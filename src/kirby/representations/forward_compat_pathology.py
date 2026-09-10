"""
Forward-compatibility pathology / whole-slide representations (c13 candidates).

One function per newly published pathology encoder. Each turns a directory of
whole-slide tile features into ONE ROW PER TCGA SAMPLE of float32 features,
frozen — no fine-tuning, no label use.

Contract every function here keeps:
  f(slide_dir, sample_ids) -> np.ndarray of shape (len(sample_ids), D), float32,
  rows in the SAME ORDER as sample_ids. A sample with no slide features gets a
  zero row and a printed count of how many rows were zero-filled; nothing is
  dropped and nothing is reordered. If EVERY row would be zero the function
  raises RuntimeError instead of returning, so an all-zero block is never handed
  to c13 to score against the labels. A stored block whose width is not exactly
  D is a zero row, never a truncation to the first D features. Barcode matching
  is the same flexible 15-char / 12-char / full-string match that
  load_wsi_features in kirby/representations/genomic.py already does.

TILES MUST BE EXTRACTED FIRST. Every model here is a TILE encoder (MOOZY is a
case encoder over tile features). None of them reads an .svs. Nothing in this
file segments tissue, tiles a slide or reads a WSI pyramid. Each function looks
for features that already exist under slide_dir; if none are there it raises
RuntimeError naming the extraction call read off that model's own card. The
TCGA diagnostic slides themselves come from the GDC, and tiling them (CLAM or
TRIDENT, 512x512 at 40x for GPFM) is a separate job that has not been run in
this repo.

Missing packages and missing weights raise ImportError or RuntimeError naming
what to install and from where, so the TCGA loader can print a warning and move
on. Nothing here imports torch, timm or transformers at module import time.

Models (4), width and pooling:
  wsi_gpfm         1024D  mean over tiles; tile row = model(x) output
                          majiabo/GPFM, MIT, ungated
  wsi_midnight     3072D  mean over tiles; tile row = class token CONCATENATED
                          with the mean of the patch tokens (1536 + 1536)
                          kaiko-ai/midnight, MIT, ungated (the 92k variants are
                          restricted; this is Midnight-12k)
  wsi_distillpath   384D  mean over tiles; tile row = class token
                          RamonK/DistillPath-KS16-Virchow2, ungated,
                          Kaiko Non-Commercial Public License
  wsi_moozy         768D  ONE ROW PER PATIENT from MOOZY's own case transformer;
                          no pooling is done here, the package emits one vector
                          AtlasAnalyticsLab/MOOZY, CC BY-NC-SA 4.0, ungated

GATING: none of these four needs an approved HuggingFace access request, which
is why they are the four that got written. Two carry non-commercial licenses
(DistillPath, MOOZY). The three gated pathology models the candidate list cut
(COBRA, FEATHER, Hibou-L, MUSK) are not here.

Statuses: all four were written against a model card, collection page, config
file or repository README opened on 2026-09-10; each docstring names the page it
was read from. None is a stub.

Where the features are read from, in order, for a model with key K:
  1. slide_dir/K.npy      — a dict {slide_or_sample_id: vector or (n_tiles, D)},
                            the format load_wsi_features already reads.
  2. slide_dir/K/*.npy    — one file per slide, named by TCGA barcode,
     slide_dir/K/*.h5       holding (n_tiles, D) or a single vector. For .h5 the
                            dataset tried is 'features', then 'feats', then the
                            first dataset in the file.
  3. slide_dir/tiles/<barcode>*/  — a directory of tile IMAGES per slide, which
                            is encoded here on the spot (needs torch + the model
                            weights). MOOZY has no image path: its input is
                            already-extracted lunit_vit_small_patch8_dino
                            features.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np


# Loaded encoders live here between calls, keyed by model name — same pattern as
# create_molformer's module-level model cache in molecular.py.
_FC_WSI_MODEL_CACHE: dict = {}

_TILE_IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')


class _NoTileOpened(Exception):
    """One slide's tile images all failed to open.

    Deliberately NOT an ImportError or a RuntimeError. Those two propagate out
    of _rows_from_tile_images and stop the whole representation, which is right
    for a missing package or missing weights and wrong for one bad slide: that
    sample gets a zero row and every other sample still gets its features.
    """


# =============================================================================
# Shared plumbing: barcode matching, pooling, zero-fill accounting
# =============================================================================

def _report_zero_rows(name: str, n_zero: int, n_total: int) -> None:
    if n_zero:
        print(f"  WARNING {name}: {n_zero}/{n_total} rows zero-filled "
              f"(no slide features for that sample); rows kept in input order.")


def _add_to_lookup(lookup: Dict[str, list], slide_id: str, value) -> None:
    """Index one slide's features under every TCGA barcode truncation.

    A patient contributes more than one slide often enough that the value is a
    LIST — every slide found under that barcode is kept and they are pooled
    together later, rather than the last one overwriting the first.
    """
    sid = str(slide_id)
    keys = [sid]
    if sid.startswith('TCGA'):
        keys.append(sid[:12])
        if len(sid) >= 15:
            keys.append(sid[:15])
    for k in keys:
        lookup.setdefault(k, []).append(value)


def _read_h5_features(path: Path) -> Optional[np.ndarray]:
    """Read a tile-feature array out of an .h5 file, or None."""
    try:
        import h5py
    except ImportError as e:
        raise ImportError(
            "Reading .h5 slide features needs h5py. Install it with: "
            "pip install h5py. ({})".format(e)
        )
    try:
        with h5py.File(path, 'r') as f:
            for key in ('features', 'feats', 'feat'):
                if key in f:
                    return np.asarray(f[key][:], dtype=np.float32)
            for key in f.keys():
                item = f[key]
                if hasattr(item, 'shape'):
                    print(f"    {path.name}: no 'features' dataset, using '{key}'")
                    return np.asarray(item[:], dtype=np.float32)
    except Exception as e:
        print(f"    WARNING: could not read {path}: {e}")
    return None


def _collect_precomputed(slide_dir: Path, key: str) -> Dict[str, list]:
    """Build barcode -> [feature arrays] from the two precomputed layouts."""
    lookup: Dict[str, list] = {}

    dict_file = slide_dir / f'{key}.npy'
    if dict_file.exists():
        try:
            data = np.load(dict_file, allow_pickle=True).item()
        except Exception as e:
            print(f"    WARNING: could not load {dict_file}: {e}")
            data = None
        if isinstance(data, dict):
            for slide_id, emb in data.items():
                arr = np.asarray(emb, dtype=np.float32)
                _add_to_lookup(lookup, slide_id, arr)
            print(f"    {key}: {len(data)} slides read from {dict_file.name}")
        elif data is not None:
            print(f"    WARNING: {dict_file} is not a "
                  f"{{slide_id: features}} dict; ignored")

    per_slide = slide_dir / key
    if per_slide.is_dir():
        n_files = 0
        for path in sorted(per_slide.iterdir()):
            if path.suffix == '.npy':
                try:
                    arr = np.asarray(np.load(path), dtype=np.float32)
                except Exception as e:
                    print(f"    WARNING: could not load {path}: {e}")
                    continue
            elif path.suffix in ('.h5', '.hdf5'):
                arr = _read_h5_features(path)
                if arr is None:
                    continue
            else:
                continue
            _add_to_lookup(lookup, path.stem, arr)
            n_files += 1
        if n_files:
            print(f"    {key}: {n_files} per-slide files read from {per_slide}")

    return lookup


def _all_rows_zero(name: str, n_zero: int, n_total: int, what: str) -> None:
    """Every row zero-filled means the matrix carries no signal at all.

    Returning it would hand c13 a constant block to score against the labels, so
    raise instead and let the loader list this representation as one that did
    not build.
    """
    if n_total and n_zero == n_total:
        raise RuntimeError(
            f"{name}: all {n_total} rows zero-filled — {what}. No row carries "
            f"any feature, so the matrix is not returned."
        )


def _rows_from_lookup(lookup: Dict[str, list], sample_ids: Sequence[str],
                      dim: int, name: str) -> np.ndarray:
    """Mean-pool each sample's tiles into one row, in sample_ids order.

    Every slide the sample has contributes its tile vectors and they are
    averaged together. A sample whose features are absent, malformed or the
    wrong width gets a zero row; nothing is dropped and nothing is reordered.
    """
    X = np.zeros((len(sample_ids), dim), dtype=np.float32)
    n_zero = 0

    for i, sid in enumerate(sample_ids):
        found = None
        for trunc in (str(sid)[:15], str(sid)[:12], str(sid)):
            if trunc in lookup:
                found = lookup[trunc]
                break
        if not found:
            n_zero += 1
            continue

        tiles = []
        for arr in found:
            a = np.asarray(arr, dtype=np.float32)
            if a.ndim == 1:
                tiles.append(a.reshape(1, -1))
            elif a.ndim == 2:
                tiles.append(a)
            else:
                print(f"    WARNING {name}: {sid} has a {a.ndim}-D feature "
                      f"array, skipped")
        if not tiles:
            n_zero += 1
            continue

        widths = {t.shape[1] for t in tiles}
        if len(widths) > 1:
            print(f"    WARNING {name}: {sid} mixes feature widths {widths}, "
                  f"zero row")
            n_zero += 1
            continue
        width = tiles[0].shape[1]
        if width != dim:
            print(f"    WARNING {name}: {sid} has {width} features, expected "
                  f"{dim}, zero row")
            n_zero += 1
            continue

        stacked = np.vstack(tiles)
        X[i] = stacked.mean(axis=0)

    n_found = len(sample_ids) - n_zero
    print(f"    {name}: {n_found}/{len(sample_ids)} samples matched, dim={dim}")
    _report_zero_rows(name, n_zero, len(sample_ids))
    _all_rows_zero(name, n_zero, len(sample_ids),
                   f"no sample barcode matched a stored slide, or every stored "
                   f"block was the wrong width for {dim} features")
    return X


def _tile_image_dirs(slide_dir: Path) -> Dict[str, list]:
    """Barcode -> [directories of tile images], from slide_dir/tiles/."""
    root = slide_dir / 'tiles'
    lookup: Dict[str, list] = {}
    if not root.is_dir():
        return lookup
    for path in sorted(root.iterdir()):
        if path.is_dir():
            _add_to_lookup(lookup, path.name, path)
    return lookup


def _tile_paths(directory: Path) -> List[Path]:
    return sorted(p for p in directory.iterdir()
                  if p.suffix.lower() in _TILE_IMAGE_EXTS)


def _rows_from_tile_images(dir_lookup: Dict[str, list],
                           sample_ids: Sequence[str], dim: int, name: str,
                           encoder: Callable[[List[Path]], np.ndarray]
                           ) -> np.ndarray:
    """Encode tile images per sample and mean-pool them into one row each."""
    X = np.zeros((len(sample_ids), dim), dtype=np.float32)
    n_zero = 0

    for i, sid in enumerate(sample_ids):
        dirs = None
        for trunc in (str(sid)[:15], str(sid)[:12], str(sid)):
            if trunc in dir_lookup:
                dirs = dir_lookup[trunc]
                break
        if not dirs:
            n_zero += 1
            continue

        paths: List[Path] = []
        for d in dirs:
            paths.extend(_tile_paths(d))
        if not paths:
            n_zero += 1
            continue

        try:
            feats = encoder(paths)
        except (ImportError, RuntimeError):
            raise
        except Exception as e:
            print(f"    WARNING {name}: encoding {sid} failed ({e}); zero row.")
            n_zero += 1
            continue

        feats = np.asarray(feats, dtype=np.float32)
        if feats.ndim != 2 or feats.shape[1] != dim:
            print(f"    WARNING {name}: {sid} encoded to shape {feats.shape}, "
                  f"expected (n_tiles, {dim}); zero row.")
            n_zero += 1
            continue
        X[i] = feats.mean(axis=0)

    n_found = len(sample_ids) - n_zero
    print(f"    {name}: {n_found}/{len(sample_ids)} samples encoded from tile "
          f"images, dim={dim}")
    _report_zero_rows(name, n_zero, len(sample_ids))
    _all_rows_zero(name, n_zero, len(sample_ids),
                   f"no sample barcode matched a tiles/<barcode>/ directory, or "
                   f"every encode returned something other than {dim} features")
    return X


def _wsi_rows(slide_dir, sample_ids, key: str, dim: int, name: str,
              encoder: Optional[Callable[[List[Path]], np.ndarray]],
              extract_hint: str) -> np.ndarray:
    """Precomputed features first, tile images second, RuntimeError third."""
    if not len(sample_ids):
        return np.zeros((0, dim), dtype=np.float32)

    if slide_dir is None:
        raise RuntimeError(
            f"{name}: no whole-slide feature directory was passed "
            f"(slide_dir is None). {extract_hint}"
        )

    slide_dir = Path(slide_dir)
    if not slide_dir.is_dir():
        raise RuntimeError(
            f"{name}: whole-slide feature directory {slide_dir} does not "
            f"exist. {extract_hint}"
        )

    lookup = _collect_precomputed(slide_dir, key)
    if lookup:
        return _rows_from_lookup(lookup, sample_ids, dim, name)

    if encoder is not None:
        dir_lookup = _tile_image_dirs(slide_dir)
        if dir_lookup:
            return _rows_from_tile_images(dir_lookup, sample_ids, dim, name,
                                          encoder)

    raise RuntimeError(
        f"{name}: no features found under {slide_dir}. Looked for "
        f"{key}.npy (a dict of slide_id -> features), {key}/ (one .npy or .h5 "
        f"per slide) and tiles/<barcode>/ (tile images). {extract_hint}"
    )


def _torch_and_pil():
    """Import torch and PIL inside the call, never at module import."""
    try:
        import torch
        from PIL import Image
    except ImportError as e:
        raise ImportError(
            "Encoding tile images needs torch and pillow. Install them with: "
            "pip install torch torchvision pillow. ({})".format(e)
        )
    return torch, Image


def _batched_encode(tile_paths, transform, forward, batch_size, name):
    """Shared image loop: read tiles, transform, forward, stack rows.

    A tile that will not open is skipped with a printed warning; the SLIDE still
    gets a row from whatever tiles did open. A slide where nothing opens raises,
    and the caller turns that into a zero row for the sample.
    """
    torch, Image = _torch_and_pil()

    rows, batch, n_bad = [], [], 0
    with torch.no_grad():
        for path in list(tile_paths) + [None]:
            if path is not None:
                try:
                    img = Image.open(path).convert('RGB')
                    batch.append(transform(img))
                except Exception as e:
                    print(f"    WARNING {name}: tile {path} unreadable ({e})")
                    n_bad += 1
            if batch and (len(batch) >= batch_size or path is None):
                x = torch.stack(batch)
                rows.append(forward(x).float().cpu().numpy())
                batch = []

    if not rows:
        raise _NoTileOpened(
            f"{name}: no tile opened out of {len(tile_paths)}")
    if n_bad:
        print(f"    {name}: {n_bad} unreadable tiles skipped")
    return np.vstack(rows).astype(np.float32)


# =============================================================================
# GPFM — Generalizable Pathology Foundation Model, 1024 features per tile
# =============================================================================

def _encode_gpfm_tiles(tile_paths: List[Path], batch_size: int = 32
                       ) -> np.ndarray:
    """Encode tile images with GPFM. Returns (n_tiles, 1024).

    API read off https://huggingface.co/majiabo/GPFM on 2026-09-10:
        model = timm.create_model('vit_large_patch14_dinov2.lvd142m',
                                  pretrained=False, img_size=224,
                                  init_values=1.0e-05)
        ckpt_path = hf_hub_download(repo_id="majiabo/GPFM", filename="GPFM.pth")
        state_dict = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=True)
        feat = model(x)          # [1, 1024]
    with ImageNet normalisation, mean (0.485, 0.456, 0.406) and std
    (0.229, 0.224, 0.225), bicubic resize to 224x224. The repository
    https://github.com/birkhoffkiki/GPFM says tiles should be 512x512 cut at 40x.
    """
    torch, _ = _torch_and_pil()
    try:
        import timm
        from torchvision import transforms
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "wsi_gpfm needs timm, torchvision and huggingface_hub. Install "
            "them with: pip install timm torchvision huggingface_hub. "
            "({})".format(e)
        )

    if 'gpfm' not in _FC_WSI_MODEL_CACHE:
        print("  Loading GPFM (majiabo/GPFM, GPFM.pth, ViT-L/14 DINOv2)...")
        model = timm.create_model(
            'vit_large_patch14_dinov2.lvd142m',
            pretrained=False,
            img_size=224,
            init_values=1.0e-05,
        )
        try:
            ckpt_path = hf_hub_download(repo_id='majiabo/GPFM',
                                        filename='GPFM.pth')
        except Exception as e:
            raise RuntimeError(
                "wsi_gpfm: could not fetch GPFM.pth from the HuggingFace repo "
                "majiabo/GPFM ({}). The weights are MIT and ungated; download "
                "GPFM.pth by hand from https://huggingface.co/majiabo/GPFM and "
                "set HF_HOME to a directory holding it.".format(e)
            )
        state_dict = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=True)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device).eval()
        transform = transforms.Compose([
            transforms.Resize(
                (224, 224),
                interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])
        _FC_WSI_MODEL_CACHE['gpfm'] = (model, transform, device)
        print(f"    GPFM loaded on {device}")

    model, transform, device = _FC_WSI_MODEL_CACHE['gpfm']
    return _batched_encode(
        tile_paths, transform,
        lambda x: model(x.to(device)),
        batch_size, 'wsi_gpfm')


def load_gpfm_features(slide_dir, sample_ids) -> np.ndarray:
    """GPFM tile features, mean-pooled per sample. (len(sample_ids), 1024).

    Model: GPFM, Generalizable Pathology Foundation Model, unified knowledge
    distillation, Nature Biomedical Engineering s41551-025-01488-4 (online
    2 Sept 2025); preprint arXiv 2407.18449.
    Weights: HuggingFace majiabo/GPFM, file GPFM.pth, MIT, NOT gated — no access
    request is needed.
    Code: https://github.com/birkhoffkiki/GPFM

    Dimensionality: 1024 features per tile, stated on the model card and in the
    repository's own extraction example.

    POOLING: one tile row is the model's output vector, 1024 wide. One sample
    row is the arithmetic mean of the tile rows over every slide that sample
    has. No attention, no weighting.

    SETUP REQUIRED — tiles must exist before this runs. Provide one of, under
    slide_dir:
      gpfm.npy            a dict {slide_id: (n_tiles, 1024) or (1024,)}
      gpfm/<barcode>.npy  or .h5, one file per slide
      tiles/<barcode>/    tile images (.png/.jpg/.tif), encoded here; the
                          repository says 512x512 tiles cut at 40x
    Cutting TCGA diagnostic slides into tiles is not done by this file. Raises
    RuntimeError naming what was missing if none of the three is there.

    Requires for the tile-image path: pip install torch torchvision timm
    huggingface_hub pillow (about 1.1 GB of weights on first call).
    """
    return _wsi_rows(
        slide_dir, sample_ids, key='gpfm', dim=1024, name='wsi_gpfm',
        encoder=_encode_gpfm_tiles,
        extract_hint=(
            "Extract GPFM tile features first: tile the slides (512x512 at "
            "40x), then either run the repository's extract_scripts from "
            "https://github.com/birkhoffkiki/GPFM or drop the tile images at "
            "slide_dir/tiles/<TCGA barcode>/ and this function encodes them "
            "with majiabo/GPFM (MIT, ungated)."
        ),
    )


# =============================================================================
# Midnight-12k — kaiko.ai ViT-g/14, 1536 cls + 1536 mean patch = 3072 features
# =============================================================================

def _encode_midnight_tiles(tile_paths: List[Path], batch_size: int = 32
                           ) -> np.ndarray:
    """Encode tile images with Midnight-12k. Returns (n_tiles, 3072).

    API read off https://huggingface.co/kaiko-ai/midnight on 2026-09-10,
    verbatim from the card's example:
        model = AutoModel.from_pretrained('kaiko-ai/midnight')
        transform = v2.Compose([v2.Resize(224), v2.CenterCrop(224),
                                v2.ToTensor(),
                                v2.Normalize(mean=(0.5, 0.5, 0.5),
                                             std=(0.5, 0.5, 0.5))])
        def extract_classification_embedding(tensor):
            cls_embedding, patch_embeddings = tensor[:, 0, :], tensor[:, 1:, :]
            return torch.cat([cls_embedding, patch_embeddings.mean(1)], dim=-1)
        embedding = extract_classification_embedding(model(batch).last_hidden_state)

    The card does NOT print the output width. 3072 is derived: config.json at
    https://huggingface.co/kaiko-ai/midnight/raw/main/config.json gives
    hidden_size 1536 (Dinov2Model), and the card's embedding is the class token
    concatenated with the mean of the patch tokens, so 1536 + 1536.
    """
    torch, _ = _torch_and_pil()
    try:
        from transformers import AutoModel
        from torchvision.transforms import v2
    except ImportError as e:
        raise ImportError(
            "wsi_midnight needs transformers and torchvision. Install them "
            "with: pip install transformers torchvision. ({})".format(e)
        )

    if 'midnight' not in _FC_WSI_MODEL_CACHE:
        print("  Loading Midnight-12k (kaiko-ai/midnight)...")
        try:
            model = AutoModel.from_pretrained('kaiko-ai/midnight')
        except Exception as e:
            raise RuntimeError(
                "wsi_midnight: could not load kaiko-ai/midnight ({}). "
                "Midnight-12k is MIT and ungated; pre-download it with "
                "huggingface-cli download kaiko-ai/midnight. The Midnight-92k "
                "variants are restricted and are NOT what this uses.".format(e)
            )
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device).eval()
        transform = v2.Compose([
            v2.Resize(224),
            v2.CenterCrop(224),
            v2.ToTensor(),
            v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
        _FC_WSI_MODEL_CACHE['midnight'] = (model, transform, device)
        print(f"    Midnight-12k loaded on {device}")

    model, transform, device = _FC_WSI_MODEL_CACHE['midnight']

    def forward(x):
        hidden = model(x.to(device)).last_hidden_state
        cls_embedding, patch_embeddings = hidden[:, 0, :], hidden[:, 1:, :]
        return torch.cat([cls_embedding, patch_embeddings.mean(1)], dim=-1)

    return _batched_encode(tile_paths, transform, forward, batch_size,
                           'wsi_midnight')


def load_midnight_features(slide_dir, sample_ids) -> np.ndarray:
    """Midnight-12k tile features, mean-pooled per sample. (n, 3072).

    Model: Midnight-12k, kaiko.ai, ViT-g/14, MICCAI 2025 (arXiv 2504.05186).
    Weights: HuggingFace kaiko-ai/midnight, MIT, NOT gated. The Midnight-92k
    and Midnight-92k/392 variants are restricted access and are not used here.

    Dimensionality: 3072 per tile. The model card does not print a width; this
    is the card's own pooling applied to the config's hidden_size — 1536 class
    token concatenated with 1536 mean of patch tokens.

    POOLING: one tile row is the class token CONCATENATED with the mean of the
    patch tokens, which is what the model card's extract_classification_embedding
    does. One sample row is the arithmetic mean of the tile rows over every
    slide that sample has.

    SETUP REQUIRED — tiles must exist before this runs. Provide one of, under
    slide_dir:
      midnight.npy            a dict {slide_id: (n_tiles, 3072) or (3072,)}
      midnight/<barcode>.npy  or .h5, one file per slide
      tiles/<barcode>/        tile images, encoded here at 224x224
    Raises RuntimeError naming what was missing if none of the three is there.

    Requires for the tile-image path: pip install torch torchvision
    transformers pillow.
    """
    return _wsi_rows(
        slide_dir, sample_ids, key='midnight', dim=3072, name='wsi_midnight',
        encoder=_encode_midnight_tiles,
        extract_hint=(
            "Extract Midnight-12k tile features first: put tile images at "
            "slide_dir/tiles/<TCGA barcode>/ and this function encodes them "
            "with kaiko-ai/midnight (MIT, ungated), or write precomputed "
            "features to slide_dir/midnight.npy."
        ),
    )


# =============================================================================
# DistillPath — 22M distilled ViT-S/16, 384 features per tile
# =============================================================================

_DISTILLPATH_REPO = 'RamonK/DistillPath-KS16-Virchow2'


def _encode_distillpath_tiles(tile_paths: List[Path], batch_size: int = 64
                              ) -> np.ndarray:
    """Encode tile images with DistillPath-KS16-Virchow2. (n_tiles, 384).

    API read off https://huggingface.co/RamonK/DistillPath-KS16-Virchow2 on
    2026-09-10:
        model = timm.create_model("hf_hub:RamonK/DistillPath-KS16-Virchow2",
                                  pretrained=True, num_classes=0)
        model.eval()
    with the kaiko normalisation, mean (0.5, 0.5, 0.5) std (0.5, 0.5, 0.5),
    resized to 224x224. With num_classes=0 the forward pass returns the class
    token, 384 wide, which is the tile embedding.

    Which checkpoint: the collection https://huggingface.co/collections/RamonK/
    distillpath holds six — KS16 and IS16 crossed with HOpt0, UNI2h and
    Virchow2. KS16-Virchow2 is the one the candidate list quotes (EVA seven-task
    mean 0.795). The KS16-H0mini variant the list also names for the HEST number
    was NOT among the six repo ids on the collection page, so it is not wired.
    """
    torch, _ = _torch_and_pil()
    try:
        import timm
        from torchvision import transforms
    except ImportError as e:
        raise ImportError(
            "wsi_distillpath needs timm and torchvision. Install them with: "
            "pip install timm torchvision. ({})".format(e)
        )

    if 'distillpath' not in _FC_WSI_MODEL_CACHE:
        print(f"  Loading DistillPath ({_DISTILLPATH_REPO}, ViT-S/16, 22M)...")
        try:
            model = timm.create_model(
                f'hf_hub:{_DISTILLPATH_REPO}',
                pretrained=True,
                num_classes=0,
            )
        except Exception as e:
            raise RuntimeError(
                "wsi_distillpath: could not load {} through timm ({}). The "
                "weights are ungated but carry the Kaiko Non-Commercial Public "
                "License; pre-download with huggingface-cli download {}. timm "
                "must be new enough to read an hf_hub: model id.".format(
                    _DISTILLPATH_REPO, e, _DISTILLPATH_REPO)
            )
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device).eval()
        transform = transforms.Compose([
            transforms.Resize((224, 224),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
        _FC_WSI_MODEL_CACHE['distillpath'] = (model, transform, device)
        print(f"    DistillPath loaded on {device}")

    model, transform, device = _FC_WSI_MODEL_CACHE['distillpath']
    return _batched_encode(
        tile_paths, transform,
        lambda x: model(x.to(device)),
        batch_size, 'wsi_distillpath')


def load_distillpath_features(slide_dir, sample_ids) -> np.ndarray:
    """DistillPath tile features, mean-pooled per sample. (n, 384).

    Model: DistillPath KS16-Virchow2, a 22M-parameter ViT-S/16 distilled from
    larger pathology encoders. arXiv 2608.17872, ECCV 2026 MedFM-Bench workshop.
    Weights: HuggingFace RamonK/DistillPath-KS16-Virchow2, NOT gated, Kaiko
    Non-Commercial Public License — research use only.
    Code: https://github.com/RamonKaspar/DistillPath

    Dimensionality: 384 features per tile, stated on the model card and in the
    paper's methods. The narrowest block of any pathology rep here.

    POOLING: one tile row is the class token, 384 wide (timm num_classes=0
    returns it directly). One sample row is the arithmetic mean of the tile rows
    over every slide that sample has.

    SETUP REQUIRED — tiles must exist before this runs. Provide one of, under
    slide_dir:
      distillpath.npy            a dict {slide_id: (n_tiles, 384) or (384,)}
      distillpath/<barcode>.npy  or .h5, one file per slide
      tiles/<barcode>/           tile images, encoded here at 224x224
    Raises RuntimeError naming what was missing if none of the three is there.

    Requires for the tile-image path: pip install torch torchvision timm pillow.
    """
    return _wsi_rows(
        slide_dir, sample_ids, key='distillpath', dim=384,
        name='wsi_distillpath', encoder=_encode_distillpath_tiles,
        extract_hint=(
            "Extract DistillPath tile features first: put tile images at "
            "slide_dir/tiles/<TCGA barcode>/ and this function encodes them "
            "with RamonK/DistillPath-KS16-Virchow2 (ungated, non-commercial "
            "license), or write precomputed features to "
            "slide_dir/distillpath.npy."
        ),
    )


# =============================================================================
# MOOZY — case-level encoder, 768 features per PATIENT
# =============================================================================

def _run_moozy_encoding(slide_paths: List[Path], name: str) -> np.ndarray:
    """Run MOOZY's case encoder over one patient's slide feature files.

    API read off https://huggingface.co/AtlasAnalyticsLab/MOOZY and the
    repository README https://github.com/AtlasAnalyticsLab/MOOZY on 2026-09-10:
        pip install moozy
        from moozy.encoding import run_encoding
        run_encoding(slide_paths=["slide_1.h5", "slide_2.h5"],
                     output_path="case_embedding.h5")
        with h5py.File("case_embedding.h5", "r") as f:
            embedding = f["features"][:]     # (768,) float32
    The checkpoint downloads automatically on first use; nothing is gated.

    UNCONFIRMED, stated here rather than guessed: docs/encode.md says the input
    .h5 files must already hold patch features from the
    lunit_vit_small_patch8_dino encoder and be compatible with AtlasPatch or
    TRIDENT output, but the page read does NOT name the dataset keys inside
    those files. If the inputs came from another tile encoder this call will
    either fail or return a vector the model was not trained to read.
    """
    try:
        from moozy.encoding import run_encoding
    except ImportError as e:
        raise ImportError(
            "wsi_moozy needs the moozy package. Install it with: "
            "pip install moozy (the checkpoint downloads on first use from "
            "https://huggingface.co/AtlasAnalyticsLab/MOOZY, CC BY-NC-SA 4.0, "
            "ungated). ({})".format(e)
        )
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / 'case_embedding.h5'
        run_encoding(slide_paths=[str(p) for p in slide_paths],
                     output_path=str(out))
        vec = _read_h5_features(out)
    if vec is None:
        raise RuntimeError(
            f"{name}: moozy run_encoding wrote no 'features' dataset")
    return np.asarray(vec, dtype=np.float32).reshape(-1)


def load_moozy_features(slide_dir, sample_ids) -> np.ndarray:
    """MOOZY case embeddings, one row per sample. (len(sample_ids), 768).

    Model: MOOZY, patient-first pathology foundation model, Atlas Analytics Lab,
    ECCV 2026.
    Weights: HuggingFace AtlasAnalyticsLab/MOOZY, CC BY-NC-SA 4.0, NOT gated —
    the checkpoint downloads on first use of the pip package.
    Code: https://github.com/AtlasAnalyticsLab/MOOZY

    Dimensionality: 768 features, stated on the model card as a 768-d float32
    case embedding written to the output file's 'features' dataset.

    POOLING: none is done in this file. MOOZY's own case transformer attends
    over all of a patient's slide embeddings through a learnable [CASE] token
    and emits ONE vector per patient, which becomes that sample's row directly.
    That makes it the only rep here whose unit is the patient rather than the
    slide: two samples from the same TCGA patient get the same row.

    SETUP REQUIRED — patch features must exist before this runs. Provide one of,
    under slide_dir:
      moozy.npy            a dict {patient_or_sample_id: (768,)}
      moozy/<barcode>.h5   the case_embedding.h5 MOOZY already wrote, or .npy
      moozy_input/<barcode>*.h5   per-slide patch features, which this function
                           passes to moozy.encoding.run_encoding to build the
                           case embedding on the spot
    Raises RuntimeError naming what was missing if none of the three is there.

    CAVEAT ON THE INPUT FORMAT: the MOOZY docs say the input .h5 must hold
    lunit_vit_small_patch8_dino patch features compatible with AtlasPatch or
    TRIDENT, but the page read does not name the dataset keys inside them. That
    part is unverified — the run_encoding call itself is quoted from the card.

    CAVEAT ON THE RESULT: MOOZY is pretrained on 77,134 public slides, which
    very likely includes the TCGA slides both TCGA datasets evaluate on. The
    overlap is not stated anywhere in its documentation.

    Requires: pip install moozy h5py.
    """
    if not len(sample_ids):
        return np.zeros((0, 768), dtype=np.float32)

    hint = (
        "Extract patch features first: MOOZY consumes per-slide .h5 files of "
        "lunit_vit_small_patch8_dino patch features (AtlasPatch or TRIDENT "
        "output). Put them at slide_dir/moozy_input/<TCGA barcode>*.h5 and "
        "this function runs moozy.encoding.run_encoding per patient, or write "
        "finished case embeddings to slide_dir/moozy.npy."
    )

    if slide_dir is None:
        raise RuntimeError(
            f"wsi_moozy: no whole-slide feature directory was passed "
            f"(slide_dir is None). {hint}")

    slide_dir = Path(slide_dir)
    if not slide_dir.is_dir():
        raise RuntimeError(
            f"wsi_moozy: whole-slide feature directory {slide_dir} does not "
            f"exist. {hint}")

    lookup = _collect_precomputed(slide_dir, 'moozy')
    if lookup:
        return _rows_from_lookup(lookup, sample_ids, 768, 'wsi_moozy')

    input_root = slide_dir / 'moozy_input'
    if not input_root.is_dir():
        raise RuntimeError(
            f"wsi_moozy: no features found under {slide_dir}. Looked for "
            f"moozy.npy, moozy/ and moozy_input/. {hint}")

    input_lookup: Dict[str, list] = {}
    for path in sorted(input_root.iterdir()):
        if path.suffix in ('.h5', '.hdf5'):
            _add_to_lookup(input_lookup, path.stem, path)
    if not input_lookup:
        raise RuntimeError(
            f"wsi_moozy: {input_root} holds no .h5 patch-feature files. {hint}")

    X = np.zeros((len(sample_ids), 768), dtype=np.float32)
    n_zero = 0
    for i, sid in enumerate(sample_ids):
        paths = None
        for trunc in (str(sid)[:15], str(sid)[:12], str(sid)):
            if trunc in input_lookup:
                paths = input_lookup[trunc]
                break
        if not paths:
            n_zero += 1
            continue
        try:
            vec = _run_moozy_encoding(paths, 'wsi_moozy')
        except ImportError:
            raise
        except Exception as e:
            print(f"    WARNING wsi_moozy: {sid} failed ({e}); zero row.")
            n_zero += 1
            continue
        if vec.shape[0] != 768:
            print(f"    WARNING wsi_moozy: {sid} returned {vec.shape[0]} "
                  f"features, expected 768; zero row.")
            n_zero += 1
            continue
        X[i] = vec

    n_found = len(sample_ids) - n_zero
    print(f"    wsi_moozy: {n_found}/{len(sample_ids)} cases encoded, dim=768")
    _report_zero_rows('wsi_moozy', n_zero, len(sample_ids))
    _all_rows_zero('wsi_moozy', n_zero, len(sample_ids),
                   "no sample barcode matched a moozy_input/<barcode>*.h5 file, "
                   "or every run_encoding call failed or returned a width other "
                   "than 768")
    return X
