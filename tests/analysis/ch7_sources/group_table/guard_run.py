"""Run a copied answer script with every path naming patient data hidden from listings and refused on open."""
import builtins, os, glob, re, sys, runpy, io
BAD = re.compile(r'tcga|annitia|nhanes|pam50', re.I)
_open, _listdir, _glob, _scandir, _iglob = builtins.open, os.listdir, glob.glob, os.scandir, glob.iglob
REFUSED = []
def g_open(f, *a, **k):
    if isinstance(f, (str, bytes, os.PathLike)) and BAD.search(os.fsdecode(f)):
        REFUSED.append(os.fsdecode(f)); raise PermissionError('patient data: ' + os.fsdecode(f))
    return _open(f, *a, **k)
builtins.open = g_open; io.open = g_open
os.listdir = lambda p='.': [x for x in _listdir(p) if not BAD.search(x)]
glob.glob = lambda *a, **k: [x for x in _glob(*a, **k) if not BAD.search(x)]
glob.iglob = lambda *a, **k: (x for x in _iglob(*a, **k) if not BAD.search(x))
script = sys.argv[1]; sys.argv = sys.argv[1:]
try:
    runpy.run_path(script, run_name='__main__')
finally:
    print('REFUSED opens:', len(REFUSED))
