"""Recursively check a results JSON for NaN/inf and print a few key numbers."""
import json
import math
import sys

path = sys.argv[1]
with open(path) as f:
    data = json.load(f)

bad = []


def walk(obj, trail):
    if isinstance(obj, dict):
        for k, v in obj.items():
            walk(v, trail + [str(k)])
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            walk(v, trail + [f"[{i}]"])
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            bad.append(".".join(trail) + f" = {obj}")


walk(data, [])
print(f"FILE: {path}")
print(f"non-finite (NaN/inf) values: {len(bad)}")
for b in bad[:40]:
    print("  BAD:", b)
print("top-level keys:", list(data.keys()) if isinstance(data, dict) else type(data))
