import sys, json; sys.path.insert(0, sys.argv[1]); from load import c7
out = {}
for nid, label, f in c7.CHECKS:
    if nid in ('N7-01','N7-02','N7-33','N7-42','N7-43','N7-44','N7-45','N7-46','N7-47'): continue
    r = c7.Report(nid, label)
    try: f(r)
    except Exception as e: print(nid, 'ERR', e)
    out[nid] = {k: v for k, v in r.values.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
json.dump(out, open(sys.argv[1] + '/raw.json', 'w'), indent=1)
print(sum(len(v) for v in out.values()))
