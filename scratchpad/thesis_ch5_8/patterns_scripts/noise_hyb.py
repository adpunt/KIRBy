import json,collections,statistics as st
R=json.load(open('/private/tmp/claude-501/noise_rows.json'))
def gain(a,b,low): return (a-b) if low else (b-a)
for dom in ['QM9 CONTROL','small molecules','antibodies','reactions']:
    rs=[r for r in R if r['dom']==dom and r['s0'] is not None and r['h0'] is not None]
    ho=sum(1 for r in rs if r['h_ok'] and not r['s_ok']); so=sum(1 for r in rs if r['s_ok'] and not r['h_ok'])
    gs=[gain(r['s0'],r['s1'],r['low']) for r in rs]; gh=[gain(r['h0'],r['h1'],r['low']) for r in rs]
    hb=sum(1 for r in rs if gain(r['s0'],r['h0'],r['low'])>0); hbafter=sum(1 for r in rs if gain(r['s1'],r['h1'],r['low'])>0)
    keep=sum(1 for r in rs if gain(r['s0'],r['h0'],r['low'])>0 and gain(r['s1'],r['h1'],r['low'])>0)
    larger=sum(1 for a,b in zip(gs,gh) if b>a)
    print(f"{dom}: {len(rs)} rows with both cells; helped hybrid only {ho}, helped single only {so}; hybrid gain larger than single gain in {larger} rows; median gain single {st.median(gs):+.4f}, hybrid {st.median(gh):+.4f} (metric units, mixed metrics); hybrid ahead before {hb}, after {hbafter}; ahead before and still ahead after {keep}")
    for ds in sorted(set(r['ds'] for r in rs)):
        q=[r for r in rs if r['ds']==ds]
        gs=[gain(r['s0'],r['s1'],r['low']) for r in q]; gh=[gain(r['h0'],r['h1'],r['low']) for r in q]
        print(f"   {ds} ({q[0]['metric'].split()[0]}): rows {len(q)}; hybrid ahead before {sum(1 for r in q if gain(r['s0'],r['h0'],r['low'])>0)}, after {sum(1 for r in q if gain(r['s1'],r['h1'],r['low'])>0)}; median gain single {st.median(gs):+.4f} hybrid {st.median(gh):+.4f}; hybrid gain larger in {sum(1 for a,b in zip(gs,gh) if b>a)}")
