import json, re, numpy as np, pathlib
B=pathlib.Path('downstream/benchmark/outputs/tcr_generation_bench/setting_B')
ev=json.loads(open('downstream/benchmark/data/tcr_generation_bench/eval_conditional.json').read())
common=set(json.loads(open('downstream/benchmark/data/tcr_generation_bench/bioseq_unseen_pmhc.json').read()))
RES="RVRAYTYSK_HLA-A*03:01"
cs=sorted(p for p,e in ev.items() if e['category']=='benchmark14' and p in common and p!=RES)
AA="ACDEFGHIKLMNPQRSTVWY"
def core(s):
    s=s.upper()
    if s.startswith('C'): s=s[1:]
    if s and s[-1] in 'FW': s=s[:-1]
    return s
def comp(seqs):
    j="".join(core(s) for s in seqs if core(s)); n=len(j) or 1
    return np.array([j.count(a)/n for a in AA])
def jsd(p,q):
    p,q=p+1e-12,q+1e-12; m=(p+q)/2
    kl=lambda a,b: float(np.sum(a*np.log2(a/b)))
    return 0.5*kl(p,m)+0.5*kl(q,m)
ref=comp([r for p in cs for r in ev[p]['ref_binders']])
def prof(name,seqs):
    c=[core(s) for s in seqs if core(s)]; j="".join(c)
    print(f"  {name:38s} G%={j.count('G')/len(j)*100:5.1f} "
          f"GGGG={np.mean([bool(re.search(r'G{4,}',x)) for x in c])*100:5.1f}% "
          f"compJSD={jsd(comp(seqs),ref):7.4f} n={len(c)}")
print("Composition realism (real binders: G%=12.4  GGGG=0.0%  compJSD=0)\n")
for t in ("olga","tcrdiff","tcrdesign"):
    prof(f"{t} baseline",[x for l in open(B/t/'designs.jsonl') for x in json.loads(l)['sequences']])
print()
for k,v in json.load(open('/tmp/vs_s2_designs_esmc.json')).items(): prof(f"esmc189000 {k}", v['seqs'])
print()
v3=json.load(open('/tmp/vs_s2_designs_v3.json'))
for k in ("vanilla T=1.0","argmax (greedy)"): prof(f"v3_121000 {k}", v3[k]['seqs'])
