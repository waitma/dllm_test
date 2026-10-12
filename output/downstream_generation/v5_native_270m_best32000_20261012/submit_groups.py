"""Submit the prepared four 270M evaluation groups after CPU gates: activate pllm; python -B FILE."""
import json,re,subprocess
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path('/vepfs-mlp2/c20250601/251105016/project/dllm_test')
RUN=ROOT/'output/downstream_generation/v5_native_270m_best32000_20261012'
assert (RUN/'cpu_gate/cpu_checks.json').exists(), 'CPU preflight required'
records=[]
for group in range(4):
    stamp=datetime.now(timezone.utc).isoformat()
    conf=ROOT/f'eval_jobs/v5_native_270m_best32000_g{group}_20261012.yml'
    result=subprocess.run(['/root/.volc/bin/volc','ml_task','submit','--conf',str(conf)],capture_output=True,text=True)
    output=result.stdout+result.stderr
    (RUN/f'submit_g{group}.log').write_text(output)
    ids=sorted(set(re.findall(r't-\d{14}-[a-z0-9]+',output)))
    entry=dict(group=group,at_utc=stamp,task_ids=ids,returncode=result.returncode,config=str(conf),status='submitted_pending_platform_query',preemptible=False)
    records.append(entry);(RUN/'submissions.json').write_text(json.dumps(records,indent=2)+'\n')
    p=ROOT/'PROJECT_PROCESS.md';s=p.read_text()
    if ids:
        marker='| 任务 | ID | 最后记录 | 证据 |\n|---|---|---|---|'
        assert marker in s
        row=f"| 270M best32000 下游 g{group} | {', '.join(ids)} | {stamp} submitted，待平台查询；非闲时 | [提交记录]({RUN/'submissions.json'}) |"
        s=s.replace(marker,marker+'\n'+row,1)
    s=re.sub(r'Last updated: [^\n]+','Last updated: '+stamp,s,count=1)
    s+=f"\n\n## {stamp} 270M 下游提交 g{group}\n\n操作 submit；task_id={ids}；YAML={conf}；returncode={result.returncode}；初始状态 submitted_pending_platform_query；Preemptible=false。固定验证最佳32000步，37项分四组，独立评测快照为硬链接；训练不受影响。\n"
    p.write_text(s)
    print(json.dumps(entry),flush=True)
    if result.returncode or len(ids)!=1: raise SystemExit('Submission requires inspection; do not blindly repeat')
plan=json.loads((RUN/'plan.json').read_text());plan['status']='submitted';(RUN/'plan.json').write_text(json.dumps(plan,indent=2,ensure_ascii=False)+'\n')
