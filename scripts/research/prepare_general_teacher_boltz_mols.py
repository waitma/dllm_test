"""Stream official Boltz CCD assets, retaining only 21 canonical protein entries.

Activate pllm, then run this absolute file. The 1.86GB upstream tar is streamed,
not stored; only the small native RDKit pickles needed by this benchmark remain.
"""
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request

ROOT=Path('/project_c20250601/251105016/project/dllm_test/output/general_teacher_validation_20261024')
DEST=Path('/project_c20250601/251105016/model_weights/general_teacher_validation_20261024/mols')
URL='https://huggingface.co/boltz-community/boltz-2/resolve/main/mols.tar'
WANTED=set('ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL UNK'.split())


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    receipt=dict(url=URL,scope='unmodified canonical protein components from official Boltz tar',files=[])
    with urllib.request.urlopen(URL,timeout=120) as response:
        with tarfile.open(fileobj=response,mode='r|') as tar:
            for member in tar:
                p=Path(member.name)
                if member.isfile() and p.suffix=='.pkl' and p.stem in WANTED:
                    data=tar.extractfile(member).read()
                    target=DEST/p.name;target.write_bytes(data)
                    row=dict(name=p.stem,path=str(target),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
                    receipt['files'].append(row)
                    print(json.dumps(row),flush=True)
                    if {x['name'] for x in receipt['files']}==WANTED:break
    receipt['status']='passed' if {x['name'] for x in receipt['files']}==WANTED else 'failed'
    (ROOT/'boltz_mols.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if receipt['status']!='passed':raise RuntimeError('Incomplete official canonical molecule set')


if __name__=='__main__':main()
