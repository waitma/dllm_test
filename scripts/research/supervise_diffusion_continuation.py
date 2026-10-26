"""Advance the authorized EOS pilots to formal training only after passed gates.

Activate pllm; USE_TF=0 python -B -m scripts.research.supervise_diffusion_continuation.
Requires pilot_ce already enqueued. Never retries failed stages, changes budgets,
or cancels platform jobs. RUN/STOP prevents subsequent stages.
"""
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

from scripts.research.prepare_diffusion_continuation import ROOT, RUN, POOL
from scripts.research.bert_mlm_data import write_json, sha


def now():
    return datetime.now(timezone.utc).isoformat()


def wait_stage(stage):
    packages=json.loads((RUN/"packages.json").read_text())
    package=packages[stage]
    bundle=Path(package["bundle"])
    output=Path(package["output"])
    ticket=json.loads((bundle/"ticket.json").read_text())
    deadline=time.monotonic()+3600
    while time.monotonic()<deadline:
        if (RUN/"STOP").exists():
            output.mkdir(exist_ok=True)
            (output/"STOP").touch()
            raise RuntimeError("User stop marker: no further stage will start")
        for path in (POOL/"results").glob("diffusion_edit287000_*.json"):
            result=json.loads(path.read_text())
            command=result.get("argv",[])
            if str(bundle) in command and result.get("state")!="succeeded":
                raise RuntimeError("Stage failed; inspect "+str(path))
        complete=output/"COMPLETE.json"
        state=json.loads((POOL/"state.json").read_text())
        if complete.exists() and state["state"]=="standby":
            done=json.loads(complete.read_text())
            if done["status"]!="success" or done["step"]!=200:
                raise ValueError("Pilot is not fully complete")
            if list((POOL/"running").glob("*.json")) or list((POOL/"pending").glob("*.json")):
                raise ValueError("Another pool ticket is pending/running")
            return done
        write_json(RUN/"continuation_driver.json",dict(status="waiting",stage=stage,
                   updated_utc=now(),output=str(output),supervisor_sha256=sha(__file__)))
        time.sleep(5)
    raise TimeoutError("Stage exceeded its one-hour startup/pilot window")


def main():
    with (RUN/"continuation_driver.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (RUN/"formal_launch.json").exists():
            raise RuntimeError("Formal launch already exists; do not duplicate")
        try:
            wait_stage("pilot_ce")
            print(now(),"CE pilot complete; starting focal pilot",flush=True)
            subprocess.run([sys.executable,"-B","-m","scripts.research.prepare_diffusion_continuation",
                            "--phase","pilot_focal","--enqueue"],cwd=ROOT,check=True)
            wait_stage("pilot_focal")
            subprocess.run([sys.executable,"-B","-m","scripts.research.select_diffusion_focal",
                            "--run",str(RUN)],cwd=ROOT,check=True)
            if (RUN/"STOP").exists():
                raise RuntimeError("Stop requested before formal launch")
            subprocess.run([sys.executable,"-B","-m","scripts.research.prepare_diffusion_continuation",
                            "--phase","train","--enqueue"],cwd=ROOT,check=True)
            write_json(RUN/"formal_launch.json",dict(status="enqueued",created_utc=now(),
                package=json.loads((RUN/"packages.json").read_text())["train"],
                selection_sha256=sha(RUN/"focal_selection.json"),supervisor_sha256=sha(__file__)))
            write_json(RUN/"continuation_driver.json",dict(status="formal_enqueued",updated_utc=now()))
        except BaseException as exc:
            write_json(RUN/"continuation_driver.json",dict(status="failed",error=repr(exc),updated_utc=now()))
            raise


if __name__=="__main__":
    main()
