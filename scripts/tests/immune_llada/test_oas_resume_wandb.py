"""Activate pllm; python -B -m pytest THIS_FILE -q.

Run real Transformers/W&B callbacks offline, including the frozen OAS source
when available. Each source runs in a fresh isolated CPU subprocess.
"""
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
FROZEN = ROOT / 'output/v5_full_binding_20261007/launch_fixed_chain_75m_oas_queue012_20261015/source'
SMOKE = r'''
from dataclasses import asdict
import json
from pathlib import Path
import runpy
import sys

source, launcher, output = map(Path, sys.argv[1:])
sys.path.insert(0, str(source))
launch = runpy.run_path(str(launcher), run_name='_oas_resume_test')
import torch
import wandb
from transformers import TrainingArguments, TrainerState, TrainerControl
from transformers.integrations import WandbCallback
from dllm.pipelines.immune_llada.fixed_chain import FixedChainConfig

# The exact config class is sufficient: startup serialization does not depend
# on allocating the full 75.6M model or using a GPU.
config = FixedChainConfig()
model = torch.nn.Linear(2, 2)
model.config = config
before = asdict(config)
args = TrainingArguments(output_dir=str(output), use_cpu=True, report_to='none')
state = TrainerState(is_world_process_zero=True, global_step=11000)
control = TrainerControl()
assert not hasattr(config, 'to_dict')
try:
    WandbCallback().on_train_begin(args, state, control, model=model)
except AttributeError as error:
    assert 'to_dict' in str(error)
else:
    raise AssertionError('Original startup failure was not reproduced')
assert wandb.run is None

try:
    launch['install_wandb_config_adapter'](source / 'wrong-source')
except ValueError as error:
    assert 'bound training source' in str(error)
else:
    raise AssertionError('Mismatched source root accepted')
assert not hasattr(config, 'to_dict')

spec = {'wandb': dict(enabled=True, resume_old_run=False,
                     project='oas-resume-offline-test', entity='offline-test',
                     run_name='oas-resume-config-smoke')}
receipt = launch['enable_wandb'](spec, source)
assert receipt == launch['install_wandb_config_adapter'](source)
assert config.to_dict() == before == asdict(config)
exported = config.to_dict()
exported['d_model'] = -1
assert config.d_model == before['d_model']
args = TrainingArguments(output_dir=str(output), use_cpu=True, report_to='none')
assert args.report_to == ['wandb']
assert args.run_name == spec['wandb']['run_name']
callback = WandbCallback()
try:
    callback.on_train_begin(args, state, control, model=model)
    assert wandb.run is not None
    assert wandb.run.settings.mode == 'offline'
    assert wandb.config['d_model'] == config.d_model
    assert wandb.config['layout_contract'] == config.layout_contract
    callback.on_log(args, state, control, model=model, logs={'loss': 1.5})
    callback.on_train_end(args, state, control, model=model)
finally:
    wandb.finish()
assert not torch.cuda.is_initialized()
print(json.dumps(dict(status='passed', source=str(source), reproduced=True,
                      callback_start_and_log=True, adapter=receipt)))
'''


@pytest.mark.parametrize('source', [ROOT, FROZEN], ids=['workspace', 'frozen-oas'])
def test_real_wandb_callback_with_bound_config(source, tmp_path):
    if not source.is_dir():
        pytest.skip('Historical frozen OAS bundle is not present in this checkout')
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(('WANDB_', 'RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_'))}
    environment.update(CUDA_VISIBLE_DEVICES='', USE_TF='0', WANDB_MODE='offline',
                       WANDB_PROJECT='oas-resume-offline-test', WANDB_ENTITY='offline-test',
                       WANDB_API_KEY='offline-test-only', WANDB_LOG_MODEL='false',
                       WANDB_WATCH='false', WANDB_DIR=str(tmp_path),
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    result = subprocess.run(
        [sys.executable, '-I', '-B', '-c', SMOKE, str(source),
         str(ROOT / 'scripts/research/launch_oas_resume.py'), str(tmp_path)],
        env=environment, text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"callback_start_and_log": true' in result.stdout
