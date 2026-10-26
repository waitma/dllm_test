"""Activate pllm/protenix_abtcr; python -m pytest this_file.py -q.

Check native optimizer/RNG continuation against uninterrupted training, not just reload.
"""
import json
import numpy as np
import pytest
import torch
from torch.utils.data import ConcatDataset
from transformers import TrainingArguments, TrainerCallback
from dllm.pipelines.immune_llada.data.records import BioSeqRecord, BioSeqChain
from dllm.pipelines.immune_llada.fixed_chain.post_training import PostTrainingModel
from dllm.pipelines.immune_llada.fixed_chain.post_training_data import PostTrainingCollator
from scripts.research.fixed_chain_validation import TokenValidationTrainer
from scripts.research.post_training_trainer import PostTrainingTrainerMixin
from scripts.research.full_post_training_data import IndexedRows
from scripts.tests.immune_llada.test_fixed_chain_post_training import fixture


class Trainer(PostTrainingTrainerMixin, TokenValidationTrainer):
    pass


class StopAtTwo(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 2:
            control.should_training_stop = True
            control.should_save = True
        return control


@pytest.mark.parametrize('recipe', ['task_mix', 'lora'])
def test_native_checkpoint_resume_matches_uninterrupted_training(tmp_path, recipe):
    def build(path, stop=False):
        _, model, _ = fixture(recipe)
        rows = [BioSeqRecord([BioSeqChain('ACD', 'peptide'), BioSeqChain(seq, 'tcr_beta')],
                             'tcr_epitope', 'test') for seq in ('CASSF', 'CATF', 'CASRF', 'CAVF')]
        data = ConcatDataset([rows[:2] * 4, rows[2:] * 4])
        args = TrainingArguments(output_dir=str(path), use_cpu=True, report_to='none', disable_tqdm=True,
            per_device_train_batch_size=2, gradient_accumulation_steps=1, max_steps=4,
            learning_rate=.001, weight_decay=.01, lr_scheduler_type='cosine', warmup_steps=0,
            save_strategy='steps', save_steps=2, eval_strategy='no', logging_strategy='no',
            remove_unused_columns=False, seed=123, data_seed=123, dataloader_num_workers=0,
            dataloader_drop_last=True, optim='adamw_torch')
        return Trainer(model=model, args=args, train_dataset=data, data_collator=PostTrainingCollator(model.config),
                       callbacks=[StopAtTwo()] if stop else [])
    uninterrupted = build(tmp_path / 'continuous')
    uninterrupted.train()
    stopped = build(tmp_path / 'resumed', stop=True)
    stopped.train()
    checkpoint = tmp_path / 'resumed/checkpoint-2'
    assert {'model.safetensors', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth',
            'research_config.json', 'immune_config.json', 'vocabulary.json'} <= {p.name for p in checkpoint.iterdir()}
    assert json.loads((checkpoint / 'research_config.json').read_text())['recipe'] == recipe
    restored = PostTrainingModel.from_pretrained(checkpoint)
    for name, value in stopped.model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
    resumed = build(tmp_path / 'resumed')
    resumed.train(resume_from_checkpoint=str(checkpoint))
    assert resumed.state.global_step == 4
    for name, value in uninterrupted.model.state_dict().items():
        torch.testing.assert_close(value, resumed.model.state_dict()[name], rtol=0, atol=0)
    left, right = uninterrupted.optimizer.state_dict(), resumed.optimizer.state_dict()
    for key in left['state']:
        for name, value in left['state'][key].items():
            torch.testing.assert_close(value, right['state'][key][name], rtol=0, atol=0)


def test_full_data_uint64_indices_become_python_ints():
    class Rows:
        def __getitem__(self, index):
            assert type(index) is int
            return index * 2
    dataset = IndexedRows(Rows(), np.array([2, 7, 11], dtype=np.uint64))
    assert len(dataset) == 3 and dataset[1] == 14


def test_formal_launch_rejects_unconfirmed_decision(tmp_path):
    from scripts.research.prepare_fixed_chain_final_post_training import verify_decision
    path = tmp_path / 'selection.json'
    path.write_text(json.dumps({'selected_family': 'task_mix', 'status': 'pilot_only'}))
    with pytest.raises(ValueError, match='completed confirmation'):
        verify_decision(path)
