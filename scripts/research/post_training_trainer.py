"""Native Trainer adaptations for full-data post-training.

Activate pllm/protenix_abtcr; run test_fixed_chain_formal_post_training.py.
Keep training recipe masks separate from observed-end-only validation semantics.
"""
from dllm.pipelines.immune_llada.data.balanced_sampling import FullCoverageSourceSampler
from dllm.pipelines.immune_llada.fixed_chain.post_training_data import observed_validation_batch


class PostTrainingTrainerMixin:
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not model.training:
            unwrapped = self.accelerator.unwrap_model(model)
            inputs = observed_validation_batch(inputs, unwrapped.tokenizer.mask_token_id)
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

    def _get_train_sampler(self, train_dataset=None):
        dataset = self.train_dataset if train_dataset is None else train_dataset
        batch = (self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps
                 * self.args.world_size)
        return FullCoverageSourceSampler([len(part) for part in dataset.datasets],
                                         effective_batch_size=batch, seed=self.args.data_seed)
