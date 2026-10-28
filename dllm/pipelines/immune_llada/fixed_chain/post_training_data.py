"""Training-only annotations on the unchanged fixed-chain feature layout.

Run: python -m pytest scripts/tests/immune_llada/test_fixed_chain_post_training.py -q
The extra masks and labels never enter denoise's feature whitelist.
"""
import torch
from .layout import FixedChainCollator, ROLES
from ..data.records import BioSeqRecord, CHAIN_ROLE_TO_ID

REGION_ORDER = ('FR1', 'CDR1', 'FR2', 'CDR2', 'FR3', 'CDR3', 'FR4')


class PostTrainingCollator(FixedChainCollator):
    """Attach exact CDR positions, observed endpoints and auxiliary class labels."""
    def __call__(self, records):
        batch = super().__call__(records)
        batch['cdr_mask'] = torch.zeros_like(batch['eligible_mask'])
        batch['observed_terminal_mask'] = torch.zeros_like(batch['terminal_mask'])
        batch['binding_labels'] = torch.full((len(records),), -1, dtype=torch.long)
        batch['cdr_mapped_chains'] = torch.zeros(len(records), dtype=torch.long)
        for r, item in enumerate(records):
            record = BioSeqRecord.from_prepared_dict(item) if isinstance(item, dict) else item
            relation = record.labels.get('relation', record.metadata.get('relation'))
            if relation in ('binding', 'nonbinding'):
                batch['binding_labels'][r] = int(relation == 'binding')
            for chain in record.chains:
                if chain.role not in ROLES:
                    continue
                positions = batch['role_ids'][r].eq(CHAIN_ROLE_TO_ID[chain.role]).nonzero().flatten()
                synthetic = chain.metadata.get('synthetic_residue_mask', [])
                if not any(synthetic):
                    batch['observed_terminal_mask'][r, positions] = batch['terminal_mask'][r, positions]
                regions = chain.regions
                # Never guess alignment by substring search (CDRs may repeat).
                if not regions or ''.join(regions.get(k, '') for k in REGION_ORDER) != chain.sequence:
                    continue
                batch['cdr_mapped_chains'][r] += 1
                offset = 0
                for name in REGION_ORDER:
                    end = offset + len(regions.get(name, ''))
                    if name.startswith('CDR'):
                        batch['cdr_mask'][r, positions[offset:end]] = True
                    offset = end
        batch['cdr_mask'] &= batch['eligible_mask'] & batch['residue_mask']
        return batch


def observed_end_batch(batch, mask_token_id):
    """Hide unobserved template EOS as well as removing their loss labels."""
    result = dict(batch)
    uncertain = batch['terminal_mask'] & ~batch['observed_terminal_mask']
    result['input_ids'] = batch['input_ids'].masked_fill(uncertain, mask_token_id)
    result['labels'] = batch['labels'].masked_fill(uncertain, -100)
    result['eligible_mask'] = batch['eligible_mask'] & ~uncertain
    result['terminal_mask'] = batch['terminal_mask'] & ~uncertain
    return result


def relation_blind_features(model, batch):
    """A relation classifier must not read binding/nonbinding as an input token."""
    features = model.feature_args(batch)
    ids = features['input_ids'].clone()
    for name in ('<binding>', '<nonbinding>'):
        ids[ids == model.tokenizer.special_id(name)] = model.tokenizer.mask_token_id
    return {**features, 'input_ids': ids}


def observed_validation_batch(batch, mask_token_id):
    """Exclude unobserved ends from input, loss and endpoint-scoring populations."""
    result = observed_end_batch(batch, mask_token_id)
    evaluable = torch.zeros_like(result['target_slot_mask'])
    for row in range(len(result['input_ids'])):
        for uid in result['chain_ids'][row][result['observed_terminal_mask'][row]].unique():
            evaluable[row] |= result['chain_ids'][row].eq(uid)
    result['endpoint_evaluable_mask'] = evaluable
    return result
