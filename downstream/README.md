# BioSeq downstream tasks

Downstream scripts migrated from `/vepfs-mlp2/c20250601/251105016/project/airgen/AirGen-Dev/downstream`.

Data defaults: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/downstream`

## Aligned to `MultiChainOphiuchusAbModel`

These entry points use `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/common.py` and the Ophiuchus-Ab checkpoint:

| Task | Script |
|------|--------|
| Heavy → light | `comp_chain/generate_light_from_csv.py` |
| CDR infill (SAbDab) | `infill/zeroshot_cdr.py` |
| CDR infill (SAb23H2) | `infill/zeroshot_sab23h2.py` |
| Humanization | `humanization/humanize.py` |
| FLAb property regression | `flab/finetune_flab.py` |
| Developability regression | `dev/finetune_dev.py` |
| Specificity classification | `specificity/hd_flu_cov_paired.py` |
| Shared embeddings | `embeddings.py` |

## Legacy AirGen copies (optional third-party deps)

The following files were copied for reference and evaluation utilities. They may still import `byprot` or require tools such as `anarci`, `abnumber`, `abnativ`, or `wandb`:

- `comp_chain/eval_scripts/`
- `humanization/eval_scripts/`, `humanization/structure_rmsd.py`, `humanization/oasis_human_score.py`, ...
- `flab/finetune_flab_align.py`, `flab/finetune_flab_esm_ppi.py`
- `dev/finetune_dev_pplm.py`, `dev/tutorial.py`
- `in_silico/finetune_in_silico.py`
- `specificity/HD_Flu_Cov-paired.py`
- `infill/zeroshot_SAb23H2.py` (AirGen filename; prefer `zeroshot_sab23h2.py`)

Prefer the aligned scripts in the first table for Ophiuchus-Ab inference and embedding extraction.

## Examples

```bash
python /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/comp_chain/generate_light_from_csv.py \
  --df-path /path/to/input.csv --output-file /path/to/output.csv

python /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/infill/zeroshot_cdr.py \
  --test-set /vepfs-mlp2/c20250601/251105016/project/dllm_test/data/downstream/cdr_infilling/sabdab/cdrh3 \
  --mode cdrh3

python /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/humanization/humanize.py \
  --pdb-dir /path/to/cif --info-csv-fpath /path/to/chain_pairs.csv --output-csv humanized.csv

python /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/flab/finetune_flab.py \
  --train-csv /path/train.csv --test-csv /path/test.csv --target-col expression
```
