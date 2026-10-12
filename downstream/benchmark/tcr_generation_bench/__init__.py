"""TCR generation benchmark package (legacy T4 plus strict public beta Track A).

Run the public beta track with::

    /vepfs-mlp2/c20250601/251105016/conda/envs/pllm/bin/python \
      /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_generation_bench/run_public_beta.py --help

Three settings share one metric suite (``common.metrics``) and one eval set
(``data/tcr_generation_bench/``):

  A  unconditional CDR3b repertoire distribution match (+ OLGA pgen calibration)
  B  epitope-conditioned CDR3b design  (main leaderboard, TCRT5 protocol)
  C  full-length alpha/beta assembly   (validity + CDR3 extraction -> B metrics)

The newer ``track_a`` / ``run_public_beta`` path is deliberately separate: it
uses 1,000 raw candidates, strict 7--24-aa C...F/W validity, target-specific
exact/recovery/GIANA/Jaccard metrics, and excludes reconstruction-only methods.
"""

BENCH_VERSION = "irbench.tcr_generation.v2"
PUBLIC_BETA_TRACK_VERSION = "public_epitope_conditioned_cdr3beta_track_a_v1"
TCRDESIGN_QUALITATIVE_ALPHA_VERSION = "tcrdesign_generated_beta_to_alpha_v1"
TCRT5_FULL_EVAL_VERSION = "tcrt5_full_eval_v1"
