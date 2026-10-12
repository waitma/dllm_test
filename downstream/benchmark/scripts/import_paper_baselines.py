#!/usr/bin/env python
"""Write immutable, explicitly labelled anchor-paper baseline values.

Run from the repository root after editing a transcription::

    python downstream/benchmark/scripts/import_paper_baselines.py

The output is a *reference table*, not a locally reproduced leaderboard.  Every
row is labelled ``paper_reported`` and carries the paper table/figure location.
Local reruns remain in their task output directories and must never overwrite
these rows.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

BENCH = Path(__file__).resolve().parents[1]
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

from common.baseline_provenance import EXACT, PAPER_REPORTED  # noqa: E402

OUT = BENCH / "outputs" / "external"
CSV_OUT = OUT / "paper_reported_baselines.csv"
REGISTRY_OUT = OUT / "baseline_registry.json"

T1_CITATION = "doi:10.1038/s41592-025-02910-0"
T2_CITATION = "doi:10.1093/nargab/lqaf150"
T3_CITATION = "arXiv:2406.06397; SCEPTR Table SI"
T4_CITATION = "doi:10.1038/s42256-025-01096-6"
MINT_CITATION = "doi:10.1038/s41467-025-67971-3"
OPHIUCHUS_CITATION = "doi:10.64898/2026.02.02.703197"
NBBENCH_CITATION = "doi:10.1088/2632-2153/ae20ec; arXiv:2505.02022"


def _row(
    benchmark: str,
    task: str,
    method: str,
    metric: str,
    value: float,
    *,
    split: str = "",
    std: float | None = None,
    citation: str,
    location: str,
    notes: str = "",
) -> dict:
    return {
        "benchmark": benchmark,
        "task": task,
        "method": method,
        "split": split,
        "metric": metric,
        "value": float(value),
        "std": None if std is None else float(std),
        "evidence_type": PAPER_REPORTED,
        "protocol_alignment": EXACT,
        "citation": citation,
        "source_location": location,
        "notes": notes,
    }


def t1_rows() -> list[dict]:
    # Nature Methods 2025 Supplementary Table 4-1/4-2, original pretrained
    # CDR3beta-only models, antigen-specific negatives.
    seen = {
        "PanPep": (0.5065, 0.4979), "TPBTE-mc": (0.5053, 0.4993),
        "TPBTE-vdj": (0.5000, 0.5000), "SETE": (0.5463, 0.5629),
        "ERGO-AE-mc": (0.5152, 0.5798), "epiTCR": (0.5566, 0.6023),
        "ERGO-lstm-mc": (0.5614, 0.6091), "ERGO-AE-vdj": (0.5428, 0.6150),
        "ERGO-lstm-vdj": (0.5771, 0.6279), "NetTCR": (0.5880, 0.6457),
        "TEINet-large": (0.5929, 0.6588), "TEINet-small": (0.5959, 0.6593),
        "TEPCAM": (0.5988, 0.6693), "TEIM": (0.6025, 0.6756),
        "ATM-TCR": (0.6480, 0.6957),
    }
    unseen = {
        "PanPep": (0.4825, 0.4771), "epiTCR": (0.4887, 0.4879),
        "TPBTE-mc": (0.5037, 0.4988), "TPBTE-vdj": (0.5000, 0.5000),
        "TEINet-small": (0.5058, 0.5003), "ERGO-AE-vdj": (0.4995, 0.5041),
        "NetTCR": (0.5005, 0.5062), "ERGO-lstm-mc": (0.5108, 0.5067),
        "TEINet-large": (0.5082, 0.5098), "TEPCAM": (0.5120, 0.5128),
        "TEIM": (0.5161, 0.5159), "ERGO-lstm-vdj": (0.5116, 0.5171),
        "ERGO-AE-mc": (0.5051, 0.5206), "ATM-TCR": (0.5218, 0.5208),
    }
    rows = []
    for split, values, table in (
        ("seen", seen, "Supplementary Table 4-1 (S_Data1, AS)"),
        ("unseen", unseen, "Supplementary Table 4-2 (U_Data1, AS)"),
    ):
        for method, (auroc, auprc) in values.items():
            rows.append(_row("t1_binding", "original_cdr3b_as", method, "auroc", auroc,
                             split=split, citation=T1_CITATION, location=table))
            rows.append(_row("t1_binding", "original_cdr3b_as", method, "auprc", auprc,
                             split=split, citation=T1_CITATION, location=table))
    return rows


def t2_rows() -> list[dict]:
    # Figure 3A at the paper's main operating point (Th=1, i.e. cluster size > 1).
    # Taken from the value labels printed on the bars of the published figure.
    #
    # Earlier revisions carried (0.25, 0.66) for HD / LD / iSMART / GIANA.  That
    # is the *group* summary the Results text quotes for those four jointly
    # ("similar low retention (0.25 +/- 0.06) and intermediate purity scores
    # (0.66 +/- 0.09)"), not their individual values, so four of the nine methods
    # were reference rows carrying a group mean.  Cross-checks: the group mean of
    # the retentions below is 0.2525 -> 0.25 and of the purities 0.665 -> 0.66;
    # the purity half-range is exactly 0.09; every value lands in the qualitative
    # bin Table 1 assigns it (Very Low [0-0.2) / Low [0.2-0.4) / Middle
    # [0.4-0.6) / High [0.6-0.8) / Very High [0.8-1]).
    #
    # Conflict inside the paper, disclosed rather than silently resolved:
    # Supplementary Table S2's clustered-sequence counts over the Rmd's fixed
    # denominators give HD 2564/8395 = 0.31 and LD 2248/8395 = 0.27, i.e. HD and
    # LD swapped relative to Fig 3A.  The remaining seven methods agree exactly
    # across both sources.  Fig 3A is used here: it is the main-text result
    # figure, and its ordering is the one consistent with LD admitting indels and
    # therefore clustering more sequences than HD.
    values = {
        # method: (retention, purity, perc_clusters, perc_CDR3s)
        "HD": (0.27, 0.69, 0.36, 0.34),
        "LD": (0.31, 0.57, 0.34, 0.20),
        "TCRMatch": (0.09, 0.82, 0.64, 0.63),
        "iSMART": (0.18, 0.75, 0.45, 0.50),
        "GIANA": (0.25, 0.65, 0.36, 0.35),
        "clusTCR": (0.09, 0.99, 0.97, 0.98),
        "GLIPH2": (0.22, 0.80, 0.51, 0.48),
        "DeepTCR": (0.92, 0.63, 0.20, 0.28),
        "TCRdist3": (0.44, 0.42, 0.07, 0.05),
    }
    metrics = ("retention", "purity", "perc_clusters", "perc_CDR3s")
    swap_note = ("Fig 3A vs Supplementary Table S2 disagree on HD/LD retention; "
                 "S2 counts give HD 0.31 and LD 0.27. Fig 3A value used.")
    rows = []
    for method, vals in values.items():
        for metric, value in zip(metrics, vals):
            notes = swap_note if (method in ("HD", "LD") and metric == "retention") else ""
            rows.append(_row("t2_clustering", "nargab_native", method, metric, value,
                             citation=T2_CITATION,
                             location="Figure 3A (Th=1, cluster size > 1)",
                             notes=notes))
    return rows


def t3_rows() -> list[dict]:
    targets = ["GIL", "NLV", "SPR", "TFE", "TTD", "YLQ"]
    values = {
        "SCEPTR": [0.911, 0.691, 0.728, 0.976, 0.708, 0.775],
        "TCRdist": [0.904, 0.648, 0.695, 0.970, 0.720, 0.762],
        "CDR3-Levenshtein": [0.872, 0.632, 0.610, 0.964, 0.600, 0.743],
        "TCR-BERT": [0.876, 0.655, 0.637, 0.966, 0.576, 0.698],
        "ESM2-8M": [0.831, 0.652, 0.604, 0.937, 0.579, 0.697],
        "ProtBERT": [0.845, 0.629, 0.575, 0.950, 0.564, 0.669],
    }
    # These six methods do NOT receive the same input fields, which is a property
    # of the anchor paper's own protocol and is reproduced faithfully by our
    # harness.  Recording it per row keeps the registry from being read as an
    # equal-input comparison, since our own model only ever sees the two CDR3
    # loops and therefore is not comparable head-to-head with the top two.
    input_fields = {
        "SCEPTR": "CDR3b + CDR3a + TRBV + TRAV",
        "TCRdist": "CDR3b + CDR3a + TRBV + TRAV + TRBJ + TRAJ",
        "CDR3-Levenshtein": "CDR3b + CDR3a",
        "TCR-BERT": "CDR3b only",
        "ESM2-8M": "CDR3b + CDR3a",
        "ProtBERT": "CDR3b + CDR3a",
    }
    return [
        _row("t3_representation", "sceptr_six_pmhc_k200", method, "auroc", value,
             split=target, citation=T3_CITATION, location="Table SI, k=200",
             notes=(f"input fields: {input_fields[method]}. Only the k=200 column is "
                    "registered; our local runs additionally sweep k in {1,5,20,100}, "
                    "for which the paper's per-pMHC values were not transcribed."))
        for method, vals in values.items() for target, value in zip(targets, vals)
    ]


def mint_rows() -> list[dict]:
    # Nature Communications Source Data, sheet "Figure 2c-h; Table S1".  The
    # primary metric differs by task and is preserved here exactly.  For every
    # task except PDB-Bind, the paper's better separate-embedding (``_sep``)
    # result is used.  Source XLSX SHA256:
    # 28f145ca997a73790e69f07c3a674326b1c751e68755a6b8a0be55dfd2e6ba61.
    values = {
        "MINT": {
            "HumanPPI": ("accuracy", 0.8796296300, 0.0069289950),
            "YeastPPI": ("accuracy", 0.6869712350, 0.0106343530),
            "Gold-standard PPI": ("auprc", 0.6871565080, 0.0032556620),
            "MutationalPPI": ("auprc", 0.6449405850, 0.0407828340),
            "SKEMPI": ("pearson", 0.2973555440, 0.1311916000),
            "PDB-Bind": ("pearson", 0.5929999870, 0.0424326810),
        },
        "ESM2-650M": {
            "HumanPPI": ("accuracy", 0.8703703700, 0.0145814960),
            "YeastPPI": ("accuracy", 0.6074450080, 0.0083752070),
            "Gold-standard PPI": ("auprc", 0.6639443710, 0.0064906040),
            "MutationalPPI": ("auprc", 0.6350616000, 0.0595692460),
            "SKEMPI": ("pearson", 0.2562498360, 0.1617750680),
            "PDB-Bind": ("pearson", 0.5759221320, 0.0282766490),
        },
        "ESM2-150M": {
            "HumanPPI": ("accuracy", 0.8537037040, 0.0026189140),
            "YeastPPI": ("accuracy", 0.5972927240, 0.0011964580),
            "Gold-standard PPI": ("auprc", 0.6774860210, 0.0040293580),
            "MutationalPPI": ("auprc", 0.6414483480, 0.0486645190),
            "SKEMPI": ("pearson", 0.2916617680, 0.1615542820),
            "PDB-Bind": ("pearson", 0.5785236410, 0.0384835650),
        },
        "ESM-1b": {
            "HumanPPI": ("accuracy", 0.8407407410, 0.0171733680),
            "YeastPPI": ("accuracy", 0.6074450080, 0.0031655310),
            "Gold-standard PPI": ("auprc", 0.6630061790, 0.0031679810),
            "MutationalPPI": ("auprc", 0.6274973210, 0.0406057390),
            "SKEMPI": ("pearson", 0.2512056310, 0.1538185070),
            "PDB-Bind": ("pearson", 0.5817493620, 0.0411380240),
        },
        "ESM2-3B": {
            "HumanPPI": ("accuracy", 0.8759259260, 0.0094426290),
            "YeastPPI": ("accuracy", 0.6218274110, 0.0135891530),
            "Gold-standard PPI": ("auprc", 0.6513395200, 0.0033411860),
            "MutationalPPI": ("auprc", 0.6394434430, 0.0605691790),
            "SKEMPI": ("pearson", 0.2307382800, 0.1807675020),
            "PDB-Bind": ("pearson", 0.5640292060, 0.0674990980),
        },
        "ProGen2-Large": {
            "HumanPPI": ("accuracy", 0.8203703700, 0.0343467350),
            "YeastPPI": ("accuracy", 0.5837563450, 0.0109657200),
            "Gold-standard PPI": ("auprc", 0.6146099250, 0.0013535990),
            "MutationalPPI": ("auprc", 0.6196785890, 0.0729102780),
            "SKEMPI": ("pearson", 0.3036013890, 0.1204674760),
            "PDB-Bind": ("pearson", 0.5773127880, 0.0414020320),
        },
        "ProtT5-UniRef": {
            "HumanPPI": ("accuracy", 0.8703703700, 0.0130945700),
            "YeastPPI": ("accuracy", 0.5879864640, 0.0093446370),
            "Gold-standard PPI": ("auprc", 0.6545670250, 0.0096628300),
            "MutationalPPI": ("auprc", 0.6386542290, 0.0597781480),
            "SKEMPI": ("pearson", 0.2242753900, 0.1098644240),
            "PDB-Bind": ("pearson", 0.5929614560, 0.0348484290),
        },
        "ProtT5-BFD": {
            "HumanPPI": ("accuracy", 0.8537037040, 0.0145814960),
            "YeastPPI": ("accuracy", 0.6125211510, 0.0176249290),
            "Gold-standard PPI": ("auprc", 0.6536289270, 0.0039032730),
            "MutationalPPI": ("auprc", 0.6273568930, 0.0605505290),
            "SKEMPI": ("pearson", 0.2699956570, 0.0653456210),
            "PDB-Bind": ("pearson", 0.5888095640, 0.0309370290),
        },
    }
    return [
        _row("mint_generalppi", task, method, metric, value, std=std,
             citation=MINT_CITATION,
             location="Source Data: Figure 2c-h; Table S1",
             notes="separate-chain embedding except PDB-Bind concatenated input")
        for method, tasks in values.items()
        for task, (metric, value, std) in tasks.items()
    ]


def mint_flab_rows() -> list[dict]:
    # Source Data Figure 3b.  This is the paper-aligned 10-fold nested-Ridge R2
    # table; it is intentionally separate from our historical 5-fold Spearman.
    tasks = ["trastuzumab_kd", "d44_Kd", "g6_Kd", "g6_er"]
    values = {
        "AbLang": [(0.293, 0.117), (0.246, 0.038), (0.244, 0.034), (0.439, 0.027)],
        "AntiBERTy": [(0.239, 0.102), (0.217, 0.056), (0.199, 0.025), (0.401, 0.032)],
        "IgBert-unpaired": [(0.278, 0.094), (0.181, 0.040), (0.177, 0.018), (0.347, 0.023)],
        "IgBert": [(0.306, 0.114), (0.131, 0.047), (0.174, 0.032), (0.400, 0.023)],
        "IgT5-unpaired": [(0.299, 0.119), (0.245, 0.040), (0.206, 0.029), (0.567, 0.025)],
        "IgT5": [(0.274, 0.070), (0.297, 0.057), (0.179, 0.014), (0.548, 0.067)],
        "MINT": [(0.398, 0.078), (0.3788, 0.058), (0.253, 0.041), (0.657, 0.023)],
    }
    return [
        _row("flab", task, method, "r2", mean, std=std, citation=MINT_CITATION,
             location="Source Data: Figure 3b",
             notes="10-fold outer CV, nested 5-fold Ridge selection")
        for method, vals in values.items() for task, (mean, std) in zip(tasks, vals)
    ]


def ophiuchus_rows() -> list[dict]:
    """Ophiuchus-Ab Tables 1-3, transcribed from the bioRxiv full text PDF.

    Table 2 prints AAR as percentages (75.50) and Table 1 as fractions (0.811).
    Both are stored as percentages here so one benchmark carries one unit, which
    also matches how RESULTS.md ss0.5 already tabulates them.
    """
    rows = []

    # -- Table 2: SAbDab, 10-fold CV, heavy-chain CDRs.  Sixteen models.
    # (cdrh1, cdrh2, cdrh3, train/test overlap, fine-tuned on the nine train folds)
    sabdab = {
        "LSTM":          (40.98, 28.50, 15.69, False, True),
        "C-LSTM":        (40.93, 29.24, 15.48, False, True),
        "RefineGNN":     (39.40, 37.06, 21.13, False, True),
        "C-RefineGNN":   (33.19, 33.53, 18.88, False, True),
        "HERN":          (41.71, 35.46, 24.75, False, True),
        "C-HERN":        (47.36, 41.45, 28.47, False, True),
        "DiffAb":        (61.34, 37.66, 25.79, False, True),
        "AbDiffuser":    (62.76, 41.10, 29.58, False, True),
        "MEAN":          (58.29, 47.15, 36.38, False, True),
        "dyMEAN":        (63.52, 55.41, 37.19, False, True),
        "ADesigner":     (64.34, 55.52, 37.37, False, True),
        "RADD":          (67.64, 58.87, 37.71, False, True),
        "AntiBERTy":     (76.70, 71.10, 42.70, True,  False),
        "AbLang2":       (76.30, 70.60, 42.70, True,  False),
        "AbBFN":         (70.30, 64.90, 31.50, False, False),
        "Ophiuchus-Ab":  (75.50, 70.18, 43.55, False, False),
    }
    for method, (h1, h2, h3, overlap, finetuned) in sabdab.items():
        note = ("paper marks train/test overlap" if overlap
                else "paper marks no train/test overlap")
        note += "; fine-tuned on the nine training folds" if finetuned else "; zero-shot"
        for cdr, value in zip(("cdrh1", "cdrh2", "cdrh3"), (h1, h2, h3)):
            rows.append(_row("antibody_cdr", "sabdab_cdr_infill", method,
                             "aar_percent", value, split=cdr,
                             citation=OPHIUCHUS_CITATION, location="Table 2",
                             notes=note))

    # -- Table 1: SAb23H2, all six CDRs.  MEAN only has a CDR-H3 entry.
    # Printed as fractions in the paper; x100 here.  The (IgFold) / (AF3) suffix
    # is the structure predictor supplying the initial input, so the two variants
    # of a method are separate baselines rather than a repeat measurement.
    sab23h2 = {
        "DiffAb (IgFold)": (0.597, 0.598, 0.421, 0.642, 0.363, 0.214),
        "DiffAb (AF3)":    (0.608, 0.599, 0.424, 0.637, 0.394, 0.226),
        "MEAN (IgFold)":   (None, None, None, None, None, 0.248),
        "MEAN (AF3)":      (None, None, None, None, None, 0.246),
        "dyMEAN":          (0.633, 0.634, 0.570, 0.742, 0.627, 0.294),
        "IgGM (IgFold)":   (0.750, 0.743, 0.635, 0.740, 0.644, 0.360),
        "IgGM (AF3)":      (0.737, 0.735, 0.602, 0.739, 0.639, 0.330),
        "Ophiuchus-Ab":    (0.811, 0.801, 0.737, 0.748, 0.686, 0.368),
    }
    for method, vals in sab23h2.items():
        for cdr, value in zip(("l1", "l2", "l3", "h1", "h2", "h3"), vals):
            if value is None:
                continue
            rows.append(_row("antibody_cdr", "sab23h2_cdr_infill", method,
                             "aar_percent", value * 100.0, split=cdr,
                             citation=OPHIUCHUS_CITATION, location="Table 1",
                             notes="paper prints fractions; stored x100 as percent"))

    # -- Table 3: light-chain pairing, two conditioning settings.
    # The two settings are NOT interchangeable and trade off against each other:
    # three-residue prompting raises reference agreement sharply (Ophiuchus-Ab
    # V-family 0.169 -> 0.894, chain match 55.6% -> 99.7%) while lowering
    # ImmunoMatch (0.701 -> 0.695).  Any local row must state which setting it
    # matches before it is compared.
    pairing_metrics = ("immunomatch", "better_than_reference_percent",
                       "chain_match_percent", "v_gene", "j_gene", "v_family",
                       "j_family", "diversity", "validity_percent", "w_property")
    pairing = {
        "no_prompt": {
            "LiChen":       (0.657, 46.2, 62.3, 0.049, 0.156, 0.215, 0.157, 0.515, 100.0, 0.140),
            "p-IgGen":      (0.674, 50.5, 50.5, 0.045, 0.139, 0.171, 0.142, 0.709, 99.1, 0.103),
            "Ophiuchus-Ab": (0.701, 51.6, 55.6, 0.056, 0.150, 0.169, 0.152, 0.676, 100.0, 0.072),
        },
        "prompt_3res": {
            "LiChen":       (0.633, 42.7, 99.7, 0.298, 0.290, 0.890, 0.291, 0.254, 99.9, 0.150),
            "p-IgGen":      (0.669, 45.3, 98.4, 0.313, 0.283, 0.874, 0.288, 0.390, 99.2, 0.118),
            "Ophiuchus-Ab": (0.695, 49.2, 99.7, 0.356, 0.273, 0.894, 0.276, 0.335, 100.0, 0.066),
        },
    }
    setting_note = {
        "no_prompt": "light chain conditioned on the heavy chain only",
        "prompt_3res": ("light chain conditioned on the heavy chain plus the first three "
                        "light-chain residues; raises reference agreement and lowers ImmunoMatch"),
    }
    for setting, methods in pairing.items():
        for method, vals in methods.items():
            for metric, value in zip(pairing_metrics, vals):
                rows.append(_row("light_chain_pairing", "cognate_light_design", method,
                                 metric, value, split=setting,
                                 citation=OPHIUCHUS_CITATION, location="Table 3",
                                 notes=(setting_note[setting]
                                        + "; 500 held-out heavy chains, 8 light chains generated each"
                                        + ("; lower is better" if metric == "w_property" else ""))))
    return rows


def t4_rows() -> list[dict]:
    # TCRT5 Supplementary Table 2, twelve model variants x seven metrics.  Reuse
    # the audited transcription in the eval module instead of keeping a second
    # copy of 84 constants.
    #
    # These are the *paper's own* protocol and metric definitions (top-20
    # references, k=100, over its full target set).  They are NOT the sparse-13
    # view that RESULTS.md ss0.4 reports, and the metric names deliberately differ
    # (``f1_mean`` / ``native_recovery_mean`` / ``map`` here vs precision / recall
    # / f1 / d_edit / seq_recovery there).  Do not put the two side by side.
    #
    # Verification status per metric comes from the stored reconstruction check
    # ``outputs/tcrt5_full_eval/author_release/main_top20_k100/
    # paper_main_comparison.csv`` (84 rows, tolerance 0.00051).
    from tcr_generation_bench.tcrt5_full_eval import PAPER_MAIN_TABLE

    verification = {
        "f1_mean": "independently recomputed from the released sequences; 12/12 agree within 0.00051",
        "f1_median": "independently recomputed from the released sequences; 12/12 agree exactly",
        "native_recovery_mean": "independently recomputed from the released sequences; 12/12 agree within 0.00051",
        "native_recovery_median": "independently recomputed from the released sequences; 11/12 agree, worst residual 0.00064",
        "total_unique": "independently recomputed from the released sequences; 12/12 agree exactly",
        "char_bleu": ("greedy hypotheses were never released, so this rests on the authors' numeric "
                      "companion table; 9/12 agree within 0.00051, worst residual 0.0026"),
        "map": ("NOT independently verifiable: the paper's cumulative model log-likelihood ranking "
                "scores were never released, so only an ordered-release-list proxy exists. 0/12 agree, "
                "delta up to 0.189. Do not compare against any local mAP."),
    }
    return [
        _row("t4_generation", "tcr_cdr3b_conditional_design", method, metric, value,
             split="paper_full_top20_k100", citation=T4_CITATION,
             location="Supplementary Table 2", notes=verification[metric])
        for method, vals in PAPER_MAIN_TABLE.items()
        for metric, value in vals.items()
    ]


def nbbench_rows() -> list[dict]:
    # Reuse the existing audited Table-5 transcription rather than duplicating
    # 121 numeric constants in two files.
    from import_nbbench_results import OFFICIAL, TASK_METRIC

    rows = []
    tasks = list(TASK_METRIC)
    for method, vals in OFFICIAL.items():
        for task, value in zip(tasks, vals):
            rows.append(_row("nbbench", task, method, TASK_METRIC[task], value,
                             citation=NBBENCH_CITATION, location="Table 5, 3-seed mean",
                             notes="frozen backbone plus learned MLP head"))
    return rows


# Benchmarks frozen by README section 0 (the 2026-08-29 scope narrowing to
# "TCR T1-T4 + antibody generation"). Their rows stay in the CSV as an archive --
# deleting them would destroy transcribed paper values that are expensive to
# recover -- but they are flagged so no consumer can pull them into a headline by
# accident. Filter on ``in_scope`` rather than maintaining a second file.
FROZEN_BENCHMARKS = {
    "nbbench": "nanobody track, split from the AB + TCR main line",
    "mint_generalppi": "general protein PPI, not an immune receptor task",
    "flab": "anchor is MINT rather than the antibody main line",
}


def registry() -> dict:
    return {
        "schema_version": 2,
        "policy": "Always use paper_reported as the primary baseline row; local reruns are separate and never fill missing paper cells.",
        "scope_policy": (
            "Every row carries in_scope. Frozen benchmarks (" +
            ", ".join(sorted(FROZEN_BENCHMARKS)) + ") are retained for archive "
            "only: never quote them in a headline table and do not update them."
        ),
        "frozen_benchmarks": FROZEN_BENCHMARKS,
        "benchmarks": {
            "t1_binding": {
                "paper_reference": "paper_reported_baselines.csv",
                "local_status": (
                    "paper original-model values are primary; official checkpoint/code reruns "
                    "on original.zip are separate diagnostics; missing paper cells remain missing"
                ),
            },
            "t2_clustering": {"paper_reference": "paper_reported_baselines.csv", "local_status": "LD indel bug fixed and rerun; paper-primary because the paper-era database is unavailable"},
            "t3_representation": {"paper_reference": "paper_reported_baselines.csv", "local_status": "local reconstruction; compare data/input/metric/aggregation/selection explicitly; unknown or mismatched protocols have no delta or rank"},
            "t4_generation": {
                "paper_reference": "paper_reported_baselines.csv (Supplementary Table 2) + official benchmark_data_w_preds.csv",
                "local_status": (
                    "official-artifact rescoring; local generators are separate rows. "
                    "Supplementary Table 2 uses the paper's own protocol and metric names and "
                    "must not be merged with the sparse-13 view. mAP is not independently "
                    "verifiable (ranking scores unreleased). Cross-model unseen comparison "
                    "requires the fixed bioseq_unseen_common set; bioseq_unseen membership "
                    "varies with each run's coverage."
                ),
            },
            "mint_generalppi": {"paper_reference": "paper_reported_baselines.csv", "local_status": "mismatched local cap/data/head; paper rows are primary baseline"},
            "antibody_cdr": {"paper_reference": "paper_reported_baselines.csv", "local_status": "paper rows primary; local masked-fill reruns labelled separately"},
            "flab": {"paper_reference": "paper_reported_baselines.csv", "local_status": "historical 5-fold Spearman is not paper Table/Figure protocol"},
            "nbbench": {"paper_reference": "paper_reported_baselines.csv", "local_status": "paper Table 5 primary; local sklearn probes are non-comparable diagnostics"},
            "light_chain_pairing": {
                "paper_reference": "paper_reported_baselines.csv (Ophiuchus-Ab Table 3)",
                "local_status": (
                    "Table 3 reports two conditioning settings that trade off against each "
                    "other, so every local row must declare which one it matches before being "
                    "compared: heavy-chain only, or heavy chain plus the first three light-chain "
                    "residues. Prompting lifts Ophiuchus-Ab chain match 55.6% -> 99.7% and "
                    "V-family 0.169 -> 0.894 while lowering ImmunoMatch 0.701 -> 0.695. Protocol "
                    "is 500 held-out heavy chains with 8 light chains generated each."
                ),
            },
            "humanization": {"paper_reference": None, "local_status": "local reference-reconstruction diagnostic; no functional paper-comparable baseline"},
        },
    }


def main() -> None:
    rows = (
        t1_rows() + t2_rows() + t3_rows() + t4_rows() + mint_rows()
        + mint_flab_rows() + ophiuchus_rows() + nbbench_rows()
    )
    out = pd.DataFrame(rows).sort_values(
        ["benchmark", "task", "method", "split", "metric"], kind="stable"
    )
    out["in_scope"] = ~out["benchmark"].isin(FROZEN_BENCHMARKS)
    out["frozen_reason"] = out["benchmark"].map(FROZEN_BENCHMARKS).fillna("")
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(CSV_OUT, index=False)
    REGISTRY_OUT.write_text(json.dumps(registry(), indent=2) + "\n", encoding="utf-8")
    n_frozen = int((~out["in_scope"]).sum())
    print(f"wrote {len(out)} paper-reported rows -> {CSV_OUT}")
    print(f"  in scope {len(out) - n_frozen}, frozen archive {n_frozen} "
          f"({out.loc[~out['in_scope'], 'benchmark'].value_counts().to_dict()})")
    print(f"wrote benchmark policy -> {REGISTRY_OUT}")


if __name__ == "__main__":
    main()
