"""Infer dominant STRING functional channel from detailed link subscores.

STRING ``protein.links.detailed.v12.0.txt.gz`` columns::

    protein1 protein2 combined_score nscore fscore pscore ascore escore dscore tscore

Evidence channels (STRING v12 documentation):
- nscore: neighborhood / genomic context
- fscore: gene fusion
- pscore: physical / co-structure evidence  -> ``binding``
- ascore: coexpression                      -> ``expression``
- escore: experiments                       -> ``binding``
- dscore: database                          -> ``binding``
- tscore: textmining                        -> ``binding``

Activation / inhibition / catalysis / ptmod require ``protein.actions`` edges
(not present as separate per-channel download files in v12).
"""

from __future__ import annotations

from .ppi_relations import infer_grammar_relation

# (field_name, grammar_relation)
STRING_EVIDENCE_CHANNELS: tuple[tuple[str, str], ...] = (
    ("nscore", "binding"),
    ("fscore", "binding"),
    ("pscore", "binding"),
    ("ascore", "expression"),
    ("escore", "binding"),
    ("dscore", "binding"),
    ("tscore", "binding"),
)


def _parse_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def dominant_channel_from_scores(scores: dict[str, float]) -> tuple[str, str]:
    best_field = "pscore"
    best_score = -1.0
    for field, relation in STRING_EVIDENCE_CHANNELS:
        score = scores.get(field, 0.0)
        if score > best_score:
            best_score = score
            best_field = field
            best_relation = relation
    if best_score <= 0:
        return "physical", "binding"
    for field, relation in STRING_EVIDENCE_CHANNELS:
        if field == best_field:
            return field.replace("score", ""), relation
    return "physical", "binding"


def parse_string_detailed_link(line: str) -> tuple[str, str, str, str] | None:
    """Parse one STRING detailed link line -> (id_a, id_b, grammar_relation, channel)."""
    parts = line.strip().split()
    if len(parts) < 10:
        return None
    name1, name2 = parts[0], parts[1]
    scores = {
        "nscore": _parse_float(parts[3]),
        "fscore": _parse_float(parts[4]),
        "pscore": _parse_float(parts[5]),
        "ascore": _parse_float(parts[6]),
        "escore": _parse_float(parts[7]),
        "dscore": _parse_float(parts[8]),
        "tscore": _parse_float(parts[9]),
    }
    channel, relation = dominant_channel_from_scores(scores)
    grammar = infer_grammar_relation(
        source_id="stringdb_mint",
        task_family="ppi_pretraining",
        string_channel=channel,
        explicit_relation=relation,
    )
    return name1, name2, grammar, channel
