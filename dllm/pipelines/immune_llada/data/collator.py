"""Import ``GrammarBioSeqCollator`` here to batch shared immune records.

The class remains canonical in ``grammar`` so existing patches to its renderer
and private encoder helpers keep working, without a circular compatibility import::

    from dllm.pipelines.immune_llada.data.collator import GrammarBioSeqCollator
    from dllm.pipelines.immune_llada.data.grammar import GrammarTokenizer
    collator = GrammarBioSeqCollator(GrammarTokenizer())
"""

from .grammar import GrammarBioSeqCollator

__all__ = ["GrammarBioSeqCollator"]
