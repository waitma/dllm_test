"""Shared immune data records and grammar-v2 batching.

Build an in-memory batch without loading datasets or model weights::

    from dllm.pipelines.immune_llada.data import (
        BioSeqChain, BioSeqRecord, GrammarBioSeqCollator, GrammarTokenizer,
    )
    record = BioSeqRecord([BioSeqChain("ACD", "peptide")], "generic", "example")
    batch = GrammarBioSeqCollator(GrammarTokenizer())([record])

ESM adapters and record helpers are also available via the ``esm_encoding`` and
``records`` submodules. Data source and trainer selection remain caller concerns.
"""

from . import dataset, esm_encoding, records
from .collator import GrammarBioSeqCollator
from .esm_encoding import Esm2SequenceTokenizer, HuggingFaceEsmTokenizerAdapter
from .grammar import (
    DEFAULT_GRAMMAR_DATA_DIR,
    GRAMMAR_NULL_CONTEXT_TOKEN,
    GRAMMAR_RELATIONS,
    GRAMMAR_TOKENS,
    TOKEN_CLASS_NAMES,
    GrammarArrowSource,
    GrammarArrowSourceConfig,
    GrammarRenderer,
    GrammarTokenizer,
)
from .dataset import PreparedImmuneDataset, load_prepared_dataset
from .records import BioSeqChain, BioSeqRecord

__all__ = [
    "BioSeqChain",
    "BioSeqRecord",
    "PreparedImmuneDataset",
    "DEFAULT_GRAMMAR_DATA_DIR",
    "Esm2SequenceTokenizer",
    "GRAMMAR_NULL_CONTEXT_TOKEN",
    "GRAMMAR_RELATIONS",
    "GRAMMAR_TOKENS",
    "GrammarArrowSource",
    "GrammarArrowSourceConfig",
    "GrammarBioSeqCollator",
    "GrammarRenderer",
    "GrammarTokenizer",
    "HuggingFaceEsmTokenizerAdapter",
    "TOKEN_CLASS_NAMES",
    "esm_encoding",
    "records",
    "dataset",
    "load_prepared_dataset",
]
