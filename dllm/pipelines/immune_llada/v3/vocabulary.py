"""Vocabulary controls; run pytest scripts/tests/immune_llada/test_v3.py.

A0 retains the historical 126464-row decoder vocabulary and exact emitted token
IDs. Unused text rows stay in the softmax; no language-model weights are loaded.
"""
import json
from pathlib import Path

from ..data.grammar import GrammarTokenizer


class LegacyVocabulary:
    def __init__(self):
        self.contract=json.loads(Path(__file__).with_name("legacy_vocabulary.json").read_text())
        self.token_to_id=self.contract["tokens"]
        self.id_to_token={value:key for key,value in self.token_to_id.items()}
        self.vocab_size=self.contract["vocab_size"]
        self.pad_token_id=self.token_to_id["<pad>"]
        self.mask_token_id=self.token_to_id["<mask>"]
        self.eos_token_id=self.token_to_id["<eos>"]

    def special_id(self,token):
        return self.token_to_id[token]

    def encode_residues(self,sequence):
        return [self.token_to_id[aa] for aa in sequence]

    def token(self,index):
        return self.id_to_token.get(int(index),f"<unused:{int(index)}>")


def make_tokenizer(ablation):
    return LegacyVocabulary() if ablation=="A0" else GrammarTokenizer()
