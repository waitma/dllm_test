"""Resolve HLA references conservatively, allowing exact coverage differences.

Activate pllm; python -B -m pytest scripts/tests/test_mhc_reference_resolution.py -q.
Use ``HLAReference.from_fasta(path).resolve('HLA-A*02:01')`` from build scripts.
An accepted result is a database reference imputation, not an observed sample sequence.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path
import re


STANDARD_AMINO_ACIDS = frozenset('ACDEFGHIKLMNPQRSTVWY')
HLA_GENES = frozenset(('A', 'B', 'C', 'E', 'F', 'G', 'HFE', 'MICA', 'MICB',
    'DMA', 'DMB', 'DOA', 'DOB', 'DPA1', 'DPB1', 'DQA1', 'DQA2', 'DQB1',
    'DQB2', 'DRA', 'DRB1', 'DRB2', 'DRB3', 'DRB4', 'DRB5', 'DRB6', 'DRB7',
    'DRB8', 'DRB9', 'TAP1', 'TAP2'))
EXPRESSION_SUFFIXES = frozenset('NLSCAQ')
RESOLUTION_POLICY = 'hla_unique_longest_exact_coverage_v1'


def sha256_text(value):
    return hashlib.sha256(value.encode('ascii')).hexdigest()


@dataclass(frozen=True)
class Allele:
    gene: str
    fields: tuple
    suffix: str = ''

    @property
    def canonical(self):
        return self.gene + '*' + ':'.join(self.fields) + self.suffix


def parse_allele(value):
    """Retain supplied resolution and expression suffix; never infer missing fields.

    Accept HLA-A*02:01, A02:01 and unambiguous four-digit A0201 notation.
    Longer compact fields are rejected because HLA fields can exceed two digits.
    """
    if not isinstance(value, str):
        raise ValueError('allele must be a string')
    text = value.strip().upper()
    if text.startswith('HLA-'):
        text = text[4:]
    if '*' in text:
        if text.count('*') != 1:
            raise ValueError('multiple allele separators')
        gene, field_text = text.split('*')
        if gene not in HLA_GENES:
            raise ValueError('unknown HLA gene')
    else:
        gene = next((g for g in sorted(HLA_GENES, key=len, reverse=True)
                     if text.startswith(g) and text[len(g):len(g) + 1].isdigit()), None)
        if gene is None:
            raise ValueError('unknown or missing HLA gene')
        field_text = text[len(gene):]
    suffix = field_text[-1:] if field_text[-1:] in EXPRESSION_SUFFIXES else ''
    if suffix:
        field_text = field_text[:-1]
    if ':' in field_text:
        fields = tuple(field_text.split(':'))
    elif len(field_text) == 4 and field_text.isdigit():
        fields = (field_text[:2], field_text[2:])
    elif re.fullmatch(r'\d{2,3}', field_text):
        fields = (field_text,)
    else:
        raise ValueError('invalid or ambiguous compact allele fields')
    if not 1 <= len(fields) <= 4 or any(not re.fullmatch(r'\d{2,}', f) for f in fields):
        raise ValueError('invalid or ambiguous allele fields')
    return Allele(gene, fields, suffix)


def normalize_protein(sequence):
    """Remove whitespace, uppercase and one terminal stop; reject unknown residues."""
    compact = ''.join(sequence.split())
    sequence = compact.upper()
    terminal_stop_removed = sequence.endswith('*')
    if terminal_stop_removed:
        sequence = sequence[:-1]
    invalid_symbols = sorted(set(sequence) - STANDARD_AMINO_ACIDS)
    return dict(sequence=sequence, valid=bool(sequence) and not invalid_symbols,
        invalid_symbols=invalid_symbols, terminal_stop_removed=terminal_stop_removed,
        lowercase_normalized=compact != compact.upper())


class HLAReference:
    """Index immutable FASTA evidence; resolve without choosing an arbitrary variant.

    Unsuffixed queries conservatively include all expression variants at their
    supplied field precision. An explicit suffix restricts to that suffix.
    Results should be stored once per allele in a release ledger, not duplicated
    in every training record. Treat returned cached dictionaries as read-only.
    """

    def __init__(self, entries, *, reference_path, reference_sha256):
        self.reference_path = str(reference_path)
        self.reference_sha256 = reference_sha256
        self.entries = entries
        self.index = defaultdict(list)
        for i, entry in enumerate(entries):
            allele = entry['parsed_allele']
            for n in range(1, len(allele.fields) + 1):
                self.index[(allele.gene, allele.fields[:n])].append(i)

    @classmethod
    def from_fasta(cls, path):
        path = Path(path).resolve()
        raw = path.read_bytes()
        entries, header, parts = [], None, []

        def add_entry():
            if header is None:
                return
            fields = header.split()
            if len(fields) < 2:
                raise ValueError('FASTA header must contain reference ID and HLA allele: ' + header)
            allele = parse_allele(fields[1])
            entries.append(dict(reference_id=fields[0], allele=allele.canonical,
                parsed_allele=allele, **normalize_protein(''.join(parts))))

        for line in raw.decode('utf-8').splitlines():
            if line.startswith('>'):
                add_entry()
                header, parts = line[1:], []
            elif line.strip():
                if header is None:
                    raise ValueError('sequence appears before first FASTA header')
                parts.append(line)
        add_entry()
        if not entries:
            raise ValueError('empty HLA reference FASTA')
        return cls(entries, reference_path=path,
                   reference_sha256=hashlib.sha256(raw).hexdigest())

    @lru_cache(maxsize=None)
    def resolve(self, allele_name):
        result = dict(policy=RESOLUTION_POLICY, supplied_allele=allele_name,
            normalized_allele=None, accepted=False, status='invalid_allele',
            sequence=None, sequence_sha256=None, entry_count=0,
            unique_sequence_count=0, reference_path=self.reference_path,
            reference_sha256=self.reference_sha256, source_entries=[], coverage=[])
        try:
            allele = parse_allele(allele_name)
        except ValueError as error:
            result['reason'] = str(error)
            return result
        result.update(normalized_allele=allele.canonical, gene=allele.gene,
                      allele_field_count=len(allele.fields), expression_suffix=allele.suffix)
        if len(allele.fields) < 2:
            result['status'] = 'insufficient_allele_resolution'
            return result
        entries = [self.entries[i] for i in self.index.get((allele.gene, allele.fields), ())
                   if not allele.suffix or self.entries[i]['parsed_allele'].suffix == allele.suffix]
        result['entry_count'] = len(entries)
        if not entries:
            result['status'] = 'reference_not_found'
            return result
        result['source_entries'] = [dict(reference_id=e['reference_id'], allele=e['allele'],
            length=len(e['sequence']), sequence_sha256=sha256_text(e['sequence']),
            terminal_stop_removed=e['terminal_stop_removed'],
            lowercase_normalized=e['lowercase_normalized'], invalid_symbols=e['invalid_symbols'])
            for e in entries]
        sequence_counts = Counter(e['sequence'] for e in entries)
        result['unique_sequence_count'] = len(sequence_counts)
        result['reference_lengths'] = dict(sorted(Counter(len(e['sequence']) for e in entries).items()))
        invalid_entries = [e for e in entries if not e['valid']]
        if invalid_entries:
            result.update(status='invalid_reference_sequence', invalid_entry_count=len(invalid_entries),
                invalid_symbols=sorted({s for e in invalid_entries for s in e['invalid_symbols']}))
            return result
        longest_length = max(map(len, sequence_counts))
        longest = sorted(s for s in sequence_counts if len(s) == longest_length)
        result.update(longest_length=longest_length, distinct_longest_sequences=len(longest))
        if len(longest) != 1:
            result['status'] = 'conflicting_longest_sequences'
            return result
        reference = longest[0]
        conflict_count = 0
        for sequence, count in sorted(sequence_counts.items(), key=lambda item: (-len(item[0]), item[0])):
            positions = []
            offset = reference.find(sequence)
            while offset != -1:
                positions.append(offset + 1)
                offset = reference.find(sequence, offset + 1)
            if not positions:
                conflict_count += count
            result['coverage'].append(dict(sequence_sha256=sha256_text(sequence), length=len(sequence),
                reference_entry_count=count, exact_substring=bool(positions),
                start_positions_1based=positions,
                end_positions_1based_inclusive=[p + len(sequence) - 1 for p in positions]))
        if conflict_count:
            result.update(status='reference_coverage_conflict', conflicting_entry_count=conflict_count)
            return result
        result.update(accepted=True, sequence=reference, sequence_sha256=sha256_text(reference),
            sequence_length=len(reference), status='resolved_identical' if len(sequence_counts) == 1
            else 'resolved_unique_longest_covering_all')
        result.update(sequence_scope='longest_available_reference_not_verified_full_length',
            starts_with_methionine=reference.startswith('M'),
            reference_n_terminus=reference[:10], reference_c_terminus=reference[-10:],
            representative_alleles=sorted({e['allele'] for e in entries if e['sequence'] == reference}),
            representative_reference_ids=sorted({e['reference_id'] for e in entries
                                                  if e['sequence'] == reference}))
        return result
