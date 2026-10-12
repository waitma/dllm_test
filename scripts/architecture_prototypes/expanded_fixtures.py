"""Deterministic artificial sequence fixtures; no biological dataset is read.

After activating pllm:
python -c "from scripts.architecture_prototypes.expanded_fixtures import write_fixture_manifest; write_fixture_manifest('/tmp/expanded_fixture', train_size=576, valid_size=72)"
Only synthetic_fixture_manifest.json is written. Loading regenerates every record
and verifies the complete manifest; external sequence files are never loaded.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import uuid

from dllm.pipelines.immune_llada.data.records import BioSeqChain, BioSeqRecord


SCHEMA = 'immune.synthetic_fixture.v1'
KIND = 'synthetic_functional_diagnostic'
MANIFEST_NAME = 'synthetic_fixture_manifest.json'
ALPHABET = 'AGSTNQDE'
GROUPS = ('ab_antigen', 'tcr_peptide', 'tcr_pmhc',
          'unconditional_ab', 'unconditional_tcr', 'unconditional_single')
LENGTH_BINS = ('short', 'medium', 'long')
TARGETS = {
    'ab_antigen': ('antibody_heavy', 'antibody_light'),
    'tcr_peptide': ('tcr_alpha', 'tcr_beta'),
    'tcr_pmhc': ('tcr_alpha', 'tcr_beta'),
    'unconditional_ab': ('antibody_heavy', 'antibody_light'),
    'unconditional_tcr': ('tcr_alpha', 'tcr_beta'),
    'unconditional_single': ('nanobody_vhh',),
}
CONDITIONS = {'ab_antigen': ('antigen',), 'tcr_peptide': ('peptide',),
              'tcr_pmhc': ('peptide', 'mhc')}
TASKS = {'ab_antigen': 'antibody_antigen', 'tcr_peptide': 'tcr_epitope',
         'tcr_pmhc': 'tcr_pmhc', 'unconditional_ab': 'antibody',
         'unconditional_tcr': 'tcr', 'unconditional_single': 'generic'}
ROLE_MARKERS = {'antibody_heavy': 'AA', 'antibody_light': 'GG',
                'tcr_alpha': 'SS', 'tcr_beta': 'TT', 'nanobody_vhh': 'NN'}
CONFIG_KEYS = {'seed', 'train_size', 'valid_size', 'full_capacity', 'fragment_capacity', 'max_length'}


@dataclass(frozen=True)
class FixtureSplit:
    train_records: tuple[BioSeqRecord, ...]
    valid_records: tuple[BioSeqRecord, ...]
    manifest: dict


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _tile(pattern, length):
    return (pattern*((length+len(pattern)-1)//len(pattern)))[:length]


def _code(number):
    # Six base-eight digits give an injective visible signature for every record.
    return ''.join(ALPHABET[(number//(8**power)) % 8] for power in range(5, -1, -1))


def _target(sequence):
    return hashlib.sha256(sequence.encode('ascii')).hexdigest()


def _summarize(records, full_capacity):
    group_counts = Counter(r.metadata['diagnostic_group'] for r in records)
    cells = Counter(f"{r.metadata['diagnostic_group']}/{r.metadata['length_bin']}" for r in records)
    targets = [c for r in records for c in r.chains if c.role in r.metadata['target_roles']]
    lengths = [len(c.sequence) for c in targets]
    return dict(records=len(records), target_chains=len(targets),
        record_sha256=[_digest(r.to_dict()) for r in records],
        target_sequence_sha256=sorted(_target(c.sequence) for c in targets),
        group_counts={g: group_counts[g] for g in GROUPS},
        group_length_counts={f'{g}/{b}': cells[f'{g}/{b}'] for g in GROUPS for b in LENGTH_BINS},
        target_length_min=min(lengths), target_length_max=max(lengths),
        targets_with_residue_distance_gt32=sum(length > 33 for length in lengths),
        all_group_length_cells_present=all(cells[f'{g}/{b}'] for g in GROUPS for b in LENGTH_BINS),
        max_canvas_input_length=max(2+sum(1+full_capacity if c.role in r.metadata['target_roles']
            else len(c.sequence)+2 for c in r.chains) for r in records))


def build_fixture_split(*, seed=42, train_size=128, valid_size=24, full_capacity=256,
                        fragment_capacity=64, max_length=2048):
    """Return disjoint artificial train/heldout records with fixed six-by-three strata.

    Each conditional target copies a six-character context code and repeats an
    eight-character code/role motif. In pMHC fixtures the first and last three
    code characters come from peptide and MHC respectively. Context also exposes
    the length-bin marker. There is no hidden conditional target nonce.
    Unconditional targets intentionally have no context-visible identity;
    full-mask exact recovery is therefore not an attainable universal criterion.
    """
    config = dict(seed=seed, train_size=train_size, valid_size=valid_size,
                  full_capacity=full_capacity, fragment_capacity=fragment_capacity, max_length=max_length)
    if any(type(value) is not int for value in config.values()):
        raise ValueError('Fixture generation arguments must be integers')
    if seed < 0 or min(train_size, valid_size) < 1 or train_size+valid_size > 8192:
        raise ValueError('Require nonnegative seed, two nonempty splits, and at most 8192 records')
    if not 16 <= full_capacity <= 1024 or not 2 <= fragment_capacity <= 1024:
        raise ValueError('Full capacity must be 16..1024 and fragment capacity 2..1024')
    # The largest topology has two targets and two contexts, each context >=8 AA.
    if max_length < 2+2*(full_capacity+1)+2*(8+2):
        raise ValueError('Sequence budget cannot hold the required two-target/two-context topology')
    maximum = full_capacity-1  # Keep one supervised <protd> slot after genuine C...F/W.
    short = min(24, max(12, maximum//4))
    medium = min(maximum-1, max(short+1, maximum//2))
    lengths = dict(zip(LENGTH_BINS, (short, medium, maximum)))
    rng = random.Random(seed)
    multiplier, offset = rng.randrange(1, 8**6, 2), rng.randrange(8**6)

    def make_records(split, count, start):
        records = []
        for index in range(count):
            group, length_bin = GROUPS[index % len(GROUPS)], LENGTH_BINS[(index//len(GROUPS)) % 3]
            code = _code((multiplier*(start+index)+offset) % (8**6))
            target_roles, condition_roles = TARGETS[group], CONDITIONS.get(group, ())
            marker = ALPHABET[LENGTH_BINS.index(length_bin)]
            budget = max_length-2-len(target_roles)*(full_capacity+1)
            condition_length = min({'short': 12, 'medium': 24, 'long': 48}[length_bin],
                                   budget//len(condition_roles)-2) if condition_roles else 0
            chains = []
            for role in condition_roles:
                visible = code if group != 'tcr_pmhc' else (code[:3] if role == 'peptide' else code[3:])
                chains.append(BioSeqChain(_tile(visible+marker, condition_length), role,
                                         metadata={'synthetic_only': True}))
            for role in target_roles:
                terminal = 'W' if role in ('antibody_heavy', 'nanobody_vhh') else 'F'
                sequence = 'C'+_tile(code+ROLE_MARKERS[role], lengths[length_bin]-2)+terminal
                chains.append(BioSeqChain(sequence, role, metadata={'synthetic_only': True}))
            records.append(BioSeqRecord(chains, TASKS[group], 'synthetic_expanded_v1', split=split,
                metadata=dict(synthetic_only=True, diagnostic_group=group, length_bin=length_bin,
                    fixture_id=f'{split}-{index:06d}', target_roles=list(target_roles),
                    condition_roles=list(condition_roles), target_scope='full',
                    conditional_signature_visible=bool(condition_roles),
                    unconditional_full_mask_identifiable=False if not condition_roles else None)))
        return tuple(records)

    train, valid = make_records('train', train_size, 0), make_records('valid', valid_size, train_size)
    summaries = {'train': _summarize(train, full_capacity), 'valid': _summarize(valid, full_capacity)}
    train_targets, valid_targets = (set(summaries[split]['target_sequence_sha256']) for split in ('train', 'valid'))
    if train_targets & valid_targets:
        raise RuntimeError('Artificial train/heldout target sequence overlap')
    manifest = dict(schema=SCHEMA, kind=KIND, biological_quality_claim=False,
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), parameters=config,
        source='procedural_artificial_strings_only', groups=list(GROUPS), length_bins=list(LENGTH_BINS),
        metadata_fields=dict(group='diagnostic_group', length_bin='length_bin'),
        target_scope='full', target_lengths=lengths, splits=summaries,
        exact_target_sequence_overlap=len(train_targets & valid_targets),
        exact_record_overlap=len(set(summaries['train']['record_sha256']) & set(summaries['valid']['record_sha256'])),
        cross_radius32_coverage=dict(possible=maximum > 33,
            train=summaries['train']['targets_with_residue_distance_gt32'] > 0,
            valid=summaries['valid']['targets_with_residue_distance_gt32'] > 0),
        learning_rules=dict(local='Repeat the visible six-letter code and role marker with period eight.',
            conditional='Copy code from context; pMHC concatenates peptide and MHC three-letter codes.',
            length='Context contains A/G/S for short/medium/long; target terminates after the configured length.'),
        limitations=['No biological corpus, downstream data, binding truth or biological quality claim.',
            'Heldout exact target sequences are disjoint; shared artificial motifs are intentional.',
            'Unconditional full-mask inputs hide the record code and true target length; perfect recovery is not identifiable.',
            'All target observations use full scope; fragment_capacity is recorded for collator compatibility.',
            'Cross-radius32 coverage exists only when capacity and generated strata permit residue distance greater than 32.'],
        content_sha256=_digest({'train': [r.to_dict() for r in train], 'valid': [r.to_dict() for r in valid]}))
    manifest['manifest_sha256'] = _digest(manifest)
    return FixtureSplit(train, valid, manifest)


def write_fixture_manifest(directory, **kwargs):
    """Write an immutable manifest and return generated records; never write sequences."""
    fixture = build_fixture_split(**kwargs)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory/MANIFEST_NAME
    if destination.exists():
        existing = load_verified_fixture_manifest(directory)
        if existing.manifest != fixture.manifest:
            raise ValueError('Existing fixture differs; use a new directory')
        return existing
    temporary = directory/f'.{MANIFEST_NAME}.{uuid.uuid4().hex}.tmp'
    try:
        temporary.write_text(json.dumps(fixture.manifest, indent=2, allow_nan=False)+'\n')
        # Atomic publication without replacing a manifest another writer created.
        try:
            os.link(temporary, destination)
        except FileExistsError:
            existing = load_verified_fixture_manifest(directory)
            if existing.manifest != fixture.manifest:
                raise ValueError('Concurrent fixture writer used different parameters')
    finally:
        temporary.unlink(missing_ok=True)
    return fixture


def load_verified_fixture_manifest(directory):
    """Regenerate trusted fixtures and compare every saved field/hash; ignore no fields."""
    path = Path(directory)/MANIFEST_NAME
    try:
        manifest = json.loads(path.read_text())
        if (not isinstance(manifest, dict) or manifest.get('schema') != SCHEMA
                or manifest.get('kind') != KIND or set(manifest.get('parameters', {})) != CONFIG_KEYS):
            raise ValueError('Unknown synthetic fixture schema/kind/parameters')
        expected = build_fixture_split(**manifest['parameters'])
        if _canonical(manifest) != _canonical(expected.manifest):
            raise ValueError('Synthetic manifest differs from regenerated records/protocol/source hashes')
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f'Invalid synthetic fixture manifest {path}: {exc}') from exc
    return expected
