"""Check oracle conflict/tie arithmetic: pytest -q scripts/tests/immune_llada/test_expanded_identifiability.py."""
import pytest

from scripts.audit_immune_expanded_identifiability import calculate, first_terminal_bound


def test_conflicting_equal_inputs_keep_joint_terminal_tie_range():
    entries = [dict(input_sha256='same', fixture_id='a', labels=[1, 9], selected={0, 1}, aa={0}, terminal={1}),
               dict(input_sha256='same', fixture_id='b', labels=[2, 3], selected={0, 1}, aa={0, 1}, terminal=set()),
               dict(input_sha256='different', fixture_id='c', labels=[1, 9], selected={0, 1}, aa={0}, terminal={1})]
    result = calculate(entries)
    assert result['selected']['maximum_correct'] == 4 and result['selected']['count'] == 6
    assert result['aa']['maximum_correct'] == 3 and result['aa']['count'] == 4
    assert result['terminal']['maximum_correct'] == 2 and result['terminal']['count'] == 2
    assert result['joint_optimal_terminal_correct_range'] == [1, 2]
    assert result['joint_optimal_terminal_accuracy_range'] == [.5, 1.]
    assert result['input_groups'] == [['a', 'b'], ['c']]


def test_no_terminal_positions_are_undefined_not_perfect_accuracy():
    result = calculate([dict(input_sha256='x', fixture_id='a', labels=[1], selected={0}, aa={0}, terminal=set())])
    assert result['selected']['accuracy_upper_bound'] == 1.
    assert result['terminal']['accuracy_upper_bound'] is None
    assert result['joint_optimal_terminal_accuracy_range'] == [None, None]


def test_first_terminal_same_input_three_lengths_has_one_third_upper_bound():
    entries = [dict(input_sha256='same', fixture_id=f'{length}-{repeat}',
                    chain_terminal_positions={0: length, 1: length-2})
               for length in (24, 127, 255) for repeat in range(4)]
    result = first_terminal_bound(entries)
    assert (result['maximum_correct'], result['count']) == (8, 24)
    assert result['accuracy_upper_bound'] == 1/3
    assert result['input_groups'][0]['terminal_position_counts'][1] == {22: 4, 125: 4, 253: 4}
    for entry in entries:
        entry['input_sha256'] = entry['fixture_id']
    assert first_terminal_bound(entries)['accuracy_upper_bound'] == 1.


def test_first_terminal_rejects_incompatible_chain_slots():
    entries = [dict(input_sha256='same', fixture_id='a', chain_terminal_positions={0: 2}),
               dict(input_sha256='same', fixture_id='b', chain_terminal_positions={1: 2})]
    with pytest.raises(ValueError, match='Identical full-mask'):
        first_terminal_bound(entries)
    assert first_terminal_bound([])['accuracy_upper_bound'] is None
