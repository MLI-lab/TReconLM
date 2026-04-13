import random
import pytest
from src.data_pkg.data_generation import (
    sample_ground_truth_length,
    nest_strings,
    unnest_strings,
    generate_ground_truth_sequence,
    data_generation,
    generate_test_data,
)
from src.data_pkg.IDS_channel import (
    generate_insertion_pattern,
    generate_alignment_pattern,
    generate_alignment,
    replace_I,
    IDS_channel,
)


# Ground truth length sampling

def test_sample_fixed_length():
    assert sample_ground_truth_length(110) == 110


def test_sample_interval_within_range():
    random.seed(42)
    for _ in range(50):
        length = sample_ground_truth_length([60, 100])
        assert 60 <= length <= 100


def test_sample_with_rng_reproducible():
    lengths_a = [sample_ground_truth_length([50, 200], rng=random.Random(0)) for _ in range(10)]
    lengths_b = [sample_ground_truth_length([50, 200], rng=random.Random(0)) for _ in range(10)]
    assert lengths_a == lengths_b
    assert all(50 <= l <= 200 for l in lengths_a)


# Insertion pattern generation

def test_insertion_pattern():
    assert generate_insertion_pattern(3) == ["I--", "-I-", "--I"]


# Alignment

def test_alignment_pattern_pads_to_equal_length():
    seqs = ["IACG", "ACGT", "IACG"]
    result = generate_alignment_pattern(seqs)
    assert len(result) == 3
    assert len(set(len(r) for r in result)) == 1


def test_generate_alignment_resolves_I_and_D():
    result = generate_alignment(["I-A", "DAC"], rng=random.Random(42))
    for seq in result:
        assert 'I' not in seq
        assert 'D' not in seq
        assert all(c in "ACGT-" for c in seq)
    assert result[1] == "-AC"  # D always becomes '-'


def test_replace_I():
    assert replace_I("IAC", "TAC") == "TAC"
    assert replace_I("ICI", "AGT") == "ACT"
    assert replace_I("ACG", "ACG") == "ACG"
    with pytest.raises(SyntaxError):
        replace_I("IA", "ACG")  # unequal length


# IDS channel

def test_ids_channel_no_noise():
    result = IDS_channel("ACGT", {'insertion_probability': 0.0, 'deletion_probability': 0.0, 'substitution_probability': 0.0}, random.Random(0))
    assert result == "ACGT"


def test_ids_channel_all_deletions():
    result = IDS_channel("ACGT", {'insertion_probability': 0.0, 'deletion_probability': 1.0, 'substitution_probability': 0.0}, random.Random(0))
    assert result == ""


def test_ids_channel_reproducible():
    stats = {'insertion_probability': 0.1, 'deletion_probability': 0.1, 'substitution_probability': 0.1}
    seq = "ACGTACGTACGT"
    r1 = IDS_channel(seq, stats, random.Random(123))
    r2 = IDS_channel(seq, stats, random.Random(123))
    assert r1 == r2


# Generate ground truth sequence

def test_generate_ground_truth_sequence():
    seq = generate_ground_truth_sequence(110, rng=random.Random(42))
    assert len(seq) == 110
    assert all(c in "ACGT" for c in seq)


# nest_strings / unnest_strings

def test_nest_strings():
    assert nest_strings("ABC|DEF|GHI") == "ADGBEHCFI"


def test_unnest_strings():
    assert unnest_strings("ADGBEHCFI", 3) == ["ABC", "DEF", "GHI"]


def test_nest_unnest_inverse():
    original = "ACGT|TGCA|GGCC"
    num_segments = len(original.split("|"))
    nested = nest_strings(original)
    recovered = unnest_strings(nested, num_segments)
    assert recovered == original.split("|")


# Train vs test data consistency

def test_train_and_test_pipelines_identical_output_cpred():
    """With the same ground truth and RNG state, both pipelines produce identical output."""
    stats = {'insertion_probability': 0.05, 'deletion_probability': 0.05, 'substitution_probability': 0.05}

    # data_generation generates GT internally, advancing the RNG.
    # To match, we replicate that: generate GT with same seed, then pass the
    # same RNG state (after GT generation) to generate_test_data.
    rng1 = random.Random(42)
    train_data = data_generation(
        data_set_size=1, observation_size=3, length_ground_truth=60,
        channel_statistics=stats, target_type='CPRED', data_type='ids_data', rng=rng1,
    )
    train_gt = train_data[0][1][0]
    train_example = train_data[1][1][0]

    # recreate the same RNG, generate the same GT (to advance RNG to same state)
    rng2 = random.Random(42)
    gt = generate_ground_truth_sequence(60, rng=rng2)
    assert gt == train_gt

    # now rng2 is at the same state as rng1 was after GT generation
    test_example = generate_test_data(
        ground_truth_sequence=gt, observation_size=3,
        channel_statistics=stats, target_type='CPRED', data_type='ids_data', rng=rng2,
    )
    assert test_example == train_example


def test_train_and_test_pipelines_identical_output_msa():
    """With the same ground truth and RNG state, both pipelines produce identical MSA output."""
    stats = {'insertion_probability': 0.05, 'deletion_probability': 0.05, 'substitution_probability': 0.05}

    rng1 = random.Random(42)
    train_data = data_generation(
        data_set_size=1, observation_size=3, length_ground_truth=60,
        channel_statistics=stats, target_type='std_MSA', data_type='ids_data', rng=rng1,
    )
    train_gt = train_data[0][1][0]
    train_example = train_data[1][1][0]

    rng2 = random.Random(42)
    gt = generate_ground_truth_sequence(60, rng=rng2)
    assert gt == train_gt

    test_example = generate_test_data(
        ground_truth_sequence=gt, observation_size=3,
        channel_statistics=stats, target_type='std_MSA', data_type='ids_data', rng=rng2,
    )
    assert test_example == train_example


def test_train_and_test_pipelines_identical_output_nested():
    """With the same ground truth and RNG state, both pipelines produce identical NESTED output."""
    stats = {'insertion_probability': 0.05, 'deletion_probability': 0.05, 'substitution_probability': 0.05}

    rng1 = random.Random(42)
    train_data = data_generation(
        data_set_size=1, observation_size=3, length_ground_truth=60,
        channel_statistics=stats, target_type='std_NESTED', data_type='ids_data', rng=rng1,
    )
    train_gt = train_data[0][1][0]
    train_example = train_data[1][1][0]

    rng2 = random.Random(42)
    gt = generate_ground_truth_sequence(60, rng=rng2)
    assert gt == train_gt

    test_example = generate_test_data(
        ground_truth_sequence=gt, observation_size=3,
        channel_statistics=stats, target_type='std_NESTED', data_type='ids_data', rng=rng2,
    )
    assert test_example == train_example
