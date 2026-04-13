import pytest


@pytest.fixture
def stoi():
    return {'A': 0, 'C': 1, 'G': 2, 'T': 3, '-': 4, '|': 5, ':': 6, '#': 7}


@pytest.fixture
def itos():
    return {0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: '-', 5: '|', 6: ':', 7: '#'}


@pytest.fixture
def channel_statistics():
    return {
        'insertion_probability': 0.05,
        'deletion_probability': 0.05,
        'substitution_probability': 0.05,
    }
