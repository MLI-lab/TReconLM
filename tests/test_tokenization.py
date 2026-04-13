from src.data_pkg.tokenizer import encode, decode, encode_list, decode_list, pad_encoded_data


def test_encode_single_bases(stoi):
    assert encode("ACGT", stoi) == [0, 1, 2, 3]


def test_encode_special_tokens(stoi):
    assert encode("-|:#", stoi) == [4, 5, 6, 7]


def test_encode_decode_inverse(stoi, itos):
    seq = "ACGT-|:#"
    assert decode(encode(seq, stoi), itos) == seq


def test_encode_batched(stoi):
    result = encode_list(["AC", "GT"], stoi)
    assert result == [[0, 1], [2, 3]]


def test_decode_batched(itos):
    result = decode_list([[0, 1], [2, 3]], itos)
    assert result == ["AC", "GT"]


def test_encode_decode_batched_inverse(stoi, itos):
    seqs = ["ACGT", "T-A|"]
    assert decode_list(encode_list(seqs, stoi), itos) == seqs


def test_pad_encoded_data_shorter(stoi):
    data = [[0, 1]]  # "AC"
    result = pad_encoded_data(data, 5, stoi)
    assert result == [[0, 1, 7, 7, 7]]  # padded with '#' = 7


def test_pad_encoded_data_exact_length(stoi):
    data = [[0, 1, 2]]
    result = pad_encoded_data(data, 3, stoi)
    assert result == [[0, 1, 2]]


def test_pad_encoded_data_multiple(stoi):
    data = [[0], [0, 1]]
    result = pad_encoded_data(data, 3, stoi)
    assert result == [[0, 7, 7], [0, 1, 7]]
