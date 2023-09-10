from singler._utils import (
    _factorize,
    _match,
    _stable_intersect,
    _stable_union,
    _clean_matrix,
)
import numpy as np
from mattress import tatamize
from summarizedexperiment import SummarizedExperiment


def test_factorize():
    lev, ind = _factorize([1, 3, 5, 5, 3, 1])
    assert lev == [1, 3, 5]
    assert list(ind) == [0, 1, 2, 2, 1, 0]

    # Preserves the order.
    lev, ind = _factorize(["C", "D", "A", "B", "C", "A"])
    assert lev == ["C", "D", "A", "B"]
    assert list(ind) == [0, 1, 2, 3, 0, 2]


def test_match():
    mm = _match(["A", "C", "B", "D", "A", "A", "C", "D", "B"], ["D", "C", "B", "A"])
    assert list(mm) == [3, 1, 2, 0, 3, 3, 1, 0, 2]

    # Handles duplicate targets.
    x = [5, 1, 2, 3, 5, 6, 7, 7, 2, 1]
    mm = _match(x, [1, 2, 3, 3, 5, 6, 1, 7, 6])
    assert list(mm) == [4, 0, 1, 2, 4, 5, 7, 7, 1, 0]


def test_intersect():
    # Preserves the order in the first argument.
    out = _stable_intersect(["B", "C", "A", "D", "E"], ["A", "C", "E"])
    assert out == ["C", "A", "E"]

    # Works with more than 2 entries.
    out = _stable_intersect(["B", "C", "A", "D", "E"], ["A", "C", "E"], ["E", "A"])
    assert out == ["A", "E"]

    # Handles duplicates gracefully.
    out = _stable_intersect(
        ["B", "B", "C", "A", "D", "D", "E"], ["A", "A", "C", "E", "F", "F"]
    )
    assert out == ["C", "A", "E"]


def test_union():
    out = _stable_union(
        ["B", "C", "A", "D", "E"],
        [
            "A",
            "C",
            "E",
            "F",
        ],
    )
    assert out == ["B", "C", "A", "D", "E", "F"]

    # Works with more than 2 entries.
    out = _stable_union(["B", "C", "A", "D", "E"], ["A", "C", "K", "E"], ["G", "K"])
    assert out == ["B", "C", "A", "D", "E", "K", "G"]

    # Handles duplicates gracefully.
    out = _stable_union(
        ["B", "B", "C", "A", "D", "D", "E"], ["F", "A", "A", "C", "E", "F"]
    )
    assert out == ["B", "C", "A", "D", "E", "F"]


def test_clean_matrix():
    out = np.random.rand(20, 10)
    features = ["FEATURE_" + str(i) for i in range(out.shape[0])]

    ptr, feats = _clean_matrix(
        out, features, assay_type=None, check_missing=True, num_threads=1
    )
    assert feats == features
    assert (ptr.row(1) == out[1, :]).all()
    assert (ptr.column(2) == out[:, 2]).all()

    ptr, feats = _clean_matrix(
        out, features, assay_type=None, check_missing=False, num_threads=1
    )
    assert feats == features
    assert (ptr.row(3) == out[3, :]).all()
    assert (ptr.column(4) == out[:, 4]).all()

    tmp = np.copy(out)
    tmp[0, 5] = np.NaN
    ptr, feats = _clean_matrix(
        tmp, features, assay_type=None, check_missing=True, num_threads=1
    )
    assert feats == features[1:]
    assert (ptr.row(2) == out[3, :]).all()
    assert (ptr.column(4) == out[1:, 4]).all()

    ptr = tatamize(out)
    ptr2, feats = _clean_matrix(
        ptr, features, assay_type=None, check_missing=True, num_threads=1
    )
    assert ptr2.ptr == ptr.ptr

    se = SummarizedExperiment({"counts": out})
    ptr, feats = _clean_matrix(
        se, features, assay_type="counts", check_missing=True, num_threads=1
    )
    assert feats == features
    assert (ptr.row(1) == out[1, :]).all()
    assert (ptr.column(2) == out[:, 2]).all()
