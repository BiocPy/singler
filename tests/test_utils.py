from singler._utils import (
    _factorize,
    _stable_intersect,
    _stable_union,
    _clean_matrix,
)
import numpy as np
from mattress import tatamize
from summarizedexperiment import SummarizedExperiment


def test_factorize():
    lev, ind = _factorize([1, 3, 5, 5, 3, 1])
    assert list(lev) == ["1", "3", "5"]
    assert (ind == [0, 1, 2, 2, 1, 0]).all()

    # Preserves the order.
    lev, ind = _factorize(["C", "D", "A", "B", "C", "A"])
    assert list(lev) == ["C", "D", "A", "B"]
    assert (ind == [0, 1, 2, 3, 0, 2]).all()

    # Handles None-ness.
    lev, ind = _factorize([1, None, 5, None, 3, None])
    assert list(lev) == ["1", "5", "3"]
    assert (ind == [0, -1, 1, -1, 2, -1]).all()

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

    # Handles None-ness.
    out = _stable_intersect(
        ["B", None, "C", "A", None, "D", "E"], ["A", None, "C", "E", None, "F"]
    )
    assert out == ["C", "A", "E"]

    # Empty list.
    assert _stable_intersect() == []


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

    # Handles None-ness.
    out = _stable_union(
        ["B", None, "C", "A", None, "D", "E"], ["A", None, "C", "E", None, "F"]
    )
    assert out == ["B", "C", "A", "D", "E", "F"]

    # Empty list.
    assert _stable_union() == []


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
    tmp[0, 5] = np.nan
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
