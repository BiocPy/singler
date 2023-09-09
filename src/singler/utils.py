from numpy import zeros, int32, ndarray
from typing import Sequence, Tuple
from summarizedexperiment import SummarizedExperiment
from mattress import tatamize, TatamiNumericPointer


def _factorize(x: Sequence) -> Tuple[Sequence, ndarray]:
    levels = []
    mapping = {}
    indices = zeros((len(x),), dtype=int32)

    for i, lev in enumerate(x):
        if lev not in mapping:
            mapping[lev] = len(levels)
            levels.append(lev)
        indices[i] = mapping[lev]

    return levels, indices


def _match(x: Sequence, target: Sequence) -> ndarray:
    mapping = {}
    for i, lev in enumerate(target):
        mapping[lev] = i

    indices = zeros((len(x),), dtype=int32)
    for i, y in enumerate(x):
        indices[i] = mapping[y]

    return indices


def _clean_matrix(x, features, assay_type, check_missing, num_threads):
    if isinstance(x, TatamiNumericPointer):
        # Assume it's already clean of NaNs.
        return x, features

    if isinstance(x, SummarizedExperiment):
        x = x.assay(assay_type)

    ptr = tatamize(x)
    if not check_missing:
        return ptr, features

    retain = ptr.row_nan_counts(num_threads=num_threads) == 0
    if retain.all():
        return ptr, features

    new_features = []
    for i, k in enumerate(retain):
        new_features.append(features[i])
    return tatamize(x[retain, :]), new_features
