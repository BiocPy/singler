from numpy import zeros, int32, ndarray
from typing import Sequence, Tuple
from summarizedexperiment import SummarizedExperiment
from . import cpphelpers as lib


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


def _clean_matrix(x, features, assay_type, check_missing, num_threads):
    if isinstance(x, SummarizedExperiment):
        x = x.assay(assay_type)

    ptr = tatamize(x)
    retain = ptr.row_nan_counts(num_threads = num_threads) == 0
    if retain.all():
        return x, features

    new_features = []
    for i, k in enumerate(retain):
        new_features.append(features[i])
    return x[retain,:], new_features
