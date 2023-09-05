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


def _clean_matrix(x, features, assay_type, check_missing):
    if isinstance(x, SummarizedExperiment):
        x = x.assay(assay_type)

#    ptr = tatamize(x)
#    retain = ndarray(ptr.nrow(), dtype=uint8)
#    if lib.prune_missing(ptr.ptr, retain):
#        retain = retain.astype(bool_)
#        x = x[retain,:]
#        features = features[retain]

    return x, features
