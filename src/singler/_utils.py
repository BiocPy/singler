from numpy import zeros, int32, ndarray
from typing import Sequence, Tuple
from summarizedexperiment import SummarizedExperiment
from mattress import tatamize, TatamiNumericPointer
from delayedarray import DelayedArray


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


def _match(x: Sequence, levels: Sequence) -> ndarray:
    mapping = {}
    for i, lev in enumerate(levels):
        # favor the first occurrence of a duplicate level.
        if lev not in mapping:
            mapping[lev] = i

    indices = zeros((len(x),), dtype=int32)
    for i, y in enumerate(x):
        indices[i] = mapping[y]

    return indices


def _stable_intersect(*args) -> list:
    nargs = len(args)
    if nargs == 0:
        return []

    occurrences = {}
    for f in args[0]:
        if f not in occurrences:
            occurrences[f] = [1, 0]

    for i in range(1, len(args)):
        for f in args[i]:
            if f in occurrences:
                state = occurrences[f]
                if state[1] < i:
                    state[0] += 1
                    state[1] = i

    output = []
    for f in args[0]:
        if f in occurrences:
            state = occurrences[f]
            if state[0] == nargs and state[1] >= 0:
                output.append(f)
                state[1] = -1  # avoid duplicates

    return output


def _stable_union(*args) -> list:
    if len(args) == 0:
        return []

    output = []
    present = set()
    for a in args:
        for f in a:
            if f not in present:
                output.append(f)
                present.add(f)

    return output


def _clean_matrix(x, features, assay_type, check_missing, num_threads):
    if isinstance(x, TatamiNumericPointer):
        # Assume it's already clean of NaNs.
        return x, features

    if isinstance(x, SummarizedExperiment):
        x = x.assay(assay_type)

    curshape = x.shape
    if len(curshape) != 2:
        raise ValueError("each entry of 'ref' should be a 2-dimensional array")
    if curshape[0] != len(features):
        raise ValueError(
            "number of rows of 'x' should be equal to the length of 'features'"
        )

    ptr = tatamize(x)
    if not check_missing:
        return ptr, features

    retain = ptr.row_nan_counts(num_threads=num_threads) == 0
    if retain.all():
        return ptr, features

    new_features = []
    for i, k in enumerate(retain):
        if k:
            new_features.append(features[i])

    sub = DelayedArray(ptr)[retain, :]  # avoid re-tatamizing 'x'.
    return tatamize(sub), new_features
