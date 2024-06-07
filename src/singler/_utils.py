from typing import Sequence, Tuple

import biocutils as ut
import numpy as np
from delayedarray import DelayedArray
from mattress import TatamiNumericPointer, tatamize
from summarizedexperiment import SummarizedExperiment


def _factorize(x: Sequence) -> Tuple[list, np.ndarray]:
    _factor = ut.Factor.from_sequence(x, sort_levels=False)
    return _factor.levels, np.array(_factor.codes, np.int32)


def _create_map(x: Sequence) -> dict:
    mapping = {}
    for i, val in enumerate(x):
        if val is not None:
            # Again, favor the first occurrence.
            if val not in mapping:
                mapping[val] = i
    return mapping


def _stable_intersect(*args) -> list:
    nargs = len(args)
    if nargs == 0:
        return []

    occurrences = {}
    for f in args[0]:
        if f is not None and f not in occurrences:
            occurrences[f] = [1, 0]

    for i in range(1, len(args)):
        for f in args[i]:
            if f is not None and f in occurrences:
                state = occurrences[f]
                if state[1] < i:
                    state[0] += 1
                    state[1] = i

    output = []
    for f in args[0]:
        if f is not None and f in occurrences:
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
            if f is not None and f not in present:
                output.append(f)
                present.add(f)

    return output


def _clean_matrix(x, features, assay_type, check_missing, num_threads):
    if isinstance(x, TatamiNumericPointer):
        # Assume the pointer was previously generated from _clean_matrix,
        # so it's 2-dimensional, matches up with features and it's already
        # clean of NaNs... so we no-op and just return it directly.
        return x, features

    if isinstance(x, SummarizedExperiment):
        if features is None:
            features = x.get_row_names()
        elif isinstance(features, str):
            features = x.get_row_data().column(features)
        features = list(features)

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


def _restrict_features(ptr, features, restrict_to):
    if restrict_to is not None:
        keep = []
        new_features = []
        for i, x in enumerate(features):
            if x in restrict_to:
                keep.append(i)
                new_features.append(x)
        features = new_features
        ptr = tatamize(DelayedArray(ptr)[keep, :])
    return ptr, features
