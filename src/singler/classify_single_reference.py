from mattress import tatamize
from numpy import ndarray, int32, float64, uintp
from biocframe import BiocFrame

from .build_single_reference import SinglePrebuiltReference
from . import cpphelpers as lib


def classify_single_reference(
    test,
    ref: SinglePrebuiltReference,
    features: Sequence,
    quantile: float = 0.8,
    use_fine_tune: bool = true,
    fine_tune_threshold: float = 0.05,
    num_threads: int = 1,
):
    nl = ref.num_labels()
    mat_ptr = tatamize(test)
    nc = mat_ptr.ncol()

    best = ndarray((nc,), dtype = int32)
    delta = ndarray((nc,), dtype = float64)

    scores = {}
    all_labels = ref.labels
    score_ptrs = ndarray((nl,), dtype=uintp)
    for i in range(nl):
        current = ndarray((nc,), dtype = float64)
        scores[all_labels[i]] = current
        score_ptrs[i] = current.ctypes.ptr

    mapping = {}
    for i, x in enumerate(features):
        mapping[x] = i

    ref_subset = ref.subset()
    subset = ndarray((len(ref_subset),), dtype = int32)
    for i, x in enumerate(ref_subset):
        if x not in mapping:
            raise KeyError("failed to find gene '" + str(x) + "' in the test dataset")
        subset[i] = mapping[x]

    lib.classify_single_reference(
        mat_ptr.ptr, 
        subset,
        ref.ptr,
        quantile = quantile,
        use_fine_tune = use_fine_tune,
        fine_tune_threshold = fine_tune_threshold,
        nthreads = num_threads,
        scores.ctypes.data,
        best,
        delta
    )

    scores_df = BiocFrame(scores, number_of_rows = nc)
    return BiocFrame(best = best, scores = scores_df, delta = delta)
