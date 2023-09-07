from mattress import tatamize
from numpy import ndarray, int32, float64, uintp
from biocframe import BiocFrame

from .build_single_reference import SinglePrebuiltReference
from . import cpphelpers as lib


def classify_single_reference(
    test,
    features: Sequence,
    ref: SinglePrebuiltReference,
    quantile: float = 0.8,
    use_fine_tune: bool = True,
    fine_tune_threshold: float = 0.05,
    num_threads: int = 1,
):
    """
    Classify a test dataset against a reference by assigning labels
    from the latter to each column of the former using the SingleR algorithm.

    Args:
        test: A matrix-like object where each row is a feature and each column
            is a test sample (usually a single cell), containing expression values.
            Normalized and transformed expression values are also acceptable as only
            the ranking is used within this function.

        features (Sequence): Sequence of identifiers for each feature in the test
            dataset, i.e., row in ``test``.

        ref (SinglePrebuiltReference):
            A pre-built reference created with
            :py:meth:`~singler.build_single_reference.build_single_reference`.

        quantile (float):
            Quantile of the correlation distribution for computing the score for each label.
            Larger values increase sensitivity of matches at the expense of
            similarity to the average behavior of each label.

        use_fine_tune (bool):
            Whether fine-tuning should be performed. This improves accuracy for distinguishing
            between similar labels but requires more computational work.

        fine_tune_threshold (float):
            Maximum difference from the maximum correlation to use in fine-tuning.
            All labels above this threshold are used for another round of fine-tuning.

        num_threads (int):
            Number of threads to use during classification.
    """

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

    ref_subset = ref.markers(indices_only = True)
    ref_features = ref.features
    subset = ndarray((len(ref_subset),), dtype = int32)
    for i, y in enumerate(ref_subset):
        x = ref_features[y]
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