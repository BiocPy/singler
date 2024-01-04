from typing import Any, Sequence, Union

from biocframe import BiocFrame
from numpy import float64, int32, ndarray, uintp

from . import _cpphelpers as lib
from ._utils import _clean_matrix, _create_map
from .build_single_reference import SinglePrebuiltReference


def classify_single_reference(
    test_data: Any,
    test_features: Sequence,
    ref_prebuilt: SinglePrebuiltReference,
    assay_type: Union[str, int] = 0,
    check_missing: bool = True,
    quantile: float = 0.8,
    use_fine_tune: bool = True,
    fine_tune_threshold: float = 0.05,
    num_threads: int = 1,
) -> BiocFrame:
    """Classify a test dataset against a reference by assigning labels from the latter to each column of the former
    using the SingleR algorithm.

    Args:
        test_data:
            A matrix-like object where each row is a feature and each column
            is a test sample (usually a single cell), containing expression values.
            Normalized and transformed expression values are also acceptable as only
            the ranking is used within this function.

            Alternatively, a
            :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`
            containing such a matrix in one of its assays.

        test_features:
            Sequence of identifiers for each feature in the test
            dataset, i.e., row in ``test_data``.

        ref_prebuilt:
            A pre-built reference created with
            :py:meth:`~singler.build_single_reference.build_single_reference`.

        assay_type: 
            Assay containing the expression matrix,
            if `test_data` is a
            :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`.

        check_missing:
            Whether to check for and remove rows with missing (NaN) values
            from ``test_data``.

        quantile:
            Quantile of the correlation distribution for computing the score for each label.
            Larger values increase sensitivity of matches at the expense of
            similarity to the average behavior of each label.

        use_fine_tune:
            Whether fine-tuning should be performed. This improves accuracy for distinguishing
            between similar labels but requires more computational work.

        fine_tune_threshold:
            Maximum difference from the maximum correlation to use in fine-tuning.
            All labels above this threshold are used for another round of fine-tuning.

        num_threads:
            Number of threads to use during classification.

    Returns:
        A data frame containing the ``best`` label, the ``scores``
        for each label (as a nested BiocFrame), and the ``delta`` from the best
        to the second-best label.  Each row corresponds to a column of ``test``.
    """
    mat_ptr, test_features = _clean_matrix(
        test_data,
        test_features,
        assay_type=assay_type,
        check_missing=check_missing,
        num_threads=num_threads,
    )

    nl = ref_prebuilt.num_labels()
    nc = mat_ptr.ncol()

    best = ndarray((nc,), dtype=int32)
    delta = ndarray((nc,), dtype=float64)

    scores = {}
    all_labels = ref_prebuilt.labels
    score_ptrs = ndarray((nl,), dtype=uintp)
    for i in range(nl):
        current = ndarray((nc,), dtype=float64)
        scores[all_labels[i]] = current
        score_ptrs[i] = current.ctypes.data

    mapping = _create_map(test_features)

    ref_subset = ref_prebuilt.marker_subset(indices_only=True)
    ref_features = ref_prebuilt.features
    subset = ndarray((len(ref_subset),), dtype=int32)
    for i, y in enumerate(ref_subset):
        x = ref_features[y]
        if x not in mapping:
            raise KeyError("failed to find gene '" + str(x) + "' in the test dataset")
        subset[i] = mapping[x]

    lib.classify_single_reference(
        mat_ptr.ptr,
        subset,
        ref_prebuilt._ptr,
        quantile=quantile,
        use_fine_tune=use_fine_tune,
        fine_tune_threshold=fine_tune_threshold,
        nthreads=num_threads,
        scores=score_ptrs.ctypes.data,
        best=best,
        delta=delta,
    )

    scores_df = BiocFrame(scores, number_of_rows=nc)
    return BiocFrame(
        {"best": [all_labels[b] for b in best], "scores": scores_df, "delta": delta}
    )
