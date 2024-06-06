from typing import Any, Optional, Sequence, Union

from biocframe import BiocFrame
from summarizedexperiment import SummarizedExperiment

from .build_single_reference import build_single_reference
from .classify_single_reference import classify_single_reference


def _resolve_reference(ref_data, ref_labels, ref_features, build_args):
    if isinstance(ref_data, SummarizedExperiment) or issubclass(
        type(ref_data), SummarizedExperiment
    ):
        if ref_features is None:
            ref_features = ref_data.get_row_names()
        elif isinstance(ref_features, str):
            ref_features = ref_data.get_row_data().column(ref_features)

        ref_features = list(ref_features)

        if ref_labels is None:
            ref_labels = ref_data.get_column_names()
        elif isinstance(ref_labels, str):
            ref_labels = ref_data.get_column_data().column(ref_labels)

        ref_labels = list(ref_labels)

        try:
            _default_asy = "logcounts"
            if "assay_type" in build_args:
                _default_asy = build_args["assay_type"]

            ref_data = ref_data.assay(_default_asy)
        except Exception as _:
            raise ValueError(
                f"Reference dataset must contain log-normalized count ('{_default_asy}') assay."
            )

    if ref_labels is None:
        raise ValueError("'ref_labels' cannot be `None`.")

    if ref_features is None:
        raise ValueError("'ref_features' cannot be `None`.")

    return ref_data, ref_labels, ref_features


def annotate_single(
    test_data: Any,
    ref_data: Any,
    ref_labels: Optional[Union[Sequence, str]],
    test_features: Optional[Union[Sequence, str]] = None,
    ref_features: Optional[Union[Sequence, str]] = None,
    build_args: dict = {},
    classify_args: dict = {},
    num_threads: int = 1,
) -> BiocFrame:
    """Annotate a single-cell expression dataset based on the correlation
    of each cell to profiles in a labelled reference.

    Args:
        test_data:
            A matrix-like object representing the test dataset, where rows are
            features and columns are samples (usually cells). Entries should be expression
            values; only the ranking within each column will be used.

            Alternatively, a
            :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`
            containing such a matrix in one of its assays. Non-default assay
            types can be specified in ``classify_args``.

        test_features:
            Sequence of length equal to the number of rows in
            ``test_data``, containing the feature identifier for each row.

            If ``test_data`` is a ``SummarizedExperiment``, ``test_features``
            may be a string speciying the column name in `row_data`that contains the
            features. Alternatively can be set to `None`, to use the `row_names` of
            the experiment as used as features.

        ref_data:
            A matrix-like object representing the reference dataset, where rows
            are features and columns are samples. Entries should be expression values,
            usually log-transformed (see comments for the ``ref`` argument in
            :py:meth:`~singler.build_single_reference.build_single_reference`).

            Alternatively, a
            :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`
            containing such a matrix in one of its assays. Non-default assay
            types can be specified in ``classify_args``.

        ref_labels:
            If ``ref_data`` is a matrix-like object, ``ref_labels`` should be
            a sequence of length equal to the number of columns of ``ref_data``,
            containing the label associated with each column.

            If ``ref_data`` is a ``SummarizedExperiment``, ``ref_labels``
            may be a string specifying the label type to use,
            e.g., "main", "fine", "ont". Alternatively can be set to
            `None`, to use the `row_names` of the experiment as used as features.

        ref_features:
            If ``ref_data`` is a matrix-like object, ``ref_features`` should be
            a sequence of length equal to the number of rows of ``ref_data``,
            containing the feature identifier associated with each row.

            If ``ref_data`` is a ``SummarizedExperiment``, ``ref_features``
            may be a string speciying the column name in `column_data`
            that contains the features. Alternatively can be set to
            `None`, to use the `row_names` of the experiment as used as features.

        build_args:
            Further arguments to pass to
            :py:meth:`~singler.build_single_reference.build_single_reference`.

        classify_args:
            Further arguments to pass to
            :py:meth:`~singler.classify_single_reference.classify_single_reference`.

        num_threads:
            Number of threads to use for the various steps.

    Returns:
        A data frame containing the labelling results, see
        :py:meth:`~singler.classify_single_reference.classify_single_reference`
        for details. The metadata also contains a ``markers`` dictionary,
        specifying the markers that were used for each pairwise comparison
        between labels; and a list of ``unique_markers`` across all labels.
    """

    if isinstance(test_data, SummarizedExperiment):
        if test_features is None:
            test_features = test_data.get_row_names()
        elif isinstance(test_features, str):
            test_features = test_data.get_row_data().column(test_features)

    if test_features is None:
        raise ValueError("'test_features' cannot be `None`.")

    test_features_set = set(test_features)

    ref_data, ref_labels, ref_features = _resolve_reference(
        ref_data=ref_data,
        ref_labels=ref_labels,
        ref_features=ref_features,
        build_args=build_args,
    )

    built = build_single_reference(
        ref_data=ref_data,
        ref_labels=ref_labels,
        ref_features=ref_features,
        restrict_to=test_features_set,
        **build_args,
        num_threads=num_threads,
    )

    output = classify_single_reference(
        test_data,
        test_features=test_features,
        ref_prebuilt=built,
        **classify_args,
        num_threads=num_threads,
    )

    output.metadata = {
        "markers": built.markers,
        "unique_markers": built.marker_subset(),
    }
    return output
