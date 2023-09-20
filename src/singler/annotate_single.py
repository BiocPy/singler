from typing import Union, Sequence, Optional, Any
from biocframe import BiocFrame
from copy import copy

from .fetch_reference import fetch_github_reference, realize_github_markers
from .build_single_reference import build_single_reference
from .classify_single_reference import classify_single_reference
from ._utils import _clean_matrix


def _resolve_reference(ref_data, ref_labels, ref_features, cache_dir, build_args, test_features_set):
    if isinstance(ref_data, str):
        ref = fetch_github_reference(ref_data, cache_dir=cache_dir)
        ref_features = ref.row_data.column(ref_features)

        num_de = None
        if "marker_args" in build_args:
            marker_args = build_args["marker_args"]
            if "num_de" in marker_args:
                num_de = marker_args["num_de"]

        ref_markers = realize_github_markers(
            ref.metadata[ref_labels],
            ref_features,
            num_markers=num_de,
            restrict_to=test_features_set,
        )

        ref_data = ref.assay("ranks")
        ref_labels=ref.col_data.column(ref_labels)
    else:
        ref_markers = None

    return ref_data, ref_labels, ref_features, ref_markers


def _attach_markers(markers, build_args):
    if markers is not None and "markers" not in build_args:
        tmp = copy(build_args)
        tmp["markers"] = markers
        print(tmp)
        return tmp
    return build_args


def annotate_single(
    test_data: Any,
    test_features: Sequence,
    ref_data: Any,
    ref_labels: Union[Sequence, str],
    ref_features: Union[Sequence, str],
    cache_dir: Optional[str] = None,
    build_args: dict = {},
    classify_args: dict = {},
    num_threads: int = 1,
) -> BiocFrame:
    """Annotate a single-cell expression dataset based on the correlation 
    of each cell to profiles in a labelled reference.

    Args:
        test_data: A matrix-like object representing the test dataset, where rows are
            features and columns are samples (usually cells). Entries should be expression
            values; only the ranking within each column will be used.

            Alternatively, a
            :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`
            containing such a matrix in one of its assays. Non-default assay
            types can be specified in ``classify_args``.

        test_features: Sequence of length equal to the number of rows in
            ``test_data``, containing the feature identifier for each row.

        ref_data: A matrix-like object representing the reference dataset, where rows
            are features and columns are samples. Entries should be expression values,
            usually log-transformed (see comments for the ``ref`` argument in
            :py:meth:`~singler.build_single_reference.build_single_reference`).

            Alternatively, a string that can be passed as ``name`` to
            :py:meth:`~singler.fetch_reference.fetch_github_reference`.
            This will use the specified dataset as the reference.

        ref_labels:
            If ``ref_data`` is a matrix-like object, ``ref_labels`` should be
            a sequence of length equal to the number of columns of ``ref_data``,
            containing the label associated with each column.

            If ``ref_data`` is a string, ``ref_labels`` should be a string
            specifying the label type to use, e.g., "main", "fine", "ont".

        ref_features:
            If ``ref_data`` is a matrix-like object, ``ref_features`` should be
            a sequence of length equal to the number of rows of ``ref_data``,
            containing the feature identifier associated with each row.

            If ``ref_data`` is a string, ``ref_features`` should be a string
            specifying the label type to use, e.g., "ensembl", "symbol".

        cache_dir:
            Path to a cache directory for downloading reference files, see
            :py:meth:`~singler.fetch_reference.fetch_github_reference` for details.
            Only used if ``ref_data`` is a string.

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
    test_features_set = set(test_features)

    ref_data, ref_labels, ref_features, markers = _resolve_reference(
        ref_data=ref_data,
        ref_labels=ref_labels,
        ref_features=ref_features,
        cache_dir=cache_dir,
        build_args=build_args,
        test_features_set=test_features_set,
    )

    bargs = _attach_markers(markers, build_args)
    built = build_single_reference(
        ref_data=ref_data,
        ref_labels=ref_labels,
        ref_features=ref_features,
        restrict_to=test_features_set,
        **bargs,
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
