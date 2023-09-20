from typing import Union, Sequence, Optional, Any, Tuple
from biocframe import BiocFrame

from .fetch_reference import fetch_github_reference, realize_github_markers
from .build_single_reference import build_single_reference
from .classify_single_reference import classify_single_reference
from .build_integrated_references import build_integrated_references
from .classify_integrated_references import classify_integrated_references
from .annotate_single import _resolve_reference, _attach_markers
from ._utils import _clean_matrix


def annotate_integrated(
    test_data: Any,
    test_features: Sequence,
    ref_data_list: Sequence[Union[Any, str]],
    ref_labels_list: Union[str, Sequence[Union[Sequence, str]]],
    ref_features_list: Union[str, Sequence[Union[Sequence, str]]],
    test_assay_type: Union[str, int] = 0,
    test_check_missing: bool = True,
    ref_assay_type: Union[str, int] = "logcounts",
    ref_check_missing: bool = True,
    cache_dir: Optional[str] = None,
    build_single_args: dict = {},
    classify_single_args: dict = {},
    build_integrated_args: dict = {},
    classify_integrated_args: dict = {},
    num_threads: int = 1,
) -> Tuple[list[BiocFrame], BiocFrame]:
    """Annotate a single-cell expression dataset based on the correlation 
    of each cell to profiles in multiple labelled references, where the
    annotation from each reference is then integrated across references.

    Args:
        test_data: A matrix-like object representing the test dataset, where rows are
            features and columns are samples (usually cells). Entries should be expression
            values; only the ranking within each column will be used.

            Alternatively, a
            :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`
            containing such a matrix in one of its assays. 

        test_features: Sequence of length equal to the number of rows in
            ``test_data``, containing the feature identifier for each row.

        ref_data_list:
            Sequence consisting of one or more of the following:

            - A matrix-like object representing the reference dataset, where rows
              are features and columns are samples. Entries should be expression values,
              usually log-transformed (see comments for the ``ref`` argument in
              :py:meth:`~singler.build_single_reference.build_single_reference`).
            - A 
              :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`
              object containing such a matrix in its assays.
            - A string that can be passed as ``name`` to
              :py:meth:`~singler.fetch_reference.fetch_github_reference`.
              This will use the specified dataset as the reference.

        ref_labels_list:
            Sequence of the same length as ``ref_data``, where the contents
            depend on the type of value in the corresponding entry of ``ref_data``:

            - If ``ref_data_list[i]`` is a matrix-like object, ``ref_labels_list[i]`` should be
              a sequence of length equal to the number of columns of ``ref_data_list[i]``,
              containing the label associated with each column.
            - If ``ref_data_list[i]`` is a string, ``ref_labels_list[i]`` should be a string
              specifying the label type to use, e.g., "main", "fine", "ont".
              If a single string is supplied, it is recycled for all ``ref_data``.

        ref_features_list:
            Sequence of the same length as ``ref_data_list``, where the contents
            depend on the type of value in the corresponding entry of ``ref_data``:

            - If ``ref_data_list[i]`` is a matrix-like object, ``ref_features_list[i]`` should be
              a sequence of length equal to the number of rows of ``ref_data_list[i]``,
              containing the feature identifier associated with each row.
            - If ``ref_data_list[i]`` is a string, ``ref_features_list[i]`` should be a string
              specifying the feature type to use, e.g., "ensembl", "symbol".
              If a single string is supplied, it is recycled for all ``ref_data``.

        test_assay_type:
            Assay of ``test_data`` containing the expression matrix, if ``test_data`` is a
            :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`.

        test_check_missing:
            Whether to check for and remove missing (i.e., NaN) values from the test dataset.

        ref_assay_type:
            Assay containing the expression matrix for any entry of ``ref_data_list`` that is a
            :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`.

        ref_check_missing:
            Whether to check for and remove missing (i.e., NaN) values from the reference datasets.

        cache_dir:
            Path to a cache directory for downloading reference files, see
            :py:meth:`~singler.fetch_reference.fetch_github_reference` for details.
            Only used if ``ref_data`` is a string.

        build_single_args:
            Further arguments to pass to
            :py:meth:`~singler.build_single_reference.build_single_reference`.

        classify_single_args:
            Further arguments to pass to
            :py:meth:`~singler.classify_single_reference.classify_single_reference`.

        build_integrated_args:
            Further arguments to pass to
            :py:meth:`~singler.build_integrated_references.build_integrated_references`.

        classify_integrated_args:
            Further arguments to pass to
            :py:meth:`~singler.classify_integrated_references.classify_integrated_references`.

        num_threads:
            Number of threads to use for the various steps.

    Returns:
        Tuple where the first element contains per-reference results (i.e. a
        list of BiocFrame outputs equivalent to running
        :py:meth:`~singler.annotate_single.annotate_single` on each reference)
        and the second element contains integrated results across references
        (i.e., a BiocFrame from
        :py:meth:`~singler.classify_integrated_references.classify_integrated_references`).
    """
    nrefs = len(ref_data_list)
    if isinstance(ref_labels_list, str):
        ref_labels_list = [ref_labels_list] * nrefs
    elif nrefs != len(ref_labels_list):
        raise ValueError("'ref_data_list' and 'ref_labels_list' must be the same length")
    if isinstance(ref_features_list, str):
        ref_features_list = [ref_features_list] * nrefs
    elif nrefs != len(ref_features_list):
        raise ValueError("'ref_data_list' and 'ref_features_list' must be the same length")

    test_ptr, test_features = _clean_matrix(
        test_data,
        test_features,
        assay_type = test_assay_type,
        check_missing = test_check_missing,
        num_threads = num_threads,
    )

    all_ref_data = []
    all_ref_labels = []
    all_ref_features = []
    all_built = []
    all_results = []
    test_features_set = set(test_features)

    for r in range(nrefs):
        curref_mat, curref_labels, curref_features, curref_markers = _resolve_reference(
            ref_data=ref_data_list[r],
            ref_labels=ref_labels_list[r],
            ref_features=ref_features_list[r],
            cache_dir=cache_dir,
            build_args=build_single_args,
            test_features_set=test_features_set,
        )

        curref_ptr, curref_features = _clean_matrix(
            curref_mat,
            curref_features,
            assay_type = ref_assay_type,
            check_missing = ref_check_missing,
            num_threads = num_threads,
        )

        bargs = _attach_markers(curref_markers, build_single_args)
        curbuilt = build_single_reference(
            ref_data=curref_ptr,
            ref_labels=curref_labels,
            ref_features=curref_features,
            restrict_to=test_features_set,
            **bargs,
            num_threads=num_threads,
        )

        res = classify_single_reference(
            test_data,
            test_features=test_features,
            ref_prebuilt=curbuilt,
            **classify_single_args,
            num_threads=num_threads,
        )

        all_ref_data.append(curref_ptr)
        all_ref_labels.append(curref_labels)
        all_ref_features.append(curref_features)
        all_built.append(curbuilt)
        all_results.append(res)

        res.metadata = {
            "markers": curbuilt.markers,
            "unique_markers": curbuilt.marker_subset(),
        }

    ibuilt = build_integrated_references(
        test_features=test_features,
        ref_data_list=all_ref_data,
        ref_labels_list=all_ref_labels,
        ref_features_list=all_ref_features,
        ref_prebuilt_list=all_built,
        **build_integrated_args,
        num_threads=num_threads,
    )

    ires = classify_integrated_references(
        test_data=test_ptr,
        results=all_results,
        integrated_prebuilt=ibuilt,
        **classify_integrated_args,
        num_threads=num_threads,
    )

    return all_results, ires
