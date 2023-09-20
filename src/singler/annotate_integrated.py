from typing import Union, Sequence, Optional, Any, Tuple
from biocframe import BiocFrame

from .fetch_reference import fetch_github_reference, realize_github_markers
from .build_single_reference import build_single_reference
from .classify_single_reference import classify_single_reference
from .build_integrated_references import build_integrated_references
from .classify_integrated_references import classify_integrated_references
from .annotate_single import _build_reference


def annotate_integrated(
    test_data: Any,
    test_features: Sequence,
    ref_data: Sequence[Union[Any, str]],
    ref_labels: Union[str, Sequence[Union[Sequence, str]]],
    ref_features: Union[str, Sequence[Union[Sequence, str]]],
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

        test_features: Sequence of length equal to the number of rows in
            ``test_data``, containing the feature identifier for each row.

        ref_data:
            Sequence consisting of one or more of the following:

            - A matrix-like object representing the reference dataset, where rows
              are features and columns are samples. Entries should be expression values,
              usually log-transformed (see comments for the ``ref`` argument in
              :py:meth:`~singler.build_single_reference.build_single_reference`).
            - A string that can be passed as ``name`` to
              :py:meth:`~singler.fetch_reference.fetch_github_reference`.
              This will use the specified dataset as the reference.

        ref_labels:
            Sequence of the same length as ``ref_data``, where the contents
            depend on the type of value in the corresponding entry of ``ref_data``:

            - If ``ref_data[i]`` is a matrix-like object, ``ref_labels[i]`` should be
              a sequence of length equal to the number of columns of ``ref_data[i]``,
              containing the label associated with each column.
            - If ``ref_data[i]`` is a string, ``ref_labels[i]`` should be a string
              specifying the label type to use, e.g., "main", "fine", "ont".
              If a single string is supplied, it is recycled for all ``ref_data``.

        ref_features:
            Sequence of the same length as ``ref_data``, where the contents
            depend on the type of value in the corresponding entry of ``ref_data``:

            - If ``ref_data[i]`` is a matrix-like object, ``ref_features[i]`` should be
              a sequence of length equal to the number of rows of ``ref_data``,
              containing the feature identifier associated with each row.
            - If ``ref_data[i]`` is a string, ``ref_features[i]`` should be a string
              specifying the feature type to use, e.g., "ensembl", "symbol".
              If a single string is supplied, it is recycled for all ``ref_data``.

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
    nrefs = len(ref_data)
    if isinstance(ref_labels, str):
        ref_labels = [ref_labels] * nrefs
    elif nrefs != len(ref_labels):
        raise ValueError("'ref_data' and 'ref_labels' must be the same length")
    if isinstance(ref_features, str):
        ref_features = [ref_features] * nrefs
    elif nrefs != len(ref_features):
        raise ValueError("'ref_data' and 'ref_features' must be the same length")

    all_ref_data = []
    all_ref_labels = []
    all_ref_features = []
    all_built = []
    all_results = []
    test_features_set = set(test_features)

    for r in range(nrefs):
        curref_data, curref_labels, curref_features, curbuilt = _build_reference(
            ref_data=ref_data[r],
            ref_labels=ref_labels[r],
            ref_features=ref_features[r],
            test_features_set=test_features_set,
            cache_dir=cache_dir,
            build_args=build_single_args,
            num_threads=num_threads,
        )

        res = classify_single_reference(
            test_data,
            test_features=test_features,
            ref_prebuilt=curbuilt,
            **classify_single_args,
            num_threads=num_threads,
        )

        res.metadata = {
            "markers": curbuilt.markers,
            "unique_markers": curbuilt.marker_subset(),
        }

        all_ref_data.append(curref_data)
        all_ref_labels.append(curref_labels)
        all_ref_features.append(curref_features)
        all_built.append(curbuilt)
        all_results.append(res)

    ibuilt = build_integrated_references(
        test_features=test_features,
        ref_data_list=all_ref_data,
        ref_labels_list=all_ref_labels,
        ref_features_list=all_ref_features,
        ref_prebuilt_list=all_built,
        num_threads=num_threads,
        **build_integrated_args,
    )

    ires = classify_integrated_references(
        test_data=test_data,
        results=all_results,
        integrated_prebuilt=ibuilt,
        **classify_integrated_args,
        num_threads=num_threads,
    )

    return all_results, ires
