from numpy import ndarray, int32, uintp
from mattress import tatamize
from typing import Union, Sequence, Optional, Any
import delayedarray

from . import _cpphelpers as lib
from ._utils import _clean_matrix, _stable_intersect, _stable_union, _create_map
from ._Markers import _Markers


def _get_classic_markers_raw(
    ref_ptrs, ref_labels, ref_features, num_de=None, num_threads=1
):
    nrefs = len(ref_ptrs)

    # We assume that ref_ptrs and ref_features contains the outputs of
    # _clean_matrix, so there's no need to re-check their consistency.
    for i, x in enumerate(ref_ptrs):
        nc = x.ncol()
        if nc != len(ref_labels[i]):
            raise ValueError(
                "number of columns of 'ref' should be equal to the length of the corresponding 'labels'"
            )

    # Defining the intersection of features.
    common_features = _stable_intersect(*ref_features)
    if len(common_features) == 0:
        for feat in ref_features:
            if len(feat):
                raise ValueError("no common feature names across 'features'")

    common_features_map = _create_map(common_features)

    # Creating medians.
    ref2 = []
    ref2_ptrs = ndarray((nrefs,), dtype=uintp)
    tmp_labels = []

    for i, x in enumerate(ref_ptrs):
        survivors = []
        remap = [None] * len(common_features)
        for j, f in enumerate(ref_features[i]):
            if f is not None and f in common_features_map:
                survivors.append(j)
                remap[common_features_map[f]] = len(survivors) - 1

        da = delayedarray.DelayedArray(x)[survivors, :]
        ptr = tatamize(da)
        med, lev = ptr.row_medians_by_group(ref_labels[i], num_threads=num_threads)
        tmp_labels.append(lev)

        finalptr = tatamize(med[remap, :])
        ref2.append(finalptr)
        ref2_ptrs[i] = finalptr.ptr

    ref_labels = tmp_labels

    # Defining the union of labels.
    common_labels = _stable_union(*ref_labels)
    common_labels_map = _create_map(common_labels)

    labels2 = []
    labels2_ptrs = ndarray((nrefs,), dtype=uintp)
    for i, lab in enumerate(ref_labels):
        converted = ndarray(len(lab), dtype=int32)
        for j, x in enumerate(lab):
            converted[j] = common_labels_map[x]
        labels2.append(converted)
        labels2_ptrs[i] = converted.ctypes.data

    # Finally getting around to calling markers.
    if num_de is None:
        num_de = -1
    elif num_de <= 0:
        raise ValueError("'num_de' should be positive")

    raw_markers = _Markers(
        lib.find_classic_markers(
            nref=nrefs,
            labels=labels2_ptrs.ctypes.data,
            ref=ref2_ptrs.ctypes.data,
            de_n=num_de,
            nthreads=num_threads,
        )
    )

    return raw_markers, common_labels, common_features


def get_classic_markers(
    ref_data: Union[Any, list[Any]],
    ref_labels: Union[Sequence, list[Sequence]],
    ref_features: Union[Sequence, list[Sequence]],
    assay_type: Union[str, int] = "logcounts",
    check_missing: bool = True,
    num_de: Optional[int] = None,
    num_threads: int = 1,
) -> dict[Any, dict[Any, list]]:
    """Compute markers from a reference using the classic SingleR algorithm. This is typically done for reference
    datasets derived from replicated bulk transcriptomic experiments.

    Args:
        ref_data(Any | list[Any]):
            A matrix-like object containing the log-normalized expression values of a reference dataset.
            Each column is a sample and each row is a feature.
            Alternatively, this can be a :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`
            containing a matrix-like object in one of its assays.
            Alternatively, a list of such matrices or ``SummarizedExperiment`` objects,
            typically for multiple batches of the same reference;
            it is assumed that different batches exhibit at least some overlap in their ``features`` and ``labels``.

        ref_labels (Any | list[Any]):
            A sequence of length equal to the number of columns of ``ref``,
            containing a label (usually a string) for each column.
            Alternatively, a list of such sequences of length equal to that of a list ``ref``;
            each sequence should have length equal to the number of columns of the corresponding entry of ``ref``.

        ref_features (Any | list[Any]):
            A sequence of length equal to the number of rows of ``ref``,
            containing the feature name (usually a string) for each row.
            Alternatively, a list of such sequences of length equal to that of a list ``ref``;
            each sequence should have length equal to the number of rows of the corresponding entry of ``ref``.

        assay_type (str | int):
            Name or index of the assay containing the assay of interest,
            if ``ref`` is or contains
            :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment` objects.

        check_missing (bool):
            Whether to check for and remove rows with missing (NaN) values in the reference matrices.
            This can be set to False if it is known that no NaN values exist.

        num_de (int, optional):
            Number of differentially expressed genes to use as markers for each pairwise comparison between labels.
            If None, an appropriate number of genes is automatically determined.

        num_threads (int):
            Number of threads to use for the calculations.

    Returns:
        dict[Any, dict[Any, list]]: A dictionary of dictionary of lists
        containing the markers for each pairwise comparison between labels,
        i.e., ``markers[a][b]`` contains the upregulated markers for label
        ``a`` over label ``b``.
    """
    if not isinstance(ref_data, list):
        ref_data = [ref_data]
        ref_labels = [ref_labels]
        ref_features = [ref_features]

    nrefs = len(ref_data)
    if nrefs != len(ref_labels):
        raise ValueError("length of 'ref' and 'labels' should be the same")
    if nrefs != len(ref_features):
        raise ValueError("length of 'ref' and 'features' should be the same")

    ref_ptrs = []
    tmp_features = []
    for i in range(nrefs):
        r, f = _clean_matrix(
            ref_data[i],
            ref_features[i],
            assay_type=assay_type,
            check_missing=check_missing,
            num_threads=num_threads,
        )
        ref_ptrs.append(r)
        tmp_features.append(f)

    ref_features = tmp_features

    raw_markers, common_labels, common_features = _get_classic_markers_raw(
        ref_ptrs=ref_ptrs,
        ref_labels=ref_labels,
        ref_features=ref_features,
        num_de=num_de,
        num_threads=num_threads,
    )

    return raw_markers.to_dict(common_labels, common_features)


def number_of_classic_markers(num_labels: int) -> int:
    """Compute the number of markers to detect for a given number of labels, using the classic SingleR marker detection
    algorithm.

    Args:
        num_labels (int): Number of labels.

    Returns:
        int: Number of markers.
    """
    return lib.number_of_classic_markers(num_labels)
