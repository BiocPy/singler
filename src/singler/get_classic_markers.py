from numpy import ndarray, int32, float64, uintp
from delayedarray import DelayedArray
from mattress import tatamize
from typing import Union, Sequence, Optional

from . import cpphelpers as lib
from .utils import _factorize
from .InternalMarkers import InternalMarkers


def get_classic_markers(
    ref : Union[Any, list[Any]],
    labels: Union[Sequence, list[Sequence]],
    features: Union[Sequence, list[Sequence]],
    assay_type: Union[str, int] = "logcounts",
    check_missing: bool = True,
    num_de: Optional[int] = None,
    num_threads: int = 1,
):
    """Compute markers from a reference using the classic SingleR algorithm.
    This is typically done for reference datasets derived from replicated bulk transcriptomic experiments.

    Args:
        ref (Any | list[Any]):
            A matrix-like object containing the expression values of a reference dataset.
            Each column is a sample and each row is a feature.
            Alternatively, this can be a :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`
            containing a matrix-like object in one of its assays.
            Alternatively, a list of such matrices or ``SummarizedExperiment`` objects, typically for multiple batches of the same reference;
            it is assumed that different batches exhibit at least some overlap in their ``features`` and ``labels``.

        labels (Any | list[Any]):
            A sequence of length equal to the number of columns of ``ref``, 
            containing a label (usually a string) for each column.
            Alternatively, a list of such sequences of length equal to that of a list ``ref``;
            each sequence should have length equal to the number of columns of the corresponding entry of ``ref``.

        features (Any | list[Any]):
            A sequence of length equal to the number of rows of ``ref``, 
            containing the feature name (usually a string) for each row.
            Alternatively, a list of such sequences of length equal to that of a list ``ref``;
            each sequence should have length equal to the number of rows of the corresponding entry of ``ref``.

        assay_type (str | int):
            Name or index of the assay containing the assay of interest,
            if ``ref`` is or contains :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment objects.
            
        check_missing (bool):
            Whether to check for and remove rows with missing (NaN) values in a matrix ``ref`` or entries of a list ``ref``
            
        num_de (int, optional):
            Number of differentially expressed genes to use as markers for each pairwise comparison between labels.
            If None, an appropriate number of genes is automatically determined.

        num_threads (int):
            Number of threads to use for the calculations.
    """
    
    if not isinstance(ref, list):
        ref = [ref]
        labels = [labels]
        features = [features]

    # Validating everyone's lengths.
    nrefs = len(ref)
    if nrefs != len(labels):
        raise ValueError("length of 'ref' and 'labels' should be the same")
    if nrefs != len(features):
        raise ValueError("length of 'ref' and 'features' should be the same")

    for i in range(nrefs):
        curref = ref[i]
        curshape = curref.shape
        if len(curshape) != 2:
            raise ValueError("each entry of 'ref' should be a 2-dimensional array")
        if curshape[0] != len(features[i]):
            raise ValueError("number of rows of 'ref' should be equal to the length of the corresponding 'features'")
        if curshape[1] != len(labels[i]):
            raise ValueError("number of columns of 'ref' should be equal to the length of the corresponding 'labels'")

    # Defining the intersection of features.
    last = set()
    if len(features):
        last = set(features[0])
        survivors = None
        for i in range(1, len(features)):
            survivors = set()
            for f in features[i]:
                if f in last:
                    survivors.add(f)
            last = survivors

    if len(last) == 0:
        for feat in features:
            if len(feat):
                raise ValueError("no common feature names across 'features'")

    common_features = list(last)
    common_features_map = {}
    for i, x in enumerate(common_features):
        common_features_map[x] = i

    # Creating medians.
    ref2 = []
    ref2_ptrs = ndarray((nrefs,), dtype=uintp)
    for i, x in enumerate(ref):
        survivors = []
        remap = [None] * len(common_features)
        for j, f in enumerate(features[i]):
            if f in common_features_map:
                survivors.append(j)
                remap[common_features_map[f]] = len(survivors) - 1

        current = DelayedArray(x)[survivors,:]
        ptr = tatamize(current)

        flevels, findices = _factorize(labels[i])
        output = ndarray((len(survivors), len(flevels)), dtype=float64)
        lib.grouped_medians(ptr.ptr, findices, len(flevels), output, num_threads)

        finalptr = tatamize(output[remap, :])
        ref2.append(finalptr)
        ref2_ptrs[i] = finalptr.ptr

    # Defining the union of labels.
    ulabels = set()
    for l in labels:
        ulabels |= set(l)

    common_labels = list(ulabels)
    common_labels_map = {}
    for i, x in enumerate(common_labels):
        common_labels_map[x] = i

    labels2 = []
    labels2_ptrs = ndarray((nrefs,), dtype=uintp)
    for i, lab in enumerate(labels):
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

    raw_markers = InternalMarkers(
        lib.find_classic_markers(
            nref = nrefs,
            labels = labels2_ptrs.ctypes.data,
            ref = ref2_ptrs.ctypes.data,
            de_n = num_de,
            nthreads = num_threads,
        )
    )

    markers = {}
    for i, x in enumerate(common_labels):
        current = {}
        for j, y in enumerate(common_labels):
            current[y] = [common_features[k] for k in raw_markers.get(i, j)]
        markers[x] = current

    return markers 
