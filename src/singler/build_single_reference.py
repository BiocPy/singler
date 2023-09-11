from numpy import int32, array, ndarray
from typing import Sequence, Union, Any, Optional, Literal
from delayedarray import DelayedArray
from mattress import tatamize

from ._Markers import _Markers
from . import _cpphelpers as lib
from ._utils import _factorize, _match, _clean_matrix
from .get_classic_markers import _get_classic_markers_raw


class SinglePrebuiltReference:
    """A prebuilt reference object, typically created by
    :py:meth:`~singler.build_single_reference.build_single_reference`. This is intended for advanced users only and
    should not be serialized.

    Attributes:
        labels (Sequence):
            Sequence of unique label identifiers, usually strings.

        features (Sequence):
            Sequence of feature identifiers, usually strings.
            This contains the universe of all features known to the reference,
            not just the markers that are ultimately used for classification.

        markers (dict[Any, dict[Any, Sequence]]):
            Upregulated markers for each pairwise comparison between labels.
    """

    def __init__(
        self,
        ptr,
        labels: Sequence,
        features: Sequence,
        markers: dict[Any, dict[Any, Sequence]],
    ):
        self._ptr = ptr
        self._features = features
        self._labels = labels
        self._markers = markers

    def __del__(self):
        lib.free_single_reference(self._ptr)

    def num_markers(self) -> int:
        """
        Returns:
            int: Number of markers to be used for classification. This is the
            same as the size of the array from :py:meth:`~marker_subset`.
        """
        return lib.get_nsubset_from_single_reference(self._ptr)

    def num_labels(self) -> int:
        """
        Returns:
            int: Number of unique labels in this reference.
        """
        return lib.get_nlabels_from_single_reference(self._ptr)

    @property
    def features(self) -> Sequence:
        """
        Returns:
            Sequence: The universe of features known to this reference,
            usually as strings.
        """
        return self._features

    @property
    def labels(self) -> Sequence:
        """
        Returns:
            Sequence: Unique labels in this reference.
        """
        return self._labels

    @property
    def markers(self) -> dict[Any, dict[Any, Sequence]]:
        """
        Returns:
            dict[Any, dict[Any, Sequence]]: Markers for every pairwise comparison
            between labels.
        """
        return self._markers

    def marker_subset(self, indices_only: bool = False) -> Union[ndarray, list]:
        """
        Args:
            indices_only (bool): Whether to return the markers as indices
                into :py:attr:`~features`, or as a list of feature identifiers.

        Returns:
            list: List of feature identifiers for the markers, if ``indices_only = False``.

            ndarray: Integer indices of features in ``features`` that were
            chosen as markers, if ``indices_only = True``.
        """
        nmarkers = self.num_markers()
        buffer = ndarray(nmarkers, dtype=int32)
        lib.get_subset_from_single_reference(self._ptr, buffer)
        if indices_only:
            return buffer
        else:
            return [self._features[i] for i in buffer]


MARKER_DETECTION_METHODS = Literal["classic"]


def build_single_reference(
    ref_data,
    ref_labels: Sequence,
    ref_features: Sequence,
    assay_type: Union[str, int] = "logcounts",
    check_missing: bool = True,
    restrict_to: Optional[Union[set, dict]] = None,
    markers: Optional[dict[Any, dict[Any, Sequence]]] = None,
    marker_method: MARKER_DETECTION_METHODS = "classic",
    marker_args={},
    approximate: bool = True,
    num_threads: int = 1,
) -> SinglePrebuiltReference:
    """Build a single reference dataset in preparation for classification.

    Args:
        ref_data: A matrix-like object where rows are features, columns are
            reference profiles, and each entry is the expression value.
            If `markers` is not provided, expression should be normalized
            and log-transformed in preparation for marker prioritization via
            differential expression analyses. Otherwise, any expression values
            are acceptable as only the ranking within each column is used.

            Alternatively, a
            :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`
            containing such a matrix in one of its assays.

        labels (Sequence): Sequence of labels for each reference profile,
            i.e., column in ``ref``.

        features (Sequence): Sequence of identifiers for each feature,
            i.e., row in ``ref``.

        assay_type(str | int): Assay containing the expression matrix,
            if `ref_data` is a
            :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`.

        check_missing (bool):
            Whether to check for and remove rows with missing (NaN) values
            from ``ref_data``.

        restrict_to (Union[set, dict], optional):
            Subset of available features to restrict to. Only features in
            ``restrict_to`` will be used in the reference building. If None,
            no restriction is performed.

        markers (dict[Any, dict[Any, Sequence]], optional):
            Upregulated markers for each pairwise comparison between labels.
            Specifically, ``markers[a][b]`` should be a sequence of features
            that are upregulated in ``a`` compared to ``b``. All such features
            should be present in ``features``, and all labels in ``labels``
            should have keys in the inner and outer dictionaries.

        marker_method (MARKER_DETECTION_METHODS):
            Method to identify markers from each pairwise comparisons between
            labels in ``ref_data``.  If "classic", we call
            :py:meth:`~singler.get_classic_markers.get_classic_markers`.
            Only used if ``markers`` is not supplied.

        marker_args:
            Further arguments to pass to the chosen marker detection method.
            Only used if ``markers`` is not supplied.

        approximate (bool):
            Whether to use an approximate neighbor search to compute scores
            during classification.

        num_threads (int):
            Number of threads to use for reference building.

    Returns:
        SinglePrebuiltReference: The pre-built reference, ready for use in downstream
        methods like :py:meth:`~singler.classify_single_reference.classify_single_reference`.
    """

    ref_ptr, ref_features = _clean_matrix(
        ref_data,
        ref_features,
        assay_type=assay_type,
        check_missing=check_missing,
        num_threads=num_threads,
    )

    if restrict_to is not None:
        keep = []
        new_features = []
        for i, x in enumerate(ref_features):
            if x in restrict_to:
                keep.append(i)
                new_features.append(x)
        ref_features = new_features
        ref_ptr = tatamize(DelayedArray(ref_ptr)[keep,:])

    if markers is None:
        if marker_method == "classic":
            mrk, lablev, ref_features = _get_classic_markers_raw(
                ref_ptrs=[ref_ptr],
                ref_labels=[ref_labels],
                ref_features=[ref_features],
                num_threads=num_threads,
                **marker_args,
            )
            markers = mrk.to_dict(lablev, ref_features)
            labind = array(_match(ref_labels, lablev), dtype=int32)
        else:
            raise NotImplementedError("other marker methods are not implemented, sorry")
    else:
        lablev, labind = _factorize(ref_labels)
        labind = array(labind, dtype=int32)
        mrk = _Markers.from_dict(markers, lablev, ref_features)

    return SinglePrebuiltReference(
        lib.build_single_reference(
            ref_ptr.ptr,
            labels=labind,
            markers=mrk._ptr,
            approximate=approximate,
            nthreads=num_threads,
        ),
        labels=lablev,
        features=ref_features,
        markers=markers,
    )
