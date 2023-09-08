from numpy import int32, ndarray
from typing import Sequence, Union, Any, Optional, Literal

from .InternalMarkers import InternalMarkers
from . import cpphelpers as lib
from .utils import _factorize, _match


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


MarkerDetectionMethods = Literal["classic"]


def build_single_reference(
    ref,
    labels: Sequence,
    features: Sequence,
    assay_type: Union[str, int] = "logcounts",
    check_missing: bool = True,
    markers: Optional[dict[Any, dict[Any, Sequence]]] = None,
    marker_method: MarkerDetectionMethods = "classic",
    num_de: Optional[int] = None,
    approximate: bool = True,
    num_threads: int = 1,
) -> SinglePrebuiltReference:
    """Build a single reference dataset in preparation for classification.

    Args:
        ref: A matrix-like object where rows are features, columns are
            reference profiles, and each entry is the expression value.
            If `markers` is not provided, expression should be normalized
            and log-transformed in preparation for marker prioritization via
            differential expression analyses. Otherwise, any expression values
            are acceptable as only the ranking within each column is used. 

        labels (Sequence): Sequence of labels for each reference profile,
            i.e., column in ``ref``.

        features (Sequence): Sequence of identifiers for each feature,
            i.e., row in ``ref``.

        markers (dict[Any, dict[Any, Sequence]]):
            Upregulated markers for each pairwise comparison between labels.
            Specifically, ``markers[a][b]`` should be a sequence of features
            that are upregulated in ``a`` compared to ``b``. All such features
            should be present in ``features``, and all labels in ``labels``
            should have keys in the inner and outer dictionaries.

        approximate (bool):
            Whether to use an approximate neighbor search to compute scores
            during classification.

        num_threads (int):
            Number of threads to use for reference building.

    Returns:
        SinglePrebuiltReference: The pre-built reference, ready for use in downstream
        methods like :py:meth:`~singler.classify_single_reference.classify_single_reference`.
    """

    ref, features = _clean_matrix(
        ref,
        features,
        assay_type=assay_type,
        check_missing=check_missing,
        num_threads=num_threads,
    )

    if markers is None:
        if marker_method == "classic":
            mrk, lablev, common_features = _get_classic_markers_raw(
                ref,
                labels,
                features,
                check_missing=False,
                num_de=num_de,
                num_threads=num_threads,
            )
            markers = mrk.to_dict(lablev, common_features)
            labind = _match(labels, lablev)
        else:
            raise NotImplementedError("other marker methods are not implemented, sorry")
    else:
        lablev, labind = _factorize(labels)
        mrk = InternalMarkers.from_dict(markers, lablev, features)

    return SinglePrebuiltReference(
        lib.build_single_reference(
            mat_ptr.ptr,
            labels=labind,
            markers=mrk._ptr,
            approximate=approximate,
            nthreads=num_threads,
        ),
        labels=lablev,
        features=features,
        markers=markers,
    )
