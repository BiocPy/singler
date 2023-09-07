from .InternalMarkers import InternalMarkers
from . import cpphelpers as lib
from .utils import _factorize
from mattress import tatamize
from numpy import int32, ndarray


class SinglePrebuiltReference:
    """A prebuilt reference object, typically created by
    :py:meth:`~singler.build_single_reference.build_single_reference`.
    This is intended for advanced users only and should not be serialized.

    Attributes:
        labels (Sequence):
            Sequence of unique label identifiers, usually strings.

        features (Sequence):
            Sequence of feature identifiers, usually strings.
            This contains the universe of all features known to the reference,
            not just the markers that are ultimately used for classification.
    """

    def __init__(self, ptr, labels: Sequence, features: Sequence):
        self._ptr = ptr
        self._features = features
        self._labels = labels

    def __del__(self):
        lib.free_single_reference(self._ptr)

    def num_markers(self) -> int:
        """
        Returns:
            int: Number of markers to be used for classification.
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

    def markers(self, indices_only: bool = False) -> Union[ndarray, list]:
        """
        Args:
            indices_only (bool): Whether to return the markers as indices
                into :py:attr:`~features`, or as a list of feature identifiers.

        Returns:
            list: List of feature identifiers for the markers, if ``indices_only = False``.

            ndarray: Integer indices of features in ``features`` chosen as markers,
            if ``indices_only = True``.
        """
        nmarkers = self.num_features()
        buffer = ndarray(nmarkers, dtype=int32)
        lib.get_subset_from_single_reference(self._ptr, buffer);
        if indices_only:
            return buffer
        else:
            return [self._features[i] for i in buffer]


def build_single_reference(
    ref,
    labels: Sequence,
    features: Sequence,
    markers: dict[Any, dict[Any, Sequence]],
    approximate: bool = True,
    num_threads: int = 1,
):
    """Build a single reference dataset in preparation for classification.

    Args:
        ref: A matrix-like object where rows are features and columns are
            reference profiles. This should contain expression values;
            normalized and transformed values are also acceptable as only
            the ranking is used within this function.

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
    """

    lablev, labind = _factorize(labels)
    mrk = InternalMarkers.from_dict(markers, lablev, features)
    mat_ptr = tatamize(ref)

    return SinglePrebuiltReference(
        lib.build_single_reference(
            mat_ptr.ptr,
            labels = labind,
            markers = mrk.ptr,
            approximate = approximate,
            num_threads = num_threads
        ),
        labels = lablev,
        features = features,
    )
