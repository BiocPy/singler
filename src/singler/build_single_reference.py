from .InternalMarkers import InternalMarkers
from . import cpphelpers as lib
from .utils import _factorize
from mattress import tatamize
from numpy import int32


class SinglePrebuiltReference:
    def __init__(self, ptr, labels: Sequence, features: Sequence):
        self._ptr = ptr
        self._features = features
        self._labels = labels

    def __del__(self):
        lib.free_single_reference(self._ptr)

    def num_features(self):
        return lib.get_nsubset_from_single_reference(self._ptr)

    def num_labels(self):
        return lib.get_nlabels_from_single_reference(self._ptr)

    @property
    def features(self):
        return self._features

    @property
    def labels(self):
        return self._labels

    def subset(self):
        nmarkers = self.num_features()
        buffer = ndarray(nmarkers, dtype=int32)
        lib.get_subset_from_single_reference(self._ptr, buffer);
        return [self._features[i] for i in buffer]


def build_single_reference(
    ref,
    labels: Sequence,
    features: Sequence,
    markers: dict[Any, dict[Any, Sequence]],
    approximate: bool = True,
    num_threads: int = 1,
):
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
