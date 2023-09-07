from .InternalMarkers import InternalMarkers
from . import cpphelpers as lib
from .utils import _factorize
from mattress import tatamize
from numpy import int32


class SinglePrebuiltReference:
    def __init__(self, ptr):
        self._ptr = ptr

    def __del__(self):
        lib.free_prebuilt(self._ptr)

    def num_features(self):
        return lib.get_nsubset_from_prebuilt(self._ptr)

    def num_labels(self):
        return lib.get_nlabels_from_prebuilt(self._ptr)

    def subset(self, features: Sequence):
        nmarkers = self.num_features()
        buffer = ndarray(nmarkers, dtype=int32)
        lib.get_subset_from_prebuilt(self._ptr, buffer);
        return [features[i] for i in buffer]


def train_single_reference(
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
        lib.prebuild_reference(
            mat_ptr.ptr, 
            labels = labind, 
            markers = mrk.ptr, 
            approximate = approximate, 
            num_threads = num_threads
        )
    )
