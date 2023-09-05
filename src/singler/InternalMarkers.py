from . import cpphelpers as lib
from numpy import ndarray, int32


class InternalMarkers:
    def __init__(self, ptr):
        self._ptr = ptr
        self._num_labels = lib.get_nlabels_from_markers(self._ptr)

    def __del__(self):
        lib.free_markers(self._ptr)

    def num_labels(self) -> int:
        return self._num_labels

    def _check(self, i: int):
        if i < 0 or i >= self._num_labels:
            raise IndexError("label " + str(i) + " out of range for marker list")

    def get(self, first: int, second: int):
        self._check(first)
        self._check(second)
        n = lib.get_nmarkers_for_pair(self._ptr, first, second)
        output = ndarray(n, dtype=int32)
        lib.get_markers_for_pair(self._ptr, first, second, output)
        return output

    def set(self, first: int, second: int, markers: ndarray):
        self._check(first)
        self._check(second)
        out = markers.astype(int32, copy=False)
        lib.set_markers_for_pair(self._ptr, first, second, len(out), out)


def create_new_markers(num_labels: int) -> InternalMarkers:
    out = lib.create_new_markers(num_labels)
    return InternalMarkers(out)
