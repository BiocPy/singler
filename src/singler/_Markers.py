from . import cpphelpers as lib
from numpy import ndarray, int32, array
from typing import Sequence, Any


class _Markers:
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

    def set(self, first: int, second: int, markers: Sequence):
        self._check(first)
        self._check(second)
        out = array(markers, dtype=int32, copy=False)
        lib.set_markers_for_pair(self._ptr, first, second, len(out), out)

    def to_dict(
        self, labels: Sequence, features: Sequence
    ) -> dict[Any, dict[Any, Sequence]]:
        if len(labels) != self._num_labels:
            raise ValueError(
                "length of 'labels' should be equal to the number of labels"
            )

        markers = {}
        for i, x in enumerate(labels):
            current = {}
            for j, y in enumerate(labels):
                current[y] = [features[k] for k in self.get(i, j)]
            markers[x] = current

        return markers

    @classmethod
    def from_dict(
        cls,
        markers: dict[Any, dict[Any, Sequence]],
        labels: Sequence,
        features: Sequence,
    ):
        fmapping = {}
        for i, x in enumerate(features):
            fmapping[x] = i

        instance = cls(lib.create_markers(len(labels)))

        for outer_i, outer_k in enumerate(labels):
            for inner_i, inner_k in enumerate(labels):
                current = markers[outer_k][inner_k]

                mapped = []
                for x in current:
                    if x in fmapping:  # just skipping features that aren't present.
                        mapped.append(fmapping[x])

                instance.set(outer_i, inner_i, mapped)

        return instance
