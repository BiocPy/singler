from typing import Sequence
from numpy import array, ndarray, int32, uintp
from mattress import tatamize

from .build_single_reference import SinglePrebuiltReference
from . import _cpphelpers as lib
from ._utils import _stable_union, _factorize, _match


class IntegratedReferences:
    """Object containing integrated references, typically constructed by
    :py:meth:`~singler.build_integrated_references.build_integrated_references`.
    """
    def __init__(self, ptr):
        self._ptr = ptr

    def __del__(self):
        lib.free_integrated_references(self._ptr)

    def num_references(self) -> int:
        """
        Returns:
            int: Number of reference datasets in this object.
        """
        return lib.get_integrated_references_num_references(self._ptr)

    def num_labels(self, r: int) -> int:
        """
        Args:
            r (int): Reference of interest. This should be a non-negative
                index that is less than :py:meth:`~num_references`.

        Returns:
            int: Number of labels in the specified reference.
        """
        return lib.get_integrated_references_num_labels(self._ptr, r)

    def num_profiles(self, r: int) -> int:
        """
        Args:
            r (int): Reference of interest. This should be a non-negative
                index that is less than :py:meth:`~num_references`.

        Returns:
            int: Number of profiles in the specified reference.
        """
        return lib.get_integrated_references_num_profiles(self._ptr, r)


def build_integrated_references(
    test_features: Sequence,
    ref_data_list: list,
    ref_labels_list: list[Sequence],
    ref_features_list: list[Sequence],
    ref_prebuilt_list: list[SinglePrebuiltReference],
    num_threads = 1,
) -> IntegratedReferences:
    """
    Build a set of integrated references for classification of a test dataset.

    Arguments:
        test_features (Sequence): Sequence of features for the test dataset.

        ref_data_list (list): List of reference datasets, equivalent to ``ref_data`` in
            :py:meth:`~singler.build_single_reference.build_single_reference`.

        ref_labels_list (list[Sequence]): List of reference labels, equivalent to ``ref_labels`` in
            :py:meth:`~singler.build_single_reference.build_single_reference`.

        ref_features_list (list[Sequence]): List of reference features, equivalent to ``ref_features`` in
            :py:meth:`~singler.build_single_reference.build_single_reference`.

        ref_prebuilt_list (list[SinglePrebuiltReference]): List of prebuilt references.

        num_threads (int):
            Number of threads.

    Returns:
        IntegratedReferences: Integrated references for classification with
        :py:meth:`~singler.classify_integrated_references.classify_integrated_references`.
    """
    universe = _stable_union(test_features, *ref_features_list)
    test_features = array(_match(test_features, universe), dtype=int32)

    nrefs = len(ref_data_list)
    converted_ref_data = []
    ref_data_ptrs = ndarray(nrefs, dtype=uintp)
    for i, x in enumerate(ref_data_list):
        current = tatamize(x)
        converted_ref_data.append(current)
        ref_data_ptrs[i] = current.ptr

    if nrefs != len(ref_labels_list):
        raise ValueError("'ref_labels_list' and 'ref_data_list' should have the same length")
    converted_label_levels = []
    converted_label_indices = []
    ref_labels_ptrs = ndarray(nrefs, dtype=uintp)
    for i, x in enumerate(ref_labels_list):
        lev, ind = _factorize(x)
        converted_label_levels.append(lev)
        ind = array(ind, dtype=int32)
        converted_label_indices.append(ind)
        ref_labels_ptrs[i] = ind.ctypes.data

    if nrefs != len(ref_features_list):
        raise ValueError("'ref_features_list' and 'ref_data_list' should have the same length")
    converted_feature_data = []
    ref_features_ptrs = ndarray(nrefs, dtype=uintp)
    for i, x in enumerate(ref_features_list):
        ind = array(_match(x, universe), dtype=int32)
        converted_feature_data.append(ind)
        ref_features_ptrs[i] = ind.ctypes.data

    if nrefs != len(ref_prebuilt_list):
        raise ValueError("'ref_prebuilt_list' and 'ref_data_list' should have the same length")
    ref_prebuilt_ptrs = ndarray(nrefs, dtype=uintp)
    for i, x in enumerate(ref_prebuilt_list):
        ref_prebuilt_ptrs[i] = x._ptr

    output = lib.build_integrated_references(
        len(test_features),
        test_features,
        nrefs, 
        ref_data_ptrs.ctypes.data,
        ref_labels_ptrs.ctypes.data,
        ref_features_ptrs.ctypes.data,
        ref_prebuilt_ptrs.ctypes.data,
        num_threads
    )

    return IntegratedReferences(output)
