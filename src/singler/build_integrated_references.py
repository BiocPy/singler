from typing import Sequence, Optional, Union
from numpy import array, ndarray, int32, uintp
from mattress import tatamize

from .build_single_reference import SinglePrebuiltReference
from . import _cpphelpers as lib
from ._utils import _stable_union, _factorize, _match


class IntegratedReferences:
    """Object containing integrated references, typically constructed by
    :py:meth:`~singler.build_integrated_references.build_integrated_references`."""

    def __init__(self, ptr, ref_names, ref_labels, test_features):
        self._ptr = ptr
        self._names = ref_names
        self._labels = ref_labels
        self._features = test_features

    def __del__(self):
        lib.free_integrated_references(self._ptr)

    @property
    def reference_names(self) -> Union[Sequence[str], None]:
        """Sequence containing the names of the references. Alternatively
        None, if no names were supplied."""
        return self._names

    @property
    def reference_labels(self) -> list:
        """List of lists containing the names of the labels for each reference.

        Each entry corresponds to a reference in :py:attr:`~reference_names`, 
        if ``reference_names`` is not None.
        """
        return self._labels

    @property
    def test_features(self) -> list[str]:
        """Sequence containing the names of the test features."""
        return self._features


def build_integrated_references(
    test_features: Sequence,
    ref_data_list: dict,
    ref_labels_list: list[Sequence],
    ref_features_list: list[Sequence],
    ref_prebuilt_list: list[SinglePrebuiltReference],
    ref_names: Optional[Sequence[str]] = None,
    num_threads: int = 1,
) -> IntegratedReferences:
    """Build a set of integrated references for classification of a test dataset.

    Arguments:
        test_features: Sequence of features for the test dataset.

        ref_data_list: List of reference datasets, where each entry is equivalent to ``ref_data`` in
            :py:meth:`~singler.build_single_reference.build_single_reference`.

        ref_labels_list: List of reference labels, where each entry is equivalent to ``ref_labels`` in
            :py:meth:`~singler.build_single_reference.build_single_reference`.

        ref_features_list: List of reference features, where each entry is equivalent to ``ref_features`` in
            :py:meth:`~singler.build_single_reference.build_single_reference`.

        ref_prebuilt_list: List of prebuilt references, typically created by 
            calling :py:meth:`~singler.build_single_reference.build_single_reference` on the corresponding
            elements of ``ref_data_list``, ``ref_labels_list`` and ``ref_features_list``.

        ref_names: Sequence of names for the references.
            If None, these are automatically generated.

        num_threads:
            Number of threads.

    Returns:
        Integrated references for classification with
        :py:meth:`~singler.classify_integrated_references.classify_integrated_references`.
    """
    universe = _stable_union(test_features, *ref_features_list)
    original_test_features = test_features
    test_features = array(_match(test_features, universe), dtype=int32)

    nrefs = len(ref_data_list)
    converted_ref_data = []
    ref_data_ptrs = ndarray(nrefs, dtype=uintp)
    for i, x in enumerate(ref_data_list):
        current = tatamize(x)
        converted_ref_data.append(current)
        ref_data_ptrs[i] = current.ptr

    if nrefs != len(ref_labels_list):
        raise ValueError(
            "'ref_labels_list' and 'ref_data_list' should have the same length"
        )
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
        raise ValueError(
            "'ref_features_list' and 'ref_data_list' should have the same length"
        )
    converted_feature_data = []
    ref_features_ptrs = ndarray(nrefs, dtype=uintp)
    for i, x in enumerate(ref_features_list):
        ind = array(_match(x, universe), dtype=int32)
        converted_feature_data.append(ind)
        ref_features_ptrs[i] = ind.ctypes.data

    if nrefs != len(ref_prebuilt_list):
        raise ValueError(
            "'ref_prebuilt_list' and 'ref_data_list' should have the same length"
        )
    ref_prebuilt_ptrs = ndarray(nrefs, dtype=uintp)
    for i, x in enumerate(ref_prebuilt_list):
        ref_prebuilt_ptrs[i] = x._ptr

    if ref_names is not None:
        if nrefs != len(ref_names):
            raise ValueError("'ref_names' and 'ref_data_list' should have the same length")
        elif nrefs != len(set(ref_names)):
            raise ValueError("'ref_names' should contain unique names")

    output = lib.build_integrated_references(
        len(test_features),
        test_features,
        nrefs,
        ref_data_ptrs.ctypes.data,
        ref_labels_ptrs.ctypes.data,
        ref_features_ptrs.ctypes.data,
        ref_prebuilt_ptrs.ctypes.data,
        num_threads,
    )

    return IntegratedReferences(
        output, ref_names, converted_label_levels, original_test_features
    )
