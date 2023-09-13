from typing import Sequence, Optional
from numpy import array, ndarray, int32, uintp
from mattress import tatamize
from biocframe import BiocFrame

from .build_single_reference import SinglePrebuiltReference
from . import _cpphelpers as lib
from ._utils import _stable_union, _factorize, _match


def classify_integrated_references(
    test_mat,
    results: list[Union[BiocFrame, Sequence]],
    integrated_prebuilt: IntegratedReferences,
    quantile=0.8,
    num_threads=1,
) -> BiocFrame:

    test_ptr = tatamize(test_mat)
    nc = test_ptr.ncol()
    best = ndarray((nc,), dtype=int32)
    delta = ndarray((nc,), dtype=float64)

    irefs = integrated_prebuilt.references
    all_refs = list(irefs.keys())
    nref = len(all_refs)

    scores = {}
    all_levels = []
    score_ptrs = ndarray((nref,), dtype=uintp)
    assign_ptrs = ndarray((nref,), dtype=uintp)

    if len(all_refs) != len(results):
        raise ValueError("length of 'results' should equal number of references in 'integrated_prebuilt'")

    for i, r in enumerate(all_refs):
        current = ndarray((nc,), dtype=float64)
        scores[r] = current
        score_ptrs[i] = current.ctypes.data

        curlabs = results[r]
        if isinstance(curlabs, BiocFrame):
            curlabs = curlabs.column("best")
        if len(curlabs) != nc:
            raise ValueError("each entry of 'results' should have results for all cells in 'test_mat'")

        ind = array(_match(curlabs, irefs[r]), dtype=int32)
        all_labels.append(ind)
        assign_ptrs[i] = ind.ctypes.data

    output = lib.classify_integrated_references(
        test_ptr.ptr,
        assign_ptrs.ctypes.data,
        integrated_prebuilt._ptr,
        quantile,
        score_ptrs.ctypes.data,
        best,
        delta,
        num_threads,
    )

    scores_df = BiocFrame(scores, number_of_rows=nc)
    return BiocFrame(
        {"best": [all_refs[b] for b in best], "scores": scores_df, "delta": delta}
    )
