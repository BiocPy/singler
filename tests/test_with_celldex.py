import singler
import numpy
import celldex
import scrnaseq
import pandas as pd
import scipy
import pytest


def test_with_minimal_args():
    sce = scrnaseq.fetch_dataset("zeisel-brain-2015", "2023-12-14", realize_assays=True)

    immgen_ref = celldex.fetch_reference("immgen", "2024-02-26", realize_assays=True)

    with pytest.raises(Exception):
        matches = singler.annotate_single(
            test_data=sce.assays["counts"],
            ref_data=immgen_ref,
            ref_labels=immgen_ref.get_column_data().column("label.main"),
        )

    matches = singler.annotate_single(
        test_data=sce,
        ref_data=immgen_ref,
        ref_labels=immgen_ref.get_column_data().column("label.main"),
    )
    counts = pd.Series(matches["best"]).value_counts()
    assert list(counts) == [1477, 765, 396, 165, 101, 83, 18]

def test_with_all_supplied():
    sce = scrnaseq.fetch_dataset("zeisel-brain-2015", "2023-12-14", realize_assays=True)

    immgen_ref = celldex.fetch_reference("immgen", "2024-02-26", realize_assays=True)

    matches = singler.annotate_single(
        test_data=sce,
        test_features=sce.get_row_names(),
        ref_data=immgen_ref,
        ref_labels=immgen_ref.get_column_data().column("label.main"),
        ref_features=immgen_ref.get_row_names(),
    )

    counts = pd.Series(matches["best"]).value_counts()
    assert list(counts) == [1477, 765, 396, 165, 101, 83, 18]


def test_with_colname():
    sce = scrnaseq.fetch_dataset("zeisel-brain-2015", "2023-12-14", realize_assays=True)

    immgen_ref = celldex.fetch_reference("immgen", "2024-02-26", realize_assays=True)

    matches = singler.annotate_single(
        test_data=sce,
        ref_data=immgen_ref,
        ref_labels="label.main",
    )

    counts = pd.Series(matches["best"]).value_counts()
    assert list(counts) == [1477, 765, 396, 165, 101, 83, 18]
