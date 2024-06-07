import singler
import numpy
import celldex
import scrnaseq
import pandas as pd
import scipy
import pytest
from biocframe import BiocFrame


def test_with_minimal_args():
    sce = scrnaseq.fetch_dataset("zeisel-brain-2015", "2023-12-14", realize_assays=True)

    blueprint_ref = celldex.fetch_reference(
        "blueprint_encode", "2024-02-26", realize_assays=True
    )
    immune_cell_ref = celldex.fetch_reference("dice", "2024-02-26", realize_assays=True)

    with pytest.raises(Exception):
        singler.annotate_integrated(
            test_data=sce.assays["counts"],
            ref_data_list=(blueprint_ref, immune_cell_ref),
            ref_labels_list="label.main",
            num_threads=6,
        )

    single, integrated = singler.annotate_integrated(
        test_data=sce,
        ref_data_list=(blueprint_ref, immune_cell_ref),
        ref_labels_list="label.main",
        num_threads=6,
    )
    assert len(single) == 2
    assert isinstance(integrated, BiocFrame)


def test_with_all_supplied():
    sce = scrnaseq.fetch_dataset("zeisel-brain-2015", "2023-12-14", realize_assays=True)

    blueprint_ref = celldex.fetch_reference(
        "blueprint_encode", "2024-02-26", realize_assays=True
    )
    immune_cell_ref = celldex.fetch_reference("dice", "2024-02-26", realize_assays=True)

    single, integrated = singler.annotate_integrated(
        test_data=sce,
        test_features=sce.get_row_names(),
        ref_data_list=(blueprint_ref, immune_cell_ref),
        ref_labels_list=[
            x.get_column_data().column("label.main")
            for x in (blueprint_ref, immune_cell_ref)
        ],
        ref_features_list=[x.get_row_names() for x in (blueprint_ref, immune_cell_ref)],
    )

    assert len(single) == 2
    assert isinstance(integrated, BiocFrame)


def test_with_colname():
    sce = scrnaseq.fetch_dataset("zeisel-brain-2015", "2023-12-14", realize_assays=True)

    blueprint_ref = celldex.fetch_reference(
        "blueprint_encode", "2024-02-26", realize_assays=True
    )
    immune_cell_ref = celldex.fetch_reference("dice", "2024-02-26", realize_assays=True)

    single, integrated = singler.annotate_integrated(
        test_data=sce,
        ref_data_list=(blueprint_ref, immune_cell_ref),
        ref_labels_list="label.main",
    )

    assert len(single) == 2
    assert isinstance(integrated, BiocFrame)
