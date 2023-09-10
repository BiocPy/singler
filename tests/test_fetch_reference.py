import singler
import summarizedexperiment
import re
import numpy


def test_fetch_github_reference():
    out = singler.fetch_github_reference("ImmGen", cache_dir="_cache")
    assert isinstance(out, summarizedexperiment.SummarizedExperiment)

    # Checking the genes.
    assert out.row_data.column("ensembl")[0].startswith("ENS")
    assert re.match("^[0-9]+", out.row_data.column("entrez")[0]) is not None
    assert re.match("^[A-Z][a-z]+[0-9]*", out.row_data.column("symbol")[0]) is not None

    # Checking the labels.
    assert isinstance(out.col_data.column("fine")[0], str)
    assert isinstance(out.col_data.column("main")[0], str)
    assert isinstance(out.col_data.column("ont")[0], str)

    # Checking the assay.
    ass = out.assays["ranks"]
    assert ass.shape[0] > ass.shape[1]
    assert (ass.min(0) == numpy.ones(ass.shape[1])).all()

    # Checking markers.
    markers = out.metadata["fine"]
    flabs = out.col_data.column("fine")
    all_labels = sorted(list(set(flabs)))
    assert sorted(markers.keys()) == all_labels
    assert sorted(markers[all_labels[0]].keys()) == all_labels
    assert len(markers[all_labels[0]][all_labels[0]]) == 0
    assert len(markers[all_labels[0]][all_labels[1]]) > 0


def test_realize_github_markers():
    markers = { "A": { "B": [ 1, 3, 5, 7 ] }}
    out = singler.realize_github_markers(markers, ["A", "B", "C", "D", "E", "F", "G", "H"])
    assert out["A"]["B"] == [ "B", "D", "F", "H" ]

    out = singler.realize_github_markers(markers, ["A", "B", "C", "D", "E", "F", "G", "H"], number=2)
    assert out["A"]["B"] == [ "B", "D" ]

    out = singler.realize_github_markers(markers, ["A", "B", "C", None, "E", "F", "G", "H"], number=2)
    assert out["A"]["B"] == [ "B", "F" ]
