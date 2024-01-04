import singler
import numpy


def test_annotate_single_sanity():
    ref = numpy.random.rand(10000, 10) + 1
    ref[:2000, :2] = 0
    ref[2000:4000, 2:4] = 0
    ref[4000:6000, 4:6] = 0
    ref[6000:8000, 6:8] = 0
    ref[8000:, 8:] = 0
    labels = ["A", "A", "B", "B", "C", "C", "D", "D", "E", "E"]

    test = numpy.random.rand(10000, 5) + 1
    test[2000:4000, 0] = 0  # B
    test[6000:8000, 1] = 0  # D
    test[:2000, 2] = 0  # A
    test[8000:, 3] = 0  # E
    test[4000:6000, 4] = 0  # C

    all_features = [str(i) for i in range(10000)]
    output = singler.annotate_single(
        test,
        test_features=all_features,
        ref_data=ref,
        ref_features=all_features,
        ref_labels=labels,
    )

    assert output.shape[0] == 5
    assert output.column("best") == ["B", "D", "A", "E", "C"]


def test_annotate_single_intersect():
    ref = numpy.random.rand(10000, 10)
    ref_features = [str(i) for i in range(10000)]
    ref_labels = ["A", "A", "B", "B", "C", "C", "D", "D", "E", "E"]
    test = numpy.random.rand(10000, 50)
    test_features = [str(i + 2000) for i in range(10000)]

    output = singler.annotate_single(
        test,
        test_features=test_features,
        ref_data=ref,
        ref_features=ref_features,
        ref_labels=ref_labels,
    )

    built = singler.build_single_reference(
        ref[2000:, :], ref_labels=ref_labels, ref_features=ref_features[2000:]
    )
    expected = singler.classify_single_reference(
        test[:8000, :], test_features[:8000], built
    )

    assert output.column("best") == expected.column("best")
    assert (output.column("delta") == expected.column("delta")).all()
    assert (
        output.column("scores").column("B") == expected.column("scores").column("B")
    ).all()


def test_annotate_single_github():
    se = singler.fetch_github_reference("ImmGen", cache_dir="_cache")

    keep = range(5, se.shape[0], 2)
    test = numpy.random.rand(len(keep), 50)
    ref_features = se.row_data.column("symbol")
    test_features = [ref_features[i] for i in keep]

    output = singler.annotate_single(
        test,
        test_features=test_features,
        ref_data="ImmGen",
        ref_features="symbol",
        ref_labels="main",
        cache_dir="_cache",
    )
    assert output.shape[0] == 50

    expected_markers = singler.realize_github_markers(
        se.metadata["main"],
        se.row_data.column("symbol"),
        restrict_to=set(test_features),
    )
    assert output.metadata["markers"] == expected_markers

    # Checking that we handle the number of markers correctly.
    more_output = singler.annotate_single(
        test,
        test_features=test_features,
        ref_data="ImmGen",
        ref_features="symbol",
        ref_labels="main",
        build_args={"marker_args": {"num_de": 10}},
        cache_dir="_cache",
    )

    ref_labels = list(set(se.column_data.column("main")))
    assert len(more_output.metadata["markers"][ref_labels[0]][ref_labels[1]]) == 10
