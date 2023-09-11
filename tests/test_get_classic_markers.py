import numpy
import singler


def test_get_classic_markers_simple():
    ref = numpy.random.rand(10000, 10)
    labels = ["A", "B", "C", "D", "E", "E", "D", "C", "B", "A"]
    features = [str(i) for i in range(ref.shape[0])]

    markers = singler.get_classic_markers(ref, labels, features)
    assert "A" in markers
    assert len(markers) == 5
    assert "B" in markers["A"]
    assert len(markers["A"]) == 5

    current = markers["A"]["B"]
    assert len(current) < 10000
    assert singler.number_of_classic_markers(5) == len(current)

    val = int(current[0])
    assert val >= 0 and val < 10000


def test_get_classic_markers_medians():
    ref = numpy.random.rand(10000, 10)
    labels = ["A", "B", "C", "D", "E", "E", "D", "C", "B", "A"]
    features = [str(i) for i in range(ref.shape[0])]
    markers = singler.get_classic_markers(ref, labels, features)

    averaged = (ref[:, 0:5] + ref[:, [9, 8, 7, 6, 5]]) / 2
    ave_markers = singler.get_classic_markers(
        averaged, ["A", "B", "C", "D", "E"], features
    )
    assert list(markers.keys()) == list(ave_markers.keys())

    for k in markers.keys():
        alpha = markers[k]
        bravo = ave_markers[k]
        assert sorted(alpha.keys()) == sorted(bravo.keys())
        for k2 in alpha.keys():
            assert alpha[k2] == bravo[k2]


def test_get_classic_markers_batch():
    features = [str(i) for i in range(10000)]

    ref1 = numpy.random.rand(10000, 5)
    labels1 = ["A", "B", "C", "D", "E"]
    markers1 = singler.get_classic_markers(ref1, labels1, features, num_de=50)

    ref2 = numpy.random.rand(10000, 3)
    labels2 = ["B", "D", "F"]
    markers2 = singler.get_classic_markers(ref2, labels2, features, num_de=50)

    averaged = (ref1[:, [1, 3]] + ref2[:, [0, 1]]) / 2
    ave_markers = singler.get_classic_markers(averaged, ["B", "D"], features, num_de=50)

    combined_markers = singler.get_classic_markers(
        [ref1, ref2], [labels1, labels2], [features, features], num_de=50
    )

    for k in ["A", "C", "E"]:
        for k2 in ["A", "B", "C", "D", "E"]:
            assert combined_markers[k][k2] == markers1[k][k2]
            assert combined_markers[k2][k] == markers1[k2][k]
        assert len(combined_markers["F"][k]) == 0
        assert len(combined_markers[k]["F"]) == 0

    for k in ["B", "D"]:
        for k2 in ["B", "D"]:
            assert combined_markers[k][k2] == ave_markers[k][k2]
            assert combined_markers[k2][k] == ave_markers[k2][k]
        assert combined_markers[k]["F"] == markers2[k]["F"]
        assert combined_markers["F"][k] == markers2["F"][k]


def test_get_classic_markers_intersected_features():
    features = [str(i) for i in range(10000)]
    labels = ["A", "B", "C", "D", "E"]

    ref1 = numpy.random.rand(8000, 5)
    features1 = features[0:8000]
    ref2 = numpy.random.rand(8000, 5)
    features2 = features[2000:10000]
    combined = singler.get_classic_markers(
        [ref1, ref2], [labels, labels], [features1, features2]
    )

    averaged = (ref1[2000:8000, :] + ref2[0:6000, :]) / 2
    ave_markers = singler.get_classic_markers(averaged, labels, features[2000:8000])

    for k in combined.keys():
        alpha = combined[k]
        bravo = ave_markers[k]
        assert sorted(alpha.keys()) == sorted(bravo.keys())
        for k2 in alpha.keys():
            assert alpha[k2] == bravo[k2]


def test_get_classic_markers_restricted():
    ref = numpy.random.rand(10000, 10)
    labels = ["A", "B", "C", "D", "E", "E", "D", "C", "B", "A"]
    features = [str(i) for i in range(ref.shape[0])]

    keep = range(2000, 8000, 5)
    restricted = [str(i) for i in keep]
    markers = singler.get_classic_markers(ref, labels, features, restrict_to=restricted)

    expected = singler.get_classic_markers(ref[keep,:], labels, restricted)
    assert markers == expected
