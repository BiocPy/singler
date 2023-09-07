import singler
import numpy


def test_classify_single_reference_simple():
    ref = numpy.random.rand(10000, 10)
    labels = ["A", "B", "C", "D", "E", "E", "D", "C", "B", "A"]
    features = [str(i) for i in range(ref.shape[0])]
    markers = singler.get_classic_markers(ref, labels, features)
    built = singler.build_single_reference(ref, labels, features, markers)

    test = numpy.random.rand(10000, 50)
    output = singler.classify_single_reference(test, features, built)
    assert output.shape[0] == 50
    assert sorted(output.column("scores").column_names) == [ "A", "B", "C", "D", "E" ]

    all_names = set(labels)
    for x in output.column("best"):
        assert x in all_names


def test_classify_single_reference_sanity():
    ref = numpy.random.rand(10000, 10) + 1
    ref[:2000,:2] = 0
    ref[2000:4000,2:4] = 0
    ref[4000:6000,4:6] = 0
    ref[6000:8000,6:8] = 0
    ref[8000:,8:] = 0
    labels = ["A", "A", "B", "B", "C", "C", "D", "D", "E", "E"]

    features = [str(i) for i in range(ref.shape[0])]
    markers = singler.get_classic_markers(ref, labels, features)
    built = singler.build_single_reference(ref, labels, features, markers)

    test = numpy.random.rand(10000, 5) + 1
    test[2000:4000,0] = 0 # B
    test[6000:8000,1] = 0 # D
    test[:2000,2] = 0 # A
    test[8000:,3] = 0 # E
    test[4000:6000,4] = 0 # C

    output = singler.classify_single_reference(test, features, built)
    assert output.shape[0] == 5
    assert output.column("best") == [ "B", "D", "A", "E", "C" ]


def test_classify_single_reference_features():
    ref = numpy.random.rand(10000, 10)
    labels = ["A", "A", "B", "B", "C", "C", "D", "D", "E", "E"]
    features = [str(i) for i in range(ref.shape[0])]
    markers = singler.get_classic_markers(ref, labels, features)
    built = singler.build_single_reference(ref, labels, features, markers)

    test = numpy.random.rand(10000, 50)
    revfeatures = features[::-1]
    output = singler.classify_single_reference(test, revfeatures, built)

    revtest = numpy.array(test[::-1,:])
    unscrambled = singler.classify_single_reference(revtest, features, built)
    assert output.column("best") == unscrambled.column("best")
    assert (output.column("delta") == unscrambled.column("delta")).all()
    assert (output.column("scores").column("A") == unscrambled.column("scores").column("A")).all()
