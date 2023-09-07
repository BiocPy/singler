import singler
import numpy


def test_build_single_reference():
    ref = numpy.random.rand(10000, 10)
    labels = ["A", "B", "C", "D", "E", "E", "D", "C", "B", "A"]
    features = [str(i) for i in range(ref.shape[0])]
    markers = singler.get_classic_markers(ref, labels, features)

    built = singler.build_single_reference(ref, labels, features, markers)
    assert built.num_labels() == 5
    assert built.num_markers() < len(features)
    assert built.features == features
    assert built.labels == [ "A", "B", "C", "D", "E" ]

    all_markers = built.marker_subset()
    assert len(all_markers) == built.num_markers()
    feat_set = set(features)
    for m in all_markers:
        assert m in feat_set

    # Same results when run in parallel.
    pbuilt = singler.build_single_reference(ref, labels, features, markers, num_threads = 2)
    assert all_markers == pbuilt.marker_subset()
