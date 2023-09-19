import singler
import numpy


def test_build_integrated_references():
    all_features = [str(i) for i in range(10000)]

    ref1 = numpy.random.rand(8000, 10)
    labels1 = ["A", "B", "C", "D", "E", "E", "D", "C", "B", "A"]
    features1 = [all_features[i] for i in range(8000)]
    built1 = singler.build_single_reference(ref1, labels1, features1)

    ref2 = numpy.random.rand(8000, 6)
    labels2 = ["z", "y", "x", "z", "y", "z"]
    features2 = [all_features[i] for i in range(2000, 10000)]
    built2 = singler.build_single_reference(ref2, labels2, features2)

    test_features = [all_features[i] for i in range(0, 10000, 2)]
    integrated = singler.build_integrated_references(
        test_features,
        ref_data_list=[ref1, ref2],
        ref_labels_list=[labels1, labels2],
        ref_features_list=[features1, features2],
        ref_prebuilt_list=[built1, built2],
    )

    assert integrated.reference_names == None
    assert integrated.reference_labels == [["A", "B", "C", "D", "E"], ["z", "y", "x"]]
    assert integrated.test_features == test_features

    # Works in parallel.
    pintegrated = singler.build_integrated_references(
        test_features,
        ref_data_list=[ref1, ref2],
        ref_labels_list=[labels1, labels2],
        ref_features_list=[features1, features2],
        ref_prebuilt_list=[built1, built2],
        ref_names=["FOO", "BAR"],
        num_threads=3,
    )

    assert pintegrated.reference_names == ["FOO", "BAR"]
    assert pintegrated.reference_labels == integrated.reference_labels
    assert pintegrated.test_features == test_features
