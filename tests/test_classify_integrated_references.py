import singler
import numpy


def test_classify_integrated_references():
    all_features = [str(i) for i in range(10000)]
    test_features = [all_features[i] for i in range(0, 10000, 2)]
    test_set = set(test_features)

    ref1 = numpy.random.rand(8000, 10)
    labels1 = ["A", "B", "C", "D", "E", "E", "D", "C", "B", "A"]
    features1 = [all_features[i] for i in range(8000)]
    built1 = singler.build_single_reference(ref1, labels1, features1, restrict_to=test_set)

    ref2 = numpy.random.rand(8000, 6)
    labels2 = ["z", "y", "x", "z", "y", "z"]
    features2 = [all_features[i] for i in range(2000, 10000)]
    built2 = singler.build_single_reference(ref2, labels2, features2, restrict_to=test_set)

    integrated = singler.build_integrated_references(
        test_features,
        ref_data_list=[ref1, ref2],
        ref_labels_list=[labels1, labels2],
        ref_features_list=[features1, features2],
        ref_prebuilt_list=[built1, built2],
        ref_names = [ "first", "second" ]
    )

    # Running the full analysis.
    test = numpy.random.rand(len(test_features), 50)
    results1 = singler.classify_single_reference(
        test, 
        test_features,
        built1
    )
    results2 = singler.classify_single_reference(
        test, 
        test_features,
        built2
    )

    results = singler.classify_integrated_references(
        test,
        results = [results1, results2.column("best")],
        integrated_prebuilt = integrated
    )
    assert results.shape[0] == 50
    assert set(results.column("best")) == set([ "first", "second" ])
    assert results.column("scores").has_column("first")
