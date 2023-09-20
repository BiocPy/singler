import singler
import numpy


def test_annotate_integrated():
    all_features = [str(i) for i in range(10000)]

    ref1 = numpy.random.rand(8000, 10)
    labels1 = ["A", "B", "C", "D", "E", "E", "D", "C", "B", "A"]
    features1 = [all_features[i] for i in range(8000)]

    ref2 = numpy.random.rand(8000, 6)
    labels2 = ["z", "y", "x", "z", "y", "z"]
    features2 = [all_features[i] for i in range(2000, 10000)]

    test_features = [all_features[i] for i in range(0, 10000, 2)]
    test = numpy.random.rand(len(test_features), 50)

    single_results, integrated_results = singler.annotate_integrated(
        test,
        test_features=test_features,
        ref_data_list=[ref1, ref2],
        ref_labels_list=[labels1, labels2],
        ref_features_list=[features1, features2],
    )

    assert len(single_results) == 2
    assert set(single_results[0].column("best")) == set(labels1)
    assert set(single_results[1].column("best")) == set(labels2)
    assert set(integrated_results.column("best_reference")) == set([0, 1])
