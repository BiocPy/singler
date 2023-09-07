#include "utils.h" // must be before raticate, singlepp includes.

#include <vector>
#include <cstdint>
#include <algorithm>

//[[export]]
void classify_single_reference(
    void* mat, 
    const int32_t* subset /** numpy */, 
    void* prebuilt, 
    double quantile, 
    uint8_t use_fine_tune, 
    double fine_tune_threshold, 
    int32_t nthreads,
    const uintptr_t* scores /** void_p */,
    int32_t* best /** numpy */,
    double* delta /** numpy */)
{
    auto mptr = reinterpret_cast<const Mattress*>(mat);
    auto bptr = reinterpret_cast<const singlepp::BasicBuilder::Prebuilt*>(prebuilt);

    singlepp::BasicScorer runner;
    runner.set_num_threads(nthreads);
    runner.set_quantile(quantile);
    runner.set_fine_tune(use_fine_tune);
    runner.set_fine_tune_threshold(fine_tune_threshold);

    // Only necessary as BasicScorer::run() isn't templated yet.
    std::vector<int> subset_copy(subset, subset + bptr->subset.size());
    size_t NC = mptr->ptr->ncol();
    std::vector<int> best_copy(NC);

    size_t nlabels = bptr->num_labels();
    std::vector<double*> score_ptrs;
    score_ptrs.reserve(nlabels);
    for (size_t l = 0; l < nlabels; ++l) {
        score_ptrs.push_back(reinterpret_cast<double*>(scores[l]));
    }

    runner.run(
        mptr->ptr.get(), 
        *bptr, 
        subset_copy.data(),
        best_copy.data(),
        score_ptrs,
        delta
    );

    std::copy(best_copy.begin(), best_copy.end(), best);
    return;
}
