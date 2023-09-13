#include "utils.h" // must be before raticate, singlepp includes.

#include <vector>
#include <cstdint>
#include <algorithm>

//[[export]]
void classify_integrated_references(
    void* mat,
    const uintptr_t* assigned /** void_p */,
    void* prebuilt,
    double quantile,
    uintptr_t* scores /** void_p */,
    int32_t* best /** numpy */,
    double* delta /** numpy */,
    int32_t nthreads)
{
    auto mptr = reinterpret_cast<const Mattress*>(mat);
    auto NC = mptr->ptr->ncol();
    auto bptr = reinterpret_cast<const singlepp::IntegratedReferences*>(prebuilt);

    // Only necessary as IntegratedScorer::run() isn't templated yet.
    size_t nrefs = bptr->num_references();
    std::vector<std::vector<int> > single_results;
    std::vector<const int*> single_results_ptr;
    single_results.reserve(nrefs);
    single_results_ptr.reserve(nrefs);

    for (size_t r = 0; r < nrefs; ++r) {
        auto current = reinterpret_cast<const int32_t*>(assigned[r]);
        single_results.emplace_back(current, current + NC);
        single_results_ptr.emplace_back(single_results.back().data());
    }

    singlepp::IntegratedScorer runner;
    runner.set_num_threads(nthreads);
    runner.set_quantile(quantile);

    std::vector<int> best_copy(NC);
    std::vector<double*> score_ptrs(nrefs);
    for (size_t r = 0; r < nrefs; ++r) {
        score_ptrs[r] = reinterpret_cast<double*>(scores[r]);
    }

    runner.run(
        mptr->ptr.get(),
        single_results_ptr,
        *bptr,
        best_copy.data(),
        score_ptrs,
        delta
    );

    std::copy(best_copy.begin(), best_copy.end(), best);
    return;
}
