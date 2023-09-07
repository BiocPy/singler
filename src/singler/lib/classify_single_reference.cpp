#include "utils.h" // must be before raticate, singlepp includes.

#include <vector>
#include <cstdint>

//[[export]]
void classify_single_reference(
    void* mat, 
    const int32_t* subset, 
    void* prebuilt, 
    double quantile, 
    uint8_t use_fine_tune, 
    double fine_tune_threshold, 
    int32_t nthreads,
    double* scores,
    int32_t* best,
    double* delta)
{
    auto parsed = Rtatami::BoundNumericPointer(test);
    PrebuiltXPtr built(prebuilt);

    // Setting up outputs.
    size_t ncells = parsed->ptr->ncol();
    Rcpp::IntegerVector best(ncells);
    Rcpp::NumericVector delta(ncells);

    size_t nlabels = built->num_labels();
    Rcpp::NumericMatrix scores(ncells, nlabels);
    std::vector<double*> scores_ptr(nlabels);
    if (nlabels) {
        scores_ptr[0] = static_cast<double*>(scores.begin());
        for (size_t l = 1; l < nlabels; ++l) {
            scores_ptr[l] = scores_ptr[l-1] + ncells;
        }
    }

    // Running the analysis.
    singlepp::BasicScorer runner;
    runner.set_num_threads(nthreads);
    runner.set_quantile(quantile).set_fine_tune(use_fine_tune).set_fine_tune_threshold(fine_tune_threshold);
    runner.run(
        parsed->ptr.get(), 
        *built, 
        static_cast<const int*>(subset.begin()),
        static_cast<int*>(best.begin()),
        scores_ptr,
        static_cast<double*>(delta.begin())
    );

    return Rcpp::List::create(
        Rcpp::Named("best") = best,
        Rcpp::Named("scores") = scores,
        Rcpp::Named("delta") = delta
    );
}
