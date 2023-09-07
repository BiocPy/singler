#include "utils.h" // must be before all other includes.

#include <vector>
#include <cstdint>
#include <algorithm>

//[[export]]
void* train_single_reference(void* ref, const int32_t* labels /** numpy */, void* markers, uint8_t approximate, int32_t nthreads) {
    singlepp::BasicBuilder builder;
    builder.set_num_threads(nthreads);
    builder.set_top(-1); // Use all available markers; assume subsetting was applied on the Python side.
    builder.set_approximate(approximate);

    auto marker_ptr = reinterpret_cast<const singlepp::Markers*>(markers);
    const auto& ptr = reinterpret_casts<const tatami::Mattress*>(ref)->ptr;
    std::vector<int> labels2(labels, labels + ptr->ncol()); // need to copy as int may not be int32 and singlepp isn't templated on the labels (for now).
    auto built = builder.run(ptr.get(), labels2.data(), *marker_ptr);

    return new singlepp::BasicBuilder::Prebuilt(std::move(built));
}

//[[export]]
int32_t get_nsubset_from_single_reference(void* ptr) {
    return reinterpret_cast<const singlpp::BasicBuilder::Prebuilt*>(ptr)->subset.size();
}

//[[export]]
int32_t get_nlabels_from_single_reference(void* ptr) {
    return reinterpret_cast<const singlpp::BasicBuilder::Prebuilt*>(ptr)->num_labels();
}

//[[export]]
void get_subset_from_single_reference(void* ptr, int32_t* buffer /** numpy */) {
    const auto& sub = reinterpret_cast<const singlpp::BasicBuilder::Prebuilt*>(ptr)->subset;
    std::copy(sub.begin(), sub.end(), buffer);
}

//[[export]]
void free_single_reference(void* ptr) {
    delete reinterpret_cast<singlepp::BasicBuilder::Prebuilt>(pr);
}
