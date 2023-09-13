#include "utils.h" // must be before all other includes.

#include <cstdint>

//[[export]]
void* build_integrated_references(
    int32_t test_nrow,
    const int32_t* /** numpy */ test_features,
    int32_t nrefs,
    const uintptr_t* /** void_p */ references,
    const uintptr_t* /** void_p */ labels,
    const uintptr_t* /** void_p */ ref_ids,
    const uintptr_t* /** void_p */ prebuilt,
    int32_t nthreads)
{
    singlepp::IntegratedBuilder runner;
    runner.set_num_threads(nthreads);

    for (int32_t r = 0; r < nrefs; ++r) {
        const auto& ptr = reinterpret_cast<const Mattress*>(references[r])->ptr;
        runner.add(
            test_nrow,
            test_features,
            ptr.get(),
            reinterpret_cast<const int32_t*>(ref_ids[r]),
            reinterpret_cast<const int32_t*>(labels[r]),
            *reinterpret_cast<const singlepp::BasicBuilder::Prebuilt*>(labels[r])
        );
    }

    auto o = runner.finish();
    return new decltype(o)(std::move(o));
}

//[[export]]
void free_integrated_references(void* ptr) {
    delete reinterpret_cast<singlepp::IntegratedReferences*>(ptr);
}

//[[export]]
int32_t get_integrated_references_num_references(void* ptr) {
    return reinterpret_cast<singlepp::IntegratedReferences*>(ptr)->num_references();
}

//[[export]]
int32_t get_integrated_references_num_labels(void* ptr, int32_t r) {
    return reinterpret_cast<singlepp::IntegratedReferences*>(ptr)->num_labels(r);
}

//[[export]]
int32_t get_integrated_references_num_profiles(void* ptr, int32_t r) {
    return reinterpret_cast<singlepp::IntegratedReferences*>(ptr)->num_profiles(r);
}
