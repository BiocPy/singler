#include "utils.h" // must be before all other includes.

#include <vector>
#include <cstdint>

//[[export]]
void* find_classic_markers(int32_t nref, const uintptr_t* labels /** void_p */, const uintptr_t* ref /** void_p */, int32_t de_n, int32_t nthreads) {
    std::vector<const tatami::Matrix<double, int>*> ref_ptrs;
    ref_ptrs.reserve(nref);
    std::vector<const int32_t*> lab_ptrs;
    lab_ptrs.reserve(nref);

    for (int32_t r = 0; r < nref; ++r) {
        const auto& ptr = reinterpret_cast<const Mattress*>(ref[r])->ptr;
        ref_ptrs.push_back(ptr.get());
        lab_ptrs.push_back(reinterpret_cast<const int32_t*>(labels[r]));
    }

    singlepp::ChooseClassicMarkers mrk;
    mrk.set_number(de_n).set_num_threads(nthreads);
    auto store = mrk.run(ref_ptrs, lab_ptrs);
    return new singlepp::Markers(std::move(store));
}

//[[export]]
int32_t number_of_classic_markers(int32_t num_labels):
    return singlepp::ChooseClassicMarkers::number_of_markers(num_labels);
}
