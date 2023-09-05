#include "utils.h" // must be before all other includes.

#include <vector>
#include <cstdint>

//[[export]]
void grouped_medians(const void* mat, const int32_t* labels /** numpy */, int32_t num_labels, double* output /** numpy */, int32_t nthreads) {
    auto ptr = reinterpret_cast<const Mattress*>(mat)->ptr.get();

    std::vector<int> label_sizes(num_labels);
    for (size_t i = 0, end = ptr->ncol(); i < end; ++i) {
        ++(label_sizes[labels[i]]); 
    }

    tatami::row_medians_by_group(
        ptr,
        labels,
        label_sizes,
        output,
        nthreads
    );
}

