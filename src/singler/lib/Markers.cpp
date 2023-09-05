#include "utils.h"

#include <algorithm>

//[[export]]
void * create_markers(int32_t nlabels) {
    auto ptr = new singlepp::Markers(nlabels);
    auto& mrk = *ptr;
    for (int32_t l = 0; l < nlabels; ++l) {
        mrk[l].resize(nlabels);
    }
    return ptr;
}

//[[export]]
void free_markers(void * ptr) {
    delete reinterpret_cast<singlepp::Markers*>(ptr);
}

//[[export]]
int32_t get_nlabels_from_markers(void* ptr) {
    return reinterpret_cast<singlepp::Markers*>(ptr)->size();
}

//[[export]]
int32_t get_nmarkers_for_pair(void * ptr, int32_t label1, int32_t label2) {
    const auto& current = (*reinterpret_cast<singlepp::Markers*>(ptr))[label1][label2];
    return current.size();
}

//[[export]]
void get_markers_for_pair(void * ptr, int32_t label1, int32_t label2, int32_t* buffer /** numpy */) {
    const auto& current = (*reinterpret_cast<singlepp::Markers*>(ptr))[label1][label2];
    std::copy(current.begin(), current.end(), buffer);
}

//[[export]]
void set_markers_for_pair(void* ptr, int32_t label1, int32_t label2, int32_t n, const int32_t* values /** numpy */) {
    auto& current = (*reinterpret_cast<singlepp::Markers*>(ptr))[label1][label2];
    current.clear();
    current.insert(current.end(), values, values + n);
}
