#ifndef CUDA_UTIL_HH
#define CUDA_UTIL_HH
#include <cstddef>

void cuda_malloc_managed(void** ptr, size_t bytes);
void cuda_free(void* ptr);

#endif