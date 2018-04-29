#include "cuda_util.cuh"

/**
 * @brief Wrapper around cudaMallocManaged available to non-Cuda code.
 */
void cuda_malloc_managed(void*& ptr, size_t bytes) {
    cudachk(cudaMallocManaged(&ptr, bytes));
}

/**
 * @brief Wrapper around cudaFree available to non-Cuda code.
 */
void cuda_free(void* ptr) { cudachk(cudaFree(ptr)); }
