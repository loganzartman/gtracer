#ifndef CUDA_UTIL_CUH
#define CUDA_UTIL_CUH
#include "cuda_util.hh"
#include <cstdio>

#define cudachk(ans) \
    { cuda_assert((ans), __FILE__, __LINE__); }
inline static void cuda_assert(cudaError_t code, const char *file, int line,
                        bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Assert: %s %s %d\n", cudaGetErrorString(code),
                file, line);
        if (abort)
            exit(code);
    }
}

#endif