#ifndef UTIL_HH
#define UTIL_HH
#include "cuda_util.hh"

#if __CUDACC__
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_real_distribution.h>
#define HOSTDEV __host__ __device__
#define DEVICE __device__
#else
#include <random>
#define HOSTDEV
#define DEVICE
#endif

namespace util {
template <typename T>
HOSTDEV static T min(T a, T b) {
    return a < b ? a : b;
}

template <typename T>
HOSTDEV static T max(T a, T b) {
    return a < b ? b : a;
}

template <typename T>
HOSTDEV void swap(T& t1, T& t2) {
    T tmp(t1);
    t1 = t2;
    t2 = tmp;
}

/*
HOSTDEV static float mix(float a, float b, float ratio) {
    return a * ratio + b * (1 - ratio);
}
*/

#if __CUDACC__
__device__ unsigned _rand_n = 0;
__device__ static float randf(float lo, float hi) {
    thrust::minstd_rand rng;
    thrust::uniform_real_distribution<float> dist(lo, hi);
    atomicAdd(&util::_rand_n, 1l);
    rng.discard(util::_rand_n);
    return dist(rng);
}
#else
static float randf(float lo, float hi) {
    using namespace std;
    static random_device rd;
    static mt19937 mt(rd());
    return (float)mt() / mt.max() * (hi - lo) + lo;
}
#endif

/**
 * @brief Allocate bytes either on CPU or GPU managed memory
 */
static void* hostdev_alloc(size_t bytes, bool gpu) {
    if (gpu) {
        void* mem;
        cuda_malloc_managed(mem, bytes);
        return mem;
    }
    return (void*)new char[bytes];
}

/**
 * @brief Free bytes either on CPU or GPU managed memory
 */
static void hostdev_free(void* ptr, bool gpu) {
    if (gpu)
        cuda_free(ptr);
    else
        delete[]((char*)ptr);
}
}  // namespace util

#endif
