#ifndef UTIL_HH
#define UTIL_HH
#include <random>
#include "cuda_util.hh"

#if __CUDACC__
#define HOSTDEV __host__ __device__
#else
#define HOSTDEV
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

    HOSTDEV static float mix(float a, float b, float ratio) { return a * ratio + b * (1 - ratio); }

    static float randf(float lo, float hi) {
        using namespace std;
        static random_device rd;
        static mt19937 mt(rd());
        return (float)mt() / mt.max() * (hi - lo) + lo;
    }

    /**
     * @brief Allocate bytes either on CPU or GPU managed memory
     */
    static void* hostdev_alloc(size_t bytes, bool gpu) {
        if (gpu) {
            void* mem;
            cuda_malloc_managed(&mem, bytes);
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
            delete[] ((char*)ptr);
    }
}

#endif
