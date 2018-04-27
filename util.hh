#ifndef UTIL_HH
#define UTIL_HH
#include <random>

#if __CUDACC__
#define HOSTDEV __host__ __device__
#else
#define HOSTDEV
#endif

#define cudachk(ans) \
    { cuda_assert((ans), __FILE__, __LINE__); }

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
}

#endif
