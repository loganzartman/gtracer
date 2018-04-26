#ifndef UTIL_HH
#define UTIL_HH

#include <random>

#if __CUDACC__
#define HOSTDEV __host__ __device__
#else
#define HOSTDEV
#endif

float randf(float lo, float hi);
float mix(float a, float b, float ratio);

#endif
