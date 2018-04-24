#ifndef VEC3_HH
#define VEC3_HH

#if __CUDACC__
#include "Vec3_cuda.cuh"
#else
#include "Vec3_cpu.hh"
#endif

#endif