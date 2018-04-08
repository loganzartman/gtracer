#ifndef RENDER_HH
#define RENDER_HH

#include "Sphere.hh"

float3 cpu_trace(const float3 &ray_orig, const float3 &ray_dir, Sphere *spheres,
                 size_t num_spheres, int depth);
void cpu_render(float *pixels, size_t w, size_t h, Sphere *spheres,
                size_t num_spheres);

#endif
