#ifndef RENDER_HH
#define RENDER_HH

#include "Sphere.hh"

float3 trace (const float3 &ray_orig, const float3 &ray_dir, Sphere *spheres, size_t num_spheres, int depth);
float3 *cpu_render (Sphere *spheres, size_t num_spheres);

#endif
