#ifndef RENDER_HH
#define RENDER_HH

#include <vector>
#include "Sphere.hh"

float3 cpu_trace(const float3 &ray_orig, const float3 &ray_dir,
                 std::vector<Sphere> spheres, int depth);
void cpu_render(float *pixels, size_t w, size_t h, std::vector<Sphere> spheres);

#endif
