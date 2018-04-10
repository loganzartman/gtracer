#ifndef RENDER_HH
#define RENDER_HH

#include <vector>
#include "Mat.hh"
#include "Sphere.hh"

float3 cpu_trace(const float3 &ray_orig, const float3 &ray_dir,
                 std::vector<Sphere> spheres, int depth);
void cpu_render(float *pixels, size_t w, size_t h, Mat4f camera,
                std::vector<Sphere> spheres);
bool cpu_ray_intersect(const float3 &ray_orig, const float3 &ray_dir,
                       std::vector<Sphere> &spheres, float3 &intersection,
                       Sphere *&hit_sphere);

#endif
