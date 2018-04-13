#ifndef RENDER_HH
#define RENDER_HH

#include <vector>
#include "Mat.hh"
#include "Sphere.hh"

#define randf(a, b) ((float)rand() / RAND_MAX * (b - a) + a)

float3 cpu_trace(const float3 &ray_orig, const float3 &ray_dir,
                 std::vector<Sphere> spheres, int depth);
void cpu_render(float *pixels, size_t w, size_t h, Mat4f camera,
                std::vector<Sphere> spheres, unsigned iteration);
bool cpu_ray_intersect(const float3 &ray_orig, const float3 &ray_dir,
                       std::vector<Sphere> &spheres, float3 &intersection,
                       Sphere *&hit_sphere);

#endif
