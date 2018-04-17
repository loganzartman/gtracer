#ifndef RENDER_HH
#define RENDER_HH

#include <cstddef>
#include <vector>
#include "Geometry.hh"
#include "Mat.hh"

struct CPUThreadArgs {
    size_t w;
    size_t h;
    size_t pitch;
    size_t offset;
    Mat4f &camera;
    std::vector<Geometry *> &geom;
    unsigned iteration;
    float *pixels;
};

float3 cpu_trace(const float3 &ray_orig, const float3 &ray_dir,
                 std::vector<Geometry *> geom, int depth);
void cpu_render(float *pixels, size_t w, size_t h, Mat4f camera,
                std::vector<Geometry *> geom, unsigned iteration,
                unsigned n_threads);
void *cpu_render_thread(void *thread_arg);
bool cpu_ray_intersect(const float3 &ray_orig, const float3 &ray_dir,
                       std::vector<Geometry *> &geom, float3 &intersection,
                       Geometry *&hit_geom);
float fresnel(float3 dir, float3 normal, float ior);

#endif
