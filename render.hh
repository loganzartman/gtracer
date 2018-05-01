#ifndef RENDER_HH
#define RENDER_HH

#include <cstddef>
#include <vector>
#include "AABB.hh"
#include "Geometry.hh"
#include "Mat.hh"
#include "UniformGrid.hh"

struct CPUThreadArgs {
    size_t w;
    size_t h;
    size_t pitch;
    size_t offset;
    Mat4f &camera;
    AABB bounds;
    const UniformGrid &grid;
    bool accel;
    unsigned iteration;
    float *pixels;
    float *display_pixels;
};

Float3 cpu_trace(const Float3 &ray_orig, const Float3 &ray_dir,
                 AABB world_bounds, const UniformGrid &grid, bool accel,
                 int depth);

void cpu_render(float *pixels, float *display_pixels, size_t w, size_t h,
                Mat4f camera, Geometry *geom_b, Geometry *geom_e,
                unsigned iteration, unsigned n_threads, bool accel);
void *cpu_render_thread(void *thread_arg);
bool cpu_ray_intersect(const Float3 &ray_orig, const Float3 &ray_dir,
                       AABB world_bounds, const UniformGrid &grid,
                       Float3 &intersection, Geometry *&hit_geom);
template <typename II>
bool cpu_ray_intersect_items(const Float3 &ray_orig, const Float3 &ray_dir,
                             II b, II e, Float3 &intersection,
                             Geometry *&hit_geom);

#endif
