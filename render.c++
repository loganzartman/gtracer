#include "render.hh"
#include "Box.hh"
#include "Mat.hh"
#include "Material.hh"
#include "Sphere.hh"
#include "UniformGrid.hh"
#include "Vec3.hh"
#include "raytracing.hh"
#include "transform.hh"
#include "util.hh"

#include <pthread.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>

using namespace std;

void cpu_render(float *pixels, size_t w, size_t h, Mat4f camera,
                Geometry *geom_b, Geometry *geom_e, unsigned iteration,
                unsigned n_threads, bool accel) {
    // construct uniform grid
    AABB bounds = geometry_bounds(geom_b, geom_e);
    Int3 res = UniformGrid::resolution(bounds, geom_e - geom_b);
    size_t n_data = UniformGrid::data_size(res);
    size_t n_pairs = UniformGrid::count_pairs(res, bounds, geom_b, geom_e);
    ugrid_data_t *grid_data = new ugrid_data_t[n_data];
    ugrid_pair_t *grid_pairs = new ugrid_pair_t[n_pairs];
    UniformGrid grid(res, bounds, grid_data, grid_pairs, n_pairs, geom_b,
                     geom_e);

    // spawn threads
    CPUThreadArgs **args = new CPUThreadArgs *[n_threads];
    pthread_t *threads = new pthread_t[n_threads];
    for (unsigned i = 0; i < n_threads; ++i) {
        const unsigned pitch = n_threads;
        const unsigned offset = i;
        args[i] = new CPUThreadArgs{w,      h,    pitch,     offset, camera,
                                    bounds, grid, iteration, pixels, accel};

        if (n_threads > 1) {
            pthread_create(&threads[i], NULL, cpu_render_thread, args[i]);
        } else {
            cpu_render_thread(static_cast<void *>(args[i]));
        }
    }

    // wait for threads
    if (n_threads > 1) {
        for (unsigned i = 0; i < n_threads; ++i) {
            pthread_join(threads[i], NULL);
            delete args[i];
        }
    }

    delete[] args;
    delete[] threads;
    delete[] grid_data;
    delete[] grid_pairs;
}

void *cpu_render_thread(void *thread_arg) {
    CPUThreadArgs &args = *static_cast<CPUThreadArgs *>(thread_arg);

    float inv_w = 1 / float(args.w);
    float inv_h = 1 / float(args.h);
    float fov = 30;
    float aspect_ratio = float(args.w) / float(args.h);
    float angle = tan(0.5 * M_PI * fov / 180.0);

    Mat4f dir_camera = transform_clear_translate(args.camera);
    Float3 origin = args.camera * Float3();

    const size_t len = args.w * args.h;
    for (size_t p = args.offset; p < len; p += args.pitch) {
        // compute position
        const size_t x = p % args.w;
        const size_t y = p / args.w;

        // do raytracing
        Float3 color;
        for (size_t i = 0; i < PRIMARY_RAYS; ++i) {
            //  compute the x and y magnitude of each vector
            float v_x = (2 * ((x + util::randf(0, 1)) * inv_w) - 1) * angle *
                        aspect_ratio;
            float v_y = (1 - 2 * ((y + util::randf(0, 1)) * inv_h)) * angle;
            Float3 ray_dir = dir_camera * Float3(v_x, v_y, -1);
            ray_dir.normalize();

            color += raytracing::trace(origin, ray_dir, args.bounds, args.grid, args.accel, 8);
        }
        color *= 1.f / PRIMARY_RAYS;

        // compute all-time average color
        const size_t idx = p * 4;
        Float3 dst = Float3(args.pixels[idx], args.pixels[idx + 1],
                            args.pixels[idx + 2]);
        float f = 1;
        if (args.iteration > 0)
            f = 1.f / args.iteration;
        Float3 blended = color * f + dst * (1 - f);

        // write color
        args.pixels[idx] = blended.x;
        args.pixels[idx + 1] = blended.y;
        args.pixels[idx + 2] = blended.z;
        args.pixels[idx + 3] = 1;  // alpha
    }

    return nullptr;
}