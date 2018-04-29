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

#define PRIMARY_RAYS 1
#define SKY_COLOR Float3(1)

using namespace std;

void cpu_render(float *pixels, size_t w, size_t h, Mat4f camera,
                Geometry *geom_b, Geometry *geom_e, unsigned iteration,
                unsigned n_threads) {
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
                                    bounds, grid, iteration, pixels};

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

            color += cpu_trace(origin, ray_dir, args.bounds, args.grid, 8);
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

/**
 * @brief Traces a primary ray and produces a color
 *
 * @param ray_orig Ray origin point
 * @param ray_dir Ray direction as a unit vector
 * @param spheres Scene geometry
 * @param depth Maximum trace depth
 * @return Color computed for this primary ray
 */
Float3 cpu_trace(const Float3 &ray_orig, const Float3 &ray_dir,
                 AABB world_bounds, const UniformGrid &grid, int depth) {
    Float3 color = 1.0;
    Float3 light = 0.0;

    Float3 origin = ray_orig;
    Float3 direction = ray_dir;
    for (int i = 0; i < depth; ++i) {
        // cast ray
        Float3 intersection;
        Geometry *hit_geom;
        if (!cpu_ray_intersect(origin, direction, world_bounds, grid,
                               intersection, hit_geom)) {
            light += SKY_COLOR * color;
            break;
        }

        // emissive material
        if (!(hit_geom->material()->emission_color == 0)) {
            light += hit_geom->material()->emission_color * color;
            break;
        }

        color *= hit_geom->material()->surface_color;
        origin = intersection;
        Float3 normal = hit_geom->normal(ray_dir, intersection);

        if (hit_geom->material()->transparency > util::randf(0, 1)) {
            float fresneleffect = raytracing::fresnel(direction, normal, 1.1f);
            if (util::randf(0, 1) < fresneleffect) {
                // reflective material
                direction = direction.reflect(normal);
            } else {
                float refr_i;
                if (direction.dot(normal) > 0)
                    refr_i = 1.1;
                else
                    refr_i = 0.91;
                float angle = normal.dot(direction);
                float k = 1 - refr_i * refr_i * (1 - angle * angle);
                Float3 refraction_dir =
                    direction * refr_i + normal * (refr_i * angle - sqrt(k));
                refraction_dir.normalize();
                direction = refraction_dir;
            }
        } else if (hit_geom->material()->reflection > util::randf(0, 1)) {
            direction = direction.reflect(normal);
        } else {
            // diffuse material
            // generate random number on a sphere, but we want only
            // vectors pointing in the same hemisphere as the normal
            direction = Float3::random_spherical();
            if (direction.dot(normal) < 0)
                direction *= -1;
        }
        direction.normalize();
        origin += normal * 1e-6;
    }

    return light;
}

/**
 * @brief Finds the nearest intersection between a ray and scene geometry.
 * @detail Uses traversal algorithm proposed by Amanatides and Woo (1987).
 *
 * @param[in] ray_orig Ray origin point
 * @param[in] ray_dir Ray direction as unit vector
 * @param[in] geom Scene geometry
 * @param[in] world_bounds an AABB for entire world (use geometry_bounds())
 * @param[in] grid the UniformGrid acceleration structure
 * @param[out] intersection The point of intersection
 * @param[out] hit_geom The geometry that was intersected
 * @return Whether there was an intersection
 */
bool cpu_ray_intersect(const Float3 &ray_orig, const Float3 &ray_dir,
                       AABB world_bounds, const UniformGrid &grid,
                       Float3 &intersection, Geometry *&hit_geom) {
    // find ray entry point into world bounds
    const Geometry bbox(BoxData{world_bounds});
    Float3 ray_entry;
    if (world_bounds.contains(ray_orig)) {
        ray_entry = ray_orig;
    } else {
        float t;
        if (!bbox.intersect(ray_orig, ray_dir, t))
            return false;
        ray_entry = ray_orig + ray_dir * t;
    }

    const Float3 world_size = world_bounds.xmax - world_bounds.xmin;
    Float3 relative_entry = ray_entry - world_bounds.xmin;
    relative_entry = vmax(Float3(0), relative_entry);
    relative_entry = vmin(world_size - 1e-5, relative_entry);  // good tolerance

    // compute voxel parameters
    Int3 voxel_pos(floor(relative_entry.x / (grid.cell_size.x)),
                   floor(relative_entry.y / (grid.cell_size.y)),
                   floor(relative_entry.z / (grid.cell_size.z)));

    const Int3 voxel_step(ray_dir.x < 0 ? -1 : 1, ray_dir.y < 0 ? -1 : 1,
                          ray_dir.z < 0 ? -1 : 1);

    const Float3 next_voxel_bound =
        Float3(voxel_pos + vmax(Int3(0), voxel_step)) * grid.cell_size;

    // compute t values at which ray crosses voxel boundaries
    Float3 t_max = (next_voxel_bound - relative_entry) / ray_dir;
    // compute t deltas
    Float3 t_delta = grid.cell_size / ray_dir * Float3(voxel_step);

    // assert(t_delta.x >= 0 && t_delta.y >= 0 && t_delta.z >= 0);

    // handle div by zero
    if (ray_dir.x == 0)
        t_max.x = INFINITY, t_delta.x = INFINITY;
    if (ray_dir.y == 0)
        t_max.y = INFINITY, t_delta.y = INFINITY;
    if (ray_dir.z == 0)
        t_max.z = INFINITY, t_delta.z = INFINITY;

    assert(voxel_pos.x >= 0 && voxel_pos.y >= 0 && voxel_pos.z >= 0);
    assert(voxel_pos.x < grid.res.x && voxel_pos.y < grid.res.y &&
           voxel_pos.z < grid.res.z);

    // traverse the grid
    unsigned i = 0;
    do {
        // test objects
        auto b = grid.first(voxel_pos);
        auto e = grid.last(voxel_pos);
        if (b != e) {
            if (cpu_ray_intersect_items(ray_orig, ray_dir, b, e, intersection,
                                        hit_geom)) {
                return true;
            }
        }

        if (t_max.x < t_max.y) {
            if (t_max.x < t_max.z) {
                voxel_pos.x += voxel_step.x;
                if (voxel_pos.x >= grid.res.x || voxel_pos.x < 0)
                    return false;
                t_max.x += t_delta.x;
            } else {
                voxel_pos.z += voxel_step.z;
                if (voxel_pos.z >= grid.res.z || voxel_pos.z < 0)
                    return false;
                t_max.z += t_delta.z;
            }
        } else {
            if (t_max.y < t_max.z) {
                voxel_pos.y += voxel_step.y;
                if (voxel_pos.y >= grid.res.y || voxel_pos.y < 0)
                    return false;
                t_max.y += t_delta.y;
            } else {
                voxel_pos.z += voxel_step.z;
                if (voxel_pos.z >= grid.res.z || voxel_pos.z < 0)
                    return false;
                t_max.z += t_delta.z;
            }
        }

        assert(voxel_pos.x >= 0 && voxel_pos.y >= 0 && voxel_pos.z >= 0);
        assert(voxel_pos.x < grid.res.x && voxel_pos.y < grid.res.y &&
               voxel_pos.z < grid.res.z);
    } while (++i < 10000);  // arbitrary traversal length limit

    return false;
}

/**
 * @brief Classic, grid-free brute-force ray intersection
 * @details This is used as a component of cpu_ray_intersect
 * It accepts input iterators and checks intersection for each item.
 *
 * @param[in] ray_orig Ray origin point
 * @param[in] ray_dir Ray direction as unit vector
 * @param[in] geom Scene geometry
 * @param[out] intersection The point of intersection
 * @param[out] hit_geom The geometry that was intersected
 * @return Whether there was an intersection
 */
template <typename II>
bool cpu_ray_intersect_items(const Float3 &ray_orig, const Float3 &ray_dir,
                             II b, II e, Float3 &intersection,
                             Geometry *&hit_geom) {
    float near_t = INFINITY;
    Geometry *near_geom = nullptr;

    while (b != e) {
        Geometry &g = *b;
        float t;
        if (!g.intersect(ray_orig, ray_dir, t)) {
            ++b;
            continue;
        }
        if (t < near_t) {
            near_t = t;
            near_geom = &g;
        }
        ++b;
    }

    if (near_geom) {
        intersection = ray_orig + ray_dir * near_t;
        hit_geom = near_geom;
        return true;
    }
    return false;
}

// unfortunate workaround to force template instantiation for tests
template bool cpu_ray_intersect_items<vector<Geometry>::iterator>(
    const Float3 &, const Float3 &, vector<Geometry>::iterator,
    vector<Geometry>::iterator, Float3 &, Geometry *&);
