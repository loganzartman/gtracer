#include "render.hh"
#include "Box.hh"
#include "Mat.hh"
#include "Material.hh"
#include "Sphere.hh"
#include "UniformGrid.hh"
#include "Vec3.hh"
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

using namespace std;

void cpu_render(float *pixels, size_t w, size_t h, Mat4f camera,
                vector<Geometry *> geom, unsigned iteration,
                unsigned n_threads) {
    // construct uniform grid
    AABB bounds = geometry_bounds(geom.begin(), geom.end());
    int3 res = UniformGrid::resolution(bounds, geom.size());
    size_t n_data = UniformGrid::data_size(res);
    size_t n_pairs =
        UniformGrid::count_pairs(res, bounds, geom.begin(), geom.end());
    ugrid_data_t *grid_data = new ugrid_data_t[n_data];
    ugrid_pair_t *grid_pairs = new ugrid_pair_t[n_pairs];
    UniformGrid grid(res, bounds, grid_data, grid_pairs, n_pairs, geom.begin(),
                     geom.end());

    // spawn threads
    CPUThreadArgs **args = new CPUThreadArgs *[n_threads];
    pthread_t *threads = new pthread_t[n_threads];
    for (unsigned i = 0; i < n_threads; ++i) {
        const unsigned pitch = n_threads;
        const unsigned offset = i;
        args[i] = new CPUThreadArgs{w,    h,      pitch, offset,    camera,
                                    geom, bounds, grid,  iteration, pixels};

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
    float3 origin = args.camera * float3();

    const size_t len = args.w * args.h;
    for (size_t p = args.offset; p < len; p += args.pitch) {
        // compute position
        const size_t x = p % args.w;
        const size_t y = p / args.w;

        // do raytracing
        float3 color;
        for (size_t i = 0; i < PRIMARY_RAYS; ++i) {
            //  compute the x and y magnitude of each vector
            float v_x =
                (2 * ((x + randf(0, 1)) * inv_w) - 1) * angle * aspect_ratio;
            float v_y = (1 - 2 * ((y + randf(0, 1)) * inv_h)) * angle;
            float3 ray_dir = dir_camera * float3(v_x, v_y, -1);
            ray_dir.normalize();

            color += cpu_trace(origin, ray_dir, args.geom, args.bounds,
                               args.grid, 8);
        }
        color *= 1.f / PRIMARY_RAYS;

        // compute all-time average color
        const size_t idx = p * 4;
        float3 dst = float3(args.pixels[idx], args.pixels[idx + 1],
                            args.pixels[idx + 2]);
        float f = 1;
        if (args.iteration > 0)
            f = 1.f / args.iteration;
        float3 blended = color * f + dst * (1 - f);

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
float3 cpu_trace(const float3 &ray_orig, const float3 &ray_dir,
                 vector<Geometry *> geom, AABB world_bounds,
                 const UniformGrid &grid, int depth) {
    float3 color = 1.0;
    float3 light = 0.0;

    float3 origin = ray_orig;
    float3 direction = ray_dir;
    for (int i = 0; i < depth; ++i) {
        // cast ray
        float3 intersection;
        Geometry *hit_geom;
        if (!cpu_ray_intersect(origin, direction, world_bounds, grid,
                               intersection, hit_geom)) {
            light += float3(1) * color;
            break;
        }

        // emissive material
        if (!(hit_geom->material()->emission_color == 0)) {
            light += hit_geom->material()->emission_color * color;
            break;
        }

        color *= hit_geom->material()->surface_color;
        origin = intersection;
        float3 normal = hit_geom->normal(ray_dir, intersection);

        if (hit_geom->material()->transparency > randf(0, 1)) {
            float fresneleffect = fresnel(direction, normal, 1.1f);
            if (randf(0, 1) < fresneleffect) {
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
                float3 refraction_dir =
                    direction * refr_i + normal * (refr_i * angle - sqrt(k));
                refraction_dir.normalize();
                direction = refraction_dir;
            }
        } else if (hit_geom->material()->reflection > randf(0, 1)) {
            direction = direction.reflect(normal);
        } else {
            // diffuse material
            // generate random number on a sphere, but we want only
            // vectors pointing in the same hemisphere as the normal
            direction = float3::random_spherical();
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
bool cpu_ray_intersect(const float3 &ray_orig, const float3 &ray_dir,
                       AABB world_bounds, const UniformGrid &grid,
                       float3 &intersection, Geometry *&hit_geom) {
    // find ray entry point into world bounds
    const Box bbox(world_bounds);
    float3 ray_entry;
    if (world_bounds.contains(ray_orig))
        ray_entry = ray_orig;
    else {
        float t;
        if (!bbox.intersect(ray_orig, ray_dir, t))
            return false;
        ray_entry = ray_orig + ray_dir * t;
    }

    // compute voxel parameters
    const float3 relative_entry = ray_entry - world_bounds.xmin;
    int3 voxel_pos(floor(relative_entry.x / (grid.cell_size.x + 1)),
                   floor(relative_entry.y / (grid.cell_size.y + 1)),
                   floor(relative_entry.z / (grid.cell_size.z + 1)));
    const int3 voxel_step(ray_dir.x < 0 ? -1 : 1, ray_dir.y < 0 ? -1 : 1,
                          ray_dir.z < 0 ? -1 : 1);
    const float3 next_voxel_bound = (voxel_pos + voxel_step) * grid.cell_size;

    // compute t values at which ray crosses voxel boundaries
    float3 t_max = (next_voxel_bound - relative_entry) / ray_dir;
    // compute t deltas
    float3 t_delta = grid.cell_size / ray_dir * float3(voxel_step);

    // handle div by zero
    if (ray_dir.x == 0)
        t_max.x = INFINITY, t_delta.x = INFINITY;
    if (ray_dir.y == 0)
        t_max.y = INFINITY, t_delta.y = INFINITY;
    if (ray_dir.z == 0)
        t_max.z = INFINITY, t_delta.z = INFINITY;

    // traverse the grid
    unsigned i = 0;
    do {
        if (fabs(t_max.x) < fabs(t_max.y)) {
            if (fabs(t_max.x) < fabs(t_max.z)) {
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
            if (fabs(t_max.y) < fabs(t_max.z)) {
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

        // test objects
        auto b = grid.first(voxel_pos);
        auto e = grid.last(voxel_pos);
        if (b == e)
            continue;
        if (cpu_ray_intersect_items(ray_orig, ray_dir, b, e, intersection,
                                    hit_geom)) {
            return true;
        }
    } while (++i < 10000);

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
bool cpu_ray_intersect_items(const float3 &ray_orig, const float3 &ray_dir,
                             II b, II e, float3 &intersection,
                             Geometry *&hit_geom) {
    float near_t = INFINITY;
    Geometry *near_geom = nullptr;

    while (b != e) {
        Geometry *g = *b;
        float t;
        if (!g->intersect(ray_orig, ray_dir, t)) {
            ++b;
            continue;
        }
        if (t < near_t) {
            near_t = t;
            near_geom = g;
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

float fresnel(float3 dir, float3 normal, float ior) {
    float cosi = dir.dot(normal);
    float n1 = 1;
    float n2 = ior;
    if (cosi > 0.f)
        swap(n1, n2);

    float sint = (n1 / n2) * sqrt(max(0.f, 1.f - cosi * cosi));
    if (sint >= 1.f)  // total internal relfection
        return 1.f;

    float cost = sqrt(max(0.f, 1.f - sint * sint));
    cosi = abs(cosi);

    float Rs = ((n2 * cosi) - (n1 * cost)) / ((n2 * cosi) + (n1 * cost));
    float Rp = ((n1 * cosi) - (n2 * cost)) / ((n1 * cosi) + (n2 * cost));
    return (Rs * Rs + Rp * Rp) / 2;
}
