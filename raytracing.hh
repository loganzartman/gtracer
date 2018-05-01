#ifndef RAYTRACING_HH
#define RAYTRACING_HH
#include <cassert>
#include "Vec3.hh"
#include "util.hh"

#define PRIMARY_RAYS 1
#define SKY_COLOR Float3(0)

namespace raytracing {
DEVICE static Float3 trace(const Float3 &ray_orig, const Float3 &ray_dir,
                           AABB world_bounds, const UniformGrid &grid,
                           bool accel, int depth);

DEVICE static bool ray_intersect(const Float3 &ray_orig, const Float3 &ray_dir,
                                 AABB world_bounds, const UniformGrid &grid,
                                 Float3 &intersection, Geometry *&hit_geom);

template <typename II>
DEVICE static bool ray_intersect_items(const Float3 &ray_orig,
                                       const Float3 &ray_dir, II b, II e,
                                       Float3 &intersection,
                                       Geometry *&hit_geom);

DEVICE static float fresnel(Float3 dir, Float3 normal, float ior) {
    float cosi = dir.dot(normal);
    float n1 = 1;
    float n2 = ior;
    if (cosi > 0.f)
        util::swap(n1, n2);

    float sint = (n1 / n2) * sqrt(util::max(0.f, 1.f - cosi * cosi));
    if (sint >= 1.f)  // total internal relfection
        return 1.f;

    float cost = sqrt(util::max(0.f, 1.f - sint * sint));
    cosi = abs(cosi);

    float Rs = ((n2 * cosi) - (n1 * cost)) / ((n2 * cosi) + (n1 * cost));
    float Rp = ((n1 * cosi) - (n2 * cost)) / ((n1 * cosi) + (n2 * cost));
    return (Rs * Rs + Rp * Rp) / 2;
}

DEVICE static Float3 tonemap(Float3 color, float gamma = 1.f);
}  // namespace raytracing

/**
 * @brief Traces a primary ray and produces a color
 *
 * @param ray_orig Ray origin point
 * @param ray_dir Ray direction as a unit vector
 * @param spheres Scene geometry
 * @param depth Maximum trace depth
 * @return Color computed for this primary ray
 */
DEVICE static Float3 raytracing::trace(const Float3 &ray_orig,
                                       const Float3 &ray_dir, AABB world_bounds,
                                       const UniformGrid &grid, bool accel,
                                       int depth) {
    Float3 color = 1.0;
    Float3 light = 0.0;

    Float3 origin = ray_orig;
    Float3 direction = ray_dir;
    for (int i = 0; i < depth; ++i) {
        // cast ray
        Float3 intersection;
        Geometry *hit_geom;

        bool intersected;
        if (accel)
            intersected = raytracing::ray_intersect(
                origin, direction, world_bounds, grid, intersection, hit_geom);
        else
            intersected = raytracing::ray_intersect_items(
                origin, direction, grid.first(), grid.last(), intersection,
                hit_geom);

        if (!intersected) {
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
        origin += normal * 1e-4;
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
DEVICE static bool raytracing::ray_intersect(
    const Float3 &ray_orig, const Float3 &ray_dir, AABB world_bounds,
    const UniformGrid &grid, Float3 &intersection, Geometry *&hit_geom) {
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
            if (raytracing::ray_intersect_items(ray_orig, ray_dir, b, e,
                                                intersection, hit_geom)) {
                Float3 cell_pos =
                    world_bounds.xmin + Float3(voxel_pos) * grid.cell_size;
                if (AABB(cell_pos - 1e-5, cell_pos + grid.cell_size + 1e-5)
                        .contains(intersection))
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
DEVICE static bool raytracing::ray_intersect_items(const Float3 &ray_orig,
                                                   const Float3 &ray_dir, II b,
                                                   II e, Float3 &intersection,
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

/**
 * @brief Use Reinhard HDR tonemapping algorithm to transform pixels
 * @param[in] pixels the pixels to transform
 * @param[in] w the width of the screen
 * @param[in] h the height of the screen
 */
DEVICE static Float3 raytracing::tonemap(Float3 color, float gamma) {
    Float3 ldr = color / (color + Float3(1));
    ldr = pow(ldr, 1.0f / gamma);
    return ldr;
}
#endif