#include "render.hh"
#include "Sphere.hh"
#include "Vec3.hh"

#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace std;

void cpu_render(float *pixels, size_t w, size_t h, vector<Sphere> spheres) {
    if (spheres.size() <= 0)
        cerr << "\e[33mWarning: no spheres in call to cpu_render!" << endl;

    float inv_w = 1 / float(w);
    float inv_h = 1 / float(h);
    float fov = 30;
    float aspect_ratio = float(w) / float(h);
    float angle = tan(0.5 * M_PI * fov / 180.0);

    for (size_t y = 0; y < h; ++y) {
        for (size_t x = 0; x < w; ++x) {
            //  compute the x and y magnitude of each vector
            float v_x = (2 * ((x + 0.5) * inv_w) - 1) * angle * aspect_ratio;
            float v_y = (1 - 2 * ((y + 0.5) * inv_h)) * angle;
            float3 ray_dir(v_x, v_y, -1);
            ray_dir.normalize();

            float3 color = cpu_trace(float3(0), ray_dir, spheres, 0);
            const size_t idx = (y * w + x) * 4;
            pixels[idx] = color.x;
            pixels[idx + 1] = color.y;
            pixels[idx + 2] = color.z;
            pixels[idx + 3] = 1;  // alpha
        }
    }
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
                 vector<Sphere> spheres, int depth) {
    float3 intersection;
    Sphere *hit_sphere;
    float3 color = 0;
    if (!cpu_ray_intersect(ray_orig, ray_dir, spheres, intersection,
                          hit_sphere)) {
        return color;  // return background color
    }
    float3 normal = intersection - hit_sphere->center;
    
    for (size_t i = 0; i < spheres.size(); ++i) {
        if (spheres[i].emission_color.x <= 0 || &spheres[i] == hit_sphere)
            continue;

        float3 light_dir = spheres[i].center - intersection;
        light_dir.normalize();

        float3 light_intersection;
        Sphere *light;
        cpu_ray_intersect(intersection, light_dir, spheres, light_intersection, light);
        if (light != &spheres[i])
            continue;

        color += hit_sphere->surface_color * light->emission_color * light_dir.dot(normal);
    }

    return color;
    //return hit_sphere->surface_color;
}

/**
 * @brief Finds the nearest intersection between a ray and scene geometry.
 *
 * @param[in] ray_orig Ray origin point
 * @param[in] ray_dir Ray direction as unit vector
 * @param[in] spheres Scene geometry
 * @param[out] intersection The point of intersection
 * @param[out] hit_sphere The sphere that was intersected
 * @return Whether there was an intersection
 */
bool cpu_ray_intersect(const float3 &ray_orig, const float3 &ray_dir,
                       vector<Sphere> &spheres, float3 &intersection,
                       Sphere *&hit_sphere) {
    float near_t = INFINITY;
    Sphere *near_sphere = nullptr;

    for (size_t i = 0; i < spheres.size(); ++i) {
        float t0 = INFINITY;
        float t1 = INFINITY;

        if (spheres[i].intersect(ray_orig, ray_dir, t0, t1)) {
            // if t0 is negative, that's on the other side of the camera
            if (t0 < 0)
                t0 = t1;

            if (t0 < near_t) {
                near_t = t0;
                near_sphere = &spheres[i];
            }
        }
    }

    if (near_sphere) {
        intersection = ray_orig + ray_dir * near_t;
        hit_sphere = near_sphere;
        return true;
    }
    return false;
}
