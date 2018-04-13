#include "render.hh"
#include "Mat.hh"
#include "Material.hh"
#include "Sphere.hh"
#include "Vec3.hh"
#include "transform.hh"

#include <cmath>
#include <iostream>
#include <stdexcept>
#define PRIMARY_RAYS 1

using namespace std;

void cpu_render(float *pixels, size_t w, size_t h, Mat4f camera,
                vector<Sphere> spheres, unsigned iteration) {
    if (spheres.size() <= 0)
        cerr << "\e[33mWarning: no spheres in call to cpu_render!" << endl;

    float inv_w = 1 / float(w);
    float inv_h = 1 / float(h);
    float fov = 30;
    float aspect_ratio = float(w) / float(h);
    float angle = tan(0.5 * M_PI * fov / 180.0);

    Mat4f dir_camera = transform_clear_translate(camera);
    float3 origin = camera * float3();
    for (size_t y = 0; y < h; ++y) {
        for (size_t x = 0; x < w; ++x) {
            // do raytracing
            float3 color;
            for (size_t i = 0; i < PRIMARY_RAYS; ++i) {
                //  compute the x and y magnitude of each vector
                float v_x = (2 * ((x + randf(0, 1)) * inv_w) - 1) * angle * aspect_ratio;
                float v_y = (1 - 2 * ((y + randf(0, 1)) * inv_h)) * angle;
                float3 ray_dir = dir_camera * float3(v_x, v_y, -1);
                ray_dir.normalize();

                color += cpu_trace(origin, ray_dir, spheres, 0);
            }
            color *= 1.f / PRIMARY_RAYS;

            // compute all-time average color
            const size_t idx = (y * w + x) * 4;
            float3 dst = float3(pixels[idx], pixels[idx+1], pixels[idx+2]);
            float f = 1;
            if (iteration > 0)
                f = 1.f / iteration;
            float3 blended = color * f + dst * (1-f);
            
            // write color
            pixels[idx] = blended.x;
            pixels[idx + 1] = blended.y;
            pixels[idx + 2] = blended.z;
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
    float3 color = 0;

    // cast primary ray
    float3 intersection;
    Sphere *hit_sphere;
    if (!cpu_ray_intersect(ray_orig, ray_dir, spheres, intersection,
                           hit_sphere)) {
        return color;  // return background color
    }

    // emissive objects are just rendered with their emission color for now
    if (!(hit_sphere->material->emission_color == float3(0)))
        return hit_sphere->material->emission_color;

    // TODO: proper ambient color
    color += hit_sphere->material->surface_color * 0.1;

    // compute surface normal
    float3 normal = intersection - hit_sphere->center;
    normal.normalize();

    // compute direction to camera (for Phong specular lighting)
    float3 camera_dir = (ray_orig - intersection).normalize();

    // compute illumination
    for (size_t i = 0; i < spheres.size(); ++i) {
        // skip non-emissive objects and the object hit by the primary ray
        if (spheres[i].material->emission_color == 0 ||
            &spheres[i] == hit_sphere)
            continue;

        // compute shadow ray
        float3 light_dir = spheres[i].center - intersection;
        light_dir.normalize();

        // see if shadow ray hits anything before reaching the light source
        float3 light_intersection;
        Sphere *light;
        cpu_ray_intersect(intersection, light_dir, spheres, light_intersection,
                          light);
        if (light != &spheres[i])
            continue;

        // add diffuse lighting
        color += hit_sphere->material->surface_color *
                 light->material->emission_color * light_dir.dot(normal);

        // add Phong specular lighting
        if (hit_sphere->material->reflection > 0) {
            float3 reflect_dir = light_dir.reflect(normal);
            float rv = reflect_dir.dot(camera_dir);
            color += hit_sphere->material->surface_color *
                     light->material->emission_color * pow(rv, 16) *
                     hit_sphere->material->reflection;
        }
    }

    return color;
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
