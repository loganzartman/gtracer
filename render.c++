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
            float3 ray(v_x, v_y, -1);
            ray.normalize();

            float3 color = cpu_trace(float3(0), ray, spheres, 0);
            const size_t idx = (y * w + x) * 4;
            pixels[idx] = color.x;
            pixels[idx + 1] = color.y;
            pixels[idx + 2] = color.z;
            pixels[idx + 3] = 1;  // alpha
        }
    }
}

/*
 *  for every sphere, detect collision. if there is one,
 *  see if it is the closest. Once we get the closest
 *  collision, determine the color we should show
 */
float3 cpu_trace(const float3 &ray_orig, const float3 &ray_dir,
                 vector<Sphere> spheres, int depth) {
    float near = INFINITY;
    Sphere *near_sphere = nullptr;

    size_t index = -1;

    for (size_t i = 0; i < spheres.size(); ++i) {
        float t0 = INFINITY;
        float t1 = INFINITY;

        if (spheres[i].intersect(ray_orig, ray_dir, t0, t1)) {
            // if t0 is negative, that's on the other side of the camera
            if (t0 < 0)
                t0 = t1;

            if (t0 < near) {
                near = t0;
                near_sphere = &spheres[i];
                index = i;
            }
        }
    }

    if (!near_sphere)
        return float3(0);  // return background color

    // std::cout << "collision between ray with origin=" << ray_orig.print()
    // << "and dir=" << ray_dir.print() << " and sphere=" << index
    // << endl;
    return near_sphere->surface_color;
}
