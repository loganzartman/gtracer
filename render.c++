#include "Sphere.hh"
#include "Vec3.hh"
#include "render.hh"

#include <cmath>
#include <stdexcept>

using namespace std;

float3 *cpu_render (Sphere *spheres, size_t num_spheres) {
  if (spheres == nullptr)
    throw invalid_argument("Spheres is null");
  if (num_spheres <= 0)
    throw invalid_argument("There needs to be at least one sphere");

  size_t w = 640, h = 480;
  float3 *image = new float3[w*h];
  float3 *pixel = image;

  float inv_w = 1 / float(w);
  float inv_h = 1 / float(h);
  float fov = 30;
  float aspect_ratio = float(w) / float(h);
  float angle = tan(0.5 * M_PI * fov / 180.0);

  for (size_t y = 0; y < h; ++y) {
    for (size_t x = 0; x < w; ++x) {
      // compute the x and y magnitude of each vector
      // TODO: do the math, replace placeholder
      float v_x = 1;
      float v_y = 1;
      float3 ray(v_x, v_y, 1);
      ray.normalize();

      *pixel = trace(float3(0), ray, spheres, num_spheres, 0);
      ++pixel;
    }
  }

  return image;
}

float3 trace (const float3 &ray_orig, const float3 &ray_dir, Sphere *spheres, size_t num_spheres, int depth) {
  // for every sphere, detect collision. if there is one,
  // see if it is the closest. Once we get the closest
  // collision, determine the color we should show
  float near = INFINITY;
  Sphere *near_sphere = nullptr;

  for (size_t i = 0; i < num_spheres; ++i) {
    float t0 = INFINITY;
    float t1 = INFINITY;
    
    if (spheres[i].intersect(ray_orig, ray_dir, t0, t1)) {
      if (t0 < 0) //if t0 is negative, that's on the other side of the camera
        t0 = t1;

      if (t0 < near) {
        near = t0;
        near_sphere = &spheres[i];
      }
    }
  }

  if (!near_sphere)
    return float3(2); //return background color

  return float3(0);
}
