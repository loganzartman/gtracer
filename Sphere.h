#ifndef SPHERE
#define SPHERE

#include "Vec3.h"

struct Sphere {
  float3 center;
  float radius, radius2;

  float3 surface_color;
  float transparency, reflection;

  Sphere(const float3 &c, const float &r, const float3 &sc,
         const float &trans = 0, const float &refl = 0)
      : center(c), radius(r), radius2(r * r), surface_color(sc),
        transparency(trans), reflection(refl) {}

  bool intersect(const float3 &r_orig, const float3 &r_dir, float &h0,
                 float &h1) const {
    // detect intersection here, set h0, h1 if they exist
    return true;
  }
};

#endif
