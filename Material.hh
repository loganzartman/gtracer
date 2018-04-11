#ifndef MATERIAL_HH
#define MATERIAL_HH

#include "Vec3.hh"

struct Material {
    float3 surface_color, emission_color;
    float transparency, reflection;

    Material(const float3 &sc = float3(0), const float &trans = 0,
             const float &refl = 0, const float3 &ec = float3(0))
        : surface_color(sc),
          emission_color(ec),
          transparency(trans),
          reflection(refl) {}
};

#endif
