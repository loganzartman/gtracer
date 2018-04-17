#ifndef AABB_HH
#define AABB_HH

#include <algorithm>
#include "Vec3.hh"

struct AABB {
    float3 xmin;
    float3 xmax;

    AABB(const float3 &a, const float3 &b) : xmin(min(a, b)), xmax(max(a, b)) {}

    AABB bounds() const { return *this; }
};

#endif