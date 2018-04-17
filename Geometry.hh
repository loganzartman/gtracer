#ifndef GEOMETRY_HH
#define GEOMETRY_HH

#include "AABB.hh"
#include "Vec3.hh"

class Geometry {
    virtual bool intersect(const float3 &r_orig, const float3 &r_dir, float &t0,
                           float &t1) const = 0;
    virtual AABB bounds() const = 0;
};

#endif