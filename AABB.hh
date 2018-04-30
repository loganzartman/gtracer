#ifndef AABB_HH
#define AABB_HH

#include "Vec3.hh"
#include "util.hh"

struct AABB {
    Float3 xmin;
    Float3 xmax;

    HOSTDEV AABB() : xmin(0), xmax(0) {}

    HOSTDEV AABB(const Float3 &a, const Float3 &b)
        : xmin(vmin(a, b)), xmax(vmax(a, b)) {}

    HOSTDEV bool contains(const Float3 &p) const {
        return p.x >= xmin.x && p.y >= xmin.y && p.z >= xmin.z &&
               p.x <= xmax.x && p.y <= xmax.y && p.z <= xmax.z;
    }

    HOSTDEV AABB bounds() const { return *this; }
};

#endif