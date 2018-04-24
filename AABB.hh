#ifndef AABB_HH
#define AABB_HH

#include <algorithm>
#include "Vec3.hh"

struct AABB {
    Float3 xmin;
    Float3 xmax;

    AABB(const Float3 &a, const Float3 &b) : xmin(min(a, b)), xmax(max(a, b)) {}

    AABB& operator=(const AABB& other) {
    	xmin = other.xmin;
    	xmax = other.xmax;
    	return *this;
    }

    bool contains(const Float3 &p) const {
        return p.x >= xmin.x && p.y >= xmin.y && p.z >= xmin.z &&
               p.x <= xmax.x && p.y <= xmax.y && p.z <= xmax.z;
    }

    AABB bounds() const { return *this; }
};

#endif