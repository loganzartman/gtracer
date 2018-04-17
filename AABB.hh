#ifndef AABB_HH
#define AABB_HH

#include <algorithm>
#include "Vec3.hh"

struct AABB {
    float3 xmin;
    float3 xmax;

    AABB(const float3 &a, const float3 &b) : xmin(min(a, b)), xmax(max(a, b)) {}

    bool intersect(const float3 &r_orig, const float3 &r_dir, float &t0, float &t1) const {
        using namespace std;

        float tmin = (xmin.x - r_orig.x) / r_dir.x;
        float tmax = (xmax.x - r_orig.x) / r_dir.x;

        if (tmin > tmax) swap(tmin, tmax);

        float tymin = (xmin.y - r_orig.y) / r_dir.y;
        float tymax = (xmax.y - r_orig.y) / r_dir.y;

        if (tymin > tymax) swap(tymin, tymax);

        if ((tmin > tymax) || (tymin > tmax))
            return false;

        if (tymin > tmin)
            tmin = tymin;

        if (tymax < tmax)
            tmax = tymax;

        float tzmin = (xmin.z - r_orig.z) / r_dir.z;
        float tzmax = (xmax.z - r_orig.z) / r_dir.z;

        if (tzmin > tzmax) swap(tzmin, tzmax);

        if ((tmin > tzmax) || (tzmin > tmax))
            return false;

        if (tzmin > tmin)
            tmin = tzmin;

        if (tzmax < tmax)
            tmax = tzmax;

        return true; 
    }

    bool intersect(const AABB &other) const {
        return xmin <= other.xmax && xmax >= other.xmin;
    }

    AABB bounds() const {
        return *this;
    }
};

#endif