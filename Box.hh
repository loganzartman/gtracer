#ifndef BOX_HH
#define BOX_HH

#include "AABB.hh"
#include "Geometry.hh"

class Box : public Geometry {
    AABB box;

public:
    Box(const AABB& box) : box(box) {}
    Box(const float3& a, const float3& b) : box(a, b) {}

    bool intersect(const float3 &r_orig, const float3 &r_dir, float &t0,
                           float &t1) const {
    	using namespace std;

        float tmin = (box.xmin.x - r_orig.x) / r_dir.x;
        float tmax = (box.xmax.x - r_orig.x) / r_dir.x;

        if (tmin > tmax)
            swap(tmin, tmax);

        float tymin = (box.xmin.y - r_orig.y) / r_dir.y;
        float tymax = (box.xmax.y - r_orig.y) / r_dir.y;

        if (tymin > tymax)
            swap(tymin, tymax);

        if ((tmin > tymax) || (tymin > tmax))
            return false;

        if (tymin > tmin)
            tmin = tymin;

        if (tymax < tmax)
            tmax = tymax;

        float tzmin = (box.xmin.z - r_orig.z) / r_dir.z;
        float tzmax = (box.xmax.z - r_orig.z) / r_dir.z;

        if (tzmin > tzmax)
            swap(tzmin, tzmax);

        if ((tmin > tzmax) || (tzmin > tmax))
            return false;

        if (tzmin > tmin)
            tmin = tzmin;

        if (tzmax < tmax)
            tmax = tzmax;

        return true;
    }

    bool intersect(const Box &other) const {
        return box.xmin <= other.box.xmax && box.xmax >= other.box.xmin;
    }

    AABB bounds() const {
    	return box;
    }
};

#endif