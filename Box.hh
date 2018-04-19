#ifndef BOX_HH
#define BOX_HH

#include <algorithm>
#include "AABB.hh"
#include "Geometry.hh"

class Box : public Geometry {
    AABB box;
    const Material *mat;

   public:
    Box(const AABB &box) : box(box) {}
    Box(const AABB &box, Material *mat) : box(box), mat(mat) {}
    Box(const float3 &a, const float3 &b) : box(a, b) {}
    Box(const float3 &a, const float3 &b, const Material *mat)
        : box(a, b), mat(mat) {}

    const Material *material() const { return mat; }

    bool intersect(const float3 &r_orig, const float3 &r_dir, float &t) const {
        using namespace std;

        float tx1 = (box.xmin.x - r_orig.x) / r_dir.x;
        float tx2 = (box.xmax.x - r_orig.x) / r_dir.x;

        float tmin = min(tx1, tx2);
        float tmax = max(tx1, tx2);

        float ty1 = (box.xmin.y - r_orig.y) / r_dir.y;
        float ty2 = (box.xmax.y - r_orig.y) / r_dir.y;

        tmin = max(tmin, min(ty1, ty2));
        tmax = min(tmax, max(ty1, ty2));

        float tz1 = (box.xmin.z - r_orig.z) / r_dir.z;
        float tz2 = (box.xmax.z - r_orig.z) / r_dir.z;

        tmin = max(tmin, min(tz1, tz2));
        tmax = min(tmax, max(tz1, tz2));

        if (tmax < tmin)
            return false;

        if (tmax < 0)
            return false;

        float t0 = fabs(tmin);  // not sure
        float t1 = fabs(tmax);  // might be bad
        t = t0;

        return true;
    }

    bool intersect(const Box &other) const {
        return box.xmin <= other.box.xmax && box.xmax >= other.box.xmin;
    }

    float3 normal(const float3 &r_dir, const float3 &intersection) const {
        float bias = 1.000001;
        float3 p = intersection - (box.xmin * 0.5 + box.xmax * 0.5);

        float3 d((box.xmin - box.xmax) * 0.5);
        vabs(d);
        float3 normal(p / d * bias);

        return normal.normalize();
    }

    AABB bounds() const { return box; }
};

#endif
