#ifndef TRIANGLE_HH
#define TRIANGLE_HH

#include <cassert>

#include "GeomData.hh"
#include "Vec3.hh"

struct Tri {
    HOSTDEV static bool intersect(const TriData &data, const Float3& r_orig, const Float3& r_dir, float& t) {
        // edges between points
        const Float3 ab = data.b - data.a;
        const Float3 ac = data.c - data.a;

        Float3 p = r_dir.cross(ac);
        float det = ab.dot(p);

        if (fabs(det) < 0.0001)
            return false;

        float inv_det = 1 / det;

        Float3 tvec = r_orig - data.a;
        float u = tvec.dot(p) * inv_det;
        if (u < 0 || u > 1)
            return false;

        Float3 q = tvec.cross(ab);
        float v = r_dir.dot(q) * inv_det;
        if (v < 0 || u + v > 1)
            return false;

        // t0 = ab.cross(ac).dot(a - r_orig) / ab.cross(ac).dot(r_dir);
        float t0 = ac.dot(q) * inv_det;

        if (t0 < 0)
            return false;
        t = t0;

        return true;
    }

    HOSTDEV static Float3 normal(const TriData &data, const Float3& r_dir, const Float3& intersection) {
        const Float3 ab = data.b - data.a;
        const Float3 ac = data.c - data.a;

        return ab.cross(ac).normalize();
    }

    HOSTDEV static AABB bounds(const TriData &data) {
        Float3 min_corner(vmin(vmin(data.a, data.b), data.c));
        Float3 max_corner(vmax(vmax(data.a, data.b), data.c));
        return AABB(min_corner, max_corner);
    }
};

#endif
