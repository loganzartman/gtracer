#ifndef TRIANGLE_HH
#define TRIANGLE_HH

#include <algorithm>
#include <cassert>
#include <vector>

#include "Geometry.hh"
#include "Vec3.hh"

class Tri : public Geometry {
    float3 a, b, c;
    const Material* mat;

   public:
    Tri(const float3& a, const float3& b, const float3& c,
        const Material* mat = nullptr)
        : a(a), b(b), c(c), mat(mat) {}
    Tri(const std::vector<float3>& p, const Material* mat = nullptr)
        : a(p[0]), b(p[1]), c(p[2]), mat(mat) {
        assert(p.size() == 3);
    }

    bool intersect(const float3& r_orig, const float3& r_dir, float& t) const {
        // edges between points
        const float3 ab = b - a;
        const float3 ac = c - a;

        float3 p = r_dir.cross(ac);
        float det = ab.dot(p);

        if (fabs(det) < 0.0001)
            return false;

        float inv_det = 1 / det;

        float3 tvec = r_orig - a;
        float u = tvec.dot(p) * inv_det;
        if (u < 0 || u > 1)
            return false;

        float3 q = tvec.cross(ab);
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

    float3 normal(const float3& r_dir, const float3& intersection) const {
        const float3 ab = b - a;
        const float3 ac = c - a;

        return ab.cross(ac).normalize();
    }

    const Material* material() const { return mat; }

    AABB bounds() const {
        float3 min_corner(min(min(a, b), c));
        float3 max_corner(max(max(a, b), c));
        return AABB(min_corner, max_corner);
    }
};

#endif
