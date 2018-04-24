#ifndef BOX_HH
#define BOX_HH

#include <algorithm>
#include <cassert>
#include <cmath>
#include "AABB.hh"
#include "Geometry.hh"

class Box : public Geometry {
    AABB box;
    const Material *mat;

   public:
    Box(const AABB &box) : box(box) {}
    Box(const AABB &box, Material *mat) : box(box), mat(mat) {}
    Box(const Float3 &a, const Float3 &b) : box(a, b) {}
    Box(const Float3 &a, const Float3 &b, const Material *mat)
        : box(a, b), mat(mat) {}

    const Material *material() const { return mat; }

    bool intersect(const Float3 &r_orig, const Float3 &r_dir, float &t) const {
        // https://gamedev.stackexchange.com/a/18459
        using namespace std;

        // r.dir is unit direction vector of ray
        Float3 invdir(1.f / r_dir.x, 1.f / r_dir.y, 1.f / r_dir.z);
        // lb is the corner of AABB with minimal coordinates - left bottom, rt
        // is maximal corner r.org is origin of ray
        float t1 = (box.xmin.x - r_orig.x) * invdir.x;
        float t2 = (box.xmax.x - r_orig.x) * invdir.x;
        float t3 = (box.xmin.y - r_orig.y) * invdir.y;
        float t4 = (box.xmax.y - r_orig.y) * invdir.y;
        float t5 = (box.xmin.z - r_orig.z) * invdir.z;
        float t6 = (box.xmax.z - r_orig.z) * invdir.z;

        float tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6));
        float tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6));

        // if tmax < 0, ray (line) is intersecting AABB, but the whole AABB is
        // behind us
        if (tmax < 0) {
            t = tmax;
            return false;
        }

        // if tmin > tmax, ray doesn't intersect AABB
        if (tmin > tmax) {
            t = tmax;
            return false;
        }

        t = tmin;
        return true;
    }

    bool intersect(const Box &other) const {
        if (box.xmin.x > other.box.xmax.x || box.xmin.y > other.box.xmax.y ||
            box.xmin.z > other.box.xmax.z)
            return false;
        if (box.xmax.x < other.box.xmin.x || box.xmax.y < other.box.xmin.y ||
            box.xmax.z < other.box.xmin.z)
            return false;
        return true;
    }

    Float3 normal(const Float3 &r_dir, const Float3 &intersection) const {
        const Float3 x_axis = Float3(1, 0, 0);
        const Float3 y_axis = Float3(0, 1, 0);
        const Float3 z_axis = Float3(0, 0, 1);

        Float3 p = intersection - (box.xmin * 0.5 + box.xmax * 0.5);
        p = p / (box.xmax - box.xmin);
        float dx = p.dot(x_axis), dy = p.dot(y_axis), dz = p.dot(z_axis);
        float adx = fabs(dx), ady = fabs(dy), adz = fabs(dz);
        if (adx > ady && adx > adz)
            return dx < 0 ? x_axis * -1 : x_axis;
        else if (ady > adx && ady > adz)
            return dy < 0 ? y_axis * -1 : y_axis;
        else
            return dz < 0 ? z_axis * -1 : z_axis;
    }

    AABB bounds() const { return box; }
};

#endif
