#ifndef BOX_HH
#define BOX_HH

#include <cassert>
#include <cmath>
#include "AABB.hh"
#include "GeomData.hh"
#include "util.hh"

struct Box {
    HOSTDEV static bool intersect(const BoxData &data, const Float3 &r_orig,
                                  const Float3 &r_dir, float &t) {
        // https://gamedev.stackexchange.com/a/18459

        // r.dir is unit direction vector of ray
        Float3 invdir(1.f / r_dir.x, 1.f / r_dir.y, 1.f / r_dir.z);
        // lb is the corner of AABB with minimal coordinates - left bottom, rt
        // is maximal corner r.org is origin of ray
        float t1 = (data.box.xmin.x - r_orig.x) * invdir.x;
        float t2 = (data.box.xmax.x - r_orig.x) * invdir.x;
        float t3 = (data.box.xmin.y - r_orig.y) * invdir.y;
        float t4 = (data.box.xmax.y - r_orig.y) * invdir.y;
        float t5 = (data.box.xmin.z - r_orig.z) * invdir.z;
        float t6 = (data.box.xmax.z - r_orig.z) * invdir.z;

        float tmin = util::max(util::max(util::min(t1, t2), util::min(t3, t4)),
                               util::min(t5, t6));
        float tmax = util::min(util::min(util::max(t1, t2), util::max(t3, t4)),
                               util::max(t5, t6));

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

    HOSTDEV static bool intersect(const BoxData &data, const BoxData &other) {
        if (data.box.xmin.x > other.box.xmax.x ||
            data.box.xmin.y > other.box.xmax.y ||
            data.box.xmin.z > other.box.xmax.z)
            return false;
        if (data.box.xmax.x < other.box.xmin.x ||
            data.box.xmax.y < other.box.xmin.y ||
            data.box.xmax.z < other.box.xmin.z)
            return false;
        return true;
    }

    HOSTDEV static Float3 normal(const BoxData &data, const Float3 &r_dir,
                                 const Float3 &intersection) {
        const Float3 x_axis = Float3(1, 0, 0);
        const Float3 y_axis = Float3(0, 1, 0);
        const Float3 z_axis = Float3(0, 0, 1);

        Float3 p = intersection - (data.box.xmin * 0.5f + data.box.xmax * 0.5f);
        p = p / (data.box.xmax - data.box.xmin);
        float dx = p.dot(x_axis), dy = p.dot(y_axis), dz = p.dot(z_axis);
        float adx = fabs(dx), ady = fabs(dy), adz = fabs(dz);
        if (adx > ady && adx > adz)
            return dx < 0 ? x_axis * -1.f : x_axis;
        else if (ady > adx && ady > adz)
            return dy < 0 ? y_axis * -1.f : y_axis;
        else
            return dz < 0 ? z_axis * -1.f : z_axis;
    }

    HOSTDEV static AABB bounds(const BoxData &data) { return data.box; }
};

#endif
