#ifndef SPHERE_HH
#define SPHERE_HH

#include <algorithm>
#include <cassert>
#include <cmath>
#include "AABB.hh"
#include "GeomData.hh"
#include "Material.hh"
#include "Vec3.hh"
#include "util.hh"

struct Sphere {
    HOSTDEV static bool intersect(const SphereData &data, const Float3 &r_orig,
                                  const Float3 &r_dir, float &t) {
        // draw a line between the center of the sphere and ray origin
        Float3 line = data.center - r_orig;

        // tca is the vector of line projected onto r_dir
        float tca = line.dot(r_dir);
        if (tca < 0)  // the ray is going in the wrong direction
            return false;

        float dist2 = line.dot(line) - tca * tca;
        if (dist2 >
            data.radius * data.radius)  // the radius is too short to span dist
            return false;

        // to get the radius of intersection, compute how much
        // of r_dir is in the sphere
        float rad_of_inter = sqrt(data.radius * data.radius - dist2);

        // t0 and t1 are parametric coefficients of the original ray
        // AKA how far down the ray the collision occurs
        float t0 = tca - rad_of_inter;
        float t1 = tca + rad_of_inter;
        t = util::min(t0, t1);
        if (t < 0)
            t = util::max(t0, t1);
        if (t < 0)
            return false;

        return true;
    }

    HOSTDEV static Float3 normal(const SphereData &data, const Float3 &,
                                 const Float3 &intersection) {
        return (intersection - data.center).normalize();
    }

    HOSTDEV static AABB bounds(const SphereData &data) {
        const Float3 offset(data.radius);
        return AABB(data.center - offset, data.center + offset);
    }
};

#endif
