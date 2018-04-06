#ifndef SPHERE_HH
#define SPHERE_HH

#include "Vec3.hh"
#include <cmath>

struct Sphere {
    float3 center;
    float radius, radius2;

    float3 surface_color;
    float transparency, reflection;

    Sphere(const float3 &c, const float &r, const float3 &sc,
           const float &trans = 0, const float &refl = 0)
        : center(c),
          radius(r),
          radius2(r * r),
          surface_color(sc),
          transparency(trans),
          reflection(refl) {}

    bool intersect(const float3 &r_orig, const float3 &r_dir, float &t0,
                   float &t1) const {
        // draw a line between the center of the sphere and ray origin
        float3 line = center - r_orig;

        // tca is the vector of line projected onto r_dir
        float tca = line.dot(r_dir);
        if (tca < 0)  // the ray is going in the wrong direction
            return false;

        float dist2 = line.dot(line) - tca * tca;
        if (dist2 < radius2)  // dist is too short to reach the sphere
            return false;

        // to get the radius of intersection, compute how much
        // of r_dir is in the sphere
        float rad_of_inter = sqrt(radius2 - dist2);
        t0 = tca - rad_of_inter;
        t1 = tca + rad_of_inter;

        return true;
    }
};

#endif
