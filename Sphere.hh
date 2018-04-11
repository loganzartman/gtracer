#ifndef SPHERE_HH
#define SPHERE_HH

#include <cmath>
#include "Material.hh"
#include "Vec3.hh"

struct Sphere {
    float3 center;
    float radius;
    const Material *material;

    Sphere(const float3 &c, const float &r, const Material *m)
        : center(c), radius(r), material(m) {}

    bool intersect(const float3 &r_orig, const float3 &r_dir, float &t0,
                   float &t1) const {
        // draw a line between the center of the sphere and ray origin
        float3 line = center - r_orig;

        // tca is the vector of line projected onto r_dir
        float tca = line.dot(r_dir);
        if (tca < 0)  // the ray is going in the wrong direction
            return false;

        float dist2 = line.dot(line) - tca * tca;
        if (dist2 > radius * radius)  // the radius is too short to span dist
            return false;

        // to get the radius of intersection, compute how much
        // of r_dir is in the sphere
        float rad_of_inter = sqrt(radius * radius - dist2);

        // t0 and t1 are parametric coefficients of the original ray
        // AKA how far down the ray the collision occurs
        t0 = tca - rad_of_inter;
        t1 = tca + rad_of_inter;

        return true;
    }
};

#endif
