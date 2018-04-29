#ifndef RAYTRACING_HH
#define RAYTRACING_HH
#include "util.hh"
#include "Vec3.hh"

namespace raytracing {
HOSTDEV static float fresnel(Float3 dir, Float3 normal, float ior) {
    float cosi = dir.dot(normal);
    float n1 = 1;
    float n2 = ior;
    if (cosi > 0.f)
        util::swap(n1, n2);

    float sint = (n1 / n2) * sqrt(util::max(0.f, 1.f - cosi * cosi));
    if (sint >= 1.f)  // total internal relfection
        return 1.f;

    float cost = sqrt(util::max(0.f, 1.f - sint * sint));
    cosi = abs(cosi);

    float Rs = ((n2 * cosi) - (n1 * cost)) / ((n2 * cosi) + (n1 * cost));
    float Rp = ((n1 * cosi) - (n2 * cost)) / ((n1 * cosi) + (n2 * cost));
    return (Rs * Rs + Rp * Rp) / 2;
}
}

#endif