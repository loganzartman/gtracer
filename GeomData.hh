#ifndef GEOMDATA_HH
#define GEOMDATA_HH
#include "AABB.hh"
#include "Vec3.hh"
#include "util.hh"

struct SphereData {
    Float3 center;
    float radius;
};

struct TriData {
    Float3 a, b, c;
};

struct BoxData {
    AABB box;
};

struct GeomData {
    SphereData sphere;
    TriData tri;
    BoxData box;

    HOSTDEV GeomData(const SphereData& data) : sphere(data) {}
    HOSTDEV GeomData(const TriData& data) : tri(data) {}
    HOSTDEV GeomData(const BoxData& data) : box(data) {}
};

#endif