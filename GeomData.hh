#ifndef GEOMDATA_HH
#define GEOMDATA_HH
#include "Vec3.hh"

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

union GeomData {
    SphereData sphere;
    TriData tri;
    BoxData box;
    
    GeomData(const SphereData& data) : sphere(data) {}
    GeomData(const TriData& data) : tri(data) {}
    GeomData(const BoxData& data) : box(data) {}
};

#endif