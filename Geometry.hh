#ifndef GEOMETRY_HH
#define GEOMETRY_HH

#include "AABB.hh"
#include "Material.hh"
#include "Vec3.hh"

class Geometry {
   public:
    virtual const Material *material() const = 0;
    virtual bool intersect(const float3 &r_orig, const float3 &r_dir, float &t0,
                           float &t1) const = 0;
    virtual float3 normal(const float3 &r_dir,
                          const float3 &intersection) const = 0;
    virtual AABB bounds() const = 0;
};

#endif