#ifndef GEOMETRY_HH
#define GEOMETRY_HH

#include <cassert>
#include "AABB.hh"
#include "Material.hh"
#include "Sphere.hh"
#include "GeomData.hh"
#include "Tri.hh"
#include "Box.hh"
#include "Vec3.hh"
#include "util.hh"

enum class GeomType {
    Sphere,
    Tri,
    Box
};

class Geometry {
   public:
    GeomType type;
    GeomData data;
    const Material* mat;

    Geometry(GeomType type, GeomData data, const Material* mat = nullptr) : type(type), data(data), mat(mat) {}
    Geometry(SphereData data, const Material* mat = nullptr) : type(GeomType::Sphere), data(data), mat(mat) {}
    Geometry(TriData data, const Material* mat = nullptr) : type(GeomType::Tri), data(data), mat(mat) {}
    Geometry(BoxData data, const Material* mat = nullptr) : type(GeomType::Box), data(data), mat(mat) {}

    HOSTDEV const Material* material() const {
        return mat;
    }

    HOSTDEV bool intersect(const Float3 &r_orig, const Float3 &r_dir,
                           float &t) const {
        switch (type) {
            case GeomType::Sphere:
                return Sphere::intersect(data.sphere, r_orig, r_dir, t);
            case GeomType::Tri:
                return Tri::intersect(data.tri, r_orig, r_dir, t);
            case GeomType::Box:
                return Box::intersect(data.box, r_orig, r_dir, t);
            default:
                assert(false);
        }
    }
    
    HOSTDEV Float3 normal(const Float3 &r_dir,
                          const Float3 &intersection) const {
        switch (type) {
            case GeomType::Sphere:
                return Sphere::normal(data.sphere, r_dir, intersection);
            case GeomType::Tri:
                return Tri::normal(data.tri, r_dir, intersection);
            case GeomType::Box:
                return Box::normal(data.box, r_dir, intersection);
            default:
                assert(false);
        }
    }
    
    HOSTDEV AABB bounds() const {
        switch (type) {
            case GeomType::Sphere:
                return Sphere::bounds(data.sphere);
            case GeomType::Tri:
                return Tri::bounds(data.tri);
            case GeomType::Box:
                return Box::bounds(data.box);
            default:
                assert(false);
        }
    }
};

/**
 * @brief Finds the AABB of a collection of geometry
 * @details Each item in the collection must implement bounds()
 *
 * @param b Beginning InputIterator
 * @param e Ending InputIterator
 * @return AABB of the collection
 */
template <typename II>
AABB geometry_bounds(II b, II e) {
    if (b == e)
        return AABB(0, 0);

    AABB bounds = (*b)->bounds();
    unsigned i = 0;
    ++b;

    while (b != e) {
        AABB candidate = (*b)->bounds();
        bounds = AABB(vmin(bounds.xmin, candidate.xmin),
                      vmax(bounds.xmax, candidate.xmax));
        ++b;
        ++i;
    }
    return bounds;
}

#endif