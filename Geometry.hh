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
	++b;

	while (b != e) {
		AABB candidate = (*b)->bounds();
		bounds = AABB(min(bounds.xmin, candidate.xmin),
		              max(bounds.xmax, candidate.xmax));
		++b;
	}
	return bounds;
}

#endif