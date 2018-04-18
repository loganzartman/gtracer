#ifndef UNIFORMGRID_HH
#define UNIFORMGRID_HH

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>
#include "AABB.hh"
#include "Vec3.hh"
#include "Geometry.hh"

class UniformGrid {
   private:
    size_t rx;
    size_t ry;
    size_t rz;
    size_t *data;

    size_t index(size_t x, size_t y, size_t z) const {
        return z * ry * rx + y * rx + x;
    }

   public:
    UniformGrid(float3 resolution)
        : rx(ceil(resolution.x)),
          ry(ceil(resolution.y)),
          rz(ceil(resolution.z)),
          data(new size_t[rx * ry * rz * 2]) {}

    ~UniformGrid() {
        delete[] data;
    }

    /**
     * @brief Get the start index for a given grid cell
     * @details An index is used to index the geometry list stored elsewhere.
     * 
     * @param x cell x index
     * @param y cell y index
     * @param z cell z index
     * @return The first index into the external geometry array
     */
    size_t& first(size_t x, size_t y, size_t z) {
        return data[index(x, y, z)];
    }

    /**
     * @brief Get the end index for a given grid cell
     * @details An index is used to index the geometry list stored elsewhere.
     * 
     * @param x cell x index
     * @param y cell y index
     * @param z cell z index
     * @return The last index into the external geometry array
     */
    size_t& last(size_t x, size_t y, size_t z) {
        return data[index(x, y, z) + 1];
    }

    /**
     * @brief Compute heuristic for grid resolution
     * 
     * @param bounds bounding box for entire scene
     * @param n number of geometries in scene
     * @param density an arbitrary density factor
     * @return a grid resolution
     */
    static float3 resolution(AABB bounds, int n, int density = 5) {
        float3 d = bounds.xmax - bounds.xmin;
        float vol = d.x * d.y * d.z;
        float factor = cbrt(density * n / vol);
        return d * factor;
    }
};

#endif