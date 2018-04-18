#ifndef UNIFORMGRID_HH
#define UNIFORMGRID_HH

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>
#include "AABB.hh"
#include "Geometry.hh"
#include "Vec3.hh"

class UniformGrid {
   public:
    const size_t rx;
    const size_t ry;
    const size_t rz;
   
   private:
    size_t* data;
    std::pair<Geometry*, size_t>* geom_cell;

    size_t index(size_t x, size_t y, size_t z) const {
        return z * ry * rx + y * rx + x;
    }

   public:
    UniformGrid(float3 resolution)
        : rx(ceil(resolution.x)),
          ry(ceil(resolution.y)),
          rz(ceil(resolution.z)),
          data(new size_t[rx * ry * rz * 2]),
          geom_cell(nullptr) {
        using namespace std;
    }

    ~UniformGrid() {
        delete[] data;
        delete[] geom_cell;
    }

    /**
     * @brief Determine what range of cells is intersected by a geometry
     * @details x1 is guaranteed to be greater than x0, etc.
     * 
     * @param scene_bounds[in] AABB encompassing entire scene
     * @param g[in] A geometry
     * @param x0[out] coordinate
     * @param y0[out] coordinate
     * @param z0[out] coordinate
     * @param x1[out] coordinate
     * @param y1[out] coordinate
     * @param z1[out] coordinate
     */
    void geom_cell_hits(AABB scene_bounds, Geometry* g, size_t& x0, size_t& y0,
                        size_t& z0, size_t& x1, size_t& y1, size_t& z1) const {
        using namespace std;
        AABB bounds = g->bounds();
        AABB rel_bounds(bounds.xmin - scene_bounds.xmin,
                        bounds.xmax - scene_bounds.xmin);
        size_t cx0 = floor(rel_bounds.xmin.x / rx);
        size_t cy0 = floor(rel_bounds.xmin.y / ry);
        size_t cz0 = floor(rel_bounds.xmin.z / rz);
        size_t cx1 = ceil(rel_bounds.xmax.x / rx);
        size_t cy1 = ceil(rel_bounds.xmax.y / ry);
        size_t cz1 = ceil(rel_bounds.xmax.z / rz);
        x0 = min(cx0, cx1), y0 = min(cy0, cy1), z0 = min(cz0, cz1);
        x1 = max(cx0, cx1), y1 = max(cy0, cy1), z1 = max(cz0, cz1);
    }

    /**
     * @brief Counts the number of (geometry, cell) pairs
     * @details Necessary to allocate memory for geom_cell
     * 
     * @param scene_bounds AABB encompassing entire scene
     * @param b Begin iterator (random access) for list of geometry
     * @param e End iterator (random access) for list of geometry
     * @return Number of (geometry, cell) pairs
     */
    template <typename RI>
    size_t count_pairs(AABB scene_bounds, RI b, RI e) const {
        size_t pairs = 0;

        // compute geometry -> cell mappings
        while (b != e) {
            Geometry* geom = *b;
            size_t x0, y0, z0, x1, y1, z1;
            geom_cell_hits(scene_bounds, geom, x0, y0, z0, x1, y1, z1);
            pairs += (x1 - x0 + 1) * (y1 - y0 + 1) * (z1 - z0 + 1);
            
            ++b;
        }
        return pairs;
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
    size_t& first(size_t x, size_t y, size_t z) { return data[index(x, y, z)]; }

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