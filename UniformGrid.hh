#ifndef UNIFORMGRID_HH
#define UNIFORMGRID_HH

#include <cassert>
#include <cmath>
#include <cstddef>
#include <utility>
#include "AABB.hh"
#include "Geometry.hh"
#include "Vec3.hh"

typedef size_t difference_type;  // yeah
typedef std::pair<size_t, size_t> ugrid_data_t;
typedef std::pair<Geometry*, size_t> ugrid_pair_t;

class UniformGrid {
   public:
    const int3 res;
    const float3 cell_size;

   private:
    ugrid_data_t* data;
    ugrid_pair_t* geom_cell;
    const size_t n_pairs;

    size_t data_index(int3 coord) const {
        return coord.z * res.y * res.x + coord.y * res.x + coord.x;
    }

   public:
    template <typename RI>
    UniformGrid(int3 resolution, AABB scene_bounds, ugrid_data_t* data,
                ugrid_pair_t* pairs, size_t n_pairs, RI b, RI e)
        : res(resolution),
          cell_size((scene_bounds.xmax - scene_bounds.xmin) /
                    float3(resolution)),
          data(data),
          geom_cell(pairs),
          n_pairs(n_pairs) {
        using namespace std;

        // construct geometry/cell-index pairs
        ugrid_pair_t* pair = geom_cell;
        while (b != e) {
            size_t x0, y0, z0, x1, y1, z1;
            geom_cell_hits(res, scene_bounds, *b, x0, y0, z0, x1, y1, z1);
            for (size_t x = x0; x < x1; ++x) {
                for (size_t y = y0; y < y1; ++y) {
                    for (size_t z = z0; z < z1; ++z) {
                        Geometry* g = *b;
                        pair->first = g;
                        pair->second = data_index(int3(x, y, z));
                        ++pair;
                    }
                }
            }
            ++b;
        }
        assert(pair == geom_cell + n_pairs);

        // sort pairs by cell index
        auto comparator = [](const ugrid_pair_t& a,
                             const ugrid_pair_t& b) -> bool {
            return a.second < b.second;
        };
        sort(geom_cell, geom_cell + n_pairs, comparator);

        // write pair indices into grid
        size_t i = 0;
        while (i < n_pairs) {
            size_t cell_idx = geom_cell[i].second;
            assert(cell_idx < data_size(res));
            data[cell_idx].first = i;
            while (i < n_pairs && geom_cell[i].second == cell_idx)
                ++i;
            data[cell_idx].second = i;
        }
        assert(i == n_pairs);
    }

    /**
     * @brief Iterator over geometries in the UniformGrid
     */
    class iterator {
       private:
        size_t index;
        const ugrid_pair_t* gc;

       public:
        iterator(size_t index, ugrid_pair_t* gc) : index(index), gc(gc) {}
        iterator(const iterator& it) = default;

        friend bool operator==(const iterator& lhs, const iterator& rhs) {
            return lhs.index == rhs.index;
        }
        friend bool operator!=(const iterator& lhs, const iterator& rhs) {
            return !(lhs == rhs);
        }
        friend bool operator<(const iterator& lhs, const iterator& rhs) {
            return lhs.index < rhs.index;
        }
        friend bool operator>(const iterator& lhs, const iterator& rhs) {
            return lhs.index > rhs.index;
        }
        friend bool operator<=(const iterator& lhs, const iterator& rhs) {
            return lhs.index <= rhs.index;
        }
        friend bool operator>=(const iterator& lhs, const iterator& rhs) {
            return lhs.index >= rhs.index;
        }

        friend iterator operator+(const iterator& lhs, difference_type rhs) {
            iterator result(lhs);
            return result += rhs;
        }
        friend iterator operator-(const iterator& lhs, difference_type rhs) {
            iterator result(lhs);
            return result -= rhs;
        }
        friend iterator& operator+=(iterator& lhs, difference_type rhs) {
            lhs.index += rhs;
            return lhs;
        }
        friend iterator& operator-=(iterator& lhs, difference_type rhs) {
            lhs.index -= rhs;
            return lhs;
        }

        friend difference_type operator-(const iterator& lhs,
                                         const iterator& rhs) {
            return lhs.index - rhs.index;
        }

        iterator& operator++() {
            ++index;
            return *this;
        }
        iterator& operator--() {
            --index;
            return *this;
        }
        Geometry* operator*() const { return gc[index].first; }
    };

    iterator first(int3 coord) const {
        return iterator(data[data_index(coord)].first, geom_cell);
    }

    iterator last(int3 coord) const {
        return iterator(data[data_index(coord)].second, geom_cell);
    }

    /**
     * @brief Compute heuristic for grid resolution
     *
     * @param bounds bounding box for entire scene
     * @param n number of geometries in scene
     * @param density an arbitrary density factor
     * @return a grid resolution
     */
    static int3 resolution(AABB bounds, int n, int density = 5) {
        float3 d = bounds.xmax - bounds.xmin;
        float vol = d.x * d.y * d.z;
        float factor = cbrt(density * n / vol);
        d *= factor;
        return int3(ceil(d.x), ceil(d.y), ceil(d.z));
    }

    /**
     * @brief Determine how many elements must be allocated for data
     *
     * @param resolution Resolution obtained with UniformGrid::resolution()
     * @return The number of elements of ugrid_data_t to allocate
     */
    static size_t data_size(int3 resolution) {
        return (resolution.x + 1) * (resolution.y + 1) * (resolution.z + 1);
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
    static void geom_cell_hits(int3 resolution, AABB scene_bounds, Geometry* g,
                               size_t& x0, size_t& y0, size_t& z0, size_t& x1,
                               size_t& y1, size_t& z1) {
        using namespace std;
        AABB bounds = g->bounds();
        AABB rel_bounds(bounds.xmin - scene_bounds.xmin,
                        bounds.xmax - scene_bounds.xmin);
        float3 size = scene_bounds.xmax - scene_bounds.xmin;
        float3 fresolution = float3(resolution.x, resolution.y, resolution.z);
        float3 c0 = rel_bounds.xmin / size * fresolution;
        float3 c1 = rel_bounds.xmax / size * fresolution;
        size_t cx0 = floor(c0.x);
        size_t cy0 = floor(c0.y);
        size_t cz0 = floor(c0.z);
        size_t cx1 = ceil(c1.x);
        size_t cy1 = ceil(c1.y);
        size_t cz1 = ceil(c1.z);
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
    static size_t count_pairs(int3 resolution, AABB scene_bounds, RI b, RI e) {
        size_t pairs = 0;

        // compute geometry -> cell mappings
        while (b != e) {
            Geometry* geom = *b;
            size_t x0, y0, z0, x1, y1, z1;
            geom_cell_hits(resolution, scene_bounds, geom, x0, y0, z0, x1, y1,
                           z1);
            pairs += (x1 - x0) * (y1 - y0) * (z1 - z0);

            ++b;
        }
        return pairs;
    }
};

#endif