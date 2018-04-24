#include <cmath>
#include <cstddef>
#include <vector>
#include "AABB.hh"
#include "Box.hh"
#include "Geometry.hh"
#include "Mat.hh"
#include "Sphere.hh"
#include "Tri.hh"
#include "UniformGrid.hh"
#include "Vec3.hh"
#include "gtest/gtest.h"
#include "render.hh"

TEST(Vec3Test, ctor) {
    Float3 v(1, 2, 3);
    ASSERT_FLOAT_EQ(v.x, 1);
    ASSERT_FLOAT_EQ(v.y, 2);
    ASSERT_FLOAT_EQ(v.z, 3);
}

TEST(Vec3Test, add) {
    Float3 v(1, 2, 3);
    Float3 u(-1, -2, -3);
    Float3 result = v + u;
    ASSERT_FLOAT_EQ(result.x, 0);
    ASSERT_FLOAT_EQ(result.y, 0);
    ASSERT_FLOAT_EQ(result.z, 0);
}

TEST(Vec3Test, sub) {
    Float3 v(10, 20, 30);
    Float3 u(1, 2, 3);
    Float3 result = v - u;
    ASSERT_FLOAT_EQ(result.x, 9);
    ASSERT_FLOAT_EQ(result.y, 18);
    ASSERT_FLOAT_EQ(result.z, 27);
}

TEST(Vec3Test, dot) {
    Float3 v(1, -2.5, 3);
    Float3 u(4, 5, -6);
    float result = v.dot(u);
    ASSERT_FLOAT_EQ(result, -26.5);
}

TEST(Vec3Test, reflect) {
    Float3 v(1, 1, 0);
    Float3 n(0, -1, 0);
    Float3 result = v.reflect(n);
    ASSERT_FLOAT_EQ(result.x, 1);
    ASSERT_FLOAT_EQ(result.y, -1);
    ASSERT_FLOAT_EQ(result.z, 0);
}

TEST(Vec3Test, normalize) {
    Float3 v(2, 4, -8);
    v.normalize();
    ASSERT_FLOAT_EQ(v.x, 1 / sqrt(21));
    ASSERT_FLOAT_EQ(v.y, 2 / sqrt(21));
    ASSERT_FLOAT_EQ(v.z, -4 / sqrt(21));
}

TEST(Vec3Test, eq) {
    // test for good floating point equality check
    Float3 v(0);
    for (int i = 0; i < 10; ++i)
        v += Float3(0.1);
    Float3 u(1);
    ASSERT_TRUE(v == u);
}

TEST(Vec3Test, eq_2) {
    // test for good floating point equality check
    Float3 v(0);
    for (int i = 0; i < 10; ++i)
        v += Float3(0.09999);
    Float3 u(1);
    ASSERT_FALSE(v == u);
}

TEST(Vec3Test, length) {
    Float3 v(1, 1, 1);
    ASSERT_FLOAT_EQ(v.length(), 1 * sqrt(3));
}

TEST(Vec3Test, cross_1) {
    Float3 a(0, 1, 1);
    Float3 b(1, -1, 3);
    ASSERT_EQ(a.cross(b), Float3(4, 1, -1));
}

TEST(RaySphereTest, intersect) {
    // test simple ray-sphere intersect on x-axis
    Float3 ray_origin(0, 0, 0);
    Float3 ray_direction(1, 0, 0);
    Float3 sphere_pos(3, 0, 0);
    float sphere_radius = 1;
    Sphere s(sphere_pos, sphere_radius);
    float t;
    ASSERT_TRUE(s.intersect(ray_origin, ray_direction, t));
}

TEST(RaySphereTest, intersect_negative) {
    // test negative direction ray-sphere intersect with slight offset
    Float3 ray_origin(0, 0.1, 0);
    Float3 ray_direction(-1, 0, 0);
    Float3 sphere_pos(-3, 0, 0);
    float sphere_radius = 1;
    Sphere s(sphere_pos, sphere_radius);
    float t;
    ASSERT_TRUE(s.intersect(ray_origin, ray_direction, t));
}

TEST(RaySphereTest, intersect_inside) {
    // test ray-sphere intersect with ray origin inside sphere
    Float3 ray_origin(0, 0, 0);
    Float3 ray_direction(1, 0, 0);
    Float3 sphere_pos(0.5, 0, 0);
    float sphere_radius = 2;
    Sphere s(sphere_pos, sphere_radius);
    float t;
    ASSERT_TRUE(s.intersect(ray_origin, ray_direction, t));
}

TEST(RaySphereTest, no_intersection) {
    // test ray-sphere intersect with non-intersecting things
    Float3 ray_origin(0, 2, 0);
    Float3 ray_direction(1, 0, 0);
    Float3 sphere_pos(3, 0, 0);
    float sphere_radius = 1;
    Sphere s(sphere_pos, sphere_radius);
    float t;
    ASSERT_FALSE(s.intersect(ray_origin, ray_direction, t));
}

TEST(CPURayIntersect, simple) {
    Float3 ray_orig(0, 0, 0);
    Float3 ray_dir(1, 0, 0);
    Sphere sphere(Float3(2, 0, 0), 1);
    std::vector<Geometry *> geoms{&sphere};

    Float3 intersection;
    Geometry *hit_geom;
    ASSERT_TRUE(cpu_ray_intersect_items(ray_orig, ray_dir, geoms.begin(),
                                        geoms.end(), intersection, hit_geom));
    ASSERT_EQ(intersection, Float3(1, 0, 0));
    ASSERT_EQ(hit_geom, geoms[0]);
}

TEST(CPURayIntersect, negative) {
    Float3 ray_orig(0, 0, 0);
    Float3 ray_dir(-1, 0, 0);
    Sphere sphere(Float3(-2, 0, 0), 1);
    std::vector<Geometry *> geoms{&sphere};

    Float3 intersection;
    Geometry *hit_geom;
    ASSERT_TRUE(cpu_ray_intersect_items(ray_orig, ray_dir, geoms.begin(),
                                        geoms.end(), intersection, hit_geom));
    ASSERT_EQ(intersection, Float3(-1, 0, 0));
    ASSERT_EQ(hit_geom, geoms[0]);
}

TEST(CPURayIntersect, negative_2) {
    Float3 ray_orig(-2, 0, 0);
    Float3 ray_dir(1, 0, 0);
    Sphere sphere(Float3(0, 0, 0), 1);
    std::vector<Geometry *> geoms{&sphere};

    Float3 intersection;
    Geometry *hit_geom;
    ASSERT_TRUE(cpu_ray_intersect_items(ray_orig, ray_dir, geoms.begin(),
                                        geoms.end(), intersection, hit_geom));
    ASSERT_EQ(intersection, Float3(-1, 0, 0));
    ASSERT_EQ(hit_geom, geoms[0]);
}

TEST(CPURayIntersect, negative_3) {
    Float3 ray_orig(1, 0, 0);
    Float3 ray_dir(-1, 0, 0);
    Sphere sphere(Float3(-3, 0, 0), 1);
    std::vector<Geometry *> geoms{&sphere};

    Float3 intersection;
    Geometry *hit_geom;
    ASSERT_TRUE(cpu_ray_intersect_items(ray_orig, ray_dir, geoms.begin(),
                                        geoms.end(), intersection, hit_geom));
    ASSERT_EQ(intersection, Float3(-2, 0, 0));
    ASSERT_EQ(hit_geom, geoms[0]);
}

TEST(CPURayIntersect, two_negative) {
    Float3 ray_orig(0, 0, 0);
    Float3 ray_dir(-1, 0, 0);
    Sphere sphere1(Float3(-2, 0, 0), 1);
    Sphere sphere2(Float3(-4, 0, 0), 1);
    std::vector<Geometry *> geoms{&sphere1, &sphere2};

    Float3 intersection;
    Geometry *hit_geom;
    ASSERT_TRUE(cpu_ray_intersect_items(ray_orig, ray_dir, geoms.begin(),
                                        geoms.end(), intersection, hit_geom));
    ASSERT_EQ(intersection, Float3(-1, 0, 0));
    ASSERT_EQ(hit_geom, geoms[0]);
}

TEST(CPURayIntersect, two_negative_2) {
    Float3 ray_orig(4, 0, 0);
    Float3 ray_dir(-1, 0, 0);
    Sphere sphere1(Float3(-2, 1.1, 0), 1);
    Sphere sphere2(Float3(-4, 0, 0), 1);
    std::vector<Geometry *> geoms{&sphere1, &sphere2};

    Float3 intersection;
    Geometry *hit_geom;
    ASSERT_TRUE(cpu_ray_intersect_items(ray_orig, ray_dir, geoms.begin(),
                                        geoms.end(), intersection, hit_geom));
    ASSERT_EQ(intersection, Float3(-3, 0, 0));
    ASSERT_EQ(hit_geom, geoms[1]);
}

TEST(CPURayIntersect, two_negative_inside) {
    Float3 ray_orig(-1.5, 0, 0);
    Float3 ray_dir(-1, 0, 0);
    Sphere sphere1(Float3(-2, 0, 0), 1);
    Sphere sphere2(Float3(-4, 0, 0), 1);
    std::vector<Geometry *> geoms{&sphere1, &sphere2};

    Float3 intersection;
    Geometry *hit_geom;
    ASSERT_TRUE(cpu_ray_intersect_items(ray_orig, ray_dir, geoms.begin(),
                                        geoms.end(), intersection, hit_geom));
    ASSERT_EQ(intersection, Float3(-3, 0, 0));
    ASSERT_EQ(hit_geom, geoms[0]);
}

TEST(MatTest, ctor) {
    Mat<float, 4, 4> m;
    ASSERT_TRUE(true);
}

TEST(MatTest, ctor_index) {
    Mat<float, 4, 4> m;
    ASSERT_FLOAT_EQ(m(0, 0), 0);
}

TEST(MatTest, ctor_index_2) {
    Mat<float, 2, 2> m({1.f, 2.f, 3.f, 4.f});
    ASSERT_FLOAT_EQ(m(0, 0), 1);
    ASSERT_FLOAT_EQ(m(0, 1), 2);
    ASSERT_FLOAT_EQ(m(1, 0), 3);
    ASSERT_FLOAT_EQ(m(1, 1), 4);
}

TEST(MatTest, add) {
    Mat<float, 2, 2> a;
    Mat<float, 2, 2> b({1.f, 2.f, 3.f, 4.f});
    Mat<float, 2, 2> c = a + b;
    ASSERT_FLOAT_EQ(c(0, 0), 1);
    ASSERT_FLOAT_EQ(c(0, 1), 2);
    ASSERT_FLOAT_EQ(c(1, 0), 3);
    ASSERT_FLOAT_EQ(c(1, 1), 4);
}

TEST(MatTest, multiply) {
    Mat<float, 2, 3> a({1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    Mat<float, 3, 2> b({7.f, 8.f, 9.f, 10.f, 11.f, 12.f});
    Mat<float, 2, 2> c = a * b;
    ASSERT_FLOAT_EQ(c(0, 0), 58);
    ASSERT_FLOAT_EQ(c(0, 1), 64);
    ASSERT_FLOAT_EQ(c(1, 0), 139);
    ASSERT_FLOAT_EQ(c(1, 1), 154);
}

TEST(MatTest, multiply_vec3) {
    Mat<float, 4, 4> a = Mat<float, 4, 4>::identity();
    Vec3<float> b(1, 2, 3);
    Vec3<float> result = a * b;
    ASSERT_FLOAT_EQ(result.x, 1);
    ASSERT_FLOAT_EQ(result.y, 2);
    ASSERT_FLOAT_EQ(result.z, 3);
}

TEST(BoxTest, intersect_box_1) {
    Box a(Float3(0), Float3(1));
    Box b(Float3(2), Float3(3));
    ASSERT_FALSE(a.intersect(b));
}

TEST(BoxTest, intersect_box_2) {
    Box a(Float3(0), Float3(1));
    Box b(Float3(0.5), Float3(1));
    ASSERT_TRUE(a.intersect(b));
}

TEST(BoxTest, intersect_box_3) {
    Box a(Float3(0), Float3(1));
    Box b(Float3(0), Float3(1));
    ASSERT_TRUE(a.intersect(b));
}

TEST(BoxTest, intersect_box_4) {
    Box a(Float3(0), Float3(1));
    Box b(Float3(0.1), Float3(0.9));
    ASSERT_TRUE(a.intersect(b));
}

TEST(BoxTest, intersect_ray_1) {
    Box a(Float3(1), Float3(2));
    Float3 r_orig(0);
    Float3 r_dir(1);
    float t;
    ASSERT_TRUE(a.intersect(r_orig, r_dir, t));
}

TEST(BoxTest, intersect_ray_2) {
    Box a(Float3(1), Float3(2));
    Float3 r_orig(0, 2, 0);
    Float3 r_dir(1);
    float t;
    ASSERT_FALSE(a.intersect(r_orig, r_dir, t));
}

TEST(TriTest, intersect_ray_1) {
    Tri a(Float3(0), Float3(1, 0, 0), Float3(1, 0, 2));
    Float3 r_orig(0.5, 1, 0.5);
    Float3 r_dir(0, -1, 0);
    float t;
    ASSERT_TRUE(a.intersect(r_orig, r_dir, t));
    ASSERT_EQ(t, 1);
}

TEST(TriTest, intersect_ray_2) {
    Tri a(Float3(0), Float3(1, 0, 0), Float3(1, 0, 2));
    Float3 r_orig(-1.5, 1, -1.5);
    Float3 r_dir(1, -0.5, 1);
    float t;
    ASSERT_TRUE(a.intersect(r_orig, r_dir, t));
    ASSERT_EQ(t, 2);
}

TEST(TriTest, cpu_ray_intersect) {
    Float3 ray_orig(0, 2, 0);
    Float3 ray_dir(0, -1, 0);
    Tri tri1(Float3(0), Float3(1, 0, 0), Float3(1, 0, 2));
    Tri tri2(Float3(0, 1, 0), Float3(1, 1, 0), Float3(1, 1, 2));
    std::vector<Geometry *> geoms{&tri1, &tri2};

    Float3 intersection;
    Geometry *hit_geom;
    ASSERT_TRUE(cpu_ray_intersect_items(ray_orig, ray_dir, geoms.begin(),
                                        geoms.end(), intersection, hit_geom));
    ASSERT_EQ(intersection, Float3(0, 1, 0));
    ASSERT_EQ(hit_geom, geoms[1]);
}

TEST(UniformGridTest, grid_resolution) {
    AABB bounds(Float3(0), Float3(1, 2, 3));
    Int3 res = UniformGrid::resolution(bounds, 2, 6);
    ASSERT_EQ(res.x, ceil(cbrt(2) * 1));
    ASSERT_EQ(res.y, ceil(cbrt(2) * 2));
    ASSERT_EQ(res.z, ceil(cbrt(2) * 3));
}

TEST(UniformGridTest, grid_count_pairs) {
    AABB world_bounds(Float3(0), Float3(1));
    Sphere s(Float3(0.5), 0.2);
    std::vector<Geometry *> geom{&s};

    Int3 res = UniformGrid::resolution(world_bounds, 1, 64);
    ASSERT_EQ(res.x, 4);
    ASSERT_EQ(res.y, 4);
    ASSERT_EQ(res.z, 4);

    size_t pairs =
        UniformGrid::count_pairs(res, world_bounds, geom.begin(), geom.end());
    ASSERT_EQ(pairs, 8);
}

TEST(UniformGridTest, acceptance) {
    // build scene
    Sphere sphere1(Float3(-2, 0, 0), 0.5f);
    Sphere sphere2(Float3(2, 0, 0), 0.5f);
    std::vector<Geometry *> geom{&sphere1, &sphere2};

    // compute resolution
    AABB world_bounds = geometry_bounds(geom.begin(), geom.end());
    Int3 res = UniformGrid::resolution(world_bounds, geom.size());
    ASSERT_EQ(res.x, 7);
    ASSERT_EQ(res.y, 2);
    ASSERT_EQ(res.z, 2);

    // compute space requirements
    size_t n_data = UniformGrid::data_size(res);
    ASSERT_EQ(n_data, 72);
    size_t n_pairs =
        UniformGrid::count_pairs(res, world_bounds, geom.begin(), geom.end());
    ASSERT_EQ(n_pairs, 16);

    // allocate memory
    ugrid_data_t *grid_data = new ugrid_data_t[n_data];
    ugrid_pair_t *grid_pairs = new ugrid_pair_t[n_pairs];

    // build grid
    UniformGrid grid(res, world_bounds, grid_data, grid_pairs, n_pairs,
                     geom.begin(), geom.end());

    // test lookup of sphere1
    {
        Int3 cell = Int3(0, 0, 0);
        auto b = grid.first(cell);
        auto e = grid.last(cell);
        ASSERT_EQ(e - b, 1);
        ASSERT_EQ(*b, geom[0]);
    }

    // test lookup of sphere2
    {
        Int3 cell = Int3(5, 0, 0);
        auto b = grid.first(cell);
        auto e = grid.last(cell);
        ASSERT_EQ(e - b, 1);
        ASSERT_EQ(*b, geom[1]);
    }

    // test lookup of nothing
    {
        Int3 cell = Int3(3, 0, 0);
        auto b = grid.first(cell);
        auto e = grid.last(cell);
        ASSERT_EQ(e - b, 0);
    }

    // test lookup of sphere2
    {
        Int3 cell = Int3(6, 1, 1);
        auto b = grid.first(cell);
        auto e = grid.last(cell);
        ASSERT_EQ(e - b, 1);
        ASSERT_EQ(*b, geom[1]);
    }

    delete[] grid_data;
    delete[] grid_pairs;
}