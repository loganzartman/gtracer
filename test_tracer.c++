#include <cmath>
#include <vector>
#include "Mat.hh"
#include "Sphere.hh"
#include "Vec3.hh"
#include "gtest/gtest.h"
#include "render.hh"

TEST(Vec3Test, ctor) {
    float3 v(1, 2, 3);
    ASSERT_FLOAT_EQ(v.x, 1);
    ASSERT_FLOAT_EQ(v.y, 2);
    ASSERT_FLOAT_EQ(v.z, 3);
}

TEST(Vec3Test, add) {
    float3 v(1, 2, 3);
    float3 u(-1, -2, -3);
    float3 result = v + u;
    ASSERT_FLOAT_EQ(result.x, 0);
    ASSERT_FLOAT_EQ(result.y, 0);
    ASSERT_FLOAT_EQ(result.z, 0);
}

TEST(Vec3Test, sub) {
    float3 v(10, 20, 30);
    float3 u(1, 2, 3);
    float3 result = v - u;
    ASSERT_FLOAT_EQ(result.x, 9);
    ASSERT_FLOAT_EQ(result.y, 18);
    ASSERT_FLOAT_EQ(result.z, 27);
}

TEST(Vec3Test, dot) {
    float3 v(1, -2.5, 3);
    float3 u(4, 5, -6);
    float result = v.dot(u);
    ASSERT_FLOAT_EQ(result, -26.5);
}

TEST(Vec3Test, reflect) {
    float3 v(1, 1, 0);
    float3 n(0, -1, 0);
    float3 result = v.reflect(n);
    ASSERT_FLOAT_EQ(result.x, 1);
    ASSERT_FLOAT_EQ(result.y, -1);
    ASSERT_FLOAT_EQ(result.z, 0);
}

TEST(Vec3Test, normalize) {
    float3 v(2, 4, -8);
    v.normalize();
    ASSERT_FLOAT_EQ(v.x, 1 / sqrt(21));
    ASSERT_FLOAT_EQ(v.y, 2 / sqrt(21));
    ASSERT_FLOAT_EQ(v.z, -4 / sqrt(21));
}

TEST(Vec3Test, eq) {
    // test for good floating point equality check
    float3 v(0);
    for (int i = 0; i < 10; ++i)
        v += float3(0.1);
    float3 u(1);
    ASSERT_TRUE(v == u);
}

TEST(Vec3Test, eq_2) {
    // test for good floating point equality check
    float3 v(0);
    for (int i = 0; i < 10; ++i)
        v += float3(0.09999);
    float3 u(1);
    ASSERT_FALSE(v == u);
}

TEST(Vec3Test, length) {
    float3 v(1, 1, 1);
    ASSERT_FLOAT_EQ(v.length(), 1 * sqrt(3));
}

TEST(RaySphereTest, intersect) {
    // test simple ray-sphere intersect on x-axis
    float3 ray_origin(0, 0, 0);
    float3 ray_direction(1, 0, 0);
    float3 sphere_pos(3, 0, 0);
    float sphere_radius = 1;
    Sphere s(sphere_pos, sphere_radius);
    float t0, t1;
    ASSERT_TRUE(s.intersect(ray_origin, ray_direction, t0, t1));
}

TEST(RaySphereTest, intersect_negative) {
    // test negative direction ray-sphere intersect with slight offset
    float3 ray_origin(0, 0.1, 0);
    float3 ray_direction(-1, 0, 0);
    float3 sphere_pos(-3, 0, 0);
    float sphere_radius = 1;
    Sphere s(sphere_pos, sphere_radius);
    float t0, t1;
    ASSERT_TRUE(s.intersect(ray_origin, ray_direction, t0, t1));
}

TEST(RaySphereTest, intersect_inside) {
    // test ray-sphere intersect with ray origin inside sphere
    float3 ray_origin(0, 0, 0);
    float3 ray_direction(1, 0, 0);
    float3 sphere_pos(0.5, 0, 0);
    float sphere_radius = 2;
    Sphere s(sphere_pos, sphere_radius);
    float t0, t1;
    ASSERT_TRUE(s.intersect(ray_origin, ray_direction, t0, t1));
}

TEST(RaySphereTest, no_intersection) {
    // test ray-sphere intersect with non-intersecting things
    float3 ray_origin(0, 2, 0);
    float3 ray_direction(1, 0, 0);
    float3 sphere_pos(3, 0, 0);
    float sphere_radius = 1;
    Sphere s(sphere_pos, sphere_radius);
    float t0, t1;
    ASSERT_FALSE(s.intersect(ray_origin, ray_direction, t0, t1));
}

TEST(CPURayIntersect, simple) {
    float3 ray_orig(0, 0, 0);
    float3 ray_dir(1, 0, 0);
    Sphere sphere(float3(2, 0, 0), 1, float3());
    std::vector<Sphere> spheres{sphere};

    float3 intersection;
    Sphere *hit_sphere;
    ASSERT_TRUE(cpu_ray_intersect(ray_orig, ray_dir, spheres, intersection,
                                  hit_sphere));
    ASSERT_EQ(intersection, float3(1, 0, 0));
    ASSERT_EQ(hit_sphere, &spheres[0]);
}

TEST(CPURayIntersect, negative) {
    float3 ray_orig(0, 0, 0);
    float3 ray_dir(-1, 0, 0);
    Sphere sphere(float3(-2, 0, 0), 1, float3());
    std::vector<Sphere> spheres{sphere};

    float3 intersection;
    Sphere *hit_sphere;
    ASSERT_TRUE(cpu_ray_intersect(ray_orig, ray_dir, spheres, intersection,
                                  hit_sphere));
    ASSERT_EQ(intersection, float3(-1, 0, 0));
    ASSERT_EQ(hit_sphere, &spheres[0]);
}

TEST(CPURayIntersect, negative_2) {
    float3 ray_orig(-2, 0, 0);
    float3 ray_dir(1, 0, 0);
    Sphere sphere(float3(0, 0, 0), 1, float3());
    std::vector<Sphere> spheres{sphere};

    float3 intersection;
    Sphere *hit_sphere;
    ASSERT_TRUE(cpu_ray_intersect(ray_orig, ray_dir, spheres, intersection,
                                  hit_sphere));
    ASSERT_EQ(intersection, float3(-1, 0, 0));
    ASSERT_EQ(hit_sphere, &spheres[0]);
}

TEST(CPURayIntersect, negative_3) {
    float3 ray_orig(1, 0, 0);
    float3 ray_dir(-1, 0, 0);
    Sphere sphere(float3(-3, 0, 0), 1, float3());
    std::vector<Sphere> spheres{sphere};

    float3 intersection;
    Sphere *hit_sphere;
    ASSERT_TRUE(cpu_ray_intersect(ray_orig, ray_dir, spheres, intersection,
                                  hit_sphere));
    ASSERT_EQ(intersection, float3(-2, 0, 0));
    ASSERT_EQ(hit_sphere, &spheres[0]);
}

TEST(CPURayIntersect, two_negative) {
    float3 ray_orig(0, 0, 0);
    float3 ray_dir(-1, 0, 0);
    Sphere sphere1(float3(-2, 0, 0), 1, float3());
    Sphere sphere2(float3(-4, 0, 0), 1, float3());
    std::vector<Sphere> spheres{sphere1, sphere2};

    float3 intersection;
    Sphere *hit_sphere;
    ASSERT_TRUE(cpu_ray_intersect(ray_orig, ray_dir, spheres, intersection,
                                  hit_sphere));
    ASSERT_EQ(intersection, float3(-1, 0, 0));
    ASSERT_EQ(hit_sphere, &spheres[0]);
}

TEST(CPURayIntersect, two_negative_2) {
    float3 ray_orig(4, 0, 0);
    float3 ray_dir(-1, 0, 0);
    Sphere sphere1(float3(-2, 1.1, 0), 1, float3());
    Sphere sphere2(float3(-4, 0, 0), 1, float3());
    std::vector<Sphere> spheres{sphere1, sphere2};

    float3 intersection;
    Sphere *hit_sphere;
    ASSERT_TRUE(cpu_ray_intersect(ray_orig, ray_dir, spheres, intersection,
                                  hit_sphere));
    ASSERT_EQ(intersection, float3(-3, 0, 0));
    ASSERT_EQ(hit_sphere, &spheres[1]);
}

TEST(CPURayIntersect, two_negative_inside) {
    float3 ray_orig(-1.5, 0, 0);
    float3 ray_dir(-1, 0, 0);
    Sphere sphere1(float3(-2, 0, 0), 1, float3());
    Sphere sphere2(float3(-4, 0, 0), 1, float3());
    std::vector<Sphere> spheres{sphere1, sphere2};

    float3 intersection;
    Sphere *hit_sphere;
    ASSERT_TRUE(cpu_ray_intersect(ray_orig, ray_dir, spheres, intersection,
                                  hit_sphere));
    ASSERT_EQ(intersection, float3(-3, 0, 0));
    ASSERT_EQ(hit_sphere, &spheres[0]);
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