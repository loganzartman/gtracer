#include <cmath>
#include "Sphere.hh"
#include "Vec3.hh"
#include "gtest/gtest.h"

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