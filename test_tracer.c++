#include "gtest/gtest.h"
#include "Vec3.hh"
#include "Sphere.hh"

TEST(TracerFixture, ray_sphere_intersect) {
	// test simple ray-sphere intersect on x-axis
	float3 ray_origin(0, 0, 0);
	float3 ray_direction(1, 0, 0);
	float3 sphere_pos(3, 0, 0);
	float sphere_radius = 1;
	Sphere s(sphere_pos, sphere_radius);
	float t0, t1;
	ASSERT_TRUE(s.intersect(ray_origin, ray_direction, t0, t1));
}

TEST(TracerFixture, ray_sphere_intersect_2) {
	// test negative direction ray-sphere intersect with slight offset
	float3 ray_origin(0, 0.1, 0);
	float3 ray_direction(-1, 0, 0);
	float3 sphere_pos(-3, 0, 0);
	float sphere_radius = 1;
	Sphere s(sphere_pos, sphere_radius);
	float t0, t1;
	ASSERT_TRUE(s.intersect(ray_origin, ray_direction, t0, t1));
}

TEST(TracerFixture, ray_sphere_intersect_3) {
	// test ray-sphere intersect with ray origin inside sphere
	float3 ray_origin(0, 0, 0);
	float3 ray_direction(1, 0, 0);
	float3 sphere_pos(0.5, 0, 0);
	float sphere_radius = 2;
	Sphere s(sphere_pos, sphere_radius);
	float t0, t1;
	ASSERT_TRUE(s.intersect(ray_origin, ray_direction, t0, t1));
}

TEST(TracerFixture, ray_sphere_intersect_4) {
	// test ray-sphere intersect with non-intersecting things
	float3 ray_origin(0, 2, 0);
	float3 ray_direction(1, 0, 0);
	float3 sphere_pos(3, 0, 0);
	float sphere_radius = 1;
	Sphere s(sphere_pos, sphere_radius);
	float t0, t1;
	ASSERT_FALSE(s.intersect(ray_origin, ray_direction, t0, t1));
}