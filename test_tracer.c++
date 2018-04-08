#include "gtest/gtest.h"
#include "Vec3.hh"
#include "Sphere.hh"

TEST(TracerFixture, ray_sphere_intersect) {
	float3 ray_origin(0, 0, 0);
	float3 ray_direction(1, 0, 0);
	float3 sphere_pos(3, 0, 0);
	float sphere_radius = 1;
	Sphere s(sphere_pos, sphere_radius);
	float t0, t1;
	ASSERT_TRUE(s.intersect(ray_origin, ray_direction, t0, t1));
}