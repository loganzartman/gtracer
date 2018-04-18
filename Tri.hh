#ifndef TRIANGLE_HH
#define TRIANGLE_HH

#include <cassert>
#include <vector>
#include <algorithm>

#include "Vec3.hh"
#include "Geometry.hh"

class Tri : public Geometry {
    float3 a, b, c;
    const Material *mat;

  public:
    Tri(const float3& a, const float3& b, const float3& c, const Material *mat = nullptr)
        : a(a), b(b), c(c), mat(mat) {}
    Tri(const std::vector<float3>& p, const Material *mat = nullptr)
        : a(p[0]), b(p[1]), c(p[2]), mat(mat) {
        assert(p.size() == 3);
    }

    // doesn't work
    bool geo_intersect(const float3& r_orig, const float3& r_dir, float& t0, float& t1) const {
        // compute plane's normal
        float3 ab = b - a; 
        float3 ac = c - a; 
        // no need to normalize
        float3 N = ab.cross(ac); // N 
        float denom = N.dot(N); 
     
        // Step 1: finding P
     
        // check if ray and plane are parallel ?
        float NdotRayDirection = N.dot(r_dir); 
        if (fabs(NdotRayDirection) < 0.00001) // almost 0 
            return false; // they are parallel so they don't intersect ! 
     
        // compute d parameter using equation 2
        float d = N.dot(a); 
     
        // compute t (equation 3)
        t0 = (N.dot(r_orig) + d) / NdotRayDirection; 
        t1 = t0;
        // check if the triangle is in behind the ray
        if (t0 < 0) return false; // the triangle is behind 
     
        // compute the intersection point using equation 1
        float3 P = r_orig + r_dir * t0; 
     
        // Step 2: inside-outside test
        float3 C; // vector perpendicular to triangle's plane 
     
        // edge 0
        float3 edge0 = b - a; 
        float3 vp0 = P - a; 
        C = edge0.cross(vp0); 
        if (N.dot(C) < 0) return false; // P is on the right side 
     
        // edge 1
        float3 edge1 = c - b; 
        float3 vp1 = P - b; 
        C = edge1.cross(vp1); 
        if (N.dot(C) < 0)  return false; // P is on the right side 
     
        // edge 2
        float3 edge2 = a - c; 
        float3 vp2 = P - c; 
        C = edge2.cross(vp2); 
        if (N.dot(C) < 0) return false; // P is on the right side; 
     
        return true; // this ray hits the triangle 
    }

    bool intersect(const float3& r_orig, const float3& r_dir, float& t0, float& t1) const {
        // edges between points
        const float3 ab = b - a; 
        const float3 ac = c - a; 

        float3 p = r_dir.cross(ac); 
        float det = ab.dot(p); 

        if (fabs(det) < 0.0001)
            return false;

        float inv_det = 1 / det; 
 
        float3 tvec = r_orig - a; 
        float u = tvec.dot(p) * inv_det; 
        if (u < 0 || u > 1)
            return false; 
     
        float3 q = tvec.cross(ab); 
        float v = r_dir.dot(q) * inv_det; 
        if (v < 0 || u + v > 1)
            return false; 
     
        t0 = ac.dot(q) * inv_det;
        t1 = t0;
     
        return true;
    }
    
    float3 normal(const float3& r_dir, const float3& intersection) const {
        const float3 ab = b - a;
        const float3 ac = c - a;

        return ab.cross(ac).normalize();
    }

    const Material *material() const { return mat; }

    AABB bounds() const {
        float3 min_corner(min(min(a, b), c));
        float3 max_corner(max(max(a, b), c));
        return AABB(min_corner, max_corner);
    }
};

#endif
