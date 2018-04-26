#ifndef VEC3_CUDA_HH
#define VEC3_CUDA_HH
// #include <helper_math.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include "util.hh"

template <typename T>
struct Vec3;
template <>
struct Vec3<float>;
template <>
struct Vec3<int>;

template <typename T>
__host__ __device__ Vec3<T> operator+(const Vec3<T> &l, const Vec3<T> &r) {
    Vec3<T> result(l);
    return result += r;
}

template <typename T>
__host__ __device__ Vec3<T> operator-(const Vec3<T> &l, const Vec3<T> &r) {
    Vec3<T> result(l);
    return result -= r;
}

template <typename T>
__host__ __device__ Vec3<T> operator*(const Vec3<T> &l, const T &r) {
    Vec3<T> result(l);
    return result *= r;
}

template <typename T>
__host__ __device__ Vec3<T> operator/(const Vec3<T> &l, const T &r) {
    Vec3<T> result(l);
    return result *= 1.f / r;
}

template <typename T>
__host__ __device__ Vec3<T> operator*(const Vec3<T> &l, const Vec3<T> &r) {
    Vec3<T> result(l);
    return result *= r;
}

template <typename T>
__host__ __device__ Vec3<T> operator/(const Vec3<T> &l, const Vec3<T> &r) {
    Vec3<T> result(l);
    return result /= r;
}

template <typename T>
__host__ __device__ Vec3<T> &operator+=(Vec3<T> &l, const Vec3<T> &r) {
    l.x += r.x, l.y += r.y, l.z += r.z;
    return l;
}

template <typename T>
__host__ __device__ Vec3<T> &operator-=(Vec3<T> &l, const Vec3<T> &r) {
    l.x -= r.x, l.y -= r.y, l.z -= r.z;
    return l;
}

template <typename T>
__host__ __device__ Vec3<T> &operator*=(Vec3<T> &l, const T &f) {
    l.x *= f, l.y *= f, l.z *= f;
    return l;
}

template <typename T>
__host__ __device__ Vec3<T> &operator*=(Vec3<T> &l, const Vec3<T> &r) {
    l.x *= r.x;
    l.y *= r.y;
    l.z *= r.z;
    return l;
}

template <typename T>
__host__ __device__ Vec3<T> &operator/=(Vec3<T> &l, const Vec3<T> &r) {
    l.x /= r.x;
    l.y /= r.y;
    l.z /= r.z;
    return l;
}

template <typename T>
__host__ __device__ Vec3<T> operator-(const Vec3<T> &v) {
    return Vec3<T>(-v.x, -v.y, -v.z);
}

template <typename T>
__host__ __device__ Vec3<T> min(const Vec3<T> &a, const Vec3<T> &b) {
    using namespace std;
    return Vec3<T>(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}

template <typename T>
__host__ __device__ Vec3<T> max(const Vec3<T> &a, const Vec3<T> &b) {
    using namespace std;
    return Vec3<T>(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}

template <>
__host__ __device__ struct Vec3<int> {
    int3 i;
    int &x = i.x;
    int &y = i.y;
    int &z = i.z;

    __host__ __device__ Vec3() : i(make_int3(0, 0, 0)) {}

    __host__ __device__ Vec3(int v) : i(make_int3(v, v, v)) {}

    __host__ __device__ Vec3(int vx, int vy, int vz)
        : i(make_int3(vx, vy, vz)) {}

    /**
     * @brief Vector type conversion constructor
     * @details Converting between vector types is allowed (i.e. multiply
     * Float3 by Int3) but must be explicit to avoid unintended behavior.
     *
     * @param other Vector to copy
     * @tparam E Type of the other vector
     */
    template <typename E>
    __host__ __device__ explicit Vec3(const Vec3<E> &other)
        : x((int)other.x), y((int)other.y), z((int)other.z) {}

    __host__ __device__ Vec3<int> &operator=(const Vec3<int> &other) {
        i = other.i;
        return *this;
    }
};

template <>
__host__ __device__ struct Vec3<float> {
    float3 f;
    float &x = f.x;
    float &y = f.y;
    float &z = f.z;

    __host__ __device__ Vec3() : f(make_float3(0, 0, 0)) {}

    __host__ __device__ Vec3(float v) : f(make_float3(v, v, v)) {}

    __host__ __device__ Vec3(float vx, float vy, float vz)
        : f(make_float3(vx, vy, vz)) {}

    /**
     * @brief Vector type conversion constructor
     * @details Converting between vector types is allowed (i.e. multiply
     * Float3 by Int3) but must be explicit to avoid unintended behavior.
     *
     * @param other Vector to copy
     * @tparam E Type of the other vector
     */
    template <typename E>
    __host__ __device__ explicit Vec3(const Vec3<E> &other)
        : x((float)other.x), y((float)other.y), z((float)other.z) {}

    __host__ __device__ Vec3<float> &operator=(const Vec3<float> &other) {
        f = other.f;
        return *this;
    }

    __host__ __device__ Vec3 &normalize() {
        float len2 = length2();
        if (len2 > 0) {
            float scalar = 1 / sqrt(len2);

            x *= scalar;
            y *= scalar;
            z *= scalar;
        }
        return *this;
    }

    __host__ __device__ float dot(const Vec3<float> &o) const {
        return x * o.x + y * o.y + z * o.z;
    }

    __host__ __device__ Vec3<float> cross(const Vec3<float> &o) const {
        Vec3<float> result;
        result.x = y * o.z - z * o.y;
        result.y = z * o.x - x * o.z;
        result.z = x * o.y - y * o.x;
        return result;
    }

    __host__ __device__ Vec3<float> reflect(const Vec3<float> &normal) const {
        return *this - normal * (2 * (this->dot(normal)));
    }

    // shouldn't use fabs because we want to keep this generic
    __host__ __device__ friend Vec3<float> vabs(const Vec3<float> &v) {
        Vec3<float> result = v;
        if (v.x < 0)
            result.x = -v.x;
        if (v.y < 0)
            result.y = -v.y;
        if (v.z < 0)
            result.z = -v.z;
        return result;
    }

    __host__ __device__ friend bool operator==(const Vec3<float> &l,
                                               const Vec3<float> &r) {
        const float tol = 1e-6f;
        return fabs(l.x - r.x) < tol && fabs(l.y - r.y) < tol &&
               fabs(l.z - r.z) < tol;
    }

    __host__ __device__ friend bool operator!=(const Vec3<float> &l,
                                               const Vec3<float> &r) {
        return !(l == r);
    }

    friend std::ostream &operator<<(std::ostream &o, const Vec3<float> &v) {
        o << "(" << v.x << ", " << v.y << ", " << v.z << ")";
        return o;
    }

    /* Other relational ops (<, <=, >, >=) are intentionally not implemented as
     * their behavior is ambiguous. */

    __host__ __device__ float length2() const { return x * x + y * y + z * z; }

    __host__ __device__ float length() const { return sqrt(length2()); }

    __host__ __device__ static Vec3<float> random_spherical() {
        Vec3<float> result;
        float phi = randf(0., M_PI * 2);
        float costheta = randf(-1., 1.);

        float theta = acos(costheta);
        result.x = sin(theta) * cos(phi);
        result.y = sin(theta) * sin(phi);
        result.z = cos(theta);
        return result;
    }
};

typedef Vec3<float> Float3;
typedef Vec3<int> Int3;
#endif
