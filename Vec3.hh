#ifndef VEC3_HH
#define VEC3_HH

#include <algorithm>
#include <cmath>
#include <sstream>
#include "util.hh"

template <typename T>
struct Vec3 {
    friend Vec3<T> min(const Vec3<T> &a, const Vec3<T> &b) {
        using namespace std;
        return Vec3<T>(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
    }

    friend Vec3<T> max(const Vec3<T> &a, const Vec3<T> &b) {
        using namespace std;
        return Vec3<T>(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
    }

    T x, y, z;

    Vec3() : x(0), y(0), z(0) {}
    Vec3(T v) : x(v), y(v), z(v) {}
    Vec3(T vx, T vy, T vz) : x(vx), y(vy), z(vz) {}

    template <typename E>
    Vec3(Vec3<E> other) : x((T)other.x), y((T)other.y), z((T)other.z) {}

    Vec3 &normalize() {
        T len2 = length2();
        if (len2 > 0) {
            T scalar = 1 / sqrt(len2);

            x *= scalar;
            y *= scalar;
            z *= scalar;
        }
        return *this;
    }

    T dot(const Vec3<T> &o) const { return x * o.x + y * o.y + z * o.z; }

    Vec3<T> cross(const Vec3<T> &o) const {
        Vec3<T> result;
        result.x = y * o.z - z * o.y;
        result.y = z * o.x - x * o.z;
        result.z = x * o.y - y * o.x;
        return result;
    }

    // shouldn't use fabs because we want to keep this generic
    friend void vabs(Vec3 &v) {
        if (v.x < 0)
            v.x = -v.x;
        if (v.y < 0)
            v.y = -v.y;
        if (v.z < 0)
            v.z = -v.z;
    }

    Vec3<T> reflect(const Vec3<T> &normal) const {
        return *this - normal * (2 * (this->dot(normal)));
    }

    Vec3<T> operator+(const Vec3<T> &o) const {
        Vec3<T> result(*this);
        return result += o;
    }

    Vec3<T> operator-(const Vec3<T> &o) const {
        Vec3<T> result(*this);
        return result -= o;
    }

    Vec3<T> operator*(const T &o) const {
        Vec3<T> result(*this);
        return result *= o;
    }

    Vec3<T> operator/(const T &o) const {
        Vec3<T> result(*this);
        return result *= 1.f / o;
    }

    Vec3<T> operator*(const Vec3<T> &o) const {
        Vec3<T> result(*this);
        return result *= o;
    }

    Vec3<T> operator/(const Vec3<T> &o) const {
        Vec3<T> result(*this);
        return result /= o;
    }

    Vec3<T> &operator+=(const Vec3<T> &o) {
        x += o.x, y += o.y, z += o.z;
        return *this;
    }

    Vec3<T> &operator-=(const Vec3<T> &o) {
        x -= o.x, y -= o.y, z -= o.z;
        return *this;
    }

    Vec3<T> &operator*=(const T &f) {
        x *= f, y *= f, z *= f;
        return *this;
    }

    Vec3<T> &operator*=(const Vec3<T> &o) {
        this->x *= o.x;
        this->y *= o.y;
        this->z *= o.z;
        return *this;
    }

    Vec3<T> &operator/=(const Vec3<T> &o) {
        this->x /= o.x;
        this->y /= o.y;
        this->z /= o.z;
        return *this;
    }

    Vec3<T> operator-() const { return Vec3<T>(-x, -y, -z); }

    friend bool operator==(const Vec3<T> &l, const Vec3<T> &r) {
        const float tol = 1e-6f;
        return fabs(l.x - r.x) < tol && fabs(l.y - r.y) < tol &&
               fabs(l.z - r.z) < tol;
    }

    friend bool operator<(const Vec3<T> &l, const Vec3<T> &r) {
        return l.x < r.x && l.y < r.y && l.z < r.z;
    }

    friend bool operator>(const Vec3<T> &l, const Vec3<T> &r) {
        return l.x > r.x && l.y > r.y && l.z > r.z;
    }

    friend bool operator<=(const Vec3<T> &l, const Vec3<T> &r) {
        return l.x <= r.x && l.y <= r.y && l.z <= r.z;
    }

    friend bool operator>=(const Vec3<T> &l, const Vec3<T> &r) {
        return l.x >= r.x && l.y >= r.y && l.z >= r.z;
    }

    T length2() const { return x * x + y * y + z * z; }

    T length() const { return sqrt(length2()); }

    static Vec3<T> random_spherical() {
        Vec3<T> result;
        T phi = randf(0., M_PI * 2);
        T costheta = randf(-1., 1.);

        T theta = acos(costheta);
        result.x = sin(theta) * cos(phi);
        result.y = sin(theta) * sin(phi);
        result.z = cos(theta);
        return result;
    }

    std::string print() const {
        std::ostringstream o;
        o << "(" << x << ", " << y << ", " << z << ")";
        return o.str();
    }
};

typedef Vec3<float> float3;
typedef Vec3<int> int3;
#endif
