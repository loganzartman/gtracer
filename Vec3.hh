#ifndef VEC3_HH
#define VEC3_HH

#include <cmath>
#include <sstream>

template <typename T>
struct Vec3 {
    T x, y, z;

    Vec3() : x(0), y(0), z(0) {}
    Vec3(T v) : x(v), y(v), z(v) {}
    Vec3(T vx, T vy, T vz) : x(vx), y(vy), z(vz) {}

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

    Vec3<T> reflect(const Vec3<T> &normal) {
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

    Vec3<T> operator*(const Vec3<T> &o) const {
        Vec3<T> result(*this);
        result.x *= o.x;
        result.y *= o.y;
        result.z *= o.z;
        return result;
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

    Vec3<T> operator-() const { return Vec3<T>(-x, -y, -z); }

    friend bool operator==(const Vec3<T> &l, const Vec3<T> &r) {
        const float tol = 1e-6f;
        return fabs(l.x - r.x) < tol && fabs(l.y - r.y) < tol &&
               fabs(l.z - r.z) < tol;
    }

    T length2() const { return x * x + y * y + z * z; }

    T length() const { return sqrt(length2()); }

    std::string print() const {
        std::ostringstream o;
        o << "(" << x << ", " << y << ", " << z << ")";
        return o.str();
    }
};

typedef Vec3<float> float3;
#endif
