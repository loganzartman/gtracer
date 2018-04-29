#ifndef VEC3_CPU_HH
#define VEC3_CPU_HH

#include <cmath>
#include <ostream>
#include "util.hh"

template <typename T>
struct Vec3 {
    T x, y, z;

    Vec3() : x(0), y(0), z(0) {}
    Vec3(T v) : x(v), y(v), z(v) {}
    Vec3(T vx, T vy, T vz) : x(vx), y(vy), z(vz) {}

    /**
     * @brief Vector type conversion constructor
     * @details Converting between vector types is allowed (i.e. multiply
     * Float3 by Int3) but must be explicit to avoid unintended behavior.
     *
     * @param other Vector to copy
     * @tparam E Type of the other vector
     */
    template <typename E>
    explicit Vec3(const Vec3<E> &other)
        : x((T)other.x), y((T)other.y), z((T)other.z) {}

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

    friend Vec3<T> vabs(const Vec3<T> &v) {
        Vec3<T> result = v;
        if (v.x < 0)
            result.x = -v.x;
        if (v.y < 0)
            result.y = -v.y;
        if (v.z < 0)
            result.z = -v.z;
        return result;
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

    friend bool operator!=(const Vec3<T> &l, const Vec3<T> &r) {
        return !(l == r);
    }

    friend std::ostream &operator<<(std::ostream &o, const Vec3<T> &v) {
        o << "(" << v.x << ", " << v.y << ", " << v.z << ")";
        return o;
    }

    /* Other relational ops (<, <=, >, >=) are intentionally not implemented as
     * their behavior is ambiguous. */

    T length2() const { return x * x + y * y + z * z; }

    T length() const { return sqrt(length2()); }

    static Vec3<T> random_spherical() {
        Vec3<T> result;
        T phi = util::randf(0., M_PI * 2);
        T costheta = util::randf(-1., 1.);

        T theta = acos(costheta);
        result.x = sin(theta) * cos(phi);
        result.y = sin(theta) * sin(phi);
        result.z = cos(theta);
        return result;
    }
};

template <typename T>
Vec3<T> vmin(const Vec3<T> &a, const Vec3<T> &b) {
    return Vec3<T>(util::min(a.x, b.x), util::min(a.y, b.y), util::min(a.z, b.z));
}

template <typename T>
Vec3<T> vmax(const Vec3<T> &a, const Vec3<T> &b) {
    return Vec3<T>(util::max(a.x, b.x), util::max(a.y, b.y), util::max(a.z, b.z));
}

typedef Vec3<float> Float3;
typedef Vec3<int> Int3;
#endif
