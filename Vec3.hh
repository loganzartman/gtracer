#ifndef VEC3_HH
#define VEC3_HH

#include <sstream>

template <typename T>
struct Vec3 {
    T x, y, z;

    Vec3() : x(), y(), z() {}
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

    Vec3<T> operator-(const Vec3<T> &o) const {
        return Vec3<T>(x - o.x, y - o.y, z - o.z);
    }
    Vec3<T> operator+(const Vec3<T> &o) const {
        return Vec3<T>(x + o.x, y + o.y, z + o.z);
    }
    Vec3<T> &operator+=(const Vec3<T> &o) {
        x += o.x, y += o.y, z += o.z;
        return *this;
    }
    Vec3<T> &operator*=(const Vec3<T> &o) {
        x *= o.x, y *= o.y, z *= o.z;
        return *this;
    }
    Vec3<T> operator-() const { return Vec3<T>(-x, -y, -z); }

    friend bool operator==(const Vec3<T> &l, const Vec3<T> &r) {
        return l.x == r.x && l.y == r.y && l.z == r.z;
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
