#ifndef TRANSFORM_HH
#define TRANSFORM_HH

#include "Mat.hh"
#include "Vec3.hh"

template <typename T>
Mat<T, 4, 4> transform_translate(const Vec3<T> &v) {
    Mat<T, 4, 4> result = Mat<T, 4, 4>::identity();
    result(0, 3) = v.x;
    result(1, 3) = v.y;
    result(2, 3) = v.z;
    return result;
}

template <typename T>
Mat<T, 4, 4> transform_scale(const Vec3<T> &v) {
    Mat<T, 4, 4> result;
    result(0, 0) = v.x;
    result(1, 1) = v.y;
    result(2, 2) = v.z;
    result(3, 3) = 1;
    return result;
}

template <typename T>
Mat<T, 4, 4> transform_rotateX(const T &t) {
    Mat<T, 4, 4> result = Mat<T, 4, 4>::identity();
    result(1, 1) = cos(t);
    result(1, 2) = -sin(t);
    result(2, 1) = sin(t);
    result(2, 2) = cos(t);
    return result;
}

template <typename T>
Mat<T, 4, 4> transform_rotateY(const T &t) {
    Mat<T, 4, 4> result = Mat<T, 4, 4>::identity();
    result(0, 0) = cos(t);
    result(0, 2) = -sin(t);
    result(2, 0) = sin(t);
    result(2, 2) = cos(t);
    return result;
}

template <typename T>
Mat<T, 4, 4> transform_rotateZ(const T &t) {
    Mat<T, 4, 4> result = Mat<T, 4, 4>::identity();
    result(0, 0) = cos(t);
    result(0, 1) = -sin(t);
    result(1, 0) = sin(t);
    result(1, 1) = cos(t);
    return result;
}

template <typename T>
Mat<T, 4, 4> transform_clear_translate(const Mat<T, 4, 4> &a) {
    Mat<T, 4, 4> result = a;
    result(0, 3) = 0;
    result(1, 3) = 0;
    result(2, 3) = 0;
    return result;
}

#endif