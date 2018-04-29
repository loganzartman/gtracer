#ifndef TRANSFORM_HH
#define TRANSFORM_HH

#include "Mat.hh"
#include "Vec3.hh"
#include "util.hh"

/**
 * @brief Produce a transformation matrix for a given translation.
 *
 * @param v The translation vector
 * @return The resulting transformation matrix
 */
template <typename T>
HOSTDEV Mat<T, 4, 4> transform_translate(const Vec3<T> &v) {
    Mat<T, 4, 4> result = Mat<T, 4, 4>::identity();
    result(0, 3) = v.x;
    result(1, 3) = v.y;
    result(2, 3) = v.z;
    return result;
}

/**
 * @brief Produce a transformation matrix for a given scale.
 *
 * @param v The scale vector, where components represent the scale on each axis
 * @return The resulting transformation matrix.
 */
template <typename T>
HOSTDEV Mat<T, 4, 4> transform_scale(const Vec3<T> &v) {
    Mat<T, 4, 4> result;
    result(0, 0) = v.x;
    result(1, 1) = v.y;
    result(2, 2) = v.z;
    result(3, 3) = 1;
    return result;
}

/**
 * @brief Produce a transformation matrix for a given X rotation.
 * @details Rotating individual components is helpful when you want to control
 * the order of rotations, which is necessary in some cases.
 *
 * @param t the rotation amount in radians
 * @return The resulting transformation matrix.
 */
template <typename T>
HOSTDEV Mat<T, 4, 4> transform_rotateX(const T &t) {
    Mat<T, 4, 4> result = Mat<T, 4, 4>::identity();
    result(1, 1) = cos(t);
    result(1, 2) = -sin(t);
    result(2, 1) = sin(t);
    result(2, 2) = cos(t);
    return result;
}

/**
 * @brief Produce a transformation matrix for a given Y rotation.
 * @details Rotating individual components is helpful when you want to control
 * the order of rotations, which is necessary in some cases.
 *
 * @param t the rotation amount in radians
 * @return The resulting transformation matrix.
 */
template <typename T>
HOSTDEV Mat<T, 4, 4> transform_rotateY(const T &t) {
    Mat<T, 4, 4> result = Mat<T, 4, 4>::identity();
    result(0, 0) = cos(t);
    result(0, 2) = -sin(t);
    result(2, 0) = sin(t);
    result(2, 2) = cos(t);
    return result;
}

/**
 * @brief Produce a transformation matrix for a given Z rotation.
 * @details Rotating individual components is helpful when you want to control
 * the order of rotations, which is necessary in some cases.
 *
 * @param t the rotation amount in radians
 * @return The resulting transformation matrix.
 */
template <typename T>
HOSTDEV Mat<T, 4, 4> transform_rotateZ(const T &t) {
    Mat<T, 4, 4> result = Mat<T, 4, 4>::identity();
    result(0, 0) = cos(t);
    result(0, 1) = -sin(t);
    result(1, 0) = sin(t);
    result(1, 1) = cos(t);
    return result;
}

/**
 * @brief Produce a transformation matrix for a given rotation.
 * @details The rotation is performed in the order X, Y, Z. If you need a
 * particular order other than this, use the other rotation functions.
 *
 * @param v The rotation vector, where each component is radians on that axis
 * @return The resulting transformation matrix
 */
template <typename T>
HOSTDEV Mat<T, 4, 4> transform_rotate(const Vec3<T> &v) {
    Mat<T, 4, 4> result = Mat<T, 4, 4>::identity();
    result(0, 0) = cos(v.y) * cos(v.z);
    result(0, 1) = cos(v.y) * sin(v.z);
    result(0, 2) = -sin(v.y);

    result(1, 0) = sin(v.x) * sin(v.y) * cos(v.z) - cos(v.x) * sin(v.z);
    result(1, 1) = sin(v.x) * sin(v.y) * sin(v.z) + cos(v.x) * cos(v.z);
    result(1, 2) = sin(v.x) * cos(v.y);

    result(2, 0) = cos(v.x) * sin(v.y) * cos(v.z) + sin(v.x) * sin(v.z);
    result(2, 1) = cos(v.x) * sin(v.y) * sin(v.z) - sin(v.x) * cos(v.z);
    result(2, 2) = cos(v.x) * cos(v.y);
    return result;
}

/**
 * @brief Produce a copy of this transformation matrix without translation.
 * @details This is used to apply the camera transform to ray directions,
 * which should be rotated but not translated.
 *
 * @param a An existing transformation matrix
 * @return A copy of the transformation matrix with no translation
 */
template <typename T>
HOSTDEV Mat<T, 4, 4> transform_clear_translate(const Mat<T, 4, 4> &a) {
    Mat<T, 4, 4> result = a;
    result(0, 3) = 0;
    result(1, 3) = 0;
    result(2, 3) = 0;
    return result;
}

#endif