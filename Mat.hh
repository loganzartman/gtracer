#ifndef MAT_HH
#define MAT_HH

#include <initializer_list>
#include <cstddef>
#include "Vec3.hh"
#include "util.hh"

template <typename T, size_t N, size_t M>
class Mat {
   private:
    T el[N * M];

   public:
    /**
     * @brief Construct a zero matrix
     */
    HOSTDEV Mat() { 
        for (size_t i = 0; i < N * M; ++i)
            el[i] = 0;
    }

    /**
     * @brief Construct a matrix from an array of elements.
     * @details Elements should be an array of row-vectors. Ex:
     * 1, 2, 3, 4 =>
     * [1 2]
     * [3 4]
     *
     * @param elems Array of packed row-vectors
     */
    HOSTDEV Mat(std::initializer_list<T> elems) {
        T* dst = el;
        const T* src = elems.begin();
        while (src != elems.end()) {
            *dst = *src;
            ++dst;
            ++src;
        }
    }

    /**
     * @brief Add two matrices
     *
     * @param a The second matrix
     * @return A new matrix representing the sum
     */
    HOSTDEV Mat<T, N, M> operator+(const Mat<T, N, M>& a) const {
        Mat<T, N, M> out = *this;
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < M; ++j) {
                out(i, j) = (*this)(i, j) + a(i, j);
            }
        }
        return out;
    }

    /**
     * @brief Multiply two matrices
     *
     * @param b The right-hand matrix
     * @return A new matrix representing the product
     */
    template <size_t X>
    HOSTDEV Mat<T, N, X> operator*(const Mat<T, M, X>& b) const {
        Mat<T, N, X> out;
        for (size_t r = 0; r < N; ++r) {
            for (size_t c = 0; c < X; ++c) {
                for (size_t i = 0; i < M; ++i) {
                    out(r, c) += (*this)(r, i) * b(i, c);
                }
            }
        }
        return out;
    }

    /**
     * @brief Multiply 4x4 matrix by a Vec3
     * @details Augments the Vec3 to become {x, y, z, 1}.
     * Multiplies it on the right side of this matrix.
     *
     * @param b The vec3 to multiply on the right.
     * @return A new matrix representing the product
     */
    HOSTDEV Vec3<T> operator*(const Vec3<T>& b) const {
        Mat<T, 4, 1> b_vec{{b.x, b.y, b.z, 1}};
        Mat<T, 4, 1> result = (*this) * b_vec;
        const T iw = 1 / result(3, 0);
        return Vec3<T>(result(0, 0), result(1, 0), result(2, 0)) * iw;
    }

    HOSTDEV T& operator()(size_t row, size_t col) { return el[row * M + col]; }
    HOSTDEV const T& operator()(size_t row, size_t col) const {
        return el[row * M + col];
    }

    /**
     * @brief Produce an identity matrix. Dimensions must be equal.
     * @return The identity matrix
     */
    HOSTDEV static Mat<T, N, M> identity() {
        static_assert(N == M, "Identity matrix must be square");
        Mat<T, N, M> m;
        for (size_t i = 0; i < N; ++i)
            m(i, i) = 1;
        return m;
    }
};

typedef Mat<float, 4, 4> Mat4f;

#endif