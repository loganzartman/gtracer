#ifndef MAT_HH
#define MAT_HH

#include <algorithm>
#include <cstddef>

template <typename T, size_t N, size_t M>
class Mat {
   private:
    T el[N * M];

   public:
    Mat() { std::fill_n(el, N * M, 0); }
    Mat(T* elems) { std::copy_n(elems, N * M, el); }
    template <typename... E>
    Mat(E... elems) : el{elems...} {};

    Mat<T, N, M> operator+(const Mat<T, N, M>& a) {
        Mat<T, N, M> out = this->clone();
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < M; ++j) {
                out(i, j) = (*this)(i, j) + a(i, j);
            }
        }
        return out;
    }

    template <size_t X>
    Mat<T, N, X> operator*(const Mat<T, M, X>& b) {
        Mat<T, N, X> out;
        for (int r = 0; r < N; ++r) {
            for (int c = 0; c < X; ++c) {
                for (int i = 0; i < M; ++i) {
                    out(r, c) += (*this)(r, i) * b(i, c);
                }
            }
        }
        return out;
    }

    T& operator()(size_t row, size_t col) { return el[row * M + col]; }
    const T& operator()(size_t row, size_t col) const {
        return el[row * M + col];
    }

    Mat<T, N, M> clone() { return Mat<T, N, M>(el); }
};

typedef Mat<float, 4, 4> Mat4f;

#endif