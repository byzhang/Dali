#include "dali/tensor/op/reducers.h"

#include "dali/tensor/__MatMacros__.h"
#include "dali/math/TensorOps.h"
#include "dali/math/LazyTensor.h"

namespace matops {

    template<typename R>
    Mat<R> Reducers<R>::grad_norm(Mat<R> matrix) {
        auto out = Mat<R>(1, 1, weights<R>::empty());
        auto norm = GRAD(matrix).L2_norm();
        out.w(0) = norm;
        return out;
    }

    template<typename R>
    Mat<R> Reducers<R>::grad_norm_rowwise(Mat<R> matrix) {
        if (matrix.dims(1) == 1)
            return matrix;
        Mat<R> out(matrix.dims(0), 1);
        MAT(out).ravel() = reduce_to_1d<0, mshadow::red::sum>(F<TensorOps::op::square<R>>(GRAD(matrix).wrapper()));

        MAT(out) = F<TensorOps::op::sqrt_f<R>>(MAT(out).wrapper());

        return out;
    }

    template<typename R>
    Mat<R> Reducers<R>::grad_norm_colwise(Mat<R> matrix) {
        if (matrix.dims(0) == 1)
            return matrix;
        Mat<R> out(1, matrix.dims(1), weights<R>::empty());
        MAT(out).ravel() = reduce_to_1d<1, mshadow::red::sum>(F<TensorOps::op::square<R>>(GRAD(matrix).wrapper()));
        MAT(out)         = F<TensorOps::op::sqrt_f<R>>(MAT(out).wrapper());

        return out;
    }

    template<typename R>
    Mat<R> Reducers<R>::L2_norm(Mat<R> matrix) {
        auto out = Mat<R>(1, 1, weights<R>::empty());
        auto norm = MAT(matrix).L2_norm();
        out.w(0) = norm;

        if (graph::backprop_enabled() && !matrix.constant)
            graph::emplace_back([matrix, out, norm]() mutable {
                GRAD(matrix) += (MAT(matrix).wrapper() * (out.dw(0) / norm) );
            });
        return out;
    }

    template<typename R>
    Mat<R> Reducers<R>::L2_norm_rowwise(Mat<R> matrix) {
        if (matrix.dims(1) == 1)
            return matrix;
        Mat<R> out(matrix.dims(0), 1);
        MAT(out).ravel() = reduce_to_1d<0, mshadow::red::sum>(F<TensorOps::op::square<R>>(MAT(matrix).wrapper()));

        MAT(out) = F<TensorOps::op::sqrt_f<R>>(MAT(out).wrapper());

        if (graph::backprop_enabled() && !matrix.constant)
            graph::emplace_back([matrix, out]() mutable {
                TensorInternal<R,2> temp(MAT(out).shape);
                temp = GRAD(out).wrapper() / MAT(out).wrapper();
                GRAD(matrix) += MAT(matrix).wrapper() * (temp.ravel().wrapper().template broadcast<0>(GRAD(matrix).shape));
            });
        return out;
    }

    template<typename R>
    Mat<R> Reducers<R>::L2_norm_colwise(Mat<R> matrix) {
        if (matrix.dims(0) == 1)
            return matrix;
        Mat<R> out(1, matrix.dims(1), weights<R>::empty());
        MAT(out).ravel() = reduce_to_1d<1, mshadow::red::sum>(F<TensorOps::op::square<R>>(MAT(matrix).wrapper()));
        MAT(out)         = F<TensorOps::op::sqrt_f<R>>(MAT(out).wrapper());

        if (graph::backprop_enabled() && !matrix.constant)
            graph::emplace_back([matrix, out]() mutable {
                TensorInternal<R,2> temp(MAT(out).shape);
                temp = GRAD(out).wrapper() / MAT(out).wrapper();
                GRAD(matrix) += MAT(matrix).wrapper() * (temp).ravel().wrapper().template broadcast<1>(GRAD(matrix).shape);
            });
        return out;
    }



    template<typename R>
    Mat<R> Reducers<R>::sum(Mat<R> matrix) {
        if (matrix.dims(0) == 1 && matrix.dims(1) == 1)
            return matrix;
        Mat<R> out(1,1, weights<R>::empty());
        out.w(0) = MAT(matrix).sum();

        if (graph::backprop_enabled() && !matrix.constant)
            graph::emplace_back([matrix, out]() mutable {
                GRAD(matrix) += out.dw(0);
            });
        return out;
    }

    template<typename R>
    Mat<R> Reducers<R>::sum_rowwise(Mat<R> matrix) {
        if (matrix.dims(1) == 1)
            return matrix;
        Mat<R> out(matrix.dims(0), 1, weights<R>::empty());
        MAT(out).ravel() = reduce_to_1d<0, mshadow::red::sum>(MAT(matrix).wrapper());

        if (graph::backprop_enabled() && !matrix.constant)
            graph::emplace_back([matrix, out]() mutable {
                GRAD(matrix) += GRAD(out).ravel().wrapper().template broadcast<0>(GRAD(matrix).shape);
            });
        return out;
    }

    template<typename R>
    Mat<R> Reducers<R>::sum_colwise(Mat<R> matrix) {
        if (matrix.dims(0) == 1)
            return matrix;
        Mat<R> out(1, matrix.dims(1), weights<R>::empty());
        MAT(out).ravel() = reduce_to_1d<1, mshadow::red::sum>(MAT(matrix).wrapper());

        if (graph::backprop_enabled() && !matrix.constant)
            graph::emplace_back([matrix, out]() mutable {
                GRAD(matrix) += GRAD(out).ravel().wrapper().template broadcast<1>(GRAD(matrix).shape);
            });
        return out;
    }

    template<typename R>
    Mat<R> Reducers<R>::mean(Mat<R> matrix) {
        Mat<R> out (1,1, weights<R>::empty());
        auto ne = matrix.number_of_elements();
        out.w(0) = MAT(matrix).sum() / ne;
        if (graph::backprop_enabled() && !matrix.constant)
            graph::emplace_back([matrix, out, ne]() mutable {
                GRAD(matrix) += out.dw(0) / ne;
            });

        return out;
    }


    template<typename R>
    Mat<R> Reducers<R>::mean_rowwise(Mat<R> matrix) {
        if (matrix.dims(1) == 1)
            return matrix;
        Mat<R> out(matrix.dims(0), 1, weights<R>::empty());
        R ne = matrix.number_of_elements();

        MAT(out).ravel() = reduce_to_1d<0, mshadow::red::sum>(MAT(matrix).wrapper());
        MAT(out).ravel() /= ne;

        if (graph::backprop_enabled() && !matrix.constant)
            graph::emplace_back([matrix, out, ne]() mutable {
                GRAD(matrix) += GRAD(out).ravel().wrapper().template broadcast<0>(GRAD(matrix).shape) / (R)ne;
            });
        return out;
    }

    template<typename R>
    Mat<R> Reducers<R>::mean_colwise(Mat<R> matrix) {
        if (matrix.dims(0) == 1)
            return matrix;
        Mat<R> out(1, matrix.dims(1), weights<R>::empty());
        R ne = matrix.number_of_elements();
        MAT(out).ravel() = reduce_to_1d<1, mshadow::red::sum>(MAT(matrix).wrapper());
        MAT(out).ravel() /= ne;

        if (graph::backprop_enabled() && !matrix.constant)
            graph::emplace_back([matrix, out, ne]() mutable {
                GRAD(matrix) += GRAD(out).ravel().wrapper().template broadcast<1>(GRAD(matrix).shape) / ne;
            });
        return out;
    }

    template<typename R>
    Mat<R> Reducers<R>::max(Mat<R> matrix) {
        auto mat_idx = MAT(matrix).argmax();
        return matrix.ravel()[mat_idx];
    }

    template<typename R>
    Mat<R> Reducers<R>::max_rowwise(Mat<R> matrix) {
        if (matrix.dims(1) == 1)
            return matrix;
        Mat<R> out(matrix.dims(0), 1, weights<R>::empty());
        MAT(out).ravel() = reduce_to_1d<0, mshadow::red::maximum>(MAT(matrix).wrapper());

        if (graph::backprop_enabled() && !matrix.constant)
            graph::emplace_back([matrix, out]() mutable {
                GRAD(matrix) += (
                    GRAD(out).ravel().wrapper().template broadcast<0>(GRAD(matrix).shape) *
                    F<TensorOps::op::maximum_backward<R>>(
                        MAT(matrix).wrapper(),
                        MAT(out).ravel().wrapper().template broadcast<0>(GRAD(matrix).shape)
                    )
                );
            });
        return out;
    }

    template<typename R>
    Mat<R> Reducers<R>::max_colwise(Mat<R> matrix) {
        if (matrix.dims(0) == 1)
            return matrix;
        Mat<R> out(1, matrix.dims(1), weights<R>::empty());
        MAT(out).ravel() = reduce_to_1d<1, mshadow::red::maximum>(MAT(matrix).wrapper());

        if (graph::backprop_enabled() && !matrix.constant)
            graph::emplace_back([matrix, out]() mutable {
                GRAD(matrix) += (
                    GRAD(out).ravel().wrapper().template broadcast<1>(GRAD(matrix).shape) *
                    F<TensorOps::op::maximum_backward<R>>(
                        MAT(matrix).wrapper(),
                        MAT(out).ravel().wrapper().template broadcast<1>(GRAD(matrix).shape)
                    )
                );
            });
        return out;
    }

    template<typename R>
    Mat<R> Reducers<R>::min(Mat<R> matrix) {
        auto mat_idx = MAT(matrix).argmin();
        return matrix.ravel()[mat_idx];
    }

    template<typename R>
    Mat<R> Reducers<R>::min_rowwise(Mat<R> matrix) {
        if (matrix.dims(1) == 1)
            return matrix;
        Mat<R> out(matrix.dims(0), 1, weights<R>::empty());
        MAT(out).ravel() = reduce_to_1d<0, mshadow::red::minimum>(MAT(matrix).wrapper());

        if (graph::backprop_enabled() && !matrix.constant)
            graph::emplace_back([matrix, out]() mutable {
                GRAD(matrix) += (
                    GRAD(out).ravel().wrapper().template broadcast<0>(GRAD(matrix).shape) *
                    F<TensorOps::op::minimum_backward<R>>(
                        MAT(matrix).wrapper(),
                        MAT(out).ravel().wrapper().template broadcast<0>(GRAD(matrix).shape)
                    )
                );
            });
        return out;
    }

    template<typename R>
    Mat<R> Reducers<R>::min_colwise(Mat<R> matrix) {
        if (matrix.dims(0) == 1)
            return matrix;
        Mat<R> out(1, matrix.dims(1), weights<R>::empty());
        MAT(out).ravel() = reduce_to_1d<1, mshadow::red::minimum>(MAT(matrix).wrapper());

        if (graph::backprop_enabled() && !matrix.constant)
            graph::emplace_back([matrix, out]() mutable {
                GRAD(matrix) += (
                    GRAD(out).ravel().wrapper().template broadcast<1>(GRAD(matrix).shape) *
                    F<TensorOps::op::minimum_backward<R>>(
                        MAT(matrix).wrapper(),
                        MAT(out).ravel().wrapper().template broadcast<1>(GRAD(matrix).shape)
                    )
                );
            });
        return out;
    }


    template class Reducers<float>;
    template class Reducers<double>;
    template class Reducers<int>;

}
