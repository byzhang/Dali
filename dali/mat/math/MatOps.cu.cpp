#include "dali/mat/math/MatOps.h"
#include "dali/mat/math/__MatMacros__.h"
#include "dali/mat/math/TensorOps.h"

using std::vector;
using std::string;
using utils::assert2;
using utils::MS;
using utils::LambdaOperator;
using graph::backprop_enabled;
using graph::emplace_back;
using mshadow::expr::ExpEngine;
using mshadow::expr::scalar;

template<typename R>
R MatOps<R>::EPS = 1e-9;

#define DONT_COMPILE

namespace Indexing {
    typedef uint ind_t;
    class Index {
        ind_t& operator[](std::size_t idx);
        ind_t  operator[](std::size_t idx) const;
    };
}

template<typename R>
Mat<R> MatOps<R>::eltmul_broadcast(
        Mat<R> matrix1,
        Mat<R> matrix2) {
    #ifndef DONT_COMPILE
    assert2(matrix1.dims(0) == matrix2.dims(0) && matrix2.dims(1) == 1,
            MS() << "Matrices " << matrix1 << " and " << matrix2
                 << " cannot be element multiplied with broadcast,"
                 << " they do not have the same dimensions.");
    auto out = Mat<R>::empty_like(matrix1);
    MAT(out) = (MAT(matrix1).array().colwise() * MAT(matrix2).col(0).array()).matrix();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix1, matrix2, out]() {
            SAFE_GRAD(matrix1).noalias() += ((GRAD(out)).array().colwise() * (MAT(matrix2)).col(0).array()).matrix();
            SAFE_GRAD(matrix2).noalias() += ((MAT(matrix1)).array() * (GRAD(out)).array()).matrix().rowwise().sum();
        });
    return out;
    #else
    return Mat<R>(1,1);
    #endif
}

template<typename R>
Mat<R> MatOps<R>::eltdivide_broadcast(
        Mat<R> matrix1,
        Mat<R> matrix2) {
    #ifndef DONT_COMPILE
    assert2(matrix1.dims(0) == matrix2.dims(0) && matrix2.dims(1) == 1,
            MS() << "Matrices " << matrix1 << " and " << matrix2
                 << " cannot be element divided with broadcast,"
                 << " they do not have the same dimensions.");
    auto out = Mat<R>::empty_like(matrix1);
    MAT(out) = (
        MAT(matrix1).array().colwise()
        /
        MAT(matrix2).col(0).array()
    ).matrix();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix1, matrix2, out]() {
            SAFE_GRAD(matrix1).noalias() += (
                (GRAD(out)).array().colwise() *
                (MAT(matrix2)).col(0).array().inverse()
            ).matrix();
            SAFE_GRAD(matrix2).noalias() -= (
                (
                    MAT(matrix1).array().colwise() /
                    MAT(matrix2).col(0).array().square()
                ).matrix().array() * GRAD(out).array()
            ).matrix().rowwise().sum();
        });
    return out;
    #else
    return Mat<R>(1,1);
    #endif
}

template<typename R>
Mat<R> MatOps<R>::eltdivide_broadcast_reversed(
        Mat<R> matrix1,
        Mat<R> matrix2) {
    #ifndef DONT_COMPILE
    assert2(matrix1.dims(0) == matrix2.dims(0) && matrix2.dims(1) == 1,
            MS() << "Matrices " << matrix1 << " and " << matrix2
                 << " cannot be element divided with broadcast,"
                 << " they do not have the same dimensions.");
    auto out = Mat<R>::empty_like(matrix1);
    MAT(out) = (MAT(matrix1).array().inverse().colwise() * MAT(matrix2).col(0).array()).matrix();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix1, matrix2, out]() {
            SAFE_GRAD(matrix1).noalias() -= ((MAT(matrix1).array().square().inverse().colwise() * MAT(matrix2).col(0).array()).matrix().array() * GRAD(out).array()).matrix();
            SAFE_GRAD(matrix2).noalias() += (MAT(matrix1).array().inverse() * GRAD(out).array()).rowwise().sum().matrix();
        });
    return out;
    #else
    return Mat<R>(1,1);
    #endif
}

template<typename R>
Mat<R> MatOps<R>::eltmul(
        Mat<R> matrix1,
        Mat<R> matrix2) {
    #ifndef DONT_COMPILE
    if (matrix1.dims(1) != matrix2.dims(1) && (matrix1.dims(1) == 1 || matrix2.dims(1) == 1)) {
        if (matrix1.dims(1) == 1) {
            return eltmul_broadcast(matrix2, matrix1);
        }
        return eltmul_broadcast(matrix1, matrix2);
    }
    assert2(matrix1.dims(0) == matrix2.dims(0) && matrix1.dims(1) == matrix2.dims(1),
            "Matrices cannot be element-wise multiplied, they do not have the same dimensions.");
    auto out = Mat<R>::empty_like(matrix1);
    MAT(out) = (MAT(matrix1).array() * MAT(matrix2).array()).matrix();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix1, matrix2, out]() {
            SAFE_GRAD(matrix1).noalias() += ((MAT(matrix2)).array() * (GRAD(out)).array()).matrix();
            SAFE_GRAD(matrix2).noalias() += ((MAT(matrix1)).array() * (GRAD(out)).array()).matrix();
        });
    return out;
    #else
    return Mat<R>(1,1);
    #endif
}

template<typename R>
Mat<R> MatOps<R>::max(Mat<R> matrix, R lower_bound) {
    #ifndef DONT_COMPILE
    auto out = Mat<R>::empty_like(matrix);
    // out = max(matrix, lower_bound);
    MAT(out) = MAT(matrix).unaryExpr(
        LambdaOperator<R>([&lower_bound](R item) {
            return std::max(item, lower_bound);
        }));
    if (graph::backprop_enabled)
        graph::emplace_back([matrix, out, lower_bound]() {
            if (!matrix.constant) {
                // mask = (matrix >= lower_bound) ? 1.0 : 0:0;

                auto mask = MAT(matrix).unaryExpr(
                    LambdaOperator<R>([&lower_bound](R item) {
                        return item >= lower_bound ? 1.0 : 0.0;
                    }));
                GRAD(matrix).noalias() += (mask.array() * (GRAD(out)).array()).matrix();
            }
        });
    return out;
    #else
    return Mat<R>(1,1);
    #endif
}

template<typename R>
Mat<R> MatOps<R>::eltmul(
    Mat<R> matrix,
    R alpha) {
    #ifndef DONT_COMPILE
    auto out = Mat<R>::empty_like(matrix);
    MAT(out) = (MAT(matrix).array() * alpha).matrix();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix, alpha, out]() {
            SAFE_GRAD(matrix).noalias() += (alpha * (GRAD(out)).array()).matrix();
        });
    return out;
    #else
    return Mat<R>(1,1);
    #endif
}

template<typename R>
vector<Mat<R>> MatOps<R>::eltmul(const vector<Mat<R>>& seq1, const vector<Mat<R>>& seq2) {
    #ifndef DONT_COMPILE
    assert2(seq1.size() == seq2.size(), "Multiplying sequences of different sizes.");

    vector<Mat<R>> result(seq1.size());
    for (int i = 0; i < seq1.size(); ++i) {
        result[i] = seq1[i] * seq2[i];
    }
    return result;
    #else
    return {Mat<R>(1,1)};
    #endif
}


template<typename R>
Mat<R> MatOps<R>::eltdivide(
    Mat<R> matrix1,
    Mat<R> matrix2) {
    #ifndef DONT_COMPILE
    if (matrix1.dims(1) != matrix2.dims(1) && (matrix1.dims(1) == 1 || matrix2.dims(1) == 1)) {
        if (matrix1.dims(1) == 1) {
            return eltdivide_broadcast_reversed(matrix2, matrix1);
        }
        return eltdivide_broadcast(matrix1, matrix2);
    }
    assert2(matrix1.dims(0) == matrix2.dims(0) && matrix1.dims(1) == matrix2.dims(1),
            "Matrices cannot be element-wise divided, they do not have the same dimensions.");
    auto out = Mat<R>::empty_like(matrix1);
    MAT(out) = (MAT(matrix1).array() / MAT(matrix2).array()).matrix();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix1, matrix2, out]() {
            SAFE_GRAD(matrix1).noalias() += (
                MAT(matrix2).array().inverse() *
                GRAD(out).array()
            ).matrix();
            SAFE_GRAD(matrix2).noalias() -= (
                (
                    MAT(matrix1).array() /
                    MAT(matrix2).array().square()
                ) * GRAD(out).array()
            ).matrix();
        });
    return out;
    #else
    return {Mat<R>(1,1)};
    #endif
}


template<typename R>
Mat<R> MatOps<R>::eltdivide(
    Mat<R> matrix,
    R alpha) {
    #ifndef DONT_COMPILE
    auto out = Mat<R>::empty_like(matrix);
    MAT(out) = (MAT(matrix).array() / alpha).matrix();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix, alpha, out]() {
            SAFE_GRAD(matrix).noalias() += ((1.0 / alpha) * (GRAD(out)).array()).matrix();
        });
    return out;
    #else
    return Mat<R>(1,1);
    #endif
}


template<typename R>
Mat<R> MatOps<R>::eltmul_broadcast_rowwise(
    Mat<R> matrix1,
    Mat<R> row_vector) {
    #ifndef DONT_COMPILE
    if (matrix1.dims(1) != row_vector.dims(1) || row_vector.dims(0) != 1)
        throw std::invalid_argument("Matrices A and B^T cannot be element multiplied with broadcast, they do not have the same dimensions.");
    auto out = Mat<R>::empty_like(matrix1);
    MAT(out) = (MAT(matrix1).array().rowwise() * MAT(row_vector).row(0).array()).matrix();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix1, row_vector, out]() {
            SAFE_GRAD(matrix1).noalias() += ((GRAD(out)).array().rowwise() * (MAT(row_vector)).row(0).array()).matrix();
            SAFE_GRAD(row_vector).noalias() += (((MAT(matrix1)).array() * (GRAD(out)).array()).matrix().colwise().sum()).matrix();
        });
    return out;
    #else
    return Mat<R>(1,1);
    #endif
}

template<typename R>
vector<Mat<R>> MatOps<R>::eltmul_broadcast_rowwise(const vector<Mat<R>>& seq1, const vector<Mat<R>>& seq2) {
    #ifndef DONT_COMPILE
    assert2(seq1.size() == seq2.size(), "Multiplying sequences of different sizes.");

    vector<Mat<R>> result(seq1.size());
    for (int i = 0; i < seq1.size(); ++i) {
        result[i] = eltmul_broadcast_rowwise(seq1[i], seq2[i]);
    }
    return result;
    #else
    return {Mat<R>(1,1)};
    #endif
}

template<typename R>
Mat<R> MatOps<R>::eltmul_rowwise(
    Mat<R> matrix1,
    Mat<R> matrix2) {
    #ifndef DONT_COMPILE

    if (matrix1.dims(0) != matrix2.dims(1) || matrix1.dims(1) != matrix2.dims(0))
        throw std::invalid_argument("Matrices A and B^T cannot be element-wise multiplied, they do not have the same dimensions.");
    auto out = Mat<R>::empty_like(matrix1);
    MAT(out) = (MAT(matrix1).array() * MAT(matrix2).transpose().array()).matrix();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix1, matrix2, out]() {
            SAFE_GRAD(matrix1).noalias() += (
                MAT(matrix2).transpose().array() *
                GRAD(out).array()
            ).matrix();
            SAFE_GRAD(matrix2).noalias() += (
                MAT(matrix1).array() *
                GRAD(out).array()
            ).matrix().transpose();
        });
    return out;
    #else
    return Mat<R>(1,1);
    #endif
}

template<typename R>
vector<Mat<R>> MatOps<R>::eltmul_rowwise(const vector<Mat<R>>& seq1, const vector<Mat<R>>& seq2) {
    #ifndef DONT_COMPILE
    assert2(seq1.size() == seq2.size(), "Multiplying sequences of different sizes.");

    vector<Mat<R>> result(seq1.size());
    for (int i = 0; i < seq1.size(); ++i) {
        result[i] = eltmul_rowwise(seq1[i], seq2[i]);
    }
    return result;
    #else
    return {Mat<R>(1,1)};
    #endif
}


template<typename R>
Mat<R> MatOps<R>::add(
        Mat<R> matrix1,
        Mat<R> matrix2) {
    if (matrix1.dims(1) != matrix2.dims(1) && (matrix1.dims(1) == 1 || matrix2.dims(1) == 1)) {
        if (matrix1.dims(1) == 1) {
            return add_broadcast(matrix2, matrix1);
        }
        return add_broadcast(matrix1, matrix2);
    }
    assert2((matrix1.dims(0) == matrix2.dims(0)) && (matrix1.dims(1) == matrix2.dims(1)),
        "Matrices cannot be added, they do not have the same dimensions.");

    auto out = Mat<R>::empty_like(matrix1);
    DALI_FUNCTION_3_MUT(TensorOps::add, MAT(matrix1), MAT(matrix2), MAT(out), OVERWRITE);

    if (graph::backprop_enabled)
        graph::emplace_back([matrix1, matrix2, out]() {
            if (!matrix1.constant)
                // matrix1.dw() += out.dw()
                DALI_FUNCTION_2_MUT(TensorOps::add_inplace, GRAD(out), GRAD(matrix1));
            if (!matrix2.constant)
                // matrix1.dw() += out.dw()
                DALI_FUNCTION_2_MUT(TensorOps::add_inplace, GRAD(out), GRAD(matrix2));
        });
    return out;
}


template<typename R>
Mat<R> MatOps<R>::sub(
        Mat<R> matrix1,
        Mat<R> matrix2) {
    #ifndef DONT_COMPILE
    if (matrix1.dims(1) != matrix2.dims(1) && (matrix1.dims(1) == 1 || matrix2.dims(1) == 1)) {
        if (matrix1.dims(1) == 1) {
            return sub_broadcast_reversed(matrix2, matrix1);
        }
        return sub_broadcast(matrix1, matrix2);
    }
    assert2((matrix1.dims(0) == matrix2.dims(0)) && (matrix1.dims(1) == matrix2.dims(1)),
        "Matrices cannot be subtracted, they do not have the same dimensions.");
    auto out = Mat<R>::empty_like(matrix1);
    MAT(out) = MAT(matrix1) - MAT(matrix2);
    if (graph::backprop_enabled)
        graph::emplace_back([matrix1, matrix2, out]() {
            SAFE_GRAD(matrix1).noalias() += GRAD(out);
            SAFE_GRAD(matrix2).noalias() -= GRAD(out);
        });
    return out;
    #else
    return Mat<R>(1,1);
    #endif
}

template<typename R>
Mat<R> MatOps<R>::add(
        Mat<R> matrix1,
        R alpha) {
    #ifndef DONT_COMPILE
    auto out = Mat<R>::empty_like(matrix1);
    MAT(out).array() = MAT(matrix1).array() + alpha;
    if (graph::backprop_enabled)
        graph::emplace_back([matrix1, out]() {
            SAFE_GRAD(matrix1).noalias() += GRAD(out);
        });
    return out;
    #else
    return Mat<R>(1,1);
    #endif
}

template<typename R>
Mat<R> MatOps<R>::add_broadcast(Mat<R> matrix1, Mat<R> matrix2) {
    #ifndef DONT_COMPILE
    // broadcast matrix 2:
    if (matrix1.dims(0) != matrix2.dims(0) || matrix2.dims(1) != 1)
            throw std::invalid_argument("Matrices cannot be added with broadcast, they do not have the same dimensions.");
    auto out = Mat<R>::empty_like(matrix1);
    MAT(out) = (MAT(matrix1).colwise() + MAT(matrix2).col(0)).matrix();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix1, matrix2, out]() {
            SAFE_GRAD(matrix1).noalias() += GRAD(out);
            SAFE_GRAD(matrix2).noalias() += GRAD(out).rowwise().sum();
        });
    return out;
    #else
    return Mat<R>(1,1);
    #endif
}

template<typename R>
Mat<R> MatOps<R>::sub_broadcast(Mat<R> matrix1, Mat<R> matrix2) {
    #ifndef DONT_COMPILE
    // broadcast matrix 2:
    if (matrix1.dims(0) != matrix2.dims(0) || matrix2.dims(1) != 1)
        throw std::invalid_argument("Matrices cannot be substracted with broadcast, they do not have the same dimensions.");
    auto out = Mat<R>::empty_like(matrix1);
    MAT(out) = (MAT(matrix1).colwise() - MAT(matrix2).col(0)).matrix();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix1, matrix2, out]() {
            SAFE_GRAD(matrix1).noalias() += GRAD(out);
            SAFE_GRAD(matrix2).noalias() -= GRAD(out).rowwise().sum();
        });
    return out;
    #else
    return Mat<R>(1,1);
    #endif
}

template<typename R>
Mat<R> MatOps<R>::sub_broadcast_reversed(Mat<R> matrix1, Mat<R> matrix2) {
    #ifndef DONT_COMPILE
    // broadcast matrix 2:
    if (matrix1.dims(0) != matrix2.dims(0) || matrix2.dims(1) != 1)
        throw std::invalid_argument("Matrices cannot be substracted with broadcast, they do not have the same dimensions.");
    auto out = Mat<R>::empty_like(matrix1);
    MAT(out) = ((-MAT(matrix1)).colwise() + MAT(matrix2).col(0)).matrix();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix1, matrix2, out] () {
            SAFE_GRAD(matrix1).noalias() -= GRAD(out);
            SAFE_GRAD(matrix2).noalias() += GRAD(out).rowwise().sum();
        });
    return out;
    #else
    return Mat<R>(1,1);
    #endif
}

template<typename R>
Mat<R> MatOps<R>::sub_broadcast_reversed(Mat<R> matrix, R other) {
    #ifndef DONT_COMPILE
    auto out = Mat<R>::empty_like(matrix);
    MAT(out) = (other - MAT(matrix).array()).matrix();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix, out] () {
            SAFE_GRAD(matrix).noalias() -= GRAD(out);
        });
    return out;
    #else
    return Mat<R>(1,1);
    #endif
}

template<typename R>
Mat<R> MatOps<R>::add(std::initializer_list<Mat<R>> matrices) {
    #ifndef DONT_COMPILE
    auto matrices_vector = vector<Mat<R>>(matrices);
    return add(matrices_vector);
    #else
    return Mat<R>(1,1);
    #endif
}

template<typename R>
Mat<R> MatOps<R>::add(const std::vector<Mat<R>>& matrices) {
    #ifndef DONT_COMPILE
    auto out = Mat<R>::zeros_like(*matrices.begin());
    for (auto& matrix : matrices)
        MAT(out) += MAT(matrix);
    if (graph::backprop_enabled)
        graph::emplace_back([matrices, out]() {
            for (auto& matrix : matrices) {
                SAFE_GRAD(matrix).noalias() += GRAD(out);
            }
        });
    return out;
    #else
    return Mat<R>(1,1);
    #endif
}

template<typename R>
Mat<R> MatOps<R>::square(Mat<R> matrix) {
    #ifndef DONT_COMPILE
    auto out = Mat<R>::empty_like(matrix);
    MAT(out) = MAT(matrix).array().square();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix, out]() {
            SAFE_GRAD(matrix).noalias() += 2.0 * ((MAT(matrix)).array() * (GRAD(out)).array()).matrix();
        });
    return out;
    #else
    return Mat<R>(1,1);
    #endif
}

template<typename R>
Mat<R> MatOps<R>::L2_norm(Mat<R> matrix) {
    #ifndef DONT_COMPILE
    auto out = Mat<R>(1, 1, false);
    MAT(out)(0) = MAT(matrix).norm();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix, out]() {
            SAFE_GRAD(matrix).noalias() += ((MAT(matrix).array() / MAT(out)(0)) * GRAD(out)(0)).matrix();
        });
    return out;
    #else
    return Mat<R>(1,1);
    #endif
}

template<typename R>
Mat<R> MatOps<R>::sqrt(Mat<R> matrix) {
    #ifndef DONT_COMPILE
    auto out = Mat<R>::empty_like(matrix);
    MAT(out) = MAT(matrix).array().sqrt();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix, out]() {
            SAFE_GRAD(matrix).noalias() += 0.5 * ((MAT(out)).array().inverse() * (GRAD(out)).array()).matrix();
        });
    return out;
    #else
    return Mat<R>(1,1);
    #endif
}

template<typename R>
Mat<R> MatOps<R>::elt_inv(Mat<R> matrix) {
    #ifndef DONT_COMPILE
    auto out = Mat<R>::empty_like(matrix);
    MAT(out) = MAT(matrix).array().inverse();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix, out]() {
            SAFE_GRAD(matrix).noalias() += -((MAT(out)).array().square() * GRAD(out).array()).matrix();
        });
    return out;
    #else
    return Mat<R>(1,1);
    #endif
}

template<typename R>
Mat<R> MatOps<R>::fill(Mat<R> matrix, R filler) {
    auto out = Mat<R>::empty_like(matrix);
    DALI_FUNCTION_1_MUT(TensorOps::fill, MAT(out), filler);
    return out;

}

template<typename R>
Mat<R> MatOps<R>::pow(Mat<R> matrix, R other) {
    #ifndef DONT_COMPILE
    if (other == (R) -1.0) {
        return MatOps<R>::elt_inv(matrix);
    } else if (other == (R) 0.0){
        return MatOps<R>::fill(matrix, 1.0);
    } else if (other == (R)0.5) {
        return MatOps<R>::sqrt(matrix);
    } else if (other == (R)1.0) {
        return matrix;
    } else if (other == (R)2.0) {
        return MatOps<R>::square(matrix);
    }
    auto out = Mat<R>::empty_like(matrix);
    MAT(out) = MAT(matrix).array().pow(other);
    if (graph::backprop_enabled)
        graph::emplace_back([matrix, out, other]() {
            SAFE_GRAD(matrix).noalias() += other * ((MAT(matrix)).array().pow(other - 1.0) * (GRAD(out)).array()).matrix();
        });
    return out;
    #else
    return Mat<R>(1,1);
    #endif
}

template<typename R>
Mat<R> MatOps<R>::pow(Mat<R> matrix, Mat<R> other) {
    #ifndef DONT_COMPILE
    assert2(other.dims(0) == 1 && other.dims(1) == 1, "exponent must be a 1x1 matrix.");
    auto out = Mat<R>::empty_like(matrix);
    MAT(out) = MAT(matrix).array().pow(MAT(other)(0));
    if (graph::backprop_enabled)
        graph::emplace_back([matrix, out, other]() {
            SAFE_GRAD(matrix).noalias() += MAT(other)(0) * ((MAT(matrix)).array().pow(MAT(other)(0) - 1.0) * (MAT(out)).array()).matrix();
            SAFE_GRAD(other)(0) += (
                MAT(matrix).unaryExpr(utils::log_or_zero<R>()).array() * MAT(out).array() * GRAD(out).array()
            ).sum();
        });
    return out;
    #else
    return Mat<R>(1,1);
    #endif
}

template<typename R>
Mat<R> MatOps<R>::sigmoid(Mat<R> matrix) {
    #ifndef DONT_COMPILE
    auto out = Mat<R>::empty_like(matrix);
    MAT(out) = MAT(matrix).unaryExpr(utils::sigmoid_operator<R>());
    if (graph::backprop_enabled)
        graph::emplace_back([matrix, out](){
            SAFE_GRAD(matrix).noalias() += (((MAT(out)).array() - MAT(out).array().square()) * GRAD(out).array()).matrix();
        });
    return out;
    #else
    return Mat<R>(1,1);
    #endif
}


template<typename R>
Mat<R> MatOps<R>::softmax_no_grad(Mat<R> matrix, R temperature) {
    #ifndef DONT_COMPILE
    auto out = Mat<R>::empty_like(matrix);

    auto layer_max = MAT(matrix).colwise().maxCoeff().array().matrix();
    auto exped_distributions = (
        (MAT(matrix).rowwise() - layer_max.row(0)) / temperature
    ).array().exp().matrix();

    auto total_distribution = exped_distributions.colwise().sum().array().matrix();
    MAT(out) = (exped_distributions.array().rowwise() / total_distribution.row(0).array());
    return out;
    #else
    return Mat<R>(1,1);
    #endif
}

template<typename R>
Mat<R> MatOps<R>::softmax(Mat<R> matrix, R temperature) {
    #ifndef DONT_COMPILE
    Mat<R> out = MatOps<R>::softmax_no_grad(matrix, temperature);
    if (graph::backprop_enabled && !matrix.constant)
        graph::emplace_back([matrix, temperature, out]() {
            auto& dw = GRAD(matrix);
            auto& sm = MAT(out);
            auto& dy = GRAD(out);
            eigen_mat sm_times_dy = (sm.array() * dy.array());
            auto colwise_sums                      = sm_times_dy.colwise().sum();
            for (size_t i = 0; i < matrix.dims(1); ++i) {
                dw.col(i) += (sm_times_dy.col(i) - sm.col(i) * colwise_sums(i)) / temperature;
            }
        });
    return out;
    #else
    return Mat<R>(1,1);
    #endif
}

template<typename R>
Mat<R> MatOps<R>::softmax_transpose(Mat<R> matrix, R temperature) {
    #ifndef DONT_COMPILE
    Mat<R> out = MatOps<R>::softmax_no_grad_transpose(matrix, temperature);
    if (graph::backprop_enabled && !matrix.constant)
        graph::emplace_back([matrix, temperature, out]() {
            auto& dw = GRAD(matrix);
            auto& sm = MAT(out);
            auto& dy = GRAD(out);
            eigen_mat sm_times_dy = (sm.array() * dy.array());
            auto rowwise_sums                      = sm_times_dy.rowwise().sum();
            for (size_t i = 0; i < matrix.dims(0); ++i) {
                dw.row(i) += sm_times_dy.row(i) - sm.row(i) * rowwise_sums(i);
            }
        });
    return out;
    #else
    return Mat<R>(1,1);
    #endif
}

template<typename R>
Mat<R> MatOps<R>::softmax_no_grad_transpose(Mat<R> matrix, R temperature) {
    #ifndef DONT_COMPILE
    auto out = Mat<R>::empty_like(matrix);
    auto layer_max = MAT(matrix).rowwise().maxCoeff().array().matrix();
    auto exped_distributions = (MAT(matrix).colwise() - layer_max.row(0)).array().exp().matrix();

    auto total_distribution = exped_distributions.rowwise().sum().array().matrix();
    MAT(out) = (exped_distributions.array().colwise() / total_distribution.col(0).array());
    return out;
    #else
    return Mat<R>(1,1);
    #endif
}

template<typename R>
vector<Mat<R>> MatOps<R>::softmax_no_grad(const vector<Mat<R>>& matrices, R temperature) {
    #ifndef DONT_COMPILE
    vector<Mat<R>> out;
    out.reserve(matrices.size());
    assert2(matrices.size() > 0, "Must be a non empty list of vectors to softmax.");
    R layer_max = MAT(matrices[0])(0);

    for (auto& mat : matrices) {
        assert2(mat.dims(0) == 1 && mat.dims(1) == 1, "Softmax on a vector must be made on 1x1 matrices only.");
        layer_max = std::max(layer_max, MAT(mat)(0));
    }
    R total = 0.0;
    for (auto& mat : matrices) {
        out.emplace_back(1,1);
        MAT(out.back())(0) = std::exp((MAT(mat)(0) - layer_max) / temperature);
        total += MAT(out.back())(0);
    }
    for (auto& mat : out) {
        MAT(mat)(0) /= total;
    }
    return out;
    #else
    return {Mat<R>(1,1)};
    #endif
}

template<typename R>
vector<Mat<R>> MatOps<R>::softmax(const vector<Mat<R>>& matrices, R temperature) {
    #ifndef DONT_COMPILE
    vector<Mat<R>> out = MatOps<R>::softmax_no_grad(matrices, temperature);
    if (graph::backprop_enabled)
        graph::emplace_back([temperature, out, matrices]() {
            R colwise_sums = 0.0;

            for (int i = 0; i < out.size(); i++) {
                auto& dw = GRAD(matrices[i]);
                auto& sm = MAT(out[i]);
                auto& dy = GRAD(out[i]);
                colwise_sums += sm(0) * dy(0);
            }

            for (int i = 0; i < out.size(); i++) {
                if (!matrices[i].constant) {
                    auto& dw = GRAD(matrices[i]);
                    auto& sm = MAT(out[i]);
                    auto& dy = GRAD(out[i]);
                    dw(0) += ((sm(0) * dy(0)) - (sm(0) * colwise_sums)) / temperature;
                }
            }
        });
    return out;
    #else
    return {Mat<R>(1,1)};
    #endif
}

template<typename R>
Mat<R> MatOps<R>::steep_sigmoid(Mat<R> matrix, R aggressiveness) {
    #ifndef DONT_COMPILE
    auto out = Mat<R>::empty_like(matrix);
    MAT(out) = MAT(matrix).unaryExpr(utils::steep_sigmoid_operator<R>(aggressiveness));
    if (graph::backprop_enabled)
        graph::emplace_back([matrix, out, aggressiveness](){
            SAFE_GRAD(matrix).noalias() += (aggressiveness * ((MAT(out)).array() - MAT(out).array().square()) * GRAD(out).array()).matrix();
        });
    return out;
    #else
    return Mat<R>(1,1);
    #endif
}



template<typename R>
Mat<R> MatOps<R>::sum(Mat<R> matrix) {
    if (matrix.dims(0) == 1 && matrix.dims(1) == 1)
        return matrix;
    Mat<R> out(1,1, weights<R>::empty());
    out.w(0) = DALI_FUNCTION_1(TensorOps::sum, MAT(matrix), matrix.number_of_elements());
    if (backprop_enabled() && !matrix.constant)
        emplace_back([matrix, out](){
            DALI_FUNCTION_1_MUT(TensorOps::add_inplace,
                                GRAD(matrix),
                                matrix.number_of_elements(),
                                out.dw(0));
        });
    return out;
}

template<typename R>
Mat<R> MatOps<R>::mean(Mat<R> matrix) {
    Mat<R> out (1,1, weights<R>::empty());
    auto ne = matrix.number_of_elements();
    out.w(0) = DALI_FUNCTION_1(TensorOps::sum, MAT(matrix), ne) / ne;
    if (backprop_enabled() && !matrix.constant)
        emplace_back([matrix, out, ne](){
            DALI_FUNCTION_1_MUT(TensorOps::add_inplace,
                                GRAD(matrix),
                                ne,
                                out.dw(0) / ne);
        });

    return out;
}

template<typename R>
Mat<R> MatOps<R>::sigmoid_binary_cross_entropy(Mat<R> matrix, R t) {
    #ifndef DONT_COMPILE
    assert(0 <= t && t <= 1);
    assert(matrix.dims().size() > 1);
    auto out = Mat<R>::empty_like(matrix);

    auto sigmoided_input = std::make_shared<eigen_mat>(
        MAT(matrix).array().unaryExpr(utils::sigmoid_operator<R>())
    );

    MAT(out) = -(
                          t  * ( sigmoided_input->array()   + EPS      ).log()
                + ( 1.0 - t) * ( 1.00000001 - sigmoided_input->array() ).log()
    ).matrix();

    if (graph::backprop_enabled())
        graph::emplace_back([matrix, t, out, sigmoided_input](){
            SAFE_GRAD(matrix).array() += (sigmoided_input->array() - t) * GRAD(out).array();
        });
    return out;
    #else
    return Mat<R>(1,1);
    #endif
}

template<typename R>
Mat<R> MatOps<R>::binary_cross_entropy(Mat<R> matrix, R t) {
    #ifndef DONT_COMPILE
    assert(0 <= t && t <= 1);
    assert(matrix.dims().size() > 1);
    Mat<R> out =  Mat<R>(
        matrix.dims(0),
        matrix.dims(1),
        false);

    auto x = MAT(matrix).array();

    MAT(out) = (-(t * (x + EPS).log() + (1.0-t) * (1.0 - x + EPS).log())).matrix();

    DEBUG_ASSERT_MAT_NOT_NAN(out);

    if (graph::backprop_enabled())
        graph::emplace_back([matrix, t, out](){
            auto x = MAT(matrix).array();
            SAFE_GRAD(matrix).array() += (
                (
                    (t - x) /
                    (x * (x - 1.0) + EPS)
                ) * GRAD(out).array()
            );
            DEBUG_ASSERT_GRAD_NOT_NAN(matrix);
        });

    return out;
    #else
    return Mat<R>(1,1);
    #endif
}

template<typename R>
Mat<R> MatOps<R>::cross_entropy(Mat<R> matrix, uint answer_idx) {
    #ifndef DONT_COMPILE
    DEBUG_ASSERT_BOUNDS(MAT(matrix),0.0,1.0 + EPS);
    assert(matrix.dims().size() > 1);
    assert(answer_idx < matrix.dims(0));
    Mat<R> out =  Mat<R>(
        1,
        matrix.dims(1),
        false);

    auto x = MAT(matrix).array();
    MAT(out) = - (x.row(answer_idx).array() + EPS).log();

    DEBUG_ASSERT_MAT_NOT_NAN(out);

    if (graph::backprop_enabled)
        graph::emplace_back([matrix, answer_idx, out](){
            auto x = MAT(matrix).array();
            SAFE_GRAD(matrix).row(answer_idx).array() += -(x.row(answer_idx).array() + EPS).inverse() * GRAD(out).array();
        });
    return out;
    #else
    return Mat<R>(1,1);
    #endif
}

template<typename R>
Mat<R> MatOps<R>::cross_entropy(Mat<R> matrix, Mat<R> target) {
    #ifndef DONT_COMPILE
    assert2(matrix.dims(0) == target.dims(0) && matrix.dims(1) == target.dims(1),
        "Matrix and target must have same dimension");

    Mat<R> out = Mat<R>::empty_like(matrix);
    MAT(out) = -(MAT(target).array() * ((MAT(matrix).array() + EPS).log())).matrix();

    DEBUG_ASSERT_NOT_NAN(MAT(out));

    if (graph::backprop_enabled)
        graph::emplace_back([matrix, target, out](){
            auto x = MAT(matrix).array();
            SAFE_GRAD(matrix).noalias() -= (((x + EPS).inverse()) * MAT(target).array() * GRAD(out).array()).matrix();
            SAFE_GRAD(target).noalias() -= ((x.log()) * GRAD(out).array()).matrix();
        });
    return out;
    #else
    return Mat<R>(1,1);
    #endif
}


template<typename R>
Mat<R> MatOps<R>::softmax_cross_entropy(Mat<R> matrix, uint answer_idx) {
    #ifndef DONT_COMPILE
    Mat<R> out =  Mat<R>(1, 1, false);
    Mat<R> probs = softmax_no_grad(matrix);
    MAT(out)(0,0) = -std::log(MAT(probs)(answer_idx, 0));

    if (graph::backprop_enabled)
        graph::emplace_back([matrix, probs, answer_idx, out](){
            SAFE_GRAD(matrix) += MAT(probs) * GRAD(out)(0,0);
            // write gradients into log probabilities
            SAFE_GRAD(matrix)(answer_idx, 0) -= 1 * GRAD(out)(0,0);
        });
    return out;
    #else
    return Mat<R>(1,1);
    #endif
}

template<typename R>
Mat<R> MatOps<R>::softmax_cross_entropy(Mat<R> matrix, Indexing::Index targets) {
    #ifndef DONT_COMPILE
    Mat<R> out =  Mat<R>(1, targets.size(), false);
    Mat<R> probs = softmax_no_grad(matrix);
    for (int i = 0; i < targets.size(); i++) {
        MAT(out)(i) = -std::log(MAT(probs)(targets[i], i));
    }

    if (graph::backprop_enabled)
        graph::emplace_back([matrix, probs, out, targets](){
            if (!matrix.constant) {
                SAFE_GRAD(matrix).noalias() += (MAT(probs).array().rowwise() * GRAD(out).row(0).array()).matrix();
                for (int i = 0; i < targets.size(); i++) {
                    GRAD(matrix)(targets[i],i) -= 1.0 * GRAD(out)(i);
                }
            }
        });
    return out;
    #else
    return Mat<R>(1,1);
    #endif
}

template<typename R>
Mat<R> MatOps<R>::margin_loss(Mat<R> matrix, uint answer_idx, R margin) {
    #ifndef DONT_COMPILE
    // Exprected input is a column vector
    assert(answer_idx < matrix.dims(0));
    assert(matrix.dims(1) == 1);
    Mat<R> error(1,1);
    for (int idx = 0; idx < matrix.dims(0); ++idx) {
        if (idx == answer_idx) continue;
        error = error + MatOps<R>::max(matrix[idx] - matrix[answer_idx] + margin, 0.0);
    }
    return error;
    #else
    return Mat<R>(1,1);
    #endif
}

template<typename R>
Mat<R> MatOps<R>::log(Mat<R> matrix) {
    #ifndef DONT_COMPILE
    assert(matrix.dims().size() > 1);
    auto out = Mat<R>::empty_like(matrix);
    MAT(out) = MAT(matrix).array().log();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix, out](){
            SAFE_GRAD(matrix).noalias() += (
                (1.0 / MAT(matrix).array()) *
                GRAD(out).array()
            ).matrix();
        });
    return out;
    #else
    return Mat<R>(1,1);
    #endif
}

template<typename R>
Mat<R> MatOps<R>::exp(Mat<R> matrix) {
    #ifndef DONT_COMPILE
    assert(matrix.dims().size() > 1);
    auto out = Mat<R>::empty_like(matrix);
    MAT(out) = MAT(matrix).array().exp();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix, out](){
            SAFE_GRAD(matrix).noalias() += (MAT(out).array() * GRAD(out).array()).matrix();
        });
    return out;
    #else
    return Mat<R>(1,1);
    #endif
}

template<typename R>
Mat<R> MatOps<R>::hstack(Mat<R> matrix1, Mat<R> matrix2) {
    #ifndef DONT_COMPILE
    if (matrix1.dims(0) != matrix2.dims(0))
        throw std::invalid_argument("Matrices cannot be joined -- they do not have the same number of rows.");
    Mat<R> out (
        matrix1.dims(0),
        matrix1.dims(1) + matrix2.dims(1),
        false
    );
    MAT(out).block(0,0, matrix1.dims(0), matrix1.dims(1)) = MAT(matrix1);
    MAT(out).block(0,matrix1.dims(1), matrix2.dims(0), matrix2.dims(1)) = MAT(matrix2);
    if (graph::backprop_enabled)
        graph::emplace_back([matrix1, matrix2, out]() {
            SAFE_GRAD(matrix1).noalias() += GRAD(out).block(0,0, matrix1.dims(0), matrix1.dims(1));
            SAFE_GRAD(matrix2).noalias() += GRAD(out).block(0,matrix1.dims(1), matrix2.dims(0), matrix2.dims(1));
        });
    return out;
    #else
    return Mat<R>(1,1);
    #endif
}

template<typename R>
Mat<R> MatOps<R>::hstack(std::initializer_list<Mat<R>> matrices) {
    #ifndef DONT_COMPILE
    vector<Mat<R>> matrices_vector(matrices);
    return hstack(matrices_vector);
    #else
    return Mat<R>(1,1);
    #endif
}

template<typename R>
Mat<R> MatOps<R>::hstack(const std::vector<Mat<R>>& matrices) {
    #ifndef DONT_COMPILE
    int n = -1;
    int d_total = 0;
    for (auto& mat : matrices) {
        if (n == -1) {
            n = mat.dims(0);
        } else {
            if (mat.dims(0) != n) {
                throw std::invalid_argument("Matrices cannot be joined -- they do not have the same number of rows.");
            }
        }
        d_total+= mat.dims(1);
    }
    Mat<R> out (
        n,
        d_total,
        false
    );
    int offset = 0;
    for (auto& mat : matrices) {
        MAT(out).block(0, offset, mat.dims(0), mat.dims(1)) = MAT(mat);
        offset += mat.dims(1);
    }
    if (graph::backprop_enabled)
        graph::emplace_back([matrices, out]() {
            int offset = 0;
            for (auto & mat : matrices) {
                SAFE_GRAD(mat).noalias() += GRAD(out).block(0, offset, mat.dims(0), mat.dims(1));
                offset += mat.dims(1);
            }
        });
    return out;
    #else
    return Mat<R>(1,1);
    #endif
}

template<typename R>
Mat<R> MatOps<R>::vstack(Mat<R> matrix1, Mat<R> matrix2) {
    #ifndef DONT_COMPILE
    if (matrix1.dims(1) != matrix2.dims(1))
        throw std::invalid_argument("Matrices cannot be horizontally stacked -- they do not have the same number of cols.");
    Mat<R> out (
        matrix1.dims(0) + matrix2.dims(0),
        matrix1.dims(1),
        false
    );
    MAT(out).block(0,0, matrix1.dims(0), matrix1.dims(1)) = MAT(matrix1);
    MAT(out).block(matrix1.dims(0),0, matrix2.dims(0), matrix2.dims(1)) = MAT(matrix2);
    if (graph::backprop_enabled)
        graph::emplace_back([matrix1, matrix2, out]() {
            SAFE_GRAD(matrix1).noalias() += GRAD(out).block(0,0, matrix1.dims(0), matrix1.dims(1));
            SAFE_GRAD(matrix2).noalias() += GRAD(out).block(matrix1.dims(0),0, matrix2.dims(0), matrix2.dims(1));
        });
    return out;
    #else
    return Mat<R>(1,1);
    #endif
}

template<typename R>
Mat<R> MatOps<R>::vstack(std::initializer_list<Mat<R>> matrices) {
    #ifndef DONT_COMPILE
    vector<Mat<R>> matrices_vector(matrices);
    return vstack(matrices_vector);
    #else
    return Mat<R>(1,1);
    #endif
}

template<typename R>
Mat<R> MatOps<R>::vstack(const std::vector<Mat<R>>& matrices) {
    #ifndef DONT_COMPILE
    assert(matrices.size() > 0);
    assert(matrices[0].dims().size() > 1);
    int d = matrices[0].dims(1);
    int n_total = 0;
    for (auto& mat : matrices) {
        if (mat.dims(1) != d) {
            throw std::invalid_argument("Matrices cannot be horizontally stacked -- they do not have the same number of cols.");
        }
        n_total += mat.dims(0);
    }
    Mat<R> out (
        n_total,
        d,
        false
    );
    int offset = 0;
    for (auto& mat : matrices) {
        MAT(out).block(offset, 0, mat.dims(0), mat.dims(1)) = MAT(mat);
        offset += mat.dims(0);
    }
    if (graph::backprop_enabled)
        graph::emplace_back([matrices, out]() {
            int offset = 0;
            for (auto & mat : matrices) {
                SAFE_GRAD(mat).noalias() += GRAD(out).block(offset,0, mat.dims(0), mat.dims(1));
                offset += mat.dims(0);
            }
        });
    return out;
    #else
    return Mat<R>(1,1);
    #endif
}

template<typename R>
Mat<R> MatOps<R>::transpose(Mat<R> matrix) {
    #ifndef DONT_COMPILE
    assert(matrix.dims().size() > 1);
    Mat<R> out (
        matrix.dims(1),
        matrix.dims(0),
        false);
    MAT(out) = MAT(matrix).transpose();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix, out](){
            SAFE_GRAD(matrix).noalias() += (GRAD(out)).transpose();
        });
    return out;
    #else
    return Mat<R>(1,1);
    #endif
}

template<typename R>
Mat<R> MatOps<R>::tanh(Mat<R> matrix) {
    #ifndef DONT_COMPILE
    auto out = Mat<R>::empty_like(matrix);
    MAT(out) = MAT(matrix).unaryExpr(utils::tanh_operator<R>());
    if (graph::backprop_enabled)
        graph::emplace_back([matrix, out](){
            SAFE_GRAD(matrix).noalias() += (MAT(out).unaryExpr(utils::dtanh_operator<R>()).array() * GRAD(out).array()).matrix();
        });
    return out;
    #else
    return Mat<R>(1,1);
    #endif
}

template<typename R>
Mat<R> MatOps<R>::relu(Mat<R> matrix) {
    #ifndef DONT_COMPILE
    auto out = Mat<R>::empty_like(matrix);
    MAT(out) = MAT(matrix).unaryExpr(utils::relu_operator<R>());
    if (graph::backprop_enabled)
        graph::emplace_back([matrix, out](){
            SAFE_GRAD(matrix).noalias() += (MAT(out).unaryExpr(utils::max_operator<R>()).array() * GRAD(out).array()).matrix();
        });
    return out;
    #else
    return Mat<R>(1,1);
    #endif
}

template<typename R>
Mat<R> MatOps<R>::abs(Mat<R> matrix) {
    #ifndef DONT_COMPILE
    auto out = Mat<R>::empty_like(matrix);
    MAT(out) = MAT(matrix).array().abs().matrix();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix, out](){
            SAFE_GRAD(matrix).noalias() += (MAT(matrix).unaryExpr(utils::sign_operator<R>()).array() * GRAD(out).array()).matrix();
        });
    return out;
    #else
    return Mat<R>(1,1);
    #endif
}

template<typename R>
Mat<R> MatOps<R>::mul(
        Mat<R> matrix1,
        Mat<R> matrix2) {
    assert2(matrix1.dims(1) == matrix2.dims(0), "matrix product dimensions misaligned.");
    Mat<R> out (matrix1.dims(0), matrix2.dims(1), false);

    DALI_FUNCTION_3_MUT(TensorOps::dot, MAT(matrix1), MAT(matrix2), MAT(out),
                                        NO_TRANSPOSE, NO_TRANSPOSE, OVERWRITE);


    if (graph::backprop_enabled)
        graph::emplace_back([matrix1, matrix2, out](){
            if (!matrix1.constant) {
                // matrix1.g += out.g <dot> matrix2.T
                DALI_FUNCTION_3_MUT(TensorOps::dot, GRAD(out),    MAT(matrix2), GRAD(matrix1),
                                                    NO_TRANSPOSE, TRANSPOSE,    ADD_TO_EXISTING);
            }
            if (!matrix2.constant) {
                // matrix2.g += matrix1.T <dot> out.g
                DALI_FUNCTION_3_MUT(TensorOps::dot, GRAD(matrix1), GRAD(out),       GRAD(matrix2),
                                                    TRANSPOSE,     NO_TRANSPOSE,    ADD_TO_EXISTING);
            }
        });
    return out;
}

template<typename R>
Mat<R> MatOps<R>::quadratic_form(
        Mat<R> left,
        Mat<R> weights,
        Mat<R> right) {
    #ifndef DONT_COMPILE

    assert2(weights.dims(1) == right.dims(0), "Quadratic form right matrix has wrong dimensions.");
    assert2(left.dims(0) == weights.dims(0) , "Quadratic form left matrix has wrong dimensions.");

    Mat<R> out (
        left.dims(1),
        right.dims(1),
        false);
    if (graph::backprop_enabled) {
        auto left_side_mul = std::make_shared<Eigen::Matrix<R, Eigen::Dynamic, Eigen::Dynamic>>(
            left.dims(1), weights.dims(1)
        );
        (*left_side_mul) = MAT(left).transpose() * MAT(weights);
        MAT(out) = (*left_side_mul) * MAT(right);
        graph::emplace_back([left_side_mul, left, weights, right, out](){
            SAFE_GRAD(right).noalias() += (*left_side_mul).transpose() * GRAD(out);
            auto LeftT_dot_weights_grad = GRAD(out) * (MAT(right).transpose());
            SAFE_GRAD(left).noalias() += (
                LeftT_dot_weights_grad * MAT(weights).transpose()
            ).transpose();
            SAFE_GRAD(weights).noalias() += MAT(left) * LeftT_dot_weights_grad;
        });
    } else {
        MAT(out) = MAT(left).transpose() * MAT(weights) * MAT(right);
    }
    return out;
    #else
    return Mat<R>(1,1);
    #endif
}

template<typename R>
Mat<R> MatOps<R>::mul_with_bias(
        Mat<R> matrix1,
        Mat<R> matrix2,
        Mat<R> bias) {
    #ifndef DONT_COMPILE
    assert2(matrix1.dims(1) == matrix2.dims(0), "matmul dimensions misaligned.");
    if (matrix1.dims(0) != bias.dims(0) || bias.dims(1) != 1)
        throw std::invalid_argument("Matrices cannot be multiplied with broadcast, they do not have the same dimensions.");
    Mat<R> out (
            matrix1.dims(0),
            matrix2.dims(1),
            false);
    MAT(out) = ((MAT(matrix1) * MAT(matrix2)).colwise() + MAT(bias).col(0)).matrix();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix1, matrix2, bias, out]() {
            SAFE_GRAD(matrix1).noalias() += (GRAD(out)) * ((MAT(matrix2)).transpose());
            SAFE_GRAD(matrix2).noalias() += MAT(matrix1).transpose() * (GRAD(out));
            SAFE_GRAD(bias).noalias()    += GRAD(out).rowwise().sum().matrix();
        });
    return out;
    #else
    return Mat<R>(1,1);
    #endif
}

template<typename R>
Mat<R> MatOps<R>::mul_add_broadcast_mul_with_bias(
        Mat<R> matrix1,
        Mat<R> input_to_1,
        Mat<R> matrix2,
        Mat<R> input_to_2,
        Mat<R> bias) {
    #ifndef DONT_COMPILE
    assert2(matrix1.dims(1) == input_to_1.dims(0), "matmul 1 dimensions misaligned.");
    if (matrix2.dims(1) != input_to_2.dims(0))
        throw std::invalid_argument("matmul 2 dimensions misaligned.");
    if (matrix2.dims(0) != bias.dims(0) || matrix1.dims(0) != bias.dims(0) || input_to_1.dims(1) != 1 || bias.dims(1) != 1)
        throw std::invalid_argument("Matrices cannot be shigamizood, they do not have the same dimensions.");
    Mat<R> out (
            matrix1.dims(0),
            input_to_2.dims(1),
            false);
    // both input to 1 and bias are columns,
    // so we add both of those before adding the true matrix
    // product in broadcasted form
    MAT(out) = (
          (
              (
                  (MAT(matrix2) * MAT(input_to_2))
              )
          ).colwise() + (MAT(bias) + (MAT(matrix1) * MAT(input_to_1))).col(0)
      ).matrix();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix1, input_to_1, matrix2, input_to_2, bias, out] () {
            // first multiply:
            // broadcasting input means taking outer product here:
            SAFE_GRAD(matrix1) += ((GRAD(out)).rowwise().sum() * ((MAT(input_to_1)).transpose()));
            // broadcasting output means sum after the reverse product here:
            SAFE_GRAD(input_to_1).noalias() += (
                MAT(matrix1).transpose() * (GRAD(out))
            ).rowwise().sum();
            // second multiply:
            SAFE_GRAD(matrix2).noalias() += (GRAD(out)) * ((MAT(input_to_2)).transpose());

            SAFE_GRAD(input_to_2).noalias() += MAT(matrix2).transpose() * (GRAD(out));
            // bias vector:
            SAFE_GRAD(bias).noalias() += GRAD(out).rowwise().sum();
        });
    return out;
    #else
    return Mat<R>(1,1);
    #endif
}


template<typename R>
Mat<R> MatOps<R>::mul_add_mul_with_bias(std::initializer_list<Mat<R>> matrices) {
    vector<Mat<R>> matrices_vector(matrices);
    return mul_add_mul_with_bias(matrices_vector);
}

template<typename R>
Mat<R> MatOps<R>::mul_add_mul_with_bias(const vector<Mat<R>>& matrices) {
    #ifndef DONT_COMPILE
    // broacast to largest input size
    dim_t max_broadcast = matrices[1].dims(1);
    for (auto matrices_ptr = matrices.begin()+1; matrices_ptr < matrices.end(); matrices_ptr+=2) {
        max_broadcast = std::max(max_broadcast, matrices_ptr->dims(1));
    }

    Mat<R> out(
            matrices[0].dims(0),
            max_broadcast,
            true);
    DEBUG_ASSERT_MAT_NOT_NAN(out)
    auto matrices_ptr = matrices.begin();
    while (matrices_ptr != (matrices.end() - 1)) {
        DEBUG_ASSERT_MAT_NOT_NAN(*matrices_ptr)
        DEBUG_ASSERT_MAT_NOT_NAN(*(matrices_ptr + 1))
        DEBUG_ASSERT_MAT_NOT_NAN(out)
        // inputs must either match the broadcasted size, or be broadcastable by having their
        // inner dimension be 1 (a column vector essentially)
        assert(((matrices_ptr+1)->dims(1) == max_broadcast) || ((matrices_ptr+1)->dims(1) == 1));
        if ((matrices_ptr+1)->dims(1) == max_broadcast) {
            MAT(out) += MAT(*matrices_ptr) * MAT(*(matrices_ptr + 1));
        } else {
            auto el = MAT(*matrices_ptr) * MAT(*(matrices_ptr + 1));
            MAT(out).colwise() += el.col(0);
        }
        DEBUG_ASSERT_MAT_NOT_NAN(out)
        matrices_ptr+=2;
    }

    DEBUG_ASSERT_MAT_NOT_NAN(matrices.back());
    MAT(out).colwise() += MAT(matrices.back()).col(0);
    if (graph::backprop_enabled)
        graph::emplace_back([matrices, out, max_broadcast](){
            auto matrices_ptr = matrices.begin();
            while (matrices_ptr != (matrices.end() - 1)) {
                if ((matrices_ptr+1)->dims(1) == max_broadcast) {
                    SAFE_GRAD(*matrices_ptr).noalias()   += GRAD(out) * MAT(*(matrices_ptr+1)).transpose();
                    SAFE_GRAD(*(matrices_ptr+1)).noalias() += MAT(*matrices_ptr).transpose() * (GRAD(out));
                } else {
                    // broadcasting input means taking outer product here:
                    SAFE_GRAD(*matrices_ptr).noalias() += (GRAD(out).rowwise().sum() * (MAT(*(matrices_ptr+1)).transpose()));
                    // broadcasting output means sum after the reverse product here:
                    SAFE_GRAD(*(matrices_ptr+1)).noalias() += (
                        MAT(*matrices_ptr).transpose() * GRAD(out)
                    ).rowwise().sum();
                }
                matrices_ptr+=2;
            }
            SAFE_GRAD(matrices.back()).noalias() += GRAD(out).rowwise().sum();
        });

    DEBUG_ASSERT_NOT_NAN(MAT(out));
    return out;
    #else
    return Mat<R>(1,1);
    #endif
}

// operation of the form (A * x + B * y) + C, called with mul_add_mul_with_bias(A, x, B, y, C)
template<typename R>
Mat<R> MatOps<R>::mul_add_mul_with_bias(
        Mat<R> matrix1,
        Mat<R> input_to_1,
        Mat<R> matrix2,
        Mat<R> input_to_2,
        Mat<R> bias) {
    #ifndef DONT_COMPILE
    DEBUG_ASSERT_NOT_NAN(MAT(bias));
    assert2(matrix1.dims(1) == input_to_1.dims(0), "matmul 1 dimensions misaligned.");
    if (matrix2.dims(1) != input_to_2.dims(0))
        throw std::invalid_argument("matmul 2 dimensions misaligned.");
    if (matrix2.dims(0) != bias.dims(0) || matrix1.dims(0) != bias.dims(0) || bias.dims(1) != 1)
        throw std::invalid_argument("Matrices cannot be shigamizood, they do not have the same dimensions.");
    if (input_to_1.dims(1) != input_to_2.dims(1)) {
        if (input_to_1.dims(1) == 1) {
            return mul_add_broadcast_mul_with_bias(matrix1, input_to_1, matrix2, input_to_2, bias);
        }
        return mul_add_broadcast_mul_with_bias(matrix2, input_to_2, matrix1, input_to_1, bias);
    }
    Mat<R> out (
            matrix1.dims(0),
            input_to_1.dims(1),
            false);
    MAT(out) = (
                  (
                      (
                          (MAT(matrix1) * MAT(input_to_1)) +
                          (MAT(matrix2) * MAT(input_to_2))
                      )
                  ).colwise() + MAT(bias).col(0)
              ).matrix();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix1, input_to_1, matrix2, input_to_2, bias, out](){
            // first multiply:
            // broadcasting input means taking outer product here:
            SAFE_GRAD(matrix1)               += (GRAD(out) * (MAT(input_to_1)).transpose());
            // broadcasting output means sum after the reverse product here:
            SAFE_GRAD(input_to_1).noalias() += MAT(matrix1).transpose() * (GRAD(out));
            // second multiply:
            SAFE_GRAD(matrix2).noalias()     += (GRAD(out)) * (MAT(input_to_2)).transpose();

            SAFE_GRAD(input_to_2).noalias()  += MAT(matrix2).transpose() * (GRAD(out));
            // bias vector:
            SAFE_GRAD(bias).noalias()        += GRAD(out).rowwise().sum();
        });
    return out;
    #else
    return Mat<R>(1,1);
    #endif
}

template<typename R>
Mat<R> MatOps<R>::rows_pluck(
        Mat<R> matrix,
        Indexing::Index indices
        ) {
    #ifndef DONT_COMPILE
    Mat<R> out (
        matrix.dims(1),
        indices.size(),
        false);

    for (std::size_t offset = 0; offset < indices.size(); ++offset) {
        MAT(out).col(offset) = MAT(matrix).row(indices[offset]).transpose();
    }
    if (graph::backprop_enabled) {
        graph::emplace_back([matrix, out, indices](){
            auto index_ptr = indices.data();
            for (std::size_t i = 0; i < out.dims(1); ++i) {
                // for each row do the same operation as for row_pluck:
                SAFE_GRAD(matrix).row(*index_ptr).noalias() += GRAD(out).col(i).transpose();
                index_ptr++;
            }
        });
    }
    return out;
    #else
    return Mat<R>(1,1);
    #endif
}

template<typename R>
Mat<R> MatOps<R>::dropout(
        Mat<R> matrix,
        R drop_prob) {
    #ifndef DONT_COMPILE

    assert(0.0 <= drop_prob && drop_prob <= 1.0);

    // no dropout happens.
    if (drop_prob < 1e-6)
        return matrix;

    auto out = Mat<R>::empty_like(matrix);

    auto bool_mat = std::make_shared<Eigen::Matrix<R, Eigen::Dynamic, Eigen::Dynamic>>(
        matrix.dims(0),
        matrix.dims(1)
    );

    std::default_random_engine generator;
    std::bernoulli_distribution distribution(1.0 - drop_prob);
    std::random_device rd;
    generator.seed(rd());

    auto data_ptr = MAT(matrix).data();
    auto out_ptr  = MAT(out).data();
    auto bool_ptr = bool_mat->data();

    for (int i = 0; i < matrix.number_of_elements();++i) {
        (*bool_ptr) = distribution(generator) ? 1.0 : 0.0;
        (*out_ptr) = (*bool_ptr) > 0 ? *data_ptr : 0.0;
        out_ptr++;
        data_ptr++;
        bool_ptr++;
    }

    if (graph::backprop_enabled) {
        graph::emplace_back([matrix, out, bool_mat](){
            SAFE_GRAD(matrix) += (GRAD(out).array() * (*bool_mat).array()).matrix();
        });
    }
    return out;
    #else
    return Mat<R>(1,1);
    #endif
}

template<typename R>
Mat<R> MatOps<R>::dropout_normalized(
        Mat<R> matrix,
        R drop_prob) {
    #ifndef DONT_COMPILE

    assert(0.0 <= drop_prob && drop_prob <= 1.0);

    // no dropout happens.
    if (drop_prob < 1e-6)
        return matrix;

    auto out = Mat<R>::empty_like(matrix);

    auto bool_mat = std::make_shared<Eigen::Matrix<R, Eigen::Dynamic, Eigen::Dynamic>>(
        matrix.dims(0),
        matrix.dims(1)
    );

    std::default_random_engine generator;
    std::bernoulli_distribution distribution(1.0 - drop_prob);
    std::random_device rd;
    generator.seed(rd());

    auto data_ptr = MAT(matrix).data();
    auto out_ptr  = MAT(out).data();
    auto bool_ptr = bool_mat->data();

    R normalized_drop_prob = 1.0 / (1.0 - drop_prob);
    for (unsigned int i = 0; i < matrix.number_of_elements();++i) {
        (*bool_ptr) = distribution(generator) ? normalized_drop_prob : 0.0;
        (*out_ptr) = (*bool_ptr) > 0 ? *data_ptr : 0.0;
        out_ptr++;
        data_ptr++;
        bool_ptr++;
    }

    if (graph::backprop_enabled) {
        graph::emplace_back([matrix, out, bool_mat](){
            SAFE_GRAD(matrix) += (GRAD(out).array() * (*bool_mat).array()).matrix();
        });
    }
    return out;
    #else
    return Mat<R>(1,1);
    #endif
}

template<typename R>
vector<Mat<R>> MatOps<R>::dropout_normalized(
        const vector<Mat<R>>& matrices,
        R drop_prob) {
    #ifndef DONT_COMPILE
    vector<Mat<R>> dropped_matrices;
    dropped_matrices.reserve(matrices.size());
    for (auto& mat : matrices) {
        dropped_matrices.emplace_back(dropout_normalized(mat, drop_prob));
    }
    return dropped_matrices;
    #else
    return {Mat<R>(1,1)};
    #endif
}

template<typename R>
vector<Mat<R>> MatOps<R>::dropout(
        const vector<Mat<R>>& matrices,
        R drop_prob) {
    #ifndef DONT_COMPILE
    vector<Mat<R>> dropped_matrices;
    dropped_matrices.reserve(matrices.size());
    for (auto& mat : matrices) {
        dropped_matrices.emplace_back(dropout(mat, drop_prob));
    }
    return dropped_matrices;
    #else
    return {Mat<R>(1,1)};
    #endif
}

template<typename R>
Mat<R> MatOps<R>::fast_dropout(Mat<R> matrix) {
    #ifndef DONT_COMPILE
    auto out = Mat<R>::empty_like(matrix);

    auto randn_mat = std::make_shared<Eigen::Matrix<R, Eigen::Dynamic, Eigen::Dynamic>>(
        matrix.dims(0),
        matrix.dims(1)
    );

    std::default_random_engine generator;
    std::normal_distribution<R> distribution(1.0, 1.0);
    std::random_device rd;
    generator.seed(rd());

    auto data_ptr = MAT(matrix).data();
    auto out_ptr  = MAT(out).data();
    auto randn_ptr = randn_mat->data();

    for (unsigned int i = 0; i < matrix.number_of_elements();++i) {
        (*randn_ptr) = distribution(generator);
        (*out_ptr) = (*randn_ptr) * *data_ptr;
        out_ptr++;
        data_ptr++;
        randn_ptr++;
    }

    if (graph::backprop_enabled) {
        graph::emplace_back([matrix, out, randn_mat](){
            SAFE_GRAD(matrix) += (GRAD(out).array() * (*randn_mat).array()).matrix();
        });
    }
    return out;
    #else
    return Mat<R>(1,1);
    #endif
}

template<typename R>
Mat<R> MatOps<R>::rows_cols_pluck(
        Mat<R> matrix,
        Indexing::Index row_indices,
        Indexing::Index col_indices) {
    #ifndef DONT_COMPILE
    if (row_indices.size() != col_indices.size())
        throw std::invalid_argument("Cannot pluck column row pairs, not the "
            "same amount of row and column indices.");
        Mat<R> out (
            1,
            row_indices.size(),
            false);
        for (int offset = 0; offset < row_indices.size(); ++offset)
            MAT(out)(offset) = MAT(matrix)(row_indices[offset], col_indices[offset]);
    if (graph::backprop_enabled && !matrix.constant) {
        graph::emplace_back([matrix, out, row_indices, col_indices](){
            auto row_index_ptr = row_indices.data();
            auto col_index_ptr = col_indices.data();
            for (int i = 0; i < out.dims(1); ++i) {
                // for each row do the same operation as for row_pluck:
                GRAD(matrix)(*row_index_ptr, *col_index_ptr) += GRAD(out)(i);
                row_index_ptr++;
                col_index_ptr++;
            }
        });
    }
    return out;
    #else
    return Mat<R>(1,1);
    #endif
}

template<typename R>
Mat<R> MatOps<R>::row_pluck(
        Mat<R> matrix,
        int row) {
    #ifndef DONT_COMPILE
    Mat<R> out (matrix.dims(1), 1, false);
    MAT(out) = MAT(matrix).row(row).transpose();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix, out, row]() {
            SAFE_GRAD(matrix).row(row).noalias() += GRAD(out).col(0).transpose();
        });
    return out;
    #else
    return Mat<R>(1,1);
    #endif
}

template<typename R>
Mat<R> MatOps<R>::col_pluck(
        Mat<R> matrix,
        int col) {
    #ifndef DONT_COMPILE
    Mat<R> out (matrix.dims(0), 1, false);
    MAT(out) = MAT(matrix).col(col);
    if (graph::backprop_enabled)
        graph::emplace_back([matrix, out, col]() {
            SAFE_GRAD(matrix).col(col).noalias() += GRAD(out).col(0).transpose();
        });
    return out;
    #else
    return Mat<R>(1,1);
    #endif
}

template<typename R>
Mat<R> MatOps<R>::consider_constant_if(
        Mat<R> matrix,
        bool should_consider_constant) {
    if (should_consider_constant)
        return consider_constant(matrix);
    return matrix;
}

template<typename R>
Mat<R> MatOps<R>::consider_constant(
        Mat<R> matrix
        ) {
    // perform a copy of the matrix that references
    // everything and owns nothing. A true nomad.
    Mat<R> out(matrix, false, false);
    out.constant = true;
    return out;
}

template<typename R>
vector<size_t> MatOps<R>::argsort_rowwise(Mat<R> m) {
    #ifndef DONT_COMPILE
    // initialize original index locations
    vector<size_t> idx(m.dims(0));
    for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

    // sort indexes based on comparing values in v
    sort(idx.begin(), idx.end(), [&m](size_t i1, size_t i2) {
        return MAT(m)(i1) < MAT(m)(i2);
    });
    return idx;
    #else
    return {};
    #endif
}

template<typename R>
vector<size_t> MatOps<R>::argsort(const vector<Mat<R>>& v) {
    #ifndef DONT_COMPILE
    // initialize original index locations
    vector<size_t> idx(v.size());
    for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

    // sort indexes based on comparing values in v
    sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return MAT(v[i1])(0) < MAT(v[i2])(0);});

    return idx;
    #else
    return {};
    #endif
}

template<typename R>
void MatOps<R>::resize(const Mat<R>& mat, dim_t n, dim_t d) {
    #ifndef DONT_COMPILE
    mat.w()->dims[0] = n;
    mat.w()->dims[1] = d;
    MAT(mat).conservativeResize(n, d);
    GRAD(mat).conservativeResize(n, d);
    #else
    #endif
}


template<typename R>
int MatOps<R>::argmax(const Mat<R>& mat) {
    #ifndef DONT_COMPILE
    int i = 0;
    R current_max = -std::numeric_limits<R>::infinity();
    auto ptr = mat.w()->data();
    for (int j = 0; j < mat.number_of_elements(); j++) {
        if (*ptr > current_max) {
            current_max = *ptr;
            i = j;
        }
        ptr++;
    }
    return i;
    #else
    return 0;
    #endif
}

template<typename R>
int MatOps<R>::argmax_slice(const Mat<R>& mat, int lower, int upper) {
    #ifndef DONT_COMPILE
    int i = 0;
    R current_max = -std::numeric_limits<R>::infinity();
    auto ptr = mat.w()->data();
    for (int j = lower; j < upper; j++) {
        if (*ptr > current_max) {
            current_max = *ptr;
            i = j;
        }
        ptr++;
    }
    return i;
    #else
    return 0;
    #endif
}



template<typename R>
bool MatOps<R>::equals(Mat<R> a, Mat<R> b) {
    // wrong dimensions
    if (a.dims() != b.dims())
        return false;

    return DALI_FUNCTION_2(TensorOps::equals, MAT(a), MAT(b), a.number_of_elements());
}

template<typename R>
bool MatOps<R>::allclose(Mat<R> a, Mat<R> b, R tol) {
    if (a.dims() != b.dims())
        return false;
    return DALI_FUNCTION_2(TensorOps::allclose, MAT(a), MAT(b), a.number_of_elements(), tol);
}

template<typename R>
bool MatOps<R>::grad_allclose(Mat<R> a, Mat<R> b, R tol) {
    if (a.dims() != b.dims())
        return false;
    return DALI_FUNCTION_2(TensorOps::allclose, GRAD(a), GRAD(b), a.number_of_elements(), tol);
}



template class MatOps<float>;
template class MatOps<double>;
