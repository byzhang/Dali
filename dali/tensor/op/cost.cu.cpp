#include "dali/tensor/op/cost.h"

#include "dali/tensor/__MatMacros__.h"
#include "dali/math/TensorOps.h"
#include "dali/math/LazyTensor.h"

using std::vector;
using namespace TensorOps;
using std::make_shared;

namespace matops {

    // performs row wise normalization
    template<typename R>
    Mat<R> Cost<R>::softmax_no_grad_rowwise(Mat<R> matrix, R temperature) {
        auto out = Mat<R>::empty_like(matrix);
        MAT(out) = MAT(matrix).wrapper().softmax_rowwise(temperature);
        return out;
    }

    template<typename R>
    Mat<R> Cost<R>::softmax_rowwise(Mat<R> matrix, R temperature) {
        Mat<R> out = Cost<R>::softmax_no_grad_rowwise(matrix, temperature);
        if (graph::backprop_enabled() && !matrix.constant)
            graph::emplace_back([matrix, temperature, out]() mutable {
                TensorInternal<R, 1> sm_times_dy_colsum( mshadow::Shape1(matrix.dims(0)));
                sm_times_dy_colsum = sum_cols(MAT(out).wrapper() * GRAD(out).wrapper());

                GRAD(matrix) += (
                      MAT(out).wrapper() * GRAD(out).wrapper()
                    - MAT(out).wrapper() * sm_times_dy_colsum.wrapper().template broadcast<0>(GRAD(out).shape)
                ) / temperature;
            });
        return out;
    }

    template<typename R>
    vector<Mat<R>> Cost<R>::softmax_no_grad_colwise(const vector<Mat<R>>& matrices, R temperature) {
        vector<Mat<R>> out;
        out.reserve(matrices.size());
        ASSERT2(matrices.size() > 0, "Must be a non empty list of vectors to softmax.");
        R layer_max = matrices.front().w(0);
        for (auto& mat : matrices) {
            ASSERT2(mat.dims(0) == 1 && mat.dims(1) == 1, "Softmax on a vector must be made on 1x1 matrices only.");
            layer_max = std::max(layer_max, mat.w(0));
        }
        R total = 0.0;
        for (auto& mat : matrices) {
            out.emplace_back(1,1);
            out.back().w(0) = std::exp(mat.w(0) - layer_max) / temperature;
            total += out.back().w(0);
        }
        for (auto& mat : out) {
            mat.w(0) /= total;
        }
        return out;
    }

    // performs column wise normalization
    template<typename R>
    vector<Mat<R>> Cost<R>::softmax(vector<Mat<R>>& matrices, R temperature) {
        vector<Mat<R>> out = Cost<R>::softmax_no_grad_colwise(matrices, temperature);
        if (graph::backprop_enabled())
            graph::emplace_back([temperature, out, matrices]() mutable {
                R colwise_sums = 0.0;

                for (int i = 0; i < out.size(); i++) {
                    colwise_sums += out[i].w(0) * out[i].dw(0);
                }

                for (int i = 0; i < out.size(); i++) {
                    if (!matrices[i].constant) {
                        matrices[i].dw(0) += (
                            out[i].w(0) * out[i].dw(0) - out[i].w(0) * colwise_sums
                        ) / temperature;
                    }
                }
            });
        return out;
    }

    template<typename R>
    Mat<R> Cost<R>::softmax_colwise(Mat<R> matrix, R temperature) {
        Mat<R> out     = Cost<R>::softmax_no_grad_colwise(matrix, temperature);

        if (graph::backprop_enabled() && !matrix.constant)
            graph::emplace_back([matrix, temperature, out]() mutable {

                TensorInternal<R, 1> sm_times_dy_rowsum( mshadow::Shape1(matrix.dims(1)));
                sm_times_dy_rowsum = sum_rows(MAT(out).wrapper() * GRAD(out).wrapper());

                GRAD(matrix) += (
                      MAT(out).wrapper() * GRAD(out).wrapper()
                    - MAT(out).wrapper() * sm_times_dy_rowsum.wrapper().template broadcast<1>(GRAD(out).shape)
                )       / temperature;
            });
        return out;
    }

    template<typename R>
    Mat<R> Cost<R>::softmax_no_grad_colwise(Mat<R> matrix, R temperature) {
        auto out = Mat<R>::empty_like(matrix);
        MAT(out) = MAT(matrix).wrapper().softmax_colwise(temperature);
        return out;
    }

    template<typename R>
    Mat<R> Cost<R>::sigmoid_binary_cross_entropy(Mat<R> matrix, R t) {
        ASSERT2(0 <= t && t <= 1,
            "Target value for sigmoid_binary_cross_entropy must be a probability between 0 and 1.");
        auto out = Mat<R>::empty_like(matrix);

        // take sigmoid and keep it for backprop
        TensorInternal<R, 2> sigmoided_input(MAT(out).shape);
        sigmoided_input = F<op::sigmoid<R>>(MAT(matrix).wrapper());

        // take element wise binary cross entropy between target probability
        // and obtained probability
        MAT(out) = F<op::binary_cross_entropy<R>>(sigmoided_input.wrapper(), t);

        if (graph::backprop_enabled())
            graph::emplace_back([matrix, t, out, sigmoided_input]() mutable {
                SAFE_GRAD(matrix) += (sigmoided_input.wrapper() - t) * GRAD(out).wrapper();
            });
        return out;
    }

    template<typename R>
    Mat<R> Cost<R>::sigmoid_binary_cross_entropy(Mat<R> matrix, Mat<R> target) {
        ASSERT2(matrix.dims(0) == target.dims(0) && matrix.dims(1) == target.dims(1),
            "Matrix and target must have same dimension");

        Mat<R> out = Mat<R>::empty_like(matrix);

        // take sigmoid and keep it for backprop
        TensorInternal<R, 2> sigmoided_input(MAT(out).shape);
        sigmoided_input = F<op::sigmoid<R>>(MAT(matrix).wrapper());

        MAT(out) = F<op::binary_cross_entropy<R>>(sigmoided_input.wrapper(), MAT(target).wrapper());

        DEBUG_ASSERT_NOT_NAN(MAT(out));

        if (graph::backprop_enabled())
            graph::emplace_back([matrix, target, out, sigmoided_input]() mutable {
                SAFE_GRAD(matrix) += (sigmoided_input.wrapper() - MAT(target).wrapper()) * GRAD(out).wrapper();
                DEBUG_ASSERT_GRAD_NOT_NAN(matrix);
            });
        return out;
    }

    template<typename R>
    Mat<R> Cost<R>::binary_cross_entropy(Mat<R> matrix, R t) {
        assert(0 <= t && t <= 1);
        assert(matrix.dims().size() > 1);
        Mat<R> out =  Mat<R>(
            matrix.dims(0),
            matrix.dims(1),
            weights<R>::empty());

        MAT(out) = F<op::binary_cross_entropy<R>>(MAT(matrix).wrapper(), t);

        DEBUG_ASSERT_MAT_NOT_NAN(out);

        if (graph::backprop_enabled())
            graph::emplace_back([matrix, t, out]() mutable {
                SAFE_GRAD(matrix) += (
                    F<op::binary_cross_entropy_grad<R>>(
                        MAT(matrix).wrapper(), t
                    ) * GRAD(out).wrapper()
                );
                DEBUG_ASSERT_GRAD_NOT_NAN(matrix);
            });
        return out;
    }

    template<typename R>
    Mat<R> Cost<R>::binary_cross_entropy(Mat<R> matrix, Mat<R> target) {
        ASSERT2(matrix.dims(0) == target.dims(0) && matrix.dims(1) == target.dims(1),
            "Matrix and target must have same dimension");

        Mat<R> out = Mat<R>::empty_like(matrix);
        MAT(out) = F<op::binary_cross_entropy<R>>(MAT(matrix).wrapper(), MAT(target).wrapper());

        DEBUG_ASSERT_NOT_NAN(MAT(out));

        if (graph::backprop_enabled())
            graph::emplace_back([matrix, target, out]() mutable {
                SAFE_GRAD(matrix) += (
                    F<op::binary_cross_entropy_grad<R>>(
                        MAT(matrix).wrapper(), MAT(target).wrapper()
                    ) * GRAD(out).wrapper()
                );
                DEBUG_ASSERT_GRAD_NOT_NAN(matrix);
            });
        return out;
    }



    template<typename R>
    Mat<R> Cost<R>::cross_entropy_colwise(Mat<R> matrix, uint answer_idx) {
        DEBUG_ASSERT_BOUNDS(MAT(matrix),0.0,1.0 + EPS);
        ASSERT2(answer_idx < matrix.dims(0),
            utils::MS() << "Cross entropy target (" << answer_idx << ") must be less than"
                           " number of rows in predicted matrix (" << matrix.dims(0) << ").");
        Mat<R> out =  Mat<R>(1, matrix.dims(1), weights<R>::empty());
        MAT(out).ravel() =  (R)-1.0 * F<op::log<R>>(MAT(matrix)[answer_idx].wrapper());

        DEBUG_ASSERT_MAT_NOT_NAN(out);

        if (graph::backprop_enabled())
            graph::emplace_back([matrix, answer_idx, out]() mutable {
                SAFE_GRAD(matrix)[answer_idx] -= F<op::inv<R>>(MAT(matrix)[answer_idx].wrapper()) * GRAD(out).ravel().wrapper();
            });
        return out;
    }



    template<typename R>
    Mat<R> Cost<R>::cross_entropy_rowwise(Mat<R> matrix, uint answer_idx) {
        DEBUG_ASSERT_BOUNDS(MAT(matrix),0.0,1.0 + EPS);
        ASSERT2(answer_idx < matrix.dims(1),
            utils::MS() << "Cross entropy target (" << answer_idx << ") must be less than"
                           " number of columns in predicted matrix (" << matrix.dims(1) << ").");
        Mat<R> out =  Mat<R>(matrix.dims(0), 1, weights<R>::empty());
        TensorOps::col_pluck(MAT(out).ravel(), MAT(matrix), answer_idx);

        MAT(out).ravel() =  (R)-1.0 * F<op::log<R>>(MAT(out).ravel().wrapper());

        DEBUG_ASSERT_MAT_NOT_NAN(out);

        if (graph::backprop_enabled())
            graph::emplace_back([matrix, answer_idx, out]() mutable {
                TensorInternal<R,1> temp(mshadow::Shape1(matrix.dims(0)));
                TensorOps::col_pluck(temp, MAT(matrix), answer_idx);

                temp = (R)-1.0 * F<op::inv<R>>(temp.wrapper()) * GRAD(out).ravel().wrapper();

                TensorOps::col_pluck_backward(GRAD(matrix), temp, answer_idx);
            });
        return out;
    }

    template<typename R>
    Mat<R> Cost<R>::cross_entropy_colwise(Mat<R> matrix, Mat<int> targets) {
        ASSERT2(targets.number_of_elements() == matrix.dims(1), "Number of cols must be equal to the number of target in Cross Entropy colwise");
        Mat<R> out =  Mat<R>(1, targets.number_of_elements(), weights<R>::empty());

        select_from_cols(MAT(out), MAT(matrix), targets.w().ravel());

        MAT(out) = (R)-1.0 * F<op::log<R>>(MAT(out).wrapper());

        if (graph::backprop_enabled() && !matrix.constant) {
            graph::emplace_back([matrix, out, targets]() mutable {
                cross_entropy_colwise_backward(MAT(matrix), GRAD(matrix), GRAD(out), targets.w().ravel());
            });
        }
        return out;
    }

    template<typename R>
    Mat<R> Cost<R>::cross_entropy_rowwise(Mat<R> matrix, Mat<int> targets) {
        ASSERT2(targets.number_of_elements() == matrix.dims(0), "Number of rows must be equal to the number of target in Cross Entropy rowwise");
        Mat<R> out =  Mat<R>(targets.number_of_elements(), 1, weights<R>::empty());

        select_from_rows(MAT(out), MAT(matrix), targets.w().ravel());

        MAT(out) = (R)-1.0 * F<op::log<R>>(MAT(out).wrapper());

        if (graph::backprop_enabled() && !matrix.constant) {
            graph::emplace_back([matrix, out, targets]() mutable {
                cross_entropy_rowwise_backward(MAT(matrix), GRAD(matrix), GRAD(out), targets.w().ravel());
            });
        }
        return out;
    }

    template<typename R>
    Mat<R> Cost<R>::cross_entropy(Mat<R> matrix, Mat<R> target) {
        ASSERT2(matrix.dims(0) == target.dims(0) && matrix.dims(1) == target.dims(1),
            "Matrix and target must have same dimension");

        Mat<R> out = Mat<R>::empty_like(matrix);
        MAT(out) = (R)-1.0 * MAT(target).wrapper() * F<op::log<R>>(MAT(matrix).wrapper());

        DEBUG_ASSERT_NOT_NAN(MAT(out));

        if (graph::backprop_enabled())
            graph::emplace_back([matrix, target, out]() mutable {
                SAFE_GRAD(matrix) -= F<op::inv<R>>(MAT(matrix).wrapper()) * MAT(target).wrapper() * GRAD(out).wrapper();
                SAFE_GRAD(target) -= F<op::log<R>>(MAT(matrix).wrapper()) * GRAD(out).wrapper();
            });
        return out;
    }

    /////////////////////// SOFTMAX CROSSENTROPY COLWISE //////////////////////////////////

    template<typename R>
    Mat<R> Cost<R>::softmax_cross_entropy_colwise(Mat<R> matrix, uint answer_idx) {
        Mat<int> target(1,1);
        MAT(target) = answer_idx;
        return softmax_cross_entropy_colwise(matrix, target);
    }

    template<typename R>
    Mat<R> Cost<R>::softmax_cross_entropy_colwise(Mat<R> matrix, Mat<int> targets) {
        ASSERT2(targets.number_of_elements() == matrix.dims(1),
                utils::MS() << "Softmax cross entropy: Number of targets ("
                            << targets.number_of_elements() << ") should equal number of input columns ("
                            << matrix.dims(1) << ")");
        Mat<R> out =  Mat<R>(1, targets.number_of_elements(), weights<R>::empty());
        Mat<R> probs = softmax_no_grad_colwise(matrix);
        select_from_cols(MAT(out), MAT(probs), targets.w().ravel());

        MAT(out) = (R)-1.0 * F<op::log<R>>(F<op::max_scalar<R>>(
              MAT(out).wrapper(), std::numeric_limits<R>::min()));
        if (graph::backprop_enabled()) {
            graph::emplace_back([matrix, probs, out, targets]() mutable {
                if (!matrix.constant) {
                    GRAD(matrix) += (
                        MAT(probs).wrapper() *
                        GRAD(out).ravel().wrapper().template broadcast<1>(MAT(probs).shape)
                    );

                    softmax_cross_entropy_colwise_backward(GRAD(matrix), GRAD(out), targets.w().ravel());
                }
            });
        }
        return out;
    }

    template<typename R>
    Mat<R> Cost<R>::softmax_cross_entropy_colwise(Mat<R> matrix, Indexing::Index targets) {
        Mat<int> targets_mat(1, targets.size());
        for (int i = 0; i < targets.size(); ++i) {
            targets_mat.w(i) = targets[i];
        }
        return softmax_cross_entropy_colwise(matrix, targets_mat);
    }


    /////////////////////// SOFTMAX CROSSENTROPY ROWWISE //////////////////////////////////


    template<typename R>
    Mat<R> Cost<R>::softmax_cross_entropy_rowwise(Mat<R> matrix, uint answer_idx) {
        Mat<int> target(1,1);
        MAT(target) = answer_idx;
        return softmax_cross_entropy_rowwise(matrix, target);
    }

    template<typename R>
    Mat<R> Cost<R>::softmax_cross_entropy_rowwise(Mat<R> matrix, Mat<int> targets) {
        ASSERT2(targets.number_of_elements() == matrix.dims(0),
                utils::MS() << "Softmax cross entropy: Number of targets ("
                            << targets.number_of_elements() << ") should equal number of input rows ("
                            << matrix.dims(0) << ")");

        Mat<R> out =  Mat<R>(targets.number_of_elements(), 1, weights<R>::empty());

        Mat<R> probs = softmax_no_grad_rowwise(matrix);

        select_from_rows(MAT(out), MAT(probs), targets.w().ravel());


        MAT(out) = (R)-1.0 * F<op::log<R>>(F<op::max_scalar<R>>(
              MAT(out).wrapper(), std::numeric_limits<R>::min()));

        if (graph::backprop_enabled()) {
            graph::emplace_back([matrix, probs, out, targets]() mutable {

                if (!matrix.constant) {
                    GRAD(matrix) += (
                        MAT(probs).wrapper() *
                        GRAD(out).ravel().wrapper().template broadcast<0>(MAT(probs).shape)
                    );

                    softmax_cross_entropy_rowwise_backward(GRAD(matrix), GRAD(out), targets.w().ravel());
                }
            });
        }
        return out;
    }

    template<typename R>
    Mat<R> Cost<R>::softmax_cross_entropy_rowwise(Mat<R> matrix, Indexing::Index targets) {
        Mat<int> targets_mat(targets.size(), 1);
        for (int i = 0; i < targets.size(); ++i) {
            targets_mat.w(i) = targets[i];
        }
        return softmax_cross_entropy_rowwise(matrix, targets_mat);
    }



    template<typename R>
    Mat<R> Cost<R>::margin_loss_rowwise(Mat<R> matrix, uint answer_idx, R margin) {
        // Exprected input is a column vector
        ASSERT2(answer_idx < matrix.dims(1),
            utils::MS() << "Target answer ("
                        << answer_idx
                        << ")must be less than number of "
                           "cols of matrix ("
                        << matrix.dims(1) << ").");
        Mat<R> error(matrix.dims(0), 1);

        auto target_column = matrix(NULL, answer_idx);

        for (int idx = 0; idx < matrix.dims(1); ++idx) {
            if (idx == answer_idx) continue;
            error = error + MatOps<R>::eltmax(matrix(NULL, idx) - target_column + margin, 0.0);
        }
        return error;
    }

    template<typename R>
    Mat<R> Cost<R>::margin_loss_colwise(Mat<R> matrix, uint answer_idx, R margin) {
        // Exprected input is a column vector
        ASSERT2(answer_idx < matrix.dims(0),
            utils::MS() << "Target answer ("
                        << answer_idx
                        << ")must be less than number of "
                           "rows of matrix ("
                        << matrix.dims(0) << ").");
        Mat<R> error(1, matrix.dims(1));
        for (int idx = 0; idx < matrix.dims(0); ++idx) {
            if (idx == answer_idx) continue;
            error = error + MatOps<R>::eltmax(matrix[idx] - matrix[answer_idx] + margin, 0.0);
        }
        return error;
    }

    template class Cost<float>;
    template class Cost<double>;
    template class Cost<int>;
}
