#include <chrono>
#include <vector>
#include <iomanip>
#include <gtest/gtest.h>

#include "dali/test_utils.h"
#include "dali/tensor/Index.h"
#include "dali/layers/Layers.h"
#include "dali/tensor/Mat.h"
#include "dali/tensor/MatOps.h"
#include "dali/tensor/Tape.h"
#include "dali/tensor/Solver.h"

using std::vector;
using std::chrono::milliseconds;

class MatrixTests : public MemorySafeTest {
  protected:
    static void SetUpTestCase() {
        dali_init();
    }
};


TEST_F(MatrixTests, sum_test) {
    auto A = Mat<R>(10, 20, weights<R>::uniform(2.0));
    auto res = A.sum();
    R sum = 0.0;
    for (int i = 0; i < A.number_of_elements(); ++i) {
        sum += A.w(i);
    }
    ASSERT_NEAR(sum, res.w(0), 1e-4);
}

TEST_F(MatrixTests, sum_rowwise) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return MatOps<R>::sum_rowwise(Xs[0]);
    };

    EXPERIMENT_REPEAT {
        auto A = Mat<R>(10, 20, weights<R>::uniform(2.0));
        expect_args_remain_on_gpu(functor, {A});
        EXPECT_TRUE(gradient_same(functor, {A}));
    }
}

TEST_F(MatrixTests, mean_colwise) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return MatOps<R>::mean_colwise(Xs[0]);
    };

    EXPERIMENT_REPEAT {
        auto A = Mat<R>(10, 20, weights<R>::uniform(2.0));
        expect_args_remain_on_gpu(functor, {A});
        EXPECT_TRUE(gradient_same(functor, {A}));
    }
}

TEST_F(MatrixTests, mean_rowwise) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return MatOps<R>::mean_rowwise(Xs[0]);
    };

    EXPERIMENT_REPEAT {
        auto A = Mat<R>(10, 20, weights<R>::uniform(2.0));
        expect_args_remain_on_gpu(functor, {A});
        EXPECT_TRUE(gradient_same(functor, {A}));
    }
}

TEST_F(MatrixTests, max_rowwise) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return MatOps<R>::max_rowwise(Xs[0]);
    };

    EXPERIMENT_REPEAT {
        Mat<R> A;
        {
            graph::NoBackprop nb;
            auto mat  = Mat<R>(5, 10, weights<R>::uniform(0.1, 20.0));
            auto mat2 = Mat<R>(5, 10, weights<R>::uniform(-20.0, -0.1));
            A = MatOps<R>::vstack({mat, mat2});
        }
        expect_args_remain_on_gpu(functor, {A});
        EXPECT_TRUE(gradient_same(functor, {A}));
    }
}

TEST_F(MatrixTests, min_rowwise) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return MatOps<R>::min_rowwise(Xs[0]);
    };

    EXPERIMENT_REPEAT {
        Mat<R> A;
        {
            graph::NoBackprop nb;
            auto mat  = Mat<R>(5, 10, weights<R>::uniform(0.1, 20.0));
            auto mat2 = Mat<R>(5, 10, weights<R>::uniform(-20.0, -0.1));
            A = MatOps<R>::vstack({mat, mat2});
        }
        expect_args_remain_on_gpu(functor, {A});
        EXPECT_TRUE(gradient_same(functor, {A}));
    }
}

TEST_F(MatrixTests, max_colwise) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return MatOps<R>::max_colwise(Xs[0]);
    };

    EXPERIMENT_REPEAT {
        Mat<R> A;
        {
            graph::NoBackprop nb;
            auto mat  = Mat<R>(5, 10, weights<R>::uniform(0.1, 20.0));
            auto mat2 = Mat<R>(5, 10, weights<R>::uniform(-20.0, -0.1));
            A = MatOps<R>::vstack({mat, mat2});
        }
        expect_args_remain_on_gpu(functor, {A});
        EXPECT_TRUE(gradient_same(functor, {A}));
    }
}

TEST_F(MatrixTests, min_colwise) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return MatOps<R>::min_colwise(Xs[0]);
    };

    EXPERIMENT_REPEAT {
        Mat<R> A;
        {
            graph::NoBackprop nb;
            auto mat  = Mat<R>(5, 10, weights<R>::uniform(0.1, 20.0));
            auto mat2 = Mat<R>(5, 10, weights<R>::uniform(-20.0, -0.1));
            A = MatOps<R>::vstack({mat, mat2});
        }
        expect_args_remain_on_gpu(functor, {A});
        EXPECT_TRUE(gradient_same(functor, {A}));
    }
}

TEST_F(MatrixTests, sum_colwise) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return MatOps<R>::sum_colwise(Xs[0]);
    };

    EXPERIMENT_REPEAT {
        auto A = Mat<R>(10, 20, weights<R>::uniform(2.0));
        expect_args_remain_on_gpu(functor, {A});
        EXPECT_TRUE(gradient_same(functor, {A}));
    }
}

TEST_F(MatrixTests, max_min_test) {
    auto A = Mat<R>(5, 5);
    for (int i = 0; i < 25; i++) {
        A.w(i) = i - 15;
    }
    ASSERT_NEAR(A.w().min(), -15, 1e-6);
    ASSERT_NEAR(A.w().max(), 9, 1e-6);
}


TEST_F(MatrixTests, equals) {
    auto A = Mat<R>(10, 20, weights<R>::uniform(2.0));
    auto B = Mat<R>(10, 20, weights<R>::uniform(2.0));

    EXPECT_MATRIX_EQ(A, A)  << "A equals A.";
    EXPECT_MATRIX_NEQ(A, B) << "A different from B.";
    EXPECT_MATRIX_CLOSE(A, A, 1e-4)  << "A near A.";
    EXPECT_MATRIX_NOT_CLOSE(A, B, 1e-4) << "A not near B.";

    EXPECT_MAT_ON_GPU(A);
    EXPECT_MAT_ON_GPU(B);

}

TEST_F(MatrixTests, L2_norm_rowwise) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return MatOps<R>::L2_norm_rowwise(Xs[0]);
    };

    EXPERIMENT_REPEAT {
        auto A = Mat<R>(10, 20, weights<R>::uniform(2.0));
        expect_args_remain_on_gpu(functor, {A});
        EXPECT_TRUE(gradient_same(functor, {A}));
    }
}

TEST_F(MatrixTests, L2_norm_colwise) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return MatOps<R>::L2_norm_colwise(Xs[0]);
    };

    EXPERIMENT_REPEAT {
        auto A = Mat<R>(2, 3, weights<R>::ones());
        expect_args_remain_on_gpu(functor, {A});
        EXPECT_TRUE(gradient_same(functor, {A}));
    }
}


TEST_F(MatrixTests, sum) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return Xs[0].sum();
    };

    EXPERIMENT_REPEAT {
        auto A = Mat<R>(10, 20, weights<R>::uniform(2.0));
        expect_args_remain_on_gpu(functor, {A});
        EXPECT_TRUE(gradient_same(functor, {A}));
    }
}

TEST_F(MatrixTests, sigmoid_gpu_vs_cpu) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return Xs[0].sigmoid();
    };

    EXPERIMENT_REPEAT {
        auto A = Mat<R>(10, 20, weights<R>::uniform(2.0));
        EXPECT_TRUE(cpu_vs_gpu(functor, {A}));
    }
}

TEST_F(MatrixTests, identity_init) {
    R init_val = 2.0;
    auto A = Mat<R>(10, 10, weights<R>::eye(init_val));
    EXPECT_MAT_ON_GPU(A);
    for (int i = 0; i < A.dims(0); i++) {
        for (int j = 0; j < A.dims(1); j++) {
            if (i == j) {
                EXPECT_TRUE(A.w(i, j) == init_val);
            } else {
                EXPECT_TRUE(A.w(i, j) == 0.0);
            }
        }
    }
}

TEST_F(MatrixTests, recursive_sum) {
    auto functor = [](vector<Mat<R>>& Xs)-> Mat<R> {
        auto doubled = Xs[0] + Xs[0];
        return doubled.sum();
    };
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(10, 20, weights<R>::uniform(2.0));
        ASSERT_TRUE(gradient_same(functor, {A}, 1e-3, DEFAULT_GRAD_EPS, true));
    }
}

TEST_F(MatrixTests, inplace_sum) {

    EXPERIMENT_REPEAT {
        auto A = Mat<R>(3, 4, weights<R>::uniform(2.0));
        auto B = Mat<R>(3, 4, weights<R>::uniform(2.0));

        auto functor = [&A, &B](vector<Mat<R>>& Xs)-> Mat<R> {
            auto A_temp = A;
            auto B_temp = B;
            A_temp += B_temp;
            return A_temp;
        };
        ASSERT_TRUE(gradient_same(functor, {A, B}, 1e-2, DEFAULT_GRAD_EPS, true));
    }
}

TEST_F(MatrixTests, max_scalar) {
    int input_size = 5;
    int hidden_size = 3;

    EXPERIMENT_REPEAT {
        auto mat = Mat<R>(input_size, hidden_size, weights<R>::uniform(1.5, 20.0));
        auto mat2 = Mat<R>(input_size, hidden_size, weights<R>::uniform(-20.0, 1.3));
        auto combined = MatOps<R>::hstack({mat, mat2});
        R maxxand = 1.4;
        auto functor = [&maxxand](vector<Mat<R>> Xs)-> Mat<R> {
            return MatOps<R>::eltmax(Xs[0], maxxand);
        };
        ASSERT_TRUE(gradient_same(functor, {combined}, 1e-4));
    }
}

TEST_F(MatrixTests, inplace_substract) {
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(3, 4, weights<R>::uniform(1.0, 2.0));
        auto B = Mat<R>(3, 4, weights<R>::uniform(1.0, 2.0));

        auto functor = [&A, &B](vector<Mat<R>>& Xs)-> Mat<R> {
            auto A_temp = A;
            auto B_temp = B;
            A_temp -= B_temp;
            return A_temp;
        };
        ASSERT_TRUE(gradient_same(functor, {A, B}, 1e-5, DEFAULT_GRAD_EPS, true));
    }
}

TEST_F(MatrixTests, inplace_multiply) {
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(3, 4, weights<R>::uniform(2.0));
        auto B = Mat<R>(3, 4, weights<R>::uniform(2.0));

        auto functor = [&A, &B](vector<Mat<R>>& Xs)-> Mat<R> {
            auto A_temp = A;
            auto B_temp = B;
            A_temp *= B_temp;
            return A_temp;
        };
        ASSERT_TRUE(gradient_same(functor, {A, B}, 1e-5, DEFAULT_GRAD_EPS, true));
    }
}

TEST_F(MatrixTests, addition) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return Xs[0] + Xs[1];
    };
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(10, 20, weights<R>::uniform(2.0));
        auto B = Mat<R>(10, 20,  weights<R>::uniform(0.5));
        ASSERT_TRUE(gradient_same(functor, {A, B}, 1e-5, DEFAULT_GRAD_EPS, true));
    }
}

TEST_F(MatrixTests, addition_vector) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return MatOps<R>::add({ Xs[0], Xs[1], Xs[2] });
    };
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(10, 20, weights<R>::uniform(2.0));
        auto B = Mat<R>(10, 20,  weights<R>::uniform(0.5));
        auto C = Mat<R>(10, 20,  weights<R>::uniform(0.5));
        ASSERT_TRUE(gradient_same(functor, {A, B, C}, 1e-5, DEFAULT_GRAD_EPS, true));
    }
}

TEST_F(MatrixTests, broadcast_row_vector) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return MatOps<R>::broadcast_row_vector(Xs[0], 10);
    };
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(1,  20,  weights<R>::uniform(2.0));
        ASSERT_TRUE(gradient_same(functor, {A}, 1e-5, DEFAULT_GRAD_EPS, true));
    }
}

TEST_F(MatrixTests, broadcast_col_vector) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return MatOps<R>::broadcast_col_vector(Xs[0], 10);
    };
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(20,  1,  weights<R>::uniform(2.0));
        ASSERT_TRUE(gradient_same(functor, {A}, 1e-5, DEFAULT_GRAD_EPS, true));
    }
}

TEST(MatrixIOTests, load_test) {
    Mat<R> arange(utils::dir_join({STR(DALI_DATA_DIR),    "tests", "arange12.npy"}));
    Mat<R> arange_fortran(utils::dir_join({STR(DALI_DATA_DIR), "tests", "arange12.fortran.npy"}));
    ASSERT_TRUE(MatOps<R>::equals(arange, arange_fortran));
    for (int i = 0; i < 12; i++) {
        ASSERT_EQ(arange.w(i), i);
    }
}

TEST(MatrixIOTests, save_load_test) {
    // load arange, then save it to a new file
    Mat<R> arange(utils::dir_join({STR(DALI_DATA_DIR),    "tests", "arange12.npy"}));
    arange.npy_save(utils::dir_join({STR(DALI_DATA_DIR),  "tests", "arange12.temp.npy"}));
    Mat<R> reloaded(utils::dir_join({STR(DALI_DATA_DIR),  "tests", "arange12.temp.npy"}));
    ASSERT_TRUE(MatOps<R>::equals(arange, reloaded));
    for (int i = 0; i < 12; i++) {
        ASSERT_EQ(arange.w(i), i);
    }
}

TEST_F(MatrixTests, lazy_allocation) {
    // if memory must be filled with zeros,
    // then allocation is lazy
    Mat<R> zero_mat(4, 5, weights<R>::zeros());

    #ifdef DALI_USE_CUDA
    ASSERT_TRUE((!zero_mat.w().memory_->allocated_cpu) && (!zero_mat.w().memory_->allocated_gpu));
    #else
    ASSERT_TRUE(!zero_mat.w().memory_->allocated_cpu);
    #endif

    // if memory must be filled with gaussian
    // noise, allocation is immediate
    Mat<R> gauss_mat(4, 5, weights<R>::gaussian(0.5));
    #ifdef DALI_USE_CUDA
    ASSERT_TRUE((!gauss_mat.w().memory_->allocated_cpu) && (gauss_mat.w().memory_->allocated_gpu));
    #else
    ASSERT_TRUE(gauss_mat.w().memory_->allocated_cpu);
    #endif

    // the gradients are set to 0, but are also lazily
    // allocated and cleared.
    #ifdef DALI_USE_CUDA
    ASSERT_TRUE((!gauss_mat.dw().memory_->allocated_cpu) && (!gauss_mat.dw().memory_->allocated_gpu));
    ASSERT_TRUE((!zero_mat.dw().memory_->allocated_cpu) && (!zero_mat.dw().memory_->allocated_gpu));
    #else
    ASSERT_TRUE(!gauss_mat.dw().memory_->allocated_cpu);
    ASSERT_TRUE(!zero_mat.dw().memory_->allocated_cpu);
    #endif
}

TEST_F(MatrixTests, view_transpose) {
    // For 1xN or Nx1 matrices, a transpose is simply a
    // different view onto the memory
    Mat<R> row_vector(1, 5, weights<R>::zeros());
    auto transposed_row = row_vector.T();
    for (int i = 0; i < 5; i++) {
        transposed_row.w(i, 0) = i;
    }
    // see the changes reflected in the original matrix:
    for (int i = 0; i < 5; i++) {
        ASSERT_EQ(row_vector.w(0, i), i);
    }
    Mat<R> col_vector(5, 1, weights<R>::zeros());
    auto transposed_col = col_vector.T();
    for (int i = 0; i < 5; i++) {
        transposed_col.w(0, i) = i;
    }
    // see the changes reflected in the original matrix:
    for (int i = 0; i < 5; i++) {
        ASSERT_EQ(col_vector.w(i, 0), i);
    }
}

TEST_F(MatrixTests, slice) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return Xs[0].slice(2, 5);
    };
    EXPERIMENT_REPEAT {
        Mat<R> block(10, 2, weights<R>::uniform(2.0));
        ASSERT_TRUE(gradient_same(functor, {block}));
    }

    Mat<R> block(10, 2, weights<R>::uniform(2.0));
    auto subblock = block.slice(2, 5);

    // ensure the slice is a view!
    ASSERT_EQ(&subblock.w().memory() , &block.w().memory());
    ASSERT_EQ(&subblock.dw().memory() , &block.dw().memory());
}

TEST_F(MatrixTests, reshape) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return Xs[0].slice(2, 10);
    };
    EXPERIMENT_REPEAT {
        Mat<R> block(10, 2, weights<R>::uniform(2.0));
        ASSERT_TRUE(gradient_same(functor, {block}));
    }

    Mat<R> block(10, 2, weights<R>::uniform(2.0));
    auto subblock = block.reshape(20, 1);

    // ensure the slice is a view!
    ASSERT_EQ(&subblock.w().memory() , &block.w().memory());
    ASSERT_EQ(&subblock.dw().memory() , &block.dw().memory());
}

TEST_F(MatrixTests, patch2col) {
    int nbatch = 2;
    int feats = 3;
    int width = 10;
    int height = 4;

    int kwidth = 3;
    int kheight = 2;
    int kstride = 2;

    auto functor = [&](vector<Mat<R>> Xs)-> Mat<R> {
        return MatOps<R>::patch2col(
            Xs[0],
            {nbatch, feats, height, width},
            kheight,
            kwidth,
            kstride
        );
    };
    EXPERIMENT_REPEAT {
        Mat<R> image(nbatch, feats * width * height, weights<R>::uniform(2.0));
        ASSERT_TRUE(gradient_same(functor, {image}));
    }
}

TEST_F(MatrixTests, patch2col_conv2d) {
    int nbatch = 2;
    int feats = 3;
    int width = 10;
    int height = 4;

    int kwidth = 3;
    int kheight = 2;
    int kstride = 2;

    int num_kernels = 2;

    auto functor = [&](vector<Mat<R>> Xs)-> Mat<R> {
        auto out = MatOps<R>::conv2d(
            Xs[0],
            Xs[1],
            {nbatch, feats, height, width},
            kheight,
            kwidth,
            kstride
        );
        return out;
    };
    EXPERIMENT_REPEAT {
        Mat<R> image(nbatch, feats * width * height, weights<R>::uniform(2.0));
        Mat<R> kernels(num_kernels, feats * kwidth * kheight, weights<R>::uniform(2.0));
        ASSERT_TRUE(gradient_same(functor, {image, kernels}));
    }
}

TEST_F(MatrixTests, subtraction) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return Xs[0] - Xs[1];
    };
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(10, 20, weights<R>::uniform(2.0));
        auto B = Mat<R>(10, 20,  weights<R>::uniform(0.5));
        ASSERT_TRUE(gradient_same(functor, {A, B}));
    }
}

TEST_F(MatrixTests, argmax_argmin) {
    auto A = Mat<R>(5, 5, weights<R>::eye());
    // orientation in an identity matrix does not matter
    // for argmax:
    auto indices_max_col = A.argmax(0);
    EXPECT_EQ(indices_max_col, std::vector<int>({0, 1, 2, 3, 4}));
    auto indices_max_row = A.argmax(1);
    EXPECT_EQ(indices_max_row, std::vector<int>({0, 1, 2, 3, 4}));

    // however for an irregular assymetric pattern, then
    // orientation matters:
    auto B   = Mat<R>(6, 5);
    B.w(0,0) = -12.0;
    B.w(1,3) = -32.0;
    B.w(2,4) = -44.0;
    B.w(3,0) = -35.0;
    B.w(4,2) = -32.0;
    B.w(5,3) = -27.0;
    #ifdef DALI_USE_CUDA
    // force computation to happen on device if possible
    B.w().memory().to_gpu();
    #endif

    auto indices_min_col = B.argmin(0);
    EXPECT_EQ(indices_min_col, std::vector<int>({3, 0, 4, 1, 2}));

    auto indices_min_row = B.argmin(1);
    EXPECT_EQ(indices_min_row, std::vector<int>({0, 3, 4, 0, 2, 3}));

    auto Z = Mat<R>(3, 9);
    Z.w(12) = 55;

    Z.w(13) = -12;
    #ifdef DALI_USE_CUDA
    // force computation to happen on device if possible
    Z.w().memory().to_gpu();
    #endif

    // argmin without dimension argument treats
    // matrix as one long strand of memory
    EXPECT_EQ(Z.argmin(), 13);
    EXPECT_EQ(Z.argmax(), 12);
}

TEST_F(MatrixTests, argsort) {
    auto mats = vector<Mat<R>>({
        MatOps<R>::fill(Mat<R>(1,1), 3),
        MatOps<R>::fill(Mat<R>(1,1), 9),
        MatOps<R>::fill(Mat<R>(1,1), 1),
    });
    auto sorted = utils::argsort(mats);
    ASSERT_EQ(sorted, std::vector<size_t>({2, 0, 1}));
}


TEST_F(MatrixTests, mat_argsort) {
    // shape of matrix has no influence on
    // argsort
    auto A = Mat<R>(2, 2);
    A.w(0) = -5;
    A.w(1) =  12.0;
    A.w(2) = -33.0;
    A.w(3) =  66.0;

    #ifdef DALI_USE_CUDA
    A.w().memory().to_gpu();
    #endif

    auto sorted = A.argsort();
    ASSERT_EQ(sorted, std::vector<int>({2, 0, 1, 3}));
}

TEST_F(MatrixTests, mean) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return Xs[0].mean();
    };
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(10, 20, weights<R>::uniform(2.0));
        ASSERT_TRUE(gradient_same(functor, {A}));
    }
}

TEST_F(MatrixTests, max) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return Xs[0].max();
    };
    EXPERIMENT_REPEAT {
        Mat<R> A;
        {
            graph::NoBackprop nb;
            auto mat  = Mat<R>(5, 10, weights<R>::uniform(0.1, 20.0));
            auto mat2 = Mat<R>(5, 10, weights<R>::uniform(-20.0, -0.1));
            A = MatOps<R>::hstack({mat, mat2});
        }
        ASSERT_TRUE(gradient_same(functor, {A}));
    }
}

TEST_F(MatrixTests, min) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return Xs[0].min();
    };
    EXPERIMENT_REPEAT {
        Mat<R> A;
        {
            graph::NoBackprop nb;
            auto mat  = Mat<R>(5, 10, weights<R>::uniform(0.1, 20.0));
            auto mat2 = Mat<R>(5, 10, weights<R>::uniform(-20.0, -0.1));
            A = MatOps<R>::hstack({mat, mat2});
        }
        ASSERT_TRUE(gradient_same(functor, {A}));
    }
}

TEST_F(MatrixTests, square) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return Xs[0].square();
    };
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(10, 20, weights<R>::uniform(0.5, 5.0));
        ASSERT_TRUE(gradient_same(functor, {A}, 1e-3, 1e-5));
    }
}

TEST_F(MatrixTests, sqrt) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return Xs[0].sqrt();
    };
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(10, 20, weights<R>::uniform(0.5, 5.0));
        ASSERT_TRUE(gradient_same(functor, {A}, 1e-3, 1e-5));
    }
}

TEST_F(MatrixTests, elt_inv) {
    sane_crashes::activate();
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return MatOps<R>::elt_inv(Xs[0]);
    };
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(10, 20, weights<R>::uniform(0.5, 5.0));
        ASSERT_TRUE(gradient_same(functor, {A}, 1e-3, 1e-5));
    }
}

TEST_F(MatrixTests, addition_broadcast_rowwise) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return Xs[0] + Xs[1];
    };
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(20, 10, weights<R>::uniform(2.0));
        auto B = Mat<R>(1, 10, weights<R>::uniform(0.5));
        ASSERT_TRUE(gradient_same(functor, {A, B}));
    }
}



TEST_F(MatrixTests, addition_broadcast_colwise) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return MatOps<R>::add_broadcast_colwise(Xs[0], Xs[1]);
    };
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(10, 20, weights<R>::uniform(2.0));
        auto B = Mat<R>(10, 1, weights<R>::uniform(0.5));
        ASSERT_TRUE(gradient_same(functor, {A, B}));
    }
}

TEST_F(MatrixTests, substraction_broadcast) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return Xs[0] - Xs[1];
    };
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(10, 20, weights<R>::uniform(2.0));
        auto B = Mat<R>(10, 1, weights<R>::uniform(0.5));
        ASSERT_TRUE(gradient_same(functor, {A, B}));
    }
}

TEST_F(MatrixTests, substraction_reversed_broadcast) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return Xs[1] - Xs[0];
    };
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(10, 20, weights<R>::uniform(2.0));
        auto B = Mat<R>(10, 1, weights<R>::uniform(0.5));
        ASSERT_TRUE(gradient_same(functor, {A, B}));
    }
}


TEST_F(MatrixTests, sigmoid) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return Xs[0].sigmoid();
    };
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(10, 20, weights<R>::uniform(20.0));
        ASSERT_TRUE(gradient_same(functor, {A}, 1e-4, 1e-3));
    }
}

TEST_F(MatrixTests, steep_sigmoid) {
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(10, 20, weights<R>::uniform(20.0));

        R aggresiveness = utils::randdouble(1.0, 2.0);
        auto functor = [aggresiveness](vector<Mat<R>> Xs)-> Mat<R> {
            return MatOps<R>::steep_sigmoid(Xs[0], aggresiveness);
        };
        ASSERT_TRUE(gradient_same(functor, {A}, 1e-4, 1e-3));
    }
}

TEST_F(MatrixTests, relu) {
    int input_size = 5;
    int hidden_size = 3;

    EXPERIMENT_REPEAT {
        Mat<R> A;
        {
            graph::NoBackprop nb;
            auto mat  = Mat<R>(input_size, hidden_size, weights<R>::uniform(0.1, 20.0));
            auto mat2 = Mat<R>(input_size, hidden_size, weights<R>::uniform(-20.0, -0.1));
            A = MatOps<R>::hstack({mat, mat2});
        }

        auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
            return Xs[0].relu();
        };
        ASSERT_TRUE(gradient_same(functor, {A}, 1e-4));
    }
}

TEST_F(MatrixTests, tanh) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return Xs[0].tanh();
    };
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(10, 20, weights<R>::uniform(20.0));
        ASSERT_TRUE(gradient_same(functor, {A}, 1e-4, 1e-3, true));
    }
}

TEST_F(MatrixTests, binary_cross_entropy) {
    // We observe the KL divergence to 0 or 1 for each unit
    // in our input matrix with respect to the target.
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(10, 20, weights<R>::uniform(0.1, 0.9));
        R target = utils::randdouble(0.01, 0.99);
        auto functor = [target](vector<Mat<R>> Xs)-> Mat<R> {
            return MatOps<R>::binary_cross_entropy(Xs[0], target);
        };
        ASSERT_TRUE(gradient_same(functor, {A}, 1e-2));
    }
}

TEST_F(MatrixTests, binary_cross_entropy_matrix_target) {
    // We observe the KL divergence to 0 or 1 for each unit
    // in our input matrix with respect to the target.
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(10, 20, weights<R>::uniform(0.1, 0.9));
        auto target = Mat<R>(10, 20, weights<R>::uniform(0.1, 0.9));
        target.constant = true;
        auto functor = [target](vector<Mat<R>> Xs)-> Mat<R> {
            return MatOps<R>::binary_cross_entropy(Xs[0], target);
        };
        ASSERT_TRUE(gradient_same(functor, {A}, 1e-2));
    }
}

TEST_F(MatrixTests, sigmoid_binary_cross_entropy) {
    // we can now extend the range of our random numbers to be beyond
    // 0 and 1 since sigmoid will clamp them to 0 or 1.
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(10, 20, weights<R>::uniform(5.0));
        R target = utils::randdouble(0.1, 0.9);
        auto functor = [target](vector<Mat<R>> Xs)-> Mat<R> {
            return MatOps<R>::sigmoid_binary_cross_entropy(Xs[0], target);
        };
        ASSERT_TRUE(gradient_same(functor, {A}, 1e-2, 1e-4));
    }
}

TEST_F(MatrixTests, sigmoid_binary_cross_entropy_matrix_target) {
    // we can now extend the range of our random numbers to be beyond
    // 0 and 1 since sigmoid will clamp them to 0 or 1.
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(10, 20, weights<R>::uniform(5.0));
        auto target = Mat<R>(10, 20, weights<R>::uniform(0.1, 0.9));
        target.constant = true;
        auto functor = [target](vector<Mat<R>> Xs)-> Mat<R> {
            return MatOps<R>::sigmoid_binary_cross_entropy(Xs[0], target);
        };
        ASSERT_TRUE(gradient_same(functor, {A}, 1e-2, 1e-4));
    }
}

TEST_F(MatrixTests, margin_loss_colwise) {
    utils::random::set_seed(100);
    // we can now extend the range of our random numbers to be beyond
    // 0 and 1 since sigmoid will clamp them to 0 or 1.
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(10, 20, weights<R>::uniform(5.0));
        R margin = utils::randdouble(0.01, 0.1);
        uint target = utils::randinteger<uint>(0, A.dims(0) - 1);
        auto functor = [target, margin](vector<Mat<R>> Xs)-> Mat<R> {
            return MatOps<R>::margin_loss_colwise(Xs[0], target, margin);
        };
        ASSERT_TRUE(gradient_same(functor, {A}, 1e-3, 1e-4));
    }
    utils::random::reseed();
}


TEST_F(MatrixTests, margin_loss_rowwise) {
    utils::random::set_seed(100);
    // we can now extend the range of our random numbers to be beyond
    // 0 and 1 since sigmoid will clamp them to 0 or 1.
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(10, 20, weights<R>::uniform(5.0));
        R margin = utils::randdouble(0.01, 0.1);
        uint target = utils::randinteger<uint>(0, A.dims(1) - 1);
        auto functor = [target, margin](vector<Mat<R>> Xs)-> Mat<R> {
            return MatOps<R>::margin_loss_rowwise(Xs[0], target, margin);
        };
        ASSERT_TRUE(gradient_same(functor, {A}, 1e-3, 1e-4));
    }
    utils::random::reseed();
}



TEST_F(MatrixTests, norm) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return Xs[0].L2_norm();
    };
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(10, 20, weights<R>::uniform(20.0));
        ASSERT_TRUE(gradient_same(functor, {A}, 1e-4, DEFAULT_GRAD_EPS, true));
    }
}

TEST_F(MatrixTests, exp) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return Xs[0].exp();
    };
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(10, 20, weights<R>::uniform(3.0));
        ASSERT_TRUE(gradient_same(functor, {A}, 1e-3, 1e-3, true));
    }
}

TEST_F(MatrixTests, softplus) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return Xs[0].softplus();
    };
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(10, 20, weights<R>::uniform(3.0));
        ASSERT_TRUE(gradient_same(functor, {A}, 1e-3, 1e-3, true));
    }
}

TEST_F(MatrixTests, log) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return Xs[0].log();
    };
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(10, 20, weights<R>::uniform(0.1, 20.0));
        ASSERT_TRUE(gradient_same(functor, {A}, 1e-3, 1e-3, true));
    }
}

TEST_F(MatrixTests, dot) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return Xs[1].dot(Xs[0]);
    };
    int num_examples = 5;
    int hidden_size = 10;
    int input_size = 3;
    EXPERIMENT_REPEAT {
        auto X = Mat<R>(input_size, num_examples, weights<R>::uniform(20.0));
        auto W = Mat<R>(hidden_size, input_size, weights<R>::uniform(2.0));
        ASSERT_TRUE(gradient_same(functor, {X, W}, 1e-4));
    }
}

TEST_F(MatrixTests, matrix_dot_plus_bias) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        auto res = Xs[1].dot(Xs[0]) + Xs[2];
        return res;
    };
    int num_examples = 20;
    int hidden_size = 10;
    int input_size = 5;
    EXPERIMENT_REPEAT {
        auto X = Mat<R>(num_examples, input_size,   weights<R>::uniform(20.0));
        auto W = Mat<R>(input_size,   hidden_size,  weights<R>::uniform(2.0));
        auto bias = Mat<R>(1, hidden_size, weights<R>::uniform(2.0));
        ASSERT_TRUE(gradient_same(functor, {W, X, bias}, 1e-4));
    }
}

TEST_F(MatrixTests, matrix_divide) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return Xs[0] / Xs[1];
    };
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(10, 20, weights<R>::uniform(10.0, 20.0));
        auto B = Mat<R>(10, 20, weights<R>::uniform(5.0, 15.0));
        ASSERT_TRUE(gradient_same(functor, {A, B}, 1e-3));
    }
}

TEST_F(MatrixTests, matrix_divide_broadcast) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return Xs[0] / Xs[1];
    };
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(10, 20, weights<R>::uniform(0.1, 20.0));
        auto B = Mat<R>(10, 1,  weights<R>::uniform(0.5, 4.0));
        ASSERT_TRUE(gradient_same(functor, {A, B}, 5e-3, 1e-3));
    }
}

TEST_F(MatrixTests, matrix_eltmul_broadcast_rowwise_default) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return Xs[0] * Xs[1];
    };
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(20, 10, weights<R>::uniform(0.1, 20.0));
        auto B = Mat<R>(1,  10,  weights<R>::uniform(0.5, 4.0));
        ASSERT_TRUE(gradient_same(functor, {A, B}, 5e-3, 1e-3));
    }
}

TEST_F(MatrixTests, matrix_eltmul_broadcast_colwise) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return MatOps<R>::eltmul_broadcast_colwise(Xs[0], Xs[1]);
    };
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(10, 20, weights<R>::uniform(0.1, 20.0));
        auto B = Mat<R>(10, 1,  weights<R>::uniform(0.5, 4.0));
        ASSERT_TRUE(gradient_same(functor, {A, B}, 5e-3, 1e-3));
    }
}

TEST_F(MatrixTests, matrix_divide_reversed_broadcast) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return Xs[1] / Xs[0];
    };
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(10, 20, weights<R>::uniform(0.5, 5.0));
        auto B = Mat<R>(10, 1,  weights<R>::uniform(0.1, 20.0));
        ASSERT_TRUE(gradient_same(functor, {A, B}, 5e-3, 1e-3));
    }
}

TEST_F(MatrixTests, matrix_eltmul_rowwise) {
    // Operation of the form f(A,B) => A * B.T
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return MatOps<R>::eltmul_rowwise(Xs[0], Xs[1]);
    };
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(4, 5, weights<R>::uniform(10.0));
        auto B = Mat<R>(5, 4, weights<R>::uniform(10.0));
        ASSERT_TRUE(gradient_same(functor, {A, B}, 1e-3));
    }
}

TEST_F(MatrixTests, matrix_eltmul_broadcast_rowwise) {
    // Operation of the form f(A,B) => A * B.T
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return MatOps<R>::eltmul_broadcast_rowwise(Xs[0], Xs[1]);
    };
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(4, 5, weights<R>::uniform(10.0));
        auto B = Mat<R>(1, 5, weights<R>::uniform(10.0));
        ASSERT_TRUE(gradient_same(functor, {A, B}, 1e-3));
    }
}

TEST_F(MatrixTests, matrix_divide_scalar) {
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(10, 20, weights<R>::uniform(-20.0, 20.0));
        auto scalar = (R) utils::randdouble(0.1, 20.0);
        auto functor = [&scalar](vector<Mat<R>> Xs)-> Mat<R> {
            return Xs[0] / scalar;
        };
        ASSERT_TRUE(gradient_same(functor, {A}, 1e-3, DEFAULT_GRAD_EPS, true));
    }
}


TEST_F(MatrixTests, divide_inplace_matrix) {
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(3, 4, weights<R>::uniform(0.1, 20.0));
        auto B = Mat<R>(3, 4, weights<R>::uniform(1.0, 2.0));

        auto functor = [&A, &B](vector<Mat<R>>& Xs)-> Mat<R> {
            auto A_temp = A;
            auto B_temp = B;
            A_temp /= B_temp;
            return A_temp;
        };
        ASSERT_TRUE(gradient_same(functor, {A, B}, 1e-3, DEFAULT_GRAD_EPS, true));
    }
}


TEST_F(MatrixTests, divide_inplace_scalar) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        auto temp = Xs[0];
        temp /= (R)20.0;
        return (temp - 2.0) ^ 2;
    };
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(3, 4, weights<R>::uniform(0.001, 20.0));
        ASSERT_TRUE(gradient_same(functor, {A}, 1e-4, DEFAULT_GRAD_EPS, false));
    }
}

typedef MemorySafeTest MatOpsTests;

TEST_F(MatOpsTests, matrix_mul_with_bias) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return MatOps<R>::mul_with_bias(Xs[1], Xs[0], Xs[2]);
    };
    int num_examples = 20;
    int hidden_size = 10;
    int input_size = 5;
    EXPERIMENT_REPEAT {
        auto X = Mat<R>(num_examples, input_size,  weights<R>::uniform(20.0));
        auto W = Mat<R>(input_size, hidden_size,  weights<R>::uniform(2.0));
        auto bias = Mat<R>(1, hidden_size, weights<R>::uniform(2.0));
        ASSERT_TRUE(gradient_same(functor, {X, W, bias}, 1e-4));
    }
}

TEST_F(MatOpsTests, matrix_mul_add_mul_with_bias) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return MatOps<R>::mul_add_mul_with_bias({Xs[0], Xs[2]}, {Xs[1], Xs[3]}, Xs[4]);
    };
    int num_examples = 20;
    int hidden_size = 10;
    int input_size = 5;
    int other_input_size = 7;
    EXPERIMENT_REPEAT {
        auto X       = Mat<R>(num_examples, input_size,       weights<R>::uniform(20.0));
        auto W       = Mat<R>(input_size,   hidden_size,      weights<R>::uniform(2.0));

        auto X_other = Mat<R>(num_examples, other_input_size, weights<R>::uniform(20.0));
        auto W_other = Mat<R>(other_input_size, hidden_size,  weights<R>::uniform(2.0));

        auto bias    = Mat<R>(hidden_size, 1,                 weights<R>::uniform(2.0));
        ASSERT_TRUE(gradient_same(functor, {W, X, W_other, X_other, bias}, 0.0003));
    }
}

TEST_F(MatOpsTests, matrix_mul_add_mul_with_bias_fancy_broadcast) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return MatOps<R>::mul_add_mul_with_bias({Xs[0], Xs[2], Xs[4]}, {Xs[1], Xs[3], Xs[5]}, Xs[6]);
    };
    int num_examples = 20;
    int hidden_size = 10;
    int input_size = 5;
    int other_input_size = 7;
    EXPERIMENT_REPEAT {
        auto X       = Mat<R>(num_examples, input_size,       weights<R>::uniform(20.0));
        auto W       = Mat<R>(input_size,   hidden_size,      weights<R>::uniform(2.0));

        auto Xfancy   = Mat<R>(1,            input_size,       weights<R>::uniform(20.0));
        auto Wfancy   = Mat<R>(input_size,   hidden_size,      weights<R>::uniform(2.0));


        auto X_other = Mat<R>(num_examples, other_input_size, weights<R>::uniform(20.0));
        auto W_other = Mat<R>(other_input_size, hidden_size,  weights<R>::uniform(2.0));

        auto bias    = Mat<R>(1,  hidden_size,                weights<R>::uniform(2.0));
        ASSERT_TRUE(gradient_same(functor, {W, X, Wfancy, Xfancy, W_other, X_other, bias}, 0.0003));
    }
}

TEST_F(MatOpsTests, matrix_mul_add_mul_with_bias_colwise) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return MatOps<R>::mul_add_mul_with_bias_colwise({Xs[0], Xs[2], Xs[4]}, {Xs[1], Xs[3], Xs[5]}, Xs[6]);
    };
    int num_examples = 20;
    int hidden_size = 10;
    int input_size = 5;
    int other_input_size = 7;
    EXPERIMENT_REPEAT {
        auto W       = Mat<R>(hidden_size,  input_size,         weights<R>::uniform(2.0));
        auto X       = Mat<R>(input_size,   num_examples,       weights<R>::uniform(20.0));

        auto Wfancy   = Mat<R>(hidden_size, input_size,         weights<R>::uniform(2.0));
        auto Xfancy   = Mat<R>(input_size,  1,                  weights<R>::uniform(20.0));


        auto X_other = Mat<R>(other_input_size, num_examples,      weights<R>::uniform(20.0));
        auto W_other = Mat<R>(hidden_size,      other_input_size,  weights<R>::uniform(2.0));

        auto bias    = Mat<R>(1,      hidden_size,                 weights<R>::uniform(2.0));
        ASSERT_TRUE(gradient_same(functor, {W, X, Wfancy, Xfancy, W_other, X_other, bias}, 0.0003));
    }
}


TEST_F(MatrixTests, log_exp) {
    EXPERIMENT_REPEAT {
        graph::NoBackprop nb;
        auto mat = Mat<R>(10, 10, weights<R>::uniform(0.1, 20.0));
        auto log_mat = mat.log();
        auto exp_log_mat = log_mat.exp();
        #ifdef DALI_USE_CUDA
        ASSERT_MATRIX_CLOSE(mat, exp_log_mat, 1e-3);
        #else
        ASSERT_MATRIX_CLOSE(mat, exp_log_mat, 1e-6);
        #endif
    }
}

TEST_F(MatrixTests, hstack) {
    {
        graph::NoBackprop nb;

        Mat<R> a(2, 3);
        Mat<R> b(2, 4);

        // A:
        // 0 1 2
        // 7 8 9
        // B:
        // 3  4  5  6
        // 10 11 12 13

        for (int i = 0; i < 3; i++)
            a.w(i) = i;
        for (int i = 0; i < 4; i++)
            b.w(i) = i + 3;
        for (int i = 3; i < 6; i++)
            a.w(i) = i + 4;
        for (int i = 4; i < 8; i++)
            b.w(i) = i + 6;

        auto c = MatOps<R>::hstack({a, b});
        for (int i = 0; i < 14;i++) {
            ASSERT_EQ(c.w(i), i);
        }
    }

    EXPERIMENT_REPEAT {
        auto mat = Mat<R>(2, 3, weights<R>::uniform(20.0));
        auto mat2 = Mat<R>(2, 4, weights<R>::uniform(20.0));

        auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
            return MatOps<R>::hstack(Xs);
        };
        ASSERT_TRUE(gradient_same(functor, {mat, mat2}, 1e-4));
    }
}


TEST_F(MatrixTests, vstack) {
    {
        graph::NoBackprop nb;

        Mat<R> a(2, 3);
        Mat<R> b(4, 3);
        // A:
        // 0 1 2
        // 1 2 3
        // B:
        // 2 3 4
        // 3 4 5
        // 4 5 6
        // 5 6 7

        for (int row=0; row < 2; ++row) {
            for (int col = 0; col < 3; ++col) {
                a.w(row,col) = (R)(row + col);
            }
        }

        for (int row=0; row < 4; ++row) {
            for (int col = 0; col < 3; ++col) {
                b.w(row,col) = (R)(row + col + 2);
            }
        }

        auto c = MatOps<R>::vstack(a,b);

        for (int row = 0; row < 6; ++row) {
            for (int col = 0; col < 3; ++col) {
                SCOPED_TRACE("Index (" + std::to_string(row) + "," + std::to_string(col) + ")");
                ASSERT_NEAR(c.w(row, col), (R)(row + col), 1e-9);
            }
        }
    }

    EXPERIMENT_REPEAT {
        auto mat = Mat<R>(2, 3, weights<R>::uniform(20.0));
        auto mat2 = Mat<R>(4, 3, weights<R>::uniform(20.0));

        auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
            return MatOps<R>::vstack(Xs);
        };
        ASSERT_TRUE(gradient_same(functor, {mat, mat2}, 1e-4));
    }
}


TEST_F(MatrixTests, abs) {
    int input_size = 5;
    int hidden_size = 3;

    EXPERIMENT_REPEAT {
        auto mat = Mat<R>(input_size, hidden_size, weights<R>::uniform(0.1, 20.0));
        auto mat2 = Mat<R>(input_size, hidden_size, weights<R>::uniform(-20.0, -0.1));

        auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
            return MatOps<R>::hstack(Xs).abs();
        };
        ASSERT_TRUE(gradient_same(functor, {mat, mat2}, 1e-4));
    }
}


TEST_F(MatOpsTests, dropout) {
    int seed = 1234;
    auto functor = [&seed](vector<Mat<R>> Xs)-> Mat<R> {
        auto C = Xs[0] * Xs[1];
        utils::random::set_seed(seed);
        auto D = MatOps<R>::dropout(C, 0.5);
        utils::random::reseed();
        auto Z = D + Xs[2];
        return Z;
    };
    int num_examples = 3;
    int hidden_size = 4;
    int input_size = 5;
    EXPERIMENT_REPEAT {
        seed = utils::randint(0, 2000);
        auto A = Mat<R>(hidden_size, input_size, weights<R>::uniform(2.0));
        auto B = Mat<R>(hidden_size, input_size, weights<R>::uniform(20.0));
        auto C = Mat<R>(1,           input_size, weights<R>::uniform(20.0));
        ASSERT_TRUE(gradient_same(functor, {A, B, C}, 0.0003));
    }
}

TEST_F(MatOpsTests, dropout_normalized) {
    int seed = 1234;
    auto functor = [&seed](vector<Mat<R>> Xs)-> Mat<R> {
        auto C = Xs[0] * Xs[1];
        utils::random::set_seed(seed);
        auto D = MatOps<R>::dropout_normalized(C, 0.5);
        utils::random::reseed();
        auto Z = D + Xs[2];
        return Z;
    };
    int num_examples = 3;
    int hidden_size = 4;
    int input_size = 5;
    EXPERIMENT_REPEAT {
        seed = utils::randint(0, 2000);
        auto A = Mat<R>(hidden_size, input_size, weights<R>::uniform(2.0));
        auto B = Mat<R>(hidden_size, input_size, weights<R>::uniform(20.0));
        auto C = Mat<R>(1,           input_size, weights<R>::uniform(20.0));
        ASSERT_TRUE(gradient_same(functor, {A, B, C}, 0.0003));
    }
}

TEST_F(MatOpsTests, fast_dropout) {
    int seed = 1234;
    auto functor = [&seed](vector<Mat<R>> Xs)-> Mat<R> {
        auto C = Xs[0] * Xs[1];
        utils::random::set_seed(seed);
        auto D = MatOps<R>::fast_dropout(C);
        utils::random::reseed();
        auto Z = D + Xs[2];
        return Z;
    };
    int num_examples = 3;
    int hidden_size = 4;
    int input_size = 5;
    EXPERIMENT_REPEAT {
        seed = utils::randint(0, 2000);
        auto A = Mat<R>(hidden_size, input_size, weights<R>::uniform(2.0));
        auto B = Mat<R>(hidden_size, input_size, weights<R>::uniform(20.0));
        auto C = Mat<R>(1,           input_size, weights<R>::uniform(20.0));
        ASSERT_TRUE(gradient_same(functor, {A, B, C}, 0.0003));
    }
}

TEST_F(MatOpsTests, softmax_colwise) {
    int row_size = 5;
    int col_size = 10;
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(row_size, col_size, weights<R>::uniform(-3.0, 3.0));
        int row = utils::randint(0, row_size - 1);
        auto functor = [row](vector<Mat<R>> Xs)-> Mat<R>{
            auto soft = MatOps<R>::softmax_colwise(Xs[0]);
            //soft.print();
            auto g = soft[row];
            //g.print();
            return g;
        };
        ASSERT_TRUE(gradient_same(functor, {A}, 1e-3, 1e-3));
    }
}

TEST_F(MatOpsTests, softmax_rowwise) {
    int row_size = 5;
    int col_size = 10;
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(row_size, col_size, weights<R>::uniform(-3.0, 3.0));
        int col = utils::randint(0, col_size - 1);
        auto functor = [col](vector<Mat<R>> Xs)-> Mat<R>{
            auto soft = MatOps<R>::softmax_rowwise(Xs[0]);
            auto g = soft.T()[col];
            return g;
        };
        ASSERT_TRUE(gradient_same(functor, {A}, 1e-3));
    }
}

TEST_F(MatOpsTests, resize_decrease_rows) {
    int row_size = 3, col_size = 4;
    // decrease number of rows by 1
    auto A = Mat<R>(row_size, col_size);
    for (int i = 0; i < 12; i++) {
        A.w(i) = i;
    }

    auto new_shape = mshadow::Shape2(row_size - 1, col_size);
    A.w().resize(new_shape);
    for (int i = 0; i < (row_size - 1) * col_size ; i++) {
        ASSERT_EQ(A.w(i), i);
    }
    ASSERT_EQ(A.w().shape, new_shape);
}

TEST_F(MatOpsTests, resize_increase_rows) {
    int row_size = 3, col_size = 4;
    // increase number of rows by 1
    auto A = Mat<R>(row_size, col_size);
    for (int i = 0; i < row_size * col_size; i++) {
        A.w(i) = i;
    }
    auto new_shape = mshadow::Shape2(row_size + 1, col_size);
    A.w().resize(new_shape, 3.5);
    for (int i = 0; i < row_size * col_size; i++) {
        ASSERT_EQ(A.w(i), i);
    }
    for (int i = row_size * col_size; i < (row_size + 1) * col_size; i++) {
        ASSERT_EQ(A.w(i), 3.5);
    }
    ASSERT_EQ(A.w().shape, new_shape);
}

TEST_F(MatOpsTests, resize_decrease_cols) {
    int row_size = 3, col_size = 4;
    // decrease number of columns by 1
    auto A = Mat<R>(row_size, col_size);
    for (int i = 0; i < row_size * col_size; i++) {
        A.w(i) = i;
    }
    auto new_shape = mshadow::Shape2(row_size, col_size - 1);
    A.w().resize(new_shape);
    for (int i = 0; i < row_size; i++) {
        for (int j = 0; j < col_size - 1; j++) {
            ASSERT_EQ(A.w(i,j), i * col_size + j);
        }
    }
    ASSERT_EQ(A.w().shape, new_shape);
}

TEST_F(MatOpsTests, resize_increase_cols) {
    int row_size = 3, col_size = 4;
    // increase number of columns by 1
    auto A = Mat<R>(row_size, col_size);
    for (int i = 0; i < row_size * col_size; i++) {
        A.w(i) = i;
    }
    auto new_shape = mshadow::Shape2(row_size, col_size + 1);
    A.w().resize(new_shape, 4.2);
    for (int i = 0; i < row_size; i++) {
        for (int j = 0; j < col_size; j++) {
            ASSERT_EQ(A.w(i,j), i * col_size + j);
        }
    }
    for (int i = 0; i < row_size; i++) {
        for (int j = col_size; j < col_size + 1; j++) {
            ASSERT_EQ(A.w(i,j), 4.2);
        }
    }
    ASSERT_EQ(A.w().shape, new_shape);
}

TEST_F(MatOpsTests, resize_increase_rows_and_cols) {
    int row_size = 3, col_size = 4;
    // increase number of rows and columns by 1
    auto A = Mat<R>(row_size, col_size);
    for (int i = 0; i < row_size * col_size; i++) {
        A.w(i) = i;
    }
    auto new_shape = mshadow::Shape2(row_size + 1, col_size + 1);
    A.w().resize(new_shape, 4.2);
    for (int i = 0; i < row_size; i++) {
        for (int j = 0; j < col_size; j++) {
            ASSERT_EQ(A.w(i,j), i * col_size + j);
        }
    }
    for (int i = 0; i < row_size; i++) {
        for (int j = col_size; j < col_size + 1; j++) {
            ASSERT_EQ(A.w(i,j), 4.2);
        }
    }
    for (int i = row_size; i < row_size + 1; i++) {
        for (int j = 0; j < col_size + 1; j++) {
            ASSERT_EQ(A.w(i,j), 4.2);
        }
    }
    ASSERT_EQ(A.w().shape, new_shape);
}

TEST_F(MatOpsTests, resize_decrease_rows_and_cols) {
    int row_size = 3, col_size = 4;
    // decrease number of rows and columns by 1
    auto A = Mat<R>(row_size, col_size);
    for (int i = 0; i < row_size * col_size; i++) {
        A.w(i) = i;
    }
    auto new_shape = mshadow::Shape2(row_size - 1, col_size - 1);
    A.w().resize(new_shape, 4.2);
    for (int i = 0; i < row_size - 1; i++) {
        for (int j = 0; j < col_size - 1; j++) {
            ASSERT_EQ(A.w(i,j), i * col_size + j);
        }
    }
    ASSERT_EQ(A.w().shape, new_shape);
}

TEST_F(MatOpsTests, resize_1D_decrease_rows) {
    int row_size = 3;
    // decrease number of rows by 1
    TensorInternal<R,1> A(mshadow::Shape1(row_size));

    for (int i = 0; i < row_size; i++) {
        A(i) = i;
    }

    auto new_shape = mshadow::Shape1(row_size - 1);
    A.resize(new_shape);
    for (int i = 0; i < (row_size - 1); i++) {
        ASSERT_EQ(A(i), i);
    }
    ASSERT_EQ(A.shape, new_shape);
}

TEST_F(MatOpsTests, resize_1D_increase_rows) {
    int row_size = 3;
    // increase number of rows by 1
    TensorInternal<R,1> A(mshadow::Shape1(row_size));

    for (int i = 0; i < row_size; i++) {
        A(i) = i;
    }

    auto new_shape = mshadow::Shape1(row_size + 1);
    A.resize(new_shape, 666.0);
    for (int i = 0; i < (row_size); i++) {
        ASSERT_EQ(A(i), i);
    }
    ASSERT_EQ(A(row_size), 666.0);
    ASSERT_EQ(A.shape, new_shape);
}

TEST_F(MatOpsTests, circular_convolution) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return MatOps<R>::circular_convolution(Xs[0], Xs[1]);
    };
    EXPERIMENT_REPEAT {
        auto matrix = Mat<R>(4, 5, weights<R>::uniform(-20.0, 20.0));
        auto shift  = Mat<R>(4, 5, weights<R>::uniform(-20.0, 20.0));
        ASSERT_TRUE(gradient_same(functor, {matrix, shift}, 1e-4));
    }
}


TEST_F(MatOpsTests, DISABLED_matrix_conv1d_grad) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return MatOps<R>::conv1d(Xs[0], std::initializer_list<Mat<R>>({Xs[1], Xs[2]})).tanh();
    };
    EXPERIMENT_REPEAT {
        auto kernel1 = Mat<R>(5, 5, weights<R>::uniform(-20.0, 20.0));
        auto kernel2 = Mat<R>(5, 5, weights<R>::uniform(-20.0, 20.0));
        auto image = Mat<R>(5, 20, weights<R>::uniform(-20.0, 20.0));
        ASSERT_TRUE(gradient_same(functor, {image, kernel1, kernel2}, 1e-2));
    }
}

TEST_F(MatOpsTests, softmax_temperature) {
    graph::NoBackprop nb;

    auto mat = Mat<R>(10, 1);
    for (int i = 0; i < 10; i++) mat.w(i) = i;

    auto base_prob = MatOps<R>::softmax_colwise(mat, 1.0);

    auto flat = MatOps<R>::softmax_colwise(
        MatOps<R>::fill(
            Mat<R>::empty_like(mat),
            1.0
        )
    );

    auto kl = MatOps<R>::cross_entropy(
        base_prob,
        flat
    ).sum();

    // gets flatter with higher temperature
    for (int i = 2; i < 11; i++) {
        R temperature = 1.0 * i;
        auto new_prob = MatOps<R>::softmax_colwise(
            mat,
            temperature
        );
        auto new_kl = MatOps<R>::cross_entropy(
            new_prob,
            flat
        ).sum();
        ASSERT_TRUE(new_kl.w(0) < kl.w(0));
        kl = new_kl;
    }
}

TEST_F(MatOpsTests, cross_entropy_grad) {
    EXPERIMENT_REPEAT {
        const int hidden_size = 10;
        double temperature = 1.0; // utils::randdouble(0.1, 100);
        int target = utils::randint(0, hidden_size - 1);
        auto layer = Mat<R>(hidden_size, 5, weights<R>::uniform(-2.0, 2.0));
        auto input = Mat<R>(5,  3, weights<R>::uniform(-2.0, 2.0));

        auto functor = [target, temperature](vector<Mat<R>> Xs)-> Mat<R> {
            auto soft = MatOps<R>::softmax_colwise(
                    Xs[1].dot(Xs[0]),
                    temperature
                );
            return MatOps<R>::cross_entropy_colwise(
                soft,
                target);
        };

        ASSERT_TRUE(gradient_same(functor, {input, layer}, 1e-2));
    }
}

TEST_F(MatOpsTests, softmax_cross_entropy_colwise_grad) {
    EXPERIMENT_REPEAT {
        auto input = Mat<R>(3,  2, weights<R>::uniform(-2.0, 2.0));

        vector<uint> targets;
        for (int i = 0; i < input.dims(1); ++i)
            targets.push_back(utils::randint(0, input.dims(0) - 1));
        Indexing::Index indexed_targets(&targets);


        auto functor = [indexed_targets](vector<Mat<R>> Xs)-> Mat<R> {
            return MatOps<R>::softmax_cross_entropy_colwise(
                Xs[0],
                indexed_targets);
        };

        ASSERT_TRUE(gradient_same(functor, {input}, 1e-2));
    }
}

TEST_F(MatOpsTests, cross_entropy_colwise_multiindex) {
    EXPERIMENT_REPEAT {
        graph::NoBackprop nb;

        Mat<R> input (3, 5, weights<R>::uniform(0.01, 0.99));
        Mat<R> softmaxed = MatOps<R>::softmax_colwise(input);

        vector<uint> targets;
        for (int i = 0; i < input.dims(1); ++i)
            targets.push_back(utils::randint(0, input.dims(0) - 1));
        Mat<R> actual_res = MatOps<R>::softmax_cross_entropy_colwise(
                input, Indexing::Index(&targets));
        #ifdef DALI_USE_CUDA
            EXPECT_TRUE(actual_res.w().memory().gpu_fresh);
        #endif

        for (int i = 0; i < targets.size(); ++i) {
            // take each column separately
            auto expected_res = MatOps<R>::cross_entropy_colwise(softmaxed(NULL,i), targets[i]);
            ASSERT_NEAR(actual_res.w(i), expected_res.w(0), 1e-4);
        }
    }
}



TEST_F(MatOpsTests, softmax_cross_entropy_rowwise_grad) {
    // utils::random::set_seed(1234);
    EXPERIMENT_REPEAT {
        auto input = Mat<R>(2, 3, weights<R>::uniform(-2.0, 2.0));

        vector<uint> targets;
        for (int i = 0; i < input.dims(0); ++i)
            targets.push_back(utils::randint(0, input.dims(1) - 1));
        Indexing::Index indexed_targets(&targets);


        auto functor = [indexed_targets](vector<Mat<R>> Xs)-> Mat<R> {
            return MatOps<R>::softmax_cross_entropy_rowwise(
                Xs[0],
                indexed_targets);
        };

        ASSERT_TRUE(gradient_ratio_same(functor, {input}, 1e-2));
    }
    // utils::random::reseed();
}

TEST_F(MatOpsTests, cross_entropy_rowwise_multiindex) {
    EXPERIMENT_REPEAT {
        graph::NoBackprop nb;

        Mat<R> input (5, 3, weights<R>::uniform(0.01, 0.99));
        Mat<R> softmaxed = MatOps<R>::softmax_rowwise(input);

        vector<uint> targets;
        for (int i = 0; i < input.dims(0); ++i)
            targets.push_back(utils::randint(0, input.dims(1) - 1));
        Mat<R> actual_res = MatOps<R>::softmax_cross_entropy_rowwise(
                input, Indexing::Index(&targets));
        #ifdef DALI_USE_CUDA
            EXPECT_TRUE(actual_res.w().memory().gpu_fresh);
        #endif

        for (int i = 0; i < targets.size(); ++i) {
            // take each column separately
            auto expected_res = MatOps<R>::cross_entropy_rowwise(softmaxed[i], targets[i]);
            ASSERT_NEAR(actual_res.w(i), expected_res.w(0), 1e-4);
        }
    }
}

TEST_F(MatOpsTests, cross_entropy_rowwise_grad) {
    EXPERIMENT_REPEAT {
        auto input = Mat<R>(2, 3, weights<R>::uniform(0.01, 1.0));

        auto targets = Mat<int>(2, 1);
        for (int i = 0; i < input.dims(0); ++i)
            targets.w(i) = utils::randint(0, input.dims(1) - 1);

        auto functor = [&targets](vector<Mat<R>> Xs)-> Mat<R> {
            return MatOps<R>::cross_entropy_rowwise(
                Xs[0],
                targets);
        };

        ASSERT_TRUE(gradient_same(functor, {input}, 1e-2));
    }
}

TEST_F(MatOpsTests, cross_entropy_colwise_grad) {
    EXPERIMENT_REPEAT {
        auto input = Mat<R>(3, 2, weights<R>::uniform(0.01, 1.0));

        auto targets = Mat<int>(2, 1);
        for (int i = 0; i < input.dims(1); ++i)
            targets.w(i) = utils::randint(0, input.dims(0) - 1);

        auto functor = [&targets](vector<Mat<R>> Xs)-> Mat<R> {
            return MatOps<R>::cross_entropy_colwise(
                Xs[0],
                targets);
        };

        ASSERT_TRUE(gradient_same(functor, {input}, 1e-2));
    }
}

TEST_F(MatOpsTests, cross_entropy_grad_thought_target) {
    double temperature;
    int target;

    EXPERIMENT_REPEAT {
        const int hidden_size  = 10;
        const int num_examples = 3;
        const int input_size   = 5;
        double temperature = 1.0; // utils::randdouble(0.1, 100);

        auto layer = Mat<R>(hidden_size, input_size,     weights<R>::uniform(-2.0, 2.0));
        auto input = Mat<R>(input_size,  num_examples,   weights<R>::uniform(-2.0, 2.0));

        auto target = Mat<R>(hidden_size,  num_examples, weights<R>::uniform(0.15, 0.85));


        auto functor = [target, temperature](vector<Mat<R>> Xs)-> Mat<R> {
            auto soft = MatOps<R>::softmax_colwise(
                    Xs[1].dot(Xs[0]),
                    temperature
                );
            return MatOps<R>::cross_entropy(
                soft,
                Xs[2]);
        };

        ASSERT_TRUE(gradient_same(functor, {input, layer, target}, 1e-1));
    }
}

TEST_F(MatrixTests, row_pluck) {

    EXPERIMENT_REPEAT {
        Mat<R> A(5, 3, weights<R>::uniform(20.0));
        int row = utils::randint(0, A.dims(0) - 1);
        auto functor = [row](vector<Mat<R>> Xs) {
            return Xs[0][row];
        };
        ASSERT_TRUE(gradient_same(functor, {A}, 1e-4));
    }
}

TEST_F(MatrixTests, col_pluck) {
    EXPERIMENT_REPEAT {
        Mat<R> A(5, 3, weights<R>::uniform(20.0));
        const int col = utils::randint(0, A.dims(1) - 1);
        auto functor = [col](vector<Mat<R>> Xs) {
            #ifdef DALI_USE_CUDA
                // to ensure op works on gpu we force memory
                // freshness of the device
                Xs[0].w().memory().to_gpu();
            #endif
            auto res = MatOps<R>::col_pluck(Xs[0], col);
            return res;
        };
        ASSERT_TRUE(gradient_same(functor, {A}, 1e-4, 1e-2, true));
    }
}

TEST_F(MatrixTests, col_pluck_gpu_vs_cpu) {
    EXPERIMENT_REPEAT {
        Mat<R> A(5, 3, weights<R>::uniform(20.0));
        int col = utils::randint(0, A.dims(1) - 1);
        auto functor = [col](vector<Mat<R>> Xs) {
            return MatOps<R>::col_pluck(Xs[0], col);
        };
        ASSERT_TRUE(cpu_vs_gpu(functor, {A}, 1e-4));
    }
}

TEST_F(MatrixTests, rows_pluck_forward_correctness) {

    EXPERIMENT_REPEAT {
        graph::NoBackprop nb;
        const int num_plucks = 4;
        Mat<R> A(10, 5, weights<R>::uniform(20.0));

        vector<uint> plucks;
        for (int i = 0; i < num_plucks; ++i) {
            plucks.push_back(utils::randint(0, A.dims(0) - 1));
        }
        auto res = A[&plucks];
        #ifdef DALI_USE_CUDA
            EXPECT_TRUE(res.w().memory().gpu_fresh);
        #endif

        for (int pluck_idx = 0; pluck_idx < plucks.size(); ++pluck_idx) {
            auto actual_row = res[pluck_idx];
            auto expected_row = A[plucks[pluck_idx]];
            EXPECT_MATRIX_CLOSE(actual_row, expected_row, 1e-4);
        }
    }
}

TEST_F(MatrixTests, rows_pluck) {
    EXPERIMENT_REPEAT {
        const int num_plucks = 4;
        Mat<R> A(10, 5, weights<R>::uniform(20.0));

        vector<uint> plucks;
        for (int i = 0; i < num_plucks; ++i) {
            plucks.push_back(utils::randint(0, A.dims(0) - 1));
        }

        auto functor = [&plucks](vector<Mat<R>> Xs) {
            return Xs[0][&plucks];
        };
        ASSERT_TRUE(gradient_same(functor, {A}, 1e-4));
    }
}

TEST_F(MatOpsTests, vector_softmax) {
    int softmax_size = 10;
    EXPERIMENT_REPEAT {
        vector<Mat<R>> matrices;
        for (int i = 0; i < softmax_size; i++) {
            matrices.emplace_back(1,1, weights<R>::uniform(0.0, 2.0));
        }
        int row = utils::randint(0, softmax_size-1);

        auto functor = [row, softmax_size](vector<Mat<R>> Xs)-> Mat<R> {
            auto mats = MatOps<R>::softmax(Xs);
            return mats[row];
        };
        ASSERT_TRUE(gradient_same(functor, matrices, 5e-3));
    }
}


void copy_constructor_helper(bool copy_w, bool copy_dw) {
    Mat<R> original(3,3, weights<R>::uniform(20.0));
    Mat<R> copy(original, copy_w, copy_dw);

    copy.w(0,0) += 1.0;
    copy.dw(0,0) += 1.0;

    if (copy_w) {
        ASSERT_MATRIX_NEQ(original, copy);
    } else {
        ASSERT_MATRIX_EQ(original, copy);
    }

    if (copy_dw) {
        ASSERT_MATRIX_GRAD_NOT_CLOSE(original, copy, 1e-5);
    } else {
        ASSERT_MATRIX_GRAD_CLOSE(original, copy, 1e-5);
    }

    copy.w(0,0) -= 1.0;
    copy.dw(0,0) -= 1.0;
    ASSERT_MATRIX_GRAD_CLOSE(original, copy, 1e-5);
    ASSERT_MATRIX_CLOSE(original, copy, 1e-5);
}


TEST_F(MatrixTests, copy_constructor) {
    copy_constructor_helper(false, false);
    copy_constructor_helper(false, true);
    copy_constructor_helper(true,  false);
    copy_constructor_helper(true,  true);
}

TEST_F(MatrixTests, matrix_constant_check) {
    int num_examples           = 10;
    int input_size             = 3;

    auto X  = Mat<R>(input_size, num_examples, weights<R>::uniform(20.0));
    // THE ONLY VARIABLE CONSIDERED CONSTANT IN THIS TEST IS X HERE
    auto X_const = MatOps<R>::consider_constant(X);
    auto B = Mat<R>(input_size, num_examples, weights<R>::uniform(20.0));
    auto error = (((X_const * B) - 2.0) ^ 2).sum();
    error.grad();
    graph::backward();

    EXPECT_TRUE(MatOps<R>::grad_allclose(X, Mat<R>::zeros_like(X), 1e-9));
    EXPECT_FALSE(MatOps<R>::grad_allclose(B, Mat<R>::zeros_like(B), 1e-9));
    // HERE X IS NO LONGER CONST
    X = Mat<R>(input_size, num_examples, weights<R>::uniform(20.0));
    B = Mat<R>(input_size, num_examples, weights<R>::uniform(20.0));
    error = (((X * B) - 2.0) ^ 2).sum();
    error.grad();
    graph::backward();

    EXPECT_FALSE(MatOps<R>::grad_allclose(X, Mat<R>::zeros_like(X), 1e-9));
    EXPECT_FALSE(MatOps<R>::grad_allclose(B, Mat<R>::zeros_like(B), 1e-9));
}

TEST_F(MatrixTests, scalar_pow) {
    int height = 3;
    int width = 4;

    EXPERIMENT_REPEAT {
        auto mat = Mat<R>(height, width, weights<R>::uniform(1.0, 2.0));
        R exponent = utils::randdouble(0.4, 2.5);

        auto functor = [exponent](vector<Mat<R>> Xs)-> Mat<R> {
            return Xs[0] ^ exponent;
        };
        ASSERT_TRUE(gradient_same(functor, {mat}, 1e-3));
    }
}

TEST_F(MatrixTests, pow) {
    int height = 3;
    int width = 4;

    EXPERIMENT_REPEAT {

        auto mat = Mat<R>(height, width, weights<R>::uniform(0.5, 1.5));
        auto exponent = Mat<R>(1,1);
        exponent = MatOps<R>::fill(exponent, 2.4);

        auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
            return Xs[0] ^ Xs[1];
        };
        ASSERT_TRUE(gradient_same(functor, {mat, exponent}, 1e-3));
    }
}


TEST_F(MatrixTests, quadratic_form) {
    int left_size = 2;
    int right_size = 3;
    int left_size_outer = 4;
    int right_size_outer = 5;

    EXPERIMENT_REPEAT {
        auto left = Mat<R>(left_size, left_size_outer, weights<R>::uniform(20.0));
        auto middle = Mat<R>(left_size, right_size, weights<R>::uniform(20.0));
        auto right = Mat<R>(right_size, right_size_outer, weights<R>::uniform(20.0));

        auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
            return MatOps<R>::quadratic_form(Xs[0], Xs[1], Xs[2]);
        };
        ASSERT_TRUE(gradient_same(functor, {left, middle, right}, 1e-3));
    }
}

typedef std::function<std::shared_ptr<Solver::AbstractSolver<R>>(vector<Mat<R>>)> create_solver_t;

void test_solver(create_solver_t create_solver) {
    // minimize X.T() * W * X + W2 * X;
    Mat<R> X(5, 1, weights<R>::uniform(20.0));
    X = MatOps<R>::consider_constant(X);
    Mat<R> W(5, 5, weights<R>::uniform(20.0));
    Mat<R> W2(1, 5, weights<R>::uniform(20.0));

    W = W.dot(W.T()); // ensure positive definite.

    vector<Mat<R>> params({W, W2});
    auto solver = create_solver(params);

    int solver_iterations = 10;

    R last_error;

    for (int iter = 0; iter < solver_iterations; ++iter) {
        auto error = MatOps<R>::quadratic_form(X, W, X) + W2.dot(X);
        error.grad();
        graph::backward();
        solver->step(params);

        if (iter > 1) { // some solvers need an epoch to start up.
            ASSERT_LT(error.w(0) + 1e-5, last_error);
        }
        last_error = error.w(0);
    }
}

TEST(Solver, sgd) {
    test_solver([](vector<Mat<R>> params) {
        auto ret = std::make_shared<Solver::SGD<R>>(params);
        ret->step_size = 0.01;
        return ret;
    });
}

TEST(Solver, adagrad) {
    test_solver([](vector<Mat<R>> params) {
        auto ret = std::make_shared<Solver::AdaGrad<R>>(params);
        ret->step_size = 0.01;
        return ret;
    });
}

TEST(Solver, rmsprop) {
    test_solver([](vector<Mat<R>> params) {
        auto ret = std::make_shared<Solver::RMSProp<R>>(params);
        ret->step_size = 0.1;
        return ret;
    });
}

TEST(Solver, rmspropmomentum) {
    test_solver([](vector<Mat<R>> params) {
        auto ret = std::make_shared<Solver::RMSPropMomentum<R>>(params);
        // ret->step_size = 0.001;
        return ret;
    });
}

TEST(Solver, adadelta) {
    test_solver([](vector<Mat<R>> params) {
        auto ret = std::make_shared<Solver::AdaDelta<R>>(params);
        return ret;
    });
}

TEST(Solver, adam) {
    test_solver([](vector<Mat<R>> params) {
        auto ret = std::make_shared<Solver::Adam<R>>(params);
        return ret;
    });
}

Mat<R> create_dataset() {
    int num_points     = 20;
    int num_dimensions = 5;
    // create data
    auto dataset = Mat<R>();
    {
        graph::NoBackprop nb;
        auto pointsA = Mat<R>(
            num_dimensions,
            num_points,
            weights<R>::gaussian(0.0, 2.0)
        );
        auto pointsB = Mat<R>(
            num_dimensions,
            num_points,
            weights<R>::gaussian(0.0, 2.0)
        );
        auto point = Mat<R>(num_dimensions, 1);
        for (int i = 0; i < num_dimensions; i++)
            point.w(i) = 2;
        pointsA += point;

        for (int i = 0; i < num_dimensions; i++)
            point.w(i) = -2;
        pointsB += point;
        dataset = MatOps<R>::hstack({pointsA, pointsB});
    }
    return dataset;
}


void test_solver_optimization(std::string solvername) {
    utils::random::set_seed(1234);
    int num_points = 20;
    int num_dimensions = 5;
    // create data
    auto dataset = Mat<R>();
    {
        graph::NoBackprop nb;
        auto pointsA = Mat<R>(
            num_points,
            num_dimensions,
            weights<R>::gaussian(0.0, 2.0)
        );
        auto pointsB = Mat<R>(
            num_points,
            num_dimensions,
            weights<R>::gaussian(0.0, 2.0)
        );
        auto point = Mat<R>(1, num_dimensions);
        for (int i = 0; i < num_dimensions; i++)
            point.w(i) = 2;
        pointsA += point;

        for (int i = 0; i < num_dimensions; i++)
            point.w(i) = -2;
        pointsB += point;
        dataset = MatOps<R>::vstack({pointsA, pointsB});
    }

    int num_classes = 2;
    auto mat = Mat<R>(num_dimensions, num_classes, weights<R>::uniform(2.0));
    auto bias = Mat<R>(1,             num_classes, weights<R>::uniform(2.0));
    auto params = vector<Mat<R>>({mat, bias});

    auto solver = Solver::construct(solvername, params, 0.1);
    auto labels = vector<uint>();
    for (int i = 0; i < num_points * 2; i++)
        labels.emplace_back(i < num_points ? 0 : 1);

    R original_error = 0;
    {
        graph::NoBackprop nb;
        auto mat_err = MatOps<R>::softmax_cross_entropy_rowwise((dataset.dot(mat) + bias), &labels).sum();
        original_error = mat_err.w(0);
    }
    R error = original_error;
    for (int e = 0; e < 100; e++) {
        auto KL = MatOps<R>::softmax_cross_entropy_rowwise((dataset.dot(mat) + bias), &labels).sum();
        KL.grad();
        graph::backward();
        solver->step(params);
        error = KL.w(0);
    }
    // make 10x improvements (or else no VC funding)
    ASSERT_PRED_FORMAT2(SCALAR_COMP_LE, error, original_error / 10.0);
}

TEST(Solver, adagrad_optimization_test) {
    test_solver_optimization("adagrad");
}
TEST(Solver, sgd_optimization_test) {
    test_solver_optimization("sgd");
}
TEST(Solver, adadelta_optimization_test) {
    test_solver_optimization("adadelta");
}
TEST(Solver, adam_optimization_test) {
    test_solver_optimization("adam");
}
TEST(Solver, rmsprop_optimization_test) {
    test_solver_optimization("rmsprop");
}
TEST(Solver, rmspropmomentum_optimization_test) {
    test_solver_optimization("RMSPropMomentum");
}
