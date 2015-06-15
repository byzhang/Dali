#include <chrono>
#include <vector>
#include <gtest/gtest.h>

#include "dali/test_utils.h"
#include "dali/core.h"

using std::vector;
using std::chrono::milliseconds;
using ::testing::AssertionResult;
using ::testing::AssertionSuccess;
using ::testing::AssertionFailure;

#ifdef DALI_USE_CUDA
// most gpus don't support double
typedef float R;
#else
typedef double R;
#endif

#define NUM_RETRIES 10
#define EXPERIMENT_REPEAT for(int __repetition=0; __repetition < NUM_RETRIES; ++__repetition)

template<typename T>
AssertionResult buffer_equals (T* buffer1, T* buffer2, uint size1, uint size2) {
    if (size1 != size2)
        return AssertionFailure() << "Sizes differ first matrix is " << size1 << ", while second is " << size2;
    for (int i=0; i<size1; ++i) {
        if (buffer1[i] != buffer2[i])
            return AssertionFailure() << "Difference in datum " << i << " first tensor has "
                                      << buffer1[i] << ", while second has " << buffer2[i];
    }
    return AssertionSuccess();
}

template<typename T, typename J>
AssertionResult buffer_almost_equals(T* buffer1, T* buffer2, uint size1, uint size2, J eps) {
    if (size1 != size2)
        return AssertionFailure() << "Sizes differ first matrix is " << size1 << ", while second is " << size2;
    for (int i=0; i<size1; ++i) {
        if (abs(buffer1[i] - buffer2[i]) > eps)
            return AssertionFailure() << "Difference in datum " << i << " first tensor has "
                                      << buffer1[i] << ", while second has " << buffer2[i]
                                      << "(tolerance = " << eps << ")";
    }
    return AssertionSuccess();
}

#define ASSERT_MATRIX_EQ(A, B) ASSERT_TRUE(MatOps<R>::equals((A),(B)))
#define ASSERT_MATRIX_NEQ(A, B) ASSERT_FALSE(MatOps<R>::equals((A),(B)))
#define ASSERT_MATRIX_CLOSE(A, B, eps) ASSERT_TRUE(MatOps<R>::allclose((A),(B), (eps)))
#define ASSERT_MATRIX_NOT_CLOSE(A, B, eps) ASSERT_FALSE(MatOps<R>::allclose((A),(B), (eps)))
#define ASSERT_MATRIX_GRAD_CLOSE(A, B, eps) ASSERT_TRUE(MatOps<R>::grad_allclose((A),(B),(eps)))
#define ASSERT_MATRIX_GRAD_NOT_CLOSE(A, B, eps) ASSERT_FALSE(MatOps<R>::grad_allclose((A),(B),(eps)))

#define EXPECT_MATRIX_EQ(A, B) EXPECT_TRUE(MatOps<R>::equals((A),(B)))
#define EXPECT_MATRIX_NEQ(A, B) EXPECT_FALSE(MatOps<R>::equals((A),(B)))
#define EXPECT_MATRIX_CLOSE(A, B, eps) EXPECT_TRUE(MatOps<R>::allclose((A),(B), (eps)))
#define EXPECT_MATRIX_NOT_CLOSE(A, B, eps) EXPECT_FALSE(MatOps<R>::allclose((A),(B), (eps)))
#define EXPECT_MATRIX_GRAD_CLOSE(A, B, eps) EXPECT_TRUE(MatOps<R>::grad_allclose((A),(B),(eps)))
#define EXPECT_MATRIX_GRAD_NOT_CLOSE(A, B, eps) EXPECT_FALSE(MatOps<R>::grad_allclose((A),(B),(eps)))

#ifdef DALI_USE_CUDA
#define ASSERT_MAT_ON_GPU(A) ASSERT_TRUE(MAT(A).gpu_fresh)
#define EXPECT_MAT_ON_GPU(A) EXPECT_TRUE(MAT(A).gpu_fresh)
#else
#define ASSERT_MAT_ON_GPU(A)
#define EXPECT_MAT_ON_GPU(A)
#endif

class MatrixTests : public MemorySafeTest {
  protected:
    static void SetUpTestCase() {
        dali_init();
    }
};

/**
Gradient Same
-------------

Numerical gradient checking method. Performs
a finite difference estimation of the gradient
over each argument to a functor.

**/


void expect_computation_on_gpu(
        std::function<Mat<R>(std::vector<Mat<R>>&)> functor,
        std::vector<Mat<R>> arguments) {
    graph::NoBackprop nb;
    auto res = functor(arguments);
    EXPECT_MAT_ON_GPU(res);
    for (auto& arg: arguments) {
        EXPECT_MAT_ON_GPU(arg);
    }
}

void expect_args_remain_on_gpu(
        std::function<Mat<R>(std::vector<Mat<R>>&)> functor,
        std::vector<Mat<R>> arguments) {
    graph::NoBackprop nb;
    auto res = functor(arguments);
    for (auto& arg: arguments) {
        EXPECT_MAT_ON_GPU(arg);
    }
}


bool gradient_same(
        std::function<Mat<R>(std::vector<Mat<R>>&)> functor,
        std::vector<Mat<R>> arguments,
        R tolerance    = 1e-5,
        R grad_epsilon = 1e-9) {

    auto error = functor(arguments).sum();
    error.grad();
    graph::backward();

    bool worked_out = true;

    // from now on gradient is purely numerical:
    graph::NoBackprop nb;

    for (auto& arg : arguments) {
        R  Arg_prime[arg.number_of_elements()];
        R* arg_buffer = arg.w()->data();
        for (int i = 0; i < arg.number_of_elements(); i++) {
            auto prev_val     = arg_buffer[i];
            arg_buffer[i]     = prev_val +  grad_epsilon;
            auto obj_positive = MatOps<R>::consider_constant(functor(arguments)).sum().w(0);
            arg_buffer[i]     = prev_val - grad_epsilon;
            auto obj_negative = MatOps<R>::consider_constant(functor(arguments)).sum().w(0);

            arg_buffer[i]     = prev_val;
            Arg_prime[i]      = (obj_positive - obj_negative) / (2.0 * grad_epsilon);
        }
        AssertionResult did_work_out = buffer_almost_equals(
                (R*)Arg_prime,
                arg.dw()->data(),
                arg.number_of_elements(),
                arg.number_of_elements(),
                tolerance);
        // AssertionResult is a GoogleTest magic and it's castable to bool.
        worked_out = worked_out && (bool)did_work_out;
        if (!did_work_out) {
            std::cout << std::fixed
                      << std::setw( 7 ) // keep 7 digits
                      << std::setprecision( 3 ) // use 3 decimals
                      << std::setfill( ' ' ); // pad values with blanks this->w(i,j)
            std::cout << "-----------\nArg_prime:" << std::endl;
            for (uint i = 0; i < std::min(arg.number_of_elements(), (uint)12); ++i) {
                std::cout << Arg_prime[i] << " ";
            }
            std::cout << std::endl;
            std::cout << "-----------\narg.dw():" << std::endl;
            for (uint i = 0; i < std::min(arg.number_of_elements(), (uint)12); ++i) {
                std::cout << arg.dw(i) << " ";
            }
            std::cout << std::endl;
            if (arg.name != nullptr) {
                std::cout << "arg.name = " << *arg.name << std::endl;
            }
            std::cout << "-----------" << std::endl;
        }
    }

    return worked_out;
}

TEST_F(MatrixTests, sum_test) {
    auto A = Mat<R>(10, 20, weights<R>::uniform(2.0));
    auto res = A.sum();
    R sum = 0.0;
    for (int i = 0; i < A.number_of_elements(); ++i) {
        sum += A.w(i);
    }
    ASSERT_NEAR(sum, res.w(0), 1e-4);
}


TEST_F(MatrixTests, equals) {
    auto A = Mat<R>(10, 20, weights<R>::uniform(2.0));
    auto B = Mat<R>(10, 20, weights<R>::uniform(2.0));

    EXPECT_MATRIX_EQ(A, A)  << "A equals A.";
    EXPECT_MATRIX_NEQ(A, B) << "A different from B.";
//    EXPECT_MATRIX_CLOSE(A, A, 1e-4)  << "A near A.";
    EXPECT_MATRIX_NOT_CLOSE(A, B, 1e-4) << "A not near B.";

    EXPECT_MAT_ON_GPU(A);
    EXPECT_MAT_ON_GPU(B);

}


TEST_F(MatrixTests, sum_gradient) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return Xs[0].sum();
    };

    EXPERIMENT_REPEAT {
        auto A = Mat<R>(10, 20, weights<R>::uniform(2.0));
        expect_args_remain_on_gpu(functor, {A});
        EXPECT_TRUE(gradient_same(functor, {A}));
    }

}



// TEST_F(MatrixTests, recursive_sum) {
//     auto functor = [](vector<Mat<R>>& Xs)-> Mat<R> {
//         auto doubled = Xs[0] + Xs[0];
//         return doubled.sum();
//     };
//     EXPERIMENT_REPEAT {
//         auto A = Mat<R>(10, 20, weights<R>::uniform(2.0));
//         ASSERT_TRUE(gradient_same(functor, {A}, 1e-2));
//     }
// }

// TEST_F(MatrixTests, inplace_sum) {

//     EXPERIMENT_REPEAT {
//         auto A = Mat<R>(3, 4, weights<R>::uniform(2.0));
//         auto B = Mat<R>(3, 4, weights<R>::uniform(2.0));

//         auto functor = [&A, &B](vector<Mat<R>>& Xs)-> Mat<R> {
//             auto A_temp = A;
//             auto B_temp = B;
//             A_temp += B_temp;
//             return A_temp;
//         };
//         ASSERT_TRUE(gradient_same(functor, {A, B}, 1e-2));
//     }
// }

// TEST_F(MatrixTests, inplace_substract) {
//     EXPERIMENT_REPEAT {
//         auto A = Mat<R>(3, 4, weights<R>::uniform(2.0));
//         auto B = Mat<R>(3, 4, weights<R>::uniform(2.0));

//         auto functor = [&A, &B](vector<Mat<R>>& Xs)-> Mat<R> {
//             auto A_temp = A;
//             auto B_temp = B;
//             A_temp -= B_temp;
//             return A_temp;
//         };
//         ASSERT_TRUE(gradient_same(functor, {A, B}, 1e-2));
//     }
// }

// TEST_F(MatrixTests, inplace_divide) {
//     EXPERIMENT_REPEAT {
//         auto A = Mat<R>(3, 4, weights<R>::uniform(2.0));
//         auto B = Mat<R>(3, 4, weights<R>::uniform(2.0));

//         auto functor = [&A, &B](vector<Mat<R>>& Xs)-> Mat<R> {
//             auto A_temp = A;
//             auto B_temp = B;
//             A_temp /= B_temp;
//             return A_temp;
//         };
//         ASSERT_TRUE(gradient_same(functor, {A, B}, 1e-2));
//     }
// }

// TEST_F(MatrixTests, inplace_multiply) {
//     EXPERIMENT_REPEAT {
//         auto A = Mat<R>(3, 4, weights<R>::uniform(2.0));
//         auto B = Mat<R>(3, 4, weights<R>::uniform(2.0));

//         auto functor = [&A, &B](vector<Mat<R>>& Xs)-> Mat<R> {
//             auto A_temp = A;
//             auto B_temp = B;
//             A_temp *= B_temp;
//             return A_temp;
//         };
//         ASSERT_TRUE(gradient_same(functor, {A, B}, 1e-2));
//     }
// }



TEST_F(MatrixTests, addition_gradient) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return Xs[0] + Xs[1];
    };
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(10, 20, weights<R>::uniform(2.0));
        auto B = Mat<R>(10, 20,  weights<R>::uniform(0.5));
        ASSERT_TRUE(gradient_same(functor, {A, B}));
    }
}

/*
TEST_F(MatrixTests, addition_broadcast_gradient) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return Xs[0] + Xs[1];
    };
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(10, 20, weights<R>::uniform(2.0));
        auto B = Mat<R>(10, 1,  weights<R>::uniform(0.5));
        ASSERT_TRUE(gradient_same(functor, {A, B}));
    }
}

TEST_F(MatrixTests, mean_gradient) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return Xs[0].mean();
    };
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(10, 20, weights<R>::uniform(2.0));
        ASSERT_TRUE(gradient_same(functor, {A}));
    }
}

TEST_F(MatrixTests, sigmoid_gradient) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return Xs[0].sigmoid();
    };
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(10, 20, weights<R>::uniform(20.0));
        ASSERT_TRUE(gradient_same(functor, {A}, 1e-4));
    }
}

TEST_F(MatrixTests, tanh_gradient) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return Xs[0].tanh();
    };
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(10, 20, weights<R>::uniform(20.0));
        ASSERT_TRUE(gradient_same(functor, {A}, 1e-4));
    }
}

TEST_F(MatrixTests, norm_gradient) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return Xs[0].L2_norm();
    };
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(10, 20, weights<R>::uniform(20.0));
        ASSERT_TRUE(gradient_same(functor, {A}, 1e-4));
    }
}

TEST_F(MatrixTests, exp_gradient) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return Xs[0].tanh();
    };
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(10, 20, weights<R>::uniform(20.0));
        ASSERT_TRUE(gradient_same(functor, {A}, 1e-4));
    }
}

TEST_F(MatrixTests, log_gradient) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return Xs[0].tanh();
    };
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(10, 20, weights<R>::uniform(0.001, 20.0));
        ASSERT_TRUE(gradient_same(functor, {A}, 1e-4));
    }
}

TEST_F(MatrixTests, matrix_dot_plus_bias) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return Xs[1].dot(Xs[0]) + Xs[2];
    };
    int num_examples = 20;
    int hidden_size = 10;
    int input_size = 5;
    EXPERIMENT_REPEAT {
        auto X = Mat<R>(input_size, num_examples, weights<R>::uniform(20.0));
        auto W = Mat<R>(hidden_size, input_size, weights<R>::uniform(2.0));
        auto bias = Mat<R>(hidden_size, 1, weights<R>::uniform(2.0));
        ASSERT_TRUE(gradient_same(functor, {X, W, bias}, 1e-4));
    }
}

TEST_F(MatrixTests, matrix_divide) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return Xs[0] / Xs[1];
    };
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(10, 20, weights<R>::uniform(-20.0, 20.0));
        auto B = Mat<R>(10, 20, weights<R>::uniform(0.1, 20.0));
        ASSERT_TRUE(gradient_same(functor, {A, B}, 1e-4));
    }
}

TEST_F(MatrixTests, matrix_divide_broadcast) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return Xs[0] / Xs[1];
    };
    EXPERIMENT_REPEAT {
        auto A = Mat<R>(10, 20, weights<R>::uniform(-20.0, 20.0));
        auto B = Mat<R>(10, 1, weights<R>::uniform(0.1, 20.0));
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
        ASSERT_TRUE(gradient_same(functor, {A}, 1e-3));
    }
}


// TEST_F(MatrixTests, divide_inplace) {
//     auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
//         Xs[0] /= 20.0;
//         return (Xs[0] - 2.0) ^ 2;
//     };
//     EXPERIMENT_REPEAT {
//         auto A = Mat<R>(10, 20, weights<R>::uniform(0.001, 20.0));
//         ASSERT_TRUE(gradient_same(functor, {A}, 1e-4));
//     }
// }

typedef MemorySafeTest MatOpsTests;

TEST_F(MatOpsTests, matrix_mul_with_bias) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return MatOps<R>::mul_with_bias(Xs[1], Xs[0], Xs[2]);
    };
    int num_examples = 20;
    int hidden_size = 10;
    int input_size = 5;
    EXPERIMENT_REPEAT {
        auto X = Mat<R>(input_size, num_examples, weights<R>::uniform(20.0));
        auto W = Mat<R>(hidden_size, input_size, weights<R>::uniform(2.0));
        auto bias = Mat<R>(hidden_size, 1, weights<R>::uniform(2.0));
        ASSERT_TRUE(gradient_same(functor, {X, W, bias}, 1e-4));
    }
}

TEST_F(MatOpsTests, matrix_mul_add_mul_with_bias) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return MatOps<R>::mul_add_mul_with_bias(Xs[0], Xs[1], Xs[2], Xs[3], Xs[4]);
    };
    int num_examples = 20;
    int hidden_size = 10;
    int input_size = 5;
    int other_input_size = 7;
    EXPERIMENT_REPEAT {
        auto X       = Mat<R>(input_size, num_examples,       weights<R>::uniform(20.0));
        auto X_other = Mat<R>(other_input_size, num_examples, weights<R>::uniform(20.0));
        auto W       = Mat<R>(hidden_size, input_size,        weights<R>::uniform(2.0));
        auto W_other = Mat<R>(hidden_size, other_input_size,  weights<R>::uniform(2.0));
        auto bias    = Mat<R>(hidden_size, 1,                 weights<R>::uniform(2.0));
        ASSERT_TRUE(gradient_same(functor, {W, X, W_other, X_other, bias}, 0.0003));
    }
}

TEST_F(MatOpsTests, matrix_conv2d) {
    graph::NoBackprop nb;

    auto image = Mat<R>(10, 10);
    int block_width  = 4,
        block_offset = 3,
        kernel_width = 3,
        kernel_height = 3;
    R filler = 2.0;

    image.w()->w.block(
        block_offset,
        block_offset,
        block_width,
        block_width).fill(filler);

    auto kernel = Mat<R>(kernel_width, kernel_height);

    kernel = MatOps<R>::fill(kernel, 1);

    auto out = MatOps<R>::conv2d(image, kernel);

    auto expected = Mat<R>(
        image.dims(0) - kernel.dims(0) + 1,
        image.dims(1) - kernel.dims(1) + 1);

    expected.w()->w.block(
        block_offset,
        block_offset,
        block_width - kernel_width + 1,
        block_width - kernel_height + 1).fill(filler);

    ASSERT_EQ( (*out.sum().w())(0), (block_width * block_width * filler)) << "Sum of convolution with image should be sum of image";

    // TODO: test more properties here.
    ASSERT_TRUE((
        expected.w()->w.block(
            block_offset,
            block_offset,
            block_width - kernel_width + 1,
            block_width - kernel_height + 1).array() ==
        out.w()->w.block(
            block_offset,
            block_offset,
            block_width - kernel_width + 1,
            block_width - kernel_height + 1).array()).all()) << "Center of kernel activations should match up.";
}

TEST_F(MatOpsTests, matrix_conv2d_grad) {
    auto functor = [](vector<Mat<R>> Xs)-> Mat<R> {
        return MatOps<R>::conv2d(Xs[0], Xs[1]).tanh();
    };
    EXPERIMENT_REPEAT {
        auto kernel = Mat<R>(5, 5, weights<R>::uniform(-20.0, 20.0));
        auto image = Mat<R>(8, 8, weights<R>::uniform(-20.0, 20.0));
        ASSERT_TRUE(gradient_same(functor, {image, kernel}, 1e-4));
    }
}

TEST_F(MatOpsTests, softmax_temperature) {
    graph::NoBackprop nb;

    auto mat = Mat<R>(10, 1);
    for (int i = 0; i < 10; i++) mat.w(i) = i;

    auto base_prob = MatOps<R>::softmax(mat, 1.0);

    auto flat = MatOps<R>::softmax(
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
        auto new_prob = MatOps<R>::softmax(
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
    double temperature;
    int target = 8;
    auto functor = [&target, &temperature](vector<Mat<R>> Xs)-> Mat<R> {
        auto soft = MatOps<R>::softmax(
                Xs[1].dot(Xs[0]),
                temperature
            );
        return MatOps<R>::cross_entropy(
            soft,
            target);
    };
    EXPERIMENT_REPEAT {
        temperature = utils::randdouble(0.1, 100);
        auto input = Mat<R>(5,  3, weights<R>::uniform(-2.0, 2.0));
        auto layer = Mat<R>(10, 5, weights<R>::uniform(-2.0, 2.0));
        ASSERT_TRUE(gradient_same(functor, {input, layer}, 1e-4));
    }
}

TEST_F(MatOpsTests, matrix_conv1d_grad) {
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

TEST_F(MatOpsTests, vector_softmax) {
    int softmax_size = 15;
    EXPERIMENT_REPEAT {
        double temperature = utils::randdouble(0.1, 100.0);
        vector<Mat<R>> matrices;
        for (int i = 0; i < softmax_size; i++) {
            matrices.emplace_back(1,1, weights<R>::uniform(-20.0, 20.0));
        }
        auto functor = [&matrices, &temperature](vector<Mat<R>> Xs)-> Mat<R> {
            auto mats = MatOps<R>::softmax(matrices, temperature);
            return (mats[4] - 1.0) ^ 2;
        };
        ASSERT_TRUE(gradient_same(functor, {matrices}, 1e-4));
    }
}

typedef MemorySafeTest LayerTests;

TEST_F(LayerTests, layer_tanh_gradient) {
    int num_examples = 10;
    int hidden_size = 10;
    int input_size = 5;

    EXPERIMENT_REPEAT {
        auto X  = Mat<R>(input_size, num_examples,      weights<R>::uniform(20.0));
        auto mylayer = Layer<R>(input_size, hidden_size);
        auto params = mylayer.parameters();
        params.emplace_back(X);
        auto functor = [&mylayer](vector<Mat<R>> Xs)-> Mat<R> {
            return mylayer.activate(Xs.back()).tanh();
        };
        ASSERT_TRUE(gradient_same(functor, params, 0.0003));
    }
}

TEST_F(LayerTests, BroadcastMultiply) {

    int large_size = 10;
    int out_size   = 2;

    // different input sizes passed to a stacked input layer
    vector<int> input_sizes   = {5,  2,  5,  1, 5};

    // broadcast the 1s into the larger dimension:
    vector<int> example_sizes = {large_size, 1, large_size, 1, large_size};

    EXPERIMENT_REPEAT {
        // build layer
        auto mylayer = StackedInputLayer<R>(input_sizes, out_size);

        // build inputs
        vector<Mat<R>> inputs;
        for (int i = 0; i < input_sizes.size(); i++) {
            inputs.emplace_back(input_sizes[i], example_sizes[i], weights<R>::uniform(5.0));
        }

        // add the params
        auto params = mylayer.parameters();
        params.insert(params.end(), inputs.begin(), inputs.end());

        // project
        auto functor = [&mylayer, &inputs](vector<Mat<R>> Xs)-> Mat<R> {
            return mylayer.activate(inputs);
        };
        ASSERT_TRUE(gradient_same(functor, params, 0.0003));
    }
}

TEST_F(LayerTests, stacked_layer_tanh_gradient) {

    int num_examples = 10;
    int hidden_size  = 10;
    int input_size_1 = 5;
    int input_size_2 = 8;
    int input_size_3 = 12;

    EXPERIMENT_REPEAT {
        auto A  = Mat<R>(
            input_size_1,
            num_examples,
            weights<R>::uniform(20.0));
        auto B  = Mat<R>(
            input_size_2,
            num_examples,
            weights<R>::uniform(20.0));
        auto C  = Mat<R>(
            input_size_3,
            num_examples,
            weights<R>::uniform(20.0));
        auto mylayer = StackedInputLayer<R>({
            input_size_1,
            input_size_2,
            input_size_3}, hidden_size);
        auto params = mylayer.parameters();
        params.emplace_back(A);
        params.emplace_back(B);
        params.emplace_back(C);
        auto functor = [&mylayer, &A, &B, &C](vector<Mat<R>> Xs)-> Mat<R> {
            return mylayer.activate({A, B, C}).tanh();
        };
        ASSERT_TRUE(gradient_same(functor, params, 0.0003));
    }
}

TEST_F(LayerTests, LSTM_Zaremba_gradient) {

    int num_examples           = 10;
    int hidden_size            = 5;
    int input_size             = 3;

    EXPERIMENT_REPEAT {
        auto X  = Mat<R>(input_size, num_examples, weights<R>::uniform(20.0));
        auto mylayer = LSTM<R>(input_size, hidden_size, false);
        auto params = mylayer.parameters();
        params.emplace_back(X);
        auto initial_state = mylayer.initial_states();
        auto functor = [&mylayer, &X, &initial_state](vector<Mat<R>> Xs)-> Mat<R> {
            auto myout_state = mylayer.activate(X, initial_state);
            return myout_state.hidden;
        };
        ASSERT_TRUE(gradient_same(functor, params, 0.0003));
    }
}

TEST_F(LayerTests, LSTM_Graves_gradient) {

    int num_examples           = 10;
    int hidden_size            = 5;
    int input_size             = 3;

    EXPERIMENT_REPEAT {
        auto X  = Mat<R>(input_size, num_examples,      weights<R>::uniform(20.0));
        auto mylayer = LSTM<R>(input_size, hidden_size, true);
        auto params = mylayer.parameters();
        params.emplace_back(X);

        // In an LSTM we do not back prop through the cell activations when using
        // it in a gate, but to test it here we set this to true
        mylayer.backprop_through_gates = true;

        auto initial_state = mylayer.initial_states();
        auto functor = [&mylayer, &X, &initial_state](vector<Mat<R>> Xs)-> Mat<R> {
            auto myout_state = mylayer.activate(X, initial_state);
            return myout_state.hidden;
        };
        ASSERT_TRUE(gradient_same(functor, params, 0.0003));
    }
}

TEST_F(LayerTests, LSTM_Graves_shortcut_gradient) {

    int num_examples           = 10;
    int hidden_size            = 5;
    int input_size             = 3;
    int shortcut_size          = 2;

    EXPERIMENT_REPEAT {
        auto X  = Mat<R>(input_size,    num_examples, weights<R>::uniform(20.0));
        auto X_s = Mat<R>(shortcut_size, num_examples, weights<R>::uniform(20.0));
        auto mylayer = LSTM<R>({input_size, shortcut_size}, hidden_size, 1, true);
        auto params = mylayer.parameters();
        params.emplace_back(X);
        params.emplace_back(X_s);

        // In an LSTM we do not back prop through the cell activations when using
        // it in a gate:
        mylayer.backprop_through_gates = true;

        auto initial_state = mylayer.initial_states();
        auto functor = [&mylayer, &X, &X_s, &initial_state](vector<Mat<R>> Xs)-> Mat<R> {
            auto state = mylayer.activate_shortcut(X, X_s, initial_state);
            return state.hidden;
        };
        ASSERT_TRUE(gradient_same(functor, params, 0.0003));
    }
}

TEST_F(LayerTests, LSTM_Zaremba_shortcut_gradient) {
    int num_examples           = 10;
    int hidden_size            = 5;
    int input_size             = 3;
    int shortcut_size          = 2;

    EXPERIMENT_REPEAT {
        auto X  = Mat<R>(input_size,    num_examples, weights<R>::uniform(20.0));
        auto X_s = Mat<R>(shortcut_size, num_examples, weights<R>::uniform(20.0));
        auto mylayer = LSTM<R>({input_size, shortcut_size}, hidden_size, 1, false);
        auto params = mylayer.parameters();
        params.emplace_back(X);
        params.emplace_back(X_s);

        auto initial_state = mylayer.initial_states();
        auto functor = [&mylayer, &X, &X_s, &initial_state](vector<Mat<R>> Xs)-> Mat<R> {
            auto state = mylayer.activate_shortcut(X, X_s, initial_state);
            return state.hidden;
        };
        ASSERT_TRUE(gradient_same(functor, params, 0.0003));
    }
}

void copy_constructor_helper(bool copy_w, bool copy_dw) {
    Mat<R> original(10,10, weights<R>::uniform(20.0));
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


}

TEST_F(MatrixTests, copy_constructor) {
    copy_constructor_helper(false, false);
    copy_constructor_helper(false, true);
    copy_constructor_helper(true, false);
    copy_constructor_helper(true, true);
}

TEST_F(LayerTests, RNN_gradient_vs_Stacked_gradient) {
    int num_examples           = 10;
    int hidden_size            = 5;
    int input_size             = 3;

    EXPERIMENT_REPEAT {

        auto X  = Mat<R>(input_size, num_examples, weights<R>::uniform(20.0));
        auto H  = Mat<R>(hidden_size, num_examples, weights<R>::uniform(20.0));

        auto X_s  = Mat<R>(X, true, true); // perform full copies
        auto H_s  = Mat<R>(H, true, true); // perform full copies

        auto rnn_layer = RNN<R>(input_size, hidden_size);
        auto stacked_layer = StackedInputLayer<R>({input_size, hidden_size}, hidden_size);

        auto params = rnn_layer.parameters();
        auto stacked_params = stacked_layer.parameters();

        for (auto it1 = params.begin(),
                  it2 = stacked_params.begin(); (it1 != params.end()) && (it2 != stacked_params.end()); it1++, it2++) {
            ASSERT_EQ((*it1).dims(), (*it2).dims());
            std::copy(it2->w()->data(), it2->w()->data() + it2->number_of_elements(),
                      it1->w()->data());
        }

        auto error = ((rnn_layer.activate(X, H).tanh() - 1) ^ 2).sum();
        error.grad();
        auto error2 = ((stacked_layer.activate({X_s, H_s}).tanh() - 1) ^ 2).sum();
        error2.grad();
        graph::backward();

        for (auto it1 = params.begin(),
                  it2 = stacked_params.begin(); (it1 != params.end()) && (it2 != stacked_params.end()); it1++, it2++) {
            ASSERT_MATRIX_GRAD_CLOSE((*it1), (*it2), 1e-6);
        }
        ASSERT_MATRIX_GRAD_CLOSE(X, X_s, 1e-6);
        ASSERT_MATRIX_GRAD_CLOSE(H, H_s, 1e-6);
    }
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

    EXPECT_TRUE((X.dw()->dw.array() == Mat<R>::zeros_like(X).w()->w.array()).all());
    EXPECT_FALSE((B.dw()->dw.array() == Mat<R>::zeros_like(X).w()->w.array()).all());

    // HERE X IS NO LONGER CONST
    X = Mat<R>(input_size, num_examples, weights<R>::uniform(20.0));
    B = Mat<R>(input_size, num_examples, weights<R>::uniform(20.0));
    error = (((X * B) - 2.0) ^ 2).sum();
    error.grad();
    graph::backward();

    EXPECT_FALSE((X.dw()->dw.array() == Mat<R>::zeros_like(X).w()->w.array()).all());
    EXPECT_FALSE((B.dw()->dw.array() == Mat<R>::zeros_like(X).w()->w.array()).all());
}

TEST_F(LayerTests, shortcut_test) {
    int input_size = 10;
    int num_examples = 2;
    auto hidden_sizes = {40, 30};//{30, 13, 20, 1, 9, 2};

    auto model = StackedLSTM<R>(input_size, hidden_sizes, true, true);
    auto X = {Mat<R>(
        input_size,
        num_examples,
        weights<R>::uniform(20.0)
    )};

    auto out_states = model.activate_sequence(model.initial_states(),
                                              X,
                                              0.2);
}

TEST_F(LayerTests, multi_input_lstm_test) {
    int num_children = 3;
    int input_size = 4;
    int hidden_size = 2;
    int num_examples = 3;

    EXPERIMENT_REPEAT {
        auto input = Mat<R>(input_size,    num_examples, weights<R>::uniform(20.0));
        vector<LSTM<R>::State> states;
        for (int cidx = 0 ; cidx < num_children; ++cidx) {
            states.emplace_back(
                Mat<R>(hidden_size, num_examples, weights<R>::uniform(20.0)),
                Mat<R>(hidden_size, num_examples, weights<R>::uniform(20.0))
            );
        }

        auto mylayer = LSTM<R>(input_size, hidden_size, num_children);
        auto params = mylayer.parameters();
        params.emplace_back(input);
        for(auto& state: states) {
            params.emplace_back(state.memory);
            params.emplace_back(state.hidden);
        }

        auto functor = [&mylayer, &input, &states](vector<Mat<R>> Xs)-> Mat<R> {
                auto state = mylayer.activate(input, states);
                return state.hidden;
        };
        ASSERT_TRUE(gradient_same(functor, params, 0.0003));

        utils::Timer::report();
    }
}

TEST_F(MatrixTests, log_exp) {
    EXPERIMENT_REPEAT {
        graph::NoBackprop nb;
        auto mat = Mat<R>(10, 10, weights<R>::uniform(0.1, 20.0));
        auto log_mat = mat.log();
        auto exp_log_mat = log_mat.exp();
        ASSERT_MATRIX_CLOSE(mat, exp_log_mat, 1e-6);
    }
}

TEST_F(LayerTests, activate_sequence) {
    vector<int> hidden_sizes = {7, 10};
    int input_size = 5;
    int num_out_states = hidden_sizes.size();
    vector<Mat<R>> sequence;
    for (int i = 0; i < 10; i++) {
        sequence.emplace_back(input_size, 1);
    }
    auto model = StackedLSTM<R>(input_size, hidden_sizes, false, false);
    auto out_states = model.activate_sequence(model.initial_states(), sequence, 0.1);
    ASSERT_EQ(num_out_states, LSTM<R>::State::hiddens(out_states).size());
}

TEST_F(LayerTests, GRU) {
    int input_size = 3;
    int hidden_size = 5;
    int tsteps = 5;

    EXPERIMENT_REPEAT {
        auto gru = GRU<R>(input_size, hidden_size);
        auto params = gru.parameters();
        auto inputs = vector<Mat<R>>();
        for (int i = 0; i < tsteps; i++) inputs.emplace_back(Mat<R>(input_size,1, weights<R>::uniform(20.0)));
        auto functor = [&inputs, &gru,&tsteps, &hidden_size, &input_size](vector<Mat<R>> Xs)-> Mat<R> {
            auto state = Mat<R>(hidden_size, 1);
            for (int i = 0; i < tsteps; i++)
                state = gru.activate(inputs[i], state);
            return (state -1.0) ^ 2;
        };
        ASSERT_TRUE(gradient_same(functor, params, 1e-5));
    }
}

TEST_F(MatrixTests, powtest) {
    int height = 3;
    int width = 4;

    EXPERIMENT_REPEAT {

        auto mat = Mat<R>(height, width, weights<R>::uniform(0.1, 20.0));

        auto exponent = Mat<R>(1,1);
        MatOps<R>::fill(exponent, 2.0);

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

TEST_F(MatrixTests, argsort) {
    auto mats = vector<Mat<R>>({
        MatOps<R>::fill(Mat<R>(1,1), 3),
        MatOps<R>::fill(Mat<R>(1,1), 9),
        MatOps<R>::fill(Mat<R>(1,1), 1),
    });
    auto sorted = utils::argsort(mats);
    ASSERT_EQ(sorted, std::vector<size_t>({2, 0, 1}));
}


TEST(Solver, adagrad) {
    int num_points = 20;
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

    int num_classes = 2;
    auto mat = Mat<R>(num_classes, num_dimensions, weights<R>::uniform(2.0));
    auto bias = Mat<R>(num_classes, 1, weights<R>::uniform(2.0));
    auto params = vector<Mat<R>>({mat, bias});
    Solver::AdaGrad<R> solver(params);
    solver.step_size = 0.1;
    auto labels = vector<uint>();
    for (int i = 0; i < num_points * 2; i++)
        labels.emplace_back(i < num_points ? 0 : 1);

    R original_error = 0;
    {
        graph::NoBackprop nb;
        auto mat_err = MatOps<R>::softmax_cross_entropy((mat.dot(dataset) + bias), labels).sum();
        original_error = mat_err.w(0);
    }
    R error = original_error;
    for (int e = 0; e < 100; e++) {
        auto KL = MatOps<R>::softmax_cross_entropy((mat.dot(dataset) + bias), labels).sum();
        KL.grad();
        graph::backward();
        solver.step(params);
        error = KL.w(0);
    }
    // make 10x improvements (or else no VC funding)
    ASSERT_TRUE(original_error / 10.0 > error);
}

*/
