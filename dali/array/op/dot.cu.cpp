#include "dot.h"

#include <vector>

#include "dali/array/function/function.h"
#include "dali/array/shape.h"
#include "dali/array/function/args/dali_gemm_engine_exp.h"
#include "dali/array/op.h"


template<OPERATOR_T operator_t, int devT, typename T>
struct LazyGemmRunner {
    template <
        OPERATOR_T var_operator_t = operator_t,
        typename var_T = T,
        typename std::enable_if<
            !(
                (var_operator_t == OPERATOR_T_MUL) ||
                (var_operator_t == OPERATOR_T_DIV)
            )
        >::type* = nullptr
    >
    static void run(TypedArray<devT, T>& out,
                    const TypedArray<devT, T>& a,
                    const TypedArray<devT, T>& b) {
        typedef decltype(a.contiguous_d2()) mshadow_tensor_t;
        bool             a_transposed, b_transposed;
        mshadow_tensor_t a_tensor,     b_tensor;
        std::tie(a_transposed, a_tensor) = a.blas_friendly_tensor();
        std::tie(b_transposed, b_tensor) = b.blas_friendly_tensor();

        operator_assign_contiguous<operator_t, 2>(
            out,
            dali_gemm(
                a_tensor,
                b_tensor,
                a_transposed,
                b_transposed,
                (T)1.0f
            )
        );
    }

    template <
        OPERATOR_T var_operator_t = operator_t,
        typename var_T = T,
        typename std::enable_if<
            (var_operator_t == OPERATOR_T_MUL) ||
            (var_operator_t == OPERATOR_T_DIV)
        >::type* = nullptr
    >
    static void run(TypedArray<devT, T>& out,
                    const TypedArray<devT, T>& a,
                    const TypedArray<devT, T>& b) {
        ASSERT2(!(var_operator_t == OPERATOR_T_MUL) && !(var_operator_t == OPERATOR_T_MUL),
                "Matrix multiplication's result cannot be inplace-multiplied or inplace-divided.");
        ASSERT2(false, "If asserts above are complete this message should never be displayed");
    }
};


struct LazyReshapedGemm : public Function<LazyReshapedGemm, Array, Array, Array, std::vector<int>> {
    static std::vector<int> deduce_output_bshape(
            const Array& a,
            const Array& b,
            const std::vector<int>& output_shape) {
        return output_shape;
    }

    static void preprocess_out(const OPERATOR_T& operator_t, Array& out, const Array& a, const Array& b, const std::vector<int>& output_shape) {
        // make output look like 2D output
        ASSERT2(a.ndim() == 2 && b.ndim() == 2,
                utils::MS() << "Gemm inputs must be matrices, got a.ndim()=" << a.ndim() << " and b.ndim()=" << b.ndim() << " tensors.");
        ASSERT2(a.shape()[1] == b.shape()[0],
            utils::MS() << "shapes " << a.shape() << " and " << b.shape() << " not aligned: "
                        << a.shape()[1] << " (dim 1) != " << b.shape()[0] << " (dim 0)");
        out = out.copyless_reshape(
            {a.shape()[0], b.shape()[1]}
        );
    }
    static void postprocess_out(const OPERATOR_T& operator_t, Array& out, const Array& a, const Array& b, const std::vector<int>& output_shape) {
        // return to tensordot output shape
        out = out.copyless_reshape(output_shape);
    }

    template<OPERATOR_T operator_t, int devT, typename T>
    void typed_eval(
            TypedArray<devT, T> out,
            TypedArray<devT, T> a,
            TypedArray<devT, T> b,
            const std::vector<int>& output_shape) {
        LazyGemmRunner<operator_t,devT,T>::run(out, a, b);
    }
};


void check_tensordot_reduce_axes(
        const Array& operand,
        char name,
        const std::vector<int>& reduce_axes,
        const bool& batched) {
    ASSERT2(reduce_axes.size() <= operand.ndim(),
        utils::MS() << "length of argument " << name << "_reduce_axes "
                    << "should be less than the dimensions of " << name
                    << " (" << name << ".ndim()=" << operand.ndim()
                    << ", " << name << "_reduce_axes.size()="
                    << reduce_axes.size() << ")."
    );
    auto max_reduce_dim = std::max_element(
        reduce_axes.begin(), reduce_axes.end()
    );
    ASSERT2(reduce_axes.size() == 0 || (*max_reduce_dim) < operand.ndim(),
        utils::MS() << name << "_reduce_axes contains reduction dimensions "
                    << " that are greater than or equal to "
                    << name << ".ndim() ("
                    << name << ".ndim()=" << operand.ndim()
                    << ", and found max(" << name << "_reduce_axes)="
                    << *max_reduce_dim << ")."
    );
    ASSERT2(
        (!batched ||
        std::find(reduce_axes.begin(), reduce_axes.end(), 0) != reduce_axes.end()
        ),
        utils::MS() << "axes to sum over must not contain the batch axis "
                    << "(" << name << "_reduce_axes="
                    << reduce_axes << ")."
    );
}

std::vector<int> get_tensordot_dimshuffle_axes(
        const int& ndim,
        const std::vector<int>& reduce_axes,
        const bool& batched) {
    std::vector<int> other_axes;
    for (int x = 0; x < ndim; x++) {
        // when batched, 0 is always kept
        // as leading dim, and thus will not
        // be dimshuffled
        if (batched && x == 0) {
            continue;
        }
        bool not_in_reduce_axes = (
            std::find(
                reduce_axes.begin(),
                reduce_axes.end(),
                x
            ) == reduce_axes.end()
        );

        if (not_in_reduce_axes) {
            other_axes.emplace_back(x);
        }
    }
    return other_axes;
}


namespace op {
    enum DOT_TYPE_T {
        DOT_TYPE_2D_T = 0,
        DOT_TYPE_BATCHED_T = 1
    };

    AssignableArray _tensordot_as_dot(
            const Array& a,
            const Array& b,
            const int& axis,
            DOT_TYPE_T dot_type,
            bool batched) {
        // This code follows the logic from theano's tensordot as dot
        // [source https://github.com/Theano/Theano/blob/master/theano/tensor/basic.py#L5628]
        //
        // Theano code was also originally based elsewhere on
        // Tijmen Tieleman's gnumpy:
        // [source http://www.cs.toronto.edu/~tijmen/gnumpy.html]

        // if 'axes' is a single number of axes to multiply and sum over
        // (trailing axes of a, leading axes of b), we can just reshape
        // and use dot.
        // validate that the axis used for summing
        // is not out of bounds for the arguments a and b
        ASSERT2(
            axis >= 0,
            utils::MS() << "axis must be a non-negative integer (got " << axis << ")."
        );
        for (int i = 0; i < 2; i++) {
            auto& operand = i == 0 ? a : b;
            char operand_name = i == 0 ? 'a' : 'b';
            ASSERT2(
                axis <= operand.ndim(),
                utils::MS() << "axis can not be larger than the dimension of "
                            << operand_name
                            << " (" << operand_name << ".ndim()=" << operand.ndim()
                            << ", axis=" << axis <<")."
            );
            ASSERT2(!(axis == operand.ndim() && batched),
                utils::MS() << "axis to sum over must not include the batch axis "
                            << "of " << operand_name
                            << " (" << operand_name << ".ndim()=" << operand.ndim()
                            << ", axis=" << axis <<")."
            );
        }
        int batch_axes = batched ? 1 : 0;

        std::vector<int> a_shape = {1, 1};
        std::vector<int> b_shape = {1, 1};

        const auto& a_old_shape = a.shape();
        const auto& b_old_shape = b.shape();

        // compute total size of summed axes
        for (int i = 0; i < axis; i++) {
            a_shape[1] *= a_old_shape[a_old_shape.size() - (i + 1)];
            b_shape[0] *= b_old_shape[batch_axes + i];
        }
        // compute total size of other axes
        for (int i = 0; i < (a.ndim() - axis - batch_axes); i++) {
            a_shape[0] *= a_old_shape[batch_axes + i];
        }
        for (int i = 0; i < (b.ndim() - axis - batch_axes); i++) {
            b_shape[1] *= b_old_shape[b_old_shape.size() -(i + 1)];
        }

        if (batched) {
            a_shape.insert(a_shape.begin(), a_old_shape[0]);
            b_shape.insert(b_shape.begin(), b_old_shape[0]);
        }
        auto a_reshaped = a.reshape(a_shape);
        auto b_reshaped = b.reshape(b_shape);


        std::vector<int> output_shape;

        output_shape.insert(
            output_shape.begin(),
            a_old_shape.begin(),
            a_old_shape.begin() + a_old_shape.size() - axis
        );

        output_shape.insert(
            output_shape.end(),
            b_old_shape.begin() + batch_axes + axis,
            b_old_shape.end()
        );

        return LazyReshapedGemm::run(
            a_reshaped, b_reshaped, output_shape
        );
    }

    AssignableArray matrix_vector_dot(
        const Array& a,
        const Array& b) {
        ASSERT2((a.ndim() == 1 && b.ndim() == 2) || (a.ndim() == 2 && b.ndim() == 1),
                utils::MS() << "Gemv inputs must be a vector and a matrix, but got a.ndim()="
                            << a.ndim() << " and b.ndim()=" << b.ndim() << " tensors.");
        std::vector<int> outshape(1);
        if (a.ndim() == 1 && b.ndim() == 2) {
            ASSERT2(
                a.bshape()[0] == -1 || b.bshape()[0] == a.bshape()[0] || b.bshape()[0] == -1,
                utils::MS() << "shapes " << a.shape() << " and " << b.shape() << " not aligned: "
                            << a.shape()[0] << " (dim 0) != " << b.shape()[0] << " (dim 0)");
            outshape[0] = b.bshape()[1];
            return LazyReshapedGemm::run(
                a.copyless_reshape({1, a.number_of_elements()}),
                b,
                outshape
            );
        } else {
            ASSERT2(
                b.bshape()[0] == -1 || a.bshape()[1] == b.bshape()[0] || a.bshape()[0] == -1,
                utils::MS() << "shapes " << a.shape() << " and " << b.shape() << " not aligned: "
                            << a.shape()[1] << " (dim 1) != " << b.shape()[0] << " (dim 0)");
            outshape[0] = a.bshape()[0];
            return LazyReshapedGemm::run(
                a,
                b.copyless_reshape({b.number_of_elements(), 1}),
                outshape
            );
        }
    }

    AssignableArray vector_dot(
            const Array& a,
            const Array& b) {
        ASSERT2(a.ndim() == 1 && b.ndim() == 1,
            utils::MS() << "VectorDot must be called on a pair of vectors, but got a.ndim()="
                        << a.ndim() << " and b.ndim()=" << b.ndim() << " tensors.");
        ASSERT2(a.bshape()[0] == b.bshape()[0] || (a.bshape()[0] == -1) || (b.bshape()[0] == -1),
            utils::MS() << "shapes " << a.shape() << " and " << b.shape() << " not aligned: "
                        << a.shape()[0] << " (dim 0) != " << b.shape()[0] << " (dim 0)");
        return LazyReshapedGemm::run(
            a.copyless_reshape({1, a.number_of_elements()}),
            b.copyless_reshape({b.number_of_elements(), 1}),
            {}
        );
    }

    AssignableArray _tensordot_as_dot(
            const Array& a,
            const Array& b,
            const std::vector<int>& a_reduce_axes,
            const std::vector<int>& b_reduce_axes,
            DOT_TYPE_T dot_type,
            bool batched) {

        ASSERT2(a_reduce_axes.size() == b_reduce_axes.size(),
            utils::MS() << "must have as many reduction axes for a than b "
                        << "(got a_reduce_axes=" << a_reduce_axes << " and "
                        << "b_reduce_axes=" << b_reduce_axes << ")."
        );

        check_tensordot_reduce_axes(a, 'a', a_reduce_axes, batched);
        check_tensordot_reduce_axes(b, 'b', b_reduce_axes, batched);

        auto a_other_axes = get_tensordot_dimshuffle_axes(
            a.ndim(), a_reduce_axes, batched
        );
        auto b_other_axes = get_tensordot_dimshuffle_axes(
            b.ndim(), b_reduce_axes, batched
        );

        a_other_axes.insert(a_other_axes.end(), a_reduce_axes.begin(), a_reduce_axes.end());
        b_other_axes.insert(b_other_axes.begin(), b_reduce_axes.begin(), b_reduce_axes.end());

        if (batched) {
            a_other_axes.insert(a_other_axes.begin(), 0);
            b_other_axes.insert(b_other_axes.begin(), 0);
        }

        // now call dimshuffle
        auto a_shuffled = a.dimshuffle(a_other_axes);
        auto b_shuffled = b.dimshuffle(b_other_axes);

        return _tensordot_as_dot(
            a_shuffled,
            b_shuffled,
            a_reduce_axes.size(),
            dot_type,
            batched
        );
    }

    // TODO (szymon): allow for scaling with Binary expression + template redundancy trick!
    AssignableArray dot(const Array& a, const Array& b) {
        auto a_ndim = a.ndim();
        auto b_ndim = b.ndim();

        if (a_ndim == 0 || b_ndim == 0) {
            if (a_ndim == 0) {
                return a.broadcast_scalar_to_ndim(b_ndim) * b;
            } else {
                return a * b.broadcast_scalar_to_ndim(a_ndim);
            }
        } else if (a_ndim > 2 || b_ndim > 2) {
            return _tensordot_as_dot(
                a,
                b,
                {a_ndim - 1},
                {std::max(0, b_ndim - 2)},
                DOT_TYPE_2D_T,
                false
            );
        } else if (a_ndim == 1 && b_ndim == 1) {
            return vector_dot(a, b);
        } else if (a_ndim == 2 && b_ndim == 2) {
            return LazyReshapedGemm::run(a, b, {a.shape()[0], b.shape()[1]});
        } else {
            return matrix_vector_dot(a, b);
        }
    }

    AssignableArray tensordot(const Array& a, const Array& b, const int& axis) {
        return _tensordot_as_dot(
            a, b, axis, /*dot=*/DOT_TYPE_2D_T, /*batched=*/false
        );
    }

    AssignableArray tensordot(const Array& a, const Array& b, const std::vector<int>& a_reduce_axes, const std::vector<int>& b_reduce_axes) {
        return _tensordot_as_dot(
            a, b, a_reduce_axes, b_reduce_axes, /*dot=*/DOT_TYPE_2D_T, /*batched=*/false
        );
    }
}