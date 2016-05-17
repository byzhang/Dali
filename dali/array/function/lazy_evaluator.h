#ifndef DALI_ARRAY_FUNCTION_LAZY_EVALUATOR_H
#define DALI_ARRAY_FUNCTION_LAZY_EVALUATOR_H

#include "dali/array/debug.h"
#include "dali/array/dtype.h"
#include "dali/array/function/args/reduce_over_lazy_expr.h"
#include "dali/array/function/function.h"
#include "dali/array/function/lazy_function.h"
#include "dali/array/function/operator.h"
#include "dali/array/function/typed_array.h"


////////////////////////////////////////////////////////////////////////////////
//                             LAZY_EVALUATOR                                 //
////////////////////////////////////////////////////////////////////////////////

template<class LazyExpr>
struct LazyEvaluator : public Function<LazyEvaluator<LazyExpr>, Array, LazyExpr> {

    static std::vector<int> deduce_output_bshape(const LazyExpr& expr) {
        return expr.bshape();
    }

    static DType deduce_output_dtype(const LazyExpr& expr) {
        return expr.dtype();
    }

    static memory::Device deduce_output_device(const LazyExpr& expr) {
        auto res = ReduceOverLazyExpr<DeviceReducer>::reduce(expr);
        return res;
    }

    static memory::Device deduce_computation_device(const Array& out, const LazyExpr& expr) {
        return ReduceOverLazyExpr<DeviceReducer>::reduce(out, expr);
    }

    static DType deduce_computation_dtype(const Array& out, const LazyExpr& expr) {
        ASSERT2(out.dtype() == expr.dtype(),
            utils::MS() << "Output type (" << dtype_to_name(out.dtype())
                        << ") and expression type (" << dtype_to_name(expr.dtype()) << ") differ");
        return out.dtype();
    }

    template<OPERATOR_T operator_t, int devT, typename T>
    void typed_eval(TypedArray<devT,T> out, const LazyExpr& expr) {
        debug::lazy_evaluation_callback.activate(out.array);

        // out.array.shape() is passed to MshadowWrapper as final destination
        // shape this means that all the input arguments will be broadcasted
        // to fit out.array.shape(). Here we are assuming that out.array.shape()
        // is not broadcasted, so when the computation actually happens
        // the shape is already fully known every step of the way.

        operator_assign<operator_t, LazyExpr::evaluation_dim>(
            out,
            MshadowWrapper<devT,T,decltype(expr)>::wrap(expr, out.device, out.array.shape())
        );
    }
};

#include "dali/array/lazy/base_lazy_axis_reducer.h"

namespace internal {
    template<typename ExprT>
    struct NonRecursiveLazySumAxis : public BaseLazyAxisReducer<LazyFunctionNonRecusive,NonRecursiveLazySumAxis<ExprT>, ExprT, mshadow::red::sum, false> {
        using BaseLazyAxisReducer<LazyFunctionNonRecusive,NonRecursiveLazySumAxis<ExprT>, ExprT, mshadow::red::sum, false>::BaseLazyAxisReducer;
    };
}  // namespace internal

namespace internal {
    int requires_reduction(const Array& output, const std::vector<int>& in_bshape);
}

namespace lazy {
    template<typename Class, typename... Args>
    AssignableArray eval(const LazyFunctionNonRecusive<Class, Args...>& expr) {
        return LazyEvaluator<Class>::run(expr.self());
    }

    template<typename Class, typename... Args>
    AssignableArray eval(const LazyFunction<Class, Args...>& expr) {
        auto this_self   = expr.self();
        auto this_bshape = expr.bshape();

        return AssignableArray([this_self, this_bshape](Array& out, const OPERATOR_T& operator_t) {
            int reduction_dimension = -1;
            if (operator_t == OPERATOR_T_LSE) {
                reduction_dimension = internal::requires_reduction(out, this_bshape);
            }
            if (reduction_dimension != -1) {
                auto reduced_expr = internal::NonRecursiveLazySumAxis<decltype(this_self)>(
                        this_self,
                        /*axis=*/reduction_dimension,
                        /*keepdims=*/true);
                auto computation_with_reduce = lazy::eval(reduced_expr);
                computation_with_reduce.assign_to(out, operator_t);
            } else {
                LazyEvaluator<Class>::run(this_self).assign_to(out, operator_t);
            }
        });
    }

}  // namespace lazy

#endif
