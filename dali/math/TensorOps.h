#ifndef DALI_MAT_MATH_TENSOROPS_H
#define DALI_MAT_MATH_TENSOROPS_H

#include "dali/config.h"

#include <mshadow/tensor.h>
#include <math.h>
#include <functional>

#include "dali/math/TensorInternal.h"
#include "dali/math/MshadowIntegerOps.h"
#include "dali/math/TensorFunctions.h"
#include "dali/math/TensorAccessor.h"
#include "dali/math/ThrustUtils.h"


/**
Tensor Operations
-----------------

Implementations for GPU and CPU agnostic operations on
Tensors. Using a combination of Thrust and MShadow this
file has implementation for the major operations done
on tensors.

Each namespace covers an area of implementation that's
syntactically sugared for Dali:

random
------

Implements the following random samplers:

* uniform
* gaussian
* bernoulli
* bernoulli_normalized

arg
---

* argmin
* argmax
* argmin row-wise & column-wise
* argmax row-wise & column-wise

comparison
----------

* equals
* allclose

reduction
---------

* sum
* L2_norm

op
--

Kernels for MShadow

**/

/* CUDA UTILS END HERE */
#define DALI_ASSIGN(op, out, expr) if ((op) == OVERWRITE) { out = (expr); } else {  out += (expr);  }


namespace TensorOps {
    using mshadow::gpu;
    using mshadow::cpu;

    namespace comparison {
        #ifdef DALI_USE_CUDA
        template<int ndims, typename R>
        bool equals(const mshadow::Tensor<gpu, ndims, R> a, const mshadow::Tensor<gpu, ndims, R> b, int num_elts) {
            return thrust::equal(to_thrust(a), to_thrust(a) + num_elts, to_thrust(b));
        }
        #endif
        template<int ndims, typename R>
        bool equals(const mshadow::Tensor<cpu, ndims, R> a, const mshadow::Tensor<cpu, ndims, R> b, int num_elts) {
            return std::equal(a.dptr_, a.dptr_ + num_elts, b.dptr_);
        }


        #ifdef DALI_USE_CUDA
        template<int ndims, typename R>
        bool allclose(const mshadow::Tensor<gpu, ndims, R> a, const mshadow::Tensor<gpu, ndims, R> b, int num_elts, R tol) {
            return thrust::equal(to_thrust(a),
                                 to_thrust(a) + num_elts,
                                 to_thrust(b),
                                 near_equal<R>(tol));
        }

        // template<int ndims>
        // bool allclose(const mshadow::Tensor<gpu, ndims, int> a, const mshadow::Tensor<gpu, ndims, int> b, int num_elts, int tol) {
        //     using namespace thrust::placeholders;
        //     return thrust::equal(to_thrust(a),
        //                          to_thrust(a) + num_elts,
        //                          to_thrust(b),
        //                          thrust::abs(_1 - _2) < tol );
        // }
        #endif
        template<int ndims, typename R>
        bool allclose(const mshadow::Tensor<cpu, ndims, R> a, const mshadow::Tensor<cpu, ndims, R> b, int num_elts, R tol) {
            return std::equal(a.dptr_, a.dptr_ + num_elts, b.dptr_,
            [tol](const R & lhs, const R & rhs) {
                return std::fabs(lhs - rhs) < tol;
            });
        }
    }

    namespace reduction {
        #ifdef DALI_USE_CUDA
            template <typename R, int dimension>
            bool is_nan(const mshadow::Tensor<gpu, dimension, R> a, int num_elts) {
                return std::isnan(thrust::reduce(
                    to_thrust(a),
                    to_thrust(a) + num_elts,
                    0.0,
                    thrust::plus<R>()));
            }
        #endif

        template <typename R, int dimension>
        bool is_nan(const mshadow::Tensor<cpu, dimension, R> a, int num_elts) {
            return std::isnan(std::accumulate(a.dptr_, a.dptr_ + num_elts, 0.0));
        }

        #ifdef DALI_USE_CUDA
        template<int ndims, typename R>
        R sum(const mshadow::Tensor<gpu, ndims, R> a, int num_elts) {
            return thrust::reduce(
                to_thrust(a),
                to_thrust(a) + num_elts,
                0.0,
                thrust::plus<R>());
        }
        #endif

        template<int ndims, typename R>
        R sum(const mshadow::Tensor<cpu, ndims, R> a, int num_elts) {
            return std::accumulate(a.dptr_, a.dptr_ + num_elts, 0.0);
        }


        #ifdef DALI_USE_CUDA
        template <typename T>
        struct thrust_square {
            __host__ __device__
            T operator()(const T& x) const {
                return x * x;
            }
        };

        template<int ndims, typename R>
        R L2_norm(const mshadow::Tensor<gpu, ndims, R> a, int num_elts) {
            return std::sqrt(thrust::transform_reduce(to_thrust(a), to_thrust(a) + num_elts, thrust_square<R>(), 0.0, thrust::plus<R>()));
        }


        template<int ndims, typename R>
        R min(const mshadow::Tensor<gpu, ndims, R> a, int num_elts) {
            return thrust::reduce(to_thrust(a), to_thrust(a) + num_elts, std::numeric_limits<R>::infinity(), thrust::minimum<R>());
        }

        template<int ndims, typename R>
        R max(const mshadow::Tensor<gpu, ndims, R> a, int num_elts) {
            return thrust::reduce(to_thrust(a), to_thrust(a) + num_elts, -std::numeric_limits<R>::infinity(), thrust::maximum<R>());
        }
        #endif

        template <typename T>
        struct thrust_square_reduce {
            T operator()(const T& x, const T& y) const {
                return x + (y * y);
            }
        };

        template<int ndims, typename R>
        R L2_norm(const mshadow::Tensor<cpu, ndims, R> a, int num_elts) {
            return std::sqrt(std::accumulate(a.dptr_, a.dptr_ + num_elts, 0.0, thrust_square_reduce<R>()));
        }

        template <typename T>
        struct min_kernel {
            T operator()(const T& x, const T& y) const {
                return x > y ? y : x;
            }
        };

        template <typename T>
        struct max_kernel {
            T operator()(const T& x, const T& y) const {
                return x > y ? x : y;
            }
        };

        template<int ndims, typename R>
        R min(const mshadow::Tensor<cpu, ndims, R> a, int num_elts) {
            return std::accumulate(a.dptr_, a.dptr_ + num_elts, std::numeric_limits<R>::infinity(), min_kernel<R>());
        }

        template<int ndims, typename R>
        R max(const mshadow::Tensor<cpu, ndims, R> a, int num_elts) {
            return std::accumulate(a.dptr_, a.dptr_ + num_elts, -std::numeric_limits<R>::infinity(), max_kernel<R>());
        }

    }

    namespace arg {
        template <typename R, int dimension>
        std::vector<int> argsort(const mshadow::Tensor<cpu, dimension, R> A, int num_elts) {
            std::vector<int> arguments(num_elts);
            // initialize ordering data:
            for (int i=0;i<num_elts;i++)
                arguments[i] = i;

            auto ptr = A.dptr_;
            // sort in increasing order
            std::sort(arguments.begin(), arguments.end(), [&ptr](const int& lhs, const int& rhs) {
                return *(ptr + lhs) < *(ptr + rhs);
            });
            return arguments;
        }

        template<typename Class>
        class Arger {
            public:
                template<typename R, typename Device, int dimension>
                static std::vector<int> apply(const mshadow::Tensor<Device, dimension, R> A, int reduce_dim) {
                    ASSERT2(false, utils::MS() << "Operation not supported for tensor with dimension " << dimension);
                    return {};
                }

                template<typename R>
                static std::vector<int> apply(const mshadow::Tensor<cpu, 1, R> A, int reduce_dim) {
                    return apply_ptr(A.dptr_, A.shape_[0] );
                }

                template<typename R>
                static std::vector<int> apply(const mshadow::Tensor<cpu, 2, R> A, int reduce_dim) {
                    const int num_rows = A.shape_[0];
                    const int num_cols = A.shape_[1];
                    if (reduce_dim == 1) {
                        std::vector<int> arguments(num_rows);
                        for (int i = 0; i < num_rows; i++) {
                            arguments[i] = 0;
                            for (int j = 0; j < num_cols; j++) {
                                if ( Class::compare(*(A.dptr_ + (A.stride_ * i) + j), *(A.dptr_ + (A.stride_ * i) + arguments[i]) )) {
                                    arguments[i] = j;
                                }
                            }
                        }
                        return arguments;
                    } else if (reduce_dim == 0) {
                        std::vector<int> arguments(num_cols);
                        for (int j = 0; j < num_cols; j++) {
                            arguments[j] = 0;
                            for (int i = 0; i < num_rows; i++) {
                                if ( Class::compare(*(A.dptr_ + (A.stride_ * i) + j), *(A.dptr_ + (A.stride_ * arguments[j]) + j) )) {
                                    arguments[j] = i;
                                }
                            }
                        }
                        return arguments;
                    } else {
                        throw std::runtime_error("argmax can only be used with axis equal to 0 or 1.");
                    }
                }

                template<typename R>
                static std::vector<int> apply_ptr(const R* start, const int num_elts) {
                    std::vector<int> offset_row(1, 0);
                    int& offset = offset_row[0];
                    for (int i = 0; i < num_elts; i++) {
                        if (Class::compare(*(start + i), *(start + offset))) {
                            offset = i;
                        }
                    }
                    return offset_row;
                }
        };

        struct Argmax : public Arger<Argmax> {
            template<typename R>
            static MSHADOW_XINLINE bool compare(const R& left, const R& right) {
                return left > right;
            }
#ifdef DALI_USE_CUDA
            template<typename R>
            static thrust::device_ptr<R> thrust_compare(thrust::device_ptr<R> start, thrust::device_ptr<R> end) {
                return std::max_element(start, end);
            }
#endif
        };

        struct Argmin : public Arger<Argmin> {
            template<typename R>
            static MSHADOW_XINLINE bool compare(const R& left, const R& right) {
                return left < right;
            }

#ifdef DALI_USE_CUDA
            template<typename R>
            static thrust::device_ptr<R> thrust_compare(thrust::device_ptr<R> start, thrust::device_ptr<R> end) {
                return std::min_element(start, end);
            }
#endif
        };

        template<typename R, typename Device, int dimension>
        std::vector<int> argmin(const mshadow::Tensor<Device, dimension, R>& A, int reduce_dim) {
            return Argmin::apply(A, reduce_dim);
        }

        template<typename R, typename Device, int dimension>
        std::vector<int> argmax(const mshadow::Tensor<Device, dimension, R>& A, int reduce_dim) {
            return Argmax::apply(A, reduce_dim);
        }

        template<typename R>
        std::vector<int> argmin(const R* ptr, int number_of_elements) {
            return Argmin::apply_ptr(ptr, number_of_elements);
        }

        template<typename R>
        std::vector<int> argmax(const R* ptr, int number_of_elements) {
            return Argmax::apply_ptr(ptr, number_of_elements);
        }

#ifdef DALI_USE_CUDA
        // Convert a linear index to a row index
        template <typename R>
        struct linear_index_to_row_index : public thrust::unary_function<R,R> {
            R cols; // number of columns
            __host__ __device__
            linear_index_to_row_index(R _cols) : cols(_cols) {}
            __host__ __device__
            R operator()(R i) {
                return i / cols;
            }
        };

        // convert a linear index to a row index
        template <typename T>
        struct linear_index_to_col_index: public thrust::unary_function<T, T> {
            T rows; // number of rows
            __host__ __device__ linear_index_to_col_index(T rows) : rows(rows) {}

            __host__ __device__ T operator() (T i) {
                return i % rows;
            }
        };

        #define THRUST_COMP_KERNEL_RESULT_T thrust::tuple<int, R>

        #define THRUST_COMP_KERNEL(compsymbol, kernel_name) \
            template<typename R> \
            struct kernel_name : public thrust::binary_function<THRUST_COMP_KERNEL_RESULT_T ,THRUST_COMP_KERNEL_RESULT_T , THRUST_COMP_KERNEL_RESULT_T > {\
                __host__ __device__ \
                THRUST_COMP_KERNEL_RESULT_T operator()(const THRUST_COMP_KERNEL_RESULT_T & a, const THRUST_COMP_KERNEL_RESULT_T & b) const {\
                    if (thrust::get<1>(a) compsymbol thrust::get<1>(b)){\
                        return a;\
                    } else {\
                        return b;\
                    }\
                }\
            }\

        template <int extraction, typename TA, typename TB>
        struct thrust_extract_arg : public thrust::unary_function<TA,TB> {
            __host__ __device__
            auto operator()(const thrust::tuple<TA, TB>& x) -> decltype(thrust::get<extraction>(x)) const {
                return thrust::get<extraction>(x);
            }
        };

        template <int extraction, typename TA, typename TB, typename T>
        struct thrust_extract_arg_divide : public thrust::unary_function<TA,TB> {
            T divisor;
            thrust_extract_arg_divide(T _divisor) : divisor(_divisor) {}

            __host__ __device__
            auto operator()(const thrust::tuple<TA, TB>& x) -> typename thrust::tuple_element<extraction,thrust::tuple<TA, TB>>::type const {
                return thrust::get<extraction>(x) % divisor;
            }
        };

        THRUST_COMP_KERNEL(<=, argmin_op);
        THRUST_COMP_KERNEL(>=, argmax_op);

        // specialize for kernel
        #define THRUST_KERNEL_ROWWISE_FROM_2D_MSHADOW( kernel_name, fname ) \
            template <typename R> \
            std::vector<int> fname <R, gpu, 2>::apply (const mshadow::Tensor<gpu, 2, R> A, int reduce_dim) { \
                int num_rows = A.shape_[0]; \
                int num_cols = A.shape_[1]; \
                if (reduce_dim == 0) {\
                    /* reduce_by_key for thrust only works for contiguous portions, since */\
                    /* row-major memory breaks this assumption, we cannot do this without */\
                    /* many intermediate steps; we instead transpose & switch reduce dim  */\
                    mshadow::Tensor<gpu, 2, R> A_T( mshadow::Shape2(num_cols, num_rows));\
                    mshadow::AllocSpace(&A_T, false);\
                    A_T = A.T();\
                    auto sorted = fname <R, gpu, 2>::apply ( A_T , 1 );\
                    mshadow::FreeSpace(&A_T);\
                    return sorted;\
                } else if (reduce_dim == 1) {\
                    /* allocate storage for row arguments and indices */ \
                    thrust::device_vector<THRUST_COMP_KERNEL_RESULT_T> row_arguments(num_rows); \
                    thrust::device_vector<int> row_indices(num_rows); \
                    /* compute row arguments by finding argmin values with equal row indices */ \
                    thrust::reduce_by_key(\
                        thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(num_cols)), \
                        thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(num_cols)) + (num_rows * num_cols), \
                        thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<int>(0), to_thrust(A))), \
                        row_indices.begin(),                                \
                        row_arguments.begin(),                              \
                        thrust::equal_to<int>(),                            \
                        kernel_name<R>()                                    \
                    );                                                      \
                    std::vector<int> host_arguments(num_rows);              \
                    thrust::device_vector<int> row_indices2(num_rows);      \
                    thrust::transform(                                      \
                        row_arguments.begin(),                              \
                        row_arguments.end(),                                \
                        row_indices2.begin(),                               \
                        thrust_extract_arg_divide<0, int, R, int>(num_cols) \
                    );\
                    thrust::copy(row_indices2.begin(), row_indices2.end(), host_arguments.begin());\
                    return host_arguments; \
                } else {\
                    throw std::runtime_error(STR(fname) " can only be used with axis equal to 0 or 1.");\
                }\
            }

        THRUST_KERNEL_ROWWISE_FROM_2D_MSHADOW( argmin_op, Argmin )
        THRUST_KERNEL_ROWWISE_FROM_2D_MSHADOW( argmax_op, Argmax )

        template <typename R, int dimension>
        std::vector<int> argsort(const mshadow::Tensor<gpu, dimension, R> A, int num_elts) {
            thrust::device_vector<int> arguments(num_elts);
            thrust::device_vector<R> values(num_elts);
            // initialize ordering data:
            thrust::sequence(arguments.begin(), arguments.end(), 0, 1);
            // copy data for sorting:
            thrust::copy(to_thrust(A), to_thrust(A) + num_elts, values.begin());
            // sort arguments based on the values they correspond to:
            thrust::sort_by_key(
              values.begin(), values.end(), arguments.begin()
            );
            std::vector<int> host_arguments(num_elts);
            thrust::copy(arguments.begin(), arguments.end(), host_arguments.begin());
            return host_arguments;
        }

        // following the advice from this Stackoverflow:
        // http://stackoverflow.com/questions/7709181/finding-the-maximum-element-value-and-its-position-using-cuda-thrust


        #define THRUST_KERNEL_ROWWISE_FROM_1D_GPU_PTR( thrust_op_name, fname) \
            template <typename R> \
            std::vector<int> fname (thrust::device_ptr<R> start, int n_elements) { \
                auto idx = thrust_op_name (start, start + n_elements);\
                return { (int) (&idx[0] - &start[0]) };\
            }

        #define THRUST_KERNEL_ROWWISE_FROM_1D_MSHADOW( thrust_op_name, fname ) \
            template <typename R> \
            std::vector<int> fname <R, gpu, 1>::apply (const mshadow::Tensor<gpu, 1, R> A, int reduce_dim) { \
                auto start = to_thrust(A);\
                return fname ( start, A.shape_[0] );\
            }

        THRUST_KERNEL_ROWWISE_FROM_1D_GPU_PTR( thrust::min_element , argmin )
        THRUST_KERNEL_ROWWISE_FROM_1D_GPU_PTR( thrust::max_element , argmax )

        THRUST_KERNEL_ROWWISE_FROM_1D_MSHADOW( thrust::min_element , Argmin )
        THRUST_KERNEL_ROWWISE_FROM_1D_MSHADOW( thrust::max_element , Argmax )

#endif

    }

    template<int ndims, typename R>
    void eye(mshadow::Tensor<cpu,ndims,R> tc, R diag) {
        if (tc.shape_[0] != tc.shape_[1]) {
            throw std::runtime_error("Identity initialization must be called on a square matrix.");
        }
        tc = 0.0;
        for (int i = 0; i < tc.shape_[0]; i++)
            tc[i][i] = diag;
    }

    #ifdef DALI_USE_CUDA
    template<typename R>
    struct IdentityKernel : public thrust::binary_function<int,R, R> {
        R diag;
        int rows;
        IdentityKernel(R _diag, int _rows) : diag(_diag), rows(_rows) {}
        __host__ __device__ R operator()(const int& offset, const R& b) const {
            return (offset / rows) == (offset % rows) ? diag : 0.0;
        }
    };

    template<int ndims, typename R>
    void eye(mshadow::Tensor<gpu,ndims,R> tg, R diag) {
        if (tg.shape_[0] != tg.shape_[1]) {
            throw std::runtime_error("Identity initialization must be called on a square matrix.");
        }
        int num_elts = tg.shape_[0] * tg.shape_[1];
        int num_rows = tg.shape_[0];
        // counting iterators define a sequence [0, 8)
        thrust::counting_iterator<int> first(0);
        thrust::counting_iterator<int> last = first + num_elts;

        thrust::transform(
            first,
            last,
            to_thrust(tg),
            to_thrust(tg),
            IdentityKernel<R>(diag, num_rows)
        );
    }
    #endif
};

#endif
