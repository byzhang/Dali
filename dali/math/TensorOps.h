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

    } // namespace reduction

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
}; // namespace TensorOps

#endif
