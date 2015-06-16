#ifndef DALI_MAT_MATH_TENSOROPS_H
#define DALI_MAT_MATH_TENSOROPS_H

#include <mshadow/tensor.h>
#include <random>
#include <functional>


#ifdef DALI_USE_CUDA

#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>


/* CUDA UTILS START HERE */

namespace TensorOps {
    template<typename R, int ndims>
    thrust::device_ptr<R> to_thrust(const mshadow::Tensor<mshadow::gpu, ndims, R>& tg) {
        auto dev_ptr = thrust::device_pointer_cast(tg.dptr_);
        return dev_ptr;
    }

    template<typename T>
    struct near_equal {
        T tol;
        near_equal(T _tol) : tol(_tol) {}
        __host__ __device__ bool operator()(const T& lhs, const T& rhs) const {
            return std::fabs(lhs - rhs) < tol;
        }
    };
}
#endif

enum OptionalTranspose {
    TRANSPOSE,
    NO_TRANSPOSE
};

enum OperationType {
    OVERWRITE,
    ADD_TO_EXISTING
};

/* CUDA UTILS END HERE */

#define DALI_ASSIGN(condition, out, expr) if ((op) == OVERWRITE) { out = (expr); } else {  out += (expr);  }

namespace TensorOps {
    using mshadow::gpu;
    using mshadow::cpu;

    #ifdef DALI_USE_CUDA
    template<int ndims, typename R>
    bool equals(const mshadow::Tensor<gpu, ndims, R>& a, const mshadow::Tensor<gpu, ndims, R>& b, int num_elts) {
        return thrust::equal(to_thrust(a), to_thrust(a) + num_elts, to_thrust(b));
    }
    #endif
    template<int ndims, typename R>
    bool equals(const mshadow::Tensor<cpu, ndims, R>& a, const mshadow::Tensor<cpu, ndims, R>& b, int num_elts) {
        return std::equal(a.dptr_, a.dptr_ + num_elts, b.dptr_);
    }


    #ifdef DALI_USE_CUDA
    template<int ndims, typename R>
    bool allclose(const mshadow::Tensor<gpu, ndims, R>& a, const mshadow::Tensor<gpu, ndims, R>& b, int num_elts, R tol) {
        return thrust::equal(to_thrust(a),
                             to_thrust(a) + num_elts,
                             to_thrust(b),
                             near_equal<R>(tol));
    }
    #endif
    template<int ndims, typename R>
    bool allclose(const mshadow::Tensor<cpu, ndims, R>& a, const mshadow::Tensor<cpu, ndims, R>& b, int num_elts, R tol) {
        return std::equal(a.dptr_, a.dptr_ + num_elts, b.dptr_,
        [tol](const R & lhs, const R & rhs) {
            return std::fabs(lhs - rhs) < tol;
        });
    }

    template<typename tensor_t>
    void add(tensor_t& a, tensor_t& b, tensor_t& out, OperationType op) {
        DALI_ASSIGN(op, out, a + b)
    }

    template<typename Device, int ndims, typename R>
    void add_inplace(mshadow::Tensor<Device,ndims,R> a, int num_elts, R summand) {
        a += mshadow::expr::scalar<R>(summand);
    }

    template<typename tensor_t>
    void add_inplace(tensor_t& a, tensor_t& dest) {
        dest += a;
    }

    #ifdef DALI_USE_CUDA
    template<int ndims, typename R>
    R sum(const mshadow::Tensor<gpu, ndims, R>& a, int num_elts) {
        return thrust::reduce(to_thrust(a), to_thrust(a) + num_elts, 0.0, thrust::plus<R>());
    }
    #endif
    template<int ndims, typename R>
    R sum(const mshadow::Tensor<cpu, ndims, R>& a, int num_elts) {
        return std::accumulate(a.dptr_, a.dptr_ + num_elts, 0.0);
    }

    template<typename tensor_t, typename R>
    void fill(tensor_t& ts, R filler) {
        mshadow::MapExp<mshadow::sv::saveto>(&ts,
            mshadow::expr::ScalarExp<R>(filler)
        );
    }

    template<int ndims, typename R>
    void eye(mshadow::Tensor<cpu,ndims,R>& tc, R diag) {
        if (tc.shape_[0] != tc.shape_[1]) {
            throw std::runtime_error("Identity initialization must be called on a square matrix.");
        }
        fill(tc, (R)0.0);
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
    void eye(mshadow::Tensor<gpu,ndims,R>& tg, R diag) {
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

    namespace random {
        template<typename Device, int ndims, typename R, template <typename,int,typename> class tensor_t>
        void uniform(tensor_t<Device, ndims, R>& t, R lower, R upper) {
            std::random_device rd;
            mshadow::Random<Device, R> generator((int)rd());
            generator.SampleUniform(&t, lower, upper);
        }

        template<typename Device, int ndims, typename R, template <typename,int,typename> class tensor_t>
        void gaussian(tensor_t<Device, ndims, R>& t, R mean, R std) {
            std::random_device rd;
            mshadow::Random<Device, R> generator((int)rd());
            generator.SampleGaussian(&t, mean, std);
        }
    };

    template<typename tensor_t>
    void dot(tensor_t& a, tensor_t& b, tensor_t& out, OptionalTranspose t_a, OptionalTranspose t_b, OperationType op) {
        bool transpose_a = (t_a == TRANSPOSE);
        bool transpose_b = (t_b == TRANSPOSE);
        if (!transpose_a && !transpose_b) {
            DALI_ASSIGN(op, out, mshadow::expr::dot(a,b))
        } else if (transpose_a && !transpose_b) {
            DALI_ASSIGN(op, out, mshadow::expr::dot(a.T(),b))
        } else if (!transpose_a && transpose_b) {
            DALI_ASSIGN(op, out, mshadow::expr::dot(a,b.T()))
        } else if (transpose_a && transpose_b) {
            DALI_ASSIGN(op, out, mshadow::expr::dot(a.T(),b.T()))
        }
    }

    template<typename tensor_t>
    void eltmul(tensor_t& a, tensor_t& b, tensor_t& out, OperationType op) {
        DALI_ASSIGN(op, out, a * b)
    }
};

#endif
