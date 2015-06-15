#ifndef DALI_MAT_MATH_TENSOROPS_H
#define DALI_MAT_MATH_TENSOROPS_H

#include <mshadow/tensor.h>


#ifdef DALI_USE_CUDA

#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>

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
        __host__ __device__ bool operator()(const T &lhs, const T &rhs) const {
            return std::fabs(lhs - rhs) < tol;
        }
    };
}
#endif

/* CUDA UTILS END HERE */


namespace TensorOps {
    using mshadow::gpu;
    using mshadow::cpu;

    template<typename tensor_t>
    void add(tensor_t& a, tensor_t& b, tensor_t& out) {
        out = a+b;
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
        bool equals(const mshadow::Tensor<gpu,ndims,R>& a, const mshadow::Tensor<gpu,ndims,R>& b, int num_elts) {
            return thrust::equal(to_thrust(a), to_thrust(a) + num_elts, to_thrust(b));
        }
    #endif
    template<int ndims, typename R>
    bool equals(const mshadow::Tensor<cpu,ndims,R>& a, const mshadow::Tensor<cpu,ndims,R>& b, int num_elts) {
        return std::equal(a.dptr_, a.dptr_ + num_elts, b.dptr_);
    }


    #ifdef DALI_USE_CUDA
    template<int ndims, typename R>
        bool allclose(const mshadow::Tensor<gpu,ndims,R>& a, const mshadow::Tensor<gpu,ndims,R>& b, int num_elts, R tol) {
            return thrust::equal(to_thrust(a),
                                 to_thrust(a) + num_elts,
                                 to_thrust(b),
                                 near_equal<R>(tol));
        }
    #endif
    template<int ndims, typename R>
    bool allclose(const mshadow::Tensor<cpu,ndims,R>& a, const mshadow::Tensor<cpu,ndims,R>& b, int num_elts, R tol) {
        return std::equal(a.dptr_, a.dptr_ + num_elts, b.dptr_,
                [tol](const R &lhs, const R &rhs) {
                    return std::fabs(lhs - rhs) < tol;
                });
    }

    #ifdef DALI_USE_CUDA
        template<int ndims, typename R>
        R sum(const mshadow::Tensor<gpu,ndims,R>& a, int num_elts) {
            return thrust::reduce(to_thrust(a), to_thrust(a) + num_elts, 0.0, thrust::plus<R>());
        }
    #endif
    template<int ndims, typename R>
    R sum(const mshadow::Tensor<cpu,ndims,R>& a, int num_elts) {
        return std::accumulate(a.dptr_, a.dptr_ + num_elts, 0.0);
    }

    template<typename Device, int ndims, typename R, typename R2>
    void fill(mshadow::Tensor<Device, ndims, R>& ts, R2 filler) {
        mshadow::MapExp<mshadow::sv::saveto>(&ts, mshadow::expr::ScalarExp<R>((R)filler));
    }

};





#endif
