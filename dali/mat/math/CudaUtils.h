#ifndef DALI_MAT_MATH_CUDAUTILS_H
#define DALI_MAT_MATH_CUDAUTILS_H
#ifdef DALI_USE_CUDA

#include <mshadow/tensor.h>

#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>

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



#endif
#endif
