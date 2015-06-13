#ifndef DALI_MAT_MATH___MAT_MACROS___H
#define DALI_MAT_MATH___MAT_MACROS___H

#include <mshadow/tensor.h>

#include "dali/mat/math/SynchronizedTensor.h"



#define DALI_MAT_ST(matrix) ((matrix).w()->w)
#define DALI_GRAD_ST(X) ((X).dw()->dw)

#define GRAD(X) if (!(X).constant()) GET_GRAD(X)

#ifdef DALI_USE_CUDA
    #define DALI_EXECUTE_ST_FUNCTION_MUT(st, f, ...)     \
            if ((st).prefers_gpu()) {                    \
                f((st).mutable_gpu_data(), __VA_ARGS__); \
            } else {                                     \
                f((st).mutable_cpu_data(), __VA_ARGS__); \
            }
#else
    #define DALI_EXECUTE_ST_FUNCTION_MUT(st, f, ...) \
            f((st).mutable_cpu_data(), __VA_ARGS__);
#endif

#define TENSOR_TEMPLATE template<typename Device, int dims, typename R>



template<typename Device, int ndims, typename R, typename R2>
void tensor_fill(mshadow::Tensor<Device, ndims, R>& ts, R2 filler) {
    mshadow::MapExp<mshadow::sv::saveto>(&ts, mshadow::expr::ScalarExp<R>((R)filler));
}

template<typename R, typename R2>
inline void tensor_fill(SynchronizedTensor<R>& t, R2 filler) {
    DALI_EXECUTE_ST_FUNCTION_MUT(t, tensor_fill, filler);
}


#endif
