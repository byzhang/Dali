#ifndef DALI_MAT_MATH___MAT_MACROS___H
#define DALI_MAT_MATH___MAT_MACROS___H

#include <mshadow/tensor.h>

#define GET_MAT(X) (X).w()->w
#define GET_GRAD(X) (X).dw()->dw
#define GRAD(X) if (!(X).constant) GET_GRAD(X)

template<typename Device, int ndims, typename R, typename R2>
inline void tensor_fill(mshadow::Tensor<Device, ndims, R>& ts, R2 filler) {
    mshadow::MapExp<mshadow::sv::saveto>(&ts, mshadow::expr::ScalarExp<R>((R)filler));
}

#endif
