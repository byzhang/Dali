#ifndef DALI_MAT_MATH_TENSOROPS_H
#define DALI_MAT_MATH_TENSOROPS_H

#include <mshadow/tensor.h>

namespace TensorOps {
    template<typename tensor_t>
    void add(tensor_t& a, tensor_t& b, tensor_t& out) {
        out = a+b;
    }
};





#endif
