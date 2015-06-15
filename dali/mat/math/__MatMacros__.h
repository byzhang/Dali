#ifndef DALI_MAT_MATH___MAT_MACROS___H
#define DALI_MAT_MATH___MAT_MACROS___H

#include <functional>
#include <mshadow/tensor.h>

#include "dali/mat/math/SynchronizedTensor.h"

#define MAT(matrix) ((matrix).w()->w)
#define GRAD(X) ((X).dw()->dw)

#define SAFE_GRAD(X) if (!(X).constant()) GRAD(X)

#ifdef DALI_USE_CUDA

    #define DALI_FUNCTION_1_MUT(f, st, ...)              \
            (should_compute_on_gpu({std::ref(st)}) ?     \
                f((st).mutable_gpu_data(), ##__VA_ARGS__)  \
            :                                            \
                f((st).mutable_cpu_data(), ##__VA_ARGS__)  \
            )

    #define DALI_FUNCTION_1(f, st, ...)               \
            (should_compute_on_gpu({std::ref(st)}) ?  \
                f((st).gpu_data(), ##__VA_ARGS__)       \
            :                                         \
                f((st).cpu_data(), ##__VA_ARGS__)       \
            )

    #define DALI_FUNCTION_2_MUT(f, st1, st2, ...)                                   \
            (should_compute_on_gpu({std::ref(st1), std::ref(st2)}) ?                \
                f((st1).mutable_gpu_data(), (st2).mutable_gpu_data(), ##__VA_ARGS__)  \
            :                                                                       \
                f((st1).mutable_cpu_data(), (st2).mutable_cpu_data(), ##__VA_ARGS__)  \
            )

    #define DALI_FUNCTION_2(f, st1, st2, ...)                         \
            (should_compute_on_gpu({std::ref(st1), std::ref(st2)}) ?  \
                f((st1).gpu_data(), (st2).gpu_data(), ##__VA_ARGS__)    \
            :                                                         \
                f((st1).cpu_data(), (st2).cpu_data(), ##__VA_ARGS__)     \
            )

    #define DALI_FUNCTION_3_MUT(f, st1, st2, st3, ...)                                                          \
            (should_compute_on_gpu({std::ref(st1), std::ref(st2),  std::ref(st3)}) ?                            \
                f((st1).mutable_gpu_data(), (st2).mutable_gpu_data(), (st3).mutable_gpu_data(), ##__VA_ARGS__)  \
            :                                                                                                   \
                f((st1).mutable_cpu_data(), (st2).mutable_cpu_data(), (st3).mutable_cpu_data(), ##__VA_ARGS__)  \
            )


    #define DALI_FUNCTION_3(f, st1, st2, st3, ...)                                      \
            (should_compute_on_gpu({std::ref(st1), std::ref(st2),  std::ref(st3)}) ?    \
                f((st1).gpu_data(), (st2).gpu_data(), (st3).gpu_data(), ##__VA_ARGS__)  \
            :                                                                           \
                f((st1).cpu_data(), (st2).cpu_data(), (st3).cpu_data(), ##__VA_ARGS__)  \
            )

#else
    #define DALI_FUNCTION_1_MUT(f, st, ...)          \
            f((st).mutable_cpu_data(), ##__VA_ARGS__)

    #define DALI_FUNCTION_1(f, st, ...)      \
            f((st).cpu_data(), ##__VA_ARGS__)

    #define DALI_FUNCTION_2_MUT(f, st1, st2, ...)                                   \
                f((st1).mutable_cpu_data(), (st2).mutable_cpu_data(), ##__VA_ARGS__)  \

    #define DALI_FUNCTION_2(f, st1, st2, ...)                      \
                f((st1).cpu_data(), (st2).cpu_data(), ##__VA_ARGS__)  \


    #define DALI_FUNCTION_3_MUT(f, st1, st2, st3, ...)                                   \
                f((st1).mutable_cpu_data(), (st2).mutable_cpu_data(), (st3).mutable_cpu_data(), ##__VA_ARGS__)  \

    #define DALI_FUNCTION_3(f, st1, st2, st3, ...)                      \
                f((st1).cpu_data(), (st2).cpu_data(), (st3).cpu_data(), ##__VA_ARGS__)  \


#endif

#define TENSOR_TEMPLATE template<typename Device, int dims, typename R>


#endif
