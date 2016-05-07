#include "dali/math/LazyTensor.h"
#include <mshadow/extension/swapaxis.h>

#ifdef DALI_USE_CUDA
template<int a1, int a2, typename TA, typename TB, typename DType, int dimension, int ta>
inline auto swapaxis(const LazyTensor<TA, TB, DType, dimension, ta> &exp)
    -> LazyTensor<decltype(mshadow::expr::swapaxis<a1,a2>(exp.left)), decltype(mshadow::expr::swapaxis<a1,a2>(exp.right)), DType, dimension, ta> {
        return LazyTensor<decltype(mshadow::expr::swapaxis<a1,a2>(exp.left)),
                          decltype(mshadow::expr::swapaxis<a1,a2>(exp.right)),
                          DType, dimension,
                          ta>(
                              mshadow::expr::swapaxis<a1,a2>(exp.left),
                              mshadow::expr::swapaxis<a1,a2>(exp.right),
                              exp.dependent_tensors
                );

    }
#else
template<int a1, int a2, typename TA, typename DType, int dimension, int ta>
inline auto swapaxis(const LazyTensor<TA, DType, dimension, ta> &exp)
    -> LazyTensor<decltype(mshadow::expr::swapaxis<a1,a2>(exp.left)), DType, dimension, ta> {
        return LazyTensor<decltype(mshadow::expr::swapaxis<a1,a2>(exp.left)),
                          DType, dimension,
                          ta>(
            mshadow::expr::swapaxis<a1,a2>(exp.left),
            exp.dependent_tensors
        );
    }
#endif
