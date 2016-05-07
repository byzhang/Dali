#include "dali/math/LazyTensor.h"
#include <mshadow/extension/unpack_patch2col.h>
#include <mshadow/extension/pack_col2patch.h>

#ifdef DALI_USE_CUDA
template<typename TA, typename TB, typename DType, int dimension, int dstdim, int ta>
inline LazyTensor<mshadow::expr::PackColToPatchXExp<TA, DType, dstdim>,
                  mshadow::expr::PackColToPatchXExp<TB, DType, dstdim>, DType, dstdim, ta> pack_col2patch(
            const LazyTensor<TA, TB, DType, dimension, ta> &exp,
            mshadow::Shape<dstdim> imshape,
            int kernel_height,
            int kernel_width,
            int kernel_stride) {
    return LazyTensor<mshadow::expr::PackColToPatchXExp<TA, DType, dstdim>,
                      mshadow::expr::PackColToPatchXExp<TB, DType, dstdim>, DType, dstdim, ta>(
        mshadow::expr::PackColToPatchXExp<TA, DType, dstdim>(
            exp.left,
            imshape,
            kernel_height,
            kernel_width,
            kernel_stride
        ),
        mshadow::expr::PackColToPatchXExp<TB, DType, dstdim>(
            exp.right,
            imshape,
            kernel_height,
            kernel_width,
            kernel_stride
        ),
        exp.dependent_tensors
    );
}
#else
template<typename TA, typename DType, int dimension, int dstdim, int ta>
inline LazyTensor<mshadow::expr::PackColToPatchXExp<TA, DType, dstdim>, DType, dstdim, ta> pack_col2patch(
        const LazyTensor<TA, DType, dimension, ta> &exp,
        mshadow::Shape<dstdim> imshape,
        int kernel_height,
        int kernel_width,
        int kernel_stride) {
    return LazyTensor<mshadow::expr::PackColToPatchXExp<TA, DType, dstdim>, DType, dstdim, ta>(
        mshadow::expr::PackColToPatchXExp<TA, DType, dstdim>(
            exp.left,
            imshape,
            kernel_height,
            kernel_width,
            kernel_stride
        ),
        exp.dependent_tensors
    );
}
#endif

#ifdef DALI_USE_CUDA
template<typename TA, typename TB, typename DType, int dimension, int ta>
inline LazyTensor<mshadow::expr::UnpackPatchToColXExp<TA, DType, dimension>,
                  mshadow::expr::UnpackPatchToColXExp<TB, DType, dimension>, DType, 2, ta> unpack_patch2col(
        const LazyTensor<TA, TB, DType, dimension, ta> &exp,
        const int& kernel_height,
        const int& kernel_width,
        const int& kernel_stride) {
    return LazyTensor<mshadow::expr::UnpackPatchToColXExp<TA, DType, dimension>,
                      mshadow::expr::UnpackPatchToColXExp<TB, DType, dimension>, DType, 2, ta>(
        mshadow::expr::UnpackPatchToColXExp<TA, DType, dimension>(
            exp.left,
            kernel_height,
            kernel_width,
            kernel_stride
        ),
        mshadow::expr::UnpackPatchToColXExp<TB, DType, dimension>(
            exp.right,
            kernel_height,
            kernel_width,
            kernel_stride
        ),
        exp.dependent_tensors
    );
}
#else
template<typename TA, typename DType, int dimension, int ta>
inline LazyTensor<mshadow::expr::UnpackPatchToColXExp<TA, DType, dimension>, DType, 2, ta> unpack_patch2col(
        const LazyTensor<TA, DType, dimension, ta> &exp,
        const int& kernel_height,
        const int& kernel_width,
        const int& kernel_stride) {
    return LazyTensor<mshadow::expr::UnpackPatchToColXExp<TA, DType, dimension>, DType, 2, ta>(
        mshadow::expr::UnpackPatchToColXExp<TA, DType, dimension>(
            exp.left,
            kernel_height,
            kernel_width,
            kernel_stride
        ),
        exp.dependent_tensors
    );
}
#endif
