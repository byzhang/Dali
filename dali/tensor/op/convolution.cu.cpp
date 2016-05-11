#include "dali/tensor/op/convolution.h"

#include "dali/tensor/__MatMacros__.h"
#include "dali/math/TensorOps.h"
#include "dali/math/LazyTensor.h"
#include "dali/math/lazy_swapaxis.h"
#include "dali/math/lazy_patch2col.h"
#include "dali/math/TensorConvolution.h"
#include "dali/tensor/op/reshaping.h"

using utils::assert2;
using utils::MS;
using std::make_shared;
using std::shared_ptr;
using std::vector;

namespace matops {
    // Note if kernel is 3D (as in multi kernel)
    // Then result must also be a tensor (first dimension is kernel dimension)
    template<typename R>
    Mat<R> Convolution<R>::conv2d(
            Mat<R> image,
            Mat<R> kernels,
            const std::vector<int>& image_shape,
            const int& kernel_height,
            const int& kernel_width,
            const int& kernel_stride) {
        ASSERT2(image_shape.size() == 4,
            utils::MS() << "image_shape argument to patch2col must be a size "
                        << "4 vector (got " << image_shape.size() << ")"
        );
        ASSERT2(kernels.dims(1) == kernel_height * kernel_width * image_shape[1],
            utils::MS() << "kernels shape must be of dimension num_filters * filter_size, "
                        << "but inferred filter_size to be kernel_height * kernel_width * "
                        << "image_shape[1] = " << (kernel_height * kernel_width * image_shape[1])
                        << ", while kernels has second dimension = " << kernels.dims(1)
        );

        auto patched_image = Reshaping<R>::patch2col_no_grad(
            image,
            image_shape,
            kernel_height,
            kernel_width,
            kernel_stride
        );
        Mat<R> out_wrong_arrangement(
            kernels.dims(0),
            patched_image.dims(1),
            weights<R>::empty()
        );
        // convolve kernels with extracted image patches (now viewed
        // as columns)

        MAT(out_wrong_arrangement) = dot(
            MAT(kernels).wrapper(),
            MAT(patched_image).wrapper()
        );

        int oheight  = (image_shape[2] - kernel_height)/kernel_stride + 1;
        int owidth   = (image_shape[3] - kernel_width)/kernel_stride + 1;
        int nbatch   = image_shape[0];
        int nfilters = kernels.dims(0);

        Mat<R> out(
            nbatch,
            nfilters * oheight * owidth,
            weights<R>::empty()
        );
        // present activations back in their 4d shape:
        MAT(out).reshape(mshadow::Shape4(nbatch, nfilters, oheight, owidth)) = (
            swapaxis<1,0>(
                MAT(out_wrong_arrangement).reshape(
                    mshadow::Shape4(nfilters, nbatch, oheight, owidth)
                ).wrapper()
            )
        );

        // during backprop we do not keep the extracted patches
        // but instead keep the original image (because of the aliasing
        // present in patched_image we can save memory by recomputing patch2col
        // during backprop)
        if (graph::backprop_enabled() && (!image.constant || !kernels.constant))
            graph::emplace_back([
                    out,
                    image,
                    kernels,
                    image_shape,
                    nfilters,
                    nbatch,
                    oheight,
                    owidth,
                    kernel_height,
                    kernel_width,
                    kernel_stride] () {
                // run patching once more
                auto patched_image = Reshaping<R>::patch2col_no_grad(
                    image,
                    image_shape,
                    kernel_height,
                    kernel_width,
                    kernel_stride
                );

                // return activations to a 2d shape (from 4D above)
                TensorInternal<R, 2> activations_2d(
                    mshadow::Shape2(
                        nfilters, nbatch * oheight * owidth
                    )
                );

                activations_2d.reshape(
                    mshadow::Shape4(nfilters, nbatch, oheight, owidth)
                ) = swapaxis<1,0>(
                    GRAD(out).reshape(
                        mshadow::Shape4(nbatch, nfilters, oheight, owidth)
                    ).wrapper()
                );
                // backprop dot-product
                if (!kernels.constant) {
                    GRAD(kernels) = dot(
                        activations_2d.wrapper(),
                        MAT(patched_image).wrapper().T()
                    );
                }
                if (!image.constant) {
                    // backprop image gradients into
                    // the patch-columns
                    GRAD(patched_image) = dot(
                        MAT(kernels).wrapper().T(),
                        activations_2d.wrapper()
                    );

                    auto image_4dshape = mshadow::Shape4(
                        image_shape[0],
                        image_shape[1],
                        image_shape[2],
                        image_shape[3]
                    );
                    // re-pack the patched columns
                    // into the original image
                    GRAD(image).reshape(image_4dshape) += pack_col2patch(
                        GRAD(patched_image).wrapper(),
                        image_4dshape,
                        kernel_height,
                        kernel_width,
                        kernel_stride
                    );
                }
            });

        return out;
    }

    template<typename R>
    Mat<R> Convolution<R>::circular_convolution(Mat<R> matrix, Mat<R> shift) {
        assert2(matrix.dims(0) == shift.dims(0) && matrix.dims(1) == shift.dims(1),
                "Cannot perform circular convolution: matrix and shift must be of the same size.");
        auto out = Mat<R>::zeros_like(matrix);
        TensorOps::circular_convolution(MAT(out), MAT(matrix), MAT(shift));
        if (graph::backprop_enabled()) {
            graph::emplace_back([out, matrix, shift]() mutable {
                if (!matrix.constant) {
                    TensorOps::circular_convolution(GRAD(matrix), GRAD(out), MAT(shift));
                }
                if (!shift.constant) {
                    TensorOps::circular_convolution(GRAD(shift), MAT(matrix), GRAD(out));
                }
            });
        }
        return out;
    }

    template class Convolution<float>;
    template class Convolution<double>;
    template class Convolution<int>;

}
