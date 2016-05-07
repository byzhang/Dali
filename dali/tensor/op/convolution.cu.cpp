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
                        mshadow::Shape4(nfilters, nbatch, oheight, owidth)
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
    Mat<R> Convolution<R>::conv1d(Mat<R> image, Mat<R> kernel) {
        auto kerns = vector<Mat<R>>({kernel});
        return Convolution<R>::conv1d(image, kerns);
    }

    template<typename R>
    Mat<R> Convolution<R>::conv1d(Mat<R> image, Mat<R> kernel, bool pad) {
        auto kerns = vector<Mat<R>>({kernel});
        return Convolution<R>::conv1d(image, kerns, pad);
    }

    // Here multiple kernels are allowable
    template<typename R>
    Mat<R> Convolution<R>::conv1d(Mat<R> image, const vector<Mat<R>>& kernels) {
        // assert2(kernels.size() > 0, "Must pass at least 1 kernel to conv1d.");
        // int kern_col_size = kernels[0].dims(1);
        // for (auto& kernel : kernels) {
        //     assert2(image.dims(0) == kernel.dims(0),
        //         MS() << "Kernel's first dimension (" << kernel.dims(0)
        //              << ") must be equal than or equal to argument's first dimension ("
        //              << image.dims(0));
        //     assert2(image.dims(1) >= kernel.dims(1),
        //         MS() << "Kernel's second dimension (" << kernel.dims(1)
        //              << ") must be smaller than or equal to argument's first dimenion ("
        //              << image.dims(1));
        //     assert2(kern_col_size == kernel.dims(1),
        //         MS() << "All Kernel's second dimension (" << kernel.dims(1)
        //              << ") must be equal");
        // }
        // auto out = Mat<R>(
        //     kernels.size(), // 1d convolution only holds one row
        //     image.dims(1) - kern_col_size + 1, // as many times as the kernel fits
        //     false // fill zeros
        // );
        // auto& out_mat = GET_MAT(out);
        // auto& image_mat = GET_MAT(image);
        // int col=0,
        //     KSizeX = image.dims(0),
        //     SizeX  = image.dims(0),
        //     SizeY  = image.dims(1);
        // vector<R> kernel_sums;
        // kernel_sums.reserve(kernels.size());
        // std::transform(kernels.begin(), kernels.end(), std::back_inserter(kernel_sums), [](const Mat<R>& kern) {
        //     return GET_MAT(kern).sum();
        // });

        // for ( col = 0; col < out.dims(1); col++ ) {
        //     for (int i = 0; i < kernels.size();i++) {
        //         out_mat(i,col) = (image_mat.block(0, col, KSizeX, kern_col_size).array() * GET_MAT(kernels[i]).array()).sum() / kernel_sums[i];
        //     }
        // }

        // if (graph::backprop_enabled()) {
        //     graph::emplace_back([image, kernels, out, kernel_sums, kern_col_size](){
        //         auto& image_mat = GET_MAT(image);
        //         int col=0,
        //             KSizeX = image.dims(0),
        //             SizeX  = image.dims(0),
        //             SizeY  = image.dims(1);
        //         bool grad_image = !image.constant;
        //         auto& out_grad = GET_GRAD(out);
        //         auto& out_weight = GET_MAT(out);
        //         std::shared_ptr<Eigen::Matrix<R, Eigen::Dynamic, 1>> surplus;
        //         bool computed_surplus = false;
        //         for (int i=0; i < kernels.size();i++) {
        //             if (!kernels[i].constant) {
        //                 if (!computed_surplus) {
        //                     surplus = make_shared<Eigen::Matrix<R, Eigen::Dynamic, 1>>((out_weight.array() * out_grad.array()).rowwise().sum());
        //                     computed_surplus = true;
        //                 }
        //                 GET_GRAD(kernels[i]).array() -= (*surplus)(i,0) / kernel_sums[i];
        //             }
        //         }
        //         for ( col = 0; col < out.dims(1); col++ ) {
        //             if (grad_image) {
        //                 for (int i=0; i < kernels.size();i++) {
        //                     GET_GRAD(image).block(0, col, KSizeX, kern_col_size).noalias() += GET_MAT(kernels[i]) * (out_grad(i, col) / kernel_sums[i]);
        //                 }
        //             }
        //             for (int i=0; i < kernels.size();i++) {
        //                 if (!kernels[i].constant) {
        //                     GET_GRAD(kernels[i]).noalias() += (image_mat.block(0, col, KSizeX, kern_col_size).array() * (out_grad(i, col) / (kernel_sums[i]))).matrix();
        //                 }
        //             }
        //         }
        //     });
        // }
        // return out;
        return Mat<R>(1,1);
    }

    // Here multiple kernels are allowable (but only the overlap between them and image is used)
    template<typename R>
    Mat<R> Convolution<R>::conv1d(Mat<R> image, const vector<Mat<R>>& kernels, bool pad) {
        // if (!pad) {
        //     return conv1d(image, kernels);
        // }
        // assert2(kernels.size() > 0, "Must pass at least 1 kernel to conv1d.");
        // int kern_col_size = kernels[0].dims(1);
        // for (auto& kernel : kernels) {
        //     assert2(image.dims(0) <= kernel.dims(0),
        //         MS() << "Kernel's first dimension (" << kernel.dims(0)
        //              << ") must be greater than or equal to argument's first dimension ("
        //              << image.dims(0));
        //     assert2(kern_col_size == kernel.dims(1),
        //         MS() << "All Kernel's second dimension (" << kernel.dims(1)
        //              << ") must be equal");
        // }
        // if (image.dims(0) == kernels[0].dims(0) && kern_col_size <= image.dims(1)) {
        //     return conv1d(image, kernels);
        // }

        // auto out = Mat<R>(
        //     kernels.size(), // 1d convolution only holds one row
        //     1,              // kernels are larger than "image"
        //     false // fill zeros
        // );

        // auto& out_mat = GET_MAT(out);
        // auto& image_mat = GET_MAT(image);
        // int KSizeX = image.dims(0),
        //     SizeX  = image.dims(0),
        //     SizeY  = image.dims(1);
        // vector<R> kernel_sums;
        // kernel_sums.reserve(kernels.size());
        // std::transform(kernels.begin(), kernels.end(), std::back_inserter(kernel_sums), [&SizeX, &SizeY](const Mat<R>& kern) {
        //     return GET_MAT(kern).block(
        //         0,
        //         0,
        //         SizeX,
        //         SizeY
        //         ).sum();
        // });

        // for (int i = 0; i < kernels.size();i++) {
        //     out_mat(i,0) = (image_mat.array() * GET_MAT(kernels[i]).block(
        //         0,
        //         0,
        //         SizeX,
        //         SizeY
        //         ).array()).sum() / kernel_sums[i];
        // }

        // if (graph::backprop_enabled()) {
        //     graph::emplace_back([image, kernels, out, kernel_sums, kern_col_size](){
        //         auto& image_mat = GET_MAT(image);
        //         int col=0,
        //             KSizeX = image.dims(0),
        //             SizeX  = image.dims(0),
        //             SizeY  = image.dims(1);
        //         bool grad_image = !image.constant;
        //         auto& out_grad = GET_GRAD(out);
        //         auto& out_weight = GET_MAT(out);
        //         std::shared_ptr<Eigen::Matrix<R, Eigen::Dynamic, 1>> surplus;
        //         bool computed_surplus = false;
        //         for (int i=0; i < kernels.size();i++) {
        //             if (!kernels[i].constant) {
        //                 if (!computed_surplus) {
        //                     surplus = make_shared<Eigen::Matrix<R, Eigen::Dynamic, 1>>((out_weight.array() * out_grad.array()).rowwise().sum());
        //                     computed_surplus = true;
        //                 }
        //                 GET_GRAD(kernels[i]).array() -= (*surplus)(i,0) / kernel_sums[i];
        //             }
        //         }
        //         if (grad_image) {
        //             for (int i=0; i < kernels.size();i++) {
        //                 GET_GRAD(image).noalias() += GET_MAT(kernels[i]).block(0, 0, SizeX, SizeY) * (out_grad(i, 0) / kernel_sums[i]);
        //             }
        //         }
        //         for (int i=0; i < kernels.size();i++) {
        //             if (!kernels[i].constant) {
        //                 GET_GRAD(kernels[i]).block(0, 0, SizeX, SizeY).noalias() += (image_mat.array() * (out_grad(i, 0) / (kernel_sums[i]))).matrix();
        //             }
        //         }
        //     });
        // }
        // return out;
        return Mat<R>(1,1);
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
