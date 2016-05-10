#include "dali/tensor/op/reshaping.h"

#include "dali/tensor/__MatMacros__.h"
#include "dali/math/TensorOps.h"
#include "dali/math/LazyTensor.h"
#include "dali/math/lazy_patch2col.h"
#include "dali/math/lazy_swapaxis.h"
#include "dali/utils/assert2.h"

#define DONT_COMPILE

using std::vector;
using utils::assert2;

namespace matops {

    template<typename R>
    Mat<R> Reshaping<R>::rows_pluck(
            Mat<R> matrix,
            Indexing::Index indices) {
        Mat<int> indices_mat(1, indices.size());
        for (int i = 0; i < indices.size(); ++i) {
            indices_mat.w(i) = indices[i];
        }
        return Reshaping<R>::rows_pluck(matrix, indices_mat);
    }

    template<typename R>
    Mat<R> Reshaping<R>::rows_pluck(
            Mat<R> matrix,
            Mat<int> indices) {
        Mat<R> out (
            indices.number_of_elements(),
            matrix.dims(1),
            weights<R>::empty());

        TensorOps::rows_pluck(MAT(out), MAT(matrix), indices.w().ravel());

        if (graph::backprop_enabled() && !matrix.constant) {
            graph::emplace_back([matrix, out, indices]() mutable {
                TensorOps::rows_pluck_backprop(GRAD(matrix), GRAD(out), indices.w().ravel());
            });
        }
        return out;
    }

    template<typename R>
    Mat<R> Reshaping<R>::broadcast_row_vector(Mat<R> matrix, int num_rows) {
        assert2(matrix.dims(0) == 1, "broadcast: expected a row vector");
        Mat<R> out(num_rows, matrix.dims(1), weights<R>::empty());
        MAT(out) = MAT(matrix).ravel().wrapper().template broadcast<1>(MAT(out).shape);
        if (graph::backprop_enabled() && !matrix.constant) {
            graph::emplace_back([matrix, out]() mutable {
                GRAD(matrix).ravel() += sum_rows(GRAD(out).wrapper());
            });
        }
        return out;
    }

    template<typename R>
    Mat<R> Reshaping<R>::broadcast_col_vector(Mat<R> matrix, int num_cols) {
        assert2(matrix.dims(1) == 1, "broadcast: expected a column vector.");
        Mat<R> out(matrix.dims(0), num_cols, weights<R>::empty());
        MAT(out) = MAT(matrix).ravel().wrapper().template broadcast<0>(MAT(out).shape);
        if (graph::backprop_enabled() && !matrix.constant) {
            graph::emplace_back([matrix, out]() mutable {
                GRAD(matrix).ravel() += sum_cols(GRAD(out).wrapper());
            });
        }
        return out;
    }


    template<typename R>
    Mat<R> Reshaping<R>::hstack(Mat<R> matrix1, Mat<R> matrix2) {
        return Reshaping<R>::hstack({matrix1, matrix2});
    }

    template<typename R>
    Mat<R> Reshaping<R>::hstack(std::initializer_list<Mat<R>> matrices) {
        vector<Mat<R>> matrices_vector(matrices);
        return hstack(matrices_vector);
    }

    template<typename R>
    Mat<R> Reshaping<R>::hstack(const std::vector<Mat<R>>& matrices) {
        int n = -1;
        int d_total = 0;
        for (auto& mat : matrices) {
            if (n == -1) {
                n = mat.dims(0);
            } else {
                ASSERT2(mat.dims(0) == n, "Matrices cannot be joined -- they do not have the same number of rows.");
            }
            d_total+= mat.dims(1);
        }
        Mat<R> out (
            n, d_total, weights<R>::empty()
        );
        int offset = 0;
        int col, row;
        auto out_data = out.w().mutable_cpu_data();

        for (row = 0; row < n; row++) {
            offset = 0;
            for (auto& mat : matrices) {
                const int col_size = mat.dims(1);
                const auto mat_data = mat.w().cpu_data();
                for (col = 0; col < col_size; col++) {
                    *(out_data.dptr_ + (out_data.stride_ * row) + (col + offset)) = *(mat_data.dptr_ + (mat_data.stride_ * row) + col);
                }
                offset += col_size;
            }
        }

        if (graph::backprop_enabled())
            graph::emplace_back([matrices, out, n]() mutable {
                int offset = 0;
                auto out_data = out.dw().cpu_data();
                int row, col;
                for (row = 0; row < n; row++) {
                    offset = 0;
                    for (auto& mat : matrices) {
                        const int col_size = mat.dims(1);
                        auto mat_data = mat.dw().mutable_cpu_data();
                        for (col = 0; col < col_size; col++) {
                            *(mat_data.dptr_ + (mat_data.stride_ * row) + col ) += *(out_data.dptr_ + (out_data.stride_ * row) + (col + offset));
                        }
                        offset += col_size;
                    }
                }
            });
        return out;
    }

    template<typename R>
    Mat<R> Reshaping<R>::vstack(Mat<R> matrix1, Mat<R> matrix2) {
        return Reshaping<R>::vstack({matrix1, matrix2});
    }


    template<typename R>
    void Reshaping<R>::resize(Mat<R>& matrix, dim_t n, dim_t d) {
        MAT(matrix).resize(mshadow::Shape2(n, d));
        GRAD(matrix).resize(mshadow::Shape2(n, d));
    }

    template<typename R>
    Mat<R> Reshaping<R>::vstack(std::initializer_list<Mat<R>> matrices) {
        vector<Mat<R>> matrices_vector(matrices);
        return vstack(matrices_vector);
    }

    template<typename R>
    Mat<R> Reshaping<R>::vstack(const std::vector<Mat<R>>& matrices) {
        assert(matrices.size() > 0);
        int d = matrices[0].dims(1);
        int n_total = 0;
        for (auto& mat : matrices) {
            ASSERT2(mat.dims(1) == d,
                "Matrices cannot be vertically stacked -- "
                "they do not have the same number of cols.");
            n_total += mat.dims(0);
        }
        Mat<R> out (
            n_total,
            d,
            weights<R>::empty()
        );
        int offset = 0;
        for (auto& mat : matrices) {
            MAT(out).Slice(offset, offset + mat.dims(0)) = MAT(mat).wrapper() + (R)0.0;
            // MAT(out).mutable_cpu_data().Slice(offset, offset + mat.dims(0)) += MAT(mat).cpu_data();
            offset += mat.dims(0);
        }
        if (graph::backprop_enabled())
            graph::emplace_back([matrices, out]() mutable {
                int offset = 0;
                for (auto & mat : matrices) {
                    SAFE_GRAD(mat) +=
                            GRAD(out).Slice(offset, offset + mat.dims(0)).wrapper();
                    offset += mat.dims(0);
                }
            });
        return out;

    }


    template<typename R>
    Mat<R> Reshaping<R>::rows_cols_pluck(
            Mat<R> matrix,
            Indexing::Index row_indices,
            Indexing::Index col_indices) {
        #ifndef DONT_COMPILE
        ASSERT2(row_indices.size() != col_indices.size(),"Cannot pluck column row pairs, not the "
                "same amount of row and column indices.");
            Mat<R> out (
                1,
                row_indices.size(),
                weights<R>::empty());
            for (int offset = 0; offset < row_indices.size(); ++offset)
                MAT(out)(offset) = MAT(matrix)(row_indices[offset], col_indices[offset]);
        if (graph::backprop_enabled() && !matrix.constant) {
            graph::emplace_back([matrix, out, row_indices, col_indices]() mutable {
                auto row_index_ptr = row_indices.data();
                auto col_index_ptr = col_indices.data();
                for (int i = 0; i < out.dims(1); ++i) {
                    // for each row do the same operatoitn as for row_pluck:
                    GRAD(matrix)(*row_index_ptr, *col_index_ptr) += GRAD(out)(i);
                    row_index_ptr++;
                    col_index_ptr++;
                }
            });
        }
        return out;
        #else
        return Mat<R>(1,1);
        #endif
    }

    template<typename R>
    Mat<R> Reshaping<R>::row_pluck(
            Mat<R> matrix,
            int row) {
        ASSERT2(
            0 <= row && row < matrix.dims(0),
            utils::MS() << "Row (" << row
                    << ") must be positive and less than number of rows in matrix ("
                    << matrix.dims(0) << ")."
        );
        Mat<R> out(1, matrix.dims(1), weights<R>::empty());
        MAT(out)  = MAT(matrix)[row].reshape(MAT(out).shape);
        GRAD(out) = GRAD(matrix)[row].reshape(MAT(out).shape);

        return out;
    }

    template<typename R>
    Mat<R> Reshaping<R>::reshape(
            Mat<R> matrix,
            int rows, int cols) {
        ASSERT2(
            ((rows * cols) == (matrix.dims(0) * matrix.dims(1))) && rows > 0 && cols > 0 ,
            utils::MS() << "Not the same number of elements in original matrix (" << matrix.dims(0) * matrix.dims(1)
                    << ") and reshaped matrix (" << (rows * cols) << ")."
        );
        Mat<R> out(rows, cols, weights<R>::empty());
        MAT(out)  = MAT(matrix).reshape(mshadow::Shape2(rows, cols));
        GRAD(out) = GRAD(matrix).reshape(mshadow::Shape2(rows, cols));

        return out;
    }

    template<typename R>
    Mat<R> Reshaping<R>::col_pluck(
            Mat<R> matrix,
            int col) {
        ASSERT2 (0 <= col && col <= matrix.dims(1), "Wrong col index used in col_pluck");
        Mat<R> out (matrix.dims(0), 1, weights<R>::empty());

        TensorOps::col_pluck(MAT(out).ravel(), MAT(matrix), col);

        if (graph::backprop_enabled())
            graph::emplace_back([matrix, out, col]() mutable {
                TensorOps::col_pluck_backward(GRAD(matrix), GRAD(out).ravel(), col);
            });
        return out;
    }

    template<typename R>
    Mat<R> Reshaping<R>::slice(
            Mat<R> matrix,
            int rowstart, int rowwend
            ) {
        if (rowstart == rowwend) {
            return Mat<R>(0, matrix.dims(1));
        }
        ASSERT2(rowstart < rowwend,
            utils::MS()
            << "slice end must be greater than or equal to slice start (got "
            << rowstart << " > " <<  rowwend << ")");
        Mat<R> out(
            rowwend - rowstart,
            matrix.dims(1),
            weights<R>::empty());
        MAT(out) = MAT(matrix).Slice(rowstart, rowwend);
        GRAD(out) = GRAD(matrix).Slice(rowstart, rowwend);
        return out;
    }

    template<typename R>
    Mat<R> Reshaping<R>::transpose(Mat<R> matrix) {
        Mat<R> out (
            matrix.dims(1),
            matrix.dims(0),
            weights<R>::empty());
        if (matrix.dims(0) == 1 || matrix.dims(1) == 1) {
            MAT(out) = MAT(matrix).reshape(MAT(out).shape);
            GRAD(out) = GRAD(matrix).reshape(GRAD(out).shape);
        } else {
            MAT(out) = MAT(matrix).wrapper().T();
            if (graph::backprop_enabled() && !matrix.constant)
                graph::emplace_back([matrix, out]() mutable {
                    GRAD(matrix) += (GRAD(out).wrapper()).T();
                });
        }
        return out;
    }

    mshadow::Shape<2> patch2col_no_grad_size(
            const std::vector<int>& four_d_shape,
            const int& kernel_height,
            const int& kernel_width,
            const int& kernel_stride) {
        mshadow::index_t oheight  = (four_d_shape[2] - kernel_height)/kernel_stride + 1;
        mshadow::index_t owidth   = (four_d_shape[3] - kernel_width)/kernel_stride + 1;
        mshadow::index_t nbatch   = four_d_shape[0];
        return mshadow::Shape2(
            four_d_shape[1] * kernel_height * kernel_width,
            nbatch * oheight * owidth
        );
    }

    template<typename R>
    Mat<R> Reshaping<R>::patch2col_no_grad(
            Mat<R> matrix,
            const std::vector<int>& four_d_shape,
            const int& kernel_height,
            const int& kernel_width,
            const int& kernel_stride) {
        ASSERT2(four_d_shape.size() == 4,
            utils::MS() << "four_d_shape argument to patch2col must be a size "
                        << "4 vector (got " << four_d_shape.size() << ")"
        );
        ASSERT2(kernel_height > 0 && kernel_width > 0 && kernel_stride > 0,
            utils::MS() << "kernel height, width, and stride should be strictly positive (got "
                        << "height = " << kernel_height << ", width = " << kernel_width << ", and "
                        << "stride = " << kernel_stride << " instead)."
        );
        ASSERT2(four_d_shape[0] > 0 && four_d_shape[1] > 0 && four_d_shape[2] > 0 && four_d_shape[3] > 0,
            "4d shape dimensions should be strictly positive.");
        int vol = (four_d_shape[0] * four_d_shape[1] * four_d_shape[2] * four_d_shape[3]);
        ASSERT2(matrix.number_of_elements() == vol,
            utils::MS() << "hypercube volume of 4d shape different (" << vol
                        << ") from number of elements in matrix ("
                        << matrix.number_of_elements() << ")."
        );

        ASSERT2(four_d_shape[3] >= kernel_width && four_d_shape[2] >= kernel_height,
                utils::MS() << "patch2col requires kernel width and height to be less than or equal to image dimensions "
                            << "(got kernel " << kernel_height << "x" << kernel_width << " with image shape "
                            << four_d_shape[2] << "x" << four_d_shape[3] << ").");

        int oheight  = (four_d_shape[2] - kernel_height) / kernel_stride + 1;
        int owidth   = (four_d_shape[3] - kernel_width) / kernel_stride + 1;
        int nbatch   = four_d_shape[0];
        // we directly unpack all local patches and do a dot product
        // this cost lots of memory, normally for large image, only unpack several image at a time
        auto image_shape = mshadow::Shape4(
            four_d_shape[0],
            four_d_shape[1],
            four_d_shape[2],
            four_d_shape[3]
        );
        auto out_shape = patch2col_no_grad_size(
            four_d_shape,
            kernel_height,
            kernel_width,
            kernel_stride
        );
        Mat<R> out(
            out_shape[0],
            out_shape[1],
            weights<R>::empty()
        );
        MAT(out) = unpack_patch2col(
            MAT(matrix).reshape(image_shape).wrapper(),
            kernel_height,
            kernel_width,
            kernel_stride
        );
        return out;
    }

    template<typename R>
    Mat<R> Reshaping<R>::patch2col(
            Mat<R> matrix,
            const std::vector<int>& four_d_shape,
            const int& kernel_height,
            const int& kernel_width,
            const int& kernel_stride) {

        auto out = patch2col_no_grad(
            matrix,
            four_d_shape,
            kernel_height,
            kernel_width,
            kernel_stride
        );

        if (graph::backprop_enabled() && !matrix.constant) {
            graph::emplace_back([matrix, out, four_d_shape, kernel_width, kernel_height, kernel_stride]() {
                auto image_shape = mshadow::Shape4(
                    four_d_shape[0],
                    four_d_shape[1],
                    four_d_shape[2],
                    four_d_shape[3]
                );
                GRAD(matrix).reshape(
                    image_shape
                ) = pack_col2patch(
                    GRAD(out).wrapper(),
                    image_shape,
                    kernel_height,
                    kernel_width,
                    kernel_stride
                );
            });
        }
        return out;
    }

    template<int ndim>
    mshadow::Shape<ndim> vector2shape(const std::vector<int>& vshape) {
        mshadow::Shape<ndim> shape;
        for (int i = 0; i < vshape.size(); i++) {
            shape[i] = vshape[i];
        }
        return shape;
    }

    template<int operation_ndim, int axis1, int axis2, typename R>
    Mat<R> swapaxes_impl(Mat<R> mat, const std::vector<int>& reshape) {
        static_assert(axis1 >= 0 && axis1 < operation_ndim, "axis1 is outside of operation_ndim");
        static_assert(axis2 >= 0 && axis2 < operation_ndim, "axis2 is outside of operation_ndim");
        static_assert(axis1 > axis2, "axis1 must be greater than axis2");
        static_assert(axis2 != axis1, "axis1 cannot equal axis2 (this is a no-op)");
        // first assert that there is as much data in reshape as
        // in mat.
        int vol = 1;
        ASSERT2(reshape.size() == operation_ndim,
                utils::MS() << "Need a temporary shape of size " << operation_ndim
                            << " but got " << reshape.size() << " instead"
        );
        for (const auto& val : reshape) {
            ASSERT2(
                val > 0,
                utils::MS() << "all dimensions of reshape size must be strictly positive (got "
                            << reshape << ")"
            );
            vol *= val;
        }
        ASSERT2(vol == mat.number_of_elements(),
            utils::MS() << "swapaxes shape must have as many elements as original matrix (got "
                        << vol << " but should be " << mat.number_of_elements() << ")"
        );

        auto outcome_shape = reshape;
        std::swap(outcome_shape[axis1], outcome_shape[axis2]);

        // output has dimensions of Prod(dim) x dim[-1]
        Mat<R> out(
            vol / outcome_shape.back(),
            outcome_shape.back(), weights<R>::empty()
        );

        out.w().reshape(vector2shape<operation_ndim>(outcome_shape)) = swapaxis<axis1, axis2>(
            mat.w().reshape(vector2shape<operation_ndim>(reshape)).wrapper()
        );

        if (graph::backprop_enabled() && !mat.constant)
            graph::emplace_back([outcome_shape, reshape, mat, out]() {
                mat.dw().reshape(vector2shape<operation_ndim>(reshape)) += swapaxis<axis1, axis2>(
                    out.dw().reshape(vector2shape<operation_ndim>(outcome_shape)).wrapper()
                );
            });
        return out;
    }

    template<typename R>
    Mat<R> Reshaping<R>::swapaxes(Mat<R> mat, const std::vector<int>& reshape, const int& axis1, const int& axis2) {
        if (axis2 > axis1) {
            return swapaxes(mat, reshape, axis2, axis1);
        }

        #define SWAPAXES_NDIM_A1_A2(ndim, A1, A2)\
            if (axis1 == A1 && axis2 == A2) {\
                return swapaxes_impl<ndim, A1, A2>(mat, reshape);\
            }\

        switch (reshape.size()) {
            case 2:
                SWAPAXES_NDIM_A1_A2(2, 1, 0)
                ASSERT2(false, utils::MS() << "axis out of bounds for swapaxes: " << axis1 << "-" << axis2 << ".");
            case 3:
                SWAPAXES_NDIM_A1_A2(3, 1, 0)
                SWAPAXES_NDIM_A1_A2(3, 2, 0)
                SWAPAXES_NDIM_A1_A2(3, 2, 1)
                ASSERT2(false, utils::MS() << "axis out of bounds for swapaxes: " << axis1 << "-" << axis2 << ".");
            case 4:
                SWAPAXES_NDIM_A1_A2(4, 1, 0)
                SWAPAXES_NDIM_A1_A2(4, 2, 0)
                SWAPAXES_NDIM_A1_A2(4, 2, 1)
                SWAPAXES_NDIM_A1_A2(4, 3, 1)
                SWAPAXES_NDIM_A1_A2(4, 3, 2)
                ASSERT2(false, utils::MS() << "axis out of bounds for swapaxes: " << axis1 << "-" << axis2 << ".");
            default:
                ASSERT2(false, utils::MS() << "swapaxes shape dimensionality must be between 2 and 4 (got " << reshape.size() << ").");
        }
    }

    template class Reshaping<float>;
    template class Reshaping<double>;
    template class Reshaping<int>;

}
