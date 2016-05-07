#ifndef DALI_MAT_MATH_TENSOR_ARG_OPS_H
#define DALI_MAT_MATH_TENSOR_ARG_OPS_H

#include <mshadow/tensor.h>
#include "dali/math/ThrustUtils.h"

namespace TensorOps {
    using mshadow::gpu;
    using mshadow::cpu;

    namespace arg {
        template <typename R, int dimension>
        std::vector<int> argsort(const mshadow::Tensor<cpu, dimension, R> A, int num_elts) {
            std::vector<int> arguments(num_elts);
            // initialize ordering data:
            for (int i=0;i<num_elts;i++)
                arguments[i] = i;

            auto ptr = A.dptr_;
            // sort in increasing order
            std::sort(arguments.begin(), arguments.end(), [&ptr](const int& lhs, const int& rhs) {
                return *(ptr + lhs) < *(ptr + rhs);
            });
            return arguments;
        }

#ifdef DALI_USE_CUDA
        // Convert a linear index to a row index
        template <typename R>
        struct linear_index_to_row_index : public thrust::unary_function<R,R> {
            R cols; // number of columns
            __host__ __device__
            linear_index_to_row_index(R _cols) : cols(_cols) {}
            __host__ __device__
            R operator()(R i) {
                return i / cols;
            }
        };

        // convert a linear index to a row index
        template <typename T>
        struct linear_index_to_col_index: public thrust::unary_function<T, T> {
            T rows; // number of rows
            __host__ __device__ linear_index_to_col_index(T rows) : rows(rows) {}

            __host__ __device__ T operator() (T i) {
                return i % rows;
            }
        };

        template <int extraction, typename TA, typename TB>
        struct thrust_extract_arg : public thrust::unary_function<TA,TB> {
            __host__ __device__
            auto operator()(const thrust::tuple<TA, TB>& x) -> decltype(thrust::get<extraction>(x)) const {
                return thrust::get<extraction>(x);
            }
        };

        template <int extraction, typename TA, typename TB, typename T>
        struct thrust_extract_arg_divide : public thrust::unary_function<TA,TB> {
            T divisor;
            thrust_extract_arg_divide(T _divisor) : divisor(_divisor) {}

            __host__ __device__
            auto operator()(const thrust::tuple<TA, TB>& x) -> typename thrust::tuple_element<extraction,thrust::tuple<TA, TB>>::type const {
                return thrust::get<extraction>(x) % divisor;
            }
        };

        template <typename R, int dimension>
        std::vector<int> argsort(const mshadow::Tensor<gpu, dimension, R> A, int num_elts) {
            thrust::device_vector<int> arguments(num_elts);
            thrust::device_vector<R> values(num_elts);
            // initialize ordering data:
            thrust::sequence(arguments.begin(), arguments.end(), 0, 1);
            // copy data for sorting:
            thrust::copy(to_thrust(A), to_thrust(A) + num_elts, values.begin());
            // sort arguments based on the values they correspond to:
            thrust::sort_by_key(
              values.begin(), values.end(), arguments.begin()
            );
            std::vector<int> host_arguments(num_elts);
            thrust::copy(arguments.begin(), arguments.end(), host_arguments.begin());
            return host_arguments;
        }
#endif

        template<typename Class>
        class Arger {
            public:
                template<typename R>
                static std::vector<int> apply(const R* start, const int num_elts) {
                    std::vector<int> offset_row(1, 0);
                    int& offset = offset_row[0];
                    for (int i = 0; i < num_elts; i++) {
                        if (Class::compare(*(start + i), *(start + offset))) {
                            offset = i;
                        }
                    }
                    return offset_row;
                }

                template<typename R, typename Device, int dimension>
                static std::vector<int> apply(const mshadow::Tensor<Device, dimension, R> A, int reduce_dim) {
                    ASSERT2(false, utils::MS() << "Operation not supported for tensor with dimension " << dimension);
                    return {};
                }

                template<typename R>
                static std::vector<int> apply(const mshadow::Tensor<cpu, 1, R> A, int reduce_dim) {
                    return apply(A.dptr_, A.shape_[0] );
                }

                template<typename R>
                static std::vector<int> apply(const mshadow::Tensor<cpu, 2, R> A, int reduce_dim) {
                    const int num_rows = A.shape_[0];
                    const int num_cols = A.shape_[1];
                    if (reduce_dim == 1) {
                        std::vector<int> arguments(num_rows);
                        for (int i = 0; i < num_rows; i++) {
                            arguments[i] = 0;
                            for (int j = 0; j < num_cols; j++) {
                                auto rleft = (A.dptr_ + (A.stride_ * i) + j);
                                auto rright = (A.dptr_ + (A.stride_ * i) + arguments[i]);
                                if (!Class::compare(*rright, *rleft)) arguments[i] = j;
                            }
                        }
                        return arguments;
                    } else if (reduce_dim == 0) {
                        std::vector<int> arguments(num_cols);
                        for (int j = 0; j < num_cols; j++) {
                            arguments[j] = 0;
                            for (int i = 0; i < num_rows; i++) {
                                auto rleft = (A.dptr_ + (A.stride_ * i) + j);
                                auto rright = (A.dptr_ + (A.stride_ * arguments[j]) + j);
                                if (!Class::compare(*rright, *rleft)) arguments[j] = i;
                            }
                        }
                        return arguments;
                    } else {
                        throw std::runtime_error("argmax can only be used with axis equal to 0 or 1.");
                    }
                }
#ifdef DALI_USE_CUDA
                #define THRUST_COMP_KERNEL_RESULT_T thrust::tuple<int, R>

                template<typename R>
                struct arger_thrust_kernel : public thrust::binary_function<THRUST_COMP_KERNEL_RESULT_T ,THRUST_COMP_KERNEL_RESULT_T , THRUST_COMP_KERNEL_RESULT_T > {
                    __host__ __device__
                    THRUST_COMP_KERNEL_RESULT_T operator()(const THRUST_COMP_KERNEL_RESULT_T & a, const THRUST_COMP_KERNEL_RESULT_T & b) const {
                        if (Class::compare(thrust::get<1>(a), thrust::get<1>(b))) {
                            return a;
                        } else {
                            return b;
                        }
                    }
                };

                template <typename R>
                static std::vector<int> apply(const mshadow::Tensor<gpu, 2, R> A, int reduce_dim) {
                    int num_rows = A.shape_[0];
                    int num_cols = A.shape_[1];
                    if (reduce_dim == 0) {
                        /* reduce_by_key for thrust only works for contiguous portions, since */
                        /* row-major memory breaks this assumption, we cannot do this without */
                        /* many intermediate steps; we instead transpose & switch reduce dim  */
                        mshadow::Tensor<gpu, 2, R> A_T( mshadow::Shape2(num_cols, num_rows));
                        mshadow::AllocSpace(&A_T, false);
                        A_T = A.T();
                        auto sorted = apply(A_T, 1);
                        mshadow::FreeSpace(&A_T);
                        return sorted;
                    } else if (reduce_dim == 1) {
                        /* allocate storage for row arguments and indices */
                        thrust::device_vector<THRUST_COMP_KERNEL_RESULT_T> row_arguments(num_rows);
                        thrust::device_vector<int> row_indices(num_rows);
                        /* compute row arguments by finding argmin values with equal row indices */
                        thrust::reduce_by_key(
                            thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(num_cols)),
                            thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(num_cols)) + (num_rows * num_cols),
                            thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<int>(0), to_thrust(A))),
                            row_indices.begin(),
                            row_arguments.begin(),
                            thrust::equal_to<int>(),
                            arger_thrust_kernel<R>()
                        );
                        std::vector<int> host_arguments(num_rows);
                        thrust::device_vector<int> row_indices2(num_rows);
                        thrust::transform(
                            row_arguments.begin(),
                            row_arguments.end(),
                            row_indices2.begin(),
                            thrust_extract_arg_divide<0, int, R, int>(num_cols)
                        );
                        thrust::copy(row_indices2.begin(), row_indices2.end(), host_arguments.begin());
                        return host_arguments;
                    } else {
                        throw std::runtime_error("Operation can only be used with axis equal to 0 or 1.");
                    }
                }
                // following the advice from this Stackoverflow:
                // http://stackoverflow.com/questions/7709181/finding-the-maximum-element-value-and-its-position-using-cuda-thrust
                template <typename R>
                static std::vector<int> apply(const thrust::device_ptr<R>& start, int n_elements) {
                    auto idx = Class::thrust_compare(start, start + n_elements);
                    return { (int) (&idx[0] - &start[0]) };
                }
                template <typename R>
                static std::vector<int> apply(const mshadow::Tensor<gpu, 1, R> A, int reduce_dim) {
                    auto start = to_thrust(A);
                    return apply(start, A.shape_[0]);
                }
#endif
        };

        struct Argmax : public Arger<Argmax> {
            template<typename R>
            static MSHADOW_XINLINE bool compare(const R& left, const R& right) {
                return left > right;
            }
#ifdef DALI_USE_CUDA
            template<typename R>
            static thrust::device_ptr<R> thrust_compare(thrust::device_ptr<R> start, thrust::device_ptr<R> end) {
                return std::max_element(start, end);
            }
#endif
        };

        struct Argmin : public Arger<Argmin> {
            template<typename R>
            static MSHADOW_XINLINE bool compare(const R& left, const R& right) {
                return left <= right;
            }

#ifdef DALI_USE_CUDA
            template<typename R>
            static thrust::device_ptr<R> thrust_compare(thrust::device_ptr<R> start, thrust::device_ptr<R> end) {
                return std::min_element(start, end);
            }
#endif
        };

        template<typename R, typename Device, int dimension>
        std::vector<int> argmin(const mshadow::Tensor<Device, dimension, R>& A, int reduce_dim) {
            return Argmin::apply(A, reduce_dim);
        }

        template<typename R, typename Device, int dimension>
        std::vector<int> argmax(const mshadow::Tensor<Device, dimension, R>& A, int reduce_dim) {
            return Argmax::apply(A, reduce_dim);
        }

        template<typename R>
        std::vector<int> argmin(const R* ptr, int number_of_elements) {
            return Argmin::apply(ptr, number_of_elements);
        }

        template<typename R>
        std::vector<int> argmax(const R* ptr, int number_of_elements) {
            return Argmax::apply(ptr, number_of_elements);
        }
#ifdef DALI_USE_CUDA
        template<typename R>
        std::vector<int> argmax(const thrust::device_ptr<R>& ptr, int number_of_elements) {
            return Argmax::apply(ptr, number_of_elements);
        }
        template<typename R>
        std::vector<int> argmin(const thrust::device_ptr<R>& ptr, int number_of_elements) {
            return Argmin::apply(ptr, number_of_elements);
        }
#endif

    } // namespace arg
} // namespace TensorOps
#endif
