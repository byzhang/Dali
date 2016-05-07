#include "dali/math/TensorInternal.h"

#include "dali/math/TensorOps.h"
#include "dali/math/LazyTensor.h"
#include "dali/utils/core_utils.h"

using std::vector;

void dali_init() {
    mshadow::InitTensorEngine<mshadow::cpu>();
    #ifdef DALI_USE_CUDA
        mshadow::InitTensorEngine<mshadow::gpu>();
    #endif
}

template<int dimension>
std::ostream &operator <<(std::ostream &os, const mshadow::Shape<dimension> &shape) {
    if (dimension == 0) {
        return os << "<shape ()>";
    } else {
        os << "<shape (";
        for (int i = 0; i < dimension;i++) {
            os << shape[i];
            if (i != dimension - 1) os << ", ";
        }
        os << ")>";
        return os;
    }
}

template std::ostream& operator<< <0>(std::ostream& strm, const mshadow::Shape<0>& a);
template std::ostream& operator<< <1>(std::ostream& strm, const mshadow::Shape<1>& a);
template std::ostream& operator<< <2>(std::ostream& strm, const mshadow::Shape<2>& a);
template std::ostream& operator<< <3>(std::ostream& strm, const mshadow::Shape<3>& a);
template std::ostream& operator<< <4>(std::ostream& strm, const mshadow::Shape<4>& a);
template std::ostream& operator<< <5>(std::ostream& strm, const mshadow::Shape<5>& a);
template std::ostream& operator<< <6>(std::ostream& strm, const mshadow::Shape<6>& a);
template std::ostream& operator<< <7>(std::ostream& strm, const mshadow::Shape<7>& a);
template std::ostream& operator<< <8>(std::ostream& strm, const mshadow::Shape<8>& a);
template std::ostream& operator<< <9>(std::ostream& strm, const mshadow::Shape<9>& a);

template<typename R, int dimension>
TensorInternal<R,dimension>::TensorInternal(mshadow::Shape<dimension> _shape) :
        shape(_shape),
        offset(0) {
    // we treat the special case of empty matrix
    // as uninitalized memory:
    memory_ = std::make_shared<SynchronizedMemory<R>>(shape.Size(), shape[dimension - 1], default_preferred_device);
}

template<typename R, int dimension>
TensorInternal<R,dimension>::TensorInternal(const TensorInternal& other, bool copy_memory) :
        TensorInternal(other.shape, other.memory_, other.offset) {
    if (copy_memory) {
        memory_ = std::make_shared<SynchronizedMemory<R>>(*other.memory_);
    }
}

template<typename R, int dimension>
const SynchronizedMemory<R>& TensorInternal<R, dimension>::memory() const {
    return *memory_;
}

template<typename R, int dimension>
SynchronizedMemory<R>& TensorInternal<R, dimension>::memory() {
    return *memory_;
}

template<typename R, int dimension>
TensorInternal<R,dimension>::TensorInternal(mshadow::Shape<dimension> _shape,
                                            std::shared_ptr<SynchronizedMemory<R>> _memory,
                                            int _offset) :
        shape(_shape),
        memory_(_memory),
        offset(_offset) {
}

template<typename R, int dimension>
R TensorInternal<R, dimension>::sum() const {
    #ifdef DALI_USE_CUDA
        if (compute_me_on_gpu()) {
            return TensorOps::reduction::sum(this->gpu_data(), this->number_of_elements() );
        }
    #endif

    return TensorOps::reduction::sum(this->cpu_data(), this->number_of_elements() );
}

template<typename R, int dimension>
R TensorInternal<R, dimension>::max() const {
    #ifdef DALI_USE_CUDA
        if (compute_me_on_gpu()) {
            return TensorOps::reduction::max(this->gpu_data(), this->number_of_elements() );
        }
    #endif

    return TensorOps::reduction::max(this->cpu_data(), this->number_of_elements() );
}

template<typename R, int dimension>
R TensorInternal<R, dimension>::min() const {
    #ifdef DALI_USE_CUDA
        if (compute_me_on_gpu()) {
            return TensorOps::reduction::min(this->gpu_data(), this->number_of_elements() );
        }
    #endif

    return TensorOps::reduction::min(this->cpu_data(), this->number_of_elements() );
}

template<typename R, int dimension>
bool TensorInternal<R,dimension>::is_nan() const {
    #ifdef DALI_USE_CUDA
        if (compute_me_on_gpu()) {
            return TensorOps::reduction::is_nan(this->gpu_data(), this->number_of_elements() );
        }
    #endif

    return TensorOps::reduction::is_nan(this->cpu_data(), this->number_of_elements() );
}


template<typename R, int dimension>
std::vector<int> TensorInternal<R, dimension>::argmin(int reduce_dim) const {
    // reduce colwise
    #ifdef DALI_USE_CUDA
    if (compute_me_on_gpu()) {
        return TensorOps::arg::argmin(this->gpu_data(), reduce_dim);
    }
    #endif

    return TensorOps::arg::argmin(this->cpu_data(), reduce_dim);
}

template<typename R, int dimension>
std::vector<int> TensorInternal<R, dimension>::argsort() const {
    #ifdef DALI_USE_CUDA
    if (compute_me_on_gpu()) {
        return TensorOps::arg::argsort(this->gpu_data(), this->number_of_elements());
    }
    #endif

    return TensorOps::arg::argsort(this->cpu_data(), this->number_of_elements());
}

template<typename R, int dimension>
std::vector<int> TensorInternal<R, dimension>::argmax(int reduce_dim) const {
    // reduce colwise
    #ifdef DALI_USE_CUDA
    if (compute_me_on_gpu()) {
        return TensorOps::arg::argmax(this->gpu_data(), reduce_dim);
    }
    #endif

    return TensorOps::arg::argmax(this->cpu_data(), reduce_dim);
}

template<typename R, int dimension>
int TensorInternal<R, dimension>::argmin() const {
    // reduce colwise
    #ifdef DALI_USE_CUDA
    if (compute_me_on_gpu()) {
        return TensorOps::arg::argmin( TensorOps::to_thrust(this->gpu_data()), this->number_of_elements() )[0];
    }
    #endif

    return TensorOps::arg::argmin(this->cpu_data().dptr_, this->number_of_elements() )[0];
}

template<typename R, int dimension>
int TensorInternal<R, dimension>::argmax() const {
    // reduce colwise
    #ifdef DALI_USE_CUDA
    if (compute_me_on_gpu()) {
        return TensorOps::arg::argmax( TensorOps::to_thrust(this->gpu_data()), this->number_of_elements() )[0];
    }
    #endif

    return TensorOps::arg::argmax(this->cpu_data().dptr_, this->number_of_elements() )[0];
}

template<typename R, int dimension>
int TensorInternal<R, dimension>::argmax_slice(int lower, int upper) const {
    #ifdef DALI_USE_CUDA
    if (compute_me_on_gpu()) {
        return TensorOps::arg::argmax( TensorOps::to_thrust(this->gpu_data()) + lower, upper - lower )[0];
    }
    #endif

    return TensorOps::arg::argmax(this->cpu_data().dptr_ + lower, upper - lower)[0];
}
template<typename R, int dimension>
int TensorInternal<R, dimension>::argmin_slice(int lower, int upper) const {
    #ifdef DALI_USE_CUDA
    if (compute_me_on_gpu()) {
        return TensorOps::arg::argmin( TensorOps::to_thrust(this->gpu_data()) + lower, upper - lower )[0];
    }
    #endif

    return TensorOps::arg::argmin(this->cpu_data().dptr_ + lower, upper - lower)[0];
}

template<typename R, int dimension>
R TensorInternal<R, dimension>::L2_norm() const {
    #ifdef DALI_USE_CUDA
        if (compute_me_on_gpu()) {
            return TensorOps::reduction::L2_norm(this->gpu_data(), this->number_of_elements() );
        }
    #endif

    return TensorOps::reduction::L2_norm(this->cpu_data(), this->number_of_elements() );
}

template<typename R, int dimension>
bool TensorInternal<R,dimension>::operator==(const TensorInternal<R,dimension>& other) const {
    #ifdef DALI_USE_CUDA
        if (should_compute_on_gpu(vector<const SynchronizedMemory<R>*>({memory_.get(), other.memory_.get()}))) {
            return TensorOps::comparison::equals(this->gpu_data(), other.gpu_data(), this->number_of_elements());
        }
    #endif
    return TensorOps::comparison::equals(this->cpu_data(), other.cpu_data(), this->number_of_elements());
}

template<typename R, int dimension>
bool TensorInternal<R,dimension>::allclose(const TensorInternal<R,dimension>& other, R tol) const {
    #ifdef DALI_USE_CUDA
        if (should_compute_on_gpu(vector<const SynchronizedMemory<R>*>({memory_.get(), other.memory_.get()}))) {
            return TensorOps::comparison::allclose(this->gpu_data(), other.gpu_data(), this->number_of_elements(), tol);
        }
    #endif
    return TensorOps::comparison::allclose(this->cpu_data(), other.cpu_data(), this->number_of_elements(), tol);
}

template<typename R, int dimension>
typename TensorInternal<R, dimension>::lazy_t TensorInternal<R,dimension>::wrapper() const {
    return lazy_t(*this);
}

template<typename R, int dimension>
TensorInternal<R,dimension>::operator lazy_t() const {
    return wrapper();
}

template<typename R, int dimension>
bool TensorInternal<R, dimension>::compute_me_on_gpu() const {
    #ifdef DALI_USE_CUDA
        if (should_compute_on_gpu(vector<const SynchronizedMemory<R>*>({memory_.get()}))) {
            return true;
        }
    #endif
    return false;
}


template<typename R, int dimension>
R& TensorInternal<R,dimension>::operator()(int i, int j) {
    int offset = this->cpu_data().stride_  * i + j;
    return *(this->mutable_cpu_data().dptr_ + offset);
}

template<typename R, int dimension>
R TensorInternal<R,dimension>::operator()(int i, int j) const {
    // this is wrong for dimension > 2 or == 1
    int offset = this->cpu_data().stride_  * i + j;
    return *(this->cpu_data().dptr_ + offset);
}

template<typename R, int dimension>
R& TensorInternal<R,dimension>::operator()(int i) {
    return *(this->mutable_cpu_data().dptr_ + i);
}

template<typename R, int dimension>
R TensorInternal<R,dimension>::operator()(int i) const {
    return *(this->cpu_data().dptr_ + i);
}

template<typename R, int dimension>
const R* TensorInternal<R,dimension>::data() const {
    return this->cpu_data().dptr_;
}

template<typename R, int dimension>
R* TensorInternal<R,dimension>::data() {
    return this->mutable_cpu_data().dptr_;
}

#define DALI_TENSOR_INTERNAL_PRINT(dtype) \
    template <> void TensorInternal<dtype, 1>::print(std::basic_ostream<char>& stream, int indent) const {\
        stream << std::string(indent, ' ');\
        stream << "[";\
        for(int i=0; i<shape[0]; ++i) {\
            stream << std::fixed\
                      << std::setw( 7 ) /* keep 7 digits*/\
                      << std::setprecision( 3 ) /* use 3 decimals*/\
                      << std::setfill( ' ' ) << (*this)(i); /* pad values with blanks this->w(i,j)*/\
            if (i != shape[0] - 1) stream << " ";\
        }\
        stream << "]";\
        stream << std::endl;\
    }

DALI_TENSOR_INTERNAL_PRINT(float)
DALI_TENSOR_INTERNAL_PRINT(double)
DALI_TENSOR_INTERNAL_PRINT(int)

template<typename R, int dimension>
void TensorInternal<R,dimension>::print(std::basic_ostream<char>& stream, int indent) const {
    static_assert (dimension > 1, "Print called with wrong dimension.");
    stream << std::string(indent, ' ') << "[" << std::endl;
    for (int i=0; i < shape[0]; ++i)
        (*this)[i].print(stream, indent + 4);
    stream << std::string(indent, ' ') <<"]" << std::endl;
}

template<typename R, int dimension>
void TensorInternal<R,dimension>::clear() {
    if (memory_ != nullptr) {
        if (offset > 0 || number_of_elements() != memory().total_memory) {
            // only clear subset of memory
            #ifdef DALI_USE_CUDA
            if (compute_me_on_gpu()) {
                overwrite_gpu_data() = 0;
                return;
            }
            #endif
            overwrite_cpu_data() = 0;
        } else {
            memory().lazy_clear();
        }
    }
}

template<typename R, int dimension>
TensorInternal<R,dimension> TensorInternal<R, dimension>::zeros(mshadow::Shape<dimension> shape) {
    auto tensor = TensorInternal<R,dimension>(shape);
    tensor.clear();
    return tensor;
}

template<typename R, int dimension>
const typename TensorInternal<R,dimension>::cpu_tensor_t TensorInternal<R,dimension>::cpu_data() const {
    return cpu_tensor_t(memory_->cpu_data() + offset, shape);
}

template<typename R, int dimension>
typename TensorInternal<R,dimension>::cpu_tensor_t TensorInternal<R,dimension>::mutable_cpu_data() {
    return cpu_tensor_t(memory_->mutable_cpu_data() + offset, shape);
}

template<typename R, int dimension>
typename TensorInternal<R,dimension>::cpu_tensor_t TensorInternal<R,dimension>::overwrite_cpu_data() {
    // if the memory requested covers the entirety of the span of memory
    // then asking to overwrite is fair
    if (number_of_elements() == memory_->total_memory) {
        return cpu_tensor_t(memory_->overwrite_cpu_data() + offset, shape);
    }
    // if there is less memory being overwritten than the total, then we must ensure that the
    // fresh remainder comes along too.
    return cpu_tensor_t(memory_->mutable_cpu_data() + offset, shape);
}

#ifdef DALI_USE_CUDA
    template<typename R, int dimension>
    const typename TensorInternal<R,dimension>::gpu_tensor_t TensorInternal<R,dimension>::gpu_data() const {
        return gpu_tensor_t(memory_->gpu_data() + offset, shape);
    }

    template<typename R, int dimension>
    typename TensorInternal<R,dimension>::gpu_tensor_t TensorInternal<R,dimension>::mutable_gpu_data() {
        return gpu_tensor_t(memory_->mutable_gpu_data() + offset, shape);
    }

    template<typename R, int dimension>
    typename TensorInternal<R,dimension>::gpu_tensor_t TensorInternal<R,dimension>::overwrite_gpu_data() {
        if (number_of_elements() == memory_->total_memory) {
            return gpu_tensor_t(memory_->overwrite_gpu_data() + offset, shape);
        }
        return gpu_tensor_t(memory_->mutable_gpu_data() + offset, shape);
    }
#endif

template<typename R, int dimension>
int TensorInternal<R,dimension>::number_of_elements() const {
    return shape.Size();
}

template<typename R, int dimension>
TensorInternal<R, dimension - 1> TensorInternal<R,dimension>::operator[](mshadow::index_t idx) const {
    auto subshape = shape.SubShape();
    return TensorInternal<R, dimension - 1>(subshape,
                                            memory_,
                                            offset + subshape.Size() * idx);
}

template<typename R, int dimension>
TensorInternal<R, 1> TensorInternal<R,dimension>::ravel() const {
    auto newshape = mshadow::Shape1(number_of_elements());
    return TensorInternal<R, 1>(newshape, memory_, offset);
}

template<typename R, int dimension>
TensorInternal<R, dimension> TensorInternal<R,dimension>::Slice(
        mshadow::index_t begin, mshadow::index_t end) const {
    auto newshape = shape;
    newshape[0] = end - begin;
    return TensorInternal<R, dimension>(newshape,
                                        memory_,
                                        offset + shape.SubShape().Size() * begin);
}

template<typename R, int dimension>
template<int new_dimension>
TensorInternal<R, new_dimension> TensorInternal<R,dimension>::reshape(mshadow::Shape<new_dimension> new_shape, int extra_offset) const {
    ASSERT2(extra_offset + offset >= 0,
            utils::MS() << "Reshaped TensorInternal's memory offset must be positive "
                           "(was " << offset << " before reshape, and "
                           "extra_offset + offset = " << extra_offset + offset << " < 0).");
    ASSERT2(new_shape.Size() + extra_offset + offset <= memory_->total_memory,
            "Reshape dimensions exceed the size of the underlying memory buffer.");
    return TensorInternal<R, new_dimension>(new_shape,
                                            memory_,
                                            offset + extra_offset);
}

template<typename R, int dimension>
TensorInternal<R,dimension>& TensorInternal<R,dimension>::operator=(const lazy_t& expr) {
    #ifdef DALI_USE_CUDA
        if (should_compute_on_gpu(extract_memory(expr.dependent_tensors))) {
            /* refresh the gpu memory from cpu*/
            for (auto participant : expr.dependent_tensors) {
                participant->update_tensor(DEVICE_GPU);
            }
            mshadow::Copy(this->mutable_gpu_data(), expr.right);
            return *this;
        }
    #endif
    for (auto participant : expr.dependent_tensors) {
        participant->update_tensor(DEVICE_CPU);
    }
    mshadow::Copy(this->mutable_cpu_data(), expr.left);
    return *this;
}


template<typename R, int dimension>
void TensorInternal<R, dimension>::resize(mshadow::Shape<dimension> newshape, R filler) {
    if (newshape == shape)
        return;
    // if same columns
    if (newshape[1] == shape[1]) {
        if (newshape[0] != shape[0]) {
            // if we are only changing the number of rows
            // then we can simply chop off or add memory
            // at the tail end of this tensor's memory buffer.
            R* data_ptr = memory_->mutable_cpu_data();
            R* new_ptr  = (R*)realloc(data_ptr, newshape.Size() * sizeof(R));
            ASSERT2(new_ptr != NULL, "Error: Could not allocated memory for TensorInternal.");
            if (new_ptr != NULL) {
                memory_->cpu_ptr = new_ptr;
                memory_->total_memory = newshape.Size();
                #ifdef DALI_USE_CUDA
                memory_->free_gpu();
                #endif
                // fill new area with zeros.
                for (int i = shape.Size(); i < newshape.Size(); i++) {
                    *(new_ptr + i) = filler;
                }
                for (int i = 0; i < mshadow::Shape<dimension>::kDimension ; i++) {
                    shape.shape_[i] = newshape[i];
                }
            }
        }
    } else {
        if (newshape[1] > shape[1]) {
            if (newshape[0] == shape[0]) {
                // if we are increasing the columns, then
                // we move the farthest rows & columns out to
                // make room for closer columns
                R* data_ptr = memory_->mutable_cpu_data();
                R* new_ptr  = (R*)realloc(data_ptr, newshape.Size() * sizeof(R));
                ASSERT2(new_ptr != NULL, "Error: Could not allocated memory for TensorInternal.");
                if (new_ptr != NULL) {
                    memory_->cpu_ptr = new_ptr;
                    memory_->total_memory = newshape.Size();
                    #ifdef DALI_USE_CUDA
                    memory_->free_gpu();
                    #endif
                    for (int i = shape[0] - 1; i >= 0; i--) {
                        for (int j = shape[1] - 1; j >= 0; j--) {
                            int old_offset = this->cpu_data().stride_ * i + j;
                            int new_offset = newshape[1] * i + j;
                            *(new_ptr + new_offset) = *(new_ptr + old_offset);
                        }
                    }
                    for (int i = 0; i < newshape[0]; i++) {
                        for (int j = shape[1]; j < newshape[1]; j++) {
                            int offset = newshape[1] * i + j;
                            *(new_ptr + offset) = filler;
                        }
                    }
                    for (int i = 0; i < mshadow::Shape<dimension>::kDimension ; i++) {
                        shape.shape_[i] = newshape[i];
                    }
                }
            } else {
                mshadow::Shape<dimension> temp_shape = newshape;
                temp_shape.shape_[1] = shape[1];
                // 1. Ensure that all other dimensions are identical
                resize(temp_shape, filler);
                // 2. Make shape[1] match newshape[1]
                resize(newshape, filler);
            }
        } else {
            if (newshape[0] == shape[0]) {
                R* data_ptr = memory_->mutable_cpu_data();
                // if we are shrinking, then copy closest memory first
                // then move to farther memory
                for (int i = 0; i < newshape[0]; i++) {
                    for (int j = 0; j < newshape[1]; j++) {
                        int old_offset = this->cpu_data().stride_ * i + j;
                        int new_offset = newshape[1] * i + j;
                        *(data_ptr + new_offset) = *(data_ptr + old_offset);
                    }
                }
                R* new_ptr = (R*)realloc(data_ptr, newshape.Size() * sizeof(R));
                ASSERT2(new_ptr != NULL, "Error: Could not allocated memory for TensorInternal.");
                if (new_ptr != NULL) {
                    memory_->cpu_ptr = new_ptr;
                    memory_->total_memory = newshape.Size();
                    #ifdef DALI_USE_CUDA
                    memory_->free_gpu();
                    #endif
                    for (int i = 0; i < mshadow::Shape<dimension>::kDimension ; i++) {
                        shape.shape_[i] = newshape[i];
                    }
                }
            } else {
                mshadow::Shape<dimension> temp_shape = newshape;
                temp_shape.shape_[1] = shape[1];
                // 1. Ensure that all other dimensions are identical
                resize(temp_shape, filler);
                // 2. Make shape[1] match newshape[1]
                resize(newshape, filler);
            }
        }
    }
}

// 1D resize
#ifdef DALI_USE_CUDA
    #define DALI_TENSOR_INTERNAL_RESIZE(dtype) \
        template<> \
        void TensorInternal<dtype, 1>::resize(mshadow::Shape<1> newshape, dtype filler) {\
            if (newshape == shape)\
                return;\
            dtype* data_ptr = memory_->mutable_cpu_data();\
            dtype* new_ptr  = (dtype*)realloc(data_ptr, newshape.Size() * sizeof(dtype));\
            ASSERT2(new_ptr != NULL, "Error: Could not allocated memory for TensorInternal.");\
            memory_->cpu_ptr = new_ptr;\
            memory_->total_memory = newshape.Size();\
            memory_->free_gpu();\
            if (newshape[0] > shape[0]) {\
                for (int i = shape.Size(); i < newshape.Size(); i++) {\
                    *(new_ptr + i) = filler;\
                }\
            }\
            shape.shape_[0] = newshape[0];\
        }
#else
    #define DALI_TENSOR_INTERNAL_RESIZE(dtype) \
        template<> \
        void TensorInternal<dtype, 1>::resize(mshadow::Shape<1> newshape, dtype filler) {\
            if (newshape == shape)\
                return;\
            dtype* data_ptr = memory_->mutable_cpu_data();\
            dtype* new_ptr  = (dtype*)realloc(data_ptr, newshape.Size() * sizeof(dtype));\
            ASSERT2(new_ptr != NULL, "Error: Could not allocated memory for TensorInternal.");\
            memory_->cpu_ptr = new_ptr;\
            memory_->total_memory = newshape.Size();\
            if (newshape[0] > shape[0]) {\
                for (int i = shape.Size(); i < newshape.Size(); i++) {\
                    *(new_ptr + i) = filler;\
                }\
            }\
            shape.shape_[0] = newshape[0];\
        }
#endif

DALI_TENSOR_INTERNAL_RESIZE(float);
DALI_TENSOR_INTERNAL_RESIZE(double);
DALI_TENSOR_INTERNAL_RESIZE(int);

#define DALI_TENSOR_RESHAPE_INSTANTIATE(dtype, fromdim, todim)\
    template \
    TensorInternal<dtype,todim> TensorInternal<dtype,fromdim>::reshape<todim>(mshadow::Shape<todim> newshape, int) const\

#define DALI_TENSOR_RESHAPE_INSTANTIATES_SUBSHAPE(dtype, primary_dim)\
    DALI_TENSOR_RESHAPE_INSTANTIATE(dtype, primary_dim, 1);\
    DALI_TENSOR_RESHAPE_INSTANTIATE(dtype, primary_dim, 2);\
    DALI_TENSOR_RESHAPE_INSTANTIATE(dtype, primary_dim, 3);\
    DALI_TENSOR_RESHAPE_INSTANTIATE(dtype, primary_dim, 4);\

#define DALI_TENSOR_RESHAPE_INSTANTIATES(dtype)\
    DALI_TENSOR_RESHAPE_INSTANTIATES_SUBSHAPE(dtype, 1);\
    DALI_TENSOR_RESHAPE_INSTANTIATES_SUBSHAPE(dtype, 2);\
    DALI_TENSOR_RESHAPE_INSTANTIATES_SUBSHAPE(dtype, 3);\
    DALI_TENSOR_RESHAPE_INSTANTIATES_SUBSHAPE(dtype, 4);\

DALI_TENSOR_RESHAPE_INSTANTIATES(float);
DALI_TENSOR_RESHAPE_INSTANTIATES(int);
DALI_TENSOR_RESHAPE_INSTANTIATES(double);


template class TensorInternal<float, 1>;
template class TensorInternal<double,1>;
template class TensorInternal<int,1>;
template class TensorInternal<float, 2>;
template class TensorInternal<double,2>;
template class TensorInternal<int,2>;
template class TensorInternal<float, 3>;
template class TensorInternal<double,3>;
template class TensorInternal<int,3>;
template class TensorInternal<float, 4>;
template class TensorInternal<double,4>;
template class TensorInternal<int,4>;
// template class TensorInternal<float, 5>;
// template class TensorInternal<double,5>;
// template class TensorInternal<float, 6>;
// template class TensorInternal<double,6>;
// template class TensorInternal<float, 7>;
// template class TensorInternal<double,7>;
// template class TensorInternal<float, 8>;
// template class TensorInternal<double,8>;
// template class TensorInternal<float, 9>;
// template class TensorInternal<double,9>;
