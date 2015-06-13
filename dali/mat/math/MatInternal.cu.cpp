#include "dali/mat/math/MatInternal.h"

#include "dali/mat/math/__MatMacros__.h"

using std::vector;
using std::string;
using mshadow::Shape2;


#ifdef DALI_USE_CUDA
    static const PreferredDevice preferred_device = DEVICE_GPU;
#else
    static const PreferredDevice preferred_device = DEVICE_CPU;
#endif

void dali_init() {
    mshadow::InitTensorEngine<mshadow::cpu>();
#ifdef DALI_USE_CUDA
    mshadow::InitTensorEngine<mshadow::gpu>();
#endif
}

/* MatInternal */

template<typename R>
std::atomic<int> MatInternal<R>::next_matrix(0);

template<typename R>
MatInternal<R>::MatInternal(dim_t n, dim_t d, bool fill_zeros) :
        w(n, d, preferred_device),
        dims( {n, d}),
        id(next_matrix.fetch_add(1)) {
    if (fill_zeros) {
        tensor_fill(w, 0.0);
    }
}

template<typename R>
MatInternal<R>::MatInternal(const MatInternal<R>& m) :
        w(m.w),
        dims(m.dims),
        id(m.id) {
}

template<typename R>
MatInternal<R>::~MatInternal() {
    // SynchronizedTensor handles the destruction.
}

template<typename R>
MatInternal<R>::operator typename MatInternal<R>::mat_storage_t () {
    return w;
}

template<typename R>
R& MatInternal<R>::operator()(int i, int j) {
    return w.mutable_cpu_data()[i][j];
}

template<typename R>
R MatInternal<R>::operator()(int i, int j) const {
    return w.cpu_data()[i][j];
}

template<typename R>
R& MatInternal<R>::operator()(int i) {
    return *(w.mutable_cpu_data().dptr_ + i);
}

template<typename R>
R MatInternal<R>::operator()(int i) const {
    return *(w.cpu_data().dptr_ + i);
}

template<typename R>
const R* MatInternal<R>::data() const {
    return w.cpu_data().dptr_;
}

template<typename R>
R* MatInternal<R>::data() {
    return w.mutable_cpu_data().dptr_;
}

template<typename R>
void MatInternal<R>::print() const {
    const auto& data = w.cpu_data();
    for (int i = 0; i < dims[0] ; ++i) {
        std::cout << (i == 0 ? "[" : " ");
        for (int j = 0; j < dims[1]; ++j) {
            std::cout << std::fixed
                      << std::setw( 7 ) // keep 7 digits
                      << std::setprecision( 3 ) // use 3 decimals
                      << std::setfill( ' ' ) // pad values with blanks this->w(i,j)
                      << data[i][j] << " ";
        }
        std::cout << (i == dims[0] - 1 ? "]" : "\n");
    }
    std::cout << std::endl;
}

template<typename R>
void MatInternal<R>::clear() {
    tensor_fill(w, 0);
}

/* GradInternal */

template<typename R>
GradInternal<R>::GradInternal(dim_t n, dim_t d, bool fill_zeros) :
        dw(n, d, preferred_device) {
    if (fill_zeros) {
        tensor_fill(dw, 0.0);
    }
}

template<typename R>
GradInternal<R>::GradInternal(const GradInternal<R>& g) :
        dw(g.dw) {
}

template<typename R>
GradInternal<R>::~GradInternal() {
    // SynchronizedTensor handles the destruction.
}


template<typename R>
GradInternal<R>::operator typename GradInternal<R>::mat_storage_t () {
    return dw;
}


template<typename R>
R& GradInternal<R>::operator()(int i, int j) {
    return dw.mutable_cpu_data()[i][j];
}

template<typename R>
R GradInternal<R>::operator()(int i, int j) const {
    return dw.cpu_data()[i][j];
}

template<typename R>
R& GradInternal<R>::operator()(int i) {
    return *(dw.mutable_cpu_data().dptr_ + i);
}

template<typename R>
R GradInternal<R>::operator()(int i) const {
    return *(dw.cpu_data().dptr_ + i);
}

template<typename R>
const R* GradInternal<R>::data() const {
    return dw.cpu_data().dptr_;
}

template<typename R>
R* GradInternal<R>::data() {
    return dw.mutable_cpu_data().dptr_;
}

template<typename R>
void GradInternal<R>::clear() {
    tensor_fill(dw, 0);
}

template class MatInternal<float>;
template class MatInternal<double>;
template class GradInternal<float>;
template class GradInternal<double>;

/** END GRADINTERNAL **/
