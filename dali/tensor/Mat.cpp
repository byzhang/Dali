#include "dali/tensor/Mat.h"
#include "dali/tensor/Index.h"

using std::vector;
using std::string;
using std::stringstream;
using utils::assert2;
using std::make_shared;

const vector<dim_t> mat_missing_dimensions({0,0});

/* Mat */
// this does not need to initialize anything once we get rid of w and dw.
template<typename R>
Mat<R>::Mat() : Mat(0,0) {
}

template<typename R>
typename Mat<R>::storage_t& Mat<R>::w() {
    return *m;
}

template<typename R>
const typename Mat<R>::storage_t& Mat<R>::w() const {
    return *m;
}

template<typename R>
void Mat<R>::forget_w() {
    m = NULL;
}

template<typename R>
void Mat<R>::forget_dw() {
    g = NULL;
}

template<typename R>
typename Mat<R>::storage_t& Mat<R>::dw() {
    return *g;
}

template<typename R>
typename Mat<R>::storage_t& Mat<R>::dw() const {
    return *g;
}

template<typename R>
R Mat<R>::w(int i) const {
    return w()(i);
}

template<typename R>
R& Mat<R>::w(int i) {
    return w()(i);
}

template<typename R>
R Mat<R>::w(int i, int j) const {
    return w()(i,j);
}

template<typename R>
R& Mat<R>::w(int i, int j) {
    return w()(i,j);
}

template<typename R>
R Mat<R>::dw(int i) const {
    return dw()(i);
}

template<typename R>
R& Mat<R>::dw(int i) {
    return dw()(i);
}

template<typename R>
R Mat<R>::dw(int i, int j) const {
    return dw()(i,j);
}

template<typename R>
R& Mat<R>::dw(int i, int j) {
    return dw()(i,j);
}

template<typename R>
vector<dim_t> Mat<R>::dims() const {
    if (m != nullptr) {
        auto& shape = m->shape;
        std::vector<dim_t> dimensions(decltype(m->shape)::kDimension);
        for (int i = 0; i < decltype(m->shape)::kDimension;i++) {
            dimensions[i] = shape[i];
        }
        return dimensions;
    }
    return mat_missing_dimensions;
}

template<typename R>
dim_t Mat<R>::dims(int idx) const {
    if (m != nullptr)
        return m->shape[idx];
    return (dim_t) 0;
}

template<typename R>
bool Mat<R>::empty() const {
    return number_of_elements() == 0;
}

template<typename R>
Mat<R>::Mat(dim_t n, dim_t d) : Mat(n,d, true) {
}

template<typename R>
void Mat<R>::resize(dim_t n, dim_t d) {
    if (m == nullptr || g == nullptr) {
        if (n * d > 0) {
            // Don't fill with zeros - it's initializer's job.
            m = make_shared<TensorInternal<R,2>>(mshadow::Shape2(n, d));
            // We always reset the grad calculation
            g = make_shared<TensorInternal<R,2>>(mshadow::Shape2(n, d));
            g->clear();
        }
    } else if (n * d > 0) {
        MatOps<R>::resize(*this, n, d);
    } else {
        forget_w();
        forget_dw();
    }
}

/**
This is the only Matrix constructor, all other
constructors reference this one.
If this one breaks, the whole ship goes to the
bottom of the ocean.
Do not let this one break. Please

-Sincerely, the Tux Family

Note: the copy constructor below is only a sideshow,
**this** is where the action is!

**/
template<typename R>
Mat<R>::Mat(dim_t n, dim_t d, typename weights<R>::initializer_t wi) :
        name(nullptr), constant(false) {
    if (n * d > 0) {
        // Don't fill with zeros - it's initializer's job.
        m = make_shared<TensorInternal<R,2>>(mshadow::Shape2(n, d));
        // We always reset the grad calculation
        g = make_shared<TensorInternal<R,2>>(mshadow::Shape2(n, d));
        g->clear();
        wi(w());
    }
}

template<typename R>
Mat<R>::Mat (dim_t n, dim_t d, bool fill_zeros) :
        Mat(n, d, fill_zeros ? weights<R>::zeros() : weights<R>::empty()) {
}

template<typename R>
Mat<R>::Mat(string fname) :
        name(nullptr),
        constant(false) {
    npy_load(fname);
}

template<typename R>
Mat<R>::Mat(const Mat<R>& other, bool copy_w, bool copy_dw) :
        name(other.name),
        constant(other.constant) {

    if (copy_w && other.m != nullptr) {
        // This copies memory using copy constructor
        // The copy is only executed if matrix was actually initialized
        // hence the && other.m part.
        m = make_shared<TensorInternal<R,2>>(*other.m, true);
    } else {
        // This does not. (only shared_ptr is copied).
        m = other.m;
    }

    if (copy_dw && other.g != nullptr) {
        // see comment for copy_w.
        g = make_shared<TensorInternal<R,2>>(*other.g, true);
    } else {
        g = other.g;
    }
}

template<typename R>
Mat<R> Mat<R>::shallow_copy() {
    return Mat(*this, false, true);
}


template<typename R>
void Mat<R>::set_name(string& _name) {
    if (name != nullptr) {
        *name = _name;
    } else {
        name = std::make_shared<string>(_name);
    }
}

template<typename R>
void Mat<R>::set_name(char * _name) {
    if (name != nullptr) {
        *name = _name;
    } else {
        name = std::make_shared<string>(_name);
    }
}

template<typename R>
void Mat<R>::set_name(const char * _name) {
    if (name != nullptr) {
        *name = _name;
    } else {
        name = std::make_shared<string>(_name);
    }
}

template<typename R>
void Mat<R>::print(std::basic_ostream<char>& stream) const {
    if (m != nullptr) {
        stream << "m shared counts:" << m.use_count() << std::endl;
        w().print(stream);
    } else {
        stream << "[]";
    }
}

template<typename R>
void Mat<R>::grad() {
    MatOps<R>::grad(this);
}


template<typename R>
void Mat<R>::clear_grad() {
    dw().clear();
}

template<typename R>
void Mat<R>::clear() {
    w().clear();
    dw().clear();
}

template<typename R>
void Mat<R>::npy_save (string fname, string mode) {
    auto dimensions = dims();
    cnpy::npy_save(
        fname,
        w().data(),
        dimensions.data(),
        dimensions.size(),
        mode
    );
}

template<typename R>
unsigned int Mat<R>::number_of_elements() const {
    if (m != nullptr) {
        return w().number_of_elements();
    }
    return 0;
}

template<typename R>
Mat<R> Mat<R>::eltmul(R alpha) const {
    return MatOps<R>::eltmul(*this, alpha);
}

#define MAT_OP_SPECIALIZATION(fname, opname, R, ScalarType) \
        template<>                                          \
        template<>                                          \
        Mat<R> Mat<R>::fname(ScalarType power) const {      \
            return MatOps<R>::opname(*this, (R)power);      \
        }

#define MAT_OP_SPECIALIZATIONS(fname, opname)              \
        MAT_OP_SPECIALIZATION(fname,opname,float,float);   \
        MAT_OP_SPECIALIZATION(fname,opname,float,double);  \
        MAT_OP_SPECIALIZATION(fname,opname,float,int);     \
        MAT_OP_SPECIALIZATION(fname,opname,double,float);  \
        MAT_OP_SPECIALIZATION(fname,opname,double,double); \
        MAT_OP_SPECIALIZATION(fname,opname,double,int);    \
        MAT_OP_SPECIALIZATION(fname,opname,int,float);     \
        MAT_OP_SPECIALIZATION(fname,opname,int,double);    \
        MAT_OP_SPECIALIZATION(fname,opname,int,int);

MAT_OP_SPECIALIZATIONS(pow,pow);
MAT_OP_SPECIALIZATIONS(operator^,pow);

template<typename R>
Mat<R> Mat<R>::steep_sigmoid(R aggressiveness) const {
    return MatOps<R>::steep_sigmoid(*this, aggressiveness);
}

template<typename R>
bool Mat<R>::is_nan() const {
    return MatOps<R>::is_nan(*this);
}

template<typename R>
bool Mat<R>::is_grad_nan() const {
    return MatOps<R>::is_grad_nan(*this);
}


#define MAT_BINARY_OP( opname ) \
    template<typename R> \
    Mat<R> Mat<R>::opname(Mat<R> matrix) const {\
        return MatOps<R>::opname(*this, matrix);\
    }

MAT_BINARY_OP( eltmul_broadcast_colwise )
MAT_BINARY_OP( eltmul )
MAT_BINARY_OP( eltmul_broadcast_rowwise )
MAT_BINARY_OP( eltmul_rowwise )
MAT_BINARY_OP( add )
MAT_BINARY_OP( sub )
MAT_BINARY_OP( add_broadcast_rowwise )
MAT_BINARY_OP( add_broadcast_colwise )
MAT_BINARY_OP( sub_broadcast )
MAT_BINARY_OP( sub_broadcast_reversed )
MAT_BINARY_OP( mul )

// syntactic sugar
template<typename R>
Mat<R> Mat<R>::dot(Mat<R> other) const {
    return MatOps<R>::mul(*this, other);
}

#define MAT_UNARY_OP( opname ) \
    template<typename R> \
    Mat<R> Mat<R>::opname() const {\
        return MatOps<R>::opname(*this);\
    }\

MAT_UNARY_OP( square )
MAT_UNARY_OP( L2_norm )
MAT_UNARY_OP( sqrt )
MAT_UNARY_OP( elt_inv )
MAT_UNARY_OP( tanh )
MAT_UNARY_OP( softplus )
MAT_UNARY_OP( sigmoid )
MAT_UNARY_OP( sum )
MAT_UNARY_OP( mean )
MAT_UNARY_OP( max )
MAT_UNARY_OP( min )
MAT_UNARY_OP( log )
MAT_UNARY_OP( exp )
MAT_UNARY_OP( abs )
MAT_UNARY_OP( relu )

template<typename R>
Mat<R> Mat<R>::T() const {
    return MatOps<R>::transpose(*this);
}

template<typename R>
Mat<R> Mat<R>::slice(int rowstart, int rowwend) const {
    return MatOps<R>::slice(*this, rowstart, rowwend);
}

template<typename R>
Mat<R> Mat<R>::reshape(int rows, int cols) const {
    return MatOps<R>::reshape(*this, rows, cols);
}

template<typename R>
Mat<R> Mat<R>::ravel() const {
    return MatOps<R>::reshape(*this, number_of_elements(), 1);
}

template<typename R>
Mat<R> Mat<R>::col(int col) {
    return MatOps<R>::col_pluck(*this, col);
}

template<typename R>
Mat<R> Mat<R>::operator[](
        Indexing::Index indices) const {
    return MatOps<R>::rows_pluck(*this, indices);
}

template<typename R>
Mat<R> Mat<R>::operator[](
        Mat<int> indices) const {
    return MatOps<R>::rows_pluck(*this, indices);
}

template<typename R>
Mat<R> Mat<R>::operator()(
        Indexing::Index indices) const {
    return MatOps<R>::rows_pluck(*this, indices);
}

template<typename R>
Mat<R> Mat<R>::operator()(
        Indexing::Index row_indices,
        Indexing::Index col_indices) const {
    return MatOps<R>::rows_cols_pluck(*this, row_indices, col_indices);
}

template<typename R>
Mat<R> Mat<R>::operator[](
        int row) const {
    return MatOps<R>::row_pluck(*this, row);
}
template<typename R>
Mat<R> Mat<R>::operator()(
        int row) const {
    return MatOps<R>::row_pluck(*this, row);
}

template<typename R>
Mat<R> Mat<R>::operator()(
        void* nothing,
        int col) const {
    return MatOps<R>::col_pluck(*this, col);
}

template<typename R>
void Mat<R>::npy_save (FILE * fp) {
    std::vector<char> header = cnpy::create_npy_header(w().data(),dims().data(),dims().size());
    fwrite(&header[0],sizeof(char),header.size(),fp);
    fwrite(w().data(),sizeof(R), number_of_elements(), fp);
}

template<typename R>
void Mat<R>::npy_load(cnpy::NpyArray& arr) {
    int n = arr.shape[0];
    int d = arr.shape.size() > 1 ? arr.shape[1] : 1;

    g = make_shared<storage_t>(mshadow::Shape2(n,d));
    g->clear();

    m = make_shared<storage_t>(mshadow::Shape2(n,d));
    auto mut_data = w().mutable_cpu_data();
    R* data_ptr = mut_data.dptr_;

    if (arr.word_size == sizeof(double)) {
        double* loaded_data = reinterpret_cast<double*>(arr.data);
        if (arr.fortran_order) {
            for (int i = 0; i < dims(0); i++) {
                for (int j = 0; j < dims(1); j++) {
                    mut_data[i][j] = loaded_data[j * dims(0) + i];
                }
            }
        } else {
            for (int i = 0; i < dims(0); i++) {
                for (int j = 0; j < dims(1); j++) {
                    mut_data[i][j] = loaded_data[i * dims(1) + j];
                }
            }
        }
    } else if (arr.word_size == sizeof(float)) {
        float* loaded_data = reinterpret_cast<float*>(arr.data);
        if (arr.fortran_order) {
            for (int i = 0; i < dims(0); i++) {
                for (int j = 0; j < dims(1); j++) {
                    mut_data[i][j] = loaded_data[j * dims(0) + i];
                }
            }
        } else {
            for (int i = 0; i < dims(0); i++) {
                for (int j = 0; j < dims(1); j++) {
                    mut_data[i][j] = loaded_data[i * dims(1) + j];
                }
            }
        }
    } else {
        ASSERT2(arr.word_size == sizeof(double) || arr.word_size == sizeof(float),
            "Could not load numpy matrix : not recognized as float or double.");
    }
}

template<typename R>
void Mat<R>::npy_load(FILE * fp) {
    auto arr = cnpy::load_the_npy_file(fp);
    npy_load(arr);
    arr.destruct();
}

template<typename R>
void Mat<R>::npy_load(string fname) {
    auto arr = cnpy::npy_load(fname);
    npy_load(arr);
    arr.destruct();
}

template<typename R>
Mat<R>::~Mat() {}

template<typename R>
void Mat<R>::copy_from(const Mat<R>& source) {
    return MatOps<R>::copy(this, source);
}

template<typename R>
void Mat<R>::copy_grad_from(const Mat<R>& source) {
    return MatOps<R>::copy_grad(this, source);
}

template<typename R>
Mat<R> Mat<R>::Empty(dim_t n, dim_t d) {
    // use an empty matrix and modify
    // it so as to not incur the filling
    // with zeros cost.
    return Mat(n, d, false);
}



template<typename R>
Mat<R> Mat<R>::operator+(Mat<R> other) const {
    return MatOps<R>::add(*this, other);
}

template<typename R>
Mat<R> Mat<R>::operator+(R other) const {
    return MatOps<R>::add(*this, other);
}

template<typename R>
Mat<R>& Mat<R>::operator+=(Mat<R> other) {
    auto sum = MatOps<R>::add(*this, other);
    this->m = sum.m;
    this->g = sum.g;
    return *this;
}

template<typename R>
Mat<R>& Mat<R>::operator+=(R other) {
    auto sum = MatOps<R>::add(*this, other);
    this->m = sum.m;
    this->g = sum.g;
    return *this;
}

template<typename R>
Mat<R> Mat<R>::operator-(Mat<R> other) const {
    return MatOps<R>::sub(*this, other);
}

template<typename R>
Mat<R> Mat<R>::operator-(R other) const {
    return MatOps<R>::add(*this, -other);
}

template<typename R>
Mat<R>& Mat<R>::operator-=(Mat<R> other) {
    auto diff = MatOps<R>::sub(*this, other);
    this->m = diff.m;
    this->g = diff.g;
    return *this;
}

template<typename R>
Mat<R>& Mat<R>::operator-=(R other) {
    auto diff = MatOps<R>::add(*this, -other);
    this->m = diff.m;
    this->g = diff.g;
    return *this;
}

template<typename R>
Mat<R> Mat<R>::operator*(Mat<R> other) const {
    return MatOps<R>::eltmul(*this, other);
}

template<typename R>
Mat<R> Mat<R>::operator*(R alpha) const {
    return MatOps<R>::eltmul(*this, alpha);
}

template<typename R>
Mat<R>& Mat<R>::operator*=(Mat<R> other) {
    auto prod = MatOps<R>::eltmul(*this, other);
    this->m = prod.m;
    this->g = prod.g;
    return *this;
}

template<typename R>
Mat<R>& Mat<R>::operator*=(R other) {
    auto prod = MatOps<R>::eltmul(*this, other);
    this->m = prod.m;
    this->g = prod.g;
    return *this;
}


template<typename R>
Mat<R> Mat<R>::operator-() const {
    return (*this) * -1;
}

template<typename R>
Mat<R> Mat<R>::operator/(Mat<R> other) const {
    return MatOps<R>::eltdivide(*this, other);
}

template<typename R>
Mat<R> Mat<R>::operator/(R alpha) const {
    return MatOps<R>::eltdivide(*this, alpha);
}

template<typename R>
Mat<R>& Mat<R>::operator/=(Mat<R> other) {
    auto divided = MatOps<R>::eltdivide(*this, other);
    this->m = divided.m;
    this->g = divided.g;
    return *this;
}

template<typename R>
Mat<R>& Mat<R>::operator/=(R other) {
    auto divided = MatOps<R>::eltdivide(*this, other);
    this->m = divided.m;
    this->g = divided.g;
    return *this;
}

template<typename R>
Mat<R> Mat<R>::operator^(Mat<R> other) const {
    return MatOps<R>::pow(*this, other);
}


template<typename R>
Mat<R> Mat<R>::zeros_like(Mat<R> other) {
    return Mat<R>(other.dims(0), other.dims(1));
}

template<typename R>
Mat<R> Mat<R>::empty_like(Mat<R> other) {
    return Mat<R>(other.dims(0), other.dims(1), false);
}

/* External operators */
template<typename R>
Mat<R> operator+(int other, Mat<R> mat) {
    return MatOps<R>::add(mat, (R) other);
}
template<typename R>
Mat<R> operator+(float other, Mat<R> mat) {
    return MatOps<R>::add(mat, other);
}
template<typename R>
Mat<R> operator+(double other, Mat<R> mat) {
    return MatOps<R>::add(mat, other);
}


template<typename R>
Mat<R> operator-(int other, Mat<R> mat) {
    return MatOps<R>::sub_broadcast_reversed(mat, (R) other);
}
template<typename R>
Mat<R> operator-(float other, Mat<R> mat) {
    return MatOps<R>::sub_broadcast_reversed(mat, other);
}
template<typename R>
Mat<R> operator-(double other, Mat<R> mat) {
    return MatOps<R>::sub_broadcast_reversed(mat, other);
}


template<typename R>
Mat<R> operator*(int other, Mat<R> mat) {
    return MatOps<R>::eltmul(mat, (R)other);
}
template<typename R>
Mat<R> operator*(float other, Mat<R> mat) {
    return MatOps<R>::eltmul(mat, other);
}
template<typename R>
Mat<R> operator*(double other, Mat<R> mat) {
    return MatOps<R>::eltmul(mat, other);
}

template Mat<float> operator+(int, Mat<float>);
template Mat<float> operator+(float, Mat<float>);
template Mat<float> operator+(double, Mat<float>);

template Mat<double> operator+(int, Mat<double>);
template Mat<double> operator+(float, Mat<double>);
template Mat<double> operator+(double, Mat<double>);


template Mat<float> operator-(int, Mat<float>);
template Mat<float> operator-(float, Mat<float>);
template Mat<float> operator-(double, Mat<float>);

template Mat<double> operator-(int, Mat<double>);
template Mat<double> operator-(float, Mat<double>);
template Mat<double> operator-(double, Mat<double>);


template Mat<float> operator*(int, Mat<float>);
template Mat<float> operator*(float, Mat<float>);
template Mat<float> operator*(double, Mat<float>);

template Mat<double> operator*(int, Mat<double>);
template Mat<double> operator*(float, Mat<double>);
template Mat<double> operator*(double, Mat<double>);


template<typename R>
std::ostream& operator<<(std::ostream& strm, const Mat<R>& a) {
    if (a.name != nullptr) {
        return strm << "<#Mat name=\"" << *a.name<< "\" n=" << a.dims(0) << ", d=" << a.dims(1) << ">";
    } else {
        return strm << "<#Mat n=" << a.dims(0) << ", d=" << a.dims(1) << ">";
    }
}

template std::ostream& operator<< <float>(std::ostream& strm, const Mat<float>& a);
template std::ostream& operator<< <double>(std::ostream& strm, const Mat<double>& a);
template std::ostream& operator<< <int>(std::ostream& strm, const Mat<int>& a);

template <typename R>
std::size_t std::hash<Mat<R>>::operator()(const Mat<R>& k) const {
    auto ptr = &(k.w());
    auto hasher = std::hash<decltype(ptr)>();
    return hasher(ptr);
}

template std::size_t std::hash<Mat<float>>::operator()(const Mat<float>& k)   const;
template std::size_t std::hash<Mat<double>>::operator()(const Mat<double>& k) const;
template std::size_t std::hash<Mat<int>>::operator()(const Mat<int>& k) const;

template <typename R>
bool operator!=(const Mat<R>& A, const Mat<R>& B) {
    return &(A.w()) != &(B.w());
}

template bool operator!=(const Mat<float>&, const Mat<float>&);
template bool operator!=(const Mat<double>&, const Mat<double>&);

template <typename R>
bool operator==(const Mat<R>& A, const Mat<R>& B) {
    return &(A.w()) == &(B.w());
}

template bool operator==<float>(const Mat<float>&, const Mat<float>&);
template bool operator==<double>(const Mat<double>&, const Mat<double>&);

template<typename R>
int Mat<R>::argmax() const {
    return MatOps<R>::argmax(*this);
}

template<typename R>
int Mat<R>::argmin() const {
    return MatOps<R>::argmin(*this);
}

template<typename R>
vector<int> Mat<R>::argmin(int dimension) const {
    return MatOps<R>::argmin(*this, dimension);
}

template<typename R>
vector<int> Mat<R>::argmax(int dimension) const {
    return MatOps<R>::argmax(*this, dimension);
}

template <typename R>
vector<int> Mat<R>::argsort() const {
    return MatOps<R>::argsort(*this);
}

template<typename R>
int Mat<R>::argmax_slice(int lower, int upper) const {
    return MatOps<R>::argmax_slice(*this, lower, upper);
}

template<typename R>
int Mat<R>::argmin_slice(int lower, int upper) const {
    return MatOps<R>::argmin_slice(*this, lower, upper);
}

template<typename R>
void Mat<R>::to_cpu() const {
    if (m != nullptr) {
        w().memory().to_cpu();
        dw().memory().to_cpu();
    }
}

#ifdef DALI_USE_CUDA
template<typename R>
void Mat<R>::to_gpu() const {
    if (m != nullptr) {
        w().memory().to_gpu();
        dw().memory().to_gpu();
    }
}
#endif

namespace utils {
    template<typename R>
    void save_matrices(vector<Mat<R>> parameters, string dirname) {
        utils::ensure_directory(dirname);
        const char * c_dirname = dirname.c_str();
        utils::makedirs(c_dirname);
        int i = 0;
        for (auto& param : parameters) {
            stringstream param_location;
            param_location << dirname << "/param_" << i << ".npy";
            param.npy_save(param_location.str());
            i++;
        }
    }

    template<typename R>
    void load_matrices(vector<Mat<R>> parameters, string dirname) {
        utils::ensure_directory(dirname);
        int i = 0;
        for (auto& param : parameters) {
            stringstream param_location;
            param_location << dirname << "/param_" << i << ".npy";
            param.npy_load(param_location.str());
            i++;
        }
    }

    template <>
    vector<size_t> argsort(const vector<Mat<float>> &v) {
        return MatOps<float>::argsort(v);
    }

    template <>
    vector<size_t> argsort(const vector<Mat<double>> &v) {
        return MatOps<double>::argsort(v);
    }

    template void save_matrices(vector<Mat<float> >, string);
    template void save_matrices(vector<Mat<double> >, string);
    template void load_matrices(vector<Mat<float> >, string);
    template void load_matrices(vector<Mat<double> >, string);
}

template class weights<float>;
template class weights<double>;
template class Mat<float>;
template class Mat<double>;
template class Mat<int>;
