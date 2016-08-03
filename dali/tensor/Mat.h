#ifndef CORE_MAT_H
#define CORE_MAT_H

#include <atomic>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <ostream>
#include <unordered_map>

#include "dali/math/TensorInternal.h"
#include "dali/tensor/MatOps.h"
#include "dali/tensor/Weights.h"
#include "dali/tensor/Tape.h"
#include "dali/utils.h"

template<typename R> class Mat;

namespace Indexing {
    class Index;
}

template<typename R>
struct weights;
/**
Mat
---

Main matrix class for this library. The Mat
class contains two pieces of memory, `w`
and `dw`. The first is the actual weights or
values associated with this matrix, and the
second is the local contribution to the
objective function (or ∂E/∂Mat). This local
contribution can then be used in
backpropagation.

Mat is used almost everywhere in the library
except in `utils`.

Note: ideally this class would generalize to
higher and lower dimensions. For instance
see `Graph::sum` and `Graph::mean`, or
`Mat::grad` methods to see where a Scalar
type would be useful (today this is a 1x1
matrix -- mathematically sound, but this
is inefficient, and inadequate for good
broadcasting across other operations).

**/

template<typename R>
class Mat {
    public:
        typedef TensorInternal<R,2>            storage_t;
        typedef std::shared_ptr<storage_t> storage_ref_t;
    private:
        storage_ref_t m;
        mutable storage_ref_t g;
    public:

        std::shared_ptr<std::string> name = nullptr;

        bool constant;

        Mat();

        // Initializes with zeros;
        Mat (dim_t n, dim_t d, Device preferred_device = default_preferred_device);
        // sometimes we don't need to reset m
        // (for example if it's about to be assigned).
        explicit Mat (dim_t n, dim_t d, bool fill_zeros, Device preferred_device = default_preferred_device);
        explicit Mat (dim_t n, dim_t d,
             typename weights<R>::initializer_t wi, Device preferred_device = default_preferred_device);
        /*
        A copy constructor that perform shallow and deep
        copies of a Mat.
        By default shallow copy is performed.

        Key usage is for Hogwild style training of parameters
        where different computation threads share memory for
        the parameters but each compute their own gradients.
        The gradients are kept in separate `dw` memory buffers
        but `w` buffers are shared amongst threads.
        */
        Mat (const Mat<R>& m, bool copy_w=false, bool copy_d=false);

        ~Mat();

        void copy_from(const Mat<R>& source);
        void copy_grad_from(const Mat<R>& source);

        void print(std::basic_ostream<char>& stream = std::cout) const;

        /**
        Adds 1 to the gradient (`dw`) of every element in this Matrix as long
        as the matrix is 1x1 (a scalar).

        Useful for computing cross entropy, mean squared error, and other
        loss functions in vanilla ML fashion.

        **/
        void grad();

        void clear_grad();
        void clear();

        const storage_t& w() const;
        storage_t& w();

        R w(int i) const;
        R& w(int i);
        R dw(int i) const;
        R& dw(int i);
        R w(int i, int j) const;
        R& w(int i, int j);
        R dw(int i, int j) const;
        R& dw(int i, int j);

        void forget_w();
        void forget_dw();

        storage_t& dw() const;
        storage_t& dw();

        std::vector<dim_t> dims() const;
        dim_t dims(int idx) const;

        unsigned int number_of_elements() const;

        bool empty() const;

        void set_name(std::string& newname);
        void set_name(char* newname);
        void set_name(const char* newname);

        void npy_save(std::string fname, std::string mode = "w");
        void npy_save(FILE*);
        void npy_load(std::string fname);
        void npy_load(FILE*);
        void npy_load(cnpy::NpyArray&);
        Mat (std::string fname);
        static Mat Empty(dim_t n, dim_t d);
        /* A copy constructor that perform shallow copies of a Mat.

        Key usage is for Hogwild style training of parameters
        where different computation threads share memory for
        the parameters but each compute their own gradients.
        The gradients are kept in separate `dw` memory buffers
        but `w` buffers are shared amongst threads. */
        Mat shallow_copy();
        operator std::string() const;
        void resize(dim_t rows, dim_t cols);

        // Various operations on matrix.
        // Soon to be replaced by legitimate operators
        // See MatOps for documentation.
        bool is_nan() const;
        bool is_grad_nan() const;
        Mat<R> eltmul_broadcast_colwise(Mat<R>) const;
        Mat<R> eltmul(Mat<R>) const;
        Mat<R> eltmul(R) const;
        Mat<R> eltmul_broadcast_rowwise(Mat<R>) const;
        Mat<R> eltmul_rowwise(Mat<R>) const;
        Mat<R> add_broadcast_rowwise(Mat<R>) const;
        Mat<R> add_broadcast_colwise(Mat<R>) const;
        Mat<R> add(Mat<R>) const;
        Mat<R> sub(Mat<R>) const;
        Mat<R> sub_broadcast(Mat<R>) const;
        Mat<R> sub_broadcast_reversed(Mat<R>) const;
        Mat<R> square() const;
        Mat<R> L2_norm() const;
        Mat<R> sum() const;
        Mat<R> mean() const;
        Mat<R> max() const;
        Mat<R> min() const;
        Mat<R> log() const;
        Mat<R> exp() const;
        Mat<R> abs() const;
        Mat<R> sigmoid() const;
        Mat<R> steep_sigmoid(R aggressiveness = 3.75) const;
        // Warning: transpose makes a copy, uses extra memory
        Mat<R> T() const;
        Mat<R> tanh() const;
        Mat<R> softplus() const;
        Mat<R> relu() const;
        Mat<R> mul(Mat<R>) const;
        Mat<R> dot(Mat<R>) const;
        template<typename ScalarType>
        Mat<R> pow(ScalarType) const;
        Mat<R> sqrt() const;
        Mat<R> elt_inv() const;
        Mat<R> slice(int rowstart, int rowend) const;
        Mat<R> reshape(int rows, int cols) const;
        Mat<R> ravel() const;

        int argmax() const;
        int argmin() const;

        std::vector<int> argmin(int dimension) const;
        std::vector<int> argmax(int dimension) const;
        std::vector<int> argsort() const;
        /*
        Restricted range argmax: returns the index of the
        highest value between two indices, lower and upper
        (useful if a range of predictions is inadmissible,
        so we are only considering a subset of predictions)
        */
        int argmax_slice(int lower, int upper) const;
        int argmin_slice(int lower, int upper) const;

        Mat<R> operator-() const;

        Mat<R> operator+(Mat<R>) const;
        Mat<R> operator+(R) const;
        Mat<R>& operator+=(Mat<R>);
        Mat<R>& operator+=(R);

        Mat<R> operator-(Mat<R>) const;
        Mat<R> operator-(R) const;
        Mat<R>& operator-=(Mat<R>);
        Mat<R>& operator-=(R);

        Mat<R> operator*(Mat<R> other) const;
        Mat<R> operator*(R alpha) const;
        Mat<R>& operator*=(Mat<R>);
        Mat<R>& operator*=(R);

        Mat<R> operator/(Mat<R> other) const;
        Mat<R> operator/(R alpha) const;
        Mat<R>& operator/=(Mat<R>);
        Mat<R>& operator/=(R);

        template<typename ScalarType>
        Mat<R> operator^(ScalarType) const;

        Mat<R> operator^(Mat<R>) const;


        // Plucking rows and columns:
        Mat<R> col(int col);
        Mat<R> operator[](int) const;
        Mat<R> operator[](Mat<int>) const;
        Mat<R> operator()(int) const;
        Mat<R> operator[](Indexing::Index) const;
        Mat<R> operator()(Indexing::Index) const;
        Mat<R> operator()(Indexing::Index, Indexing::Index) const;
        // Mat<R> operator()(void*, Indexing::Index) const;
        Mat<R> operator()(void*, int) const;
        static Mat<R> zeros_like(Mat<R> shape);
        static Mat<R> empty_like(Mat<R> shape);

        // forcing memory location
        void to_cpu() const;

        #ifdef DALI_USE_CUDA
            void to_gpu() const;
        #endif
};
extern const std::vector<dim_t> mat_missing_dimensions;

template<typename R>
Mat<R> operator+(int other, Mat<R> mat);
template<typename R>
Mat<R> operator+(float other, Mat<R> mat);
template<typename R>
Mat<R> operator+(double other, Mat<R> mat);

template<typename R>
Mat<R> operator-(int other, Mat<R> mat);
template<typename R>
Mat<R> operator-(float other, Mat<R> mat);
template<typename R>
Mat<R> operator-(double other, Mat<R> mat);

template<typename R>
Mat<R> operator*(int other, Mat<R> mat);
template<typename R>
Mat<R> operator*(float other, Mat<R> mat);
template<typename R>
Mat<R> operator*(double other, Mat<R> mat);

template<typename R>
std::ostream& operator<<(std::ostream&, const Mat<R>&);

// define hash code for matrices:
namespace std {
    template <typename R> struct hash<Mat<R>> {
        std::size_t operator()(const Mat<R>&) const;
    };
}

namespace utils {

    template<typename R>
    void save_matrices(std::vector<Mat<R>>, std::string);

    template<typename R>
    void load_matrices(std::vector<Mat<R>>, std::string);

}

template <typename R>
bool operator!=(const Mat<R>&, const Mat<R>&);

template <typename R>
bool operator==(const Mat<R>&, const Mat<R>&);

#endif
