#include "dali/mat/math/Weights.h"
#include "mshadow/random.h"
#include "dali/mat/math/__MatMacros__.h"
#include "dali/mat/math/TensorOps.h"
#include "dali/mat/math/SynchronizedTensor.h"

template<typename R>
typename weights<R>::initializer_t weights<R>::empty() {
    return [](SynchronizedTensor<R>&){};
};

template<typename R>
typename weights<R>::initializer_t weights<R>::zeros() {
    return [](SynchronizedTensor<R>& matrix){
        DALI_FUNCTION_1_MUT(TensorOps::fill, matrix, (R)0.0);
    };
};

template<typename R>
typename weights<R>::initializer_t weights<R>::eye(R diag) {
    return [diag](SynchronizedTensor<R>& matrix) {
        // assert2(matrix.dims(0) == matrix.dims(1), "Identity initialization requires square matrix.");
        // auto& mat = GET_MAT(matrix);
        // mat.fill(0);
        // for (int i = 0; i < matrix.dims(0); i++)
        //     mat(i,i) = diag;
    };
};

template<typename R>
typename weights<R>::initializer_t weights<R>::uniform(R lower, R upper) {
    return [lower, upper](SynchronizedTensor<R>& matrix){
        DALI_FUNCTION_1_MUT(TensorOps::fill_uniform, matrix, lower, upper);
    };
};

template<typename R>
typename weights<R>::initializer_t weights<R>::uniform(R bound) {
    return uniform(-bound/2.0, bound/2.0);
}

template<typename R>
typename weights<R>::initializer_t weights<R>::gaussian(R mean, R std) {
    return [mean, std](SynchronizedTensor<R>& matrix){
        // std::default_random_engine generator;
        // std::normal_distribution<R> distribution(mean, std);
        // std::random_device rd;
        // generator.seed(rd());
        // auto randn = [&distribution, &generator] (int) {return distribution(generator);};
        // GET_MAT(matrix) = MatInternal<R>::eigen_mat::NullaryExpr(
        //     matrix.dims(0),
        //     matrix.dims(1),
        //     randn);
    };
};

template<typename R>
typename weights<R>::initializer_t weights<R>::gaussian(R std) {
    return gaussian(0.0, std);
}

template<typename R>
typename weights<R>::initializer_t weights<R>::svd(initializer_t preinitializer) {
    return [preinitializer](SynchronizedTensor<R>& matrix) {
        // assert(matrix.dims().size() == 2);
        // preinitializer(matrix);
        // auto svd = GET_MAT(matrix).jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
        // int n = matrix.dims(0);
        // int d = matrix.dims(1);
        // if (n < d) {
        //     GET_MAT(matrix) = svd.matrixV().block(0, 0, n, d);
        // } else {
        //     GET_MAT(matrix) = svd.matrixU().block(0, 0, n, d);
        // }
    };
}

template struct weights<float>;
template struct weights<double>;
