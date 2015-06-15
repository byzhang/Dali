#include "dali/mat/math/Weights.h"

#include "mshadow/random.h"

#include "dali/mat/math/__MatMacros__.h"

using utils::assert2;

template<typename R>
typename weights<R>::initializer_t weights<R>::uninitialized() {
    return [](Mat<R>&){};
};

template<typename R>
typename weights<R>::initializer_t weights<R>::zeros() {
    return [](Mat<R>& matrix){
        tensor_fill(MAT(matrix), 0);
    };
};

template<typename R>
typename weights<R>::initializer_t weights<R>::eye(R diag) {
    return [diag](Mat<R>& matrix){
        // assert2(matrix.dims(0) == matrix.dims(1), "Identity initialization requires square matrix.");
        // auto& mat = GET_MAT(matrix);
        // mat.fill(0);
        // for (int i = 0; i < matrix.dims(0); i++)
        //     mat(i,i) = diag;
    };
};

TENSOR_TEMPLATE
void uniform_fill_helper(mshadow::Tensor<Device,dims,R>& t, R lower, R upper) {
    std::random_device rd;
    mshadow::Random<Device, R> generator((int)rd());
    generator.SampleUniform(&t, lower, upper);
}

template<typename R>
typename weights<R>::initializer_t weights<R>::uniform(R lower, R upper) {
    return [lower, upper](Mat<R>& matrix){
        DALI_FUNCTION_1_MUT(uniform_fill_helper,
                            MAT(matrix),
                            lower,
                            upper);
    };
};

template<typename R>
typename weights<R>::initializer_t weights<R>::uniform(R bound) {
    return uniform(-bound/2.0, bound/2.0);
}

template<typename R>
typename weights<R>::initializer_t weights<R>::gaussian(R mean, R std) {
    return [mean, std](Mat<R>& matrix){
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
    return [preinitializer](Mat<R>& matrix) {
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
