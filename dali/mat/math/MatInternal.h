#ifndef DALI_MAT_MATH_MAT_INTERNAL_H
#define DALI_MAT_MATH_MAT_INTERNAL_H

#include <atomic>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <mshadow/tensor.h>

typedef unsigned int dim_t;

template<typename R>
class MatInternal {
    private:
        static std::atomic<int> next_matrix;
    public:
        typedef mshadow::Tensor<mshadow::gpu, 2, R> mat_storage_t;

        mat_storage_t w;

        std::vector<dim_t> dims;
        const size_t id;

        MatInternal(dim_t n, dim_t d, bool empty=false);
        MatInternal(const MatInternal<R>& m);

        ~MatInternal();

        R& operator()(int i, int j);
        R operator()(int i, int j) const;

        R& operator()(int i);
        R operator()(int i) const;

        const R* data() const;
        R* data();

        void print() const;

        void clear();

        operator mat_storage_t();
};

template<typename R>
class GradInternal {
    public:
        typedef mshadow::Tensor<mshadow::gpu, 2, R> mat_storage_t;

        mat_storage_t dw;

        GradInternal(dim_t n, dim_t d, bool empty=true);
        GradInternal(const GradInternal<R>& g);

        ~GradInternal();


        R& operator()(int i, int j);
        R operator()(int i, int j) const;

        R& operator()(int i);
        R operator()(int i) const;

        const R* data() const;
        R* data();

        void clear();

        operator mat_storage_t();
};

#endif
