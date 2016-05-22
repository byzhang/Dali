#ifndef DALI_ARRAY_LAZY_BINARY_H
#define DALI_ARRAY_LAZY_BINARY_H

#include "dali/array/functor.h"

template<template<class>class Functor, typename LeftT, typename RightT>
struct LazyBinary;

namespace lazy {
    template <typename T, typename T2>
    LazyBinary<functor::add, T, T2> add(T a, T2 b);

    template <typename T, typename T2>
    LazyBinary<functor::sub, T, T2> sub(T a, T2 b);

    template <typename T, typename T2>
    LazyBinary<functor::eltmul, T, T2> eltmul(T a, T2 b);

    template <typename T, typename T2>
    LazyBinary<functor::eltdiv, T, T2> eltdiv(T a, T2 b);

    template <typename T, typename T2>
    LazyBinary<functor::power, T, T2> pow(T a, T2 b);

    template <typename T, typename T2>
    LazyBinary<functor::max_scalar, T, T2> max(T a, T2 b);

    template <typename T, typename T2>
    LazyBinary<functor::min_scalar, T, T2> min(T a, T2 b);

    template <typename T, typename T2>
    LazyBinary<functor::binary_cross_entropy, T, T2> binary_cross_entropy(T a, T2 b);

    template <typename T, typename T2>
    LazyBinary<functor::binary_cross_entropy_grad, T, T2> binary_cross_entropy_grad(T a, T2 b);


}  // namespace lazy

#include "dali/array/lazy/binary-impl.h"

#endif