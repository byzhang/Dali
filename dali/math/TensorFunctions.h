#ifndef DALI_MATH_TENSOR_FUNCTIONS_H
#define DALI_MATH_TENSOR_FUNCTIONS_H

#include "dali/config.h"
#include <mshadow/tensor.h>
#include <math.h>

#ifdef DALI_USE_CUDA
    #define TANH_F tanhf
    #define LOG_F  logf
    #define EXP_F  expf
    #define POW_F  powf
#else
    #define TANH_F std::tanh
    #define LOG_F  std::log
    #define EXP_F  std::exp
    #define POW_F  pow
#endif

#define EPS 1e-6


namespace TensorOps {
    template<typename T>
    struct near_equal {
        T tol;
        near_equal(T _tol) : tol(_tol) {}
        MSHADOW_XINLINE bool operator()(const T& lhs, const T& rhs) const {
            return std::abs(lhs - rhs) < tol;
        }
    };

    namespace op {
        template<typename R>
        struct square {
            MSHADOW_XINLINE static R Map(const R& a) {
                return a * a;
            }
        };

        template<>
        MSHADOW_XINLINE
        int square<int>::Map(const int& a) {
            return a*a;
        }

        template<typename R>
        struct sqrt_f {
            MSHADOW_XINLINE static R Map(const R& a) {
                return sqrt(a);
            }
        };

        template<>
        MSHADOW_XINLINE
        int sqrt_f<int>::Map(const int& a) {
            return (int)sqrt((float)a);
        }


        template<typename R>
        struct inv {
            MSHADOW_XINLINE static R Map(const R& a) {
                return ((R)1.0) / a;
            }
        };

        template<typename R>
        struct sigmoid {
            MSHADOW_XINLINE static R Map(const R& a) {
                return 1.0 / (1.0 + EXP_F(-a));
            }
        };

        template<typename R>
        struct log {
            MSHADOW_XINLINE static R Map(const R& a) {
                return LOG_F(a);
            }
        };

        template<typename R>
        struct safe_entropy_log {
            MSHADOW_XINLINE static R Map(const R& a) {
                R a_safe = a;
                const R lower_bound = (R)EPS;
                const R upper_bound = (R)(1.0 - EPS);
                if (a_safe > upper_bound) {
                    a_safe = upper_bound;
                }
                if (a_safe < lower_bound) {
                    a_safe = lower_bound;
                }
                return LOG_F(a_safe);
            }
        };

        template<typename R>
        struct exp {
            MSHADOW_XINLINE static R Map(const R& a) {
                return EXP_F(a);
            }
        };

        template<typename R>
        struct div_grad {
            MSHADOW_XINLINE static R Map(const R& a, const R& b) {
                return a / (b * b);
            }
        };

        template<typename R>
        struct dsigmoid {
            MSHADOW_XINLINE static R Map(const R& a) {
                return a * (((R)1.0) - a);
            }
        };

        template<typename R>
        struct tanh {
            MSHADOW_XINLINE static R Map(const R& a) {
                // if (a < -30.0f)
                //     return -1.0f;
                // if (a >  30.0f)
                //     return 1.0f;
                return TANH_F(a);
            }
        };

        template<typename R>
        struct dtanh {
            MSHADOW_XINLINE static R Map(const R& a) {
                return 1.0 - a * a;
            }
        };

        template<typename R>
        struct power {
            MSHADOW_XINLINE static R Map(const R& a, const R& b) {
                return POW_F(a, b);
            }
        };

        template<typename R>
        struct abs {
            MSHADOW_XINLINE static R Map(const R& a) {
                return std::abs(a);
            }
        };

        template<typename R>
        struct log_or_zero {
            MSHADOW_XINLINE static R Map(const R& a) {
                return a > 0 ? LOG_F(a) : 0;
            }
        };

        template<typename R>
        struct sign {
            MSHADOW_XINLINE static R Map(const R& x) {
                return x > 0.0 ? 1.0 : -1.0;
            }
        };

        template<typename R>
        struct threshold {
            MSHADOW_XINLINE static R Map(const R& a, const R& b) {
                return a < b ? 1.0 : 0.0;
            }
        };

        template<typename R>
        struct max_scalar {
            MSHADOW_XINLINE static R Map(const R& x, const R& y) {
                return x > y ? x : y;
            }
        };

        template<typename R>
        struct  max_scalar_mask {
            MSHADOW_XINLINE static R Map(const R& m, const R& lower_bound) {
                return (m >= lower_bound) ? 1.0 : 0.0;
            }
        };

        template<typename R>
        struct  steep_sigmoid {
            MSHADOW_XINLINE static R Map(const R& x, const R& aggressiveness) {
                return 1.0 / (1.0 + EXP_F( - aggressiveness * x));
            }
        };

        template<typename R>
        struct  steep_sigmoid_backward {
            MSHADOW_XINLINE static R Map(const R& x, const R& aggressiveness) {
                return aggressiveness * (x - x * x);
            }
        };

        template<typename R>
        struct relu {
            MSHADOW_XINLINE static R Map(const R& x) {
                return x > 0.0 ? x : 0.0;
            }
        };
        template<typename R>
        struct relu_backward {
            MSHADOW_XINLINE static R Map(const R& x) {
                return x > 0.0 ? 1.0 : 0.0;
            }
        };

        template<typename R>
        struct minimum_backward {
            MSHADOW_XINLINE static R Map(const R& redres, const R& redsrc) {
                return redres == redsrc ? 1: 0;
            }
        };
        // identical to minimum backward
        template<typename R>
        struct maximum_backward {
            MSHADOW_XINLINE static R Map(const R& redres, const R& redsrc) {
                return redres == redsrc ? 1: 0;
            }
        };

        template<typename R>
        struct clip {
            MSHADOW_XINLINE static R Map(const R& x, const R& clipping_val) {
                if (x > clipping_val) {
                    return clipping_val;
                } else if (x < -clipping_val) {
                    return -clipping_val;
                } else {
                    return x;
                }
            }
        };

        template<typename R>
        struct binary_cross_entropy {
            MSHADOW_XINLINE static R Map(const R& x, const R& t ) {
                R distance_from1 =        t  * LOG_F(x);
                R distance_from0 = (1.0 - t) * LOG_F(1. - x);
                return -(distance_from1 + distance_from0);
            }
        };

        template<typename R>
        struct binary_cross_entropy_grad {
            MSHADOW_XINLINE static R Map(const R& x, const R& t ) {
                R numerator   = t - x;
                R denominator = (x * (x - 1.0));
                return numerator / denominator;
            }
        };

        template<typename R>
        struct softplus {
            MSHADOW_XINLINE static R Map(const R& x) {
                if (x > 20.0) {
                    return x;
                } else {
                    return LOG_F((R)1.0 + EXP_F(x));
                }
            }
        };

        template<typename R>
        struct softplus_backward {
            MSHADOW_XINLINE static R Map(const R& x) {
                if (x > 40.0) {
                    return 1.0;
                } else {
                    return EXP_F(x) / ((R)1.0 + EXP_F(x));
                }
            }
        };

    }
} //namespace TensorOps

#endif
