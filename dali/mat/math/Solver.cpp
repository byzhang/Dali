#include "Solver.h"

#include "dali/mat/math/__MatMacros__.h"

using std::vector;
#define PARAM_KEY_FOR_LOOKUP_TABLE param.id()

DEFINE_string(solver, "adadelta", "What solver to use (adadelta, sgd, adam, rmsprop, adagrad)");
DEFINE_double(learning_rate, 0.01, "Learning rate for SGD and Adagrad.");

namespace Solver {

    #define DECLARE_TYPENAME_GETTER(CLASSNAME, METHODNAME, IS_CLASSNAME) \
    template<typename R> \
    bool CLASSNAME<R>::is_ ##METHODNAME () const {\
        return IS_CLASSNAME;\
    }

    /* Abstract Solver */
    template<typename R>
    AbstractSolver<R>::AbstractSolver() : clipval(std::numeric_limits<R>::infinity),
     smooth_eps(SMOOTH_DEFAULT), regc(0.0) {}

    template<typename R>
    AbstractSolver<R>::AbstractSolver(R _clipval, R _smooth_eps, R _regc) : clipval(_clipval),
     smooth_eps(_smooth_eps), regc(_regc) {}

    template<typename R>
    void AbstractSolver<R>::reset_caches(vector<Mat<R>>& parameters) {}

    DECLARE_TYPENAME_GETTER(AbstractSolver, sgd,      false)
    DECLARE_TYPENAME_GETTER(AbstractSolver, adagrad,  false)
    DECLARE_TYPENAME_GETTER(AbstractSolver, adam,     false)
    DECLARE_TYPENAME_GETTER(AbstractSolver, rmsprop,  false)
    DECLARE_TYPENAME_GETTER(AbstractSolver, adadelta, false)

    /* SGD */
    template<typename R>
    SGD<R>::SGD (R clipval, R regc) : AbstractSolver<R>(clipval, 0.0, regc) {};

    template<typename R>
    SGD<R>::SGD (
        vector<Mat<R>>& parameters,
        R clipval,
        R regc) : AbstractSolver<R>(clipval, 0.0, regc) {};

    template<typename R>
    void SGD<R>::step (vector<Mat<R>>& parameters, R step_size) {
        // for (auto& param : parameters) {
        //     DEBUG_ASSERT_NOT_NAN(GET_MAT(param));
        //     if (param.sparse) {
        //         for (auto& i : *(param.sparse_row_keys)) {
        //             if (this->regc > 0) {
        //                 GET_MAT(param).row(i) -= (step_size * GET_GRAD(param).row(i).array().min(this->clipval).max(-this->clipval)).matrix() - (this->regc * GET_MAT(param).row(i));
        //             } else {
        //                 GET_MAT(param).row(i) -= (step_size * GET_GRAD(param).row(i).array().min(this->clipval).max(-this->clipval)).matrix();
        //             }
        //             // reset gradient
        //             GET_GRAD(param).row(i).fill(0);
        //         }
        //     } else {
        //         if (this->regc > 0) {
        //             GET_MAT(param) -= (step_size * GET_GRAD(param).array().min(this->clipval).max(-this->clipval)).matrix() - (this->regc * GET_MAT(param));
        //         } else {
        //             GET_MAT(param) -= (step_size * GET_GRAD(param).array().min(this->clipval).max(-this->clipval)).matrix();
        //         }
        //         // reset gradient
        //         GET_GRAD(param).fill(0);
        //     }
        //     DEBUG_ASSERT_NOT_NAN(GET_MAT(param));
        // }
    }

    template<typename R>
    void SGD<R>::step (vector<Mat<R>>& parameters) {
        return step(parameters, this->step_size);
    }

    template class SGD<float>;
    template class SGD<double>;

    DECLARE_TYPENAME_GETTER(SGD, sgd, true)

    /* AdaGrad */
    template<typename R>
    AdaGrad<R>::AdaGrad (
        R smooth_eps,
        R clipval,
        R regc) : AbstractSolver<R>(clipval, smooth_eps, regc) {};

    template<typename R>
    AdaGrad<R>::AdaGrad (
        vector<Mat<R>>& parameters,
        R smooth_eps,
        R clipval,
        R regc) : AbstractSolver<R>(clipval, smooth_eps, regc) {
        create_gradient_caches(parameters);
    };

    template<typename R>
    void AdaGrad<R>::create_gradient_caches(
            vector<Mat<R>>& parameters) {
        // for (auto& param : parameters) {
        //     // this operation should be run once unless
        //     // we expect the parameters of the model
        //     // to change online (probably not the case)

        //     if (gsums.count(PARAM_KEY_FOR_LOOKUP_TABLE) == 0) {
        //         auto new_cache = gsums.emplace(
        //             std::piecewise_construct,
        //         std::forward_as_tuple(PARAM_KEY_FOR_LOOKUP_TABLE),
        //         std::forward_as_tuple(param.dims(0), param.dims(1)));
        //         // initialize values for step cache to zero:
        //         new_cache.first->second.fill(0);
        //     }

        // }
    }

    template<typename R>
    void AdaGrad<R>::reset_caches(
            vector<Mat<R>>& parameters) {
        // for (auto& param : parameters) {
        //     auto& s = gsums[PARAM_KEY_FOR_LOOKUP_TABLE];
        //     s.fill(0);
        // }
    }

    template<typename R>
    void AdaGrad<R>::step(
            vector<Mat<R>>& parameters, R step_size) {
        // for (auto& param : parameters) {
        //     auto& s = gsums[PARAM_KEY_FOR_LOOKUP_TABLE];
        //     if (param.sparse) {
        //         for (auto& i : *(param.sparse_row_keys)) {
        //         if (this->regc > 0) {
        //             GET_GRAD(param).row(i) = GET_GRAD(param).row(i).array().min(this->clipval).max(-this->clipval).matrix() + (this->regc * GET_MAT(param).row(i));
        //         } else {
        //             GET_GRAD(param).row(i) = GET_GRAD(param).row(i).array().min(this->clipval).max(-this->clipval).matrix();
        //         }
        //         // update gradient cache using decay rule:
        //         s.row(i) += GET_GRAD(param).row(i).array().square().matrix();
        //         // clip the gradient to prevent explosions:
        //         // update gradient using RMSprop rule
        //         DEBUG_ASSERT_POSITIVE((s.row(i).array() + this->smooth_eps).matrix());
        //         GET_MAT(param).row(i) -= step_size * (GET_GRAD(param).row(i).array() / (s.row(i).array() + this->smooth_eps).sqrt() ).matrix();
        //         // reset gradient
        //         GET_GRAD(param).row(i).fill(0);
        //         }
        //     } else {
        //         if (this->regc > 0) {
        //             GET_GRAD(param) = GET_GRAD(param).array().min(this->clipval).max(-this->clipval).matrix() + (this->regc * GET_MAT(param));
        //         } else {
        //             GET_GRAD(param) = GET_GRAD(param).array().min(this->clipval).max(-this->clipval).matrix();
        //         }
        //         // update gradient cache using decay rule:
        //         s += GET_GRAD(param).array().square().matrix();
        //         // clip the gradient to prevent explosions:
        //         // update gradient using RMSprop rule
        //         DEBUG_ASSERT_POSITIVE((s.array() + this->smooth_eps).matrix());
        //         GET_MAT(param) -= step_size * (GET_GRAD(param).array() / (s.array() + this->smooth_eps).sqrt() ).matrix();
        //         // reset gradient
        //         GET_GRAD(param).fill(0);
        //     }
        //     DEBUG_ASSERT_NOT_NAN(GET_MAT(param));
        // }
    }

    template<typename R>
    void AdaGrad<R>::step(
        vector<Mat<R>>& parameters) {
        return step(parameters, this->step_size);
    }

    template class AdaGrad<float>;
    template class AdaGrad<double>;

    DECLARE_TYPENAME_GETTER(AdaGrad, adagrad, true)

    /* RMSProp */
    template<typename R>
    RMSProp<R>::RMSProp (
        R _decay_rate,
        R smooth_eps,
        R clipval,
        R regc) : AdaGrad<R>(clipval, smooth_eps, regc), decay_rate(_decay_rate) {};

    template<typename R>
    RMSProp<R>::RMSProp (
        vector<Mat<R>>& parameters,
        R _decay_rate,
        R smooth_eps,
        R clipval,
        R regc) : AdaGrad<R>(parameters, clipval, smooth_eps, regc), decay_rate(_decay_rate) {};

    template<typename R>
    void RMSProp<R>::step(
            vector<Mat<R>>& parameters,
            R step_size
            ) {
        // for (auto& param : parameters) {
        //     auto& s = this->gsums[PARAM_KEY_FOR_LOOKUP_TABLE];

        //     if (param.sparse) {
        //         for (auto& i : *(param.sparse_row_keys)) {
        //             s.row(i) = s.row(i) * decay_rate + (1.0 - decay_rate) * GET_GRAD(param).row(i).array().square().matrix();
        //             // clip the gradient to prevent explosions:
        //             if (this->regc > 0) {
        //                 GET_GRAD(param).row(i) = GET_GRAD(param).row(i).array().min(this->clipval).max(-this->clipval).matrix() + (this->regc * GET_MAT(param).row(i));
        //             } else {
        //                 GET_GRAD(param).row(i) = GET_GRAD(param).row(i).array().min(this->clipval).max(-this->clipval).matrix();
        //             }
        //             // update gradient using RMSprop rule
        //             DEBUG_ASSERT_POSITIVE((s.row(i).array() + this->smooth_eps).matrix());
        //             GET_MAT(param).row(i) -= step_size * (GET_GRAD(param).row(i).array() / (s.row(i).array() + this->smooth_eps).sqrt() ).matrix()  - (this->regc * GET_MAT(param).row(i));
        //             // reset gradient
        //             GET_GRAD(param).row(i).fill(0);
        //         }
        //     } else {
        //         s = s * decay_rate + (1.0 - decay_rate) * GET_GRAD(param).array().square().matrix();
        //         // clip the gradient to prevent explosions:
        //         if (this->regc > 0) {
        //             GET_GRAD(param) = GET_GRAD(param).array().min(this->clipval).max(-this->clipval).matrix() + (this->regc * GET_MAT(param));
        //         } else {
        //             GET_GRAD(param) = GET_GRAD(param).array().min(this->clipval).max(-this->clipval).matrix();
        //         }
        //         // update gradient using RMSprop rule
        //         DEBUG_ASSERT_POSITIVE((s.array() + this->smooth_eps).matrix());
        //         GET_MAT(param) -= step_size * (GET_GRAD(param).array() / (s.array() + this->smooth_eps).sqrt() ).matrix()  - (this->regc * GET_MAT(param));
        //         // reset gradient
        //         GET_GRAD(param).fill(0);
        //     }
        //     DEBUG_ASSERT_NOT_NAN(GET_MAT(param));
        // }
    }

    template<typename R>
    void RMSProp<R>::step(
        vector<Mat<R>>& parameters) {
        return step(parameters, step_size);
    }

    template class RMSProp<float>;
    template class RMSProp<double>;

    DECLARE_TYPENAME_GETTER(RMSProp, rmsprop, true)

    /* AdaDelta */
    template<typename R>
    AdaDelta<R>::AdaDelta (
        R _rho,
        R smooth_eps,
        R clipval,
        R regc) : AbstractSolver<R>(clipval, smooth_eps, regc), rho(_rho) {};

    template<typename R>
    AdaDelta<R>::AdaDelta (
        vector<Mat<R>>& parameters,
        R _rho,
        R smooth_eps,
        R clipval,
        R regc) : AbstractSolver<R>(clipval, smooth_eps, regc), rho(_rho) {
        create_gradient_caches(parameters);
    };

    template<typename R>
    void AdaDelta<R>::create_gradient_caches(
            vector<Mat<R>>& parameters) {
        // for (auto& param : parameters) {
        //     // this operation should be run once unless
        //     // we expect the parameters of the model
        //     // to change online (probably not the case)
        //     if (!(gsums.count(PARAM_KEY_FOR_LOOKUP_TABLE) > 0)) {
        //         auto new_cache = gsums.emplace(
        //             std::piecewise_construct,
        //         std::forward_as_tuple(PARAM_KEY_FOR_LOOKUP_TABLE),
        //         std::forward_as_tuple(param.dims(0), param.dims(1)));
        //         // initialize values for step cache to zero:
        //         new_cache.first->second.fill(0);

        //         new_cache = xsums.emplace(
        //             std::piecewise_construct,
        //         std::forward_as_tuple(PARAM_KEY_FOR_LOOKUP_TABLE),
        //         std::forward_as_tuple(param.dims(0), param.dims(1)));
        //         // initialize values for step cache to zero:
        //         new_cache.first->second.fill(0);
        //     }
        // }
    }

    template<typename R>
    void AdaDelta<R>::reset_caches(
            vector<Mat<R>>& parameters) {
        // for (auto& param : parameters) {
        //     auto& s = gsums[PARAM_KEY_FOR_LOOKUP_TABLE];
        //     s.fill(0);
        //     auto& x = xsums[PARAM_KEY_FOR_LOOKUP_TABLE];
        //     x.fill(0);
        // }
    }

    template<typename R>
    void AdaDelta<R>::step (vector<Mat<R>>& parameters) {
        // for (auto& param : parameters) {
        //     auto& gsum = gsums[PARAM_KEY_FOR_LOOKUP_TABLE];
        //     auto& xsum = xsums[PARAM_KEY_FOR_LOOKUP_TABLE];
        //     if (param.sparse) {
        //         for (auto& i : *(param.sparse_row_keys)) {
        //             if (this->regc > 0) {
        //                 GET_GRAD(param).row(i) = GET_GRAD(param).row(i).array().min(this->clipval).max(-this->clipval).matrix() + (this->regc * GET_MAT(param).row(i));
        //             } else {
        //                 GET_GRAD(param).row(i) = GET_GRAD(param).row(i).array().min(this->clipval).max(-this->clipval).matrix();
        //             }
        //             // update gradient cache using decay rule:
        //             DEBUG_ASSERT_POSITIVE(gsum.row(i).matrix());
        //             gsum.row(i) = (gsum.row(i) * rho) + ((1.0 - rho) * (GET_GRAD(param).row(i).array().square()).matrix());

        //             DEBUG_ASSERT_NOT_NAN(((gsum.row(i).array()  + this->smooth_eps).matrix()));
        //             DEBUG_ASSERT_POSITIVE(((gsum.row(i).array() + this->smooth_eps)).matrix());
        //             DEBUG_ASSERT_POSITIVE(((xsum.row(i).array() + this->smooth_eps) / (gsum.row(i).array() + this->smooth_eps)).matrix());
        //             auto dparam = -(((xsum.row(i).array() + this->smooth_eps) / (gsum.row(i).array() + this->smooth_eps)).sqrt() * GET_GRAD(param).row(i).array()).matrix();

        //             xsum.row(i) = (xsum.row(i) * rho) + ((1.0 - rho) * (dparam.array().square())).matrix();
        //             // update gradient using AdaDelta rule
        //             GET_MAT(param).row(i) += dparam;
        //             // reset gradient
        //             GET_GRAD(param).row(i).fill(0);
        //         }
        //     } else {
        //         if (this->regc > 0) {
        //             GET_GRAD(param) = GET_GRAD(param).array().min(this->clipval).max(-this->clipval).matrix() + (this->regc * GET_MAT(param));
        //         } else {
        //             GET_GRAD(param) = GET_GRAD(param).array().min(this->clipval).max(-this->clipval).matrix();
        //         }
        //         // update gradient cache using decay rule:
        //         gsum = (gsum * rho) + ((1.0 - rho) * (GET_GRAD(param).array().square()).matrix());
        //         DEBUG_ASSERT_POSITIVE((gsum.array()  + this->smooth_eps).matrix());
        //         DEBUG_ASSERT_POSITIVE(((xsum.array() + this->smooth_eps) / (gsum.array() + this->smooth_eps)).matrix());
        //         auto dparam = -(((xsum.array() + this->smooth_eps) / (gsum.array() + this->smooth_eps)).sqrt() * GET_GRAD(param).array()).matrix();

        //         xsum = (xsum * rho) + ((1.0 - rho) * (dparam.array().square())).matrix();
        //         // update gradient using AdaDelta rule
        //         GET_MAT(param) += dparam;
        //         // reset gradient
        //         GET_GRAD(param).fill(0);
        //     }
        //     DEBUG_ASSERT_NOT_NAN(GET_MAT(param));
        // }
    }

    template class AdaDelta<float>;
    template class AdaDelta<double>;

    DECLARE_TYPENAME_GETTER(AdaDelta, adadelta, true)

    /* Adam */
    template<typename R>
    Adam<R>::Adam (
        R _b1,
        R _b2,
        R smooth_eps,
        R clipval,
        R regc) : AbstractSolver<R>(clipval, smooth_eps, regc), b1(_b1), b2(_b2), epoch(0) {};

    template<typename R>
    Adam<R>::Adam (
        vector<Mat<R>>& parameters,
        R _b1,
        R _b2,
        R smooth_eps,
        R clipval,
        R regc) : AbstractSolver<R>(clipval, smooth_eps, regc), b1(_b1), b2(_b2), epoch(0) {
        create_gradient_caches(parameters);
    };

    template<typename R>
    void Adam<R>::create_gradient_caches(
            vector<Mat<R>>& parameters) {
        // for (auto& param : parameters) {
        //     // this operation should be run once unless
        //     // we expect the parameters of the model
        //     // to change online (probably not the case)
        //     if (!(gsums.count(PARAM_KEY_FOR_LOOKUP_TABLE) > 0)) {
        //         auto new_cache = gsums.emplace(
        //             std::piecewise_construct,
        //         std::forward_as_tuple(PARAM_KEY_FOR_LOOKUP_TABLE),
        //         std::forward_as_tuple(param.dims(0), param.dims(1)));
        //         // initialize values for step cache to zero:
        //         new_cache.first->second.fill(0);

        //         new_cache = xsums.emplace(
        //             std::piecewise_construct,
        //         std::forward_as_tuple(PARAM_KEY_FOR_LOOKUP_TABLE),
        //         std::forward_as_tuple(param.dims(0), param.dims(1)));
        //         // initialize values for step cache to zero:
        //         new_cache.first->second.fill(0);
        //     }
        // }
    }

    template<typename R>
    void Adam<R>::reset_caches(
            vector<Mat<R>>& parameters) {
        // for (auto& param : parameters) {
        //     auto& s = gsums[PARAM_KEY_FOR_LOOKUP_TABLE];
        //     s.fill(0);
        //     auto& x = xsums[PARAM_KEY_FOR_LOOKUP_TABLE];
        //     x.fill(0);
        // }
        // epoch = 0;
    }

    template<typename R>
    void Adam<R>::step (vector<Mat<R>>& parameters, R step_size) {
        // // increase timesteps:
        // epoch+=1;
        // // this affects the learning rate:
        // auto fix1 = 1.0 - std::pow(b1, epoch);
        // auto fix2 = 1.0 - std::pow(b2, epoch);
        // auto lr_t = step_size * sqrt(fix2 / fix1);

        // assert(lr_t == lr_t);

        // for (auto& param : parameters) {
        //     auto& m = gsums[PARAM_KEY_FOR_LOOKUP_TABLE];
        //     auto& v = xsums[PARAM_KEY_FOR_LOOKUP_TABLE];

        //     if (param.sparse) {
        //         for (auto& i : *(param.sparse_row_keys)) {
        //             GET_GRAD(param).row(i) = GET_GRAD(param).row(i).array().min(this->clipval).max(-this->clipval).matrix();
        //             // update m acculumulator
        //             m.row(i) = ((b1 * GET_GRAD(param).row(i).array()) + ((1. - b1) * m.row(i).array())).matrix();

        //             // update v acculumulator
        //             v.row(i) = ((b2 * GET_GRAD(param).row(i).array().square()) + ((1. - b2) * v.row(i).array())).matrix();

        //             // regularize using L2 norm:
        //             if (this->regc > 0) {
        //                 GET_GRAD(param).row(i)  = (m.row(i).array() / (v.row(i).array().sqrt() + this->smooth_eps)).matrix() + (this->regc * GET_MAT(param).row(i));
        //             } else {
        //                 GET_GRAD(param).row(i)  = (m.row(i).array() / (v.row(i).array().sqrt() + this->smooth_eps)).matrix();
        //             }

        //             // take gradient step
        //             GET_MAT(param).row(i) -= lr_t * GET_GRAD(param).row(i);

        //             // reset gradient
        //             GET_GRAD(param).row(i).fill(0);
        //         }
        //     } else {

        //         GET_GRAD(param) = GET_GRAD(param).array().min(this->clipval).max(-this->clipval).matrix();

        //         // update m acculumulator
        //         m = ((b1 * GET_GRAD(param).array()) + ((1. - b1) * m.array())).matrix();

        //         // update v acculumulator
        //         v = ((b2 * GET_GRAD(param).array().square()) + ((1. - b2) * v.array())).matrix();

        //         // regularize using L2 norm:
        //         if (this->regc > 0) {
        //             GET_GRAD(param)  = (m.array() / (v.array().sqrt() + this->smooth_eps)).matrix() + (this->regc * GET_MAT(param));
        //         } else {
        //             GET_GRAD(param)  = (m.array() / (v.array().sqrt() + this->smooth_eps)).matrix();
        //         }

        //         // take gradient step
        //         GET_MAT(param) -= lr_t * GET_GRAD(param);

        //         // reset gradient
        //         GET_GRAD(param).fill(0);
        //     }
        // }
    }


    template<typename R>
    void Adam<R>::step (vector<Mat<R>>& parameters) {
        return step(parameters, 0.0002);
    }

    template<typename R>
    std::shared_ptr<AbstractSolver<R>> construct(std::string solver_name, std::vector<Mat<R>>& params, R learning_rate, R regc) {
        std::shared_ptr<AbstractSolver<R>> solver;
        std::transform(solver_name.begin(), solver_name.end(), solver_name.begin(), ::tolower);
        if (solver_name        == "adadelta") {
            solver = std::make_shared<AdaDelta<R>>(params, 0.95, 1e-9, 100.0, regc);
        } else if (solver_name == "adam") {
            solver = std::make_shared<Adam<R>>(params, 0.1, 0.001, 1e-9, 100.0, regc);
        } else if (solver_name == "sgd") {
            solver = std::make_shared<SGD<R>>(params, 100.0, regc);
            dynamic_cast<SGD<R>*>(solver.get())->step_size = learning_rate;
        } else if (solver_name == "adagrad") {
            solver = std::make_shared<AdaGrad<R>>(params, 1e-9, 100.0, regc);
            dynamic_cast<Solver::AdaGrad<R>*>(solver.get())->step_size = learning_rate;
        } else if (solver_name == "rmsprop") {
            solver = std::make_shared<RMSProp<R>>(params, 0.999, 1e-9, 100.0, regc);
            dynamic_cast<Solver::RMSProp<R>*>(solver.get())->step_size = learning_rate;
        } else {
            utils::exit_with_message("Did not recognize this solver type.");
        }
        return solver;
    }

    template std::shared_ptr<AbstractSolver<float>> construct(std::string solver_name, std::vector<Mat<float>>& params, float learning_rate, float regc);
    template std::shared_ptr<AbstractSolver<double>> construct(std::string solver_name, std::vector<Mat<double>>& params, double learning_rate, double regc);

    template class Adam<float>;
    template class Adam<double>;

    DECLARE_TYPENAME_GETTER(Adam, adam, true)
}
