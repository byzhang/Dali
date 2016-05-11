#include "dali/tensor/Solver.h"
#include "dali/tensor/__MatMacros__.h"


using std::vector;
#define PARAM_KEY_FOR_LOOKUP_TABLE &MAT(param)



namespace Solver {
    bool nan_protection = true;

    template<typename R>
    void create_cache(Mat<R>& param,
            std::unordered_map<cache_key_t<R>, cache_t<R>>& caches) {
        if (caches.count(PARAM_KEY_FOR_LOOKUP_TABLE) == 0) {
            auto new_cache = caches.emplace(
                std::piecewise_construct,
            std::forward_as_tuple(PARAM_KEY_FOR_LOOKUP_TABLE),
            std::forward_as_tuple(param.dims(0), param.dims(1)));
            // initialize values for step cache to zero:
            new_cache.first->second.clear();
        }
    }

    template<typename R>
    void create_cache(vector<Mat<R>>& params,
            std::unordered_map<cache_key_t<R>, cache_t<R>>& caches) {
        for (auto& param: params) {
            create_cache(param, caches);
        }
    }

    template<typename R>
    void reset_cache(Mat<R>& param,
        std::unordered_map<cache_key_t<R>, cache_t<R>>& caches) {
        auto& s = caches.at(PARAM_KEY_FOR_LOOKUP_TABLE);
        s.clear();
    }

    template<typename R>
    void reset_cache(vector<Mat<R>>& params,
        std::unordered_map<cache_key_t<R>, cache_t<R>>& caches) {
        for (auto& param: params) {
            reset_cache(param, caches);
        }
    }


    /* Abstract Solver */
    template<typename R>
    AbstractSolver<R>::AbstractSolver() :
            clipval(std::numeric_limits<R>::infinity),
            smooth_eps(SMOOTH_DEFAULT),
            regc(0.0),
            method(METHOD_UNINITIALIZED) {
    }

    template<typename R>
    AbstractSolver<R>::AbstractSolver(R _clipval,
                                      R _smooth_eps,
                                      R _regc,
                                      Method _method) :
            clipval(_clipval),
            smooth_eps(_smooth_eps),
            regc(_regc),
            method(_method) {
    }

    template<typename R>
    void AbstractSolver<R>::reset_caches(vector<Mat<R>>& parameters) {}
    template<typename R>
    void AbstractSolver<R>::create_gradient_caches(vector<Mat<R>>& parameters) {}

    /* SGD */
    template<typename R>
    SGD<R>::SGD (R clipval, R regc) :
            AbstractSolver<R>(clipval, 0.0, regc, METHOD_SGD) {
    }

    template<typename R>
    SGD<R>::SGD (
        vector<Mat<R>>& parameters,
        R clipval,
        R regc) : AbstractSolver<R>(clipval, 0.0, regc, METHOD_SGD) {};

    template<typename R>
    void SGD<R>::step (vector<Mat<R>>& parameters, R step_size) {
        for (auto& param : parameters) {
            if (nan_protection && param.is_grad_nan()) {
                std::cout << "WARNING: Ignoring gradient update because of NaNs." << std::endl;
            } else {
                MatOps<R>::clip_and_regularize(param, this->clipval, this->regc);
                MatOps<R>::sgd_update(param, step_size);
            }
            // reset gradient
            GRAD(param).clear();
        }
    }

    template<typename R>
    void SGD<R>::step (vector<Mat<R>>& parameters) {
        return step(parameters, this->step_size);
    }

    template class SGD<float>;
    template class SGD<double>;

    /* AdaGrad */
    template<typename R>
    AdaGrad<R>::AdaGrad (
            R smooth_eps,
            R clipval,
            R regc) : AbstractSolver<R>(clipval, smooth_eps, regc, METHOD_ADAGRAD) {
    }

    template<typename R>
    AdaGrad<R>::AdaGrad (
            vector<Mat<R>>& parameters,
            R smooth_eps,
            R clipval,
            R regc) : AbstractSolver<R>(clipval, smooth_eps, regc, METHOD_ADAGRAD) {
        create_gradient_caches(parameters);
    }

    template<typename R>
    void AdaGrad<R>::create_gradient_caches(
            vector<Mat<R>>& parameters) {
        create_cache(parameters, gsums);
    }

    template<typename R>
    void AdaGrad<R>::reset_caches(
            vector<Mat<R>>& parameters) {
        reset_cache(parameters, gsums);
    }

    template<typename R>
    void AdaGrad<R>::step(
            vector<Mat<R>>& parameters, R step_size) {
        for (auto& param : parameters) {

            if (nan_protection && param.is_grad_nan()) {
                std::cout << "WARNING: Ignoring gradient update because of NaNs." << std::endl;
            } else {
                auto& s = gsums.at(PARAM_KEY_FOR_LOOKUP_TABLE);

                MatOps<R>::clip_and_regularize(param, this->clipval, this->regc);
                MatOps<R>::adagrad_update(param, s, step_size, this->smooth_eps);
            }

            // reset gradient
            GRAD(param).clear();
        }
    }

    template<typename R>
    void AdaGrad<R>::step(
        vector<Mat<R>>& parameters) {
        return step(parameters, this->step_size);
    }

    template class AdaGrad<float>;
    template class AdaGrad<double>;

    /* RMSProp */
    template<typename R>
    RMSProp<R>::RMSProp (
            R _decay_rate,
            R smooth_eps,
            R clipval,
            R regc) : AdaGrad<R>(smooth_eps, clipval, regc),
                      decay_rate(_decay_rate) {
        this->method = METHOD_RMSPROP;
    }

    template<typename R>
    RMSProp<R>::RMSProp (
            vector<Mat<R>>& parameters,
            R _decay_rate,
            R smooth_eps,
            R clipval,
            R regc) : AdaGrad<R>(parameters, smooth_eps, clipval, regc),
                      decay_rate(_decay_rate) {
        this->method = METHOD_RMSPROP;
    }

    template<typename R>
    void RMSProp<R>::step(
            vector<Mat<R>>& parameters,
            R step_size
            ) {
        for (auto& param : parameters) {

            if (nan_protection && param.is_grad_nan()) {
                std::cout << "WARNING: Ignoring gradient update because of NaNs." << std::endl;
            } else {
                auto& s = this->gsums[PARAM_KEY_FOR_LOOKUP_TABLE];

                MatOps<R>::clip_and_regularize(param, this->clipval, this->regc);
                MatOps<R>::rmsprop_update(param, s, decay_rate, step_size, this->smooth_eps);
            }

            // reset gradient
            GRAD(param).clear();
            DEBUG_ASSERT_NOT_NAN(MAT(param));
        }
    }

    template<typename R>
    void RMSProp<R>::step(
        vector<Mat<R>>& parameters) {
        return step(parameters, step_size);
    }

    template class RMSProp<float>;
    template class RMSProp<double>;

    /* RMSPropMomentum */
    template<typename R>
    RMSPropMomentum<R>::RMSPropMomentum (
            R decay_rate,
            R momentum,
            R step_size,
            R smooth_eps,
            R clipval,
            R regc) : AbstractSolver<R>(clipval, smooth_eps, regc, METHOD_RMSPROPMOMENTUM),
                      decay_rate(decay_rate),
                      momentum(momentum),
                      step_size(step_size) {
    }

    template<typename R>
    RMSPropMomentum<R>::RMSPropMomentum (
            vector<Mat<R>>& parameters,
            R decay_rate,
            R momentum,
            R step_size,
            R smooth_eps,
            R clipval,
            R regc) : AbstractSolver<R>(clipval, smooth_eps, regc, METHOD_RMSPROPMOMENTUM),
                      decay_rate(decay_rate),
                      momentum(momentum),
                      step_size(step_size) {
        create_gradient_caches(parameters);
    }


    template<typename R>
    void RMSPropMomentum<R>::create_gradient_caches(
            vector<Mat<R>>& parameters) {
        create_cache(parameters, n_cache);
        create_cache(parameters, g_cache);
        create_cache(parameters, momentum_cache);
    }

    template<typename R>
    void RMSPropMomentum<R>::reset_caches(
            vector<Mat<R>>& parameters) {

        reset_cache(parameters, n_cache);
        reset_cache(parameters, g_cache);
        reset_cache(parameters, momentum_cache);
    }

    template<typename R>
    void RMSPropMomentum<R>::step(
            vector<Mat<R>>& parameters, R step_size_override) {
        for (auto& param : parameters) {

            if (nan_protection && param.is_grad_nan()) {
                std::cout << "WARNING: Ignoring gradient update because of NaNs." << std::endl;
            } else {
                auto& n = n_cache.at(PARAM_KEY_FOR_LOOKUP_TABLE);
                auto& g = g_cache.at(PARAM_KEY_FOR_LOOKUP_TABLE);
                auto& m = momentum_cache.at(PARAM_KEY_FOR_LOOKUP_TABLE);

                MatOps<R>::clip_and_regularize(param, this->clipval, this->regc);
                MatOps<R>::rmsprop_momentum_update(
                        param, n, g, m, decay_rate, momentum,
                        step_size_override, this->smooth_eps);
            }

            // reset gradient
            GRAD(param).clear();
        }
    }

    template<typename R>
    void RMSPropMomentum<R>::step(vector<Mat<R>>& parameters) {
        return step(parameters, this->step_size);
    }

    template class RMSPropMomentum<float>;
    template class RMSPropMomentum<double>;


    /* AdaDelta */
    template<typename R>
    AdaDelta<R>::AdaDelta (
            R _rho,
            R smooth_eps,
            R clipval,
            R regc) : AbstractSolver<R>(clipval, smooth_eps, regc, METHOD_ADADELTA),
                      rho(_rho) {
    }

    template<typename R>
    AdaDelta<R>::AdaDelta (
            vector<Mat<R>>& parameters,
            R _rho,
            R smooth_eps,
            R clipval,
            R regc) : AbstractSolver<R>(clipval, smooth_eps, regc, METHOD_ADADELTA),
                      rho(_rho) {
        create_gradient_caches(parameters);
    }

    template<typename R>
    void AdaDelta<R>::create_gradient_caches(
            vector<Mat<R>>& parameters) {
        create_cache(parameters, gsums);
        create_cache(parameters, xsums);
    }

    template<typename R>
    void AdaDelta<R>::reset_caches(
            vector<Mat<R>>& parameters) {
        reset_cache(parameters, gsums);
        reset_cache(parameters, xsums);
    }


    template<typename R>
    void AdaDelta<R>::step (vector<Mat<R>>& parameters) {
        for (auto& param : parameters) {
            if (nan_protection && param.is_grad_nan()) {
                std::cout << "WARNING: Ignoring gradient update because of NaNs." << std::endl;
            } else {
                auto& gsum = gsums[PARAM_KEY_FOR_LOOKUP_TABLE];
                auto& xsum = xsums[PARAM_KEY_FOR_LOOKUP_TABLE];

                MatOps<R>::clip_and_regularize(param, this->clipval, this->regc);
                MatOps<R>::adadelta_update(param, gsum, xsum, rho, this->smooth_eps);
            }

            // reset gradient
            GRAD(param).clear();

            DEBUG_ASSERT_NOT_NAN(GET_MAT(param));
        }
    }

    template class AdaDelta<float>;
    template class AdaDelta<double>;

    /* Adam */
    template<typename R>
    Adam<R>::Adam (
            R _step_size,
            R _b1,
            R _b2,
            R smooth_eps,
            R clipval,
            R regc) : AbstractSolver<R>(clipval, smooth_eps, regc, METHOD_ADAM),
                      step_size(_step_size), b1(_b1), b2(_b2), epoch(0) {
    }

    template<typename R>
    Adam<R>::Adam (
            vector<Mat<R>>& parameters,
            R _step_size,
            R _b1,
            R _b2,
            R smooth_eps,
            R clipval,
            R regc) : AbstractSolver<R>(clipval, smooth_eps, regc, METHOD_ADAM),
                      step_size(_step_size), b1(_b1), b2(_b2), epoch(0) {
        create_gradient_caches(parameters);
    }

    template<typename R>
    void Adam<R>::create_gradient_caches(
            vector<Mat<R>>& parameters) {
        create_cache(parameters, gsums);
        create_cache(parameters, xsums);
    }

    template<typename R>
    void Adam<R>::reset_caches(
            vector<Mat<R>>& parameters) {
        reset_cache(parameters, gsums);
        reset_cache(parameters, xsums);
        epoch = 0;
    }

    template<typename R>
    void Adam<R>::step (vector<Mat<R>>& parameters, R step_size) {
        // increase timesteps:
        epoch += 1;

        for (auto& param : parameters) {
            if (nan_protection && param.is_grad_nan()) {
                std::cout << "WARNING: Ignoring gradient update because of NaNs." << std::endl;
            } else {
                auto& m = gsums[PARAM_KEY_FOR_LOOKUP_TABLE];
                auto& v = xsums[PARAM_KEY_FOR_LOOKUP_TABLE];

                MatOps<R>::clip_and_regularize(param, this->clipval, this->regc);

                MatOps<R>::adam_update(param, m, v, b1, b2, this->smooth_eps, step_size, epoch);
            }

            // reset gradient
            GRAD(param).clear();
        }
    }


    template<typename R>
    void Adam<R>::step (vector<Mat<R>>& parameters) {
        return step(parameters, step_size);
    }

    template<typename R>
    std::shared_ptr<AbstractSolver<R>> construct(
            std::string solver_name,
            std::vector<Mat<R>>& params,
            R step_size,
            R regc) {
        std::shared_ptr<AbstractSolver<R>> solver;
        std::transform(solver_name.begin(), solver_name.end(), solver_name.begin(), ::tolower);
        if (solver_name        == "adadelta") {
            solver = std::make_shared<AdaDelta<R>>(params, 0.95, 1e-4, 100.0, regc);
        } else if (solver_name == "adam") {
            solver = std::make_shared<Adam<R>>(params, step_size, 0.55, 1e-6, 1e-9, 100.0, regc);
        } else if (solver_name == "sgd") {
            solver = std::make_shared<SGD<R>>(params, 100.0, regc);
            dynamic_cast<SGD<R>*>(solver.get())->step_size = step_size;
        } else if (solver_name == "adagrad") {
            solver = std::make_shared<AdaGrad<R>>(params, 1e-9, 100.0, regc);
            dynamic_cast<Solver::AdaGrad<R>*>(solver.get())->step_size = step_size;
        } else if (solver_name == "rmsprop") {
            solver = std::make_shared<RMSProp<R>>(params, 0.999, 1e-9, 100.0, regc);
            dynamic_cast<Solver::RMSProp<R>*>(solver.get())->step_size = step_size;
        } else if (solver_name == "rmspropmomentum") {
            solver = std::make_shared<RMSPropMomentum<R>>(params);
            dynamic_cast<Solver::RMSPropMomentum<R>*>(solver.get())->step_size = step_size;
            dynamic_cast<Solver::RMSPropMomentum<R>*>(solver.get())->regc = regc;
        } else {
            utils::exit_with_message("Did not recognize this solver type.");
        }
        return solver;
    }

    template std::shared_ptr<AbstractSolver<float>> construct(std::string solver_name, std::vector<Mat<float>>& params, float learning_rate, float regc);
    template std::shared_ptr<AbstractSolver<double>> construct(std::string solver_name, std::vector<Mat<double>>& params, double learning_rate, double regc);

    template class Adam<float>;
    template class Adam<double>;

}
