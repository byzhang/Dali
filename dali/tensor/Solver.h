#ifndef SOLVER_MAT_H
#define SOLVER_MAT_H

#include "dali/tensor/Mat.h"
#include "dali/utils/core_utils.h"

#define SOLVER_MAT_DEFAULT_STEP_SIZE_H 0.035

namespace Solver {
    template<typename R>
    using cache_key_t = void*;
    template<typename R>
    using cache_t = Mat<R>;

    enum Method {
        METHOD_UNINITIALIZED,
        METHOD_ADAGRAD,
        METHOD_ADADELTA,
        METHOD_SGD,
        METHOD_RMSPROP,
        METHOD_ADAM,
        METHOD_RMSPROPMOMENTUM,
    };

    extern bool nan_protection;

    const double SMOOTH_DEFAULT = 1e-4;

    template<typename R> class AbstractSolver {
        public:
            Method method;

            R clip_abs;
            R clip_norm;

            R smooth_eps;
            R regc;
            AbstractSolver();
            AbstractSolver(R clip_norm, R smooth_eps, R regc, Method method);
            virtual void step( std::vector<Mat<R>>& ) = 0;
            virtual void reset_caches( std::vector<Mat<R>>&);
            virtual void create_gradient_caches(std::vector<Mat<R>>&);
    };

    template<typename R> class SGD : public AbstractSolver<R> {
        public:
            // This can be overriden by parameter passed to step function.
            R step_size = SOLVER_MAT_DEFAULT_STEP_SIZE_H;

            SGD (R clip_norm = 5.0, R regc = 0.0);
            SGD (std::vector<Mat<R>>&, R clip_norm = 5.0, R regc = 0.0);
            virtual void step( std::vector<Mat<R>>&);
            virtual void step( std::vector<Mat<R>>&, R step_size);
    };

    template<typename R> class AdaGrad : public AbstractSolver<R> {
        public:
            // This can be overriden by parameter passed to step function.
            R step_size = SOLVER_MAT_DEFAULT_STEP_SIZE_H;

            std::unordered_map<cache_key_t<R>, cache_t<R>> gsums;
            AdaGrad (R smooth_eps = SMOOTH_DEFAULT, R clip_norm = 100.0, R regc = 0.0);
            AdaGrad (std::vector<Mat<R>>&, R smooth_eps = SMOOTH_DEFAULT, R clip_norm = 100.0, R regc = 0.0);
            virtual void step( std::vector<Mat<R>>&);
            virtual void step( std::vector<Mat<R>>&, R step_size);
            virtual void create_gradient_caches(std::vector<Mat<R>>&);
            virtual void reset_caches(std::vector<Mat<R>>&);
    };

    template<typename R> class RMSProp : public AdaGrad<R> {
        public:
            R step_size = SOLVER_MAT_DEFAULT_STEP_SIZE_H;
            R decay_rate;

            RMSProp (R _decay_rate= 0.999, R smooth_eps = SMOOTH_DEFAULT, R clip_norm = 100.0, R regc = 0.0);
            RMSProp (std::vector<Mat<R>>&, R _decay_rate= 0.999, R smooth_eps = SMOOTH_DEFAULT, R clip_norm = 100.0, R regc = 0.0);
            virtual void step(std::vector<Mat<R>>&);
            virtual void step(std::vector<Mat<R>>&, R step_size);
    };

    template<typename R> class RMSPropMomentum : public AbstractSolver<R> {
        public:
            R step_size = SOLVER_MAT_DEFAULT_STEP_SIZE_H;
            R decay_rate;
            R momentum;

            std::unordered_map<cache_key_t<R>, cache_t<R>> n_cache;
            std::unordered_map<cache_key_t<R>, cache_t<R>> g_cache;
            std::unordered_map<cache_key_t<R>, cache_t<R>> momentum_cache;

            RMSPropMomentum (R decay_rate= 0.95, R momentum=0.9, R step_size=1e-4, R smooth_eps = 1e-4, R clip_norm = 100.0, R regc = 0.0);
            RMSPropMomentum (std::vector<Mat<R>>&, R decay_rate= 0.95, R momentum=0.9, R step_size=1e-4, R smooth_eps = 1e-4, R clip_norm = 100.0, R regc = 0.0);
            virtual void step(std::vector<Mat<R>>&);
            virtual void step(std::vector<Mat<R>>&, R step_size);
            virtual void create_gradient_caches(std::vector<Mat<R>>&);
            virtual void reset_caches(std::vector<Mat<R>>&);
    };

    template<typename R> class AdaDelta : public AbstractSolver<R> {
        public:
            R rho;
            std::unordered_map<cache_key_t<R>, cache_t<R>> xsums;
            std::unordered_map<cache_key_t<R>, cache_t<R>> gsums;
            AdaDelta (R rho= 0.95, R smooth_eps = 1e-4, R clip_norm = 100.0, R regc = 0.0);
            AdaDelta (std::vector<Mat<R>>&, R rho= 0.95, R smooth_eps = 1e-4, R clip_norm = 100.0, R regc = 0.0);
            virtual void step(std::vector<Mat<R>>&);
            virtual void create_gradient_caches(std::vector<Mat<R>>&);
            virtual void reset_caches(std::vector<Mat<R>>&);
    };

    template<typename R> class Adam : public AbstractSolver<R> {
        public:
            R b1;
            R b2;
            R step_size;
            // This is a large integer:
            unsigned long long epoch;
            std::unordered_map<cache_key_t<R>, cache_t<R>> xsums;
            std::unordered_map<cache_key_t<R>, cache_t<R>> gsums;
            Adam (R step_size = 0.0002, R b1 = 0.5, R b2 = 1e-6, R smooth_eps = SMOOTH_DEFAULT, R clip_norm = 100.0, R regc = 0.0);
            Adam (std::vector<Mat<R>>&, R step_size = 0.0002, R b1 = 0.5, R b2 = 1e-6, R smooth_eps = SMOOTH_DEFAULT, R clip_norm = 100.0, R regc = 0.0);
            virtual void step(std::vector<Mat<R>>&);
            virtual void step(std::vector<Mat<R>>&, R step_size);
            virtual void create_gradient_caches(std::vector<Mat<R>>&);
            virtual void reset_caches(std::vector<Mat<R>>&);
    };

    template<typename R>
    std::shared_ptr<AbstractSolver<R>> construct(std::string solver_name,
                                                 std::vector<Mat<R>>& params,
                                                 R step_size = 0.01,
                                                 R regc = 0.0);

}
#endif
