#ifndef DALI_MAT_MATH_SYNCHRONIZED_TENSOR_H
#define DALI_MAT_MATH_SYNCHRONIZED_TENSOR_H

#include <functional>
#include <mshadow/tensor.h>

// This a small file keeping track of freshness of memory on CPU.
// The whole reason this is done is because not operations are
// available on GPU and therefore sometimes we need to copy
// the memory to CPU before executing an operation.
// On top of that we want to minimize the number of copies,
// which SynchronizedTensor helps achieving.

enum PreferredDevice {
    DEVICE_GPU,
    DEVICE_CPU
};
#ifdef DALI_USE_CUDA
    template<typename R>
    class SynchronizedTensor {
      private:
        mutable mshadow::Tensor<mshadow::cpu, 2, R> mem_cpu;
        mutable mshadow::Tensor<mshadow::gpu, 2, R> mem_gpu;
        PreferredDevice preferred_device;
      public:
        mutable bool cpu_fresh;
        mutable bool gpu_fresh;
        SynchronizedTensor(int n, int d, PreferredDevice preferred_device);
        // inherits preferred device and copies memory to it.
        SynchronizedTensor(const SynchronizedTensor& other);
        ~SynchronizedTensor();

        const mshadow::Tensor<mshadow::cpu, 2, R>&   cpu_data() const;
        mshadow::Tensor<mshadow::cpu, 2, R>& mutable_cpu_data();

        const mshadow::Tensor<mshadow::gpu, 2, R>&   gpu_data() const;
        mshadow::Tensor<mshadow::gpu, 2, R>& mutable_gpu_data();

        bool prefers_cpu() const;
        bool prefers_gpu() const;

        SynchronizedTensor& operator=(const SynchronizedTensor&) = delete;

        // tie-breaker for operations involving multiple tensors
        // on mixed devices.
        static PreferredDevice tie_breaker_device;
      private:
        void to_gpu() const;
        void to_cpu() const;

        // only used by copy constructor.
        template<typename SourceType>
        void copy_data_from(SourceType& src);
    };
#else
    template<typename R>
    class SynchronizedTensor {
      private:
        mutable mshadow::Tensor<mshadow::cpu, 2, R> mem_cpu;
        mutable bool cpu_fresh;
        PreferredDevice preferred_device;
      public:
        SynchronizedTensor(int n, int d, PreferredDevice preferred_device);
        // inherits preferred device and copies memory to it.
        SynchronizedTensor(const SynchronizedTensor& other);
        ~SynchronizedTensor();

        const mshadow::Tensor<mshadow::cpu, 2, R>&   cpu_data() const;
        mshadow::Tensor<mshadow::cpu, 2, R>& mutable_cpu_data();

        bool prefers_cpu() const;
        bool prefers_gpu() const;

        SynchronizedTensor& operator=(const SynchronizedTensor&) = delete;

      private:
        void to_cpu() const;
        template<typename SourceType>
        void copy_data_from(SourceType& src);
    };

#endif


template<typename R>
bool should_compute_on_gpu(
        std::initializer_list<std::reference_wrapper<SynchronizedTensor<R>>> sts);


#endif
