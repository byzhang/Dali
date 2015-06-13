#ifndef DALI_MAT_MATH_SYNCHRONIZED_TENSOR_H
#define DALI_MAT_MATH_SYNCHRONIZED_TENSOR_H

#include <mshadow/tensor.h>

// This a small file keeping track of freshness of memory on CPU.
// The whole reason this is done is because not operations are
// available on GPU and therefore sometimes we need to copy
// the memory to CPU before executing an operation.
// On top of that we want to minimize the number of copies,
// which SynchronizedTensor helps achieving.

#include <mshadow/tensor.h>

enum PreferredDevice {
    DEVICE_GPU,
    DEVICE_CPU
};

template<typename R>
class SynchronizedTensor {
  private:
    mutable mshadow::Tensor<mshadow::cpu, 2, R> mem_cpu;
    mutable mshadow::Tensor<mshadow::gpu, 2, R> mem_gpu;
    mutable bool cpu_fresh;
    mutable bool gpu_fresh;
    PreferredDevice preferred_device;
  public:
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
  private:
    void to_gpu() const;
    void to_cpu() const;

    // only used by copy constructor.
    template<typename SourceType>
    void copy_data_from(SourceType& src);
};

#endif
