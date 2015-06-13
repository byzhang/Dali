#include "dali/mat/math/SynchronizedTensor.h"

using mshadow::AllocSpace;
using mshadow::FreeSpace;
using mshadow::Shape2;
using mshadow::Tensor;
using mshadow::Copy;

template<typename R>
SynchronizedTensor<R>::SynchronizedTensor(int n, int d, PreferredDevice _preferred_device) :
#ifdef DALI_USE_CUDA
    mem_gpu(Shape2(n, d)),
    gpu_fresh(false),
#endif
    mem_cpu(Shape2(n, d)),
    cpu_fresh(false),
    preferred_device(_preferred_device) {
}

template<typename R>
SynchronizedTensor<R>::SynchronizedTensor(const SynchronizedTensor& other) :
#ifdef DALI_USE_CUDA
        mem_gpu(other.mem_gpu.shape_),
        gpu_fresh(false),
#endif
        mem_cpu(other.mem_cpu.shape_),
        cpu_fresh(false),
        preferred_device(other.preferred_device) {
    if (other.cpu_fresh) {
        const auto& data_source = other.cpu_data();
        copy_data_from(data_source);
#ifdef DALI_USE_CUDA
    } else if (other.gpu_fresh) {
        const auto& data_source = other.gpu_data();
        copy_data_from(data_source);
#endif
    } else {
        // data was not initialized on the source
        // so we also choose not to initialize.
        return;
    }

}


template<typename R>
SynchronizedTensor<R>::~SynchronizedTensor() {
    if (mem_cpu.stream_ != NULL)
        FreeSpace(&mem_cpu);
#ifdef DALI_USE_CUDA
    if (mem_gpu.stream_ != NULL)
        FreeSpace(&mem_gpu);
#endif
}

template<typename R>
const Tensor<mshadow::cpu, 2, R>& SynchronizedTensor<R>::cpu_data() const {
    to_cpu();
    return mem_cpu;
}

template<typename R>
Tensor<mshadow::cpu, 2, R>& SynchronizedTensor<R>::mutable_cpu_data() {
    to_cpu();
#ifdef DALI_USE_CUDA
    gpu_fresh = false;
#endif
    return mem_cpu;
}

#ifdef DALI_USE_CUDA
    template<typename R>
    const Tensor<mshadow::gpu, 2, R>& SynchronizedTensor<R>::gpu_data() const {
        to_gpu();
        return mem_gpu;
    }

    template<typename R>
    Tensor<mshadow::gpu, 2, R>& SynchronizedTensor<R>::mutable_gpu_data() {
        to_gpu();
        cpu_fresh = false;
        return mem_gpu;
    }
#endif

template<typename R>
bool SynchronizedTensor<R>::prefers_cpu() const {
    return preferred_device == DEVICE_CPU;
}

template<typename R>
bool SynchronizedTensor<R>::prefers_gpu() const {
    return preferred_device == DEVICE_GPU;
}

#ifdef DALI_USE_CUDA
    template<typename R>
    void SynchronizedTensor<R>::to_gpu() const {
        if (!gpu_fresh) {
            if (mem_gpu.stream_ == NULL)
                AllocSpace(&mem_gpu, false);
            if (cpu_fresh)
                Copy(mem_gpu, mem_cpu);
            gpu_fresh = true;
        }
    }
#endif

template<typename R>
void SynchronizedTensor<R>::to_cpu() const {
    if (!cpu_fresh) {
        if (mem_cpu.stream_ == NULL)
            AllocSpace(&mem_cpu, false);
#ifdef DALI_USE_CUDA
        if (gpu_fresh)
            Copy(mem_cpu, mem_gpu);
#endif
        cpu_fresh = true;
    }
}

template<typename R>
template<typename SourceType>
void SynchronizedTensor<R>::copy_data_from(SourceType& data_source) {
    if (prefers_cpu()) {
        AllocSpace(&mem_cpu, false);
        Copy(mem_cpu, data_source);
        cpu_fresh = true;
    } else {
#ifdef DALI_USE_CUDA
        AllocSpace(&mem_gpu, false);
        Copy(mem_gpu, data_source);
        gpu_fresh = true;
#endif
    }
}

template class SynchronizedTensor<float>;
template class SynchronizedTensor<double>;
