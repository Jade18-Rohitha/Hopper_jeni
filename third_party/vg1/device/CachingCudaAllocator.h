// Minimal stand-in for OwnTensor's device/CachingCudaAllocator.h (BluTrain Tensor-Implementations), so the
// colleague's Vg1 kernels (src/attention/GQA_fwd_sm90.cu, GQA_bwd_vj2_sm90.cu) build unchanged in this repo.
// Same namespace and the two calls Vg1 uses: instance().allocate(bytes) / deallocate(ptr).
// Vg1's launcher allocates its scratch once (static, grown only when a call needs more), so plain
// cudaMalloc/cudaFree here does not change what the sweep times.
#pragma once
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

namespace OwnTensor {
class CachingCUDAAllocator {
public:
    static CachingCUDAAllocator& instance() { static CachingCUDAAllocator a; return a; }
    void* allocate(size_t bytes) {
        void* p = nullptr;
        cudaError_t e = cudaMalloc(&p, bytes);
        if (e != cudaSuccess) { fprintf(stderr, "vg1 shim allocate(%zu): %s\n", bytes, cudaGetErrorString(e)); abort(); }
        return p;
    }
    void deallocate(void* p) { if (p) cudaFree(p); }
};
} // namespace OwnTensor
