cmake_minimum_required(VERSION 3.22)
project(HOPPER LANGUAGES CUDA CXX)

# ── Export compile_commands.json so VSCode IntelliSense resolves includes ──────
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── CUDA Toolkit ───────────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

# ── Language standards ─────────────────────────────────────────────────────────
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD  17)

# ── Architecture ──────────────────────────────────────────────────────────────
# sm_90a = H100 / H200 (Hopper). GQA_bwd V1/V2 use only WMMA (SM_70+ ISA),
# so they run here without wgmma/tcgen05.
set(CMAKE_CUDA_ARCHITECTURES 90a)

# ── Compiler flags ─────────────────────────────────────────────────────────────
# -lineinfo : maps profiler samples to source lines (Nsight Compute)
# -O3       : full optimization
# NOTE: --use_fast_math is NOT set — the backward kernels use IEEE-compliant math
#       for tighter precision vs PyTorch.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -O3")

# ── Common include directory ───────────────────────────────────────────────────
# Kernels include headers as "utils/kernelBench.cuh" etc.
set(COMMON_INCLUDES ${CMAKE_SOURCE_DIR}/include)

# ── Output directory ───────────────────────────────────────────────────────────
set(BIN_DIR ${CMAKE_BINARY_DIR}/bin)

# ── Helper macro: add a CUDA kernel executable ────────────────────────────────
macro(add_kernel TARGET SOURCE)
    add_executable(${TARGET} ${SOURCE})
    target_include_directories(${TARGET} PRIVATE ${COMMON_INCLUDES})
    target_link_libraries(${TARGET} PRIVATE CUDA::cudart)
    set_target_properties(${TARGET} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${BIN_DIR}
    )
endmacro()

# ── Kernel targets ─────────────────────────────────────────────────────────────
add_kernel(gqa_bwd  src/attention/GQA_bwd.cu)
# V4 uses the CUDA Driver API (cuTensorMapEncodeTiled) for TMA descriptors.
target_link_libraries(gqa_bwd PRIVATE CUDA::cuda_driver)

add_kernel(gqa_bwd_baseline  src/attention/GQA_bwd_baseline.cu)
target_link_libraries(gqa_bwd_baseline PRIVATE CUDA::cuda_driver)

# standalone head-to-head: Vm1 vs Vj1d backward (individual test before any sweep)
add_kernel(gqa_bwd_vm1_test  src/attention/GQA_bwd_vm1_test.cu)
target_link_libraries(gqa_bwd_vm1_test PRIVATE CUDA::cuda_driver CUDA::nvml)   # NVML: sustained-mode energy

# fwd+bwd chain: V62 forward -> Vj1d backward (vs cuDNN sdpa fwd+bwd)
add_kernel(gqa_fwd_bwd  src/attention/GQA_fwd_bwd.cu)
target_link_libraries(gqa_fwd_bwd PRIVATE CUDA::cuda_driver)
# Vg1 = colleague's Hopper fwd/bwd (2026-10-19), unchanged, each its own translation unit; the allocator shim in
# third_party/vg1 stands in for OwnTensor's device/CachingCudaAllocator.h.
target_sources(gqa_fwd_bwd PRIVATE src/attention/GQA_fwd_sm90.cu src/attention/GQA_bwd_vj2_sm90.cu)
target_include_directories(gqa_fwd_bwd PRIVATE ${CMAKE_SOURCE_DIR}/third_party/vg1 ${CMAKE_SOURCE_DIR}/src/attention)

# (V45 standalone target removed)
# add_kernel(gqa_fwd_v62  src/attention/GQA_fwd_ref.cu)
# target_link_libraries(gqa_fwd_v62 PRIVATE CUDA::cuda_driver)
