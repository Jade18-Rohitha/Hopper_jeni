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

add_kernel(gqa_bwd_v44  src/attention/GQA_bwd_v44.cu)
target_link_libraries(gqa_bwd_v44 PRIVATE CUDA::cuda_driver)

add_kernel(gqa_bwd_v45  src/attention/GQA_bwd_v45.cu)
target_link_libraries(gqa_bwd_v45 PRIVATE CUDA::cuda_driver)

add_kernel(gqa_fwd_v62  src/attention/GQA_fwd_ref.cu)
target_link_libraries(gqa_fwd_v62 PRIVATE CUDA::cuda_driver)
