// Vg1 = the colleague's Hopper GQA kernels, added to the sweep 2026-10-19 (files unchanged, compiled as their own
// translation units and linked into gqa_fwd_bwd):
//   Vg1 fwd = src/attention/GQA_fwd_sm90.cu       (OwnTensor::cuda::gqa_v62_causal_launch_d128)
//   Vg1 bwd = src/attention/GQA_bwd_vj2_sm90.cu   (OwnTensor::cuda::gqa_vj2_causal_bwd_launch_d128)
// Same layouts as ours: Q/O/dO/dQ [B*Hq*S, 128] bf16, K/V/dK/dV [B*Hkv*S, 128] bf16, LSE [B*Hq*S] fp32 natural log.
// Dense path only (chat lists = nullptr).
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace OwnTensor { namespace cuda {
void gqa_v62_causal_launch_d128(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* O, float* LSE,
    int B, int Hq, int Hkv, int S, float scale, cudaStream_t stream,
    const int32_t* chat_units, int n_units);
void gqa_vj2_causal_bwd_launch_d128(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* O, const __nv_bfloat16* dO, const float* LSE,
    __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV,
    int B, int Hq, int Hkv, int G, int S, float scale,
    const int32_t* chat_blocks, int n_chat_blocks, const int32_t* dq_tiles, int n_dq_tiles);
}} // namespace OwnTensor::cuda
