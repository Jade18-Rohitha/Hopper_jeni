// ============================================================================
//  GQA_bwd_baseline.cu  —  STANDALONE DELIVERABLE
//
//  Grouped-Query-Attention flash-attention BACKWARD kernel for Hopper (H200, sm_90a).
//  This file contains ONLY the final kernel, "Vj1": a per-query-head, wgmma,
//  2-CTA/SM design whose dK/dV are reduced across the G query heads of each KV
//  group directly in L2 via cp.reduce.async.bulk.tensor (TMA-reduce) — no scratch
//  buffer, no separate reduction kernel. At locked clock (1980 MHz) it beats the
//  cuDNN reference (PyTorch SDPA enable_gqa bwd) on all 12 shapes of the sweep
//  {B in 2,4,8} x {Hq/Hkv in 12/4,16/4,24/8,32/8}, S=4096, D=128, bf16, causal.
//
//  Build:  nvcc -gencode arch=compute_90a,code=sm_90a -Iinclude \
//               src/attention/GQA_bwd_baseline.cu -o gqa_bwd_baseline
//  Run:    python precision/baseline_gqa.py B Hq Hkv   # writes data/gqa_*.bin
//          ./gqa_bwd_baseline B Hq Hkv                 # correctness + latency
//
//  Only Vj1 lives here; the full experiment log (V1..V45, Vz2, V44) stays in
//  GQA_bwd.cu.  See docs/report.md and docs/learnings.md for the derivation.
// ============================================================================
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <cuda_bf16.h>
#include "utils/kernelUtils.cuh"
#include "utils/kernelBench.cuh"

using namespace nvcuda;
using bf16 = __nv_bfloat16;

__device__ __forceinline__ void wgmma_m64n64k16(float d[32], uint64_t descA, uint64_t descB) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
        "%32, %33, 1, 1, 1, 0, 0;\n"
        : "+f"(d[0]),  "+f"(d[1]),  "+f"(d[2]),  "+f"(d[3]),
          "+f"(d[4]),  "+f"(d[5]),  "+f"(d[6]),  "+f"(d[7]),
          "+f"(d[8]),  "+f"(d[9]),  "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "l"(descA), "l"(descB));
}
__device__ __forceinline__ void wgmma_fence()  { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_wait0()  { asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); }

__device__ __forceinline__ void wgmma_wait1()  { asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory"); }

template<int N>
__device__ __forceinline__ void fence_operandN(float *d) {
#pragma unroll
    for (int i = 0; i < N; i++) asm volatile("" : "+f"(d[i]) :: "memory");
}

__device__ __forceinline__ void fence_proxy_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void zeroN(float *d) {
#pragma unroll
    for (int i = 0; i < N; i++) d[i] = 0.0f;
}

__device__ __forceinline__ void mbar_init_v4(uint64_t* mbar, uint32_t count) {
    uint32_t p = (uint32_t)__cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(p), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx_v4(uint64_t* mbar, uint32_t tx_bytes) {
    uint32_t p = (uint32_t)__cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" :: "r"(p), "r"(tx_bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait_v4(uint64_t* mbar, uint32_t parity) {
    uint32_t p = (uint32_t)__cvta_generic_to_shared(mbar);
    asm volatile(
        "{\n .reg .pred _P;\n"
        "_MBAR_LOOP_V4:\n"
        "mbarrier.try_wait.parity.shared.b64 _P, [%0], %1;\n"
        "@!_P bra _MBAR_LOOP_V4;\n }\n"
        :: "r"(p), "r"(parity) : "memory");
}

__device__ __forceinline__ void tma_load_2d_v4(
    const void* tma_desc, void* smem_dst, uint64_t* mbar, uint32_t cx, uint32_t cy) {
    uint32_t dst = (uint32_t)__cvta_generic_to_shared(smem_dst);
    uint32_t mb  = (uint32_t)__cvta_generic_to_shared(mbar);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
        :: "r"(dst), "l"((uint64_t)tma_desc), "r"(cx), "r"(cy), "r"(mb) : "memory");
}

__device__ __forceinline__ void tma_store_commit_v34() { asm volatile("cp.async.bulk.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void tma_store_wait_v34()   { asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory"); }

__device__ __forceinline__ void tma_bulk_wait0_v43() { asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory"); }

__device__ __forceinline__ uint64_t make_desc_sw128_K(const bf16* smem_ptr) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint64_t d = 0;
    d |= (uint64_t)((addr >> 4) & 0x3FFFu);   // start_address       [13:0]
    d |= (uint64_t)1u        << 16;           // leading_byte_offset [29:16] (unused; =1 per CUTLASS)
    d |= (uint64_t)(1024>>4) << 32;           // stride_byte_offset  [45:32] = 1024 B (SBO)
    d |= (uint64_t)1u        << 62;           // layout_type = B128   [63:62]
    return d;
}

__global__ void convert_dq_accum_to_bf16_v6(const float4 * __restrict__ in, uint2 * __restrict__ out, long n4) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        float4 v = in[i];
        bf16 o[4] = { __float2bfloat16(v.x), __float2bfloat16(v.y),
                      __float2bfloat16(v.z), __float2bfloat16(v.w) };
        out[i] = *reinterpret_cast<const uint2*>(o);
    }
}

constexpr float LOG2E_V29 = 1.4426950408889634f;   // V29: fold scale·log2e for exp2f softmax
__device__ __forceinline__ float ex2_approx_v29(float x) {   // V29: raw SFU 2^x (the forward's exp path)
    float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y;
}   // sS / sdP float row stride (Bc + 8)

__device__ __forceinline__ void wgmma_m64n64k16_tB(float d[32], uint64_t descA, uint64_t descB) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
        "%32, %33, 1, 1, 1, 0, 1;\n"   // scaleD, scaleA, scaleB, transA=0, transB=1
        : "+f"(d[0]),  "+f"(d[1]),  "+f"(d[2]),  "+f"(d[3]),
          "+f"(d[4]),  "+f"(d[5]),  "+f"(d[6]),  "+f"(d[7]),
          "+f"(d[8]),  "+f"(d[9]),  "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "l"(descA), "l"(descB));
}

__device__ __forceinline__ void wgmma_m64n64k16_tAtB(float d[32], uint64_t descA, uint64_t descB) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
        "%32, %33, 1, 1, 1, 1, 1;\n"   // scaleD, scaleA, scaleB, transA=1, transB=1
        : "+f"(d[0]),  "+f"(d[1]),  "+f"(d[2]),  "+f"(d[3]),
          "+f"(d[4]),  "+f"(d[5]),  "+f"(d[6]),  "+f"(d[7]),
          "+f"(d[8]),  "+f"(d[9]),  "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "l"(descA), "l"(descB));
}

__device__ __forceinline__ uint64_t make_desc_sw128_MN(const bf16* smem_ptr) {
    return make_desc_sw128_K(smem_ptr);
}

__device__ __forceinline__ void run_gemm_n64_sw2_hoB_issue(float acc[32], const bf16* A_sw, uint64_t descB_base) {
    fence_proxy_async_shared();
    fence_operandN<32>(acc);
    wgmma_fence();
#pragma unroll
    for (int k = 0; k < 8; k++) {
        uint64_t dA = make_desc_sw128_K(A_sw + (k >> 2) * 4096 + (k & 3) * 16);
        uint64_t dB = descB_base + (uint64_t)((k >> 2) * 512 + (k & 3) * 2);
        wgmma_m64n64k16(acc, dA, dB);
    }
    wgmma_commit();
}

template<int NDATA, int ROWSTRIDE>
__device__ __forceinline__ void stage_acc_bf16_s(const float *d, bf16 *stage, int tid, float scl) {
    int w = tid >> 5, lane = tid & 31;
    int r0 = w * 16 + (lane >> 2), r1 = r0 + 8, cc = (lane & 3) * 2;
#pragma unroll
    for (int nt = 0; nt < NDATA / 8; nt++) {
        int c = nt * 8 + cc;
        stage[r0 * ROWSTRIDE + c + 0] = __float2bfloat16(d[nt * 4 + 0] * scl);
        stage[r0 * ROWSTRIDE + c + 1] = __float2bfloat16(d[nt * 4 + 1] * scl);
        stage[r1 * ROWSTRIDE + c + 0] = __float2bfloat16(d[nt * 4 + 2] * scl);
        stage[r1 * ROWSTRIDE + c + 1] = __float2bfloat16(d[nt * 4 + 3] * scl);
    }
}

__device__ __forceinline__ void run_gemm_dVdK_half_te_issue_hoA(float acc[32], uint64_t descA_base,
                                                                const bf16* B_sw_half) {
    fence_proxy_async_shared();
    fence_operandN<32>(acc);
    wgmma_fence();
#pragma unroll
    for (int k = 0; k < 4; k++) {
        uint64_t dA = descA_base + (uint64_t)(k * 128);
        uint64_t dB = make_desc_sw128_MN(B_sw_half + k * 1024);
        wgmma_m64n64k16_tAtB(acc, dA, dB);
    }
    wgmma_commit();
}

__device__ __forceinline__ void stmatrix_x4(uint32_t dst, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};\n"
                 :: "r"(dst), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t src, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(src) : "memory");
}

template<int Bc, bool MASK>
__device__ __forceinline__ void fused_p_stsm(
    const float acc[32], bf16* sP, const float* sLSE, int wtid,
    int q_row0, int k_row0, float scale2)
{
    const int w = wtid >> 5, lane = wtid & 31;
    const int r0 = w * 16 + (lane >> 2), r1 = r0 + 8, cc = (lane & 3) * 2;
    const float l0 = sLSE[r0] * LOG2E_V29, l1 = sLSE[r1] * LOG2E_V29;
    const int gr0 = q_row0 + r0, gr1 = q_row0 + r1;

    // 32 P-values → 16 bf162 regs. pr0[nt]={P(r0,nt*8+cc),P(r0,+1)}; pr1[nt]={P(r1,..),..}.
    uint32_t pr0[8], pr1[8];
#pragma unroll
    for (int nt = 0; nt < 8; nt++) {
        float e00 = ex2_approx_v29(__fmaf_rn(acc[nt*4+0], scale2, -l0));
        float e01 = ex2_approx_v29(__fmaf_rn(acc[nt*4+1], scale2, -l0));
        float e10 = ex2_approx_v29(__fmaf_rn(acc[nt*4+2], scale2, -l1));
        float e11 = ex2_approx_v29(__fmaf_rn(acc[nt*4+3], scale2, -l1));
        if (MASK) {                              // branchless: ex2 already computed → SELECT → FSEL
            const int gc0 = k_row0 + nt*8 + cc, gc1 = gc0 + 1;
            e00 = (gc0 > gr0) ? 0.f : e00;
            e01 = (gc1 > gr0) ? 0.f : e01;
            e10 = (gc0 > gr1) ? 0.f : e10;
            e11 = (gc1 > gr1) ? 0.f : e11;
        }
        *reinterpret_cast<__nv_bfloat162*>(&pr0[nt]) = __float22bfloat162_rn(make_float2(e00, e01));
        *reinterpret_cast<__nv_bfloat162*>(&pr1[nt]) = __float22bfloat162_rn(make_float2(e10, e11));
    }

    // 4 stmatrix.x4 calls. Addr: lane l → matrix (l>>3)=key-block-in-group, row (l&7)=query phase.
    const int ph  = lane & 7;                    // (query_row & 7)
    const int mm  = lane >> 3;                    // key-block within the 4-group (0..3)
    const int qr0 = 16 * w + ph;                  // query row, qb=2w   (calls #0,#1)
    const int qr1 = 16 * w + 8 + ph;              // query row, qb=2w+1 (calls #2,#3)
    const uint32_t d0 = (uint32_t)__cvta_generic_to_shared(&sP[qr0 * 64 + (((mm    ) ^ ph) << 3)]);
    const uint32_t d1 = (uint32_t)__cvta_generic_to_shared(&sP[qr0 * 64 + (((mm + 4) ^ ph) << 3)]);
    const uint32_t d2 = (uint32_t)__cvta_generic_to_shared(&sP[qr1 * 64 + (((mm    ) ^ ph) << 3)]);
    const uint32_t d3 = (uint32_t)__cvta_generic_to_shared(&sP[qr1 * 64 + (((mm + 4) ^ ph) << 3)]);
    stmatrix_x4(d0, pr0[0], pr0[1], pr0[2], pr0[3]);   // qb=2w,   cb 0..3
    stmatrix_x4(d1, pr0[4], pr0[5], pr0[6], pr0[7]);   // qb=2w,   cb 4..7
    stmatrix_x4(d2, pr1[0], pr1[1], pr1[2], pr1[3]);   // qb=2w+1, cb 0..3
    stmatrix_x4(d3, pr1[4], pr1[5], pr1[6], pr1[7]);   // qb=2w+1, cb 4..7
}

__global__ void compute_drowsum_v22(
    const bf16 * __restrict__ d_dO, const bf16 * __restrict__ d_O,
    float * __restrict__ d_Drow, long nRows)
{
    const int  warpsPerBlock = blockDim.x >> 5;
    const int  lane          = threadIdx.x & 31;
    const long warp0         = (long)blockIdx.x * warpsPerBlock + (threadIdx.x >> 5);
    const long stride        = (long)gridDim.x * warpsPerBlock;
    // Grid-stride over rows → EVERY row in [0, nRows) is covered. VECTORIZED: each lane reads 4 CONTIGUOUS
    // elems as a uint2 (8B) from dO and O -> one coalesced transaction each vs 4 strided. D within tolerance
    // (contiguous vs strided sum order; not strictly bit-identical to v20, but |Δ|~1e-6 << 2e-2 check tol).
    for (long row = warp0; row < nRows; row += stride) {
        const long base = row * 128 + lane * 4;
        const uint2 vdO = *reinterpret_cast<const uint2*>(d_dO + base);   // 4 bf16
        const uint2 vO  = *reinterpret_cast<const uint2*>(d_O  + base);
        const bf16* a = reinterpret_cast<const bf16*>(&vdO);
        const bf16* b = reinterpret_cast<const bf16*>(&vO);
        float partial = 0.f;
        #pragma unroll
        for (int k = 0; k < 4; k++) partial += __bfloat162float(a[k]) * __bfloat162float(b[k]);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            partial += __shfl_down_sync(0xFFFFFFFFu, partial, off);
        if (lane == 0) d_Drow[row] = partial;
    }
}

template<int Bc>
__device__ __forceinline__ void fuse_dS_ldstsm(
    const bf16* __restrict__ sP, const float dP[32], const float* __restrict__ sD,
    bf16* __restrict__ sDS, int wtid) {
    const int w = wtid >> 5, lane = wtid & 31;
    const int r0 = w * 16 + (lane >> 2), r1 = r0 + 8;
    const float D0 = sD[r0], D1 = sD[r1];
    const int ph = lane & 7, mm = lane >> 3;
    const int qr0 = 16 * w + ph, qr1 = 16 * w + 8 + ph;
    const uint32_t s0 = (uint32_t)__cvta_generic_to_shared(&sP[qr0 * 64 + (((mm    ) ^ ph) << 3)]);
    const uint32_t s1 = (uint32_t)__cvta_generic_to_shared(&sP[qr0 * 64 + (((mm + 4) ^ ph) << 3)]);
    const uint32_t s2 = (uint32_t)__cvta_generic_to_shared(&sP[qr1 * 64 + (((mm    ) ^ ph) << 3)]);
    const uint32_t s3 = (uint32_t)__cvta_generic_to_shared(&sP[qr1 * 64 + (((mm + 4) ^ ph) << 3)]);
    uint32_t pP0[8], pP1[8];
    ldmatrix_x4(s0, pP0[0], pP0[1], pP0[2], pP0[3]);
    ldmatrix_x4(s1, pP0[4], pP0[5], pP0[6], pP0[7]);
    ldmatrix_x4(s2, pP1[0], pP1[1], pP1[2], pP1[3]);
    ldmatrix_x4(s3, pP1[4], pP1[5], pP1[6], pP1[7]);
    uint32_t pr0[8], pr1[8];
#pragma unroll
    for (int nt = 0; nt < 8; nt++) {
        const float2 pf0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&pP0[nt]));
        const float2 pf1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&pP1[nt]));
        const float d00 = pf0.x * (dP[nt*4+0] - D0), d01 = pf0.y * (dP[nt*4+1] - D0);
        const float d10 = pf1.x * (dP[nt*4+2] - D1), d11 = pf1.y * (dP[nt*4+3] - D1);
        *reinterpret_cast<__nv_bfloat162*>(&pr0[nt]) = __float22bfloat162_rn(make_float2(d00, d01));
        *reinterpret_cast<__nv_bfloat162*>(&pr1[nt]) = __float22bfloat162_rn(make_float2(d10, d11));
    }
    const uint32_t d0 = (uint32_t)__cvta_generic_to_shared(&sDS[qr0 * 64 + (((mm    ) ^ ph) << 3)]);
    const uint32_t d1 = (uint32_t)__cvta_generic_to_shared(&sDS[qr0 * 64 + (((mm + 4) ^ ph) << 3)]);
    const uint32_t d2 = (uint32_t)__cvta_generic_to_shared(&sDS[qr1 * 64 + (((mm    ) ^ ph) << 3)]);
    const uint32_t d3 = (uint32_t)__cvta_generic_to_shared(&sDS[qr1 * 64 + (((mm + 4) ^ ph) << 3)]);
    stmatrix_x4(d0, pr0[0], pr0[1], pr0[2], pr0[3]);
    stmatrix_x4(d1, pr0[4], pr0[5], pr0[6], pr0[7]);
    stmatrix_x4(d2, pr1[0], pr1[1], pr1[2], pr1[3]);
    stmatrix_x4(d3, pr1[4], pr1[5], pr1[6], pr1[7]);
}

__device__ __forceinline__ void run_gemm_dKdQ_te_issue_ho(
    float dk[32], float dq[32], uint64_t descDSmn_base, uint64_t descDSk_base,
    uint64_t descKhalf_base, const bf16* sQ_half) {
    fence_proxy_async_shared();
    fence_operandN<32>(dk);  fence_operandN<32>(dq);
    wgmma_fence();
#pragma unroll
    for (int k = 0; k < 4; k++)
        wgmma_m64n64k16_tAtB(dk, descDSmn_base + (uint64_t)(k * 128),
                                 make_desc_sw128_MN(sQ_half + k * 1024));
#pragma unroll
    for (int k = 0; k < 4; k++)
        wgmma_m64n64k16_tB (dq, descDSk_base   + (uint64_t)(k * 2),
                                 descKhalf_base + (uint64_t)(k * 128));
    wgmma_commit();
}

__device__ __forceinline__ void run_gemm_dVdKdQ_te_wait(float dv[32], float dk[32], float dq[32]) {
    wgmma_wait0();
    fence_operandN<32>(dv);  fence_operandN<32>(dk);  fence_operandN<32>(dq);
}

__device__ __forceinline__ void tma_reduce_add_2d_v43(const void* tma_desc, const float* smem, uint32_t cx, uint32_t cy) {
    uint32_t src = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile(
        "cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group [%0, {%1, %2}], [%3];\n"
        :: "l"((uint64_t)tma_desc), "r"(cx), "r"(cy), "r"(src) : "memory");
}

__device__ __forceinline__ void tma_reduce_add_2d_bf16(const void* tma_desc, const bf16* smem, uint32_t cx, uint32_t cy) {
    uint32_t src = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile(
        "cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group [%0, {%1, %2}], [%3];\n"
        :: "l"((uint64_t)tma_desc), "r"(cx), "r"(cy), "r"(src) : "memory");
}

__device__ __forceinline__ void store_acc_sw128_f32(const float* d, float* base, int tid, float scl) {
    int w = tid >> 5, lane = tid & 31;
    int r0 = w * 16 + (lane >> 2), r1 = r0 + 8, cc = (lane & 3) * 2;
    const int ph0 = r0 & 7, ph1 = r1 & 7;
#pragma unroll
    for (int nt = 0; nt < 8; nt++) {
        const int c   = nt * 8 + cc;               // 0..63 (even)
        const int atom = c >> 5, col32 = c & 31;   // 128B atom (0/1), col within atom
        const int chunk = col32 >> 2, lo = col32 & 3;
        const int abase = atom * (64 * 32);
        // c and c+1 share the same 4-fp32 chunk (c even) -> adjacent phys slots (lo, lo+1)
        base[abase + r0 * 32 + ((chunk ^ ph0) << 2) + lo    ] = d[nt * 4 + 0] * scl;
        base[abase + r0 * 32 + ((chunk ^ ph0) << 2) + lo + 1] = d[nt * 4 + 1] * scl;
        base[abase + r1 * 32 + ((chunk ^ ph1) << 2) + lo    ] = d[nt * 4 + 2] * scl;
        base[abase + r1 * 32 + ((chunk ^ ph1) << 2) + lo + 1] = d[nt * 4 + 3] * scl;
    }
}

template<int Br, int Bc, int D>
__global__ void __launch_bounds__(128, 1)
gqa_bwd_vj1(
    const __grid_constant__ CUtensorMap tma_K,  const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_Q,  const __grid_constant__ CUtensorMap tma_dO,
    const __grid_constant__ CUtensorMap tma_dV_st, const __grid_constant__ CUtensorMap tma_dK_st,
    const __grid_constant__ CUtensorMap tma_dq_red,
    const float* __restrict__ LSE, const float* __restrict__ Drow,
    int B, int Hq, int Hkv, int G, int S, float scale) {
    const int kt = blockIdx.x, hq = blockIdx.y, b = blockIdx.z;
    const int hkv = hq / G;
    const int wtid = threadIdx.x;
    const int nQ = S / Br;
    const int k_row0 = kt * Bc;
    const float scale2 = scale * LOG2E_V29;

    __shared__ __align__(128)  bf16  sK_sw[Bc*D], sV_sw[Bc*D], sQ_sw[Br*D], sdO_sw[Br*D];
    __shared__ __align__(1024) bf16  sP[Br*64], sDS[Br*64];
    __shared__ __align__(1024) float sStage[2][Br*64];   // double-buffered: both dQ halves deferred
    __shared__ float sLSE[Br], sD[Br];
    __shared__ __align__(8) uint64_t mbar_kv, mbar_qo;

    if (wtid == 0) { mbar_init_v4(&mbar_kv, 1); mbar_init_v4(&mbar_qo, 1); }
    __syncthreads();

    const uint64_t descK    = make_desc_sw128_K (sK_sw);
    const uint64_t descV    = make_desc_sw128_K (sV_sw);
    const uint64_t descP    = make_desc_sw128_MN(sP);
    const uint64_t descKh0  = make_desc_sw128_MN(sK_sw);
    const uint64_t descKh1  = make_desc_sw128_MN(sK_sw + 4096);
    const uint64_t descDSmn = make_desc_sw128_MN(sDS);
    const uint64_t descDSk  = make_desc_sw128_K (sDS);

    const uint32_t kvRow = (uint32_t)((b*Hkv+hkv)*S + k_row0);
    if (wtid == 0) {
        mbar_expect_tx_v4(&mbar_kv, (uint32_t)(Bc*64*sizeof(bf16))*4);
        tma_load_2d_v4(&tma_K, sK_sw,       &mbar_kv, 0,  kvRow);
        tma_load_2d_v4(&tma_K, sK_sw+64*64, &mbar_kv, 64, kvRow);
        tma_load_2d_v4(&tma_V, sV_sw,       &mbar_kv, 0,  kvRow);
        tma_load_2d_v4(&tma_V, sV_sw+64*64, &mbar_kv, 64, kvRow);
    }
    mbar_wait_v4(&mbar_kv, 0);

    float dv[64], dk[64];
    zeroN<64>(dv); zeroN<64>(dk);

    // SINGLE-BUFFER SOFTWARE PREFETCH: issue tile it+1's Q/dO TMA into the SAME buffer right after the
    // half1 dK-gemm frees sQ/sdO, overlapping the dQ-reduce (which touches only sStage). No ring, no
    // smem cost, 2 CTAs preserved. Attacks long_scoreboard 2.51 (loads were issued-then-immediately-waited).
    const int nq_ = nQ - kt, nIter = nq_;   // Vj1 per-hq: ONE query head per CTA
    auto issue = [&](int it) {
        const int q = kt + it;
        const uint32_t qRow = (uint32_t)((b*Hq+hq)*S + q*Br);
        const long lb = (long)(b*Hq+hq)*S + (long)q*Br;
        if (wtid == 0) {
            mbar_expect_tx_v4(&mbar_qo, (uint32_t)(Br*D*sizeof(bf16))*2);
            tma_load_2d_v4(&tma_Q,  sQ_sw,        &mbar_qo, 0,  qRow);
            tma_load_2d_v4(&tma_Q,  sQ_sw+64*64,  &mbar_qo, 64, qRow);
            tma_load_2d_v4(&tma_dO, sdO_sw,       &mbar_qo, 0,  qRow);
            tma_load_2d_v4(&tma_dO, sdO_sw+64*64, &mbar_qo, 64, qRow);
        }
        if (wtid < Br) { sLSE[wtid] = LSE[lb+wtid]; sD[wtid] = Drow[lb+wtid]; }
    };
    uint32_t qopar = 0;
    issue(0);                                          // prologue: load tile 0

    for (int it = 0; it < nIter; it++) {
        const int q = kt + it;
        const uint32_t qRow = (uint32_t)((b*Hq+hq)*S + q*Br);
        mbar_wait_v4(&mbar_qo, qopar); qopar ^= 1;
        // NOTE: no post-load barrier — sQ visibility is guaranteed by mbar_wait (TMA completion), and
        // sLSE/sD were written+published by the PREVIOUS tile's store-barrier (prefetch). One barrier saved.

        // S = Q@Kᵀ and dP = dO@Vᵀ ISSUED TOGETHER (overlap on the tensor pipe), waited once — instead of
        // S(wait)→fused_p→dP(wait) which serialized the two GEMMs. Attacks wait 1.28 + 44%-idle pipe.
        float acc[32];   zeroN<32>(acc);
        float dPacc[32]; zeroN<32>(dPacc);
        run_gemm_n64_sw2_hoB_issue(acc,   sQ_sw,  descK);   // S (group A)
        run_gemm_n64_sw2_hoB_issue(dPacc, sdO_sw, descV);   // dP (group B)
        wgmma_wait1();                       // wait S only; dP GEMM keeps running on the tensor pipe
        fence_operandN<32>(acc);
        if (q == kt) fused_p_stsm<Bc,true >(acc, sP, sLSE, wtid, q*Br, k_row0, scale2);
        else         fused_p_stsm<Bc,false>(acc, sP, sLSE, wtid, 0, 0, scale2);  // softmax OVERLAPS dP GEMM
        wgmma_wait0();                       // now wait dP
        fence_operandN<32>(dPacc);
        __syncthreads();

        run_gemm_dVdK_half_te_issue_hoA(dv,    descP, sdO_sw + 0);
        run_gemm_dVdK_half_te_issue_hoA(dv+32, descP, sdO_sw + 4096);
        fuse_dS_ldstsm<Bc>(sP, dPacc, sD, sDS, wtid);
        __syncthreads();

        // dQ split: half0 & half1 kept SEPARATE — half0's committed reduce overlaps half1's GEMM, and the
        // prefetch overlaps half1's store+reduce. (Combining into one wait0 lost that overlap → slower.)
        {   float dq[32]; zeroN<32>(dq);
            run_gemm_dKdQ_te_issue_ho(dk, dq, descDSmn, descDSk, descKh0, sQ_sw + 0);
            if (wtid == 0) tma_bulk_wait0_v43();        // drain PREV reduces — moved BEFORE the wgmma wait
            run_gemm_dVdKdQ_te_wait(dv, dk, dq);        //   so thread 0's drain overlaps the dK/dQ GEMM
            __syncthreads();
            store_acc_sw128_f32(dq, sStage[0], wtid, scale);
            __syncthreads(); fence_proxy_async_shared();
            if (wtid == 0) {
                tma_reduce_add_2d_v43(&tma_dq_red, sStage[0],       0,  qRow);
                tma_reduce_add_2d_v43(&tma_dq_red, sStage[0]+64*32, 32, qRow);
                tma_store_commit_v34();   // NO wait — deferred (double-buffered sStage)
            }
        }
        {   float dq[32]; zeroN<32>(dq);
            run_gemm_dKdQ_te_issue_ho(dk+32, dq, descDSmn, descDSk, descKh1, sQ_sw + 4096);
            run_gemm_dVdKdQ_te_wait(dv, dk+32, dq);
            __syncthreads();                            // all threads done reading sQ/sdO
            if (it + 1 < nIter) issue(it + 1);          // prefetch next tile into same buffer (overlap)
            store_acc_sw128_f32(dq, sStage[1], wtid, scale);
            __syncthreads(); fence_proxy_async_shared();
            if (wtid == 0) {
                tma_reduce_add_2d_v43(&tma_dq_red, sStage[1],       64, qRow);
                tma_reduce_add_2d_v43(&tma_dq_red, sStage[1]+64*32, 96, qRow);
                tma_store_commit_v34();   // NO wait0 — deferred to next tile's half0
            }
        }
    }
    if (wtid == 0) tma_bulk_wait0_v43();   // drain the last tile's deferred half1 reduce before epilogue
    __syncthreads();
    // epilogue: stage + TMA-store dV, dK (full [64x128] each, 2 col-halves)
    fence_operandN<64>(dv);
    stage_acc_bf16_s<64,64>(dv,    sQ_sw + 0,    wtid, 1.0f);
    stage_acc_bf16_s<64,64>(dv+32, sQ_sw + 4096, wtid, 1.0f);
    fence_operandN<64>(dk);
    stage_acc_bf16_s<64,64>(dk,    sdO_sw + 0,    wtid, scale);
    stage_acc_bf16_s<64,64>(dk+32, sdO_sw + 4096, wtid, scale);
    __syncthreads(); fence_proxy_async_shared();
    // Vj1: TMA-REDUCE (add) each per-hq dV/dK straight into the FINAL [B,Hkv,S,D] at kvRow. The G query
    // heads of this KV-group (separate CTAs, same kvRow) sum in L2 — no scratch buffer, no greduce kernel.
    if (wtid == 0) {
        tma_reduce_add_2d_bf16(&tma_dV_st, sQ_sw + 0,    0,  kvRow);
        tma_reduce_add_2d_bf16(&tma_dV_st, sQ_sw + 4096, 64, kvRow);
        tma_reduce_add_2d_bf16(&tma_dK_st, sdO_sw + 0,    0,  kvRow);
        tma_reduce_add_2d_bf16(&tma_dK_st, sdO_sw + 4096, 64, kvRow);
        tma_store_commit_v34(); tma_store_wait_v34();
    }
}

template<int Br, int Bc, int D>
void launch_gqa_backward_vj1(
    const bf16 *d_Q, const bf16 *d_K, const bf16 *d_V, const bf16 *d_O,
    const bf16 *d_dO, const float *d_LSE,
    bf16 *d_dQ, bf16 *d_dK, bf16 *d_dV,
    int B, int Hq, int Hkv, int G, int S, float scale) {
    auto make_tma_sw128 = [&](const bf16* ptr, uint64_t total_rows, uint32_t tile_rows) {
        CUtensorMap desc{}; uint64_t gS[2]={(uint64_t)D,total_rows}; uint64_t gT[1]={(uint64_t)D*sizeof(bf16)};
        uint32_t bx[2]={64u,tile_rows}, eS[2]={1,1};
        CUresult r=cuTensorMapEncodeTiled(&desc,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,(void*)ptr,gS,gT,bx,eS,
            CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if(r!=CUDA_SUCCESS){const char*e;cuGetErrorString(r,&e);fprintf(stderr,"sw128 vz2: %s\n",e);exit(1);} return desc; };
    auto make_tma_out = [&](const bf16* ptr, uint64_t rows) {
        CUtensorMap desc{}; uint64_t gS[2]={(uint64_t)D,rows}; uint64_t gT[1]={(uint64_t)D*sizeof(bf16)};
        uint32_t bx[2]={64u,64u}, eS[2]={1,1};
        CUresult r=cuTensorMapEncodeTiled(&desc,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,(void*)ptr,gS,gT,bx,eS,
            CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if(r!=CUDA_SUCCESS){const char*e;cuGetErrorString(r,&e);fprintf(stderr,"out vz2: %s\n",e);exit(1);} return desc; };
    auto make_tma_red = [&](const float* ptr, uint64_t rows) {
        CUtensorMap desc{}; uint64_t gS[2]={(uint64_t)D,rows}; uint64_t gT[1]={(uint64_t)D*sizeof(float)};
        uint32_t bx[2]={32u,64u}, eS[2]={1,1};
        CUresult r=cuTensorMapEncodeTiled(&desc,CU_TENSOR_MAP_DATA_TYPE_FLOAT32,2,(void*)ptr,gS,gT,bx,eS,
            CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if(r!=CUDA_SUCCESS){const char*e;cuGetErrorString(r,&e);fprintf(stderr,"red vz2: %s\n",e);exit(1);} return desc; };

    const uint64_t Rq=(uint64_t)B*Hq*S, Rkv=(uint64_t)B*Hkv*S;
    CUtensorMap tK=make_tma_sw128(d_K,Rkv,Bc), tV=make_tma_sw128(d_V,Rkv,Bc);
    CUtensorMap tQ=make_tma_sw128(d_Q,Rq,Br),  tdO=make_tma_sw128(d_dO,Rq,Br);
    // Vj1 per-hq: main kernel TMA-reduce-adds dK/dV DIRECTLY into the final [B,Hkv,S,D] (the G heads sum
    // in L2), so d_dK/d_dV must start zeroed. No scratch buffer, no greduce kernel — that 69us DRAM-bound
    // serial reduce (2.5% of runtime) is gone; the reduction now overlaps the main kernel's compute.
    CUDA_CHECK(cudaMemset(d_dV,0,(size_t)Rkv*D*sizeof(bf16)));
    CUDA_CHECK(cudaMemset(d_dK,0,(size_t)Rkv*D*sizeof(bf16)));
    CUtensorMap tdV=make_tma_out(d_dV,Rkv), tdK=make_tma_out(d_dK,Rkv);

    const long drowN=(long)B*Hq*S; static float* d_Drow=nullptr; static long drc=0;
    if(drowN>drc){ if(d_Drow)CUDA_CHECK(cudaFree(d_Drow)); CUDA_CHECK(cudaMalloc(&d_Drow,drowN*sizeof(float))); drc=drowN; }
    const long dqN=(long)B*Hq*S*D; static float* d_dqa=nullptr; static long dqc=0;
    if(dqN>dqc){ if(d_dqa)CUDA_CHECK(cudaFree(d_dqa)); CUDA_CHECK(cudaMalloc(&d_dqa,dqN*sizeof(float))); dqc=dqN; }
    CUDA_CHECK(cudaMemset(d_dqa,0,dqN*sizeof(float)));
    CUtensorMap tRed=make_tma_red(d_dqa,(uint64_t)B*Hq*S);

    const int dBlock=256; const long dGrid=(drowN+(dBlock/32)-1)/(dBlock/32);
    compute_drowsum_v22<<<(unsigned)dGrid,dBlock>>>(d_dO,d_O,d_Drow,drowN);

    dim3 GRID(S/Bc, Hq, B);   // Vj1 per-hq: 3x the CTAs (Hq not Hkv) to fill the GPU at low B
    gqa_bwd_vj1<Br,Bc,D><<<GRID,128>>>(tK,tV,tQ,tdO,tdV,tdK,tRed,d_LSE,d_Drow,B,Hq,Hkv,G,S,scale);
    // dK/dV reduced in-kernel via TMA-reduce — greduce kernel removed.

    const int cB=256; const long dqN4=dqN/4; const int cG=(int)((dqN4+cB-1)/cB);
    convert_dq_accum_to_bf16_v6<<<cG,cB>>>(reinterpret_cast<const float4*>(d_dqa), reinterpret_cast<uint2*>(d_dQ), dqN4);
}

// ============================================================================
//  Driver: load the PyTorch reference, run Vj1, verify dQ/dK/dV, benchmark.
// ============================================================================
int main(int argc, char** argv){
    std::cout << "GQA Backward (Vj1) — standalone deliverable  [Hopper SM_90a / H200]\n";
    std::cout << "Prerequisite: python precision/baseline_gqa.py B Hq Hkv\n\n";

    const int B   = argc > 1 ? atoi(argv[1]) : 4;
    const int Hq  = argc > 2 ? atoi(argv[2]) : 24;
    const int Hkv = argc > 3 ? atoi(argv[3]) : 8;
    const int G   = Hq / Hkv;
    std::cout << "  shape: B=" << B << " Hq=" << Hq << " Hkv=" << Hkv << " (G=" << G << ")\n\n";
    constexpr int S = 4096, D = 128, Br = 64, Bc = 64;

    const float  scale = 1.0f / sqrtf((float)D);
    const size_t Nq    = (size_t)B * Hq  * S * D;
    const size_t Nkv   = (size_t)B * Hkv * S * D;
    const size_t Nlse  = (size_t)B * Hq  * S;

    std::vector<__nv_bfloat16> h_Q(Nq), h_K(Nkv), h_V(Nkv), h_O(Nq), h_dO(Nq);
    std::vector<float>         h_LSE(Nlse), h_dQ_ref(Nq), h_dK_ref(Nkv), h_dV_ref(Nkv);

    auto fileSz = [](const char *p, size_t n) -> bool {
        FILE *f = fopen(p, "rb"); if (!f) return false;
        fseek(f, 0, SEEK_END); size_t b = (size_t)ftell(f); fclose(f);
        return b == n * sizeof(float);
    };
    auto loadBF16 = [](const char *p, std::vector<__nv_bfloat16> &dst, size_t n){
        std::vector<float> tmp(n); loadBin(p, tmp.data(), n);
        for (size_t i = 0; i < n; ++i) dst[i] = __float2bfloat16(tmp[i]);
    };
    if (!fileSz("data/gqa_q.bin",Nq)||!fileSz("data/gqa_k.bin",Nkv)||!fileSz("data/gqa_v.bin",Nkv)||
        !fileSz("data/gqa_o.bin",Nq)||!fileSz("data/gqa_do.bin",Nq)||!fileSz("data/gqa_lse.bin",Nlse)||
        !fileSz("data/gqa_dq.bin",Nq)||!fileSz("data/gqa_dk.bin",Nkv)||!fileSz("data/gqa_dv.bin",Nkv)){
        std::cerr << "ERROR: reference files not found. Run: python precision/baseline_gqa.py "
                  << B << " " << Hq << " " << Hkv << "\n";
        return 1;
    }
    loadBF16("data/gqa_q.bin",h_Q,Nq);   loadBF16("data/gqa_k.bin",h_K,Nkv);
    loadBF16("data/gqa_v.bin",h_V,Nkv);  loadBF16("data/gqa_o.bin",h_O,Nq);
    loadBF16("data/gqa_do.bin",h_dO,Nq);
    loadBin("data/gqa_lse.bin",h_LSE.data(),Nlse);
    loadBin("data/gqa_dq.bin",h_dQ_ref.data(),Nq);
    loadBin("data/gqa_dk.bin",h_dK_ref.data(),Nkv);
    loadBin("data/gqa_dv.bin",h_dV_ref.data(),Nkv);
    std::cout << "Loaded PyTorch reference from data/gqa_*.bin\n\n";

    __nv_bfloat16 *d_Q,*d_K,*d_V,*d_O,*d_dO,*d_dQ,*d_dK,*d_dV; float *d_LSE;
    CUDA_CHECK(cudaMalloc(&d_Q,Nq*sizeof(bf16)));   CUDA_CHECK(cudaMalloc(&d_K,Nkv*sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&d_V,Nkv*sizeof(bf16)));  CUDA_CHECK(cudaMalloc(&d_O,Nq*sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&d_dO,Nq*sizeof(bf16)));  CUDA_CHECK(cudaMalloc(&d_LSE,Nlse*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dQ,Nq*sizeof(bf16)));  CUDA_CHECK(cudaMalloc(&d_dK,Nkv*sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&d_dV,Nkv*sizeof(bf16)));
    CUDA_CHECK(cudaMemcpy(d_Q,h_Q.data(),Nq*sizeof(bf16),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K,h_K.data(),Nkv*sizeof(bf16),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V,h_V.data(),Nkv*sizeof(bf16),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_O,h_O.data(),Nq*sizeof(bf16),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dO,h_dO.data(),Nq*sizeof(bf16),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_LSE,h_LSE.data(),Nlse*sizeof(float),cudaMemcpyHostToDevice));

    // ---- correctness ----
    launch_gqa_backward_vj1<Br,Bc,D>(d_Q,d_K,d_V,d_O,d_dO,d_LSE,d_dQ,d_dK,d_dV,B,Hq,Hkv,G,S,scale);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    {
        std::vector<__nv_bfloat16> hQ(Nq),hK(Nkv),hV(Nkv);
        CUDA_CHECK(cudaMemcpy(hQ.data(),d_dQ,Nq*sizeof(bf16),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hK.data(),d_dK,Nkv*sizeof(bf16),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hV.data(),d_dV,Nkv*sizeof(bf16),cudaMemcpyDeviceToHost));
        std::vector<float> qf(Nq),kf(Nkv),vf(Nkv);
        for(size_t i=0;i<Nq;++i)  qf[i]=__bfloat162float(hQ[i]);
        for(size_t i=0;i<Nkv;++i) kf[i]=__bfloat162float(hK[i]);
        for(size_t i=0;i<Nkv;++i) vf[i]=__bfloat162float(hV[i]);
        std::cout << "-- Vj1 (per-hq, TMA-reduce dK/dV) --\n";
        reportPrecision("  dQ", h_dQ_ref.data(), qf.data(), Nq);
        reportPrecision("  dK", h_dK_ref.data(), kf.data(), Nkv);
        reportPrecision("  dV", h_dV_ref.data(), vf.data(), Nkv);
        std::cout << "  dQ: "; checkResult(h_dQ_ref.data(), qf.data(), Nq,  2e-2f, 2e-2f);
        std::cout << "  dK: "; checkResult(h_dK_ref.data(), kf.data(), Nkv, 2e-2f, 2e-2f);
        std::cout << "  dV: "; checkResult(h_dV_ref.data(), vf.data(), Nkv, 2e-2f, 2e-2f);
        std::cout << "\n";
    }

    // ---- latency (FLOP convention matches baseline_gqa.py) ----
    const long long bwd_flops = 16LL * B * Hq * (long long)S * S * D;
    {
        KernelStats s = benchmarkKernel(
            [&](){ launch_gqa_backward_vj1<Br,Bc,D>(
                d_Q,d_K,d_V,d_O,d_dO,d_LSE,d_dQ,d_dK,d_dV,B,Hq,Hkv,G,S,scale); },
            100, 10, bwd_flops);
        displayStats("GQA bwd Vj1  (per-hq, TMA-reduce dK/dV)  (Hopper SM_90a)", s);
    }

    CUDA_CHECK(cudaFree(d_Q));  CUDA_CHECK(cudaFree(d_K));  CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));  CUDA_CHECK(cudaFree(d_dO)); CUDA_CHECK(cudaFree(d_LSE));
    CUDA_CHECK(cudaFree(d_dQ)); CUDA_CHECK(cudaFree(d_dK)); CUDA_CHECK(cudaFree(d_dV));
    return 0;
}
