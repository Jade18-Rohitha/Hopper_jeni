// ═════════════════════════════════════════════════════════════════════════════
// GQA_bwd_v45.cu — STANDALONE single-kernel extraction of V45 (swizzled TMA-reduce dQ)
// Extracted mechanically from src/attention/GQA_bwd.cu (V1–V45).  Contains ONLY
// V45's transitive dependency closure.  Kernel + launcher are byte-identical to
// the original.  sm_90a (Hopper H100/H200).
// ═════════════════════════════════════════════════════════════════════════════
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>
#include <cassert>
#include <cmath>
#include <cuda_bf16.h>
#include <iostream>
#include <vector>
#include "utils/kernelUtils.cuh"
#include "utils/kernelBench.cuh"

using bf16 = __nv_bfloat16;

constexpr float LOG2E_V29 = 1.4426950408889634f;   // V29: fold scale·log2e for exp2f softmax
__device__ __forceinline__ float ex2_approx_v29(float x) {   // V29: raw SFU 2^x (the forward's exp path)
    float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y;
}

// ═════════════════════════════════════════════════════════════════════════════
// DEVICE HELPERS — V45 closure (filled iteratively, dependency order)
// ═════════════════════════════════════════════════════════════════════════════
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

__device__ __forceinline__ void tma_store_2d_v34(const void* tma_desc, const bf16* smem, uint32_t cx, uint32_t cy) {
    uint32_t src = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];\n"
        :: "l"((uint64_t)tma_desc), "r"(cx), "r"(cy), "r"(src) : "memory");
}

__device__ __forceinline__ void tma_store_commit_v34() { asm volatile("cp.async.bulk.commit_group;\n" ::: "memory"); }

__device__ __forceinline__ void tma_store_wait_v34()   { asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory"); }

__device__ __forceinline__ void tma_bulk_wait1_v43() { asm volatile("cp.async.bulk.wait_group 1;\n" ::: "memory"); }

__device__ __forceinline__ uint64_t make_desc_sw128_K(const bf16* smem_ptr) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint64_t d = 0;
    d |= (uint64_t)((addr >> 4) & 0x3FFFu);   // start_address       [13:0]
    d |= (uint64_t)1u        << 16;           // leading_byte_offset [29:16] (unused; =1 per CUTLASS)
    d |= (uint64_t)(1024>>4) << 32;           // stride_byte_offset  [45:32] = 1024 B (SBO)
    d |= (uint64_t)1u        << 62;           // layout_type = B128   [63:62]
    return d;
}

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

__device__ __forceinline__ void run_gemm_n64_sw2_hoB(float acc[32], const bf16* A_sw, uint64_t descB_base) {
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
    wgmma_wait0();
    fence_operandN<32>(acc);
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

__device__ __forceinline__ void reg_dec_producer_v30() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(40)); }

__device__ __forceinline__ void reg_inc_consumer_v30() { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(232)); }

__device__ __forceinline__ void consumer_sync() {
    asm volatile("bar.sync 1, 256;\n" ::: "memory");
}

__device__ __forceinline__ void consumer_sync_wg0() {
    asm volatile("bar.sync 3, 128;\n" ::: "memory");
}

__device__ __forceinline__ void consumer_sync_wg1() {
    asm volatile("bar.sync 4, 128;\n" ::: "memory");
}

__device__ __forceinline__ void mbar_arrive_v11(uint64_t* mbar) {
    uint32_t p = (uint32_t)__cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];\n" :: "r"(p) : "memory");
}

__device__ __forceinline__ void producer_sync() {
    asm volatile("bar.sync 2, 128;\n" ::: "memory");
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

// ═════════════════════════════════════════════════════════════════════════════
// AUX KERNELS (verbatim)
// ═════════════════════════════════════════════════════════════════════════════

// Converts the fp32 dq_accum scratch (written via atomicAdd by V5) to bf16 d_dQ.
__global__ void convert_dq_accum_to_bf16_v5(const float * __restrict__ d_dq_accum, bf16 * __restrict__ d_dQ, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d_dQ[i] = __float2bfloat16(d_dq_accum[i]);
}

// One warp per row, grid-stride over rows.  D fixed at 128 (32 lanes × 4 strided
// columns).  Light memory-bound reduction (~200 MB read: dO+O).
__global__ void compute_drowsum_v22(
    const bf16 * __restrict__ d_dO, const bf16 * __restrict__ d_O,
    float * __restrict__ d_Drow, long nRows)
{
    const int  warpsPerBlock = blockDim.x >> 5;
    const int  lane          = threadIdx.x & 31;
    const long warp0         = (long)blockIdx.x * warpsPerBlock + (threadIdx.x >> 5);
    const long stride        = (long)gridDim.x * warpsPerBlock;
    for (long row = warp0; row < nRows; row += stride) {
        const long base = row * 128;
        float partial = 0.f;
        #pragma unroll
        for (int j = lane; j < 128; j += 32)
            partial += __bfloat162float(d_dO[base + j]) * __bfloat162float(d_O[base + j]);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            partial += __shfl_down_sync(0xFFFFFFFFu, partial, off);
        if (lane == 0) d_Drow[row] = partial;
    }
}

__device__ __forceinline__ void store_acc_sw128_f32_v45(const float* d, float* base, int tid, float scl) {
    int w = tid >> 5, lane = tid & 31;
    int r0 = w * 16 + (lane >> 2), r1 = r0 + 8, cc = (lane & 3) * 2;
    const int ph0 = r0 & 7, ph1 = r1 & 7;
#pragma unroll
    for (int nt = 0; nt < 8; nt++) {
        const int c   = nt * 8 + cc;               
        const int atom = c >> 5, col32 = c & 31;   
        const int chunk = col32 >> 2, lo = col32 & 3;
        const int abase = atom * (64 * 32);
        
        uint32_t addr0 = (uint32_t)__cvta_generic_to_shared(&base[abase + r0 * 32 + ((chunk ^ ph0) << 2) + lo]);
        uint32_t addr1 = (uint32_t)__cvta_generic_to_shared(&base[abase + r1 * 32 + ((chunk ^ ph1) << 2) + lo]);
        float v0 = d[nt * 4 + 0] * scl, v1 = d[nt * 4 + 1] * scl;
        float v2 = d[nt * 4 + 2] * scl, v3 = d[nt * 4 + 3] * scl;
        asm volatile("st.shared.v2.f32 [%0], {%1, %2};\n" :: "r"(addr0), "f"(v0), "f"(v1) : "memory");
        asm volatile("st.shared.v2.f32 [%0], {%1, %2};\n" :: "r"(addr1), "f"(v2), "f"(v3) : "memory");
    }
}

template<int Br, int Bc, int D>
__global__ void __launch_bounds__(384, 1)
gqa_backward_v45_kv(
    const __grid_constant__ CUtensorMap tma_K_sw,
    const __grid_constant__ CUtensorMap tma_V_sw,
    const __grid_constant__ CUtensorMap tma_Q_sw,
    const __grid_constant__ CUtensorMap tma_dO_sw,
    const __grid_constant__ CUtensorMap tma_dV_st,
    const __grid_constant__ CUtensorMap tma_dK_st,
    const __grid_constant__ CUtensorMap tma_dq_red,   // fp32 dq_accum (SWIZZLE_128B, box 32x64) — swizzled TMA-reduce add target
    const float * __restrict__ d_Drow,
    const float * __restrict__ d_LSE,
    bf16 * __restrict__ d_dK, bf16 * __restrict__ d_dV,
    int B, int Hq, int Hkv, int G, int S, float scale
) {
    static_assert(Br == 64 && Bc == 64 && D == 128, "V45 requires Br=Bc=64, D=128");
    constexpr int CONS = 256;
    constexpr int PD   = 3;

    __shared__ __align__(128)  bf16 sK_sw[Bc * D];
    __shared__ __align__(128)  bf16 sV_sw[Bc * D];
    __shared__ __align__(128)  bf16 sQ_sw [PD][Br * D];
    __shared__ __align__(128)  bf16 sdO_sw[PD][Br * D];
    __shared__ __align__(1024) float sS [2][Br * 64];   // V45: DOUBLE-BUFFERED SWIZZLED dQ stage (wg0); each buf = 2 SW128B atoms of [Br*32]
    __shared__ __align__(1024) float sdP[2][Br * 64];   // V45: DOUBLE-BUFFERED SWIZZLED dQ stage (wg1); atom a at [a*Br*32], SW128B chunk^ (row&7)
    __shared__ __align__(1024) bf16  sP [Br * 64];
    __shared__ __align__(1024) bf16  sDS[Br * 64];
    __shared__                 float sLSE[PD][Br];
    __shared__                 float sD  [PD][Br];
    __shared__ __align__(8)    uint64_t mbar_kv;
    __shared__ __align__(8)    uint64_t full   [PD];
    __shared__ __align__(8)    uint64_t empty  [PD];
    __shared__ __align__(8)    uint64_t d_ready[PD];

    const int tid   = threadIdx.x;
    const int wg    = tid >> 7;
    const int wtid  = tid & 127;
    const int b = blockIdx.x, hkv = blockIdx.y, k_tile = blockIdx.z;
    const int k_row0 = k_tile * Bc;
    const int nQTiles = S / Br;

    const long     kvBase     = ((long)(b * Hkv + hkv) * S + k_row0) * D;
    const uint32_t kvFlatRow  = (uint32_t)((b * Hkv + hkv) * S + k_row0);
    const uint32_t bytesTile  = (uint32_t)(Br * D * sizeof(bf16));
    const uint32_t bytesAtom  = (uint32_t)(Bc * 64 * sizeof(bf16));

    if (tid == 0) {
        mbar_init_v4(&mbar_kv, 1);
        #pragma unroll
        for (int i = 0; i < PD; i++) {
            mbar_init_v4(&full[i], 1);
            mbar_init_v4(&empty[i], 1);
            mbar_init_v4(&d_ready[i], 1);
        }
    }
    __syncthreads();

    const int qc0    = k_row0 / Br;
    const int perG   = nQTiles - qc0;
    const int nIter  = G * perG;

    auto qFlatRowOf = [&](int g, int qc) -> uint32_t {
        const int hq = hkv * G + g;
        return (uint32_t)((b * Hq + hq) * S + qc * Br);
    };
    auto lBaseOf = [&](int g, int qc) -> long {
        const int hq = hkv * G + g;
        return (long)(b * Hq + hq) * S + (long)qc * Br;
    };

    if (wg == 2) {
        reg_dec_producer_v30();
        const bool leader = (tid == 256);
        const int  pl     = tid - 256;
        if (leader) {
            mbar_expect_tx_v4(&mbar_kv, bytesAtom * 4);
            tma_load_2d_v4(&tma_K_sw, sK_sw,           &mbar_kv, 0,  kvFlatRow);
            tma_load_2d_v4(&tma_K_sw, sK_sw + 64 * 64, &mbar_kv, 64, kvFlatRow);
            tma_load_2d_v4(&tma_V_sw, sV_sw,           &mbar_kv, 0,  kvFlatRow);
            tma_load_2d_v4(&tma_V_sw, sV_sw + 64 * 64, &mbar_kv, 64, kvFlatRow);
        }
        uint32_t epar[PD] = {0};
        int gP = 0, qcP = qc0;
        for (int it = 0; it < nIter; it++) {
            const int s = it % PD;
            if (it >= PD) { mbar_wait_v4(&empty[s], epar[s]); epar[s] ^= 1; }
            const long dbase = lBaseOf(gP, qcP);
            if (leader) {
                const uint32_t r = qFlatRowOf(gP, qcP);
                mbar_expect_tx_v4(&full[s], bytesTile * 2);
                tma_load_2d_v4(&tma_Q_sw,  sQ_sw [s],           &full[s], 0,  r);
                tma_load_2d_v4(&tma_Q_sw,  sQ_sw [s] + 64 * 64, &full[s], 64, r);
                tma_load_2d_v4(&tma_dO_sw, sdO_sw[s],           &full[s], 0,  r);
                tma_load_2d_v4(&tma_dO_sw, sdO_sw[s] + 64 * 64, &full[s], 64, r);
            }
            if (pl < Br) { sD[s][pl] = d_Drow[dbase + pl]; sLSE[s][pl] = d_LSE[dbase + pl]; }
            producer_sync();
            if (leader) mbar_arrive_v11(&d_ready[s]);
            if (++qcP == nQTiles) { qcP = qc0; ++gP; }
        }
        return;
    }

    reg_inc_consumer_v30();
    const float scale2 = scale * LOG2E_V29;
    mbar_wait_v4(&mbar_kv, 0);

    // V42: hoist the invariant-buffer wgmma base descriptors — computed once, reused every kv-tile.
    const uint64_t descGemmB = make_desc_sw128_K((wg == 0) ? sK_sw : sV_sw);   // S/dP GEMM B (K-major)
    const uint64_t descP     = make_desc_sw128_MN(sP);                          // dV A (Major::MN)
    const uint64_t descDSmn  = make_desc_sw128_MN(sDS);                         // dK A (Major::MN)
    const uint64_t descDSk   = make_desc_sw128_K (sDS);                         // dQ A (Major::K)
    const uint64_t descKhalf = make_desc_sw128_MN(sK_sw + wg * 4096);           // dQ B (Major::MN)

    float dv[32]; zeroN<32>(dv);
    float dk[32]; zeroN<32>(dk);

    uint32_t cpar[PD] = {0}, dpar[PD] = {0};
    int gC = 0, qcC = qc0;
    for (int it = 0; it < nIter; it++) {
        const int s = it % PD;
        const int q_row0 = qcC * Br;
        mbar_wait_v4(&full[s], cpar[s]); cpar[s] ^= 1;
        mbar_wait_v4(&d_ready[s], dpar[s]); dpar[s] ^= 1;

        float dPacc[32];
        if (wg == 0) {
            float acc[32]; zeroN<32>(acc);
            run_gemm_n64_sw2_hoB(acc, sQ_sw[s], descGemmB);
            if (qcC == qc0) fused_p_stsm<Bc, true >(acc, sP, sLSE[s], wtid, q_row0, k_row0, scale2);
            else            fused_p_stsm<Bc, false>(acc, sP, sLSE[s], wtid, 0,       0,      scale2);
        } else {
            zeroN<32>(dPacc);
            run_gemm_n64_sw2_hoB(dPacc, sdO_sw[s], descGemmB);
        }
        consumer_sync();

        run_gemm_dVdK_half_te_issue_hoA(dv, descP, sdO_sw[s] + wg * 4096);

        if (wg == 1) fuse_dS_ldstsm<Bc>(sP, dPacc, sD[s], sDS, wtid);
        consumer_sync();

        float dq[32]; zeroN<32>(dq);
        run_gemm_dKdQ_te_issue_ho(dk, dq, descDSmn, descDSk, descKhalf, sQ_sw[s] + wg * 4096);
        run_gemm_dVdKdQ_te_wait(dv, dk, dq);
        
        const int db = it & 1;                                        
        float* stageDQ = (wg == 0) ? sS[db] : sdP[db];
        
        store_acc_sw128_f32_v45(dq, stageDQ, wtid, scale); 

        if (wg == 0) consumer_sync_wg0(); else consumer_sync_wg1();
        fence_proxy_async_shared();                 
        if (wtid == 0) {                            
            const uint32_t crow = (uint32_t)lBaseOf(gC, qcC);
            tma_reduce_add_2d_v43(&tma_dq_red, stageDQ,             (uint32_t)(wg * 64),      crow);   
            tma_reduce_add_2d_v43(&tma_dq_red, stageDQ + 64 * 32,   (uint32_t)(wg * 64 + 32), crow);   
            tma_store_commit_v34();
            tma_bulk_wait1_v43();                   
        }
        consumer_sync();
        if (tid == 0) mbar_arrive_v11(&empty[s]);
        if (++qcC == nQTiles) { qcC = qc0; ++gC; }
    }

    bf16 *qflat    = reinterpret_cast<bf16*>(&sQ_sw[0][0]);
    bf16 *stage_dv = qflat + wg * 4096;
    bf16 *stage_dk = qflat + 8192 + wg * 4096;
    fence_operandN<32>(dv);
    stage_acc_bf16_s<64, 64>(dv, stage_dv, wtid, 1.0f);
    fence_operandN<32>(dk);
    stage_acc_bf16_s<64, 64>(dk, stage_dk, wtid, scale);
    consumer_sync();
    fence_proxy_async_shared();
    if (wtid == 0) {
        tma_store_2d_v34(&tma_dV_st, stage_dv, (uint32_t)(wg * 64), kvFlatRow);
        tma_store_2d_v34(&tma_dK_st, stage_dk, (uint32_t)(wg * 64), kvFlatRow);
        tma_store_commit_v34();
        tma_store_wait_v34();
    }
}

template<int Br, int Bc, int D>
void launch_gqa_backward_v45(
    const bf16 *d_Q, const bf16 *d_K, const bf16 *d_V, const bf16 *d_O,
    const bf16 *d_dO, const float *d_LSE,
    bf16 *d_dQ, bf16 *d_dK, bf16 *d_dV,
    int B, int Hq, int Hkv, int G, int S, float scale
) {
    static_assert(Br == 64 && Bc == 64 && D == 128, "V45 requires Br=Bc=64, D=128");
    auto make_tma_sw128 = [&](const bf16* ptr, uint64_t total_rows, uint32_t tile_rows) {
        CUtensorMap desc{};
        uint64_t gSize[2]   = {(uint64_t)D, total_rows};
        uint64_t gStride[1] = {(uint64_t)D * sizeof(bf16)};
        uint32_t box[2]     = {64u, tile_rows};
        uint32_t eStride[2] = {1, 1};
        CUresult r = cuTensorMapEncodeTiled(
            &desc, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)ptr,
            gSize, gStride, box, eStride,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (r != CUDA_SUCCESS) { const char* e; cuGetErrorString(r, &e);
            fprintf(stderr, "cuTensorMapEncodeTiled(sw128) failed: %s\n", e); exit(1); }
        return desc;
    };
    const uint64_t Rq  = (uint64_t)B * Hq  * S;
    const uint64_t Rkv = (uint64_t)B * Hkv * S;
    CUtensorMap tma_K_sw  = make_tma_sw128(d_K,  Rkv, Bc);
    CUtensorMap tma_V_sw  = make_tma_sw128(d_V,  Rkv, Bc);
    CUtensorMap tma_Q_sw  = make_tma_sw128(d_Q,  Rq,  Br);
    CUtensorMap tma_dO_sw = make_tma_sw128(d_dO, Rq,  Br);
    auto make_tma_out = [&](const bf16* ptr, uint64_t rows) {
        CUtensorMap desc{};
        uint64_t gSize[2]={(uint64_t)D, rows}; uint64_t gStride[1]={(uint64_t)D*sizeof(bf16)};
        uint32_t box[2]={64u,64u}; uint32_t eStride[2]={1,1};
        CUresult r=cuTensorMapEncodeTiled(&desc,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,(void*)ptr,gSize,gStride,box,eStride,
            CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if(r!=CUDA_SUCCESS){const char*e;cuGetErrorString(r,&e);fprintf(stderr,"tma_out v42: %s\n",e);exit(1);} return desc; };
    CUtensorMap tma_dV_st = make_tma_out(d_dV, Rkv);
    CUtensorMap tma_dK_st = make_tma_out(d_dK, Rkv);
    // fp32 dq_accum swizzled TMA-reduce descriptor. SW128B needs box inner = 32 fp32 (=128B atom); the 64-wide
    // dQ D-half is reduced as TWO 32-wide atoms. Probe-confirmed: FP32 SW128B box{64,64} is rejected, box{32,64} OK.
    auto make_tma_red = [&](const float* ptr, uint64_t rows) {
        CUtensorMap desc{};
        uint64_t gSize[2]={(uint64_t)D, rows}; uint64_t gStride[1]={(uint64_t)D*sizeof(float)};
        uint32_t box[2]={32u,64u}; uint32_t eStride[2]={1,1};
        CUresult r=cuTensorMapEncodeTiled(&desc,CU_TENSOR_MAP_DATA_TYPE_FLOAT32,2,(void*)ptr,gSize,gStride,box,eStride,
            CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if(r!=CUDA_SUCCESS){const char*e;cuGetErrorString(r,&e);fprintf(stderr,"tma_red v45: %s\n",e);exit(1);} return desc; };

    const long drowN = (long)B * Hq * S;
    static float* d_Drow  = nullptr;
    static long   drow_cap = 0;
    if (drowN > drow_cap) {
        if (d_Drow) CUDA_CHECK(cudaFree(d_Drow));
        CUDA_CHECK(cudaMalloc(&d_Drow, drowN * sizeof(float)));
        drow_cap = drowN;
    }
    const long dqN = (long)B * Hq * S * D;
    static float* d_dq_accum = nullptr;
    static long   dq_cap     = 0;
    if (dqN > dq_cap) {
        if (d_dq_accum) CUDA_CHECK(cudaFree(d_dq_accum));
        CUDA_CHECK(cudaMalloc(&d_dq_accum, dqN * sizeof(float)));
        dq_cap = dqN;
    }
    CUDA_CHECK(cudaMemset(d_dq_accum, 0, dqN * sizeof(float)));
    CUtensorMap tma_dq_red = make_tma_red(d_dq_accum, (uint64_t)B * Hq * S);

    const int  dBlock = 256;
    const long dGrid  = (drowN + (dBlock / 32) - 1) / (dBlock / 32);
    compute_drowsum_v22<<<(unsigned)dGrid, dBlock>>>(d_dO, d_O, d_Drow, drowN);

    constexpr dim3 BLOCK(384);
    dim3 GRID(B, Hkv, S / Bc);
    gqa_backward_v45_kv<Br,Bc,D><<<GRID, BLOCK>>>(
        tma_K_sw, tma_V_sw, tma_Q_sw, tma_dO_sw, tma_dV_st, tma_dK_st, tma_dq_red,
        d_Drow, d_LSE, d_dK, d_dV, B, Hq, Hkv, G, S, scale);

    const int convBlock = 256;
    const int convGrid  = (int)((dqN + convBlock - 1) / convBlock);
    convert_dq_accum_to_bf16_v5<<<convGrid, convBlock>>>(d_dq_accum, d_dQ, dqN);
}


// ═════════════════════════════════════════════════════════════════════════════
// main — precision-check + latency-benchmark for B=2 and B=4
// ═════════════════════════════════════════════════════════════════════════════
static void run_for_B(int B) {
    constexpr int Hq = 12, Hkv = 4, G = Hq / Hkv;
    constexpr int S = 4096, D = 128;
    constexpr int Br2 = 64, Bc2 = 64;
    const float scale = 1.0f / sqrtf((float)D);

    const size_t Nq   = (size_t)B * Hq  * S * D;
    const size_t Nkv  = (size_t)B * Hkv * S * D;
    const size_t Nlse = (size_t)B * Hq  * S;

    auto fileSz = [](const char *p, size_t n) -> bool {
        FILE *f = fopen(p, "rb"); if (!f) return false;
        fseek(f, 0, SEEK_END); size_t b = (size_t)ftell(f); fclose(f);
        return b == n * sizeof(float);
    };
    if (!fileSz("data/gqa_q.bin",   Nq)  ||
        !fileSz("data/gqa_k.bin",   Nkv) ||
        !fileSz("data/gqa_v.bin",   Nkv) ||
        !fileSz("data/gqa_o.bin",   Nq)  ||
        !fileSz("data/gqa_do.bin",  Nq)  ||
        !fileSz("data/gqa_lse.bin", Nlse)||
        !fileSz("data/gqa_dq.bin",  Nq)  ||
        !fileSz("data/gqa_dk.bin",  Nkv) ||
        !fileSz("data/gqa_dv.bin",  Nkv)) {
        printf("[B=%d] reference size mismatch — run baseline_gqa.py for B=%d, skipping\n", B, B);
        return;
    }

    std::vector<__nv_bfloat16> h_Q(Nq), h_K(Nkv), h_V(Nkv), h_O(Nq), h_dO(Nq);
    std::vector<float>         h_LSE(Nlse);
    std::vector<float>         h_dQ_ref(Nq), h_dK_ref(Nkv), h_dV_ref(Nkv);

    auto loadBF16 = [](const char *p, std::vector<__nv_bfloat16> &dst, size_t n){
        std::vector<float> tmp(n); loadBin(p, tmp.data(), n);
        for (size_t i = 0; i < n; ++i) dst[i] = __float2bfloat16(tmp[i]);
    };
    loadBF16("data/gqa_q.bin",  h_Q,  Nq);
    loadBF16("data/gqa_k.bin",  h_K,  Nkv);
    loadBF16("data/gqa_v.bin",  h_V,  Nkv);
    loadBF16("data/gqa_o.bin",  h_O,  Nq);
    loadBF16("data/gqa_do.bin", h_dO, Nq);
    loadBin("data/gqa_lse.bin", h_LSE.data(),    Nlse);
    loadBin("data/gqa_dq.bin",  h_dQ_ref.data(), Nq);
    loadBin("data/gqa_dk.bin",  h_dK_ref.data(), Nkv);
    loadBin("data/gqa_dv.bin",  h_dV_ref.data(), Nkv);

    __nv_bfloat16 *d_Q, *d_K, *d_V, *d_O, *d_dO, *d_dQ, *d_dK, *d_dV;
    float *d_LSE;
    CUDA_CHECK(cudaMalloc(&d_Q,   Nq   * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_K,   Nkv  * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_V,   Nkv  * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_O,   Nq   * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_dO,  Nq   * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_LSE, Nlse * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dQ,  Nq   * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_dK,  Nkv  * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_dV,  Nkv  * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemcpy(d_Q,   h_Q.data(),   Nq   * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K,   h_K.data(),   Nkv  * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V,   h_V.data(),   Nkv  * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_O,   h_O.data(),   Nq   * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dO,  h_dO.data(),  Nq   * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_LSE, h_LSE.data(), Nlse * sizeof(float),         cudaMemcpyHostToDevice));

    launch_gqa_backward_v45<Br2,Bc2,D>(d_Q,d_K,d_V,d_O,d_dO,d_LSE,d_dQ,d_dK,d_dV,B,Hq,Hkv,G,S,scale);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());

    {
        std::vector<__nv_bfloat16> hQ_(Nq), hK_(Nkv), hV_(Nkv);
        CUDA_CHECK(cudaMemcpy(hQ_.data(), d_dQ, Nq  * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hK_.data(), d_dK, Nkv * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hV_.data(), d_dV, Nkv * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        std::vector<float> qf(Nq), kf(Nkv), vf(Nkv);
        for (size_t i = 0; i < Nq;  ++i) qf[i] = __bfloat162float(hQ_[i]);
        for (size_t i = 0; i < Nkv; ++i) kf[i] = __bfloat162float(hK_[i]);
        for (size_t i = 0; i < Nkv; ++i) vf[i] = __bfloat162float(hV_[i]);
        printf("── V45 standalone  B=%d ──\n", B);
        reportPrecision("  dQ", h_dQ_ref.data(), qf.data(), Nq);
        reportPrecision("  dK", h_dK_ref.data(), kf.data(), Nkv);
        reportPrecision("  dV", h_dV_ref.data(), vf.data(), Nkv);
        std::cout << "  dQ: "; checkResult(h_dQ_ref.data(), qf.data(), Nq,  2e-2f, 2e-2f);
        std::cout << "  dK: "; checkResult(h_dK_ref.data(), kf.data(), Nkv, 2e-2f, 2e-2f);
        std::cout << "  dV: "; checkResult(h_dV_ref.data(), vf.data(), Nkv, 2e-2f, 2e-2f);
        std::cout << "\n";
    }

    const long long bwd_flops = 16LL * B * Hq * (long long)S * S * D;
    {
        char label[128];
        snprintf(label, sizeof(label), "V45 standalone  B=%d  swizzled TMA-reduce dQ  (Hopper SM_90)", B);
        KernelStats s = benchmarkKernel(
            [&](){ launch_gqa_backward_v45<Br2,Bc2,D>(
                d_Q,d_K,d_V,d_O,d_dO,d_LSE,d_dQ,d_dK,d_dV,B,Hq,Hkv,G,S,scale); },
            100, 10, bwd_flops);
        displayStats(label, s);
    }

    CUDA_CHECK(cudaFree(d_Q));  CUDA_CHECK(cudaFree(d_K));  CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));  CUDA_CHECK(cudaFree(d_dO)); CUDA_CHECK(cudaFree(d_LSE));
    CUDA_CHECK(cudaFree(d_dQ)); CUDA_CHECK(cudaFree(d_dK)); CUDA_CHECK(cudaFree(d_dV));
}

int main(){
    std::cout << "GQA Backward V45 — standalone precision + latency  [Hopper SM_90 / H200]\n";
    std::cout << "Prerequisite: python precision/baseline_gqa.py (per B)\n\n";
    run_for_B(2);
    run_for_B(4);
    run_for_B(8);
    return 0;
}
