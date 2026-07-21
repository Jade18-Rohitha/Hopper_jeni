// ─────────────────────────────────────────────────────────────────────────────
// GQA Backward — precision test  (Hopper SM_90 / H200)
//
// Two portable WMMA versions, no tcgen05 / TMA — run on any SM_70+ device:
//   V1  Br=16, Bc=32   single-warp (32 threads) per block
//   V2  Br=64, Bc=64   4-warp (128 threads) per block
//
// Both check bf16 dQ/dK/dV against the PyTorch reference produced by
//   python precision/baseline_gqa.py
// ─────────────────────────────────────────────────────────────────────────────
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>
#include <cassert>
#include <cmath>
#include <cuda_bf16.h>
#include "utils/kernelUtils.cuh"
#include "utils/kernelBench.cuh"

using namespace nvcuda;

// ─────────────────────────────────────────────────────────────────────────────
// V1 — Kernel 1 — dQ
// Grid  : (B, Hq, S/Br)  — each block owns one Q tile exclusively
// Block : 32 threads (one warp)
// ─────────────────────────────────────────────────────────────────────────────
template<int Br, int Bc, int D>
__global__ void gqa_backward_dQ(
    const __nv_bfloat16 *d_Q,
    const __nv_bfloat16 *d_K,
    const __nv_bfloat16 *d_V,
    const __nv_bfloat16 *d_O,
    const __nv_bfloat16 *d_dO,
    const float         *d_LSE,
          __nv_bfloat16 *d_dQ,
    int B, int Hq, int Hkv, int G, int S, float scale
){
    const int b       = blockIdx.x;
    const int hq      = blockIdx.y;
    const int q_tile  = blockIdx.z;
    const int hkv     = hq / G;
    const int lane    = threadIdx.x;
    const int q_row0  = q_tile * Br;
    const int nKTiles = S / Bc;

    const long qBase  = ((long)(b * Hq  + hq)  * S + q_row0) * D;
    const long kvBase = ((long)(b * Hkv + hkv) * S) * D;
    const long lBase  =  (long)(b * Hq  + hq)  * S + q_row0;

    __shared__ __align__(16) __nv_bfloat16 sQ  [Br * D];
    __shared__ __align__(16) __nv_bfloat16 sdO [Br * D];
    __shared__ __align__(16) __nv_bfloat16 sO  [Br * D];
    __shared__ __align__(16) __nv_bfloat16 sK  [Bc * D];
    __shared__ __align__(16) __nv_bfloat16 sV  [Bc * D];
    __shared__ __align__(16) float          sS  [Br * Bc];
    __shared__ __align__(16) float          sdP [Br * Bc];
    __shared__ __align__(16) __nv_bfloat16 sP  [Br * Bc];
    __shared__ __align__(16) __nv_bfloat16 sdS [Br * Bc];
    __shared__ float sLSE[Br];
    __shared__ float sD  [Br];

    for(int i = lane; i < Br * D; i += 32){
        sQ [i] = d_Q [qBase + i];
        sdO[i] = d_dO[qBase + i];
        sO [i] = d_O [qBase + i];
    }
    if(lane < Br) sLSE[lane] = d_LSE[lBase + lane];
    __syncwarp();

    if(lane < Br){
        float d = 0.0f;
        for(int j = 0; j < D; j++)
            d += __bfloat162float(sdO[lane * D + j]) *
                 __bfloat162float(sO [lane * D + j]);
        sD[lane] = d;
    }
    __syncwarp();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dQ_acc[D / 16];
    for(int t = 0; t < D / 16; t++) wmma::fill_fragment(dQ_acc[t], 0.0f);

    for(int kc = 0; kc < nKTiles; kc++){
        if (kc * Bc >= q_row0 + Br) break;   // causal: all remaining K tiles are fully masked

        const long kBase = kvBase + (long)kc * Bc * D;

        for(int i = lane; i < Bc * D; i += 32){
            sK[i] = d_K[kBase + i];
            sV[i] = d_V[kBase + i];
        }
        __syncwarp();

        // S = Q @ K^T * scale
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> qf;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> kf;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            for(int nt = 0; nt < Bc / 16; nt++){
                wmma::fill_fragment(acc, 0.0f);
                for(int kt = 0; kt < D / 16; kt++){
                    wmma::load_matrix_sync(qf, sQ + kt * 16,            D);
                    wmma::load_matrix_sync(kf, sK + nt * 16 * D + kt * 16, D);
                    wmma::mma_sync(acc, qf, kf, acc);
                }
                for(int t = 0; t < acc.num_elements; t++) acc.x[t] *= scale;
                wmma::store_matrix_sync(sS + nt * 16, acc, Bc, wmma::mem_row_major);
            }
        }
        __syncwarp();

        // P = exp(S - LSE) with causal mask: P[r,j] = 0 if K-col > Q-row
        if(lane < Br){
            float lse = sLSE[lane];
            int global_row = q_row0 + lane;
            for(int j = 0; j < Bc; j++){
                int global_col = kc * Bc + j;
                sP[lane * Bc + j] = (global_col > global_row)
                    ? __float2bfloat16(0.f)
                    : __float2bfloat16(expf(sS[lane * Bc + j] - lse));
            }
        }
        __syncwarp();

        // dP = dO @ V^T
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> dof;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> vf;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            for(int nt = 0; nt < Bc / 16; nt++){
                wmma::fill_fragment(acc, 0.0f);
                for(int kt = 0; kt < D / 16; kt++){
                    wmma::load_matrix_sync(dof, sdO + kt * 16,             D);
                    wmma::load_matrix_sync(vf,  sV  + nt * 16 * D + kt * 16, D);
                    wmma::mma_sync(acc, dof, vf, acc);
                }
                wmma::store_matrix_sync(sdP + nt * 16, acc, Bc, wmma::mem_row_major);
            }
        }
        __syncwarp();

        // dS = P ⊙ (dP - D)
        if(lane < Br){
            float d = sD[lane];
            for(int j = 0; j < Bc; j++){
                float p  = __bfloat162float(sP [lane * Bc + j]);
                float dp = sdP[lane * Bc + j];
                sdS[lane * Bc + j] = __float2bfloat16(p * (dp - d));
            }
        }
        __syncwarp();

        // dQ_acc += dS @ K
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> dsf;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> kf;
            for(int nt = 0; nt < D / 16; nt++){
                for(int kt = 0; kt < Bc / 16; kt++){
                    wmma::load_matrix_sync(dsf, sdS + kt * 16,             Bc);
                    wmma::load_matrix_sync(kf,  sK  + kt * 16 * D + nt * 16, D);
                    wmma::mma_sync(dQ_acc[nt], dsf, kf, dQ_acc[nt]);
                }
            }
        }
        __syncwarp();
    }

    for(int nt = 0; nt < D / 16; nt++){
        for(int t = 0; t < dQ_acc[nt].num_elements; t++)
            dQ_acc[nt].x[t] *= scale;
        wmma::store_matrix_sync(sdP, dQ_acc[nt], 16, wmma::mem_row_major);
        __syncwarp();
        for(int i = lane; i < Br * 16; i += 32){
            int r = i / 16, c = i % 16;
            d_dQ[qBase + r * D + nt * 16 + c] = __float2bfloat16(sdP[i]);
        }
        __syncwarp();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// V1 — Kernel 2 — dK, dV
// Grid  : (B, Hkv, S/Bc)  — each block owns one K tile exclusively
// Block : 32 threads (one warp)
// ─────────────────────────────────────────────────────────────────────────────
template<int Br, int Bc, int D>
__global__ void gqa_backward_dKdV(
    const __nv_bfloat16 *d_Q,
    const __nv_bfloat16 *d_K,
    const __nv_bfloat16 *d_V,
    const __nv_bfloat16 *d_O,
    const __nv_bfloat16 *d_dO,
    const float         *d_LSE,
          __nv_bfloat16 *d_dK,
          __nv_bfloat16 *d_dV,
    int B, int Hq, int Hkv, int G, int S, float scale
){
    const int b      = blockIdx.x;
    const int hkv    = blockIdx.y;
    const int k_tile = blockIdx.z;
    const int lane   = threadIdx.x;
    const int k_row0 = k_tile * Bc;
    const int nQTiles = S / Br;

    const long kvBase = ((long)(b * Hkv + hkv) * S + k_row0) * D;

    __shared__ __align__(16) __nv_bfloat16 sK  [Bc * D];
    __shared__ __align__(16) __nv_bfloat16 sV  [Bc * D];
    __shared__ __align__(16) __nv_bfloat16 sQ  [Br * D];
    __shared__ __align__(16) __nv_bfloat16 sdO [Br * D];
    __shared__ __align__(16) __nv_bfloat16 sO  [Br * D];
    __shared__ __align__(16) float          sS  [Br * Bc];
    __shared__ __align__(16) float          sdP [Br * Bc];
    __shared__ __align__(16) __nv_bfloat16 sP  [Br * Bc];
    __shared__ __align__(16) __nv_bfloat16 sdS [Br * Bc];
    __shared__ float sLSE[Br];
    __shared__ float sD  [Br];

    for(int i = lane; i < Bc * D; i += 32){
        sK[i] = d_K[kvBase + i];
        sV[i] = d_V[kvBase + i];
    }
    __syncwarp();

    constexpr int nAccTiles = (Bc / 16) * (D / 16);
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dV_acc[nAccTiles];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dK_acc[nAccTiles];
    for(int t = 0; t < nAccTiles; t++){
        wmma::fill_fragment(dV_acc[t], 0.0f);
        wmma::fill_fragment(dK_acc[t], 0.0f);
    }

    for(int g = 0; g < G; g++){
        const int hq = hkv * G + g;
        for(int qc = k_row0 / Br; qc < nQTiles; qc++){
            const int  q_row0 = qc * Br;
            const long qBase  = ((long)(b * Hq + hq) * S + q_row0) * D;
            const long lBase  =  (long)(b * Hq + hq) * S + q_row0;

            for(int i = lane; i < Br * D; i += 32){
                sQ [i] = d_Q [qBase + i];
                sdO[i] = d_dO[qBase + i];
                sO [i] = d_O [qBase + i];
            }
            if(lane < Br) sLSE[lane] = d_LSE[lBase + lane];
            __syncwarp();

            if(lane < Br){
                float d = 0.0f;
                for(int j = 0; j < D; j++)
                    d += __bfloat162float(sdO[lane * D + j]) *
                         __bfloat162float(sO [lane * D + j]);
                sD[lane] = d;
            }
            __syncwarp();

            // S = Q @ K^T * scale
            {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> qf;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> kf;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
                for(int nt = 0; nt < Bc / 16; nt++){
                    wmma::fill_fragment(acc, 0.0f);
                    for(int kt = 0; kt < D / 16; kt++){
                        wmma::load_matrix_sync(qf, sQ + kt * 16,            D);
                        wmma::load_matrix_sync(kf, sK + nt * 16 * D + kt * 16, D);
                        wmma::mma_sync(acc, qf, kf, acc);
                    }
                    for(int t = 0; t < acc.num_elements; t++) acc.x[t] *= scale;
                    wmma::store_matrix_sync(sS + nt * 16, acc, Bc, wmma::mem_row_major);
                }
            }
            __syncwarp();

            // P = exp(S - LSE) with causal mask: P[r,j] = 0 if K-col > Q-row
            if(lane < Br){
                float lse = sLSE[lane];
                int global_row = q_row0 + lane;
                for(int j = 0; j < Bc; j++){
                    int global_col = k_row0 + j;
                    sP[lane * Bc + j] = (global_col > global_row)
                        ? __float2bfloat16(0.f)
                        : __float2bfloat16(expf(sS[lane * Bc + j] - lse));
                }
            }
            __syncwarp();

            // dV_acc += P^T @ dO
            {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> pf;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> dof;
                for(int vt = 0; vt < Bc / 16; vt++){
                    for(int nt = 0; nt < D / 16; nt++){
                        for(int kt = 0; kt < Br / 16; kt++){
                            wmma::load_matrix_sync(pf,  sP  + kt * 16 * Bc + vt * 16, Bc);
                            wmma::load_matrix_sync(dof, sdO + kt * 16 * D  + nt * 16, D);
                            wmma::mma_sync(dV_acc[vt * (D/16) + nt], pf, dof,
                                           dV_acc[vt * (D/16) + nt]);
                        }
                    }
                }
            }
            __syncwarp();

            // dP = dO @ V^T
            {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> dof;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> vf;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
                for(int nt = 0; nt < Bc / 16; nt++){
                    wmma::fill_fragment(acc, 0.0f);
                    for(int kt = 0; kt < D / 16; kt++){
                        wmma::load_matrix_sync(dof, sdO + kt * 16,             D);
                        wmma::load_matrix_sync(vf,  sV  + nt * 16 * D + kt * 16, D);
                        wmma::mma_sync(acc, dof, vf, acc);
                    }
                    wmma::store_matrix_sync(sdP + nt * 16, acc, Bc, wmma::mem_row_major);
                }
            }
            __syncwarp();

            // dS = P ⊙ (dP - D)
            if(lane < Br){
                float d = sD[lane];
                for(int j = 0; j < Bc; j++){
                    float p  = __bfloat162float(sP [lane * Bc + j]);
                    float dp = sdP[lane * Bc + j];
                    sdS[lane * Bc + j] = __float2bfloat16(p * (dp - d));
                }
            }
            __syncwarp();

            // dK_acc += dS^T @ Q
            {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> dsf;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> qf;
                for(int vt = 0; vt < Bc / 16; vt++){
                    for(int nt = 0; nt < D / 16; nt++){
                        for(int kt = 0; kt < Br / 16; kt++){
                            wmma::load_matrix_sync(dsf, sdS + kt * 16 * Bc + vt * 16, Bc);
                            wmma::load_matrix_sync(qf,  sQ  + kt * 16 * D  + nt * 16, D);
                            wmma::mma_sync(dK_acc[vt * (D/16) + nt], dsf, qf,
                                           dK_acc[vt * (D/16) + nt]);
                        }
                    }
                }
            }
            __syncwarp();
        }
    }

    for(int vt = 0; vt < Bc / 16; vt++){
        for(int nt = 0; nt < D / 16; nt++){
            wmma::store_matrix_sync(sdP, dV_acc[vt * (D/16) + nt], 16, wmma::mem_row_major);
            __syncwarp();
            for(int i = lane; i < 16 * 16; i += 32){
                int r = i / 16, c = i % 16;
                d_dV[kvBase + (vt * 16 + r) * D + nt * 16 + c] = __float2bfloat16(sdP[i]);
            }
            __syncwarp();

            for(int t = 0; t < dK_acc[vt * (D/16) + nt].num_elements; t++)
                dK_acc[vt * (D/16) + nt].x[t] *= scale;
            wmma::store_matrix_sync(sdP, dK_acc[vt * (D/16) + nt], 16, wmma::mem_row_major);
            __syncwarp();
            for(int i = lane; i < 16 * 16; i += 32){
                int r = i / 16, c = i % 16;
                d_dK[kvBase + (vt * 16 + r) * D + nt * 16 + c] = __float2bfloat16(sdP[i]);
            }
            __syncwarp();
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// V1 Launcher
// ─────────────────────────────────────────────────────────────────────────────
template<int Br, int Bc, int D>
void launch_gqa_backward_v1(
    const __nv_bfloat16 *d_Q,
    const __nv_bfloat16 *d_K,
    const __nv_bfloat16 *d_V,
    const __nv_bfloat16 *d_O,
    const __nv_bfloat16 *d_dO,
    const float         *d_LSE,
          __nv_bfloat16 *d_dQ,
          __nv_bfloat16 *d_dK,
          __nv_bfloat16 *d_dV,
    int B, int Hq, int Hkv, int G, int S, float scale
){
    static_assert(Br % 16 == 0, "Br must be a multiple of 16");
    static_assert(Bc % 16 == 0, "Bc must be a multiple of 16");
    static_assert(D  % 16 == 0, "D  must be a multiple of 16");

    dim3 BLOCK(32);
    dim3 GRID1(B, Hq,  S / Br);
    dim3 GRID2(B, Hkv, S / Bc);

    gqa_backward_dQ<Br, Bc, D><<<GRID1, BLOCK>>>(
        d_Q, d_K, d_V, d_O, d_dO, d_LSE, d_dQ,
        B, Hq, Hkv, G, S, scale
    );
    gqa_backward_dKdV<Br, Bc, D><<<GRID2, BLOCK>>>(
        d_Q, d_K, d_V, d_O, d_dO, d_LSE, d_dK, d_dV,
        B, Hq, Hkv, G, S, scale
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// V2 — WMMA 4 warps (Br=64, Bc=64) — SM_70+ compatible, no wgmma
// ─────────────────────────────────────────────────────────────────────────────

template<int Br, int Bc, int D>
__global__ void __launch_bounds__(128, 1)
gqa_backward_dQ_v2(
    const __nv_bfloat16 * __restrict__ d_Q,
    const __nv_bfloat16 * __restrict__ d_K,
    const __nv_bfloat16 * __restrict__ d_V,
    const __nv_bfloat16 * __restrict__ d_O,
    const __nv_bfloat16 * __restrict__ d_dO,
    const float         * __restrict__ d_LSE,
          __nv_bfloat16 * __restrict__ d_dQ,
    int B, int Hq, int Hkv, int G, int S, float scale
) {
    static_assert(Br == 64 && Bc == 64 && D == 64, "V2 requires Br=Bc=D=64");

    __shared__ __align__(16) __nv_bfloat16 sQ  [Br * D];
    __shared__ __align__(16) __nv_bfloat16 sdO [Br * D];
    __shared__ __align__(16) __nv_bfloat16 sO  [Br * D];
    __shared__ __align__(16) __nv_bfloat16 sK  [Bc * D];
    __shared__ __align__(16) __nv_bfloat16 sV  [Bc * D];
    __shared__ __align__(16) float          sS  [Br * Bc];
    __shared__ __align__(16) float          sdP [Br * Bc];
    __shared__ __align__(16) __nv_bfloat16 sP  [Br * Bc]; // reused as sdS
    __shared__               float          sLSE[Br];
    __shared__               float          sD  [Br];

    const int tid    = threadIdx.x;
    const int lane   = tid % 32;
    const int warpId = tid / 32;
    const int row0   = warpId * 16;

    const int b       = blockIdx.x;
    const int hq      = blockIdx.y;
    const int q_tile  = blockIdx.z;
    const int hkv     = hq / G;
    const int q_row0  = q_tile * Br;
    const int nKTiles = S / Bc;

    const long qBase  = ((long)(b * Hq  + hq)  * S + q_row0) * D;
    const long kvBase = ((long)(b * Hkv + hkv) * S) * D;
    const long lBase  =  (long)(b * Hq  + hq)  * S + q_row0;

    for (int i = tid; i < Br * D; i += 128) {
        sQ [i] = d_Q [qBase + i];
        sdO[i] = d_dO[qBase + i];
        sO [i] = d_O [qBase + i];
    }
    if (tid < Br) sLSE[tid] = d_LSE[lBase + tid];
    __syncthreads();

    if (lane < 16) {
        int r = row0 + lane;
        float d = 0.f;
        for (int j = 0; j < D; j++)
            d += __bfloat162float(sdO[r * D + j]) * __bfloat162float(sO[r * D + j]);
        sD[r] = d;
    }
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dQ_acc[D / 16];
    for (int t = 0; t < D / 16; t++) wmma::fill_fragment(dQ_acc[t], 0.0f);

    for (int kc = 0; kc < nKTiles; kc++) {
        if (kc * Bc >= q_row0 + Br) break;   // causal: all remaining K tiles are fully masked

        const long kBase = kvBase + (long)kc * Bc * D;
        for (int i = tid; i < Bc * D; i += 128) {
            sK[i] = d_K[kBase + i];
            sV[i] = d_V[kBase + i];
        }
        __syncthreads();

        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> qf;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> kf;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            for (int nt = 0; nt < Bc / 16; nt++) {
                wmma::fill_fragment(acc, 0.0f);
                for (int kt = 0; kt < D / 16; kt++) {
                    wmma::load_matrix_sync(qf, sQ + row0 * D + kt * 16, D);
                    wmma::load_matrix_sync(kf, sK + nt * 16 * D + kt * 16, D);
                    wmma::mma_sync(acc, qf, kf, acc);
                }
                for (int t = 0; t < acc.num_elements; t++) acc.x[t] *= scale;
                wmma::store_matrix_sync(sS + row0 * Bc + nt * 16, acc, Bc, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // P = exp(S - LSE) with causal mask: P[r,j] = 0 if K-col > Q-row
        for (int i = tid; i < Br * Bc; i += 128) {
            int r = i / Bc, c = i % Bc;
            sP[i] = (kc * Bc + c > q_row0 + r)
                ? __float2bfloat16(0.f)
                : __float2bfloat16(expf(sS[i] - sLSE[r]));
        }
        __syncthreads();

        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> dof;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> vf;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            for (int nt = 0; nt < Bc / 16; nt++) {
                wmma::fill_fragment(acc, 0.0f);
                for (int kt = 0; kt < D / 16; kt++) {
                    wmma::load_matrix_sync(dof, sdO + row0 * D + kt * 16, D);
                    wmma::load_matrix_sync(vf,  sV  + nt * 16 * D + kt * 16, D);
                    wmma::mma_sync(acc, dof, vf, acc);
                }
                wmma::store_matrix_sync(sdP + row0 * Bc + nt * 16, acc, Bc, wmma::mem_row_major);
            }
        }
        __syncthreads();

        for (int i = tid; i < Br * Bc; i += 128) {
            int r = i / Bc;
            sP[i] = __float2bfloat16(__bfloat162float(sP[i]) * (sdP[i] - sD[r]));
        }
        __syncthreads();

        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> dsf;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> kf;
            for (int nt = 0; nt < D / 16; nt++) {
                for (int kt = 0; kt < Bc / 16; kt++) {
                    wmma::load_matrix_sync(dsf, sP + row0 * Bc + kt * 16, Bc);
                    wmma::load_matrix_sync(kf,  sK + kt * 16 * D + nt * 16, D);
                    wmma::mma_sync(dQ_acc[nt], dsf, kf, dQ_acc[nt]);
                }
            }
        }
        __syncthreads();
    }

    float * const warp_tmp = sdP + warpId * 16 * 16;
    for (int nt = 0; nt < D / 16; nt++) {
        for (int t = 0; t < dQ_acc[nt].num_elements; t++) dQ_acc[nt].x[t] *= scale;
        wmma::store_matrix_sync(warp_tmp, dQ_acc[nt], 16, wmma::mem_row_major);
        for (int i = lane; i < 16 * 16; i += 32) {
            int r = i / 16, c = i % 16;
            d_dQ[qBase + (row0 + r) * D + nt * 16 + c] = __float2bfloat16(warp_tmp[i]);
        }
    }
}

template<int Br, int Bc, int D>
__global__ void __launch_bounds__(128, 1)
gqa_backward_dKdV_v2(
    const __nv_bfloat16 * __restrict__ d_Q,
    const __nv_bfloat16 * __restrict__ d_K,
    const __nv_bfloat16 * __restrict__ d_V,
    const __nv_bfloat16 * __restrict__ d_O,
    const __nv_bfloat16 * __restrict__ d_dO,
    const float         * __restrict__ d_LSE,
          __nv_bfloat16 * __restrict__ d_dK,
          __nv_bfloat16 * __restrict__ d_dV,
    int B, int Hq, int Hkv, int G, int S, float scale
) {
    static_assert(Br == 64 && Bc == 64 && D == 64, "V2 requires Br=Bc=D=64");

    __shared__ __align__(16) __nv_bfloat16 sK  [Bc * D];
    __shared__ __align__(16) __nv_bfloat16 sV  [Bc * D];
    __shared__ __align__(16) __nv_bfloat16 sQ  [Br * D];
    __shared__ __align__(16) __nv_bfloat16 sdO [Br * D];
    __shared__ __align__(16) __nv_bfloat16 sO  [Br * D];
    __shared__ __align__(16) float          sS  [Br * Bc];
    __shared__ __align__(16) float          sdP [Br * Bc];
    __shared__ __align__(16) __nv_bfloat16 sP  [Br * Bc]; // reused as sdS
    __shared__               float          sLSE[Br];
    __shared__               float          sD  [Br];

    const int tid    = threadIdx.x;
    const int lane   = tid % 32;
    const int warpId = tid / 32;
    const int row0   = warpId * 16;

    const int b       = blockIdx.x;
    const int hkv     = blockIdx.y;
    const int k_tile  = blockIdx.z;
    const int k_row0  = k_tile * Bc;
    const int nQTiles = S / Br;

    const long kvBase = ((long)(b * Hkv + hkv) * S + k_row0) * D;

    for (int i = tid; i < Bc * D; i += 128) {
        sK[i] = d_K[kvBase + i];
        sV[i] = d_V[kvBase + i];
    }
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dV_acc[D / 16];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dK_acc[D / 16];
    for (int t = 0; t < D / 16; t++) {
        wmma::fill_fragment(dV_acc[t], 0.0f);
        wmma::fill_fragment(dK_acc[t], 0.0f);
    }

    for (int g = 0; g < G; g++) {
        const int hq = hkv * G + g;
        for (int qc = k_row0 / Br; qc < nQTiles; qc++) {
            const int  q_row0 = qc * Br;
            const long qBase  = ((long)(b * Hq + hq) * S + q_row0) * D;
            const long lBase  =  (long)(b * Hq + hq) * S + q_row0;

            for (int i = tid; i < Br * D; i += 128) {
                sQ [i] = d_Q [qBase + i];
                sdO[i] = d_dO[qBase + i];
                sO [i] = d_O [qBase + i];
            }
            if (tid < Br) sLSE[tid] = d_LSE[lBase + tid];
            __syncthreads();

            if (tid < Br) {
                float d = 0.f;
                for (int j = 0; j < D; j++)
                    d += __bfloat162float(sdO[tid * D + j]) * __bfloat162float(sO[tid * D + j]);
                sD[tid] = d;
            }
            __syncthreads();

            {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> qf;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> kf;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
                for (int nt = 0; nt < Bc / 16; nt++) {
                    wmma::fill_fragment(acc, 0.0f);
                    for (int kt = 0; kt < D / 16; kt++) {
                        wmma::load_matrix_sync(qf, sQ + row0 * D + kt * 16, D);
                        wmma::load_matrix_sync(kf, sK + nt * 16 * D + kt * 16, D);
                        wmma::mma_sync(acc, qf, kf, acc);
                    }
                    for (int t = 0; t < acc.num_elements; t++) acc.x[t] *= scale;
                    wmma::store_matrix_sync(sS + row0 * Bc + nt * 16, acc, Bc, wmma::mem_row_major);
                }
            }
            __syncthreads();

            // P = exp(S - LSE) with causal mask: P[r,j] = 0 if K-col > Q-row
            for (int i = tid; i < Br * Bc; i += 128) {
                int r = i / Bc, c = i % Bc;
                sP[i] = (k_row0 + c > q_row0 + r)
                    ? __float2bfloat16(0.f)
                    : __float2bfloat16(expf(sS[i] - sLSE[r]));
            }
            __syncthreads();

            // dV_acc += P^T @ dO  (col_major trick)
            {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> pf;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> dof;
                for (int nt = 0; nt < D / 16; nt++) {
                    for (int kt = 0; kt < Br / 16; kt++) {
                        wmma::load_matrix_sync(pf,  sP  + kt * 16 * Bc + row0, Bc);
                        wmma::load_matrix_sync(dof, sdO + kt * 16 * D  + nt * 16, D);
                        wmma::mma_sync(dV_acc[nt], pf, dof, dV_acc[nt]);
                    }
                }
            }

            // dP + sync
            {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> dof;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> vf;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
                for (int nt = 0; nt < Bc / 16; nt++) {
                    wmma::fill_fragment(acc, 0.0f);
                    for (int kt = 0; kt < D / 16; kt++) {
                        wmma::load_matrix_sync(dof, sdO + row0 * D + kt * 16, D);
                        wmma::load_matrix_sync(vf,  sV  + nt * 16 * D + kt * 16, D);
                        wmma::mma_sync(acc, dof, vf, acc);
                    }
                    wmma::store_matrix_sync(sdP + row0 * Bc + nt * 16, acc, Bc, wmma::mem_row_major);
                }
            }
            __syncthreads();

            for (int i = tid; i < Br * Bc; i += 128) {
                int r = i / Bc;
                sP[i] = __float2bfloat16(__bfloat162float(sP[i]) * (sdP[i] - sD[r]));
            }
            __syncthreads();

            // dK_acc += dS^T @ Q  (col_major trick)
            {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> dsf;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> qf;
                for (int nt = 0; nt < D / 16; nt++) {
                    for (int kt = 0; kt < Br / 16; kt++) {
                        wmma::load_matrix_sync(dsf, sP  + kt * 16 * Bc + row0, Bc);
                        wmma::load_matrix_sync(qf,  sQ  + kt * 16 * D  + nt * 16, D);
                        wmma::mma_sync(dK_acc[nt], dsf, qf, dK_acc[nt]);
                    }
                }
            }
            __syncthreads();
        }
    }

    float * const warp_tmp = sdP + warpId * 16 * 16;
    for (int nt = 0; nt < D / 16; nt++) {
        wmma::store_matrix_sync(warp_tmp, dV_acc[nt], 16, wmma::mem_row_major);
        for (int i = lane; i < 16 * 16; i += 32) {
            int r = i / 16, c = i % 16;
            d_dV[kvBase + (row0 + r) * D + nt * 16 + c] = __float2bfloat16(warp_tmp[i]);
        }
        for (int t = 0; t < dK_acc[nt].num_elements; t++) dK_acc[nt].x[t] *= scale;
        wmma::store_matrix_sync(warp_tmp, dK_acc[nt], 16, wmma::mem_row_major);
        for (int i = lane; i < 16 * 16; i += 32) {
            int r = i / 16, c = i % 16;
            d_dK[kvBase + (row0 + r) * D + nt * 16 + c] = __float2bfloat16(warp_tmp[i]);
        }
    }
}

template<int Br, int Bc, int D>
void launch_gqa_backward_v2(
    const __nv_bfloat16 *d_Q, const __nv_bfloat16 *d_K,
    const __nv_bfloat16 *d_V, const __nv_bfloat16 *d_O,
    const __nv_bfloat16 *d_dO, const float *d_LSE,
    __nv_bfloat16 *d_dQ, __nv_bfloat16 *d_dK, __nv_bfloat16 *d_dV,
    int B, int Hq, int Hkv, int G, int S, float scale
) {
    static_assert(Br == 64 && Bc == 64 && D == 64, "V2 requires Br=Bc=D=64");

    constexpr dim3 BLOCK(128);
    dim3 GRID1(B, Hq,  S / Br);
    dim3 GRID2(B, Hkv, S / Bc);
    gqa_backward_dQ_v2  <Br,Bc,D><<<GRID1, BLOCK>>>(
        d_Q, d_K, d_V, d_O, d_dO, d_LSE, d_dQ, B, Hq, Hkv, G, S, scale);
    gqa_backward_dKdV_v2<Br,Bc,D><<<GRID2, BLOCK>>>(
        d_Q, d_K, d_V, d_O, d_dO, d_LSE, d_dK, d_dV, B, Hq, Hkv, G, S, scale);
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main(){
    std::cout << "GQA Backward — precision test (V1 + V2 vs PyTorch reference)  [Hopper SM_90 / H200]\n";
    std::cout << "Prerequisite: python precision/baseline_gqa.py\n\n";

    constexpr int B   = 8, Hq  = 12, Hkv = 4, G = Hq / Hkv;
    constexpr int S   = 4096, D = 64, Br = 16, Bc = 32;
    constexpr int Br2 = 64, Bc2 = 64;

    const float scale = 1.0f / sqrtf((float)D);
    const size_t Nq   = (size_t)B * Hq  * S * D;
    const size_t Nkv  = (size_t)B * Hkv * S * D;
    const size_t Nlse = (size_t)B * Hq  * S;

    std::vector<__nv_bfloat16> h_Q(Nq), h_K(Nkv), h_V(Nkv), h_O(Nq), h_dO(Nq);
    std::vector<float>         h_LSE(Nlse);
    std::vector<float>         h_dQ_ref(Nq), h_dK_ref(Nkv), h_dV_ref(Nkv);

    auto fileSz = [](const char *p, size_t n) -> bool {
        FILE *f = fopen(p, "rb"); if (!f) return false;
        fseek(f, 0, SEEK_END); size_t b = (size_t)ftell(f); fclose(f);
        return b == n * sizeof(float);
    };
    auto loadBF16 = [](const char *p, std::vector<__nv_bfloat16> &dst, size_t n){
        std::vector<float> tmp(n); loadBin(p, tmp.data(), n);
        for (size_t i = 0; i < n; ++i) dst[i] = __float2bfloat16(tmp[i]);
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
        std::cerr << "ERROR: reference files not found. Run: python precision/baseline_gqa.py\n";
        return 1;
    }
    loadBF16("data/gqa_q.bin",  h_Q,  Nq);
    loadBF16("data/gqa_k.bin",  h_K,  Nkv);
    loadBF16("data/gqa_v.bin",  h_V,  Nkv);
    loadBF16("data/gqa_o.bin",  h_O,  Nq);
    loadBF16("data/gqa_do.bin", h_dO, Nq);
    loadBin("data/gqa_lse.bin", h_LSE.data(),    Nlse);
    loadBin("data/gqa_dq.bin",  h_dQ_ref.data(), Nq);
    loadBin("data/gqa_dk.bin",  h_dK_ref.data(), Nkv);
    loadBin("data/gqa_dv.bin",  h_dV_ref.data(), Nkv);
    std::cout << "Loaded PyTorch reference from data/gqa_*.bin\n\n";

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

    auto check = [&](const char *label, size_t Nd, size_t Nkv_,
                     __nv_bfloat16 *d_dQ_, __nv_bfloat16 *d_dK_, __nv_bfloat16 *d_dV_) {
        std::vector<__nv_bfloat16> hQ_(Nd), hK_(Nkv_), hV_(Nkv_);
        CUDA_CHECK(cudaMemcpy(hQ_.data(), d_dQ_, Nd   * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hK_.data(), d_dK_, Nkv_ * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hV_.data(), d_dV_, Nkv_ * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        std::vector<float> qf(Nd), kf(Nkv_), vf(Nkv_);
        for (size_t i = 0; i < Nd;   ++i) qf[i] = __bfloat162float(hQ_[i]);
        for (size_t i = 0; i < Nkv_; ++i) kf[i] = __bfloat162float(hK_[i]);
        for (size_t i = 0; i < Nkv_; ++i) vf[i] = __bfloat162float(hV_[i]);
        std::cout << label << "\n";
        reportPrecision("  dQ", h_dQ_ref.data(), qf.data(), Nd);
        reportPrecision("  dK", h_dK_ref.data(), kf.data(), Nkv_);
        reportPrecision("  dV", h_dV_ref.data(), vf.data(), Nkv_);
        std::cout << "  dQ: "; checkResult(h_dQ_ref.data(), qf.data(), Nd,   2e-2f, 2e-2f);
        std::cout << "  dK: "; checkResult(h_dK_ref.data(), kf.data(), Nkv_, 2e-2f, 2e-2f);
        std::cout << "  dV: "; checkResult(h_dV_ref.data(), vf.data(), Nkv_, 2e-2f, 2e-2f);
        std::cout << "\n";
    };

    launch_gqa_backward_v1<Br,Bc,D>(d_Q,d_K,d_V,d_O,d_dO,d_LSE,d_dQ,d_dK,d_dV,B,Hq,Hkv,G,S,scale);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    check("── V1  Br=16, Bc=32 ──", Nq, Nkv, d_dQ, d_dK, d_dV);

    launch_gqa_backward_v2<Br2,Bc2,D>(d_Q,d_K,d_V,d_O,d_dO,d_LSE,d_dQ,d_dK,d_dV,B,Hq,Hkv,G,S,scale);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    check("── V2  Br=64, Bc=64 ──", Nq, Nkv, d_dQ, d_dK, d_dV);

    // ─────────────────────────────────────────────────────────────────────────
    // Latency benchmark — full backward (dQ + dKdV kernels), median over 100
    // iterations with the L2 flushed between reps (mirrors triton.testing.do_bench).
    // FLOP convention matches baseline_gqa.py: bwd_flops = 4 * (4·B·Hq·S·S·D),
    // the non-causal algorithmic count, so these TFLOP/s are directly comparable
    // to the SDPA / Triton numbers that script prints (our kernels are causal, so
    // the effective work is ~½ that — treat TFLOP/s as a comparison metric, not
    // a hardware-utilization figure).
    // ─────────────────────────────────────────────────────────────────────────
    const long long bwd_flops = 16LL * B * Hq * (long long)S * S * D;

    std::cout << "=== Latency  (B=" << B << " Hq=" << Hq << " Hkv=" << Hkv
              << " S=" << S << " D=" << D << ", GQA G=" << G << ", bf16, causal) ===\n";
    {
        KernelStats s = benchmarkKernel(
            [&](){ launch_gqa_backward_v1<Br,Bc,D>(
                d_Q,d_K,d_V,d_O,d_dO,d_LSE,d_dQ,d_dK,d_dV,B,Hq,Hkv,G,S,scale); },
            100, 10, bwd_flops);
        displayStats("GQA bwd V1  Br=16, Bc=32  (Hopper SM_90)", s);
    }
    {
        KernelStats s = benchmarkKernel(
            [&](){ launch_gqa_backward_v2<Br2,Bc2,D>(
                d_Q,d_K,d_V,d_O,d_dO,d_LSE,d_dQ,d_dK,d_dV,B,Hq,Hkv,G,S,scale); },
            100, 10, bwd_flops);
        displayStats("GQA bwd V2  Br=64, Bc=64  (Hopper SM_90)", s);
    }

    CUDA_CHECK(cudaFree(d_Q));  CUDA_CHECK(cudaFree(d_K));  CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));  CUDA_CHECK(cudaFree(d_dO)); CUDA_CHECK(cudaFree(d_LSE));
    CUDA_CHECK(cudaFree(d_dQ)); CUDA_CHECK(cudaFree(d_dK)); CUDA_CHECK(cudaFree(d_dV));
    return 0;
}
