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
//D
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
//D
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
//D
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
//S
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
//dP
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
//dS
        for (int i = tid; i < Br * Bc; i += 128) {
            int r = i / Bc;
            sP[i] = __float2bfloat16(__bfloat162float(sP[i]) * (sdP[i] - sD[r]));
        }
        __syncthreads();
//dQ
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
//D
            if (tid < Br) {
                float d = 0.f;
                for (int j = 0; j < D; j++)
                    d += __bfloat162float(sdO[tid * D + j]) * __bfloat162float(sO[tid * D + j]);
                sD[tid] = d;
            }
            __syncthreads();
//S
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
//dS
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

// ═════════════════════════════════════════════════════════════════════════════
// V3 — Hopper wgmma (warpgroup MMA) tensor-core baseline  [SM_90a only]
//
//   Direct, correctness-first port of V2: identical (Br=Bc=D=64) tiling, grids,
//   two-kernel split, softmax / causal mask / D-correction math, and plain
//   row-major shared-memory staging.  The ONLY change is that every V2
//   `wmma::mma_sync` (16x16x16) becomes a single Hopper warpgroup MMA
//   `wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16`.
//   One block = ONE warpgroup = 128 threads.  bf16 in/out, fp32 accumulate.
//   NOTE: wgmma.* is sm_90a-only, so V3 needs a real Hopper GPU to RUN; on
//   non-Hopper it is compile-only.  (The wgmma PTX coexists fine with the wmma
//   V1/V2 in this file — they are just never reached on non-Hopper at runtime.)
// ═════════════════════════════════════════════════════════════════════════════
using bf16 = __nv_bfloat16;

// ── wgmma shared-memory matrix descriptor (64-bit) ───────────────────────────
// Bit layout (PTX ISA §9.7.14): [13:0]=start addr(enc), [29:16]=leading byte
// offset LBO(enc), [45:32]=stride byte offset SBO(enc), [51:49]=matrix base
// offset (swizzle only), [63:62]=swizzle mode (0=none).  enc(x)=(x&0x3FFFF)>>4.
__device__ __forceinline__ uint64_t desc_encode(uint64_t x) { return (x & 0x3FFFFull) >> 4; }

// ── Core-matrix-tiled shared-memory layout, NO SWIZZLE ───────────────────────
// A 16-bit "core matrix" is 8 rows (strided) x 16 bytes (contiguous) = 8x8 bf16,
// stored as 128 CONTIGUOUS bytes.  wgmma has NO within-core-matrix row-stride
// field, so a plain row-major [R][C] tile is not directly consumable — operands
// are re-tiled so each 8x8 core matrix is contiguous.  Every wgmma operand is a
// [R][C=64] K-major tile (C = contraction dim = contiguous), core matrices in
// row-block-major / col-block-minor order:
//   tiled_off(r,c) = ((r/8)*(C/8) + (c/8))*64 + (r%8)*8 + (c%8)     (C=64 -> C/8=8)
// For one wgmma k-step (K=16 = 2 col-blocks): base = &tile[128*kt], LBO=128B
// (stride between the 2 K core-matrices), SBO=1024B (stride between the 8 M/N
// core-matrices), base advances 128 elements (256B) per step.  (Matches the
// Colfax / accelerated-computing.academy no-swizzle m64n_k16 example.)
__device__ __forceinline__ int tiled_off(int r, int c) {
    return ((r >> 3) * 8 + (c >> 3)) * 64 + (r & 7) * 8 + (c & 7);
}
__device__ __forceinline__ uint64_t make_desc(const bf16 *smem_ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint64_t d = 0;
    d |= desc_encode((uint64_t)addr);          // start address  -> bits [13:0]
    d |= desc_encode((uint64_t)128)  << 16;    // LBO = 128 bytes -> bits [29:16]
    d |= desc_encode((uint64_t)1024) << 32;    // SBO = 1024 bytes-> bits [45:32]
    return d;                                   // swizzle=0, matrix base offset=0
}

// Fill a K-major tiled operand buffer from a plain row-major [64][64] tile.
__device__ __forceinline__ void fill_copy(bf16 *dst, const bf16 *src, int tid) {  // dst_tiled(r,c)=src[r][c]
    for (int i = tid; i < 64 * 64; i += 128) { int r = i >> 6, c = i & 63; dst[tiled_off(r, c)] = src[r * 64 + c]; }
}
__device__ __forceinline__ void fill_trans(bf16 *dst, const bf16 *src, int tid) { // dst_tiled(r,c)=src[c][r]
    for (int i = tid; i < 64 * 64; i += 128) { int r = i >> 6, c = i & 63; dst[tiled_off(r, c)] = src[c * 64 + r]; }
}

// One warpgroup MMA: D(m64,n64) += A(m64,k16)*B(k16,n64).  scaleD=1 (accumulate),
// scaleA=1, scaleB=1, transA=0, transB=0.
//
// TRANS SEMANTICS (the subtle part — got this wrong in the first cut): for bf16
// wgmma the NATIVE ("TN") layout is trans=0 => K-contiguous (contraction dim is
// the contiguous 16-byte core-matrix direction) for BOTH A and B.  trans=1 flips
// an operand to MN-contiguous.  We stage EVERY operand contraction-contiguous
// (via fill_copy / fill_trans), so both flags are 0.  Using transB=1 here made
// wgmma read the K-contiguous B tile as if it were N-contiguous, contracting the
// wrong index (Σ_k Q[i][k]·K[k][n] instead of Q·K^T) -> huge S -> expf overflow
// -> dQ NaN.  32 fp32 accumulators/thread in registers.
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

// One full m64n64 GEMM (K=64 = 4 k-steps of 16), accumulating into `acc`.
// fence -> 4 wgmma (one group) -> commit -> wait, per the async discipline.
__device__ __forceinline__ void run_gemm(float acc[32], const bf16 *sA_t, const bf16 *sB_t) {
    wgmma_fence();                    // order the (non-wgmma) zeroing / prior writes of acc
#pragma unroll
    for (int kt = 0; kt < 4; kt++) {  // base advances 128 elems per k-step
        wgmma_m64n64k16(acc, make_desc(sA_t + 128 * kt), make_desc(sB_t + 128 * kt));
    }
    wgmma_commit();
    wgmma_wait0();
}

// Accumulator-register -> memory mapping (m64n64k16 f32): standard mma.m16n8k16
// D-fragment tiled over 4 warps (rows) x 8 n-subtiles (cols).
//   warp w=tid/32 owns rows [16w,16w+16); r0=lane/4, r1=r0+8, col_base=(lane%4)*2
//   nt(0..7): d[nt*4+{0,1,2,3}] -> (r0,c),(r0,c+1),(r1,c),(r1,c+1), c=nt*8+col_base
__device__ __forceinline__ void store_acc_smem(const float d[32], float *smem, int tid, float scl) {
    int w = tid >> 5, lane = tid & 31;
    int r0 = w * 16 + (lane >> 2), r1 = r0 + 8, cc = (lane & 3) * 2;
#pragma unroll
    for (int nt = 0; nt < 8; nt++) {
        int c = nt * 8 + cc;
        smem[r0 * 64 + c + 0] = d[nt * 4 + 0] * scl;
        smem[r0 * 64 + c + 1] = d[nt * 4 + 1] * scl;
        smem[r1 * 64 + c + 0] = d[nt * 4 + 2] * scl;
        smem[r1 * 64 + c + 1] = d[nt * 4 + 3] * scl;
    }
}
__device__ __forceinline__ void store_acc_global(const float d[32], bf16 *g, long base, int D, int tid, float scl) {
    int w = tid >> 5, lane = tid & 31;
    int r0 = w * 16 + (lane >> 2), r1 = r0 + 8, cc = (lane & 3) * 2;
#pragma unroll
    for (int nt = 0; nt < 8; nt++) {
        int c = nt * 8 + cc;
        g[base + (long)r0 * D + c + 0] = __float2bfloat16(d[nt * 4 + 0] * scl);
        g[base + (long)r0 * D + c + 1] = __float2bfloat16(d[nt * 4 + 1] * scl);
        g[base + (long)r1 * D + c + 0] = __float2bfloat16(d[nt * 4 + 2] * scl);
        g[base + (long)r1 * D + c + 1] = __float2bfloat16(d[nt * 4 + 3] * scl);
    }
}
__device__ __forceinline__ void zero32(float d[32]) {
#pragma unroll
    for (int i = 0; i < 32; i++) d[i] = 0.0f;
}

// ── V3 — Kernel 1 — dQ ──  Grid (B,Hq,S/Br), 128 threads (one warpgroup).
//   S = Q·K^T·scale , dP = dO·V^T , dQ += (dS·K)·scale.  All natural K-major.
template<int Br, int Bc, int D>
__global__ void __launch_bounds__(128, 1)
gqa_backward_dQ_v3(
    const bf16 * __restrict__ d_Q, const bf16 * __restrict__ d_K,
    const bf16 * __restrict__ d_V, const bf16 * __restrict__ d_O,
    const bf16 * __restrict__ d_dO, const float * __restrict__ d_LSE,
          bf16 * __restrict__ d_dQ,
    int B, int Hq, int Hkv, int G, int S, float scale
) {
    static_assert(Br == 64 && Bc == 64 && D == 64, "V3 requires Br=Bc=D=64");

    __shared__ __align__(16)  bf16  sQ [Br * D];
    __shared__ __align__(16)  bf16  sdO[Br * D];
    __shared__ __align__(16)  bf16  sO [Br * D];
    __shared__ __align__(16)  bf16  sK [Bc * D];
    __shared__ __align__(16)  bf16  sV [Bc * D];
    __shared__ __align__(16)  float sS [Br * Bc];
    __shared__ __align__(16)  float sdP[Br * Bc];
    __shared__ __align__(16)  bf16  sP [Br * Bc];   // reused as sdS
    __shared__                float sLSE[Br];
    __shared__                float sD  [Br];
    __shared__ __align__(128) bf16  sA_t[64 * 64];  // wgmma K-major tiled scratch
    __shared__ __align__(128) bf16  sB_t[64 * 64];

    const int tid = threadIdx.x;
    const int b = blockIdx.x, hq = blockIdx.y, q_tile = blockIdx.z;
    const int hkv = hq / G;
    const int q_row0 = q_tile * Br;
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

    if (tid < Br) {
        float acc = 0.f;
        for (int j = 0; j < D; j++)
            acc += __bfloat162float(sdO[tid * D + j]) * __bfloat162float(sO[tid * D + j]);
        sD[tid] = acc;
    }
    __syncthreads();

    float dq[32]; zero32(dq);   // dQ running total (Br x D) — folded via register adds, NOT a wgmma accumulator

    for (int kc = 0; kc < nKTiles; kc++) {
        if (kc * Bc >= q_row0 + Br) break;   // causal: remaining K tiles fully masked

        const long kBase = kvBase + (long)kc * Bc * D;
        for (int i = tid; i < Bc * D; i += 128) { sK[i] = d_K[kBase + i]; sV[i] = d_V[kBase + i]; }
        __syncthreads();

        // S = Q·K^T·scale   (A=Q[i][d], B=K[j][d]; K-major, no transpose)
        fill_copy(sA_t, sQ, tid); fill_copy(sB_t, sK, tid);
        __syncthreads();
        { float acc[32]; zero32(acc); run_gemm(acc, sA_t, sB_t); store_acc_smem(acc, sS, tid, scale); }
        __syncthreads();

        // P = exp(S - LSE) with causal mask
        for (int i = tid; i < Br * Bc; i += 128) {
            int r = i / Bc, c = i % Bc;
            sP[i] = (kc * Bc + c > q_row0 + r) ? __float2bfloat16(0.f)
                                               : __float2bfloat16(expf(sS[i] - sLSE[r]));
        }
        __syncthreads();

        // dP = dO·V^T   (A=dO[i][d], B=V[j][d])
        fill_copy(sA_t, sdO, tid); fill_copy(sB_t, sV, tid);
        __syncthreads();
        { float acc[32]; zero32(acc); run_gemm(acc, sA_t, sB_t); store_acc_smem(acc, sdP, tid, 1.0f); }
        __syncthreads();

#ifdef V3_DEBUG
        // Decisive probe: dump FULL row 0 of S (Q·Kᵀ·scale) and dP (dO·Vᵀ) vs a
        // scalar hand-compute, for block(0,0,0), kc==0. If BOTH match across all 64
        // columns, the straight-GEMM path (descriptor + readout) is fully correct
        // and the bug is downstream (transposed dV/dK GEMMs or accumulation). If
        // they diverge at some column, the descriptor/readout is wrong there.
        if (b == 0 && hq == 0 && q_tile == 0 && kc == 0 && tid == 0) {
            int nbad_s = 0, nbad_dp = 0;
            for (int c = 0; c < 64; c++) {
                float sref = 0.f, dpref = 0.f;
                for (int d = 0; d < D; d++) {
                    sref  += __bfloat162float(sQ [0 * D + d]) * __bfloat162float(sK[c * D + d]);
                    dpref += __bfloat162float(sdO[0 * D + d]) * __bfloat162float(sV[c * D + d]);
                }
                sref *= scale;
                if (fabsf(sS [c] - sref)  > 1e-2f) { if (++nbad_s  <= 4) printf("  S[0][%2d]  got=% .5f ref=% .5f\n", c, sS [c], sref);  }
                if (fabsf(sdP[c] - dpref) > 1e-2f) { if (++nbad_dp <= 4) printf("  dP[0][%2d] got=% .5f ref=% .5f\n", c, sdP[c], dpref); }
            }
            printf("[V3_DEBUG] block(0,0,0) row0: S mismatches=%d/64, dP mismatches=%d/64\n", nbad_s, nbad_dp);
        }
        __syncthreads();
#endif

        // dS = P ⊙ (dP - D)   (into sP)
        for (int i = tid; i < Br * Bc; i += 128) {
            int r = i / Bc;
            sP[i] = __float2bfloat16(__bfloat162float(sP[i]) * (sdP[i] - sD[r]));
        }
        __syncthreads();

        // dQ += dS·K   (A=dS[i][j] copy; B K-major [D][Bc] with j contiguous = K^T -> transpose sK)
        // Compute this tile's dS·K into a FRESH transient accumulator, then fold it
        // into dq with ordinary register adds.  We must NOT hold dq as a wgmma
        // accumulator across the interleaved S/dP wgmma groups of the next tile:
        // that persistent-accumulator-across-groups pattern was non-deterministic
        // / produced +inf on the H200.  reg i maps to the same (row,col) every
        // tile, so dq[i]+=acc[i] accumulates the correct output element.
        fill_copy(sA_t, sP, tid); fill_trans(sB_t, sK, tid);
        __syncthreads();
        { float acc[32]; zero32(acc); run_gemm(acc, sA_t, sB_t);
#ifdef V3_DEBUG
          // Verify the transposed-B GEMM dQ=dS·K directly. block(0,0,0) has only
          // kc==0, so `acc` IS the complete (unscaled) dQ tile. sP holds dS (bf16),
          // sK holds K (bf16); sS is free (S already consumed into P). If this
          // matches, fill_trans + the transposed wgmma are correct and the bug is
          // in accumulation; if not, the transposed path itself is broken.
          if (kc == 0 && b == 0 && hq == 0 && q_tile == 0) {
              store_acc_smem(acc, sS, tid, 1.0f);
              __syncthreads();
              if (tid == 0) {
                  int nbad = 0;
                  for (int d = 0; d < 64; d++) {
                      float ref = 0.f;
                      for (int j = 0; j < 64; j++)
                          ref += __bfloat162float(sP[0 * 64 + j]) * __bfloat162float(sK[j * 64 + d]);
                      if (fabsf(sS[d] - ref) > 1e-2f) { if (++nbad <= 6) printf("  dSK[0][%2d] got=% .5f ref=% .5f\n", d, sS[d], ref); }
                  }
                  printf("[V3_DEBUG] block(0,0,0) dQ=dS.K row0 (transposed-B GEMM): mismatches=%d/64\n", nbad);
              }
              __syncthreads();
          }
#endif
          for (int i = 0; i < 32; i++) dq[i] += acc[i]; }
        __syncthreads();
    }

    store_acc_global(dq, d_dQ, qBase, D, tid, scale);   // row=i, col=d; *scale
}

// ── V3 — Kernel 2 — dK, dV ──  Grid (B,Hkv,S/Bc), 128 threads (one warpgroup).
//   S=Q·K^T·scale , dV += P^T·dO , dP=dO·V^T , dK += (dS^T·Q)·scale.
//   The transposed GEMMs (dV, dK) are handled by transposing the A tile (P/dS)
//   and B tile (dO/Q) during the smem repack so their contraction dim is also
//   contiguous, letting all six GEMMs use the single native config
//   (transA=0, transB=0, K-major both operands) — see the report notes.
template<int Br, int Bc, int D>
__global__ void __launch_bounds__(128, 1)
gqa_backward_dKdV_v3(
    const bf16 * __restrict__ d_Q, const bf16 * __restrict__ d_K,
    const bf16 * __restrict__ d_V, const bf16 * __restrict__ d_O,
    const bf16 * __restrict__ d_dO, const float * __restrict__ d_LSE,
          bf16 * __restrict__ d_dK, bf16 * __restrict__ d_dV,
    int B, int Hq, int Hkv, int G, int S, float scale
) {
    static_assert(Br == 64 && Bc == 64 && D == 64, "V3 requires Br=Bc=D=64");

    __shared__ __align__(16)  bf16  sK [Bc * D];
    __shared__ __align__(16)  bf16  sV [Bc * D];
    __shared__ __align__(16)  bf16  sQ [Br * D];
    __shared__ __align__(16)  bf16  sdO[Br * D];
    __shared__ __align__(16)  bf16  sO [Br * D];
    __shared__ __align__(16)  float sS [Br * Bc];
    __shared__ __align__(16)  float sdP[Br * Bc];
    __shared__ __align__(16)  bf16  sP [Br * Bc];   // reused as sdS
    __shared__                float sLSE[Br];
    __shared__                float sD  [Br];
    __shared__ __align__(128) bf16  sA_t[64 * 64];
    __shared__ __align__(128) bf16  sB_t[64 * 64];

    const int tid = threadIdx.x;
    const int b = blockIdx.x, hkv = blockIdx.y, k_tile = blockIdx.z;
    const int k_row0 = k_tile * Bc;
    const int nQTiles = S / Br;

    const long kvBase = ((long)(b * Hkv + hkv) * S + k_row0) * D;

    for (int i = tid; i < Bc * D; i += 128) { sK[i] = d_K[kvBase + i]; sV[i] = d_V[kvBase + i]; }
    __syncthreads();

    float dv[32]; zero32(dv);   // dV running total (Bc x D) — folded via register adds, NOT a wgmma accumulator
    float dk[32]; zero32(dk);   // dK running total (Bc x D) — folded via register adds, NOT a wgmma accumulator

    for (int g = 0; g < G; g++) {
        const int hq = hkv * G + g;
        for (int qc = k_row0 / Br; qc < nQTiles; qc++) {
            const int  q_row0 = qc * Br;
            const long qBase  = ((long)(b * Hq + hq) * S + q_row0) * D;
            const long lBase  =  (long)(b * Hq + hq) * S + q_row0;

            for (int i = tid; i < Br * D; i += 128) {
                sQ [i] = d_Q [qBase + i]; sdO[i] = d_dO[qBase + i]; sO [i] = d_O [qBase + i];
            }
            if (tid < Br) sLSE[tid] = d_LSE[lBase + tid];
            __syncthreads();

            if (tid < Br) {
                float acc = 0.f;
                for (int j = 0; j < D; j++)
                    acc += __bfloat162float(sdO[tid * D + j]) * __bfloat162float(sO[tid * D + j]);
                sD[tid] = acc;
            }
            __syncthreads();

            // S = Q·K^T·scale
            fill_copy(sA_t, sQ, tid); fill_copy(sB_t, sK, tid);
            __syncthreads();
            { float acc[32]; zero32(acc); run_gemm(acc, sA_t, sB_t); store_acc_smem(acc, sS, tid, scale); }
            __syncthreads();

            // P = exp(S - LSE) with causal mask
            for (int i = tid; i < Br * Bc; i += 128) {
                int r = i / Bc, c = i % Bc;
                sP[i] = (k_row0 + c > q_row0 + r) ? __float2bfloat16(0.f)
                                                  : __float2bfloat16(expf(sS[i] - sLSE[r]));
            }
            __syncthreads();

            // dV += P^T·dO   (A tiled[j][i]=P[i][j] -> transpose sP; B [D][Br] i-contig = dO^T -> transpose sdO)
            // Fresh transient accumulator + ordinary register add (see dQ note):
            // never hold dv as a wgmma accumulator across the other GEMM groups.
            fill_trans(sA_t, sP, tid); fill_trans(sB_t, sdO, tid);
            __syncthreads();
            { float acc[32]; zero32(acc); run_gemm(acc, sA_t, sB_t);
#ifdef V3_DEBUG
              // Verify dV = P^T·dO (the trans-A, trans-B GEMM) on the FIRST iteration
              // of block(0,0,0). acc is this iteration's contribution. Output row j:
              // dV[j][d] = sum_i P[i][j]·dO[i][d].  sP holds P (bf16), sdO holds dO.
              if (b == 0 && hkv == 0 && k_tile == 0 && g == 0 && qc == k_row0 / Br) {
                  store_acc_smem(acc, sS, tid, 1.0f);
                  __syncthreads();
                  if (tid == 0) {
                      int nbad = 0;
                      for (int d = 0; d < 64; d++) {
                          float ref = 0.f;
                          for (int i = 0; i < 64; i++)
                              ref += __bfloat162float(sP[i * 64 + 0]) * __bfloat162float(sdO[i * 64 + d]);
                          if (fabsf(sS[d] - ref) > 1e-2f) { if (++nbad <= 6) printf("  dV[0][%2d] got=% .5f ref=% .5f\n", d, sS[d], ref); }
                      }
                      printf("[V3_DEBUG] block(0,0,0) dV=P^T.dO row0 (trans-A GEMM): mismatches=%d/64\n", nbad);
                  }
                  __syncthreads();
              }
#endif
              for (int i = 0; i < 32; i++) dv[i] += acc[i]; }
            __syncthreads();

            // dP = dO·V^T
            fill_copy(sA_t, sdO, tid); fill_copy(sB_t, sV, tid);
            __syncthreads();
            { float acc[32]; zero32(acc); run_gemm(acc, sA_t, sB_t); store_acc_smem(acc, sdP, tid, 1.0f); }
            __syncthreads();

            // dS = P ⊙ (dP - D)   (into sP)
            for (int i = tid; i < Br * Bc; i += 128) {
                int r = i / Bc;
                sP[i] = __float2bfloat16(__bfloat162float(sP[i]) * (sdP[i] - sD[r]));
            }
            __syncthreads();

            // dK += dS^T·Q   (A tiled[j][i]=dS[i][j] -> transpose sP; B [D][Br] i-contig = Q^T -> transpose sQ)
            // Fresh transient accumulator + ordinary register add (see dQ note).
            fill_trans(sA_t, sP, tid); fill_trans(sB_t, sQ, tid);
            __syncthreads();
            { float acc[32]; zero32(acc); run_gemm(acc, sA_t, sB_t);
              for (int i = 0; i < 32; i++) dk[i] += acc[i]; }
            __syncthreads();
        }
    }

    store_acc_global(dv, d_dV, kvBase, D, tid, 1.0f);    // dV : no scale
    store_acc_global(dk, d_dK, kvBase, D, tid, scale);   // dK : *scale
}

// ── V3 launcher — same signature as launch_gqa_backward_v2 ──
template<int Br, int Bc, int D>
void launch_gqa_backward_v3(
    const bf16 *d_Q, const bf16 *d_K, const bf16 *d_V, const bf16 *d_O,
    const bf16 *d_dO, const float *d_LSE,
    bf16 *d_dQ, bf16 *d_dK, bf16 *d_dV,
    int B, int Hq, int Hkv, int G, int S, float scale
) {
    static_assert(Br == 64 && Bc == 64 && D == 64, "V3 requires Br=Bc=D=64");
    constexpr dim3 BLOCK(128);
    dim3 GRID1(B, Hq,  S / Br);
    dim3 GRID2(B, Hkv, S / Bc);
    gqa_backward_dQ_v3  <Br,Bc,D><<<GRID1, BLOCK>>>(
        d_Q, d_K, d_V, d_O, d_dO, d_LSE, d_dQ, B, Hq, Hkv, G, S, scale);
    gqa_backward_dKdV_v3<Br,Bc,D><<<GRID2, BLOCK>>>(
        d_Q, d_K, d_V, d_O, d_dO, d_LSE, d_dK, d_dV, B, Hq, Hkv, G, S, scale);
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main(){
    std::cout << "GQA Backward — precision test  [Hopper SM_90 / H200]\n";
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

    launch_gqa_backward_v3<Br2,Bc2,D>(d_Q,d_K,d_V,d_O,d_dO,d_LSE,d_dQ,d_dK,d_dV,B,Hq,Hkv,G,S,scale);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    check("── V3  Br=64, Bc=64  wgmma ──", Nq, Nkv, d_dQ, d_dK, d_dV);

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
    {
        KernelStats s = benchmarkKernel(
            [&](){ launch_gqa_backward_v3<Br2,Bc2,D>(
                d_Q,d_K,d_V,d_O,d_dO,d_LSE,d_dQ,d_dK,d_dV,B,Hq,Hkv,G,S,scale); },
            100, 10, bwd_flops);
        displayStats("GQA bwd V3  Br=64, Bc=64  wgmma  (Hopper SM_90)", s);
    }

    CUDA_CHECK(cudaFree(d_Q));  CUDA_CHECK(cudaFree(d_K));  CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));  CUDA_CHECK(cudaFree(d_dO)); CUDA_CHECK(cudaFree(d_LSE));
    CUDA_CHECK(cudaFree(d_dQ)); CUDA_CHECK(cudaFree(d_dK)); CUDA_CHECK(cudaFree(d_dV));
    return 0;
}
