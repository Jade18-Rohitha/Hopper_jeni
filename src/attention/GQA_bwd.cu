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
    static_assert(Br == 64 && Bc == 64 && D % 16 == 0, "V2 requires Br=Bc=64, D%16==0");

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
    static_assert(Br == 64 && Bc == 64 && D % 16 == 0, "V2 requires Br=Bc=64, D%16==0");

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
    static_assert(Br == 64 && Bc == 64 && D % 16 == 0, "V2 requires Br=Bc=64, D%16==0");

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
// are re-tiled so each 8x8 core matrix is contiguous.  Every wgmma operand is an
// [MN][K] K-major tile (MN = M for A / N for B; K = contraction dim = contiguous),
// core matrices in MN-block-major / K-block-minor order:
//   tiled_off(mn,k,K) = ((mn/8)*(K/8) + (k/8))*64 + (mn%8)*8 + (k%8)
// For one wgmma k-step (K=16 = 2 col-blocks): base = &tile[128*kt], LBO=128B
// (stride between the 2 K core-matrices), base advances 128 elements (256B)/step.
// SBO (stride between adjacent MN core-matrices) = (K/8)*128 B and DEPENDS on the
// buffer's full contraction width K — 1024B for K=64 tiles, 2048B for K=128 (the
// D-contraction S/dP GEMMs).  Passed into make_desc explicitly.  (Matches the
// Colfax / accelerated-computing.academy no-swizzle m64n_k16 example, generalized.)
__device__ __forceinline__ int tiled_off(int mn, int k, int K) {
    return ((mn >> 3) * (K >> 3) + (k >> 3)) * 64 + (mn & 7) * 8 + (k & 7);
}
__device__ __forceinline__ uint64_t make_desc(const bf16 *smem_ptr, uint64_t sbo) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint64_t d = 0;
    d |= desc_encode((uint64_t)addr);          // start address  -> bits [13:0]
    d |= desc_encode((uint64_t)128) << 16;     // LBO = 128 bytes -> bits [29:16]
    d |= desc_encode(sbo)           << 32;     // SBO = (K/8)*128 -> bits [45:32]
    return d;                                   // swizzle=0, matrix base offset=0
}

// Fill a K-major tiled operand buffer (MN rows x K contraction).
//   fill_copy : src is [MN][K] row-major (stride K)  -> dst_tiled(mn,k) = src[mn][k]
//   fill_trans: src is [K][MN] row-major (stride MN)  -> dst_tiled(mn,k) = src[k][mn]
// MN and K are compile-time (powers of two) so /K, %K and tiled_off fold to shifts.
template<int MN, int K>
__device__ __forceinline__ void fill_copy(bf16 *dst, const bf16 *src, int tid) {
    for (int i = tid; i < MN * K; i += 128) { int mn = i / K, k = i % K; dst[tiled_off(mn, k, K)] = src[mn * K + k]; }
}
template<int MN, int K>
__device__ __forceinline__ void fill_trans(bf16 *dst, const bf16 *src, int tid) {
    for (int i = tid; i < MN * K; i += 128) { int mn = i / K, k = i % K; dst[tiled_off(mn, k, K)] = src[k * MN + mn]; }
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

// Accumulator-operand fence (CUTLASS `warpgroup_fence_operand`, mma_sm90_gmma.hpp).
// wgmma.mma_async writes its accumulator registers ASYNCHRONOUSLY; they only become
// valid at wgmma.wait_group.  `wgmma.fence` (a "memory" clobber) orders MEMORY but
// NOT these register accesses, so without a per-register compiler barrier ptxas is
// free to hoist the post-GEMM reads of `acc` (store_acc_smem / dq[i]+=acc[i]) ahead
// of the async writes retiring — reading garbage.  That is exactly the multi-tile
// hazard here: it was masked in the single-tile probe (the probe's extra
// store+sync+printf perturbs scheduling) but fired in the `for kc` / `for g,qc`
// loops → non-deterministic dV/dK and, via garbage S → expf overflow, dQ = +inf.
// Bracket the accumulator with this barrier BEFORE wgmma.fence and AFTER wait_group,
// exactly as CUTLASS does.  It is a no-op asm (0 extra registers / 0 spills).
template<int N>
__device__ __forceinline__ void fence_operandN(float *d) {
#pragma unroll
    for (int i = 0; i < N; i++) asm volatile("" : "+f"(d[i]) :: "memory");
}

// Cross-proxy shared-memory fence.  fill_copy/fill_trans/fill_trans_A write the
// wgmma operand buffers sA_t/sB_t with ORDINARY stores (the GENERIC proxy).
// wgmma.mma_async READS those operands through the ASYNC proxy.  __syncthreads
// orders the generic proxy across threads, but it does NOT bridge to the async
// proxy — so at full speed the async operand read can observe STALE smem (the
// result is correct ONLY when the sanitizer/serialization happens to let the
// generic writes drain first).  `fence.proxy.async.shared::cta` establishes the
// generic→async ordering for CTA-scoped shared memory; it is the SAME fence
// CUTLASS issues before a TMA store that reads generically-written smem
// (cute::tma_store_fence / cutlass::arch::fence_view_async_shared).  It must be
// executed by the whole warpgroup AFTER the fills' __syncthreads and BEFORE the
// wgmma — hence it is the first thing every run_gemm_* does.
__device__ __forceinline__ void fence_proxy_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

// One full m64n64 GEMM over NK k-steps of 16 (K = 16*NK), accumulating into `acc`.
// fence -> NK wgmma (one group) -> commit -> wait, per the async discipline.
// Used for the D-contraction GEMMs S=Q·Kᵀ and dP=dO·Vᵀ: K = D = 128 -> NK = 8,
// SBO = (D/8)*128 = 2048.  base advances 128 elems / k-step for both operands.
template<int NK>
__device__ __forceinline__ void run_gemm_n64(float acc[32], const bf16 *sA_t, const bf16 *sB_t, uint64_t sbo) {
    fence_proxy_async_shared();       // generic (fill) writes → async (wgmma) operand reads
    fence_operandN<32>(acc);          // bracket the async region (see fence_operandN note)
    wgmma_fence();                    // order the (non-wgmma) zeroing / prior writes of acc
#pragma unroll
    for (int kt = 0; kt < NK; kt++) {
        wgmma_m64n64k16(acc, make_desc(sA_t + 128 * kt, sbo), make_desc(sB_t + 128 * kt, sbo));
    }
    wgmma_commit();
    wgmma_wait0();
    fence_operandN<32>(acc);          // block hoisting acc reads before wait_group retires
}

// ── wgmma m64n128k16 (bf16) — 64 fp32 accumulators/thread ────────────────────
// The D-wide OUTPUT GEMMs (dQ=dS·K, dV=Pᵀ·dO, dK=dSᵀ·Q) produce an N = head-dim
// = 128 result, so a single m64n128 wgmma covers the whole output row in one
// instruction (cleaner than 2×m64n64 and identical register cost: 64 regs).
// The accumulator layout is the m64n64 fragment extended to N/8 = 16 n-subtiles
// (see store_acc_* below) — trans flags never touch the D-fragment layout.
__device__ __forceinline__ void wgmma_m64n128k16(float d[64], uint64_t descA, uint64_t descB) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
        "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
        "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, "
        "%64, %65, 1, 1, 1, 0, 0;\n"
        : "+f"(d[0]),  "+f"(d[1]),  "+f"(d[2]),  "+f"(d[3]),  "+f"(d[4]),  "+f"(d[5]),  "+f"(d[6]),  "+f"(d[7]),
          "+f"(d[8]),  "+f"(d[9]),  "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
          "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
          "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
          "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
          "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
        : "l"(descA), "l"(descB));
}
// Transposed-A variant: trans-a = 1 (Major::MN A), trans-b = 0 (K-major B).
__device__ __forceinline__ void wgmma_m64n128k16_tA(float d[64], uint64_t descA, uint64_t descB) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
        "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
        "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, "
        "%64, %65, 1, 1, 1, 1, 0;\n"   // scaleD, scaleA, scaleB, transA=1, transB=0
        : "+f"(d[0]),  "+f"(d[1]),  "+f"(d[2]),  "+f"(d[3]),  "+f"(d[4]),  "+f"(d[5]),  "+f"(d[6]),  "+f"(d[7]),
          "+f"(d[8]),  "+f"(d[9]),  "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
          "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
          "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
          "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
          "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
        : "l"(descA), "l"(descB));
}
// One full m64n128 GEMM over NK k-steps (K-major A, K-major B).  For dQ=dS·K:
// contraction K = Bc = 64 -> NK = 4, SBO = (Bc/8)*128 = 1024.
template<int NK>
__device__ __forceinline__ void run_gemm_n128(float acc[64], const bf16 *sA_t, const bf16 *sB_t, uint64_t sbo) {
    fence_proxy_async_shared();
    fence_operandN<64>(acc);
    wgmma_fence();
#pragma unroll
    for (int kt = 0; kt < NK; kt++) {
        wgmma_m64n128k16(acc, make_desc(sA_t + 128 * kt, sbo), make_desc(sB_t + 128 * kt, sbo));
    }
    wgmma_commit();
    wgmma_wait0();
    fence_operandN<64>(acc);
}

// ── Transposed-A GEMM (dV = Pᵀ·dO, dK = dSᵀ·Q), m64n128 ──────────────────────
// Contraction is over the ROW dim `i` of BOTH operands, so operand A is supplied
// transposed.  ISA rule (PTX ISA §9.7.14 / CUTLASS make_gmma_desc): a K-major
// operand (trans=0) has intra-core order (mn_local*8 + k_local); to transpose you
// present Major::MN (trans=1), whose core matrix is the TRANSPOSE — contiguous dim
// = MN, intra-core order (k_local*8 + mn_local).  The descriptor byte offsets are
// assigned identically for Major::K and Major::MN in the no-swizzle case, so
// `make_desc` is reused verbatim.  Only two things change vs the K-major path:
//   1. the intra-core element order is transposed  →  `mn_off` below;
//   2. the wgmma `trans-a` immediate flips 0 → 1   →  `wgmma_m64n128k16_tA`.
// Operand B is left K-major (trans-b=0, `fill_trans`).  For dV/dK: contraction
// K = Br = 64 -> NK = 4, SBO = 1024.
// mn_off(mn,k,K): mn = M (output row), k = K (contraction).  SAME block arrangement
// as tiled_off (base still advances 128 elems / k-step) — only the intra-core term
// (k&7)*8+(mn&7) is transposed.
__device__ __forceinline__ int mn_off(int mn, int k, int K) {
    return ((mn >> 3) * (K >> 3) + (k >> 3)) * 64 + (k & 7) * 8 + (mn & 7);
}
// Stage operand A = srcᵀ (A[mn][k] = src[k][mn]) into the Major::MN core layout.
// src is [K][MN] row-major (stride MN).
template<int MN, int K>
__device__ __forceinline__ void fill_trans_A(bf16 *dst, const bf16 *src, int tid) {
    for (int i = tid; i < MN * K; i += 128) { int mn = i / K, k = i % K; dst[mn_off(mn, k, K)] = src[k * MN + mn]; }
}
template<int NK>
__device__ __forceinline__ void run_gemm_n128_tA(float acc[64], const bf16 *sA_t, const bf16 *sB_t, uint64_t sbo) {
    fence_proxy_async_shared();
    fence_operandN<64>(acc);
    wgmma_fence();
#pragma unroll
    for (int kt = 0; kt < NK; kt++) {
        wgmma_m64n128k16_tA(acc, make_desc(sA_t + 128 * kt, sbo), make_desc(sB_t + 128 * kt, sbo));
    }
    wgmma_commit();
    wgmma_wait0();
    fence_operandN<64>(acc);
}

// Accumulator-register -> memory mapping (m64nNk16 f32): standard mma.m16n8k16
// D-fragment tiled over 4 warps (rows) x N/8 n-subtiles (cols).  NCOL = N.
//   warp w=tid/32 owns rows [16w,16w+16); r0=lane/4, r1=r0+8, col_base=(lane%4)*2
//   nt(0..N/8): d[nt*4+{0,1,2,3}] -> (r0,c),(r0,c+1),(r1,c),(r1,c+1), c=nt*8+col_base
// NCOL=64  (32-reg S/dP tiles); NCOL=128 (64-reg dQ/dV/dK output tiles).
template<int NCOL>
__device__ __forceinline__ void store_acc_smem(const float *d, float *smem, int tid, float scl) {
    int w = tid >> 5, lane = tid & 31;
    int r0 = w * 16 + (lane >> 2), r1 = r0 + 8, cc = (lane & 3) * 2;
#pragma unroll
    for (int nt = 0; nt < NCOL / 8; nt++) {
        int c = nt * 8 + cc;
        smem[r0 * NCOL + c + 0] = d[nt * 4 + 0] * scl;
        smem[r0 * NCOL + c + 1] = d[nt * 4 + 1] * scl;
        smem[r1 * NCOL + c + 0] = d[nt * 4 + 2] * scl;
        smem[r1 * NCOL + c + 1] = d[nt * 4 + 3] * scl;
    }
}
// NCOL = output width = D; global row stride is also D (full [64][D] tile).
template<int NCOL>
__device__ __forceinline__ void store_acc_global(const float *d, bf16 *g, long base, int D, int tid, float scl) {
    int w = tid >> 5, lane = tid & 31;
    int r0 = w * 16 + (lane >> 2), r1 = r0 + 8, cc = (lane & 3) * 2;
#pragma unroll
    for (int nt = 0; nt < NCOL / 8; nt++) {
        int c = nt * 8 + cc;
        g[base + (long)r0 * D + c + 0] = __float2bfloat16(d[nt * 4 + 0] * scl);
        g[base + (long)r0 * D + c + 1] = __float2bfloat16(d[nt * 4 + 1] * scl);
        g[base + (long)r1 * D + c + 0] = __float2bfloat16(d[nt * 4 + 2] * scl);
        g[base + (long)r1 * D + c + 1] = __float2bfloat16(d[nt * 4 + 3] * scl);
    }
}
template<int N>
__device__ __forceinline__ void zeroN(float *d) {
#pragma unroll
    for (int i = 0; i < N; i++) d[i] = 0.0f;
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
    static_assert(Br == 64 && Bc == 64 && D == 128, "V3 requires Br=Bc=64, D=128");

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
    __shared__ __align__(128) bf16  sA_t[64 * 128]; // wgmma tiled scratch (max operand 64x128)
    __shared__ __align__(128) bf16  sB_t[64 * 128];

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

    float dq[64]; zeroN<64>(dq);   // PERSISTENT dQ[Br x D=128] wgmma accumulator (m64n128 → 64 regs)

    for (int kc = 0; kc < nKTiles; kc++) {
        if (kc * Bc >= q_row0 + Br) break;   // causal: remaining K tiles fully masked

        const long kBase = kvBase + (long)kc * Bc * D;
        for (int i = tid; i < Bc * D; i += 128) { sK[i] = d_K[kBase + i]; sV[i] = d_V[kBase + i]; }
        __syncthreads();

        // S = Q·K^T·scale   (A=Q[i][d], B=K[j][d]; K-major over D=128 → D/16=8 k-steps, SBO=2048)
        fill_copy<Br, D>(sA_t, sQ, tid); fill_copy<Bc, D>(sB_t, sK, tid);
        __syncthreads();
        { float acc[32]; zeroN<32>(acc); run_gemm_n64<D / 16>(acc, sA_t, sB_t, (uint64_t)(D >> 3) * 128); store_acc_smem<Bc>(acc, sS, tid, scale); }
        __syncthreads();

        // P = exp(S - LSE) with causal mask
        for (int i = tid; i < Br * Bc; i += 128) {
            int r = i / Bc, c = i % Bc;
            sP[i] = (kc * Bc + c > q_row0 + r) ? __float2bfloat16(0.f)
                                               : __float2bfloat16(expf(sS[i] - sLSE[r]));
        }
        __syncthreads();

        // dP = dO·V^T   (A=dO[i][d], B=V[j][d]; K-major over D → 8 k-steps, SBO=2048)
        fill_copy<Br, D>(sA_t, sdO, tid); fill_copy<Bc, D>(sB_t, sV, tid);
        __syncthreads();
        { float acc[32]; zeroN<32>(acc); run_gemm_n64<D / 16>(acc, sA_t, sB_t, (uint64_t)(D >> 3) * 128); store_acc_smem<Bc>(acc, sdP, tid, 1.0f); }
        __syncthreads();

        // dS = P ⊙ (dP - D)   (into sP)
        for (int i = tid; i < Br * Bc; i += 128) {
            int r = i / Bc;
            sP[i] = __float2bfloat16(__bfloat162float(sP[i]) * (sdP[i] - sD[r]));
        }
        __syncthreads();

        // dQ += dS·K   (single m64n128: N = D = 128 in one wgmma; contraction j = Bc = 64
        // → Bc/16 = 4 k-steps, SBO=1024.  A = dS copy [i][j] (K-major, K=j=Bc); B = K staged
        // [d][j] via fill_trans (N=D rows, K=Bc contiguous → j-major = Kᵀ).
        // PERSISTENT wgmma accumulator: dq accumulated IN-HARDWARE (scaleD=1) across the kc
        // loop and NEVER read mid-loop — stored once after.  dq zeroed once before the loop,
        // so the first tile's scaleD=1 gives dq = dS·K + 0.  Holding dq live across the
        // interleaved S/dP groups is safe: each GEMM commits+waits its own group (dq never
        // in-flight across them) and run_gemm_n128 brackets dq with fence_operandN.
        fill_copy<Br, Bc>(sA_t, sP, tid); fill_trans<D, Bc>(sB_t, sK, tid);
        __syncthreads();
        run_gemm_n128<Bc / 16>(dq, sA_t, sB_t, (uint64_t)(Bc >> 3) * 128);   // dq += dS·K
        __syncthreads();
    }

    fence_operandN<64>(dq);                                 // ensure final read follows all wgmma
    store_acc_global<D>(dq, d_dQ, qBase, D, tid, scale);    // row=i, col=d; *scale
}

// ── V3 — Kernel 2 — dK, dV ──  Grid (B,Hkv,S/Bc), 128 threads (one warpgroup).
//   S=Q·K^T·scale , dV += P^T·dO , dP=dO·V^T , dK += (dS^T·Q)·scale.
//   S/dP contract over the head dim (K-major both operands, trans-a=0/trans-b=0).
//   dV/dK contract over the row dim i, so their A operand (P/dS) is transposed via
//   Major::MN staging (mn_off) + trans-a=1, while B (dO/Q) stays K-major + trans-b=0
//   (see fill_trans_A / run_gemm_n128_tA).  dV, dK, dQ are PERSISTENT wgmma accumulators
//   accumulated in-hardware (scaleD=1) across the loop and read only once at the end.
template<int Br, int Bc, int D>
__global__ void __launch_bounds__(128, 1)
gqa_backward_dKdV_v3(
    const bf16 * __restrict__ d_Q, const bf16 * __restrict__ d_K,
    const bf16 * __restrict__ d_V, const bf16 * __restrict__ d_O,
    const bf16 * __restrict__ d_dO, const float * __restrict__ d_LSE,
          bf16 * __restrict__ d_dK, bf16 * __restrict__ d_dV,
    int B, int Hq, int Hkv, int G, int S, float scale
) {
    static_assert(Br == 64 && Bc == 64 && D == 128, "V3 requires Br=Bc=64, D=128");

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
    __shared__ __align__(128) bf16  sA_t[64 * 128]; // wgmma tiled scratch (max operand 64x128)
    __shared__ __align__(128) bf16  sB_t[64 * 128];

    const int tid = threadIdx.x;
    const int b = blockIdx.x, hkv = blockIdx.y, k_tile = blockIdx.z;
    const int k_row0 = k_tile * Bc;
    const int nQTiles = S / Br;

    const long kvBase = ((long)(b * Hkv + hkv) * S + k_row0) * D;

    for (int i = tid; i < Bc * D; i += 128) { sK[i] = d_K[kvBase + i]; sV[i] = d_V[kvBase + i]; }
    __syncthreads();

    float dv[64]; zeroN<64>(dv);   // PERSISTENT dV[Bc x D=128] wgmma accumulator (m64n128 → 64 regs)
    float dk[64]; zeroN<64>(dk);   // PERSISTENT dK[Bc x D=128] wgmma accumulator (m64n128 → 64 regs)

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

            // S = Q·K^T·scale   (K-major over D=128 → 8 k-steps, SBO=2048)
            fill_copy<Br, D>(sA_t, sQ, tid); fill_copy<Bc, D>(sB_t, sK, tid);
            __syncthreads();
            { float acc[32]; zeroN<32>(acc); run_gemm_n64<D / 16>(acc, sA_t, sB_t, (uint64_t)(D >> 3) * 128); store_acc_smem<Bc>(acc, sS, tid, scale); }
            __syncthreads();

            // P = exp(S - LSE) with causal mask
            for (int i = tid; i < Br * Bc; i += 128) {
                int r = i / Bc, c = i % Bc;
                sP[i] = (k_row0 + c > q_row0 + r) ? __float2bfloat16(0.f)
                                                  : __float2bfloat16(expf(sS[i] - sLSE[r]));
            }
            __syncthreads();

            // dV += P^T·dO   (single m64n128: N = D = 128.  A = Pᵀ, contraction over row
            // i = Br → transposed A operand via Major::MN staging (fill_trans_A) + trans-a=1;
            // B = dO staged [d][i] via fill_trans (K-major, trans-b=0).  Br/16 = 4 k-steps,
            // SBO=1024.  PERSISTENT wgmma accumulator dv (scaleD=1), accumulated in-hardware
            // across the G×qc loop, never read mid-loop — see the dQ persistent-acc note.
            fill_trans_A<Bc, Br>(sA_t, sP, tid); fill_trans<D, Br>(sB_t, sdO, tid);
            __syncthreads();
            run_gemm_n128_tA<Br / 16>(dv, sA_t, sB_t, (uint64_t)(Br >> 3) * 128);   // dv += Pᵀ·dO
            __syncthreads();

            // dP = dO·V^T   (K-major over D → 8 k-steps, SBO=2048)
            fill_copy<Br, D>(sA_t, sdO, tid); fill_copy<Bc, D>(sB_t, sV, tid);
            __syncthreads();
            { float acc[32]; zeroN<32>(acc); run_gemm_n64<D / 16>(acc, sA_t, sB_t, (uint64_t)(D >> 3) * 128); store_acc_smem<Bc>(acc, sdP, tid, 1.0f); }
            __syncthreads();

            // dS = P ⊙ (dP - D)   (into sP)
            for (int i = tid; i < Br * Bc; i += 128) {
                int r = i / Bc;
                sP[i] = __float2bfloat16(__bfloat162float(sP[i]) * (sdP[i] - sD[r]));
            }
            __syncthreads();

            // dK += dS^T·Q   (single m64n128: N = D = 128.  A = dSᵀ, contraction over row
            // i = Br → transposed A operand (fill_trans_A) + trans-a=1; B = Q staged [d][i]
            // via fill_trans, trans-b=0.  4 k-steps, SBO=1024.  PERSISTENT accumulator dk.)
            fill_trans_A<Bc, Br>(sA_t, sP, tid); fill_trans<D, Br>(sB_t, sQ, tid);
            __syncthreads();
            run_gemm_n128_tA<Br / 16>(dk, sA_t, sB_t, (uint64_t)(Br >> 3) * 128);   // dk += dSᵀ·Q
            __syncthreads();
        }
    }

    fence_operandN<64>(dv); fence_operandN<64>(dk);         // ensure final reads follow all wgmma
    store_acc_global<D>(dv, d_dV, kvBase, D, tid, 1.0f);    // dV : no scale
    store_acc_global<D>(dk, d_dK, kvBase, D, tid, scale);   // dK : *scale
}

// ── V3 launcher — same signature as launch_gqa_backward_v2 ──
template<int Br, int Bc, int D>
void launch_gqa_backward_v3(
    const bf16 *d_Q, const bf16 *d_K, const bf16 *d_V, const bf16 *d_O,
    const bf16 *d_dO, const float *d_LSE,
    bf16 *d_dQ, bf16 *d_dK, bf16 *d_dV,
    int B, int Hq, int Hkv, int G, int S, float scale
) {
    static_assert(Br == 64 && Bc == 64 && D == 128, "V3 requires Br=Bc=64, D=128");
    constexpr dim3 BLOCK(128);
    dim3 GRID1(B, Hq,  S / Br);
    dim3 GRID2(B, Hkv, S / Bc);
    gqa_backward_dQ_v3  <Br,Bc,D><<<GRID1, BLOCK>>>(
        d_Q, d_K, d_V, d_O, d_dO, d_LSE, d_dQ, B, Hq, Hkv, G, S, scale);
    gqa_backward_dKdV_v3<Br,Bc,D><<<GRID2, BLOCK>>>(
        d_Q, d_K, d_V, d_O, d_dO, d_LSE, d_dK, d_dV, B, Hq, Hkv, G, S, scale);
}

// ═════════════════════════════════════════════════════════════════════════════
// V4 — TMA (cp.async.bulk.tensor) 128B-swizzled loads + double-buffered wgmma
//                                                            [SM_90a only]
//
// Same math / tiling / grids / two-kernel split / correctness mechanisms as V3.
// Two changes vs V3:
//   (1) The operands read in their NATURAL K-major-over-D orientation (S=Q·Kᵀ and
//       dP=dO·Vᵀ) are TMA-loaded into a 128B-SWIZZLED smem layout and fed to wgmma
//       DIRECTLY via a Major::K SW128 descriptor — NO per-GEMM fill_copy repack.
//       (dQ kernel: Q,V swizzled;  dKdV kernel: K,V swizzled.)
//   (2) The per-K-tile loads are DOUBLE-BUFFERED with mbarriers so tile k+1's TMA
//       overlaps tile k's wgmma compute (single warpgroup, no warp-spec — that's V5).
//
// WHAT STILL REPACKS AND WHY (unchanged from V3, verified path):
//   • Any operand read TRANSPOSED (contraction over the token dim Br/Bc, not D):
//       dQ+=dS·K   B = Kᵀ  (fill_trans of plain K)
//       dV+=Pᵀ·dO  A = Pᵀ  (fill_trans_A, intermediate), B = dOᵀ (fill_trans plain dO)
//       dK+=dSᵀ·Q  A = dSᵀ (fill_trans_A, intermediate), B = Qᵀ (fill_trans plain Q)
//     The 128B-swizzled buffer is D-contiguous; a transposed read would need a
//     Major::MN SW128 descriptor whose geometry is UNVERIFIED here, so we keep the
//     verified no-swizzle fill_trans/fill_trans_A path (source: plain-TMA copy).
//   • S / dP intermediates (dS, P) are kernel-computed, never loaded — no TMA help.
//   • O is only used elementwise (D[r]=rowsum(dO·O)); loaded PLAIN (swizzle would
//     make the element-wise read opaque). dO is also plain (needed both by dP's
//     fill_copy-A and by the D computation) — so only V (dP-B) is swizzled in dQ,
//     and only K/V are swizzled in dKdV. This deliberately keeps EVERY wgmma read
//     on either the verified no-swizzle path OR the Major::K SW128 path (the direct
//     analog of the HW-confirmed single-atom SW128 case), avoiding Major::MN swz.
//
// D=128 ⇒ each row is 256 B = TWO 128-byte swizzle atoms. One SWIZZLE_128B TMA
// atom box is 64 bf16 wide (128 B). So each swizzled operand is loaded as TWO
// (rows×64) atom sub-tiles (col origins 0 and 64) stacked contiguously in smem;
// the S/dP k-loop walks atom = k/4, lk = k%4 (see run_gemm_n64_sw). Each atom is a
// clean single-atom SW128 tile == the B300-confirmed case, done twice.
// ═════════════════════════════════════════════════════════════════════════════

// ── mbarrier + TMA device helpers (adapted from Blackwell GQA_sm103_bwd.cu) ────
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
// 2-D TMA bulk async: global tile → swizzled/plain smem, tracked by `mbar`.
__device__ __forceinline__ void tma_load_2d_v4(
    const void* tma_desc, void* smem_dst, uint64_t* mbar, uint32_t cx, uint32_t cy) {
    uint32_t dst = (uint32_t)__cvta_generic_to_shared(smem_dst);
    uint32_t mb  = (uint32_t)__cvta_generic_to_shared(mbar);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
        :: "r"(dst), "l"((uint64_t)tma_desc), "r"(cx), "r"(cy), "r"(mb) : "memory");
}

// ── wgmma SS descriptor for a 128B-SWIZZLED, Major::K (K-contiguous) bf16 operand.
// Authoritative field values from CUTLASS cute::GmmaDescriptor + make_gmma_desc
// (pytorch/third_party/cutlass/.../mma_sm90_desc.hpp, mma_traits_sm90_gmma.hpp):
//   start_address_ [13:0]  = smem_addr >> 4
//   leading_byte_offset_ [29:16] : UNUSED under swizzle ("assumed to be 1"); set 1 to match CUTLASS
//   stride_byte_offset_  [45:32] = SBO = stride between 8-row MN-blocks = one 8×64-bf16
//                                  swizzle sub-unit = 8*128 = 1024 B → encoded 1024>>4 = 64
//   base_offset_ [51:49] = 0  (atom buffers are 1024-B-aligned; sub-atom k-advance is in the addr)
//   layout_type_ [63:62] = 1  (SM90::GMMA::LayoutType::B128 — 128B swizzle; NOTE this is the
//                              wgmma 2-bit field, NOT the tcgen05 3-bit "2<<61" field)
// The per-k16 address advance (+lk*32 B within an atom) matches the B300-confirmed
// make_smem_desc_sw128(ptr + k*32) pattern; here lk = k%4 within each 64-wide atom.
__device__ __forceinline__ uint64_t make_desc_sw128_K(const bf16* smem_ptr) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint64_t d = 0;
    d |= (uint64_t)((addr >> 4) & 0x3FFFu);   // start_address       [13:0]
    d |= (uint64_t)1u        << 16;           // leading_byte_offset [29:16] (unused; =1 per CUTLASS)
    d |= (uint64_t)(1024>>4) << 32;           // stride_byte_offset  [45:32] = 1024 B (SBO)
    d |= (uint64_t)1u        << 62;           // layout_type = B128   [63:62]
    return d;
}

// S=Q·Kᵀ / dP=dO·Vᵀ over D=128: one operand is 128B-swizzled (2 atoms, addr math),
// the other is a no-swizzle tiled_off K-major operand (K=D=128 → SBO=2048, base +128
// elems/k-step, exactly V3's run_gemm_n64 B path).  Mixing one swizzled + one
// no-swizzle operand in a single wgmma is legal (each descriptor is independent).
//   A_swz=true  : A swizzled, B no-swizzle   (S:   A=Q_sw,  B=K_fill)
//   A_swz=false : A no-swizzle, B swizzled   (dP:  A=dO_fill,B=V_sw )
// ATOM_ELEMS = rows*64 = 64*64 = 4096 (fixed Br=Bc=64).  8 k16 steps over D=128.
template<bool A_swz>
__device__ __forceinline__ void run_gemm_n64_sw(float acc[32], const bf16* swz, const bf16* nosw) {
    fence_proxy_async_shared();       // orders the generic fill of `nosw` before the async wgmma read
    fence_operandN<32>(acc);          // (the swizzled operand is TMA-written + mbar-ordered already)
    wgmma_fence();
#pragma unroll
    for (int k = 0; k < 8; k++) {
        uint64_t dsw = make_desc_sw128_K(swz + (k >> 2) * 4096 + (k & 3) * 16); // atom=k/4, lk=k%4
        uint64_t dno = make_desc(nosw + 128 * k, (uint64_t)2048);              // tiled_off K=D=128
        if (A_swz) wgmma_m64n64k16(acc, dsw, dno);
        else       wgmma_m64n64k16(acc, dno, dsw);
    }
    wgmma_commit();
    wgmma_wait0();
    fence_operandN<32>(acc);
}

// ── V4 — Kernel 1 — dQ ──  Grid (B,Hq,S/Br), 128 threads (one warpgroup).
//   Persistent (per Q-tile): Q swizzled, dO plain, O plain, LSE, D.
//   Loop-variant (per K-tile, DOUBLE-BUFFERED): K plain, V swizzled.
template<int Br, int Bc, int D>
__global__ void __launch_bounds__(128, 1)
gqa_backward_dQ_v4(
    const __grid_constant__ CUtensorMap tma_Q_sw,   // Q  128B-swizzled (S=Q·Kᵀ, A)
    const __grid_constant__ CUtensorMap tma_V_sw,   // V  128B-swizzled (dP=dO·Vᵀ, B)
    const __grid_constant__ CUtensorMap tma_K_pl,   // K  plain  (S-B fill_copy + dQ-B fill_trans)
    const __grid_constant__ CUtensorMap tma_dO_pl,  // dO plain  (dP-A fill_copy + D compute)
    const __grid_constant__ CUtensorMap tma_O_pl,   // O  plain  (D compute)
    const float * __restrict__ d_LSE, bf16 * __restrict__ d_dQ,
    int B, int Hq, int Hkv, int G, int S, float scale
) {
    static_assert(Br == 64 && Bc == 64 && D == 128, "V4 requires Br=Bc=64, D=128");

    // Persistent operands
    __shared__ __align__(128) bf16 sQ_sw [Br * D];        // 2 swizzle atoms, contiguous
    __shared__ __align__(128) bf16 sdO_pl[Br * D];
    __shared__ __align__(128) bf16 sO_pl [Br * D];
    // Loop-variant, double-buffered [stage]
    __shared__ __align__(128) bf16 sK_pl [2][Bc * D];
    __shared__ __align__(128) bf16 sV_sw [2][Bc * D];     // 2 swizzle atoms per stage
    // Compute scratch / intermediates
    __shared__ __align__(16)  float sS [Br * Bc];
    __shared__ __align__(16)  float sdP[Br * Bc];
    __shared__ __align__(16)  bf16  sP [Br * Bc];         // reused as sdS
    __shared__                float sLSE[Br];
    __shared__                float sD  [Br];
    __shared__ __align__(128) bf16  sA_t[64 * 128];       // no-swizzle wgmma scratch
    __shared__ __align__(128) bf16  sB_t[64 * 128];
    __shared__ __align__(8)   uint64_t mbar_p;            // persistent-load barrier
    __shared__ __align__(8)   uint64_t mbar[2];           // double-buffer barriers

    const int tid = threadIdx.x;
    const int b = blockIdx.x, hq = blockIdx.y, q_tile = blockIdx.z;
    const int hkv = hq / G;
    const int q_row0 = q_tile * Br;
    const int nKTiles = S / Bc;

    const long     qBase    = ((long)(b * Hq  + hq)  * S + q_row0) * D;
    const long     lBase    =  (long)(b * Hq  + hq)  * S + q_row0;
    const uint32_t qFlatRow = (uint32_t)((b * Hq  + hq)  * S + q_row0);
    const uint32_t kvFlatBase = (uint32_t)((b * Hkv + hkv) * S);
    const uint32_t bytesTile  = (uint32_t)(Br * D * sizeof(bf16));   // 16384 (one full tile)
    const uint32_t bytesAtom  = (uint32_t)(Bc * 64 * sizeof(bf16));  // 8192  (one swizzle atom)

    if (tid == 0) { mbar_init_v4(&mbar_p, 1); mbar_init_v4(&mbar[0], 1); mbar_init_v4(&mbar[1], 1); }
    __syncthreads();

    // ── Persistent loads: Q swizzled (2 atoms), dO plain, O plain ──────────────
    uint32_t par_p = 0;
    if (tid == 0) {
        mbar_expect_tx_v4(&mbar_p, bytesAtom * 2 + bytesTile * 2);
        tma_load_2d_v4(&tma_Q_sw,  sQ_sw,           &mbar_p, 0,  qFlatRow);  // atom 0 (cols 0..63)
        tma_load_2d_v4(&tma_Q_sw,  sQ_sw + 64 * 64, &mbar_p, 64, qFlatRow);  // atom 1 (cols 64..127)
        tma_load_2d_v4(&tma_dO_pl, sdO_pl,          &mbar_p, 0,  qFlatRow);
        tma_load_2d_v4(&tma_O_pl,  sO_pl,           &mbar_p, 0,  qFlatRow);
    }
    if (tid < Br) sLSE[tid] = d_LSE[lBase + tid];
    mbar_wait_v4(&mbar_p, par_p);
    __syncthreads();

    // D[r] = rowsum(dO · O) (both plain)
    if (tid < Br) {
        float acc = 0.f;
        for (int j = 0; j < D; j++)
            acc += __bfloat162float(sdO_pl[tid * D + j]) * __bfloat162float(sO_pl[tid * D + j]);
        sD[tid] = acc;
    }
    __syncthreads();

    // Number of non-masked K-tiles (causal)
    const int kc_max = min(nKTiles, (q_row0 + Br + Bc - 1) / Bc);

    // Prologue: TMA K-tile 0 (K plain + V swizzled) → stage 0
    uint32_t par[2] = {0, 0};
    if (kc_max > 0 && tid == 0) {
        mbar_expect_tx_v4(&mbar[0], bytesTile + bytesAtom * 2);
        tma_load_2d_v4(&tma_K_pl, sK_pl[0],           &mbar[0], 0,  kvFlatBase);
        tma_load_2d_v4(&tma_V_sw, sV_sw[0],           &mbar[0], 0,  kvFlatBase);
        tma_load_2d_v4(&tma_V_sw, sV_sw[0] + 64 * 64, &mbar[0], 64, kvFlatBase);
    }

    float dq[64]; zeroN<64>(dq);   // PERSISTENT dQ[Br×D] wgmma accumulator (m64n128, 64 regs)

    for (int kc = 0; kc < kc_max; kc++) {
        const int s = kc & 1;
        // Prefetch NEXT K-tile into the other stage (overlaps this tile's compute)
        if (kc + 1 < kc_max && tid == 0) {
            const uint32_t nr = kvFlatBase + (uint32_t)((kc + 1) * Bc);
            mbar_expect_tx_v4(&mbar[s ^ 1], bytesTile + bytesAtom * 2);
            tma_load_2d_v4(&tma_K_pl, sK_pl[s ^ 1],           &mbar[s ^ 1], 0,  nr);
            tma_load_2d_v4(&tma_V_sw, sV_sw[s ^ 1],           &mbar[s ^ 1], 0,  nr);
            tma_load_2d_v4(&tma_V_sw, sV_sw[s ^ 1] + 64 * 64, &mbar[s ^ 1], 64, nr);
        }
        mbar_wait_v4(&mbar[s], par[s]); par[s] ^= 1;
        __syncthreads();

        // S = Q·Kᵀ·scale  (A=Q swizzled, B=K no-swizzle via fill_copy)
        fill_copy<Bc, D>(sB_t, sK_pl[s], tid);
        __syncthreads();
        { float acc[32]; zeroN<32>(acc); run_gemm_n64_sw<true>(acc, sQ_sw, sB_t); store_acc_smem<Bc>(acc, sS, tid, scale); }
        __syncthreads();

        // P = exp(S - LSE), causal mask → sP
        for (int i = tid; i < Br * Bc; i += 128) {
            int r = i / Bc, c = i % Bc;
            sP[i] = (kc * Bc + c > q_row0 + r) ? __float2bfloat16(0.f)
                                               : __float2bfloat16(expf(sS[i] - sLSE[r]));
        }
        __syncthreads();

        // dP = dO·Vᵀ  (A=dO no-swizzle via fill_copy, B=V swizzled)
        fill_copy<Br, D>(sA_t, sdO_pl, tid);
        __syncthreads();
        { float acc[32]; zeroN<32>(acc); run_gemm_n64_sw<false>(acc, sV_sw[s], sA_t); store_acc_smem<Bc>(acc, sdP, tid, 1.0f); }
        __syncthreads();

        // dS = P ⊙ (dP - D) → sP
        for (int i = tid; i < Br * Bc; i += 128) {
            int r = i / Bc;
            sP[i] = __float2bfloat16(__bfloat162float(sP[i]) * (sdP[i] - sD[r]));
        }
        __syncthreads();

        // dQ += dS·K  (A=dS no-swizzle fill_copy, B=Kᵀ no-swizzle fill_trans of plain K)
        fill_copy<Br, Bc>(sA_t, sP, tid); fill_trans<D, Bc>(sB_t, sK_pl[s], tid);
        __syncthreads();
        run_gemm_n128<Bc / 16>(dq, sA_t, sB_t, (uint64_t)(Bc >> 3) * 128);   // SBO=1024
        __syncthreads();
    }

    fence_operandN<64>(dq);
    store_acc_global<D>(dq, d_dQ, qBase, D, tid, scale);
}

// ── V4 — Kernel 2 — dK, dV ──  Grid (B,Hkv,S/Bc), 128 threads (one warpgroup).
//   Persistent (per K-tile): K swizzled, V swizzled.
//   Loop-variant (per Q-tile over G×qc, DOUBLE-BUFFERED): Q plain, dO plain, O plain.
template<int Br, int Bc, int D>
__global__ void __launch_bounds__(128, 1)
gqa_backward_dKdV_v4(
    const __grid_constant__ CUtensorMap tma_K_sw,   // K  128B-swizzled (S=Q·Kᵀ, B)
    const __grid_constant__ CUtensorMap tma_V_sw,   // V  128B-swizzled (dP=dO·Vᵀ, B)
    const __grid_constant__ CUtensorMap tma_Q_pl,   // Q  plain (S-A fill_copy + dK-B fill_trans)
    const __grid_constant__ CUtensorMap tma_dO_pl,  // dO plain (dP-A fill_copy + dV-B fill_trans + D)
    const __grid_constant__ CUtensorMap tma_O_pl,   // O  plain (D compute)
    const float * __restrict__ d_LSE, bf16 * __restrict__ d_dK, bf16 * __restrict__ d_dV,
    int B, int Hq, int Hkv, int G, int S, float scale
) {
    static_assert(Br == 64 && Bc == 64 && D == 128, "V4 requires Br=Bc=64, D=128");

    // Persistent operands (K-tile fixed for the block)
    __shared__ __align__(128) bf16 sK_sw[Bc * D];
    __shared__ __align__(128) bf16 sV_sw[Bc * D];
    // Loop-variant, double-buffered [stage]
    __shared__ __align__(128) bf16 sQ_pl [2][Br * D];
    __shared__ __align__(128) bf16 sdO_pl[2][Br * D];
    __shared__ __align__(128) bf16 sO_pl [2][Br * D];
    // Compute scratch / intermediates
    __shared__ __align__(16)  float sS [Br * Bc];
    __shared__ __align__(16)  float sdP[Br * Bc];
    __shared__ __align__(16)  bf16  sP [Br * Bc];         // reused as sdS
    __shared__                float sLSE[Br];
    __shared__                float sD  [Br];
    __shared__ __align__(128) bf16  sA_t[64 * 128];
    __shared__ __align__(128) bf16  sB_t[64 * 128];
    __shared__ __align__(8)   uint64_t mbar_p;
    __shared__ __align__(8)   uint64_t mbar[2];

    const int tid = threadIdx.x;
    const int b = blockIdx.x, hkv = blockIdx.y, k_tile = blockIdx.z;
    const int k_row0 = k_tile * Bc;
    const int nQTiles = S / Br;

    const long     kvBase     = ((long)(b * Hkv + hkv) * S + k_row0) * D;
    const uint32_t kvFlatRow  = (uint32_t)((b * Hkv + hkv) * S + k_row0);
    const uint32_t bytesTile  = (uint32_t)(Br * D * sizeof(bf16));   // 16384
    const uint32_t bytesAtom  = (uint32_t)(Bc * 64 * sizeof(bf16));  // 8192

    if (tid == 0) { mbar_init_v4(&mbar_p, 1); mbar_init_v4(&mbar[0], 1); mbar_init_v4(&mbar[1], 1); }
    __syncthreads();

    // ── Persistent loads: K swizzled (2 atoms), V swizzled (2 atoms) ───────────
    uint32_t par_p = 0;
    if (tid == 0) {
        mbar_expect_tx_v4(&mbar_p, bytesAtom * 4);
        tma_load_2d_v4(&tma_K_sw, sK_sw,           &mbar_p, 0,  kvFlatRow);
        tma_load_2d_v4(&tma_K_sw, sK_sw + 64 * 64, &mbar_p, 64, kvFlatRow);
        tma_load_2d_v4(&tma_V_sw, sV_sw,           &mbar_p, 0,  kvFlatRow);
        tma_load_2d_v4(&tma_V_sw, sV_sw + 64 * 64, &mbar_p, 64, kvFlatRow);
    }
    mbar_wait_v4(&mbar_p, par_p);
    __syncthreads();

    float dv[64]; zeroN<64>(dv);   // PERSISTENT dV[Bc×D] accumulator (64 regs)
    float dk[64]; zeroN<64>(dk);   // PERSISTENT dK[Bc×D] accumulator (64 regs)

    // Flatten the G × qc iteration space so the double-buffer prefetch-ahead is simple.
    const int qc0    = k_row0 / Br;
    const int perG   = nQTiles - qc0;   // Q-tiles per group
    const int nIter  = G * perG;

    auto qFlatRowOf = [&](int it) -> uint32_t {
        const int g  = it / perG;
        const int qc = qc0 + (it % perG);
        const int hq = hkv * G + g;
        return (uint32_t)((b * Hq + hq) * S + qc * Br);
    };
    auto lBaseOf = [&](int it) -> long {
        const int g  = it / perG;
        const int qc = qc0 + (it % perG);
        const int hq = hkv * G + g;
        return (long)(b * Hq + hq) * S + (long)qc * Br;
    };

    // Prologue: TMA iter 0 (Q,dO,O plain) → stage 0
    uint32_t par[2] = {0, 0};
    if (nIter > 0 && tid == 0) {
        const uint32_t r0 = qFlatRowOf(0);
        mbar_expect_tx_v4(&mbar[0], bytesTile * 3);
        tma_load_2d_v4(&tma_Q_pl,  sQ_pl [0], &mbar[0], 0, r0);
        tma_load_2d_v4(&tma_dO_pl, sdO_pl[0], &mbar[0], 0, r0);
        tma_load_2d_v4(&tma_O_pl,  sO_pl [0], &mbar[0], 0, r0);
    }

    for (int it = 0; it < nIter; it++) {
        const int s = it & 1;
        const int q_row0 = (qc0 + (it % perG)) * Br;
        // Prefetch NEXT iter → other stage
        if (it + 1 < nIter && tid == 0) {
            const uint32_t nr = qFlatRowOf(it + 1);
            mbar_expect_tx_v4(&mbar[s ^ 1], bytesTile * 3);
            tma_load_2d_v4(&tma_Q_pl,  sQ_pl [s ^ 1], &mbar[s ^ 1], 0, nr);
            tma_load_2d_v4(&tma_dO_pl, sdO_pl[s ^ 1], &mbar[s ^ 1], 0, nr);
            tma_load_2d_v4(&tma_O_pl,  sO_pl [s ^ 1], &mbar[s ^ 1], 0, nr);
        }
        mbar_wait_v4(&mbar[s], par[s]); par[s] ^= 1;
        if (tid < Br) sLSE[tid] = d_LSE[lBaseOf(it) + tid];
        __syncthreads();

        // D[r] = rowsum(dO · O)
        if (tid < Br) {
            float acc = 0.f;
            for (int j = 0; j < D; j++)
                acc += __bfloat162float(sdO_pl[s][tid * D + j]) * __bfloat162float(sO_pl[s][tid * D + j]);
            sD[tid] = acc;
        }
        __syncthreads();

        // S = Q·Kᵀ·scale  (A=Q no-swizzle fill_copy, B=K swizzled)
        fill_copy<Br, D>(sA_t, sQ_pl[s], tid);
        __syncthreads();
        { float acc[32]; zeroN<32>(acc); run_gemm_n64_sw<false>(acc, sK_sw, sA_t); store_acc_smem<Bc>(acc, sS, tid, scale); }
        __syncthreads();

        // P = exp(S - LSE), causal mask → sP
        for (int i = tid; i < Br * Bc; i += 128) {
            int r = i / Bc, c = i % Bc;
            sP[i] = (k_row0 + c > q_row0 + r) ? __float2bfloat16(0.f)
                                              : __float2bfloat16(expf(sS[i] - sLSE[r]));
        }
        __syncthreads();

        // dV += Pᵀ·dO  (A=Pᵀ fill_trans_A intermediate, B=dOᵀ fill_trans of plain dO)
        fill_trans_A<Bc, Br>(sA_t, sP, tid); fill_trans<D, Br>(sB_t, sdO_pl[s], tid);
        __syncthreads();
        run_gemm_n128_tA<Br / 16>(dv, sA_t, sB_t, (uint64_t)(Br >> 3) * 128);   // SBO=1024
        __syncthreads();

        // dP = dO·Vᵀ  (A=dO no-swizzle fill_copy, B=V swizzled)
        fill_copy<Br, D>(sA_t, sdO_pl[s], tid);
        __syncthreads();
        { float acc[32]; zeroN<32>(acc); run_gemm_n64_sw<false>(acc, sV_sw, sA_t); store_acc_smem<Bc>(acc, sdP, tid, 1.0f); }
        __syncthreads();

        // dS = P ⊙ (dP - D) → sP
        for (int i = tid; i < Br * Bc; i += 128) {
            int r = i / Bc;
            sP[i] = __float2bfloat16(__bfloat162float(sP[i]) * (sdP[i] - sD[r]));
        }
        __syncthreads();

        // dK += dSᵀ·Q  (A=dSᵀ fill_trans_A intermediate, B=Qᵀ fill_trans of plain Q)
        fill_trans_A<Bc, Br>(sA_t, sP, tid); fill_trans<D, Br>(sB_t, sQ_pl[s], tid);
        __syncthreads();
        run_gemm_n128_tA<Br / 16>(dk, sA_t, sB_t, (uint64_t)(Br >> 3) * 128);
        __syncthreads();
    }

    fence_operandN<64>(dv); fence_operandN<64>(dk);
    store_acc_global<D>(dv, d_dV, kvBase, D, tid, 1.0f);
    store_acc_global<D>(dk, d_dK, kvBase, D, tid, scale);
}

// ── V4 launcher — builds TMA descriptors (SWIZZLE_128B for MMA operands, NONE for
// plain), raises the dynamic-smem cap defensively, launches both kernels. ────────
template<int Br, int Bc, int D>
void launch_gqa_backward_v4(
    const bf16 *d_Q, const bf16 *d_K, const bf16 *d_V, const bf16 *d_O,
    const bf16 *d_dO, const float *d_LSE,
    bf16 *d_dQ, bf16 *d_dK, bf16 *d_dV,
    int B, int Hq, int Hkv, int G, int S, float scale
) {
    static_assert(Br == 64 && Bc == 64 && D == 128, "V4 requires Br=Bc=64, D=128");

    // 128B-swizzled descriptor: box is ONE 64-wide (128 B) atom × tile_rows; the
    // kernel issues 2 loads (col origins 0 and 64) to cover D=128 = 2 atoms.
    auto make_tma_sw128 = [&](const bf16* ptr, uint64_t total_rows, uint32_t tile_rows) {
        CUtensorMap desc{};
        uint64_t gSize[2]   = {(uint64_t)D, total_rows};              // {128, rows}, dim0 contiguous
        uint64_t gStride[1] = {(uint64_t)D * sizeof(bf16)};           // 256 B row stride
        uint32_t box[2]     = {64u, tile_rows};                       // one 128-B swizzle atom wide
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
    // Plain (row-major) descriptor: full D-wide box, no swizzle.
    auto make_tma_plain = [&](const bf16* ptr, uint64_t total_rows, uint32_t tile_rows) {
        CUtensorMap desc{};
        uint64_t gSize[2]   = {(uint64_t)D, total_rows};
        uint64_t gStride[1] = {(uint64_t)D * sizeof(bf16)};
        uint32_t box[2]     = {(uint32_t)D, tile_rows};
        uint32_t eStride[2] = {1, 1};
        CUresult r = cuTensorMapEncodeTiled(
            &desc, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)ptr,
            gSize, gStride, box, eStride,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (r != CUDA_SUCCESS) { const char* e; cuGetErrorString(r, &e);
            fprintf(stderr, "cuTensorMapEncodeTiled(plain) failed: %s\n", e); exit(1); }
        return desc;
    };

    const uint64_t Rq  = (uint64_t)B * Hq  * S;
    const uint64_t Rkv = (uint64_t)B * Hkv * S;

    CUtensorMap tma_Q_sw  = make_tma_sw128(d_Q,  Rq,  Br);
    CUtensorMap tma_Q_pl  = make_tma_plain(d_Q,  Rq,  Br);
    CUtensorMap tma_K_sw  = make_tma_sw128(d_K,  Rkv, Bc);
    CUtensorMap tma_K_pl  = make_tma_plain(d_K,  Rkv, Bc);
    CUtensorMap tma_V_sw  = make_tma_sw128(d_V,  Rkv, Bc);
    CUtensorMap tma_dO_pl = make_tma_plain(d_dO, Rq,  Br);
    CUtensorMap tma_O_pl  = make_tma_plain(d_O,  Rq,  Br);

    // NOTE on smem: V4 uses only STATIC __shared__ (dQ 188,952 B; dKdV 205,336 B),
    // both < the 227 KB H200 limit. This mirrors V3's large-static path (156 KB),
    // which runs on H200 with no opt-in call — static shared is reserved at launch
    // automatically. cudaFuncAttributeMaxDynamicSharedMemorySize governs only the
    // *dynamic* (extern __shared__) portion, which V4 does not use, so no attribute
    // call is needed here. FALLBACK if a driver ever rejects the large static block:
    // move the buffers to one extern __shared__ arena + manual offsets and call
    // cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes).

    constexpr dim3 BLOCK(128);
    dim3 GRID1(B, Hq,  S / Br);
    dim3 GRID2(B, Hkv, S / Bc);
    gqa_backward_dQ_v4  <Br,Bc,D><<<GRID1, BLOCK>>>(
        tma_Q_sw, tma_V_sw, tma_K_pl, tma_dO_pl, tma_O_pl, d_LSE, d_dQ,
        B, Hq, Hkv, G, S, scale);
    gqa_backward_dKdV_v4<Br,Bc,D><<<GRID2, BLOCK>>>(
        tma_K_sw, tma_V_sw, tma_Q_pl, tma_dO_pl, tma_O_pl, d_LSE, d_dK, d_dV,
        B, Hq, Hkv, G, S, scale);
}

// ═════════════════════════════════════════════════════════════════════════════
// V5 — SINGLE FUSED, KV-CENTRIC kernel (replaces V4's two-kernel dQ + dKdV split)
//                                                            [SM_90a only]
//
// One kernel, grid (B, Hkv, S/Bc) — one block per (batch, kv-head, kv-tile).  The
// block OWNS its KV-tile's dK and dV (they reduce over the QUERY axis, so a single
// block sees every contribution → persistent wgmma accumulators dk[64]/dv[64], no
// atomics, one writeback at the end — identical to V4's dKdV kernel).
//
// dQ reduces over the KEY axis, so a given query row receives a partial from EVERY
// KV-tile it attends → NOT owned by one block.  Each (g, qc) iteration computes that
// Q-tile's dQ contribution from THIS block's KV-tile alone into a TRANSIENT wgmma
// accumulator dq[64] (m64n128, dQ_tile = dS·K), then flushes it via fp32 atomicAdd
// into a scratch d_dq_accum[B*Hq*S*D] buffer (the FA2/FA3 dq_accum pattern the
// Blackwell GQA_sm103_bwd.cu v11_kv kernel uses).  A tiny convert kernel turns the
// fp32 scratch into the final bf16 d_dQ once the fused kernel is done.  The launcher
// owns the scratch: cudaMalloc + cudaMemset(0) before, convert + cudaFree after.
//
// WHY fp32 atomic + convert (not a bf16 atomic on d_dQ directly): bf16 atomicAdd is
// unsupported on sm_90, and even where a 16-bit atomic exists, summing many KV-tile
// partials in bf16 would lose precision catastrophically (each Q row accumulates
// O(S/Bc) terms).  Accumulating in fp32 then a single bf16 round matches the
// reference math and V4's dQ (which also accumulates in fp32 registers, then rounds
// once in store_acc_global).
//
// The per-(g,qc) compute is V4's dKdV body VERBATIM (same TMA double-buffer, same
// swizzled/plain operand roles, same fences, same fill_*/run_gemm_* helpers) with
// ONE addition: the dQ = dS·K wgmma + atomic flush, reusing V4's dQ-kernel dQ path
// (fill_copy dS  →  A;  fill_trans of PLAIN K  →  B;  run_gemm_n128, SBO=1024).
// That fill_trans needs K in the PLAIN (D-contiguous) layout, whereas S=Q·Kᵀ reads K
// SWIZZLED — so V5 loads K BOTH ways into two persistent buffers (sK_sw + sK_pl),
// each on its own already-verified path (V4 dKdV's swizzled-K S-GEMM; V4 dQ's
// plain-K fill_trans dQ-GEMM).  +16 KB of persistent smem vs V4 dKdV.
// ═════════════════════════════════════════════════════════════════════════════

// Register-accumulator → fp32 scratch, ATOMIC accumulate.  Same m64nNk16 D-fragment
// mapping as store_acc_global, but atomicAdd into an fp32 buffer instead of a plain
// bf16 store (many KV-tiles race to the same query rows).  NCOL = output width = D.
// `coloff` is the D-column base for the fragment (0 or 64 for the two m64n64 halves
// of the D=128 dQ tile — see the split note in the kernel).
template<int NCOL>
__device__ __forceinline__ void atomic_acc_global(const float *d, float *g, long base, int D, int coloff, int tid, float scl) {
    int w = tid >> 5, lane = tid & 31;
    int r0 = w * 16 + (lane >> 2), r1 = r0 + 8, cc = (lane & 3) * 2;
#pragma unroll
    for (int nt = 0; nt < NCOL / 8; nt++) {
        int c = coloff + nt * 8 + cc;
        atomicAdd(&g[base + (long)r0 * D + c + 0], d[nt * 4 + 0] * scl);
        atomicAdd(&g[base + (long)r0 * D + c + 1], d[nt * 4 + 1] * scl);
        atomicAdd(&g[base + (long)r1 * D + c + 0], d[nt * 4 + 2] * scl);
        atomicAdd(&g[base + (long)r1 * D + c + 1], d[nt * 4 + 3] * scl);
    }
}

// Converts the fp32 dq_accum scratch (written via atomicAdd by V5) to bf16 d_dQ.
__global__ void convert_dq_accum_to_bf16_v5(
    const float * __restrict__ d_dq_accum, bf16 * __restrict__ d_dQ, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d_dQ[i] = __float2bfloat16(d_dq_accum[i]);
}

// ── V5 — fused dQ + dK + dV ──  Grid (B,Hkv,S/Bc), 128 threads (one warpgroup).
//   Persistent (per K-tile): K swizzled (S=Q·Kᵀ B), K plain (dQ=dS·K B), V swizzled.
//   Loop-variant (per Q-tile over G×qc, DOUBLE-BUFFERED): Q plain, dO plain, O plain.
template<int Br, int Bc, int D>
__global__ void __launch_bounds__(128, 1)
gqa_backward_v5_kv(
    const __grid_constant__ CUtensorMap tma_K_sw,   // K 128B-swizzled (S=Q·Kᵀ, B)
    const __grid_constant__ CUtensorMap tma_K_pl,   // K plain         (dQ=dS·K, B via fill_trans)
    const __grid_constant__ CUtensorMap tma_V_sw,   // V 128B-swizzled (dP=dO·Vᵀ, B)
    const __grid_constant__ CUtensorMap tma_Q_pl,   // Q plain (S-A fill_copy + dK-B fill_trans)
    const __grid_constant__ CUtensorMap tma_dO_pl,  // dO plain (dP-A fill_copy + dV-B fill_trans + D)
    const __grid_constant__ CUtensorMap tma_O_pl,   // O  plain (D compute)
    const float * __restrict__ d_LSE,
    bf16 * __restrict__ d_dK, bf16 * __restrict__ d_dV, float * __restrict__ d_dq_accum,
    int B, int Hq, int Hkv, int G, int S, float scale
) {
    static_assert(Br == 64 && Bc == 64 && D == 128, "V5 requires Br=Bc=64, D=128");

    // Persistent operands (K-tile fixed for the block)
    __shared__ __align__(128) bf16 sK_sw[Bc * D];        // 2 swizzle atoms (S-B)
    __shared__ __align__(128) bf16 sK_pl[Bc * D];        // plain (dQ-B fill_trans)
    __shared__ __align__(128) bf16 sV_sw[Bc * D];        // 2 swizzle atoms (dP-B)
    // Loop-variant, double-buffered [stage]
    __shared__ __align__(128) bf16 sQ_pl [2][Br * D];
    __shared__ __align__(128) bf16 sdO_pl[2][Br * D];
    __shared__ __align__(128) bf16 sO_pl [2][Br * D];
    // Compute scratch / intermediates
    __shared__ __align__(16)  float sS [Br * Bc];
    __shared__ __align__(16)  float sdP[Br * Bc];
    __shared__ __align__(16)  bf16  sP [Br * Bc];         // reused as sdS
    __shared__                float sLSE[Br];
    __shared__                float sD  [Br];
    __shared__ __align__(128) bf16  sA_t[64 * 128];
    __shared__ __align__(128) bf16  sB_t[64 * 128];
    __shared__ __align__(8)   uint64_t mbar_p;
    __shared__ __align__(8)   uint64_t mbar[2];

    const int tid = threadIdx.x;
    const int b = blockIdx.x, hkv = blockIdx.y, k_tile = blockIdx.z;
    const int k_row0 = k_tile * Bc;
    const int nQTiles = S / Br;

    const long     kvBase     = ((long)(b * Hkv + hkv) * S + k_row0) * D;
    const uint32_t kvFlatRow  = (uint32_t)((b * Hkv + hkv) * S + k_row0);
    const uint32_t bytesTile  = (uint32_t)(Br * D * sizeof(bf16));   // 16384
    const uint32_t bytesAtom  = (uint32_t)(Bc * 64 * sizeof(bf16));  // 8192

    if (tid == 0) { mbar_init_v4(&mbar_p, 1); mbar_init_v4(&mbar[0], 1); mbar_init_v4(&mbar[1], 1); }
    __syncthreads();

    // ── Persistent loads: K swizzled (2 atoms) + K plain (1 full-D) + V swizzled (2)
    uint32_t par_p = 0;
    if (tid == 0) {
        mbar_expect_tx_v4(&mbar_p, bytesAtom * 4 + bytesTile);
        tma_load_2d_v4(&tma_K_sw, sK_sw,           &mbar_p, 0,  kvFlatRow);
        tma_load_2d_v4(&tma_K_sw, sK_sw + 64 * 64, &mbar_p, 64, kvFlatRow);
        tma_load_2d_v4(&tma_V_sw, sV_sw,           &mbar_p, 0,  kvFlatRow);
        tma_load_2d_v4(&tma_V_sw, sV_sw + 64 * 64, &mbar_p, 64, kvFlatRow);
        tma_load_2d_v4(&tma_K_pl, sK_pl,           &mbar_p, 0,  kvFlatRow);
    }
    mbar_wait_v4(&mbar_p, par_p);
    __syncthreads();

    float dv[64]; zeroN<64>(dv);   // PERSISTENT dV[Bc×D] accumulator (owned by this block)
    float dk[64]; zeroN<64>(dk);   // PERSISTENT dK[Bc×D] accumulator (owned by this block)

    // Flatten the G × qc causal iteration space (qc from k_row0/Br upward).
    const int qc0    = k_row0 / Br;
    const int perG   = nQTiles - qc0;
    const int nIter  = G * perG;

    auto qFlatRowOf = [&](int it) -> uint32_t {
        const int g  = it / perG;
        const int qc = qc0 + (it % perG);
        const int hq = hkv * G + g;
        return (uint32_t)((b * Hq + hq) * S + qc * Br);
    };
    auto lBaseOf = [&](int it) -> long {
        const int g  = it / perG;
        const int qc = qc0 + (it % perG);
        const int hq = hkv * G + g;
        return (long)(b * Hq + hq) * S + (long)qc * Br;
    };

    // Prologue: TMA iter 0 (Q,dO,O plain) → stage 0
    uint32_t par[2] = {0, 0};
    if (nIter > 0 && tid == 0) {
        const uint32_t r0 = qFlatRowOf(0);
        mbar_expect_tx_v4(&mbar[0], bytesTile * 3);
        tma_load_2d_v4(&tma_Q_pl,  sQ_pl [0], &mbar[0], 0, r0);
        tma_load_2d_v4(&tma_dO_pl, sdO_pl[0], &mbar[0], 0, r0);
        tma_load_2d_v4(&tma_O_pl,  sO_pl [0], &mbar[0], 0, r0);
    }

    for (int it = 0; it < nIter; it++) {
        const int s = it & 1;
        const int q_row0 = (qc0 + (it % perG)) * Br;
        // Prefetch NEXT iter → other stage
        if (it + 1 < nIter && tid == 0) {
            const uint32_t nr = qFlatRowOf(it + 1);
            mbar_expect_tx_v4(&mbar[s ^ 1], bytesTile * 3);
            tma_load_2d_v4(&tma_Q_pl,  sQ_pl [s ^ 1], &mbar[s ^ 1], 0, nr);
            tma_load_2d_v4(&tma_dO_pl, sdO_pl[s ^ 1], &mbar[s ^ 1], 0, nr);
            tma_load_2d_v4(&tma_O_pl,  sO_pl [s ^ 1], &mbar[s ^ 1], 0, nr);
        }
        mbar_wait_v4(&mbar[s], par[s]); par[s] ^= 1;
        if (tid < Br) sLSE[tid] = d_LSE[lBaseOf(it) + tid];
        __syncthreads();

        // D[r] = rowsum(dO · O)
        if (tid < Br) {
            float acc = 0.f;
            for (int j = 0; j < D; j++)
                acc += __bfloat162float(sdO_pl[s][tid * D + j]) * __bfloat162float(sO_pl[s][tid * D + j]);
            sD[tid] = acc;
        }
        __syncthreads();

        // S = Q·Kᵀ·scale  (A=Q no-swizzle fill_copy, B=K swizzled)
        fill_copy<Br, D>(sA_t, sQ_pl[s], tid);
        __syncthreads();
        { float acc[32]; zeroN<32>(acc); run_gemm_n64_sw<false>(acc, sK_sw, sA_t); store_acc_smem<Bc>(acc, sS, tid, scale); }
        __syncthreads();

        // P = exp(S - LSE), causal mask → sP
        for (int i = tid; i < Br * Bc; i += 128) {
            int r = i / Bc, c = i % Bc;
            sP[i] = (k_row0 + c > q_row0 + r) ? __float2bfloat16(0.f)
                                              : __float2bfloat16(expf(sS[i] - sLSE[r]));
        }
        __syncthreads();

        // dV += Pᵀ·dO  (A=Pᵀ fill_trans_A intermediate, B=dOᵀ fill_trans of plain dO)
        fill_trans_A<Bc, Br>(sA_t, sP, tid); fill_trans<D, Br>(sB_t, sdO_pl[s], tid);
        __syncthreads();
        run_gemm_n128_tA<Br / 16>(dv, sA_t, sB_t, (uint64_t)(Br >> 3) * 128);   // SBO=1024
        __syncthreads();

        // dP = dO·Vᵀ  (A=dO no-swizzle fill_copy, B=V swizzled)
        fill_copy<Br, D>(sA_t, sdO_pl[s], tid);
        __syncthreads();
        { float acc[32]; zeroN<32>(acc); run_gemm_n64_sw<false>(acc, sV_sw, sA_t); store_acc_smem<Bc>(acc, sdP, tid, 1.0f); }
        __syncthreads();

        // dS = P ⊙ (dP - D) → sP
        for (int i = tid; i < Br * Bc; i += 128) {
            int r = i / Bc;
            sP[i] = __float2bfloat16(__bfloat162float(sP[i]) * (sdP[i] - sD[r]));
        }
        __syncthreads();

        // dK += dSᵀ·Q  (A=dSᵀ fill_trans_A intermediate, B=Qᵀ fill_trans of plain Q)
        fill_trans_A<Bc, Br>(sA_t, sP, tid); fill_trans<D, Br>(sB_t, sQ_pl[s], tid);
        __syncthreads();
        run_gemm_n128_tA<Br / 16>(dk, sA_t, sB_t, (uint64_t)(Br >> 3) * 128);
        __syncthreads();

        // dQ_tile = dS·K  (TRANSIENT; A=dS no-swizzle fill_copy, B=Kᵀ no-swizzle
        // fill_trans of PLAIN K).  Reset every iteration (each is a different Q-tile's
        // partial), then flush via fp32 atomicAdd into d_dq_accum.
        //
        // REGISTER NOTE: dv[64]+dk[64] persist across the loop, so a full m64n128 dq[64]
        // here would peak at 192 accumulator regs → 255 + 4-byte spill.  Instead the D=128
        // output is done as TWO m64n64 halves (acc[32] each, reused): peak transient drops
        // to 32 regs → 0 spills.  Half 0 reads B cols [0,64) at sB_t+0; half 1 reads cols
        // [64,128) at sB_t+4096 (= tiled_off(64,0,Bc); the 8 mn-blocks 8..15, SBO=1024
        // strides them exactly as run_gemm_n128 does the full 16).  Both halves share the
        // one fill of sA_t/sB_t.  run_gemm_n64 is V3's HW-verified no-swizzle GEMM.
        fill_copy<Br, Bc>(sA_t, sP, tid); fill_trans<D, Bc>(sB_t, sK_pl, tid);
        __syncthreads();
        { float acc[32]; zeroN<32>(acc);
          run_gemm_n64<Bc / 16>(acc, sA_t, sB_t, (uint64_t)(Bc >> 3) * 128);          // SBO=1024
          atomic_acc_global<64>(acc, d_dq_accum, lBaseOf(it) * D, D, 0,  tid, scale); }
        { float acc[32]; zeroN<32>(acc);
          run_gemm_n64<Bc / 16>(acc, sA_t, sB_t + 4096, (uint64_t)(Bc >> 3) * 128);   // SBO=1024
          atomic_acc_global<64>(acc, d_dq_accum, lBaseOf(it) * D, D, 64, tid, scale); }
        __syncthreads();
    }

    fence_operandN<64>(dv); fence_operandN<64>(dk);
    store_acc_global<D>(dv, d_dV, kvBase, D, tid, 1.0f);
    store_acc_global<D>(dk, d_dK, kvBase, D, tid, scale);
}

// ── V5 launcher — single fused kernel + fp32 dQ-scratch (alloc/zero/convert/free).
// Grid (B,Hkv,S/Bc).  Descriptors: K swizzled + K plain + V swizzled (persistent),
// Q/dO/O plain (double-buffered).  Same make_tma helpers as V4.
template<int Br, int Bc, int D>
void launch_gqa_backward_v5(
    const bf16 *d_Q, const bf16 *d_K, const bf16 *d_V, const bf16 *d_O,
    const bf16 *d_dO, const float *d_LSE,
    bf16 *d_dQ, bf16 *d_dK, bf16 *d_dV,
    int B, int Hq, int Hkv, int G, int S, float scale
) {
    static_assert(Br == 64 && Bc == 64 && D == 128, "V5 requires Br=Bc=64, D=128");

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
    auto make_tma_plain = [&](const bf16* ptr, uint64_t total_rows, uint32_t tile_rows) {
        CUtensorMap desc{};
        uint64_t gSize[2]   = {(uint64_t)D, total_rows};
        uint64_t gStride[1] = {(uint64_t)D * sizeof(bf16)};
        uint32_t box[2]     = {(uint32_t)D, tile_rows};
        uint32_t eStride[2] = {1, 1};
        CUresult r = cuTensorMapEncodeTiled(
            &desc, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)ptr,
            gSize, gStride, box, eStride,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (r != CUDA_SUCCESS) { const char* e; cuGetErrorString(r, &e);
            fprintf(stderr, "cuTensorMapEncodeTiled(plain) failed: %s\n", e); exit(1); }
        return desc;
    };

    const uint64_t Rq  = (uint64_t)B * Hq  * S;
    const uint64_t Rkv = (uint64_t)B * Hkv * S;

    CUtensorMap tma_K_sw  = make_tma_sw128(d_K,  Rkv, Bc);
    CUtensorMap tma_K_pl  = make_tma_plain(d_K,  Rkv, Bc);
    CUtensorMap tma_V_sw  = make_tma_sw128(d_V,  Rkv, Bc);
    CUtensorMap tma_Q_pl  = make_tma_plain(d_Q,  Rq,  Br);
    CUtensorMap tma_dO_pl = make_tma_plain(d_dO, Rq,  Br);
    CUtensorMap tma_O_pl  = make_tma_plain(d_O,  Rq,  Br);

    // fp32 dQ accumulator scratch — this launcher OWNS it (alloc/zero/convert/free).
    // Each query row receives a partial from every KV-tile it attends; those partials
    // race across blocks and must sum in fp32 before a single bf16 round.
    const long dqN = (long)B * Hq * S * D;
    float* d_dq_accum = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dq_accum, dqN * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dq_accum, 0, dqN * sizeof(float)));

    constexpr dim3 BLOCK(128);
    dim3 GRID(B, Hkv, S / Bc);
    gqa_backward_v5_kv<Br,Bc,D><<<GRID, BLOCK>>>(
        tma_K_sw, tma_K_pl, tma_V_sw, tma_Q_pl, tma_dO_pl, tma_O_pl,
        d_LSE, d_dK, d_dV, d_dq_accum, B, Hq, Hkv, G, S, scale);

    const int convBlock = 256;
    const int convGrid  = (int)((dqN + convBlock - 1) / convBlock);
    convert_dq_accum_to_bf16_v5<<<convGrid, convBlock>>>(d_dq_accum, d_dQ, dqN);

    CUDA_CHECK(cudaFree(d_dq_accum));
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main(){
    std::cout << "GQA Backward — precision test  [Hopper SM_90 / H200]\n";
    std::cout << "Prerequisite: python precision/baseline_gqa.py\n\n";

    constexpr int B   = 8, Hq  = 12, Hkv = 4, G = Hq / Hkv;
    constexpr int S   = 4096, D = 128, Br = 16, Bc = 32;
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

    launch_gqa_backward_v4<Br2,Bc2,D>(d_Q,d_K,d_V,d_O,d_dO,d_LSE,d_dQ,d_dK,d_dV,B,Hq,Hkv,G,S,scale);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    check("── V4  Br=64, Bc=64  TMA+wgmma (128B swizzle, double-buffered) ──", Nq, Nkv, d_dQ, d_dK, d_dV);

    launch_gqa_backward_v5<Br2,Bc2,D>(d_Q,d_K,d_V,d_O,d_dO,d_LSE,d_dQ,d_dK,d_dV,B,Hq,Hkv,G,S,scale);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    check("── V5  Br=64, Bc=64  FUSED KV-centric TMA+wgmma (dQ fp32 atomic) ──", Nq, Nkv, d_dQ, d_dK, d_dV);

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
    {
        KernelStats s = benchmarkKernel(
            [&](){ launch_gqa_backward_v4<Br2,Bc2,D>(
                d_Q,d_K,d_V,d_O,d_dO,d_LSE,d_dQ,d_dK,d_dV,B,Hq,Hkv,G,S,scale); },
            100, 10, bwd_flops);
        displayStats("GQA bwd V4  Br=64, Bc=64  TMA+wgmma  (Hopper SM_90)", s);
    }
    {
        KernelStats s = benchmarkKernel(
            [&](){ launch_gqa_backward_v5<Br2,Bc2,D>(
                d_Q,d_K,d_V,d_O,d_dO,d_LSE,d_dQ,d_dK,d_dV,B,Hq,Hkv,G,S,scale); },
            100, 10, bwd_flops);
        displayStats("GQA bwd V5  Br=64, Bc=64  FUSED KV-centric  (Hopper SM_90)", s);
    }

    CUDA_CHECK(cudaFree(d_Q));  CUDA_CHECK(cudaFree(d_K));  CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));  CUDA_CHECK(cudaFree(d_dO)); CUDA_CHECK(cudaFree(d_LSE));
    CUDA_CHECK(cudaFree(d_dQ)); CUDA_CHECK(cudaFree(d_dK)); CUDA_CHECK(cudaFree(d_dV));
    return 0;
}
