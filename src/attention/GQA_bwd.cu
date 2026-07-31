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
__global__ void convert_dq_accum_to_bf16_v5(const float * __restrict__ d_dq_accum, bf16 * __restrict__ d_dQ, long n) {
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

    // fp32 dQ accumulator scratch. Each query row receives a partial from every KV-tile
    // it attends; those partials race across blocks and must sum in fp32 before a single
    // bf16 round. CACHED across calls (static) so the benchmark isn't dominated by per-call
    // cudaMalloc/cudaFree; still zeroed every call since dQ atomic-accumulates.
    const long dqN = (long)B * Hq * S * D;
    static float* d_dq_accum = nullptr;
    static long   dq_cap     = 0;
    if (dqN > dq_cap) {
        if (d_dq_accum) CUDA_CHECK(cudaFree(d_dq_accum));
        CUDA_CHECK(cudaMalloc(&d_dq_accum, dqN * sizeof(float)));
        dq_cap = dqN;
    }
    CUDA_CHECK(cudaMemset(d_dq_accum, 0, dqN * sizeof(float)));

    constexpr dim3 BLOCK(128);
    dim3 GRID(B, Hkv, S / Bc);
    gqa_backward_v5_kv<Br,Bc,D><<<GRID, BLOCK>>>(
        tma_K_sw, tma_K_pl, tma_V_sw, tma_Q_pl, tma_dO_pl, tma_O_pl,
        d_LSE, d_dK, d_dV, d_dq_accum, B, Hq, Hkv, G, S, scale);

    const int convBlock = 256;
    const int convGrid  = (int)((dqN + convBlock - 1) / convBlock);
    convert_dq_accum_to_bf16_v5<<<convGrid, convBlock>>>(d_dq_accum, d_dQ, dqN);
    // d_dq_accum is cached (static) — intentionally not freed; reclaimed at process exit.
}

// ═════════════════════════════════════════════════════════════════════════════
// V6 — BANK-CONFLICT-FREE variant of V5 (identical math, padded smem layouts)
//                                                            [SM_90a only]
//
// V5's Nsight profile (docs/V5_analysis.md) is dominated by shared-memory bank
// conflicts: 17.9-way on loads (94.4% of wavefronts), tensor cores idle at 13.5%
// Compute SOL.  Every conflict has the SAME root cause — a shared row/atom stride
// that is a MULTIPLE of the 128-B (32 banks × 4 B) width, so the rows/core-matrices
// that a warp (or the wgmma operand fetch) touches together alias onto the same
// banks.  V6 fixes this purely by PADDING the storage stride of the regular-thread
// buffers so the stride is no longer 128-B-aligned; NO computed value changes, so
// V6 is bit-identical to V5 in the precision check.  (The TMA-loaded plain buffers
// sQ_pl/sdO_pl/sO_pl/sK_pl and the 128B-swizzled sK_sw/sV_sw are NOT padded — TMA
// writes them contiguously and the swizzle already de-aliases, respectively.)
//
// Three padded layouts:
//   1. sA_t/sB_t no-swizzle tiled operands — MN core-matrix stride (the wgmma SBO)
//      is (K/8)*128 B, a multiple of 128 B → all 8/16 MN core-matrices alias.
//      Pad the MN-block-row stride by PAD_V6=8 bf16 (16 B, the minimum that keeps
//      SBO 16-byte-encodable in the descriptor): SBO gains +16 B → SBO mod 128 = 16
//      ≠ 0, spreading the MN core-matrices onto distinct banks.  The padded SBO is
//      threaded into make_desc for BOTH operands (tiled_off_v6 and mn_off_v6 share
//      the same block layout, so LBO=128 and the +128-elem/k-step base advance are
//      unchanged — only the SBO field moves).
//   2. sS/sdP (float, row stride Bc=64 → 256 B): store_acc_smem writes smem[r0*64+c];
//      the 8 distinct r0 at fixed c collide (r0*64 mod 32 = 0).  Pad row stride to
//      65 (odd word stride, coprime with 32) → the 8 r0 hit 8 distinct banks.
//   3. sP (bf16, row stride Bc=64 → 128 B): the transposed fill_trans_A read
//      sP[k*64+mn] strides one full bank width per thread → 32-way.  Pad row stride
//      to 66 (even so word stride is exact; 66/2=33 coprime with 32) → conflict-free,
//      while the row-major fill_copy / elementwise reads stay contiguous.
// ═════════════════════════════════════════════════════════════════════════════

constexpr int PAD_V6        = 8;        // bf16 elems (16 B) between MN core-matrix rows
constexpr int SS_STRIDE_V6  = 64 + 1;   // sS / sdP float row stride (Bc + 1)
constexpr int SP_STRIDE_V6  = 64 + 2;   // sP bf16 row stride (Bc + 2, kept even)
constexpr int SMEM_TILED_V6 = 8320;     // max padded sA_t/sB_t = (128/8)*rowpad_v6(64)

// Padded MN-block-row stride (elements) for a K-contraction-width-K tiled operand.
__device__ __forceinline__ int      rowpad_v6(int K) { return (K >> 3) * 64 + PAD_V6; }
// Padded SBO (bytes) matching rowpad_v6, to feed make_desc.
__device__ __forceinline__ uint64_t sbo_pad_v6(int K) { return (uint64_t)(((K >> 3) * 64 + PAD_V6) * 2); }

// Padded no-swizzle K-major tiled offset (== tiled_off but MN-block-row stride padded).
__device__ __forceinline__ int tiled_off_v6(int mn, int k, int K) {
    return (mn >> 3) * rowpad_v6(K) + (k >> 3) * 64 + (mn & 7) * 8 + (k & 7);
}
// Padded Major::MN offset (== mn_off but MN-block-row stride padded; transposed intra-core).
__device__ __forceinline__ int mn_off_v6(int mn, int k, int K) {
    return (mn >> 3) * rowpad_v6(K) + (k >> 3) * 64 + (k & 7) * 8 + (mn & 7);
}

// V6 fills — same semantics as fill_copy / fill_trans / fill_trans_A but (a) write
// the PADDED tiled layout and (b) take an explicit src row stride so a padded source
// (sP, stride SP_STRIDE_V6) can be read as well as the unpadded TMA buffers (stride K/MN).
//   fill_copy_v6   : src [MN][src_stride] row-major -> dst_tiled(mn,k) = src[mn][k]
//   fill_trans_v6  : src [K][src_stride]  row-major -> dst_tiled(mn,k) = src[k][mn]
//   fill_trans_A_v6: src [K][src_stride]  row-major -> dst_mnoff(mn,k) = src[k][mn]
template<int MN, int K>
__device__ __forceinline__ void fill_copy_v6(bf16 *dst, const bf16 *src, int src_stride, int tid) {
    for (int i = tid; i < MN * K; i += 128) { int mn = i / K, k = i % K; dst[tiled_off_v6(mn, k, K)] = src[mn * src_stride + k]; }
}
template<int MN, int K>
__device__ __forceinline__ void fill_trans_v6(bf16 *dst, const bf16 *src, int src_stride, int tid) {
    for (int i = tid; i < MN * K; i += 128) { int mn = i / K, k = i % K; dst[tiled_off_v6(mn, k, K)] = src[k * src_stride + mn]; }
}
template<int MN, int K>
__device__ __forceinline__ void fill_trans_A_v6(bf16 *dst, const bf16 *src, int src_stride, int tid) {
    for (int i = tid; i < MN * K; i += 128) { int mn = i / K, k = i % K; dst[mn_off_v6(mn, k, K)] = src[k * src_stride + mn]; }
}

// S=Q·Kᵀ / dP=dO·Vᵀ — one 128B-swizzled operand + one PADDED no-swizzle operand
// (K=D=128 → padded SBO = sbo_pad_v6(128) = 2064 B).  Mirrors run_gemm_n64_sw but
// feeds the padded SBO for the no-swizzle side (the swizzled side is untouched).
template<bool A_swz>
__device__ __forceinline__ void run_gemm_n64_sw_v6(float acc[32], const bf16* swz, const bf16* nosw) {
    fence_proxy_async_shared();
    fence_operandN<32>(acc);
    wgmma_fence();
#pragma unroll
    for (int k = 0; k < 8; k++) {
        uint64_t dsw = make_desc_sw128_K(swz + (k >> 2) * 4096 + (k & 3) * 16); // atom=k/4, lk=k%4
        uint64_t dno = make_desc(nosw + 128 * k, sbo_pad_v6(128));              // padded tiled K=128
        if (A_swz) wgmma_m64n64k16(acc, dsw, dno);
        else       wgmma_m64n64k16(acc, dno, dsw);
    }
    wgmma_commit();
    wgmma_wait0();
    fence_operandN<32>(acc);
}

// store_acc_smem with an explicit (padded) row stride, decoupled from NCOL.
template<int NCOL, int ROWSTRIDE>
__device__ __forceinline__ void store_acc_smem_v6(const float *d, float *smem, int tid, float scl) {
    int w = tid >> 5, lane = tid & 31;
    int r0 = w * 16 + (lane >> 2), r1 = r0 + 8, cc = (lane & 3) * 2;
#pragma unroll
    for (int nt = 0; nt < NCOL / 8; nt++) {
        int c = nt * 8 + cc;
        smem[r0 * ROWSTRIDE + c + 0] = d[nt * 4 + 0] * scl;
        smem[r0 * ROWSTRIDE + c + 1] = d[nt * 4 + 1] * scl;
        smem[r1 * ROWSTRIDE + c + 0] = d[nt * 4 + 2] * scl;
        smem[r1 * ROWSTRIDE + c + 1] = d[nt * 4 + 3] * scl;
    }
}

// ── V6 — fused dQ + dK + dV, bank-conflict-free ──  Grid/threads identical to V5.
template<int Br, int Bc, int D>
__global__ void __launch_bounds__(128, 1)
gqa_backward_v6_kv(
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
    static_assert(Br == 64 && Bc == 64 && D == 128, "V6 requires Br=Bc=64, D=128");

    // Persistent operands (K-tile fixed for the block) — TMA/swizzle, NOT padded.
    __shared__ __align__(128) bf16 sK_sw[Bc * D];        // 2 swizzle atoms (S-B)
    __shared__ __align__(128) bf16 sK_pl[Bc * D];        // plain (dQ-B fill_trans)
    __shared__ __align__(128) bf16 sV_sw[Bc * D];        // 2 swizzle atoms (dP-B)
    // Loop-variant, double-buffered [stage] — TMA plain, NOT padded.
    __shared__ __align__(128) bf16 sQ_pl [2][Br * D];
    __shared__ __align__(128) bf16 sdO_pl[2][Br * D];
    __shared__ __align__(128) bf16 sO_pl [2][Br * D];
    // Compute scratch / intermediates — PADDED row strides (bank-conflict-free).
    __shared__ __align__(16)  float sS [Br * SS_STRIDE_V6];  // float, stride Bc+1
    __shared__ __align__(16)  float sdP[Br * SS_STRIDE_V6];  // float, stride Bc+1
    __shared__ __align__(16)  bf16  sP [Br * SP_STRIDE_V6];  // bf16,  stride Bc+2 (reused as sdS)
    __shared__                float sLSE[Br];
    __shared__                float sD  [Br];
    __shared__ __align__(128) bf16  sA_t[SMEM_TILED_V6];     // padded no-swizzle tiled scratch
    __shared__ __align__(128) bf16  sB_t[SMEM_TILED_V6];
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

        // D[r] = rowsum(dO · O)  (reads unpadded TMA buffers; unchanged from V5)
        if (tid < Br) {
            float acc = 0.f;
            for (int j = 0; j < D; j++)
                acc += __bfloat162float(sdO_pl[s][tid * D + j]) * __bfloat162float(sO_pl[s][tid * D + j]);
            sD[tid] = acc;
        }
        __syncthreads();

        // S = Q·Kᵀ·scale  (A=Q no-swizzle PADDED fill_copy, B=K swizzled)
        fill_copy_v6<Br, D>(sA_t, sQ_pl[s], D, tid);
        __syncthreads();
        { float acc[32]; zeroN<32>(acc); run_gemm_n64_sw_v6<false>(acc, sK_sw, sA_t); store_acc_smem_v6<Bc, SS_STRIDE_V6>(acc, sS, tid, scale); }
        __syncthreads();

        // P = exp(S - LSE), causal mask → sP  (padded strides)
        for (int i = tid; i < Br * Bc; i += 128) {
            int r = i / Bc, c = i % Bc;
            sP[r * SP_STRIDE_V6 + c] = (k_row0 + c > q_row0 + r) ? __float2bfloat16(0.f)
                                              : __float2bfloat16(expf(sS[r * SS_STRIDE_V6 + c] - sLSE[r]));
        }
        __syncthreads();

        // dV += Pᵀ·dO  (A=Pᵀ PADDED fill_trans_A from padded sP, B=dOᵀ PADDED fill_trans of plain dO)
        fill_trans_A_v6<Bc, Br>(sA_t, sP, SP_STRIDE_V6, tid); fill_trans_v6<D, Br>(sB_t, sdO_pl[s], D, tid);
        __syncthreads();
        run_gemm_n128_tA<Br / 16>(dv, sA_t, sB_t, sbo_pad_v6(Br));   // padded SBO
        __syncthreads();

        // dP = dO·Vᵀ  (A=dO no-swizzle PADDED fill_copy, B=V swizzled)
        fill_copy_v6<Br, D>(sA_t, sdO_pl[s], D, tid);
        __syncthreads();
        { float acc[32]; zeroN<32>(acc); run_gemm_n64_sw_v6<false>(acc, sV_sw, sA_t); store_acc_smem_v6<Bc, SS_STRIDE_V6>(acc, sdP, tid, 1.0f); }
        __syncthreads();

        // dS = P ⊙ (dP - D) → sP  (padded strides)
        for (int i = tid; i < Br * Bc; i += 128) {
            int r = i / Bc, c = i % Bc;
            int pidx = r * SP_STRIDE_V6 + c;
            sP[pidx] = __float2bfloat16(__bfloat162float(sP[pidx]) * (sdP[r * SS_STRIDE_V6 + c] - sD[r]));
        }
        __syncthreads();

        // dK += dSᵀ·Q  (A=dSᵀ PADDED fill_trans_A from padded sP, B=Qᵀ PADDED fill_trans of plain Q)
        fill_trans_A_v6<Bc, Br>(sA_t, sP, SP_STRIDE_V6, tid); fill_trans_v6<D, Br>(sB_t, sQ_pl[s], D, tid);
        __syncthreads();
        run_gemm_n128_tA<Br / 16>(dk, sA_t, sB_t, sbo_pad_v6(Br));
        __syncthreads();

        // dQ_tile = dS·K  (TRANSIENT; A=dS no-swizzle PADDED fill_copy from padded sP, B=Kᵀ
        // no-swizzle PADDED fill_trans of PLAIN K).  As in V5, the D=128 output is done as
        // TWO m64n64 halves (acc[32] each) to cap transient accumulator pressure at 32 regs
        // → 0 spills.  Half 1 reads B mn-blocks 8..15, whose padded base is 8*rowpad_v6(Bc)
        // (V5's 4096 → 4160), and both halves use the padded SBO.
        fill_copy_v6<Br, Bc>(sA_t, sP, SP_STRIDE_V6, tid); fill_trans_v6<D, Bc>(sB_t, sK_pl, D, tid);
        __syncthreads();
        { float acc[32]; zeroN<32>(acc);
          run_gemm_n64<Bc / 16>(acc, sA_t, sB_t, sbo_pad_v6(Bc));                          // padded SBO
          atomic_acc_global<64>(acc, d_dq_accum, lBaseOf(it) * D, D, 0,  tid, scale); }
        { float acc[32]; zeroN<32>(acc);
          run_gemm_n64<Bc / 16>(acc, sA_t, sB_t + 8 * rowpad_v6(Bc), sbo_pad_v6(Bc));      // mn-blocks 8..15
          atomic_acc_global<64>(acc, d_dq_accum, lBaseOf(it) * D, D, 64, tid, scale); }
        __syncthreads();
    }

    fence_operandN<64>(dv); fence_operandN<64>(dk);
    store_acc_global<D>(dv, d_dV, kvBase, D, tid, 1.0f);
    store_acc_global<D>(dk, d_dK, kvBase, D, tid, scale);
}

// ── V6 launcher — identical to V5 except it launches gqa_backward_v6_kv. ─────────
template<int Br, int Bc, int D>
void launch_gqa_backward_v6(
    const bf16 *d_Q, const bf16 *d_K, const bf16 *d_V, const bf16 *d_O,
    const bf16 *d_dO, const float *d_LSE,
    bf16 *d_dQ, bf16 *d_dK, bf16 *d_dV,
    int B, int Hq, int Hkv, int G, int S, float scale
) {
    static_assert(Br == 64 && Bc == 64 && D == 128, "V6 requires Br=Bc=64, D=128");

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

    const long dqN = (long)B * Hq * S * D;
    static float* d_dq_accum = nullptr;
    static long   dq_cap     = 0;
    if (dqN > dq_cap) {
        if (d_dq_accum) CUDA_CHECK(cudaFree(d_dq_accum));
        CUDA_CHECK(cudaMalloc(&d_dq_accum, dqN * sizeof(float)));
        dq_cap = dqN;
    }
    CUDA_CHECK(cudaMemset(d_dq_accum, 0, dqN * sizeof(float)));

    constexpr dim3 BLOCK(128);
    dim3 GRID(B, Hkv, S / Bc);
    gqa_backward_v6_kv<Br,Bc,D><<<GRID, BLOCK>>>(
        tma_K_sw, tma_K_pl, tma_V_sw, tma_Q_pl, tma_dO_pl, tma_O_pl,
        d_LSE, d_dK, d_dV, d_dq_accum, B, Hq, Hkv, G, S, scale);

    const int convBlock = 256;
    const int convGrid  = (int)((dqN + convBlock - 1) / convBlock);
    convert_dq_accum_to_bf16_v5<<<convGrid, convBlock>>>(d_dq_accum, d_dQ, dqN);
    // d_dq_accum is cached (static) — intentionally not freed; reclaimed at process exit.
}

// ═════════════════════════════════════════════════════════════════════════════
// V7 — SWIZZLED-TMA + wgmma Major::MN TRANSPOSE  (full transpose-buffer elimination)
//                                                            [SM_90a only]
//
// V6 padded away every *paddable* bank conflict, but its Nsight profile
// (docs/V6_analysis.md) is still stuck at a 15.1-way shared-LOAD conflict on 93% of
// wavefronts.  The root cause is UNPADDABLE: the three transposed `fill_trans` reads
// of the TMA-plain loaded operands feeding the transposed GEMMs —
//     dQ = dS·K   (B = Kᵀ,  fill_trans of plain sK_pl)
//     dV = Pᵀ·dO  (B = dOᵀ, fill_trans of plain sdO_pl)
//     dK = dSᵀ·Q  (B = Qᵀ,  fill_trans of plain sQ_pl)
// each a strided `src[k*stride + mn]` read = one 128-B bank line per thread → collide.
// Padding can't help (the read pattern, not the storage stride, is the aliaser).
//
// V7 removes the strided read ENTIRELY, exactly as the Blackwell V8–V10 arc did:
// load K / Q / dO via a 128-B-SWIZZLED TMA descriptor into a SINGLE buffer, and feed
// the transposed GEMM from those SAME bytes via the wgmma **Major::MN transpose
// immediate (trans-b = 1)** — the Hopper analogue of Blackwell's idesc `b_major` bit.
// No `fill_trans`, no separate plain duplicate. The conflict has no buffer to live in.
//
//   ┌── operand ──┬── non-transposed use (Major::K, trans=0) ─┬── transposed use (Major::MN, trans-b=1)
//   │ K  (sK_sw)  │ S  = Q·Kᵀ   B, contraction=D              │ dQ = dS·K   B, contraction=Bc
//   │ Q  (sQ_sw)  │ S  = Q·Kᵀ   A, contraction=D              │ dK = dSᵀ·Q  B, contraction=Br
//   │ dO (sdO_sw) │ dP = dO·Vᵀ  A, contraction=D              │ dV = Pᵀ·dO  B, contraction=Br
//   │ V  (sV_sw)  │ dP = dO·Vᵀ  B, contraction=D              │ (none)
//   └─────────────┴────────────────────────────────────────────┴────────────────────
// Both roles read the SAME physical [rows][D] D-contiguous swizzled buffer:
//   • Major::K  (trans=0): contiguous dim D IS the contraction → base +(k>>2)*4096
//     +(k&3)*16 elems, 8 k-steps (the V4–V6-verified make_desc_sw128_K path).
//   • Major::MN (trans-b=1): contiguous dim D IS the N output; contraction = the ROW
//     dim (Bc or Br). Base walks the rows: +k*1024 elems (= k*2048 B = 16 rows × 128 B)
//     per MMA-K step, 4 k-steps. SAME descriptor bits (SBO=1024, swz=B128) as Major::K.
//
// ── DERIVED Major::MN @ D=128 geometry (THE correctness risk; re-derived, not copied)
// D=128 bf16 = 256 B = TWO 128-B swizzle atoms wide in the N direction.  A single
// m64n128 MN-major read would span both atoms (the UNVALIDATED multi-atom case, see
// reference_tcgen05_swizzle_descriptor memory §"multi-atom … OPEN").  We AVOID it by
// splitting every transposed-B GEMM into TWO m64n64 halves of N=64 (= exactly one
// 128-B atom).  Each half is then byte-for-byte Blackwell's B300-CONFIRMED single-atom
// recipe (SBO=8·128=1024 B, LBO implicit, base +k·2048 B, b_major).  V6's dQ already
// split this way for register pressure; V7 splits dV/dK the same way (2× m64n64 into
// the dv[64]/dk[64] halves — bit-identical accumulator layout to one m64n128).
//   Per-half B base:  half0 = buf + 0     (N cols [0,64)   → K/Q/dO D-cols 0..63)
//                     half1 = buf + 4096  (N cols [64,128) → D-cols 64..127, atom 1)
//   Per-k-step:       += 1024 elems (2048 B), k = 0..3, contraction = 64 rows.
//   How Major::K @ D=128 differs: contiguous dim = K = full 128-wide D (8 k-steps of
//   +16 elems, atom switch every 4 steps at +4096); SBO field is unused/implicit on
//   the swizzled side. Major::MN reuses the identical descriptor bits — ONLY the base
//   advance (k·1024 elems along rows vs k·16 along cols) and trans-b (1 vs 0) change.
//
// ── V10 twist (dO double presence): D[r]=rowsum(dO·O) is a PLAIN elementwise read and
// cannot index swizzled bytes. So dO is loaded TWICE — swizzled sdO_sw (feeds dP-A and
// dV-B MMAs) AND plain sdO_pl (feeds ONLY the D-rowsum). O stays plain (sO_pl). This
// costs +1 tile of TMA bandwidth on dO per iter; no restructuring of D[r].
//
// ── Deleted vs V6:  sK_pl (K plain dup), sQ_pl→sQ_sw (Q now swizzled, no dup), and
// sB_t ENTIRELY (no loaded-operand fill_trans left — sA_t still stages the in-kernel
// dS/Pᵀ/dSᵀ intermediates via fill_copy_v6 / fill_trans_A_v6, unchanged).  Added:
// sdO_sw (the V10 double-load).  S and dP become BOTH-operand-swizzled (Q_sw·K_sw,
// dO_sw·V_sw) — their fill_copy of Q/dO is gone too.  Math is bit-identical to V6.
// ═════════════════════════════════════════════════════════════════════════════

// wgmma m64n64k16, trans-a=0, trans-b=1 (B read Major::MN).  For dQ = dS·K:
// A = dS (K-major, no-swizzle), B = K (Major::MN, swizzled).
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
// wgmma m64n64k16, trans-a=1, trans-b=1 (BOTH Major::MN).  For dV=Pᵀ·dO, dK=dSᵀ·Q:
// A = Pᵀ/dSᵀ (Major::MN, no-swizzle mn_off layout — same as V6 run_gemm_n128_tA's A),
// B = dO/Q (Major::MN, swizzled).
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

// Major::MN swizzled descriptor.  DERIVED as bit-identical to the Major::K descriptor
// make_desc_sw128_K (SBO=1024 B, LBO implicit=1, swizzle-field 1<<62 = B128) — under
// 128-B swizzle the HW ignores LBO and the same SBO strides the 8-row atom groups for
// BOTH orientations.  ONLY the base-pointer advance (caller: +k·1024 elems along rows)
// and the wgmma trans-b immediate (=1) distinguish the MN read from the K read.  If
// the H200 run shows a transposed-gradient error, LBO (0 vs 1024) is the first field
// to sweep here (Blackwell harness confirmed LBO=0 for the single-atom MN read).
__device__ __forceinline__ uint64_t make_desc_sw128_MN(const bf16* smem_ptr) {
    return make_desc_sw128_K(smem_ptr);
}

// S = Q·Kᵀ / dP = dO·Vᵀ — BOTH operands 128-B-swizzled, Major::K (trans 0,0).
// N = Bc = 64, contraction K = D = 128 → 8 k-steps.  A = Q_sw/dO_sw, B = K_sw/V_sw.
// Mirror of run_gemm_n64_sw_v6 but with the no-swizzle side promoted to a second
// swizzled descriptor (Q/dO now arrive swizzled instead of via fill_copy).
__device__ __forceinline__ void run_gemm_n64_sw2(float acc[32], const bf16* A_sw, const bf16* B_sw) {
    fence_proxy_async_shared();
    fence_operandN<32>(acc);
    wgmma_fence();
#pragma unroll
    for (int k = 0; k < 8; k++) {
        uint64_t dA = make_desc_sw128_K(A_sw + (k >> 2) * 4096 + (k & 3) * 16); // atom=k/4, lk=k%4
        uint64_t dB = make_desc_sw128_K(B_sw + (k >> 2) * 4096 + (k & 3) * 16);
        wgmma_m64n64k16(acc, dA, dB);   // trans 0,0
    }
    wgmma_commit();
    wgmma_wait0();
    fence_operandN<32>(acc);
}

// dQ half = dS·K, ONE N=64 half (one B swizzle atom).  A = dS no-swizzle K-major
// (sA_t, fill_copy_v6 from padded sP), B = K Major::MN swizzled single-atom.
// contraction K = Bc = 64 → 4 k-steps.  sbo_A = sbo_pad_v6(Bc) for the padded A tile.
__device__ __forceinline__ void run_gemm_dQ_half(float acc[32], const bf16* sA_t,
                                                  const bf16* K_sw_atom, uint64_t sbo_A) {
    fence_proxy_async_shared();       // orders the generic fill of dS(sA_t) → async wgmma read
    fence_operandN<32>(acc);
    wgmma_fence();
#pragma unroll
    for (int k = 0; k < 4; k++) {
        uint64_t dA = make_desc(sA_t + 128 * k, sbo_A);          // no-swizzle K-major, +128 elem/step
        uint64_t dB = make_desc_sw128_MN(K_sw_atom + k * 1024);  // swizzled Major::MN, +1024 elem/step
        wgmma_m64n64k16_tB(acc, dA, dB);                         // trans-a=0, trans-b=1
    }
    wgmma_commit();
    wgmma_wait0();
    fence_operandN<32>(acc);
}

// dV += Pᵀ·dO  /  dK += dSᵀ·Q — N = D = 128 done as TWO m64n64 halves into acc[64].
// A = Pᵀ/dSᵀ no-swizzle Major::MN (sA_t, fill_trans_A_v6 from padded sP; trans-a=1,
// identical to V6 run_gemm_n128_tA's A operand), B = dO/Q Major::MN swizzled.
// Half 0 → B atom at +0 → acc cols [0,64) = acc[0:32]; half 1 → B atom +4096 → acc[32:64].
// contraction K = Br = 64 → 4 k-steps.  sbo_A = sbo_pad_v6(Br).
__device__ __forceinline__ void run_gemm_dVdK(float acc[64], const bf16* sA_t,
                                              const bf16* B_sw, uint64_t sbo_A) {
    fence_proxy_async_shared();       // orders the generic fill of Pᵀ/dSᵀ(sA_t) → async wgmma read
    fence_operandN<64>(acc);
    wgmma_fence();
#pragma unroll
    for (int k = 0; k < 4; k++) {
        uint64_t dA  = make_desc(sA_t + 128 * k, sbo_A);          // no-swizzle Major::MN A, +128 elem
        uint64_t dB0 = make_desc_sw128_MN(B_sw + 0    + k * 1024); // N cols [0,64),  atom 0
        uint64_t dB1 = make_desc_sw128_MN(B_sw + 4096 + k * 1024); // N cols [64,128), atom 1
        wgmma_m64n64k16_tAtB(acc + 0,  dA, dB0);                  // → cols 0..63
        wgmma_m64n64k16_tAtB(acc + 32, dA, dB1);                  // → cols 64..127
    }
    wgmma_commit();
    wgmma_wait0();
    fence_operandN<64>(acc);
}

// ── V7 — fused dQ + dK + dV, transpose-buffer-free ──  Grid/threads identical to V6.
template<int Br, int Bc, int D>
__global__ void __launch_bounds__(128, 1)
gqa_backward_v7_kv(
    const __grid_constant__ CUtensorMap tma_K_sw,   // K 128B-swizzled (S-B Major::K + dQ-B Major::MN)
    const __grid_constant__ CUtensorMap tma_V_sw,   // V 128B-swizzled (dP-B Major::K)
    const __grid_constant__ CUtensorMap tma_Q_sw,   // Q 128B-swizzled (S-A Major::K + dK-B Major::MN)
    const __grid_constant__ CUtensorMap tma_dO_sw,  // dO 128B-swizzled (dP-A Major::K + dV-B Major::MN)
    const __grid_constant__ CUtensorMap tma_dO_pl,  // dO plain (D-rowsum ONLY — V10 double presence)
    const __grid_constant__ CUtensorMap tma_O_pl,   // O  plain (D-rowsum)
    const float * __restrict__ d_LSE,
    bf16 * __restrict__ d_dK, bf16 * __restrict__ d_dV, float * __restrict__ d_dq_accum,
    int B, int Hq, int Hkv, int G, int S, float scale
) {
    static_assert(Br == 64 && Bc == 64 && D == 128, "V7 requires Br=Bc=64, D=128");

    // Persistent operands (K-tile fixed) — swizzled, no plain K duplicate.
    __shared__ __align__(128) bf16 sK_sw[Bc * D];        // 2 swizzle atoms (S-B + dQ-B)
    __shared__ __align__(128) bf16 sV_sw[Bc * D];        // 2 swizzle atoms (dP-B)
    // Loop-variant, double-buffered [stage].
    __shared__ __align__(128) bf16 sQ_sw [2][Br * D];    // swizzled (S-A + dK-B)
    __shared__ __align__(128) bf16 sdO_sw[2][Br * D];    // swizzled (dP-A + dV-B)
    __shared__ __align__(128) bf16 sdO_pl[2][Br * D];    // plain (D-rowsum only)
    __shared__ __align__(128) bf16 sO_pl [2][Br * D];    // plain (D-rowsum)
    // Compute scratch / intermediates — PADDED (bank-conflict-free, unchanged from V6).
    __shared__ __align__(16)  float sS [Br * SS_STRIDE_V6];
    __shared__ __align__(16)  float sdP[Br * SS_STRIDE_V6];
    __shared__ __align__(16)  bf16  sP [Br * SP_STRIDE_V6];  // reused as sdS
    __shared__                float sLSE[Br];
    __shared__                float sD  [Br];
    __shared__ __align__(128) bf16  sA_t[SMEM_TILED_V6];     // in-kernel dS/Pᵀ/dSᵀ ONLY (sB_t deleted)
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

    // ── Persistent loads: K swizzled (2 atoms) + V swizzled (2 atoms).  No K plain.
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

    float dv[64]; zeroN<64>(dv);   // PERSISTENT dV[Bc×D] accumulator (owned by this block)
    float dk[64]; zeroN<64>(dk);   // PERSISTENT dK[Bc×D] accumulator (owned by this block)

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

    // Prologue: TMA iter 0 → stage 0.  Q swizzled (2 atoms), dO swizzled (2 atoms),
    // dO plain (1 tile, D-rowsum), O plain (1 tile).  bytes = 4·bytesTile.
    uint32_t par[2] = {0, 0};
    if (nIter > 0 && tid == 0) {
        const uint32_t r0 = qFlatRowOf(0);
        mbar_expect_tx_v4(&mbar[0], bytesTile * 4);
        tma_load_2d_v4(&tma_Q_sw,  sQ_sw [0],            &mbar[0], 0,  r0);
        tma_load_2d_v4(&tma_Q_sw,  sQ_sw [0] + 64 * 64,  &mbar[0], 64, r0);
        tma_load_2d_v4(&tma_dO_sw, sdO_sw[0],            &mbar[0], 0,  r0);
        tma_load_2d_v4(&tma_dO_sw, sdO_sw[0] + 64 * 64,  &mbar[0], 64, r0);
        tma_load_2d_v4(&tma_dO_pl, sdO_pl[0],            &mbar[0], 0,  r0);
        tma_load_2d_v4(&tma_O_pl,  sO_pl [0],            &mbar[0], 0,  r0);
    }

    for (int it = 0; it < nIter; it++) {
        const int s = it & 1;
        const int q_row0 = (qc0 + (it % perG)) * Br;
        // Prefetch NEXT iter → other stage.
        if (it + 1 < nIter && tid == 0) {
            const uint32_t nr = qFlatRowOf(it + 1);
            const int sn = s ^ 1;
            mbar_expect_tx_v4(&mbar[sn], bytesTile * 4);
            tma_load_2d_v4(&tma_Q_sw,  sQ_sw [sn],           &mbar[sn], 0,  nr);
            tma_load_2d_v4(&tma_Q_sw,  sQ_sw [sn] + 64 * 64, &mbar[sn], 64, nr);
            tma_load_2d_v4(&tma_dO_sw, sdO_sw[sn],           &mbar[sn], 0,  nr);
            tma_load_2d_v4(&tma_dO_sw, sdO_sw[sn] + 64 * 64, &mbar[sn], 64, nr);
            tma_load_2d_v4(&tma_dO_pl, sdO_pl[sn],           &mbar[sn], 0,  nr);
            tma_load_2d_v4(&tma_O_pl,  sO_pl [sn],           &mbar[sn], 0,  nr);
        }
        mbar_wait_v4(&mbar[s], par[s]); par[s] ^= 1;
        if (tid < Br) sLSE[tid] = d_LSE[lBaseOf(it) + tid];
        __syncthreads();

        // D[r] = rowsum(dO · O)  — PLAIN buffers (swizzled bytes can't be plain-indexed).
        if (tid < Br) {
            float acc = 0.f;
            for (int j = 0; j < D; j++)
                acc += __bfloat162float(sdO_pl[s][tid * D + j]) * __bfloat162float(sO_pl[s][tid * D + j]);
            sD[tid] = acc;
        }
        __syncthreads();

        // S = Q·Kᵀ·scale  (A=Q swizzled Major::K, B=K swizzled Major::K — no fill_copy).
        { float acc[32]; zeroN<32>(acc); run_gemm_n64_sw2(acc, sQ_sw[s], sK_sw); store_acc_smem_v6<Bc, SS_STRIDE_V6>(acc, sS, tid, scale); }
        __syncthreads();

        // P = exp(S - LSE), causal mask → sP  (padded strides).
        for (int i = tid; i < Br * Bc; i += 128) {
            int r = i / Bc, c = i % Bc;
            sP[r * SP_STRIDE_V6 + c] = (k_row0 + c > q_row0 + r) ? __float2bfloat16(0.f)
                                              : __float2bfloat16(expf(sS[r * SS_STRIDE_V6 + c] - sLSE[r]));
        }
        __syncthreads();

        // dV += Pᵀ·dO  (A=Pᵀ padded fill_trans_A → sA_t, B=dO swizzled Major::MN).
        fill_trans_A_v6<Bc, Br>(sA_t, sP, SP_STRIDE_V6, tid);
        __syncthreads();
        run_gemm_dVdK(dv, sA_t, sdO_sw[s], sbo_pad_v6(Br));
        __syncthreads();

        // dP = dO·Vᵀ  (A=dO swizzled Major::K, B=V swizzled Major::K — no fill_copy).
        { float acc[32]; zeroN<32>(acc); run_gemm_n64_sw2(acc, sdO_sw[s], sV_sw); store_acc_smem_v6<Bc, SS_STRIDE_V6>(acc, sdP, tid, 1.0f); }
        __syncthreads();

        // dS = P ⊙ (dP - D) → sP  (padded strides).
        for (int i = tid; i < Br * Bc; i += 128) {
            int r = i / Bc, c = i % Bc;
            int pidx = r * SP_STRIDE_V6 + c;
            sP[pidx] = __float2bfloat16(__bfloat162float(sP[pidx]) * (sdP[r * SS_STRIDE_V6 + c] - sD[r]));
        }
        __syncthreads();

        // dK += dSᵀ·Q  (A=dSᵀ padded fill_trans_A → sA_t, B=Q swizzled Major::MN).
        fill_trans_A_v6<Bc, Br>(sA_t, sP, SP_STRIDE_V6, tid);
        __syncthreads();
        run_gemm_dVdK(dk, sA_t, sQ_sw[s], sbo_pad_v6(Br));
        __syncthreads();

        // dQ_tile = dS·K  (TRANSIENT; A=dS padded fill_copy → sA_t, B=K swizzled Major::MN).
        // Two N=64 halves (single-atom MN read each) → fp32 atomic flush, as in V6.
        fill_copy_v6<Br, Bc>(sA_t, sP, SP_STRIDE_V6, tid);
        __syncthreads();
        { float acc[32]; zeroN<32>(acc);
          run_gemm_dQ_half(acc, sA_t, sK_sw,        sbo_pad_v6(Bc));   // N cols [0,64),  K atom 0
          atomic_acc_global<64>(acc, d_dq_accum, lBaseOf(it) * D, D, 0,  tid, scale); }
        { float acc[32]; zeroN<32>(acc);
          run_gemm_dQ_half(acc, sA_t, sK_sw + 4096, sbo_pad_v6(Bc));   // N cols [64,128), K atom 1
          atomic_acc_global<64>(acc, d_dq_accum, lBaseOf(it) * D, D, 64, tid, scale); }
        __syncthreads();
    }

    fence_operandN<64>(dv); fence_operandN<64>(dk);
    store_acc_global<D>(dv, d_dV, kvBase, D, tid, 1.0f);
    store_acc_global<D>(dk, d_dK, kvBase, D, tid, scale);
}

// ── V7 launcher — like V6 but K/Q/dO all swizzled (K plain + Q plain deleted; dO
// double-loaded swizzled + plain).  Same fp32 dQ-scratch alloc/zero/convert.
template<int Br, int Bc, int D>
void launch_gqa_backward_v7(
    const bf16 *d_Q, const bf16 *d_K, const bf16 *d_V, const bf16 *d_O,
    const bf16 *d_dO, const float *d_LSE,
    bf16 *d_dQ, bf16 *d_dK, bf16 *d_dV,
    int B, int Hq, int Hkv, int G, int S, float scale
) {
    static_assert(Br == 64 && Bc == 64 && D == 128, "V7 requires Br=Bc=64, D=128");

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
    CUtensorMap tma_V_sw  = make_tma_sw128(d_V,  Rkv, Bc);
    CUtensorMap tma_Q_sw  = make_tma_sw128(d_Q,  Rq,  Br);
    CUtensorMap tma_dO_sw = make_tma_sw128(d_dO, Rq,  Br);
    CUtensorMap tma_dO_pl = make_tma_plain(d_dO, Rq,  Br);
    CUtensorMap tma_O_pl  = make_tma_plain(d_O,  Rq,  Br);

    const long dqN = (long)B * Hq * S * D;
    static float* d_dq_accum = nullptr;
    static long   dq_cap     = 0;
    if (dqN > dq_cap) {
        if (d_dq_accum) CUDA_CHECK(cudaFree(d_dq_accum));
        CUDA_CHECK(cudaMalloc(&d_dq_accum, dqN * sizeof(float)));
        dq_cap = dqN;
    }
    CUDA_CHECK(cudaMemset(d_dq_accum, 0, dqN * sizeof(float)));

    constexpr dim3 BLOCK(128);
    dim3 GRID(B, Hkv, S / Bc);
    gqa_backward_v7_kv<Br,Bc,D><<<GRID, BLOCK>>>(
        tma_K_sw, tma_V_sw, tma_Q_sw, tma_dO_sw, tma_dO_pl, tma_O_pl,
        d_LSE, d_dK, d_dV, d_dq_accum, B, Hq, Hkv, G, S, scale);

    const int convBlock = 256;
    const int convGrid  = (int)((dqN + convBlock - 1) / convBlock);
    convert_dq_accum_to_bf16_v5<<<convGrid, convBlock>>>(d_dq_accum, d_dQ, dqN);
    // d_dq_accum is cached (static) — intentionally not freed; reclaimed at process exit.
}

// ── V8 — fused dQ + dK + dV.  SHIPS: the PROVEN V7 no-swizzle A-operand GEMMs (bit-
// identical to V7) PLUS the Blackwell dKdV_v10 warp-per-row shuffle reduction for
// D[r]=rowsum(dO·O), which removes V7's 32-way same-bank aliasing on the plain sdO/sO
// reads (the D-rowsum residual).
// Historical note: an attempted 128B-swizzled sA_t A-operand conflict fix was proven
// INVALID on Hopper wgmma — a swizzled-Major::K A operand paired with a Major::MN (trans-b=1)
// B is an unsupported operand combination (finite-wrong output, HW-confirmed via -DV8_DEBUG).
// The swizzle scaffold + self-test were removed once the finding was recorded.
template<int Br, int Bc, int D>
__global__ void __launch_bounds__(128, 1)
gqa_backward_v8_kv(
    const __grid_constant__ CUtensorMap tma_K_sw,   // K 128B-swizzled (S-B Major::K + dQ-B Major::MN)
    const __grid_constant__ CUtensorMap tma_V_sw,   // V 128B-swizzled (dP-B Major::K)
    const __grid_constant__ CUtensorMap tma_Q_sw,   // Q 128B-swizzled (S-A Major::K + dK-B Major::MN)
    const __grid_constant__ CUtensorMap tma_dO_sw,  // dO 128B-swizzled (dP-A Major::K + dV-B Major::MN)
    const __grid_constant__ CUtensorMap tma_dO_pl,  // dO plain (D-rowsum ONLY — V10 double presence)
    const __grid_constant__ CUtensorMap tma_O_pl,   // O  plain (D-rowsum)
    const float * __restrict__ d_LSE,
    bf16 * __restrict__ d_dK, bf16 * __restrict__ d_dV, float * __restrict__ d_dq_accum,
    int B, int Hq, int Hkv, int G, int S, float scale
) {
    static_assert(Br == 64 && Bc == 64 && D == 128, "V8 requires Br=Bc=64, D=128");

    // Persistent operands (K-tile fixed) — swizzled, no plain K duplicate.
    __shared__ __align__(128) bf16 sK_sw[Bc * D];        // 2 swizzle atoms (S-B + dQ-B)
    __shared__ __align__(128) bf16 sV_sw[Bc * D];        // 2 swizzle atoms (dP-B)
    // Loop-variant, double-buffered [stage].
    __shared__ __align__(128) bf16 sQ_sw [2][Br * D];    // swizzled (S-A + dK-B)
    __shared__ __align__(128) bf16 sdO_sw[2][Br * D];    // swizzled (dP-A + dV-B)
    __shared__ __align__(128) bf16 sdO_pl[2][Br * D];    // plain (D-rowsum only)
    __shared__ __align__(128) bf16 sO_pl [2][Br * D];    // plain (D-rowsum)
    // Compute scratch / intermediates.
    __shared__ __align__(16)  float sS [Br * SS_STRIDE_V6];
    __shared__ __align__(16)  float sdP[Br * SS_STRIDE_V6];
    __shared__ __align__(16)  bf16  sP [Br * SP_STRIDE_V6];  // reused as sdS (padded, conflict-free)
    __shared__                float sLSE[Br];
    __shared__                float sD  [Br];
    // A-operand staging for the in-kernel dS/Pᵀ/dSᵀ intermediates.  V8 ships the
    // PROVEN no-swizzle padded layout (bit-identical to V7).
    __shared__ __align__(128) bf16  sA_t[SMEM_TILED_V6];     // no-swizzle padded (V7 layout)
    __shared__ __align__(8)   uint64_t mbar_p;
    __shared__ __align__(8)   uint64_t mbar[2];

    const int tid = threadIdx.x;
    const int warpId = tid >> 5, lane = tid & 31;
    const int b = blockIdx.x, hkv = blockIdx.y, k_tile = blockIdx.z;
    const int k_row0 = k_tile * Bc;
    const int nQTiles = S / Br;

    const long     kvBase     = ((long)(b * Hkv + hkv) * S + k_row0) * D;
    const uint32_t kvFlatRow  = (uint32_t)((b * Hkv + hkv) * S + k_row0);
    const uint32_t bytesTile  = (uint32_t)(Br * D * sizeof(bf16));   // 16384
    const uint32_t bytesAtom  = (uint32_t)(Bc * 64 * sizeof(bf16));  // 8192

    if (tid == 0) { mbar_init_v4(&mbar_p, 1); mbar_init_v4(&mbar[0], 1); mbar_init_v4(&mbar[1], 1); }
    __syncthreads();

    // ── Persistent loads: K swizzled (2 atoms) + V swizzled (2 atoms).  No K plain.
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

    float dv[64]; zeroN<64>(dv);   // PERSISTENT dV[Bc×D] accumulator (owned by this block)
    float dk[64]; zeroN<64>(dk);   // PERSISTENT dK[Bc×D] accumulator (owned by this block)

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

    // Prologue: TMA iter 0 → stage 0 (Q sw 2 atoms, dO sw 2 atoms, dO plain, O plain).
    uint32_t par[2] = {0, 0};
    if (nIter > 0 && tid == 0) {
        const uint32_t r0 = qFlatRowOf(0);
        mbar_expect_tx_v4(&mbar[0], bytesTile * 4);
        tma_load_2d_v4(&tma_Q_sw,  sQ_sw [0],            &mbar[0], 0,  r0);
        tma_load_2d_v4(&tma_Q_sw,  sQ_sw [0] + 64 * 64,  &mbar[0], 64, r0);
        tma_load_2d_v4(&tma_dO_sw, sdO_sw[0],            &mbar[0], 0,  r0);
        tma_load_2d_v4(&tma_dO_sw, sdO_sw[0] + 64 * 64,  &mbar[0], 64, r0);
        tma_load_2d_v4(&tma_dO_pl, sdO_pl[0],            &mbar[0], 0,  r0);
        tma_load_2d_v4(&tma_O_pl,  sO_pl [0],            &mbar[0], 0,  r0);
    }

    for (int it = 0; it < nIter; it++) {
        const int s = it & 1;
        const int q_row0 = (qc0 + (it % perG)) * Br;
        // Prefetch NEXT iter → other stage.
        if (it + 1 < nIter && tid == 0) {
            const uint32_t nr = qFlatRowOf(it + 1);
            const int sn = s ^ 1;
            mbar_expect_tx_v4(&mbar[sn], bytesTile * 4);
            tma_load_2d_v4(&tma_Q_sw,  sQ_sw [sn],           &mbar[sn], 0,  nr);
            tma_load_2d_v4(&tma_Q_sw,  sQ_sw [sn] + 64 * 64, &mbar[sn], 64, nr);
            tma_load_2d_v4(&tma_dO_sw, sdO_sw[sn],           &mbar[sn], 0,  nr);
            tma_load_2d_v4(&tma_dO_sw, sdO_sw[sn] + 64 * 64, &mbar[sn], 64, nr);
            tma_load_2d_v4(&tma_dO_pl, sdO_pl[sn],           &mbar[sn], 0,  nr);
            tma_load_2d_v4(&tma_O_pl,  sO_pl [sn],           &mbar[sn], 0,  nr);
        }
        mbar_wait_v4(&mbar[s], par[s]); par[s] ^= 1;
        if (tid < Br) sLSE[tid] = d_LSE[lBaseOf(it) + tid];
        __syncthreads();

        // D[r] = rowsum(dO · O) — PLAIN buffers, warp-per-row coalesced + shuffle reduce
        // (Blackwell dKdV_v10; replaces V7's fixed-j / stride-D 32-way plain read).
        for (int row = warpId; row < Br; row += 4) {
            float partial = 0.f;
            for (int j = lane; j < D; j += 32)
                partial += __bfloat162float(sdO_pl[s][row * D + j]) * __bfloat162float(sO_pl[s][row * D + j]);
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                partial += __shfl_down_sync(0xFFFFFFFFu, partial, off);
            if (lane == 0) sD[row] = partial;
        }
        __syncthreads();

        // S = Q·Kᵀ·scale  (A=Q swizzled Major::K, B=K swizzled Major::K — UNCHANGED from V7).
        { float acc[32]; zeroN<32>(acc); run_gemm_n64_sw2(acc, sQ_sw[s], sK_sw); store_acc_smem_v6<Bc, SS_STRIDE_V6>(acc, sS, tid, scale); }
        __syncthreads();

        // P = exp(S - LSE), causal mask → sP  (padded strides).
        for (int i = tid; i < Br * Bc; i += 128) {
            int r = i / Bc, c = i % Bc;
            sP[r * SP_STRIDE_V6 + c] = (k_row0 + c > q_row0 + r) ? __float2bfloat16(0.f)
                                              : __float2bfloat16(expf(sS[r * SS_STRIDE_V6 + c] - sLSE[r]));
        }
        __syncthreads();

        // dV += Pᵀ·dO  (A=Pᵀ no-swizzle Major::MN via fill_trans_A_v6 → sA_t, B=dO
        // swizzled Major::MN — PROVEN V7 path).
        fill_trans_A_v6<Bc, Br>(sA_t, sP, SP_STRIDE_V6, tid);
        __syncthreads();
        run_gemm_dVdK(dv, sA_t, sdO_sw[s], sbo_pad_v6(Br));
        __syncthreads();

        // dP = dO·Vᵀ  (A=dO swizzled Major::K, B=V swizzled Major::K — UNCHANGED from V7).
        { float acc[32]; zeroN<32>(acc); run_gemm_n64_sw2(acc, sdO_sw[s], sV_sw); store_acc_smem_v6<Bc, SS_STRIDE_V6>(acc, sdP, tid, 1.0f); }
        __syncthreads();

        // dS = P ⊙ (dP - D) → sP  (padded strides).
        for (int i = tid; i < Br * Bc; i += 128) {
            int r = i / Bc, c = i % Bc;
            int pidx = r * SP_STRIDE_V6 + c;
            sP[pidx] = __float2bfloat16(__bfloat162float(sP[pidx]) * (sdP[r * SS_STRIDE_V6 + c] - sD[r]));
        }
        __syncthreads();

        // dK += dSᵀ·Q  (A=dSᵀ no-swizzle Major::MN via fill_trans_A_v6 → sA_t, B=Q
        // swizzled Major::MN — PROVEN V7 path).
        fill_trans_A_v6<Bc, Br>(sA_t, sP, SP_STRIDE_V6, tid);
        __syncthreads();
        run_gemm_dVdK(dk, sA_t, sQ_sw[s], sbo_pad_v6(Br));
        __syncthreads();

        // dQ_tile = dS·K  (TRANSIENT; A=dS no-swizzle Major::K via fill_copy_v6 → sA_t,
        // B=K swizzled Major::MN — PROVEN V7 path).  Two N=64 halves → fp32 atomic flush.
        fill_copy_v6<Br, Bc>(sA_t, sP, SP_STRIDE_V6, tid);
        __syncthreads();
        { float acc[32]; zeroN<32>(acc);
          run_gemm_dQ_half(acc, sA_t, sK_sw,        sbo_pad_v6(Bc));   // N cols [0,64),  K atom 0
          atomic_acc_global<64>(acc, d_dq_accum, lBaseOf(it) * D, D, 0,  tid, scale); }
        { float acc[32]; zeroN<32>(acc);
          run_gemm_dQ_half(acc, sA_t, sK_sw + 4096, sbo_pad_v6(Bc));   // N cols [64,128), K atom 1
          atomic_acc_global<64>(acc, d_dq_accum, lBaseOf(it) * D, D, 64, tid, scale); }
        __syncthreads();
    }

    fence_operandN<64>(dv); fence_operandN<64>(dk);
    store_acc_global<D>(dv, d_dV, kvBase, D, tid, 1.0f);
    store_acc_global<D>(dk, d_dK, kvBase, D, tid, scale);
}

// ── V8 launcher — identical to V7 (same swizzled/plain TMA descriptors, same fp32 dQ
// scratch); only the launched kernel differs.
template<int Br, int Bc, int D>
void launch_gqa_backward_v8(
    const bf16 *d_Q, const bf16 *d_K, const bf16 *d_V, const bf16 *d_O,
    const bf16 *d_dO, const float *d_LSE,
    bf16 *d_dQ, bf16 *d_dK, bf16 *d_dV,
    int B, int Hq, int Hkv, int G, int S, float scale
) {
    static_assert(Br == 64 && Bc == 64 && D == 128, "V8 requires Br=Bc=64, D=128");

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
    CUtensorMap tma_V_sw  = make_tma_sw128(d_V,  Rkv, Bc);
    CUtensorMap tma_Q_sw  = make_tma_sw128(d_Q,  Rq,  Br);
    CUtensorMap tma_dO_sw = make_tma_sw128(d_dO, Rq,  Br);
    CUtensorMap tma_dO_pl = make_tma_plain(d_dO, Rq,  Br);
    CUtensorMap tma_O_pl  = make_tma_plain(d_O,  Rq,  Br);

    const long dqN = (long)B * Hq * S * D;
    static float* d_dq_accum = nullptr;
    static long   dq_cap     = 0;
    if (dqN > dq_cap) {
        if (d_dq_accum) CUDA_CHECK(cudaFree(d_dq_accum));
        CUDA_CHECK(cudaMalloc(&d_dq_accum, dqN * sizeof(float)));
        dq_cap = dqN;
    }
    CUDA_CHECK(cudaMemset(d_dq_accum, 0, dqN * sizeof(float)));

    constexpr dim3 BLOCK(128);
    dim3 GRID(B, Hkv, S / Bc);
    gqa_backward_v8_kv<Br,Bc,D><<<GRID, BLOCK>>>(
        tma_K_sw, tma_V_sw, tma_Q_sw, tma_dO_sw, tma_dO_pl, tma_O_pl,
        d_LSE, d_dK, d_dV, d_dq_accum, B, Hq, Hkv, G, S, scale);

    const int convBlock = 256;
    const int convGrid  = (int)((dqN + convBlock - 1) / convBlock);
    convert_dq_accum_to_bf16_v5<<<convGrid, convBlock>>>(d_dq_accum, d_dQ, dqN);
    // d_dq_accum is cached (static) — intentionally not freed; reclaimed at process exit.
}

// ═════════════════════════════════════════════════════════════════════════════
// V9 — OCCUPANCY via a SECOND WARPGROUP (Route B).  Same 222,744 B smem, same
// pipeline/double-buffer/descriptor machinery as V8 — the ONLY structural change
// is the block grows from 128 threads (1 warpgroup, 4 warps) to 256 threads
// (2 warpgroups, 8 warps).  smem is unchanged, so the block still maps 1-per-SM
// (222.7 KB > half of the 227 KB opt-in cap → 2 blocks/SM impossible without
// wrecking the pipeline), but occupancy rises 6.25% → 12.5% (8 warps / 64).
//
// WHY this helps (V8 profile: pure latency-bound, no-eligible 77.4%, 1.0 warp/
// scheduler): each SM has 4 schedulers; 4 warps = 1 warp/scheduler, so any stall
// (wgmma.wait_group, mbar.wait, dependent expf chain) idles the scheduler.  8
// warps = 2 warps/scheduler → the sibling warpgroup can issue while the other
// stalls.  On top of pure latency-hiding, the two warpgroups do GENUINELY
// PARALLEL tensor work: V8's 5 GEMMs are fully serial; V9 runs the two
// independent m64n64 GEMMs (S=Q·Kᵀ ∥ dP=dO·Vᵀ) concurrently and splits each
// D=128 output GEMM (dV, dK, dQ) into its two m64n64 column-halves — one per
// warpgroup — so per-iteration tensor issue roughly halves in wall time.
//
// wgmma is a WARPGROUP-collective instruction (PTX ISA 9.7.14: wgmma.mma_async.
// sync.aligned — all 128 threads of the issuing warpgroup participate; the
// commit_group / wait_group retire that warpgroup's own async group).  Two
// distinct warpgroups therefore each drive their own independent wgmma pipeline
// and accumulator registers.  The `if (wg==0) … else …` around the GEMM calls is
// warpgroup-UNIFORM (wg = tid>>7), so `.aligned` is satisfied.  The wgmma
// accumulator→row map is warpgroup-relative (warp (tid&127)>>5 owns rows
// [16w,16w+16)), so every register↔memory helper is fed the warpgroup-local
// tid `wtid = tid & 127`, not the block tid.
//
// Correctness invariants preserved verbatim from V8:
//   • fence.proxy.async.shared::cta before every wgmma (generic fills → async
//     reads) — issued per-warpgroup inside run_gemm_*, harmless idempotent CTA
//     fence; the preceding __syncthreads(256) already drained all 256 threads'
//     generic writes CTA-wide.
//   • fence_operandN brackets every async accumulator region.
//   • All phase barriers are block-scope __syncthreads() (all 256 threads reach
//     each one — no partial-CTA named barriers, so no bar.sync count hazards).
//   • Persistent dv/dk split by COLUMN: WG0 owns dV/dK cols [0,64), WG1 owns
//     [64,128).  Disjoint global regions → no atomics (unchanged from V8's
//     block-owns-KV-tile invariant); dQ still fp32-atomic-accumulated (reduces
//     over the KEY axis, not owned by one block).
// ═════════════════════════════════════════════════════════════════════════════

// store_acc_global with a global-column offset (for the two m64n64 halves of a
// D=128 output tile: WG0 → coloff 0, WG1 → coloff 64).  `tid` is warpgroup-local.
template<int NCOL>
__device__ __forceinline__ void store_acc_global_col(const float *d, bf16 *g, long base, int D, int coloff, int tid, float scl) {
    int w = tid >> 5, lane = tid & 31;
    int r0 = w * 16 + (lane >> 2), r1 = r0 + 8, cc = (lane & 3) * 2;
#pragma unroll
    for (int nt = 0; nt < NCOL / 8; nt++) {
        int c = coloff + nt * 8 + cc;
        g[base + (long)r0 * D + c + 0] = __float2bfloat16(d[nt * 4 + 0] * scl);
        g[base + (long)r0 * D + c + 1] = __float2bfloat16(d[nt * 4 + 1] * scl);
        g[base + (long)r1 * D + c + 0] = __float2bfloat16(d[nt * 4 + 2] * scl);
        g[base + (long)r1 * D + c + 1] = __float2bfloat16(d[nt * 4 + 3] * scl);
    }
}

// One m64n64 HALF of dV=Pᵀ·dO / dK=dSᵀ·Q (V8's run_gemm_dVdK does both halves;
// here each warpgroup issues one, into its own persistent acc[32]).  A = Pᵀ/dSᵀ
// no-swizzle Major::MN (sA_t, fill_trans_A_v9; trans-a=1), B = dO/Q Major::MN
// swizzled single atom (trans-b=1).  Caller passes B_sw_half = B_sw + wg*4096
// (WG0 atom0 → cols [0,64); WG1 atom1 → cols [64,128)).  K = Br = 64 → 4 k-steps.
__device__ __forceinline__ void run_gemm_dVdK_half(float acc[32], const bf16* sA_t,
                                                   const bf16* B_sw_half, uint64_t sbo_A) {
    fence_proxy_async_shared();       // orders the generic fill of Pᵀ/dSᵀ(sA_t) → async wgmma read
    fence_operandN<32>(acc);
    wgmma_fence();
#pragma unroll
    for (int k = 0; k < 4; k++) {
        uint64_t dA = make_desc(sA_t + 128 * k, sbo_A);            // no-swizzle Major::MN A, +128 elem
        uint64_t dB = make_desc_sw128_MN(B_sw_half + k * 1024);    // swizzled Major::MN, +1024 elem
        wgmma_m64n64k16_tAtB(acc, dA, dB);
    }
    wgmma_commit();
    wgmma_wait0();
    fence_operandN<32>(acc);
}

// blockDim.x-strided variants of fill_copy_v6 / fill_trans_A_v6 (V6/V7/V8 hardcode
// a 128-thread stride).  Identical padded-tiled scatter formulae — the index map
// mn_off_v6 / tiled_off_v6 is a bijection over [0,MN*K) independent of how many
// threads cover the range, so widening the stride to blockDim.x (256) is exact.
template<int MN, int K>
__device__ __forceinline__ void fill_copy_v9(bf16 *dst, const bf16 *src, int src_stride, int tid) {
    for (int i = tid; i < MN * K; i += blockDim.x) { int mn = i / K, k = i % K; dst[tiled_off_v6(mn, k, K)] = src[mn * src_stride + k]; }
}
template<int MN, int K>
__device__ __forceinline__ void fill_trans_A_v9(bf16 *dst, const bf16 *src, int src_stride, int tid) {
    for (int i = tid; i < MN * K; i += blockDim.x) { int mn = i / K, k = i % K; dst[mn_off_v6(mn, k, K)] = src[k * src_stride + mn]; }
}

// ── V9 kernel — byte-identical smem to V8, 256 threads (2 warpgroups) ─────────
template<int Br, int Bc, int D>
__global__ void __launch_bounds__(256, 1)
gqa_backward_v9_kv(
    const __grid_constant__ CUtensorMap tma_K_sw,   // K 128B-swizzled (S-B Major::K + dQ-B Major::MN)
    const __grid_constant__ CUtensorMap tma_V_sw,   // V 128B-swizzled (dP-B Major::K)
    const __grid_constant__ CUtensorMap tma_Q_sw,   // Q 128B-swizzled (S-A Major::K + dK-B Major::MN)
    const __grid_constant__ CUtensorMap tma_dO_sw,  // dO 128B-swizzled (dP-A Major::K + dV-B Major::MN)
    const __grid_constant__ CUtensorMap tma_dO_pl,  // dO plain (D-rowsum ONLY)
    const __grid_constant__ CUtensorMap tma_O_pl,   // O  plain (D-rowsum)
    const float * __restrict__ d_LSE,
    bf16 * __restrict__ d_dK, bf16 * __restrict__ d_dV, float * __restrict__ d_dq_accum,
    int B, int Hq, int Hkv, int G, int S, float scale
) {
    static_assert(Br == 64 && Bc == 64 && D == 128, "V9 requires Br=Bc=64, D=128");

    // ── smem: BYTE-IDENTICAL to V8 (single sA_t shared read-only across both WGs).
    __shared__ __align__(128) bf16 sK_sw[Bc * D];
    __shared__ __align__(128) bf16 sV_sw[Bc * D];
    __shared__ __align__(128) bf16 sQ_sw [2][Br * D];
    __shared__ __align__(128) bf16 sdO_sw[2][Br * D];
    __shared__ __align__(128) bf16 sdO_pl[2][Br * D];
    __shared__ __align__(128) bf16 sO_pl [2][Br * D];
    __shared__ __align__(16)  float sS [Br * SS_STRIDE_V6];
    __shared__ __align__(16)  float sdP[Br * SS_STRIDE_V6];
    __shared__ __align__(16)  bf16  sP [Br * SP_STRIDE_V6];
    __shared__                float sLSE[Br];
    __shared__                float sD  [Br];
    __shared__ __align__(128) bf16  sA_t[SMEM_TILED_V6];
    __shared__ __align__(8)   uint64_t mbar_p;
    __shared__ __align__(8)   uint64_t mbar[2];

    const int tid   = threadIdx.x;
    const int wg    = tid >> 7;          // warpgroup 0/1
    const int wtid  = tid & 127;         // warpgroup-local tid (register↔row map)
    const int gwarp = tid >> 5;          // global warp 0..7
    const int lane  = tid & 31;
    const int b = blockIdx.x, hkv = blockIdx.y, k_tile = blockIdx.z;
    const int k_row0 = k_tile * Bc;
    const int nQTiles = S / Br;

    const long     kvBase     = ((long)(b * Hkv + hkv) * S + k_row0) * D;
    const uint32_t kvFlatRow  = (uint32_t)((b * Hkv + hkv) * S + k_row0);
    const uint32_t bytesTile  = (uint32_t)(Br * D * sizeof(bf16));   // 16384
    const uint32_t bytesAtom  = (uint32_t)(Bc * 64 * sizeof(bf16));  // 8192

    if (tid == 0) { mbar_init_v4(&mbar_p, 1); mbar_init_v4(&mbar[0], 1); mbar_init_v4(&mbar[1], 1); }
    __syncthreads();

    // ── Persistent loads: K swizzled (2 atoms) + V swizzled (2 atoms).
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

    // Persistent accumulators — this warpgroup owns dV/dK columns [wg*64, wg*64+64).
    float dv[32]; zeroN<32>(dv);
    float dk[32]; zeroN<32>(dk);

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

    // Prologue: TMA iter 0 → stage 0.
    uint32_t par[2] = {0, 0};
    if (nIter > 0 && tid == 0) {
        const uint32_t r0 = qFlatRowOf(0);
        mbar_expect_tx_v4(&mbar[0], bytesTile * 4);
        tma_load_2d_v4(&tma_Q_sw,  sQ_sw [0],            &mbar[0], 0,  r0);
        tma_load_2d_v4(&tma_Q_sw,  sQ_sw [0] + 64 * 64,  &mbar[0], 64, r0);
        tma_load_2d_v4(&tma_dO_sw, sdO_sw[0],            &mbar[0], 0,  r0);
        tma_load_2d_v4(&tma_dO_sw, sdO_sw[0] + 64 * 64,  &mbar[0], 64, r0);
        tma_load_2d_v4(&tma_dO_pl, sdO_pl[0],            &mbar[0], 0,  r0);
        tma_load_2d_v4(&tma_O_pl,  sO_pl [0],            &mbar[0], 0,  r0);
    }

    for (int it = 0; it < nIter; it++) {
        const int s = it & 1;
        const int q_row0 = (qc0 + (it % perG)) * Br;
        // Prefetch NEXT iter → other stage (tid 0 only, unchanged from V8).
        if (it + 1 < nIter && tid == 0) {
            const uint32_t nr = qFlatRowOf(it + 1);
            const int sn = s ^ 1;
            mbar_expect_tx_v4(&mbar[sn], bytesTile * 4);
            tma_load_2d_v4(&tma_Q_sw,  sQ_sw [sn],           &mbar[sn], 0,  nr);
            tma_load_2d_v4(&tma_Q_sw,  sQ_sw [sn] + 64 * 64, &mbar[sn], 64, nr);
            tma_load_2d_v4(&tma_dO_sw, sdO_sw[sn],           &mbar[sn], 0,  nr);
            tma_load_2d_v4(&tma_dO_sw, sdO_sw[sn] + 64 * 64, &mbar[sn], 64, nr);
            tma_load_2d_v4(&tma_dO_pl, sdO_pl[sn],           &mbar[sn], 0,  nr);
            tma_load_2d_v4(&tma_O_pl,  sO_pl [sn],           &mbar[sn], 0,  nr);
        }
        mbar_wait_v4(&mbar[s], par[s]); par[s] ^= 1;
        if (tid < Br) sLSE[tid] = d_LSE[lBaseOf(it) + tid];
        __syncthreads();

        // D[r] = rowsum(dO · O) — 8 warps, warp-per-row shuffle reduce (V8's 4-warp
        // loop, now blockDim.x/32 = 8 warps → 8 rows/step instead of 4).
        for (int row = gwarp; row < Br; row += (blockDim.x >> 5)) {
            float partial = 0.f;
            for (int j = lane; j < D; j += 32)
                partial += __bfloat162float(sdO_pl[s][row * D + j]) * __bfloat162float(sO_pl[s][row * D + j]);
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                partial += __shfl_down_sync(0xFFFFFFFFu, partial, off);
            if (lane == 0) sD[row] = partial;
        }
        __syncthreads();

        // ── S = Q·Kᵀ·scale  ∥  dP = dO·Vᵀ  (two INDEPENDENT m64n64 GEMMs, one per
        // warpgroup, issued concurrently — V8 ran these serially).  WG0→sS, WG1→sdP.
        if (wg == 0) {
            float acc[32]; zeroN<32>(acc);
            run_gemm_n64_sw2(acc, sQ_sw[s], sK_sw);
            store_acc_smem_v6<Bc, SS_STRIDE_V6>(acc, sS, wtid, scale);
        } else {
            float acc[32]; zeroN<32>(acc);
            run_gemm_n64_sw2(acc, sdO_sw[s], sV_sw);
            store_acc_smem_v6<Bc, SS_STRIDE_V6>(acc, sdP, wtid, 1.0f);
        }
        __syncthreads();

        // P = exp(S - LSE), causal mask → sP  (all 256 threads).
        for (int i = tid; i < Br * Bc; i += blockDim.x) {
            int r = i / Bc, c = i % Bc;
            sP[r * SP_STRIDE_V6 + c] = (k_row0 + c > q_row0 + r) ? __float2bfloat16(0.f)
                                              : __float2bfloat16(expf(sS[r * SS_STRIDE_V6 + c] - sLSE[r]));
        }
        __syncthreads();

        // dV += Pᵀ·dO — fill Pᵀ→sA_t (256 threads), then WG0 half0 / WG1 half1.
        fill_trans_A_v9<Bc, Br>(sA_t, sP, SP_STRIDE_V6, tid);
        __syncthreads();
        run_gemm_dVdK_half(dv, sA_t, sdO_sw[s] + wg * 4096, sbo_pad_v6(Br));
        __syncthreads();

        // dS = P ⊙ (dP - D) → sP  (all 256 threads).
        for (int i = tid; i < Br * Bc; i += blockDim.x) {
            int r = i / Bc, c = i % Bc;
            int pidx = r * SP_STRIDE_V6 + c;
            sP[pidx] = __float2bfloat16(__bfloat162float(sP[pidx]) * (sdP[r * SS_STRIDE_V6 + c] - sD[r]));
        }
        __syncthreads();

        // dK += dSᵀ·Q — fill dSᵀ→sA_t (256), then WG0 half0 / WG1 half1.
        fill_trans_A_v9<Bc, Br>(sA_t, sP, SP_STRIDE_V6, tid);
        __syncthreads();
        run_gemm_dVdK_half(dk, sA_t, sQ_sw[s] + wg * 4096, sbo_pad_v6(Br));
        __syncthreads();

        // dQ_tile = dS·K (TRANSIENT) — fill dS(K-major)→sA_t (256), then WG0 does
        // N cols [0,64) (K atom0), WG1 does [64,128) (K atom1) → fp32 atomic flush.
        fill_copy_v9<Br, Bc>(sA_t, sP, SP_STRIDE_V6, tid);
        __syncthreads();
        { float acc[32]; zeroN<32>(acc);
          run_gemm_dQ_half(acc, sA_t, sK_sw + wg * 4096, sbo_pad_v6(Bc));
          atomic_acc_global<64>(acc, d_dq_accum, lBaseOf(it) * D, D, wg * 64, wtid, scale); }
        __syncthreads();
    }

    // Writeback — each warpgroup stores its owned dV/dK column-half.
    fence_operandN<32>(dv); fence_operandN<32>(dk);
    store_acc_global_col<64>(dv, d_dV, kvBase, D, wg * 64, wtid, 1.0f);
    store_acc_global_col<64>(dk, d_dK, kvBase, D, wg * 64, wtid, scale);
}

// ── V9 launcher — identical descriptors/scratch to V8; BLOCK(256), V9 kernel.
template<int Br, int Bc, int D>
void launch_gqa_backward_v9(
    const bf16 *d_Q, const bf16 *d_K, const bf16 *d_V, const bf16 *d_O,
    const bf16 *d_dO, const float *d_LSE,
    bf16 *d_dQ, bf16 *d_dK, bf16 *d_dV,
    int B, int Hq, int Hkv, int G, int S, float scale
) {
    static_assert(Br == 64 && Bc == 64 && D == 128, "V9 requires Br=Bc=64, D=128");

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
    CUtensorMap tma_V_sw  = make_tma_sw128(d_V,  Rkv, Bc);
    CUtensorMap tma_Q_sw  = make_tma_sw128(d_Q,  Rq,  Br);
    CUtensorMap tma_dO_sw = make_tma_sw128(d_dO, Rq,  Br);
    CUtensorMap tma_dO_pl = make_tma_plain(d_dO, Rq,  Br);
    CUtensorMap tma_O_pl  = make_tma_plain(d_O,  Rq,  Br);

    const long dqN = (long)B * Hq * S * D;
    static float* d_dq_accum = nullptr;
    static long   dq_cap     = 0;
    if (dqN > dq_cap) {
        if (d_dq_accum) CUDA_CHECK(cudaFree(d_dq_accum));
        CUDA_CHECK(cudaMalloc(&d_dq_accum, dqN * sizeof(float)));
        dq_cap = dqN;
    }
    CUDA_CHECK(cudaMemset(d_dq_accum, 0, dqN * sizeof(float)));

    constexpr dim3 BLOCK(256);                          // 2 warpgroups (V8 was 128)
    dim3 GRID(B, Hkv, S / Bc);
    gqa_backward_v9_kv<Br,Bc,D><<<GRID, BLOCK>>>(
        tma_K_sw, tma_V_sw, tma_Q_sw, tma_dO_sw, tma_dO_pl, tma_O_pl,
        d_LSE, d_dK, d_dV, d_dq_accum, B, Hq, Hkv, G, S, scale);

    const int convBlock = 256;
    const int convGrid  = (int)((dqN + convBlock - 1) / convBlock);
    convert_dq_accum_to_bf16_v5<<<convGrid, convBlock>>>(d_dq_accum, d_dQ, dqN);
}

// ═════════════════════════════════════════════════════════════════════════════
// V10 — COALESCED GLOBAL WRITEBACK.  Structurally IDENTICAL to V9 (same 256-thread
// 2-warpgroup split, same 222,744 B smem, same pipeline/double-buffer/GEMMs) with
// ONE mechanical change: dK/dV/dQ leave the SM through smem-staged, ADDRESS-
// CONTIGUOUS stores instead of the mma-fragment scatter.
//
// The mma m16n8k16 D-fragment map (store_acc_global) hands consecutive lanes
// STRIDED global addresses (r0=w*16+lane/4, r1=r0+8, cc=(lane&3)*2): within any
// single store INSTRUCTION a warp touches only 8/32 B of a bf16 sector (dK/dV) or
// 16/32 B of an fp32 sector (dQ atomics) — the other bytes are filled by *other*
// instructions, so per-transaction efficiency is 8/32 (V9 profile).  V10 reshuffles
// each accumulator through a DEAD smem buffer into ROW-MAJOR order, then re-reads it
// so consecutive threads emit consecutive global addresses → full-sector coalescing.
//
// smem reuse (ZERO net smem — V9 is already at the 222,744 B / 1-block-per-SM cap):
//   • dK/dV (one-shot epilogue, all compute buffers dead): staged in the DEAD sA_t
//     (bf16[8320]=16,640 B, last touched by the loop's dQ GEMM).  WG0 → sA_t[0..4096),
//     WG1 → sA_t[4096..8192) — the 8320-elem buffer holds both 64×64 bf16 halves.
//     → 128-bit (uint4 = 8×bf16) vector stores; consecutive threads = consecutive
//     16-B chunks of a row → 32/32-B coalesced.
//   • dQ (per-iteration fp32 atomic flush): staged in the DEAD sS (WG0) / sdP (WG1)
//     — both fp32[64×65]=16,640 B, dead from the P (reads sS) / dS (reads sdP)
//     computations onward, i.e. well before the flush.  Row-major [64×64]=4096 ≤ 4160.
//     → CONTIGUOUS fp32 atomicAdds; consecutive threads = consecutive addresses so
//     L2 coalesces the atomics (16/32 → 32/32-B).
//
// Staging is a pure data-reshuffle: every (row,col) the fragment owns is written to
// smem once and read once, so the value stored to each global address is BIT-
// IDENTICAL to V9 (for dQ each global location still receives exactly one addend per
// flush — only the issuing thread changes; the per-location sum is unchanged).
// No proxy fence is needed on the reused buffers: the last async user of sA_t
// (run_gemm_dQ_half) retired via wgmma_wait0 + the end-of-iter __syncthreads before
// the epilogue overwrites it, and the staged data is re-read by GENERIC loads
// (uint4 / fp32), never by wgmma.  A __syncthreads() separates each register→smem
// scatter from the smem→global read (RAW), and the dV read from the dK overwrite (WAR).
// ═════════════════════════════════════════════════════════════════════════════

// mma D-fragment (float regs) → ROW-MAJOR bf16 smem stage (stride NCOL, no coloff).
// Same (r0,r1,cc) map as store_acc_global_col; `tid` is warpgroup-local (wtid).
template<int NCOL>
__device__ __forceinline__ void stage_acc_bf16(const float *d, bf16 *stage, int tid, float scl) {
    int w = tid >> 5, lane = tid & 31;
    int r0 = w * 16 + (lane >> 2), r1 = r0 + 8, cc = (lane & 3) * 2;
#pragma unroll
    for (int nt = 0; nt < NCOL / 8; nt++) {
        int c = nt * 8 + cc;
        stage[r0 * NCOL + c + 0] = __float2bfloat16(d[nt * 4 + 0] * scl);
        stage[r0 * NCOL + c + 1] = __float2bfloat16(d[nt * 4 + 1] * scl);
        stage[r1 * NCOL + c + 0] = __float2bfloat16(d[nt * 4 + 2] * scl);
        stage[r1 * NCOL + c + 1] = __float2bfloat16(d[nt * 4 + 3] * scl);
    }
}

// mma D-fragment (float regs) → ROW-MAJOR fp32 smem stage (stride NCOL, no coloff).
template<int NCOL>
__device__ __forceinline__ void stage_acc_f32(const float *d, float *stage, int tid, float scl) {
    int w = tid >> 5, lane = tid & 31;
    int r0 = w * 16 + (lane >> 2), r1 = r0 + 8, cc = (lane & 3) * 2;
#pragma unroll
    for (int nt = 0; nt < NCOL / 8; nt++) {
        int c = nt * 8 + cc;
        stage[r0 * NCOL + c + 0] = d[nt * 4 + 0] * scl;
        stage[r0 * NCOL + c + 1] = d[nt * 4 + 1] * scl;
        stage[r1 * NCOL + c + 0] = d[nt * 4 + 2] * scl;
        stage[r1 * NCOL + c + 1] = d[nt * 4 + 3] * scl;
    }
}

// ROW-MAJOR bf16 stage [ROWS×NCOL] → global g[base + row*D + coloff + col] via
// 128-bit (uint4 = 8 bf16) vector stores.  `tid` warpgroup-local (stride 128).
// NCOL%8==0; base/coloff/row*D all 16-B aligned (cudaMalloc base, D=128, coloff=64).
template<int ROWS, int NCOL>
__device__ __forceinline__ void store_stage_vec(const bf16 *stage, bf16 *g, long base, int D, int coloff, int tid) {
    constexpr int CPR = NCOL / 8;          // 128-bit chunks per row
    constexpr int NCH = ROWS * CPR;        // total chunks
    for (int ci = tid; ci < NCH; ci += 128) {
        int row = ci / CPR;
        int cc  = (ci % CPR) * 8;
        uint4 v = *reinterpret_cast<const uint4*>(&stage[row * NCOL + cc]);
        *reinterpret_cast<uint4*>(&g[base + (long)row * D + coloff + cc]) = v;
    }
}

// ROW-MAJOR fp32 stage [ROWS×NCOL] → global atomicAdd g[base + row*D + coloff + col].
// Consecutive threads → consecutive addresses (same row) → L2 coalesces the atomics.
template<int ROWS, int NCOL>
__device__ __forceinline__ void atomic_flush_stage(const float *stage, float *g, long base, int D, int coloff, int tid) {
    constexpr int N = ROWS * NCOL;
    for (int li = tid; li < N; li += 128) {
        int row = li / NCOL, col = li % NCOL;
        atomicAdd(&g[base + (long)row * D + coloff + col], stage[row * NCOL + col]);
    }
}
// ── V10 kernel — byte-identical smem to V9, 256 threads (2 warpgroups) ─────────
template<int Br, int Bc, int D>
__global__ void __launch_bounds__(256, 1)
gqa_backward_v10_kv(
    const __grid_constant__ CUtensorMap tma_K_sw,   // K 128B-swizzled (S-B Major::K + dQ-B Major::MN)
    const __grid_constant__ CUtensorMap tma_V_sw,   // V 128B-swizzled (dP-B Major::K)
    const __grid_constant__ CUtensorMap tma_Q_sw,   // Q 128B-swizzled (S-A Major::K + dK-B Major::MN)
    const __grid_constant__ CUtensorMap tma_dO_sw,  // dO 128B-swizzled (dP-A Major::K + dV-B Major::MN)
    const __grid_constant__ CUtensorMap tma_dO_pl,  // dO plain (D-rowsum ONLY)
    const __grid_constant__ CUtensorMap tma_O_pl,   // O  plain (D-rowsum)
    const float * __restrict__ d_LSE,
    bf16 * __restrict__ d_dK, bf16 * __restrict__ d_dV, float * __restrict__ d_dq_accum,
    int B, int Hq, int Hkv, int G, int S, float scale
) {
    static_assert(Br == 64 && Bc == 64 && D == 128, "V10 requires Br=Bc=64, D=128");

    // ── smem: BYTE-IDENTICAL to V9/V8 (single sA_t shared read-only across both WGs).
    __shared__ __align__(128) bf16 sK_sw[Bc * D];
    __shared__ __align__(128) bf16 sV_sw[Bc * D];
    __shared__ __align__(128) bf16 sQ_sw [2][Br * D];
    __shared__ __align__(128) bf16 sdO_sw[2][Br * D];
    __shared__ __align__(128) bf16 sdO_pl[2][Br * D];
    __shared__ __align__(128) bf16 sO_pl [2][Br * D];
    __shared__ __align__(16)  float sS [Br * SS_STRIDE_V6];
    __shared__ __align__(16)  float sdP[Br * SS_STRIDE_V6];
    __shared__ __align__(16)  bf16  sP [Br * SP_STRIDE_V6];
    __shared__                float sLSE[Br];
    __shared__                float sD  [Br];
    __shared__ __align__(128) bf16  sA_t[SMEM_TILED_V6];
    __shared__ __align__(8)   uint64_t mbar_p;
    __shared__ __align__(8)   uint64_t mbar[2];

    const int tid   = threadIdx.x;
    const int wg    = tid >> 7;          // warpgroup 0/1
    const int wtid  = tid & 127;         // warpgroup-local tid (register↔row map)
    const int gwarp = tid >> 5;          // global warp 0..7
    const int lane  = tid & 31;
    const int b = blockIdx.x, hkv = blockIdx.y, k_tile = blockIdx.z;
    const int k_row0 = k_tile * Bc;
    const int nQTiles = S / Br;

    const long     kvBase     = ((long)(b * Hkv + hkv) * S + k_row0) * D;
    const uint32_t kvFlatRow  = (uint32_t)((b * Hkv + hkv) * S + k_row0);
    const uint32_t bytesTile  = (uint32_t)(Br * D * sizeof(bf16));   // 16384
    const uint32_t bytesAtom  = (uint32_t)(Bc * 64 * sizeof(bf16));  // 8192

    if (tid == 0) { mbar_init_v4(&mbar_p, 1); mbar_init_v4(&mbar[0], 1); mbar_init_v4(&mbar[1], 1); }
    __syncthreads();

    // ── Persistent loads: K swizzled (2 atoms) + V swizzled (2 atoms).
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

    // Persistent accumulators — this warpgroup owns dV/dK columns [wg*64, wg*64+64).
    float dv[32]; zeroN<32>(dv);
    float dk[32]; zeroN<32>(dk);

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

    // Prologue: TMA iter 0 → stage 0.
    uint32_t par[2] = {0, 0};
    if (nIter > 0 && tid == 0) {
        const uint32_t r0 = qFlatRowOf(0);
        mbar_expect_tx_v4(&mbar[0], bytesTile * 4);
        tma_load_2d_v4(&tma_Q_sw,  sQ_sw [0],            &mbar[0], 0,  r0);
        tma_load_2d_v4(&tma_Q_sw,  sQ_sw [0] + 64 * 64,  &mbar[0], 64, r0);
        tma_load_2d_v4(&tma_dO_sw, sdO_sw[0],            &mbar[0], 0,  r0);
        tma_load_2d_v4(&tma_dO_sw, sdO_sw[0] + 64 * 64,  &mbar[0], 64, r0);
        tma_load_2d_v4(&tma_dO_pl, sdO_pl[0],            &mbar[0], 0,  r0);
        tma_load_2d_v4(&tma_O_pl,  sO_pl [0],            &mbar[0], 0,  r0);
    }

    for (int it = 0; it < nIter; it++) {
        const int s = it & 1;
        const int q_row0 = (qc0 + (it % perG)) * Br;
        // Prefetch NEXT iter → other stage (tid 0 only, unchanged from V9).
        if (it + 1 < nIter && tid == 0) {
            const uint32_t nr = qFlatRowOf(it + 1);
            const int sn = s ^ 1;
            mbar_expect_tx_v4(&mbar[sn], bytesTile * 4);
            tma_load_2d_v4(&tma_Q_sw,  sQ_sw [sn],           &mbar[sn], 0,  nr);
            tma_load_2d_v4(&tma_Q_sw,  sQ_sw [sn] + 64 * 64, &mbar[sn], 64, nr);
            tma_load_2d_v4(&tma_dO_sw, sdO_sw[sn],           &mbar[sn], 0,  nr);
            tma_load_2d_v4(&tma_dO_sw, sdO_sw[sn] + 64 * 64, &mbar[sn], 64, nr);
            tma_load_2d_v4(&tma_dO_pl, sdO_pl[sn],           &mbar[sn], 0,  nr);
            tma_load_2d_v4(&tma_O_pl,  sO_pl [sn],           &mbar[sn], 0,  nr);
        }
        mbar_wait_v4(&mbar[s], par[s]); par[s] ^= 1;
        if (tid < Br) sLSE[tid] = d_LSE[lBaseOf(it) + tid];
        __syncthreads();

        // D[r] = rowsum(dO · O) — 8 warps, warp-per-row shuffle reduce.
        for (int row = gwarp; row < Br; row += (blockDim.x >> 5)) {
            float partial = 0.f;
            for (int j = lane; j < D; j += 32)
                partial += __bfloat162float(sdO_pl[s][row * D + j]) * __bfloat162float(sO_pl[s][row * D + j]);
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                partial += __shfl_down_sync(0xFFFFFFFFu, partial, off);
            if (lane == 0) sD[row] = partial;
        }
        __syncthreads();

        // ── S = Q·Kᵀ·scale  ∥  dP = dO·Vᵀ  (two INDEPENDENT m64n64 GEMMs, one per WG).
        if (wg == 0) {
            float acc[32]; zeroN<32>(acc);
            run_gemm_n64_sw2(acc, sQ_sw[s], sK_sw);
            store_acc_smem_v6<Bc, SS_STRIDE_V6>(acc, sS, wtid, scale);
        } else {
            float acc[32]; zeroN<32>(acc);
            run_gemm_n64_sw2(acc, sdO_sw[s], sV_sw);
            store_acc_smem_v6<Bc, SS_STRIDE_V6>(acc, sdP, wtid, 1.0f);
        }
        __syncthreads();

        // P = exp(S - LSE), causal mask → sP  (all 256 threads).
        for (int i = tid; i < Br * Bc; i += blockDim.x) {
            int r = i / Bc, c = i % Bc;
            sP[r * SP_STRIDE_V6 + c] = (k_row0 + c > q_row0 + r) ? __float2bfloat16(0.f)
                                              : __float2bfloat16(expf(sS[r * SS_STRIDE_V6 + c] - sLSE[r]));
        }
        __syncthreads();

        // dV += Pᵀ·dO — fill Pᵀ→sA_t (256 threads), then WG0 half0 / WG1 half1.
        fill_trans_A_v9<Bc, Br>(sA_t, sP, SP_STRIDE_V6, tid);
        __syncthreads();
        run_gemm_dVdK_half(dv, sA_t, sdO_sw[s] + wg * 4096, sbo_pad_v6(Br));
        __syncthreads();

        // dS = P ⊙ (dP - D) → sP  (all 256 threads).
        for (int i = tid; i < Br * Bc; i += blockDim.x) {
            int r = i / Bc, c = i % Bc;
            int pidx = r * SP_STRIDE_V6 + c;
            sP[pidx] = __float2bfloat16(__bfloat162float(sP[pidx]) * (sdP[r * SS_STRIDE_V6 + c] - sD[r]));
        }
        __syncthreads();

        // dK += dSᵀ·Q — fill dSᵀ→sA_t (256), then WG0 half0 / WG1 half1.
        fill_trans_A_v9<Bc, Br>(sA_t, sP, SP_STRIDE_V6, tid);
        __syncthreads();
        run_gemm_dVdK_half(dk, sA_t, sQ_sw[s] + wg * 4096, sbo_pad_v6(Br));
        __syncthreads();

        // dQ_tile = dS·K (TRANSIENT) — fill dS(K-major)→sA_t (256), then WG0 N cols
        // [0,64) (K atom0), WG1 [64,128) (K atom1).  V10: stage the fp32 half
        // ROW-MAJOR into a DEAD buffer (WG0→sS, WG1→sdP; both dead here), __sync,
        // then flush with CONTIGUOUS fp32 atomicAdds (was: 16/32-B scattered atomics).
        fill_copy_v9<Br, Bc>(sA_t, sP, SP_STRIDE_V6, tid);
        __syncthreads();
        { float acc[32]; zeroN<32>(acc);
          run_gemm_dQ_half(acc, sA_t, sK_sw + wg * 4096, sbo_pad_v6(Bc));
          stage_acc_f32<64>(acc, (wg == 0) ? sS : sdP, wtid, scale); }
        __syncthreads();
        atomic_flush_stage<Br, 64>((wg == 0) ? sS : sdP, d_dq_accum, lBaseOf(it) * D, D, wg * 64, wtid);
        __syncthreads();
    }

    // ── Writeback — COALESCED.  Reuse the DEAD sA_t as bf16 staging (2 halves 64×64;
    // 8320-elem buffer holds sA_t[0..4096)=WG0, sA_t[4096..8192)=WG1).  dV then dK
    // reuse the same region serially (WAR-fenced by the middle __syncthreads).
    bf16 *stage = sA_t + wg * 4096;
    fence_operandN<32>(dv);
    stage_acc_bf16<64>(dv, stage, wtid, 1.0f);
    __syncthreads();
    store_stage_vec<Bc, 64>(stage, d_dV, kvBase, D, wg * 64, wtid);
    __syncthreads();
    fence_operandN<32>(dk);
    stage_acc_bf16<64>(dk, stage, wtid, scale);
    __syncthreads();
    store_stage_vec<Bc, 64>(stage, d_dK, kvBase, D, wg * 64, wtid);
}

// ── V10 launcher — identical descriptors/scratch to V9; BLOCK(256), V10 kernel.
template<int Br, int Bc, int D>
void launch_gqa_backward_v10(
    const bf16 *d_Q, const bf16 *d_K, const bf16 *d_V, const bf16 *d_O,
    const bf16 *d_dO, const float *d_LSE,
    bf16 *d_dQ, bf16 *d_dK, bf16 *d_dV,
    int B, int Hq, int Hkv, int G, int S, float scale
) {
    static_assert(Br == 64 && Bc == 64 && D == 128, "V10 requires Br=Bc=64, D=128");

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
    CUtensorMap tma_V_sw  = make_tma_sw128(d_V,  Rkv, Bc);
    CUtensorMap tma_Q_sw  = make_tma_sw128(d_Q,  Rq,  Br);
    CUtensorMap tma_dO_sw = make_tma_sw128(d_dO, Rq,  Br);
    CUtensorMap tma_dO_pl = make_tma_plain(d_dO, Rq,  Br);
    CUtensorMap tma_O_pl  = make_tma_plain(d_O,  Rq,  Br);

    const long dqN = (long)B * Hq * S * D;
    static float* d_dq_accum = nullptr;
    static long   dq_cap     = 0;
    if (dqN > dq_cap) {
        if (d_dq_accum) CUDA_CHECK(cudaFree(d_dq_accum));
        CUDA_CHECK(cudaMalloc(&d_dq_accum, dqN * sizeof(float)));
        dq_cap = dqN;
    }
    CUDA_CHECK(cudaMemset(d_dq_accum, 0, dqN * sizeof(float)));

    constexpr dim3 BLOCK(256);                          // 2 warpgroups (same as V9)
    dim3 GRID(B, Hkv, S / Bc);
    gqa_backward_v10_kv<Br,Bc,D><<<GRID, BLOCK>>>(
        tma_K_sw, tma_V_sw, tma_Q_sw, tma_dO_sw, tma_dO_pl, tma_O_pl,
        d_LSE, d_dK, d_dV, d_dq_accum, B, Hq, Hkv, G, S, scale);

    const int convBlock = 256;
    const int convGrid  = (int)((dqN + convBlock - 1) / convBlock);
    convert_dq_accum_to_bf16_v5<<<convGrid, convBlock>>>(d_dq_accum, d_dQ, dqN);
}

// ═════════════════════════════════════════════════════════════════════════════
// V11 — 3-warpgroup, PRODUCER/CONSUMER warp specialization (FA3-style).      [SM_90a]
//
//   Threads = 384 = 3 warpgroups.  __launch_bounds__(384,1) ⇒ THEORETICAL
//   OCCUPANCY 18.75% (12 warps / 64), the exact shape cuDNN uses.  smem is byte-
//   identical to V10's compute layout (222,744 B of operand/score buffers) plus a
//   handful of 8-byte mbarriers, so it stays 1 block/SM (smem-pinned, as V10) —
//   the occupancy rises purely from the extra resident warpgroup.
//
//   ROLES  (wg = tid>>7):
//     • wg 0,1  (tid   0..255) = CONSUMERS.  Do ALL compute exactly as V10's two
//                                warpgroups did (D-rowsum, S∥dP, P, dV, dS, dK, dQ,
//                                flush, epilogue).  Both warpgroups issue their
//                                wgmma GEMMs warpgroup-uniformly, unchanged.
//     • wg 2    (tid 256..383) = PRODUCER.  Issues every cp.async.bulk.tensor load
//                                and drives the double buffer.  Only the producer
//                                LEADER (tid==256) emits the TMA/mbarrier ops; the
//                                other 96 producer threads stay resident and spin on
//                                the `empty` barriers (FA3 producer-warp residency),
//                                contributing eligible warps to hide the consumers'
//                                fixed-latency wgmma dependency stalls.
//
//   The compute path's intra-iteration barriers become `bar.sync 1, 256`
//   (consumer_sync, id 1, 256 threads) — NOT __syncthreads() (id 0, all 384): the
//   producer never joins the compute barrier, so putting it on id 0 would deadlock.
//
//   FULL/EMPTY PROTOCOL  (2 double-buffer stages s∈{0,1}):
//     full[s]  (init count 1): producer→consumer.  Leader does expect_tx+6×TMA on
//              full[s]; TMA complete_tx flips the phase; consumers mbar_wait(full[s]).
//     empty[s] (init count 1): consumer→producer.  ONE consumer (tid==0) arrives
//              empty[s] right AFTER the dK GEMM (the last read of the staged Q/dO
//              buffers — and run_gemm_* internally wgmma.wait_group(0)s, so those
//              async operand reads are fully drained before we release the buffer).
//              Producer mbar_wait(empty[s]) before overwriting stage s.
//     mbar_kv  (init count 1): one-shot persistent K/V load (never overwritten).
//
//   DEADLOCK / RACE SAFETY (reasoned, since this can only be compile-verified here):
//     • Priming: producer fills it=0→stage0 and it=1→stage1 WITHOUT an empty-wait
//       (buffers start free); it waits empty[s] only for it≥2.  Consumers always
//       wait full[s].  So the only cross-role waits are full (provided by producer)
//       and empty (provided by consumer at it−2) — a strictly forward chain, no cycle.
//     • Run-ahead bound: producer at iter `it` blocks on empty[it−2] = consumer(it−2)
//       has passed its dK phase ⇒ producer is at most 2 tiles (= buffer depth) ahead.
//       No buffer is overwritten while any consumer read of it is outstanding.
//     • Early empty is safe: after dK, the remaining consumer work (dQ, stage,
//       atomic-flush, epilogue) touches only sK_sw (persistent), sA_t, sS/sdP and
//       the dv/dk register accumulators — NEVER the staged sQ_sw[s]/sdO_sw[s]/
//       sdO_pl[s]/sO_pl[s].  So the producer may refill stage s during the dQ tail.
//     • Termination: producer issues all nIter fills then returns; the last two
//       empty arrivals have no waiter (harmless).  Consumers finish the loop +
//       epilogue independently.  nIter≥1 always for a launched (b,hkv,k_tile).
//
//   ISA notes:
//     • wgmma.mma_async.sync.aligned is a WARPGROUP instruction — all 128 lanes of a
//       warpgroup must issue it collectively (PTX ISA §9.7.14.3).  The `if(wg==0)…
//       else…` and column-half splits are warpgroup-uniform, so `.aligned` holds; the
//       producer warpgroup simply never issues wgmma.
//     • bar.sync 1,256 : named barrier id 1 over 256 threads (PTX ISA §9.7.12.1) —
//       same arrive+wait + shared-memory ordering as __syncthreads, scoped to the
//       consumers.  Producer uses no bar.sync, so it cannot deadlock the barrier.
//     • mbarrier phase/parity: try_wait.parity returns true once the barrier completes
//       the phase of the given parity; caller toggles its own parity each round
//       (PTX ISA §9.7.13.15.{15,18}) — identical to the V10/V4 scheme, just split by role.
//     • setmaxnreg is NOT used: at 127 regs occupancy is smem-pinned, not reg-pinned
//       (384×127 = 48,768 < 65,536), so dealloc/alloc buys no block and only adds risk.
// ═════════════════════════════════════════════════════════════════════════════

// Consumer-only named barrier (id 1, the 256 consumer threads = wg 0+1).
__device__ __forceinline__ void consumer_sync() {
    asm volatile("bar.sync 1, 256;\n" ::: "memory");
}
// Plain directional arrival (no tx bytes) — consumer→producer "buffer empty".
__device__ __forceinline__ void mbar_arrive_v11(uint64_t* mbar) {
    uint32_t p = (uint32_t)__cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];\n" :: "r"(p) : "memory");
}
// blockDim.x-independent fill variants: consumers number exactly `nt` (=256) here,
// so the stride must be `nt`, not blockDim.x (=384) — else producer-range indices
// [256,384),… would never be written.  Same padded-tiled bijection as fill_*_v9.
template<int MN, int K>
__device__ __forceinline__ void fill_copy_v11(bf16 *dst, const bf16 *src, int src_stride, int tid, int nt) {
    for (int i = tid; i < MN * K; i += nt) { int mn = i / K, k = i % K; dst[tiled_off_v6(mn, k, K)] = src[mn * src_stride + k]; }
}
template<int MN, int K>
__device__ __forceinline__ void fill_trans_A_v11(bf16 *dst, const bf16 *src, int src_stride, int tid, int nt) {
    for (int i = tid; i < MN * K; i += nt) { int mn = i / K, k = i % K; dst[mn_off_v6(mn, k, K)] = src[k * src_stride + mn]; }
}

template<int Br, int Bc, int D>
__global__ void __launch_bounds__(384, 1)
gqa_backward_v11_kv(
    const __grid_constant__ CUtensorMap tma_K_sw,
    const __grid_constant__ CUtensorMap tma_V_sw,
    const __grid_constant__ CUtensorMap tma_Q_sw,
    const __grid_constant__ CUtensorMap tma_dO_sw,
    const __grid_constant__ CUtensorMap tma_dO_pl,
    const __grid_constant__ CUtensorMap tma_O_pl,
    const float * __restrict__ d_LSE,
    bf16 * __restrict__ d_dK, bf16 * __restrict__ d_dV, float * __restrict__ d_dq_accum,
    int B, int Hq, int Hkv, int G, int S, float scale
) {
    static_assert(Br == 64 && Bc == 64 && D == 128, "V11 requires Br=Bc=64, D=128");
    constexpr int CONS = 256;   // consumer thread count (wg 0+1)

    // smem: byte-identical compute layout to V10 (single sA_t shared across WGs).
    __shared__ __align__(128) bf16 sK_sw[Bc * D];
    __shared__ __align__(128) bf16 sV_sw[Bc * D];
    __shared__ __align__(128) bf16 sQ_sw [2][Br * D];
    __shared__ __align__(128) bf16 sdO_sw[2][Br * D];
    __shared__ __align__(128) bf16 sdO_pl[2][Br * D];
    __shared__ __align__(128) bf16 sO_pl [2][Br * D];
    __shared__ __align__(16)  float sS [Br * SS_STRIDE_V6];
    __shared__ __align__(16)  float sdP[Br * SS_STRIDE_V6];
    __shared__ __align__(16)  bf16  sP [Br * SP_STRIDE_V6];
    __shared__                float sLSE[Br];
    __shared__                float sD  [Br];
    __shared__ __align__(128) bf16  sA_t[SMEM_TILED_V6];
    __shared__ __align__(8)   uint64_t mbar_kv;     // one-shot persistent K/V
    __shared__ __align__(8)   uint64_t full [2];    // producer → consumer
    __shared__ __align__(8)   uint64_t empty[2];    // consumer → producer

    const int tid   = threadIdx.x;
    const int wg    = tid >> 7;          // 0,1 = consumers ; 2 = producer
    const int wtid  = tid & 127;         // warpgroup-local (register↔row map)
    const int gwarp = tid >> 5;          // global warp 0..11
    const int lane  = tid & 31;
    const int b = blockIdx.x, hkv = blockIdx.y, k_tile = blockIdx.z;
    const int k_row0 = k_tile * Bc;
    const int nQTiles = S / Br;

    const long     kvBase     = ((long)(b * Hkv + hkv) * S + k_row0) * D;
    const uint32_t kvFlatRow  = (uint32_t)((b * Hkv + hkv) * S + k_row0);
    const uint32_t bytesTile  = (uint32_t)(Br * D * sizeof(bf16));   // 16384
    const uint32_t bytesAtom  = (uint32_t)(Bc * 64 * sizeof(bf16));  // 8192

    if (tid == 0) {
        mbar_init_v4(&mbar_kv, 1);
        mbar_init_v4(&full[0], 1);  mbar_init_v4(&full[1], 1);
        mbar_init_v4(&empty[0], 1); mbar_init_v4(&empty[1], 1);
    }
    __syncthreads();                     // the ONLY all-384 barrier — publishes mbar init

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

    // ── PRODUCER (wg 2) ──────────────────────────────────────────────────────
    if (wg == 2) {
        const bool leader = (tid == 256);
        // Persistent K/V load (2 swizzle atoms each), one-shot.
        if (leader) {
            mbar_expect_tx_v4(&mbar_kv, bytesAtom * 4);
            tma_load_2d_v4(&tma_K_sw, sK_sw,           &mbar_kv, 0,  kvFlatRow);
            tma_load_2d_v4(&tma_K_sw, sK_sw + 64 * 64, &mbar_kv, 64, kvFlatRow);
            tma_load_2d_v4(&tma_V_sw, sV_sw,           &mbar_kv, 0,  kvFlatRow);
            tma_load_2d_v4(&tma_V_sw, sV_sw + 64 * 64, &mbar_kv, 64, kvFlatRow);
        }
        uint32_t epar[2] = {0, 0};
        for (int it = 0; it < nIter; it++) {
            const int s = it & 1;
            // Wait for consumers to release stage s (except the first fill of each
            // buffer — it0→stage0, it1→stage1 write fresh buffers, no producer wait).
            if (it >= 2) { mbar_wait_v4(&empty[s], epar[s]); epar[s] ^= 1; }
            if (leader) {
                const uint32_t r = qFlatRowOf(it);
                mbar_expect_tx_v4(&full[s], bytesTile * 4);
                tma_load_2d_v4(&tma_Q_sw,  sQ_sw [s],           &full[s], 0,  r);
                tma_load_2d_v4(&tma_Q_sw,  sQ_sw [s] + 64 * 64, &full[s], 64, r);
                tma_load_2d_v4(&tma_dO_sw, sdO_sw[s],           &full[s], 0,  r);
                tma_load_2d_v4(&tma_dO_sw, sdO_sw[s] + 64 * 64, &full[s], 64, r);
                tma_load_2d_v4(&tma_dO_pl, sdO_pl[s],           &full[s], 0,  r);
                tma_load_2d_v4(&tma_O_pl,  sO_pl [s],           &full[s], 0,  r);
            }
        }
        return;   // producer never touches the compute barrier / epilogue
    }

    // ── CONSUMERS (wg 0,1) ───────────────────────────────────────────────────
    mbar_wait_v4(&mbar_kv, 0);           // persistent K/V visible (TMA→async-proxy ordered)

    float dv[32]; zeroN<32>(dv);
    float dk[32]; zeroN<32>(dk);

    uint32_t cpar[2] = {0, 0};
    for (int it = 0; it < nIter; it++) {
        const int s = it & 1;
        const int q_row0 = (qc0 + (it % perG)) * Br;
        mbar_wait_v4(&full[s], cpar[s]); cpar[s] ^= 1;
        if (tid < Br) sLSE[tid] = d_LSE[lBaseOf(it) + tid];
        consumer_sync();

        // D[r] = rowsum(dO·O) — 8 consumer warps (stride 8), warp-per-row reduce.
        for (int row = gwarp; row < Br; row += 8) {
            float partial = 0.f;
            for (int j = lane; j < D; j += 32)
                partial += __bfloat162float(sdO_pl[s][row * D + j]) * __bfloat162float(sO_pl[s][row * D + j]);
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                partial += __shfl_down_sync(0xFFFFFFFFu, partial, off);
            if (lane == 0) sD[row] = partial;
        }
        consumer_sync();

        // S = Q·Kᵀ·scale  ∥  dP = dO·Vᵀ  (one m64n64 GEMM per consumer WG).
        if (wg == 0) {
            float acc[32]; zeroN<32>(acc);
            run_gemm_n64_sw2(acc, sQ_sw[s], sK_sw);
            store_acc_smem_v6<Bc, SS_STRIDE_V6>(acc, sS, wtid, scale);
        } else {
            float acc[32]; zeroN<32>(acc);
            run_gemm_n64_sw2(acc, sdO_sw[s], sV_sw);
            store_acc_smem_v6<Bc, SS_STRIDE_V6>(acc, sdP, wtid, 1.0f);
        }
        consumer_sync();

        // P = exp(S - LSE), causal mask → sP  (256 consumers, stride 256).
        for (int i = tid; i < Br * Bc; i += CONS) {
            int r = i / Bc, c = i % Bc;
            sP[r * SP_STRIDE_V6 + c] = (k_row0 + c > q_row0 + r) ? __float2bfloat16(0.f)
                                              : __float2bfloat16(expf(sS[r * SS_STRIDE_V6 + c] - sLSE[r]));
        }
        consumer_sync();

        // dV += Pᵀ·dO  (last read of sdO_sw[s]).
        fill_trans_A_v11<Bc, Br>(sA_t, sP, SP_STRIDE_V6, tid, CONS);
        consumer_sync();
        run_gemm_dVdK_half(dv, sA_t, sdO_sw[s] + wg * 4096, sbo_pad_v6(Br));
        consumer_sync();

        // dS = P ⊙ (dP - D) → sP  (256 consumers, stride 256).
        for (int i = tid; i < Br * Bc; i += CONS) {
            int r = i / Bc, c = i % Bc;
            int pidx = r * SP_STRIDE_V6 + c;
            sP[pidx] = __float2bfloat16(__bfloat162float(sP[pidx]) * (sdP[r * SS_STRIDE_V6 + c] - sD[r]));
        }
        consumer_sync();

        // dK += dSᵀ·Q  (last read of sQ_sw[s]; run_gemm wgmma.wait_group(0)s inside).
        fill_trans_A_v11<Bc, Br>(sA_t, sP, SP_STRIDE_V6, tid, CONS);
        consumer_sync();
        run_gemm_dVdK_half(dk, sA_t, sQ_sw[s] + wg * 4096, sbo_pad_v6(Br));
        consumer_sync();                                // all consumers past dK
        if (tid == 0) mbar_arrive_v11(&empty[s]);       // stage s fully consumed → release

        // dQ_tile = dS·K (transient) — reads sK_sw (persistent) + sA_t only.
        fill_copy_v11<Br, Bc>(sA_t, sP, SP_STRIDE_V6, tid, CONS);
        consumer_sync();
        { float acc[32]; zeroN<32>(acc);
          run_gemm_dQ_half(acc, sA_t, sK_sw + wg * 4096, sbo_pad_v6(Bc));
          stage_acc_f32<64>(acc, (wg == 0) ? sS : sdP, wtid, scale); }
        consumer_sync();
        atomic_flush_stage<Br, 64>((wg == 0) ? sS : sdP, d_dq_accum, lBaseOf(it) * D, D, wg * 64, wtid);
        consumer_sync();
    }

    // ── Epilogue — coalesced dV/dK writeback (consumer-only, bar.sync id 1). ──
    bf16 *stage = sA_t + wg * 4096;
    fence_operandN<32>(dv);
    stage_acc_bf16<64>(dv, stage, wtid, 1.0f);
    consumer_sync();
    store_stage_vec<Bc, 64>(stage, d_dV, kvBase, D, wg * 64, wtid);
    consumer_sync();
    fence_operandN<32>(dk);
    stage_acc_bf16<64>(dk, stage, wtid, scale);
    consumer_sync();
    store_stage_vec<Bc, 64>(stage, d_dK, kvBase, D, wg * 64, wtid);
}

// ── V11 launcher — identical descriptors/scratch to V10; BLOCK(384), V11 kernel.
template<int Br, int Bc, int D>
void launch_gqa_backward_v11(
    const bf16 *d_Q, const bf16 *d_K, const bf16 *d_V, const bf16 *d_O,
    const bf16 *d_dO, const float *d_LSE,
    bf16 *d_dQ, bf16 *d_dK, bf16 *d_dV,
    int B, int Hq, int Hkv, int G, int S, float scale
) {
    static_assert(Br == 64 && Bc == 64 && D == 128, "V11 requires Br=Bc=64, D=128");

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
    CUtensorMap tma_V_sw  = make_tma_sw128(d_V,  Rkv, Bc);
    CUtensorMap tma_Q_sw  = make_tma_sw128(d_Q,  Rq,  Br);
    CUtensorMap tma_dO_sw = make_tma_sw128(d_dO, Rq,  Br);
    CUtensorMap tma_dO_pl = make_tma_plain(d_dO, Rq,  Br);
    CUtensorMap tma_O_pl  = make_tma_plain(d_O,  Rq,  Br);

    const long dqN = (long)B * Hq * S * D;
    static float* d_dq_accum = nullptr;
    static long   dq_cap     = 0;
    if (dqN > dq_cap) {
        if (d_dq_accum) CUDA_CHECK(cudaFree(d_dq_accum));
        CUDA_CHECK(cudaMalloc(&d_dq_accum, dqN * sizeof(float)));
        dq_cap = dqN;
    }
    CUDA_CHECK(cudaMemset(d_dq_accum, 0, dqN * sizeof(float)));

    constexpr dim3 BLOCK(384);                          // 3 warpgroups → occupancy 18.75%
    dim3 GRID(B, Hkv, S / Bc);
    gqa_backward_v11_kv<Br,Bc,D><<<GRID, BLOCK>>>(
        tma_K_sw, tma_V_sw, tma_Q_sw, tma_dO_sw, tma_dO_pl, tma_O_pl,
        d_LSE, d_dK, d_dV, d_dq_accum, B, Hq, Hkv, G, S, scale);

    const int convBlock = 256;
    const int convGrid  = (int)((dqN + convBlock - 1) / convBlock);
    convert_dq_accum_to_bf16_v5<<<convGrid, convBlock>>>(d_dq_accum, d_dQ, dqN);
}

// ═════════════════════════════════════════════════════════════════════════════
// V12 — ldmatrix→stmatrix conflict-free sP→sA_t reshuffle  [SM_90a only]
//
// V11 is memory-latency bound (docs/V11_analysis.md): the dominant post-occupancy
// cost is SHARED traffic — a 3.8-way shared-STORE conflict (~33.8%) and excessive
// shared wavefronts (~38.6%), sourced by the three `fill_*_v11` scatters that build
// the wgmma A-operand (Pᵀ/dSᵀ for dV/dK, dS for dQ) out of `sP`.  Each scatter does
// 2-BYTE (bf16) stores at an 8-element stride into the core-matrix-tiled `sA_t`; the
// stride-8 bf16 pattern hits only banks {0,4,8,…,28} → 4-way store conflict, and the
// 2-byte granularity inflates wavefront count.
//
// ── Mechanism (Option B: keep the PROVEN SS-wgmma + descriptors byte-identical) ──
// Replace each scatter with a matched **ldmatrix.x4 → stmatrix.x4** pair (PTX ISA
// §9.7.13.4.{15,16}), 16-BYTE (8×bf16 = one 8×8 b16 core-matrix row) transactions on
// the HW matrix path — the conflict-optimized load/store engine — instead of 2-byte
// scalar stores.  We do NOT feed wgmma from registers (RS form): V11 sits at 168/170
// regs, and RS-wgmma would blow the ceiling; here the 4 ldmatrix regs are transient
// (consumed immediately by stmatrix), so the register budget is essentially unchanged
// and the wgmma reads `sA_t` from SHARED exactly as in V11 (run_gemm_* untouched).
//
// ── Why this is BIT-IDENTICAL by construction (the #1 correctness risk, disarmed) ──
// ldmatrix and stmatrix use the SAME per-lane semantics: lane l supplies the address
// of row (l%8) of matrix (l/8), and the fragment distribution of stmatrix is the exact
// inverse of ldmatrix.  So if I load 4 sub-blocks with ldmatrix.x4 and store the SAME
// opaque {r0,r1,r2,r3} with stmatrix.x4, element (row ρ, col γ) of matrix m is copied
// from ldmatrix-src-addr(l=m*8+ρ)+γ to stmatrix-dst-addr(l=m*8+ρ)+γ — I never touch or
// reinterpret the fragment layout; I only choose the src/dst ROW addresses.  The
// per-lane addresses below are derived from the SAME index formulae as the proven
// fill_trans_A_v11 (mn_off_v6) and fill_copy_v11 (tiled_off_v6), so the resulting
// `sA_t` bytes are identical to V11's → the SS-wgmma result is identical.
//
// The "transpose" that fill_trans_A does (Pᵀ) is NOT an element-level transpose here —
// it is purely a BLOCK-PLACEMENT swap: an 8×8 sub-block P[8·qb:+8][8·cb:+8] whose 8
// rows are 8 CONTIGUOUS bf16 in sP maps to sA_t core (MB=cb, KB=qb) (dV/dK) vs core
// (MB=qb, KB=cb) (dQ), stored row-major inside the core either way.  So no .trans is
// needed on either ldmatrix or stmatrix — the src rows are contiguous and land as
// contiguous core rows; only the destination core BASE differs between the two fills.
//
// ── sP repad: SP_STRIDE_V12 = 72 (vs V11's 66) ──
// ldmatrix requires each lane's row address to be 16-byte aligned (one row = 8×bf16 =
// 16 B).  V11's SP_STRIDE_V6 = 66 elems = 132 B stride is only 4-byte aligned on odd
// rows → illegal for ldmatrix.  72 elems = 144 B (= 9×16 B) makes every sP row start
// 16-byte-aligned, AND its word stride 36 → the 8 ldmatrix row addresses land on 8
// distinct banks (bank += 4/row) → conflict-free matrix load.  Cost: sP grows 64·66·2
// → 64·72·2 = +768 B smem (222,760 → 223,528 B), still < 227 KB H200 limit → still
// exactly 1 block/SM → occupancy 18.75% UNCHANGED.  All other V12 smem/logic == V11.
//
// ── Fences: UNCHANGED from V11 ──  ldmatrix (read sP) and stmatrix (write sA_t) are
// GENERIC-proxy ops, so (a) the consumer_sync (bar.sync) BEFORE each fill still orders
// the generic sP writes (P/dS) → the ldmatrix read across warps; (b) the consumer_sync
// AFTER each fill still orders the stmatrix sA_t writes across warps; (c) the
// fence.proxy.async.shared::cta at the top of every run_gemm_* still bridges the
// generic sA_t writes → the ASYNC wgmma operand read.  stmatrix does NOT change the
// proxy story (it is not an async/TMA op).  The ld→st register dependency orders the
// load before the store within each lane.  Discipline is preserved verbatim.
// ═════════════════════════════════════════════════════════════════════════════

constexpr int SP_STRIDE_V12 = 72;              // sP row stride (elems): 16-B-aligned for ldmatrix
constexpr int ROWPAD64_V12  = (64 >> 3) * 64 + PAD_V6;  // = 520 = rowpad_v6(64), K=Br=Bc=64

// One matched ldmatrix.x4 → stmatrix.x4 (4× 8×8 b16 core matrices), register-passthrough.
// Warp-scoped: EACH lane supplies its own src (ldmatrix) and dst (stmatrix) row address.
// The {r0..r3} fragment is opaque — never reinterpreted — so the copy is exact for ANY
// per-lane address pair (see the by-construction argument in the section header).
__device__ __forceinline__ void ldst_matrix_x4(uint32_t src_saddr, uint32_t dst_saddr) {
    uint32_t r0, r1, r2, r3;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(src_saddr));
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};\n"
                 :: "r"(dst_saddr), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
}

// Common lane→sub-block geometry for the V12 fills (K = MN = 64 → 8×8 grid of 8×8
// sub-blocks).  8 consumer warps × 2 ldmatrix.x4 groups × 4 matrices = 64 sub-blocks.
//   group g = warp*2 + gg (0..15);  the 4 sub-blocks 4g..4g+3 share query-block qb and
//   span kv-blocks cb0..cb0+3 (cb0 ∈ {0,4}).  lane l → matrix mm=l/8, row ρ=l/8-local
//   = l&7; matrix mm's kv-block = cb0+mm.  src (sP) row = 8·qb+ρ, col = 8·(cb0+mm).
// `TRANS` selects the sA_t core BASE:  fill_trans_A → core (MB=cb, KB=qb) = mn_off_v6
// (dV/dK, Pᵀ/dSᵀ);  fill_copy → core (MB=qb, KB=cb) = tiled_off_v6 (dQ, dS).
template<bool TRANS>
__device__ __forceinline__ void sp_to_sAt_v12(bf16* sA_t, const bf16* sP, int tid) {
    const int warp = tid >> 5, lane = tid & 31;
    const int mm = lane >> 3, rho = lane & 7;      // matrix (0..3), row-in-matrix (0..7)
#pragma unroll
    for (int gg = 0; gg < 2; gg++) {
        const int g   = warp * 2 + gg;             // ldmatrix.x4 group (0..15)
        const int qb  = (4 * g) >> 3;              // query-block shared by the 4 sub-blocks
        const int cb  = ((4 * g) & 7) + mm;        // this lane's kv-block (cb0 + mm)
        const bf16* src = sP + (8 * qb + rho) * SP_STRIDE_V12 + 8 * cb;      // contiguous 8-col row
        // Core base: mn_off_v6(mn=8cb,k=8qb) = cb*520+qb*64  (TRANS, Pᵀ/dSᵀ);
        //            tiled_off_v6(mn=8qb,k=8cb) = qb*520+cb*64 (¬TRANS, dS)  — then + row ρ*8.
        bf16* dst = TRANS ? (sA_t + cb * ROWPAD64_V12 + qb * 64 + rho * 8)
                          : (sA_t + qb * ROWPAD64_V12 + cb * 64 + rho * 8);
        ldst_matrix_x4((uint32_t)__cvta_generic_to_shared(src),
                       (uint32_t)__cvta_generic_to_shared(dst));
    }
}

// V12 kernel — clone of gqa_backward_v11_kv; ONLY the sP row stride (→ SP_STRIDE_V12)
// and the three fill_*_v11 scatters (→ ldmatrix→stmatrix sp_to_sAt_v12) change.
template<int Br, int Bc, int D>
__global__ void __launch_bounds__(384, 1)
gqa_backward_v12_kv(
    const __grid_constant__ CUtensorMap tma_K_sw,
    const __grid_constant__ CUtensorMap tma_V_sw,
    const __grid_constant__ CUtensorMap tma_Q_sw,
    const __grid_constant__ CUtensorMap tma_dO_sw,
    const __grid_constant__ CUtensorMap tma_dO_pl,
    const __grid_constant__ CUtensorMap tma_O_pl,
    const float * __restrict__ d_LSE,
    bf16 * __restrict__ d_dK, bf16 * __restrict__ d_dV, float * __restrict__ d_dq_accum,
    int B, int Hq, int Hkv, int G, int S, float scale
) {
    static_assert(Br == 64 && Bc == 64 && D == 128, "V12 requires Br=Bc=64, D=128");
    constexpr int CONS = 256;   // consumer thread count (wg 0+1)

    __shared__ __align__(128) bf16 sK_sw[Bc * D];
    __shared__ __align__(128) bf16 sV_sw[Bc * D];
    __shared__ __align__(128) bf16 sQ_sw [2][Br * D];
    __shared__ __align__(128) bf16 sdO_sw[2][Br * D];
    __shared__ __align__(128) bf16 sdO_pl[2][Br * D];
    __shared__ __align__(128) bf16 sO_pl [2][Br * D];
    __shared__ __align__(16)  float sS [Br * SS_STRIDE_V6];
    __shared__ __align__(16)  float sdP[Br * SS_STRIDE_V6];
    __shared__ __align__(16)  bf16  sP [Br * SP_STRIDE_V12];   // V12: 16-B-aligned stride for ldmatrix
    __shared__                float sLSE[Br];
    __shared__                float sD  [Br];
    __shared__ __align__(128) bf16  sA_t[SMEM_TILED_V6];
    __shared__ __align__(8)   uint64_t mbar_kv;
    __shared__ __align__(8)   uint64_t full [2];
    __shared__ __align__(8)   uint64_t empty[2];

    const int tid   = threadIdx.x;
    const int wg    = tid >> 7;
    const int wtid  = tid & 127;
    const int gwarp = tid >> 5;
    const int lane  = tid & 31;
    const int b = blockIdx.x, hkv = blockIdx.y, k_tile = blockIdx.z;
    const int k_row0 = k_tile * Bc;
    const int nQTiles = S / Br;

    const long     kvBase     = ((long)(b * Hkv + hkv) * S + k_row0) * D;
    const uint32_t kvFlatRow  = (uint32_t)((b * Hkv + hkv) * S + k_row0);
    const uint32_t bytesTile  = (uint32_t)(Br * D * sizeof(bf16));
    const uint32_t bytesAtom  = (uint32_t)(Bc * 64 * sizeof(bf16));

    if (tid == 0) {
        mbar_init_v4(&mbar_kv, 1);
        mbar_init_v4(&full[0], 1);  mbar_init_v4(&full[1], 1);
        mbar_init_v4(&empty[0], 1); mbar_init_v4(&empty[1], 1);
    }
    __syncthreads();

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

    // ── PRODUCER (wg 2) ──────────────────────────────────────────────────────
    if (wg == 2) {
        const bool leader = (tid == 256);
        if (leader) {
            mbar_expect_tx_v4(&mbar_kv, bytesAtom * 4);
            tma_load_2d_v4(&tma_K_sw, sK_sw,           &mbar_kv, 0,  kvFlatRow);
            tma_load_2d_v4(&tma_K_sw, sK_sw + 64 * 64, &mbar_kv, 64, kvFlatRow);
            tma_load_2d_v4(&tma_V_sw, sV_sw,           &mbar_kv, 0,  kvFlatRow);
            tma_load_2d_v4(&tma_V_sw, sV_sw + 64 * 64, &mbar_kv, 64, kvFlatRow);
        }
        uint32_t epar[2] = {0, 0};
        for (int it = 0; it < nIter; it++) {
            const int s = it & 1;
            if (it >= 2) { mbar_wait_v4(&empty[s], epar[s]); epar[s] ^= 1; }
            if (leader) {
                const uint32_t r = qFlatRowOf(it);
                mbar_expect_tx_v4(&full[s], bytesTile * 4);
                tma_load_2d_v4(&tma_Q_sw,  sQ_sw [s],           &full[s], 0,  r);
                tma_load_2d_v4(&tma_Q_sw,  sQ_sw [s] + 64 * 64, &full[s], 64, r);
                tma_load_2d_v4(&tma_dO_sw, sdO_sw[s],           &full[s], 0,  r);
                tma_load_2d_v4(&tma_dO_sw, sdO_sw[s] + 64 * 64, &full[s], 64, r);
                tma_load_2d_v4(&tma_dO_pl, sdO_pl[s],           &full[s], 0,  r);
                tma_load_2d_v4(&tma_O_pl,  sO_pl [s],           &full[s], 0,  r);
            }
        }
        return;
    }

    // ── CONSUMERS (wg 0,1) ───────────────────────────────────────────────────
    mbar_wait_v4(&mbar_kv, 0);

    float dv[32]; zeroN<32>(dv);
    float dk[32]; zeroN<32>(dk);

    uint32_t cpar[2] = {0, 0};
    for (int it = 0; it < nIter; it++) {
        const int s = it & 1;
        const int q_row0 = (qc0 + (it % perG)) * Br;
        mbar_wait_v4(&full[s], cpar[s]); cpar[s] ^= 1;
        if (tid < Br) sLSE[tid] = d_LSE[lBaseOf(it) + tid];
        consumer_sync();

        // D[r] = rowsum(dO·O).
        for (int row = gwarp; row < Br; row += 8) {
            float partial = 0.f;
            for (int j = lane; j < D; j += 32)
                partial += __bfloat162float(sdO_pl[s][row * D + j]) * __bfloat162float(sO_pl[s][row * D + j]);
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                partial += __shfl_down_sync(0xFFFFFFFFu, partial, off);
            if (lane == 0) sD[row] = partial;
        }
        consumer_sync();

        // S = Q·Kᵀ·scale  ∥  dP = dO·Vᵀ.
        if (wg == 0) {
            float acc[32]; zeroN<32>(acc);
            run_gemm_n64_sw2(acc, sQ_sw[s], sK_sw);
            store_acc_smem_v6<Bc, SS_STRIDE_V6>(acc, sS, wtid, scale);
        } else {
            float acc[32]; zeroN<32>(acc);
            run_gemm_n64_sw2(acc, sdO_sw[s], sV_sw);
            store_acc_smem_v6<Bc, SS_STRIDE_V6>(acc, sdP, wtid, 1.0f);
        }
        consumer_sync();

        // P = exp(S - LSE), causal mask → sP  (V12 stride).
        for (int i = tid; i < Br * Bc; i += CONS) {
            int r = i / Bc, c = i % Bc;
            sP[r * SP_STRIDE_V12 + c] = (k_row0 + c > q_row0 + r) ? __float2bfloat16(0.f)
                                              : __float2bfloat16(expf(sS[r * SS_STRIDE_V6 + c] - sLSE[r]));
        }
        consumer_sync();

        // dV += Pᵀ·dO  (Pᵀ built by ldmatrix→stmatrix; last read of sdO_sw[s]).
        sp_to_sAt_v12<true>(sA_t, sP, tid);
        consumer_sync();
        run_gemm_dVdK_half(dv, sA_t, sdO_sw[s] + wg * 4096, sbo_pad_v6(Br));
        consumer_sync();

        // dS = P ⊙ (dP - D) → sP  (V12 stride).
        for (int i = tid; i < Br * Bc; i += CONS) {
            int r = i / Bc, c = i % Bc;
            int pidx = r * SP_STRIDE_V12 + c;
            sP[pidx] = __float2bfloat16(__bfloat162float(sP[pidx]) * (sdP[r * SS_STRIDE_V6 + c] - sD[r]));
        }
        consumer_sync();

        // dK += dSᵀ·Q  (dSᵀ built by ldmatrix→stmatrix; last read of sQ_sw[s]).
        sp_to_sAt_v12<true>(sA_t, sP, tid);
        consumer_sync();
        run_gemm_dVdK_half(dk, sA_t, sQ_sw[s] + wg * 4096, sbo_pad_v6(Br));
        consumer_sync();
        if (tid == 0) mbar_arrive_v11(&empty[s]);

        // dQ_tile = dS·K (transient) — dS built by ldmatrix→stmatrix (K-major layout).
        sp_to_sAt_v12<false>(sA_t, sP, tid);
        consumer_sync();
        { float acc[32]; zeroN<32>(acc);
          run_gemm_dQ_half(acc, sA_t, sK_sw + wg * 4096, sbo_pad_v6(Bc));
          stage_acc_f32<64>(acc, (wg == 0) ? sS : sdP, wtid, scale); }
        consumer_sync();
        atomic_flush_stage<Br, 64>((wg == 0) ? sS : sdP, d_dq_accum, lBaseOf(it) * D, D, wg * 64, wtid);
        consumer_sync();
    }

    // ── Epilogue — coalesced dV/dK writeback (identical to V11). ──
    bf16 *stage = sA_t + wg * 4096;
    fence_operandN<32>(dv);
    stage_acc_bf16<64>(dv, stage, wtid, 1.0f);
    consumer_sync();
    store_stage_vec<Bc, 64>(stage, d_dV, kvBase, D, wg * 64, wtid);
    consumer_sync();
    fence_operandN<32>(dk);
    stage_acc_bf16<64>(dk, stage, wtid, scale);
    consumer_sync();
    store_stage_vec<Bc, 64>(stage, d_dK, kvBase, D, wg * 64, wtid);
}

// ── V12 launcher — identical descriptors/scratch to V11; BLOCK(384), V12 kernel.
template<int Br, int Bc, int D>
void launch_gqa_backward_v12(
    const bf16 *d_Q, const bf16 *d_K, const bf16 *d_V, const bf16 *d_O,
    const bf16 *d_dO, const float *d_LSE,
    bf16 *d_dQ, bf16 *d_dK, bf16 *d_dV,
    int B, int Hq, int Hkv, int G, int S, float scale
) {
    static_assert(Br == 64 && Bc == 64 && D == 128, "V12 requires Br=Bc=64, D=128");

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
    CUtensorMap tma_V_sw  = make_tma_sw128(d_V,  Rkv, Bc);
    CUtensorMap tma_Q_sw  = make_tma_sw128(d_Q,  Rq,  Br);
    CUtensorMap tma_dO_sw = make_tma_sw128(d_dO, Rq,  Br);
    CUtensorMap tma_dO_pl = make_tma_plain(d_dO, Rq,  Br);
    CUtensorMap tma_O_pl  = make_tma_plain(d_O,  Rq,  Br);

    const long dqN = (long)B * Hq * S * D;
    static float* d_dq_accum = nullptr;
    static long   dq_cap     = 0;
    if (dqN > dq_cap) {
        if (d_dq_accum) CUDA_CHECK(cudaFree(d_dq_accum));
        CUDA_CHECK(cudaMalloc(&d_dq_accum, dqN * sizeof(float)));
        dq_cap = dqN;
    }
    CUDA_CHECK(cudaMemset(d_dq_accum, 0, dqN * sizeof(float)));

    constexpr dim3 BLOCK(384);
    dim3 GRID(B, Hkv, S / Bc);
    gqa_backward_v12_kv<Br,Bc,D><<<GRID, BLOCK>>>(
        tma_K_sw, tma_V_sw, tma_Q_sw, tma_dO_sw, tma_dO_pl, tma_O_pl,
        d_LSE, d_dK, d_dV, d_dq_accum, B, Hq, Hkv, G, S, scale);

    const int convBlock = 256;
    const int convGrid  = (int)((dqN + convBlock - 1) / convBlock);
    convert_dq_accum_to_bf16_v5<<<convGrid, convBlock>>>(d_dq_accum, d_dQ, dqN);
}

// ═════════════════════════════════════════════════════════════════════════════
// V13 — register-fused softmax + producer-side D-rowsum        [SM_90a only]
//
// Two changes over V12; output bit-identical (2e-2), smem ≈ V12, ≤170 regs.
//
//  PART B (register-fused softmax): after run_gemm_n64_sw2, wg0's 128 threads
//    already hold the FULL 64×64 S = Q·Kᵀ in acc[32] (the store_acc_smem_v6
//    fragment map covers all 4096 elements).  Instead of store S→sS, sync, reload
//    sS to form P, wg0 computes P = exp(S·scale − LSE) + causal mask DIRECTLY on
//    acc and writes straight to sP (SP_STRIDE_V12).  Removes the sS store, the
//    sS→P reload loop, and ONE consumer_sync per tile.  acc*scale (fp32 reg) ==
//    the old sS value (fp32) exactly, so exp() is bit-identical.
//    NOTE: sS is NOT deleted — it is reused as wg0's dQ fp32 staging buffer in the
//    dQ phase; deleting it would require an equal-size fp32 stage elsewhere (a
//    net-zero smem move, and no free 16 KB region exists: sQ_sw[s]/sdO_sw[s] are
//    released to the producer by empty[s] before the dQ phase).  So smem is kept
//    flat and occupancy (18.75%, 1 block/SM) is unchanged.
//
//  PART A (producer-side D-rowsum): D[r]=rowsum(dO·O) moves OFF the consumers ONTO
//    wg2's 128 (otherwise-idle) producer threads, overlapping the consumer GEMMs.
//    The producer computes D for tile it−1 one tile LAGGED (so its TMA-issue loop
//    keeps 2 tiles in flight), into double-buffered sD[2][Br], syncs its own
//    warpgroup (bar.sync 2,128), and the leader arrives d_ready[sp].  Consumers
//    drop their D-rowsum phase (+ its sync) and wait d_ready[s] before the dS
//    phase reads sD[s].
//    Deadlock argument: the producer's D-compute for a tile depends ONLY on that
//    tile's TMA completing (full[sp], which the producer itself issued) — never on
//    any consumer action — so d_ready[sp] is always reachable.  Consumers' single
//    new blocking point (d_ready[s] before dS) is therefore always satisfied.
//    full[sp] is waited by BOTH roles (non-consuming try_wait.parity; per-role
//    parity counters), safe for many observers.  sD is double-buffered because the
//    producer runs ≤2 tiles ahead: producer(it+2) overwrites sD[s] only after
//    gating on empty[s], which consumer(it) arrives AFTER its dS read of sD[s].
// ═════════════════════════════════════════════════════════════════════════════

// wg0: P = exp(S·scale − LSE) + causal mask, DIRECT from the m64n64 S fragment →
// sP.  (r,c)→acc index map is store_acc_smem_v6's, retargeted to SP_STRIDE_V12.
template<int Bc>
__device__ __forceinline__ void fused_p_from_acc_v13(
    const float acc[32], bf16* sP, const float* sLSE, int wtid,
    int q_row0, int k_row0, float scale)
{
    const int w = wtid >> 5, lane = wtid & 31;
    const int r0 = w * 16 + (lane >> 2), r1 = r0 + 8, cc = (lane & 3) * 2;
    const float l0 = sLSE[r0], l1 = sLSE[r1];
    const int gr0 = q_row0 + r0, gr1 = q_row0 + r1;
#pragma unroll
    for (int nt = 0; nt < Bc / 8; nt++) {
        const int c   = nt * 8 + cc;
        const int gc0 = k_row0 + c, gc1 = gc0 + 1;
        sP[r0 * SP_STRIDE_V12 + c + 0] = (gc0 > gr0) ? __float2bfloat16(0.f)
                                       : __float2bfloat16(expf(acc[nt * 4 + 0] * scale - l0));
        sP[r0 * SP_STRIDE_V12 + c + 1] = (gc1 > gr0) ? __float2bfloat16(0.f)
                                       : __float2bfloat16(expf(acc[nt * 4 + 1] * scale - l0));
        sP[r1 * SP_STRIDE_V12 + c + 0] = (gc0 > gr1) ? __float2bfloat16(0.f)
                                       : __float2bfloat16(expf(acc[nt * 4 + 2] * scale - l1));
        sP[r1 * SP_STRIDE_V12 + c + 1] = (gc1 > gr1) ? __float2bfloat16(0.f)
                                       : __float2bfloat16(expf(acc[nt * 4 + 3] * scale - l1));
    }
}

// Producer warpgroup (128 thr): D[row] = Σ_j dO[row,j]·O[row,j].  pwarp∈0..3,
// stride 4 covers all Br rows; per-row j-order + shuffle-tree are identical to the
// consumer version → bit-identical D.
template<int Br, int D>
__device__ __forceinline__ void producer_drowsum_v13(
    float* sDrow, const bf16* sdO, const bf16* sO, int pwarp, int lane)
{
    for (int row = pwarp; row < Br; row += 4) {
        float partial = 0.f;
        for (int j = lane; j < D; j += 32)
            partial += __bfloat162float(sdO[row * D + j]) * __bfloat162float(sO[row * D + j]);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            partial += __shfl_down_sync(0xFFFFFFFFu, partial, off);
        if (lane == 0) sDrow[row] = partial;
    }
}

// Producer-only named barrier (id 2, the 128 producer threads = wg2).  Distinct
// from id 0 (__syncthreads, 384) and id 1 (consumer_sync, 256).
__device__ __forceinline__ void producer_sync() {
    asm volatile("bar.sync 2, 128;\n" ::: "memory");
}

template<int Br, int Bc, int D>
__global__ void
gqa_backward_v13_kv(
    const __grid_constant__ CUtensorMap tma_K_sw,
    const __grid_constant__ CUtensorMap tma_V_sw,
    const __grid_constant__ CUtensorMap tma_Q_sw,
    const __grid_constant__ CUtensorMap tma_dO_sw,
    const __grid_constant__ CUtensorMap tma_dO_pl,
    const __grid_constant__ CUtensorMap tma_O_pl,
    const float * __restrict__ d_LSE,
    bf16 * __restrict__ d_dK, bf16 * __restrict__ d_dV, float * __restrict__ d_dq_accum,
    int B, int Hq, int Hkv, int G, int S, float scale
) {
    static_assert(Br == 64 && Bc == 64 && D == 128, "V13 requires Br=Bc=64, D=128");
    constexpr int CONS = 256;   // consumer thread count (wg 0+1)

    __shared__ __align__(128) bf16 sK_sw[Bc * D];
    __shared__ __align__(128) bf16 sV_sw[Bc * D];
    __shared__ __align__(128) bf16 sQ_sw [2][Br * D];
    __shared__ __align__(128) bf16 sdO_sw[2][Br * D];
    __shared__ __align__(128) bf16 sdO_pl[2][Br * D];
    __shared__ __align__(128) bf16 sO_pl [2][Br * D];
    __shared__ __align__(16)  float sS [Br * SS_STRIDE_V6];   // V13: S store DROPPED; kept as wg0 dQ stage
    __shared__ __align__(16)  float sdP[Br * SS_STRIDE_V6];
    __shared__ __align__(16)  bf16  sP [Br * SP_STRIDE_V12];  // V12: 16-B-aligned stride for ldmatrix
    __shared__                float sLSE[Br];
    __shared__                float sD  [2][Br];              // V13: double-buffered, producer-written
    __shared__ __align__(128) bf16  sA_t[SMEM_TILED_V6];
    __shared__ __align__(8)   uint64_t mbar_kv;
    __shared__ __align__(8)   uint64_t full   [2];
    __shared__ __align__(8)   uint64_t empty  [2];
    __shared__ __align__(8)   uint64_t d_ready[2];            // V13: producer→consumer D signal

    const int tid   = threadIdx.x;
    const int wg    = tid >> 7;
    const int wtid  = tid & 127;
    const int lane  = tid & 31;
    const int b = blockIdx.x, hkv = blockIdx.y, k_tile = blockIdx.z;
    const int k_row0 = k_tile * Bc;
    const int nQTiles = S / Br;

    const long     kvBase     = ((long)(b * Hkv + hkv) * S + k_row0) * D;
    const uint32_t kvFlatRow  = (uint32_t)((b * Hkv + hkv) * S + k_row0);
    const uint32_t bytesTile  = (uint32_t)(Br * D * sizeof(bf16));
    const uint32_t bytesAtom  = (uint32_t)(Bc * 64 * sizeof(bf16));

    if (tid == 0) {
        mbar_init_v4(&mbar_kv, 1);
        mbar_init_v4(&full[0], 1);    mbar_init_v4(&full[1], 1);
        mbar_init_v4(&empty[0], 1);   mbar_init_v4(&empty[1], 1);
        mbar_init_v4(&d_ready[0], 1); mbar_init_v4(&d_ready[1], 1);
    }
    __syncthreads();

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

    // ── PRODUCER (wg 2): TMA loads + LAGGED D-rowsum (PART A) ─────────────────
    if (wg == 2) {
        const bool leader = (tid == 256);
        const int pwarp   = (tid - 256) >> 5;   // 0..3 within the producer warpgroup
        if (leader) {
            mbar_expect_tx_v4(&mbar_kv, bytesAtom * 4);
            tma_load_2d_v4(&tma_K_sw, sK_sw,           &mbar_kv, 0,  kvFlatRow);
            tma_load_2d_v4(&tma_K_sw, sK_sw + 64 * 64, &mbar_kv, 64, kvFlatRow);
            tma_load_2d_v4(&tma_V_sw, sV_sw,           &mbar_kv, 0,  kvFlatRow);
            tma_load_2d_v4(&tma_V_sw, sV_sw + 64 * 64, &mbar_kv, 64, kvFlatRow);
        }
        uint32_t epar[2] = {0, 0}, fpar[2] = {0, 0};
        // Compute D for tile `td` (waits its TMA, reduces, arrives d_ready).
        auto do_drowsum = [&](int td) {
            const int sp = td & 1;
            mbar_wait_v4(&full[sp], fpar[sp]); fpar[sp] ^= 1;
            producer_drowsum_v13<Br, D>(sD[sp], sdO_pl[sp], sO_pl[sp], pwarp, lane);
            producer_sync();
            if (leader) mbar_arrive_v11(&d_ready[sp]);
        };
        for (int it = 0; it < nIter; it++) {
            const int s = it & 1;
            if (it >= 2) { mbar_wait_v4(&empty[s], epar[s]); epar[s] ^= 1; }
            if (leader) {
                const uint32_t r = qFlatRowOf(it);
                mbar_expect_tx_v4(&full[s], bytesTile * 4);
                tma_load_2d_v4(&tma_Q_sw,  sQ_sw [s],           &full[s], 0,  r);
                tma_load_2d_v4(&tma_Q_sw,  sQ_sw [s] + 64 * 64, &full[s], 64, r);
                tma_load_2d_v4(&tma_dO_sw, sdO_sw[s],           &full[s], 0,  r);
                tma_load_2d_v4(&tma_dO_sw, sdO_sw[s] + 64 * 64, &full[s], 64, r);
                tma_load_2d_v4(&tma_dO_pl, sdO_pl[s],           &full[s], 0,  r);
                tma_load_2d_v4(&tma_O_pl,  sO_pl [s],           &full[s], 0,  r);
            }
            if (it >= 1) do_drowsum(it - 1);   // lagged one tile → 2 TMAs in flight
        }
        do_drowsum(nIter - 1);                 // tail: final tile's D
        return;
    }

    // ── CONSUMERS (wg 0,1) ───────────────────────────────────────────────────
    mbar_wait_v4(&mbar_kv, 0);

    float dv[32]; zeroN<32>(dv);
    float dk[32]; zeroN<32>(dk);

    uint32_t cpar[2] = {0, 0}, dpar[2] = {0, 0};
    for (int it = 0; it < nIter; it++) {
        const int s = it & 1;
        const int q_row0 = (qc0 + (it % perG)) * Br;
        mbar_wait_v4(&full[s], cpar[s]); cpar[s] ^= 1;
        if (tid < Br) sLSE[tid] = d_LSE[lBaseOf(it) + tid];
        consumer_sync();

        // S = Q·Kᵀ·scale → P = exp(S − LSE)+causal DIRECT to sP (wg0, PART B)
        //  ∥  dP = dO·Vᵀ → sdP (wg1).   [D-rowsum now on the producer, PART A.]
        if (wg == 0) {
            float acc[32]; zeroN<32>(acc);
            run_gemm_n64_sw2(acc, sQ_sw[s], sK_sw);
            fused_p_from_acc_v13<Bc>(acc, sP, sLSE, wtid, q_row0, k_row0, scale);
        } else {
            float acc[32]; zeroN<32>(acc);
            run_gemm_n64_sw2(acc, sdO_sw[s], sV_sw);
            store_acc_smem_v6<Bc, SS_STRIDE_V6>(acc, sdP, wtid, 1.0f);
        }
        consumer_sync();

        // dV += Pᵀ·dO  (Pᵀ built by ldmatrix→stmatrix; last read of sdO_sw[s]).
        sp_to_sAt_v12<true>(sA_t, sP, tid);
        consumer_sync();
        run_gemm_dVdK_half(dv, sA_t, sdO_sw[s] + wg * 4096, sbo_pad_v6(Br));
        consumer_sync();

        // dS = P ⊙ (dP − D) → sP.  D produced by wg2 → wait d_ready[s] first.
        mbar_wait_v4(&d_ready[s], dpar[s]); dpar[s] ^= 1;
        for (int i = tid; i < Br * Bc; i += CONS) {
            int r = i / Bc, c = i % Bc;
            int pidx = r * SP_STRIDE_V12 + c;
            sP[pidx] = __float2bfloat16(__bfloat162float(sP[pidx]) * (sdP[r * SS_STRIDE_V6 + c] - sD[s][r]));
        }
        consumer_sync();

        // dK += dSᵀ·Q  (dSᵀ built by ldmatrix→stmatrix; last read of sQ_sw[s]).
        sp_to_sAt_v12<true>(sA_t, sP, tid);
        consumer_sync();
        run_gemm_dVdK_half(dk, sA_t, sQ_sw[s] + wg * 4096, sbo_pad_v6(Br));
        consumer_sync();
        if (tid == 0) mbar_arrive_v11(&empty[s]);

        // dQ_tile = dS·K (transient) — dS built by ldmatrix→stmatrix (K-major).
        sp_to_sAt_v12<false>(sA_t, sP, tid);
        consumer_sync();
        { float acc[32]; zeroN<32>(acc);
          run_gemm_dQ_half(acc, sA_t, sK_sw + wg * 4096, sbo_pad_v6(Bc));
          stage_acc_f32<64>(acc, (wg == 0) ? sS : sdP, wtid, scale); }
        consumer_sync();
        atomic_flush_stage<Br, 64>((wg == 0) ? sS : sdP, d_dq_accum, lBaseOf(it) * D, D, wg * 64, wtid);
        consumer_sync();
    }

    // ── Epilogue — coalesced dV/dK writeback (identical to V11/V12). ──
    bf16 *stage = sA_t + wg * 4096;
    fence_operandN<32>(dv);
    stage_acc_bf16<64>(dv, stage, wtid, 1.0f);
    consumer_sync();
    store_stage_vec<Bc, 64>(stage, d_dV, kvBase, D, wg * 64, wtid);
    consumer_sync();
    fence_operandN<32>(dk);
    stage_acc_bf16<64>(dk, stage, wtid, scale);
    consumer_sync();
    store_stage_vec<Bc, 64>(stage, d_dK, kvBase, D, wg * 64, wtid);
}

// ── V13 launcher — identical descriptors/scratch to V12; BLOCK(384), V13 kernel.
template<int Br, int Bc, int D>
void launch_gqa_backward_v13(
    const bf16 *d_Q, const bf16 *d_K, const bf16 *d_V, const bf16 *d_O,
    const bf16 *d_dO, const float *d_LSE,
    bf16 *d_dQ, bf16 *d_dK, bf16 *d_dV,
    int B, int Hq, int Hkv, int G, int S, float scale
) {
    static_assert(Br == 64 && Bc == 64 && D == 128, "V13 requires Br=Bc=64, D=128");

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
    CUtensorMap tma_V_sw  = make_tma_sw128(d_V,  Rkv, Bc);
    CUtensorMap tma_Q_sw  = make_tma_sw128(d_Q,  Rq,  Br);
    CUtensorMap tma_dO_sw = make_tma_sw128(d_dO, Rq,  Br);
    CUtensorMap tma_dO_pl = make_tma_plain(d_dO, Rq,  Br);
    CUtensorMap tma_O_pl  = make_tma_plain(d_O,  Rq,  Br);

    const long dqN = (long)B * Hq * S * D;
    static float* d_dq_accum = nullptr;
    static long   dq_cap     = 0;
    if (dqN > dq_cap) {
        if (d_dq_accum) CUDA_CHECK(cudaFree(d_dq_accum));
        CUDA_CHECK(cudaMalloc(&d_dq_accum, dqN * sizeof(float)));
        dq_cap = dqN;
    }
    CUDA_CHECK(cudaMemset(d_dq_accum, 0, dqN * sizeof(float)));

    constexpr dim3 BLOCK(384);
    dim3 GRID(B, Hkv, S / Bc);
    gqa_backward_v13_kv<Br,Bc,D><<<GRID, BLOCK>>>(
        tma_K_sw, tma_V_sw, tma_Q_sw, tma_dO_sw, tma_dO_pl, tma_O_pl,
        d_LSE, d_dK, d_dV, d_dq_accum, B, Hq, Hkv, G, S, scale);

    const int convBlock = 256;
    const int convGrid  = (int)((dqN + convBlock - 1) / convBlock);
    convert_dq_accum_to_bf16_v5<<<convGrid, convBlock>>>(d_dq_accum, d_dQ, dqN);
}

// ─────────────────────────────────────────────────────────────────────────────

// V14 — V13 clone; ONLY change: fused-P softmax uses __expf (fast SFU exp,

// strips the range-reduce/reconstruct ALU around EX2 on wg0's critical path).

// ─────────────────────────────────────────────────────────────────────────────

template<int Bc>
__device__ __forceinline__ void fused_p_from_acc_v14(
    const float acc[32], bf16* sP, const float* sLSE, int wtid,
    int q_row0, int k_row0, float scale)
{
    const int w = wtid >> 5, lane = wtid & 31;
    const int r0 = w * 16 + (lane >> 2), r1 = r0 + 8, cc = (lane & 3) * 2;
    const float l0 = sLSE[r0], l1 = sLSE[r1];
    const int gr0 = q_row0 + r0, gr1 = q_row0 + r1;
#pragma unroll
    for (int nt = 0; nt < Bc / 8; nt++) {
        const int c   = nt * 8 + cc;
        const int gc0 = k_row0 + c, gc1 = gc0 + 1;
        sP[r0 * SP_STRIDE_V12 + c + 0] = (gc0 > gr0) ? __float2bfloat16(0.f)
                                       : __float2bfloat16(__expf(acc[nt * 4 + 0] * scale - l0));
        sP[r0 * SP_STRIDE_V12 + c + 1] = (gc1 > gr0) ? __float2bfloat16(0.f)
                                       : __float2bfloat16(__expf(acc[nt * 4 + 1] * scale - l0));
        sP[r1 * SP_STRIDE_V12 + c + 0] = (gc0 > gr1) ? __float2bfloat16(0.f)
                                       : __float2bfloat16(__expf(acc[nt * 4 + 2] * scale - l1));
        sP[r1 * SP_STRIDE_V12 + c + 1] = (gc1 > gr1) ? __float2bfloat16(0.f)
                                       : __float2bfloat16(__expf(acc[nt * 4 + 3] * scale - l1));
    }
}

template<int Br, int Bc, int D>
__global__ void
gqa_backward_v14_kv(
    const __grid_constant__ CUtensorMap tma_K_sw,
    const __grid_constant__ CUtensorMap tma_V_sw,
    const __grid_constant__ CUtensorMap tma_Q_sw,
    const __grid_constant__ CUtensorMap tma_dO_sw,
    const __grid_constant__ CUtensorMap tma_dO_pl,
    const __grid_constant__ CUtensorMap tma_O_pl,
    const float * __restrict__ d_LSE,
    bf16 * __restrict__ d_dK, bf16 * __restrict__ d_dV, float * __restrict__ d_dq_accum,
    int B, int Hq, int Hkv, int G, int S, float scale
) {
    static_assert(Br == 64 && Bc == 64 && D == 128, "V13 requires Br=Bc=64, D=128");
    constexpr int CONS = 256;   // consumer thread count (wg 0+1)

    __shared__ __align__(128) bf16 sK_sw[Bc * D];
    __shared__ __align__(128) bf16 sV_sw[Bc * D];
    __shared__ __align__(128) bf16 sQ_sw [2][Br * D];
    __shared__ __align__(128) bf16 sdO_sw[2][Br * D];
    __shared__ __align__(128) bf16 sdO_pl[2][Br * D];
    __shared__ __align__(128) bf16 sO_pl [2][Br * D];
    __shared__ __align__(16)  float sS [Br * SS_STRIDE_V6];   // V13: S store DROPPED; kept as wg0 dQ stage
    __shared__ __align__(16)  float sdP[Br * SS_STRIDE_V6];
    __shared__ __align__(16)  bf16  sP [Br * SP_STRIDE_V12];  // V12: 16-B-aligned stride for ldmatrix
    __shared__                float sLSE[Br];
    __shared__                float sD  [2][Br];              // V13: double-buffered, producer-written
    __shared__ __align__(128) bf16  sA_t[SMEM_TILED_V6];
    __shared__ __align__(8)   uint64_t mbar_kv;
    __shared__ __align__(8)   uint64_t full   [2];
    __shared__ __align__(8)   uint64_t empty  [2];
    __shared__ __align__(8)   uint64_t d_ready[2];            // V13: producer→consumer D signal

    const int tid   = threadIdx.x;
    const int wg    = tid >> 7;
    const int wtid  = tid & 127;
    const int lane  = tid & 31;
    const int b = blockIdx.x, hkv = blockIdx.y, k_tile = blockIdx.z;
    const int k_row0 = k_tile * Bc;
    const int nQTiles = S / Br;

    const long     kvBase     = ((long)(b * Hkv + hkv) * S + k_row0) * D;
    const uint32_t kvFlatRow  = (uint32_t)((b * Hkv + hkv) * S + k_row0);
    const uint32_t bytesTile  = (uint32_t)(Br * D * sizeof(bf16));
    const uint32_t bytesAtom  = (uint32_t)(Bc * 64 * sizeof(bf16));

    if (tid == 0) {
        mbar_init_v4(&mbar_kv, 1);
        mbar_init_v4(&full[0], 1);    mbar_init_v4(&full[1], 1);
        mbar_init_v4(&empty[0], 1);   mbar_init_v4(&empty[1], 1);
        mbar_init_v4(&d_ready[0], 1); mbar_init_v4(&d_ready[1], 1);
    }
    __syncthreads();

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

    // ── PRODUCER (wg 2): TMA loads + LAGGED D-rowsum (PART A) ─────────────────
    if (wg == 2) {
        const bool leader = (tid == 256);
        const int pwarp   = (tid - 256) >> 5;   // 0..3 within the producer warpgroup
        if (leader) {
            mbar_expect_tx_v4(&mbar_kv, bytesAtom * 4);
            tma_load_2d_v4(&tma_K_sw, sK_sw,           &mbar_kv, 0,  kvFlatRow);
            tma_load_2d_v4(&tma_K_sw, sK_sw + 64 * 64, &mbar_kv, 64, kvFlatRow);
            tma_load_2d_v4(&tma_V_sw, sV_sw,           &mbar_kv, 0,  kvFlatRow);
            tma_load_2d_v4(&tma_V_sw, sV_sw + 64 * 64, &mbar_kv, 64, kvFlatRow);
        }
        uint32_t epar[2] = {0, 0}, fpar[2] = {0, 0};
        // Compute D for tile `td` (waits its TMA, reduces, arrives d_ready).
        auto do_drowsum = [&](int td) {
            const int sp = td & 1;
            mbar_wait_v4(&full[sp], fpar[sp]); fpar[sp] ^= 1;
            producer_drowsum_v13<Br, D>(sD[sp], sdO_pl[sp], sO_pl[sp], pwarp, lane);
            producer_sync();
            if (leader) mbar_arrive_v11(&d_ready[sp]);
        };
        for (int it = 0; it < nIter; it++) {
            const int s = it & 1;
            if (it >= 2) { mbar_wait_v4(&empty[s], epar[s]); epar[s] ^= 1; }
            if (leader) {
                const uint32_t r = qFlatRowOf(it);
                mbar_expect_tx_v4(&full[s], bytesTile * 4);
                tma_load_2d_v4(&tma_Q_sw,  sQ_sw [s],           &full[s], 0,  r);
                tma_load_2d_v4(&tma_Q_sw,  sQ_sw [s] + 64 * 64, &full[s], 64, r);
                tma_load_2d_v4(&tma_dO_sw, sdO_sw[s],           &full[s], 0,  r);
                tma_load_2d_v4(&tma_dO_sw, sdO_sw[s] + 64 * 64, &full[s], 64, r);
                tma_load_2d_v4(&tma_dO_pl, sdO_pl[s],           &full[s], 0,  r);
                tma_load_2d_v4(&tma_O_pl,  sO_pl [s],           &full[s], 0,  r);
            }
            if (it >= 1) do_drowsum(it - 1);   // lagged one tile → 2 TMAs in flight
        }
        do_drowsum(nIter - 1);                 // tail: final tile's D
        return;
    }

    // ── CONSUMERS (wg 0,1) ───────────────────────────────────────────────────
    mbar_wait_v4(&mbar_kv, 0);

    float dv[32]; zeroN<32>(dv);
    float dk[32]; zeroN<32>(dk);

    uint32_t cpar[2] = {0, 0}, dpar[2] = {0, 0};
    for (int it = 0; it < nIter; it++) {
        const int s = it & 1;
        const int q_row0 = (qc0 + (it % perG)) * Br;
        mbar_wait_v4(&full[s], cpar[s]); cpar[s] ^= 1;
        if (tid < Br) sLSE[tid] = d_LSE[lBaseOf(it) + tid];
        consumer_sync();

        // S = Q·Kᵀ·scale → P = exp(S − LSE)+causal DIRECT to sP (wg0, PART B)
        //  ∥  dP = dO·Vᵀ → sdP (wg1).   [D-rowsum now on the producer, PART A.]
        if (wg == 0) {
            float acc[32]; zeroN<32>(acc);
            run_gemm_n64_sw2(acc, sQ_sw[s], sK_sw);
            fused_p_from_acc_v14<Bc>(acc, sP, sLSE, wtid, q_row0, k_row0, scale);
        } else {
            float acc[32]; zeroN<32>(acc);
            run_gemm_n64_sw2(acc, sdO_sw[s], sV_sw);
            store_acc_smem_v6<Bc, SS_STRIDE_V6>(acc, sdP, wtid, 1.0f);
        }
        consumer_sync();

        // dV += Pᵀ·dO  (Pᵀ built by ldmatrix→stmatrix; last read of sdO_sw[s]).
        sp_to_sAt_v12<true>(sA_t, sP, tid);
        consumer_sync();
        run_gemm_dVdK_half(dv, sA_t, sdO_sw[s] + wg * 4096, sbo_pad_v6(Br));
        consumer_sync();

        // dS = P ⊙ (dP − D) → sP.  D produced by wg2 → wait d_ready[s] first.
        mbar_wait_v4(&d_ready[s], dpar[s]); dpar[s] ^= 1;
        for (int i = tid; i < Br * Bc; i += CONS) {
            int r = i / Bc, c = i % Bc;
            int pidx = r * SP_STRIDE_V12 + c;
            sP[pidx] = __float2bfloat16(__bfloat162float(sP[pidx]) * (sdP[r * SS_STRIDE_V6 + c] - sD[s][r]));
        }
        consumer_sync();

        // dK += dSᵀ·Q  (dSᵀ built by ldmatrix→stmatrix; last read of sQ_sw[s]).
        sp_to_sAt_v12<true>(sA_t, sP, tid);
        consumer_sync();
        run_gemm_dVdK_half(dk, sA_t, sQ_sw[s] + wg * 4096, sbo_pad_v6(Br));
        consumer_sync();
        if (tid == 0) mbar_arrive_v11(&empty[s]);

        // dQ_tile = dS·K (transient) — dS built by ldmatrix→stmatrix (K-major).
        sp_to_sAt_v12<false>(sA_t, sP, tid);
        consumer_sync();
        { float acc[32]; zeroN<32>(acc);
          run_gemm_dQ_half(acc, sA_t, sK_sw + wg * 4096, sbo_pad_v6(Bc));
          stage_acc_f32<64>(acc, (wg == 0) ? sS : sdP, wtid, scale); }
        consumer_sync();
        atomic_flush_stage<Br, 64>((wg == 0) ? sS : sdP, d_dq_accum, lBaseOf(it) * D, D, wg * 64, wtid);
        consumer_sync();
    }

    // ── Epilogue — coalesced dV/dK writeback (identical to V11/V12). ──
    bf16 *stage = sA_t + wg * 4096;
    fence_operandN<32>(dv);
    stage_acc_bf16<64>(dv, stage, wtid, 1.0f);
    consumer_sync();
    store_stage_vec<Bc, 64>(stage, d_dV, kvBase, D, wg * 64, wtid);
    consumer_sync();
    fence_operandN<32>(dk);
    stage_acc_bf16<64>(dk, stage, wtid, scale);
    consumer_sync();
    store_stage_vec<Bc, 64>(stage, d_dK, kvBase, D, wg * 64, wtid);
}

// ── V14 launcher — identical to V13; only fused-P uses __expf (fast SFU exp).
template<int Br, int Bc, int D>
void launch_gqa_backward_v14(
    const bf16 *d_Q, const bf16 *d_K, const bf16 *d_V, const bf16 *d_O,
    const bf16 *d_dO, const float *d_LSE,
    bf16 *d_dQ, bf16 *d_dK, bf16 *d_dV,
    int B, int Hq, int Hkv, int G, int S, float scale
) {
    static_assert(Br == 64 && Bc == 64 && D == 128, "V14 requires Br=Bc=64, D=128");

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
    CUtensorMap tma_V_sw  = make_tma_sw128(d_V,  Rkv, Bc);
    CUtensorMap tma_Q_sw  = make_tma_sw128(d_Q,  Rq,  Br);
    CUtensorMap tma_dO_sw = make_tma_sw128(d_dO, Rq,  Br);
    CUtensorMap tma_dO_pl = make_tma_plain(d_dO, Rq,  Br);
    CUtensorMap tma_O_pl  = make_tma_plain(d_O,  Rq,  Br);

    const long dqN = (long)B * Hq * S * D;
    static float* d_dq_accum = nullptr;
    static long   dq_cap     = 0;
    if (dqN > dq_cap) {
        if (d_dq_accum) CUDA_CHECK(cudaFree(d_dq_accum));
        CUDA_CHECK(cudaMalloc(&d_dq_accum, dqN * sizeof(float)));
        dq_cap = dqN;
    }
    CUDA_CHECK(cudaMemset(d_dq_accum, 0, dqN * sizeof(float)));

    constexpr dim3 BLOCK(384);
    dim3 GRID(B, Hkv, S / Bc);
    gqa_backward_v14_kv<Br,Bc,D><<<GRID, BLOCK>>>(
        tma_K_sw, tma_V_sw, tma_Q_sw, tma_dO_sw, tma_dO_pl, tma_O_pl,
        d_LSE, d_dK, d_dV, d_dq_accum, B, Hq, Hkv, G, S, scale);

    const int convBlock = 256;
    const int convGrid  = (int)((dqN + convBlock - 1) / convBlock);
    convert_dq_accum_to_bf16_v5<<<convGrid, convBlock>>>(d_dq_accum, d_dQ, dqN);
}

// ─────────────────────────────────────────────────────────────────────────────
// V15 — V14 clone; ONLY change: per-iteration (g,qc) address indices maintained
// incrementally instead of it/perG (IDIV) + it%perG (IREM) — removes the runtime
// integer divide/modulo from the address→TMA critical path (chain-shortening).
// ─────────────────────────────────────────────────────────────────────────────
template<int Br, int Bc, int D>
__global__ void
gqa_backward_v15_kv(
    const __grid_constant__ CUtensorMap tma_K_sw,
    const __grid_constant__ CUtensorMap tma_V_sw,
    const __grid_constant__ CUtensorMap tma_Q_sw,
    const __grid_constant__ CUtensorMap tma_dO_sw,
    const __grid_constant__ CUtensorMap tma_dO_pl,
    const __grid_constant__ CUtensorMap tma_O_pl,
    const float * __restrict__ d_LSE,
    bf16 * __restrict__ d_dK, bf16 * __restrict__ d_dV, float * __restrict__ d_dq_accum,
    int B, int Hq, int Hkv, int G, int S, float scale
) {
    static_assert(Br == 64 && Bc == 64 && D == 128, "V13 requires Br=Bc=64, D=128");
    constexpr int CONS = 256;   // consumer thread count (wg 0+1)

    __shared__ __align__(128) bf16 sK_sw[Bc * D];
    __shared__ __align__(128) bf16 sV_sw[Bc * D];
    __shared__ __align__(128) bf16 sQ_sw [2][Br * D];
    __shared__ __align__(128) bf16 sdO_sw[2][Br * D];
    __shared__ __align__(128) bf16 sdO_pl[2][Br * D];
    __shared__ __align__(128) bf16 sO_pl [2][Br * D];
    __shared__ __align__(16)  float sS [Br * SS_STRIDE_V6];   // V13: S store DROPPED; kept as wg0 dQ stage
    __shared__ __align__(16)  float sdP[Br * SS_STRIDE_V6];
    __shared__ __align__(16)  bf16  sP [Br * SP_STRIDE_V12];  // V12: 16-B-aligned stride for ldmatrix
    __shared__                float sLSE[Br];
    __shared__                float sD  [2][Br];              // V13: double-buffered, producer-written
    __shared__ __align__(128) bf16  sA_t[SMEM_TILED_V6];
    __shared__ __align__(8)   uint64_t mbar_kv;
    __shared__ __align__(8)   uint64_t full   [2];
    __shared__ __align__(8)   uint64_t empty  [2];
    __shared__ __align__(8)   uint64_t d_ready[2];            // V13: producer→consumer D signal

    const int tid   = threadIdx.x;
    const int wg    = tid >> 7;
    const int wtid  = tid & 127;
    const int lane  = tid & 31;
    const int b = blockIdx.x, hkv = blockIdx.y, k_tile = blockIdx.z;
    const int k_row0 = k_tile * Bc;
    const int nQTiles = S / Br;

    const long     kvBase     = ((long)(b * Hkv + hkv) * S + k_row0) * D;
    const uint32_t kvFlatRow  = (uint32_t)((b * Hkv + hkv) * S + k_row0);
    const uint32_t bytesTile  = (uint32_t)(Br * D * sizeof(bf16));
    const uint32_t bytesAtom  = (uint32_t)(Bc * 64 * sizeof(bf16));

    if (tid == 0) {
        mbar_init_v4(&mbar_kv, 1);
        mbar_init_v4(&full[0], 1);    mbar_init_v4(&full[1], 1);
        mbar_init_v4(&empty[0], 1);   mbar_init_v4(&empty[1], 1);
        mbar_init_v4(&d_ready[0], 1); mbar_init_v4(&d_ready[1], 1);
    }
    __syncthreads();

    const int qc0    = k_row0 / Br;
    const int perG   = nQTiles - qc0;
    const int nIter  = G * perG;

    // V15: (g,qc) maintained incrementally in each loop → no per-iter IDIV/IREM.
    auto qFlatRowOf = [&](int g, int qc) -> uint32_t {
        const int hq = hkv * G + g;
        return (uint32_t)((b * Hq + hq) * S + qc * Br);
    };
    auto lBaseOf = [&](int g, int qc) -> long {
        const int hq = hkv * G + g;
        return (long)(b * Hq + hq) * S + (long)qc * Br;
    };

    // ── PRODUCER (wg 2): TMA loads + LAGGED D-rowsum (PART A) ─────────────────
    if (wg == 2) {
        const bool leader = (tid == 256);
        const int pwarp   = (tid - 256) >> 5;   // 0..3 within the producer warpgroup
        if (leader) {
            mbar_expect_tx_v4(&mbar_kv, bytesAtom * 4);
            tma_load_2d_v4(&tma_K_sw, sK_sw,           &mbar_kv, 0,  kvFlatRow);
            tma_load_2d_v4(&tma_K_sw, sK_sw + 64 * 64, &mbar_kv, 64, kvFlatRow);
            tma_load_2d_v4(&tma_V_sw, sV_sw,           &mbar_kv, 0,  kvFlatRow);
            tma_load_2d_v4(&tma_V_sw, sV_sw + 64 * 64, &mbar_kv, 64, kvFlatRow);
        }
        uint32_t epar[2] = {0, 0}, fpar[2] = {0, 0};
        int gP = 0, qcP = qc0;   // V15 incremental (g,qc)
        // Compute D for tile `td` (waits its TMA, reduces, arrives d_ready).
        auto do_drowsum = [&](int td) {
            const int sp = td & 1;
            mbar_wait_v4(&full[sp], fpar[sp]); fpar[sp] ^= 1;
            producer_drowsum_v13<Br, D>(sD[sp], sdO_pl[sp], sO_pl[sp], pwarp, lane);
            producer_sync();
            if (leader) mbar_arrive_v11(&d_ready[sp]);
        };
        for (int it = 0; it < nIter; it++) {
            const int s = it & 1;
            if (it >= 2) { mbar_wait_v4(&empty[s], epar[s]); epar[s] ^= 1; }
            if (leader) {
                const uint32_t r = qFlatRowOf(gP, qcP);
                mbar_expect_tx_v4(&full[s], bytesTile * 4);
                tma_load_2d_v4(&tma_Q_sw,  sQ_sw [s],           &full[s], 0,  r);
                tma_load_2d_v4(&tma_Q_sw,  sQ_sw [s] + 64 * 64, &full[s], 64, r);
                tma_load_2d_v4(&tma_dO_sw, sdO_sw[s],           &full[s], 0,  r);
                tma_load_2d_v4(&tma_dO_sw, sdO_sw[s] + 64 * 64, &full[s], 64, r);
                tma_load_2d_v4(&tma_dO_pl, sdO_pl[s],           &full[s], 0,  r);
                tma_load_2d_v4(&tma_O_pl,  sO_pl [s],           &full[s], 0,  r);
            }
            if (it >= 1) do_drowsum(it - 1);   // lagged one tile → 2 TMAs in flight
            if (++qcP == nQTiles) { qcP = qc0; ++gP; }
        }
        do_drowsum(nIter - 1);                 // tail: final tile's D
        return;
    }

    // ── CONSUMERS (wg 0,1) ───────────────────────────────────────────────────
    mbar_wait_v4(&mbar_kv, 0);

    float dv[32]; zeroN<32>(dv);
    float dk[32]; zeroN<32>(dk);

    uint32_t cpar[2] = {0, 0}, dpar[2] = {0, 0};
    int gC = 0, qcC = qc0;   // V15 incremental (g,qc)
    for (int it = 0; it < nIter; it++) {
        const int s = it & 1;
        const int q_row0 = qcC * Br;
        mbar_wait_v4(&full[s], cpar[s]); cpar[s] ^= 1;
        if (tid < Br) sLSE[tid] = d_LSE[lBaseOf(gC, qcC) + tid];
        consumer_sync();

        // S = Q·Kᵀ·scale → P = exp(S − LSE)+causal DIRECT to sP (wg0, PART B)
        //  ∥  dP = dO·Vᵀ → sdP (wg1).   [D-rowsum now on the producer, PART A.]
        if (wg == 0) {
            float acc[32]; zeroN<32>(acc);
            run_gemm_n64_sw2(acc, sQ_sw[s], sK_sw);
            fused_p_from_acc_v14<Bc>(acc, sP, sLSE, wtid, q_row0, k_row0, scale);
        } else {
            float acc[32]; zeroN<32>(acc);
            run_gemm_n64_sw2(acc, sdO_sw[s], sV_sw);
            store_acc_smem_v6<Bc, SS_STRIDE_V6>(acc, sdP, wtid, 1.0f);
        }
        consumer_sync();

        // dV += Pᵀ·dO  (Pᵀ built by ldmatrix→stmatrix; last read of sdO_sw[s]).
        sp_to_sAt_v12<true>(sA_t, sP, tid);
        consumer_sync();
        run_gemm_dVdK_half(dv, sA_t, sdO_sw[s] + wg * 4096, sbo_pad_v6(Br));
        consumer_sync();

        // dS = P ⊙ (dP − D) → sP.  D produced by wg2 → wait d_ready[s] first.
        mbar_wait_v4(&d_ready[s], dpar[s]); dpar[s] ^= 1;
        for (int i = tid; i < Br * Bc; i += CONS) {
            int r = i / Bc, c = i % Bc;
            int pidx = r * SP_STRIDE_V12 + c;
            sP[pidx] = __float2bfloat16(__bfloat162float(sP[pidx]) * (sdP[r * SS_STRIDE_V6 + c] - sD[s][r]));
        }
        consumer_sync();

        // dK += dSᵀ·Q  (dSᵀ built by ldmatrix→stmatrix; last read of sQ_sw[s]).
        sp_to_sAt_v12<true>(sA_t, sP, tid);
        consumer_sync();
        run_gemm_dVdK_half(dk, sA_t, sQ_sw[s] + wg * 4096, sbo_pad_v6(Br));
        consumer_sync();
        if (tid == 0) mbar_arrive_v11(&empty[s]);

        // dQ_tile = dS·K (transient) — dS built by ldmatrix→stmatrix (K-major).
        sp_to_sAt_v12<false>(sA_t, sP, tid);
        consumer_sync();
        { float acc[32]; zeroN<32>(acc);
          run_gemm_dQ_half(acc, sA_t, sK_sw + wg * 4096, sbo_pad_v6(Bc));
          stage_acc_f32<64>(acc, (wg == 0) ? sS : sdP, wtid, scale); }
        consumer_sync();
        atomic_flush_stage<Br, 64>((wg == 0) ? sS : sdP, d_dq_accum, lBaseOf(gC, qcC) * D, D, wg * 64, wtid);
        consumer_sync();
        if (++qcC == nQTiles) { qcC = qc0; ++gC; }
    }

    // ── Epilogue — coalesced dV/dK writeback (identical to V11/V12). ──
    bf16 *stage = sA_t + wg * 4096;
    fence_operandN<32>(dv);
    stage_acc_bf16<64>(dv, stage, wtid, 1.0f);
    consumer_sync();
    store_stage_vec<Bc, 64>(stage, d_dV, kvBase, D, wg * 64, wtid);
    consumer_sync();
    fence_operandN<32>(dk);
    stage_acc_bf16<64>(dk, stage, wtid, scale);
    consumer_sync();
    store_stage_vec<Bc, 64>(stage, d_dK, kvBase, D, wg * 64, wtid);
}

// ── V15 launcher — identical to V14; kernel hoists per-iter (g,qc) address math.
template<int Br, int Bc, int D>
void launch_gqa_backward_v15(
    const bf16 *d_Q, const bf16 *d_K, const bf16 *d_V, const bf16 *d_O,
    const bf16 *d_dO, const float *d_LSE,
    bf16 *d_dQ, bf16 *d_dK, bf16 *d_dV,
    int B, int Hq, int Hkv, int G, int S, float scale
) {
    static_assert(Br == 64 && Bc == 64 && D == 128, "V14 requires Br=Bc=64, D=128");

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
    CUtensorMap tma_V_sw  = make_tma_sw128(d_V,  Rkv, Bc);
    CUtensorMap tma_Q_sw  = make_tma_sw128(d_Q,  Rq,  Br);
    CUtensorMap tma_dO_sw = make_tma_sw128(d_dO, Rq,  Br);
    CUtensorMap tma_dO_pl = make_tma_plain(d_dO, Rq,  Br);
    CUtensorMap tma_O_pl  = make_tma_plain(d_O,  Rq,  Br);

    const long dqN = (long)B * Hq * S * D;
    static float* d_dq_accum = nullptr;
    static long   dq_cap     = 0;
    if (dqN > dq_cap) {
        if (d_dq_accum) CUDA_CHECK(cudaFree(d_dq_accum));
        CUDA_CHECK(cudaMalloc(&d_dq_accum, dqN * sizeof(float)));
        dq_cap = dqN;
    }
    CUDA_CHECK(cudaMemset(d_dq_accum, 0, dqN * sizeof(float)));

    constexpr dim3 BLOCK(384);
    dim3 GRID(B, Hkv, S / Bc);
    gqa_backward_v15_kv<Br,Bc,D><<<GRID, BLOCK>>>(
        tma_K_sw, tma_V_sw, tma_Q_sw, tma_dO_sw, tma_dO_pl, tma_O_pl,
        d_LSE, d_dK, d_dV, d_dq_accum, B, Hq, Hkv, G, S, scale);

    const int convBlock = 256;
    const int convGrid  = (int)((dqN + convBlock - 1) / convBlock);
    convert_dq_accum_to_bf16_v5<<<convGrid, convBlock>>>(d_dq_accum, d_dQ, dqN);
}

// ═════════════════════════════════════════════════════════════════════════════
// V16 — V15 clone + TRANSPOSE-ELIMINATION for the dV and dK GEMMs.
//
// WHAT CHANGES vs V15.  V15's consumer loop builds Pᵀ (for dV) and dSᵀ (for dK) via
// sp_to_sAt_v12<true> — an ldmatrix→stmatrix round-trip that WRITES then READS sA_t in
// shared.  That write-then-read is the #1 stall in the V15 profile (short_scoreboard
// 2.16 = shared-load latency).  V16 reads sP DIRECTLY through a TRANSPOSED wgmma
// descriptor (layout aliasing, ZERO data movement), deleting the two sA_t transposed-
// staging round-trips (and their 2 consumer_sync barriers).  This is FA3/cuDNN's method
// (mainloop_bwd_sm90_tma_gmma_ws.hpp) and it targets the DOMINANT stall now.
//
// THE OPERAND ARRANGEMENT (HW-verified recipe — do NOT re-derive):
//   dV += Pᵀ·dO :  A = Pᵀ  read from swizzled sP (Major::MN, trans-a=1),
//                  B = dO   from swizzled sdO_sw  (Major::MN, trans-b=1).
//   dK += dSᵀ·Q :  A = dSᵀ read from swizzled sP (Major::MN, trans-a=1),
//                  B = Q    from swizzled sQ_sw   (Major::MN, trans-b=1).
//   The A read is BYTE-IDENTICAL to V7's proven B=dO read: make_desc_sw128_MN + k*1024
//   advance, 4 k-steps (contraction = Br query rows).  BOTH operands swizzled Major::MN
//   is the correct arrangement (V8's failure was A-swizzled-Major::K + B-swizzled-MN,
//   a layout mismatch for a TRANSPOSED operand — NOT this case).
//
// THE ALIGNMENT GOTCHA (root-caused; cost the original attempt a full debug cycle):
//   make_desc_sw128_MN sets base_offset=0 → it ASSUMES the swizzled buffer is 1024-B
//   aligned (one B128 swizzle atom).  So sP MUST be __align__(1024) (V15's was 16).
//   A non-1024 base injects a fixed row-phase XOR → permutation-like same-magnitude
//   wrong dV/dK (dQ immune).  k-advances of +1024 elems (=+2048 B) preserve the phase.
//
// SWIZZLED sP LAYOUT.  sP is now a single 64×64 (=128-B-per-row) swizzle atom, 4096
// bf16, stride 64.  It is WRITTEN in swizzled layout (fused_p + the dS = P⊙(dP−D) step
// both use sw128_idx) so the descriptor reads it correctly.  The swizzle byte-formula
// sw128_idx and the +k*1024 k-advance were HW-verified (V8 scaffold, since stripped;
// reconstructed here from agent-mem reference_wgmma_swizzle_atom_alignment).
//
// dQ PATH UNCHANGED IN SPIRIT.  dQ still stages dS into sA_t via ldmatrix→stmatrix and
// runs the proven no-swizzle run_gemm_dQ_half — ONLY the ldmatrix SOURCE addresses are
// XOR-adjusted (sp_to_sAt_v16_dq) to read the now-swizzled sP.  That read is relative-
// addressed (phase-independent → robust to the align bug).  sA_t is RETAINED (dQ stage
// + epilogue writeback stage `sA_t + wg*4096`).
//
// BIT-IDENTICAL EXPECTATION.  This is a pure data-movement change: the P/dS VALUES and
// the contraction are unchanged, only WHERE the operand bytes live and how the tensor
// core addresses them.  If the swizzle/align layout is right, dV/dK/dQ are all bit-
// identical to V15.  (HONEST CAUTION: an earlier transpose-elim attempt measured ~FLAT
// on wall-clock; we rebuild it because it targets the dominant stall AND frees a little
// smem — the user measures the H200 wall-clock.)
// ═════════════════════════════════════════════════════════════════════════════

// 128-B swizzle index for a 64-wide (128-B-per-row) bf16 atom.  Element (row,col) of a
// logical [rows][64] tile lives at this offset in the swizzled buffer.  HW-verified to
// match make_desc_sw128_K/MN's descriptor read (agent-mem
// reference_wgmma_swizzle_atom_alignment).  row∈[0,64), col∈[0,64) → offset∈[0,4096).
__device__ __forceinline__ int sw128_idx(int row, int col) {
    return row * 64 + (((col >> 3) ^ (row & 7)) << 3) + (col & 7);
}

// wg0: P = exp(S·scale − LSE) + causal mask, DIRECT from the m64n64 S fragment → sP,
// written in the 128-B-SWIZZLED layout (sw128_idx) so the transposed wgmma A-read sees
// it correctly.  Identical VALUES to fused_p_from_acc_v14 (same __expf, same (r,c)→acc
// map); ONLY the destination index changes (sw128_idx vs r*SP_STRIDE_V12+c).  c is even
// and c,c+1 share col>>3 → adjacent swizzled offsets (base, base+1).
template<int Bc>
__device__ __forceinline__ void fused_p_from_acc_v16(
    const float acc[32], bf16* sP, const float* sLSE, int wtid,
    int q_row0, int k_row0, float scale)
{
    const int w = wtid >> 5, lane = wtid & 31;
    const int r0 = w * 16 + (lane >> 2), r1 = r0 + 8, cc = (lane & 3) * 2;
    const float l0 = sLSE[r0], l1 = sLSE[r1];
    const int gr0 = q_row0 + r0, gr1 = q_row0 + r1;
#pragma unroll
    for (int nt = 0; nt < Bc / 8; nt++) {
        const int c   = nt * 8 + cc;
        const int gc0 = k_row0 + c, gc1 = gc0 + 1;
        const int b0  = sw128_idx(r0, c);   // (r0,c) and (r0,c+1) → b0, b0+1
        const int b1  = sw128_idx(r1, c);   // (r1,c) and (r1,c+1) → b1, b1+1
        sP[b0 + 0] = (gc0 > gr0) ? __float2bfloat16(0.f)
                                 : __float2bfloat16(__expf(acc[nt * 4 + 0] * scale - l0));
        sP[b0 + 1] = (gc1 > gr0) ? __float2bfloat16(0.f)
                                 : __float2bfloat16(__expf(acc[nt * 4 + 1] * scale - l0));
        sP[b1 + 0] = (gc0 > gr1) ? __float2bfloat16(0.f)
                                 : __float2bfloat16(__expf(acc[nt * 4 + 2] * scale - l1));
        sP[b1 + 1] = (gc1 > gr1) ? __float2bfloat16(0.f)
                                 : __float2bfloat16(__expf(acc[nt * 4 + 3] * scale - l1));
    }
}

// dQ-only variant of sp_to_sAt_v12<false>: reads the now-SWIZZLED sP and stages dS into
// sA_t (tiled_off, K-major, no-swizzle) for the UNCHANGED run_gemm_dQ_half.  The dst is
// identical to sp_to_sAt_v12<false>; ONLY the ldmatrix src address changes to the
// swizzled offset.  For the 8-col run at (row=8·qb+rho, colblock=8·cb): the 8 elements
// are contiguous at base = row·64 + ((cb ^ (row&7))<<3), with row&7 = rho.  Relative-
// addressed → phase-independent (robust to the 1024-B align gotcha).
__device__ __forceinline__ void sp_to_sAt_v16_dq(bf16* sA_t, const bf16* sP, int tid) {
    const int warp = tid >> 5, lane = tid & 31;
    const int mm = lane >> 3, rho = lane & 7;      // matrix (0..3), row-in-matrix (0..7)
#pragma unroll
    for (int gg = 0; gg < 2; gg++) {
        const int g   = warp * 2 + gg;             // ldmatrix.x4 group (0..15)
        const int qb  = (4 * g) >> 3;              // query-block shared by the 4 sub-blocks
        const int cb  = ((4 * g) & 7) + mm;        // this lane's kv-block (cb0 + mm)
        const int row = 8 * qb + rho;
        const bf16* src = sP + row * 64 + ((cb ^ rho) << 3);          // swizzled 8-col run
        bf16* dst = sA_t + qb * ROWPAD64_V12 + cb * 64 + rho * 8;     // tiled_off (¬TRANS), unchanged
        ldst_matrix_x4((uint32_t)__cvta_generic_to_shared(src),
                       (uint32_t)__cvta_generic_to_shared(dst));
    }
}

// dV += Pᵀ·dO / dK += dSᵀ·Q — one m64n64 HALF, TRANSPOSE-ELIMINATED.  A = Pᵀ/dSᵀ read
// DIRECTLY from swizzled sP (Major::MN, trans-a=1); B = dO/Q from swizzled sdO_sw/sQ_sw
// (Major::MN, trans-b=1).  Both A and B advance +k*1024 elems (16 rows / one 128-B atom
// per MMA-K step), K = Br = 64 → 4 k-steps.  A-read is byte-identical to the B-read.
// sP MUST be __align__(1024).  No sbo_A (both operands swizzled).
__device__ __forceinline__ void run_gemm_dVdK_half_te(float acc[32], const bf16* sP_sw,
                                                      const bf16* B_sw_half) {
    fence_proxy_async_shared();       // orders the generic fused_p/dS write of sP → async wgmma read
    fence_operandN<32>(acc);
    wgmma_fence();
#pragma unroll
    for (int k = 0; k < 4; k++) {
        uint64_t dA = make_desc_sw128_MN(sP_sw     + k * 1024);    // swizzled Major::MN A (Pᵀ/dSᵀ)
        uint64_t dB = make_desc_sw128_MN(B_sw_half + k * 1024);    // swizzled Major::MN B (dO/Q)
        wgmma_m64n64k16_tAtB(acc, dA, dB);                         // trans-a=1, trans-b=1
    }
    wgmma_commit();
    wgmma_wait0();
    fence_operandN<32>(acc);
}


template<int Br, int Bc, int D>
__global__ void
gqa_backward_v16_kv(
    const __grid_constant__ CUtensorMap tma_K_sw,
    const __grid_constant__ CUtensorMap tma_V_sw,
    const __grid_constant__ CUtensorMap tma_Q_sw,
    const __grid_constant__ CUtensorMap tma_dO_sw,
    const __grid_constant__ CUtensorMap tma_dO_pl,
    const __grid_constant__ CUtensorMap tma_O_pl,
    const float * __restrict__ d_LSE,
    bf16 * __restrict__ d_dK, bf16 * __restrict__ d_dV, float * __restrict__ d_dq_accum,
    int B, int Hq, int Hkv, int G, int S, float scale
) {
    static_assert(Br == 64 && Bc == 64 && D == 128, "V16 requires Br=Bc=64, D=128");
    constexpr int CONS = 256;   // consumer thread count (wg 0+1)

    __shared__ __align__(128)  bf16 sK_sw[Bc * D];
    __shared__ __align__(128)  bf16 sV_sw[Bc * D];
    __shared__ __align__(128)  bf16 sQ_sw [2][Br * D];
    __shared__ __align__(128)  bf16 sdO_sw[2][Br * D];
    __shared__ __align__(128)  bf16 sdO_pl[2][Br * D];
    __shared__ __align__(128)  bf16 sO_pl [2][Br * D];
    __shared__ __align__(16)   float sS [Br * SS_STRIDE_V6];   // V13: S store DROPPED; kept as wg0 dQ stage
    __shared__ __align__(16)   float sdP[Br * SS_STRIDE_V6];
    __shared__ __align__(1024) bf16  sP [Br * 64];             // V16: 64-wide 128-B swizzle atom, 1024-B aligned
    __shared__                 float sLSE[Br];
    __shared__                 float sD  [2][Br];              // V13: double-buffered, producer-written
    __shared__ __align__(128)  bf16  sA_t[SMEM_TILED_V6];      // dQ stage + epilogue writeback stage
    __shared__ __align__(8)    uint64_t mbar_kv;
    __shared__ __align__(8)    uint64_t full   [2];
    __shared__ __align__(8)    uint64_t empty  [2];
    __shared__ __align__(8)    uint64_t d_ready[2];            // V13: producer→consumer D signal

    const int tid   = threadIdx.x;
    const int wg    = tid >> 7;
    const int wtid  = tid & 127;
    const int lane  = tid & 31;
    const int b = blockIdx.x, hkv = blockIdx.y, k_tile = blockIdx.z;
    const int k_row0 = k_tile * Bc;
    const int nQTiles = S / Br;

    const long     kvBase     = ((long)(b * Hkv + hkv) * S + k_row0) * D;
    const uint32_t kvFlatRow  = (uint32_t)((b * Hkv + hkv) * S + k_row0);
    const uint32_t bytesTile  = (uint32_t)(Br * D * sizeof(bf16));
    const uint32_t bytesAtom  = (uint32_t)(Bc * 64 * sizeof(bf16));

    if (tid == 0) {
        mbar_init_v4(&mbar_kv, 1);
        mbar_init_v4(&full[0], 1);    mbar_init_v4(&full[1], 1);
        mbar_init_v4(&empty[0], 1);   mbar_init_v4(&empty[1], 1);
        mbar_init_v4(&d_ready[0], 1); mbar_init_v4(&d_ready[1], 1);
    }
    __syncthreads();

    const int qc0    = k_row0 / Br;
    const int perG   = nQTiles - qc0;
    const int nIter  = G * perG;

    // V15: (g,qc) maintained incrementally in each loop → no per-iter IDIV/IREM.
    auto qFlatRowOf = [&](int g, int qc) -> uint32_t {
        const int hq = hkv * G + g;
        return (uint32_t)((b * Hq + hq) * S + qc * Br);
    };
    auto lBaseOf = [&](int g, int qc) -> long {
        const int hq = hkv * G + g;
        return (long)(b * Hq + hq) * S + (long)qc * Br;
    };

    // ── PRODUCER (wg 2): TMA loads + LAGGED D-rowsum (PART A) ─────────────────
    if (wg == 2) {
        const bool leader = (tid == 256);
        const int pwarp   = (tid - 256) >> 5;   // 0..3 within the producer warpgroup
        if (leader) {
            mbar_expect_tx_v4(&mbar_kv, bytesAtom * 4);
            tma_load_2d_v4(&tma_K_sw, sK_sw,           &mbar_kv, 0,  kvFlatRow);
            tma_load_2d_v4(&tma_K_sw, sK_sw + 64 * 64, &mbar_kv, 64, kvFlatRow);
            tma_load_2d_v4(&tma_V_sw, sV_sw,           &mbar_kv, 0,  kvFlatRow);
            tma_load_2d_v4(&tma_V_sw, sV_sw + 64 * 64, &mbar_kv, 64, kvFlatRow);
        }
        uint32_t epar[2] = {0, 0}, fpar[2] = {0, 0};
        int gP = 0, qcP = qc0;   // V15 incremental (g,qc)
        // Compute D for tile `td` (waits its TMA, reduces, arrives d_ready).
        auto do_drowsum = [&](int td) {
            const int sp = td & 1;
            mbar_wait_v4(&full[sp], fpar[sp]); fpar[sp] ^= 1;
            producer_drowsum_v13<Br, D>(sD[sp], sdO_pl[sp], sO_pl[sp], pwarp, lane);
            producer_sync();
            if (leader) mbar_arrive_v11(&d_ready[sp]);
        };
        for (int it = 0; it < nIter; it++) {
            const int s = it & 1;
            if (it >= 2) { mbar_wait_v4(&empty[s], epar[s]); epar[s] ^= 1; }
            if (leader) {
                const uint32_t r = qFlatRowOf(gP, qcP);
                mbar_expect_tx_v4(&full[s], bytesTile * 4);
                tma_load_2d_v4(&tma_Q_sw,  sQ_sw [s],           &full[s], 0,  r);
                tma_load_2d_v4(&tma_Q_sw,  sQ_sw [s] + 64 * 64, &full[s], 64, r);
                tma_load_2d_v4(&tma_dO_sw, sdO_sw[s],           &full[s], 0,  r);
                tma_load_2d_v4(&tma_dO_sw, sdO_sw[s] + 64 * 64, &full[s], 64, r);
                tma_load_2d_v4(&tma_dO_pl, sdO_pl[s],           &full[s], 0,  r);
                tma_load_2d_v4(&tma_O_pl,  sO_pl [s],           &full[s], 0,  r);
            }
            if (it >= 1) do_drowsum(it - 1);   // lagged one tile → 2 TMAs in flight
            if (++qcP == nQTiles) { qcP = qc0; ++gP; }
        }
        do_drowsum(nIter - 1);                 // tail: final tile's D
        return;
    }

    // ── CONSUMERS (wg 0,1) ───────────────────────────────────────────────────
    mbar_wait_v4(&mbar_kv, 0);

    float dv[32]; zeroN<32>(dv);
    float dk[32]; zeroN<32>(dk);

    uint32_t cpar[2] = {0, 0}, dpar[2] = {0, 0};
    int gC = 0, qcC = qc0;   // V15 incremental (g,qc)
    for (int it = 0; it < nIter; it++) {
        const int s = it & 1;
        const int q_row0 = qcC * Br;
        mbar_wait_v4(&full[s], cpar[s]); cpar[s] ^= 1;
        if (tid < Br) sLSE[tid] = d_LSE[lBaseOf(gC, qcC) + tid];
        consumer_sync();

        // S = Q·Kᵀ·scale → P = exp(S − LSE)+causal DIRECT to swizzled sP (wg0, PART B)
        //  ∥  dP = dO·Vᵀ → sdP (wg1).   [D-rowsum now on the producer, PART A.]
        if (wg == 0) {
            float acc[32]; zeroN<32>(acc);
            run_gemm_n64_sw2(acc, sQ_sw[s], sK_sw);
            fused_p_from_acc_v16<Bc>(acc, sP, sLSE, wtid, q_row0, k_row0, scale);
        } else {
            float acc[32]; zeroN<32>(acc);
            run_gemm_n64_sw2(acc, sdO_sw[s], sV_sw);
            store_acc_smem_v6<Bc, SS_STRIDE_V6>(acc, sdP, wtid, 1.0f);
        }
        consumer_sync();   // wg0's swizzled sP writes visible cross-warp before the transposed reads

        // dV += Pᵀ·dO  — TRANSPOSE-ELIMINATED: A = Pᵀ read DIRECTLY from swizzled sP.
        run_gemm_dVdK_half_te(dv, sP, sdO_sw[s] + wg * 4096);
        consumer_sync();

        // dS = P ⊙ (dP − D) → swizzled sP.  D produced by wg2 → wait d_ready[s] first.
        mbar_wait_v4(&d_ready[s], dpar[s]); dpar[s] ^= 1;
        for (int i = tid; i < Br * Bc; i += CONS) {
            int r = i / Bc, c = i % Bc;
            int pidx = sw128_idx(r, c);
            sP[pidx] = __float2bfloat16(__bfloat162float(sP[pidx]) * (sdP[r * SS_STRIDE_V6 + c] - sD[s][r]));
        }
        consumer_sync();   // dS writes visible cross-warp before the transposed dK read

        // dK += dSᵀ·Q  — TRANSPOSE-ELIMINATED: A = dSᵀ read DIRECTLY from swizzled sP.
        run_gemm_dVdK_half_te(dk, sP, sQ_sw[s] + wg * 4096);
        consumer_sync();
        if (tid == 0) mbar_arrive_v11(&empty[s]);

        // dQ_tile = dS·K (transient) — dS staged K-major into sA_t from swizzled sP.
        sp_to_sAt_v16_dq(sA_t, sP, tid);
        consumer_sync();
        { float acc[32]; zeroN<32>(acc);
          run_gemm_dQ_half(acc, sA_t, sK_sw + wg * 4096, sbo_pad_v6(Bc));
          stage_acc_f32<64>(acc, (wg == 0) ? sS : sdP, wtid, scale); }
        consumer_sync();
        atomic_flush_stage<Br, 64>((wg == 0) ? sS : sdP, d_dq_accum, lBaseOf(gC, qcC) * D, D, wg * 64, wtid);
        consumer_sync();
        if (++qcC == nQTiles) { qcC = qc0; ++gC; }
    }

    // ── Epilogue — coalesced dV/dK writeback (identical to V11/V12). ──
    bf16 *stage = sA_t + wg * 4096;
    fence_operandN<32>(dv);
    stage_acc_bf16<64>(dv, stage, wtid, 1.0f);
    consumer_sync();
    store_stage_vec<Bc, 64>(stage, d_dV, kvBase, D, wg * 64, wtid);
    consumer_sync();
    fence_operandN<32>(dk);
    stage_acc_bf16<64>(dk, stage, wtid, scale);
    consumer_sync();
    store_stage_vec<Bc, 64>(stage, d_dK, kvBase, D, wg * 64, wtid);
}

// ── V16 launcher — identical to V15 (TMA descriptors, dQ scratch); only the kernel
// symbol changes.  sP swizzled-atom layout is internal to the kernel.
template<int Br, int Bc, int D>
void launch_gqa_backward_v16(
    const bf16 *d_Q, const bf16 *d_K, const bf16 *d_V, const bf16 *d_O,
    const bf16 *d_dO, const float *d_LSE,
    bf16 *d_dQ, bf16 *d_dK, bf16 *d_dV,
    int B, int Hq, int Hkv, int G, int S, float scale
) {
    static_assert(Br == 64 && Bc == 64 && D == 128, "V16 requires Br=Bc=64, D=128");

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
    CUtensorMap tma_V_sw  = make_tma_sw128(d_V,  Rkv, Bc);
    CUtensorMap tma_Q_sw  = make_tma_sw128(d_Q,  Rq,  Br);
    CUtensorMap tma_dO_sw = make_tma_sw128(d_dO, Rq,  Br);
    CUtensorMap tma_dO_pl = make_tma_plain(d_dO, Rq,  Br);
    CUtensorMap tma_O_pl  = make_tma_plain(d_O,  Rq,  Br);

    const long dqN = (long)B * Hq * S * D;
    static float* d_dq_accum = nullptr;
    static long   dq_cap     = 0;
    if (dqN > dq_cap) {
        if (d_dq_accum) CUDA_CHECK(cudaFree(d_dq_accum));
        CUDA_CHECK(cudaMalloc(&d_dq_accum, dqN * sizeof(float)));
        dq_cap = dqN;
    }
    CUDA_CHECK(cudaMemset(d_dq_accum, 0, dqN * sizeof(float)));

    constexpr dim3 BLOCK(384);
    dim3 GRID(B, Hkv, S / Bc);
    gqa_backward_v16_kv<Br,Bc,D><<<GRID, BLOCK>>>(
        tma_K_sw, tma_V_sw, tma_Q_sw, tma_dO_sw, tma_dO_pl, tma_O_pl,
        d_LSE, d_dK, d_dV, d_dq_accum, B, Hq, Hkv, G, S, scale);

    const int convBlock = 256;
    const int convGrid  = (int)((dqN + convBlock - 1) / convBlock);
    convert_dq_accum_to_bf16_v5<<<convGrid, convBlock>>>(d_dq_accum, d_dQ, dqN);
}

// ─────────────────────────────────────────────────────────────────────────────
// V17 — V16 clone; hoist swizzle-index invariants on the sP write path (fused_p +
// dS write) to trim the vector-ALU the swizzled store added. Bit-identical to V16.
// ─────────────────────────────────────────────────────────────────────────────
// wg0 fused-P (V16 swizzled write) with V17 hoisted swizzle-index:
// sw128_idx(r,c)=r*64+(((c>>3)^(r&7))<<3)+(c&7); c=nt*8+cc, cc<8 ⇒ c>>3=nt (compile-time),
// c&7=cc ⇒ b = (r*64+cc) + ((nt^(r&7))<<3).  Bit-identical to fused_p_from_acc_v16.
template<int Bc>
__device__ __forceinline__ void fused_p_from_acc_v17(
    const float acc[32], bf16* sP, const float* sLSE, int wtid,
    int q_row0, int k_row0, float scale)
{
    const int w = wtid >> 5, lane = wtid & 31;
    const int r0 = w * 16 + (lane >> 2), r1 = r0 + 8, cc = (lane & 3) * 2;
    const float l0 = sLSE[r0], l1 = sLSE[r1];
    const int gr0 = q_row0 + r0, gr1 = q_row0 + r1;
    const int base0 = r0 * 64 + cc, m0 = r0 & 7;   // V17: hoisted swizzle-index invariants
    const int base1 = r1 * 64 + cc, m1 = r1 & 7;
#pragma unroll
    for (int nt = 0; nt < Bc / 8; nt++) {
        const int c   = nt * 8 + cc;
        const int gc0 = k_row0 + c, gc1 = gc0 + 1;
        const int b0  = base0 + ((nt ^ m0) << 3);
        const int b1  = base1 + ((nt ^ m1) << 3);
        sP[b0 + 0] = (gc0 > gr0) ? __float2bfloat16(0.f)
                                 : __float2bfloat16(__expf(acc[nt * 4 + 0] * scale - l0));
        sP[b0 + 1] = (gc1 > gr0) ? __float2bfloat16(0.f)
                                 : __float2bfloat16(__expf(acc[nt * 4 + 1] * scale - l0));
        sP[b1 + 0] = (gc0 > gr1) ? __float2bfloat16(0.f)
                                 : __float2bfloat16(__expf(acc[nt * 4 + 2] * scale - l1));
        sP[b1 + 1] = (gc1 > gr1) ? __float2bfloat16(0.f)
                                 : __float2bfloat16(__expf(acc[nt * 4 + 3] * scale - l1));
    }
}

template<int Br, int Bc, int D>
__global__ void
gqa_backward_v17_kv(
    const __grid_constant__ CUtensorMap tma_K_sw,
    const __grid_constant__ CUtensorMap tma_V_sw,
    const __grid_constant__ CUtensorMap tma_Q_sw,
    const __grid_constant__ CUtensorMap tma_dO_sw,
    const __grid_constant__ CUtensorMap tma_dO_pl,
    const __grid_constant__ CUtensorMap tma_O_pl,
    const float * __restrict__ d_LSE,
    bf16 * __restrict__ d_dK, bf16 * __restrict__ d_dV, float * __restrict__ d_dq_accum,
    int B, int Hq, int Hkv, int G, int S, float scale
) {
    static_assert(Br == 64 && Bc == 64 && D == 128, "V16 requires Br=Bc=64, D=128");
    constexpr int CONS = 256;   // consumer thread count (wg 0+1)

    __shared__ __align__(128)  bf16 sK_sw[Bc * D];
    __shared__ __align__(128)  bf16 sV_sw[Bc * D];
    __shared__ __align__(128)  bf16 sQ_sw [2][Br * D];
    __shared__ __align__(128)  bf16 sdO_sw[2][Br * D];
    __shared__ __align__(128)  bf16 sdO_pl[2][Br * D];
    __shared__ __align__(128)  bf16 sO_pl [2][Br * D];
    __shared__ __align__(16)   float sS [Br * SS_STRIDE_V6];   // V13: S store DROPPED; kept as wg0 dQ stage
    __shared__ __align__(16)   float sdP[Br * SS_STRIDE_V6];
    __shared__ __align__(1024) bf16  sP [Br * 64];             // V16: 64-wide 128-B swizzle atom, 1024-B aligned
    __shared__                 float sLSE[Br];
    __shared__                 float sD  [2][Br];              // V13: double-buffered, producer-written
    __shared__ __align__(128)  bf16  sA_t[SMEM_TILED_V6];      // dQ stage + epilogue writeback stage
    __shared__ __align__(8)    uint64_t mbar_kv;
    __shared__ __align__(8)    uint64_t full   [2];
    __shared__ __align__(8)    uint64_t empty  [2];
    __shared__ __align__(8)    uint64_t d_ready[2];            // V13: producer→consumer D signal

    const int tid   = threadIdx.x;
    const int wg    = tid >> 7;
    const int wtid  = tid & 127;
    const int lane  = tid & 31;
    const int b = blockIdx.x, hkv = blockIdx.y, k_tile = blockIdx.z;
    const int k_row0 = k_tile * Bc;
    const int nQTiles = S / Br;

    const long     kvBase     = ((long)(b * Hkv + hkv) * S + k_row0) * D;
    const uint32_t kvFlatRow  = (uint32_t)((b * Hkv + hkv) * S + k_row0);
    const uint32_t bytesTile  = (uint32_t)(Br * D * sizeof(bf16));
    const uint32_t bytesAtom  = (uint32_t)(Bc * 64 * sizeof(bf16));

    if (tid == 0) {
        mbar_init_v4(&mbar_kv, 1);
        mbar_init_v4(&full[0], 1);    mbar_init_v4(&full[1], 1);
        mbar_init_v4(&empty[0], 1);   mbar_init_v4(&empty[1], 1);
        mbar_init_v4(&d_ready[0], 1); mbar_init_v4(&d_ready[1], 1);
    }
    __syncthreads();

    const int qc0    = k_row0 / Br;
    const int perG   = nQTiles - qc0;
    const int nIter  = G * perG;

    // V15: (g,qc) maintained incrementally in each loop → no per-iter IDIV/IREM.
    auto qFlatRowOf = [&](int g, int qc) -> uint32_t {
        const int hq = hkv * G + g;
        return (uint32_t)((b * Hq + hq) * S + qc * Br);
    };
    auto lBaseOf = [&](int g, int qc) -> long {
        const int hq = hkv * G + g;
        return (long)(b * Hq + hq) * S + (long)qc * Br;
    };

    // ── PRODUCER (wg 2): TMA loads + LAGGED D-rowsum (PART A) ─────────────────
    if (wg == 2) {
        const bool leader = (tid == 256);
        const int pwarp   = (tid - 256) >> 5;   // 0..3 within the producer warpgroup
        if (leader) {
            mbar_expect_tx_v4(&mbar_kv, bytesAtom * 4);
            tma_load_2d_v4(&tma_K_sw, sK_sw,           &mbar_kv, 0,  kvFlatRow);
            tma_load_2d_v4(&tma_K_sw, sK_sw + 64 * 64, &mbar_kv, 64, kvFlatRow);
            tma_load_2d_v4(&tma_V_sw, sV_sw,           &mbar_kv, 0,  kvFlatRow);
            tma_load_2d_v4(&tma_V_sw, sV_sw + 64 * 64, &mbar_kv, 64, kvFlatRow);
        }
        uint32_t epar[2] = {0, 0}, fpar[2] = {0, 0};
        int gP = 0, qcP = qc0;   // V15 incremental (g,qc)
        // Compute D for tile `td` (waits its TMA, reduces, arrives d_ready).
        auto do_drowsum = [&](int td) {
            const int sp = td & 1;
            mbar_wait_v4(&full[sp], fpar[sp]); fpar[sp] ^= 1;
            producer_drowsum_v13<Br, D>(sD[sp], sdO_pl[sp], sO_pl[sp], pwarp, lane);
            producer_sync();
            if (leader) mbar_arrive_v11(&d_ready[sp]);
        };
        for (int it = 0; it < nIter; it++) {
            const int s = it & 1;
            if (it >= 2) { mbar_wait_v4(&empty[s], epar[s]); epar[s] ^= 1; }
            if (leader) {
                const uint32_t r = qFlatRowOf(gP, qcP);
                mbar_expect_tx_v4(&full[s], bytesTile * 4);
                tma_load_2d_v4(&tma_Q_sw,  sQ_sw [s],           &full[s], 0,  r);
                tma_load_2d_v4(&tma_Q_sw,  sQ_sw [s] + 64 * 64, &full[s], 64, r);
                tma_load_2d_v4(&tma_dO_sw, sdO_sw[s],           &full[s], 0,  r);
                tma_load_2d_v4(&tma_dO_sw, sdO_sw[s] + 64 * 64, &full[s], 64, r);
                tma_load_2d_v4(&tma_dO_pl, sdO_pl[s],           &full[s], 0,  r);
                tma_load_2d_v4(&tma_O_pl,  sO_pl [s],           &full[s], 0,  r);
            }
            if (it >= 1) do_drowsum(it - 1);   // lagged one tile → 2 TMAs in flight
            if (++qcP == nQTiles) { qcP = qc0; ++gP; }
        }
        do_drowsum(nIter - 1);                 // tail: final tile's D
        return;
    }

    // ── CONSUMERS (wg 0,1) ───────────────────────────────────────────────────
    mbar_wait_v4(&mbar_kv, 0);

    float dv[32]; zeroN<32>(dv);
    float dk[32]; zeroN<32>(dk);

    uint32_t cpar[2] = {0, 0}, dpar[2] = {0, 0};
    int gC = 0, qcC = qc0;   // V15 incremental (g,qc)
    for (int it = 0; it < nIter; it++) {
        const int s = it & 1;
        const int q_row0 = qcC * Br;
        mbar_wait_v4(&full[s], cpar[s]); cpar[s] ^= 1;
        if (tid < Br) sLSE[tid] = d_LSE[lBaseOf(gC, qcC) + tid];
        consumer_sync();

        // S = Q·Kᵀ·scale → P = exp(S − LSE)+causal DIRECT to swizzled sP (wg0, PART B)
        //  ∥  dP = dO·Vᵀ → sdP (wg1).   [D-rowsum now on the producer, PART A.]
        if (wg == 0) {
            float acc[32]; zeroN<32>(acc);
            run_gemm_n64_sw2(acc, sQ_sw[s], sK_sw);
            fused_p_from_acc_v17<Bc>(acc, sP, sLSE, wtid, q_row0, k_row0, scale);
        } else {
            float acc[32]; zeroN<32>(acc);
            run_gemm_n64_sw2(acc, sdO_sw[s], sV_sw);
            store_acc_smem_v6<Bc, SS_STRIDE_V6>(acc, sdP, wtid, 1.0f);
        }
        consumer_sync();   // wg0's swizzled sP writes visible cross-warp before the transposed reads

        // dV += Pᵀ·dO  — TRANSPOSE-ELIMINATED: A = Pᵀ read DIRECTLY from swizzled sP.
        run_gemm_dVdK_half_te(dv, sP, sdO_sw[s] + wg * 4096);
        consumer_sync();

        // dS = P ⊙ (dP − D) → swizzled sP.  D produced by wg2 → wait d_ready[s] first.
        mbar_wait_v4(&d_ready[s], dpar[s]); dpar[s] ^= 1;
        {   // V17: c = i%Bc invariant (CONS%Bc==0) → hoist c/c8/clo out of the loop.
            const int c = tid % Bc, c8 = c >> 3, clo = c & 7;
            for (int i = tid; i < Br * Bc; i += CONS) {
                const int r = i / Bc;
                const int pidx = r * 64 + ((c8 ^ (r & 7)) << 3) + clo;
                sP[pidx] = __float2bfloat16(__bfloat162float(sP[pidx]) * (sdP[r * SS_STRIDE_V6 + c] - sD[s][r]));
            }
        }
        consumer_sync();   // dS writes visible cross-warp before the transposed dK read

        // dK += dSᵀ·Q  — TRANSPOSE-ELIMINATED: A = dSᵀ read DIRECTLY from swizzled sP.
        run_gemm_dVdK_half_te(dk, sP, sQ_sw[s] + wg * 4096);
        consumer_sync();
        if (tid == 0) mbar_arrive_v11(&empty[s]);

        // dQ_tile = dS·K (transient) — dS staged K-major into sA_t from swizzled sP.
        sp_to_sAt_v16_dq(sA_t, sP, tid);
        consumer_sync();
        { float acc[32]; zeroN<32>(acc);
          run_gemm_dQ_half(acc, sA_t, sK_sw + wg * 4096, sbo_pad_v6(Bc));
          stage_acc_f32<64>(acc, (wg == 0) ? sS : sdP, wtid, scale); }
        consumer_sync();
        atomic_flush_stage<Br, 64>((wg == 0) ? sS : sdP, d_dq_accum, lBaseOf(gC, qcC) * D, D, wg * 64, wtid);
        consumer_sync();
        if (++qcC == nQTiles) { qcC = qc0; ++gC; }
    }

    // ── Epilogue — coalesced dV/dK writeback (identical to V11/V12). ──
    bf16 *stage = sA_t + wg * 4096;
    fence_operandN<32>(dv);
    stage_acc_bf16<64>(dv, stage, wtid, 1.0f);
    consumer_sync();
    store_stage_vec<Bc, 64>(stage, d_dV, kvBase, D, wg * 64, wtid);
    consumer_sync();
    fence_operandN<32>(dk);
    stage_acc_bf16<64>(dk, stage, wtid, scale);
    consumer_sync();
    store_stage_vec<Bc, 64>(stage, d_dK, kvBase, D, wg * 64, wtid);
}

// ── V17 launcher — identical to V15 (TMA descriptors, dQ scratch); only the kernel
// symbol changes.  sP swizzled-atom layout is internal to the kernel.
template<int Br, int Bc, int D>
void launch_gqa_backward_v17(
    const bf16 *d_Q, const bf16 *d_K, const bf16 *d_V, const bf16 *d_O,
    const bf16 *d_dO, const float *d_LSE,
    bf16 *d_dQ, bf16 *d_dK, bf16 *d_dV,
    int B, int Hq, int Hkv, int G, int S, float scale
) {
    static_assert(Br == 64 && Bc == 64 && D == 128, "V16 requires Br=Bc=64, D=128");

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
    CUtensorMap tma_V_sw  = make_tma_sw128(d_V,  Rkv, Bc);
    CUtensorMap tma_Q_sw  = make_tma_sw128(d_Q,  Rq,  Br);
    CUtensorMap tma_dO_sw = make_tma_sw128(d_dO, Rq,  Br);
    CUtensorMap tma_dO_pl = make_tma_plain(d_dO, Rq,  Br);
    CUtensorMap tma_O_pl  = make_tma_plain(d_O,  Rq,  Br);

    const long dqN = (long)B * Hq * S * D;
    static float* d_dq_accum = nullptr;
    static long   dq_cap     = 0;
    if (dqN > dq_cap) {
        if (d_dq_accum) CUDA_CHECK(cudaFree(d_dq_accum));
        CUDA_CHECK(cudaMalloc(&d_dq_accum, dqN * sizeof(float)));
        dq_cap = dqN;
    }
    CUDA_CHECK(cudaMemset(d_dq_accum, 0, dqN * sizeof(float)));

    constexpr dim3 BLOCK(384);
    dim3 GRID(B, Hkv, S / Bc);
    gqa_backward_v17_kv<Br,Bc,D><<<GRID, BLOCK>>>(
        tma_K_sw, tma_V_sw, tma_Q_sw, tma_dO_sw, tma_dO_pl, tma_O_pl,
        d_LSE, d_dK, d_dV, d_dq_accum, B, Hq, Hkv, G, S, scale);

    const int convBlock = 256;
    const int convGrid  = (int)((dqN + convBlock - 1) / convBlock);
    convert_dq_accum_to_bf16_v5<<<convGrid, convBlock>>>(d_dq_accum, d_dQ, dqN);
}

// ─────────────────────────────────────────────────────────────────────────────
// V18 — V17 clone; VECTORIZE dS=P⊙(dP−D) (bf16×2/float2, 2 adjacent cols/step) to
// halve the cvt→mul→cvt RAW chain (the #2 'wait' stall).  Bit-identical to V17.
// ─────────────────────────────────────────────────────────────────────────────
template<int Br, int Bc, int D>
__global__ void
gqa_backward_v18_kv(
    const __grid_constant__ CUtensorMap tma_K_sw,
    const __grid_constant__ CUtensorMap tma_V_sw,
    const __grid_constant__ CUtensorMap tma_Q_sw,
    const __grid_constant__ CUtensorMap tma_dO_sw,
    const __grid_constant__ CUtensorMap tma_dO_pl,
    const __grid_constant__ CUtensorMap tma_O_pl,
    const float * __restrict__ d_LSE,
    bf16 * __restrict__ d_dK, bf16 * __restrict__ d_dV, float * __restrict__ d_dq_accum,
    int B, int Hq, int Hkv, int G, int S, float scale
) {
    static_assert(Br == 64 && Bc == 64 && D == 128, "V16 requires Br=Bc=64, D=128");
    constexpr int CONS = 256;   // consumer thread count (wg 0+1)

    __shared__ __align__(128)  bf16 sK_sw[Bc * D];
    __shared__ __align__(128)  bf16 sV_sw[Bc * D];
    __shared__ __align__(128)  bf16 sQ_sw [2][Br * D];
    __shared__ __align__(128)  bf16 sdO_sw[2][Br * D];
    __shared__ __align__(128)  bf16 sdO_pl[2][Br * D];
    __shared__ __align__(128)  bf16 sO_pl [2][Br * D];
    __shared__ __align__(16)   float sS [Br * SS_STRIDE_V6];   // V13: S store DROPPED; kept as wg0 dQ stage
    __shared__ __align__(16)   float sdP[Br * SS_STRIDE_V6];
    __shared__ __align__(1024) bf16  sP [Br * 64];             // V16: 64-wide 128-B swizzle atom, 1024-B aligned
    __shared__                 float sLSE[Br];
    __shared__                 float sD  [2][Br];              // V13: double-buffered, producer-written
    __shared__ __align__(128)  bf16  sA_t[SMEM_TILED_V6];      // dQ stage + epilogue writeback stage
    __shared__ __align__(8)    uint64_t mbar_kv;
    __shared__ __align__(8)    uint64_t full   [2];
    __shared__ __align__(8)    uint64_t empty  [2];
    __shared__ __align__(8)    uint64_t d_ready[2];            // V13: producer→consumer D signal

    const int tid   = threadIdx.x;
    const int wg    = tid >> 7;
    const int wtid  = tid & 127;
    const int lane  = tid & 31;
    const int b = blockIdx.x, hkv = blockIdx.y, k_tile = blockIdx.z;
    const int k_row0 = k_tile * Bc;
    const int nQTiles = S / Br;

    const long     kvBase     = ((long)(b * Hkv + hkv) * S + k_row0) * D;
    const uint32_t kvFlatRow  = (uint32_t)((b * Hkv + hkv) * S + k_row0);
    const uint32_t bytesTile  = (uint32_t)(Br * D * sizeof(bf16));
    const uint32_t bytesAtom  = (uint32_t)(Bc * 64 * sizeof(bf16));

    if (tid == 0) {
        mbar_init_v4(&mbar_kv, 1);
        mbar_init_v4(&full[0], 1);    mbar_init_v4(&full[1], 1);
        mbar_init_v4(&empty[0], 1);   mbar_init_v4(&empty[1], 1);
        mbar_init_v4(&d_ready[0], 1); mbar_init_v4(&d_ready[1], 1);
    }
    __syncthreads();

    const int qc0    = k_row0 / Br;
    const int perG   = nQTiles - qc0;
    const int nIter  = G * perG;

    // V15: (g,qc) maintained incrementally in each loop → no per-iter IDIV/IREM.
    auto qFlatRowOf = [&](int g, int qc) -> uint32_t {
        const int hq = hkv * G + g;
        return (uint32_t)((b * Hq + hq) * S + qc * Br);
    };
    auto lBaseOf = [&](int g, int qc) -> long {
        const int hq = hkv * G + g;
        return (long)(b * Hq + hq) * S + (long)qc * Br;
    };

    // ── PRODUCER (wg 2): TMA loads + LAGGED D-rowsum (PART A) ─────────────────
    if (wg == 2) {
        const bool leader = (tid == 256);
        const int pwarp   = (tid - 256) >> 5;   // 0..3 within the producer warpgroup
        if (leader) {
            mbar_expect_tx_v4(&mbar_kv, bytesAtom * 4);
            tma_load_2d_v4(&tma_K_sw, sK_sw,           &mbar_kv, 0,  kvFlatRow);
            tma_load_2d_v4(&tma_K_sw, sK_sw + 64 * 64, &mbar_kv, 64, kvFlatRow);
            tma_load_2d_v4(&tma_V_sw, sV_sw,           &mbar_kv, 0,  kvFlatRow);
            tma_load_2d_v4(&tma_V_sw, sV_sw + 64 * 64, &mbar_kv, 64, kvFlatRow);
        }
        uint32_t epar[2] = {0, 0}, fpar[2] = {0, 0};
        int gP = 0, qcP = qc0;   // V15 incremental (g,qc)
        // Compute D for tile `td` (waits its TMA, reduces, arrives d_ready).
        auto do_drowsum = [&](int td) {
            const int sp = td & 1;
            mbar_wait_v4(&full[sp], fpar[sp]); fpar[sp] ^= 1;
            producer_drowsum_v13<Br, D>(sD[sp], sdO_pl[sp], sO_pl[sp], pwarp, lane);
            producer_sync();
            if (leader) mbar_arrive_v11(&d_ready[sp]);
        };
        for (int it = 0; it < nIter; it++) {
            const int s = it & 1;
            if (it >= 2) { mbar_wait_v4(&empty[s], epar[s]); epar[s] ^= 1; }
            if (leader) {
                const uint32_t r = qFlatRowOf(gP, qcP);
                mbar_expect_tx_v4(&full[s], bytesTile * 4);
                tma_load_2d_v4(&tma_Q_sw,  sQ_sw [s],           &full[s], 0,  r);
                tma_load_2d_v4(&tma_Q_sw,  sQ_sw [s] + 64 * 64, &full[s], 64, r);
                tma_load_2d_v4(&tma_dO_sw, sdO_sw[s],           &full[s], 0,  r);
                tma_load_2d_v4(&tma_dO_sw, sdO_sw[s] + 64 * 64, &full[s], 64, r);
                tma_load_2d_v4(&tma_dO_pl, sdO_pl[s],           &full[s], 0,  r);
                tma_load_2d_v4(&tma_O_pl,  sO_pl [s],           &full[s], 0,  r);
            }
            if (it >= 1) do_drowsum(it - 1);   // lagged one tile → 2 TMAs in flight
            if (++qcP == nQTiles) { qcP = qc0; ++gP; }
        }
        do_drowsum(nIter - 1);                 // tail: final tile's D
        return;
    }

    // ── CONSUMERS (wg 0,1) ───────────────────────────────────────────────────
    mbar_wait_v4(&mbar_kv, 0);

    float dv[32]; zeroN<32>(dv);
    float dk[32]; zeroN<32>(dk);

    uint32_t cpar[2] = {0, 0}, dpar[2] = {0, 0};
    int gC = 0, qcC = qc0;   // V15 incremental (g,qc)
    for (int it = 0; it < nIter; it++) {
        const int s = it & 1;
        const int q_row0 = qcC * Br;
        mbar_wait_v4(&full[s], cpar[s]); cpar[s] ^= 1;
        if (tid < Br) sLSE[tid] = d_LSE[lBaseOf(gC, qcC) + tid];
        consumer_sync();

        // S = Q·Kᵀ·scale → P = exp(S − LSE)+causal DIRECT to swizzled sP (wg0, PART B)
        //  ∥  dP = dO·Vᵀ → sdP (wg1).   [D-rowsum now on the producer, PART A.]
        if (wg == 0) {
            float acc[32]; zeroN<32>(acc);
            run_gemm_n64_sw2(acc, sQ_sw[s], sK_sw);
            fused_p_from_acc_v17<Bc>(acc, sP, sLSE, wtid, q_row0, k_row0, scale);
        } else {
            float acc[32]; zeroN<32>(acc);
            run_gemm_n64_sw2(acc, sdO_sw[s], sV_sw);
            store_acc_smem_v6<Bc, SS_STRIDE_V6>(acc, sdP, wtid, 1.0f);
        }
        consumer_sync();   // wg0's swizzled sP writes visible cross-warp before the transposed reads

        // dV += Pᵀ·dO  — TRANSPOSE-ELIMINATED: A = Pᵀ read DIRECTLY from swizzled sP.
        run_gemm_dVdK_half_te(dv, sP, sdO_sw[s] + wg * 4096);
        consumer_sync();

        // dS = P ⊙ (dP − D) → swizzled sP.  D produced by wg2 → wait d_ready[s] first.
        mbar_wait_v4(&d_ready[s], dpar[s]); dpar[s] ^= 1;
        {   // V18: vectorize dS — 2 adjacent columns/step.  cbase=(2·tid)%Bc is even ⇒
            // sP[pidx],sP[pidx+1] adjacent in the swizzle (cbase,cbase+1 same 8-group) ⇒
            // bf16×2 load/store + float2 arith, halving the cvt→mul→cvt RAW chain.  Each
            // element computed once with identical ops → bit-identical to V17.
            const int cbase = (2 * tid) % Bc, c8 = cbase >> 3, clo = cbase & 7;
            for (int pp = tid; pp < Br * Bc / 2; pp += CONS) {
                const int r    = (2 * pp) / Bc;
                const int pidx = r * 64 + ((c8 ^ (r & 7)) << 3) + clo;
                const int sdi  = r * SS_STRIDE_V6 + cbase;
                const float d  = sD[s][r];
                const __nv_bfloat162 p2 = *reinterpret_cast<const __nv_bfloat162*>(&sP[pidx]);
                const float2 pf  = __bfloat1622float2(p2);
                const float2 res = make_float2(pf.x * (sdP[sdi] - d), pf.y * (sdP[sdi + 1] - d));
                *reinterpret_cast<__nv_bfloat162*>(&sP[pidx]) = __float22bfloat162_rn(res);
            }
        }
        consumer_sync();   // dS writes visible cross-warp before the transposed dK read

        // dK += dSᵀ·Q  — TRANSPOSE-ELIMINATED: A = dSᵀ read DIRECTLY from swizzled sP.
        run_gemm_dVdK_half_te(dk, sP, sQ_sw[s] + wg * 4096);
        consumer_sync();
        if (tid == 0) mbar_arrive_v11(&empty[s]);

        // dQ_tile = dS·K (transient) — dS staged K-major into sA_t from swizzled sP.
        sp_to_sAt_v16_dq(sA_t, sP, tid);
        consumer_sync();
        { float acc[32]; zeroN<32>(acc);
          run_gemm_dQ_half(acc, sA_t, sK_sw + wg * 4096, sbo_pad_v6(Bc));
          stage_acc_f32<64>(acc, (wg == 0) ? sS : sdP, wtid, scale); }
        consumer_sync();
        atomic_flush_stage<Br, 64>((wg == 0) ? sS : sdP, d_dq_accum, lBaseOf(gC, qcC) * D, D, wg * 64, wtid);
        consumer_sync();
        if (++qcC == nQTiles) { qcC = qc0; ++gC; }
    }

    // ── Epilogue — coalesced dV/dK writeback (identical to V11/V12). ──
    bf16 *stage = sA_t + wg * 4096;
    fence_operandN<32>(dv);
    stage_acc_bf16<64>(dv, stage, wtid, 1.0f);
    consumer_sync();
    store_stage_vec<Bc, 64>(stage, d_dV, kvBase, D, wg * 64, wtid);
    consumer_sync();
    fence_operandN<32>(dk);
    stage_acc_bf16<64>(dk, stage, wtid, scale);
    consumer_sync();
    store_stage_vec<Bc, 64>(stage, d_dK, kvBase, D, wg * 64, wtid);
}

// ── V18 launcher — identical to V15 (TMA descriptors, dQ scratch); only the kernel
// symbol changes.  sP swizzled-atom layout is internal to the kernel.
template<int Br, int Bc, int D>
void launch_gqa_backward_v18(
    const bf16 *d_Q, const bf16 *d_K, const bf16 *d_V, const bf16 *d_O,
    const bf16 *d_dO, const float *d_LSE,
    bf16 *d_dQ, bf16 *d_dK, bf16 *d_dV,
    int B, int Hq, int Hkv, int G, int S, float scale
) {
    static_assert(Br == 64 && Bc == 64 && D == 128, "V16 requires Br=Bc=64, D=128");

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
    CUtensorMap tma_V_sw  = make_tma_sw128(d_V,  Rkv, Bc);
    CUtensorMap tma_Q_sw  = make_tma_sw128(d_Q,  Rq,  Br);
    CUtensorMap tma_dO_sw = make_tma_sw128(d_dO, Rq,  Br);
    CUtensorMap tma_dO_pl = make_tma_plain(d_dO, Rq,  Br);
    CUtensorMap tma_O_pl  = make_tma_plain(d_O,  Rq,  Br);

    const long dqN = (long)B * Hq * S * D;
    static float* d_dq_accum = nullptr;
    static long   dq_cap     = 0;
    if (dqN > dq_cap) {
        if (d_dq_accum) CUDA_CHECK(cudaFree(d_dq_accum));
        CUDA_CHECK(cudaMalloc(&d_dq_accum, dqN * sizeof(float)));
        dq_cap = dqN;
    }
    CUDA_CHECK(cudaMemset(d_dq_accum, 0, dqN * sizeof(float)));

    constexpr dim3 BLOCK(384);
    dim3 GRID(B, Hkv, S / Bc);
    gqa_backward_v18_kv<Br,Bc,D><<<GRID, BLOCK>>>(
        tma_K_sw, tma_V_sw, tma_Q_sw, tma_dO_sw, tma_dO_pl, tma_O_pl,
        d_LSE, d_dK, d_dV, d_dq_accum, B, Hq, Hkv, G, S, scale);

    const int convBlock = 256;
    const int convGrid  = (int)((dqN + convBlock - 1) / convBlock);
    convert_dq_accum_to_bf16_v5<<<convGrid, convBlock>>>(d_dq_accum, d_dQ, dqN);
}


// ─────────────────────────────────────────────────────────────────────────────
// V19 — V18 clone; VECTORIZE the fused_p P-write (bf16×2), same lever as V18's dS.
// P feeds all downstream GEMMs → critical path.  Bit-identical to V18.
// ─────────────────────────────────────────────────────────────────────────────
// V19: vectorized P-write.  Two adjacent P values per row (cols c,c+1) packed into one
// bf16×2 store; b0,b1 even ⇒ 4-byte aligned.  Bit-identical to fused_p_from_acc_v17.
template<int Bc>
__device__ __forceinline__ void fused_p_from_acc_v19(
    const float acc[32], bf16* sP, const float* sLSE, int wtid,
    int q_row0, int k_row0, float scale)
{
    const int w = wtid >> 5, lane = wtid & 31;
    const int r0 = w * 16 + (lane >> 2), r1 = r0 + 8, cc = (lane & 3) * 2;
    const float l0 = sLSE[r0], l1 = sLSE[r1];
    const int gr0 = q_row0 + r0, gr1 = q_row0 + r1;
    const int base0 = r0 * 64 + cc, m0 = r0 & 7;
    const int base1 = r1 * 64 + cc, m1 = r1 & 7;
#pragma unroll
    for (int nt = 0; nt < Bc / 8; nt++) {
        const int c   = nt * 8 + cc;
        const int gc0 = k_row0 + c, gc1 = gc0 + 1;
        const int b0  = base0 + ((nt ^ m0) << 3);
        const int b1  = base1 + ((nt ^ m1) << 3);
        const float2 p0 = make_float2((gc0 > gr0) ? 0.f : __expf(acc[nt * 4 + 0] * scale - l0),
                                      (gc1 > gr0) ? 0.f : __expf(acc[nt * 4 + 1] * scale - l0));
        const float2 p1 = make_float2((gc0 > gr1) ? 0.f : __expf(acc[nt * 4 + 2] * scale - l1),
                                      (gc1 > gr1) ? 0.f : __expf(acc[nt * 4 + 3] * scale - l1));
        *reinterpret_cast<__nv_bfloat162*>(&sP[b0]) = __float22bfloat162_rn(p0);
        *reinterpret_cast<__nv_bfloat162*>(&sP[b1]) = __float22bfloat162_rn(p1);
    }
}

template<int Br, int Bc, int D>
__global__ void
gqa_backward_v19_kv(
    const __grid_constant__ CUtensorMap tma_K_sw,
    const __grid_constant__ CUtensorMap tma_V_sw,
    const __grid_constant__ CUtensorMap tma_Q_sw,
    const __grid_constant__ CUtensorMap tma_dO_sw,
    const __grid_constant__ CUtensorMap tma_dO_pl,
    const __grid_constant__ CUtensorMap tma_O_pl,
    const float * __restrict__ d_LSE,
    bf16 * __restrict__ d_dK, bf16 * __restrict__ d_dV, float * __restrict__ d_dq_accum,
    int B, int Hq, int Hkv, int G, int S, float scale
) {
    static_assert(Br == 64 && Bc == 64 && D == 128, "V16 requires Br=Bc=64, D=128");
    constexpr int CONS = 256;   // consumer thread count (wg 0+1)

    __shared__ __align__(128)  bf16 sK_sw[Bc * D];
    __shared__ __align__(128)  bf16 sV_sw[Bc * D];
    __shared__ __align__(128)  bf16 sQ_sw [2][Br * D];
    __shared__ __align__(128)  bf16 sdO_sw[2][Br * D];
    __shared__ __align__(128)  bf16 sdO_pl[2][Br * D];
    __shared__ __align__(128)  bf16 sO_pl [2][Br * D];
    __shared__ __align__(16)   float sS [Br * SS_STRIDE_V6];   // V13: S store DROPPED; kept as wg0 dQ stage
    __shared__ __align__(16)   float sdP[Br * SS_STRIDE_V6];
    __shared__ __align__(1024) bf16  sP [Br * 64];             // V16: 64-wide 128-B swizzle atom, 1024-B aligned
    __shared__                 float sLSE[Br];
    __shared__                 float sD  [2][Br];              // V13: double-buffered, producer-written
    __shared__ __align__(128)  bf16  sA_t[SMEM_TILED_V6];      // dQ stage + epilogue writeback stage
    __shared__ __align__(8)    uint64_t mbar_kv;
    __shared__ __align__(8)    uint64_t full   [2];
    __shared__ __align__(8)    uint64_t empty  [2];
    __shared__ __align__(8)    uint64_t d_ready[2];            // V13: producer→consumer D signal

    const int tid   = threadIdx.x;
    const int wg    = tid >> 7;
    const int wtid  = tid & 127;
    const int lane  = tid & 31;
    const int b = blockIdx.x, hkv = blockIdx.y, k_tile = blockIdx.z;
    const int k_row0 = k_tile * Bc;
    const int nQTiles = S / Br;

    const long     kvBase     = ((long)(b * Hkv + hkv) * S + k_row0) * D;
    const uint32_t kvFlatRow  = (uint32_t)((b * Hkv + hkv) * S + k_row0);
    const uint32_t bytesTile  = (uint32_t)(Br * D * sizeof(bf16));
    const uint32_t bytesAtom  = (uint32_t)(Bc * 64 * sizeof(bf16));

    if (tid == 0) {
        mbar_init_v4(&mbar_kv, 1);
        mbar_init_v4(&full[0], 1);    mbar_init_v4(&full[1], 1);
        mbar_init_v4(&empty[0], 1);   mbar_init_v4(&empty[1], 1);
        mbar_init_v4(&d_ready[0], 1); mbar_init_v4(&d_ready[1], 1);
    }
    __syncthreads();

    const int qc0    = k_row0 / Br;
    const int perG   = nQTiles - qc0;
    const int nIter  = G * perG;

    // V15: (g,qc) maintained incrementally in each loop → no per-iter IDIV/IREM.
    auto qFlatRowOf = [&](int g, int qc) -> uint32_t {
        const int hq = hkv * G + g;
        return (uint32_t)((b * Hq + hq) * S + qc * Br);
    };
    auto lBaseOf = [&](int g, int qc) -> long {
        const int hq = hkv * G + g;
        return (long)(b * Hq + hq) * S + (long)qc * Br;
    };

    // ── PRODUCER (wg 2): TMA loads + LAGGED D-rowsum (PART A) ─────────────────
    if (wg == 2) {
        const bool leader = (tid == 256);
        const int pwarp   = (tid - 256) >> 5;   // 0..3 within the producer warpgroup
        if (leader) {
            mbar_expect_tx_v4(&mbar_kv, bytesAtom * 4);
            tma_load_2d_v4(&tma_K_sw, sK_sw,           &mbar_kv, 0,  kvFlatRow);
            tma_load_2d_v4(&tma_K_sw, sK_sw + 64 * 64, &mbar_kv, 64, kvFlatRow);
            tma_load_2d_v4(&tma_V_sw, sV_sw,           &mbar_kv, 0,  kvFlatRow);
            tma_load_2d_v4(&tma_V_sw, sV_sw + 64 * 64, &mbar_kv, 64, kvFlatRow);
        }
        uint32_t epar[2] = {0, 0}, fpar[2] = {0, 0};
        int gP = 0, qcP = qc0;   // V15 incremental (g,qc)
        // Compute D for tile `td` (waits its TMA, reduces, arrives d_ready).
        auto do_drowsum = [&](int td) {
            const int sp = td & 1;
            mbar_wait_v4(&full[sp], fpar[sp]); fpar[sp] ^= 1;
            producer_drowsum_v13<Br, D>(sD[sp], sdO_pl[sp], sO_pl[sp], pwarp, lane);
            producer_sync();
            if (leader) mbar_arrive_v11(&d_ready[sp]);
        };
        for (int it = 0; it < nIter; it++) {
            const int s = it & 1;
            if (it >= 2) { mbar_wait_v4(&empty[s], epar[s]); epar[s] ^= 1; }
            if (leader) {
                const uint32_t r = qFlatRowOf(gP, qcP);
                mbar_expect_tx_v4(&full[s], bytesTile * 4);
                tma_load_2d_v4(&tma_Q_sw,  sQ_sw [s],           &full[s], 0,  r);
                tma_load_2d_v4(&tma_Q_sw,  sQ_sw [s] + 64 * 64, &full[s], 64, r);
                tma_load_2d_v4(&tma_dO_sw, sdO_sw[s],           &full[s], 0,  r);
                tma_load_2d_v4(&tma_dO_sw, sdO_sw[s] + 64 * 64, &full[s], 64, r);
                tma_load_2d_v4(&tma_dO_pl, sdO_pl[s],           &full[s], 0,  r);
                tma_load_2d_v4(&tma_O_pl,  sO_pl [s],           &full[s], 0,  r);
            }
            if (it >= 1) do_drowsum(it - 1);   // lagged one tile → 2 TMAs in flight
            if (++qcP == nQTiles) { qcP = qc0; ++gP; }
        }
        do_drowsum(nIter - 1);                 // tail: final tile's D
        return;
    }

    // ── CONSUMERS (wg 0,1) ───────────────────────────────────────────────────
    mbar_wait_v4(&mbar_kv, 0);

    float dv[32]; zeroN<32>(dv);
    float dk[32]; zeroN<32>(dk);

    uint32_t cpar[2] = {0, 0}, dpar[2] = {0, 0};
    int gC = 0, qcC = qc0;   // V15 incremental (g,qc)
    for (int it = 0; it < nIter; it++) {
        const int s = it & 1;
        const int q_row0 = qcC * Br;
        mbar_wait_v4(&full[s], cpar[s]); cpar[s] ^= 1;
        if (tid < Br) sLSE[tid] = d_LSE[lBaseOf(gC, qcC) + tid];
        consumer_sync();

        // S = Q·Kᵀ·scale → P = exp(S − LSE)+causal DIRECT to swizzled sP (wg0, PART B)
        //  ∥  dP = dO·Vᵀ → sdP (wg1).   [D-rowsum now on the producer, PART A.]
        if (wg == 0) {
            float acc[32]; zeroN<32>(acc);
            run_gemm_n64_sw2(acc, sQ_sw[s], sK_sw);
            fused_p_from_acc_v19<Bc>(acc, sP, sLSE, wtid, q_row0, k_row0, scale);
        } else {
            float acc[32]; zeroN<32>(acc);
            run_gemm_n64_sw2(acc, sdO_sw[s], sV_sw);
            store_acc_smem_v6<Bc, SS_STRIDE_V6>(acc, sdP, wtid, 1.0f);
        }
        consumer_sync();   // wg0's swizzled sP writes visible cross-warp before the transposed reads

        // dV += Pᵀ·dO  — TRANSPOSE-ELIMINATED: A = Pᵀ read DIRECTLY from swizzled sP.
        run_gemm_dVdK_half_te(dv, sP, sdO_sw[s] + wg * 4096);
        consumer_sync();

        // dS = P ⊙ (dP − D) → swizzled sP.  D produced by wg2 → wait d_ready[s] first.
        mbar_wait_v4(&d_ready[s], dpar[s]); dpar[s] ^= 1;
        {   // V18: vectorize dS — 2 adjacent columns/step.  cbase=(2·tid)%Bc is even ⇒
            // sP[pidx],sP[pidx+1] adjacent in the swizzle (cbase,cbase+1 same 8-group) ⇒
            // bf16×2 load/store + float2 arith, halving the cvt→mul→cvt RAW chain.  Each
            // element computed once with identical ops → bit-identical to V17.
            const int cbase = (2 * tid) % Bc, c8 = cbase >> 3, clo = cbase & 7;
            for (int pp = tid; pp < Br * Bc / 2; pp += CONS) {
                const int r    = (2 * pp) / Bc;
                const int pidx = r * 64 + ((c8 ^ (r & 7)) << 3) + clo;
                const int sdi  = r * SS_STRIDE_V6 + cbase;
                const float d  = sD[s][r];
                const __nv_bfloat162 p2 = *reinterpret_cast<const __nv_bfloat162*>(&sP[pidx]);
                const float2 pf  = __bfloat1622float2(p2);
                const float2 res = make_float2(pf.x * (sdP[sdi] - d), pf.y * (sdP[sdi + 1] - d));
                *reinterpret_cast<__nv_bfloat162*>(&sP[pidx]) = __float22bfloat162_rn(res);
            }
        }
        consumer_sync();   // dS writes visible cross-warp before the transposed dK read

        // dK += dSᵀ·Q  — TRANSPOSE-ELIMINATED: A = dSᵀ read DIRECTLY from swizzled sP.
        run_gemm_dVdK_half_te(dk, sP, sQ_sw[s] + wg * 4096);
        consumer_sync();
        if (tid == 0) mbar_arrive_v11(&empty[s]);

        // dQ_tile = dS·K (transient) — dS staged K-major into sA_t from swizzled sP.
        sp_to_sAt_v16_dq(sA_t, sP, tid);
        consumer_sync();
        { float acc[32]; zeroN<32>(acc);
          run_gemm_dQ_half(acc, sA_t, sK_sw + wg * 4096, sbo_pad_v6(Bc));
          stage_acc_f32<64>(acc, (wg == 0) ? sS : sdP, wtid, scale); }
        consumer_sync();
        atomic_flush_stage<Br, 64>((wg == 0) ? sS : sdP, d_dq_accum, lBaseOf(gC, qcC) * D, D, wg * 64, wtid);
        consumer_sync();
        if (++qcC == nQTiles) { qcC = qc0; ++gC; }
    }

    // ── Epilogue — coalesced dV/dK writeback (identical to V11/V12). ──
    bf16 *stage = sA_t + wg * 4096;
    fence_operandN<32>(dv);
    stage_acc_bf16<64>(dv, stage, wtid, 1.0f);
    consumer_sync();
    store_stage_vec<Bc, 64>(stage, d_dV, kvBase, D, wg * 64, wtid);
    consumer_sync();
    fence_operandN<32>(dk);
    stage_acc_bf16<64>(dk, stage, wtid, scale);
    consumer_sync();
    store_stage_vec<Bc, 64>(stage, d_dK, kvBase, D, wg * 64, wtid);
}

// ── V19 launcher — identical to V15 (TMA descriptors, dQ scratch); only the kernel
// symbol changes.  sP swizzled-atom layout is internal to the kernel.
template<int Br, int Bc, int D>
void launch_gqa_backward_v19(
    const bf16 *d_Q, const bf16 *d_K, const bf16 *d_V, const bf16 *d_O,
    const bf16 *d_dO, const float *d_LSE,
    bf16 *d_dQ, bf16 *d_dK, bf16 *d_dV,
    int B, int Hq, int Hkv, int G, int S, float scale
) {
    static_assert(Br == 64 && Bc == 64 && D == 128, "V16 requires Br=Bc=64, D=128");

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
    CUtensorMap tma_V_sw  = make_tma_sw128(d_V,  Rkv, Bc);
    CUtensorMap tma_Q_sw  = make_tma_sw128(d_Q,  Rq,  Br);
    CUtensorMap tma_dO_sw = make_tma_sw128(d_dO, Rq,  Br);
    CUtensorMap tma_dO_pl = make_tma_plain(d_dO, Rq,  Br);
    CUtensorMap tma_O_pl  = make_tma_plain(d_O,  Rq,  Br);

    const long dqN = (long)B * Hq * S * D;
    static float* d_dq_accum = nullptr;
    static long   dq_cap     = 0;
    if (dqN > dq_cap) {
        if (d_dq_accum) CUDA_CHECK(cudaFree(d_dq_accum));
        CUDA_CHECK(cudaMalloc(&d_dq_accum, dqN * sizeof(float)));
        dq_cap = dqN;
    }
    CUDA_CHECK(cudaMemset(d_dq_accum, 0, dqN * sizeof(float)));

    constexpr dim3 BLOCK(384);
    dim3 GRID(B, Hkv, S / Bc);
    gqa_backward_v19_kv<Br,Bc,D><<<GRID, BLOCK>>>(
        tma_K_sw, tma_V_sw, tma_Q_sw, tma_dO_sw, tma_dO_pl, tma_O_pl,
        d_LSE, d_dK, d_dV, d_dq_accum, B, Hq, Hkv, G, S, scale);

    const int convBlock = 256;
    const int convGrid  = (int)((dqN + convBlock - 1) / convBlock);
    convert_dq_accum_to_bf16_v5<<<convGrid, convBlock>>>(d_dq_accum, d_dQ, dqN);
}

// ═════════════════════════════════════════════════════════════════════════════
// V20 — STRUCTURAL smem-unblock: swizzled-D-rowsum (Phase 1) + 3-deep TMA
// pipeline (Phase 2).  Attacks the long_scoreboard parking-spot floor (consumers
// spinning on full[] waiting for the producer's cp.async.bulk.tensor) by letting
// the producer run 3 tiles ahead instead of 2.
//
// PHASE 1 (free ~32 KB): the producer D-rowsum D[r]=Σ_d dO[r,d]·O[r,d] no longer
// needs PLAIN copies of dO/O.  A row-sum over d is order-independent, and sdO_sw
// / sO_sw share a byte-identical 128-B-swizzle layout, so sdO_sw[sidx]·sO_sw[sidx]
// pairs the SAME (row,col) in both buffers.  We iterate the SAME global column j
// (=lane,lane+32,…) as the plain V13/V19 rowsum but read through the swizzle
// address → the per-lane products and their order are IDENTICAL ⇒ D is fp32-bit-
// identical (not merely <2e-2).  DELETE sdO_pl (−32 KB); sO_pl→sO_sw (loaded with
// the swizzled TMA descriptor, same size).
//
// swizzle address for the 64-wide 128-B atom (matches fused_p_from_acc / sp_to_sAt):
//   element (row, c∈[0,64)) → row*64 + (((c>>3) ^ (row&7))<<3) + (c&7)
// D=128 buffer = two atoms: atom = col>>6, local c = col&63, + atom*4096.
// ═════════════════════════════════════════════════════════════════════════════
template<int Br, int D>
__device__ __forceinline__ void producer_drowsum_v20_sw(
    float* sDrow, const bf16* sdO_sw, const bf16* sO_sw, int pwarp, int lane)
{
    static_assert(D == 128, "V20 swizzled D-rowsum assumes D=128 = two 64-wide atoms");
    for (int row = pwarp; row < Br; row += 4) {
        const int m = row & 7;
        float partial = 0.f;
        // Iterate the SAME global columns j as the plain rowsum (lane, lane+32,
        // lane+64, lane+96) in the SAME order → bit-identical partial per lane.
        for (int j = lane; j < D; j += 32) {
            const int atom = j >> 6;                 // 0 or 1
            const int c    = j & 63;                 // local column in atom
            const int sidx = atom * 4096 + row * 64 + (((c >> 3) ^ m) << 3) + (c & 7);
            partial += __bfloat162float(sdO_sw[sidx]) * __bfloat162float(sO_sw[sidx]);
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            partial += __shfl_down_sync(0xFFFFFFFFu, partial, off);
        if (lane == 0) sDrow[row] = partial;
    }
}

template<int Br, int Bc, int D>
__global__ void
gqa_backward_v20_kv(
    const __grid_constant__ CUtensorMap tma_K_sw,
    const __grid_constant__ CUtensorMap tma_V_sw,
    const __grid_constant__ CUtensorMap tma_Q_sw,
    const __grid_constant__ CUtensorMap tma_dO_sw,
    const __grid_constant__ CUtensorMap tma_O_sw,     // V20: swizzled O (was tma_dO_pl + tma_O_pl)
    const float * __restrict__ d_LSE,
    bf16 * __restrict__ d_dK, bf16 * __restrict__ d_dV, float * __restrict__ d_dq_accum,
    int B, int Hq, int Hkv, int G, int S, float scale
) {
    static_assert(Br == 64 && Bc == 64 && D == 128, "V20 requires Br=Bc=64, D=128");
    constexpr int CONS = 256;   // consumer thread count (wg 0+1)
    constexpr int PD   = 3;     // V20 pipeline depth for sQ_sw / sdO_sw (Phase 1: 2, Phase 2: 3)

    __shared__ __align__(128)  bf16 sK_sw[Bc * D];
    __shared__ __align__(128)  bf16 sV_sw[Bc * D];
    __shared__ __align__(128)  bf16 sQ_sw [PD][Br * D];      // V20: 3-deep
    __shared__ __align__(128)  bf16 sdO_sw[PD][Br * D];      // V20: 3-deep
    __shared__ __align__(128)  bf16 sO_sw [2][Br * D];       // V20: swizzled (was sO_pl); depth 2 (producer-only, lag-1 D-rowsum)
    __shared__ __align__(16)   float sS [Br * SS_STRIDE_V6];   // V13: S store DROPPED; kept as wg0 dQ stage
    __shared__ __align__(16)   float sdP[Br * SS_STRIDE_V6];
    __shared__ __align__(1024) bf16  sP [Br * 64];             // V16: 64-wide 128-B swizzle atom, 1024-B aligned
    __shared__                 float sLSE[Br];
    __shared__                 float sD  [2][Br];              // V13: double-buffered, producer-written
    __shared__ __align__(128)  bf16  sA_t[SMEM_TILED_V6];      // dQ stage + epilogue writeback stage
    __shared__ __align__(8)    uint64_t mbar_kv;
    __shared__ __align__(8)    uint64_t full   [PD];
    __shared__ __align__(8)    uint64_t empty  [PD];
    __shared__ __align__(8)    uint64_t d_ready[PD];           // V13: producer→consumer D signal

    const int tid   = threadIdx.x;
    const int wg    = tid >> 7;
    const int wtid  = tid & 127;
    const int lane  = tid & 31;
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

    // V15: (g,qc) maintained incrementally in each loop → no per-iter IDIV/IREM.
    auto qFlatRowOf = [&](int g, int qc) -> uint32_t {
        const int hq = hkv * G + g;
        return (uint32_t)((b * Hq + hq) * S + qc * Br);
    };
    auto lBaseOf = [&](int g, int qc) -> long {
        const int hq = hkv * G + g;
        return (long)(b * Hq + hq) * S + (long)qc * Br;
    };

    // ── PRODUCER (wg 2): TMA loads (3-deep) + LAGGED swizzled D-rowsum (PART A) ──
    if (wg == 2) {
        const bool leader = (tid == 256);
        const int pwarp   = (tid - 256) >> 5;   // 0..3 within the producer warpgroup
        if (leader) {
            mbar_expect_tx_v4(&mbar_kv, bytesAtom * 4);
            tma_load_2d_v4(&tma_K_sw, sK_sw,           &mbar_kv, 0,  kvFlatRow);
            tma_load_2d_v4(&tma_K_sw, sK_sw + 64 * 64, &mbar_kv, 64, kvFlatRow);
            tma_load_2d_v4(&tma_V_sw, sV_sw,           &mbar_kv, 0,  kvFlatRow);
            tma_load_2d_v4(&tma_V_sw, sV_sw + 64 * 64, &mbar_kv, 64, kvFlatRow);
        }
        uint32_t epar[PD] = {0}, fpar[PD] = {0};
        int gP = 0, qcP = qc0;   // V15 incremental (g,qc)
        // Compute D for tile `td` (waits its TMA, reduces from swizzled buffers).
        // sdO_sw is PD-deep (slot td%PD); sO_sw is 2-deep (slot td&1, producer-only,
        // safe under lag-1 because D-rowsum(td) completes before O(td+2) overwrites).
        auto do_drowsum = [&](int td) {
            const int fp = td % PD;
            const int op = td & 1;
            mbar_wait_v4(&full[fp], fpar[fp]); fpar[fp] ^= 1;
            producer_drowsum_v20_sw<Br, D>(sD[op], sdO_sw[fp], sO_sw[op], pwarp, lane);
            producer_sync();
            if (leader) mbar_arrive_v11(&d_ready[fp]);
        };
        for (int it = 0; it < nIter; it++) {
            const int s  = it % PD;      // Q/dO slot (3-deep)
            const int os = it & 1;       // O slot (2-deep)
            if (it >= PD) { mbar_wait_v4(&empty[s], epar[s]); epar[s] ^= 1; }
            if (leader) {
                const uint32_t r = qFlatRowOf(gP, qcP);
                mbar_expect_tx_v4(&full[s], bytesTile * 3);
                tma_load_2d_v4(&tma_Q_sw,  sQ_sw [s],           &full[s], 0,  r);
                tma_load_2d_v4(&tma_Q_sw,  sQ_sw [s] + 64 * 64, &full[s], 64, r);
                tma_load_2d_v4(&tma_dO_sw, sdO_sw[s],           &full[s], 0,  r);
                tma_load_2d_v4(&tma_dO_sw, sdO_sw[s] + 64 * 64, &full[s], 64, r);
                tma_load_2d_v4(&tma_O_sw,  sO_sw [os],           &full[s], 0,  r);
                tma_load_2d_v4(&tma_O_sw,  sO_sw [os] + 64 * 64, &full[s], 64, r);
            }
            if (it >= 1) do_drowsum(it - 1);   // lagged one tile → keeps TMAs in flight
            if (++qcP == nQTiles) { qcP = qc0; ++gP; }
        }
        do_drowsum(nIter - 1);                 // tail: final tile's D
        return;
    }

    // ── CONSUMERS (wg 0,1) ───────────────────────────────────────────────────
    mbar_wait_v4(&mbar_kv, 0);

    float dv[32]; zeroN<32>(dv);
    float dk[32]; zeroN<32>(dk);

    uint32_t cpar[PD] = {0}, dpar[PD] = {0};
    int gC = 0, qcC = qc0;   // V15 incremental (g,qc)
    for (int it = 0; it < nIter; it++) {
        const int s = it % PD;
        const int q_row0 = qcC * Br;
        mbar_wait_v4(&full[s], cpar[s]); cpar[s] ^= 1;
        if (tid < Br) sLSE[tid] = d_LSE[lBaseOf(gC, qcC) + tid];
        consumer_sync();

        // S = Q·Kᵀ·scale → P = exp(S − LSE)+causal DIRECT to swizzled sP (wg0, PART B)
        //  ∥  dP = dO·Vᵀ → sdP (wg1).   [D-rowsum now on the producer, PART A.]
        if (wg == 0) {
            float acc[32]; zeroN<32>(acc);
            run_gemm_n64_sw2(acc, sQ_sw[s], sK_sw);
            fused_p_from_acc_v19<Bc>(acc, sP, sLSE, wtid, q_row0, k_row0, scale);
        } else {
            float acc[32]; zeroN<32>(acc);
            run_gemm_n64_sw2(acc, sdO_sw[s], sV_sw);
            store_acc_smem_v6<Bc, SS_STRIDE_V6>(acc, sdP, wtid, 1.0f);
        }
        consumer_sync();   // wg0's swizzled sP writes visible cross-warp before the transposed reads

        // dV += Pᵀ·dO  — TRANSPOSE-ELIMINATED: A = Pᵀ read DIRECTLY from swizzled sP.
        run_gemm_dVdK_half_te(dv, sP, sdO_sw[s] + wg * 4096);
        consumer_sync();

        // dS = P ⊙ (dP − D) → swizzled sP.  D produced by wg2 → wait d_ready[s] first.
        mbar_wait_v4(&d_ready[s], dpar[s]); dpar[s] ^= 1;
        {   // V18/V19: vectorize dS — 2 adjacent columns/step (bf16×2 + float2).
            const int cbase = (2 * tid) % Bc, c8 = cbase >> 3, clo = cbase & 7;
            for (int pp = tid; pp < Br * Bc / 2; pp += CONS) {
                const int r    = (2 * pp) / Bc;
                const int pidx = r * 64 + ((c8 ^ (r & 7)) << 3) + clo;
                const int sdi  = r * SS_STRIDE_V6 + cbase;
                const float d  = sD[it & 1][r];   // sD is 2-deep by TILE parity (producer op=td&1)
                const __nv_bfloat162 p2 = *reinterpret_cast<const __nv_bfloat162*>(&sP[pidx]);
                const float2 pf  = __bfloat1622float2(p2);
                const float2 res = make_float2(pf.x * (sdP[sdi] - d), pf.y * (sdP[sdi + 1] - d));
                *reinterpret_cast<__nv_bfloat162*>(&sP[pidx]) = __float22bfloat162_rn(res);
            }
        }
        consumer_sync();   // dS writes visible cross-warp before the transposed dK read

        // dK += dSᵀ·Q  — TRANSPOSE-ELIMINATED: A = dSᵀ read DIRECTLY from swizzled sP.
        run_gemm_dVdK_half_te(dk, sP, sQ_sw[s] + wg * 4096);
        consumer_sync();
        if (tid == 0) mbar_arrive_v11(&empty[s]);

        // dQ_tile = dS·K (transient) — dS staged K-major into sA_t from swizzled sP.
        sp_to_sAt_v16_dq(sA_t, sP, tid);
        consumer_sync();
        { float acc[32]; zeroN<32>(acc);
          run_gemm_dQ_half(acc, sA_t, sK_sw + wg * 4096, sbo_pad_v6(Bc));
          stage_acc_f32<64>(acc, (wg == 0) ? sS : sdP, wtid, scale); }
        consumer_sync();
        atomic_flush_stage<Br, 64>((wg == 0) ? sS : sdP, d_dq_accum, lBaseOf(gC, qcC) * D, D, wg * 64, wtid);
        consumer_sync();
        if (++qcC == nQTiles) { qcC = qc0; ++gC; }
    }

    // ── Epilogue — coalesced dV/dK writeback (identical to V11/V12). ──
    bf16 *stage = sA_t + wg * 4096;
    fence_operandN<32>(dv);
    stage_acc_bf16<64>(dv, stage, wtid, 1.0f);
    consumer_sync();
    store_stage_vec<Bc, 64>(stage, d_dV, kvBase, D, wg * 64, wtid);
    consumer_sync();
    fence_operandN<32>(dk);
    stage_acc_bf16<64>(dk, stage, wtid, scale);
    consumer_sync();
    store_stage_vec<Bc, 64>(stage, d_dK, kvBase, D, wg * 64, wtid);
}

// ── V20 launcher — like V19 but O is loaded SWIZZLED (tma_O_sw); the plain
// dO/O descriptors are gone.
template<int Br, int Bc, int D>
void launch_gqa_backward_v20(
    const bf16 *d_Q, const bf16 *d_K, const bf16 *d_V, const bf16 *d_O,
    const bf16 *d_dO, const float *d_LSE,
    bf16 *d_dQ, bf16 *d_dK, bf16 *d_dV,
    int B, int Hq, int Hkv, int G, int S, float scale
) {
    static_assert(Br == 64 && Bc == 64 && D == 128, "V20 requires Br=Bc=64, D=128");

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
    CUtensorMap tma_O_sw  = make_tma_sw128(d_O,  Rq,  Br);   // V20: O now swizzled

    const long dqN = (long)B * Hq * S * D;
    static float* d_dq_accum = nullptr;
    static long   dq_cap     = 0;
    if (dqN > dq_cap) {
        if (d_dq_accum) CUDA_CHECK(cudaFree(d_dq_accum));
        CUDA_CHECK(cudaMalloc(&d_dq_accum, dqN * sizeof(float)));
        dq_cap = dqN;
    }
    CUDA_CHECK(cudaMemset(d_dq_accum, 0, dqN * sizeof(float)));

    constexpr dim3 BLOCK(384);
    dim3 GRID(B, Hkv, S / Bc);
    gqa_backward_v20_kv<Br,Bc,D><<<GRID, BLOCK>>>(
        tma_K_sw, tma_V_sw, tma_Q_sw, tma_dO_sw, tma_O_sw,
        d_LSE, d_dK, d_dV, d_dq_accum, B, Hq, Hkv, G, S, scale);

    const int convBlock = 256;
    const int convGrid  = (int)((dqN + convBlock - 1) / convBlock);
    convert_dq_accum_to_bf16_v5<<<convGrid, convBlock>>>(d_dq_accum, d_dQ, dqN);
}

// ─────────────────────────────────────────────────────────────────────────────
// V21 — V20 clone; CAUSAL-MASK SPECIALIZATION: only the diagonal tile (qc==qc0) runs the
// masked fused-P; all off-diagonal tiles skip the mask+index math.  Bit-identical.
// ─────────────────────────────────────────────────────────────────────────────
// V21: causal-mask-FREE fused-P for non-diagonal tiles (qc>qc0 ⇒ every k<q ⇒ the mask
// never fires ⇒ bit-identical to fused_p_from_acc_v19).  Drops gc/gr index math + ISETP/SEL
// from the critical-path exp — used on ~97% of tiles (only the diagonal keeps the mask).
template<int Bc>
__device__ __forceinline__ void fused_p_nomask_v21(
    const float acc[32], bf16* sP, const float* sLSE, int wtid, float scale)
{
    const int w = wtid >> 5, lane = wtid & 31;
    const int r0 = w * 16 + (lane >> 2), r1 = r0 + 8, cc = (lane & 3) * 2;
    const float l0 = sLSE[r0], l1 = sLSE[r1];
    const int base0 = r0 * 64 + cc, m0 = r0 & 7;
    const int base1 = r1 * 64 + cc, m1 = r1 & 7;
#pragma unroll
    for (int nt = 0; nt < Bc / 8; nt++) {
        const int b0 = base0 + ((nt ^ m0) << 3);
        const int b1 = base1 + ((nt ^ m1) << 3);
        const float2 p0 = make_float2(__expf(acc[nt * 4 + 0] * scale - l0), __expf(acc[nt * 4 + 1] * scale - l0));
        const float2 p1 = make_float2(__expf(acc[nt * 4 + 2] * scale - l1), __expf(acc[nt * 4 + 3] * scale - l1));
        *reinterpret_cast<__nv_bfloat162*>(&sP[b0]) = __float22bfloat162_rn(p0);
        *reinterpret_cast<__nv_bfloat162*>(&sP[b1]) = __float22bfloat162_rn(p1);
    }
}

template<int Br, int Bc, int D>
__global__ void
gqa_backward_v21_kv(
    const __grid_constant__ CUtensorMap tma_K_sw,
    const __grid_constant__ CUtensorMap tma_V_sw,
    const __grid_constant__ CUtensorMap tma_Q_sw,
    const __grid_constant__ CUtensorMap tma_dO_sw,
    const __grid_constant__ CUtensorMap tma_O_sw,     // V20: swizzled O (was tma_dO_pl + tma_O_pl)
    const float * __restrict__ d_LSE,
    bf16 * __restrict__ d_dK, bf16 * __restrict__ d_dV, float * __restrict__ d_dq_accum,
    int B, int Hq, int Hkv, int G, int S, float scale
) {
    static_assert(Br == 64 && Bc == 64 && D == 128, "V20 requires Br=Bc=64, D=128");
    constexpr int CONS = 256;   // consumer thread count (wg 0+1)
    constexpr int PD   = 3;     // V20 pipeline depth for sQ_sw / sdO_sw (Phase 1: 2, Phase 2: 3)

    __shared__ __align__(128)  bf16 sK_sw[Bc * D];
    __shared__ __align__(128)  bf16 sV_sw[Bc * D];
    __shared__ __align__(128)  bf16 sQ_sw [PD][Br * D];      // V20: 3-deep
    __shared__ __align__(128)  bf16 sdO_sw[PD][Br * D];      // V20: 3-deep
    __shared__ __align__(128)  bf16 sO_sw [2][Br * D];       // V20: swizzled (was sO_pl); depth 2 (producer-only, lag-1 D-rowsum)
    __shared__ __align__(16)   float sS [Br * SS_STRIDE_V6];   // V13: S store DROPPED; kept as wg0 dQ stage
    __shared__ __align__(16)   float sdP[Br * SS_STRIDE_V6];
    __shared__ __align__(1024) bf16  sP [Br * 64];             // V16: 64-wide 128-B swizzle atom, 1024-B aligned
    __shared__                 float sLSE[Br];
    __shared__                 float sD  [2][Br];              // V13: double-buffered, producer-written
    __shared__ __align__(128)  bf16  sA_t[SMEM_TILED_V6];      // dQ stage + epilogue writeback stage
    __shared__ __align__(8)    uint64_t mbar_kv;
    __shared__ __align__(8)    uint64_t full   [PD];
    __shared__ __align__(8)    uint64_t empty  [PD];
    __shared__ __align__(8)    uint64_t d_ready[PD];           // V13: producer→consumer D signal

    const int tid   = threadIdx.x;
    const int wg    = tid >> 7;
    const int wtid  = tid & 127;
    const int lane  = tid & 31;
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

    // V15: (g,qc) maintained incrementally in each loop → no per-iter IDIV/IREM.
    auto qFlatRowOf = [&](int g, int qc) -> uint32_t {
        const int hq = hkv * G + g;
        return (uint32_t)((b * Hq + hq) * S + qc * Br);
    };
    auto lBaseOf = [&](int g, int qc) -> long {
        const int hq = hkv * G + g;
        return (long)(b * Hq + hq) * S + (long)qc * Br;
    };

    // ── PRODUCER (wg 2): TMA loads (3-deep) + LAGGED swizzled D-rowsum (PART A) ──
    if (wg == 2) {
        const bool leader = (tid == 256);
        const int pwarp   = (tid - 256) >> 5;   // 0..3 within the producer warpgroup
        if (leader) {
            mbar_expect_tx_v4(&mbar_kv, bytesAtom * 4);
            tma_load_2d_v4(&tma_K_sw, sK_sw,           &mbar_kv, 0,  kvFlatRow);
            tma_load_2d_v4(&tma_K_sw, sK_sw + 64 * 64, &mbar_kv, 64, kvFlatRow);
            tma_load_2d_v4(&tma_V_sw, sV_sw,           &mbar_kv, 0,  kvFlatRow);
            tma_load_2d_v4(&tma_V_sw, sV_sw + 64 * 64, &mbar_kv, 64, kvFlatRow);
        }
        uint32_t epar[PD] = {0}, fpar[PD] = {0};
        int gP = 0, qcP = qc0;   // V15 incremental (g,qc)
        // Compute D for tile `td` (waits its TMA, reduces from swizzled buffers).
        // sdO_sw is PD-deep (slot td%PD); sO_sw is 2-deep (slot td&1, producer-only,
        // safe under lag-1 because D-rowsum(td) completes before O(td+2) overwrites).
        auto do_drowsum = [&](int td) {
            const int fp = td % PD;
            const int op = td & 1;
            mbar_wait_v4(&full[fp], fpar[fp]); fpar[fp] ^= 1;
            producer_drowsum_v20_sw<Br, D>(sD[op], sdO_sw[fp], sO_sw[op], pwarp, lane);
            producer_sync();
            if (leader) mbar_arrive_v11(&d_ready[fp]);
        };
        for (int it = 0; it < nIter; it++) {
            const int s  = it % PD;      // Q/dO slot (3-deep)
            const int os = it & 1;       // O slot (2-deep)
            if (it >= PD) { mbar_wait_v4(&empty[s], epar[s]); epar[s] ^= 1; }
            if (leader) {
                const uint32_t r = qFlatRowOf(gP, qcP);
                mbar_expect_tx_v4(&full[s], bytesTile * 3);
                tma_load_2d_v4(&tma_Q_sw,  sQ_sw [s],           &full[s], 0,  r);
                tma_load_2d_v4(&tma_Q_sw,  sQ_sw [s] + 64 * 64, &full[s], 64, r);
                tma_load_2d_v4(&tma_dO_sw, sdO_sw[s],           &full[s], 0,  r);
                tma_load_2d_v4(&tma_dO_sw, sdO_sw[s] + 64 * 64, &full[s], 64, r);
                tma_load_2d_v4(&tma_O_sw,  sO_sw [os],           &full[s], 0,  r);
                tma_load_2d_v4(&tma_O_sw,  sO_sw [os] + 64 * 64, &full[s], 64, r);
            }
            if (it >= 1) do_drowsum(it - 1);   // lagged one tile → keeps TMAs in flight
            if (++qcP == nQTiles) { qcP = qc0; ++gP; }
        }
        do_drowsum(nIter - 1);                 // tail: final tile's D
        return;
    }

    // ── CONSUMERS (wg 0,1) ───────────────────────────────────────────────────
    mbar_wait_v4(&mbar_kv, 0);

    float dv[32]; zeroN<32>(dv);
    float dk[32]; zeroN<32>(dk);

    uint32_t cpar[PD] = {0}, dpar[PD] = {0};
    int gC = 0, qcC = qc0;   // V15 incremental (g,qc)
    for (int it = 0; it < nIter; it++) {
        const int s = it % PD;
        const int q_row0 = qcC * Br;
        mbar_wait_v4(&full[s], cpar[s]); cpar[s] ^= 1;
        if (tid < Br) sLSE[tid] = d_LSE[lBaseOf(gC, qcC) + tid];
        consumer_sync();

        // S = Q·Kᵀ·scale → P = exp(S − LSE)+causal DIRECT to swizzled sP (wg0, PART B)
        //  ∥  dP = dO·Vᵀ → sdP (wg1).   [D-rowsum now on the producer, PART A.]
        if (wg == 0) {
            float acc[32]; zeroN<32>(acc);
            run_gemm_n64_sw2(acc, sQ_sw[s], sK_sw);
            if (qcC == qc0) fused_p_from_acc_v19<Bc>(acc, sP, sLSE, wtid, q_row0, k_row0, scale);
            else            fused_p_nomask_v21<Bc>(acc, sP, sLSE, wtid, scale);   // V21: no mask off-diagonal
        } else {
            float acc[32]; zeroN<32>(acc);
            run_gemm_n64_sw2(acc, sdO_sw[s], sV_sw);
            store_acc_smem_v6<Bc, SS_STRIDE_V6>(acc, sdP, wtid, 1.0f);
        }
        consumer_sync();   // wg0's swizzled sP writes visible cross-warp before the transposed reads

        // dV += Pᵀ·dO  — TRANSPOSE-ELIMINATED: A = Pᵀ read DIRECTLY from swizzled sP.
        run_gemm_dVdK_half_te(dv, sP, sdO_sw[s] + wg * 4096);
        consumer_sync();

        // dS = P ⊙ (dP − D) → swizzled sP.  D produced by wg2 → wait d_ready[s] first.
        mbar_wait_v4(&d_ready[s], dpar[s]); dpar[s] ^= 1;
        {   // V18/V19: vectorize dS — 2 adjacent columns/step (bf16×2 + float2).
            const int cbase = (2 * tid) % Bc, c8 = cbase >> 3, clo = cbase & 7;
            for (int pp = tid; pp < Br * Bc / 2; pp += CONS) {
                const int r    = (2 * pp) / Bc;
                const int pidx = r * 64 + ((c8 ^ (r & 7)) << 3) + clo;
                const int sdi  = r * SS_STRIDE_V6 + cbase;
                const float d  = sD[it & 1][r];   // sD is 2-deep by TILE parity (producer op=td&1)
                const __nv_bfloat162 p2 = *reinterpret_cast<const __nv_bfloat162*>(&sP[pidx]);
                const float2 pf  = __bfloat1622float2(p2);
                const float2 res = make_float2(pf.x * (sdP[sdi] - d), pf.y * (sdP[sdi + 1] - d));
                *reinterpret_cast<__nv_bfloat162*>(&sP[pidx]) = __float22bfloat162_rn(res);
            }
        }
        consumer_sync();   // dS writes visible cross-warp before the transposed dK read

        // dK += dSᵀ·Q  — TRANSPOSE-ELIMINATED: A = dSᵀ read DIRECTLY from swizzled sP.
        run_gemm_dVdK_half_te(dk, sP, sQ_sw[s] + wg * 4096);
        consumer_sync();
        if (tid == 0) mbar_arrive_v11(&empty[s]);

        // dQ_tile = dS·K (transient) — dS staged K-major into sA_t from swizzled sP.
        sp_to_sAt_v16_dq(sA_t, sP, tid);
        consumer_sync();
        { float acc[32]; zeroN<32>(acc);
          run_gemm_dQ_half(acc, sA_t, sK_sw + wg * 4096, sbo_pad_v6(Bc));
          stage_acc_f32<64>(acc, (wg == 0) ? sS : sdP, wtid, scale); }
        consumer_sync();
        atomic_flush_stage<Br, 64>((wg == 0) ? sS : sdP, d_dq_accum, lBaseOf(gC, qcC) * D, D, wg * 64, wtid);
        consumer_sync();
        if (++qcC == nQTiles) { qcC = qc0; ++gC; }
    }

    // ── Epilogue — coalesced dV/dK writeback (identical to V11/V12). ──
    bf16 *stage = sA_t + wg * 4096;
    fence_operandN<32>(dv);
    stage_acc_bf16<64>(dv, stage, wtid, 1.0f);
    consumer_sync();
    store_stage_vec<Bc, 64>(stage, d_dV, kvBase, D, wg * 64, wtid);
    consumer_sync();
    fence_operandN<32>(dk);
    stage_acc_bf16<64>(dk, stage, wtid, scale);
    consumer_sync();
    store_stage_vec<Bc, 64>(stage, d_dK, kvBase, D, wg * 64, wtid);
}

// ── V21 launcher — like V19 but O is loaded SWIZZLED (tma_O_sw); the plain
// dO/O descriptors are gone.
template<int Br, int Bc, int D>
void launch_gqa_backward_v21(
    const bf16 *d_Q, const bf16 *d_K, const bf16 *d_V, const bf16 *d_O,
    const bf16 *d_dO, const float *d_LSE,
    bf16 *d_dQ, bf16 *d_dK, bf16 *d_dV,
    int B, int Hq, int Hkv, int G, int S, float scale
) {
    static_assert(Br == 64 && Bc == 64 && D == 128, "V20 requires Br=Bc=64, D=128");

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
    CUtensorMap tma_O_sw  = make_tma_sw128(d_O,  Rq,  Br);   // V20: O now swizzled

    const long dqN = (long)B * Hq * S * D;
    static float* d_dq_accum = nullptr;
    static long   dq_cap     = 0;
    if (dqN > dq_cap) {
        if (d_dq_accum) CUDA_CHECK(cudaFree(d_dq_accum));
        CUDA_CHECK(cudaMalloc(&d_dq_accum, dqN * sizeof(float)));
        dq_cap = dqN;
    }
    CUDA_CHECK(cudaMemset(d_dq_accum, 0, dqN * sizeof(float)));

    constexpr dim3 BLOCK(384);
    dim3 GRID(B, Hkv, S / Bc);
    gqa_backward_v21_kv<Br,Bc,D><<<GRID, BLOCK>>>(
        tma_K_sw, tma_V_sw, tma_Q_sw, tma_dO_sw, tma_O_sw,
        d_LSE, d_dK, d_dV, d_dq_accum, B, Hq, Hkv, G, S, scale);

    const int convBlock = 256;
    const int convGrid  = (int)((dqN + convBlock - 1) / convBlock);
    convert_dq_accum_to_bf16_v5<<<convGrid, convBlock>>>(d_dq_accum, d_dQ, dqN);
}

// ═════════════════════════════════════════════════════════════════════════════
// V22 — D-ROWSUM SPLIT (cuDNN-style multi-kernel).  D[row]=Σ_d dO[row,d]·O[row,d]
// is precomputed in a SEPARATE light kernel `compute_drowsum_v22` and written to a
// global fp32 buffer d_Drow[B*Hq*S].  The main kernel `gqa_backward_v22_kv` (V21
// clone) then DROPS the inline producer D-rowsum reduction (our measured #1
// short_scoreboard stall — the producer_drowsum shuffle-tree FADD cluster) and the
// swizzled O TMA load entirely; the producer instead loads d_Drow[q_rows] into
// sD[slot] (coalesced global read) in place of the deleted compute, keeping the
// existing d_ready[]/full[]/empty[] mbarrier protocol byte-for-byte.
//
// DELETIONS vs V21: sO_sw (2×Br×D bf16 = 32,768 B smem freed) + tma_O_sw + O TMA
// traffic + producer_drowsum_v20_sw call.  full[] expect_tx drops 3→2 tiles.
//
// WITHIN-TOLERANCE (NOT bit-identical): compute_drowsum_v22 uses a fast coalesced
// vectorized (bf16×2, 2 accumulators) warp-per-row reduction whose accumulation
// ORDER differs from V21's producer_drowsum (strided lane-j + shfl tree).  D shifts
// in the last fp32 ULP ⇒ dS/dQ/dK are WITHIN 2e-2, not bit-identical.  Approved
// break of the 22-version bit-identical streak for the cleaner/faster D kernel.
// ═════════════════════════════════════════════════════════════════════════════

// One warp per row.  D=128 = 32 lanes × 4 contiguous elems/lane (8-byte coalesced
// loads of two __nv_bfloat162 each).  2 accumulators (acc0/acc1) for ILP, then a
// 5-step warp-shuffle reduction.  Memory-bound: reads ~2·B·Hq·S·D·2 B ≈ 200 MB.
__global__ void compute_drowsum_v22(
    const bf16 * __restrict__ d_dO, const bf16 * __restrict__ d_O,
    float * __restrict__ d_Drow, long nRows)
{
    // D is fixed at 128 for this kernel family (32 lanes × 4 elems).
    const int  gtid   = blockIdx.x * blockDim.x + threadIdx.x;
    const int  warpId = gtid >> 5;
    const int  lane   = gtid & 31;
    if ((long)warpId >= nRows) return;

    const long e = (long)warpId * 128 + lane * 4;   // lane owns 4 contiguous elems
    const __nv_bfloat162 do01 = *reinterpret_cast<const __nv_bfloat162*>(&d_dO[e]);
    const __nv_bfloat162 do23 = *reinterpret_cast<const __nv_bfloat162*>(&d_dO[e + 2]);
    const __nv_bfloat162 o01  = *reinterpret_cast<const __nv_bfloat162*>(&d_O[e]);
    const __nv_bfloat162 o23  = *reinterpret_cast<const __nv_bfloat162*>(&d_O[e + 2]);
    const float2 df0 = __bfloat1622float2(do01), df1 = __bfloat1622float2(do23);
    const float2 of0 = __bfloat1622float2(o01),  of1 = __bfloat1622float2(o23);
    float acc0 = df0.x * of0.x + df0.y * of0.y;   // 2 independent partials → ILP
    float acc1 = df1.x * of1.x + df1.y * of1.y;
    float acc  = acc0 + acc1;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
    if (lane == 0) d_Drow[warpId] = acc;
}

template<int Br, int Bc, int D>
__global__ void
gqa_backward_v22_kv(
    const __grid_constant__ CUtensorMap tma_K_sw,
    const __grid_constant__ CUtensorMap tma_V_sw,
    const __grid_constant__ CUtensorMap tma_Q_sw,
    const __grid_constant__ CUtensorMap tma_dO_sw,
    const float * __restrict__ d_Drow,               // V22: precomputed D=Σ dO·O
    const float * __restrict__ d_LSE,
    bf16 * __restrict__ d_dK, bf16 * __restrict__ d_dV, float * __restrict__ d_dq_accum,
    int B, int Hq, int Hkv, int G, int S, float scale
) {
    static_assert(Br == 64 && Bc == 64 && D == 128, "V22 requires Br=Bc=64, D=128");
    constexpr int CONS = 256;   // consumer thread count (wg 0+1)
    constexpr int PD   = 3;     // pipeline depth for sQ_sw / sdO_sw

    __shared__ __align__(128)  bf16 sK_sw[Bc * D];
    __shared__ __align__(128)  bf16 sV_sw[Bc * D];
    __shared__ __align__(128)  bf16 sQ_sw [PD][Br * D];      // 3-deep
    __shared__ __align__(128)  bf16 sdO_sw[PD][Br * D];      // 3-deep
    // V22: sO_sw DELETED (−32,768 B) — O only fed the inline D-rowsum, now a
    // separate kernel.  dV uses dO (sdO_sw), not O.
    __shared__ __align__(16)   float sS [Br * SS_STRIDE_V6];   // wg0 dQ stage
    __shared__ __align__(16)   float sdP[Br * SS_STRIDE_V6];
    __shared__ __align__(1024) bf16  sP [Br * 64];             // 64-wide 128-B swizzle atom
    __shared__                 float sLSE[Br];
    __shared__                 float sD  [2][Br];              // double-buffered, producer-loaded from d_Drow
    __shared__ __align__(128)  bf16  sA_t[SMEM_TILED_V6];      // dQ stage + epilogue writeback stage
    __shared__ __align__(8)    uint64_t mbar_kv;
    __shared__ __align__(8)    uint64_t full   [PD];
    __shared__ __align__(8)    uint64_t empty  [PD];
    __shared__ __align__(8)    uint64_t d_ready[PD];           // producer→consumer D-ready signal

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

    // V15: (g,qc) maintained incrementally in each loop → no per-iter IDIV/IREM.
    auto qFlatRowOf = [&](int g, int qc) -> uint32_t {
        const int hq = hkv * G + g;
        return (uint32_t)((b * Hq + hq) * S + qc * Br);
    };
    auto lBaseOf = [&](int g, int qc) -> long {
        const int hq = hkv * G + g;
        return (long)(b * Hq + hq) * S + (long)qc * Br;
    };

    // ── PRODUCER (wg 2): TMA loads (3-deep) + LAGGED d_Drow load (replaces the
    // V21 inline swizzled D-rowsum).  D is now precomputed in d_Drow[flatRow]. ──
    if (wg == 2) {
        const bool leader = (tid == 256);
        const int  pl     = tid - 256;          // producer-local thread id [0,128)
        if (leader) {
            mbar_expect_tx_v4(&mbar_kv, bytesAtom * 4);
            tma_load_2d_v4(&tma_K_sw, sK_sw,           &mbar_kv, 0,  kvFlatRow);
            tma_load_2d_v4(&tma_K_sw, sK_sw + 64 * 64, &mbar_kv, 64, kvFlatRow);
            tma_load_2d_v4(&tma_V_sw, sV_sw,           &mbar_kv, 0,  kvFlatRow);
            tma_load_2d_v4(&tma_V_sw, sV_sw + 64 * 64, &mbar_kv, 64, kvFlatRow);
        }
        uint32_t epar[PD] = {0}, fpar[PD] = {0};
        int gP = 0, qcP = qc0;   // TMA (g,qc), tile `it`
        int gD = 0, qcD = qc0;   // D-load (g,qc), monotonic over td = 0,1,2,…
        // Load D for tile `td` from d_Drow → sD[td&1].  Keeps the full[fp] wait so
        // the sD 2-deep buffer reuse timing is byte-identical to V21 (harmless
        // over-sync: producer is not the bottleneck).  d_Drow does not depend on
        // the Q/dO TMA, but gating on it preserves the proven safety invariant.
        auto do_dload = [&](int td) {
            const int fp = td % PD;
            const int op = td & 1;
            mbar_wait_v4(&full[fp], fpar[fp]); fpar[fp] ^= 1;
            const long dbase = lBaseOf(gD, qcD);           // flat base for tile td
            if (pl < Br) sD[op][pl] = d_Drow[dbase + pl];  // coalesced 64×fp32 = 256 B
            producer_sync();
            if (leader) mbar_arrive_v11(&d_ready[fp]);
            if (++qcD == nQTiles) { qcD = qc0; ++gD; }
        };
        for (int it = 0; it < nIter; it++) {
            const int s = it % PD;      // Q/dO slot (3-deep)
            if (it >= PD) { mbar_wait_v4(&empty[s], epar[s]); epar[s] ^= 1; }
            if (leader) {
                const uint32_t r = qFlatRowOf(gP, qcP);
                mbar_expect_tx_v4(&full[s], bytesTile * 2);   // V22: 2 tiles (Q,dO); O gone
                tma_load_2d_v4(&tma_Q_sw,  sQ_sw [s],           &full[s], 0,  r);
                tma_load_2d_v4(&tma_Q_sw,  sQ_sw [s] + 64 * 64, &full[s], 64, r);
                tma_load_2d_v4(&tma_dO_sw, sdO_sw[s],           &full[s], 0,  r);
                tma_load_2d_v4(&tma_dO_sw, sdO_sw[s] + 64 * 64, &full[s], 64, r);
            }
            if (it >= 1) do_dload(it - 1);   // lagged one tile → keeps TMAs in flight
            if (++qcP == nQTiles) { qcP = qc0; ++gP; }
        }
        do_dload(nIter - 1);                 // tail: final tile's D
        return;
    }

    // ── CONSUMERS (wg 0,1) — IDENTICAL to V21 ────────────────────────────────
    mbar_wait_v4(&mbar_kv, 0);

    float dv[32]; zeroN<32>(dv);
    float dk[32]; zeroN<32>(dk);

    uint32_t cpar[PD] = {0}, dpar[PD] = {0};
    int gC = 0, qcC = qc0;   // V15 incremental (g,qc)
    for (int it = 0; it < nIter; it++) {
        const int s = it % PD;
        const int q_row0 = qcC * Br;
        mbar_wait_v4(&full[s], cpar[s]); cpar[s] ^= 1;
        if (tid < Br) sLSE[tid] = d_LSE[lBaseOf(gC, qcC) + tid];
        consumer_sync();

        // S = Q·Kᵀ·scale → P = exp(S − LSE)+causal DIRECT to swizzled sP (wg0)
        //  ∥  dP = dO·Vᵀ → sdP (wg1).
        if (wg == 0) {
            float acc[32]; zeroN<32>(acc);
            run_gemm_n64_sw2(acc, sQ_sw[s], sK_sw);
            if (qcC == qc0) fused_p_from_acc_v19<Bc>(acc, sP, sLSE, wtid, q_row0, k_row0, scale);
            else            fused_p_nomask_v21<Bc>(acc, sP, sLSE, wtid, scale);
        } else {
            float acc[32]; zeroN<32>(acc);
            run_gemm_n64_sw2(acc, sdO_sw[s], sV_sw);
            store_acc_smem_v6<Bc, SS_STRIDE_V6>(acc, sdP, wtid, 1.0f);
        }
        consumer_sync();   // wg0's swizzled sP writes visible cross-warp before the transposed reads

        // dV += Pᵀ·dO  — TRANSPOSE-ELIMINATED: A = Pᵀ read DIRECTLY from swizzled sP.
        run_gemm_dVdK_half_te(dv, sP, sdO_sw[s] + wg * 4096);
        consumer_sync();

        // dS = P ⊙ (dP − D) → swizzled sP.  D loaded by wg2 → wait d_ready[s] first.
        mbar_wait_v4(&d_ready[s], dpar[s]); dpar[s] ^= 1;
        {   // V18/V19: vectorize dS — 2 adjacent columns/step (bf16×2 + float2).
            const int cbase = (2 * tid) % Bc, c8 = cbase >> 3, clo = cbase & 7;
            for (int pp = tid; pp < Br * Bc / 2; pp += CONS) {
                const int r    = (2 * pp) / Bc;
                const int pidx = r * 64 + ((c8 ^ (r & 7)) << 3) + clo;
                const int sdi  = r * SS_STRIDE_V6 + cbase;
                const float d  = sD[it & 1][r];   // sD is 2-deep by TILE parity
                const __nv_bfloat162 p2 = *reinterpret_cast<const __nv_bfloat162*>(&sP[pidx]);
                const float2 pf  = __bfloat1622float2(p2);
                const float2 res = make_float2(pf.x * (sdP[sdi] - d), pf.y * (sdP[sdi + 1] - d));
                *reinterpret_cast<__nv_bfloat162*>(&sP[pidx]) = __float22bfloat162_rn(res);
            }
        }
        consumer_sync();   // dS writes visible cross-warp before the transposed dK read

        // dK += dSᵀ·Q  — TRANSPOSE-ELIMINATED: A = dSᵀ read DIRECTLY from swizzled sP.
        run_gemm_dVdK_half_te(dk, sP, sQ_sw[s] + wg * 4096);
        consumer_sync();
        if (tid == 0) mbar_arrive_v11(&empty[s]);

        // dQ_tile = dS·K (transient) — dS staged K-major into sA_t from swizzled sP.
        sp_to_sAt_v16_dq(sA_t, sP, tid);
        consumer_sync();
        { float acc[32]; zeroN<32>(acc);
          run_gemm_dQ_half(acc, sA_t, sK_sw + wg * 4096, sbo_pad_v6(Bc));
          stage_acc_f32<64>(acc, (wg == 0) ? sS : sdP, wtid, scale); }
        consumer_sync();
        atomic_flush_stage<Br, 64>((wg == 0) ? sS : sdP, d_dq_accum, lBaseOf(gC, qcC) * D, D, wg * 64, wtid);
        consumer_sync();
        if (++qcC == nQTiles) { qcC = qc0; ++gC; }
    }

    // ── Epilogue — coalesced dV/dK writeback (identical to V21). ──
    bf16 *stage = sA_t + wg * 4096;
    fence_operandN<32>(dv);
    stage_acc_bf16<64>(dv, stage, wtid, 1.0f);
    consumer_sync();
    store_stage_vec<Bc, 64>(stage, d_dV, kvBase, D, wg * 64, wtid);
    consumer_sync();
    fence_operandN<32>(dk);
    stage_acc_bf16<64>(dk, stage, wtid, scale);
    consumer_sync();
    store_stage_vec<Bc, 64>(stage, d_dK, kvBase, D, wg * 64, wtid);
}

// ── V22 launcher — runs compute_drowsum_v22 FIRST (fills d_Drow), then the main
// kernel (reads d_Drow, no O TMA), then convert_dq_accum_to_bf16_v5.  All three
// inside the launcher so the benchmark times the TOTAL (D-kernel + main + convert),
// an apples-to-apples comparison with cuDNN whose 2.7 ms already includes its D
// kernel.  d_Drow is a cached static alloc (like d_dq_accum).
template<int Br, int Bc, int D>
void launch_gqa_backward_v22(
    const bf16 *d_Q, const bf16 *d_K, const bf16 *d_V, const bf16 *d_O,
    const bf16 *d_dO, const float *d_LSE,
    bf16 *d_dQ, bf16 *d_dK, bf16 *d_dV,
    int B, int Hq, int Hkv, int G, int S, float scale
) {
    static_assert(Br == 64 && Bc == 64 && D == 128, "V22 requires Br=Bc=64, D=128");

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
    CUtensorMap tma_dO_sw = make_tma_sw128(d_dO, Rq,  Br);   // V22: no tma_O_sw

    // Cached static d_Drow buffer — one fp32 per (b,hq,s) row = B*Hq*S floats.
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

    // (1) Precompute D = Σ_d dO·O → d_Drow (one warp / row, 8 rows / block).
    const int  dBlock = 256;                       // 8 warps / block → 8 rows
    const long dGrid  = (drowN + (dBlock / 32) - 1) / (dBlock / 32);
    compute_drowsum_v22<<<(unsigned)dGrid, dBlock>>>(d_dO, d_O, d_Drow, drowN);

    // (2) Main backward kernel (reads d_Drow, drops the inline D-rowsum + O TMA).
    constexpr dim3 BLOCK(384);
    dim3 GRID(B, Hkv, S / Bc);
    gqa_backward_v22_kv<Br,Bc,D><<<GRID, BLOCK>>>(
        tma_K_sw, tma_V_sw, tma_Q_sw, tma_dO_sw,
        d_Drow, d_LSE, d_dK, d_dV, d_dq_accum, B, Hq, Hkv, G, S, scale);

    // (3) dQ fp32 accumulator → bf16.
    const int convBlock = 256;
    const int convGrid  = (int)((dqN + convBlock - 1) / convBlock);
    convert_dq_accum_to_bf16_v5<<<convGrid, convBlock>>>(d_dq_accum, d_dQ, dqN);
}





// ═════════════════════════════════════════════════════════════════════════════

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

    launch_gqa_backward_v6<Br2,Bc2,D>(d_Q,d_K,d_V,d_O,d_dO,d_LSE,d_dQ,d_dK,d_dV,B,Hq,Hkv,G,S,scale);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    check("── V6  Br=64, Bc=64  FUSED KV-centric, BANK-CONFLICT-FREE (padded smem) ──", Nq, Nkv, d_dQ, d_dK, d_dV);

    launch_gqa_backward_v7<Br2,Bc2,D>(d_Q,d_K,d_V,d_O,d_dO,d_LSE,d_dQ,d_dK,d_dV,B,Hq,Hkv,G,S,scale);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    check("── V7  Br=64, Bc=64  SWIZZLED-TMA + wgmma Major::MN (transpose-buffer-free) ──", Nq, Nkv, d_dQ, d_dK, d_dV);

    launch_gqa_backward_v8<Br2,Bc2,D>(d_Q,d_K,d_V,d_O,d_dO,d_LSE,d_dQ,d_dK,d_dV,B,Hq,Hkv,G,S,scale);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    check("── V8  Br=64, Bc=64  warp-shuffle D-rowsum reduction ──", Nq, Nkv, d_dQ, d_dK, d_dV);

    launch_gqa_backward_v9<Br2,Bc2,D>(d_Q,d_K,d_V,d_O,d_dO,d_LSE,d_dQ,d_dK,d_dV,B,Hq,Hkv,G,S,scale);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    check("── V9  Br=64, Bc=64  2-warpgroup work-split (256 thr, occupancy 12.5%) ──", Nq, Nkv, d_dQ, d_dK, d_dV);

    launch_gqa_backward_v10<Br2,Bc2,D>(d_Q,d_K,d_V,d_O,d_dO,d_LSE,d_dQ,d_dK,d_dV,B,Hq,Hkv,G,S,scale);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    check("── V10 Br=64, Bc=64  coalesced writeback (Hopper SM_90) ──", Nq, Nkv, d_dQ, d_dK, d_dV);

    launch_gqa_backward_v11<Br2,Bc2,D>(d_Q,d_K,d_V,d_O,d_dO,d_LSE,d_dQ,d_dK,d_dV,B,Hq,Hkv,G,S,scale);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    check("── V11 Br=64 Bc=64 3-warpgroup (occ 18.75%) (Hopper SM_90) ──", Nq, Nkv, d_dQ, d_dK, d_dV);

    launch_gqa_backward_v12<Br2,Bc2,D>(d_Q,d_K,d_V,d_O,d_dO,d_LSE,d_dQ,d_dK,d_dV,B,Hq,Hkv,G,S,scale);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    check("── V12 Br=64 Bc=64 ldmatrix/stmatrix shared-traffic (Hopper SM_90) ──", Nq, Nkv, d_dQ, d_dK, d_dV);

    launch_gqa_backward_v13<Br2,Bc2,D>(d_Q,d_K,d_V,d_O,d_dO,d_LSE,d_dQ,d_dK,d_dV,B,Hq,Hkv,G,S,scale);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    check("── V13 Br=64 Bc=64 fused-softmax + producer D-rowsum (Hopper SM_90) ──", Nq, Nkv, d_dQ, d_dK, d_dV);

    launch_gqa_backward_v14<Br2,Bc2,D>(d_Q,d_K,d_V,d_O,d_dO,d_LSE,d_dQ,d_dK,d_dV,B,Hq,Hkv,G,S,scale);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    check("── V14 Br=64 Bc=64 __expf fast-SFU softmax (Hopper SM_90) ──", Nq, Nkv, d_dQ, d_dK, d_dV);

    launch_gqa_backward_v15<Br2,Bc2,D>(d_Q,d_K,d_V,d_O,d_dO,d_LSE,d_dQ,d_dK,d_dV,B,Hq,Hkv,G,S,scale);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    check("── V15 Br=64 Bc=64 incremental (g,qc) address (Hopper SM_90) ──", Nq, Nkv, d_dQ, d_dK, d_dV);

    launch_gqa_backward_v16<Br2,Bc2,D>(d_Q,d_K,d_V,d_O,d_dO,d_LSE,d_dQ,d_dK,d_dV,B,Hq,Hkv,G,S,scale);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    check("── V16 Br=64 Bc=64 transpose-elim direct-sP wgmma (Hopper SM_90) ──", Nq, Nkv, d_dQ, d_dK, d_dV);

    launch_gqa_backward_v17<Br2,Bc2,D>(d_Q,d_K,d_V,d_O,d_dO,d_LSE,d_dQ,d_dK,d_dV,B,Hq,Hkv,G,S,scale);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    check("── V17 Br=64 Bc=64 hoisted swizzle-index write (Hopper SM_90) ──", Nq, Nkv, d_dQ, d_dK, d_dV);

    launch_gqa_backward_v18<Br2,Bc2,D>(d_Q,d_K,d_V,d_O,d_dO,d_LSE,d_dQ,d_dK,d_dV,B,Hq,Hkv,G,S,scale);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    check("── V18 Br=64 Bc=64 vectorized dS bf16x2 (Hopper SM_90) ──", Nq, Nkv, d_dQ, d_dK, d_dV);

    launch_gqa_backward_v19<Br2,Bc2,D>(d_Q,d_K,d_V,d_O,d_dO,d_LSE,d_dQ,d_dK,d_dV,B,Hq,Hkv,G,S,scale);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    check("── V19 Br=64 Bc=64 vectorized P-write bf16x2 (Hopper SM_90) ──", Nq, Nkv, d_dQ, d_dK, d_dV);

    launch_gqa_backward_v20<Br2,Bc2,D>(d_Q,d_K,d_V,d_O,d_dO,d_LSE,d_dQ,d_dK,d_dV,B,Hq,Hkv,G,S,scale);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    check("── V20 Br=64 Bc=64 swizzled-Drowsum + 3-deep pipeline (Hopper SM_90) ──", Nq, Nkv, d_dQ, d_dK, d_dV);

    launch_gqa_backward_v21<Br2,Bc2,D>(d_Q,d_K,d_V,d_O,d_dO,d_LSE,d_dQ,d_dK,d_dV,B,Hq,Hkv,G,S,scale);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    check("── V21 Br=64 Bc=64 causal-mask specialization (Hopper SM_90) ──", Nq, Nkv, d_dQ, d_dK, d_dV);

    launch_gqa_backward_v22<Br2,Bc2,D>(d_Q,d_K,d_V,d_O,d_dO,d_LSE,d_dQ,d_dK,d_dV,B,Hq,Hkv,G,S,scale);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    check("── V22 Br=64 Bc=64 D-rowsum split (separate kernel) (Hopper SM_90) ──", Nq, Nkv, d_dQ, d_dK, d_dV);

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
    {
        KernelStats s = benchmarkKernel(
            [&](){ launch_gqa_backward_v6<Br2,Bc2,D>(
                d_Q,d_K,d_V,d_O,d_dO,d_LSE,d_dQ,d_dK,d_dV,B,Hq,Hkv,G,S,scale); },
            100, 10, bwd_flops);
        displayStats("GQA bwd V6  Br=64, Bc=64  BANK-CONFLICT-FREE  (Hopper SM_90)", s);
    }
    {
        KernelStats s = benchmarkKernel(
            [&](){ launch_gqa_backward_v7<Br2,Bc2,D>(
                d_Q,d_K,d_V,d_O,d_dO,d_LSE,d_dQ,d_dK,d_dV,B,Hq,Hkv,G,S,scale); },
            100, 10, bwd_flops);
        displayStats("GQA bwd V7  Br=64, Bc=64  SWIZZLED-MN transpose-free  (Hopper SM_90)", s);
    }
    {
        KernelStats s = benchmarkKernel(
            [&](){ launch_gqa_backward_v8<Br2,Bc2,D>(
                d_Q,d_K,d_V,d_O,d_dO,d_LSE,d_dQ,d_dK,d_dV,B,Hq,Hkv,G,S,scale); },
            100, 10, bwd_flops);
        displayStats("GQA bwd V8  Br=64, Bc=64  warp-shuffle D-rowsum reduction  (Hopper SM_90)", s);
    }
    {
        KernelStats s = benchmarkKernel(
            [&](){ launch_gqa_backward_v9<Br2,Bc2,D>(
                d_Q,d_K,d_V,d_O,d_dO,d_LSE,d_dQ,d_dK,d_dV,B,Hq,Hkv,G,S,scale); },
            100, 10, bwd_flops);
        displayStats("GQA bwd V9  Br=64, Bc=64  2-warpgroup work-split  (Hopper SM_90)", s);
    }
    {
        KernelStats s = benchmarkKernel(
            [&](){ launch_gqa_backward_v10<Br2,Bc2,D>(
                d_Q,d_K,d_V,d_O,d_dO,d_LSE,d_dQ,d_dK,d_dV,B,Hq,Hkv,G,S,scale); },
            100, 10, bwd_flops);
        displayStats("GQA bwd V10 Br=64, Bc=64  coalesced writeback  (Hopper SM_90)", s);
    }
    {
        KernelStats s = benchmarkKernel(
            [&](){ launch_gqa_backward_v11<Br2,Bc2,D>(
                d_Q,d_K,d_V,d_O,d_dO,d_LSE,d_dQ,d_dK,d_dV,B,Hq,Hkv,G,S,scale); },
            100, 10, bwd_flops);
        displayStats("GQA bwd V11 Br=64, Bc=64  3-warpgroup (occ 18.75%)  (Hopper SM_90)", s);
    }
    {
        KernelStats s = benchmarkKernel(
            [&](){ launch_gqa_backward_v12<Br2,Bc2,D>(
                d_Q,d_K,d_V,d_O,d_dO,d_LSE,d_dQ,d_dK,d_dV,B,Hq,Hkv,G,S,scale); },
            100, 10, bwd_flops);
        displayStats("GQA bwd V12 Br=64, Bc=64  ldmatrix/stmatrix shared-traffic  (Hopper SM_90)", s);
    }
    {
        KernelStats s = benchmarkKernel(
            [&](){ launch_gqa_backward_v13<Br2,Bc2,D>(
                d_Q,d_K,d_V,d_O,d_dO,d_LSE,d_dQ,d_dK,d_dV,B,Hq,Hkv,G,S,scale); },
            100, 10, bwd_flops);
        displayStats("GQA bwd V13 Br=64, Bc=64  fused-softmax + producer D-rowsum  (Hopper SM_90)", s);
    }
    {
        KernelStats s = benchmarkKernel(
            [&](){ launch_gqa_backward_v14<Br2,Bc2,D>(
                d_Q,d_K,d_V,d_O,d_dO,d_LSE,d_dQ,d_dK,d_dV,B,Hq,Hkv,G,S,scale); },
            100, 10, bwd_flops);
        displayStats("GQA bwd V14 Br=64, Bc=64  __expf fast-SFU softmax  (Hopper SM_90)", s);
    }
    {
        KernelStats s = benchmarkKernel(
            [&](){ launch_gqa_backward_v15<Br2,Bc2,D>(
                d_Q,d_K,d_V,d_O,d_dO,d_LSE,d_dQ,d_dK,d_dV,B,Hq,Hkv,G,S,scale); },
            100, 10, bwd_flops);
        displayStats("GQA bwd V15 Br=64, Bc=64  incremental (g,qc) address  (Hopper SM_90)", s);
    }
    {
        KernelStats s = benchmarkKernel(
            [&](){ launch_gqa_backward_v16<Br2,Bc2,D>(
                d_Q,d_K,d_V,d_O,d_dO,d_LSE,d_dQ,d_dK,d_dV,B,Hq,Hkv,G,S,scale); },
            100, 10, bwd_flops);
        displayStats("GQA bwd V16 Br=64, Bc=64  transpose-elim  (Hopper SM_90)", s);
    }
    {
        KernelStats s = benchmarkKernel(
            [&](){ launch_gqa_backward_v17<Br2,Bc2,D>(
                d_Q,d_K,d_V,d_O,d_dO,d_LSE,d_dQ,d_dK,d_dV,B,Hq,Hkv,G,S,scale); },
            100, 10, bwd_flops);
        displayStats("GQA bwd V17 Br=64, Bc=64  hoisted swizzle-index  (Hopper SM_90)", s);
    }
    {
        KernelStats s = benchmarkKernel(
            [&](){ launch_gqa_backward_v18<Br2,Bc2,D>(
                d_Q,d_K,d_V,d_O,d_dO,d_LSE,d_dQ,d_dK,d_dV,B,Hq,Hkv,G,S,scale); },
            100, 10, bwd_flops);
        displayStats("GQA bwd V18 Br=64, Bc=64  vectorized dS bf16x2  (Hopper SM_90)", s);
    }
    {
        KernelStats s = benchmarkKernel(
            [&](){ launch_gqa_backward_v19<Br2,Bc2,D>(
                d_Q,d_K,d_V,d_O,d_dO,d_LSE,d_dQ,d_dK,d_dV,B,Hq,Hkv,G,S,scale); },
            100, 10, bwd_flops);
        displayStats("GQA bwd V19 Br=64, Bc=64  vectorized P-write bf16x2  (Hopper SM_90)", s);
    }
    {
        KernelStats s = benchmarkKernel(
            [&](){ launch_gqa_backward_v20<Br2,Bc2,D>(
                d_Q,d_K,d_V,d_O,d_dO,d_LSE,d_dQ,d_dK,d_dV,B,Hq,Hkv,G,S,scale); },
            100, 10, bwd_flops);
        displayStats("GQA bwd V20 Br=64, Bc=64  swizzled-Drowsum + 3-deep pipeline  (Hopper SM_90)", s);
    }
    {
        KernelStats s = benchmarkKernel(
            [&](){ launch_gqa_backward_v21<Br2,Bc2,D>(
                d_Q,d_K,d_V,d_O,d_dO,d_LSE,d_dQ,d_dK,d_dV,B,Hq,Hkv,G,S,scale); },
            100, 10, bwd_flops);
        displayStats("GQA bwd V21 Br=64, Bc=64  causal-mask specialization  (Hopper SM_90)", s);
    }
    {
        KernelStats s = benchmarkKernel(
            [&](){ launch_gqa_backward_v22<Br2,Bc2,D>(
                d_Q,d_K,d_V,d_O,d_dO,d_LSE,d_dQ,d_dK,d_dV,B,Hq,Hkv,G,S,scale); },
            100, 10, bwd_flops);
        displayStats("GQA bwd V22 Br=64, Bc=64  D-rowsum split (separate kernel)  (Hopper SM_90)", s);
    }
    CUDA_CHECK(cudaFree(d_Q));  CUDA_CHECK(cudaFree(d_K));  CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));  CUDA_CHECK(cudaFree(d_dO)); CUDA_CHECK(cudaFree(d_LSE));
    CUDA_CHECK(cudaFree(d_dQ)); CUDA_CHECK(cudaFree(d_dK)); CUDA_CHECK(cudaFree(d_dV));
    return 0;
}