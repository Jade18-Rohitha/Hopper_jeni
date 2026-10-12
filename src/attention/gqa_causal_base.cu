// ─────────────────────────────────────────────────────────────────────────────
// GQA Backward (causal) — base version  (V2 from GQA_bwd.cu)
//
//   V2  Br=64, Bc=64   4-warp (128 threads) per block
//   WMMA tensor cores, SM_70+ compatible, no wgmma / TMA.
//
//   Kernel 1 (dQ)   grid (B, Hq,  S/Br) — one block per Q tile
//   Kernel 2 (dKdV) grid (B, Hkv, S/Bc) — one block per K/V tile, loops over G
// ─────────────────────────────────────────────────────────────────────────────
//
// Standalone: no project headers.  Build & run (from Hopper/):
//   nvcc -O3 -arch=sm_86 -std=c++17 src/attention/gqa_causal_base.cu -o gqa_causal_base
//   python precision/baseline_gqa.py      # writes data/gqa_*.bin
//   ./gqa_causal_base [B Hq Hkv]
// ─────────────────────────────────────────────────────────────────────────────
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

using namespace nvcuda;

#define CUDA_CHECK(call) do {                                                  \
    cudaError_t err_ = (call);                                                 \
    if (err_ != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                            \
                cudaGetErrorString(err_), __FILE__, __LINE__);                 \
        exit(1);                                                               \
    }                                                                          \
} while (0)

// ─────────────────────────────────────────────────────────────────────────────
// V2 — WMMA 4 warps (Br=64, Bc=64) — SM_70+ compatible, no wgmma
//
// Target: RTX 3060 (sm_86) — max 99 KB dynamic shared memory per block.
// The original V2 used 123 KB of static __shared__ at D=128, so two
// math-neutral cuts bring it to 88.5 KB:
//   1. no sO tile — O is only used in the D = rowsum(dO*O) dot product, so it
//      is read straight from global there (same summation order).
//   2. sdP aliases sS — sS is last read by the P = exp(S - LSE) pass, which is
//      followed by __syncthreads() before dP is stored.
// Layout (every buffer is a multiple of 16 B, so each carve stays aligned):
//   sQ sdO sK sV bf16 [64*D] ×4 | sS(=sdP) fp32 [64*64] | sP bf16 [64*64]
//   sLSE sD fp32 [64] ×2
// ─────────────────────────────────────────────────────────────────────────────

template<int Br, int Bc, int D>
constexpr size_t gqa_v2_smem_bytes() {
    return (size_t)(2 * Br * D + 2 * Bc * D) * sizeof(__nv_bfloat16)   // sQ sdO sK sV
         + (size_t)(Br * Bc) * sizeof(float)                           // sS (= sdP)
         + (size_t)(Br * Bc) * sizeof(__nv_bfloat16)                   // sP
         + (size_t)(2 * Br) * sizeof(float);                           // sLSE sD
}

template<int Br, int Bc, int D>
struct GqaV2Smem {
    __nv_bfloat16 *sQ, *sdO, *sK, *sV, *sP;
    float *sS, *sdP, *sLSE, *sD;
    __device__ explicit GqaV2Smem(unsigned char *base) {
        __nv_bfloat16 *h = reinterpret_cast<__nv_bfloat16 *>(base);
        sQ  = h; h += Br * D;
        sdO = h; h += Br * D;
        sK  = h; h += Bc * D;
        sV  = h; h += Bc * D;
        float *f = reinterpret_cast<float *>(h);
        sS  = f;
        sdP = f; f += Br * Bc;   // alias: sS is dead before dP is stored
        sP  = reinterpret_cast<__nv_bfloat16 *>(f);
        f   = reinterpret_cast<float *>(sP + Br * Bc);
        sLSE = f; f += Br;
        sD   = f;
    }
};

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

    extern __shared__ __align__(16) unsigned char smem_v2[];
    const GqaV2Smem<Br, Bc, D> sm(smem_v2);
    __nv_bfloat16 * const sQ  = sm.sQ;
    __nv_bfloat16 * const sdO = sm.sdO;
    __nv_bfloat16 * const sK  = sm.sK;
    __nv_bfloat16 * const sV  = sm.sV;
    float         * const sS  = sm.sS;
    float         * const sdP = sm.sdP;
    __nv_bfloat16 * const sP  = sm.sP;    // reused as sdS
    float         * const sLSE = sm.sLSE;
    float         * const sD   = sm.sD;

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
    }
    if (tid < Br) sLSE[tid] = d_LSE[lBase + tid];
    __syncthreads();
//D
    if (lane < 16) {
        int r = row0 + lane;
        float d = 0.f;
        for (int j = 0; j < D; j++)
            d += __bfloat162float(sdO[r * D + j]) * __bfloat162float(d_O[qBase + r * D + j]);
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

    extern __shared__ __align__(16) unsigned char smem_v2[];
    const GqaV2Smem<Br, Bc, D> sm(smem_v2);
    __nv_bfloat16 * const sQ  = sm.sQ;
    __nv_bfloat16 * const sdO = sm.sdO;
    __nv_bfloat16 * const sK  = sm.sK;
    __nv_bfloat16 * const sV  = sm.sV;
    float         * const sS  = sm.sS;
    float         * const sdP = sm.sdP;
    __nv_bfloat16 * const sP  = sm.sP;    // reused as sdS
    float         * const sLSE = sm.sLSE;
    float         * const sD   = sm.sD;

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
            }
            if (tid < Br) sLSE[tid] = d_LSE[lBase + tid];
            __syncthreads();
//D
            if (tid < Br) {
                float d = 0.f;
                for (int j = 0; j < D; j++)
                    d += __bfloat162float(sdO[tid * D + j]) * __bfloat162float(d_O[qBase + tid * D + j]);
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

    constexpr size_t SMEM = gqa_v2_smem_bytes<Br, Bc, D>();
    static bool attr_set = false;
    if (!attr_set) {
        CUDA_CHECK(cudaFuncSetAttribute(gqa_backward_dQ_v2<Br,Bc,D>,
                   cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM));
        CUDA_CHECK(cudaFuncSetAttribute(gqa_backward_dKdV_v2<Br,Bc,D>,
                   cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM));
        attr_set = true;
    }

    constexpr dim3 BLOCK(128);
    dim3 GRID1(B, Hq,  S / Br);
    dim3 GRID2(B, Hkv, S / Bc);
    gqa_backward_dQ_v2  <Br,Bc,D><<<GRID1, BLOCK, SMEM>>>(
        d_Q, d_K, d_V, d_O, d_dO, d_LSE, d_dQ, B, Hq, Hkv, G, S, scale);
    gqa_backward_dKdV_v2<Br,Bc,D><<<GRID2, BLOCK, SMEM>>>(
        d_Q, d_K, d_V, d_O, d_dO, d_LSE, d_dK, d_dV, B, Hq, Hkv, G, S, scale);
}

// ─────────────────────────────────────────────────────────────────────────────
// Host: load PyTorch reference, run V2, check precision, time it
// ─────────────────────────────────────────────────────────────────────────────
static bool load_f32(const char *path, std::vector<float> &dst, size_t n) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    size_t bytes = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    if (bytes != n * sizeof(float)) { fclose(f); return false; }
    dst.resize(n);
    size_t got = fread(dst.data(), sizeof(float), n, f);
    fclose(f);
    return got == n;
}

static bool check(const char *name, const std::vector<float> &ref,
                  const std::vector<__nv_bfloat16> &out, float atol, float rtol) {
    double max_abs = 0.0, sum_abs = 0.0;
    size_t bad = 0, first_bad = (size_t)-1;
    for (size_t i = 0; i < ref.size(); ++i) {
        float o = __bfloat162float(out[i]);
        double e = fabs((double)o - (double)ref[i]);
        if (!(e <= atol + rtol * fabs((double)ref[i]))) {   // also catches NaN
            if (bad == 0) first_bad = i;
            ++bad;
        }
        if (e > max_abs || std::isnan(e)) max_abs = e;
        sum_abs += e;
    }
    printf("  %s: max_abs=%.3e  mean_abs=%.3e  mismatches=%zu/%zu  %s\n",
           name, max_abs, sum_abs / ref.size(), bad, ref.size(), bad ? "FAIL" : "PASS");
    if (bad)
        printf("      first mismatch @%zu: ref=%.6f got=%.6f\n",
               first_bad, ref[first_bad], __bfloat162float(out[first_bad]));
    return bad == 0;
}

int main(int argc, char **argv) {
    const int B   = argc > 1 ? atoi(argv[1]) : 4;
    const int Hq  = argc > 2 ? atoi(argv[2]) : 24;
    const int Hkv = argc > 3 ? atoi(argv[3]) : 8;
    const int G   = Hq / Hkv;
    constexpr int S = 4096, D = 128, Br = 64, Bc = 64;
    const float scale = 1.0f / sqrtf((float)D);

    printf("GQA causal backward — V2 base (WMMA, Br=Bc=64)\n");
    printf("  shape: B=%d Hq=%d Hkv=%d (G=%d) S=%d D=%d\n\n", B, Hq, Hkv, G, S, D);
    if (Hq % Hkv != 0) { fprintf(stderr, "ERROR: Hq must be a multiple of Hkv\n"); return 1; }

    const size_t Nq   = (size_t)B * Hq  * S * D;
    const size_t Nkv  = (size_t)B * Hkv * S * D;
    const size_t Nlse = (size_t)B * Hq  * S;

    std::vector<float> fQ, fK, fV, fO, fdO, hLSE, rdQ, rdK, rdV;
    if (!load_f32("data/gqa_q.bin",   fQ,   Nq)   || !load_f32("data/gqa_k.bin",  fK,  Nkv) ||
        !load_f32("data/gqa_v.bin",   fV,   Nkv)  || !load_f32("data/gqa_o.bin",  fO,  Nq)  ||
        !load_f32("data/gqa_do.bin",  fdO,  Nq)   || !load_f32("data/gqa_lse.bin", hLSE, Nlse) ||
        !load_f32("data/gqa_dq.bin",  rdQ,  Nq)   || !load_f32("data/gqa_dk.bin", rdK, Nkv) ||
        !load_f32("data/gqa_dv.bin",  rdV,  Nkv)) {
        fprintf(stderr, "ERROR: reference files missing or wrong size for this shape.\n"
                        "       Run: python precision/baseline_gqa.py\n");
        return 1;
    }

    auto to_bf16 = [](const std::vector<float> &src) {
        std::vector<__nv_bfloat16> dst(src.size());
        for (size_t i = 0; i < src.size(); ++i) dst[i] = __float2bfloat16(src[i]);
        return dst;
    };
    std::vector<__nv_bfloat16> hQ = to_bf16(fQ), hK = to_bf16(fK), hV = to_bf16(fV),
                               hO = to_bf16(fO), hdO = to_bf16(fdO);

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
    CUDA_CHECK(cudaMemcpy(d_Q,   hQ.data(),   Nq   * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K,   hK.data(),   Nkv  * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V,   hV.data(),   Nkv  * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_O,   hO.data(),   Nq   * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dO,  hdO.data(),  Nq   * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_LSE, hLSE.data(), Nlse * sizeof(float),         cudaMemcpyHostToDevice));

    // ── correctness ──
    launch_gqa_backward_v2<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_dO, d_LSE, d_dQ, d_dK, d_dV,
                                      B, Hq, Hkv, G, S, scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> odQ(Nq), odK(Nkv), odV(Nkv);
    CUDA_CHECK(cudaMemcpy(odQ.data(), d_dQ, Nq  * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(odK.data(), d_dK, Nkv * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(odV.data(), d_dV, Nkv * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    printf("── V2  Br=64, Bc=64 ──\n");
    bool ok = true;
    ok &= check("dQ", rdQ, odQ, 2e-2f, 2e-2f);
    ok &= check("dK", rdK, odK, 2e-2f, 2e-2f);
    ok &= check("dV", rdV, odV, 2e-2f, 2e-2f);

    // ── timing ──
    const int warmup = 3, iters = 10;
    for (int i = 0; i < warmup; ++i)
        launch_gqa_backward_v2<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_dO, d_LSE, d_dQ, d_dK, d_dV,
                                          B, Hq, Hkv, G, S, scale);
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < iters; ++i)
        launch_gqa_backward_v2<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_dO, d_LSE, d_dQ, d_dK, d_dV,
                                          B, Hq, Hkv, G, S, scale);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaGetLastError());
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    ms /= iters;
    // causal bwd: 5 GEMMs of 2*S*S*D each over Hq heads, halved by the mask
    const double flops = 5.0 * 2.0 * B * Hq * (double)S * S * D * 0.5;
    printf("\n  time: %.3f ms   (%.1f TFLOP/s causal)\n", ms, flops / (ms * 1e-3) / 1e12);
    printf("\n%s\n", ok ? "ALL PASS" : "SOME CHECKS FAILED");

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O); cudaFree(d_dO);
    cudaFree(d_LSE); cudaFree(d_dQ); cudaFree(d_dK); cudaFree(d_dV);
    return ok ? 0 : 1;
}
