// Standalone probe for Vr1's dK/dV G-head reduce kernel (cuDNN's fmha_reduce_head equivalent).
// Sums the G query-heads that share a KV head: partial[B,Hq,S,D] -> out[B,Hkv,S,D], Hq=Hkv*G.
// Goal: confirm it's ~18-40us (cuDNN measured 18.78+18.62us for dK+dV) BEFORE reworking the main kernel.
// If the reduce is this cheap, offloading the inline G-accumulation is worth it iff the per-hq main
// kernel (G x more, shorter blocks) gains more than the reduce costs.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
typedef __nv_bfloat16 bf16;
#define CK(x) do{cudaError_t e=(x);if(e){printf("cuda %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)

constexpr int RPB = 8;   // s-rows per block (cuDNN grid = S/8 x Hkv x B)

// grid (S/RPB, Hkv, B), block (D=128). thread d sums G heads for RPB rows.
__global__ void greduce(const bf16* __restrict__ partial, bf16* __restrict__ out,
                        int Hq, int Hkv, int G, int S, int D) {
    int b = blockIdx.z, hkv = blockIdx.y, s0 = blockIdx.x * RPB, d = threadIdx.x;
#pragma unroll
    for (int r = 0; r < RPB; r++) {
        int s = s0 + r;
        float acc = 0.f;
#pragma unroll 1
        for (int g = 0; g < G; g++) {
            int hq = hkv * G + g;
            acc += __bfloat162float(partial[((long)(b * Hq + hq) * S + s) * D + d]);
        }
        out[((long)(b * Hkv + hkv) * S + s) * D + d] = __float2bfloat16(acc);
    }
}

int main() {
    const int B = 4, Hkv = 4, G = 3, S = 4096, D = 128;
    const int Hq = Hkv * G;                       // 12
    const long np = (long)B * Hq  * S * D;        // partial elems  = 25.2M -> 50MB bf16
    const long no = (long)B * Hkv * S * D;        // out elems      = 8.4M  -> 16MB
    bf16 *dpK, *dpV, *doK, *doV;
    CK(cudaMalloc(&dpK, np * 2)); CK(cudaMalloc(&dpV, np * 2));
    CK(cudaMalloc(&doK, no * 2)); CK(cudaMalloc(&doV, no * 2));
    // fill partials with a known pattern for correctness check
    bf16* h = (bf16*)malloc(np * 2);
    for (long i = 0; i < np; i++) h[i] = __float2bfloat16((float)((i % 7) - 3) * 0.25f);
    CK(cudaMemcpy(dpK, h, np * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dpV, h, np * 2, cudaMemcpyHostToDevice));

    dim3 grid(S / RPB, Hkv, B), block(D);
    // warmup
    for (int i = 0; i < 5; i++) {
        greduce<<<grid, block>>>(dpK, doK, Hq, Hkv, G, S, D);
        greduce<<<grid, block>>>(dpV, doV, Hq, Hkv, G, S, D);
    }
    CK(cudaDeviceSynchronize());
    // time (dK + dV = one "reduce pass", like cuDNN's two fmha_reduce_head launches)
    cudaEvent_t a, bE; cudaEventCreate(&a); cudaEventCreate(&bE);
    const int IT = 200;
    cudaEventRecord(a);
    for (int i = 0; i < IT; i++) {
        greduce<<<grid, block>>>(dpK, doK, Hq, Hkv, G, S, D);
        greduce<<<grid, block>>>(dpV, doV, Hq, Hkv, G, S, D);
    }
    cudaEventRecord(bE); CK(cudaEventSynchronize(bE));
    float ms; cudaEventElapsedTime(&ms, a, bE);
    double per = ms / IT * 1000.0;  // us per (dK+dV) pass
    printf("G-reduce dK+dV: %.2f us/pass  (single kernel %.2f us)  [B=%d Hq=%d Hkv=%d G=%d S=%d D=%d]\n",
           per, per / 2, B, Hq, Hkv, G, S, D);

    // correctness: out[b,hkv,s,d] == sum_g partial[b,hkv*G+g,s,d]
    bf16* ho = (bf16*)malloc(no * 2); CK(cudaMemcpy(ho, doK, no * 2, cudaMemcpyDeviceToHost));
    int bad = 0;
    for (long t = 0; t < 20000 && bad < 5; t++) {
        long idx = (t * 9973) % no;
        int d = idx % D; long sh = idx / D; int s = sh % S; long bh = sh / S; int hkv = bh % Hkv, b = bh / Hkv;
        float want = 0.f;
        for (int g = 0; g < G; g++) want += __bfloat162float(h[((long)(b * Hq + hkv * G + g) * S + s) * D + d]);
        float got = __bfloat162float(ho[idx]);
        if (fabsf(got - want) > 0.02f) { printf("mismatch @%ld got %.3f want %.3f\n", idx, got, want); bad++; }
    }
    printf(bad ? "FAIL\n" : "PASS: G-reduce sums correctly\n");
    return 0;
}
