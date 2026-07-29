// ═════════════════════════════════════════════════════════════════════════════
// wgmma_transpose_probe.cu  —  Hopper sm_90a  —  RESEARCH / de-risking probe
// ─────────────────────────────────────────────────────────────────────────────
// PURPOSE
//   Decide whether V14 can read a TRANSPOSED A operand (Pᵀ for dV=Pᵀ·dO) DIRECTLY
//   from a resident swizzled smem buffer via a transposed wgmma descriptor — the
//   FA3 "layout-aliasing" trick that deletes V13's sA_t staging round-trip
//   (sp_to_sAt_v12 / fill_trans_A).  This is a STANDALONE probe: it is NOT wired
//   into GQA_bwd.cu and changes no kernel.  Compile-verified for sm_90a on a
//   non-Hopper box; RUN ON THE H200.
//
// WHAT IT TESTS  (single m64n64k16 bf16 wgmma, dV = Pᵀ·B style)
//   Contraction K = r (query rows, 64).  Output M = c (key cols, 64).  Output N = n
//   (64).  Two swizzled-A candidates, SAME logical GEMM, SAME CPU reference:
//
//     Candidate FA3  (the V14 proposal — what flash-attention bwd actually does):
//       A = Pᵀ read GMMA::Major::MN, trans-a=1, from a 128B-swizzled buffer whose
//           CONTIGUOUS dim is the OUTPUT key(c);  B read Major::MN, trans-b=1.
//       → wgmma_m64n64k16_tAtB, BOTH operands make_desc_sw128_MN(+k*1024).
//       The B-side recipe is byte-identical to V7's HW-proven dO read, so a FAIL
//       here isolates the A transposed read as the cause.
//
//     Candidate V8   (the arrangement that FAILED on the H200, never root-caused):
//       A = Pᵀ read GMMA::Major::K, trans-a=0, from a 128B-swizzled buffer whose
//           CONTIGUOUS dim is the CONTRACTION query(r);  B read Major::MN, trans-b=1.
//       → wgmma_m64n64k16_tB, A make_desc_sw128_K(+k*16), B make_desc_sw128_MN.
//
//   EXPECTED on H200:  FA3 = PASS,  V8 = FAIL  →  confirms the root cause and
//   green-lights V14 transpose-elimination.  If FA3 FAILs too, descriptor-based
//   transpose-elimination is NOT viable for this buffer; fall back to reducing
//   the staging cost another way (see report).
//
// GROUND TRUTH
//   FA3 hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp:  PdS_Major = GMMA::Major::K
//   (P/dS stored swizzled, KEY-contiguous), SmemLayoutPdSt = cute::composition
//   transpose view (no data movement), PdSt_Major = GMMA::Major::MN → the dV/dK
//   GEMM reads P/dS as operand A with GMMA::Major::MN (trans-a=1).  B (dO/Q) is
//   also Major::MN.  → Candidate FA3 = both-Major::MN swizzled = wgmma tAtB.
//   PTX ISA §9.7.14: transpose (trans=1) is legal for .f16/.bf16 ONLY, and only
//   for SS-form (A from shared memory).  Both our operands are bf16 + SS-form. ✓
//
// BUILD (sm_90a; heed nvcc-13 -arch gotcha — MUST use -gencode, not -arch):
//   nvcc -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a \
//        experiments/wgmma_transpose_probe.cu -o build/wgmma_transpose_probe
//   (Compiles on any box.  Only RUN on an H100/H200.)
// ═════════════════════════════════════════════════════════════════════════════
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>

using bf16 = __nv_bfloat16;

#define CUDA_OK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
    printf("CUDA ERROR %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    exit(1);} } while(0)

// ── Ramp generators (IDENTICAL host & device) ──────────────────────────────
// Small integers → bf16-EXACT, products exact, fp32 sums exact → the PASS/FAIL is
// crisp (a correct read reproduces the reference bit-for-bit).  Asymmetric in
// (r,c)/(r,n) so a non-transposed / mis-swizzled read is DETECTABLY wrong.
__host__ __device__ __forceinline__ float rampP(int r, int c) { return (float)(((r*3 + c*5) % 17) - 8); }
__host__ __device__ __forceinline__ float rampB(int r, int n) { return (float)(((r*2 + n*7) % 11) - 5); }

// ── 128B-swizzle byte permutation for one 64-wide (128 B) bf16 atom ─────────
// row = strided index (0..63), col = contiguous index (0..63).  This is the EXACT
// byte layout CU_TENSOR_MAP_SWIZZLE_128B produces (HW-confirmed vs a TMA probe on
// sm_120 — see reference_hopper_wgmma_swizzled_inkernel_A).  Writing a buffer this
// way and reading it with make_desc_sw128_K/MN is correct by transitivity.
__host__ __device__ __forceinline__ int sw128_idx(int row, int col) {
    return row*64 + (((col>>3) ^ (row&7))<<3) + (col&7);
}

// ═════════════════════════════════════════════════════════════════════════════
// Descriptor + wgmma helpers — COPIED VERBATIM from src/attention/GQA_bwd.cu so
// the probe exercises the EXACT descriptor bits V14 will use.  Do not edit here;
// if a bit is wrong it is wrong in the kernel too.
// ═════════════════════════════════════════════════════════════════════════════
__device__ __forceinline__ uint64_t desc_encode(uint64_t x) { return (x & 0x3FFFFull) >> 4; }

// no-swizzle core-matrix-tiled K-major descriptor (unused by the candidates here,
// kept for parity / possible future no-swizzle-A comparison).
__device__ __forceinline__ uint64_t make_desc(const bf16 *smem_ptr, uint64_t sbo) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint64_t d = 0;
    d |= desc_encode((uint64_t)addr);
    d |= desc_encode((uint64_t)128) << 16;
    d |= desc_encode(sbo)           << 32;
    return d;
}

// 128B-swizzled Major::K descriptor (SBO=1024, LBO=1, swizzle-field 1<<62=B128).
__device__ __forceinline__ uint64_t make_desc_sw128_K(const bf16* smem_ptr) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint64_t d = 0;
    d |= (uint64_t)((addr >> 4) & 0x3FFFu);   // start_address       [13:0]
    d |= (uint64_t)1u        << 16;           // leading_byte_offset [29:16] (=1 per CUTLASS)
    d |= (uint64_t)(1024>>4) << 32;           // stride_byte_offset  [45:32] = 1024 B
    d |= (uint64_t)1u        << 62;           // layout_type = B128   [63:62]
    return d;
}
// Major::MN swizzled descriptor == Major::K descriptor (per reference_hopper_wgmma_mn_swizzle_d128):
// under 128B swizzle the HW ignores LBO and the same SBO strides the 8-row atom
// groups both ways; only the base-advance (caller: +k*1024) and the wgmma trans
// immediate differ.
__device__ __forceinline__ uint64_t make_desc_sw128_MN(const bf16* smem_ptr) {
    return make_desc_sw128_K(smem_ptr);
}

// m64n64k16, trans-a=0, trans-b=1  (Candidate V8: A Major::K, B Major::MN)
__device__ __forceinline__ void wgmma_m64n64k16_tB(float d[32], uint64_t descA, uint64_t descB) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
        "%32, %33, 1, 1, 1, 0, 1;\n"
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
// m64n64k16, trans-a=1, trans-b=1  (Candidate FA3 / V14: BOTH Major::MN)
__device__ __forceinline__ void wgmma_m64n64k16_tAtB(float d[32], uint64_t descA, uint64_t descB) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
        "%32, %33, 1, 1, 1, 1, 1;\n"
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
__device__ __forceinline__ void fence_proxy_async_shared() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
template<int N>
__device__ __forceinline__ void fence_operandN(float *d) {
#pragma unroll
    for (int i = 0; i < N; i++) asm volatile("" : "+f"(d[i]) :: "memory");
}
template<int N>
__device__ __forceinline__ void zeroN(float *d) {
#pragma unroll
    for (int i = 0; i < N; i++) d[i] = 0.f;
}

// Accumulator (m64n64 f32, 32 regs/thread) → global float [M=64][N=64], row stride 64.
// EXACT mapping copied from store_acc_smem<64> in GQA_bwd.cu:  row = M = key(c),
// col = N = n.  warp w owns rows [16w,16w+16); r0=lane/4, r1=r0+8, cc=(lane%4)*2.
__device__ __forceinline__ void store_acc_out(const float *d, float *g, int tid) {
    int w = tid >> 5, lane = tid & 31;
    int r0 = w*16 + (lane>>2), r1 = r0 + 8, cc = (lane & 3) * 2;
#pragma unroll
    for (int nt = 0; nt < 64/8; nt++) {
        int c = nt*8 + cc;
        g[r0*64 + c + 0] = d[nt*4 + 0];
        g[r0*64 + c + 1] = d[nt*4 + 1];
        g[r1*64 + c + 0] = d[nt*4 + 2];
        g[r1*64 + c + 1] = d[nt*4 + 3];
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Probe kernel — 1 CTA, 128 threads (one warpgroup).
//   out layout: out[cand][M=key*64 + N=n],  cand 0 = FA3,  cand 1 = V8.
// ═════════════════════════════════════════════════════════════════════════════
__global__ void __launch_bounds__(128,1) probe_kernel(float* __restrict__ out) {
    const int tid = threadIdx.x;

    __shared__ __align__(128) bf16 sP_sw [4096];  // FA3: P swizzled, KEY(c)-contiguous
    __shared__ __align__(128) bf16 sP_swK[4096];  // V8 : P swizzled, QUERY(r)-contiguous (transposed staging)
    __shared__ __align__(128) bf16 B_sw  [4096];  // B(dO-like) swizzled, N(n)-contiguous

    // ── Fill the ramps into their swizzled layouts ─────────────────────────
    for (int i = tid; i < 4096; i += 128) {
        int a = i >> 6, b = i & 63;              // a = strided index, b = contiguous index
        // FA3 sP_sw: P with key(c) CONTIGUOUS, query(r) strided → pos sw128_idx(row=query,col=key),
        //   value P[query][key]=rampP(query,key).  (a=query, b=key)
        sP_sw [sw128_idx(a, b)] = __float2bfloat16(rampP(a, b));
        // V8 sP_swK: A=Pᵀ read Major::K ⇒ contraction query(r) CONTIGUOUS, output key(c) strided →
        //   pos sw128_idx(row=key,col=query), value A[key][query]=P[query][key]=rampP(query,key).
        //   (a=key, b=query) ⇒ value rampP(b, a).
        sP_swK[sw128_idx(a, b)] = __float2bfloat16(rampP(b, a));
        // B_sw: B(dO-like) with n CONTIGUOUS, query(r) strided → pos sw128_idx(row=query,col=n),
        //   value B[query][n]=rampB(query,n).  (a=query, b=n)
        B_sw  [sw128_idx(a, b)] = __float2bfloat16(rampB(a, b));
    }
    __syncthreads();
    fence_proxy_async_shared();   // generic fills → async wgmma read (CTA scope)

    // ── Candidate FA3 (V14): A=Pᵀ Major::MN swizzled (trans-a=1), B Major::MN (trans-b=1) ──
    {
        float acc[32]; zeroN<32>(acc);
        fence_operandN<32>(acc); wgmma_fence();
#pragma unroll
        for (int k = 0; k < 4; k++) {                 // K = r = 64 → 4 k-steps of 16
            uint64_t dA = make_desc_sw128_MN(sP_sw + k*1024);  // single atom, +1024 elem/step (== proven dO read)
            uint64_t dB = make_desc_sw128_MN(B_sw  + k*1024);
            wgmma_m64n64k16_tAtB(acc, dA, dB);
        }
        wgmma_commit(); wgmma_wait0(); fence_operandN<32>(acc);
        store_acc_out(acc, out + 0*4096, tid);
    }

    // ── Candidate V8 (failed): A=Pᵀ Major::K swizzled (trans-a=0), B Major::MN (trans-b=1) ──
    {
        float acc[32]; zeroN<32>(acc);
        fence_operandN<32>(acc); wgmma_fence();
#pragma unroll
        for (int k = 0; k < 4; k++) {                 // K = r = 64 → 4 k-steps of 16
            uint64_t dA = make_desc_sw128_K (sP_swK + k*16);   // Major::K, contraction(r) contiguous, +16 elem/step
            uint64_t dB = make_desc_sw128_MN(B_sw   + k*1024);
            wgmma_m64n64k16_tB(acc, dA, dB);
        }
        wgmma_commit(); wgmma_wait0(); fence_operandN<32>(acc);
        store_acc_out(acc, out + 1*4096, tid);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Host driver + CPU reference.
//   ref[c][n] = Σ_r P[r][c]·B[r][n]  = (Pᵀ·B)[c][n].   out[cand] compared to ref.
// ═════════════════════════════════════════════════════════════════════════════
static void check(const char* name, const float* got, const float* ref) {
    double maxd = 0.0; int nbad = 0, badc = -1, badn = -1;
    for (int c = 0; c < 64; c++)
        for (int n = 0; n < 64; n++) {
            double dd = fabs((double)got[c*64+n] - (double)ref[c*64+n]);
            if (dd > maxd) { maxd = dd; badc = c; badn = n; }
            if (dd > 0.5) nbad++;
        }
    bool pass = (maxd <= 0.5);   // exact-integer arithmetic → any real mismatch >> 0.5
    printf("  %-14s : %s   max|Δ| = %.6g   (%d / 4096 elems wrong)\n",
           name, pass ? "PASS ✅" : "FAIL ❌", maxd, nbad);
    if (!pass) {
        printf("      worst @ (key=%d, n=%d): got %.3f  expected %.3f\n",
               badc, badn, got[badc*64+badn], ref[badc*64+badn]);
        // a few samples to eyeball the failure structure
        for (int s = 0; s < 4; s++) {
            int c = s*11 % 64, n = s*23 % 64;
            printf("      sample (key=%2d,n=%2d): got %8.3f  exp %8.3f\n",
                   c, n, got[c*64+n], ref[c*64+n]);
        }
    }
}

int main() {
    // Device sanity — the probe is MEANINGLESS on non-Hopper (wgmma won't execute).
    int dev = 0; cudaDeviceProp p;
    CUDA_OK(cudaGetDevice(&dev)); CUDA_OK(cudaGetDeviceProperties(&p, dev));
    printf("Device: %s  (sm_%d%d)\n", p.name, p.major, p.minor);
    if (p.major != 9) {
        printf("!! WARNING: wgmma requires sm_90 (Hopper).  Results on this GPU are meaningless.\n"
               "!! This binary is COMPILE-VERIFY only unless run on an H100/H200.\n");
    }

    // CPU reference.
    float ref[64*64];
    for (int c = 0; c < 64; c++)
        for (int n = 0; n < 64; n++) {
            float acc = 0.f;
            for (int r = 0; r < 64; r++) acc += rampP(r,c) * rampB(r,n);
            ref[c*64+n] = acc;
        }

    float *d_out; CUDA_OK(cudaMalloc(&d_out, 2*4096*sizeof(float)));
    CUDA_OK(cudaMemset(d_out, 0, 2*4096*sizeof(float)));
    probe_kernel<<<1,128>>>(d_out);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());

    float h_out[2*4096];
    CUDA_OK(cudaMemcpy(h_out, d_out, 2*4096*sizeof(float), cudaMemcpyDeviceToHost));

    printf("\nwgmma transposed-A read probe  (dV = Pᵀ·B, m64n64k16 bf16)\n");
    printf("  reference = (Pᵀ·B)[key][n], exact integer ramps\n\n");
    check("FA3 (V14)", h_out + 0*4096, ref);
    check("V8 (failed)", h_out + 1*4096, ref);
    printf("\nInterpretation:\n"
           "  FA3 PASS  → transpose-via-descriptor is VIABLE; adopt in V14 (drop sA_t staging for dV/dK).\n"
           "  FA3 FAIL  → NOT viable for this buffer; keep staging, reduce its cost another way.\n"
           "  V8 FAIL   → confirms the swizzled-Major::K-A + Major::MN-B combo is the V8 root cause.\n"
           "  (V8 PASS would be a surprise — re-examine the V8 post-mortem.)\n");

    CUDA_OK(cudaFree(d_out));
    return 0;
}
