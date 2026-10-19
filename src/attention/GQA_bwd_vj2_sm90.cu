// ============================================================================
//  GQA_bwd_vj2_sm90.cu -- FA3-structured GQA causal backward for Hopper (sm_90a), head_dim 128.
//  Added 2026-10-12. Since 2026-10-15 the DEFAULT Hopper backward (USE_VJ2=0 selects Vj1d, GQA_bwd_vj1d_sm90.cu,
//  which is not changed by this file).
//
//  WHY: measured on H200 (dense B=8, 18/6 heads, T=4096, ncu, 2026-10-12), Vj1d takes 3.63 ms against FA3's
//  2.49 ms with the SAME tensor-core busy time (1.66 vs 1.74 ms): the gap is time with idle tensor cores.
//  Vj1d's 64-key blocks reload Q/dO and add dQ back twice as often per key (L2 read requests 81.0M vs 40.9M),
//  one warpgroup does the loads, the math and the dQ add-back behind __syncthreads (~14% of stall samples at
//  the fences before the add-back, ~10% waiting on GEMMs), and P makes a round trip through shared memory
//  (~7%). Records: profiling_reports/v2_bwd_rework_research_2026_10_12/RESEARCH.md and DESIGN_VJ2.md.
//
//  WHAT (FA3's head-dim-128 causal layout, hopper/flash_bwd_launch_template.h:351):
//    * 128 keys per block, 64 queries per step, 384 threads = 3 warpgroups:
//        WG0 loader: warp 0 issues the TMA loads (K/V once; Q, dO, LSE, D two steps ahead),
//                    warp 1 is the dQ writer (1-D bulk reduce-add of the compute groups' fp32 dQ);
//        WG1/WG2 compute: group c owns keys k0+64c .. k0+64c+63.
//      setmaxnreg 24 / 240, as FA3 (flash_bwd_kernel_sm90.h:64-65).
//    * The score GEMMs are "swapped" (S^T = K.Q^T, dP^T = V.dO^T, FA3 SdP_swapAB), so P^T and dS^T sit in
//      registers in the layout the dV and dK GEMMs read directly (register-A wgmma, FA3 Mma_dKV_is_RS).
//      Only dS goes to shared memory, for the dQ GEMM.
//    * dS is computed from the fp32 P (FA3 mainloop_bwd_sm90_tma_gmma_ws.hpp:868,893). Vj1d computes it from
//      the bf16 copy of P re-read from shared memory (GQA_bwd_vj1d_sm90.cu:338-347). CPU emulation
//      (RESEARCH.md section 4): RMS error of dQ/dK 17-30% lower on normal inputs.
//    * The compute groups never stop the whole block: they hand dQ to the writer warp with named barriers
//      (FA3 :953-957 / :645-654) instead of __syncthreads + thread 0 issuing the reduce.
//  KEPT from Vj1d (proven there): the TMA maps and 128-B swizzle, the one-writer bf16 dK/dV partial per query
//  head + the fixed-order head sum (dK/dV are bit-identical run to run). The kernels are copied into this file
//  (each .cu is its own translation unit, no device linking).
//  dQ (changed 2026-10-13, FA3 mainloop_bwd_sm90_tma_gmma_ws.hpp:195-198, 645-654, 951-957): each thread stores
//  its fp32 dQ fragment to a flat staging buffer with 128-bit stores in thread order, and the writer adds the
//  16 KB buffer to global with ONE 1-D bulk reduce-add. The global fp32 accumulator therefore holds each
//  64 x 128 tile in that fragment order, and convert_dq_frag_vj2 maps it back to row-major bf16. The first
//  version staged Vj1d's SW128 layout for a 2-D tensor reduce: 16 STS.64 per thread per step with 2-way bank
//  conflicts (ncu 2026-10-13: 56.3M store bank conflicts vs FA3's 1.8M; the fence after them 4.2% + the
//  barrier arrive 3.1% of stall samples).
//  SCOPE: causal; dense rows, and packed rows given as a per-chat list (v2 design A, the cu_seqlens way of
//  FlashAttention varlen; added 2026-10-15, replacing the per-token seg_start version written earlier that day).
//  See the kernel comment for how a chat's work is cut.
//
//  Device code is guarded by __CUDA_ARCH__ >= 900 && < 1000 so the file still compiles for other targets.
// ============================================================================
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#include "device/CachingCudaAllocator.h"

namespace OwnTensor {
namespace cuda {
namespace gqa_bwd_vj2 {

using bf16 = __nv_bfloat16;

#define GQA_BWD_VJ2_CUDA_CHECK(x) do {                                        \
    cudaError_t _e = (x);                                                     \
    if (_e != cudaSuccess) {                                                  \
        fprintf(stderr, "GQA_bwd_vj2_sm90 %s:%d: %s\n", __FILE__, __LINE__,  \
                cudaGetErrorString(_e));                                      \
        abort();                                                              \
    }                                                                         \
} while (0)

constexpr int D        = 128;           // head dim
constexpr int BM       = 64;            // queries per step
constexpr int BN       = 128;           // keys per block (64 per compute group)
constexpr int WGT      = 128;           // threads per warpgroup
constexpr int NTHREADS = 3 * WGT;       // loader + 2 compute warpgroups
constexpr int KV_HALF  = BN * 64;       // one SW128 head-half block of K or V: [128 keys][64]  (elements)
constexpr int KV_TILE  = 2 * KV_HALF;
constexpr int Q_HALF   = BM * 64;       // one SW128 head-half block of Q or dO: [64 queries][64]
constexpr int Q_TILE   = 2 * Q_HALF;
constexpr int DS_TILE  = BN * BM;       // dS^T: [128 keys][64 queries], one SW128 block
constexpr int DQ_WG    = BM * 64;       // fp32 dQ staging of one compute group: 64 queries x 64 head cols,
                                        // in fragment order (see store_acc_frag_f32)
constexpr float LOG2E  = 1.4426950408889634f;

// Byte offsets inside the dynamic shared memory (base rounded up to 1024 B in the kernel).
constexpr uint32_t OFF_K   = 0;
constexpr uint32_t OFF_V   = OFF_K   + KV_TILE * 2;          //  32768
constexpr uint32_t OFF_Q   = OFF_V   + KV_TILE * 2;          //  65536  (2 stages)
constexpr uint32_t OFF_DO  = OFF_Q   + 2 * Q_TILE * 2;       //  98304  (2 stages)
constexpr uint32_t OFF_DS  = OFF_DO  + 2 * Q_TILE * 2;       // 131072  (2 stages)
constexpr uint32_t OFF_DQ  = OFF_DS  + 2 * DS_TILE * 2;      // 163840  (one per compute group)
constexpr uint32_t OFF_LSE = OFF_DQ  + 2 * DQ_WG * 4;        // 196608  (2 stages x 64 fp32, LSE*log2e)
constexpr uint32_t OFF_D   = OFF_LSE + 2 * BM * 4;           // 197120  (2 stages x 64 fp32)
constexpr uint32_t OFF_BAR = OFF_D   + 2 * BM * 4;           // 197632  (kv_full, q_full[2], q_empty[2])
constexpr uint32_t SMEM_USED  = OFF_BAR + 8 * 8;
constexpr uint32_t SMEM_BYTES = SMEM_USED + 1024;            // + room to round the base up to 1024 B
static_assert(SMEM_BYTES <= 232448, "over the 227 KB a Hopper block may use");
static_assert(OFF_V % 1024 == 0 && OFF_Q % 1024 == 0 && OFF_DO % 1024 == 0 && OFF_DS % 1024 == 0 &&
              OFF_DQ % 1024 == 0, "128-B swizzled operands need 1024-B aligned bases");
static_assert(OFF_LSE % 16 == 0 && OFF_D % 16 == 0 && OFF_BAR % 8 == 0, "bulk copy / mbarrier alignment");
constexpr uint32_t QSTEP_TX = 2 * Q_TILE * 2 + 2 * BM * 4;   // Q + dO tiles + LSE + D rows = 33280 B
constexpr uint32_t KV_TX    = 2 * KV_TILE * 2;               // K + V tiles = 65536 B

// Named barriers (id 0 is __syncthreads, used once before the warpgroups split).
constexpr int BAR_PDS      = 1;   // both compute groups (256): dS written / epilogue join
constexpr int BAR_DQ_FULL  = 2;   // + c: compute group c (128) arrives, the writer warp (32) syncs
constexpr int BAR_DQ_EMPTY = 4;   // + c: the writer warp (32) arrives, compute group c (128) syncs
constexpr int BAR_EPI      = 6;   // + c: one compute group (128)

#if __CUDA_ARCH__ >= 900 && __CUDA_ARCH__ < 1000   // Hopper only

// ---- synchronisation helpers (copies of the GQA_fwd_sm90.cu / GQA_bwd_vj1d_sm90.cu primitives) ----------
__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(a), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, int phase) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile(
        "{\n\t.reg .pred P;\n"
        "LW_%=:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P, [%0], %1;\n\t"
        "@P bra.uni LD_%=;\n\t"
        "bra.uni LW_%=;\n"
        "LD_%=:\n\t}" :: "r"(a), "r"(phase) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(a) : "memory");
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(a), "r"(bytes) : "memory");
}
__device__ __forceinline__ void tma_load_2d(void* smem_dst, const void* tmap, int cx, int cy, uint64_t* bar) {
    uint32_t dst = (uint32_t)__cvta_generic_to_shared(smem_dst);
    uint32_t b   = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"((uint64_t)tmap), "r"(cx), "r"(cy), "r"(b) : "memory");
}
// 1-D bulk copy global -> shared on an mbarrier (FA3 loads LSE and dPsum this way, mainloop_bwd...:514-515).
// Needs 16-B aligned addresses and a size multiple of 16 (256 B here).
__device__ __forceinline__ void bulk_load_1d(void* smem_dst, const void* gsrc, uint32_t bytes, uint64_t* bar) {
    uint32_t dst = (uint32_t)__cvta_generic_to_shared(smem_dst);
    uint32_t b   = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
        :: "r"(dst), "l"((uint64_t)gsrc), "r"(bytes), "r"(b) : "memory");
}
__device__ __forceinline__ void tma_store_2d(const void* tmap, const void* smem_src, int cx, int cy) {
    uint32_t src = (uint32_t)__cvta_generic_to_shared(smem_src);
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
                 :: "l"((uint64_t)tmap), "r"(cx), "r"(cy), "r"(src) : "memory");
}
// 1-D bulk reduce-add shared -> global, fp32 (FA3's SM90_BULK_REDUCE_ADD, mainloop_bwd...:647). 16-B aligned
// addresses, size a multiple of 16.
__device__ __forceinline__ void bulk_reduce_add_1d(float* gdst, const float* smem_src, uint32_t bytes) {
    uint32_t src = (uint32_t)__cvta_generic_to_shared(smem_src);
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], %2;"
                 :: "l"(gdst), "r"(src), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_commit()     { asm volatile("cp.async.bulk.commit_group;"       ::: "memory"); }
__device__ __forceinline__ void bulk_wait_all()   { asm volatile("cp.async.bulk.wait_group 0;"       ::: "memory"); }
__device__ __forceinline__ void bulk_wait_read()  { asm volatile("cp.async.bulk.wait_group.read 0;"  ::: "memory"); }
__device__ __forceinline__ void fence_proxy_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ void bar_sync(int id, int n)   { asm volatile("bar.sync %0, %1;"   :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void bar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void reg_dec_loader()  { asm volatile("setmaxnreg.dec.sync.aligned.u32 24;"  ::: "memory"); }
__device__ __forceinline__ void reg_inc_compute() { asm volatile("setmaxnreg.inc.sync.aligned.u32 240;" ::: "memory"); }

// ---- wgmma helpers --------------------------------------------------------------------------------------
__device__ __forceinline__ void wgmma_fence()  { asm volatile("wgmma.fence.sync.aligned;"        ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wgmma_wait0()  { asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory"); }
__device__ __forceinline__ void wgmma_wait1()  { asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory"); }
template<int N>
__device__ __forceinline__ void fence_f(float* d) {
#pragma unroll
    for (int i = 0; i < N; i++) asm volatile("" : "+f"(d[i]) :: "memory");
}
template<int N>
__device__ __forceinline__ void fence_u(uint32_t* d) {
#pragma unroll
    for (int i = 0; i < N; i++) asm volatile("" : "+r"(d[i]) :: "memory");
}
template<int N>
__device__ __forceinline__ void zero_f(float* d) {
#pragma unroll
    for (int i = 0; i < N; i++) d[i] = 0.0f;
}

// 128-B swizzle smem descriptor: LBO 16 B, SBO 1024 B, layout SW128. Same bits as make_desc_sw128_K in
// GQA_bwd_vj1d_sm90.cu:126-134 and make_swz_desc in GQA_fwd_sm90.cu:257-265.
__device__ __forceinline__ uint64_t desc_sw128(const void* smem_ptr) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint64_t d = 0;
    d |= (uint64_t)((addr >> 4) & 0x3FFFu);
    d |= (uint64_t)1u << 16;
    d |= (uint64_t)(1024 >> 4) << 32;
    d |= (uint64_t)1u << 62;
    return d;
}

// Sibling descriptor at a bf16 ELEMENT offset E (a multiple of 8): the address field is (byte address >> 4), so the
// sibling is desc + E/8, added in the low 32 bits only (no carry can leave the 14-bit field: shared memory < 227 KB).
// Copy of desc_add_lo, GQA_fwd_sm90.cu:253-256. Every GEMM below builds its descriptors this way from bases made
// once per block: the first version rebuilt each one from a pointer (cvta + mask + shifts) for every wgmma, 5.4 extra
// instructions per HGMMA, 23% of all instructions (ncu 2026-10-12).
__device__ __forceinline__ uint64_t desc_add(uint64_t d, int elem_off) {
    const uint32_t lo = (uint32_t)d + (uint32_t)(elem_off >> 3);
    return ((uint64_t)(uint32_t)(d >> 32) << 32) | (uint64_t)lo;
}

#define VJ2_ACC32 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15," \
                  "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}"
#define VJ2_ACC32_OPS(d) \
    "+f"(d[0]),  "+f"(d[1]),  "+f"(d[2]),  "+f"(d[3]),  "+f"(d[4]),  "+f"(d[5]),  "+f"(d[6]),  "+f"(d[7]),  \
    "+f"(d[8]),  "+f"(d[9]),  "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), \
    "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), \
    "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])

// m64n64k16, A and B from shared memory, both K-major (no transpose) -- Vj1d wgmma_m64n64k16 (:49-64).
__device__ __forceinline__ void wgmma_ss(float d[32], uint64_t descA, uint64_t descB) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 " VJ2_ACC32 ", %32, %33, 1, 1, 1, 0, 0;"
                 : VJ2_ACC32_OPS(d) : "l"(descA), "l"(descB));
}
// m64n64k16, A and B from shared memory, both MN-major (transposed) -- Vj1d wgmma_m64n64k16_tAtB (:168-183).
__device__ __forceinline__ void wgmma_ss_tAtB(float d[32], uint64_t descA, uint64_t descB) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 " VJ2_ACC32 ", %32, %33, 1, 1, 1, 1, 1;"
                 : VJ2_ACC32_OPS(d) : "l"(descA), "l"(descB));
}
#define VJ2_ACC64 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15," \
                  "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31," \
                  "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47," \
                  "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}"
// 4 x m64n128k16 with A from REGISTERS (bf16 fragments a[4k..4k+3] for k-step k) and B from shared memory,
// MN-major (trans-b = 1). One asm block for the 4 k-steps so ptxas cannot put a sync between them.
// Full-width since 2026-10-13 (G1): FA3/FA4/cuDNN all run their register-A backward GEMMs as m64n128k16 (8 per step);
// the first version issued 2 x m64n64k16 per k-step (one per head half, 16 per step; ncu 2026-10-13 SASS).
// The m64n128 accumulator holds n8 block j in d[4j..4j+3]: blocks 0-7 = head cols 0-63 = the old first half's
// d[0..31], blocks 8-15 = cols 64-127 = the old second half's d[32..63], so the epilogue is unchanged.
__device__ __forceinline__ void wgmma_rs128_x4(float d[64], const uint32_t a[16], const uint64_t b[4]) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 " VJ2_ACC64 ", {%64,%65,%66,%67}, %80, 1, 1, 1, 1;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 " VJ2_ACC64 ", {%68,%69,%70,%71}, %81, 1, 1, 1, 1;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 " VJ2_ACC64 ", {%72,%73,%74,%75}, %82, 1, 1, 1, 1;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 " VJ2_ACC64 ", {%76,%77,%78,%79}, %83, 1, 1, 1, 1;\n"
        : VJ2_ACC32_OPS(d), VJ2_ACC32_OPS((d + 32))
        : "r"(a[0]),  "r"(a[1]),  "r"(a[2]),  "r"(a[3]),  "r"(a[4]),  "r"(a[5]),  "r"(a[6]),  "r"(a[7]),
          "r"(a[8]),  "r"(a[9]),  "r"(a[10]), "r"(a[11]), "r"(a[12]), "r"(a[13]), "r"(a[14]), "r"(a[15]),
          "l"(b[0]), "l"(b[1]), "l"(b[2]), "l"(b[3]));
}
// The same descriptor with its leading byte offset (bits 16-29, 16-B units) set. For an MN-major SW128 operand
// the LBO is the distance between the 64-element column blocks of N (CUTLASS make_gmma_desc<Major::MN>, B128:
// ((T,8,m),(8,k)):((1,T,LBO),(8T,SBO))); N = 128 spans the tile's 2 head-half blocks, Q_HALF elements apart.
__device__ __forceinline__ uint64_t desc_lbo(uint64_t d, uint32_t lbo_bytes) {
    return (d & ~(0x3FFFull << 16)) | ((uint64_t)((lbo_bytes >> 4) & 0x3FFFu) << 16);
}

__device__ __forceinline__ float ex2_approx(float x) {
    float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y;
}
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return reinterpret_cast<uint32_t&>(v);
}
__device__ __forceinline__ void stmatrix_x4(uint32_t dst, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};"
                 :: "r"(dst), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
}

// ---- GEMMs of one step --------------------------------------------------------------------------------------
// S^T (or dP^T) of this group: rows = its 64 keys, cols = the step's 64 queries, k = 128 head dims.
// A = K (or V) rows 64c.. of a [128][64]x2 tile, K-major; B = Q (or dO) [64][64]x2, K-major.
// Vj1d's S GEMM (run_gemm_n64_sw2_hoB_issue, :189-200) with A and B swapped (FA3 SdP_swapAB).
__device__ __forceinline__ void gemm_kq_issue(float acc[32], uint64_t dA_rows, uint64_t dB_tile) {
    fence_f<32>(acc);
    wgmma_fence();
#pragma unroll
    for (int k = 0; k < 8; k++)
        wgmma_ss(acc, desc_add(dA_rows, (k >> 2) * KV_HALF + (k & 3) * 16),
                      desc_add(dB_tile, (k >> 2) * Q_HALF  + (k & 3) * 16));
    wgmma_commit();
}
// dV (or dK) of this group += P^T (or dS^T) . dO (or Q): 64 keys x 128 head cols, k = 64 queries.
// A = registers; B = [64 queries][64]x2 tile, MN-major: 16 queries per k-step = 1024 elements; one m64n128k16 per
// k-step reads both head halves (LBO = Q_HALF * 2 bytes).
__device__ __forceinline__ void gemm_rs_issue(float acc[64], const uint32_t a[16], uint64_t dB_tile) {
    fence_f<64>(acc);
    const uint64_t dB = desc_lbo(dB_tile, Q_HALF * 2);
    uint64_t b[4];
#pragma unroll
    for (int kc = 0; kc < 4; kc++) b[kc] = desc_add(dB, kc * 1024);
    wgmma_rs128_x4(acc, a, b);
}
// dQ of this group (64 queries x head cols 64c..64c+63) = dS (64 queries x 128 keys) . K (128 keys x 64).
// A = dS^T stored [128 keys][64 queries] -> the queries (M) are contiguous: MN-major, trans-a. 16 keys per k-step
// = 16 rows = 1024 elements. B = K head-half c [128 keys][64], MN-major, trans-b, same stepping.
// The same descriptor forms as Vj1d's dK GEMM (run_gemm_dKdQ_te_issue_ho, :361-376).
__device__ __forceinline__ void gemm_dq_issue(float acc[32], uint64_t dS_tile, uint64_t dK_half) {
    fence_f<32>(acc);
    wgmma_fence();
#pragma unroll
    for (int k = 0; k < 8; k++)
        wgmma_ss_tAtB(acc, desc_add(dS_tile, k * 1024), desc_add(dK_half, k * 1024));
    wgmma_commit();
}

// fp32 accumulator (64 rows x 64 cols, wgmma C layout) -> flat staging in FRAGMENT order: value block j
// (d[4j..4j+3] = n8 block j: rows r0, r0, r0+8, r0+8; cols 8j+2t, +1, 8j+2t, +1) of thread tid goes to
// base[512 j + 4 tid]. One 128-bit store per block; a warp writes 512 contiguous bytes, no bank conflicts.
// FA3's R2STiledCopydQaccum (mainloop_bwd...:195-198): 128-bit atoms over the threads of a warpgroup.
// convert_dq_frag_vj2 holds the inverse map. The values are stored unscaled: the softmax scale is applied once
// in convert_dq_frag_vj2, as FA3 does (flash_bwd_postprocess_kernel.h:213). G4 2026-10-14: the per-step multiply
// was 32 FMUL per thread per step, 39M of our 80M FMUL (ncu, B8 18/6).
__device__ __forceinline__ void store_acc_frag_f32(const float* d, float* base, int tid) {
#pragma unroll
    for (int j = 0; j < 8; j++)
        *reinterpret_cast<float4*>(base + 512 * j + 4 * tid) =
            make_float4(d[4 * j], d[4 * j + 1], d[4 * j + 2], d[4 * j + 3]);
}
// fp32 accumulator (64 x 64) -> bf16 [64][64] in the 128-B swizzle (16-B chunk k of row r stored at chunk
// k ^ (r & 7)) for the SWIZZLE_128B TMA store (box {64, 64}); the base must be 1024-B aligned. The 8 rows a warp
// writes then fall in 8 different chunks: 32 lanes, 32 banks. The first version (Vj1d stage_acc_bf16_s<64,64>,
// :202-214) wrote plain 128-B rows for an unswizzled store: 8-way bank conflicts (ncu 2026-10-13, 2.2% of stall
// samples). Same bf16 rounding (round to nearest) as before.
__device__ __forceinline__ void stage_acc_bf16_sw128(const float* d, bf16* stage, int tid, float scl) {
    const int w = tid >> 5, lane = tid & 31;
    const int r0 = w * 16 + (lane >> 2), r1 = r0 + 8, cc = (lane & 3) * 2;
    const int ph = r0 & 7;                        // == r1 & 7
#pragma unroll
    for (int nt = 0; nt < 8; nt++) {
        const int off = ((nt ^ ph) << 3) + cc;    // cols 8nt + cc, +1 of the row, in the swizzled chunk
        *reinterpret_cast<__nv_bfloat162*>(&stage[r0 * 64 + off]) =
            __floats2bfloat162_rn(d[nt * 4 + 0] * scl, d[nt * 4 + 1] * scl);
        *reinterpret_cast<__nv_bfloat162*>(&stage[r1 * 64 + off]) =
            __floats2bfloat162_rn(d[nt * 4 + 2] * scl, d[nt * 4 + 3] * scl);
    }
}

// D = rowsum(dO * O) per query row (copy of Vj1d compute_drowsum_v22, :296-320) and, new, LSE * log2e, so the
// compute groups use ex2(S * scale * log2e - lse2) with no extra multiply (FA3 stores LSE in log2 units too).
// Both outputs are read by 256-B bulk copies, one per 64-row step.
// It also zeroes the dQ accumulator (B-1, 2026-10-14; FA3 clears dQaccum in its preprocess kernel the same way):
// the accumulator has 128 floats per query row, so each warp clears its row's 128 floats with one 16-B store per
// lane. This replaces a separate cudaMemset of the whole accumulator (nsys, B8 18/6: 67.9 us).
__global__ void bwd_prep_vj2(const bf16* __restrict__ d_dO, const bf16* __restrict__ d_O,
                             const float* __restrict__ d_LSE, float* __restrict__ d_Drow,
                             float* __restrict__ d_Lse2, float* __restrict__ d_dqa, long nRows) {
    const int  warpsPerBlock = blockDim.x >> 5;
    const int  lane          = threadIdx.x & 31;
    const long warp0         = (long)blockIdx.x * warpsPerBlock + (threadIdx.x >> 5);
    const long stride        = (long)gridDim.x * warpsPerBlock;
    for (long row = warp0; row < nRows; row += stride) {
        const long base = row * 128 + lane * 4;
        *reinterpret_cast<float4*>(d_dqa + base) = make_float4(0.f, 0.f, 0.f, 0.f);
        const uint2 vdO = *reinterpret_cast<const uint2*>(d_dO + base);
        const uint2 vO  = *reinterpret_cast<const uint2*>(d_O  + base);
        const bf16* a = reinterpret_cast<const bf16*>(&vdO);
        const bf16* b = reinterpret_cast<const bf16*>(&vO);
        float partial = 0.f;
#pragma unroll
        for (int k = 0; k < 4; k++) partial += __bfloat162float(a[k]) * __bfloat162float(b[k]);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            partial += __shfl_down_sync(0xFFFFFFFFu, partial, off);
        if (lane == 0) { d_Drow[row] = partial; d_Lse2[row] = d_LSE[row] * LOG2E; }
    }
}

// Design A (per-chat lists): the same per-row work, written to the plan's PADDED layout. A chat can start at any token,
// but the 256-B LSE/D bulk loads and the fragment-order dQ tiles need 64-row alignment, so every chat gets its own
// 64-row-aligned slot per query head -- FA3's varlen backward does the same (seqlen.h offset_padded: each sequence of
// dQaccum / LSE_log2 / dPsum starts on a kBlockM boundary). Each chat owns ceil(L/64) tiles per head: padded row
// p = (hq * nT + T) * 64 + r is token row0 + r of row b when r < nrows (dq_tiles[T] = {b, row0, nrows, 0}), else
// padding. Padding gets LSE2 = +inf and D = 0, so P = ex2(S * scale2 - inf) = 0 and dS = P * (dP - D) = 0 there:
// the rows past a chat's end that its last step reads add exact zeros (FlashAttention 2 sets LSE = +inf past the
// sequence end the same way, csrc/flash_attn/src/flash_bwd_kernel.h:411-413). The fp32 dQ accumulator uses the same
// padded rows (128 floats per row, zeroed here). D is computed exactly as above, so a chat's D is bit-identical to the
// dense run of the same chat.
__global__ void bwd_prep_vj2_chats(const bf16* __restrict__ d_dO, const bf16* __restrict__ d_O,
                                   const float* __restrict__ d_LSE, float* __restrict__ d_Drow,
                                   float* __restrict__ d_Lse2, float* __restrict__ d_dqa,
                                   const int4* __restrict__ dq_tiles, int nT, int Hq, int S, long nRowsP) {
    const int  warpsPerBlock = blockDim.x >> 5;
    const int  lane          = threadIdx.x & 31;
    const long warp0         = (long)blockIdx.x * warpsPerBlock + (threadIdx.x >> 5);
    const long stride        = (long)gridDim.x * warpsPerBlock;
    for (long row = warp0; row < nRowsP; row += stride) {
        *reinterpret_cast<float4*>(d_dqa + row * 128 + lane * 4) = make_float4(0.f, 0.f, 0.f, 0.f);
        const long tp = row >> 6;                         // padded tile over all heads: hq * nT + T
        const int  r  = (int)(row & 63);
        const int  hq = (int)(tp / nT);
        const int4 dt = dq_tiles[tp - (long)hq * nT];     // {b, row0, nrows, 0}
        if (r < dt.z) {                                    // one row per warp: the same in every lane
            const long src  = ((long)dt.x * Hq + hq) * S + dt.y + r;
            const long base = src * 128 + lane * 4;
            const uint2 vdO = *reinterpret_cast<const uint2*>(d_dO + base);
            const uint2 vO  = *reinterpret_cast<const uint2*>(d_O  + base);
            const bf16* a = reinterpret_cast<const bf16*>(&vdO);
            const bf16* b = reinterpret_cast<const bf16*>(&vO);
            float partial = 0.f;
#pragma unroll
            for (int k = 0; k < 4; k++) partial += __bfloat162float(a[k]) * __bfloat162float(b[k]);
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                partial += __shfl_down_sync(0xFFFFFFFFu, partial, off);
            if (lane == 0) { d_Drow[row] = partial; d_Lse2[row] = d_LSE[src] * LOG2E; }
        } else if (lane == 0) {
            d_Drow[row] = 0.f; d_Lse2[row] = INFINITY;
        }
    }
}

// ---- the kernel ---------------------------------------------------------------------------------------------
// One block per (128-key block kt, query head hq, batch b). Key block kt is seen by the query steps
// q = 2kt .. S/64-1 (causal); the first two steps hold the diagonal and are the only masked ones.
// Block order (added 2026-10-13, FA3 SingleTileBwdLPTScheduler, tile_scheduler.hpp:368-469): a 1-D grid in
// sections of `sw` (batch, head) pairs; inside a section key block 0 (the most steps) of every pair runs first,
// then key block 1, ... The last blocks to start are the shortest, so the SMs finish together. The first
// version ran kt fastest (grid kt x Hq x B): the heaviest blocks of the last heads started in the last wave
// (ncu 2026-10-13: SMs busy 96.9% of the elapsed cycles, FA3 98.6%).
// CHATS (v2 design A, the cu_seqlens way; 2026-10-15): one block per (128-key tile of a chat, query head), read from
// the plan: blocks[2i] = {b, t0, L, kt}, blocks[2i+1] = {hq, dqt0, 0, 0} = key tile kt of segment [t0, t0+L) of row b
// (a chat, or a row's pad tail as its own segment). Every index is segment-relative, so the steps, the diagonal mask
// and the trip count are the dense ones with S = L (Vj1d's MODE_CHATS does the same with 64-key tiles). The block
// order is made by the plan builder: FA3's LPT order inside sections of `sw` (batch, head) pairs, as the dense grid.
// What differs from dense:
//   * Q/dO/K/V rows are offset by t0. A chat's last step can hold rows past L (the next segment's rows, or rows past
//     the tensor end that TMA fills with zeros): their LSE/D come from the padded slot as +inf / 0, so P = 0, dS = 0.
//   * LSE, D and the fp32 dQ accumulator are in the plan's PADDED layout (bwd_prep_vj2_chats): dqt0 = the slot tile
//     of this block's first step; step i reads and adds tile dqt0 + i.
//   * keys past L sit only in the chat's last key tile. Causal hides them: every query of the chat is before them,
//     and every step that holds such keys is a diagonal step. Their dK/dV rows belong to the next segment, so a
//     compute group whose 64 keys cross L stores only its rows < L, with plain stores, and a group entirely past L
//     stores nothing (FA3 stores all varlen dK/dV with plain stores, hopper/epilogue_bwd.hpp:31,263-268).
// CHATS = false is the dense kernel, unchanged (SASS checked identical to the first Vj2). The three A-only parameters
// are the LAST ones so the older ones keep their offsets.
template<bool CHATS>
__global__ void __launch_bounds__(NTHREADS, 1)
gqa_bwd_vj2_kernel(
    const __grid_constant__ CUtensorMap tma_K,  const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_Q,  const __grid_constant__ CUtensorMap tma_dO,
    const __grid_constant__ CUtensorMap tma_dV_st, const __grid_constant__ CUtensorMap tma_dK_st,
    float* __restrict__ dQacc,                          // fp32, 64 x 128 tiles in fragment order
    const float* __restrict__ Lse2, const float* __restrict__ Drow,
    int B, int Hq, int Hkv, int G, int S, float scale, int sw,
    const int4* __restrict__ blocks, bf16* __restrict__ dVp, bf16* __restrict__ dKp) {   // A only
    int kt, hq, b;
    int t0 = 0, L = 0, dqt0 = 0;                        // A only (dense: 0)
    if constexpr (CHATS) {
        const int4 w0 = blocks[2 * blockIdx.x], w1 = blocks[2 * blockIdx.x + 1];
        b = w0.x; t0 = w0.y; L = w0.z; kt = w0.w; hq = w1.x; dqt0 = w1.y;
    } else {
        const int nkt = S / BN, HB = B * Hq;
        const int sec = blockIdx.x / (sw * nkt), in = blockIdx.x % (sw * nkt);
        const int n = (sec < HB / sw) ? sw : HB % sw;       // pairs in this section (the last one may be short)
        kt = in / n;
        const int hb = sec * sw + in % n;
        b = hb / Hq; hq = hb % Hq;
    }
    const int hkv = hq / G;
    const int tid = threadIdx.x, wg = tid / WGT;
    // Warp / lane numbers are taken AFTER each warpgroup's setmaxnreg, inside its branch: computed here, the
    // loader's warp number was kept across the register re-split by a 4-byte spill (ptxas -v, first build).
    const int k0 = kt * BN;                             // relative to the row (dense) or the segment (A)
    const int q_first = 2 * kt;
    int nIter = S / BM - q_first;                       // >= 1: k0 < S and S % 64 == 0
    if constexpr (CHATS) nIter = (L + BM - 1) / BM - q_first;   // A: >= 1 because k0 < L

    extern __shared__ __align__(1024) char smem_raw[];
    // Round the base up to 1024 B by pointer arithmetic on smem_raw, so the compiler still knows every pointer below
    // is shared memory. The first version went through (uintptr_t) and back: all sLSE/sD/sdQ/staging accesses became
    // generic LD.E/ST.E (ncu 2026-10-12: 0 LDS/STS, 19.5M LD.E.64 + 25.4M ST.E*, lg_throttle stalls, ~11% of samples).
    char* sm = smem_raw + ((1024u - ((uint32_t)__cvta_generic_to_shared(smem_raw) & 1023u)) & 1023u);
    bf16*  sK   = reinterpret_cast<bf16*>(sm + OFF_K);
    bf16*  sV   = reinterpret_cast<bf16*>(sm + OFF_V);
    bf16*  sQ   = reinterpret_cast<bf16*>(sm + OFF_Q);     // [stage][Q_TILE]
    bf16*  sdO  = reinterpret_cast<bf16*>(sm + OFF_DO);    // [stage][Q_TILE]
    bf16*  sdS  = reinterpret_cast<bf16*>(sm + OFF_DS);    // [stage][DS_TILE]
    float* sdQ  = reinterpret_cast<float*>(sm + OFF_DQ);   // [group][DQ_WG]
    float* sLSE = reinterpret_cast<float*>(sm + OFF_LSE);  // [stage][BM]
    float* sD   = reinterpret_cast<float*>(sm + OFF_D);    // [stage][BM]
    uint64_t* bars    = reinterpret_cast<uint64_t*>(sm + OFF_BAR);
    uint64_t* kv_full = bars;            // 1 arrival (+ 64 KB of TMA bytes)
    uint64_t* q_full  = bars + 1;        // [2]: 1 arrival (+ Q, dO, LSE, D bytes)
    uint64_t* q_empty = bars + 3;        // [2]: 1 arrival per compute group

    if (tid == 0) {
        mbar_init(kv_full, 1);
        mbar_init(&q_full[0], 1);  mbar_init(&q_full[1], 1);
        mbar_init(&q_empty[0], 2); mbar_init(&q_empty[1], 2);
    }
    __syncthreads();   // the only block-wide barrier: from here on the warpgroups only meet on named barriers

    if (wg == 0) {
        reg_dec_loader();
        const int warp = __shfl_sync(0xffffffffu, (int)(threadIdx.x >> 5), 0), lane = (int)(threadIdx.x & 31);
        if (warp == 0) {
            if (lane == 0) {
                const int kvRow = (b * Hkv + hkv) * S + t0 + k0;   // t0: A only (else 0)
                mbar_arrive_expect_tx(kv_full, KV_TX);
                tma_load_2d(sK,           &tma_K, 0,  kvRow, kv_full);
                tma_load_2d(sK + KV_HALF, &tma_K, 64, kvRow, kv_full);
                tma_load_2d(sV,           &tma_V, 0,  kvRow, kv_full);
                tma_load_2d(sV + KV_HALF, &tma_V, 64, kvRow, kv_full);
                for (int i = 0; i < nIter; i++) {
                    const int s = i & 1;
                    if (i >= 2) mbar_wait(&q_empty[s], ((i >> 1) - 1) & 1);   // both groups done with step i-2
                    const int qRow = (b * Hq + hq) * S + t0 + (q_first + i) * BM;   // t0: A only (else 0)
                    bf16* q_s  = sQ  + s * Q_TILE;
                    bf16* do_s = sdO + s * Q_TILE;
                    mbar_arrive_expect_tx(&q_full[s], QSTEP_TX);
                    tma_load_2d(q_s,           &tma_Q,  0,  qRow, &q_full[s]);
                    tma_load_2d(q_s + Q_HALF,  &tma_Q,  64, qRow, &q_full[s]);
                    tma_load_2d(do_s,          &tma_dO, 0,  qRow, &q_full[s]);
                    tma_load_2d(do_s + Q_HALF, &tma_dO, 64, qRow, &q_full[s]);
                    if constexpr (CHATS) {   // A: the step's padded slot (64-row aligned, see bwd_prep_vj2_chats)
                        bulk_load_1d(sLSE + s * BM, Lse2 + (long)(dqt0 + i) * BM, BM * 4, &q_full[s]);
                        bulk_load_1d(sD   + s * BM, Drow + (long)(dqt0 + i) * BM, BM * 4, &q_full[s]);
                    } else {
                        bulk_load_1d(sLSE + s * BM, Lse2 + qRow, BM * 4, &q_full[s]);
                        bulk_load_1d(sD   + s * BM, Drow + qRow, BM * 4, &q_full[s]);
                    }
                }
            }
        } else if (warp == 1) {
            // dQ writer: for every step, group 0's then group 1's fp32 dQ (16 KB, fragment order) goes to its
            // slot of the tile with one 1-D bulk reduce-add; the buffer is handed back once the copy has read it.
            for (int i = 0; i < nIter; i++) {
                float* tile = dQacc + ((long)(b * Hq + hq) * (S / BM) + (q_first + i)) * (BM * D);
                if constexpr (CHATS) tile = dQacc + (long)(dqt0 + i) * (BM * D);   // A: the step's padded slot
#pragma unroll
                for (int c = 0; c < 2; c++) {
                    bar_sync(BAR_DQ_FULL + c, WGT + 32);
                    if (lane == 0) {
                        bulk_reduce_add_1d(tile + c * DQ_WG, sdQ + c * DQ_WG, DQ_WG * 4);
                        bulk_commit();
                        bulk_wait_read();
                    }
                    __syncwarp();
                    if (i + 1 < nIter) bar_arrive(BAR_DQ_EMPTY + c, WGT + 32);
                }
            }
            if (lane == 0) bulk_wait_all();   // the last reduce-adds have landed before the block exits
        }
        return;
    }

    // ---- compute groups ------------------------------------------------------------------------------------
    reg_inc_compute();
    const int tw = (int)(threadIdx.x % WGT), warp = tw >> 5, lane = tw & 31;
    // 0 or 1: keys k0 + 64c .. k0 + 64c + 63. Broadcast from lane 0 so ptxas knows it is the same in the whole warp
    // and keeps everything built from it (smem bases, wgmma descriptors) in uniform registers, as CUTLASS's
    // canonical_warp_group_idx / FA3's warp_uniform (hopper/utils.h:621-623) do. Taken from threadIdx directly
    // (first version), the descriptors lived in vector registers and every HGMMA needed R2UR moves
    // (ncu 2026-10-13: R2UR 59M, IMAD 74M, LOP3 46M executed vs FA3's 2M / 8M / 0.6M).
    const int c = __shfl_sync(0xffffffffu, (int)(threadIdx.x / WGT), 0) - 1;
    const int kbase = k0 + 64 * c;
    const float scale2 = scale * LOG2E;
    const int t  = lane & 3;                      // accumulator column pair: cols 8nt + 2t, +1
    const int r0 = 16 * warp + (lane >> 2);       // accumulator rows (keys of this group): r0, r0 + 8
    const int ph = lane & 7, mm = lane >> 3;      // stmatrix addressing (Vj1d fused_p_stsm, :281-289)
    // Base descriptors, made once: this group's K/V rows, the Q / dO / dS stage-0 tiles, K head-half c.
    const uint64_t dK_rows  = desc_sw128(sK + 64 * c * 64);
    const uint64_t dV_rows  = desc_sw128(sV + 64 * c * 64);
    const uint64_t dQ_base  = desc_sw128(sQ);
    const uint64_t dO_base  = desc_sw128(sdO);
    const uint64_t dS_base  = desc_sw128(sdS);
    const uint64_t dK_halfc = desc_sw128(sK + c * KV_HALF);
    float* dq_stage = sdQ + c * DQ_WG;

    float dv[64], dk[64];
    zero_f<64>(dv); zero_f<64>(dk);
    mbar_wait(kv_full, 0);

    for (int i = 0; i < nIter; i++) {
        const int s = i & 1;
        const int q0 = (q_first + i) * BM;
        bf16* ds_s = sdS + s * DS_TILE;
        const uint64_t dq_s  = desc_add(dQ_base, s * Q_TILE);
        const uint64_t ddo_s = desc_add(dO_base, s * Q_TILE);
        const uint64_t dds_s = desc_add(dS_base, s * DS_TILE);
        mbar_wait(&q_full[s], (i >> 1) & 1);

        // LSE * log2e of this thread's 16 query columns, read BEFORE the score GEMMs are issued (G3, 2026-10-13).
        // The wgmma instructions wait in the same issue queue as these loads, so a load placed after them only
        // returns once they are dispatched: read after the dP issue (first versions) or between the two GEMMs
        // (tried, null), the first FFMA using each value waited (short_scoreboard 10-12% of all samples). FA3's
        // SASS issues the same 8 LDS right after its first S HGMMA and those FFMAs barely wait (ncu 2026-10-13).
        float l2[16];
#pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            l2[2 * nt]     = sLSE[s * BM + 8 * nt + 2 * t];
            l2[2 * nt + 1] = sLSE[s * BM + 8 * nt + 2 * t + 1];
        }
        float S_[32], dP[32];
        zero_f<32>(S_); zero_f<32>(dP);
        gemm_kq_issue(S_, dK_rows, dq_s);         // group A: S^T
        gemm_kq_issue(dP, dV_rows, ddo_s);        // group B: dP^T
        wgmma_wait1();                            // S^T ready, dP^T still running
        fence_f<32>(S_);
        // P = exp(S * scale - LSE); on the first two steps (the ones holding the diagonal) hide key > query.
        // A: the same, in segment-relative indices; rows past L have LSE2 = +inf, so their P is 0 on every step.
        // Two copies of the exp loop behind one branch: with the mask (the diagonal steps) and without (all others).
        // The first version tested `diag` inside the one unrolled element loop; ptxas made that predicated code, so
        // EVERY step ran the key/query index math + compare + select (ncu 2026-10-16, B8 18/6: FSEL 38.9M = one per
        // MUFU.EX2; 705M instructions vs FA3's 626M). Tried and dropped (2026-10-16): (1) two step loops, masked then
        // unmasked (FA3's way): ptxas spilled 128 B per step in the unmasked copy; (2) the mask as its own block after
        // the exp loop: 617M instructions, but the diagonal steps got slower (the mask no longer hides in the exp
        // latency), packed backward +0.58% in 12 of 12 race rounds. Here the diagonal steps run the original
        // interleaved code and the other steps skip the mask.
        const bool diag = (q0 < k0 + BN);
        if (diag) {
#pragma unroll
            for (int nt = 0; nt < 8; nt++) {
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    float p = ex2_approx(__fmaf_rn(S_[4 * nt + e], scale2, -l2[2 * nt + (e & 1)]));
                    const int key = kbase + r0 + ((e & 2) ? 8 : 0);
                    const int qry = q0 + 8 * nt + 2 * t + (e & 1);
                    S_[4 * nt + e] = (key > qry) ? 0.f : p;
                }
            }
        } else {
#pragma unroll
            for (int nt = 0; nt < 8; nt++) {
#pragma unroll
                for (int e = 0; e < 4; e++)
                    S_[4 * nt + e] = ex2_approx(__fmaf_rn(S_[4 * nt + e], scale2, -l2[2 * nt + (e & 1)]));
            }
        }
        float dd[16];                             // D of the same 16 query columns
#pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            dd[2 * nt]     = sD[s * BM + 8 * nt + 2 * t];
            dd[2 * nt + 1] = sD[s * BM + 8 * nt + 2 * t + 1];
        }
        wgmma_wait0();                            // dP^T ready
        fence_f<32>(dP);
        // dS = P * (dP - D), from the fp32 P (FA3 :893), then both to bf16.
        uint32_t pP0[8], pP1[8], pS0[8], pS1[8];  // [n8 block]: rows r0 / r0 + 8
#pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            const float d0 = dd[2 * nt], d1 = dd[2 * nt + 1];
            const float s0 = S_[4 * nt + 0] * (dP[4 * nt + 0] - d0);
            const float s1 = S_[4 * nt + 1] * (dP[4 * nt + 1] - d1);
            const float s2 = S_[4 * nt + 2] * (dP[4 * nt + 2] - d0);
            const float s3 = S_[4 * nt + 3] * (dP[4 * nt + 3] - d1);
            pP0[nt] = pack_bf16(S_[4 * nt + 0], S_[4 * nt + 1]);
            pP1[nt] = pack_bf16(S_[4 * nt + 2], S_[4 * nt + 3]);
            pS0[nt] = pack_bf16(s0, s1);
            pS1[nt] = pack_bf16(s2, s3);
        }
        // dS^T -> sdS[s] rows 64c.. (row = key, 64 queries = one 128-B swizzled row), for the dQ GEMM.
        {
            const int row0 = 64 * c + 16 * warp + ph, row1 = row0 + 8;   // (row & 7) == ph for both
            const uint32_t a0 = (uint32_t)__cvta_generic_to_shared(&ds_s[row0 * 64 + (((mm    ) ^ ph) << 3)]);
            const uint32_t a1 = (uint32_t)__cvta_generic_to_shared(&ds_s[row0 * 64 + (((mm + 4) ^ ph) << 3)]);
            const uint32_t a2 = (uint32_t)__cvta_generic_to_shared(&ds_s[row1 * 64 + (((mm    ) ^ ph) << 3)]);
            const uint32_t a3 = (uint32_t)__cvta_generic_to_shared(&ds_s[row1 * 64 + (((mm + 4) ^ ph) << 3)]);
            stmatrix_x4(a0, pS0[0], pS0[1], pS0[2], pS0[3]);
            stmatrix_x4(a1, pS0[4], pS0[5], pS0[6], pS0[7]);
            stmatrix_x4(a2, pS1[0], pS1[1], pS1[2], pS1[3]);
            stmatrix_x4(a3, pS1[4], pS1[5], pS1[6], pS1[7]);
        }
        // A fragments for k-step kc (queries 16kc..16kc+15): {row r0 cols 2t.., row r0+8, row r0 cols 8+2t.., row
        // r0+8} = n8 blocks 2kc and 2kc+1 -- the forward's P@V packing (GQA_fwd_sm90.cu:638-643).
        uint32_t aP[16], aS[16];
#pragma unroll
        for (int kc = 0; kc < 4; kc++) {
            aP[4 * kc + 0] = pP0[2 * kc];  aP[4 * kc + 1] = pP1[2 * kc];
            aP[4 * kc + 2] = pP0[2 * kc + 1]; aP[4 * kc + 3] = pP1[2 * kc + 1];
            aS[4 * kc + 0] = pS0[2 * kc];  aS[4 * kc + 1] = pS1[2 * kc];
            aS[4 * kc + 2] = pS0[2 * kc + 1]; aS[4 * kc + 3] = pS1[2 * kc + 1];
        }
        fence_u<16>(aP); fence_u<16>(aS);
        wgmma_fence();                            // after every register write the GEMMs below read
        gemm_rs_issue(dv, aP, ddo_s);             // dV += P^T . dO
        wgmma_commit();                           // group 1: dV
        fence_proxy_async_shared();               // this thread's dS stores -> visible to the tensor cores
        bar_sync(BAR_PDS, 2 * WGT);               // both groups' dS halves are in sdS[s]
        float dq[32];
        zero_f<32>(dq);
        gemm_dq_issue(dq, dds_s, dK_halfc);       // group 2: dQ (head cols 64c..)
        // FA3's order (mainloop_bwd_sm90_tma_gmma_ws.hpp:941-951, B-2a 2026-10-14): wait until dV is done before
        // issuing dK, then wait until dQ is done. The first version issued dK right after dQ with no wait in
        // between; the 4 warps then waited for each other at the dK GEMM's first HGMMA ("warpgroup arrive",
        // 215 us-equivalent vs FA3's 141 on B8 18/6, ncu 2026-10-14).
        wgmma_wait1();                            // dV done, dQ still running
        fence_f<64>(dv);
        wgmma_fence();
        gemm_rs_issue(dk, aS, dq_s);              // dK += dS^T . Q
        wgmma_commit();                           // group 3: dK
        wgmma_wait1();                            // dQ done, dK still running
        fence_f<32>(dq);
        if (i > 0) bar_sync(BAR_DQ_EMPTY + c, WGT + 32);   // the writer has read the previous dQ
        store_acc_frag_f32(dq, dq_stage, tw);     // unscaled; convert_dq_frag_vj2 applies the scale
        fence_proxy_async_shared();
        bar_arrive(BAR_DQ_FULL + c, WGT + 32);    // hand this step's dQ to the writer, do not wait
        wgmma_wait0();                            // dK done: nothing of this step reads sQ/sdO/sdS any more
        fence_f<64>(dk); fence_u<16>(aP); fence_u<16>(aS);
        if (tw == 0) mbar_arrive(&q_empty[s]);
    }

    // ---- epilogue: dV, dK of this group's 64 keys -> bf16 partial of this query head ----------------------
    // Both groups must be done with every GEMM that reads sQ/sdO before either reuses them as staging.
    bar_sync(BAR_PDS, 2 * WGT);
    bf16* stg = (c == 0) ? sQ : sdO;              // 2 x Q_TILE = 4 blocks of [64][64] per group
    stage_acc_bf16_sw128(dv,      stg,              tw, 1.0f);   // stg and each Q_HALF step: 1024-B aligned
    stage_acc_bf16_sw128(dv + 32, stg + Q_HALF,     tw, 1.0f);
    stage_acc_bf16_sw128(dk,      stg + 2 * Q_HALF, tw, scale);
    stage_acc_bf16_sw128(dk + 32, stg + 3 * Q_HALF, tw, scale);
    fence_proxy_async_shared();
    bar_sync(BAR_EPI + c, WGT);
    if constexpr (CHATS) {
        // A: this group's keys are rows kbase .. kbase+63 of the segment; only the rows < L are the segment's.
        const int nrow = L - kbase;
        const int hqRow = (b * Hq + hq) * S + t0 + kbase;
        if (nrow >= 64) {
            if (tw == 0) {
                tma_store_2d(&tma_dV_st, stg,              0,  hqRow);
                tma_store_2d(&tma_dV_st, stg + Q_HALF,     64, hqRow);
                tma_store_2d(&tma_dK_st, stg + 2 * Q_HALF, 0,  hqRow);
                tma_store_2d(&tma_dK_st, stg + 3 * Q_HALF, 64, hqRow);
                bulk_commit();
                bulk_wait_all();
            }
        } else if (nrow > 0) {
            // Plain 16-B stores of rows < nrow from the swizzled staging (stage_acc_bf16_sw128: chunk k of row r sits
            // at chunk k ^ (r & 7)). Blocks: dV cols 0-63, dV cols 64-127, dK cols 0-63, dK cols 64-127.
            for (int e = tw; e < 4 * 64 * 8; e += WGT) {
                const int blk = e >> 9, r = (e >> 3) & 63, k = e & 7;
                if (r >= nrow) continue;
                const bf16* src = stg + blk * Q_HALF + r * 64 + ((k ^ (r & 7)) << 3);
                bf16* dst = ((blk < 2) ? dVp : dKp) + ((long)hqRow + r) * D + (blk & 1) * 64 + k * 8;
                *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
            }
        }
    } else {
    if (tw == 0 && kbase < S) {                   // S % 128 == 64: group 1 of the last block has no keys
        const int hqRow = (b * Hq + hq) * S + kbase;
        tma_store_2d(&tma_dV_st, stg,              0,  hqRow);
        tma_store_2d(&tma_dV_st, stg + Q_HALF,     64, hqRow);
        tma_store_2d(&tma_dK_st, stg + 2 * Q_HALF, 0,  hqRow);
        tma_store_2d(&tma_dK_st, stg + 3 * Q_HALF, 64, hqRow);
        bulk_commit();
        bulk_wait_all();
    }
    }
}

// Fixed-order head sum of the per-query-head partials (copy of Vj1d gqa_dkdv_greduce, :408-426).
__global__ void gqa_dkdv_greduce_vj2(const bf16* __restrict__ partial, bf16* __restrict__ out,
                                     int Hq, int Hkv, int G, int S) {
    constexpr int Dd = 128, RPB = 8;
    int b = blockIdx.z, hkv = blockIdx.y, s0 = blockIdx.x * RPB;
    int t = threadIdx.x, row = t >> 4, dg = t & 15;
    int s = s0 + row;
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int g = 0; g < G; g++) {
        const bf16* p = partial + (((long)(b * Hq + hkv * G + g) * S + s) * Dd + dg * 8);
        uint4 v = *reinterpret_cast<const uint4*>(p);
        const bf16* vb = reinterpret_cast<const bf16*>(&v);
#pragma unroll
        for (int k = 0; k < 8; k++) acc[k] += __bfloat162float(vb[k]);
    }
    bf16 o[8];
#pragma unroll
    for (int k = 0; k < 8; k++) o[k] = __float2bfloat16(acc[k]);
    *reinterpret_cast<uint4*>(out + (((long)(b * Hkv + hkv) * S + s) * Dd + dg * 8)) =
        *reinterpret_cast<const uint4*>(o);
}

// dQ fp32 accumulator (fragment order, see store_acc_frag_f32) -> row-major bf16 [rows][128]. One block per
// 64 x 128 tile (tile t = rows 64t..64t+63). Float4 k of the tile = group c = k >> 10, n8 block j = (k >> 7) & 7,
// thread tw = k & 127 (warp w = tw >> 5, lane l): rows 16w + (l >> 2) and +8, cols 64c + 8j + 2(l & 3), +1.
// The tile goes through shared memory (rows padded to 136 bf16: the 4-byte stores of a warp hit 32 different
// banks) so the global stores are whole 16-B row chunks. The accumulator holds unscaled dQ; the softmax scale
// is applied here, once per value (see store_acc_frag_f32).
// A (CHATS): tile = padded tile hq * nT + T of the plan; it holds token rows row0 .. row0 + nrows - 1 of row b
// (dq_tiles[T] = {b, row0, nrows, 0}); its padding rows (past the chat's end) are not written.
constexpr int CVT_THREADS = 256, CVT_LD = 136;
template<bool CHATS>
__global__ void __launch_bounds__(CVT_THREADS)
convert_dq_frag_vj2(const float4* __restrict__ in, bf16* __restrict__ out, float scale,
                    const int4* __restrict__ dq_tiles, int nT, int Hq, int S) {   // the last four: A only
    __shared__ __align__(16) bf16 st[BM * CVT_LD];
    const long tile = blockIdx.x;
    const float4* src = in + tile * (BM * D / 4);
#pragma unroll
    for (int m = 0; m < (BM * D / 4) / CVT_THREADS; m++) {
        const int k = threadIdx.x + m * CVT_THREADS;
        const int c = k >> 10, j = (k >> 7) & 7, tw = k & 127, w = tw >> 5, l = tw & 31;
        const int r0 = 16 * w + (l >> 2), col = 64 * c + 8 * j + 2 * (l & 3);
        const float4 v = src[k];
        *reinterpret_cast<__nv_bfloat162*>(&st[r0 * CVT_LD + col]) =
            __floats2bfloat162_rn(v.x * scale, v.y * scale);
        *reinterpret_cast<__nv_bfloat162*>(&st[(r0 + 8) * CVT_LD + col]) =
            __floats2bfloat162_rn(v.z * scale, v.w * scale);
    }
    __syncthreads();
    uint4* dst = reinterpret_cast<uint4*>(out + tile * (BM * D));
    int nrows = BM;
    if constexpr (CHATS) {
        const int hq = (int)(tile / nT);
        const int4 dt = dq_tiles[tile - (long)hq * nT];
        nrows = dt.z;
        dst = reinterpret_cast<uint4*>(out + (((long)dt.x * Hq + hq) * S + dt.y) * D);
    }
#pragma unroll
    for (int m = 0; m < (BM * D / 8) / CVT_THREADS; m++) {
        const int k = threadIdx.x + m * CVT_THREADS, row = k >> 4, ch = k & 15;
        if (!CHATS || row < nrows) dst[k] = *reinterpret_cast<const uint4*>(&st[row * CVT_LD + ch * 8]);
    }
}

#else   // ---- non-Hopper: keep the translation unit compilable (same signatures, empty bodies) ----
__global__ void bwd_prep_vj2(const bf16* __restrict__ d_dO, const bf16* __restrict__ d_O,
                             const float* __restrict__ d_LSE, float* __restrict__ d_Drow,
                             float* __restrict__ d_Lse2, float* __restrict__ d_dqa, long nRows) {}
__global__ void bwd_prep_vj2_chats(const bf16* __restrict__ d_dO, const bf16* __restrict__ d_O,
                                   const float* __restrict__ d_LSE, float* __restrict__ d_Drow,
                                   float* __restrict__ d_Lse2, float* __restrict__ d_dqa,
                                   const int4* __restrict__ dq_tiles, int nT, int Hq, int S, long nRowsP) {}
template<bool CHATS>
__global__ void gqa_bwd_vj2_kernel(
    const __grid_constant__ CUtensorMap tma_K,  const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_Q,  const __grid_constant__ CUtensorMap tma_dO,
    const __grid_constant__ CUtensorMap tma_dV_st, const __grid_constant__ CUtensorMap tma_dK_st,
    float* __restrict__ dQacc,
    const float* __restrict__ Lse2, const float* __restrict__ Drow,
    int B, int Hq, int Hkv, int G, int S, float scale, int sw,
    const int4* __restrict__ blocks, bf16* __restrict__ dVp, bf16* __restrict__ dKp) {}
__global__ void gqa_dkdv_greduce_vj2(const bf16* __restrict__ partial, bf16* __restrict__ out,
                                     int Hq, int Hkv, int G, int S) {}
constexpr int CVT_THREADS = 256;
template<bool CHATS>
__global__ void convert_dq_frag_vj2(const float4* __restrict__ in, bf16* __restrict__ out, float scale,
                                    const int4* __restrict__ dq_tiles, int nT, int Hq, int S) {}
#endif  // __CUDA_ARCH__

// ---- host launcher -------------------------------------------------------------------------------------------
// Block-order section size, as FA3 (tile_scheduler.hpp:389-398): as many (batch, head) pairs as fit their Q + dO (bf16)
// and dQ accumulator (fp32) in 40 MB of L2, rounded down to a power of 2, at least 1. The dense grid uses it below;
// gqa_chat_plan_build orders design A's blocks with the same sections (gqa_vj2_section_pairs).
static int section_pairs(int S) {
    const long one_head = (long)S * (2 * D) * 2 + (long)S * D * 4;
    int sw = 1;
    while (2L * sw * one_head <= 40L * 1024 * 1024) sw *= 2;
    return sw;
}

static void launch_gqa_backward_vj2(
    const bf16* d_Q, const bf16* d_K, const bf16* d_V, const bf16* d_O,
    const bf16* d_dO, const float* d_LSE,
    bf16* d_dQ, bf16* d_dK, bf16* d_dV,
    int B, int Hq, int Hkv, int G, int S, float scale,
    const int32_t* d_blocks, int n_blocks,              // design A: the plan's Vj2 block list, or nullptr = dense
    const int32_t* d_dqt, int n_dqt) {                  //   and its padded dQ tiles (n_dqt per query head)
    const bool chats = d_blocks != nullptr;
    auto encode = [](CUtensorMap* m, CUtensorMapDataType dt, const void* ptr, uint64_t cols, uint64_t rows,
                     uint64_t row_bytes, uint32_t box_c, uint32_t box_r, CUtensorMapSwizzle sw, const char* what) {
        uint64_t gS[2] = { cols, rows }; uint64_t gT[1] = { row_bytes };
        uint32_t bx[2] = { box_c, box_r }, eS[2] = { 1, 1 };
        CUresult r = cuTensorMapEncodeTiled(m, dt, 2, const_cast<void*>(ptr), gS, gT, bx, eS,
                                            CU_TENSOR_MAP_INTERLEAVE_NONE, sw, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (r != CUDA_SUCCESS) { const char* e; cuGetErrorString(r, &e); fprintf(stderr, "vj2 %s map: %s\n", what, e); exit(1); }
    };
    const uint64_t Rq = (uint64_t)B * Hq * S, Rkv = (uint64_t)B * Hkv * S;
    CUtensorMap tK, tV, tQ, tdO, tdV, tdK;
    encode(&tK,  CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, d_K,  D, Rkv, D * 2, 64, BN, CU_TENSOR_MAP_SWIZZLE_128B, "K");
    encode(&tV,  CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, d_V,  D, Rkv, D * 2, 64, BN, CU_TENSOR_MAP_SWIZZLE_128B, "V");
    encode(&tQ,  CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, d_Q,  D, Rq,  D * 2, 64, BM, CU_TENSOR_MAP_SWIZZLE_128B, "Q");
    encode(&tdO, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, d_dO, D, Rq,  D * 2, 64, BM, CU_TENSOR_MAP_SWIZZLE_128B, "dO");

    // Scratch from the caching allocator, sized on first use and grown when a call needs more (as Vj1d).
    // Drow / Lse2 get 64 spare rows so a 256-B bulk copy can never read past their end.
    auto& alloc = CachingCUDAAllocator::instance();
    const long partN = (long)B * Hq * S * D;
    static bf16 *d_dKp = nullptr, *d_dVp = nullptr; static long pcap = 0;
    if (partN > pcap) {
        if (d_dKp) alloc.deallocate(d_dKp);
        if (d_dVp) alloc.deallocate(d_dVp);
        d_dKp = static_cast<bf16*>(alloc.allocate(partN * sizeof(bf16)));
        d_dVp = static_cast<bf16*>(alloc.allocate(partN * sizeof(bf16)));
        pcap = partN;
    }
    const long rowsN = (long)B * Hq * S;
    // LSE*log2e / D rows and the fp32 dQ accumulator: one per token row (dense), or one per PADDED row of the plan
    // (design A: every chat 64-row aligned per head, see bwd_prep_vj2_chats), which is never fewer.
    const long rowsP = chats ? (long)Hq * n_dqt * BM : rowsN;
    static float *d_Drow = nullptr, *d_Lse2 = nullptr; static long rcap = 0;
    if (rowsP > rcap) {
        if (d_Drow) alloc.deallocate(d_Drow);
        if (d_Lse2) alloc.deallocate(d_Lse2);
        d_Drow = static_cast<float*>(alloc.allocate((rowsP + 64) * sizeof(float)));
        d_Lse2 = static_cast<float*>(alloc.allocate((rowsP + 64) * sizeof(float)));
        rcap = rowsP;
    }
    if (((uintptr_t)d_Drow & 15) || ((uintptr_t)d_Lse2 & 15)) {
        fprintf(stderr, "vj2: LSE/D scratch is not 16-B aligned (bulk copy needs it)\n"); exit(1);
    }
    const long dqN = rowsP * D;
    static float* d_dqa = nullptr; static long dqc = 0;
    if (dqN > dqc) {
        if (d_dqa) alloc.deallocate(d_dqa);
        d_dqa = static_cast<float*>(alloc.allocate(dqN * sizeof(float)));
        dqc = dqN;
    }
    // dQ accumulator: zeroed by bwd_prep_vj2(_chats) below (B-1), no separate cudaMemset (dqN == rowsP * 128).
    encode(&tdV,  CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, d_dVp, D, Rq, D * 2, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, "dV out");
    encode(&tdK,  CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, d_dKp, D, Rq, D * 2, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, "dK out");
    if ((uintptr_t)d_dqa & 15) { fprintf(stderr, "vj2: dQ accumulator is not 16-B aligned (bulk reduce needs it)\n"); exit(1); }

    const int pBlock = 256; const long pGrid = (rowsP + (pBlock / 32) - 1) / (pBlock / 32);
    const int4* dqt = reinterpret_cast<const int4*>(d_dqt);
    if (chats)
        bwd_prep_vj2_chats<<<(unsigned)pGrid, pBlock>>>(d_dO, d_O, d_LSE, d_Drow, d_Lse2, d_dqa, dqt, n_dqt, Hq, S, rowsP);
    else
        bwd_prep_vj2<<<(unsigned)pGrid, pBlock>>>(d_dO, d_O, d_LSE, d_Drow, d_Lse2, d_dqa, rowsN);

    static bool attr_set = false, attr_set_chats = false;
    if (!chats && !attr_set) {
        GQA_BWD_VJ2_CUDA_CHECK(cudaFuncSetAttribute(gqa_bwd_vj2_kernel<false>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_BYTES));
        attr_set = true;
    }
    if (chats && !attr_set_chats) {
        GQA_BWD_VJ2_CUDA_CHECK(cudaFuncSetAttribute(gqa_bwd_vj2_kernel<true>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_BYTES));
        attr_set_chats = true;
    }
    const int sw = section_pairs(S);
    if (chats) {   // design A: one block per plan entry, in the plan's order (LPT inside sections of sw pairs)
        gqa_bwd_vj2_kernel<true><<<(unsigned)n_blocks, NTHREADS, SMEM_BYTES>>>(tK, tV, tQ, tdO, tdV, tdK, d_dqa, d_Lse2,
                                    d_Drow, B, Hq, Hkv, G, S, scale, sw, reinterpret_cast<const int4*>(d_blocks), d_dVp, d_dKp);
    } else {
        const unsigned nBlocks = (unsigned)((S / BN) * B * Hq);  // S % 128 == 0 here (the router's condition)
        gqa_bwd_vj2_kernel<false><<<nBlocks, NTHREADS, SMEM_BYTES>>>(tK, tV, tQ, tdO, tdV, tdK, d_dqa, d_Lse2, d_Drow,
                                                                    B, Hq, Hkv, G, S, scale, sw, nullptr, nullptr, nullptr);
    }
    dim3 grGrid(S / 8, Hkv, B);
    gqa_dkdv_greduce_vj2<<<grGrid, 128>>>(d_dVp, d_dV, Hq, Hkv, G, S);
    gqa_dkdv_greduce_vj2<<<grGrid, 128>>>(d_dKp, d_dK, Hq, Hkv, G, S);
    if (chats) {
        const long nTiles = (long)Hq * n_dqt;               // every padded tile of every head
        convert_dq_frag_vj2<true><<<(unsigned)nTiles, CVT_THREADS>>>(reinterpret_cast<const float4*>(d_dqa), d_dQ,
                                                                     scale, dqt, n_dqt, Hq, S);
    } else {
        const long nTiles = (long)B * Hq * (S / BM);         // S % 128 == 0 here (the router's condition)
        convert_dq_frag_vj2<false><<<(unsigned)nTiles, CVT_THREADS>>>(reinterpret_cast<const float4*>(d_dqa), d_dQ,
                                                                      scale, nullptr, 0, 0, 0);
    }
}

} // namespace gqa_bwd_vj2

// ---------------------------------------------------------------------------
// Library-facing entry point, called from GQA_bwd_generic.cu: the default Hopper backward since 2026-10-15 (causal,
// head_dim 128, S % 128 == 0, H100/H200; USE_VJ2=0 selects Vj1d instead). chat_blocks / dq_tiles = the Vj2 lists of
// a GqaChatPlan (packed rows, design A), or nullptr = dense.
// ---------------------------------------------------------------------------
void gqa_vj2_causal_bwd_launch_d128(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* O, const __nv_bfloat16* dO, const float* LSE,
    __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV,
    int B, int Hq, int Hkv, int G, int S, float scale,
    const int32_t* chat_blocks, int n_chat_blocks, const int32_t* dq_tiles, int n_dq_tiles) {
    gqa_bwd_vj2::launch_gqa_backward_vj2(Q, K, V, O, dO, LSE, dQ, dK, dV, B, Hq, Hkv, G, S, scale,
                                         chat_blocks, n_chat_blocks, dq_tiles, n_dq_tiles);
}
// The section size of Vj2's block order for rows of S tokens (gqa_chat_plan_build orders design A's blocks with it).
int gqa_vj2_section_pairs(int S) { return gqa_bwd_vj2::section_pairs(S); }

} // namespace cuda
} // namespace OwnTensor
