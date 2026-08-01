// =====================================================================================
// Standalone causal GQA attention for Hopper (sm_90a) -- kernel v62.
//
// Extracted from the tier3 development ladder into a single self-contained file: the
// constants, the 26 device helpers v62 actually reaches (transitively), the kernel, and
// the host-side launcher. Nothing here depends on the rest of the project.
//
// WHAT IT IS: a persistent, warp-specialised FlashAttention-style kernel.
//   * 132 CTAs (one per H200 SM), 384 threads = 1 TMA producer warpgroup + 2 consumer
//     warpgroups, setmaxnreg 40/232.
//   * BLOCK_M=64 queries x BLOCK_N=128 keys, head_dim 128, bf16 in / bf16 out + fp32 LSE.
//   * NSTAGES=3 mbarrier K/V pipeline fed by TMA (cp.async.bulk.tensor.2d).
//   * 128B-swizzled smem operands; native m64n128k16 wgmma for Q@K^T and m64n64k16
//     (trans_b) for P@V, each group emitted from ONE asm block so ptxas cannot insert
//     per-wgmma sync.
//   * FA3-style consumer ping-pong on named barriers 1/2, so one warpgroup's softmax
//     overlaps the other's tensor-core work.
//   * Online softmax with the scale folded into the exponent FFMA, ex2.approx, a branched
//     causal mask, seeded accumulators, and a deferred row-sum reduction.
//   * QK(n+1) is issued before P@V(n) is waited on (cross-iteration GEMM overlap), and the
//     Q buffer is released before the epilogue so the producer can prefetch the next tile.
//
// Measured on H200 at B=8 Hq=12 Hkv=4 S=4096 D=128 (causal, bf16): 0.7075 ms median,
// 579 TFLOPS -- about 1.6% faster than PyTorch SDPA's cuDNN path on the same shape.
// =====================================================================================

// Link with -lcuda: the TMA descriptors are built with cuTensorMapEncodeTiled, which is a
// CUDA *driver* API entry point and is not in cudart.
#include "gqa_v62.cuh"

#include <cuda_bf16.h>
#include <cuda.h>       // CUtensorMap, cuTensorMapEncodeTiled (TMA)
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cassert>   // the TMA builders assert on cuTensorMapEncodeTiled

namespace gqa62 {

constexpr int HEAD_DIM = 128;
constexpr int WG = 128;                   // threads per warpgroup
constexpr int ACC_PV = 64;                // m64n128 output (head=128)
constexpr float LOG2E = 1.4426950408889634f;
constexpr float LN2   = 0.6931471805599453f;
constexpr int NUM_WG = 3;                  // 1 producer + 2 consumers
constexpr int NUM_THREADS = NUM_WG * WG;   // 384
constexpr int NUM_SMS = 132;
// v60: 2-warpgroup config. probe4 established that ptxas charges the register cap per WARPGROUP --
// 65536 / (128 * ceil(T/128)) -- so 257..384 threads all get 168 and the cap only opens at <=256,
// where it is 254. Dropping the (permanently idle) producer warpgroup is the only way to reach it.
constexpr int BLOCK_M = 64, BLOCK_N = 128;
constexpr int D_CHUNKS = HEAD_DIM / 16;   // 8  (QK head contraction)
constexpr int N_CHUNKS = BLOCK_N / 16;    // 8  (P@V key contraction)
constexpr int ACC_QK   = 64;              // m64n128 scores (2 x m64n64 key-halves)
constexpr int TILE_K   = BLOCK_N * HEAD_DIM;   // 16384
constexpr int TILE_Q   = BLOCK_M * HEAD_DIM;   // 8192
constexpr int SWZ_BLK  = BLOCK_N * 64;         // 8192  (one SW128 head-half block: [128 keys][64])
// KHALF (SWZ_BLK/2 = 4096, the keys-64..127 offset inside a head-half block) is not needed at
// n=128: the native m64n128k16 wgmma reads the whole [128 keys][64] block, so nothing selects
// a key-half any more. Kept only as a comment because the swizzle layout is easier to read
// with it named.
constexpr uint32_t TMA_BYTES = BLOCK_N * HEAD_DIM * sizeof(__nv_bfloat16); // 32768 (one K or V tile)
constexpr int NSTAGES  = 3;
// How far ahead the merged issuer runs. NOT NSTAGES: refilling the buffer the consumer just released
// would make the issuer block on BOTH warpgroups' empty[] arrivals for the tile that just finished,
// serialising it against the ping-pong baton. At distance 2 the target buffer was released a full
// iteration earlier, so the empty[] wait is already satisfied in steady state.
constexpr int SWZ_BLK_Q = BLOCK_M * 64;                       // 4096 (Q head-half block)
constexpr uint32_t Q_TMA_BYTES = BLOCK_M * 64 * sizeof(__nv_bfloat16); // 8192 (one Q SW128 box)

// ---- guards for the two constant-capture bugs the tier2->tier3 port hit --------------------------
// Both were "same name, different value in another namespace" traps that only blow up at runtime
// (code 700). These make them compile-time errors instead.
static_assert(SWZ_BLK_Q == BLOCK_M * 64, "Q head-half stride must follow BLOCK_M (Q box rows)");
static_assert(SWZ_BLK   == BLOCK_N * 64, "K/V head-half stride must follow BLOCK_N (K/V box rows)");
static_assert(TILE_Q == 2 * SWZ_BLK_Q,   "Qt holds exactly 2 head-half boxes");
static_assert(TILE_K == 2 * SWZ_BLK,     "a K (or V) tile holds exactly 2 head-half boxes");
// highest element any Q / K descriptor can address must stay inside its buffer:
static_assert(((D_CHUNKS - 1) / 4) * SWZ_BLK_Q + ((D_CHUNKS - 1) % 4) * 16 + 16 <= TILE_Q,
              "Q descriptor base overruns Qt (this is the code-700 the port hit)");
static_assert(((D_CHUNKS - 1) / 4) * SWZ_BLK   + ((D_CHUNKS - 1) % 4) * 16 + 16 <= TILE_K,
              "K descriptor base overruns the K tile");
// TMA expect_tx must match what the 2+2 box loads actually transfer:
static_assert(2 * Q_TMA_BYTES == 2 * BLOCK_M * 64 * sizeof(__nv_bfloat16), "Q expect_tx mismatch");
static_assert(2 * TMA_BYTES   == 4 * BLOCK_N * 64 * sizeof(__nv_bfloat16), "K+V expect_tx mismatch");

// ---- shared-memory budget (v34 layout) + the v52 all-ones tile ----------------------------------
// The v34 layout is 229456 B of the 232448 B (227 KB) a Hopper CTA may hold => 2992 B spare.
// v52 spends 2048 of that on a tile of 1.0f used as the B operand of the row-sum wgmma. Its real
// footprint is ~256 B (16k x 8n x 2B); the tile is oversized on purpose so that no plausible
// misreading of the descriptor's LBO/SBO can walk off the end of the CTA's shared window.
constexpr size_t T3_SMEM_V34   = (size_t)2 * NSTAGES * TILE_K * sizeof(__nv_bfloat16)   // sK + sV
                               + (size_t)2 * TILE_Q * sizeof(__nv_bfloat16)             // Qt
                               + (size_t)(2 * NSTAGES + 4) * sizeof(uint64_t);          // mbarriers
                                                                       // low 4 address bits; 128-B
                                                                       // align so none of the ones
                                                                       // region rounds into a mbarrier

// ---- host TMA tensor maps -------------------------------------------------
static CUtensorMap make_tma_kv128(const __nv_bfloat16* g, uint64_t rows, uint64_t cols) {
    CUtensorMap tmap{};
    uint64_t gdim[2]    = { cols, rows };
    uint64_t gstride[1] = { cols * sizeof(__nv_bfloat16) };
    uint32_t bdim[2]    = { 64, (uint32_t)BLOCK_N };   // [128 keys][64 head] box
    uint32_t estride[2] = { 1, 1 };
    CUresult res = cuTensorMapEncodeTiled(
        &tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<__nv_bfloat16*>(g),
        gdim, gstride, bdim, estride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    assert(res == CUDA_SUCCESS); (void)res;
    return tmap;
}

// Q tensor map: SW128 with a [BLOCK_M][64] box. NOTE the box rows are BLOCK_M (=64), NOT BLOCK_N.
// (tier2's v15 version used BLOCK_N because v15 inherited v13's BLOCK_N=64, where the two happened to
// be equal; copying it into t3 -- which has BLOCK_N=128 -- silently made a [128][64] Q box, so the
// producer's 2 TMAs wrote 16384 elems into the 8192-elem Qt buffer -> smem overflow -> code 700.)
static CUtensorMap make_tma_q64(const __nv_bfloat16* g, uint64_t rows, uint64_t cols) {
    CUtensorMap tmap{};
    uint64_t gdim[2]    = { cols, rows };
    uint64_t gstride[1] = { cols * sizeof(__nv_bfloat16) };
    uint32_t bdim[2]    = { 64, (uint32_t)BLOCK_M };
    uint32_t estride[2] = { 1, 1 };
    CUresult res = cuTensorMapEncodeTiled(
        &tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<__nv_bfloat16*>(g),
        gdim, gstride, bdim, estride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    assert(res == CUDA_SUCCESS); (void)res;
    return tmap;
}

#if __CUDA_ARCH__ >= 900 && __CUDA_ARCH__ < 1000   // Hopper only

// ---- device helpers (copies of the validated tier2 primitives) ------------
__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {
    uint32_t a = __cvta_generic_to_shared(bar);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(a), "r"(count));
}

__device__ __forceinline__ void mbar_wait(uint64_t* bar, int phase) {
    uint32_t a = __cvta_generic_to_shared(bar);
    asm volatile(
        "{\n\t.reg .pred P;\n"
        "LW2_%=:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P, [%0], %1;\n\t"
        "@P bra.uni LD2_%=;\n\t"
        "bra.uni LW2_%=;\n"
        "LD2_%=:\n\t}" :: "r"(a), "r"(phase));
}

__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
    uint32_t a = __cvta_generic_to_shared(bar);
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(a) : "memory");
}

__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
    uint32_t a = __cvta_generic_to_shared(bar);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(a), "r"(bytes) : "memory");
}

__device__ __forceinline__ void tma_load_2d(uint32_t smem_addr, const void* tmap,
                                             int c, int r, uint64_t* bar) {
    uint32_t b = __cvta_generic_to_shared(bar);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(smem_addr), "l"(tmap), "r"(c), "r"(r), "r"(b) : "memory");
}

__device__ __forceinline__ void wgmma_fence()  { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }

__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }

__device__ __forceinline__ void wgmma_wait0()  { asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); }

__device__ __forceinline__ float rcp_approx(float x) {
    float r; asm("rcp.approx.f32 %0, %1;" : "=f"(r) : "f"(x)); return r;
}

__device__ __forceinline__ void wgmma_wait1()  { asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory"); }  // v58: retire the older group, keep 1 in flight

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return reinterpret_cast<uint32_t&>(v);
}

__device__ __forceinline__ float quad_max(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
    return v;
}

__device__ __forceinline__ float quad_sum(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    return v;
}

__device__ __forceinline__ float ex2_approx(float x) {
    float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y;
}

__device__ __forceinline__ void reg_dec_producer() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(40));
}

__device__ __forceinline__ void reg_inc_consumer() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(232));
}

// Q and K/V have DIFFERENT head-half strides: a Q SW128 box is [BLOCK_M=64 rows][64] = 4096 elems,
// a K/V box is [BLOCK_N=128 rows][64] = 8192. So they need SEPARATE base helpers.
// (TRAP: in tier2 these were accidentally interchangeable -- Q used v15::swz2_base, whose SWZ_BLK was
// v13::BLOCK_N*64 = 4096 = SWZ_BLK_Q, while K used v33::SWZ_BLK = 8192. Two different values behind one
// name. Flattening both into t3 collapsed them to 8192, so the Q descriptor for kc>=4 indexed 8192 into
// the 8192-elem Qt buffer -> 4096 elems out of bounds -> code 700.)
__device__ __forceinline__ int swzq_base(int kc)  { return (kc / 4) * SWZ_BLK_Q + (kc % 4) * 16; }  // Q

__device__ __forceinline__ int swzkv_base(int kc) { return (kc / 4) * SWZ_BLK   + (kc % 4) * 16; }  // K/V

// Derive a SIBLING wgmma descriptor by a compile-time ELEMENT offset, instead of running a whole
// second cvta+mask+shift+or chain. The descriptor's address field is bits 0..13 = (a & 0x3FFFF) >> 4,
// so for a bf16 element offset E (= 2E bytes) the sibling is simply  desc + (2E)/16 == desc + (E >> 3).
// LBO/SBO/swizzle-mode bits are identical between siblings, so nothing else changes.
// SAFE (no carry out of the 14-bit address field): smem here is 229456 B < 0x40000, and the largest
// (base + offset) is V stage 2 == 98304 + 2*32768 + 30720 = 194560 B, so (a & 0x3FFFF) + 2E never wraps
// the 18-bit mask. Every offset used below is a multiple of 8 elements (=16 B), asserted at the call.
// Same as desc_add, but does the arithmetic in 32 BITS ONLY. The address field is bits 0..13, so a
// sibling offset can never carry out of the low word -- yet ptxas emits a 64-bit add, i.e. MOV + IADD3
// + IMAD.X (carry propagate) per descriptor. The v43 profile shows those 16 IMAD.X in the P@V chain at
// ~0.18% each (~2.3% of samples) doing nothing. Splicing the constant high word back drops the carry.
// ---- FA3-style CONSUMER PING-PONG (v45) ----------------------------------------------------------
// WHY: the two consumer warpgroups work on DIFFERENT q-tiles with NO phase relationship, so they drift
// into phase and hit their wgmma DEPBARs together. The v43 roofline says exactly that: 2.99 active but
// only **0.52 ELIGIBLE** warps/scheduler, 61.6% of cycles with ZERO issuable warp. And more occupancy
// is IMPOSSIBLE -- one CTA is 384x168 = 64512 of 65536 registers (98.4% of the file), so 2 CTAs/SM
// would need <=85 regs/thread and a 4th warpgroup <=128. The only remaining slack is SCHEDULING the 12
// warps we have: make the softmax an ALTERNATING critical section so that while one warpgroup is in the
// ALU/XU-heavy softmax, the other is inside its tensor GEMMs.
// MECHANISM: named barriers with count 256 (= both consumer warpgroups; id 0 is reserved for
// __syncthreads, which the SASS shows as BAR.SYNC.DEFER_BLOCKING 0x0). `bar.arrive` registers this
// warpgroup's 128 arrivals WITHOUT blocking; `bar.sync` blocks until the OTHER warpgroup has arrived
// too (128+128 = 256). So barrier 1 = WG1 hands the baton to WG0, barrier 2 = WG0 hands it to WG1.
// The producer never touches ids 1/2. Safe from divergent-trip-count deadlock because adjacent pairing
// gives my_nkv == nkv_shared == pair+1 for BOTH consumers (verified), and the else-branch below keeps
// the counts paired even if that ever stopped holding.
__device__ __forceinline__ void pp_sync1()   { asm volatile("bar.sync   1, 256;" ::: "memory"); }

__device__ __forceinline__ void pp_sync2()   { asm volatile("bar.sync   2, 256;" ::: "memory"); }

__device__ __forceinline__ void pp_arrive1() { asm volatile("bar.arrive 1, 256;" ::: "memory"); }

__device__ __forceinline__ void pp_arrive2() { asm volatile("bar.arrive 2, 256;" ::: "memory"); }

__device__ __forceinline__ uint64_t desc_add_lo(uint64_t d, int elem_off) {
    uint32_t lo = (uint32_t)d + (uint32_t)(elem_off >> 3);
    return ((uint64_t)(uint32_t)(d >> 32) << 32) | (uint64_t)lo;
}

__device__ __forceinline__ uint64_t make_swz_desc(const void* p) {
    uint32_t a = __cvta_generic_to_shared(p);
    uint64_t d = 0;
    d |= (uint64_t)((a & 0x3FFFFu) >> 4);
    d |= ((uint64_t)(16u   >> 4) << 16);
    d |= ((uint64_t)(1024u >> 4) << 32);
    d |= (1ull << 62);
    return d;
}

    // ---- FUSED wgmma groups (v35+) ------------------------------------------------------------
    // WHY: the v34 SASS showed ptxas inserting a WARPGROUP.ARRIVE + WARPGROUP.DEPBAR pair around
    // EVERY wgmma -- about 15 pct of all samples (10.1 in QK + 5.2 in P@V) -- serializing the
    // accumulation chain. Our PTX was already optimal (2 fences / 2 commits / 2 waits for 24
    // wgmma), so ptxas added that sync itself. cuDNN's SASS has NO DEPBAR between its wgmma, only
    // one at the end of the group. Emitting a whole group from ONE asm volatile block leaves
    // ptxas no gap to insert sync into, so the group pipelines the way cuDNN's does.
    __device__ __forceinline__ void wgmma_qk128_x8(const uint64_t da[8], const uint64_t db[8],
                                                   float acc[64]) {
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %64, %65, 1, 1, 1, 0, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %66, %67, 1, 1, 1, 0, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %68, %69, 1, 1, 1, 0, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %70, %71, 1, 1, 1, 0, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %72, %73, 1, 1, 1, 0, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %74, %75, 1, 1, 1, 0, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %76, %77, 1, 1, 1, 0, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %78, %79, 1, 1, 1, 0, 0;\n"
            :
          "+f"(acc[0]),"+f"(acc[1]),"+f"(acc[2]),"+f"(acc[3]),"+f"(acc[4]),"+f"(acc[5]),"+f"(acc[6]),"+f"(acc[7]),
          "+f"(acc[8]),"+f"(acc[9]),"+f"(acc[10]),"+f"(acc[11]),"+f"(acc[12]),"+f"(acc[13]),"+f"(acc[14]),"+f"(acc[15]),
          "+f"(acc[16]),"+f"(acc[17]),"+f"(acc[18]),"+f"(acc[19]),"+f"(acc[20]),"+f"(acc[21]),"+f"(acc[22]),"+f"(acc[23]),
          "+f"(acc[24]),"+f"(acc[25]),"+f"(acc[26]),"+f"(acc[27]),"+f"(acc[28]),"+f"(acc[29]),"+f"(acc[30]),"+f"(acc[31]),
          "+f"(acc[32]),"+f"(acc[33]),"+f"(acc[34]),"+f"(acc[35]),"+f"(acc[36]),"+f"(acc[37]),"+f"(acc[38]),"+f"(acc[39]),
          "+f"(acc[40]),"+f"(acc[41]),"+f"(acc[42]),"+f"(acc[43]),"+f"(acc[44]),"+f"(acc[45]),"+f"(acc[46]),"+f"(acc[47]),
          "+f"(acc[48]),"+f"(acc[49]),"+f"(acc[50]),"+f"(acc[51]),"+f"(acc[52]),"+f"(acc[53]),"+f"(acc[54]),"+f"(acc[55]),
          "+f"(acc[56]),"+f"(acc[57]),"+f"(acc[58]),"+f"(acc[59]),"+f"(acc[60]),"+f"(acc[61]),"+f"(acc[62]),"+f"(acc[63])
            : "l"(da[0]), "l"(db[0]),
              "l"(da[1]), "l"(db[1]),
              "l"(da[2]), "l"(db[2]),
              "l"(da[3]), "l"(db[3]),
              "l"(da[4]), "l"(db[4]),
              "l"(da[5]), "l"(db[5]),
              "l"(da[6]), "l"(db[6]),
              "l"(da[7]), "l"(db[7]));
    }

    __device__ __forceinline__ void wgmma_pv_half_x8(float o[32], const uint32_t a[32],
                                                     const uint64_t b[8]) {
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %64, 1, 1, 1, 1;\n"
            "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%36,%37,%38,%39}, %65, 1, 1, 1, 1;\n"
            "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%40,%41,%42,%43}, %66, 1, 1, 1, 1;\n"
            "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%44,%45,%46,%47}, %67, 1, 1, 1, 1;\n"
            "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%48,%49,%50,%51}, %68, 1, 1, 1, 1;\n"
            "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%52,%53,%54,%55}, %69, 1, 1, 1, 1;\n"
            "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%56,%57,%58,%59}, %70, 1, 1, 1, 1;\n"
            "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%60,%61,%62,%63}, %71, 1, 1, 1, 1;\n"
            :
          "+f"(o[0]),"+f"(o[1]),"+f"(o[2]),"+f"(o[3]),"+f"(o[4]),"+f"(o[5]),"+f"(o[6]),"+f"(o[7]),
          "+f"(o[8]),"+f"(o[9]),"+f"(o[10]),"+f"(o[11]),"+f"(o[12]),"+f"(o[13]),"+f"(o[14]),"+f"(o[15]),
          "+f"(o[16]),"+f"(o[17]),"+f"(o[18]),"+f"(o[19]),"+f"(o[20]),"+f"(o[21]),"+f"(o[22]),"+f"(o[23]),
          "+f"(o[24]),"+f"(o[25]),"+f"(o[26]),"+f"(o[27]),"+f"(o[28]),"+f"(o[29]),"+f"(o[30]),"+f"(o[31])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
              "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]),
              "r"(a[8]), "r"(a[9]), "r"(a[10]), "r"(a[11]),
              "r"(a[12]), "r"(a[13]), "r"(a[14]), "r"(a[15]),
              "r"(a[16]), "r"(a[17]), "r"(a[18]), "r"(a[19]),
              "r"(a[20]), "r"(a[21]), "r"(a[22]), "r"(a[23]),
              "r"(a[24]), "r"(a[25]), "r"(a[26]), "r"(a[27]),
              "r"(a[28]), "r"(a[29]), "r"(a[30]), "r"(a[31]),
              "l"(b[0]), "l"(b[1]), "l"(b[2]), "l"(b[3]), "l"(b[4]), "l"(b[5]), "l"(b[6]), "l"(b[7]));
    }

__global__ __launch_bounds__(NUM_THREADS, 1) void gqa_v62(
    const __grid_constant__ CUtensorMap tmapKsw,
    const __grid_constant__ CUtensorMap tmapVsw,
    const __grid_constant__ CUtensorMap tmapQsw,
    const __nv_bfloat16* __restrict__ Q,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int Hq, int Hkv, int S, float scale)
{
    const int G = Hq / Hkv;
    const int tid = threadIdx.x;
    const int wg = tid / WG;
    const int lane_wg = tid % WG;
    const float scale2 = scale * LOG2E;

    const int nTiles = S / BLOCK_M;
    const int total_units = B * Hq * (nTiles / 4);
    auto decode_u = [&](int u, int half, int& b, int& hq, int& hkv, int& pair) {
        const int NC = nTiles / 4;
        int cpl = u % NC, bh = u / NC;
        pair = half ? (nTiles / 2 - 1 - cpl) : cpl;
        b = bh / Hq; hq = bh % Hq; hkv = hq / G;
    };

    extern __shared__ __align__(128) char smem_raw[];
    __nv_bfloat16* sK   = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* sV   = sK + NSTAGES * TILE_K;
    __nv_bfloat16* Qt   = sV + NSTAGES * TILE_K;
    uint64_t*      full = reinterpret_cast<uint64_t*>(Qt + 2 * TILE_Q);
    uint64_t*      empty= full + NSTAGES;
    uint64_t*      qfull = empty + NSTAGES;
    uint64_t*      qempty= qfull + 2;

    if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < NSTAGES; ++i) { mbar_init(&full[i], 1); mbar_init(&empty[i], 2); }
        #pragma unroll
        for (int i = 0; i < 2; ++i) { mbar_init(&qfull[i], 1); mbar_init(&qempty[i], 1); }
    }
    __syncthreads();

    auto shared_nkv = [&](int pair) { return pair + 1; };

    if (wg == 0) {
        // PRODUCER: unchanged from v49/v53. It is ~100% idle (its whole sample share is spin), so
        // there is nothing to win here; leaving it alone keeps the diff to the consumer.
        reg_dec_producer();
        int g = 0, j = 0, stage = 0, ph = 0;
        for (int uidx = blockIdx.x; uidx < total_units; uidx += NUM_SMS) {
            #pragma unroll 1
            for (int half = 0; half < 2; ++half) {
            int b, hq, hkv, pair; decode_u(uidx, half, b, hq, hkv, pair);
            if (j > 0) { mbar_wait(&qempty[0], (j - 1) & 1); mbar_wait(&qempty[1], (j - 1) & 1); }
            if (lane_wg == 0) {
                #pragma unroll
                for (int c = 0; c < 2; ++c) {
                    int q_tile = 2 * pair + c;
                    int q_row = (b * Hq + hq) * S + q_tile * BLOCK_M;
                    __nv_bfloat16* Qd = &Qt[c * TILE_Q];
                    mbar_expect_tx(&qfull[c], 2 * Q_TMA_BYTES);
                    tma_load_2d(__cvta_generic_to_shared(&Qd[0]),         &tmapQsw, 0,  q_row, &qfull[c]);
                    tma_load_2d(__cvta_generic_to_shared(&Qd[SWZ_BLK_Q]), &tmapQsw, 64, q_row, &qfull[c]);
                }
            }
            const int nkv = shared_nkv(pair);
            const int kv_row_base = (b * Hkv + hkv) * S;
            for (int n = 0; n < nkv; ++n) {
                if (g >= NSTAGES) mbar_wait(&empty[stage], ph ^ 1);
                if (lane_wg == 0) {
                    int row = kv_row_base + n * BLOCK_N;
                    __nv_bfloat16* Kd = &sK[stage * TILE_K];
                    __nv_bfloat16* Vd = &sV[stage * TILE_K];
                    mbar_expect_tx(&full[stage], 2 * TMA_BYTES);
                    tma_load_2d(__cvta_generic_to_shared(&Kd[0]),      &tmapKsw, 0,  row, &full[stage]);
                    tma_load_2d(__cvta_generic_to_shared(&Kd[SWZ_BLK]),&tmapKsw, 64, row, &full[stage]);
                    tma_load_2d(__cvta_generic_to_shared(&Vd[0]),      &tmapVsw, 0,  row, &full[stage]);
                    tma_load_2d(__cvta_generic_to_shared(&Vd[SWZ_BLK]),&tmapVsw, 64, row, &full[stage]);
                }
                ++g;
                if (++stage == NSTAGES) { stage = 0; ph ^= 1; }
            }
            ++j;
        }
        }
        return;
    }
    {
        const int c = wg - 1;
        reg_inc_consumer();
        const int warp = lane_wg / 32, lane = lane_wg % 32;
        const int groupID = lane / 4, tig = lane % 4;
        const int R0 = 16 * warp + groupID, R1 = R0 + 8;
        __nv_bfloat16* myQt = &Qt[c * TILE_Q];
        int g = 0, j = 0, stage = 0, ph = 0;
        if (c == 1) pp_arrive1();

        for (int uidx = blockIdx.x; uidx < total_units; uidx += NUM_SMS) {
            #pragma unroll 1
            for (int half = 0; half < 2; ++half) {
            int b, hq, hkv, pair; decode_u(uidx, half, b, hq, hkv, pair);
            const int q_tile = 2 * pair + c;
            const int gR0 = q_tile * BLOCK_M + R0, gR1 = q_tile * BLOCK_M + R1;
            const long long q_base = ((long long)(b * Hq + hq) * S + (long long)q_tile * BLOCK_M) * HEAD_DIM;
            const int nkv_shared = shared_nkv(pair);
            const int my_nkv = (q_tile * BLOCK_M + BLOCK_M - 1) / BLOCK_N + 1;

            mbar_wait(&qfull[c], j & 1);

            float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
            float O_acc[ACC_PV];
            #pragma unroll
            for (int i = 0; i < ACC_PV; ++i) O_acc[i] = 0.0f;

            // ---- v58: QK(n+1) prefetch. S_acc is dead once packed into pa, so tile n+1's QK can be
            // issued before tile n's P@V is waited on. Commit P@V FIRST so the group FIFO is
            // [PV(n), QK(n+1)] and `wgmma.wait_group 1` retires P@V while QK stays in flight.
            float S_acc[ACC_QK];
            auto issue_qk = [&](__nv_bfloat16* Kt_) {
                #pragma unroll
                for (int i = 0; i < ACC_QK; ++i) S_acc[i] = 0.0f;
                wgmma_fence();
                const uint64_t qd0 = make_swz_desc(myQt), kd0 = make_swz_desc(Kt_);
                uint64_t qd[8], kd[8];
                #pragma unroll
                for (int kc = 0; kc < D_CHUNKS; ++kc) {
                    qd[kc] = desc_add_lo(qd0, swzq_base(kc));
                    kd[kc] = desc_add_lo(kd0, swzkv_base(kc));
                }
                wgmma_qk128_x8(qd, kd, S_acc);
                wgmma_commit();
            };
            if (nkv_shared > 0) {                       // prologue: put QK(0) in flight
                mbar_wait(&full[stage], ph);
                issue_qk(&sK[stage * TILE_K]);
            }
            // ACTIVE range only. Splitting the loop keeps the wgmma group bookkeeping UNIFORM inside the
            // body -- the first attempt kept `if (n < my_nkv)` around the wgmma with wait1 on one path and
            // wait0 on the other, and ptxas answered with C7518: it could not track the pending-group count
            // across the divergence and gave EVERY HGMMA its own ARRIVE/DEPBAR pair (24 groups per iteration
            // instead of 8-then-1), serialising the tensor pipe completely. CUTLASS splits for this reason.
            for (int n = 0; n < my_nkv; ++n) {
                const int cur = stage;
                int nstage = stage + 1, nph = ph;
                if (nstage == NSTAGES) { nstage = 0; nph ^= 1; }
                __nv_bfloat16* Vt = &sV[cur * TILE_K];
                wgmma_wait0();                          // QK(n) -- issued one iteration ago

                    const bool diag = (n * BLOCK_N + BLOCK_N - 1 >= q_tile * BLOCK_M);
                    float mA[8], mB[8];
                    if (!diag) {
                        #pragma unroll
                        for (int reg = 0; reg < ACC_QK; ++reg) {
                            int s = reg % 4, bq = (reg >> 2) & 7;
                            float v = S_acc[reg];
                            S_acc[reg] = v;
                            const bool seed = (reg < 32) && ((s & 1) == 0);
                            if (s < 2) mA[bq] = seed ? v : fmaxf(mA[bq], v);
                            else       mB[bq] = seed ? v : fmaxf(mB[bq], v);
                        }
                    } else {
                        #pragma unroll
                        for (int reg = 0; reg < ACC_QK; ++reg) {
                            int s = reg % 4, bq = (reg >> 2) & 7;
                            int keycol = 8 * (reg / 4) + 2 * tig + (reg & 1);
                            float v = S_acc[reg];
                            int gcol = n * BLOCK_N + keycol; int grow = (s < 2) ? gR0 : gR1;
                            if (gcol > grow) v = -INFINITY;
                            S_acc[reg] = v;
                            const bool seed = (reg < 32) && ((s & 1) == 0);
                            if (s < 2) mA[bq] = seed ? v : fmaxf(mA[bq], v);
                            else       mB[bq] = seed ? v : fmaxf(mB[bq], v);
                        }
                    }
                    float lmax0 = fmaxf(fmaxf(fmaxf(mA[0], mA[1]), fmaxf(mA[2], mA[3])),
                                        fmaxf(fmaxf(mA[4], mA[5]), fmaxf(mA[6], mA[7])));
                    float lmax1 = fmaxf(fmaxf(fmaxf(mB[0], mB[1]), fmaxf(mB[2], mB[3])),
                                        fmaxf(fmaxf(mB[4], mB[5]), fmaxf(mB[6], mB[7])));
                    float mt0 = quad_max(lmax0), mt1 = quad_max(lmax1);
                    float nm0 = fmaxf(m0, mt0), nm1 = fmaxf(m1, mt1);
                    float cc0 = ex2_approx((m0 - nm0) * scale2), cc1 = ex2_approx((m1 - nm1) * scale2);
                    const float b0 = -nm0 * scale2, b1 = -nm1 * scale2;
                    if (c == 0) pp_sync1(); else pp_sync2();   // v56: baton acquired HERE, not 154+120 instrs earlier
                    float sA[8], sB[8];
                    #pragma unroll
                    for (int reg = 0; reg < ACC_QK; ++reg) {
                        int s = reg % 4, bq = (reg >> 2) & 7;
                        float p = ex2_approx(__fmaf_rn(S_acc[reg], scale2, (s < 2) ? b0 : b1));
                        S_acc[reg] = p;
                        const bool seed = (reg < 32) && ((s & 1) == 0);
                        if (s < 2) sA[bq] = seed ? p : sA[bq] + p;
                        else       sB[bq] = seed ? p : sB[bq] + p;
                    }
                    float ls0 = ((sA[0] + sA[1]) + (sA[2] + sA[3])) + ((sA[4] + sA[5]) + (sA[6] + sA[7]));
                    float ls1 = ((sB[0] + sB[1]) + (sB[2] + sB[3])) + ((sB[4] + sB[5]) + (sB[6] + sB[7]));
                    l0 = l0 * cc0 + ls0;   // v57: per-lane partial, no SHFL in the section
                    l1 = l1 * cc1 + ls1;   // v57: reduced once in the epilogue instead
                    m0 = nm0; m1 = nm1;
                    if (!__all_sync(0xffffffffu, cc0 == 1.0f && cc1 == 1.0f)) {
                        #pragma unroll
                        for (int reg = 0; reg < ACC_PV; ++reg) O_acc[reg] *= (reg % 4 < 2) ? cc0 : cc1;
                    }

                    wgmma_fence();
                    uint32_t pa[32]; uint64_t vd[8];
                    #pragma unroll
                    for (int kc = 0; kc < N_CHUNKS; ++kc) {
                        int r = 8 * kc;
                        pa[4*kc+0] = pack_bf16(S_acc[r + 0], S_acc[r + 1]);
                        pa[4*kc+1] = pack_bf16(S_acc[r + 2], S_acc[r + 3]);
                        pa[4*kc+2] = pack_bf16(S_acc[r + 4], S_acc[r + 5]);
                        pa[4*kc+3] = pack_bf16(S_acc[r + 6], S_acc[r + 7]);
                    }
                    if (c == 0) pp_arrive2(); else pp_arrive1();
                    static_assert(1024 % 8 == 0, "desc_add_lo needs 8-element (16 B) aligned offsets");
                    const uint64_t vd0 = make_swz_desc(Vt);
                    #pragma unroll
                    for (int hh = 0; hh < 2; ++hh) {
                        #pragma unroll
                        for (int kc = 0; kc < N_CHUNKS; ++kc)
                            vd[kc] = desc_add_lo(vd0, hh * SWZ_BLK + kc * 1024);
                        wgmma_pv_half_x8(&O_acc[hh * 32], pa, vd);
                    }
                wgmma_commit();                         // groups: [PV(n)]
                if (n + 1 < nkv_shared) mbar_wait(&full[nstage], nph);
                // Unconditional so the wait depth is uniform. On the final active tile this QK is redundant
                // and its S_acc is discarded (re-zeroed by the next prologue); it targets `nstage`, never the
                // buffer we are about to release, so `empty[cur]` stays safe to signal.
                issue_qk(&sK[nstage * TILE_K]);
                wgmma_wait1();                          // retire P@V(n), keep QK in flight
                if (lane_wg == 0) mbar_arrive(&empty[cur]);
                ++g;
                stage = nstage; ph = nph;
            }
            wgmma_wait0();                              // drain the trailing QK
            // ---- v61: RELEASE Qt[c] HERE, not after the epilogue. The last reader of Qt is the QK
            // wgmma, and the drain above has retired it; the epilogue below touches only O_acc/l/m.
            // In v58 this arrive sat after the O stores and the LSE writes, so the Q pipeline was a
            // full round trip at every pair boundary: consumer finishes -> runs the whole epilogue ->
            // releases qempty -> producer issues the Q TMA -> consumer waits qfull. The v58 SASS bills
            // that at 4.14% on the consumer's qfull wait (0x38030, line 184) plus 4.44%+2.19% on the
            // producer's qempty waits (0x38040/48, lines 1773/1775) -- ~11% of all samples on a
            // dependency that only exists because of where this line sat. Moving it up hands the
            // producer the buffer ~200 instructions early (quad_sum, 2 reciprocals, 64 O stores,
            // 2 logf polynomials), which is exactly the latency the Q TMA needs.
            if (lane_wg == 0) mbar_arrive(&qempty[c]);
            for (int n = my_nkv; n < nkv_shared; ++n) {  // inactive tail: baton + handshake only
                const int cur = stage;
                int nstage = stage + 1, nph = ph;
                if (nstage == NSTAGES) { nstage = 0; nph ^= 1; }
                if (c == 0) { pp_sync1(); pp_arrive2(); } else { pp_sync2(); pp_arrive1(); }
                if (n + 1 < nkv_shared) mbar_wait(&full[nstage], nph);
                if (lane_wg == 0) mbar_arrive(&empty[cur]);
                ++g;
                stage = nstage; ph = nph;
            }

            l0 = quad_sum(l0); l1 = quad_sum(l1);   // v57: the deferred reduction, once per q-tile
            const float r0 = rcp_approx(l0), r1 = rcp_approx(l1);   // v61 (was 1.0f/l, v53)
            // v62: pair the stores. In the accumulator map col = 8*bcol + 2*tig + (s&1), so regs
            // (reg, reg+1) for reg%4 in {0,2} share bcol AND row and land on col, col+1 -- ADJACENT in
            // O. Emitting them as one bf16x2 halves both the F2FP and the STG count, and widens each
            // store from 2 B to 4 B: across a warp the four tig lanes now cover 16 CONTIGUOUS bytes
            // per row instead of four 2-byte writes on a 4-byte stride, so a row costs one sector for
            // 16 B of payload instead of one sector for 8 B. Address is 4 B aligned (head is even and
            // q_base + row*HEAD_DIM is a multiple of 128).
            #pragma unroll
            for (int reg = 0; reg < ACC_PV; reg += 2) {
                int half2 = reg / 32, rr = reg % 32;
                int bcol = rr / 4, s = rr % 4;          // s is 0 or 2 -- reg+1 is the s|1 sibling
                int row = (s < 2) ? R0 : R1;
                int head = half2 * 64 + 8 * bcol + 2 * tig;
                float rcp = (s < 2) ? r0 : r1;
                *reinterpret_cast<__nv_bfloat162*>(&O[q_base + (long long)row * HEAD_DIM + head]) =
                    __floats2bfloat162_rn(O_acc[reg] * rcp, O_acc[reg + 1] * rcp);
            }
            if (tig == 0) {
                LSE[(long long)(b * Hq + hq) * S + gR0] = (m0 * scale2) * LN2 + logf(l0);
                LSE[(long long)(b * Hq + hq) * S + gR1] = (m1 * scale2) * LN2 + logf(l1);
            }
            ++j;
        }
        }
    }
}

#else   // ---- non-Hopper: keep the translation unit compilable ----
__global__ void gqa_v62(const __grid_constant__ CUtensorMap, const __grid_constant__ CUtensorMap,
                        const __grid_constant__ CUtensorMap, const __nv_bfloat16*,
                        __nv_bfloat16*, float*, int, int, int, int, float) {}
#endif

size_t smem_bytes() { return T3_SMEM_V34; }   // sK+sV + Qt + mbarriers

} // namespace gqa62

void launch_gqa_v62(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* O, float* LSE,
    int B, int Hq, int Hkv, int S, float scale, cudaStream_t stream)
{
    int device = 0; cudaGetDevice(&device);
    cudaDeviceProp prop{}; cudaGetDeviceProperties(&prop, device);
    if (prop.major != 9) {
        fprintf(stderr, "launch_gqa_v62: Hopper only; sm_%d%d -- skipping.\n", prop.major, prop.minor);
        return;
    }
    const uint64_t kv_rows = (uint64_t)B * Hkv * S;
    CUtensorMap tmapKsw = gqa62::make_tma_kv128(K, kv_rows, gqa62::HEAD_DIM);
    CUtensorMap tmapVsw = gqa62::make_tma_kv128(V, kv_rows, gqa62::HEAD_DIM);
    CUtensorMap tmapQsw = gqa62::make_tma_q64(Q, (uint64_t)B * Hq * S, gqa62::HEAD_DIM);  // Q box [64][64] SW128
    static bool cfg = false;
    size_t smem_bytes = gqa62::smem_bytes();
    if (!cfg) { cudaFuncSetAttribute(gqa62::gqa_v62, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes); cfg = true; }
    dim3 grid(gqa62::NUM_SMS); dim3 block(gqa62::NUM_THREADS);
    gqa62::gqa_v62<<<grid, block, smem_bytes, stream>>>(tmapKsw, tmapVsw, tmapQsw, Q, O, LSE, B, Hq, Hkv, S, scale);
}

size_t gqa_v62_smem_bytes() { return gqa62::smem_bytes(); }