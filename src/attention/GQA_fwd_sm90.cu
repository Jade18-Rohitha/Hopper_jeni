#define KV0_FIRST 1
#define SM3 1   // 2026-10-19 candidate PF = SM3m + the first K/V tile before the Q-buffer wait (FA3's producer order, work doc 45.53)
#define L2_HINT 2
#define L2_HINT_DENSE 1   // 2026-10-19 candidate LHOd = LHO (K5q f35982af + FA3's L2 hints + O evict-first) in the DENSE kernel only (work doc 45.50)
#define K4_EPI 1
#define K1_SPLIT 16
#define K1_DB 0   // 2026-10-17 candidate switch, see the K1 / K2 / K4 / K5 / K6 notes in the file
// SM90 (Hopper H100/H200) GQA causal FORWARD -- kernel v62.
//
// Persistent, warp-specialised FlashAttention-style kernel:
//   * 132 CTAs (one per H200 SM), 384 threads = 1 TMA producer warpgroup + 2
//     consumer warpgroups, setmaxnreg 40/232 (F3, 2026-10-16, tried FA3's 24/240: dense forward +1.1%).
//   * BLOCK_M=64 queries x BLOCK_N=128 keys, head_dim 128, bf16 in / bf16 out +
//     fp32 LSE.  NSTAGES=3 mbarrier K/V pipeline fed by TMA.
//   * 128B-swizzled smem operands; native m64n128k16 wgmma for Q@K^T and
//     m64n128k16 (trans_b, register A) for P@V (m64n64k16 per head half until
//     2026-10-16), each group emitted from ONE asm block so ptxas cannot insert
//     per-wgmma sync.
//   * FA3-style consumer ping-pong on named barriers 1/2, online softmax with the
//     scale folded into the exponent FFMA, ex2.approx, branched causal mask.
//
// Measured on H200 at B=8 Hq=12 Hkv=4 S=4096 D=128 (causal, bf16): 0.7075 ms
// median, 579 TFLOPS -- ~1.6% faster than PyTorch SDPA's cuDNN path.
//
// CONSTRAINTS (checked by the caller in GQA_fwd_generic.cu before dispatch):
//   causal only, head_dim == 128, S % 256 == 0, and the grid is a fixed 132 CTAs.
// The device code is guarded by __CUDA_ARCH__ >= 900 && < 1000, so this file
// still compiles (as an empty stub kernel) for non-Hopper archs.
//
// Requires -lcuda: cuTensorMapEncodeTiled is a CUDA *driver* API entry point.

#include <cuda_bf16.h>
#include <cuda.h>       // CUtensorMap, cuTensorMapEncodeTiled (TMA)
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cassert>
// K1 switches (2026-10-17 candidates): K1_DB = 1 two P register sets (default), 0 one set; V62_NSTAGES = K/V stages (3).
#ifndef K1_DB
#define K1_DB 1
#endif
#ifndef V62_NSTAGES
#define V62_NSTAGES 3
#endif
// K1_SPLIT = how many of the 64 scores are exponentiated BEFORE the wait for P@V(n-1) (64 = K1 as raced 2026-10-17 12:30; the
// rest are exponentiated after it, where ptxas can interleave them with the O rescale and the P packs, as FA3 does: FA3 has
// 13 exps before its wait, 53 after, mixed with its 64 rescale FMULs; trace 2026-10-17: K1's softmax 1,413 cycles per tile,
// all of it before the wait, + 104 for the rescale after).
#ifndef K1_SPLIT
#define K1_SPLIT 64
#endif
// K2_EPI = 1: the O store through shared memory (stmatrix + 16-byte stores); needs V62_NSTAGES 2 (32 KB of shared memory).
#ifndef K2_EPI
#define K2_EPI 0
#endif
// K6_PREF = 1: design A loads the NEXT unit's descriptor (units[next]) as soon as its index is known (during the last P@V)
// instead of at the unit's start (a global load the whole group waited for at every unit: trace 2026-10-17, packed).
#ifndef K6_PREF
#define K6_PREF 0
#endif
// K4_EPI = 1: the same O store, staged in the LAST tile's K buffer (keeps the 3 stages; see the epilogue).
#ifndef K4_EPI
#define K4_EPI 0
#endif
// L2_HINT (2026-10-19): L2 cache policies on the global loads / stores, as FA3 (flash-attention 94e22c9 hopper/
// mainloop_fwd_sm90_tma_gmma_ws.hpp: K and V loaded EVICT_LAST 753 / 767, Q EVICT_FIRST 820; FA3's O = TMA store, no hint).
// ncu 2026-10-19 (same box, dense B8 18/6): FA3 146M L2 sectors evict_last + 7M evict_first, ours all evict_normal; our L2
// misses 29.9M vs 23.1M sectors, DRAM reads 520 vs 365 MB with the same work order (G5 = FA3's L2 sections + LPT).
// 0: no hint (K5q); 1: FA3's hints (Q first, K / V last); 2: 1 + the O stores evict-first.
#ifndef L2_HINT
#define L2_HINT 0
#endif
// L2_HINT_DENSE = 1: the hints only in the dense kernel; the packed kernel keeps K5q's loads and stores (race lh_1019: LH /
// LHO dense -1.18 / -1.49%, packed +0.56 / +0.68%: in packed each unit's first K / V tile waited longer, DRAM unchanged).
#ifndef L2_HINT_DENSE
#define L2_HINT_DENSE 0
#endif
// SM3 (2026-10-19, work doc 45.52): the softmax in FA3's structure (flash-attention 94e22c9 hopper/softmax.h): the row max as
// ONE chain per row from the previous max, columns in order (thread_reduce_ with zero_init false, then quad_allreduce_), the
// exps row by row (scale_apply_exp2: row 0's 32, then row 1's), the row sums one chain per row from l * cc, columns in order
// (max_get_scale's row_sum *= scale, then reduce_sum). Same math and instruction counts as before; what changes is what ptxas
// can overlap: FA3's compiled loop runs row 0's exps while row 1's max chain is still going (2 FMNMX between exps), ours ran
// all the max first and then the exps nearly back to back (PC sampling, same box: +119 cycles per step of "wait" stalls on
// exps issued 0-2 instructions after the previous one). Part A = the first K1_SPLIT exps of ROW 0; the P.V-wait pin
// (order_after) takes row 0's running sum, i.e. the end of part A. 0: LHOd's softmax.
#ifndef SM3
#define SM3 0
#endif
// KV0_FIRST (2026-10-19, work doc 45.53): the producer loads a unit's FIRST K / V tile as soon as its slot is free, before
// waiting for both groups to release the previous Q (qempty), as FA3 does (flash-attention 94e22c9 hopper/
// mainloop_fwd_sm90_tma_gmma_ws.hpp: load_K :769, then the QueryEmpty barrier :776, then Q :780). Before: qempty -> Q -> K / V(0),
// so the first tile's 64 KB were requested only after both groups' last QK of the previous unit: ncu PC sampling (dense,
// same box) 1,498 cycles per unit at that wait, FA3 50 at its first-K wait. The ring order per barrier is unchanged
// (protocol model deep/protocol_sim_pf.py: clean on 3,000 + 3,000 random unit lists; its twin without the slot wait is caught).
#ifndef KV0_FIRST
#define KV0_FIRST 0
#endif
#include <type_traits>   // K1: std::true_type / false_type (mask instance of the softmax)

namespace OwnTensor {
namespace cuda {

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
constexpr int NSTAGES = V62_NSTAGES;   // K1: candidate switch (FA3 hdim 128 uses 2)
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
                               + (size_t)(2 * NSTAGES + 6) * sizeof(uint64_t)           // mbarriers (+2 sched_full, G5)
                               + 2 * sizeof(int)                                        // sched_unit[2] (G5)
                               + 2 * sizeof(uint64_t)                                   // order_bar[2] (K1)
                               + 2 * sizeof(uint64_t)                                   // kcount[2] (K4, 8 B used)
                               + (size_t)(K2_EPI ? 2 * TILE_Q : 0) * sizeof(__nv_bfloat16);  // sO (K2)
                                                                       // low 4 address bits; 128-B
                                                                       // align so none of the ones
                                                                       // region rounds into a mbarrier
static_assert(T3_SMEM_V34 <= 232448, "shared memory over the 227 KB a Hopper CTA may hold (K2_EPI needs V62_NSTAGES 2)");

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

// ---- G5 (2026-10-16): dynamic work order, as FA3's persistent tile scheduler (hopper/tile_scheduler.hpp, causal: LPT +
// a work counter). Each CTA takes its first unit at blockIdx.x, every next one from this counter (+ NUM_SMS), so an SM that
// finishes early takes more work instead of idling at the end. v62 gave CTA i the units i, i + 132, ... (static): dense
// B8 18/6 = 2304 equal units = 17.45 rounds, 60 SMs did 18 units and 72 did 17 (SMs busy 95.7% vs FA3's 99.6%, ncu
// 2026-10-13). The counter resets itself: every launch makes exactly total_units claims (each CTA's last claim is past
// the end), so the claim that returns total_units - 1 is the last one and sets it back to 0 for the next launch. One
// forward kernel at a time per device (the trainer's attention runs on one stream). Defined outside the Hopper guard so
// the host and device passes see the same variable.
__device__ int g_v62_next_unit = 0;

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

constexpr uint64_t L2_EVICT_FIRST = 0x12F0000000000000ull;   // L2_HINT: CUTLASS CacheHintSm90::EVICT_FIRST / EVICT_LAST
constexpr uint64_t L2_EVICT_LAST  = 0x14F0000000000000ull;

template <bool HINT>   // HINT: with the L2 cache policy (L2_HINT), else K5q's plain load
__device__ __forceinline__ void tma_load_2d(uint32_t smem_addr, const void* tmap,
                                             int c, int r, uint64_t* bar, uint64_t policy = 0) {
    uint32_t b = __cvta_generic_to_shared(bar);
    if constexpr (HINT) {
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
            " [%0], [%1, {%2, %3}], [%4], %5;"
            :: "r"(smem_addr), "l"(tmap), "r"(c), "r"(r), "r"(b), "l"(policy) : "memory");
    } else {
        (void)policy;
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%2, %3}], [%4];"
            :: "r"(smem_addr), "l"(tmap), "r"(c), "r"(r), "r"(b) : "memory");
    }
}

__device__ __forceinline__ void wgmma_fence()  { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }

__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }

__device__ __forceinline__ void wgmma_wait0()  { asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); }

__device__ __forceinline__ float rcp_approx(float x) {
    float r; asm("rcp.approx.f32 %0, %1;" : "=f"(r) : "f"(x)); return r;
}

__device__ __forceinline__ void wgmma_wait1()  { asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory"); }  // v58: retire the older group, keep 1 in flight

// K2: one warp-wide 8x8 x4 matrix store (each register = this thread's 2 elements of one matrix, as the mma C layout) and
// one 16-byte shared load.
// K4: release store / acquire load of a shared counter (the 'last QK done' count of a consumer group).
__device__ __forceinline__ void st_release_shared(unsigned int* p, unsigned int v) {
    asm volatile("st.release.cta.shared::cta.u32 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(p)), "r"(v) : "memory");
}
__device__ __forceinline__ unsigned int ld_acquire_shared(const unsigned int* p) {
    unsigned int v;
    asm volatile("ld.acquire.cta.shared::cta.u32 %0, [%1];" : "=r"(v) : "r"((uint32_t)__cvta_generic_to_shared(p)) : "memory");
    return v;
}
__device__ __forceinline__ void stmatrix_x4(uint32_t addr, const uint32_t v[4]) {
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                 :: "r"(addr), "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3]) : "memory");
}
__device__ __forceinline__ uint4 lds128(uint32_t addr) {
    uint4 d;
    asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];\n" : "=r"(d.x), "=r"(d.y), "=r"(d.z), "=r"(d.w) : "r"(addr) : "memory");
    return d;
}
// (a uint4 store through a cast pointer was split into four 4-byte STG by the compiler: SASS 2026-10-17)
template <bool HINT>   // HINT: the O store evict-first (L2_HINT 2), else K5q's plain store
__device__ __forceinline__ void stg128(void* p, const uint4& d) {
    if constexpr (HINT)
        asm volatile("st.global.L2::cache_hint.v4.b32 [%0], {%1, %2, %3, %4}, %5;\n"
                     :: "l"(p), "r"(d.x), "r"(d.y), "r"(d.z), "r"(d.w), "l"(L2_EVICT_FIRST) : "memory");
    else
        asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};\n" :: "l"(p), "r"(d.x), "r"(d.y), "r"(d.z), "r"(d.w) : "memory");
}
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
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(40));   // 40 / 232 (F3 tried FA3's 24 / 240: dense +1.1%)
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

    // P@V of one key tile: 8 x m64n128k16 (one per 16 keys), A = P from REGISTERS (bf16 fragments a[4kc..4kc+3]),
    // B = V from shared memory, MN-major (trans-b = 1). Full-width since 2026-10-16 (G1 for the forward, as the backward's
    // dV/dK on 2026-10-13): FA3/FA4/cuDNN all run P@V as 8 m64n128k16 per key tile; v62 issued 2 x 8 m64n64k16 (one
    // group per head half) and so read the P fragments twice. The m64n128 accumulator holds n8 block j in o[4j..4j+3]:
    // blocks 0-7 = head cols 0-63 = the old first half's o[0..31], blocks 8-15 = cols 64-127 = the old o[32..63], so
    // the softmax rescale and the epilogue are unchanged. b[kc]: see make_pv_desc.
    __device__ __forceinline__ void wgmma_pv128_x8(float o[64], const uint32_t a[32], const uint64_t b[8]) {
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, {%64,%65,%66,%67}, %96, 1, 1, 1, 1;\n"
            "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, {%68,%69,%70,%71}, %97, 1, 1, 1, 1;\n"
            "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, {%72,%73,%74,%75}, %98, 1, 1, 1, 1;\n"
            "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, {%76,%77,%78,%79}, %99, 1, 1, 1, 1;\n"
            "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, {%80,%81,%82,%83}, %100, 1, 1, 1, 1;\n"
            "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, {%84,%85,%86,%87}, %101, 1, 1, 1, 1;\n"
            "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, {%88,%89,%90,%91}, %102, 1, 1, 1, 1;\n"
            "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, {%92,%93,%94,%95}, %103, 1, 1, 1, 1;\n"
            :
          "+f"(o[0]),"+f"(o[1]),"+f"(o[2]),"+f"(o[3]),"+f"(o[4]),"+f"(o[5]),"+f"(o[6]),"+f"(o[7]),
          "+f"(o[8]),"+f"(o[9]),"+f"(o[10]),"+f"(o[11]),"+f"(o[12]),"+f"(o[13]),"+f"(o[14]),"+f"(o[15]),
          "+f"(o[16]),"+f"(o[17]),"+f"(o[18]),"+f"(o[19]),"+f"(o[20]),"+f"(o[21]),"+f"(o[22]),"+f"(o[23]),
          "+f"(o[24]),"+f"(o[25]),"+f"(o[26]),"+f"(o[27]),"+f"(o[28]),"+f"(o[29]),"+f"(o[30]),"+f"(o[31]),
          "+f"(o[32]),"+f"(o[33]),"+f"(o[34]),"+f"(o[35]),"+f"(o[36]),"+f"(o[37]),"+f"(o[38]),"+f"(o[39]),
          "+f"(o[40]),"+f"(o[41]),"+f"(o[42]),"+f"(o[43]),"+f"(o[44]),"+f"(o[45]),"+f"(o[46]),"+f"(o[47]),
          "+f"(o[48]),"+f"(o[49]),"+f"(o[50]),"+f"(o[51]),"+f"(o[52]),"+f"(o[53]),"+f"(o[54]),"+f"(o[55]),
          "+f"(o[56]),"+f"(o[57]),"+f"(o[58]),"+f"(o[59]),"+f"(o[60]),"+f"(o[61]),"+f"(o[62]),"+f"(o[63])
            :
              "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
              "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]),
              "r"(a[8]), "r"(a[9]), "r"(a[10]), "r"(a[11]),
              "r"(a[12]), "r"(a[13]), "r"(a[14]), "r"(a[15]),
              "r"(a[16]), "r"(a[17]), "r"(a[18]), "r"(a[19]),
              "r"(a[20]), "r"(a[21]), "r"(a[22]), "r"(a[23]),
              "r"(a[24]), "r"(a[25]), "r"(a[26]), "r"(a[27]),
              "r"(a[28]), "r"(a[29]), "r"(a[30]), "r"(a[31]),
              "l"(b[0]), "l"(b[1]), "l"(b[2]), "l"(b[3]), "l"(b[4]), "l"(b[5]), "l"(b[6]), "l"(b[7]));
    }

    // V descriptor for the full-width P@V: the SW128 descriptor of the V tile with its leading byte offset (bits 16-29,
    // 16-B units) set to the distance between the tile's two head-half blocks. For an MN-major SW128 operand the LBO is
    // the distance between the 64-element column blocks of N (CUTLASS make_gmma_desc<Major::MN>, B128:
    // ((T,8,m),(8,k)):((1,T,LBO),(8T,SBO))); N = 128 head cols span both blocks, SWZ_BLK elements = 16384 B apart.
    // Copy of desc_lbo, GQA_bwd_vj2_sm90.cu:255-257 (there LBO = 8192 B: its tiles hold 64 rows, here 128 keys).
    __device__ __forceinline__ uint64_t make_pv_desc(const void* vt) {
        const uint64_t d = make_swz_desc(vt);
        return (d & ~(0x3FFFull << 16)) | ((uint64_t)(((SWZ_BLK * 2) >> 4) & 0x3FFFu) << 16);
    }

// v2 packing: design B (one seg_start value per token, the FlexAttention document-mask way; added 2026-10-10)
// was REMOVED on 2026-10-15. Design A below does the same job faster (forward 2-16% on the 2026-10-10/12
// measurements: B's tiles that hold a chat border mix two chats, A starts every chat on its own tiles) and is
// the cu_seqlens way of FlashAttention 2/3/4, TE/Megatron THD, PyTorch varlen_attn and TRL. One packing
// representation keeps one tested path.
// v2 packing, design A (per-segment work list, the cu_seqlens way; added 2026-10-10), MODE_CHATS:
// the same packed rows, but the work is cut per segment instead of per row. units[u] =
// {b, t0, L, (cpl << 8) | hq}: segment [t0, t0+L) of row b (a chat or a pad tail; the segments of
// a row cover all of it), query head hq, and the couple cpl of the segment's query PAIRS -- pair cpl
// with pair nP-1-cpl (nP = ceil(L/128)), the same balancing as the dense decode_u inside the
// segment; for odd nP the middle pair is a 1-pair unit. Changes against the dense kernel:
//   * Q/K/V rows are offset by t0, and every tile index is segment-relative, so the diagonal mask is
//     exactly the dense causal mask of a sequence of length L. Keys past L only appear in a row's
//     diagonal tile, where causal already hides them (they are after every row of the segment).
//   * rows past L (the tail of the last pair) are computed but never stored (O and LSE); a q-tile
//     entirely past L gets no active tiles and runs only the existing inactive-tail handshake.
//   * no -inf guard: every row of a segment sees key t0 of its own segment.
// The same split is made by FlashAttention varlen (hopper/seqlen.h, tile_scheduler.hpp:593-700;
// rows past seqlen are not stored, epilogue_fwd.hpp:303,331,358) and TE/Megatron THD.
constexpr int MODE_DENSE = 0, MODE_CHATS = 1;

template <int MODE>
__global__ __launch_bounds__(NUM_THREADS, 1) void gqa_v62_sm90(
    const __grid_constant__ CUtensorMap tmapKsw,
    const __grid_constant__ CUtensorMap tmapVsw,
    const __grid_constant__ CUtensorMap tmapQsw,
    const __nv_bfloat16* __restrict__ Q,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int Hq, int Hkv, int S, float scale,
    const int4* __restrict__ units,          // v2 A: work list, read only when CHATS
    int n_units)
{
    constexpr bool CHATS  = (MODE == MODE_CHATS);
    constexpr bool HINT_LD = L2_HINT != 0 && !(L2_HINT_DENSE && CHATS);   // L2_HINT: Q / K / V loads with an L2 policy
    constexpr bool HINT_O  = L2_HINT >= 2 && !(L2_HINT_DENSE && CHATS);   // L2_HINT 2: the O stores too
    const int G = Hq / Hkv;
    const int tid = threadIdx.x;
    const int wg = tid / WG;
    const int lane_wg = tid % WG;
    const float scale2 = scale * LOG2E;

    const int nTiles = S / BLOCK_M;
    // G5: a dense unit is ONE query pair (128 rows, both consumer groups) instead of a couple (pair cpl with pair
    // NP-1-cpl). Order, as FA3's SingleTileLPTScheduler: sections of SW (batch, head) pairs, as many as keep their K/V
    // in 32 MB of L2 (FA3 tile_scheduler.hpp: size_l2 / one kv head's K+V, times the query heads per kv head; B8 18/6
    // S 4096: 16 kv heads = 48 query heads), inside a section the heaviest pairs first (pair NP-1 has NP key tiles),
    // heads inner. Design A's units stay couples from the plan (gqa_chat_plan_build: longest segment first).
    const int NP = nTiles / 2;
    int total_units = B * Hq * NP;
    if constexpr (CHATS) total_units = n_units;
    const int SW = min(B * Hq, max(1, (int)((32ll << 20) / ((long long)S * 4 * HEAD_DIM))) * G);
    auto decode_u = [&](int u, int& b, int& hq, int& hkv, int& pair) {
        const int sec = u / (SW * NP), in = u % (SW * NP);
        const int h0 = sec * SW, n = min(SW, B * Hq - h0);   // the last section may be short
        pair = NP - 1 - in / n;
        const int bh = h0 + in % n;
        b = bh / Hq; hq = bh % Hq; hkv = hq / G;
    };

    extern __shared__ __align__(128) char smem_raw[];
    __nv_bfloat16* sK   = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* sV   = sK + NSTAGES * TILE_K;
    __nv_bfloat16* Qt   = sV + NSTAGES * TILE_K;
#if K2_EPI
    __nv_bfloat16* sO   = Qt + 2 * TILE_Q;                      // K2: O staging, [2 groups][64 rows][128] bf16
    uint64_t*      full = reinterpret_cast<uint64_t*>(sO + 2 * TILE_Q);
#else
    uint64_t*      full = reinterpret_cast<uint64_t*>(Qt + 2 * TILE_Q);
#endif
    uint64_t*      empty= full + NSTAGES;
    uint64_t*      qfull = empty + NSTAGES;
    uint64_t*      qempty= qfull + 2;
    uint64_t*      sched_full = qempty + 2;     // G5: [2] unit number published (1 arrival: the producer)
    int*           sched_unit = reinterpret_cast<int*>(sched_full + 2);   // G5: [2] the unit, -1 = no more work
    uint64_t*      order_bar  = reinterpret_cast<uint64_t*>(sched_unit + 2);  // K1: [2] never waited on (see order_after)
    // K4: [2] how many pairs' last QK group c has finished (its K buffer may then be reused). A COUNTER, not an mbarrier: the
    // first K4 (2026-10-17, mbarrier + parity wait) hung in the quick suite. An inactive group (A) signals at the start of its
    // pair and can run a whole pair ahead; its second arrival then completed the phase the other group was about to wait
    // for, the parity looked unfinished, and that group waited for ever. A count that only grows cannot be overtaken.
    unsigned int*  kcount     = reinterpret_cast<unsigned int*>(order_bar + 2);

    if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < NSTAGES; ++i) { mbar_init(&full[i], 1); mbar_init(&empty[i], 2); }
        #pragma unroll
        for (int i = 0; i < 2; ++i) { mbar_init(&qfull[i], 1); mbar_init(&qempty[i], 1); mbar_init(&sched_full[i], 1); mbar_init(&order_bar[i], 1); kcount[i] = 0u; }
    }
    __syncthreads();

    auto shared_nkv = [&](int pair) { return pair + 1; };

    if (wg == 0) {
        // PRODUCER: one warp (G5; v49-v62 ran the loop in all 4 warps and 3 of them only spun on the barriers, ~25% of
        // all stall samples, ncu 2026-10-13). setmaxnreg is warpgroup-wide, so the other warps leave after it.
        reg_dec_producer();
        if (lane_wg >= 32) return;
        const int lane = lane_wg;
        int g = 0, j = 0, ju = 0, stage = 0, ph = 0;
        int uidx = blockIdx.x;
        // F3: the consumers read the NEXT unit before their epilogue (FA3: get_next_work is called "before the epilogue
        // so that the next tile is ready to go", flash_fwd_kernel_sm90.h:435), so it is published as soon as this
        // unit's loads are issued (FA3 claims it inside mainloop.load, prefetch_next_work, tile_scheduler.hpp:117-140).
        // G5 published a unit at the start of its Q load and the consumers read it after their epilogue: a shared load
        // and two shuffles in a row before the next unit's K/V wait (ncu 2026-10-16: unit start 0.148M cycles vs FA3's
        // 0.097M). Slot reuse: slot k & 1 is rewritten with unit k at the end of unit k-1's loads, after the Q load of
        // unit k-1, which waited for both groups to finish unit k-2's QK; they read unit k-2's number at the end of
        // unit k-3, earlier.
        if (lane == 0) { sched_unit[0] = (uidx < total_units) ? uidx : -1; mbar_arrive(&sched_full[0]); }
        while (uidx < total_units) {
            int4 un{};
            if constexpr (CHATS) un = units[uidx];
            #pragma unroll 1
            for (int half = 0; half < 2; ++half) {
            int b, hq, hkv, pair, t0 = 0;
            if constexpr (CHATS) {   // A: segment [t0, t0+L) of row b; pairs cpl and nP-1-cpl
                b = un.x; t0 = un.y; hq = un.w & 0xff; hkv = hq / G;
                const int cpl = un.w >> 8, nP = (un.z + 2 * BLOCK_M - 1) / (2 * BLOCK_M);
                pair = half ? (nP - 1 - cpl) : cpl;
                if (half == 1 && pair == cpl) break;   // middle pair of an odd count: 1-pair unit
            } else {
                if (half == 1) break;                  // G5: a dense unit is one pair
                decode_u(uidx, b, hq, hkv, pair);
            }
            const int nkv = shared_nkv(pair);
            const int kv_row_base = (b * Hkv + hkv) * S + t0;   // t0: A only (else 0)
            auto load_kv = [&](const int n) {          // tile n's K and V into the next stage (waits for the slot)
                if (g >= NSTAGES) mbar_wait(&empty[stage], ph ^ 1);
                if (lane_wg == 0) {
                    int row = kv_row_base + n * BLOCK_N;
                    __nv_bfloat16* Kd = &sK[stage * TILE_K];
                    __nv_bfloat16* Vd = &sV[stage * TILE_K];
                    mbar_expect_tx(&full[stage], 2 * TMA_BYTES);
                    tma_load_2d<HINT_LD>(__cvta_generic_to_shared(&Kd[0]),      &tmapKsw, 0,  row, &full[stage], L2_EVICT_LAST);
                    tma_load_2d<HINT_LD>(__cvta_generic_to_shared(&Kd[SWZ_BLK]),&tmapKsw, 64, row, &full[stage], L2_EVICT_LAST);
                    tma_load_2d<HINT_LD>(__cvta_generic_to_shared(&Vd[0]),      &tmapVsw, 0,  row, &full[stage], L2_EVICT_LAST);
                    tma_load_2d<HINT_LD>(__cvta_generic_to_shared(&Vd[SWZ_BLK]),&tmapVsw, 64, row, &full[stage], L2_EVICT_LAST);
                }
                ++g;
                if (++stage == NSTAGES) { stage = 0; ph ^= 1; }
            };
            int n0 = 0;
#if KV0_FIRST
            if (nkv > 0) { load_kv(0); n0 = 1; }       // KV0_FIRST: the first tile before the Q buffer (FA3's order)
#endif
            if (j > 0) { mbar_wait(&qempty[0], (j - 1) & 1); mbar_wait(&qempty[1], (j - 1) & 1); }
            if (lane_wg == 0) {
                #pragma unroll
                for (int c = 0; c < 2; ++c) {
                    int q_tile = 2 * pair + c;
                    int q_row = (b * Hq + hq) * S + t0 + q_tile * BLOCK_M;   // t0: A only (else 0)
                    __nv_bfloat16* Qd = &Qt[c * TILE_Q];
                    mbar_expect_tx(&qfull[c], 2 * Q_TMA_BYTES);
                    tma_load_2d<HINT_LD>(__cvta_generic_to_shared(&Qd[0]),         &tmapQsw, 0,  q_row, &qfull[c], L2_EVICT_FIRST);
                    tma_load_2d<HINT_LD>(__cvta_generic_to_shared(&Qd[SWZ_BLK_Q]), &tmapQsw, 64, q_row, &qfull[c], L2_EVICT_FIRST);
                }
            }
            for (int n = n0; n < nkv; ++n) load_kv(n);
            ++j;
            }
            ++ju;
            int claim = 0;
            if (lane == 0) {
                claim = atomicAdd(&g_v62_next_unit, 1);
                if (claim == total_units - 1) atomicExch(&g_v62_next_unit, 0);   // the launch's last claim: reset
            }
            uidx = __shfl_sync(0xffffffffu, claim, 0) + NUM_SMS;
            if (lane == 0) { sched_unit[ju & 1] = (uidx < total_units) ? uidx : -1; mbar_arrive(&sched_full[ju & 1]); }
        }
        return;
    }
    {
        reg_inc_consumer();
        // Consumer index, broadcast from lane 0 so ptxas knows it is the same in the whole warp and keeps what is
        // built from it (this group's Q buffer, the wgmma descriptors) in uniform registers, as CUTLASS's
        // canonical_warp_group_idx / FA3's warp_uniform (hopper/utils.h:621-623) do. Taken from threadIdx
        // directly (v62 as first written), the descriptors lived in vector registers and every HGMMA needed R2UR
        // moves (ncu 2026-10-13, dense B8 18/6: R2UR 40.2M executed vs FA3's 0.7M).
        const int c = __shfl_sync(0xffffffffu, (int)(threadIdx.x / WG), 0) - 1;
        const int warp = lane_wg / 32, lane = lane_wg % 32;
        const int groupID = lane / 4, tig = lane % 4;
        const int R0 = 16 * warp + groupID, R1 = R0 + 8;
        __nv_bfloat16* myQt = &Qt[c * TILE_Q];
        int g = 0, j = 0, stage = 0, ph = 0;
        if (c == 1) pp_arrive1();

        // G5/F3: unit ju is published in slot ju & 1 (sched_full[ju & 1], phase (ju >> 1) & 1). F2: the number is
        // broadcast from lane 0, so ptxas knows the unit is the same in the whole warp and so is everything built from
        // it (b, hq, the pair, the trip counts, the K/V stage, the wgmma descriptors): they then live in uniform
        // registers. Read from shared memory by every thread (G5 as first written), they did not, and every HGMMA needed
        // its descriptor moved over first (ncu 2026-10-16: 32 R2UR + 30 IMAD.MOV/VIADD per key tile in the issue phase;
        // FA3 builds them with UMOV/UIADD3). Same idea as G2's `c` above.
        auto fetch_unit = [&](int k) {
            mbar_wait(&sched_full[k & 1], (k >> 1) & 1);
            asm volatile("" ::: "memory");                   // (mbar_wait has no memory clobber: read after it)
            return __shfl_sync(0xffffffffu, *reinterpret_cast<volatile int*>(&sched_unit[k & 1]), 0);
        };
        int next_uidx = fetch_unit(0);
#if K6_PREF
        int4 next_un{};                                 // K6: the next unit's descriptor, loaded ahead
        if constexpr (CHATS) { if (next_uidx >= 0) next_un = units[next_uidx]; }
#endif
        for (int ju = 0; ; ++ju) {
            const int uidx = next_uidx;
            if (uidx < 0) break;
            int4 un{};
            if constexpr (CHATS) {
#if K6_PREF
                un = next_un;                           // K6: loaded during the previous unit's last P@V
#else
                un = units[uidx];
#endif
                un.x = __shfl_sync(0xffffffffu, un.x, 0); un.y = __shfl_sync(0xffffffffu, un.y, 0);
                un.z = __shfl_sync(0xffffffffu, un.z, 0); un.w = __shfl_sync(0xffffffffu, un.w, 0);
            }
            #pragma unroll 1
            for (int half = 0; half < 2; ++half) {
            int b, hq, hkv, pair, t0 = 0, L = 0;
            bool last_pair = true;                     // F3: the next unit is fetched before this pair's epilogue
            if constexpr (CHATS) {   // A: the same decode as the producer (same unit, same pairs)
                b = un.x; t0 = un.y; L = un.z; hq = un.w & 0xff; hkv = hq / G;
                const int cpl = un.w >> 8, nP = (L + 2 * BLOCK_M - 1) / (2 * BLOCK_M);
                pair = half ? (nP - 1 - cpl) : cpl;
                if (half == 1 && pair == cpl) break;
                last_pair = (half == 1) || (nP - 1 - cpl == cpl);
            } else {
                if (half == 1) break;                  // G5: a dense unit is one pair
                decode_u(uidx, b, hq, hkv, pair);
            }
            const int q_tile = 2 * pair + c;
            // gR0/gR1: rows of this thread, relative to the row start (dense) or the segment start (A)
            const int gR0 = q_tile * BLOCK_M + R0, gR1 = q_tile * BLOCK_M + R1;
            const long long q_base = ((long long)(b * Hq + hq) * S + t0 + (long long)q_tile * BLOCK_M) * HEAD_DIM;
            const int nkv_shared = shared_nkv(pair);
            int my_nkv = (q_tile * BLOCK_M + BLOCK_M - 1) / BLOCK_N + 1;
            // A: a q-tile that starts at or past L holds no row of the segment (the second tile of a
            // last pair when L % 128 is in 1..64): no active tile, only the inactive-tail handshake.
            if constexpr (CHATS) { if (q_tile * BLOCK_M >= L) my_nkv = 0; }

            mbar_wait(&qfull[c], j & 1);
#if K4_EPI
            if (my_nkv == 0 && lane_wg == 0) st_release_shared(&kcount[c], (unsigned)(j + 1));   // K4: inactive: never reads K
#endif

            float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
            float O_acc[ACC_PV];
            #pragma unroll
            for (int i = 0; i < ACC_PV; ++i) O_acc[i] = 0.0f;

            // ---- F1 (2026-10-16): FA3's loop order (IntraWGOverlap, hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp:1170-1207).
            // Per key tile n >= 1: start QK(n) and P@V(n-1) together, wait for QK(n) only, run softmax(n) WHILE P@V(n-1) is
            // still on the tensor cores, then wait for P@V(n-1), rescale O by softmax(n)'s factor and pack P(n). v58-v62
            // (until 2026-10-16) waited for both GEMMs before each softmax: QK(n+1) overlapped only the wait for P@V(n), and
            // a group's softmax ran with nothing of its own on the tensor cores (ncu 2026-10-13, B8 18/6: tensor cores busy
            // 64% vs FA3's 83%). The math and its order are unchanged: O(n) = O(n-1) * cc(n) + P(n) . V(n), tile 0 first.
            // Ping-pong baton: now around the GEMM issue, as FA3 (warp_scheduler_barrier_sync / _arrive, :1176 / :1183): one
            // group issues its two GEMMs, then the other (v45-v62 held it around the exp). One handshake per active tile (the
            // prologue counts as one), so the inactive tail below keeps its nkv_shared - my_nkv handshakes.
            // Registers: P(n-1) (pa, 32) stays live while its GEMM runs, on top of S (64) and O (64); FA3 does the same with
            // 240 registers per thread, ours has 232 (ptxas -v: 0 spills).
            // SASS (2026-10-16): wait-for-QK, then the mask and row max (~260 instructions) overlap P@V(n-1); ptxas places
            // the wait-for-P@V before the exps (it reuses P(n-1)'s registers for softmax temporaries; empty-asm register
            // fences before / after the wait did not move it: they act in NVVM, ptxas allocates on its own). FA3's SASS has
            // the same shape: 13 of its 64 exps before its wait-for-P@V, 51 after (ncu 2026-10-13); the gain is the
            // GEMM-issue baton (the other group's two GEMMs are queued while this group runs its softmax).
            float S_acc[ACC_QK];
            // K1 (2026-10-17): P as bf16 pairs in TWO register sets, P(n) in pa0 if n is even, else pa1. P@V(n-1) reads one
            // set while softmax(n) packs P(n) into the other, so packing no longer waits for P@V(n-1). With ONE set (v62-F3c)
            // the pack had to follow the wait for P@V(n-1), and ptxas put that wait right after the first row's max: the
            // other row's max and all 64 exps ran after it, not under P@V(n-1) (SASS 2026-10-17, in every softmax form tried;
            // the PTX had them all before the wait; registers were not the limit: 184 of 232 used in the loop). +32 registers.
#if K1_DB
            uint32_t pa0[32], pa1[32];
#else
            uint32_t pa0[32];
            uint32_t (&pa1)[32] = pa0;                  // K1_DB 0: one set (P(n) is packed after the wait for P@V(n-1))
#endif
            float cc0 = 1.0f, cc1 = 1.0f;               // rescale factors of O for the tile just softmaxed
            float this_bb0 = 0.0f, this_bb1 = 0.0f;     // K1: -max * scale2 of the tile being softmaxed (part A -> part B)
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
            auto issue_pv = [&](__nv_bfloat16* Vt_, auto buf_tag) {   // buf_tag: parity of the tile whose P is multiplied
                wgmma_fence();                          // after the O rescale and the P packing: the GEMM reads both
                static_assert(1024 % 8 == 0, "desc_add_lo needs 8-element (16 B) aligned offsets");
                const uint64_t vd0 = make_pv_desc(Vt_);
                uint64_t vd[8];
                #pragma unroll
                for (int kc = 0; kc < N_CHUNKS; ++kc)
                    vd[kc] = desc_add_lo(vd0, kc * 1024);   // 16 keys = 16 rows of 128 B = 1024 elements
                if constexpr (decltype(buf_tag)::value == 0) wgmma_pv128_x8(O_acc, pa0, vd);
                else                                         wgmma_pv128_x8(O_acc, pa1, vd);
                wgmma_commit();
            };
            // Softmax of key tile nt from its scores in S_acc: causal mask (MASK instance only), row max, the rescale factors
            // cc0/cc1, P = exp2(S * scale2 - max) left in S_acc, running row sums l0/l1. O is NOT touched here: the previous
            // tile's P@V may still be accumulating into it.
            // K1 (2026-10-17): as FA3 (hopper/softmax.h max_get_scale / online_softmax, mainloop_fwd_sm90_tma_gmma_ws.hpp
            // 1209-1227). (1) The mask is a compile-time instance, used only for the diagonal tile, which is peeled out of
            // the loop below: until K1 a `diag` branch sat inside every tile's softmax, so the loop held both versions and
            // ptxas could not schedule the max and the exps as one block. (2) One running max and one running sum per
            // accumulator column parity and row (4 + 4 registers) instead of 8 partial maxes and 8 partial sums per row (32
            // registers). The SASS before K1 (ncu 2026-10-17, dense B8 18/6, equal clock): after the wait for P@V(n-1) ours ran
            // 328 instructions per tile (all 64 exps, 19 of the row max's FMNMX) vs FA3's 258 (FA3 has the whole max, 33 FFMA
            // and 13 exps before that wait): +0.198M of the forward's +0.288M cycles vs FA3.
            float sm[4];                                // K1: running row sums per accumulator column parity and row (part A + B)
#if SM3
            float rs0 = 0.0f;                           // SM3: row 0's running sum, l0 * cc0 + its exps in column order
#endif
            auto softmax_tile = [&](const int nt, auto mask_tag) {   // part A: mask, max, cc, exps of regs [0, K1_SPLIT)
                constexpr bool MASK = decltype(mask_tag)::value;
                if constexpr (MASK) {
#if SM3
                    // SM3 (45.52): FA3's mask order (hopper/mask.h, causal): row by row, each element's column offset (a
                    // constant) against ONE per-row limit; the same condition as below (nt * BLOCK_N + keycol > gR). The
                    // mixed-row order made ptxas pack the 64 conditions into a bitmask in SM3 (LOP3 per element, packed ncu).
                    const int lim0 = gR0 - nt * BLOCK_N - 2 * tig, lim1 = gR1 - nt * BLOCK_N - 2 * tig;
                    #pragma unroll
                    for (int k = 0; k < 32; ++k) { if (8 * (k / 2) + (k % 2) > lim0) S_acc[4 * (k / 2) + (k % 2)] = -INFINITY; }
                    #pragma unroll
                    for (int k = 0; k < 32; ++k) { if (8 * (k / 2) + (k % 2) > lim1) S_acc[4 * (k / 2) + (k % 2) + 2] = -INFINITY; }
#else
                    #pragma unroll
                    for (int reg = 0; reg < ACC_QK; ++reg) {
                        const int keycol = 8 * (reg / 4) + 2 * tig + (reg & 1);
                        if (nt * BLOCK_N + keycol > ((reg % 4 < 2) ? gR0 : gR1)) S_acc[reg] = -INFINITY;
                    }
#endif
                }
#if SM3
                static_assert(!K1_DB && K1_SPLIT < 32, "SM3: one P set, part A = row 0's first K1_SPLIT exps");
                float nm0 = m0, nm1 = m1;               // FA3 thread_reduce_: one chain per row, from the previous max
                #pragma unroll
                for (int j = 0; j < ACC_QK / 4; ++j) {
                    #pragma unroll
                    for (int e = 0; e < 2; ++e) { nm0 = fmaxf(nm0, S_acc[4 * j + e]); nm1 = fmaxf(nm1, S_acc[4 * j + 2 + e]); }
                }
                nm0 = quad_max(nm0); nm1 = quad_max(nm1);
#else
                float mx[4];
                #pragma unroll
                for (int reg = 0; reg < ACC_QK; ++reg) { const int s = reg % 4; mx[s] = (reg < 4) ? S_acc[reg] : fmaxf(mx[s], S_acc[reg]); }
                const float mt0 = quad_max(fmaxf(mx[0], mx[1])), mt1 = quad_max(fmaxf(mx[2], mx[3]));
                const float nm0 = fmaxf(m0, mt0), nm1 = fmaxf(m1, mt1);
#endif
                // No -inf guard is needed: every row (dense, or A's segment-relative rows) sees key 0
                // of its own row or segment in the first tile, so nm0/nm1 are finite from tile 0 on.
                cc0 = ex2_approx((m0 - nm0) * scale2); cc1 = ex2_approx((m1 - nm1) * scale2);
                this_bb0 = -nm0 * scale2; this_bb1 = -nm1 * scale2;
                const float bb0 = this_bb0, bb1 = this_bb1;
                m0 = nm0; m1 = nm1;
#if SM3
                (void)bb1;
                rs0 = l0 * cc0;                         // FA3 max_get_scale: row_sum *= scores_scale
                #pragma unroll
                for (int k = 0; k < K1_SPLIT; ++k) {    // FA3 scale_apply_exp2, row 0 first; then reduce_sum's chain of row 0
                    const int reg = 4 * (k / 2) + (k % 2);
                    S_acc[reg] = ex2_approx(__fmaf_rn(S_acc[reg], scale2, bb0));
                }
                #pragma unroll
                for (int k = 0; k < K1_SPLIT; ++k) rs0 += S_acc[4 * (k / 2) + (k % 2)];
#else
                #pragma unroll
                for (int reg = 0; reg < ACC_QK; ++reg) {
                    if (reg >= K1_SPLIT) break;
                    const int s = reg % 4;
                    const float p = ex2_approx(__fmaf_rn(S_acc[reg], scale2, (s < 2) ? bb0 : bb1));
                    S_acc[reg] = p;
                    sm[s] = (reg < 4) ? p : sm[s] + p;
                }
#endif
            };
            auto softmax_rest = [&](const float after) {   // part B: exps of regs [K1_SPLIT, 64) and the row sums
                // `after` is 0, but read from a register only valid after the wait for P@V(n-1): it keeps ptxas from moving part
                // B in front of that wait (it did, in one of the two unrolled loop copies, SASS 2026-10-17). 0 elsewhere.
                const float bb0 = this_bb0 + after, bb1 = this_bb1 + after;
#if SM3
                #pragma unroll
                for (int k = K1_SPLIT; k < 32; ++k) {   // the rest of row 0, then row 1 (FA3 scale_apply_exp2)
                    const int reg = 4 * (k / 2) + (k % 2);
                    S_acc[reg] = ex2_approx(__fmaf_rn(S_acc[reg], scale2, bb0));
                }
                #pragma unroll
                for (int k = 0; k < 32; ++k) {
                    const int reg = 4 * (k / 2) + (k % 2) + 2;
                    S_acc[reg] = ex2_approx(__fmaf_rn(S_acc[reg], scale2, bb1));
                }
                float rs1 = l1 * cc1;                   // FA3 reduce_sum: one chain per row, columns in order (rows interleaved)
                #pragma unroll
                for (int k = 0; k < 32; ++k) {
                    if (k >= K1_SPLIT) rs0 += S_acc[4 * (k / 2) + (k % 2)];
                    rs1 += S_acc[4 * (k / 2) + (k % 2) + 2];
                }
                l0 = rs0; l1 = rs1;                     // per-lane partials, reduced once in the epilogue (as before)
#else
                #pragma unroll
                for (int reg = 0; reg < ACC_QK; ++reg) {
                    if (reg < K1_SPLIT) continue;
                    const int s = reg % 4;
                    const float p = ex2_approx(__fmaf_rn(S_acc[reg], scale2, (s < 2) ? bb0 : bb1));
                    S_acc[reg] = p;
                    sm[s] = (reg < 4) ? p : sm[s] + p;
                }
                l0 = l0 * cc0 + (sm[0] + sm[1]);   // v57: per-lane partial, no SHFL in the section
                l1 = l1 * cc1 + (sm[2] + sm[3]);   // v57: reduced once in the epilogue instead
#endif
            };
            // After the previous P@V has finished: P (fp32 in S_acc) -> bf16 pairs in pa, then O rescaled for the new max
            // (FA3's order: convert_type_out, then rescale_o). F2 (2026-10-16): O is rescaled on EVERY tile, as FA3; until F2
            // a warp vote skipped the 64 FMUL when no row max had changed, and its branch cut the softmax into blocks ptxas
            // could not mix. K1: the prologue packs only (O is still 0 there, its rescale was 64 FMUL per unit for nothing).
            auto pack_p = [&](auto buf_tag) {          // buf_tag: parity of the tile just softmaxed
                uint32_t* pa = (decltype(buf_tag)::value == 0) ? pa0 : pa1;
                #pragma unroll
                for (int kc = 0; kc < N_CHUNKS; ++kc) {
                    int r = 8 * kc;
                    pa[4*kc+0] = pack_bf16(S_acc[r + 0], S_acc[r + 1]);
                    pa[4*kc+1] = pack_bf16(S_acc[r + 2], S_acc[r + 3]);
                    pa[4*kc+2] = pack_bf16(S_acc[r + 4], S_acc[r + 5]);
                    pa[4*kc+3] = pack_bf16(S_acc[r + 6], S_acc[r + 7]);
                }
            };
            auto rescale_o = [&]() {
                #pragma unroll
                for (int reg = 0; reg < ACC_PV; ++reg) O_acc[reg] *= (reg % 4 < 2) ? cc0 : cc1;
            };
            // F2: the K/V stage index broadcast from lane 0 (here and at the top of every tile), so ptxas knows it is
            // warp-uniform and builds the K and V descriptors from it in uniform registers. As a loop-carried counter it
            // stayed in a vector register and every K/V descriptor was built there and moved over with R2UR before its
            // HGMMA (SASS 2026-10-16), even after the unit itself was broadcast.
            stage = __shfl_sync(0xffffffffu, stage, 0);
            if (nkv_shared > 0) mbar_wait(&full[stage], ph);   // tile 0's K/V (the inactive tail is paced by it too)
            if (my_nkv > 0) {                           // prologue: QK(0), softmax(0), P(0); O is still 0
                if (c == 0) pp_sync1(); else pp_sync2();
                issue_qk(&sK[stage * TILE_K]);
                if (c == 0) pp_arrive2(); else pp_arrive1();
                wgmma_wait0();
#if K4_EPI
                if (my_nkv == 1 && lane_wg == 0) st_release_shared(&kcount[c], (unsigned)(j + 1));   // K4: last QK (K(0)) done
#endif
                if (my_nkv == 1) softmax_tile(0, std::true_type{});   // tile 0 is the diagonal one only when it is the last
                else             softmax_tile(0, std::false_type{});
                softmax_rest(0.0f);
                pack_p(std::integral_constant<int, 0>{});   // P(0) -> pa0
            }
            // ACTIVE range only. Splitting the loop keeps the wgmma group bookkeeping UNIFORM inside the
            // body -- the first attempt kept `if (n < my_nkv)` around the wgmma with wait1 on one path and
            // wait0 on the other, and ptxas answered with C7518: it could not track the pending-group count
            // across the divergence and gave EVERY HGMMA its own ARRIVE/DEPBAR pair (24 groups per iteration
            // instead of 8-then-1), serialising the tensor pipe completely. CUTLASS splits for this reason.
            // K1 (2026-10-17): ptxas keeps the wait for P@V(n-1) (WARPGROUP.DEPBAR.LE gsb0, 0x0) behind an mbarrier arrive
            // that comes before it in the program. Without one, it put the wait right after the first row's max and ran the
            // rest of the softmax behind it, in every form of the softmax tried (SASS 2026-10-17; the PTX had it all in front).
            // FA3 gets the same effect from its K release (pipeline_k.consumer_release, mainloop :1185): ptxas schedules that
            // arrive late and the wait after it (FA3 SASS: 13 exps and the whole max before its wait). Here the arrive is on a
            // barrier nobody waits on (count 1: every arrive completes a phase, it never blocks anything), and WHICH of the two
            // is chosen depends on `v`, a value computed at the end of the softmax: the arrive, and so the wait, can only come
            // after it. Both choices are harmless; `v` only creates the dependency. One predicated arrive per warp per tile.
            auto order_after = [&](uint32_t v) {
                if (lane_wg == 0) mbar_arrive(&order_bar[(v == 0xffffffffu) ? 1 : 0]);
            };
            // One key tile n >= 1 (FA3's fwd_step, mainloop_fwd_sm90_tma_gmma_ws.hpp:1170-1207).
            // K1: only group 0 waits for the K/V of tile n (FA3 :1175 / :1180, `!UseSchedulerBarrier || warp_group_idx == 0`):
            // group 1 issues tile n only after the baton from group 0, which group 0 passes after its own wait for the same
            // stage and its two GEMMs, so the data has landed. The prologue and the inactive tail keep their waits in both
            // groups (a group-1-only tail has no group 0 ahead of it on that tile).
            auto step = [&](const int n, auto mask_tag, auto buf_tag) {   // buf_tag: n & 1 (compile time)
                constexpr int B = decltype(buf_tag)::value;
                stage = __shfl_sync(0xffffffffu, stage, 0);   // F2: keep the stage (and the descriptors) uniform
                const int prev = stage;                 // stage of tile n-1: its V is read by P@V(n-1)
                int nstage = stage + 1, nph = ph;
                if (nstage == NSTAGES) { nstage = 0; nph ^= 1; }
                if (c == 0) mbar_wait(&full[nstage], nph);   // K/V of tile n have landed
                if (c == 0) pp_sync1(); else pp_sync2();
                issue_qk(&sK[nstage * TILE_K]);         // group: QK(n)
                issue_pv(&sV[prev * TILE_K], std::integral_constant<int, 1 - B>{});   // group: P@V(n-1), from the other set
                if (c == 0) pp_arrive2(); else pp_arrive1();
                wgmma_wait1();                          // QK(n) done, P@V(n-1) still running
#if K4_EPI
                if constexpr (decltype(mask_tag)::value) { if (lane_wg == 0) st_release_shared(&kcount[c], (unsigned)(j + 1)); }   // K4
#endif
                softmax_tile(n, mask_tag);              // ... while P@V(n-1) runs
#if K1_DB
#if K1_SPLIT >= 64
                softmax_rest(0.0f);                     // (nothing left: only the row sums)
                pack_p(buf_tag);                        // P(n) into this tile's set, also while P@V(n-1) runs
                {
                    uint32_t* pp = (B == 0) ? pa0 : pa1;
                    order_after(__float_as_uint(l0 + l1) ^ pp[0] ^ pp[31]);   // after the sums and the packing
                }
                wgmma_wait0();                          // P@V(n-1) done
                if (lane_wg == 0) mbar_arrive(&empty[prev]);
                rescale_o();                            // the only work left behind the wait
#else
                order_after(__float_as_uint((sm[0] + sm[1]) + (sm[2] + sm[3])));   // after ALL of part A's exps (their sums)
                wgmma_wait0();                          // P@V(n-1) done
                if (lane_wg == 0) mbar_arrive(&empty[prev]);
                {
                    float zero;                         // O_acc[0] - O_acc[0]: 0, readable only after the wait (O is finite)
                    asm volatile("sub.f32 %0, %1, %1;" : "=f"(zero) : "f"(O_acc[0]));
                    softmax_rest(zero);                 // the other exps, mixed by ptxas with the rescale and the packs
                }
                rescale_o();
                pack_p(buf_tag);
#endif
#else
#if K1_SPLIT >= 64
                softmax_rest(0.0f);
                order_after(__float_as_uint(l0 + l1));  // after the exps and the sums
                wgmma_wait0();                          // P@V(n-1) done
                if (lane_wg == 0) mbar_arrive(&empty[prev]);
                pack_p(buf_tag);                        // one set: only after P@V(n-1) has read it
                rescale_o();
#else
                // K5 (2026-10-17): the split with ONE P register set. With the split, all of P(n) is packed after the wait
                // for P@V(n-1) anyway (part B), so the second set (32 registers, and the 2x unrolled loop) buys nothing; it was
                // what pushed K4h / K4hb into 76-112 B of spills at every unit's start and end (K4hb packed +17% vs K4b).
#if SM3
                order_after(__float_as_uint(rs0));      // SM3: after row 0's running sum, i.e. after ALL of part A's exps
#else
                order_after(__float_as_uint((sm[0] + sm[1]) + (sm[2] + sm[3])));   // after ALL of part A's exps (their sums)
#endif
                wgmma_wait0();                          // P@V(n-1) done: it has read pa
                if (lane_wg == 0) mbar_arrive(&empty[prev]);
                {
                    float zero;                         // O_acc[0] - O_acc[0]: 0, readable only after the wait (O is finite)
                    asm volatile("sub.f32 %0, %1, %1;" : "=f"(zero) : "f"(O_acc[0]));
                    softmax_rest(zero);                 // the other exps, mixed by ptxas with the rescale and the packs
                }
                rescale_o();
                pack_p(buf_tag);
#endif
#endif
                ++g;
                stage = nstage; ph = nph;
            };
            // K1: the causal diagonal is always this group's LAST active tile: q-tile 2*pair + c holds rows 128*pair + 64*c +
            // 0..63 (of the row, or of A's segment), my_nkv - 1 = pair, so tile pair is the only one with keys past a row.
            // FA3 runs its masked tiles first (it walks the keys backwards); ours walks forwards, so the diagonal comes last.
            // The register set alternates with n, so the body is unrolled by two (n odd, then n even) to keep it static.
            using I0 = std::integral_constant<int, 0>; using I1 = std::integral_constant<int, 1>;
#if K1_DB
            int n = 1;
            #pragma unroll 1
            for (; n + 1 < my_nkv - 1; n += 2) {       // two mask-free tiles per trip
                step(n,     std::false_type{}, I1{});
                step(n + 1, std::false_type{}, I0{});
            }
            if (n < my_nkv - 1) { step(n, std::false_type{}, I1{}); ++n; }   // one mask-free tile left (n odd)
            if (my_nkv > 1) {                           // the diagonal tile, n == my_nkv - 1
                if (n & 1) step(n, std::true_type{}, I1{});
                else       step(n, std::true_type{}, I0{});
            }
#else
            for (int n = 1; n < my_nkv - 1; ++n) step(n, std::false_type{}, I0{});   // mask-free body
            if (my_nkv > 1) step(my_nkv - 1, std::true_type{}, I0{});                // the diagonal tile
#endif
            // ---- v61: RELEASE Qt[c] as soon as its last reader, the last QK, is done, not after the epilogue: in v58
            // this arrive sat after the O stores and the LSE writes, so the Q pipeline was a full round trip at every
            // pair boundary (consumer finishes -> epilogue -> qempty -> producer issues the Q TMA -> consumer waits
            // qfull; ~11% of all samples in the v58 SASS). With F1 the last QK is done before the last P@V is issued.
            if (lane_wg == 0) mbar_arrive(&qempty[c]);
            int s_last = 0;                             // K4: stage of this group's last active tile
            if (my_nkv > 0) {                           // the last tile's P@V
                stage = __shfl_sync(0xffffffffu, stage, 0);
                s_last = stage;
                if ((my_nkv - 1) & 1) issue_pv(&sV[stage * TILE_K], std::integral_constant<int, 1>{});   // P(my_nkv - 1)
                else                  issue_pv(&sV[stage * TILE_K], std::integral_constant<int, 0>{});
                // F3: fetch the next unit while the last P@V runs (FA3: before the epilogue) -- only when this group has no
                // inactive tail (my_nkv == nkv_shared, always true dense)
                if (last_pair && my_nkv == nkv_shared) {
                    next_uidx = fetch_unit(ju + 1);
#if K6_PREF
                    if constexpr (CHATS) { if (next_uidx >= 0) next_un = units[next_uidx]; }   // K6: in flight during the epilogue
#endif
                }
                wgmma_wait0();
#if !K4_EPI
                if (lane_wg == 0) mbar_arrive(&empty[stage]);
#endif
                ++g;
                if (++stage == NSTAGES) { stage = 0; ph ^= 1; }
                if (my_nkv < nkv_shared) mbar_wait(&full[stage], ph);   // the tail's first tile (v62 waited it in the loop)
            }
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

            // F3: a group with an inactive tail (design A: its q-tile is past the segment's end, my_nkv == 0) fetches the
            // next unit only HERE, after the tail. The tail releases this unit's K/V stages, and the producer can publish the
            // next unit only after issuing all of this unit's loads, which for 4+ key tiles wait for those releases: the
            // first F3 build fetched before the tail and deadlocked (GPU 2026-10-16 16:57 IST, quick suite timeout;
            // reproduced by the CPU protocol model profiling_reports/fwd_deep_2026_10_16/protocol_sim.py: 118 of 300
            // random packed unit lists deadlock, 0 with this order).
            if (last_pair && my_nkv < nkv_shared) {
                next_uidx = fetch_unit(ju + 1);
#if K6_PREF
                if constexpr (CHATS) { if (next_uidx >= 0) next_un = units[next_uidx]; }
#endif
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
            // A: rows past L belong to the next segment (computed by its own unit): never stored.
            bool st0 = true, st1 = true;
            if constexpr (CHATS) { st0 = gR0 < L; st1 = gR1 < L; }
#if K2_EPI || K4_EPI
            // K2 (2026-10-17): O through shared memory, as FA3's epilogue (hopper/epilogue_fwd.hpp: stmatrix into smem_o).
            // Until K2 each thread stored its 64 values straight from the accumulators: 32 four-byte stores per thread, each
            // warp instruction 8 rows x 16 B; the unit end took 109k cycles vs FA3's ~32k (ncu 2026-10-17, dense B8 18/6,
            // equal clock; stalls: the store queue full, and waits until stored registers may be reused). Now: normalise +
            // convert as before, 8 stmatrix.x4 (32 matrices of 8 x 8 = this warp's 16 rows x 128 columns) into this group's
            // 16 KB block, then 8 x (16-byte shared load + 16-byte global store) per thread: a warp store writes 2 full rows
            // (512 B). Each warp reads back only its own 16 rows, so a __syncwarp is the only synchronisation. Rows are 256 B;
            // the 16-byte chunk index is XORed with (row & 7) so the 8 rows of one matrix fall on different banks.
            // A (packed): rows past L belong to the next segment and are not stored, as before.
            // K4 (2026-10-17): the 16 KB per group are this group's half of the LAST tile's K buffer, so the 3 K/V stages stay
            // (K2's own 32 KB needed 2 stages: +39% dense). That K tile is free once BOTH groups' last QK is done: each group
            // sets kcount[c] = pairs done after its last QK (an inactive group at once), and waits here for the other group's count
            // (as FA3's barrier_O guards smem_o, which aliases smem_v). The stage is released to the producer (empty arrive)
            // only after all 4 warps of the group have read their rows back: proxy fence (TMA will write it next), group barrier.
            // A group with no active tile (A: q-tile past the segment end) has nothing to store and releases its stages in the
            // inactive tail as before. Active groups never have a tail (my_nkv is 0 or nkv_shared), so the delayed release
            // cannot hold back a tile this group still waits for.
#if K4_EPI
            if (my_nkv > 0)
#endif
            {
                const int wrow = 16 * warp;             // this warp's first row in the q-tile
#if K4_EPI
                while (ld_acquire_shared(&kcount[1 - c]) < (unsigned)(j + 1)) { }   // the other group's last QK has read K(last)
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");      // its async (wgmma) reads before our writes
                const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(sK + s_last * TILE_K + c * TILE_Q + wrow * HEAD_DIM);
#else
                const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(sO + c * TILE_Q + wrow * HEAD_DIM);
#endif
                #pragma unroll
                for (int i = 0; i < 8; ++i) {           // matrices m = 0..3: row block m & 1 (R0 / R1), column block 2i + (m >> 1)
                    uint32_t v[4];
                    #pragma unroll
                    for (int m = 0; m < 4; ++m) {
                        const int rb = m & 1, cb = 2 * i + (m >> 1);
                        const int reg = 32 * (cb / 8) + 4 * (cb % 8) + 2 * rb;   // the accumulator pair of (row block, column block)
                        const float rcp = rb ? r1 : r0;
                        v[m] = pack_bf16(O_acc[reg] * rcp, O_acc[reg + 1] * rcp);
                    }
                    const int mi = lane >> 3, row = 8 * (mi & 1) + (lane & 7), cb = 2 * i + (mi >> 1);   // lane -> row of matrix mi
                    stmatrix_x4(sbase + row * 256 + ((cb ^ (row & 7)) << 4), v);
                }
                __syncwarp();
                #pragma unroll
                for (int k = 0; k < 8; ++k) {           // lanes 0-15: row 2k, lanes 16-31: row 2k + 1; 16 B each
                    const int row = 2 * k + (lane >> 4), ch = lane & 15;
                    const uint4 d = lds128(sbase + row * 256 + ((ch ^ (row & 7)) << 4));
                    if (!CHATS || q_tile * BLOCK_M + wrow + row < L)
                        stg128<HINT_O>(&O[q_base + (long long)(wrow + row) * HEAD_DIM + ch * 8], d);
                }
                __syncwarp();                           // all lanes have read their rows before the next unit's stmatrix
#if K4_EPI
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");   // generic accesses before the next TMA write
                asm volatile("bar.sync %0, 128;" :: "r"(3 + c) : "memory");    // the group's 4 warps are done with the stage
                if (lane_wg == 0) mbar_arrive(&empty[s_last]);
#endif
            }
#else
            #pragma unroll
            for (int reg = 0; reg < ACC_PV; reg += 2) {
                int half2 = reg / 32, rr = reg % 32;
                int bcol = rr / 4, s = rr % 4;          // s is 0 or 2 -- reg+1 is the s|1 sibling
                int row = (s < 2) ? R0 : R1;
                int head = half2 * 64 + 8 * bcol + 2 * tig;
                float rcp = (s < 2) ? r0 : r1;
                if (!CHATS || ((s < 2) ? st0 : st1))
                *reinterpret_cast<__nv_bfloat162*>(&O[q_base + (long long)row * HEAD_DIM + head]) =
                    __floats2bfloat162_rn(O_acc[reg] * rcp, O_acc[reg + 1] * rcp);
            }
#endif
            if (tig == 0) {
                if (st0) LSE[(long long)(b * Hq + hq) * S + t0 + gR0] = (m0 * scale2) * LN2 + logf(l0);
                if (st1) LSE[(long long)(b * Hq + hq) * S + t0 + gR1] = (m1 * scale2) * LN2 + logf(l1);
            }
            ++j;
        }
        }
    }
}

#else   // ---- non-Hopper: keep the translation unit compilable ----
// Same parameter types as the Hopper definition, __restrict__ included: the host pass sees only
// this stub, and nvcc's generated explicit specializations of a TEMPLATE kernel must match it
// exactly (the old non-template stub could differ in __restrict__; a template cannot).
constexpr int MODE_DENSE = 0, MODE_CHATS = 1;
template <int MODE>
__global__ void gqa_v62_sm90(const __grid_constant__ CUtensorMap, const __grid_constant__ CUtensorMap,
                        const __grid_constant__ CUtensorMap, const __nv_bfloat16* __restrict__,
                        __nv_bfloat16* __restrict__, float* __restrict__, int, int, int, int, float,
                        const int4* __restrict__, int) {}
#endif

size_t smem_bytes() { return T3_SMEM_V34; }   // sK+sV + Qt + mbarriers

} // namespace gqa62

// v2 (2026-10-10): chat_units/n_units (device work list of a GqaChatPlan, design A) select the
// per-segment variant; nullptr keeps the dense kernel exactly as before. (Design B's seg_start
// parameter was removed on 2026-10-15, see the kernel comment.)
void gqa_v62_causal_launch_d128(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* O, float* LSE,
    int B, int Hq, int Hkv, int S, float scale, cudaStream_t stream,
    const int32_t* chat_units, int n_units)
{
    int device = 0; cudaGetDevice(&device);
    cudaDeviceProp prop{}; cudaGetDeviceProperties(&prop, device);
    if (prop.major != 9) {
        fprintf(stderr, "gqa_v62_causal_launch_d128: Hopper only; sm_%d%d -- skipping.\n", prop.major, prop.minor);
        return;
    }
    const uint64_t kv_rows = (uint64_t)B * Hkv * S;
    CUtensorMap tmapKsw = gqa62::make_tma_kv128(K, kv_rows, gqa62::HEAD_DIM);
    CUtensorMap tmapVsw = gqa62::make_tma_kv128(V, kv_rows, gqa62::HEAD_DIM);
    CUtensorMap tmapQsw = gqa62::make_tma_q64(Q, (uint64_t)B * Hq * S, gqa62::HEAD_DIM);  // Q box [64][64] SW128
    // The dynamic-smem opt-in is per kernel, so each variant gets its own once-flag.
    static bool cfg_dense = false, cfg_chats = false;
    size_t smem_bytes = gqa62::smem_bytes();
    dim3 grid(gqa62::NUM_SMS); dim3 block(gqa62::NUM_THREADS);
    const int4* units = reinterpret_cast<const int4*>(chat_units);
    if (chat_units != nullptr) {
        if (!cfg_chats) { cudaFuncSetAttribute(gqa62::gqa_v62_sm90<gqa62::MODE_CHATS>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes); cfg_chats = true; }
        gqa62::gqa_v62_sm90<gqa62::MODE_CHATS><<<grid, block, smem_bytes, stream>>>(tmapKsw, tmapVsw, tmapQsw, Q, O, LSE, B, Hq, Hkv, S, scale, units, n_units);
    } else {
        if (!cfg_dense) { cudaFuncSetAttribute(gqa62::gqa_v62_sm90<gqa62::MODE_DENSE>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes); cfg_dense = true; }
        gqa62::gqa_v62_sm90<gqa62::MODE_DENSE><<<grid, block, smem_bytes, stream>>>(tmapKsw, tmapVsw, tmapQsw, Q, O, LSE, B, Hq, Hkv, S, scale, nullptr, 0);
    }
}



} // namespace cuda
} // namespace OwnTensor
