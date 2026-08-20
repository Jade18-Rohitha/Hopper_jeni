// Determinism probe for Vj1d (DETERMINISTIC fixed-order greduce dK/dV). Includes the baseline for
// shared helpers + Vj1, defines Vj1d on top, runs N times on identical inputs. Expect dK/dV
// BIT-IDENTICAL (max|Δ|=0) — matching cuDNN's one-writer reproducibility.
// Build: nvcc -gencode arch=compute_90a,code=sm_90a -Iinclude precision/vj1d_determinism_probe.cu -o vj1d_det -lcuda
// Run:   python precision/baseline_gqa.py B Hq Hkv ; ./vj1d_det B Hq Hkv [N]
#define VJ1_NO_MAIN
#include "../src/attention/GQA_bwd_baseline.cu"

// --- plain TMA store (from GQA_bwd.cu; baseline Vj1 uses reduce, not store) ---
__device__ __forceinline__ void tma_store_2d_v34(const void* tma_desc, const bf16* smem, uint32_t cx, uint32_t cy) {
    uint32_t src = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];\n"
        :: "l"((uint64_t)tma_desc), "r"(cx), "r"(cy), "r"(src) : "memory");
}

// --- fixed-order deterministic G-head reducer (from GQA_bwd.cu) ---
__global__ void gqa_dkdv_greduce(const bf16* __restrict__ partial, bf16* __restrict__ out,
                                 int Hq, int Hkv, int G, int S) {
    constexpr int Dd = 128, RPB = 8;
    int b = blockIdx.z, hkv = blockIdx.y, s0 = blockIdx.x * RPB;
    int t = threadIdx.x, row = t >> 4, dg = t & 15;          // 128 thr = 8 rows x 16 uint4-groups
    int s = s0 + row;
    float acc[8] = {0,0,0,0,0,0,0,0};
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
    *reinterpret_cast<uint4*>(out + (((long)(b * Hkv + hkv) * S + s) * Dd + dg * 8)) = *reinterpret_cast<const uint4*>(o);
}

template<int Br, int Bc, int D>
__global__ void __launch_bounds__(128, 1)
gqa_bwd_vj1d(
    const __grid_constant__ CUtensorMap tma_K,  const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_Q,  const __grid_constant__ CUtensorMap tma_dO,
    const __grid_constant__ CUtensorMap tma_dV_st, const __grid_constant__ CUtensorMap tma_dK_st,
    const __grid_constant__ CUtensorMap tma_dq_red,
    const float* __restrict__ LSE, const float* __restrict__ Drow,
    int B, int Hq, int Hkv, int G, int S, float scale) {
    const int kt = blockIdx.x, hq = blockIdx.y, b = blockIdx.z;
    const int hkv = hq / G;
    const int wtid = threadIdx.x;
    const int nQ = S / Br;
    const int k_row0 = kt * Bc;
    const float scale2 = scale * LOG2E_V29;

    __shared__ __align__(128)  bf16  sK_sw[Bc*D], sV_sw[Bc*D], sQ_sw[Br*D], sdO_sw[Br*D];
    __shared__ __align__(1024) bf16  sP[Br*64], sDS[Br*64];
    __shared__ __align__(1024) float sStage[2][Br*64];   // double-buffered: both dQ halves deferred
    __shared__ float sLSE[Br], sD[Br];
    __shared__ __align__(8) uint64_t mbar_kv, mbar_qo;

    if (wtid == 0) { mbar_init_v4(&mbar_kv, 1); mbar_init_v4(&mbar_qo, 1); }
    __syncthreads();

    const uint64_t descK    = make_desc_sw128_K (sK_sw);
    const uint64_t descV    = make_desc_sw128_K (sV_sw);
    const uint64_t descP    = make_desc_sw128_MN(sP);
    const uint64_t descKh0  = make_desc_sw128_MN(sK_sw);
    const uint64_t descKh1  = make_desc_sw128_MN(sK_sw + 4096);
    const uint64_t descDSmn = make_desc_sw128_MN(sDS);
    const uint64_t descDSk  = make_desc_sw128_K (sDS);

    const uint32_t kvRow = (uint32_t)((b*Hkv+hkv)*S + k_row0);
    if (wtid == 0) {
        mbar_expect_tx_v4(&mbar_kv, (uint32_t)(Bc*64*sizeof(bf16))*4);
        tma_load_2d_v4(&tma_K, sK_sw,       &mbar_kv, 0,  kvRow);
        tma_load_2d_v4(&tma_K, sK_sw+64*64, &mbar_kv, 64, kvRow);
        tma_load_2d_v4(&tma_V, sV_sw,       &mbar_kv, 0,  kvRow);
        tma_load_2d_v4(&tma_V, sV_sw+64*64, &mbar_kv, 64, kvRow);
    }
    mbar_wait_v4(&mbar_kv, 0);

    float dv[64], dk[64];
    zeroN<64>(dv); zeroN<64>(dk);

    // SINGLE-BUFFER SOFTWARE PREFETCH: issue tile it+1's Q/dO TMA into the SAME buffer right after the
    // half1 dK-gemm frees sQ/sdO, overlapping the dQ-reduce (which touches only sStage). No ring, no
    // smem cost, 2 CTAs preserved. Attacks long_scoreboard 2.51 (loads were issued-then-immediately-waited).
    const int nq_ = nQ - kt, nIter = nq_;   // Vj1 per-hq: ONE query head per CTA
    auto issue = [&](int it) {
        const int q = kt + it;
        const uint32_t qRow = (uint32_t)((b*Hq+hq)*S + q*Br);
        const long lb = (long)(b*Hq+hq)*S + (long)q*Br;
        if (wtid == 0) {
            mbar_expect_tx_v4(&mbar_qo, (uint32_t)(Br*D*sizeof(bf16))*2);
            tma_load_2d_v4(&tma_Q,  sQ_sw,        &mbar_qo, 0,  qRow);
            tma_load_2d_v4(&tma_Q,  sQ_sw+64*64,  &mbar_qo, 64, qRow);
            tma_load_2d_v4(&tma_dO, sdO_sw,       &mbar_qo, 0,  qRow);
            tma_load_2d_v4(&tma_dO, sdO_sw+64*64, &mbar_qo, 64, qRow);
        }
        if (wtid < Br) { sLSE[wtid] = LSE[lb+wtid]; sD[wtid] = Drow[lb+wtid]; }
    };
    uint32_t qopar = 0;
    issue(0);                                          // prologue: load tile 0

    for (int it = 0; it < nIter; it++) {
        const int q = kt + it;
        const uint32_t qRow = (uint32_t)((b*Hq+hq)*S + q*Br);
        mbar_wait_v4(&mbar_qo, qopar); qopar ^= 1;
        // NOTE: no post-load barrier — sQ visibility is guaranteed by mbar_wait (TMA completion), and
        // sLSE/sD were written+published by the PREVIOUS tile's store-barrier (prefetch). One barrier saved.

        // S = Q@Kᵀ and dP = dO@Vᵀ ISSUED TOGETHER (overlap on the tensor pipe), waited once — instead of
        // S(wait)→fused_p→dP(wait) which serialized the two GEMMs. Attacks wait 1.28 + 44%-idle pipe.
        float acc[32];   zeroN<32>(acc);
        float dPacc[32]; zeroN<32>(dPacc);
        run_gemm_n64_sw2_hoB_issue(acc,   sQ_sw,  descK);   // S (group A)
        run_gemm_n64_sw2_hoB_issue(dPacc, sdO_sw, descV);   // dP (group B)
        wgmma_wait1();                       // wait S only; dP GEMM keeps running on the tensor pipe
        fence_operandN<32>(acc);
        if (q == kt) fused_p_stsm<Bc,true >(acc, sP, sLSE, wtid, q*Br, k_row0, scale2);
        else         fused_p_stsm<Bc,false>(acc, sP, sLSE, wtid, 0, 0, scale2);  // softmax OVERLAPS dP GEMM
        wgmma_wait0();                       // now wait dP
        fence_operandN<32>(dPacc);
        __syncthreads();

        run_gemm_dVdK_half_te_issue_hoA(dv,    descP, sdO_sw + 0);
        run_gemm_dVdK_half_te_issue_hoA(dv+32, descP, sdO_sw + 4096);
        fuse_dS_ldstsm<Bc>(sP, dPacc, sD, sDS, wtid);
        __syncthreads();

        // dQ split: half0 & half1 kept SEPARATE — half0's committed reduce overlaps half1's GEMM, and the
        // prefetch overlaps half1's store+reduce. (Combining into one wait0 lost that overlap → slower.)
        {   float dq[32]; zeroN<32>(dq);
            run_gemm_dKdQ_te_issue_ho(dk, dq, descDSmn, descDSk, descKh0, sQ_sw + 0);
            if (wtid == 0) tma_bulk_wait0_v43();        // drain PREV reduces — moved BEFORE the wgmma wait
            run_gemm_dVdKdQ_te_wait(dv, dk, dq);        //   so thread 0's drain overlaps the dK/dQ GEMM
            __syncthreads();
            store_acc_sw128_f32(dq, sStage[0], wtid, scale);
            __syncthreads(); fence_proxy_async_shared();
            if (wtid == 0) {
                tma_reduce_add_2d_v43(&tma_dq_red, sStage[0],       0,  qRow);
                tma_reduce_add_2d_v43(&tma_dq_red, sStage[0]+64*32, 32, qRow);
                tma_store_commit_v34();   // NO wait — deferred (double-buffered sStage)
            }
        }
        {   float dq[32]; zeroN<32>(dq);
            run_gemm_dKdQ_te_issue_ho(dk+32, dq, descDSmn, descDSk, descKh1, sQ_sw + 4096);
            run_gemm_dVdKdQ_te_wait(dv, dk+32, dq);
            __syncthreads();                            // all threads done reading sQ/sdO
            if (it + 1 < nIter) issue(it + 1);          // prefetch next tile into same buffer (overlap)
            store_acc_sw128_f32(dq, sStage[1], wtid, scale);
            __syncthreads(); fence_proxy_async_shared();
            if (wtid == 0) {
                tma_reduce_add_2d_v43(&tma_dq_red, sStage[1],       64, qRow);
                tma_reduce_add_2d_v43(&tma_dq_red, sStage[1]+64*32, 96, qRow);
                tma_store_commit_v34();   // NO wait0 — deferred to next tile's half0
            }
        }
    }
    if (wtid == 0) tma_bulk_wait0_v43();   // drain the last tile's deferred half1 reduce before epilogue
    __syncthreads();
    // epilogue: stage + TMA-store dV, dK (full [64x128] each, 2 col-halves)
    fence_operandN<64>(dv);
    stage_acc_bf16_s<64,64>(dv,    sQ_sw + 0,    wtid, 1.0f);
    stage_acc_bf16_s<64,64>(dv+32, sQ_sw + 4096, wtid, 1.0f);
    fence_operandN<64>(dk);
    stage_acc_bf16_s<64,64>(dk,    sdO_sw + 0,    wtid, scale);
    stage_acc_bf16_s<64,64>(dk+32, sdO_sw + 4096, wtid, scale);
    __syncthreads(); fence_proxy_async_shared();
    // Vj1d: one-writer bf16 PARTIAL store to [B,Hq,S,D] at hqRow (each per-hq CTA owns its own slot),
    // then a fixed-order greduce sums the G heads DETERMINISTICALLY (no atomics) -> bit-identical dK/dV
    // run-to-run, matching cuDNN's one-writer reproducibility. Price = the ~69us greduce pass.
    const uint32_t hqRow = (uint32_t)((b*Hq+hq)*S + k_row0);
    if (wtid == 0) {
        tma_store_2d_v34(&tma_dV_st, sQ_sw + 0,    0,  hqRow);
        tma_store_2d_v34(&tma_dV_st, sQ_sw + 4096, 64, hqRow);
        tma_store_2d_v34(&tma_dK_st, sdO_sw + 0,    0,  hqRow);
        tma_store_2d_v34(&tma_dK_st, sdO_sw + 4096, 64, hqRow);
        tma_store_commit_v34(); tma_store_wait_v34();
    }
}

template<int Br, int Bc, int D>
void launch_gqa_backward_vj1d(
    const bf16 *d_Q, const bf16 *d_K, const bf16 *d_V, const bf16 *d_O,
    const bf16 *d_dO, const float *d_LSE,
    bf16 *d_dQ, bf16 *d_dK, bf16 *d_dV,
    int B, int Hq, int Hkv, int G, int S, float scale) {
    auto make_tma_sw128 = [&](const bf16* ptr, uint64_t total_rows, uint32_t tile_rows) {
        CUtensorMap desc{}; uint64_t gS[2]={(uint64_t)D,total_rows}; uint64_t gT[1]={(uint64_t)D*sizeof(bf16)};
        uint32_t bx[2]={64u,tile_rows}, eS[2]={1,1};
        CUresult r=cuTensorMapEncodeTiled(&desc,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,(void*)ptr,gS,gT,bx,eS,
            CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if(r!=CUDA_SUCCESS){const char*e;cuGetErrorString(r,&e);fprintf(stderr,"sw128 vz2: %s\n",e);exit(1);} return desc; };
    auto make_tma_out = [&](const bf16* ptr, uint64_t rows) {
        CUtensorMap desc{}; uint64_t gS[2]={(uint64_t)D,rows}; uint64_t gT[1]={(uint64_t)D*sizeof(bf16)};
        uint32_t bx[2]={64u,64u}, eS[2]={1,1};
        CUresult r=cuTensorMapEncodeTiled(&desc,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,(void*)ptr,gS,gT,bx,eS,
            CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if(r!=CUDA_SUCCESS){const char*e;cuGetErrorString(r,&e);fprintf(stderr,"out vz2: %s\n",e);exit(1);} return desc; };
    auto make_tma_red = [&](const float* ptr, uint64_t rows) {
        CUtensorMap desc{}; uint64_t gS[2]={(uint64_t)D,rows}; uint64_t gT[1]={(uint64_t)D*sizeof(float)};
        uint32_t bx[2]={32u,64u}, eS[2]={1,1};
        CUresult r=cuTensorMapEncodeTiled(&desc,CU_TENSOR_MAP_DATA_TYPE_FLOAT32,2,(void*)ptr,gS,gT,bx,eS,
            CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if(r!=CUDA_SUCCESS){const char*e;cuGetErrorString(r,&e);fprintf(stderr,"red vz2: %s\n",e);exit(1);} return desc; };

    const uint64_t Rq=(uint64_t)B*Hq*S, Rkv=(uint64_t)B*Hkv*S;
    CUtensorMap tK=make_tma_sw128(d_K,Rkv,Bc), tV=make_tma_sw128(d_V,Rkv,Bc);
    CUtensorMap tQ=make_tma_sw128(d_Q,Rq,Br),  tdO=make_tma_sw128(d_dO,Rq,Br);
    // Vj1d: one-writer bf16 PARTIAL dK/dV [B,Hq,S,D]; a fixed-order greduce (no atomics) sums the G heads
    // DETERMINISTICALLY -> bit-identical dK/dV. Trade vs Vj1: the ~69us greduce pass instead of the atomic reduce.
    const long partN=(long)B*Hq*S*D; static bf16 *d_dKp=nullptr,*d_dVp=nullptr; static long pcap=0;
    if(partN>pcap){ if(d_dKp)CUDA_CHECK(cudaFree(d_dKp)); if(d_dVp)CUDA_CHECK(cudaFree(d_dVp));
        CUDA_CHECK(cudaMalloc(&d_dKp,partN*sizeof(bf16))); CUDA_CHECK(cudaMalloc(&d_dVp,partN*sizeof(bf16))); pcap=partN; }
    CUtensorMap tdV=make_tma_out(d_dVp,Rq), tdK=make_tma_out(d_dKp,Rq);

    const long drowN=(long)B*Hq*S; static float* d_Drow=nullptr; static long drc=0;
    if(drowN>drc){ if(d_Drow)CUDA_CHECK(cudaFree(d_Drow)); CUDA_CHECK(cudaMalloc(&d_Drow,drowN*sizeof(float))); drc=drowN; }
    const long dqN=(long)B*Hq*S*D; static float* d_dqa=nullptr; static long dqc=0;
    if(dqN>dqc){ if(d_dqa)CUDA_CHECK(cudaFree(d_dqa)); CUDA_CHECK(cudaMalloc(&d_dqa,dqN*sizeof(float))); dqc=dqN; }
    CUDA_CHECK(cudaMemset(d_dqa,0,dqN*sizeof(float)));
    CUtensorMap tRed=make_tma_red(d_dqa,(uint64_t)B*Hq*S);

    const int dBlock=256; const long dGrid=(drowN+(dBlock/32)-1)/(dBlock/32);
    compute_drowsum_v22<<<(unsigned)dGrid,dBlock>>>(d_dO,d_O,d_Drow,drowN);

    dim3 GRID(S/Bc, Hq, B);   // Vj1 per-hq: 3x the CTAs (Hq not Hkv) to fill the GPU at low B
    gqa_bwd_vj1d<Br,Bc,D><<<GRID,128>>>(tK,tV,tQ,tdO,tdV,tdK,tRed,d_LSE,d_Drow,B,Hq,Hkv,G,S,scale);
    // fixed-order deterministic G-head reduction (no atomics) -> bit-identical dK/dV.
    dim3 grGRID(S/8, Hkv, B);
    gqa_dkdv_greduce<<<grGRID,128>>>(d_dVp, d_dV, Hq, Hkv, G, S);
    gqa_dkdv_greduce<<<grGRID,128>>>(d_dKp, d_dK, Hq, Hkv, G, S);
    const int cB=256; const long dqN4=dqN/4; const int cG=(int)((dqN4+cB-1)/cB);
    convert_dq_accum_to_bf16_v6<<<cG,cB>>>(reinterpret_cast<const float4*>(d_dqa), reinterpret_cast<uint2*>(d_dQ), dqN4);
}

int main(int argc,char**argv){
    const int B=argc>1?atoi(argv[1]):4, Hq=argc>2?atoi(argv[2]):24, Hkv=argc>3?atoi(argv[3]):8;
    const int G=Hq/Hkv, N=argc>4?atoi(argv[4]):8;
    const int S=4096,D=128; const float scale=1.0f/sqrtf((float)D);
    const size_t Nq=(size_t)B*Hq*S*D, Nkv=(size_t)B*Hkv*S*D, Nlse=(size_t)B*Hq*S;
    std::cout<<"Vj1d (deterministic fixed-order dK/dV) probe  B="<<B<<" Hq="<<Hq<<" Hkv="<<Hkv
             <<" | "<<N<<" runs, identical inputs\n\n";
    std::vector<bf16> hQ(Nq),hK(Nkv),hV(Nkv),hO(Nq),hdO(Nq); std::vector<float> hLSE(Nlse);
    auto ldbf=[&](const char*p,std::vector<bf16>&d,size_t n){std::vector<float> t(n);loadBin(p,t.data(),n);
        for(size_t i=0;i<n;++i)d[i]=__float2bfloat16(t[i]);};
    ldbf("data/gqa_q.bin",hQ,Nq); ldbf("data/gqa_k.bin",hK,Nkv); ldbf("data/gqa_v.bin",hV,Nkv);
    ldbf("data/gqa_o.bin",hO,Nq); ldbf("data/gqa_do.bin",hdO,Nq); loadBin("data/gqa_lse.bin",hLSE.data(),Nlse);
    bf16*dQ,*dK,*dV,*dO,*O,*Q,*K,*V; float*LSE;
    CUDA_CHECK(cudaMalloc(&Q,Nq*sizeof(bf16)));CUDA_CHECK(cudaMalloc(&K,Nkv*sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&V,Nkv*sizeof(bf16)));CUDA_CHECK(cudaMalloc(&O,Nq*sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&dO,Nq*sizeof(bf16)));CUDA_CHECK(cudaMalloc(&LSE,Nlse*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dQ,Nq*sizeof(bf16)));CUDA_CHECK(cudaMalloc(&dK,Nkv*sizeof(bf16)));CUDA_CHECK(cudaMalloc(&dV,Nkv*sizeof(bf16)));
    CUDA_CHECK(cudaMemcpy(Q,hQ.data(),Nq*sizeof(bf16),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(K,hK.data(),Nkv*sizeof(bf16),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(V,hV.data(),Nkv*sizeof(bf16),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(O,hO.data(),Nq*sizeof(bf16),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dO,hdO.data(),Nq*sizeof(bf16),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(LSE,hLSE.data(),Nlse*sizeof(float),cudaMemcpyHostToDevice));
    auto grab=[&](std::vector<bf16>&q,std::vector<bf16>&k,std::vector<bf16>&v){
        launch_gqa_backward_vj1d<64,64,128>(Q,K,V,O,dO,LSE,dQ,dK,dV,B,Hq,Hkv,G,S,scale);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        q.resize(Nq);k.resize(Nkv);v.resize(Nkv);
        CUDA_CHECK(cudaMemcpy(q.data(),dQ,Nq*sizeof(bf16),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(k.data(),dK,Nkv*sizeof(bf16),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(v.data(),dV,Nkv*sizeof(bf16),cudaMemcpyDeviceToHost));};
    auto maxd=[&](const std::vector<bf16>&a,const std::vector<bf16>&b){float m=0;bool eq=true;
        for(size_t i=0;i<a.size();++i){float d=fabsf(__bfloat162float(a[i])-__bfloat162float(b[i]));
            if(d>m)m=d; if(a[i]!=b[i])eq=false;} return std::make_pair(eq,m);};
    std::vector<bf16> q0,k0,v0,q1,k1,v1; grab(q0,k0,v0);
    const char* nm[3]={"dQ","dK","dV"}; bool anyQ=false,anyKV=false;
    for(int r=1;r<N;r++){ grab(q1,k1,v1);
        std::pair<bool,float> res[3]={maxd(q0,q1),maxd(k0,k1),maxd(v0,v1)};
        for(int g=0;g<3;g++){ printf("run %d: %s %-14s max|Δ|=%.3e\n",r,nm[g],res[g].first?"bit-identical":"DIFFERS",res[g].second);
            if(!res[g].first){ if(g==0)anyQ=true; else anyKV=true; } }
        printf("\n"); }
    printf("============================================================\n");
    printf("Vj1d  dQ: %s (K-tile atomic reduce, like cuDNN dQ)   dK/dV: %s\n",
           anyQ?"NON-DET":"deterministic", anyKV?"NON-DET":"BIT-IDENTICAL (fixed-order greduce = cuDNN-grade)");
    return 0;
}
