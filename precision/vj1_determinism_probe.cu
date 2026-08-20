// Determinism probe for OUR Vj1 kernel — the mirror of check_cudnn_determinism.py.
// Runs launch_gqa_backward_vj1 N times on byte-identical device inputs and checks
// whether dQ/dK/dV come back bit-for-bit equal. Reports max|Δ| per gradient.
//
// Expectation (given the analysis): dQ non-deterministic (fp32 cross-K-tile TMA-reduce,
// like cuDNN), AND dK/dV non-deterministic too (bf16 cross-head TMA-reduce — the per-hq
// price cuDNN doesn't pay). This measures the magnitude of that dK/dV wobble.
//
// Build: nvcc -gencode arch=compute_90a,code=sm_90a -Iinclude \
//             precision/vj1_determinism_probe.cu -o vj1_det -lcuda
// Run:   python precision/baseline_gqa.py B Hq Hkv ; ./vj1_det B Hq Hkv [N]

#define VJ1_NO_MAIN
#include "../src/attention/GQA_bwd_baseline.cu"

static void run_into(const bf16*Q,const bf16*K,const bf16*V,const bf16*O,const bf16*dO,
                     const float*LSE, bf16*dQ,bf16*dK,bf16*dV,
                     int B,int Hq,int Hkv,int G,int S,float scale){
    launch_gqa_backward_vj1<64,64,128>(Q,K,V,O,dO,LSE,dQ,dK,dV,B,Hq,Hkv,G,S,scale);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
}

int main(int argc,char**argv){
    const int B=argc>1?atoi(argv[1]):4, Hq=argc>2?atoi(argv[2]):24, Hkv=argc>3?atoi(argv[3]):8;
    const int G=Hq/Hkv, N=argc>4?atoi(argv[4]):8;
    const int S=4096,D=128; const float scale=1.0f/sqrtf((float)D);
    const size_t Nq=(size_t)B*Hq*S*D, Nkv=(size_t)B*Hkv*S*D, Nlse=(size_t)B*Hq*S;
    std::cout<<"Vj1 determinism probe  B="<<B<<" Hq="<<Hq<<" Hkv="<<Hkv<<" S="<<S<<" D="<<D
             <<" bf16 causal | "<<N<<" runs, identical inputs\n\n";

    std::vector<bf16> hQ(Nq),hK(Nkv),hV(Nkv),hO(Nq),hdO(Nq); std::vector<float> hLSE(Nlse);
    auto ldbf=[&](const char*p,std::vector<bf16>&d,size_t n){std::vector<float> t(n);loadBin(p,t.data(),n);
        for(size_t i=0;i<n;++i)d[i]=__float2bfloat16(t[i]);};
    ldbf("data/gqa_q.bin",hQ,Nq); ldbf("data/gqa_k.bin",hK,Nkv); ldbf("data/gqa_v.bin",hV,Nkv);
    ldbf("data/gqa_o.bin",hO,Nq); ldbf("data/gqa_do.bin",hdO,Nq); loadBin("data/gqa_lse.bin",hLSE.data(),Nlse);

    bf16*dQ,*dK,*dV,*dO,*O,*Q,*K,*V; float*LSE;
    CUDA_CHECK(cudaMalloc(&Q,Nq*sizeof(bf16)));CUDA_CHECK(cudaMalloc(&K,Nkv*sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&V,Nkv*sizeof(bf16)));CUDA_CHECK(cudaMalloc(&O,Nq*sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&dO,Nq*sizeof(bf16)));CUDA_CHECK(cudaMalloc(&LSE,Nlse*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dQ,Nq*sizeof(bf16)));CUDA_CHECK(cudaMalloc(&dK,Nkv*sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&dV,Nkv*sizeof(bf16)));
    CUDA_CHECK(cudaMemcpy(Q,hQ.data(),Nq*sizeof(bf16),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(K,hK.data(),Nkv*sizeof(bf16),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(V,hV.data(),Nkv*sizeof(bf16),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(O,hO.data(),Nq*sizeof(bf16),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dO,hdO.data(),Nq*sizeof(bf16),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(LSE,hLSE.data(),Nlse*sizeof(float),cudaMemcpyHostToDevice));

    auto grab=[&](std::vector<bf16>&q,std::vector<bf16>&k,std::vector<bf16>&v){
        run_into(Q,K,V,O,dO,LSE,dQ,dK,dV,B,Hq,Hkv,G,S,scale);
        q.resize(Nq);k.resize(Nkv);v.resize(Nkv);
        CUDA_CHECK(cudaMemcpy(q.data(),dQ,Nq*sizeof(bf16),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(k.data(),dK,Nkv*sizeof(bf16),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(v.data(),dV,Nkv*sizeof(bf16),cudaMemcpyDeviceToHost));};
    auto maxd=[&](const std::vector<bf16>&a,const std::vector<bf16>&b){
        float m=0; bool eq=true;
        for(size_t i=0;i<a.size();++i){float d=fabsf(__bfloat162float(a[i])-__bfloat162float(b[i]));
            if(d>m)m=d; if(a[i]!=b[i])eq=false;} return std::make_pair(eq,m);};

    std::vector<bf16> q0,k0,v0,q1,k1,v1; grab(q0,k0,v0);
    const char* nm[3]={"dQ","dK","dV"}; bool anyQ=false,anyKV=false;
    for(int r=1;r<N;r++){ grab(q1,k1,v1);
        std::pair<bool,float> res[3]={maxd(q0,q1),maxd(k0,k1),maxd(v0,v1)};
        for(int g=0;g<3;g++){ bool eq=res[g].first; float md=res[g].second;
            printf("run %d: %s %-14s max|Δ|=%.3e\n",r,nm[g],eq?"bit-identical":"DIFFERS",md);
            if(!eq){ if(g==0)anyQ=true; else anyKV=true; } }
        printf("\n");
    }
    printf("============================================================\n");
    printf("Vj1  dQ: %s   dK/dV: %s\n", anyQ?"NON-DET (fp32 K-tile reduce)":"deterministic",
           anyKV?"NON-DET (bf16 cross-head reduce — per-hq price)":"deterministic");
    CUDA_CHECK(cudaFree(Q));CUDA_CHECK(cudaFree(K));CUDA_CHECK(cudaFree(V));CUDA_CHECK(cudaFree(O));
    CUDA_CHECK(cudaFree(dO));CUDA_CHECK(cudaFree(LSE));CUDA_CHECK(cudaFree(dQ));CUDA_CHECK(cudaFree(dK));CUDA_CHECK(cudaFree(dV));
    return 0;
}
