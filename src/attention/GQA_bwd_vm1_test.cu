// ============================================================================
//  GQA_bwd_vm1_test.cu  —  standalone head-to-head: Vj1d vs Vm3 vs Vm4 (backward only)  [H200, sm_90a]
//
//  Same inputs and checks as gqa_bwd_baseline (PyTorch reference from data/gqa_*.bin), then:
//    1. correctness of dQ/dK/dV for Vj1d and Vm1 (tol 2e-2)
//    2. Vm3 dK/dV bit-identical over 6 runs
//    3. latency of both, two ways:
//       - benchmarkKernel (per-iteration sync; same method gqa_bwd_baseline reports)
//       - do_bench-style median (async queue, one sync; same method precision/cudnn_time.py uses
//         for cuDNN, so "do_bench" lines are directly comparable to "cuDNN bwd-only")
//
//  Build:  cmake --build build --target gqa_bwd_vm1_test -j
//  Run:    python3 precision/baseline_gqa.py B Hq Hkv
//          ./build/bin/gqa_bwd_vm1_test B Hq Hkv
//          python3 precision/cudnn_time.py B Hq Hkv        # cuDNN reference, same inputs
// ============================================================================
#define GQA_BWD_NO_MAIN
#include "GQA_bwd.cu"            // launch_gqa_backward_vj1d / _vm3 / _vm4

#include <algorithm>
#include <cstring>

template<typename F>
static float median_ms_dobench(F work, int iters = 100, int warmup = 25) {
    constexpr size_t kFlushBytes = 256ULL * 1024 * 1024;
    int *flush = nullptr;
    CUDA_CHECK(cudaMalloc(&flush, kFlushBytes));
    for (int i = 0; i < warmup; ++i) work();
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<cudaEvent_t> st(iters), en(iters);
    for (int i = 0; i < iters; ++i) { cudaEventCreate(&st[i]); cudaEventCreate(&en[i]); }
    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaMemsetAsync(flush, 0, kFlushBytes));
        CUDA_CHECK(cudaEventRecord(st[i]));
        work();
        CUDA_CHECK(cudaEventRecord(en[i]));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> t(iters);
    for (int i = 0; i < iters; ++i) {
        cudaEventElapsedTime(&t[i], st[i], en[i]);
        cudaEventDestroy(st[i]); cudaEventDestroy(en[i]);
    }
    CUDA_CHECK(cudaFree(flush));
    std::sort(t.begin(), t.end());
    return t[iters / 2];
}

int main(int argc, char** argv) {
    const int B   = argc > 1 ? atoi(argv[1]) : 8;
    const int Hq  = argc > 2 ? atoi(argv[2]) : 12;
    const int Hkv = argc > 3 ? atoi(argv[3]) : 4;
    const int G   = Hq / Hkv;
    constexpr int S = 4096, D = 128, Br = 64, Bc = 64;
    printf("GQA bwd head-to-head: Vj1d vs Vm3 vs Vm4   B=%d Hq=%d Hkv=%d (G=%d) S=%d D=%d causal bf16\n\n",
           B, Hq, Hkv, G, S, D);

    const float  scale = 1.0f / sqrtf((float)D);
    const size_t Nq = (size_t)B * Hq * S * D, Nkv = (size_t)B * Hkv * S * D, Nlse = (size_t)B * Hq * S;
    auto fileSz = [](const char *p, size_t n) -> bool {
        FILE *f = fopen(p, "rb"); if (!f) return false;
        fseek(f, 0, SEEK_END); size_t b = (size_t)ftell(f); fclose(f); return b == n * sizeof(float);
    };
    if (!fileSz("data/gqa_q.bin",Nq)||!fileSz("data/gqa_k.bin",Nkv)||!fileSz("data/gqa_v.bin",Nkv)||
        !fileSz("data/gqa_o.bin",Nq)||!fileSz("data/gqa_do.bin",Nq)||!fileSz("data/gqa_lse.bin",Nlse)||
        !fileSz("data/gqa_dq.bin",Nq)||!fileSz("data/gqa_dk.bin",Nkv)||!fileSz("data/gqa_dv.bin",Nkv)) {
        fprintf(stderr, "ERROR: reference missing/size mismatch. Run: python3 precision/baseline_gqa.py %d %d %d\n",
                B, Hq, Hkv);
        return 1;
    }
    std::vector<bf16> h_Q(Nq), h_K(Nkv), h_V(Nkv), h_O(Nq), h_dO(Nq);
    std::vector<float> h_LSE(Nlse), h_dQref(Nq), h_dKref(Nkv), h_dVref(Nkv);
    auto loadBF16 = [](const char *p, std::vector<bf16> &dst, size_t n) {
        std::vector<float> tmp(n); loadBin(p, tmp.data(), n);
        for (size_t i = 0; i < n; ++i) dst[i] = __float2bfloat16(tmp[i]);
    };
    loadBF16("data/gqa_q.bin", h_Q, Nq);  loadBF16("data/gqa_k.bin", h_K, Nkv);
    loadBF16("data/gqa_v.bin", h_V, Nkv); loadBF16("data/gqa_o.bin", h_O, Nq);
    loadBF16("data/gqa_do.bin", h_dO, Nq);
    loadBin("data/gqa_lse.bin", h_LSE.data(),  Nlse);
    loadBin("data/gqa_dq.bin",  h_dQref.data(), Nq);
    loadBin("data/gqa_dk.bin",  h_dKref.data(), Nkv);
    loadBin("data/gqa_dv.bin",  h_dVref.data(), Nkv);

    bf16 *d_Q,*d_K,*d_V,*d_O,*d_dO,*d_dQ,*d_dK,*d_dV; float *d_LSE;
    CUDA_CHECK(cudaMalloc(&d_Q,Nq*sizeof(bf16)));   CUDA_CHECK(cudaMalloc(&d_K,Nkv*sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&d_V,Nkv*sizeof(bf16)));  CUDA_CHECK(cudaMalloc(&d_O,Nq*sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&d_dO,Nq*sizeof(bf16)));  CUDA_CHECK(cudaMalloc(&d_LSE,Nlse*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dQ,Nq*sizeof(bf16)));  CUDA_CHECK(cudaMalloc(&d_dK,Nkv*sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&d_dV,Nkv*sizeof(bf16)));
    CUDA_CHECK(cudaMemcpy(d_Q,h_Q.data(),Nq*sizeof(bf16),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K,h_K.data(),Nkv*sizeof(bf16),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V,h_V.data(),Nkv*sizeof(bf16),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_O,h_O.data(),Nq*sizeof(bf16),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dO,h_dO.data(),Nq*sizeof(bf16),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_LSE,h_LSE.data(),Nlse*sizeof(float),cudaMemcpyHostToDevice));

    auto vj1d = [&]() { launch_gqa_backward_vj1d<Br,Bc,D>(d_Q,d_K,d_V,d_O,d_dO,d_LSE,d_dQ,d_dK,d_dV,B,Hq,Hkv,G,S,scale); };
    auto vm3  = [&]() { launch_gqa_backward_vm3 <Br,Bc,D>(d_Q,d_K,d_V,d_O,d_dO,d_LSE,d_dQ,d_dK,d_dV,B,Hq,Hkv,G,S,scale); };
    auto vm4  = [&]() { launch_gqa_backward_vm4 <Br,Bc,D>(d_Q,d_K,d_V,d_O,d_dO,d_LSE,d_dQ,d_dK,d_dV,B,Hq,Hkv,G,S,scale); };

    auto toF = [](const bf16 *d, size_t n) {
        std::vector<bf16> h(n); CUDA_CHECK(cudaMemcpy(h.data(), d, n*sizeof(bf16), cudaMemcpyDeviceToHost));
        std::vector<float> f(n); for (size_t i = 0; i < n; ++i) f[i] = __bfloat162float(h[i]); return f;
    };
    auto check = [&](const char *name, auto launch) {
        CUDA_CHECK(cudaMemset(d_dQ, 0, Nq*sizeof(bf16)));
        CUDA_CHECK(cudaMemset(d_dK, 0, Nkv*sizeof(bf16)));
        CUDA_CHECK(cudaMemset(d_dV, 0, Nkv*sizeof(bf16)));
        launch(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> dq = toF(d_dQ, Nq), dk = toF(d_dK, Nkv), dv = toF(d_dV, Nkv);
        printf("-- %s vs PyTorch reference --\n", name);
        reportPrecision("  dQ", h_dQref.data(), dq.data(), Nq);
        reportPrecision("  dK", h_dKref.data(), dk.data(), Nkv);
        reportPrecision("  dV", h_dVref.data(), dv.data(), Nkv);
        std::cout << "CHECK " << name << " dQ:  "; checkResult(h_dQref.data(), dq.data(), Nq,  2e-2f, 2e-2f);
        std::cout << "CHECK " << name << " dK:  "; checkResult(h_dKref.data(), dk.data(), Nkv, 2e-2f, 2e-2f);
        std::cout << "CHECK " << name << " dV:  "; checkResult(h_dVref.data(), dv.data(), Nkv, 2e-2f, 2e-2f);
    };
    check("Vj1d", vj1d);
    check("Vm3 ", vm3);
    check("Vm4 ", vm4);
    auto determinism = [&](const char *name, auto launch) {   // dK/dV bit-identical run to run
        std::vector<bf16> k0(Nkv), v0(Nkv), k1(Nkv), v1(Nkv); bool same = true;
        launch(); CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(k0.data(), d_dK, Nkv*sizeof(bf16), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(v0.data(), d_dV, Nkv*sizeof(bf16), cudaMemcpyDeviceToHost));
        for (int r = 0; r < 5 && same; r++) {
            launch(); CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(k1.data(), d_dK, Nkv*sizeof(bf16), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(v1.data(), d_dV, Nkv*sizeof(bf16), cudaMemcpyDeviceToHost));
            same = !memcmp(k0.data(), k1.data(), Nkv*sizeof(bf16)) && !memcmp(v0.data(), v1.data(), Nkv*sizeof(bf16));
        }
        printf("CHECK %s dK/dV bit-identical over 6 runs: %s\n", name, same ? "Values match!" : "Values don't match (NON-DETERMINISTIC)");
    };
    determinism("Vm3 ", vm3);
    determinism("Vm4 ", vm4);
    printf("\n");

    // latency, interleaved round-robin so clock/thermal drift hits all kernels equally
    const long long bwd_flops = 16LL * B * Hq * (long long)S * S * D;
    std::vector<float> t[3];
    for (int r = 0; r < 3; r++) { t[0].push_back(median_ms_dobench(vj1d)); t[1].push_back(median_ms_dobench(vm3));
                                  t[2].push_back(median_ms_dobench(vm4)); }
    for (auto &v : t) std::sort(v.begin(), v.end());
    {   KernelStats s = benchmarkKernel(vj1d, 100, 10, bwd_flops); displayStats("GQA bwd Vj1d  (benchmarkKernel, per-iteration sync)", s); }
    {   KernelStats s = benchmarkKernel(vm3,  100, 10, bwd_flops); displayStats("GQA bwd Vm3   (benchmarkKernel, per-iteration sync)", s); }
    {   KernelStats s = benchmarkKernel(vm4,  100, 10, bwd_flops); displayStats("GQA bwd Vm4   (benchmarkKernel, per-iteration sync)", s); }
    const char *nm[3] = {"Vj1d", "Vm3 ", "Vm4 "};
    printf("\n");
    for (int k = 0; k < 3; k++)
        printf("OURS bwd %s   do_bench median-of-3  %.4f ms   (runs %.4f %.4f %.4f)   vs Vj1d %+.4f ms\n",
               nm[k], t[k][1], t[k][0], t[k][1], t[k][2], t[k][1] - t[0][1]);
    printf("(vs Vj1d: negative = faster than Vj1d)\n");
    printf("Compare with: python3 precision/cudnn_time.py %d %d %d   (line 'cuDNN bwd-only')\n", B, Hq, Hkv);

    CUDA_CHECK(cudaFree(d_Q));  CUDA_CHECK(cudaFree(d_K));  CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));  CUDA_CHECK(cudaFree(d_dO)); CUDA_CHECK(cudaFree(d_LSE));
    CUDA_CHECK(cudaFree(d_dQ)); CUDA_CHECK(cudaFree(d_dK)); CUDA_CHECK(cudaFree(d_dV));
    return 0;
}
