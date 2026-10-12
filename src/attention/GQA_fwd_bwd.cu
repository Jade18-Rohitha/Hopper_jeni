// ============================================================================
//  GQA_fwd_bwd.cu  —  FORWARD + BACKWARD chain: V62 forward -> Vj1d backward  [H200, sm_90a]
//
//  Apples-to-apples with cuDNN's training step (sdpa(...).backward(dO)): our V62 forward
//  produces O + LSE, and Vj1d consumes THOSE (not the PyTorch reference O/LSE), so the
//  chain is exactly what a training step would run.
//
//  Timing mirrors triton.testing.do_bench (what precision/baseline_gqa.py uses for cuDNN):
//  per rep = L2 flush, start event, work, end event; all reps enqueued back-to-back, ONE
//  sync at the end -> host launch overhead overlaps GPU work on both sides.
//
//  Build:  cmake --build build --target gqa_fwd_bwd -j
//  Run:    python precision/baseline_gqa.py B Hq Hkv     # writes data/gqa_*.bin
//          ./build/bin/gqa_fwd_bwd B Hq Hkv
// ============================================================================
#define VJ1_NO_MAIN
#include "GQA_bwd_baseline.cu"   // Vj1d: launch_gqa_backward_vj1d<Br,Bc,D>
#include "GQA_fwd_ref.cu"        // V62:  launch_gqa_v62

#include <algorithm>

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
    printf("GQA fwd+bwd chain: V62 fwd -> Vj1d bwd   B=%d Hq=%d Hkv=%d (G=%d) S=%d D=%d causal bf16\n\n",
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
        fprintf(stderr, "ERROR: reference missing/size mismatch. Run: python precision/baseline_gqa.py %d %d %d\n",
                B, Hq, Hkv);
        return 1;
    }
    std::vector<bf16> h_Q(Nq), h_K(Nkv), h_V(Nkv), h_dO(Nq);
    std::vector<float> h_Oref(Nq), h_LSEref(Nlse), h_dQref(Nq), h_dKref(Nkv), h_dVref(Nkv);
    auto loadBF16 = [](const char *p, std::vector<bf16> &dst, size_t n) {
        std::vector<float> tmp(n); loadBin(p, tmp.data(), n);
        for (size_t i = 0; i < n; ++i) dst[i] = __float2bfloat16(tmp[i]);
    };
    loadBF16("data/gqa_q.bin", h_Q, Nq);  loadBF16("data/gqa_k.bin", h_K, Nkv);
    loadBF16("data/gqa_v.bin", h_V, Nkv); loadBF16("data/gqa_do.bin", h_dO, Nq);
    loadBin("data/gqa_o.bin",   h_Oref.data(),   Nq);
    loadBin("data/gqa_lse.bin", h_LSEref.data(), Nlse);
    loadBin("data/gqa_dq.bin",  h_dQref.data(),  Nq);
    loadBin("data/gqa_dk.bin",  h_dKref.data(),  Nkv);
    loadBin("data/gqa_dv.bin",  h_dVref.data(),  Nkv);

    bf16 *d_Q,*d_K,*d_V,*d_O,*d_dO,*d_dQ,*d_dK,*d_dV; float *d_LSE;
    CUDA_CHECK(cudaMalloc(&d_Q,Nq*sizeof(bf16)));   CUDA_CHECK(cudaMalloc(&d_K,Nkv*sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&d_V,Nkv*sizeof(bf16)));  CUDA_CHECK(cudaMalloc(&d_O,Nq*sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&d_dO,Nq*sizeof(bf16)));  CUDA_CHECK(cudaMalloc(&d_LSE,Nlse*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dQ,Nq*sizeof(bf16)));  CUDA_CHECK(cudaMalloc(&d_dK,Nkv*sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&d_dV,Nkv*sizeof(bf16)));
    CUDA_CHECK(cudaMemcpy(d_Q,h_Q.data(),Nq*sizeof(bf16),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K,h_K.data(),Nkv*sizeof(bf16),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V,h_V.data(),Nkv*sizeof(bf16),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dO,h_dO.data(),Nq*sizeof(bf16),cudaMemcpyHostToDevice));

    auto fwd = [&]() { launch_gqa_v62(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, scale, 0); };
    auto bwd = [&]() { launch_gqa_backward_vj1d<Br,Bc,D>(d_Q,d_K,d_V,d_O,d_dO,d_LSE,d_dQ,d_dK,d_dV,
                                                         B,Hq,Hkv,G,S,scale); };

    // ── correctness: chained (bwd consumes OUR forward's O/LSE) ──
    fwd(); CUDA_CHECK(cudaGetLastError());
    bwd(); CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto toF = [](const bf16 *d, size_t n) {
        std::vector<bf16> h(n); CUDA_CHECK(cudaMemcpy(h.data(), d, n*sizeof(bf16), cudaMemcpyDeviceToHost));
        std::vector<float> f(n); for (size_t i = 0; i < n; ++i) f[i] = __bfloat162float(h[i]); return f;
    };
    {
        std::vector<float> o = toF(d_O, Nq), lse(Nlse);
        CUDA_CHECK(cudaMemcpy(lse.data(), d_LSE, Nlse*sizeof(float), cudaMemcpyDeviceToHost));
        std::vector<float> dq = toF(d_dQ, Nq), dk = toF(d_dK, Nkv), dv = toF(d_dV, Nkv);
        printf("-- V62 forward vs PyTorch reference --\n");
        reportPrecision("  O  ", h_Oref.data(),   o.data(),   Nq);
        reportPrecision("  LSE", h_LSEref.data(), lse.data(), Nlse);
        std::cout << "  O:   "; checkResult(h_Oref.data(),   o.data(),   Nq,   2e-2f, 2e-2f);
        std::cout << "  LSE: "; checkResult(h_LSEref.data(), lse.data(), Nlse, 2e-2f, 2e-2f);
        printf("-- Vj1d backward on V62's O/LSE vs PyTorch reference --\n");
        reportPrecision("  dQ", h_dQref.data(), dq.data(), Nq);
        reportPrecision("  dK", h_dKref.data(), dk.data(), Nkv);
        reportPrecision("  dV", h_dVref.data(), dv.data(), Nkv);
        std::cout << "  dQ: "; checkResult(h_dQref.data(), dq.data(), Nq,  2e-2f, 2e-2f);
        std::cout << "  dK: "; checkResult(h_dKref.data(), dk.data(), Nkv, 2e-2f, 2e-2f);
        std::cout << "  dV: "; checkResult(h_dVref.data(), dv.data(), Nkv, 2e-2f, 2e-2f);
        printf("\n");
    }

    // ── latency (do_bench-style medians) ──
    const float t_fwd = median_ms_dobench(fwd);
    const float t_bwd = median_ms_dobench(bwd);
    const float t_fb  = median_ms_dobench([&]() { fwd(); bwd(); });
    printf("OURS fwd V62              %.4f ms\n", t_fwd);
    printf("OURS bwd Vj1d             %.4f ms\n", t_bwd);
    printf("OURS fwd+bwd V62+Vj1d     %.4f ms\n", t_fb);

    CUDA_CHECK(cudaFree(d_Q));  CUDA_CHECK(cudaFree(d_K));  CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));  CUDA_CHECK(cudaFree(d_dO)); CUDA_CHECK(cudaFree(d_LSE));
    CUDA_CHECK(cudaFree(d_dQ)); CUDA_CHECK(cudaFree(d_dK)); CUDA_CHECK(cudaFree(d_dV));
    return 0;
}
