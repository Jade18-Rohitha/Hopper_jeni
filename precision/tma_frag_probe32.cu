// Probe: cuDNN-style dQ store, ALL-32-LANE version (fix for V5c's even-only + laneHi-64 alias).
// Goal: prove (a) bit-exact global result AND (b) low shared store-conflicts, in isolation, on H200.
//
// mma-C fragment: lane holds (r0,c),(r0,c+1),(r1,c),(r1,c+1); r0=w*16+(lane>>2), r1=r0+8,
//   cc=(lane&3)*2, c=nt*8+cc, nt=0..7.  Native run = 2 cols (8B) < TMA 16B min inner box.
// Fix: shfl_xor(1) pairs lane with lane^1 (adjacent col-pair) -> 4 contiguous cols (16B).
//   EVEN lane -> assembles r0's 4-col float4 -> s0.   ODD lane -> r1's 4-col float4 -> s1.
//   ALL 32 lanes active (V5c left the 16 odd lanes idle -> conflicts + half BW).
//   pk = laneHi*2 + colGroup (0..15) packs the 16 active lanes contiguous per (w,nt).
//   store base ((w*8+nt)*16+pk)*4 == col4 + colGroup*4 + laneHi*8 + nt*64 + w*512
//     == the 5D descriptor's contiguous [w][nt][laneHi][colGroup][col4] layout.  <-- key identity
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#define CK(x) do{cudaError_t e=(x);if(e){printf("cuda err %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)

__device__ __forceinline__ void red5(const void* d,const float* s,uint32_t c0,uint32_t c1,uint32_t c2,uint32_t c3,uint32_t c4){
    uint32_t src=(uint32_t)__cvta_generic_to_shared(s);
    asm volatile("cp.reduce.async.bulk.tensor.5d.global.shared::cta.add.tile.bulk_group [%0,{%1,%2,%3,%4,%5}],[%6];\n"
        ::"l"((uint64_t)d),"r"(c0),"r"(c1),"r"(c2),"r"(c3),"r"(c4),"r"(src):"memory");
}
__global__ void k(const float* __restrict__ T, const __grid_constant__ CUtensorMap desc0,
                                                const __grid_constant__ CUtensorMap desc1){
    __shared__ __align__(1024) float s0[2048];
    __shared__ __align__(1024) float s1[2048];
    int tid=threadIdx.x, w=tid>>5, lane=tid&31;
    int laneHi=lane>>2, colGroup=(lane&3)>>1;
    int r0=w*16+laneHi, r1=r0+8, cc=(lane&3)*2;
    bool r0lane = !(lane&1);
    int pk = laneHi*2 + colGroup;
    #pragma unroll
    for(int nt=0;nt<8;nt++){
        int c=nt*8+cc;
        float v0=T[r0*64+c], v1=T[r0*64+c+1], v2=T[r1*64+c], v3=T[r1*64+c+1]; // r0c0,r0c1,r1c0,r1c1
        float p0=__shfl_xor_sync(~0u,v0,1), p1=__shfl_xor_sync(~0u,v1,1);       // partner's r0c2,r0c3
        float p2=__shfl_xor_sync(~0u,v2,1), p3=__shfl_xor_sync(~0u,v3,1);       // partner's r1c2,r1c3
        int base=((w*8+nt)*16+pk)*4;
        if(r0lane){ s0[base+0]=v0; s0[base+1]=v1; s0[base+2]=p0; s0[base+3]=p1; } // r0: cols c..c+3
        else      { s1[base+0]=p2; s1[base+1]=p3; s1[base+2]=v2; s1[base+3]=v3; } // r1: cols (c-2)..c+1
    }
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;\n":::"memory");
    if(tid==0){
        red5(&desc0, s0, 0,0,0,0,0);
        red5(&desc1, s1, 0,0,0,0,0);
        asm volatile("cp.async.bulk.commit_group;\n":::"memory");
        asm volatile("cp.async.bulk.wait_group 0;\n":::"memory");
    }
}
int main(){
    float* T=(float*)malloc(64*64*4); for(int i=0;i<64*64;i++)T[i]=(float)(i%97)-48.f;
    float *dT,*dG; CK(cudaMalloc(&dT,64*64*4)); CK(cudaMalloc(&dG,64*64*4));
    CK(cudaMemcpy(dT,T,64*64*4,cudaMemcpyHostToDevice)); CK(cudaMemset(dG,0,64*64*4));
    // box dims inner->outer {col4:4, colGroup:2, laneHi:8, nt:8, w:4}; global elem strides:
    //   col4=1, colGroup=4, laneHi=64(row), nt=8, w=16*64(row). gStr = strides of dims 1..4 in bytes.
    uint64_t gS[5]={4,2,8,8,4};
    uint64_t gStr[4]={4*4ull, 64*4ull, 8*4ull, (uint64_t)16*64*4};
    uint32_t bx[5]={4,2,8,8,4}; uint32_t eS[5]={1,1,1,1,1};
    auto enc=[&](CUtensorMap* m,float* base){
        CUresult r=cuTensorMapEncodeTiled(m,CU_TENSOR_MAP_DATA_TYPE_FLOAT32,5,base,gS,gStr,bx,eS,
            CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if(r!=CUDA_SUCCESS){const char*e;cuGetErrorString(r,&e);printf("encode: %s\n",e);exit(1);} };
    CUtensorMap d0{},d1{}; enc(&d0,dG); enc(&d1,dG+8*64);   // r0 base, r1 base = +8 rows
    k<<<1,128>>>(dT,d0,d1); CK(cudaDeviceSynchronize());
    float* G=(float*)malloc(64*64*4); CK(cudaMemcpy(G,dG,64*64*4,cudaMemcpyDeviceToHost));
    int bad=0; for(int i=0;i<64*64&&bad<10;i++) if(G[i]!=T[i]){printf("mismatch row %d col %d: got %.0f want %.0f\n",i/64,i%64,G[i],T[i]);bad++;}
    printf(bad?"FAIL\n":"PASS: all-32-lane coalesced store + 5D descriptor maps correctly\n");
    return 0;
}
