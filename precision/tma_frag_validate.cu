// Standalone validation: cuDNN-style coalesced dQ store + 5D TMA-reduce, proving the store<->descriptor
// mapping is correct in isolation before wiring into the attention kernel. Run on H200 (sm_90a).
//
// One warpgroup (128 thr) owns a 64x64 fp32 tile T. mma-C fragment: lane holds (r0,c),(r0,c+1),(r1,c),
// (r1,c+1); r0=w*16+(lane>>2), r1=r0+8, cc=(lane&3)*2, c=nt*8+cc, nt=0..7.
// TMA inner box must be >=4 fp32, fragment gives 2 -> shfl-extend cols to 4 (even lanes gather lane^1).
// The fragment needs 6 logical dims (i splits row/col); TMA is 5D -> split r0/r1 into TWO descriptors.
//   stage_r0 / stage_r1 laid out [w(4)][laneHi=lane>>2(8)][nt(8)][ccBit=cc>>2(2)][col4(4)] = 2048 fp32.
//   r0 rows = w*16+laneHi (32 rows: 0-7,16-23,32-39,48-55) ; r1 rows = +8 (the gaps).
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
    int r0=w*16+(lane>>2), r1=r0+8, cc=(lane&3)*2;
    float d[32];
    #pragma unroll
    for(int nt=0;nt<8;nt++){int c=nt*8+cc;
        d[nt*4+0]=T[r0*64+c]; d[nt*4+1]=T[r0*64+c+1];
        d[nt*4+2]=T[r1*64+c]; d[nt*4+3]=T[r1*64+c+1];}
    bool even=!(lane&1); int laneHi=lane>>2, ccBit=cc>>2;
    #pragma unroll
    for(int nt=0;nt<8;nt++){
        float n0=__shfl_xor_sync(~0u,d[nt*4+0],1), n1=__shfl_xor_sync(~0u,d[nt*4+1],1);
        float n2=__shfl_xor_sync(~0u,d[nt*4+2],1), n3=__shfl_xor_sync(~0u,d[nt*4+3],1);
        if(even){
            int base=((((w*8+laneHi)*8+nt)*2+ccBit)*4);   // [w][laneHi][nt][ccBit][col4]
            s0[base+0]=d[nt*4+0]; s0[base+1]=d[nt*4+1]; s0[base+2]=n0; s0[base+3]=n1;
            s1[base+0]=d[nt*4+2]; s1[base+1]=d[nt*4+3]; s1[base+2]=n2; s1[base+3]=n3;
        }
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
    // box dims inner->outer {col4, ccBit, nt, laneHi, w}; global strides(elem): col4=1, ccBit=4, nt=8, laneHi=64, w=16*64
    uint64_t gS[5]={4,2,8,8,4}; uint64_t gStr[4]={4*4ull,8*4ull,64*4ull,(uint64_t)16*64*4};
    uint32_t bx[5]={4,2,8,8,4}; uint32_t eS[5]={1,1,1,1,1};
    auto enc=[&](CUtensorMap* m,float* base){
        CUresult r=cuTensorMapEncodeTiled(m,CU_TENSOR_MAP_DATA_TYPE_FLOAT32,5,base,gS,gStr,bx,eS,
            CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if(r!=CUDA_SUCCESS){const char*e;cuGetErrorString(r,&e);printf("encode: %s\n",e);exit(1);} };
    CUtensorMap d0{},d1{}; enc(&d0,dG); enc(&d1,dG+8*64);   // r0 base, r1 base = +8 rows
    k<<<1,128>>>(dT,d0,d1); CK(cudaDeviceSynchronize());
    float* G=(float*)malloc(64*64*4); CK(cudaMemcpy(G,dG,64*64*4,cudaMemcpyDeviceToHost));
    int bad=0; for(int i=0;i<64*64&&bad<10;i++) if(G[i]!=T[i]){printf("mismatch row %d col %d: got %.0f want %.0f\n",i/64,i%64,G[i],T[i]);bad++;}
    printf(bad?"FAIL\n":"PASS: coalesced store + 5D descriptor maps correctly (foundation valid)\n",bad);
    return 0;
}
