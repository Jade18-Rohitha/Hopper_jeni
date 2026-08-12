// Validation 2: FULL-TENSOR 5D descriptor with per-tile coords (what the real kernel needs).
// dq_accum [TR rows, 128 cols]. A (wg, rowtile) writes its 64x64 half-tile via coalesced store; the ONE
// descriptor + reduce COORDS place it. Proves tile placement (rowbase/16 on w, wg*8 on nt, 0/8 on laneHi).
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#define CK(x) do{cudaError_t e=(x);if(e){printf("cuda %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
__device__ __forceinline__ void red5(const void* d,const float* s,uint32_t c0,uint32_t c1,uint32_t c2,uint32_t c3,uint32_t c4){
    uint32_t src=(uint32_t)__cvta_generic_to_shared(s);
    asm volatile("cp.reduce.async.bulk.tensor.5d.global.shared::cta.add.tile.bulk_group [%0,{%1,%2,%3,%4,%5}],[%6];\n"
        ::"l"((uint64_t)d),"r"(c0),"r"(c1),"r"(c2),"r"(c3),"r"(c4),"r"(src):"memory");
}
// launch 128 thr; params: which wg-half (0/1) and rowtile (rowbase=rowtile*64). T is the full [TR,128].
__global__ void k(const float* __restrict__ T, const __grid_constant__ CUtensorMap desc,
                  int wgHalf, int rowtile){
    __shared__ __align__(1024) float s0[2048]; __shared__ __align__(1024) float s1[2048];
    int tid=threadIdx.x, w=tid>>5, lane=tid&31;
    int r0=w*16+(lane>>2), r1=r0+8, cc=(lane&3)*2, colbase=wgHalf*64, rowbase=rowtile*64;
    float d[32];
    #pragma unroll
    for(int nt=0;nt<8;nt++){int c=nt*8+cc;
        d[nt*4+0]=T[(rowbase+r0)*128+colbase+c];   d[nt*4+1]=T[(rowbase+r0)*128+colbase+c+1];
        d[nt*4+2]=T[(rowbase+r1)*128+colbase+c];   d[nt*4+3]=T[(rowbase+r1)*128+colbase+c+1];}
    bool even=!(lane&1); int laneHi=lane>>2, ccBit=(lane&3)>>1;
    #pragma unroll
    for(int nt=0;nt<8;nt++){
        float n0=__shfl_xor_sync(~0u,d[nt*4+0],1),n1=__shfl_xor_sync(~0u,d[nt*4+1],1);
        float n2=__shfl_xor_sync(~0u,d[nt*4+2],1),n3=__shfl_xor_sync(~0u,d[nt*4+3],1);
        if(even){int base=((((w*8+laneHi)*8+nt)*2+ccBit)*4);
            s0[base+0]=d[nt*4+0];s0[base+1]=d[nt*4+1];s0[base+2]=n0;s0[base+3]=n1;
            s1[base+0]=d[nt*4+2];s1[base+1]=d[nt*4+3];s1[base+2]=n2;s1[base+3]=n3;}
    }
    __syncthreads(); asm volatile("fence.proxy.async.shared::cta;\n":::"memory");
    if(tid==0){
        // coords {col4=0, ccBit=0, nt=wgHalf*8, laneHi=0/8 (r0/r1), w=rowbase/16}
        red5(&desc, s0, 0,0,(uint32_t)(wgHalf*8),0,           (uint32_t)(rowbase/16));
        red5(&desc, s1, 0,0,(uint32_t)(wgHalf*8),8,           (uint32_t)(rowbase/16));
        asm volatile("cp.async.bulk.commit_group;\ncp.async.bulk.wait_group 0;\n":::"memory");
    }
}
int main(){
    const int TR=128; // 2 row-tiles x 2 wg-halves = 4 half-tiles to place
    float* T=(float*)malloc(TR*128*4); for(int i=0;i<TR*128;i++)T[i]=(float)((i*7)%101)-50.f;
    float *dT,*dG; CK(cudaMalloc(&dT,TR*128*4)); CK(cudaMalloc(&dG,TR*128*4));
    CK(cudaMemcpy(dT,T,TR*128*4,cudaMemcpyHostToDevice)); CK(cudaMemset(dG,0,TR*128*4));
    // full-tensor 5D {col4,ccBit,nt,laneHi,w}; gDim total, gStride elem: col4=1,ccBit=4,nt=8,laneHi=128,w=16*128
    uint64_t gS[5]={4,2,16,16,(uint64_t)TR/16};
    uint64_t gStr[4]={4*4ull,8*4ull,128*4ull,(uint64_t)16*128*4};
    uint32_t bx[5]={4,2,8,8,4}; uint32_t eS[5]={1,1,1,1,1};
    CUtensorMap desc{};
    CUresult r=cuTensorMapEncodeTiled(&desc,CU_TENSOR_MAP_DATA_TYPE_FLOAT32,5,dG,gS,gStr,bx,eS,
        CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if(r!=CUDA_SUCCESS){const char*e;cuGetErrorString(r,&e);printf("encode: %s\n",e);return 1;}
    for(int rt=0;rt<2;rt++)for(int wg=0;wg<2;wg++){ k<<<1,128>>>(dT,desc,wg,rt); CK(cudaDeviceSynchronize()); }
    float* G=(float*)malloc(TR*128*4); CK(cudaMemcpy(G,dG,TR*128*4,cudaMemcpyDeviceToHost));
    int bad=0; for(int i=0;i<TR*128&&bad<10;i++) if(G[i]!=T[i]){printf("mismatch row %d col %d: got %.0f want %.0f\n",i/128,i%128,G[i],T[i]);bad++;}
    printf("%s\n",bad?"FAIL":"PASS: full-tensor 5D descriptor + per-tile coords correct");
    return 0;
}
