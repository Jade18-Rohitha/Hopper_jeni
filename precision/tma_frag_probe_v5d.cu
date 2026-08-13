// Probe V5d: cuDNN's 4D-reduce geometry — 8 (here 4 per D-half) contiguous [row][col16] D-slab blocks,
// each reduced by a COMPACT descriptor (inner run = 16 contiguous cols = 64B), vs V5c's 5D-strided 16B scatter.
// Validates: (a) bit-exact global reconstruction, (b) the 32-lane coalesced row-major store is conflict-free.
//
// mma-C fragment: lane(laneHi=lane>>2, cc=lane&3) holds per nt: (r0,c),(r0,c+1),(r1,c),(r1,c+1);
//   r0=w*16+laneHi, r1=r0+8, c=nt*8+cc*2, nt=0..7.
// Block s (0..3) = D-cols [s*16 : s*16+16] x 64 rows, laid out dense [row][col16]. 4 blocks = 64x64 tile.
// TRANSPOSE (cuDNN repack): store lane l writes block[s] at [row=w*16+h*8+(l>>2)][col16=q*4], q=l&3 -> a
//   32-lane STS.128 (512B = 4 phases -> conflict-free). Each target float4 = 4 contiguous slab-cols gathered
//   from fragment lanes (laneHi, cc_lo=(q&1)*2) and (laneHi, cc_hi=cc_lo+1), r-half via h. Brute 32-shfl
//   gather (probe only — cost doesn't affect the store-conflict / correctness metrics measured here).
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#define CK(x) do{cudaError_t e=(x);if(e){printf("cuda err %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)

__device__ __forceinline__ void red2(const void* d,const float* s,uint32_t c0,uint32_t c1){
    uint32_t src=(uint32_t)__cvta_generic_to_shared(s);
    asm volatile("cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group [%0,{%1,%2}],[%3];\n"
        ::"l"((uint64_t)d),"r"(c0),"r"(c1),"r"(src):"memory");
}
__global__ void k(const float* __restrict__ T, const __grid_constant__ CUtensorMap desc){
    __shared__ __align__(1024) float sB[4][1024];   // 4 D-slab blocks, dense [row64][col16]
    int tid=threadIdx.x, w=tid>>5, lane=tid&31, laneHi=lane>>2, q=lane&3;
    float d[32];
    #pragma unroll
    for(int nt=0;nt<8;nt++){int r0=w*16+laneHi,r1=r0+8,c=nt*8+(lane&3)*2;
        d[nt*4+0]=T[r0*64+c]; d[nt*4+1]=T[r0*64+c+1];
        d[nt*4+2]=T[r1*64+c]; d[nt*4+3]=T[r1*64+c+1];}
    #pragma unroll
    for(int h=0;h<2;h++){
        #pragma unroll
        for(int s=0;s<4;s++){
            int nt=2*s+(q>>1), cc_lo=(q&1)*2;
            int srcA=laneHi*4+cc_lo, srcB=laneHi*4+cc_lo+1, regidx=nt*4+h*2;
            float a0=0,a1=0,b0=0,b1=0;
            #pragma unroll
            for(int jj=0;jj<32;jj++){
                float tA=__shfl_sync(~0u,d[jj],srcA), tB=__shfl_sync(~0u,d[jj],srcB);
                if(jj==regidx){a0=tA;b0=tB;} else if(jj==regidx+1){a1=tA;b1=tB;}
            }
            int row=w*16+h*8+laneHi, base=row*16+q*4;
            sB[s][base+0]=a0; sB[s][base+1]=a1; sB[s][base+2]=b0; sB[s][base+3]=b1;
        }
    }
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;\n":::"memory");
    if(tid==0){
        #pragma unroll
        for(int s=0;s<4;s++) red2(&desc, sB[s], (uint32_t)(s*16), 0);   // coord = D-col slab offset
        asm volatile("cp.async.bulk.commit_group;\n":::"memory");
        asm volatile("cp.async.bulk.wait_group 0;\n":::"memory");
    }
}
int main(){
    float* T=(float*)malloc(64*64*4); for(int i=0;i<64*64;i++)T[i]=(float)(i%97)-48.f;
    float *dT,*dG; CK(cudaMalloc(&dT,64*64*4)); CK(cudaMalloc(&dG,64*64*4));
    CK(cudaMemcpy(dT,T,64*64*4,cudaMemcpyHostToDevice)); CK(cudaMemset(dG,0,64*64*4));
    // 2D descriptor over the 64x64 tile (row-major, elem(row,col)=row*64+col). dim0=col(fast), dim1=row.
    uint64_t gS[2]={64,64}; uint64_t gStr[1]={64*4ull};        // row stride bytes
    uint32_t bx[2]={16,64}; uint32_t eS[2]={1,1};              // box = 16 cols x 64 rows (inner run 64B)
    CUtensorMap desc{};
    CUresult r=cuTensorMapEncodeTiled(&desc,CU_TENSOR_MAP_DATA_TYPE_FLOAT32,2,dG,gS,gStr,bx,eS,
        CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if(r!=CUDA_SUCCESS){const char*e;cuGetErrorString(r,&e);printf("encode: %s\n",e);exit(1);}
    k<<<1,128>>>(dT,desc); CK(cudaDeviceSynchronize());
    float* G=(float*)malloc(64*64*4); CK(cudaMemcpy(G,dG,64*64*4,cudaMemcpyDeviceToHost));
    int bad=0; for(int i=0;i<64*64&&bad<10;i++) if(G[i]!=T[i]){printf("mismatch row %d col %d: got %.0f want %.0f\n",i/64,i%64,G[i],T[i]);bad++;}
    printf(bad?"FAIL\n":"PASS: V5d 4-block [row][col16] store + compact 2D reduce maps correctly\n");
    return 0;
}
