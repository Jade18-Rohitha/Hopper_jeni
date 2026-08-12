// Probe: can cuTensorMapEncodeTiled express a de-aliasing fp32 SWIZZLE_128B layout for the dQ reduce?
// The 5.3-way floor is r0/r1 phase-alias (r1=r0+8, phase=row&7 identical). De-alias needs the 64-row
// tile reshaped so the swizzle's low bits see bg=row>>3 (differs for r0/r1) instead of a=row&7.
// That means a >=3D box where the row dim is split [a(8)][bg(8)] with the ordering/strides that put bg
// in the inner swizzle position. Question: which of these does the encoder ACCEPT for FLOAT32 + SW128B?
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>

static const char* es(CUresult r){ const char* e; cuGetErrorString(r,&e); return e; }

int main(){
    cudaFree(0);   // init a primary context via the runtime; driver API calls use it
    const int D=128, S=4096;
    void* ptr=(void*)0x100000000ull;   // dummy aligned device-ish addr (encode only, no launch)

    auto try2d=[&](const char* tag,uint32_t bx0,uint32_t bx1){
        CUtensorMap m{}; uint64_t gS[2]={(uint64_t)D,(uint64_t)S}; uint64_t gStr[1]={(uint64_t)D*4};
        uint32_t bx[2]={bx0,bx1}; uint32_t eStr[2]={1,1};
        CUresult r=cuTensorMapEncodeTiled(&m,CU_TENSOR_MAP_DATA_TYPE_FLOAT32,2,ptr,gS,gStr,bx,eStr,
            CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        printf("2D %-22s box{%u,%u} -> %s\n",tag,bx0,bx1,r==CUDA_SUCCESS?"OK":es(r));
    };
    // 3D: dims (inner->outer) = {col, dimA, dimB}. gSize/gStride/box in that order.
    auto try3d=[&](const char* tag,uint32_t bcol,uint32_t bA,uint32_t bB,
                   uint64_t gcol,uint64_t gA,uint64_t gB,uint64_t sA,uint64_t sB,CUtensorMapSwizzle sw){
        CUtensorMap m{}; uint64_t gS[3]={gcol,gA,gB}; uint64_t gStr[2]={sA,sB};
        uint32_t bx[3]={bcol,bA,bB}; uint32_t eStr[3]={1,1,1};
        CUresult r=cuTensorMapEncodeTiled(&m,CU_TENSOR_MAP_DATA_TYPE_FLOAT32,3,ptr,gS,gStr,bx,eStr,
            CU_TENSOR_MAP_INTERLEAVE_NONE,sw,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        printf("3D %-22s box{%u,%u,%u} gStr{%llu,%llu} -> %s\n",tag,bcol,bA,bB,
               (unsigned long long)sA,(unsigned long long)sB,r==CUDA_SUCCESS?"OK":es(r));
    };

    printf("== 2D baselines ==\n");
    try2d("v44 32x64",32,64);
    try2d("64x64(known-bad)",64,64);

    printf("\n== 3D de-alias candidates (col=32) ==\n");
    // A=a(row&7,stride D*4), B=bg(row>>3,stride 8*D*4): smem_row = B*8+A -> &7=A = ALIAS (monotonic strides)
    try3d("A=a B=bg mono",32,8,8, D,8,S/8, (uint64_t)D*4,(uint64_t)8*D*4,CU_TENSOR_MAP_SWIZZLE_128B);
    // swap: A=bg(stride 8D) B=a(stride D): smem_row = A? decreasing stride (dim1>dim2) -> de-alias if accepted
    try3d("A=bg B=a decr",32,8,8, D,S/8,8, (uint64_t)8*D*4,(uint64_t)D*4,CU_TENSOR_MAP_SWIZZLE_128B);
    // col=64 inner variants
    try3d("col64 A=a B=bg",64,8,8, D,8,S/8, (uint64_t)D*4,(uint64_t)8*D*4,CU_TENSOR_MAP_SWIZZLE_128B);
    try3d("col64 A=bg B=a decr",64,8,8, D,S/8,8, (uint64_t)8*D*4,(uint64_t)D*4,CU_TENSOR_MAP_SWIZZLE_128B);
    // SWIZZLE_NONE 3D (padding-style, no swizzle) for reference
    try3d("NONE A=bg B=a decr",32,8,8, D,S/8,8, (uint64_t)8*D*4,(uint64_t)D*4,CU_TENSOR_MAP_SWIZZLE_NONE);
    printf("\ndone\n");
    return 0;
}
