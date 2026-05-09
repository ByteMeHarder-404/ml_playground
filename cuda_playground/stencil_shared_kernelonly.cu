#include<cuda_runtime.h>
#define TILE 18
__global__ void stencil(float* in,float* out,unsigned int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x-1;
    int j=blockIdx.y*blockDim.y+threadIdx.y-1;
    int k=blockIdx.z*blockDim.z+threadIdx.z-1;
    __shared__ float ins[TILE][TILE][TILE];
    if(i>=0&&i<n&&j>=0&&j<n&&k>=0&&k<n){
        ins[threadIdx.z][threadIdx.y][threadIdx.x]=in[i+j*n+k*n*n];
    }
    __syncthreads();
    if(i>=1&&i<n-1&&k>=1&&j<n-1&&j>=1&&k<n-1){
        if(threadIdx.x>=1&&threadIdx.x<TILE-1&&threadIdx.y>=1&&threadIdx.y<TILE-1&&threadIdx.z>=1&&threadIdx.z<TILE-1)
            out[i+j*n+k*n*n]=c0*ins[threadIdx.z][threadIdx.y][threadIdx.x]
                            +c1*ins[threadIdx.z][threadIdx.y][threadIdx.x-1]
                            +c2*ins[threadIdx.z][threadIdx.y][threadIdx.x+1]
                            +c3*ins[threadIdx.z][threadIdx.y-1][threadIdx.x]
                            +c4*ins[threadIdx.z][threadIdx.y+1][threadIdx.x]
                            +c5*ins[threadIdx.z-1][threadIdx.y][threadIdx.x]
                            +c6*ins[threadIdx.z+1][threadIdx.y][threadIdx.x];
    }
}