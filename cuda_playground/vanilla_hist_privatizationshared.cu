#include<cuda_runtime.h>
#include<stdio.h>
#define BINS 7
__global__ void histogram(char* data,unsigned int l,unsigned int* hist){
    __shared__ unsigned int hs[BINS];
    for(unsigned int j=threadIdx.x;j<BINS;j+=blockDim.x){
        hs[j]=0u;
    }
    __syncthreads();
    unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<l){
        int alp=data[i]-'a';
        if(alp>=0 && apl<26)
            atomicAdd(&hs[alp/4],1);
    }
    __syncthreads();
    for(unsigned int j=threadIdx.x;j<BINS;j+=blockDim.x){
        unsigned int bv=hs[j];
        if (bv>0){
            atomicAdd(&(hist[j]),bv);
        }
    }
}