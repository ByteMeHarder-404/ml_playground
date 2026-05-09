#include<cuda_runtime.h>
#include<stdio.h>
#define BINS 7
#define CF 3
__global__ void histogram(char* data,unsigned int l,unsigned int* hist){
    __shared__ unsigned int hs[BINS];
    for (unsigned int j=threadIdx.x;j<BINS;j+=blockDim.x)
        hs[j]=0u;
    __syncthreads();
    unsigned int tid=blockIdx.x*blockDim.x+threadIdx.x;
    for(unsigned int i=tid*CF;i<min((tid+1)*CF,l);++i){
        int alp=data[i]-'a';
        if(alp>=0 && alp<26)
            atomicAdd(&hs[alp/4],1);
    }
    __syncthreads();
    for(unsigned int k=threadIdx.x;k<BINS;k+=blockDim.x){
        unsigned int bv=hs[k];
        if(bv>0){
            atomicAdd(&(hist[k]),bv);
        }
    }
}