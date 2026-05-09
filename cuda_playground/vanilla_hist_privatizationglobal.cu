#include<cuda_runtime.h>
#include<stdio.h>
#define BIN 7
__global__ void histo(char* data,unsigned int l,unsigned int* hist){
    unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<l){
        int alp=data[i]-'a';
        if(alp>=0 && alp<26)
            atomicAdd(&(hist[blockIdx.x*BINS+alp/4]),1);
    }
    __syncthreads();
    if (blockIdx.x>0){
        for(unsigned int j=threadIdx.x;j<BINS;j+=blockDim.x){
            unsigned int bv=hist[blockIdx.x*BINS+j];
            if(bv>0)
                atomicAdd(&hist[j],bv);
        }
    }
}