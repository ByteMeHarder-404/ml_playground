#include<cuda_runtime.h>
#include<stdio.h>
__global__ void histo(char* data,unsigned int l,unsigned int* hist){
    unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<l){
        int alp=data[i]-'a';
        if(alp>=0 && alp<26)
            atomicAdd(&(hist[alp/4]),1);
    }
}