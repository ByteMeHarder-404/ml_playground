#include<cuda_runtime.h>
#include<stdio.h>
__global__ void sumred(float* input,float* output){
    unsigned int i=2*threadIdx.x;
    for(unsigned int stride=1;stride<blockDim.x;stride*=2)
        input[i]+=input[i+stride];
    __syncthreads();
    if(threadIdx.x==0)
        *output=input[0];
}