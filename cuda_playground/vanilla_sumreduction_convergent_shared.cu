#include<cuda_runtime.h>
#include<stdio.h>
__global__ void sumred(float* input,float* output){
    extern __shared__ float ins[];
    unsigned int i=threadIdx.x;
    ins[i]=input[i]+input[i+blockDim.x];
    __syncthreads();
    for(unsigned int stride=blockDim.x/2;stride>=1;stride/=2){
        if(threadIdx.x<stride)
            ins[i]+=ins[i+stride];
        __syncthreads();
    }
    if(threadIdx.x==0)
        *output=ins[0];
}