#include<cuda_runtime.h>
#include<stdio.h>
__global__ void sumred(float* input,float* output){
    __shared__ float ins[1024];
    unsigned int seg=2*blockDim.x*blockIdx.x;
    unsigned int i=seg+threadIdx.x;
    ins[threadIdx.x]=input[i]+input[i+blockDim.x];
    for (unsigned int stride=blockDim.x/2;stride>=1;stride/=2){
        __syncthreads();
        if(threadIdx.x<stride)
            ins[threadIdx.x]+=ins[threadIdx.x+stride];
    }
    if(threadIdx.x==0){
        atomicAdd(output,ins[0]);
    }
}