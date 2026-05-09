#include<cuda_runtime.h>
#include<stdio.h>
#define CF 2
__global__ void sumred(float* input,float* output){
    __shared__ float ins[1024];
    unsigned int seg=CF*2*blockDim.x*blockIdx.x;
    unsigned int i=seg+threadIdx.x;
    unsigned int t=threadIdx.x;
    float s=input[i];
    for(unsigned int tile=1;tile<CF*2;++tile)
        s+=input[i+tile*1024];
    ins[t]=s;
    for(unsigned int stride=blockDim.x/2;stride>=1;stride/=2){
        if(t<stride)
            ins[t]+=ins[t+stride];
        __syncthreads();
    }
    if (t==0)
        atomicAdd(output,ins[0]);
}