#include<cuda_runtime.h>
#include<stdio.h>
#define BINS 7
__global__ void hist(char* data,unsigned int l,unsigned int* hist){
    __shared__ unsigned int hs[BINS];
    for(unsigned int j=threadIdx.x;j<BINS;j+=blockDim.x)
        hs[j]=0u;
    __syncthreads();
    unsigned int acc=0;
    int pre=-1;
    unsigned int tid=blockIdx.x*blockDim.x+threadIdx.x;
    for(unsigned int i=tid;i<l;i+=blockDim.x*gridDim.x){
        int alp=data[i]-'a';
        if(alp>=0 && alp<26){
            int b=alp/4;
            if(b==pre)
                ++acc;
            else{
                if(acc>0)
                    atomicAdd(&(hs[pre]),acc);
                acc=1;
                pre=b;
            }
        }
    }
    if(acc>0)
        atomicAdd(&(hs[pre]),acc);
    __syncthreads();
    for(unsigned int b=threadIdx.x;b<BINS;b+=blockDim.x){
        unsigned int bv=hs[b];
        if(bv>0)
            atomicAdd(&(hist[b]),bv);
    }
}