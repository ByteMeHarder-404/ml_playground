#include<cuda_runtime.h>
#define TILE 32
#define FILTER 2
__constant__ float F_c[2*FILTER+1][2*FILTER+1];
__global__ void conv(float* N,float* P,int w,int h){
    int col=blockIdx.x*TILE+threadIdx.x;
    int row=blockIdx.y*TILE+threadIdx.y;
    __shared__ float Ns[TILE][TILE];
    if(row<h && col<w){
        Ns[threadIdx.y][threadIdx.x]=N[row*w+col];
    }
    else{
        Ns[threadIdx.y][threadIdx.x]=0.0f;
    }
    __syncthreads();
    if(col<w && row<h){
        float Pv=0.0f;
        for(int frow=0;frow<2*FILTER+1;frow++){
            for(int fcol=0;fcol<2*FILTER+1;fcol++){
                if(threadIdx.x-FILTER+fcol>=0 && threadIdx.x-FILTER+fcol<TILE &&
                threadIdx.y-FILTER+frow>=0 && threadIdx.y-FILTER+frow<TILE){
                    Pv+=F_c[frow][fcol]*Ns[threadIdx.y+frow][threadIdx.x+fcol];
                }
                else{
                    if(row-FILTER+frow>=0 && row-FILTER+frow<h
                    &&col-FILTER+fcol>=0 && col-FILTER+fcol<w){
                        Pv+=F_c[frow][fcol]*N[(row-FILTER+frow)*w+col-FILTER+fcol];
                    }
                }
            }
            P[row*w+col]=Pv;
        }
    }
}