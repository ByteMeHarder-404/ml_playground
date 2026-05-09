//https://leetgpu.com/playground/027ba99e-84d6-47c8-8c45-4dc1b1dc1363
#include <stdio.h>
#include<cuda_runtime.h>
#define TILE 16
__global__ void matmul(float *M,float *N,float* P,int W){
    extern __shared__ float shr[];
    float *Mds=(float*)shr;
    float *Nds=(float*)shr+TILE*TILE;

    int bx=blockIdx.x; int by=blockIdx.y;
    int tx=threadIdx.x;int ty=threadIdx.y;
    int R=by*TILE+ty;
    int C=bx*TILE+tx;
    float pro=0;

    for(int i=0;i<ceil(W/(float)TILE);i++){
        if((R<W)&&(i*TILE+tx)<W)
            Mds[ty*TILE+tx]=M[R*W+i*TILE+tx];
        else Mds[ty*TILE+tx]=0.0f;
        if ((i*TILE+ty)<W && C<W)
            Nds[ty*TILE+tx]=N[(i*TILE+ty)*W+C];
        else Nds[ty*TILE+tx]=0.0f;
        __syncthreads();
        for (int k=0;k<TILE;++k){
            pro+=Mds[ty*TILE+k]*Nds[k*TILE+tx];
        }
        __syncthreads();
    }
    if (R<W && C<W)
        P[R*W+C]=pro;
}
