https://leetgpu.com/playground/b8dda1df-616e-413b-af49-b239b7fccfb4
#include<cuda_runtime.h>
#include<stdio.h>
#define FILTER 2
#define INTILE 32
#define OUTTILE ((INTILE)-2*(FILTER))

__constant__ float F_c[2*FILTER+1][2*FILTER+1];
__global__ void conv(float* N,float* P, int w,int h){
    int col=blockIdx.x*OUTTILE+threadIdx.x-FILTER;
    int row=blockIdx.y*OUTTILE+threadIdx.y-FILTER;
    __shared__ float Ns[INTILE][INTILE];
    if(row>=0 && row<h && col>=0 && col<w){
        Ns[threadIdx.y][threadIdx.x]=N[row*w+col];
    }
    else{
        Ns[threadIdx.y][threadIdx.x]=0.0f;
    }
    __syncthreads();
    int tileCol=threadIdx.x-FILTER;
    int tileRow=threadIdx.y-FILTER;
    if (col>=0 && col<w && row>=0 && row<h){
        if (tileCol>=0 && tileCol<OUTTILE && tileRow>=0 && tileRow<OUTTILE){
            float Pvalue=0.0f;
            for(int frow=0;frow<2*FILTER+1;frow++){
                for(int fcol=0;fcol<2*FILTER+1;fcol++){
                    Pvalue+=F_c[frow][fcol]*Ns[tileRow+frow][tileCol+fcol];
                }
            }
            P[row*w+col]=Pvalue;
        }
    }
}
int run(float* h_N, float* h_P, int w, int h)
{
    float* d_N;
    float* d_P;
    cudaMalloc(&d_N, w * h * sizeof(float));
    cudaMalloc(&d_P, w * h * sizeof(float));
    cudaMemcpy(d_N, h_N, w * h * sizeof(float), cudaMemcpyHostToDevice);
    dim3 blockDim(INTILE, INTILE);
    dim3 gridDim(
        (w + OUTTILE - 1) / OUTTILE,
        (h + OUTTILE - 1) / OUTTILE
    );
    conv<<<gridDim, blockDim>>>(d_N, d_P, w, h);
    cudaDeviceSynchronize();
    cudaMemcpy(h_P, d_P, w * h * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_N);
    cudaFree(d_P);

    return 0;
}