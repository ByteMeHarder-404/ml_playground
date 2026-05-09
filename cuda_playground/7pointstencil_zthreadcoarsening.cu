#include<cuda_runtime.h>
#include<stdio.h>
#define INTILE 18
#define OUTTILE 16
__constant__ float c0, c1, c2, c3, c4, c5, c6;
__global__ void stencil(const float* inn,float* out,unsigned int n){
    int start=blockIdx.z*OUTTILE;
    int j=blockIdx.y*OUTTILE+threadIdx.y-1;
    int k=blockIdx.x*OUTTILE+threadIdx.x-1;
    __shared__ float ps[INTILE][INTILE];
    __shared__ float cs[INTILE][INTILE];
    __shared__ float ns[INTILE][INTILE];
    if(start-1>=0 &&start-1<n && j>=0 &&j<n && k>=0 && k<n)
        ps[threadIdx.y][threadIdx.x]=inn[(start-1)*n*n+j*n+k];
    if(start>=0 &&start<n && j>=0 &&j<n && k>=0 && k<n)
        cs[threadIdx.y][threadIdx.x]=inn[(start)*n*n+j*n+k];
    __syncthreads();
    for(int i=start;i<start+OUTTILE;++i){
        if(i+1>=0 && i+1<n && j>=0 && j<n && k>=0 && k<n)
            ns[threadIdx.y][threadIdx.x]=inn[(i+1)*n*n+j*n+k];
        __syncthreads();
        if(i>=1&&i<n-1&&j>=1&&j<n-1&&k>=1&&k<n-1){
            if(threadIdx.y>=1 && threadIdx.y<INTILE-1 &&threadIdx.x>=1 && threadIdx.x<INTILE-1)
                out[i*n*n+j*n+k]=c0*cs[threadIdx.y][threadIdx.x]+c1*cs[threadIdx.y][threadIdx.x-1]+
                                c2*cs[threadIdx.y][threadIdx.x+1]+c3*cs[threadIdx.y-1][threadIdx.x]+
                                c4*cs[threadIdx.y+1][threadIdx.x]+c5*ps[threadIdx.y][threadIdx.x]+
                                c6*ns[threadIdx.y][threadIdx.x];
        }
        __syncthreads();
        ps[threadIdx.y][threadIdx.x]=cs[threadIdx.y][threadIdx.x];
        cs[threadIdx.y][threadIdx.x]=ns[threadIdx.y][threadIdx.x];
    }
}
void launch(const float* d_in,float* d_out,unsigned int n,float h_c0, float h_c1, float h_c2,float h_c3, float h_c4, float h_c5, float h_c6)
    {
    cudaMemcpyToSymbolAsync(c0, &h_c0, sizeof(float), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbolAsync(c1, &h_c1, sizeof(float), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbolAsync(c2, &h_c2, sizeof(float), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbolAsync(c3, &h_c3, sizeof(float), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbolAsync(c4, &h_c4, sizeof(float), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbolAsync(c5, &h_c5, sizeof(float), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbolAsync(c6, &h_c6, sizeof(float), 0, cudaMemcpyHostToDevice);
    dim3 blockDim(INTILE, INTILE);
    dim3 gridDim((n + OUTTILE - 1) / OUTTILE,(n + OUTTILE - 1) / OUTTILE,(n + OUTTILE - 1) / OUTTILE);
    stencil<<<gridDim, blockDim, 0>>>(d_in,d_out,n);
}