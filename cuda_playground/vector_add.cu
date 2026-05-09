//https://leetgpu.com/playground/5d0854bd-9d90-4225-a76d-88d1f9777db5
#include<stdio.h>
#include<cuda_runtime.h>
__global__
void vecAddKernel(float* A,float*B,float*C,int n){
    int i=threadIdx.x+blockDim.x*blockIdx.x;
    if(i<n){
        C[i]=A[i]+B[i];
    }
}
void vecAdd(float* A_h,float* B_h,float* C_h,int n){
    int size=n*sizeof(float);
    float* A_d,*B_d,*C_d;
    cudaMalloc((void**)&A_d,size);
    cudaMalloc((void**)&B_d,size);
    cudaMalloc((void**)&C_d,size);
    cudaMemcpy(A_d,A_h,size,cudaMemcpyHostToDevice);
    cudaMemcpy(B_d,B_h,size,cudaMemcpyHostToDevice);
    vecAddKernel<<<ceil(n/256.0),256>>>(A_d,B_d,C_d,n);
    cudaDeviceSynchronize();
    cudaMemcpy(C_h,C_d,size,cudaMemcpyDeviceToHost);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}
int main(){
    int n=1000;
    float *A_h,*B_h,*C_h;
    A_h=(float*)malloc(n*sizeof(float));
    B_h=(float*)malloc(n*sizeof(float));
    C_h=(float*)malloc(n*sizeof(float));
    for (int i=0;i<n;i++){
        A_h[i]=(float)i;
        B_h[i]=(float)(i*2);
    }
    vecAdd(A_h,B_h,C_h,n);
    for (int i=0;i<10;i++){
        printf("%.1f + %.1f=%.1f\n",A_h[i],B_h[i],C_h[i]);
    }
    free(A_h);
    free(B_h);
    free(C_h);
    return 0;
}
