#include<cuda_runtime.h>
__global__ void stencil(float* in,float* out,unsigned int n){
    unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int j=blockIdx.y*blockDim.y+threadIdx.y;
    unsigned int k=blockIdx.z*blockDim.z+threadIdx.z;
    if(i>=1&&i<n&&k>=1&&j<n&&j>=1&&k<n){
        out[i+j*n+k*n*n]=c0*in[i+j*n+k*n*n]+c1*in[(i-1)+j*n+k*n*n]+c2*in[(i+1)+j*n+k*n*n]+
                        c3*in[i+(j+1)*n+k*n*n]+c4*in[i+(j-1)*n+k*n*n]+c5*in[i+j*n+(k+1)*n*n]+c6*in[i+j*n+(k-1)*n*n];
    }
}