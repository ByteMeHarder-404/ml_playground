#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
__device__ float sig(float x){
    return 1.0f/(1.0f + expf(x));
}
__global__ void swigelu(float* A,float* B.foat* Y,int N){
    int i=blockDim.x*blockIdx+threadIdx.x;
    int stride =blockDim.x*gridDim.x;
    for (int j=idx;i<N;i+=stride){
        float b=B[i];
        float swis=b*sig(b);
        Y[i]=A[i]*swis;
    }
}