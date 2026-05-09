//https://leetgpu.com/playground/55febd78-6c10-4f77-8eca-43bd690b4d8e
#include <stdio.h>
#include <cuda_runtime.h>
__global__
void matmull(const float* M, const float* N, float* Pro, int Width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < Width && col < Width) {
        float P = 0.0f;
        for (int k = 0; k < Width; ++k) {
            P += M[row * Width + k] * N[k * Width + col];
        }
        Pro[row * Width + col] = P;
    }
}
int main() {
    int Width = 3;
    int size = Width * Width * sizeof(float);
    float h_M[] = {1, 2, 3,
                   4, 5, 6,
                   7, 8, 9};

    float h_N[] = {9, 8, 7,
                   6, 5, 4,
                   3, 2, 1};

    float h_Pro[Width * Width] = {0};
    float *d_M, *d_N, *d_Pro;

    cudaMalloc((void**)&d_M, size);
    cudaMalloc((void**)&d_N, size);
    cudaMalloc((void**)&d_Pro, size);
    cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((Width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (Width + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matmull<<<numBlocks, threadsPerBlock>>>(d_M, d_N, d_Pro, Width);

    cudaDeviceSynchronize();
    cudaMemcpy(h_Pro, d_Pro, size, cudaMemcpyDeviceToHost);

    printf("Product:\n");
    for (int i = 0; i < Width; ++i) {
        for (int j = 0; j < Width; ++j) {
            printf("%6.1f ", h_Pro[i * Width + j]);
        }
        printf("\n");
    }
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_Pro);

    return 0;
}