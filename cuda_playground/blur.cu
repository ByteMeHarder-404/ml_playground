//https://leetgpu.com/playground/bce9e85d-60be-4464-8bed-5259d59d12af
#include <stdio.h>
#include <cuda_runtime.h>
#define BLUR_SIZE 3 //(7x7) kernel
__global__
void blur(const unsigned char* in, unsigned char* out, int w, int h) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < w && row < h) {
        int sum = 0;
        int pixels = 0;
        for (int rowb = -BLUR_SIZE; rowb <= BLUR_SIZE; ++rowb) {
            for (int colb = -BLUR_SIZE; colb <= BLUR_SIZE; ++colb) {
                int srow = row + rowb;
                int scol = col + colb;
                if (srow >= 0 && srow < h && scol >= 0 && scol < w) {
                    sum += in[srow * w + scol];
                    ++pixels;
                }
            }
        }
        out[row * w + col] = (unsigned char)(sum / pixels);
    }
}
int main() {
    int w = 64;
    int h = 64;
    int size = w * h * sizeof(unsigned char);
    unsigned char *in= (unsigned char*)malloc(size);
    unsigned char *out= (unsigned char*)malloc(size);
    for (int i = 0; i < w * h; ++i)
        in[i] = (unsigned char)(i * 4);
    unsigned char *d_in, *d_out;
    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);
    cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);
    dim3 blockSize(4, 4);
    dim3 gridSize((w + blockSize.x - 1) / blockSize.x,
                  (h + blockSize.y - 1) / blockSize.y);
    blur<<<gridSize, blockSize>>>(d_in, d_out, w, h);
    cudaDeviceSynchronize();
    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);
    printf("Input image:\n");
    for (int r = 0; r < h; ++r) {
        for (int c = 0; c < w; ++c) {
            printf("%3d ", in[r * w + c]);
        }
        printf("\n");
    }
    printf("\nBlurred image:\n");
    for (int r = 0; r < h; ++r) {
        for (int c = 0; c < w; ++c) {
            printf("%3d ", out[r * w + c]);
        }
        printf("\n");
    }
    cudaFree(d_in);
    cudaFree(d_out);
    free(in);
    free(out);

    return 0;
}
