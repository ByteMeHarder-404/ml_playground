#include <cuda_runtime.h>
#include <math.h>

__global__
void leakyrelu(const float* input, float alpha,
               float* output, size_t n, size_t m)
{
    int row = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    int col = 4 * (blockIdx.y * blockDim.y + threadIdx.y);

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            int r = row + i;
            int c = col + j;

            if (r < n && c < m) {
                float x = input[r * m + c];
                output[r * m + c] = fmaxf(alpha * x, x);
            }
        }
    }
}

extern "C"
void solution(const float* input, float alpha,
              float* output, size_t n, size_t m)
{
    dim3 block(16, 16);

    // each thread handles 4 rows & 4 cols
    dim3 grid((n + 4 * block.x - 1) / (4 * block.x),
              (m + 4 * block.y - 1) / (4 * block.y));

    leakyrelu<<<grid, block>>>(input, alpha, output, n, m);
}
