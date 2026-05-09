#include <cuda_runtime.h>
#include <cuda_runtime.h>
#include <math.h>

#define TILE 16
#define PI 3.14159265359f

__global__
void gellu(const float* input, float* output, size_t n, size_t m)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= n || col >= m) return;

    float x = input[row * m + col];

    output[row * m + col] =
        0.5f * x *
        (1.0f + tanhf(sqrtf(2.0f / PI) * (x + 0.044715f * x * x * x)));
}
// Note: input, output are all device pointers to float32 arrays
extern "C" void solution(const float* input, float* output, size_t n, size_t m) {
    dim3 block(TILE, TILE);
dim3 grid((n + TILE - 1) / TILE,
          (m + TILE - 1) / TILE);

gellu<<<grid, block>>>(input, output, n, m);

}