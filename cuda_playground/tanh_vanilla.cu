#include <cuda_runtime.h>
#include <math.h>
__global__
void tanh(const float* input, float* output, size_t n, size_t m){
    int r = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    int c = 2 * (blockIdx.y * blockDim.y + threadIdx.y);

    for (int dr = 0; dr < 2; dr++) {
        for (int dc = 0; dc < 2; dc++) {
            int rr = r + dr;
            int cc = c + dc;
            if (rr < m && cc < n) {
                int idx = rr * n + cc;
                output[idx] = tanhf(input[idx]);
        }
    }
}

    }

// Note: input, output are all device pointers to float32 arrays
extern "C" void solution(const float* input, float* output, size_t n, size_t m) {
    dim3 block(16, 16);
    dim3 grid((m + block.x - 1) / block.x,
    (n + block.y - 1) / block.y);
    tanh<<<grid, block>>>(input,output, n, m);

}