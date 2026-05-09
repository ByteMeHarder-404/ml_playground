#include <cuda_runtime.h>
#include <stdio.h>
__global__
void conv1d(const float* A, const float* B, float* C, int N, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        float sum = 0.0f;
        int half = (K - 1) / 2;

        for (int j = -half; j <= half; ++j) {
            int a_idx = i + j;
            if (a_idx >= 0 && a_idx < N) {
                sum += A[a_idx] * B[j + half];
            }
        }

        C[i] = sum;
    }
}
extern "C"
void solution(const float* A, const float* B, float* C, size_t N, size_t K) {
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    conv1d<<<blocks, threadsPerBlock>>>(
        A,
        B,
        C,
        static_cast<int>(N),
        static_cast<int>(K)
    );
}

