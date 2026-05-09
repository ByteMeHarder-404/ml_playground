#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#define BLOCK_SIZE 16
__global__ void flash_attention_forward(const float* __restrict__ Q,const float* __restrict__ K,const float* __restrict__ V,float* __restrict__ O,int N,int d) {
    int q_idx = blockIdx.x;
    int tx = threadIdx.x;
    __shared__ float K_tile[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float V_tile[BLOCK_SIZE][BLOCK_SIZE];
    float q_reg[BLOCK_SIZE];
    for (int i = 0; i < d; i++) {
        q_reg[i] = Q[q_idx * d + i];
    }
    float m_i = -INFINITY;
    float l_i = 0.0f;
    float o_reg[BLOCK_SIZE] = {0.0f};
    for (int kv_start = 0; kv_start < N; kv_start += BLOCK_SIZE) {
        if (tx < BLOCK_SIZE && kv_start + tx < N) {
            for (int j = 0; j < d; j++) {
                K_tile[tx][j] = K[(kv_start + tx) * d + j];
                V_tile[tx][j] = V[(kv_start + tx) * d + j];
            }
        }
        __syncthreads();
        float scores[BLOCK_SIZE];
        for (int j = 0; j < BLOCK_SIZE; j++) {
            float dot = 0.0f;
            for (int k = 0; k < d; k++) {
                dot += q_reg[k] * K_tile[j][k];
            }
            scores[j] = dot / sqrtf((float)d);
        }
        float m_ij = -INFINITY;
        for (int j = 0; j < BLOCK_SIZE; j++) {
            m_ij = fmaxf(m_ij, scores[j]);
        }
        float m_new = fmaxf(m_i, m_ij);
        float l_ij = 0.0f;
        float exp_scores[BLOCK_SIZE];
        for (int j = 0; j < BLOCK_SIZE; j++) {
            exp_scores[j] = expf(scores[j] - m_new);
            l_ij += exp_scores[j];
        }
        float l_new = expf(m_i - m_new) * l_i + l_ij;
        for (int j = 0; j < BLOCK_SIZE; j++) {
            for (int k = 0; k < d; k++) {
                o_reg[k] += exp_scores[j] * V_tile[j][k];
            }
        }
        for (int k = 0; k < d; k++) {
            o_reg[k] = o_reg[k] / l_new;
        }
        m_i = m_new;
        l_i = l_new;
        __syncthreads();
    }
    for (int i = 0; i < d; i++) {
        O[q_idx * d + i] = o_reg[i];
    }
}