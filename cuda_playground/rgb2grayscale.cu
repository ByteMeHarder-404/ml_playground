//https://leetgpu.com/playground/cb7fc5ef-6d6f-4bd4-8b78-98b1d7ae6164
#include<stdio.h>
#include<cuda_runtime.h>
#define CHANNELS 3
__global__
void c2Gconv(unsigned char* Out,unsigned char* In,int w,int h){
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    int row=blockIdx.y*blockDim.y+threadIdx.y;

    if (col<w && row<h){
        int goff=row*w+col;
        int roff=goff*CHANNELS;
        unsigned char r=In[roff];
        unsigned char g=In[roff+1];
        unsigned char b=In[roff+2];
        Out[goff]=0.21f*r+0.71f*g+0.07f*b;
    }
}

int main(){
    int w=8;
    int h=8;
    int rimg=w*h*CHANNELS;
    int gimg=w*h;
    unsigned char* In=(unsigned char*)malloc(rimg);
    unsigned char* Out=(unsigned char*)malloc(gimg);

    for(int i =0;i<rimg;i++){
        In[i]=i%256;
    }
    unsigned char* In_d,*Out_d;
    cudaMalloc((void**)&In_d,rimg);
    cudaMalloc((void**)&Out_d,gimg);
    cudaMemcpy(In_d,In,rimg,cudaMemcpyHostToDevice);
    dim3 block(16,16);
    dim3 grid((w+block.x-1)/block.x,(h+block.y-1)/block.y);
    c2Gconv<<<grid,block>>>(Out_d,In_d,w,h);
    cudaDeviceSynchronize();
    cudaMemcpy(Out,Out_d,gimg,cudaMemcpyDeviceToHost);
    printf("Grayscale output (first few values):\n");
    for (int i = 0; i < 10 && i < gimg; i++) {
        printf("%d ", Out[i]);
    }
    printf("\n");
    cudaFree(Out_d);
    cudaFree(In_d);
    free(In);
    free(Out);

}
