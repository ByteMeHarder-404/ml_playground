//https://leetgpu.com/playground/b4d5c3e8-4f81-4f4d-b1b2-7cc1ead054ec
#include <cuda_runtime.h>
__global__
void conv(float* inn,float* out,int W,int H,int r,float* filter){
    int iTW=blockDim.x;
    int iTH=blockDim.y;
    int Ox=iTW-2*r;
    int Oy=iTH-2*r;
    int outx=blockIdx.x*Ox;
    int outy=blockIdx.y*Oy;

    int inx=outx-r;
    int iny=outy-r;

    int tx=threadIdx.x;
    int ty=threadIdx.y;

    int gx=inx+tx;
    int gy=iny+ty;

    extern __shared__ float mem[];
    int pit=iTW+1;
    float* tile=mem;
    float val=0.0f;
    if(gx>=0 && gx<W && gy>=0 && gy<H){
        val=inn[gy*W+gx];
    }
    tile[ty*pit+tx]=val;
    __syncthreads();
    if(!((tx>=r)&&(tx<r+Ox)&&(ty>=r)&&(ty<r+Oy))){
        return;
    }
    int cx=tx;
    int cy=ty;
    int outtx=outx+(tx-r);
    int outty=outy+(ty-r);
    float sum=0.0f;

    int F=2*r+1;
    for(int z=-r;z<=r;++z){
        int sy=cy+z;
        int frow=(z+r)*F;
        for(int j=-r;j<=r;++j){
            int sx=cx+j;
            float a=tile[sy*pit+sx];
            float w=filter[frow+(j+r)];
            sum+=a*w;
        }
    }
    if(outtx>=0 &&outtx<W && outty>=0 && outty<H){
        out[outty*W+outtx]=sum;
    }
}