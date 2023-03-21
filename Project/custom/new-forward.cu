#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#define TILE_WIDTH 16 // for general one
#define TILE_WIDTH_LAYER1 8 // for map size 4
#define TILE_WIDTH_LAYER2 16 //for map size 16
#define Const_MEM_SIZE 4096

// OPtimization 1: Weight matrix (kernel values) in constant memory
// OPtimization 2: Tiled shared memory convolution
// Optimization 3: Shared memory matrix multiplication and input matrix unrolling
// Optimization 4: Multiple kernel implementations for different layer sizes
// Optimization 5: Tune parameters
// Optimization 6: Tuning with restrict and loop unrolling
__constant__ float ConstMem[Const_MEM_SIZE];


// Opt5.
__global__ void conv_forward_kernel_opt_layer2(float *output, const float *input, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    int sharedWidth = TILE_WIDTH_LAYER2 + K - 1;
    extern __shared__ float shared[];

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    // #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int W_grid = ceil(1.0*Width/TILE_WIDTH_LAYER2);
    int m = blockIdx.x; //indicate which feature map it belongs to
    int h = (blockIdx.y / W_grid) * TILE_WIDTH_LAYER2 + ty;
    int w = (blockIdx.y % W_grid) * TILE_WIDTH_LAYER2 + tx;
    // int image = blockIdx.z;

    float acc = 0.0f;
    // Optimization 2
    for (int c = 0; c < Channel; c++) // sum over all input channels
    { 
        for(int i = ty; i < sharedWidth; i += TILE_WIDTH_LAYER2)
        {
            for(int j = tx; j < sharedWidth; j += TILE_WIDTH_LAYER2)
            {
                // int i1 = h+i-ty;
                // int i0 = w+j-tx;
                if ((h+i-ty) < Height && (w+j-tx) < Width)
                {
                    shared[c*sharedWidth*sharedWidth+i*sharedWidth+j] = in_4d(blockIdx.z, c, h+i-ty, w+j-tx);
                }
            }
        }
    }
    __syncthreads();

    if (h<Height_out && w<Width_out)
    {
        for (int c = 0; c < Channel; c++)
        {
            for (int p = 0; p < K; p++) // loop over KxK filter
            {
                // for (int q = 0; q < K; q++) 
                // {
                    // int row = ty+p;
                    // int col = tx+q;
                    // if ((ty+p) < sharedWidth && (tx+0) < sharedWidth)
                    // {
                    //     acc += shared[c*sharedWidth*sharedWidth+(ty+p)*sharedWidth+(tx+0)] * ConstMem[m*Channel*K*K+c*K*K+p*K+0];
                    // }
                    // if ((ty+p) < sharedWidth && (tx+1) < sharedWidth)
                    // {
                    //     acc += shared[c*sharedWidth*sharedWidth+(ty+p)*sharedWidth+(tx+1)] * ConstMem[m*Channel*K*K+c*K*K+p*K+1];
                    // }
                    // if ((ty+p) < sharedWidth && (tx+2) < sharedWidth)
                    // {
                    //     acc += shared[c*sharedWidth*sharedWidth+(ty+p)*sharedWidth+(tx+2)] * ConstMem[m*Channel*K*K+c*K*K+p*K+2];
                    // }
                    // if ((ty+p) < sharedWidth && (tx+3) < sharedWidth)
                    // {
                    //     acc += shared[c*sharedWidth*sharedWidth+(ty+p)*sharedWidth+(tx+3)] * ConstMem[m*Channel*K*K+c*K*K+p*K+3];
                    // }
                    // if ((ty+p) < sharedWidth && (tx+4) < sharedWidth)
                    // {
                    //     acc += shared[c*sharedWidth*sharedWidth+(ty+p)*sharedWidth+(tx+4)] * ConstMem[m*Channel*K*K+c*K*K+p*K+4];
                    // }
                    // if ((ty+p) < sharedWidth && (tx+5) < sharedWidth)
                    // {
                    //     acc += shared[c*sharedWidth*sharedWidth+(ty+p)*sharedWidth+(tx+5)] * ConstMem[m*Channel*K*K+c*K*K+p*K+5];
                    // }
                    // if ((ty+p) < sharedWidth && (tx+6) < sharedWidth)
                    // {
                    //     acc += shared[c*sharedWidth*sharedWidth+(ty+p)*sharedWidth+(tx+6)] * ConstMem[m*Channel*K*K+c*K*K+p*K+6];
                    // }
                    // if ((ty+p) < sharedWidth && (tx+7) < sharedWidth)
                    // {
                    //     acc += shared[c*sharedWidth*sharedWidth+(ty+p)*sharedWidth+(tx+7)] * ConstMem[m*Channel*K*K+c*K*K+p*K+7];
                    // }
                    // if ((ty+p) < sharedWidth && (tx+8) < sharedWidth)
                    // {
                    //     acc += shared[c*sharedWidth*sharedWidth+(ty+p)*sharedWidth+(tx+8)] * ConstMem[m*Channel*K*K+c*K*K+p*K+8];
                    // }
                    // if ((ty+p) < sharedWidth && (tx+9) < sharedWidth)
                    // {
                    //     acc += shared[c*sharedWidth*sharedWidth+(ty+p)*sharedWidth+(tx+9)] * ConstMem[m*Channel*K*K+c*K*K+p*K+9];
                    // }
                // }
                for (int q = 0; q < K; q++) 
                {
                    // int row = ty+p;
                    // int col = tx+q;
                    if ((ty+p) < sharedWidth && (tx+q) < sharedWidth)
                    {
                        acc += shared[c*sharedWidth*sharedWidth+(ty+p)*sharedWidth+(tx+q)] * ConstMem[m*Channel*K*K+c*K*K+p*K+q];
                    }
                }
            }
        }
        out_4d(blockIdx.z, m, h, w) = acc;
    }

    #undef out_4d
    #undef in_4d
    // #undef mask_4d
}

__global__ void conv_forward_kernel_opt_layer1(float *output, const float *input, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    int sharedWidth = TILE_WIDTH_LAYER1 + K - 1;
    extern __shared__ float shared[];

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    // #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int W_grid = ceil(1.0*Width/TILE_WIDTH_LAYER1);
    int m = blockIdx.x; //indicate which feature map it belongs to
    int h = (blockIdx.y / W_grid) * TILE_WIDTH_LAYER1 + ty;
    int w = (blockIdx.y % W_grid) * TILE_WIDTH_LAYER1 + tx;
    // int image = blockIdx.z;

    float acc = 0.0f;
    // Optimization 2
    for (int c = 0; c < Channel; c++) // sum over all input channels
    { 
        for(int i = ty; i < sharedWidth; i += TILE_WIDTH_LAYER1)
        {
            for(int j = tx; j < sharedWidth; j += TILE_WIDTH_LAYER1)
            {
                // int i1 = h+i-ty;
                // int i0 = w+j-tx;
                if ((h+i-ty) < Height && (w+j-tx) < Width)
                {
                    shared[c*sharedWidth*sharedWidth+i*sharedWidth+j] = in_4d(blockIdx.z, c, h+i-ty, w+j-tx);
                }
            }
        }
    }
    __syncthreads();

    if (h<Height_out && w<Width_out)
    {
        for (int c = 0; c < Channel; c++)
        {
            for (int p = 0; p < K; p++) // loop over KxK filter
            {
                // // for (int q = 0; q < K; q++) 
                // // {
                //     // int row = ty+p;
                //     // int col = tx+q;
                //     if ((ty+p) < sharedWidth && (tx+0) < sharedWidth)
                //     {
                //         acc += shared[c*sharedWidth*sharedWidth+(ty+p)*sharedWidth+(tx+0)] * ConstMem[m*Channel*K*K+c*K*K+p*K+0];
                //     }
                //     if ((ty+p) < sharedWidth && (tx+1) < sharedWidth)
                //     {
                //         acc += shared[c*sharedWidth*sharedWidth+(ty+p)*sharedWidth+(tx+1)] * ConstMem[m*Channel*K*K+c*K*K+p*K+1];
                //     }
                //     if ((ty+p) < sharedWidth && (tx+2) < sharedWidth)
                //     {
                //         acc += shared[c*sharedWidth*sharedWidth+(ty+p)*sharedWidth+(tx+2)] * ConstMem[m*Channel*K*K+c*K*K+p*K+2];
                //     }
                //     if ((ty+p) < sharedWidth && (tx+3) < sharedWidth)
                //     {
                //         acc += shared[c*sharedWidth*sharedWidth+(ty+p)*sharedWidth+(tx+3)] * ConstMem[m*Channel*K*K+c*K*K+p*K+3];
                //     }
                //     if ((ty+p) < sharedWidth && (tx+4) < sharedWidth)
                //     {
                //         acc += shared[c*sharedWidth*sharedWidth+(ty+p)*sharedWidth+(tx+4)] * ConstMem[m*Channel*K*K+c*K*K+p*K+4];
                //     }
                //     if ((ty+p) < sharedWidth && (tx+5) < sharedWidth)
                //     {
                //         acc += shared[c*sharedWidth*sharedWidth+(ty+p)*sharedWidth+(tx+5)] * ConstMem[m*Channel*K*K+c*K*K+p*K+5];
                //     }
                //     if ((ty+p) < sharedWidth && (tx+6) < sharedWidth)
                //     {
                //         acc += shared[c*sharedWidth*sharedWidth+(ty+p)*sharedWidth+(tx+6)] * ConstMem[m*Channel*K*K+c*K*K+p*K+6];
                //     }
                //     // if ((ty+p) < sharedWidth && (tx+7) < sharedWidth)
                //     // {
                //     //     acc += shared[c*sharedWidth*sharedWidth+(ty+p)*sharedWidth+(tx+7)] * ConstMem[m*Channel*K*K+c*K*K+p*K+7];
                //     // }
                //     // if ((ty+p) < sharedWidth && (tx+8) < sharedWidth)
                //     // {
                //     //     acc += shared[c*sharedWidth*sharedWidth+(ty+p)*sharedWidth+(tx+8)] * ConstMem[m*Channel*K*K+c*K*K+p*K+8];
                //     // }
                //     // if ((ty+p) < sharedWidth && (tx+9) < sharedWidth)
                //     // {
                //     //     acc += shared[c*sharedWidth*sharedWidth+(ty+p)*sharedWidth+(tx+9)] * ConstMem[m*Channel*K*K+c*K*K+p*K+9];
                //     // }
                // // }
                for (int q = 0; q < K; q++) 
                {
                    // int row = ty+p;
                    // int col = tx+q;
                    if ((ty+p) < sharedWidth && (tx+q) < sharedWidth)
                    {
                        acc += shared[c*sharedWidth*sharedWidth+(ty+p)*sharedWidth+(tx+q)] * ConstMem[m*Channel*K*K+c*K*K+p*K+q];
                    }
                }
            }
        }
        out_4d(blockIdx.z, m, h, w) = acc;
    }

    #undef out_4d
    #undef in_4d
    // #undef mask_4d
}



// Opt2. Opt3.
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns)
{
    //@@ Insert code to implement matrix multiplication here
    __shared__ float subTileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float subTileB[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = bx * blockDim.x + tx;
    int Col = by * blockDim.y + ty;
    float Pvalue = 0;

    for (int m = 0; m < ceil(1.0*numAColumns/TILE_WIDTH); ++m)
    {
        if (Row < numARows && m*TILE_WIDTH+ty < numAColumns)
        {
            subTileA[tx][ty] = A[Row*numAColumns + m*TILE_WIDTH+ty];
        } 
        else 
        {
            subTileA[tx][ty] = 0;
        }

        if (m*TILE_WIDTH+tx < numBRows && Col < numBColumns)
        {
            subTileB[tx][ty] = B[(m*TILE_WIDTH+tx)*numBColumns+Col];
        }
        else
        {
            subTileB[tx][ty] = 0;
        }

        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; ++k){
            Pvalue += subTileA[tx][k] * subTileB[k][ty];
        }
        __syncthreads();
    }
    if (Row < numCRows && Col < numCColumns){
        C[Row*numCColumns+Col] = Pvalue;
    }
}

__global__ void UnrollInput(float *output, const float *input, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int Y = Height_out * Width_out;

    #define in_3d(i2, i1, i0) input[(i2) * (Height * Width) + (i1) * (Width) + i0]
    #define unroll_2d(i1, i0) output[(i1) * (Y) + i0]
    
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(idx < Channel*Y){
        int h = (idx % Y) / Width_out;
        int w = (idx % Y) % Width_out;
        int c = idx / Y;
        int w_base = c * (K*K);
        for(int p = 0; p < K; p++){
            for(int q = 0; q < K; q++){
                int h_unroll = w_base + p * K + q; // data needed by one thread
                int w_unroll = h * Width_out + w;
                unroll_2d(h_unroll, w_unroll) = in_3d(c, h + p, w + q);
            }
        }
    }
    #undef in_3d
    #undef unroll_2d
}



// Opt1. Opt2.
__global__ void conv_forward_kernel_opt(float *output, const float *input, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    int sharedWidth = TILE_WIDTH + K - 1;
    extern __shared__ float shared[];

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    // #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int W_grid = ceil(1.0*Width/TILE_WIDTH);
    int m = blockIdx.x; //indicate which feature map it belongs to
    int h = (blockIdx.y / W_grid) * TILE_WIDTH + ty;
    int w = (blockIdx.y % W_grid) * TILE_WIDTH + tx;
    // int image = blockIdx.z;

    float acc = 0.0f;
    // Optimization 2
    for (int c = 0; c < Channel; c++) // sum over all input channels
    { 
        for(int i = ty; i < sharedWidth; i += TILE_WIDTH)
        {
            for(int j = tx; j < sharedWidth; j += TILE_WIDTH)
            {
                // int i1 = h+i-ty;
                // int i0 = w+j-tx;
                if ((h+i-ty) < Height && (w+j-tx) < Width)
                {
                    shared[c*sharedWidth*sharedWidth+i*sharedWidth+j] = in_4d(blockIdx.z, c, h+i-ty, w+j-tx);
                }
            }
        }
    }
    __syncthreads();

    if (h<Height_out && w<Width_out)
    {
        for (int c = 0; c < Channel; c++)
        {
            for (int p = 0; p < K; p++) // loop over KxK filter
            {
                for (int q = 0; q < K; q++) 
                {
                    // int row = ty+p;
                    // int col = tx+q;
                    if ((ty+p) < sharedWidth && (tx+q) < sharedWidth)
                    {
                        acc += shared[c*sharedWidth*sharedWidth+(ty+p)*sharedWidth+(tx+q)] * ConstMem[m*Channel*K*K+c*K*K+p*K+q];
                    }
                }
            }
        }
        out_4d(blockIdx.z, m, h, w) = acc;
    }

    #undef out_4d
    #undef in_4d
    // #undef mask_4d
}

__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int W_grid = ceil(1.0*Width/TILE_WIDTH);
    int m = blockIdx.x; //indicate which feature map it belongs to
    int h = (blockIdx.y / W_grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % W_grid) * TILE_WIDTH + threadIdx.x;
    int image = blockIdx.z;
    
    float acc = 0.0f;
    for (int c = 0; c < Channel; c++) // sum over all input channels
    { 
        for (int p = 0; p < K; p++) // loop over KxK filter
        {
            for (int q = 0; q < K; q++) 
            {
                acc += in_4d(image, c, h + p, w + q) * mask_4d(m, c, p, q);
            }
        }
    }
    if (h<Height_out && w<Width_out)
    {
        out_4d(image, m, h, w) = acc;
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
    // int inputNum = Batch * Channel * Height * Width;
    // int outputNum = Batch * Map_out * (Height - K + 1) * (Width - K + 1);
    // int maskNum = K * K * Channel * Map_out;
    cudaMalloc((void **)device_input_ptr, Batch * Channel * Height * Width * sizeof(float));
    cudaMalloc((void **)device_output_ptr, Batch * Map_out * (Height - K + 1) * (Width - K + 1) * sizeof(float));
    // cudaMalloc((void **)device_mask_ptr, maskNum * sizeof(float)); //comment this line for opt.
    cudaMemcpy(*device_input_ptr, host_input, Batch * Channel * Height * Width*sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(*device_mask_ptr, host_mask, maskNum*sizeof(float), cudaMemcpyHostToDevice); //comment this line for opt.
    //Optimization 1: Weight matrix (kernel values) in constant memory
    cudaMemcpyToSymbol(ConstMem, host_mask, K * K * Channel * Map_out*sizeof(float));
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // int W_grid = ceil(1.0*Width/TILE_WIDTH);
    // int H_grid = ceil(1.0*Height/TILE_WIDTH);
    // int Y = H_grid * W_grid;

    // // Set the kernel dimensions
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(Map_out, ceil(1.0*Width/TILE_WIDTH)*ceil(1.0*Height/TILE_WIDTH), Batch);
    dim3 blockDim1(TILE_WIDTH_LAYER1, TILE_WIDTH_LAYER1, 1);
    dim3 gridDim1(Map_out, ceil(1.0*Width/TILE_WIDTH_LAYER1)*ceil(1.0*Height/TILE_WIDTH_LAYER1), Batch);
    dim3 blockDim2(TILE_WIDTH_LAYER2, TILE_WIDTH_LAYER2, 1);
    dim3 gridDim2(Map_out, ceil(1.0*Width/TILE_WIDTH_LAYER2)*ceil(1.0*Height/TILE_WIDTH_LAYER2), Batch);
    // int sharedMemSize = Channel*(TILE_WIDTH + K - 1)*(TILE_WIDTH + K - 1);
    // // call the kernel
    // conv_forward_kernel<<<gridDim, blockDim>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K); //comment this line for optimization

    // Optimizaztion 1: Weight matrix (kernel values) in constant memory
    // OPtimization 2: Tiled shared memory convolution
    // std::cout<<Map_out<<std::endl;
    // std::cout<<K<<std::endl;
    // if (K!=7)
    // {
    //     conv_forward_kernel_opt<<<gridDim, blockDim, Channel*(TILE_WIDTH + K - 1)*(TILE_WIDTH + K - 1)*sizeof(float)>>>(device_output, device_input, Batch, Map_out, Channel, Height, Width, K);
    // }
    if (Map_out==16)
    {
        conv_forward_kernel_opt_layer2<<<gridDim2, blockDim2, Channel*(TILE_WIDTH_LAYER2 + K - 1)*(TILE_WIDTH_LAYER2 + K - 1)*sizeof(float)>>>(device_output, device_input, Batch, Map_out, Channel, Height, Width, K);
    }
    else
    {
        conv_forward_kernel_opt_layer1<<<gridDim1, blockDim1, Channel*(TILE_WIDTH_LAYER1 + K - 1)*(TILE_WIDTH_LAYER1 + K - 1)*sizeof(float)>>>(device_output, device_input, Batch, Map_out, Channel, Height, Width, K);
    }
    

    // OPtimization 3: Shared memory matrix multiplication and input matrix unrolling
    // int UnrollWidth = (Height - K + 1)*(Width - K + 1);
    // int UnrollHeight = Channel * K * K;
    // float* Input_Unroll;
    // cudaMalloc((void **)&Input_Unroll, UnrollWidth * UnrollHeight * sizeof(float));
    // dim3 gridDim_Unroll(ceil(1.0*Channel*UnrollWidth/TILE_WIDTH), 1, 1);
    // dim3 blockDim_Unroll(TILE_WIDTH, 1, 1);
    // dim3 gridDim_Matrix(ceil(1.0*Map_out/TILE_WIDTH),ceil(1.0 *UnrollWidth/TILE_WIDTH),1);
    // dim3 blockDim_Matrix(TILE_WIDTH, TILE_WIDTH, 1);
    // int image = 0;
    // while(image<Batch){
    //     UnrollInput<<<gridDim_Unroll, blockDim_Unroll>>>(Input_Unroll, &device_input[image*Channel*Height*Width],  Channel, Height, Width, K);
    //     matrixMultiply<<<gridDim_Matrix, blockDim_Matrix>>>((float*)device_mask, Input_Unroll, &device_output[image*Map_out*UnrollWidth], Map_out, UnrollHeight, UnrollHeight, UnrollWidth, Map_out, UnrollWidth);
    //     image++;
    // }
    // cudaFree(Input_Unroll);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    // int outputNum = Batch * Map_out * (Height - K + 1) * (Width - K + 1);
    cudaMemcpy(host_output, device_output, Batch * Map_out * (Height - K + 1) * (Width - K + 1) * sizeof(float), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(device_output);
    cudaFree(device_input);
    // cudaFree(device_mask); comment for opt.
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
