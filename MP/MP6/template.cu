// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define GRID_SIZE 32

//@@ insert code here

__global__ void gray2histogram (unsigned char *input, unsigned int *histogram, int width, int height){
  __shared__ unsigned int h[HISTOGRAM_LENGTH];
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;

  if (tid < HISTOGRAM_LENGTH) h[tid] = 0.;

  __syncthreads();  

  if (tx < width && ty < height) atomicAdd(&(h[input[ty * width + tx]]), 1);

  __syncthreads();

  if (tid < HISTOGRAM_LENGTH) atomicAdd(&(histogram[tid]), h[tid]);
}

__global__ void scan(unsigned int *input, float *output, int size, int len) {
  __shared__ float T[HISTOGRAM_LENGTH];
  int idx = 2 * blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < len) T[threadIdx.x] = input[idx];
  else T[threadIdx.x] = 0.;

  if (idx + blockDim.x < len) T[blockDim.x + threadIdx.x] = input[idx + blockDim.x];
  else T[blockDim.x+threadIdx.x] = 0.;

  for (int stride = 1; stride < HISTOGRAM_LENGTH; stride <<= 1){
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if (index < HISTOGRAM_LENGTH && index >= stride) T[index] += T[index - stride];
  }
  
  for (int stride = 64; stride > 0; stride >>= 1){
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if (index + stride < HISTOGRAM_LENGTH) T[index + stride] += T[index];
  }
  
  __syncthreads();

  if (idx < len) output[idx] = 1. * T[threadIdx.x] / size;
  else output[idx] = 0.;

  if (idx + blockDim.x < len) output[idx + blockDim.x] = 1. * T[blockDim.x + threadIdx.x] / size;
  else output[idx + blockDim.x] = 0.;
}

__global__ void correct_color(float* old_cdf, unsigned char* input, unsigned char* output, float ss, int size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    unsigned char new_val = (unsigned char) 255 * (old_cdf[input[index]] - ss) / (1.0 - ss);
    if (new_val > 255) new_val = 255;
    output[index] = new_val;
  }
}

__global__ void Convert(float* input, unsigned char* output, int len, int mode) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < len) {
    if (mode == 0) output[index] = (unsigned char) (input[index] * 255); 
    else input[index] = (float) (output[index] / 255.); 
  }
}

__global__ void Color2Gray(unsigned char* input, unsigned char* output, int height, int width) {
  int w = blockIdx.x * blockDim.x + threadIdx.x;
  int h = blockIdx.y * blockDim.y + threadIdx.y;
  if (w < width && h < height){
    int index = 3 * (h * width + w);
    output[h * width + w] = (unsigned char) (0.21 * input[index] + 0.71 * input[index + 1] + 0.07 * input[index + 2]);
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float *inputImage_f, *histogram, *outputImage_f, *ss;
  unsigned char *inputImage_c, *grayImage_c, *deviceOutputImage;
  unsigned int *deviceHistogram;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  cudaMalloc((void **)&inputImage_f, imageChannels * imageWidth * imageHeight * sizeof(float));
  cudaMalloc((void **)&outputImage_f, imageChannels * imageWidth * imageHeight * sizeof(float));
  cudaMalloc((void **)&histogram, HISTOGRAM_LENGTH * sizeof(float));
  cudaMalloc((void **)&inputImage_c, imageChannels * imageWidth * imageHeight * sizeof(unsigned char));
  cudaMalloc((void **)&deviceOutputImage, imageChannels * imageWidth * imageHeight * sizeof(unsigned char));
  cudaMalloc((void **)&grayImage_c, imageWidth * imageHeight * sizeof(unsigned char));
  cudaMalloc((void **)&deviceHistogram, HISTOGRAM_LENGTH * sizeof(unsigned int));
  ss = (float*)malloc(sizeof(float));

  cudaMemcpy(inputImage_f, hostInputImageData, imageChannels * imageWidth * imageHeight * sizeof(float), cudaMemcpyHostToDevice);
  
  dim3 DimGrid(ceil(1. * imageChannels * imageHeight * imageWidth / HISTOGRAM_LENGTH), 1, 1);
  dim3 DimBlock(HISTOGRAM_LENGTH, 1, 1);
  dim3 DimGrid1(ceil(1. * imageWidth / GRID_SIZE), ceil(1. * imageHeight / GRID_SIZE), 1);
  dim3 DimBlock1(GRID_SIZE, GRID_SIZE, 1);
  dim3 DimGrid2(1, 1, 1);
  dim3 DimBlock2(128, 1, 1);

  Convert<<<DimGrid,DimBlock>>>(inputImage_f, inputImage_c, imageChannels * imageWidth * imageHeight, 0);
  cudaDeviceSynchronize();

  Color2Gray<<<DimGrid1, DimBlock1>>>(inputImage_c, grayImage_c, imageHeight, imageWidth);
  cudaDeviceSynchronize(); 

  gray2histogram<<<DimGrid1,DimBlock1>>>(grayImage_c, deviceHistogram, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  scan<<<DimGrid2, DimBlock2>>>(deviceHistogram, histogram, imageHeight * imageWidth, HISTOGRAM_LENGTH);
  cudaDeviceSynchronize();

  cudaMemcpy(ss, histogram, sizeof(float), cudaMemcpyDeviceToHost);
  correct_color<<<DimGrid,DimBlock>>>(histogram, inputImage_c, deviceOutputImage, *ss, imageChannels * imageHeight * imageWidth);
  Convert<<<DimGrid, DimBlock>>>(outputImage_f, deviceOutputImage, imageChannels * imageWidth * imageHeight, 1);
  cudaDeviceSynchronize();
  
  cudaMemcpy(hostOutputImageData, outputImage_f, imageChannels * imageWidth * imageHeight * sizeof(float), cudaMemcpyDeviceToHost);

  wbSolution(args, outputImage);

  //@@ insert code here
  free(hostInputImageData);
  free(hostOutputImageData);
  cudaFree(inputImage_f);
  cudaFree(inputImage_c);
  cudaFree(grayImage_c);
  cudaFree(histogram);
  cudaFree(deviceHistogram);
  cudaFree(deviceOutputImage);

  return 0;
}
