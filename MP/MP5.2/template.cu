// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float s[2 * BLOCK_SIZE];
  
  int bt1 = 2 * blockDim.x * blockIdx.x + threadIdx.x;
  int bt2 = bt1 + blockDim.x;

  if (bt1 < len) s[threadIdx.x] = input[bt1];
  else s[threadIdx.x] = 0;

  if (bt2 < len) s[blockDim.x+threadIdx.x] = input[bt2];
  else s[blockDim.x+threadIdx.x] = 0;

  for (int stride = 1; stride < 2 * BLOCK_SIZE; stride <<= 1) {
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if (index < 2 * BLOCK_SIZE && (index - stride) >= 0) s[index] += s[index - stride];
  }
  
  for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if ((index + stride) < 2 * BLOCK_SIZE) s[index + stride] += s[index];
  }
  
  __syncthreads();
  if (bt1 < len) output[bt1] = s[threadIdx.x];
  else output[bt1] = 0;

  if (bt2 < len) output[bt2] = s[blockDim.x + threadIdx.x];
  else output[bt2] = 0;
}

__global__ void scan_sum(float *input, float *output,int len){
  
  if (threadIdx.x == 0) output[threadIdx.x] = 0;
  else {
    if (threadIdx.x < len) output[threadIdx.x] = input[threadIdx.x*2*BLOCK_SIZE-1];
  }
}

__global__ void cal_sum(float *output, float *partial, int len) {
  int bt1 = 2 * blockDim.x * blockIdx.x + threadIdx.x;
  int bt2 = bt1 + blockDim.x;
  if (bt1 < len) output[bt1] += partial[blockIdx.x];
  if (bt2 < len) output[bt2] += partial[blockIdx.x];
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list
  float *partial;


  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");


  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  int grid = ceil(numElements / 2. / BLOCK_SIZE);
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&partial, grid * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 GridDim(grid, 1, 1);
  dim3 GridDimm(1, 1, 1);
  dim3 BlockDim(BLOCK_SIZE, 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<GridDim, BlockDim>>>(deviceInput, deviceOutput, numElements);
  cudaDeviceSynchronize();
  scan_sum<<<GridDimm, BlockDim>>>(deviceOutput, partial, grid);
  cudaDeviceSynchronize();
  scan<<<GridDimm, BlockDim>>>(partial, partial, grid);
  cudaDeviceSynchronize();
  cal_sum<<<GridDim, BlockDim>>>(deviceOutput, partial, numElements);
  cudaDeviceSynchronize();
  
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
