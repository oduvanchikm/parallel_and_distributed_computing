#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define SIZE_BLOCK 512

__global__ void reverse_array_function(float* input_array, float* output_arra, int n)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += gridDim.x * blockDim.x)
    {
        output_arra[n - 1 - idx] = input_array[idx];
    }
}

int main()
{
    int n;
    scanf("%d", &n);

    float* input_array = (float*)malloc(sizeof(float) * n);
    float* reverse_array = (float*)malloc(sizeof(float) * n);
    if (!input_array || !reverse_array)
    {
        printf("Allocation error in malloc\n");
        return 1;
    }

    for (int i = 0; i < n; ++i)
    {
        scanf("%f", &input_array[i]);
    }

    float* input_array_cuda;
    float* reverse_array_cuda;

    cudaError_t error_1 = cudaMalloc(&input_array_cuda, sizeof(float)* n);
    if (error_1 != cudaSuccess)
    {
      printf("cudaMalloc1 failed: %s\n", cudaGetErrorString(error_1));
      return 1;
    }


    cudaError_t error_2 = cudaMalloc(&reverse_array_cuda,sizeof(float)* n);
    if (error_2 != cudaSuccess)
    {
      printf("Allocation error in cudaMalloc2");
      return 1;
    }

    cudaError_t error_3 = cudaMemcpy(input_array_cuda, input_array, n * sizeof(float), cudaMemcpyHostToDevice);
    if (error_3 != cudaSuccess)
    {
      printf("Allocation error in cudaMemcpy1");
      return 1;
    }

    reverse_array_function<<<SIZE_BLOCK, SIZE_BLOCK>>>(input_array_cuda, reverse_array_cuda, n);

    cudaError_t error_4 = cudaMemcpy(reverse_array, reverse_array_cuda,sizeof(float)* n, cudaMemcpyDeviceToHost);
    if (error_4 != cudaSuccess)
    {
      printf("Allocation error in cudaMemcpy2");
      return 1;
    }

    for (int i = 0; i < n; ++i)
    {
        printf("%.10e ", reverse_array[i]);
    }
    printf("\n");

    free(input_array);
    free(reverse_array);
    cudaFree(input_array_cuda);
    cudaFree(reverse_array_cuda);

    return 0;
}
