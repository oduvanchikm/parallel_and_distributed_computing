#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define EPSILON 1e-6

typedef enum {
    ANY,
    INCORRECT,
    ONE_ROOT,
    TWO_ROOTS,
    IMAGINARY
} status;

__global__ void find_roots(float* a, float* b, float* c, float* x1, float* x2, status* status_code)
{
  if (*a == 0) 
  {
    if (*b == 0) 
    {
      *status_code = (*c == 0) ? ANY : INCORRECT;
    } 
    else 
    {
      *x1 = -(*c) / (*b);
      *status_code = ONE_ROOT;
    }

    return;
  }

  float discriminant = (*b) * (*b) - 4 * (*a) * (*c);

  if (discriminant > EPSILON)
  {
    *x1 = (-(*b) + sqrt(discriminant)) / (2 * (*a));
    *x2 = (-(*b) - sqrt(discriminant)) / (2 * (*a));
    *status_code = TWO_ROOTS;
    return;
  }
  else if (fabs(discriminant) < EPSILON)
  {
    *x1 = -(*b) / (2 * (*a));
    *status_code = ONE_ROOT;
    return;
  }
  else
  {
    *status_code = IMAGINARY;
    return;
  }
}

void free_function(float* a, float *b, float* c, float* x1, float* x2, status* status)
{
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
  cudaFree(x1);
  cudaFree(x2);
  cudaFree(status);
}

int main()
{
  float a, b, c;
  float x1, x2;
  status status_code;

  if (scanf("%f %f %f", &a, &b, &c) != 3)
  {
    printf("Error wiyj count of arguments\n");
    return 1;
  }

  float* a_gpu, *b_gpu, *c_gpu;
  float* x1_gpu, *x2_gpu;
  status* status_code_gpu;

  cudaError_t error_1 = cudaMalloc(&a_gpu, sizeof(float));
  if (error_1 != cudaSuccess)
  {
    printf("cudaMalloc1 failed: %s\n", cudaGetErrorString(error_1));
    return 1;
  }

  cudaError_t error_2 = cudaMalloc(&b_gpu, sizeof(float));
  if (error_2 != cudaSuccess)
  {
    printf("cudaMalloc1 failed: %s\n", cudaGetErrorString(error_2));
    return 1;
  }

  cudaError_t error_3 = cudaMalloc(&c_gpu, sizeof(float));
  if (error_3 != cudaSuccess)
  {
    printf("cudaMalloc1 failed: %s\n", cudaGetErrorString(error_3));
    return 1;
  }

  cudaError_t error_4 = cudaMalloc(&x1_gpu, sizeof(float));
  if (error_4 != cudaSuccess)
  {
    printf("cudaMalloc1 failed: %s\n", cudaGetErrorString(error_4));
    return 1;
  }

  cudaError_t error_5 = cudaMalloc(&x2_gpu, sizeof(float));
  if (error_5 != cudaSuccess)
  {
    printf("cudaMalloc1 failed: %s\n", cudaGetErrorString(error_5));
    return 1;
  }

  cudaError_t error_enum = cudaMalloc(&status_code_gpu, sizeof(status));
  if (error_enum != cudaSuccess)
  {
    printf("cudaMalloc1 failed: %s\n", cudaGetErrorString(error_enum));
    return 1;
  }


  cudaError_t error_6 = cudaMemcpy(a_gpu, &a, sizeof(float), cudaMemcpyHostToDevice);
  if (error_6 != cudaSuccess)
  {
    printf("Allocation error in cudaMemcpy1 %s", cudaGetErrorString(error_6));
    return 1;
  }

  cudaError_t error_7 = cudaMemcpy(b_gpu, &b, sizeof(float), cudaMemcpyHostToDevice);
  if (error_7 != cudaSuccess)
  {
    printf("Allocation error in cudaMemcpy1 %s", cudaGetErrorString(error_7));
    return 1;
  }

  cudaError_t error_8 = cudaMemcpy(c_gpu, &c, sizeof(float), cudaMemcpyHostToDevice);
  if (error_8 != cudaSuccess)
  {
    printf("Allocation error in cudaMemcpy1 %s", cudaGetErrorString(error_8));
    return 1;
  }

  find_roots<<<1, 1>>>(a_gpu, b_gpu, c_gpu, x1_gpu, x2_gpu, status_code_gpu);

  cudaError_t error_9 = cudaMemcpy(&x1, x1_gpu, sizeof(float), cudaMemcpyDeviceToHost);
  if (error_9 != cudaSuccess)
  {
    printf("Allocation error in cudaMemcpy1 %s", cudaGetErrorString(error_9));
    return 1;
  }

  cudaError_t error_10 = cudaMemcpy(&x2, x2_gpu, sizeof(float), cudaMemcpyDeviceToHost);
  if (error_10 != cudaSuccess)
  {
    printf("Allocation error in cudaMemcpy1 %s", cudaGetErrorString(error_10));
    return 1;
  }

  cudaError_t error_enum_mem = cudaMemcpy(&status_code, status_code_gpu, sizeof(status), cudaMemcpyDeviceToHost);
  if (error_enum_mem != cudaSuccess)
  {
    printf("Allocation error in cudaMemcpy1 %s", cudaGetErrorString(error_enum_mem));
    return 1;
  }

  switch (status_code) {
    case ANY:
      printf("any\n");
      break;

    case INCORRECT:
      printf("incorrect\n");
      break;

    case ONE_ROOT:
      printf("%.6f\n", x1);
      break;

    case TWO_ROOTS:
      printf("%.6f %.6f\n", x1, x2);
      break;

    case IMAGINARY:
      printf("imaginary\n");
      break;
  }

  free_function(a_gpu, b_gpu, c_gpu, x1_gpu, x2_gpu, status_code_gpu);

  return 0;
}
