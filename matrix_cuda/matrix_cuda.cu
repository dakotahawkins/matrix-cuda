/*
 *  file name: matrix.cu
 *
 *  matrix.cu contains the code that realize some common used matrix operations in CUDA
 *  
 *  this is a toy program for learning CUDA, some functions are reusable in other project
 *  
 */
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "TimingCPU.h"
#include "TimingGPU.cuh"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <typeinfo>

#define BLOCK_SIZE 16
#define NO_CPU_TEST

/*
*********************************************************************
function name: gpu_matrix_mult

description: dot product of two matrix (not only square)

parameters: 
            &a GPU device pointer to a m X n matrix (A)
            &b GPU device pointer to a n X k matrix (B)
            &c GPU device output purpose pointer to a m X k matrix (C) 
            to store the result

Note:
    grid and block should be configured as:
        dim3 dimGrid((k + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    further sppedup can be obtained by using shared memory to decrease global memory access times
return: none
*********************************************************************
*/
template<class matrixT>
__global__ void gpu_matrix_mult(matrixT *a, matrixT *b, matrixT *c, std::size_t m, std::size_t n, std::size_t k)
{ 
    std::size_t row = blockIdx.y * blockDim.y + threadIdx.y; 
    std::size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    matrixT sum = 0;
    if( col < k && row < m) 
    {
        for(int i = 0; i < n; i++) 
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
} 

/*
*********************************************************************
function name: gpu_square_matrix_mult

description: dot product of two matrix (not only square) in GPU

parameters: 
            &a GPU device pointer to a n X n matrix (A)
            &b GPU device pointer to a n X n matrix (B)
            &c GPU device output purpose pointer to a n X n matrix (C) 
            to store the result
Note:
    grid and block should be configured as:

        dim3 dim_grid((n - 1) / BLOCK_SIZE + 1, (n - 1) / BLOCK_SIZE + 1, 1);
        dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);

return: none
*********************************************************************
*/
template<class matrixT>
__global__ void gpu_square_matrix_mult(matrixT *d_a, matrixT *d_b, matrixT *d_result, std::size_t n)
{
    __shared__ matrixT tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ matrixT tile_b[BLOCK_SIZE][BLOCK_SIZE];

    std::size_t row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    std::size_t col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    matrixT tmp = 0;
    std::size_t idx;

    for (std::size_t sub = 0; sub < gridDim.x; ++sub)
    {
        idx = row * n + sub * BLOCK_SIZE + threadIdx.x;
        if(idx >= n*n)
        {
            // n may not divisible by BLOCK_SIZE
            tile_a[threadIdx.y][threadIdx.x] = 0;
        }
        else
        {
            tile_a[threadIdx.y][threadIdx.x] = d_a[idx];
        }

        idx = (sub * BLOCK_SIZE + threadIdx.y) * n + col;
        if(idx >= n*n)
        {
            tile_b[threadIdx.y][threadIdx.x] = 0;
        }  
        else
        {
            tile_b[threadIdx.y][threadIdx.x] = d_b[idx];
        }
        __syncthreads();

        for (std::size_t k = 0; k < BLOCK_SIZE; ++k)
        {
            tmp += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }
        __syncthreads();
    }
    if(row < n && col < n)
    {
        d_result[row * n + col] = tmp;
    }
}

/*
*********************************************************************
function name: gpu_matrix_transpose

description: matrix transpose

parameters: 
            &mat_in GPU device pointer to a rows X cols matrix
            &mat_out GPU device output purpose pointer to a cols X rows matrix 
            to store the result
Note:
    grid and block should be configured as:
        dim3 dim_grid((n - 1) / BLOCK_SIZE + 1, (n - 1) / BLOCK_SIZE + 1, 1);
        dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);

return: none
*********************************************************************
*/
template<class matrixT>
__global__ void gpu_matrix_transpose(matrixT* mat_in, matrixT* mat_out, std::size_t rows, std::size_t cols)
{
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows) 
    {
        std::size_t pos = idy * cols + idx;
        std::size_t trans_pos = idx * rows + idy;
        mat_out[trans_pos] = mat_in[pos];
    }
}
/*
*********************************************************************
function name: cpu_matrix_mult

description: dot product of two matrix (not only square) in CPU, 
             for validating GPU results

parameters: 
            &a CPU host pointer to a m X n matrix (A)
            &b CPU host pointer to a n X k matrix (B)
            &c CPU host output purpose pointer to a m X k matrix (C) 
            to store the result
return: none
*********************************************************************
*/
template<class matrixT>
void cpu_matrix_mult(matrixT *h_a, matrixT *h_b, matrixT *h_result, std::size_t m, std::size_t n, std::size_t k) {
    for (std::size_t i = 0; i < m; ++i)
    {
        std::size_t in = i * n;
        std::size_t ik = i * k;
        for (std::size_t j = 0; j < k; ++j)
        {
            matrixT tmp = (matrixT)0.0;
            for (std::size_t h = 0; h < n; ++h)
            {
                tmp += h_a[in + h] * h_b[h * k + j];
            }
            h_result[ik + j] = tmp;
        }
    }
}



template<class matrixT>
void test(std::size_t m, std::size_t n, std::size_t k)
{
    std::cout << "\n----------------------------------------------------------------------------------\nMatrix type: " << typeid(matrixT).name() << '\n';

    // allocate memory in host RAM, h_cc is used to store CPU result
    matrixT *h_a, *h_b, *h_c;
#ifndef NO_CPU_TEST
    matrixT *h_cc;
#endif

    cudaMallocHost((void **)&h_a, sizeof(matrixT)*m*n);
    cudaMallocHost((void **)&h_b, sizeof(matrixT)*n*k);
    cudaMallocHost((void **)&h_c, sizeof(matrixT)*m*k);
#ifndef NO_CPU_TEST
    cudaMallocHost((void **)&h_cc, sizeof(matrixT)*m*k);
#endif

    /* Fixed seed for illustration */
    std::srand(3333);

    // random initialize matrix A
    for (std::size_t i = 0; i < m; ++i) {
        std::size_t in = i * n;
        for (std::size_t j = 0; j < n; ++j) {
            h_a[in + j] = (matrixT)(std::rand() % 1024);
        }
    }

    // random initialize matrix B
    for (std::size_t i = 0; i < n; ++i) {
        std::size_t ik = i * k;
        for (std::size_t j = 0; j < k; ++j) {
            h_b[ik + j] = (matrixT)(std::rand() % 1024);
        }
    }

    double gpu_elapsed_time_ms = 0.0;
#ifndef NO_CPU_TEST
    double cpu_elapsed_time_ms = 0.0;
#endif
    {
        TimingGPU timer_GPU;
        timer_GPU.StartCounter();

        // Allocate memory space on the device 
        matrixT *d_a, *d_b, *d_c;
        cudaMalloc((void **)&d_a, sizeof(matrixT)*m*n);
        cudaMalloc((void **)&d_b, sizeof(matrixT)*n*k);
        cudaMalloc((void **)&d_c, sizeof(matrixT)*m*k);

        // copy matrix A and B from host to device memory
        cudaMemcpy(d_a, h_a, sizeof(matrixT)*m*n, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, sizeof(matrixT)*n*k, cudaMemcpyHostToDevice);

        std::size_t grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
        std::size_t grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dim3 dimGrid((unsigned int)grid_cols, (unsigned int)grid_rows);
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

        // Launch kernel 
        if (m == n && n == k)
        {
            gpu_square_matrix_mult << <dimGrid, dimBlock >> > (d_a, d_b, d_c, n);
        }
        else
        {
            gpu_matrix_mult << <dimGrid, dimBlock >> > (d_a, d_b, d_c, m, n, k);
        }
        // Transefr results from device to host 
        cudaMemcpy(h_c, d_c, sizeof(matrixT)*m*k, cudaMemcpyDeviceToHost);
        cudaThreadSynchronize();

        gpu_elapsed_time_ms = (double)timer_GPU.GetCounter();

        // compute time elapse on GPU computing
        std::cout << "Time elapsed on matrix multiplication of " << m << 'x' << n << " . " << n << 'x' << k << " on GPU: " << gpu_elapsed_time_ms << "ms.\n";

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }

#ifndef NO_CPU_TEST
    {
        TimingCPU timer_CPU;
        timer_CPU.StartCounter();

        cpu_matrix_mult(h_a, h_b, h_cc, m, n, k);

        cpu_elapsed_time_ms = timer_CPU.GetCounter();
        std::cout << "Time elapsed on matrix multiplication of " << m << 'x' << n << " . " << n << 'x' << k << " on CPU: " << cpu_elapsed_time_ms << "ms.\n";
    }
    // validate results computed by GPU
    int all_ok = 1;
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            //printf("[%d][%d]:%d == [%d][%d]:%d, ", i, j, h_c[i*k + j], i, j, h_c[i*k + j]);
            if (h_c[i*k + j] != h_c[i*k + j])
            {
                all_ok = 0;
            }
        }
        //printf("\n");
    }

    // roughly compute speedup
    if (all_ok)
    {
        std::cout << "All results are correct!!!, speedup = " << cpu_elapsed_time_ms / gpu_elapsed_time_ms << "\n";
    }
    else
    {
        std::cout << "Incorrect results\n";
    }

#endif // NO_CPU_TEST

    // free memory
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);

#ifndef NO_CPU_TEST
    cudaFreeHost(h_cc);
#endif
}

/*
*********************************************************************
function name: main

description: test and compare

parameters: 
            none

return: none
*********************************************************************
*/
int main(int argc, char const *argv[])
{
    std::cout << "==================================================================================\n";
    std::size_t m, n, k;

    std::cout << "Enter m, n, and k: ";
    std::cin >> m >> n >> k;

    test<std::int32_t>(m, n, k);
    test<float>(m, n, k);
    test<double>(m, n, k);

    std::cout << "\n==================================================================================\n";
    return 0;
}
