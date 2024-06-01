#include <iostream>
#include <math.h>

using namespace std;

#define TILE_DIM 32
#define BLOCK_ROWS 8
#define NUM_REPS 100


__global__ void blockTransposeBankConflicts(float *odata, const float *idata) {
  __shared__ float tile[TILE_DIM][TILE_DIM];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}


void simple_exec(int N) {
    // * Define the size of blocks and threads to be allocated on GPU
    dim3 dimGrid(N/TILE_DIM, N/TILE_DIM, 1);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
    int memory_size = N * N * sizeof(float);
    // * Allocate memory on CPU
    float *X = (float*) malloc(memory_size);
    float *Y = (float*) malloc(memory_size);
    memset(Y, 0, memory_size);
    for (int i = 0; i < N; i++) { for (int j = 0; j < N; j++) { X[i + j*N] = i + j*N; } }

    float (*pX), (*pY);
    cudaMalloc(&pX, memory_size);
    cudaMalloc(&pY, memory_size);

    cudaEvent_t start, stop, startk, stopk;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&startk);
    cudaEventCreate(&stopk);

    cudaEventRecord(start);
    cudaMemcpy(pX, X, memory_size, cudaMemcpyHostToDevice);
    cudaMemcpy(pY, Y, memory_size, cudaMemcpyHostToDevice);
    cudaEventRecord(startk);

    for (int i = 0; i < NUM_REPS; i++)
        blockTransposeBankConflicts<<<dimGrid, dimBlock>>>(pY, pX);
    cudaEventRecord(stopk);
    cudaEventSynchronize(stopk);
    cudaMemcpy(Y, pY, memory_size, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    float milliseconds_konly = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventElapsedTime(&milliseconds_konly, startk, stopk);
    float milliseconds_mem_only = milliseconds - milliseconds_konly;
    float throughput_konly = 2 * N * N * sizeof(float) * 1e-6 * NUM_REPS / milliseconds_konly;
    float throughput = 2 * N * N * sizeof(float) * 1e-6 * NUM_REPS / milliseconds;
    printf("%d, %f, %f, %f, %f\n", N, milliseconds, throughput, milliseconds_konly, throughput_konly);
    //printf("Kernel Only Time: %f ms ( for %d Repetitions )\n", milliseconds_konly, NUM_REPS);
    //printf("Memory Allocation + Kernel Time: %f ms\n", milliseconds);
    //printf("Memory Only Time: %f ms\n", milliseconds_mem_only);
    //printf("Kernel Only Throughput in GB/s: %20.2f\n", 2 * N * N * sizeof(float) * 1e-6 * NUM_REPS / milliseconds_konly);
    //printf("Memory Alloc + Kernel Throughput in GB/s: %20.2f\n", 2 * N * N * sizeof(float) * 1e-6 * NUM_REPS / milliseconds);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(startk);
    cudaEventDestroy(stopk);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (X[i + j*N] != Y[j + i*N]) {
                printf("Error in the transposition!!!\n");
                cudaFree(pX);
                cudaFree(pY);
                free(X);
                free(Y);
                return;
            }
        }
    }
    cudaFree(pX);
    cudaFree(pY);
    free(X);
    free(Y);
    return;
}


int main(int argc, char* argv[]) {
    printf("Block Transpose With Bank Conflicts:\n");
    if (argc >= 2) {
        int N = (1 << atoi(argv[1]));
        simple_exec(N);
    } else {
        int N[10] = { 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384 };
        printf("N, OpTime[ms], OpThroughput[GB/s], KTime[ms], KThroughput[GB/s]\n");
        for (int i = 0; i < 10; i++) { simple_exec(N[i]); }
    }
    printf("\n\n");
    return 0;
}
