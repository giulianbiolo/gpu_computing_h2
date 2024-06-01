#include <iostream>
#include <math.h>

using namespace std;

#define B_TILE 32
#define B_ROWS 8
#define NUM_REPS 100


__global__ void naive_transpose(float *odata, const float *idata, int N) {
    int x = blockIdx.x * B_TILE + threadIdx.x;
    int y = blockIdx.y * B_TILE + threadIdx.y;
    
    int idxi = x + N * y;
    int idxo = y + N * x;
    
    for (int i = 0; i < B_TILE; i+= B_ROWS) {
        odata[idxo + i] = idata[idxi + i * N];
    }
}

void simple_exec(int N) {
    // * Define the size of blocks and threads to be allocated on GPU
    dim3 gridB(N/B_TILE, N/B_TILE);
    dim3 blockB(B_TILE, B_ROWS);
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
        naive_transpose<<<gridB, blockB>>>(pY, pX, N);
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
    printf("Naive Transpose:\n");
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
