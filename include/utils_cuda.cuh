#ifndef BETWEENNESS_CENTRALITY_BRANDES_UTILS_CUDA_CUH
#define BETWEENNESS_CENTRALITY_BRANDES_UTILS_CUDA_CUH

#define HANDLE_ERROR(err) (HandleError( err, __FILE__, __LINE__ ))

static void HandleError(cudaError_t err,
                        const char *file,
                        int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
               file, line);
        exit(EXIT_FAILURE);
    }
}

std::pair<cudaEvent_t, cudaEvent_t> start_benchmark() {
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, 0));
    return {start, stop};
}

float end_benchmark(cudaEvent_t &start, cudaEvent_t &stop) {
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    float elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));
    return elapsedTime;
}

#endif //BETWEENNESS_CENTRALITY_BRANDES_UTILS_CUDA_CUH
