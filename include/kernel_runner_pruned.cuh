#ifndef BETWEENNESS_CENTRALITY_BRANDES_KERNEL_RUNNER_PRUNED_CUH
#define BETWEENNESS_CENTRALITY_BRANDES_KERNEL_RUNNER_PRUNED_CUH

#include <iostream>
#include <math.h>
#include "utils.h"
#include "utils_cuda.cuh"
#include "graph_pruned.h"

#define THREADS_PER_BLOCK 256


dim3 get_dim_grid(int n_vertices) {
    // E.g. 700 vertices, 256 threads_per_block -> we want 3 blocks
    // Calculation explanation : (700 + 256) / 256 = 3
    unsigned int n_blocks = (n_vertices + THREADS_PER_BLOCK) / THREADS_PER_BLOCK;
    return dim3{n_blocks, 1, 1};
}

dim3 get_dim_block() {
    return dim3{THREADS_PER_BLOCK, 1, 1};
}


struct BrandesDataDevice {
    // Graph data
    int *offset, *vmap, *nvir, *ptrs, *adjs;
    // Algorithm data
    int *distance, *sigma;
    const int distance_size, sigma_size;
    double *delta, *bc;
    const int delta_size;
    bool *cont;
    const bool cont_size = sizeof(bool);
    int64_t *reach;

    BrandesDataDevice(BrandesData &brandes_host_data, int n_real_vertices) :
            distance_size(sizeof(int) * n_real_vertices), sigma_size(sizeof(int) * n_real_vertices),
            delta_size(sizeof(double) * n_real_vertices) {
        // Graph data
        // Move cc to device -> ONCE
        HANDLE_ERROR(cudaMalloc(&offset, brandes_host_data.offset_size));
        HANDLE_ERROR(
                cudaMemcpy(offset, brandes_host_data.offset, brandes_host_data.offset_size, cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMalloc(&vmap, brandes_host_data.vmap_size));
        HANDLE_ERROR(cudaMemcpy(vmap, brandes_host_data.vmap, brandes_host_data.vmap_size, cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMalloc(&nvir, brandes_host_data.nvir_size));
        HANDLE_ERROR(cudaMemcpy(nvir, brandes_host_data.nvir, brandes_host_data.nvir_size, cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMalloc(&ptrs, brandes_host_data.ptrs_size));
        HANDLE_ERROR(cudaMemcpy(ptrs, brandes_host_data.ptrs, brandes_host_data.ptrs_size, cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMalloc(&adjs, brandes_host_data.adjs_size));
        HANDLE_ERROR(cudaMemcpy(adjs, brandes_host_data.adjs, brandes_host_data.adjs_size, cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMalloc(&bc, brandes_host_data.bc_size));
        HANDLE_ERROR(cudaMemset(bc, 0.0, brandes_host_data.bc_size));
        HANDLE_ERROR(cudaMalloc(&reach, brandes_host_data.reach_size));
        HANDLE_ERROR(cudaMemcpy(reach, brandes_host_data.reach, brandes_host_data.reach_size, cudaMemcpyHostToDevice));

        // Algorithm data
        HANDLE_ERROR(cudaMalloc(&distance, distance_size));
        HANDLE_ERROR(cudaMalloc(&sigma, sigma_size));
        HANDLE_ERROR(cudaMalloc(&delta, delta_size));
        HANDLE_ERROR(cudaMalloc(&cont, cont_size));
    }

    ~BrandesDataDevice() {
        HANDLE_ERROR(cudaFree(offset));
        HANDLE_ERROR(cudaFree(vmap));
        HANDLE_ERROR(cudaFree(nvir));
        HANDLE_ERROR(cudaFree(ptrs));
        HANDLE_ERROR(cudaFree(adjs));
        HANDLE_ERROR(cudaFree(bc));
        HANDLE_ERROR(cudaFree(distance));
        HANDLE_ERROR(cudaFree(sigma));
        HANDLE_ERROR(cudaFree(delta));
        HANDLE_ERROR(cudaFree(cont));
        HANDLE_ERROR(cudaFree(reach));
    }

    /**
     * Initialize data for given source
     */
    void init(int source) {
        HANDLE_ERROR(cudaMemset(distance, -1, distance_size));
        HANDLE_ERROR(cudaMemset(sigma, 0, sigma_size));
        HANDLE_ERROR(cudaMemset(delta, 0.0, delta_size));
        // distance[source] = 0;
        HANDLE_ERROR(cudaMemset(distance + source, 0, sizeof(int))); // set 4bytes to 0, the rest is -1
        // sigma[source] = 1;
        HANDLE_ERROR(cudaMemset(sigma + source, 1, 1)); // set 1byte to 1, the rest is 0
    }
};

__global__ void forward_step_kernel(const int level, bool *cont,
                                    int *distance, int *sigma,
                                    const int *offset, const int *vmap, const int *nvir, const int *ptrs,
                                    const int *adjs,
                                    const int n_virtual_vertices) {
    // Each thread gets EXCLUSIVELY 'virtual vertex' to processes
    int tid = threadIdx.x + blockIdx.x * blockDim.x; // thread_offset_within_block + (blockIdx * threads_per_block)
    int stride = blockDim.x * gridDim.x; // (threads_per_block) * (num_blocks)
    int vertex, neighbor;

    for (int virtual_vertex = tid; virtual_vertex < n_virtual_vertices; virtual_vertex += stride) {
        vertex = vmap[virtual_vertex];
        if (distance[vertex] == level) {
            int end_of_neighbors = ptrs[vertex + 1];
            int neighbor_stride = nvir[vertex];
            // [CONTINUOUS segments of neighbors] are processed by multiple threads, thus improving coalesced memory access
            for (int neighbor_idx = ptrs[vertex] + offset[virtual_vertex];
                 neighbor_idx < end_of_neighbors; neighbor_idx += neighbor_stride) {
                neighbor = adjs[neighbor_idx];
                // Discovered unvisited vertex
                if (distance[neighbor] == -1) {
                    distance[neighbor] = level + 1;
                    *cont = true;
                }
                // Discovered new shortest path (possibly one of many)
                if (distance[neighbor] == level + 1) {
                    atomicAdd(&sigma[neighbor], sigma[vertex]);
                }
            }
        }
    }
}

__global__ void backward_step_kernel(const int level,
                                     const int *distance, double *delta,
                                     const int *offset, const int *vmap, const int *nvir, const int *ptrs,
                                     const int *adjs,
                                     const int n_virtual_vertices) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x; // thread_offset_within_block + (blockIdx * threads_per_block)
    const int stride = blockDim.x * gridDim.x; // (threads_per_block) * (num_blocks)
    int vertex, neighbor;

    for (int virtual_vertex = tid; virtual_vertex < n_virtual_vertices; virtual_vertex += stride) {
        vertex = vmap[virtual_vertex];
        if (distance[vertex] == level) {
            double thread_sum = 0;
            int end_of_neighbors = ptrs[vertex + 1];
            int neighbor_stride = nvir[vertex];

            for (int neighbor_idx = ptrs[vertex] + offset[virtual_vertex];
                 neighbor_idx < end_of_neighbors; neighbor_idx += neighbor_stride) {
                neighbor = adjs[neighbor_idx];
                if (distance[neighbor] == level + 1) { // alternatively, we could store predecessors array here
                    thread_sum += delta[neighbor];
                }
            }
            atomicAdd(&delta[vertex], thread_sum);
        }
    }
}


__global__ void delta_update_kernel(int n_vertices, double *delta, int *sigma, int64_t *reach) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x; // thread_offset_within_block + (blockIdx * threads_per_block)
    int stride = blockDim.x * gridDim.x; // (threads_per_block) * (num_blocks)
    int cache_sigma;

    for (int vertex = tid; vertex < n_vertices; vertex += stride) {
        // Might be 0, if it's not in the same CC - won't happen in this implementation, but it's left for clarity.
        cache_sigma = sigma[vertex];
        if (cache_sigma != 0) {
            delta[vertex] = double(reach[vertex]) / double(cache_sigma);
        }
        // otherwise delta is supposed to be initialized to 0.0
        // it's done out of the kernel via Memset, which is much faster than random writes from subsets of threads
        // this ensures better coalesce
    }
}

__global__ void bc_update_kernel(int source, int n_vertices, double *bc, double *delta, int *sigma, int64_t *reach) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x; // thread_offset_within_block + (blockIdx * threads_per_block)
    int stride = blockDim.x * gridDim.x; // (threads_per_block) * (num_blocks)

    double update;
    for (int vertex = tid; vertex < n_vertices; vertex += stride) {
        if (vertex != source) {
            update = max(delta[vertex] * double(sigma[vertex]) - 1.0, 0.0) * double(reach[source]);
            // It's better to check if there is need to update than do unnecessary write.
            // Perhaps none of the threads in the warp will execute write.
            if (update > 0) {
                bc[vertex] += update;
            }
        }
    }
}

struct KernelRunner {
    int get_distance_size(int n_vertices) {
        return sizeof(int) * n_vertices;
    }

    int get_sigma_size(int n_vertices) {
        return sizeof(int) * n_vertices;
    }

    int get_delta_size(int n_vertices) {
        return sizeof(double) * n_vertices;
    }

    virtual void run_kernels(Graph &graph) = 0;
};


/**
 * Kernel used to calculate BC of a graph composed of multiple CC.
 * It's used when the whole graph is read as multiple CC.
 *
 * Description:
 * for each CC:
 *   for each source:
 *     1. while not_done:
 *           perform one bfs step (unwind frontier)
 *     2. init_deltas
 *     3. while level > 1: // starting from leaves of the bfs-tree
 *           update delta
 *     4. for each node:
 *           update bc
 */
struct KernelRunnerMultiCCPruned : KernelRunner {
    void run_kernels(Graph &graph) override {
        double total_memory_transfer_time = 0;
        double total_kernel_execution_time = 0;
        std::pair<cudaEvent_t, cudaEvent_t> memory_transfer_bench;
        std::pair<cudaEvent_t, cudaEvent_t> kernel_execution_bench;

        for (auto &cc: graph.connected_components) {
            BrandesDataDevice dev_data(cc.brandes, cc.get_n_real_vertices());

            for (int source = 0; source < cc.get_n_real_vertices(); ++source) {
                memory_transfer_bench = start_benchmark();
                dev_data.init(source);
                total_memory_transfer_time += end_benchmark(memory_transfer_bench.first, memory_transfer_bench.second);

                int level = 0;
                bool host_cont = true;
                kernel_execution_bench = start_benchmark();
                while (host_cont) {
                    HANDLE_ERROR(cudaMemset(dev_data.cont, false, dev_data.cont_size));
                    forward_step_kernel<<<get_dim_grid(cc.get_n_virtual_vertices()), get_dim_block()>>>(
                            level,
                            dev_data.cont,
                            dev_data.distance,
                            dev_data.sigma,
                            dev_data.offset,
                            dev_data.vmap,
                            dev_data.nvir,
                            dev_data.ptrs,
                            dev_data.adjs,
                            cc.get_n_virtual_vertices());
                    HANDLE_ERROR(cudaMemcpy(&host_cont, dev_data.cont, dev_data.cont_size, cudaMemcpyDeviceToHost));
                    level = level + 1;

                }

                // Delta's are already initialize to 0.0
                delta_update_kernel<<<get_dim_grid(cc.get_n_real_vertices()), get_dim_block()>>>(
                        cc.get_n_real_vertices(), dev_data.delta, dev_data.sigma, dev_data.reach);

                while (level > 1) { // Update value for all vertices, despite source
                    level = level - 1;
                    backward_step_kernel<<<get_dim_grid(cc.get_n_virtual_vertices()
                    ),
                    get_dim_block()>>>(level,
                                       dev_data.distance,
                                       dev_data.delta,
                                       dev_data.offset,
                                       dev_data.vmap,
                                       dev_data.nvir,
                                       dev_data.ptrs,
                                       dev_data.adjs,
                                       cc.get_n_virtual_vertices());
                }

                bc_update_kernel<<<get_dim_grid(cc.get_n_real_vertices()), get_dim_block()>>>(
                        source, cc.get_n_real_vertices(), dev_data.bc, dev_data.delta, dev_data.sigma, dev_data.reach);
                total_kernel_execution_time += end_benchmark(kernel_execution_bench.first,
                                                             kernel_execution_bench.second);
            }

            // After processing the whole CC, simply copy data to the corresponding location
            // It's much faster than updating it at each step
            // and turns out to be faster than splitting it into multiple threads (heterogeneous approach)
            memory_transfer_bench = start_benchmark();
            HANDLE_ERROR(cudaMemcpy(cc.brandes.bc, dev_data.bc, cc.brandes.bc_size, cudaMemcpyDeviceToHost));
            total_memory_transfer_time += end_benchmark(memory_transfer_bench.first, memory_transfer_bench.second);
        }

        std::cerr << total_kernel_execution_time << std::endl;
        std::cerr << total_kernel_execution_time + total_memory_transfer_time << std::endl;
    }
};


void benchmark(Graph &graph, KernelRunner &runner) {
    // Start of benchmark
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, 0));

    runner.run_kernels(graph);

    // End of benchmark
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    float elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
}

#endif //BETWEENNESS_CENTRALITY_BRANDES_KERNEL_RUNNER_PRUNED_CUH
