#include "../include/kernel_runner_pruned.cuh"

/**
 * 1. Build "big graph".
 * 2. Iteratively, until convergence: prune vertices of degree 1
 * 3. From 'big graph', build C graphs, where C is the number of connected components
 * - this requires mapping: vertex -> CC
 * - each CC calculates its own BC -> allows for BC updates on Kernel
 */


int main(int argc, char *argv[]) {
    if (argc != 3)
        printf("Incorrect program arguments, usage:\n  ./brandes input-file output-file");
    std::string input_file = argv[1], output_file = argv[2];

    Graph graph = read_graph(input_file);
    // Prune deg-1 vertices
    graph.prune_cpu();
    // Build graph based on CC decomposition
    graph.build();
    // Run GPU Brandes algorithm
    KernelRunnerMultiCCPruned runner{};
    runner.run_kernels(graph);
    // Save results to file
    graph.write_bc(output_file);

    return 0;
}
