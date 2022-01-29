# CUDA - Brandes betweenness centrality

CUDA implementation of [Brande's](https://www.tandfonline.com/doi/abs/10.1080/0022250X.2001.9990249) algorithm for finding betweenness centrality of the graph. Logic follows the solution prosposed in [Betweenness Centrality on GPUs and Heterogeneous
Architectures](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.301.6142&rep=rep1&type=pdf) where authors investigate a set of novel techniques to make BC computations faster with CUDA on GPUs.

"Centrality metrics, such as betweenness and closeness, quantify how central a node is in a network. They have been used successfully to carry analyses for various purposes such as structural analysis of knowledge networks, power grid contingency analysis, quantifying importance in social networks, analysis of covert networks and decision/action networks, and even for finding the best store locations in cities." 

## Project structure
- `src/main.cu` contains main function
- `include/graph_pruned.h` contains implementation of `struct Graph` (storing Graph on CPU, pruning, and decomposition into CC), `struct ConnectedComponent` (for building and storing in CPU format, as well as GPU Stride-CSR format), and `struct BrandesData` (Stride-CSR format)
- `include/kernel_runner_pruned.h` contains implementation of `cuda_kernels`, `KernelRunnerMultiCCPruned` (struct that orchestrates algorithm), `BrandesDataDevice` (Stride-CSR format)
- `include/common.h` common definitions
- `include/utils.h`
- `include/utils_cuda.cuh` cuda specific utils
- `archive/` contains other implementation in a raw stage (lot's of debug/commented out/TODO code). I attach them as a proof of concept.

## Introduction

They idea is inspired by https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.301.6142&rep=rep1&type=pdf

- I have implemented and fined-tuned Algorithm 4 from the paper.
- On GPU graph is stored using Stride-CSR representation and virtual vertices are introduced, as in the paper.

To get the most our of the chosen algorithm and corresponding data representation I have designed several enchancements, such as:
- *internal virtual vertices* (it's not the same as virtual-vertices used in Stride-CSR)
- *connected components decomposition*
- *deg-1 vertices pruning* 

Overall, I've especially focused on fine-tuning memory layout to ensure best possible coalescing. In addition, I've moved update stages (for both `delta` and `bc` values) to the device, which resulted in crucial performance and made `total_memory_transfer_time` neglibile. Empirically, those improvements resulted in significant speed up on vast majority of graphs.

### High-level algorithm description:

1. Read graph into Graph struct.
2. Iteratively, until convergence: prune vertices of degree 1. Results in $G_1$.
3. From $G_1$ build $c$ subgraphs, where $c$ is number of connected components.
4. For each Connected Component: run Brandes algorithm (see below for details).

It's crucial that step 3 is performed after step 2. This leads to *compact memory layout* and great *memory coalescing*.

## Key enhancements:
### SoA memory layout instead of naive AoS.
### *Internal Virtual Vertices* 
- can be thought of as "deg-0" vertices removal
- it's performed twice:
    - first layer of mapping is "original vertex name in input graph -> graph internal vertex id".
    - second layer is introduced during CC creation. "graph internal vertex id -> CC internal vertex id".
- the idea is to assign next_free_id to newly encountered vertices. This ensures that in practive no vertices of degree 0 are stored (when writing bc to the result file, the inverse mapping is used to simply write 0 for those vertices).
- in the end, in a single connected component, data for all the vertices can be stored in a continuous array in which non of the vertices is "trivial" or "not-relevent", thus all the reads/writes are necessary, and much more likely to occur in a the same warp of threads.
- **Speed up:** this leads to compact memory composition on both *graph* and *connected component* level, especially the second one is important due to the usage of SoA. Among others, this reduces overall memory usage (both host and device), memory transport time, overall computation time (as there are less vertices to go through, even for lookup), and the most importantly, it leads to better memory coalescing.
- for details see `graph_pruned.h/Graph: vertex_to_id`, `graph_pruned.h/ConnectedComponent: vertex_to_id` fields.

### *Connected components decomposition*
-   Naive implementation of Algorithm 4 during kernel's execution traverses the whole grap at each step (in case of forward/backward kernels it's a huge overhead because those steps are repeated until BFS is finished). Trivial observation is that traversing graph from source corresponds exactly to traversing EXACTLY connected component for this source.
-   To leverage this fact, each connected component is stored and processed as independed graph, allowing for concise memory layout.
-    **Speed up:** This gives significant boost when $G$ is composed of many connected components, especially if each of them is "big", that is vertices are distributed uniformly across different CC. All the vertices that are processed are "relevant" with respect to current source.

### *Deg-1 vertices pruning*
-  Basically, it follows the procedure described in the paper. To make it compatible with *connected components decomposition* I had to dive deep into paper and change formulas a little bit.
-  Allows for processing huge graphs which in form are similar to "long paths". 
-  *Speed-up*:
    -  Firstly, it's achieved by the fact that problem size has been reduced. There are far less sources to be considered and vertices to be traversed at EACH step. In other words, each "per source" traverse is faster, as well as, there are less sources to consider.
    -  Secondly, this allows for superior memory layout, as vertices we are left with are of non-trivial degree, which improves probability that threads in the warp are working on the same vertex/neighbors (SoA layout), thus leading to better *memory coalescing*.


## Key ingredients
### CPU Graph/Connected Component representation:
- It's rather standard, based on STL library, in a way that allows very fast *pruning stage* and *connected components decomposition*.
- They key enhancement is addition of *virtual vertices mapping* described above.
- for details see `graph_pruned.h`. Especially `Graph: build()`, `Graph: create_cc()` and `ConnectedComponent: build()`.

### CPU Pruning step:
- Instead on moving pruning stage to GPU (as suggested in the paper) I've decided to stay on CPU. Justification is that it allows us to:
    -  get the most out of the STL library
    -  avoid moving graph back and forth between host/device. After each pruning stage, graph would have to be: moved back to host, re-built on host, moved to the device. It would be inefficient.
    -  at each pruning stage, the whole graph would be traversed, which would be wastefull as only vertices of deg-1 and their neighbors have to be updated (fast `unordered_set` manipulations are perfect for this kind of tasks).
    -  it's less error-prone
- Using, among others, `unordered_sets` I was able to achieve linear time pruning stage. 
- see `graph_pruned.h` for details. Especially functions: `prune_cpu()` and `build_deg_to_vertices()`.


### GPU ConnectedComponents representation:
- General scheme follows Stride-CSR (SoA) described in the paper - see `struct BrandesData` in `graph_pruned.h` and `struct BrandesDataDevice` in `kernel_runner_pruned.h` for details.
- Significant improvement is achieved via applied enchancements (described above), as they ensure that each field in memory "is relevant", and "non-trivial" with respect to current source/vertex/neighbor within connected component. Therefore all reads/writes within a single warp are much more likely to coalesce. Moreover, this compact representation minimizes the memory used, which in addition with moving updates of `bc` and `deltas` to device resulted in negligible `memory_transfer` times - see exemplary `results` below,

## GPU/CPU computations - Brandes algorithm

### Pseudocode
``` 
for each CC:
    copy data device
    for each source in current CC:
        initialize values on device
        while cont = true:
            forward_step_kernel()     
        delta_update_kernel()         
        while level > 1:
            backward_step_kernel()    
            level = level-1
        bc_update_kernel()            
    copy bc results to host
```


### CUDA kernels description:
#### forward_step_kernel
- follows implementation from the paper.
#### backward_step_kernel
- follows implementation from the paper.
#### delta_update_kernel
- to leverage GPU speed-up and avoid unnecessary movements of data (host <-> device) `delta's` values are updated on GPU.
- to avoid reading `sigma's` value twice corresponding value is cached (see code for details).

#### bc_update_kernel
- to leverage GPU speed-up and avoid unnecessary movements of data (host <-> device) `bc's` values are updated on GPU.
- to avoid unnecessary writes (updates) `bc` value is updated only if `update != 0` (see code for details).

As already mentioned, thanks to this implementation of `update kernels` total `memory_transfer_time` (in both directions) is neglibile, as for each CC, its representation is moved once to the device at the beginning, and final results are moved once in the end.

## Results comparison
I have run additional comparison results for approaches:
- multi-connected-components-pruned (decomposition into connected components + prunning)
- multi-connected-components (decomposition into connected components)
- single-connected-component (graph is treated as single CC, though it might not be connected)

Implementation for `mcc-pruned` is the one being compiled, and can be found in `src`, `include` directories.
Implementation for `mcc` can be found in `archive/multi-cc-no-pruning` (files that differ).
Implementation for `scc` can be found in `archive/single-cc-no-pruning` (files that differ).
Note: Please keep in mind that archive directory is mainly for archive purposes and contains rather messy code + a lot of commented out / debug code.
All of above use *internal virtual vertices* which boosts them significantly. Otherwise, for large graphs, they algorithm won't finish in reasonable time.

Based on my empirical testing, cuda-excel-calculator, and my local profiling tool I have decided to pick mdeg4, nth256 as my final setup.

Here I present some results. Additional results are stored in `results` folder. Naming convention:
- `mcc` stands for `multi-connected-components`
- `scc` stands for `single-connectedc-component`
- `mdeg` stands for `MDEG` hyperparameter for creating virtual vertices (as in the paper)
- `nth{val}` stands for `THREADS_PER_BLOCK` 

Topology of below graphs can be found on http://snap.stanford.edu/data/

As we can see below, speed up is significant for graphs with multiple CC, and where deg-1 pruning can be performed. However, this additional operations and decomposition does not slow down computations in other cases. Thus resulting in much better overall algorithm.

### mcc-prune-mdeg4-nth256

| test | wtime | kernel time | kernel + memory_transfer time |
|--- |--- |---|--- |
|as20000102.txt|1.280444064|575.135|629.245|
|ca-GrQc.txt|2.073422989|900.124|958.065|
|com-amazon-ungraph.txt|425.490112520|408370|413716|
|com-dblp-ungraph.txt|207.661284876|192933|197238|
|email-Enron.txt|8.095895277|5274.26|5629.11|
|facebook_combined.txt|1.489015967|693.414|744.437|
|graph-random-001.txt|9.853228337|972.572|1089.47|
|graph-random-002.txt|6.350137512|1333.77|1477.36|
|loc-gowalla.txt|73.452852078|64663.8|66574.5|
|graph_long_path.txt|6.526242292|0.093152|0.125376|

Note that this implementation allows for handling graphs that normally are not feasible e.g. `graph_long_path.txt` which is a graph of 1milion vertices connected in one long path. Pruning stage in no-time compresses it.

### mcc-mdeg4-nth256

| test | wtime | kernel time | kernel + memory_transfer time |
|--- |--- |---|--- |
|as20000102.txt|2.077557189|1087.79|1172.51|
|ca-GrQc.txt|2.258878356|1134.01|1205.42|
|com-amazon-ungraph.txt|490.699155496|472566|478719|
|com-dblp-ungraph.txt|261.881511633|245689|251348|
|email-Enron.txt|10.911580631|7950.25|8429.96|
|facebook_combined.txt|1.546048585|717.139|771.492|
|graph-random-001.txt|9.014721803|964.742|1084.52|
|graph-random-002.txt|6.081331511|1342.04|1481.26|
|loc-gowalla.txt|112.840204420|101955|104856|

### scc-mdeg4-nth256
| test | wtime | kernel time | kernel + memory_transfer time |
|--- |--- |---|--- |
|as20000102.txt|2.052223182|1077.2|1166.84|
|ca-GrQc.txt|2.000914568|1132.9|1200.7|
|com-amazon-ungraph.txt|489.156334185|471441|477596|
|com-dblp-ungraph.txt|261.495491053|245194|250850|
|email-Enron.txt|10.052875935|7889.38|8376.92|
|facebook_combined.txt|1.778788633|707.563|760.51|
|graph-random-001.txt|9.406165272|1418.62|1549.14|
|graph-random-002.txt|6.431886729|1735.86|1891.71|
|loc-gowalla.txt|112.571165645|101842|104743|



## Other ideas:

Here I list out other ideas I have tried and implemented. All of them resulted in worse times. However, I do admit that I haven't fine-tuned them, which means that the comparison is not 100% fair. Implementations for those ideas can be found in `archive/other`

- Heterogeneous computations as suggested in the paper. Namely, one-thread for handling GPU/CPU operations (as in single thread implementation). 7 additional threads for computing `bc` in parallel on `CPU`. It actually slowed down the process, thus I decided to focused on fine-tuning `pure-CUDA` solution (I've decided to focus on learning and understanding CUDA better).
- CUDA based solution with `deltas` && `bc` updates performed on `host`. This solution allowed me to see that moving data between host and device is a significant bottleneck, and it would be much better to move those updates into device.
- Multi-threaded CPU implementation of brandes algorithm
- Simple one-thread CPU implementation of brandes algorithm
- Probably multiple other things that I have already forgot :)

