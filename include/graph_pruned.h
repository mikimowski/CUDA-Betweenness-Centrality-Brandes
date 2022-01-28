#ifndef BETWEENNESS_CENTRALITY_BRANDES_GRAPH_PRUNED_H
#define BETWEENNESS_CENTRALITY_BRANDES_GRAPH_PRUNED_H

#include "common.h"
#include <unordered_map>
#include <unordered_set>
#include <cassert>
#include <fstream>
#include <queue>
#include <cstring>
#include <iostream>

#define MDEG 4

/**
 * @param vertex_deg
 * @return number of virtual vertices corresponding to vertex. Based on MDEG and number deg(vertex) == n_neighbors
 */
int get_number_of_virtual_vertices(int vertex_deg) {
    return (vertex_deg + MDEG - 1) / MDEG;
}

struct BrandesData {
    // BrandesData algorithm's variables
    // Id of virtual_vertex for current real vertex. Offset within virtual_vertices assigned to real vertex.
    // Virtual vertex number withing the real vertex.
    int *offset = nullptr; // virtual_offset
    size_t offset_size{};
    // Mapping: virtual vertex -> real vertex
    int *vmap = nullptr; // virtual_to_real
    size_t vmap_size{};
    // Number of virtual vertices for given real. Mapping: real vertex -> cnt_virtual
    int *nvir = nullptr; // cnt_virtuals
    size_t nvir_size{};
    // Mapping: real vertex -> idx of first neighbor in adjacency_list
    int *ptrs = nullptr;
    size_t ptrs_size{};
    // Mapping: virtual vertex -> neighbor idx
    int *adjs = nullptr;
    size_t adjs_size{};
    double *bc = nullptr;
    size_t bc_size{};
    // Reach array from pruning stage. int64_t is necessary for big graphs.
    int64_t *reach = nullptr;
    size_t reach_size{};

    BrandesData() = default;

    BrandesData(int n_real_vertices, int n_virtual_vertices, int n_edges) :
            offset_size(n_virtual_vertices * sizeof(int)),
            vmap_size(n_virtual_vertices * sizeof(int)),
            nvir_size(n_real_vertices * sizeof(int)),
            ptrs_size((n_real_vertices + 1) * sizeof(int)),
            adjs_size(n_edges * sizeof(int)),
            reach_size(n_real_vertices * sizeof(int64_t)),
            bc_size(n_real_vertices * sizeof(double)) {
        offset = new int[n_virtual_vertices];
        vmap = new int[n_virtual_vertices];
        nvir = new int[n_real_vertices];
        ptrs = new int[n_real_vertices + 1];
        adjs = new int[n_edges];
        bc = new double[n_real_vertices];
        memset(bc, 0.0, bc_size);
        reach = new int64_t[n_real_vertices];
        memset(reach, 1, reach_size); // initialize reach to 1
    }

    ~BrandesData() {
        // Deleting nullptr has no effect
        delete offset;
        delete vmap;
        delete nvir;
        delete ptrs;
        delete adjs;
        delete bc;
        delete reach;
    }

    // Move assignment
    BrandesData &operator=(BrandesData &&rhs) noexcept {
        offset_size = rhs.offset_size;
        offset = rhs.offset;
        rhs.offset = nullptr;

        vmap_size = rhs.vmap_size;
        vmap = rhs.vmap;
        rhs.vmap = nullptr;

        nvir_size = rhs.nvir_size;
        nvir = rhs.nvir;
        rhs.nvir = nullptr;

        ptrs_size = rhs.ptrs_size;
        ptrs = rhs.ptrs;
        rhs.ptrs = nullptr;

        adjs_size = rhs.adjs_size;
        adjs = rhs.adjs;
        rhs.adjs = nullptr;

        bc_size = rhs.bc_size;
        bc = rhs.bc;
        rhs.bc = nullptr;

        reach_size = rhs.reach_size;
        reach = rhs.reach;
        rhs.reach = nullptr;

        return *this;
    }
};


struct ConnectedComponent {
    BrandesData brandes;

    // internal
    int n_edges = 0;
    int n_virtual_vertices = 0;
    int n_original_vertices = 0;

    // Mapping from external (e.g. original) vertex's name to internal vertex id.
    // It's used to "squash" graph, that is, not to store
    // maps original id to "first layer of virtual vertices".
    // Note that "external" for Graph means the original input name,
    // whereas "external" for CC would be id from the Graph
    std::unordered_map<VertexName, VertexId> vertex_to_id;
    // Mapping: vertex -> vertex's neighbors
    AdjacencyList adjacency_list;

    int get_vertex_id(VertexName vertex_name) {
        auto it = vertex_to_id.find(vertex_name);
        if (it == vertex_to_id.end()) {
            n_original_vertices = std::max(n_original_vertices - 1, vertex_name) + 1;
            int idx = vertex_to_id.size();
            vertex_to_id[vertex_name] = idx;
            return idx;
        } else {
            return it->second;
        }
    }

    // Used to avoid adding duplicated edges when building CC from Graph
    void add_edge_directed(int v1_name, int v2_name) {
        if (v1_name != v2_name) {
            int v1_id = get_vertex_id(v1_name);
            int v2_id = get_vertex_id(v2_name);
            this->adjacency_list[v1_id].insert(v2_id);
            this->n_edges += 1;
        }
    }

    void recalculate_n_virtual_vertices() {
        n_virtual_vertices = 0;
        for (AdjacencyList::iterator it_adj = adjacency_list.begin(); it_adj != adjacency_list.end(); ++it_adj) {
            n_virtual_vertices += get_number_of_virtual_vertices(it_adj->second.size());
        }
    }

    void build() {
//        printf("Starting building CC...\n");
        recalculate_n_virtual_vertices();
        brandes = BrandesData(get_n_real_vertices(), get_n_virtual_vertices(), get_n_edges());

        int offset_idx = 0, adjs_idx = 0, v;
        for (AdjacencyList::iterator it_adj = adjacency_list.begin(); it_adj != adjacency_list.end(); ++it_adj) {
            v = it_adj->first;

            const int num_neighbors = adjacency_list[v].size();
            brandes.nvir[v] = get_number_of_virtual_vertices(num_neighbors);
            // Fill offset && vmap arrays for current vertex
            for (int i = 0; i < brandes.nvir[v]; ++i) {
                brandes.offset[offset_idx] = i;
                brandes.vmap[offset_idx] = v;
                ++offset_idx;
            }

            // Set pointer to 'first neighbor' location in adjs_list
            brandes.ptrs[v] = adjs_idx;
            for (auto it = adjacency_list[v].begin(); it != adjacency_list[v].end(); ++it) {
                brandes.adjs[adjs_idx] = *it;
                adjs_idx++;
            }
        }
        // Set last pointer to point after the array <- dummy sentry
        brandes.ptrs[get_n_real_vertices()] = adjs_idx;
    }


    /**
     * Number of vertices actually being stored in adjacency_list && vertex_to_id
     * === n_sources
     */
    int get_n_real_vertices() const {
        return vertex_to_id.size();
    }

    /**
     * Number of original input vertices
     */
    int get_n_original_vertices() const {
        return n_original_vertices;
    }

    int get_n_virtual_vertices() const {
        return n_virtual_vertices;
    }

    int get_n_edges() const {
        return n_edges;
    }

    double get_bc(int v_name) {
        return brandes.bc[vertex_to_id[v_name]];
    }

    void set_reach(int v_name, int64_t val) {
        brandes.reach[vertex_to_id[v_name]] = val;
    }

    /********************************************** STATS UTILS *******************************************************/
    void write_adjacency_list() {
        for (auto &it : this->adjacency_list) {
            printf("%d: ", it.first);
            for (int it_neigh : it.second) {
                printf("%d ", it_neigh);
            }
            printf("\n");
        }
    }


    void write_stats() {
        uint64_t max_out_deg = 0, min_out_deg = n_virtual_vertices;
        int cnt_details = 10, out_deg;
        int deg_cnt[cnt_details];
        memset(deg_cnt, 0, sizeof(int) * 10);

        for (auto &it : this->adjacency_list) {
            out_deg = it.second.size();
            max_out_deg = std::max(max_out_deg, it.second.size());
            min_out_deg = std::min(min_out_deg, it.second.size());
            if (out_deg < cnt_details)
                deg_cnt[out_deg]++;
        }
        printf("############ Stats ############\n");
        printf("original vertices: %d\n", n_original_vertices);
        printf("real vertices: %lu\n", vertex_to_id.size());
        printf("virtual vertices: %d\n", n_virtual_vertices);
        printf("edges: %d\n", n_edges);
        printf("min out_deg: %lu\n", min_out_deg);
        printf("max out_deg: %lu\n", max_out_deg);
        printf("average out_deg: %f\n", double(n_edges) / vertex_to_id.size());
        for (int i = 0; i < cnt_details; ++i) {
            printf("cnt of deg_out %d: %d\n", i, deg_cnt[i]);
        }
    }
};

/**
 * Graph is built in the following way:
 * 1. add all edges
 * 2. create_cc()
 */
struct Graph {
    // internal
    // Edges are calculated as if they were directed edges.
    // Thus n_edges == sum(deg(v)) for all v
    int n_edges = 0;
    int n_original_vertices = 0;

    // Mapping: vertex_id -> bc
    std::vector<double> pruned_bc;
    // Mapping: vertex_id -> reach
    std::vector<int64_t> pruned_reach;


    /** Maps original id to "first layer of virtual vertices" - that is different than virtual vertices used in cuda_kernels
    * This is used to "shrink" the graph and ensure that "meaningful" memory is allocated.
    * Namely, vertices with deg(v) == 0, are not stored at all.
    * E.g. for graph given as:
     * 4 5
     * 5 6
     * 6 7
     * We will have corresponding representation:
     * 4 -> 0
     * 5 -> 1
     * 6 -> 2
     * 7 -> 3
     * All internal operations on the graph operate on VertexId. VertexName is used only for external representation
    */
    std::unordered_map<VertexName, VertexId> vertex_to_id;

    // Mapping: vertex -> vertex's neighbors
    AdjacencyList adjacency_list;

    std::vector<ConnectedComponent> connected_components;

    // Mapping: VertexId -> its connected component id
    std::unordered_map<VertexId, int> vertex_to_cc;

    int get_vertex_id(VertexName vertex_name) {
        std::unordered_map<VertexName, VertexId>::iterator it = vertex_to_id.find(vertex_name);
        // If seen for the first time
        if (it == vertex_to_id.end()) {
            // n_original_vertices is defined as max vertex name encountered in the edge description
            n_original_vertices = std::max(n_original_vertices - 1, vertex_name) + 1;
            int idx = vertex_to_id.size();
            vertex_to_id[vertex_name] = idx;
            return idx;
        } else {
            return it->second;
        }
    }

    void add_edge(int v1_name, int v2_name) {
        if (v1_name != v2_name) {
            // Get VertexId
            int v1_id = get_vertex_id(v1_name);
            int v2_id = get_vertex_id(v2_name);
            this->adjacency_list[v1_id].insert(v2_id);
            this->adjacency_list[v2_id].insert(v1_id);
            this->n_edges += 2;
        }
    }

    void build(bool as_single_cc = false) {
        vertex_to_cc.clear();
        if (as_single_cc) {
            // Not optimal way to preserve the same interface but simulating graph as a single CC.
            // Note that in this case graph would actually have multiple CC -> might not be connected
            int cc = 0;
            for (auto &it : vertex_to_id) {
                vertex_to_cc[it.first] = cc;
            }
            create_cc(cc + 1);
        } else {
            int num_cc = discover_cc();
            create_cc(num_cc);
            for (auto &cc: connected_components) {
                cc.build();
            }
        }

        // Initialize reach values calculated during pruning
        int v;
        for (auto &it : vertex_to_cc) {
            v = it.first;
            connected_components[vertex_to_cc[v]].set_reach(v, pruned_reach[v]);
        }
    }

    /**
     * Each CC will have it's own mapping "from original vertices to idx",
     * but now, those original vertices are w.r.t. this graph, not the input.
     */
    void create_cc(int num_cc) {
        connected_components = std::vector<ConnectedComponent>(num_cc);
        int cc, v;
        // for each vertex, IN INCREASING ORDER <- preserving what we were given
        for (auto &it : adjacency_list) {
//            for (int v = 0; v < adjacency_list.size(); v++) {
//                if (is_pruned())
            v = it.first;
            cc = vertex_to_cc[v];
            // Put it into its CC and add all edges <- again, we preserve ordering of the edges (lexi)
            for (int neigh: adjacency_list[v]) {
                connected_components[cc].add_edge_directed(v, neigh);
            }
        }
    }

    /**
     * Builds mapping from vertices to corresponding CC
     * VertexId -> CC
     * @return number of connected components
     */
    int discover_cc() {
        int cc = 0;
        for (auto it_adj = adjacency_list.begin(); it_adj != adjacency_list.end(); ++it_adj) {
            // If new CC
            if (vertex_to_cc.find(it_adj->first) == vertex_to_cc.end()) {
                mark_cc_bfs(it_adj->first, cc);
                ++cc;
            }
        }
        return cc;
    }

    /***
     * Writes BC to the output file
     */
    void write_bc(std::string &outfile) {
        std::ofstream file;
        file.open(outfile);
        file.precision(dbl::max_digits10);
        int vertex_id;
        double bc;

        for (int v_name = 0; v_name < n_original_vertices; ++v_name) {
            if (vertex_to_id.find(v_name) == vertex_to_id.end()) {
                file << 0 << std::endl;
            } else {
                // Mapping from external to internal ID
                vertex_id = vertex_to_id[v_name];
                bc = pruned_bc[vertex_id];
                if (!is_pruned(vertex_id)) {
                    bc += connected_components[vertex_to_cc[vertex_id]].get_bc(vertex_id);
                }
                file << bc << std::endl;
            }
        }
        file.close();
    }

    int get_n_real_vertices() const {
        return vertex_to_id.size();
    }


    /**
     * vertex_id  not a name!
     * @param vertex_id
     * @return
     */
    bool is_pruned(int vertex_id) {
        return adjacency_list.find(vertex_id) == adjacency_list.end();
    }

    bool is_in_cc_with_two_vertices(int vertex_id) {
        if (adjacency_list[vertex_id].size() == 1) {
            return adjacency_list[*adjacency_list[vertex_id].begin()].size() == 1;
        }
        return false;
    }


    /************************************************* PRUNING ********************************************************/
    void prune_cpu() {
        pruned_bc.assign(vertex_to_id.size(), 0.0);
        pruned_reach.assign(vertex_to_id.size(), 1);

        auto cc_to_size = prune_discover_cc();
        auto deg_to_vertices = build_deg_to_vertices();

        int64_t N;
        int total_pruned = 0, pruned = 1;
        std::vector<VertexId> to_remove;
        while (pruned > 0) {
            pruned = 0;
            to_remove.resize(0);
            std::unordered_set<int> next_iteration_deg1;

            // For each vertex with deg(v) == 1
            for (auto tr: deg_to_vertices[1]) {
                // If in CC with two vertices, then skip (not pruned, so we can calculate it easily later on)
                if (is_in_cc_with_two_vertices(tr)) {
                    continue;
                }

                // As far as I understand the papers, N should be the INITIAL number of vertices in CC (not in the graph!)
                N = cc_to_size[vertex_to_cc[tr]] - 1;
                pruned_bc[tr] += (pruned_reach[tr] - 1) * (N - pruned_reach[tr]);
                // Update all neighbors
                for (auto neigh: adjacency_list[tr]) {
                    adjacency_list[neigh].erase(tr);
                    // Update deg_to_vertices
                    size_t new_deg = adjacency_list[neigh].size();
                    deg_to_vertices[new_deg + 1].erase(neigh);
                    if (new_deg == 1) {
                        next_iteration_deg1.insert(neigh);
                    } else {
                        deg_to_vertices[new_deg].insert(neigh);
                    }

                    pruned_reach[neigh] += pruned_reach[tr];
                    pruned_bc[neigh] += pruned_reach[tr] * (N - pruned_reach[tr] - 1);
                }

                // Remove vertex
                adjacency_list.erase(tr);
                total_pruned++;
                pruned++;
                to_remove.push_back(tr);
            }

            deg_to_vertices[1] = std::move(next_iteration_deg1);
        }

        // Clear mapping as CC should be built again after pruning (excluding pruned vertices)
        vertex_to_cc.clear();
    }

    /**
     * @return Mapping: deg(n) -> set of vertices with this deg
     */
    std::unordered_map<int, std::unordered_set<VertexId>> build_deg_to_vertices() {
        std::unordered_map<int, std::unordered_set<VertexId>> deg_to_vertices;
        for (const auto &it : adjacency_list) {
            deg_to_vertices[it.second.size()].insert(it.first);
        }
        return deg_to_vertices;
    }

    /**
     * Builds mapping from vertices to corresponding CC
     * @return Mapping from CC to its size
     * */
    std::unordered_map<VertexId, int> prune_discover_cc() {
        std::unordered_map<VertexId, int> cc_to_size;
        int cc = 0;
        for (auto &it_adj : adjacency_list) {
            // If new CC
            if (vertex_to_cc.find(it_adj.first) == vertex_to_cc.end()) {
                cc_to_size[cc] = mark_cc_bfs(it_adj.first, cc);
                ++cc;
            }
        }
        return cc_to_size;
    }

    /**
     * Maps all the vertices in v's CC to the same CC
     * @return Number of processed vertices == size of CC
     */
    int mark_cc_bfs(int v, int cc) {
        std::queue<int> q;
        q.push(v);
        int cnt = 0;

        while (!q.empty()) {
            cnt++;
            v = q.front();
            q.pop();
            for (int neigh: adjacency_list[v]) {
                // if not visited == not assigned cc
                if (vertex_to_cc.find(neigh) == vertex_to_cc.end()) {
                    vertex_to_cc[neigh] = cc;
                    q.push(neigh);
                } else {
                    assert(vertex_to_cc[neigh] == cc);
                }
            }
        }
        return cnt;
    }

    /********************************************** STATS UTILS *******************************************************/
    void write_stats() {
        printf("Graph:\n"
               "connected components: %lu\n"
               "original vertices: %d\n"
               "real vertices: %d\n", connected_components.size(), n_original_vertices, get_n_real_vertices());
        for (int i = 0; i < connected_components.size(); i++) {
            printf("CC %d: ", i);
            connected_components[i].write_stats();
        }
    }

    void write_adjacency_list() {
        for (auto &it : this->adjacency_list) {
            printf("%d: ", it.first);
            for (int it_neigh : it.second) {
                printf("%d ", it_neigh);
            }
            printf("\n");
        }
    }
};


Graph read_graph(const std::string &filepath) {
    struct Graph graph;
    std::ifstream infile(filepath);
    int u, v;
    // Assumption: Two integers per line describing the edges
    while (infile >> u >> v) {
        graph.add_edge(u, v);
    }

    return graph;
}

#endif //BETWEENNESS_CENTRALITY_BRANDES_GRAPH_PRUNED_H
