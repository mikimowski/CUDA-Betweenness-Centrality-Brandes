#ifndef BETWEENNESS_CENTRALITY_BRANDES_UTILS_H
#define BETWEENNESS_CENTRALITY_BRANDES_UTILS_H

#include <fstream>
#include <iostream>
#include <vector>
#include "common.h"


template<typename T>
void write_vector(std::vector<T> &arr) {
    for (int i = 0; i < arr.size(); ++i) {
        std::cout << arr[i] << "\n";
    }
    std::cout << std::endl;
}

template<typename T>
void write_array(T *arr, int N) {
    for (int i = 0; i < N; ++i) {
        std::cout << arr[i] << "\n";
    }
    std::cout << std::endl;
}


void write_adjacency_list(AdjacencyList &adjacency_list) {
    for (auto & it : adjacency_list) {
        printf("%d: ", it.first);
        for (int it_neigh : it.second) {
            printf("%d ", it_neigh);
        }
        printf("\n");
    }
}


void write_result(const double *arr, int N) {
    for (int i = 0; i < N; ++i) {
        std::cout << arr[i] << std::endl;
    }
}

void write_result(const double *arr, int N, std::string &outfile) {
    std::ofstream file;
    file.open(outfile);
    for (int i = 0; i < N; ++i) {
        file << arr[i] << std::endl;
    }
    file.close();
}

#endif //BETWEENNESS_CENTRALITY_BRANDES_UTILS_H
