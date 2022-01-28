#ifndef BETWEENNESS_CENTRALITY_BRANDES_COMMON_H
#define BETWEENNESS_CENTRALITY_BRANDES_COMMON_H

#include <map>
#include <set>
#include <vector>
#include <limits>

typedef int VertexName;
typedef int VertexId;
typedef std::map<VertexId, std::set<VertexId>> AdjacencyList;
typedef std::numeric_limits<double> dbl;

#endif //BETWEENNESS_CENTRALITY_BRANDES_COMMON_H
