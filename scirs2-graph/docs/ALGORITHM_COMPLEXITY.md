# Algorithm Complexity Reference for scirs2-graph

This document provides a comprehensive reference for the time and space complexity of all algorithms implemented in scirs2-graph.

## Notation

- **V**: Number of vertices (nodes) in the graph
- **E**: Number of edges in the graph  
- **k**: Parameter specific to the algorithm (e.g., number of shortest paths)
- **d**: Maximum degree of any vertex
- **D**: Diameter of the graph

## Graph Traversal Algorithms

### Breadth-First Search (BFS)
- **Function**: `breadth_first_search`
- **Time Complexity**: O(V + E)
- **Space Complexity**: O(V)
- **Notes**: Visits each vertex and edge exactly once

### Depth-First Search (DFS)
- **Function**: `depth_first_search`
- **Time Complexity**: O(V + E)
- **Space Complexity**: O(V) for the recursion stack
- **Notes**: Can be iterative or recursive implementation

### Bidirectional Search
- **Function**: `bidirectional_search`
- **Time Complexity**: O(b^(d/2)) where b is branching factor, d is depth
- **Space Complexity**: O(b^(d/2))
- **Notes**: Much faster than standard BFS for finding path between two specific nodes

## Shortest Path Algorithms

### Dijkstra's Algorithm
- **Function**: `shortest_path`
- **Time Complexity**: 
  - With binary heap: O((V + E) log V)
  - With Fibonacci heap: O(E + V log V)
- **Space Complexity**: O(V)
- **Notes**: Only works with non-negative edge weights

### A* Search
- **Function**: `astar_search`
- **Time Complexity**: O(E) in the worst case, often much better with good heuristic
- **Space Complexity**: O(V)
- **Notes**: Performance depends heavily on the quality of the heuristic function

### Floyd-Warshall Algorithm
- **Function**: `floyd_warshall`
- **Time Complexity**: O(V³)
- **Space Complexity**: O(V²)
- **Notes**: Computes all-pairs shortest paths, handles negative weights

### k-Shortest Paths
- **Function**: `k_shortest_paths`
- **Time Complexity**: O(k * V * (E + V log V))
- **Space Complexity**: O(k * V)
- **Notes**: Finds k shortest paths between two nodes

## Connectivity Algorithms

### Connected Components (Undirected)
- **Function**: `connected_components`
- **Time Complexity**: O(V + E)
- **Space Complexity**: O(V)
- **Notes**: Uses DFS or BFS internally

### Strongly Connected Components (Directed)
- **Function**: `strongly_connected_components`
- **Time Complexity**: O(V + E)
- **Space Complexity**: O(V)
- **Notes**: Uses Tarjan's or Kosaraju's algorithm

### Articulation Points
- **Function**: `articulation_points`
- **Time Complexity**: O(V + E)
- **Space Complexity**: O(V)
- **Notes**: Vertices whose removal increases connected components

### Bridges
- **Function**: `bridges`
- **Time Complexity**: O(V + E)
- **Space Complexity**: O(V)
- **Notes**: Edges whose removal increases connected components

## Centrality Algorithms

### Degree Centrality
- **Function**: `centrality` (with `CentralityType::Degree`)
- **Time Complexity**: O(V)
- **Space Complexity**: O(V)
- **Notes**: Simply counts edges per vertex

### Betweenness Centrality
- **Function**: `betweenness_centrality`
- **Time Complexity**: O(V * E) for unweighted, O(V * E + V² log V) for weighted
- **Space Complexity**: O(V + E)
- **Notes**: Measures how often a vertex lies on shortest paths

### Closeness Centrality
- **Function**: `closeness_centrality`
- **Time Complexity**: O(V * (E + V log V))
- **Space Complexity**: O(V)
- **Notes**: Based on average shortest path length

### Eigenvector Centrality
- **Function**: `eigenvector_centrality`
- **Time Complexity**: O(k * E) where k is iterations (typically small)
- **Space Complexity**: O(V)
- **Notes**: Uses power iteration method

### PageRank
- **Function**: `pagerank_centrality`
- **Time Complexity**: O(k * (V + E)) where k is iterations
- **Space Complexity**: O(V)
- **Notes**: Converges quickly in practice (k ≈ 20-100)

### Katz Centrality
- **Function**: `katz_centrality`
- **Time Complexity**: O(k * (V + E)) where k is iterations
- **Space Complexity**: O(V)
- **Notes**: Similar to eigenvector centrality with attenuation factor

### HITS Algorithm
- **Function**: `hits_algorithm`
- **Time Complexity**: O(k * E) where k is iterations
- **Space Complexity**: O(V)
- **Notes**: Computes hub and authority scores

## Spanning Tree Algorithms

### Minimum Spanning Tree (Kruskal's)
- **Function**: `minimum_spanning_tree`
- **Time Complexity**: O(E log E)
- **Space Complexity**: O(V)
- **Notes**: Uses union-find data structure

### Minimum Spanning Tree (Prim's)
- **Function**: Also available via `minimum_spanning_tree`
- **Time Complexity**: O(E log V) with binary heap
- **Space Complexity**: O(V)
- **Notes**: Better for dense graphs

## Community Detection Algorithms

### Louvain Method
- **Function**: `louvain_communities`
- **Time Complexity**: O(V log V) typically, O(V²) worst case
- **Space Complexity**: O(V)
- **Notes**: Greedy modularity optimization

### Label Propagation
- **Function**: `label_propagation`
- **Time Complexity**: O(k * E) where k is iterations (usually small)
- **Space Complexity**: O(V)
- **Notes**: Near-linear time community detection

### Infomap
- **Function**: `infomap_communities`
- **Time Complexity**: O(E * log V)
- **Space Complexity**: O(V)
- **Notes**: Based on information theory

### Fluid Communities
- **Function**: `fluid_communities`
- **Time Complexity**: O(k * E) where k is iterations
- **Space Complexity**: O(V)
- **Notes**: Propagates community labels like fluid

### Hierarchical Clustering
- **Function**: `hierarchical_communities`
- **Time Complexity**: O(V² log V)
- **Space Complexity**: O(V²)
- **Notes**: Builds dendrogram of communities

## Matching Algorithms

### Maximum Bipartite Matching
- **Function**: `maximum_bipartite_matching`
- **Time Complexity**: O(E * √V)
- **Space Complexity**: O(V)
- **Notes**: Uses Hopcroft-Karp algorithm

### Maximum Cardinality Matching
- **Function**: `maximum_cardinality_matching`
- **Time Complexity**: O(E * √V)
- **Space Complexity**: O(V)
- **Notes**: For general graphs, uses Edmonds' algorithm

### Stable Marriage
- **Function**: `stable_marriage`
- **Time Complexity**: O(V²)
- **Space Complexity**: O(V)
- **Notes**: Gale-Shapley algorithm

## Flow Algorithms

### Ford-Fulkerson (Max Flow)
- **Function**: `ford_fulkerson_max_flow`
- **Time Complexity**: O(E * f) where f is maximum flow value
- **Space Complexity**: O(V)
- **Notes**: Can be slow for large flow values

### Dinic's Algorithm (Max Flow)
- **Function**: `dinic_max_flow`
- **Time Complexity**: O(V² * E)
- **Space Complexity**: O(V + E)
- **Notes**: More efficient than Ford-Fulkerson in practice

### Push-Relabel (Max Flow)
- **Function**: `push_relabel_max_flow`
- **Time Complexity**: O(V³)
- **Space Complexity**: O(V²)
- **Notes**: Good for dense graphs

### Minimum Cut
- **Function**: `minimum_cut`
- **Time Complexity**: Same as max flow algorithm used
- **Space Complexity**: O(V)
- **Notes**: Derived from max flow computation

## Graph Coloring

### Greedy Coloring
- **Function**: `greedy_coloring`
- **Time Complexity**: O(V + E)
- **Space Complexity**: O(V)
- **Notes**: Not optimal but fast

### Chromatic Number (Exact)
- **Function**: `chromatic_number`
- **Time Complexity**: O(V * 2^V) worst case
- **Space Complexity**: O(V)
- **Notes**: NP-complete problem, exponential time

## Isomorphism Testing

### VF2 Algorithm
- **Function**: `are_graphs_isomorphic`
- **Time Complexity**: O(V! * V) worst case, often much better
- **Space Complexity**: O(V)
- **Notes**: State-space search with pruning

### Subgraph Isomorphism
- **Function**: `find_subgraph_matches`
- **Time Complexity**: O(V^k) where k is size of pattern
- **Space Complexity**: O(V)
- **Notes**: NP-complete problem

## Spectral Algorithms

### Laplacian Matrix
- **Function**: `laplacian`
- **Time Complexity**: O(V + E)
- **Space Complexity**: O(V²)
- **Notes**: Creates V×V matrix

### Spectral Radius
- **Function**: `spectral_radius`
- **Time Complexity**: O(V³) for exact, O(k * V²) for approximation
- **Space Complexity**: O(V²)
- **Notes**: Largest eigenvalue magnitude

### Normalized Cut
- **Function**: `normalized_cut`
- **Time Complexity**: O(V³)
- **Space Complexity**: O(V²)
- **Notes**: Spectral clustering algorithm

## Graph Generation

### Erdős–Rényi Random Graph
- **Function**: `erdos_renyi_graph`
- **Time Complexity**: O(V²) for G(n,p) model
- **Space Complexity**: O(V + E)
- **Notes**: Each edge exists with probability p

### Barabási–Albert Graph
- **Function**: `barabasi_albert_graph`
- **Time Complexity**: O(V * m) where m is edges per new node
- **Space Complexity**: O(V + E)
- **Notes**: Preferential attachment model

### Watts-Strogatz Small World
- **Function**: `watts_strogatz_graph`
- **Time Complexity**: O(V * k) where k is initial degree
- **Space Complexity**: O(V + E)
- **Notes**: Rewiring model for small-world networks

## Special Operations

### Graph Edit Distance
- **Function**: `graph_edit_distance`
- **Time Complexity**: O(V! * V) worst case
- **Space Complexity**: O(V²)
- **Notes**: NP-hard problem

### k-Core Decomposition
- **Function**: `k_core_decomposition`
- **Time Complexity**: O(V + E)
- **Space Complexity**: O(V)
- **Notes**: Linear time algorithm

### Motif Finding
- **Function**: `find_motifs`
- **Time Complexity**: O(V^k) where k is motif size
- **Space Complexity**: O(V)
- **Notes**: Searches for specific patterns

## Memory Considerations

### Sparse vs Dense Representations
- Adjacency List: O(V + E) space - better for sparse graphs
- Adjacency Matrix: O(V²) space - better for dense graphs
- scirs2-graph uses adjacency list (via petgraph)

### Large Graph Processing
- Streaming algorithms available for graphs that don't fit in memory
- Chunk-based processing for very large graphs
- External memory algorithms for disk-based graphs

## Parallel Complexity

Many algorithms have parallel implementations that can achieve better time complexity:

- BFS/DFS: O(D) time with O(V) processors
- Connected Components: O(log V) time with O(V + E) work
- PageRank: Near-linear speedup with multiple cores
- Community Detection: Varies by algorithm, typically good parallelization

## Practical Performance Notes

1. **Cache Effects**: Real performance often depends on cache locality
2. **Graph Structure**: Performance can vary significantly based on graph properties:
   - Sparse vs dense
   - Degree distribution
   - Clustering coefficient
   - Diameter

3. **Implementation Details**: 
   - scirs2-graph uses efficient data structures from petgraph
   - SIMD optimizations for certain operations
   - Parallel execution via Rayon when beneficial

4. **Algorithm Selection**:
   - For sparse graphs: prefer algorithms with O(V + E) complexity
   - For dense graphs: O(V²) algorithms may be acceptable
   - Consider approximation algorithms for NP-hard problems

## References

1. Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2009). Introduction to algorithms.
2. Newman, M. (2018). Networks (2nd ed.). Oxford University Press.
3. Sedgewick, R., & Wayne, K. (2011). Algorithms (4th ed.). Addison-Wesley.