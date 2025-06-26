# NetworkX to SciRS2-Graph Migration Guide

This guide helps users transition from NetworkX to scirs2-graph, providing side-by-side comparisons and practical migration examples.

## Table of Contents

1. [Key Differences](#key-differences)
2. [Installation and Setup](#installation-and-setup)
3. [Basic Graph Operations](#basic-graph-operations)
4. [Graph Construction](#graph-construction)
5. [Algorithm Migration](#algorithm-migration)
6. [Advanced Features](#advanced-features)
7. [Performance Considerations](#performance-considerations)
8. [API Mapping Reference](#api-mapping-reference)

## Key Differences

### Conceptual Differences

| Aspect | NetworkX | scirs2-graph |
|--------|----------|--------------|
| **Language** | Python | Rust |
| **Performance** | Interpreted, single-threaded by default | Compiled, parallel processing built-in |
| **Memory Model** | Python objects, higher overhead | Zero-cost abstractions, efficient memory layout |
| **Type Safety** | Dynamic typing | Static typing with compile-time guarantees |
| **Node IDs** | Any hashable object | Currently `usize` (integers) |
| **Edge Weights** | Any object | Numeric types (f32, f64, etc.) |
| **Error Handling** | Exceptions | Result<T, E> type |

### Design Philosophy

- **NetworkX**: Flexibility and ease of use, supports any Python object as nodes/edges
- **scirs2-graph**: Performance and type safety, optimized for numerical computing

## Installation and Setup

### NetworkX
```python
pip install networkx
import networkx as nx
```

### scirs2-graph
```toml
# Cargo.toml
[dependencies]
scirs2-graph = "0.1.0-beta.1"
```

```rust
use scirs2_graph::{Graph, DiGraph, algorithms, measures};
use scirs2_core::error::CoreResult;
```

## Basic Graph Operations

### Creating a Graph

**NetworkX:**
```python
# Undirected graph
G = nx.Graph()

# Directed graph
G = nx.DiGraph()

# With initial data
G = nx.Graph([(1, 2), (2, 3)])
```

**scirs2-graph:**
```rust
// Undirected graph
let mut g = Graph::new();

// Directed graph
let mut g = DiGraph::new();

// With initial data
let mut g = Graph::new();
g.add_edge(1, 2, 1.0)?;
g.add_edge(2, 3, 1.0)?;
```

### Adding Nodes and Edges

**NetworkX:**
```python
# Add single node
G.add_node(1)
G.add_node('A', color='red')

# Add multiple nodes
G.add_nodes_from([2, 3, 4])
G.add_nodes_from(['B', 'C'], color='blue')

# Add edges
G.add_edge(1, 2, weight=4.0)
G.add_edges_from([(1, 3), (2, 4)], weight=2.0)
```

**scirs2-graph:**
```rust
// Add single node
g.add_node(1);

// Add multiple nodes
for node in vec![2, 3, 4] {
    g.add_node(node);
}

// Add edges
g.add_edge(1, 2, 4.0)?;
for (u, v) in vec![(1, 3), (2, 4)] {
    g.add_edge(u, v, 2.0)?;
}
```

### Accessing Graph Properties

**NetworkX:**
```python
# Number of nodes and edges
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()

# Check if node/edge exists
has_node = 1 in G
has_edge = G.has_edge(1, 2)

# Get neighbors
neighbors = list(G.neighbors(1))

# Get degree
degree = G.degree(1)
```

**scirs2-graph:**
```rust
// Number of nodes and edges
let num_nodes = g.node_count();
let num_edges = g.edge_count();

// Check if node/edge exists
let has_node = g.has_node(1);
let has_edge = g.has_edge(1, 2);

// Get neighbors
let neighbors: Vec<usize> = g.neighbors(1).collect();

// Get degree
let degree = g.degree(1);
```

## Graph Construction

### From Edge List

**NetworkX:**
```python
# From list of tuples
edges = [(1, 2), (2, 3), (3, 4)]
G = nx.from_edgelist(edges)

# From file
G = nx.read_edgelist('edges.txt')
```

**scirs2-graph:**
```rust
// From vector of tuples
let edges = vec![(1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0)];
let mut g = Graph::new();
for (u, v, w) in edges {
    g.add_edge(u, v, w)?;
}

// From file
use scirs2_graph::io;
let g = io::read_edgelist("edges.txt", false)?;
```

### Graph Generators

**NetworkX:**
```python
# Random graphs
G = nx.erdos_renyi_graph(100, 0.1)
G = nx.barabasi_albert_graph(100, 3)
G = nx.watts_strogatz_graph(100, 6, 0.3)

# Regular graphs
G = nx.complete_graph(10)
G = nx.cycle_graph(20)
G = nx.path_graph(15)
```

**scirs2-graph:**
```rust
use scirs2_graph::generators;

// Random graphs
let g = generators::erdos_renyi_graph(100, 0.1, None)?;
let g = generators::barabasi_albert_graph(100, 3, None)?;
let g = generators::watts_strogatz_graph(100, 6, 0.3, None)?;

// Regular graphs
let g = generators::complete_graph(10);
let g = generators::cycle_graph(20);
let g = generators::path_graph(15);
```

## Algorithm Migration

### Shortest Paths

**NetworkX:**
```python
# Single source shortest path
path = nx.shortest_path(G, source=1, target=5)
length = nx.shortest_path_length(G, source=1, target=5)

# Dijkstra's algorithm
path = nx.dijkstra_path(G, 1, 5)
distances = nx.single_source_dijkstra_path_length(G, 1)

# All pairs shortest paths
paths = dict(nx.all_pairs_shortest_path(G))
```

**scirs2-graph:**
```rust
use scirs2_graph::algorithms::{shortest_path, dijkstra};

// Single source shortest path
let path = shortest_path(&g, 1, 5)?;
let length = shortest_path_length(&g, 1, 5)?;

// Dijkstra's algorithm
let (distances, predecessors) = dijkstra(&g, 1)?;
let path = reconstruct_path(&predecessors, 1, 5);

// All pairs shortest paths
let paths = all_pairs_shortest_paths(&g)?;
```

### Centrality Measures

**NetworkX:**
```python
# Degree centrality
dc = nx.degree_centrality(G)

# Betweenness centrality
bc = nx.betweenness_centrality(G)

# Closeness centrality
cc = nx.closeness_centrality(G)

# PageRank
pr = nx.pagerank(G, alpha=0.85)
```

**scirs2-graph:**
```rust
use scirs2_graph::measures::centrality;

// Degree centrality
let dc = centrality::degree_centrality(&g)?;

// Betweenness centrality
let bc = centrality::betweenness_centrality(&g)?;

// Closeness centrality
let cc = centrality::closeness_centrality(&g)?;

// PageRank
let pr = centrality::pagerank(&g, 0.85, None)?;
```

### Community Detection

**NetworkX:**
```python
import networkx.algorithms.community as nx_comm

# Modularity-based communities
communities = nx_comm.louvain_communities(G)

# Label propagation
communities = nx_comm.label_propagation_communities(G)

# K-clique communities
communities = list(nx_comm.k_clique_communities(G, 3))
```

**scirs2-graph:**
```rust
use scirs2_graph::algorithms::community;

// Modularity-based communities
let communities = community::louvain_communities(&g)?;

// Label propagation
let communities = community::label_propagation(&g)?;

// K-clique communities
let communities = community::k_clique_communities(&g, 3)?;
```

### Graph Traversal

**NetworkX:**
```python
# BFS
bfs_tree = nx.bfs_tree(G, 1)
bfs_edges = list(nx.bfs_edges(G, 1))

# DFS
dfs_tree = nx.dfs_tree(G, 1)
dfs_edges = list(nx.dfs_edges(G, 1))

# Connected components
components = list(nx.connected_components(G))
```

**scirs2-graph:**
```rust
use scirs2_graph::algorithms::{bfs, dfs, connectivity};

// BFS
let bfs_tree = bfs::bfs_tree(&g, 1)?;
let bfs_order: Vec<usize> = bfs::bfs_order(&g, 1)?;

// DFS
let dfs_tree = dfs::dfs_tree(&g, 1)?;
let dfs_order: Vec<usize> = dfs::dfs_order(&g, 1)?;

// Connected components
let components = connectivity::connected_components(&g)?;
```

## Advanced Features

### Attributed Graphs

**NetworkX:**
```python
# Node attributes
G.add_node(1, label='A', weight=10)
G.nodes[1]['color'] = 'red'

# Edge attributes
G.add_edge(1, 2, weight=4.0, capacity=10)
G[1][2]['flow'] = 5

# Access attributes
node_data = G.nodes[1]
edge_data = G[1][2]
```

**scirs2-graph:**
```rust
use scirs2_graph::AttributedGraph;

// Create attributed graph
let mut g = AttributedGraph::new();

// Node attributes (using a struct)
#[derive(Clone)]
struct NodeData {
    label: String,
    weight: f64,
}

g.add_node_with_attrs(1, NodeData {
    label: "A".to_string(),
    weight: 10.0,
});

// Edge attributes
g.add_edge_with_attrs(1, 2, 4.0, EdgeData {
    capacity: 10,
    flow: 5,
})?;
```

### Subgraphs and Views

**NetworkX:**
```python
# Subgraph
nodes = [1, 2, 3]
subgraph = G.subgraph(nodes)

# Edge subgraph
edges = [(1, 2), (2, 3)]
edge_subgraph = G.edge_subgraph(edges)

# Node-induced subgraph
induced = G.induced_subgraph(nodes)
```

**scirs2-graph:**
```rust
use scirs2_graph::views;

// Subgraph
let nodes = vec![1, 2, 3];
let subgraph = g.subgraph(&nodes)?;

// Edge subgraph
let edges = vec![(1, 2), (2, 3)];
let edge_subgraph = g.edge_subgraph(&edges)?;

// Node-induced subgraph
let induced = g.induced_subgraph(&nodes)?;
```

### Graph I/O

**NetworkX:**
```python
# Write to file
nx.write_graphml(G, 'graph.graphml')
nx.write_gml(G, 'graph.gml')
nx.write_edgelist(G, 'edges.txt')

# Read from file
G = nx.read_graphml('graph.graphml')
G = nx.read_gml('graph.gml')
G = nx.read_edgelist('edges.txt')
```

**scirs2-graph:**
```rust
use scirs2_graph::io;

// Write to file
io::write_graphml(&g, "graph.graphml")?;
io::write_gml(&g, "graph.gml")?;
io::write_edgelist(&g, "edges.txt")?;

// Read from file
let g = io::read_graphml("graph.graphml")?;
let g = io::read_gml("graph.gml")?;
let g = io::read_edgelist("edges.txt", false)?;
```

## Performance Considerations

### When to Use scirs2-graph

1. **Large graphs** (>100K nodes/edges)
2. **Performance-critical applications**
3. **Numerical computations** on graphs
4. **Parallel processing** requirements
5. **Memory-constrained** environments

### Migration Tips for Performance

1. **Batch operations**: Instead of adding edges one by one, collect them and add in batches
2. **Use appropriate data structures**: Choose between Graph, DiGraph, and specialized types
3. **Enable parallelism**: Use the `parallel` feature for automatic speedups
4. **Preallocate when possible**: Use `with_capacity` constructors for known sizes

### Performance Comparison Example

**NetworkX (Python):**
```python
import time

# Creating a large graph
start = time.time()
G = nx.barabasi_albert_graph(10000, 5)
pr = nx.pagerank(G)
end = time.time()
print(f"Time: {end - start:.2f}s")
```

**scirs2-graph (Rust):**
```rust
use std::time::Instant;

// Creating a large graph
let start = Instant::now();
let g = generators::barabasi_albert_graph(10000, 5, None)?;
let pr = algorithms::pagerank(&g, 0.85, None)?;
let duration = start.elapsed();
println!("Time: {:.2f}s", duration.as_secs_f64());
```

Typical speedup: 10-100x depending on algorithm and graph size.

## API Mapping Reference

### Core Graph Methods

| NetworkX | scirs2-graph |
|----------|--------------|
| `G.add_node(n)` | `g.add_node(n)` |
| `G.add_edge(u, v)` | `g.add_edge(u, v, weight)?` |
| `G.remove_node(n)` | `g.remove_node(n)?` |
| `G.remove_edge(u, v)` | `g.remove_edge(u, v)?` |
| `G.number_of_nodes()` | `g.node_count()` |
| `G.number_of_edges()` | `g.edge_count()` |
| `G.nodes()` | `g.nodes()` |
| `G.edges()` | `g.edges()` |
| `G.degree(n)` | `g.degree(n)` |
| `G.neighbors(n)` | `g.neighbors(n)` |
| `n in G` | `g.has_node(n)` |
| `G.has_edge(u, v)` | `g.has_edge(u, v)` |

### Algorithms

| NetworkX | scirs2-graph |
|----------|--------------|
| `nx.shortest_path()` | `algorithms::shortest_path()` |
| `nx.dijkstra_path()` | `algorithms::dijkstra()` |
| `nx.bellman_ford_path()` | `algorithms::bellman_ford()` |
| `nx.connected_components()` | `algorithms::connected_components()` |
| `nx.strongly_connected_components()` | `algorithms::strongly_connected_components()` |
| `nx.minimum_spanning_tree()` | `algorithms::minimum_spanning_tree()` |
| `nx.maximum_flow()` | `algorithms::ford_fulkerson_max_flow()` |
| `nx.pagerank()` | `algorithms::pagerank()` |
| `nx.betweenness_centrality()` | `measures::centrality::betweenness_centrality()` |
| `nx.clustering()` | `measures::clustering_coefficient()` |
| `nx.triangles()` | `algorithms::count_triangles()` |

### Graph Generators

| NetworkX | scirs2-graph |
|----------|--------------|
| `nx.complete_graph()` | `generators::complete_graph()` |
| `nx.cycle_graph()` | `generators::cycle_graph()` |
| `nx.path_graph()` | `generators::path_graph()` |
| `nx.star_graph()` | `generators::star_graph()` |
| `nx.wheel_graph()` | `generators::wheel_graph()` |
| `nx.grid_graph()` | `generators::grid_graph()` |
| `nx.erdos_renyi_graph()` | `generators::erdos_renyi_graph()` |
| `nx.barabasi_albert_graph()` | `generators::barabasi_albert_graph()` |
| `nx.watts_strogatz_graph()` | `generators::watts_strogatz_graph()` |

### I/O Operations

| NetworkX | scirs2-graph |
|----------|--------------|
| `nx.read_graphml()` | `io::read_graphml()` |
| `nx.write_graphml()` | `io::write_graphml()` |
| `nx.read_gml()` | `io::read_gml()` |
| `nx.write_gml()` | `io::write_gml()` |
| `nx.read_edgelist()` | `io::read_edgelist()` |
| `nx.write_edgelist()` | `io::write_edgelist()` |
| `nx.read_adjlist()` | `io::read_adjlist()` |
| `nx.write_adjlist()` | `io::write_adjlist()` |

## Common Migration Patterns

### Pattern 1: Graph Analysis Pipeline

**NetworkX:**
```python
# Load graph
G = nx.read_edgelist('network.txt')

# Basic analysis
print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
print(f"Density: {nx.density(G)}")

# Centrality analysis
degree_cent = nx.degree_centrality(G)
between_cent = nx.betweenness_centrality(G)

# Find communities
communities = nx.community.louvain_communities(G)

# Export results
nx.write_graphml(G, 'analyzed.graphml')
```

**scirs2-graph:**
```rust
use scirs2_graph::{io, measures, algorithms};

// Load graph
let g = io::read_edgelist("network.txt", false)?;

// Basic analysis
println!("Nodes: {}", g.node_count());
println!("Edges: {}", g.edge_count());
println!("Density: {}", measures::graph_density(&g));

// Centrality analysis
let degree_cent = measures::centrality::degree_centrality(&g)?;
let between_cent = measures::centrality::betweenness_centrality(&g)?;

// Find communities
let communities = algorithms::community::louvain_communities(&g)?;

// Export results
io::write_graphml(&g, "analyzed.graphml")?;
```

### Pattern 2: Dynamic Graph Manipulation

**NetworkX:**
```python
G = nx.Graph()

# Build graph dynamically
for i in range(100):
    G.add_node(i, weight=i*0.1)
    if i > 0:
        G.add_edge(i-1, i, weight=1.0)
    if i % 10 == 0 and i > 0:
        G.add_edge(0, i, weight=2.0)

# Modify graph
for u, v in list(G.edges()):
    if G[u][v]['weight'] < 1.5:
        G.remove_edge(u, v)
```

**scirs2-graph:**
```rust
let mut g = Graph::new();

// Build graph dynamically
for i in 0..100 {
    g.add_node(i);
    if i > 0 {
        g.add_edge(i-1, i, 1.0)?;
    }
    if i % 10 == 0 && i > 0 {
        g.add_edge(0, i, 2.0)?;
    }
}

// Modify graph
let edges_to_remove: Vec<_> = g.edges()
    .filter(|&&(u, v, w)| w < 1.5)
    .map(|&(u, v, _)| (u, v))
    .collect();

for (u, v) in edges_to_remove {
    g.remove_edge(u, v)?;
}
```

## Troubleshooting Common Issues

### Issue 1: Node Type Differences

NetworkX allows any hashable object as nodes, while scirs2-graph currently uses integers.

**Solution**: Create a mapping between your node identifiers and integers:

```rust
use std::collections::HashMap;

let mut node_map = HashMap::new();
let mut reverse_map = HashMap::new();
let mut next_id = 0;

// Map string nodes to integers
for node_name in ["A", "B", "C"] {
    node_map.insert(node_name, next_id);
    reverse_map.insert(next_id, node_name);
    next_id += 1;
}
```

### Issue 2: Missing NetworkX Features

Some specialized NetworkX algorithms may not yet be implemented in scirs2-graph.

**Solution**: 
1. Check if an equivalent algorithm exists under a different name
2. Implement the algorithm using available primitives
3. Consider contributing the implementation to scirs2-graph

### Issue 3: Error Handling

NetworkX raises exceptions while scirs2-graph returns Results.

**Solution**: Use `?` operator or match on Results:

```rust
// Using ? operator
let path = algorithms::shortest_path(&g, 1, 5)?;

// Explicit handling
match algorithms::shortest_path(&g, 1, 5) {
    Ok(path) => println!("Path found: {:?}", path),
    Err(e) => println!("No path exists: {}", e),
}
```

## Conclusion

Migrating from NetworkX to scirs2-graph offers significant performance benefits for large-scale graph processing. While there are some API differences and current limitations (like node types), the performance gains and type safety make it an excellent choice for production systems requiring high-performance graph analytics.

For the latest updates and additional examples, see the [scirs2-graph documentation](https://docs.rs/scirs2-graph).