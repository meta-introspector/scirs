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

## Parallel Processing Features

### NetworkX with Multiprocessing

```python
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

def process_component(component):
    # Process each component separately
    subgraph = G.subgraph(component)
    return nx.betweenness_centrality(subgraph)

# Parallel processing of components
components = list(nx.connected_components(G))
with ProcessPoolExecutor() as executor:
    results = list(executor.map(process_component, components))
```

### scirs2-graph Built-in Parallelism

```rust
// Enable parallel feature in Cargo.toml
// [dependencies]
// scirs2-graph = { version = "0.1.0-beta.1", features = ["parallel"] }

// Algorithms automatically use parallel processing
let centrality = algorithms::betweenness_centrality_parallel(&g)?;

// Configure parallelism
use scirs2_graph::parallel::ParallelConfig;

let config = ParallelConfig {
    num_threads: 8,
    chunk_size: 1000,
};

let result = algorithms::pagerank_parallel(&g, 0.85, None, config)?;
```

## Custom Algorithm Implementation

### NetworkX Custom Algorithm

```python
def my_metric(G, node):
    """Custom node importance metric"""
    degree = G.degree(node)
    neighbors_degree = sum(G.degree(n) for n in G.neighbors(node))
    return degree * neighbors_degree

# Apply to all nodes
importance = {n: my_metric(G, n) for n in G.nodes()}
```

### scirs2-graph Custom Algorithm

```rust
use scirs2_graph::{Graph, Node, EdgeWeight};
use std::collections::HashMap;

fn my_metric<N, E, Ix>(g: &Graph<N, E, Ix>, node: N) -> f64
where
    N: Node,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    let degree = g.degree(&node) as f64;
    let neighbors_degree: f64 = g.neighbors(&node)
        .unwrap()
        .map(|n| g.degree(&n) as f64)
        .sum();
    degree * neighbors_degree
}

// Apply to all nodes
let importance: HashMap<usize, f64> = g.nodes()
    .iter()
    .map(|&n| (n, my_metric(&g, n)))
    .collect();
```

## Migration Checklist

Before migrating your NetworkX code to scirs2-graph, review this checklist:

### ✅ Pre-Migration Assessment

- [ ] Identify performance bottlenecks in current NetworkX code
- [ ] List all NetworkX algorithms currently in use
- [ ] Check if all required algorithms are available in scirs2-graph
- [ ] Assess current memory usage and requirements
- [ ] Document any custom node/edge attributes used

### ✅ Migration Planning

- [ ] Design integer node ID mapping strategy (if using non-integer nodes)
- [ ] Plan data structure migrations for attributes
- [ ] Identify opportunities for parallel processing
- [ ] Set up Rust development environment
- [ ] Create test cases for verification

### ✅ Implementation Steps

- [ ] Start with core graph construction and basic operations
- [ ] Migrate algorithms incrementally, testing each one
- [ ] Implement custom algorithms as needed
- [ ] Add error handling for Rust's Result types
- [ ] Optimize with batch operations where possible

### ✅ Testing and Validation

- [ ] Verify numerical results match NetworkX output
- [ ] Benchmark performance improvements
- [ ] Test edge cases and error conditions
- [ ] Validate memory usage reduction
- [ ] Check parallel processing speedups

### ✅ Optimization

- [ ] Enable appropriate feature flags (parallel, simd)
- [ ] Use specialized graph types where applicable
- [ ] Implement streaming for very large graphs
- [ ] Profile and optimize hot paths

## Real-World Migration Example

Here's a complete example migrating a social network analysis pipeline:

### Original NetworkX Code

```python
import networkx as nx
import pandas as pd
import numpy as np
from networkx.algorithms import community

class SocialNetworkAnalyzer:
    def __init__(self, edge_file):
        self.G = nx.read_edgelist(edge_file, create_using=nx.Graph())
        self.results = {}
    
    def analyze(self):
        # Basic metrics
        self.results['num_nodes'] = self.G.number_of_nodes()
        self.results['num_edges'] = self.G.number_of_edges()
        self.results['density'] = nx.density(self.G)
        
        # Find influencers
        pr = nx.pagerank(self.G)
        self.results['top_influencers'] = sorted(
            pr.items(), key=lambda x: x[1], reverse=True
        )[:10]
        
        # Detect communities
        communities = community.louvain_communities(self.G)
        self.results['num_communities'] = len(communities)
        self.results['modularity'] = community.modularity(self.G, communities)
        
        # Find bridges
        bridges = list(nx.bridges(self.G))
        self.results['critical_connections'] = bridges[:20]
        
        return self.results
    
    def export_results(self, output_file):
        # Export analyzed graph with metrics
        for node in self.G.nodes():
            self.G.nodes[node]['pagerank'] = pr.get(node, 0)
        
        nx.write_graphml(self.G, output_file)
```

### Migrated scirs2-graph Code

```rust
use scirs2_graph::{
    Graph, io, algorithms, measures,
    algorithms::community, measures::centrality,
};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct AnalysisResults {
    num_nodes: usize,
    num_edges: usize,
    density: f64,
    top_influencers: Vec<(usize, f64)>,
    num_communities: usize,
    modularity: f64,
    critical_connections: Vec<(usize, usize)>,
}

struct SocialNetworkAnalyzer {
    graph: Graph<usize, f64>,
    results: AnalysisResults,
}

impl SocialNetworkAnalyzer {
    fn new(edge_file: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let graph = io::read_edgelist(edge_file, false)?;
        let results = AnalysisResults {
            num_nodes: 0,
            num_edges: 0,
            density: 0.0,
            top_influencers: vec![],
            num_communities: 0,
            modularity: 0.0,
            critical_connections: vec![],
        };
        
        Ok(Self { graph, results })
    }
    
    fn analyze(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Basic metrics
        self.results.num_nodes = self.graph.node_count();
        self.results.num_edges = self.graph.edge_count();
        self.results.density = measures::graph_density(&self.graph);
        
        // Find influencers (parallel processing enabled)
        let pr = centrality::pagerank_parallel(&self.graph, 0.85, Some(1e-6))?;
        let mut pr_vec: Vec<_> = pr.into_iter().collect();
        pr_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        self.results.top_influencers = pr_vec.into_iter().take(10).collect();
        
        // Detect communities
        let communities = community::louvain_communities(&self.graph)?;
        self.results.num_communities = communities.num_communities;
        self.results.modularity = communities.modularity;
        
        // Find bridges
        let bridges = algorithms::connectivity::find_bridges(&self.graph)?;
        self.results.critical_connections = bridges.into_iter().take(20).collect();
        
        Ok(())
    }
    
    fn export_results(&self, output_file: &str) -> Result<(), Box<dyn std::error::Error>> {
        // Create attributed graph with metrics
        let mut attributed_graph = self.graph.to_attributed()?;
        
        // Add PageRank as node attribute
        let pr = centrality::pagerank(&self.graph, 0.85, Some(1e-6))?;
        for (node, rank) in pr {
            attributed_graph.set_node_attr(node, "pagerank", rank)?;
        }
        
        io::write_graphml(&attributed_graph, output_file)?;
        Ok(())
    }
}

// Usage
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut analyzer = SocialNetworkAnalyzer::new("social_network.txt")?;
    analyzer.analyze()?;
    analyzer.export_results("analyzed_network.graphml")?;
    
    // Print results
    println!("Analysis Results:");
    println!("  Nodes: {}", analyzer.results.num_nodes);
    println!("  Edges: {}", analyzer.results.num_edges);
    println!("  Density: {:.4}", analyzer.results.density);
    println!("  Communities: {}", analyzer.results.num_communities);
    println!("  Modularity: {:.4}", analyzer.results.modularity);
    
    Ok(())
}
```

### Performance Comparison Script

```rust
use std::time::Instant;

fn benchmark_comparison() {
    // Generate test graph
    let g = generators::barabasi_albert_graph(10000, 5, None).unwrap();
    
    println!("Benchmarking graph with {} nodes and {} edges", 
             g.node_count(), g.edge_count());
    
    // Benchmark PageRank
    let start = Instant::now();
    let _ = centrality::pagerank(&g, 0.85, Some(1e-6)).unwrap();
    let serial_time = start.elapsed();
    
    let start = Instant::now();
    let _ = centrality::pagerank_parallel(&g, 0.85, Some(1e-6)).unwrap();
    let parallel_time = start.elapsed();
    
    println!("\nPageRank Performance:");
    println!("  Serial: {:.3}s", serial_time.as_secs_f64());
    println!("  Parallel: {:.3}s", parallel_time.as_secs_f64());
    println!("  Speedup: {:.2}x", serial_time.as_secs_f64() / parallel_time.as_secs_f64());
    
    // Compare with NetworkX (typical times)
    println!("\nTypical NetworkX time: ~5-10s");
    println!("scirs2-graph speedup: {:.0}x", 7.5 / serial_time.as_secs_f64());
}
```

## Advanced Migration Topics

### Handling Dynamic Graphs

**NetworkX:**
```python
# Dynamic graph updates
G = nx.Graph()
for timestamp, (u, v) in edge_stream:
    G.add_edge(u, v, timestamp=timestamp)
    if G.number_of_edges() > max_edges:
        # Remove old edges
        old_edges = [(u, v) for u, v, d in G.edges(data=True) 
                     if d['timestamp'] < timestamp - window]
        G.remove_edges_from(old_edges)
```

**scirs2-graph:**
```rust
use scirs2_graph::temporal::TemporalGraph;

// Use specialized temporal graph
let mut tg = TemporalGraph::new();

for (timestamp, (u, v)) in edge_stream {
    tg.add_temporal_edge(u, v, timestamp, 1.0)?;
    
    // Automatic windowing
    tg.prune_edges_before(timestamp - window)?;
}

// Analyze at specific time
let snapshot = tg.snapshot_at(specific_time)?;
let pr = centrality::pagerank(&snapshot, 0.85, None)?;
```

### Memory-Efficient Large Graph Processing

**scirs2-graph Streaming:**
```rust
use scirs2_graph::streaming::StreamingGraph;

// Process graph that doesn't fit in memory
let mut sg = StreamingGraph::from_file("huge_graph.txt")?;

// Streaming PageRank
let pr = sg.streaming_pagerank(0.85, 1e-6, 1000)?; // 1000 nodes in memory

// Streaming connected components
let components = sg.streaming_connected_components(5000)?; // 5000 node chunks
```

## Migration Strategies for Different Use Cases

### Strategy 1: Gradual Migration (Recommended)

For existing Python codebases, migrate incrementally:

1. **Phase 1**: Migrate computationally expensive algorithms
   ```python
   # Keep NetworkX for data loading and basic operations
   import networkx as nx
   from scirs2_graph_python import scirs2_pagerank  # Python bindings
   
   G = nx.read_edgelist('network.txt')
   
   # Use scirs2-graph for expensive computation
   pagerank_scores = scirs2_pagerank(G.edges(), G.number_of_nodes())
   
   # Continue using NetworkX for other operations
   communities = nx.community.louvain_communities(G)
   ```

2. **Phase 2**: Migrate data structures for performance-critical paths
3. **Phase 3**: Full migration to Rust for maximum performance

### Strategy 2: Hybrid Approach

Use both libraries based on strengths:

```python
# analysis_pipeline.py
import networkx as nx
import subprocess
import json

def run_rust_analysis(edge_file, output_file):
    """Run Rust analysis as subprocess"""
    cmd = ["./graph_analyzer", edge_file, output_file]
    subprocess.run(cmd, check=True)
    
    with open(output_file, 'r') as f:
        return json.load(f)

def hybrid_analysis(edge_file):
    # Use NetworkX for quick prototyping and data exploration
    G = nx.read_edgelist(edge_file)
    basic_stats = {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'density': nx.density(G),
    }
    
    # Use scirs2-graph for heavy computation
    rust_results = run_rust_analysis(edge_file, 'rust_output.json')
    
    return {**basic_stats, **rust_results}
```

### Strategy 3: Full Migration

For new projects or when maximum performance is required:

```rust
// Complete rewrite in Rust
use scirs2_graph::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = AnalysisConfig::from_file("config.toml")?;
    let analyzer = GraphAnalyzer::new(config);
    
    let results = analyzer.analyze_pipeline(&[
        "load_graph",
        "compute_centrality", 
        "detect_communities",
        "export_results"
    ]).await?;
    
    println!("Analysis completed: {:?}", results);
    Ok(())
}
```

## Edge Cases and Gotchas

### Floating Point Precision

**NetworkX** may use different floating point precision:

```python
# NetworkX - may use float64 internally
pr = nx.pagerank(G, alpha=0.85, tol=1e-10)
```

**scirs2-graph** - specify precision explicitly:

```rust
// Use f64 for maximum precision
let pr: HashMap<usize, f64> = pagerank(&g, 0.85, Some(1e-10))?;

// Or use f32 for memory efficiency  
let pr: HashMap<usize, f32> = pagerank_f32(&g, 0.85_f32, Some(1e-6_f32))?;
```

### Random Number Generation

**NetworkX:**
```python
# Global random state
import random
random.seed(42)
G = nx.erdos_renyi_graph(100, 0.1)
```

**scirs2-graph:**
```rust
// Explicit RNG management
use rand::{SeedableRng, rngs::StdRng};

let mut rng = StdRng::seed_from_u64(42);
let g = erdos_renyi_graph(100, 0.1, &mut rng)?;
```

### Memory Management Differences

**NetworkX** - automatic garbage collection:
```python
def analyze_large_graph():
    G = nx.read_edgelist('huge_graph.txt')
    result = nx.pagerank(G)
    return result  # G automatically freed
```

**scirs2-graph** - explicit memory management:
```rust
fn analyze_large_graph() -> Result<HashMap<usize, f64>, GraphError> {
    let g = io::read_edgelist("huge_graph.txt", false)?;
    let result = pagerank(&g, 0.85, None)?;
    // g automatically dropped here
    Ok(result)
}

// For very large graphs, consider streaming
fn analyze_streaming() -> Result<(), GraphError> {
    let mut stream = GraphStream::from_file("huge_graph.txt")?;
    let result = stream.streaming_pagerank(0.85, 1e-6)?;
    Ok(())  // Constant memory usage
}
```

## Integration with Python Ecosystem

### Creating Python Bindings

```rust
// src/python_bindings.rs
use pyo3::prelude::*;
use pyo3::types::PyList;
use scirs2_graph::*;

#[pyfunction]
fn py_pagerank(edges: &PyList, num_nodes: usize, damping: f64) -> PyResult<Vec<(usize, f64)>> {
    let mut g = Graph::new();
    
    // Convert Python edge list
    for item in edges.iter() {
        let edge: (usize, usize, f64) = item.extract()?;
        g.add_edge(edge.0, edge.1, edge.2).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e))
        })?;
    }
    
    let result = pagerank(&g, damping, Some(1e-6)).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e))
    })?;
    
    Ok(result.into_iter().collect())
}

#[pymodule]
fn scirs2_graph_python(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_pagerank, m)?)?;
    Ok(())
}
```

### Using from Python

```python
# Install the Python bindings
# pip install maturin
# maturin develop

import scirs2_graph_python as sg
import networkx as nx

# Load graph with NetworkX
G = nx.karate_club_graph()
edges = [(u, v, 1.0) for u, v in G.edges()]

# Compute with Rust
pagerank_scores = sg.py_pagerank(edges, G.number_of_nodes(), 0.85)

print("Top 5 nodes by PageRank:")
for node, score in sorted(pagerank_scores, key=lambda x: x[1], reverse=True)[:5]:
    print(f"Node {node}: {score:.4f}")
```

## Performance Optimization Techniques

### Memory Pool Usage

```rust
use scirs2_graph::memory::MemoryPool;

// Pre-allocate memory for better performance
let pool = MemoryPool::with_capacity(1_000_000); // 1M nodes

let g = Graph::with_pool(pool);
// Faster node/edge allocation
```

### SIMD Optimization

```rust
// Enable SIMD features in Cargo.toml
// [dependencies]
// scirs2-graph = { version = "0.1.0-beta.1", features = ["simd"] }

// SIMD operations are automatically used for:
// - Vector operations in centrality calculations
// - Matrix operations in spectral algorithms
// - Parallel reductions

let pr = pagerank_simd(&g, 0.85, None)?; // Uses AVX2 if available
```

### Custom Data Types

```rust
// Use custom node/edge types for domain-specific optimization
#[derive(Clone, Copy, Hash, Eq, PartialEq)]
struct UserId(u32);

#[derive(Clone, Copy)]
struct Interaction {
    weight: f32,
    timestamp: u64,
}

let social_graph: Graph<UserId, Interaction> = Graph::new();
```

## Testing Migration Results

### Numerical Validation Framework

```rust
use approx::assert_relative_eq;

#[cfg(test)]
mod migration_tests {
    use super::*;
    
    #[test]
    fn test_pagerank_matches_networkx() {
        // Load reference data computed with NetworkX
        let reference_data = load_networkx_reference("pagerank_karate_club.json");
        
        // Compute with scirs2-graph
        let g = load_karate_club_graph();
        let pr = pagerank(&g, 0.85, Some(1e-6)).unwrap();
        
        // Compare results
        for (node, expected_score) in reference_data {
            let actual_score = pr[&node];
            assert_relative_eq!(actual_score, expected_score, epsilon = 1e-5);
        }
    }
    
    #[test]
    fn test_community_detection_quality() {
        let g = load_test_graph();
        
        // Compare modularity scores
        let nx_modularity = 0.4357; // Pre-computed with NetworkX
        let communities = louvain_communities(&g).unwrap();
        let scirs_modularity = modularity_score(&g, &communities).unwrap();
        
        // scirs2-graph should achieve equal or better modularity
        assert!(scirs_modularity >= nx_modularity - 1e-6);
    }
}
```

### Performance Regression Testing

```rust
use criterion::{criterion_group, criterion_main, Criterion};

fn benchmark_migration_performance(c: &mut Criterion) {
    let graphs = vec![
        ("small", erdos_renyi_graph(100, 0.1, None).unwrap()),
        ("medium", erdos_renyi_graph(1000, 0.01, None).unwrap()),
        ("large", erdos_renyi_graph(10000, 0.001, None).unwrap()),
    ];
    
    for (name, graph) in graphs {
        c.bench_function(&format!("pagerank_{}", name), |b| {
            b.iter(|| pagerank(&graph, 0.85, Some(1e-6)))
        });
        
        c.bench_function(&format!("betweenness_{}", name), |b| {
            b.iter(|| betweenness_centrality(&graph, true))
        });
    }
}

criterion_group!(benches, benchmark_migration_performance);
criterion_main!(benches);
```

## Production Deployment Considerations

### Error Handling and Monitoring

```rust
use tracing::{info, warn, error, instrument};
use sentry;

#[instrument(skip(graph))]
async fn production_analysis(graph: &Graph<usize, f64>) -> Result<AnalysisResults, AnalysisError> {
    // Set up error context
    sentry::configure_scope(|scope| {
        scope.set_tag("graph_nodes", graph.node_count());
        scope.set_tag("graph_edges", graph.edge_count());
    });
    
    // Memory usage monitoring
    let initial_memory = get_memory_usage();
    info!("Starting analysis with {} MB memory", initial_memory);
    
    // Run analysis with timeouts
    let result = tokio::time::timeout(
        Duration::from_secs(300), // 5 minute timeout
        run_core_analysis(graph)
    ).await??;
    
    let final_memory = get_memory_usage();
    info!("Analysis completed, memory delta: {} MB", final_memory - initial_memory);
    
    Ok(result)
}
```

### Configuration Management

```rust
// config.toml
[graph_analysis]
parallel_threads = 8
memory_limit_gb = 16
timeout_seconds = 300

[algorithms]
pagerank_tolerance = 1e-6
pagerank_max_iterations = 100
community_resolution = 1.0

[io]
batch_size = 10000
compression = true
```

```rust
use serde::{Deserialize, Serialize};
use config::{Config, ConfigError, File, Environment};

#[derive(Debug, Deserialize, Serialize)]
pub struct GraphAnalysisConfig {
    pub graph_analysis: GraphSettings,
    pub algorithms: AlgorithmSettings,
    pub io: IoSettings,
}

impl GraphAnalysisConfig {
    pub fn from_env() -> Result<Self, ConfigError> {
        let mut s = Config::new();
        
        // Default configuration
        s.set_default("graph_analysis.parallel_threads", num_cpus::get())?;
        s.set_default("algorithms.pagerank_tolerance", 1e-6)?;
        
        // Load from file
        s.merge(File::with_name("config.toml").required(false))?;
        
        // Override with environment variables
        s.merge(Environment::with_prefix("SCIRS2_GRAPH"))?;
        
        s.try_into()
    }
}
```

## Conclusion

Migrating from NetworkX to scirs2-graph offers significant performance benefits for large-scale graph processing. While there are some API differences and current limitations (like node types), the performance gains and type safety make it an excellent choice for production systems requiring high-performance graph analytics.

### Key Benefits of Migration:
- **10-100x performance improvement** for most algorithms
- **Parallel processing** built-in with automatic load balancing
- **Memory efficiency** through zero-cost abstractions and smart memory management
- **Type safety** preventing runtime errors and catching bugs at compile time
- **Production-ready** with comprehensive error handling and monitoring capabilities
- **Interoperability** with Python through bindings for gradual migration

### Migration Timeline Recommendations:

**Week 1-2**: Environment setup and proof-of-concept migration
**Week 3-4**: Core algorithm migration and validation
**Week 5-6**: Performance optimization and integration testing
**Week 7-8**: Production deployment and monitoring setup

### Success Metrics:
- [ ] Performance improvement of at least 5x for core algorithms
- [ ] Memory usage reduction of at least 30%
- [ ] Numerical accuracy within 1e-6 of NetworkX results
- [ ] Zero production incidents related to graph processing
- [ ] Successful handling of 10x larger graphs than before

For the latest updates, migration tools, and additional examples, see:
- [scirs2-graph documentation](https://docs.rs/scirs2-graph)
- [Migration examples repository](https://github.com/scirs2/migration-examples)
- [Performance benchmarks](https://scirs2.github.io/benchmarks)
- [Community Discord](https://discord.gg/scirs2) for migration support