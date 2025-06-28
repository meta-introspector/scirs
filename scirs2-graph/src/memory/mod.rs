//! Memory profiling and optimization utilities for graph data structures
//!
//! This module provides tools to analyze and optimize memory usage in graph operations.

pub mod compact;

pub use compact::{BitPackedGraph, CSRGraph, CompressedAdjacencyList, HybridGraph, MemmapGraph};

use crate::{DiGraph, Graph};
use std::collections::HashMap;
use std::mem;

/// Memory usage statistics for a graph
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Total memory used by the graph structure (bytes)
    pub total_bytes: usize,
    /// Memory used by node storage
    pub node_bytes: usize,
    /// Memory used by edge storage
    pub edge_bytes: usize,
    /// Memory used by adjacency lists
    pub adjacency_bytes: usize,
    /// Overhead from allocator metadata
    pub overhead_bytes: usize,
    /// Memory efficiency (useful data / total memory)
    pub efficiency: f64,
}

/// Memory profiler for graph structures
pub struct MemoryProfiler;

impl MemoryProfiler {
    /// Calculate memory statistics for an undirected graph
    pub fn profile_graph(graph: &Graph) -> MemoryStats {
        let node_count = graph.node_count();
        let edge_count = graph.edge_count();

        // Calculate node storage
        // Each node is stored as usize in a Vec
        let node_bytes = node_count * mem::size_of::<usize>() + mem::size_of::<Vec<usize>>(); // Vec overhead

        // Calculate adjacency list storage
        // Each node has a Vec of (neighbor, weight) pairs
        let mut adjacency_bytes = 0;
        for node in 0..node_count {
            let neighbors = graph.neighbors(node).count();
            adjacency_bytes += neighbors * mem::size_of::<(usize, f64)>() // (neighbor, weight)
                + mem::size_of::<Vec<(usize, f64)>>(); // Vec overhead per node
        }

        // Calculate edge storage (if separate edge list is maintained)
        let edge_bytes = edge_count * mem::size_of::<(usize, usize, f64)>();

        // Estimate allocator overhead (typically 8-16 bytes per allocation)
        let allocation_count = node_count + 1; // nodes + main structure
        let overhead_bytes = allocation_count * 16;

        let total_bytes = node_bytes + adjacency_bytes + edge_bytes + overhead_bytes;
        let useful_bytes = node_bytes + adjacency_bytes;
        let efficiency = useful_bytes as f64 / total_bytes as f64;

        MemoryStats {
            total_bytes,
            node_bytes,
            edge_bytes,
            adjacency_bytes,
            overhead_bytes,
            efficiency,
        }
    }

    /// Calculate memory statistics for a directed graph
    pub fn profile_digraph<N, E>(graph: &DiGraph<N, E>) -> MemoryStats {
        let node_count = graph.node_count();
        let edge_count = graph.edge_count();

        // Similar to undirected but with separate in/out adjacency lists
        let node_bytes = node_count * mem::size_of::<usize>() + mem::size_of::<Vec<usize>>();

        // Both in-edges and out-edges storage
        let mut adjacency_bytes = 0;
        for node in graph.nodes() {
            let out_neighbors = graph.neighbors(&node).unwrap_or(vec![]).len();
            let in_neighbors = out_neighbors; // For undirected graphs, in=out
            adjacency_bytes += (out_neighbors + in_neighbors) * mem::size_of::<(usize, f64)>()
                + 2 * mem::size_of::<Vec<(usize, f64)>>(); // Two Vecs per node
        }

        let edge_bytes = edge_count * mem::size_of::<(usize, usize, f64)>();

        let allocation_count = node_count * 2 + 1; // in/out vecs + main structure
        let overhead_bytes = allocation_count * 16;

        let total_bytes = node_bytes + adjacency_bytes + edge_bytes + overhead_bytes;
        let useful_bytes = node_bytes + adjacency_bytes;
        let efficiency = useful_bytes as f64 / total_bytes as f64;

        MemoryStats {
            total_bytes,
            node_bytes,
            edge_bytes,
            adjacency_bytes,
            overhead_bytes,
            efficiency,
        }
    }

    /// Estimate memory usage for a graph of given size
    pub fn estimate_memory(nodes: usize, edges: usize, directed: bool) -> usize {
        let avg_degree = if nodes > 0 {
            edges as f64 / nodes as f64
        } else {
            0.0
        };

        // Base node storage
        let node_bytes = nodes * mem::size_of::<usize>();

        // Adjacency list storage
        let edge_entry_size = mem::size_of::<(usize, f64)>();
        let adjacency_multiplier = if directed { 2.0 } else { 1.0 };
        let adjacency_bytes =
            (edges as f64 * adjacency_multiplier * edge_entry_size as f64) as usize;

        // Vec overhead (capacity often > size)
        let vec_overhead =
            nodes * mem::size_of::<Vec<(usize, f64)>>() * if directed { 2 } else { 1 };

        // Allocator overhead
        let overhead = (nodes + edges / 100) * 16;

        node_bytes + adjacency_bytes + vec_overhead + overhead
    }

    /// Analyze memory fragmentation in the graph
    pub fn analyze_fragmentation(graph: &Graph) -> FragmentationReport {
        let mut degree_distribution = HashMap::new();
        let mut total_capacity = 0;
        let mut total_used = 0;

        for node in 0..graph.node_count() {
            let degree = graph.degree(node);
            *degree_distribution.entry(degree).or_insert(0) += 1;

            // Estimate Vec capacity vs actual usage
            // Vecs typically grow by 2x when resizing
            let capacity = degree.next_power_of_two().max(4);
            total_capacity += capacity;
            total_used += degree;
        }

        let fragmentation = if total_capacity > 0 {
            1.0 - (total_used as f64 / total_capacity as f64)
        } else {
            0.0
        };

        FragmentationReport {
            degree_distribution,
            total_capacity,
            total_used,
            fragmentation_ratio: fragmentation,
            wasted_bytes: (total_capacity - total_used) * mem::size_of::<(usize, f64)>(),
        }
    }
}

/// Report on memory fragmentation in graph structures
#[derive(Debug)]
pub struct FragmentationReport {
    /// Distribution of node degrees
    pub degree_distribution: HashMap<usize, usize>,
    /// Total capacity allocated for adjacency lists
    pub total_capacity: usize,
    /// Total capacity actually used
    pub total_used: usize,
    /// Fragmentation ratio (0.0 = no fragmentation, 1.0 = all wasted)
    pub fragmentation_ratio: f64,
    /// Estimated wasted bytes due to over-allocation
    pub wasted_bytes: usize,
}

/// Memory-optimized graph builder
pub struct OptimizedGraphBuilder {
    nodes: Vec<usize>,
    edges: Vec<(usize, usize, f64)>,
    estimated_edges_per_node: Option<usize>,
}

impl OptimizedGraphBuilder {
    /// Create a new optimized graph builder
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            estimated_edges_per_node: None,
        }
    }

    /// Set expected number of edges per node for better memory allocation
    pub fn with_estimated_edges_per_node(mut self, edges_per_node: usize) -> Self {
        self.estimated_edges_per_node = Some(edges_per_node);
        self
    }

    /// Reserve capacity for nodes
    pub fn reserve_nodes(mut self, capacity: usize) -> Self {
        self.nodes.reserve(capacity);
        self
    }

    /// Reserve capacity for edges
    pub fn reserve_edges(mut self, capacity: usize) -> Self {
        self.edges.reserve(capacity);
        self
    }

    /// Add a node
    pub fn add_node(&mut self, node: usize) {
        self.nodes.push(node);
    }

    /// Add an edge
    pub fn add_edge(&mut self, from: usize, to: usize, weight: f64) {
        self.edges.push((from, to, weight));
    }

    /// Build the optimized graph
    pub fn build(self) -> Result<Graph<usize, f64>, String> {
        let mut graph = Graph::new();

        // Pre-allocate with estimated sizes
        if let Some(_epn) = self.estimated_edges_per_node {
            // Reserve capacity in adjacency lists
            for &node in &self.nodes {
                let _ = graph.add_node(node);
                // Internal method to reserve adjacency list capacity
                // This would need to be added to Graph API
            }
        } else {
            for &node in &self.nodes {
                let _ = graph.add_node(node);
            }
        }

        // Add edges
        for (from, to, weight) in self.edges {
            graph
                .add_edge(from, to, weight)
                .map_err(|e| format!("Failed to add edge: {:?}", e))?;
        }

        Ok(graph)
    }
}

/// Memory optimization suggestions
#[derive(Debug)]
pub struct OptimizationSuggestions {
    pub suggestions: Vec<String>,
    pub potential_savings: usize,
}

/// Analyze a graph and provide memory optimization suggestions
pub fn suggest_optimizations(
    stats: &MemoryStats,
    fragmentation: &FragmentationReport,
) -> OptimizationSuggestions {
    let mut suggestions = Vec::new();
    let mut potential_savings = 0;

    // Check efficiency
    if stats.efficiency < 0.7 {
        suggestions.push(format!(
            "Low memory efficiency ({:.1}%). Consider using a more compact representation.",
            stats.efficiency * 100.0
        ));
    }

    // Check fragmentation
    if fragmentation.fragmentation_ratio > 0.3 {
        suggestions.push(format!(
            "High fragmentation ({:.1}%). Pre-allocate adjacency lists based on expected degree.",
            fragmentation.fragmentation_ratio * 100.0
        ));
        potential_savings += fragmentation.wasted_bytes;
    }

    // Check degree distribution
    let max_degree = fragmentation
        .degree_distribution
        .keys()
        .max()
        .copied()
        .unwrap_or(0);
    let avg_degree = if fragmentation.total_used > 0 {
        fragmentation.total_used as f64 / fragmentation.degree_distribution.len() as f64
    } else {
        0.0
    };

    if max_degree > avg_degree as usize * 10 {
        suggestions.push(
            "Highly skewed degree distribution. Consider using a hybrid representation \
             with different storage for high-degree nodes."
                .to_string(),
        );
    }

    // Check for sparse graphs
    if avg_degree < 5.0 {
        suggestions.push(
            "Very sparse graph. Consider using a sparse matrix representation \
             or compressed adjacency lists."
                .to_string(),
        );
    }

    OptimizationSuggestions {
        suggestions,
        potential_savings,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generators;

    #[test]
    fn test_memory_profiling() {
        let graph = generators::complete_graph(100).unwrap();
        let stats = MemoryProfiler::profile_graph(&graph);

        assert!(stats.total_bytes > 0);
        assert!(stats.efficiency > 0.0 && stats.efficiency <= 1.0);
        assert_eq!(graph.node_count(), 100);
        assert_eq!(graph.edge_count(), 100 * 99 / 2); // Complete graph
    }

    #[test]
    fn test_memory_estimation() {
        let estimated = MemoryProfiler::estimate_memory(1000, 5000, false);
        assert!(estimated > 0);

        let estimated_directed = MemoryProfiler::estimate_memory(1000, 5000, true);
        assert!(estimated_directed > estimated); // Directed graphs use more memory
    }

    #[test]
    fn test_fragmentation_analysis() {
        let mut graph = Graph::new();

        // Create a graph with varied degrees
        for i in 0..100 {
            graph.add_node(i);
        }

        // Add edges to create different degree nodes
        for i in 0..10 {
            for j in 0..50 {
                if i != j {
                    graph.add_edge(i, j, 1.0).unwrap();
                }
            }
        }

        let report = MemoryProfiler::analyze_fragmentation(&graph);
        assert!(report.fragmentation_ratio >= 0.0 && report.fragmentation_ratio <= 1.0);
    }

    #[test]
    fn test_optimized_builder() {
        let mut builder = OptimizedGraphBuilder::new()
            .reserve_nodes(100)
            .reserve_edges(200)
            .with_estimated_edges_per_node(4);

        for i in 0..100 {
            builder.add_node(i);
        }

        for i in 0..99 {
            builder.add_edge(i, i + 1, 1.0);
        }

        let graph = builder.build().unwrap();
        assert_eq!(graph.node_count(), 100);
        assert_eq!(graph.edge_count(), 99);
    }
}
