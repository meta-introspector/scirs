//! Compact graph representations for memory efficiency
//!
//! This module provides memory-efficient graph storage formats optimized for
//! different graph characteristics (sparse, dense, regular degree, etc.).

use crate::error::GraphError;
use ndarray::{Array1, Array2};
use std::mem;

/// Compressed Sparse Row (CSR) format for sparse graphs
///
/// This format is highly memory-efficient for sparse graphs and provides
/// fast row (neighbor) access.
#[derive(Debug, Clone)]
pub struct CSRGraph {
    /// Number of nodes
    n_nodes: usize,
    /// Number of edges
    n_edges: usize,
    /// Row pointers - indices where each node's edges start
    row_ptr: Vec<usize>,
    /// Column indices - destination nodes
    col_idx: Vec<usize>,
    /// Edge weights
    weights: Vec<f64>,
}

impl CSRGraph {
    /// Create a new CSR graph from edge list
    pub fn from_edges(n_nodes: usize, edges: Vec<(usize, usize, f64)>) -> Result<Self, GraphError> {
        let n_edges = edges.len();

        // Count degree of each node
        let mut degree = vec![0; n_nodes];
        for &(src, _, _) in &edges {
            if src >= n_nodes {
                return Err(GraphError::InvalidNode(src));
            }
            degree[src] += 1;
        }

        // Build row pointers
        let mut row_ptr = vec![0; n_nodes + 1];
        for i in 0..n_nodes {
            row_ptr[i + 1] = row_ptr[i] + degree[i];
        }

        // Sort edges by source node
        let mut sorted_edges = edges;
        sorted_edges.sort_by_key(|&(src, _, _)| src);

        // Build column indices and weights
        let mut col_idx = Vec::with_capacity(n_edges);
        let mut weights = Vec::with_capacity(n_edges);

        for (_, dst, weight) in sorted_edges {
            col_idx.push(dst);
            weights.push(weight);
        }

        Ok(CSRGraph {
            n_nodes,
            n_edges,
            row_ptr,
            col_idx,
            weights,
        })
    }

    /// Get neighbors of a node
    pub fn neighbors(&self, node: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        let start = self.row_ptr[node];
        let end = self.row_ptr[node + 1];

        self.col_idx[start..end]
            .iter()
            .zip(&self.weights[start..end])
            .map(|(&idx, &weight)| (idx, weight))
    }

    /// Get degree of a node
    pub fn degree(&self, node: usize) -> usize {
        self.row_ptr[node + 1] - self.row_ptr[node]
    }

    /// Memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        mem::size_of_val(&self.n_nodes)
            + mem::size_of_val(&self.n_edges)
            + mem::size_of_val(&self.row_ptr[..])
            + mem::size_of_val(&self.col_idx[..])
            + mem::size_of_val(&self.weights[..])
    }

    /// Convert to adjacency matrix (for dense operations)
    pub fn to_adjacency_matrix(&self) -> Array2<f64> {
        let mut matrix = Array2::zeros((self.n_nodes, self.n_nodes));

        for src in 0..self.n_nodes {
            for (dst, weight) in self.neighbors(src) {
                matrix[[src, dst]] = weight;
            }
        }

        matrix
    }
}

/// Bit-packed representation for unweighted graphs
///
/// Uses 1 bit per potential edge, extremely memory efficient for unweighted graphs.
#[derive(Debug, Clone)]
pub struct BitPackedGraph {
    /// Number of nodes
    n_nodes: usize,
    /// Bit array storing adjacency information
    /// For undirected graphs, only upper triangle is stored
    bits: Vec<u64>,
    /// Whether the graph is directed
    directed: bool,
}

impl BitPackedGraph {
    /// Create a new bit-packed graph
    pub fn new(n_nodes: usize, directed: bool) -> Self {
        let bits_needed = if directed {
            n_nodes * n_nodes
        } else {
            n_nodes * (n_nodes + 1) / 2 // Upper triangle including diagonal
        };

        let words_needed = (bits_needed + 63) / 64;

        BitPackedGraph {
            n_nodes,
            bits: vec![0; words_needed],
            directed,
        }
    }

    /// Calculate bit position for an edge
    fn bit_position(&self, from: usize, to: usize) -> Option<usize> {
        if from >= self.n_nodes || to >= self.n_nodes {
            return None;
        }

        if self.directed {
            Some(from * self.n_nodes + to)
        } else {
            // For undirected, normalize to upper triangle
            let (u, v) = if from <= to { (from, to) } else { (to, from) };
            Some(u * self.n_nodes - u * (u - 1) / 2 + v - u)
        }
    }

    /// Add an edge
    pub fn add_edge(&mut self, from: usize, to: usize) -> Result<(), GraphError> {
        let bit_pos = self
            .bit_position(from, to)
            .ok_or(GraphError::InvalidNode(from.max(to)))?;

        let word_idx = bit_pos / 64;
        let bit_idx = bit_pos % 64;

        self.bits[word_idx] |= 1u64 << bit_idx;

        Ok(())
    }

    /// Check if edge exists
    pub fn has_edge(&self, from: usize, to: usize) -> bool {
        if let Some(bit_pos) = self.bit_position(from, to) {
            let word_idx = bit_pos / 64;
            let bit_idx = bit_pos % 64;

            (self.bits[word_idx] & (1u64 << bit_idx)) != 0
        } else {
            false
        }
    }

    /// Get neighbors of a node
    pub fn neighbors(&self, node: usize) -> Vec<usize> {
        let mut neighbors = Vec::new();

        for other in 0..self.n_nodes {
            if self.has_edge(node, other) {
                neighbors.push(other);
            }
        }

        neighbors
    }

    /// Memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        mem::size_of_val(&self.n_nodes)
            + mem::size_of_val(&self.bits[..])
            + mem::size_of_val(&self.directed)
    }
}

/// Compressed adjacency list using variable-length encoding
///
/// Uses delta encoding and variable-length integers for neighbor lists.
#[derive(Debug, Clone)]
pub struct CompressedAdjacencyList {
    /// Number of nodes
    n_nodes: usize,
    /// Compressed neighbor data
    data: Vec<u8>,
    /// Offsets into data for each node
    offsets: Vec<usize>,
}

impl CompressedAdjacencyList {
    /// Create from adjacency lists
    pub fn from_adjacency(adj_lists: Vec<Vec<usize>>) -> Self {
        let n_nodes = adj_lists.len();
        let mut data = Vec::new();
        let mut offsets = Vec::with_capacity(n_nodes + 1);

        offsets.push(0);

        for neighbors in adj_lists {
            let start_pos = data.len();

            // Sort neighbors for delta encoding
            let mut sorted_neighbors = neighbors;
            sorted_neighbors.sort_unstable();

            // Encode count
            Self::encode_varint(sorted_neighbors.len(), &mut data);

            // Delta encode neighbors
            let mut prev = 0;
            for &neighbor in &sorted_neighbors {
                let delta = neighbor - prev;
                Self::encode_varint(delta, &mut data);
                prev = neighbor;
            }

            offsets.push(data.len());
        }

        CompressedAdjacencyList {
            n_nodes,
            data,
            offsets,
        }
    }

    /// Variable-length integer encoding
    fn encode_varint(mut value: usize, output: &mut Vec<u8>) {
        while value >= 0x80 {
            output.push((value & 0x7F) as u8 | 0x80);
            value >>= 7;
        }
        output.push(value as u8);
    }

    /// Variable-length integer decoding
    fn decode_varint(data: &[u8], pos: &mut usize) -> usize {
        let mut value = 0;
        let mut shift = 0;

        loop {
            let byte = data[*pos];
            *pos += 1;

            value |= ((byte & 0x7F) as usize) << shift;

            if byte & 0x80 == 0 {
                break;
            }

            shift += 7;
        }

        value
    }

    /// Get neighbors of a node
    pub fn neighbors(&self, node: usize) -> Vec<usize> {
        if node >= self.n_nodes {
            return Vec::new();
        }

        let start = self.offsets[node];
        let end = self.offsets[node + 1];
        let data_slice = &self.data[start..end];

        let mut pos = 0;
        let count = Self::decode_varint(data_slice, &mut pos);

        let mut neighbors = Vec::with_capacity(count);
        let mut current = 0;

        for _ in 0..count {
            let delta = Self::decode_varint(data_slice, &mut pos);
            current += delta;
            neighbors.push(current);
        }

        neighbors
    }

    /// Memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        mem::size_of_val(&self.n_nodes)
            + mem::size_of_val(&self.data[..])
            + mem::size_of_val(&self.offsets[..])
    }
}

/// Hybrid graph representation that chooses optimal format based on graph properties
pub enum HybridGraph {
    /// Use CSR for sparse graphs
    CSR(CSRGraph),
    /// Use bit-packed for dense unweighted graphs
    BitPacked(BitPackedGraph),
    /// Use compressed adjacency for medium density
    Compressed(CompressedAdjacencyList),
}

impl HybridGraph {
    /// Automatically choose the best representation based on graph properties
    pub fn auto_select(
        n_nodes: usize,
        edges: Vec<(usize, usize, Option<f64>)>,
        directed: bool,
    ) -> Result<Self, GraphError> {
        let n_edges = edges.len();
        let density = n_edges as f64 / (n_nodes * n_nodes) as f64;
        let all_unweighted = edges.iter().all(|(_, _, w)| w.is_none());

        if all_unweighted && density > 0.1 {
            // Dense unweighted - use bit-packed
            let mut graph = BitPackedGraph::new(n_nodes, directed);
            for (src, dst, _) in edges {
                graph.add_edge(src, dst)?;
            }
            Ok(HybridGraph::BitPacked(graph))
        } else if density < 0.01 {
            // Very sparse - use CSR
            let weighted_edges: Vec<(usize, usize, f64)> = edges
                .into_iter()
                .map(|(s, d, w)| (s, d, w.unwrap_or(1.0)))
                .collect();
            let graph = CSRGraph::from_edges(n_nodes, weighted_edges)?;
            Ok(HybridGraph::CSR(graph))
        } else {
            // Medium density - use compressed adjacency
            let mut adj_lists = vec![Vec::new(); n_nodes];
            for (src, dst, _) in edges {
                adj_lists[src].push(dst);
                if !directed {
                    adj_lists[dst].push(src);
                }
            }
            let graph = CompressedAdjacencyList::from_adjacency(adj_lists);
            Ok(HybridGraph::Compressed(graph))
        }
    }

    /// Get memory usage
    pub fn memory_usage(&self) -> usize {
        match self {
            HybridGraph::CSR(g) => g.memory_usage(),
            HybridGraph::BitPacked(g) => g.memory_usage(),
            HybridGraph::Compressed(g) => g.memory_usage(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csr_graph() {
        let edges = vec![(0, 1, 1.0), (0, 2, 2.0), (1, 2, 3.0), (2, 3, 4.0)];

        let graph = CSRGraph::from_edges(4, edges).unwrap();

        assert_eq!(graph.degree(0), 2);
        assert_eq!(graph.degree(3), 0);

        let neighbors: Vec<_> = graph.neighbors(0).collect();
        assert_eq!(neighbors, vec![(1, 1.0), (2, 2.0)]);
    }

    #[test]
    fn test_bit_packed_graph() {
        let mut graph = BitPackedGraph::new(4, false);

        graph.add_edge(0, 1).unwrap();
        graph.add_edge(1, 2).unwrap();
        graph.add_edge(0, 3).unwrap();

        assert!(graph.has_edge(0, 1));
        assert!(graph.has_edge(1, 0)); // Undirected
        assert!(!graph.has_edge(2, 3));

        let neighbors = graph.neighbors(0);
        assert!(neighbors.contains(&1));
        assert!(neighbors.contains(&3));
    }

    #[test]
    fn test_compressed_adjacency() {
        let adj_lists = vec![
            vec![1, 2, 5],
            vec![0, 2],
            vec![0, 1, 3],
            vec![2],
            vec![],
            vec![0],
        ];

        let graph = CompressedAdjacencyList::from_adjacency(adj_lists.clone());

        for (node, expected) in adj_lists.iter().enumerate() {
            let neighbors = graph.neighbors(node);
            assert_eq!(&neighbors, expected);
        }

        // Check memory compression
        let uncompressed_size = adj_lists
            .iter()
            .map(|list| list.len() * mem::size_of::<usize>())
            .sum::<usize>();

        assert!(graph.memory_usage() < uncompressed_size);
    }
}
