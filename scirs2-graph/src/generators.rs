//! Graph generation algorithms
//!
//! This module provides functions for generating various types of graphs:
//! - Random graphs (Erdős–Rényi, Barabási–Albert, etc.)
//! - Regular graphs (complete, star, path, cycle)
//! - Lattice graphs
//! - Small-world networks

use rand::prelude::*;
use std::collections::HashSet;

use crate::base::{DiGraph, Graph};
use crate::error::{GraphError, Result};

/// Create a new empty undirected graph
pub fn create_graph<N: crate::base::Node, E: crate::base::EdgeWeight>() -> Graph<N, E> {
    Graph::new()
}

/// Create a new empty directed graph
pub fn create_digraph<N: crate::base::Node, E: crate::base::EdgeWeight>() -> DiGraph<N, E> {
    DiGraph::new()
}

/// Generates an Erdős–Rényi random graph
///
/// # Arguments
/// * `n` - Number of nodes
/// * `p` - Probability of edge creation between any two nodes
/// * `rng` - Random number generator
///
/// # Returns
/// * `Result<Graph<usize, f64>>` - The generated graph with node IDs 0..n-1
pub fn erdos_renyi_graph<R: Rng>(n: usize, p: f64, rng: &mut R) -> Result<Graph<usize, f64>> {
    if !(0.0..=1.0).contains(&p) {
        return Err(GraphError::InvalidGraph(
            "Probability must be between 0 and 1".to_string(),
        ));
    }

    let mut graph = Graph::new();

    // Add all nodes
    for i in 0..n {
        graph.add_node(i);
    }

    // Add edges with probability p
    for i in 0..n {
        for j in i + 1..n {
            if rng.random::<f64>() < p {
                graph.add_edge(i, j, 1.0)?;
            }
        }
    }

    Ok(graph)
}

/// Generates a Barabási–Albert preferential attachment graph
///
/// # Arguments
/// * `n` - Total number of nodes
/// * `m` - Number of edges to attach from a new node to existing nodes
/// * `rng` - Random number generator
///
/// # Returns
/// * `Result<Graph<usize, f64>>` - The generated graph with node IDs 0..n-1
pub fn barabasi_albert_graph<R: Rng>(n: usize, m: usize, rng: &mut R) -> Result<Graph<usize, f64>> {
    if m >= n {
        return Err(GraphError::InvalidGraph(
            "m must be less than n".to_string(),
        ));
    }
    if m == 0 {
        return Err(GraphError::InvalidGraph("m must be positive".to_string()));
    }

    let mut graph = Graph::new();

    // Start with a complete graph of m+1 nodes
    for i in 0..=m {
        graph.add_node(i);
    }

    for i in 0..=m {
        for j in i + 1..=m {
            graph.add_edge(i, j, 1.0)?;
        }
    }

    // Keep track of node degrees for preferential attachment
    let mut degrees = vec![m; m + 1];
    let mut total_degree = m * (m + 1);

    // Add remaining nodes
    for new_node in (m + 1)..n {
        graph.add_node(new_node);

        let mut targets = HashSet::new();

        // Select m nodes to connect to based on preferential attachment
        while targets.len() < m {
            let mut cumulative_prob = 0.0;
            let random_value = rng.random::<f64>() * total_degree as f64;

            for (node_id, &degree) in degrees.iter().enumerate() {
                cumulative_prob += degree as f64;
                if random_value <= cumulative_prob && !targets.contains(&node_id) {
                    targets.insert(node_id);
                    break;
                }
            }
        }

        // Add edges to selected targets
        for &target in &targets {
            graph.add_edge(new_node, target, 1.0)?;
            degrees[target] += 1;
            total_degree += 2; // Each edge adds 2 to total degree
        }

        degrees.push(m); // New node has degree m
    }

    Ok(graph)
}

/// Generates a complete graph (clique)
///
/// # Arguments
/// * `n` - Number of nodes
///
/// # Returns
/// * `Result<Graph<usize, f64>>` - A complete graph with n nodes
pub fn complete_graph(n: usize) -> Result<Graph<usize, f64>> {
    let mut graph = Graph::new();

    // Add all nodes
    for i in 0..n {
        graph.add_node(i);
    }

    // Add all possible edges
    for i in 0..n {
        for j in i + 1..n {
            graph.add_edge(i, j, 1.0)?;
        }
    }

    Ok(graph)
}

/// Generates a star graph with one central node connected to all others
///
/// # Arguments
/// * `n` - Total number of nodes (must be >= 1)
///
/// # Returns
/// * `Result<Graph<usize, f64>>` - A star graph with node 0 as the center
pub fn star_graph(n: usize) -> Result<Graph<usize, f64>> {
    if n == 0 {
        return Err(GraphError::InvalidGraph(
            "Star graph must have at least 1 node".to_string(),
        ));
    }

    let mut graph = Graph::new();

    // Add all nodes
    for i in 0..n {
        graph.add_node(i);
    }

    // Connect center (node 0) to all other nodes
    for i in 1..n {
        graph.add_edge(0, i, 1.0)?;
    }

    Ok(graph)
}

/// Generates a path graph (nodes connected in a line)
///
/// # Arguments
/// * `n` - Number of nodes
///
/// # Returns
/// * `Result<Graph<usize, f64>>` - A path graph with nodes 0, 1, ..., n-1
pub fn path_graph(n: usize) -> Result<Graph<usize, f64>> {
    let mut graph = Graph::new();

    // Add all nodes
    for i in 0..n {
        graph.add_node(i);
    }

    // Connect consecutive nodes
    for i in 0..n.saturating_sub(1) {
        graph.add_edge(i, i + 1, 1.0)?;
    }

    Ok(graph)
}

/// Generates a cycle graph (circular arrangement of nodes)
///
/// # Arguments
/// * `n` - Number of nodes (must be >= 3 for a meaningful cycle)
///
/// # Returns
/// * `Result<Graph<usize, f64>>` - A cycle graph with nodes 0, 1, ..., n-1
pub fn cycle_graph(n: usize) -> Result<Graph<usize, f64>> {
    if n < 3 {
        return Err(GraphError::InvalidGraph(
            "Cycle graph must have at least 3 nodes".to_string(),
        ));
    }

    let mut graph = Graph::new();

    // Add all nodes
    for i in 0..n {
        graph.add_node(i);
    }

    // Connect consecutive nodes
    for i in 0..n {
        graph.add_edge(i, (i + 1) % n, 1.0)?;
    }

    Ok(graph)
}

/// Generates a 2D grid/lattice graph
///
/// # Arguments
/// * `rows` - Number of rows
/// * `cols` - Number of columns
///
/// # Returns
/// * `Result<Graph<usize, f64>>` - A grid graph where node ID = row * cols + col
pub fn grid_2d_graph(rows: usize, cols: usize) -> Result<Graph<usize, f64>> {
    if rows == 0 || cols == 0 {
        return Err(GraphError::InvalidGraph(
            "Grid dimensions must be positive".to_string(),
        ));
    }

    let mut graph = Graph::new();

    // Add all nodes
    for i in 0..(rows * cols) {
        graph.add_node(i);
    }

    // Add edges to adjacent nodes (4-connectivity)
    for row in 0..rows {
        for col in 0..cols {
            let node_id = row * cols + col;

            // Connect to right neighbor
            if col + 1 < cols {
                let right_neighbor = row * cols + (col + 1);
                graph.add_edge(node_id, right_neighbor, 1.0)?;
            }

            // Connect to bottom neighbor
            if row + 1 < rows {
                let bottom_neighbor = (row + 1) * cols + col;
                graph.add_edge(node_id, bottom_neighbor, 1.0)?;
            }
        }
    }

    Ok(graph)
}

/// Generates a Watts-Strogatz small-world graph
///
/// # Arguments
/// * `n` - Number of nodes
/// * `k` - Each node is connected to k nearest neighbors in ring topology (must be even)
/// * `p` - Probability of rewiring each edge
/// * `rng` - Random number generator
///
/// # Returns
/// * `Result<Graph<usize, f64>>` - A small-world graph
pub fn watts_strogatz_graph<R: Rng>(
    n: usize,
    k: usize,
    p: f64,
    rng: &mut R,
) -> Result<Graph<usize, f64>> {
    if k >= n || k % 2 != 0 {
        return Err(GraphError::InvalidGraph(
            "k must be even and less than n".to_string(),
        ));
    }
    if !(0.0..=1.0).contains(&p) {
        return Err(GraphError::InvalidGraph(
            "Probability must be between 0 and 1".to_string(),
        ));
    }

    let mut graph = Graph::new();

    // Add all nodes
    for i in 0..n {
        graph.add_node(i);
    }

    // Create regular ring lattice
    for i in 0..n {
        for j in 1..=(k / 2) {
            let neighbor = (i + j) % n;
            graph.add_edge(i, neighbor, 1.0)?;
        }
    }

    // Rewire edges with probability p
    let edges_to_process: Vec<_> = graph.edges().into_iter().collect();

    for edge in edges_to_process {
        if rng.random::<f64>() < p {
            // Remove the original edge (we'll recreate the graph to do this)
            let mut new_graph = Graph::new();

            // Add all nodes
            for i in 0..n {
                new_graph.add_node(i);
            }

            // Add all edges except the one we're rewiring
            for existing_edge in graph.edges() {
                if (existing_edge.source != edge.source || existing_edge.target != edge.target)
                    && (existing_edge.source != edge.target || existing_edge.target != edge.source)
                {
                    new_graph.add_edge(
                        existing_edge.source,
                        existing_edge.target,
                        existing_edge.weight,
                    )?;
                }
            }

            // Add rewired edge to a random node
            let mut new_target = rng.random_range(0..n);
            while new_target == edge.source || new_graph.has_node(&new_target) {
                new_target = rng.random_range(0..n);
            }

            new_graph.add_edge(edge.source, new_target, 1.0)?;
            graph = new_graph;
        }
    }

    Ok(graph)
}

/// Generates a graph using the Stochastic Block Model (SBM)
///
/// The SBM generates a graph where nodes are divided into communities (blocks)
/// and edge probabilities depend on which communities the nodes belong to.
///
/// # Arguments
/// * `block_sizes` - Vector specifying the size of each block/community
/// * `block_matrix` - Probability matrix where entry (i,j) is the probability
///   of an edge between nodes in block i and block j
/// * `rng` - Random number generator
///
/// # Returns
/// * `Result<Graph<usize, f64>>` - The generated graph with node IDs 0..n-1
///   where nodes 0..block_sizes[0]-1 are in block 0, etc.
pub fn stochastic_block_model<R: Rng>(
    block_sizes: &[usize],
    block_matrix: &[Vec<f64>],
    rng: &mut R,
) -> Result<Graph<usize, f64>> {
    if block_sizes.is_empty() {
        return Err(GraphError::InvalidGraph(
            "At least one block must be specified".to_string(),
        ));
    }

    if block_matrix.len() != block_sizes.len() {
        return Err(GraphError::InvalidGraph(
            "Block matrix dimensions must match number of blocks".to_string(),
        ));
    }

    for row in block_matrix {
        if row.len() != block_sizes.len() {
            return Err(GraphError::InvalidGraph(
                "Block matrix must be square".to_string(),
            ));
        }
        for &prob in row {
            if !(0.0..=1.0).contains(&prob) {
                return Err(GraphError::InvalidGraph(
                    "All probabilities must be between 0 and 1".to_string(),
                ));
            }
        }
    }

    let total_nodes: usize = block_sizes.iter().sum();
    let mut graph = Graph::new();

    // Add all nodes
    for i in 0..total_nodes {
        graph.add_node(i);
    }

    // Create mapping from node to block
    let mut node_to_block = vec![0; total_nodes];
    let mut current_node = 0;
    for (block_id, &block_size) in block_sizes.iter().enumerate() {
        for _ in 0..block_size {
            node_to_block[current_node] = block_id;
            current_node += 1;
        }
    }

    // Generate edges based on block probabilities
    for i in 0..total_nodes {
        for j in (i + 1)..total_nodes {
            let block_i = node_to_block[i];
            let block_j = node_to_block[j];
            let prob = block_matrix[block_i][block_j];

            if rng.random::<f64>() < prob {
                graph.add_edge(i, j, 1.0)?;
            }
        }
    }

    Ok(graph)
}

/// Generates a simple stochastic block model with two communities
///
/// This is a convenience function for creating a two-community SBM with
/// high intra-community probability and low inter-community probability.
///
/// # Arguments
/// * `n1` - Size of first community
/// * `n2` - Size of second community
/// * `p_in` - Probability of edges within communities
/// * `p_out` - Probability of edges between communities
/// * `rng` - Random number generator
///
/// # Returns
/// * `Result<Graph<usize, f64>>` - The generated graph
pub fn two_community_sbm<R: Rng>(
    n1: usize,
    n2: usize,
    p_in: f64,
    p_out: f64,
    rng: &mut R,
) -> Result<Graph<usize, f64>> {
    let block_sizes = vec![n1, n2];
    let block_matrix = vec![vec![p_in, p_out], vec![p_out, p_in]];

    stochastic_block_model(&block_sizes, &block_matrix, rng)
}

/// Generates a planted partition model (special case of SBM)
///
/// In this model, there are k communities of equal size, with high
/// intra-community probability and low inter-community probability.
///
/// # Arguments
/// * `n` - Total number of nodes (must be divisible by k)
/// * `k` - Number of communities
/// * `p_in` - Probability of edges within communities
/// * `p_out` - Probability of edges between communities
/// * `rng` - Random number generator
///
/// # Returns
/// * `Result<Graph<usize, f64>>` - The generated graph
pub fn planted_partition_model<R: Rng>(
    n: usize,
    k: usize,
    p_in: f64,
    p_out: f64,
    rng: &mut R,
) -> Result<Graph<usize, f64>> {
    if n % k != 0 {
        return Err(GraphError::InvalidGraph(
            "Number of nodes must be divisible by number of communities".to_string(),
        ));
    }

    let community_size = n / k;
    let block_sizes = vec![community_size; k];

    // Create block matrix
    let mut block_matrix = vec![vec![p_out; k]; k];
    for i in 0..k {
        block_matrix[i][i] = p_in;
    }

    stochastic_block_model(&block_sizes, &block_matrix, rng)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_erdos_renyi_graph() {
        let mut rng = StdRng::seed_from_u64(42);
        let graph = erdos_renyi_graph(10, 0.3, &mut rng).unwrap();

        assert_eq!(graph.node_count(), 10);
        // With p=0.3 and 45 possible edges, we expect around 13-14 edges
        // but this is random, so we just check it's reasonable
        assert!(graph.edge_count() <= 45);
    }

    #[test]
    fn test_complete_graph() {
        let graph = complete_graph(5).unwrap();

        assert_eq!(graph.node_count(), 5);
        assert_eq!(graph.edge_count(), 10); // n*(n-1)/2 = 5*4/2 = 10
    }

    #[test]
    fn test_star_graph() {
        let graph = star_graph(6).unwrap();

        assert_eq!(graph.node_count(), 6);
        assert_eq!(graph.edge_count(), 5); // n-1 edges
    }

    #[test]
    fn test_path_graph() {
        let graph = path_graph(5).unwrap();

        assert_eq!(graph.node_count(), 5);
        assert_eq!(graph.edge_count(), 4); // n-1 edges
    }

    #[test]
    fn test_cycle_graph() {
        let graph = cycle_graph(5).unwrap();

        assert_eq!(graph.node_count(), 5);
        assert_eq!(graph.edge_count(), 5); // n edges

        // Test error case
        assert!(cycle_graph(2).is_err());
    }

    #[test]
    fn test_grid_2d_graph() {
        let graph = grid_2d_graph(3, 4).unwrap();

        assert_eq!(graph.node_count(), 12); // 3*4 = 12 nodes
        assert_eq!(graph.edge_count(), 17); // (3-1)*4 + 3*(4-1) = 8 + 9 = 17 edges
    }

    #[test]
    fn test_barabasi_albert_graph() {
        let mut rng = StdRng::seed_from_u64(42);
        let graph = barabasi_albert_graph(10, 2, &mut rng).unwrap();

        assert_eq!(graph.node_count(), 10);
        // Should have 3 + 2*7 = 17 edges (3 initial edges + 2 for each of the 7 new nodes)
        assert_eq!(graph.edge_count(), 17);
    }

    #[test]
    fn test_stochastic_block_model() {
        let mut rng = StdRng::seed_from_u64(42);

        // Two blocks of size 3 and 4
        let block_sizes = vec![3, 4];
        // High intra-block probability, low inter-block probability
        let block_matrix = vec![vec![0.8, 0.1], vec![0.1, 0.8]];

        let graph = stochastic_block_model(&block_sizes, &block_matrix, &mut rng).unwrap();

        assert_eq!(graph.node_count(), 7); // 3 + 4 = 7 nodes

        // Check that all nodes are present
        for i in 0..7 {
            assert!(graph.has_node(&i));
        }
    }

    #[test]
    fn test_two_community_sbm() {
        let mut rng = StdRng::seed_from_u64(42);

        let graph = two_community_sbm(5, 5, 0.8, 0.1, &mut rng).unwrap();

        assert_eq!(graph.node_count(), 10);

        // Should have some edges within communities and fewer between
        // This is probabilistic so we can't test exact numbers
        assert!(graph.edge_count() > 0);
    }

    #[test]
    fn test_planted_partition_model() {
        let mut rng = StdRng::seed_from_u64(42);

        let graph = planted_partition_model(12, 3, 0.7, 0.1, &mut rng).unwrap();

        assert_eq!(graph.node_count(), 12); // 12 nodes total

        // 3 communities of size 4 each
        // Should have some edges
        assert!(graph.edge_count() > 0);
    }

    #[test]
    fn test_stochastic_block_model_errors() {
        let mut rng = StdRng::seed_from_u64(42);

        // Empty blocks
        assert!(stochastic_block_model(&[], &[], &mut rng).is_err());

        // Mismatched dimensions
        let block_sizes = vec![3, 4];
        let wrong_matrix = vec![vec![0.5]];
        assert!(stochastic_block_model(&block_sizes, &wrong_matrix, &mut rng).is_err());

        // Invalid probabilities
        let bad_matrix = vec![vec![1.5, 0.5], vec![0.5, 0.5]];
        assert!(stochastic_block_model(&block_sizes, &bad_matrix, &mut rng).is_err());

        // Non-divisible nodes for planted partition
        assert!(planted_partition_model(10, 3, 0.5, 0.1, &mut rng).is_err());
    }
}
