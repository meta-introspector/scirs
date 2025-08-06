//! Graph clustering and community detection algorithms
//!
//! This module provides implementations of various graph clustering algorithms for
//! detecting communities and clusters in network data. These algorithms work with
//! graph representations where nodes represent data points and edges represent
//! similarities or connections between them.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_traits::{Float, FromPrimitive};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::Debug;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::error::{ClusteringError, Result};

/// Graph representation for clustering algorithms
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Graph<F: Float> {
    /// Number of nodes in the graph
    pub n_nodes: usize,
    /// Adjacency list representation: node_id -> [(neighbor_id, weight), ...]
    pub adjacency: Vec<Vec<(usize, F)>>,
    /// Optional node labels/features
    pub node_features: Option<Array2<F>>,
}

impl<F: Float + FromPrimitive + Debug> Graph<F> {
    /// Create a new empty graph with specified number of nodes
    pub fn new(_nnodes: usize) -> Self {
        Self {
            n_nodes,
            adjacency: vec![Vec::new(); _n_nodes],
            node_features: None,
        }
    }

    /// Create a graph from an adjacency matrix
    pub fn from_adjacency_matrix(_adjacencymatrix: ArrayView2<F>) -> Result<Self> {
        let n_nodes = adjacency_matrix.shape()[0];
        if adjacency_matrix.shape()[1] != n_nodes {
            return Err(ClusteringError::InvalidInput(
                "Adjacency _matrix must be square".to_string(),
            ));
        }

        let mut graph = Self::new(n_nodes);

        for i in 0..n_nodes {
            for j in 0..n_nodes {
                let weight = adjacency_matrix[[i, j]];
                if weight > F::zero() && i != j {
                    graph.add_edge(i, j, weight)?;
                }
            }
        }

        Ok(graph)
    }

    /// Create a k-nearest neighbor graph from data points
    pub fn from_knn_graph(data: ArrayView2<F>, k: usize) -> Result<Self> {
        let n_samples = data.shape()[0];
        let mut graph = Self::new(n_samples);
        graph.node_features = Some(_data.to_owned());

        // For each point, find k nearest neighbors
        for i in 0..n_samples {
            let mut distances: Vec<(usize, F)> = Vec::new();

            for j in 0..n_samples {
                if i != j {
                    let dist = euclidean_distance(_data.row(i), data.row(j));
                    distances.push((j, dist));
                }
            }

            // Sort by distance and take k nearest
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            for &(neighbor_idx, distance) in distances.iter().take(k) {
                // Use similarity (inverse of distance) as edge weight
                let similarity = F::one() / (F::one() + distance);
                graph.add_edge(i, neighbor_idx, similarity)?;
            }
        }

        Ok(graph)
    }

    /// Add an edge between two nodes
    pub fn add_edge(&mut self, node1: usize, node2: usize, weight: F) -> Result<()> {
        if node1 >= self.n_nodes || node2 >= self.n_nodes {
            return Err(ClusteringError::InvalidInput(
                "Node index out of bounds".to_string(),
            ));
        }

        if node1 != node2 {
            self.adjacency[node1].push((node2, weight));
            self.adjacency[node2].push((node1, weight)); // Undirected graph
        }

        Ok(())
    }

    /// Get the degree of a node (number of neighbors)
    pub fn degree(&self, node: usize) -> usize {
        if node < self.n_nodes {
            self.adjacency[node].len()
        } else {
            0
        }
    }

    /// Get the weighted degree of a node (sum of edge weights)
    pub fn weighted_degree(&self, node: usize) -> F {
        if node < self.n_nodes {
            self.adjacency[node].iter().map(|(_, weight)| *weight).sum()
        } else {
            F::zero()
        }
    }

    /// Get all neighbors of a node
    pub fn neighbors(&self, node: usize) -> &[(usize, F)] {
        if node < self.n_nodes {
            &self.adjacency[node]
        } else {
            &[]
        }
    }

    /// Calculate modularity of a given community assignment
    pub fn modularity(&self, communities: &[usize]) -> F {
        let total_weight = self.total_edge_weight();
        if total_weight == F::zero() {
            return F::zero();
        }

        let mut modularity = F::zero();

        for i in 0..self.n_nodes {
            for j in 0..self.n_nodes {
                if communities[i] == communities[j] {
                    let edge_weight = self.get_edge_weight(i, j);
                    let degree_i = self.weighted_degree(i);
                    let degree_j = self.weighted_degree(j);

                    let expected = degree_i * degree_j / (F::from(2.0).unwrap() * total_weight);
                    modularity = modularity + edge_weight - expected;
                }
            }
        }

        modularity / (F::from(2.0).unwrap() * total_weight)
    }

    /// Get edge weight between two nodes
    fn get_edge_weight(&self, node1: usize, node2: usize) -> F {
        if node1 < self.n_nodes {
            for &(neighbor, weight) in &self.adjacency[node1] {
                if neighbor == node2 {
                    return weight;
                }
            }
        }
        F::zero()
    }

    /// Calculate total weight of all edges in the graph
    fn total_edge_weight(&self) -> F {
        let mut total = F::zero();
        for node in 0..self.n_nodes {
            for &(_, weight) in &self.adjacency[node] {
                total = total + weight;
            }
        }
        total / F::from(2.0).unwrap() // Divide by 2 because each edge is counted twice
    }
}

/// Louvain community detection algorithm
///
/// The Louvain algorithm is a greedy optimization method that attempts to optimize
/// the modularity of a partition of the network. It produces high quality communities
/// and has excellent performance on large networks.
///
/// # Arguments
///
/// * `graph` - Input graph
/// * `resolution` - Resolution parameter (higher values lead to smaller communities)
/// * `max_iterations` - Maximum number of iterations
///
/// # Returns
///
/// Community assignments for each node
///
/// # Example
///
/// ```
/// use ndarray::Array2;
/// use scirs2__cluster::graph::{Graph, louvain};
///
/// // Create a simple graph
/// let adjacency = Array2::fromshape_vec((4, 4), vec![
///     0.0, 1.0, 1.0, 0.0,
///     1.0, 0.0, 0.0, 0.0,
///     1.0, 0.0, 0.0, 1.0,
///     0.0, 0.0, 1.0, 0.0,
/// ]).unwrap();
///
/// let graph = Graph::from_adjacency_matrix(adjacency.view()).unwrap();
/// let communities = louvain(&graph, 1.0, 100).unwrap();
/// ```
#[allow(dead_code)]
pub fn louvain<F>(
    _graph: &Graph<F>,
    resolution: f64,
    max_iterations: usize,
) -> Result<Array1<usize>>
where
    F: Float + FromPrimitive + Debug + 'static,
    f64: From<F>,
{
    let n_nodes = graph.n_nodes;
    let mut communities: Array1<usize> = Array1::from_iter(0..n_nodes);
    let mut improved = true;
    let mut iteration = 0;

    while improved && iteration < max_iterations {
        improved = false;
        iteration += 1;

        // Phase 1: Optimize modularity by moving nodes
        for node in 0..n_nodes {
            let current_community = communities[node];
            let mut best_community = current_community;
            let mut best_gain = F::zero();

            // Try moving node to each neighbor's community
            let mut candidate_communities = HashSet::new();
            candidate_communities.insert(current_community);

            for &(neighbor_) in graph.neighbors(node) {
                candidate_communities.insert(communities[neighbor]);
            }

            for &candidate_community in &candidate_communities {
                if candidate_community != current_community {
                    // Calculate modularity gain from moving to this community
                    let gain = modularity_gain(
                        graph,
                        &communities,
                        node,
                        current_community,
                        candidate_community,
                        resolution,
                    );

                    if gain > best_gain {
                        best_gain = gain;
                        best_community = candidate_community;
                    }
                }
            }

            // Move node to best community if improvement found
            if best_community != current_community && best_gain > F::zero() {
                communities[node] = best_community;
                improved = true;
            }
        }
    }

    Ok(communities)
}

/// Calculate modularity gain from moving a node to a different community
#[allow(dead_code)]
fn modularity_gain<F>(
    graph: &Graph<F>,
    communities: &Array1<usize>,
    node: usize,
    from_community: usize,
    to_community: usize,
    resolution: f64,
) -> F
where
    F: Float + FromPrimitive + Debug + 'static,
    f64: From<F>,
{
    let total_weight = graph.total_edge_weight();
    if total_weight == F::zero() {
        return F::zero();
    }

    let node_degree = graph.weighted_degree(node);
    let resolution_f = F::from(resolution).unwrap();

    // Calculate connections within target _community
    let mut edges_to_target = F::zero();
    let mut edges_from_source = F::zero();

    for &(neighbor, weight) in graph.neighbors(node) {
        if communities[neighbor] == to_community {
            edges_to_target = edges_to_target + weight;
        }
        if communities[neighbor] == from_community && neighbor != node {
            edges_from_source = edges_from_source + weight;
        }
    }

    // Calculate _community weights
    let target_community_weight = calculate_community_weight(graph, communities, to_community);
    let source_community_weight = calculate_community_weight(graph, communities, from_community);

    // Calculate modularity gain
    let gain_to = edges_to_target
        - resolution_f * node_degree * target_community_weight
            / (F::from(2.0).unwrap() * total_weight);
    let loss_from = edges_from_source
        - resolution_f * node_degree * (source_community_weight - node_degree)
            / (F::from(2.0).unwrap() * total_weight);

    gain_to - loss_from
}

/// Calculate total weight of a community
#[allow(dead_code)]
fn calculate_community_weight<F>(
    graph: &Graph<F>,
    communities: &Array1<usize>,
    community: usize,
) -> F
where
    F: Float + FromPrimitive + Debug + 'static,
{
    let mut weight = F::zero();
    for node in 0..graph.n_nodes {
        if communities[node] == community {
            weight = weight + graph.weighted_degree(node);
        }
    }
    weight
}

/// Label propagation algorithm for community detection
///
/// A fast algorithm where each node adopts the label that most of its neighbors have.
/// This process continues iteratively until convergence.
///
/// # Arguments
///
/// * `graph` - Input graph
/// * `max_iterations` - Maximum number of iterations
/// * `tolerance` - Convergence tolerance
///
/// # Returns
///
/// Community assignments for each node
#[allow(dead_code)]
pub fn label_propagation<F>(
    graph: &Graph<F>,
    max_iterations: usize,
    tolerance: f64,
) -> Result<Array1<usize>>
where
    F: Float + FromPrimitive + Debug + 'static,
    f64: From<F>,
{
    let n_nodes = graph.n_nodes;
    let mut labels: Array1<usize> = Array1::from_iter(0..n_nodes);
    let mut changed_nodes = n_nodes;
    let tolerance_f = F::from(tolerance).unwrap();

    for _iteration in 0..max_iterations {
        let mut new_labels = labels.clone();
        changed_nodes = 0;

        // Process nodes in random order
        let mut node_order: Vec<usize> = (0..n_nodes).collect();
        // For deterministic results, we'll use a simple shuffle based on node index
        node_order.sort_by_key(|&i| i * 17 % n_nodes);

        for &node in &node_order {
            // Count label frequencies among neighbors
            let mut label_weights: HashMap<usize, F> = HashMap::new();

            for &(neighbor, weight) in graph.neighbors(node) {
                let label = labels[neighbor];
                *labelweights.entry(label).or_insert(F::zero()) += weight;
            }

            // Choose label with highest weight
            if let Some((&best_label_)) = label_weights
                .iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            {
                if best_label != labels[node] {
                    new_labels[node] = best_label;
                    changed_nodes += 1;
                }
            }
        }

        labels = new_labels;

        // Check convergence
        let change_ratio = changed_nodes as f64 / n_nodes as f64;
        if change_ratio < tolerance {
            break;
        }
    }

    // Relabel communities to be consecutive integers starting from 0
    let unique_labels: HashSet<usize> = labels.iter().cloned().collect();
    let label_mapping: HashMap<usize, usize> = unique_labels
        .into_iter()
        .enumerate()
        .map(|(new_label, old_label)| (old_label, new_label))
        .collect();

    for label in labels.iter_mut() {
        *label = label_mapping[label];
    }

    Ok(labels)
}

/// Girvan-Newman algorithm for community detection
///
/// This algorithm removes edges with highest betweenness centrality iteratively
/// to reveal community structure. It's more computationally expensive but can
/// produce hierarchical community structures.
///
/// # Arguments
///
/// * `graph` - Input graph
/// * `n_communities` - Desired number of communities (algorithm stops when reached)
///
/// # Returns
///
/// Community assignments for each node
#[allow(dead_code)]
pub fn girvan_newman<F>(_graph: &Graph<F>, ncommunities: usize) -> Result<Array1<usize>>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    if n_communities > graph.n_nodes {
        return Err(ClusteringError::InvalidInput(
            "Number of _communities cannot exceed number of nodes".to_string(),
        ));
    }

    let mut working_graph = graph.clone();
    let mut _communities = find_connected_components(&working_graph);

    while count_communities(&_communities) < n_communities && has_edges(&working_graph) {
        // Calculate edge betweenness centrality
        let edge_betweenness = calculate_edge_betweenness(&working_graph)?;

        // Find edge with highest betweenness
        if let Some((max_edge_)) = edge_betweenness
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        {
            // Remove the edge with highest betweenness
            remove_edge(&mut working_graph, max_edge.0, max_edge.1);

            // Recalculate connected components
            _communities = find_connected_components(&working_graph);
        } else {
            break; // No more edges to remove
        }
    }

    Ok(Array1::from_vec(_communities))
}

/// Calculate edge betweenness centrality for all edges
#[allow(dead_code)]
fn calculate_edge_betweenness<F>(graph: &Graph<F>) -> Result<HashMap<(usize, usize), f64>>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    let mut edge_betweenness = HashMap::new();

    // Initialize all edges with zero betweenness
    for node in 0.._graph.n_nodes {
        for &(neighbor_) in graph.neighbors(node) {
            if node < neighbor {
                // Count each edge only once
                edge_betweenness.insert((node, neighbor), 0.0);
            }
        }
    }

    // For each pair of nodes, calculate shortest paths and update edge betweenness
    for source in 0.._graph.n_nodes {
        for target in (source + 1).._graph.n_nodes {
            let paths = find_all_shortest_paths(_graph, source, target);

            if !paths.is_empty() {
                let contribution = 1.0 / paths.len() as f64;

                for path in paths {
                    for i in 0..(path.len() - 1) {
                        let (u, v) = if path[i] < path[i + 1] {
                            (path[i], path[i + 1])
                        } else {
                            (path[i + 1], path[i])
                        };

                        *edge_betweenness.entry((u, v)).or_insert(0.0) += contribution;
                    }
                }
            }
        }
    }

    Ok(edge_betweenness)
}

/// Find all shortest paths between two nodes using BFS
#[allow(dead_code)]
fn find_all_shortest_paths<F>(graph: &Graph<F>, source: usize, target: usize) -> Vec<Vec<usize>>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    let mut distances = vec![None; graph.n_nodes];
    let mut predecessors: Vec<Vec<usize>> = vec![Vec::new(); graph.n_nodes];
    let mut queue = VecDeque::new();

    distances[source] = Some(0);
    queue.push_back(source);

    while let Some(current) = queue.pop_front() {
        let current_dist = distances[current].unwrap();

        for &(neighbor_) in graph.neighbors(current) {
            if distances[neighbor].is_none() {
                // First time visiting this node
                distances[neighbor] = Some(current_dist + 1);
                predecessors[neighbor].push(current);
                queue.push_back(neighbor);
            } else if distances[neighbor] == Some(current_dist + 1) {
                // Another shortest path found
                predecessors[neighbor].push(current);
            }
        }
    }

    // Reconstruct all shortest paths
    if distances[target].is_none() {
        return Vec::new(); // No path exists
    }

    let mut paths = Vec::new();
    let mut current_paths = vec![vec![target]];

    while !current_paths.is_empty() {
        let mut next_paths = Vec::new();

        for path in current_paths {
            let last_node = path[path.len() - 1];

            if last_node == source {
                let mut complete_path = path.clone();
                complete_path.reverse();
                paths.push(complete_path);
            } else {
                for &pred in &predecessors[last_node] {
                    let mut new_path = path.clone();
                    new_path.push(pred);
                    next_paths.push(new_path);
                }
            }
        }

        current_paths = next_paths;
    }

    paths
}

/// Remove an edge from the graph
#[allow(dead_code)]
fn remove_edge<F>(graph: &mut Graph<F>, node1: usize, node2: usize)
where
    F: Float + FromPrimitive + Debug + 'static,
{
    graph.adjacency[node1].retain(|(neighbor_)| *neighbor != node2);
    graph.adjacency[node2].retain(|(neighbor_)| *neighbor != node1);
}

/// Check if the graph has any edges
#[allow(dead_code)]
fn has_edges<F>(graph: &Graph<F>) -> bool
where
    F: Float + FromPrimitive + Debug + 'static,
{
    _graph
        .adjacency
        .iter()
        .any(|neighbors| !neighbors.is_empty())
}

/// Find connected components in the graph
#[allow(dead_code)]
fn find_connected_components<F>(graph: &Graph<F>) -> Vec<usize>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    let mut visited = vec![false; graph.n_nodes];
    let mut components = vec![0; graph.n_nodes];
    let mut component_id = 0;

    for node in 0.._graph.n_nodes {
        if !visited[node] {
            dfs_component(_graph, node, component_id, &mut visited, &mut components);
            component_id += 1;
        }
    }

    components
}

/// Depth-first search to mark connected component
#[allow(dead_code)]
fn dfs_component<F>(
    graph: &Graph<F>,
    node: usize,
    component_id: usize,
    visited: &mut [bool],
    components: &mut [usize],
) where
    F: Float + FromPrimitive + Debug + 'static,
{
    visited[node] = true;
    components[node] = component_id;

    for &(neighbor_) in graph.neighbors(node) {
        if !visited[neighbor] {
            dfs_component(graph, neighbor, component_id, visited, components);
        }
    }
}

/// Count the number of unique communities
#[allow(dead_code)]
fn count_communities(communities: &[usize]) -> usize {
    let mut unique: HashSet<usize> = HashSet::new();
    for &community in _communities {
        unique.insert(community);
    }
    unique.len()
}

/// Helper function to calculate Euclidean distance between two points
#[allow(dead_code)]
fn euclidean_distance<F>(a: ArrayView1<F>, b: ArrayView1<F>) -> F
where
    F: Float + std::iter::Sum,
{
    let diff = &a.to_owned() - &b.to_owned();
    diff.dot(&diff).sqrt()
}

/// Configuration for graph clustering algorithms
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GraphClusteringConfig {
    /// Algorithm to use for clustering
    pub algorithm: GraphClusteringAlgorithm,
    /// Maximum number of iterations (for iterative algorithms)
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Resolution parameter (for modularity-based algorithms)
    pub resolution: f64,
    /// Target number of communities (for hierarchical algorithms)
    pub n_communities: Option<usize>,
}

/// Available graph clustering algorithms
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum GraphClusteringAlgorithm {
    /// Louvain community detection
    Louvain,
    /// Label propagation algorithm
    LabelPropagation,
    /// Girvan-Newman algorithm
    GirvanNewman,
}

impl Default for GraphClusteringConfig {
    fn default() -> Self {
        Self {
            algorithm: GraphClusteringAlgorithm::Louvain,
            max_iterations: 100,
            tolerance: 1e-6,
            resolution: 1.0,
            n_communities: None,
        }
    }
}

/// Perform graph clustering using the specified configuration
///
/// # Arguments
///
/// * `graph` - Input graph
/// * `config` - Clustering configuration
///
/// # Returns
///
/// Community assignments for each node
#[allow(dead_code)]
pub fn graph_clustering<F>(
    graph: &Graph<F>,
    config: &GraphClusteringConfig,
) -> Result<Array1<usize>>
where
    F: Float + FromPrimitive + Debug + 'static,
    f64: From<F>,
{
    match config.algorithm {
        GraphClusteringAlgorithm::Louvain => {
            louvain(graph, config.resolution, config.max_iterations)
        }
        GraphClusteringAlgorithm::LabelPropagation => {
            label_propagation(graph, config.max_iterations, config.tolerance)
        }
        GraphClusteringAlgorithm::GirvanNewman => {
            let n_communities = config.n_communities.unwrap_or(2);
            girvan_newman(graph, n_communities)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_graph_creation() {
        let graph = Graph::<f64>::new(5);
        assert_eq!(graph.n_nodes, 5);
        assert_eq!(graph.adjacency.len(), 5);
    }

    #[test]
    fn test_graph_from_adjacency_matrix() {
        let adjacency =
            Array2::fromshape_vec((3, 3), vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
                .unwrap();

        let graph = Graph::from_adjacency_matrix(adjacency.view()).unwrap();
        assert_eq!(graph.n_nodes, 3);
        assert_eq!(graph.degree(0), 1);
        assert_eq!(graph.degree(1), 2);
        assert_eq!(graph.degree(2), 1);
    }

    #[test]
    fn test_louvain_clustering() {
        // Create a simple graph with two obvious communities
        let adjacency = Array2::fromshape_vec(
            (4, 4),
            vec![
                0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
            ],
        )
        .unwrap();

        let graph = Graph::from_adjacency_matrix(adjacency.view()).unwrap();
        let communities = louvain(&graph, 1.0, 100).unwrap();

        // Nodes 0,1 should be in one community and nodes 2,3 in another
        assert_eq!(communities.len(), 4);
        assert_eq!(communities[0], communities[1]);
        assert_eq!(communities[2], communities[3]);
        assert_ne!(communities[0], communities[2]);
    }

    #[test]
    fn test_label_propagation() {
        let adjacency = Array2::fromshape_vec(
            (4, 4),
            vec![
                0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
            ],
        )
        .unwrap();

        let graph = Graph::from_adjacency_matrix(adjacency.view()).unwrap();
        let communities = label_propagation(&graph, 100, 1e-6).unwrap();

        assert_eq!(communities.len(), 4);
        // Should detect two communities
        let unique_communities: HashSet<usize> = communities.iter().cloned().collect();
        assert_eq!(unique_communities.len(), 2);
    }

    #[test]
    fn test_knn_graph_creation() {
        let data =
            Array2::fromshape_vec((4, 2), vec![0.0, 0.0, 0.1, 0.1, 5.0, 5.0, 5.1, 5.1]).unwrap();

        let graph = Graph::from_knn_graph(data.view(), 2).unwrap();
        assert_eq!(graph.n_nodes, 4);

        // Each node should have exactly 2 neighbors
        for node in 0..4 {
            assert_eq!(graph.degree(node), 2);
        }
    }
}
