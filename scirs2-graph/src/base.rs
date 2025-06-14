//! Base graph structures and operations
//!
//! This module provides the core graph data structures and interfaces
//! for representing and working with graphs.

use ndarray::{Array1, Array2};
pub use petgraph::graph::IndexType;
use petgraph::graph::{Graph as PetGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::{Directed, Undirected};
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

use crate::error::{GraphError, Result};

/// A trait representing a node in a graph
pub trait Node: Clone + Eq + Hash + Send + Sync {}

/// Implements Node for common types
impl<T: Clone + Eq + Hash + Send + Sync> Node for T {}

/// A trait for edge weights in a graph
pub trait EdgeWeight: Clone + PartialOrd + Send + Sync {}

/// Implements EdgeWeight for common types
impl<T: Clone + PartialOrd + Send + Sync> EdgeWeight for T {}

/// An undirected graph structure
pub struct Graph<N: Node, E: EdgeWeight, Ix: IndexType = u32> {
    graph: PetGraph<N, E, Undirected, Ix>,
    node_indices: HashMap<N, NodeIndex<Ix>>,
}

/// A directed graph structure
pub struct DiGraph<N: Node, E: EdgeWeight, Ix: IndexType = u32> {
    graph: PetGraph<N, E, Directed, Ix>,
    node_indices: HashMap<N, NodeIndex<Ix>>,
}

/// Represents an edge in a graph
#[derive(Debug, Clone)]
pub struct Edge<N: Node, E: EdgeWeight> {
    /// Source node
    pub source: N,
    /// Target node
    pub target: N,
    /// Edge weight
    pub weight: E,
}

impl<N: Node, E: EdgeWeight, Ix: IndexType> Default for Graph<N, E, Ix> {
    fn default() -> Self {
        Self::new()
    }
}

impl<N: Node, E: EdgeWeight, Ix: IndexType> Graph<N, E, Ix> {
    /// Create a new empty undirected graph
    pub fn new() -> Self {
        Graph {
            graph: PetGraph::default(),
            node_indices: HashMap::new(),
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, node: N) -> NodeIndex<Ix> {
        if let Some(idx) = self.node_indices.get(&node) {
            return *idx;
        }

        let idx = self.graph.add_node(node.clone());
        self.node_indices.insert(node, idx);
        idx
    }

    /// Add an edge between two nodes with a given weight
    pub fn add_edge(&mut self, source: N, target: N, weight: E) -> Result<()> {
        let source_idx = self.add_node(source);
        let target_idx = self.add_node(target);

        self.graph.add_edge(source_idx, target_idx, weight);
        Ok(())
    }

    /// Get the adjacency matrix representation of the graph
    pub fn adjacency_matrix(&self) -> Array2<E>
    where
        E: num_traits::Zero + num_traits::One + Copy,
    {
        let n = self.graph.node_count();
        let mut adj_mat = Array2::zeros((n, n));

        for edge in self.graph.edge_references() {
            let (src, tgt) = (edge.source().index(), edge.target().index());
            adj_mat[[src, tgt]] = *edge.weight();
            adj_mat[[tgt, src]] = *edge.weight(); // Undirected graph
        }

        adj_mat
    }

    /// Get the degree vector of the graph
    pub fn degree_vector(&self) -> Array1<usize> {
        let n = self.graph.node_count();
        let mut degrees = Array1::zeros(n);

        for (idx, node) in self.graph.node_indices().enumerate() {
            degrees[idx] = self.graph.neighbors(node).count();
        }

        degrees
    }

    /// Get all nodes in the graph
    pub fn nodes(&self) -> Vec<&N> {
        self.graph.node_weights().collect()
    }

    /// Get all edges in the graph
    pub fn edges(&self) -> Vec<Edge<N, E>>
    where
        N: Clone,
        E: Clone,
    {
        let mut result = Vec::new();
        let node_map: HashMap<NodeIndex<Ix>, &N> = self
            .graph
            .node_indices()
            .map(|idx| (idx, self.graph.node_weight(idx).unwrap()))
            .collect();

        for edge in self.graph.edge_references() {
            let source = node_map[&edge.source()].clone();
            let target = node_map[&edge.target()].clone();
            let weight = edge.weight().clone();

            result.push(Edge {
                source,
                target,
                weight,
            });
        }

        result
    }

    /// Number of nodes in the graph
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Number of edges in the graph
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Check if the graph has a node
    pub fn has_node(&self, node: &N) -> bool {
        self.node_indices.contains_key(node)
    }

    /// Get neighbors of a node
    pub fn neighbors(&self, node: &N) -> Result<Vec<N>>
    where
        N: Clone,
    {
        if let Some(&idx) = self.node_indices.get(node) {
            let neighbors: Vec<N> = self
                .graph
                .neighbors(idx)
                .map(|neighbor_idx| self.graph[neighbor_idx].clone())
                .collect();
            Ok(neighbors)
        } else {
            Err(GraphError::NodeNotFound)
        }
    }

    /// Check if an edge exists between two nodes
    pub fn has_edge(&self, source: &N, target: &N) -> bool {
        if let (Some(&src_idx), Some(&tgt_idx)) =
            (self.node_indices.get(source), self.node_indices.get(target))
        {
            self.graph.contains_edge(src_idx, tgt_idx)
        } else {
            false
        }
    }

    /// Get the weight of an edge between two nodes
    pub fn edge_weight(&self, source: &N, target: &N) -> Result<E>
    where
        E: Clone,
    {
        if let (Some(&src_idx), Some(&tgt_idx)) =
            (self.node_indices.get(source), self.node_indices.get(target))
        {
            if let Some(edge_ref) = self.graph.find_edge(src_idx, tgt_idx) {
                Ok(self.graph[edge_ref].clone())
            } else {
                Err(GraphError::EdgeNotFound)
            }
        } else {
            Err(GraphError::NodeNotFound)
        }
    }

    /// Get the degree of a node (total number of incident edges)
    pub fn degree(&self, node: &N) -> usize {
        if let Some(idx) = self.node_indices.get(node) {
            self.graph.neighbors(*idx).count()
        } else {
            0
        }
    }

    /// Check if the graph contains a specific node
    pub fn contains_node(&self, node: &N) -> bool {
        self.node_indices.contains_key(node)
    }

    /// Get the internal petgraph structure for more advanced operations
    pub fn inner(&self) -> &PetGraph<N, E, Undirected, Ix> {
        &self.graph
    }

    /// Get a mutable reference to the internal petgraph structure
    pub fn inner_mut(&mut self) -> &mut PetGraph<N, E, Undirected, Ix> {
        &mut self.graph
    }
}

impl<N: Node, E: EdgeWeight, Ix: IndexType> Default for DiGraph<N, E, Ix> {
    fn default() -> Self {
        Self::new()
    }
}

impl<N: Node, E: EdgeWeight, Ix: IndexType> DiGraph<N, E, Ix> {
    /// Create a new empty directed graph
    pub fn new() -> Self {
        DiGraph {
            graph: PetGraph::default(),
            node_indices: HashMap::new(),
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, node: N) -> NodeIndex<Ix> {
        if let Some(idx) = self.node_indices.get(&node) {
            return *idx;
        }

        let idx = self.graph.add_node(node.clone());
        self.node_indices.insert(node, idx);
        idx
    }

    /// Add a directed edge from source to target with a given weight
    pub fn add_edge(&mut self, source: N, target: N, weight: E) -> Result<()> {
        let source_idx = self.add_node(source);
        let target_idx = self.add_node(target);

        self.graph.add_edge(source_idx, target_idx, weight);
        Ok(())
    }

    /// Get the adjacency matrix representation of the graph
    pub fn adjacency_matrix(&self) -> Array2<E>
    where
        E: num_traits::Zero + num_traits::One + Copy,
    {
        let n = self.graph.node_count();
        let mut adj_mat = Array2::zeros((n, n));

        for edge in self.graph.edge_references() {
            let (src, tgt) = (edge.source().index(), edge.target().index());
            adj_mat[[src, tgt]] = *edge.weight();
        }

        adj_mat
    }

    /// Get the in-degree vector of the graph
    pub fn in_degree_vector(&self) -> Array1<usize> {
        let n = self.graph.node_count();
        let mut degrees = Array1::zeros(n);

        for (idx, node) in self.graph.node_indices().enumerate() {
            degrees[idx] = self
                .graph
                .neighbors_directed(node, petgraph::Direction::Incoming)
                .count();
        }

        degrees
    }

    /// Get the out-degree vector of the graph
    pub fn out_degree_vector(&self) -> Array1<usize> {
        let n = self.graph.node_count();
        let mut degrees = Array1::zeros(n);

        for (idx, node) in self.graph.node_indices().enumerate() {
            degrees[idx] = self
                .graph
                .neighbors_directed(node, petgraph::Direction::Outgoing)
                .count();
        }

        degrees
    }

    /// Get all nodes in the graph
    pub fn nodes(&self) -> Vec<&N> {
        self.graph.node_weights().collect()
    }

    /// Get all edges in the graph
    pub fn edges(&self) -> Vec<Edge<N, E>>
    where
        N: Clone,
        E: Clone,
    {
        let mut result = Vec::new();
        let node_map: HashMap<NodeIndex<Ix>, &N> = self
            .graph
            .node_indices()
            .map(|idx| (idx, self.graph.node_weight(idx).unwrap()))
            .collect();

        for edge in self.graph.edge_references() {
            let source = node_map[&edge.source()].clone();
            let target = node_map[&edge.target()].clone();
            let weight = edge.weight().clone();

            result.push(Edge {
                source,
                target,
                weight,
            });
        }

        result
    }

    /// Number of nodes in the graph
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Number of edges in the graph
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Check if the graph has a node
    pub fn has_node(&self, node: &N) -> bool {
        self.node_indices.contains_key(node)
    }

    /// Get successors (outgoing neighbors) of a node
    pub fn successors(&self, node: &N) -> Result<Vec<N>>
    where
        N: Clone,
    {
        if let Some(&idx) = self.node_indices.get(node) {
            let successors: Vec<N> = self
                .graph
                .neighbors_directed(idx, petgraph::Direction::Outgoing)
                .map(|neighbor_idx| self.graph[neighbor_idx].clone())
                .collect();
            Ok(successors)
        } else {
            Err(GraphError::NodeNotFound)
        }
    }

    /// Get predecessors (incoming neighbors) of a node
    pub fn predecessors(&self, node: &N) -> Result<Vec<N>>
    where
        N: Clone,
    {
        if let Some(&idx) = self.node_indices.get(node) {
            let predecessors: Vec<N> = self
                .graph
                .neighbors_directed(idx, petgraph::Direction::Incoming)
                .map(|neighbor_idx| self.graph[neighbor_idx].clone())
                .collect();
            Ok(predecessors)
        } else {
            Err(GraphError::NodeNotFound)
        }
    }

    /// Check if an edge exists between two nodes
    pub fn has_edge(&self, source: &N, target: &N) -> bool {
        if let (Some(&src_idx), Some(&tgt_idx)) =
            (self.node_indices.get(source), self.node_indices.get(target))
        {
            self.graph.contains_edge(src_idx, tgt_idx)
        } else {
            false
        }
    }

    /// Get the weight of an edge between two nodes
    pub fn edge_weight(&self, source: &N, target: &N) -> Result<E>
    where
        E: Clone,
    {
        if let (Some(&src_idx), Some(&tgt_idx)) =
            (self.node_indices.get(source), self.node_indices.get(target))
        {
            if let Some(edge_ref) = self.graph.find_edge(src_idx, tgt_idx) {
                Ok(self.graph[edge_ref].clone())
            } else {
                Err(GraphError::EdgeNotFound)
            }
        } else {
            Err(GraphError::NodeNotFound)
        }
    }

    /// Check if the graph contains a specific node
    pub fn contains_node(&self, node: &N) -> bool {
        self.node_indices.contains_key(node)
    }

    /// Get the internal petgraph structure for more advanced operations
    pub fn inner(&self) -> &PetGraph<N, E, Directed, Ix> {
        &self.graph
    }

    /// Get a mutable reference to the internal petgraph structure
    pub fn inner_mut(&mut self) -> &mut PetGraph<N, E, Directed, Ix> {
        &mut self.graph
    }
}

/// A multi-graph structure that supports parallel edges
///
/// Unlike Graph, MultiGraph allows multiple edges between the same pair of nodes.
/// This is useful for modeling scenarios where multiple connections of different types
/// or weights can exist between nodes.
pub struct MultiGraph<N: Node, E: EdgeWeight, Ix: IndexType = u32> {
    /// Adjacency list representation: node -> list of (neighbor, edge_weight, edge_id)
    adjacency: HashMap<N, Vec<(N, E, usize)>>,
    /// All nodes in the graph
    nodes: std::collections::HashSet<N>,
    /// Edge counter for unique edge IDs
    edge_id_counter: usize,
    /// All edges in the graph with their IDs
    edges: HashMap<usize, Edge<N, E>>,
    /// Phantom data for index type
    _phantom: std::marker::PhantomData<Ix>,
}

/// A directed multi-graph structure that supports parallel edges
pub struct MultiDiGraph<N: Node, E: EdgeWeight, Ix: IndexType = u32> {
    /// Outgoing adjacency list: node -> list of (target, edge_weight, edge_id)
    out_adjacency: HashMap<N, Vec<(N, E, usize)>>,
    /// Incoming adjacency list: node -> list of (source, edge_weight, edge_id)
    in_adjacency: HashMap<N, Vec<(N, E, usize)>>,
    /// All nodes in the graph
    nodes: std::collections::HashSet<N>,
    /// Edge counter for unique edge IDs
    edge_id_counter: usize,
    /// All edges in the graph with their IDs
    edges: HashMap<usize, Edge<N, E>>,
    /// Phantom data for index type
    _phantom: std::marker::PhantomData<Ix>,
}

impl<N: Node, E: EdgeWeight, Ix: IndexType> Default for MultiGraph<N, E, Ix> {
    fn default() -> Self {
        Self::new()
    }
}

impl<N: Node, E: EdgeWeight, Ix: IndexType> MultiGraph<N, E, Ix> {
    /// Create a new empty multi-graph
    pub fn new() -> Self {
        MultiGraph {
            adjacency: HashMap::new(),
            nodes: std::collections::HashSet::new(),
            edge_id_counter: 0,
            edges: HashMap::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, node: N) {
        if !self.nodes.contains(&node) {
            self.nodes.insert(node.clone());
            self.adjacency.insert(node, Vec::new());
        }
    }

    /// Add an edge between two nodes with a given weight
    /// Returns the edge ID for reference
    pub fn add_edge(&mut self, source: N, target: N, weight: E) -> usize
    where
        N: Clone,
        E: Clone,
    {
        // Ensure both nodes exist
        self.add_node(source.clone());
        self.add_node(target.clone());

        let edge_id = self.edge_id_counter;
        self.edge_id_counter += 1;

        // Add to adjacency lists (undirected, so add both directions)
        self.adjacency
            .get_mut(&source)
            .unwrap()
            .push((target.clone(), weight.clone(), edge_id));

        if source != target {
            self.adjacency.get_mut(&target).unwrap().push((
                source.clone(),
                weight.clone(),
                edge_id,
            ));
        }

        // Store edge information
        self.edges.insert(
            edge_id,
            Edge {
                source,
                target,
                weight,
            },
        );

        edge_id
    }

    /// Get all parallel edges between two nodes
    pub fn get_edges_between(&self, source: &N, target: &N) -> Vec<(usize, &E)>
    where
        E: Clone,
    {
        let mut result = Vec::new();

        if let Some(neighbors) = self.adjacency.get(source) {
            for (neighbor, weight, edge_id) in neighbors {
                if neighbor == target {
                    result.push((*edge_id, weight));
                }
            }
        }

        result
    }

    /// Remove an edge by its ID
    pub fn remove_edge(&mut self, edge_id: usize) -> Result<Edge<N, E>>
    where
        N: Clone,
        E: Clone,
    {
        if let Some(edge) = self.edges.remove(&edge_id) {
            // Remove from adjacency lists
            if let Some(neighbors) = self.adjacency.get_mut(&edge.source) {
                neighbors.retain(|(_, _, id)| *id != edge_id);
            }

            if edge.source != edge.target {
                if let Some(neighbors) = self.adjacency.get_mut(&edge.target) {
                    neighbors.retain(|(_, _, id)| *id != edge_id);
                }
            }

            Ok(edge)
        } else {
            Err(GraphError::EdgeNotFound)
        }
    }

    /// Get all nodes in the graph
    pub fn nodes(&self) -> std::collections::hash_set::Iter<'_, N> {
        self.nodes.iter()
    }

    /// Get all edges in the graph
    pub fn edges(&self) -> std::collections::hash_map::Values<'_, usize, Edge<N, E>> {
        self.edges.values()
    }

    /// Number of nodes in the graph
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges in the graph
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Get neighbors of a node with edge weights and IDs
    pub fn neighbors_with_edges(&self, node: &N) -> Option<&Vec<(N, E, usize)>> {
        self.adjacency.get(node)
    }

    /// Get simple neighbors (without edge information)
    pub fn neighbors(&self, node: &N) -> Vec<&N> {
        if let Some(neighbors) = self.adjacency.get(node) {
            neighbors.iter().map(|(neighbor, _, _)| neighbor).collect()
        } else {
            Vec::new()
        }
    }

    /// Check if the graph contains a node
    pub fn has_node(&self, node: &N) -> bool {
        self.nodes.contains(node)
    }

    /// Get the degree of a node (total number of incident edges)
    pub fn degree(&self, node: &N) -> usize {
        self.adjacency
            .get(node)
            .map_or(0, |neighbors| neighbors.len())
    }
}

impl<N: Node, E: EdgeWeight, Ix: IndexType> Default for MultiDiGraph<N, E, Ix> {
    fn default() -> Self {
        Self::new()
    }
}

impl<N: Node, E: EdgeWeight, Ix: IndexType> MultiDiGraph<N, E, Ix> {
    /// Create a new empty directed multi-graph
    pub fn new() -> Self {
        MultiDiGraph {
            out_adjacency: HashMap::new(),
            in_adjacency: HashMap::new(),
            nodes: std::collections::HashSet::new(),
            edge_id_counter: 0,
            edges: HashMap::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, node: N) {
        if !self.nodes.contains(&node) {
            self.nodes.insert(node.clone());
            self.out_adjacency.insert(node.clone(), Vec::new());
            self.in_adjacency.insert(node, Vec::new());
        }
    }

    /// Add an edge from source to target with given weight
    /// Returns the edge ID for reference
    pub fn add_edge(&mut self, source: N, target: N, weight: E) -> usize
    where
        N: Clone,
        E: Clone,
    {
        // Ensure both nodes exist
        self.add_node(source.clone());
        self.add_node(target.clone());

        let edge_id = self.edge_id_counter;
        self.edge_id_counter += 1;

        // Add to outgoing adjacency list
        self.out_adjacency.get_mut(&source).unwrap().push((
            target.clone(),
            weight.clone(),
            edge_id,
        ));

        // Add to incoming adjacency list
        self.in_adjacency
            .get_mut(&target)
            .unwrap()
            .push((source.clone(), weight.clone(), edge_id));

        // Store edge information
        self.edges.insert(
            edge_id,
            Edge {
                source,
                target,
                weight,
            },
        );

        edge_id
    }

    /// Get all parallel edges between two nodes
    pub fn get_edges_between(&self, source: &N, target: &N) -> Vec<(usize, &E)> {
        let mut result = Vec::new();

        if let Some(neighbors) = self.out_adjacency.get(source) {
            for (neighbor, weight, edge_id) in neighbors {
                if neighbor == target {
                    result.push((*edge_id, weight));
                }
            }
        }

        result
    }

    /// Remove an edge by its ID
    pub fn remove_edge(&mut self, edge_id: usize) -> Result<Edge<N, E>>
    where
        N: Clone,
        E: Clone,
    {
        if let Some(edge) = self.edges.remove(&edge_id) {
            // Remove from outgoing adjacency list
            if let Some(neighbors) = self.out_adjacency.get_mut(&edge.source) {
                neighbors.retain(|(_, _, id)| *id != edge_id);
            }

            // Remove from incoming adjacency list
            if let Some(neighbors) = self.in_adjacency.get_mut(&edge.target) {
                neighbors.retain(|(_, _, id)| *id != edge_id);
            }

            Ok(edge)
        } else {
            Err(GraphError::EdgeNotFound)
        }
    }

    /// Get all nodes in the graph
    pub fn nodes(&self) -> std::collections::hash_set::Iter<'_, N> {
        self.nodes.iter()
    }

    /// Get all edges in the graph
    pub fn edges(&self) -> std::collections::hash_map::Values<'_, usize, Edge<N, E>> {
        self.edges.values()
    }

    /// Number of nodes in the graph
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges in the graph
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Get outgoing neighbors of a node with edge weights and IDs
    pub fn successors_with_edges(&self, node: &N) -> Option<&Vec<(N, E, usize)>> {
        self.out_adjacency.get(node)
    }

    /// Get incoming neighbors of a node with edge weights and IDs
    pub fn predecessors_with_edges(&self, node: &N) -> Option<&Vec<(N, E, usize)>> {
        self.in_adjacency.get(node)
    }

    /// Get simple successors (without edge information)
    pub fn successors(&self, node: &N) -> Vec<&N> {
        if let Some(neighbors) = self.out_adjacency.get(node) {
            neighbors.iter().map(|(neighbor, _, _)| neighbor).collect()
        } else {
            Vec::new()
        }
    }

    /// Get simple predecessors (without edge information)
    pub fn predecessors(&self, node: &N) -> Vec<&N> {
        if let Some(neighbors) = self.in_adjacency.get(node) {
            neighbors.iter().map(|(neighbor, _, _)| neighbor).collect()
        } else {
            Vec::new()
        }
    }

    /// Check if the graph contains a node
    pub fn has_node(&self, node: &N) -> bool {
        self.nodes.contains(node)
    }

    /// Get the out-degree of a node
    pub fn out_degree(&self, node: &N) -> usize {
        self.out_adjacency
            .get(node)
            .map_or(0, |neighbors| neighbors.len())
    }

    /// Get the in-degree of a node
    pub fn in_degree(&self, node: &N) -> usize {
        self.in_adjacency
            .get(node)
            .map_or(0, |neighbors| neighbors.len())
    }

    /// Get the total degree of a node (in-degree + out-degree)
    pub fn degree(&self, node: &N) -> usize {
        self.in_degree(node) + self.out_degree(node)
    }
}

/// A specialized bipartite graph structure
///
/// A bipartite graph is a graph whose vertices can be divided into two disjoint sets
/// such that no two vertices within the same set are adjacent. This implementation
/// enforces the bipartite property and provides optimized operations.
pub struct BipartiteGraph<N: Node, E: EdgeWeight, Ix: IndexType = u32> {
    /// The underlying undirected graph
    graph: Graph<N, E, Ix>,
    /// Set A of the bipartition
    set_a: std::collections::HashSet<N>,
    /// Set B of the bipartition
    set_b: std::collections::HashSet<N>,
}

impl<N: Node, E: EdgeWeight, Ix: IndexType> Default for BipartiteGraph<N, E, Ix> {
    fn default() -> Self {
        Self::new()
    }
}

impl<N: Node, E: EdgeWeight, Ix: IndexType> BipartiteGraph<N, E, Ix> {
    /// Create a new empty bipartite graph
    pub fn new() -> Self {
        BipartiteGraph {
            graph: Graph::new(),
            set_a: std::collections::HashSet::new(),
            set_b: std::collections::HashSet::new(),
        }
    }

    /// Create a bipartite graph from a regular graph if it's bipartite
    ///
    /// # Arguments
    /// * `graph` - The input graph to convert
    ///
    /// # Returns
    /// * `Result<BipartiteGraph<N, E, Ix>>` - The bipartite graph if conversion is successful
    pub fn from_graph(graph: Graph<N, E, Ix>) -> Result<Self>
    where
        N: Clone,
        E: Clone,
    {
        // Check if the graph is bipartite using the existing algorithm
        let bipartite_result = crate::algorithms::connectivity::is_bipartite(&graph);

        if !bipartite_result.is_bipartite {
            return Err(GraphError::InvalidGraph(
                "Input graph is not bipartite".to_string(),
            ));
        }

        let mut set_a = std::collections::HashSet::new();
        let mut set_b = std::collections::HashSet::new();

        // Partition nodes based on coloring
        for (node, &color) in &bipartite_result.coloring {
            if color == 0 {
                set_a.insert(node.clone());
            } else {
                set_b.insert(node.clone());
            }
        }

        Ok(BipartiteGraph {
            graph,
            set_a,
            set_b,
        })
    }

    /// Add a node to set A of the bipartition
    pub fn add_node_to_set_a(&mut self, node: N) {
        if !self.set_b.contains(&node) {
            self.graph.add_node(node.clone());
            self.set_a.insert(node);
        }
    }

    /// Add a node to set B of the bipartition
    pub fn add_node_to_set_b(&mut self, node: N) {
        if !self.set_a.contains(&node) {
            self.graph.add_node(node.clone());
            self.set_b.insert(node);
        }
    }

    /// Add an edge between nodes from different sets
    ///
    /// # Arguments
    /// * `source` - Source node (must be in one set)
    /// * `target` - Target node (must be in the other set)
    /// * `weight` - Edge weight
    ///
    /// # Returns
    /// * `Result<()>` - Success or error if nodes are in the same set
    pub fn add_edge(&mut self, source: N, target: N, weight: E) -> Result<()>
    where
        N: Clone,
    {
        // Validate bipartite property
        let source_in_a = self.set_a.contains(&source);
        let source_in_b = self.set_b.contains(&source);
        let target_in_a = self.set_a.contains(&target);
        let target_in_b = self.set_b.contains(&target);

        // Check if both nodes exist in the graph
        if (!source_in_a && !source_in_b) || (!target_in_a && !target_in_b) {
            return Err(GraphError::NodeNotFound);
        }

        // Check bipartite constraint: nodes must be in different sets
        if (source_in_a && target_in_a) || (source_in_b && target_in_b) {
            return Err(GraphError::InvalidGraph(
                "Cannot add edge between nodes in the same partition".to_string(),
            ));
        }

        self.graph.add_edge(source, target, weight)
    }

    /// Get all nodes in set A
    pub fn set_a(&self) -> &std::collections::HashSet<N> {
        &self.set_a
    }

    /// Get all nodes in set B
    pub fn set_b(&self) -> &std::collections::HashSet<N> {
        &self.set_b
    }

    /// Get the size of set A
    pub fn set_a_size(&self) -> usize {
        self.set_a.len()
    }

    /// Get the size of set B
    pub fn set_b_size(&self) -> usize {
        self.set_b.len()
    }

    /// Check which set a node belongs to
    ///
    /// # Arguments
    /// * `node` - The node to check
    ///
    /// # Returns
    /// * `Some(0)` if node is in set A, `Some(1)` if in set B, `None` if not found
    pub fn node_set(&self, node: &N) -> Option<u8> {
        if self.set_a.contains(node) {
            Some(0)
        } else if self.set_b.contains(node) {
            Some(1)
        } else {
            None
        }
    }

    /// Get neighbors of a node (always from the opposite set)
    pub fn neighbors(&self, node: &N) -> Result<Vec<N>>
    where
        N: Clone,
    {
        self.graph.neighbors(node)
    }

    /// Get neighbors in set A for a node in set B
    pub fn neighbors_in_a(&self, node: &N) -> Result<Vec<N>>
    where
        N: Clone,
    {
        if !self.set_b.contains(node) {
            return Err(GraphError::InvalidGraph(
                "Node must be in set B to get neighbors in set A".to_string(),
            ));
        }

        let all_neighbors = self.graph.neighbors(node)?;
        Ok(all_neighbors
            .into_iter()
            .filter(|n| self.set_a.contains(n))
            .collect())
    }

    /// Get neighbors in set B for a node in set A
    pub fn neighbors_in_b(&self, node: &N) -> Result<Vec<N>>
    where
        N: Clone,
    {
        if !self.set_a.contains(node) {
            return Err(GraphError::InvalidGraph(
                "Node must be in set A to get neighbors in set B".to_string(),
            ));
        }

        let all_neighbors = self.graph.neighbors(node)?;
        Ok(all_neighbors
            .into_iter()
            .filter(|n| self.set_b.contains(n))
            .collect())
    }

    /// Get the degree of a node
    pub fn degree(&self, node: &N) -> usize {
        self.graph.degree(node)
    }

    /// Get all nodes in the graph
    pub fn nodes(&self) -> Vec<&N> {
        self.graph.nodes()
    }

    /// Get all edges in the graph
    pub fn edges(&self) -> Vec<Edge<N, E>>
    where
        N: Clone,
        E: Clone,
    {
        self.graph.edges()
    }

    /// Number of nodes in the graph
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Number of edges in the graph
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Check if the graph has a node
    pub fn has_node(&self, node: &N) -> bool {
        self.graph.has_node(node)
    }

    /// Check if an edge exists between two nodes
    pub fn has_edge(&self, source: &N, target: &N) -> bool {
        self.graph.has_edge(source, target)
    }

    /// Get the weight of an edge between two nodes
    pub fn edge_weight(&self, source: &N, target: &N) -> Result<E>
    where
        E: Clone,
    {
        self.graph.edge_weight(source, target)
    }

    /// Get the adjacency matrix representation of the bipartite graph
    ///
    /// Returns a matrix where rows correspond to set A and columns to set B
    pub fn biadjacency_matrix(&self) -> Array2<E>
    where
        E: num_traits::Zero + Copy,
        N: Clone,
    {
        let a_size = self.set_a.len();
        let b_size = self.set_b.len();
        let mut biadj_mat = Array2::zeros((a_size, b_size));

        // Create mappings from nodes to indices
        let a_nodes: Vec<&N> = self.set_a.iter().collect();
        let b_nodes: Vec<&N> = self.set_b.iter().collect();

        let a_to_idx: HashMap<&N, usize> =
            a_nodes.iter().enumerate().map(|(i, &n)| (n, i)).collect();
        let b_to_idx: HashMap<&N, usize> =
            b_nodes.iter().enumerate().map(|(i, &n)| (n, i)).collect();

        // Fill the biadjacency matrix
        for edge in self.graph.edges() {
            let (a_idx, b_idx) = if self.set_a.contains(&edge.source) {
                (a_to_idx[&edge.source], b_to_idx[&edge.target])
            } else {
                (a_to_idx[&edge.target], b_to_idx[&edge.source])
            };
            biadj_mat[[a_idx, b_idx]] = edge.weight;
        }

        biadj_mat
    }

    /// Convert to a regular graph
    pub fn to_graph(self) -> Graph<N, E, Ix> {
        self.graph
    }

    /// Get a reference to the underlying graph
    pub fn as_graph(&self) -> &Graph<N, E, Ix> {
        &self.graph
    }

    /// Check if the graph is complete bipartite (all possible edges exist)
    pub fn is_complete(&self) -> bool {
        let expected_edges = self.set_a.len() * self.set_b.len();
        self.edge_count() == expected_edges
    }

    /// Get the maximum possible number of edges for this bipartite graph
    pub fn max_edges(&self) -> usize {
        self.set_a.len() * self.set_b.len()
    }

    /// Get the density of the bipartite graph (actual edges / max possible edges)
    pub fn density(&self) -> f64 {
        if self.max_edges() == 0 {
            0.0
        } else {
            self.edge_count() as f64 / self.max_edges() as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_undirected_graph_creation() {
        let mut graph: Graph<&str, f64> = Graph::new();

        graph.add_node("A");
        graph.add_node("B");
        graph.add_node("C");

        graph.add_edge("A", "B", 1.0).unwrap();
        graph.add_edge("B", "C", 2.0).unwrap();

        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 2);
        assert!(graph.has_node(&"A"));
        assert!(graph.has_node(&"B"));
        assert!(graph.has_node(&"C"));
    }

    #[test]
    fn test_directed_graph_creation() {
        let mut graph: DiGraph<i32, f64> = DiGraph::new();

        graph.add_node(1);
        graph.add_node(2);
        graph.add_node(3);

        graph.add_edge(1, 2, 1.5).unwrap();
        graph.add_edge(2, 3, 2.5).unwrap();

        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 2);
        assert!(graph.has_node(&1));
        assert!(graph.has_node(&2));
        assert!(graph.has_node(&3));
    }

    #[test]
    fn test_adjacency_matrix() {
        let mut graph: Graph<u8, f64> = Graph::new();

        graph.add_node(0);
        graph.add_node(1);
        graph.add_node(2);

        graph.add_edge(0, 1, 1.0).unwrap();
        graph.add_edge(1, 2, 2.0).unwrap();

        let adj_mat = graph.adjacency_matrix();

        // Expected matrix:
        // [[0.0, 1.0, 0.0],
        //  [1.0, 0.0, 2.0],
        //  [0.0, 2.0, 0.0]]

        assert_eq!(adj_mat.shape(), &[3, 3]);
        assert_eq!(adj_mat[[0, 1]], 1.0);
        assert_eq!(adj_mat[[1, 0]], 1.0);
        assert_eq!(adj_mat[[1, 2]], 2.0);
        assert_eq!(adj_mat[[2, 1]], 2.0);
        assert_eq!(adj_mat[[0, 2]], 0.0);
        assert_eq!(adj_mat[[2, 0]], 0.0);
    }

    #[test]
    fn test_degree_vector() {
        let mut graph: Graph<char, f64> = Graph::new();

        graph.add_node('A');
        graph.add_node('B');
        graph.add_node('C');
        graph.add_node('D');

        graph.add_edge('A', 'B', 1.0).unwrap();
        graph.add_edge('B', 'C', 1.0).unwrap();
        graph.add_edge('C', 'D', 1.0).unwrap();
        graph.add_edge('D', 'A', 1.0).unwrap();
        graph.add_edge('A', 'C', 1.0).unwrap();

        let degrees = graph.degree_vector();

        // A connects to B, D, C = 3
        // B connects to A, C = 2
        // C connects to B, D, A = 3
        // D connects to C, A = 2

        assert_eq!(degrees, Array1::from_vec(vec![3, 2, 3, 2]));
    }

    #[test]
    fn test_multigraph_parallel_edges() {
        let mut graph: MultiGraph<&str, f64> = MultiGraph::new();

        // Add parallel edges between A and B
        let _edge1 = graph.add_edge("A", "B", 1.0);
        let edge2 = graph.add_edge("A", "B", 2.0);
        let _edge3 = graph.add_edge("A", "B", 3.0);

        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 3);

        // Check that we can get all parallel edges
        let edges_ab = graph.get_edges_between(&"A", &"B");
        assert_eq!(edges_ab.len(), 3);

        // Check edge weights
        let weights: Vec<f64> = edges_ab.iter().map(|(_, &weight)| weight).collect();
        assert!(weights.contains(&1.0));
        assert!(weights.contains(&2.0));
        assert!(weights.contains(&3.0));

        // Remove one edge
        let removed_edge = graph.remove_edge(edge2).unwrap();
        assert_eq!(removed_edge.weight, 2.0);
        assert_eq!(graph.edge_count(), 2);

        // Check remaining edges
        let edges_ab = graph.get_edges_between(&"A", &"B");
        assert_eq!(edges_ab.len(), 2);
    }

    #[test]
    fn test_multidigraph_parallel_edges() {
        let mut graph: MultiDiGraph<&str, f64> = MultiDiGraph::new();

        // Add parallel directed edges from A to B
        let edge1 = graph.add_edge("A", "B", 1.0);
        let _edge2 = graph.add_edge("A", "B", 2.0);

        // Add edge in opposite direction
        let _edge3 = graph.add_edge("B", "A", 3.0);

        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 3);

        // Check outgoing edges from A
        let edges_ab = graph.get_edges_between(&"A", &"B");
        assert_eq!(edges_ab.len(), 2);

        // Check outgoing edges from B
        let edges_ba = graph.get_edges_between(&"B", &"A");
        assert_eq!(edges_ba.len(), 1);

        // Check degrees
        assert_eq!(graph.out_degree(&"A"), 2);
        assert_eq!(graph.in_degree(&"A"), 1);
        assert_eq!(graph.out_degree(&"B"), 1);
        assert_eq!(graph.in_degree(&"B"), 2);

        // Remove edge and check
        graph.remove_edge(edge1).unwrap();
        assert_eq!(graph.edge_count(), 2);
        assert_eq!(graph.out_degree(&"A"), 1);
        assert_eq!(graph.in_degree(&"B"), 1);
    }

    #[test]
    fn test_bipartite_graph_creation() {
        let mut bipartite: BipartiteGraph<&str, f64> = BipartiteGraph::new();

        // Add nodes to different sets
        bipartite.add_node_to_set_a("A1");
        bipartite.add_node_to_set_a("A2");
        bipartite.add_node_to_set_b("B1");
        bipartite.add_node_to_set_b("B2");

        assert_eq!(bipartite.set_a_size(), 2);
        assert_eq!(bipartite.set_b_size(), 2);
        assert_eq!(bipartite.node_count(), 4);

        // Add valid edges (between different sets)
        assert!(bipartite.add_edge("A1", "B1", 1.0).is_ok());
        assert!(bipartite.add_edge("A2", "B2", 2.0).is_ok());

        assert_eq!(bipartite.edge_count(), 2);
    }

    #[test]
    fn test_bipartite_graph_invalid_edges() {
        let mut bipartite: BipartiteGraph<i32, f64> = BipartiteGraph::new();

        bipartite.add_node_to_set_a(1);
        bipartite.add_node_to_set_a(2);
        bipartite.add_node_to_set_b(3);
        bipartite.add_node_to_set_b(4);

        // Try to add edge within same set (should fail)
        assert!(bipartite.add_edge(1, 2, 1.0).is_err());
        assert!(bipartite.add_edge(3, 4, 1.0).is_err());

        // Valid edge should work
        assert!(bipartite.add_edge(1, 3, 1.0).is_ok());
    }

    #[test]
    fn test_bipartite_graph_from_regular_graph() {
        // Create a bipartite graph (square)
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_edge(1, 2, 1.0).unwrap();
        graph.add_edge(2, 3, 2.0).unwrap();
        graph.add_edge(3, 4, 3.0).unwrap();
        graph.add_edge(4, 1, 4.0).unwrap();

        // Convert to bipartite graph
        let bipartite = BipartiteGraph::from_graph(graph).unwrap();

        assert_eq!(bipartite.node_count(), 4);
        assert_eq!(bipartite.edge_count(), 4);
        assert_eq!(bipartite.set_a_size() + bipartite.set_b_size(), 4);

        // Check that nodes are properly partitioned
        assert!(bipartite.set_a_size() == 2);
        assert!(bipartite.set_b_size() == 2);
    }

    #[test]
    fn test_bipartite_graph_from_non_bipartite_graph() {
        // Create a non-bipartite graph (triangle)
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_edge(1, 2, 1.0).unwrap();
        graph.add_edge(2, 3, 2.0).unwrap();
        graph.add_edge(3, 1, 3.0).unwrap();

        // Should fail to convert
        assert!(BipartiteGraph::from_graph(graph).is_err());
    }

    #[test]
    fn test_bipartite_graph_node_set_identification() {
        let mut bipartite: BipartiteGraph<char, f64> = BipartiteGraph::new();

        bipartite.add_node_to_set_a('A');
        bipartite.add_node_to_set_b('B');

        assert_eq!(bipartite.node_set(&'A'), Some(0));
        assert_eq!(bipartite.node_set(&'B'), Some(1));
        assert_eq!(bipartite.node_set(&'C'), None);
    }

    #[test]
    fn test_bipartite_graph_neighbors() {
        let mut bipartite: BipartiteGraph<&str, f64> = BipartiteGraph::new();

        bipartite.add_node_to_set_a("A1");
        bipartite.add_node_to_set_a("A2");
        bipartite.add_node_to_set_b("B1");
        bipartite.add_node_to_set_b("B2");

        bipartite.add_edge("A1", "B1", 1.0).unwrap();
        bipartite.add_edge("A1", "B2", 2.0).unwrap();
        bipartite.add_edge("A2", "B1", 3.0).unwrap();

        // Test neighbors_in_b for nodes in set A
        let a1_neighbors = bipartite.neighbors_in_b(&"A1").unwrap();
        assert_eq!(a1_neighbors.len(), 2);
        assert!(a1_neighbors.contains(&"B1"));
        assert!(a1_neighbors.contains(&"B2"));

        // Test neighbors_in_a for nodes in set B
        let b1_neighbors = bipartite.neighbors_in_a(&"B1").unwrap();
        assert_eq!(b1_neighbors.len(), 2);
        assert!(b1_neighbors.contains(&"A1"));
        assert!(b1_neighbors.contains(&"A2"));

        // Test invalid neighbor queries
        assert!(bipartite.neighbors_in_b(&"B1").is_err()); // B1 is not in set A
        assert!(bipartite.neighbors_in_a(&"A1").is_err()); // A1 is not in set B
    }

    #[test]
    fn test_bipartite_graph_biadjacency_matrix() {
        let mut bipartite: BipartiteGraph<i32, f64> = BipartiteGraph::new();

        bipartite.add_node_to_set_a(1);
        bipartite.add_node_to_set_a(2);
        bipartite.add_node_to_set_b(3);
        bipartite.add_node_to_set_b(4);

        bipartite.add_edge(1, 3, 5.0).unwrap();
        bipartite.add_edge(2, 4, 7.0).unwrap();

        let biadj = bipartite.biadjacency_matrix();
        assert_eq!(biadj.shape(), &[2, 2]);

        // Check that the matrix has the expected structure
        // Note: exact positions may vary based on hash set iteration order
        let total_sum: f64 = biadj.iter().sum();
        assert_eq!(total_sum, 12.0); // 5.0 + 7.0 = 12.0
    }

    #[test]
    fn test_bipartite_graph_completeness() {
        let mut bipartite: BipartiteGraph<i32, f64> = BipartiteGraph::new();

        bipartite.add_node_to_set_a(1);
        bipartite.add_node_to_set_a(2);
        bipartite.add_node_to_set_b(3);
        bipartite.add_node_to_set_b(4);

        // Not complete initially
        assert!(!bipartite.is_complete());
        assert_eq!(bipartite.max_edges(), 4); // 2 * 2 = 4
        assert_eq!(bipartite.density(), 0.0);

        // Add all possible edges
        bipartite.add_edge(1, 3, 1.0).unwrap();
        bipartite.add_edge(1, 4, 1.0).unwrap();
        bipartite.add_edge(2, 3, 1.0).unwrap();
        bipartite.add_edge(2, 4, 1.0).unwrap();

        // Now it should be complete
        assert!(bipartite.is_complete());
        assert_eq!(bipartite.density(), 1.0);
    }

    #[test]
    fn test_bipartite_graph_conversion() {
        let mut bipartite: BipartiteGraph<&str, f64> = BipartiteGraph::new();

        bipartite.add_node_to_set_a("A");
        bipartite.add_node_to_set_b("B");
        bipartite.add_edge("A", "B", 3.15).unwrap();

        // Convert to regular graph
        let regular_graph = bipartite.to_graph();
        assert_eq!(regular_graph.node_count(), 2);
        assert_eq!(regular_graph.edge_count(), 1);
        assert!(regular_graph.has_edge(&"A", &"B"));
    }

    #[test]
    fn test_bipartite_graph_empty() {
        let bipartite: BipartiteGraph<i32, f64> = BipartiteGraph::new();

        assert_eq!(bipartite.node_count(), 0);
        assert_eq!(bipartite.edge_count(), 0);
        assert_eq!(bipartite.set_a_size(), 0);
        assert_eq!(bipartite.set_b_size(), 0);
        assert_eq!(bipartite.max_edges(), 0);
        assert_eq!(bipartite.density(), 0.0);
        assert!(bipartite.is_complete()); // Vacuously true for empty graph
    }

    #[test]
    fn test_multigraph_self_loops() {
        let mut graph: MultiGraph<i32, f64> = MultiGraph::new();

        // Add self loops
        let _edge1 = graph.add_edge(1, 1, 10.0);
        let _edge2 = graph.add_edge(1, 1, 20.0);

        assert_eq!(graph.node_count(), 1);
        assert_eq!(graph.edge_count(), 2);

        // Self loops should appear in neighbors
        let neighbors = graph.neighbors(&1);
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.iter().all(|&&n| n == 1));

        // Degree should count self loops
        assert_eq!(graph.degree(&1), 2);

        // Check self-loop edges
        let self_edges = graph.get_edges_between(&1, &1);
        assert_eq!(self_edges.len(), 2);
    }
}
