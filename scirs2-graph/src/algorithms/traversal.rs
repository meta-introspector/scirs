//! Graph traversal algorithms
//!
//! This module provides breadth-first search (BFS) and depth-first search (DFS)
//! algorithms for both directed and undirected graphs.

use std::collections::{HashSet, VecDeque};

use crate::base::{DiGraph, EdgeWeight, Graph, Node};
use crate::error::{GraphError, Result};

/// Performs breadth-first search (BFS) from a given starting node
///
/// # Arguments
/// * `graph` - The graph to traverse
/// * `start` - The starting node
///
/// # Returns
/// * `Result<Vec<N>>` - The nodes visited in BFS order
pub fn breadth_first_search<N, E, Ix>(graph: &Graph<N, E, Ix>, start: &N) -> Result<Vec<N>>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    if !graph.has_node(start) {
        return Err(GraphError::InvalidGraph(format!(
            "Start node {:?} not found",
            start
        )));
    }

    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    let mut result = Vec::new();

    // Find the starting node index
    let start_idx = graph
        .inner()
        .node_indices()
        .find(|&idx| graph.inner()[idx] == *start)
        .unwrap();

    queue.push_back(start_idx);
    visited.insert(start_idx);

    while let Some(current_idx) = queue.pop_front() {
        result.push(graph.inner()[current_idx].clone());

        // Visit all unvisited neighbors
        for neighbor_idx in graph.inner().neighbors(current_idx) {
            if !visited.contains(&neighbor_idx) {
                visited.insert(neighbor_idx);
                queue.push_back(neighbor_idx);
            }
        }
    }

    Ok(result)
}

/// Performs breadth-first search (BFS) from a given starting node in a directed graph
///
/// # Arguments
/// * `graph` - The directed graph to traverse
/// * `start` - The starting node
///
/// # Returns
/// * `Result<Vec<N>>` - The nodes visited in BFS order
pub fn breadth_first_search_digraph<N, E, Ix>(
    graph: &DiGraph<N, E, Ix>,
    start: &N,
) -> Result<Vec<N>>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    if !graph.has_node(start) {
        return Err(GraphError::InvalidGraph(format!(
            "Start node {:?} not found",
            start
        )));
    }

    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    let mut result = Vec::new();

    // Find the starting node index
    let start_idx = graph
        .inner()
        .node_indices()
        .find(|&idx| graph.inner()[idx] == *start)
        .unwrap();

    queue.push_back(start_idx);
    visited.insert(start_idx);

    while let Some(current_idx) = queue.pop_front() {
        result.push(graph.inner()[current_idx].clone());

        // Visit all unvisited neighbors (outgoing edges only for directed graph)
        for neighbor_idx in graph
            .inner()
            .neighbors_directed(current_idx, petgraph::Direction::Outgoing)
        {
            if !visited.contains(&neighbor_idx) {
                visited.insert(neighbor_idx);
                queue.push_back(neighbor_idx);
            }
        }
    }

    Ok(result)
}

/// Performs depth-first search (DFS) from a given starting node
///
/// # Arguments
/// * `graph` - The graph to traverse
/// * `start` - The starting node
///
/// # Returns
/// * `Result<Vec<N>>` - The nodes visited in DFS order
pub fn depth_first_search<N, E, Ix>(graph: &Graph<N, E, Ix>, start: &N) -> Result<Vec<N>>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    if !graph.has_node(start) {
        return Err(GraphError::InvalidGraph(format!(
            "Start node {:?} not found",
            start
        )));
    }

    let mut visited = HashSet::new();
    let mut stack = Vec::new();
    let mut result = Vec::new();

    // Find the starting node index
    let start_idx = graph
        .inner()
        .node_indices()
        .find(|&idx| graph.inner()[idx] == *start)
        .unwrap();

    stack.push(start_idx);

    while let Some(current_idx) = stack.pop() {
        if !visited.contains(&current_idx) {
            visited.insert(current_idx);
            result.push(graph.inner()[current_idx].clone());

            // Add all unvisited neighbors to the stack (in reverse order for consistent traversal)
            let mut neighbors: Vec<_> = graph.inner().neighbors(current_idx).collect();
            neighbors.reverse();
            for neighbor_idx in neighbors {
                if !visited.contains(&neighbor_idx) {
                    stack.push(neighbor_idx);
                }
            }
        }
    }

    Ok(result)
}

/// Performs depth-first search (DFS) from a given starting node in a directed graph
///
/// # Arguments
/// * `graph` - The directed graph to traverse
/// * `start` - The starting node
///
/// # Returns
/// * `Result<Vec<N>>` - The nodes visited in DFS order
pub fn depth_first_search_digraph<N, E, Ix>(graph: &DiGraph<N, E, Ix>, start: &N) -> Result<Vec<N>>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    if !graph.has_node(start) {
        return Err(GraphError::InvalidGraph(format!(
            "Start node {:?} not found",
            start
        )));
    }

    let mut visited = HashSet::new();
    let mut stack = Vec::new();
    let mut result = Vec::new();

    // Find the starting node index
    let start_idx = graph
        .inner()
        .node_indices()
        .find(|&idx| graph.inner()[idx] == *start)
        .unwrap();

    stack.push(start_idx);

    while let Some(current_idx) = stack.pop() {
        if !visited.contains(&current_idx) {
            visited.insert(current_idx);
            result.push(graph.inner()[current_idx].clone());

            // Add all unvisited neighbors to the stack (outgoing edges only for directed graph)
            let mut neighbors: Vec<_> = graph
                .inner()
                .neighbors_directed(current_idx, petgraph::Direction::Outgoing)
                .collect();
            neighbors.reverse();
            for neighbor_idx in neighbors {
                if !visited.contains(&neighbor_idx) {
                    stack.push(neighbor_idx);
                }
            }
        }
    }

    Ok(result)
}

/// Performs bidirectional breadth-first search to find a path between two nodes
///
/// This algorithm searches from both the start and goal nodes simultaneously,
/// which can be more efficient than unidirectional search for finding paths
/// in large graphs.
///
/// # Arguments
/// * `graph` - The graph to search
/// * `start` - The starting node
/// * `goal` - The goal node
///
/// # Returns
/// * `Result<Option<Vec<N>>>` - The path from start to goal, or None if no path exists
pub fn bidirectional_search<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    start: &N,
    goal: &N,
) -> Result<Option<Vec<N>>>
where
    N: Node + std::fmt::Debug + Clone + std::hash::Hash + Eq,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    if !graph.has_node(start) || !graph.has_node(goal) {
        return Err(GraphError::InvalidGraph(
            "Start or goal node not found".to_string(),
        ));
    }

    if start == goal {
        return Ok(Some(vec![start.clone()]));
    }

    // Find node indices
    let start_idx = graph
        .inner()
        .node_indices()
        .find(|&idx| graph.inner()[idx] == *start)
        .unwrap();
    let goal_idx = graph
        .inner()
        .node_indices()
        .find(|&idx| graph.inner()[idx] == *goal)
        .unwrap();

    let mut forward_queue = VecDeque::new();
    let mut backward_queue = VecDeque::new();
    let mut forward_visited = HashSet::new();
    let mut backward_visited = HashSet::new();
    let mut forward_parent: std::collections::HashMap<
        petgraph::graph::NodeIndex<Ix>,
        petgraph::graph::NodeIndex<Ix>,
    > = std::collections::HashMap::new();
    let mut backward_parent: std::collections::HashMap<
        petgraph::graph::NodeIndex<Ix>,
        petgraph::graph::NodeIndex<Ix>,
    > = std::collections::HashMap::new();

    forward_queue.push_back(start_idx);
    backward_queue.push_back(goal_idx);
    forward_visited.insert(start_idx);
    backward_visited.insert(goal_idx);

    while !forward_queue.is_empty() || !backward_queue.is_empty() {
        // Forward search step
        if !forward_queue.is_empty() {
            let current = forward_queue.pop_front().unwrap();

            // Check if we've met the backward search
            if backward_visited.contains(&current) {
                return Ok(Some(reconstruct_bidirectional_path(
                    graph,
                    start_idx,
                    goal_idx,
                    current,
                    &forward_parent,
                    &backward_parent,
                )));
            }

            for neighbor in graph.inner().neighbors(current) {
                if !forward_visited.contains(&neighbor) {
                    forward_visited.insert(neighbor);
                    forward_parent.insert(neighbor, current);
                    forward_queue.push_back(neighbor);
                }
            }
        }

        // Backward search step
        if !backward_queue.is_empty() {
            let current = backward_queue.pop_front().unwrap();

            // Check if we've met the forward search
            if forward_visited.contains(&current) {
                return Ok(Some(reconstruct_bidirectional_path(
                    graph,
                    start_idx,
                    goal_idx,
                    current,
                    &forward_parent,
                    &backward_parent,
                )));
            }

            for neighbor in graph.inner().neighbors(current) {
                if !backward_visited.contains(&neighbor) {
                    backward_visited.insert(neighbor);
                    backward_parent.insert(neighbor, current);
                    backward_queue.push_back(neighbor);
                }
            }
        }
    }

    Ok(None)
}

/// Bidirectional search for directed graphs
pub fn bidirectional_search_digraph<N, E, Ix>(
    graph: &DiGraph<N, E, Ix>,
    start: &N,
    goal: &N,
) -> Result<Option<Vec<N>>>
where
    N: Node + std::fmt::Debug + Clone + std::hash::Hash + Eq,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    if !graph.has_node(start) || !graph.has_node(goal) {
        return Err(GraphError::InvalidGraph(
            "Start or goal node not found".to_string(),
        ));
    }

    if start == goal {
        return Ok(Some(vec![start.clone()]));
    }

    // Find node indices
    let start_idx = graph
        .inner()
        .node_indices()
        .find(|&idx| graph.inner()[idx] == *start)
        .unwrap();
    let goal_idx = graph
        .inner()
        .node_indices()
        .find(|&idx| graph.inner()[idx] == *goal)
        .unwrap();

    let mut forward_queue = VecDeque::new();
    let mut backward_queue = VecDeque::new();
    let mut forward_visited = HashSet::new();
    let mut backward_visited = HashSet::new();
    let mut forward_parent: std::collections::HashMap<
        petgraph::graph::NodeIndex<Ix>,
        petgraph::graph::NodeIndex<Ix>,
    > = std::collections::HashMap::new();
    let mut backward_parent: std::collections::HashMap<
        petgraph::graph::NodeIndex<Ix>,
        petgraph::graph::NodeIndex<Ix>,
    > = std::collections::HashMap::new();

    forward_queue.push_back(start_idx);
    backward_queue.push_back(goal_idx);
    forward_visited.insert(start_idx);
    backward_visited.insert(goal_idx);

    while !forward_queue.is_empty() || !backward_queue.is_empty() {
        // Forward search step (follow outgoing edges)
        if !forward_queue.is_empty() {
            let current = forward_queue.pop_front().unwrap();

            // Check if we've met the backward search
            if backward_visited.contains(&current) {
                return Ok(Some(reconstruct_bidirectional_path_digraph(
                    graph,
                    start_idx,
                    goal_idx,
                    current,
                    &forward_parent,
                    &backward_parent,
                )));
            }

            for neighbor in graph
                .inner()
                .neighbors_directed(current, petgraph::Direction::Outgoing)
            {
                if !forward_visited.contains(&neighbor) {
                    forward_visited.insert(neighbor);
                    forward_parent.insert(neighbor, current);
                    forward_queue.push_back(neighbor);
                }
            }
        }

        // Backward search step (follow incoming edges)
        if !backward_queue.is_empty() {
            let current = backward_queue.pop_front().unwrap();

            // Check if we've met the forward search
            if forward_visited.contains(&current) {
                return Ok(Some(reconstruct_bidirectional_path_digraph(
                    graph,
                    start_idx,
                    goal_idx,
                    current,
                    &forward_parent,
                    &backward_parent,
                )));
            }

            for neighbor in graph
                .inner()
                .neighbors_directed(current, petgraph::Direction::Incoming)
            {
                if !backward_visited.contains(&neighbor) {
                    backward_visited.insert(neighbor);
                    backward_parent.insert(neighbor, current);
                    backward_queue.push_back(neighbor);
                }
            }
        }
    }

    Ok(None)
}

/// Helper function to reconstruct path from bidirectional search
fn reconstruct_bidirectional_path<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    start_idx: petgraph::graph::NodeIndex<Ix>,
    goal_idx: petgraph::graph::NodeIndex<Ix>,
    meeting_point: petgraph::graph::NodeIndex<Ix>,
    forward_parent: &std::collections::HashMap<
        petgraph::graph::NodeIndex<Ix>,
        petgraph::graph::NodeIndex<Ix>,
    >,
    backward_parent: &std::collections::HashMap<
        petgraph::graph::NodeIndex<Ix>,
        petgraph::graph::NodeIndex<Ix>,
    >,
) -> Vec<N>
where
    N: Node + Clone,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    let mut forward_path = Vec::new();
    let mut current = meeting_point;

    // Build forward path (from start to meeting point)
    while current != start_idx {
        forward_path.push(graph.inner()[current].clone());
        current = forward_parent[&current];
    }
    forward_path.push(graph.inner()[start_idx].clone());
    forward_path.reverse();

    // Build backward path (from meeting point to goal)
    let mut backward_path = Vec::new();
    current = meeting_point;
    while current != goal_idx {
        if let Some(&parent) = backward_parent.get(&current) {
            current = parent;
            backward_path.push(graph.inner()[current].clone());
        } else {
            break;
        }
    }

    // Combine paths
    let mut full_path = forward_path;
    full_path.extend(backward_path);
    full_path
}

/// Helper function to reconstruct path from bidirectional search for directed graphs
fn reconstruct_bidirectional_path_digraph<N, E, Ix>(
    graph: &DiGraph<N, E, Ix>,
    start_idx: petgraph::graph::NodeIndex<Ix>,
    goal_idx: petgraph::graph::NodeIndex<Ix>,
    meeting_point: petgraph::graph::NodeIndex<Ix>,
    forward_parent: &std::collections::HashMap<
        petgraph::graph::NodeIndex<Ix>,
        petgraph::graph::NodeIndex<Ix>,
    >,
    backward_parent: &std::collections::HashMap<
        petgraph::graph::NodeIndex<Ix>,
        petgraph::graph::NodeIndex<Ix>,
    >,
) -> Vec<N>
where
    N: Node + Clone,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    let mut forward_path = Vec::new();
    let mut current = meeting_point;

    // Build forward path (from start to meeting point)
    while current != start_idx {
        forward_path.push(graph.inner()[current].clone());
        current = forward_parent[&current];
    }
    forward_path.push(graph.inner()[start_idx].clone());
    forward_path.reverse();

    // Build backward path (from meeting point to goal)
    let mut backward_path = Vec::new();
    current = meeting_point;
    while current != goal_idx {
        if let Some(&parent) = backward_parent.get(&current) {
            current = parent;
            backward_path.push(graph.inner()[current].clone());
        } else {
            break;
        }
    }

    // Combine paths
    let mut full_path = forward_path;
    full_path.extend(backward_path);
    full_path
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_breadth_first_search() {
        let mut graph: Graph<i32, f64> = Graph::new();

        // Create a simple graph: 1 -- 2 -- 3
        //                             |
        //                             4
        graph.add_edge(1, 2, 1.0).unwrap();
        graph.add_edge(2, 3, 1.0).unwrap();
        graph.add_edge(2, 4, 1.0).unwrap();

        let result = breadth_first_search(&graph, &1).unwrap();

        // BFS should visit nodes level by level
        assert_eq!(result[0], 1);
        assert_eq!(result[1], 2);
        // 3 and 4 can be in any order as they're at the same level
        assert!(result.contains(&3));
        assert!(result.contains(&4));
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_breadth_first_search_digraph() {
        let mut graph: DiGraph<i32, f64> = DiGraph::new();

        // Create a directed graph: 1 -> 2 -> 3
        //                               |
        //                               v
        //                               4
        graph.add_edge(1, 2, 1.0).unwrap();
        graph.add_edge(2, 3, 1.0).unwrap();
        graph.add_edge(2, 4, 1.0).unwrap();

        let result = breadth_first_search_digraph(&graph, &1).unwrap();

        assert_eq!(result[0], 1);
        assert_eq!(result[1], 2);
        assert!(result.contains(&3));
        assert!(result.contains(&4));
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_depth_first_search() {
        let mut graph: Graph<i32, f64> = Graph::new();

        // Create a simple graph
        graph.add_edge(1, 2, 1.0).unwrap();
        graph.add_edge(1, 3, 1.0).unwrap();
        graph.add_edge(2, 4, 1.0).unwrap();

        let result = depth_first_search(&graph, &1).unwrap();

        // DFS should visit nodes depth-first
        assert_eq!(result[0], 1);
        assert_eq!(result.len(), 4);
        assert!(result.contains(&2));
        assert!(result.contains(&3));
        assert!(result.contains(&4));
    }

    #[test]
    fn test_depth_first_search_digraph() {
        let mut graph: DiGraph<i32, f64> = DiGraph::new();

        // Create a directed graph
        graph.add_edge(1, 2, 1.0).unwrap();
        graph.add_edge(1, 3, 1.0).unwrap();
        graph.add_edge(2, 4, 1.0).unwrap();

        let result = depth_first_search_digraph(&graph, &1).unwrap();

        assert_eq!(result[0], 1);
        assert_eq!(result.len(), 4);
        assert!(result.contains(&2));
        assert!(result.contains(&3));
        assert!(result.contains(&4));
    }

    #[test]
    fn test_bidirectional_search() {
        let mut graph: Graph<i32, f64> = Graph::new();

        // Create a chain: 1 -- 2 -- 3 -- 4 -- 5
        graph.add_edge(1, 2, 1.0).unwrap();
        graph.add_edge(2, 3, 1.0).unwrap();
        graph.add_edge(3, 4, 1.0).unwrap();
        graph.add_edge(4, 5, 1.0).unwrap();

        // Test finding path from 1 to 5
        let path = bidirectional_search(&graph, &1, &5).unwrap().unwrap();
        assert_eq!(path[0], 1);
        assert_eq!(path[path.len() - 1], 5);
        assert!(path.len() <= 5);

        // Test path from node to itself
        let same_path = bidirectional_search(&graph, &3, &3).unwrap().unwrap();
        assert_eq!(same_path, vec![3]);

        // Test disconnected graph
        let mut disconnected: Graph<i32, f64> = Graph::new();
        disconnected.add_node(1);
        disconnected.add_node(2);
        let no_path = bidirectional_search(&disconnected, &1, &2).unwrap();
        assert_eq!(no_path, None);
    }

    #[test]
    fn test_bidirectional_search_digraph() {
        let mut graph: DiGraph<i32, f64> = DiGraph::new();

        // Create a directed chain: 1 -> 2 -> 3 -> 4 -> 5
        graph.add_edge(1, 2, 1.0).unwrap();
        graph.add_edge(2, 3, 1.0).unwrap();
        graph.add_edge(3, 4, 1.0).unwrap();
        graph.add_edge(4, 5, 1.0).unwrap();

        // Test finding path from 1 to 5
        let path = bidirectional_search_digraph(&graph, &1, &5)
            .unwrap()
            .unwrap();
        assert_eq!(path[0], 1);
        assert_eq!(path[path.len() - 1], 5);

        // Test no path in wrong direction
        let no_path = bidirectional_search_digraph(&graph, &5, &1).unwrap();
        assert_eq!(no_path, None);
    }
}
