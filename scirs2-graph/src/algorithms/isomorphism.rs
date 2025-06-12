//! Graph isomorphism and subgraph matching algorithms
//!
//! This module contains algorithms for graph isomorphism testing and subgraph matching.

use crate::base::{EdgeWeight, Graph, IndexType, Node};
use std::collections::HashMap;
use std::hash::Hash;

/// Find all subgraph matches of a pattern graph in a target graph
///
/// Returns a vector of mappings from pattern nodes to target nodes for each match found.
pub fn find_subgraph_matches<N1, N2, E, Ix>(
    pattern: &Graph<N1, E, Ix>,
    target: &Graph<N2, E, Ix>,
) -> Vec<HashMap<N1, N2>>
where
    N1: Node + Clone + Hash + Eq,
    N2: Node + Clone + Hash + Eq,
    E: EdgeWeight,
    Ix: IndexType,
{
    let pattern_nodes: Vec<N1> = pattern.nodes().into_iter().cloned().collect();
    let target_nodes: Vec<N2> = target.nodes().into_iter().cloned().collect();

    if pattern_nodes.is_empty() || pattern_nodes.len() > target_nodes.len() {
        return vec![];
    }

    let mut matches = Vec::new();
    let mut current_mapping = HashMap::new();

    // Try to match starting from each target node
    for start_node in &target_nodes {
        find_matches_recursive(
            &pattern_nodes,
            pattern,
            target,
            &mut current_mapping,
            0,
            start_node,
            &mut matches,
        );
    }

    matches
}

fn find_matches_recursive<N1, N2, E, Ix>(
    pattern_nodes: &[N1],
    pattern: &Graph<N1, E, Ix>,
    target: &Graph<N2, E, Ix>,
    current_mapping: &mut HashMap<N1, N2>,
    pattern_idx: usize,
    target_node: &N2,
    matches: &mut Vec<HashMap<N1, N2>>,
) where
    N1: Node + Clone + Hash + Eq,
    N2: Node + Clone + Hash + Eq,
    E: EdgeWeight,
    Ix: IndexType,
{
    if pattern_idx >= pattern_nodes.len() {
        // Found a complete match
        matches.push(current_mapping.clone());
        return;
    }

    let pattern_node = &pattern_nodes[pattern_idx];

    // Check if target_node is already mapped
    if current_mapping.values().any(|n| n == target_node) {
        return;
    }

    // Try to map pattern_node to target_node
    current_mapping.insert(pattern_node.clone(), target_node.clone());

    // Check if this mapping is consistent with edges
    if is_mapping_consistent(pattern, target, current_mapping) {
        if pattern_idx + 1 < pattern_nodes.len() {
            // Continue mapping with remaining nodes
            if let Ok(target_neighbors) = target.neighbors(target_node) {
                for next_target in target_neighbors {
                    find_matches_recursive(
                        pattern_nodes,
                        pattern,
                        target,
                        current_mapping,
                        pattern_idx + 1,
                        &next_target,
                        matches,
                    );
                }
            }

            // Also try non-neighbors
            for next_target in &target.nodes().into_iter().cloned().collect::<Vec<_>>() {
                if !current_mapping.values().any(|n| n == next_target) {
                    find_matches_recursive(
                        pattern_nodes,
                        pattern,
                        target,
                        current_mapping,
                        pattern_idx + 1,
                        next_target,
                        matches,
                    );
                }
            }
        } else {
            // Last node - check if complete mapping is valid
            find_matches_recursive(
                pattern_nodes,
                pattern,
                target,
                current_mapping,
                pattern_idx + 1,
                target_node,
                matches,
            );
        }
    }

    // Backtrack
    current_mapping.remove(pattern_node);
}

fn is_mapping_consistent<N1, N2, E, Ix>(
    pattern: &Graph<N1, E, Ix>,
    target: &Graph<N2, E, Ix>,
    mapping: &HashMap<N1, N2>,
) -> bool
where
    N1: Node + Hash + Eq,
    N2: Node + Hash + Eq,
    E: EdgeWeight,
    Ix: IndexType,
{
    // Check that all edges in the pattern exist in the target under the mapping
    for (n1, n2) in mapping {
        for (m1, m2) in mapping {
            if n1 != m1 {
                let pattern_has_edge = pattern.has_edge(n1, m1);
                let target_has_edge = target.has_edge(n2, m2);

                if pattern_has_edge && !target_has_edge {
                    return false;
                }
            }
        }
    }

    true
}

/// Check if two graphs are isomorphic
///
/// Two graphs are isomorphic if there exists a bijection between their vertices
/// that preserves the edge-adjacency relationship.
///
/// # Arguments
/// * `graph1` - The first graph
/// * `graph2` - The second graph
///
/// # Returns
/// * `bool` - True if the graphs are isomorphic, false otherwise
pub fn are_graphs_isomorphic<N1, N2, E, Ix>(
    graph1: &Graph<N1, E, Ix>,
    graph2: &Graph<N2, E, Ix>,
) -> bool
where
    N1: Node + Clone + Hash + Eq,
    N2: Node + Clone + Hash + Eq,
    E: EdgeWeight,
    Ix: IndexType,
{
    // Quick checks first
    if graph1.node_count() != graph2.node_count() || graph1.edge_count() != graph2.edge_count() {
        return false;
    }

    // Check degree sequence
    if !have_same_degree_sequence(graph1, graph2) {
        return false;
    }

    // If either graph is empty, they're isomorphic
    if graph1.node_count() == 0 {
        return true;
    }

    // Try to find an isomorphism
    find_isomorphism(graph1, graph2).is_some()
}

/// Find an isomorphism between two graphs if one exists
///
/// # Arguments
/// * `graph1` - The first graph
/// * `graph2` - The second graph
///
/// # Returns
/// * `Option<HashMap<N1, N2>>` - Mapping from graph1 nodes to graph2 nodes if isomorphic
pub fn find_isomorphism<N1, N2, E, Ix>(
    graph1: &Graph<N1, E, Ix>,
    graph2: &Graph<N2, E, Ix>,
) -> Option<HashMap<N1, N2>>
where
    N1: Node + Clone + Hash + Eq,
    N2: Node + Clone + Hash + Eq,
    E: EdgeWeight,
    Ix: IndexType,
{
    let nodes1: Vec<N1> = graph1.nodes().into_iter().cloned().collect();
    let nodes2: Vec<N2> = graph2.nodes().into_iter().cloned().collect();

    if nodes1.len() != nodes2.len() {
        return None;
    }

    let mut mapping = HashMap::new();
    if backtrack_isomorphism(&nodes1, &nodes2, graph1, graph2, &mut mapping, 0) {
        Some(mapping)
    } else {
        None
    }
}

/// Check if two graphs have the same degree sequence
fn have_same_degree_sequence<N1, N2, E, Ix>(
    graph1: &Graph<N1, E, Ix>,
    graph2: &Graph<N2, E, Ix>,
) -> bool
where
    N1: Node,
    N2: Node,
    E: EdgeWeight,
    Ix: IndexType,
{
    let mut degrees1: Vec<usize> = graph1
        .nodes()
        .iter()
        .map(|node| {
            graph1
                .neighbors(node)
                .map_or(0, |neighbors| neighbors.len())
        })
        .collect();

    let mut degrees2: Vec<usize> = graph2
        .nodes()
        .iter()
        .map(|node| {
            graph2
                .neighbors(node)
                .map_or(0, |neighbors| neighbors.len())
        })
        .collect();

    degrees1.sort_unstable();
    degrees2.sort_unstable();

    degrees1 == degrees2
}

/// Backtracking algorithm to find isomorphism
fn backtrack_isomorphism<N1, N2, E, Ix>(
    nodes1: &[N1],
    nodes2: &[N2],
    graph1: &Graph<N1, E, Ix>,
    graph2: &Graph<N2, E, Ix>,
    mapping: &mut HashMap<N1, N2>,
    depth: usize,
) -> bool
where
    N1: Node + Clone + Hash + Eq,
    N2: Node + Clone + Hash + Eq,
    E: EdgeWeight,
    Ix: IndexType,
{
    // Base case: all nodes mapped
    if depth == nodes1.len() {
        return is_valid_isomorphism(graph1, graph2, mapping);
    }

    let node1 = &nodes1[depth];

    for node2 in nodes2 {
        // Skip if this node2 is already mapped
        if mapping.values().any(|mapped| mapped == node2) {
            continue;
        }

        // Check degree compatibility
        let degree1 = graph1
            .neighbors(node1)
            .map_or(0, |neighbors| neighbors.len());
        let degree2 = graph2
            .neighbors(node2)
            .map_or(0, |neighbors| neighbors.len());

        if degree1 != degree2 {
            continue;
        }

        // Try this mapping
        mapping.insert(node1.clone(), node2.clone());

        // Check if current partial mapping is consistent
        if is_partial_mapping_valid(graph1, graph2, mapping, depth + 1) 
            && backtrack_isomorphism(nodes1, nodes2, graph1, graph2, mapping, depth + 1) {
            return true;
        }

        // Backtrack
        mapping.remove(node1);
    }

    false
}

/// Check if a partial mapping is valid (preserves edges among mapped nodes)
fn is_partial_mapping_valid<N1, N2, E, Ix>(
    graph1: &Graph<N1, E, Ix>,
    graph2: &Graph<N2, E, Ix>,
    mapping: &HashMap<N1, N2>,
    _mapped_count: usize,
) -> bool
where
    N1: Node + Hash + Eq,
    N2: Node + Hash + Eq,
    E: EdgeWeight,
    Ix: IndexType,
{
    for (n1, n2) in mapping {
        for (m1, m2) in mapping {
            if n1 != m1 {
                let edge1_exists = graph1.has_edge(n1, m1);
                let edge2_exists = graph2.has_edge(n2, m2);

                if edge1_exists != edge2_exists {
                    return false;
                }
            }
        }
    }
    true
}

/// Check if a complete mapping is a valid isomorphism
fn is_valid_isomorphism<N1, N2, E, Ix>(
    graph1: &Graph<N1, E, Ix>,
    graph2: &Graph<N2, E, Ix>,
    mapping: &HashMap<N1, N2>,
) -> bool
where
    N1: Node + Hash + Eq,
    N2: Node + Hash + Eq,
    E: EdgeWeight,
    Ix: IndexType,
{
    // Check that the mapping preserves all edges
    for (n1, n2) in mapping {
        for (m1, m2) in mapping {
            if n1 != m1 {
                let edge1_exists = graph1.has_edge(n1, m1);
                let edge2_exists = graph2.has_edge(n2, m2);

                if edge1_exists != edge2_exists {
                    return false;
                }
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Result as GraphResult;
    use crate::generators::create_graph;

    #[test]
    fn test_find_subgraph_matches() -> GraphResult<()> {
        // Create a pattern graph (triangle)
        let mut pattern = create_graph::<&str, ()>();
        pattern.add_edge("A", "B", ())?;
        pattern.add_edge("B", "C", ())?;
        pattern.add_edge("C", "A", ())?;

        // Create a target graph with two triangles
        let mut target = create_graph::<&str, ()>();
        // First triangle
        target.add_edge("1", "2", ())?;
        target.add_edge("2", "3", ())?;
        target.add_edge("3", "1", ())?;
        // Second triangle
        target.add_edge("4", "5", ())?;
        target.add_edge("5", "6", ())?;
        target.add_edge("6", "4", ())?;
        // Connect them
        target.add_edge("3", "4", ())?;

        let matches = find_subgraph_matches(&pattern, &target);

        // Should find at least 2 triangles
        assert!(matches.len() >= 2);

        // Each match should have 3 mappings
        for match_map in &matches {
            assert_eq!(match_map.len(), 3);
        }

        Ok(())
    }

    #[test]
    fn test_no_subgraph_match() -> GraphResult<()> {
        // Create a pattern graph (triangle)
        let mut pattern = create_graph::<&str, ()>();
        pattern.add_edge("A", "B", ())?;
        pattern.add_edge("B", "C", ())?;
        pattern.add_edge("C", "A", ())?;

        // Create a target graph with no triangles (path)
        let mut target = create_graph::<&str, ()>();
        target.add_edge("1", "2", ())?;
        target.add_edge("2", "3", ())?;
        target.add_edge("3", "4", ())?;

        let matches = find_subgraph_matches(&pattern, &target);

        // Should find no matches
        assert_eq!(matches.len(), 0);

        Ok(())
    }

    #[test]
    fn test_isomorphic_graphs() -> GraphResult<()> {
        // Create two isomorphic triangles with different node labels
        let mut graph1 = create_graph::<&str, ()>();
        graph1.add_edge("A", "B", ())?;
        graph1.add_edge("B", "C", ())?;
        graph1.add_edge("C", "A", ())?;

        let mut graph2 = create_graph::<i32, ()>();
        graph2.add_edge(1, 2, ())?;
        graph2.add_edge(2, 3, ())?;
        graph2.add_edge(3, 1, ())?;

        assert!(are_graphs_isomorphic(&graph1, &graph2));

        let isomorphism = find_isomorphism(&graph1, &graph2);
        assert!(isomorphism.is_some());

        Ok(())
    }

    #[test]
    fn test_non_isomorphic_graphs() -> GraphResult<()> {
        // Triangle vs path
        let mut triangle = create_graph::<i32, ()>();
        triangle.add_edge(1, 2, ())?;
        triangle.add_edge(2, 3, ())?;
        triangle.add_edge(3, 1, ())?;

        let mut path = create_graph::<i32, ()>();
        path.add_edge(1, 2, ())?;
        path.add_edge(2, 3, ())?;

        assert!(!are_graphs_isomorphic(&triangle, &path));
        assert!(find_isomorphism(&triangle, &path).is_none());

        Ok(())
    }

    #[test]
    fn test_different_size_graphs() -> GraphResult<()> {
        let mut small = create_graph::<i32, ()>();
        small.add_edge(1, 2, ())?;

        let mut large = create_graph::<i32, ()>();
        large.add_edge(1, 2, ())?;
        large.add_edge(2, 3, ())?;

        assert!(!are_graphs_isomorphic(&small, &large));
        assert!(find_isomorphism(&small, &large).is_none());

        Ok(())
    }

    #[test]
    fn test_empty_graphs() {
        let graph1 = create_graph::<i32, ()>();
        let graph2 = create_graph::<&str, ()>();

        assert!(are_graphs_isomorphic(&graph1, &graph2));
        assert!(find_isomorphism(&graph1, &graph2).is_some());
    }
}
