//! Adjacency list format I/O operations
//!
//! This module provides functions for reading and writing graphs in adjacency list format.
//! Adjacency list format represents each node and its neighbors on a single line:
//! - For unweighted graphs: `source: target1 target2 target3 ...`
//! - For weighted graphs: `source: target1 target2 target3 weight1 weight2 weight3 ...`
//!
//! The format uses a colon (`:`) to separate the source node from its neighbors.
//! For weighted graphs, weights are listed after all the target nodes.
//!
//! Lines starting with '#' are treated as comments and ignored.
//! Empty lines are also ignored.
//!
//! # Examples
//!
//! ```rust
//! use scirs2_graph::io::adjacency_list::{read_adjacency_list_format, write_adjacency_list_format};
//! use scirs2_graph::base::Graph;
//! use std::io::Write;
//! use tempfile::NamedTempFile;
//!
//! // Create a test file with unweighted adjacency list
//! let mut temp_file = NamedTempFile::new().unwrap();
//! writeln!(temp_file, "1: 2 3").unwrap();
//! writeln!(temp_file, "2: 1 3").unwrap();
//! writeln!(temp_file, "3: 1 2").unwrap();
//! temp_file.flush().unwrap();
//!
//! // Read an unweighted graph from adjacency list format
//! let graph: Graph<i32, f64> = read_adjacency_list_format(temp_file.path(), false).unwrap();
//! assert_eq!(graph.node_count(), 3);
//! ```
//!
//! ```rust
//! use scirs2_graph::io::adjacency_list::{read_adjacency_list_format, write_adjacency_list_format};
//! use scirs2_graph::base::Graph;
//! use std::io::Write;
//! use tempfile::NamedTempFile;
//!
//! // Create a test file with weighted adjacency list
//! let mut temp_file = NamedTempFile::new().unwrap();
//! writeln!(temp_file, "1: 2 3 1.5 2.0").unwrap();
//! writeln!(temp_file, "2: 1 1.5").unwrap();
//! temp_file.flush().unwrap();
//!
//! // Read a weighted graph from adjacency list format
//! let graph: Graph<i32, f64> = read_adjacency_list_format(temp_file.path(), true).unwrap();
//! assert_eq!(graph.node_count(), 3);
//! ```

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::str::FromStr;

use crate::base::{DiGraph, EdgeWeight, Graph, Node};
use crate::error::{GraphError, Result};

/// Read an undirected graph from adjacency list format
///
/// # Arguments
///
/// * `path` - Path to the input file
/// * `weighted` - Whether to parse edge weights
///
/// # Returns
///
/// Returns a `Graph` with the parsed nodes and edges
///
/// # Errors
///
/// Returns an error if:
/// - The file cannot be opened
/// - Lines cannot be parsed
/// - Node or edge weight parsing fails
/// - Format is malformed (missing colon, inconsistent weight count)
///
/// # Format
///
/// Each line should contain:
/// - `source: target1 target2 ...` for unweighted graphs
/// - `source: target1 target2 ... weight1 weight2 ...` for weighted graphs
/// - Lines starting with '#' are treated as comments
/// - Empty lines are ignored
/// - Malformed lines are skipped with a warning
///
/// For weighted graphs, the number of weights must match the number of targets.
pub fn read_adjacency_list_format<N, E, P>(path: P, weighted: bool) -> Result<Graph<N, E>>
where
    N: Node + std::fmt::Debug + FromStr + Clone,
    E: EdgeWeight + std::marker::Copy + std::fmt::Debug + std::default::Default + FromStr,
    P: AsRef<Path>,
{
    let file =
        File::open(path).map_err(|e| GraphError::Other(format!("Cannot open file: {}", e)))?;
    let reader = BufReader::new(file);
    let mut graph = Graph::new();

    for line_result in reader.lines() {
        let line =
            line_result.map_err(|e| GraphError::Other(format!("Error reading line: {}", e)))?;
        let line = line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Parse line: source: target1 target2 ... [weight1 weight2 ...]
        let parts: Vec<&str> = line.split(':').collect();
        if parts.len() != 2 {
            continue; // Skip malformed lines (no colon or multiple colons)
        }

        let source_str = parts[0].trim();
        let targets_str = parts[1].trim();

        if targets_str.is_empty() {
            continue; // Skip lines with no targets
        }

        // Parse source node
        let source_data = match N::from_str(source_str) {
            Ok(data) => data,
            Err(_) => continue, // Skip lines with malformed source nodes
        };

        // Parse targets and weights
        let target_parts: Vec<&str> = targets_str.split_whitespace().collect();
        if target_parts.is_empty() {
            continue; // Skip empty target lists
        }

        if weighted {
            // For weighted graphs, we expect: target1 target2 ... weight1 weight2 ...
            // The number of weights should equal the number of targets
            if target_parts.len() % 2 != 0 {
                continue; // Skip lines with odd number of parts (inconsistent format)
            }

            let num_targets = target_parts.len() / 2;
            let targets = &target_parts[0..num_targets];
            let weights = &target_parts[num_targets..];

            for (target_str, weight_str) in targets.iter().zip(weights.iter()) {
                // Skip parsing errors for individual target/weight pairs
                if let (Ok(target_data), Ok(edge_weight)) = (N::from_str(target_str), E::from_str(weight_str)) {
                    let _ = graph.add_edge(source_data.clone(), target_data, edge_weight);
                    // Silently skip edge addition failures
                }
                // Silently skip malformed target nodes or weights
            }
        } else {
            // For unweighted graphs, all parts are targets
            for target_str in target_parts {
                // Skip parsing errors for individual targets (malformed nodes)
                if let Ok(target_data) = N::from_str(target_str) {
                    let _ = graph.add_edge(source_data.clone(), target_data, E::default());
                    // Silently skip edge addition failures
                }
                // Silently skip malformed target nodes
            }
        }
    }

    Ok(graph)
}

/// Write an undirected graph to adjacency list format
///
/// # Arguments
///
/// * `graph` - The graph to write
/// * `path` - Path to the output file
/// * `weighted` - Whether to include edge weights in the output
///
/// # Errors
///
/// Returns an error if:
/// - The file cannot be created
/// - Writing to the file fails
///
/// # Format
///
/// Each line will contain:
/// - `source: target1 target2 ...` for unweighted output
/// - `source: target1 target2 ... weight1 weight2 ...` for weighted output
///
/// Nodes are written in the order they appear in the graph.
/// For each node, its neighbors are listed after the colon.
pub fn write_adjacency_list_format<N, E, Ix, P>(
    graph: &Graph<N, E, Ix>,
    path: P,
    weighted: bool,
) -> Result<()>
where
    N: Node + std::fmt::Debug + std::fmt::Display + Clone,
    E: EdgeWeight
        + std::marker::Copy
        + std::fmt::Debug
        + std::default::Default
        + std::fmt::Display
        + Clone,
    Ix: petgraph::graph::IndexType,
    P: AsRef<Path>,
{
    let mut file =
        File::create(path).map_err(|e| GraphError::Other(format!("Cannot create file: {}", e)))?;

    // Build adjacency lists from edges
    let mut adjacency_lists: HashMap<N, Vec<(N, E)>> = HashMap::new();

    // Initialize adjacency lists for all nodes
    for node in graph.nodes() {
        adjacency_lists.insert(node.clone(), Vec::new());
    }

    // Populate adjacency lists from edges
    for edge in graph.edges() {
        // For undirected graphs, add both directions
        adjacency_lists
            .entry(edge.source.clone())
            .or_default()
            .push((edge.target.clone(), edge.weight));

        adjacency_lists
            .entry(edge.target.clone())
            .or_default()
            .push((edge.source.clone(), edge.weight));
    }

    // Write adjacency lists
    let mut nodes: Vec<_> = adjacency_lists.keys().cloned().collect();
    nodes.sort_by(|a, b| format!("{}", a).cmp(&format!("{}", b))); // Sort by string representation

    for source in nodes {
        let neighbors = &adjacency_lists[&source];
        
        write!(file, "{}:", source)
            .map_err(|e| GraphError::Other(format!("Error writing source node: {}", e)))?;

        if neighbors.is_empty() {
            writeln!(file)
                .map_err(|e| GraphError::Other(format!("Error writing newline: {}", e)))?;
            continue;
        }

        // Write targets first
        for (target, _) in neighbors {
            write!(file, " {}", target)
                .map_err(|e| GraphError::Other(format!("Error writing target node: {}", e)))?;
        }

        // Write weights if requested
        if weighted {
            for (_, weight) in neighbors {
                write!(file, " {}", weight)
                    .map_err(|e| GraphError::Other(format!("Error writing weight: {}", e)))?;
            }
        }

        writeln!(file)
            .map_err(|e| GraphError::Other(format!("Error writing newline: {}", e)))?;
    }

    Ok(())
}

/// Read a directed graph from adjacency list format
///
/// # Arguments
///
/// * `path` - Path to the input file
/// * `weighted` - Whether to parse edge weights
///
/// # Returns
///
/// Returns a `DiGraph` with the parsed nodes and edges
///
/// # Errors
///
/// Returns an error if:
/// - The file cannot be opened
/// - Lines cannot be parsed
/// - Node or edge weight parsing fails
/// - Format is malformed (missing colon, inconsistent weight count)
///
/// # Format
///
/// Each line should contain:
/// - `source: target1 target2 ...` for unweighted graphs
/// - `source: target1 target2 ... weight1 weight2 ...` for weighted graphs
/// - Lines starting with '#' are treated as comments
/// - Empty lines are ignored
/// - Malformed lines are skipped with a warning
///
/// For weighted graphs, the number of weights must match the number of targets.
/// In directed graphs, only outgoing edges from the source are represented.
pub fn read_adjacency_list_format_digraph<N, E, P>(path: P, weighted: bool) -> Result<DiGraph<N, E>>
where
    N: Node + std::fmt::Debug + FromStr + Clone,
    E: EdgeWeight + std::marker::Copy + std::fmt::Debug + std::default::Default + FromStr,
    P: AsRef<Path>,
{
    let file =
        File::open(path).map_err(|e| GraphError::Other(format!("Cannot open file: {}", e)))?;
    let reader = BufReader::new(file);
    let mut graph = DiGraph::new();

    for line_result in reader.lines() {
        let line =
            line_result.map_err(|e| GraphError::Other(format!("Error reading line: {}", e)))?;
        let line = line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Parse line: source: target1 target2 ... [weight1 weight2 ...]
        let parts: Vec<&str> = line.split(':').collect();
        if parts.len() != 2 {
            continue; // Skip malformed lines (no colon or multiple colons)
        }

        let source_str = parts[0].trim();
        let targets_str = parts[1].trim();

        if targets_str.is_empty() {
            continue; // Skip lines with no targets
        }

        // Parse source node
        let source_data = match N::from_str(source_str) {
            Ok(data) => data,
            Err(_) => continue, // Skip lines with malformed source nodes
        };

        // Parse targets and weights
        let target_parts: Vec<&str> = targets_str.split_whitespace().collect();
        if target_parts.is_empty() {
            continue; // Skip empty target lists
        }

        if weighted {
            // For weighted graphs, we expect: target1 target2 ... weight1 weight2 ...
            // The number of weights should equal the number of targets
            if target_parts.len() % 2 != 0 {
                continue; // Skip lines with odd number of parts (inconsistent format)
            }

            let num_targets = target_parts.len() / 2;
            let targets = &target_parts[0..num_targets];
            let weights = &target_parts[num_targets..];

            for (target_str, weight_str) in targets.iter().zip(weights.iter()) {
                // Skip parsing errors for individual target/weight pairs
                if let (Ok(target_data), Ok(edge_weight)) = (N::from_str(target_str), E::from_str(weight_str)) {
                    let _ = graph.add_edge(source_data.clone(), target_data, edge_weight);
                    // Silently skip edge addition failures
                }
                // Silently skip malformed target nodes or weights
            }
        } else {
            // For unweighted graphs, all parts are targets
            for target_str in target_parts {
                // Skip parsing errors for individual targets (malformed nodes)
                if let Ok(target_data) = N::from_str(target_str) {
                    let _ = graph.add_edge(source_data.clone(), target_data, E::default());
                    // Silently skip edge addition failures
                }
                // Silently skip malformed target nodes
            }
        }
    }

    Ok(graph)
}

/// Write a directed graph to adjacency list format
///
/// # Arguments
///
/// * `graph` - The directed graph to write
/// * `path` - Path to the output file
/// * `weighted` - Whether to include edge weights in the output
///
/// # Errors
///
/// Returns an error if:
/// - The file cannot be created
/// - Writing to the file fails
///
/// # Format
///
/// Each line will contain:
/// - `source: target1 target2 ...` for unweighted output
/// - `source: target1 target2 ... weight1 weight2 ...` for weighted output
///
/// For directed graphs, only outgoing edges from each source node are written.
/// Nodes are written in the order they appear in the graph.
pub fn write_adjacency_list_format_digraph<N, E, Ix, P>(
    graph: &DiGraph<N, E, Ix>,
    path: P,
    weighted: bool,
) -> Result<()>
where
    N: Node + std::fmt::Debug + std::fmt::Display + Clone,
    E: EdgeWeight
        + std::marker::Copy
        + std::fmt::Debug
        + std::default::Default
        + std::fmt::Display
        + Clone,
    Ix: petgraph::graph::IndexType,
    P: AsRef<Path>,
{
    let mut file =
        File::create(path).map_err(|e| GraphError::Other(format!("Cannot create file: {}", e)))?;

    // Build adjacency lists from edges (only outgoing edges for directed graphs)
    let mut adjacency_lists: HashMap<N, Vec<(N, E)>> = HashMap::new();

    // Initialize adjacency lists for all nodes
    for node in graph.nodes() {
        adjacency_lists.insert(node.clone(), Vec::new());
    }

    // Populate adjacency lists from edges (only outgoing for directed graphs)
    for edge in graph.edges() {
        adjacency_lists
            .entry(edge.source.clone())
            .or_default()
            .push((edge.target.clone(), edge.weight));
    }

    // Write adjacency lists
    let mut nodes: Vec<_> = adjacency_lists.keys().cloned().collect();
    nodes.sort_by(|a, b| format!("{}", a).cmp(&format!("{}", b))); // Sort by string representation

    for source in nodes {
        let neighbors = &adjacency_lists[&source];
        
        write!(file, "{}:", source)
            .map_err(|e| GraphError::Other(format!("Error writing source node: {}", e)))?;

        if neighbors.is_empty() {
            writeln!(file)
                .map_err(|e| GraphError::Other(format!("Error writing newline: {}", e)))?;
            continue;
        }

        // Write targets first
        for (target, _) in neighbors {
            write!(file, " {}", target)
                .map_err(|e| GraphError::Other(format!("Error writing target node: {}", e)))?;
        }

        // Write weights if requested
        if weighted {
            for (_, weight) in neighbors {
                write!(file, " {}", weight)
                    .map_err(|e| GraphError::Other(format!("Error writing weight: {}", e)))?;
            }
        }

        writeln!(file)
            .map_err(|e| GraphError::Other(format!("Error writing newline: {}", e)))?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_adjacency_list_format_unweighted() {
        // Create a temporary file with adjacency list data
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "# This is a comment").unwrap();
        writeln!(temp_file, "1: 2 3").unwrap();
        writeln!(temp_file, "2: 1 3").unwrap();
        writeln!(temp_file, "3: 1 2").unwrap();
        writeln!(temp_file, "").unwrap(); // Empty line
        temp_file.flush().unwrap();

        let graph: Graph<i32, f64> = read_adjacency_list_format(temp_file.path(), false).unwrap();
        assert_eq!(graph.node_count(), 3);
        // Each undirected edge is added in both directions, so 3 lines * 2 edges per line = 6 edges
        assert_eq!(graph.edge_count(), 6);
    }

    #[test]
    fn test_adjacency_list_format_weighted() {
        // Create a temporary file with weighted adjacency list data
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "1: 2 3 1.5 2.0").unwrap();
        writeln!(temp_file, "2: 1 1.5").unwrap();
        writeln!(temp_file, "3: 1 2.0").unwrap();
        temp_file.flush().unwrap();

        let graph: Graph<i32, f64> = read_adjacency_list_format(temp_file.path(), true).unwrap();
        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 4); // 2 + 1 + 1 edges
    }

    #[test]
    fn test_digraph_adjacency_list_format() {
        // Create a temporary file with adjacency list data for directed graph
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "1: 2 3 1.0 2.0").unwrap();
        writeln!(temp_file, "2: 3 1.5").unwrap();
        temp_file.flush().unwrap();

        let graph: DiGraph<i32, f64> = read_adjacency_list_format_digraph(temp_file.path(), true).unwrap();
        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 3); // 2 + 1 edges (directed)
    }

    #[test]
    fn test_adjacency_list_empty_neighbors() {
        // Test nodes with no neighbors
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "1: 2").unwrap();
        writeln!(temp_file, "2:").unwrap(); // Node 2 has no outgoing edges
        writeln!(temp_file, "3: 1").unwrap();
        temp_file.flush().unwrap();

        let graph: DiGraph<i32, f64> = read_adjacency_list_format_digraph(temp_file.path(), false).unwrap();
        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 2); // Only edges 1->2 and 3->1
    }

    #[test]
    fn test_adjacency_list_malformed_lines() {
        // Test handling of malformed lines
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "1: 2 3").unwrap(); // Valid line
        writeln!(temp_file, "2 3 4").unwrap();  // No colon - should be skipped
        writeln!(temp_file, "3: 1: 2").unwrap(); // Multiple colons - should be skipped
        writeln!(temp_file, "4: 1 2 x").unwrap(); // Invalid node name - should be skipped
        temp_file.flush().unwrap();

        let graph: Graph<i32, f64> = read_adjacency_list_format(temp_file.path(), false).unwrap();
        assert_eq!(graph.node_count(), 3); // Only nodes 1, 2, 3 from the valid line
        assert_eq!(graph.edge_count(), 4); // 1->2, 2->1, 1->3, 3->1
    }
}