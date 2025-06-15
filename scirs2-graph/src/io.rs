//! Input/output operations for graphs
//!
//! This module provides functions for reading and writing graph data
//! in various formats.

use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::str::FromStr;

use crate::base::{DiGraph, EdgeWeight, Graph, Node};
use crate::error::{GraphError, Result};

/// Supported file formats for graph I/O
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphFormat {
    /// Edge list format (one edge per line: source target [weight])
    EdgeList,
    /// Adjacency list format (source: target1 target2 ...)
    AdjacencyList,
    /// Matrix Market format (sparse matrix format)
    MatrixMarket,
    /// GraphML format (XML-based format for graphs)
    GraphML,
    /// DOT format (Graphviz format)
    Dot,
    /// JSON graph format
    Json,
}

/// Reads a graph from a file
///
/// # Arguments
/// * `path` - Path to the file
/// * `format` - Format of the file
/// * `weighted` - Whether the graph has edge weights
/// * `directed` - Whether the graph is directed
///
/// # Returns
/// * `Ok(Graph)` - The graph read from the file
/// * `Err(GraphError)` - If there was an error reading the file
pub fn read_graph<N, E, P>(
    path: P,
    format: GraphFormat,
    weighted: bool,
    _directed: bool,
) -> Result<Graph<N, E>>
where
    N: Node + std::fmt::Debug + FromStr + Clone,
    E: EdgeWeight + std::marker::Copy + std::fmt::Debug + std::default::Default + FromStr,
    P: AsRef<Path>,
{
    match format {
        GraphFormat::EdgeList => read_edge_list_format(path, weighted),
        GraphFormat::AdjacencyList => read_adjacency_list_format(path, weighted),
        GraphFormat::Dot => read_dot_format(path, weighted),
        GraphFormat::Json => read_json_format(path, weighted),
        _ => Err(GraphError::Other(format!(
            "Format {:?} not implemented yet",
            format
        ))),
    }
}

/// Reads a directed graph from a file - stubbed implementation
///
/// # Arguments
/// * `path` - Path to the file
/// * `format` - Format of the file
/// * `weighted` - Whether the graph has edge weights
///
/// # Returns
/// * `Ok(DiGraph)` - The directed graph read from the file
/// * `Err(GraphError)` - If there was an error reading the file
pub fn read_digraph<N, E, P>(
    _path: P,
    _format: GraphFormat,
    _weighted: bool,
) -> Result<DiGraph<N, E>>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight + std::marker::Copy + std::fmt::Debug + std::default::Default,
    P: AsRef<Path>,
{
    Err(GraphError::Other(
        "Function not implemented yet".to_string(),
    ))
}

/// Writes a graph to a file
///
/// # Arguments
/// * `graph` - The graph to write
/// * `path` - Path to the file
/// * `format` - Format of the file
/// * `weighted` - Whether to include edge weights
///
/// # Returns
/// * `Ok(())` - If the graph was written successfully
/// * `Err(GraphError)` - If there was an error writing the file
pub fn write_graph<N, E, Ix, P>(
    graph: &Graph<N, E, Ix>,
    path: P,
    format: GraphFormat,
    weighted: bool,
) -> Result<()>
where
    N: Node + std::fmt::Debug + std::fmt::Display,
    E: EdgeWeight + std::marker::Copy + std::fmt::Debug + std::default::Default + std::fmt::Display,
    Ix: petgraph::graph::IndexType,
    P: AsRef<Path>,
{
    match format {
        GraphFormat::EdgeList => write_edge_list_format(graph, path, weighted),
        GraphFormat::AdjacencyList => write_adjacency_list_format(graph, path, weighted),
        GraphFormat::Dot => write_dot_format(graph, path, weighted),
        GraphFormat::Json => write_json_format(graph, path, weighted),
        _ => Err(GraphError::Other(format!(
            "Format {:?} not implemented yet",
            format
        ))),
    }
}

/// Writes a directed graph to a file - stubbed implementation
///
/// # Arguments
/// * `graph` - The directed graph to write
/// * `path` - Path to the file
/// * `format` - Format of the file
/// * `weighted` - Whether to include edge weights
///
/// # Returns
/// * `Ok(())` - If the graph was written successfully
/// * `Err(GraphError)` - If there was an error writing the file
pub fn write_digraph<N, E, Ix, P>(
    _graph: &DiGraph<N, E, Ix>,
    _path: P,
    _format: GraphFormat,
    _weighted: bool,
) -> Result<()>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight + std::marker::Copy + std::fmt::Debug + std::default::Default,
    Ix: petgraph::graph::IndexType,
    P: AsRef<Path>,
{
    Err(GraphError::Other(
        "Function not implemented yet".to_string(),
    ))
}

/// Helper function to read edge list format
fn read_edge_list_format<N, E, P>(path: P, weighted: bool) -> Result<Graph<N, E>>
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

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 {
            continue; // Skip malformed lines
        }

        // Parse source and target nodes
        let source_str = parts[0];
        let target_str = parts[1];

        let source_data = N::from_str(source_str)
            .map_err(|_| GraphError::Other(format!("Cannot parse source node: {}", source_str)))?;
        let target_data = N::from_str(target_str)
            .map_err(|_| GraphError::Other(format!("Cannot parse target node: {}", target_str)))?;

        // Parse edge weight if needed
        let edge_weight = if weighted && parts.len() > 2 {
            E::from_str(parts[2])
                .map_err(|_| GraphError::Other(format!("Cannot parse edge weight: {}", parts[2])))?
        } else {
            E::default()
        };

        // Add edge (this will automatically add nodes if they don't exist)
        graph.add_edge(source_data, target_data, edge_weight)?;
    }

    Ok(graph)
}

/// Helper function to write edge list format
fn write_edge_list_format<N, E, Ix, P>(
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

    // Write edges
    for edge in graph.edges() {
        if weighted {
            writeln!(file, "{} {} {}", edge.source, edge.target, edge.weight)
                .map_err(|e| GraphError::Other(format!("Error writing line: {}", e)))?;
        } else {
            writeln!(file, "{} {}", edge.source, edge.target)
                .map_err(|e| GraphError::Other(format!("Error writing line: {}", e)))?;
        }
    }

    Ok(())
}

/// Helper function to read DOT format (Graphviz)
fn read_dot_format<N, E, P>(path: P, weighted: bool) -> Result<Graph<N, E>>
where
    N: Node + std::fmt::Debug + FromStr + Clone,
    E: EdgeWeight + std::marker::Copy + std::fmt::Debug + std::default::Default + FromStr,
    P: AsRef<Path>,
{
    let file =
        File::open(path).map_err(|e| GraphError::Other(format!("Cannot open file: {}", e)))?;
    let reader = BufReader::new(file);
    let mut graph = Graph::new();
    let mut in_graph = false;

    for line_result in reader.lines() {
        let line =
            line_result.map_err(|e| GraphError::Other(format!("Error reading line: {}", e)))?;
        let line = line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with("//") {
            continue;
        }

        // Look for graph declaration
        if line.contains("graph") && line.contains("{") {
            in_graph = true;
            continue;
        }

        // End of graph
        if line.contains("}") {
            break;
        }

        if in_graph {
            // Parse edge declarations (node1 -- node2 [label="weight"])
            if line.contains("--") {
                let parts: Vec<&str> = line.split("--").collect();
                if parts.len() >= 2 {
                    let source_str = parts[0].trim();
                    let rest = parts[1].trim();

                    // Extract target node (before any attributes)
                    let target_str = if rest.contains("[") {
                        rest.split("[").next().unwrap_or(rest).trim()
                    } else {
                        rest.split(";").next().unwrap_or(rest).trim()
                    };

                    let source_data = N::from_str(source_str).map_err(|_| {
                        GraphError::Other(format!("Cannot parse source node: {}", source_str))
                    })?;
                    let target_data = N::from_str(target_str).map_err(|_| {
                        GraphError::Other(format!("Cannot parse target node: {}", target_str))
                    })?;

                    // Parse weight from label attribute if present and weighted
                    let edge_weight = if weighted && line.contains("label=") {
                        // Extract label value
                        if let Some(label_start) = line.find("label=\"") {
                            let label_content = &line[label_start + 7..];
                            if let Some(label_end) = label_content.find("\"") {
                                let weight_str = &label_content[..label_end];
                                E::from_str(weight_str).map_err(|_| {
                                    GraphError::Other(format!(
                                        "Cannot parse edge weight: {}",
                                        weight_str
                                    ))
                                })?
                            } else {
                                E::default()
                            }
                        } else {
                            E::default()
                        }
                    } else {
                        E::default()
                    };

                    graph.add_edge(source_data, target_data, edge_weight)?;
                }
            }
        }
    }

    Ok(graph)
}

/// Helper function to write DOT format (Graphviz)
fn write_dot_format<N, E, Ix, P>(graph: &Graph<N, E, Ix>, path: P, weighted: bool) -> Result<()>
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

    // Write DOT header
    writeln!(file, "graph G {{")
        .map_err(|e| GraphError::Other(format!("Error writing header: {}", e)))?;

    // Write edges
    for edge in graph.edges() {
        if weighted {
            writeln!(
                file,
                "  {} -- {} [label=\"{}\"];",
                edge.source, edge.target, edge.weight
            )
            .map_err(|e| GraphError::Other(format!("Error writing edge: {}", e)))?;
        } else {
            writeln!(file, "  {} -- {};", edge.source, edge.target)
                .map_err(|e| GraphError::Other(format!("Error writing edge: {}", e)))?;
        }
    }

    // Write DOT footer
    writeln!(file, "}}").map_err(|e| GraphError::Other(format!("Error writing footer: {}", e)))?;

    Ok(())
}

/// Helper function to read adjacency list format
fn read_adjacency_list_format<N, E, P>(path: P, weighted: bool) -> Result<Graph<N, E>>
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

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        // First part is the source node
        let source_str = parts[0].trim_end_matches(':');
        let source_data = N::from_str(source_str)
            .map_err(|_| GraphError::Other(format!("Cannot parse source node: {}", source_str)))?;

        // Remaining parts are target nodes (and weights if weighted)
        let mut i = 1;
        while i < parts.len() {
            let target_str = parts[i];
            let target_data = N::from_str(target_str).map_err(|_| {
                GraphError::Other(format!("Cannot parse target node: {}", target_str))
            })?;

            let edge_weight = if weighted && i + 1 < parts.len() {
                // Next part should be the weight
                E::from_str(parts[i + 1]).map_err(|_| {
                    GraphError::Other(format!("Cannot parse edge weight: {}", parts[i + 1]))
                })?
            } else {
                E::default()
            };

            graph.add_edge(source_data.clone(), target_data, edge_weight)?;

            i += if weighted { 2 } else { 1 };
        }
    }

    Ok(graph)
}

/// Helper function to write adjacency list format
fn write_adjacency_list_format<N, E, Ix, P>(
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

    // Build adjacency lists
    let mut adjacency: std::collections::HashMap<N, Vec<(N, E)>> = std::collections::HashMap::new();

    for edge in graph.edges() {
        adjacency
            .entry(edge.source.clone())
            .or_default()
            .push((edge.target.clone(), edge.weight));
        // For undirected graphs, add the reverse edge too
        adjacency
            .entry(edge.target.clone())
            .or_default()
            .push((edge.source.clone(), edge.weight));
    }

    // Write adjacency lists
    for (source, neighbors) in adjacency {
        write!(file, "{}:", source)
            .map_err(|e| GraphError::Other(format!("Error writing source: {}", e)))?;

        for (target, weight) in neighbors {
            if weighted {
                write!(file, " {} {}", target, weight)
                    .map_err(|e| GraphError::Other(format!("Error writing neighbor: {}", e)))?;
            } else {
                write!(file, " {}", target)
                    .map_err(|e| GraphError::Other(format!("Error writing neighbor: {}", e)))?;
            }
        }

        writeln!(file).map_err(|e| GraphError::Other(format!("Error writing newline: {}", e)))?;
    }

    Ok(())
}

/// Helper function to read JSON format
fn read_json_format<N, E, P>(path: P, _weighted: bool) -> Result<Graph<N, E>>
where
    N: Node + std::fmt::Debug + FromStr + Clone,
    E: EdgeWeight + std::marker::Copy + std::fmt::Debug + std::default::Default + FromStr,
    P: AsRef<Path>,
{
    use serde_json::Value;

    let content = std::fs::read_to_string(path)
        .map_err(|e| GraphError::Other(format!("Cannot read file: {}", e)))?;

    let json: Value = serde_json::from_str(&content)
        .map_err(|e| GraphError::Other(format!("Cannot parse JSON: {}", e)))?;

    let mut graph = Graph::new();

    // Parse nodes if present
    if let Some(nodes) = json.get("nodes").and_then(|n| n.as_array()) {
        for node_value in nodes {
            if let Some(node_str) = node_value.as_str() {
                let node_data = N::from_str(node_str)
                    .map_err(|_| GraphError::Other(format!("Cannot parse node: {}", node_str)))?;
                graph.add_node(node_data);
            }
        }
    }

    // Parse edges
    if let Some(edges) = json.get("edges").and_then(|e| e.as_array()) {
        for edge_value in edges {
            if let Some(edge_obj) = edge_value.as_object() {
                let source_str = edge_obj
                    .get("source")
                    .and_then(|s| s.as_str())
                    .ok_or_else(|| GraphError::Other("Missing source in edge".to_string()))?;

                let target_str = edge_obj
                    .get("target")
                    .and_then(|t| t.as_str())
                    .ok_or_else(|| GraphError::Other("Missing target in edge".to_string()))?;

                let source_data = N::from_str(source_str).map_err(|_| {
                    GraphError::Other(format!("Cannot parse source node: {}", source_str))
                })?;
                let target_data = N::from_str(target_str).map_err(|_| {
                    GraphError::Other(format!("Cannot parse target node: {}", target_str))
                })?;

                let edge_weight = if let Some(weight_value) = edge_obj.get("weight") {
                    if let Some(weight_str) = weight_value.as_str() {
                        E::from_str(weight_str).map_err(|_| {
                            GraphError::Other(format!("Cannot parse edge weight: {}", weight_str))
                        })?
                    } else if let Some(weight_num) = weight_value.as_f64() {
                        // Try to parse as string representation of the number
                        E::from_str(&weight_num.to_string()).map_err(|_| {
                            GraphError::Other(format!("Cannot parse edge weight: {}", weight_num))
                        })?
                    } else {
                        E::default()
                    }
                } else {
                    E::default()
                };

                graph.add_edge(source_data, target_data, edge_weight)?;
            }
        }
    }

    Ok(graph)
}

/// Helper function to write JSON format
fn write_json_format<N, E, Ix, P>(graph: &Graph<N, E, Ix>, path: P, _weighted: bool) -> Result<()>
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

    // Write a simple JSON representation
    writeln!(file, "{{")
        .map_err(|e| GraphError::Other(format!("Error writing JSON start: {}", e)))?;

    // Write nodes
    writeln!(file, "  \"nodes\": [")
        .map_err(|e| GraphError::Other(format!("Error writing nodes array: {}", e)))?;

    let nodes = graph.nodes();
    for (i, node) in nodes.iter().enumerate() {
        let comma = if i < nodes.len() - 1 { "," } else { "" };
        writeln!(file, "    \"{}\"{}", node, comma)
            .map_err(|e| GraphError::Other(format!("Error writing node: {}", e)))?;
    }

    writeln!(file, "  ],")
        .map_err(|e| GraphError::Other(format!("Error writing nodes end: {}", e)))?;

    // Write edges
    writeln!(file, "  \"edges\": [")
        .map_err(|e| GraphError::Other(format!("Error writing edges array: {}", e)))?;

    let edges = graph.edges();
    for (i, edge) in edges.iter().enumerate() {
        let comma = if i < edges.len() - 1 { "," } else { "" };
        writeln!(
            file,
            "    {{\"source\": \"{}\", \"target\": \"{}\", \"weight\": \"{}\"}}{}",
            edge.source, edge.target, edge.weight, comma
        )
        .map_err(|e| GraphError::Other(format!("Error writing edge: {}", e)))?;
    }

    writeln!(file, "  ]")
        .map_err(|e| GraphError::Other(format!("Error writing edges end: {}", e)))?;

    writeln!(file, "}}")
        .map_err(|e| GraphError::Other(format!("Error writing JSON end: {}", e)))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::NamedTempFile;

    #[test]
    fn test_edge_list_format() {
        let mut graph: Graph<&str, f64> = Graph::new();
        graph.add_edge("A", "B", 1.0).unwrap();
        graph.add_edge("B", "C", 2.0).unwrap();
        graph.add_edge("C", "A", 3.0).unwrap();

        // Test writing
        let temp_file = NamedTempFile::new().unwrap();
        write_graph(&graph, temp_file.path(), GraphFormat::EdgeList, true).unwrap();

        // Test reading
        let read_graph: Graph<String, f64> =
            read_graph(temp_file.path(), GraphFormat::EdgeList, true, false).unwrap();
        assert_eq!(read_graph.node_count(), 3);
        assert_eq!(read_graph.edge_count(), 3);
    }

    #[test]
    fn test_dot_format() {
        let mut graph: Graph<&str, f64> = Graph::new();
        graph.add_edge("A", "B", 1.5).unwrap();
        graph.add_edge("B", "C", 2.5).unwrap();

        // Test writing
        let temp_file = NamedTempFile::new().unwrap();
        write_graph(&graph, temp_file.path(), GraphFormat::Dot, true).unwrap();

        // Verify the content
        let content = fs::read_to_string(temp_file.path()).unwrap();
        assert!(content.contains("graph G {"));
        assert!(content.contains("A -- B [label=\"1.5\"];"));
        assert!(content.contains("B -- C [label=\"2.5\"];"));
        assert!(content.contains("}"));
    }

    #[test]
    fn test_adjacency_list_format() {
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_edge(1, 2, 1.0).unwrap();
        graph.add_edge(2, 3, 2.0).unwrap();

        // Test writing
        let temp_file = NamedTempFile::new().unwrap();
        write_graph(&graph, temp_file.path(), GraphFormat::AdjacencyList, true).unwrap();

        // Test reading
        let read_graph: Graph<i32, f64> =
            read_graph(temp_file.path(), GraphFormat::AdjacencyList, true, false).unwrap();
        assert_eq!(read_graph.node_count(), 3);
        assert!(read_graph.has_edge(&1, &2));
        assert!(read_graph.has_edge(&2, &3));
    }

    #[test]
    fn test_json_format_write() {
        let mut graph: Graph<&str, f64> = Graph::new();
        graph.add_edge("X", "Y", 1.0).unwrap();
        graph.add_edge("Y", "Z", 2.0).unwrap();

        // Test writing
        let temp_file = NamedTempFile::new().unwrap();
        write_graph(&graph, temp_file.path(), GraphFormat::Json, true).unwrap();

        // Verify the content contains JSON structure
        let content = fs::read_to_string(temp_file.path()).unwrap();
        assert!(content.contains("\"nodes\":"));
        assert!(content.contains("\"edges\":"));
        assert!(content.contains("\"source\":"));
        assert!(content.contains("\"target\":"));
        assert!(content.contains("\"weight\":"));
    }

    #[test]
    fn test_json_format_round_trip() {
        let mut original_graph: Graph<String, f64> = Graph::new();
        original_graph
            .add_edge("A".to_string(), "B".to_string(), 1.5)
            .unwrap();
        original_graph
            .add_edge("B".to_string(), "C".to_string(), 2.5)
            .unwrap();
        original_graph
            .add_edge("C".to_string(), "A".to_string(), 3.5)
            .unwrap();

        // Write to JSON
        let temp_file = NamedTempFile::new().unwrap();
        write_graph(&original_graph, temp_file.path(), GraphFormat::Json, true).unwrap();

        // Read back from JSON
        let read_graph: Graph<String, f64> =
            read_graph(temp_file.path(), GraphFormat::Json, true, false).unwrap();

        // Verify the graph structure is preserved
        assert_eq!(read_graph.node_count(), 3);
        assert_eq!(read_graph.edge_count(), 3);
        assert!(read_graph.has_edge(&"A".to_string(), &"B".to_string()));
        assert!(read_graph.has_edge(&"B".to_string(), &"C".to_string()));
        assert!(read_graph.has_edge(&"C".to_string(), &"A".to_string()));
    }

    #[test]
    fn test_unweighted_formats() {
        let mut graph: Graph<&str, i32> = Graph::new();
        graph.add_edge("P", "Q", 1).unwrap();
        graph.add_edge("Q", "R", 1).unwrap();

        // Test DOT format without weights
        let temp_file_dot = NamedTempFile::new().unwrap();
        write_graph(&graph, temp_file_dot.path(), GraphFormat::Dot, false).unwrap();

        let content = fs::read_to_string(temp_file_dot.path()).unwrap();
        assert!(content.contains("P -- Q;"));
        assert!(!content.contains("label="));

        // Test adjacency list format without weights
        let temp_file_adj = NamedTempFile::new().unwrap();
        write_graph(
            &graph,
            temp_file_adj.path(),
            GraphFormat::AdjacencyList,
            false,
        )
        .unwrap();

        let content = fs::read_to_string(temp_file_adj.path()).unwrap();
        // For undirected graphs, we should see entries for each node
        assert!(content.contains("P:") && content.contains("Q") && content.contains("R"));
    }
}
