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
    /// GML format (Graph Modeling Language)
    Gml,
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
        GraphFormat::GraphML => read_graphml_format(path, weighted),
        GraphFormat::Gml => read_gml_format(path, weighted),
        GraphFormat::MatrixMarket => read_matrix_market_format(path, weighted),
    }
}

/// Reads a directed graph from a file
///
/// # Arguments
/// * `path` - Path to the file
/// * `format` - Format of the file
/// * `weighted` - Whether the graph has edge weights
///
/// # Returns
/// * `Ok(DiGraph)` - The directed graph read from the file
/// * `Err(GraphError)` - If there was an error reading the file
pub fn read_digraph<N, E, P>(path: P, format: GraphFormat, weighted: bool) -> Result<DiGraph<N, E>>
where
    N: Node + std::fmt::Debug + FromStr + Clone,
    E: EdgeWeight + std::marker::Copy + std::fmt::Debug + std::default::Default + FromStr,
    P: AsRef<Path>,
{
    match format {
        GraphFormat::EdgeList => read_edge_list_format_digraph(path, weighted),
        GraphFormat::AdjacencyList => read_adjacency_list_format_digraph(path, weighted),
        GraphFormat::Dot => read_dot_format_digraph(path, weighted),
        GraphFormat::Json => read_json_format_digraph(path, weighted),
        GraphFormat::GraphML => read_graphml_format_digraph(path, weighted),
        GraphFormat::Gml => read_gml_format_digraph(path, weighted),
        GraphFormat::MatrixMarket => read_matrix_market_format_digraph(path, weighted),
    }
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
        GraphFormat::GraphML => write_graphml_format(graph, path, weighted),
        GraphFormat::Gml => write_gml_format(graph, path, weighted),
        GraphFormat::MatrixMarket => write_matrix_market_format(graph, path, weighted),
    }
}

/// Writes a directed graph to a file
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
    graph: &DiGraph<N, E, Ix>,
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
        GraphFormat::EdgeList => write_edge_list_format_digraph(graph, path, weighted),
        GraphFormat::AdjacencyList => write_adjacency_list_format_digraph(graph, path, weighted),
        GraphFormat::Dot => write_dot_format_digraph(graph, path, weighted),
        GraphFormat::Json => write_json_format_digraph(graph, path, weighted),
        GraphFormat::GraphML => write_graphml_format_digraph(graph, path, weighted),
        GraphFormat::Gml => write_gml_format_digraph(graph, path, weighted),
        GraphFormat::MatrixMarket => write_matrix_market_format_digraph(graph, path, weighted),
    }
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

/// Helper function to read GraphML format
fn read_graphml_format<N, E, P>(path: P, _weighted: bool) -> Result<Graph<N, E>>
where
    N: Node + std::fmt::Debug + FromStr + Clone,
    E: EdgeWeight + std::marker::Copy + std::fmt::Debug + std::default::Default + FromStr,
    P: AsRef<Path>,
{
    let content = std::fs::read_to_string(path)
        .map_err(|e| GraphError::Other(format!("Cannot read file: {}", e)))?;

    let mut graph = Graph::new();
    let mut in_graph = false;

    // Simple GraphML parser - handles basic structure
    for line in content.lines() {
        let line = line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with("<!--") {
            continue;
        }

        // Look for graph opening tag
        if line.contains("<graph") {
            in_graph = true;
            continue;
        }

        // End of graph
        if line.contains("</graph>") {
            break;
        }

        if in_graph {
            // Parse node declarations
            if line.contains("<node") && line.contains("id=") {
                if let Some(id_start) = line.find("id=\"") {
                    let id_content = &line[id_start + 4..];
                    if let Some(id_end) = id_content.find('"') {
                        let node_id = &id_content[..id_end];
                        let node_data = N::from_str(node_id).map_err(|_| {
                            GraphError::Other(format!("Cannot parse node: {}", node_id))
                        })?;
                        graph.add_node(node_data);
                    }
                }
            }

            // Parse edge declarations
            if line.contains("<edge") && line.contains("source=") && line.contains("target=") {
                let source_str = if let Some(source_start) = line.find("source=\"") {
                    let source_content = &line[source_start + 8..];
                    source_content
                        .find('"')
                        .map(|source_end| &source_content[..source_end])
                } else {
                    None
                };

                let target_str = if let Some(target_start) = line.find("target=\"") {
                    let target_content = &line[target_start + 8..];
                    target_content
                        .find('"')
                        .map(|target_end| &target_content[..target_end])
                } else {
                    None
                };

                if let (Some(source_id), Some(target_id)) = (source_str, target_str) {
                    let source_data = N::from_str(source_id).map_err(|_| {
                        GraphError::Other(format!("Cannot parse source node: {}", source_id))
                    })?;
                    let target_data = N::from_str(target_id).map_err(|_| {
                        GraphError::Other(format!("Cannot parse target node: {}", target_id))
                    })?;

                    // For now, use default weight - could be extended to parse weight attributes
                    let edge_weight = E::default();
                    graph.add_edge(source_data, target_data, edge_weight)?;
                }
            }
        }
    }

    Ok(graph)
}

/// Helper function to write GraphML format
fn write_graphml_format<N, E, Ix, P>(graph: &Graph<N, E, Ix>, path: P, weighted: bool) -> Result<()>
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

    // Write GraphML header
    writeln!(file, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>")
        .map_err(|e| GraphError::Other(format!("Error writing header: {}", e)))?;
    writeln!(
        file,
        "<graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\""
    )
    .map_err(|e| GraphError::Other(format!("Error writing graphml tag: {}", e)))?;
    writeln!(
        file,
        "         xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\""
    )
    .map_err(|e| GraphError::Other(format!("Error writing schema: {}", e)))?;
    writeln!(file, "         xsi:schemaLocation=\"http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd\">")
        .map_err(|e| GraphError::Other(format!("Error writing schema location: {}", e)))?;

    // Add edge weight key if weighted
    if weighted {
        writeln!(
            file,
            "  <key id=\"weight\" for=\"edge\" attr.name=\"weight\" attr.type=\"double\"/>"
        )
        .map_err(|e| GraphError::Other(format!("Error writing weight key: {}", e)))?;
    }

    // Write graph opening tag (undirected by default)
    writeln!(file, "  <graph id=\"G\" edgedefault=\"undirected\">")
        .map_err(|e| GraphError::Other(format!("Error writing graph tag: {}", e)))?;

    // Write nodes
    for node in graph.nodes() {
        writeln!(file, "    <node id=\"{}\"/>", node)
            .map_err(|e| GraphError::Other(format!("Error writing node: {}", e)))?;
    }

    // Write edges
    for edge in graph.edges() {
        if weighted {
            writeln!(
                file,
                "    <edge source=\"{}\" target=\"{}\">",
                edge.source, edge.target
            )
            .map_err(|e| GraphError::Other(format!("Error writing edge: {}", e)))?;
            writeln!(file, "      <data key=\"weight\">{}</data>", edge.weight)
                .map_err(|e| GraphError::Other(format!("Error writing weight: {}", e)))?;
            writeln!(file, "    </edge>")
                .map_err(|e| GraphError::Other(format!("Error writing edge end: {}", e)))?;
        } else {
            writeln!(
                file,
                "    <edge source=\"{}\" target=\"{}\"/>",
                edge.source, edge.target
            )
            .map_err(|e| GraphError::Other(format!("Error writing edge: {}", e)))?;
        }
    }

    // Write closing tags
    writeln!(file, "  </graph>")
        .map_err(|e| GraphError::Other(format!("Error writing graph end: {}", e)))?;
    writeln!(file, "</graphml>")
        .map_err(|e| GraphError::Other(format!("Error writing graphml end: {}", e)))?;

    Ok(())
}

/// Helper function to read Matrix Market format
fn read_matrix_market_format<N, E, P>(path: P, weighted: bool) -> Result<Graph<N, E>>
where
    N: Node + std::fmt::Debug + FromStr + Clone,
    E: EdgeWeight + std::marker::Copy + std::fmt::Debug + std::default::Default + FromStr,
    P: AsRef<Path>,
{
    let file =
        File::open(path).map_err(|e| GraphError::Other(format!("Cannot open file: {}", e)))?;
    let reader = BufReader::new(file);
    let mut graph = Graph::new();
    let mut header_parsed = false;

    for line_result in reader.lines() {
        let line =
            line_result.map_err(|e| GraphError::Other(format!("Error reading line: {}", e)))?;
        let line = line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('%') {
            continue;
        }

        if !header_parsed {
            // Skip the matrix dimension line (first non-comment line)
            // Format: rows cols entries
            header_parsed = true;
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 {
            continue; // Skip malformed lines
        }

        // Matrix Market format uses 1-based indexing, convert to 0-based if numeric
        let source_str = parts[0];
        let target_str = parts[1];

        let source_data = N::from_str(source_str)
            .map_err(|_| GraphError::Other(format!("Cannot parse source node: {}", source_str)))?;
        let target_data = N::from_str(target_str)
            .map_err(|_| GraphError::Other(format!("Cannot parse target node: {}", target_str)))?;

        // Parse edge weight if available and weighted
        let edge_weight = if weighted && parts.len() > 2 {
            E::from_str(parts[2])
                .map_err(|_| GraphError::Other(format!("Cannot parse edge weight: {}", parts[2])))?
        } else {
            E::default()
        };

        graph.add_edge(source_data, target_data, edge_weight)?;
    }

    Ok(graph)
}

/// Helper function to write Matrix Market format
fn write_matrix_market_format<N, E, Ix, P>(
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

    // Write Matrix Market header
    writeln!(file, "%%MatrixMarket matrix coordinate real general")
        .map_err(|e| GraphError::Other(format!("Error writing header: {}", e)))?;
    writeln!(file, "%").map_err(|e| GraphError::Other(format!("Error writing comment: {}", e)))?;

    let nodes = graph.nodes();
    let edges = graph.edges();

    // Write matrix dimensions: rows cols entries
    writeln!(file, "{} {} {}", nodes.len(), nodes.len(), edges.len())
        .map_err(|e| GraphError::Other(format!("Error writing dimensions: {}", e)))?;

    // Create node to index mapping for consistent ordering
    let node_to_index: std::collections::HashMap<N, usize> = nodes
        .iter()
        .enumerate()
        .map(|(i, node)| ((*node).clone(), i + 1)) // Matrix Market uses 1-based indexing
        .collect();

    // Write edges
    for edge in edges {
        let source_idx = node_to_index
            .get(&edge.source)
            .ok_or_else(|| GraphError::Other("Source node not found in mapping".to_string()))?;
        let target_idx = node_to_index
            .get(&edge.target)
            .ok_or_else(|| GraphError::Other("Target node not found in mapping".to_string()))?;

        if weighted {
            writeln!(file, "{} {} {}", source_idx, target_idx, edge.weight)
                .map_err(|e| GraphError::Other(format!("Error writing edge: {}", e)))?;
        } else {
            writeln!(file, "{} {} 1.0", source_idx, target_idx)
                .map_err(|e| GraphError::Other(format!("Error writing edge: {}", e)))?;
        }
    }

    Ok(())
}

/// Helper function to read GML format
fn read_gml_format<N, E, P>(path: P, _weighted: bool) -> Result<Graph<N, E>>
where
    N: Node + std::fmt::Debug + FromStr + Clone,
    E: EdgeWeight + std::marker::Copy + std::fmt::Debug + std::default::Default + FromStr,
    P: AsRef<Path>,
{
    let content = std::fs::read_to_string(path)
        .map_err(|e| GraphError::Other(format!("Cannot read file: {}", e)))?;

    let mut graph = Graph::new();
    let lines = content.lines();
    let mut in_graph = false;
    let mut in_node = false;
    let mut in_edge = false;
    let mut current_node_id: Option<String> = None;
    let mut current_edge_source: Option<String> = None;
    let mut current_edge_target: Option<String> = None;
    let mut current_edge_value: Option<String> = None;

    for line in lines {
        let line = line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Look for graph opening
        if line.contains("graph") && line.contains("[") {
            in_graph = true;
            continue;
        }

        // End of graph
        if in_graph && line == "]" && !in_node && !in_edge {
            break;
        }

        if in_graph {
            // Handle node declarations
            if line.contains("node") && line.contains("[") {
                in_node = true;
                continue;
            }

            if in_node {
                if line == "]" {
                    // End of node - add it to graph
                    if let Some(id) = current_node_id.take() {
                        if let Ok(node_data) = N::from_str(&id) {
                            graph.add_node(node_data);
                        }
                    }
                    in_node = false;
                    continue;
                }

                // Parse node properties
                if line.starts_with("id") {
                    if let Some(value_start) = line.find("\"") {
                        let rest = &line[value_start + 1..];
                        if let Some(value_end) = rest.find("\"") {
                            current_node_id = Some(rest[..value_end].to_string());
                        }
                    } else if let Some(value) = line.split_whitespace().nth(1) {
                        current_node_id = Some(value.to_string());
                    }
                }
            }

            // Handle edge declarations
            if line.contains("edge") && line.contains("[") {
                in_edge = true;
                continue;
            }

            if in_edge {
                if line == "]" {
                    // End of edge - add it to graph
                    if let (Some(source), Some(target)) =
                        (current_edge_source.take(), current_edge_target.take())
                    {
                        if let (Ok(source_data), Ok(target_data)) =
                            (N::from_str(&source), N::from_str(&target))
                        {
                            let edge_weight = if let Some(value_str) = current_edge_value.take() {
                                E::from_str(&value_str).unwrap_or_else(|_| E::default())
                            } else {
                                E::default()
                            };
                            let _ = graph.add_edge(source_data, target_data, edge_weight);
                        }
                    }
                    in_edge = false;
                    continue;
                }

                // Parse edge properties
                if line.starts_with("source") {
                    if let Some(value_start) = line.find("\"") {
                        let rest = &line[value_start + 1..];
                        if let Some(value_end) = rest.find("\"") {
                            current_edge_source = Some(rest[..value_end].to_string());
                        }
                    } else if let Some(value) = line.split_whitespace().nth(1) {
                        current_edge_source = Some(value.to_string());
                    }
                }

                if line.starts_with("target") {
                    if let Some(value_start) = line.find("\"") {
                        let rest = &line[value_start + 1..];
                        if let Some(value_end) = rest.find("\"") {
                            current_edge_target = Some(rest[..value_end].to_string());
                        }
                    } else if let Some(value) = line.split_whitespace().nth(1) {
                        current_edge_target = Some(value.to_string());
                    }
                }

                if line.starts_with("value") {
                    if let Some(value_start) = line.find("\"") {
                        let rest = &line[value_start + 1..];
                        if let Some(value_end) = rest.find("\"") {
                            current_edge_value = Some(rest[..value_end].to_string());
                        }
                    } else if let Some(value) = line.split_whitespace().nth(1) {
                        current_edge_value = Some(value.to_string());
                    }
                }
            }
        }
    }

    Ok(graph)
}

/// Helper function to write GML format
fn write_gml_format<N, E, Ix, P>(graph: &Graph<N, E, Ix>, path: P, weighted: bool) -> Result<()>
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

    // Write GML header
    writeln!(file, "graph [")
        .map_err(|e| GraphError::Other(format!("Error writing header: {}", e)))?;

    // Write nodes
    for node in graph.nodes() {
        writeln!(file, "  node [")
            .map_err(|e| GraphError::Other(format!("Error writing node start: {}", e)))?;
        writeln!(file, "    id \"{}\"", node)
            .map_err(|e| GraphError::Other(format!("Error writing node id: {}", e)))?;
        writeln!(file, "  ]")
            .map_err(|e| GraphError::Other(format!("Error writing node end: {}", e)))?;
    }

    // Write edges
    for edge in graph.edges() {
        writeln!(file, "  edge [")
            .map_err(|e| GraphError::Other(format!("Error writing edge start: {}", e)))?;
        writeln!(file, "    source \"{}\"", edge.source)
            .map_err(|e| GraphError::Other(format!("Error writing edge source: {}", e)))?;
        writeln!(file, "    target \"{}\"", edge.target)
            .map_err(|e| GraphError::Other(format!("Error writing edge target: {}", e)))?;

        if weighted {
            writeln!(file, "    value \"{}\"", edge.weight)
                .map_err(|e| GraphError::Other(format!("Error writing edge weight: {}", e)))?;
        }

        writeln!(file, "  ]")
            .map_err(|e| GraphError::Other(format!("Error writing edge end: {}", e)))?;
    }

    // Write GML footer
    writeln!(file, "]").map_err(|e| GraphError::Other(format!("Error writing footer: {}", e)))?;

    Ok(())
}

/// Helper function to read GML format for directed graphs
fn read_gml_format_digraph<N, E, P>(path: P, _weighted: bool) -> Result<DiGraph<N, E>>
where
    N: Node + std::fmt::Debug + FromStr + Clone,
    E: EdgeWeight + std::marker::Copy + std::fmt::Debug + std::default::Default + FromStr,
    P: AsRef<Path>,
{
    let content = std::fs::read_to_string(path)
        .map_err(|e| GraphError::Other(format!("Cannot read file: {}", e)))?;

    let mut graph = DiGraph::new();
    let lines = content.lines();
    let mut in_graph = false;
    let mut in_node = false;
    let mut in_edge = false;
    let mut current_node_id: Option<String> = None;
    let mut current_edge_source: Option<String> = None;
    let mut current_edge_target: Option<String> = None;
    let mut current_edge_value: Option<String> = None;

    for line in lines {
        let line = line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Look for graph opening
        if line.contains("graph") && line.contains("[") {
            in_graph = true;
            continue;
        }

        // End of graph
        if in_graph && line == "]" && !in_node && !in_edge {
            break;
        }

        if in_graph {
            // Handle node declarations
            if line.contains("node") && line.contains("[") {
                in_node = true;
                continue;
            }

            if in_node {
                if line == "]" {
                    // End of node - add it to graph
                    if let Some(id) = current_node_id.take() {
                        if let Ok(node_data) = N::from_str(&id) {
                            graph.add_node(node_data);
                        }
                    }
                    in_node = false;
                    continue;
                }

                // Parse node properties
                if line.starts_with("id") {
                    if let Some(value_start) = line.find("\"") {
                        let rest = &line[value_start + 1..];
                        if let Some(value_end) = rest.find("\"") {
                            current_node_id = Some(rest[..value_end].to_string());
                        }
                    } else if let Some(value) = line.split_whitespace().nth(1) {
                        current_node_id = Some(value.to_string());
                    }
                }
            }

            // Handle edge declarations
            if line.contains("edge") && line.contains("[") {
                in_edge = true;
                continue;
            }

            if in_edge {
                if line == "]" {
                    // End of edge - add it to graph
                    if let (Some(source), Some(target)) =
                        (current_edge_source.take(), current_edge_target.take())
                    {
                        if let (Ok(source_data), Ok(target_data)) =
                            (N::from_str(&source), N::from_str(&target))
                        {
                            let edge_weight = if let Some(value_str) = current_edge_value.take() {
                                E::from_str(&value_str).unwrap_or_else(|_| E::default())
                            } else {
                                E::default()
                            };
                            let _ = graph.add_edge(source_data, target_data, edge_weight);
                        }
                    }
                    in_edge = false;
                    continue;
                }

                // Parse edge properties
                if line.starts_with("source") {
                    if let Some(value_start) = line.find("\"") {
                        let rest = &line[value_start + 1..];
                        if let Some(value_end) = rest.find("\"") {
                            current_edge_source = Some(rest[..value_end].to_string());
                        }
                    } else if let Some(value) = line.split_whitespace().nth(1) {
                        current_edge_source = Some(value.to_string());
                    }
                }

                if line.starts_with("target") {
                    if let Some(value_start) = line.find("\"") {
                        let rest = &line[value_start + 1..];
                        if let Some(value_end) = rest.find("\"") {
                            current_edge_target = Some(rest[..value_end].to_string());
                        }
                    } else if let Some(value) = line.split_whitespace().nth(1) {
                        current_edge_target = Some(value.to_string());
                    }
                }

                if line.starts_with("value") {
                    if let Some(value_start) = line.find("\"") {
                        let rest = &line[value_start + 1..];
                        if let Some(value_end) = rest.find("\"") {
                            current_edge_value = Some(rest[..value_end].to_string());
                        }
                    } else if let Some(value) = line.split_whitespace().nth(1) {
                        current_edge_value = Some(value.to_string());
                    }
                }
            }
        }
    }

    Ok(graph)
}

/// Helper function to write GML format for directed graphs
fn write_gml_format_digraph<N, E, Ix, P>(
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

    // Write GML header
    writeln!(file, "graph [")
        .map_err(|e| GraphError::Other(format!("Error writing header: {}", e)))?;
    writeln!(file, "  directed 1")
        .map_err(|e| GraphError::Other(format!("Error writing directed flag: {}", e)))?;

    // Write nodes
    for node in graph.nodes() {
        writeln!(file, "  node [")
            .map_err(|e| GraphError::Other(format!("Error writing node start: {}", e)))?;
        writeln!(file, "    id \"{}\"", node)
            .map_err(|e| GraphError::Other(format!("Error writing node id: {}", e)))?;
        writeln!(file, "  ]")
            .map_err(|e| GraphError::Other(format!("Error writing node end: {}", e)))?;
    }

    // Write edges
    for edge in graph.edges() {
        writeln!(file, "  edge [")
            .map_err(|e| GraphError::Other(format!("Error writing edge start: {}", e)))?;
        writeln!(file, "    source \"{}\"", edge.source)
            .map_err(|e| GraphError::Other(format!("Error writing edge source: {}", e)))?;
        writeln!(file, "    target \"{}\"", edge.target)
            .map_err(|e| GraphError::Other(format!("Error writing edge target: {}", e)))?;

        if weighted {
            writeln!(file, "    value \"{}\"", edge.weight)
                .map_err(|e| GraphError::Other(format!("Error writing edge weight: {}", e)))?;
        }

        writeln!(file, "  ]")
            .map_err(|e| GraphError::Other(format!("Error writing edge end: {}", e)))?;
    }

    // Write GML footer
    writeln!(file, "]").map_err(|e| GraphError::Other(format!("Error writing footer: {}", e)))?;

    Ok(())
}

// Directed graph helper functions

/// Helper function to read edge list format for directed graphs
fn read_edge_list_format_digraph<N, E, P>(path: P, weighted: bool) -> Result<DiGraph<N, E>>
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

/// Helper function to write edge list format for directed graphs
fn write_edge_list_format_digraph<N, E, Ix, P>(
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

/// Helper function to read adjacency list format for directed graphs
fn read_adjacency_list_format_digraph<N, E, P>(path: P, weighted: bool) -> Result<DiGraph<N, E>>
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

/// Helper function to write adjacency list format for directed graphs
fn write_adjacency_list_format_digraph<N, E, Ix, P>(
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

    // Build adjacency lists for directed graph (only outgoing edges)
    let mut adjacency: std::collections::HashMap<N, Vec<(N, E)>> = std::collections::HashMap::new();

    for edge in graph.edges() {
        adjacency
            .entry(edge.source.clone())
            .or_default()
            .push((edge.target.clone(), edge.weight));
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

/// Helper function to read DOT format for directed graphs
fn read_dot_format_digraph<N, E, P>(path: P, weighted: bool) -> Result<DiGraph<N, E>>
where
    N: Node + std::fmt::Debug + FromStr + Clone,
    E: EdgeWeight + std::marker::Copy + std::fmt::Debug + std::default::Default + FromStr,
    P: AsRef<Path>,
{
    let file =
        File::open(path).map_err(|e| GraphError::Other(format!("Cannot open file: {}", e)))?;
    let reader = BufReader::new(file);
    let mut graph = DiGraph::new();
    let mut in_graph = false;

    for line_result in reader.lines() {
        let line =
            line_result.map_err(|e| GraphError::Other(format!("Error reading line: {}", e)))?;
        let line = line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with("//") {
            continue;
        }

        // Look for digraph declaration
        if line.contains("digraph") && line.contains("{") {
            in_graph = true;
            continue;
        }

        // End of graph
        if line.contains("}") {
            break;
        }

        if in_graph {
            // Parse edge declarations (node1 -> node2 [label="weight"])
            if line.contains("->") {
                let parts: Vec<&str> = line.split("->").collect();
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

/// Helper function to write DOT format for directed graphs
fn write_dot_format_digraph<N, E, Ix, P>(
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

    // Write DOT header for directed graph
    writeln!(file, "digraph G {{")
        .map_err(|e| GraphError::Other(format!("Error writing header: {}", e)))?;

    // Write edges
    for edge in graph.edges() {
        if weighted {
            writeln!(
                file,
                "  {} -> {} [label=\"{}\"];",
                edge.source, edge.target, edge.weight
            )
            .map_err(|e| GraphError::Other(format!("Error writing edge: {}", e)))?;
        } else {
            writeln!(file, "  {} -> {};", edge.source, edge.target)
                .map_err(|e| GraphError::Other(format!("Error writing edge: {}", e)))?;
        }
    }

    // Write DOT footer
    writeln!(file, "}}").map_err(|e| GraphError::Other(format!("Error writing footer: {}", e)))?;

    Ok(())
}

/// Helper function to read JSON format for directed graphs
fn read_json_format_digraph<N, E, P>(path: P, _weighted: bool) -> Result<DiGraph<N, E>>
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

    let mut graph = DiGraph::new();

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

/// Helper function to write JSON format for directed graphs
fn write_json_format_digraph<N, E, Ix, P>(
    graph: &DiGraph<N, E, Ix>,
    path: P,
    _weighted: bool,
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

/// Helper function to read GraphML format for directed graphs
fn read_graphml_format_digraph<N, E, P>(path: P, _weighted: bool) -> Result<DiGraph<N, E>>
where
    N: Node + std::fmt::Debug + FromStr + Clone,
    E: EdgeWeight + std::marker::Copy + std::fmt::Debug + std::default::Default + FromStr,
    P: AsRef<Path>,
{
    let content = std::fs::read_to_string(path)
        .map_err(|e| GraphError::Other(format!("Cannot read file: {}", e)))?;

    let mut graph = DiGraph::new();
    let mut in_graph = false;

    // Simple GraphML parser - handles basic structure
    for line in content.lines() {
        let line = line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with("<!--") {
            continue;
        }

        // Look for graph opening tag (directed)
        if line.contains("<graph") {
            in_graph = true;
            continue;
        }

        // End of graph
        if line.contains("</graph>") {
            break;
        }

        if in_graph {
            // Parse node declarations
            if line.contains("<node") && line.contains("id=") {
                if let Some(id_start) = line.find("id=\"") {
                    let id_content = &line[id_start + 4..];
                    if let Some(id_end) = id_content.find('"') {
                        let node_id = &id_content[..id_end];
                        let node_data = N::from_str(node_id).map_err(|_| {
                            GraphError::Other(format!("Cannot parse node: {}", node_id))
                        })?;
                        graph.add_node(node_data);
                    }
                }
            }

            // Parse edge declarations
            if line.contains("<edge") && line.contains("source=") && line.contains("target=") {
                let source_str = if let Some(source_start) = line.find("source=\"") {
                    let source_content = &line[source_start + 8..];
                    source_content
                        .find('"')
                        .map(|source_end| &source_content[..source_end])
                } else {
                    None
                };

                let target_str = if let Some(target_start) = line.find("target=\"") {
                    let target_content = &line[target_start + 8..];
                    target_content
                        .find('"')
                        .map(|target_end| &target_content[..target_end])
                } else {
                    None
                };

                if let (Some(source_id), Some(target_id)) = (source_str, target_str) {
                    let source_data = N::from_str(source_id).map_err(|_| {
                        GraphError::Other(format!("Cannot parse source node: {}", source_id))
                    })?;
                    let target_data = N::from_str(target_id).map_err(|_| {
                        GraphError::Other(format!("Cannot parse target node: {}", target_id))
                    })?;

                    // For now, use default weight - could be extended to parse weight attributes
                    let edge_weight = E::default();
                    graph.add_edge(source_data, target_data, edge_weight)?;
                }
            }
        }
    }

    Ok(graph)
}

/// Helper function to write GraphML format for directed graphs
fn write_graphml_format_digraph<N, E, Ix, P>(
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

    // Write GraphML header
    writeln!(file, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>")
        .map_err(|e| GraphError::Other(format!("Error writing header: {}", e)))?;
    writeln!(
        file,
        "<graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\""
    )
    .map_err(|e| GraphError::Other(format!("Error writing graphml tag: {}", e)))?;
    writeln!(
        file,
        "         xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\""
    )
    .map_err(|e| GraphError::Other(format!("Error writing schema: {}", e)))?;
    writeln!(file, "         xsi:schemaLocation=\"http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd\">")
        .map_err(|e| GraphError::Other(format!("Error writing schema location: {}", e)))?;

    // Add edge weight key if weighted
    if weighted {
        writeln!(
            file,
            "  <key id=\"weight\" for=\"edge\" attr.name=\"weight\" attr.type=\"double\"/>"
        )
        .map_err(|e| GraphError::Other(format!("Error writing weight key: {}", e)))?;
    }

    // Write graph opening tag (directed)
    writeln!(file, "  <graph id=\"G\" edgedefault=\"directed\">")
        .map_err(|e| GraphError::Other(format!("Error writing graph tag: {}", e)))?;

    // Write nodes
    for node in graph.nodes() {
        writeln!(file, "    <node id=\"{}\"/>", node)
            .map_err(|e| GraphError::Other(format!("Error writing node: {}", e)))?;
    }

    // Write edges
    for edge in graph.edges() {
        if weighted {
            writeln!(
                file,
                "    <edge source=\"{}\" target=\"{}\">",
                edge.source, edge.target
            )
            .map_err(|e| GraphError::Other(format!("Error writing edge: {}", e)))?;
            writeln!(file, "      <data key=\"weight\">{}</data>", edge.weight)
                .map_err(|e| GraphError::Other(format!("Error writing weight: {}", e)))?;
            writeln!(file, "    </edge>")
                .map_err(|e| GraphError::Other(format!("Error writing edge end: {}", e)))?;
        } else {
            writeln!(
                file,
                "    <edge source=\"{}\" target=\"{}\"/>",
                edge.source, edge.target
            )
            .map_err(|e| GraphError::Other(format!("Error writing edge: {}", e)))?;
        }
    }

    // Write closing tags
    writeln!(file, "  </graph>")
        .map_err(|e| GraphError::Other(format!("Error writing graph end: {}", e)))?;
    writeln!(file, "</graphml>")
        .map_err(|e| GraphError::Other(format!("Error writing graphml end: {}", e)))?;

    Ok(())
}

/// Helper function to read Matrix Market format for directed graphs
fn read_matrix_market_format_digraph<N, E, P>(path: P, weighted: bool) -> Result<DiGraph<N, E>>
where
    N: Node + std::fmt::Debug + FromStr + Clone,
    E: EdgeWeight + std::marker::Copy + std::fmt::Debug + std::default::Default + FromStr,
    P: AsRef<Path>,
{
    let file =
        File::open(path).map_err(|e| GraphError::Other(format!("Cannot open file: {}", e)))?;
    let reader = BufReader::new(file);
    let mut graph = DiGraph::new();
    let mut header_parsed = false;

    for line_result in reader.lines() {
        let line =
            line_result.map_err(|e| GraphError::Other(format!("Error reading line: {}", e)))?;
        let line = line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('%') {
            continue;
        }

        if !header_parsed {
            // Skip the matrix dimension line (first non-comment line)
            // Format: rows cols entries
            header_parsed = true;
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 {
            continue; // Skip malformed lines
        }

        // Matrix Market format uses 1-based indexing, convert to 0-based if numeric
        let source_str = parts[0];
        let target_str = parts[1];

        let source_data = N::from_str(source_str)
            .map_err(|_| GraphError::Other(format!("Cannot parse source node: {}", source_str)))?;
        let target_data = N::from_str(target_str)
            .map_err(|_| GraphError::Other(format!("Cannot parse target node: {}", target_str)))?;

        // Parse edge weight if available and weighted
        let edge_weight = if weighted && parts.len() > 2 {
            E::from_str(parts[2])
                .map_err(|_| GraphError::Other(format!("Cannot parse edge weight: {}", parts[2])))?
        } else {
            E::default()
        };

        graph.add_edge(source_data, target_data, edge_weight)?;
    }

    Ok(graph)
}

/// Helper function to write Matrix Market format for directed graphs
fn write_matrix_market_format_digraph<N, E, Ix, P>(
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

    // Write Matrix Market header
    writeln!(file, "%%MatrixMarket matrix coordinate real general")
        .map_err(|e| GraphError::Other(format!("Error writing header: {}", e)))?;
    writeln!(file, "%").map_err(|e| GraphError::Other(format!("Error writing comment: {}", e)))?;

    let nodes = graph.nodes();
    let edges = graph.edges();

    // Write matrix dimensions: rows cols entries
    writeln!(file, "{} {} {}", nodes.len(), nodes.len(), edges.len())
        .map_err(|e| GraphError::Other(format!("Error writing dimensions: {}", e)))?;

    // Create node to index mapping for consistent ordering
    let node_to_index: std::collections::HashMap<N, usize> = nodes
        .iter()
        .enumerate()
        .map(|(i, node)| ((*node).clone(), i + 1)) // Matrix Market uses 1-based indexing
        .collect();

    // Write edges
    for edge in edges {
        let source_idx = node_to_index
            .get(&edge.source)
            .ok_or_else(|| GraphError::Other("Source node not found in mapping".to_string()))?;
        let target_idx = node_to_index
            .get(&edge.target)
            .ok_or_else(|| GraphError::Other("Target node not found in mapping".to_string()))?;

        if weighted {
            writeln!(file, "{} {} {}", source_idx, target_idx, edge.weight)
                .map_err(|e| GraphError::Other(format!("Error writing edge: {}", e)))?;
        } else {
            writeln!(file, "{} {} 1.0", source_idx, target_idx)
                .map_err(|e| GraphError::Other(format!("Error writing edge: {}", e)))?;
        }
    }

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

    #[test]
    fn test_graphml_format() {
        let mut graph: Graph<&str, f64> = Graph::new();
        graph.add_edge("A", "B", 1.5).unwrap();
        graph.add_edge("B", "C", 2.5).unwrap();

        // Test writing
        let temp_file = NamedTempFile::new().unwrap();
        write_graph(&graph, temp_file.path(), GraphFormat::GraphML, true).unwrap();

        // Verify the content contains GraphML structure
        let content = fs::read_to_string(temp_file.path()).unwrap();
        assert!(content.contains("<?xml version=\"1.0\""));
        assert!(content.contains("<graphml"));
        assert!(content.contains("<graph"));
        assert!(content.contains("<node id=\"A\""));
        assert!(content.contains("<node id=\"B\""));
        assert!(content.contains("<node id=\"C\""));
        assert!(content.contains("<edge source=\"A\" target=\"B\""));
        assert!(content.contains("<edge source=\"B\" target=\"C\""));
        assert!(content.contains("</graphml>"));
    }

    #[test]
    fn test_graphml_format_round_trip() {
        let mut original_graph: Graph<String, f64> = Graph::new();
        original_graph
            .add_edge("X".to_string(), "Y".to_string(), 1.0)
            .unwrap();
        original_graph
            .add_edge("Y".to_string(), "Z".to_string(), 2.0)
            .unwrap();

        // Write to GraphML
        let temp_file = NamedTempFile::new().unwrap();
        write_graph(
            &original_graph,
            temp_file.path(),
            GraphFormat::GraphML,
            true,
        )
        .unwrap();

        // Read back from GraphML
        let read_graph: Graph<String, f64> =
            read_graph(temp_file.path(), GraphFormat::GraphML, true, false).unwrap();

        // Verify the graph structure is preserved
        assert_eq!(read_graph.node_count(), 3);
        assert_eq!(read_graph.edge_count(), 2);
        assert!(read_graph.has_edge(&"X".to_string(), &"Y".to_string()));
        assert!(read_graph.has_edge(&"Y".to_string(), &"Z".to_string()));
    }

    #[test]
    fn test_matrix_market_format() {
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_edge(1, 2, 1.5).unwrap();
        graph.add_edge(2, 3, 2.5).unwrap();

        // Test writing
        let temp_file = NamedTempFile::new().unwrap();
        write_graph(&graph, temp_file.path(), GraphFormat::MatrixMarket, true).unwrap();

        // Verify the content contains Matrix Market structure
        let content = fs::read_to_string(temp_file.path()).unwrap();
        assert!(content.contains("%%MatrixMarket"));
        assert!(content.contains("matrix coordinate"));
        // Should have dimensions line
        assert!(content.contains("3 3 2")); // 3 nodes, 3 nodes, 2 edges
    }

    #[test]
    fn test_matrix_market_format_round_trip() {
        let mut original_graph: Graph<i32, f64> = Graph::new();
        original_graph.add_edge(1, 2, 1.0).unwrap();
        original_graph.add_edge(2, 3, 2.0).unwrap();
        original_graph.add_edge(3, 1, 3.0).unwrap();

        // Write to Matrix Market
        let temp_file = NamedTempFile::new().unwrap();
        write_graph(
            &original_graph,
            temp_file.path(),
            GraphFormat::MatrixMarket,
            true,
        )
        .unwrap();

        // Read back from Matrix Market
        let read_graph: Graph<i32, f64> =
            read_graph(temp_file.path(), GraphFormat::MatrixMarket, true, false).unwrap();

        // Verify the graph structure is preserved
        assert_eq!(read_graph.node_count(), 3);
        assert_eq!(read_graph.edge_count(), 3);
        assert!(read_graph.has_edge(&1, &2));
        assert!(read_graph.has_edge(&2, &3));
        assert!(read_graph.has_edge(&3, &1));
    }

    #[test]
    fn test_gml_format() {
        let mut graph: Graph<&str, f64> = Graph::new();
        graph.add_edge("A", "B", 1.5).unwrap();
        graph.add_edge("B", "C", 2.5).unwrap();

        // Test writing
        let temp_file = NamedTempFile::new().unwrap();
        write_graph(&graph, temp_file.path(), GraphFormat::Gml, true).unwrap();

        // Verify the content contains GML structure
        let content = fs::read_to_string(temp_file.path()).unwrap();
        assert!(content.contains("graph ["));
        assert!(content.contains("node ["));
        assert!(content.contains("id \"A\""));
        assert!(content.contains("id \"B\""));
        assert!(content.contains("id \"C\""));
        assert!(content.contains("edge ["));
        assert!(content.contains("source \"A\""));
        assert!(content.contains("target \"B\""));
        assert!(content.contains("source \"B\""));
        assert!(content.contains("target \"C\""));
        assert!(content.contains("value \"1.5\""));
        assert!(content.contains("value \"2.5\""));
        assert!(content.contains("]"));
    }

    #[test]
    fn test_gml_format_round_trip() {
        let mut original_graph: Graph<String, f64> = Graph::new();
        original_graph
            .add_edge("X".to_string(), "Y".to_string(), 1.0)
            .unwrap();
        original_graph
            .add_edge("Y".to_string(), "Z".to_string(), 2.0)
            .unwrap();

        // Write to GML
        let temp_file = NamedTempFile::new().unwrap();
        write_graph(&original_graph, temp_file.path(), GraphFormat::Gml, true).unwrap();

        // Read back from GML
        let read_graph: Graph<String, f64> =
            read_graph(temp_file.path(), GraphFormat::Gml, true, false).unwrap();

        // Verify the graph structure is preserved
        assert_eq!(read_graph.node_count(), 3);
        assert_eq!(read_graph.edge_count(), 2);
        assert!(read_graph.has_edge(&"X".to_string(), &"Y".to_string()));
        assert!(read_graph.has_edge(&"Y".to_string(), &"Z".to_string()));
    }

    #[test]
    fn test_gml_format_unweighted() {
        let mut graph: Graph<&str, i32> = Graph::new();
        graph.add_edge("P", "Q", 1).unwrap();
        graph.add_edge("Q", "R", 1).unwrap();

        // Test GML format without weights
        let temp_file = NamedTempFile::new().unwrap();
        write_graph(&graph, temp_file.path(), GraphFormat::Gml, false).unwrap();

        let content = fs::read_to_string(temp_file.path()).unwrap();
        assert!(content.contains("source \"P\""));
        assert!(content.contains("target \"Q\""));
        assert!(content.contains("source \"Q\""));
        assert!(content.contains("target \"R\""));
        assert!(!content.contains("value"));
    }

    #[test]
    fn test_digraph_gml_format() {
        let mut graph: DiGraph<&str, f64> = DiGraph::new();
        graph.add_edge("A", "B", 1.5).unwrap();
        graph.add_edge("B", "C", 2.5).unwrap();

        // Test writing
        let temp_file = NamedTempFile::new().unwrap();
        write_digraph(&graph, temp_file.path(), GraphFormat::Gml, true).unwrap();

        // Verify the content contains GML structure with directed graph
        let content = fs::read_to_string(temp_file.path()).unwrap();
        assert!(content.contains("graph ["));
        assert!(content.contains("directed 1"));
        assert!(content.contains("node ["));
        assert!(content.contains("id \"A\""));
        assert!(content.contains("id \"B\""));
        assert!(content.contains("id \"C\""));
        assert!(content.contains("edge ["));
        assert!(content.contains("source \"A\""));
        assert!(content.contains("target \"B\""));
        assert!(content.contains("source \"B\""));
        assert!(content.contains("target \"C\""));
        assert!(content.contains("value \"1.5\""));
        assert!(content.contains("value \"2.5\""));
        assert!(content.contains("]"));
    }

    #[test]
    fn test_digraph_gml_format_round_trip() {
        let mut original_graph: DiGraph<String, f64> = DiGraph::new();
        original_graph
            .add_edge("X".to_string(), "Y".to_string(), 1.0)
            .unwrap();
        original_graph
            .add_edge("Y".to_string(), "Z".to_string(), 2.0)
            .unwrap();

        // Write to GML
        let temp_file = NamedTempFile::new().unwrap();
        write_digraph(&original_graph, temp_file.path(), GraphFormat::Gml, true).unwrap();

        // Read back from GML
        let read_graph: DiGraph<String, f64> =
            read_digraph(temp_file.path(), GraphFormat::Gml, true).unwrap();

        // Verify the graph structure is preserved
        assert_eq!(read_graph.node_count(), 3);
        assert_eq!(read_graph.edge_count(), 2);
        assert!(read_graph.has_edge(&"X".to_string(), &"Y".to_string()));
        assert!(read_graph.has_edge(&"Y".to_string(), &"Z".to_string()));
        // For directed graphs, reverse edges should not exist
        assert!(!read_graph.has_edge(&"Y".to_string(), &"X".to_string()));
        assert!(!read_graph.has_edge(&"Z".to_string(), &"Y".to_string()));
    }

    // Directed graph tests

    #[test]
    fn test_digraph_edge_list_format() {
        let mut graph: DiGraph<&str, f64> = DiGraph::new();
        graph.add_edge("A", "B", 1.0).unwrap();
        graph.add_edge("B", "C", 2.0).unwrap();
        graph.add_edge("C", "A", 3.0).unwrap();

        // Test writing
        let temp_file = NamedTempFile::new().unwrap();
        write_digraph(&graph, temp_file.path(), GraphFormat::EdgeList, true).unwrap();

        // Test reading
        let read_graph: DiGraph<String, f64> =
            read_digraph(temp_file.path(), GraphFormat::EdgeList, true).unwrap();
        assert_eq!(read_graph.node_count(), 3);
        assert_eq!(read_graph.edge_count(), 3);
    }

    #[test]
    fn test_digraph_dot_format() {
        let mut graph: DiGraph<&str, f64> = DiGraph::new();
        graph.add_edge("A", "B", 1.5).unwrap();
        graph.add_edge("B", "C", 2.5).unwrap();

        // Test writing
        let temp_file = NamedTempFile::new().unwrap();
        write_digraph(&graph, temp_file.path(), GraphFormat::Dot, true).unwrap();

        // Verify the content
        let content = fs::read_to_string(temp_file.path()).unwrap();
        assert!(content.contains("digraph G {"));
        assert!(content.contains("A -> B [label=\"1.5\"];"));
        assert!(content.contains("B -> C [label=\"2.5\"];"));
        assert!(content.contains("}"));
    }

    #[test]
    fn test_digraph_adjacency_list_format() {
        let mut graph: DiGraph<i32, f64> = DiGraph::new();
        graph.add_edge(1, 2, 1.0).unwrap();
        graph.add_edge(2, 3, 2.0).unwrap();

        // Test writing
        let temp_file = NamedTempFile::new().unwrap();
        write_digraph(&graph, temp_file.path(), GraphFormat::AdjacencyList, true).unwrap();

        // Test reading
        let read_graph: DiGraph<i32, f64> =
            read_digraph(temp_file.path(), GraphFormat::AdjacencyList, true).unwrap();
        assert_eq!(read_graph.node_count(), 3);
        assert!(read_graph.has_edge(&1, &2));
        assert!(read_graph.has_edge(&2, &3));
        // For directed graphs, reverse edges should not exist
        assert!(!read_graph.has_edge(&2, &1));
        assert!(!read_graph.has_edge(&3, &2));
    }

    #[test]
    fn test_digraph_json_format_round_trip() {
        let mut original_graph: DiGraph<String, f64> = DiGraph::new();
        original_graph
            .add_edge("A".to_string(), "B".to_string(), 1.5)
            .unwrap();
        original_graph
            .add_edge("B".to_string(), "C".to_string(), 2.5)
            .unwrap();

        // Write to JSON
        let temp_file = NamedTempFile::new().unwrap();
        write_digraph(&original_graph, temp_file.path(), GraphFormat::Json, true).unwrap();

        // Read back from JSON
        let read_graph: DiGraph<String, f64> =
            read_digraph(temp_file.path(), GraphFormat::Json, true).unwrap();

        // Verify the graph structure is preserved
        assert_eq!(read_graph.node_count(), 3);
        assert_eq!(read_graph.edge_count(), 2);
        assert!(read_graph.has_edge(&"A".to_string(), &"B".to_string()));
        assert!(read_graph.has_edge(&"B".to_string(), &"C".to_string()));
        // For directed graphs, reverse edges should not exist
        assert!(!read_graph.has_edge(&"B".to_string(), &"A".to_string()));
        assert!(!read_graph.has_edge(&"C".to_string(), &"B".to_string()));
    }

    #[test]
    fn test_digraph_graphml_format() {
        let mut graph: DiGraph<&str, f64> = DiGraph::new();
        graph.add_edge("A", "B", 1.5).unwrap();
        graph.add_edge("B", "C", 2.5).unwrap();

        // Test writing
        let temp_file = NamedTempFile::new().unwrap();
        write_digraph(&graph, temp_file.path(), GraphFormat::GraphML, true).unwrap();

        // Verify the content contains GraphML structure with directed graph
        let content = fs::read_to_string(temp_file.path()).unwrap();
        assert!(content.contains("<?xml version=\"1.0\""));
        assert!(content.contains("<graphml"));
        assert!(content.contains("edgedefault=\"directed\""));
        assert!(content.contains("<node id=\"A\""));
        assert!(content.contains("<node id=\"B\""));
        assert!(content.contains("<node id=\"C\""));
        assert!(content.contains("<edge source=\"A\" target=\"B\""));
        assert!(content.contains("<edge source=\"B\" target=\"C\""));
        assert!(content.contains("</graphml>"));
    }

    #[test]
    fn test_digraph_graphml_format_round_trip() {
        let mut original_graph: DiGraph<String, f64> = DiGraph::new();
        original_graph
            .add_edge("X".to_string(), "Y".to_string(), 1.0)
            .unwrap();
        original_graph
            .add_edge("Y".to_string(), "Z".to_string(), 2.0)
            .unwrap();

        // Write to GraphML
        let temp_file = NamedTempFile::new().unwrap();
        write_digraph(
            &original_graph,
            temp_file.path(),
            GraphFormat::GraphML,
            true,
        )
        .unwrap();

        // Read back from GraphML
        let read_graph: DiGraph<String, f64> =
            read_digraph(temp_file.path(), GraphFormat::GraphML, true).unwrap();

        // Verify the graph structure is preserved
        assert_eq!(read_graph.node_count(), 3);
        assert_eq!(read_graph.edge_count(), 2);
        assert!(read_graph.has_edge(&"X".to_string(), &"Y".to_string()));
        assert!(read_graph.has_edge(&"Y".to_string(), &"Z".to_string()));
        // For directed graphs, reverse edges should not exist
        assert!(!read_graph.has_edge(&"Y".to_string(), &"X".to_string()));
        assert!(!read_graph.has_edge(&"Z".to_string(), &"Y".to_string()));
    }

    #[test]
    fn test_digraph_matrix_market_format() {
        let mut graph: DiGraph<i32, f64> = DiGraph::new();
        graph.add_edge(1, 2, 1.5).unwrap();
        graph.add_edge(2, 3, 2.5).unwrap();

        // Test writing
        let temp_file = NamedTempFile::new().unwrap();
        write_digraph(&graph, temp_file.path(), GraphFormat::MatrixMarket, true).unwrap();

        // Verify the content contains Matrix Market structure
        let content = fs::read_to_string(temp_file.path()).unwrap();
        assert!(content.contains("%%MatrixMarket"));
        assert!(content.contains("matrix coordinate"));
        // Should have dimensions line
        assert!(content.contains("3 3 2")); // 3 nodes, 3 nodes, 2 edges
    }

    #[test]
    fn test_digraph_matrix_market_format_round_trip() {
        let mut original_graph: DiGraph<i32, f64> = DiGraph::new();
        original_graph.add_edge(1, 2, 1.0).unwrap();
        original_graph.add_edge(2, 3, 2.0).unwrap();

        // Write to Matrix Market
        let temp_file = NamedTempFile::new().unwrap();
        write_digraph(
            &original_graph,
            temp_file.path(),
            GraphFormat::MatrixMarket,
            true,
        )
        .unwrap();

        // Read back from Matrix Market
        let read_graph: DiGraph<i32, f64> =
            read_digraph(temp_file.path(), GraphFormat::MatrixMarket, true).unwrap();

        // Verify the graph structure is preserved
        assert_eq!(read_graph.node_count(), 3);
        assert_eq!(read_graph.edge_count(), 2);
        assert!(read_graph.has_edge(&1, &2));
        assert!(read_graph.has_edge(&2, &3));
        // For directed graphs, reverse edges should not exist
        assert!(!read_graph.has_edge(&2, &1));
        assert!(!read_graph.has_edge(&3, &2));
    }

    #[test]
    fn test_digraph_unweighted_formats() {
        let mut graph: DiGraph<&str, i32> = DiGraph::new();
        graph.add_edge("P", "Q", 1).unwrap();
        graph.add_edge("Q", "R", 1).unwrap();

        // Test DOT format without weights
        let temp_file_dot = NamedTempFile::new().unwrap();
        write_digraph(&graph, temp_file_dot.path(), GraphFormat::Dot, false).unwrap();

        let content = fs::read_to_string(temp_file_dot.path()).unwrap();
        assert!(content.contains("P -> Q;"));
        assert!(content.contains("Q -> R;"));
        assert!(!content.contains("label="));

        // Test adjacency list format without weights
        let temp_file_adj = NamedTempFile::new().unwrap();
        write_digraph(
            &graph,
            temp_file_adj.path(),
            GraphFormat::AdjacencyList,
            false,
        )
        .unwrap();

        let content = fs::read_to_string(temp_file_adj.path()).unwrap();
        // For directed graphs, we should only see outgoing edges
        assert!(content.contains("P: Q"));
        assert!(content.contains("Q: R"));
        // R should not have any outgoing edges
        assert!(!content.contains("R:"));
    }
}
