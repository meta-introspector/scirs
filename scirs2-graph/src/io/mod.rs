//! Input/output operations for graphs
//!
//! This module provides functions for reading and writing graph data
//! in various formats.

use std::path::Path;
use std::str::FromStr;

use crate::base::{DiGraph, EdgeWeight, Graph, Node};
use crate::error::{GraphError, Result};

pub mod adjacency_list;
pub mod dot;
pub mod edge_list;
pub mod matrix_market;

use adjacency_list::{
    read_adjacency_list_format, read_adjacency_list_format_digraph, write_adjacency_list_format,
    write_adjacency_list_format_digraph,
};
use dot::{
    read_dot_format, read_dot_format_digraph, write_dot_format, write_dot_format_digraph,
};
use edge_list::{
    read_edge_list_format, read_edge_list_format_digraph, write_edge_list_format,
    write_edge_list_format_digraph,
};
use matrix_market::{
    read_matrix_market_format, read_matrix_market_format_digraph,
    write_matrix_market_format, write_matrix_market_format_digraph,
};

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
pub fn read_graph<N, E, P>(path: P, format: GraphFormat, weighted: bool) -> Result<Graph<N, E>>
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
/// * `path` - Path to the output file
/// * `format` - Format to write the file in
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
/// * `path` - Path to the output file
/// * `format` - Format to write the file in
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

// TODO: Implement the remaining format functions
// For now, these are placeholders that return errors

// DOT format is now implemented in the dot module

fn read_json_format<N, E, P>(_path: P, _weighted: bool) -> Result<Graph<N, E>>
where
    N: Node + std::fmt::Debug + FromStr + Clone,
    E: EdgeWeight + std::marker::Copy + std::fmt::Debug + std::default::Default + FromStr,
    P: AsRef<Path>,
{
    Err(GraphError::Other(
        "JSON format not yet implemented".to_string(),
    ))
}

fn read_graphml_format<N, E, P>(_path: P, _weighted: bool) -> Result<Graph<N, E>>
where
    N: Node + std::fmt::Debug + FromStr + Clone,
    E: EdgeWeight + std::marker::Copy + std::fmt::Debug + std::default::Default + FromStr,
    P: AsRef<Path>,
{
    Err(GraphError::Other(
        "GraphML format not yet implemented".to_string(),
    ))
}

fn read_gml_format<N, E, P>(_path: P, _weighted: bool) -> Result<Graph<N, E>>
where
    N: Node + std::fmt::Debug + FromStr + Clone,
    E: EdgeWeight + std::marker::Copy + std::fmt::Debug + std::default::Default + FromStr,
    P: AsRef<Path>,
{
    Err(GraphError::Other(
        "GML format not yet implemented".to_string(),
    ))
}

// Matrix Market format is now implemented in the matrix_market module

// Digraph placeholder functions

// DOT digraph format is now implemented in the dot module

fn read_json_format_digraph<N, E, P>(_path: P, _weighted: bool) -> Result<DiGraph<N, E>>
where
    N: Node + std::fmt::Debug + FromStr + Clone,
    E: EdgeWeight + std::marker::Copy + std::fmt::Debug + std::default::Default + FromStr,
    P: AsRef<Path>,
{
    Err(GraphError::Other(
        "JSON format for digraph not yet implemented".to_string(),
    ))
}

fn read_graphml_format_digraph<N, E, P>(_path: P, _weighted: bool) -> Result<DiGraph<N, E>>
where
    N: Node + std::fmt::Debug + FromStr + Clone,
    E: EdgeWeight + std::marker::Copy + std::fmt::Debug + std::default::Default + FromStr,
    P: AsRef<Path>,
{
    Err(GraphError::Other(
        "GraphML format for digraph not yet implemented".to_string(),
    ))
}

fn read_gml_format_digraph<N, E, P>(_path: P, _weighted: bool) -> Result<DiGraph<N, E>>
where
    N: Node + std::fmt::Debug + FromStr + Clone,
    E: EdgeWeight + std::marker::Copy + std::fmt::Debug + std::default::Default + FromStr,
    P: AsRef<Path>,
{
    Err(GraphError::Other(
        "GML format for digraph not yet implemented".to_string(),
    ))
}

// Matrix Market digraph format is now implemented in the matrix_market module

// Write placeholder functions

// DOT write format is now implemented in the dot module

fn write_json_format<N, E, Ix, P>(_graph: &Graph<N, E, Ix>, _path: P, _weighted: bool) -> Result<()>
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
    Err(GraphError::Other(
        "JSON write format not yet implemented".to_string(),
    ))
}

fn write_graphml_format<N, E, Ix, P>(
    _graph: &Graph<N, E, Ix>,
    _path: P,
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
    Err(GraphError::Other(
        "GraphML write format not yet implemented".to_string(),
    ))
}

fn write_gml_format<N, E, Ix, P>(_graph: &Graph<N, E, Ix>, _path: P, _weighted: bool) -> Result<()>
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
    Err(GraphError::Other(
        "GML write format not yet implemented".to_string(),
    ))
}

// Matrix Market write format is now implemented in the matrix_market module

// Digraph write placeholder functions

// DOT digraph write format is now implemented in the dot module

fn write_json_format_digraph<N, E, Ix, P>(
    _graph: &DiGraph<N, E, Ix>,
    _path: P,
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
    Err(GraphError::Other(
        "JSON write format for digraph not yet implemented".to_string(),
    ))
}

fn write_graphml_format_digraph<N, E, Ix, P>(
    _graph: &DiGraph<N, E, Ix>,
    _path: P,
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
    Err(GraphError::Other(
        "GraphML write format for digraph not yet implemented".to_string(),
    ))
}

fn write_gml_format_digraph<N, E, Ix, P>(
    _graph: &DiGraph<N, E, Ix>,
    _path: P,
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
    Err(GraphError::Other(
        "GML write format for digraph not yet implemented".to_string(),
    ))
}

// Matrix Market digraph write format is now implemented in the matrix_market module
