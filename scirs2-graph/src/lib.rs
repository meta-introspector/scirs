//! Graph processing module for SciRS2
//!
//! This module provides graph algorithms and data structures
//! for scientific computing and machine learning applications.
//!
//! ## Features
//!
//! - Basic graph representations and operations
//! - Graph algorithms (traversal, shortest paths, etc.)
//! - Network analysis (centrality measures, community detection)
//! - Spectral graph theory
//! - Support for graph neural networks
//!
//! ## Stability
//!
//! This module follows semantic versioning. Most APIs are stable and covered
//! by our compatibility guarantee. Experimental features are clearly marked.

#![warn(missing_docs)]
#![cfg_attr(feature = "unstable", feature(stability))]

/// Stability attributes for API items
///
/// These macros help track API stability across the library
#[cfg(feature = "unstable")]
macro_rules! stable {
    ($feature:expr, $since:expr) => {
        #[stable(feature = $feature, since = $since)]
    };
}

#[cfg(not(feature = "unstable"))]
macro_rules! stable {
    ($feature:expr, $since:expr) => {
        // No-op when unstable feature is not enabled
    };
}

#[cfg(feature = "unstable")]
macro_rules! unstable {
    ($feature:expr, $issue:expr) => {
        #[unstable(feature = $feature, issue = $issue)]
    };
}

#[cfg(not(feature = "unstable"))]
macro_rules! unstable {
    ($feature:expr, $issue:expr) => {
        // No-op when unstable feature is not enabled
    };
}

// Temporarily commenting out OpenBLAS to fix build issues
// extern crate blas;
// extern crate openblas_src;

pub mod algorithms;
pub mod attributes;
pub mod base;
pub mod embeddings;
pub mod error;
pub mod generators;
pub mod io;
pub mod layout;
pub mod measures;
pub mod memory;
pub mod performance;
pub mod spectral;
pub mod temporal;
pub mod weighted;

// Re-export important types and functions
stable!("graph_core", "0.1.0-beta.1");
pub use algorithms::{
    // Core algorithms - stable for 1.0
    articulation_points,
    astar_search,
    astar_search_digraph,
    betweenness_centrality,
    bidirectional_search,
    bidirectional_search_digraph,
    breadth_first_search,
    breadth_first_search_digraph,
    bridges,
    // Additional stable algorithms for 1.0
    center_nodes,
    // Centrality measures - stable for 1.0
    closeness_centrality,
    // Graph transformations - stable for 1.0
    complement,
    connected_components,
    cosine_similarity,
    depth_first_search,
    depth_first_search_digraph,
    diameter,
    dijkstra_path,
    // Flow algorithms - stable for 1.0
    dinic_max_flow,
    edge_subgraph,

    eigenvector_centrality,

    eulerian_type,
    floyd_warshall,
    floyd_warshall_digraph,
    fluid_communities_result,
    greedy_coloring,
    greedy_modularity_optimization_result,
    hierarchical_communities_result,
    infomap_communities,
    is_bipartite,
    jaccard_similarity,
    k_core_decomposition,
    k_shortest_paths,
    label_propagation_result,
    line_digraph,
    line_graph,
    // Community detection algorithms - stable for 1.0
    louvain_communities_result,
    parallel_louvain_communities_result,
    // Matching algorithms - stable for 1.0
    maximal_matching,
    maximum_bipartite_matching,
    maximum_cardinality_matching,
    minimum_cut,
    minimum_spanning_tree,
    minimum_weight_bipartite_matching,
    modularity,

    modularity_optimization_result,
    pagerank,
    personalized_pagerank,
    push_relabel_max_flow,

    radius,
    random_walk,
    shortest_path_digraph,
    stable_marriage,

    strongly_connected_components,
    subdigraph,
    subgraph,
    tensor_product,
    topological_sort,

    transition_matrix,
    weight_filtered_subgraph,

    // Result types - stable for 1.0
    AStarResult,
    BipartiteMatching,
    BipartiteResult,
    CommunityResult,
    CommunityStructure,
    EulerianType,
    GraphColoring,
    InfomapResult,
    MaximumMatching,
    MotifType,
};

// Experimental algorithms - unstable, may change in future versions
unstable!("graph_advanced", "none");
pub use algorithms::{
    // Isomorphism and advanced matching - experimental
    are_graphs_isomorphic,
    are_graphs_isomorphic_enhanced,
    cartesian_product,
    chromatic_number,
    find_isomorphism,
    find_isomorphism_vf2,
    find_motifs,
    find_subgraph_matches,
    graph_edit_distance,
    has_hamiltonian_circuit,
    has_hamiltonian_path,
};

// Add deprecation warnings for legacy functions
#[deprecated(
    since = "0.1.0-beta.2",
    note = "Use `dijkstra_path` for future compatibility. This function will return PathResult in v1.0"
)]
pub use algorithms::shortest_path;

#[deprecated(
    since = "0.1.0-beta.2",
    note = "Use `louvain_communities_result` instead"
)]
pub use algorithms::louvain_communities;

#[deprecated(
    since = "0.1.0-beta.2",
    note = "Use `label_propagation_result` instead"
)]
pub use algorithms::label_propagation;

#[deprecated(
    since = "0.1.0-beta.2",
    note = "Use `fluid_communities_result` instead"
)]
pub use algorithms::fluid_communities;

#[deprecated(
    since = "0.1.0-beta.2",
    note = "Use `hierarchical_communities_result` instead"
)]
pub use algorithms::hierarchical_communities;

#[deprecated(
    since = "0.1.0-beta.2",
    note = "Use `modularity_optimization_result` instead"
)]
pub use algorithms::modularity_optimization;

#[deprecated(
    since = "0.1.0-beta.2",
    note = "Use `greedy_modularity_optimization_result` instead"
)]
pub use algorithms::greedy_modularity_optimization;

#[deprecated(
    since = "0.1.0-beta.2",
    note = "Use `parallel_louvain_communities_result` instead"
)]
pub use algorithms::parallel_louvain_communities;

// Attribute system - stable for 1.0
stable!("graph_attributes", "0.1.0-beta.1");
pub use attributes::{
    AttributeSummary, AttributeValue, AttributeView, AttributedDiGraph, AttributedGraph, Attributes,
};

// Core graph types - stable for 1.0
stable!("graph_types", "0.1.0-beta.1");
pub use base::{
    BipartiteGraph, DiGraph, Edge, EdgeWeight, Graph, Hyperedge, Hypergraph, IndexType,
    MultiDiGraph, MultiGraph, Node,
};
// Graph embeddings - experimental features
unstable!("graph_embeddings", "none");
pub use embeddings::{
    DeepWalk, DeepWalkConfig, Embedding, EmbeddingModel, Node2Vec, Node2VecConfig, RandomWalk,
    RandomWalkGenerator,
};

// Error handling - stable for 1.0
stable!("graph_errors", "0.1.0-beta.1");
pub use error::{GraphError, Result};

// Graph generators - stable for 1.0
stable!("graph_generators", "0.1.0-beta.1");
pub use generators::{
    barabasi_albert_graph, complete_graph, cycle_graph, erdos_renyi_graph, grid_2d_graph,
    grid_3d_graph, hexagonal_lattice_graph, path_graph, planted_partition_model, star_graph,
    stochastic_block_model, triangular_lattice_graph, two_community_sbm, watts_strogatz_graph,
};

// Layout algorithms - experimental features
unstable!("graph_layout", "none");
pub use layout::{circular_layout, hierarchical_layout, spectral_layout, spring_layout, Position};

// Graph measures - stable for 1.0
stable!("graph_measures", "0.1.0-beta.1");
pub use measures::{
    centrality, clustering_coefficient, graph_density, hits_algorithm, katz_centrality,
    katz_centrality_digraph, pagerank_centrality, pagerank_centrality_digraph,
    parallel_pagerank_centrality, CentralityType, HitsScores,
};

// Memory optimization - stable for 1.0
stable!("graph_memory", "0.1.0-beta.1");
pub use memory::{
    suggest_optimizations, BitPackedGraph, CSRGraph, CompressedAdjacencyList, FragmentationReport,
    HybridGraph, MemoryProfiler, MemoryStats, OptimizedGraphBuilder, MemorySample, OptimizationSuggestions,
};

// Performance monitoring - stable for 1.0
stable!("graph_performance", "0.1.0-beta.1");
pub use performance::{
    LargeGraphIterator, LargeGraphOps, MemoryMetrics, ParallelConfig, PerformanceMonitor,
    PerformanceReport, StreamingGraphProcessor,
};

// Spectral analysis - stable for 1.0
stable!("graph_spectral", "0.1.0-beta.1");
pub use spectral::{laplacian, normalized_cut, spectral_radius};

// Temporal graphs - experimental features
unstable!("graph_temporal", "none");
pub use temporal::{
    temporal_betweenness_centrality, temporal_reachability, TemporalGraph, TemporalPath,
    TimeInstant, TimeInterval,
};

// Weighted operations - stable for 1.0
stable!("graph_weighted", "0.1.0-beta.1");
pub use weighted::{
    MultiWeight, NormalizationMethod, WeightStatistics, WeightTransform, WeightedOps,
};
