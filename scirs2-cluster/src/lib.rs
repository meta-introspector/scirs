//! Clustering algorithms module for SciRS2
//!
//! This module provides implementations of various clustering algorithms such as:
//! - Vector quantization (k-means, etc.)
//! - Hierarchical clustering
//! - Density-based clustering (DBSCAN, OPTICS, etc.)
//! - Mean Shift clustering
//! - Spectral clustering
//! - Affinity Propagation
//!
//! ## Features
//!
//! * **Vector Quantization**: K-means and K-means++ for partitioning data
//! * **Hierarchical Clustering**: Agglomerative clustering with various linkage methods
//! * **Density-based Clustering**: DBSCAN and OPTICS for finding clusters of arbitrary shape
//! * **Mean Shift**: Non-parametric clustering based on density estimation
//! * **Spectral Clustering**: Graph-based clustering using eigenvectors of the graph Laplacian
//! * **Affinity Propagation**: Message-passing based clustering that identifies exemplars
//! * **Evaluation Metrics**: Silhouette coefficient, Davies-Bouldin index, and other measures to evaluate clustering quality
//! * **Data Preprocessing**: Utilities for normalizing, standardizing, and whitening data before clustering
//!
//! ## Examples
//!
//! ```
//! use ndarray::{Array2, ArrayView2};
//! use scirs2_cluster::vq::kmeans;
//! use scirs2_cluster::preprocess::standardize;
//!
//! // Example data with two clusters
//! let data = Array2::from_shape_vec((6, 2), vec![
//!     1.0, 2.0,
//!     1.2, 1.8,
//!     0.8, 1.9,
//!     3.7, 4.2,
//!     3.9, 3.9,
//!     4.2, 4.1,
//! ]).unwrap();
//!
//! // Standardize the data
//! let standardized = standardize(data.view(), true).unwrap();
//!
//! // Run k-means with k=2
//! let (centroids, labels) = kmeans(standardized.view(), 2, None, None, None, None).unwrap();
//!
//! // Print the results
//! println!("Centroids: {:?}", centroids);
//! println!("Cluster assignments: {:?}", labels);
//! ```

#![warn(missing_docs)]

/// Cutting-edge clustering algorithms including quantum-inspired methods and advanced online learning.
///
/// This module provides state-of-the-art clustering algorithms that push the boundaries
/// of traditional clustering methods. It includes quantum-inspired algorithms that leverage
/// quantum computing principles and advanced online learning variants with concept drift detection.
///
/// # Features
///
/// * **Quantum K-means**: Uses quantum superposition principles for potentially better optimization
/// * **Adaptive Online Clustering**: Automatically adapts to changing data distributions
/// * **Concept Drift Detection**: Detects and adapts to changes in streaming data
/// * **Dynamic Cluster Management**: Creates, merges, and removes clusters automatically
/// * **Quantum Annealing**: Simulated quantum annealing for global optimization
pub mod advanced;
pub mod affinity;
pub mod birch;
pub mod density;
/// Distributed clustering algorithms for large-scale datasets.
///
/// This module provides distributed implementations of clustering algorithms that can
/// handle datasets too large to fit in memory on a single machine. It supports
/// distributed K-means, hierarchical clustering, and various data partitioning strategies.
///
/// # Features
///
/// * **Distributed K-means**: Multi-node K-means with coordination rounds
/// * **Distributed Hierarchical Clustering**: Large-scale hierarchical clustering
/// * **Data Partitioning**: Multiple strategies for distributing data across workers
/// * **Load Balancing**: Dynamic and static load balancing strategies
/// * **Memory Management**: Configurable memory limits and optimization
/// * **Fault Tolerance**: Worker failure detection and recovery mechanisms
pub mod distributed;
/// Ensemble clustering methods for improved robustness.
///
/// This module provides ensemble clustering techniques that combine multiple
/// clustering algorithms or multiple runs of the same algorithm to achieve
/// more robust and stable clustering results.
pub mod ensemble;
pub mod error;
pub mod gmm;
/// GPU acceleration module for clustering algorithms.
///
/// This module provides GPU acceleration interfaces and implementations for clustering
/// algorithms. It supports multiple GPU backends including CUDA, OpenCL, ROCm, and others.
/// When GPU acceleration is not available or disabled, algorithms automatically fall back
/// to optimized CPU implementations.
///
/// # Features
///
/// * **Multiple GPU Backends**: Support for CUDA, OpenCL, ROCm, Intel OneAPI, and Metal
/// * **Automatic Fallback**: Seamless fallback to CPU when GPU is not available
/// * **Memory Management**: Efficient GPU memory allocation and pooling
/// * **Performance Monitoring**: Built-in benchmarking and performance statistics
/// * **Device Selection**: Automatic or manual GPU device selection strategies
#[cfg(feature = "gpu")]
pub mod gpu;
/// Graph clustering and community detection algorithms.
///
/// This module provides implementations of various graph clustering algorithms for
/// detecting communities and clusters in network data. These algorithms work with
/// graph representations where nodes represent data points and edges represent
/// similarities or connections between them.
///
/// # Features
///
/// * **Community Detection**: Louvain algorithm for modularity optimization
/// * **Label Propagation**: Fast algorithm for community detection
/// * **Hierarchical Methods**: Girvan-Newman algorithm for hierarchical communities
/// * **Graph Construction**: k-NN graphs, adjacency matrix support
/// * **Quality Metrics**: Modularity calculation and community evaluation
pub mod graph;
pub mod hierarchy;
pub mod input_validation;
pub mod leader;
/// Mean Shift clustering implementation.
///
/// This module provides the Mean Shift clustering algorithm, which is a centroid-based
/// algorithm that works by updating candidates for centroids to be the mean of the points
/// within a given region. These candidates are then filtered in a post-processing stage to
/// eliminate near-duplicates, forming the final set of centroids.
///
/// Mean Shift is a non-parametric clustering technique that doesn't require specifying the
/// number of clusters in advance and can find clusters of arbitrary shapes.
pub mod meanshift;
pub mod metrics;
pub mod neighbor_search;
pub mod preprocess;
pub mod serialization;
pub mod sparse;
pub mod spectral;
pub mod stability;
pub mod streaming;
/// Text clustering algorithms with semantic similarity support.
///
/// This module provides specialized clustering algorithms for text data that leverage
/// semantic similarity measures rather than traditional distance metrics. It includes
/// algorithms optimized for document clustering, sentence clustering, and topic modeling.
///
/// # Features
///
/// * **Semantic K-means**: K-means clustering with semantic similarity metrics
/// * **Hierarchical Text Clustering**: Agglomerative clustering for text data
/// * **Topic-based Clustering**: Clustering based on topic modeling approaches
/// * **Multiple Text Representations**: Support for TF-IDF, word embeddings, contextualized embeddings
/// * **Semantic Similarity Metrics**: Cosine, Jaccard, Jensen-Shannon, and other text-specific metrics
pub mod text_clustering;
/// Time series clustering algorithms with specialized distance metrics.
///
/// This module provides clustering algorithms specifically designed for time series data,
/// including dynamic time warping (DTW) distance and other temporal similarity measures.
/// These algorithms can handle time series of different lengths and temporal alignments.
///
/// # Features
///
/// * **Dynamic Time Warping**: DTW distance with optional constraints
/// * **Soft DTW**: Differentiable variant for gradient-based optimization
/// * **Time Series K-means**: Clustering with DTW barycenter averaging
/// * **Time Series K-medoids**: Robust clustering using actual time series as centers
/// * **Hierarchical Clustering**: Agglomerative clustering with DTW distance
pub mod time_series;
/// Automatic hyperparameter tuning for clustering algorithms.
///
/// This module provides comprehensive hyperparameter optimization capabilities
/// for all clustering algorithms in the scirs2-cluster crate. It supports
/// grid search, random search, Bayesian optimization, and adaptive strategies.
pub mod tuning;
pub mod vq;

// Re-exports
pub use advanced::{
    adaptive_online_clustering, quantum_kmeans, rl_clustering, transfer_learning_clustering,
    AdaptiveOnlineClustering, AdaptiveOnlineConfig, FeatureAlignment, QuantumConfig, QuantumKMeans,
    RLClustering, RLClusteringConfig, RewardFunction, TransferLearningClustering,
    TransferLearningConfig,
};
pub use affinity::{affinity_propagation, AffinityPropagationOptions};
pub use birch::{birch, Birch, BirchOptions, BirchStatistics};
pub use density::hdbscan::{
    dbscan_clustering, hdbscan, ClusterSelectionMethod, HDBSCANOptions, HDBSCANResult, StoreCenter,
};
pub use density::optics::{extract_dbscan_clustering, extract_xi_clusters, OPTICSResult};
pub use density::*;
pub use ensemble::convenience::{
    bootstrap_ensemble, ensemble_clustering, multi_algorithm_ensemble,
};
pub use ensemble::{
    ClusteringAlgorithm, ClusteringResult, ConsensusMethod, ConsensusStatistics, DiversityMetrics,
    DiversityStrategy, EnsembleClusterer, EnsembleConfig, EnsembleResult, NoiseType,
    ParameterRange, SamplingStrategy,
};
pub use gmm::{gaussian_mixture, CovarianceType, GMMInit, GMMOptions, GaussianMixture};
pub use graph::{
    girvan_newman, graph_clustering, label_propagation, louvain, Graph, GraphClusteringAlgorithm,
    GraphClusteringConfig,
};
pub use hierarchy::*;
pub use input_validation::{
    check_duplicate_points, suggest_clustering_algorithm, validate_clustering_data,
    validate_convergence_parameters, validate_distance_parameter, validate_integer_parameter,
    validate_n_clusters, validate_sample_weights, ValidationConfig,
};
pub use leader::{
    euclidean_distance, leader_clustering, manhattan_distance, LeaderClustering, LeaderNode,
    LeaderTree,
};
pub use meanshift::{estimate_bandwidth, get_bin_seeds, mean_shift, MeanShift, MeanShiftOptions};
pub use metrics::{
    adjusted_rand_index, calinski_harabasz_score, davies_bouldin_score,
    homogeneity_completeness_v_measure, normalized_mutual_info, silhouette_samples,
    silhouette_score,
};

// Re-export ensemble validation methods
pub use metrics::ensemble::{
    bootstrap_confidence_interval, consensus_clustering_score, cross_validation_score,
    multi_criterion_validation, robust_validation,
};

// Re-export information-theoretic methods
pub use metrics::information_theory::{
    information_cluster_quality, jensen_shannon_divergence, variation_of_information,
};

// Re-export stability-based methods
pub use metrics::stability::{cluster_stability_bootstrap, optimal_clusters_stability};

// Re-export advanced metrics
pub use metrics::advanced::{bic_score, dunn_index};
pub use neighbor_search::{
    create_neighbor_searcher, BallTree, BruteForceSearch, KDTree, NeighborResult,
    NeighborSearchAlgorithm, NeighborSearchConfig, NeighborSearcher,
};
pub use preprocess::{min_max_scale, normalize, standardize, whiten, NormType};
pub use serialization::{
    affinity_propagation_to_model, birch_to_model, compatibility, dbscan_to_model, gmm_to_model,
    hierarchy_to_model, kmeans_to_model, leader_to_model, leader_tree_to_model, meanshift_to_model,
    save_affinity_propagation, save_birch, save_gmm, save_hierarchy, save_kmeans, save_leader,
    save_leader_tree, save_spectral_clustering, spectral_clustering_to_model, AdvancedExport,
    AffinityPropagationModel, BirchModel, DBSCANModel, ExportFormat, GMMModel, HierarchicalModel,
    KMeansModel, LeaderModel, LeaderNodeModel, LeaderTreeModel, MeanShiftModel, ModelMetadata,
    SerializableModel, SpectralClusteringModel,
};
pub use sparse::{
    sparse_epsilon_graph, sparse_knn_graph, SparseDistanceMatrix, SparseHierarchicalClustering,
};
pub use spectral::{
    spectral_bipartition, spectral_clustering, AffinityMode, SpectralClusteringOptions,
};
pub use stability::{
    BootstrapValidator, ConsensusClusterer, OptimalKSelector, StabilityConfig, StabilityResult,
};
pub use streaming::{
    ChunkedDistanceMatrix, ProgressiveHierarchical, StreamingConfig, StreamingKMeans,
};
pub use text_clustering::{
    semantic_hierarchical, semantic_kmeans, topic_clustering, SemanticClusteringConfig,
    SemanticHierarchical, SemanticKMeans, SemanticSimilarity, TextPreprocessing,
    TextRepresentation, TopicBasedClustering,
};
pub use time_series::{
    dtw_barycenter_averaging, dtw_distance, dtw_distance_custom, dtw_hierarchical_clustering,
    dtw_k_means, dtw_k_medoids, soft_dtw_distance, time_series_clustering, TimeSeriesAlgorithm,
    TimeSeriesClusteringConfig,
};
pub use tuning::{
    AcquisitionFunction, AutoTuner, BayesianState, CVStrategy, ConvergenceInfo,
    CrossValidationConfig, EarlyStoppingConfig, EnsembleResults, EvaluationMetric,
    EvaluationResult, ExplorationStats, HyperParameter, KernelType, LoadBalancingStrategy,
    ParallelConfig, ResourceConstraints, SearchSpace, SearchStrategy, StandardSearchSpaces,
    StoppingReason, SurrogateModel, TuningConfig, TuningResult,
};

// Re-export visualization and animation capabilities
pub use visualization::{
    create_scatter_plot_2d, create_scatter_plot_3d, AnimationConfig, BoundaryType, ClusterBoundary,
    ColorScheme, DimensionalityReduction, EasingFunction, LegendEntry, ScatterPlot2D,
    ScatterPlot3D, VisualizationConfig,
};

// Re-export animation features
pub use visualization::animation::{
    AnimationFrame, ConvergenceInfo, IterativeAnimationConfig, IterativeAnimationRecorder,
    StreamingConfig, StreamingVisualizer,
};

// Re-export interactive visualization features
pub use visualization::interactive::{
    ClusterStats, InteractiveConfig, InteractiveState, InteractiveVisualizer,
};

// Re-export export capabilities
pub use visualization::export::{
    export_scatter_2d_to_html, export_scatter_2d_to_json, export_scatter_3d_to_html,
    export_scatter_3d_to_json, save_visualization_to_file, ExportFormat,
};

// Re-export distributed clustering capabilities
pub use distributed::{
    ConvergenceMetrics, DataPartition, DistributedConfig, DistributedHierarchical,
    DistributedKMeans, LinkageMethod, LocalDendrogram, PartitioningStrategy, WorkerStatistics,
};

// Re-export distributed utilities
pub use distributed::utils::{estimate_optimal_workers, generate_large_dataset};
pub use vq::*;

// GPU acceleration re-exports (when GPU feature is enabled)
#[cfg(feature = "gpu")]
pub use gpu::{
    DeviceSelection, DistanceMetric as GpuDistanceMetric, GpuBackend, GpuConfig, GpuContext,
    GpuDevice, GpuDistanceMatrix, GpuKMeans, GpuKMeansConfig, GpuMemoryManager, GpuStats,
    MemoryStats, MemoryStrategy,
};

#[cfg(feature = "gpu")]
/// GPU acceleration benchmark utilities
pub mod gpu_benchmark {
    pub use crate::gpu::benchmark::*;
}

#[cfg(feature = "gpu")]
/// High-level GPU-accelerated clustering with automatic fallback
pub mod accelerated {
    pub use crate::gpu::accelerated::*;
}

// Always available GPU acceleration interface (with CPU fallback)
/// GPU-accelerated clustering with automatic CPU fallback
///
/// This module provides high-level clustering algorithms that automatically
/// use GPU acceleration when available, falling back to CPU implementations
/// when GPU is not available or optimal.
pub mod gpu_accelerated {
    pub use crate::gpu::accelerated::*;
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod property_tests;
