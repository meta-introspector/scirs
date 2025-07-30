//! Distributed clustering algorithms for large-scale datasets
//!
//! This module provides distributed implementations of clustering algorithms that can
//! handle datasets too large to fit in memory on a single machine. It supports
//! distributed K-means, hierarchical clustering, and data partitioning strategies.
//!
//! The implementation has been modularized for better organization and maintainability:
//!
//! ## Architecture
//!
//! - **Core Algorithm**: Main distributed K-means implementation with fault tolerance
//! - **Message Passing**: Communication infrastructure for worker coordination
//! - **Fault Tolerance**: Worker health monitoring and failure recovery mechanisms
//! - **Data Partitioning**: Strategies for distributing data across workers
//! - **Load Balancing**: Dynamic workload optimization and resource allocation
//! - **Performance Monitoring**: Comprehensive metrics collection and analysis
//!
//! ## Key Features
//!
//! - **Scalability**: Handle datasets that don't fit in memory on a single machine
//! - **Fault Tolerance**: Automatic recovery from worker failures with multiple strategies
//! - **Load Balancing**: Dynamic redistribution of work based on worker performance
//! - **Performance Monitoring**: Real-time metrics and optimization recommendations
//! - **Flexible Partitioning**: Multiple data distribution strategies (random, stratified, hash-based)
//! - **Convergence Detection**: Automatic detection of algorithm convergence
//! - **Checkpointing**: Periodic state saves for recovery from catastrophic failures
//!
//! ## Example Usage
//!
//! ```rust
//! use scirs2_cluster::distributed::{DistributedKMeans, DistributedKMeansConfig};
//! use ndarray::Array2;
//!
//! // Create sample data
//! let data = Array2::from_shape_vec((10000, 5), 
//!     (0..50000).map(|x| x as f64 / 1000.0).collect()).unwrap();
//!
//! // Configure distributed clustering
//! let config = DistributedKMeansConfig {
//!     max_iterations: 100,
//!     tolerance: 1e-4,
//!     n_workers: 8,
//!     enable_fault_tolerance: true,
//!     enable_load_balancing: true,
//!     enable_monitoring: true,
//!     verbose: true,
//!     ..Default::default()
//! };
//!
//! // Create and fit distributed K-means
//! let mut kmeans = DistributedKMeans::new(10, config)?;
//! let result = kmeans.fit(data.view())?;
//!
//! // Analyze results
//! println!("Clustering completed in {} iterations", result.n_iterations);
//! println!("Final inertia: {:.6}", result.inertia);
//! println!("Worker efficiency: {:.2}%", result.performance_stats.worker_efficiency * 100.0);
//! println!("Load balance score: {:.3}", result.performance_stats.load_balance_score);
//!
//! // Use the trained model for prediction
//! let new_data = Array2::from_shape_vec((100, 5), 
//!     (0..500).map(|x| x as f64 / 100.0).collect()).unwrap();
//! let predictions = kmeans.predict(new_data.view())?;
//! ```
//!
//! ## Advanced Configuration
//!
//! ```rust
//! use scirs2_cluster::distributed::{
//!     DistributedKMeans, DistributedKMeansConfig, InitializationMethod,
//!     PartitioningStrategy, LoadBalancingStrategy, OptimizationObjective
//! };
//!
//! let config = DistributedKMeansConfig {
//!     max_iterations: 200,
//!     tolerance: 1e-6,
//!     n_workers: 16,
//!     init_method: InitializationMethod::KMeansPlusPlus,
//!     enable_fault_tolerance: true,
//!     enable_load_balancing: true,
//!     enable_monitoring: true,
//!     convergence_check_interval: 5,
//!     checkpoint_interval: 20,
//!     verbose: true,
//!     random_seed: Some(42),
//! };
//!
//! let mut kmeans = DistributedKMeans::new(20, config)?;
//! 
//! // Configure advanced partitioning
//! let partition_config = PartitioningConfig {
//!     strategy: PartitioningStrategy::Stratified { n_strata: 8 },
//!     balance_threshold: 0.05,
//!     enable_load_balancing: true,
//!     preserve_locality: true,
//!     ..Default::default()
//! };
//!
//! // Configure load balancing strategy
//! let load_strategy = LoadBalancingStrategy::MultiObjective {
//!     objectives: vec![
//!         OptimizationObjective::MaximizeThroughput,
//!         OptimizationObjective::MinimizeCommunication,
//!         OptimizationObjective::MaximizeReliability,
//!     ],
//!     weights: vec![0.4, 0.3, 0.3],
//! };
//! ```

// Import all modular components
mod distributed;

// Re-export all components for backward compatibility
pub use distributed::*;

// Re-export individual modules for fine-grained access
pub use distributed::{
    core, message_passing, fault_tolerance, partitioning, 
    load_balancing, monitoring
};

// Convenient type aliases
pub type DistributedKMeansF64 = DistributedKMeans<f64>;
pub type DistributedKMeansF32 = DistributedKMeans<f32>;

// Legacy compatibility - these functions provide the same interface as the original monolithic implementation
use ndarray::{Array1, Array2, ArrayView2};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;
use crate::error::Result;

/// Legacy function: Perform distributed K-means clustering (backward compatibility)
pub fn distributed_kmeans<F>(
    data: ArrayView2<F>,
    k: usize,
    n_workers: usize,
    max_iterations: usize,
    tolerance: f64,
) -> Result<(Array2<F>, Array1<usize>, f64)>
where
    F: Float + FromPrimitive + Debug + Send + Sync + 'static,
{
    let config = DistributedKMeansConfig {
        max_iterations,
        tolerance,
        n_workers,
        ..Default::default()
    };

    let mut kmeans = DistributedKMeans::new(k, config)?;
    let result = kmeans.fit(data)?;

    Ok((result.centroids, result.labels, result.inertia))
}

/// Legacy function: Perform distributed K-means with custom initialization
pub fn distributed_kmeans_with_init<F>(
    data: ArrayView2<F>,
    initial_centroids: ArrayView2<F>,
    n_workers: usize,
    max_iterations: usize,
    tolerance: f64,
) -> Result<(Array2<F>, Array1<usize>, f64)>
where
    F: Float + FromPrimitive + Debug + Send + Sync + 'static,
{
    let k = initial_centroids.nrows();
    let custom_centroids = initial_centroids.to_owned();
    
    let config = DistributedKMeansConfig {
        max_iterations,
        tolerance,
        n_workers,
        init_method: InitializationMethod::Custom(
            custom_centroids.mapv(|x| x.to_f64().unwrap_or(0.0))
        ),
        ..Default::default()
    };

    let mut kmeans = DistributedKMeans::new(k, config)?;
    let result = kmeans.fit(data)?;

    Ok((result.centroids, result.labels, result.inertia))
}

/// Legacy function: Perform distributed K-means with fault tolerance
pub fn distributed_kmeans_fault_tolerant<F>(
    data: ArrayView2<F>,
    k: usize,
    n_workers: usize,
    max_iterations: usize,
    tolerance: f64,
    max_failures: usize,
) -> Result<(Array2<F>, Array1<usize>, f64, Vec<ConvergenceInfo>)>
where
    F: Float + FromPrimitive + Debug + Send + Sync + 'static,
{
    let config = DistributedKMeansConfig {
        max_iterations,
        tolerance,
        n_workers,
        enable_fault_tolerance: true,
        enable_monitoring: true,
        ..Default::default()
    };

    let mut kmeans = DistributedKMeans::new(k, config)?;
    let result = kmeans.fit(data)?;

    Ok((
        result.centroids,
        result.labels,
        result.inertia,
        kmeans.convergence_history().to_vec(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_legacy_distributed_kmeans() {
        let data = Array2::from_shape_vec(
            (100, 3),
            (0..300).map(|x| x as f64 / 10.0).collect(),
        ).unwrap();

        let result = distributed_kmeans(data.view(), 3, 2, 50, 1e-4);
        assert!(result.is_ok());

        let (centroids, labels, inertia) = result.unwrap();
        assert_eq!(centroids.nrows(), 3);
        assert_eq!(centroids.ncols(), 3);
        assert_eq!(labels.len(), 100);
        assert!(inertia >= 0.0);
    }

    #[test]
    fn test_legacy_distributed_kmeans_with_init() {
        let data = Array2::from_shape_vec(
            (50, 2),
            (0..100).map(|x| x as f64).collect(),
        ).unwrap();

        let initial_centroids = Array2::from_shape_vec(
            (2, 2),
            vec![0.0, 0.0, 10.0, 10.0],
        ).unwrap();

        let result = distributed_kmeans_with_init(
            data.view(),
            initial_centroids.view(),
            2,
            30,
            1e-3,
        );
        assert!(result.is_ok());

        let (centroids, labels, inertia) = result.unwrap();
        assert_eq!(centroids.nrows(), 2);
        assert_eq!(labels.len(), 50);
        assert!(inertia >= 0.0);
    }

    #[test]
    fn test_legacy_fault_tolerant_kmeans() {
        let data = Array2::from_shape_vec(
            (80, 2),
            (0..160).map(|x| (x as f64).sin()).collect(),
        ).unwrap();

        let result = distributed_kmeans_fault_tolerant(
            data.view(),
            4,
            3,
            40,
            1e-4,
            1,
        );
        assert!(result.is_ok());

        let (centroids, labels, inertia, convergence_history) = result.unwrap();
        assert_eq!(centroids.nrows(), 4);
        assert_eq!(labels.len(), 80);
        assert!(inertia >= 0.0);
        assert!(!convergence_history.is_empty());
    }

    #[test]
    fn test_distributed_kmeans_config_validation() {
        // Test invalid k
        let result = DistributedKMeans::<f64>::new(0, DistributedKMeansConfig::default());
        assert!(result.is_err());

        // Test invalid n_workers
        let config = DistributedKMeansConfig {
            n_workers: 0,
            ..Default::default()
        };
        let result = DistributedKMeans::<f64>::new(3, config);
        assert!(result.is_err());

        // Test valid configuration
        let result = DistributedKMeans::<f64>::new(3, DistributedKMeansConfig::default());
        assert!(result.is_ok());
    }

    #[test]
    fn test_modular_imports() {
        // Test that all modules are accessible
        use super::core::DistributedKMeans;
        use super::message_passing::MessagePassingCoordinator;
        use super::fault_tolerance::FaultToleranceCoordinator;
        use super::partitioning::DataPartitioner;
        use super::load_balancing::LoadBalancingCoordinator;
        use super::monitoring::PerformanceMonitor;

        // If compilation succeeds, the imports work correctly
        assert!(true);
    }
}