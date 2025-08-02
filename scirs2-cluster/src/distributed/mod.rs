//! Distributed clustering algorithms and infrastructure
//!
//! This module provides a comprehensive distributed clustering framework with
//! support for fault tolerance, load balancing, performance monitoring, and
//! various data partitioning strategies.
//!
//! ## Architecture
//!
//! The distributed clustering system is organized into several key components:
//!
//! - **Core**: Main distributed K-means algorithm implementation
//! - **Message Passing**: Communication infrastructure for worker coordination  
//! - **Fault Tolerance**: Worker health monitoring and failure recovery
//! - **Partitioning**: Data distribution strategies across workers
//! - **Load Balancing**: Dynamic workload optimization algorithms
//! - **Monitoring**: Performance metrics and system health analysis
//!
//! ## Example Usage
//!
//! ```rust
//! use scirs2_cluster::distributed::{DistributedKMeans, DistributedKMeansConfig};
//! use ndarray::Array2;
//!
//! // Create sample data
//! let data = Array2::fromshape_vec((1000, 2), (0..2000).map(|x| x as f64).collect()).unwrap();
//!
//! // Configure distributed clustering
//! let config = DistributedKMeansConfig {
//!     max_iterations: 100,
//!     n_workers: 4,
//!     enable_fault_tolerance: true,
//!     enable_load_balancing: true,
//!     ..Default::default()
//! };
//!
//! // Create and fit distributed K-means
//! let mut kmeans = DistributedKMeans::new(5, config).unwrap();
//! let result = kmeans.fit(data.view()).unwrap();
//!
//! println!("Clustering completed in {} iterations", result.n_iterations);
//! println!("Final inertia: {:.6}", result.inertia);
//! ```

pub mod core;
pub mod message_passing;
pub mod fault_tolerance;
pub mod partitioning;
pub mod load_balancing;
pub mod monitoring;

// Re-export main types for convenience
pub use core::{
    DistributedKMeans, DistributedKMeansConfig, ClusteringResult, 
    ConvergenceInfo, InitializationMethod, PerformanceStatistics
};

pub use message_passing::{
    MessagePassingCoordinator, ClusteringMessage, MessagePriority,
    RecoveryStrategy, MessageEnvelope, WorkerStatus as MessageWorkerStatus
};

pub use fault_tolerance::{
    FaultToleranceCoordinator, WorkerHealthInfo, WorkerStatus,
    FaultToleranceConfig, ClusteringCheckpoint, DataPartition
};

pub use partitioning::{
    DataPartitioner, PartitioningConfig, PartitioningStrategy,
    PartitioningStatistics
};

pub use load_balancing::{
    LoadBalancingCoordinator, WorkerProfile, LoadBalancingConfig,
    LoadBalancingStrategy, OptimizationObjective, LoadBalanceDecision
};

pub use monitoring::{
    PerformanceMonitor, MonitoringConfig, PerformanceMetrics,
    ResourceUsage, WorkerMetrics, PerformanceAlert, AlertType,
    AlertSeverity, EfficiencyAnalysis, MonitoringReport
};

/// Convenient type alias for f64-based distributed K-means
pub type DistributedKMeansF64 = DistributedKMeans<f64>;

/// Convenient type alias for f32-based distributed K-means  
pub type DistributedKMeansF32 = DistributedKMeans<f32>;