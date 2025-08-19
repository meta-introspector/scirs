//! Quantum-Inspired Algorithms
//!
//! This module provides implementations of quantum-inspired algorithms for spatial
//! computing, including clustering, search, optimization, and machine learning algorithms
//! that leverage quantum computing principles for enhanced performance.

pub mod quantum_clustering;
pub mod quantum_search;
pub mod quantum_optimization;

// Export main algorithm structures
pub use quantum_clustering::QuantumClusterer;
pub use quantum_search::QuantumNearestNeighbor;
pub use quantum_optimization::QuantumSpatialOptimizer;

// TODO: Add these modules as they are implemented
// pub mod quantum_machine_learning;