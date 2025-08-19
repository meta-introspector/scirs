//! Neuromorphic Computing Algorithms
//!
//! This module provides implementations of neuromorphic computing algorithms for
//! spatial data processing, including spiking neural networks, competitive learning,
//! and memristive computing approaches.

pub mod spiking_clustering;
pub mod competitive_learning;
pub mod memristive_learning;
pub mod processing;

// Re-export main algorithm structures
pub use spiking_clustering::SpikingNeuralClusterer;
pub use competitive_learning::{CompetitiveNeuralClusterer, HomeostaticNeuralClusterer};
pub use memristive_learning::AdvancedMemristiveLearning;
pub use processing::NeuromorphicProcessor;