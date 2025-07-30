//! Advanced analysis tools for dynamical systems
//!
//! This module provides tools for analyzing the behavior of dynamical systems,
//! including bifurcation analysis and stability assessment.

pub mod types;
pub mod bifurcation;
pub mod stability;
pub mod advanced;
pub mod ml_prediction;
pub mod utils;

// Re-export all types and analyzers for backward compatibility
pub use types::*;
pub use bifurcation::BifurcationAnalyzer;
pub use stability::StabilityAnalyzer;