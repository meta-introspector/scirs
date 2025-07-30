//! Quantum mechanics solvers and algorithms
//!
//! This module provides comprehensive quantum mechanics functionality including:
//! - Core quantum state representations and Schrödinger equation solvers
//! - Advanced quantum algorithms (annealing, VQE, error correction)
//! - Multi-particle entanglement systems and Bell states
//! - Advanced basis sets for quantum calculations
//! - GPU-accelerated quantum computations

pub mod core;
pub mod algorithms;
pub mod entanglement;
pub mod basis_sets;
pub mod gpu;

// Re-export all public types for backward compatibility
pub use core::*;
pub use algorithms::*;
pub use entanglement::*;
pub use basis_sets::*;
pub use gpu::*;