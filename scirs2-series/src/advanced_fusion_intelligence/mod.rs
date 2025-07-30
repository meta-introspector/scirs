//! Advanced Fusion Intelligence Modules
//!
//! This module provides the modular implementation of the Advanced Fusion Intelligence
//! system, breaking down the large monolithic implementation into focused modules
//! for better maintainability and organization.

pub mod quantum;
pub mod neuromorphic;
pub mod meta_learning;
pub mod evolution;
pub mod consciousness;
pub mod temporal;
pub mod distributed;

// Re-export all public types for backward compatibility
pub use quantum::*;
pub use neuromorphic::*;
pub use meta_learning::*;
pub use evolution::*;
pub use consciousness::*;
pub use temporal::*;
pub use distributed::*;