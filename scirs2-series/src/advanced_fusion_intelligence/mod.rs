//! Advanced Fusion Intelligence Modules
//!
//! This module provides the modular implementation of the Advanced Fusion Intelligence
//! system, breaking down the large monolithic implementation into focused modules
//! for better maintainability and organization.

pub mod consciousness;
pub mod distributed;
pub mod evolution;
pub mod meta_learning;
pub mod neuromorphic;
pub mod quantum;
pub mod temporal;

// Re-export all public types for backward compatibility
pub use consciousness::*;
pub use distributed::*;
pub use evolution::*;
pub use meta_learning::*;
pub use neuromorphic::*;
pub use quantum::*;
pub use temporal::*;
