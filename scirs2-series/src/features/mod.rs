//! Time series feature extraction modules
//!
//! This module provides a modular approach to time series feature extraction,
//! organized by functionality for better maintainability and performance.

pub mod complexity;
pub mod config;
pub mod frequency;
pub mod statistical;
pub mod temporal;
pub mod turning_points;
pub mod utils;
pub mod wavelet;
pub mod window_based;

// Re-export commonly used items for convenience
pub use complexity::*;
pub use config::*;
pub use frequency::*;
pub use statistical::*;
pub use temporal::*;
pub use turning_points::*;
pub use utils::*;
pub use wavelet::*;
pub use window_based::*;
