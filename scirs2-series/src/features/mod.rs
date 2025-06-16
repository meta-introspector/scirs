//! Time series feature extraction modules
//!
//! This module provides a modular approach to time series feature extraction,
//! organized by functionality for better maintainability and performance.

pub mod config;
pub mod utils;
pub mod statistical;
pub mod complexity;
pub mod frequency;
pub mod temporal;
pub mod wavelet;
pub mod turning_points;
pub mod window_based;

// Re-export commonly used items for convenience
pub use config::*;
pub use utils::*;
pub use statistical::*;
pub use complexity::*;
pub use frequency::*;
pub use temporal::*;
pub use wavelet::*;
pub use turning_points::*;
pub use window_based::*;