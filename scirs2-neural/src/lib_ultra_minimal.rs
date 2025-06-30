//! Neural network building blocks module for SciRS2 - Ultra minimal working version
//!
//! This is an ultra minimal version that includes only the error module
//! to establish basic compilation.

#![warn(missing_docs)]

// Ultra minimal - just error handling
pub mod error;

// Re-export the error type
pub use error::{Error, NeuralError, Result};