//! Error types for the ML optimization module

use std::error::Error;
use std::fmt;

/// Error type for ML optimization operations
#[derive(Debug)]
pub enum OptimError {
    /// Invalid configuration
    InvalidConfig(String),
    /// Optimization error
    OptimizationError(String),
    /// Dimension mismatch error
    DimensionMismatch(String),
    /// Privacy budget exhausted
    PrivacyBudgetExhausted {
        consumed_epsilon: f64,
        target_epsilon: f64,
    },
    /// Invalid privacy configuration
    InvalidPrivacyConfig(String),
    /// Privacy accounting error
    PrivacyAccountingError(String),
    /// Other error
    Other(String),
}

/// Alias for backward compatibility
pub type OptimizerError = OptimError;

impl fmt::Display for OptimError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            OptimError::InvalidConfig(msg) => write!(f, "Invalid configuration: {}", msg),
            OptimError::OptimizationError(msg) => write!(f, "Optimization error: {}", msg),
            OptimError::DimensionMismatch(msg) => write!(f, "Dimension mismatch: {}", msg),
            OptimError::PrivacyBudgetExhausted {
                consumed_epsilon,
                target_epsilon,
            } => {
                write!(
                    f,
                    "Privacy budget exhausted: consumed ε={:.4}, target ε={:.4}",
                    consumed_epsilon, target_epsilon
                )
            }
            OptimError::InvalidPrivacyConfig(msg) => {
                write!(f, "Invalid privacy configuration: {}", msg)
            }
            OptimError::PrivacyAccountingError(msg) => {
                write!(f, "Privacy accounting error: {}", msg)
            }
            OptimError::Other(msg) => write!(f, "Error: {}", msg),
        }
    }
}

impl Error for OptimError {}

/// Result type for ML optimization operations
pub type Result<T> = std::result::Result<T, OptimError>;
