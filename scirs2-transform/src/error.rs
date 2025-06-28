//! Error types for the data transformation module

use thiserror::Error;

/// Error type for data transformation operations
#[derive(Error, Debug)]
pub enum TransformError {
    /// Invalid input data
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Transformation error
    #[error("Transformation error: {0}")]
    TransformationError(String),

    /// Core error
    #[error("Core error: {0}")]
    CoreError(#[from] scirs2_core::error::CoreError),

    /// Linear algebra error
    #[error("Linear algebra error: {0}")]
    LinalgError(#[from] scirs2_linalg::error::LinalgError),

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Computation error
    #[error("Computation error: {0}")]
    ComputationError(String),

    /// Model not fitted error
    #[error("Model not fitted: {0}")]
    NotFitted(String),

    /// Feature not enabled error
    #[error("Feature not enabled: {0}")]
    FeatureNotEnabled(String),

    /// GPU error
    #[error("GPU error: {0}")]
    GpuError(String),

    /// Distributed processing error
    #[error("Distributed processing error: {0}")]
    DistributedError(String),

    /// Monitoring error
    #[error("Monitoring error: {0}")]
    MonitoringError(String),

    /// Memory allocation error
    #[error("Memory error: {0}")]
    MemoryError(String),

    /// Convergence failure in iterative algorithms
    #[error("Convergence error: {0}")]
    ConvergenceError(String),

    /// Data quality or validation error
    #[error("Data validation error: {0}")]
    DataValidationError(String),

    /// Threading or parallel processing error
    #[error("Parallel processing error: {0}")]
    ParallelError(String),

    /// Configuration validation error
    #[error("Configuration error: {0}")]
    ConfigurationError(String),

    /// Timeout error for long-running operations
    #[error("Timeout error: {0}")]
    TimeoutError(String),

    /// SIMD operation error
    #[error("SIMD error: {0}")]
    SimdError(String),

    /// Streaming data pipeline error
    #[error("Streaming error: {0}")]
    StreamingError(String),

    /// Cross-validation error
    #[error("Cross-validation error: {0}")]
    CrossValidationError(String),

    /// Prometheus error
    #[cfg(feature = "monitoring")]
    #[error("Prometheus error: {0}")]
    PrometheusError(#[from] prometheus::Error),

    /// Serialization error
    #[cfg(feature = "distributed")]
    #[error("Serialization error: {0}")]
    SerializationError(#[from] bincode::Error),

    /// Not implemented error
    #[error("Not implemented: {0}")]
    NotImplemented(String),

    /// Parse error
    #[error("Parse error: {0}")]
    ParseError(String),

    /// Other error
    #[error("Error: {0}")]
    Other(String),
}

/// Result type for data transformation operations
pub type Result<T> = std::result::Result<T, TransformError>;
