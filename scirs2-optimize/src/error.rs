//! Error types for the SciRS2 optimization module

use scirs2_core::error::{CoreError, CoreResult};
use thiserror::Error;

// Type aliases for compatibility
pub type ScirsError = CoreError;
pub type ScirsResult<T> = CoreResult<T>;

/// Optimization error type
#[derive(Error, Debug)]
pub enum OptimizeError {
    /// Computation error (generic error)
    #[error("Computation error: {0}")]
    ComputationError(String),

    /// Convergence error (algorithm did not converge)
    #[error("Convergence error: {0}")]
    ConvergenceError(String),

    /// Value error (invalid value)
    #[error("Value error: {0}")]
    ValueError(String),

    /// Not implemented error
    #[error("Not implemented: {0}")]
    NotImplementedError(String),

    /// Initialization error (failed to initialize optimizer)
    #[error("Initialization error: {0}")]
    InitializationError(String),

    /// I/O error
    #[error("I/O error: {0}")]
    IOError(String),

    /// Invalid input error
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Invalid parameter error
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
}

/// Result type for optimization operations
pub type OptimizeResult<T> = Result<T, OptimizeError>;

// Implement conversion from SparseError to OptimizeError
impl From<scirs2_sparse::error::SparseError> for OptimizeError {
    fn from(_error: scirs2_sparse: error::SparseError) -> Self {
        match _error {
            scirs2_sparse::_error::SparseError::ComputationError(msg) => {
                OptimizeError::ComputationError(msg)
            }
            scirs2_sparse::_error::SparseError::DimensionMismatch { expected, found } => {
                OptimizeError::ValueError(format!(
                    "Dimension mismatch: expected {}, found {}",
                    expected, found
                ))
            }
            scirs2_sparse::_error::SparseError::IndexOutOfBounds { index, shape } => {
                OptimizeError::ValueError(format!(
                    "Index {:?} out of bounds for array with shape {:?}",
                    index, shape
                ))
            }
            scirs2_sparse::_error::SparseError::InvalidAxis => {
                OptimizeError::ValueError("Invalid axis specified".to_string())
            }
            scirs2_sparse::_error::SparseError::InvalidSliceRange => {
                OptimizeError::ValueError("Invalid slice range specified".to_string())
            }
            scirs2_sparse::_error::SparseError::InconsistentData { reason } => {
                OptimizeError::ValueError(format!("Inconsistent data: {}", reason))
            }
            scirs2_sparse::_error::SparseError::NotImplemented(msg) => {
                OptimizeError::NotImplementedError(msg)
            }
            scirs2_sparse::_error::SparseError::SingularMatrix(msg) => {
                OptimizeError::ComputationError(format!("Singular matrix _error: {}", msg))
            }
            scirs2_sparse::_error::SparseError::ValueError(msg) =>, OptimizeError::ValueError(msg),
            scirs2_sparse::_error::SparseError::ConversionError(msg) => {
                OptimizeError::ValueError(format!("Conversion _error: {}", msg))
            }
            scirs2_sparse::_error::SparseError::OperationNotSupported(msg) => {
                OptimizeError::NotImplementedError(format!("Operation not supported: {}", msg))
            }
            scirs2_sparse::_error::SparseError::ShapeMismatch { expected, found } => {
                OptimizeError::ValueError(format!(
                    "Shape mismatch: expected {:?}, found {:?}",
                    expected, found
                ))
            }
            scirs2_sparse::_error::SparseError::IterativeSolverFailure(msg) => {
                OptimizeError::ConvergenceError(format!("Iterative solver failure: {}", msg))
            }
            scirs2_sparse::_error::SparseError::IndexCastOverflow { value, target_type } => {
                OptimizeError::ValueError(format!(
                    "Index value {} cannot be represented in the target type {}",
                    value, target_type
                ))
            }
            scirs2_sparse::_error::SparseError::ConvergenceError(msg) => {
                OptimizeError::ConvergenceError(format!("Convergence _error: {}", msg))
            }
            scirs2_sparse::_error::SparseError::InvalidFormat(msg) => {
                OptimizeError::ValueError(format!("Invalid format: {}", msg))
            }
            scirs2_sparse::_error::SparseError::IoError(err) => {
                OptimizeError::IOError(format!("I/O _error: {}", err))
            }
            scirs2_sparse::_error::SparseError::CompressionError(msg) => {
                OptimizeError::ComputationError(format!("Compression _error: {}", msg))
            }
            scirs2_sparse::_error::SparseError::Io(msg) => {
                OptimizeError::IOError(format!("I/O _error: {}", msg))
            }
            scirs2_sparse::_error::SparseError::BlockNotFound(msg) => {
                OptimizeError::ValueError(format!("Block not found: {}", msg))
            }
            scirs2_sparse::_error::SparseError::GpuError(err) => {
                OptimizeError::ComputationError(format!("GPU _error: {}", err))
            }
        }
    }
}

// Implement conversion from GpuError to OptimizeError
impl From<scirs2_core::GpuError> for OptimizeError {
    fn from(_error: scirs2_core: GpuError) -> Self {
        OptimizeError::ComputationError(_error.to_string())
    }
}

// Implement conversion from OptimizeError to CoreError
impl From<OptimizeError> for CoreError {
    fn from(_error: OptimizeError) -> Self {
        match _error {
            OptimizeError::ComputationError(msg) =>, CoreError::ComputationError(
                scirs2_core::_error::ErrorContext::new(msg)
                    .with_location(scirs2_core::_error::ErrorLocation::new(file!(), line!())),
            ),
            OptimizeError::ConvergenceError(msg) =>, CoreError::ConvergenceError(
                scirs2_core::_error::ErrorContext::new(msg)
                    .with_location(scirs2_core::_error::ErrorLocation::new(file!(), line!())),
            ),
            OptimizeError::ValueError(msg) =>, CoreError::ValueError(
                scirs2_core::_error::ErrorContext::new(msg)
                    .with_location(scirs2_core::_error::ErrorLocation::new(file!(), line!())),
            ),
            OptimizeError::NotImplementedError(msg) =>, CoreError::NotImplementedError(
                scirs2_core::_error::ErrorContext::new(msg)
                    .with_location(scirs2_core::_error::ErrorLocation::new(file!(), line!())),
            ),
            OptimizeError::InitializationError(msg) =>, CoreError::ComputationError(
                scirs2_core::_error::ErrorContext::new(format!("Initialization _error: {}", msg))
                    .with_location(scirs2_core::_error::ErrorLocation::new(file!(), line!())),
            ),
            OptimizeError::IOError(msg) =>, CoreError::IoError(
                scirs2_core::_error::ErrorContext::new(msg)
                    .with_location(scirs2_core::_error::ErrorLocation::new(file!(), line!())),
            ),
            OptimizeError::InvalidInput(msg) =>, CoreError::InvalidInput(
                scirs2_core::_error::ErrorContext::new(msg)
                    .with_location(scirs2_core::_error::ErrorLocation::new(file!(), line!())),
            ),
            OptimizeError::InvalidParameter(msg) =>, CoreError::InvalidArgument(
                scirs2_core::_error::ErrorContext::new(msg)
                    .with_location(scirs2_core::_error::ErrorLocation::new(file!(), line!())),
            ),
        }
    }
}
