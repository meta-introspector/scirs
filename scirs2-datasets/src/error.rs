//! Error types for the datasets module

use std::io;
use thiserror::Error;

/// Error type for datasets operations
#[derive(Error, Debug)]
pub enum DatasetsError {
    /// Invalid data format
    #[error("Invalid format: {0}")]
    InvalidFormat(String),

    /// Data loading error
    #[error("Loading error: {0}")]
    LoadingError(String),

    /// Format error
    #[error("Format error: {0}")]
    FormatError(String),

    /// Not found error
    #[error("Not found: {0}")]
    NotFound(String),

    /// Authentication error
    #[error("Authentication error: {0}")]
    AuthenticationError(String),

    /// Download error
    #[error("Download error: {0}")]
    DownloadError(String),

    /// Cache error
    #[error("Cache error: {0}")]
    CacheError(String),

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] io::Error),

    /// Serialization/Deserialization error
    #[error("Serialization error: {0}")]
    SerdeError(String),

    /// Other error
    #[error("Error: {0}")]
    Other(String),
}

/// Result type for datasets operations
pub type Result<T> = std::result::Result<T, DatasetsError>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;

    #[test]
    fn test_invalid_format_error() {
        let error = DatasetsError::InvalidFormat("test format".to_string());
        assert_eq!(error.to_string(), "Invalid format: test format");
    }

    #[test]
    fn test_loading_error() {
        let error = DatasetsError::LoadingError("test loading".to_string());
        assert_eq!(error.to_string(), "Loading error: test loading");
    }

    #[test]
    fn test_download_error() {
        let error = DatasetsError::DownloadError("test download".to_string());
        assert_eq!(error.to_string(), "Download error: test download");
    }

    #[test]
    fn test_cache_error() {
        let error = DatasetsError::CacheError("test cache".to_string());
        assert_eq!(error.to_string(), "Cache error: test cache");
    }

    #[test]
    fn test_io_error_conversion() {
        let io_error = io::Error::new(io::ErrorKind::NotFound, "file not found");
        let datasets_error: DatasetsError = io_error.into();

        match datasets_error {
            DatasetsError::IoError(_) => {
                assert!(datasets_error.to_string().contains("file not found"));
            }
            _ => panic!("Expected IoError variant"),
        }
    }

    #[test]
    fn test_serde_error() {
        let error = DatasetsError::SerdeError("serialization failed".to_string());
        assert_eq!(
            error.to_string(),
            "Serialization error: serialization failed"
        );
    }

    #[test]
    fn test_other_error() {
        let error = DatasetsError::Other("generic error".to_string());
        assert_eq!(error.to_string(), "Error: generic error");
    }

    #[test]
    fn test_error_debug_format() {
        let error = DatasetsError::InvalidFormat("debug test".to_string());
        let debug_str = format!("{:?}", error);
        assert!(debug_str.contains("InvalidFormat"));
        assert!(debug_str.contains("debug test"));
    }

    #[test]
    fn test_result_type() {
        // Test Ok case
        let ok_result: Result<i32> = Ok(42);
        assert_eq!(ok_result.unwrap(), 42);

        // Test Err case
        let err_result: Result<i32> = Err(DatasetsError::Other("test".to_string()));
        assert!(err_result.is_err());
    }

    #[test]
    fn test_error_from_io_error() {
        let io_err = io::Error::new(io::ErrorKind::PermissionDenied, "access denied");
        let datasets_err = DatasetsError::from(io_err);

        if let DatasetsError::IoError(ref inner) = datasets_err {
            assert_eq!(inner.kind(), io::ErrorKind::PermissionDenied);
        } else {
            panic!("Expected IoError variant");
        }
    }

    #[test]
    fn test_error_chain() {
        // Test that error displays work correctly in error chains
        let error = DatasetsError::LoadingError("failed to parse CSV".to_string());
        let result: Result<()> = Err(error);

        match result {
            Ok(_) => panic!("Expected error"),
            Err(e) => {
                assert_eq!(e.to_string(), "Loading error: failed to parse CSV");
            }
        }
    }
}
