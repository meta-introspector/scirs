//! Error types for the IO module

use std::error::Error;
use std::fmt;

/// Error type for IO operations
#[derive(Debug)]
pub enum IoError {
    /// File error
    FileError(String),
    /// Format error
    FormatError(String),
    /// Serialization error
    SerializationError(String),
    /// Deserialization error
    DeserializationError(String),
    /// Compression error
    CompressionError(String),
    /// Decompression error
    DecompressionError(String),
    /// Unsupported compression algorithm
    UnsupportedCompressionAlgorithm(String),
    /// Unsupported format
    UnsupportedFormat(String),
    /// Conversion error
    ConversionError(String),
    /// File not found
    FileNotFound(String),
    /// Parse error
    ParseError(String),
    /// Standard I/O error
    Io(std::io::Error),
    /// Validation error
    ValidationError(String),
    /// Checksum error
    ChecksumError(String),
    /// Integrity error
    IntegrityError(String),
    /// Configuration error
    ConfigError(String),
    /// Network error
    NetworkError(String),
    /// Other error
    Other(String),
}

impl fmt::Display for IoError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            IoError::FileError(msg) => write!(f, "File error: {}", msg),
            IoError::FormatError(msg) => write!(f, "Format error: {}", msg),
            IoError::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
            IoError::DeserializationError(msg) => write!(f, "Deserialization error: {}", msg),
            IoError::CompressionError(msg) => write!(f, "Compression error: {}", msg),
            IoError::DecompressionError(msg) => write!(f, "Decompression error: {}", msg),
            IoError::UnsupportedCompressionAlgorithm(algo) => {
                write!(f, "Unsupported compression algorithm: {}", algo)
            }
            IoError::UnsupportedFormat(fmt) => write!(f, "Unsupported format: {}", fmt),
            IoError::ConversionError(msg) => write!(f, "Conversion error: {}", msg),
            IoError::FileNotFound(path) => write!(f, "File not found: {}", path),
            IoError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            IoError::Io(e) => write!(f, "I/O error: {}", e),
            IoError::ValidationError(msg) => write!(f, "Validation error: {}", msg),
            IoError::ChecksumError(msg) => write!(f, "Checksum error: {}", msg),
            IoError::IntegrityError(msg) => write!(f, "Integrity error: {}", msg),
            IoError::ConfigError(msg) => write!(f, "Configuration error: {}", msg),
            IoError::NetworkError(msg) => write!(f, "Network error: {}", msg),
            IoError::Other(msg) => write!(f, "Error: {}", msg),
        }
    }
}

impl Error for IoError {}

/// Result type for IO operations
pub type Result<T> = std::result::Result<T, IoError>;
