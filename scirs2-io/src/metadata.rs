//! Advanced metadata management for scientific data
//!
//! Provides comprehensive metadata handling across different file formats with
//! unified interfaces for storing, retrieving, and transforming metadata.

use crate::error::{IoError, Result};
use chrono::{DateTime, Utc};
use indexmap::{IndexMap, indexmap};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::path::Path;

/// Standard metadata keys commonly used across scientific data formats
pub mod standard_keys {
    pub const TITLE: &str = "title";
    pub const AUTHOR: &str = "author";
    pub const DESCRIPTION: &str = "description";
    pub const CREATION_DATE: &str = "creation_date";
    pub const MODIFICATION_DATE: &str = "modification_date";
    pub const VERSION: &str = "version";
    pub const LICENSE: &str = "license";
    pub const KEYWORDS: &str = "keywords";
    pub const UNITS: &str = "units";
    pub const DIMENSIONS: &str = "dimensions";
    pub const COORDINATE_SYSTEM: &str = "coordinate_system";
    pub const INSTRUMENT: &str = "instrument";
    pub const EXPERIMENT: &str = "experiment";
    pub const PROCESSING_HISTORY: &str = "processing_history";
    pub const REFERENCES: &str = "references";
    pub const PROVENANCE: &str = "provenance";
}

/// Metadata value types supporting various data formats
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MetadataValue {
    /// String value
    String(String),
    /// Integer value
    Integer(i64),
    /// Floating point value
    Float(f64),
    /// Boolean value
    Boolean(bool),
    /// Date/time value
    DateTime(DateTime<Utc>),
    /// Array of values
    Array(Vec<MetadataValue>),
    /// Nested metadata object
    Object(IndexMap<String, MetadataValue>),
    /// Binary data
    Binary(Vec<u8>),
}

impl fmt::Display for MetadataValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::String(s) => write!(f, "{}", s),
            Self::Integer(i) => write!(f, "{}", i),
            Self::Float(fl) => write!(f, "{}", fl),
            Self::Boolean(b) => write!(f, "{}", b),
            Self::DateTime(dt) => write!(f, "{}", dt.to_rfc3339()),
            Self::Array(arr) => write!(f, "[{}]", arr.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ")),
            Self::Object(_) => write!(f, "[object]"),
            Self::Binary(b) => write!(f, "[binary: {} bytes]", b.len()),
        }
    }
}

/// Advanced metadata container with rich functionality
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Metadata {
    /// Core metadata stored in insertion order
    data: IndexMap<String, MetadataValue>,
    /// Format-specific metadata extensions
    extensions: HashMap<String, IndexMap<String, MetadataValue>>,
    /// Metadata schema version
    schema_version: String,
}

impl Metadata {
    /// Create a new empty metadata container
    pub fn new() -> Self {
        Self {
            data: IndexMap::new(),
            extensions: HashMap::new(),
            schema_version: "1.0".to_string(),
        }
    }

    /// Create metadata with a specific schema version
    pub fn with_schema(version: &str) -> Self {
        Self {
            data: IndexMap::new(),
            extensions: HashMap::new(),
            schema_version: version.to_string(),
        }
    }

    /// Set a metadata value
    pub fn set(&mut self, key: impl Into<String>, value: impl Into<MetadataValue>) {
        self.data.insert(key.into(), value.into());
    }

    /// Get a metadata value
    pub fn get(&self, key: &str) -> Option<&MetadataValue> {
        self.data.get(key)
    }

    /// Get a typed metadata value
    pub fn get_string(&self, key: &str) -> Option<&str> {
        match self.get(key)? {
            MetadataValue::String(s) => Some(s),
            _ => None,
        }
    }

    /// Get integer metadata value
    pub fn get_integer(&self, key: &str) -> Option<i64> {
        match self.get(key)? {
            MetadataValue::Integer(i) => Some(*i),
            _ => None,
        }
    }

    /// Get float metadata value
    pub fn get_float(&self, key: &str) -> Option<f64> {
        match self.get(key)? {
            MetadataValue::Float(f) => Some(*f),
            MetadataValue::Integer(i) => Some(*i as f64),
            _ => None,
        }
    }

    /// Set format-specific extension metadata
    pub fn set_extension(&mut self, format: &str, key: impl Into<String>, value: impl Into<MetadataValue>) {
        self.extensions
            .entry(format.to_string())
            .or_insert_with(IndexMap::new)
            .insert(key.into(), value.into());
    }

    /// Get format-specific extension metadata
    pub fn get_extension(&self, format: &str) -> Option<&IndexMap<String, MetadataValue>> {
        self.extensions.get(format)
    }

    /// Merge metadata from another container
    pub fn merge(&mut self, other: &Metadata) {
        for (key, value) in &other.data {
            self.data.insert(key.clone(), value.clone());
        }
        for (format, ext_data) in &other.extensions {
            let ext = self.extensions.entry(format.clone()).or_insert_with(IndexMap::new);
            for (key, value) in ext_data {
                ext.insert(key.clone(), value.clone());
            }
        }
    }

    /// Validate metadata against a schema
    pub fn validate(&self, schema: &MetadataSchema) -> Result<()> {
        schema.validate(self)
    }

    /// Convert metadata to a specific format
    pub fn to_format(&self, format: MetadataFormat) -> Result<String> {
        match format {
            MetadataFormat::Json => serde_json::to_string_pretty(self)
                .map_err(|e| IoError::SerializationError(e.to_string())),
            MetadataFormat::Yaml => serde_yaml::to_string(self)
                .map_err(|e| IoError::SerializationError(e.to_string())),
            MetadataFormat::Toml => toml::to_string_pretty(self)
                .map_err(|e| IoError::SerializationError(e.to_string())),
        }
    }

    /// Load metadata from a file
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path)
            .map_err(|e| IoError::Io(e))?;
        
        let extension = path.extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("");
        
        match extension {
            "json" => serde_json::from_str(&content)
                .map_err(|e| IoError::SerializationError(e.to_string())),
            "yaml" | "yml" => serde_yaml::from_str(&content)
                .map_err(|e| IoError::SerializationError(e.to_string())),
            "toml" => toml::from_str(&content)
                .map_err(|e| IoError::SerializationError(e.to_string())),
            _ => Err(IoError::UnsupportedFormat(format!("Unknown metadata format: {}", extension))),
        }
    }

    /// Save metadata to a file
    pub fn to_file(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let extension = path.extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("");
        
        let format = match extension {
            "json" => MetadataFormat::Json,
            "yaml" | "yml" => MetadataFormat::Yaml,
            "toml" => MetadataFormat::Toml,
            _ => return Err(IoError::UnsupportedFormat(format!("Unknown metadata format: {}", extension))),
        };
        
        let content = self.to_format(format)?;
        std::fs::write(path, content)
            .map_err(|e| IoError::Io(e))
    }

    /// Add processing history entry
    pub fn add_processing_history(&mut self, entry: ProcessingHistoryEntry) {
        let history = match self.data.get_mut(standard_keys::PROCESSING_HISTORY) {
            Some(MetadataValue::Array(arr)) => arr,
            _ => {
                self.data.insert(
                    standard_keys::PROCESSING_HISTORY.to_string(),
                    MetadataValue::Array(Vec::new()),
                );
                match self.data.get_mut(standard_keys::PROCESSING_HISTORY).unwrap() {
                    MetadataValue::Array(arr) => arr,
                    _ => unreachable!(),
                }
            }
        };
        
        let entry_obj = indexmap! {
            "timestamp".to_string() => MetadataValue::DateTime(entry.timestamp),
            "operation".to_string() => MetadataValue::String(entry.operation),
            "parameters".to_string() => MetadataValue::Object(entry.parameters),
            "user".to_string() => MetadataValue::String(entry.user.unwrap_or_else(|| "unknown".to_string())),
        };
        
        history.push(MetadataValue::Object(entry_obj));
    }

    /// Update modification timestamp
    pub fn update_modification_date(&mut self) {
        self.set(standard_keys::MODIFICATION_DATE, MetadataValue::DateTime(Utc::now()));
    }
}

/// Processing history entry for tracking data transformations
#[derive(Debug, Clone)]
pub struct ProcessingHistoryEntry {
    pub timestamp: DateTime<Utc>,
    pub operation: String,
    pub parameters: IndexMap<String, MetadataValue>,
    pub user: Option<String>,
}

impl ProcessingHistoryEntry {
    pub fn new(operation: impl Into<String>) -> Self {
        Self {
            timestamp: Utc::now(),
            operation: operation.into(),
            parameters: IndexMap::new(),
            user: std::env::var("USER").ok(),
        }
    }

    pub fn with_parameter(mut self, key: impl Into<String>, value: impl Into<MetadataValue>) -> Self {
        self.parameters.insert(key.into(), value.into());
        self
    }
}

/// Metadata output formats
#[derive(Debug, Clone, Copy)]
pub enum MetadataFormat {
    Json,
    Yaml,
    Toml,
}

/// Metadata schema for validation
#[derive(Debug, Clone)]
pub struct MetadataSchema {
    required_fields: Vec<String>,
    field_types: HashMap<String, MetadataFieldType>,
    constraints: Vec<MetadataConstraint>,
}

#[derive(Debug, Clone)]
pub enum MetadataFieldType {
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
    Array(Box<MetadataFieldType>),
    Object,
}

#[derive(Debug, Clone)]
pub enum MetadataConstraint {
    MinValue(String, f64),
    MaxValue(String, f64),
    Pattern(String, String),
    OneOf(String, Vec<MetadataValue>),
}

impl MetadataSchema {
    pub fn new() -> Self {
        Self {
            required_fields: Vec::new(),
            field_types: HashMap::new(),
            constraints: Vec::new(),
        }
    }

    pub fn require(mut self, field: impl Into<String>) -> Self {
        self.required_fields.push(field.into());
        self
    }

    pub fn field_type(mut self, field: impl Into<String>, field_type: MetadataFieldType) -> Self {
        self.field_types.insert(field.into(), field_type);
        self
    }

    pub fn constraint(mut self, constraint: MetadataConstraint) -> Self {
        self.constraints.push(constraint);
        self
    }

    pub fn validate(&self, metadata: &Metadata) -> Result<()> {
        // Check required fields
        for field in &self.required_fields {
            if metadata.get(field).is_none() {
                return Err(IoError::ValidationError(format!("Required field '{}' is missing", field)));
            }
        }

        // Validate field types
        for (field, expected_type) in &self.field_types {
            if let Some(value) = metadata.get(field) {
                if !self.validate_type(value, expected_type) {
                    return Err(IoError::ValidationError(
                        format!("Field '{}' has incorrect type", field)
                    ));
                }
            }
        }

        // Apply constraints
        for constraint in &self.constraints {
            self.apply_constraint(metadata, constraint)?;
        }

        Ok(())
    }

    fn validate_type(&self, value: &MetadataValue, expected: &MetadataFieldType) -> bool {
        match (value, expected) {
            (MetadataValue::String(_), MetadataFieldType::String) => true,
            (MetadataValue::Integer(_), MetadataFieldType::Integer) => true,
            (MetadataValue::Float(_), MetadataFieldType::Float) => true,
            (MetadataValue::Boolean(_), MetadataFieldType::Boolean) => true,
            (MetadataValue::DateTime(_), MetadataFieldType::DateTime) => true,
            (MetadataValue::Array(arr), MetadataFieldType::Array(elem_type)) => {
                arr.iter().all(|v| self.validate_type(v, elem_type))
            }
            (MetadataValue::Object(_), MetadataFieldType::Object) => true,
            _ => false,
        }
    }

    fn apply_constraint(&self, metadata: &Metadata, constraint: &MetadataConstraint) -> Result<()> {
        match constraint {
            MetadataConstraint::MinValue(field, min) => {
                if let Some(val) = metadata.get_float(field) {
                    if val < *min {
                        return Err(IoError::ValidationError(
                            format!("Field '{}' value {} is less than minimum {}", field, val, min)
                        ));
                    }
                }
            }
            MetadataConstraint::MaxValue(field, max) => {
                if let Some(val) = metadata.get_float(field) {
                    if val > *max {
                        return Err(IoError::ValidationError(
                            format!("Field '{}' value {} is greater than maximum {}", field, val, max)
                        ));
                    }
                }
            }
            MetadataConstraint::Pattern(field, pattern) => {
                if let Some(val) = metadata.get_string(field) {
                    let re = regex::Regex::new(pattern)
                        .map_err(|e| IoError::ValidationError(format!("Invalid regex pattern: {}", e)))?;
                    if !re.is_match(val) {
                        return Err(IoError::ValidationError(
                            format!("Field '{}' value '{}' does not match pattern '{}'", field, val, pattern)
                        ));
                    }
                }
            }
            MetadataConstraint::OneOf(field, allowed) => {
                if let Some(val) = metadata.get(field) {
                    if !allowed.contains(val) {
                        return Err(IoError::ValidationError(
                            format!("Field '{}' value is not in allowed set", field)
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

/// Metadata transformer for converting between different metadata formats
pub struct MetadataTransformer {
    mappings: HashMap<String, String>,
    transformations: HashMap<String, Box<dyn Fn(&MetadataValue) -> MetadataValue>>,
}

impl MetadataTransformer {
    pub fn new() -> Self {
        Self {
            mappings: HashMap::new(),
            transformations: HashMap::new(),
        }
    }

    /// Add a field mapping
    pub fn map_field(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        self.mappings.insert(from.into(), to.into());
        self
    }

    /// Transform metadata using configured mappings
    pub fn transform(&self, input: &Metadata) -> Metadata {
        let mut output = Metadata::new();
        output.schema_version = input.schema_version.clone();

        for (key, value) in &input.data {
            let new_key = self.mappings.get(key).cloned().unwrap_or_else(|| key.clone());
            let new_value = if let Some(transform) = self.transformations.get(key) {
                transform(value)
            } else {
                value.clone()
            };
            output.set(new_key, new_value);
        }

        output.extensions = input.extensions.clone();
        output
    }
}

/// Predefined metadata schemas for common scientific data types
pub mod schemas {
    use super::*;

    /// Schema for image metadata
    pub fn image_schema() -> MetadataSchema {
        MetadataSchema::new()
            .require("width")
            .require("height")
            .field_type("width", MetadataFieldType::Integer)
            .field_type("height", MetadataFieldType::Integer)
            .field_type("channels", MetadataFieldType::Integer)
            .field_type("bit_depth", MetadataFieldType::Integer)
            .constraint(MetadataConstraint::MinValue("width".to_string(), 1.0))
            .constraint(MetadataConstraint::MinValue("height".to_string(), 1.0))
    }

    /// Schema for time series metadata
    pub fn time_series_schema() -> MetadataSchema {
        MetadataSchema::new()
            .require("start_time")
            .require("sampling_rate")
            .field_type("start_time", MetadataFieldType::DateTime)
            .field_type("sampling_rate", MetadataFieldType::Float)
            .field_type("units", MetadataFieldType::String)
            .constraint(MetadataConstraint::MinValue("sampling_rate".to_string(), 0.0))
    }

    /// Schema for geospatial metadata
    pub fn geospatial_schema() -> MetadataSchema {
        MetadataSchema::new()
            .require("coordinate_system")
            .field_type("coordinate_system", MetadataFieldType::String)
            .field_type("bounds", MetadataFieldType::Array(Box::new(MetadataFieldType::Float)))
            .field_type("projection", MetadataFieldType::String)
    }
}

impl From<String> for MetadataValue {
    fn from(s: String) -> Self {
        MetadataValue::String(s)
    }
}

impl From<&str> for MetadataValue {
    fn from(s: &str) -> Self {
        MetadataValue::String(s.to_string())
    }
}

impl From<i64> for MetadataValue {
    fn from(i: i64) -> Self {
        MetadataValue::Integer(i)
    }
}

impl From<f64> for MetadataValue {
    fn from(f: f64) -> Self {
        MetadataValue::Float(f)
    }
}

impl From<bool> for MetadataValue {
    fn from(b: bool) -> Self {
        MetadataValue::Boolean(b)
    }
}

impl From<DateTime<Utc>> for MetadataValue {
    fn from(dt: DateTime<Utc>) -> Self {
        MetadataValue::DateTime(dt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_basic_operations() {
        let mut metadata = Metadata::new();
        metadata.set("title", "Test Dataset");
        metadata.set("version", 1i64);
        metadata.set("temperature", 25.5f64);
        
        assert_eq!(metadata.get_string("title"), Some("Test Dataset"));
        assert_eq!(metadata.get_integer("version"), Some(1));
        assert_eq!(metadata.get_float("temperature"), Some(25.5));
    }

    #[test]
    fn test_metadata_schema_validation() {
        let schema = MetadataSchema::new()
            .require("title")
            .require("version")
            .field_type("version", MetadataFieldType::Integer)
            .constraint(MetadataConstraint::MinValue("version".to_string(), 1.0));
        
        let mut metadata = Metadata::new();
        metadata.set("title", "Test");
        metadata.set("version", 2i64);
        
        assert!(schema.validate(&metadata).is_ok());
    }

    #[test]
    fn test_processing_history() {
        let mut metadata = Metadata::new();
        
        let entry = ProcessingHistoryEntry::new("normalize")
            .with_parameter("method", "z-score")
            .with_parameter("mean", 0.0)
            .with_parameter("std", 1.0);
        
        metadata.add_processing_history(entry);
        
        let history = metadata.get(standard_keys::PROCESSING_HISTORY);
        assert!(matches!(history, Some(MetadataValue::Array(_))));
    }
}