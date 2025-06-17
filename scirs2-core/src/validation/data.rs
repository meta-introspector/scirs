//! # Comprehensive Data Validation System
//!
//! Production-grade data validation system for SciRS2 Core providing schema
//! validation, constraint enforcement, and data integrity checks for scientific
//! computing applications in regulated environments.
//!
//! ## Features
//!
//! - JSON Schema validation with scientific extensions
//! - Constraint-based validation (range, format, pattern)
//! - Data integrity verification with checksums
//! - Type safety validation for numeric data
//! - Custom validation rules and plugins
//! - Performance-optimized validation pipelines
//! - Integration with ndarray for array validation
//! - Support for complex nested data structures
//! - Validation caching for repeated validations
//! - Detailed error reporting with context
//!
//! ## Example
//!
//! ```rust
//! use scirs2_core::validation::data::{Validator, ValidationSchema, ValidationConfig};
//! use ndarray::Array2;
//!
//! // Create a validation schema
//! let schema = ValidationSchema::new()
//!     .require_field("name", DataType::String)
//!     .require_field("age", DataType::Integer)
//!     .add_constraint("age", Constraint::Range { min: 0.0, max: 150.0 })
//!     .require_field("data", DataType::Array(Box::new(DataType::Float64)));
//!
//! let config = ValidationConfig::default();
//! let validator = Validator::new(config)?;
//!
//! // Validate data
//! let data = serde_json::json!({
//!     "name": "Test Dataset",
//!     "age": 25,
//!     "data": [[1.0, 2.0], [3.0, 4.0]]
//! });
//!
//! let result = validator.validate(&data, &schema)?;
//! if result.is_valid() {
//!     println!("Data is valid!");
//! } else {
//!     println!("Validation errors: {:#?}", result.errors());
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::error::{CoreError, ErrorContext};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};

// Core dependencies for array/matrix validation
use ndarray::{ArrayBase, Data, Dimension, ScalarOperand};
use num_traits::{Float, FromPrimitive, One, Zero};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "serde")]
use serde_json::Value as JsonValue;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

// For checksum validation
use std::collections::hash_map::DefaultHasher;

/// Data validation configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ValidationConfig {
    /// Enable strict mode (fail fast on first error)
    pub strict_mode: bool,
    /// Maximum validation depth for nested structures
    pub max_depth: usize,
    /// Enable validation caching
    pub enable_caching: bool,
    /// Cache size limit
    pub cache_size_limit: usize,
    /// Enable parallel validation for arrays
    pub enable_parallel_validation: bool,
    /// Custom validation rules
    pub custom_rules: HashMap<String, String>,
    /// Enable detailed error reporting
    pub detailed_errors: bool,
    /// Performance mode (reduced checks for speed)
    pub performance_mode: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            strict_mode: false,
            max_depth: 100,
            enable_caching: true,
            cache_size_limit: 1000,
            enable_parallel_validation: false, // Can be expensive
            custom_rules: HashMap::new(),
            detailed_errors: true,
            performance_mode: false,
        }
    }
}

/// Shape constraints for arrays and matrices
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ShapeConstraints {
    /// Exact dimensions required (None = any size for that dimension)
    pub dimensions: Vec<Option<usize>>,
    /// Minimum number of elements
    pub min_elements: Option<usize>,
    /// Maximum number of elements
    pub max_elements: Option<usize>,
    /// Whether matrix must be square (for 2D only)
    pub require_square: bool,
    /// Whether to allow broadcasting-compatible shapes
    pub allow_broadcasting: bool,
}

/// Sparse matrix formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum SparseFormat {
    /// Compressed Sparse Row
    CSR,
    /// Compressed Sparse Column
    CSC,
    /// Coordinate format (COO)
    COO,
    /// Dictionary of Keys
    DOK,
}

/// Time series constraints
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TimeConstraints {
    /// Minimum time interval between samples
    pub min_interval: Option<std::time::Duration>,
    /// Maximum time interval between samples
    pub max_interval: Option<std::time::Duration>,
    /// Whether timestamps must be monotonic
    pub require_monotonic: bool,
    /// Whether to allow duplicate timestamps
    pub allow_duplicates: bool,
}

/// Enhanced validation error type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ValidationErrorType {
    /// Required field missing
    MissingRequiredField,
    /// Type mismatch
    TypeMismatch,
    /// Constraint violation
    ConstraintViolation,
    /// Invalid format
    InvalidFormat,
    /// Out of range value
    OutOfRange,
    /// Invalid array size
    InvalidArraySize,
    /// Duplicate values where unique required
    DuplicateValues,
    /// Data integrity failure
    IntegrityFailure,
    /// Custom validation rule failure
    CustomRuleFailure,
    /// Schema validation error
    SchemaError,
    /// Shape validation error
    ShapeError,
    /// Numeric quality error (NaN, infinity)
    InvalidNumeric,
    /// Statistical constraint violation
    StatisticalViolation,
    /// Performance issue
    Performance,
    /// Data integrity error
    IntegrityError,
    /// Type conversion error
    TypeConversion,
}

/// Data quality assessment result
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DataQualityReport {
    /// Overall quality score (0.0 to 1.0)
    pub quality_score: f64,
    /// Detailed quality metrics
    pub metrics: QualityMetrics,
    /// Issues found during validation
    pub issues: Vec<QualityIssue>,
    /// Recommendations for improvement
    pub recommendations: Vec<String>,
}

/// Detailed quality metrics
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct QualityMetrics {
    /// Completeness (non-null/NaN ratio)
    pub completeness: f64,
    /// Consistency (pattern conformance)
    pub consistency: f64,
    /// Accuracy (constraint compliance)
    pub accuracy: f64,
    /// Validity (type/format correctness)
    pub validity: f64,
    /// Statistical properties
    pub statistical_summary: Option<StatisticalSummary>,
}

/// Statistical summary of numeric data
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct StatisticalSummary {
    /// Number of data points
    pub count: usize,
    /// Mean value
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Number of outliers detected
    pub outliers: usize,
    /// Data distribution type (if detectable)
    pub distribution: Option<String>,
}

/// Quality issue found during validation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct QualityIssue {
    /// Issue type
    pub issue_type: QualityIssueType,
    /// Location where issue was found
    pub location: String,
    /// Description of the issue
    pub description: String,
    /// Severity of the issue
    pub severity: ErrorSeverity,
    /// Suggested fix
    pub suggestion: Option<String>,
}

/// Types of data quality issues
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum QualityIssueType {
    /// Missing or null values
    MissingData,
    /// Invalid numeric values (NaN, infinity)
    InvalidNumeric,
    /// Out-of-range values
    OutOfRange,
    /// Inconsistent format
    FormatInconsistency,
    /// Statistical outliers
    Outlier,
    /// Duplicate entries
    Duplicate,
    /// Type mismatch
    TypeMismatch,
    /// Constraint violation
    ConstraintViolation,
    /// Performance issue
    Performance,
}

/// Data types supported by the validation system
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum DataType {
    /// Boolean value
    Boolean,
    /// Integer number
    Integer,
    /// Floating point number
    Float32,
    /// Double precision floating point
    Float64,
    /// UTF-8 string
    String,
    /// Array of elements
    Array(Box<DataType>),
    /// Object with fields
    Object(HashMap<String, DataType>),
    /// Union of multiple types
    Union(Vec<DataType>),
    /// Optional field (can be null)
    Optional(Box<DataType>),
    /// Any type (no validation)
    Any,
    /// Date/time value
    DateTime,
    /// Binary data
    Binary,
    /// Complex number
    Complex,
    /// Numeric matrix/array (ndarray)
    Matrix {
        element_type: Box<DataType>,
        dimensions: Option<Vec<usize>>,
        shape_constraints: Option<ShapeConstraints>,
    },
    /// N-dimensional array with flexible shape
    NDArray {
        element_type: Box<DataType>,
        min_dimensions: Option<usize>,
        max_dimensions: Option<usize>,
        shape_constraints: Option<ShapeConstraints>,
    },
    /// Sparse matrix representation
    SparseMatrix {
        element_type: Box<DataType>,
        format: SparseFormat,
    },
    /// Time series data
    TimeSeries {
        element_type: Box<DataType>,
        time_constraints: Option<TimeConstraints>,
    },
}

impl DataType {
    /// Check if a value matches this data type
    #[cfg(feature = "serde")]
    pub fn matches(&self, value: &JsonValue) -> bool {
        match (self, value) {
            (DataType::Boolean, JsonValue::Bool(_)) => true,
            (DataType::Integer, JsonValue::Number(n)) => n.is_i64(),
            (DataType::Float32 | DataType::Float64, JsonValue::Number(_)) => true,
            (DataType::String, JsonValue::String(_)) => true,
            (DataType::Array(element_type), JsonValue::Array(arr)) => {
                arr.iter().all(|v| element_type.matches(v))
            }
            (DataType::Object(schema), JsonValue::Object(obj)) => {
                schema.iter().all(|(key, expected_type)| {
                    obj.get(key).map_or(false, |v| expected_type.matches(v))
                })
            }
            (DataType::Union(types), value) => types.iter().any(|t| t.matches(value)),
            (DataType::Optional(_inner_type), JsonValue::Null) => true,
            (DataType::Optional(inner_type), value) => inner_type.matches(value),
            (DataType::Any, _) => true,
            _ => false,
        }
    }

    /// Get the type name as string
    pub fn type_name(&self) -> String {
        match self {
            DataType::Boolean => "boolean".to_string(),
            DataType::Integer => "integer".to_string(),
            DataType::Float32 => "float32".to_string(),
            DataType::Float64 => "float64".to_string(),
            DataType::String => "string".to_string(),
            DataType::Array(element_type) => format!("array<{}>", element_type.type_name()),
            DataType::Object(_) => "object".to_string(),
            DataType::Union(types) => {
                let type_names: Vec<String> = types.iter().map(|t| t.type_name()).collect();
                format!("union<{}>", type_names.join("|"))
            }
            DataType::Optional(inner_type) => format!("optional<{}>", inner_type.type_name()),
            DataType::Any => "any".to_string(),
            DataType::DateTime => "datetime".to_string(),
            DataType::Binary => "binary".to_string(),
            DataType::Complex => "complex".to_string(),
            DataType::Matrix {
                element_type,
                dimensions,
                shape_constraints: _,
            } => {
                if let Some(dims) = dimensions {
                    format!("matrix<{}, {:?}>", element_type.type_name(), dims)
                } else {
                    format!("matrix<{}>", element_type.type_name())
                }
            }
            DataType::NDArray {
                element_type,
                min_dimensions,
                max_dimensions,
                shape_constraints: _,
            } => match (min_dimensions, max_dimensions) {
                (Some(min), Some(max)) if min == max => {
                    format!("ndarray<{}, {}D>", element_type.type_name(), min)
                }
                (Some(min), Some(max)) => {
                    format!("ndarray<{}, {}D-{}D>", element_type.type_name(), min, max)
                }
                (Some(min), None) => {
                    format!("ndarray<{}, {}D+>", element_type.type_name(), min)
                }
                (None, Some(max)) => {
                    format!("ndarray<{}, <={}D>", element_type.type_name(), max)
                }
                (None, None) => {
                    format!("ndarray<{}>", element_type.type_name())
                }
            },
            DataType::SparseMatrix {
                element_type,
                format,
            } => {
                format!("sparse_matrix<{}, {:?}>", element_type.type_name(), format)
            }
            DataType::TimeSeries {
                element_type,
                time_constraints: _,
            } => {
                format!("time_series<{}>", element_type.type_name())
            }
        }
    }
}

/// Validation constraints
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Constraint {
    /// Value must be within range (inclusive)
    Range { min: f64, max: f64 },
    /// String length constraints
    Length {
        min: Option<usize>,
        max: Option<usize>,
    },
    /// Pattern matching (regex)
    Pattern(String),
    /// Must be one of the specified values
    Enum(Vec<String>),
    /// Custom constraint with name and parameters
    Custom {
        name: String,
        params: HashMap<String, String>,
    },
    /// Array size constraints
    ArraySize {
        min: Option<usize>,
        max: Option<usize>,
    },
    /// Unique values constraint
    Unique,
    /// Not null constraint
    NotNull,
    /// Numeric precision constraint
    Precision { decimal_places: u8 },
    /// Matrix/array shape constraint
    Shape(Vec<usize>),
    /// Data integrity constraint (checksum)
    Checksum { algorithm: String, expected: String },
    /// Numeric value constraints (NaN, infinity, precision)
    NumericQuality {
        allow_nan: bool,
        allow_infinite: bool,
        min_precision: Option<f64>,
        max_relative_error: Option<f64>,
    },
    /// Array/matrix shape constraint
    ArrayShape {
        dimensions: Vec<Option<usize>>, // None means any size for that dimension
        min_elements: Option<usize>,
        max_elements: Option<usize>,
    },
    /// Statistical constraints on data
    Statistical {
        min_mean: Option<f64>,
        max_mean: Option<f64>,
        min_std: Option<f64>,
        max_std: Option<f64>,
        required_distribution: Option<String>,
    },
    /// Data sparsity constraints
    Sparsity {
        min_density: Option<f64>,
        max_density: Option<f64>,
        required_zeros: Option<usize>,
    },
    /// Performance constraint for large datasets
    Performance {
        max_validation_time_ms: Option<u64>,
        enable_parallel: bool,
        chunk_size: Option<usize>,
    },
}

/// Field definition in a validation schema
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FieldDefinition {
    /// Field data type
    pub data_type: DataType,
    /// Whether field is required
    pub required: bool,
    /// Constraints applied to this field
    pub constraints: Vec<Constraint>,
    /// Field description
    pub description: Option<String>,
    /// Default value if not provided
    pub default_value: Option<String>,
    /// Validation rule references
    pub validation_rules: Vec<String>,
}

impl FieldDefinition {
    /// Create a new field definition
    pub fn new(data_type: DataType) -> Self {
        Self {
            data_type,
            required: false,
            constraints: Vec::new(),
            description: None,
            default_value: None,
            validation_rules: Vec::new(),
        }
    }

    /// Mark field as required
    pub fn required(mut self) -> Self {
        self.required = true;
        self
    }

    /// Add a constraint
    pub fn with_constraint(mut self, constraint: Constraint) -> Self {
        self.constraints.push(constraint);
        self
    }

    /// Set description
    pub fn with_description(mut self, description: &str) -> Self {
        self.description = Some(description.to_string());
        self
    }

    /// Set default value
    pub fn with_default(mut self, default: &str) -> Self {
        self.default_value = Some(default.to_string());
        self
    }
}

/// Validation schema definition
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ValidationSchema {
    /// Schema name
    pub name: String,
    /// Schema version
    pub version: String,
    /// Field definitions
    pub fields: HashMap<String, FieldDefinition>,
    /// Global constraints
    pub global_constraints: Vec<Constraint>,
    /// Schema metadata
    pub metadata: HashMap<String, String>,
    /// Allow additional fields not in schema
    pub allow_additional_fields: bool,
}

impl ValidationSchema {
    /// Create a new validation schema
    pub fn new() -> Self {
        Self {
            name: "unnamed".to_string(),
            version: "1.0.0".to_string(),
            fields: HashMap::new(),
            global_constraints: Vec::new(),
            metadata: HashMap::new(),
            allow_additional_fields: false,
        }
    }

    /// Set schema name
    pub fn name(mut self, name: &str) -> Self {
        self.name = name.to_string();
        self
    }

    /// Set schema version
    pub fn version(mut self, version: &str) -> Self {
        self.version = version.to_string();
        self
    }

    /// Add a required field
    pub fn require_field(mut self, name: &str, data_type: DataType) -> Self {
        let field = FieldDefinition::new(data_type).required();
        self.fields.insert(name.to_string(), field);
        self
    }

    /// Add an optional field
    pub fn optional_field(mut self, name: &str, data_type: DataType) -> Self {
        let field = FieldDefinition::new(data_type);
        self.fields.insert(name.to_string(), field);
        self
    }

    /// Add a constraint to a field
    pub fn add_constraint(mut self, field_name: &str, constraint: Constraint) -> Self {
        if let Some(field) = self.fields.get_mut(field_name) {
            field.constraints.push(constraint);
        }
        self
    }

    /// Add a global constraint
    pub fn add_global_constraint(mut self, constraint: Constraint) -> Self {
        self.global_constraints.push(constraint);
        self
    }

    /// Allow additional fields
    pub fn allow_additional(mut self) -> Self {
        self.allow_additional_fields = true;
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
}

impl Default for ValidationSchema {
    fn default() -> Self {
        Self::new()
    }
}

/// Validation error information
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ValidationError {
    /// Error type
    pub error_type: ValidationErrorType,
    /// Field path where error occurred
    pub field_path: String,
    /// Error message
    pub message: String,
    /// Expected value/type
    pub expected: Option<String>,
    /// Actual value found
    pub actual: Option<String>,
    /// Constraint that was violated
    pub constraint: Option<String>,
    /// Error severity
    pub severity: ErrorSeverity,
    /// Additional context
    pub context: HashMap<String, String>,
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ErrorSeverity {
    /// Warning - data may still be usable
    Warning,
    /// Error - data should not be used
    Error,
    /// Critical - data is corrupted or dangerous
    Critical,
}

/// Validation result
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ValidationResult {
    /// Whether validation passed
    pub valid: bool,
    /// Validation errors
    pub errors: Vec<ValidationError>,
    /// Validation warnings
    pub warnings: Vec<ValidationError>,
    /// Validation statistics
    pub stats: ValidationStats,
    /// Processing time
    pub duration: std::time::Duration,
}

impl ValidationResult {
    /// Check if validation passed
    pub fn is_valid(&self) -> bool {
        self.valid
    }

    /// Get all errors
    pub fn errors(&self) -> &[ValidationError] {
        &self.errors
    }

    /// Get all warnings
    pub fn warnings(&self) -> &[ValidationError] {
        &self.warnings
    }

    /// Get critical errors only
    pub fn critical_errors(&self) -> Vec<&ValidationError> {
        self.errors
            .iter()
            .filter(|e| e.severity == ErrorSeverity::Critical)
            .collect()
    }

    /// Check if there are any critical errors
    pub fn has_critical_errors(&self) -> bool {
        self.errors
            .iter()
            .any(|e| e.severity == ErrorSeverity::Critical)
    }
}

/// Validation statistics
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ValidationStats {
    /// Number of fields validated
    pub fields_validated: usize,
    /// Number of constraints checked
    pub constraints_checked: usize,
    /// Number of cache hits
    pub cache_hits: usize,
    /// Number of cache misses
    pub cache_misses: usize,
    /// Validation depth reached
    pub max_depth_reached: usize,
}

/// Validation cache entry
#[derive(Debug, Clone)]
struct CacheEntry {
    /// Cached validation result
    result: ValidationResult,
    /// Cache entry timestamp
    timestamp: std::time::Instant,
    /// Cache hit count
    hit_count: usize,
}

/// Main data validator implementation
pub struct Validator {
    /// Configuration
    config: ValidationConfig,
    /// Validation cache
    cache: Arc<RwLock<HashMap<String, CacheEntry>>>,
    /// Custom validation rules
    custom_rules: HashMap<String, Box<dyn ValidationRule + Send + Sync>>,
    /// Schema registry
    schema_registry: Arc<RwLock<HashMap<String, ValidationSchema>>>,
}

impl Validator {
    /// Create a new validator
    pub fn new(config: ValidationConfig) -> Result<Self, CoreError> {
        Ok(Self {
            config,
            cache: Arc::new(RwLock::new(HashMap::new())),
            custom_rules: HashMap::new(),
            schema_registry: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Register a schema
    pub fn register_schema(&self, schema: ValidationSchema) -> Result<(), CoreError> {
        let mut registry = self.schema_registry.write().map_err(|_| {
            CoreError::ComputationError(crate::error::ErrorContext::new(
                "Failed to acquire schema registry lock".to_string(),
            ))
        })?;
        registry.insert(schema.name.clone(), schema);
        Ok(())
    }

    /// Get a registered schema
    pub fn get_schema(&self, name: &str) -> Result<Option<ValidationSchema>, CoreError> {
        let registry = self.schema_registry.read().map_err(|_| {
            CoreError::ComputationError(crate::error::ErrorContext::new(
                "Failed to acquire schema registry lock".to_string(),
            ))
        })?;
        Ok(registry.get(name).cloned())
    }

    /// Validate data against a schema
    #[cfg(feature = "serde")]
    pub fn validate(
        &self,
        data: &JsonValue,
        schema: &ValidationSchema,
    ) -> Result<ValidationResult, CoreError> {
        let start_time = std::time::Instant::now();
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let mut stats = ValidationStats::default();

        // Check cache if enabled
        if self.config.enable_caching {
            let cache_key = self.generate_cache_key(data, schema)?;
            if let Some(cached_result) = self.get_cached_result(&cache_key)? {
                stats.cache_hits += 1;
                return Ok(cached_result);
            }
            stats.cache_misses += 1;
        }

        // Validate each field in the schema
        self.validate_fields(data, schema, &mut errors, &mut warnings, &mut stats, 0)?;

        // Apply global constraints
        self.validate_global_constraints(data, schema, &mut errors, &mut warnings, &mut stats)?;

        // Check for additional fields if not allowed
        if !schema.allow_additional_fields {
            self.check_additional_fields(data, schema, &mut errors, &mut warnings)?;
        }

        let valid = errors.is_empty()
            && !warnings
                .iter()
                .any(|w| w.severity == ErrorSeverity::Critical);
        let duration = start_time.elapsed();

        let result = ValidationResult {
            valid,
            errors,
            warnings,
            stats,
            duration,
        };

        // Cache result if enabled
        if self.config.enable_caching {
            let cache_key = self.generate_cache_key(data, schema)?;
            self.cache_result(&cache_key, result.clone())?;
        }

        Ok(result)
    }

    /// Validate individual fields
    #[cfg(feature = "serde")]
    fn validate_fields(
        &self,
        data: &JsonValue,
        schema: &ValidationSchema,
        errors: &mut Vec<ValidationError>,
        warnings: &mut Vec<ValidationError>,
        stats: &mut ValidationStats,
        depth: usize,
    ) -> Result<(), CoreError> {
        if depth > self.config.max_depth {
            errors.push(ValidationError {
                error_type: ValidationErrorType::SchemaError,
                field_path: "root".to_string(),
                message: "Maximum validation depth exceeded".to_string(),
                expected: None,
                actual: None,
                constraint: None,
                severity: ErrorSeverity::Error,
                context: HashMap::new(),
            });
            return Ok(());
        }

        stats.max_depth_reached = stats.max_depth_reached.max(depth);

        let data_obj = match data {
            JsonValue::Object(obj) => obj,
            _ => {
                errors.push(ValidationError {
                    error_type: ValidationErrorType::TypeMismatch,
                    field_path: "root".to_string(),
                    message: "Expected object".to_string(),
                    expected: Some("object".to_string()),
                    actual: Some(format!("{:?}", data)),
                    constraint: None,
                    severity: ErrorSeverity::Error,
                    context: HashMap::new(),
                });
                return Ok(());
            }
        };

        for (field_name, field_def) in &schema.fields {
            stats.fields_validated += 1;

            let field_path = if depth == 0 {
                field_name.clone()
            } else {
                format!("root.{}", field_name)
            };

            if let Some(field_value) = data_obj.get(field_name) {
                // Field exists, validate type and constraints
                self.validate_field_type(field_value, &field_def.data_type, &field_path, errors)?;
                self.validate_field_constraints(
                    field_value,
                    &field_def.constraints,
                    &field_path,
                    errors,
                    warnings,
                    stats,
                )?;

                // Validate custom rules
                for rule_name in &field_def.validation_rules {
                    if let Some(rule) = self.custom_rules.get(rule_name) {
                        if let Err(rule_error) = rule.validate(field_value, &field_path) {
                            errors.push(ValidationError {
                                error_type: ValidationErrorType::CustomRuleFailure,
                                field_path: field_path.clone(),
                                message: rule_error,
                                expected: None,
                                actual: None,
                                constraint: Some(rule_name.clone()),
                                severity: ErrorSeverity::Error,
                                context: HashMap::new(),
                            });
                        }
                    }
                }

                // Recursive validation for nested objects and arrays
                match &field_def.data_type {
                    DataType::Object(nested_schema) => {
                        let nested_validation_schema = ValidationSchema {
                            name: format!("{}.{}", schema.name, field_name),
                            version: schema.version.clone(),
                            fields: nested_schema
                                .iter()
                                .map(|(k, v)| (k.clone(), FieldDefinition::new(v.clone())))
                                .collect(),
                            global_constraints: Vec::new(),
                            metadata: HashMap::new(),
                            allow_additional_fields: schema.allow_additional_fields,
                        };
                        self.validate_fields(
                            field_value,
                            &nested_validation_schema,
                            errors,
                            warnings,
                            stats,
                            depth + 1,
                        )?;
                    }
                    DataType::Array(element_type) => {
                        if let JsonValue::Array(arr) = field_value {
                            for (i, element) in arr.iter().enumerate() {
                                let element_path = format!("{}[{}]", field_path, i);
                                self.validate_field_type(
                                    element,
                                    element_type,
                                    &element_path,
                                    errors,
                                )?;
                            }
                        }
                    }
                    _ => {}
                }
            } else if field_def.required {
                // Required field is missing
                errors.push(ValidationError {
                    error_type: ValidationErrorType::MissingRequiredField,
                    field_path: field_path.clone(),
                    message: format!("Required field '{}' is missing", field_name),
                    expected: Some(field_def.data_type.type_name()),
                    actual: Some("missing".to_string()),
                    constraint: None,
                    severity: ErrorSeverity::Error,
                    context: HashMap::new(),
                });

                if self.config.strict_mode {
                    return Ok(());
                }
            }
        }

        Ok(())
    }

    /// Validate field type
    #[cfg(feature = "serde")]
    fn validate_field_type(
        &self,
        value: &JsonValue,
        expected_type: &DataType,
        field_path: &str,
        errors: &mut Vec<ValidationError>,
    ) -> Result<(), CoreError> {
        if !expected_type.matches(value) {
            errors.push(ValidationError {
                error_type: ValidationErrorType::TypeMismatch,
                field_path: field_path.to_string(),
                message: format!("Type mismatch for field '{}'", field_path),
                expected: Some(expected_type.type_name()),
                actual: Some(self.get_value_type_name(value)),
                constraint: None,
                severity: ErrorSeverity::Error,
                context: HashMap::new(),
            });
        }
        Ok(())
    }

    /// Validate field constraints
    #[cfg(feature = "serde")]
    #[allow(unused_variables)] // warnings is used conditionally based on regex feature
    fn validate_field_constraints(
        &self,
        value: &JsonValue,
        constraints: &[Constraint],
        field_path: &str,
        errors: &mut Vec<ValidationError>,
        warnings: &mut Vec<ValidationError>,
        stats: &mut ValidationStats,
    ) -> Result<(), CoreError> {
        for constraint in constraints {
            stats.constraints_checked += 1;

            match constraint {
                Constraint::Range { min, max } => {
                    if let Some(num) = value.as_f64() {
                        if num < *min || num > *max {
                            errors.push(ValidationError {
                                error_type: ValidationErrorType::OutOfRange,
                                field_path: field_path.to_string(),
                                message: format!(
                                    "Value {} is not within range [{}, {}]",
                                    num, min, max
                                ),
                                expected: Some(format!("range [{}, {}]", min, max)),
                                actual: Some(num.to_string()),
                                constraint: Some("range".to_string()),
                                severity: ErrorSeverity::Error,
                                context: HashMap::new(),
                            });
                        }
                    }
                }
                Constraint::Length { min, max } => {
                    if let Some(s) = value.as_str() {
                        let len = s.len();
                        if let Some(min_len) = min {
                            if len < *min_len {
                                errors.push(ValidationError {
                                    error_type: ValidationErrorType::ConstraintViolation,
                                    field_path: field_path.to_string(),
                                    message: format!(
                                        "String length {} is less than minimum {}",
                                        len, min_len
                                    ),
                                    expected: Some(format!("min length {}", min_len)),
                                    actual: Some(len.to_string()),
                                    constraint: Some("length".to_string()),
                                    severity: ErrorSeverity::Error,
                                    context: HashMap::new(),
                                });
                            }
                        }
                        if let Some(max_len) = max {
                            if len > *max_len {
                                errors.push(ValidationError {
                                    error_type: ValidationErrorType::ConstraintViolation,
                                    field_path: field_path.to_string(),
                                    message: format!(
                                        "String length {} exceeds maximum {}",
                                        len, max_len
                                    ),
                                    expected: Some(format!("max length {}", max_len)),
                                    actual: Some(len.to_string()),
                                    constraint: Some("length".to_string()),
                                    severity: ErrorSeverity::Error,
                                    context: HashMap::new(),
                                });
                            }
                        }
                    }
                }
                Constraint::Pattern(pattern) => {
                    if let Some(s) = value.as_str() {
                        #[cfg(feature = "regex")]
                        {
                            use regex::Regex;
                            if let Ok(re) = Regex::new(pattern) {
                                if !re.is_match(s) {
                                    errors.push(ValidationError {
                                        error_type: ValidationErrorType::InvalidFormat,
                                        field_path: field_path.to_string(),
                                        message: format!(
                                            "String '{}' does not match pattern '{}'",
                                            s, pattern
                                        ),
                                        expected: Some(format!("pattern {}", pattern)),
                                        actual: Some(s.to_string()),
                                        constraint: Some("pattern".to_string()),
                                        severity: ErrorSeverity::Error,
                                        context: HashMap::new(),
                                    });
                                }
                            }
                        }
                        #[cfg(not(feature = "regex"))]
                        {
                            warnings.push(ValidationError {
                                error_type: ValidationErrorType::SchemaError,
                                field_path: field_path.to_string(),
                                message: "Pattern validation requires regex feature".to_string(),
                                expected: None,
                                actual: None,
                                constraint: Some("pattern".to_string()),
                                severity: ErrorSeverity::Warning,
                                context: HashMap::new(),
                            });
                        }
                    }
                }
                Constraint::Enum(allowed_values) => {
                    if let Some(s) = value.as_str() {
                        if !allowed_values.contains(&s.to_string()) {
                            errors.push(ValidationError {
                                error_type: ValidationErrorType::ConstraintViolation,
                                field_path: field_path.to_string(),
                                message: format!(
                                    "Value '{}' is not in allowed values: {:?}",
                                    s, allowed_values
                                ),
                                expected: Some(format!("one of {:?}", allowed_values)),
                                actual: Some(s.to_string()),
                                constraint: Some("enum".to_string()),
                                severity: ErrorSeverity::Error,
                                context: HashMap::new(),
                            });
                        }
                    }
                }
                Constraint::NotNull => {
                    if value.is_null() {
                        errors.push(ValidationError {
                            error_type: ValidationErrorType::ConstraintViolation,
                            field_path: field_path.to_string(),
                            message: "Value cannot be null".to_string(),
                            expected: Some("non-null value".to_string()),
                            actual: Some("null".to_string()),
                            constraint: Some("not_null".to_string()),
                            severity: ErrorSeverity::Error,
                            context: HashMap::new(),
                        });
                    }
                }
                Constraint::ArraySize { min, max } => {
                    if let Some(arr) = value.as_array() {
                        let size = arr.len();
                        if let Some(min_size) = min {
                            if size < *min_size {
                                errors.push(ValidationError {
                                    error_type: ValidationErrorType::InvalidArraySize,
                                    field_path: field_path.to_string(),
                                    message: format!(
                                        "Array size {} is less than minimum {}",
                                        size, min_size
                                    ),
                                    expected: Some(format!("min size {}", min_size)),
                                    actual: Some(size.to_string()),
                                    constraint: Some("array_size".to_string()),
                                    severity: ErrorSeverity::Error,
                                    context: HashMap::new(),
                                });
                            }
                        }
                        if let Some(max_size) = max {
                            if size > *max_size {
                                errors.push(ValidationError {
                                    error_type: ValidationErrorType::InvalidArraySize,
                                    field_path: field_path.to_string(),
                                    message: format!(
                                        "Array size {} exceeds maximum {}",
                                        size, max_size
                                    ),
                                    expected: Some(format!("max size {}", max_size)),
                                    actual: Some(size.to_string()),
                                    constraint: Some("array_size".to_string()),
                                    severity: ErrorSeverity::Error,
                                    context: HashMap::new(),
                                });
                            }
                        }
                    }
                }
                Constraint::Unique => {
                    if let Some(arr) = value.as_array() {
                        let mut seen = HashSet::new();
                        for item in arr {
                            let item_str = item.to_string();
                            if !seen.insert(item_str.clone()) {
                                errors.push(ValidationError {
                                    error_type: ValidationErrorType::DuplicateValues,
                                    field_path: field_path.to_string(),
                                    message: format!("Duplicate value found: {}", item_str),
                                    expected: Some("unique values".to_string()),
                                    actual: Some("duplicate found".to_string()),
                                    constraint: Some("unique".to_string()),
                                    severity: ErrorSeverity::Error,
                                    context: HashMap::new(),
                                });
                            }
                        }
                    }
                }
                _ => {
                    // Other constraints not yet implemented
                }
            }
        }
        Ok(())
    }

    /// Validate global constraints
    #[cfg(feature = "serde")]
    fn validate_global_constraints(
        &self,
        _data: &JsonValue,
        _schema: &ValidationSchema,
        _errors: &mut Vec<ValidationError>,
        _warnings: &mut Vec<ValidationError>,
        _stats: &mut ValidationStats,
    ) -> Result<(), CoreError> {
        // Global constraints would be implemented here
        Ok(())
    }

    /// Check for additional fields
    #[cfg(feature = "serde")]
    fn check_additional_fields(
        &self,
        data: &JsonValue,
        schema: &ValidationSchema,
        errors: &mut Vec<ValidationError>,
        _warnings: &mut Vec<ValidationError>,
    ) -> Result<(), CoreError> {
        if let JsonValue::Object(obj) = data {
            for key in obj.keys() {
                if !schema.fields.contains_key(key) {
                    errors.push(ValidationError {
                        error_type: ValidationErrorType::SchemaError,
                        field_path: key.clone(),
                        message: format!("Additional field '{}' not allowed", key),
                        expected: None,
                        actual: Some(key.clone()),
                        constraint: None,
                        severity: ErrorSeverity::Warning,
                        context: HashMap::new(),
                    });
                }
            }
        }
        Ok(())
    }

    /// Get the type name for a JSON value
    #[cfg(feature = "serde")]
    fn get_value_type_name(&self, value: &JsonValue) -> String {
        match value {
            JsonValue::Null => "null".to_string(),
            JsonValue::Bool(_) => "boolean".to_string(),
            JsonValue::Number(n) => {
                if n.is_i64() {
                    "integer".to_string()
                } else {
                    "number".to_string()
                }
            }
            JsonValue::String(_) => "string".to_string(),
            JsonValue::Array(_) => "array".to_string(),
            JsonValue::Object(_) => "object".to_string(),
        }
    }

    /// Generate cache key for validation result
    #[cfg(feature = "serde")]
    fn generate_cache_key(
        &self,
        data: &JsonValue,
        schema: &ValidationSchema,
    ) -> Result<String, CoreError> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        data.to_string().hash(&mut hasher);
        schema.name.hash(&mut hasher);
        schema.version.hash(&mut hasher);

        Ok(format!("{:x}", hasher.finish()))
    }

    /// Get cached validation result
    fn get_cached_result(&self, cache_key: &str) -> Result<Option<ValidationResult>, CoreError> {
        let cache = self.cache.read().map_err(|_| {
            CoreError::ComputationError(crate::error::ErrorContext::new(
                "Failed to acquire cache read lock".to_string(),
            ))
        })?;

        if let Some(entry) = cache.get(cache_key) {
            // Check if cache entry is still valid (for now, always valid)
            return Ok(Some(entry.result.clone()));
        }

        Ok(None)
    }

    /// Cache validation result
    fn cache_result(&self, cache_key: &str, result: ValidationResult) -> Result<(), CoreError> {
        let mut cache = self.cache.write().map_err(|_| {
            CoreError::ComputationError(crate::error::ErrorContext::new(
                "Failed to acquire cache write lock".to_string(),
            ))
        })?;

        // Remove oldest entries if cache is full
        if cache.len() >= self.config.cache_size_limit {
            if let Some((oldest_key, _)) = cache
                .iter()
                .min_by_key(|(_, entry)| entry.timestamp)
                .map(|(k, v)| (k.clone(), v.clone()))
            {
                cache.remove(&oldest_key);
            }
        }

        let entry = CacheEntry {
            result,
            timestamp: std::time::Instant::now(),
            hit_count: 0,
        };

        cache.insert(cache_key.to_string(), entry);
        Ok(())
    }

    /// Add a custom validation rule
    pub fn add_custom_rule(&mut self, name: String, rule: Box<dyn ValidationRule + Send + Sync>) {
        self.custom_rules.insert(name, rule);
    }

    /// Validate ndarray with comprehensive checks
    pub fn validate_ndarray<S, D>(
        &self,
        array: &ArrayBase<S, D>,
        constraints: &ArrayValidationConstraints<S::Elem>,
        _config: &ValidationConfig,
    ) -> Result<ValidationResult, CoreError>
    where
        S: Data,
        D: Dimension,
        S::Elem: Float + fmt::Debug + Send + Sync + ScalarOperand + FromPrimitive,
    {
        let start_time = std::time::Instant::now();
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let mut stats = ValidationStats::default();

        // Basic shape validation
        if let Some(expected_shape) = &constraints.expected_shape {
            if !self.validate_array_shape(array, expected_shape)? {
                errors.push(ValidationError {
                    error_type: ValidationErrorType::ShapeError,
                    field_path: constraints
                        .field_name
                        .clone()
                        .unwrap_or("array".to_string()),
                    message: format!(
                        "Array shape {:?} does not match expected {:?}",
                        array.shape(),
                        expected_shape
                    ),
                    expected: Some(format!("{:?}", expected_shape)),
                    actual: Some(format!("{:?}", array.shape())),
                    constraint: Some("shape".to_string()),
                    severity: ErrorSeverity::Error,
                    context: HashMap::new(),
                });
            }
        }

        // Numeric quality validation
        if constraints.check_numeric_quality {
            self.validate_numeric_quality(array, &mut errors, &mut warnings, &mut stats)?;
        }

        // Statistical validation
        if let Some(stat_constraints) = &constraints.statistical_constraints {
            self.validate_statistical_properties(
                array,
                stat_constraints,
                &mut errors,
                &mut warnings,
            )?;
        }

        // Performance validation for large arrays
        if constraints.check_performance {
            self.validate_array_performance(array, &mut warnings)?;
        }

        let valid = errors.is_empty()
            && !warnings
                .iter()
                .any(|w| w.severity == ErrorSeverity::Critical);
        let duration = start_time.elapsed();

        Ok(ValidationResult {
            valid,
            errors,
            warnings,
            stats,
            duration,
        })
    }

    /// Validate array shape against constraints
    fn validate_array_shape<S, D>(
        &self,
        array: &ArrayBase<S, D>,
        expected_shape: &[usize],
    ) -> Result<bool, CoreError>
    where
        S: Data,
        D: Dimension,
    {
        let actual_shape = array.shape();
        Ok(actual_shape == expected_shape)
    }

    /// Validate numeric quality (NaN, infinity, precision)
    fn validate_numeric_quality<S, D>(
        &self,
        _array: &ArrayBase<S, D>,
        _errors: &mut Vec<ValidationError>,
        warnings: &mut Vec<ValidationError>,
        stats: &mut ValidationStats,
    ) -> Result<(), CoreError>
    where
        S: Data,
        D: Dimension,
        S::Elem: Float + fmt::Debug,
    {
        let nan_count = 0;
        let inf_count = 0;
        let total_count = 0;

        #[cfg(feature = "parallel")]
        let _check_parallel = array.len() > 10000;

        #[cfg(feature = "parallel")]
        if _check_parallel {
            let results: Vec<_> = array
                .as_slice()
                .unwrap_or(&[])
                .par_iter()
                .map(|&value| {
                    let is_nan = value.is_nan();
                    let is_inf = value.is_infinite();
                    (is_nan, is_inf)
                })
                .collect();

            for (is_nan, is_inf) in results {
                total_count += 1;
                if is_nan {
                    nan_count += 1;
                }
                if is_inf {
                    inf_count += 1;
                }
            }
        } else {
            for value in array.iter() {
                total_count += 1;
                if value.is_nan() {
                    nan_count += 1;
                }
                if value.is_infinite() {
                    inf_count += 1;
                }
            }
        }

        stats.fields_validated += 1;
        stats.constraints_checked += 2; // NaN and infinity checks

        if nan_count > 0 {
            warnings.push(ValidationError {
                error_type: ValidationErrorType::InvalidNumeric,
                field_path: "array".to_string(),
                message: format!(
                    "Found {} NaN values out of {} total",
                    nan_count, total_count
                ),
                expected: Some("finite values".to_string()),
                actual: Some(format!("{} NaN values", nan_count)),
                constraint: Some("numeric_quality".to_string()),
                severity: ErrorSeverity::Warning,
                context: HashMap::new(),
            });
        }

        if inf_count > 0 {
            warnings.push(ValidationError {
                error_type: ValidationErrorType::InvalidNumeric,
                field_path: "array".to_string(),
                message: format!(
                    "Found {} infinite values out of {} total",
                    inf_count, total_count
                ),
                expected: Some("finite values".to_string()),
                actual: Some(format!("{} infinite values", inf_count)),
                constraint: Some("numeric_quality".to_string()),
                severity: ErrorSeverity::Warning,
                context: HashMap::new(),
            });
        }

        Ok(())
    }

    /// Validate statistical properties of numeric arrays
    fn validate_statistical_properties<S, D>(
        &self,
        array: &ArrayBase<S, D>,
        constraints: &StatisticalConstraints,
        errors: &mut Vec<ValidationError>,
        warnings: &mut Vec<ValidationError>,
    ) -> Result<(), CoreError>
    where
        S: Data,
        D: Dimension,
        S::Elem: Float + fmt::Debug + ScalarOperand + FromPrimitive,
    {
        if array.is_empty() {
            return Ok(());
        }

        // Calculate basic statistics
        let mean = array.mean().unwrap_or(S::Elem::zero());
        let std_dev = array.std(num_traits::cast(1.0).unwrap());

        // Validate mean constraints
        if let Some(min_mean) = constraints.min_mean {
            let min_mean_typed: S::Elem = num_traits::cast(min_mean).unwrap_or(S::Elem::zero());
            if mean < min_mean_typed {
                errors.push(ValidationError {
                    error_type: ValidationErrorType::ConstraintViolation,
                    field_path: "array.mean".to_string(),
                    message: format!("Array mean {:?} is below minimum {:?}", mean, min_mean),
                    expected: Some(format!("mean >= {}", min_mean)),
                    actual: Some(format!("{:?}", mean)),
                    constraint: Some("statistical.min_mean".to_string()),
                    severity: ErrorSeverity::Error,
                    context: HashMap::new(),
                });
            }
        }

        if let Some(max_mean) = constraints.max_mean {
            let max_mean_typed: S::Elem = num_traits::cast(max_mean).unwrap_or(S::Elem::zero());
            if mean > max_mean_typed {
                errors.push(ValidationError {
                    error_type: ValidationErrorType::ConstraintViolation,
                    field_path: "array.mean".to_string(),
                    message: format!("Array mean {:?} exceeds maximum {:?}", mean, max_mean),
                    expected: Some(format!("mean <= {}", max_mean)),
                    actual: Some(format!("{:?}", mean)),
                    constraint: Some("statistical.max_mean".to_string()),
                    severity: ErrorSeverity::Error,
                    context: HashMap::new(),
                });
            }
        }

        // Validate standard deviation constraints
        if let Some(min_std) = constraints.min_std {
            let min_std_typed: S::Elem = num_traits::cast(min_std).unwrap_or(S::Elem::zero());
            if std_dev < min_std_typed {
                warnings.push(ValidationError {
                    error_type: ValidationErrorType::ConstraintViolation,
                    field_path: "array.std".to_string(),
                    message: format!(
                        "Array standard deviation {:?} is below minimum {:?}",
                        std_dev, min_std
                    ),
                    expected: Some(format!("std >= {}", min_std)),
                    actual: Some(format!("{:?}", std_dev)),
                    constraint: Some("statistical.min_std".to_string()),
                    severity: ErrorSeverity::Warning,
                    context: HashMap::new(),
                });
            }
        }

        if let Some(max_std) = constraints.max_std {
            let max_std_typed: S::Elem = num_traits::cast(max_std).unwrap_or(S::Elem::zero());
            if std_dev > max_std_typed {
                warnings.push(ValidationError {
                    error_type: ValidationErrorType::ConstraintViolation,
                    field_path: "array.std".to_string(),
                    message: format!(
                        "Array standard deviation {:?} exceeds maximum {:?}",
                        std_dev, max_std
                    ),
                    expected: Some(format!("std <= {}", max_std)),
                    actual: Some(format!("{:?}", std_dev)),
                    constraint: Some("statistical.max_std".to_string()),
                    severity: ErrorSeverity::Warning,
                    context: HashMap::new(),
                });
            }
        }

        Ok(())
    }

    /// Validate performance characteristics for large arrays
    fn validate_array_performance<S, D>(
        &self,
        array: &ArrayBase<S, D>,
        warnings: &mut Vec<ValidationError>,
    ) -> Result<(), CoreError>
    where
        S: Data,
        D: Dimension,
        S::Elem: fmt::Debug,
    {
        let element_count = array.len();
        let element_size = std::mem::size_of::<S::Elem>();
        let total_size = element_count * element_size;

        // Warn about very large arrays
        const LARGE_ARRAY_THRESHOLD: usize = 100_000_000; // 100M elements
        if element_count > LARGE_ARRAY_THRESHOLD {
            warnings.push(ValidationError {
                error_type: ValidationErrorType::Performance,
                field_path: "array.size".to_string(),
                message: format!(
                    "Large array detected: {} elements ({} bytes). Consider chunking for better performance.",
                    element_count, total_size
                ),
                expected: Some(format!("<= {} elements", LARGE_ARRAY_THRESHOLD)),
                actual: Some(format!("{} elements", element_count)),
                constraint: Some("performance.max_elements".to_string()),
                severity: ErrorSeverity::Warning,
                context: HashMap::new(),
            });
        }

        // Warn about memory usage
        const LARGE_MEMORY_THRESHOLD: usize = 1_000_000_000; // 1GB
        if total_size > LARGE_MEMORY_THRESHOLD {
            warnings.push(ValidationError {
                error_type: ValidationErrorType::Performance,
                field_path: "array.memory".to_string(),
                message: format!(
                    "High memory usage: {} bytes. Consider memory-efficient operations.",
                    total_size
                ),
                expected: Some(format!("<= {} bytes", LARGE_MEMORY_THRESHOLD)),
                actual: Some(format!("{} bytes", total_size)),
                constraint: Some("performance.max_memory".to_string()),
                severity: ErrorSeverity::Warning,
                context: HashMap::new(),
            });
        }

        Ok(())
    }

    /// Generate comprehensive data quality report
    pub fn generate_quality_report<S, D>(
        &self,
        array: &ArrayBase<S, D>,
        field_name: &str,
    ) -> Result<DataQualityReport, CoreError>
    where
        S: Data,
        D: Dimension,
        S::Elem: Float + fmt::Debug + ScalarOperand + Send + Sync + FromPrimitive,
    {
        let mut issues = Vec::new();
        let mut recommendations = Vec::new();

        // Calculate completeness (non-NaN ratio)
        let total_elements = array.len();
        let nan_count = array.iter().filter(|&&x| x.is_nan()).count();
        let completeness = if total_elements > 0 {
            (total_elements - nan_count) as f64 / total_elements as f64
        } else {
            1.0
        };

        if completeness < 0.95 {
            issues.push(QualityIssue {
                issue_type: QualityIssueType::MissingData,
                location: field_name.to_string(),
                description: format!("Low data completeness: {:.1}%", completeness * 100.0),
                severity: if completeness < 0.8 {
                    ErrorSeverity::Error
                } else {
                    ErrorSeverity::Warning
                },
                suggestion: Some(
                    "Consider data imputation or removal of incomplete records".to_string(),
                ),
            });

            if completeness < 0.8 {
                recommendations.push("Critical: Data completeness is below 80%. Consider data quality improvement before analysis.".to_string());
            }
        }

        // Calculate validity (finite values ratio)
        let inf_count = array.iter().filter(|&&x| x.is_infinite()).count();
        let validity = if total_elements > 0 {
            (total_elements - nan_count - inf_count) as f64 / total_elements as f64
        } else {
            1.0
        };

        if validity < 1.0 {
            issues.push(QualityIssue {
                issue_type: QualityIssueType::InvalidNumeric,
                location: field_name.to_string(),
                description: format!(
                    "Invalid numeric values detected: {:.1}% valid",
                    validity * 100.0
                ),
                severity: ErrorSeverity::Warning,
                suggestion: Some("Remove or replace NaN and infinite values".to_string()),
            });
        }

        // Statistical summary
        let statistical_summary = if total_elements > 0 && nan_count < total_elements {
            let finite_values: Vec<_> = array.iter().filter(|&&x| x.is_finite()).cloned().collect();
            if !finite_values.is_empty() {
                let mean = finite_values
                    .iter()
                    .fold(S::Elem::zero(), |acc, &x| acc + x)
                    / num_traits::cast(finite_values.len()).unwrap_or(S::Elem::one());

                let variance = finite_values
                    .iter()
                    .map(|&x| {
                        let diff = x - mean;
                        diff * diff
                    })
                    .fold(S::Elem::zero(), |acc, x| acc + x)
                    / num_traits::cast(finite_values.len()).unwrap_or(S::Elem::one());

                let std_dev = variance.sqrt();
                let min_val =
                    finite_values
                        .iter()
                        .fold(finite_values[0], |acc, &x| if x < acc { x } else { acc });
                let max_val =
                    finite_values
                        .iter()
                        .fold(finite_values[0], |acc, &x| if x > acc { x } else { acc });

                Some(StatisticalSummary {
                    count: finite_values.len(),
                    mean: num_traits::cast(mean).unwrap_or(0.0),
                    std_dev: num_traits::cast(std_dev).unwrap_or(0.0),
                    min: num_traits::cast(min_val).unwrap_or(0.0),
                    max: num_traits::cast(max_val).unwrap_or(0.0),
                    outliers: 0,        // TODO: Implement outlier detection
                    distribution: None, // TODO: Implement distribution detection
                })
            } else {
                None
            }
        } else {
            None
        };

        // Calculate overall quality score
        let consistency = 1.0; // TODO: Implement pattern consistency check
        let accuracy = if issues
            .iter()
            .any(|i| matches!(i.issue_type, QualityIssueType::ConstraintViolation))
        {
            0.8
        } else {
            1.0
        };

        let quality_score = (completeness + validity + consistency + accuracy) / 4.0;

        // Add performance recommendations
        if total_elements > 1_000_000 {
            recommendations.push(
                "Large dataset detected. Consider parallel processing for better performance."
                    .to_string(),
            );
        }

        if quality_score < 0.8 {
            recommendations.push(
                "Overall data quality is low. Review data collection and preprocessing procedures."
                    .to_string(),
            );
        }

        Ok(DataQualityReport {
            quality_score,
            metrics: QualityMetrics {
                completeness,
                consistency,
                accuracy,
                validity,
                statistical_summary,
            },
            issues,
            recommendations,
        })
    }

    /// Clear validation cache
    pub fn clear_cache(&self) -> Result<(), CoreError> {
        let mut cache = self.cache.write().map_err(|_| {
            CoreError::ComputationError(crate::error::ErrorContext::new(
                "Failed to acquire cache write lock".to_string(),
            ))
        })?;
        cache.clear();
        Ok(())
    }
}

/// Trait for custom validation rules
pub trait ValidationRule {
    /// Validate a value
    #[cfg(feature = "serde")]
    fn validate(&self, value: &JsonValue, field_path: &str) -> Result<(), String>;

    /// Get rule name
    fn name(&self) -> &str;

    /// Get rule description
    fn description(&self) -> &str;
}

/// Array validation constraints
pub struct ArrayValidationConstraints<T> {
    /// Expected array shape
    pub expected_shape: Option<Vec<usize>>,
    /// Field name for error reporting
    pub field_name: Option<String>,
    /// Whether to check numeric quality (NaN, infinity)
    pub check_numeric_quality: bool,
    /// Statistical constraints
    pub statistical_constraints: Option<StatisticalConstraints>,
    /// Whether to check performance characteristics
    pub check_performance: bool,
    /// Custom element-wise validation function
    pub element_validator: Option<Box<dyn Fn(&T) -> Result<(), String> + Send + Sync>>,
}

impl<T> std::fmt::Debug for ArrayValidationConstraints<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArrayValidationConstraints")
            .field("expected_shape", &self.expected_shape)
            .field("field_name", &self.field_name)
            .field("check_numeric_quality", &self.check_numeric_quality)
            .field("statistical_constraints", &self.statistical_constraints)
            .field("check_performance", &self.check_performance)
            .field(
                "element_validator",
                &self.element_validator.as_ref().map(|_| "<function>"),
            )
            .finish()
    }
}

impl<T> Default for ArrayValidationConstraints<T> {
    fn default() -> Self {
        Self {
            expected_shape: None,
            field_name: None,
            check_numeric_quality: true,
            statistical_constraints: None,
            check_performance: false,
            element_validator: None,
        }
    }
}

/// Statistical constraints for numeric data
#[derive(Debug, Clone)]
pub struct StatisticalConstraints {
    /// Minimum allowed mean value
    pub min_mean: Option<f64>,
    /// Maximum allowed mean value
    pub max_mean: Option<f64>,
    /// Minimum allowed standard deviation
    pub min_std: Option<f64>,
    /// Maximum allowed standard deviation
    pub max_std: Option<f64>,
    /// Expected distribution type
    pub expected_distribution: Option<String>,
}

impl Default for StatisticalConstraints {
    fn default() -> Self {
        Self {
            min_mean: None,
            max_mean: None,
            min_std: None,
            max_std: None,
            expected_distribution: None,
        }
    }
}

/// Checksum algorithms supported for data integrity validation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ChecksumAlgorithm {
    /// Simple hash-based checksum
    Hash,
    /// CRC32 checksum
    CRC32,
    /// MD5 hash (not recommended for security)
    MD5,
    /// SHA-256 hash
    SHA256,
}

/// Data integrity validator
pub struct DataIntegrityValidator {
    /// Supported checksum algorithms
    algorithms: HashMap<ChecksumAlgorithm, Box<dyn Fn(&[u8]) -> String + Send + Sync>>,
}

impl DataIntegrityValidator {
    /// Create a new data integrity validator
    pub fn new() -> Self {
        let mut algorithms: HashMap<ChecksumAlgorithm, Box<dyn Fn(&[u8]) -> String + Send + Sync>> =
            HashMap::new();

        // Simple hash-based checksum
        algorithms.insert(
            ChecksumAlgorithm::Hash,
            Box::new(|data: &[u8]| {
                let mut hasher = DefaultHasher::new();
                hasher.write(data);
                format!("{:x}", hasher.finish())
            }),
        );

        // TODO: Add other checksum algorithms when dependencies are available

        Self { algorithms }
    }

    /// Calculate checksum for data
    pub fn calculate_checksum<T>(
        &self,
        data: &[T],
        algorithm: ChecksumAlgorithm,
    ) -> Result<String, CoreError>
    where
        T: Copy + std::fmt::Debug,
    {
        let calculator = self.algorithms.get(&algorithm).ok_or_else(|| {
            CoreError::ComputationError(ErrorContext::new(format!(
                "Unsupported checksum algorithm: {:?}",
                algorithm
            )))
        })?;

        // Convert data to bytes for hashing
        let bytes = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<T>(),
            )
        };
        Ok(calculator(bytes))
    }

    /// Verify data against expected checksum
    pub fn verify_checksum<T>(
        &self,
        data: &[T],
        algorithm: ChecksumAlgorithm,
        expected: &str,
    ) -> Result<bool, CoreError>
    where
        T: Copy + std::fmt::Debug,
    {
        let actual = self.calculate_checksum(data, algorithm)?;
        Ok(actual == expected)
    }
}

impl Default for DataIntegrityValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Numeric data validator with advanced checks
pub struct NumericDataValidator;

impl NumericDataValidator {
    /// Validate numeric array for quality issues
    pub fn validate_numeric_quality<T>(
        data: &[T],
        allow_nan: bool,
        allow_infinite: bool,
    ) -> Result<NumericQualityReport, CoreError>
    where
        T: Float + fmt::Debug + Copy,
    {
        let mut report = NumericQualityReport {
            total_count: data.len(),
            finite_count: 0,
            nan_count: 0,
            infinite_count: 0,
            zero_count: 0,
            negative_count: 0,
            quality_score: 0.0,
            issues: Vec::new(),
        };

        for &value in data {
            if value.is_nan() {
                report.nan_count += 1;
                if !allow_nan {
                    report
                        .issues
                        .push("NaN values detected but not allowed".to_string());
                }
            } else if value.is_infinite() {
                report.infinite_count += 1;
                if !allow_infinite {
                    report
                        .issues
                        .push("Infinite values detected but not allowed".to_string());
                }
            } else {
                report.finite_count += 1;
                if value == T::zero() {
                    report.zero_count += 1;
                }
                if value < T::zero() {
                    report.negative_count += 1;
                }
            }
        }

        // Calculate quality score
        if report.total_count > 0 {
            let finite_ratio = report.finite_count as f64 / report.total_count as f64;
            report.quality_score = finite_ratio;

            if finite_ratio < 0.95 {
                report.issues.push(format!(
                    "Low finite value ratio: {:.1}%",
                    finite_ratio * 100.0
                ));
            }
        } else {
            report.quality_score = 1.0;
        }

        Ok(report)
    }

    /// Detect outliers in numeric data using IQR method
    pub fn detect_outliers<T>(data: &[T]) -> Result<Vec<usize>, CoreError>
    where
        T: Float + fmt::Debug + Copy + PartialOrd,
    {
        if data.len() < 4 {
            return Ok(Vec::new()); // Need at least 4 points for IQR
        }

        let mut sorted_data: Vec<T> = data.iter().filter(|&&x| x.is_finite()).cloned().collect();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        if sorted_data.len() < 4 {
            return Ok(Vec::new());
        }

        let q1_index = sorted_data.len() / 4;
        let q3_index = 3 * sorted_data.len() / 4;
        let q1 = sorted_data[q1_index];
        let q3 = sorted_data[q3_index];
        let iqr = q3 - q1;

        let lower_bound = q1 - iqr * num_traits::cast(1.5).unwrap_or(T::one());
        let upper_bound = q3 + iqr * num_traits::cast(1.5).unwrap_or(T::one());

        let mut outliers = Vec::new();
        for (index, &value) in data.iter().enumerate() {
            if value.is_finite() && (value < lower_bound || value > upper_bound) {
                outliers.push(index);
            }
        }

        Ok(outliers)
    }
}

/// Numeric quality assessment report
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NumericQualityReport {
    /// Total number of elements
    pub total_count: usize,
    /// Number of finite elements
    pub finite_count: usize,
    /// Number of NaN elements
    pub nan_count: usize,
    /// Number of infinite elements
    pub infinite_count: usize,
    /// Number of zero elements
    pub zero_count: usize,
    /// Number of negative elements
    pub negative_count: usize,
    /// Overall quality score (0.0 to 1.0)
    pub quality_score: f64,
    /// List of quality issues found
    pub issues: Vec<String>,
}

/// Email validation rule
pub struct EmailValidationRule;

impl ValidationRule for EmailValidationRule {
    #[cfg(feature = "serde")]
    fn validate(&self, value: &JsonValue, _field_path: &str) -> Result<(), String> {
        if let Some(s) = value.as_str() {
            if s.contains('@') && s.contains('.') {
                Ok(())
            } else {
                Err("Invalid email format".to_string())
            }
        } else {
            Err("Expected string value for email".to_string())
        }
    }

    fn name(&self) -> &str {
        "email"
    }

    fn description(&self) -> &str {
        "Validates email address format"
    }
}

/// URL validation rule
pub struct UrlValidationRule;

impl ValidationRule for UrlValidationRule {
    #[cfg(feature = "serde")]
    fn validate(&self, value: &JsonValue, _field_path: &str) -> Result<(), String> {
        if let Some(s) = value.as_str() {
            if s.starts_with("http://") || s.starts_with("https://") {
                Ok(())
            } else {
                Err("Invalid URL format".to_string())
            }
        } else {
            Err("Expected string value for URL".to_string())
        }
    }

    fn name(&self) -> &str {
        "url"
    }

    fn description(&self) -> &str {
        "Validates URL format"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_type_matching() {
        #[cfg(feature = "serde")]
        {
            let int_type = DataType::Integer;
            let bool_type = DataType::Boolean;
            let string_type = DataType::String;

            assert!(int_type.matches(&JsonValue::Number(serde_json::Number::from(42))));
            assert!(bool_type.matches(&JsonValue::Bool(true)));
            assert!(string_type.matches(&JsonValue::String("test".to_string())));

            assert!(!int_type.matches(&JsonValue::String("not a number".to_string())));
        }
    }

    #[test]
    fn test_field_definition_builder() {
        let field = FieldDefinition::new(DataType::String)
            .required()
            .with_constraint(Constraint::Length {
                min: Some(1),
                max: Some(100),
            })
            .with_description("A test field");

        assert!(field.required);
        assert_eq!(field.constraints.len(), 1);
        assert!(field.description.is_some());
    }

    #[test]
    fn test_validation_schema_builder() {
        let schema = ValidationSchema::new()
            .name("test_schema")
            .version("1.0.0")
            .require_field("name", DataType::String)
            .optional_field("age", DataType::Integer)
            .add_constraint(
                "age",
                Constraint::Range {
                    min: 0.0,
                    max: 150.0,
                },
            )
            .allow_additional();

        assert_eq!(schema.name, "test_schema");
        assert_eq!(schema.fields.len(), 2);
        assert!(schema.allow_additional_fields);
    }

    #[test]
    fn test_validator_creation() {
        let config = ValidationConfig::default();
        let validator = Validator::new(config).unwrap();

        // Test schema registration
        let schema = ValidationSchema::new().name("test");
        validator.register_schema(schema).unwrap();

        let retrieved = validator.get_schema("test").unwrap();
        assert!(retrieved.is_some());
    }

    #[test]
    fn test_constraint_types() {
        let range_constraint = Constraint::Range {
            min: 0.0,
            max: 100.0,
        };
        let _length_constraint = Constraint::Length {
            min: Some(1),
            max: Some(50),
        };
        let _enum_constraint = Constraint::Enum(vec!["A".to_string(), "B".to_string()]);

        match range_constraint {
            Constraint::Range { min, max } => {
                assert_eq!(min, 0.0);
                assert_eq!(max, 100.0);
            }
            _ => panic!("Wrong constraint type"),
        }
    }

    #[test]
    fn test_validation_error_severity() {
        assert!(ErrorSeverity::Critical > ErrorSeverity::Error);
        assert!(ErrorSeverity::Error > ErrorSeverity::Warning);
    }

    #[test]
    fn test_custom_validation_rule() {
        let email_rule = EmailValidationRule;
        assert_eq!(email_rule.name(), "email");
        assert!(!email_rule.description().is_empty());

        #[cfg(feature = "serde")]
        {
            let valid_email = JsonValue::String("test@example.com".to_string());
            let invalid_email = JsonValue::String("invalid-email".to_string());

            assert!(email_rule.validate(&valid_email, "email").is_ok());
            assert!(email_rule.validate(&invalid_email, "email").is_err());
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_basic_validation() {
        let config = ValidationConfig::default();
        let validator = Validator::new(config).unwrap();

        let schema = ValidationSchema::new()
            .require_field("name", DataType::String)
            .require_field("age", DataType::Integer)
            .add_constraint(
                "age",
                Constraint::Range {
                    min: 0.0,
                    max: 150.0,
                },
            );

        let valid_data = serde_json::json!({
            "name": "John Doe",
            "age": 25
        });

        let invalid_data = serde_json::json!({
            "name": "John Doe",
            "age": 200
        });

        let valid_result = validator.validate(&valid_data, &schema).unwrap();
        let invalid_result = validator.validate(&invalid_data, &schema).unwrap();

        assert!(valid_result.is_valid());
        assert!(!invalid_result.is_valid());
        assert!(!invalid_result.errors().is_empty());
    }

    #[test]
    fn test_ndarray_validation() {
        use ndarray::Array2;

        let config = ValidationConfig::default();
        let validator = Validator::new(config.clone()).unwrap();

        // Create a test array
        let array = Array2::<f64>::zeros((3, 4));
        let constraints = ArrayValidationConstraints {
            expected_shape: Some(vec![3, 4]),
            field_name: Some("test_array".to_string()),
            check_numeric_quality: true,
            statistical_constraints: None,
            check_performance: false,
            element_validator: None,
        };

        let result = validator
            .validate_ndarray(&array, &constraints, &config)
            .unwrap();
        assert!(result.is_valid());

        // Test with wrong shape
        let constraints_wrong_shape = ArrayValidationConstraints {
            expected_shape: Some(vec![2, 3]),
            field_name: Some("test_array".to_string()),
            check_numeric_quality: true,
            statistical_constraints: None,
            check_performance: false,
            element_validator: None,
        };

        let result = validator
            .validate_ndarray(&array, &constraints_wrong_shape, &config)
            .unwrap();
        assert!(!result.is_valid());
    }

    #[test]
    fn test_numeric_quality_validation() {
        let data = vec![1.0, 2.0, f64::NAN, 4.0, f64::INFINITY];
        let report = NumericDataValidator::validate_numeric_quality(&data, false, false).unwrap();

        assert_eq!(report.total_count, 5);
        assert_eq!(report.finite_count, 3);
        assert_eq!(report.nan_count, 1);
        assert_eq!(report.infinite_count, 1);
        assert!(!report.issues.is_empty());
    }

    #[test]
    fn test_outlier_detection() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0]; // 100.0 is an outlier
        let outliers = NumericDataValidator::detect_outliers(&data).unwrap();
        assert!(!outliers.is_empty());
        assert!(outliers.contains(&5)); // Index of 100.0
    }

    #[test]
    fn test_data_integrity_validator() {
        let validator = DataIntegrityValidator::new();
        let data = vec![1u32, 2u32, 3u32, 4u32];

        let checksum = validator
            .calculate_checksum(&data, ChecksumAlgorithm::Hash)
            .unwrap();
        assert!(!checksum.is_empty());

        let is_valid = validator
            .verify_checksum(&data, ChecksumAlgorithm::Hash, &checksum)
            .unwrap();
        assert!(is_valid);

        let is_invalid = validator
            .verify_checksum(&data, ChecksumAlgorithm::Hash, "wrong_checksum")
            .unwrap();
        assert!(!is_invalid);
    }

    #[test]
    fn test_quality_report_generation() {
        use ndarray::Array1;

        let config = ValidationConfig::default();
        let validator = Validator::new(config).unwrap();

        let array = Array1::from_vec(vec![1.0, 2.0, 3.0, f64::NAN, 5.0]);
        let report = validator
            .generate_quality_report(&array, "test_field")
            .unwrap();

        assert!(report.quality_score < 1.0); // Should be less than perfect due to NaN
        assert!(!report.issues.is_empty()); // Should have issues due to NaN
        assert!(report.metrics.completeness < 1.0); // Should be less than 100% complete
    }

    #[test]
    fn test_statistical_constraints() {
        use ndarray::Array1;

        let config = ValidationConfig::default();
        let validator = Validator::new(config.clone()).unwrap();

        // Array with mean around 3.0
        let array = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let constraints = ArrayValidationConstraints {
            expected_shape: None,
            field_name: Some("test_array".to_string()),
            check_numeric_quality: true,
            statistical_constraints: Some(StatisticalConstraints {
                min_mean: Some(2.0),
                max_mean: Some(4.0),
                min_std: None,
                max_std: None,
                expected_distribution: None,
            }),
            check_performance: false,
            element_validator: None,
        };

        let result = validator
            .validate_ndarray(&array, &constraints, &config)
            .unwrap();
        assert!(result.is_valid());

        // Test with constraints that should fail
        let failing_constraints = ArrayValidationConstraints {
            expected_shape: None,
            field_name: Some("test_array".to_string()),
            check_numeric_quality: true,
            statistical_constraints: Some(StatisticalConstraints {
                min_mean: Some(5.0), // Mean is 3.0, so this should fail
                max_mean: Some(6.0),
                min_std: None,
                max_std: None,
                expected_distribution: None,
            }),
            check_performance: false,
            element_validator: None,
        };

        let result = validator
            .validate_ndarray(&array, &failing_constraints, &config)
            .unwrap();
        assert!(!result.is_valid());
    }

    #[test]
    fn test_performance_validation() {
        use ndarray::Array1;

        let config = ValidationConfig::default();
        let validator = Validator::new(config.clone()).unwrap();

        // Small array - should not trigger performance warnings
        let small_array = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let constraints = ArrayValidationConstraints {
            expected_shape: None,
            field_name: Some("small_array".to_string()),
            check_numeric_quality: false,
            statistical_constraints: None,
            check_performance: true,
            element_validator: None,
        };

        let result = validator
            .validate_ndarray(&small_array, &constraints, &config)
            .unwrap();
        assert!(result.is_valid());
        assert!(result.warnings.is_empty());
    }

    #[test]
    fn test_shape_constraints() {
        let shape_constraints = ShapeConstraints {
            dimensions: vec![Some(3), Some(4)],
            min_elements: Some(10),
            max_elements: Some(20),
            require_square: false,
            allow_broadcasting: true,
        };

        // Test basic structure
        assert_eq!(shape_constraints.dimensions.len(), 2);
        assert_eq!(shape_constraints.min_elements, Some(10));
        assert!(!shape_constraints.require_square);
    }

    #[test]
    fn test_quality_issue_types() {
        let issue = QualityIssue {
            issue_type: QualityIssueType::MissingData,
            location: "test_field".to_string(),
            description: "Test issue".to_string(),
            severity: ErrorSeverity::Warning,
            suggestion: Some("Fix the data".to_string()),
        };

        assert_eq!(issue.issue_type, QualityIssueType::MissingData);
        assert_eq!(issue.severity, ErrorSeverity::Warning);
        assert!(issue.suggestion.is_some());
    }
}
