//! Main validator implementation
//!
//! This module provides the core Validator struct that orchestrates all validation
//! operations and manages caching, custom rules, and configuration.

use crate::error::{CoreError, ErrorContext};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::time::Instant;

use super::array_validation::ArrayValidator;
use super::config::{ErrorSeverity, ValidationConfig, ValidationErrorType};
use super::constraints::{ArrayValidationConstraints, Constraint};
use super::errors::{ValidationError, ValidationResult, ValidationStats};
use super::quality::{DataQualityReport, QualityAnalyzer};
use super::schema::{DataType, ValidationSchema};

// Core dependencies for array/matrix validation
use ndarray::{ArrayBase, Data, Dimension, ScalarOperand};
use num_traits::{Float, FromPrimitive};
use std::fmt;

#[cfg(feature = "serde")]
use serde_json::Value as JsonValue;

#[cfg(feature = "parallel")]
// For checksum validation
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Cache entry for validation results
#[derive(Debug, Clone)]
struct CacheEntry {
    result: ValidationResult,
    timestamp: Instant,
    hit_count: usize,
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

/// Main data validator with comprehensive validation capabilities
pub struct Validator {
    /// Validation configuration
    config: ValidationConfig,
    /// Validation result cache
    cache: Arc<RwLock<HashMap<String, CacheEntry>>>,
    /// Custom validation rules
    custom_rules: HashMap<String, Box<dyn ValidationRule + Send + Sync>>,
    /// Array validator
    array_validator: ArrayValidator,
    /// Quality analyzer
    quality_analyzer: QualityAnalyzer,
}

impl Validator {
    /// Create a new validator with configuration
    pub fn new(config: ValidationConfig) -> Result<Self, CoreError> {
        Ok(Self {
            config,
            cache: Arc::new(RwLock::new(HashMap::new())),
            custom_rules: HashMap::new(),
            array_validator: ArrayValidator::new(),
            quality_analyzer: QualityAnalyzer::new(),
        })
    }

    /// Validate data against a schema
    #[cfg(feature = "serde")]
    pub fn validate(
        &self,
        data: &JsonValue,
        schema: &ValidationSchema,
    ) -> Result<ValidationResult, CoreError> {
        let start_time = Instant::now();
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let mut stats = ValidationStats::default();

        // Check cache if enabled
        if self.config.enable_caching {
            let cache_key = self.generate_cache_key(data, schema)?;
            if let Some(mut cached_result) = self.get_cached_result(&cache_key)? {
                // Update cache hit rate
                let cache_hit_rate = self.calculate_cache_hit_rate()?;
                cached_result.stats.set_cache_hit_rate(cache_hit_rate);
                return Ok(cached_result);
            }
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

        let mut result = ValidationResult {
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

        // Update cache hit rate
        if self.config.enable_caching {
            let cache_hit_rate = self.calculate_cache_hit_rate()?;
            result.stats.set_cache_hit_rate(cache_hit_rate);
        }

        Ok(result)
    }

    /// Validate ndarray with comprehensive checks
    pub fn validate_ndarray<S, D>(
        &self,
        array: &ArrayBase<S, D>,
        constraints: &ArrayValidationConstraints,
        config: &ValidationConfig,
    ) -> Result<ValidationResult, CoreError>
    where
        S: Data,
        D: Dimension,
        S::Elem: Float + fmt::Debug + Send + Sync + ScalarOperand + FromPrimitive,
    {
        self.array_validator
            .validate_ndarray(array, constraints, config)
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
        self.quality_analyzer
            .generate_quality_report(array, field_name)
    }

    /// Add a custom validation rule
    pub fn add_custom_rule(&mut self, name: String, rule: Box<dyn ValidationRule + Send + Sync>) {
        self.custom_rules.insert(name, rule);
    }

    /// Clear validation cache
    pub fn clear_cache(&self) -> Result<(), CoreError> {
        let mut cache = self.cache.write().map_err(|_| {
            CoreError::ComputationError(ErrorContext::new(
                "Failed to acquire cache write lock".to_string(),
            ))
        })?;
        cache.clear();
        Ok(())
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> Result<(usize, f64), CoreError> {
        let cache = self.cache.read().map_err(|_| {
            CoreError::ComputationError(ErrorContext::new(
                "Failed to acquire cache read lock".to_string(),
            ))
        })?;

        let size = cache.len();
        let hit_rate = self.calculate_cache_hit_rate()?;

        Ok((size, hit_rate))
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

        let data_obj = match data {
            JsonValue::Object(obj) => obj,
            _ => {
                errors.push(ValidationError {
                    error_type: ValidationErrorType::TypeMismatch,
                    field_path: "root".to_string(),
                    message: "Expected object".to_string(),
                    expected: Some("object".to_string()),
                    actual: Some(self.get_value_type_name(data)),
                    constraint: None,
                    severity: ErrorSeverity::Error,
                    context: HashMap::new(),
                });
                return Ok(());
            }
        };

        for (field_name, field_def) in &schema.fields {
            stats.add_field_validation();

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
            } else if field_def.required {
                // Required field is missing
                errors.push(ValidationError {
                    error_type: ValidationErrorType::MissingRequiredField,
                    field_path,
                    message: format!("Required field '{}' is missing", field_name),
                    expected: Some(format!("{:?}", field_def.data_type)),
                    actual: Some("missing".to_string()),
                    constraint: Some("required".to_string()),
                    severity: ErrorSeverity::Error,
                    context: HashMap::new(),
                });
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
        let type_matches = match expected_type {
            DataType::Boolean => value.is_boolean(),
            DataType::Integer => value.is_i64(),
            DataType::Float32 | DataType::Float64 => value.is_f64() || value.is_i64(),
            DataType::String => value.is_string(),
            DataType::Array(_) => value.is_array(),
            DataType::Object => value.is_object(),
            DataType::Null => value.is_null(),
            _ => true, // Other types not yet implemented
        };

        if !type_matches {
            errors.push(ValidationError {
                error_type: ValidationErrorType::TypeMismatch,
                field_path: field_path.to_string(),
                message: format!(
                    "Type mismatch: expected {:?}, got {}",
                    expected_type,
                    self.get_value_type_name(value)
                ),
                expected: Some(format!("{:?}", expected_type)),
                actual: Some(self.get_value_type_name(value)),
                constraint: Some("type".to_string()),
                severity: ErrorSeverity::Error,
                context: HashMap::new(),
            });
        }

        Ok(())
    }

    /// Validate field constraints
    #[cfg(feature = "serde")]
    fn validate_field_constraints(
        &self,
        value: &JsonValue,
        constraints: &[Constraint],
        field_path: &str,
        errors: &mut Vec<ValidationError>,
        _warnings: &mut Vec<ValidationError>,
        stats: &mut ValidationStats,
    ) -> Result<(), CoreError> {
        for constraint in constraints {
            stats.add_constraint_check();

            match constraint {
                Constraint::Range { min, max } => {
                    if let Some(num) = value.as_f64() {
                        if num < *min || num > *max {
                            errors.push(ValidationError {
                                error_type: ValidationErrorType::OutOfRange,
                                field_path: field_path.to_string(),
                                message: format!(
                                    "Value {} is out of range [{}, {}]",
                                    num, min, max
                                ),
                                expected: Some(format!("[{}, {}]", min, max)),
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
                        if len < *min || len > *max {
                            errors.push(ValidationError {
                                error_type: ValidationErrorType::ConstraintViolation,
                                field_path: field_path.to_string(),
                                message: format!(
                                    "String length {} is out of range [{}, {}]",
                                    len, min, max
                                ),
                                expected: Some(format!("length in [{}, {}]", min, max)),
                                actual: Some(len.to_string()),
                                constraint: Some("length".to_string()),
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
        let mut hasher = DefaultHasher::new();
        data.to_string().hash(&mut hasher);
        schema.name.hash(&mut hasher);
        schema.version.hash(&mut hasher);

        Ok(format!("{:x}", hasher.finish()))
    }

    /// Get cached validation result
    fn get_cached_result(&self, cache_key: &str) -> Result<Option<ValidationResult>, CoreError> {
        let cache = self.cache.read().map_err(|_| {
            CoreError::ComputationError(ErrorContext::new(
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
            CoreError::ComputationError(ErrorContext::new(
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
            timestamp: Instant::now(),
            hit_count: 0,
        };

        cache.insert(cache_key.to_string(), entry);
        Ok(())
    }

    /// Calculate cache hit rate
    fn calculate_cache_hit_rate(&self) -> Result<f64, CoreError> {
        let cache = self.cache.read().map_err(|_| {
            CoreError::ComputationError(ErrorContext::new(
                "Failed to acquire cache read lock".to_string(),
            ))
        })?;

        if cache.is_empty() {
            return Ok(0.0);
        }

        let total_hits: usize = cache.values().map(|entry| entry.hit_count).sum();
        let total_entries = cache.len();

        Ok(total_hits as f64 / total_entries as f64)
    }
}

impl Default for Validator {
    fn default() -> Self {
        Self::new(ValidationConfig::default()).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_validator_creation() {
        let config = ValidationConfig::default();
        let validator = Validator::new(config).unwrap();

        // Test basic properties
        assert!(!validator.config.strict_mode);
        assert_eq!(validator.config.max_depth, 100);
    }

    #[test]
    fn test_array_validation() {
        let config = ValidationConfig::default();
        let validator = Validator::new(config.clone()).unwrap();
        let array = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let constraints = ArrayValidationConstraints::new()
            .with_shape(vec![5])
            .with_field_name("test_array")
            .check_numeric_quality();

        let result = validator
            .validate_ndarray(&array, &constraints, &config)
            .unwrap();
        assert!(result.is_valid());
    }

    #[test]
    fn test_quality_report_generation() {
        let config = ValidationConfig::default();
        let validator = Validator::new(config).unwrap();
        let array = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let report = validator
            .generate_quality_report(&array, "test_field")
            .unwrap();

        assert!(report.quality_score > 0.9); // Should be high quality
        assert_eq!(report.metrics.completeness, 1.0); // No missing values
    }

    #[test]
    fn test_cache_management() {
        let config = ValidationConfig::default();
        let validator = Validator::new(config).unwrap();

        // Test cache clearing
        assert!(validator.clear_cache().is_ok());

        // Test cache stats
        let (size, hit_rate) = validator.get_cache_stats().unwrap();
        assert_eq!(size, 0); // Should be empty after clearing
        assert_eq!(hit_rate, 0.0); // No hits yet
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_json_validation() {
        let config = ValidationConfig::default();
        let validator = Validator::new(config).unwrap();

        let schema = ValidationSchema::new()
            .name("test_schema")
            .require_field("name", DataType::String)
            .require_field("age", DataType::Integer);

        let valid_data = serde_json::json!({
            "name": "John Doe",
            "age": 30
        });

        let result = validator.validate(&valid_data, &schema).unwrap();
        assert!(result.is_valid());

        let invalid_data = serde_json::json!({
            "name": "John Doe"
            // Missing required "age" field
        });

        let result = validator.validate(&invalid_data, &schema).unwrap();
        assert!(!result.is_valid());
        assert_eq!(result.errors().len(), 1);
    }
}
