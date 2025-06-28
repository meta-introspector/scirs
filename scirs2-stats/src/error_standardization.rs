//! Error message standardization for consistent error handling
//!
//! This module provides standardized error messages and recovery suggestions
//! that are used consistently across all statistical modules.

use crate::error::{StatsError, StatsResult};
use std::collections::HashMap;

/// Standardized error message templates
pub struct ErrorMessages;

impl ErrorMessages {
    /// Standard dimension mismatch messages
    pub fn dimension_mismatch(expected: &str, actual: &str) -> StatsError {
        StatsError::dimension_mismatch(format!(
            "Array dimension mismatch: expected {}, got {}. {}",
            expected,
            actual,
            "Ensure all input arrays have compatible dimensions for the operation."
        ))
    }

    /// Standard array length mismatch messages
    pub fn length_mismatch(
        array1_name: &str,
        len1: usize,
        array2_name: &str,
        len2: usize,
    ) -> StatsError {
        StatsError::dimension_mismatch(format!(
            "Array length mismatch: {} has {} elements, {} has {} elements. {}",
            array1_name,
            len1,
            array2_name,
            len2,
            "Both arrays must have the same number of elements."
        ))
    }

    /// Standard empty array messages
    pub fn empty_array(array_name: &str) -> StatsError {
        StatsError::invalid_argument(format!(
            "Array '{}' cannot be empty. {}",
            array_name, "Provide an array with at least one element."
        ))
    }

    /// Standard insufficient data messages
    pub fn insufficient_data(operation: &str, required: usize, actual: usize) -> StatsError {
        StatsError::invalid_argument(format!(
            "Insufficient data for {}: requires at least {} elements, got {}. {}",
            operation,
            required,
            actual,
            if required == 2 {
                "Statistical calculations typically require at least 2 data points."
            } else {
                "Increase the sample size or use a different method."
            }
        ))
    }

    /// Standard non-positive value messages
    pub fn non_positive_value(parameter: &str, value: f64) -> StatsError {
        StatsError::domain(format!(
            "Parameter '{}' must be positive, got {}. {}",
            parameter, value, "Ensure the value is greater than 0."
        ))
    }

    /// Standard probability range messages
    pub fn invalid_probability(parameter: &str, value: f64) -> StatsError {
        StatsError::domain(format!(
            "Parameter '{}' must be a valid probability between 0 and 1, got {}. {}",
            parameter, value, "Probability values must be in the range [0, 1]."
        ))
    }

    /// Standard NaN detection messages
    pub fn nan_detected(context: &str) -> StatsError {
        StatsError::invalid_argument(format!(
            "NaN (Not a Number) values detected in {}. {}",
            context, "Remove NaN values or use functions that handle missing data explicitly."
        ))
    }

    /// Standard infinite value messages
    pub fn infinite_value_detected(context: &str) -> StatsError {
        StatsError::invalid_argument(format!(
            "Infinite values detected in {}. {}",
            context, "Check for overflow conditions or extreme values in your data."
        ))
    }

    /// Standard matrix not positive definite messages
    pub fn not_positive_definite(matrix_name: &str) -> StatsError {
        StatsError::computation(format!(
            "Matrix '{}' is not positive definite. {}",
            matrix_name,
            "Ensure the matrix is symmetric and all eigenvalues are positive, or use regularization."
        ))
    }

    /// Standard singular matrix messages
    pub fn singular_matrix(matrix_name: &str) -> StatsError {
        StatsError::computation(format!(
            "Matrix '{}' is singular (non-invertible). {}",
            matrix_name, "Check for linear dependencies in your data or add regularization."
        ))
    }

    /// Standard convergence failure messages
    pub fn convergence_failure(algorithm: &str, iterations: usize) -> StatsError {
        StatsError::ConvergenceError(format!(
            "{} failed to converge after {} iterations. {}",
            algorithm, iterations,
            "Try increasing the maximum iterations, adjusting tolerance, or using different initial values."
        ))
    }

    /// Standard numerical instability messages
    pub fn numerical_instability(operation: &str, suggestion: &str) -> StatsError {
        StatsError::computation(format!(
            "Numerical instability detected in {}. {}",
            operation, suggestion
        ))
    }

    /// Standard unsupported operation messages
    pub fn unsupported_operation(operation: &str, context: &str) -> StatsError {
        StatsError::not_implemented(format!(
            "Operation '{}' is not supported for {}. {}",
            operation,
            context,
            "Check the documentation for supported operations or consider alternative methods."
        ))
    }
}

/// Context-aware error validation
pub struct ErrorValidator;

impl ErrorValidator {
    /// Validate array for common issues
    pub fn validate_array<T>(data: &[T], name: &str) -> StatsResult<()>
    where
        T: PartialOrd + Copy,
    {
        if data.is_empty() {
            return Err(ErrorMessages::empty_array(name));
        }
        Ok(())
    }

    /// Validate array for finite values (for float types)
    pub fn validate_finite_array(data: &[f64], name: &str) -> StatsResult<()> {
        if data.is_empty() {
            return Err(ErrorMessages::empty_array(name));
        }

        for (i, &value) in data.iter().enumerate() {
            if value.is_nan() {
                return Err(ErrorMessages::nan_detected(&format!("{}[{}]", name, i)));
            }
            if value.is_infinite() {
                return Err(ErrorMessages::infinite_value_detected(&format!(
                    "{}[{}]",
                    name, i
                )));
            }
        }
        Ok(())
    }

    /// Validate probability value
    pub fn validate_probability(value: f64, name: &str) -> StatsResult<()> {
        if value < 0.0 || value > 1.0 {
            return Err(ErrorMessages::invalid_probability(name, value));
        }
        if value.is_nan() {
            return Err(ErrorMessages::nan_detected(name));
        }
        Ok(())
    }

    /// Validate positive value
    pub fn validate_positive(value: f64, name: &str) -> StatsResult<()> {
        if value <= 0.0 {
            return Err(ErrorMessages::non_positive_value(name, value));
        }
        if value.is_nan() {
            return Err(ErrorMessages::nan_detected(name));
        }
        if value.is_infinite() {
            return Err(ErrorMessages::infinite_value_detected(name));
        }
        Ok(())
    }

    /// Validate arrays have same length
    pub fn validate_same_length<T, U>(
        arr1: &[T],
        arr1_name: &str,
        arr2: &[U],
        arr2_name: &str,
    ) -> StatsResult<()> {
        if arr1.len() != arr2.len() {
            return Err(ErrorMessages::length_mismatch(
                arr1_name,
                arr1.len(),
                arr2_name,
                arr2.len(),
            ));
        }
        Ok(())
    }

    /// Validate minimum sample size
    pub fn validate_sample_size(size: usize, minimum: usize, operation: &str) -> StatsResult<()> {
        if size < minimum {
            return Err(ErrorMessages::insufficient_data(operation, minimum, size));
        }
        Ok(())
    }
}

/// Performance impact assessment for error recovery
#[derive(Debug, Clone, Copy)]
pub enum PerformanceImpact {
    /// No performance impact
    None,
    /// Minimal performance impact (< 5%)
    Minimal,
    /// Moderate performance impact (5-20%)
    Moderate,
    /// Significant performance impact (> 20%)
    Significant,
}

/// Standardized error recovery suggestions
pub struct RecoverySuggestions;

impl RecoverySuggestions {
    /// Get recovery suggestions for common statistical errors
    pub fn get_suggestions(error: &StatsError) -> Vec<(String, PerformanceImpact)> {
        match error {
            StatsError::DimensionMismatch(_) => vec![
                (
                    "Reshape arrays to have compatible dimensions".to_string(),
                    PerformanceImpact::None,
                ),
                (
                    "Use broadcasting-compatible operations".to_string(),
                    PerformanceImpact::Minimal,
                ),
                (
                    "Transpose matrices if needed".to_string(),
                    PerformanceImpact::Minimal,
                ),
            ],
            StatsError::InvalidArgument(msg) if msg.contains("empty") => vec![
                (
                    "Provide non-empty input arrays".to_string(),
                    PerformanceImpact::None,
                ),
                (
                    "Use default values for empty inputs".to_string(),
                    PerformanceImpact::Minimal,
                ),
                (
                    "Filter out empty arrays before processing".to_string(),
                    PerformanceImpact::Minimal,
                ),
            ],
            StatsError::InvalidArgument(msg) if msg.contains("NaN") => vec![
                (
                    "Remove NaN values using data.dropna()".to_string(),
                    PerformanceImpact::Minimal,
                ),
                (
                    "Use interpolation to fill NaN values".to_string(),
                    PerformanceImpact::Moderate,
                ),
                (
                    "Use statistical methods that handle NaN explicitly".to_string(),
                    PerformanceImpact::Minimal,
                ),
            ],
            StatsError::ComputationError(msg) if msg.contains("singular") => vec![
                (
                    "Add regularization (e.g., ridge regression)".to_string(),
                    PerformanceImpact::Minimal,
                ),
                (
                    "Use pseudo-inverse instead of inverse".to_string(),
                    PerformanceImpact::Moderate,
                ),
                (
                    "Check for multicollinearity in data".to_string(),
                    PerformanceImpact::None,
                ),
            ],
            StatsError::ConvergenceError(_) => vec![
                (
                    "Increase maximum iterations".to_string(),
                    PerformanceImpact::Moderate,
                ),
                (
                    "Adjust convergence tolerance".to_string(),
                    PerformanceImpact::None,
                ),
                (
                    "Use different initial values".to_string(),
                    PerformanceImpact::None,
                ),
                (
                    "Try a different optimization algorithm".to_string(),
                    PerformanceImpact::Significant,
                ),
            ],
            StatsError::DomainError(_) => vec![
                (
                    "Check parameter bounds and constraints".to_string(),
                    PerformanceImpact::None,
                ),
                (
                    "Scale or normalize input data".to_string(),
                    PerformanceImpact::Minimal,
                ),
                (
                    "Use robust statistical methods".to_string(),
                    PerformanceImpact::Moderate,
                ),
            ],
            _ => vec![
                (
                    "Check input data for validity".to_string(),
                    PerformanceImpact::None,
                ),
                (
                    "Refer to function documentation".to_string(),
                    PerformanceImpact::None,
                ),
            ],
        }
    }

    /// Get context-specific suggestions for statistical operations
    pub fn get_context_suggestions(operation: &str) -> HashMap<String, Vec<String>> {
        let mut suggestions = HashMap::new();

        match operation {
            "correlation" => {
                suggestions.insert(
                    "data_preparation".to_string(),
                    vec![
                        "Ensure data is numeric and finite".to_string(),
                        "Consider outlier detection and removal".to_string(),
                        "Check for missing values".to_string(),
                    ],
                );
                suggestions.insert(
                    "performance".to_string(),
                    vec![
                        "Use SIMD-optimized functions for large datasets".to_string(),
                        "Consider parallel computation for correlation matrices".to_string(),
                    ],
                );
            }
            "regression" => {
                suggestions.insert(
                    "data_preparation".to_string(),
                    vec![
                        "Check for multicollinearity".to_string(),
                        "Normalize features if needed".to_string(),
                        "Consider feature selection".to_string(),
                    ],
                );
                suggestions.insert(
                    "model_selection".to_string(),
                    vec![
                        "Use regularization for high-dimensional data".to_string(),
                        "Consider robust regression for outliers".to_string(),
                    ],
                );
            }
            "hypothesis_testing" => {
                suggestions.insert(
                    "assumptions".to_string(),
                    vec![
                        "Check normality assumptions".to_string(),
                        "Verify independence of observations".to_string(),
                        "Consider non-parametric alternatives".to_string(),
                    ],
                );
                suggestions.insert(
                    "interpretation".to_string(),
                    vec![
                        "Adjust for multiple comparisons if needed".to_string(),
                        "Consider effect size in addition to p-values".to_string(),
                    ],
                );
            }
            _ => {
                suggestions.insert(
                    "general".to_string(),
                    vec![
                        "Validate input data quality".to_string(),
                        "Check function prerequisites".to_string(),
                    ],
                );
            }
        }

        suggestions
    }
}

/// Comprehensive error reporting with standardized messages
pub struct StandardizedErrorReporter;

impl StandardizedErrorReporter {
    /// Generate a comprehensive error report
    pub fn generate_report(error: &StatsError, context: Option<&str>) -> String {
        let mut report = String::new();

        // Main error message
        report.push_str(&format!("❌ Error: {}\n\n", error));

        // Context information
        if let Some(ctx) = context {
            report.push_str(&format!("📍 Context: {}\n\n", ctx));
        }

        // Recovery suggestions
        let suggestions = RecoverySuggestions::get_suggestions(error);
        if !suggestions.is_empty() {
            report.push_str("💡 Suggested Solutions:\n");
            for (i, (suggestion, impact)) in suggestions.iter().enumerate() {
                let impact_icon = match impact {
                    PerformanceImpact::None => "⚡",
                    PerformanceImpact::Minimal => "🔋",
                    PerformanceImpact::Moderate => "⏱️",
                    PerformanceImpact::Significant => "⚠️",
                };
                report.push_str(&format!("   {}. {} {}\n", i + 1, impact_icon, suggestion));
            }
            report.push('\n');
        }

        // Performance impact legend
        report.push_str("Legend: ⚡ No impact, 🔋 Minimal, ⏱️ Moderate, ⚠️ Significant\n");

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_messages() {
        let err = ErrorMessages::length_mismatch("x", 5, "y", 3);
        assert!(err.to_string().contains("Array length mismatch"));
        assert!(err.to_string().contains("same number of elements"));
    }

    #[test]
    fn test_error_validator() {
        let empty_data: &[f64] = &[];
        assert!(ErrorValidator::validate_array(empty_data, "test").is_err());

        let finite_data = [1.0, 2.0, 3.0];
        assert!(ErrorValidator::validate_finite_array(&finite_data, "test").is_ok());

        let nan_data = [1.0, f64::NAN, 3.0];
        assert!(ErrorValidator::validate_finite_array(&nan_data, "test").is_err());
    }

    #[test]
    fn test_recovery_suggestions() {
        let err = ErrorMessages::empty_array("data");
        let suggestions = RecoverySuggestions::get_suggestions(&err);
        assert!(!suggestions.is_empty());
    }
}
