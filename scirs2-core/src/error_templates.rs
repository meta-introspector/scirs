//! Standardized error message templates for consistent error reporting
//!
//! This module provides template functions for generating consistent error
//! messages across the entire codebase.

use std::fmt::Display;

/// Generate an error message for dimension mismatch
pub fn dimension_mismatch(operation: &str, expected: impl Display, actual: impl Display) -> String {
    format!("{operation}: dimension mismatch (expected: {expected}, actual: {actual})")
}

/// Generate an error message for shape mismatch in array operations
pub fn shape_mismatch(operation: &str, shape1: &[usize], shape2: &[usize]) -> String {
    format!("{operation}: incompatible shapes ({shape1:?} vs {shape2:?})")
}

/// Generate an error message for invalid parameter values
pub fn invalid_parameter(param_name: &str, constraint: &str, actual_value: impl Display) -> String {
    format!("Parameter '{param_name}': {constraint} (got: {actual_value})")
}

/// Generate an error message for out of bounds access
pub fn index_out_of_bounds(index: usize, length: usize) -> String {
    format!("Index out of bounds: index {index} is invalid for length {length}")
}

/// Generate an error message for empty input
pub fn empty_input(operation: &str) -> String {
    format!("{operation}: input array/collection is empty")
}

/// Generate an error message for numerical computation errors
pub fn numerical_error(operation: &str, issue: &str) -> String {
    format!("{operation}: {issue} - check input values for numerical issues")
}

/// Generate an error message for convergence failures
pub fn convergence_failed(algorithm: &str, iterations: usize, tolerance: impl Display) -> String {
    format!(
        "{algorithm}: failed to converge after {iterations} iterations (tolerance: {tolerance})"
    )
}

/// Generate an error message for not implemented features
pub fn not_implemented(feature: &str) -> String {
    format!("Feature not implemented: {feature}")
}

/// Generate an error message for invalid array dimensions
pub fn invalid_dimensions(operation: &str, requirement: &str, actual_dims: &[usize]) -> String {
    format!("{operation}: {requirement} (got: {actual_dims:?})")
}

/// Generate an error message for domain errors
pub fn domain_error(value_desc: &str, constraint: &str, value: impl Display) -> String {
    format!("{value_desc} must be {constraint} (got: {value})")
}

/// Generate an error message for allocation failures
pub fn allocation_failed(size: usize, element_type: &str) -> String {
    format!("Failed to allocate memory for {size} elements of type {element_type}")
}

/// Generate an error message for file I/O errors
pub fn io_error(operation: &str, path: &str, details: &str) -> String {
    format!("{operation} failed for '{path}': {details}")
}

/// Generate an error message for parse errors
pub fn parse_error(type_name: &str, input: &str, reason: &str) -> String {
    format!("Failed to parse '{input}' as {type_name}: {reason}")
}

/// Generate an error message for invalid state
pub fn invalid_state(object: &str, expected_state: &str, actual_state: &str) -> String {
    format!("{object} is in invalid state: expected {expected_state}, but was {actual_state}")
}

/// Generate an error message with recovery suggestion
pub fn with_suggestion(error_msg: &str, suggestion: &str) -> String {
    format!("{error_msg}\nSuggestion: {suggestion}")
}

/// Common parameter constraints
pub mod constraints {
    /// Generate constraint message for positive values
    pub fn positive() -> &'static str {
        "must be positive (> 0)"
    }

    /// Generate constraint message for non-negative values
    pub fn non_negative() -> &'static str {
        "must be non-negative (>= 0)"
    }

    /// Generate constraint message for probability values
    pub fn probability() -> &'static str {
        "must be a valid probability in [0, 1]"
    }

    /// Generate constraint message for finite values
    pub fn finite() -> &'static str {
        "must be finite (not NaN or infinite)"
    }

    /// Generate constraint message for non-empty
    pub fn non_empty() -> &'static str {
        "must not be empty"
    }

    /// Generate constraint message for square matrix
    pub fn square_matrix() -> &'static str {
        "must be a square matrix"
    }

    /// Generate constraint message for positive definite matrix
    pub fn positive_definite() -> &'static str {
        "must be a positive definite matrix"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimension_mismatch() {
        let msg = dimension_mismatch("matrix multiplication", "3x4", "5x3");
        assert_eq!(
            msg,
            "matrix multiplication: dimension mismatch (expected: 3x4, actual: 5x3)"
        );
    }

    #[test]
    fn test_shape_mismatch() {
        let msg = shape_mismatch("element-wise operation", &[3, 4], &[3, 5]);
        assert_eq!(
            msg,
            "element-wise operation: incompatible shapes ([3, 4] vs [3, 5])"
        );
    }

    #[test]
    fn test_invalid_parameter() {
        let msg = invalid_parameter("alpha", constraints::positive(), -0.5);
        assert_eq!(msg, "Parameter 'alpha': must be positive (> 0) (got: -0.5)");
    }

    #[test]
    fn test_with_suggestion() {
        let error = domain_error("input", constraints::positive(), -1.0);
        let msg = with_suggestion(&error, "use absolute value or check input data");
        assert!(msg.contains("input must be must be positive (> 0) (got: -1)"));
        assert!(msg.contains("Suggestion: use absolute value or check input data"));
    }

    #[test]
    fn test_convergence_failed() {
        let msg = convergence_failed("Newton-Raphson", 100, 1e-6);
        assert_eq!(
            msg,
            "Newton-Raphson: failed to converge after 100 iterations (tolerance: 0.000001)"
        );
    }
}
