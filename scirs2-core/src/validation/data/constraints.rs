//! Constraint types and validation logic
//!
//! This module provides various constraint types used for data validation,
//! including range constraints, pattern matching, and statistical constraints.

use std::time::Duration;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Validation constraints for data fields
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Constraint {
    /// Value must be within range (inclusive)
    Range { min: f64, max: f64 },
    /// String must match pattern
    Pattern(String),
    /// Value must be one of the allowed values
    AllowedValues(Vec<String>),
    /// String length constraints
    Length { min: usize, max: usize },
    /// Numeric precision constraints
    Precision { decimal_places: usize },
    /// Uniqueness constraint
    Unique,
    /// Non-null constraint
    NotNull,
    /// Custom validation rule
    Custom(String),
    /// Array element constraints
    ArrayElements(Box<Constraint>),
    /// Array size constraints
    ArraySize { min: usize, max: usize },
    /// Statistical constraints for numeric data
    Statistical(StatisticalConstraints),
    /// Time-based constraints
    Temporal(TimeConstraints),
    /// Matrix/array shape constraints
    Shape(ShapeConstraints),
}

/// Statistical constraints for numeric data
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct StatisticalConstraints {
    /// Minimum allowed mean value
    pub min_mean: Option<f64>,
    /// Maximum allowed mean value
    pub max_mean: Option<f64>,
    /// Minimum allowed standard deviation
    pub min_std: Option<f64>,
    /// Maximum allowed standard deviation
    pub max_std: Option<f64>,
    /// Expected statistical distribution
    pub expected_distribution: Option<String>,
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

/// Time series constraints
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TimeConstraints {
    /// Minimum time interval between samples
    pub min_interval: Option<Duration>,
    /// Maximum time interval between samples
    pub max_interval: Option<Duration>,
    /// Whether timestamps must be monotonic
    pub require_monotonic: bool,
    /// Whether to allow duplicate timestamps
    pub allow_duplicates: bool,
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

/// Element validation function type
pub type ElementValidatorFn<T> = Box<dyn Fn(&T) -> bool + Send + Sync>;

/// Array validation constraints
pub struct ArrayValidationConstraints {
    /// Expected array shape
    pub expected_shape: Option<Vec<usize>>,
    /// Field name for error reporting
    pub field_name: Option<String>,
    /// Check for NaN and infinity values
    pub check_numeric_quality: bool,
    /// Statistical constraints
    pub statistical_constraints: Option<StatisticalConstraints>,
    /// Check performance characteristics
    pub check_performance: bool,
    /// Element-wise validation function
    pub element_validator: Option<ElementValidatorFn<f64>>,
}

impl ArrayValidationConstraints {
    /// Create new array validation constraints
    pub fn new() -> Self {
        Self {
            expected_shape: None,
            field_name: None,
            check_numeric_quality: false,
            statistical_constraints: None,
            check_performance: false,
            element_validator: None,
        }
    }

    /// Set expected shape
    pub fn with_shape(mut self, shape: Vec<usize>) -> Self {
        self.expected_shape = Some(shape);
        self
    }

    /// Set field name
    pub fn with_field_name(mut self, name: &str) -> Self {
        self.field_name = Some(name.to_string());
        self
    }

    /// Enable numeric quality checks
    pub fn check_numeric_quality(mut self) -> Self {
        self.check_numeric_quality = true;
        self
    }

    /// Set statistical constraints
    pub fn with_statistical_constraints(mut self, constraints: StatisticalConstraints) -> Self {
        self.statistical_constraints = Some(constraints);
        self
    }

    /// Enable performance checks
    pub fn check_performance(mut self) -> Self {
        self.check_performance = true;
        self
    }
}

impl Default for ArrayValidationConstraints {
    fn default() -> Self {
        Self::new()
    }
}

impl StatisticalConstraints {
    /// Create new statistical constraints
    pub fn new() -> Self {
        Self {
            min_mean: None,
            max_mean: None,
            min_std: None,
            max_std: None,
            expected_distribution: None,
        }
    }

    /// Set mean range
    pub fn with_mean_range(mut self, min: f64, max: f64) -> Self {
        self.min_mean = Some(min);
        self.max_mean = Some(max);
        self
    }

    /// Set standard deviation range
    pub fn with_std_range(mut self, min: f64, max: f64) -> Self {
        self.min_std = Some(min);
        self.max_std = Some(max);
        self
    }

    /// Set expected distribution
    pub fn with_distribution(mut self, distribution: &str) -> Self {
        self.expected_distribution = Some(distribution.to_string());
        self
    }
}

impl Default for StatisticalConstraints {
    fn default() -> Self {
        Self::new()
    }
}

impl ShapeConstraints {
    /// Create new shape constraints
    pub fn new() -> Self {
        Self {
            dimensions: Vec::new(),
            min_elements: None,
            max_elements: None,
            require_square: false,
            allow_broadcasting: false,
        }
    }

    /// Set exact dimensions
    pub fn with_dimensions(mut self, dimensions: Vec<Option<usize>>) -> Self {
        self.dimensions = dimensions;
        self
    }

    /// Set element count range
    pub fn with_element_range(mut self, min: usize, max: usize) -> Self {
        self.min_elements = Some(min);
        self.max_elements = Some(max);
        self
    }

    /// Require square matrix
    pub fn require_square(mut self) -> Self {
        self.require_square = true;
        self
    }

    /// Allow broadcasting
    pub fn allow_broadcasting(mut self) -> Self {
        self.allow_broadcasting = true;
        self
    }
}

impl Default for ShapeConstraints {
    fn default() -> Self {
        Self::new()
    }
}

impl TimeConstraints {
    /// Create new time constraints
    pub fn new() -> Self {
        Self {
            min_interval: None,
            max_interval: None,
            require_monotonic: false,
            allow_duplicates: true,
        }
    }

    /// Set interval range
    pub fn with_interval_range(mut self, min: Duration, max: Duration) -> Self {
        self.min_interval = Some(min);
        self.max_interval = Some(max);
        self
    }

    /// Require monotonic timestamps
    pub fn require_monotonic(mut self) -> Self {
        self.require_monotonic = true;
        self
    }

    /// Disallow duplicate timestamps
    pub fn disallow_duplicates(mut self) -> Self {
        self.allow_duplicates = false;
        self
    }
}

impl Default for TimeConstraints {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_range_constraint() {
        let constraint = Constraint::Range {
            min: 0.0,
            max: 100.0,
        };
        match constraint {
            Constraint::Range { min, max } => {
                assert_eq!(min, 0.0);
                assert_eq!(max, 100.0);
            }
            _ => panic!("Expected Range constraint"),
        }
    }

    #[test]
    fn test_statistical_constraints() {
        let constraints = StatisticalConstraints::new()
            .with_mean_range(0.0, 10.0)
            .with_std_range(1.0, 5.0)
            .with_distribution("normal");

        assert_eq!(constraints.min_mean, Some(0.0));
        assert_eq!(constraints.max_mean, Some(10.0));
        assert_eq!(constraints.min_std, Some(1.0));
        assert_eq!(constraints.max_std, Some(5.0));
        assert_eq!(
            constraints.expected_distribution,
            Some("normal".to_string())
        );
    }

    #[test]
    fn test_shape_constraints() {
        let constraints = ShapeConstraints::new()
            .with_dimensions(vec![Some(10), Some(20)])
            .with_element_range(100, 500)
            .require_square();

        assert_eq!(constraints.dimensions, vec![Some(10), Some(20)]);
        assert_eq!(constraints.min_elements, Some(100));
        assert_eq!(constraints.max_elements, Some(500));
        assert!(constraints.require_square);
    }

    #[test]
    fn test_array_validation_constraints() {
        let constraints = ArrayValidationConstraints::new()
            .with_shape(vec![10, 20])
            .with_field_name("test_array")
            .check_numeric_quality()
            .check_performance();

        assert_eq!(constraints.expected_shape, Some(vec![10, 20]));
        assert_eq!(constraints.field_name, Some("test_array".to_string()));
        assert!(constraints.check_numeric_quality);
        assert!(constraints.check_performance);
    }

    #[test]
    fn test_sparse_format() {
        let format = SparseFormat::CSR;
        assert_eq!(format, SparseFormat::CSR);
    }
}
