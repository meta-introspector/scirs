//! Common traits for interpolation types
//!
//! This module defines standard trait bounds used throughout the interpolation library
//! to ensure API consistency and reduce repetition.

use ndarray::{ArrayView1, ArrayView2};
use num_traits::{Float, FromPrimitive};
use std::fmt::{Debug, Display, LowerExp};
use std::ops::{AddAssign, SubAssign};

/// Standard floating-point type for interpolation operations
///
/// This trait combines all the common bounds needed for interpolation algorithms,
/// providing a single consistent constraint across the library.
pub trait InterpolationFloat:
    Float + FromPrimitive + Debug + Display + LowerExp + AddAssign + SubAssign + Send + Sync + 'static
{
    /// Default epsilon value for this floating-point type
    fn default_epsilon() -> Self {
        Self::from_f64(1e-9).unwrap_or_else(|| Self::epsilon())
    }

    /// Default tolerance for iterative algorithms
    fn default_tolerance() -> Self {
        Self::from_f64(1e-12).unwrap_or_else(|| Self::epsilon() * Self::from_f64(100.0).unwrap())
    }
}

// Implement for standard floating-point types
impl InterpolationFloat for f32 {}
impl InterpolationFloat for f64 {}

/// Standard input data format for interpolation
///
/// This trait defines the expected format for input data points and values
pub trait InterpolationData<T: InterpolationFloat> {
    /// Get the spatial coordinates of the data points
    fn points(&self) -> ArrayView2<T>;

    /// Get the function values at the data points
    fn values(&self) -> ArrayView1<T>;

    /// Get the number of data points
    fn len(&self) -> usize {
        self.values().len()
    }

    /// Check if the data is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the spatial dimension of the data
    fn dim(&self) -> usize {
        self.points().ncols()
    }
}

/// Standard query interface for interpolators
pub trait Interpolator<T: InterpolationFloat> {
    /// Evaluate the interpolator at given query points
    fn evaluate(&self, query_points: &ArrayView2<T>) -> crate::InterpolateResult<Vec<T>>;

    /// Evaluate the interpolator at a single point
    fn evaluate_single(&self, point: &ArrayView1<T>) -> crate::InterpolateResult<T> {
        let query = point.view().insert_axis(ndarray::Axis(0));
        self.evaluate(&query).map(|v| v[0])
    }

    /// Evaluate derivatives at query points (if supported)
    fn evaluate_derivatives(
        &self,
        query_points: &ArrayView2<T>,
        order: usize,
    ) -> crate::InterpolateResult<Vec<Vec<T>>> {
        let _ = (query_points, order);
        Err(crate::InterpolateError::NotImplemented(
            "Derivative evaluation not implemented for this interpolator".to_string(),
        ))
    }
}

/// Configuration trait for interpolation methods
pub trait InterpolationConfig: Clone + Debug {
    /// Validate the configuration
    fn validate(&self) -> crate::InterpolateResult<()>;

    /// Get default configuration
    fn default() -> Self;
}

/// Builder pattern trait for consistent API
pub trait InterpolatorBuilder<T: InterpolationFloat> {
    /// The interpolator type this builder creates
    type Interpolator: Interpolator<T>;

    /// The configuration type for this builder
    type Config: InterpolationConfig;

    /// Build the interpolator with the given data and configuration
    fn build(
        points: &ArrayView2<T>,
        values: &ArrayView1<T>,
        config: Self::Config,
    ) -> crate::InterpolateResult<Self::Interpolator>;
}

/// Standard evaluation options
#[derive(Debug, Clone)]
pub struct EvaluationOptions {
    /// Number of parallel workers (None for automatic)
    pub workers: Option<usize>,

    /// Whether to use caching for repeated evaluations
    pub use_cache: bool,

    /// Whether to validate input bounds
    pub validate_bounds: bool,

    /// Fill value for out-of-bounds queries
    pub fill_value: Option<f64>,
}

impl Default for EvaluationOptions {
    fn default() -> Self {
        Self {
            workers: None,
            use_cache: false,
            validate_bounds: true,
            fill_value: None,
        }
    }
}

/// Standard result for batch evaluation
pub struct BatchEvaluationResult<T: InterpolationFloat> {
    /// The interpolated values
    pub values: Vec<T>,

    /// Optional uncertainty estimates (for methods that support it)
    pub uncertainties: Option<Vec<T>>,

    /// Indices of out-of-bounds points (if any)
    pub out_of_bounds: Vec<usize>,
}

/// Common validation utilities
pub mod validation {
    use super::*;

    /// Validate that points and values have consistent dimensions
    pub fn validate_data_consistency<T: InterpolationFloat>(
        points: &ArrayView2<T>,
        values: &ArrayView1<T>,
    ) -> crate::InterpolateResult<()> {
        if points.nrows() != values.len() {
            return Err(crate::InterpolateError::invalid_input(format!(
                "Inconsistent data dimensions: {} points but {} values",
                points.nrows(),
                values.len()
            )));
        }

        if points.is_empty() || values.is_empty() {
            return Err(crate::InterpolateError::invalid_input("Empty input data"));
        }

        Ok(())
    }

    /// Validate query points have correct dimension
    pub fn validate_query_dimension<T: InterpolationFloat>(
        data_dim: usize,
        query_points: &ArrayView2<T>,
    ) -> crate::InterpolateResult<()> {
        if query_points.ncols() != data_dim {
            return Err(crate::InterpolateError::invalid_input(format!(
                "Query dimension {} does not match data dimension {}",
                query_points.ncols(),
                data_dim
            )));
        }
        Ok(())
    }
}
