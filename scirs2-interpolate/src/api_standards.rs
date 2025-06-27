//! API standardization guidelines and examples
//!
//! This module demonstrates the standardized API patterns that should be used
//! throughout the interpolation library for consistency.

use crate::traits::*;
use crate::{InterpolateError, InterpolateResult};
use ndarray::{ArrayView1, ArrayView2};

/// Standard factory function pattern
///
/// All factory functions should follow this pattern:
/// 1. Name: `make_<interpolator_name>`
/// 2. Parameters: points, values, config (optional)
/// 3. Return: InterpolateResult<Interpolator>
/// 
/// # Example Implementation
/// ```ignore
/// pub fn make_example_interpolator<T: InterpolationFloat>(
///     points: &ArrayView2<T>,
///     values: &ArrayView1<T>,
///     config: Option<ExampleConfig>,
/// ) -> InterpolateResult<ExampleInterpolator<T>> {
///     // Validate input data
///     validation::validate_data_consistency(points, values)?;
///     
///     // Use default config if not provided
///     let config = config.unwrap_or_default();
///     config.validate()?;
///     
///     // Build and return interpolator
///     ExampleInterpolator::new(points, values, config)
/// }
/// ```
pub mod factory_pattern {
    use super::*;
    
    /// Example configuration structure
    #[derive(Debug, Clone)]
    pub struct StandardConfig<T: InterpolationFloat> {
        /// Smoothing parameter
        pub smoothing: Option<T>,
        
        /// Regularization parameter
        pub regularization: Option<T>,
        
        /// Maximum iterations for iterative methods
        pub max_iterations: usize,
        
        /// Convergence tolerance
        pub tolerance: T,
    }
    
    impl<T: InterpolationFloat> Default for StandardConfig<T> {
        fn default() -> Self {
            Self {
                smoothing: None,
                regularization: None,
                max_iterations: 100,
                tolerance: T::default_tolerance(),
            }
        }
    }
    
    impl<T: InterpolationFloat> InterpolationConfig for StandardConfig<T> {
        fn validate(&self) -> InterpolateResult<()> {
            if self.max_iterations == 0 {
                return Err(InterpolateError::InvalidInput(
                    "max_iterations must be greater than 0".to_string()
                ));
            }
            
            if let Some(s) = self.smoothing {
                if s <= T::zero() {
                    return Err(InterpolateError::InvalidInput(
                        "smoothing parameter must be positive".to_string()
                    ));
                }
            }
            
            Ok(())
        }
        
        fn default() -> Self {
            Self::default()
        }
    }
}

/// Standard builder pattern
///
/// For more complex interpolators, use a builder pattern that follows
/// these conventions:
/// 
/// # Example
/// ```ignore
/// let interpolator = ExampleInterpolatorBuilder::new()
///     .with_smoothing(0.1)
///     .with_regularization(0.01)
///     .with_max_iterations(200)
///     .build(points, values)?;
/// ```
pub mod builder_pattern {
    use super::*;
    
    /// Example builder structure
    #[derive(Debug, Clone)]
    pub struct StandardInterpolatorBuilder<T: InterpolationFloat> {
        config: factory_pattern::StandardConfig<T>,
    }
    
    impl<T: InterpolationFloat> StandardInterpolatorBuilder<T> {
        /// Create a new builder with default configuration
        pub fn new() -> Self {
            Self {
                config: Default::default(),
            }
        }
        
        /// Set smoothing parameter
        pub fn with_smoothing(mut self, smoothing: T) -> Self {
            self.config.smoothing = Some(smoothing);
            self
        }
        
        /// Set regularization parameter  
        pub fn with_regularization(mut self, regularization: T) -> Self {
            self.config.regularization = Some(regularization);
            self
        }
        
        /// Set maximum iterations
        pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
            self.config.max_iterations = max_iterations;
            self
        }
        
        /// Set convergence tolerance
        pub fn with_tolerance(mut self, tolerance: T) -> Self {
            self.config.tolerance = tolerance;
            self
        }
        
        /// Build the interpolator
        pub fn build<I>(
            self,
            points: &ArrayView2<T>,
            values: &ArrayView1<T>,
        ) -> InterpolateResult<I>
        where
            I: From<(ArrayView2<T>, ArrayView1<T>, factory_pattern::StandardConfig<T>)>,
        {
            validation::validate_data_consistency(points, values)?;
            self.config.validate()?;
            
            Ok(I::from((points.view(), values.view(), self.config)))
        }
    }
    
    impl<T: InterpolationFloat> Default for StandardInterpolatorBuilder<T> {
        fn default() -> Self {
            Self::new()
        }
    }
}

/// Standard evaluation interface
///
/// All interpolators should implement consistent evaluation methods
pub mod evaluation_pattern {
    use super::*;
    
    /// Standard batch evaluation with options
    pub fn evaluate_batch<T, I>(
        interpolator: &I,
        query_points: &ArrayView2<T>,
        options: Option<EvaluationOptions>,
    ) -> InterpolateResult<BatchEvaluationResult<T>>
    where
        T: InterpolationFloat,
        I: Interpolator<T>,
    {
        let options = options.unwrap_or_default();
        
        // Validate query dimension
        // validation::validate_query_dimension(interpolator.data_dim(), query_points)?;
        
        // Perform evaluation
        let values = interpolator.evaluate(query_points)?;
        
        Ok(BatchEvaluationResult {
            values,
            uncertainties: None,
            out_of_bounds: Vec::new(),
        })
    }
}

/// Standard error messages
///
/// Consistent error messages across the library
pub mod error_messages {
    /// Generate standard dimension mismatch error
    pub fn dimension_mismatch(expected: usize, actual: usize, context: &str) -> String {
        format!(
            "Dimension mismatch in {}: expected {}, got {}",
            context, expected, actual
        )
    }
    
    /// Generate standard empty data error
    pub fn empty_data(context: &str) -> String {
        format!("Empty input data provided to {}", context)
    }
    
    /// Generate standard invalid parameter error
    pub fn invalid_parameter(param: &str, reason: &str) -> String {
        format!("Invalid parameter '{}': {}", param, reason)
    }
    
    /// Generate standard convergence failure error
    pub fn convergence_failure(method: &str, iterations: usize) -> String {
        format!(
            "{} failed to converge after {} iterations",
            method, iterations
        )
    }
}

/// Migration examples
///
/// Examples of how to migrate existing APIs to the new standard
pub mod migration_examples {
    use super::*;
    
    // Old API:
    // pub fn make_rbf_interpolator<F>(
    //     x: &ArrayView2<F>,
    //     y: &ArrayView1<F>,
    //     kernel: RBFKernel,
    //     epsilon: F,
    // ) -> InterpolateResult<RBFInterpolator<F>>
    
    // New standardized API:
    pub struct RBFConfig<T: InterpolationFloat> {
        pub kernel: RBFKernel,
        pub epsilon: T,
    }
    
    #[derive(Debug)]
    pub enum RBFKernel {
        Gaussian,
        Multiquadric,
        InverseMultiquadric,
        ThinPlate,
    }
    
    impl<T: InterpolationFloat> Default for RBFConfig<T> {
        fn default() -> Self {
            Self {
                kernel: RBFKernel::Gaussian,
                epsilon: T::from_f64(1.0).unwrap(),
            }
        }
    }
    
    impl<T: InterpolationFloat> InterpolationConfig for RBFConfig<T> {
        fn validate(&self) -> InterpolateResult<()> {
            if self.epsilon <= T::zero() {
                return Err(InterpolateError::InvalidInput(
                    "epsilon must be positive".to_string()
                ));
            }
            Ok(())
        }
        
        fn default() -> Self {
            Self::default()
        }
    }
    
    /// Standardized RBF factory function
    pub fn make_rbf_interpolator<T: InterpolationFloat, I>(
        points: &ArrayView2<T>,
        values: &ArrayView1<T>,
        config: Option<RBFConfig<T>>,
    ) -> InterpolateResult<I>
    where
        I: From<(ArrayView2<T>, ArrayView1<T>, RBFConfig<T>)>,
    {
        validation::validate_data_consistency(points, values)?;
        
        let config = config.unwrap_or_default();
        config.validate()?;
        
        Ok(I::from((points.view(), values.view(), config)))
    }
}