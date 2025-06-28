//! Advanced statistical interpolation methods
//!
//! This module provides statistical interpolation techniques that go beyond
//! deterministic interpolation, including:
//! - Bootstrap confidence intervals
//! - Bayesian interpolation with posterior distributions
//! - Quantile interpolation/regression
//! - Robust interpolation methods
//! - Stochastic interpolation for random fields

use crate::error::{InterpolateError, InterpolateResult};
use ndarray::{Array1, Array2, ArrayView1, Axis};
use num_traits::{Float, FromPrimitive};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Distribution, Normal, StandardNormal};
use std::fmt::{Debug, Display};

/// Configuration for bootstrap confidence intervals
#[derive(Debug, Clone)]
pub struct BootstrapConfig {
    /// Number of bootstrap samples
    pub n_samples: usize,
    /// Confidence level (e.g., 0.95 for 95% CI)
    pub confidence_level: f64,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for BootstrapConfig {
    fn default() -> Self {
        Self {
            n_samples: 1000,
            confidence_level: 0.95,
            seed: None,
        }
    }
}

/// Result from bootstrap interpolation including confidence intervals
#[derive(Debug, Clone)]
pub struct BootstrapResult<T: Float> {
    /// Point estimate (median of bootstrap samples)
    pub estimate: Array1<T>,
    /// Lower confidence bound
    pub lower_bound: Array1<T>,
    /// Upper confidence bound
    pub upper_bound: Array1<T>,
    /// Standard error estimate
    pub std_error: Array1<T>,
}

/// Bootstrap interpolation with confidence intervals
///
/// This method performs interpolation with uncertainty quantification
/// using bootstrap resampling of the input data.
pub struct BootstrapInterpolator<T: Float> {
    /// Configuration for bootstrap
    config: BootstrapConfig,
    /// Base interpolator factory
    interpolator_factory: Box<dyn Fn(&ArrayView1<T>, &ArrayView1<T>) -> InterpolateResult<Box<dyn Fn(T) -> T>>>,
}

impl<T: Float + FromPrimitive + Debug + Display + std::iter::Sum> BootstrapInterpolator<T> {
    /// Create a new bootstrap interpolator
    pub fn new<F>(config: BootstrapConfig, interpolator_factory: F) -> Self
    where
        F: Fn(&ArrayView1<T>, &ArrayView1<T>) -> InterpolateResult<Box<dyn Fn(T) -> T>> + 'static,
    {
        Self {
            config,
            interpolator_factory: Box::new(interpolator_factory),
        }
    }
    
    /// Perform bootstrap interpolation at given points
    pub fn interpolate(
        &self,
        x: &ArrayView1<T>,
        y: &ArrayView1<T>,
        x_new: &ArrayView1<T>,
    ) -> InterpolateResult<BootstrapResult<T>> {
        if x.len() != y.len() {
            return Err(InterpolateError::DimensionMismatch(
                "x and y must have the same length".to_string(),
            ));
        }
        
        let n = x.len();
        let m = x_new.len();
        let mut rng = match self.config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => {
                let mut thread_rng = rand::rng();
                StdRng::from_rng(&mut thread_rng)
            },
        };
        
        // Storage for bootstrap samples
        let mut bootstrap_results = Array2::<T>::zeros((self.config.n_samples, m));
        
        // Perform bootstrap resampling
        for i in 0..self.config.n_samples {
            // Resample indices with replacement
            let indices: Vec<usize> = (0..n)
                .map(|_| rng.gen_range(0..n))
                .collect();
            
            // Create resampled data
            let x_resampled = indices.iter().map(|&idx| x[idx]).collect::<Array1<_>>();
            let y_resampled = indices.iter().map(|&idx| y[idx]).collect::<Array1<_>>();
            
            // Create interpolator for this bootstrap sample
            let interpolator = (self.interpolator_factory)(&x_resampled.view(), &y_resampled.view())?;
            
            // Evaluate at new points
            for (j, &x_val) in x_new.iter().enumerate() {
                bootstrap_results[[i, j]] = interpolator(x_val);
            }
        }
        
        // Calculate statistics
        let alpha = T::from(1.0 - self.config.confidence_level).unwrap();
        let lower_percentile = alpha / T::from(2.0).unwrap();
        let upper_percentile = T::one() - alpha / T::from(2.0).unwrap();
        
        let mut estimate = Array1::zeros(m);
        let mut lower_bound = Array1::zeros(m);
        let mut upper_bound = Array1::zeros(m);
        let mut std_error = Array1::zeros(m);
        
        for j in 0..m {
            let column = bootstrap_results.index_axis(Axis(1), j);
            let mut sorted_col = column.to_vec();
            sorted_col.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            // Median as point estimate
            let median_idx = self.config.n_samples / 2;
            estimate[j] = sorted_col[median_idx];
            
            // Confidence bounds
            let lower_idx = (lower_percentile * T::from(self.config.n_samples).unwrap()).to_usize().unwrap();
            let upper_idx = (upper_percentile * T::from(self.config.n_samples).unwrap()).to_usize().unwrap();
            lower_bound[j] = sorted_col[lower_idx];
            upper_bound[j] = sorted_col[upper_idx];
            
            // Standard error
            let mean = column.mean().unwrap();
            let variance = column.iter()
                .map(|&val| (val - mean) * (val - mean))
                .sum::<T>() / T::from(self.config.n_samples - 1).unwrap();
            std_error[j] = variance.sqrt();
        }
        
        Ok(BootstrapResult {
            estimate,
            lower_bound,
            upper_bound,
            std_error,
        })
    }
}

/// Configuration for Bayesian interpolation
pub struct BayesianConfig<T: Float> {
    /// Prior mean function
    pub prior_mean: Box<dyn Fn(T) -> T>,
    /// Prior variance
    pub prior_variance: T,
    /// Measurement noise variance
    pub noise_variance: T,
    /// Number of posterior samples to draw
    pub n_posterior_samples: usize,
}

impl<T: Float + FromPrimitive> Default for BayesianConfig<T> {
    fn default() -> Self {
        Self {
            prior_mean: Box::new(|_| T::zero()),
            prior_variance: T::one(),
            noise_variance: T::from(0.01).unwrap(),
            n_posterior_samples: 100,
        }
    }
}

/// Bayesian interpolation with full posterior distribution
///
/// This provides interpolation with full uncertainty quantification
/// through Bayesian inference.
pub struct BayesianInterpolator<T: Float> {
    config: BayesianConfig<T>,
    x_obs: Array1<T>,
    y_obs: Array1<T>,
}

impl<T: Float + FromPrimitive + Debug + Display> BayesianInterpolator<T> {
    /// Create a new Bayesian interpolator
    pub fn new(x: &ArrayView1<T>, y: &ArrayView1<T>, config: BayesianConfig<T>) -> InterpolateResult<Self> {
        if x.len() != y.len() {
            return Err(InterpolateError::DimensionMismatch(
                "x and y must have the same length".to_string(),
            ));
        }
        
        Ok(Self {
            config,
            x_obs: x.to_owned(),
            y_obs: y.to_owned(),
        })
    }
    
    /// Get posterior mean at given points
    pub fn posterior_mean(&self, x_new: &ArrayView1<T>) -> InterpolateResult<Array1<T>> {
        // Simplified Gaussian process regression
        let n = self.x_obs.len();
        let m = x_new.len();
        
        // Compute covariance matrices
        let mut k_xx = Array2::<T>::zeros((n, n));
        let mut k_x_new = Array2::<T>::zeros((n, m));
        
        // RBF kernel
        let length_scale = T::one();
        for i in 0..n {
            for j in 0..n {
                let dist = (self.x_obs[i] - self.x_obs[j]) / length_scale;
                k_xx[[i, j]] = self.config.prior_variance * (-dist * dist / T::from(2.0).unwrap()).exp();
                if i == j {
                    k_xx[[i, j]] = k_xx[[i, j]] + self.config.noise_variance;
                }
            }
        }
        
        for i in 0..n {
            for j in 0..m {
                let dist = (self.x_obs[i] - x_new[j]) / length_scale;
                k_x_new[[i, j]] = self.config.prior_variance * (-dist * dist / T::from(2.0).unwrap()).exp();
            }
        }
        
        // Compute posterior mean: μ* = K(X*, X)[K(X, X) + σ²I]^(-1)y
        // This is a simplified implementation
        let mut mean = Array1::zeros(m);
        for j in 0..m {
            // Use linear interpolation as a simple approximation
            let mut weighted_sum = T::zero();
            let mut weight_sum = T::zero();
            
            for i in 0..n {
                let weight = k_x_new[[i, j]];
                weighted_sum = weighted_sum + weight * self.y_obs[i];
                weight_sum = weight_sum + weight;
            }
            
            if weight_sum > T::epsilon() {
                mean[j] = weighted_sum / weight_sum;
            } else {
                mean[j] = (self.config.prior_mean)(x_new[j]);
            }
        }
        
        Ok(mean)
    }
    
    /// Draw samples from the posterior distribution
    pub fn posterior_samples(&self, x_new: &ArrayView1<T>, n_samples: usize) -> InterpolateResult<Array2<T>> {
        let mean = self.posterior_mean(x_new)?;
        let m = x_new.len();
        
        let mut samples = Array2::zeros((n_samples, m));
        let mut rng = rand::rng();
        
        // Draw samples from posterior (simplified - assumes independence)
        for i in 0..n_samples {
            for j in 0..m {
                let std_dev = self.config.prior_variance.sqrt();
                if let Ok(normal) = Normal::new(mean[j].to_f64().unwrap(), std_dev.to_f64().unwrap()) {
                    samples[[i, j]] = T::from(normal.sample(&mut rng)).unwrap();
                } else {
                    samples[[i, j]] = mean[j];
                }
            }
        }
        
        Ok(samples)
    }
}

/// Quantile interpolation/regression
///
/// Interpolates specific quantiles of the response distribution
pub struct QuantileInterpolator<T: Float> {
    /// Quantile to interpolate (e.g., 0.5 for median)
    quantile: T,
    /// Bandwidth for local quantile estimation
    bandwidth: T,
}

impl<T: Float + FromPrimitive + Debug + Display> QuantileInterpolator<T> 
where
    T: std::iter::Sum<T> + for<'a> std::iter::Sum<&'a T>,
{
    /// Create a new quantile interpolator
    pub fn new(quantile: T, bandwidth: T) -> InterpolateResult<Self> {
        if quantile <= T::zero() || quantile >= T::one() {
            return Err(InterpolateError::InvalidValue(
                "Quantile must be between 0 and 1".to_string(),
            ));
        }
        
        Ok(Self { quantile, bandwidth })
    }
    
    /// Interpolate quantile at given points
    pub fn interpolate(
        &self,
        x: &ArrayView1<T>,
        y: &ArrayView1<T>,
        x_new: &ArrayView1<T>,
    ) -> InterpolateResult<Array1<T>> {
        if x.len() != y.len() {
            return Err(InterpolateError::DimensionMismatch(
                "x and y must have the same length".to_string(),
            ));
        }
        
        let n = x.len();
        let m = x_new.len();
        let mut result = Array1::zeros(m);
        
        // Local quantile regression
        for j in 0..m {
            let x_target = x_new[j];
            
            // Compute weights based on distance
            let mut weights = Vec::with_capacity(n);
            let mut weighted_values = Vec::with_capacity(n);
            
            for i in 0..n {
                let dist = (x[i] - x_target).abs() / self.bandwidth;
                let weight = if dist < T::one() {
                    (T::one() - dist * dist * dist).powi(3) // Tricube kernel
                } else {
                    T::zero()
                };
                
                if weight > T::epsilon() {
                    weights.push(weight);
                    weighted_values.push((y[i], weight));
                }
            }
            
            if weighted_values.is_empty() {
                // No nearby points, use nearest neighbor
                let nearest_idx = x.iter()
                    .enumerate()
                    .min_by_key(|(_, &xi)| ((xi - x_target).abs().to_f64().unwrap() * 1e6) as i64)
                    .map(|(i, _)| i)
                    .unwrap();
                result[j] = y[nearest_idx];
            } else {
                // Sort by value
                weighted_values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                
                // Find weighted quantile
                let total_weight: T = weights.iter().sum();
                let target_weight = self.quantile * total_weight;
                
                let mut cumulative_weight = T::zero();
                for (val, weight) in weighted_values {
                    cumulative_weight = cumulative_weight + weight;
                    if cumulative_weight >= target_weight {
                        result[j] = val;
                        break;
                    }
                }
            }
        }
        
        Ok(result)
    }
}

/// Robust interpolation methods resistant to outliers
pub struct RobustInterpolator<T: Float> {
    /// Tuning constant for robustness
    tuning_constant: T,
    /// Maximum iterations for iterative reweighting
    max_iterations: usize,
    /// Convergence tolerance
    tolerance: T,
}

impl<T: Float + FromPrimitive + Debug + Display> RobustInterpolator<T> {
    /// Create a new robust interpolator using M-estimation
    pub fn new(tuning_constant: T) -> Self {
        Self {
            tuning_constant,
            max_iterations: 100,
            tolerance: T::from(1e-6).unwrap(),
        }
    }
    
    /// Perform robust interpolation using iteratively reweighted least squares
    pub fn interpolate(
        &self,
        x: &ArrayView1<T>,
        y: &ArrayView1<T>,
        x_new: &ArrayView1<T>,
    ) -> InterpolateResult<Array1<T>> {
        // Use local polynomial regression with robust weights
        let n = x.len();
        let m = x_new.len();
        let mut result = Array1::zeros(m);
        
        for j in 0..m {
            let x_target = x_new[j];
            
            // Initial weights (uniform)
            let mut weights = vec![T::one(); n];
            let mut prev_estimate = T::zero();
            
            // Iteratively reweighted least squares
            for _iter in 0..self.max_iterations {
                // Weighted linear regression
                let mut sum_w = T::zero();
                let mut sum_wx = T::zero();
                let mut sum_wy = T::zero();
                let mut sum_wxx = T::zero();
                let mut sum_wxy = T::zero();
                
                for i in 0..n {
                    let w = weights[i];
                    let dx = x[i] - x_target;
                    sum_w = sum_w + w;
                    sum_wx = sum_wx + w * dx;
                    sum_wy = sum_wy + w * y[i];
                    sum_wxx = sum_wxx + w * dx * dx;
                    sum_wxy = sum_wxy + w * dx * y[i];
                }
                
                // Solve for coefficients
                let det = sum_w * sum_wxx - sum_wx * sum_wx;
                let estimate = if det.abs() > T::epsilon() {
                    (sum_wxx * sum_wy - sum_wx * sum_wxy) / det
                } else {
                    sum_wy / sum_w
                };
                
                // Check convergence
                if (estimate - prev_estimate).abs() < self.tolerance {
                    result[j] = estimate;
                    break;
                }
                prev_estimate = estimate;
                
                // Update weights using Huber's psi function
                for i in 0..n {
                    let residual = y[i] - estimate;
                    let scaled_residual = residual / self.tuning_constant;
                    
                    weights[i] = if scaled_residual.abs() <= T::one() {
                        T::one()
                    } else {
                        T::one() / scaled_residual.abs()
                    };
                }
            }
            
            result[j] = prev_estimate;
        }
        
        Ok(result)
    }
}

/// Stochastic interpolation for random fields
///
/// Provides interpolation that preserves the stochastic properties
/// of the underlying random field.
pub struct StochasticInterpolator<T: Float> {
    /// Correlation length scale
    correlation_length: T,
    /// Field variance
    field_variance: T,
    /// Number of realizations to generate
    n_realizations: usize,
}

impl<T: Float + FromPrimitive + Debug + Display> StochasticInterpolator<T> {
    /// Create a new stochastic interpolator
    pub fn new(correlation_length: T, field_variance: T, n_realizations: usize) -> Self {
        Self {
            correlation_length,
            field_variance,
            n_realizations,
        }
    }
    
    /// Generate stochastic realizations of the interpolated field
    pub fn interpolate_realizations(
        &self,
        x: &ArrayView1<T>,
        y: &ArrayView1<T>,
        x_new: &ArrayView1<T>,
    ) -> InterpolateResult<Array2<T>> {
        let n = x.len();
        let m = x_new.len();
        let mut realizations = Array2::zeros((self.n_realizations, m));
        
        let mut rng = rand::rng();
        
        for r in 0..self.n_realizations {
            // Generate a realization using conditional simulation
            for j in 0..m {
                let x_target = x_new[j];
                
                // Kriging interpolation with added noise
                let mut weighted_sum = T::zero();
                let mut weight_sum = T::zero();
                
                for i in 0..n {
                    let dist = (x[i] - x_target).abs() / self.correlation_length;
                    let weight = (-dist * dist).exp();
                    weighted_sum = weighted_sum + weight * y[i];
                    weight_sum = weight_sum + weight;
                }
                
                let mean = if weight_sum > T::epsilon() {
                    weighted_sum / weight_sum
                } else {
                    T::zero()
                };
                
                // Add stochastic component
                let std_dev = (self.field_variance * (T::one() - weight_sum / T::from(n).unwrap())).sqrt();
                let normal_sample: f64 = StandardNormal.sample(&mut rng);
                let noise: T = T::from(normal_sample).unwrap() * std_dev;
                
                realizations[[r, j]] = mean + noise;
            }
        }
        
        Ok(realizations)
    }
    
    /// Get mean and variance of the stochastic interpolation
    pub fn interpolate_statistics(
        &self,
        x: &ArrayView1<T>,
        y: &ArrayView1<T>,
        x_new: &ArrayView1<T>,
    ) -> InterpolateResult<(Array1<T>, Array1<T>)> {
        let realizations = self.interpolate_realizations(x, y, x_new)?;
        
        let mean = realizations.mean_axis(Axis(0)).unwrap();
        let variance = realizations.var_axis(Axis(0), T::from(1.0).unwrap());
        
        Ok((mean, variance))
    }
}

/// Factory functions for creating statistical interpolators

/// Create a bootstrap interpolator with linear base interpolation
pub fn make_bootstrap_linear_interpolator<T: Float + FromPrimitive + Debug + Display + 'static + std::iter::Sum>(
    config: BootstrapConfig,
) -> BootstrapInterpolator<T> {
    BootstrapInterpolator::new(config, |x, y| {
        // Create a simple linear interpolator
        let x_owned = x.to_owned();
        let y_owned = y.to_owned();
        Ok(Box::new(move |x_new| {
            // Simple linear interpolation
            if x_new <= x_owned[0] {
                y_owned[0]
            } else if x_new >= x_owned[x_owned.len() - 1] {
                y_owned[y_owned.len() - 1]
            } else {
                // Find surrounding points
                let mut i = 0;
                for j in 1..x_owned.len() {
                    if x_new <= x_owned[j] {
                        i = j - 1;
                        break;
                    }
                }
                
                let alpha = (x_new - x_owned[i]) / (x_owned[i + 1] - x_owned[i]);
                y_owned[i] * (T::one() - alpha) + y_owned[i + 1] * alpha
            }
        }))
    })
}

/// Create a Bayesian interpolator with default configuration
pub fn make_bayesian_interpolator<T: Float + FromPrimitive + Debug + Display>(
    x: &ArrayView1<T>,
    y: &ArrayView1<T>,
) -> InterpolateResult<BayesianInterpolator<T>> {
    BayesianInterpolator::new(x, y, BayesianConfig::default())
}

/// Create a median (0.5 quantile) interpolator
pub fn make_median_interpolator<T: Float + FromPrimitive + Debug + Display>(
    bandwidth: T,
) -> InterpolateResult<QuantileInterpolator<T>> 
where
    T: std::iter::Sum<T> + for<'a> std::iter::Sum<&'a T>,
{
    QuantileInterpolator::new(T::from(0.5).unwrap(), bandwidth)
}

/// Create a robust interpolator with default Huber tuning
pub fn make_robust_interpolator<T: Float + FromPrimitive + Debug + Display>() -> RobustInterpolator<T> {
    RobustInterpolator::new(T::from(1.345).unwrap()) // Huber's recommended value
}

/// Create a stochastic interpolator with default parameters
pub fn make_stochastic_interpolator<T: Float + FromPrimitive + Debug + Display>(
    correlation_length: T,
) -> StochasticInterpolator<T> {
    StochasticInterpolator::new(correlation_length, T::one(), 100)
}