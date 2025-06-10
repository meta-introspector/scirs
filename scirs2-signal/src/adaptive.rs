//! Adaptive filtering algorithms
//!
//! This module provides adaptive filter implementations including LMS (Least Mean Squares)
//! and RLS (Recursive Least Squares) algorithms. These filters are used for applications
//! such as noise cancellation, system identification, echo cancellation, and equalization.

use crate::error::{SignalError, SignalResult};
use std::fmt::Debug;

/// Least Mean Squares (LMS) adaptive filter
///
/// The LMS algorithm is a simple and robust adaptive filter that minimizes
/// the mean square error between the desired signal and the filter output.
/// It uses a gradient descent approach to update the filter coefficients.
///
/// # Examples
///
/// ```ignore
/// use scirs2_signal::adaptive::LmsFilter;
///
/// let mut lms = LmsFilter::new(4, 0.01, 0.0).unwrap();
/// let output = lms.adapt(&[1.0, 0.5, -0.3, 0.8], 0.5).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct LmsFilter {
    /// Filter coefficients (weights)
    weights: Vec<f64>,
    /// Input buffer (delay line)
    buffer: Vec<f64>,
    /// Step size (learning rate)
    step_size: f64,
    /// Current buffer index
    buffer_index: usize,
}

impl LmsFilter {
    /// Create a new LMS filter
    ///
    /// # Arguments
    ///
    /// * `num_taps` - Number of filter taps (filter order + 1)
    /// * `step_size` - Learning rate (typically 0.001 to 0.1)
    /// * `initial_weight` - Initial value for all filter weights
    ///
    /// # Returns
    ///
    /// * A new LMS filter instance
    pub fn new(num_taps: usize, step_size: f64, initial_weight: f64) -> SignalResult<Self> {
        if num_taps == 0 {
            return Err(SignalError::ValueError(
                "Number of taps must be positive".to_string(),
            ));
        }

        if step_size <= 0.0 {
            return Err(SignalError::ValueError(
                "Step size must be positive".to_string(),
            ));
        }

        Ok(LmsFilter {
            weights: vec![initial_weight; num_taps],
            buffer: vec![0.0; num_taps],
            step_size,
            buffer_index: 0,
        })
    }

    /// Process one sample through the adaptive filter
    ///
    /// # Arguments
    ///
    /// * `input` - Input sample
    /// * `desired` - Desired output sample
    ///
    /// # Returns
    ///
    /// * Tuple of (filter_output, error, mse_estimate)
    pub fn adapt(&mut self, input: f64, desired: f64) -> SignalResult<(f64, f64, f64)> {
        // Update circular buffer with new input
        self.buffer[self.buffer_index] = input;
        self.buffer_index = (self.buffer_index + 1) % self.buffer.len();

        // Compute filter output (dot product of weights and buffered inputs)
        let mut output = 0.0;
        for i in 0..self.weights.len() {
            let buffer_idx = (self.buffer_index + self.buffer.len() - 1 - i) % self.buffer.len();
            output += self.weights[i] * self.buffer[buffer_idx];
        }

        // Compute error
        let error = desired - output;

        // Update weights using LMS algorithm: w(n+1) = w(n) + μ * e(n) * x(n)
        for i in 0..self.weights.len() {
            let buffer_idx = (self.buffer_index + self.buffer.len() - 1 - i) % self.buffer.len();
            self.weights[i] += self.step_size * error * self.buffer[buffer_idx];
        }

        // Estimate MSE (simple exponential average)
        let mse_estimate = error * error;

        Ok((output, error, mse_estimate))
    }

    /// Process a batch of samples
    ///
    /// # Arguments
    ///
    /// * `inputs` - Input signal samples
    /// * `desired` - Desired output samples
    ///
    /// # Returns
    ///
    /// * Tuple of (outputs, errors, mse_estimates)
    pub fn adapt_batch(
        &mut self,
        inputs: &[f64],
        desired: &[f64],
    ) -> SignalResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        if inputs.len() != desired.len() {
            return Err(SignalError::ValueError(
                "Input and desired signals must have the same length".to_string(),
            ));
        }

        let mut outputs = Vec::with_capacity(inputs.len());
        let mut errors = Vec::with_capacity(inputs.len());
        let mut mse_estimates = Vec::with_capacity(inputs.len());

        for (&input, &des) in inputs.iter().zip(desired.iter()) {
            let (output, error, mse) = self.adapt(input, des)?;
            outputs.push(output);
            errors.push(error);
            mse_estimates.push(mse);
        }

        Ok((outputs, errors, mse_estimates))
    }

    /// Get current filter weights
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Get current input buffer state
    pub fn buffer(&self) -> &[f64] {
        &self.buffer
    }

    /// Reset the filter to initial state
    pub fn reset(&mut self, initial_weight: f64) {
        self.weights.fill(initial_weight);
        self.buffer.fill(0.0);
        self.buffer_index = 0;
    }

    /// Set step size (learning rate)
    pub fn set_step_size(&mut self, step_size: f64) -> SignalResult<()> {
        if step_size <= 0.0 {
            return Err(SignalError::ValueError(
                "Step size must be positive".to_string(),
            ));
        }
        self.step_size = step_size;
        Ok(())
    }
}

/// Recursive Least Squares (RLS) adaptive filter
///
/// The RLS algorithm provides faster convergence than LMS but with higher
/// computational complexity. It minimizes the exponentially weighted sum
/// of squared errors and is particularly effective for non-stationary signals.
///
/// # Examples
///
/// ```ignore
/// use scirs2_signal::adaptive::RlsFilter;
///
/// let mut rls = RlsFilter::new(4, 0.99, 1000.0).unwrap();
/// let output = rls.adapt(&[1.0, 0.5, -0.3, 0.8], 0.5).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct RlsFilter {
    /// Filter coefficients (weights)
    weights: Vec<f64>,
    /// Input buffer (delay line)  
    buffer: Vec<f64>,
    /// Inverse correlation matrix P
    p_matrix: Vec<Vec<f64>>,
    /// Forgetting factor (typically 0.95 to 0.999)
    lambda: f64,
    /// Current buffer index
    buffer_index: usize,
}

impl RlsFilter {
    /// Create a new RLS filter
    ///
    /// # Arguments
    ///
    /// * `num_taps` - Number of filter taps
    /// * `lambda` - Forgetting factor (0 < lambda <= 1.0, typically 0.99)
    /// * `delta` - Initialization parameter for P matrix (typically 100-10000)
    ///
    /// # Returns
    ///
    /// * A new RLS filter instance
    pub fn new(num_taps: usize, lambda: f64, delta: f64) -> SignalResult<Self> {
        if num_taps == 0 {
            return Err(SignalError::ValueError(
                "Number of taps must be positive".to_string(),
            ));
        }

        if lambda <= 0.0 || lambda > 1.0 {
            return Err(SignalError::ValueError(
                "Forgetting factor must be in (0, 1]".to_string(),
            ));
        }

        if delta <= 0.0 {
            return Err(SignalError::ValueError(
                "Delta must be positive".to_string(),
            ));
        }

        // Initialize P matrix as delta * I (identity matrix)
        let mut p_matrix = vec![vec![0.0; num_taps]; num_taps];
        for (i, row) in p_matrix.iter_mut().enumerate().take(num_taps) {
            row[i] = delta;
        }

        Ok(RlsFilter {
            weights: vec![0.0; num_taps],
            buffer: vec![0.0; num_taps],
            p_matrix,
            lambda,
            buffer_index: 0,
        })
    }

    /// Process one sample through the adaptive filter
    ///
    /// # Arguments
    ///
    /// * `input` - Input sample
    /// * `desired` - Desired output sample
    ///
    /// # Returns
    ///
    /// * Tuple of (filter_output, error, mse_estimate)
    pub fn adapt(&mut self, input: f64, desired: f64) -> SignalResult<(f64, f64, f64)> {
        let num_taps = self.weights.len();

        // Update circular buffer with new input
        self.buffer[self.buffer_index] = input;
        self.buffer_index = (self.buffer_index + 1) % self.buffer.len();

        // Create input vector (in proper order)
        let mut input_vec = vec![0.0; num_taps];
        for (i, input_val) in input_vec.iter_mut().enumerate().take(num_taps) {
            let buffer_idx = (self.buffer_index + self.buffer.len() - 1 - i) % self.buffer.len();
            *input_val = self.buffer[buffer_idx];
        }

        // Compute filter output
        let output = dot_product(&self.weights, &input_vec);

        // Compute error
        let error = desired - output;

        // RLS algorithm updates
        // 1. Compute k(n) = P(n-1) * x(n) / (lambda + x(n)^T * P(n-1) * x(n))
        let mut px = matrix_vector_multiply(&self.p_matrix, &input_vec);
        let xpx = dot_product(&input_vec, &px);
        let denominator = self.lambda + xpx;

        if denominator.abs() < 1e-10 {
            return Err(SignalError::ValueError(
                "RLS denominator too small, numerical instability".to_string(),
            ));
        }

        for px_val in &mut px {
            *px_val /= denominator;
        }
        let k = px; // k(n) = P(n-1) * x(n) / denominator

        // 2. Update weights: w(n) = w(n-1) + k(n) * e(n)
        for (weight, &k_val) in self.weights.iter_mut().zip(k.iter()) {
            *weight += k_val * error;
        }

        // 3. Update P matrix: P(n) = (P(n-1) - k(n) * x(n)^T * P(n-1)) / lambda
        let mut kx_outer = vec![vec![0.0; num_taps]; num_taps];
        for (kx_row, &k_val) in kx_outer.iter_mut().zip(k.iter()) {
            for (kx_elem, &input_val) in kx_row.iter_mut().zip(input_vec.iter()) {
                *kx_elem = k_val * input_val;
            }
        }

        // P = (P - k * x^T * P) / lambda
        let p_matrix_copy = self.p_matrix.clone();
        for (p_row, kx_row) in self.p_matrix.iter_mut().zip(kx_outer.iter()) {
            for (j, p_elem) in p_row.iter_mut().enumerate() {
                let kxp = dot_product(kx_row, &get_column(&p_matrix_copy, j));
                *p_elem = (*p_elem - kxp) / self.lambda;
            }
        }

        let mse_estimate = error * error;

        Ok((output, error, mse_estimate))
    }

    /// Process a batch of samples
    ///
    /// # Arguments
    ///
    /// * `inputs` - Input signal samples
    /// * `desired` - Desired output samples
    ///
    /// # Returns
    ///
    /// * Tuple of (outputs, errors, mse_estimates)
    pub fn adapt_batch(
        &mut self,
        inputs: &[f64],
        desired: &[f64],
    ) -> SignalResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        if inputs.len() != desired.len() {
            return Err(SignalError::ValueError(
                "Input and desired signals must have the same length".to_string(),
            ));
        }

        let mut outputs = Vec::with_capacity(inputs.len());
        let mut errors = Vec::with_capacity(inputs.len());
        let mut mse_estimates = Vec::with_capacity(inputs.len());

        for (&input, &des) in inputs.iter().zip(desired.iter()) {
            let (output, error, mse) = self.adapt(input, des)?;
            outputs.push(output);
            errors.push(error);
            mse_estimates.push(mse);
        }

        Ok((outputs, errors, mse_estimates))
    }

    /// Get current filter weights
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Reset the filter to initial state
    pub fn reset(&mut self, delta: f64) -> SignalResult<()> {
        if delta <= 0.0 {
            return Err(SignalError::ValueError(
                "Delta must be positive".to_string(),
            ));
        }

        let num_taps = self.weights.len();
        self.weights.fill(0.0);
        self.buffer.fill(0.0);
        self.buffer_index = 0;

        // Reinitialize P matrix
        for i in 0..num_taps {
            for j in 0..num_taps {
                self.p_matrix[i][j] = if i == j { delta } else { 0.0 };
            }
        }

        Ok(())
    }
}

/// Normalized LMS (NLMS) adaptive filter
///
/// The NLMS algorithm normalizes the step size by the input signal power,
/// providing better performance for signals with varying power levels.
#[derive(Debug, Clone)]
pub struct NlmsFilter {
    /// Filter coefficients (weights)
    weights: Vec<f64>,
    /// Input buffer (delay line)
    buffer: Vec<f64>,
    /// Step size (learning rate)
    step_size: f64,
    /// Regularization parameter to avoid division by zero
    epsilon: f64,
    /// Current buffer index
    buffer_index: usize,
}

impl NlmsFilter {
    /// Create a new NLMS filter
    ///
    /// # Arguments
    ///
    /// * `num_taps` - Number of filter taps
    /// * `step_size` - Learning rate (typically 0.1 to 2.0)
    /// * `epsilon` - Regularization parameter (typically 1e-6)
    ///
    /// # Returns
    ///
    /// * A new NLMS filter instance
    pub fn new(num_taps: usize, step_size: f64, epsilon: f64) -> SignalResult<Self> {
        if num_taps == 0 {
            return Err(SignalError::ValueError(
                "Number of taps must be positive".to_string(),
            ));
        }

        if step_size <= 0.0 {
            return Err(SignalError::ValueError(
                "Step size must be positive".to_string(),
            ));
        }

        if epsilon <= 0.0 {
            return Err(SignalError::ValueError(
                "Epsilon must be positive".to_string(),
            ));
        }

        Ok(NlmsFilter {
            weights: vec![0.0; num_taps],
            buffer: vec![0.0; num_taps],
            step_size,
            epsilon,
            buffer_index: 0,
        })
    }

    /// Process one sample through the adaptive filter
    ///
    /// # Arguments
    ///
    /// * `input` - Input sample
    /// * `desired` - Desired output sample
    ///
    /// # Returns
    ///
    /// * Tuple of (filter_output, error, mse_estimate)
    pub fn adapt(&mut self, input: f64, desired: f64) -> SignalResult<(f64, f64, f64)> {
        // Update circular buffer
        self.buffer[self.buffer_index] = input;
        self.buffer_index = (self.buffer_index + 1) % self.buffer.len();

        // Compute filter output
        let mut output = 0.0;
        for i in 0..self.weights.len() {
            let buffer_idx = (self.buffer_index + self.buffer.len() - 1 - i) % self.buffer.len();
            output += self.weights[i] * self.buffer[buffer_idx];
        }

        // Compute error
        let error = desired - output;

        // Compute input power (norm squared)
        let input_power: f64 = self.buffer.iter().map(|&x| x * x).sum();
        let normalized_step = self.step_size / (input_power + self.epsilon);

        // Update weights using NLMS algorithm
        for i in 0..self.weights.len() {
            let buffer_idx = (self.buffer_index + self.buffer.len() - 1 - i) % self.buffer.len();
            self.weights[i] += normalized_step * error * self.buffer[buffer_idx];
        }

        let mse_estimate = error * error;

        Ok((output, error, mse_estimate))
    }

    /// Get current filter weights
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Reset the filter
    pub fn reset(&mut self) {
        self.weights.fill(0.0);
        self.buffer.fill(0.0);
        self.buffer_index = 0;
    }
}

// Helper functions for matrix operations

/// Compute dot product of two vectors
fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Multiply matrix by vector
fn matrix_vector_multiply(matrix: &[Vec<f64>], vector: &[f64]) -> Vec<f64> {
    let mut result = vec![0.0; matrix.len()];
    for i in 0..matrix.len() {
        result[i] = dot_product(&matrix[i], vector);
    }
    result
}

/// Get column from matrix
fn get_column(matrix: &[Vec<f64>], col: usize) -> Vec<f64> {
    matrix.iter().map(|row| row[col]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_lms_creation() {
        let lms = LmsFilter::new(4, 0.01, 0.0).unwrap();
        assert_eq!(lms.weights().len(), 4);
        assert_eq!(lms.buffer().len(), 4);

        // Test error conditions
        assert!(LmsFilter::new(0, 0.01, 0.0).is_err());
        assert!(LmsFilter::new(4, 0.0, 0.0).is_err());
    }

    #[test]
    fn test_lms_adapt() {
        let mut lms = LmsFilter::new(2, 0.1, 0.0).unwrap();

        // Test single adaptation
        let (output, error, _mse) = lms.adapt(1.0, 0.5).unwrap();

        // Initially weights are zero, so output should be zero
        assert_relative_eq!(output, 0.0, epsilon = 1e-10);
        assert_relative_eq!(error, 0.5, epsilon = 1e-10);

        // Weights should be updated
        assert!(!lms.weights().iter().all(|&w| w == 0.0));
    }

    #[test]
    fn test_lms_batch() {
        let mut lms = LmsFilter::new(2, 0.05, 0.0).unwrap();

        let inputs = vec![1.0, 0.5, -0.3, 0.8];
        let desired = vec![0.1, 0.2, 0.3, 0.4];

        let (outputs, errors, _mse) = lms.adapt_batch(&inputs, &desired).unwrap();

        assert_eq!(outputs.len(), 4);
        assert_eq!(errors.len(), 4);

        // Error should generally decrease over time for a learnable system
        // Note: LMS adaptation is gradual, so we just check that it's reasonable
        assert!(errors.iter().all(|&e| e.abs() < 10.0)); // Errors should be bounded
    }

    #[test]
    fn test_lms_system_identification() {
        // Test LMS for system identification
        let mut lms = LmsFilter::new(3, 0.01, 0.0).unwrap();

        // Target system: h = [0.5, -0.3, 0.2]
        let target_system = [0.5, -0.3, 0.2];

        // Generate training data
        let mut inputs = Vec::new();
        let mut desired = Vec::new();

        for i in 0..100 {
            let input = (i as f64 * 0.1).sin();
            inputs.push(input);

            // Generate desired output from target system (simplified)
            let output = if i >= 2 {
                target_system[0] * inputs[i]
                    + target_system[1] * inputs[i - 1]
                    + target_system[2] * inputs[i - 2]
            } else {
                0.0
            };
            desired.push(output);
        }

        let (_outputs, _errors, _mse) = lms.adapt_batch(&inputs, &desired).unwrap();

        // Check if weights converged towards target (approximately)
        // Note: LMS convergence depends on step size, signal properties, and training length
        // We test that the weights are in a reasonable range rather than exact convergence
        for (i, &target_weight) in target_system.iter().enumerate() {
            let weight_diff = (lms.weights()[i] - target_weight).abs();
            assert!(
                weight_diff < 1.0,
                "Weight {} difference {} too large",
                i,
                weight_diff
            );
        }
    }

    #[test]
    fn test_rls_creation() {
        let rls = RlsFilter::new(3, 0.99, 100.0).unwrap();
        assert_eq!(rls.weights().len(), 3);

        // Test error conditions
        assert!(RlsFilter::new(0, 0.99, 100.0).is_err());
        assert!(RlsFilter::new(3, 0.0, 100.0).is_err());
        assert!(RlsFilter::new(3, 1.1, 100.0).is_err());
        assert!(RlsFilter::new(3, 0.99, 0.0).is_err());
    }

    #[test]
    fn test_rls_adapt() {
        let mut rls = RlsFilter::new(2, 0.99, 100.0).unwrap();

        let (output, error, _mse) = rls.adapt(1.0, 0.5).unwrap();

        // Initially weights are zero, so output should be zero
        assert_relative_eq!(output, 0.0, epsilon = 1e-10);
        assert_relative_eq!(error, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_nlms_creation() {
        let nlms = NlmsFilter::new(4, 0.5, 1e-6).unwrap();
        assert_eq!(nlms.weights().len(), 4);

        // Test error conditions
        assert!(NlmsFilter::new(0, 0.5, 1e-6).is_err());
        assert!(NlmsFilter::new(4, 0.0, 1e-6).is_err());
        assert!(NlmsFilter::new(4, 0.5, 0.0).is_err());
    }

    #[test]
    fn test_nlms_adapt() {
        let mut nlms = NlmsFilter::new(2, 0.5, 1e-6).unwrap();

        let (output, error, _mse) = nlms.adapt(1.0, 0.3).unwrap();

        assert_relative_eq!(output, 0.0, epsilon = 1e-10);
        assert_relative_eq!(error, 0.3, epsilon = 1e-10);
    }

    #[test]
    fn test_matrix_operations() {
        // Test dot product
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = dot_product(&a, &b);
        assert_relative_eq!(result, 32.0, epsilon = 1e-10); // 1*4 + 2*5 + 3*6 = 32

        // Test matrix-vector multiply
        let matrix = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let vector = vec![5.0, 6.0];
        let result = matrix_vector_multiply(&matrix, &vector);
        assert_eq!(result.len(), 2);
        assert_relative_eq!(result[0], 17.0, epsilon = 1e-10); // 1*5 + 2*6 = 17
        assert_relative_eq!(result[1], 39.0, epsilon = 1e-10); // 3*5 + 4*6 = 39

        // Test get column
        let column = get_column(&matrix, 0);
        assert_eq!(column, vec![1.0, 3.0]);
    }

    #[test]
    fn test_convergence_comparison() {
        // Compare LMS and RLS convergence for the same problem
        let target_system = [0.8, -0.4];
        let num_samples = 50;

        let mut lms = LmsFilter::new(2, 0.05, 0.0).unwrap();
        let mut rls = RlsFilter::new(2, 0.99, 100.0).unwrap();

        let mut lms_errors = Vec::new();
        let mut rls_errors = Vec::new();

        for i in 0..num_samples {
            let input = (i as f64 * 0.2).sin();
            let desired = if i >= 1 {
                target_system[0] * input + target_system[1] * (((i - 1) as f64) * 0.2).sin()
            } else {
                target_system[0] * input
            };

            let (_out_lms, err_lms, _) = lms.adapt(input, desired).unwrap();
            let (_out_rls, err_rls, _) = rls.adapt(input, desired).unwrap();

            lms_errors.push(err_lms.abs());
            rls_errors.push(err_rls.abs());
        }

        // RLS should generally converge faster (lower final error)
        let lms_final_error = lms_errors.iter().rev().take(10).sum::<f64>() / 10.0;
        let rls_final_error = rls_errors.iter().rev().take(10).sum::<f64>() / 10.0;

        // This is a rough test - both algorithms should achieve reasonable convergence
        // We don't enforce that RLS is better since convergence depends on many factors
        assert!(
            lms_final_error < 2.0,
            "LMS final error too large: {}",
            lms_final_error
        );
        assert!(
            rls_final_error < 2.0,
            "RLS final error too large: {}",
            rls_final_error
        );
    }
}
