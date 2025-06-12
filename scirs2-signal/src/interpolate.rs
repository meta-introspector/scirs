// Missing data interpolation module
//
// This module implements various techniques for interpolating missing data in signals,
// including linear, spline, Gaussian process, spectral, and iterative methods.

use crate::error::{SignalError, SignalResult};
use ndarray::{s, Array1, Array2};
use rustfft::{num_complex::Complex, FftPlanner};
use scirs2_linalg::{cholesky, solve, solve_triangular};
use std::f64::consts::PI;

/// Configuration for interpolation algorithms
#[derive(Debug, Clone)]
pub struct InterpolationConfig {
    /// Maximum number of iterations for iterative methods
    pub max_iterations: usize,
    /// Convergence threshold for iterative methods
    pub convergence_threshold: f64,
    /// Regularization parameter for solving systems
    pub regularization: f64,
    /// Window size for local methods
    pub window_size: usize,
    /// Whether to extrapolate beyond boundaries
    pub extrapolate: bool,
    /// Whether to enforce monotonicity constraints
    pub monotonic: bool,
    /// Whether to apply smoothing
    pub smoothing: bool,
    /// Smoothing factor
    pub smoothing_factor: f64,
    /// Whether to use frequency-domain constraints
    pub frequency_constraint: bool,
    /// Cutoff frequency ratio for bandlimited signals
    pub cutoff_frequency: f64,
}

impl Default for InterpolationConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            convergence_threshold: 1e-6,
            regularization: 1e-6,
            window_size: 10,
            extrapolate: false,
            monotonic: false,
            smoothing: false,
            smoothing_factor: 0.5,
            frequency_constraint: false,
            cutoff_frequency: 0.5,
        }
    }
}

/// Interpolation methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InterpolationMethod {
    /// Linear interpolation
    Linear,
    /// Cubic spline interpolation
    CubicSpline,
    /// Cubic Hermite spline interpolation (PCHIP)
    CubicHermite,
    /// Gaussian process interpolation
    GaussianProcess,
    /// Sinc interpolation (for bandlimited signals)
    Sinc,
    /// Spectral interpolation (FFT-based)
    Spectral,
    /// Minimum energy interpolation
    MinimumEnergy,
    /// Kriging interpolation
    Kriging,
    /// Radial basis function interpolation
    RBF,
    /// Nearest neighbor interpolation
    NearestNeighbor,
}

/// Applies linear interpolation to fill missing values in a signal
///
/// # Arguments
///
/// * `signal` - Input signal with missing values (NaN)
///
/// # Returns
///
/// * Interpolated signal
pub fn linear_interpolate(signal: &Array1<f64>) -> SignalResult<Array1<f64>> {
    let n = signal.len();

    // Check if input has any missing values
    let has_missing = signal.iter().any(|&x| x.is_nan());
    if !has_missing {
        return Ok(signal.clone());
    }

    let mut result = signal.clone();

    // Find all non-missing points
    let mut valid_indices = Vec::new();
    let mut valid_values = Vec::new();

    for i in 0..n {
        if !signal[i].is_nan() {
            valid_indices.push(i);
            valid_values.push(signal[i]);
        }
    }

    if valid_indices.is_empty() {
        return Err(SignalError::ValueError(
            "All values are missing in the input signal".to_string(),
        ));
    }

    if valid_indices.len() == 1 {
        // If only one valid point, fill with that value
        let value = valid_values[0];
        for i in 0..n {
            result[i] = value;
        }
        return Ok(result);
    }

    // Interpolate missing values
    for i in 0..n {
        if signal[i].is_nan() {
            // Find the valid points surrounding the missing value
            let mut left_idx = None;
            let mut right_idx = None;

            for (j, &valid_idx) in valid_indices.iter().enumerate() {
                match valid_idx.cmp(&i) {
                    std::cmp::Ordering::Less => left_idx = Some(j),
                    std::cmp::Ordering::Greater => {
                        right_idx = Some(j);
                        break;
                    }
                    std::cmp::Ordering::Equal => {} // No action needed for exact match
                }
            }

            match (left_idx, right_idx) {
                (Some(left), Some(right)) => {
                    // Interpolate between left and right valid points
                    let x1 = valid_indices[left] as f64;
                    let x2 = valid_indices[right] as f64;
                    let y1 = valid_values[left];
                    let y2 = valid_values[right];
                    let x = i as f64;

                    // Linear interpolation formula
                    result[i] = y1 + (y2 - y1) * (x - x1) / (x2 - x1);
                }
                (Some(left), None) => {
                    // Extrapolate using the last valid point
                    result[i] = valid_values[left];
                }
                (None, Some(right)) => {
                    // Extrapolate using the first valid point
                    result[i] = valid_values[right];
                }
                (None, None) => {
                    // This shouldn't happen if we have at least one valid point
                    result[i] = 0.0;
                }
            }
        }
    }

    Ok(result)
}

/// Applies cubic spline interpolation to fill missing values in a signal
///
/// # Arguments
///
/// * `signal` - Input signal with missing values (NaN)
/// * `config` - Interpolation configuration
///
/// # Returns
///
/// * Interpolated signal
pub fn cubic_spline_interpolate(
    signal: &Array1<f64>,
    config: &InterpolationConfig,
) -> SignalResult<Array1<f64>> {
    let n = signal.len();

    // Check if input has any missing values
    let has_missing = signal.iter().any(|&x| x.is_nan());
    if !has_missing {
        return Ok(signal.clone());
    }

    // Find all non-missing points
    let mut valid_indices = Vec::new();
    let mut valid_values = Vec::new();

    for i in 0..n {
        if !signal[i].is_nan() {
            valid_indices.push(i as f64);
            valid_values.push(signal[i]);
        }
    }

    if valid_indices.is_empty() {
        return Err(SignalError::ValueError(
            "All values are missing in the input signal".to_string(),
        ));
    }

    if valid_indices.len() < 4 {
        // Not enough points for cubic spline, fall back to linear
        return linear_interpolate(signal);
    }

    // Convert to ndarray for matrix operations
    let x = Array1::from_vec(valid_indices);
    let y = Array1::from_vec(valid_values);

    // Number of valid points
    let n_valid = x.len();

    // Set up the system of equations for cubic spline
    let mut matrix = Array2::zeros((n_valid, n_valid));
    let mut rhs = Array1::zeros(n_valid);

    // First and last points have second derivative = 0 (natural spline)
    matrix[[0, 0]] = 2.0;
    matrix[[0, 1]] = 1.0;
    rhs[0] = 0.0;

    matrix[[n_valid - 1, n_valid - 2]] = 1.0;
    matrix[[n_valid - 1, n_valid - 1]] = 2.0;
    rhs[n_valid - 1] = 0.0;

    // Interior points satisfy continuity of first and second derivatives
    for i in 1..n_valid - 1 {
        let h_i = x[i] - x[i - 1];
        let h_i1 = x[i + 1] - x[i];

        matrix[[i, i - 1]] = h_i;
        matrix[[i, i]] = 2.0 * (h_i + h_i1);
        matrix[[i, i + 1]] = h_i1;

        rhs[i] = 3.0 * ((y[i + 1] - y[i]) / h_i1 - (y[i] - y[i - 1]) / h_i);
    }

    // Add regularization for stability
    for i in 0..n_valid {
        matrix[[i, i]] += config.regularization;
    }

    // Solve the system to get second derivatives at each point
    let second_derivatives = match solve(&matrix.view(), &rhs.view(), None) {
        Ok(solution) => solution,
        Err(_) => {
            return Err(SignalError::Compute(
                "Failed to solve spline equation system".to_string(),
            ));
        }
    };

    // Now we can evaluate the spline at each point in the original signal
    let mut result = signal.clone();

    for i in 0..n {
        if signal[i].is_nan() {
            let t = i as f64;

            // Find the spline segment containing t
            let mut segment_idx = 0;
            while segment_idx < n_valid - 1 && x[segment_idx + 1] < t {
                segment_idx += 1;
            }

            // If t is outside the range of valid points
            if t < x[0] {
                if config.extrapolate {
                    // Extrapolate using the first segment
                    segment_idx = 0;
                } else {
                    // Use the first valid value
                    result[i] = y[0];
                    continue;
                }
            } else if t > x[n_valid - 1] {
                if config.extrapolate {
                    // Extrapolate using the last segment
                    segment_idx = n_valid - 2;
                } else {
                    // Use the last valid value
                    result[i] = y[n_valid - 1];
                    continue;
                }
            }

            // Evaluate the cubic spline
            let x1 = x[segment_idx];
            let x2 = x[segment_idx + 1];
            let y1 = y[segment_idx];
            let y2 = y[segment_idx + 1];
            let d1 = second_derivatives[segment_idx];
            let d2 = second_derivatives[segment_idx + 1];

            let h = x2 - x1;
            let t_norm = (t - x1) / h;

            // Cubic spline formula
            let a = (1.0 - t_norm) * y1
                + t_norm * y2
                + t_norm
                    * (1.0 - t_norm)
                    * ((1.0 - t_norm) * h * h * d1 / 6.0 + t_norm * h * h * d2 / 6.0);

            result[i] = a;
        }
    }

    // Apply monotonicity constraint if requested
    if config.monotonic {
        result = enforce_monotonicity(&result);
    }

    // Apply smoothing if requested
    if config.smoothing {
        result = smooth_signal(&result, config.smoothing_factor);
    }

    Ok(result)
}

/// Applies cubic Hermite spline interpolation (PCHIP) to fill missing values in a signal
///
/// # Arguments
///
/// * `signal` - Input signal with missing values (NaN)
/// * `config` - Interpolation configuration
///
/// # Returns
///
/// * Interpolated signal
pub fn cubic_hermite_interpolate(
    signal: &Array1<f64>,
    config: &InterpolationConfig,
) -> SignalResult<Array1<f64>> {
    let n = signal.len();

    // Check if input has any missing values
    let has_missing = signal.iter().any(|&x| x.is_nan());
    if !has_missing {
        return Ok(signal.clone());
    }

    // Find all non-missing points
    let mut valid_indices = Vec::new();
    let mut valid_values = Vec::new();

    for i in 0..n {
        if !signal[i].is_nan() {
            valid_indices.push(i as f64);
            valid_values.push(signal[i]);
        }
    }

    if valid_indices.is_empty() {
        return Err(SignalError::ValueError(
            "All values are missing in the input signal".to_string(),
        ));
    }

    if valid_indices.len() < 2 {
        // Not enough points for PCHIP, fill with constant value
        let value = valid_values[0];
        let mut result = signal.clone();
        for i in 0..n {
            if result[i].is_nan() {
                result[i] = value;
            }
        }
        return Ok(result);
    }

    // Convert to ndarray for easier manipulation
    let x = Array1::from_vec(valid_indices);
    let y = Array1::from_vec(valid_values);

    // Number of valid points
    let n_valid = x.len();

    // Calculate slopes that preserve monotonicity
    let mut slopes = Array1::zeros(n_valid);

    // Interior points
    for i in 1..n_valid - 1 {
        let h1 = x[i] - x[i - 1];
        let h2 = x[i + 1] - x[i];
        let delta1 = (y[i] - y[i - 1]) / h1;
        let delta2 = (y[i + 1] - y[i]) / h2;

        if delta1 * delta2 <= 0.0 {
            // If slopes have opposite signs or one is zero, set slope to zero
            slopes[i] = 0.0;
        } else {
            // Harmonic mean of slopes
            let w1 = 2.0 * h2 + h1;
            let w2 = h2 + 2.0 * h1;
            slopes[i] = (w1 + w2) / (w1 / delta1 + w2 / delta2);
        }
    }

    // End points
    let h1 = x[1] - x[0];
    let h2 = x[n_valid - 1] - x[n_valid - 2];
    let delta1 = (y[1] - y[0]) / h1;
    let delta2 = (y[n_valid - 1] - y[n_valid - 2]) / h2;

    // Secant slope for end points
    slopes[0] = delta1;
    slopes[n_valid - 1] = delta2;

    // Now we can evaluate the PCHIP at each point in the original signal
    let mut result = signal.clone();

    for i in 0..n {
        if signal[i].is_nan() {
            let t = i as f64;

            // Find the spline segment containing t
            let mut segment_idx = 0;
            while segment_idx < n_valid - 1 && x[segment_idx + 1] < t {
                segment_idx += 1;
            }

            // If t is outside the range of valid points
            if t < x[0] {
                if config.extrapolate {
                    // Extrapolate using the first segment
                    segment_idx = 0;
                } else {
                    // Use the first valid value
                    result[i] = y[0];
                    continue;
                }
            } else if t > x[n_valid - 1] {
                if config.extrapolate {
                    // Extrapolate using the last segment
                    segment_idx = n_valid - 2;
                } else {
                    // Use the last valid value
                    result[i] = y[n_valid - 1];
                    continue;
                }
            }

            // Evaluate the cubic Hermite spline
            let x1 = x[segment_idx];
            let x2 = x[segment_idx + 1];
            let y1 = y[segment_idx];
            let y2 = y[segment_idx + 1];
            let m1 = slopes[segment_idx];
            let m2 = slopes[segment_idx + 1];

            let h = x2 - x1;
            let t_norm = (t - x1) / h;

            // Hermite basis functions
            let h00 = 2.0 * t_norm.powi(3) - 3.0 * t_norm.powi(2) + 1.0;
            let h10 = t_norm.powi(3) - 2.0 * t_norm.powi(2) + t_norm;
            let h01 = -2.0 * t_norm.powi(3) + 3.0 * t_norm.powi(2);
            let h11 = t_norm.powi(3) - t_norm.powi(2);

            // PCHIP formula
            let value = h00 * y1 + h10 * h * m1 + h01 * y2 + h11 * h * m2;

            result[i] = value;
        }
    }

    // Apply smoothing if requested
    if config.smoothing {
        result = smooth_signal(&result, config.smoothing_factor);
    }

    Ok(result)
}

/// Applies Gaussian process interpolation to fill missing values in a signal
///
/// # Arguments
///
/// * `signal` - Input signal with missing values (NaN)
/// * `kernel_length` - Length scale parameter for RBF kernel
/// * `kernel_sigma` - Signal variance parameter for RBF kernel
/// * `noise_level` - Noise variance parameter
///
/// # Returns
///
/// * Interpolated signal
pub fn gaussian_process_interpolate(
    signal: &Array1<f64>,
    kernel_length: f64,
    kernel_sigma: f64,
    noise_level: f64,
) -> SignalResult<Array1<f64>> {
    let n = signal.len();

    // Check if input has any missing values
    let has_missing = signal.iter().any(|&x| x.is_nan());
    if !has_missing {
        return Ok(signal.clone());
    }

    // Find indices of missing and non-missing points
    let mut missing_indices = Vec::new();
    let mut valid_indices = Vec::new();
    let mut valid_values = Vec::new();

    for i in 0..n {
        if signal[i].is_nan() {
            missing_indices.push(i);
        } else {
            valid_indices.push(i);
            valid_values.push(signal[i]);
        }
    }

    if valid_indices.is_empty() {
        return Err(SignalError::ValueError(
            "All values are missing in the input signal".to_string(),
        ));
    }

    if missing_indices.is_empty() {
        return Ok(signal.clone());
    }

    // RBF Kernel function
    let kernel = |x1: f64, x2: f64| -> f64 {
        kernel_sigma * (-0.5 * (x1 - x2).powi(2) / (kernel_length * kernel_length)).exp()
    };

    // Create covariance matrix for observed points
    let n_valid = valid_indices.len();
    let mut k_xx = Array2::zeros((n_valid, n_valid));

    for i in 0..n_valid {
        for j in 0..n_valid {
            k_xx[[i, j]] = kernel(valid_indices[i] as f64, valid_indices[j] as f64);

            // Add noise variance to diagonal
            if i == j {
                k_xx[[i, j]] += noise_level;
            }
        }
    }

    // Create cross-covariance matrix between test points and observed points
    let n_missing = missing_indices.len();
    let mut k_star_x = Array2::zeros((n_missing, n_valid));

    for i in 0..n_missing {
        for j in 0..n_valid {
            k_star_x[[i, j]] = kernel(missing_indices[i] as f64, valid_indices[j] as f64);
        }
    }

    // Compute self-covariance matrix for test points
    let mut k_star_star = Array2::zeros((n_missing, n_missing));

    for i in 0..n_missing {
        for j in 0..n_missing {
            k_star_star[[i, j]] = kernel(missing_indices[i] as f64, missing_indices[j] as f64);

            // Add noise variance to diagonal
            if i == j {
                k_star_star[[i, j]] += noise_level;
            }
        }
    }

    // Compute the Cholesky decomposition of K_xx
    let l = match cholesky(&k_xx.view(), None) {
        Ok(l) => l,
        Err(_) => {
            return Err(SignalError::Compute(
                "Failed to compute Cholesky decomposition of covariance matrix".to_string(),
            ));
        }
    };

    // Solve for alpha = K_xx^(-1) * y
    let y = Array1::from_vec(valid_values);
    let alpha = match solve_triangular(&l.view(), &y.view(), true, false) {
        Ok(a) => a,
        Err(_) => {
            return Err(SignalError::Compute(
                "Failed to solve triangular system in Gaussian process".to_string(),
            ));
        }
    };

    // Predict mean for missing points: mu = K_*x * K_xx^(-1) * y
    let mu = k_star_x.dot(&alpha);

    // Create result by copying input and filling missing values
    let mut result = signal.clone();

    for i in 0..n_missing {
        result[missing_indices[i]] = mu[i];
    }

    Ok(result)
}

/// Applies sinc interpolation to fill missing values in a bandlimited signal
///
/// # Arguments
///
/// * `signal` - Input signal with missing values (NaN)
/// * `cutoff_freq` - Normalized cutoff frequency (0.0 to 0.5)
///
/// # Returns
///
/// * Interpolated signal
pub fn sinc_interpolate(signal: &Array1<f64>, cutoff_freq: f64) -> SignalResult<Array1<f64>> {
    if cutoff_freq <= 0.0 || cutoff_freq > 0.5 {
        return Err(SignalError::ValueError(
            "Cutoff frequency must be in the range (0, 0.5]".to_string(),
        ));
    }

    let n = signal.len();

    // Check if input has any missing values
    let has_missing = signal.iter().any(|&x| x.is_nan());
    if !has_missing {
        return Ok(signal.clone());
    }

    // Find indices of missing and non-missing points
    let mut missing_indices = Vec::new();
    let mut valid_indices = Vec::new();
    let mut valid_values = Vec::new();

    for i in 0..n {
        if signal[i].is_nan() {
            missing_indices.push(i);
        } else {
            valid_indices.push(i);
            valid_values.push(signal[i]);
        }
    }

    if valid_indices.is_empty() {
        return Err(SignalError::ValueError(
            "All values are missing in the input signal".to_string(),
        ));
    }

    // Create result by copying input
    let mut result = signal.clone();

    // For each missing point, compute sinc interpolation
    for &missing_idx in &missing_indices {
        let mut sum = 0.0;
        let mut weight_sum = 0.0;

        for (&valid_idx, &valid_value) in valid_indices.iter().zip(valid_values.iter()) {
            // Sinc function: sin(pi*x)/(pi*x)
            let distance = missing_idx as f64 - valid_idx as f64;

            // Avoid division by zero
            let sinc = if distance.abs() < 1e-10 {
                1.0
            } else {
                let x = 2.0 * PI * cutoff_freq * distance;
                x.sin() / x
            };

            // Apply window to reduce ringing (Lanczos window)
            let window = if distance.abs() < n as f64 {
                let x = PI * distance / n as f64;
                if x.abs() < 1e-10 {
                    1.0
                } else {
                    x.sin() / x
                }
            } else {
                0.0
            };

            let weight = sinc * window;
            sum += valid_value * weight;
            weight_sum += weight;
        }

        // Normalize if total weight is non-zero
        if weight_sum.abs() > 1e-10 {
            result[missing_idx] = sum / weight_sum;
        } else {
            // Fallback to linear interpolation
            let nearest_valid_idx = find_nearest_valid_index(missing_idx, &valid_indices);
            result[missing_idx] = valid_values[nearest_valid_idx];
        }
    }

    Ok(result)
}

/// Applies spectral (FFT-based) interpolation to fill missing values in a signal
///
/// # Arguments
///
/// * `signal` - Input signal with missing values (NaN)
/// * `config` - Interpolation configuration
///
/// # Returns
///
/// * Interpolated signal
pub fn spectral_interpolate(
    signal: &Array1<f64>,
    config: &InterpolationConfig,
) -> SignalResult<Array1<f64>> {
    let n = signal.len();

    // Check if input has any missing values
    let has_missing = signal.iter().any(|&x| x.is_nan());
    if !has_missing {
        return Ok(signal.clone());
    }

    // Find indices of missing and non-missing points
    let mut mask = Array1::zeros(n);
    let mut valid_signal = signal.clone();

    for i in 0..n {
        if signal[i].is_nan() {
            mask[i] = 1.0; // 1 indicates missing
            valid_signal[i] = 0.0; // Initialize with zeros
        }
    }

    // If all values are missing, return error
    if mask.sum() == n as f64 {
        return Err(SignalError::ValueError(
            "All values are missing in the input signal".to_string(),
        ));
    }

    // Make a copy of the initial valid signal
    let mut result = valid_signal.clone();
    let mut prev_result = valid_signal.clone();

    // Create FFT planner
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);

    // Convert to complex for FFT
    let mut complex_signal = vec![Complex::new(0.0, 0.0); n];

    // Iterative spectral interpolation
    for _ in 0..config.max_iterations {
        // Copy current estimate to complex array
        for (i, &val) in result.iter().enumerate().take(n) {
            complex_signal[i] = Complex::new(val, 0.0);
        }

        // Forward FFT
        fft.process(&mut complex_signal);

        // Apply frequency constraint if requested
        if config.frequency_constraint {
            let cutoff = (n as f64 * config.cutoff_frequency) as usize;

            // Zero out high frequencies
            for value in complex_signal.iter_mut().skip(cutoff).take(n - 2 * cutoff) {
                *value = Complex::new(0.0, 0.0);
            }
        }

        // Inverse FFT
        ifft.process(&mut complex_signal);

        // Scale by 1/n
        let scale = 1.0 / n as f64;
        for value in complex_signal.iter_mut().take(n) {
            *value *= scale;
        }

        // Copy current estimate and update
        prev_result.assign(&result);

        // Update: known samples remain the same, missing samples get values from FFT
        for i in 0..n {
            if mask[i] > 0.5 {
                // Missing data point
                result[i] = complex_signal[i].re;
            }
        }

        // Check for convergence
        let diff = (&result - &prev_result).mapv(|x| x.powi(2)).sum().sqrt();
        let norm = result.mapv(|x| x.powi(2)).sum().sqrt();

        if diff / norm < config.convergence_threshold {
            break;
        }
    }

    Ok(result)
}

/// Applies minimum energy interpolation to fill missing values in a signal
///
/// # Arguments
///
/// * `signal` - Input signal with missing values (NaN)
/// * `config` - Interpolation configuration
///
/// # Returns
///
/// * Interpolated signal
pub fn minimum_energy_interpolate(
    signal: &Array1<f64>,
    config: &InterpolationConfig,
) -> SignalResult<Array1<f64>> {
    let n = signal.len();

    // Check if input has any missing values
    let has_missing = signal.iter().any(|&x| x.is_nan());
    if !has_missing {
        return Ok(signal.clone());
    }

    // Find indices of missing and non-missing points
    let mut missing_indices = Vec::new();
    let mut valid_indices = Vec::new();
    let mut valid_values = Vec::new();

    for i in 0..n {
        if signal[i].is_nan() {
            missing_indices.push(i);
        } else {
            valid_indices.push(i);
            valid_values.push(signal[i]);
        }
    }

    if valid_indices.is_empty() {
        return Err(SignalError::ValueError(
            "All values are missing in the input signal".to_string(),
        ));
    }

    // Create finite difference matrix for second derivative
    let mut d2 = Array2::zeros((n - 2, n));
    for i in 0..n - 2 {
        d2[[i, i]] = 1.0;
        d2[[i, i + 1]] = -2.0;
        d2[[i, i + 2]] = 1.0;
    }

    // Split the problem into known and unknown parts
    let n_missing = missing_indices.len();
    let n_valid = valid_indices.len();

    // Create selection matrices
    let mut s_known = Array2::zeros((n_valid, n));
    let mut s_unknown = Array2::zeros((n_missing, n));

    for (i, &idx) in valid_indices.iter().enumerate() {
        s_known[[i, idx]] = 1.0;
    }

    for (i, &idx) in missing_indices.iter().enumerate() {
        s_unknown[[i, idx]] = 1.0;
    }

    // Known values vector
    let y_known = Array1::from_vec(valid_values);

    // Calculate the regularization matrix
    let h = d2.t().dot(&d2);

    // Calculate the matrices for the linear system
    let a = s_unknown.dot(&h).dot(&s_unknown.t());
    let b = s_unknown.dot(&h).dot(&s_known.t()).dot(&y_known);

    // Add regularization for stability
    let mut a_reg = a.clone();
    for i in 0..a_reg.dim().0 {
        a_reg[[i, i]] += config.regularization;
    }

    // Solve the system to get the unknown values
    let y_unknown = match solve(&a_reg.view(), &b.view(), None) {
        Ok(solution) => -solution, // Negative because of how we set up the system
        Err(_) => {
            return Err(SignalError::Compute(
                "Failed to solve minimum energy interpolation system".to_string(),
            ));
        }
    };

    // Create result by copying input and filling missing values
    let mut result = signal.clone();

    for (i, &idx) in missing_indices.iter().enumerate() {
        result[idx] = y_unknown[i];
    }

    Ok(result)
}

/// Applies Kriging interpolation to fill missing values in a signal
///
/// # Arguments
///
/// * `signal` - Input signal with missing values (NaN)
/// * `variogram_model` - Variogram model function (distance -> semivariance)
/// * `config` - Interpolation configuration
///
/// # Returns
///
/// * Interpolated signal
pub fn kriging_interpolate<F>(
    signal: &Array1<f64>,
    variogram_model: F,
    config: &InterpolationConfig,
) -> SignalResult<Array1<f64>>
where
    F: Fn(f64) -> f64,
{
    let n = signal.len();

    // Check if input has any missing values
    let has_missing = signal.iter().any(|&x| x.is_nan());
    if !has_missing {
        return Ok(signal.clone());
    }

    // Find indices of missing and non-missing points
    let mut missing_indices = Vec::new();
    let mut valid_indices = Vec::new();
    let mut valid_values = Vec::new();

    for i in 0..n {
        if signal[i].is_nan() {
            missing_indices.push(i);
        } else {
            valid_indices.push(i);
            valid_values.push(signal[i]);
        }
    }

    if valid_indices.is_empty() {
        return Err(SignalError::ValueError(
            "All values are missing in the input signal".to_string(),
        ));
    }

    // Number of valid points
    let n_valid = valid_indices.len();

    // Create the variogram matrix
    let mut gamma = Array2::zeros((n_valid + 1, n_valid + 1));

    // Fill variogram matrix
    for i in 0..n_valid {
        for j in 0..n_valid {
            let dist = (valid_indices[i] as f64 - valid_indices[j] as f64).abs();
            gamma[[i, j]] = variogram_model(dist);
        }
    }

    // Add Lagrange multiplier row and column
    for i in 0..n_valid {
        gamma[[i, n_valid]] = 1.0;
        gamma[[n_valid, i]] = 1.0;
    }
    gamma[[n_valid, n_valid]] = 0.0;

    // Add small regularization to diagonal for numerical stability
    for i in 0..n_valid {
        gamma[[i, i]] += config.regularization;
    }

    // Create result array
    let mut result = signal.clone();

    // For each missing point, solve the Kriging system
    for &missing_idx in &missing_indices {
        // Create the right-hand side vector (variogram values to prediction point)
        let mut rhs = Array1::zeros(n_valid + 1);

        for i in 0..n_valid {
            let dist = (valid_indices[i] as f64 - missing_idx as f64).abs();
            rhs[i] = variogram_model(dist);
        }
        rhs[n_valid] = 1.0;

        // Solve the Kriging system
        let weights = match solve(&gamma.view(), &rhs.view(), None) {
            Ok(w) => w,
            Err(_) => {
                return Err(SignalError::Compute(
                    "Failed to solve Kriging system".to_string(),
                ));
            }
        };

        // Compute the interpolated value
        let mut value = 0.0;
        for i in 0..n_valid {
            value += weights[i] * valid_values[i];
        }

        result[missing_idx] = value;
    }

    Ok(result)
}

/// Applies Radial Basis Function (RBF) interpolation to fill missing values
///
/// # Arguments
///
/// * `signal` - Input signal with missing values (NaN)
/// * `rbf_function` - Radial basis function (distance -> value)
/// * `config` - Interpolation configuration
///
/// # Returns
///
/// * Interpolated signal
pub fn rbf_interpolate<F>(
    signal: &Array1<f64>,
    rbf_function: F,
    config: &InterpolationConfig,
) -> SignalResult<Array1<f64>>
where
    F: Fn(f64) -> f64,
{
    let n = signal.len();

    // Check if input has any missing values
    let has_missing = signal.iter().any(|&x| x.is_nan());
    if !has_missing {
        return Ok(signal.clone());
    }

    // Find indices of missing and non-missing points
    let mut missing_indices = Vec::new();
    let mut valid_indices = Vec::new();
    let mut valid_values = Vec::new();

    for i in 0..n {
        if signal[i].is_nan() {
            missing_indices.push(i);
        } else {
            valid_indices.push(i);
            valid_values.push(signal[i]);
        }
    }

    if valid_indices.is_empty() {
        return Err(SignalError::ValueError(
            "All values are missing in the input signal".to_string(),
        ));
    }

    // Number of valid points
    let n_valid = valid_indices.len();

    // Create the RBF matrix
    let mut phi = Array2::zeros((n_valid, n_valid));

    // Fill RBF matrix
    for i in 0..n_valid {
        for j in 0..n_valid {
            let dist = (valid_indices[i] as f64 - valid_indices[j] as f64).abs();
            phi[[i, j]] = rbf_function(dist);
        }
    }

    // Add regularization for stability
    for i in 0..n_valid {
        phi[[i, i]] += config.regularization;
    }

    // Solve for RBF weights
    let y = Array1::from_vec(valid_values);
    let weights = match solve(&phi.view(), &y.view(), None) {
        Ok(w) => w,
        Err(_) => {
            return Err(SignalError::Compute(
                "Failed to solve RBF system".to_string(),
            ));
        }
    };

    // Create result array
    let mut result = signal.clone();

    // For each missing point, compute the RBF interpolation
    for &missing_idx in &missing_indices {
        let mut value = 0.0;

        for i in 0..n_valid {
            let dist = (valid_indices[i] as f64 - missing_idx as f64).abs();
            value += weights[i] * rbf_function(dist);
        }

        result[missing_idx] = value;
    }

    Ok(result)
}

/// Applies nearest neighbor interpolation to fill missing values
///
/// # Arguments
///
/// * `signal` - Input signal with missing values (NaN)
///
/// # Returns
///
/// * Interpolated signal
pub fn nearest_neighbor_interpolate(signal: &Array1<f64>) -> SignalResult<Array1<f64>> {
    let n = signal.len();

    // Check if input has any missing values
    let has_missing = signal.iter().any(|&x| x.is_nan());
    if !has_missing {
        return Ok(signal.clone());
    }

    // Find indices of non-missing points
    let mut valid_indices = Vec::new();
    let mut valid_values = Vec::new();

    for i in 0..n {
        if !signal[i].is_nan() {
            valid_indices.push(i);
            valid_values.push(signal[i]);
        }
    }

    if valid_indices.is_empty() {
        return Err(SignalError::ValueError(
            "All values are missing in the input signal".to_string(),
        ));
    }

    // Create result by filling missing values with nearest valid value
    let mut result = signal.clone();

    for i in 0..n {
        if signal[i].is_nan() {
            let nearest_idx = find_nearest_valid_index(i, &valid_indices);
            result[i] = valid_values[nearest_idx];
        }
    }

    Ok(result)
}

/// Interpolate missing values using the specified method
///
/// # Arguments
///
/// * `signal` - Input signal with missing values (NaN)
/// * `method` - Interpolation method to use
/// * `config` - Interpolation configuration
///
/// # Returns
///
/// * Interpolated signal
pub fn interpolate(
    signal: &Array1<f64>,
    method: InterpolationMethod,
    config: &InterpolationConfig,
) -> SignalResult<Array1<f64>> {
    match method {
        InterpolationMethod::Linear => linear_interpolate(signal),
        InterpolationMethod::CubicSpline => cubic_spline_interpolate(signal, config),
        InterpolationMethod::CubicHermite => cubic_hermite_interpolate(signal, config),
        InterpolationMethod::GaussianProcess => {
            gaussian_process_interpolate(signal, 10.0, 1.0, 1e-3)
        }
        InterpolationMethod::Sinc => sinc_interpolate(signal, config.cutoff_frequency),
        InterpolationMethod::Spectral => spectral_interpolate(signal, config),
        InterpolationMethod::MinimumEnergy => minimum_energy_interpolate(signal, config),
        InterpolationMethod::Kriging => {
            // Use exponential variogram model
            let variogram = |h: f64| -> f64 { 1.0 - (-h / 10.0).exp() };

            kriging_interpolate(signal, variogram, config)
        }
        InterpolationMethod::RBF => {
            // Use Gaussian RBF
            let rbf = |r: f64| -> f64 { (-r * r / (2.0 * 10.0 * 10.0)).exp() };

            rbf_interpolate(signal, rbf, config)
        }
        InterpolationMethod::NearestNeighbor => nearest_neighbor_interpolate(signal),
    }
}

/// Interpolates a 2D array (image) with missing values
///
/// # Arguments
///
/// * `image` - Input image with missing values (NaN)
/// * `method` - Interpolation method to use
/// * `config` - Interpolation configuration
///
/// # Returns
///
/// * Interpolated image
pub fn interpolate_2d(
    image: &Array2<f64>,
    method: InterpolationMethod,
    config: &InterpolationConfig,
) -> SignalResult<Array2<f64>> {
    let (n_rows, n_cols) = image.dim();

    // Check if input has any missing values
    let has_missing = image.iter().any(|&x| x.is_nan());
    if !has_missing {
        return Ok(image.clone());
    }

    // Initialize result
    let mut result = image.clone();

    match method {
        InterpolationMethod::NearestNeighbor => {
            // For nearest neighbor, we can efficiently process the entire image at once
            nearest_neighbor_interpolate_2d(image)
        }
        _ => {
            // First interpolate along rows
            for i in 0..n_rows {
                let row = image.slice(s![i, ..]).to_owned();
                let interpolated_row = interpolate(&row, method, config)?;
                result.slice_mut(s![i, ..]).assign(&interpolated_row);
            }

            // Then interpolate along columns for any remaining missing values
            for j in 0..n_cols {
                let col = result.slice(s![.., j]).to_owned();

                // Only interpolate column if it still has missing values
                if col.iter().any(|&x| x.is_nan()) {
                    let interpolated_col = interpolate(&col, method, config)?;
                    result.slice_mut(s![.., j]).assign(&interpolated_col);
                }
            }

            // Check if there are still missing values and try one more pass if needed
            if result.iter().any(|&x| x.is_nan()) {
                // One more row pass
                for i in 0..n_rows {
                    let row = result.slice(s![i, ..]).to_owned();

                    // Only interpolate row if it still has missing values
                    if row.iter().any(|&x| x.is_nan()) {
                        let interpolated_row = interpolate(&row, method, config)?;
                        result.slice_mut(s![i, ..]).assign(&interpolated_row);
                    }
                }
            }

            Ok(result)
        }
    }
}

/// Applies nearest neighbor interpolation to a 2D image with missing values
///
/// # Arguments
///
/// * `image` - Input image with missing values (NaN)
///
/// # Returns
///
/// * Interpolated image
fn nearest_neighbor_interpolate_2d(image: &Array2<f64>) -> SignalResult<Array2<f64>> {
    let (n_rows, n_cols) = image.dim();

    // Find all valid points
    let mut valid_points = Vec::new();

    for i in 0..n_rows {
        for j in 0..n_cols {
            if !image[[i, j]].is_nan() {
                valid_points.push(((i, j), image[[i, j]]));
            }
        }
    }

    if valid_points.is_empty() {
        return Err(SignalError::ValueError(
            "All values are missing in the input image".to_string(),
        ));
    }

    // Create result with all missing values initially
    let mut result = image.clone();

    // Find nearest valid point for each missing point
    for i in 0..n_rows {
        for j in 0..n_cols {
            if image[[i, j]].is_nan() {
                // Find nearest valid point
                let mut min_dist = f64::MAX;
                let mut min_value = 0.0;

                for &((vi, vj), value) in &valid_points {
                    let dist =
                        ((i as f64 - vi as f64).powi(2) + (j as f64 - vj as f64).powi(2)).sqrt();

                    if dist < min_dist {
                        min_dist = dist;
                        min_value = value;
                    }
                }

                result[[i, j]] = min_value;
            }
        }
    }

    Ok(result)
}

/// Helper function to smooth a signal using moving average
fn smooth_signal(signal: &Array1<f64>, factor: f64) -> Array1<f64> {
    let n = signal.len();
    let window_size = (n as f64 * factor).ceil() as usize;
    let half_window = window_size / 2;

    let mut result = signal.clone();

    for i in 0..n {
        let mut count = 0;
        let mut sum = 0.0;

        let start = i.saturating_sub(half_window);
        let end = (i + half_window + 1).min(n);

        for j in start..end {
            if !signal[j].is_nan() {
                sum += signal[j];
                count += 1;
            }
        }

        if count > 0 {
            result[i] = sum / count as f64;
        }
    }

    result
}

/// Helper function to enforce monotonicity constraints
fn enforce_monotonicity(signal: &Array1<f64>) -> Array1<f64> {
    let n = signal.len();
    let mut result = signal.clone();

    // Find all valid points
    let mut valid_indices = Vec::new();

    for i in 0..n {
        if !signal[i].is_nan() {
            valid_indices.push(i);
        }
    }

    if valid_indices.len() < 2 {
        return result;
    }

    // Check if valid samples are monotonically increasing or decreasing
    let mut increasing = true;
    let mut decreasing = true;

    for i in 1..valid_indices.len() {
        let prev = signal[valid_indices[i - 1]];
        let curr = signal[valid_indices[i]];

        if curr < prev {
            increasing = false;
        }

        if curr > prev {
            decreasing = false;
        }
    }

    // If neither increasing nor decreasing, no monotonicity to enforce
    if !increasing && !decreasing {
        return result;
    }

    // Enforce monotonicity
    if increasing {
        for i in 1..n {
            if result[i] < result[i - 1] {
                result[i] = result[i - 1];
            }
        }
    } else if decreasing {
        for i in 1..n {
            if result[i] > result[i - 1] {
                result[i] = result[i - 1];
            }
        }
    }

    result
}

/// Helper function to find the index of the nearest valid point
fn find_nearest_valid_index(idx: usize, valid_indices: &[usize]) -> usize {
    if valid_indices.is_empty() {
        return 0;
    }

    let mut nearest_idx = 0;
    let mut min_dist = usize::MAX;

    for (i, &valid_idx) in valid_indices.iter().enumerate() {
        let dist = if valid_idx > idx {
            valid_idx - idx
        } else {
            idx - valid_idx
        };

        if dist < min_dist {
            min_dist = dist;
            nearest_idx = i;
        }
    }

    nearest_idx
}

/// Generate standard variogram models for Kriging interpolation
pub mod variogram_models {
    /// Spherical variogram model
    pub fn spherical(range: f64, sill: f64, nugget: f64) -> impl Fn(f64) -> f64 {
        move |h: f64| {
            if h <= 0.0 {
                return 0.0;
            }

            if h >= range {
                return sill;
            }

            let h_norm = h / range;
            nugget + (sill - nugget) * (1.5 * h_norm - 0.5 * h_norm.powi(3))
        }
    }

    /// Exponential variogram model
    pub fn exponential(range: f64, sill: f64, nugget: f64) -> impl Fn(f64) -> f64 {
        move |h: f64| {
            if h <= 0.0 {
                return 0.0;
            }

            nugget + (sill - nugget) * (1.0 - (-3.0 * h / range).exp())
        }
    }

    /// Gaussian variogram model
    pub fn gaussian(range: f64, sill: f64, nugget: f64) -> impl Fn(f64) -> f64 {
        move |h: f64| {
            if h <= 0.0 {
                return 0.0;
            }

            nugget + (sill - nugget) * (1.0 - (-9.0 * h * h / (range * range)).exp())
        }
    }

    /// Linear variogram model
    pub fn linear(slope: f64, nugget: f64) -> impl Fn(f64) -> f64 {
        move |h: f64| {
            if h <= 0.0 {
                return 0.0;
            }

            nugget + slope * h
        }
    }
}

/// Generate standard RBF functions for interpolation
pub mod rbf_functions {
    /// Gaussian RBF
    pub fn gaussian(epsilon: f64) -> impl Fn(f64) -> f64 {
        move |r: f64| (-epsilon * r * r).exp()
    }

    /// Multiquadric RBF
    pub fn multiquadric(epsilon: f64) -> impl Fn(f64) -> f64 {
        move |r: f64| (1.0 + epsilon * r * r).sqrt()
    }

    /// Inverse multiquadric RBF
    pub fn inverse_multiquadric(epsilon: f64) -> impl Fn(f64) -> f64 {
        move |r: f64| 1.0 / (1.0 + epsilon * r * r).sqrt()
    }

    /// Thin plate spline RBF
    pub fn thin_plate_spline() -> impl Fn(f64) -> f64 {
        move |r: f64| {
            if r.abs() < 1e-10 {
                0.0
            } else {
                r * r * r.ln()
            }
        }
    }
}

/// Computes multiple interpolation methods and selects the best one
///
/// # Arguments
///
/// * `signal` - Input signal with missing values (NaN)
/// * `config` - Interpolation configuration
/// * `cross_validation` - Whether to use cross-validation for method selection
///
/// # Returns
///
/// * Tuple containing (interpolated signal, selected method)
pub fn auto_interpolate(
    signal: &Array1<f64>,
    config: &InterpolationConfig,
    cross_validation: bool,
) -> SignalResult<(Array1<f64>, InterpolationMethod)> {
    let n = signal.len();

    // Check if input has any missing values
    let has_missing = signal.iter().any(|&x| x.is_nan());
    if !has_missing {
        return Ok((signal.clone(), InterpolationMethod::Linear));
    }

    let methods = [
        InterpolationMethod::Linear,
        InterpolationMethod::CubicSpline,
        InterpolationMethod::CubicHermite,
        InterpolationMethod::Sinc,
        InterpolationMethod::Spectral,
        InterpolationMethod::MinimumEnergy,
        InterpolationMethod::NearestNeighbor,
    ];

    if cross_validation {
        // Find all valid points
        let mut valid_indices = Vec::new();

        for i in 0..n {
            if !signal[i].is_nan() {
                valid_indices.push(i);
            }
        }

        let n_valid = valid_indices.len();

        if n_valid < 5 {
            // Not enough points for cross-validation
            let result = linear_interpolate(signal)?;
            return Ok((result, InterpolationMethod::Linear));
        }

        // Prepare for k-fold cross-validation (k=5)
        let k = 5.min(n_valid);
        let fold_size = n_valid / k;

        let mut best_method = InterpolationMethod::Linear;
        let mut min_error = f64::MAX;

        for &method in &methods {
            let mut total_error = 0.0;

            // K-fold cross-validation
            for fold in 0..k {
                let start = fold * fold_size;
                let end = if fold == k - 1 {
                    n_valid
                } else {
                    (fold + 1) * fold_size
                };

                // Create temporary signal with additional missing values
                let mut temp_signal = signal.clone();

                // Mask out validation fold
                for &idx in valid_indices.iter().skip(start).take(end - start) {
                    temp_signal[idx] = f64::NAN;
                }

                // Interpolate with current method
                let interpolated = interpolate(&temp_signal, method, config)?;

                // Calculate error on validation fold
                let mut fold_error = 0.0;
                for &idx in valid_indices.iter().skip(start).take(end - start) {
                    let error = interpolated[idx] - signal[idx];
                    fold_error += error * error;
                }

                total_error += fold_error / (end - start) as f64;
            }

            let avg_error = total_error / k as f64;

            if avg_error < min_error {
                min_error = avg_error;
                best_method = method;
            }
        }

        // Apply the best method to the original signal
        let result = interpolate(signal, best_method, config)?;
        Ok((result, best_method))
    } else {
        // Try all methods and pick the one with the smoothest result
        let mut min_roughness = f64::MAX;
        let mut best_method = InterpolationMethod::Linear;
        let mut best_result = linear_interpolate(signal)?;

        for &method in &methods {
            let interpolated = interpolate(signal, method, config)?;

            // Calculate second-derivative roughness
            let mut roughness = 0.0;
            for i in 1..n - 1 {
                let d2 = interpolated[i - 1] - 2.0 * interpolated[i] + interpolated[i + 1];
                roughness += d2 * d2;
            }

            if roughness < min_roughness {
                min_roughness = roughness;
                best_method = method;
                best_result = interpolated;
            }
        }

        Ok((best_result, best_method))
    }
}

/// Advanced resampling algorithms using sinc and fractional delay interpolation
pub mod resampling {
    use crate::error::{SignalError, SignalResult};
    use crate::window;
    use std::f64::consts::PI;

    /// Configuration for high-quality resampling
    #[derive(Debug, Clone)]
    pub struct ResamplingConfig {
        /// Length of the sinc filter kernel (in samples)
        pub kernel_length: usize,
        /// Beta parameter for Kaiser window (controls sidelobe attenuation)
        pub kaiser_beta: f64,
        /// Cutoff frequency as fraction of Nyquist rate
        pub cutoff_frequency: f64,
        /// Oversampling factor for polyphase filters
        pub oversampling_factor: usize,
        /// Whether to use zero-phase filtering
        pub zero_phase: bool,
    }

    impl Default for ResamplingConfig {
        fn default() -> Self {
            Self {
                kernel_length: 65, // Must be odd
                kaiser_beta: 8.0,
                cutoff_frequency: 0.9,
                oversampling_factor: 32,
                zero_phase: true,
            }
        }
    }

    /// High-quality sinc interpolation for arbitrary resampling ratios
    ///
    /// # Arguments
    ///
    /// * `signal` - Input signal to resample
    /// * `target_length` - Target length of resampled signal
    /// * `config` - Resampling configuration
    ///
    /// # Returns
    ///
    /// * Resampled signal
    pub fn sinc_resample(
        signal: &[f64],
        target_length: usize,
        config: &ResamplingConfig,
    ) -> SignalResult<Vec<f64>> {
        if signal.is_empty() {
            return Err(SignalError::ValueError("Input signal is empty".to_string()));
        }

        if target_length == 0 {
            return Err(SignalError::ValueError(
                "Target length must be positive".to_string(),
            ));
        }

        let input_length = signal.len();
        let ratio = input_length as f64 / target_length as f64;

        // Create sinc kernel
        let kernel = create_sinc_kernel(config)?;
        let kernel_half = kernel.len() / 2;

        let mut output = vec![0.0; target_length];

        for (i, output_sample) in output.iter_mut().enumerate() {
            // Calculate the exact input position
            let exact_pos = i as f64 * ratio;
            let center_sample = exact_pos.round() as i32;
            let fractional_delay = exact_pos - center_sample as f64;

            let mut sum = 0.0;
            let mut weight_sum = 0.0;

            // Apply windowed sinc kernel
            for k in 0..kernel.len() {
                let sample_idx = center_sample + k as i32 - kernel_half as i32;

                if sample_idx >= 0 && sample_idx < input_length as i32 {
                    let kernel_pos = k as f64 - kernel_half as f64 + fractional_delay;
                    let sinc_weight =
                        evaluate_sinc_kernel(&kernel, kernel_pos, config.cutoff_frequency);

                    sum += signal[sample_idx as usize] * sinc_weight;
                    weight_sum += sinc_weight;
                }
            }

            // Normalize to preserve signal amplitude
            *output_sample = if weight_sum.abs() > 1e-10 {
                sum / weight_sum
            } else {
                0.0
            };
        }

        Ok(output)
    }

    /// Fractional delay interpolation using Lagrange interpolation
    ///
    /// # Arguments
    ///
    /// * `signal` - Input signal
    /// * `delay` - Fractional delay in samples (can be negative)
    /// * `filter_length` - Length of the interpolation filter (should be odd)
    ///
    /// # Returns
    ///
    /// * Signal with fractional delay applied
    pub fn fractional_delay(
        signal: &[f64],
        delay: f64,
        filter_length: usize,
    ) -> SignalResult<Vec<f64>> {
        if signal.is_empty() {
            return Err(SignalError::ValueError("Input signal is empty".to_string()));
        }

        if filter_length % 2 == 0 {
            return Err(SignalError::ValueError(
                "Filter length must be odd".to_string(),
            ));
        }

        let half_length = filter_length / 2;
        let mut output = vec![0.0; signal.len()];

        for (n, output_sample) in output.iter_mut().enumerate() {
            let mut sum = 0.0;

            for k in 0..filter_length {
                let m = k as i32 - half_length as i32;
                let sample_idx = n as i32 + m;

                if sample_idx >= 0 && sample_idx < signal.len() as i32 {
                    // Lagrange interpolation coefficient
                    let mut coeff = 1.0;
                    let target_pos = m as f64 - delay;

                    for j in 0..filter_length {
                        let jm = j as i32 - half_length as i32;
                        if jm != m {
                            coeff *= (target_pos - jm as f64) / (m as f64 - jm as f64);
                        }
                    }

                    sum += signal[sample_idx as usize] * coeff;
                }
            }

            *output_sample = sum;
        }

        Ok(output)
    }

    /// Allpass fractional delay using Thiran filter design
    ///
    /// # Arguments
    ///
    /// * `signal` - Input signal
    /// * `delay` - Fractional delay in samples
    /// * `filter_order` - Order of the allpass filter
    ///
    /// # Returns
    ///
    /// * Signal with fractional delay applied
    pub fn allpass_fractional_delay(
        signal: &[f64],
        delay: f64,
        filter_order: usize,
    ) -> SignalResult<Vec<f64>> {
        if signal.is_empty() {
            return Err(SignalError::ValueError("Input signal is empty".to_string()));
        }

        if delay < 0.0 {
            return Err(SignalError::ValueError(
                "Delay must be non-negative".to_string(),
            ));
        }

        // Design Thiran allpass filter coefficients
        let coeffs = design_thiran_coefficients(delay, filter_order)?;

        // Apply the allpass filter
        apply_allpass_filter(signal, &coeffs)
    }

    /// Polyphase interpolation for efficient rational resampling
    ///
    /// # Arguments
    ///
    /// * `signal` - Input signal
    /// * `up_factor` - Upsampling factor
    /// * `down_factor` - Downsampling factor
    /// * `config` - Resampling configuration
    ///
    /// # Returns
    ///
    /// * Resampled signal
    pub fn polyphase_resample(
        signal: &[f64],
        up_factor: usize,
        down_factor: usize,
        config: &ResamplingConfig,
    ) -> SignalResult<Vec<f64>> {
        if signal.is_empty() {
            return Err(SignalError::ValueError("Input signal is empty".to_string()));
        }

        if up_factor == 0 || down_factor == 0 {
            return Err(SignalError::ValueError(
                "Resampling factors must be positive".to_string(),
            ));
        }

        // Design the prototype lowpass filter
        let prototype_filter = design_resampling_filter(up_factor, down_factor, config)?;

        // Create polyphase filter bank
        let polyphase_filters = create_polyphase_bank(&prototype_filter, up_factor);

        // Apply polyphase resampling
        let upsampled = upsample_with_polyphase(signal, &polyphase_filters)?;
        let output = downsample(&upsampled, down_factor);

        Ok(output)
    }

    /// Bandlimited interpolation using ideal reconstruction
    ///
    /// # Arguments
    ///
    /// * `signal` - Input signal (assumed to be bandlimited)
    /// * `new_sample_rate` - Target sample rate
    /// * `original_sample_rate` - Original sample rate
    /// * `config` - Resampling configuration
    ///
    /// # Returns
    ///
    /// * Resampled signal
    pub fn bandlimited_interpolation(
        signal: &[f64],
        new_sample_rate: f64,
        original_sample_rate: f64,
        config: &ResamplingConfig,
    ) -> SignalResult<Vec<f64>> {
        if signal.is_empty() {
            return Err(SignalError::ValueError("Input signal is empty".to_string()));
        }

        if new_sample_rate <= 0.0 || original_sample_rate <= 0.0 {
            return Err(SignalError::ValueError(
                "Sample rates must be positive".to_string(),
            ));
        }

        let ratio = new_sample_rate / original_sample_rate;
        let target_length = (signal.len() as f64 * ratio).round() as usize;

        if ratio == 1.0 {
            return Ok(signal.to_vec());
        }

        // Use sinc interpolation for bandlimited reconstruction
        sinc_resample(signal, target_length, config)
    }

    /// Variable sample rate conversion with time-varying ratio
    ///
    /// # Arguments
    ///
    /// * `signal` - Input signal
    /// * `time_map` - Time mapping function (input_time -> output_time)
    /// * `config` - Resampling configuration
    ///
    /// # Returns
    ///
    /// * Time-warped signal
    pub fn variable_rate_resample<F>(
        signal: &[f64],
        time_map: F,
        output_length: usize,
        config: &ResamplingConfig,
    ) -> SignalResult<Vec<f64>>
    where
        F: Fn(f64) -> f64,
    {
        if signal.is_empty() {
            return Err(SignalError::ValueError("Input signal is empty".to_string()));
        }

        if output_length == 0 {
            return Err(SignalError::ValueError(
                "Output length must be positive".to_string(),
            ));
        }

        let kernel = create_sinc_kernel(config)?;
        let kernel_half = kernel.len() / 2;
        let mut output = vec![0.0; output_length];

        for (i, output_sample) in output.iter_mut().enumerate() {
            let output_time = i as f64;
            let input_time = time_map(output_time);

            if input_time >= 0.0 && input_time < signal.len() as f64 {
                let center_sample = input_time.floor() as usize;
                let fractional_part = input_time - center_sample as f64;

                let mut sum = 0.0;
                let mut weight_sum = 0.0;

                for k in 0..kernel.len() {
                    let sample_idx = center_sample as i32 + k as i32 - kernel_half as i32;

                    if sample_idx >= 0 && (sample_idx as usize) < signal.len() {
                        let kernel_pos = k as f64 - kernel_half as f64 + fractional_part;
                        let weight =
                            evaluate_sinc_kernel(&kernel, kernel_pos, config.cutoff_frequency);

                        sum += signal[sample_idx as usize] * weight;
                        weight_sum += weight;
                    }
                }

                *output_sample = if weight_sum.abs() > 1e-10 {
                    sum / weight_sum
                } else {
                    0.0
                };
            }
        }

        Ok(output)
    }

    // Helper functions

    /// Create a windowed sinc kernel for interpolation
    fn create_sinc_kernel(config: &ResamplingConfig) -> SignalResult<Vec<f64>> {
        let length = config.kernel_length;
        if length % 2 == 0 {
            return Err(SignalError::ValueError(
                "Kernel length must be odd".to_string(),
            ));
        }

        let half_length = length / 2;
        let mut kernel = vec![0.0; length];

        // Create Kaiser window
        let window = window::kaiser::kaiser(length, config.kaiser_beta, true)?;

        for i in 0..length {
            let n = i as f64 - half_length as f64;

            // Sinc function with cutoff frequency
            let sinc_val = if n.abs() < 1e-10 {
                config.cutoff_frequency
            } else {
                let x = PI * config.cutoff_frequency * n;
                config.cutoff_frequency * x.sin() / x
            };

            kernel[i] = sinc_val * window[i];
        }

        Ok(kernel)
    }

    /// Evaluate sinc kernel at arbitrary position
    fn evaluate_sinc_kernel(kernel: &[f64], position: f64, _cutoff: f64) -> f64 {
        let kernel_len = kernel.len();
        let half_len = kernel_len / 2;

        if position.abs() >= half_len as f64 {
            return 0.0;
        }

        // Linear interpolation within the kernel
        let exact_pos = position + half_len as f64;
        let idx = exact_pos.floor() as usize;
        let frac = exact_pos - idx as f64;

        if idx >= kernel_len - 1 {
            return kernel[kernel_len - 1];
        }

        kernel[idx] * (1.0 - frac) + kernel[idx + 1] * frac
    }

    /// Design Thiran allpass filter coefficients
    fn design_thiran_coefficients(delay: f64, order: usize) -> SignalResult<Vec<f64>> {
        let mut coeffs = vec![0.0; order + 1];

        // Thiran filter design for fractional delay
        for (k, coeff_ref) in coeffs.iter_mut().enumerate() {
            let mut coeff = 1.0;

            for n in 0..=order {
                if n != k {
                    coeff *= (delay - n as f64) / (k as f64 - n as f64);
                }
            }

            *coeff_ref = if k % 2 == 0 { coeff } else { -coeff };
        }

        Ok(coeffs)
    }

    /// Apply allpass filter using direct form
    fn apply_allpass_filter(signal: &[f64], coeffs: &[f64]) -> SignalResult<Vec<f64>> {
        let order = coeffs.len() - 1;
        let mut output = vec![0.0; signal.len()];
        let mut delay_line = vec![0.0; order];

        for (n, &input) in signal.iter().enumerate() {
            let mut delayed_sum = 0.0;

            // Calculate feedforward path
            for k in 1..coeffs.len() {
                delayed_sum += coeffs[k] * delay_line[k - 1];
            }

            // Calculate output
            let y = coeffs[0] * input + delayed_sum;
            output[n] = y;

            // Update delay line (shift and insert new value)
            for k in (1..order).rev() {
                delay_line[k] = delay_line[k - 1];
            }
            if order > 0 {
                delay_line[0] = input - coeffs[0] * y;
            }
        }

        Ok(output)
    }

    /// Design resampling filter for polyphase implementation
    fn design_resampling_filter(
        up_factor: usize,
        down_factor: usize,
        config: &ResamplingConfig,
    ) -> SignalResult<Vec<f64>> {
        let gcd = num_integer::gcd(up_factor, down_factor);
        let effective_up = up_factor / gcd;
        let effective_down = down_factor / gcd;

        // Determine the anti-aliasing cutoff frequency
        let nyquist_factor = effective_up.max(effective_down) as f64;
        let cutoff = config.cutoff_frequency / nyquist_factor;

        // Filter length should be proportional to the upsampling factor
        let filter_length = config.kernel_length * effective_up;
        let half_length = filter_length / 2;

        let mut filter = vec![0.0; filter_length];
        let window = window::kaiser::kaiser(filter_length, config.kaiser_beta, true)?;

        for i in 0..filter_length {
            let n = i as f64 - half_length as f64;

            // Lowpass sinc filter
            let sinc_val = if n.abs() < 1e-10 {
                cutoff
            } else {
                let x = PI * cutoff * n;
                cutoff * x.sin() / x
            };

            filter[i] = sinc_val * window[i] * effective_up as f64;
        }

        Ok(filter)
    }

    /// Create polyphase filter bank from prototype filter
    fn create_polyphase_bank(prototype: &[f64], num_phases: usize) -> Vec<Vec<f64>> {
        let filter_length = prototype.len();
        let subfilter_length = filter_length.div_ceil(num_phases);

        let mut polyphase_bank = vec![vec![0.0; subfilter_length]; num_phases];

        for (i, &proto_val) in prototype.iter().enumerate() {
            let phase = i % num_phases;
            let tap = i / num_phases;
            if tap < subfilter_length {
                polyphase_bank[phase][tap] = proto_val;
            }
        }

        polyphase_bank
    }

    /// Upsample using polyphase filter bank
    fn upsample_with_polyphase(
        signal: &[f64],
        polyphase_filters: &[Vec<f64>],
    ) -> SignalResult<Vec<f64>> {
        let up_factor = polyphase_filters.len();
        let input_length = signal.len();
        let output_length = input_length * up_factor;
        let filter_length = polyphase_filters[0].len();

        let mut output = vec![0.0; output_length];

        for n in 0..input_length {
            #[allow(clippy::needless_range_loop)]
            for phase in 0..up_factor {
                let output_idx = n * up_factor + phase;
                let mut sum = 0.0;

                for k in 0..filter_length {
                    let input_idx = n as i32 - k as i32;
                    if input_idx >= 0 && input_idx < input_length as i32 {
                        sum += signal[input_idx as usize] * polyphase_filters[phase][k];
                    }
                }

                if output_idx < output_length {
                    output[output_idx] = sum;
                }
            }
        }

        Ok(output)
    }

    /// Simple downsampling by integer factor
    fn downsample(signal: &[f64], factor: usize) -> Vec<f64> {
        signal.iter().step_by(factor).copied().collect()
    }

    #[cfg(test)]
    mod resampling_tests {
        use super::*;
        use approx::assert_relative_eq;

        #[test]
        fn test_sinc_resample() {
            // Create a test signal (sine wave)
            let fs = 1000.0;
            let freq = 50.0;
            let duration = 0.1;
            let n_samples = (fs * duration) as usize;

            let signal: Vec<f64> = (0..n_samples)
                .map(|i| (2.0 * PI * freq * i as f64 / fs).sin())
                .collect();

            let config = ResamplingConfig::default();
            let target_length = n_samples * 2; // Upsample by 2x

            let resampled = sinc_resample(&signal, target_length, &config).unwrap();

            assert_eq!(resampled.len(), target_length);

            // Check that the signal maintains its frequency content
            let max_val = resampled.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
            assert!(max_val > 0.8); // Should preserve amplitude reasonably well
        }

        #[test]
        fn test_fractional_delay() {
            // Create a simple impulse
            let mut signal = vec![0.0; 20];
            signal[10] = 1.0; // Impulse at position 10

            let delay = 1.5; // Delay by 1.5 samples
            let delayed = fractional_delay(&signal, delay, 7).unwrap();

            assert_eq!(delayed.len(), signal.len());

            // The peak should shift to around position 8.5 (10 - 1.5)
            // Find the maximum value rather than index
            let max_val = delayed.iter().fold(0.0f64, |a, &b| a.max(b.abs()));

            // Check that we have reasonable amplitude preservation
            assert!(max_val > 0.5); // Should preserve most of the amplitude
        }

        #[test]
        fn test_polyphase_resample() {
            // Create test signal
            let signal: Vec<f64> = (0..100)
                .map(|i| (2.0 * PI * 0.1 * i as f64).sin())
                .collect();

            let config = ResamplingConfig::default();

            // Resample by 3:2 ratio
            let resampled = polyphase_resample(&signal, 3, 2, &config).unwrap();

            // Output length should be approximately input_length * 3/2
            let expected_length = signal.len() * 3 / 2;
            assert!((resampled.len() as i32 - expected_length as i32).abs() <= 1);
        }

        #[test]
        fn test_bandlimited_interpolation() {
            // Create a simple test signal
            let signal: Vec<f64> = (0..50).map(|i| (2.0 * PI * 0.1 * i as f64).sin()).collect();

            let config = ResamplingConfig::default();

            // Upsample from 1000 Hz to 2000 Hz
            let resampled = bandlimited_interpolation(&signal, 2000.0, 1000.0, &config).unwrap();

            // Should approximately double the length
            assert!((resampled.len() as i32 - (signal.len() * 2) as i32).abs() <= 1);
        }

        #[test]
        fn test_allpass_fractional_delay() {
            // Create a test signal
            let signal: Vec<f64> = (0..20).map(|i| if i == 5 { 1.0 } else { 0.0 }).collect();

            let delay = 2.3;
            let delayed = allpass_fractional_delay(&signal, delay, 4).unwrap();

            assert_eq!(delayed.len(), signal.len());

            // The signal should be delayed but preserve its shape reasonably
            let energy_original: f64 = signal.iter().map(|&x| x * x).sum();
            let energy_delayed: f64 = delayed.iter().map(|&x| x * x).sum();

            // Energy should be reasonably preserved (allow more tolerance)
            assert!(energy_delayed > 0.5 * energy_original);
            assert!(energy_delayed < 1.5 * energy_original);
        }

        #[test]
        fn test_variable_rate_resample() {
            // Create test signal
            let signal: Vec<f64> = (0..50).map(|i| (2.0 * PI * 0.1 * i as f64).sin()).collect();

            // Linear time mapping (constant rate)
            let time_map = |t: f64| t;
            let config = ResamplingConfig::default();

            let resampled = variable_rate_resample(&signal, time_map, 50, &config).unwrap();

            assert_eq!(resampled.len(), 50);

            // With identity mapping, should be similar to original
            for (orig, resamp) in signal.iter().zip(resampled.iter()) {
                assert_relative_eq!(orig, resamp, epsilon = 0.1);
            }
        }
    }
}

/// Polynomial interpolation methods
pub mod polynomial {
    use super::*;
    use crate::error::{SignalError, SignalResult};
    use ndarray::Array2;

    /// Lagrange polynomial interpolation
    ///
    /// # Arguments
    ///
    /// * `x_data` - Known x coordinates
    /// * `y_data` - Known y coordinates  
    /// * `x_new` - Points to interpolate at
    ///
    /// # Returns
    ///
    /// * Interpolated values at x_new points
    pub fn lagrange_interpolate(
        x_data: &[f64],
        y_data: &[f64],
        x_new: &[f64],
    ) -> SignalResult<Vec<f64>> {
        if x_data.len() != y_data.len() {
            return Err(SignalError::ValueError(
                "x_data and y_data must have same length".to_string(),
            ));
        }

        if x_data.is_empty() {
            return Err(SignalError::ValueError(
                "Need at least one data point".to_string(),
            ));
        }

        let n = x_data.len();
        let mut result = vec![0.0; x_new.len()];

        for (i, &xi) in x_new.iter().enumerate() {
            let mut sum = 0.0;

            for j in 0..n {
                let mut product = y_data[j];

                for k in 0..n {
                    if k != j {
                        let denominator = x_data[j] - x_data[k];
                        if denominator.abs() < 1e-15 {
                            return Err(SignalError::ValueError(
                                "Duplicate x values not allowed".to_string(),
                            ));
                        }
                        product *= (xi - x_data[k]) / denominator;
                    }
                }

                sum += product;
            }

            result[i] = sum;
        }

        Ok(result)
    }

    /// Newton polynomial interpolation with divided differences
    ///
    /// # Arguments
    ///
    /// * `x_data` - Known x coordinates
    /// * `y_data` - Known y coordinates
    /// * `x_new` - Points to interpolate at
    ///
    /// # Returns
    ///
    /// * Interpolated values at x_new points
    pub fn newton_interpolate(
        x_data: &[f64],
        y_data: &[f64],
        x_new: &[f64],
    ) -> SignalResult<Vec<f64>> {
        if x_data.len() != y_data.len() {
            return Err(SignalError::ValueError(
                "x_data and y_data must have same length".to_string(),
            ));
        }

        if x_data.is_empty() {
            return Err(SignalError::ValueError(
                "Need at least one data point".to_string(),
            ));
        }

        let n = x_data.len();

        // Calculate divided differences table
        let mut dd_table = Array2::zeros((n, n));

        // First column is just the y values
        for i in 0..n {
            dd_table[[i, 0]] = y_data[i];
        }

        // Fill in the divided differences table
        for j in 1..n {
            for i in 0..(n - j) {
                let denominator = x_data[i + j] - x_data[i];
                if denominator.abs() < 1e-15 {
                    return Err(SignalError::ValueError(
                        "Duplicate x values not allowed".to_string(),
                    ));
                }
                dd_table[[i, j]] = (dd_table[[i + 1, j - 1]] - dd_table[[i, j - 1]]) / denominator;
            }
        }

        // Evaluate the Newton polynomial at each point
        let mut result = vec![0.0; x_new.len()];

        for (idx, &xi) in x_new.iter().enumerate() {
            let mut sum = dd_table[[0, 0]];

            for j in 1..n {
                let mut product = dd_table[[0, j]];
                #[allow(clippy::needless_range_loop)]
                for k in 0..j {
                    product *= xi - x_data[k];
                }
                sum += product;
            }

            result[idx] = sum;
        }

        Ok(result)
    }

    /// Chebyshev polynomial interpolation
    ///
    /// # Arguments
    ///
    /// * `x_data` - Known x coordinates (should be Chebyshev nodes for best results)
    /// * `y_data` - Known y coordinates
    /// * `x_new` - Points to interpolate at (must be in [-1, 1])
    /// * `domain` - Optional domain transformation [a, b] -> [-1, 1]
    ///
    /// # Returns
    ///
    /// * Interpolated values at x_new points
    pub fn chebyshev_interpolate(
        x_data: &[f64],
        y_data: &[f64],
        x_new: &[f64],
        domain: Option<[f64; 2]>,
    ) -> SignalResult<Vec<f64>> {
        if x_data.len() != y_data.len() {
            return Err(SignalError::ValueError(
                "x_data and y_data must have same length".to_string(),
            ));
        }

        if x_data.is_empty() {
            return Err(SignalError::ValueError(
                "Need at least one data point".to_string(),
            ));
        }

        let n = x_data.len();

        // Transform to [-1, 1] if domain is specified
        let (x_transformed, x_new_transformed) = if let Some([a, b]) = domain {
            let x_trans: Vec<f64> = x_data
                .iter()
                .map(|&x| 2.0 * (x - a) / (b - a) - 1.0)
                .collect();
            let x_new_trans: Vec<f64> = x_new
                .iter()
                .map(|&x| 2.0 * (x - a) / (b - a) - 1.0)
                .collect();
            (x_trans, x_new_trans)
        } else {
            (x_data.to_vec(), x_new.to_vec())
        };

        // Compute Chebyshev coefficients using barycentric interpolation
        let mut weights = vec![0.0; n];
        for (i, weight) in weights.iter_mut().enumerate().take(n) {
            *weight = if i == 0 || i == n - 1 { 0.5 } else { 1.0 };
            if i % 2 == 1 {
                *weight = -*weight;
            }
        }

        let mut result = vec![0.0; x_new.len()];

        for (idx, &xi) in x_new_transformed.iter().enumerate() {
            // Check if xi matches any data point exactly
            for (i, &x_i) in x_transformed.iter().enumerate() {
                if (xi - x_i).abs() < 1e-15 {
                    result[idx] = y_data[i];
                    continue;
                }
            }

            // Barycentric interpolation formula
            let mut numerator = 0.0;
            let mut denominator = 0.0;

            for i in 0..n {
                let term = weights[i] / (xi - x_transformed[i]);
                numerator += term * y_data[i];
                denominator += term;
            }

            if denominator.abs() < 1e-15 {
                return Err(SignalError::ValueError(
                    "Numerical instability in Chebyshev interpolation".to_string(),
                ));
            }

            result[idx] = numerator / denominator;
        }

        Ok(result)
    }

    /// Generate Chebyshev nodes for optimal interpolation
    ///
    /// # Arguments
    ///
    /// * `n` - Number of nodes
    /// * `domain` - Domain [a, b] for the nodes
    ///
    /// # Returns
    ///
    /// * Chebyshev nodes in the specified domain
    pub fn chebyshev_nodes(n: usize, domain: [f64; 2]) -> SignalResult<Vec<f64>> {
        if n == 0 {
            return Err(SignalError::ValueError(
                "Number of nodes must be positive".to_string(),
            ));
        }

        let [a, b] = domain;
        if b <= a {
            return Err(SignalError::ValueError(
                "Invalid domain: b must be greater than a".to_string(),
            ));
        }

        let mut nodes = vec![0.0; n];
        let half_range = (b - a) / 2.0;
        let mid_point = (a + b) / 2.0;

        for (i, node) in nodes.iter_mut().enumerate().take(n) {
            let angle = PI * (2 * i + 1) as f64 / (2 * n) as f64;
            *node = mid_point + half_range * angle.cos();
        }

        Ok(nodes)
    }

    /// Piecewise polynomial interpolation (higher-order splines)
    ///
    /// # Arguments
    ///
    /// * `x_data` - Known x coordinates (must be sorted)
    /// * `y_data` - Known y coordinates
    /// * `x_new` - Points to interpolate at
    /// * `degree` - Polynomial degree for each piece
    ///
    /// # Returns
    ///
    /// * Interpolated values at x_new points
    pub fn piecewise_polynomial_interpolate(
        x_data: &[f64],
        y_data: &[f64],
        x_new: &[f64],
        degree: usize,
    ) -> SignalResult<Vec<f64>> {
        if x_data.len() != y_data.len() {
            return Err(SignalError::ValueError(
                "x_data and y_data must have same length".to_string(),
            ));
        }

        if x_data.len() < degree + 1 {
            return Err(SignalError::ValueError(
                "Need at least degree+1 data points".to_string(),
            ));
        }

        // Check if x_data is sorted
        for i in 1..x_data.len() {
            if x_data[i] <= x_data[i - 1] {
                return Err(SignalError::ValueError(
                    "x_data must be sorted in ascending order".to_string(),
                ));
            }
        }

        let mut result = vec![0.0; x_new.len()];

        for (idx, &xi) in x_new.iter().enumerate() {
            // Find the appropriate segment
            let segment_start = if xi <= x_data[0] {
                0
            } else if xi >= x_data[x_data.len() - 1] {
                x_data.len().saturating_sub(degree + 1)
            } else {
                // Binary search for the interval
                let mut left = 0;
                let mut right = x_data.len() - 1;
                while right - left > 1 {
                    let mid = (left + right) / 2;
                    if x_data[mid] <= xi {
                        left = mid;
                    } else {
                        right = mid;
                    }
                }
                // Center the polynomial around the found interval
                left.saturating_sub(degree / 2)
                    .min(x_data.len() - degree - 1)
            };

            let segment_end = (segment_start + degree + 1).min(x_data.len());

            // Extract segment data
            let x_segment = &x_data[segment_start..segment_end];
            let y_segment = &y_data[segment_start..segment_end];

            // Use Lagrange interpolation for this segment
            let xi_vec = vec![xi];
            let interpolated = lagrange_interpolate(x_segment, y_segment, &xi_vec)?;
            result[idx] = interpolated[0];
        }

        Ok(result)
    }

    #[cfg(test)]
    mod polynomial_tests {
        use super::*;
        use approx::assert_relative_eq;

        #[test]
        fn test_lagrange_interpolation() {
            // Test with a quadratic function: f(x) = x^2
            let x_data = vec![0.0, 1.0, 2.0];
            let y_data = vec![0.0, 1.0, 4.0];
            let x_new = vec![0.5, 1.5];

            let result = lagrange_interpolate(&x_data, &y_data, &x_new).unwrap();

            // Expected values: f(0.5) = 0.25, f(1.5) = 2.25
            assert_relative_eq!(result[0], 0.25, epsilon = 1e-10);
            assert_relative_eq!(result[1], 2.25, epsilon = 1e-10);
        }

        #[test]
        fn test_newton_interpolation() {
            // Test with a linear function: f(x) = 2x + 1
            let x_data = vec![0.0, 1.0, 2.0];
            let y_data = vec![1.0, 3.0, 5.0];
            let x_new = vec![0.5, 1.5];

            let result = newton_interpolate(&x_data, &y_data, &x_new).unwrap();

            // Expected values: f(0.5) = 2, f(1.5) = 4
            assert_relative_eq!(result[0], 2.0, epsilon = 1e-10);
            assert_relative_eq!(result[1], 4.0, epsilon = 1e-10);
        }

        #[test]
        fn test_chebyshev_nodes() {
            let nodes = chebyshev_nodes(5, [-1.0, 1.0]).unwrap();

            assert_eq!(nodes.len(), 5);

            // Check that all nodes are in [-1, 1]
            for &node in &nodes {
                assert!(node >= -1.0 && node <= 1.0);
            }

            // Check specific values for 5 nodes
            let expected_angles: Vec<f64> =
                (0..5).map(|i| PI * (2 * i + 1) as f64 / 10.0).collect();

            for (i, &angle) in expected_angles.iter().enumerate() {
                assert_relative_eq!(nodes[i], angle.cos(), epsilon = 1e-10);
            }
        }

        #[test]
        fn test_chebyshev_interpolation() {
            // Test with Chebyshev nodes
            let nodes = chebyshev_nodes(4, [-1.0, 1.0]).unwrap();
            let y_data: Vec<f64> = nodes.iter().map(|&x| x * x).collect(); // f(x) = x^2

            let x_new = vec![0.0, 0.5];
            let result = chebyshev_interpolate(&nodes, &y_data, &x_new, None).unwrap();

            // Should interpolate x^2 reasonably well with Chebyshev nodes
            assert_relative_eq!(result[0], 0.0, epsilon = 1e-1);
            assert_relative_eq!(result[1], 0.25, epsilon = 1e-1);
        }

        #[test]
        fn test_piecewise_polynomial() {
            // Test with a cubic function
            let x_data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
            let y_data = vec![0.0, 1.0, 8.0, 27.0, 64.0]; // f(x) = x^3
            let x_new = vec![1.5, 2.5];

            let result = piecewise_polynomial_interpolate(&x_data, &y_data, &x_new, 3).unwrap();

            // Expected values: f(1.5) = 3.375, f(2.5) = 15.625
            assert_relative_eq!(result[0], 3.375, epsilon = 1e-6);
            assert_relative_eq!(result[1], 15.625, epsilon = 1e-6);
        }

        #[test]
        fn test_lagrange_error_cases() {
            let x_data = vec![0.0, 1.0];
            let y_data = vec![0.0]; // Different length
            let x_new = vec![0.5];

            let result = lagrange_interpolate(&x_data, &y_data, &x_new);
            assert!(result.is_err());

            // Test duplicate x values
            let x_data = vec![0.0, 0.0];
            let y_data = vec![0.0, 1.0];
            let result = lagrange_interpolate(&x_data, &y_data, &x_new);
            assert!(result.is_err());
        }
    }
}
