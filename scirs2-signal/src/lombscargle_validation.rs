//! Validation utilities for Lomb-Scargle periodogram
//!
//! This module provides comprehensive validation functions for Lomb-Scargle
//! implementations, including numerical stability checks, comparison with
//! reference implementations, and edge case handling.

use crate::error::{SignalError, SignalResult};
use crate::lombscargle::{lombscargle, AutoFreqMethod};
use crate::lombscargle_enhanced::{enhanced_lombscargle, LombScargleConfig, WindowType};
use ndarray::{Array1, ArrayView1};
use num_traits::{Float, NumCast};
use scirs2_core::validation::{check_finite, check_positive};
use std::f64::consts::PI;

/// Validation result structure
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Maximum relative error compared to reference
    pub max_relative_error: f64,
    /// Mean relative error
    pub mean_relative_error: f64,
    /// Numerical stability score (0-1, higher is better)
    pub stability_score: f64,
    /// Peak frequency accuracy
    pub peak_freq_error: f64,
    /// Issues found during validation
    pub issues: Vec<String>,
}

/// Validate Lomb-Scargle implementation against known analytical cases
///
/// # Arguments
///
/// * `implementation` - Name of implementation to test
/// * `tolerance` - Tolerance for numerical comparison
///
/// # Returns
///
/// * Validation result with detailed metrics
pub fn validate_analytical_cases(
    implementation: &str,
    tolerance: f64,
) -> SignalResult<ValidationResult> {
    let mut issues = Vec::new();
    let mut errors = Vec::new();
    let mut peak_errors = Vec::new();

    // Test case 1: Pure sinusoid (should have exact peak at frequency)
    let n = 1000;
    let fs = 100.0;
    let f_signal = 10.0;
    let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
    let signal: Vec<f64> = t
        .iter()
        .map(|&ti| (2.0 * PI * f_signal * ti).sin())
        .collect();

    // Compute periodogram
    let (freqs, power) = match implementation {
        "standard" => lombscargle(
            &t,
            &signal,
            None,
            Some("standard"),
            Some(true),
            Some(false),
            None,
            None,
        )?,
        "enhanced" => {
            let config = LombScargleConfig {
                window: WindowType::None,
                custom_window: None,
                oversample: 5.0,
                f_min: Some(1.0),
                f_max: Some(50.0),
                bootstrap_iter: None,
                confidence: None,
                tolerance: 1e-10,
                use_fast: true,
            };
            let (f, p, _) = enhanced_lombscargle(&t, &signal, &config)?;
            (f, p)
        }
        _ => {
            return Err(SignalError::ValueError(
                "Unknown implementation".to_string(),
            ))
        }
    };

    // Find peak
    let (peak_idx, &peak_power) = power
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    let peak_freq = freqs[peak_idx];

    // Check peak frequency accuracy
    let freq_error = (peak_freq - f_signal).abs() / f_signal;
    peak_errors.push(freq_error);

    if freq_error > tolerance {
        issues.push(format!(
            "Pure sinusoid: Peak frequency error {:.6} exceeds tolerance",
            freq_error
        ));
    }

    // Test case 2: Two sinusoids (should have two distinct peaks)
    let f_signal2 = 25.0;
    let signal2: Vec<f64> = t
        .iter()
        .map(|&ti| (2.0 * PI * f_signal * ti).sin() + 0.5 * (2.0 * PI * f_signal2 * ti).sin())
        .collect();

    let (freqs2, power2) = match implementation {
        "standard" => lombscargle(
            &t,
            &signal2,
            None,
            Some("standard"),
            Some(true),
            Some(false),
            None,
            None,
        )?,
        "enhanced" => {
            let config = LombScargleConfig {
                window: WindowType::None,
                custom_window: None,
                oversample: 5.0,
                f_min: Some(1.0),
                f_max: Some(50.0),
                bootstrap_iter: None,
                confidence: None,
                tolerance: 1e-10,
                use_fast: true,
            };
            let (f, p, _) = enhanced_lombscargle(&t, &signal2, &config)?;
            (f, p)
        }
        _ => {
            return Err(SignalError::ValueError(
                "Unknown implementation".to_string(),
            ))
        }
    };

    // Find two highest peaks
    let mut power_sorted: Vec<(usize, f64)> =
        power2.iter().enumerate().map(|(i, &p)| (i, p)).collect();
    power_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let peak1_freq = freqs2[power_sorted[0].0];
    let peak2_freq = freqs2[power_sorted[1].0];

    // Check if both frequencies are detected
    let freq1_found = (peak1_freq - f_signal).abs() < 1.0 || (peak2_freq - f_signal).abs() < 1.0;
    let freq2_found = (peak1_freq - f_signal2).abs() < 1.0 || (peak2_freq - f_signal2).abs() < 1.0;

    if !freq1_found || !freq2_found {
        issues.push("Two sinusoids: Failed to detect both frequency components".to_string());
    }

    // Test case 3: DC offset (should not affect spectrum except at f=0)
    let dc_offset = 5.0;
    let signal_dc: Vec<f64> = signal.iter().map(|&s| s + dc_offset).collect();

    let (_, power_dc) = match implementation {
        "standard" => lombscargle(
            &t,
            &signal_dc,
            None,
            Some("standard"),
            Some(true),
            Some(false),
            None,
            None,
        )?,
        "enhanced" => {
            let config = LombScargleConfig {
                window: WindowType::None,
                custom_window: None,
                oversample: 5.0,
                f_min: Some(1.0),
                f_max: Some(50.0),
                bootstrap_iter: None,
                confidence: None,
                tolerance: 1e-10,
                use_fast: true,
            };
            let (f, p, _) = enhanced_lombscargle(&t, &signal_dc, &config)?;
            (f, p)
        }
        _ => {
            return Err(SignalError::ValueError(
                "Unknown implementation".to_string(),
            ))
        }
    };

    // Compare spectra (should be identical except near DC)
    for i in 1..power.len().min(power_dc.len()) {
        if freqs[i] > 1.0 {
            let rel_error = (power[i] - power_dc[i]).abs() / power[i].max(1e-10);
            errors.push(rel_error);
            if rel_error > tolerance * 10.0 {
                issues.push(format!(
                    "DC offset test: Spectrum affected at f={:.2} Hz",
                    freqs[i]
                ));
            }
        }
    }

    // Calculate validation metrics
    let max_relative_error = errors.iter().cloned().fold(0.0, f64::max);
    let mean_relative_error = if errors.is_empty() {
        0.0
    } else {
        errors.iter().sum::<f64>() / errors.len() as f64
    };

    let peak_freq_error = peak_errors.iter().cloned().fold(0.0, f64::max);

    // Stability score based on number of issues
    let stability_score = 1.0 - (issues.len() as f64 / 10.0).min(1.0);

    Ok(ValidationResult {
        max_relative_error,
        mean_relative_error,
        stability_score,
        peak_freq_error,
        issues,
    })
}

/// Validate numerical stability with edge cases
///
/// # Arguments
///
/// * `implementation` - Implementation to test
///
/// # Returns
///
/// * Validation result
pub fn validate_numerical_stability(implementation: &str) -> SignalResult<ValidationResult> {
    let mut issues = Vec::new();
    let mut stability_tests_passed = 0;
    let total_tests = 5;

    // Test 1: Very small signal values
    let t_small: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let signal_small: Vec<f64> = t_small.iter().map(|&ti| 1e-10 * ti.sin()).collect();

    match compute_test_periodogram(&t_small, &signal_small, implementation) {
        Ok((_, power)) => {
            if power.iter().all(|&p| p.is_finite()) {
                stability_tests_passed += 1;
            } else {
                issues.push("Failed with very small signal values".to_string());
            }
        }
        Err(_) => issues.push("Error with very small signal values".to_string()),
    }

    // Test 2: Very large signal values
    let signal_large: Vec<f64> = t_small.iter().map(|&ti| 1e10 * ti.sin()).collect();

    match compute_test_periodogram(&t_small, &signal_large, implementation) {
        Ok((_, power)) => {
            if power.iter().all(|&p| p.is_finite()) {
                stability_tests_passed += 1;
            } else {
                issues.push("Failed with very large signal values".to_string());
            }
        }
        Err(_) => issues.push("Error with very large signal values".to_string()),
    }

    // Test 3: Highly irregular sampling
    let mut t_irregular = vec![0.0];
    let mut rng = rand::rng();
    for i in 1..50 {
        t_irregular.push(t_irregular[i - 1] + rng.random_range(0.1..10.0));
    }
    let signal_irregular: Vec<f64> = t_irregular.iter().map(|&ti| ti.sin()).collect();

    match compute_test_periodogram(&t_irregular, &signal_irregular, implementation) {
        Ok((_, power)) => {
            if power.iter().all(|&p| p.is_finite() && p >= 0.0) {
                stability_tests_passed += 1;
            } else {
                issues.push("Failed with irregular sampling".to_string());
            }
        }
        Err(_) => issues.push("Error with irregular sampling".to_string()),
    }

    // Test 4: Nearly constant signal
    let signal_const: Vec<f64> = vec![1.0 + 1e-15; 100];
    let t_const: Vec<f64> = (0..100).map(|i| i as f64).collect();

    match compute_test_periodogram(&t_const, &signal_const, implementation) {
        Ok((_, power)) => {
            if power.iter().all(|&p| p.is_finite()) {
                stability_tests_passed += 1;
            } else {
                issues.push("Failed with nearly constant signal".to_string());
            }
        }
        Err(_) => issues.push("Error with nearly constant signal".to_string()),
    }

    // Test 5: Signal with outliers
    let mut signal_outliers: Vec<f64> = t_small.iter().map(|&ti| ti.sin()).collect();
    signal_outliers[25] = 1000.0; // Add outlier
    signal_outliers[75] = -1000.0; // Add outlier

    match compute_test_periodogram(&t_small, &signal_outliers, implementation) {
        Ok((_, power)) => {
            if power.iter().all(|&p| p.is_finite()) {
                stability_tests_passed += 1;
            } else {
                issues.push("Failed with outliers".to_string());
            }
        }
        Err(_) => issues.push("Error with outliers".to_string()),
    }

    let stability_score = stability_tests_passed as f64 / total_tests as f64;

    Ok(ValidationResult {
        max_relative_error: 0.0,
        mean_relative_error: 0.0,
        stability_score,
        peak_freq_error: 0.0,
        issues,
    })
}

/// Helper function to compute periodogram for testing
fn compute_test_periodogram(
    times: &[f64],
    values: &[f64],
    implementation: &str,
) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    match implementation {
        "standard" => lombscargle(
            times,
            values,
            None,
            Some("standard"),
            Some(true),
            Some(false),
            None,
            None,
        ),
        "enhanced" => {
            let config = LombScargleConfig {
                window: WindowType::None,
                custom_window: None,
                oversample: 3.0,
                f_min: None,
                f_max: None,
                bootstrap_iter: None,
                confidence: None,
                tolerance: 1e-10,
                use_fast: true,
            };
            let (f, p, _) = enhanced_lombscargle(times, values, &config)?;
            Ok((f, p))
        }
        _ => Err(SignalError::ValueError(
            "Unknown implementation".to_string(),
        )),
    }
}

/// Compare implementations for consistency
///
/// # Arguments
///
/// * `times` - Sample times
/// * `values` - Signal values
/// * `tolerance` - Tolerance for comparison
///
/// # Returns
///
/// * Maximum relative difference between implementations
pub fn compare_implementations<T, U>(times: &[T], values: &[U], tolerance: f64) -> SignalResult<f64>
where
    T: Float + NumCast,
    U: Float + NumCast,
{
    // Convert to f64
    let times_f64: Vec<f64> = times.iter().map(|&t| NumCast::from(t).unwrap()).collect();
    let values_f64: Vec<f64> = values.iter().map(|&v| NumCast::from(v).unwrap()).collect();

    // Compute with standard implementation
    let (freqs1, power1) = lombscargle(
        &times_f64,
        &values_f64,
        None,
        Some("standard"),
        Some(true),
        Some(false),
        None,
        None,
    )?;

    // Compute with enhanced implementation
    let config = LombScargleConfig {
        window: WindowType::None,
        custom_window: None,
        oversample: 3.0,
        f_min: None,
        f_max: None,
        bootstrap_iter: None,
        confidence: None,
        tolerance: 1e-10,
        use_fast: false, // Use standard algorithm for comparison
    };
    let (freqs2, power2, _) = enhanced_lombscargle(&times_f64, &values_f64, &config)?;

    // Find common frequency range
    let f_min = freqs1[0].max(freqs2[0]);
    let f_max = freqs1[freqs1.len() - 1].min(freqs2[freqs2.len() - 1]);

    // Interpolate to common grid for comparison
    let mut max_diff = 0.0;

    for i in 0..freqs1.len() {
        if freqs1[i] >= f_min && freqs1[i] <= f_max {
            // Find nearest frequency in freqs2
            let j = freqs2
                .iter()
                .position(|&f| (f - freqs1[i]).abs() < tolerance)
                .unwrap_or(0);

            if j < power2.len() {
                let rel_diff = (power1[i] - power2[j]).abs() / power1[i].max(1e-10);
                max_diff = max_diff.max(rel_diff);
            }
        }
    }

    Ok(max_diff)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_analytical() {
        let result = validate_analytical_cases("standard", 0.01).unwrap();
        assert!(result.peak_freq_error < 0.01);
        assert!(result.stability_score > 0.8);
    }

    #[test]
    fn test_numerical_stability() {
        let result = validate_numerical_stability("standard").unwrap();
        assert!(result.stability_score > 0.6);
    }
}
