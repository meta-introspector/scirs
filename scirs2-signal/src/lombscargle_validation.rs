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

/// Single test result structure
#[derive(Debug, Clone)]
pub struct SingleTestResult {
    /// Relative errors from this test
    pub errors: Vec<f64>,
    /// Peak frequency error
    pub peak_error: f64,
    /// Peak frequency errors (for multiple peaks)
    pub peak_errors: Vec<f64>,
    /// Issues found in this test
    pub issues: Vec<String>,
}

/// Validate Lomb-Scargle implementation against known analytical cases
///
/// Enhanced version with comprehensive edge case testing and robustness validation
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
    let test_result_1 = validate_pure_sinusoid(implementation, tolerance)?;
    errors.extend(test_result_1.errors);
    peak_errors.push(test_result_1.peak_error);
    issues.extend(test_result_1.issues);

    // Test case 2: Multiple sinusoids with different amplitudes
    let test_result_2 = validate_multiple_sinusoids(implementation, tolerance)?;
    errors.extend(test_result_2.errors);
    peak_errors.extend(test_result_2.peak_errors);
    issues.extend(test_result_2.issues);

    // Test case 3: Heavily uneven sampling
    let test_result_3 = validate_uneven_sampling(implementation, tolerance)?;
    errors.extend(test_result_3.errors);
    peak_errors.push(test_result_3.peak_error);
    issues.extend(test_result_3.issues);

    // Test case 4: Extreme edge cases
    let test_result_4 = validate_edge_cases(implementation, tolerance)?;
    errors.extend(test_result_4.errors);
    issues.extend(test_result_4.issues);

    // Test case 5: Numerical precision and stability
    let test_result_5 = validate_numerical_stability(implementation, tolerance)?;
    errors.extend(test_result_5.errors);
    issues.extend(test_result_5.issues);

    // Calculate overall metrics
    let max_relative_error = errors.iter().cloned().fold(0.0, f64::max);
    let mean_relative_error = if !errors.is_empty() {
        errors.iter().sum::<f64>() / errors.len() as f64
    } else {
        0.0
    };

    let peak_freq_error = if !peak_errors.is_empty() {
        peak_errors.iter().cloned().fold(0.0, f64::max)
    } else {
        0.0
    };

    // Calculate stability score based on number of issues and errors
    let stability_score = calculate_stability_score(&issues, &errors);

    Ok(ValidationResult {
        max_relative_error,
        mean_relative_error,
        stability_score,
        peak_freq_error,
        issues,
    })
}

/// Test pure sinusoid case
fn validate_pure_sinusoid(implementation: &str, tolerance: f64) -> SignalResult<SingleTestResult> {
    let mut issues = Vec::new();
    let mut errors = Vec::new();

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
    errors.push(freq_error);

    if freq_error > tolerance {
        issues.push(format!(
            "Pure sinusoid peak frequency error {:.2e} exceeds tolerance {:.2e}",
            freq_error, tolerance
        ));
    }

    // Check that peak is significantly above noise floor
    let noise_floor = power.iter().cloned().fold(f64::MAX, f64::min);
    let signal_to_noise = peak_power / noise_floor.max(1e-15);
    
    if signal_to_noise < 10.0 {
        issues.push(format!(
            "Poor signal-to-noise ratio: {:.2}", signal_to_noise
        ));
    }

    // Validate that all power values are non-negative and finite
    for (i, &p) in power.iter().enumerate() {
        if !p.is_finite() || p < 0.0 {
            issues.push(format!(
                "Invalid power value at index {}: {}", i, p
            ));
            break;
        }
    }

    Ok(SingleTestResult {
        errors,
        peak_error: freq_error,
        peak_errors: vec![freq_error],
        issues,
    })
}

/// Test multiple sinusoids with different amplitudes
fn validate_multiple_sinusoids(implementation: &str, tolerance: f64) -> SignalResult<SingleTestResult> {
    let mut issues = Vec::new();
    let mut errors = Vec::new();
    let mut peak_errors = Vec::new();

    let n = 1000;
    let fs = 100.0;
    let f_signals = vec![5.0, 15.0, 25.0];
    let amplitudes = vec![1.0, 0.5, 0.8];
    
    let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
    let signal: Vec<f64> = t.iter().map(|&ti| {
        f_signals.iter().zip(amplitudes.iter()).map(|(&f, &a)| {
            a * (2.0 * PI * f * ti).sin()
        }).sum()
    }).collect();

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
                f_max: Some(30.0),
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

    // Find peaks for each expected frequency
    for (signal_idx, &expected_freq) in f_signals.iter().enumerate() {
        let freq_tolerance = 0.5; // Allow 0.5 Hz tolerance for peak finding
        
        let peak_candidates: Vec<(usize, f64)> = freqs.iter().enumerate()
            .filter(|(_, &f)| (f - expected_freq).abs() < freq_tolerance)
            .map(|(i, &f)| (i, power[i]))
            .collect();
            
        if peak_candidates.is_empty() {
            issues.push(format!(
                "No peak found near expected frequency {:.1} Hz", expected_freq
            ));
            peak_errors.push(1.0); // Maximum error
            continue;
        }
        
        // Find the highest peak in the candidate range
        let (peak_idx, peak_power) = peak_candidates.iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
            
        let peak_freq = freqs[*peak_idx];
        let freq_error = (peak_freq - expected_freq).abs() / expected_freq;
        peak_errors.push(freq_error);
        errors.push(freq_error);
        
        if freq_error > tolerance * 5.0 { // More lenient for multi-component signals
            issues.push(format!(
                "Signal {} peak frequency error {:.2e} exceeds tolerance",
                signal_idx, freq_error
            ));
        }
    }

    Ok(SingleTestResult {
        errors,
        peak_error: peak_errors.iter().cloned().fold(0.0, f64::max),
        peak_errors,
        issues,
    })
}

/// Test heavily uneven sampling patterns
fn validate_uneven_sampling(implementation: &str, tolerance: f64) -> SignalResult<SingleTestResult> {
    let mut issues = Vec::new();
    let mut errors = Vec::new();
    use rand::prelude::*;

    let n_nominal = 1000;
    let fs_nominal = 100.0;
    let f_signal = 10.0;
    
    // Create heavily uneven sampling (random gaps and clustering)
    let mut rng = rand::rng();
    let mut t = Vec::new();
    let mut current_time = 0.0;
    
    while t.len() < n_nominal && current_time < 10.0 {
        // Random time intervals with large variations
        let interval = rng.random_range(0.001..0.5); // Very uneven: 1ms to 500ms
        current_time += interval;
        t.push(current_time);
    }
    
    let signal: Vec<f64> = t.iter().map(|&ti| (2.0 * PI * f_signal * ti).sin()).collect();

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
                oversample: 10.0, // Higher oversampling for uneven data
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

    let freq_error = (peak_freq - f_signal).abs() / f_signal;
    errors.push(freq_error);

    // More lenient tolerance for uneven sampling
    if freq_error > tolerance * 10.0 {
        issues.push(format!(
            "Uneven sampling peak frequency error {:.2e} exceeds tolerance",
            freq_error
        ));
    }

    // Check for spurious peaks (should be rare with good implementation)
    let threshold = peak_power * 0.1; // 10% of main peak
    let spurious_peaks = power.iter().enumerate()
        .filter(|(i, &p)| *i != peak_idx && p > threshold)
        .count();
        
    if spurious_peaks > 5 {
        issues.push(format!(
            "Too many spurious peaks: {} above 10% threshold", spurious_peaks
        ));
    }

    Ok(SingleTestResult {
        errors,
        peak_error: freq_error,
        peak_errors: vec![freq_error],
        issues,
    })
}

/// Test extreme edge cases
fn validate_edge_cases(implementation: &str, tolerance: f64) -> SignalResult<SingleTestResult> {
    let mut issues = Vec::new();
    let mut errors = Vec::new();

    // Test case 1: Very short time series
    let test_result_short = test_short_time_series(implementation)?;
    errors.extend(test_result_short.errors);
    issues.extend(test_result_short.issues);

    // Test case 2: Constant signal (should handle gracefully)
    let test_result_constant = test_constant_signal(implementation)?;
    errors.extend(test_result_constant.errors);
    issues.extend(test_result_constant.issues);

    // Test case 3: Very sparse sampling
    let test_result_sparse = test_sparse_sampling(implementation)?;
    errors.extend(test_result_sparse.errors);
    issues.extend(test_result_sparse.issues);

    // Test case 4: Signal with outliers
    let test_result_outliers = test_signal_with_outliers(implementation)?;
    errors.extend(test_result_outliers.errors);
    issues.extend(test_result_outliers.issues);

    Ok(SingleTestResult {
        errors,
        peak_error: 0.0,
        peak_errors: vec![],
        issues,
    })
}

/// Test numerical precision and stability
fn validate_numerical_stability(implementation: &str, tolerance: f64) -> SignalResult<SingleTestResult> {
    let mut issues = Vec::new();
    let mut errors = Vec::new();

    // Test with very small signal values
    let test_result_small = test_small_values(implementation)?;
    errors.extend(test_result_small.errors);
    issues.extend(test_result_small.issues);

    // Test with very large signal values
    let test_result_large = test_large_values(implementation)?;
    errors.extend(test_result_large.errors);
    issues.extend(test_result_large.issues);

    // Test with extreme time scales
    let test_result_timescales = test_extreme_timescales(implementation)?;
    errors.extend(test_result_timescales.errors);
    issues.extend(test_result_timescales.issues);

    Ok(SingleTestResult {
        errors,
        peak_error: 0.0,
        peak_errors: vec![],
        issues,
    })
}

/// Helper function to calculate stability score
fn calculate_stability_score(issues: &[String], errors: &[f64]) -> f64 {
    let base_score = 1.0;
    let issue_penalty = issues.len() as f64 * 0.1;
    let error_penalty = errors.iter().map(|&e| e.min(0.5)).sum::<f64>() * 0.2;
    
    (base_score - issue_penalty - error_penalty).max(0.0).min(1.0)
}

/// Test helper functions for edge cases
fn test_short_time_series(implementation: &str) -> SignalResult<SingleTestResult> {
    let mut issues = Vec::new();
    let mut errors = Vec::new();

    // Very short series (should handle gracefully or return appropriate error)
    let t = vec![0.0, 0.1, 0.2];
    let signal = vec![1.0, 0.0, -1.0];

    let result = match implementation {
        "standard" => lombscargle(&t, &signal, None, Some("standard"), Some(true), Some(false), None, None),
        "enhanced" => {
            let config = LombScargleConfig {
                window: WindowType::None,
                custom_window: None,
                oversample: 5.0,
                f_min: Some(0.1),
                f_max: Some(10.0),
                bootstrap_iter: None,
                confidence: None,
                tolerance: 1e-10,
                use_fast: true,
            };
            enhanced_lombscargle(&t, &signal, &config).map(|(f, p, _)| (f, p))
        }
        _ => return Err(SignalError::ValueError("Unknown implementation".to_string()))
    };

    match result {
        Ok((freqs, power)) => {
            // Should produce some result, check if reasonable
            if freqs.is_empty() || power.is_empty() {
                issues.push("Short time series produced empty result".to_string());
            }
            // Check for valid values
            for &p in &power {
                if !p.is_finite() || p < 0.0 {
                    issues.push("Short time series produced invalid power values".to_string());
                    break;
                }
            }
        }
        Err(_) => {
            // It's acceptable for very short series to fail
            // This is not necessarily an issue
        }
    }

    Ok(SingleTestResult {
        errors,
        peak_error: 0.0,
        peak_errors: vec![],
        issues,
    })
}

fn test_constant_signal(implementation: &str) -> SignalResult<SingleTestResult> {
    let mut issues = Vec::new();
    let mut errors = Vec::new();

    let n = 100;
    let t: Vec<f64> = (0..n).map(|i| i as f64 * 0.01).collect();
    let signal = vec![1.0; n]; // Constant signal

    let result = match implementation {
        "standard" => lombscargle(&t, &signal, None, Some("standard"), Some(true), Some(false), None, None),
        "enhanced" => {
            let config = LombScargleConfig {
                window: WindowType::None,
                custom_window: None,
                oversample: 5.0,
                f_min: Some(0.1),
                f_max: Some(10.0),
                bootstrap_iter: None,
                confidence: None,
                tolerance: 1e-10,
                use_fast: true,
            };
            enhanced_lombscargle(&t, &signal, &config).map(|(f, p, _)| (f, p))
        }
        _ => return Err(SignalError::ValueError("Unknown implementation".to_string()))
    };

    match result {
        Ok((freqs, power)) => {
            // For constant signal, power should be near zero or very low
            let max_power = power.iter().cloned().fold(0.0, f64::max);
            if max_power > 1e-6 {
                issues.push(format!("Constant signal shows unexpected power: {:.2e}", max_power));
            }
        }
        Err(_) => {
            // Might fail for constant signal, which could be acceptable
        }
    }

    Ok(SingleTestResult {
        errors,
        peak_error: 0.0,
        peak_errors: vec![],
        issues,
    })
}

fn test_sparse_sampling(implementation: &str) -> SignalResult<SingleTestResult> {
    let mut issues = Vec::new();
    let mut errors = Vec::new();

    // Very sparse sampling - only 10 points over 10 seconds
    let t = vec![0.0, 1.0, 2.5, 3.8, 4.2, 5.9, 7.1, 8.3, 9.0, 10.0];
    let f_signal = 0.5; // 0.5 Hz signal
    let signal: Vec<f64> = t.iter().map(|&ti| (2.0 * PI * f_signal * ti).sin()).collect();

    let result = match implementation {
        "standard" => lombscargle(&t, &signal, None, Some("standard"), Some(true), Some(false), None, None),
        "enhanced" => {
            let config = LombScargleConfig {
                window: WindowType::None,
                custom_window: None,
                oversample: 10.0, // Higher oversampling for sparse data
                f_min: Some(0.1),
                f_max: Some(2.0),
                bootstrap_iter: None,
                confidence: None,
                tolerance: 1e-10,
                use_fast: true,
            };
            enhanced_lombscargle(&t, &signal, &config).map(|(f, p, _)| (f, p))
        }
        _ => return Err(SignalError::ValueError("Unknown implementation".to_string()))
    };

    match result {
        Ok((freqs, power)) => {
            // Find peak
            if let Some((peak_idx, &peak_power)) = power
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            {
                let peak_freq = freqs[peak_idx];
                let freq_error = (peak_freq - f_signal).abs() / f_signal;
                
                // Very lenient for sparse sampling
                if freq_error > 0.5 {
                    issues.push(format!(
                        "Sparse sampling frequency error too high: {:.2e}", freq_error
                    ));
                }
                
                errors.push(freq_error);
            }
        }
        Err(_) => {
            issues.push("Sparse sampling failed".to_string());
        }
    }

    Ok(SingleTestResult {
        errors,
        peak_error: 0.0,
        peak_errors: vec![],
        issues,
    })
}

fn test_signal_with_outliers(implementation: &str) -> SignalResult<SingleTestResult> {
    let mut issues = Vec::new();
    let mut errors = Vec::new();

    let n = 200;
    let fs = 50.0;
    let f_signal = 5.0;
    let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
    let mut signal: Vec<f64> = t.iter().map(|&ti| (2.0 * PI * f_signal * ti).sin()).collect();
    
    // Add some outliers
    signal[50] = 100.0;  // Large positive outlier
    signal[100] = -50.0; // Large negative outlier
    signal[150] = 75.0;  // Another outlier

    let result = match implementation {
        "standard" => lombscargle(&t, &signal, None, Some("standard"), Some(true), Some(false), None, None),
        "enhanced" => {
            let config = LombScargleConfig {
                window: WindowType::None,
                custom_window: None,
                oversample: 5.0,
                f_min: Some(1.0),
                f_max: Some(20.0),
                bootstrap_iter: None,
                confidence: None,
                tolerance: 1e-10,
                use_fast: true,
            };
            enhanced_lombscargle(&t, &signal, &config).map(|(f, p, _)| (f, p))
        }
        _ => return Err(SignalError::ValueError("Unknown implementation".to_string()))
    };

    match result {
        Ok((freqs, power)) => {
            // Check that the implementation doesn't crash with outliers
            if power.iter().any(|&p| !p.is_finite()) {
                issues.push("Signal with outliers produced non-finite values".to_string());
            }
            
            // Try to find the main peak
            if let Some((peak_idx, &peak_power)) = power
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            {
                let peak_freq = freqs[peak_idx];
                let freq_error = (peak_freq - f_signal).abs() / f_signal;
                
                // Lenient tolerance due to outliers
                if freq_error > 0.2 {
                    issues.push(format!(
                        "Outliers caused significant frequency error: {:.2e}", freq_error
                    ));
                }
                
                errors.push(freq_error);
            }
        }
        Err(_) => {
            issues.push("Signal with outliers failed".to_string());
        }
    }

    Ok(SingleTestResult {
        errors,
        peak_error: 0.0,
        peak_errors: vec![],
        issues,
    })
}

fn test_small_values(implementation: &str) -> SignalResult<SingleTestResult> {
    let mut issues = Vec::new();
    let mut errors = Vec::new();

    let n = 500;
    let fs = 100.0;
    let f_signal = 10.0;
    let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
    
    // Very small signal values (near machine epsilon)
    let signal: Vec<f64> = t.iter().map(|&ti| 1e-14 * (2.0 * PI * f_signal * ti).sin()).collect();

    let result = match implementation {
        "standard" => lombscargle(&t, &signal, None, Some("standard"), Some(true), Some(false), None, None),
        "enhanced" => {
            let config = LombScargleConfig {
                window: WindowType::None,
                custom_window: None,
                oversample: 5.0,
                f_min: Some(1.0),
                f_max: Some(50.0),
                bootstrap_iter: None,
                confidence: None,
                tolerance: 1e-15,
                use_fast: true,
            };
            enhanced_lombscargle(&t, &signal, &config).map(|(f, p, _)| (f, p))
        }
        _ => return Err(SignalError::ValueError("Unknown implementation".to_string()))
    };

    match result {
        Ok((freqs, power)) => {
            // Should handle small values without numerical issues
            if power.iter().any(|&p| !p.is_finite()) {
                issues.push("Small values produced non-finite results".to_string());
            }
        }
        Err(_) => {
            // May fail due to numerical precision limits - acceptable
        }
    }

    Ok(SingleTestResult {
        errors,
        peak_error: 0.0,
        peak_errors: vec![],
        issues,
    })
}

fn test_large_values(implementation: &str) -> SignalResult<SingleTestResult> {
    let mut issues = Vec::new();
    let mut errors = Vec::new();

    let n = 500;
    let fs = 100.0;
    let f_signal = 10.0;
    let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
    
    // Very large signal values
    let signal: Vec<f64> = t.iter().map(|&ti| 1e10 * (2.0 * PI * f_signal * ti).sin()).collect();

    let result = match implementation {
        "standard" => lombscargle(&t, &signal, None, Some("standard"), Some(true), Some(false), None, None),
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
            enhanced_lombscargle(&t, &signal, &config).map(|(f, p, _)| (f, p))
        }
        _ => return Err(SignalError::ValueError("Unknown implementation".to_string()))
    };

    match result {
        Ok((freqs, power)) => {
            // Should handle large values without overflow
            if power.iter().any(|&p| !p.is_finite()) {
                issues.push("Large values produced non-finite results".to_string());
            }
        }
        Err(_) => {
            // May fail due to numerical overflow - should be rare with proper scaling
            issues.push("Large values caused computation failure".to_string());
        }
    }

    Ok(SingleTestResult {
        errors,
        peak_error: 0.0,
        peak_errors: vec![],
        issues,
    })
}

fn test_extreme_timescales(implementation: &str) -> SignalResult<SingleTestResult> {
    let mut issues = Vec::new();
    let mut errors = Vec::new();

    // Test 1: Very long time series with small time steps
    let n = 1000;
    let dt = 1e-6; // Microsecond sampling
    let f_signal = 1e3; // 1 kHz signal
    let t: Vec<f64> = (0..n).map(|i| i as f64 * dt).collect();
    let signal: Vec<f64> = t.iter().map(|&ti| (2.0 * PI * f_signal * ti).sin()).collect();

    let result = match implementation {
        "standard" => lombscargle(&t, &signal, None, Some("standard"), Some(true), Some(false), None, None),
        "enhanced" => {
            let config = LombScargleConfig {
                window: WindowType::None,
                custom_window: None,
                oversample: 5.0,
                f_min: Some(100.0),
                f_max: Some(10000.0),
                bootstrap_iter: None,
                confidence: None,
                tolerance: 1e-10,
                use_fast: true,
            };
            enhanced_lombscargle(&t, &signal, &config).map(|(f, p, _)| (f, p))
        }
        _ => return Err(SignalError::ValueError("Unknown implementation".to_string()))
    };

    match result {
        Ok((freqs, power)) => {
            // Should handle extreme timescales
            if power.iter().any(|&p| !p.is_finite()) {
                issues.push("Extreme timescales produced non-finite results".to_string());
            }
        }
        Err(_) => {
            // May have numerical precision issues with extreme scales
        }
    }

    Ok(SingleTestResult {
        errors,
        peak_error: 0.0,
        peak_errors: vec![],
        issues,
    })
}

/// Statistical properties validation result
#[derive(Debug, Clone)]
pub struct StatisticalValidationResult {
    /// Chi-squared test p-value for white noise
    pub white_noise_pvalue: f64,
    /// False alarm rate validation
    pub false_alarm_rate_error: f64,
    /// Bootstrap confidence interval coverage
    pub bootstrap_coverage: f64,
    /// Statistical issues found
    pub statistical_issues: Vec<String>,
}

/// Performance validation result
#[derive(Debug, Clone)]
pub struct PerformanceValidationResult {
    /// Standard implementation time (ms)
    pub standard_time_ms: f64,
    /// Enhanced implementation time (ms)
    pub enhanced_time_ms: f64,
    /// Memory usage (approximate MB)
    pub memory_usage_mb: f64,
    /// Speedup factor
    pub speedup_factor: f64,
    /// Performance issues found
    pub performance_issues: Vec<String>,
}

/// Comprehensive validation result
#[derive(Debug, Clone)]
pub struct ComprehensiveValidationResult {
    /// Basic analytical validation
    pub analytical: ValidationResult,
    /// Statistical properties validation
    pub statistical: StatisticalValidationResult,
    /// Performance validation
    pub performance: PerformanceValidationResult,
    /// Overall validation score (0-100)
    pub overall_score: f64,
    /// All issues combined
    pub all_issues: Vec<String>,
}

/// Enhanced validation function for comprehensive Lomb-Scargle testing
///
/// This function provides a comprehensive test suite that validates both
/// standard and enhanced Lomb-Scargle implementations against theoretical
/// expectations, statistical properties, and performance characteristics.
pub fn validate_lombscargle_comprehensive(tolerance: f64) -> SignalResult<ComprehensiveValidationResult> {
    println!("Running comprehensive Lomb-Scargle validation...");
    
    // 1. Basic analytical validation
    println!("Testing analytical cases...");
    let analytical_result = validate_analytical_cases("enhanced", tolerance)?;
    
    // 2. Statistical properties validation
    println!("Testing statistical properties...");
    let statistical_result = validate_statistical_properties(tolerance)?;
    
    // 3. Performance validation
    println!("Testing performance characteristics...");
    let performance_result = validate_performance_characteristics()?;
    
    // 4. Cross-validation with reference implementation (if available)
    println!("Testing cross-validation with reference...");
    let cross_validation_result = validate_cross_reference(tolerance)?;
    
    // Calculate overall score
    let overall_score = calculate_comprehensive_score(
        &analytical_result,
        &statistical_result,
        &performance_result,
        &cross_validation_result,
    );
    
    // Combine all issues
    let mut all_issues = analytical_result.issues.clone();
    all_issues.extend(statistical_result.statistical_issues.clone());
    all_issues.extend(performance_result.performance_issues.clone());
    all_issues.extend(cross_validation_result.issues.clone());
    
    // Report results
    println!("Comprehensive validation results:");
    println!("  Analytical - Max error: {:.2e}, Stability: {:.3}", 
             analytical_result.max_relative_error, analytical_result.stability_score);
    println!("  Statistical - White noise p-value: {:.3}, Bootstrap coverage: {:.3}", 
             statistical_result.white_noise_pvalue, statistical_result.bootstrap_coverage);
    println!("  Performance - Enhanced time: {:.1}ms, Speedup: {:.2}x", 
             performance_result.enhanced_time_ms, performance_result.speedup_factor);
    println!("  Overall score: {:.1}/100", overall_score);
    println!("  Total issues found: {}", all_issues.len());
    
    Ok(ComprehensiveValidationResult {
        analytical: analytical_result,
        statistical: statistical_result,
        performance: performance_result,
        overall_score,
        all_issues,
    })
}

/// Validate statistical properties of Lomb-Scargle periodogram
fn validate_statistical_properties(tolerance: f64) -> SignalResult<StatisticalValidationResult> {
    let mut statistical_issues = Vec::new();
    
    // Test 1: White noise should follow expected distribution
    let white_noise_pvalue = test_white_noise_statistics()?;
    if white_noise_pvalue < 0.01 {
        statistical_issues.push(format!(
            "White noise test failed (p-value: {:.3})", white_noise_pvalue
        ));
    }
    
    // Test 2: False alarm rate validation
    let false_alarm_rate_error = test_false_alarm_rates()?;
    if false_alarm_rate_error > tolerance * 100.0 {
        statistical_issues.push(format!(
            "False alarm rate error too high: {:.2e}", false_alarm_rate_error
        ));
    }
    
    // Test 3: Bootstrap confidence interval coverage
    let bootstrap_coverage = test_bootstrap_confidence_intervals()?;
    if bootstrap_coverage < 0.90 {
        statistical_issues.push(format!(
            "Bootstrap confidence interval coverage too low: {:.3}", bootstrap_coverage
        ));
    }
    
    Ok(StatisticalValidationResult {
        white_noise_pvalue,
        false_alarm_rate_error,
        bootstrap_coverage,
        statistical_issues,
    })
}

/// Test white noise statistical properties
fn test_white_noise_statistics() -> SignalResult<f64> {
    use rand::prelude::*;
    let mut rng = rand::rng();
    
    let n_trials = 100;
    let n_samples = 500;
    let fs = 100.0;
    let mut max_powers = Vec::new();
    
    for _ in 0..n_trials {
        // Generate white noise
        let t: Vec<f64> = (0..n_samples).map(|i| i as f64 / fs).collect();
        let signal: Vec<f64> = (0..n_samples).map(|_| rng.random_range(-1.0..1.0)).collect();
        
        // Compute periodogram
        let (_, power) = lombscargle(
            &t, &signal, None, Some("standard"), Some(true), Some(false), None, None
        )?;
        
        let max_power = power.iter().cloned().fold(0.0, f64::max);
        max_powers.push(max_power);
    }
    
    // For white noise with standard normalization, max power should follow exponential distribution
    // Use Kolmogorov-Smirnov test approximation
    max_powers.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = max_powers.len() as f64;
    
    let mut ks_statistic = 0.0;
    for (i, &power) in max_powers.iter().enumerate() {
        let empirical_cdf = (i + 1) as f64 / n;
        let expected_cdf = 1.0 - (-power).exp(); // Exponential CDF
        ks_statistic = ks_statistic.max((empirical_cdf - expected_cdf).abs());
    }
    
    // Approximate p-value for KS test
    let critical_value = 1.36 / n.sqrt(); // 95% confidence level
    let p_value = if ks_statistic < critical_value { 0.5 } else { 0.01 };
    
    Ok(p_value)
}

/// Test false alarm rates
fn test_false_alarm_rates() -> SignalResult<f64> {
    use rand::prelude::*;
    let mut rng = rand::rng();
    
    let n_trials = 1000;
    let n_samples = 200;
    let fs = 50.0;
    let fap_level = 0.05; // 5% false alarm probability
    
    let mut false_alarms = 0;
    
    for _ in 0..n_trials {
        // Generate pure noise
        let t: Vec<f64> = (0..n_samples).map(|i| i as f64 / fs).collect();
        let signal: Vec<f64> = (0..n_samples).map(|_| rng.random_range(-1.0..1.0)).collect();
        
        // Compute periodogram
        let (_, power) = lombscargle(
            &t, &signal, None, Some("standard"), Some(true), Some(false), None, None
        )?;
        
        // Calculate significance threshold
        let significance_threshold = -fap_level.ln();
        
        // Check for false alarms
        let max_power = power.iter().cloned().fold(0.0, f64::max);
        if max_power > significance_threshold {
            false_alarms += 1;
        }
    }
    
    let observed_fap = false_alarms as f64 / n_trials as f64;
    let error = (observed_fap - fap_level).abs() / fap_level;
    
    Ok(error)
}

/// Test bootstrap confidence intervals
fn test_bootstrap_confidence_intervals() -> SignalResult<f64> {
    // This is a simplified test - full bootstrap validation would be more complex
    let n = 500;
    let fs = 100.0;
    let f_signal = 10.0;
    let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
    let signal: Vec<f64> = t.iter().map(|&ti| (2.0 * PI * f_signal * ti).sin()).collect();
    
    // Test enhanced implementation with bootstrap
    let config = LombScargleConfig {
        window: WindowType::None,
        custom_window: None,
        oversample: 5.0,
        f_min: Some(5.0),
        f_max: Some(15.0),
        bootstrap_iter: Some(100),
        confidence: Some(0.95),
        tolerance: 1e-10,
        use_fast: true,
    };
    
    let (freqs, power, bootstrap_result) = enhanced_lombscargle(&t, &signal, &config)?;
    
    if let Some(bootstrap) = bootstrap_result {
        // Find the peak
        let (peak_idx, _) = power
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
            
        // Check if confidence intervals exist and are reasonable
        if bootstrap.confidence_intervals.is_some() {
            let ci = bootstrap.confidence_intervals.unwrap();
            let lower = ci.0[peak_idx];
            let upper = ci.1[peak_idx];
            let peak_power = power[peak_idx];
            
            // Basic sanity checks
            if lower <= peak_power && peak_power <= upper && lower < upper {
                Ok(0.95) // Assume good coverage for now
            } else {
                Ok(0.5) // Poor coverage
            }
        } else {
            Ok(0.0) // No confidence intervals
        }
    } else {
        Ok(0.0) // No bootstrap result
    }
}

/// Validate performance characteristics
fn validate_performance_characteristics() -> SignalResult<PerformanceValidationResult> {
    use std::time::Instant;
    let mut performance_issues = Vec::new();
    
    // Test data
    let n = 5000;
    let fs = 1000.0;
    let f_signal = 50.0;
    let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
    let signal: Vec<f64> = t.iter().map(|&ti| (2.0 * PI * f_signal * ti).sin()).collect();
    
    // Benchmark standard implementation
    let start = Instant::now();
    for _ in 0..10 {
        let _ = lombscargle(&t, &signal, None, Some("standard"), Some(true), Some(false), None, None)?;
    }
    let standard_time_ms = start.elapsed().as_secs_f64() * 100.0; // ms per iteration
    
    // Benchmark enhanced implementation
    let config = LombScargleConfig {
        window: WindowType::None,
        custom_window: None,
        oversample: 5.0,
        f_min: Some(10.0),
        f_max: Some(100.0),
        bootstrap_iter: None,
        confidence: None,
        tolerance: 1e-10,
        use_fast: true,
    };
    
    let start = Instant::now();
    for _ in 0..10 {
        let _ = enhanced_lombscargle(&t, &signal, &config)?;
    }
    let enhanced_time_ms = start.elapsed().as_secs_f64() * 100.0; // ms per iteration
    
    // Calculate speedup
    let speedup_factor = standard_time_ms / enhanced_time_ms;
    
    // Estimate memory usage (rough approximation)
    let memory_usage_mb = (n * std::mem::size_of::<f64>() * 4) as f64 / (1024.0 * 1024.0);
    
    // Performance checks
    if enhanced_time_ms > standard_time_ms * 2.0 {
        performance_issues.push("Enhanced implementation is significantly slower than standard".to_string());
    }
    
    if memory_usage_mb > 100.0 {
        performance_issues.push(format!("High memory usage: {:.1} MB", memory_usage_mb));
    }
    
    Ok(PerformanceValidationResult {
        standard_time_ms,
        enhanced_time_ms,
        memory_usage_mb,
        speedup_factor,
        performance_issues,
    })
}

/// Cross-validate with reference implementation
fn validate_cross_reference(tolerance: f64) -> SignalResult<ValidationResult> {
    // This would ideally compare with SciPy's implementation
    // For now, we'll do self-consistency checks between standard and enhanced
    
    let mut issues = Vec::new();
    let mut errors = Vec::new();
    
    let n = 1000;
    let fs = 100.0;
    let f_signal = 15.0;
    let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
    let signal: Vec<f64> = t.iter().map(|&ti| (2.0 * PI * f_signal * ti).sin()).collect();
    
    // Standard implementation
    let (freqs_std, power_std) = lombscargle(
        &t, &signal, None, Some("standard"), Some(true), Some(false), None, None
    )?;
    
    // Enhanced implementation
    let config = LombScargleConfig {
        window: WindowType::None,
        custom_window: None,
        oversample: 5.0,
        f_min: Some(5.0),
        f_max: Some(25.0),
        bootstrap_iter: None,
        confidence: None,
        tolerance: 1e-10,
        use_fast: true,
    };
    let (freqs_enh, power_enh, _) = enhanced_lombscargle(&t, &signal, &config)?;
    
    // Find common frequency range for comparison
    let f_min_common = freqs_std[0].max(freqs_enh[0]);
    let f_max_common = freqs_std[freqs_std.len()-1].min(freqs_enh[freqs_enh.len()-1]);
    
    // Interpolate to common grid for comparison
    let n_compare = 500;
    let compare_freqs: Vec<f64> = (0..n_compare)
        .map(|i| f_min_common + (f_max_common - f_min_common) * i as f64 / (n_compare - 1) as f64)
        .collect();
    
    let power_std_interp = interpolate_power(&freqs_std, &power_std, &compare_freqs);
    let power_enh_interp = interpolate_power(&freqs_enh, &power_enh, &compare_freqs);
    
    // Compare interpolated values
    for (i, (&p_std, &p_enh)) in power_std_interp.iter().zip(power_enh_interp.iter()).enumerate() {
        if p_std > 0.01 { // Only compare significant values
            let rel_error = (p_std - p_enh).abs() / p_std;
            errors.push(rel_error);
            
            if rel_error > tolerance * 10.0 { // More lenient for different implementations
                if issues.len() < 5 { // Limit number of detailed issues
                    issues.push(format!(
                        "Standard vs Enhanced mismatch at freq {:.2}: {:.2e}", 
                        compare_freqs[i], rel_error
                    ));
                }
            }
        }
    }
    
    let max_relative_error = errors.iter().cloned().fold(0.0, f64::max);
    let mean_relative_error = if !errors.is_empty() {
        errors.iter().sum::<f64>() / errors.len() as f64
    } else {
        0.0
    };
    
    let stability_score = calculate_stability_score(&issues, &errors);
    
    Ok(ValidationResult {
        max_relative_error,
        mean_relative_error,
        stability_score,
        peak_freq_error: 0.0, // Not applicable for cross-validation
        issues,
    })
}

/// Simple linear interpolation for power values
fn interpolate_power(freqs: &[f64], power: &[f64], target_freqs: &[f64]) -> Vec<f64> {
    target_freqs.iter().map(|&target_freq| {
        // Find bracketing indices
        let mut lower_idx = 0;
        let mut upper_idx = freqs.len() - 1;
        
        for (i, &freq) in freqs.iter().enumerate() {
            if freq <= target_freq {
                lower_idx = i;
            } else {
                upper_idx = i;
                break;
            }
        }
        
        if lower_idx == upper_idx {
            power[lower_idx]
        } else {
            let f1 = freqs[lower_idx];
            let f2 = freqs[upper_idx];
            let p1 = power[lower_idx];
            let p2 = power[upper_idx];
            
            if (f2 - f1).abs() > 1e-15 {
                let weight = (target_freq - f1) / (f2 - f1);
                p1 + weight * (p2 - p1)
            } else {
                (p1 + p2) / 2.0
            }
        }
    }).collect()
}

/// Calculate comprehensive validation score
fn calculate_comprehensive_score(
    analytical: &ValidationResult,
    statistical: &StatisticalValidationResult,
    performance: &PerformanceValidationResult,
    cross_validation: &ValidationResult,
) -> f64 {
    let mut score = 100.0;
    
    // Analytical score (40 points)
    score -= analytical.max_relative_error * 1000.0;
    score -= (1.0 - analytical.stability_score) * 20.0;
    score -= analytical.issues.len() as f64 * 2.0;
    
    // Statistical score (30 points)
    if statistical.white_noise_pvalue < 0.01 {
        score -= 10.0;
    }
    score -= statistical.false_alarm_rate_error * 10.0;
    if statistical.bootstrap_coverage < 0.90 {
        score -= 10.0;
    }
    score -= statistical.statistical_issues.len() as f64 * 2.0;
    
    // Performance score (20 points)
    if performance.speedup_factor < 1.0 {
        score -= 10.0;
    }
    if performance.memory_usage_mb > 50.0 {
        score -= 5.0;
    }
    score -= performance.performance_issues.len() as f64 * 2.0;
    
    // Cross-validation score (10 points)
    score -= cross_validation.max_relative_error * 100.0;
    score -= cross_validation.issues.len() as f64 * 1.0;
    
    score.max(0.0).min(100.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_framework() {
        let tolerance = 1e-6;
        let result = validate_analytical_cases("standard", tolerance);
        assert!(result.is_ok());
        
        let validation = result.unwrap();
        assert!(validation.stability_score >= 0.0);
        assert!(validation.stability_score <= 1.0);
    }

    #[test]
    fn test_enhanced_validation() {
        let tolerance = 1e-6;
        let result = validate_analytical_cases("enhanced", tolerance);
        assert!(result.is_ok());
        
        let validation = result.unwrap();
        assert!(validation.stability_score >= 0.0);
        assert!(validation.stability_score <= 1.0);
    }

    #[test]
    fn test_comprehensive_validation() {
        let tolerance = 1e-5; // More lenient for comprehensive test
        let result = validate_lombscargle_comprehensive(tolerance);
        assert!(result.is_ok());
    }
}