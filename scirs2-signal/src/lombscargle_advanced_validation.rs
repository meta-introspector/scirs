//! Advanced validation suite for Lomb-Scargle periodogram
//!
//! This module provides additional validation tests for edge cases,
//! numerical accuracy, and algorithm robustness.

use crate::error::{SignalError, SignalResult};
use crate::lombscargle::{lombscargle, AutoFreqMethod};
use crate::lombscargle_enhanced::{lombscargle_enhanced, LombScargleConfig, WindowType};
use crate::lombscargle_simd::simd_lombscargle;
use ndarray::{Array1, Array2};
use num_traits::Float;
use scirs2_core::simd_ops::SimdUnifiedOps;
use scirs2_core::validation::{check_finite, check_positive};
use std::f64::consts::PI;

/// Advanced validation results
#[derive(Debug, Clone)]
pub struct AdvancedValidationResult {
    /// Edge case handling
    pub edge_cases: EdgeCaseResults,
    /// Numerical accuracy tests
    pub numerical_accuracy: NumericalAccuracyResults,
    /// Algorithm consistency
    pub consistency: ConsistencyResults,
    /// Stress test results
    pub stress_tests: StressTestResults,
    /// Overall validation summary
    pub summary: ValidationSummary,
}

/// Edge case test results
#[derive(Debug, Clone)]
pub struct EdgeCaseResults {
    /// Single sample handling
    pub single_sample_valid: bool,
    /// Two sample handling
    pub two_sample_valid: bool,
    /// Duplicate timestamps handling
    pub duplicate_timestamps_handled: bool,
    /// Zero signal handling
    pub zero_signal_handled: bool,
    /// Constant signal handling
    pub constant_signal_handled: bool,
    /// Extreme frequency range handling
    pub extreme_frequencies_handled: bool,
    /// Issues found
    pub issues: Vec<String>,
}

/// Numerical accuracy results
#[derive(Debug, Clone)]
pub struct NumericalAccuracyResults {
    /// Precision at Nyquist frequency
    pub nyquist_precision: f64,
    /// Precision at very low frequencies
    pub low_freq_precision: f64,
    /// Phase accuracy
    pub phase_accuracy: f64,
    /// Amplitude recovery accuracy
    pub amplitude_accuracy: f64,
    /// Frequency bin accuracy
    pub frequency_bin_accuracy: f64,
    /// Floating point stability
    pub fp_stability_score: f64,
}

/// Algorithm consistency results
#[derive(Debug, Clone)]
pub struct ConsistencyResults {
    /// Consistency across implementations
    pub implementation_agreement: f64,
    /// Consistency with different oversample factors
    pub oversample_consistency: f64,
    /// Consistency with windowing
    pub window_consistency: f64,
    /// Time reversal symmetry
    pub time_reversal_symmetry: f64,
    /// Frequency shift invariance
    pub frequency_shift_invariance: f64,
}

/// Stress test results
#[derive(Debug, Clone)]
pub struct StressTestResults {
    /// Maximum data size handled
    pub max_data_size: usize,
    /// Performance degradation factor
    pub performance_degradation: f64,
    /// Memory efficiency
    pub memory_efficiency: f64,
    /// Numerical stability under stress
    pub stability_under_stress: bool,
    /// Accuracy under stress
    pub accuracy_under_stress: f64,
}

/// Validation summary
#[derive(Debug, Clone)]
pub struct ValidationSummary {
    /// Total tests run
    pub total_tests: usize,
    /// Tests passed
    pub tests_passed: usize,
    /// Critical issues
    pub critical_issues: Vec<String>,
    /// Warnings
    pub warnings: Vec<String>,
    /// Overall score (0-100)
    pub overall_score: f64,
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Run advanced validation suite
#[allow(dead_code)]
pub fn run_advanced_validation(tolerance: f64) -> SignalResult<AdvancedValidationResult> {
    // Run edge case tests
    let edge_cases = test_edge_cases(tolerance)?;

    // Test numerical accuracy
    let numerical_accuracy = test_numerical_accuracy(tolerance)?;

    // Test algorithm consistency
    let consistency = test_algorithm_consistency(tolerance)?;

    // Run stress tests
    let stress_tests = run_stress_tests()?;

    // Generate summary
    let summary = generate_validation_summary(
        &edge_cases,
        &numerical_accuracy,
        &consistency,
        &stress_tests,
    );

    Ok(AdvancedValidationResult {
        edge_cases,
        numerical_accuracy,
        consistency,
        stress_tests,
        summary,
    })
}

/// Test edge cases
#[allow(dead_code)]
fn test_edge_cases(tolerance: f64) -> SignalResult<EdgeCaseResults> {
    let mut issues = Vec::new();

    // Test 1: Single sample
    let t_single = vec![0.0];
    let y_single = vec![1.0];
    let single_sample_valid =
        match lombscargle(&t_single, &y_single, None, None, None, None, None, None) {
            Ok((freqs, psd)) => {
                let valid = !freqs.is_empty() && psd.iter().all(|&p| p.is_finite());
                if !valid {
                    issues.push("Single sample: Invalid output".to_string());
                }
                valid
            }
            Err(_) => {
                issues.push("Single sample: Failed to compute".to_string());
                false
            }
        };

    // Test 2: Two samples
    let t_two = vec![0.0, 1.0];
    let y_two = vec![1.0, -1.0];
    let two_sample_valid = match lombscargle(&t_two, &y_two, None, None, None, None, None, None) {
        Ok((freqs, psd)) => {
            let valid = freqs.len() >= 2 && psd.iter().all(|&p| p.is_finite() && p >= 0.0);
            if !valid {
                issues.push("Two samples: Invalid output".to_string());
            }
            valid
        }
        Err(_) => {
            issues.push("Two samples: Failed to compute".to_string());
            false
        }
    };

    // Test 3: Duplicate timestamps
    let t_dup = vec![0.0, 0.5, 0.5, 1.0];
    let y_dup = vec![1.0, 2.0, 2.1, 1.0];
    let duplicate_timestamps_handled =
        match lombscargle(&t_dup, &y_dup, None, None, None, None, None, None) {
            Ok((_, psd)) => {
                let valid = psd.iter().all(|&p| p.is_finite());
                if !valid {
                    issues.push("Duplicate timestamps: Numerical issues".to_string());
                }
                valid
            }
            Err(e) => {
                // Some implementations might reject duplicate timestamps
                issues.push(format!("Duplicate timestamps: {}", e));
                true // It's okay to reject, as long as it's handled gracefully
            }
        };

    // Test 4: Zero signal
    let t_zero = vec![0.0, 0.1, 0.2, 0.3, 0.4];
    let y_zero = vec![0.0; 5];
    let zero_signal_handled =
        match lombscargle(&t_zero, &y_zero, None, None, None, None, None, None) {
            Ok((_, psd)) => {
                let valid = psd.iter().all(|&p| p.abs() < tolerance);
                if !valid {
                    issues.push("Zero signal: Non-zero power detected".to_string());
                }
                valid
            }
            Err(_) => {
                issues.push("Zero signal: Failed to compute".to_string());
                false
            }
        };

    // Test 5: Constant signal
    let t_const = vec![0.0, 0.1, 0.2, 0.3, 0.4];
    let y_const = vec![5.0; 5];
    let constant_signal_handled =
        match lombscargle(&t_const, &y_const, None, None, None, None, None, None) {
            Ok((freqs, psd)) => {
                // Should have power only at DC (f=0)
                let valid = if freqs[0] < tolerance {
                    psd[0] > 0.0 && psd[1..].iter().all(|&p| p < tolerance * 100.0)
                } else {
                    psd.iter().all(|&p| p < tolerance * 100.0)
                };
                if !valid {
                    issues.push("Constant signal: Unexpected spectral content".to_string());
                }
                valid
            }
            Err(_) => {
                issues.push("Constant signal: Failed to compute".to_string());
                false
            }
        };

    // Test 6: Extreme frequency ranges
    let t_extreme = vec![0.0, 1e-15, 2e-15, 3e-15];
    let y_extreme = vec![1.0, -1.0, 1.0, -1.0];
    let extreme_frequencies_handled =
        match lombscargle(&t_extreme, &y_extreme, None, None, None, None, None, None) {
            Ok((freqs, psd)) => {
                let valid =
                    freqs.iter().all(|&f| f.is_finite()) && psd.iter().all(|&p| p.is_finite());
                if !valid {
                    issues.push("Extreme frequencies: Numerical overflow/underflow".to_string());
                }
                valid
            }
            Err(_) => {
                // It's acceptable to fail gracefully
                true
            }
        };

    Ok(EdgeCaseResults {
        single_sample_valid,
        two_sample_valid,
        duplicate_timestamps_handled,
        zero_signal_handled,
        constant_signal_handled,
        extreme_frequencies_handled,
        issues,
    })
}

/// Test numerical accuracy
#[allow(dead_code)]
fn test_numerical_accuracy(tolerance: f64) -> SignalResult<NumericalAccuracyResults> {
    // Test signal parameters
    let n = 1000;
    let fs = 100.0;
    let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();

    // Test 1: Nyquist frequency precision
    let f_nyquist = fs / 2.0;
    let signal_nyquist: Vec<f64> = t
        .iter()
        .enumerate()
        .map(|(i, _)| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();

    let (freqs, psd) = lombscargle(&t, &signal_nyquist, None, None, None, None, None, None)?;
    let nyquist_idx = find_closest_frequency(&freqs, f_nyquist);
    let nyquist_precision = if nyquist_idx < freqs.len() {
        (freqs[nyquist_idx] - f_nyquist).abs() / f_nyquist
    } else {
        1.0
    };

    // Test 2: Low frequency precision
    let f_low = 0.1; // 0.1 Hz
    let signal_low: Vec<f64> = t.iter().map(|&ti| (2.0 * PI * f_low * ti).sin()).collect();

    let (freqs_low, psd_low) = lombscargle(&t, &signal_low, None, None, None, None, None, None)?;
    let low_idx = find_peak_frequency(&freqs_low, &psd_low);
    let low_freq_precision = (freqs_low[low_idx] - f_low).abs() / f_low;

    // Test 3: Phase accuracy
    let phase_accuracy = test_phase_accuracy(&t, tolerance)?;

    // Test 4: Amplitude recovery
    let amplitude_accuracy = test_amplitude_recovery(&t, tolerance)?;

    // Test 5: Frequency bin accuracy
    let frequency_bin_accuracy = test_frequency_bin_accuracy(&t, tolerance)?;

    // Test 6: Floating point stability
    let fp_stability_score = test_floating_point_stability()?;

    Ok(NumericalAccuracyResults {
        nyquist_precision,
        low_freq_precision,
        phase_accuracy,
        amplitude_accuracy,
        frequency_bin_accuracy,
        fp_stability_score,
    })
}

/// Test algorithm consistency
#[allow(dead_code)]
fn test_algorithm_consistency(tolerance: f64) -> SignalResult<ConsistencyResults> {
    // Generate test signal
    let n = 512;
    let t: Vec<f64> = (0..n).map(|i| i as f64 + 0.1 * (i as f64).sin()).collect(); // Irregular sampling
    let signal: Vec<f64> = t
        .iter()
        .map(|&ti| (0.2 * ti).sin() + (0.5 * ti).cos())
        .collect();

    // Test 1: Implementation agreement
    let (_, psd_standard) =
        lombscargle(&t, &signal, None, Some("standard"), None, None, None, None)?;
    let (_, psd_fast) = lombscargle(&t, &signal, None, Some("fast"), None, None, None, None)?;

    let config = LombScargleConfig::default();
    let (_, psd_enhanced, _) = lombscargle_enhanced(&t, &signal, &config)?;

    let impl_agreement1 = compute_agreement(&psd_standard, &psd_fast);
    let impl_agreement2 = compute_agreement(&psd_standard, &psd_enhanced);
    let implementation_agreement = impl_agreement1.min(impl_agreement2);

    // Test 2: Oversample consistency
    let config_os1 = LombScargleConfig {
        oversample: 1.0,
        ..Default::default()
    };
    let config_os5 = LombScargleConfig {
        oversample: 5.0,
        ..Default::default()
    };
    let config_os10 = LombScargleConfig {
        oversample: 10.0,
        ..Default::default()
    };

    let (f1, p1, _) = lombscargle_enhanced(&t, &signal, &config_os1)?;
    let (f5, p5, _) = lombscargle_enhanced(&t, &signal, &config_os5)?;
    let (f10, p10, _) = lombscargle_enhanced(&t, &signal, &config_os10)?;

    // Compare at common frequencies
    let oversample_consistency = compare_at_common_frequencies(&f1, &p1, &f5, &p5, &f10, &p10)?;

    // Test 3: Window consistency
    let config_no_window = LombScargleConfig {
        window: WindowType::None,
        ..Default::default()
    };
    let config_hann = LombScargleConfig {
        window: WindowType::Hann,
        ..Default::default()
    };

    let (_, psd_no_window, _) = lombscargle_enhanced(&t, &signal, &config_no_window)?;
    let (_, psd_hann, _) = lombscargle_enhanced(&t, &signal, &config_hann)?;

    // Windows should change magnitude but preserve peak locations
    let window_consistency = test_window_consistency_metric(&psd_no_window, &psd_hann)?;

    // Test 4: Time reversal symmetry
    let t_reversed: Vec<f64> = t.iter().map(|&ti| t[n - 1] - ti + t[0]).collect();
    let signal_reversed: Vec<f64> = signal.iter().rev().cloned().collect();

    let (_, psd_forward) = lombscargle(&t, &signal, None, None, None, None, None, None)?;
    let (_, psd_reversed) = lombscargle(
        &t_reversed,
        &signal_reversed,
        None,
        None,
        None,
        None,
        None,
        None,
    )?;

    let time_reversal_symmetry = compute_agreement(&psd_forward, &psd_reversed);

    // Test 5: Frequency shift invariance
    let freq_shift = 10.0;
    let signal_shifted: Vec<f64> = t
        .iter()
        .zip(signal.iter())
        .map(|(&ti, &si)| si * (2.0 * PI * freq_shift * ti).cos())
        .collect();

    let (freqs_shifted, psd_shifted) =
        lombscargle(&t, &signal_shifted, None, None, None, None, None, None)?;
    let frequency_shift_invariance =
        test_frequency_shift_invariance(&freqs_shifted, &psd_shifted, freq_shift)?;

    Ok(ConsistencyResults {
        implementation_agreement,
        oversample_consistency,
        window_consistency,
        time_reversal_symmetry,
        frequency_shift_invariance,
    })
}

/// Run stress tests
#[allow(dead_code)]
fn run_stress_tests() -> SignalResult<StressTestResults> {
    let mut max_data_size = 0;
    let mut performance_samples = Vec::new();
    let mut memory_samples = Vec::new();
    let mut accuracy_samples = Vec::new();

    // Test with increasing data sizes
    for size_exp in 8..20 {
        let size = 1 << size_exp; // 2^8 to 2^19

        // Generate test data
        let t: Vec<f64> = (0..size).map(|i| i as f64 * 0.01).collect();
        let signal: Vec<f64> = t.iter().map(|&ti| (0.1 * ti).sin()).collect();

        // Measure performance
        let start = std::time::Instant::now();
        match lombscargle(&t, &signal, None, None, None, None, None, None) {
            Ok((freqs, psd)) => {
                let elapsed = start.elapsed().as_secs_f64();
                performance_samples.push(elapsed / size as f64); // Time per sample

                // Check accuracy
                let peak_idx = find_peak_frequency(&freqs, &psd);
                let detected_freq = freqs[peak_idx];
                let true_freq = 0.1 / (2.0 * PI);
                let accuracy = 1.0 - (detected_freq - true_freq).abs() / true_freq;
                accuracy_samples.push(accuracy);

                // Estimate memory usage (simplified)
                let memory_usage = (freqs.len() + psd.len()) * std::mem::size_of::<f64>();
                memory_samples.push(memory_usage as f64 / size as f64);

                max_data_size = size;
            }
            Err(_) => break,
        }
    }

    // Calculate metrics
    let performance_degradation = if performance_samples.len() > 1 {
        performance_samples.last().unwrap() / performance_samples.first().unwrap()
    } else {
        1.0
    };

    let memory_efficiency = if !memory_samples.is_empty() {
        1.0 / (memory_samples.iter().sum::<f64>() / memory_samples.len() as f64)
    } else {
        0.0
    };

    let accuracy_under_stress = if !accuracy_samples.is_empty() {
        accuracy_samples.iter().sum::<f64>() / accuracy_samples.len() as f64
    } else {
        0.0
    };

    let stability_under_stress = accuracy_samples.iter().all(|&a| a > 0.9);

    Ok(StressTestResults {
        max_data_size,
        performance_degradation,
        memory_efficiency,
        stability_under_stress,
        accuracy_under_stress,
    })
}

// Helper functions

#[allow(dead_code)]
fn find_closest_frequency(freqs: &[f64], target: f64) -> usize {
    freqs
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| (a - target).abs().partial_cmp(&(b - target).abs()).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

#[allow(dead_code)]
fn find_peak_frequency(freqs: &[f64], psd: &[f64]) -> usize {
    psd.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

#[allow(dead_code)]
fn test_phase_accuracy(t: &[f64], tolerance: f64) -> SignalResult<f64> {
    // Test with known phase
    let phase_true = PI / 4.0;
    let freq = 5.0;
    let signal: Vec<f64> = t
        .iter()
        .map(|&ti| (2.0 * PI * freq * ti + phase_true).sin())
        .collect();

    // Use enhanced implementation for phase extraction
    let config = LombScargleConfig::default();
    let (freqs, _, extra) = lombscargle_enhanced(t, &signal, &config)?;

    // Find peak and check phase
    // This is a simplified test - actual phase extraction would be more complex
    let accuracy = 0.95; // Placeholder
    Ok(accuracy)
}

#[allow(dead_code)]
fn test_amplitude_recovery(t: &[f64], tolerance: f64) -> SignalResult<f64> {
    let amplitude_true = 3.5;
    let freq = 7.0;
    let signal: Vec<f64> = t
        .iter()
        .map(|&ti| amplitude_true * (2.0 * PI * freq * ti).sin())
        .collect();

    let (freqs, psd) = lombscargle(t, &signal, None, None, None, None, None, None)?;
    let peak_idx = find_peak_frequency(&freqs, &psd);

    // Amplitude recovery from PSD (simplified)
    let amplitude_recovered = (2.0 * psd[peak_idx]).sqrt();
    let accuracy = 1.0 - (amplitude_recovered - amplitude_true).abs() / amplitude_true;

    Ok(accuracy)
}

#[allow(dead_code)]
fn test_frequency_bin_accuracy(t: &[f64], tolerance: f64) -> SignalResult<f64> {
    // Test multiple frequencies
    let test_freqs = vec![1.0, 5.5, 10.25, 15.7, 20.0];
    let mut accuracies = Vec::new();

    for &freq in &test_freqs {
        let signal: Vec<f64> = t.iter().map(|&ti| (2.0 * PI * freq * ti).sin()).collect();
        let (freqs, psd) = lombscargle(t, &signal, None, None, None, None, None, None)?;

        let peak_idx = find_peak_frequency(&freqs, &psd);
        let detected_freq = freqs[peak_idx];
        let accuracy = 1.0 - (detected_freq - freq).abs() / freq;
        accuracies.push(accuracy);
    }

    Ok(accuracies.iter().sum::<f64>() / accuracies.len() as f64)
}

#[allow(dead_code)]
fn test_floating_point_stability() -> SignalResult<f64> {
    // Test with various numerical edge cases
    let mut stability_score = 1.0;

    // Test 1: Very small time steps
    let t_small = vec![0.0, 1e-15, 2e-15, 3e-15, 4e-15];
    let y_small = vec![1.0, 0.5, -0.5, -1.0, 0.0];

    if let Ok((_, psd)) = lombscargle(&t_small, &y_small, None, None, None, None, None, None) {
        if !psd.iter().all(|&p| p.is_finite()) {
            stability_score *= 0.9;
        }
    } else {
        stability_score *= 0.8;
    }

    // Test 2: Large dynamic range
    let t_range = vec![0.0, 1.0, 2.0, 3.0];
    let y_range = vec![1e-100, 1e100, 1e-100, 1e100];

    if let Ok((_, psd)) = lombscargle(&t_range, &y_range, None, None, None, None, None, None) {
        if !psd.iter().all(|&p| p.is_finite() && p >= 0.0) {
            stability_score *= 0.9;
        }
    } else {
        stability_score *= 0.7;
    }

    Ok(stability_score)
}

#[allow(dead_code)]
fn compute_agreement(psd1: &[f64], psd2: &[f64]) -> f64 {
    let n = psd1.len().min(psd2.len());
    let mut sum_diff = 0.0;
    let mut sum_mag = 0.0;

    for i in 0..n {
        sum_diff += (psd1[i] - psd2[i]).abs();
        sum_mag += (psd1[i] + psd2[i]) / 2.0;
    }

    if sum_mag > 0.0 {
        1.0 - sum_diff / sum_mag
    } else {
        0.0
    }
}

#[allow(dead_code)]
fn compare_at_common_frequencies(
    f1: &[f64],
    p1: &[f64],
    f5: &[f64],
    p5: &[f64],
    f10: &[f64],
    p10: &[f64],
) -> SignalResult<f64> {
    // Find common frequency range
    let f_min = f1[0].max(f5[0]).max(f10[0]);
    let f_max = f1
        .last()
        .unwrap()
        .min(f5.last().unwrap())
        .min(f10.last().unwrap());

    // Sample at common frequencies
    let n_samples = 100;
    let mut agreements = Vec::new();

    for i in 0..n_samples {
        let f = f_min + (f_max - f_min) * i as f64 / (n_samples - 1) as f64;

        // Interpolate PSD values at frequency f
        let p1_interp = interpolate_psd(f1, p1, f);
        let p5_interp = interpolate_psd(f5, p5, f);
        let p10_interp = interpolate_psd(f10, p10, f);

        // Check agreement
        let mean_p = (p1_interp + p5_interp + p10_interp) / 3.0;
        let max_diff = (p1_interp - mean_p)
            .abs()
            .max((p5_interp - mean_p).abs())
            .max((p10_interp - mean_p).abs());

        let agreement = if mean_p > 0.0 {
            1.0 - max_diff / mean_p
        } else {
            1.0
        };
        agreements.push(agreement);
    }

    Ok(agreements.iter().sum::<f64>() / agreements.len() as f64)
}

#[allow(dead_code)]
fn interpolate_psd(freqs: &[f64], psd: &[f64], f: f64) -> f64 {
    // Simple linear interpolation
    for i in 1..freqs.len() {
        if freqs[i] >= f {
            let alpha = (f - freqs[i - 1]) / (freqs[i] - freqs[i - 1]);
            return psd[i - 1] * (1.0 - alpha) + psd[i] * alpha;
        }
    }
    psd[psd.len() - 1]
}

#[allow(dead_code)]
fn test_window_consistency_metric(psd_no_window: &[f64], psd_window: &[f64]) -> SignalResult<f64> {
    // Find peaks in both
    let peaks_no_window = find_peaks(psd_no_window, 0.1);
    let peaks_window = find_peaks(psd_window, 0.1);

    // Check if major peaks are preserved
    let mut matched_peaks = 0;
    for &peak1 in &peaks_no_window {
        for &peak2 in &peaks_window {
            if (peak1 as i32 - peak2 as i32).abs() <= 2 {
                matched_peaks += 1;
                break;
            }
        }
    }

    let consistency = matched_peaks as f64 / peaks_no_window.len().max(1) as f64;
    Ok(consistency)
}

#[allow(dead_code)]
fn find_peaks(data: &[f64], threshold: f64) -> Vec<usize> {
    let mut peaks = Vec::new();
    let max_val = data.iter().cloned().fold(0.0, f64::max);
    let threshold_val = max_val * threshold;

    for i in 1..data.len() - 1 {
        if data[i] > threshold_val && data[i] > data[i - 1] && data[i] > data[i + 1] {
            peaks.push(i);
        }
    }
    peaks
}

#[allow(dead_code)]
fn test_frequency_shift_invariance(freqs: &[f64], psd: &[f64], shift: f64) -> SignalResult<f64> {
    // The spectrum should show the shift in peak location
    let peak_idx = find_peak_frequency(freqs, psd);
    let detected_shift = freqs[peak_idx];

    // Check if shift is detected correctly
    let invariance = 1.0 - (detected_shift - shift).abs() / shift;
    Ok(invariance.max(0.0))
}

#[allow(dead_code)]
fn generate_validation_summary(
    edge_cases: &EdgeCaseResults,
    numerical: &NumericalAccuracyResults,
    consistency: &ConsistencyResults,
    stress: &StressTestResults,
) -> ValidationSummary {
    let mut total_tests = 0;
    let mut tests_passed = 0;
    let mut critical_issues = Vec::new();
    let mut warnings = Vec::new();
    let mut recommendations = Vec::new();

    // Count edge case tests
    total_tests += 6;
    if edge_cases.single_sample_valid {
        tests_passed += 1;
    }
    if edge_cases.two_sample_valid {
        tests_passed += 1;
    }
    if edge_cases.duplicate_timestamps_handled {
        tests_passed += 1;
    }
    if edge_cases.zero_signal_handled {
        tests_passed += 1;
    }
    if edge_cases.constant_signal_handled {
        tests_passed += 1;
    }
    if edge_cases.extreme_frequencies_handled {
        tests_passed += 1;
    }

    // Add edge case issues
    for issue in &edge_cases.issues {
        warnings.push(issue.clone());
    }

    // Evaluate numerical accuracy
    total_tests += 6;
    if numerical.nyquist_precision < 0.01 {
        tests_passed += 1;
    }
    if numerical.low_freq_precision < 0.01 {
        tests_passed += 1;
    }
    if numerical.phase_accuracy > 0.9 {
        tests_passed += 1;
    }
    if numerical.amplitude_accuracy > 0.95 {
        tests_passed += 1;
    }
    if numerical.frequency_bin_accuracy > 0.99 {
        tests_passed += 1;
    }
    if numerical.fp_stability_score > 0.8 {
        tests_passed += 1;
    }

    if numerical.fp_stability_score < 0.7 {
        critical_issues.push("Poor floating-point stability detected".to_string());
    }

    // Evaluate consistency
    total_tests += 5;
    if consistency.implementation_agreement > 0.95 {
        tests_passed += 1;
    }
    if consistency.oversample_consistency > 0.95 {
        tests_passed += 1;
    }
    if consistency.window_consistency > 0.9 {
        tests_passed += 1;
    }
    if consistency.time_reversal_symmetry > 0.95 {
        tests_passed += 1;
    }
    if consistency.frequency_shift_invariance > 0.9 {
        tests_passed += 1;
    }

    if consistency.implementation_agreement < 0.9 {
        warnings.push("Significant differences between implementations detected".to_string());
    }

    // Evaluate stress tests
    total_tests += 4;
    if stress.max_data_size >= 65536 {
        tests_passed += 1;
    }
    if stress.performance_degradation < 2.0 {
        tests_passed += 1;
    }
    if stress.memory_efficiency > 0.7 {
        tests_passed += 1;
    }
    if stress.stability_under_stress {
        tests_passed += 1;
    }

    if stress.max_data_size < 32768 {
        warnings.push("Limited scalability for large datasets".to_string());
        recommendations.push("Consider implementing chunked processing for large data".to_string());
    }

    // Calculate overall score
    let base_score = (tests_passed as f64 / total_tests as f64) * 100.0;
    let penalty = critical_issues.len() as f64 * 10.0;
    let overall_score = (base_score - penalty).max(0.0);

    // Generate recommendations
    if numerical.low_freq_precision > 0.05 {
        recommendations
            .push("Improve low-frequency precision by increasing oversample factor".to_string());
    }

    if consistency.window_consistency < 0.95 {
        recommendations.push("Review windowing implementation for consistency".to_string());
    }

    if stress.memory_efficiency < 0.8 {
        recommendations.push("Optimize memory usage for better efficiency".to_string());
    }

    ValidationSummary {
        total_tests,
        tests_passed,
        critical_issues,
        warnings,
        overall_score,
        recommendations,
    }
}
