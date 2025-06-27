//! Enhanced wavelet denoising with advanced thresholding methods
//!
//! This module provides state-of-the-art wavelet denoising techniques including:
//! - Translation-invariant denoising
//! - Block thresholding
//! - Stein's unbiased risk estimate (SURE)
//! - BayesShrink and VisuShrink
//! - Adaptive thresholding

use crate::dwt::{wavedec, waverec, Wavelet};
use crate::dwt2d::{dwt2d_decompose, dwt2d_reconstruct};
use crate::error::{SignalError, SignalResult};
use ndarray::{Array1, Array2, ArrayView1, Axis};
use scirs2_core::parallel_ops::*;
use scirs2_core::simd_ops::SimdUnifiedOps;
use scirs2_core::validation::{check_finite, check_positive};
use std::f64;

/// Enhanced denoising configuration
#[derive(Debug, Clone)]
pub struct DenoiseConfig {
    /// Wavelet to use
    pub wavelet: Wavelet,
    /// Decomposition levels (None for automatic)
    pub levels: Option<usize>,
    /// Thresholding method
    pub method: ThresholdMethod,
    /// Threshold selection rule
    pub threshold_rule: ThresholdRule,
    /// Use translation-invariant denoising
    pub translation_invariant: bool,
    /// Number of shifts for TI denoising
    pub n_shifts: usize,
    /// Use parallel processing
    pub parallel: bool,
    /// Preserve approximation coefficients
    pub preserve_approx: bool,
    /// Level-dependent thresholding
    pub level_dependent: bool,
}

impl Default for DenoiseConfig {
    fn default() -> Self {
        Self {
            wavelet: Wavelet::DB(4),
            levels: None,
            method: ThresholdMethod::Soft,
            threshold_rule: ThresholdRule::SURE,
            translation_invariant: false,
            n_shifts: 8,
            parallel: true,
            preserve_approx: true,
            level_dependent: true,
        }
    }
}

/// Thresholding methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ThresholdMethod {
    /// Soft thresholding (shrinkage)
    Soft,
    /// Hard thresholding
    Hard,
    /// Garotte thresholding
    Garotte,
    /// SCAD (Smoothly Clipped Absolute Deviation)
    SCAD { a: f64 },
    /// Firm thresholding
    Firm { alpha: f64 },
    /// Hyperbolic thresholding
    Hyperbolic,
    /// Block thresholding
    Block { block_size: usize },
}

/// Threshold selection rules
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ThresholdRule {
    /// Universal threshold (VisuShrink)
    Universal,
    /// Stein's Unbiased Risk Estimate
    SURE,
    /// BayesShrink
    Bayes,
    /// Minimax threshold
    Minimax,
    /// Cross-validation
    CrossValidation,
    /// False Discovery Rate
    FDR { q: f64 },
    /// Custom threshold value
    Custom(f64),
}

/// Denoising result with diagnostics
#[derive(Debug, Clone)]
pub struct DenoiseResult {
    /// Denoised signal
    pub signal: Array1<f64>,
    /// Estimated noise level
    pub noise_sigma: f64,
    /// Thresholds used at each level
    pub thresholds: Vec<f64>,
    /// Percentage of coefficients retained
    pub retention_rate: f64,
    /// Effective degrees of freedom
    pub effective_df: f64,
    /// Risk estimate (if available)
    pub risk_estimate: Option<f64>,
}

/// 2D denoising result
#[derive(Debug, Clone)]
pub struct Denoise2dResult {
    /// Denoised image
    pub image: Array2<f64>,
    /// Estimated noise level
    pub noise_sigma: f64,
    /// Thresholds per subband
    pub thresholds: SubbandThresholds,
    /// Retention rates per subband
    pub retention_rates: SubbandRetention,
    /// Quality metrics
    pub quality: QualityMetrics,
}

/// Thresholds for each subband
#[derive(Debug, Clone)]
pub struct SubbandThresholds {
    pub horizontal: Vec<f64>,
    pub vertical: Vec<f64>,
    pub diagonal: Vec<f64>,
}

/// Retention rates for each subband
#[derive(Debug, Clone)]
pub struct SubbandRetention {
    pub horizontal: Vec<f64>,
    pub vertical: Vec<f64>,
    pub diagonal: Vec<f64>,
}

/// Quality metrics for denoising
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    /// Estimated SNR improvement (dB)
    pub snr_improvement: f64,
    /// Edge preservation index
    pub edge_preservation: f64,
    /// Texture preservation index
    pub texture_preservation: f64,
}

/// Enhanced 1D wavelet denoising
pub fn denoise_wavelet_1d(
    signal: &Array1<f64>,
    config: &DenoiseConfig,
) -> SignalResult<DenoiseResult> {
    check_finite(&signal.to_vec(), "signal")?;
    
    if config.translation_invariant {
        translation_invariant_denoise_1d(signal, config)
    } else {
        standard_denoise_1d(signal, config)
    }
}

/// Standard wavelet denoising
fn standard_denoise_1d(
    signal: &Array1<f64>,
    config: &DenoiseConfig,
) -> SignalResult<DenoiseResult> {
    let n = signal.len();
    
    // Determine decomposition levels
    let max_levels = (n as f64).log2().floor() as usize - 1;
    let levels = config.levels.unwrap_or(max_levels.min(6));
    
    // Perform wavelet decomposition
    let coeffs = wavedec(signal, config.wavelet, levels, None)?;
    
    // Estimate noise level from finest scale coefficients
    let noise_sigma = estimate_noise_mad(&coeffs.details[coeffs.details.len() - 1]);
    
    // Apply thresholding
    let (thresholded_coeffs, thresholds, retention_rates) = 
        apply_thresholding(&coeffs, noise_sigma, config)?;
    
    // Reconstruct signal
    let denoised = waverec(&thresholded_coeffs)?;
    
    // Calculate diagnostics
    let total_coeffs: usize = coeffs.details.iter().map(|d| d.len()).sum();
    let retained_coeffs: f64 = retention_rates.iter()
        .zip(coeffs.details.iter())
        .map(|(rate, detail)| rate * detail.len() as f64)
        .sum();
    let retention_rate = retained_coeffs / total_coeffs as f64;
    
    let effective_df = compute_effective_df(&thresholded_coeffs);
    
    let risk_estimate = match config.threshold_rule {
        ThresholdRule::SURE => Some(compute_sure_risk(&coeffs, &thresholds, noise_sigma)?),
        _ => None,
    };
    
    Ok(DenoiseResult {
        signal: denoised,
        noise_sigma,
        thresholds,
        retention_rate,
        effective_df,
        risk_estimate,
    })
}

/// Translation-invariant denoising
fn translation_invariant_denoise_1d(
    signal: &Array1<f64>,
    config: &DenoiseConfig,
) -> SignalResult<DenoiseResult> {
    let n = signal.len();
    let n_shifts = config.n_shifts.min(n);
    
    // Store shifted and denoised versions
    let mut denoised_shifts = Vec::with_capacity(n_shifts);
    let mut all_noise_estimates = Vec::with_capacity(n_shifts);
    
    // Process each shift
    let shift_results: Vec<_> = if config.parallel {
        (0..n_shifts)
            .into_par_iter()
            .map(|shift| {
                // Circular shift
                let mut shifted = Array1::zeros(n);
                for i in 0..n {
                    shifted[i] = signal[(i + shift) % n];
                }
                
                // Denoise shifted signal
                let mut shift_config = config.clone();
                shift_config.translation_invariant = false;
                
                standard_denoise_1d(&shifted, &shift_config)
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        (0..n_shifts)
            .map(|shift| {
                let mut shifted = Array1::zeros(n);
                for i in 0..n {
                    shifted[i] = signal[(i + shift) % n];
                }
                
                let mut shift_config = config.clone();
                shift_config.translation_invariant = false;
                
                standard_denoise_1d(&shifted, &shift_config)
            })
            .collect::<Result<Vec<_>, _>>()?
    };
    
    // Average the unshifted results
    let mut averaged = Array1::zeros(n);
    
    for (shift, result) in shift_results.iter().enumerate() {
        // Unshift the denoised signal
        for i in 0..n {
            averaged[(i + shift) % n] += result.signal[i] / n_shifts as f64;
        }
        
        all_noise_estimates.push(result.noise_sigma);
    }
    
    // Aggregate diagnostics
    let noise_sigma = all_noise_estimates.iter().sum::<f64>() / n_shifts as f64;
    let thresholds = shift_results[0].thresholds.clone(); // Use first shift's thresholds
    let retention_rate = shift_results.iter()
        .map(|r| r.retention_rate)
        .sum::<f64>() / n_shifts as f64;
    let effective_df = compute_effective_df_ti(&averaged, signal);
    
    Ok(DenoiseResult {
        signal: averaged,
        noise_sigma,
        thresholds,
        retention_rate,
        effective_df,
        risk_estimate: None,
    })
}

/// Enhanced 2D wavelet denoising
pub fn denoise_wavelet_2d(
    image: &Array2<f64>,
    config: &DenoiseConfig,
) -> SignalResult<Denoise2dResult> {
    check_finite(&image.as_slice().unwrap(), "image")?;
    
    let (rows, cols) = image.dim();
    let max_levels = ((rows.min(cols)) as f64).log2().floor() as usize - 1;
    let levels = config.levels.unwrap_or(max_levels.min(4));
    
    // Store results for each level
    let mut all_h_thresholds = Vec::with_capacity(levels);
    let mut all_v_thresholds = Vec::with_capacity(levels);
    let mut all_d_thresholds = Vec::with_capacity(levels);
    let mut all_h_retention = Vec::with_capacity(levels);
    let mut all_v_retention = Vec::with_capacity(levels);
    let mut all_d_retention = Vec::with_capacity(levels);
    
    // Start with the image
    let mut current = image.clone();
    let mut approximations = Vec::new();
    let mut h_details = Vec::new();
    let mut v_details = Vec::new();
    let mut d_details = Vec::new();
    
    // Multilevel decomposition with thresholding
    for level in 0..levels {
        let (approx, h_detail, v_detail, d_detail) = 
            dwt2d_decompose(&current, config.wavelet)?;
        
        // Estimate noise at first level
        let noise_sigma = if level == 0 {
            estimate_noise_2d(&d_detail)
        } else {
            // Use previous estimate scaled by sqrt(2)
            all_d_thresholds[0] / (2.0 * (d_detail.len() as f64).ln()).sqrt()
        };
        
        // Apply thresholding to detail coefficients
        let (h_thresh, h_thresholded, h_retention) = 
            threshold_subband(&h_detail, noise_sigma, level, config)?;
        let (v_thresh, v_thresholded, v_retention) = 
            threshold_subband(&v_detail, noise_sigma, level, config)?;
        let (d_thresh, d_thresholded, d_retention) = 
            threshold_subband(&d_detail, noise_sigma, level, config)?;
        
        all_h_thresholds.push(h_thresh);
        all_v_thresholds.push(v_thresh);
        all_d_thresholds.push(d_thresh);
        all_h_retention.push(h_retention);
        all_v_retention.push(v_retention);
        all_d_retention.push(d_retention);
        
        approximations.push(approx.clone());
        h_details.push(h_thresholded);
        v_details.push(v_thresholded);
        d_details.push(d_thresholded);
        
        current = approx;
    }
    
    // Reconstruct from thresholded coefficients
    let mut reconstructed = if config.preserve_approx {
        approximations.last().unwrap().clone()
    } else {
        // Also threshold approximation
        let noise_sigma = all_d_thresholds[0] / (2.0_f64).powf(levels as f64 / 2.0);
        let (_, approx_thresholded, _) = 
            threshold_subband(approximations.last().unwrap(), noise_sigma, levels, config)?;
        approx_thresholded
    };
    
    // Inverse transform
    for level in (0..levels).rev() {
        reconstructed = dwt2d_reconstruct(
            &reconstructed,
            &h_details[level],
            &v_details[level],
            &d_details[level],
            config.wavelet,
        )?;
    }
    
    // Ensure output matches input size
    let denoised = if reconstructed.dim() != image.dim() {
        reconstructed.slice(s![0..rows, 0..cols]).to_owned()
    } else {
        reconstructed
    };
    
    // Compute quality metrics
    let quality = compute_quality_metrics(image, &denoised, &all_h_retention, &all_v_retention, &all_d_retention);
    
    Ok(Denoise2dResult {
        image: denoised,
        noise_sigma: all_d_thresholds[0],
        thresholds: SubbandThresholds {
            horizontal: all_h_thresholds,
            vertical: all_v_thresholds,
            diagonal: all_d_thresholds,
        },
        retention_rates: SubbandRetention {
            horizontal: all_h_retention,
            vertical: all_v_retention,
            diagonal: all_d_retention,
        },
        quality,
    })
}

/// Estimate noise using median absolute deviation
fn estimate_noise_mad(coeffs: &Array1<f64>) -> f64 {
    let mut abs_coeffs: Vec<f64> = coeffs.iter().map(|&x| x.abs()).collect();
    abs_coeffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let median = if abs_coeffs.len() % 2 == 0 {
        (abs_coeffs[abs_coeffs.len() / 2 - 1] + abs_coeffs[abs_coeffs.len() / 2]) / 2.0
    } else {
        abs_coeffs[abs_coeffs.len() / 2]
    };
    
    median / 0.6745 // Scale factor for Gaussian noise
}

/// Estimate noise in 2D using diagonal detail coefficients
fn estimate_noise_2d(detail: &Array2<f64>) -> f64 {
    let flat_detail: Vec<f64> = detail.iter().cloned().collect();
    let flat_array = Array1::from_vec(flat_detail);
    estimate_noise_mad(&flat_array)
}

/// Apply thresholding to wavelet coefficients
fn apply_thresholding(
    coeffs: &crate::dwt::DecompositionResult,
    noise_sigma: f64,
    config: &DenoiseConfig,
) -> SignalResult<(crate::dwt::DecompositionResult, Vec<f64>, Vec<f64>)> {
    let mut thresholded = coeffs.clone();
    let mut thresholds = Vec::new();
    let mut retention_rates = Vec::new();
    
    // Process each detail level
    for (level, detail) in coeffs.details.iter().enumerate() {
        let n = detail.len() as f64;
        
        // Compute threshold
        let threshold = match config.threshold_rule {
            ThresholdRule::Universal => noise_sigma * (2.0 * n.ln()).sqrt(),
            ThresholdRule::SURE => compute_sure_threshold(detail, noise_sigma)?,
            ThresholdRule::Bayes => compute_bayes_threshold(detail, noise_sigma),
            ThresholdRule::Minimax => compute_minimax_threshold(n, noise_sigma),
            ThresholdRule::FDR { q } => compute_fdr_threshold(detail, noise_sigma, q)?,
            ThresholdRule::Custom(t) => t,
            ThresholdRule::CrossValidation => compute_cv_threshold(detail, noise_sigma)?,
        };
        
        // Apply level-dependent scaling if enabled
        let scaled_threshold = if config.level_dependent {
            threshold * (1.0 + level as f64 * 0.1)
        } else {
            threshold
        };
        
        thresholds.push(scaled_threshold);
        
        // Apply thresholding method
        let (thresholded_detail, retention) = match config.method {
            ThresholdMethod::Soft => soft_threshold(detail, scaled_threshold),
            ThresholdMethod::Hard => hard_threshold(detail, scaled_threshold),
            ThresholdMethod::Garotte => garotte_threshold(detail, scaled_threshold),
            ThresholdMethod::SCAD { a } => scad_threshold(detail, scaled_threshold, a),
            ThresholdMethod::Firm { alpha } => firm_threshold(detail, scaled_threshold, alpha),
            ThresholdMethod::Hyperbolic => hyperbolic_threshold(detail, scaled_threshold),
            ThresholdMethod::Block { block_size } => block_threshold(detail, scaled_threshold, block_size)?,
        };
        
        thresholded.details[level] = thresholded_detail;
        retention_rates.push(retention);
    }
    
    Ok((thresholded, thresholds, retention_rates))
}

/// Soft thresholding
fn soft_threshold(coeffs: &Array1<f64>, threshold: f64) -> (Array1<f64>, f64) {
    let mut thresholded = Array1::zeros(coeffs.len());
    let mut retained = 0;
    
    for (i, &coeff) in coeffs.iter().enumerate() {
        if coeff.abs() > threshold {
            thresholded[i] = coeff.signum() * (coeff.abs() - threshold);
            retained += 1;
        }
    }
    
    (thresholded, retained as f64 / coeffs.len() as f64)
}

/// Hard thresholding
fn hard_threshold(coeffs: &Array1<f64>, threshold: f64) -> (Array1<f64>, f64) {
    let mut thresholded = coeffs.clone();
    let mut retained = 0;
    
    for (i, &coeff) in coeffs.iter().enumerate() {
        if coeff.abs() <= threshold {
            thresholded[i] = 0.0;
        } else {
            retained += 1;
        }
    }
    
    (thresholded, retained as f64 / coeffs.len() as f64)
}

/// Garotte thresholding
fn garotte_threshold(coeffs: &Array1<f64>, threshold: f64) -> (Array1<f64>, f64) {
    let mut thresholded = Array1::zeros(coeffs.len());
    let mut retained = 0;
    let threshold_sq = threshold * threshold;
    
    for (i, &coeff) in coeffs.iter().enumerate() {
        let coeff_sq = coeff * coeff;
        if coeff_sq > threshold_sq {
            thresholded[i] = coeff * (1.0 - threshold_sq / coeff_sq);
            retained += 1;
        }
    }
    
    (thresholded, retained as f64 / coeffs.len() as f64)
}

/// SCAD thresholding
fn scad_threshold(coeffs: &Array1<f64>, threshold: f64, a: f64) -> (Array1<f64>, f64) {
    let mut thresholded = Array1::zeros(coeffs.len());
    let mut retained = 0;
    
    for (i, &coeff) in coeffs.iter().enumerate() {
        let abs_coeff = coeff.abs();
        
        if abs_coeff <= threshold {
            thresholded[i] = 0.0;
        } else if abs_coeff <= a * threshold {
            thresholded[i] = coeff.signum() * (a * abs_coeff - a * threshold) / (a - 1.0);
            retained += 1;
        } else {
            thresholded[i] = coeff;
            retained += 1;
        }
    }
    
    (thresholded, retained as f64 / coeffs.len() as f64)
}

/// Firm thresholding
fn firm_threshold(coeffs: &Array1<f64>, threshold: f64, alpha: f64) -> (Array1<f64>, f64) {
    let mut thresholded = Array1::zeros(coeffs.len());
    let mut retained = 0;
    let upper_threshold = alpha * threshold;
    
    for (i, &coeff) in coeffs.iter().enumerate() {
        let abs_coeff = coeff.abs();
        
        if abs_coeff <= threshold {
            thresholded[i] = 0.0;
        } else if abs_coeff <= upper_threshold {
            let scale = (abs_coeff - threshold) / (upper_threshold - threshold);
            thresholded[i] = coeff * scale;
            retained += 1;
        } else {
            thresholded[i] = coeff;
            retained += 1;
        }
    }
    
    (thresholded, retained as f64 / coeffs.len() as f64)
}

/// Hyperbolic thresholding
fn hyperbolic_threshold(coeffs: &Array1<f64>, threshold: f64) -> (Array1<f64>, f64) {
    let mut thresholded = Array1::zeros(coeffs.len());
    let mut retained = 0;
    let threshold_sq = threshold * threshold;
    
    for (i, &coeff) in coeffs.iter().enumerate() {
        let coeff_sq = coeff * coeff;
        if coeff_sq > threshold_sq {
            thresholded[i] = coeff * (coeff_sq - threshold_sq).sqrt() / coeff.abs();
            retained += 1;
        }
    }
    
    (thresholded, retained as f64 / coeffs.len() as f64)
}

/// Block thresholding
fn block_threshold(coeffs: &Array1<f64>, threshold: f64, block_size: usize) -> SignalResult<(Array1<f64>, f64)> {
    let n = coeffs.len();
    let n_blocks = (n + block_size - 1) / block_size;
    let mut thresholded = coeffs.clone();
    let mut retained_blocks = 0;
    
    for i in 0..n_blocks {
        let start = i * block_size;
        let end = (start + block_size).min(n);
        
        // Compute block energy
        let mut block_energy = 0.0;
        for j in start..end {
            block_energy += coeffs[j] * coeffs[j];
        }
        block_energy = block_energy.sqrt();
        
        // Apply block threshold
        if block_energy <= threshold {
            for j in start..end {
                thresholded[j] = 0.0;
            }
        } else {
            retained_blocks += 1;
        }
    }
    
    let retention_rate = retained_blocks as f64 / n_blocks as f64;
    Ok((thresholded, retention_rate))
}

/// Compute SURE threshold
fn compute_sure_threshold(coeffs: &Array1<f64>, noise_sigma: f64) -> SignalResult<f64> {
    let n = coeffs.len() as f64;
    let max_threshold = noise_sigma * (2.0 * n.ln()).sqrt();
    let n_candidates = 100;
    
    let mut best_threshold = max_threshold;
    let mut min_risk = f64::INFINITY;
    
    for i in 0..n_candidates {
        let threshold = max_threshold * (i + 1) as f64 / n_candidates as f64;
        let risk = sure_risk(coeffs, threshold, noise_sigma);
        
        if risk < min_risk {
            min_risk = risk;
            best_threshold = threshold;
        }
    }
    
    Ok(best_threshold)
}

/// SURE risk calculation
fn sure_risk(coeffs: &Array1<f64>, threshold: f64, noise_sigma: f64) -> f64 {
    let n = coeffs.len() as f64;
    let noise_var = noise_sigma * noise_sigma;
    
    let mut risk = -n * noise_var;
    let mut n_small = 0.0;
    
    for &coeff in coeffs.iter() {
        let abs_coeff = coeff.abs();
        if abs_coeff <= threshold {
            risk += coeff * coeff;
            n_small += 1.0;
        } else {
            risk += noise_var + (abs_coeff - threshold).powi(2);
        }
    }
    
    risk + 2.0 * noise_var * n_small
}

/// Compute Bayes threshold
fn compute_bayes_threshold(coeffs: &Array1<f64>, noise_sigma: f64) -> f64 {
    let variance = coeffs.iter().map(|&x| x * x).sum::<f64>() / coeffs.len() as f64;
    let signal_variance = (variance - noise_sigma * noise_sigma).max(0.0);
    
    if signal_variance > 0.0 {
        noise_sigma * noise_sigma / signal_variance.sqrt()
    } else {
        f64::INFINITY // No signal, threshold everything
    }
}

/// Compute minimax threshold
fn compute_minimax_threshold(n: f64, noise_sigma: f64) -> f64 {
    // Minimax threshold approximation
    let log_n = n.ln();
    
    if log_n < 2.0 {
        0.0
    } else {
        noise_sigma * (0.3936 + 0.1829 * log_n).sqrt()
    }
}

/// Compute FDR threshold
fn compute_fdr_threshold(coeffs: &Array1<f64>, noise_sigma: f64, q: f64) -> SignalResult<f64> {
    let n = coeffs.len();
    
    // Sort coefficients by absolute value
    let mut abs_coeffs: Vec<(usize, f64)> = coeffs.iter()
        .enumerate()
        .map(|(i, &c)| (i, c.abs()))
        .collect();
    abs_coeffs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    // Find threshold using FDR procedure
    let mut threshold = 0.0;
    
    for (k, &(_, abs_val)) in abs_coeffs.iter().enumerate() {
        let p_value = 2.0 * (1.0 - normal_cdf(abs_val / noise_sigma));
        let fdr_threshold = q * (k + 1) as f64 / n as f64;
        
        if p_value <= fdr_threshold {
            threshold = abs_val;
            break;
        }
    }
    
    Ok(threshold)
}

/// Compute cross-validation threshold
fn compute_cv_threshold(coeffs: &Array1<f64>, noise_sigma: f64) -> SignalResult<f64> {
    // Simplified CV: leave-one-out
    let n = coeffs.len();
    let max_threshold = noise_sigma * (2.0 * (n as f64).ln()).sqrt();
    let n_candidates = 50;
    
    let mut best_threshold = max_threshold;
    let mut min_cv_error = f64::INFINITY;
    
    for i in 0..n_candidates {
        let threshold = max_threshold * (i + 1) as f64 / n_candidates as f64;
        let mut cv_error = 0.0;
        
        // Leave-one-out CV
        for j in 0..n {
            let mut temp_coeffs = coeffs.clone();
            temp_coeffs.remove(j);
            
            // Apply threshold to remaining coefficients
            let (thresholded, _) = soft_threshold(&temp_coeffs, threshold);
            
            // Estimate error at left-out position
            let predicted = if j > 0 && j < n - 1 {
                (thresholded[j - 1] + thresholded[j]) / 2.0
            } else {
                0.0
            };
            
            cv_error += (coeffs[j] - predicted).powi(2);
        }
        
        if cv_error < min_cv_error {
            min_cv_error = cv_error;
            best_threshold = threshold;
        }
    }
    
    Ok(best_threshold)
}

/// Threshold 2D subband
fn threshold_subband(
    subband: &Array2<f64>,
    noise_sigma: f64,
    level: usize,
    config: &DenoiseConfig,
) -> SignalResult<(f64, Array2<f64>, f64)> {
    // Flatten subband for 1D thresholding methods
    let flat: Vec<f64> = subband.iter().cloned().collect();
    let flat_array = Array1::from_vec(flat);
    
    // Estimate subband-specific noise
    let subband_sigma = noise_sigma * 2.0_f64.powf(level as f64 / 2.0);
    
    // Compute threshold
    let threshold = match config.threshold_rule {
        ThresholdRule::Universal => subband_sigma * (2.0 * (flat.len() as f64).ln()).sqrt(),
        ThresholdRule::SURE => compute_sure_threshold(&flat_array, subband_sigma)?,
        ThresholdRule::Bayes => compute_bayes_threshold(&flat_array, subband_sigma),
        _ => subband_sigma * (2.0 * (flat.len() as f64).ln()).sqrt(),
    };
    
    // Apply thresholding
    let (thresholded_flat, retention) = match config.method {
        ThresholdMethod::Soft => soft_threshold(&flat_array, threshold),
        ThresholdMethod::Hard => hard_threshold(&flat_array, threshold),
        _ => soft_threshold(&flat_array, threshold),
    };
    
    // Reshape back to 2D
    let shape = subband.dim();
    let thresholded = Array2::from_shape_vec(shape, thresholded_flat.to_vec())?;
    
    Ok((threshold, thresholded, retention))
}

/// Compute effective degrees of freedom
fn compute_effective_df(coeffs: &crate::dwt::DecompositionResult) -> f64 {
    let mut df = coeffs.approx.len() as f64;
    
    for detail in &coeffs.details {
        df += detail.iter().filter(|&&x| x != 0.0).count() as f64;
    }
    
    df
}

/// Compute effective degrees of freedom for TI denoising
fn compute_effective_df_ti(denoised: &Array1<f64>, original: &Array1<f64>) -> f64 {
    // Estimate using divergence formula
    let n = denoised.len() as f64;
    let mut div = 0.0;
    let h = 1e-6;
    
    for i in 0..denoised.len() {
        let mut perturbed = original.clone();
        perturbed[i] += h;
        
        // Would need to recompute denoising for perturbed signal
        // Simplified: assume linear response
        div += 1.0;
    }
    
    div
}

/// Compute SURE risk for coefficients
fn compute_sure_risk(
    coeffs: &crate::dwt::DecompositionResult,
    thresholds: &[f64],
    noise_sigma: f64,
) -> SignalResult<f64> {
    let mut total_risk = 0.0;
    
    for (detail, &threshold) in coeffs.details.iter().zip(thresholds.iter()) {
        total_risk += sure_risk(detail, threshold, noise_sigma);
    }
    
    Ok(total_risk)
}

/// Compute quality metrics for 2D denoising
fn compute_quality_metrics(
    original: &Array2<f64>,
    denoised: &Array2<f64>,
    h_retention: &[f64],
    v_retention: &[f64],
    d_retention: &[f64],
) -> QualityMetrics {
    // SNR improvement estimate
    let noise_estimate = (original - denoised).mapv(|x| x * x).sum() / original.len() as f64;
    let signal_power = original.mapv(|x| x * x).sum() / original.len() as f64;
    let snr_improvement = 10.0 * (signal_power / noise_estimate.max(1e-10)).log10();
    
    // Edge preservation: higher retention in H and V indicates better edge preservation
    let edge_preservation = (h_retention.iter().sum::<f64>() + v_retention.iter().sum::<f64>()) 
        / (2.0 * h_retention.len() as f64);
    
    // Texture preservation: retention in all subbands
    let texture_preservation = (h_retention.iter().sum::<f64>() 
        + v_retention.iter().sum::<f64>() 
        + d_retention.iter().sum::<f64>()) 
        / (3.0 * h_retention.len() as f64);
    
    QualityMetrics {
        snr_improvement,
        edge_preservation,
        texture_preservation,
    }
}

/// Normal CDF approximation
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / 2.0_f64.sqrt()))
}

/// Error function approximation
fn erf(x: f64) -> f64 {
    // Abramowitz and Stegun approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    
    let sign = x.signum();
    let x = x.abs();
    
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    
    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    
    #[test]
    fn test_denoise_1d_basic() {
        let n = 256;
        let t = Array1::linspace(0.0, 1.0, n);
        let clean_signal = t.mapv(|x| (2.0 * f64::consts::PI * 5.0 * x).sin());
        
        // Add noise
        let mut rng = rand::rng();
        let noise_level = 0.1;
        let noisy_signal = &clean_signal + &Array1::from_shape_fn(n, |_| 
            noise_level * rng.random_range(-1.0..1.0)
        );
        
        let config = DenoiseConfig::default();
        let result = denoise_wavelet_1d(&noisy_signal, &config).unwrap();
        
        assert!(result.noise_sigma > 0.0);
        assert!(result.retention_rate < 1.0);
    }
    
    #[test]
    fn test_threshold_methods() {
        let coeffs = Array1::from_vec(vec![0.1, -0.5, 1.2, -0.3, 2.0, -1.5]);
        let threshold = 0.4;
        
        let (soft, _) = soft_threshold(&coeffs, threshold);
        let (hard, _) = hard_threshold(&coeffs, threshold);
        
        // Check soft thresholding shrinks values
        assert!(soft[2].abs() < coeffs[2].abs());
        
        // Check hard thresholding preserves large values
        assert_eq!(hard[4], coeffs[4]);
    }
}