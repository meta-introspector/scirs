//! Enhanced multitaper spectral estimation with SIMD and parallel processing
//!
//! This module provides high-performance implementations of multitaper spectral
//! estimation using scirs2-core's SIMD and parallel processing capabilities.

use super::windows::dpss;
use crate::error::{SignalError, SignalResult};
use ndarray::{Array1, Array2, ArrayView1, Axis};
use num_complex::Complex64;
use num_traits::{Float, NumCast};
use scirs2_core::parallel_ops::*;
use scirs2_core::simd_ops::{PlatformCapabilities, SimdUnifiedOps};
use scirs2_core::validation::{check_finite, check_positive};
use std::fmt::Debug;
use std::sync::Arc;

/// Enhanced multitaper PSD result with additional statistics
#[derive(Debug, Clone)]
pub struct EnhancedMultitaperResult {
    /// Frequencies
    pub frequencies: Vec<f64>,
    /// Power spectral density
    pub psd: Vec<f64>,
    /// Confidence intervals (if requested)
    pub confidence_intervals: Option<(Vec<f64>, Vec<f64>)>,
    /// Effective degrees of freedom
    pub dof: Option<f64>,
    /// DPSS tapers used (if requested)
    pub tapers: Option<Array2<f64>>,
    /// Eigenvalues (if requested)
    pub eigenvalues: Option<Array1<f64>>,
}

/// Configuration for enhanced multitaper estimation
#[derive(Debug, Clone)]
pub struct MultitaperConfig {
    /// Sampling frequency
    pub fs: f64,
    /// Time-bandwidth product
    pub nw: f64,
    /// Number of tapers
    pub k: usize,
    /// FFT length
    pub nfft: Option<usize>,
    /// Return one-sided spectrum
    pub onesided: bool,
    /// Use adaptive weighting
    pub adaptive: bool,
    /// Compute confidence intervals
    pub confidence: Option<f64>,
    /// Return tapers and eigenvalues
    pub return_tapers: bool,
    /// Use parallel processing
    pub parallel: bool,
    /// Minimum chunk size for parallel processing
    pub parallel_threshold: usize,
}

impl Default for MultitaperConfig {
    fn default() -> Self {
        Self {
            fs: 1.0,
            nw: 4.0,
            k: 7, // 2*nw - 1
            nfft: None,
            onesided: true,
            adaptive: true,
            confidence: None,
            return_tapers: false,
            parallel: true,
            parallel_threshold: 1024,
        }
    }
}

/// Enhanced multitaper power spectral density estimation with SIMD and parallel processing
///
/// This function provides a high-performance implementation of the multitaper method
/// using scirs2-core's acceleration capabilities.
///
/// # Arguments
///
/// * `x` - Input signal
/// * `config` - Multitaper configuration
///
/// # Returns
///
/// * Enhanced multitaper result with PSD and optional statistics
///
/// # Examples
///
/// ```
/// use scirs2_signal::multitaper::enhanced::{enhanced_pmtm, MultitaperConfig};
/// use std::f64::consts::PI;
///
/// // Generate test signal
/// let n = 1024;
/// let fs = 100.0;
/// let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
/// use rand::Rng;
/// let mut rng = rand::rng();
/// let signal: Vec<f64> = t.iter()
///     .map(|&ti| (2.0 * PI * 10.0 * ti).sin() + 0.1 * rng.random_range(0.0..1.0))
///     .collect();
///
/// // Configure multitaper estimation
/// let config = MultitaperConfig {
///     fs,
///     nw: 4.0,
///     k: 7,
///     confidence: Some(0.95),
///     ..Default::default()
/// };
///
/// // Compute enhanced multitaper PSD
/// let result = enhanced_pmtm(&signal, &config).unwrap();
/// assert!(result.frequencies.len() > 0);
/// assert!(result.confidence_intervals.is_some());
/// ```
pub fn enhanced_pmtm<T>(x: &[T], config: &MultitaperConfig) -> SignalResult<EnhancedMultitaperResult>
where
    T: Float + NumCast + Debug + Send + Sync,
{
    // Validate input
    if x.is_empty() {
        return Err(SignalError::ValueError("Input signal is empty".to_string()));
    }
    
    check_positive(config.nw, "nw")?;
    check_positive(config.k, "k")?;
    check_positive(config.fs, "fs")?;
    
    // Convert input to f64
    let x_f64: Vec<f64> = x
        .iter()
        .map(|&val| {
            NumCast::from(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<f64>>>()?;
    
    check_finite(&x_f64, "signal")?;
    
    let n = x_f64.len();
    let nfft = config.nfft.unwrap_or(next_power_of_two(n));
    
    // Compute DPSS tapers
    let (tapers, eigenvalues_opt) = dpss(n, config.nw, config.k, true)?;
    
    let eigenvalues = eigenvalues_opt.ok_or_else(|| {
        SignalError::ComputationError("Eigenvalues required but not returned from dpss".to_string())
    })?;
    
    // Compute tapered FFTs using parallel processing if enabled
    let spectra = if config.parallel && n >= config.parallel_threshold {
        compute_tapered_ffts_parallel(&x_f64, &tapers, nfft)?
    } else {
        compute_tapered_ffts_simd(&x_f64, &tapers, nfft)?
    };
    
    // Combine spectra using adaptive or standard weighting
    let (frequencies, psd) = if config.adaptive {
        combine_spectra_adaptive(&spectra, &eigenvalues, config.fs, nfft, config.onesided)?
    } else {
        combine_spectra_standard(&spectra, &eigenvalues, config.fs, nfft, config.onesided)?
    };
    
    // Compute confidence intervals if requested
    let confidence_intervals = if let Some(confidence_level) = config.confidence {
        Some(compute_confidence_intervals(&spectra, &eigenvalues, confidence_level)?)
    } else {
        None
    };
    
    // Compute effective degrees of freedom
    let dof = Some(compute_effective_dof(&eigenvalues));
    
    Ok(EnhancedMultitaperResult {
        frequencies,
        psd,
        confidence_intervals,
        dof,
        tapers: if config.return_tapers { Some(tapers) } else { None },
        eigenvalues: if config.return_tapers { Some(eigenvalues) } else { None },
    })
}

/// Compute tapered FFTs using SIMD operations
fn compute_tapered_ffts_simd(
    signal: &[f64],
    tapers: &Array2<f64>,
    nfft: usize,
) -> SignalResult<Array2<f64>> {
    let k = tapers.nrows();
    let n = signal.len();
    let mut spectra = Array2::zeros((k, nfft));
    
    // Get SIMD capabilities
    let _caps = PlatformCapabilities::detect();
    
    for i in 0..k {
        // Apply taper using SIMD operations
        let taper_view = tapers.row(i);
        let signal_view = ArrayView1::from(signal);
        
        // Use SIMD multiplication for tapering
        let mut tapered = vec![0.0; n];
        let tapered_view = ArrayView1::from_shape(n, &mut tapered).unwrap();
        
        // SIMD element-wise multiplication
        f64::simd_mul(&signal_view, &taper_view, &tapered_view);
        
        // Compute FFT (using enhanced FFT with SIMD)
        let spectrum = simd_fft(&tapered, nfft)?;
        
        // Store power spectrum
        for (j, &val) in spectrum.iter().enumerate() {
            spectra[[i, j]] = val.norm_sqr();
        }
    }
    
    Ok(spectra)
}

/// Compute tapered FFTs using parallel processing
fn compute_tapered_ffts_parallel(
    signal: &[f64],
    tapers: &Array2<f64>,
    nfft: usize,
) -> SignalResult<Array2<f64>> {
    let k = tapers.nrows();
    let n = signal.len();
    let signal_arc = Arc::new(signal.to_vec());
    
    // Process tapers in parallel
    let results: Vec<Vec<f64>> = (0..k)
        .into_par_iter()
        .map(|i| {
            let signal_ref = signal_arc.clone();
            let taper = tapers.row(i).to_owned();
            
            // Apply taper
            let mut tapered = vec![0.0; n];
            for j in 0..n {
                tapered[j] = signal_ref[j] * taper[j];
            }
            
            // Compute FFT
            let spectrum = simd_fft(&tapered, nfft).unwrap();
            
            // Return power spectrum
            spectrum.iter().map(|c| c.norm_sqr()).collect()
        })
        .collect();
    
    // Convert to Array2
    let mut spectra = Array2::zeros((k, nfft));
    for (i, row) in results.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            spectra[[i, j]] = val;
        }
    }
    
    Ok(spectra)
}

/// Enhanced FFT using SIMD operations
fn simd_fft(x: &[f64], nfft: usize) -> SignalResult<Vec<Complex64>> {
    // Pad or truncate to nfft
    let mut padded = vec![Complex64::new(0.0, 0.0); nfft];
    let copy_len = x.len().min(nfft);
    
    for i in 0..copy_len {
        padded[i] = Complex64::new(x[i], 0.0);
    }
    
    // Use rustfft with pre-planning for better performance
    use rustfft::{FftPlanner, num_complex::Complex};
    
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(nfft);
    let mut buffer = padded.clone();
    
    fft.process(&mut buffer);
    
    Ok(buffer)
}

/// Combine spectra using standard eigenvalue weighting
fn combine_spectra_standard(
    spectra: &Array2<f64>,
    eigenvalues: &Array1<f64>,
    fs: f64,
    nfft: usize,
    onesided: bool,
) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    let k = spectra.nrows();
    
    // Create frequency array
    let frequencies = if onesided {
        let n_freqs = nfft / 2 + 1;
        (0..n_freqs)
            .map(|i| i as f64 * fs / nfft as f64)
            .collect()
    } else {
        (0..nfft)
            .map(|i| {
                if i <= nfft / 2 {
                    i as f64 * fs / nfft as f64
                } else {
                    (i as f64 - nfft as f64) * fs / nfft as f64
                }
            })
            .collect()
    };
    
    // Combine spectra with eigenvalue weights
    let n_freqs = if onesided { nfft / 2 + 1 } else { nfft };
    let mut psd = vec![0.0; n_freqs];
    
    let weight_sum: f64 = eigenvalues.sum();
    let scaling = if onesided {
        2.0 / (fs * weight_sum)
    } else {
        1.0 / (fs * weight_sum)
    };
    
    for j in 0..n_freqs {
        let mut weighted_sum = 0.0;
        for i in 0..k {
            weighted_sum += eigenvalues[i] * spectra[[i, j]];
        }
        psd[j] = weighted_sum * scaling;
    }
    
    Ok((frequencies, psd))
}

/// Combine spectra using Thomson's adaptive weighting method
fn combine_spectra_adaptive(
    spectra: &Array2<f64>,
    eigenvalues: &Array1<f64>,
    fs: f64,
    nfft: usize,
    onesided: bool,
) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    let k = spectra.nrows();
    let n_freqs = if onesided { nfft / 2 + 1 } else { nfft };
    
    // Initialize adaptive weights
    let mut weights = Array2::zeros((k, n_freqs));
    let mut psd = vec![0.0; n_freqs];
    
    // Iterative adaptive algorithm (simplified version)
    let max_iter = 5;
    let tolerance = 1e-6;
    
    // Initialize with eigenvalue weights
    for i in 0..k {
        for j in 0..n_freqs {
            weights[[i, j]] = eigenvalues[i];
        }
    }
    
    // Adaptive iteration
    for _ in 0..max_iter {
        let old_psd = psd.clone();
        
        // Update PSD estimate
        for j in 0..n_freqs {
            let mut weighted_sum = 0.0;
            let mut weight_sum = 0.0;
            
            for i in 0..k {
                let w = weights[[i, j]];
                weighted_sum += w * spectra[[i, j]];
                weight_sum += w;
            }
            
            psd[j] = weighted_sum / weight_sum;
        }
        
        // Update weights based on current PSD estimate
        for j in 0..n_freqs {
            for i in 0..k {
                let lambda = eigenvalues[i];
                let bias_factor = 1.0 / (1.0 + (psd[j] / spectra[[i, j]]).powi(2));
                weights[[i, j]] = lambda * bias_factor;
            }
        }
        
        // Check convergence
        let max_change = old_psd.iter()
            .zip(psd.iter())
            .map(|(old, new)| ((old - new) / old.max(1e-10)).abs())
            .fold(0.0, f64::max);
            
        if max_change < tolerance {
            break;
        }
    }
    
    // Create frequency array
    let frequencies = if onesided {
        (0..n_freqs)
            .map(|i| i as f64 * fs / nfft as f64)
            .collect()
    } else {
        (0..nfft)
            .map(|i| {
                if i <= nfft / 2 {
                    i as f64 * fs / nfft as f64
                } else {
                    (i as f64 - nfft as f64) * fs / nfft as f64
                }
            })
            .collect()
    };
    
    // Apply final scaling
    let scaling = if onesided { 2.0 / fs } else { 1.0 / fs };
    psd.iter_mut().for_each(|p| *p *= scaling);
    
    Ok((frequencies, psd))
}

/// Compute confidence intervals using chi-squared approximation
fn compute_confidence_intervals(
    spectra: &Array2<f64>,
    eigenvalues: &Array1<f64>,
    confidence_level: f64,
) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    use statrs::distribution::{ChiSquared, ContinuousCDF};
    
    let k = spectra.nrows() as f64;
    let dof = 2.0 * k; // Degrees of freedom for multitaper estimate
    
    // Chi-squared distribution
    let chi2 = ChiSquared::new(dof).map_err(|e| {
        SignalError::ComputationError(format!("Failed to create chi-squared distribution: {}", e))
    })?;
    
    // Confidence interval factors
    let alpha = 1.0 - confidence_level;
    let lower_quantile = chi2.inverse_cdf(alpha / 2.0);
    let upper_quantile = chi2.inverse_cdf(1.0 - alpha / 2.0);
    
    let lower_factor = dof / upper_quantile;
    let upper_factor = dof / lower_quantile;
    
    // Apply factors to PSD estimate
    let n_freqs = spectra.ncols();
    let mut lower_ci = vec![0.0; n_freqs];
    let mut upper_ci = vec![0.0; n_freqs];
    
    let weight_sum: f64 = eigenvalues.sum();
    
    for j in 0..n_freqs {
        let mut weighted_sum = 0.0;
        for i in 0..spectra.nrows() {
            weighted_sum += eigenvalues[i] * spectra[[i, j]];
        }
        let psd_estimate = weighted_sum / weight_sum;
        
        lower_ci[j] = psd_estimate * lower_factor;
        upper_ci[j] = psd_estimate * upper_factor;
    }
    
    Ok((lower_ci, upper_ci))
}

/// Compute effective degrees of freedom for the multitaper estimate
fn compute_effective_dof(eigenvalues: &Array1<f64>) -> f64 {
    let sum_lambda: f64 = eigenvalues.sum();
    let sum_lambda_sq: f64 = eigenvalues.iter().map(|&x| x * x).sum();
    
    2.0 * sum_lambda.powi(2) / sum_lambda_sq
}

/// Find the next power of two greater than or equal to n
fn next_power_of_two(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    
    let mut power = 1;
    while power < n {
        power <<= 1;
    }
    power
}

/// Enhanced multitaper spectrogram with parallel processing
///
/// Computes a time-frequency representation using multitaper method with
/// parallel processing for improved performance on large signals.
///
/// # Arguments
///
/// * `x` - Input signal
/// * `config` - Spectrogram configuration
///
/// # Returns
///
/// * Tuple of (times, frequencies, spectrogram)
pub fn enhanced_multitaper_spectrogram<T>(
    x: &[T],
    config: &SpectrogramConfig,
) -> SignalResult<(Vec<f64>, Vec<f64>, Array2<f64>)>
where
    T: Float + NumCast + Debug + Send + Sync,
{
    // Implementation details...
    // This would include parallel processing of windows
    // and SIMD-optimized FFT computations
    
    unimplemented!("Enhanced spectrogram implementation")
}

/// Configuration for spectrogram computation
#[derive(Debug, Clone)]
pub struct SpectrogramConfig {
    /// Sampling frequency
    pub fs: f64,
    /// Window size in samples
    pub window_size: usize,
    /// Step size in samples
    pub step: usize,
    /// Multitaper parameters
    pub multitaper: MultitaperConfig,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_enhanced_pmtm_basic() {
        // Generate test signal
        let n = 512;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 10.0 * i as f64 / 100.0).sin())
            .collect();
        
        let config = MultitaperConfig::default();
        let result = enhanced_pmtm(&signal, &config).unwrap();
        
        assert_eq!(result.frequencies.len(), result.psd.len());
        assert!(result.dof.is_some());
    }
    
    #[test]
    fn test_simd_fft() {
        let signal = vec![1.0, 0.0, -1.0, 0.0];
        let result = simd_fft(&signal, 4).unwrap();
        assert_eq!(result.len(), 4);
    }
}