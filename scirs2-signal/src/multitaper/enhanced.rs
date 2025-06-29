//! Enhanced multitaper spectral estimation with SIMD and parallel processing
//!
//! This module provides high-performance implementations of multitaper spectral
//! estimation using scirs2-core's SIMD and parallel processing capabilities.
//!
//! Key improvements in this version:
//! - Enhanced numerical stability in adaptive weighting
//! - Better convergence detection and error handling
//! - Improved memory efficiency for large signals
//! - More robust confidence interval computation
//! - Better parameter validation and edge case handling

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
    /// Force memory-optimized processing for large signals
    pub memory_optimized: bool,
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
            memory_optimized: false,
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
/// use rand::prelude::*;
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
pub fn enhanced_pmtm<T>(
    x: &[T],
    config: &MultitaperConfig,
) -> SignalResult<EnhancedMultitaperResult>
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

    // Convert input to f64 for numerical computations
    let x_f64: Vec<f64> = x
        .iter()
        .map(|&val| {
            NumCast::from(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<f64>>>()?;

    // Validate confidence level if provided
    if let Some(confidence) = config.confidence {
        if confidence <= 0.0 || confidence >= 1.0 {
            return Err(SignalError::ValueError(format!(
                "Confidence level must be between 0 and 1, got {}",
                confidence
            )));
        }
    }

    // Enhanced multitaper parameter validation
    if config.k > (2.0 * config.nw) as usize {
        return Err(SignalError::ValueError(format!(
            "Number of tapers k={} should not exceed 2*nw={}",
            config.k,
            2.0 * config.nw
        )));
    }

    // Additional validation for numerical stability
    if config.nw < 1.0 {
        return Err(SignalError::ValueError(format!(
            "Time-bandwidth product nw={} must be at least 1.0",
            config.nw
        )));
    }

    if config.k == 0 {
        return Err(SignalError::ValueError(
            "Number of tapers k must be at least 1".to_string(),
        ));
    }

    // Check if signal is long enough for meaningful spectral estimation
    let min_signal_length = (4.0 * config.nw) as usize;
    if x_f64.len() < min_signal_length {
        return Err(SignalError::ValueError(format!(
            "Signal length {} too short for nw={}. Minimum length is {}",
            x_f64.len(),
            config.nw,
            min_signal_length
        )));
    }

    // Warn if signal is very short relative to nw
    if x_f64.len() < (8.0 * config.nw) as usize {
        eprintln!("Warning: Signal length {} is relatively short for nw={}. Consider reducing nw or using a longer signal.", 
                  x_f64.len(), config.nw);
    }

    check_finite(&x_f64, "signal")?;

    let n = x_f64.len();
    let nfft = config.nfft.unwrap_or(next_power_of_two(n));

    // Enhanced memory management: Adaptive threshold based on available system memory
    let memory_threshold = if config.k > 10 {
        500_000 // Reduce threshold for many tapers
    } else {
        1_000_000 // 1M samples for normal cases
    };
    let use_chunked_processing = n > memory_threshold || config.memory_optimized;

    if use_chunked_processing {
        return compute_pmtm_chunked(&x_f64, config, nfft);
    }

    // Compute DPSS tapers with enhanced validation
    let (tapers, eigenvalues_opt) = dpss(n, config.nw, config.k, true)?;

    let eigenvalues = eigenvalues_opt.ok_or_else(|| {
        SignalError::ComputationError("Eigenvalues required but not returned from dpss".to_string())
    })?;

    // Enhanced validation of DPSS results
    // Check eigenvalue ordering (should be descending)
    for i in 1..eigenvalues.len() {
        if eigenvalues[i] > eigenvalues[i - 1] {
            return Err(SignalError::ComputationError(
                "DPSS eigenvalues are not in descending order".to_string(),
            ));
        }
    }

    // Check eigenvalue concentration (should be close to 1 for good tapers)
    let min_concentration = 0.9;
    for (i, &eigenval) in eigenvalues.iter().enumerate() {
        if eigenval < min_concentration && i < config.k {
            eprintln!("Warning: Taper {} has low concentration ratio {:.3}. Consider reducing k or increasing nw.", 
                      i, eigenval);
        }
    }

    // Verify taper orthogonality
    for i in 0..config.k {
        for j in (i + 1)..config.k {
            let dot_product: f64 = tapers.row(i).dot(&tapers.row(j));
            if dot_product.abs() > 1e-10 {
                eprintln!(
                    "Warning: Tapers {} and {} have non-orthogonal dot product {:.2e}",
                    i, j, dot_product
                );
            }
        }
    }

    // Compute tapered FFTs using parallel processing if enabled
    let spectra = if config.parallel && n >= config.parallel_threshold {
        compute_tapered_ffts_parallel(&x_f64, &tapers, nfft)?
    } else {
        compute_tapered_ffts_simd(&x_f64, &tapers, nfft)?
    };

    // Enhanced spectral validation before combination
    for i in 0..spectra.nrows() {
        for j in 0..spectra.ncols() {
            let val = spectra[[i, j]];
            if !val.is_finite() || val < 0.0 {
                return Err(SignalError::ComputationError(format!(
                    "Invalid spectral value at taper {}, frequency bin {}: {}",
                    i, j, val
                )));
            }
        }
    }

    // Combine spectra using adaptive or standard weighting
    let (frequencies, psd) = if config.adaptive {
        combine_spectra_adaptive(&spectra, &eigenvalues, config.fs, nfft, config.onesided)?
    } else {
        combine_spectra_standard(&spectra, &eigenvalues, config.fs, nfft, config.onesided)?
    };

    // Final validation of PSD results
    for (i, &val) in psd.iter().enumerate() {
        if !val.is_finite() || val < 0.0 {
            return Err(SignalError::ComputationError(format!(
                "Invalid PSD value at frequency bin {}: {}",
                i, val
            )));
        }
    }

    // Compute confidence intervals if requested
    let confidence_intervals = if let Some(confidence_level) = config.confidence {
        Some(compute_confidence_intervals(
            &spectra,
            &eigenvalues,
            confidence_level,
        )?)
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
        tapers: if config.return_tapers {
            Some(tapers)
        } else {
            None
        },
        eigenvalues: if config.return_tapers {
            Some(eigenvalues)
        } else {
            None
        },
    })
}

/// Compute tapered FFTs using enhanced SIMD operations
fn compute_tapered_ffts_simd(
    signal: &[f64],
    tapers: &Array2<f64>,
    nfft: usize,
) -> SignalResult<Array2<f64>> {
    let k = tapers.nrows();
    let n = signal.len();
    let mut spectra = Array2::zeros((k, nfft));

    // Get SIMD capabilities for optimal performance
    let caps = PlatformCapabilities::detect();
    let use_advanced_simd = caps.has_avx2 || caps.has_avx512;

    // Enhanced memory management for large datasets
    let memory_efficient = k > 20 || n > 50_000;

    if memory_efficient {
        // Process tapers in smaller batches to reduce memory pressure
        let batch_size = if k > 50 { 8 } else { k };

        for batch_start in (0..k).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(k);

            for i in batch_start..batch_end {
                let result = compute_single_tapered_fft_simd(
                    signal,
                    tapers.row(i),
                    nfft,
                    use_advanced_simd,
                )?;
                for (j, &val) in result.iter().enumerate() {
                    spectra[[i, j]] = val;
                }
            }
        }
    } else {
        // Process all tapers at once for smaller datasets
        for i in 0..k {
            let result =
                compute_single_tapered_fft_simd(signal, tapers.row(i), nfft, use_advanced_simd)?;
            for (j, &val) in result.iter().enumerate() {
                spectra[[i, j]] = val;
            }
        }
    }

    // Enhanced validation of spectral results
    validate_spectral_matrix(&spectra)?;

    Ok(spectra)
}

/// Compute single tapered FFT with optimized SIMD operations
fn compute_single_tapered_fft_simd(
    signal: &[f64],
    taper: ArrayView1<f64>,
    nfft: usize,
    use_advanced_simd: bool,
) -> SignalResult<Vec<f64>> {
    let n = signal.len();

    // Enhanced tapering with SIMD optimizations
    let mut tapered = vec![0.0; n];

    if use_advanced_simd && n >= 64 {
        // Use advanced SIMD operations for larger signals
        use crate::simd_advanced::{simd_apply_window, SimdConfig};
        let config = SimdConfig::default();

        // Convert taper to Vec for SIMD operations
        let taper_vec: Vec<f64> = taper.iter().copied().collect();

        match simd_apply_window(signal, &taper_vec, &mut tapered, &config) {
            Ok(()) => {
                // SIMD tapering successful
            }
            Err(_) => {
                // Fallback to basic SIMD operations
                let signal_view = ArrayView1::from(signal);
                let tapered_view = ArrayView1::from_shape(n, &mut tapered).unwrap();
                f64::simd_mul(&signal_view, &taper, &tapered_view);
            }
        }
    } else {
        // Use basic SIMD operations for smaller signals
        let signal_view = ArrayView1::from(signal);
        let tapered_view = ArrayView1::from_shape(n, &mut tapered).unwrap();
        f64::simd_mul(&signal_view, &taper, &tapered_view);
    }

    // Enhanced validation of tapered signal
    for (i, &val) in tapered.iter().enumerate() {
        if !val.is_finite() {
            return Err(SignalError::ComputationError(format!(
                "Non-finite value in tapered signal at index {}: {}",
                i, val
            )));
        }
    }

    // Compute FFT with enhanced error handling
    let spectrum = simd_fft(&tapered, nfft)?;

    // Compute power spectrum with overflow protection
    let mut power_spectrum = Vec::with_capacity(nfft);
    for &val in spectrum.iter() {
        let power = val.norm_sqr();

        // Enhanced validation for power values
        if !power.is_finite() || power < 0.0 {
            return Err(SignalError::ComputationError(format!(
                "Invalid power spectrum value: {}",
                power
            )));
        }

        // Protect against extremely large values that might cause issues
        if power > 1e100 {
            eprintln!("Warning: Very large power spectrum value: {:.2e}", power);
        }

        power_spectrum.push(power);
    }

    Ok(power_spectrum)
}

/// Validate spectral matrix for numerical stability
fn validate_spectral_matrix(spectra: &Array2<f64>) -> SignalResult<()> {
    let (k, nfft) = spectra.dim();

    for i in 0..k {
        for j in 0..nfft {
            let val = spectra[[i, j]];

            if !val.is_finite() {
                return Err(SignalError::ComputationError(format!(
                    "Non-finite spectral value at taper {}, frequency bin {}: {}",
                    i, j, val
                )));
            }

            if val < 0.0 {
                return Err(SignalError::ComputationError(format!(
                    "Negative spectral value at taper {}, frequency bin {}: {}",
                    i, j, val
                )));
            }

            // Check for extremely large values that might indicate computational issues
            if val > 1e200 {
                return Err(SignalError::ComputationError(format!(
                    "Extremely large spectral value at taper {}, frequency bin {}: {:.2e}",
                    i, j, val
                )));
            }
        }
    }

    // Additional validation: check for reasonable energy distribution
    for i in 0..k {
        let row_sum: f64 = (0..nfft).map(|j| spectra[[i, j]]).sum();

        if row_sum < 1e-100 {
            return Err(SignalError::ComputationError(format!(
                "Taper {} has extremely low total energy: {:.2e}",
                i, row_sum
            )));
        }

        if row_sum > 1e100 {
            eprintln!(
                "Warning: Taper {} has very high total energy: {:.2e}",
                i, row_sum
            );
        }
    }

    Ok(())
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
    let results: Result<Vec<Vec<f64>>, SignalError> = (0..k)
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
            let spectrum = simd_fft(&tapered, nfft)?;

            // Return power spectrum
            Ok(spectrum.iter().map(|c| c.norm_sqr()).collect())
        })
        .collect();

    let results = results?;

    // Convert to Array2
    let mut spectra = Array2::zeros((k, nfft));
    for (i, row) in results.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            spectra[[i, j]] = val;
        }
    }

    Ok(spectra)
}

/// Enhanced FFT using SIMD operations and optimized planning
fn simd_fft(x: &[f64], nfft: usize) -> SignalResult<Vec<Complex64>> {
    // Enhanced input validation
    if nfft == 0 {
        return Err(SignalError::ValueError(
            "FFT length cannot be zero".to_string(),
        ));
    }

    if !nfft.is_power_of_two() {
        return Err(SignalError::ValueError(
            "FFT length must be a power of two for optimal performance".to_string(),
        ));
    }

    // Pad or truncate to nfft with improved memory management
    let mut padded = vec![Complex64::new(0.0, 0.0); nfft];
    let copy_len = x.len().min(nfft);

    // Use SIMD-optimized copying when possible
    if copy_len >= 64 {
        use crate::simd_advanced::{simd_apply_window, SimdConfig};
        let config = SimdConfig::default();
        let unity_window = vec![1.0; copy_len];
        let mut temp_real = vec![0.0; copy_len];

        // Copy using SIMD operations
        if simd_apply_window(&x[..copy_len], &unity_window, &mut temp_real, &config).is_ok() {
            for (i, &val) in temp_real.iter().enumerate() {
                padded[i] = Complex64::new(val, 0.0);
            }
        } else {
            // Fallback to scalar copy
            for i in 0..copy_len {
                padded[i] = Complex64::new(x[i], 0.0);
            }
        }
    } else {
        for i in 0..copy_len {
            padded[i] = Complex64::new(x[i], 0.0);
        }
    }

    // Use rustfft with enhanced error handling and performance optimization
    use rustfft::{num_complex::Complex, FftPlanner};

    let mut planner = FftPlanner::new();

    // Create FFT with proper error handling
    let fft = planner.plan_fft_forward(nfft);
    let mut buffer = padded.clone();

    // Validate buffer before FFT
    for (i, &val) in buffer.iter().enumerate() {
        if !val.is_finite() {
            return Err(SignalError::ComputationError(format!(
                "Non-finite value in FFT input at index {}: {}",
                i, val
            )));
        }
    }

    // Perform FFT with timing for large transforms
    if nfft > 8192 {
        let start = std::time::Instant::now();
        fft.process(&mut buffer);
        let duration = start.elapsed();

        // Warn for very slow FFTs
        if duration.as_millis() > 1000 {
            eprintln!(
                "Warning: Large FFT took {:.2}s for length {}",
                duration.as_secs_f64(),
                nfft
            );
        }
    } else {
        fft.process(&mut buffer);
    }

    // Validate output
    for (i, &val) in buffer.iter().enumerate() {
        if !val.is_finite() {
            return Err(SignalError::ComputationError(format!(
                "Non-finite value in FFT output at index {}: {}",
                i, val
            )));
        }
    }

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
        (0..n_freqs).map(|i| i as f64 * fs / nfft as f64).collect()
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

/// Combine spectra using Thomson's adaptive weighting method with improved robustness
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

    // Enhanced adaptive algorithm with improved stabilization
    let max_iter = 15; // Increased for better convergence
    let tolerance = 1e-10; // Tighter tolerance for better accuracy
    let min_weight = 1e-15; // Reduced for better precision
    let regularization = 1e-12; // Tighter regularization
    let damping_start = 7; // Start damping later
    let damping_factor = 0.9; // Less aggressive damping

    // Initialize with normalized eigenvalue weights
    let lambda_sum: f64 = eigenvalues.sum();
    for i in 0..k {
        for j in 0..n_freqs {
            weights[[i, j]] = eigenvalues[i] / lambda_sum;
        }
    }

    // Adaptive iteration with convergence checks
    let mut converged = false;
    let mut last_old_psd = psd.clone();
    for iter in 0..max_iter {
        let old_psd = psd.clone();
        last_old_psd = old_psd.clone();

        // Update PSD estimate with numerical stabilization
        for j in 0..n_freqs {
            let mut weighted_sum = 0.0;
            let mut weight_sum = 0.0;

            for i in 0..k {
                let w = weights[[i, j]];
                weighted_sum += w * spectra[[i, j]];
                weight_sum += w;
            }

            // Prevent division by zero
            if weight_sum > min_weight {
                psd[j] = weighted_sum / weight_sum;
            } else {
                // Fallback to eigenvalue-weighted average
                let mut fallback_sum = 0.0;
                for i in 0..k {
                    fallback_sum += eigenvalues[i] * spectra[[i, j]];
                }
                psd[j] = fallback_sum / lambda_sum;
            }

            // Ensure PSD is positive
            psd[j] = psd[j].max(regularization);
        }

        // Update weights with improved Thomson's method
        for j in 0..n_freqs {
            let mut new_weight_sum = 0.0;

            for i in 0..k {
                let lambda = eigenvalues[i];
                let spectrum_val = spectra[[i, j]].max(regularization);
                let psd_val = psd[j].max(regularization);

                // Improved bias factor calculation
                let ratio = psd_val / spectrum_val;
                let bias_factor = if ratio > 1e-6 {
                    lambda / (lambda + ratio.powi(2))
                } else {
                    lambda // Fallback for very small ratios
                };

                weights[[i, j]] = bias_factor.max(min_weight);
                new_weight_sum += weights[[i, j]];
            }

            // Normalize weights for this frequency bin
            if new_weight_sum > min_weight {
                for i in 0..k {
                    weights[[i, j]] /= new_weight_sum;
                }
            } else {
                // Fallback to equal weights
                for i in 0..k {
                    weights[[i, j]] = 1.0 / k as f64;
                }
            }
        }

        // Enhanced convergence criteria with multiple metrics
        let max_change = old_psd
            .iter()
            .zip(psd.iter())
            .map(|(old, new)| {
                let denominator = old.abs().max(new.abs()).max(1e-12);
                ((old - new) / denominator).abs()
            })
            .fold(0.0, f64::max);

        let mean_change = old_psd
            .iter()
            .zip(psd.iter())
            .map(|(old, new)| {
                let denominator = old.abs().max(new.abs()).max(1e-12);
                ((old - new) / denominator).abs()
            })
            .sum::<f64>()
            / n_freqs as f64;

        // Additional convergence criterion: RMS change
        let rms_change = (old_psd
            .iter()
            .zip(psd.iter())
            .map(|(old, new)| {
                let denominator = old.abs().max(new.abs()).max(1e-12);
                ((old - new) / denominator).powi(2)
            })
            .sum::<f64>()
            / n_freqs as f64)
            .sqrt();

        // Convergence check with all three criteria
        if max_change < tolerance && mean_change < tolerance * 0.1 && rms_change < tolerance * 0.5 {
            converged = true;
            break;
        }

        // Add adaptive damping for later iterations to ensure convergence
        if iter > damping_start {
            // Adaptive damping based on convergence rate
            let adaptive_damping = if mean_change > tolerance * 10.0 {
                0.7 // More aggressive damping for slow convergence
            } else {
                damping_factor // Standard damping
            };

            for j in 0..n_freqs {
                psd[j] = adaptive_damping * psd[j] + (1.0 - adaptive_damping) * old_psd[j];
            }
        }
    }

    // Enhanced convergence diagnostics
    if !converged {
        // Check if we're close to convergence
        let final_change = last_old_psd
            .iter()
            .zip(psd.iter())
            .map(|(old, new)| {
                let denominator = old.abs().max(new.abs()).max(1e-12);
                ((old - new) / denominator).abs()
            })
            .fold(0.0, f64::max);

        if final_change < tolerance * 10.0 {
            // Close enough for practical purposes
            eprintln!(
                "Warning: Adaptive multitaper algorithm nearly converged (final change: {:.2e})",
                final_change
            );
        } else {
            eprintln!("Warning: Adaptive multitaper algorithm did not converge within {} iterations (final change: {:.2e})", max_iter, final_change);
            // Could potentially return an error here in stricter implementations
        }
    }

    // Create frequency array
    let frequencies = if onesided {
        (0..n_freqs).map(|i| i as f64 * fs / nfft as f64).collect()
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

    // Apply final scaling with improved normalization
    let scaling = if onesided { 2.0 / fs } else { 1.0 / fs };
    psd.iter_mut().for_each(|p| *p *= scaling);

    Ok((frequencies, psd))
}

/// Compute confidence intervals using enhanced chi-squared approximation
///
/// This implementation includes improved DOF calculation and better handling
/// of edge cases for more accurate confidence intervals.
fn compute_confidence_intervals(
    spectra: &Array2<f64>,
    eigenvalues: &Array1<f64>,
    confidence_level: f64,
) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    use statrs::distribution::{ChiSquared, ContinuousCDF};

    let k = spectra.nrows() as f64;
    // Enhanced DOF calculation using effective number of tapers
    let effective_k = compute_effective_dof(eigenvalues) / 2.0;
    let dof = 2.0 * effective_k; // More accurate degrees of freedom

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

    // Apply factors to PSD estimate with improved scaling
    let n_freqs = spectra.ncols();
    let mut lower_ci = vec![0.0; n_freqs];
    let mut upper_ci = vec![0.0; n_freqs];

    let weight_sum: f64 = eigenvalues.sum();

    for j in 0..n_freqs {
        let mut weighted_sum = 0.0;
        let mut variance_estimate = 0.0;

        // Compute weighted mean and variance estimate
        for i in 0..spectra.nrows() {
            weighted_sum += eigenvalues[i] * spectra[[i, j]];
        }
        let psd_estimate = weighted_sum / weight_sum;

        // Improved variance estimation for better confidence intervals
        for i in 0..spectra.nrows() {
            let deviation = spectra[[i, j]] - psd_estimate;
            variance_estimate += eigenvalues[i] * deviation * deviation;
        }
        variance_estimate /= weight_sum;

        // Apply chi-squared scaling with variance correction
        let scale_factor = (1.0 + variance_estimate / (psd_estimate * psd_estimate + 1e-15)).sqrt();

        lower_ci[j] = psd_estimate * lower_factor / scale_factor;
        upper_ci[j] = psd_estimate * upper_factor * scale_factor;

        // Ensure positive confidence intervals
        lower_ci[j] = lower_ci[j].max(1e-15);
        upper_ci[j] = upper_ci[j].max(lower_ci[j] * 1.01); // Ensure upper > lower
    }

    Ok((lower_ci, upper_ci))
}

/// Compute effective degrees of freedom for the multitaper estimate
fn compute_effective_dof(eigenvalues: &Array1<f64>) -> f64 {
    let sum_lambda: f64 = eigenvalues.sum();
    let sum_lambda_sq: f64 = eigenvalues.iter().map(|&x| x * x).sum();

    2.0 * sum_lambda.powi(2) / sum_lambda_sq
}

/// Enhanced memory-efficient multitaper estimation for very large signals
///
/// This implementation includes improved chunking strategy and better
/// statistical combination of results across chunks.
fn compute_pmtm_chunked(
    signal: &[f64],
    config: &MultitaperConfig,
    nfft: usize,
) -> SignalResult<EnhancedMultitaperResult> {
    let n = signal.len();
    // Adaptive chunk size based on available memory and signal characteristics
    let base_chunk_size = if config.k > 20 { 50_000 } else { 100_000 };
    let chunk_size = base_chunk_size.min(n / 10).max(config.k * 20); // Ensure minimum viable chunk
    let overlap = (chunk_size as f64 * 0.2) as usize; // Increased overlap for better continuity
    let step = chunk_size - overlap;

    // Calculate number of chunks
    let n_chunks = (n + step - 1) / step; // Ceiling division

    // Initialize accumulators
    let n_freqs = if config.onesided { nfft / 2 + 1 } else { nfft };
    let mut psd_accumulator = vec![0.0; n_freqs];
    let mut weight_accumulator = vec![0.0; n_freqs];
    let mut frequencies = Vec::new();

    // Process each chunk
    for chunk_idx in 0..n_chunks {
        let start = chunk_idx * step;
        let end = (start + chunk_size).min(n);

        let chunk_len = end - start;
        if chunk_len < config.k * 15 {
            // Skip chunks that are too small for reliable estimation
            // Increased minimum size for better statistical properties
            continue;
        }

        // Additional validation for chunk quality
        let chunk = &signal[start..end];
        let chunk_energy: f64 = chunk.iter().map(|&x| x * x).sum();
        if chunk_energy < 1e-20 {
            // Skip near-zero energy chunks
            continue;
        }

        let chunk = &signal[start..end];
        let chunk_len = chunk.len();

        // Compute DPSS for this chunk size
        let (tapers, eigenvalues_opt) = dpss(chunk_len, config.nw, config.k, true)?;
        let eigenvalues = eigenvalues_opt.ok_or_else(|| {
            SignalError::ComputationError(
                "Eigenvalues required but not returned from dpss".to_string(),
            )
        })?;

        // Use a smaller nfft for chunks
        let chunk_nfft = next_power_of_two(chunk_len);

        // Compute tapered FFTs for this chunk
        let spectra = if config.parallel && chunk_len >= config.parallel_threshold {
            compute_tapered_ffts_parallel(chunk, &tapers, chunk_nfft)?
        } else {
            compute_tapered_ffts_simd(chunk, &tapers, chunk_nfft)?
        };

        // Combine spectra for this chunk
        let (chunk_freqs, chunk_psd) = if config.adaptive {
            combine_spectra_adaptive(
                &spectra,
                &eigenvalues,
                config.fs,
                chunk_nfft,
                config.onesided,
            )?
        } else {
            combine_spectra_standard(
                &spectra,
                &eigenvalues,
                config.fs,
                chunk_nfft,
                config.onesided,
            )?
        };

        // Store frequencies from first chunk
        if chunk_idx == 0 {
            frequencies = chunk_freqs.clone();
        }

        // Interpolate chunk PSD to match target frequency grid if needed
        let interpolated_psd = if chunk_freqs.len() != frequencies.len() {
            interpolate_psd(&chunk_freqs, &chunk_psd, &frequencies)?
        } else {
            chunk_psd
        };

        // Enhanced weighted accumulation with variance tracking
        let chunk_len_actual = end - start;
        let chunk_weight = (chunk_len_actual as f64 / n as f64)
            * (chunk_len_actual as f64 / chunk_size as f64).sqrt(); // Quality factor

        for (i, &psd_val) in interpolated_psd.iter().enumerate() {
            if i < psd_accumulator.len() && psd_val.is_finite() && psd_val > 0.0 {
                psd_accumulator[i] += psd_val * chunk_weight;
                weight_accumulator[i] += chunk_weight;
            }
        }
    }

    // Normalize accumulated PSD
    for i in 0..psd_accumulator.len() {
        if weight_accumulator[i] > 0.0 {
            psd_accumulator[i] /= weight_accumulator[i];
        }
    }

    // Note: For chunked processing, we don't compute confidence intervals
    // as they would require more complex statistical handling across chunks

    Ok(EnhancedMultitaperResult {
        frequencies,
        psd: psd_accumulator,
        confidence_intervals: None, // Not supported for chunked processing
        dof: Some(2.0 * config.k as f64 * n_chunks as f64), // Approximate DOF
        tapers: None,               // Not returned for memory efficiency
        eigenvalues: None,          // Not returned for memory efficiency
    })
}

/// Simple linear interpolation for PSD values
fn interpolate_psd(
    source_freqs: &[f64],
    source_psd: &[f64],
    target_freqs: &[f64],
) -> SignalResult<Vec<f64>> {
    if source_freqs.is_empty() || source_psd.is_empty() || target_freqs.is_empty() {
        return Err(SignalError::ValueError(
            "Empty frequency or PSD arrays".to_string(),
        ));
    }

    let mut result = vec![0.0; target_freqs.len()];

    for (i, &target_freq) in target_freqs.iter().enumerate() {
        // Find bracketing indices
        let mut lower_idx = 0;
        let mut upper_idx = source_freqs.len() - 1;

        for (j, &freq) in source_freqs.iter().enumerate() {
            if freq <= target_freq {
                lower_idx = j;
            } else {
                upper_idx = j;
                break;
            }
        }

        if lower_idx == upper_idx {
            // Exact match or at boundary
            result[i] = source_psd[lower_idx];
        } else {
            // Linear interpolation
            let f1 = source_freqs[lower_idx];
            let f2 = source_freqs[upper_idx];
            let p1 = source_psd[lower_idx];
            let p2 = source_psd[upper_idx];

            if (f2 - f1).abs() > 1e-15 {
                let weight = (target_freq - f1) / (f2 - f1);
                result[i] = p1 + weight * (p2 - p1);
            } else {
                result[i] = (p1 + p2) / 2.0;
            }
        }
    }

    Ok(result)
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
    // Validate input
    if x.is_empty() {
        return Err(SignalError::ValueError("Input signal is empty".to_string()));
    }

    check_positive(config.window_size, "window_size")?;
    check_positive(config.step, "step")?;

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
    let window_size = config.window_size;
    let step = config.step;

    // Calculate number of windows
    if window_size > n {
        return Err(SignalError::ValueError(
            "Window size larger than signal length".to_string(),
        ));
    }

    let n_windows = (n - window_size) / step + 1;
    if n_windows == 0 {
        return Err(SignalError::ValueError(
            "No complete windows in signal".to_string(),
        ));
    }

    // Prepare multitaper config for each window
    let mut mt_config = config.multitaper.clone();
    mt_config.nfft = Some(config.window_size);

    // Calculate time points
    let times: Vec<f64> = (0..n_windows)
        .map(|i| (i * step + window_size / 2) as f64 / config.fs)
        .collect();

    // Process windows in parallel if enabled
    let results: Vec<EnhancedMultitaperResult> = if config.multitaper.parallel
        && n_windows >= config.multitaper.parallel_threshold / window_size
    {
        let x_arc = Arc::new(x_f64);

        (0..n_windows)
            .into_par_iter()
            .map(|i| {
                let start = i * step;
                let end = start + window_size;
                let window = &x_arc[start..end];

                enhanced_pmtm(window, &mt_config).unwrap()
            })
            .collect()
    } else {
        // Sequential processing
        (0..n_windows)
            .map(|i| {
                let start = i * step;
                let end = start + window_size;
                let window = &x_f64[start..end];

                enhanced_pmtm(window, &mt_config)
            })
            .collect::<SignalResult<Vec<_>>>()?
    };

    // Extract frequencies from first result
    let frequencies = results[0].frequencies.clone();
    let n_freqs = frequencies.len();

    // Build spectrogram matrix
    let mut spectrogram = Array2::zeros((n_freqs, n_windows));

    for (j, result) in results.iter().enumerate() {
        for (i, &psd_val) in result.psd.iter().enumerate() {
            spectrogram[[i, j]] = psd_val;
        }
    }

    // Apply logarithmic scaling if requested (common for spectrograms)
    let epsilon = 1e-10;
    spectrogram.mapv_inplace(|x| (x + epsilon).log10() * 10.0); // Convert to dB

    Ok((times, frequencies, spectrogram))
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
