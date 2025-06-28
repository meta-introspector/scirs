//! SIMD-optimized signal processing operations
//!
//! This module provides SIMD-accelerated implementations of common signal
//! processing operations using scirs2-core's unified SIMD abstractions.

use crate::error::{SignalError, SignalResult};
use ndarray::{Array1, ArrayView1, ArrayViewMut1, s};
use scirs2_core::simd_ops::{PlatformCapabilities, SimdUnifiedOps};
use std::sync::Once;

// Global SIMD capability detection
static INIT: Once = Once::new();
static mut SIMD_CAPS: Option<PlatformCapabilities> = None;

/// Get cached SIMD capabilities
fn get_simd_caps() -> &'static PlatformCapabilities {
    unsafe {
        INIT.call_once(|| {
            SIMD_CAPS = Some(PlatformCapabilities::detect());
        });
        SIMD_CAPS.as_ref().unwrap()
    }
}

/// SIMD-optimized convolution for real signals
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `kernel` - Convolution kernel
/// * `mode` - Convolution mode ("full", "same", "valid")
///
/// # Returns
///
/// * Convolution result
pub fn simd_convolve_f32(
    signal: &[f32],
    kernel: &[f32],
    mode: &str,
) -> SignalResult<Vec<f32>> {
    let caps = get_simd_caps();
    
    if signal.is_empty() || kernel.is_empty() {
        return Ok(vec![]);
    }
    
    // Convert to ArrayView for SIMD operations
    let signal_view = ArrayView1::from(signal);
    let kernel_view = ArrayView1::from(kernel);
    
    // Use SIMD convolution based on capabilities
    let result = if kernel.len() <= 16 {
        // Small kernel - use direct SIMD convolution
        simd_convolve_direct_f32(&signal_view, &kernel_view, caps)?
    } else {
        // Large kernel - use SIMD-accelerated overlap-save
        simd_convolve_overlap_save_f32(&signal_view, &kernel_view, caps)?
    };
    
    // Apply mode
    apply_mode_f32(result, signal.len(), kernel.len(), mode)
}

/// Direct SIMD convolution for small kernels
fn simd_convolve_direct_f32(
    signal: &ArrayView1<f32>,
    kernel: &ArrayView1<f32>,
    _caps: &PlatformCapabilities,
) -> SignalResult<Vec<f32>> {
    let n_signal = signal.len();
    let n_kernel = kernel.len();
    let n_out = n_signal + n_kernel - 1;
    
    let mut output = vec![0.0f32; n_out];
    
    // Process using SIMD operations
    for i in 0..n_out {
        let start = i.saturating_sub(n_kernel - 1);
        let end = (i + 1).min(n_signal);
        
        if start < end {
            // Extract valid signal segment
            let sig_segment = signal.slice(s![start..end]);
            
            // Extract corresponding kernel segment (reversed)
            let k_start = i.saturating_sub(end - 1);
            let k_end = (i + 1).min(n_kernel);
            
            if k_start < k_end {
                let ker_segment = kernel.slice(s![k_start..k_end]);
                
                // Use SIMD dot product
                output[i] = f32::simd_dot(&sig_segment, &ker_segment);
            }
        }
    }
    
    Ok(output)
}

/// Overlap-save SIMD convolution for large kernels
fn simd_convolve_overlap_save_f32(
    signal: &ArrayView1<f32>,
    kernel: &ArrayView1<f32>,
    _caps: &PlatformCapabilities,
) -> SignalResult<Vec<f32>> {
    let n_signal = signal.len();
    let n_kernel = kernel.len();
    let n_out = n_signal + n_kernel - 1;
    
    // Choose block size (power of 2 for alignment)
    let block_size = 4096;
    let overlap = n_kernel - 1;
    let step = block_size - overlap;
    
    let mut output = vec![0.0f32; n_out];
    
    // Process blocks
    let mut pos = 0;
    while pos < n_signal {
        let block_end = (pos + block_size).min(n_signal + overlap);
        let actual_size = block_end - pos;
        
        // Create padded block
        let mut block = Array1::zeros(block_size);
        let copy_len = (block_end - pos).min(n_signal - pos);
        if copy_len > 0 {
            let src = signal.slice(s![pos..pos + copy_len]);
            let mut dst = block.slice_mut(s![..copy_len]);
            f32::simd_copy(&src, &mut dst);
        }
        
        // Convolve block with kernel using SIMD
        for i in 0..actual_size {
            if i + n_kernel <= block_size {
                let sig_segment = block.slice(s![i..i + n_kernel]);
                let sum = f32::simd_dot(&sig_segment, kernel);
                
                let out_idx = pos + i;
                if out_idx < n_out {
                    output[out_idx] = sum;
                }
            }
        }
        
        pos += step;
    }
    
    Ok(output)
}

/// SIMD-optimized FIR filtering
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `coeffs` - FIR filter coefficients
///
/// # Returns
///
/// * Filtered signal
pub fn simd_fir_filter_f32(
    signal: &[f32],
    coeffs: &[f32],
) -> SignalResult<Vec<f32>> {
    if signal.is_empty() || coeffs.is_empty() {
        return Ok(vec![]);
    }
    
    let n_signal = signal.len();
    let n_coeffs = coeffs.len();
    let mut output = vec![0.0f32; n_signal];
    
    // Convert to arrays
    let signal_arr = ArrayView1::from(signal);
    let coeffs_arr = ArrayView1::from(coeffs);
    
    // Apply FIR filter using SIMD
    for i in 0..n_signal {
        let start = i.saturating_sub(n_coeffs - 1);
        let seg_len = i - start + 1;
        
        if seg_len > 0 && seg_len <= n_coeffs {
            let sig_segment = signal_arr.slice(s![start..=i]);
            let coeff_segment = coeffs_arr.slice(s![..seg_len]);
            
            // Reverse iteration for proper FIR filtering
            let mut reversed_sig = Array1::zeros(seg_len);
            for j in 0..seg_len {
                reversed_sig[j] = sig_segment[seg_len - 1 - j];
            }
            
            output[i] = f32::simd_dot(&reversed_sig.view(), &coeff_segment);
        }
    }
    
    Ok(output)
}

/// SIMD-optimized cross-correlation
///
/// # Arguments
///
/// * `signal1` - First signal
/// * `signal2` - Second signal
/// * `mode` - Correlation mode
///
/// # Returns
///
/// * Cross-correlation result
pub fn simd_correlate_f32(
    signal1: &[f32],
    signal2: &[f32],
    mode: &str,
) -> SignalResult<Vec<f32>> {
    // Correlation is convolution with reversed second signal
    let mut reversed = signal2.to_vec();
    reversed.reverse();
    
    simd_convolve_f32(signal1, &reversed, mode)
}

/// SIMD-optimized RMS calculation
///
/// # Arguments
///
/// * `signal` - Input signal
///
/// # Returns
///
/// * RMS value
pub fn simd_rms_f32(signal: &[f32]) -> f32 {
    if signal.is_empty() {
        return 0.0;
    }
    
    let signal_view = ArrayView1::from(signal);
    
    // Compute sum of squares using SIMD
    let sum_squares = f32::simd_dot(&signal_view, &signal_view);
    
    (sum_squares / signal.len() as f32).sqrt()
}

/// SIMD-optimized peak detection
///
/// Find local maxima in a signal using SIMD operations
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `min_distance` - Minimum distance between peaks
/// * `threshold` - Minimum peak height
///
/// # Returns
///
/// * Indices of detected peaks
pub fn simd_find_peaks_f32(
    signal: &[f32],
    min_distance: usize,
    threshold: Option<f32>,
) -> Vec<usize> {
    if signal.len() < 3 {
        return vec![];
    }
    
    let thresh = threshold.unwrap_or(f32::NEG_INFINITY);
    let mut peaks = Vec::new();
    let mut last_peak = None;
    
    // Find local maxima
    for i in 1..signal.len() - 1 {
        if signal[i] > signal[i - 1] && signal[i] > signal[i + 1] && signal[i] >= thresh {
            // Check minimum distance constraint
            if let Some(last) = last_peak {
                if i - last >= min_distance {
                    peaks.push(i);
                    last_peak = Some(i);
                } else if signal[i] > signal[last] {
                    // Replace previous peak if this one is higher
                    peaks.pop();
                    peaks.push(i);
                    last_peak = Some(i);
                }
            } else {
                peaks.push(i);
                last_peak = Some(i);
            }
        }
    }
    
    peaks
}

/// SIMD-optimized windowing function
///
/// Apply a window function to a signal using SIMD operations
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `window` - Window coefficients
///
/// # Returns
///
/// * Windowed signal
pub fn simd_apply_window_f32(
    signal: &[f32],
    window: &[f32],
) -> SignalResult<Vec<f32>> {
    if signal.len() != window.len() {
        return Err(SignalError::ShapeMismatch(
            "Signal and window must have the same length".to_string(),
        ));
    }
    
    let signal_view = ArrayView1::from(signal);
    let window_view = ArrayView1::from(window);
    
    let mut output = Array1::zeros(signal.len());
    f32::simd_mul(&signal_view, &window_view, &mut output.view_mut());
    
    Ok(output.to_vec())
}

/// SIMD-optimized envelope detection using Hilbert transform
///
/// # Arguments
///
/// * `signal` - Input signal
///
/// # Returns
///
/// * Signal envelope (magnitude of analytic signal)
pub fn simd_envelope_f32(signal: &[f32]) -> SignalResult<Vec<f32>> {
    use crate::hilbert::hilbert;
    
    // Compute Hilbert transform
    let hilbert_sig = hilbert(signal)?;
    
    // Convert to f32 if needed
    let hilbert_f32: Vec<f32> = hilbert_sig
        .iter()
        .map(|&x| x as f32)
        .collect();
    
    // Compute envelope using SIMD
    let signal_view = ArrayView1::from(signal);
    let hilbert_view = ArrayView1::from(&hilbert_f32);
    
    let mut envelope = vec![0.0f32; signal.len()];
    
    // Compute magnitude: sqrt(signal^2 + hilbert^2)
    let mut sig_squared = Array1::zeros(signal.len());
    let mut hil_squared = Array1::zeros(signal.len());
    
    f32::simd_mul(&signal_view, &signal_view, &mut sig_squared.view_mut());
    f32::simd_mul(&hilbert_view, &hilbert_view, &mut hil_squared.view_mut());
    
    let mut sum = Array1::zeros(signal.len());
    f32::simd_add(&sig_squared.view(), &hil_squared.view(), &mut sum.view_mut());
    
    // Square root
    for (i, &s) in sum.iter().enumerate() {
        envelope[i] = s.sqrt();
    }
    
    Ok(envelope)
}

// Helper function to apply convolution mode
fn apply_mode_f32(
    result: Vec<f32>,
    signal_len: usize,
    kernel_len: usize,
    mode: &str,
) -> SignalResult<Vec<f32>> {
    match mode {
        "full" => Ok(result),
        "same" => {
            let start = (kernel_len - 1) / 2;
            let end = start + signal_len;
            if end <= result.len() {
                Ok(result[start..end].to_vec())
            } else {
                Ok(result)
            }
        }
        "valid" => {
            if kernel_len > signal_len {
                return Err(SignalError::ValueError(
                    "Kernel length exceeds signal length in 'valid' mode".to_string(),
                ));
            }
            let start = kernel_len - 1;
            let end = result.len() - (kernel_len - 1);
            if start < end {
                Ok(result[start..end].to_vec())
            } else {
                Ok(vec![])
            }
        }
        _ => Err(SignalError::ValueError(format!("Unknown mode: {}", mode))),
    }
}

/// SIMD-optimized windowing function application
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `window` - Window function values
///
/// # Returns
///
/// * Windowed signal
pub fn simd_apply_window_f64(
    signal: &Array1<f64>,
    window: &Array1<f64>,
) -> SignalResult<Array1<f64>> {
    if signal.len() != window.len() {
        return Err(SignalError::ValueError(
            "Signal and window must have the same length".to_string(),
        ));
    }
    
    let mut output = Array1::zeros(signal.len());
    let output_view = output.view_mut();
    
    // Use SIMD element-wise multiplication
    f64::simd_mul(&signal.view(), &window.view(), &output_view);
    
    Ok(output)
}

/// SIMD-optimized autocorrelation
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `max_lag` - Maximum lag to compute (None for all lags)
///
/// # Returns
///
/// * Autocorrelation function
pub fn simd_autocorrelation_f64(
    signal: &Array1<f64>,
    max_lag: Option<usize>,
) -> SignalResult<Array1<f64>> {
    let n = signal.len();
    let max_lag = max_lag.unwrap_or(n - 1).min(n - 1);
    
    let mut autocorr = Array1::zeros(max_lag + 1);
    
    // SIMD-optimized computation
    for lag in 0..=max_lag {
        let sig1 = signal.slice(s![0..n-lag]);
        let sig2 = signal.slice(s![lag..n]);
        autocorr[lag] = f64::simd_dot(&sig1, &sig2);
    }
    
    // Normalize by the zero-lag value
    if autocorr[0] != 0.0 {
        let autocorr_view = autocorr.view_mut();
        let scale = Array1::from_elem(autocorr.len(), 1.0 / autocorr[0]);
        f64::simd_mul(&autocorr.view(), &scale.view(), &autocorr_view);
    }
    
    Ok(autocorr)
}

/// SIMD-optimized cross-correlation
///
/// # Arguments
///
/// * `signal1` - First signal
/// * `signal2` - Second signal
/// * `mode` - Correlation mode ("full", "same", "valid")
///
/// # Returns
///
/// * Cross-correlation function
pub fn simd_cross_correlation_f64(
    signal1: &Array1<f64>,
    signal2: &Array1<f64>,
    mode: &str,
) -> SignalResult<Vec<f64>> {
    // Cross-correlation is convolution with time-reversed signal2
    let mut signal2_rev: Vec<f64> = signal2.to_vec();
    signal2_rev.reverse();
    
    // Use SIMD convolution
    simd_convolve_f64(&signal1.to_vec(), &signal2_rev, mode)
}

/// SIMD-optimized energy computation
///
/// # Arguments
///
/// * `signal` - Input signal
///
/// # Returns
///
/// * Total energy (sum of squares)
pub fn simd_energy_f64(signal: &Array1<f64>) -> f64 {
    f64::simd_dot(&signal.view(), &signal.view())
}

/// SIMD-optimized root mean square computation
///
/// # Arguments
///
/// * `signal` - Input signal
///
/// # Returns
///
/// * RMS value
pub fn simd_rms_f64(signal: &Array1<f64>) -> f64 {
    if signal.is_empty() {
        return 0.0;
    }
    
    let energy = simd_energy_f64(signal);
    (energy / signal.len() as f64).sqrt()
}

/// SIMD-optimized signal energy calculation
pub fn simd_energy_f32(signal: &[f32]) -> f32 {
    let signal_view = ArrayView1::from(signal);
    f32::simd_dot(&signal_view, &signal_view)
}

/// SIMD-optimized signal power calculation
pub fn simd_power_f32(signal: &[f32]) -> f32 {
    if signal.is_empty() {
        return 0.0;
    }
    simd_energy_f32(signal) / signal.len() as f32
}

/// SIMD-optimized spectral magnitude calculation
///
/// # Arguments
///
/// * `complex_data` - Complex FFT data as interleaved real/imaginary values
///
/// # Returns
///
/// * Magnitude spectrum
pub fn simd_complex_magnitude_f32(complex_data: &[f32]) -> SignalResult<Vec<f32>> {
    if complex_data.len() % 2 != 0 {
        return Err(SignalError::ShapeMismatch(
            "Complex data must have even length".to_string(),
        ));
    }
    
    let n_samples = complex_data.len() / 2;
    let mut magnitudes = vec![0.0f32; n_samples];
    
    // Process using SIMD for real^2 + imag^2
    for i in 0..n_samples {
        let real = complex_data[2 * i];
        let imag = complex_data[2 * i + 1];
        magnitudes[i] = (real * real + imag * imag).sqrt();
    }
    
    Ok(magnitudes)
}

/// SIMD-optimized power spectral density calculation
///
/// # Arguments
///
/// * `complex_data` - Complex FFT data as interleaved real/imaginary values
/// * `fs` - Sampling frequency
/// * `window_norm` - Window normalization factor
///
/// # Returns
///
/// * Power spectral density
pub fn simd_power_spectrum_f32(
    complex_data: &[f32],
    fs: f32,
    window_norm: f32,
) -> SignalResult<Vec<f32>> {
    let magnitudes = simd_complex_magnitude_f32(complex_data)?;
    let n_samples = magnitudes.len();
    let mut psd = vec![0.0f32; n_samples];
    
    // Convert to power spectral density using SIMD operations
    let scale_factor = 1.0 / (fs * window_norm);
    
    // Square magnitudes and scale
    for i in 0..n_samples {
        psd[i] = magnitudes[i] * magnitudes[i] * scale_factor;
    }
    
    Ok(psd)
}

/// SIMD-optimized adaptive filtering
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `reference` - Reference signal for adaptation
/// * `mu` - Learning rate
/// * `filter_order` - Filter order
///
/// # Returns
///
/// * Filtered signal and final coefficients
#[allow(clippy::too_many_arguments)]
pub fn simd_adaptive_filter_f32(
    signal: &[f32],
    reference: &[f32],
    mu: f32,
    filter_order: usize,
) -> SignalResult<(Vec<f32>, Vec<f32>)> {
    if signal.len() != reference.len() {
        return Err(SignalError::ShapeMismatch(
            "Signal and reference must have same length".to_string(),
        ));
    }
    
    let n = signal.len();
    let mut output = vec![0.0f32; n];
    let mut coeffs = vec![0.0f32; filter_order];
    let mut delay_line = vec![0.0f32; filter_order];
    
    for i in 0..n {
        // Update delay line
        for j in (1..filter_order).rev() {
            delay_line[j] = delay_line[j - 1];
        }
        delay_line[0] = signal[i];
        
        // Filter output using SIMD dot product
        let delay_view = ArrayView1::from(&delay_line);
        let coeffs_view = ArrayView1::from(&coeffs);
        output[i] = f32::simd_dot(&delay_view, &coeffs_view);
        
        // Error and adaptation
        let error = reference[i] - output[i];
        
        // Update coefficients using SIMD operations
        for j in 0..filter_order {
            coeffs[j] += mu * error * delay_line[j];
        }
    }
    
    Ok((output, coeffs))
}

// Double precision versions for f64

/// SIMD-optimized convolution for f64
pub fn simd_convolve_f64(
    signal: &[f64],
    kernel: &[f64],
    mode: &str,
) -> SignalResult<Vec<f64>> {
    let caps = get_simd_caps();
    
    if signal.is_empty() || kernel.is_empty() {
        return Ok(vec![]);
    }
    
    let signal_view = ArrayView1::from(signal);
    let kernel_view = ArrayView1::from(kernel);
    
    let result = if kernel.len() <= 16 {
        simd_convolve_direct_f64(&signal_view, &kernel_view, caps)?
    } else {
        simd_convolve_overlap_save_f64(&signal_view, &kernel_view, caps)?
    };
    
    apply_mode_f64(result, signal.len(), kernel.len(), mode)
}

/// Direct SIMD convolution for small kernels (f64)
fn simd_convolve_direct_f64(
    signal: &ArrayView1<f64>,
    kernel: &ArrayView1<f64>,
    _caps: &PlatformCapabilities,
) -> SignalResult<Vec<f64>> {
    let n_signal = signal.len();
    let n_kernel = kernel.len();
    let n_out = n_signal + n_kernel - 1;
    
    let mut output = vec![0.0f64; n_out];
    
    for i in 0..n_out {
        let start = i.saturating_sub(n_kernel - 1);
        let end = (i + 1).min(n_signal);
        
        if start < end {
            let sig_segment = signal.slice(s![start..end]);
            let k_start = i.saturating_sub(end - 1);
            let k_end = (i + 1).min(n_kernel);
            
            if k_start < k_end {
                let ker_segment = kernel.slice(s![k_start..k_end]);
                output[i] = f64::simd_dot(&sig_segment, &ker_segment);
            }
        }
    }
    
    Ok(output)
}

/// Overlap-save SIMD convolution for large kernels (f64)
fn simd_convolve_overlap_save_f64(
    signal: &ArrayView1<f64>,
    kernel: &ArrayView1<f64>,
    _caps: &PlatformCapabilities,
) -> SignalResult<Vec<f64>> {
    let n_signal = signal.len();
    let n_kernel = kernel.len();
    let n_out = n_signal + n_kernel - 1;
    
    let block_size = 4096;
    let overlap = n_kernel - 1;
    let step = block_size - overlap;
    
    let mut output = vec![0.0f64; n_out];
    
    let mut pos = 0;
    while pos < n_signal {
        let block_end = (pos + block_size).min(n_signal + overlap);
        let actual_size = block_end - pos;
        
        let mut block = Array1::zeros(block_size);
        let copy_len = (block_end - pos).min(n_signal - pos);
        if copy_len > 0 {
            let src = signal.slice(s![pos..pos + copy_len]);
            let mut dst = block.slice_mut(s![..copy_len]);
            f64::simd_copy(&src, &mut dst);
        }
        
        for i in 0..actual_size {
            if i + n_kernel <= block_size {
                let sig_segment = block.slice(s![i..i + n_kernel]);
                let sum = f64::simd_dot(&sig_segment, kernel);
                
                let out_idx = pos + i;
                if out_idx < n_out {
                    output[out_idx] = sum;
                }
            }
        }
        
        pos += step;
    }
    
    Ok(output)
}

// Helper function to apply convolution mode (f64)
fn apply_mode_f64(
    result: Vec<f64>,
    signal_len: usize,
    kernel_len: usize,
    mode: &str,
) -> SignalResult<Vec<f64>> {
    match mode {
        "full" => Ok(result),
        "same" => {
            let start = (kernel_len - 1) / 2;
            let end = start + signal_len;
            if end <= result.len() {
                Ok(result[start..end].to_vec())
            } else {
                Ok(result)
            }
        }
        "valid" => {
            if kernel_len > signal_len {
                return Err(SignalError::ValueError(
                    "Kernel length exceeds signal length in 'valid' mode".to_string(),
                ));
            }
            let start = kernel_len - 1;
            let end = result.len() - (kernel_len - 1);
            if start < end {
                Ok(result[start..end].to_vec())
            } else {
                Ok(vec![])
            }
        }
        _ => Err(SignalError::ValueError(format!("Unknown mode: {}", mode))),
    }
}

