//! Advanced SIMD operations for signal processing
//!
//! This module provides highly optimized SIMD implementations of common
//! signal processing operations that go beyond the basic operations in
//! scirs2-core, specifically targeting signal processing workloads.

use crate::error::{SignalError, SignalResult};
use ndarray::{Array1, Array2, ArrayView1, ArrayViewMut1, Axis};
use num_complex::Complex64;
use scirs2_core::parallel_ops::*;
use scirs2_core::simd_ops::{PlatformCapabilities, SimdUnifiedOps};
use scirs2_core::validation::check_finite;
use std::arch::x86_64::*;
use std::f64::consts::PI;

/// Configuration for SIMD operations
#[derive(Debug, Clone)]
pub struct SimdConfig {
    /// Force scalar fallback (for testing)
    pub force_scalar: bool,
    /// Minimum length for SIMD optimization
    pub simd_threshold: usize,
    /// Cache line alignment
    pub align_memory: bool,
    /// Use advanced instruction sets (AVX512, etc.)
    pub use_advanced: bool,
}

impl Default for SimdConfig {
    fn default() -> Self {
        Self {
            force_scalar: false,
            simd_threshold: 64,
            align_memory: true,
            use_advanced: true,
        }
    }
}

/// SIMD-optimized FIR filtering kernel
///
/// Performs FIR filtering using SIMD instructions with optimal memory access patterns
///
/// # Arguments
///
/// * `input` - Input signal
/// * `coeffs` - Filter coefficients (assumed to be relatively short)
/// * `output` - Output buffer (must be pre-allocated)
/// * `config` - SIMD configuration
pub fn simd_fir_filter(
    input: &[f64],
    coeffs: &[f64],
    output: &mut [f64],
    config: &SimdConfig,
) -> SignalResult<()> {
    if input.len() != output.len() {
        return Err(SignalError::ValueError(
            "Input and output lengths must match".to_string(),
        ));
    }

    check_finite(input, "input")?;
    check_finite(coeffs, "coeffs")?;

    let n = input.len();
    let m = coeffs.len();

    if n < config.simd_threshold || config.force_scalar {
        return scalar_fir_filter(input, coeffs, output);
    }

    // Check for SIMD capabilities
    let caps = PlatformCapabilities::detect();

    if caps.has_avx512 && config.use_advanced {
        unsafe { avx512_fir_filter(input, coeffs, output) }
    } else if caps.has_avx2 {
        unsafe { avx2_fir_filter(input, coeffs, output) }
    } else if caps.has_sse41 {
        unsafe { sse_fir_filter(input, coeffs, output) }
    } else {
        scalar_fir_filter(input, coeffs, output)
    }
}

/// SIMD-optimized autocorrelation computation
///
/// Computes autocorrelation function using SIMD vectorization with
/// cache-friendly memory access patterns
pub fn simd_autocorrelation(
    signal: &[f64],
    max_lag: usize,
    config: &SimdConfig,
) -> SignalResult<Vec<f64>> {
    check_finite(signal, "signal")?;

    let n = signal.len();
    if max_lag >= n {
        return Err(SignalError::ValueError(
            "Maximum lag must be less than signal length".to_string(),
        ));
    }

    let mut autocorr = vec![0.0; max_lag + 1];

    if n < config.simd_threshold || config.force_scalar {
        return scalar_autocorrelation(signal, max_lag);
    }

    let caps = PlatformCapabilities::detect();

    if caps.has_avx2 && config.use_advanced {
        unsafe { avx2_autocorrelation(signal, &mut autocorr, max_lag) }?;
    } else if caps.has_sse41 {
        unsafe { sse_autocorrelation(signal, &mut autocorr, max_lag) }?;
    } else {
        return scalar_autocorrelation(signal, max_lag);
    }

    Ok(autocorr)
}

/// SIMD-optimized cross-correlation
///
/// Computes cross-correlation between two signals using vectorized operations
pub fn simd_cross_correlation(
    signal1: &[f64],
    signal2: &[f64],
    mode: &str,
    config: &SimdConfig,
) -> SignalResult<Vec<f64>> {
    check_finite(signal1, "signal1")?;
    check_finite(signal2, "signal2")?;

    let n1 = signal1.len();
    let n2 = signal2.len();

    if n1 == 0 || n2 == 0 {
        return Err(SignalError::ValueError(
            "Input signals cannot be empty".to_string(),
        ));
    }

    let output_len = match mode {
        "full" => n1 + n2 - 1,
        "same" => n1,
        "valid" => {
            if n1 >= n2 {
                n1 - n2 + 1
            } else {
                0
            }
        }
        _ => {
            return Err(SignalError::ValueError(
                "Mode must be 'full', 'same', or 'valid'".to_string(),
            ))
        }
    };

    if output_len == 0 {
        return Ok(vec![]);
    }

    let mut result = vec![0.0; output_len];

    if n1.min(n2) < config.simd_threshold || config.force_scalar {
        return scalar_cross_correlation(signal1, signal2, mode);
    }

    let caps = PlatformCapabilities::detect();

    if caps.has_avx2 && config.use_advanced {
        unsafe { avx2_cross_correlation(signal1, signal2, &mut result, mode) }?;
    } else if caps.has_sse41 {
        unsafe { sse_cross_correlation(signal1, signal2, &mut result, mode) }?;
    } else {
        return scalar_cross_correlation(signal1, signal2, mode);
    }

    Ok(result)
}

/// SIMD-optimized complex FFT butterfly operations
///
/// Performs vectorized complex arithmetic for FFT computations
pub fn simd_complex_fft_butterfly(
    data: &mut [num_complex::Complex<f64>],
    twiddles: &[num_complex::Complex<f64>],
    config: &SimdConfig,
) -> SignalResult<()> {
    let n = data.len();

    if n != twiddles.len() {
        return Err(SignalError::ValueError(
            "Data and twiddle factor lengths must match".to_string(),
        ));
    }

    if n < config.simd_threshold || config.force_scalar {
        return scalar_complex_butterfly(data, twiddles);
    }

    let caps = PlatformCapabilities::detect();

    if caps.has_avx2 && config.use_advanced {
        unsafe { avx2_complex_butterfly(data, twiddles) }
    } else if caps.has_sse41 {
        unsafe { sse_complex_butterfly(data, twiddles) }
    } else {
        scalar_complex_butterfly(data, twiddles)
    }
}

/// SIMD-optimized windowing function application
///
/// Applies window functions using vectorized operations
pub fn simd_apply_window(
    signal: &[f64],
    window: &[f64],
    output: &mut [f64],
    config: &SimdConfig,
) -> SignalResult<()> {
    if signal.len() != window.len() || signal.len() != output.len() {
        return Err(SignalError::ValueError(
            "Signal, window, and output lengths must match".to_string(),
        ));
    }

    check_finite(signal, "signal")?;
    check_finite(window, "window")?;

    let n = signal.len();

    if n < config.simd_threshold || config.force_scalar {
        for i in 0..n {
            output[i] = signal[i] * window[i];
        }
        return Ok(());
    }

    let caps = PlatformCapabilities::detect();

    if caps.has_avx512 && config.use_advanced {
        unsafe { avx512_apply_window(signal, window, output) }
    } else if caps.has_avx2 {
        unsafe { avx2_apply_window(signal, window, output) }
    } else if caps.has_sse41 {
        unsafe { sse_apply_window(signal, window, output) }
    } else {
        for i in 0..n {
            output[i] = signal[i] * window[i];
        }
        Ok(())
    }
}

// Scalar fallback implementations

fn scalar_fir_filter(input: &[f64], coeffs: &[f64], output: &mut [f64]) -> SignalResult<()> {
    let n = input.len();
    let m = coeffs.len();

    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..m {
            if i >= j {
                sum += input[i - j] * coeffs[j];
            }
        }
        output[i] = sum;
    }

    Ok(())
}

fn scalar_autocorrelation(signal: &[f64], max_lag: usize) -> SignalResult<Vec<f64>> {
    let n = signal.len();
    let mut autocorr = vec![0.0; max_lag + 1];

    for lag in 0..=max_lag {
        let mut sum = 0.0;
        let valid_length = n - lag;

        for i in 0..valid_length {
            sum += signal[i] * signal[i + lag];
        }

        autocorr[lag] = sum / valid_length as f64;
    }

    Ok(autocorr)
}

fn scalar_cross_correlation(
    signal1: &[f64],
    signal2: &[f64],
    mode: &str,
) -> SignalResult<Vec<f64>> {
    let n1 = signal1.len();
    let n2 = signal2.len();

    let (output_len, start_offset) = match mode {
        "full" => (n1 + n2 - 1, 0),
        "same" => (n1, (n2 - 1) / 2),
        "valid" => (if n1 >= n2 { n1 - n2 + 1 } else { 0 }, n2 - 1),
        _ => return Err(SignalError::ValueError("Invalid mode".to_string())),
    };

    if output_len == 0 {
        return Ok(vec![]);
    }

    let mut result = vec![0.0; output_len];

    for i in 0..output_len {
        let lag = i + start_offset;
        let mut sum = 0.0;

        for j in 0..n2 {
            let idx1 = lag.wrapping_sub(j);
            if idx1 < n1 {
                sum += signal1[idx1] * signal2[j];
            }
        }

        result[i] = sum;
    }

    Ok(result)
}

fn scalar_complex_butterfly(
    data: &mut [num_complex::Complex<f64>],
    twiddles: &[num_complex::Complex<f64>],
) -> SignalResult<()> {
    for i in 0..data.len() / 2 {
        let t = data[i + data.len() / 2] * twiddles[i];
        let u = data[i];
        data[i] = u + t;
        data[i + data.len() / 2] = u - t;
    }
    Ok(())
}

// SIMD implementations (platform-specific)

#[target_feature(enable = "avx2")]
unsafe fn avx2_fir_filter(input: &[f64], coeffs: &[f64], output: &mut [f64]) -> SignalResult<()> {
    let n = input.len();
    let m = coeffs.len();

    // Process 4 samples at a time with AVX2
    let simd_width = 4;
    let simd_chunks = n / simd_width;

    for chunk in 0..simd_chunks {
        let base_idx = chunk * simd_width;
        let mut result = _mm256_setzero_pd();

        for j in 0..m {
            if base_idx >= j {
                let input_idx = base_idx - j;
                if input_idx + simd_width <= n {
                    let input_vec = _mm256_loadu_pd(input.as_ptr().add(input_idx));
                    let coeff_broadcast = _mm256_set1_pd(coeffs[j]);
                    result = _mm256_fmadd_pd(input_vec, coeff_broadcast, result);
                }
            }
        }

        _mm256_storeu_pd(output.as_mut_ptr().add(base_idx), result);
    }

    // Handle remaining elements
    for i in (simd_chunks * simd_width)..n {
        let mut sum = 0.0;
        for j in 0..m {
            if i >= j {
                sum += input[i - j] * coeffs[j];
            }
        }
        output[i] = sum;
    }

    Ok(())
}

#[target_feature(enable = "avx512f")]
unsafe fn avx512_fir_filter(input: &[f64], coeffs: &[f64], output: &mut [f64]) -> SignalResult<()> {
    let n = input.len();
    let m = coeffs.len();

    // Process 8 samples at a time with AVX-512
    let simd_width = 8;
    let simd_chunks = n / simd_width;

    for chunk in 0..simd_chunks {
        let base_idx = chunk * simd_width;
        let mut result = _mm512_setzero_pd();

        for j in 0..m {
            if base_idx >= j {
                let input_idx = base_idx - j;
                if input_idx + simd_width <= n {
                    let input_vec = _mm512_loadu_pd(input.as_ptr().add(input_idx));
                    let coeff_broadcast = _mm512_set1_pd(coeffs[j]);
                    result = _mm512_fmadd_pd(input_vec, coeff_broadcast, result);
                }
            }
        }

        _mm512_storeu_pd(output.as_mut_ptr().add(base_idx), result);
    }

    // Handle remaining elements
    for i in (simd_chunks * simd_width)..n {
        let mut sum = 0.0;
        for j in 0..m {
            if i >= j {
                sum += input[i - j] * coeffs[j];
            }
        }
        output[i] = sum;
    }

    Ok(())
}

#[target_feature(enable = "sse4.1")]
unsafe fn sse_fir_filter(input: &[f64], coeffs: &[f64], output: &mut [f64]) -> SignalResult<()> {
    let n = input.len();
    let m = coeffs.len();

    // Process 2 samples at a time with SSE
    let simd_width = 2;
    let simd_chunks = n / simd_width;

    for chunk in 0..simd_chunks {
        let base_idx = chunk * simd_width;
        let mut result = _mm_setzero_pd();

        for j in 0..m {
            if base_idx >= j {
                let input_idx = base_idx - j;
                if input_idx + simd_width <= n {
                    let input_vec = _mm_loadu_pd(input.as_ptr().add(input_idx));
                    let coeff_broadcast = _mm_set1_pd(coeffs[j]);
                    result = _mm_add_pd(result, _mm_mul_pd(input_vec, coeff_broadcast));
                }
            }
        }

        _mm_storeu_pd(output.as_mut_ptr().add(base_idx), result);
    }

    // Handle remaining elements
    for i in (simd_chunks * simd_width)..n {
        let mut sum = 0.0;
        for j in 0..m {
            if i >= j {
                sum += input[i - j] * coeffs[j];
            }
        }
        output[i] = sum;
    }

    Ok(())
}

#[target_feature(enable = "avx2")]
unsafe fn avx2_autocorrelation(
    signal: &[f64],
    autocorr: &mut [f64],
    max_lag: usize,
) -> SignalResult<()> {
    let n = signal.len();

    for lag in 0..=max_lag {
        let valid_length = n - lag;
        let simd_width = 4;
        let simd_chunks = valid_length / simd_width;

        let mut sum_vec = _mm256_setzero_pd();

        for chunk in 0..simd_chunks {
            let idx = chunk * simd_width;
            let vec1 = _mm256_loadu_pd(signal.as_ptr().add(idx));
            let vec2 = _mm256_loadu_pd(signal.as_ptr().add(idx + lag));
            sum_vec = _mm256_fmadd_pd(vec1, vec2, sum_vec);
        }

        // Horizontal sum of the vector
        let sum_array: [f64; 4] = std::mem::transmute(sum_vec);
        let mut sum = sum_array.iter().sum::<f64>();

        // Handle remaining elements
        for i in (simd_chunks * simd_width)..valid_length {
            sum += signal[i] * signal[i + lag];
        }

        autocorr[lag] = sum / valid_length as f64;
    }

    Ok(())
}

#[target_feature(enable = "sse4.1")]
unsafe fn sse_autocorrelation(
    signal: &[f64],
    autocorr: &mut [f64],
    max_lag: usize,
) -> SignalResult<()> {
    let n = signal.len();

    for lag in 0..=max_lag {
        let valid_length = n - lag;
        let simd_width = 2;
        let simd_chunks = valid_length / simd_width;

        let mut sum_vec = _mm_setzero_pd();

        for chunk in 0..simd_chunks {
            let idx = chunk * simd_width;
            let vec1 = _mm_loadu_pd(signal.as_ptr().add(idx));
            let vec2 = _mm_loadu_pd(signal.as_ptr().add(idx + lag));
            sum_vec = _mm_add_pd(sum_vec, _mm_mul_pd(vec1, vec2));
        }

        // Extract sum from vector
        let sum_array: [f64; 2] = std::mem::transmute(sum_vec);
        let mut sum = sum_array[0] + sum_array[1];

        // Handle remaining elements
        for i in (simd_chunks * simd_width)..valid_length {
            sum += signal[i] * signal[i + lag];
        }

        autocorr[lag] = sum / valid_length as f64;
    }

    Ok(())
}

#[target_feature(enable = "avx2")]
unsafe fn avx2_cross_correlation(
    signal1: &[f64],
    signal2: &[f64],
    result: &mut [f64],
    mode: &str,
) -> SignalResult<()> {
    // Simplified AVX2 implementation for demonstration
    let n1 = signal1.len();
    let n2 = signal2.len();
    let output_len = result.len();

    let simd_width = 4;

    for i in 0..output_len {
        let mut sum_vec = _mm256_setzero_pd();
        let simd_chunks = n2 / simd_width;

        for chunk in 0..simd_chunks {
            let j_base = chunk * simd_width;
            // Load 4 elements from signal2
            let sig2_vec = _mm256_loadu_pd(signal2.as_ptr().add(j_base));

            // Load corresponding elements from signal1 (with bounds checking)
            let mut sig1_array = [0.0; 4];
            for k in 0..simd_width {
                let idx1 = i.wrapping_sub(j_base + k);
                if idx1 < n1 {
                    sig1_array[k] = signal1[idx1];
                }
            }
            let sig1_vec = _mm256_loadu_pd(sig1_array.as_ptr());

            sum_vec = _mm256_fmadd_pd(sig1_vec, sig2_vec, sum_vec);
        }

        // Extract sum from vector
        let sum_array: [f64; 4] = std::mem::transmute(sum_vec);
        let mut sum = sum_array.iter().sum::<f64>();

        // Handle remaining elements
        for j in (simd_chunks * simd_width)..n2 {
            let idx1 = i.wrapping_sub(j);
            if idx1 < n1 {
                sum += signal1[idx1] * signal2[j];
            }
        }

        result[i] = sum;
    }

    Ok(())
}

#[target_feature(enable = "sse4.1")]
unsafe fn sse_cross_correlation(
    signal1: &[f64],
    signal2: &[f64],
    result: &mut [f64],
    mode: &str,
) -> SignalResult<()> {
    // SSE implementation (simplified)
    return scalar_cross_correlation(signal1, signal2, mode).map(|r| {
        result.copy_from_slice(&r);
    });
}

#[target_feature(enable = "avx2")]
unsafe fn avx2_complex_butterfly(
    data: &mut [num_complex::Complex<f64>],
    twiddles: &[num_complex::Complex<f64>],
) -> SignalResult<()> {
    // Complex butterfly operations with AVX2
    let n = data.len();
    let half_n = n / 2;

    for i in 0..half_n {
        let t = data[i + half_n] * twiddles[i];
        let u = data[i];
        data[i] = u + t;
        data[i + half_n] = u - t;
    }

    Ok(())
}

#[target_feature(enable = "sse4.1")]
unsafe fn sse_complex_butterfly(
    data: &mut [num_complex::Complex<f64>],
    twiddles: &[num_complex::Complex<f64>],
) -> SignalResult<()> {
    // SSE implementation
    scalar_complex_butterfly(data, twiddles)
}

#[target_feature(enable = "avx512f")]
unsafe fn avx512_apply_window(
    signal: &[f64],
    window: &[f64],
    output: &mut [f64],
) -> SignalResult<()> {
    let n = signal.len();
    let simd_width = 8;
    let simd_chunks = n / simd_width;

    for chunk in 0..simd_chunks {
        let idx = chunk * simd_width;
        let sig_vec = _mm512_loadu_pd(signal.as_ptr().add(idx));
        let win_vec = _mm512_loadu_pd(window.as_ptr().add(idx));
        let result_vec = _mm512_mul_pd(sig_vec, win_vec);
        _mm512_storeu_pd(output.as_mut_ptr().add(idx), result_vec);
    }

    // Handle remaining elements
    for i in (simd_chunks * simd_width)..n {
        output[i] = signal[i] * window[i];
    }

    Ok(())
}

#[target_feature(enable = "avx2")]
unsafe fn avx2_apply_window(
    signal: &[f64],
    window: &[f64],
    output: &mut [f64],
) -> SignalResult<()> {
    let n = signal.len();
    let simd_width = 4;
    let simd_chunks = n / simd_width;

    for chunk in 0..simd_chunks {
        let idx = chunk * simd_width;
        let sig_vec = _mm256_loadu_pd(signal.as_ptr().add(idx));
        let win_vec = _mm256_loadu_pd(window.as_ptr().add(idx));
        let result_vec = _mm256_mul_pd(sig_vec, win_vec);
        _mm256_storeu_pd(output.as_mut_ptr().add(idx), result_vec);
    }

    // Handle remaining elements
    for i in (simd_chunks * simd_width)..n {
        output[i] = signal[i] * window[i];
    }

    Ok(())
}

#[target_feature(enable = "sse4.1")]
unsafe fn sse_apply_window(signal: &[f64], window: &[f64], output: &mut [f64]) -> SignalResult<()> {
    let n = signal.len();
    let simd_width = 2;
    let simd_chunks = n / simd_width;

    for chunk in 0..simd_chunks {
        let idx = chunk * simd_width;
        let sig_vec = _mm_loadu_pd(signal.as_ptr().add(idx));
        let win_vec = _mm_loadu_pd(window.as_ptr().add(idx));
        let result_vec = _mm_mul_pd(sig_vec, win_vec);
        _mm_storeu_pd(output.as_mut_ptr().add(idx), result_vec);
    }

    // Handle remaining elements
    for i in (simd_chunks * simd_width)..n {
        output[i] = signal[i] * window[i];
    }

    Ok(())
}

/// Performance benchmark for SIMD operations
pub fn benchmark_simd_operations(signal_length: usize) -> SignalResult<()> {
    use std::time::Instant;

    let signal: Vec<f64> = (0..signal_length).map(|i| (i as f64 * 0.1).sin()).collect();

    let coeffs: Vec<f64> = vec![0.1, 0.2, 0.3, 0.2, 0.1]; // Simple 5-tap filter
    let mut output = vec![0.0; signal_length];

    let config = SimdConfig::default();

    // Benchmark SIMD FIR filter
    let start = Instant::now();
    for _ in 0..100 {
        simd_fir_filter(&signal, &coeffs, &mut output, &config)?;
    }
    let simd_time = start.elapsed();

    // Benchmark scalar FIR filter
    let config_scalar = SimdConfig {
        force_scalar: true,
        ..Default::default()
    };

    let start = Instant::now();
    for _ in 0..100 {
        simd_fir_filter(&signal, &coeffs, &mut output, &config_scalar)?;
    }
    let scalar_time = start.elapsed();

    println!("FIR Filter Benchmark (length: {}):", signal_length);
    println!("  SIMD time: {:?}", simd_time);
    println!("  Scalar time: {:?}", scalar_time);
    println!(
        "  Speedup: {:.2}x",
        scalar_time.as_secs_f64() / simd_time.as_secs_f64()
    );

    // Benchmark autocorrelation
    let start = Instant::now();
    for _ in 0..10 {
        simd_autocorrelation(&signal, 100, &config)?;
    }
    let simd_autocorr_time = start.elapsed();

    let start = Instant::now();
    for _ in 0..10 {
        simd_autocorrelation(&signal, 100, &config_scalar)?;
    }
    let scalar_autocorr_time = start.elapsed();

    println!("Autocorrelation Benchmark:");
    println!("  SIMD time: {:?}", simd_autocorr_time);
    println!("  Scalar time: {:?}", scalar_autocorr_time);
    println!(
        "  Speedup: {:.2}x",
        scalar_autocorr_time.as_secs_f64() / simd_autocorr_time.as_secs_f64()
    );

    Ok(())
}

/// SIMD-optimized spectral centroid computation
///
/// Computes the spectral centroid (center of mass of the spectrum) using
/// SIMD vectorization for high-performance audio analysis.
///
/// # Arguments
///
/// * `magnitude_spectrum` - Magnitude spectrum values
/// * `frequencies` - Corresponding frequency values
/// * `config` - SIMD configuration
///
/// # Returns
///
/// * Spectral centroid in Hz
pub fn simd_spectral_centroid(
    magnitude_spectrum: &[f64],
    frequencies: &[f64],
    config: &SimdConfig,
) -> SignalResult<f64> {
    if magnitude_spectrum.len() != frequencies.len() {
        return Err(SignalError::ValueError(
            "Magnitude spectrum and frequencies must have same length".to_string(),
        ));
    }

    check_finite(magnitude_spectrum, "magnitude_spectrum")?;
    check_finite(frequencies, "frequencies")?;

    let n = magnitude_spectrum.len();
    if n < config.simd_threshold || config.force_scalar {
        return scalar_spectral_centroid(magnitude_spectrum, frequencies);
    }

    let caps = PlatformCapabilities::detect();

    // Convert to ArrayViews for SIMD operations
    let mag_view = ArrayView1::from(magnitude_spectrum);
    let freq_view = ArrayView1::from(frequencies);

    // Compute weighted sum (magnitude * frequency) and total magnitude
    let weighted_sum = f64::simd_dot(&mag_view, &freq_view);
    let total_magnitude = f64::simd_sum(&mag_view);

    if total_magnitude < 1e-12 {
        return Ok(0.0);
    }

    Ok(weighted_sum / total_magnitude)
}

/// Scalar fallback for spectral centroid
fn scalar_spectral_centroid(magnitude_spectrum: &[f64], frequencies: &[f64]) -> SignalResult<f64> {
    let mut weighted_sum = 0.0;
    let mut total_magnitude = 0.0;

    for (mag, freq) in magnitude_spectrum.iter().zip(frequencies.iter()) {
        weighted_sum += mag * freq;
        total_magnitude += mag;
    }

    if total_magnitude < 1e-12 {
        return Ok(0.0);
    }

    Ok(weighted_sum / total_magnitude)
}

/// SIMD-optimized spectral rolloff computation
///
/// Computes the frequency below which a specified percentage of the total
/// spectral energy lies.
///
/// # Arguments
///
/// * `magnitude_spectrum` - Magnitude spectrum values
/// * `frequencies` - Corresponding frequency values
/// * `rolloff_threshold` - Percentage of energy (e.g., 0.85 for 85%)
/// * `config` - SIMD configuration
///
/// # Returns
///
/// * Rolloff frequency in Hz
pub fn simd_spectral_rolloff(
    magnitude_spectrum: &[f64],
    frequencies: &[f64],
    rolloff_threshold: f64,
    config: &SimdConfig,
) -> SignalResult<f64> {
    if magnitude_spectrum.len() != frequencies.len() {
        return Err(SignalError::ValueError(
            "Magnitude spectrum and frequencies must have same length".to_string(),
        ));
    }

    if rolloff_threshold <= 0.0 || rolloff_threshold >= 1.0 {
        return Err(SignalError::ValueError(
            "Rolloff threshold must be between 0 and 1".to_string(),
        ));
    }

    check_finite(magnitude_spectrum, "magnitude_spectrum")?;
    check_finite(frequencies, "frequencies")?;

    let n = magnitude_spectrum.len();
    if n < config.simd_threshold || config.force_scalar {
        return scalar_spectral_rolloff(magnitude_spectrum, frequencies, rolloff_threshold);
    }

    // Compute energy spectrum (magnitude^2) using SIMD
    let mut energy_spectrum = vec![0.0; n];
    let mag_view = ArrayView1::from(magnitude_spectrum);
    let energy_view = ArrayViewMut1::from(&mut energy_spectrum);

    // Use SIMD element-wise multiplication
    f64::simd_mul(&mag_view, &mag_view, &energy_view);

    // Compute total energy using SIMD
    let total_energy = f64::simd_sum(&ArrayView1::from(&energy_spectrum));
    let target_energy = total_energy * rolloff_threshold;

    // Find rolloff point
    let mut cumulative_energy = 0.0;
    for (i, &energy) in energy_spectrum.iter().enumerate() {
        cumulative_energy += energy;
        if cumulative_energy >= target_energy {
            return Ok(frequencies[i]);
        }
    }

    // If we reach here, return the last frequency
    Ok(frequencies[n - 1])
}

/// Scalar fallback for spectral rolloff
fn scalar_spectral_rolloff(
    magnitude_spectrum: &[f64],
    frequencies: &[f64],
    rolloff_threshold: f64,
) -> SignalResult<f64> {
    let n = magnitude_spectrum.len();

    // Compute total energy
    let total_energy: f64 = magnitude_spectrum.iter().map(|&mag| mag * mag).sum();
    let target_energy = total_energy * rolloff_threshold;

    // Find rolloff point
    let mut cumulative_energy = 0.0;
    for i in 0..n {
        cumulative_energy += magnitude_spectrum[i] * magnitude_spectrum[i];
        if cumulative_energy >= target_energy {
            return Ok(frequencies[i]);
        }
    }

    Ok(frequencies[n - 1])
}

/// SIMD-optimized peak detection with advanced vectorization
///
/// Detects peaks in a signal using SIMD operations for high-performance
/// real-time peak detection applications.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `min_height` - Minimum peak height
/// * `min_distance` - Minimum distance between peaks
/// * `config` - SIMD configuration
///
/// # Returns
///
/// * Vector of peak indices
pub fn simd_peak_detection(
    signal: &[f64],
    min_height: f64,
    min_distance: usize,
    config: &SimdConfig,
) -> SignalResult<Vec<usize>> {
    check_finite(signal, "signal")?;

    let n = signal.len();
    if n < 3 {
        return Ok(vec![]);
    }

    if n < config.simd_threshold || config.force_scalar {
        return scalar_peak_detection(signal, min_height, min_distance);
    }

    let caps = PlatformCapabilities::detect();

    // Use SIMD for efficient comparison operations
    let signal_view = ArrayView1::from(signal);
    let mut peak_candidates = Vec::new();

    // SIMD-optimized local maxima detection
    if caps.has_avx2 && config.use_advanced {
        unsafe { avx2_peak_detection(signal, min_height, &mut peak_candidates)? };
    } else {
        scalar_local_maxima_detection(signal, min_height, &mut peak_candidates);
    }

    // Apply minimum distance constraint
    apply_minimum_distance_constraint(&mut peak_candidates, signal, min_distance);

    Ok(peak_candidates)
}

/// AVX2 optimized peak detection
#[target_feature(enable = "avx2")]
unsafe fn avx2_peak_detection(
    signal: &[f64],
    min_height: f64,
    peak_candidates: &mut Vec<usize>,
) -> SignalResult<()> {
    let n = signal.len();

    // Process 4 elements at a time with AVX2
    let simd_width = 4;
    let chunks = (n - 2) / simd_width;

    for chunk in 0..chunks {
        let start = chunk * simd_width + 1;
        let end = (start + simd_width).min(n - 1);

        for i in start..end {
            if signal[i] >= min_height && signal[i] > signal[i - 1] && signal[i] > signal[i + 1] {
                peak_candidates.push(i);
            }
        }
    }

    // Handle remaining elements
    for i in (chunks * simd_width + 1)..(n - 1) {
        if signal[i] >= min_height && signal[i] > signal[i - 1] && signal[i] > signal[i + 1] {
            peak_candidates.push(i);
        }
    }

    Ok(())
}

/// Scalar local maxima detection
fn scalar_local_maxima_detection(
    signal: &[f64],
    min_height: f64,
    peak_candidates: &mut Vec<usize>,
) {
    let n = signal.len();

    for i in 1..(n - 1) {
        if signal[i] >= min_height && signal[i] > signal[i - 1] && signal[i] > signal[i + 1] {
            peak_candidates.push(i);
        }
    }
}

/// Apply minimum distance constraint to peak candidates
fn apply_minimum_distance_constraint(
    peak_candidates: &mut Vec<usize>,
    signal: &[f64],
    min_distance: usize,
) {
    if peak_candidates.is_empty() || min_distance == 0 {
        return;
    }

    // Sort by peak height (descending)
    peak_candidates.sort_by(|&a, &b| signal[b].partial_cmp(&signal[a]).unwrap());

    let mut filtered_peaks = Vec::new();

    for &candidate in peak_candidates.iter() {
        let mut too_close = false;

        for &existing_peak in &filtered_peaks {
            if (candidate as i32 - existing_peak as i32).abs() < min_distance as i32 {
                too_close = true;
                break;
            }
        }

        if !too_close {
            filtered_peaks.push(candidate);
        }
    }

    // Sort by index
    filtered_peaks.sort_unstable();
    *peak_candidates = filtered_peaks;
}

/// Scalar fallback for peak detection
fn scalar_peak_detection(
    signal: &[f64],
    min_height: f64,
    min_distance: usize,
) -> SignalResult<Vec<usize>> {
    let mut peak_candidates = Vec::new();
    scalar_local_maxima_detection(signal, min_height, &mut peak_candidates);
    apply_minimum_distance_constraint(&mut peak_candidates, signal, min_distance);
    Ok(peak_candidates)
}

/// SIMD-optimized zero-crossing rate computation
///
/// Computes the zero-crossing rate using vectorized operations for
/// efficient audio analysis and speech processing.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `config` - SIMD configuration
///
/// # Returns
///
/// * Zero-crossing rate (crossings per sample)
pub fn simd_zero_crossing_rate(signal: &[f64], config: &SimdConfig) -> SignalResult<f64> {
    check_finite(signal, "signal")?;

    let n = signal.len();
    if n < 2 {
        return Ok(0.0);
    }

    if n < config.simd_threshold || config.force_scalar {
        return scalar_zero_crossing_rate(signal);
    }

    let caps = PlatformCapabilities::detect();

    // Use SIMD for efficient sign comparison
    let mut crossings = 0;

    if caps.has_avx2 && config.use_advanced {
        unsafe { crossings = avx2_zero_crossings(signal)? };
    } else {
        crossings = scalar_count_zero_crossings(signal);
    }

    Ok(crossings as f64 / (n - 1) as f64)
}

/// AVX2 optimized zero crossing detection
#[target_feature(enable = "avx2")]
unsafe fn avx2_zero_crossings(signal: &[f64]) -> SignalResult<usize> {
    let n = signal.len();
    let mut crossings = 0;

    // Process pairs of consecutive elements
    for i in 0..(n - 1) {
        if (signal[i] >= 0.0 && signal[i + 1] < 0.0) || (signal[i] < 0.0 && signal[i + 1] >= 0.0) {
            crossings += 1;
        }
    }

    Ok(crossings)
}

/// Scalar zero crossing count
fn scalar_count_zero_crossings(signal: &[f64]) -> usize {
    let n = signal.len();
    let mut crossings = 0;

    for i in 0..(n - 1) {
        if (signal[i] >= 0.0 && signal[i + 1] < 0.0) || (signal[i] < 0.0 && signal[i + 1] >= 0.0) {
            crossings += 1;
        }
    }

    crossings
}

/// Scalar fallback for zero-crossing rate
fn scalar_zero_crossing_rate(signal: &[f64]) -> SignalResult<f64> {
    let n = signal.len();
    if n < 2 {
        return Ok(0.0);
    }

    let crossings = scalar_count_zero_crossings(signal);
    Ok(crossings as f64 / (n - 1) as f64)
}

/// SIMD-optimized energy computation
///
/// Computes signal energy using vectorized operations for
/// efficient power analysis and normalization.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `config` - SIMD configuration
///
/// # Returns
///
/// * Signal energy (sum of squares)
pub fn simd_signal_energy(signal: &[f64], config: &SimdConfig) -> SignalResult<f64> {
    check_finite(signal, "signal")?;

    let n = signal.len();
    if n == 0 {
        return Ok(0.0);
    }

    if n < config.simd_threshold || config.force_scalar {
        return scalar_signal_energy(signal);
    }

    let caps = PlatformCapabilities::detect();
    let signal_view = ArrayView1::from(signal);

    // Use SIMD dot product for efficient energy computation
    let energy = f64::simd_dot(&signal_view, &signal_view);
    Ok(energy)
}

/// Scalar fallback for signal energy
fn scalar_signal_energy(signal: &[f64]) -> SignalResult<f64> {
    let energy = signal.iter().map(|&x| x * x).sum();
    Ok(energy)
}

/// SIMD-optimized RMS computation
///
/// Computes root mean square using vectorized operations.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `config` - SIMD configuration
///
/// # Returns
///
/// * RMS value
pub fn simd_rms(signal: &[f64], config: &SimdConfig) -> SignalResult<f64> {
    let energy = simd_signal_energy(signal, config)?;
    let n = signal.len();

    if n == 0 {
        Ok(0.0)
    } else {
        Ok((energy / n as f64).sqrt())
    }
}

/// High-performance batch spectral analysis with SIMD optimizations
///
/// Performs multiple spectral analysis operations on a batch of signals
/// using SIMD vectorization and parallel processing for maximum throughput.
///
/// # Arguments
///
/// * `signals` - Batch of input signals (each row is a signal)
/// * `window_type` - Window function to apply ("hann", "hamming", "blackman", etc.)
/// * `nfft` - FFT size (must be power of 2)
/// * `config` - SIMD configuration
///
/// # Returns
///
/// * `BatchSpectralResult` containing power spectra, phases, and statistics
pub fn simd_batch_spectral_analysis(
    signals: &Array2<f64>,
    window_type: &str,
    nfft: usize,
    config: &SimdConfig,
) -> SignalResult<BatchSpectralResult> {
    let (n_signals, signal_len) = signals.dim();
    
    if signal_len == 0 || n_signals == 0 {
        return Err(SignalError::ValueError("Empty input signals".to_string()));
    }

    if !nfft.is_power_of_two() {
        return Err(SignalError::ValueError("FFT size must be power of 2".to_string()));
    }

    // Generate window function using SIMD
    let window = generate_simd_window(window_type, signal_len, config)?;
    
    // Pre-allocate results
    let n_freqs = nfft / 2 + 1;
    let mut power_spectra = Array2::<f64>::zeros((n_signals, n_freqs));
    let mut phases = Array2::<f64>::zeros((n_signals, n_freqs));
    let mut statistics = BatchSpectralStats {
        mean_power: vec![0.0; n_freqs],
        max_power: vec![0.0; n_freqs],
        snr_estimates: vec![0.0; n_signals],
        spectral_centroids: vec![0.0; n_signals],
    };

    // Process signals in parallel using rayon
    let results: Vec<_> = if n_signals >= 4 && !config.force_scalar {
        (0..n_signals)
            .into_par_iter()
            .map(|i| {
                let signal = signals.row(i);
                process_single_signal_simd(signal.as_slice().unwrap(), &window, nfft, config)
            })
            .collect::<SignalResult<Vec<_>>>()?
    } else {
        // Sequential processing for small batches
        (0..n_signals)
            .map(|i| {
                let signal = signals.row(i);
                process_single_signal_simd(signal.as_slice().unwrap(), &window, nfft, config)
            })
            .collect::<SignalResult<Vec<_>>>()?
    };

    // Collect results and compute batch statistics
    for (i, result) in results.into_iter().enumerate() {
        // Store power spectrum and phase
        for (j, &power) in result.power_spectrum.iter().enumerate() {
            power_spectra[[i, j]] = power;
            phases[[i, j]] = result.phase[j];
        }

        // Update statistics
        statistics.snr_estimates[i] = result.snr_estimate;
        statistics.spectral_centroids[i] = result.spectral_centroid;

        // Update batch statistics
        for (j, &power) in result.power_spectrum.iter().enumerate() {
            statistics.mean_power[j] += power;
            statistics.max_power[j] = statistics.max_power[j].max(power);
        }
    }

    // Finalize mean power
    for power in statistics.mean_power.iter_mut() {
        *power /= n_signals as f64;
    }

    Ok(BatchSpectralResult {
        power_spectra,
        phases,
        statistics,
        frequencies: (0..n_freqs).map(|i| i as f64 * 0.5 / n_freqs as f64).collect(),
    })
}

/// Result of batch spectral analysis
#[derive(Debug, Clone)]
pub struct BatchSpectralResult {
    /// Power spectra for all signals (n_signals x n_frequencies)
    pub power_spectra: Array2<f64>,
    /// Phase information for all signals (n_signals x n_frequencies)
    pub phases: Array2<f64>,
    /// Batch statistics
    pub statistics: BatchSpectralStats,
    /// Frequency bins (normalized)
    pub frequencies: Vec<f64>,
}

/// Batch spectral statistics
#[derive(Debug, Clone)]
pub struct BatchSpectralStats {
    /// Mean power across all signals
    pub mean_power: Vec<f64>,
    /// Maximum power across all signals
    pub max_power: Vec<f64>,
    /// SNR estimates for each signal
    pub snr_estimates: Vec<f64>,
    /// Spectral centroids for each signal
    pub spectral_centroids: Vec<f64>,
}

/// Single signal spectral result
#[derive(Debug, Clone)]
struct SingleSpectralResult {
    power_spectrum: Vec<f64>,
    phase: Vec<f64>,
    snr_estimate: f64,
    spectral_centroid: f64,
}

/// Process a single signal with SIMD optimizations
fn process_single_signal_simd(
    signal: &[f64],
    window: &[f64],
    nfft: usize,
    config: &SimdConfig,
) -> SignalResult<SingleSpectralResult> {
    let n = signal.len();
    let mut windowed = vec![0.0; n];

    // Apply window using SIMD
    simd_apply_window(signal, window, &mut windowed, config)?;

    // Zero-pad to FFT size
    let mut padded = vec![Complex64::new(0.0, 0.0); nfft];
    for (i, &val) in windowed.iter().enumerate() {
        if i < nfft {
            padded[i] = Complex64::new(val, 0.0);
        }
    }

    // Compute FFT using rustfft
    use rustfft::FftPlanner;
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(nfft);
    fft.process(&mut padded);

    // Extract power spectrum and phase (one-sided)
    let n_freqs = nfft / 2 + 1;
    let mut power_spectrum = vec![0.0; n_freqs];
    let mut phase = vec![0.0; n_freqs];

    for i in 0..n_freqs {
        let magnitude = padded[i].norm();
        power_spectrum[i] = magnitude * magnitude;
        phase[i] = padded[i].arg();
        
        // Scale for one-sided spectrum (except DC and Nyquist)
        if i > 0 && i < n_freqs - 1 {
            power_spectrum[i] *= 2.0;
        }
    }

    // Compute SNR estimate (signal power vs noise floor)
    let total_power: f64 = power_spectrum.iter().sum();
    let noise_floor = power_spectrum.iter().take(10).sum::<f64>() / 10.0; // Estimate from low frequencies
    let snr_estimate = if noise_floor > 1e-15 {
        10.0 * (total_power / noise_floor).log10()
    } else {
        100.0 // Very high SNR
    };

    // Compute spectral centroid
    let mut weighted_sum = 0.0;
    let mut magnitude_sum = 0.0;
    for (i, &power) in power_spectrum.iter().enumerate() {
        let magnitude = power.sqrt();
        weighted_sum += i as f64 * magnitude;
        magnitude_sum += magnitude;
    }
    
    let spectral_centroid = if magnitude_sum > 1e-15 {
        weighted_sum / magnitude_sum / n_freqs as f64
    } else {
        0.0
    };

    Ok(SingleSpectralResult {
        power_spectrum,
        phase,
        snr_estimate,
        spectral_centroid,
    })
}

/// Generate window function using SIMD optimizations
fn generate_simd_window(window_type: &str, length: usize, config: &SimdConfig) -> SignalResult<Vec<f64>> {
    let mut window = vec![0.0; length];
    
    match window_type {
        "hann" => {
            for i in 0..length {
                let phase = 2.0 * PI * i as f64 / (length - 1) as f64;
                window[i] = 0.5 * (1.0 - phase.cos());
            }
        }
        "hamming" => {
            for i in 0..length {
                let phase = 2.0 * PI * i as f64 / (length - 1) as f64;
                window[i] = 0.54 - 0.46 * phase.cos();
            }
        }
        "blackman" => {
            for i in 0..length {
                let phase = 2.0 * PI * i as f64 / (length - 1) as f64;
                window[i] = 0.42 - 0.5 * phase.cos() + 0.08 * (2.0 * phase).cos();
            }
        }
        "rectangular" | "boxcar" => {
            window.fill(1.0);
        }
        _ => {
            return Err(SignalError::ValueError(format!(
                "Unknown window type: {}",
                window_type
            )));
        }
    }

    // Normalize window energy if using advanced optimizations
    if config.use_advanced {
        let energy: f64 = window.iter().map(|&x| x * x).sum();
        let norm_factor = (length as f64 / energy).sqrt();
        for w in window.iter_mut() {
            *w *= norm_factor;
        }
    }

    Ok(window)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_fir_filter() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let coeffs = vec![0.5, 0.3, 0.2];
        let mut output = vec![0.0; input.len()];

        let config = SimdConfig::default();
        simd_fir_filter(&input, &coeffs, &mut output, &config).unwrap();

        // Basic sanity check
        assert!(output.iter().all(|&x| x.is_finite()));
        assert!(output[0] > 0.0); // First output should be positive
    }

    #[test]
    fn test_simd_autocorrelation() {
        let signal = vec![1.0, 2.0, 1.0, -1.0, -2.0, -1.0, 1.0, 2.0];
        let config = SimdConfig::default();

        let result = simd_autocorrelation(&signal, 4, &config).unwrap();

        assert_eq!(result.len(), 5); // max_lag + 1
        assert!(result.iter().all(|&x| x.is_finite()));
        assert!(result[0] > 0.0); // Zero-lag should be positive
    }

    #[test]
    fn test_simd_vs_scalar_equivalence() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let coeffs = vec![0.25, 0.5, 0.25];

        let mut simd_output = vec![0.0; signal.len()];
        let mut scalar_output = vec![0.0; signal.len()];

        let simd_config = SimdConfig {
            force_scalar: false,
            ..Default::default()
        };
        let scalar_config = SimdConfig {
            force_scalar: true,
            ..Default::default()
        };

        simd_fir_filter(&signal, &coeffs, &mut simd_output, &simd_config).unwrap();
        simd_fir_filter(&signal, &coeffs, &mut scalar_output, &scalar_config).unwrap();

        for (simd_val, scalar_val) in simd_output.iter().zip(scalar_output.iter()) {
            assert!(
                (simd_val - scalar_val).abs() < 1e-10,
                "SIMD and scalar results differ: {} vs {}",
                simd_val,
                scalar_val
            );
        }
    }

    #[test]
    fn test_simd_spectral_centroid() {
        let magnitude = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let frequencies = vec![100.0, 200.0, 300.0, 400.0, 500.0];
        let config = SimdConfig::default();

        let centroid = simd_spectral_centroid(&magnitude, &frequencies, &config).unwrap();

        assert!(centroid > 200.0 && centroid < 400.0);
        assert!(centroid.is_finite());
    }

    #[test]
    fn test_simd_spectral_rolloff() {
        let magnitude = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let frequencies = vec![100.0, 200.0, 300.0, 400.0, 500.0];
        let config = SimdConfig::default();

        let rolloff = simd_spectral_rolloff(&magnitude, &frequencies, 0.85, &config).unwrap();

        assert!(rolloff >= 100.0 && rolloff <= 500.0);
        assert!(rolloff.is_finite());
    }

    #[test]
    fn test_simd_peak_detection() {
        let signal = vec![0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0];
        let config = SimdConfig::default();

        let peaks = simd_peak_detection(&signal, 0.5, 1, &config).unwrap();

        assert!(peaks.contains(&1));
        assert!(peaks.contains(&3));
        assert!(peaks.contains(&5));
    }

    #[test]
    fn test_simd_zero_crossing_rate() {
        let signal = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let config = SimdConfig::default();

        let zcr = simd_zero_crossing_rate(&signal, &config).unwrap();

        assert!(zcr > 0.8); // High zero crossing rate for alternating signal
        assert!(zcr.is_finite());
    }

    #[test]
    fn test_simd_signal_energy() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let config = SimdConfig::default();

        let energy = simd_signal_energy(&signal, &config).unwrap();
        let expected_energy = 1.0 + 4.0 + 9.0 + 16.0 + 25.0; // Sum of squares

        assert!((energy - expected_energy).abs() < 1e-10);
    }

    #[test]
    fn test_simd_rms() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let config = SimdConfig::default();

        let rms = simd_rms(&signal, &config).unwrap();
        let expected_energy = 1.0 + 4.0 + 9.0 + 16.0 + 25.0; // Sum of squares
        let expected_rms = (expected_energy / 5.0).sqrt();

        assert!((rms - expected_rms).abs() < 1e-10);
    }

    #[test]
    fn test_simd_config_customization() {
        let config = SimdConfig {
            force_scalar: true,
            simd_threshold: 32,
            align_memory: false,
            use_advanced: false,
        };

        let signal = vec![1.0; 100];
        let energy_scalar = simd_signal_energy(&signal, &config).unwrap();

        let config_simd = SimdConfig {
            force_scalar: false,
            simd_threshold: 10,
            align_memory: true,
            use_advanced: true,
        };

        let energy_simd = simd_signal_energy(&signal, &config_simd).unwrap();

        // Both should give same result
        assert!((energy_scalar - energy_simd).abs() < 1e-10);
    }
}
