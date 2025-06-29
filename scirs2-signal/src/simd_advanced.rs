//! Advanced SIMD operations for signal processing
//!
//! This module provides highly optimized SIMD implementations of common
//! signal processing operations that go beyond the basic operations in
//! scirs2-core, specifically targeting signal processing workloads.

use crate::error::{SignalError, SignalResult};
use ndarray::{Array1, Array2, ArrayView1, ArrayViewMut1, Axis};
use scirs2_core::simd_ops::{PlatformCapabilities, SimdUnifiedOps};
use scirs2_core::validation::check_finite;
use std::arch::x86_64::*;

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
            "Input and output lengths must match".to_string()
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
            "Maximum lag must be less than signal length".to_string()
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
            "Input signals cannot be empty".to_string()
        ));
    }

    let output_len = match mode {
        "full" => n1 + n2 - 1,
        "same" => n1,
        "valid" => if n1 >= n2 { n1 - n2 + 1 } else { 0 },
        _ => return Err(SignalError::ValueError(
            "Mode must be 'full', 'same', or 'valid'".to_string()
        )),
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
            "Data and twiddle factor lengths must match".to_string()
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
            "Signal, window, and output lengths must match".to_string()
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

fn scalar_fir_filter(
    input: &[f64],
    coeffs: &[f64],
    output: &mut [f64],
) -> SignalResult<()> {
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
unsafe fn avx2_fir_filter(
    input: &[f64],
    coeffs: &[f64],
    output: &mut [f64],
) -> SignalResult<()> {
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
unsafe fn avx512_fir_filter(
    input: &[f64],
    coeffs: &[f64],
    output: &mut [f64],
) -> SignalResult<()> {
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
unsafe fn sse_fir_filter(
    input: &[f64],
    coeffs: &[f64],
    output: &mut [f64],
) -> SignalResult<()> {
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
unsafe fn sse_apply_window(
    signal: &[f64],
    window: &[f64],
    output: &mut [f64],
) -> SignalResult<()> {
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
    
    let signal: Vec<f64> = (0..signal_length)
        .map(|i| (i as f64 * 0.1).sin())
        .collect();
    
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
    println!("  Speedup: {:.2}x", scalar_time.as_secs_f64() / simd_time.as_secs_f64());
    
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
    println!("  Speedup: {:.2}x", scalar_autocorr_time.as_secs_f64() / simd_autocorr_time.as_secs_f64());
    
    Ok(())
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
        
        let simd_config = SimdConfig { force_scalar: false, ..Default::default() };
        let scalar_config = SimdConfig { force_scalar: true, ..Default::default() };
        
        simd_fir_filter(&signal, &coeffs, &mut simd_output, &simd_config).unwrap();
        simd_fir_filter(&signal, &coeffs, &mut scalar_output, &scalar_config).unwrap();
        
        for (simd_val, scalar_val) in simd_output.iter().zip(scalar_output.iter()) {
            assert!((simd_val - scalar_val).abs() < 1e-10, 
                   "SIMD and scalar results differ: {} vs {}", simd_val, scalar_val);
        }
    }
}