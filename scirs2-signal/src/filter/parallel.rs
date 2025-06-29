//! Parallel filtering operations using scirs2-core parallel abstractions
//!
//! This module provides parallel implementations of filtering operations
//! for improved performance on multi-core systems.

use crate::error::{SignalError, SignalResult};
use ndarray::{s, Array1, Array2, ArrayView1, Axis};
use num_traits::{Float, NumCast};
use scirs2_core::parallel_ops::*;
use scirs2_core::validation::check_finite;
use std::fmt::Debug;
use std::sync::Arc;

/// Parallel implementation of filtfilt (zero-phase filtering)
///
/// Applies a digital filter forward and backward to achieve zero-phase
/// distortion, using parallel processing for improved performance.
///
/// # Arguments
///
/// * `b` - Numerator coefficients
/// * `a` - Denominator coefficients
/// * `x` - Input signal
/// * `chunk_size` - Size of chunks for parallel processing (None for auto)
///
/// # Returns
///
/// * Zero-phase filtered signal
pub fn parallel_filtfilt<T>(
    b: &[f64],
    a: &[f64],
    x: &[T],
    chunk_size: Option<usize>,
) -> SignalResult<Vec<f64>>
where
    T: Float + NumCast + Debug + Send + Sync,
{
    if a.is_empty() || a[0] == 0.0 {
        return Err(SignalError::ValueError(
            "First denominator coefficient cannot be zero".to_string(),
        ));
    }

    // Convert input to f64 Array1
    let x_array = Array1::from_iter(
        x.iter()
            .map(|&val| {
                NumCast::from(val).ok_or_else(|| {
                    SignalError::ValueError(format!("Could not convert {:?} to f64", val))
                })
            })
            .collect::<SignalResult<Vec<f64>>>()?,
    );

    // Forward filtering with overlap-save method for parallelization
    let forward_filtered = parallel_filter_overlap_save(b, a, &x_array, chunk_size)?;

    // Reverse the signal
    let mut reversed = forward_filtered.to_vec();
    reversed.reverse();
    let reversed_array = Array1::from(reversed);

    // Backward filtering
    let backward_filtered = parallel_filter_overlap_save(b, a, &reversed_array, chunk_size)?;

    // Reverse again to get final result
    let mut result = backward_filtered.to_vec();
    result.reverse();

    Ok(result)
}

/// Parallel convolution using overlap-save method
///
/// Performs convolution of two signals using parallel processing
/// with the overlap-save method for efficiency.
///
/// # Arguments
///
/// * `a` - First signal
/// * `v` - Second signal (kernel)
/// * `mode` - Convolution mode ("full", "same", "valid")
/// * `chunk_size` - Chunk size for parallel processing
///
/// # Returns
///
/// * Convolution result
pub fn parallel_convolve<T, U>(
    a: &[T],
    v: &[U],
    mode: &str,
    chunk_size: Option<usize>,
) -> SignalResult<Vec<f64>>
where
    T: Float + NumCast + Debug + Send + Sync,
    U: Float + NumCast + Debug + Send + Sync,
{
    // Convert inputs to Array1
    let a_array = Array1::from_iter(
        a.iter()
            .map(|&val| NumCast::from(val).unwrap_or(0.0))
            .collect::<Vec<f64>>(),
    );

    let v_array = Array1::from_iter(
        v.iter()
            .map(|&val| NumCast::from(val).unwrap_or(0.0))
            .collect::<Vec<f64>>(),
    );

    // Use overlap-save for efficiency with long signals
    if a_array.len() > 1000 && v_array.len() > 10 {
        parallel_convolve_overlap_save(&a_array, &v_array, mode, chunk_size)
    } else {
        // For small signals, use direct convolution
        parallel_convolve_direct(&a_array, &v_array, mode)
    }
}

/// Overlap-save method for parallel filtering
fn parallel_filter_overlap_save(
    b: &[f64],
    a: &[f64],
    x: &Array1<f64>,
    chunk_size: Option<usize>,
) -> SignalResult<Array1<f64>> {
    let n = x.len();
    let filter_len = b.len().max(a.len());

    // Determine chunk size
    let chunk = chunk_size.unwrap_or_else(|| {
        // Auto-determine based on signal length and available cores
        let n_cores = num_cpus::get();
        ((n / n_cores).max(filter_len * 4)).min(8192)
    });

    // Overlap needed for continuity
    let overlap = filter_len - 1;

    // Process chunks in parallel
    let n_chunks = (n + chunk - overlap - 1) / (chunk - overlap);
    let mut results = vec![Vec::new(); n_chunks];

    par_iter_with_setup(
        0..n_chunks,
        || {},
        |_, &i| {
            let start = i * (chunk - overlap);
            let end = ((start + chunk).min(n)).max(start + 1);

            // Extract chunk with proper overlap
            let chunk_start = start.saturating_sub(overlap);
            let chunk_data = x.slice(s![chunk_start..end]).to_vec();

            // Apply filter to chunk
            let filtered = filter_direct(b, a, &chunk_data)?;

            // Extract valid portion (discard transient response)
            let valid_start = if i == 0 { 0 } else { overlap };
            let valid_filtered = filtered[valid_start..].to_vec();

            Ok(valid_filtered)
        },
        |i, result: SignalResult<Vec<f64>>| {
            results[i] = result?;
            Ok(())
        },
    )?;

    // Concatenate results
    let mut output = Vec::with_capacity(n);
    for chunk_result in results {
        output.extend(chunk_result);
    }

    // Trim to exact length
    output.truncate(n);

    Ok(Array1::from(output))
}

/// Direct filtering implementation (for chunks)
fn filter_direct(b: &[f64], a: &[f64], x: &[f64]) -> SignalResult<Vec<f64>> {
    let n = x.len();
    let nb = b.len();
    let na = a.len();

    // Normalize by a[0]
    let a0 = a[0];
    if a0.abs() < 1e-10 {
        return Err(SignalError::ValueError(
            "First denominator coefficient cannot be zero".to_string(),
        ));
    }

    let mut y = vec![0.0; n];

    for i in 0..n {
        // Feedforward path
        for j in 0..nb.min(i + 1) {
            y[i] += b[j] * x[i - j] / a0;
        }

        // Feedback path
        for j in 1..na.min(i + 1) {
            y[i] -= a[j] * y[i - j] / a0;
        }
    }

    Ok(y)
}

/// Overlap-save convolution for parallel processing
fn parallel_convolve_overlap_save(
    a: &Array1<f64>,
    v: &Array1<f64>,
    mode: &str,
    chunk_size: Option<usize>,
) -> SignalResult<Vec<f64>> {
    let na = a.len();
    let nv = v.len();

    // For overlap-save, process in chunks
    let chunk = chunk_size.unwrap_or(4096);
    let overlap = nv - 1;

    // Full convolution length
    let n_full = na + nv - 1;
    let mut result = vec![0.0; n_full];

    // Process chunks in parallel
    let n_chunks = (na + chunk - overlap - 1) / (chunk - overlap);
    let chunk_results: Vec<Vec<f64>> = par_iter_with_setup(
        0..n_chunks,
        || {},
        |_, &i| {
            let start = i * (chunk - overlap);
            let end = (start + chunk).min(na);

            // Extract chunk with zero padding if needed
            let mut chunk_data = vec![0.0; chunk];
            for j in start..end {
                chunk_data[j - start] = a[j];
            }

            // Convolve chunk with kernel
            let mut chunk_result = vec![0.0; chunk + nv - 1];
            for j in 0..chunk {
                for k in 0..nv {
                    chunk_result[j + k] += chunk_data[j] * v[k];
                }
            }

            Ok(chunk_result)
        },
        |results, chunk_res| {
            results.push(chunk_res?);
            Ok(())
        },
    )?;

    // Combine chunk results
    for (i, chunk_res) in chunk_results.iter().enumerate() {
        let start = i * (chunk - overlap);
        for (j, &val) in chunk_res.iter().enumerate() {
            if start + j < n_full {
                result[start + j] += val;
            }
        }
    }

    // Apply mode
    match mode {
        "full" => Ok(result),
        "same" => {
            let start = (nv - 1) / 2;
            let end = start + na;
            Ok(result[start..end].to_vec())
        }
        "valid" => {
            if nv > na {
                return Err(SignalError::ValueError(
                    "In 'valid' mode, kernel must not be larger than signal".to_string(),
                ));
            }
            let start = nv - 1;
            let end = n_full - (nv - 1);
            Ok(result[start..end].to_vec())
        }
        _ => Err(SignalError::ValueError(format!("Unknown mode: {}", mode))),
    }
}

/// Direct convolution for small signals
fn parallel_convolve_direct(
    a: &Array1<f64>,
    v: &Array1<f64>,
    mode: &str,
) -> SignalResult<Vec<f64>> {
    let na = a.len();
    let nv = v.len();
    let n_full = na + nv - 1;

    // Use parallel iteration for the outer loop
    let result: Vec<f64> = par_iter_with_setup(
        0..n_full,
        || {},
        |_, &i| {
            let mut sum = 0.0;
            for j in 0..nv {
                if i >= j && i - j < na {
                    sum += a[i - j] * v[j];
                }
            }
            Ok(sum)
        },
        |results, val| {
            results.push(val?);
            Ok(())
        },
    )?;

    // Apply mode
    match mode {
        "full" => Ok(result),
        "same" => {
            let start = (nv - 1) / 2;
            let end = start + na;
            Ok(result[start..end].to_vec())
        }
        "valid" => {
            if nv > na {
                return Err(SignalError::ValueError(
                    "In 'valid' mode, kernel must not be larger than signal".to_string(),
                ));
            }
            let start = nv - 1;
            let end = n_full - (nv - 1);
            Ok(result[start..end].to_vec())
        }
        _ => Err(SignalError::ValueError(format!("Unknown mode: {}", mode))),
    }
}

/// Parallel 2D convolution for image filtering
///
/// # Arguments
///
/// * `image` - 2D input array (image)
/// * `kernel` - 2D convolution kernel
/// * `mode` - Convolution mode
/// * `boundary` - Boundary handling ("zero", "reflect", "wrap")
///
/// # Returns
///
/// * Filtered 2D array
pub fn parallel_convolve2d(
    image: &Array2<f64>,
    kernel: &Array2<f64>,
    mode: &str,
    boundary: &str,
) -> SignalResult<Array2<f64>> {
    let (img_rows, img_cols) = image.dim();
    let (ker_rows, ker_cols) = kernel.dim();

    // Validate inputs
    if ker_rows > img_rows || ker_cols > img_cols {
        return Err(SignalError::ValueError(
            "Kernel dimensions must not exceed image dimensions".to_string(),
        ));
    }

    // Determine output size based on mode
    let (out_rows, out_cols) = match mode {
        "full" => (img_rows + ker_rows - 1, img_cols + ker_cols - 1),
        "same" => (img_rows, img_cols),
        "valid" => (img_rows - ker_rows + 1, img_cols - ker_cols + 1),
        _ => return Err(SignalError::ValueError(format!("Unknown mode: {}", mode))),
    };

    // Padding for boundary handling
    let pad_rows = ker_rows - 1;
    let pad_cols = ker_cols - 1;

    // Create padded image based on boundary condition
    let padded = pad_image(image, pad_rows, pad_cols, boundary)?;

    // Parallel convolution over rows
    let result_vec: Vec<Vec<f64>> = par_iter_with_setup(
        0..out_rows,
        || {},
        |_, &i| {
            let mut row_result = vec![0.0; out_cols];

            // Adjust indices based on mode
            let row_offset = match mode {
                "full" => 0,
                "same" => ker_rows / 2,
                "valid" => ker_rows - 1,
                _ => 0,
            };

            let col_offset = match mode {
                "full" => 0,
                "same" => ker_cols / 2,
                "valid" => ker_cols - 1,
                _ => 0,
            };

            for j in 0..out_cols {
                let mut sum = 0.0;

                // Convolution at position (i, j)
                for ki in 0..ker_rows {
                    for kj in 0..ker_cols {
                        let pi = i + row_offset + ki;
                        let pj = j + col_offset + kj;

                        if pi < padded.nrows() && pj < padded.ncols() {
                            sum +=
                                padded[[pi, pj]] * kernel[[ker_rows - 1 - ki, ker_cols - 1 - kj]];
                        }
                    }
                }

                row_result[j] = sum;
            }

            Ok(row_result)
        },
        |results, row| {
            results.push(row?);
            Ok(())
        },
    )?;

    // Convert to Array2
    let mut output = Array2::zeros((out_rows, out_cols));
    for (i, row) in result_vec.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            output[[i, j]] = val;
        }
    }

    Ok(output)
}

/// Pad image for boundary handling
fn pad_image(
    image: &Array2<f64>,
    pad_rows: usize,
    pad_cols: usize,
    boundary: &str,
) -> SignalResult<Array2<f64>> {
    let (rows, cols) = image.dim();
    let padded_rows = rows + 2 * pad_rows;
    let padded_cols = cols + 2 * pad_cols;

    let mut padded = Array2::zeros((padded_rows, padded_cols));

    // Copy original image to center
    for i in 0..rows {
        for j in 0..cols {
            padded[[i + pad_rows, j + pad_cols]] = image[[i, j]];
        }
    }

    // Apply boundary condition
    match boundary {
        "zero" => {
            // Already zero-padded
        }
        "reflect" => {
            // Reflect padding
            // Top and bottom
            for i in 0..pad_rows {
                for j in 0..cols {
                    padded[[i, j + pad_cols]] = image[[pad_rows - i - 1, j]];
                    padded[[rows + pad_rows + i, j + pad_cols]] = image[[rows - i - 1, j]];
                }
            }

            // Left and right (including corners)
            for i in 0..padded_rows {
                for j in 0..pad_cols {
                    let src_i = i.saturating_sub(pad_rows).min(rows - 1);
                    padded[[i, j]] = padded[[i, 2 * pad_cols - j - 1]];
                    padded[[i, cols + pad_cols + j]] = padded[[i, cols + pad_cols - j - 1]];
                }
            }
        }
        "wrap" => {
            // Periodic boundary
            for i in 0..padded_rows {
                for j in 0..padded_cols {
                    let src_i = (i + rows - pad_rows) % rows;
                    let src_j = (j + cols - pad_cols) % cols;
                    padded[[i, j]] = image[[src_i, src_j]];
                }
            }
        }
        _ => {
            return Err(SignalError::ValueError(format!(
                "Unknown boundary condition: {}",
                boundary
            )));
        }
    }

    Ok(padded)
}

/// Parallel Savitzky-Golay filtering for smoothing large datasets
///
/// # Arguments
///
/// * `data` - Input data array
/// * `window_length` - Length of the filter window (must be odd)
/// * `polyorder` - Order of the polynomial used to fit the samples
/// * `deriv` - Order of the derivative to compute (0 = smoothing)
/// * `delta` - Spacing of samples (used for derivatives)
///
/// # Returns
///
/// * Filtered data
pub fn parallel_savgol_filter(
    data: &Array1<f64>,
    window_length: usize,
    polyorder: usize,
    deriv: usize,
    delta: f64,
) -> SignalResult<Array1<f64>> {
    use crate::savgol::savgol_coeffs;

    // Get filter coefficients
    let coeffs = savgol_coeffs(window_length, polyorder, deriv, delta, "interp")?;

    // Apply filter using parallel convolution
    let filtered = parallel_convolve(data.as_slice().unwrap(), &coeffs, "same", None)?;

    Ok(Array1::from(filtered))
}

/// Parallel batch filtering for multiple signals
///
/// Applies the same digital filter to multiple signals in parallel.
/// Useful for processing multiple channels simultaneously.
///
/// # Arguments
///
/// * `b` - Numerator coefficients
/// * `a` - Denominator coefficients
/// * `signals` - Array of input signals (each row is a signal)
/// * `chunk_size` - Chunk size for parallel processing
///
/// # Returns
///
/// * Array of filtered signals
#[allow(clippy::too_many_arguments)]
pub fn parallel_batch_filter(
    b: &[f64],
    a: &[f64],
    signals: &Array2<f64>,
    chunk_size: Option<usize>,
) -> SignalResult<Array2<f64>> {
    let (n_signals, signal_len) = signals.dim();
    let mut results = Array2::zeros((n_signals, signal_len));

    // Process each signal in parallel
    let signal_refs: Vec<_> = (0..n_signals).map(|i| signals.row(i)).collect();

    let processed: Vec<Vec<f64>> = par_iter_with_setup(
        signal_refs.iter().enumerate(),
        || {},
        |_, (i, signal)| {
            // Apply filter to each signal
            let filtered = parallel_filter_overlap_save(
                b,
                a,
                &Array1::from_iter(signal.iter().cloned()),
                chunk_size,
            )?;
            Ok(filtered.to_vec())
        },
        |results, processed_signal| {
            results.push(processed_signal?);
            Ok(())
        },
    )
    .map_err(|e| SignalError::ComputationError(format!("Batch filtering failed: {:?}", e)))?;

    // Copy results back
    for (i, signal_result) in processed.into_iter().enumerate() {
        for (j, &val) in signal_result.iter().enumerate() {
            if j < signal_len {
                results[[i, j]] = val;
            }
        }
    }

    Ok(results)
}

/// Parallel multi-rate filtering with decimation
///
/// Applies filtering followed by downsampling in parallel chunks.
/// Useful for efficiently reducing sample rate while filtering.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `b` - Filter numerator coefficients
/// * `a` - Filter denominator coefficients
/// * `decimation_factor` - Downsampling factor
/// * `chunk_size` - Chunk size for processing
///
/// # Returns
///
/// * Filtered and decimated signal
#[allow(dead_code)]
pub fn parallel_decimate_filter(
    signal: &[f64],
    b: &[f64],
    a: &[f64],
    decimation_factor: usize,
    chunk_size: Option<usize>,
) -> SignalResult<Vec<f64>> {
    if decimation_factor == 0 {
        return Err(SignalError::ValueError(
            "Decimation factor must be greater than 0".to_string(),
        ));
    }

    // First apply the filter
    let filtered = parallel_filtfilt(b, a, signal, chunk_size)?;

    // Then decimate
    let decimated: Vec<f64> = filtered
        .into_iter()
        .enumerate()
        .filter_map(|(i, val)| {
            if i % decimation_factor == 0 {
                Some(val)
            } else {
                None
            }
        })
        .collect();

    Ok(decimated)
}

/// Parallel FIR filter bank processing
///
/// Applies multiple FIR filters to the same input signal in parallel.
/// Useful for multi-band processing and feature extraction.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `filter_bank` - Collection of FIR filter coefficients
/// * `chunk_size` - Chunk size for parallel processing
///
/// # Returns
///
/// * Vector of filtered outputs, one for each filter
pub fn parallel_fir_filter_bank(
    signal: &[f64],
    filter_bank: &[Vec<f64>],
    chunk_size: Option<usize>,
) -> SignalResult<Vec<Vec<f64>>> {
    if filter_bank.is_empty() {
        return Err(SignalError::ValueError(
            "Filter bank cannot be empty".to_string(),
        ));
    }

    let signal_array = Array1::from(signal.to_vec());
    
    // Process each filter in parallel
    let results: Vec<Vec<f64>> = par_iter_with_setup(
        filter_bank.iter().enumerate(),
        || {},
        |_, (i, filter_coeffs)| {
            // Use parallel convolution for each filter
            let dummy_denominator = vec![1.0]; // FIR filter has denominator [1.0]
            parallel_filter_overlap_save(
                filter_coeffs,
                &dummy_denominator,
                &signal_array,
                chunk_size,
            )
        },
        |results, filter_result| {
            results.push(filter_result?);
            Ok(())
        },
    )?;

    Ok(results)
}

/// Parallel IIR filter bank processing
///
/// Applies multiple IIR filters to the same input signal in parallel.
/// Useful for multiband equalization and analysis.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `numerators` - Collection of numerator coefficient arrays
/// * `denominators` - Collection of denominator coefficient arrays
/// * `chunk_size` - Chunk size for parallel processing
///
/// # Returns
///
/// * Vector of filtered outputs, one for each filter
pub fn parallel_iir_filter_bank(
    signal: &[f64],
    numerators: &[Vec<f64>],
    denominators: &[Vec<f64>],
    chunk_size: Option<usize>,
) -> SignalResult<Vec<Vec<f64>>> {
    if numerators.len() != denominators.len() {
        return Err(SignalError::ValueError(
            "Number of numerators must match number of denominators".to_string(),
        ));
    }

    if numerators.is_empty() {
        return Err(SignalError::ValueError(
            "Filter bank cannot be empty".to_string(),
        ));
    }

    let signal_array = Array1::from(signal.to_vec());
    
    // Process each filter in parallel
    let results: Vec<Vec<f64>> = par_iter_with_setup(
        numerators.iter().zip(denominators.iter()).enumerate(),
        || {},
        |_, (i, (num_coeffs, den_coeffs))| {
            parallel_filter_overlap_save(
                num_coeffs,
                den_coeffs,
                &signal_array,
                chunk_size,
            )
        },
        |results, filter_result| {
            results.push(filter_result?);
            Ok(())
        },
    )?;

    Ok(results)
}

/// Parallel adaptive filter implementation
///
/// Implements LMS adaptive filtering with parallel processing for
/// the convolution operations.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `desired` - Desired response signal
/// * `filter_length` - Length of adaptive filter
/// * `step_size` - LMS step size (learning rate)
/// * `chunk_size` - Chunk size for parallel processing
///
/// # Returns
///
/// * Tuple of (filtered output, final filter coefficients, error signal)
pub fn parallel_adaptive_lms_filter(
    signal: &[f64],
    desired: &[f64],
    filter_length: usize,
    step_size: f64,
    chunk_size: Option<usize>,
) -> SignalResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    if signal.len() != desired.len() {
        return Err(SignalError::ValueError(
            "Signal and desired response must have same length".to_string(),
        ));
    }

    if filter_length == 0 {
        return Err(SignalError::ValueError(
            "Filter length must be greater than 0".to_string(),
        ));
    }

    let n = signal.len();
    let chunk = chunk_size.unwrap_or(1024.min(n / 4));
    
    let mut coeffs = vec![0.0; filter_length];
    let mut output = vec![0.0; n];
    let mut error = vec![0.0; n];
    let mut delay_line = vec![0.0; filter_length];

    // Process in chunks for parallel efficiency
    let n_chunks = (n + chunk - 1) / chunk;
    
    for chunk_idx in 0..n_chunks {
        let start = chunk_idx * chunk;
        let end = (start + chunk).min(n);
        let chunk_len = end - start;
        
        // Process each sample in the chunk
        for i in start..end {
            // Update delay line
            for j in (1..filter_length).rev() {
                delay_line[j] = delay_line[j - 1];
            }
            delay_line[0] = signal[i];
            
            // Filter output using parallel dot product
            let delay_array = Array1::from(delay_line.clone());
            let coeffs_array = Array1::from(coeffs.clone());
            
            // Use SIMD operations from scirs2-core
            use scirs2_core::simd_ops::SimdUnifiedOps;
            output[i] = f64::simd_dot(&delay_array.view(), &coeffs_array.view());
            
            // Error calculation
            error[i] = desired[i] - output[i];
            
            // Coefficient update using parallel operations
            for j in 0..filter_length {
                coeffs[j] += 2.0 * step_size * error[i] * delay_line[j];
            }
        }
    }

    Ok((output, coeffs, error))
}

/// Parallel wavelet filter bank
///
/// Applies a wavelet decomposition using parallel processing.
/// Implements a filter bank approach to wavelet transforms.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `wavelet_filters` - Lowpass and highpass filter coefficients
/// * `levels` - Number of decomposition levels
/// * `chunk_size` - Chunk size for parallel processing
///
/// # Returns
///
/// * Wavelet coefficients organized by level and type (approximation/detail)
pub fn parallel_wavelet_filter_bank(
    signal: &[f64],
    wavelet_filters: &(Vec<f64>, Vec<f64>), // (lowpass, highpass)
    levels: usize,
    chunk_size: Option<usize>,
) -> SignalResult<Vec<(Vec<f64>, Vec<f64>)>> {
    if levels == 0 {
        return Err(SignalError::ValueError(
            "Number of levels must be greater than 0".to_string(),
        ));
    }

    let (lowpass, highpass) = wavelet_filters;
    let mut results = Vec::with_capacity(levels);
    let mut current_signal = signal.to_vec();

    for level in 0..levels {
        // Apply both filters in parallel
        let filter_outputs: Vec<Vec<f64>> = par_iter_with_setup(
            [lowpass, highpass].iter().enumerate(),
            || {},
            |_, (filter_idx, filter_coeffs)| {
                let signal_array = Array1::from(current_signal.clone());
                let dummy_denominator = vec![1.0];
                parallel_filter_overlap_save(
                    filter_coeffs,
                    &dummy_denominator,
                    &signal_array,
                    chunk_size,
                )
            },
            |outputs, filter_result| {
                outputs.push(filter_result?);
                Ok(())
            },
        )?;

        let approximation = &filter_outputs[0];
        let detail = &filter_outputs[1];

        // Downsample both outputs by 2 (critical sampling)
        let approx_downsampled: Vec<f64> = approximation
            .iter()
            .enumerate()
            .filter_map(|(i, &val)| if i % 2 == 0 { Some(val) } else { None })
            .collect();

        let detail_downsampled: Vec<f64> = detail
            .iter()
            .enumerate()
            .filter_map(|(i, &val)| if i % 2 == 0 { Some(val) } else { None })
            .collect();

        results.push((approx_downsampled.clone(), detail_downsampled));

        // Continue with approximation coefficients for next level
        current_signal = approx_downsampled;
        
        // Stop if signal becomes too short
        if current_signal.len() < lowpass.len() * 2 {
            break;
        }
    }

    Ok(results)
}

/// Parallel polyphase filter implementation
///
/// Efficient implementation of polyphase filtering for multirate processing.
/// Useful for efficient decimation and interpolation.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `polyphase_filters` - Array of polyphase filter coefficients
/// * `decimation_factor` - Decimation factor
/// * `chunk_size` - Chunk size for parallel processing
///
/// # Returns
///
/// * Filtered and decimated output
pub fn parallel_polyphase_filter(
    signal: &[f64],
    polyphase_filters: &[Vec<f64>],
    decimation_factor: usize,
    chunk_size: Option<usize>,
) -> SignalResult<Vec<f64>> {
    if polyphase_filters.is_empty() || decimation_factor == 0 {
        return Err(SignalError::ValueError(
            "Invalid polyphase filter parameters".to_string(),
        ));
    }

    let n_phases = polyphase_filters.len();
    let filter_length = polyphase_filters[0].len();
    
    // Verify all polyphase filters have same length
    for filter in polyphase_filters {
        if filter.len() != filter_length {
            return Err(SignalError::ValueError(
                "All polyphase filters must have same length".to_string(),
            ));
        }
    }

    let output_length = signal.len() / decimation_factor;
    let mut output = vec![0.0; output_length];
    
    // Process decimated samples in parallel
    let output_indices: Vec<usize> = (0..output_length).collect();
    
    let parallel_results: Vec<f64> = par_iter_with_setup(
        output_indices.iter(),
        || {},
        |_, &out_idx| {
            let input_idx = out_idx * decimation_factor;
            if input_idx >= signal.len() {
                return Ok(0.0);
            }
            
            let mut sample_sum = 0.0;
            
            // Sum contributions from all polyphase filters
            for (phase, filter_coeffs) in polyphase_filters.iter().enumerate() {
                let mut filter_sum = 0.0;
                
                for (tap, &coeff) in filter_coeffs.iter().enumerate() {
                    let signal_idx = input_idx + phase + tap * n_phases;
                    if signal_idx < signal.len() {
                        filter_sum += coeff * signal[signal_idx];
                    }
                }
                
                sample_sum += filter_sum;
            }
            
            Ok(sample_sum)
        },
        |results, sample| {
            results.push(sample?);
            Ok(())
        },
    )?;

    Ok(parallel_results)
}

/// Parallel frequency-domain filtering using FFT convolution
///
/// Efficient for long filters using overlap-add method with parallel FFTs.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `impulse_response` - Filter impulse response
/// * `chunk_size` - FFT chunk size (should be power of 2)
///
/// # Returns
///
/// * Filtered signal
pub fn parallel_fft_filter(
    signal: &[f64],
    impulse_response: &[f64],
    chunk_size: Option<usize>,
) -> SignalResult<Vec<f64>> {
    use rustfft::{FftPlanner, num_complex::Complex};
    
    let fft_size = chunk_size.unwrap_or(4096);
    let ir_len = impulse_response.len();
    let useful_size = fft_size - ir_len + 1;
    
    if useful_size <= 0 {
        return Err(SignalError::ValueError(
            "FFT size too small for impulse response length".to_string(),
        ));
    }

    // Prepare FFT planner
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);
    let ifft = planner.plan_fft_inverse(fft_size);

    // Zero-pad and FFT the impulse response
    let mut ir_padded: Vec<Complex<f64>> = impulse_response
        .iter()
        .map(|&x| Complex::new(x, 0.0))
        .collect();
    ir_padded.resize(fft_size, Complex::new(0.0, 0.0));
    
    let mut ir_fft = ir_padded.clone();
    fft.process(&mut ir_fft);

    // Calculate number of chunks needed
    let n_chunks = (signal.len() + useful_size - 1) / useful_size;
    
    // Process chunks in parallel
    let chunk_results: Vec<Vec<f64>> = par_iter_with_setup(
        0..n_chunks,
        || {},
        |_, &chunk_idx| {
            let start = chunk_idx * useful_size;
            let end = (start + useful_size).min(signal.len());
            
            // Prepare input chunk
            let mut chunk_data: Vec<Complex<f64>> = (start..end)
                .map(|i| Complex::new(signal[i], 0.0))
                .collect();
            chunk_data.resize(fft_size, Complex::new(0.0, 0.0));
            
            // FFT of input chunk
            let mut chunk_fft = chunk_data;
            fft.process(&mut chunk_fft);
            
            // Frequency domain multiplication
            for i in 0..fft_size {
                chunk_fft[i] *= ir_fft[i];
            }
            
            // IFFT to get time domain result
            ifft.process(&mut chunk_fft);
            
            // Extract real part and normalize
            let chunk_result: Vec<f64> = chunk_fft
                .iter()
                .map(|c| c.re / fft_size as f64)
                .collect();
            
            Ok(chunk_result)
        },
        |results, chunk_result| {
            results.push(chunk_result?);
            Ok(())
        },
    )?;

    // Overlap-add to combine results
    let output_len = signal.len() + ir_len - 1;
    let mut output = vec![0.0; output_len];
    
    for (chunk_idx, chunk_result) in chunk_results.iter().enumerate() {
        let start = chunk_idx * useful_size;
        for (i, &val) in chunk_result.iter().enumerate() {
            if start + i < output_len {
                output[start + i] += val;
            }
        }
    }

    // Trim to input signal length for "same" mode
    output.truncate(signal.len());
    Ok(output)
}

/// Configuration for parallel filter operations
#[derive(Debug, Clone)]
pub struct ParallelFilterConfig {
    /// Chunk size for parallel processing
    pub chunk_size: Option<usize>,
    /// Number of threads to use (None for automatic)
    pub num_threads: Option<usize>,
    /// Enable SIMD optimizations
    pub use_simd: bool,
    /// Memory optimization mode
    pub memory_efficient: bool,
}

impl Default for ParallelFilterConfig {
    fn default() -> Self {
        Self {
            chunk_size: None,
            num_threads: None,
            use_simd: true,
            memory_efficient: false,
        }
    }
}

/// Advanced parallel filtering with configuration options
///
/// Provides a unified interface for various parallel filtering operations
/// with configuration options for optimization.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `filter_type` - Type of filter to apply
/// * `config` - Configuration for parallel processing
///
/// # Returns
///
/// * Filtered signal
pub fn parallel_filter_advanced(
    signal: &[f64],
    filter_type: &ParallelFilterType,
    config: &ParallelFilterConfig,
) -> SignalResult<Vec<f64>> {
    match filter_type {
        ParallelFilterType::FIR { coeffs } => {
            let dummy_denom = vec![1.0];
            let signal_array = Array1::from(signal.to_vec());
            parallel_filter_overlap_save(coeffs, &dummy_denom, &signal_array, config.chunk_size)
        }
        
        ParallelFilterType::IIR { numerator, denominator } => {
            let signal_array = Array1::from(signal.to_vec());
            parallel_filter_overlap_save(numerator, denominator, &signal_array, config.chunk_size)
        }
        
        ParallelFilterType::Adaptive { desired, filter_length, step_size } => {
            let (output, _, _) = parallel_adaptive_lms_filter(
                signal,
                desired,
                *filter_length,
                *step_size,
                config.chunk_size,
            )?;
            Ok(output)
        }
        
        ParallelFilterType::FFT { impulse_response } => {
            parallel_fft_filter(signal, impulse_response, config.chunk_size)
        }
    }
}

/// Types of parallel filters available
#[derive(Debug, Clone)]
pub enum ParallelFilterType {
    /// FIR filter with coefficients
    FIR {
        coeffs: Vec<f64>,
    },
    /// IIR filter with numerator and denominator
    IIR {
        numerator: Vec<f64>,
        denominator: Vec<f64>,
    },
    /// Adaptive filter
    Adaptive {
        desired: Vec<f64>,
        filter_length: usize,
        step_size: f64,
    },
    /// FFT-based convolution filter
    FFT {
        impulse_response: Vec<f64>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_parallel_fir_filter_bank() {
        let signal: Vec<f64> = (0..1000).map(|i| (2.0 * PI * i as f64 / 100.0).sin()).collect();
        
        // Create simple filter bank
        let filter_bank = vec![
            vec![0.5, 0.5], // Simple averaging filter
            vec![1.0, -1.0], // Simple differencing filter
        ];
        
        let results = parallel_fir_filter_bank(&signal, &filter_bank, None).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), signal.len());
        assert_eq!(results[1].len(), signal.len());
    }

    #[test]
    fn test_parallel_adaptive_lms() {
        let n = 100;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / 10.0).sin()).collect();
        let desired: Vec<f64> = signal.iter().map(|&x| x * 0.5).collect(); // Attenuated version
        
        let (output, coeffs, _error) = parallel_adaptive_lms_filter(
            &signal, &desired, 10, 0.01, None
        ).unwrap();
        
        assert_eq!(output.len(), n);
        assert_eq!(coeffs.len(), 10);
    }

    #[test]
    fn test_parallel_wavelet_filter_bank() {
        let signal: Vec<f64> = (0..512).map(|i| (2.0 * PI * i as f64 / 32.0).sin()).collect();
        
        // Simple Haar wavelet filters
        let lowpass = vec![0.7071, 0.7071];
        let highpass = vec![0.7071, -0.7071];
        let wavelet_filters = (lowpass, highpass);
        
        let results = parallel_wavelet_filter_bank(&signal, &wavelet_filters, 3, None).unwrap();
        assert_eq!(results.len(), 3); // 3 levels of decomposition
        
        // Check that each level has approximation and detail coefficients
        for (approx, detail) in &results {
            assert!(!approx.is_empty());
            assert!(!detail.is_empty());
        }
    }

    #[test]
    fn test_parallel_filter_advanced() {
        let signal: Vec<f64> = (0..100).map(|i| (2.0 * PI * i as f64 / 10.0).sin()).collect();
        
        let filter_type = ParallelFilterType::FIR {
            coeffs: vec![0.25, 0.5, 0.25], // Simple smoothing filter
        };
        
        let config = ParallelFilterConfig::default();
        let result = parallel_filter_advanced(&signal, &filter_type, &config).unwrap();
        
        assert_eq!(result.len(), signal.len());
    }
}
