//! SIMD-accelerated normalization operations
//!
//! This module provides SIMD-optimized implementations of normalization operations
//! using the unified SIMD operations from scirs2-core.

use ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix1, Ix2};
use num_traits::{Float, NumCast};
use scirs2_core::simd_ops::SimdUnifiedOps;

use crate::error::{Result, TransformError};
use crate::normalize::{NormalizationMethod, EPSILON};

/// SIMD-accelerated min-max normalization for 1D arrays
pub fn simd_minmax_normalize_1d<F>(array: &Array1<F>) -> Result<Array1<F>>
where
    F: Float + NumCast + SimdUnifiedOps,
{
    if array.is_empty() {
        return Err(TransformError::InvalidInput(
            "Input array is empty".to_string(),
        ));
    }

    let min = F::simd_min_element(&array.view());
    let max = F::simd_max_element(&array.view());
    let range = max - min;

    if range.abs() <= F::from(EPSILON).unwrap() {
        // Constant feature, return array of 0.5
        return Ok(Array1::from_elem(array.len(), F::from(0.5).unwrap()));
    }

    // Normalize: (x - min) / range
    let min_array = Array1::from_elem(array.len(), min);
    let normalized = F::simd_sub(&array.view(), &min_array.view());
    let range_array = Array1::from_elem(array.len(), range);
    let result = F::simd_div(&normalized.view(), &range_array.view());

    Ok(result)
}

/// SIMD-accelerated Z-score normalization for 1D arrays
pub fn simd_zscore_normalize_1d<F>(array: &Array1<F>) -> Result<Array1<F>>
where
    F: Float + NumCast + SimdUnifiedOps,
{
    if array.is_empty() {
        return Err(TransformError::InvalidInput(
            "Input array is empty".to_string(),
        ));
    }

    let mean = F::simd_mean(&array.view());
    let n = F::from(array.len()).unwrap();

    // Compute variance
    let mean_array = Array1::from_elem(array.len(), mean);
    let centered = F::simd_sub(&array.view(), &mean_array.view());
    let squared = F::simd_mul(&centered.view(), &centered.view());
    let variance = F::simd_sum(&squared.view()) / n;
    let std_dev = variance.sqrt();

    if std_dev <= F::from(EPSILON).unwrap() {
        // Constant feature, return zeros
        return Ok(Array1::zeros(array.len()));
    }

    // Normalize: (x - mean) / std_dev
    let std_array = Array1::from_elem(array.len(), std_dev);
    let result = F::simd_div(&centered.view(), &std_array.view());

    Ok(result)
}

/// SIMD-accelerated L2 normalization for 1D arrays
pub fn simd_l2_normalize_1d<F>(array: &Array1<F>) -> Result<Array1<F>>
where
    F: Float + NumCast + SimdUnifiedOps,
{
    if array.is_empty() {
        return Err(TransformError::InvalidInput(
            "Input array is empty".to_string(),
        ));
    }

    let l2_norm = F::simd_norm(&array.view());

    if l2_norm <= F::from(EPSILON).unwrap() {
        // Zero vector, return zeros
        return Ok(Array1::zeros(array.len()));
    }

    // Normalize: x / l2_norm
    let norm_array = Array1::from_elem(array.len(), l2_norm);
    let result = F::simd_div(&array.view(), &norm_array.view());

    Ok(result)
}

/// SIMD-accelerated max absolute scaling for 1D arrays
pub fn simd_maxabs_normalize_1d<F>(array: &Array1<F>) -> Result<Array1<F>>
where
    F: Float + NumCast + SimdUnifiedOps,
{
    if array.is_empty() {
        return Err(TransformError::InvalidInput(
            "Input array is empty".to_string(),
        ));
    }

    let abs_array = F::simd_abs(&array.view());
    let max_abs = F::simd_max_element(&abs_array.view());

    if max_abs <= F::from(EPSILON).unwrap() {
        // All zeros, return zeros
        return Ok(Array1::zeros(array.len()));
    }

    // Normalize: x / max_abs
    let max_abs_array = Array1::from_elem(array.len(), max_abs);
    let result = F::simd_div(&array.view(), &max_abs_array.view());

    Ok(result)
}

/// Advanced SIMD-accelerated normalization for 2D arrays with optimized memory access patterns
pub fn simd_normalize_array<S, F>(
    array: &ArrayBase<S, Ix2>,
    method: NormalizationMethod,
    axis: usize,
) -> Result<Array2<F>>
where
    S: Data<Elem = F>,
    F: Float + NumCast + SimdUnifiedOps,
{
    if !array.is_standard_layout() {
        return Err(TransformError::InvalidInput(
            "Input array must be in standard memory layout".to_string(),
        ));
    }

    if array.ndim() != 2 {
        return Err(TransformError::InvalidInput(
            "Only 2D arrays are supported".to_string(),
        ));
    }

    if axis >= array.ndim() {
        return Err(TransformError::InvalidInput(format!(
            "Invalid axis {} for array with {} dimensions",
            axis,
            array.ndim()
        )));
    }

    let shape = array.shape();
    let mut normalized = Array2::zeros((shape[0], shape[1]));

    match method {
        NormalizationMethod::MinMax => {
            simd_normalize_block_minmax(array, &mut normalized, axis)?
        }
        NormalizationMethod::ZScore => {
            simd_normalize_block_zscore(array, &mut normalized, axis)?
        }
        NormalizationMethod::L2 => {
            simd_normalize_block_l2(array, &mut normalized, axis)?
        }
        NormalizationMethod::MaxAbs => {
            simd_normalize_block_maxabs(array, &mut normalized, axis)?
        }
        _ => {
            // Fall back to non-SIMD implementation for other methods
            return Err(TransformError::InvalidInput(
                "SIMD implementation not available for this normalization method".to_string(),
            ));
        }
    }

    Ok(normalized)
}

/// Block-wise SIMD min-max normalization with optimized memory access
fn simd_normalize_block_minmax<S, F>(
    array: &ArrayBase<S, Ix2>,
    normalized: &mut Array2<F>,
    axis: usize,
) -> Result<()>
where
    S: Data<Elem = F>,
    F: Float + NumCast + SimdUnifiedOps,
{
    const BLOCK_SIZE: usize = 64;
    let shape = array.shape();

    if axis == 0 {
        // Column-wise normalization with block processing
        let mut global_mins = Array1::zeros(shape[1]);
        let mut global_maxs = Array1::zeros(shape[1]);

        // First pass: compute global min/max for each column in blocks
        for block_start in (0..shape[1]).step_by(BLOCK_SIZE) {
            let block_end = (block_start + BLOCK_SIZE).min(shape[1]);
            
            for j in block_start..block_end {
                let col = array.column(j);
                let col_array = col.to_owned();
                global_mins[j] = F::simd_min_element(&col_array.view());
                global_maxs[j] = F::simd_max_element(&col_array.view());
            }
        }

        // Second pass: normalize using pre-computed min/max
        for block_start in (0..shape[1]).step_by(BLOCK_SIZE) {
            let block_end = (block_start + BLOCK_SIZE).min(shape[1]);
            
            for j in block_start..block_end {
                let col = array.column(j);
                let col_array = col.to_owned();
                let range = global_maxs[j] - global_mins[j];
                
                if range.abs() <= F::from(EPSILON).unwrap() {
                    // Constant feature
                    for i in 0..shape[0] {
                        normalized[[i, j]] = F::from(0.5).unwrap();
                    }
                } else {
                    // Vectorized normalization
                    let min_array = Array1::from_elem(shape[0], global_mins[j]);
                    let range_array = Array1::from_elem(shape[0], range);
                    let centered = F::simd_sub(&col_array.view(), &min_array.view());
                    let norm_col = F::simd_div(&centered.view(), &range_array.view());
                    
                    for i in 0..shape[0] {
                        normalized[[i, j]] = norm_col[i];
                    }
                }
            }
        }
    } else {
        // Row-wise normalization with block processing
        for block_start in (0..shape[0]).step_by(BLOCK_SIZE) {
            let block_end = (block_start + BLOCK_SIZE).min(shape[0]);
            
            for i in block_start..block_end {
                let row = array.row(i);
                let row_array = row.to_owned();
                let norm_row = simd_minmax_normalize_1d(&row_array)?;
                
                for j in 0..shape[1] {
                    normalized[[i, j]] = norm_row[j];
                }
            }
        }
    }
    Ok(())
}

/// Block-wise SIMD Z-score normalization with optimized memory access
fn simd_normalize_block_zscore<S, F>(
    array: &ArrayBase<S, Ix2>,
    normalized: &mut Array2<F>,
    axis: usize,
) -> Result<()>
where
    S: Data<Elem = F>,
    F: Float + NumCast + SimdUnifiedOps,
{
    const BLOCK_SIZE: usize = 64;
    let shape = array.shape();

    if axis == 0 {
        // Column-wise normalization
        let mut global_means = Array1::zeros(shape[1]);
        let mut global_stds = Array1::zeros(shape[1]);
        let n_samples_f = F::from(shape[0]).unwrap();

        // First pass: compute means and standard deviations
        for block_start in (0..shape[1]).step_by(BLOCK_SIZE) {
            let block_end = (block_start + BLOCK_SIZE).min(shape[1]);
            
            for j in block_start..block_end {
                let col = array.column(j);
                let col_array = col.to_owned();
                
                // Compute mean
                global_means[j] = F::simd_sum(&col_array.view()) / n_samples_f;
                
                // Compute standard deviation
                let mean_array = Array1::from_elem(shape[0], global_means[j]);
                let centered = F::simd_sub(&col_array.view(), &mean_array.view());
                let squared = F::simd_mul(&centered.view(), &centered.view());
                let variance = F::simd_sum(&squared.view()) / n_samples_f;
                global_stds[j] = variance.sqrt();
                
                // Avoid division by zero
                if global_stds[j] <= F::from(EPSILON).unwrap() {
                    global_stds[j] = F::one();
                }
            }
        }

        // Second pass: normalize using pre-computed statistics
        for block_start in (0..shape[1]).step_by(BLOCK_SIZE) {
            let block_end = (block_start + BLOCK_SIZE).min(shape[1]);
            
            for j in block_start..block_end {
                let col = array.column(j);
                let col_array = col.to_owned();
                
                if global_stds[j] <= F::from(EPSILON).unwrap() {
                    // Constant feature
                    for i in 0..shape[0] {
                        normalized[[i, j]] = F::zero();
                    }
                } else {
                    // Vectorized normalization
                    let mean_array = Array1::from_elem(shape[0], global_means[j]);
                    let std_array = Array1::from_elem(shape[0], global_stds[j]);
                    let centered = F::simd_sub(&col_array.view(), &mean_array.view());
                    let norm_col = F::simd_div(&centered.view(), &std_array.view());
                    
                    for i in 0..shape[0] {
                        normalized[[i, j]] = norm_col[i];
                    }
                }
            }
        }
    } else {
        // Row-wise normalization with block processing
        for block_start in (0..shape[0]).step_by(BLOCK_SIZE) {
            let block_end = (block_start + BLOCK_SIZE).min(shape[0]);
            
            for i in block_start..block_end {
                let row = array.row(i);
                let row_array = row.to_owned();
                let norm_row = simd_zscore_normalize_1d(&row_array)?;
                
                for j in 0..shape[1] {
                    normalized[[i, j]] = norm_row[j];
                }
            }
        }
    }
    Ok(())
}

/// Block-wise SIMD L2 normalization with optimized memory access
fn simd_normalize_block_l2<S, F>(
    array: &ArrayBase<S, Ix2>,
    normalized: &mut Array2<F>,
    axis: usize,
) -> Result<()>
where
    S: Data<Elem = F>,
    F: Float + NumCast + SimdUnifiedOps,
{
    const BLOCK_SIZE: usize = 64;
    let shape = array.shape();

    if axis == 0 {
        // Column-wise L2 normalization
        for block_start in (0..shape[1]).step_by(BLOCK_SIZE) {
            let block_end = (block_start + BLOCK_SIZE).min(shape[1]);
            
            for j in block_start..block_end {
                let col = array.column(j);
                let col_array = col.to_owned();
                let norm_col = simd_l2_normalize_1d(&col_array)?;
                
                for i in 0..shape[0] {
                    normalized[[i, j]] = norm_col[i];
                }
            }
        }
    } else {
        // Row-wise L2 normalization with SIMD optimization
        for block_start in (0..shape[0]).step_by(BLOCK_SIZE) {
            let block_end = (block_start + BLOCK_SIZE).min(shape[0]);
            
            for i in block_start..block_end {
                let row = array.row(i);
                let row_array = row.to_owned();
                let l2_norm = F::simd_norm(&row_array.view());
                
                if l2_norm <= F::from(EPSILON).unwrap() {
                    // Zero vector
                    for j in 0..shape[1] {
                        normalized[[i, j]] = F::zero();
                    }
                } else {
                    // Vectorized division
                    let norm_array = Array1::from_elem(shape[1], l2_norm);
                    let norm_row = F::simd_div(&row_array.view(), &norm_array.view());
                    
                    for j in 0..shape[1] {
                        normalized[[i, j]] = norm_row[j];
                    }
                }
            }
        }
    }
    Ok(())
}

/// Block-wise SIMD max-absolute normalization with optimized memory access
fn simd_normalize_block_maxabs<S, F>(
    array: &ArrayBase<S, Ix2>,
    normalized: &mut Array2<F>,
    axis: usize,
) -> Result<()>
where
    S: Data<Elem = F>,
    F: Float + NumCast + SimdUnifiedOps,
{
    const BLOCK_SIZE: usize = 64;
    let shape = array.shape();

    if axis == 0 {
        // Column-wise max-abs normalization
        for block_start in (0..shape[1]).step_by(BLOCK_SIZE) {
            let block_end = (block_start + BLOCK_SIZE).min(shape[1]);
            
            for j in block_start..block_end {
                let col = array.column(j);
                let col_array = col.to_owned();
                let norm_col = simd_maxabs_normalize_1d(&col_array)?;
                
                for i in 0..shape[0] {
                    normalized[[i, j]] = norm_col[i];
                }
            }
        }
    } else {
        // Row-wise max-abs normalization with SIMD optimization
        for block_start in (0..shape[0]).step_by(BLOCK_SIZE) {
            let block_end = (block_start + BLOCK_SIZE).min(shape[0]);
            
            for i in block_start..block_end {
                let row = array.row(i);
                let row_array = row.to_owned();
                let abs_array = F::simd_abs(&row_array.view());
                let max_abs = F::simd_max_element(&abs_array.view());
                
                if max_abs <= F::from(EPSILON).unwrap() {
                    // All zeros
                    for j in 0..shape[1] {
                        normalized[[i, j]] = F::zero();
                    }
                } else {
                    // Vectorized division
                    let max_abs_array = Array1::from_elem(shape[1], max_abs);
                    let norm_row = F::simd_div(&row_array.view(), &max_abs_array.view());
                    
                    for j in 0..shape[1] {
                        normalized[[i, j]] = norm_row[j];
                    }
                }
            }
        }
    }
    Ok(())
}
