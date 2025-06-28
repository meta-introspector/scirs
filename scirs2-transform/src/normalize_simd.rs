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

/// SIMD-accelerated normalization for 2D arrays along a specified axis
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
            // Process along the specified axis
            if axis == 0 {
                // Normalize each column
                for j in 0..shape[1] {
                    let col = array.column(j);
                    let col_array = col.to_owned();
                    let norm_col = simd_minmax_normalize_1d(&col_array)?;
                    for i in 0..shape[0] {
                        normalized[[i, j]] = norm_col[i];
                    }
                }
            } else {
                // Normalize each row
                for i in 0..shape[0] {
                    let row = array.row(i);
                    let row_array = row.to_owned();
                    let norm_row = simd_minmax_normalize_1d(&row_array)?;
                    for j in 0..shape[1] {
                        normalized[[i, j]] = norm_row[j];
                    }
                }
            }
        }
        NormalizationMethod::ZScore => {
            // Process along the specified axis
            if axis == 0 {
                // Normalize each column
                for j in 0..shape[1] {
                    let col = array.column(j);
                    let col_array = col.to_owned();
                    let norm_col = simd_zscore_normalize_1d(&col_array)?;
                    for i in 0..shape[0] {
                        normalized[[i, j]] = norm_col[i];
                    }
                }
            } else {
                // Normalize each row
                for i in 0..shape[0] {
                    let row = array.row(i);
                    let row_array = row.to_owned();
                    let norm_row = simd_zscore_normalize_1d(&row_array)?;
                    for j in 0..shape[1] {
                        normalized[[i, j]] = norm_row[j];
                    }
                }
            }
        }
        NormalizationMethod::L2 => {
            // Process along the specified axis
            if axis == 0 {
                // Normalize each column
                for j in 0..shape[1] {
                    let col = array.column(j);
                    let col_array = col.to_owned();
                    let norm_col = simd_l2_normalize_1d(&col_array)?;
                    for i in 0..shape[0] {
                        normalized[[i, j]] = norm_col[i];
                    }
                }
            } else {
                // Normalize each row
                for i in 0..shape[0] {
                    let row = array.row(i);
                    let row_array = row.to_owned();
                    let norm_row = simd_l2_normalize_1d(&row_array)?;
                    for j in 0..shape[1] {
                        normalized[[i, j]] = norm_row[j];
                    }
                }
            }
        }
        NormalizationMethod::MaxAbs => {
            // Process along the specified axis
            if axis == 0 {
                // Normalize each column
                for j in 0..shape[1] {
                    let col = array.column(j);
                    let col_array = col.to_owned();
                    let norm_col = simd_maxabs_normalize_1d(&col_array)?;
                    for i in 0..shape[0] {
                        normalized[[i, j]] = norm_col[i];
                    }
                }
            } else {
                // Normalize each row
                for i in 0..shape[0] {
                    let row = array.row(i);
                    let row_array = row.to_owned();
                    let norm_row = simd_maxabs_normalize_1d(&row_array)?;
                    for j in 0..shape[1] {
                        normalized[[i, j]] = norm_row[j];
                    }
                }
            }
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
