//! SIMD-optimized specialized filter functions
//!
//! This module provides highly optimized SIMD implementations for
//! specialized filtering operations that benefit from vectorization.

use ndarray::{s, Array, ArrayView, ArrayView1, ArrayView2, ArrayViewMut1, Axis, Dimension, Ix2};
use num_traits::{Float, FromPrimitive};
use scirs2_core::simd_ops::SimdUnifiedOps;
use std::cmp;
use std::fmt::Debug;

use crate::error::NdimageResult;
use crate::filters::BoundaryMode;

/// SIMD-optimized bilateral filter for edge-preserving smoothing
///
/// The bilateral filter is a non-linear filter that smooths an image while preserving edges.
/// It considers both spatial distance and intensity difference when computing weights.
pub fn simd_bilateral_filter<T>(
    input: ArrayView2<T>,
    spatial_sigma: T,
    range_sigma: T,
    window_size: Option<usize>,
) -> NdimageResult<Array<T, Ix2>>
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + SimdUnifiedOps,
{
    let (height, width) = input.dim();
    let window_size = window_size.unwrap_or_else(|| {
        // Automatically determine window size based on spatial sigma
        let radius = (spatial_sigma * T::from_f64(3.0).unwrap()).to_usize().unwrap();
        2 * radius + 1
    });
    let half_window = window_size / 2;
    
    let mut output = Array::zeros((height, width));
    
    // Pre-compute spatial weights
    let spatial_weights = compute_spatial_weights(window_size, spatial_sigma);
    
    // Process image in parallel chunks
    let chunk_size = if height * width > 10000 { 64 } else { height };
    
    output
        .axis_chunks_iter_mut(Axis(0), chunk_size)
        .into_par_iter()
        .enumerate()
        .for_each(|(chunk_idx, mut chunk)| {
            let y_start = chunk_idx * chunk_size;
            
            for (local_y, mut row) in chunk.axis_iter_mut(Axis(0)).enumerate() {
                let y = y_start + local_y;
                simd_bilateral_filter_row(
                    &input,
                    &mut row,
                    y,
                    half_window,
                    &spatial_weights,
                    range_sigma,
                );
            }
        });
    
    Ok(output)
}

/// Process a single row with SIMD optimization
fn simd_bilateral_filter_row<T>(
    input: &ArrayView2<T>,
    output_row: &mut ArrayViewMut1<T>,
    y: usize,
    half_window: usize,
    spatial_weights: &Array<T, Ix2>,
    range_sigma: T,
) where
    T: Float + FromPrimitive + Debug + Clone + SimdUnifiedOps,
{
    let (height, width) = input.dim();
    let range_factor = T::from_f64(-0.5).unwrap() / (range_sigma * range_sigma);
    
    // Process pixels in SIMD chunks
    let simd_width = T::simd_width();
    let num_full_chunks = width / simd_width;
    
    // Process full SIMD chunks
    for chunk_idx in 0..num_full_chunks {
        let x_start = chunk_idx * simd_width;
        
        // Gather center pixel values for SIMD chunk
        let mut center_values = vec![T::zero(); simd_width];
        for i in 0..simd_width {
            center_values[i] = input[(y, x_start + i)];
        }
        
        let mut sum_weights = vec![T::zero(); simd_width];
        let mut sum_values = vec![T::zero(); simd_width];
        
        // Compute bilateral filter for window
        for dy in 0..2 * half_window + 1 {
            let ny = (y as isize + dy as isize - half_window as isize)
                .clamp(0, height as isize - 1) as usize;
            
            for dx in 0..2 * half_window + 1 {
                // Process SIMD width pixels at once
                let mut neighbor_values = vec![T::zero(); simd_width];
                let mut valid_mask = vec![true; simd_width];
                
                for i in 0..simd_width {
                    let x = x_start + i;
                    let nx = (x as isize + dx as isize - half_window as isize)
                        .clamp(0, width as isize - 1) as usize;
                    neighbor_values[i] = input[(ny, nx)];
                    valid_mask[i] = nx < width;
                }
                
                // Compute range weights using SIMD
                let mut range_diffs = vec![T::zero(); simd_width];
                for i in 0..simd_width {
                    range_diffs[i] = neighbor_values[i] - center_values[i];
                }
                
                // Square differences
                let range_diffs_sq = T::simd_mul(&range_diffs[..], &range_diffs[..]);
                
                // Apply range factor
                let mut range_exp_args = vec![T::zero(); simd_width];
                for i in 0..simd_width {
                    range_exp_args[i] = range_diffs_sq[i] * range_factor;
                }
                
                // Compute exponential (approximation for SIMD)
                let range_weights = simd_exp_approx(&range_exp_args);
                
                // Combine with spatial weight
                let spatial_weight = spatial_weights[(dy, dx)];
                
                for i in 0..simd_width {
                    if valid_mask[i] {
                        let weight = spatial_weight * range_weights[i];
                        sum_weights[i] = sum_weights[i] + weight;
                        sum_values[i] = sum_values[i] + weight * neighbor_values[i];
                    }
                }
            }
        }
        
        // Write results
        for i in 0..simd_width {
            if x_start + i < width {
                output_row[x_start + i] = sum_values[i] / sum_weights[i];
            }
        }
    }
    
    // Process remaining pixels
    for x in (num_full_chunks * simd_width)..width {
        let center_value = input[(y, x)];
        let mut sum_weight = T::zero();
        let mut sum_value = T::zero();
        
        for dy in 0..2 * half_window + 1 {
            let ny = (y as isize + dy as isize - half_window as isize)
                .clamp(0, height as isize - 1) as usize;
            
            for dx in 0..2 * half_window + 1 {
                let nx = (x as isize + dx as isize - half_window as isize)
                    .clamp(0, width as isize - 1) as usize;
                
                let neighbor_value = input[(ny, nx)];
                let range_diff = neighbor_value - center_value;
                let range_weight = (range_diff * range_diff * range_factor).exp();
                let spatial_weight = spatial_weights[(dy, dx)];
                
                let weight = spatial_weight * range_weight;
                sum_weight = sum_weight + weight;
                sum_value = sum_value + weight * neighbor_value;
            }
        }
        
        output_row[x] = sum_value / sum_weight;
    }
}

/// Compute spatial weights for bilateral filter
fn compute_spatial_weights<T>(window_size: usize, sigma: T) -> Array<T, Ix2>
where
    T: Float + FromPrimitive,
{
    let half_window = window_size / 2;
    let factor = T::from_f64(-0.5).unwrap() / (sigma * sigma);
    let mut weights = Array::zeros((window_size, window_size));
    
    for dy in 0..window_size {
        for dx in 0..window_size {
            let y_dist = T::from_isize(dy as isize - half_window as isize).unwrap();
            let x_dist = T::from_isize(dx as isize - half_window as isize).unwrap();
            let dist_sq = y_dist * y_dist + x_dist * x_dist;
            weights[(dy, dx)] = (dist_sq * factor).exp();
        }
    }
    
    weights
}

/// SIMD approximation of exponential function
fn simd_exp_approx<T>(values: &[T]) -> Vec<T>
where
    T: Float + FromPrimitive,
{
    // Use Taylor series approximation for exp(x) ≈ 1 + x + x²/2 + x³/6
    // This is accurate for small values (which we have after multiplying by range_factor)
    let mut result = vec![T::one(); values.len()];
    
    for i in 0..values.len() {
        let x = values[i];
        let x2 = x * x;
        let x3 = x2 * x;
        result[i] = T::one() + x + x2 / T::from_f64(2.0).unwrap() + x3 / T::from_f64(6.0).unwrap();
        
        // Clamp to positive values
        if result[i] < T::zero() {
            result[i] = T::zero();
        }
    }
    
    result
}

/// SIMD-optimized non-local means filter
///
/// Non-local means is a denoising algorithm that averages similar patches
/// throughout the image, not just in a local neighborhood.
pub fn simd_non_local_means<T>(
    input: ArrayView2<T>,
    patch_size: usize,
    search_window: usize,
    h: T, // Filter strength
) -> NdimageResult<Array<T, Ix2>>
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + SimdUnifiedOps,
{
    let (height, width) = input.dim();
    let half_patch = patch_size / 2;
    let half_search = search_window / 2;
    
    let mut output = Array::zeros((height, width));
    let h_squared = h * h;
    
    // Pre-compute patch normalization factor
    let patch_norm = T::from_usize(patch_size * patch_size).unwrap();
    
    // Process in parallel chunks
    output
        .axis_chunks_iter_mut(Axis(0), 32)
        .into_par_iter()
        .enumerate()
        .for_each(|(chunk_idx, mut chunk)| {
            let y_start = chunk_idx * 32;
            
            for (local_y, mut row) in chunk.axis_iter_mut(Axis(0)).enumerate() {
                let y = y_start + local_y;
                if y >= half_patch && y < height - half_patch {
                    simd_nlm_process_row(
                        &input,
                        &mut row,
                        y,
                        half_patch,
                        half_search,
                        h_squared,
                        patch_norm,
                    );
                }
            }
        });
    
    Ok(output)
}

/// Process a row with non-local means using SIMD
fn simd_nlm_process_row<T>(
    input: &ArrayView2<T>,
    output_row: &mut ArrayViewMut1<T>,
    y: usize,
    half_patch: usize,
    half_search: usize,
    h_squared: T,
    patch_norm: T,
) where
    T: Float + FromPrimitive + Debug + Clone + SimdUnifiedOps,
{
    let (height, width) = input.dim();
    let simd_width = T::simd_width();
    
    for x in half_patch..width - half_patch {
        let mut weight_sum = T::zero();
        let mut value_sum = T::zero();
        
        // Define search region
        let search_y_min = (y as isize - half_search as isize).max(half_patch as isize) as usize;
        let search_y_max = (y + half_search + 1).min(height - half_patch);
        let search_x_min = (x as isize - half_search as isize).max(half_patch as isize) as usize;
        let search_x_max = (x + half_search + 1).min(width - half_patch);
        
        // Extract reference patch
        let ref_patch = input.slice(s![
            y - half_patch..=y + half_patch,
            x - half_patch..=x + half_patch
        ]);
        
        // Search for similar patches
        for sy in search_y_min..search_y_max {
            for sx in search_x_min..search_x_max {
                // Extract comparison patch
                let comp_patch = input.slice(s![
                    sy - half_patch..=sy + half_patch,
                    sx - half_patch..=sx + half_patch
                ]);
                
                // Compute patch distance using SIMD
                let distance = simd_patch_distance(&ref_patch, &comp_patch) / patch_norm;
                
                // Compute weight
                let weight = (-distance / h_squared).exp();
                weight_sum = weight_sum + weight;
                value_sum = value_sum + weight * input[(sy, sx)];
            }
        }
        
        output_row[x] = value_sum / weight_sum;
    }
}

/// Compute L2 distance between patches using SIMD
fn simd_patch_distance<T>(patch1: &ArrayView2<T>, patch2: &ArrayView2<T>) -> T
where
    T: Float + FromPrimitive + SimdUnifiedOps,
{
    let flat1 = patch1.as_slice().unwrap();
    let flat2 = patch2.as_slice().unwrap();
    
    let mut sum = T::zero();
    let simd_width = T::simd_width();
    let num_chunks = flat1.len() / simd_width;
    
    // Process SIMD chunks
    for i in 0..num_chunks {
        let start = i * simd_width;
        let end = start + simd_width;
        
        let diff = T::simd_sub(&flat1[start..end], &flat2[start..end]);
        let diff_sq = T::simd_mul(&diff, &diff);
        
        for &val in &diff_sq {
            sum = sum + val;
        }
    }
    
    // Process remaining elements
    for i in (num_chunks * simd_width)..flat1.len() {
        let diff = flat1[i] - flat2[i];
        sum = sum + diff * diff;
    }
    
    sum
}

/// SIMD-optimized anisotropic diffusion filter
///
/// This filter performs edge-preserving smoothing by varying the diffusion
/// coefficient based on the local image gradient.
pub fn simd_anisotropic_diffusion<T>(
    input: ArrayView2<T>,
    iterations: usize,
    kappa: T,     // Edge threshold parameter
    lambda: T,    // Diffusion rate (0 < lambda <= 0.25)
    option: usize, // 1: exponential, 2: quadratic
) -> NdimageResult<Array<T, Ix2>>
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + SimdUnifiedOps,
{
    let (height, width) = input.dim();
    let mut current = input.to_owned();
    let mut next = Array::zeros((height, width));
    
    let kappa_sq = kappa * kappa;
    
    for _ in 0..iterations {
        // Compute gradients and diffusion coefficients in parallel
        next.axis_chunks_iter_mut(Axis(0), 32)
            .into_par_iter()
            .enumerate()
            .for_each(|(chunk_idx, mut chunk)| {
                let y_start = chunk_idx * 32;
                
                for (local_y, mut row) in chunk.axis_iter_mut(Axis(0)).enumerate() {
                    let y = y_start + local_y;
                    simd_diffusion_row(
                        &current.view(),
                        &mut row,
                        y,
                        kappa_sq,
                        lambda,
                        option,
                    );
                }
            });
        
        // Swap buffers
        std::mem::swap(&mut current, &mut next);
    }
    
    Ok(current)
}

/// Process a row with anisotropic diffusion using SIMD
fn simd_diffusion_row<T>(
    input: &ArrayView2<T>,
    output_row: &mut ArrayViewMut1<T>,
    y: usize,
    kappa_sq: T,
    lambda: T,
    option: usize,
) where
    T: Float + FromPrimitive + Debug + Clone + SimdUnifiedOps,
{
    let (height, width) = input.dim();
    let simd_width = T::simd_width();
    
    // Process SIMD chunks
    let num_chunks = width / simd_width;
    
    for chunk_idx in 0..num_chunks {
        let x_start = chunk_idx * simd_width;
        
        let mut center_vals = vec![T::zero(); simd_width];
        let mut north_vals = vec![T::zero(); simd_width];
        let mut south_vals = vec![T::zero(); simd_width];
        let mut east_vals = vec![T::zero(); simd_width];
        let mut west_vals = vec![T::zero(); simd_width];
        
        // Gather neighborhood values
        for i in 0..simd_width {
            let x = x_start + i;
            center_vals[i] = input[(y, x)];
            
            north_vals[i] = if y > 0 { input[(y - 1, x)] } else { center_vals[i] };
            south_vals[i] = if y < height - 1 { input[(y + 1, x)] } else { center_vals[i] };
            west_vals[i] = if x > 0 { input[(y, x - 1)] } else { center_vals[i] };
            east_vals[i] = if x < width - 1 { input[(y, x + 1)] } else { center_vals[i] };
        }
        
        // Compute gradients using SIMD
        let grad_n = T::simd_sub(&north_vals[..], &center_vals[..]);
        let grad_s = T::simd_sub(&south_vals[..], &center_vals[..]);
        let grad_e = T::simd_sub(&east_vals[..], &center_vals[..]);
        let grad_w = T::simd_sub(&west_vals[..], &center_vals[..]);
        
        // Compute diffusion coefficients
        let coeff_n = compute_diffusion_coeff(&grad_n, kappa_sq, option);
        let coeff_s = compute_diffusion_coeff(&grad_s, kappa_sq, option);
        let coeff_e = compute_diffusion_coeff(&grad_e, kappa_sq, option);
        let coeff_w = compute_diffusion_coeff(&grad_w, kappa_sq, option);
        
        // Update values
        for i in 0..simd_width {
            if x_start + i < width {
                let flux = coeff_n[i] * grad_n[i] + coeff_s[i] * grad_s[i]
                    + coeff_e[i] * grad_e[i] + coeff_w[i] * grad_w[i];
                output_row[x_start + i] = center_vals[i] + lambda * flux;
            }
        }
    }
    
    // Process remaining pixels
    for x in (num_chunks * simd_width)..width {
        let center = input[(y, x)];
        
        let north = if y > 0 { input[(y - 1, x)] } else { center };
        let south = if y < height - 1 { input[(y + 1, x)] } else { center };
        let west = if x > 0 { input[(y, x - 1)] } else { center };
        let east = if x < width - 1 { input[(y, x + 1)] } else { center };
        
        let grad_n = north - center;
        let grad_s = south - center;
        let grad_e = east - center;
        let grad_w = west - center;
        
        let coeff_n = compute_single_diffusion_coeff(grad_n, kappa_sq, option);
        let coeff_s = compute_single_diffusion_coeff(grad_s, kappa_sq, option);
        let coeff_e = compute_single_diffusion_coeff(grad_e, kappa_sq, option);
        let coeff_w = compute_single_diffusion_coeff(grad_w, kappa_sq, option);
        
        let flux = coeff_n * grad_n + coeff_s * grad_s + coeff_e * grad_e + coeff_w * grad_w;
        output_row[x] = center + lambda * flux;
    }
}

/// Compute diffusion coefficients for SIMD values
fn compute_diffusion_coeff<T>(gradients: &[T], kappa_sq: T, option: usize) -> Vec<T>
where
    T: Float + FromPrimitive,
{
    gradients
        .iter()
        .map(|&g| compute_single_diffusion_coeff(g, kappa_sq, option))
        .collect()
}

/// Compute single diffusion coefficient
fn compute_single_diffusion_coeff<T>(gradient: T, kappa_sq: T, option: usize) -> T
where
    T: Float + FromPrimitive,
{
    match option {
        1 => {
            // Exponential: c(g) = exp(-(g/kappa)²)
            (-(gradient * gradient) / kappa_sq).exp()
        }
        2 => {
            // Quadratic: c(g) = 1 / (1 + (g/kappa)²)
            T::one() / (T::one() + gradient * gradient / kappa_sq)
        }
        _ => T::one(),
    }
}

// Conditional compilation for parallel iterator
#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(not(feature = "parallel"))]
trait IntoParallelIterator {
    type Iter;
    fn into_par_iter(self) -> Self::Iter;
}

#[cfg(not(feature = "parallel"))]
impl<T> IntoParallelIterator for T
where
    T: IntoIterator,
{
    type Iter = T::IntoIter;
    fn into_par_iter(self) -> Self::Iter {
        self.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    
    #[test]
    fn test_bilateral_filter() {
        let input = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ];
        
        let result = simd_bilateral_filter(input.view(), 1.0, 2.0, Some(3)).unwrap();
        assert_eq!(result.shape(), input.shape());
    }
    
    #[test]
    fn test_anisotropic_diffusion() {
        let input = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ];
        
        let result = simd_anisotropic_diffusion(input.view(), 5, 2.0, 0.1, 1).unwrap();
        assert_eq!(result.shape(), input.shape());
    }
}