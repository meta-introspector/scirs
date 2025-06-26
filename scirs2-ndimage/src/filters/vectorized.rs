//! Vectorized filtering for batch operations on multiple images
//!
//! This module provides efficient batch processing of multiple images with the same filter,
//! enabling significant performance improvements through:
//! - Parallel processing across images
//! - Shared kernel computations
//! - Optimized memory access patterns
//! - SIMD operations where applicable

use ndarray::{Array, Array2, Array3, Array4, ArrayView2, ArrayView3, Axis, Dimension};
use num_traits::{Float, FromPrimitive};
use scirs2_core::parallel_ops::*;
use scirs2_core::simd_ops::SimdUnifiedOps;
use std::fmt::Debug;

use super::{boundary_optimized::*, BorderMode};
use crate::error::{NdimageError, NdimageResult};

/// Batch processing configuration
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Number of threads to use for parallel processing
    pub num_threads: Option<usize>,
    /// Chunk size for processing batches
    pub chunk_size: Option<usize>,
    /// Whether to use SIMD optimizations
    pub use_simd: bool,
    /// Memory limit hint in bytes
    pub memory_limit: Option<usize>,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            num_threads: None,
            chunk_size: None,
            use_simd: true,
            memory_limit: None,
        }
    }
}

/// Apply Gaussian filter to a batch of 2D images
///
/// # Arguments
/// * `batch` - 3D array where first dimension is the batch dimension
/// * `sigma` - Standard deviation for Gaussian kernel
/// * `mode` - Boundary handling mode
/// * `cval` - Constant value for constant mode
/// * `config` - Batch processing configuration
pub fn gaussian_filter_batch<T>(
    batch: &Array3<T>,
    sigma: T,
    mode: BorderMode,
    cval: Option<T>,
    config: Option<BatchConfig>,
) -> NdimageResult<Array3<T>>
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + 'static,
{
    let config = config.unwrap_or_default();
    let (batch_size, height, width) = batch.dim();

    // Create Gaussian kernel once for all images
    let kernel_size = (T::from_f64(6.0).unwrap() * sigma)
        .ceil()
        .to_usize()
        .unwrap();
    let kernel_size = kernel_size | 1; // Ensure odd size
    let kernel = create_gaussian_kernel_2d(sigma, kernel_size)?;

    // Process batch
    if batch_size == 1 {
        // Single image, no need for parallel processing
        let img = batch.index_axis(Axis(0), 0);
        let result = convolve2d_optimized(&img.to_owned(), &kernel, mode, cval)?;
        let mut output = Array3::zeros((1, height, width));
        output.index_axis_mut(Axis(0), 0).assign(&result);
        Ok(output)
    } else if config.num_threads.unwrap_or(num_threads()) > 1 && batch_size > 2 {
        // Parallel processing across batch
        let indices: Vec<usize> = (0..batch_size).collect();

        let process_image = |&idx: &usize| -> Result<Array2<T>, scirs2_core::CoreError> {
            let img = batch.index_axis(Axis(0), idx).to_owned();
            convolve2d_optimized(&img, &kernel, mode, cval)
                .map_err(|e| scirs2_core::CoreError::Parallel(e.to_string()))
        };

        let results = parallel_map_result(&indices, process_image)?;

        // Combine results
        let mut output = Array3::zeros((batch_size, height, width));
        for (idx, result) in results.into_iter().enumerate() {
            output.index_axis_mut(Axis(0), idx).assign(&result);
        }

        Ok(output)
    } else {
        // Sequential processing
        let mut output = Array3::zeros((batch_size, height, width));

        for idx in 0..batch_size {
            let img = batch.index_axis(Axis(0), idx).to_owned();
            let result = convolve2d_optimized(&img, &kernel, mode, cval)?;
            output.index_axis_mut(Axis(0), idx).assign(&result);
        }

        Ok(output)
    }
}

/// Apply median filter to a batch of 2D images
pub fn median_filter_batch<T>(
    batch: &Array3<T>,
    size: &[usize],
    mode: BorderMode,
    config: Option<BatchConfig>,
) -> NdimageResult<Array3<T>>
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + PartialOrd + 'static,
{
    let config = config.unwrap_or_default();
    let (batch_size, height, width) = batch.dim();

    if size.len() != 2 {
        return Err(NdimageError::InvalidInput(
            "Size must have length 2 for 2D median filter".into(),
        ));
    }

    // Process batch
    if batch_size == 1 {
        // Single image
        let img = batch.index_axis(Axis(0), 0).to_owned();
        let result = crate::filters::median::median_filter(&img, size, Some(mode))?;
        let mut output = Array3::zeros((1, height, width));
        output.index_axis_mut(Axis(0), 0).assign(&result);
        Ok(output)
    } else if config.num_threads.unwrap_or(num_threads()) > 1 && batch_size > 2 {
        // Parallel processing
        let indices: Vec<usize> = (0..batch_size).collect();

        let process_image = |&idx: &usize| -> Result<Array2<T>, scirs2_core::CoreError> {
            let img = batch.index_axis(Axis(0), idx).to_owned();
            crate::filters::median::median_filter(&img, size, Some(mode))
                .map_err(|e| scirs2_core::CoreError::Parallel(e.to_string()))
        };

        let results = parallel_map_result(&indices, process_image)?;

        // Combine results
        let mut output = Array3::zeros((batch_size, height, width));
        for (idx, result) in results.into_iter().enumerate() {
            output.index_axis_mut(Axis(0), idx).assign(&result);
        }

        Ok(output)
    } else {
        // Sequential processing
        let mut output = Array3::zeros((batch_size, height, width));

        for idx in 0..batch_size {
            let img = batch.index_axis(Axis(0), idx).to_owned();
            let result = crate::filters::median::median_filter(&img, size, Some(mode))?;
            output.index_axis_mut(Axis(0), idx).assign(&result);
        }

        Ok(output)
    }
}

/// Apply generic convolution to a batch of images with the same kernel
pub fn convolve_batch<T>(
    batch: &Array3<T>,
    kernel: &Array2<T>,
    mode: BorderMode,
    cval: Option<T>,
    config: Option<BatchConfig>,
) -> NdimageResult<Array3<T>>
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + 'static,
{
    let config = config.unwrap_or_default();
    let (batch_size, height, width) = batch.dim();

    // Determine optimal processing strategy
    let chunk_size = config.chunk_size.unwrap_or_else(|| {
        // Auto-determine chunk size based on available memory and image size
        let image_bytes = height * width * std::mem::size_of::<T>();
        let available_memory = config.memory_limit.unwrap_or(1 << 30); // 1GB default
        let max_chunk = available_memory / (image_bytes * 4); // Conservative estimate
        max_chunk.max(1).min(batch_size)
    });

    if batch_size <= chunk_size {
        // Process entire batch at once
        process_batch_chunk(batch, kernel, mode, cval, &config)
    } else {
        // Process in chunks to manage memory
        let mut output = Array3::zeros((batch_size, height, width));

        for chunk_start in (0..batch_size).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(batch_size);
            let chunk_slice =
                batch.slice_axis(Axis(0), ndarray::Slice::from(chunk_start..chunk_end));

            let chunk_result =
                process_batch_chunk(&chunk_slice.to_owned(), kernel, mode, cval, &config)?;

            for (i, idx) in (chunk_start..chunk_end).enumerate() {
                output
                    .index_axis_mut(Axis(0), idx)
                    .assign(&chunk_result.index_axis(Axis(0), i));
            }
        }

        Ok(output)
    }
}

/// Process a chunk of images
fn process_batch_chunk<T>(
    chunk: &Array3<T>,
    kernel: &Array2<T>,
    mode: BorderMode,
    cval: Option<T>,
    config: &BatchConfig,
) -> NdimageResult<Array3<T>>
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + 'static,
{
    let (chunk_size, height, width) = chunk.dim();

    if config.num_threads.unwrap_or(num_threads()) > 1 && chunk_size > 1 {
        // Parallel processing
        let indices: Vec<usize> = (0..chunk_size).collect();

        let process_image = |&idx: &usize| -> Result<Array2<T>, scirs2_core::CoreError> {
            let img = chunk.index_axis(Axis(0), idx).to_owned();
            convolve2d_optimized(&img, kernel, mode, cval)
                .map_err(|e| scirs2_core::CoreError::Parallel(e.to_string()))
        };

        let results = parallel_map_result(&indices, process_image)?;

        // Combine results
        let mut output = Array3::zeros((chunk_size, height, width));
        for (idx, result) in results.into_iter().enumerate() {
            output.index_axis_mut(Axis(0), idx).assign(&result);
        }

        Ok(output)
    } else {
        // Sequential processing
        let mut output = Array3::zeros((chunk_size, height, width));

        for idx in 0..chunk_size {
            let img = chunk.index_axis(Axis(0), idx).to_owned();
            let result = convolve2d_optimized(&img, kernel, mode, cval)?;
            output.index_axis_mut(Axis(0), idx).assign(&result);
        }

        Ok(output)
    }
}

/// Apply Sobel edge detection to a batch of images
pub fn sobel_batch<T>(
    batch: &Array3<T>,
    axis: Option<usize>,
    mode: BorderMode,
    cval: Option<T>,
    config: Option<BatchConfig>,
) -> NdimageResult<Array3<T>>
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + 'static,
{
    let config = config.unwrap_or_default();
    let (batch_size, height, width) = batch.dim();

    // Create Sobel kernels
    let (kernel_x, kernel_y) = create_sobel_kernels();

    let process_fn = |img: &Array2<T>| -> NdimageResult<Array2<T>> {
        match axis {
            Some(0) => convolve2d_optimized(img, &kernel_y, mode, cval),
            Some(1) => convolve2d_optimized(img, &kernel_x, mode, cval),
            None => {
                // Gradient magnitude
                let gx = convolve2d_optimized(img, &kernel_x, mode, cval)?;
                let gy = convolve2d_optimized(img, &kernel_y, mode, cval)?;
                Ok((&gx * &gx + &gy * &gy).mapv(|v| v.sqrt()))
            }
            _ => Err(NdimageError::InvalidInput(
                "Invalid axis for Sobel filter".into(),
            )),
        }
    };

    if batch_size == 1 {
        let img = batch.index_axis(Axis(0), 0).to_owned();
        let result = process_fn(&img)?;
        let mut output = Array3::zeros((1, height, width));
        output.index_axis_mut(Axis(0), 0).assign(&result);
        Ok(output)
    } else if config.num_threads.unwrap_or(num_threads()) > 1 && batch_size > 2 {
        // Parallel processing
        let indices: Vec<usize> = (0..batch_size).collect();

        let process_image = |&idx: &usize| -> Result<Array2<T>, scirs2_core::CoreError> {
            let img = batch.index_axis(Axis(0), idx).to_owned();
            process_fn(&img).map_err(|e| scirs2_core::CoreError::Parallel(e.to_string()))
        };

        let results = parallel_map_result(&indices, process_image)?;

        // Combine results
        let mut output = Array3::zeros((batch_size, height, width));
        for (idx, result) in results.into_iter().enumerate() {
            output.index_axis_mut(Axis(0), idx).assign(&result);
        }

        Ok(output)
    } else {
        // Sequential processing
        let mut output = Array3::zeros((batch_size, height, width));

        for idx in 0..batch_size {
            let img = batch.index_axis(Axis(0), idx).to_owned();
            let result = process_fn(&img)?;
            output.index_axis_mut(Axis(0), idx).assign(&result);
        }

        Ok(output)
    }
}

/// Apply a custom filter function to a batch of images
pub fn apply_filter_batch<T, F>(
    batch: &Array3<T>,
    filter_fn: F,
    config: Option<BatchConfig>,
) -> NdimageResult<Array3<T>>
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + 'static,
    F: Fn(&Array2<T>) -> NdimageResult<Array2<T>> + Clone + Send + Sync,
{
    let config = config.unwrap_or_default();
    let (batch_size, height, width) = batch.dim();

    if batch_size == 1 {
        let img = batch.index_axis(Axis(0), 0).to_owned();
        let result = filter_fn(&img)?;
        let mut output = Array3::zeros((1, height, width));
        output.index_axis_mut(Axis(0), 0).assign(&result);
        Ok(output)
    } else if config.num_threads.unwrap_or(num_threads()) > 1 && batch_size > 2 {
        // Parallel processing
        let indices: Vec<usize> = (0..batch_size).collect();

        let process_image = |&idx: &usize| -> Result<Array2<T>, scirs2_core::CoreError> {
            let img = batch.index_axis(Axis(0), idx).to_owned();
            filter_fn(&img).map_err(|e| scirs2_core::CoreError::Parallel(e.to_string()))
        };

        let results = parallel_map_result(&indices, process_image)?;

        // Combine results
        let mut output = Array3::zeros((batch_size, height, width));
        for (idx, result) in results.into_iter().enumerate() {
            output.index_axis_mut(Axis(0), idx).assign(&result);
        }

        Ok(output)
    } else {
        // Sequential processing
        let mut output = Array3::zeros((batch_size, height, width));

        for idx in 0..batch_size {
            let img = batch.index_axis(Axis(0), idx).to_owned();
            let result = filter_fn(&img)?;
            output.index_axis_mut(Axis(0), idx).assign(&result);
        }

        Ok(output)
    }
}

// Helper functions

fn create_gaussian_kernel_2d<T>(sigma: T, size: usize) -> NdimageResult<Array2<T>>
where
    T: Float + FromPrimitive + Debug,
{
    let mut kernel = Array2::zeros((size, size));
    let center = (size / 2) as f64;
    let sigma_f64 = sigma.to_f64().unwrap();
    let two_sigma_sq = 2.0 * sigma_f64 * sigma_f64;

    let mut sum = 0.0;

    for i in 0..size {
        for j in 0..size {
            let x = i as f64 - center;
            let y = j as f64 - center;
            let dist_sq = x * x + y * y;
            let val = (-dist_sq / two_sigma_sq).exp();
            kernel[[i, j]] = T::from_f64(val).unwrap();
            sum += val;
        }
    }

    // Normalize
    let sum_t = T::from_f64(sum).unwrap();
    kernel.mapv_inplace(|v| v / sum_t);

    Ok(kernel)
}

fn create_sobel_kernels<T>() -> (Array2<T>, Array2<T>)
where
    T: Float + FromPrimitive,
{
    let kernel_x = Array2::from_shape_vec(
        (3, 3),
        vec![
            T::from_i32(-1).unwrap(),
            T::zero(),
            T::from_i32(1).unwrap(),
            T::from_i32(-2).unwrap(),
            T::zero(),
            T::from_i32(2).unwrap(),
            T::from_i32(-1).unwrap(),
            T::zero(),
            T::from_i32(1).unwrap(),
        ],
    )
    .unwrap();

    let kernel_y = Array2::from_shape_vec(
        (3, 3),
        vec![
            T::from_i32(-1).unwrap(),
            T::from_i32(-2).unwrap(),
            T::from_i32(-1).unwrap(),
            T::zero(),
            T::zero(),
            T::zero(),
            T::from_i32(1).unwrap(),
            T::from_i32(2).unwrap(),
            T::from_i32(1).unwrap(),
        ],
    )
    .unwrap();

    (kernel_x, kernel_y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr3;

    #[test]
    fn test_gaussian_filter_batch() {
        // Create a batch of 3 simple 3x3 images
        let batch = arr3(&[
            [[1.0, 1.0, 1.0], [1.0, 5.0, 1.0], [1.0, 1.0, 1.0]],
            [[0.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 0.0]],
            [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
        ]);

        let result =
            gaussian_filter_batch(&batch, 1.0, BorderMode::Constant, Some(0.0), None).unwrap();

        assert_eq!(result.shape(), batch.shape());

        // First image should be smoothed
        assert!(result[[0, 1, 1]] < 5.0);
        assert!(result[[0, 1, 1]] > 1.0);

        // Second image should be smoothed
        assert!(result[[1, 1, 1]] < 10.0);
        assert!(result[[1, 1, 1]] > 0.0);

        // Third image should remain relatively unchanged
        assert!((result[[2, 1, 1]] - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_convolve_batch() {
        let batch = arr3(&[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]);

        let kernel = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();

        let config = BatchConfig {
            num_threads: Some(1), // Force sequential for deterministic test
            ..Default::default()
        };

        let result = convolve_batch(
            &batch,
            &kernel,
            BorderMode::Constant,
            Some(0.0),
            Some(config),
        )
        .unwrap();

        assert_eq!(result.shape(), batch.shape());

        // Check convolution results
        // For convolution, kernel is flipped, so it becomes [[1,0],[0,1]]
        // At position (0,0) of first image: 1*1 + 0*0 = 1
        assert_eq!(result[[0, 0, 0]], 1.0);

        // At position (0,0) of second image: 5*1 + 0*0 = 5
        assert_eq!(result[[1, 0, 0]], 5.0);
    }
}
