//! Parallel implementations of vector operations for iterative solvers
//!
//! This module provides SIMD and parallel accelerated implementations of common
//! vector operations used in iterative solvers, leveraging scirs2-core infrastructure.

use ndarray::ArrayView1;
use num_traits::{Float, NumAssign};
use std::iter::Sum;

// Import parallel and SIMD operations from scirs2-core
use scirs2_core::parallel_ops::*;
use scirs2_core::simd_ops::SimdUnifiedOps;

/// Configuration options for parallel vector operations
#[derive(Debug, Clone)]
pub struct ParallelVectorOptions {
    /// Use parallel processing for operations
    pub use_parallel: bool,
    /// Minimum vector length to trigger parallel processing
    pub parallel_threshold: usize,
    /// Chunk size for parallel processing
    pub chunk_size: usize,
    /// Use SIMD acceleration
    pub use_simd: bool,
    /// Minimum vector length to trigger SIMD processing
    pub simd_threshold: usize,
}

impl Default for ParallelVectorOptions {
    fn default() -> Self {
        Self {
            use_parallel: true,
            parallel_threshold: 10000,
            chunk_size: 1024,
            use_simd: true,
            simd_threshold: 32,
        }
    }
}

/// Parallel and SIMD accelerated dot product
///
/// Computes the dot product x^T * y using parallel processing and SIMD acceleration
/// when beneficial.
///
/// # Arguments
///
/// * `x` - First vector
/// * `y` - Second vector
/// * `options` - Optional configuration (uses default if None)
///
/// # Returns
///
/// The dot product sum(x[i] * y[i])
///
/// # Panics
///
/// Panics if vectors have different lengths
pub fn parallel_dot<T>(x: &[T], y: &[T], options: Option<ParallelVectorOptions>) -> T
where
    T: Float + NumAssign + Sum + Copy + Send + Sync + SimdUnifiedOps,
{
    assert_eq!(
        x.len(),
        y.len(),
        "Vector lengths must be equal for dot product"
    );

    if x.is_empty() {
        return T::zero();
    }

    let opts = options.unwrap_or_default();

    if opts.use_parallel && x.len() >= opts.parallel_threshold {
        // Parallel computation using work-stealing
        let chunks = x
            .chunks(opts.chunk_size)
            .zip(y.chunks(opts.chunk_size))
            .collect::<Vec<_>>();

        let partial_sums: Vec<T> = parallel_map(&chunks, |(x_chunk, y_chunk)| {
            if opts.use_simd && x_chunk.len() >= opts.simd_threshold {
                // Use SIMD acceleration for large chunks
                let x_view = ArrayView1::from(x_chunk);
                let y_view = ArrayView1::from(*y_chunk);
                T::simd_dot(&x_view, &y_view)
            } else {
                // Scalar computation for small chunks
                x_chunk
                    .iter()
                    .zip(y_chunk.iter())
                    .map(|(&xi, &yi)| xi * yi)
                    .sum()
            }
        });

        // Sum the partial results
        partial_sums.into_iter().sum()
    } else if opts.use_simd && x.len() >= opts.simd_threshold {
        // Use SIMD without parallelization
        let x_view = ArrayView1::from(x);
        let y_view = ArrayView1::from(y);
        T::simd_dot(&x_view, &y_view)
    } else {
        // Fallback to scalar computation
        x.iter().zip(y).map(|(&xi, &yi)| xi * yi).sum()
    }
}

/// Parallel and SIMD accelerated 2-norm computation
///
/// Computes the Euclidean norm ||x||_2 = sqrt(x^T * x) using parallel processing
/// and SIMD acceleration when beneficial.
///
/// # Arguments
///
/// * `x` - Input vector
/// * `options` - Optional configuration (uses default if None)
///
/// # Returns
///
/// The 2-norm of the vector
pub fn parallel_norm2<T>(x: &[T], options: Option<ParallelVectorOptions>) -> T
where
    T: Float + NumAssign + Sum + Copy + Send + Sync + SimdUnifiedOps,
{
    if x.is_empty() {
        return T::zero();
    }

    let opts = options.unwrap_or_default();

    if opts.use_parallel && x.len() >= opts.parallel_threshold {
        // Parallel computation
        let chunks = x.chunks(opts.chunk_size).collect::<Vec<_>>();

        let partial_sums: Vec<T> = parallel_map(&chunks, |chunk| {
            if opts.use_simd && chunk.len() >= opts.simd_threshold {
                // Use SIMD acceleration for large chunks
                let chunk_view = ArrayView1::from(*chunk);
                T::simd_dot(&chunk_view, &chunk_view)
            } else {
                // Scalar computation for small chunks
                chunk.iter().map(|&xi| xi * xi).sum()
            }
        });

        // Sum partial results and take square root
        partial_sums.into_iter().sum::<T>().sqrt()
    } else if opts.use_simd && x.len() >= opts.simd_threshold {
        // Use SIMD without parallelization
        let x_view = ArrayView1::from(x);
        T::simd_dot(&x_view, &x_view).sqrt()
    } else {
        // Fallback to scalar computation
        x.iter().map(|&xi| xi * xi).sum::<T>().sqrt()
    }
}

/// Parallel vector addition: z = x + y
///
/// Computes element-wise vector addition using parallel processing when beneficial.
///
/// # Arguments
///
/// * `x` - First vector
/// * `y` - Second vector  
/// * `z` - Output vector (will be overwritten)
/// * `options` - Optional configuration (uses default if None)
///
/// # Panics
///
/// Panics if vectors have different lengths
pub fn parallel_vector_add<T>(x: &[T], y: &[T], z: &mut [T], options: Option<ParallelVectorOptions>)
where
    T: Float + NumAssign + Send + Sync + Copy + SimdUnifiedOps,
{
    assert_eq!(x.len(), y.len(), "Input vector lengths must be equal");
    assert_eq!(x.len(), z.len(), "Output vector length must match input");

    if x.is_empty() {
        return;
    }

    let opts = options.unwrap_or_default();

    if opts.use_parallel && x.len() >= opts.parallel_threshold {
        // Parallel computation using chunks
        let chunk_size = opts.chunk_size;
        let chunks: Vec<_> = (0..x.len()).step_by(chunk_size).collect();

        // For now, fallback to sequential processing to avoid unsafe parallel writes
        for start in chunks {
            let end = (start + chunk_size).min(x.len());
            let x_slice = &x[start..end];
            let y_slice = &y[start..end];
            let z_slice = &mut z[start..end];

            if opts.use_simd && x_slice.len() >= opts.simd_threshold {
                // Use SIMD acceleration for large chunks
                let x_view = ArrayView1::from(x_slice);
                let y_view = ArrayView1::from(y_slice);
                let result = T::simd_add(&x_view, &y_view);
                z_slice.copy_from_slice(result.as_slice().unwrap());
            } else {
                // Scalar computation for small chunks
                for ((xi, yi), zi) in x_slice.iter().zip(y_slice).zip(z_slice.iter_mut()) {
                    *zi = *xi + *yi;
                }
            }
        }
    } else if opts.use_simd && x.len() >= opts.simd_threshold {
        // Use SIMD without parallelization
        let x_view = ArrayView1::from(x);
        let y_view = ArrayView1::from(y);
        let result = T::simd_add(&x_view, &y_view);
        z.copy_from_slice(result.as_slice().unwrap());
    } else {
        // Fallback to scalar computation
        for ((xi, yi), zi) in x.iter().zip(y).zip(z) {
            *zi = *xi + *yi;
        }
    }
}

/// Parallel vector subtraction: z = x - y
///
/// Computes element-wise vector subtraction using parallel processing when beneficial.
///
/// # Arguments
///
/// * `x` - First vector (minuend)
/// * `y` - Second vector (subtrahend)
/// * `z` - Output vector (will be overwritten)
/// * `options` - Optional configuration (uses default if None)
///
/// # Panics
///
/// Panics if vectors have different lengths
pub fn parallel_vector_sub<T>(x: &[T], y: &[T], z: &mut [T], options: Option<ParallelVectorOptions>)
where
    T: Float + NumAssign + Send + Sync + Copy + SimdUnifiedOps,
{
    assert_eq!(x.len(), y.len(), "Input vector lengths must be equal");
    assert_eq!(x.len(), z.len(), "Output vector length must match input");

    if x.is_empty() {
        return;
    }

    let opts = options.unwrap_or_default();

    if opts.use_parallel && x.len() >= opts.parallel_threshold {
        // Parallel computation
        let chunks: Vec<_> = x
            .chunks(opts.chunk_size)
            .zip(y.chunks(opts.chunk_size))
            .zip(z.chunks_mut(opts.chunk_size))
            .collect();

        // Sequential processing to avoid copy trait issues with mutable references
        for ((x_chunk, y_chunk), z_chunk) in chunks {
            if opts.use_simd && x_chunk.len() >= opts.simd_threshold {
                // Use SIMD acceleration for large chunks
                let x_view = ArrayView1::from(x_chunk);
                let y_view = ArrayView1::from(y_chunk);
                let result = T::simd_sub(&x_view, &y_view);
                z_chunk.copy_from_slice(result.as_slice().unwrap());
            } else {
                // Scalar computation for small chunks
                for ((xi, yi), zi) in x_chunk.iter().zip(y_chunk).zip(z_chunk) {
                    *zi = *xi - *yi;
                }
            }
        }
    } else if opts.use_simd && x.len() >= opts.simd_threshold {
        // Use SIMD without parallelization
        let x_view = ArrayView1::from(x);
        let y_view = ArrayView1::from(y);
        let result = T::simd_sub(&x_view, &y_view);
        z.copy_from_slice(result.as_slice().unwrap());
    } else {
        // Fallback to scalar computation
        for ((xi, yi), zi) in x.iter().zip(y).zip(z) {
            *zi = *xi - *yi;
        }
    }
}

/// Parallel AXPY operation: y = a*x + y
///
/// Computes the AXPY operation (scalar times vector plus vector) using parallel
/// processing when beneficial.
///
/// # Arguments
///
/// * `a` - Scalar multiplier
/// * `x` - Input vector to be scaled
/// * `y` - Input/output vector (will be modified in place)
/// * `options` - Optional configuration (uses default if None)
///
/// # Panics
///
/// Panics if vectors have different lengths
pub fn parallel_axpy<T>(a: T, x: &[T], y: &mut [T], options: Option<ParallelVectorOptions>)
where
    T: Float + NumAssign + Send + Sync + Copy + SimdUnifiedOps,
{
    assert_eq!(x.len(), y.len(), "Vector lengths must be equal for AXPY");

    if x.is_empty() {
        return;
    }

    let opts = options.unwrap_or_default();

    if opts.use_parallel && x.len() >= opts.parallel_threshold {
        // Parallel computation
        let chunks: Vec<_> = x
            .chunks(opts.chunk_size)
            .zip(y.chunks_mut(opts.chunk_size))
            .collect();

        // Sequential processing to avoid copy trait issues with mutable references
        for (x_chunk, y_chunk) in chunks {
            if opts.use_simd && x_chunk.len() >= opts.simd_threshold {
                // Use SIMD acceleration for large chunks
                let x_view = ArrayView1::from(x_chunk);
                let scaled = T::simd_scalar_mul(&x_view, a);
                let y_view = ArrayView1::from(y_chunk);
                let result = T::simd_add(&scaled.view(), &y_view);
                y_chunk.copy_from_slice(result.as_slice().unwrap());
            } else {
                // Scalar computation for small chunks
                for (xi, yi) in x_chunk.iter().zip(y_chunk) {
                    *yi = a * *xi + *yi;
                }
            }
        }
    } else if opts.use_simd && x.len() >= opts.simd_threshold {
        // Use SIMD without parallelization
        let x_view = ArrayView1::from(x);
        let scaled = T::simd_scalar_mul(&x_view, a);
        let y_view = ArrayView1::from(y);
        let result = T::simd_add(&scaled.view(), &y_view);
        y.copy_from_slice(result.as_slice().unwrap());
    } else {
        // Fallback to scalar computation
        for (xi, yi) in x.iter().zip(y) {
            *yi = a * *xi + *yi;
        }
    }
}

/// Parallel vector scaling: y = a*x
///
/// Computes element-wise vector scaling using parallel processing when beneficial.
///
/// # Arguments
///
/// * `a` - Scalar multiplier
/// * `x` - Input vector
/// * `y` - Output vector (will be overwritten)
/// * `options` - Optional configuration (uses default if None)
///
/// # Panics
///
/// Panics if vectors have different lengths
pub fn parallel_vector_scale<T>(a: T, x: &[T], y: &mut [T], options: Option<ParallelVectorOptions>)
where
    T: Float + NumAssign + Send + Sync + Copy + SimdUnifiedOps,
{
    assert_eq!(x.len(), y.len(), "Vector lengths must be equal for scaling");

    if x.is_empty() {
        return;
    }

    let opts = options.unwrap_or_default();

    if opts.use_parallel && x.len() >= opts.parallel_threshold {
        // Parallel computation
        let chunks: Vec<_> = x
            .chunks(opts.chunk_size)
            .zip(y.chunks_mut(opts.chunk_size))
            .collect();

        // Sequential processing to avoid copy trait issues with mutable references
        for (x_chunk, y_chunk) in chunks {
            if opts.use_simd && x_chunk.len() >= opts.simd_threshold {
                // Use SIMD acceleration for large chunks
                let x_view = ArrayView1::from(x_chunk);
                let result = T::simd_scalar_mul(&x_view, a);
                y_chunk.copy_from_slice(result.as_slice().unwrap());
            } else {
                // Scalar computation for small chunks
                for (xi, yi) in x_chunk.iter().zip(y_chunk) {
                    *yi = a * *xi;
                }
            }
        }
    } else if opts.use_simd && x.len() >= opts.simd_threshold {
        // Use SIMD without parallelization
        let x_view = ArrayView1::from(x);
        let result = T::simd_scalar_mul(&x_view, a);
        y.copy_from_slice(result.as_slice().unwrap());
    } else {
        // Fallback to scalar computation
        for (xi, yi) in x.iter().zip(y) {
            *yi = a * *xi;
        }
    }
}

/// Parallel vector copy: y = x
///
/// Copies vector x to y using parallel processing when beneficial.
///
/// # Arguments
///
/// * `x` - Source vector
/// * `y` - Destination vector (will be overwritten)
/// * `options` - Optional configuration (uses default if None)
///
/// # Panics
///
/// Panics if vectors have different lengths
pub fn parallel_vector_copy<T>(x: &[T], y: &mut [T], options: Option<ParallelVectorOptions>)
where
    T: Float + NumAssign + Send + Sync + Copy,
{
    assert_eq!(x.len(), y.len(), "Vector lengths must be equal for copy");

    if x.is_empty() {
        return;
    }

    let opts = options.unwrap_or_default();

    if opts.use_parallel && x.len() >= opts.parallel_threshold {
        // Parallel computation
        let chunks: Vec<_> = x
            .chunks(opts.chunk_size)
            .zip(y.chunks_mut(opts.chunk_size))
            .collect();

        // Sequential processing to avoid copy trait issues with mutable references
        for (x_chunk, y_chunk) in chunks {
            y_chunk.copy_from_slice(x_chunk);
        }
    } else {
        // Direct copy (already optimized by the compiler/runtime)
        y.copy_from_slice(x);
    }
}

/// Enhanced parallel linear combination: z = a*x + b*y
///
/// Computes element-wise linear combination using parallel processing when beneficial.
///
/// # Arguments
///
/// * `a` - Scalar multiplier for x
/// * `x` - First vector
/// * `b` - Scalar multiplier for y  
/// * `y` - Second vector
/// * `z` - Output vector (will be overwritten)
/// * `options` - Optional configuration (uses default if None)
///
/// # Panics
///
/// Panics if vectors have different lengths
pub fn parallel_linear_combination<T>(
    a: T,
    x: &[T],
    b: T,
    y: &[T],
    z: &mut [T],
    options: Option<ParallelVectorOptions>,
) where
    T: Float + NumAssign + Send + Sync + Copy + SimdUnifiedOps,
{
    assert_eq!(x.len(), y.len(), "Input vector lengths must be equal");
    assert_eq!(x.len(), z.len(), "Output vector length must match input");

    if x.is_empty() {
        return;
    }

    let opts = options.unwrap_or_default();

    if opts.use_parallel && x.len() >= opts.parallel_threshold {
        // Parallel computation
        let chunks: Vec<_> = x
            .chunks(opts.chunk_size)
            .zip(y.chunks(opts.chunk_size))
            .zip(z.chunks_mut(opts.chunk_size))
            .collect();

        // Sequential processing to avoid copy trait issues with mutable references
        for ((x_chunk, y_chunk), z_chunk) in chunks {
            if opts.use_simd && x_chunk.len() >= opts.simd_threshold {
                // Use SIMD acceleration for large chunks
                let x_view = ArrayView1::from(x_chunk);
                let y_view = ArrayView1::from(y_chunk);
                let ax = T::simd_scalar_mul(&x_view, a);
                let by = T::simd_scalar_mul(&y_view, b);
                let result = T::simd_add(&ax.view(), &by.view());
                z_chunk.copy_from_slice(result.as_slice().unwrap());
            } else {
                // Scalar computation for small chunks
                for ((xi, yi), zi) in x_chunk.iter().zip(y_chunk).zip(z_chunk) {
                    *zi = a * *xi + b * *yi;
                }
            }
        }
    } else if opts.use_simd && x.len() >= opts.simd_threshold {
        // Use SIMD without parallelization
        let x_view = ArrayView1::from(x);
        let y_view = ArrayView1::from(y);
        let ax = T::simd_scalar_mul(&x_view, a);
        let by = T::simd_scalar_mul(&y_view, b);
        let result = T::simd_add(&ax.view(), &by.view());
        z.copy_from_slice(result.as_slice().unwrap());
    } else {
        // Fallback to scalar computation
        for ((xi, yi), zi) in x.iter().zip(y).zip(z) {
            *zi = a * *xi + b * *yi;
        }
    }
}

// Removed parallel_for_chunks function as it's not used anymore

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_parallel_dot() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 3.0, 4.0, 5.0];

        let result = parallel_dot(&x, &y, None);
        let expected = 1.0 * 2.0 + 2.0 * 3.0 + 3.0 * 4.0 + 4.0 * 5.0; // = 40.0

        assert_relative_eq!(result, expected);
    }

    #[test]
    fn test_parallel_norm2() {
        let x = vec![3.0, 4.0]; // ||(3,4)|| = 5.0

        let result = parallel_norm2(&x, None);
        assert_relative_eq!(result, 5.0);
    }

    #[test]
    fn test_parallel_vector_add() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];
        let mut z = vec![0.0; 3];

        parallel_vector_add(&x, &y, &mut z, None);

        assert_relative_eq!(z[0], 5.0);
        assert_relative_eq!(z[1], 7.0);
        assert_relative_eq!(z[2], 9.0);
    }

    #[test]
    fn test_parallel_axpy() {
        let a = 2.0;
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![1.0, 1.0, 1.0];

        parallel_axpy(a, &x, &mut y, None);

        // y = 2*x + y = 2*[1,2,3] + [1,1,1] = [3,5,7]
        assert_relative_eq!(y[0], 3.0);
        assert_relative_eq!(y[1], 5.0);
        assert_relative_eq!(y[2], 7.0);
    }

    #[test]
    fn test_parallel_vector_scale() {
        let a = 3.0;
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0; 3];

        parallel_vector_scale(a, &x, &mut y, None);

        assert_relative_eq!(y[0], 3.0);
        assert_relative_eq!(y[1], 6.0);
        assert_relative_eq!(y[2], 9.0);
    }

    #[test]
    fn test_parallel_linear_combination() {
        let a = 2.0;
        let x = vec![1.0, 2.0];
        let b = 3.0;
        let y = vec![1.0, 1.0];
        let mut z = vec![0.0; 2];

        parallel_linear_combination(a, &x, b, &y, &mut z, None);

        // z = 2*x + 3*y = 2*[1,2] + 3*[1,1] = [5,7]
        assert_relative_eq!(z[0], 5.0);
        assert_relative_eq!(z[1], 7.0);
    }

    #[test]
    fn test_empty_vectors() {
        let x: Vec<f64> = vec![];
        let y: Vec<f64> = vec![];

        assert_eq!(parallel_dot(&x, &y, None), 0.0);
        assert_eq!(parallel_norm2(&x, None), 0.0);
    }

    #[test]
    fn test_large_vectors_trigger_parallel() {
        let opts = ParallelVectorOptions {
            parallel_threshold: 100,
            ..Default::default()
        };

        let x: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..1000).map(|i| (i + 1) as f64).collect();

        let result = parallel_dot(&x, &y, Some(opts));

        // Should use parallel computation for vectors of length 1000
        // Expected result: sum(i * (i+1)) for i in 0..1000
        let expected: f64 = (0..1000).map(|i| (i * (i + 1)) as f64).sum();
        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }
}
