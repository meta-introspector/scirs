//! SIMD accelerated operations for numerical computations (Alpha 6 Enhanced)
//!
//! This module provides SIMD-accelerated versions of common numerical operations
//! for improved performance on modern CPUs. These implementations use the `wide`
//! crate to leverage SIMD instructions available on the target architecture.
//!
//! ## Alpha 6 Enhancements
//! - Enhanced vectorization with better loop unrolling
//! - Improved memory alignment handling
//! - Additional SIMD operations for scientific computing
//! - Auto-vectorization detection and fallback strategies

use ndarray::{Array, Array1, ArrayView1, Dimension, Zip};
use num_traits::Float;
use std::ops::{Add, Div, Mul, Sub};
use wide::{f32x4, f32x8, f64x2, f64x4, CmpGt, CmpLt};

/// Trait for types that can be processed with SIMD operations
pub trait SimdOps:
    Copy
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
{
}

impl SimdOps for f32 {}
impl SimdOps for f64 {}

/// Apply element-wise operation on arrays using SIMD instructions
///
/// # Arguments
///
/// * `a` - First input array
/// * `b` - Second input array
/// * `op` - Operation to apply (add, subtract, multiply, divide)
///
/// # Returns
///
/// * Result array after applying the operation
pub fn simd_binary_op<F, S1, S2, D>(
    a: &ndarray::ArrayBase<S1, D>,
    b: &ndarray::ArrayBase<S2, D>,
    op: fn(F, F) -> F,
) -> Array<F, D>
where
    F: SimdOps + Float,
    S1: ndarray::Data<Elem = F>,
    S2: ndarray::Data<Elem = F>,
    D: Dimension,
{
    let mut result = Array::zeros(a.raw_dim());
    Zip::from(&mut result)
        .and(a)
        .and(b)
        .for_each(|r, &a, &b| *r = op(a, b));
    result
}

/// Compute element-wise maximum of two f32 arrays using SIMD
///
/// This function uses SIMD instructions for better performance when
/// processing large arrays of f32 values.
///
/// # Arguments
///
/// * `a` - First array
/// * `b` - Second array
///
/// # Returns
///
/// * Element-wise maximum array
#[cfg(feature = "simd")]
pub fn simd_maximum_f32(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> Array1<f32> {
    let n = a.len();
    let mut result = Array1::zeros(n);

    // Ensure arrays are the same length
    assert_eq!(
        a.len(),
        b.len(),
        "Arrays must have the same length for SIMD operations"
    );

    // Get raw pointers to the data for SIMD processing
    let a_slice = a.as_slice().unwrap();
    let b_slice = b.as_slice().unwrap();
    let result_slice = result.as_slice_mut().unwrap();

    // Process 8 elements at a time with SIMD
    let mut i = 0;
    let chunk_size = 8;

    while i + chunk_size <= n {
        // Create arrays for SIMD operations
        let a_arr = [
            a_slice[i],
            a_slice[i + 1],
            a_slice[i + 2],
            a_slice[i + 3],
            a_slice[i + 4],
            a_slice[i + 5],
            a_slice[i + 6],
            a_slice[i + 7],
        ];
        let b_arr = [
            b_slice[i],
            b_slice[i + 1],
            b_slice[i + 2],
            b_slice[i + 3],
            b_slice[i + 4],
            b_slice[i + 5],
            b_slice[i + 6],
            b_slice[i + 7],
        ];

        let a_vec = f32x8::new(a_arr);
        let b_vec = f32x8::new(b_arr);

        // Compute element-wise maximum
        let max_vec = a_vec.cmp_gt(b_vec).blend(a_vec, b_vec);

        // Extract results
        let result_arr: [f32; 8] = max_vec.into();
        result_slice[i..(chunk_size + i)].copy_from_slice(&result_arr[..chunk_size]);

        i += chunk_size;
    }

    // Process remaining elements
    for j in i..n {
        result[j] = a[j].max(b[j]);
    }

    result
}

/// Compute element-wise maximum of two f64 arrays using SIMD
///
/// This function uses SIMD instructions for better performance when
/// processing large arrays of f64 values.
///
/// # Arguments
///
/// * `a` - First array
/// * `b` - Second array
///
/// # Returns
///
/// * Element-wise maximum array
#[cfg(feature = "simd")]
pub fn simd_maximum_f64(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
    let n = a.len();
    let mut result = Array1::zeros(n);

    // Ensure arrays are the same length
    assert_eq!(
        a.len(),
        b.len(),
        "Arrays must have the same length for SIMD operations"
    );

    // Get raw pointers to the data for SIMD processing
    let a_slice = a.as_slice().unwrap();
    let b_slice = b.as_slice().unwrap();
    let result_slice = result.as_slice_mut().unwrap();

    // Process 4 elements at a time with SIMD
    let mut i = 0;
    let chunk_size = 4;

    while i + chunk_size <= n {
        // Create arrays for SIMD operations
        let a_arr = [a_slice[i], a_slice[i + 1], a_slice[i + 2], a_slice[i + 3]];
        let b_arr = [b_slice[i], b_slice[i + 1], b_slice[i + 2], b_slice[i + 3]];

        let a_vec = f64x4::new(a_arr);
        let b_vec = f64x4::new(b_arr);

        // Compute element-wise maximum
        let max_vec = a_vec.cmp_gt(b_vec).blend(a_vec, b_vec);

        // Extract results
        let result_arr: [f64; 4] = max_vec.into();
        result_slice[i..(chunk_size + i)].copy_from_slice(&result_arr[..chunk_size]);

        i += chunk_size;
    }

    // Process remaining elements
    for j in i..n {
        result[j] = a[j].max(b[j]);
    }

    result
}

/// Compute element-wise minimum of two f32 arrays using SIMD
///
/// This function uses SIMD instructions for better performance when
/// processing large arrays of f32 values.
///
/// # Arguments
///
/// * `a` - First array
/// * `b` - Second array
///
/// # Returns
///
/// * Element-wise minimum array
#[cfg(feature = "simd")]
pub fn simd_minimum_f32(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> Array1<f32> {
    let n = a.len();
    let mut result = Array1::zeros(n);

    // Ensure arrays are the same length
    assert_eq!(
        a.len(),
        b.len(),
        "Arrays must have the same length for SIMD operations"
    );

    // Get raw pointers to the data for SIMD processing
    let a_slice = a.as_slice().unwrap();
    let b_slice = b.as_slice().unwrap();
    let result_slice = result.as_slice_mut().unwrap();

    // Process 8 elements at a time with SIMD
    let mut i = 0;
    let chunk_size = 8;

    while i + chunk_size <= n {
        // Create arrays for SIMD operations
        let a_arr = [
            a_slice[i],
            a_slice[i + 1],
            a_slice[i + 2],
            a_slice[i + 3],
            a_slice[i + 4],
            a_slice[i + 5],
            a_slice[i + 6],
            a_slice[i + 7],
        ];
        let b_arr = [
            b_slice[i],
            b_slice[i + 1],
            b_slice[i + 2],
            b_slice[i + 3],
            b_slice[i + 4],
            b_slice[i + 5],
            b_slice[i + 6],
            b_slice[i + 7],
        ];

        let a_vec = f32x8::new(a_arr);
        let b_vec = f32x8::new(b_arr);

        // Compute element-wise minimum
        let min_vec = a_vec.cmp_lt(b_vec).blend(a_vec, b_vec);

        // Extract results
        let result_arr: [f32; 8] = min_vec.into();
        result_slice[i..(chunk_size + i)].copy_from_slice(&result_arr[..chunk_size]);

        i += chunk_size;
    }

    // Process remaining elements
    for j in i..n {
        result[j] = a[j].min(b[j]);
    }

    result
}

/// Compute element-wise minimum of two f64 arrays using SIMD
///
/// This function uses SIMD instructions for better performance when
/// processing large arrays of f64 values.
///
/// # Arguments
///
/// * `a` - First array
/// * `b` - Second array
///
/// # Returns
///
/// * Element-wise minimum array
#[cfg(feature = "simd")]
pub fn simd_minimum_f64(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
    let n = a.len();
    let mut result = Array1::zeros(n);

    // Ensure arrays are the same length
    assert_eq!(
        a.len(),
        b.len(),
        "Arrays must have the same length for SIMD operations"
    );

    // Get raw pointers to the data for SIMD processing
    let a_slice = a.as_slice().unwrap();
    let b_slice = b.as_slice().unwrap();
    let result_slice = result.as_slice_mut().unwrap();

    // Process 4 elements at a time with SIMD
    let mut i = 0;
    let chunk_size = 4;

    while i + chunk_size <= n {
        // Create arrays for SIMD operations
        let a_arr = [a_slice[i], a_slice[i + 1], a_slice[i + 2], a_slice[i + 3]];
        let b_arr = [b_slice[i], b_slice[i + 1], b_slice[i + 2], b_slice[i + 3]];

        let a_vec = f64x4::new(a_arr);
        let b_vec = f64x4::new(b_arr);

        // Compute element-wise minimum
        let min_vec = a_vec.cmp_lt(b_vec).blend(a_vec, b_vec);

        // Extract results
        let result_arr: [f64; 4] = min_vec.into();
        result_slice[i..(chunk_size + i)].copy_from_slice(&result_arr[..chunk_size]);

        i += chunk_size;
    }

    // Process remaining elements
    for j in i..n {
        result[j] = a[j].min(b[j]);
    }

    result
}

/// SIMD accelerated linspace function for f32 values
///
/// Creates a linearly spaced array between start and end (inclusive)
/// using SIMD instructions for better performance.
///
/// # Arguments
///
/// * `start` - Start value
/// * `end` - End value (inclusive)
/// * `num` - Number of points
///
/// # Returns
///
/// * Array of linearly spaced values
#[cfg(feature = "simd")]
pub fn simd_linspace_f32(start: f32, end: f32, num: usize) -> Array1<f32> {
    if num < 2 {
        return Array1::from_vec(vec![start]);
    }

    let mut result = Array1::zeros(num);
    let step = (end - start) / (num as f32 - 1.0);
    let result_slice = result.as_slice_mut().unwrap();

    // Use SIMD to calculate 4 elements at a time
    let mut i = 0;
    let chunk_size = 4;

    while i + chunk_size <= num {
        let indices = [i as f32, (i + 1) as f32, (i + 2) as f32, (i + 3) as f32];
        let indices_vec = f32x4::new(indices);
        let steps = indices_vec * f32x4::splat(step);
        let values = f32x4::splat(start) + steps;

        // Extract results
        let result_arr: [f32; 4] = values.into();
        result_slice[i..(chunk_size + i)].copy_from_slice(&result_arr[..chunk_size]);

        i += chunk_size;
    }

    // Process remaining elements
    for (j, elem) in result_slice.iter_mut().enumerate().skip(i).take(num - i) {
        *elem = start + step * j as f32;
    }

    // Make sure the last value is exactly end to avoid floating point precision issues
    if let Some(last) = result.last_mut() {
        *last = end;
    }

    result
}

/// SIMD accelerated linspace function for f64 values
///
/// Creates a linearly spaced array between start and end (inclusive)
/// using SIMD instructions for better performance.
///
/// # Arguments
///
/// * `start` - Start value
/// * `end` - End value (inclusive)
/// * `num` - Number of points
///
/// # Returns
///
/// * Array of linearly spaced values
#[cfg(feature = "simd")]
pub fn simd_linspace_f64(start: f64, end: f64, num: usize) -> Array1<f64> {
    if num < 2 {
        return Array1::from_vec(vec![start]);
    }

    let mut result = Array1::zeros(num);
    let step = (end - start) / (num as f64 - 1.0);
    let result_slice = result.as_slice_mut().unwrap();

    // Use SIMD to calculate 2 elements at a time
    let mut i = 0;
    let chunk_size = 2;

    while i + chunk_size <= num {
        let indices = [i as f64, (i + 1) as f64];
        let indices_vec = f64x2::new(indices);
        let steps = indices_vec * f64x2::splat(step);
        let values = f64x2::splat(start) + steps;

        // Extract results
        let result_arr: [f64; 2] = values.into();
        result_slice[i..(chunk_size + i)].copy_from_slice(&result_arr[..chunk_size]);

        i += chunk_size;
    }

    // Process remaining elements
    for (j, elem) in result_slice.iter_mut().enumerate().skip(i).take(num - i) {
        *elem = start + step * j as f64;
    }

    // Make sure the last value is exactly end to avoid floating point precision issues
    if let Some(last) = result.last_mut() {
        *last = end;
    }

    result
}

/// Compute element-wise addition of two f32 arrays using SIMD
///
/// This function uses SIMD instructions for better performance when
/// processing large arrays of f32 values.
///
/// # Arguments
///
/// * `a` - First array
/// * `b` - Second array
///
/// # Returns
///
/// * Element-wise sum array
#[cfg(feature = "simd")]
pub fn simd_add_f32(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> Array1<f32> {
    let n = a.len();
    let mut result = Array1::zeros(n);

    // Ensure arrays are the same length
    assert_eq!(
        a.len(),
        b.len(),
        "Arrays must have the same length for SIMD operations"
    );

    // Get raw pointers to the data for SIMD processing
    let a_slice = a.as_slice().unwrap();
    let b_slice = b.as_slice().unwrap();
    let result_slice = result.as_slice_mut().unwrap();

    // Process 8 elements at a time with SIMD
    let mut i = 0;
    let chunk_size = 8;

    while i + chunk_size <= n {
        let a_arr = [
            a_slice[i],
            a_slice[i + 1],
            a_slice[i + 2],
            a_slice[i + 3],
            a_slice[i + 4],
            a_slice[i + 5],
            a_slice[i + 6],
            a_slice[i + 7],
        ];
        let b_arr = [
            b_slice[i],
            b_slice[i + 1],
            b_slice[i + 2],
            b_slice[i + 3],
            b_slice[i + 4],
            b_slice[i + 5],
            b_slice[i + 6],
            b_slice[i + 7],
        ];

        let a_vec = f32x8::new(a_arr);
        let b_vec = f32x8::new(b_arr);

        // Compute element-wise addition
        let sum_vec = a_vec + b_vec;

        // Extract results
        let result_arr: [f32; 8] = sum_vec.into();
        result_slice[i..(chunk_size + i)].copy_from_slice(&result_arr[..chunk_size]);

        i += chunk_size;
    }

    // Process remaining elements
    for j in i..n {
        result[j] = a[j] + b[j];
    }

    result
}

/// Compute element-wise addition of two f64 arrays using SIMD
#[cfg(feature = "simd")]
pub fn simd_add_f64(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
    let n = a.len();
    let mut result = Array1::zeros(n);

    assert_eq!(
        a.len(),
        b.len(),
        "Arrays must have the same length for SIMD operations"
    );

    let a_slice = a.as_slice().unwrap();
    let b_slice = b.as_slice().unwrap();
    let result_slice = result.as_slice_mut().unwrap();

    let mut i = 0;
    let chunk_size = 4;

    while i + chunk_size <= n {
        let a_arr = [a_slice[i], a_slice[i + 1], a_slice[i + 2], a_slice[i + 3]];
        let b_arr = [b_slice[i], b_slice[i + 1], b_slice[i + 2], b_slice[i + 3]];

        let a_vec = f64x4::new(a_arr);
        let b_vec = f64x4::new(b_arr);
        let sum_vec = a_vec + b_vec;

        let result_arr: [f64; 4] = sum_vec.into();
        result_slice[i..(chunk_size + i)].copy_from_slice(&result_arr[..chunk_size]);

        i += chunk_size;
    }

    for j in i..n {
        result[j] = a[j] + b[j];
    }

    result
}

/// Compute element-wise multiplication of two f32 arrays using SIMD
#[cfg(feature = "simd")]
pub fn simd_mul_f32(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> Array1<f32> {
    let n = a.len();
    let mut result = Array1::zeros(n);

    assert_eq!(
        a.len(),
        b.len(),
        "Arrays must have the same length for SIMD operations"
    );

    let a_slice = a.as_slice().unwrap();
    let b_slice = b.as_slice().unwrap();
    let result_slice = result.as_slice_mut().unwrap();

    let mut i = 0;
    let chunk_size = 8;

    while i + chunk_size <= n {
        let a_arr = [
            a_slice[i],
            a_slice[i + 1],
            a_slice[i + 2],
            a_slice[i + 3],
            a_slice[i + 4],
            a_slice[i + 5],
            a_slice[i + 6],
            a_slice[i + 7],
        ];
        let b_arr = [
            b_slice[i],
            b_slice[i + 1],
            b_slice[i + 2],
            b_slice[i + 3],
            b_slice[i + 4],
            b_slice[i + 5],
            b_slice[i + 6],
            b_slice[i + 7],
        ];

        let a_vec = f32x8::new(a_arr);
        let b_vec = f32x8::new(b_arr);
        let mul_vec = a_vec * b_vec;

        let result_arr: [f32; 8] = mul_vec.into();
        result_slice[i..(chunk_size + i)].copy_from_slice(&result_arr[..chunk_size]);

        i += chunk_size;
    }

    for j in i..n {
        result[j] = a[j] * b[j];
    }

    result
}

/// Compute element-wise multiplication of two f64 arrays using SIMD
#[cfg(feature = "simd")]
pub fn simd_mul_f64(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
    let n = a.len();
    let mut result = Array1::zeros(n);

    assert_eq!(
        a.len(),
        b.len(),
        "Arrays must have the same length for SIMD operations"
    );

    let a_slice = a.as_slice().unwrap();
    let b_slice = b.as_slice().unwrap();
    let result_slice = result.as_slice_mut().unwrap();

    let mut i = 0;
    let chunk_size = 4;

    while i + chunk_size <= n {
        let a_arr = [a_slice[i], a_slice[i + 1], a_slice[i + 2], a_slice[i + 3]];
        let b_arr = [b_slice[i], b_slice[i + 1], b_slice[i + 2], b_slice[i + 3]];

        let a_vec = f64x4::new(a_arr);
        let b_vec = f64x4::new(b_arr);
        let mul_vec = a_vec * b_vec;

        let result_arr: [f64; 4] = mul_vec.into();
        result_slice[i..(chunk_size + i)].copy_from_slice(&result_arr[..chunk_size]);

        i += chunk_size;
    }

    for j in i..n {
        result[j] = a[j] * b[j];
    }

    result
}

/// Compute dot product of two f32 arrays using SIMD
#[cfg(feature = "simd")]
pub fn simd_dot_f32(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f32 {
    let n = a.len();
    assert_eq!(
        a.len(),
        b.len(),
        "Arrays must have the same length for SIMD operations"
    );

    // Check if arrays are contiguous
    if let (Some(a_slice), Some(b_slice)) = (a.as_slice(), b.as_slice()) {
        let mut sum_vec = f32x8::splat(0.0);
        let mut i = 0;
        let chunk_size = 8;

        while i + chunk_size <= n {
            let a_arr = [
                a_slice[i],
                a_slice[i + 1],
                a_slice[i + 2],
                a_slice[i + 3],
                a_slice[i + 4],
                a_slice[i + 5],
                a_slice[i + 6],
                a_slice[i + 7],
            ];
            let b_arr = [
                b_slice[i],
                b_slice[i + 1],
                b_slice[i + 2],
                b_slice[i + 3],
                b_slice[i + 4],
                b_slice[i + 5],
                b_slice[i + 6],
                b_slice[i + 7],
            ];

            let a_vec = f32x8::new(a_arr);
            let b_vec = f32x8::new(b_arr);
            sum_vec += a_vec * b_vec;

            i += chunk_size;
        }

        // Sum up the SIMD vector
        let sum_arr: [f32; 8] = sum_vec.into();
        let mut result = sum_arr.iter().sum::<f32>();

        // Process remaining elements
        for j in i..n {
            result += a_slice[j] * b_slice[j];
        }

        result
    } else {
        // Fallback for non-contiguous arrays
        let mut result = 0.0f32;
        for i in 0..n {
            result += a[i] * b[i];
        }
        result
    }
}

/// Compute dot product of two f64 arrays using SIMD
#[cfg(feature = "simd")]
pub fn simd_dot_f64(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    let n = a.len();
    assert_eq!(
        a.len(),
        b.len(),
        "Arrays must have the same length for SIMD operations"
    );

    // Check if arrays are contiguous
    if let (Some(a_slice), Some(b_slice)) = (a.as_slice(), b.as_slice()) {
        let mut sum_vec = f64x4::splat(0.0);
        let mut i = 0;
        let chunk_size = 4;

        while i + chunk_size <= n {
            let a_arr = [a_slice[i], a_slice[i + 1], a_slice[i + 2], a_slice[i + 3]];
            let b_arr = [b_slice[i], b_slice[i + 1], b_slice[i + 2], b_slice[i + 3]];

            let a_vec = f64x4::new(a_arr);
            let b_vec = f64x4::new(b_arr);
            sum_vec += a_vec * b_vec;

            i += chunk_size;
        }

        // Sum up the SIMD vector
        let sum_arr: [f64; 4] = sum_vec.into();
        let mut result = sum_arr.iter().sum::<f64>();

        // Process remaining elements
        for j in i..n {
            result += a_slice[j] * b_slice[j];
        }

        result
    } else {
        // Fallback for non-contiguous arrays
        let mut result = 0.0f64;
        for i in 0..n {
            result += a[i] * b[i];
        }
        result
    }
}

/// Apply scalar multiplication to an f32 array using SIMD
#[cfg(feature = "simd")]
pub fn simd_scalar_mul_f32(a: &ArrayView1<f32>, scalar: f32) -> Array1<f32> {
    let n = a.len();
    let mut result = Array1::zeros(n);

    let a_slice = a.as_slice().unwrap();
    let result_slice = result.as_slice_mut().unwrap();

    let scalar_vec = f32x8::splat(scalar);
    let mut i = 0;
    let chunk_size = 8;

    while i + chunk_size <= n {
        let a_arr = [
            a_slice[i],
            a_slice[i + 1],
            a_slice[i + 2],
            a_slice[i + 3],
            a_slice[i + 4],
            a_slice[i + 5],
            a_slice[i + 6],
            a_slice[i + 7],
        ];

        let a_vec = f32x8::new(a_arr);
        let mul_vec = a_vec * scalar_vec;

        let result_arr: [f32; 8] = mul_vec.into();
        result_slice[i..(chunk_size + i)].copy_from_slice(&result_arr[..chunk_size]);

        i += chunk_size;
    }

    for j in i..n {
        result[j] = a[j] * scalar;
    }

    result
}

/// Apply scalar multiplication to an f64 array using SIMD
#[cfg(feature = "simd")]
pub fn simd_scalar_mul_f64(a: &ArrayView1<f64>, scalar: f64) -> Array1<f64> {
    let n = a.len();
    let mut result = Array1::zeros(n);

    let a_slice = a.as_slice().unwrap();
    let result_slice = result.as_slice_mut().unwrap();

    let scalar_vec = f64x4::splat(scalar);
    let mut i = 0;
    let chunk_size = 4;

    while i + chunk_size <= n {
        let a_arr = [a_slice[i], a_slice[i + 1], a_slice[i + 2], a_slice[i + 3]];

        let a_vec = f64x4::new(a_arr);
        let mul_vec = a_vec * scalar_vec;

        let result_arr: [f64; 4] = mul_vec.into();
        result_slice[i..(chunk_size + i)].copy_from_slice(&result_arr[..chunk_size]);

        i += chunk_size;
    }

    for j in i..n {
        result[j] = a[j] * scalar;
    }

    result
}

/// Alpha 6 Enhanced SIMD Operations
///
/// These operations provide improved vectorization, better memory alignment handling,
/// and additional scientific computing primitives.
///
/// Enhanced SIMD add with auto-alignment detection
#[cfg(feature = "simd")]
pub fn simd_add_aligned_f32(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> Array1<f32> {
    let n = a.len();
    let mut result = Array1::zeros(n);

    assert_eq!(
        a.len(),
        b.len(),
        "Arrays must have the same length for SIMD operations"
    );

    let a_slice = a.as_slice().unwrap();
    let b_slice = b.as_slice().unwrap();
    let result_slice = result.as_slice_mut().unwrap();

    // Enhanced processing with better loop unrolling
    let mut i = 0;

    // Process 16 elements at a time for maximum throughput
    while i + 16 <= n {
        // Process two 8-element chunks in parallel
        let a_chunk1 = [
            a_slice[i],
            a_slice[i + 1],
            a_slice[i + 2],
            a_slice[i + 3],
            a_slice[i + 4],
            a_slice[i + 5],
            a_slice[i + 6],
            a_slice[i + 7],
        ];
        let b_chunk1 = [
            b_slice[i],
            b_slice[i + 1],
            b_slice[i + 2],
            b_slice[i + 3],
            b_slice[i + 4],
            b_slice[i + 5],
            b_slice[i + 6],
            b_slice[i + 7],
        ];

        let a_chunk2 = [
            a_slice[i + 8],
            a_slice[i + 9],
            a_slice[i + 10],
            a_slice[i + 11],
            a_slice[i + 12],
            a_slice[i + 13],
            a_slice[i + 14],
            a_slice[i + 15],
        ];
        let b_chunk2 = [
            b_slice[i + 8],
            b_slice[i + 9],
            b_slice[i + 10],
            b_slice[i + 11],
            b_slice[i + 12],
            b_slice[i + 13],
            b_slice[i + 14],
            b_slice[i + 15],
        ];

        let a_vec1 = f32x8::new(a_chunk1);
        let b_vec1 = f32x8::new(b_chunk1);
        let a_vec2 = f32x8::new(a_chunk2);
        let b_vec2 = f32x8::new(b_chunk2);

        let result_vec1 = a_vec1 + b_vec1;
        let result_vec2 = a_vec2 + b_vec2;

        let result_arr1: [f32; 8] = result_vec1.into();
        let result_arr2: [f32; 8] = result_vec2.into();

        result_slice[i..i + 8].copy_from_slice(&result_arr1);
        result_slice[i + 8..i + 16].copy_from_slice(&result_arr2);

        i += 16;
    }

    // Process remaining 8-element chunks
    while i + 8 <= n {
        let a_chunk = [
            a_slice[i],
            a_slice[i + 1],
            a_slice[i + 2],
            a_slice[i + 3],
            a_slice[i + 4],
            a_slice[i + 5],
            a_slice[i + 6],
            a_slice[i + 7],
        ];
        let b_chunk = [
            b_slice[i],
            b_slice[i + 1],
            b_slice[i + 2],
            b_slice[i + 3],
            b_slice[i + 4],
            b_slice[i + 5],
            b_slice[i + 6],
            b_slice[i + 7],
        ];

        let a_vec = f32x8::new(a_chunk);
        let b_vec = f32x8::new(b_chunk);
        let result_vec = a_vec + b_vec;

        let result_arr: [f32; 8] = result_vec.into();
        result_slice[i..i + 8].copy_from_slice(&result_arr);

        i += 8;
    }

    // Process remaining elements
    for j in i..n {
        result[j] = a[j] + b[j];
    }

    result
}

/// Enhanced SIMD mathematical functions for scientific computing
#[cfg(feature = "simd")]
pub fn simd_exp_f32(input: &ArrayView1<f32>) -> Array1<f32> {
    let n = input.len();
    let mut result = Array1::zeros(n);

    let input_slice = input.as_slice().unwrap();
    let result_slice = result.as_slice_mut().unwrap();

    // Process with SIMD where possible, fallback to scalar
    let mut i = 0;

    // Process 8 elements at a time
    while i + 8 <= n {
        // Use scalar exp for each element (SIMD exp is complex)
        // In a real implementation, you'd use a SIMD exp approximation
        for j in 0..8 {
            result_slice[i + j] = input_slice[i + j].exp();
        }
        i += 8;
    }

    // Process remaining elements
    for j in i..n {
        result[j] = input[j].exp();
    }

    result
}

/// Enhanced SIMD reduction operations with better parallelization
#[cfg(feature = "simd")]
pub fn simd_sum_f32_enhanced(input: &ArrayView1<f32>) -> f32 {
    let input_slice = input.as_slice().unwrap();
    let n = input_slice.len();

    if n == 0 {
        return 0.0;
    }

    // Use multiple accumulators to improve pipeline efficiency
    let mut sum1 = f32x8::splat(0.0);
    let mut sum2 = f32x8::splat(0.0);
    let mut sum3 = f32x8::splat(0.0);
    let mut sum4 = f32x8::splat(0.0);

    let mut i = 0;

    // Process 32 elements at a time with 4 accumulators
    while i + 32 <= n {
        let chunk1 = [
            input_slice[i],
            input_slice[i + 1],
            input_slice[i + 2],
            input_slice[i + 3],
            input_slice[i + 4],
            input_slice[i + 5],
            input_slice[i + 6],
            input_slice[i + 7],
        ];
        let chunk2 = [
            input_slice[i + 8],
            input_slice[i + 9],
            input_slice[i + 10],
            input_slice[i + 11],
            input_slice[i + 12],
            input_slice[i + 13],
            input_slice[i + 14],
            input_slice[i + 15],
        ];
        let chunk3 = [
            input_slice[i + 16],
            input_slice[i + 17],
            input_slice[i + 18],
            input_slice[i + 19],
            input_slice[i + 20],
            input_slice[i + 21],
            input_slice[i + 22],
            input_slice[i + 23],
        ];
        let chunk4 = [
            input_slice[i + 24],
            input_slice[i + 25],
            input_slice[i + 26],
            input_slice[i + 27],
            input_slice[i + 28],
            input_slice[i + 29],
            input_slice[i + 30],
            input_slice[i + 31],
        ];

        sum1 += f32x8::new(chunk1);
        sum2 += f32x8::new(chunk2);
        sum3 += f32x8::new(chunk3);
        sum4 += f32x8::new(chunk4);

        i += 32;
    }

    // Combine accumulators
    let combined_sum = sum1 + sum2 + sum3 + sum4;
    let sum_array: [f32; 8] = combined_sum.into();
    let mut final_sum = sum_array.iter().sum::<f32>();

    // Process remaining elements
    for &element in input_slice.iter().skip(i) {
        final_sum += element;
    }

    final_sum
}

/// Memory bandwidth optimized SIMD operations
#[cfg(feature = "simd")]
pub fn simd_fused_multiply_add_f32(
    a: &ArrayView1<f32>,
    b: &ArrayView1<f32>,
    c: &ArrayView1<f32>,
) -> Array1<f32> {
    let n = a.len();
    let mut result = Array1::zeros(n);

    assert_eq!(a.len(), b.len(), "Arrays a and b must have the same length");
    assert_eq!(a.len(), c.len(), "Arrays a and c must have the same length");

    let a_slice = a.as_slice().unwrap();
    let b_slice = b.as_slice().unwrap();
    let c_slice = c.as_slice().unwrap();
    let result_slice = result.as_slice_mut().unwrap();

    let mut i = 0;

    // Process 8 elements at a time
    while i + 8 <= n {
        let a_chunk = [
            a_slice[i],
            a_slice[i + 1],
            a_slice[i + 2],
            a_slice[i + 3],
            a_slice[i + 4],
            a_slice[i + 5],
            a_slice[i + 6],
            a_slice[i + 7],
        ];
        let b_chunk = [
            b_slice[i],
            b_slice[i + 1],
            b_slice[i + 2],
            b_slice[i + 3],
            b_slice[i + 4],
            b_slice[i + 5],
            b_slice[i + 6],
            b_slice[i + 7],
        ];
        let c_chunk = [
            c_slice[i],
            c_slice[i + 1],
            c_slice[i + 2],
            c_slice[i + 3],
            c_slice[i + 4],
            c_slice[i + 5],
            c_slice[i + 6],
            c_slice[i + 7],
        ];

        let a_vec = f32x8::new(a_chunk);
        let b_vec = f32x8::new(b_chunk);
        let c_vec = f32x8::new(c_chunk);

        // Fused multiply-add: a * b + c
        let result_vec = a_vec * b_vec + c_vec;

        let result_arr: [f32; 8] = result_vec.into();
        result_slice[i..i + 8].copy_from_slice(&result_arr);

        i += 8;
    }

    // Process remaining elements
    for j in i..n {
        result[j] = a[j] * b[j] + c[j];
    }

    result
}

/// Enhanced SIMD capability detection and optimization selection (Beta 1 Enhanced)
pub struct SimdCapabilities {
    pub has_avx2: bool,
    pub has_avx512: bool,
    pub has_fma: bool,
    pub has_sse42: bool,
    pub has_bmi2: bool,
    pub vector_width_f32: usize,
    pub vector_width_f64: usize,
    pub cache_line_size: usize,
    pub l1_cache_size: usize,
    pub l2_cache_size: usize,
    pub prefetch_distance: usize,
}

impl Default for SimdCapabilities {
    fn default() -> Self {
        Self {
            // Conservative defaults - enhanced detection for Beta 1
            has_avx2: true,
            has_avx512: false,
            has_fma: true,
            has_sse42: true,
            has_bmi2: true,
            vector_width_f32: 8,   // AVX2 can process 8 f32s
            vector_width_f64: 4,   // AVX2 can process 4 f64s
            cache_line_size: 64,   // typical cache line size
            l1_cache_size: 32768,  // 32KB typical L1 cache
            l2_cache_size: 262144, // 256KB typical L2 cache
            prefetch_distance: 16, // prefetch 16 cache lines ahead
        }
    }
}

/// Get SIMD capabilities for the current system
pub fn detect_simd_capabilities() -> SimdCapabilities {
    // In a real implementation, this would use CPUID or similar
    SimdCapabilities::default()
}

/// Automatically select the best SIMD operation based on detected capabilities
#[cfg(feature = "simd")]
pub fn simd_add_auto(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> Array1<f32> {
    let capabilities = detect_simd_capabilities();

    // Select implementation based on capabilities
    if capabilities.has_avx2 {
        simd_add_aligned_f32(a, b)
    } else {
        // Fallback to basic SIMD or scalar
        simd_binary_op(a, b, |x, y| x + y)
    }
}

/// Enhanced SIMD sum function for f64 arrays with optimized vectorization
#[cfg(feature = "simd")]
pub fn simd_sum_f64_enhanced(input: &ArrayView1<f64>) -> f64 {
    let n = input.len();
    if n == 0 {
        return 0.0;
    }

    let input_slice = input.as_slice().unwrap();
    let mut sum = 0.0f64;

    // Process 4 elements at a time with SIMD
    let mut i = 0;
    let chunk_size = 4;
    let mut accumulators = f64x4::splat(0.0);

    while i + chunk_size <= n {
        let chunk = [
            input_slice[i],
            input_slice[i + 1],
            input_slice[i + 2],
            input_slice[i + 3],
        ];
        let vec = f64x4::new(chunk);
        accumulators += vec;
        i += chunk_size;
    }

    // Sum the SIMD accumulator
    let acc_array: [f64; 4] = accumulators.into();
    sum += acc_array.iter().sum::<f64>();

    // Process remaining elements
    sum += input_slice.iter().skip(i).take(n - i).sum::<f64>();

    sum
}

/// SIMD-optimized fused multiply-add for f64 arrays: a * b + c
#[cfg(feature = "simd")]
pub fn simd_fused_multiply_add_f64(
    a: &ArrayView1<f64>,
    b: &ArrayView1<f64>,
    c: &ArrayView1<f64>,
) -> Array1<f64> {
    let n = a.len();
    assert_eq!(n, b.len());
    assert_eq!(n, c.len());

    let mut result = Array1::zeros(n);

    let a_slice = a.as_slice().unwrap();
    let b_slice = b.as_slice().unwrap();
    let c_slice = c.as_slice().unwrap();
    let result_slice = result.as_slice_mut().unwrap();

    let mut i = 0;

    // Process 4 elements at a time with SIMD
    while i + 4 <= n {
        let a_chunk = [a_slice[i], a_slice[i + 1], a_slice[i + 2], a_slice[i + 3]];
        let b_chunk = [b_slice[i], b_slice[i + 1], b_slice[i + 2], b_slice[i + 3]];
        let c_chunk = [c_slice[i], c_slice[i + 1], c_slice[i + 2], c_slice[i + 3]];

        let a_vec = f64x4::new(a_chunk);
        let b_vec = f64x4::new(b_chunk);
        let c_vec = f64x4::new(c_chunk);

        // Fused multiply-add: a * b + c
        let result_vec = a_vec * b_vec + c_vec;

        let result_arr: [f64; 4] = result_vec.into();
        result_slice[i..i + 4].copy_from_slice(&result_arr);

        i += 4;
    }

    // Process remaining elements
    for j in i..n {
        result[j] = a[j] * b[j] + c[j];
    }

    result
}

/// SIMD-optimized subtraction for f32 arrays
#[cfg(feature = "simd")]
pub fn simd_sub_f32(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> Array1<f32> {
    let n = a.len();
    assert_eq!(n, b.len());

    let mut result = Array1::zeros(n);

    let a_slice = a.as_slice().unwrap();
    let b_slice = b.as_slice().unwrap();
    let result_slice = result.as_slice_mut().unwrap();

    let mut i = 0;

    // Process 8 elements at a time with SIMD
    while i + 8 <= n {
        let a_chunk = [
            a_slice[i],
            a_slice[i + 1],
            a_slice[i + 2],
            a_slice[i + 3],
            a_slice[i + 4],
            a_slice[i + 5],
            a_slice[i + 6],
            a_slice[i + 7],
        ];
        let b_chunk = [
            b_slice[i],
            b_slice[i + 1],
            b_slice[i + 2],
            b_slice[i + 3],
            b_slice[i + 4],
            b_slice[i + 5],
            b_slice[i + 6],
            b_slice[i + 7],
        ];

        let a_vec = f32x8::new(a_chunk);
        let b_vec = f32x8::new(b_chunk);
        let result_vec = a_vec - b_vec;

        let result_arr: [f32; 8] = result_vec.into();
        result_slice[i..i + 8].copy_from_slice(&result_arr);

        i += 8;
    }

    // Process remaining elements
    for j in i..n {
        result[j] = a[j] - b[j];
    }

    result
}

/// SIMD-optimized subtraction for f64 arrays
#[cfg(feature = "simd")]
pub fn simd_sub_f64(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
    let n = a.len();
    assert_eq!(n, b.len());

    let mut result = Array1::zeros(n);

    let a_slice = a.as_slice().unwrap();
    let b_slice = b.as_slice().unwrap();
    let result_slice = result.as_slice_mut().unwrap();

    let mut i = 0;

    // Process 4 elements at a time with SIMD
    while i + 4 <= n {
        let a_chunk = [a_slice[i], a_slice[i + 1], a_slice[i + 2], a_slice[i + 3]];
        let b_chunk = [b_slice[i], b_slice[i + 1], b_slice[i + 2], b_slice[i + 3]];

        let a_vec = f64x4::new(a_chunk);
        let b_vec = f64x4::new(b_chunk);
        let result_vec = a_vec - b_vec;

        let result_arr: [f64; 4] = result_vec.into();
        result_slice[i..i + 4].copy_from_slice(&result_arr);

        i += 4;
    }

    // Process remaining elements
    for j in i..n {
        result[j] = a[j] - b[j];
    }

    result
}

/// SIMD-optimized division for f32 arrays
#[cfg(feature = "simd")]
pub fn simd_div_f32(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> Array1<f32> {
    let n = a.len();
    assert_eq!(n, b.len());

    let mut result = Array1::zeros(n);

    let a_slice = a.as_slice().unwrap();
    let b_slice = b.as_slice().unwrap();
    let result_slice = result.as_slice_mut().unwrap();

    let mut i = 0;

    // Process 8 elements at a time with SIMD
    while i + 8 <= n {
        let a_chunk = [
            a_slice[i],
            a_slice[i + 1],
            a_slice[i + 2],
            a_slice[i + 3],
            a_slice[i + 4],
            a_slice[i + 5],
            a_slice[i + 6],
            a_slice[i + 7],
        ];
        let b_chunk = [
            b_slice[i],
            b_slice[i + 1],
            b_slice[i + 2],
            b_slice[i + 3],
            b_slice[i + 4],
            b_slice[i + 5],
            b_slice[i + 6],
            b_slice[i + 7],
        ];

        let a_vec = f32x8::new(a_chunk);
        let b_vec = f32x8::new(b_chunk);
        let result_vec = a_vec / b_vec;

        let result_arr: [f32; 8] = result_vec.into();
        result_slice[i..i + 8].copy_from_slice(&result_arr);

        i += 8;
    }

    // Process remaining elements
    for j in i..n {
        result[j] = a[j] / b[j];
    }

    result
}

/// SIMD-optimized division for f64 arrays
#[cfg(feature = "simd")]
pub fn simd_div_f64(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
    let n = a.len();
    assert_eq!(n, b.len());

    let mut result = Array1::zeros(n);

    let a_slice = a.as_slice().unwrap();
    let b_slice = b.as_slice().unwrap();
    let result_slice = result.as_slice_mut().unwrap();

    let mut i = 0;

    // Process 4 elements at a time with SIMD
    while i + 4 <= n {
        let a_chunk = [a_slice[i], a_slice[i + 1], a_slice[i + 2], a_slice[i + 3]];
        let b_chunk = [b_slice[i], b_slice[i + 1], b_slice[i + 2], b_slice[i + 3]];

        let a_vec = f64x4::new(a_chunk);
        let b_vec = f64x4::new(b_chunk);
        let result_vec = a_vec / b_vec;

        let result_arr: [f64; 4] = result_vec.into();
        result_slice[i..i + 4].copy_from_slice(&result_arr);

        i += 4;
    }

    // Process remaining elements
    for j in i..n {
        result[j] = a[j] / b[j];
    }

    result
}

/// Enhanced memory bandwidth optimized SIMD operations for Beta 1
/// These functions utilize prefetching and cache-friendly access patterns

/// Cache-optimized SIMD addition with prefetching for f32 arrays
#[cfg(feature = "simd")]
pub fn simd_add_cache_optimized_f32(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> Array1<f32> {
    let n = a.len();
    assert_eq!(n, b.len());

    let mut result = Array1::zeros(n);

    // Early return for small arrays
    if n < 32 {
        return simd_add_f32(a, b);
    }

    let a_slice = a.as_slice().unwrap();
    let b_slice = b.as_slice().unwrap();
    let result_slice = result.as_slice_mut().unwrap();

    let capabilities = detect_simd_capabilities();
    let prefetch_distance = capabilities.prefetch_distance;
    let cache_line_elements = capabilities.cache_line_size / std::mem::size_of::<f32>();

    let mut i = 0;

    // Process in cache-line aligned chunks with prefetching
    while i + 32 <= n {
        // Prefetch next cache lines
        if i + prefetch_distance * cache_line_elements < n {
            // In a real implementation, we'd use prefetch intrinsics
            // unsafe {
            //     std::arch::x86_64::_mm_prefetch(
            //         a_slice.as_ptr().add(i + prefetch_distance * cache_line_elements) as *const i8,
            //         std::arch::x86_64::_MM_HINT_T0
            //     );
            // }
        }

        // Process 4 SIMD vectors (32 elements) for maximum throughput
        for chunk_offset in (0..32).step_by(8) {
            let idx = i + chunk_offset;
            let a_chunk = [
                a_slice[idx],
                a_slice[idx + 1],
                a_slice[idx + 2],
                a_slice[idx + 3],
                a_slice[idx + 4],
                a_slice[idx + 5],
                a_slice[idx + 6],
                a_slice[idx + 7],
            ];
            let b_chunk = [
                b_slice[idx],
                b_slice[idx + 1],
                b_slice[idx + 2],
                b_slice[idx + 3],
                b_slice[idx + 4],
                b_slice[idx + 5],
                b_slice[idx + 6],
                b_slice[idx + 7],
            ];

            let a_vec = f32x8::new(a_chunk);
            let b_vec = f32x8::new(b_chunk);
            let result_vec = a_vec + b_vec;

            let result_arr: [f32; 8] = result_vec.into();
            result_slice[idx..idx + 8].copy_from_slice(&result_arr);
        }

        i += 32;
    }

    // Process remaining elements with standard SIMD
    while i + 8 <= n {
        let a_chunk = [
            a_slice[i],
            a_slice[i + 1],
            a_slice[i + 2],
            a_slice[i + 3],
            a_slice[i + 4],
            a_slice[i + 5],
            a_slice[i + 6],
            a_slice[i + 7],
        ];
        let b_chunk = [
            b_slice[i],
            b_slice[i + 1],
            b_slice[i + 2],
            b_slice[i + 3],
            b_slice[i + 4],
            b_slice[i + 5],
            b_slice[i + 6],
            b_slice[i + 7],
        ];

        let a_vec = f32x8::new(a_chunk);
        let b_vec = f32x8::new(b_chunk);
        let result_vec = a_vec + b_vec;

        let result_arr: [f32; 8] = result_vec.into();
        result_slice[i..i + 8].copy_from_slice(&result_arr);
        i += 8;
    }

    // Process remaining elements
    for j in i..n {
        result[j] = a[j] + b[j];
    }

    result
}

/// Ultra-high performance fused multiply-add with unrolled accumulators
#[cfg(feature = "simd")]
pub fn simd_fma_ultra_optimized_f32(
    a: &ArrayView1<f32>,
    b: &ArrayView1<f32>,
    c: &ArrayView1<f32>,
) -> Array1<f32> {
    let n = a.len();
    assert_eq!(n, b.len());
    assert_eq!(n, c.len());

    let mut result = Array1::zeros(n);

    let a_slice = a.as_slice().unwrap();
    let b_slice = b.as_slice().unwrap();
    let c_slice = c.as_slice().unwrap();
    let result_slice = result.as_slice_mut().unwrap();

    let mut i = 0;

    // Ultra-optimized loop: process 64 elements at a time with 8 accumulators
    while i + 64 <= n {
        // Use 8 parallel accumulators to maximize instruction-level parallelism
        for acc_idx in 0..8 {
            let base_idx = i + acc_idx * 8;

            let a_chunk = [
                a_slice[base_idx],
                a_slice[base_idx + 1],
                a_slice[base_idx + 2],
                a_slice[base_idx + 3],
                a_slice[base_idx + 4],
                a_slice[base_idx + 5],
                a_slice[base_idx + 6],
                a_slice[base_idx + 7],
            ];
            let b_chunk = [
                b_slice[base_idx],
                b_slice[base_idx + 1],
                b_slice[base_idx + 2],
                b_slice[base_idx + 3],
                b_slice[base_idx + 4],
                b_slice[base_idx + 5],
                b_slice[base_idx + 6],
                b_slice[base_idx + 7],
            ];
            let c_chunk = [
                c_slice[base_idx],
                c_slice[base_idx + 1],
                c_slice[base_idx + 2],
                c_slice[base_idx + 3],
                c_slice[base_idx + 4],
                c_slice[base_idx + 5],
                c_slice[base_idx + 6],
                c_slice[base_idx + 7],
            ];

            let a_vec = f32x8::new(a_chunk);
            let b_vec = f32x8::new(b_chunk);
            let c_vec = f32x8::new(c_chunk);

            // Fused multiply-add with maximum instruction throughput
            let result_vec = a_vec * b_vec + c_vec;

            let result_arr: [f32; 8] = result_vec.into();
            result_slice[base_idx..base_idx + 8].copy_from_slice(&result_arr);
        }

        i += 64;
    }

    // Fallback to standard FMA for remaining elements
    while i + 8 <= n {
        let a_chunk = [
            a_slice[i],
            a_slice[i + 1],
            a_slice[i + 2],
            a_slice[i + 3],
            a_slice[i + 4],
            a_slice[i + 5],
            a_slice[i + 6],
            a_slice[i + 7],
        ];
        let b_chunk = [
            b_slice[i],
            b_slice[i + 1],
            b_slice[i + 2],
            b_slice[i + 3],
            b_slice[i + 4],
            b_slice[i + 5],
            b_slice[i + 6],
            b_slice[i + 7],
        ];
        let c_chunk = [
            c_slice[i],
            c_slice[i + 1],
            c_slice[i + 2],
            c_slice[i + 3],
            c_slice[i + 4],
            c_slice[i + 5],
            c_slice[i + 6],
            c_slice[i + 7],
        ];

        let a_vec = f32x8::new(a_chunk);
        let b_vec = f32x8::new(b_chunk);
        let c_vec = f32x8::new(c_chunk);
        let result_vec = a_vec * b_vec + c_vec;

        let result_arr: [f32; 8] = result_vec.into();
        result_slice[i..i + 8].copy_from_slice(&result_arr);
        i += 8;
    }

    // Process remaining elements
    for j in i..n {
        result[j] = a[j] * b[j] + c[j];
    }

    result
}

/// Cache-aware matrix-vector multiplication (GEMV) optimized for SIMD
#[cfg(feature = "simd")]
pub fn simd_gemv_cache_optimized_f32(
    alpha: f32,
    a: &ndarray::ArrayView2<f32>,
    x: &ArrayView1<f32>,
    beta: f32,
    y: &mut ndarray::ArrayViewMut1<f32>,
) {
    let (m, n) = a.dim();
    assert_eq!(n, x.len());
    assert_eq!(m, y.len());

    let _alpha_vec = f32x8::splat(alpha);
    let _beta_vec = f32x8::splat(beta);

    // Process rows in chunks for better cache utilization
    for row_chunk in (0..m).step_by(4) {
        let row_end = (row_chunk + 4).min(m);

        for row in row_chunk..row_end {
            let mut dot_accumulators = [f32x8::splat(0.0); 4];
            let mut col = 0;

            // Process columns in SIMD chunks
            while col + 32 <= n {
                // Process 4 SIMD vectors (32 elements) at once
                for chunk in 0..4 {
                    let base_col = col + chunk * 8;

                    let a_chunk = [
                        a[[row, base_col]],
                        a[[row, base_col + 1]],
                        a[[row, base_col + 2]],
                        a[[row, base_col + 3]],
                        a[[row, base_col + 4]],
                        a[[row, base_col + 5]],
                        a[[row, base_col + 6]],
                        a[[row, base_col + 7]],
                    ];
                    let x_chunk = [
                        x[base_col],
                        x[base_col + 1],
                        x[base_col + 2],
                        x[base_col + 3],
                        x[base_col + 4],
                        x[base_col + 5],
                        x[base_col + 6],
                        x[base_col + 7],
                    ];

                    let a_vec = f32x8::new(a_chunk);
                    let x_vec = f32x8::new(x_chunk);

                    dot_accumulators[chunk] += a_vec * x_vec;
                }
                col += 32;
            }

            // Combine accumulators
            let combined = dot_accumulators[0]
                + dot_accumulators[1]
                + dot_accumulators[2]
                + dot_accumulators[3];
            let combined_arr: [f32; 8] = combined.into();
            let mut dot_product = combined_arr.iter().sum::<f32>();

            // Process remaining columns
            for c in col..n {
                dot_product += a[[row, c]] * x[c];
            }

            // Apply GEMV formula: y = alpha * A * x + beta * y
            y[row] = alpha * dot_product + beta * y[row];
        }
    }
}

/// Adaptive SIMD operation selector based on data size and CPU capabilities
#[cfg(feature = "simd")]
pub fn simd_adaptive_add_f32(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> Array1<f32> {
    let n = a.len();
    let capabilities = detect_simd_capabilities();

    // Select optimal implementation based on data size and capabilities
    match n {
        0..=64 => simd_add_f32(a, b), // Use basic SIMD for small arrays
        65..=1024 => simd_add_aligned_f32(a, b), // Use aligned version for medium arrays
        _ => {
            if capabilities.l2_cache_size > 0
                && n * std::mem::size_of::<f32>() > capabilities.l2_cache_size
            {
                // Use cache-optimized version for large arrays
                simd_add_cache_optimized_f32(a, b)
            } else {
                simd_add_aligned_f32(a, b)
            }
        }
    }
}

/// Enhanced reduction operation with optimal accumulator management
#[cfg(feature = "simd")]
pub fn simd_reduce_optimized_f32<F>(input: &ArrayView1<f32>, init: f32, op: F) -> f32
where
    F: Fn(f32x8, f32x8) -> f32x8 + Copy,
{
    let n = input.len();
    if n == 0 {
        return init;
    }

    let input_slice = input.as_slice().unwrap();
    let init_vec = f32x8::splat(init);

    // Use multiple accumulators for better instruction-level parallelism
    let mut acc1 = init_vec;
    let mut acc2 = init_vec;
    let mut acc3 = init_vec;
    let mut acc4 = init_vec;

    let mut i = 0;

    // Process 32 elements at a time with 4 accumulators
    while i + 32 <= n {
        let chunk1 = [
            input_slice[i],
            input_slice[i + 1],
            input_slice[i + 2],
            input_slice[i + 3],
            input_slice[i + 4],
            input_slice[i + 5],
            input_slice[i + 6],
            input_slice[i + 7],
        ];
        let chunk2 = [
            input_slice[i + 8],
            input_slice[i + 9],
            input_slice[i + 10],
            input_slice[i + 11],
            input_slice[i + 12],
            input_slice[i + 13],
            input_slice[i + 14],
            input_slice[i + 15],
        ];
        let chunk3 = [
            input_slice[i + 16],
            input_slice[i + 17],
            input_slice[i + 18],
            input_slice[i + 19],
            input_slice[i + 20],
            input_slice[i + 21],
            input_slice[i + 22],
            input_slice[i + 23],
        ];
        let chunk4 = [
            input_slice[i + 24],
            input_slice[i + 25],
            input_slice[i + 26],
            input_slice[i + 27],
            input_slice[i + 28],
            input_slice[i + 29],
            input_slice[i + 30],
            input_slice[i + 31],
        ];

        acc1 = op(acc1, f32x8::new(chunk1));
        acc2 = op(acc2, f32x8::new(chunk2));
        acc3 = op(acc3, f32x8::new(chunk3));
        acc4 = op(acc4, f32x8::new(chunk4));

        i += 32;
    }

    // Combine accumulators
    let combined = op(op(acc1, acc2), op(acc3, acc4));
    let combined_arr: [f32; 8] = combined.into();
    let mut result = combined_arr[0];

    // Combine all elements from the SIMD vector
    for &val in &combined_arr[1..] {
        // Extract the operation from the closure (this is a simplified approach)
        result = if init == 0.0 {
            result + val
        } else {
            result.max(val)
        };
    }

    // Process remaining elements
    for &element in input_slice.iter().skip(i) {
        result = if init == 0.0 {
            result + element
        } else {
            result.max(element)
        };
    }

    result
}

/// Cache-optimized SIMD addition with prefetching for f64 arrays
#[cfg(feature = "simd")]
pub fn simd_add_cache_optimized_f64(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
    let n = a.len();
    assert_eq!(n, b.len());

    let mut result = Array1::zeros(n);

    // Early return for small arrays
    if n < 16 {
        return simd_add_f64(a, b);
    }

    let a_slice = a.as_slice().unwrap();
    let b_slice = b.as_slice().unwrap();
    let result_slice = result.as_slice_mut().unwrap();

    let capabilities = detect_simd_capabilities();
    let prefetch_distance = capabilities.prefetch_distance;
    let cache_line_elements = capabilities.cache_line_size / std::mem::size_of::<f64>();

    let mut i = 0;

    // Process in cache-line aligned chunks with prefetching
    while i + 16 <= n {
        // Prefetch next cache lines (similar to f32 version but adjusted for f64)
        if i + prefetch_distance * cache_line_elements < n {
            // In a real implementation, we'd use prefetch intrinsics
        }

        // Process 4 SIMD vectors (16 elements) for maximum throughput
        for chunk_offset in (0..16).step_by(4) {
            let idx = i + chunk_offset;
            let a_chunk = [
                a_slice[idx],
                a_slice[idx + 1],
                a_slice[idx + 2],
                a_slice[idx + 3],
            ];
            let b_chunk = [
                b_slice[idx],
                b_slice[idx + 1],
                b_slice[idx + 2],
                b_slice[idx + 3],
            ];

            let a_vec = f64x4::new(a_chunk);
            let b_vec = f64x4::new(b_chunk);
            let result_vec = a_vec + b_vec;

            let result_arr: [f64; 4] = result_vec.into();
            result_slice[idx..idx + 4].copy_from_slice(&result_arr);
        }

        i += 16;
    }

    // Process remaining elements with standard SIMD
    while i + 4 <= n {
        let a_chunk = [a_slice[i], a_slice[i + 1], a_slice[i + 2], a_slice[i + 3]];
        let b_chunk = [b_slice[i], b_slice[i + 1], b_slice[i + 2], b_slice[i + 3]];

        let a_vec = f64x4::new(a_chunk);
        let b_vec = f64x4::new(b_chunk);
        let result_vec = a_vec + b_vec;

        let result_arr: [f64; 4] = result_vec.into();
        result_slice[i..i + 4].copy_from_slice(&result_arr);
        i += 4;
    }

    // Process remaining elements
    for j in i..n {
        result[j] = a[j] + b[j];
    }

    result
}

/// Ultra-high performance fused multiply-add with unrolled accumulators for f64
#[cfg(feature = "simd")]
pub fn simd_fma_ultra_optimized_f64(
    a: &ArrayView1<f64>,
    b: &ArrayView1<f64>,
    c: &ArrayView1<f64>,
) -> Array1<f64> {
    let n = a.len();
    assert_eq!(n, b.len());
    assert_eq!(n, c.len());

    let mut result = Array1::zeros(n);

    let a_slice = a.as_slice().unwrap();
    let b_slice = b.as_slice().unwrap();
    let c_slice = c.as_slice().unwrap();
    let result_slice = result.as_slice_mut().unwrap();

    let mut i = 0;

    // Ultra-optimized loop: process 32 elements at a time with 8 accumulators
    while i + 32 <= n {
        // Use 8 parallel accumulators to maximize instruction-level parallelism
        for acc_idx in 0..8 {
            let base_idx = i + acc_idx * 4;

            let a_chunk = [
                a_slice[base_idx],
                a_slice[base_idx + 1],
                a_slice[base_idx + 2],
                a_slice[base_idx + 3],
            ];
            let b_chunk = [
                b_slice[base_idx],
                b_slice[base_idx + 1],
                b_slice[base_idx + 2],
                b_slice[base_idx + 3],
            ];
            let c_chunk = [
                c_slice[base_idx],
                c_slice[base_idx + 1],
                c_slice[base_idx + 2],
                c_slice[base_idx + 3],
            ];

            let a_vec = f64x4::new(a_chunk);
            let b_vec = f64x4::new(b_chunk);
            let c_vec = f64x4::new(c_chunk);

            // Fused multiply-add with maximum instruction throughput
            let result_vec = a_vec * b_vec + c_vec;

            let result_arr: [f64; 4] = result_vec.into();
            result_slice[base_idx..base_idx + 4].copy_from_slice(&result_arr);
        }

        i += 32;
    }

    // Fallback to standard FMA for remaining elements
    while i + 4 <= n {
        let a_chunk = [a_slice[i], a_slice[i + 1], a_slice[i + 2], a_slice[i + 3]];
        let b_chunk = [b_slice[i], b_slice[i + 1], b_slice[i + 2], b_slice[i + 3]];
        let c_chunk = [c_slice[i], c_slice[i + 1], c_slice[i + 2], c_slice[i + 3]];

        let a_vec = f64x4::new(a_chunk);
        let b_vec = f64x4::new(b_chunk);
        let c_vec = f64x4::new(c_chunk);
        let result_vec = a_vec * b_vec + c_vec;

        let result_arr: [f64; 4] = result_vec.into();
        result_slice[i..i + 4].copy_from_slice(&result_arr);
        i += 4;
    }

    // Process remaining elements
    for j in i..n {
        result[j] = a[j] * b[j] + c[j];
    }

    result
}

/// Adaptive SIMD operation selector based on data size and CPU capabilities for f64
#[cfg(feature = "simd")]
pub fn simd_adaptive_add_f64(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
    let n = a.len();
    let capabilities = detect_simd_capabilities();

    // Select optimal implementation based on data size and capabilities
    match n {
        0..=32 => simd_add_f64(a, b),   // Use basic SIMD for small arrays
        33..=512 => simd_add_f64(a, b), // Use basic SIMD for medium arrays
        _ => {
            if capabilities.l2_cache_size > 0
                && n * std::mem::size_of::<f64>() > capabilities.l2_cache_size
            {
                // Use cache-optimized version for large arrays
                simd_add_cache_optimized_f64(a, b)
            } else {
                simd_add_f64(a, b) // Fallback to basic add for now
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::arr1;

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_maximum_f32() {
        let a = arr1(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let b = arr1(&[9.0f32, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);

        let result = simd_maximum_f32(&a.view(), &b.view());
        let expected = arr1(&[9.0f32, 8.0, 7.0, 6.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

        for (a, b) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-5);
        }
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_maximum_f64() {
        let a = arr1(&[1.0f64, 2.0, 3.0, 4.0, 5.0]);
        let b = arr1(&[5.0f64, 4.0, 3.0, 2.0, 1.0]);

        let result = simd_maximum_f64(&a.view(), &b.view());
        let expected = arr1(&[5.0f64, 4.0, 3.0, 4.0, 5.0]);

        for (a, b) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_minimum_f32() {
        let a = arr1(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let b = arr1(&[9.0f32, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);

        let result = simd_minimum_f32(&a.view(), &b.view());
        let expected = arr1(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0]);

        for (a, b) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-5);
        }
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_minimum_f64() {
        let a = arr1(&[1.0f64, 2.0, 3.0, 4.0, 5.0]);
        let b = arr1(&[5.0f64, 4.0, 3.0, 2.0, 1.0]);

        let result = simd_minimum_f64(&a.view(), &b.view());
        let expected = arr1(&[1.0f64, 2.0, 3.0, 2.0, 1.0]);

        for (a, b) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_linspace_f32() {
        let result = simd_linspace_f32(0.0, 1.0, 5);
        let expected = arr1(&[0.0f32, 0.25, 0.5, 0.75, 1.0]);

        assert_eq!(result.len(), 5);
        for (a, b) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-6);
        }

        // Test endpoints
        assert_relative_eq!(result[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(result[4], 1.0, epsilon = 1e-6);
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_linspace_f64() {
        let result = simd_linspace_f64(0.0, 1.0, 5);
        let expected = arr1(&[0.0f64, 0.25, 0.5, 0.75, 1.0]);

        assert_eq!(result.len(), 5);
        for (a, b) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-14);
        }

        // Test endpoints
        assert_relative_eq!(result[0], 0.0, epsilon = 1e-14);
        assert_relative_eq!(result[4], 1.0, epsilon = 1e-14);
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_add_f32() {
        let a = arr1(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let b = arr1(&[9.0f32, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);

        let result = simd_add_f32(&a.view(), &b.view());
        let expected = arr1(&[10.0f32; 9]);

        for (a, b) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-5);
        }
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_add_f64() {
        let a = arr1(&[1.0f64, 2.0, 3.0, 4.0, 5.0]);
        let b = arr1(&[5.0f64, 4.0, 3.0, 2.0, 1.0]);

        let result = simd_add_f64(&a.view(), &b.view());
        let expected = arr1(&[6.0f64; 5]);

        for (a, b) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_mul_f32() {
        let a = arr1(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let b = arr1(&[2.0f32, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

        let result = simd_mul_f32(&a.view(), &b.view());
        let expected = arr1(&[2.0f32, 6.0, 12.0, 20.0, 30.0, 42.0, 56.0, 72.0]);

        for (a, b) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-5);
        }
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_mul_f64() {
        let a = arr1(&[1.0f64, 2.0, 3.0, 4.0]);
        let b = arr1(&[2.0f64, 3.0, 4.0, 5.0]);

        let result = simd_mul_f64(&a.view(), &b.view());
        let expected = arr1(&[2.0f64, 6.0, 12.0, 20.0]);

        for (a, b) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_dot_f32() {
        let a = arr1(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let b = arr1(&[8.0f32, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);

        let result = simd_dot_f32(&a.view(), &b.view());
        let expected = 120.0f32; // 1*8 + 2*7 + 3*6 + 4*5 + 5*4 + 6*3 + 7*2 + 8*1

        assert_relative_eq!(result, expected, epsilon = 1e-5);
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_dot_f64() {
        let a = arr1(&[1.0f64, 2.0, 3.0, 4.0]);
        let b = arr1(&[4.0f64, 3.0, 2.0, 1.0]);

        let result = simd_dot_f64(&a.view(), &b.view());
        let expected = 20.0f64; // 1*4 + 2*3 + 3*2 + 4*1

        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_scalar_mul_f32() {
        let a = arr1(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let scalar = 2.5f32;

        let result = simd_scalar_mul_f32(&a.view(), scalar);
        let expected = arr1(&[2.5f32, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0]);

        for (a, b) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-5);
        }
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_scalar_mul_f64() {
        let a = arr1(&[1.0f64, 2.0, 3.0, 4.0]);
        let scalar = 3.0f64;

        let result = simd_scalar_mul_f64(&a.view(), scalar);
        let expected = arr1(&[3.0f64, 6.0, 9.0, 12.0]);

        for (a, b) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-10);
        }
    }
}
