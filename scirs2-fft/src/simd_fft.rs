//! Minimal SIMD-accelerated FFT operations stub
//!
//! This module provides minimal stubs for SIMD-accelerated FFT operations.
//! All actual SIMD operations are delegated to scirs2-core when available.

use crate::error::FFTResult;
use crate::fft;
use ndarray::{Array2, ArrayD};
use num_complex::Complex64;
use num_traits::NumCast;
use scirs2_core::simd_ops::PlatformCapabilities;
use std::fmt::Debug;

/// Normalization mode for FFT operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormMode {
    None,
    Backward,
    Ortho,
    Forward,
}

/// Check if SIMD support is available
pub fn simd_support_available() -> bool {
    let caps = PlatformCapabilities::detect();
    caps.simd_available
}

/// Apply SIMD normalization (stub - not used in current implementation)
pub fn apply_simd_normalization(data: &mut [Complex64], scale: f64) {
    for c in data.iter_mut() {
        *c *= scale;
    }
}

/// SIMD-accelerated 1D FFT
pub fn fft_simd<T>(x: &[T], _norm: Option<&str>) -> FFTResult<Vec<Complex64>>
where
    T: NumCast + Copy + Debug,
{
    fft::fft(x, None)
}

/// SIMD-accelerated 1D inverse FFT
pub fn ifft_simd<T>(x: &[T], _norm: Option<&str>) -> FFTResult<Vec<Complex64>>
where
    T: NumCast + Copy + Debug,
{
    fft::ifft(x, None)
}

/// SIMD-accelerated 2D FFT
pub fn fft2_simd<T>(
    x: &[T],
    _shape: Option<(usize, usize)>,
    _norm: Option<&str>,
) -> FFTResult<Array2<Complex64>>
where
    T: NumCast + Copy + Debug,
{
    // For now, just create a simple error
    Err(crate::error::FFTError::NotImplementedError(
        "2D FFT from slice not yet implemented".to_string(),
    ))
}

/// SIMD-accelerated 2D inverse FFT
pub fn ifft2_simd<T>(
    x: &[T],
    _shape: Option<(usize, usize)>,
    _norm: Option<&str>,
) -> FFTResult<Array2<Complex64>>
where
    T: NumCast + Copy + Debug,
{
    // For now, just create a simple error
    Err(crate::error::FFTError::NotImplementedError(
        "2D inverse FFT from slice not yet implemented".to_string(),
    ))
}

/// SIMD-accelerated N-dimensional FFT
pub fn fftn_simd<T>(
    x: &[T],
    _shape: Option<&[usize]>,
    _axes: Option<&[usize]>,
    _norm: Option<&str>,
) -> FFTResult<ArrayD<Complex64>>
where
    T: NumCast + Copy + Debug,
{
    // For now, just create a simple error
    Err(crate::error::FFTError::NotImplementedError(
        "N-dimensional FFT from slice not yet implemented".to_string(),
    ))
}

/// SIMD-accelerated N-dimensional inverse FFT
pub fn ifftn_simd<T>(
    x: &[T],
    _shape: Option<&[usize]>,
    _axes: Option<&[usize]>,
    _norm: Option<&str>,
) -> FFTResult<ArrayD<Complex64>>
where
    T: NumCast + Copy + Debug,
{
    // For now, just create a simple error
    Err(crate::error::FFTError::NotImplementedError(
        "N-dimensional inverse FFT from slice not yet implemented".to_string(),
    ))
}

/// Adaptive FFT
pub fn fft_adaptive<T>(x: &[T], norm: Option<&str>) -> FFTResult<Vec<Complex64>>
where
    T: NumCast + Copy + Debug,
{
    fft_simd(x, norm)
}

/// Adaptive inverse FFT
pub fn ifft_adaptive<T>(x: &[T], norm: Option<&str>) -> FFTResult<Vec<Complex64>>
where
    T: NumCast + Copy + Debug,
{
    ifft_simd(x, norm)
}

/// Adaptive 2D FFT
pub fn fft2_adaptive<T>(
    x: &[T],
    shape: Option<(usize, usize)>,
    norm: Option<&str>,
) -> FFTResult<Array2<Complex64>>
where
    T: NumCast + Copy + Debug,
{
    fft2_simd(x, shape, norm)
}

/// Adaptive 2D inverse FFT
pub fn ifft2_adaptive<T>(
    x: &[T],
    shape: Option<(usize, usize)>,
    norm: Option<&str>,
) -> FFTResult<Array2<Complex64>>
where
    T: NumCast + Copy + Debug,
{
    ifft2_simd(x, shape, norm)
}

/// Adaptive N-dimensional FFT
pub fn fftn_adaptive<T>(
    x: &[T],
    shape: Option<&[usize]>,
    axes: Option<&[usize]>,
    norm: Option<&str>,
) -> FFTResult<ArrayD<Complex64>>
where
    T: NumCast + Copy + Debug,
{
    fftn_simd(x, shape, axes, norm)
}

/// Adaptive N-dimensional inverse FFT
pub fn ifftn_adaptive<T>(
    x: &[T],
    shape: Option<&[usize]>,
    axes: Option<&[usize]>,
    norm: Option<&str>,
) -> FFTResult<ArrayD<Complex64>>
where
    T: NumCast + Copy + Debug,
{
    ifftn_simd(x, shape, axes, norm)
}
