//! Fourier domain filtering functions
//!
//! This module provides filters that operate in the Fourier domain,
//! which can be more efficient for certain operations, especially
//! with large kernels or when multiple filters are applied.

use ndarray::{Array, Array1, Array2, Dimension};
use num_complex::Complex64;
use num_traits::{Float, FromPrimitive, NumCast};
use std::f64::consts::PI;
use std::fmt::Debug;

use crate::error::{NdimageError, NdimageResult};
use scirs2_fft::{fft, fft2, fftfreq, ifft, ifft2, FFTError};

// Conversion from FFTError to NdimageError
impl From<FFTError> for NdimageError {
    fn from(err: FFTError) -> Self {
        NdimageError::ComputationError(format!("FFT error: {}", err))
    }
}

/// Apply a Gaussian filter in the Fourier domain
///
/// This is more efficient than spatial domain filtering for large sigma values.
///
/// # Arguments
///
/// * `input` - Input array to filter
/// * `sigma` - Standard deviation for Gaussian kernel (can be different for each axis)
/// * `truncate` - Truncate the filter at this many standard deviations (not used in Fourier domain)
///
/// # Returns
///
/// * `Result<Array<T, D>>` - Filtered array
pub fn fourier_gaussian<T, D>(
    input: &Array<T, D>,
    sigma: &[T],
    _truncate: Option<T>,
) -> NdimageResult<Array<T, D>>
where
    T: Float + FromPrimitive + NumCast + Debug + Clone + Send + Sync + 'static,
    D: Dimension,
{
    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    if sigma.len() != input.ndim() {
        return Err(NdimageError::DimensionError(format!(
            "Sigma must have same length as input dimensions (got {} expected {})",
            sigma.len(),
            input.ndim()
        )));
    }

    for &s in sigma {
        if s <= T::zero() {
            return Err(NdimageError::InvalidInput("Sigma must be positive".into()));
        }
    }

    match input.ndim() {
        1 => {
            // 1D case
            let input_1d = input
                .view()
                .into_dimensionality::<ndarray::Ix1>()
                .map_err(|_| NdimageError::DimensionError("Failed to convert to 1D".into()))?;
            let input_1d_owned = input_1d.to_owned();
            let result = fourier_gaussian_1d(&input_1d_owned, sigma[0])?;
            result
                .into_dimensionality::<D>()
                .map_err(|_| NdimageError::DimensionError("Failed to convert back".into()))
        }
        2 => {
            // 2D case
            let input_2d = input
                .view()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| NdimageError::DimensionError("Failed to convert to 2D".into()))?;
            let input_2d_owned = input_2d.to_owned();
            let result = fourier_gaussian_2d(&input_2d_owned, sigma[0], sigma[1])?;
            result
                .into_dimensionality::<D>()
                .map_err(|_| NdimageError::DimensionError("Failed to convert back".into()))
        }
        _ => Err(NdimageError::NotImplementedError(
            "Fourier Gaussian filter for dimensions > 2 not yet implemented".into(),
        )),
    }
}

/// Apply 1D Gaussian filter in Fourier domain
fn fourier_gaussian_1d<T>(input: &Array1<T>, sigma: T) -> NdimageResult<Array1<T>>
where
    T: Float + FromPrimitive + NumCast + Debug + Clone,
{
    let n = input.len();

    // Convert to f64 for FFT
    let input_f64: Vec<f64> = input
        .iter()
        .map(|&x| NumCast::from(x).unwrap_or(0.0))
        .collect();

    // Perform FFT
    let mut spectrum = fft(&input_f64, None)?;

    // Get frequencies
    let freqs = fftfreq(n, 1.0)?;

    // Apply Gaussian filter
    let sigma_f64: f64 = NumCast::from(sigma).unwrap_or(1.0);
    let two_pi = 2.0 * PI;

    for (i, freq) in freqs.iter().enumerate() {
        let factor = (-0.5 * (two_pi * freq * sigma_f64).powi(2)).exp();
        spectrum[i] *= factor;
    }

    // Inverse FFT
    let result = ifft(&spectrum, None)?;

    // Convert back to T and extract real part
    let output: Array1<T> = Array1::from_vec(
        result
            .iter()
            .map(|c| T::from(c.re).unwrap_or(T::zero()))
            .collect(),
    );

    Ok(output)
}

/// Apply 2D Gaussian filter in Fourier domain
fn fourier_gaussian_2d<T>(input: &Array2<T>, sigma_y: T, sigma_x: T) -> NdimageResult<Array2<T>>
where
    T: Float + FromPrimitive + NumCast + Debug + Clone,
{
    let (ny, nx) = input.dim();

    // Convert to f64 Array2 for FFT
    let mut input_f64 = Array2::<f64>::zeros((ny, nx));
    for i in 0..ny {
        for j in 0..nx {
            input_f64[[i, j]] = NumCast::from(input[[i, j]]).unwrap_or(0.0);
        }
    }

    // Perform 2D FFT
    let spectrum = fft2(&input_f64, None, None, None)?;

    // Get frequencies
    let freqs_y = fftfreq(ny, 1.0)?;
    let freqs_x = fftfreq(nx, 1.0)?;

    // Apply Gaussian filter
    let sigma_y_f64: f64 = NumCast::from(sigma_y).unwrap_or(1.0);
    let sigma_x_f64: f64 = NumCast::from(sigma_x).unwrap_or(1.0);
    let two_pi = 2.0 * PI;

    let mut filtered_spectrum = spectrum.clone();
    for (i, &fy) in freqs_y.iter().enumerate() {
        for (j, &fx) in freqs_x.iter().enumerate() {
            let factor_y = (-0.5 * (two_pi * fy * sigma_y_f64).powi(2)).exp();
            let factor_x = (-0.5 * (two_pi * fx * sigma_x_f64).powi(2)).exp();
            filtered_spectrum[[i, j]] *= factor_y * factor_x;
        }
    }

    // Inverse FFT
    let result = ifft2(&filtered_spectrum, None, None, None)?;

    // Convert back to T and extract real part
    let mut output = Array2::zeros((ny, nx));
    for i in 0..ny {
        for j in 0..nx {
            output[[i, j]] = T::from(result[[i, j]].re).unwrap_or(T::zero());
        }
    }

    Ok(output)
}

/// Apply a uniform (box) filter in the Fourier domain
///
/// # Arguments
///
/// * `input` - Input array to filter
/// * `size` - Size of the box filter in each dimension
///
/// # Returns
///
/// * `Result<Array<T, D>>` - Filtered array
pub fn fourier_uniform<T, D>(input: &Array<T, D>, size: &[usize]) -> NdimageResult<Array<T, D>>
where
    T: Float + FromPrimitive + NumCast + Debug + Clone + Send + Sync + 'static,
    D: Dimension,
{
    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    if size.len() != input.ndim() {
        return Err(NdimageError::DimensionError(format!(
            "Size must have same length as input dimensions (got {} expected {})",
            size.len(),
            input.ndim()
        )));
    }

    for &s in size {
        if s == 0 {
            return Err(NdimageError::InvalidInput(
                "Filter size cannot be zero".into(),
            ));
        }
    }

    match input.ndim() {
        1 => {
            let input_1d = input
                .view()
                .into_dimensionality::<ndarray::Ix1>()
                .map_err(|_| NdimageError::DimensionError("Failed to convert to 1D".into()))?;
            let input_1d_owned = input_1d.to_owned();
            let result = fourier_uniform_1d(&input_1d_owned, size[0])?;
            result
                .into_dimensionality::<D>()
                .map_err(|_| NdimageError::DimensionError("Failed to convert back".into()))
        }
        2 => {
            let input_2d = input
                .view()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| NdimageError::DimensionError("Failed to convert to 2D".into()))?;
            let input_2d_owned = input_2d.to_owned();
            let result = fourier_uniform_2d(&input_2d_owned, size[0], size[1])?;
            result
                .into_dimensionality::<D>()
                .map_err(|_| NdimageError::DimensionError("Failed to convert back".into()))
        }
        _ => Err(NdimageError::NotImplementedError(
            "Fourier uniform filter for dimensions > 2 not yet implemented".into(),
        )),
    }
}

/// Apply 1D uniform filter in Fourier domain
fn fourier_uniform_1d<T>(input: &Array1<T>, size: usize) -> NdimageResult<Array1<T>>
where
    T: Float + FromPrimitive + NumCast + Debug + Clone,
{
    let n = input.len();

    // Convert to f64 for FFT
    let input_f64: Vec<f64> = input
        .iter()
        .map(|&x| NumCast::from(x).unwrap_or(0.0))
        .collect();

    // Perform FFT
    let mut spectrum = fft(&input_f64, None)?;

    // Get frequencies
    let freqs = fftfreq(n, 1.0)?;

    // Apply sinc filter (Fourier transform of box function)
    for (i, freq) in freqs.iter().enumerate() {
        let x = size as f64 * freq * 2.0 * PI;
        let sinc = if x.abs() < 1e-10 { 1.0 } else { x.sin() / x };
        spectrum[i] *= sinc;
    }

    // Inverse FFT
    let result = ifft(&spectrum, None)?;

    // Convert back to T and extract real part
    let output: Array1<T> = Array1::from_vec(
        result
            .iter()
            .map(|c| T::from(c.re).unwrap_or(T::zero()))
            .collect(),
    );

    Ok(output)
}

/// Apply 2D uniform filter in Fourier domain
fn fourier_uniform_2d<T>(input: &Array2<T>, size_y: usize, size_x: usize) -> NdimageResult<Array2<T>>
where
    T: Float + FromPrimitive + NumCast + Debug + Clone,
{
    let (ny, nx) = input.dim();

    // Convert to f64 Array2 for FFT
    let mut input_f64 = Array2::<f64>::zeros((ny, nx));
    for i in 0..ny {
        for j in 0..nx {
            input_f64[[i, j]] = NumCast::from(input[[i, j]]).unwrap_or(0.0);
        }
    }

    // Perform 2D FFT
    let spectrum = fft2(&input_f64, None, None, None)?;

    // Get frequencies
    let freqs_y = fftfreq(ny, 1.0)?;
    let freqs_x = fftfreq(nx, 1.0)?;

    // Apply sinc filter
    let mut filtered_spectrum = spectrum.clone();
    for (i, &fy) in freqs_y.iter().enumerate() {
        for (j, &fx) in freqs_x.iter().enumerate() {
            let x_y = size_y as f64 * fy * 2.0 * PI;
            let x_x = size_x as f64 * fx * 2.0 * PI;

            let sinc_y = if x_y.abs() < 1e-10 {
                1.0
            } else {
                x_y.sin() / x_y
            };
            let sinc_x = if x_x.abs() < 1e-10 {
                1.0
            } else {
                x_x.sin() / x_x
            };

            filtered_spectrum[[i, j]] *= sinc_y * sinc_x;
        }
    }

    // Inverse FFT
    let result = ifft2(&filtered_spectrum, None, None, None)?;

    // Convert back to T and extract real part
    let mut output = Array2::zeros((ny, nx));
    for i in 0..ny {
        for j in 0..nx {
            output[[i, j]] = T::from(result[[i, j]].re).unwrap_or(T::zero());
        }
    }

    Ok(output)
}

/// Apply an ellipsoid filter in the Fourier domain
///
/// This creates a filter that passes frequencies within an ellipsoid in frequency space.
///
/// # Arguments
///
/// * `input` - Input array to filter
/// * `size` - Semi-axes lengths of the ellipsoid in each dimension
/// * `mode` - 'lowpass' or 'highpass'
///
/// # Returns
///
/// * `Result<Array<T, D>>` - Filtered array
pub fn fourier_ellipsoid<T, D>(input: &Array<T, D>, size: &[T], mode: &str) -> NdimageResult<Array<T, D>>
where
    T: Float + FromPrimitive + NumCast + Debug + Clone + Send + Sync + 'static,
    D: Dimension,
{
    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    if size.len() != input.ndim() {
        return Err(NdimageError::DimensionError(format!(
            "Size must have same length as input dimensions (got {} expected {})",
            size.len(),
            input.ndim()
        )));
    }

    for &s in size {
        if s <= T::zero() {
            return Err(NdimageError::InvalidInput(
                "Ellipsoid size must be positive".into(),
            ));
        }
    }

    let is_lowpass = match mode {
        "lowpass" => true,
        "highpass" => false,
        _ => {
            return Err(NdimageError::InvalidInput(
                "Mode must be 'lowpass' or 'highpass'".into(),
            ))
        }
    };

    if input.ndim() == 2 {
        let input_2d = input
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|_| NdimageError::DimensionError("Failed to convert to 2D".into()))?;
        let input_2d_owned = input_2d.to_owned();
        let result = fourier_ellipsoid_2d(&input_2d_owned, size[0], size[1], is_lowpass)?;
        result
            .into_dimensionality::<D>()
            .map_err(|_| NdimageError::DimensionError("Failed to convert back".into()))
    } else {
        Err(NdimageError::NotImplementedError(
            "Fourier ellipsoid filter currently only implemented for 2D arrays".into(),
        ))
    }
}

/// Apply 2D ellipsoid filter in Fourier domain
fn fourier_ellipsoid_2d<T>(
    input: &Array2<T>,
    size_y: T,
    size_x: T,
    is_lowpass: bool,
) -> NdimageResult<Array2<T>>
where
    T: Float + FromPrimitive + NumCast + Debug + Clone,
{
    let (ny, nx) = input.dim();

    // Convert to f64 Array2 for FFT
    let mut input_f64 = Array2::<f64>::zeros((ny, nx));
    for i in 0..ny {
        for j in 0..nx {
            input_f64[[i, j]] = NumCast::from(input[[i, j]]).unwrap_or(0.0);
        }
    }

    // Perform 2D FFT
    let spectrum = fft2(&input_f64, None, None, None)?;

    // Get frequencies
    let freqs_y = fftfreq(ny, 1.0)?;
    let freqs_x = fftfreq(nx, 1.0)?;

    // Convert sizes to f64
    let size_y_f64: f64 = NumCast::from(size_y).unwrap_or(1.0);
    let size_x_f64: f64 = NumCast::from(size_x).unwrap_or(1.0);

    // Apply ellipsoid filter
    let mut filtered_spectrum = spectrum.clone();
    for (i, &fy) in freqs_y.iter().enumerate() {
        for (j, &fx) in freqs_x.iter().enumerate() {
            let dist_sq = (fy / size_y_f64).powi(2) + (fx / size_x_f64).powi(2);

            let mask = if is_lowpass {
                if dist_sq <= 1.0 {
                    1.0
                } else {
                    0.0
                }
            } else {
                if dist_sq > 1.0 {
                    1.0
                } else {
                    0.0
                }
            };

            filtered_spectrum[[i, j]] *= mask;
        }
    }

    // Inverse FFT
    let result = ifft2(&filtered_spectrum, None, None, None)?;

    // Convert back to T and extract real part
    let mut output = Array2::zeros((ny, nx));
    for i in 0..ny {
        for j in 0..nx {
            output[[i, j]] = T::from(result[[i, j]].re).unwrap_or(T::zero());
        }
    }

    Ok(output)
}

/// Shift an array in the Fourier domain
///
/// This is equivalent to rolling the array but can be more efficient
/// for non-integer shifts or when combined with other Fourier operations.
///
/// # Arguments
///
/// * `input` - Input array to shift
/// * `shift` - Shift amount for each axis (can be fractional)
///
/// # Returns
///
/// * `Result<Array<T, D>>` - Shifted array
pub fn fourier_shift<T, D>(input: &Array<T, D>, shift: &[T]) -> NdimageResult<Array<T, D>>
where
    T: Float + FromPrimitive + NumCast + Debug + Clone + Send + Sync + 'static,
    D: Dimension,
{
    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    if shift.len() != input.ndim() {
        return Err(NdimageError::DimensionError(format!(
            "Shift must have same length as input dimensions (got {} expected {})",
            shift.len(),
            input.ndim()
        )));
    }

    match input.ndim() {
        1 => {
            let input_1d = input
                .view()
                .into_dimensionality::<ndarray::Ix1>()
                .map_err(|_| NdimageError::DimensionError("Failed to convert to 1D".into()))?;
            let input_1d_owned = input_1d.to_owned();
            let result = fourier_shift_1d(&input_1d_owned, shift[0])?;
            result
                .into_dimensionality::<D>()
                .map_err(|_| NdimageError::DimensionError("Failed to convert back".into()))
        }
        2 => {
            let input_2d = input
                .view()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| NdimageError::DimensionError("Failed to convert to 2D".into()))?;
            let input_2d_owned = input_2d.to_owned();
            let result = fourier_shift_2d(&input_2d_owned, shift[0], shift[1])?;
            result
                .into_dimensionality::<D>()
                .map_err(|_| NdimageError::DimensionError("Failed to convert back".into()))
        }
        _ => Err(NdimageError::NotImplementedError(
            "Fourier shift for dimensions > 2 not yet implemented".into(),
        )),
    }
}

/// Apply 1D shift in Fourier domain
fn fourier_shift_1d<T>(input: &Array1<T>, shift: T) -> NdimageResult<Array1<T>>
where
    T: Float + FromPrimitive + NumCast + Debug + Clone,
{
    let n = input.len();

    // Convert to f64 for FFT
    let input_f64: Vec<f64> = input
        .iter()
        .map(|&x| NumCast::from(x).unwrap_or(0.0))
        .collect();

    // Perform FFT
    let mut spectrum = fft(&input_f64, None)?;

    // Get frequencies
    let freqs = fftfreq(n, 1.0)?;

    // Apply phase shift
    let shift_f64: f64 = NumCast::from(shift).unwrap_or(0.0);
    let two_pi = 2.0 * PI;

    for (i, freq) in freqs.iter().enumerate() {
        let phase = -two_pi * freq * shift_f64;
        let shift_factor = Complex64::new(phase.cos(), phase.sin());
        spectrum[i] *= shift_factor;
    }

    // Inverse FFT
    let result = ifft(&spectrum, None)?;

    // Convert back to T and extract real part
    let output: Array1<T> = Array1::from_vec(
        result
            .iter()
            .map(|c| T::from(c.re).unwrap_or(T::zero()))
            .collect(),
    );

    Ok(output)
}

/// Apply 2D shift in Fourier domain
fn fourier_shift_2d<T>(input: &Array2<T>, shift_y: T, shift_x: T) -> NdimageResult<Array2<T>>
where
    T: Float + FromPrimitive + NumCast + Debug + Clone,
{
    let (ny, nx) = input.dim();

    // Convert to f64 Array2 for FFT
    let mut input_f64 = Array2::<f64>::zeros((ny, nx));
    for i in 0..ny {
        for j in 0..nx {
            input_f64[[i, j]] = NumCast::from(input[[i, j]]).unwrap_or(0.0);
        }
    }

    // Perform 2D FFT
    let spectrum = fft2(&input_f64, None, None, None)?;

    // Get frequencies
    let freqs_y = fftfreq(ny, 1.0)?;
    let freqs_x = fftfreq(nx, 1.0)?;

    // Convert shifts to f64
    let shift_y_f64: f64 = NumCast::from(shift_y).unwrap_or(0.0);
    let shift_x_f64: f64 = NumCast::from(shift_x).unwrap_or(0.0);
    let two_pi = 2.0 * PI;

    // Apply phase shift
    let mut shifted_spectrum = spectrum.clone();
    for (i, &fy) in freqs_y.iter().enumerate() {
        for (j, &fx) in freqs_x.iter().enumerate() {
            let phase = -two_pi * (fy * shift_y_f64 + fx * shift_x_f64);
            let shift_factor = Complex64::new(phase.cos(), phase.sin());
            shifted_spectrum[[i, j]] *= shift_factor;
        }
    }

    // Inverse FFT
    let result = ifft2(&shifted_spectrum, None, None, None)?;

    // Convert back to T and extract real part
    let mut output = Array2::zeros((ny, nx));
    for i in 0..ny {
        for j in 0..nx {
            output[[i, j]] = T::from(result[[i, j]].re).unwrap_or(T::zero());
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{arr1, arr2, Array1, Array2};

    #[test]
    fn test_fourier_gaussian_1d() {
        // Create a simple 1D signal with a spike
        let mut signal = Array1::zeros(64);
        signal[32] = 1.0;

        // Apply Fourier Gaussian filter
        let sigma = vec![2.0];
        let filtered = fourier_gaussian(&signal, &sigma, None).unwrap();

        // Check that the spike is smoothed
        assert!(filtered[32] < 1.0);
        assert!(filtered[32] > 0.0);

        // Check symmetry
        for i in 1..10 {
            assert_abs_diff_eq!(filtered[32 - i], filtered[32 + i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_fourier_gaussian_2d() {
        // Create a 2D array with a centered spike
        let mut image = Array2::zeros((32, 32));
        image[[16, 16]] = 1.0;

        // Apply Fourier Gaussian filter
        let sigma = vec![1.5, 1.5];
        let filtered = fourier_gaussian(&image, &sigma, None).unwrap();

        // Check that the spike is smoothed
        assert!(filtered[[16, 16]] < 1.0);
        assert!(filtered[[16, 16]] > 0.0);

        // Check that surrounding pixels have non-zero values
        assert!(filtered[[15, 16]] > 0.0);
        assert!(filtered[[17, 16]] > 0.0);
        assert!(filtered[[16, 15]] > 0.0);
        assert!(filtered[[16, 17]] > 0.0);
    }

    #[test]
    fn test_fourier_uniform_1d() {
        // Create a step function
        let mut signal = Array1::zeros(64);
        for i in 30..35 {
            signal[i] = 1.0;
        }

        // Apply Fourier uniform filter
        let size = vec![5];
        let filtered = fourier_uniform(&signal, &size).unwrap();

        // The filter should smooth the edges
        assert!(filtered[29] > 0.0);
        assert!(filtered[35] > 0.0);
        assert!(filtered[32] > 0.0);
    }

    #[test]
    fn test_fourier_ellipsoid_lowpass() {
        // Create a 2D array with high and low frequency components
        let mut image = Array2::zeros((32, 32));

        // Add a low frequency component (large scale structure)
        for i in 0..32 {
            for j in 0..32 {
                image[[i, j]] =
                    ((i as f64 / 32.0 * 2.0 * PI).sin() + (j as f64 / 32.0 * 2.0 * PI).sin()) / 2.0;
            }
        }

        // Add high frequency noise
        for i in 0..32 {
            for j in 0..32 {
                if (i + j) % 2 == 0 {
                    image[[i, j]] += 0.5;
                }
            }
        }

        // Apply lowpass filter
        let size = vec![0.2, 0.2];
        let filtered = fourier_ellipsoid(&image, &size, "lowpass").unwrap();

        // The high frequency checkerboard pattern should be reduced
        let noise_original = (image[[0, 0]] - image[[0, 1]]).abs();
        let noise_filtered = (filtered[[0, 0]] - filtered[[0, 1]]).abs();
        assert!(noise_filtered < noise_original);
    }

    #[test]
    fn test_fourier_shift_1d() {
        // Create a simple signal
        let signal = arr1(&[1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]);

        // Shift by 2 positions
        let shift = vec![2.0];
        let shifted = fourier_shift(&signal, &shift).unwrap();

        // Check that values are shifted (with some tolerance for numerical errors)
        assert_abs_diff_eq!(shifted[2], 1.0, epsilon = 0.1);
        assert_abs_diff_eq!(shifted[3], 2.0, epsilon = 0.1);
        assert_abs_diff_eq!(shifted[4], 3.0, epsilon = 0.1);
        assert_abs_diff_eq!(shifted[5], 4.0, epsilon = 0.1);
    }

    #[test]
    fn test_fourier_shift_fractional() {
        // Create a smooth signal
        let mut signal = Array1::zeros(64);
        for i in 20..40 {
            signal[i] = 1.0;
        }

        // Shift by a fractional amount
        let shift = vec![0.5];
        let shifted = fourier_shift(&signal, &shift).unwrap();

        // The edges should be smoothed due to fractional shift
        assert!(shifted[19] > 0.0);
        assert!(shifted[19] < 0.5);
        assert!(shifted[40] > 0.0);
        assert!(shifted[40] < 0.5);
    }
}
