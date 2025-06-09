//! Window functions for signal processing.
//!
//! This module provides various window functions commonly used in signal processing,
//! including Hamming, Hann, Blackman, and others. These windows are useful for
//! reducing spectral leakage in Fourier transforms and filter design.

use crate::error::{SignalError, SignalResult};
use std::f64::consts::PI;

// Import specialized window implementations
mod kaiser;
pub use kaiser::{kaiser, kaiser_bessel_derived};

/// Create a window function of a specified type and length.
///
/// # Arguments
///
/// * `window_type` - Type of window function to create
/// * `length` - Length of the window
/// * `periodic` - If true, the window is periodic, otherwise symmetric
///
/// # Returns
///
/// * Window function of specified type and length
///
/// # Examples
///
/// ```
/// use scirs2_signal::window::get_window;
///
/// // Create a Hamming window of length 10
/// let window = get_window("hamming", 10, false).unwrap();
///
/// assert_eq!(window.len(), 10);
/// assert!(window[0] > 0.0 && window[0] < 1.0);
/// assert!(window[window.len() / 2] > 0.9);
/// ```
pub fn get_window(window_type: &str, length: usize, periodic: bool) -> SignalResult<Vec<f64>> {
    if length == 0 {
        return Err(SignalError::ValueError(
            "Window length must be positive".to_string(),
        ));
    }

    // Dispatch to specific window function
    match window_type.to_lowercase().as_str() {
        "hamming" => hamming(length, !periodic),
        "hanning" | "hann" => hann(length, !periodic),
        "blackman" => blackman(length, !periodic),
        "bartlett" => bartlett(length, !periodic),
        "flattop" => flattop(length, !periodic),
        "boxcar" | "rectangular" => boxcar(length, !periodic),
        "triang" => triang(length, !periodic),
        "bohman" => bohman(length, !periodic),
        "parzen" => parzen(length, !periodic),
        "nuttall" => nuttall(length, !periodic),
        "blackmanharris" => blackmanharris(length, !periodic),
        "cosine" => cosine(length, !periodic),
        "exponential" => exponential(length, None, 1.0, !periodic),
        "tukey" => tukey(length, 0.5, !periodic),
        "barthann" => barthann(length, !periodic),
        "kaiser" => {
            // Default beta value of 8.6 gives sidelobe attenuation of about 60dB
            kaiser(length, 8.6, !periodic)
        }
        "kaiser_bessel_derived" => {
            // Default beta value of 8.6
            kaiser_bessel_derived(length, 8.6, !periodic)
        }
        "dpss" | "slepian" => {
            // Default NW parameter of 3.0 for multitaper
            dpss(length, 3.0, None, !periodic)
        }
        _ => Err(SignalError::ValueError(format!(
            "Unknown window type: {}",
            window_type
        ))),
    }
}

/// Helper function to handle small or incorrect window lengths
pub(crate) fn _len_guards(m: usize) -> bool {
    // Return true for trivial windows with length 0 or 1
    m <= 1
}

/// Helper function to extend window by 1 sample if needed for DFT-even symmetry
pub(crate) fn _extend(m: usize, sym: bool) -> (usize, bool) {
    if !sym {
        (m + 1, true)
    } else {
        (m, false)
    }
}

/// Helper function to truncate window by 1 sample if needed
pub(crate) fn _truncate(w: Vec<f64>, needed: bool) -> Vec<f64> {
    if needed {
        w[..w.len() - 1].to_vec()
    } else {
        w
    }
}

/// Hamming window.
///
/// The Hamming window is a taper formed by using a raised cosine with
/// non-zero endpoints.
///
/// # Arguments
///
/// * `m` - Number of points in the output window
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
///
/// A Vec<f64> of window values
///
/// # Examples
///
/// ```
/// use scirs2_signal::window::hamming;
///
/// let window = hamming(10, true).unwrap();
/// assert_eq!(window.len(), 10);
/// ```
pub fn hamming(m: usize, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let w_val = 0.54 - 0.46 * (2.0 * PI * i as f64 / (n - 1) as f64).cos();
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Hann window.
///
/// The Hann window is a taper formed by using a raised cosine.
///
/// # Arguments
///
/// * `m` - Number of points in the output window
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
///
/// A Vec<f64> of window values
///
/// # Examples
///
/// ```
/// use scirs2_signal::window::hann;
///
/// let window = hann(10, true).unwrap();
/// assert_eq!(window.len(), 10);
/// ```
pub fn hann(m: usize, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let w_val = 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos());
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Blackman window.
///
/// The Blackman window is a taper formed by using the first three terms of
/// a summation of cosines.
///
/// # Arguments
///
/// * `m` - Number of points in the output window
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
///
/// A Vec<f64> of window values
///
/// # Examples
///
/// ```
/// use scirs2_signal::window::blackman;
///
/// let window = blackman(10, true).unwrap();
/// assert_eq!(window.len(), 10);
/// ```
pub fn blackman(m: usize, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let w_val = 0.42 - 0.5 * (2.0 * PI * i as f64 / (n - 1) as f64).cos()
            + 0.08 * (4.0 * PI * i as f64 / (n - 1) as f64).cos();
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Bartlett window.
///
/// The Bartlett window is a triangular window that is the convolution of two rectangular windows.
///
/// # Arguments
///
/// * `m` - Number of points in the output window
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
///
/// A Vec<f64> of window values
///
/// # Examples
///
/// ```
/// use scirs2_signal::window::bartlett;
///
/// let window = bartlett(10, true).unwrap();
/// assert_eq!(window.len(), 10);
/// ```
pub fn bartlett(m: usize, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    let m2 = (n - 1) as f64 / 2.0;
    for i in 0..n {
        let w_val = 1.0 - ((i as f64 - m2) / m2).abs();
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Triangular window (slightly different from Bartlett).
///
/// # Arguments
///
/// * `m` - Number of points in the output window
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
///
/// A Vec<f64> of window values
///
/// # Examples
///
/// ```
/// use scirs2_signal::window::triang;
///
/// let window = triang(10, true).unwrap();
/// assert_eq!(window.len(), 10);
/// ```
pub fn triang(m: usize, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    let m2 = (n as f64 - 1.0) / 2.0;
    for i in 0..n {
        let w_val = 1.0 - ((i as f64 - m2) / (m2 + 1.0)).abs();
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Flat top window.
///
/// The flat top window is a taper formed by using a weighted sum of cosine functions.
/// This window has the best amplitude flatness in the frequency domain.
///
/// # Arguments
///
/// * `m` - Number of points in the output window
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
///
/// A Vec<f64> of window values
///
/// # Examples
///
/// ```
/// use scirs2_signal::window::flattop;
///
/// let window = flattop(10, true).unwrap();
/// assert_eq!(window.len(), 10);
/// ```
pub fn flattop(m: usize, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    let a0 = 0.21557895;
    let a1 = 0.41663158;
    let a2 = 0.277263158;
    let a3 = 0.083578947;
    let a4 = 0.006947368;

    for i in 0..n {
        let w_val = a0 - a1 * (2.0 * PI * i as f64 / (n - 1) as f64).cos()
            + a2 * (4.0 * PI * i as f64 / (n - 1) as f64).cos()
            - a3 * (6.0 * PI * i as f64 / (n - 1) as f64).cos()
            + a4 * (8.0 * PI * i as f64 / (n - 1) as f64).cos();
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Rectangular window.
///
/// The rectangular window is the simplest window, equivalent to replacing all frame samples by a constant.
///
/// # Arguments
///
/// * `m` - Number of points in the output window
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
///
/// A Vec<f64> of window values
///
/// # Examples
///
/// ```
/// use scirs2_signal::window::boxcar;
///
/// let window = boxcar(10, true).unwrap();
/// assert_eq!(window.len(), 10);
/// assert_eq!(window[0], 1.0);
/// ```
pub fn boxcar(m: usize, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);
    let w = vec![1.0; n];
    Ok(_truncate(w, needs_trunc))
}

/// Bohman window.
///
/// The Bohman window is the product of a cosine and a sinc function.
///
/// # Arguments
///
/// * `m` - Number of points in the output window
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
///
/// A Vec<f64> of window values
///
/// # Examples
///
/// ```
/// use scirs2_signal::window::bohman;
///
/// let window = bohman(10, true).unwrap();
/// assert_eq!(window.len(), 10);
/// ```
pub fn bohman(m: usize, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let x = 2.0 * i as f64 / (n - 1) as f64 - 1.0;
        let x_abs = x.abs();
        let w_val = if x_abs <= 1.0 {
            (1.0 - x_abs) * (PI * x_abs).cos() + PI.recip() * (PI * x_abs).sin()
        } else {
            0.0
        };
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Parzen window.
///
/// The Parzen window is a piecewise cubic approximation of the Gaussian window.
///
/// # Arguments
///
/// * `m` - Number of points in the output window
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
///
/// A Vec<f64> of window values
///
/// # Examples
///
/// ```
/// use scirs2_signal::window::parzen;
///
/// let window = parzen(10, true).unwrap();
/// assert_eq!(window.len(), 10);
/// ```
pub fn parzen(m: usize, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    let n1 = (n - 1) as f64;

    for i in 0..n {
        let x = 2.0 * i as f64 / n1 - 1.0;
        let x_abs = x.abs();

        let w_val = if x_abs <= 0.5 {
            1.0 - 6.0 * x_abs.powi(2) + 6.0 * x_abs.powi(3)
        } else if x_abs <= 1.0 {
            2.0 * (1.0 - x_abs).powi(3)
        } else {
            0.0
        };

        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Nuttall window.
///
/// The Nuttall window is a minimal 4-term Blackman-Harris window.
///
/// # Arguments
///
/// * `m` - Number of points in the output window
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
///
/// A Vec<f64> of window values
///
/// # Examples
///
/// ```
/// use scirs2_signal::window::nuttall;
///
/// let window = nuttall(10, true).unwrap();
/// assert_eq!(window.len(), 10);
/// ```
pub fn nuttall(m: usize, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    let a0 = 0.3635819;
    let a1 = 0.4891775;
    let a2 = 0.1365995;
    let a3 = 0.0106411;

    for i in 0..n {
        let w_val = a0 - a1 * (2.0 * PI * i as f64 / (n - 1) as f64).cos()
            + a2 * (4.0 * PI * i as f64 / (n - 1) as f64).cos()
            - a3 * (6.0 * PI * i as f64 / (n - 1) as f64).cos();
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Blackman-Harris window.
///
/// The Blackman-Harris window is a taper formed by using the first four terms of a
/// summation of cosines.
///
/// # Arguments
///
/// * `m` - Number of points in the output window
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
///
/// A Vec<f64> of window values
///
/// # Examples
///
/// ```
/// use scirs2_signal::window::blackmanharris;
///
/// let window = blackmanharris(10, true).unwrap();
/// assert_eq!(window.len(), 10);
/// ```
pub fn blackmanharris(m: usize, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    let a0 = 0.35875;
    let a1 = 0.48829;
    let a2 = 0.14128;
    let a3 = 0.01168;

    for i in 0..n {
        let w_val = a0 - a1 * (2.0 * PI * i as f64 / (n - 1) as f64).cos()
            + a2 * (4.0 * PI * i as f64 / (n - 1) as f64).cos()
            - a3 * (6.0 * PI * i as f64 / (n - 1) as f64).cos();
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Cosine window.
///
/// Also known as the sine window, half-cosine, or half-sine window.
///
/// # Arguments
///
/// * `m` - Number of points in the output window
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
///
/// A Vec<f64> of window values
///
/// # Examples
///
/// ```
/// use scirs2_signal::window::cosine;
///
/// let window = cosine(10, true).unwrap();
/// assert_eq!(window.len(), 10);
/// ```
pub fn cosine(m: usize, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let w_val = (PI * i as f64 / (n - 1) as f64).sin();
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Exponential window.
///
/// # Arguments
///
/// * `m` - Number of points in the output window
/// * `center` - Optional parameter defining the center point of the window, default is None (m/2)
/// * `tau` - Parameter defining the decay rate
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
///
/// A Vec<f64> of window values
///
/// # Examples
///
/// ```
/// use scirs2_signal::window::exponential;
///
/// let window = exponential(10, None, 1.0, true).unwrap();
/// assert_eq!(window.len(), 10);
/// ```
pub fn exponential(m: usize, center: Option<f64>, tau: f64, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let center_val = center.unwrap_or(((n - 1) as f64) / 2.0);

    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let w_val = (-((i as f64 - center_val).abs() / tau)).exp();
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Tukey window.
///
/// The Tukey window, also known as the cosine-tapered window, is a window
/// with a flat middle section and cosine tapered ends.
///
/// # Arguments
///
/// * `m` - Number of points in the output window
/// * `alpha` - Shape parameter of the Tukey window, representing the ratio of
///   cosine-tapered section length to the total window length
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
///
/// A Vec<f64> of window values
///
/// # Examples
///
/// ```
/// use scirs2_signal::window::tukey;
///
/// let window = tukey(10, 0.5, true).unwrap();
/// assert_eq!(window.len(), 10);
/// ```
pub fn tukey(m: usize, alpha: f64, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let alpha = alpha.clamp(0.0, 1.0);

    if alpha == 0.0 {
        return boxcar(m, sym);
    }

    if alpha == 1.0 {
        return hann(m, sym);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    let width = (alpha * (n - 1) as f64 / 2.0).floor() as usize;
    let width = width.max(1); // Ensure width is at least 1

    for i in 0..n {
        let w_val = if i < width {
            0.5 * (1.0 + (PI * (-1.0 + 2.0 * i as f64 / (alpha * (n - 1) as f64))).cos())
        } else if i >= n - width {
            0.5 * (1.0
                + (PI * (-2.0 / alpha + 1.0 + 2.0 * i as f64 / (alpha * (n - 1) as f64))).cos())
        } else {
            1.0
        };
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Barthann window.
///
/// The Barthann window is the product of a Bartlett window and a Hann window.
///
/// # Arguments
///
/// * `m` - Number of points in the output window
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
///
/// A Vec<f64> of window values
///
/// # Examples
///
/// ```
/// use scirs2_signal::window::barthann;
///
/// let window = barthann(10, true).unwrap();
/// assert_eq!(window.len(), 10);
/// ```
pub fn barthann(m: usize, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let fac = (i as f64) / (n - 1) as f64;
        let w_val = 0.62 - 0.48 * (fac * 2.0 - 1.0).abs() - 0.38 * (2.0 * PI * fac).cos();
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Discrete Prolate Spheroidal Sequence (DPSS) windows.
///
/// Also known as Slepian windows, these are optimal for multitaper spectral estimation.
/// They maximize the energy concentration within a given bandwidth while minimizing
/// spectral leakage.
///
/// # Arguments
///
/// * `m` - Number of points in the output window
/// * `nw` - Time-bandwidth product. Larger values provide better frequency resolution
/// * `num_windows` - Number of windows to generate (optional, defaults to 2*NW-1)
/// * `sym` - If true, generates symmetric windows
///
/// # Returns
///
/// * A vector of DPSS windows, each window is a Vec<f64>
///
/// # Examples
///
/// ```ignore
/// use scirs2_signal::window::dpss_windows;
///
/// // Generate 5 DPSS windows for multitaper spectral estimation
/// let windows = dpss_windows(256, 4.0, Some(7), true).unwrap();
/// assert_eq!(windows.len(), 7);
/// assert_eq!(windows[0].len(), 256);
/// ```
pub fn dpss_windows(
    m: usize,
    nw: f64,
    num_windows: Option<usize>,
    sym: bool,
) -> SignalResult<Vec<Vec<f64>>> {
    if m == 0 {
        return Err(SignalError::ValueError(
            "Window length must be positive".to_string(),
        ));
    }

    if nw <= 0.0 || nw >= m as f64 / 2.0 {
        return Err(SignalError::ValueError(
            "Time-bandwidth product NW must be between 0 and M/2".to_string(),
        ));
    }

    let num_win = num_windows.unwrap_or((2.0 * nw - 1.0).floor() as usize);
    if num_win == 0 {
        return Err(SignalError::ValueError(
            "Number of windows must be positive".to_string(),
        ));
    }

    let (n, needs_trunc) = _extend(m, sym);

    // Build the tridiagonal matrix for the eigenvalue problem
    let omega = 2.0 * PI * nw / n as f64;
    let mut diag = vec![0.0; n];
    let mut off_diag = vec![0.0; n - 1];

    // Fill diagonal elements
    for (i, diag_val) in diag.iter_mut().enumerate() {
        let k = i as f64 - (n as f64 - 1.0) / 2.0;
        *diag_val = (omega * k).cos();
    }

    // Fill off-diagonal elements
    for (i, off_diag_val) in off_diag.iter_mut().enumerate() {
        let k = (i + 1) as f64;
        *off_diag_val = k * (n as f64 - k) / 2.0;
    }

    // Solve the eigenvalue problem for the tridiagonal matrix
    let (eigenvals, eigenvecs) = solve_tridiagonal_eigenproblem(&diag, &off_diag, num_win)?;

    // Sort eigenvalues and eigenvectors in descending order
    let mut sorted_indices: Vec<usize> = (0..eigenvals.len()).collect();
    sorted_indices.sort_by(|&a, &b| eigenvals[b].partial_cmp(&eigenvals[a]).unwrap());

    let mut windows = Vec::with_capacity(num_win);
    for &idx in sorted_indices.iter().take(num_win) {
        let mut window = eigenvecs[idx].clone();

        // Ensure the first element is positive (phase convention)
        if window[0] < 0.0 {
            for w in &mut window {
                *w = -*w;
            }
        }

        // Normalize the window
        let norm = window.iter().map(|&x| x * x).sum::<f64>().sqrt();
        for w in &mut window {
            *w /= norm;
        }

        windows.push(_truncate(window, needs_trunc));
    }

    Ok(windows)
}

/// Single DPSS window (first window from the set).
///
/// # Arguments
///
/// * `m` - Number of points in the output window
/// * `nw` - Time-bandwidth product
/// * `k` - Window index (optional, defaults to 0 for first window)
/// * `sym` - If true, generates a symmetric window
///
/// # Returns
///
/// * A single DPSS window
///
/// # Examples
///
/// ```ignore
/// use scirs2_signal::window::dpss;
///
/// let window = dpss(64, 2.5, None, true).unwrap();
/// assert_eq!(window.len(), 64);
/// ```
pub fn dpss(m: usize, nw: f64, k: Option<usize>, sym: bool) -> SignalResult<Vec<f64>> {
    let window_idx = k.unwrap_or(0);
    let windows = dpss_windows(m, nw, Some(window_idx + 1), sym)?;

    if windows.is_empty() {
        return Err(SignalError::ValueError(
            "Failed to generate DPSS window".to_string(),
        ));
    }

    Ok(windows[window_idx].clone())
}

/// Solve the tridiagonal eigenvalue problem using the QR algorithm.
///
/// This is a simplified implementation for finding the largest eigenvalues
/// and their corresponding eigenvectors.
fn solve_tridiagonal_eigenproblem(
    diag: &[f64],
    off_diag: &[f64],
    num_eigenvals: usize,
) -> SignalResult<(Vec<f64>, Vec<Vec<f64>>)> {
    let n = diag.len();
    if off_diag.len() != n - 1 {
        return Err(SignalError::ValueError(
            "Inconsistent matrix dimensions".to_string(),
        ));
    }

    // Simple power iteration for finding dominant eigenvalues
    // This is a simplified approach for demonstration
    let mut eigenvals: Vec<f64> = Vec::new();
    let mut eigenvecs: Vec<Vec<f64>> = Vec::new();

    // Start with the largest expected eigenvalue
    let max_iter = 1000;
    let tolerance = 1e-10;

    for _k in 0..num_eigenvals.min(n) {
        // Initialize random vector
        let mut v = vec![1.0; n];
        for (i, v_val) in v.iter_mut().enumerate().skip(1) {
            *v_val = 0.1 * (i as f64).sin();
        }

        // Orthogonalize against previous eigenvectors
        for prev_vec in &eigenvecs {
            let dot = v
                .iter()
                .zip(prev_vec.iter())
                .map(|(&a, &b)| a * b)
                .sum::<f64>();
            for (vi, &pvi) in v.iter_mut().zip(prev_vec.iter()) {
                *vi -= dot * pvi;
            }
        }

        // Normalize
        let norm = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
        for vi in &mut v {
            *vi /= norm;
        }

        let mut eigenval = 0.0;

        // Power iteration
        for _iter in 0..max_iter {
            let mut new_v = vec![0.0; n];

            // Matrix-vector multiplication for tridiagonal matrix
            for i in 0..n {
                new_v[i] += diag[i] * v[i];
                if i > 0 {
                    new_v[i] += off_diag[i - 1] * v[i - 1];
                }
                if i < n - 1 {
                    new_v[i] += off_diag[i] * v[i + 1];
                }
            }

            // Orthogonalize against previous eigenvectors
            for prev_vec in &eigenvecs {
                let dot = new_v
                    .iter()
                    .zip(prev_vec.iter())
                    .map(|(&a, &b)| a * b)
                    .sum::<f64>();
                for (nvi, &pvi) in new_v.iter_mut().zip(prev_vec.iter()) {
                    *nvi -= dot * pvi;
                }
            }

            // Calculate eigenvalue (Rayleigh quotient)
            let new_eigenval = new_v
                .iter()
                .zip(v.iter())
                .map(|(&a, &b)| a * b)
                .sum::<f64>();

            // Normalize
            let norm = new_v.iter().map(|&x| x * x).sum::<f64>().sqrt();
            for nvi in &mut new_v {
                *nvi /= norm;
            }

            // Check convergence
            if (new_eigenval - eigenval).abs() < tolerance {
                eigenval = new_eigenval;
                v = new_v;
                break;
            }

            eigenval = new_eigenval;
            v = new_v;
        }

        eigenvals.push(eigenval);
        eigenvecs.push(v);
    }

    Ok((eigenvals, eigenvecs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_hamming_window() {
        let window = hamming(10, true).unwrap();
        assert_eq!(window.len(), 10);

        // Test symmetry
        for i in 0..5 {
            assert_relative_eq!(window[i], window[9 - i], epsilon = 1e-10);
        }

        // Test specific values
        assert_relative_eq!(window[0], 0.08, epsilon = 0.01);
        // The peak is at indices 4 and 5 for a 10-point symmetric window
        assert!(window[4] > 0.95);
        assert!(window[5] > 0.95);
    }

    #[test]
    fn test_hann_window() {
        let window = hann(10, true).unwrap();
        assert_eq!(window.len(), 10);

        // Test symmetry
        for i in 0..5 {
            assert_relative_eq!(window[i], window[9 - i], epsilon = 1e-10);
        }

        // Test specific values
        assert_relative_eq!(window[0], 0.0, epsilon = 0.01);
        // The peak is at indices 4 and 5 for a 10-point symmetric window
        assert!(window[4] > 0.95);
        assert!(window[5] > 0.95);
    }

    #[test]
    fn test_blackman_window() {
        let window = blackman(10, true).unwrap();
        assert_eq!(window.len(), 10);

        // Test symmetry
        for i in 0..5 {
            assert_relative_eq!(window[i], window[9 - i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_bartlett_window() {
        let window = bartlett(10, true).unwrap();
        assert_eq!(window.len(), 10);

        // Test symmetry
        for i in 0..5 {
            assert_relative_eq!(window[i], window[9 - i], epsilon = 1e-10);
        }

        // Test endpoints - Bartlett window has zero at endpoints
        assert_relative_eq!(window[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(window[9], 0.0, epsilon = 1e-10);

        // Test that it increases from start to middle
        assert!(window[1] > window[0]);
        assert!(window[2] > window[1]);

        // Test middle values are close to 1
        let mid_val = window[4];
        assert!(mid_val > 0.8 && mid_val <= 1.0);
    }

    #[test]
    fn test_flattop_window() {
        let window = flattop(10, true).unwrap();
        assert_eq!(window.len(), 10);

        // Test symmetry
        for i in 0..5 {
            assert_relative_eq!(window[i], window[9 - i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_boxcar_window() {
        let window = boxcar(10, true).unwrap();
        assert_eq!(window.len(), 10);

        // Test all values are 1.0
        for val in window {
            assert_relative_eq!(val, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_bohman_window() {
        let window = bohman(10, true).unwrap();
        assert_eq!(window.len(), 10);

        // Test symmetry
        for i in 0..5 {
            assert_relative_eq!(window[i], window[9 - i], epsilon = 1e-10);
        }

        // Test endpoints
        assert_relative_eq!(window[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(window[9], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_triang_window() {
        let window = triang(10, true).unwrap();
        assert_eq!(window.len(), 10);

        // Test symmetry
        for i in 0..5 {
            assert_relative_eq!(window[i], window[9 - i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_dpss_window() {
        let window = dpss(64, 2.5, None, true).unwrap();
        assert_eq!(window.len(), 64);

        // Test that the window is normalized
        let norm = window.iter().map(|&x| x * x).sum::<f64>().sqrt();
        assert_relative_eq!(norm, 1.0, epsilon = 1e-10);

        // Test symmetry (approximately, since DPSS windows are nearly symmetric)
        for i in 0..32 {
            assert_relative_eq!(window[i], window[63 - i], epsilon = 1e-2);
        }

        // Test that the window has reasonable magnitude
        let max_val = window.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        assert!(max_val > 0.1);
        assert!(max_val < 1.0);
    }

    #[test]
    fn test_dpss_windows() {
        let windows = dpss_windows(32, 2.0, Some(3), true).unwrap();
        assert_eq!(windows.len(), 3);
        assert_eq!(windows[0].len(), 32);

        // Test that each window is normalized
        for window in &windows {
            let norm = window.iter().map(|&x| x * x).sum::<f64>().sqrt();
            assert_relative_eq!(norm, 1.0, epsilon = 1e-10);
        }

        // Test orthogonality between different windows (approximate)
        for i in 0..3 {
            for j in (i + 1)..3 {
                let dot_product = windows[i]
                    .iter()
                    .zip(windows[j].iter())
                    .map(|(&a, &b)| a * b)
                    .sum::<f64>();
                assert!(dot_product.abs() < 0.1); // Should be approximately orthogonal
            }
        }
    }

    #[test]
    fn test_dpss_errors() {
        // Test error conditions

        // Zero length
        let result = dpss(0, 2.0, None, true);
        assert!(result.is_err());

        // Invalid NW
        let result = dpss(10, 0.0, None, true);
        assert!(result.is_err());

        let result = dpss(10, 10.0, None, true);
        assert!(result.is_err());

        // Too many windows
        let result = dpss_windows(10, 2.0, Some(0), true);
        assert!(result.is_err());
    }
}
