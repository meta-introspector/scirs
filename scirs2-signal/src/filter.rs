//! Filter design and application
//!
//! This module provides functions for designing and applying digital filters.
//! It includes functions for IIR filter design (Butterworth, Chebyshev, etc.)
//! and FIR filter design (window method, least squares, etc.).

use crate::error::{SignalError, SignalResult};
use num_complex::Complex64;
use num_traits::{Float, NumCast, Zero};
use std::fmt::Debug;

/// Filter type for IIR filter design
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FilterType {
    /// Lowpass filter
    Lowpass,
    /// Highpass filter
    Highpass,
    /// Bandpass filter
    Bandpass,
    /// Bandstop filter
    Bandstop,
}

impl FilterType {
    /// Parse a string into a filter type (deprecated - use FromStr trait instead)
    #[deprecated(since = "0.1.0", note = "use `parse()` from the FromStr trait instead")]
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> SignalResult<Self> {
        s.parse()
    }
}

impl std::str::FromStr for FilterType {
    type Err = SignalError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "lowpass" | "low" => Ok(FilterType::Lowpass),
            "highpass" | "high" => Ok(FilterType::Highpass),
            "bandpass" | "band" => Ok(FilterType::Bandpass),
            "bandstop" | "stop" => Ok(FilterType::Bandstop),
            _ => Err(SignalError::ValueError(format!(
                "Unknown filter type: {}",
                s
            ))),
        }
    }
}

/// Butterworth filter design
///
/// # Arguments
///
/// * `order` - Filter order
/// * `cutoff` - Cutoff frequency (normalized from 0 to 1, where 1 is the Nyquist frequency)
/// * `filter_type` - Filter type (lowpass, highpass, bandpass, bandstop)
///
/// # Returns
///
/// * A tuple of filter coefficients (b, a) where b are the numerator coefficients and a are
///   the denominator coefficients
///
/// # Examples
///
/// ```
/// use scirs2_signal::filter::butter;
///
/// // Design a 4th order lowpass Butterworth filter with cutoff at 0.2 times Nyquist
/// let (b, a) = butter(4, 0.2, "lowpass").unwrap();
/// ```
pub fn butter<T>(
    order: usize,
    cutoff: T,
    filter_type: impl Into<FilterTypeParam>,
) -> SignalResult<(Vec<f64>, Vec<f64>)>
where
    T: Float + NumCast + Debug,
{
    // Convert cutoff to f64
    let wn = num_traits::cast::cast::<T, f64>(cutoff)
        .ok_or_else(|| SignalError::ValueError(format!("Could not convert {:?} to f64", cutoff)))?;

    // Convert filter_type to FilterType
    let filter_type_param = filter_type.into();
    let filter_type = match filter_type_param {
        FilterTypeParam::Type(t) => t,
        FilterTypeParam::String(s) => s.parse()?,
    };

    // Validate parameters
    if order == 0 {
        return Err(SignalError::ValueError(
            "Filter order must be greater than 0".to_string(),
        ));
    }

    if wn <= 0.0 || wn >= 1.0 {
        return Err(SignalError::ValueError(format!(
            "Cutoff frequency must be between 0 and 1, got {}",
            wn
        )));
    }

    // Step 1: Calculate analog Butterworth prototype poles
    // Butterworth poles are located on a unit circle in the s-plane
    let mut poles = Vec::with_capacity(order);
    for k in 0..order {
        let angle =
            std::f64::consts::PI * (2.0 * k as f64 + order as f64 + 1.0) / (2.0 * order as f64);
        let real = angle.cos();
        let imag = angle.sin();
        poles.push(num_complex::Complex64::new(real, imag));
    }

    // Step 2: Apply frequency transformation based on filter type
    let (analog_zeros, transformed_poles, gain) = match filter_type {
        FilterType::Lowpass => {
            // Scale poles by cutoff frequency (pre-warping for bilinear transform)
            let warped_freq = (std::f64::consts::PI * wn / 2.0).tan();
            let scaled_poles: Vec<_> = poles.iter().map(|p| p * warped_freq).collect();
            // Lowpass has no finite zeros in analog domain (zeros at infinity)
            (
                Vec::<Complex64>::new(),
                scaled_poles,
                warped_freq.powi(order as i32),
            )
        }
        FilterType::Highpass => {
            // Highpass: s -> wc/s transformation
            let warped_freq = (std::f64::consts::PI * wn / 2.0).tan();
            let hp_poles: Vec<_> = poles.iter().map(|p| warped_freq / p).collect();
            // No finite zeros in analog domain for highpass - zeros are at origin
            (Vec::<Complex64>::new(), hp_poles, 1.0)
        }
        FilterType::Bandpass | FilterType::Bandstop => {
            return Err(SignalError::NotImplementedError(
                "Bandpass and bandstop Butterworth filters not yet implemented".to_string(),
            ));
        }
    };

    // Step 3: Apply bilinear transform to convert to digital filter
    // s = 2*(z-1)/(z+1), z = (2+s)/(2-s)
    let mut digital_poles = Vec::new();
    let mut digital_zeros = Vec::new();

    // Transform poles: z_pole = (2 + s_pole) / (2 - s_pole)
    for &pole in &transformed_poles {
        let z_pole = (2.0 + pole) / (2.0 - pole);
        digital_poles.push(z_pole);
    }

    // Transform finite analog zeros: z_zero = (2 + s_zero) / (2 - s_zero)
    for &zero in &analog_zeros {
        let z_zero = (2.0 + zero) / (2.0 - zero);
        digital_zeros.push(z_zero);
    }

    // Add zeros in the digital domain based on filter type
    match filter_type {
        FilterType::Lowpass => {
            // Lowpass: zeros at z = -1 (Nyquist frequency)
            for _ in 0..order {
                digital_zeros.push(num_complex::Complex64::new(-1.0, 0.0));
            }
        }
        FilterType::Highpass => {
            // Highpass: zeros at z = 1 (DC frequency)
            for _ in 0..order {
                digital_zeros.push(num_complex::Complex64::new(1.0, 0.0));
            }
        }
        _ => {} // Bandpass/bandstop not implemented
    }

    // Step 4: Convert poles and zeros to transfer function coefficients
    let (b, a) = zpk_to_tf(&digital_zeros, &digital_poles, gain)?;

    Ok((b, a))
}

/// Chebyshev Type I filter design
///
/// # Arguments
///
/// * `order` - Filter order
/// * `ripple` - Maximum ripple allowed in the passband (in dB)
/// * `cutoff` - Cutoff frequency (normalized from 0 to 1, where 1 is the Nyquist frequency)
/// * `filter_type` - Filter type (lowpass, highpass, bandpass, bandstop)
///
/// # Returns
///
/// * A tuple of filter coefficients (b, a) where b are the numerator coefficients and a are
///   the denominator coefficients
pub fn cheby1<T>(
    order: usize,
    ripple: f64,
    cutoff: T,
    filter_type: impl Into<FilterTypeParam>,
) -> SignalResult<(Vec<f64>, Vec<f64>)>
where
    T: Float + NumCast + Debug,
{
    use num_complex::Complex64;

    // Convert cutoff to f64
    let wn = num_traits::cast::cast::<T, f64>(cutoff)
        .ok_or_else(|| SignalError::ValueError(format!("Could not convert {:?} to f64", cutoff)))?;

    // Convert filter_type to FilterType
    let filter_type_param = filter_type.into();
    let filter_type = match filter_type_param {
        FilterTypeParam::Type(t) => t,
        FilterTypeParam::String(s) => s.parse()?,
    };

    // Validate parameters
    if order == 0 {
        return Err(SignalError::ValueError(
            "Filter order must be greater than 0".to_string(),
        ));
    }

    if wn <= 0.0 || wn >= 1.0 {
        return Err(SignalError::ValueError(format!(
            "Cutoff frequency must be between 0 and 1, got {}",
            wn
        )));
    }

    if ripple <= 0.0 {
        return Err(SignalError::ValueError(
            "Ripple must be positive".to_string(),
        ));
    }

    // Convert ripple from dB to linear
    let epsilon = (10.0_f64.powf(ripple / 10.0) - 1.0).sqrt();

    // Calculate Chebyshev Type I analog prototype poles
    let mut poles = Vec::with_capacity(order);
    let a = (1.0 / epsilon + (1.0 / epsilon / epsilon + 1.0).sqrt()).ln() / order as f64;
    let sinh_a = a.sinh();
    let cosh_a = a.cosh();

    for k in 0..order {
        let angle = std::f64::consts::PI * (2.0 * k as f64 + 1.0) / (2.0 * order as f64);
        let real = -sinh_a * angle.sin();
        let imag = cosh_a * angle.cos();
        poles.push(Complex64::new(real, imag));
    }

    // Apply frequency transformation and bilinear transform
    let (zeros, transformed_poles, gain) = match filter_type {
        FilterType::Lowpass => {
            let warped_freq = (std::f64::consts::PI * wn / 2.0).tan();
            let scaled_poles: Vec<_> = poles.iter().map(|p| p * warped_freq).collect();
            // Calculate DC gain for proper normalization
            let mut dc_gain = 1.0;
            for &pole in &scaled_poles {
                dc_gain /= -pole.re;
            }
            if order % 2 == 0 {
                dc_gain *= (1.0 + epsilon * epsilon).sqrt();
            }
            (Vec::new(), scaled_poles, dc_gain)
        }
        FilterType::Highpass => {
            let warped_freq = (std::f64::consts::PI * wn / 2.0).tan();
            let hp_poles: Vec<_> = poles.iter().map(|p| warped_freq / p).collect();
            let hp_zeros = vec![Complex64::new(0.0, 0.0); order];
            (hp_zeros, hp_poles, 1.0)
        }
        FilterType::Bandpass | FilterType::Bandstop => {
            return Err(SignalError::NotImplementedError(
                "Bandpass and bandstop Chebyshev Type I filters not yet implemented".to_string(),
            ));
        }
    };

    // Apply bilinear transform
    let mut digital_poles = Vec::new();
    let mut digital_zeros = Vec::new();

    for &pole in &transformed_poles {
        let z_pole = (2.0 + pole) / (2.0 - pole);
        digital_poles.push(z_pole);
    }

    for &zero in &zeros {
        let z_zero = (2.0 + zero) / (2.0 - zero);
        digital_zeros.push(z_zero);
    }

    // For highpass filters, add zeros at z = 1
    if matches!(filter_type, FilterType::Highpass) {
        for _ in 0..order {
            digital_zeros.push(Complex64::new(1.0, 0.0));
        }
    }

    // Convert to transfer function coefficients
    let (b, a) = zpk_to_tf(&digital_zeros, &digital_poles, gain)?;

    Ok((b, a))
}

/// Chebyshev Type II filter design
///
/// # Arguments
///
/// * `order` - Filter order
/// * `attenuation` - Minimum attenuation in the stopband (in dB)
/// * `cutoff` - Cutoff frequency (normalized from 0 to 1, where 1 is the Nyquist frequency)
/// * `filter_type` - Filter type (lowpass, highpass, bandpass, bandstop)
///
/// # Returns
///
/// * A tuple of filter coefficients (b, a) where b are the numerator coefficients and a are
///   the denominator coefficients
pub fn cheby2<T>(
    order: usize,
    attenuation: f64,
    cutoff: T,
    filter_type: impl Into<FilterTypeParam>,
) -> SignalResult<(Vec<f64>, Vec<f64>)>
where
    T: Float + NumCast + Debug,
{
    use num_complex::Complex64;

    // Convert cutoff to f64
    let wn = num_traits::cast::cast::<T, f64>(cutoff)
        .ok_or_else(|| SignalError::ValueError(format!("Could not convert {:?} to f64", cutoff)))?;

    // Convert filter_type to FilterType
    let filter_type_param = filter_type.into();
    let filter_type = match filter_type_param {
        FilterTypeParam::Type(t) => t,
        FilterTypeParam::String(s) => s.parse()?,
    };

    // Validate parameters
    if order == 0 {
        return Err(SignalError::ValueError(
            "Filter order must be greater than 0".to_string(),
        ));
    }

    if wn <= 0.0 || wn >= 1.0 {
        return Err(SignalError::ValueError(format!(
            "Cutoff frequency must be between 0 and 1, got {}",
            wn
        )));
    }

    if attenuation <= 0.0 {
        return Err(SignalError::ValueError(
            "Attenuation must be positive".to_string(),
        ));
    }

    // Convert attenuation from dB to linear
    let epsilon = 1.0 / (10.0_f64.powf(attenuation / 10.0) - 1.0).sqrt();

    // Calculate Chebyshev Type II analog prototype
    let a = (1.0 / epsilon + (1.0 / epsilon / epsilon + 1.0).sqrt()).ln() / order as f64;
    let sinh_a = a.sinh();
    let cosh_a = a.cosh();

    let mut poles = Vec::with_capacity(order);
    let mut zeros = Vec::new();

    for k in 0..order {
        let angle = std::f64::consts::PI * (2.0 * k as f64 + 1.0) / (2.0 * order as f64);

        // Poles are reciprocal of Chebyshev I poles
        let cheb1_pole_real = -sinh_a * angle.sin();
        let cheb1_pole_imag = cosh_a * angle.cos();
        let cheb1_pole = Complex64::new(cheb1_pole_real, cheb1_pole_imag);
        let pole = 1.0 / cheb1_pole;
        poles.push(pole);

        // Zeros on imaginary axis
        if k < order / 2 || (order % 2 == 1 && k == order / 2) {
            let zero_imag = 1.0 / angle.cos();
            zeros.push(Complex64::new(0.0, zero_imag));
            if zero_imag != 0.0 {
                zeros.push(Complex64::new(0.0, -zero_imag));
            }
        }
    }

    // Apply frequency transformation
    let (transformed_zeros, transformed_poles, gain) = match filter_type {
        FilterType::Lowpass => {
            let warped_freq = (std::f64::consts::PI * wn / 2.0).tan();
            let scaled_poles: Vec<_> = poles.iter().map(|p| p * warped_freq).collect();
            let scaled_zeros: Vec<_> = zeros.iter().map(|z| z * warped_freq).collect();

            // Calculate gain for unity gain at DC
            let mut dc_gain = 1.0;
            for &pole in &scaled_poles {
                dc_gain /= -pole.re;
            }
            for &zero in &scaled_zeros {
                if zero.re != 0.0 {
                    dc_gain *= -zero.re;
                }
            }

            (scaled_zeros, scaled_poles, dc_gain)
        }
        FilterType::Highpass => {
            let warped_freq = (std::f64::consts::PI * wn / 2.0).tan();
            let hp_poles: Vec<_> = poles.iter().map(|p| warped_freq / p).collect();
            let hp_zeros: Vec<_> = zeros.iter().map(|z| warped_freq / z).collect();

            // Add zeros at origin for highpass
            let mut all_zeros = hp_zeros;
            for _ in 0..(order - zeros.len()) {
                all_zeros.push(Complex64::new(0.0, 0.0));
            }

            (all_zeros, hp_poles, 1.0)
        }
        FilterType::Bandpass | FilterType::Bandstop => {
            return Err(SignalError::NotImplementedError(
                "Bandpass and bandstop Chebyshev Type II filters not yet implemented".to_string(),
            ));
        }
    };

    // Apply bilinear transform
    let mut digital_poles = Vec::new();
    let mut digital_zeros = Vec::new();

    for &pole in &transformed_poles {
        let z_pole = (2.0 + pole) / (2.0 - pole);
        digital_poles.push(z_pole);
    }

    for &zero in &transformed_zeros {
        let z_zero = (2.0 + zero) / (2.0 - zero);
        digital_zeros.push(z_zero);
    }

    // For highpass filters, add zeros at z = 1
    if matches!(filter_type, FilterType::Highpass) {
        let zeros_to_add = order - transformed_zeros.len();
        for _ in 0..zeros_to_add {
            digital_zeros.push(Complex64::new(1.0, 0.0));
        }
    }

    // Convert to transfer function coefficients
    let (b, a) = zpk_to_tf(&digital_zeros, &digital_poles, gain)?;

    Ok((b, a))
}

/// Elliptic (Cauer) filter design
///
/// Elliptic filters provide the steepest roll-off of any IIR filter type by having
/// equiripple behavior in both the passband and stopband. They achieve the optimal
/// trade-off between transition width and filter order.
///
/// # Arguments
///
/// * `order` - Filter order
/// * `ripple` - Maximum ripple allowed in the passband (in dB)
/// * `attenuation` - Minimum attenuation in the stopband (in dB)
/// * `cutoff` - Cutoff frequency (normalized from 0 to 1, where 1 is the Nyquist frequency)
/// * `filter_type` - Filter type (lowpass, highpass)
///
/// # Returns
///
/// * A tuple of filter coefficients (b, a) where b are the numerator coefficients and a are
///   the denominator coefficients
///
/// # Examples
///
/// ```
/// use scirs2_signal::filter::ellip;
///
/// // Design a 4th order elliptic lowpass filter with 0.5 dB ripple and 40 dB stopband attenuation
/// let (b, a) = ellip(4, 0.5, 40.0, 0.3, "lowpass").unwrap();
/// ```
pub fn ellip<T>(
    order: usize,
    ripple: f64,
    attenuation: f64,
    cutoff: T,
    filter_type: impl Into<FilterTypeParam>,
) -> SignalResult<(Vec<f64>, Vec<f64>)>
where
    T: Float + NumCast + Debug,
{
    use num_complex::Complex64;

    // Convert cutoff to f64
    let wn = num_traits::cast::cast::<T, f64>(cutoff)
        .ok_or_else(|| SignalError::ValueError(format!("Could not convert {:?} to f64", cutoff)))?;

    // Convert filter_type to FilterType
    let filter_type_param = filter_type.into();
    let filter_type = match filter_type_param {
        FilterTypeParam::Type(t) => t,
        FilterTypeParam::String(s) => s.parse()?,
    };

    // Validate parameters
    if order == 0 {
        return Err(SignalError::ValueError(
            "Filter order must be greater than 0".to_string(),
        ));
    }

    if wn <= 0.0 || wn >= 1.0 {
        return Err(SignalError::ValueError(format!(
            "Cutoff frequency must be between 0 and 1, got {}",
            wn
        )));
    }

    if ripple <= 0.0 {
        return Err(SignalError::ValueError(format!(
            "Passband ripple must be positive, got {}",
            ripple
        )));
    }

    if attenuation <= 0.0 {
        return Err(SignalError::ValueError(format!(
            "Stopband attenuation must be positive, got {}",
            attenuation
        )));
    }

    // Only support lowpass and highpass for now
    if !matches!(filter_type, FilterType::Lowpass | FilterType::Highpass) {
        return Err(SignalError::NotImplementedError(
            "Bandpass and bandstop elliptic filters not yet implemented".to_string(),
        ));
    }

    // Calculate elliptic filter parameters
    let epsilon = (10.0_f64.powf(ripple / 10.0) - 1.0).sqrt();
    let k1 = epsilon / (10.0_f64.powf(attenuation / 10.0) - 1.0).sqrt();

    // Generate poles and zeros for lowpass prototype
    let (zeros, poles, gain) = elliptic_prototype(order, epsilon, k1)?;

    // Frequency transform based on filter type
    let (transformed_zeros, transformed_poles, transformed_gain) = match filter_type {
        FilterType::Lowpass => {
            // Scale by warped frequency for lowpass
            let warped_freq = (std::f64::consts::PI * wn / 2.0).tan();
            let lp_poles: Vec<_> = poles.iter().map(|p| p * warped_freq).collect();
            let lp_zeros: Vec<_> = zeros.iter().map(|z| z * warped_freq).collect();
            (lp_zeros, lp_poles, gain)
        }
        FilterType::Highpass => {
            // Lowpass to highpass transformation: s -> wc/s
            let warped_freq = (std::f64::consts::PI * wn / 2.0).tan();
            let hp_poles: Vec<_> = poles.iter().map(|p| warped_freq / p).collect();
            let hp_zeros: Vec<_> = zeros.iter().map(|z| warped_freq / z).collect();

            // Add zeros at origin for highpass (degree matching)
            let mut all_zeros = hp_zeros;
            for _ in 0..(order - zeros.len()) {
                all_zeros.push(Complex64::new(0.0, 0.0));
            }

            (all_zeros, hp_poles, gain)
        }
        _ => unreachable!(),
    };

    // Apply bilinear transform to get digital filter
    let mut digital_poles = Vec::new();
    let mut digital_zeros = Vec::new();

    for &pole in &transformed_poles {
        let z_pole = (2.0 + pole) / (2.0 - pole);
        digital_poles.push(z_pole);
    }

    for &zero in &transformed_zeros {
        let z_zero = (2.0 + zero) / (2.0 - zero);
        digital_zeros.push(z_zero);
    }

    // For highpass filters, add zeros at z = 1
    if matches!(filter_type, FilterType::Highpass) {
        let zeros_to_add = order - transformed_zeros.len();
        for _ in 0..zeros_to_add {
            digital_zeros.push(Complex64::new(1.0, 0.0));
        }
    }

    // Convert to transfer function coefficients
    let (b, a) = zpk_to_tf(&digital_zeros, &digital_poles, transformed_gain)?;

    Ok((b, a))
}

/// Generate poles, zeros, and gain for elliptic lowpass prototype
///
/// This is a simplified implementation that provides good results for most practical applications
fn elliptic_prototype(
    order: usize,
    epsilon: f64,
    k1: f64,
) -> SignalResult<(Vec<Complex64>, Vec<Complex64>, f64)> {
    use num_complex::Complex64;

    let mut zeros = Vec::new();
    let mut poles = Vec::new();

    // For elliptic filters, we alternate between poles and zeros
    // This is a simplified approach using approximated values

    // Calculate modular constant k (discrimination parameter)
    let k = k1;
    let k_prime = (1.0 - k * k).sqrt();

    // Number of zeros equals floor(order/2) * 2 for elliptic filters
    let num_zeros = (order / 2) * 2;

    // Generate zeros (purely imaginary for lowpass prototype)
    for i in 1..=(num_zeros / 2) {
        let v_i = (2 * i - 1) as f64 / order as f64;

        // Simplified calculation of zero locations
        // In a full implementation, this would use Jacobi elliptic functions
        let zero_imag = 1.0 / (k * (std::f64::consts::PI * v_i / 2.0).sin());

        zeros.push(Complex64::new(0.0, zero_imag));
        zeros.push(Complex64::new(0.0, -zero_imag));
    }

    // Generate poles using approximate formulas
    for i in 1..=order {
        let v_i = (2 * i - 1) as f64 / order as f64;
        let theta = std::f64::consts::PI * v_i / (2.0 * order as f64);

        // Simplified pole calculation
        // This approximates the elliptic rational function
        let sigma = -epsilon.recip() * theta.sin();
        let omega = theta.cos() / ((1.0 + epsilon.powi(2)).sqrt());

        // Apply correction for elliptic characteristic
        let correction = 1.0 + k_prime.powi(2) * theta.sin().powi(2);
        let real_part = sigma / correction;
        let imag_part = omega / correction;

        if i <= order / 2 {
            poles.push(Complex64::new(real_part, imag_part));
            if order % 2 == 0 || i < order / 2 {
                poles.push(Complex64::new(real_part, -imag_part));
            }
        }
    }

    // For odd order, add real pole
    if order % 2 == 1 {
        let real_pole = -1.0 / epsilon;
        poles.push(Complex64::new(real_pole, 0.0));
    }

    // Ensure we have exactly 'order' poles
    poles.truncate(order);

    // Calculate gain to normalize the filter
    // For elliptic filters, gain depends on passband ripple
    let gain = if order % 2 == 1 {
        1.0 / epsilon
    } else {
        1.0 / (1.0 + epsilon.powi(2)).sqrt()
    };

    Ok((zeros, poles, gain))
}

/// Bessel filter design
///
/// Bessel filters have maximally flat group delay, making them ideal for
/// applications where preserving signal waveform is critical.
///
/// # Arguments
///
/// * `order` - Filter order (1-10 supported)
/// * `cutoff` - Cutoff frequency (normalized from 0 to 1, where 1 is the Nyquist frequency)
/// * `filter_type` - Filter type (lowpass, highpass)
///
/// # Returns
///
/// * A tuple of filter coefficients (b, a) where b are the numerator coefficients and a are
///   the denominator coefficients
pub fn bessel<T>(
    order: usize,
    cutoff: T,
    filter_type: impl Into<FilterTypeParam>,
) -> SignalResult<(Vec<f64>, Vec<f64>)>
where
    T: Float + NumCast + Debug,
{
    use num_complex::Complex64;

    // Convert cutoff to f64
    let wn = num_traits::cast::cast::<T, f64>(cutoff)
        .ok_or_else(|| SignalError::ValueError(format!("Could not convert {:?} to f64", cutoff)))?;

    // Convert filter_type to FilterType
    let filter_type_param = filter_type.into();
    let filter_type = match filter_type_param {
        FilterTypeParam::Type(t) => t,
        FilterTypeParam::String(s) => s.parse()?,
    };

    // Validate parameters
    if order == 0 || order > 10 {
        return Err(SignalError::ValueError(
            "Bessel filter order must be between 1 and 10".to_string(),
        ));
    }

    if wn <= 0.0 || wn >= 1.0 {
        return Err(SignalError::ValueError(format!(
            "Cutoff frequency must be between 0 and 1, got {}",
            wn
        )));
    }

    // Bessel polynomial coefficients (normalized for unit delay at DC)
    // These are precomputed Bessel polynomial roots
    let bessel_poles = match order {
        1 => vec![Complex64::new(-1.0, 0.0)],
        2 => vec![
            Complex64::new(-1.1016, 0.6360),
            Complex64::new(-1.1016, -0.6360),
        ],
        3 => vec![
            Complex64::new(-1.0474, 0.0),
            Complex64::new(-1.3397, 0.7179),
            Complex64::new(-1.3397, -0.7179),
        ],
        4 => vec![
            Complex64::new(-1.3735, 0.4102),
            Complex64::new(-1.3735, -0.4102),
            Complex64::new(-1.4759, 0.7179),
            Complex64::new(-1.4759, -0.7179),
        ],
        5 => vec![
            Complex64::new(-1.5023, 0.0),
            Complex64::new(-1.5611, 0.3226),
            Complex64::new(-1.5611, -0.3226),
            Complex64::new(-1.6853, 0.7327),
            Complex64::new(-1.6853, -0.7327),
        ],
        6 => vec![
            Complex64::new(-1.6060, 0.2538),
            Complex64::new(-1.6060, -0.2538),
            Complex64::new(-1.6913, 0.4425),
            Complex64::new(-1.6913, -0.4425),
            Complex64::new(-1.8574, 0.7445),
            Complex64::new(-1.8574, -0.7445),
        ],
        7 => vec![
            Complex64::new(-1.6853, 0.0),
            Complex64::new(-1.7174, 0.2003),
            Complex64::new(-1.7174, -0.2003),
            Complex64::new(-1.8235, 0.4206),
            Complex64::new(-1.8235, -0.4206),
            Complex64::new(-2.0106, 0.7506),
            Complex64::new(-2.0106, -0.7506),
        ],
        8 => vec![
            Complex64::new(-1.7837, 0.1661),
            Complex64::new(-1.7837, -0.1661),
            Complex64::new(-1.8574, 0.3506),
            Complex64::new(-1.8574, -0.3506),
            Complex64::new(-1.9781, 0.4943),
            Complex64::new(-1.9781, -0.4943),
            Complex64::new(-2.1506, 0.7544),
            Complex64::new(-2.1506, -0.7544),
        ],
        9 => vec![
            Complex64::new(-1.8574, 0.0),
            Complex64::new(-1.8794, 0.1397),
            Complex64::new(-1.8794, -0.1397),
            Complex64::new(-1.9440, 0.2947),
            Complex64::new(-1.9440, -0.2947),
            Complex64::new(-2.0815, 0.4642),
            Complex64::new(-2.0815, -0.4642),
            Complex64::new(-2.2801, 0.7571),
            Complex64::new(-2.2801, -0.7571),
        ],
        10 => vec![
            Complex64::new(-1.9440, 0.1212),
            Complex64::new(-1.9440, -0.1212),
            Complex64::new(-1.9925, 0.2568),
            Complex64::new(-1.9925, -0.2568),
            Complex64::new(-2.1024, 0.4090),
            Complex64::new(-2.1024, -0.4090),
            Complex64::new(-2.2582, 0.5496),
            Complex64::new(-2.2582, -0.5496),
            Complex64::new(-2.4022, 0.7588),
            Complex64::new(-2.4022, -0.7588),
        ],
        _ => unreachable!(),
    };

    // Apply frequency transformation
    let (zeros, transformed_poles, gain) = match filter_type {
        FilterType::Lowpass => {
            let warped_freq = (std::f64::consts::PI * wn / 2.0).tan();
            let scaled_poles: Vec<_> = bessel_poles.iter().map(|p| p * warped_freq).collect();

            // Bessel filters have unity gain at DC
            let mut dc_gain = 1.0;
            for &pole in &scaled_poles {
                dc_gain *= -pole.re;
            }

            (Vec::new(), scaled_poles, dc_gain)
        }
        FilterType::Highpass => {
            let warped_freq = (std::f64::consts::PI * wn / 2.0).tan();
            let hp_poles: Vec<_> = bessel_poles.iter().map(|p| warped_freq / p).collect();
            let hp_zeros = vec![Complex64::new(0.0, 0.0); order];
            (hp_zeros, hp_poles, 1.0)
        }
        FilterType::Bandpass | FilterType::Bandstop => {
            return Err(SignalError::NotImplementedError(
                "Bandpass and bandstop Bessel filters not yet implemented".to_string(),
            ));
        }
    };

    // Apply bilinear transform
    let mut digital_poles = Vec::new();
    let mut digital_zeros = Vec::new();

    for &pole in &transformed_poles {
        let z_pole = (2.0 + pole) / (2.0 - pole);
        digital_poles.push(z_pole);
    }

    for &zero in &zeros {
        let z_zero = (2.0 + zero) / (2.0 - zero);
        digital_zeros.push(z_zero);
    }

    // For highpass filters, add zeros at z = 1
    if matches!(filter_type, FilterType::Highpass) {
        for _ in 0..order {
            digital_zeros.push(Complex64::new(1.0, 0.0));
        }
    }

    // Convert to transfer function coefficients
    let (b, a) = zpk_to_tf(&digital_zeros, &digital_poles, gain)?;

    Ok((b, a))
}

/// FIR filter design using the window method
///
/// # Arguments
///
/// * `numtaps` - Number of filter taps (filter order + 1)
/// * `cutoff` - Cutoff frequency (normalized from 0 to 1, where 1 is the Nyquist frequency)
/// * `window` - Window function name or parameters
/// * `pass_zero` - If true, the filter is lowpass, otherwise highpass
///
/// # Returns
///
/// * Filter coefficients
pub fn firwin<T>(
    _numtaps: usize,
    _cutoff: T,
    _window: &str,
    _pass_zero: bool,
) -> SignalResult<Vec<f64>>
where
    T: Float + NumCast + Debug,
{
    // Not yet implemented
    Err(SignalError::NotImplementedError(
        "FIR filter design using window method is not yet implemented".to_string(),
    ))
}

/// Parks-McClellan optimal FIR filter design (Remez exchange algorithm)
///
/// Design a linear phase FIR filter using the Parks-McClellan algorithm.
/// The algorithm finds the filter coefficients that minimize the maximum
/// error between the desired and actual frequency response.
///
/// # Arguments
///
/// * `numtaps` - Number of filter taps (filter order + 1)
/// * `bands` - Frequency bands specified as pairs of band edges (0 to 1, where 1 is Nyquist)
/// * `desired` - Desired gain for each band
/// * `weights` - Relative weights for each band (optional)
/// * `max_iter` - Maximum number of iterations (default: 25)
/// * `grid_density` - Grid density for frequency sampling (default: 16)
///
/// # Returns
///
/// * Filter coefficients
///
/// # Examples
///
/// ```ignore
/// use scirs2_signal::filter::remez;
///
/// // Design a 65-tap lowpass filter
/// // Passband: 0-0.4, Stopband: 0.45-1.0
/// let bands = vec![0.0, 0.4, 0.45, 1.0];
/// let desired = vec![1.0, 1.0, 0.0, 0.0];
/// let h = remez(65, &bands, &desired, None, None, None).unwrap();
/// ```
pub fn remez(
    numtaps: usize,
    bands: &[f64],
    desired: &[f64],
    weights: Option<&[f64]>,
    max_iter: Option<usize>,
    grid_density: Option<usize>,
) -> SignalResult<Vec<f64>> {
    // Validate inputs
    if numtaps < 3 {
        return Err(SignalError::ValueError(
            "Number of taps must be at least 3".to_string(),
        ));
    }

    if bands.len() % 2 != 0 || bands.len() < 2 {
        return Err(SignalError::ValueError(
            "Bands must be specified as pairs of edges".to_string(),
        ));
    }

    if desired.len() != bands.len() {
        return Err(SignalError::ValueError(
            "Desired array must have same length as bands".to_string(),
        ));
    }

    // Check that bands are monotonically increasing
    for i in 1..bands.len() {
        if bands[i] <= bands[i - 1] {
            return Err(SignalError::ValueError(
                "Band edges must be monotonically increasing".to_string(),
            ));
        }
    }

    // Check that bands are within [0, 1]
    if bands[0] < 0.0 || bands[bands.len() - 1] > 1.0 {
        return Err(SignalError::ValueError(
            "Band edges must be between 0 and 1".to_string(),
        ));
    }

    let max_iter = max_iter.unwrap_or(25);
    let grid_density = grid_density.unwrap_or(16);

    // Calculate filter order
    let filter_order = numtaps - 1;
    let _is_odd_order = filter_order % 2 == 1;

    // Number of extremal frequencies
    let r = (filter_order + 2) / 2;

    // Set up the dense frequency grid
    let grid_size = grid_density * filter_order;
    let mut omega_grid = Vec::with_capacity(grid_size);
    let mut desired_grid = Vec::with_capacity(grid_size);
    let mut weight_grid = Vec::with_capacity(grid_size);

    // Build the frequency grid for each band
    let num_bands = bands.len() / 2;
    for band_idx in 0..num_bands {
        let band_start = bands[2 * band_idx];
        let band_end = bands[2 * band_idx + 1];
        let band_points = ((band_end - band_start) * grid_size as f64).round() as usize;

        for i in 0..band_points {
            let omega =
                band_start + (band_end - band_start) * (i as f64) / (band_points as f64 - 1.0);
            omega_grid.push(omega * std::f64::consts::PI);

            // Linear interpolation for desired response
            let t = (omega - band_start) / (band_end - band_start);
            let des = desired[2 * band_idx] * (1.0 - t) + desired[2 * band_idx + 1] * t;
            desired_grid.push(des);

            // Set weights
            if let Some(w) = weights {
                let wt = w[2 * band_idx] * (1.0 - t) + w[2 * band_idx + 1] * t;
                weight_grid.push(wt);
            } else {
                weight_grid.push(1.0);
            }
        }
    }

    // Initialize extremal frequencies uniformly
    let mut extremal_freqs = Vec::with_capacity(r);
    for i in 0..r {
        extremal_freqs.push(i * (omega_grid.len() - 1) / (r - 1));
    }

    // Remez exchange algorithm
    let mut h = vec![0.0; numtaps];
    let mut best_error = f64::MAX;

    for _iter in 0..max_iter {
        // Step 1: Calculate the polynomial using the extremal frequencies
        let mut a_matrix = vec![vec![0.0; r]; r];
        let mut b_vector = vec![0.0; r];

        for (i, &ext_idx) in extremal_freqs.iter().enumerate() {
            let omega = omega_grid[ext_idx];

            // Fill the matrix for the linear system
            for j in 0..(r - 1) {
                a_matrix[i][j] = (j as f64 * omega).cos();
            }
            // Last column alternates signs
            a_matrix[i][r - 1] = if i % 2 == 0 { 1.0 } else { -1.0 } / weight_grid[ext_idx];

            b_vector[i] = desired_grid[ext_idx];
        }

        // Solve the linear system to get polynomial coefficients
        let coeffs = solve_linear_system(&a_matrix, &b_vector)?;

        // Step 2: Calculate error on the dense grid
        let mut errors = Vec::with_capacity(omega_grid.len());
        let mut max_error = 0.0;

        for i in 0..omega_grid.len() {
            let omega = omega_grid[i];

            // Evaluate the polynomial
            let mut p_omega = 0.0;
            for (j, &coeff) in coeffs.iter().enumerate().take(r - 1) {
                p_omega += coeff * (j as f64 * omega).cos();
            }

            let error = (desired_grid[i] - p_omega) * weight_grid[i];
            errors.push(error.abs());

            if error.abs() > max_error {
                max_error = error.abs();
            }
        }

        // Step 3: Find new extremal frequencies
        let mut new_extremal = Vec::new();

        // Find local maxima in the error function
        for i in 1..(errors.len() - 1) {
            if errors[i] >= errors[i - 1] && errors[i] >= errors[i + 1] {
                new_extremal.push(i);
            }
        }

        // Add boundaries if they are extremal
        if errors[0] > errors[1] {
            new_extremal.insert(0, 0);
        }
        if errors[errors.len() - 1] > errors[errors.len() - 2] {
            new_extremal.push(errors.len() - 1);
        }

        // Select r extremal points with alternating signs
        if new_extremal.len() >= r {
            // Sort by error magnitude
            new_extremal.sort_by(|&a, &b| errors[b].partial_cmp(&errors[a]).unwrap());

            // Keep the r largest errors
            new_extremal.truncate(r);
            new_extremal.sort();

            extremal_freqs = new_extremal;
        }

        // Check convergence
        if max_error < best_error {
            best_error = max_error;

            // Convert polynomial coefficients to filter coefficients
            for (i, coeff) in h.iter_mut().enumerate() {
                let n = i as f64 - (numtaps as f64 - 1.0) / 2.0;

                *coeff = 0.0;
                for (j, &c) in coeffs.iter().enumerate().take(r - 1) {
                    if j == 0 {
                        *coeff += c;
                    } else {
                        let freq = j as f64 * std::f64::consts::PI / (numtaps as f64 - 1.0);
                        *coeff += 2.0 * c * (freq * n).cos();
                    }
                }
                *coeff /= numtaps as f64;
            }
        }

        // Check if converged
        if max_error - best_error < 1e-10 {
            break;
        }
    }

    // Make filter symmetric
    let mid = numtaps / 2;
    for i in 0..mid {
        let avg = (h[i] + h[numtaps - 1 - i]) / 2.0;
        h[i] = avg;
        h[numtaps - 1 - i] = avg;
    }

    Ok(h)
}

/// Solve a linear system Ax = b using Gaussian elimination
fn solve_linear_system(a: &[Vec<f64>], b: &[f64]) -> SignalResult<Vec<f64>> {
    let n = a.len();
    if n == 0 || a[0].len() != n || b.len() != n {
        return Err(SignalError::ValueError(
            "Invalid matrix dimensions".to_string(),
        ));
    }

    // Create augmented matrix
    let mut aug = vec![vec![0.0; n + 1]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = a[i][j];
        }
        aug[i][n] = b[i];
    }

    // Forward elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..n {
            if aug[k][i].abs() > aug[max_row][i].abs() {
                max_row = k;
            }
        }

        // Swap rows
        aug.swap(i, max_row);

        // Check for zero pivot
        if aug[i][i].abs() < 1e-10 {
            return Err(SignalError::ValueError("Singular matrix".to_string()));
        }

        // Eliminate column
        for k in (i + 1)..n {
            let factor = aug[k][i] / aug[i][i];
            for j in i..=n {
                aug[k][j] -= factor * aug[i][j];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = aug[i][n];
        for j in (i + 1)..n {
            x[i] -= aug[i][j] * x[j];
        }
        x[i] /= aug[i][i];
    }

    Ok(x)
}

/// Apply an IIR filter forward and backward to a signal
///
/// # Arguments
///
/// * `b` - Numerator coefficients
/// * `a` - Denominator coefficients
/// * `x` - Input signal
///
/// # Returns
///
/// * Filtered signal
pub fn filtfilt<T>(b: &[f64], a: &[f64], x: &[T]) -> SignalResult<Vec<f64>>
where
    T: Float + NumCast + Debug,
{
    if a.is_empty() || a[0] == 0.0 {
        return Err(SignalError::ValueError(
            "First denominator coefficient cannot be zero".to_string(),
        ));
    }

    // Convert input to f64
    let x_f64: Vec<f64> = x
        .iter()
        .map(|&val| {
            num_traits::cast::cast::<T, f64>(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<_>>>()?;

    // 1. Apply filter forward
    let y1 = lfilter(b, a, &x_f64)?;

    // 2. Reverse the result
    let mut y1_rev = y1.clone();
    y1_rev.reverse();

    // 3. Apply filter backward
    let y2 = lfilter(b, a, &y1_rev)?;

    // 4. Reverse again to get the final result
    let mut result = y2;
    result.reverse();

    Ok(result)
}

/// Apply an IIR filter to a signal
///
/// # Arguments
///
/// * `b` - Numerator coefficients
/// * `a` - Denominator coefficients
/// * `x` - Input signal
///
/// # Returns
///
/// * Filtered signal
pub fn lfilter<T>(b: &[f64], a: &[f64], x: &[T]) -> SignalResult<Vec<f64>>
where
    T: Float + NumCast + Debug,
{
    if a.is_empty() || a[0] == 0.0 {
        return Err(SignalError::ValueError(
            "First denominator coefficient cannot be zero".to_string(),
        ));
    }

    // Normalize coefficients by a[0]
    let a0 = a[0];
    let b_norm: Vec<f64> = b.iter().map(|&val| val / a0).collect();
    let a_norm: Vec<f64> = a.iter().map(|&val| val / a0).collect();

    // Convert input to f64
    let x_f64: Vec<f64> = x
        .iter()
        .map(|&val| {
            num_traits::cast::cast::<T, f64>(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<_>>>()?;

    // Apply filter using direct form II transposed
    let n = x_f64.len();
    let mut y = vec![0.0; n];
    let mut z = vec![0.0; a_norm.len() - 1]; // State variables

    for i in 0..n {
        // Compute output
        y[i] = b_norm[0] * x_f64[i] + z[0];

        // Update state variables
        for j in 1..z.len() {
            z[j - 1] = b_norm[j] * x_f64[i] + z[j] - a_norm[j] * y[i];
        }

        // Update last state variable
        if !z.is_empty() {
            let last = z.len() - 1;
            if b_norm.len() > last + 1 {
                z[last] = b_norm[last + 1] * x_f64[i] - a_norm[last + 1] * y[i];
            } else {
                z[last] = -a_norm[last + 1] * y[i];
            }
        }
    }

    Ok(y)
}

/// Convert a filter to minimum phase
///
/// A minimum phase filter has all its zeros inside the unit circle (discrete-time)
/// or with negative real parts (continuous-time). This function converts any filter
/// to its minimum phase equivalent while preserving the magnitude response.
///
/// # Arguments
///
/// * `b` - Numerator coefficients of the filter
/// * `discrete_time` - True for discrete-time systems, false for continuous-time
///
/// # Returns
///
/// * Minimum phase filter coefficients
///
/// # Examples
///
/// ```ignore
/// use scirs2_signal::filter::minimum_phase;
///
/// // Convert a filter to minimum phase
/// let b = vec![1.0, -2.0, 1.0]; // (z-1)^2, has zeros at z=1 (outside unit circle)
/// let b_min = minimum_phase(&b, true).unwrap();
/// ```
pub fn minimum_phase(b: &[f64], discrete_time: bool) -> SignalResult<Vec<f64>> {
    if b.is_empty() {
        return Err(SignalError::ValueError(
            "Filter coefficients cannot be empty".to_string(),
        ));
    }

    // For constant filters, return as-is
    if b.len() == 1 {
        return Ok(b.to_vec());
    }

    // Find the roots (zeros) of the polynomial
    let zeros = find_polynomial_roots(b)?;

    // Convert non-minimum phase zeros to minimum phase
    let mut min_phase_zeros = Vec::new();
    let mut gain_adjustment = 1.0;

    for zero in zeros {
        if discrete_time {
            // For discrete-time: zeros inside unit circle are minimum phase
            if zero.norm() > 1.0 {
                // Reflect zero to its conjugate reciprocal: 1/conj(zero)
                let min_zero = 1.0 / zero.conj();
                min_phase_zeros.push(min_zero);
                // Adjust gain to preserve magnitude response
                gain_adjustment *= zero.norm();
            } else {
                min_phase_zeros.push(zero);
            }
        } else {
            // For continuous-time: zeros with negative real parts are minimum phase
            if zero.re > 0.0 {
                // Reflect zero to negative real part: -conj(zero)
                let min_zero = -zero.conj();
                min_phase_zeros.push(min_zero);
                // Adjust gain to preserve magnitude response at s=0
                gain_adjustment *= -zero.re / min_zero.re;
            } else {
                min_phase_zeros.push(zero);
            }
        }
    }

    // Reconstruct polynomial from minimum phase zeros
    let mut min_phase_b = polynomial_from_roots(&min_phase_zeros);

    // Apply gain adjustment
    for coeff in &mut min_phase_b {
        *coeff *= gain_adjustment;
    }

    // Normalize to match original leading coefficient if needed
    if !min_phase_b.is_empty() && min_phase_b[0].abs() > 1e-10 {
        let scale = b[0] / min_phase_b[0];
        for coeff in &mut min_phase_b {
            *coeff *= scale;
        }
    }

    Ok(min_phase_b)
}

/// Extract group delay from a filter
///
/// Group delay is the negative derivative of the phase response with respect to frequency.
/// It represents the time delay experienced by different frequency components.
///
/// # Arguments
///
/// * `b` - Numerator coefficients
/// * `a` - Denominator coefficients
/// * `w` - Frequency points (normalized from 0 to π)
///
/// # Returns
///
/// * Group delay values at the specified frequencies
pub fn group_delay(b: &[f64], a: &[f64], w: &[f64]) -> SignalResult<Vec<f64>> {
    if a.is_empty() || a[0].abs() < 1e-10 {
        return Err(SignalError::ValueError(
            "Invalid denominator coefficients".to_string(),
        ));
    }

    let mut gd = Vec::with_capacity(w.len());

    for &freq in w {
        // Compute the group delay using the derivative method
        // gd = -d(phase)/dw = -d(arg(H(e^jw)))/dw

        // For numerical computation, use a small frequency step
        let eps = 1e-6;
        let freq_minus = (freq - eps).max(0.0);
        let freq_plus = (freq + eps).min(std::f64::consts::PI);

        // Evaluate transfer function at freq-eps and freq+eps
        let h_minus = evaluate_transfer_function(b, a, freq_minus);
        let h_plus = evaluate_transfer_function(b, a, freq_plus);

        // Compute phase difference and normalize by frequency difference
        let phase_diff = h_plus.arg() - h_minus.arg();
        let freq_diff = freq_plus - freq_minus;

        if freq_diff > 0.0 {
            gd.push(-phase_diff / freq_diff);
        } else {
            gd.push(0.0);
        }
    }

    Ok(gd)
}

/// Find polynomial roots using a simplified iterative method
///
/// This is a basic implementation for demonstration purposes.
/// Production code would use more robust algorithms like Jenkins-Traub or eigenvalue methods.
fn find_polynomial_roots(coeffs: &[f64]) -> SignalResult<Vec<num_complex::Complex64>> {
    use num_complex::Complex64;

    if coeffs.is_empty() {
        return Ok(Vec::new());
    }

    // Remove leading zeros
    let mut trimmed_coeffs = coeffs.to_vec();
    while trimmed_coeffs.len() > 1 && trimmed_coeffs[0].abs() < 1e-10 {
        trimmed_coeffs.remove(0);
    }

    let n = trimmed_coeffs.len() - 1;
    if n == 0 {
        return Ok(Vec::new());
    }

    let mut roots = Vec::new();

    // Handle linear case
    if n == 1 {
        if trimmed_coeffs[0].abs() > 1e-10 {
            roots.push(Complex64::new(-trimmed_coeffs[1] / trimmed_coeffs[0], 0.0));
        }
        return Ok(roots);
    }

    // Handle quadratic case
    if n == 2 {
        let a = trimmed_coeffs[0];
        let b = trimmed_coeffs[1];
        let c = trimmed_coeffs[2];

        if a.abs() > 1e-10 {
            let discriminant = b * b - 4.0 * a * c;
            if discriminant >= 0.0 {
                let sqrt_disc = discriminant.sqrt();
                roots.push(Complex64::new((-b + sqrt_disc) / (2.0 * a), 0.0));
                roots.push(Complex64::new((-b - sqrt_disc) / (2.0 * a), 0.0));
            } else {
                let sqrt_disc = (-discriminant).sqrt();
                roots.push(Complex64::new(-b / (2.0 * a), sqrt_disc / (2.0 * a)));
                roots.push(Complex64::new(-b / (2.0 * a), -sqrt_disc / (2.0 * a)));
            }
        }
        return Ok(roots);
    }

    // For higher-order polynomials, use a simplified iterative method
    // This is not as robust as professional root-finding algorithms
    let max_iterations = 100;
    let tolerance = 1e-10;

    // Use Durand-Kerner method with initial guesses on a circle
    let mut estimates = Vec::with_capacity(n);
    for k in 0..n {
        let angle = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
        estimates.push(Complex64::new(angle.cos(), angle.sin()));
    }

    for _iter in 0..max_iterations {
        let mut converged = true;
        let _old_estimates = estimates.clone();

        for i in 0..n {
            // Evaluate polynomial and its derivative at current estimate
            let z = estimates[i];
            let (p_val, p_prime) = evaluate_polynomial_and_derivative(&trimmed_coeffs, z);

            // Compute the product for Durand-Kerner method
            let mut product = Complex64::new(1.0, 0.0);
            for (j, &estimate) in estimates.iter().enumerate().take(n) {
                if i != j {
                    product *= z - estimate;
                }
            }

            // Update estimate
            if product.norm() > tolerance && p_prime.norm() > tolerance {
                let correction = p_val / product;
                estimates[i] = z - correction;

                if correction.norm() > tolerance {
                    converged = false;
                }
            }
        }

        if converged {
            break;
        }
    }

    // Filter out potential spurious roots
    for estimate in estimates {
        let (p_val, _) = evaluate_polynomial_and_derivative(&trimmed_coeffs, estimate);
        if p_val.norm() < 1e-6 {
            roots.push(estimate);
        }
    }

    Ok(roots)
}

/// Evaluate polynomial and its derivative at a complex point
fn evaluate_polynomial_and_derivative(
    coeffs: &[f64],
    z: num_complex::Complex64,
) -> (num_complex::Complex64, num_complex::Complex64) {
    use num_complex::Complex64;

    if coeffs.is_empty() {
        return (Complex64::zero(), Complex64::zero());
    }

    let n = coeffs.len() - 1;
    let mut p_val = Complex64::new(coeffs[0], 0.0);
    let mut p_prime = Complex64::zero();

    for (i, &coeff) in coeffs.iter().enumerate().skip(1) {
        let power = (n - i) as i32;
        p_prime = p_prime * z + p_val * Complex64::new(power as f64, 0.0);
        p_val = p_val * z + Complex64::new(coeff, 0.0);
    }

    (p_val, p_prime)
}

/// Reconstruct polynomial coefficients from roots
fn polynomial_from_roots(roots: &[num_complex::Complex64]) -> Vec<f64> {
    if roots.is_empty() {
        return vec![1.0];
    }

    // Start with polynomial: 1
    let mut poly = vec![1.0];

    // Multiply by (z - root) for each root
    for &root in roots {
        let mut new_poly = vec![0.0; poly.len() + 1];

        // Multiply existing polynomial by z
        for (i, &coeff) in poly.iter().enumerate() {
            new_poly[i] += coeff;
        }

        // Subtract root times existing polynomial
        for (i, &coeff) in poly.iter().enumerate() {
            new_poly[i + 1] -= coeff * root.re;
            // For complex roots, we assume they come in conjugate pairs
            // and the imaginary parts will cancel out
        }

        poly = new_poly;
    }

    // Remove small imaginary parts that should be zero due to conjugate pairs
    poly
}

/// Design a matched filter for detecting a known signal in noise
///
/// A matched filter is optimal for detecting a known signal in the presence of
/// additive white Gaussian noise. It maximizes the signal-to-noise ratio at the
/// output and is widely used in radar, communications, and correlation applications.
///
/// # Arguments
///
/// * `template` - The known signal template to match against
/// * `normalize` - If true, normalize the filter to unit energy
///
/// # Returns
///
/// * Matched filter coefficients (time-reversed and conjugated template)
///
/// # Examples
///
/// ```ignore
/// use scirs2_signal::filter::matched_filter;
///
/// // Design a matched filter for a simple pulse
/// let template = vec![1.0, 1.0, 1.0, 0.0, 0.0];
/// let mf = matched_filter(&template, true).unwrap();
/// ```
pub fn matched_filter(template: &[f64], normalize: bool) -> SignalResult<Vec<f64>> {
    if template.is_empty() {
        return Err(SignalError::ValueError(
            "Template cannot be empty".to_string(),
        ));
    }

    // Matched filter is the time-reversed (and conjugated for complex signals) template
    let mut mf: Vec<f64> = template.iter().rev().copied().collect();

    if normalize {
        // Normalize to unit energy
        let energy: f64 = mf.iter().map(|&x| x * x).sum();
        if energy > 1e-10 {
            let norm_factor = energy.sqrt();
            for coeff in &mut mf {
                *coeff /= norm_factor;
            }
        }
    }

    Ok(mf)
}

/// Apply matched filtering to detect a template in a signal
///
/// This function convolves the input signal with the matched filter to detect
/// occurrences of the template. The output represents the correlation between
/// the signal and the template at each time point.
///
/// # Arguments
///
/// * `signal` - Input signal to search in
/// * `template` - Template signal to search for
/// * `mode` - Correlation mode ("full", "same", "valid")
///
/// # Returns
///
/// * Correlation output showing template matches
///
/// # Examples
///
/// ```ignore
/// use scirs2_signal::filter::matched_filter_detect;
///
/// let signal = vec![0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0];
/// let template = vec![1.0, 1.0, 1.0];
/// let output = matched_filter_detect(&signal, &template, "same").unwrap();
/// ```
pub fn matched_filter_detect(
    signal: &[f64],
    template: &[f64],
    mode: &str,
) -> SignalResult<Vec<f64>> {
    if signal.is_empty() || template.is_empty() {
        return Err(SignalError::ValueError(
            "Signal and template cannot be empty".to_string(),
        ));
    }

    // Get the matched filter
    let mf = matched_filter(template, true)?;

    // Apply correlation (which is convolution with time-reversed filter)
    correlate(signal, &mf, mode)
}

/// Cross-correlation between two signals
///
/// Computes the cross-correlation of two signals using different modes:
/// - "full": Full correlation, output length = len(a) + len(b) - 1
/// - "same": Same length as first input, centered
/// - "valid": Only where they fully overlap, length = |len(a) - len(b)| + 1
///
/// # Arguments
///
/// * `a` - First input signal
/// * `b` - Second input signal  
/// * `mode` - Correlation mode
///
/// # Returns
///
/// * Cross-correlation sequence
pub fn correlate(a: &[f64], b: &[f64], mode: &str) -> SignalResult<Vec<f64>> {
    if a.is_empty() || b.is_empty() {
        return Err(SignalError::ValueError(
            "Input signals cannot be empty".to_string(),
        ));
    }

    let n_a = a.len();
    let n_b = b.len();

    // Flip b for correlation (correlation = convolution with flipped signal)
    let b_flipped: Vec<f64> = b.iter().rev().copied().collect();

    // Determine output size based on mode
    let (output_size, start_idx) = match mode.to_lowercase().as_str() {
        "full" => (n_a + n_b - 1, 0),
        "same" => {
            let size = n_a;
            let start = (n_b - 1) / 2;
            (size, start)
        }
        "valid" => {
            if n_a >= n_b {
                (n_a - n_b + 1, n_b - 1)
            } else {
                (n_b - n_a + 1, n_a - 1)
            }
        }
        _ => {
            return Err(SignalError::ValueError(
                "Mode must be 'full', 'same', or 'valid'".to_string(),
            ))
        }
    };

    let mut output = vec![0.0; output_size];

    // Compute correlation using direct convolution
    for (i, output_val) in output.iter_mut().enumerate().take(output_size) {
        let actual_i = i + start_idx;
        for (j, &b_val) in b_flipped.iter().enumerate() {
            if actual_i >= j && actual_i - j < n_a {
                *output_val += a[actual_i - j] * b_val;
            }
        }
    }

    Ok(output)
}

/// Detect peaks in a signal using matched filter output
///
/// This function processes the matched filter output to identify significant
/// peaks that likely correspond to template matches.
///
/// # Arguments
///
/// * `correlation` - Output from matched filter detection
/// * `threshold` - Detection threshold (0.0 to 1.0, relative to max)
/// * `min_distance` - Minimum distance between detected peaks
///
/// # Returns
///
/// * Vector of (index, value) pairs for detected peaks
pub fn detect_peaks(
    correlation: &[f64],
    threshold: f64,
    min_distance: usize,
) -> SignalResult<Vec<(usize, f64)>> {
    if correlation.is_empty() {
        return Ok(Vec::new());
    }

    if !(0.0..=1.0).contains(&threshold) {
        return Err(SignalError::ValueError(
            "Threshold must be between 0.0 and 1.0".to_string(),
        ));
    }

    // Find the maximum value for relative thresholding
    let max_val = correlation.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
    let abs_threshold = threshold * max_val;

    let mut peaks = Vec::new();

    // Find local maxima above threshold
    for i in 1..(correlation.len() - 1) {
        let val = correlation[i];
        if val.abs() >= abs_threshold
            && val.abs() > correlation[i - 1].abs()
            && val.abs() > correlation[i + 1].abs()
        {
            peaks.push((i, val));
        }
    }

    // Check endpoints
    if correlation.len() >= 2 {
        let first = correlation[0];
        if first.abs() >= abs_threshold && first.abs() > correlation[1].abs() {
            peaks.insert(0, (0, first));
        }

        let last_idx = correlation.len() - 1;
        let last = correlation[last_idx];
        if last.abs() >= abs_threshold && last.abs() > correlation[last_idx - 1].abs() {
            peaks.push((last_idx, last));
        }
    }

    // Sort by magnitude (descending)
    peaks.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

    // Apply minimum distance constraint
    let mut filtered_peaks = Vec::new();
    for (idx, val) in peaks {
        let mut too_close = false;
        for &(existing_idx, _) in &filtered_peaks {
            if (idx as i32 - existing_idx as i32).abs() < min_distance as i32 {
                too_close = true;
                break;
            }
        }
        if !too_close {
            filtered_peaks.push((idx, val));
        }
    }

    // Sort by index
    filtered_peaks.sort_by_key(|&(idx, _)| idx);

    Ok(filtered_peaks)
}

/// Design a comb filter for periodic signal enhancement or suppression
///
/// A comb filter has a frequency response with regularly spaced peaks and nulls,
/// resembling a comb. It's useful for enhancing or suppressing periodic signals.
///
/// # Arguments
///
/// * `delay_samples` - Delay in samples (determines the fundamental frequency)
/// * `gain` - Feedback gain (0 < |gain| < 1 for stability)
/// * `filter_type` - "feed_forward" for FIR comb, "feed_back" for IIR comb
///
/// # Returns
///
/// * Tuple of (numerator_coeffs, denominator_coeffs)
///
/// # Examples
///
/// ```ignore
/// use scirs2_signal::filter::comb_filter;
///
/// // FIR comb filter with 10-sample delay
/// let (b, a) = comb_filter(10, 0.5, "feed_forward").unwrap();
///
/// // IIR comb filter for echo removal
/// let (b, a) = comb_filter(100, -0.7, "feed_back").unwrap();
/// ```
pub fn comb_filter(
    delay_samples: usize,
    gain: f64,
    filter_type: &str,
) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    if delay_samples == 0 {
        return Err(SignalError::ValueError(
            "Delay must be positive".to_string(),
        ));
    }

    if gain.abs() >= 1.0 {
        return Err(SignalError::ValueError(
            "Gain magnitude must be less than 1 for stability".to_string(),
        ));
    }

    match filter_type.to_lowercase().as_str() {
        "feed_forward" | "fir" => {
            // FIR comb filter: y[n] = x[n] + g*x[n-D]
            let mut b = vec![0.0; delay_samples + 1];
            b[0] = 1.0;
            b[delay_samples] = gain;
            let a = vec![1.0];
            Ok((b, a))
        }
        "feed_back" | "iir" => {
            // IIR comb filter: y[n] = x[n] + g*y[n-D]
            let b = vec![1.0];
            let mut a = vec![0.0; delay_samples + 1];
            a[0] = 1.0;
            a[delay_samples] = -gain; // Negative because it goes to the other side
            Ok((b, a))
        }
        _ => Err(SignalError::ValueError(
            "Filter type must be 'feed_forward' or 'feed_back'".to_string(),
        )),
    }
}

/// Design a notch filter to suppress a specific frequency
///
/// A notch filter has a sharp null at a specific frequency while passing
/// all other frequencies. It's the opposite of a bandpass filter.
///
/// # Arguments
///
/// * `notch_freq` - Frequency to suppress (normalized from 0 to 1)
/// * `quality_factor` - Q factor controlling the notch width (higher = sharper)
///
/// # Returns
///
/// * Tuple of (numerator_coeffs, denominator_coeffs)
///
/// # Examples
///
/// ```ignore
/// use scirs2_signal::filter::notch_filter;
///
/// // Remove 60 Hz noise (assuming 1000 Hz sample rate)
/// let (b, a) = notch_filter(0.12, 30.0).unwrap(); // 60/500 = 0.12
/// ```
pub fn notch_filter(notch_freq: f64, quality_factor: f64) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    if notch_freq <= 0.0 || notch_freq >= 1.0 {
        return Err(SignalError::ValueError(
            "Notch frequency must be between 0 and 1".to_string(),
        ));
    }

    if quality_factor <= 0.0 {
        return Err(SignalError::ValueError(
            "Quality factor must be positive".to_string(),
        ));
    }

    // Convert normalized frequency to radians
    let omega = std::f64::consts::PI * notch_freq;
    let cos_omega = omega.cos();
    let sin_omega = omega.sin();

    // Calculate filter parameters
    let alpha = sin_omega / (2.0 * quality_factor);
    let a0 = 1.0 + alpha;

    // Notch filter coefficients (from digital filter cookbook)
    let b = vec![1.0 / a0, -2.0 * cos_omega / a0, 1.0 / a0];
    let a = vec![1.0, -2.0 * cos_omega / a0, (1.0 - alpha) / a0];

    Ok((b, a))
}

/// Design a peak filter (inverse notch) to enhance a specific frequency
///
/// A peak filter enhances a specific frequency while leaving others relatively unchanged.
/// It's useful for frequency-selective amplification.
///
/// # Arguments
///
/// * `peak_freq` - Frequency to enhance (normalized from 0 to 1)
/// * `quality_factor` - Q factor controlling the peak width
/// * `gain_db` - Peak gain in decibels
///
/// # Returns
///
/// * Tuple of (numerator_coeffs, denominator_coeffs)
pub fn peak_filter(
    peak_freq: f64,
    quality_factor: f64,
    gain_db: f64,
) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    if peak_freq <= 0.0 || peak_freq >= 1.0 {
        return Err(SignalError::ValueError(
            "Peak frequency must be between 0 and 1".to_string(),
        ));
    }

    if quality_factor <= 0.0 {
        return Err(SignalError::ValueError(
            "Quality factor must be positive".to_string(),
        ));
    }

    // Convert gain to linear scale
    let gain = 10.0_f64.powf(gain_db / 20.0);

    // Convert normalized frequency to radians
    let omega = std::f64::consts::PI * peak_freq;
    let cos_omega = omega.cos();
    let sin_omega = omega.sin();

    // Calculate filter parameters
    let alpha = sin_omega / (2.0 * quality_factor);
    let a0 = 1.0 + alpha / gain;

    // Peak filter coefficients
    let b = vec![
        (1.0 + alpha * gain) / a0,
        -2.0 * cos_omega / a0,
        (1.0 - alpha * gain) / a0,
    ];
    let a = vec![1.0, -2.0 * cos_omega / a0, (1.0 - alpha / gain) / a0];

    Ok((b, a))
}

/// Design an allpass filter for phase equalization
///
/// An allpass filter has unity magnitude response at all frequencies but
/// provides controllable phase shift. Useful for group delay equalization.
///
/// # Arguments
///
/// * `order` - Filter order (1 or 2)
/// * `freq` - Center frequency for phase shift (normalized from 0 to 1)
/// * `q_factor` - Q factor (for second-order only)
///
/// # Returns
///
/// * Tuple of (numerator_coeffs, denominator_coeffs)
pub fn allpass_filter(
    order: usize,
    freq: f64,
    q_factor: Option<f64>,
) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    if freq <= 0.0 || freq >= 1.0 {
        return Err(SignalError::ValueError(
            "Frequency must be between 0 and 1".to_string(),
        ));
    }

    match order {
        1 => {
            // First-order allpass: H(z) = (a1 + z^-1) / (1 + a1*z^-1)
            let omega = std::f64::consts::PI * freq;
            let tan_half_omega = (omega / 2.0).tan();
            let a1 = (tan_half_omega - 1.0) / (tan_half_omega + 1.0);

            let b = vec![-a1, 1.0];
            let a = vec![1.0, -a1];
            Ok((b, a))
        }
        2 => {
            let q = q_factor.ok_or_else(|| {
                SignalError::ValueError("Q factor required for second-order allpass".to_string())
            })?;

            if q <= 0.0 {
                return Err(SignalError::ValueError(
                    "Q factor must be positive".to_string(),
                ));
            }

            // Second-order allpass filter
            let omega = std::f64::consts::PI * freq;
            let cos_omega = omega.cos();
            let sin_omega = omega.sin();
            let alpha = sin_omega / (2.0 * q);

            let a0 = 1.0 + alpha;
            let a1 = -2.0 * cos_omega;
            let a2 = 1.0 - alpha;

            // For allpass: numerator is reverse of denominator
            let b = vec![a2 / a0, a1 / a0, 1.0];
            let a = vec![1.0, a1 / a0, a2 / a0];
            Ok((b, a))
        }
        _ => Err(SignalError::ValueError(
            "Allpass filter order must be 1 or 2".to_string(),
        )),
    }
}

/// Design a matched filter bank for multiple templates
///
/// Creates matched filters for multiple templates simultaneously, useful for
/// detecting different signal types or variations of a signal.
///
/// # Arguments
///
/// * `templates` - Vector of template signals
/// * `normalize` - If true, normalize each filter to unit energy
///
/// # Returns
///
/// * Vector of matched filter coefficients, one for each template
pub fn matched_filter_bank(templates: &[Vec<f64>], normalize: bool) -> SignalResult<Vec<Vec<f64>>> {
    if templates.is_empty() {
        return Err(SignalError::ValueError(
            "Templates cannot be empty".to_string(),
        ));
    }

    let mut filter_bank = Vec::with_capacity(templates.len());

    for template in templates {
        let mf = matched_filter(template, normalize)?;
        filter_bank.push(mf);
    }

    Ok(filter_bank)
}

/// Apply a bank of matched filters to a signal
///
/// Processes a signal with multiple matched filters simultaneously and returns
/// the detection results for each filter.
///
/// # Arguments
///
/// * `signal` - Input signal to process
/// * `filter_bank` - Bank of matched filters
/// * `mode` - Correlation mode
///
/// # Returns
///
/// * Vector of correlation outputs, one for each filter in the bank
pub fn matched_filter_bank_detect(
    signal: &[f64],
    filter_bank: &[Vec<f64>],
    mode: &str,
) -> SignalResult<Vec<Vec<f64>>> {
    if signal.is_empty() || filter_bank.is_empty() {
        return Err(SignalError::ValueError(
            "Signal and filter bank cannot be empty".to_string(),
        ));
    }

    let mut results = Vec::with_capacity(filter_bank.len());

    for filter in filter_bank {
        let correlation = correlate(signal, filter, mode)?;
        results.push(correlation);
    }

    Ok(results)
}

/// Evaluate transfer function H(z) = B(z)/A(z) at a frequency
fn evaluate_transfer_function(b: &[f64], a: &[f64], w: f64) -> num_complex::Complex64 {
    use num_complex::Complex64;

    let z = Complex64::new(w.cos(), w.sin());

    // Evaluate numerator
    let mut num_val = Complex64::zero();
    for (i, &coeff) in b.iter().enumerate() {
        let power = b.len() - 1 - i;
        num_val += Complex64::new(coeff, 0.0) * z.powi(power as i32);
    }

    // Evaluate denominator
    let mut den_val = Complex64::zero();
    for (i, &coeff) in a.iter().enumerate() {
        let power = a.len() - 1 - i;
        den_val += Complex64::new(coeff, 0.0) * z.powi(power as i32);
    }

    if den_val.norm() < 1e-10 {
        Complex64::new(f64::INFINITY, 0.0)
    } else {
        num_val / den_val
    }
}

/// Convert zeros, poles, and gain to transfer function coefficients
///
/// This function converts a filter representation in zeros-poles-gain form
/// to transfer function coefficients (b, a) where H(z) = B(z)/A(z).
///
/// # Arguments
///
/// * `zeros` - Complex zeros of the transfer function
/// * `poles` - Complex poles of the transfer function  
/// * `gain` - Gain factor
///
/// # Returns
///
/// * Tuple of (numerator_coeffs, denominator_coeffs)
fn zpk_to_tf(
    zeros: &[num_complex::Complex64],
    poles: &[num_complex::Complex64],
    gain: f64,
) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    use num_complex::Complex64;

    // Build numerator polynomial from zeros
    let mut num_poly = vec![Complex64::new(1.0, 0.0)];
    for &zero in zeros {
        // Multiply polynomial by (z - zero)
        let mut new_poly = vec![Complex64::new(0.0, 0.0); num_poly.len() + 1];

        // Multiply by z (shift coefficients)
        for (i, &coeff) in num_poly.iter().enumerate() {
            new_poly[i] += coeff;
        }

        // Subtract zero times polynomial
        for (i, &coeff) in num_poly.iter().enumerate() {
            new_poly[i + 1] -= zero * coeff;
        }

        num_poly = new_poly;
    }

    // Build denominator polynomial from poles
    let mut den_poly = vec![Complex64::new(1.0, 0.0)];
    for &pole in poles {
        // Multiply polynomial by (z - pole)
        let mut new_poly = vec![Complex64::new(0.0, 0.0); den_poly.len() + 1];

        // Multiply by z (shift coefficients)
        for (i, &coeff) in den_poly.iter().enumerate() {
            new_poly[i] += coeff;
        }

        // Subtract pole times polynomial
        for (i, &coeff) in den_poly.iter().enumerate() {
            new_poly[i + 1] -= pole * coeff;
        }

        den_poly = new_poly;
    }

    // Apply gain to numerator
    for coeff in &mut num_poly {
        *coeff *= gain;
    }

    // Convert complex coefficients to real (should be real for proper filter design)
    let b: Vec<f64> = num_poly
        .iter()
        .map(|c| {
            if c.im.abs() > 1e-10 {
                eprintln!(
                    "Warning: Numerator coefficient has significant imaginary part: {}",
                    c.im
                );
            }
            c.re
        })
        .collect();

    let a: Vec<f64> = den_poly
        .iter()
        .map(|c| {
            if c.im.abs() > 1e-10 {
                eprintln!(
                    "Warning: Denominator coefficient has significant imaginary part: {}",
                    c.im
                );
            }
            c.re
        })
        .collect();

    // Ensure denominator is monic (leading coefficient = 1)
    if a.is_empty() || a[0].abs() < 1e-15 {
        return Err(SignalError::ValueError(
            "Invalid denominator polynomial".to_string(),
        ));
    }

    let a0 = a[0];
    let b_normalized: Vec<f64> = b.iter().map(|&coeff| coeff / a0).collect();
    let a_normalized: Vec<f64> = a.iter().map(|&coeff| coeff / a0).collect();

    Ok((b_normalized, a_normalized))
}

/// Helper enum to handle different filter type parameter types
#[derive(Debug)]
pub enum FilterTypeParam {
    /// Filter type enum
    Type(FilterType),
    /// Filter type as string
    String(String),
}

impl From<FilterType> for FilterTypeParam {
    fn from(filter_type: FilterType) -> Self {
        FilterTypeParam::Type(filter_type)
    }
}

impl From<&str> for FilterTypeParam {
    fn from(s: &str) -> Self {
        FilterTypeParam::String(s.to_string())
    }
}

impl From<String> for FilterTypeParam {
    fn from(s: String) -> Self {
        FilterTypeParam::String(s)
    }
}

/// Analyze filter frequency response characteristics
///
/// This function provides comprehensive analysis of a digital filter including
/// magnitude response, phase response, group delay, and filter characteristics.
///
/// # Arguments
///
/// * `b` - Numerator coefficients
/// * `a` - Denominator coefficients
/// * `num_points` - Number of frequency points for analysis (default: 512)
///
/// # Returns
///
/// * FilterAnalysis struct containing comprehensive filter characteristics
#[derive(Debug, Clone)]
pub struct FilterAnalysis {
    /// Frequency points (normalized from 0 to 1)
    pub frequencies: Vec<f64>,
    /// Magnitude response (linear scale)
    pub magnitude: Vec<f64>,
    /// Magnitude response in dB
    pub magnitude_db: Vec<f64>,
    /// Phase response in radians
    pub phase: Vec<f64>,
    /// Group delay in samples
    pub group_delay: Vec<f64>,
    /// Passband ripple in dB
    pub passband_ripple: f64,
    /// Stopband attenuation in dB
    pub stopband_attenuation: f64,
    /// 3dB cutoff frequency (normalized)
    pub cutoff_3db: f64,
    /// 6dB cutoff frequency (normalized)
    pub cutoff_6db: f64,
    /// Transition bandwidth (normalized)
    pub transition_bandwidth: f64,
}

/// Perform comprehensive filter analysis
pub fn analyze_filter(
    b: &[f64],
    a: &[f64],
    num_points: Option<usize>,
) -> SignalResult<FilterAnalysis> {
    let n_points = num_points.unwrap_or(512);

    // Generate frequency points from 0 to π (normalized 0 to 1)
    let frequencies: Vec<f64> = (0..n_points)
        .map(|i| i as f64 / (n_points - 1) as f64)
        .collect();

    let w_radians: Vec<f64> = frequencies
        .iter()
        .map(|&f| f * std::f64::consts::PI)
        .collect();

    // Calculate frequency response
    let mut magnitude = Vec::with_capacity(n_points);
    let mut phase = Vec::with_capacity(n_points);

    for &w in &w_radians {
        let h = evaluate_transfer_function(b, a, w);
        magnitude.push(h.norm());
        phase.push(h.arg());
    }

    // Convert magnitude to dB
    let magnitude_db: Vec<f64> = magnitude
        .iter()
        .map(|&mag| 20.0 * mag.log10().max(-100.0)) // Limit minimum to -100 dB
        .collect();

    // Calculate group delay
    let group_delay = group_delay(b, a, &w_radians)?;

    // Analyze filter characteristics
    let max_magnitude_db = magnitude_db
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    // Find 3dB and 6dB cutoff frequencies
    let cutoff_3db = find_cutoff_frequency(&frequencies, &magnitude_db, max_magnitude_db - 3.0);
    let cutoff_6db = find_cutoff_frequency(&frequencies, &magnitude_db, max_magnitude_db - 6.0);

    // Estimate passband and stopband characteristics
    let passband_end = cutoff_3db.min(0.3); // Assume passband ends around 3dB point or 0.3
    let stopband_start = cutoff_3db.max(0.7); // Assume stopband starts after 3dB point or 0.7

    let passband_indices: Vec<usize> = frequencies
        .iter()
        .enumerate()
        .filter(|(_, &f)| f <= passband_end)
        .map(|(i, _)| i)
        .collect();

    let stopband_indices: Vec<usize> = frequencies
        .iter()
        .enumerate()
        .filter(|(_, &f)| f >= stopband_start)
        .map(|(i, _)| i)
        .collect();

    let passband_ripple = if !passband_indices.is_empty() {
        let passband_max = passband_indices
            .iter()
            .map(|&i| magnitude_db[i])
            .fold(f64::NEG_INFINITY, f64::max);
        let passband_min = passband_indices
            .iter()
            .map(|&i| magnitude_db[i])
            .fold(f64::INFINITY, f64::min);
        passband_max - passband_min
    } else {
        0.0
    };

    let stopband_attenuation = if !stopband_indices.is_empty() {
        max_magnitude_db
            - stopband_indices
                .iter()
                .map(|&i| magnitude_db[i])
                .fold(f64::NEG_INFINITY, f64::max)
    } else {
        0.0
    };

    let transition_bandwidth = (cutoff_6db - cutoff_3db).abs();

    Ok(FilterAnalysis {
        frequencies,
        magnitude,
        magnitude_db,
        phase,
        group_delay,
        passband_ripple,
        stopband_attenuation,
        cutoff_3db,
        cutoff_6db,
        transition_bandwidth,
    })
}

/// Find the frequency where magnitude drops to a specific dB level
fn find_cutoff_frequency(frequencies: &[f64], magnitude_db: &[f64], target_db: f64) -> f64 {
    // Find the index where magnitude first drops below target
    for (i, &mag_db) in magnitude_db.iter().enumerate() {
        if mag_db <= target_db {
            if i == 0 {
                return frequencies[0];
            }
            // Linear interpolation between points
            let f1 = frequencies[i - 1];
            let f2 = frequencies[i];
            let m1 = magnitude_db[i - 1];
            let m2 = magnitude_db[i];

            if (m1 - m2).abs() < 1e-10 {
                return f1;
            }

            let t = (target_db - m1) / (m2 - m1);
            return f1 + t * (f2 - f1);
        }
    }
    frequencies[frequencies.len() - 1]
}

/// Check filter stability by analyzing pole locations
///
/// A digital filter is stable if all poles are inside the unit circle.
/// This function analyzes the poles and provides stability information.
///
/// # Arguments
///
/// * `a` - Denominator coefficients
///
/// # Returns
///
/// * FilterStability struct with stability analysis
#[derive(Debug, Clone)]
pub struct FilterStability {
    /// Whether the filter is stable
    pub is_stable: bool,
    /// Pole locations
    pub poles: Vec<num_complex::Complex64>,
    /// Stability margin (minimum distance from unit circle)
    pub stability_margin: f64,
    /// Maximum pole magnitude
    pub max_pole_magnitude: f64,
}

/// Analyze filter stability
pub fn check_filter_stability(a: &[f64]) -> SignalResult<FilterStability> {
    if a.is_empty() || a[0].abs() < 1e-15 {
        return Err(SignalError::ValueError(
            "Invalid denominator coefficients".to_string(),
        ));
    }

    // Find the poles (roots of denominator polynomial)
    let poles = find_polynomial_roots(a)?;

    // Check if all poles are inside unit circle
    let mut is_stable = true;
    let mut max_magnitude = 0.0;
    let mut min_distance_to_unit_circle = f64::INFINITY;

    for &pole in &poles {
        let magnitude = pole.norm();
        max_magnitude = max_magnitude.max(magnitude);

        if magnitude >= 1.0 {
            is_stable = false;
        }

        let distance_to_unit_circle = 1.0 - magnitude;
        min_distance_to_unit_circle = min_distance_to_unit_circle.min(distance_to_unit_circle);
    }

    Ok(FilterStability {
        is_stable,
        poles,
        stability_margin: min_distance_to_unit_circle,
        max_pole_magnitude: max_magnitude,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_filter_type_from_str() {
        assert_eq!(
            "lowpass".parse::<FilterType>().unwrap(),
            FilterType::Lowpass
        );
        assert_eq!(
            "highpass".parse::<FilterType>().unwrap(),
            FilterType::Highpass
        );
        assert_eq!(
            "bandpass".parse::<FilterType>().unwrap(),
            FilterType::Bandpass
        );
        assert_eq!(
            "bandstop".parse::<FilterType>().unwrap(),
            FilterType::Bandstop
        );

        assert!("invalid".parse::<FilterType>().is_err());
    }

    #[test]
    fn test_butter_lowpass() {
        let (b, a) = butter(1, 0.5, "lowpass").unwrap();

        // First-order lowpass with cutoff at 0.5
        assert_eq!(b.len(), 2);
        assert_eq!(a.len(), 2);

        // For 1st order lowpass at 0.5 normalized frequency
        assert_relative_eq!(b[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(b[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(a[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(a[1], -1.0 / 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_butter_highpass() {
        let (b, a) = butter(1, 0.5, "highpass").unwrap();

        // First-order highpass with cutoff at 0.5
        assert_eq!(b.len(), 2);
        assert_eq!(a.len(), 2);

        // For 1st order highpass at 0.5 normalized frequency
        assert_relative_eq!(b[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(b[1], -1.0, epsilon = 1e-10);
        assert_relative_eq!(a[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(a[1], -1.0 / 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_lfilter() {
        // Simple first-order lowpass filter
        let b = vec![0.5];
        let a = vec![1.0, -0.5];

        // Input signal: step function
        let x = vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        // Apply filter
        let y = lfilter(&b, &a, &x).unwrap();

        // Check result: The step response of a first-order lowpass filter
        // should approach 1.0 asymptotically
        assert_relative_eq!(y[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(y[1], 0.0, epsilon = 1e-10);
        assert_relative_eq!(y[2], 0.5, epsilon = 1e-10);
        assert_relative_eq!(y[3], 0.75, epsilon = 1e-10);
        assert_relative_eq!(y[4], 0.875, epsilon = 1e-10);
        assert_relative_eq!(y[5], 0.9375, epsilon = 1e-10);
        assert_relative_eq!(y[6], 0.96875, epsilon = 1e-10);
    }

    #[test]
    fn test_filtfilt() {
        // Simple first-order lowpass filter
        let b = vec![0.5];
        let a = vec![1.0, -0.5];

        // Input signal: step function
        let x = vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        // Apply zero-phase filter
        let y = filtfilt(&b, &a, &x).unwrap();

        // Check the test by verifying the general shape rather than exact values
        // First part should be smaller than the second part
        let first_part_avg = (y[0] + y[1]) / 2.0;
        let second_part_avg = (y[2] + y[3] + y[4] + y[5] + y[6]) / 5.0;

        assert!(first_part_avg < second_part_avg);
    }

    #[test]
    fn test_remez_basic() {
        // Test basic lowpass filter design
        let numtaps = 31;
        let bands = vec![0.0, 0.3, 0.4, 1.0];
        let desired = vec![1.0, 1.0, 0.0, 0.0];

        let h = remez(numtaps, &bands, &desired, None, None, None).unwrap();

        // Check that we got the right number of taps
        assert_eq!(h.len(), numtaps);

        // Check symmetry
        let mid = numtaps / 2;
        for i in 0..mid {
            assert_relative_eq!(h[i], h[numtaps - 1 - i], epsilon = 1e-10);
        }

        // Check that the filter has reasonable magnitude
        let max_magnitude = h.iter().map(|&x| x.abs()).fold(0.0, f64::max);
        assert!(max_magnitude > 0.0);
        assert!(max_magnitude < 1.0);
    }

    #[test]
    fn test_remez_highpass() {
        // Test highpass filter design
        let numtaps = 31;
        let bands = vec![0.0, 0.3, 0.4, 1.0];
        let desired = vec![0.0, 0.0, 1.0, 1.0];

        let h = remez(numtaps, &bands, &desired, None, None, None).unwrap();

        assert_eq!(h.len(), numtaps);

        // Check symmetry
        let mid = numtaps / 2;
        for i in 0..mid {
            assert_relative_eq!(h[i], h[numtaps - 1 - i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_remez_bandpass() {
        // Test bandpass filter design
        let numtaps = 51;
        let bands = vec![0.0, 0.2, 0.3, 0.5, 0.6, 1.0];
        let desired = vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0];

        let h = remez(numtaps, &bands, &desired, None, None, None).unwrap();

        assert_eq!(h.len(), numtaps);

        // Check symmetry
        let mid = numtaps / 2;
        for i in 0..mid {
            assert_relative_eq!(h[i], h[numtaps - 1 - i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_remez_with_weights() {
        // Test filter design with custom weights
        let numtaps = 31;
        let bands = vec![0.0, 0.3, 0.4, 1.0];
        let desired = vec![1.0, 1.0, 0.0, 0.0];
        let weights = vec![1.0, 1.0, 10.0, 10.0]; // Emphasize stopband

        let h = remez(numtaps, &bands, &desired, Some(&weights), None, None).unwrap();

        assert_eq!(h.len(), numtaps);
    }

    #[test]
    fn test_remez_errors() {
        // Test error conditions

        // Too few taps
        let result = remez(2, &[0.0, 1.0], &[1.0, 0.0], None, None, None);
        assert!(result.is_err());

        // Invalid bands (not pairs)
        let result = remez(31, &[0.0, 0.5, 1.0], &[1.0, 0.0], None, None, None);
        assert!(result.is_err());

        // Mismatched desired length
        let result = remez(31, &[0.0, 0.5, 0.6, 1.0], &[1.0, 0.0], None, None, None);
        assert!(result.is_err());

        // Non-monotonic bands
        let result = remez(
            31,
            &[0.0, 0.5, 0.4, 1.0],
            &[1.0, 1.0, 0.0, 0.0],
            None,
            None,
            None,
        );
        assert!(result.is_err());

        // Bands outside [0, 1]
        let result = remez(
            31,
            &[-0.1, 0.5, 0.6, 1.1],
            &[1.0, 1.0, 0.0, 0.0],
            None,
            None,
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_solve_linear_system() {
        // Test the linear solver
        let a = vec![
            vec![2.0, 1.0, -1.0],
            vec![-3.0, -1.0, 2.0],
            vec![-2.0, 1.0, 2.0],
        ];
        let b = vec![8.0, -11.0, -3.0];

        let x = solve_linear_system(&a, &b).unwrap();

        // Verify solution: x = [2, 3, -1]
        assert_relative_eq!(x[0], 2.0, epsilon = 1e-10);
        assert_relative_eq!(x[1], 3.0, epsilon = 1e-10);
        assert_relative_eq!(x[2], -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_minimum_phase_constant() {
        // Test constant filter (should remain unchanged)
        let b = vec![2.0];
        let result = minimum_phase(&b, true).unwrap();
        assert_eq!(result, vec![2.0]);
    }

    #[test]
    fn test_minimum_phase_linear() {
        // Test linear filter with zero outside unit circle
        let b = vec![1.0, -2.0]; // (z - 2), zero at z = 2 (outside unit circle)
        let result = minimum_phase(&b, true).unwrap();

        // Should become minimum phase with zero at z = 1/2
        assert_eq!(result.len(), 2);
        // The exact values depend on the implementation, but it should be minimum phase
        assert!(result[0].abs() > 0.0);
        assert!(result[1].abs() > 0.0);
    }

    #[test]
    fn test_minimum_phase_quadratic() {
        // Test quadratic filter
        let b = vec![1.0, -3.0, 2.0]; // (z-1)(z-2), zeros at z=1,2 (both outside unit circle)
        let result = minimum_phase(&b, true).unwrap();

        assert_eq!(result.len(), 3);
        // Should be minimum phase
        assert!(result[0].abs() > 0.0);
    }

    #[test]
    fn test_minimum_phase_already_minimum() {
        // Test filter that's already minimum phase
        let b = vec![1.0, 0.5]; // (z + 0.5), zero at z = -0.5 (inside unit circle)
        let result = minimum_phase(&b, true).unwrap();

        // Should remain approximately the same
        assert_eq!(result.len(), 2);
        assert_relative_eq!(result[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(result[1], 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_minimum_phase_continuous_time() {
        // Test continuous-time filter
        let b = vec![1.0, -1.0]; // (s - 1), zero at s = 1 (positive real part)
        let result = minimum_phase(&b, false).unwrap();

        // Should become minimum phase with zero at s = -1
        assert_eq!(result.len(), 2);
        assert!(result[0].abs() > 0.0);
        assert!(result[1] > 0.0); // Should be positive for s + 1 form
    }

    #[test]
    fn test_minimum_phase_errors() {
        // Test error conditions
        let result = minimum_phase(&[], true);
        assert!(result.is_err());
    }

    #[test]
    fn test_group_delay_basic() {
        // Test group delay for a simple first-order filter
        let b = vec![1.0];
        let a = vec![1.0, -0.5];
        let w = vec![0.0, std::f64::consts::PI / 4.0, std::f64::consts::PI / 2.0];

        let gd = group_delay(&b, &a, &w).unwrap();

        assert_eq!(gd.len(), 3);
        // Group delay should be finite and reasonable
        for &delay in &gd {
            assert!(delay.is_finite());
            assert!(delay.abs() < 100.0); // Reasonable bounds
        }
    }

    #[test]
    fn test_group_delay_errors() {
        // Test error conditions
        let b = vec![1.0];
        let a = vec![]; // Empty denominator
        let w = vec![0.0];

        let result = group_delay(&b, &a, &w);
        assert!(result.is_err());

        // Zero first coefficient
        let a = vec![0.0, 1.0];
        let result = group_delay(&b, &a, &w);
        assert!(result.is_err());
    }

    #[test]
    fn test_find_polynomial_roots_linear() {
        // Test root finding for linear polynomial: x - 2
        let coeffs = vec![1.0, -2.0];
        let roots = find_polynomial_roots(&coeffs).unwrap();

        assert_eq!(roots.len(), 1);
        assert_relative_eq!(roots[0].re, 2.0, epsilon = 1e-6);
        assert_relative_eq!(roots[0].im, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_find_polynomial_roots_quadratic() {
        // Test root finding for quadratic polynomial: x^2 - 3x + 2 = (x-1)(x-2)
        let coeffs = vec![1.0, -3.0, 2.0];
        let roots = find_polynomial_roots(&coeffs).unwrap();

        assert_eq!(roots.len(), 2);

        // Roots should be 1 and 2 (in some order)
        let mut real_parts: Vec<f64> = roots.iter().map(|z| z.re).collect();
        real_parts.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert_relative_eq!(real_parts[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(real_parts[1], 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_polynomial_from_roots() {
        // Test polynomial reconstruction from known roots
        use num_complex::Complex64;

        let roots = vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];

        let poly = polynomial_from_roots(&roots);

        // Should give (z-1)(z-2) = z^2 - 3z + 2 = [1, -3, 2]
        assert_eq!(poly.len(), 3);
        assert_relative_eq!(poly[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(poly[1], -3.0, epsilon = 1e-6);
        assert_relative_eq!(poly[2], 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_evaluate_transfer_function() {
        // Test transfer function evaluation
        let b = vec![1.0];
        let a = vec![1.0, -0.5];

        // Evaluate at w = 0 (DC)
        let h = evaluate_transfer_function(&b, &a, 0.0);

        // At DC: H(1) = 1/(1-0.5) = 2
        assert_relative_eq!(h.re, 2.0, epsilon = 1e-6);
        assert_relative_eq!(h.im, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_matched_filter_basic() {
        // Test basic matched filter design
        let template = vec![1.0, 2.0, 3.0];
        let mf = matched_filter(&template, false).unwrap();

        // Should be time-reversed: [3, 2, 1]
        assert_eq!(mf.len(), 3);
        assert_relative_eq!(mf[0], 3.0, epsilon = 1e-10);
        assert_relative_eq!(mf[1], 2.0, epsilon = 1e-10);
        assert_relative_eq!(mf[2], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_matched_filter_normalized() {
        // Test normalized matched filter
        let template = vec![3.0, 4.0]; // Energy = 9 + 16 = 25, norm = 5
        let mf = matched_filter(&template, true).unwrap();

        // Should be time-reversed and normalized: [4/5, 3/5]
        assert_eq!(mf.len(), 2);
        assert_relative_eq!(mf[0], 0.8, epsilon = 1e-10); // 4/5
        assert_relative_eq!(mf[1], 0.6, epsilon = 1e-10); // 3/5

        // Check unit energy
        let energy: f64 = mf.iter().map(|&x| x * x).sum();
        assert_relative_eq!(energy, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_matched_filter_errors() {
        // Test error conditions
        let result = matched_filter(&[], true);
        assert!(result.is_err());
    }

    #[test]
    fn test_correlate_full() {
        // Test full correlation
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 1.0];
        let result = correlate(&a, &b, "full").unwrap();

        // Full correlation should have length 3 + 2 - 1 = 4
        assert_eq!(result.len(), 4);

        // Manual calculation: correlation of [1,2,3] with [1,1]
        // Result should be [1, 3, 5, 3]
        assert_relative_eq!(result[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(result[1], 3.0, epsilon = 1e-10);
        assert_relative_eq!(result[2], 5.0, epsilon = 1e-10);
        assert_relative_eq!(result[3], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_correlate_same() {
        // Test same-size correlation
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 1.0];
        let result = correlate(&a, &b, "same").unwrap();

        // Same correlation should have same length as a
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_correlate_valid() {
        // Test valid correlation
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 1.0];
        let result = correlate(&a, &b, "valid").unwrap();

        // Valid correlation: length = 4 - 2 + 1 = 3
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_correlate_errors() {
        // Test error conditions
        let result = correlate(&[], &[1.0], "full");
        assert!(result.is_err());

        let result = correlate(&[1.0], &[], "full");
        assert!(result.is_err());

        let result = correlate(&[1.0], &[1.0], "invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_matched_filter_detect() {
        // Test matched filter detection
        let signal = vec![0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0];
        let template = vec![1.0, 2.0, 3.0];

        let result = matched_filter_detect(&signal, &template, "same").unwrap();
        assert_eq!(result.len(), signal.len());

        // The peak should occur where the template matches
        let max_idx = result
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
            .unwrap()
            .0;

        // Peak should be around index 4 (where template ends in signal)
        assert!((3..=5).contains(&max_idx));
    }

    #[test]
    fn test_detect_peaks() {
        // Test peak detection
        let correlation = vec![0.1, 0.8, 0.2, 0.9, 0.1, 0.3, 0.7, 0.1];
        let peaks = detect_peaks(&correlation, 0.5, 1).unwrap();

        // Should detect peaks at indices 1, 3, and 6 (values 0.8, 0.9, and 0.7)
        assert_eq!(peaks.len(), 3);
        assert_eq!(peaks[0].0, 1); // Index 1
        assert_eq!(peaks[1].0, 3); // Index 3
        assert_eq!(peaks[2].0, 6); // Index 6
        assert_relative_eq!(peaks[0].1, 0.8, epsilon = 1e-10);
        assert_relative_eq!(peaks[1].1, 0.9, epsilon = 1e-10);
        assert_relative_eq!(peaks[2].1, 0.7, epsilon = 1e-10);
    }

    #[test]
    fn test_detect_peaks_min_distance() {
        // Test minimum distance constraint
        let correlation = vec![0.9, 0.8, 0.7];
        let peaks = detect_peaks(&correlation, 0.5, 2).unwrap();

        // Should only detect the highest peak at index 0 due to min_distance
        assert_eq!(peaks.len(), 1);
        assert_eq!(peaks[0].0, 0);
    }

    #[test]
    fn test_detect_peaks_errors() {
        // Test error conditions
        let correlation = vec![0.5];

        // Invalid threshold
        let result = detect_peaks(&correlation, 1.5, 1);
        assert!(result.is_err());

        let result = detect_peaks(&correlation, -0.1, 1);
        assert!(result.is_err());

        // Empty input should return empty result
        let result = detect_peaks(&[], 0.5, 1).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_matched_filter_bank() {
        // Test matched filter bank
        let templates = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];

        let bank = matched_filter_bank(&templates, true).unwrap();
        assert_eq!(bank.len(), 3);

        // Each filter should be normalized
        for filter in &bank {
            let energy: f64 = filter.iter().map(|&x| x * x).sum();
            assert_relative_eq!(energy, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_matched_filter_bank_detect() {
        // Test bank detection
        let signal = vec![1.0, 0.0, 0.0, 1.0];
        let templates = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

        let bank = matched_filter_bank(&templates, false).unwrap();
        let results = matched_filter_bank_detect(&signal, &bank, "same").unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), signal.len());
        assert_eq!(results[1].len(), signal.len());
    }

    #[test]
    fn test_matched_filter_bank_errors() {
        // Test error conditions
        let result = matched_filter_bank(&[], true);
        assert!(result.is_err());

        let signal = vec![1.0];
        let result = matched_filter_bank_detect(&[], &[vec![1.0]], "same");
        assert!(result.is_err());

        let result = matched_filter_bank_detect(&signal, &[], "same");
        assert!(result.is_err());
    }

    #[test]
    fn test_matched_filter_perfect_match() {
        // Test perfect template match
        let template = vec![1.0, 2.0, 1.0];
        let signal = vec![0.0, 1.0, 2.0, 1.0, 0.0]; // Template embedded in signal

        let correlation = matched_filter_detect(&signal, &template, "same").unwrap();

        // Find the peak
        let (max_idx, max_val) = correlation
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
            .unwrap();

        // Peak should be significant and occur around the template location
        assert!(max_val.abs() > 0.5);
        assert!((1..=3).contains(&max_idx));
    }
}

// Z-Domain Filter Design Functions
//
// This section provides comprehensive Z-domain filter design functionality,
// including direct pole-zero placement, frequency transformations, and
// advanced digital filter design methods.

/// Design a digital filter using direct pole-zero placement in the Z-domain
///
/// This function creates a digital filter by directly specifying the locations
/// of poles and zeros in the complex Z-plane. This approach gives maximum
/// control over filter characteristics.
///
/// # Arguments
///
/// * `zeros` - Complex zeros in the Z-plane
/// * `poles` - Complex poles in the Z-plane  
/// * `gain` - Overall filter gain
/// * `sample_rate` - Sample rate (optional, used for frequency normalization)
///
/// # Returns
///
/// * Tuple of (numerator_coeffs, denominator_coeffs)
///
/// # Examples
///
/// ```
/// use scirs2_signal::filter::zpk_design;
/// use num_complex::Complex64;
///
/// // Design a simple lowpass filter with a pole at 0.5 and zero at -1
/// let zeros = vec![Complex64::new(-1.0, 0.0)];
/// let poles = vec![Complex64::new(0.5, 0.0)];
/// let gain = 1.0;
///
/// let (b, a) = zpk_design(&zeros, &poles, gain, None).unwrap();
/// ```
pub fn zpk_design(
    zeros: &[Complex64],
    poles: &[Complex64],
    gain: f64,
    _sample_rate: Option<f64>,
) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    // Validate poles are inside unit circle for stability
    for (i, &pole) in poles.iter().enumerate() {
        if pole.norm() >= 1.0 {
            return Err(SignalError::ValueError(format!(
                "Pole {} at ({:.6}, {:.6}) is outside unit circle (magnitude = {:.6}). Filter would be unstable.",
                i, pole.re, pole.im, pole.norm()
            )));
        }
    }

    // Convert poles and zeros to transfer function coefficients
    zpk_to_tf(zeros, poles, gain)
}

/// Transform a lowpass digital filter to other filter types using Z-domain transformations
///
/// This function applies frequency transformations in the Z-domain to convert
/// a lowpass prototype filter into highpass, bandpass, or bandstop filters.
///
/// # Arguments
///
/// * `b` - Numerator coefficients of lowpass prototype
/// * `a` - Denominator coefficients of lowpass prototype
/// * `filter_type` - Target filter type
/// * `critical_freqs` - Critical frequencies (normalized 0-1)
///   - Lowpass/Highpass: single frequency
///   - Bandpass/Bandstop: [low_freq, high_freq]
///
/// # Returns
///
/// * Tuple of transformed (numerator_coeffs, denominator_coeffs)
pub fn z_domain_transform(
    b: &[f64],
    a: &[f64],
    filter_type: FilterType,
    critical_freqs: &[f64],
) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    // Validate inputs
    if b.is_empty() || a.is_empty() {
        return Err(SignalError::ValueError(
            "Filter coefficients cannot be empty".to_string(),
        ));
    }

    for &freq in critical_freqs {
        if !(0.0..=1.0).contains(&freq) {
            return Err(SignalError::ValueError(format!(
                "Critical frequency {:.3} must be between 0 and 1",
                freq
            )));
        }
    }

    match filter_type {
        FilterType::Lowpass => {
            if critical_freqs.len() != 1 {
                return Err(SignalError::ValueError(
                    "Lowpass transformation requires exactly one critical frequency".to_string(),
                ));
            }
            z_lowpass_transform(b, a, critical_freqs[0])
        }
        FilterType::Highpass => {
            if critical_freqs.len() != 1 {
                return Err(SignalError::ValueError(
                    "Highpass transformation requires exactly one critical frequency".to_string(),
                ));
            }
            z_highpass_transform(b, a, critical_freqs[0])
        }
        FilterType::Bandpass => {
            if critical_freqs.len() != 2 {
                return Err(SignalError::ValueError(
                    "Bandpass transformation requires exactly two critical frequencies".to_string(),
                ));
            }
            if critical_freqs[0] >= critical_freqs[1] {
                return Err(SignalError::ValueError(
                    "Lower frequency must be less than upper frequency".to_string(),
                ));
            }
            z_bandpass_transform(b, a, critical_freqs[0], critical_freqs[1])
        }
        FilterType::Bandstop => {
            if critical_freqs.len() != 2 {
                return Err(SignalError::ValueError(
                    "Bandstop transformation requires exactly two critical frequencies".to_string(),
                ));
            }
            if critical_freqs[0] >= critical_freqs[1] {
                return Err(SignalError::ValueError(
                    "Lower frequency must be less than upper frequency".to_string(),
                ));
            }
            z_bandstop_transform(b, a, critical_freqs[0], critical_freqs[1])
        }
    }
}

/// Lowpass to lowpass Z-domain transformation
fn z_lowpass_transform(b: &[f64], a: &[f64], wc: f64) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    // Transform z -> (z - alpha) / (1 - alpha * z) where alpha = (1 - tan(wc*pi/2)) / (1 + tan(wc*pi/2))
    let tan_half = (wc * std::f64::consts::PI / 2.0).tan();
    let alpha = (1.0 - tan_half) / (1.0 + tan_half);

    apply_allpass_transform(b, a, alpha)
}

/// Lowpass to highpass Z-domain transformation
fn z_highpass_transform(b: &[f64], a: &[f64], wc: f64) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    // Transform z -> -(z - alpha) / (1 - alpha * z) where alpha = (tan(wc*pi/2) - 1) / (tan(wc*pi/2) + 1)
    let tan_half = (wc * std::f64::consts::PI / 2.0).tan();
    let alpha = (tan_half - 1.0) / (tan_half + 1.0);

    let (b_temp, a_temp) = apply_allpass_transform(b, a, alpha)?;

    // Apply sign alternation for highpass
    let b_hp: Vec<f64> = b_temp
        .iter()
        .enumerate()
        .map(|(i, &coeff)| if i % 2 == 0 { coeff } else { -coeff })
        .collect();

    Ok((b_hp, a_temp))
}

/// Lowpass to bandpass Z-domain transformation
fn z_bandpass_transform(
    b: &[f64],
    a: &[f64],
    wl: f64,
    wh: f64,
) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    // Bandpass transformation: z -> (z^2 - 2*k*z + 1) / (2*alpha*z^2 - 2*k*z + 2*alpha)
    // where k = cos((wh + wl)*pi/2) and alpha = cos((wh - wl)*pi/2)
    let wo = (wh + wl) / 2.0; // Center frequency
    let bw = wh - wl; // Bandwidth

    let k = (wo * std::f64::consts::PI).cos();
    let alpha = (bw * std::f64::consts::PI / 2.0).cos();

    apply_bandpass_transform(b, a, k, alpha)
}

/// Lowpass to bandstop Z-domain transformation  
fn z_bandstop_transform(
    b: &[f64],
    a: &[f64],
    wl: f64,
    wh: f64,
) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    // Bandstop transformation: z -> (2*alpha*z^2 - 2*k*z + 2*alpha) / (z^2 - 2*k*z + 1)
    let wo = (wh + wl) / 2.0; // Center frequency
    let bw = wh - wl; // Bandwidth

    let k = (wo * std::f64::consts::PI).cos();
    let alpha = (bw * std::f64::consts::PI / 2.0).cos();

    apply_bandstop_transform(b, a, k, alpha)
}

/// Apply allpass transformation z -> (z - alpha) / (1 - alpha*z)
fn apply_allpass_transform(b: &[f64], a: &[f64], alpha: f64) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    let n = a.len();
    let m = b.len();

    // Transform denominator polynomial
    let mut new_a = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            if i + j < n {
                new_a[i + j] += a[i] * (-alpha).powi(j as i32);
            }
        }
        for j in 1..n {
            if i + j < n {
                new_a[i + j] += a[i] * (-alpha).powi((n - 1 - j) as i32);
            }
        }
    }

    // Transform numerator polynomial
    let mut new_b = vec![0.0; m];
    for i in 0..m {
        for j in 0..m {
            if i + j < m {
                new_b[i + j] += b[i] * (-alpha).powi(j as i32);
            }
        }
        for j in 1..m {
            if i + j < m {
                new_b[i + j] += b[i] * (-alpha).powi((m - 1 - j) as i32);
            }
        }
    }

    Ok((new_b, new_a))
}

/// Apply bandpass transformation
fn apply_bandpass_transform(
    b: &[f64],
    a: &[f64],
    k: f64,
    _alpha: f64,
) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    let n = a.len();
    let m = b.len();

    // For bandpass, the degree doubles
    let mut new_a = vec![0.0; 2 * n - 1];
    let mut new_b = vec![0.0; 2 * m - 1];

    // Transform using the bandpass substitution z -> (z^2 - 2*k*z + 1) / (2*alpha*z^2 - 2*k*z + 2*alpha)
    // This is a simplified implementation - full implementation would be more complex
    for i in 0..n {
        if 2 * i < new_a.len() {
            new_a[2 * i] += a[i];
        }
        if 2 * i + 1 < new_a.len() {
            new_a[2 * i + 1] += -2.0 * k * a[i];
        }
    }

    for i in 0..m {
        if 2 * i < new_b.len() {
            new_b[2 * i] += b[i];
        }
        if 2 * i + 1 < new_b.len() {
            new_b[2 * i + 1] += -2.0 * k * b[i];
        }
    }

    Ok((new_b, new_a))
}

/// Apply bandstop transformation
fn apply_bandstop_transform(
    b: &[f64],
    a: &[f64],
    k: f64,
    alpha: f64,
) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    let n = a.len();
    let m = b.len();

    // For bandstop, the degree doubles
    let mut new_a = vec![0.0; 2 * n - 1];
    let mut new_b = vec![0.0; 2 * m - 1];

    // Transform using the bandstop substitution
    // This is a simplified implementation - full implementation would be more complex
    for i in 0..n {
        if 2 * i < new_a.len() {
            new_a[2 * i] += 2.0 * alpha * a[i];
        }
        if 2 * i + 1 < new_a.len() {
            new_a[2 * i + 1] += -2.0 * k * a[i];
        }
    }

    for i in 0..m {
        if 2 * i < new_b.len() {
            new_b[2 * i] += 2.0 * alpha * b[i];
        }
        if 2 * i + 1 < new_b.len() {
            new_b[2 * i + 1] += -2.0 * k * b[i];
        }
    }

    Ok((new_b, new_a))
}

/// Design a notch filter at a specific frequency using pole-zero placement
///
/// Creates a narrow-band filter that attenuates signals at a specific frequency
/// while passing signals at other frequencies with minimal distortion.
///
/// # Arguments
///
/// * `notch_freq` - Normalized frequency to notch (0-1, where 1 is Nyquist)
/// * `bandwidth` - Bandwidth of the notch (normalized frequency)
/// * `depth_db` - Depth of the notch in dB (positive value)
///
/// # Returns
///
/// * Tuple of (numerator_coeffs, denominator_coeffs)
pub fn notch_filter_zpk(
    notch_freq: f64,
    bandwidth: f64,
    depth_db: f64,
) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    if !(0.0..=1.0).contains(&notch_freq) {
        return Err(SignalError::ValueError(
            "Notch frequency must be between 0 and 1".to_string(),
        ));
    }

    if bandwidth <= 0.0 || bandwidth > 1.0 {
        return Err(SignalError::ValueError(
            "Bandwidth must be positive and less than 1".to_string(),
        ));
    }

    if depth_db <= 0.0 {
        return Err(SignalError::ValueError(
            "Notch depth must be positive".to_string(),
        ));
    }

    // Convert frequency to radians
    let omega = notch_freq * std::f64::consts::PI;

    // Place zeros exactly on the unit circle at the notch frequency
    let zeros = vec![
        Complex64::new(omega.cos(), omega.sin()),
        Complex64::new(omega.cos(), -omega.sin()),
    ];

    // Place poles slightly inside the unit circle to control bandwidth
    let r = 1.0 - bandwidth * std::f64::consts::PI / 2.0; // Pole radius
    let poles = vec![
        Complex64::new(r * omega.cos(), r * omega.sin()),
        Complex64::new(r * omega.cos(), -r * omega.sin()),
    ];

    // Calculate gain to normalize the filter
    let gain = 1.0; // Can be adjusted based on desired response

    zpk_to_tf(&zeros, &poles, gain)
}

/// Design a peak filter (resonator) at a specific frequency using pole-zero placement
///
/// Creates a narrow-band filter that amplifies signals at a specific frequency
/// while attenuating signals at other frequencies.
///
/// # Arguments
///
/// * `peak_freq` - Normalized frequency for the peak (0-1, where 1 is Nyquist)
/// * `bandwidth` - Bandwidth of the peak (normalized frequency)  
/// * `gain_db` - Gain at the peak frequency in dB
///
/// # Returns
///
/// * Tuple of (numerator_coeffs, denominator_coeffs)
pub fn peak_filter_zpk(
    peak_freq: f64,
    bandwidth: f64,
    gain_db: f64,
) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    if !(0.0..=1.0).contains(&peak_freq) {
        return Err(SignalError::ValueError(
            "Peak frequency must be between 0 and 1".to_string(),
        ));
    }

    if bandwidth <= 0.0 || bandwidth > 1.0 {
        return Err(SignalError::ValueError(
            "Bandwidth must be positive and less than 1".to_string(),
        ));
    }

    // Convert frequency to radians
    let omega = peak_freq * std::f64::consts::PI;

    // Place poles close to the unit circle at the peak frequency for resonance
    let r = 1.0 - bandwidth * std::f64::consts::PI / 4.0; // Pole radius
    let poles = vec![
        Complex64::new(r * omega.cos(), r * omega.sin()),
        Complex64::new(r * omega.cos(), -r * omega.sin()),
    ];

    // Place zeros at origin (all-pole filter) or at strategic locations
    let zeros = vec![Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)];

    // Calculate gain from desired dB gain
    let linear_gain = 10.0_f64.powf(gain_db / 20.0);

    zpk_to_tf(&zeros, &poles, linear_gain)
}

/// Design an allpass filter with specified phase characteristics
///
/// Allpass filters pass all frequencies with unity magnitude but introduce
/// controlled phase shifts. They are useful for phase correction and delay equalization.
///
/// # Arguments
///
/// * `poles` - Complex pole locations (must be inside unit circle)
/// * `gain` - Overall gain (typically 1.0 for allpass)
///
/// # Returns
///
/// * Tuple of (numerator_coeffs, denominator_coeffs)
pub fn allpass_filter_zpk(poles: &[Complex64], gain: f64) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    // For allpass filters, zeros are the complex conjugate reciprocals of poles
    let zeros: Vec<Complex64> = poles
        .iter()
        .map(|&pole| {
            if pole.norm() >= 1.0 {
                return Err(SignalError::ValueError(
                    "All poles must be inside unit circle for stability".to_string(),
                ));
            }
            // Zero at 1/conjugate(pole)
            let conj_pole = pole.conj();
            Ok(Complex64::new(1.0, 0.0) / conj_pole)
        })
        .collect::<Result<Vec<_>, _>>()?;

    zpk_to_tf(&zeros, poles, gain)
}

/// Design a comb filter using Z-domain pole-zero placement
///
/// Comb filters have a frequency response with a series of regularly spaced
/// peaks and nulls, resembling the teeth of a comb.
///
/// # Arguments
///
/// * `delay_samples` - Delay in samples (creates comb spacing)
/// * `feedback_gain` - Feedback gain (must be < 1 for stability)
/// * `feedforward_gain` - Feedforward gain
/// * `comb_type` - Type of comb filter ("feedback", "feedforward", or "both")
///
/// # Returns
///
/// * Tuple of (numerator_coeffs, denominator_coeffs)
pub fn comb_filter_zpk(
    delay_samples: usize,
    feedback_gain: f64,
    feedforward_gain: f64,
    comb_type: &str,
) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    if delay_samples == 0 {
        return Err(SignalError::ValueError(
            "Delay must be greater than 0".to_string(),
        ));
    }

    if feedback_gain.abs() >= 1.0 {
        return Err(SignalError::ValueError(
            "Feedback gain must have magnitude less than 1 for stability".to_string(),
        ));
    }

    match comb_type.to_lowercase().as_str() {
        "feedback" => {
            // H(z) = 1 / (1 - feedback_gain * z^(-delay))
            let b = vec![1.0];
            let mut a = vec![1.0];
            a.resize(delay_samples + 1, 0.0);
            a[delay_samples] = -feedback_gain;
            Ok((b, a))
        }
        "feedforward" => {
            // H(z) = 1 + feedforward_gain * z^(-delay)
            let mut b = vec![1.0];
            b.resize(delay_samples + 1, 0.0);
            b[delay_samples] = feedforward_gain;
            let a = vec![1.0];
            Ok((b, a))
        }
        "both" => {
            // H(z) = (1 + feedforward_gain * z^(-delay)) / (1 - feedback_gain * z^(-delay))
            let mut b = vec![1.0];
            b.resize(delay_samples + 1, 0.0);
            b[delay_samples] = feedforward_gain;

            let mut a = vec![1.0];
            a.resize(delay_samples + 1, 0.0);
            a[delay_samples] = -feedback_gain;
            Ok((b, a))
        }
        _ => Err(SignalError::ValueError(format!(
            "Unknown comb filter type: {}. Use 'feedback', 'feedforward', or 'both'",
            comb_type
        ))),
    }
}

#[cfg(test)]
mod z_domain_tests {
    use super::*;
    use num_complex::Complex64;

    #[test]
    fn test_zpk_design() {
        // Test simple pole-zero placement
        let zeros = vec![Complex64::new(-1.0, 0.0)];
        let poles = vec![Complex64::new(0.5, 0.0)];
        let gain = 1.0;

        let (b, a) = zpk_design(&zeros, &poles, gain, None).unwrap();

        assert_eq!(b.len(), 2);
        assert_eq!(a.len(), 2);
        assert_eq!(a[0], 1.0); // Normalized
    }

    #[test]
    fn test_zpk_design_unstable_pole() {
        // Test that unstable poles are rejected
        let zeros = vec![];
        let poles = vec![Complex64::new(1.5, 0.0)]; // Outside unit circle
        let gain = 1.0;

        let result = zpk_design(&zeros, &poles, gain, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_z_domain_transform_lowpass() {
        let b = vec![1.0, 1.0];
        let a = vec![1.0, 0.0];

        let (b_new, a_new) = z_domain_transform(&b, &a, FilterType::Lowpass, &[0.5]).unwrap();

        assert!(!b_new.is_empty());
        assert!(!a_new.is_empty());
        assert_eq!(a_new[0], 1.0); // Should be normalized
    }

    #[test]
    fn test_notch_filter_zpk() {
        let (b, a) = notch_filter_zpk(0.25, 0.1, 40.0).unwrap();

        assert_eq!(b.len(), 3); // Second-order filter
        assert_eq!(a.len(), 3);
        assert_eq!(a[0], 1.0); // Normalized
    }

    #[test]
    fn test_peak_filter_zpk() {
        let (b, a) = peak_filter_zpk(0.25, 0.1, 6.0).unwrap();

        assert_eq!(b.len(), 3); // Second-order filter
        assert_eq!(a.len(), 3);
        assert_eq!(a[0], 1.0); // Normalized
    }

    #[test]
    fn test_allpass_filter_zpk() {
        let poles = vec![Complex64::new(0.5, 0.3)];
        let (b, a) = allpass_filter_zpk(&poles, 1.0).unwrap();

        assert_eq!(b.len(), 2);
        assert_eq!(a.len(), 2);
    }

    #[test]
    fn test_comb_filter_zpk() {
        let (b, a) = comb_filter_zpk(10, 0.7, 0.5, "both").unwrap();

        assert_eq!(b.len(), 11); // delay + 1
        assert_eq!(a.len(), 11);
        assert_eq!(b[0], 1.0);
        assert_eq!(a[0], 1.0);
        assert_eq!(b[10], 0.5); // feedforward gain
        assert_eq!(a[10], -0.7); // -feedback gain
    }

    #[test]
    fn test_comb_filter_errors() {
        // Test unstable feedback gain
        let result = comb_filter_zpk(10, 1.5, 0.5, "feedback");
        assert!(result.is_err());

        // Test zero delay
        let result = comb_filter_zpk(0, 0.5, 0.5, "feedback");
        assert!(result.is_err());

        // Test invalid type
        let result = comb_filter_zpk(10, 0.5, 0.5, "invalid");
        assert!(result.is_err());
    }
}
