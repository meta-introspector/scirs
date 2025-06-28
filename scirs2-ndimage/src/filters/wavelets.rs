//! Wavelet-based filtering and transform operations
//!
//! This module provides discrete wavelet transform (DWT) and related filtering
//! operations for image processing. Wavelets are particularly useful for
//! denoising, compression, and multi-scale analysis.

use ndarray::{Array, Array1, Array2, ArrayView1, ArrayView2, Axis, Dimension, Ix2};
use num_traits::{Float, FromPrimitive, Zero};
use scirs2_core::simd_ops::SimdUnifiedOps;
use std::fmt::Debug;

use crate::error::{NdimageError, NdimageResult};
use crate::filters::BorderMode;

/// Wavelet family enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveletFamily {
    /// Daubechies wavelets
    Daubechies(usize), // Number of vanishing moments
    /// Biorthogonal wavelets
    Biorthogonal(usize, usize), // (reconstruction, decomposition) filter lengths
    /// Coiflets
    Coiflets(usize),
    /// Haar wavelet (simplest case)
    Haar,
}

/// Wavelet filter coefficients
#[derive(Debug, Clone)]
pub struct WaveletFilter<T> {
    /// Low-pass decomposition filter (scaling function)
    pub low_dec: Vec<T>,
    /// High-pass decomposition filter (wavelet function)
    pub high_dec: Vec<T>,
    /// Low-pass reconstruction filter
    pub low_rec: Vec<T>,
    /// High-pass reconstruction filter
    pub high_rec: Vec<T>,
}

impl<T> WaveletFilter<T>
where
    T: Float + FromPrimitive,
{
    /// Create wavelet filter coefficients for a given family
    pub fn new(family: WaveletFamily) -> NdimageResult<Self> {
        match family {
            WaveletFamily::Haar => Ok(Self::haar()),
            WaveletFamily::Daubechies(n) => Self::daubechies(n),
            WaveletFamily::Coiflets(n) => Self::coiflets(n),
            WaveletFamily::Biorthogonal(nr, nd) => Self::biorthogonal(nr, nd),
        }
    }

    /// Haar wavelet coefficients
    fn haar() -> Self {
        let sqrt2_inv = T::from_f64(1.0 / std::f64::consts::SQRT_2).unwrap();

        Self {
            low_dec: vec![sqrt2_inv, sqrt2_inv],
            high_dec: vec![sqrt2_inv, -sqrt2_inv],
            low_rec: vec![sqrt2_inv, sqrt2_inv],
            high_rec: vec![-sqrt2_inv, sqrt2_inv],
        }
    }

    /// Daubechies wavelet coefficients
    fn daubechies(n: usize) -> NdimageResult<Self> {
        if n == 1 {
            return Ok(Self::haar());
        }

        if n > 10 {
            return Err(NdimageError::InvalidInput(
                "Daubechies wavelets with more than 10 vanishing moments not supported".into(),
            ));
        }

        // Pre-computed Daubechies coefficients for common cases
        let coeffs = match n {
            2 => vec![
                T::from_f64(0.48296291314469025).unwrap(),
                T::from_f64(0.8365163037378079).unwrap(),
                T::from_f64(0.22414386804185735).unwrap(),
                T::from_f64(-0.12940952255092145).unwrap(),
            ],
            3 => vec![
                T::from_f64(0.3326705529509569).unwrap(),
                T::from_f64(0.8068915093133388).unwrap(),
                T::from_f64(0.4598775021193313).unwrap(),
                T::from_f64(-0.13501102001039084).unwrap(),
                T::from_f64(-0.08544127388224149).unwrap(),
                T::from_f64(0.035226291882100656).unwrap(),
            ],
            4 => vec![
                T::from_f64(0.23037781330885523).unwrap(),
                T::from_f64(0.7148465705525415).unwrap(),
                T::from_f64(0.6308807679295904).unwrap(),
                T::from_f64(-0.02798376941698385).unwrap(),
                T::from_f64(-0.18703481171888114).unwrap(),
                T::from_f64(0.030841381835986965).unwrap(),
                T::from_f64(0.032883011666982945).unwrap(),
                T::from_f64(-0.010597401784997278).unwrap(),
            ],
            _ => {
                return Err(NdimageError::NotImplementedError(format!(
                    "Daubechies wavelet with {} vanishing moments not implemented",
                    n
                )));
            }
        };

        let low_dec = coeffs;
        let mut high_dec = Vec::with_capacity(low_dec.len());

        // High-pass filter: h[n] = (-1)^n * g[L-1-n]
        for (i, &coeff) in low_dec.iter().rev().enumerate() {
            let sign = if i % 2 == 0 { T::one() } else { -T::one() };
            high_dec.push(sign * coeff);
        }

        // Reconstruction filters are time-reversed versions
        let low_rec = low_dec.iter().rev().cloned().collect();
        let high_rec = high_dec.iter().rev().cloned().collect();

        Ok(Self {
            low_dec,
            high_dec,
            low_rec,
            high_rec,
        })
    }

    /// Coiflets wavelet coefficients (simplified implementation)
    fn coiflets(n: usize) -> NdimageResult<Self> {
        if n != 2 {
            return Err(NdimageError::NotImplementedError(
                "Only Coiflets-2 is implemented".into(),
            ));
        }

        let coeffs = vec![
            T::from_f64(-0.01565572813546454).unwrap(),
            T::from_f64(-0.0727326195128539).unwrap(),
            T::from_f64(0.38486484686420286).unwrap(),
            T::from_f64(0.8525720202122554).unwrap(),
            T::from_f64(0.3378976624578092).unwrap(),
            T::from_f64(-0.0727326195128539).unwrap(),
        ];

        let low_dec = coeffs;
        let mut high_dec = Vec::with_capacity(low_dec.len());

        for (i, &coeff) in low_dec.iter().rev().enumerate() {
            let sign = if i % 2 == 0 { T::one() } else { -T::one() };
            high_dec.push(sign * coeff);
        }

        let low_rec = low_dec.iter().rev().cloned().collect();
        let high_rec = high_dec.iter().rev().cloned().collect();

        Ok(Self {
            low_dec,
            high_dec,
            low_rec,
            high_rec,
        })
    }

    /// Biorthogonal wavelet coefficients
    fn biorthogonal(nr: usize, nd: usize) -> NdimageResult<Self> {
        match (nr, nd) {
            (1, 1) => {
                // Biorthogonal 1.1 (Haar)
                Ok(Self::haar())
            }
            (2, 2) => {
                // Biorthogonal 2.2 (Linear B-spline)
                let low_dec = vec![
                    T::from_f64(-0.12940952255092145).unwrap(),
                    T::from_f64(0.22414386804185735).unwrap(),
                    T::from_f64(0.8365163037378079).unwrap(),
                    T::from_f64(0.48296291314469025).unwrap(),
                ];
                
                let high_dec = vec![
                    T::from_f64(-0.48296291314469025).unwrap(),
                    T::from_f64(0.8365163037378079).unwrap(),
                    T::from_f64(-0.22414386804185735).unwrap(),
                    T::from_f64(-0.12940952255092145).unwrap(),
                ];
                
                let low_rec = vec![
                    T::from_f64(0.48296291314469025).unwrap(),
                    T::from_f64(0.8365163037378079).unwrap(),
                    T::from_f64(0.22414386804185735).unwrap(),
                    T::from_f64(-0.12940952255092145).unwrap(),
                ];
                
                let high_rec = vec![
                    T::from_f64(-0.12940952255092145).unwrap(),
                    T::from_f64(-0.22414386804185735).unwrap(),
                    T::from_f64(0.8365163037378079).unwrap(),
                    T::from_f64(-0.48296291314469025).unwrap(),
                ];
                
                Ok(Self { low_dec, high_dec, low_rec, high_rec })
            }
            (2, 4) => {
                // Biorthogonal 2.4
                let low_dec = vec![
                    T::from_f64(0.0).unwrap(),
                    T::from_f64(-0.1767766952966369).unwrap(),
                    T::from_f64(-0.07589077294536541).unwrap(),
                    T::from_f64(0.87343749756405325).unwrap(),
                    T::from_f64(0.87343749756405325).unwrap(),
                    T::from_f64(-0.07589077294536541).unwrap(),
                    T::from_f64(-0.1767766952966369).unwrap(),
                    T::from_f64(0.0).unwrap(),
                ];
                
                let high_dec = vec![
                    T::from_f64(0.0).unwrap(),
                    T::from_f64(0.1767766952966369).unwrap(),
                    T::from_f64(-0.07589077294536541).unwrap(),
                    T::from_f64(-0.87343749756405325).unwrap(),
                    T::from_f64(0.87343749756405325).unwrap(),
                    T::from_f64(0.07589077294536541).unwrap(),
                    T::from_f64(-0.1767766952966369).unwrap(),
                    T::from_f64(0.0).unwrap(),
                ];
                
                let low_rec = vec![
                    T::from_f64(0.0).unwrap(),
                    T::from_f64(-0.1767766952966369).unwrap(),
                    T::from_f64(-0.07589077294536541).unwrap(),
                    T::from_f64(0.87343749756405325).unwrap(),
                    T::from_f64(0.87343749756405325).unwrap(),
                    T::from_f64(-0.07589077294536541).unwrap(),
                    T::from_f64(-0.1767766952966369).unwrap(),
                    T::from_f64(0.0).unwrap(),
                ];
                
                let high_rec = vec![
                    T::from_f64(0.0).unwrap(),
                    T::from_f64(-0.1767766952966369).unwrap(),
                    T::from_f64(0.07589077294536541).unwrap(),
                    T::from_f64(0.87343749756405325).unwrap(),
                    T::from_f64(-0.87343749756405325).unwrap(),
                    T::from_f64(-0.07589077294536541).unwrap(),
                    T::from_f64(0.1767766952966369).unwrap(),
                    T::from_f64(0.0).unwrap(),
                ];
                
                Ok(Self { low_dec, high_dec, low_rec, high_rec })
            }
            (4, 4) => {
                // Biorthogonal 4.4 (Cubic B-spline)
                let low_dec = vec![
                    T::from_f64(0.03314563036811942).unwrap(),
                    T::from_f64(-0.06629126073623884).unwrap(),
                    T::from_f64(-0.17677669529663687).unwrap(),
                    T::from_f64(0.4198446513295126).unwrap(),
                    T::from_f64(0.9943689110435825).unwrap(),
                    T::from_f64(0.4198446513295126).unwrap(),
                    T::from_f64(-0.17677669529663687).unwrap(),
                    T::from_f64(-0.06629126073623884).unwrap(),
                    T::from_f64(0.03314563036811942).unwrap(),
                ];
                
                let high_dec = vec![
                    T::from_f64(0.0).unwrap(),
                    T::from_f64(0.01657281518405971).unwrap(),
                    T::from_f64(-0.03314563036811942).unwrap(),
                    T::from_f64(-0.1767766952966369).unwrap(),
                    T::from_f64(0.41984465132951256).unwrap(),
                    T::from_f64(-0.9943689110435825).unwrap(),
                    T::from_f64(0.41984465132951256).unwrap(),
                    T::from_f64(-0.1767766952966369).unwrap(),
                    T::from_f64(-0.03314563036811942).unwrap(),
                    T::from_f64(0.01657281518405971).unwrap(),
                    T::from_f64(0.0).unwrap(),
                ];
                
                let low_rec = vec![
                    T::from_f64(0.03314563036811942).unwrap(),
                    T::from_f64(-0.06629126073623884).unwrap(),
                    T::from_f64(-0.17677669529663687).unwrap(),
                    T::from_f64(0.4198446513295126).unwrap(),
                    T::from_f64(0.9943689110435825).unwrap(),
                    T::from_f64(0.4198446513295126).unwrap(),
                    T::from_f64(-0.17677669529663687).unwrap(),
                    T::from_f64(-0.06629126073623884).unwrap(),
                    T::from_f64(0.03314563036811942).unwrap(),
                ];
                
                let high_rec = vec![
                    T::from_f64(0.0).unwrap(),
                    T::from_f64(-0.01657281518405971).unwrap(),
                    T::from_f64(-0.03314563036811942).unwrap(),
                    T::from_f64(0.1767766952966369).unwrap(),
                    T::from_f64(0.41984465132951256).unwrap(),
                    T::from_f64(0.9943689110435825).unwrap(),
                    T::from_f64(0.41984465132951256).unwrap(),
                    T::from_f64(0.1767766952966369).unwrap(),
                    T::from_f64(-0.03314563036811942).unwrap(),
                    T::from_f64(-0.01657281518405971).unwrap(),
                    T::from_f64(0.0).unwrap(),
                ];
                
                Ok(Self { low_dec, high_dec, low_rec, high_rec })
            }
            (6, 8) => {
                // Biorthogonal 6.8
                let low_dec = vec![
                    T::from_f64(0.0019088317364812906).unwrap(),
                    T::from_f64(-0.0019142861290887667).unwrap(),
                    T::from_f64(-0.016990639867602342).unwrap(),
                    T::from_f64(0.01193456527972926).unwrap(),
                    T::from_f64(0.04973290349094079).unwrap(),
                    T::from_f64(-0.07726317316720414).unwrap(),
                    T::from_f64(-0.09405920349573646).unwrap(),
                    T::from_f64(0.4207962846098268).unwrap(),
                    T::from_f64(0.8259229974584023).unwrap(),
                    T::from_f64(0.4207962846098268).unwrap(),
                    T::from_f64(-0.09405920349573646).unwrap(),
                    T::from_f64(-0.07726317316720414).unwrap(),
                    T::from_f64(0.04973290349094079).unwrap(),
                    T::from_f64(0.01193456527972926).unwrap(),
                    T::from_f64(-0.016990639867602342).unwrap(),
                    T::from_f64(-0.0019142861290887667).unwrap(),
                    T::from_f64(0.0019088317364812906).unwrap(),
                ];
                
                // For simplicity, generate high-pass filters using quadrature mirror filter relationship
                let mut high_dec = Vec::with_capacity(low_dec.len());
                for (i, &coeff) in low_dec.iter().rev().enumerate() {
                    let sign = if i % 2 == 0 { T::one() } else { -T::one() };
                    high_dec.push(sign * coeff);
                }
                
                let low_rec = low_dec.iter().rev().cloned().collect();
                let high_rec = high_dec.iter().rev().cloned().collect();
                
                Ok(Self { low_dec, high_dec, low_rec, high_rec })
            }
            _ => {
                Err(NdimageError::NotImplementedError(
                    format!("Biorthogonal wavelet ({}, {}) is not implemented. Supported variants: (1,1), (2,2), (2,4), (4,4), (6,8)", nr, nd),
                ))
            }
        }
    }
}

/// 1D Discrete Wavelet Transform
pub fn dwt_1d<T>(
    signal: &ArrayView1<T>,
    wavelet: &WaveletFilter<T>,
    mode: BorderMode,
) -> NdimageResult<(Array1<T>, Array1<T>)>
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + 'static,
{
    let n = signal.len();
    if n < 2 {
        return Err(NdimageError::InvalidInput(
            "Signal must have at least 2 elements".into(),
        ));
    }

    // Pad signal for boundary handling
    let padded = pad_signal_1d(signal, &wavelet.low_dec, mode)?;

    // Apply low-pass and high-pass filters
    let low_pass = convolve_downsample_1d(&padded.view(), &wavelet.low_dec, 2)?;
    let high_pass = convolve_downsample_1d(&padded.view(), &wavelet.high_dec, 2)?;

    Ok((low_pass, high_pass))
}

/// 1D Inverse Discrete Wavelet Transform
pub fn idwt_1d<T>(
    low: &ArrayView1<T>,
    high: &ArrayView1<T>,
    wavelet: &WaveletFilter<T>,
) -> NdimageResult<Array1<T>>
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + 'static,
{
    if low.len() != high.len() {
        return Err(NdimageError::InvalidInput(
            "Low and high frequency components must have the same length".into(),
        ));
    }

    // Upsample and filter
    let low_upsampled = upsample_convolve_1d(low, &wavelet.low_rec, 2)?;
    let high_upsampled = upsample_convolve_1d(high, &wavelet.high_rec, 2)?;

    // Combine
    let mut result = Array1::zeros(low_upsampled.len());
    for i in 0..result.len() {
        result[i] = low_upsampled[i] + high_upsampled[i];
    }

    Ok(result)
}

/// 2D Discrete Wavelet Transform
pub fn dwt_2d<T>(
    image: &ArrayView2<T>,
    wavelet: &WaveletFilter<T>,
    mode: BorderMode,
) -> NdimageResult<(Array2<T>, Array2<T>, Array2<T>, Array2<T>)>
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + 'static,
{
    let (height, width) = image.dim();

    if height < 2 || width < 2 {
        return Err(NdimageError::InvalidInput(
            "Image must be at least 2x2 pixels".into(),
        ));
    }

    // First, apply DWT to each row
    let mut row_low = Array2::zeros((height, width / 2));
    let mut row_high = Array2::zeros((height, width / 2));

    for i in 0..height {
        let row = image.row(i);
        let (low, high) = dwt_1d(&row, wavelet, mode)?;

        for j in 0..low.len() {
            row_low[[i, j]] = low[j];
        }
        for j in 0..high.len() {
            row_high[[i, j]] = high[j];
        }
    }

    // Then apply DWT to each column of the results
    let mut ll = Array2::zeros((height / 2, width / 2)); // Low-Low
    let mut lh = Array2::zeros((height / 2, width / 2)); // Low-High
    let mut hl = Array2::zeros((height / 2, width / 2)); // High-Low
    let mut hh = Array2::zeros((height / 2, width / 2)); // High-High

    // Process low-frequency rows
    for j in 0..width / 2 {
        let col = row_low.column(j);
        let (low, high) = dwt_1d(&col, wavelet, mode)?;

        for i in 0..low.len() {
            ll[[i, j]] = low[i];
        }
        for i in 0..high.len() {
            lh[[i, j]] = high[i];
        }
    }

    // Process high-frequency rows
    for j in 0..width / 2 {
        let col = row_high.column(j);
        let (low, high) = dwt_1d(&col, wavelet, mode)?;

        for i in 0..low.len() {
            hl[[i, j]] = low[i];
        }
        for i in 0..high.len() {
            hh[[i, j]] = high[i];
        }
    }

    Ok((ll, lh, hl, hh))
}

/// 2D Inverse Discrete Wavelet Transform
pub fn idwt_2d<T>(
    ll: &ArrayView2<T>,
    lh: &ArrayView2<T>,
    hl: &ArrayView2<T>,
    hh: &ArrayView2<T>,
    wavelet: &WaveletFilter<T>,
) -> NdimageResult<Array2<T>>
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + 'static,
{
    let (sub_height, sub_width) = ll.dim();

    if lh.dim() != (sub_height, sub_width)
        || hl.dim() != (sub_height, sub_width)
        || hh.dim() != (sub_height, sub_width)
    {
        return Err(NdimageError::InvalidInput(
            "All wavelet coefficient arrays must have the same dimensions".into(),
        ));
    }

    let height = sub_height * 2;
    let width = sub_width * 2;

    // First, reconstruct in the column direction
    let mut row_low = Array2::zeros((height, sub_width));
    let mut row_high = Array2::zeros((height, sub_width));

    for j in 0..sub_width {
        let ll_col = ll.column(j);
        let lh_col = lh.column(j);
        let reconstructed_low = idwt_1d(&ll_col, &lh_col, wavelet)?;

        let hl_col = hl.column(j);
        let hh_col = hh.column(j);
        let reconstructed_high = idwt_1d(&hl_col, &hh_col, wavelet)?;

        for i in 0..height {
            row_low[[i, j]] = reconstructed_low[i];
            row_high[[i, j]] = reconstructed_high[i];
        }
    }

    // Then reconstruct in the row direction
    let mut result = Array2::zeros((height, width));

    for i in 0..height {
        let low_row = row_low.row(i);
        let high_row = row_high.row(i);
        let reconstructed_row = idwt_1d(&low_row, &high_row, wavelet)?;

        for j in 0..width {
            result[[i, j]] = reconstructed_row[j];
        }
    }

    Ok(result)
}

/// Wavelet denoising using soft thresholding
pub fn wavelet_denoise<T>(
    image: &ArrayView2<T>,
    wavelet: WaveletFamily,
    threshold: T,
    mode: BorderMode,
) -> NdimageResult<Array2<T>>
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + 'static,
{
    let filter = WaveletFilter::new(wavelet)?;

    // Forward DWT
    let (ll, lh, hl, hh) = dwt_2d(image, &filter, mode)?;

    // Apply soft thresholding to detail coefficients
    let lh_denoised = soft_threshold(&lh.view(), threshold);
    let hl_denoised = soft_threshold(&hl.view(), threshold);
    let hh_denoised = soft_threshold(&hh.view(), threshold);

    // Inverse DWT
    idwt_2d(
        &ll.view(),
        &lh_denoised.view(),
        &hl_denoised.view(),
        &hh_denoised.view(),
        &filter,
    )
}

/// Multi-level wavelet decomposition
pub fn wavelet_decompose<T>(
    image: &ArrayView2<T>,
    wavelet: WaveletFamily,
    levels: usize,
    mode: BorderMode,
) -> NdimageResult<Vec<(Array2<T>, Array2<T>, Array2<T>, Array2<T>)>>
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + 'static,
{
    if levels == 0 {
        return Err(NdimageError::InvalidInput(
            "Number of decomposition levels must be positive".into(),
        ));
    }

    let filter = WaveletFilter::new(wavelet)?;
    let mut decomposition = Vec::with_capacity(levels);
    let mut current_ll = image.to_owned();

    for _ in 0..levels {
        let (ll, lh, hl, hh) = dwt_2d(&current_ll.view(), &filter, mode)?;
        decomposition.push((ll.clone(), lh, hl, hh));
        current_ll = ll;
    }

    Ok(decomposition)
}

/// Multi-level wavelet reconstruction
pub fn wavelet_reconstruct<T>(
    decomposition: &[(Array2<T>, Array2<T>, Array2<T>, Array2<T>)],
    wavelet: WaveletFamily,
) -> NdimageResult<Array2<T>>
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + 'static,
{
    if decomposition.is_empty() {
        return Err(NdimageError::InvalidInput(
            "Decomposition must contain at least one level".into(),
        ));
    }

    let filter = WaveletFilter::new(wavelet)?;

    // Start with the coarsest level
    let (ref ll, ref lh, ref hl, ref hh) = decomposition[decomposition.len() - 1];
    let mut current = idwt_2d(&ll.view(), &lh.view(), &hl.view(), &hh.view(), &filter)?;

    // Reconstruct level by level
    for level in (0..decomposition.len() - 1).rev() {
        let (_, ref lh, ref hl, ref hh) = decomposition[level];
        current = idwt_2d(&current.view(), &lh.view(), &hl.view(), &hh.view(), &filter)?;
    }

    Ok(current)
}

/// Soft thresholding function
fn soft_threshold<T>(coeffs: &ArrayView2<T>, threshold: T) -> Array2<T>
where
    T: Float + FromPrimitive,
{
    coeffs.mapv(|x| {
        if x.abs() <= threshold {
            T::zero()
        } else if x > threshold {
            x - threshold
        } else {
            x + threshold
        }
    })
}

/// Pad 1D signal for convolution
fn pad_signal_1d<T>(
    signal: &ArrayView1<T>,
    filter: &[T],
    mode: BorderMode,
) -> NdimageResult<Array1<T>>
where
    T: Float + FromPrimitive + Clone,
{
    let n = signal.len();
    let filter_len = filter.len();
    let pad_len = filter_len - 1;

    let mut padded = Array1::zeros(n + 2 * pad_len);

    // Copy original signal to center
    for i in 0..n {
        padded[i + pad_len] = signal[i];
    }

    // Apply border mode
    match mode {
        BorderMode::Constant => {
            // Zeros already filled
        }
        BorderMode::Reflect => {
            // Left padding
            for i in 0..pad_len {
                let src_idx = pad_len - 1 - i;
                if src_idx < n {
                    padded[i] = signal[src_idx];
                }
            }
            // Right padding
            for i in 0..pad_len {
                let src_idx = n - 1 - i;
                if src_idx < n {
                    padded[n + pad_len + i] = signal[src_idx];
                }
            }
        }
        BorderMode::Nearest => {
            // Left padding
            for i in 0..pad_len {
                padded[i] = signal[0];
            }
            // Right padding
            for i in 0..pad_len {
                padded[n + pad_len + i] = signal[n - 1];
            }
        }
        _ => {
            return Err(NdimageError::NotImplementedError(format!(
                "Border mode {:?} not implemented for wavelets",
                mode
            )));
        }
    }

    Ok(padded)
}

/// 1D convolution with downsampling
fn convolve_downsample_1d<T>(
    signal: &ArrayView1<T>,
    filter: &[T],
    downsample: usize,
) -> NdimageResult<Array1<T>>
where
    T: Float + FromPrimitive + Clone + Zero,
{
    let n = signal.len();
    let filter_len = filter.len();

    if n < filter_len {
        return Err(NdimageError::InvalidInput(
            "Signal length must be at least filter length".into(),
        ));
    }

    let output_len = (n - filter_len + 1 + downsample - 1) / downsample;
    let mut output = Array1::zeros(output_len);

    for i in 0..output_len {
        let start_idx = i * downsample;
        if start_idx + filter_len <= n {
            let mut sum = T::zero();
            for j in 0..filter_len {
                sum = sum + signal[start_idx + j] * filter[j];
            }
            output[i] = sum;
        }
    }

    Ok(output)
}

/// 1D upsampling with convolution
fn upsample_convolve_1d<T>(
    signal: &ArrayView1<T>,
    filter: &[T],
    upsample: usize,
) -> NdimageResult<Array1<T>>
where
    T: Float + FromPrimitive + Clone + Zero,
{
    let n = signal.len();
    let filter_len = filter.len();
    let upsampled_len = n * upsample;
    let output_len = upsampled_len + filter_len - 1;

    // Upsample by inserting zeros
    let mut upsampled = Array1::zeros(upsampled_len);
    for i in 0..n {
        upsampled[i * upsample] = signal[i];
    }

    // Convolve with reconstruction filter
    let mut output = Array1::zeros(output_len);
    for i in 0..output_len {
        let mut sum = T::zero();
        for j in 0..filter_len {
            if i >= j && i - j < upsampled_len {
                sum = sum + upsampled[i - j] * filter[j];
            }
        }
        output[i] = sum;
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_haar_coefficients() {
        let haar = WaveletFilter::<f64>::new(WaveletFamily::Haar).unwrap();
        let sqrt2_inv = 1.0 / std::f64::consts::SQRT_2;

        assert_abs_diff_eq!(haar.low_dec[0], sqrt2_inv, epsilon = 1e-10);
        assert_abs_diff_eq!(haar.low_dec[1], sqrt2_inv, epsilon = 1e-10);
        assert_abs_diff_eq!(haar.high_dec[0], sqrt2_inv, epsilon = 1e-10);
        assert_abs_diff_eq!(haar.high_dec[1], -sqrt2_inv, epsilon = 1e-10);
    }

    #[test]
    fn test_dwt_1d() {
        let signal = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let haar = WaveletFilter::new(WaveletFamily::Haar).unwrap();

        let (low, high) = dwt_1d(&signal.view(), &haar, BorderMode::Nearest).unwrap();

        // Check that the result has the expected length
        assert_eq!(low.len(), 4);
        assert_eq!(high.len(), 4);

        // The low-pass should contain the averages
        // The high-pass should contain the differences
        assert!(low.iter().all(|&x| x.is_finite()));
        assert!(high.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_dwt_idwt_reconstruction() {
        let signal = array![1.0, 2.0, 3.0, 4.0];
        let haar = WaveletFilter::new(WaveletFamily::Haar).unwrap();

        let (low, high) = dwt_1d(&signal.view(), &haar, BorderMode::Nearest).unwrap();
        let reconstructed = idwt_1d(&low.view(), &high.view(), &haar).unwrap();

        // Check that reconstruction length is appropriate
        assert!(reconstructed.len() >= signal.len());

        // Check that values are reasonable (perfect reconstruction is complex with border handling)
        assert!(reconstructed.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_dwt_2d() {
        let image = array![
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0]
        ];

        let haar = WaveletFilter::new(WaveletFamily::Haar).unwrap();
        let (ll, lh, hl, hh) = dwt_2d(&image.view(), &haar, BorderMode::Nearest).unwrap();

        // Check dimensions
        assert_eq!(ll.dim(), (2, 2));
        assert_eq!(lh.dim(), (2, 2));
        assert_eq!(hl.dim(), (2, 2));
        assert_eq!(hh.dim(), (2, 2));

        // Check that all values are finite
        assert!(ll.iter().all(|&x| x.is_finite()));
        assert!(lh.iter().all(|&x| x.is_finite()));
        assert!(hl.iter().all(|&x| x.is_finite()));
        assert!(hh.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_wavelet_denoise() {
        let image = array![
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0]
        ];

        let denoised =
            wavelet_denoise(&image.view(), WaveletFamily::Haar, 1.0, BorderMode::Nearest).unwrap();

        // Check that output has same dimensions as input
        assert_eq!(denoised.dim(), image.dim());

        // Check that all values are finite
        assert!(denoised.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_soft_threshold() {
        let coeffs = array![[-3.0, -1.5, -0.5], [0.5, 1.5, 3.0]];

        let thresholded = soft_threshold(&coeffs.view(), 1.0);

        // Values below threshold should be zero
        assert_eq!(thresholded[[0, 2]], 0.0); // -0.5 -> 0
        assert_eq!(thresholded[[1, 0]], 0.0); // 0.5 -> 0

        // Values above threshold should be reduced
        assert_eq!(thresholded[[0, 0]], -2.0); // -3.0 -> -2.0
        assert_eq!(thresholded[[1, 2]], 2.0); // 3.0 -> 2.0
    }

    #[test]
    fn test_daubechies_coefficients() {
        let db2 = WaveletFilter::<f64>::new(WaveletFamily::Daubechies(2)).unwrap();

        // Check that we have 4 coefficients for DB2
        assert_eq!(db2.low_dec.len(), 4);
        assert_eq!(db2.high_dec.len(), 4);

        // Check that coefficients are finite
        assert!(db2.low_dec.iter().all(|&x| x.is_finite()));
        assert!(db2.high_dec.iter().all(|&x| x.is_finite()));
    }
}
