//! Time series feature extraction
//!
//! This module provides utilities for extracting features from time series data,
//! including Fourier features, wavelet features, and lag features.

use ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix1, Ix2};
use num_traits::{Float, NumCast};
use scirs2_fft::{fft, Complex};
use std::f64::consts::PI;

use crate::error::{Result, TransformError};

/// Fourier feature extractor for time series
///
/// Extracts frequency domain features using Fast Fourier Transform (FFT).
/// Useful for capturing periodic patterns in time series data.
#[derive(Debug, Clone)]
pub struct FourierFeatures {
    /// Number of Fourier components to extract
    n_components: usize,
    /// Whether to include phase information
    include_phase: bool,
    /// Whether to normalize by series length
    normalize: bool,
    /// Sampling frequency (if known)
    sampling_freq: Option<f64>,
}

impl FourierFeatures {
    /// Create a new FourierFeatures extractor
    ///
    /// # Arguments
    /// * `n_components` - Number of frequency components to extract
    pub fn new(n_components: usize) -> Self {
        FourierFeatures {
            n_components,
            include_phase: false,
            normalize: true,
            sampling_freq: None,
        }
    }

    /// Include phase information in features
    pub fn with_phase(mut self) -> Self {
        self.include_phase = true;
        self
    }

    /// Set sampling frequency
    pub fn with_sampling_freq(mut self, freq: f64) -> Self {
        self.sampling_freq = Some(freq);
        self
    }

    /// Extract Fourier features from a single time series
    fn extract_features_1d<S>(&self, x: &ArrayBase<S, Ix1>) -> Result<Array1<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let n = x.len();
        if n == 0 {
            return Err(TransformError::InvalidInput("Empty time series".to_string()));
        }

        // Convert to complex for FFT
        let mut complex_data = vec![Complex::new(0.0, 0.0); n];
        for (i, &val) in x.iter().enumerate() {
            complex_data[i] = Complex::new(
                num_traits::cast::<S::Elem, f64>(val).unwrap_or(0.0),
                0.0,
            );
        }

        // Compute FFT
        fft(&mut complex_data);

        // Extract features (only positive frequencies due to symmetry)
        let n_freq = (n / 2).min(self.n_components);
        let mut features = if self.include_phase {
            Array1::zeros(n_freq * 2)
        } else {
            Array1::zeros(n_freq)
        };

        let norm_factor = if self.normalize {
            1.0 / n as f64
        } else {
            1.0
        };

        for i in 0..n_freq {
            let magnitude = (complex_data[i].re * complex_data[i].re
                + complex_data[i].im * complex_data[i].im)
                .sqrt()
                * norm_factor;

            features[i] = magnitude;

            if self.include_phase && magnitude > 1e-10 {
                let phase = complex_data[i].im.atan2(complex_data[i].re);
                features[n_freq + i] = phase;
            }
        }

        Ok(features)
    }

    /// Transform time series data to Fourier features
    ///
    /// # Arguments
    /// * `x` - Time series data, shape (n_samples, n_timesteps)
    ///
    /// # Returns
    /// * Fourier features, shape (n_samples, n_features)
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let n_samples = x.shape()[0];
        let n_features = if self.include_phase {
            self.n_components * 2
        } else {
            self.n_components
        };

        let mut result = Array2::zeros((n_samples, n_features));

        for i in 0..n_samples {
            let features = self.extract_features_1d(&x.row(i))?;
            let feat_len = features.len().min(n_features);
            result.slice_mut(ndarray::s![i, ..feat_len]).assign(&features.slice(ndarray::s![..feat_len]));
        }

        Ok(result)
    }
}

/// Lag feature extractor for time series
///
/// Creates lagged versions of time series as features. Useful for
/// autoregressive modeling and capturing temporal dependencies.
#[derive(Debug, Clone)]
pub struct LagFeatures {
    /// List of lags to include
    lags: Vec<usize>,
    /// Whether to drop NaN values resulting from lagging
    drop_na: bool,
}

impl LagFeatures {
    /// Create a new LagFeatures extractor
    ///
    /// # Arguments
    /// * `lags` - List of lag values (e.g., vec![1, 2, 3] for lags 1, 2, and 3)
    pub fn new(lags: Vec<usize>) -> Self {
        LagFeatures {
            lags,
            drop_na: true,
        }
    }

    /// Create with a range of lags
    pub fn with_range(start: usize, end: usize) -> Self {
        let lags = (start..=end).collect();
        LagFeatures {
            lags,
            drop_na: true,
        }
    }

    /// Set whether to drop NaN values
    pub fn with_drop_na(mut self, drop_na: bool) -> Self {
        self.drop_na = drop_na;
        self
    }

    /// Transform time series data to lag features
    ///
    /// # Arguments
    /// * `x` - Time series data, shape (n_timesteps,) or (n_samples, n_timesteps)
    ///
    /// # Returns
    /// * Lag features
    pub fn transform_1d<S>(&self, x: &ArrayBase<S, Ix1>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let n = x.len();
        let max_lag = *self.lags.iter().max().unwrap_or(&0);

        if max_lag >= n {
            return Err(TransformError::InvalidInput(format!(
                "Maximum lag {} must be less than series length {}",
                max_lag, n
            )));
        }

        let start_idx = if self.drop_na { max_lag } else { 0 };
        let n_samples = n - start_idx;
        let n_features = self.lags.len() + 1; // Original + lags

        let mut result = Array2::zeros((n_samples, n_features));

        // Original series
        for i in 0..n_samples {
            result[[i, 0]] = num_traits::cast::<S::Elem, f64>(x[start_idx + i]).unwrap_or(0.0);
        }

        // Lagged features
        for (lag_idx, &lag) in self.lags.iter().enumerate() {
            for i in 0..n_samples {
                let idx = start_idx + i;
                if idx >= lag {
                    result[[i, lag_idx + 1]] =
                        num_traits::cast::<S::Elem, f64>(x[idx - lag]).unwrap_or(0.0);
                } else if !self.drop_na {
                    result[[i, lag_idx + 1]] = f64::NAN;
                }
            }
        }

        Ok(result)
    }

    /// Transform multiple time series
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Vec<Array2<f64>>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let n_series = x.shape()[0];
        let mut results = Vec::new();

        for i in 0..n_series {
            let series = x.row(i);
            let lag_features = self.transform_1d(&series)?;
            results.push(lag_features);
        }

        Ok(results)
    }
}

/// Wavelet feature extractor for time series
///
/// Extracts features using wavelet decomposition. Useful for
/// multi-resolution analysis of time series.
#[derive(Debug, Clone)]
pub struct WaveletFeatures {
    /// Wavelet type: 'db1' (Haar), 'db2', 'db3', 'db4'
    wavelet: String,
    /// Decomposition level
    level: usize,
    /// Whether to include approximation coefficients
    include_approx: bool,
}

impl WaveletFeatures {
    /// Create a new WaveletFeatures extractor
    ///
    /// # Arguments
    /// * `wavelet` - Wavelet type (e.g., "db1" for Haar wavelet)
    /// * `level` - Decomposition level
    pub fn new(wavelet: &str, level: usize) -> Self {
        WaveletFeatures {
            wavelet: wavelet.to_string(),
            level,
            include_approx: true,
        }
    }

    /// Set whether to include approximation coefficients
    pub fn with_include_approx(mut self, include: bool) -> Self {
        self.include_approx = include;
        self
    }

    /// Haar wavelet transform (simplified)
    fn haar_transform(&self, x: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = x.len();
        let mut approx = Vec::with_capacity(n / 2);
        let mut detail = Vec::with_capacity(n / 2);

        for i in (0..n).step_by(2) {
            if i + 1 < n {
                approx.push((x[i] + x[i + 1]) / 2.0_f64.sqrt());
                detail.push((x[i] - x[i + 1]) / 2.0_f64.sqrt());
            } else {
                // Handle odd length
                approx.push(x[i]);
            }
        }

        (approx, detail)
    }

    /// Multi-level wavelet decomposition
    fn wavelet_decompose(&self, x: &[f64]) -> Result<Vec<Vec<f64>>> {
        if self.wavelet != "db1" && self.wavelet != "haar" {
            return Err(TransformError::InvalidInput(
                "Only Haar wavelet (db1) is currently implemented".to_string(),
            ));
        }

        let mut coefficients = Vec::new();
        let mut current = x.to_vec();

        for _ in 0..self.level {
            let (approx, detail) = self.haar_transform(&current);
            coefficients.push(detail);
            current = approx;

            if current.len() < 2 {
                break;
            }
        }

        if self.include_approx {
            coefficients.push(current);
        }

        Ok(coefficients)
    }

    /// Extract wavelet features from a single time series
    fn extract_features_1d<S>(&self, x: &ArrayBase<S, Ix1>) -> Result<Array1<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x_vec: Vec<f64> = x
            .iter()
            .map(|&v| num_traits::cast::<S::Elem, f64>(v).unwrap_or(0.0))
            .collect();

        let coefficients = self.wavelet_decompose(&x_vec)?;

        // Calculate total number of features
        let n_features: usize = coefficients.iter().map(|c| c.len()).sum();
        let mut features = Array1::zeros(n_features);

        let mut idx = 0;
        for coeff_level in coefficients {
            for &coeff in &coeff_level {
                features[idx] = coeff;
                idx += 1;
            }
        }

        Ok(features)
    }

    /// Transform time series data to wavelet features
    ///
    /// # Arguments
    /// * `x` - Time series data, shape (n_samples, n_timesteps)
    ///
    /// # Returns
    /// * Wavelet features
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Vec<Array1<f64>>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let n_samples = x.shape()[0];
        let mut results = Vec::new();

        for i in 0..n_samples {
            let features = self.extract_features_1d(&x.row(i))?;
            results.push(features);
        }

        Ok(results)
    }
}

/// Combined time series feature extractor
///
/// Combines multiple feature extraction methods for comprehensive
/// time series feature engineering.
#[derive(Debug, Clone)]
pub struct TimeSeriesFeatures {
    /// Whether to include Fourier features
    use_fourier: bool,
    /// Whether to include lag features
    use_lags: bool,
    /// Whether to include wavelet features
    use_wavelets: bool,
    /// Fourier feature configuration
    fourier_config: Option<FourierFeatures>,
    /// Lag feature configuration
    lag_config: Option<LagFeatures>,
    /// Wavelet feature configuration
    wavelet_config: Option<WaveletFeatures>,
}

impl TimeSeriesFeatures {
    /// Create a new combined feature extractor
    pub fn new() -> Self {
        TimeSeriesFeatures {
            use_fourier: true,
            use_lags: true,
            use_wavelets: false,
            fourier_config: Some(FourierFeatures::new(10)),
            lag_config: Some(LagFeatures::with_range(1, 5)),
            wavelet_config: None,
        }
    }

    /// Configure Fourier features
    pub fn with_fourier(mut self, n_components: usize, include_phase: bool) -> Self {
        self.use_fourier = true;
        let mut fourier = FourierFeatures::new(n_components);
        if include_phase {
            fourier = fourier.with_phase();
        }
        self.fourier_config = Some(fourier);
        self
    }

    /// Configure lag features
    pub fn with_lags(mut self, lags: Vec<usize>) -> Self {
        self.use_lags = true;
        self.lag_config = Some(LagFeatures::new(lags));
        self
    }

    /// Configure wavelet features
    pub fn with_wavelets(mut self, wavelet: &str, level: usize) -> Self {
        self.use_wavelets = true;
        self.wavelet_config = Some(WaveletFeatures::new(wavelet, level));
        self
    }
}

impl Default for TimeSeriesFeatures {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn test_fourier_features() {
        // Create a simple sinusoidal signal
        let n = 100;
        let mut signal = Vec::new();
        for i in 0..n {
            let t = i as f64 / n as f64 * 4.0 * PI;
            signal.push((t).sin() + 0.5 * (2.0 * t).sin());
        }
        let x = Array::from_shape_vec((1, n), signal).unwrap();

        let fourier = FourierFeatures::new(10);
        let features = fourier.transform(&x).unwrap();

        assert_eq!(features.shape(), &[1, 10]);

        // First few components should have high magnitude
        assert!(features[[0, 0]] > 0.1);
    }

    #[test]
    fn test_lag_features() {
        let x = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let lag_extractor = LagFeatures::new(vec![1, 2]);
        let features = lag_extractor.transform_1d(&x.view()).unwrap();

        // Should have 4 samples (6 - max_lag(2)) and 3 features (original + 2 lags)
        assert_eq!(features.shape(), &[4, 3]);

        // Check first row: x[2]=3, x[1]=2, x[0]=1
        assert_eq!(features[[0, 0]], 3.0);
        assert_eq!(features[[0, 1]], 2.0);
        assert_eq!(features[[0, 2]], 1.0);
    }

    #[test]
    fn test_wavelet_features() {
        let x = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let x_2d = x.clone().into_shape((1, 8)).unwrap();

        let wavelet = WaveletFeatures::new("db1", 2);
        let features = wavelet.transform(&x_2d).unwrap();

        assert!(!features.is_empty());
        assert!(features[0].len() > 0);
    }
}