//! Unified denoising API
//!
//! This module provides a consistent, high-level interface to all denoising
//! methods available in the scirs2-signal library. It allows users to easily
//! switch between different denoising algorithms and compare their performance.

use crate::denoise::{denoise_wavelet, ThresholdMethod, ThresholdSelect};
use crate::denoise_advanced::{advanced_denoise, AdvancedDenoiseConfig, NoiseEstimation};
use crate::denoise_cutting_edge::{denoise_dictionary_learning, DictionaryDenoiseConfig};
use crate::denoise_enhanced::{
    denoise_median_1d, denoise_total_variation_1d, denoise_wiener_1d, BilateralConfig, WienerConfig,
};
use crate::denoise_ultra_advanced::{ultra_advanced_denoise, UltraAdvancedDenoisingConfig};
use crate::dwt::Wavelet;
use crate::error::{SignalError, SignalResult};
use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Unified denoising method selector
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DenoisingMethod {
    /// Basic wavelet denoising
    WaveletBasic {
        wavelet: Wavelet,
        levels: Option<usize>,
        threshold_method: ThresholdMethod,
        threshold_select: ThresholdSelect,
    },
    /// Advanced wavelet denoising with multiple techniques
    WaveletAdvanced { config: AdvancedDenoiseConfig },
    /// Dictionary learning based denoising
    DictionaryLearning { config: DictionaryDenoiseConfig },
    /// Wiener filtering
    Wiener { config: WienerConfig },
    /// Median filtering
    Median { window_size: usize },
    /// Total variation denoising
    TotalVariation { lambda: f64, iterations: usize },
    /// Ultra-advanced denoising with modern techniques
    UltraAdvanced {
        config: UltraAdvancedDenoisingConfig,
    },
}

impl Default for DenoisingMethod {
    fn default() -> Self {
        Self::WaveletBasic {
            wavelet: Wavelet::DB(4),
            levels: Some(4),
            threshold_method: ThresholdMethod::Soft,
            threshold_select: ThresholdSelect::Universal,
        }
    }
}

/// Unified denoising configuration
#[derive(Debug, Clone)]
pub struct UnifiedDenoisingConfig {
    /// Primary denoising method
    pub method: DenoisingMethod,
    /// Noise level estimate (if known)
    pub noise_level: Option<f64>,
    /// Enable preprocessing
    pub enable_preprocessing: bool,
    /// Enable postprocessing
    pub enable_postprocessing: bool,
    /// Enable performance benchmarking
    pub benchmark: bool,
}

impl Default for UnifiedDenoisingConfig {
    fn default() -> Self {
        Self {
            method: DenoisingMethod::default(),
            noise_level: None,
            enable_preprocessing: true,
            enable_postprocessing: true,
            benchmark: false,
        }
    }
}

/// Unified denoising result with comprehensive information
#[derive(Debug, Clone)]
pub struct UnifiedDenoisingResult {
    /// Denoised signal
    pub denoised: Array1<f64>,
    /// Estimated noise level
    pub estimated_noise_level: f64,
    /// Signal-to-noise ratio improvement (in dB)
    pub snr_improvement: Option<f64>,
    /// Method used for denoising
    pub method_used: DenoisingMethod,
    /// Processing time (if benchmarking enabled)
    pub processing_time_ms: Option<f64>,
    /// Quality metrics
    pub quality_metrics: QualityMetrics,
}

/// Quality metrics for denoising assessment
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    /// Mean squared error (if reference signal available)
    pub mse: Option<f64>,
    /// Peak signal-to-noise ratio (if reference signal available)
    pub psnr: Option<f64>,
    /// Structural similarity index (if reference signal available)
    pub ssim: Option<f64>,
    /// Signal preservation metric (0-1, higher is better)
    pub signal_preservation: f64,
    /// Noise reduction metric (0-1, higher is better)
    pub noise_reduction: f64,
}

/// Unified denoising function
///
/// This function provides a single entry point for all denoising methods
/// available in the scirs2-signal library, with automatic method selection
/// and parameter optimization.
///
/// # Arguments
///
/// * `signal` - Input noisy signal
/// * `config` - Denoising configuration
/// * `reference` - Optional reference clean signal for quality assessment
///
/// # Returns
///
/// * Comprehensive denoising result with quality metrics
///
/// # Examples
///
/// ```
/// use ndarray::Array1;
/// use scirs2_signal::denoise_unified::{denoise_unified, UnifiedDenoisingConfig, DenoisingMethod};
/// use scirs2_signal::denoise::{ThresholdMethod, ThresholdSelect};
/// use scirs2_signal::dwt::Wavelet;
///
/// // Create a noisy signal
/// let signal = Array1::from_vec(vec![1.0, 2.0, 1.5, 3.0, 2.5, 1.0]);
///
/// // Use default wavelet denoising
/// let config = UnifiedDenoisingConfig::default();
/// let result = denoise_unified(&signal, &config, None).unwrap();
///
/// // Use advanced wavelet denoising
/// let advanced_config = UnifiedDenoisingConfig {
///     method: DenoisingMethod::WaveletAdvanced {
///         config: scirs2_signal::denoise_advanced::AdvancedDenoiseConfig::default(),
///     },
///     ..Default::default()
/// };
/// let advanced_result = denoise_unified(&signal, &advanced_config, None).unwrap();
/// ```
pub fn denoise_unified(
    signal: &Array1<f64>,
    config: &UnifiedDenoisingConfig,
    reference: Option<&Array1<f64>>,
) -> SignalResult<UnifiedDenoisingResult> {
    let start_time = if config.benchmark {
        Some(std::time::Instant::now())
    } else {
        None
    };

    // Preprocessing
    let preprocessed = if config.enable_preprocessing {
        preprocess_signal(signal)?
    } else {
        signal.clone()
    };

    // Apply denoising method
    let (denoised, estimated_noise_level) = match &config.method {
        DenoisingMethod::WaveletBasic {
            wavelet,
            levels,
            threshold_method,
            threshold_select,
        } => {
            let denoised_vec = denoise_wavelet(
                preprocessed.as_slice().unwrap(),
                *wavelet,
                *levels,
                *threshold_method,
                *threshold_select,
                config.noise_level,
            )?;
            let denoised = Array1::from_vec(denoised_vec);
            let noise_level = config
                .noise_level
                .unwrap_or_else(|| estimate_noise_level(&preprocessed));
            (denoised, noise_level)
        }
        DenoisingMethod::WaveletAdvanced { config: adv_config } => {
            let result = advanced_denoise(preprocessed.as_slice().unwrap(), adv_config)?;
            (Array1::from_vec(result.signal), result.noise_level)
        }
        DenoisingMethod::DictionaryLearning {
            config: dict_config,
        } => {
            let denoised = denoise_dictionary_learning(&preprocessed, dict_config)?;
            let noise_level = config
                .noise_level
                .unwrap_or_else(|| estimate_noise_level(&preprocessed));
            (denoised, noise_level)
        }
        DenoisingMethod::Wiener {
            config: wiener_config,
        } => {
            let denoised = denoise_wiener_1d(&preprocessed, wiener_config)?;
            let noise_level = config
                .noise_level
                .unwrap_or_else(|| estimate_noise_level(&preprocessed));
            (denoised, noise_level)
        }
        DenoisingMethod::Median { window_size } => {
            let denoised = denoise_median_1d(&preprocessed, *window_size)?;
            let noise_level = config
                .noise_level
                .unwrap_or_else(|| estimate_noise_level(&preprocessed));
            (denoised, noise_level)
        }
        DenoisingMethod::TotalVariation { lambda, iterations } => {
            let denoised = denoise_total_variation_1d(&preprocessed, *lambda, *iterations)?;
            let noise_level = config
                .noise_level
                .unwrap_or_else(|| estimate_noise_level(&preprocessed));
            (denoised, noise_level)
        }
        DenoisingMethod::UltraAdvanced {
            config: ultra_config,
        } => {
            let result = ultra_advanced_denoise(&preprocessed, ultra_config)?;
            (
                result.denoised_signal,
                estimate_noise_level(&result.noise_estimate),
            )
        }
    };

    // Postprocessing
    let final_denoised = if config.enable_postprocessing {
        postprocess_signal(&denoised)?
    } else {
        denoised
    };

    // Calculate processing time
    let processing_time_ms = start_time.map(|start| start.elapsed().as_secs_f64() * 1000.0);

    // Calculate quality metrics
    let quality_metrics = calculate_quality_metrics(signal, &final_denoised, reference);

    // Calculate SNR improvement
    let snr_improvement = calculate_snr_improvement(signal, &final_denoised);

    Ok(UnifiedDenoisingResult {
        denoised: final_denoised,
        estimated_noise_level,
        snr_improvement,
        method_used: config.method.clone(),
        processing_time_ms,
        quality_metrics,
    })
}

/// Automatically select the best denoising method for a given signal
///
/// This function analyzes the input signal characteristics and automatically
/// selects the most appropriate denoising method.
///
/// # Arguments
///
/// * `signal` - Input noisy signal
/// * `noise_level` - Optional known noise level
///
/// # Returns
///
/// * Recommended denoising configuration
pub fn auto_select_denoising_method(
    signal: &Array1<f64>,
    noise_level: Option<f64>,
) -> SignalResult<UnifiedDenoisingConfig> {
    // Analyze signal characteristics
    let signal_length = signal.len();
    let signal_complexity = analyze_signal_complexity(signal);
    let estimated_noise = noise_level.unwrap_or_else(|| estimate_noise_level(signal));

    // Select method based on characteristics
    let method = if signal_length < 256 {
        // Short signals: use median filtering or simple wavelet
        if estimated_noise > 0.1 {
            DenoisingMethod::Median { window_size: 3 }
        } else {
            DenoisingMethod::WaveletBasic {
                wavelet: Wavelet::Haar,
                levels: Some(3),
                threshold_method: ThresholdMethod::Soft,
                threshold_select: ThresholdSelect::Universal,
            }
        }
    } else if signal_complexity > 0.7 {
        // Complex signals: use advanced methods
        DenoisingMethod::WaveletAdvanced {
            config: AdvancedDenoiseConfig {
                translation_invariant: true,
                adaptive: true,
                noise_estimation: NoiseEstimation::MAD,
                ..Default::default()
            },
        }
    } else if estimated_noise > 0.05 {
        // High noise: use robust methods
        DenoisingMethod::TotalVariation {
            lambda: 0.1,
            iterations: 100,
        }
    } else {
        // Low noise: use gentle wavelet denoising
        DenoisingMethod::WaveletBasic {
            wavelet: Wavelet::DB(4),
            levels: Some(4),
            threshold_method: ThresholdMethod::Soft,
            threshold_select: ThresholdSelect::Sure,
        }
    };

    Ok(UnifiedDenoisingConfig {
        method,
        noise_level,
        enable_preprocessing: true,
        enable_postprocessing: true,
        benchmark: false,
    })
}

/// Preprocess signal before denoising
fn preprocess_signal(signal: &Array1<f64>) -> SignalResult<Array1<f64>> {
    // Basic preprocessing: remove DC offset
    let mean = signal.mean().unwrap_or(0.0);
    Ok(signal.mapv(|x| x - mean))
}

/// Postprocess signal after denoising
fn postprocess_signal(signal: &Array1<f64>) -> SignalResult<Array1<f64>> {
    // Basic postprocessing: ensure no NaN or infinite values
    let cleaned = signal.mapv(|x| if x.is_finite() { x } else { 0.0 });
    Ok(cleaned)
}

/// Estimate noise level from signal
fn estimate_noise_level(signal: &Array1<f64>) -> f64 {
    if signal.len() < 2 {
        return 0.0;
    }

    // Use differences between adjacent samples as noise estimate
    let mut diffs = Vec::with_capacity(signal.len() - 1);
    for i in 0..signal.len() - 1 {
        diffs.push((signal[i + 1] - signal[i]).abs());
    }

    // Use median of differences as robust noise estimate
    diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    diffs[diffs.len() / 2] * 0.6745 // Scale factor for Gaussian noise
}

/// Analyze signal complexity
fn analyze_signal_complexity(signal: &Array1<f64>) -> f64 {
    if signal.len() < 4 {
        return 0.0;
    }

    // Calculate second derivative as complexity measure
    let mut complexity_sum = 0.0;
    for i in 1..signal.len() - 1 {
        let second_deriv = signal[i + 1] - 2.0 * signal[i] + signal[i - 1];
        complexity_sum += second_deriv.abs();
    }

    let signal_range = signal.mapv(|x| x.abs()).into_iter().fold(0.0, f64::max);
    if signal_range > 0.0 {
        (complexity_sum / ((signal.len() - 2) as f64)) / signal_range
    } else {
        0.0
    }
}

/// Calculate quality metrics
fn calculate_quality_metrics(
    original: &Array1<f64>,
    denoised: &Array1<f64>,
    reference: Option<&Array1<f64>>,
) -> QualityMetrics {
    let mse = reference.map(|ref_sig| {
        if ref_sig.len() == denoised.len() {
            ref_sig
                .iter()
                .zip(denoised.iter())
                .map(|(&r, &d)| (r - d).powi(2))
                .sum::<f64>()
                / ref_sig.len() as f64
        } else {
            f64::NAN
        }
    });

    let psnr = mse.map(|mse_val| {
        if mse_val > 0.0 {
            let max_val = reference
                .unwrap()
                .mapv(|x| x.abs())
                .into_iter()
                .fold(0.0, f64::max);
            20.0 * (max_val / mse_val.sqrt()).log10()
        } else {
            f64::INFINITY
        }
    });

    // Simple signal preservation metric
    let signal_power = original.mapv(|x| x * x).sum();
    let denoised_power = denoised.mapv(|x| x * x).sum();
    let signal_preservation = if signal_power > 0.0 {
        (denoised_power / signal_power).min(1.0)
    } else {
        1.0
    };

    // Simple noise reduction metric based on high-frequency content reduction
    let original_hf = calculate_high_frequency_energy(original);
    let denoised_hf = calculate_high_frequency_energy(denoised);
    let noise_reduction = if original_hf > 0.0 {
        ((original_hf - denoised_hf) / original_hf)
            .max(0.0)
            .min(1.0)
    } else {
        0.0
    };

    QualityMetrics {
        mse,
        psnr,
        ssim: None, // Complex SSIM calculation not implemented here
        signal_preservation,
        noise_reduction,
    }
}

/// Calculate high-frequency energy (simple approximation)
fn calculate_high_frequency_energy(signal: &Array1<f64>) -> f64 {
    if signal.len() < 2 {
        return 0.0;
    }

    signal
        .windows(2)
        .into_iter()
        .map(|window| (window[1] - window[0]).powi(2))
        .sum()
}

/// Calculate SNR improvement
fn calculate_snr_improvement(original: &Array1<f64>, denoised: &Array1<f64>) -> Option<f64> {
    if original.len() != denoised.len() {
        return None;
    }

    // Estimate noise as difference
    let noise: Array1<f64> = original - denoised;

    let signal_power = denoised.mapv(|x| x * x).sum() / denoised.len() as f64;
    let noise_power = noise.mapv(|x| x * x).sum() / noise.len() as f64;

    if noise_power > 0.0 && signal_power > 0.0 {
        Some(10.0 * (signal_power / noise_power).log10())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_unified_denoising_basic() {
        // Create a simple test signal
        let n = 64;
        let signal: Array1<f64> = Array1::from_shape_fn(n, |i| {
            (2.0 * PI * i as f64 / n as f64 * 4.0).sin() + 0.1 * (i as f64 * 0.1).sin()
        });

        let config = UnifiedDenoisingConfig::default();
        let result = denoise_unified(&signal, &config, None).unwrap();

        assert_eq!(result.denoised.len(), signal.len());
        assert!(result.estimated_noise_level >= 0.0);
        assert!(result.quality_metrics.signal_preservation >= 0.0);
        assert!(result.quality_metrics.signal_preservation <= 1.0);
    }

    #[test]
    fn test_auto_select_denoising_method() {
        // Test with a simple signal
        let signal = Array1::from_vec(vec![1.0, 2.0, 1.5, 3.0, 2.5, 1.0]);
        let config = auto_select_denoising_method(&signal, None).unwrap();

        // Should select a method appropriate for short signals
        match config.method {
            DenoisingMethod::Median { .. } | DenoisingMethod::WaveletBasic { .. } => {}
            _ => panic!("Unexpected method selection for short signal"),
        }
    }

    #[test]
    fn test_noise_level_estimation() {
        // Create a signal with known characteristics
        let clean_signal = Array1::from_vec(vec![1.0; 100]);
        let noise_level = estimate_noise_level(&clean_signal);

        // Should detect very low noise in constant signal
        assert!(noise_level < 0.1);
    }
}
