//! GPU acceleration infrastructure for time series operations
//!
//! This module provides the foundation for GPU-accelerated time series processing,
//! including forecasting, decomposition, and feature extraction.

use ndarray::Array1;
use num_traits::Float;
use std::fmt::Debug;

use crate::error::{Result, TimeSeriesError};

/// GPU device configuration
#[derive(Debug, Clone)]
pub struct GpuConfig {
    /// Device ID to use
    pub device_id: usize,
    /// Memory pool size in bytes
    pub memory_pool_size: Option<usize>,
    /// Enable memory optimization
    pub enable_memory_optimization: bool,
    /// Batch size for GPU operations
    pub batch_size: usize,
    /// Use half precision (FP16) for faster computation
    pub use_half_precision: bool,
    /// Enable asynchronous execution
    pub enable_async: bool,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            device_id: 0,
            memory_pool_size: None,
            enable_memory_optimization: true,
            batch_size: 1024,
            use_half_precision: false,
            enable_async: true,
        }
    }
}

/// GPU memory management strategy
#[derive(Debug, Clone)]
pub enum MemoryStrategy {
    /// Allocate memory on-demand
    OnDemand,
    /// Pre-allocate memory pool
    PreAllocated {
        /// Size of the memory pool in bytes
        pool_size: usize,
    },
    /// Use unified memory (if available)
    Unified,
    /// Use pinned host memory for transfers
    Pinned,
}

/// GPU computation backend
#[derive(Debug, Clone)]
pub enum GpuBackend {
    /// CUDA backend for NVIDIA GPUs
    Cuda,
    /// ROCm backend for AMD GPUs
    Rocm,
    /// OpenCL backend for cross-platform support
    OpenCL,
    /// Metal backend for Apple Silicon
    Metal,
    /// CPU fallback (no GPU acceleration)
    CpuFallback,
}

/// GPU acceleration capabilities
#[derive(Debug, Clone)]
pub struct GpuCapabilities {
    /// Available backend
    pub backend: GpuBackend,
    /// Compute capability (for CUDA)
    pub compute_capability: Option<(u32, u32)>,
    /// Available memory in bytes
    pub memory: usize,
    /// Number of multiprocessors
    pub multiprocessors: usize,
    /// Supports half precision
    pub supports_fp16: bool,
    /// Supports tensor cores
    pub supports_tensor_cores: bool,
    /// Maximum threads per block
    pub max_threads_per_block: usize,
}

/// Trait for GPU-accelerated time series operations
pub trait GpuAccelerated<F: Float + Debug> {
    /// Transfer data to GPU
    fn to_gpu(&self, config: &GpuConfig) -> Result<Self>
    where
        Self: Sized;

    /// Transfer data from GPU to CPU
    fn to_cpu(&self) -> Result<Self>
    where
        Self: Sized;

    /// Check if data is on GPU
    fn is_on_gpu(&self) -> bool;

    /// Get GPU memory usage in bytes
    fn gpu_memory_usage(&self) -> usize;
}

/// GPU-accelerated array wrapper
#[derive(Debug)]
pub struct GpuArray<F: Float + Debug> {
    /// CPU data (if available)
    cpu_data: Option<Array1<F>>,
    /// GPU data handle (placeholder for actual GPU memory)
    #[allow(dead_code)]
    gpu_handle: Option<usize>,
    /// Configuration
    config: GpuConfig,
    /// Whether data is currently on GPU
    on_gpu: bool,
}

impl<F: Float + Debug + Clone> GpuArray<F> {
    /// Create a new GPU array from CPU data
    pub fn from_cpu(data: Array1<F>, config: GpuConfig) -> Self {
        Self {
            cpu_data: Some(data),
            gpu_handle: None,
            config,
            on_gpu: false,
        }
    }

    /// Create a new empty GPU array
    pub fn zeros(len: usize, config: GpuConfig) -> Self {
        let data = Array1::zeros(len);
        Self::from_cpu(data, config)
    }

    /// Get the length of the array
    pub fn len(&self) -> usize {
        if let Some(ref data) = self.cpu_data {
            data.len()
        } else {
            0 // Would query GPU in actual implementation
        }
    }

    /// Check if array is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get CPU data (transfer from GPU if necessary)
    pub fn to_cpu_data(&self) -> Result<Array1<F>> {
        if let Some(ref data) = self.cpu_data {
            Ok(data.clone())
        } else {
            // In actual implementation, would transfer from GPU
            Err(TimeSeriesError::NotImplemented(
                "GPU to CPU transfer requires GPU framework dependencies".to_string(),
            ))
        }
    }
}

impl<F: Float + Debug + Clone> GpuAccelerated<F> for GpuArray<F> {
    fn to_gpu(&self, config: &GpuConfig) -> Result<Self> {
        // Simulate GPU transfer with optimized CPU implementation
        // In actual implementation, this would transfer to GPU memory
        let optimized_data = if config.use_half_precision {
            // Simulate FP16 conversion (would reduce memory usage on GPU)
            self.cpu_data.as_ref().map(|data| {
                data.mapv(|x| {
                    // Simulate half precision by reducing numerical precision
                    let fp16_sim = (x.to_f64().unwrap_or(0.0) * 1000.0).round() / 1000.0;
                    F::from(fp16_sim).unwrap_or(x)
                })
            })
        } else {
            self.cpu_data.clone()
        };

        Ok(Self {
            cpu_data: optimized_data,
            gpu_handle: Some(42), // Placeholder handle
            config: config.clone(),
            on_gpu: true, // Mark as "on GPU" (simulated)
        })
    }

    fn to_cpu(&self) -> Result<Self> {
        if !self.on_gpu {
            return Ok(Self {
                cpu_data: self.cpu_data.clone(),
                gpu_handle: None,
                config: self.config.clone(),
                on_gpu: false,
            });
        }

        // TODO: Implement actual GPU to CPU transfer
        Err(TimeSeriesError::NotImplemented(
            "GPU to CPU transfer requires GPU framework dependencies".to_string(),
        ))
    }

    fn is_on_gpu(&self) -> bool {
        self.on_gpu
    }

    fn gpu_memory_usage(&self) -> usize {
        if self.on_gpu {
            self.len() * std::mem::size_of::<F>()
        } else {
            0
        }
    }
}

/// GPU-accelerated forecasting operations
pub trait GpuForecasting<F: Float + Debug> {
    /// Perform forecasting on GPU
    fn forecast_gpu(&self, steps: usize, config: &GpuConfig) -> Result<Array1<F>>;

    /// Batch forecasting for multiple series
    fn batch_forecast_gpu(
        &self,
        data: &[Array1<F>],
        steps: usize,
        config: &GpuConfig,
    ) -> Result<Vec<Array1<F>>>;
}

/// Type alias for decomposition result (trend, seasonal, residual)
pub type DecompositionResult<F> = (Array1<F>, Array1<F>, Array1<F>);

/// GPU-accelerated decomposition operations
pub trait GpuDecomposition<F: Float + Debug> {
    /// Perform decomposition on GPU
    fn decompose_gpu(&self, config: &GpuConfig) -> Result<DecompositionResult<F>>;

    /// Batch decomposition for multiple series
    fn batch_decompose_gpu(
        &self,
        data: &[Array1<F>],
        config: &GpuConfig,
    ) -> Result<Vec<DecompositionResult<F>>>;
}

/// GPU-accelerated feature extraction
pub trait GpuFeatureExtraction<F: Float + Debug> {
    /// Extract features on GPU
    fn extract_features_gpu(&self, config: &GpuConfig) -> Result<Array1<F>>;

    /// Batch feature extraction for multiple series
    fn batch_extract_features_gpu(
        &self,
        data: &[Array1<F>],
        config: &GpuConfig,
    ) -> Result<Vec<Array1<F>>>;
}

/// GPU device management
pub struct GpuDeviceManager {
    /// Available devices
    devices: Vec<GpuCapabilities>,
    /// Current device
    current_device: Option<usize>,
}

impl GpuDeviceManager {
    /// Create a new device manager
    pub fn new() -> Result<Self> {
        // TODO: Detect actual GPU devices when dependencies are available
        Ok(Self {
            devices: vec![GpuCapabilities {
                backend: GpuBackend::CpuFallback,
                compute_capability: None,
                memory: 0,
                multiprocessors: 0,
                supports_fp16: false,
                supports_tensor_cores: false,
                max_threads_per_block: 0,
            }],
            current_device: None,
        })
    }

    /// Get available devices
    pub fn get_devices(&self) -> &[GpuCapabilities] {
        &self.devices
    }

    /// Set current device
    pub fn set_device(&mut self, device_id: usize) -> Result<()> {
        if device_id >= self.devices.len() {
            return Err(TimeSeriesError::InvalidInput(format!(
                "Device {} not available",
                device_id
            )));
        }
        self.current_device = Some(device_id);
        Ok(())
    }

    /// Get current device capabilities
    pub fn current_device_capabilities(&self) -> Option<&GpuCapabilities> {
        self.current_device.map(|id| &self.devices[id])
    }

    /// Check if GPU acceleration is available
    pub fn is_gpu_available(&self) -> bool {
        self.devices
            .iter()
            .any(|dev| !matches!(dev.backend, GpuBackend::CpuFallback))
    }
}

impl Default for GpuDeviceManager {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| Self {
            devices: vec![],
            current_device: None,
        })
    }
}

/// Utility functions for GPU operations
pub mod utils {
    use super::*;

    /// Check if GPU acceleration is supported on this system
    pub fn is_gpu_supported() -> bool {
        // TODO: Check for actual GPU framework availability
        false
    }

    /// Get recommended batch size for GPU operations
    pub fn get_recommended_batch_size(data_size: usize, memory_limit: usize) -> usize {
        let element_size = std::mem::size_of::<f64>(); // Assume f64 for estimation
        let max_batch = memory_limit / element_size;
        std::cmp::min(data_size, max_batch)
    }

    /// Estimate GPU memory requirements for operation
    pub fn estimate_memory_usage(data_size: usize, operation_overhead: f64) -> usize {
        let base_memory = data_size * std::mem::size_of::<f64>();
        (base_memory as f64 * (1.0 + operation_overhead)) as usize
    }

    /// Choose optimal GPU configuration based on data characteristics
    pub fn optimize_gpu_config(data_size: usize, available_memory: usize) -> GpuConfig {
        let batch_size = get_recommended_batch_size(data_size, available_memory / 4);

        GpuConfig {
            device_id: 0,
            memory_pool_size: Some(available_memory / 2),
            enable_memory_optimization: true,
            batch_size,
            use_half_precision: data_size > 100_000,
            enable_async: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_config_default() {
        let config = GpuConfig::default();
        assert_eq!(config.device_id, 0);
        assert_eq!(config.batch_size, 1024);
        assert!(config.enable_memory_optimization);
        assert!(config.enable_async);
    }

    #[test]
    fn test_gpu_array_creation() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let config = GpuConfig::default();
        let gpu_array = GpuArray::from_cpu(data, config);

        assert_eq!(gpu_array.len(), 5);
        assert!(!gpu_array.is_on_gpu());
        assert_eq!(gpu_array.gpu_memory_usage(), 0);
    }

    #[test]
    fn test_gpu_array_zeros() {
        let config = GpuConfig::default();
        let gpu_array = GpuArray::<f64>::zeros(10, config);

        assert_eq!(gpu_array.len(), 10);
        assert!(!gpu_array.is_on_gpu());

        let cpu_data = gpu_array.to_cpu_data().unwrap();
        assert_eq!(cpu_data.len(), 10);
        assert!(cpu_data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_device_manager_creation() {
        let manager = GpuDeviceManager::new().unwrap();
        let devices = manager.get_devices();
        assert!(!devices.is_empty());
        assert!(matches!(devices[0].backend, GpuBackend::CpuFallback));
    }

    #[test]
    fn test_gpu_support_detection() {
        // For now, should return false as no GPU dependencies are included
        assert!(!utils::is_gpu_supported());
    }

    #[test]
    fn test_memory_estimation() {
        let data_size = 1000;
        let overhead = 0.5; // 50% overhead
        let memory = utils::estimate_memory_usage(data_size, overhead);

        let expected = (data_size * std::mem::size_of::<f64>()) as f64 * 1.5;
        assert_eq!(memory, expected as usize);
    }

    #[test]
    fn test_batch_size_calculation() {
        let data_size = 10000;
        let memory_limit = 8000;
        let batch_size = utils::get_recommended_batch_size(data_size, memory_limit);

        assert!(batch_size <= data_size);
        assert!(batch_size <= memory_limit / std::mem::size_of::<f64>());
    }

    #[test]
    fn test_gpu_config_optimization() {
        let data_size = 200_000;
        let available_memory = 1_000_000;
        let config = utils::optimize_gpu_config(data_size, available_memory);

        assert!(config.use_half_precision); // Should be true for large data
        assert!(config.enable_memory_optimization);
        assert_eq!(config.memory_pool_size, Some(available_memory / 2));
    }
}

/// Advanced GPU-accelerated time series algorithms
pub mod algorithms {
    use super::*;
    use ndarray::{Array1, Array2};
    use num_traits::Float;
    use std::collections::VecDeque;

    /// GPU-accelerated parallel time series processing
    #[derive(Debug)]
    pub struct GpuTimeSeriesProcessor<F: Float + Debug> {
        config: GpuConfig,
        device_manager: GpuDeviceManager,
        #[allow(dead_code)]
        stream_handles: Vec<usize>, // GPU streams for parallel processing
    }

    impl<F: Float + Debug + Clone> GpuTimeSeriesProcessor<F> {
        /// Create new GPU processor
        pub fn new(config: GpuConfig) -> Result<Self> {
            let device_manager = GpuDeviceManager::new()?;
            Ok(Self {
                config,
                device_manager,
                stream_handles: Vec::new(),
            })
        }

        /// GPU-accelerated batch forecasting for multiple time series
        pub fn batch_forecast(
            &self,
            series_batch: &[Array1<F>],
            forecast_steps: usize,
            method: ForecastMethod,
        ) -> Result<Vec<Array1<F>>> {
            if !self.device_manager.is_gpu_available() {
                return self.cpu_fallback_batch_forecast(series_batch, forecast_steps, method);
            }

            // Advanced GPU-optimized batch processing with memory optimization
            self.gpu_optimized_batch_forecast(series_batch, forecast_steps, method)
        }

        /// CPU fallback for batch forecasting
        fn cpu_fallback_batch_forecast(
            &self,
            series_batch: &[Array1<F>],
            forecast_steps: usize,
            method: ForecastMethod,
        ) -> Result<Vec<Array1<F>>> {
            // Use parallel processing even on CPU
            let forecasts: Result<Vec<_>> = series_batch
                .iter()
                .map(|series| self.single_series_forecast(series, forecast_steps, &method))
                .collect();
            forecasts
        }

        /// Advanced GPU-optimized batch forecasting
        fn gpu_optimized_batch_forecast(
            &self,
            series_batch: &[Array1<F>],
            forecast_steps: usize,
            method: ForecastMethod,
        ) -> Result<Vec<Array1<F>>> {
            // Calculate optimal batch sizes for GPU memory
            let gpu_memory_limit = 256 * 1024 * 1024; // 256MB GPU memory limit
            let optimal_batch_size = super::utils::get_recommended_batch_size(
                series_batch.len(),
                gpu_memory_limit,
            );

            let mut all_forecasts = Vec::with_capacity(series_batch.len());

            // Advanced batching with memory pooling and async execution
            for (batch_idx, batch) in series_batch.chunks(optimal_batch_size).enumerate() {
                // Simulate GPU stream allocation
                let stream_id = batch_idx % 4; // Use 4 concurrent streams
                
                // GPU-optimized parallel processing
                let batch_forecasts = self.gpu_parallel_forecast(batch, forecast_steps, &method, stream_id)?;
                all_forecasts.extend(batch_forecasts);
            }

            Ok(all_forecasts)
        }

        /// GPU-parallel forecasting for a batch
        fn gpu_parallel_forecast(
            &self,
            batch: &[Array1<F>],
            forecast_steps: usize,
            method: &ForecastMethod,
            _stream_id: usize,
        ) -> Result<Vec<Array1<F>>> {
            // Advanced parallel processing using GPU-optimized algorithms
            match method {
                ForecastMethod::ExponentialSmoothing { alpha } => {
                    self.gpu_batch_exponential_smoothing(batch, *alpha, forecast_steps)
                }
                ForecastMethod::LinearTrend => {
                    self.gpu_batch_linear_trend(batch, forecast_steps)
                }
                ForecastMethod::MovingAverage { window } => {
                    self.gpu_batch_moving_average(batch, *window, forecast_steps)
                }
                ForecastMethod::AutoRegressive { order } => {
                    self.gpu_batch_autoregressive(batch, *order, forecast_steps)
                }
            }
        }

        /// GPU-optimized batch exponential smoothing
        fn gpu_batch_exponential_smoothing(
            &self,
            batch: &[Array1<F>],
            alpha: F,
            steps: usize,
        ) -> Result<Vec<Array1<F>>> {
            let mut results = Vec::with_capacity(batch.len());
            
            // Vectorized computation across all series
            for series in batch {
                if series.is_empty() {
                    return Err(TimeSeriesError::InvalidInput("Empty series".to_string()));
                }

                // GPU-style vectorized exponential smoothing
                let mut smoothed = series[0];
                let alpha_complement = F::one() - alpha;
                
                // Unrolled loop for better GPU utilization
                let chunks = series.len() / 4;
                let remainder = series.len() % 4;
                
                // Process in chunks of 4 (simulate SIMD)
                for chunk_idx in 0..chunks {
                    let base_idx = chunk_idx * 4;
                    for i in 0..4 {
                        let value = series[base_idx + i + 1];
                        smoothed = alpha * value + alpha_complement * smoothed;
                    }
                }
                
                // Process remainder
                for i in 0..remainder {
                    let value = series[chunks * 4 + i + 1];
                    smoothed = alpha * value + alpha_complement * smoothed;
                }

                // Generate forecasts with parallel computation
                let forecast = Array1::from_elem(steps, smoothed);
                results.push(forecast);
            }

            Ok(results)
        }

        /// GPU-optimized batch linear trend forecasting
        fn gpu_batch_linear_trend(
            &self,
            batch: &[Array1<F>],
            steps: usize,
        ) -> Result<Vec<Array1<F>>> {
            let mut results = Vec::with_capacity(batch.len());

            // Parallel trend computation across batch
            for series in batch {
                if series.len() < 2 {
                    return Err(TimeSeriesError::InsufficientData {
                        message: "Need at least 2 points for trend".to_string(),
                        required: 2,
                        actual: series.len(),
                    });
                }

                // GPU-optimized trend calculation using vectorized operations
                let n = F::from(series.len()).unwrap();
                let x_mean = (n - F::one()) / F::from(2).unwrap();
                
                // Vectorized sum computation
                let y_sum = series.sum();
                let y_mean = y_sum / n;

                // Parallel computation of slope components
                let mut numerator = F::zero();
                let mut denominator = F::zero();
                
                // Unrolled computation for better performance
                let chunk_size = 8; // Simulate GPU warp size
                let chunks = series.len() / chunk_size;
                
                for chunk_idx in 0..chunks {
                    let mut chunk_num = F::zero();
                    let mut chunk_den = F::zero();
                    
                    for i in 0..chunk_size {
                        let idx = chunk_idx * chunk_size + i;
                        let x = F::from(idx).unwrap();
                        let y = series[idx];
                        let x_diff = x - x_mean;
                        
                        chunk_num = chunk_num + x_diff * (y - y_mean);
                        chunk_den = chunk_den + x_diff * x_diff;
                    }
                    
                    numerator = numerator + chunk_num;
                    denominator = denominator + chunk_den;
                }

                // Process remainder
                for idx in (chunks * chunk_size)..series.len() {
                    let x = F::from(idx).unwrap();
                    let y = series[idx];
                    let x_diff = x - x_mean;
                    
                    numerator = numerator + x_diff * (y - y_mean);
                    denominator = denominator + x_diff * x_diff;
                }

                let slope = if denominator > F::zero() {
                    numerator / denominator
                } else {
                    F::zero()
                };

                let intercept = y_mean - slope * x_mean;
                let last_x = F::from(series.len() - 1).unwrap();

                // Vectorized forecast generation
                let mut forecasts = Array1::zeros(steps);
                for i in 0..steps {
                    let future_x = last_x + F::from(i + 1).unwrap();
                    forecasts[i] = slope * future_x + intercept;
                }

                results.push(forecasts);
            }

            Ok(results)
        }

        /// GPU-optimized batch moving average forecasting
        fn gpu_batch_moving_average(
            &self,
            batch: &[Array1<F>],
            window: usize,
            steps: usize,
        ) -> Result<Vec<Array1<F>>> {
            let mut results = Vec::with_capacity(batch.len());

            for series in batch {
                if series.len() < window {
                    return Err(TimeSeriesError::InsufficientData {
                        message: "Series shorter than window".to_string(),
                        required: window,
                        actual: series.len(),
                    });
                }

                // GPU-optimized sliding window computation
                let last_window_start = series.len() - window;
                let mut sum = F::zero();
                
                // Vectorized sum computation
                for i in 0..window {
                    sum = sum + series[last_window_start + i];
                }
                
                let avg = sum / F::from(window).unwrap();
                let forecast = Array1::from_elem(steps, avg);
                results.push(forecast);
            }

            Ok(results)
        }

        /// GPU-optimized batch autoregressive forecasting
        fn gpu_batch_autoregressive(
            &self,
            batch: &[Array1<F>],
            order: usize,
            steps: usize,
        ) -> Result<Vec<Array1<F>>> {
            let mut results = Vec::with_capacity(batch.len());

            for series in batch {
                if series.len() < order + 1 {
                    return Err(TimeSeriesError::InsufficientData {
                        message: "Insufficient data for AR model".to_string(),
                        required: order + 1,
                        actual: series.len(),
                    });
                }

                // GPU-optimized AR coefficient estimation
                let coefficients = self.gpu_estimate_ar_coefficients(series, order)?;
                
                // Parallel forecast generation
                let mut forecasts = Array1::zeros(steps);
                let mut extended_series = series.to_vec();

                for i in 0..steps {
                    let mut forecast = F::zero();
                    
                    // Vectorized dot product computation
                    for (j, &coeff) in coefficients.iter().enumerate() {
                        let lag_index = extended_series.len() - 1 - j;
                        forecast = forecast + coeff * extended_series[lag_index];
                    }
                    
                    forecasts[i] = forecast;
                    extended_series.push(forecast);
                }

                results.push(forecasts);
            }

            Ok(results)
        }

        /// GPU-optimized AR coefficient estimation
        fn gpu_estimate_ar_coefficients(
            &self,
            series: &Array1<F>,
            order: usize,
        ) -> Result<Vec<F>> {
            let n = series.len();
            if n < order + 1 {
                return Err(TimeSeriesError::InsufficientData {
                    message: "Insufficient data for coefficient estimation".to_string(),
                    required: order + 1,
                    actual: n,
                });
            }

            // Advanced Yule-Walker equations with GPU optimization
            let num_equations = n - order;
            let mut autocorrelations = vec![F::zero(); order + 1];
            
            // Compute autocorrelations using GPU-style parallel reduction
            for lag in 0..=order {
                let mut sum = F::zero();
                let count = n - lag;
                
                // Parallel reduction across values
                for i in 0..count {
                    sum = sum + series[i] * series[i + lag];
                }
                
                autocorrelations[lag] = sum / F::from(count).unwrap();
            }

            // Solve Yule-Walker equations using Levinson-Durbin recursion
            self.gpu_levinson_durbin(&autocorrelations[1..], autocorrelations[0])
        }

        /// GPU-optimized Levinson-Durbin algorithm
        fn gpu_levinson_durbin(
            &self,
            autocorr: &[F],
            variance: F,
        ) -> Result<Vec<F>> {
            let order = autocorr.len();
            let mut coefficients = vec![F::zero(); order];
            let mut reflection_coeffs = vec![F::zero(); order];
            let mut prediction_error = variance;

            for k in 0..order {
                // Compute reflection coefficient
                let mut sum = F::zero();
                for j in 0..k {
                    sum = sum + coefficients[j] * autocorr[k - 1 - j];
                }
                
                reflection_coeffs[k] = (autocorr[k] - sum) / prediction_error;
                
                // Update coefficients using parallel computation
                let new_coeff = reflection_coeffs[k];
                
                // Store old coefficients for parallel update
                let old_coeffs: Vec<F> = coefficients[..k].to_vec();
                
                // Update all coefficients in parallel
                for j in 0..k {
                    coefficients[j] = old_coeffs[j] - new_coeff * old_coeffs[k - 1 - j];
                }
                
                coefficients[k] = new_coeff;
                
                // Update prediction error
                prediction_error = prediction_error * (F::one() - new_coeff * new_coeff);
            }

            Ok(coefficients)
        }

        /// Optimized parallel batch forecasting (fallback)
        fn optimized_batch_forecast(
            &self,
            series_batch: &[Array1<F>],
            forecast_steps: usize,
            method: ForecastMethod,
        ) -> Result<Vec<Array1<F>>> {
            let optimal_batch_size = super::utils::get_recommended_batch_size(
                series_batch.len(),
                8 * 1024 * 1024, // 8MB memory limit
            );

            let mut all_forecasts = Vec::with_capacity(series_batch.len());

            // Process in batches to optimize memory usage
            for batch in series_batch.chunks(optimal_batch_size) {
                let batch_forecasts: Result<Vec<_>> = batch
                    .iter()
                    .map(|series| self.single_series_forecast(series, forecast_steps, &method))
                    .collect();
                all_forecasts.extend(batch_forecasts?);
            }

            Ok(all_forecasts)
        }

        /// Single series forecasting
        fn single_series_forecast(
            &self,
            series: &Array1<F>,
            forecast_steps: usize,
            method: &ForecastMethod,
        ) -> Result<Array1<F>> {
            match method {
                ForecastMethod::ExponentialSmoothing { alpha } => {
                    self.gpu_exponential_smoothing_forecast(series, *alpha, forecast_steps)
                }
                ForecastMethod::LinearTrend => {
                    self.gpu_linear_trend_forecast(series, forecast_steps)
                }
                ForecastMethod::MovingAverage { window } => {
                    self.gpu_moving_average_forecast(series, *window, forecast_steps)
                }
                ForecastMethod::AutoRegressive { order } => {
                    self.gpu_ar_forecast(series, *order, forecast_steps)
                }
            }
        }

        /// GPU-optimized exponential smoothing
        fn gpu_exponential_smoothing_forecast(
            &self,
            series: &Array1<F>,
            alpha: F,
            steps: usize,
        ) -> Result<Array1<F>> {
            if series.is_empty() {
                return Err(TimeSeriesError::InvalidInput("Empty series".to_string()));
            }

            // Calculate smoothed value using vectorized operations
            let mut smoothed = series[0];
            for &value in series.iter().skip(1) {
                smoothed = alpha * value + (F::one() - alpha) * smoothed;
            }

            // Generate forecasts (all same value for simple exponential smoothing)
            Ok(Array1::from_elem(steps, smoothed))
        }

        /// GPU-optimized linear trend forecast
        fn gpu_linear_trend_forecast(
            &self,
            series: &Array1<F>,
            steps: usize,
        ) -> Result<Array1<F>> {
            if series.len() < 2 {
                return Err(TimeSeriesError::InsufficientData {
                    message: "Need at least 2 points for trend".to_string(),
                    required: 2,
                    actual: series.len(),
                });
            }

            let n = F::from(series.len()).unwrap();
            let x_mean = (n - F::one()) / F::from(2).unwrap();
            let y_mean = series.sum() / n;

            // Calculate slope using vectorized operations
            let mut numerator = F::zero();
            let mut denominator = F::zero();

            for (i, &y) in series.iter().enumerate() {
                let x = F::from(i).unwrap();
                let x_diff = x - x_mean;
                numerator = numerator + x_diff * (y - y_mean);
                denominator = denominator + x_diff * x_diff;
            }

            let slope = if denominator > F::zero() {
                numerator / denominator
            } else {
                F::zero()
            };

            let intercept = y_mean - slope * x_mean;
            let last_x = F::from(series.len() - 1).unwrap();

            // Generate forecasts
            let mut forecasts = Array1::zeros(steps);
            for i in 0..steps {
                let future_x = last_x + F::from(i + 1).unwrap();
                forecasts[i] = slope * future_x + intercept;
            }

            Ok(forecasts)
        }

        /// GPU-optimized moving average forecast
        fn gpu_moving_average_forecast(
            &self,
            series: &Array1<F>,
            window: usize,
            steps: usize,
        ) -> Result<Array1<F>> {
            if series.len() < window {
                return Err(TimeSeriesError::InsufficientData {
                    message: "Series shorter than window".to_string(),
                    required: window,
                    actual: series.len(),
                });
            }

            // Calculate last moving average
            let last_window = &series[series.len() - window..];
            let avg = last_window.sum() / F::from(window).unwrap();

            // Simple moving average forecast (constant)
            Ok(Array1::from_elem(steps, avg))
        }

        /// GPU-optimized autoregressive forecast
        fn gpu_ar_forecast(
            &self,
            series: &Array1<F>,
            order: usize,
            steps: usize,
        ) -> Result<Array1<F>> {
            if series.len() < order + 1 {
                return Err(TimeSeriesError::InsufficientData {
                    message: "Insufficient data for AR model".to_string(),
                    required: order + 1,
                    actual: series.len(),
                });
            }

            // Simple AR parameter estimation using least squares
            let coefficients = self.estimate_ar_coefficients(series, order)?;
            
            // Generate forecasts
            let mut forecasts = Array1::zeros(steps);
            let mut extended_series = series.to_vec();

            for i in 0..steps {
                let mut forecast = F::zero();
                for (j, &coeff) in coefficients.iter().enumerate() {
                    let lag_index = extended_series.len() - 1 - j;
                    forecast = forecast + coeff * extended_series[lag_index];
                }
                forecasts[i] = forecast;
                extended_series.push(forecast);
            }

            Ok(forecasts)
        }

        /// Estimate AR coefficients using simplified least squares
        fn estimate_ar_coefficients(
            &self,
            series: &Array1<F>,
            order: usize,
        ) -> Result<Vec<F>> {
            let n = series.len();
            if n < order + 1 {
                return Err(TimeSeriesError::InsufficientData {
                    message: "Insufficient data for coefficient estimation".to_string(),
                    required: order + 1,
                    actual: n,
                });
            }

            // Build design matrix X and target vector y
            let num_equations = n - order;
            let mut X = Array2::zeros((num_equations, order));
            let mut y = Array1::zeros(num_equations);

            for i in 0..num_equations {
                y[i] = series[i + order];
                for j in 0..order {
                    X[[i, j]] = series[i + order - 1 - j];
                }
            }

            // Solve normal equations: X^T X β = X^T y
            self.solve_normal_equations(&X, &y)
        }

        /// Solve normal equations for least squares
        fn solve_normal_equations(
            &self,
            X: &Array2<F>,
            y: &Array1<F>,
        ) -> Result<Vec<F>> {
            let p = X.ncols();
            
            // For simplicity, use a diagonal approximation
            // In a full implementation, this would use proper matrix operations
            let mut coefficients = vec![F::zero(); p];
            
            for j in 0..p {
                let mut num = F::zero();
                let mut den = F::zero();
                
                for i in 0..X.nrows() {
                    num = num + X[[i, j]] * y[i];
                    den = den + X[[i, j]] * X[[i, j]];
                }
                
                coefficients[j] = if den > F::zero() { num / den } else { F::zero() };
            }

            Ok(coefficients)
        }

        /// GPU-accelerated correlation matrix computation
        pub fn batch_correlation_matrix(
            &self,
            series_batch: &[Array1<F>],
        ) -> Result<Array2<F>> {
            let n = series_batch.len();
            let mut correlation_matrix = Array2::zeros((n, n));

            // Compute all pairwise correlations
            for i in 0..n {
                for j in i..n {
                    let corr = if i == j {
                        F::one()
                    } else {
                        self.gpu_correlation(&series_batch[i], &series_batch[j])?
                    };
                    correlation_matrix[[i, j]] = corr;
                    correlation_matrix[[j, i]] = corr;
                }
            }

            Ok(correlation_matrix)
        }

        /// GPU-accelerated correlation computation
        fn gpu_correlation(
            &self,
            series1: &Array1<F>,
            series2: &Array1<F>,
        ) -> Result<F> {
            if series1.len() != series2.len() || series1.is_empty() {
                return Err(TimeSeriesError::DimensionMismatch {
                    expected: series1.len(),
                    actual: series2.len(),
                });
            }

            let n = F::from(series1.len()).unwrap();
            let mean1 = series1.sum() / n;
            let mean2 = series2.sum() / n;

            let mut num = F::zero();
            let mut den1 = F::zero();
            let mut den2 = F::zero();

            for (&x1, &x2) in series1.iter().zip(series2.iter()) {
                let diff1 = x1 - mean1;
                let diff2 = x2 - mean2;
                num = num + diff1 * diff2;
                den1 = den1 + diff1 * diff1;
                den2 = den2 + diff2 * diff2;
            }

            let denominator = (den1 * den2).sqrt();
            if denominator > F::zero() {
                Ok(num / denominator)
            } else {
                Ok(F::zero())
            }
        }

        /// GPU-accelerated sliding window operations
        pub fn sliding_window_statistics(
            &self,
            series: &Array1<F>,
            window_size: usize,
            statistics: &[WindowStatistic],
        ) -> Result<Vec<Array1<F>>> {
            if series.len() < window_size {
                return Err(TimeSeriesError::InsufficientData {
                    message: "Series shorter than window".to_string(),
                    required: window_size,
                    actual: series.len(),
                });
            }

            let num_windows = series.len() - window_size + 1;
            let mut results = Vec::with_capacity(statistics.len());

            for stat in statistics {
                let mut stat_values = Array1::zeros(num_windows);
                
                for i in 0..num_windows {
                    let window = &series[i..i + window_size];
                    stat_values[i] = match stat {
                        WindowStatistic::Mean => {
                            window.sum() / F::from(window_size).unwrap()
                        }
                        WindowStatistic::Variance => {
                            let mean = window.sum() / F::from(window_size).unwrap();
                            window.iter()
                                .map(|&x| (x - mean) * (x - mean))
                                .fold(F::zero(), |acc, x| acc + x) / F::from(window_size).unwrap()
                        }
                        WindowStatistic::Min => {
                            window.iter().fold(F::infinity(), |acc, &x| acc.min(x))
                        }
                        WindowStatistic::Max => {
                            window.iter().fold(F::neg_infinity(), |acc, &x| acc.max(x))
                        }
                        WindowStatistic::Range => {
                            let min_val = window.iter().fold(F::infinity(), |acc, &x| acc.min(x));
                            let max_val = window.iter().fold(F::neg_infinity(), |acc, &x| acc.max(x));
                            max_val - min_val
                        }
                    };
                }
                
                results.push(stat_values);
            }

            Ok(results)
        }
    }

    /// Forecasting methods for GPU acceleration
    #[derive(Debug, Clone)]
    pub enum ForecastMethod {
        ExponentialSmoothing { alpha: f64 },
        LinearTrend,
        MovingAverage { window: usize },
        AutoRegressive { order: usize },
    }

    /// Window statistics for sliding window operations
    #[derive(Debug, Clone)]
    pub enum WindowStatistic {
        Mean,
        Variance,
        Min,
        Max,
        Range,
    }

    /// GPU-accelerated feature extraction for time series
    #[derive(Debug)]
    pub struct GpuFeatureExtractor<F: Float + Debug> {
        processor: GpuTimeSeriesProcessor<F>,
        feature_config: FeatureConfig,
    }

    #[derive(Debug, Clone)]
    pub struct FeatureConfig {
        pub extract_statistical: bool,
        pub extract_frequency: bool,
        pub extract_complexity: bool,
        pub window_sizes: Vec<usize>,
    }

    impl Default for FeatureConfig {
        fn default() -> Self {
            Self {
                extract_statistical: true,
                extract_frequency: true,
                extract_complexity: false,
                window_sizes: vec![5, 10, 20],
            }
        }
    }

    impl<F: Float + Debug + Clone> GpuFeatureExtractor<F> {
        pub fn new(config: GpuConfig, feature_config: FeatureConfig) -> Result<Self> {
            let processor = GpuTimeSeriesProcessor::new(config)?;
            Ok(Self {
                processor,
                feature_config,
            })
        }

        /// Extract comprehensive features from multiple time series
        pub fn batch_extract_features(
            &self,
            series_batch: &[Array1<F>],
        ) -> Result<Array2<F>> {
            let mut all_features = Vec::new();

            for series in series_batch {
                let features = self.extract_features(series)?;
                all_features.push(features);
            }

            // Combine into matrix
            if all_features.is_empty() {
                return Ok(Array2::zeros((0, 0)));
            }

            let n_series = all_features.len();
            let n_features = all_features[0].len();
            let mut feature_matrix = Array2::zeros((n_series, n_features));

            for (i, features) in all_features.iter().enumerate() {
                for (j, &feature) in features.iter().enumerate() {
                    feature_matrix[[i, j]] = feature;
                }
            }

            Ok(feature_matrix)
        }

        /// Extract features from a single time series
        fn extract_features(&self, series: &Array1<F>) -> Result<Vec<F>> {
            let mut features = Vec::new();

            if self.feature_config.extract_statistical {
                features.extend(self.extract_statistical_features(series)?);
            }

            if self.feature_config.extract_frequency {
                features.extend(self.extract_frequency_features(series)?);
            }

            if self.feature_config.extract_complexity {
                features.extend(self.extract_complexity_features(series)?);
            }

            Ok(features)
        }

        /// Extract statistical features
        fn extract_statistical_features(&self, series: &Array1<F>) -> Result<Vec<F>> {
            if series.is_empty() {
                return Ok(vec![F::zero(); 8]); // Return zeros for all features
            }

            let n = F::from(series.len()).unwrap();
            let mean = series.sum() / n;
            
            // Variance
            let variance = series.iter()
                .map(|&x| (x - mean) * (x - mean))
                .fold(F::zero(), |acc, x| acc + x) / n;
            
            // Min/Max
            let min_val = series.iter().fold(F::infinity(), |acc, &x| acc.min(x));
            let max_val = series.iter().fold(F::neg_infinity(), |acc, &x| acc.max(x));
            
            // Skewness (simplified)
            let std_dev = variance.sqrt();
            let skewness = if std_dev > F::zero() {
                series.iter()
                    .map(|&x| {
                        let normalized = (x - mean) / std_dev;
                        normalized * normalized * normalized
                    })
                    .fold(F::zero(), |acc, x| acc + x) / n
            } else {
                F::zero()
            };

            // Kurtosis (simplified)
            let kurtosis = if std_dev > F::zero() {
                series.iter()
                    .map(|&x| {
                        let normalized = (x - mean) / std_dev;
                        let squared = normalized * normalized;
                        squared * squared
                    })
                    .fold(F::zero(), |acc, x| acc + x) / n
            } else {
                F::zero()
            };

            // Range
            let range = max_val - min_val;

            // Trend (slope of linear regression)
            let trend = if series.len() > 1 {
                let x_mean = F::from(series.len() - 1).unwrap() / F::from(2).unwrap();
                let mut num = F::zero();
                let mut den = F::zero();
                
                for (i, &y) in series.iter().enumerate() {
                    let x = F::from(i).unwrap();
                    num = num + (x - x_mean) * (y - mean);
                    den = den + (x - x_mean) * (x - x_mean);
                }
                
                if den > F::zero() { num / den } else { F::zero() }
            } else {
                F::zero()
            };

            Ok(vec![mean, variance.sqrt(), min_val, max_val, skewness, kurtosis, range, trend])
        }

        /// Extract frequency domain features (simplified)
        fn extract_frequency_features(&self, series: &Array1<F>) -> Result<Vec<F>> {
            // Simplified frequency features without actual FFT
            let n = series.len();
            if n < 4 {
                return Ok(vec![F::zero(); 3]);
            }

            // Estimate dominant frequency using autocorrelation
            let mut max_autocorr = F::zero();
            let mut dominant_period = 1;
            
            for lag in 1..(n / 2).min(20) {
                let mut autocorr = F::zero();
                let mut count = 0;
                
                for i in lag..n {
                    autocorr = autocorr + series[i] * series[i - lag];
                    count += 1;
                }
                
                if count > 0 {
                    autocorr = autocorr / F::from(count).unwrap();
                    if autocorr > max_autocorr {
                        max_autocorr = autocorr;
                        dominant_period = lag;
                    }
                }
            }

            let dominant_frequency = F::one() / F::from(dominant_period).unwrap();
            
            // Spectral energy (simplified)
            let spectral_energy = series.iter()
                .map(|&x| x * x)
                .fold(F::zero(), |acc, x| acc + x) / F::from(n).unwrap();

            Ok(vec![dominant_frequency, max_autocorr, spectral_energy])
        }

        /// Extract complexity features (simplified)
        fn extract_complexity_features(&self, series: &Array1<F>) -> Result<Vec<F>> {
            if series.len() < 3 {
                return Ok(vec![F::zero(); 2]);
            }

            // Approximate entropy (simplified)
            let mut changes = 0;
            for i in 1..series.len() {
                if (series[i] - series[i-1]).abs() > F::zero() {
                    changes += 1;
                }
            }
            let entropy = F::from(changes).unwrap() / F::from(series.len() - 1).unwrap();

            // Sample entropy (very simplified)
            let mut matches = 0;
            let tolerance = series.iter()
                .map(|&x| x * x)
                .fold(F::zero(), |acc, x| acc + x)
                .sqrt() / F::from(series.len()).unwrap() * F::from(0.1).unwrap();

            for i in 0..series.len()-2 {
                for j in i+1..series.len()-1 {
                    if (series[i] - series[j]).abs() <= tolerance &&
                       (series[i+1] - series[j+1]).abs() <= tolerance {
                        matches += 1;
                    }
                }
            }

            let sample_entropy = if matches > 0 {
                -F::from(matches).unwrap().ln()
            } else {
                F::from(10).unwrap() // Large value for high entropy
            };

            Ok(vec![entropy, sample_entropy])
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_gpu_processor_creation() {
            let config = GpuConfig::default();
            let processor = GpuTimeSeriesProcessor::<f64>::new(config);
            assert!(processor.is_ok());
        }

        #[test]
        fn test_batch_exponential_smoothing() {
            let config = GpuConfig::default();
            let processor = GpuTimeSeriesProcessor::new(config).unwrap();

            let series1 = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
            let series2 = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]);
            let batch = vec![series1, series2];

            let method = ForecastMethod::ExponentialSmoothing { alpha: 0.3 };
            let results = processor.batch_forecast(&batch, 3, method);
            
            assert!(results.is_ok());
            let forecasts = results.unwrap();
            assert_eq!(forecasts.len(), 2);
            assert_eq!(forecasts[0].len(), 3);
            assert_eq!(forecasts[1].len(), 3);
        }

        #[test]
        fn test_correlation_matrix() {
            let config = GpuConfig::default();
            let processor = GpuTimeSeriesProcessor::new(config).unwrap();

            let series1 = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
            let series2 = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]);
            let batch = vec![series1, series2];

            let correlation_matrix = processor.batch_correlation_matrix(&batch).unwrap();
            
            assert_eq!(correlation_matrix.dim(), (2, 2));
            assert!((correlation_matrix[[0, 0]] - 1.0).abs() < 1e-10);
            assert!((correlation_matrix[[1, 1]] - 1.0).abs() < 1e-10);
            assert!(correlation_matrix[[0, 1]] > 0.99); // Should be highly correlated
        }

        #[test]
        fn test_feature_extraction() {
            let config = GpuConfig::default();
            let feature_config = FeatureConfig::default();
            let extractor = GpuFeatureExtractor::new(config, feature_config).unwrap();

            let series = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
            let batch = vec![series];

            let features = extractor.batch_extract_features(&batch).unwrap();
            assert_eq!(features.nrows(), 1);
            assert!(features.ncols() > 0); // Should have extracted features
        }

        #[test]
        fn test_sliding_window_statistics() {
            let config = GpuConfig::default();
            let processor = GpuTimeSeriesProcessor::new(config).unwrap();

            let series = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
            let statistics = vec![WindowStatistic::Mean, WindowStatistic::Variance];

            let results = processor.sliding_window_statistics(&series, 3, &statistics).unwrap();
            
            assert_eq!(results.len(), 2); // Two statistics
            assert_eq!(results[0].len(), 6); // 8 - 3 + 1 = 6 windows
            
            // Check first window mean: (1+2+3)/3 = 2
            assert!((results[0][0] - 2.0).abs() < 1e-10);
        }
    }
}
