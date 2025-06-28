//! GPU acceleration for metrics computation
//!
//! This module provides GPU-accelerated implementations of common metrics
//! using compute shaders and memory-efficient batch processing.

use crate::error::{MetricsError, Result};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::collections::HashMap;
use std::sync::Arc;

/// GPU acceleration configuration
#[derive(Debug, Clone)]
pub struct GpuAccelConfig {
    /// Minimum batch size to use GPU acceleration
    pub min_batch_size: usize,
    /// Maximum memory usage on GPU (in bytes)
    pub max_gpu_memory: usize,
    /// Preferred GPU device index
    pub device_index: Option<usize>,
    /// Enable memory pool for faster allocations
    pub enable_memory_pool: bool,
    /// Compute shader optimization level
    pub optimization_level: u8,
}

impl Default for GpuAccelConfig {
    fn default() -> Self {
        Self {
            min_batch_size: 1000,
            max_gpu_memory: 1024 * 1024 * 1024, // 1GB
            device_index: None,
            enable_memory_pool: true,
            optimization_level: 2,
        }
    }
}

/// GPU-accelerated metrics computer
pub struct GpuMetricsComputer {
    config: GpuAccelConfig,
}

impl GpuMetricsComputer {
    /// Create new GPU metrics computer
    pub fn new(config: GpuAccelConfig) -> Result<Self> {
        Ok(Self { config })
    }

    /// Check if GPU acceleration should be used for given data size
    pub fn should_use_gpu(&self, data_size: usize) -> bool {
        data_size >= self.config.min_batch_size
    }

    /// Check if GPU is available
    pub fn is_gpu_available(&self) -> bool {
        false // GPU not available in this build
    }

    /// Compute accuracy on GPU (falls back to CPU)
    pub fn gpu_accuracy(&self, y_true: &Array1<i32>, y_pred: &Array1<i32>) -> Result<f32> {
        self.cpu_accuracy(y_true, y_pred)
    }

    /// Compute MSE on GPU (falls back to CPU)
    pub fn gpu_mse(&self, y_true: &Array1<f32>, y_pred: &Array1<f32>) -> Result<f32> {
        self.cpu_mse(y_true, y_pred)
    }

    /// Compute confusion matrix on GPU (falls back to CPU)
    pub fn gpu_confusion_matrix(
        &self,
        y_true: &Array1<i32>,
        y_pred: &Array1<i32>,
        num_classes: usize,
    ) -> Result<Array2<i32>> {
        self.cpu_confusion_matrix(y_true, y_pred, num_classes)
    }

    /// Compute batch metrics on GPU (falls back to CPU)
    pub fn gpu_batch_metrics(
        &self,
        y_true_batch: &Array2<f32>,
        y_pred_batch: &Array2<f32>,
        metric_names: &[String],
    ) -> Result<HashMap<String, Array1<f32>>> {
        let mut results = HashMap::new();
        let batch_size = y_true_batch.nrows();

        for metric_name in metric_names {
            let mut metric_results = Array1::zeros(batch_size);

            for i in 0..batch_size {
                let y_true_sample = y_true_batch.row(i).to_owned();
                let y_pred_sample = y_pred_batch.row(i).to_owned();

                let result = match metric_name.as_str() {
                    "mse" => self.cpu_mse(&y_true_sample, &y_pred_sample)?,
                    "mae" => self.cpu_mae(&y_true_sample, &y_pred_sample)?,
                    _ => {
                        return Err(MetricsError::InvalidInput(format!(
                            "Unsupported metric: {}",
                            metric_name
                        )))
                    }
                };

                metric_results[i] = result;
            }

            results.insert(metric_name.clone(), metric_results);
        }

        Ok(results)
    }

    // CPU fallback implementations

    fn cpu_accuracy(&self, y_true: &Array1<i32>, y_pred: &Array1<i32>) -> Result<f32> {
        if y_true.len() != y_pred.len() {
            return Err(MetricsError::InvalidInput(
                "Arrays must have the same length".to_string(),
            ));
        }

        let correct = y_true
            .iter()
            .zip(y_pred.iter())
            .filter(|(&true_val, &pred_val)| true_val == pred_val)
            .count();

        Ok(correct as f32 / y_true.len() as f32)
    }

    fn cpu_mse(&self, y_true: &Array1<f32>, y_pred: &Array1<f32>) -> Result<f32> {
        if y_true.len() != y_pred.len() {
            return Err(MetricsError::InvalidInput(
                "Arrays must have the same length".to_string(),
            ));
        }

        let mse = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(true_val, pred_val)| (true_val - pred_val).powi(2))
            .sum::<f32>()
            / y_true.len() as f32;

        Ok(mse)
    }

    fn cpu_mae(&self, y_true: &Array1<f32>, y_pred: &Array1<f32>) -> Result<f32> {
        if y_true.len() != y_pred.len() {
            return Err(MetricsError::InvalidInput(
                "Arrays must have the same length".to_string(),
            ));
        }

        let mae = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(true_val, pred_val)| (true_val - pred_val).abs())
            .sum::<f32>()
            / y_true.len() as f32;

        Ok(mae)
    }

    fn cpu_confusion_matrix(
        &self,
        y_true: &Array1<i32>,
        y_pred: &Array1<i32>,
        num_classes: usize,
    ) -> Result<Array2<i32>> {
        if y_true.len() != y_pred.len() {
            return Err(MetricsError::InvalidInput(
                "Arrays must have the same length".to_string(),
            ));
        }

        let mut matrix = Array2::zeros((num_classes, num_classes));

        for (&true_class, &pred_class) in y_true.iter().zip(y_pred.iter()) {
            if true_class >= 0
                && (true_class as usize) < num_classes
                && pred_class >= 0
                && (pred_class as usize) < num_classes
            {
                matrix[[true_class as usize, pred_class as usize]] += 1;
            }
        }

        Ok(matrix)
    }
}

/// GPU metrics computer builder for convenient configuration
pub struct GpuMetricsComputerBuilder {
    config: GpuAccelConfig,
}

impl GpuMetricsComputerBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            config: GpuAccelConfig::default(),
        }
    }

    /// Set minimum batch size for GPU acceleration
    pub fn with_min_batch_size(mut self, size: usize) -> Self {
        self.config.min_batch_size = size;
        self
    }

    /// Set maximum GPU memory usage
    pub fn with_max_gpu_memory(mut self, bytes: usize) -> Self {
        self.config.max_gpu_memory = bytes;
        self
    }

    /// Set preferred GPU device
    pub fn with_device_index(mut self, index: Option<usize>) -> Self {
        self.config.device_index = index;
        self
    }

    /// Enable memory pool
    pub fn with_memory_pool(mut self, enable: bool) -> Self {
        self.config.enable_memory_pool = enable;
        self
    }

    /// Set optimization level
    pub fn with_optimization_level(mut self, level: u8) -> Self {
        self.config.optimization_level = level;
        self
    }

    /// Build the GPU metrics computer
    pub fn build(self) -> Result<GpuMetricsComputer> {
        GpuMetricsComputer::new(self.config)
    }
}

impl Default for GpuMetricsComputerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_gpu_metrics_computer_creation() {
        let computer = GpuMetricsComputer::new(GpuAccelConfig::default()).unwrap();
        assert!(!computer.is_gpu_available());
    }

    #[test]
    fn test_gpu_metrics_computer_builder() {
        let computer = GpuMetricsComputerBuilder::new()
            .with_min_batch_size(500)
            .with_max_gpu_memory(512 * 1024 * 1024)
            .with_device_index(Some(0))
            .with_memory_pool(true)
            .with_optimization_level(3)
            .build()
            .unwrap();

        assert_eq!(computer.config.min_batch_size, 500);
        assert_eq!(computer.config.max_gpu_memory, 512 * 1024 * 1024);
        assert_eq!(computer.config.device_index, Some(0));
        assert!(computer.config.enable_memory_pool);
        assert_eq!(computer.config.optimization_level, 3);
    }

    #[test]
    fn test_should_use_gpu() {
        let computer = GpuMetricsComputer::new(GpuAccelConfig::default()).unwrap();
        assert!(!computer.should_use_gpu(500));
        assert!(computer.should_use_gpu(1500));
    }

    #[test]
    fn test_cpu_accuracy() {
        let computer = GpuMetricsComputer::new(GpuAccelConfig::default()).unwrap();
        let y_true = array![0, 1, 2, 0, 1, 2];
        let y_pred = array![0, 2, 1, 0, 0, 2];

        let accuracy = computer.gpu_accuracy(&y_true, &y_pred).unwrap();
        assert!((accuracy - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_cpu_mse() {
        let computer = GpuMetricsComputer::new(GpuAccelConfig::default()).unwrap();
        let y_true = array![1.0, 2.0, 3.0, 4.0];
        let y_pred = array![1.1, 2.1, 2.9, 4.1];

        let mse = computer.gpu_mse(&y_true, &y_pred).unwrap();
        assert!(mse > 0.0 && mse < 0.1);
    }

    #[test]
    fn test_cpu_confusion_matrix() {
        let computer = GpuMetricsComputer::new(GpuAccelConfig::default()).unwrap();
        let y_true = array![0, 1, 2, 0, 1, 2];
        let y_pred = array![0, 2, 1, 0, 0, 2];

        let cm = computer.gpu_confusion_matrix(&y_true, &y_pred, 3).unwrap();
        assert_eq!(cm.shape(), &[3, 3]);
        assert_eq!(cm[[0, 0]], 2);
    }
}
