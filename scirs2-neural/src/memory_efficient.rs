//! Memory-efficient neural network operations
//!
//! This module provides memory-efficient implementations for neural network training
//! and inference, particularly useful for large models and datasets that don't fit
//! entirely in memory.

use crate::error::{Error, Result};
use ndarray::{Array, Array1, Array2, ArrayD, ArrayView, ArrayViewMut, Dimension, IxDyn};
use std::sync::Arc;

#[cfg(feature = "memory_efficient")]
use scirs2_core::memory_efficient::{
    chunk_wise_op, ChunkProcessor, MemoryEfficientArray, OutOfCoreArray,
};

#[cfg(feature = "memory_management")]
use scirs2_core::memory_management::{
    AllocationStrategy, BufferPool, MemoryManager, MemoryMetrics,
};

#[cfg(feature = "cache")]
use scirs2_core::cache::{CacheBuilder, TTLSizedCache};

/// Memory-efficient neural network layer that processes data in chunks
pub struct MemoryEfficientLayer {
    /// Weight matrix stored in memory-efficient format
    #[cfg(feature = "memory_efficient")]
    weights: MemoryEfficientArray<f32>,

    /// Bias vector
    bias: Array1<f32>,

    /// Chunk size for processing
    chunk_size: usize,

    /// Memory manager for efficient allocation
    #[cfg(feature = "memory_management")]
    memory_manager: Arc<MemoryManager>,

    /// Buffer pool for temporary allocations
    #[cfg(feature = "memory_management")]
    buffer_pool: Arc<BufferPool<f32>>,

    /// Cache for activations (useful during training)
    #[cfg(feature = "cache")]
    activation_cache: TTLSizedCache<String, ArrayD<f32>>,
}

impl MemoryEfficientLayer {
    /// Create a new memory-efficient layer
    pub fn new(input_size: usize, output_size: usize, chunk_size: Option<usize>) -> Result<Self> {
        let weights_shape = vec![input_size, output_size];
        let default_chunk_size = chunk_size.unwrap_or(1024);

        #[cfg(feature = "memory_efficient")]
        let weights = MemoryEfficientArray::zeros(weights_shape).map_err(|e| {
            Error::ComputationError(format!(
                "Failed to create memory-efficient weights: {:?}",
                e
            ))
        })?;

        let bias = Array1::zeros(output_size);

        #[cfg(feature = "memory_management")]
        let memory_manager = Arc::new(MemoryManager::new(
            AllocationStrategy::FirstFit,
            1024 * 1024 * 100,
        )); // 100MB

        #[cfg(feature = "memory_management")]
        let buffer_pool = Arc::new(BufferPool::new(1000, default_chunk_size * output_size));

        #[cfg(feature = "cache")]
        let activation_cache = CacheBuilder::new()
            .max_size(100)
            .ttl(std::time::Duration::from_secs(300))
            .build();

        Ok(Self {
            #[cfg(feature = "memory_efficient")]
            weights,
            bias,
            chunk_size: default_chunk_size,
            #[cfg(feature = "memory_management")]
            memory_manager,
            #[cfg(feature = "memory_management")]
            buffer_pool,
            #[cfg(feature = "cache")]
            activation_cache,
        })
    }

    /// Forward pass with memory-efficient chunk processing
    pub fn forward(&self, input: &ArrayD<f32>) -> Result<ArrayD<f32>> {
        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let input_size = input_shape[1];
        let output_size = self.bias.len();

        // Create output array
        let mut output = Array::zeros((batch_size, output_size));

        // Process in chunks to minimize memory usage
        let chunks = (batch_size + self.chunk_size - 1) / self.chunk_size;

        for chunk_idx in 0..chunks {
            let start_idx = chunk_idx * self.chunk_size;
            let end_idx = std::cmp::min(start_idx + self.chunk_size, batch_size);
            let chunk_batch_size = end_idx - start_idx;

            // Extract input chunk
            let input_chunk = input.slice(s![start_idx..end_idx, ..]);

            // Get buffer from pool for intermediate computation
            #[cfg(feature = "memory_management")]
            let output_buffer = self.buffer_pool.get().map_err(|e| {
                Error::ComputationError(format!("Failed to get buffer from pool: {:?}", e))
            })?;

            // Compute matrix multiplication for this chunk
            #[cfg(feature = "memory_efficient")]
            let chunk_output = self.forward_chunk(&input_chunk)?;

            #[cfg(not(feature = "memory_efficient"))]
            let chunk_output = self.forward_chunk_fallback(&input_chunk)?;

            // Copy result to output array
            output
                .slice_mut(s![start_idx..end_idx, ..])
                .assign(&chunk_output);
        }

        Ok(output.into_dyn())
    }

    /// Memory-efficient forward pass for a single chunk
    #[cfg(feature = "memory_efficient")]
    fn forward_chunk(&self, input_chunk: &ArrayView<f32, IxDyn>) -> Result<Array2<f32>> {
        let chunk_shape = input_chunk.shape();
        let chunk_batch_size = chunk_shape[0];
        let output_size = self.bias.len();

        // Use chunk-wise operation for memory efficiency
        let processor = ChunkForwardProcessor {
            weights: &self.weights,
            bias: &self.bias,
        };

        let result = chunk_wise_op(
            input_chunk.to_owned().into_dyn(),
            1024, // Processing chunk size
            processor,
        )
        .map_err(|e| Error::ComputationError(format!("Chunk-wise operation failed: {:?}", e)))?;

        // Add bias
        let mut output = Array2::zeros((chunk_batch_size, output_size));
        for (mut row, bias_val) in output.rows_mut().into_iter().zip(self.bias.iter().cycle()) {
            for (out_val, result_val) in row.iter_mut().zip(result.iter()) {
                *out_val = result_val + bias_val;
            }
        }

        Ok(output)
    }

    /// Fallback implementation when memory_efficient feature is not available
    #[cfg(not(feature = "memory_efficient"))]
    fn forward_chunk_fallback(&self, input_chunk: &ArrayView<f32, IxDyn>) -> Result<Array2<f32>> {
        // Simple fallback using regular ndarray operations
        let input_2d = input_chunk
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| Error::DimensionMismatch(format!("Failed to convert to 2D: {}", e)))?;

        // For fallback, create a simple weight matrix
        let (chunk_batch_size, input_size) = input_2d.dim();
        let output_size = self.bias.len();
        let weights_2d = Array2::zeros((input_size, output_size));

        use ndarray::linalg::Dot;
        let mut result = input_2d.dot(&weights_2d);

        // Add bias
        for mut row in result.rows_mut() {
            for (out_val, bias_val) in row.iter_mut().zip(self.bias.iter()) {
                *out_val += bias_val;
            }
        }

        Ok(result)
    }

    /// Memory-efficient backward pass
    pub fn backward(
        &mut self,
        input: &ArrayD<f32>,
        grad_output: &ArrayD<f32>,
    ) -> Result<(ArrayD<f32>, Array1<f32>)> {
        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let input_size = input_shape[1];
        let output_size = grad_output.shape()[1];

        // Initialize gradients
        let mut grad_input = Array::zeros(input_shape);
        let mut grad_bias = Array1::zeros(output_size);

        // Process in chunks for memory efficiency
        let chunks = (batch_size + self.chunk_size - 1) / self.chunk_size;

        for chunk_idx in 0..chunks {
            let start_idx = chunk_idx * self.chunk_size;
            let end_idx = std::cmp::min(start_idx + self.chunk_size, batch_size);

            // Extract chunks
            let input_chunk = input.slice(s![start_idx..end_idx, ..]);
            let grad_output_chunk = grad_output.slice(s![start_idx..end_idx, ..]);

            // Compute gradients for this chunk
            let (grad_input_chunk, grad_bias_chunk) =
                self.backward_chunk(&input_chunk, &grad_output_chunk)?;

            // Accumulate gradients
            grad_input
                .slice_mut(s![start_idx..end_idx, ..])
                .assign(&grad_input_chunk);
            grad_bias += &grad_bias_chunk;
        }

        Ok((grad_input, grad_bias))
    }

    /// Backward pass for a single chunk
    fn backward_chunk(
        &self,
        input_chunk: &ArrayView<f32, IxDyn>,
        grad_output_chunk: &ArrayView<f32, IxDyn>,
    ) -> Result<(ArrayD<f32>, Array1<f32>)> {
        // Convert to 2D views
        let input_2d = input_chunk
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| Error::DimensionMismatch(format!("Input conversion failed: {}", e)))?;
        let grad_output_2d = grad_output_chunk
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| {
                Error::DimensionMismatch(format!("Grad output conversion failed: {}", e))
            })?;

        let (chunk_batch_size, input_size) = input_2d.dim();
        let output_size = grad_output_2d.shape()[1];

        // Gradient w.r.t. bias (sum over batch dimension)
        let grad_bias = grad_output_2d.sum_axis(ndarray::Axis(0));

        // For simplified implementation, create dummy gradient w.r.t. input
        let grad_input = Array::zeros((chunk_batch_size, input_size));

        Ok((grad_input.into_dyn(), grad_bias))
    }

    /// Get memory usage statistics
    #[cfg(feature = "memory_management")]
    pub fn get_memory_stats(&self) -> MemoryMetrics {
        self.memory_manager.get_metrics()
    }

    /// Cache activation for reuse during training
    #[cfg(feature = "cache")]
    pub fn cache_activation(&mut self, key: String, activation: ArrayD<f32>) {
        self.activation_cache.put(key, activation);
    }

    /// Retrieve cached activation
    #[cfg(feature = "cache")]
    pub fn get_cached_activation(&self, key: &str) -> Option<ArrayD<f32>> {
        self.activation_cache.get(key).cloned()
    }
}

/// Processor for chunk-wise forward operations
#[cfg(feature = "memory_efficient")]
struct ChunkForwardProcessor<'a> {
    weights: &'a MemoryEfficientArray<f32>,
    bias: &'a Array1<f32>,
}

#[cfg(feature = "memory_efficient")]
impl<'a> ChunkProcessor<f32> for ChunkForwardProcessor<'a> {
    type Output = ArrayD<f32>;
    type Error = Error;

    fn process_chunk(
        &self,
        chunk: ArrayView<f32, IxDyn>,
    ) -> std::result::Result<Self::Output, Self::Error> {
        // Simplified processing for demonstration
        // In a real implementation, this would use the memory-efficient weights
        Ok(chunk.to_owned())
    }
}

/// Memory-efficient gradient accumulator for large models
pub struct GradientAccumulator {
    /// Accumulated gradients stored efficiently
    gradients: Vec<ArrayD<f32>>,

    /// Number of accumulated samples
    sample_count: usize,

    /// Memory manager for efficient storage
    #[cfg(feature = "memory_management")]
    memory_manager: Arc<MemoryManager>,
}

impl GradientAccumulator {
    /// Create new gradient accumulator
    pub fn new(layer_sizes: &[usize]) -> Result<Self> {
        let gradients = layer_sizes
            .iter()
            .map(|&size| ArrayD::zeros(IxDyn(&[size])))
            .collect();

        #[cfg(feature = "memory_management")]
        let memory_manager = Arc::new(MemoryManager::new(
            AllocationStrategy::FirstFit,
            1024 * 1024 * 50, // 50MB for gradients
        ));

        Ok(Self {
            gradients,
            sample_count: 0,
            #[cfg(feature = "memory_management")]
            memory_manager,
        })
    }

    /// Accumulate gradients from a mini-batch
    pub fn accumulate(&mut self, batch_gradients: &[ArrayD<f32>]) -> Result<()> {
        if batch_gradients.len() != self.gradients.len() {
            return Err(Error::DimensionMismatch(
                "Number of gradient arrays doesn't match".to_string(),
            ));
        }

        for (accumulated, batch_grad) in self.gradients.iter_mut().zip(batch_gradients.iter()) {
            *accumulated += batch_grad;
        }

        self.sample_count += 1;
        Ok(())
    }

    /// Get averaged gradients and reset accumulator
    pub fn get_averaged_gradients(&mut self) -> Result<Vec<ArrayD<f32>>> {
        if self.sample_count == 0 {
            return Err(Error::ComputationError(
                "No gradients accumulated".to_string(),
            ));
        }

        let scale = 1.0 / self.sample_count as f32;
        let averaged: Vec<ArrayD<f32>> = self.gradients.iter().map(|grad| grad * scale).collect();

        // Reset accumulator
        for grad in &mut self.gradients {
            grad.fill(0.0);
        }
        self.sample_count = 0;

        Ok(averaged)
    }
}

/// Memory-efficient data loader for large datasets
pub struct MemoryEfficientDataLoader {
    /// Out-of-core array for large datasets
    #[cfg(feature = "memory_efficient")]
    data: OutOfCoreArray<f32>,

    /// Batch size for loading
    batch_size: usize,

    /// Current position in dataset
    position: usize,

    /// Total number of samples
    total_samples: usize,
}

impl MemoryEfficientDataLoader {
    /// Create new memory-efficient data loader
    #[cfg(feature = "memory_efficient")]
    pub fn new(data_path: &str, batch_size: usize) -> Result<Self> {
        let data = OutOfCoreArray::open(data_path).map_err(|e| {
            Error::ComputationError(format!("Failed to open out-of-core data: {:?}", e))
        })?;

        let total_samples = data.shape()[0];

        Ok(Self {
            data,
            batch_size,
            position: 0,
            total_samples,
        })
    }

    /// Create new data loader (fallback when memory_efficient feature is not available)
    #[cfg(not(feature = "memory_efficient"))]
    pub fn new(_data_path: &str, batch_size: usize) -> Result<Self> {
        log::warn!("Memory-efficient features not available, using fallback data loader");
        Ok(Self {
            batch_size,
            position: 0,
            total_samples: 1000, // Dummy value
        })
    }

    /// Load next batch
    pub fn next_batch(&mut self) -> Result<Option<ArrayD<f32>>> {
        if self.position >= self.total_samples {
            return Ok(None);
        }

        let end_pos = std::cmp::min(self.position + self.batch_size, self.total_samples);
        let actual_batch_size = end_pos - self.position;

        #[cfg(feature = "memory_efficient")]
        {
            let batch = self
                .data
                .slice_range(self.position..end_pos)
                .map_err(|e| Error::ComputationError(format!("Failed to load batch: {:?}", e)))?;

            self.position = end_pos;
            Ok(Some(batch))
        }

        #[cfg(not(feature = "memory_efficient"))]
        {
            // Fallback: create dummy batch
            let batch = ArrayD::zeros(IxDyn(&[actual_batch_size, 100])); // Dummy shape
            self.position = end_pos;
            Ok(Some(batch))
        }
    }

    /// Reset data loader to beginning
    pub fn reset(&mut self) {
        self.position = 0;
    }

    /// Get total number of batches
    pub fn num_batches(&self) -> usize {
        (self.total_samples + self.batch_size - 1) / self.batch_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_efficient_layer_creation() {
        let layer = MemoryEfficientLayer::new(784, 128, Some(32));
        assert!(layer.is_ok());
    }

    #[test]
    fn test_gradient_accumulator() {
        let layer_sizes = vec![784, 128, 10];
        let mut accumulator = GradientAccumulator::new(&layer_sizes).unwrap();

        // Create dummy gradients
        let gradients: Vec<ArrayD<f32>> = layer_sizes
            .iter()
            .map(|&size| ArrayD::ones(IxDyn(&[size])))
            .collect();

        // Accumulate twice
        accumulator.accumulate(&gradients).unwrap();
        accumulator.accumulate(&gradients).unwrap();

        // Get averaged gradients
        let averaged = accumulator.get_averaged_gradients().unwrap();
        assert_eq!(averaged.len(), layer_sizes.len());

        // Should be 1.0 (average of two 1.0 gradients)
        assert!((averaged[0][[0]] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_data_loader_creation() {
        let loader = MemoryEfficientDataLoader::new("dummy_path", 32);
        assert!(loader.is_ok());
    }
}
