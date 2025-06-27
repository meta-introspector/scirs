//! GPU acceleration support for optimizers
//!
//! This module provides GPU-accelerated implementations of optimizers using CUDA kernels.

use ndarray::{Array, ArrayBase, Data, DataMut, Dimension};
use num_traits::Float;
use std::sync::Arc;

#[cfg(feature = "gpu")]
use scirs2_core::gpu::{GpuBackend, GpuBuffer, GpuContext, GpuError};

pub mod adam_gpu;
pub mod adamw_gpu;
pub mod adagrad_gpu;
pub mod lamb_gpu;
pub mod memory_pool;
pub mod mixed_precision;
pub mod multi_gpu;
pub mod rmsprop_gpu;
pub mod rocm_backend;
pub mod sgd_gpu;

/// Trait for GPU-accelerated optimizers
pub trait GpuOptimizer<A: Float, D: Dimension> {
    /// Check if GPU acceleration is available
    fn is_gpu_available(&self) -> bool;

    /// Move optimizer state to GPU
    fn to_gpu(&mut self) -> Result<(), GpuOptimizerError>;

    /// Move optimizer state back to CPU
    fn to_cpu(&mut self) -> Result<(), GpuOptimizerError>;

    /// Perform optimization step on GPU
    fn step_gpu(
        &mut self,
        params: &mut Array<A, D>,
        gradients: &Array<A, D>,
    ) -> Result<(), GpuOptimizerError>;
}

/// Error type for GPU optimizer operations
#[derive(Debug, thiserror::Error)]
pub enum GpuOptimizerError {
    /// GPU backend error
    #[error("GPU error: {0}")]
    GpuError(#[from] GpuError),

    /// Unsupported operation
    #[error("Operation not supported: {0}")]
    UnsupportedOperation(String),

    /// Invalid state
    #[error("Invalid optimizer state: {0}")]
    InvalidState(String),

    /// Dimension mismatch
    #[error("Dimension mismatch: expected {expected:?}, got {actual:?}")]
    DimensionMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    /// Not initialized
    #[error("GPU optimizer not initialized")]
    NotInitialized,
}

/// GPU optimizer configuration
#[derive(Debug, Clone)]
pub struct GpuOptimizerConfig {
    /// GPU backend to use
    pub backend: GpuBackend,

    /// Enable mixed precision training
    pub mixed_precision: bool,

    /// Loss scaling factor for mixed precision
    pub loss_scale: f32,

    /// Dynamic loss scaling
    pub dynamic_loss_scaling: bool,

    /// Memory pool size in bytes
    pub memory_pool_size: usize,

    /// Number of GPUs for multi-GPU training
    pub num_gpus: usize,

    /// Enable gradient clipping
    pub gradient_clipping: bool,

    /// Maximum gradient norm for clipping
    pub max_grad_norm: f32,
}

impl Default for GpuOptimizerConfig {
    fn default() -> Self {
        Self {
            backend: GpuBackend::default(),
            mixed_precision: false,
            loss_scale: 1024.0,
            dynamic_loss_scaling: true,
            memory_pool_size: 1024 * 1024 * 1024, // 1GB
            num_gpus: 1,
            gradient_clipping: false,
            max_grad_norm: 1.0,
        }
    }
}

/// GPU memory manager for optimizer states
pub struct GpuOptimizerMemory<A: Float> {
    /// GPU context
    context: Arc<GpuContext>,

    /// Parameter buffer on GPU
    params_gpu: Option<GpuBuffer<A>>,

    /// Gradient buffer on GPU
    grads_gpu: Option<GpuBuffer<A>>,

    /// First moment buffer (for Adam-like optimizers)
    m_gpu: Option<GpuBuffer<A>>,

    /// Second moment buffer (for Adam-like optimizers)
    v_gpu: Option<GpuBuffer<A>>,

    /// Size of buffers
    size: usize,

    /// Configuration
    config: GpuOptimizerConfig,
}

impl<A: Float> GpuOptimizerMemory<A> {
    /// Create new GPU memory manager
    pub fn new(size: usize, config: GpuOptimizerConfig) -> Result<Self, GpuOptimizerError> {
        let context = Arc::new(GpuContext::new(config.backend)?);
        
        Ok(Self {
            context,
            params_gpu: None,
            grads_gpu: None,
            m_gpu: None,
            v_gpu: None,
            size,
            config,
        })
    }

    /// Allocate GPU buffers
    pub fn allocate(&mut self) -> Result<(), GpuOptimizerError> {
        self.params_gpu = Some(self.context.create_buffer::<A>(self.size));
        self.grads_gpu = Some(self.context.create_buffer::<A>(self.size));
        self.m_gpu = Some(self.context.create_buffer::<A>(self.size));
        self.v_gpu = Some(self.context.create_buffer::<A>(self.size));
        Ok(())
    }

    /// Copy parameters to GPU
    pub fn copy_params_to_gpu<S, D>(&mut self, params: &ArrayBase<S, D>) -> Result<(), GpuOptimizerError>
    where
        S: Data<Elem = A>,
        D: Dimension,
    {
        if let Some(ref params_gpu) = self.params_gpu {
            let params_slice = params.as_slice()
                .ok_or_else(|| GpuOptimizerError::InvalidState("Parameters must be contiguous".to_string()))?;
            params_gpu.copy_from_host(params_slice);
            Ok(())
        } else {
            Err(GpuOptimizerError::NotInitialized)
        }
    }

    /// Copy parameters from GPU
    pub fn copy_params_from_gpu<S, D>(&self, params: &mut ArrayBase<S, D>) -> Result<(), GpuOptimizerError>
    where
        S: DataMut<Elem = A>,
        D: Dimension,
    {
        if let Some(ref params_gpu) = self.params_gpu {
            let params_slice = params.as_slice_mut()
                .ok_or_else(|| GpuOptimizerError::InvalidState("Parameters must be contiguous".to_string()))?;
            params_gpu.copy_to_host(params_slice);
            Ok(())
        } else {
            Err(GpuOptimizerError::NotInitialized)
        }
    }

    /// Get GPU context
    pub fn context(&self) -> &Arc<GpuContext> {
        &self.context
    }

    /// Get configuration
    pub fn config(&self) -> &GpuOptimizerConfig {
        &self.config
    }
}

/// Helper functions for GPU operations
pub mod utils {
    use super::*;

    /// Check if GPU acceleration is available for the given backend
    pub fn is_gpu_available(backend: GpuBackend) -> bool {
        match GpuContext::new(backend) {
            Ok(_) => true,
            Err(_) => false,
        }
    }

    /// Get the optimal GPU backend for the current system
    pub fn get_optimal_backend() -> GpuBackend {
        // Try backends in order of preference
        let backends = [
            GpuBackend::Cuda,
            GpuBackend::Metal,
            GpuBackend::Rocm,
            GpuBackend::Wgpu,
        ];

        for backend in &backends {
            if is_gpu_available(*backend) {
                return *backend;
            }
        }

        GpuBackend::Cpu
    }

    /// Calculate optimal block size for GPU kernels
    pub fn calculate_block_size(n: usize, max_threads: usize) -> (usize, usize) {
        let block_size = 256.min(max_threads);
        let grid_size = (n + block_size - 1) / block_size;
        (grid_size, block_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_gpu_optimizer_config_default() {
        let config = GpuOptimizerConfig::default();
        assert!(!config.mixed_precision);
        assert_eq!(config.loss_scale, 1024.0);
        assert!(config.dynamic_loss_scaling);
        assert_eq!(config.num_gpus, 1);
    }

    #[test]
    fn test_gpu_memory_creation() {
        let config = GpuOptimizerConfig {
            backend: GpuBackend::Cpu, // Use CPU backend for testing
            ..Default::default()
        };

        let memory = GpuOptimizerMemory::<f32>::new(1000, config);
        assert!(memory.is_ok());
    }

    #[test]
    fn test_optimal_backend_selection() {
        let backend = utils::get_optimal_backend();
        // Should at least return CPU backend
        assert!(matches!(
            backend,
            GpuBackend::Cuda | GpuBackend::Metal | GpuBackend::Rocm | GpuBackend::Wgpu | GpuBackend::Cpu
        ));
    }

    #[test]
    fn test_block_size_calculation() {
        let (grid, block) = utils::calculate_block_size(1000, 1024);
        assert_eq!(block, 256);
        assert_eq!(grid, 4); // (1000 + 255) / 256 = 4

        let (grid, block) = utils::calculate_block_size(100, 128);
        assert_eq!(block, 128);
        assert_eq!(grid, 1); // (100 + 127) / 128 = 1
    }
}