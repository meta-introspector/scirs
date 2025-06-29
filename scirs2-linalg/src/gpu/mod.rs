//! GPU acceleration foundations for linear algebra operations
//!
//! This module provides the foundation for GPU-accelerated linear algebra operations
//! including CUDA, OpenCL, and ROCm support. It defines traits and abstractions
//! for different GPU backends while maintaining a unified interface.

use crate::error::{LinalgError, LinalgResult};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_traits::{Float, NumAssign, Zero};
use std::fmt::Debug;

pub mod backends;
pub mod device_info;
pub mod memory;
pub mod operations;

/// GPU device types supported by the library
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuDeviceType {
    /// NVIDIA GPU with CUDA support
    Cuda,
    /// OpenCL-compatible GPU
    OpenCl,
    /// AMD GPU with ROCm support
    Rocm,
    /// Vulkan compute support
    Vulkan,
    /// Apple Metal GPU
    Metal,
}

/// GPU device information
#[derive(Debug, Clone)]
pub struct GpuDeviceInfo {
    /// Device type
    pub device_type: GpuDeviceType,
    /// Device name/description
    pub name: String,
    /// Total global memory in bytes
    pub total_memory: usize,
    /// Number of compute units/cores
    pub compute_units: u32,
    /// Clock frequency in MHz
    pub clock_frequency: u32,
    /// Whether the device supports double precision
    pub supports_fp64: bool,
    /// Whether the device supports half precision
    pub supports_fp16: bool,
    /// Maximum work group size
    pub max_work_group_size: usize,
    /// Device vendor
    pub vendor: String,
}

/// GPU memory buffer abstraction
pub trait GpuBuffer<T>: Send + Sync {
    /// Get the size of the buffer in elements
    fn len(&self) -> usize;

    /// Check if the buffer is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Copy data from host to device
    fn copy_from_host(&mut self, data: &[T]) -> LinalgResult<()>;

    /// Copy data from device to host
    fn copy_to_host(&self, data: &mut [T]) -> LinalgResult<()>;

    /// Get device pointer (platform-specific)
    fn device_ptr(&self) -> *mut std::ffi::c_void;
}

/// GPU context abstraction for managing device state (dyn compatible)
pub trait GpuContext: Send + Sync {
    /// Get device information
    fn device_info(&self) -> &GpuDeviceInfo;

    /// Synchronize all operations
    fn synchronize(&self) -> LinalgResult<()>;

    /// Get available memory in bytes
    fn available_memory(&self) -> LinalgResult<usize>;

    /// Get total memory in bytes
    fn total_memory(&self) -> usize {
        self.device_info().total_memory
    }
}

/// GPU context with generic operations (separate trait for dyn compatibility)
pub trait GpuContextAlloc: GpuContext {
    /// Allocate buffer on GPU
    fn allocate_buffer<T: Clone + Send + Sync>(
        &self,
        size: usize,
    ) -> LinalgResult<Box<dyn GpuBuffer<T>>>;
}

/// GPU linear algebra operations trait
pub trait GpuLinalgOps<T>: Send + Sync
where
    T: Float + NumAssign + Zero + Send + Sync + Debug + 'static,
{
    /// GPU matrix-vector multiplication
    fn gpu_matvec(
        &self,
        ctx: &dyn GpuContext,
        a: &ArrayView2<T>,
        x: &ArrayView1<T>,
    ) -> LinalgResult<Array1<T>>;

    /// GPU matrix-matrix multiplication  
    fn gpu_matmul(
        &self,
        ctx: &dyn GpuContext,
        a: &ArrayView2<T>,
        b: &ArrayView2<T>,
    ) -> LinalgResult<Array2<T>>;

    /// GPU vector dot product
    fn gpu_dot(
        &self,
        ctx: &dyn GpuContext,
        x: &ArrayView1<T>,
        y: &ArrayView1<T>,
    ) -> LinalgResult<T>;

    /// GPU vector norm computation
    fn gpu_norm(&self, ctx: &dyn GpuContext, x: &ArrayView1<T>) -> LinalgResult<T>;

    /// GPU element-wise operations
    fn gpu_elementwise_add(
        &self,
        ctx: &dyn GpuContext,
        a: &ArrayView2<T>,
        b: &ArrayView2<T>,
    ) -> LinalgResult<Array2<T>>;

    /// GPU element-wise multiplication
    fn gpu_elementwise_mul(
        &self,
        ctx: &dyn GpuContext,
        a: &ArrayView2<T>,
        b: &ArrayView2<T>,
    ) -> LinalgResult<Array2<T>>;
}

/// GPU backend factory for creating contexts
pub trait GpuBackend: Send + Sync {
    /// Get backend name
    fn name(&self) -> &str;

    /// Check if backend is available
    fn is_available(&self) -> bool;

    /// List available devices
    fn list_devices(&self) -> LinalgResult<Vec<GpuDeviceInfo>>;

    /// Create context for specified device
    fn create_context(&self, device_id: usize) -> LinalgResult<Box<dyn GpuContext>>;

    /// Create context for best available device
    fn create_best_context(&self) -> LinalgResult<Box<dyn GpuContext>> {
        let devices = self.list_devices()?;
        if devices.is_empty() {
            return Err(LinalgError::ComputationError(
                "No GPU devices available".to_string(),
            ));
        }

        // Find device with most memory as a simple heuristic
        let best_device = devices
            .iter()
            .enumerate()
            .max_by_key(|(_, device)| device.total_memory)
            .map(|(idx, _)| idx)
            .unwrap();

        self.create_context(best_device)
    }
}

/// GPU manager for handling multiple backends
#[derive(Default)]
pub struct GpuManager {
    backends: Vec<Box<dyn GpuBackend>>,
}

impl GpuManager {
    /// Create a new GPU manager
    pub fn new() -> Self {
        Self {
            backends: Vec::new(),
        }
    }

    /// Register a GPU backend
    pub fn register_backend(&mut self, backend: Box<dyn GpuBackend>) {
        if backend.is_available() {
            self.backends.push(backend);
        }
    }

    /// Get all available backends
    pub fn available_backends(&self) -> &[Box<dyn GpuBackend>] {
        &self.backends
    }

    /// Get backend by name
    pub fn get_backend(&self, name: &str) -> Option<&dyn GpuBackend> {
        self.backends
            .iter()
            .find(|backend| backend.name() == name)
            .map(|b| b.as_ref())
    }

    /// Create context using the best available backend
    pub fn create_best_context(&self) -> LinalgResult<Box<dyn GpuContext>> {
        if self.backends.is_empty() {
            return Err(LinalgError::ComputationError(
                "No GPU backends available".to_string(),
            ));
        }

        // Try backends in order of preference
        for backend in &self.backends {
            if let Ok(context) = backend.create_best_context() {
                return Ok(context);
            }
        }

        Err(LinalgError::ComputationError(
            "Failed to create GPU context with any backend".to_string(),
        ))
    }

    /// List all available devices across all backends
    pub fn list_all_devices(&self) -> LinalgResult<Vec<(String, Vec<GpuDeviceInfo>)>> {
        let mut all_devices = Vec::new();

        for backend in &self.backends {
            let devices = backend.list_devices()?;
            all_devices.push((backend.name().to_string(), devices));
        }

        Ok(all_devices)
    }
}

/// Initialize GPU manager with all available backends
pub fn initialize_gpu_manager() -> LinalgResult<GpuManager> {
    let mut manager = GpuManager::new();

    // Register CUDA backend if available
    #[cfg(feature = "cuda")]
    {
        if let Ok(cuda_backend) = backends::cuda::CudaBackend::new() {
            manager.register_backend(Box::new(cuda_backend));
        }
    }

    // Register OpenCL backend if available
    #[cfg(feature = "opencl")]
    {
        if let Ok(opencl_backend) = backends::opencl::OpenClBackend::new() {
            manager.register_backend(Box::new(opencl_backend));
        }
    }

    // Register ROCm backend if available
    #[cfg(feature = "rocm")]
    {
        if let Ok(rocm_backend) = backends::rocm::RocmBackend::new() {
            manager.register_backend(Box::new(rocm_backend));
        }
    }

    // Register Metal backend if available
    #[cfg(feature = "metal")]
    {
        if let Ok(metal_backend) = backends::metal::MetalBackend::new() {
            manager.register_backend(Box::new(metal_backend));
        }
    }

    Ok(manager)
}

/// Determine if GPU acceleration should be used based on problem size
pub fn should_use_gpu(
    matrix_elements: usize,
    threshold: usize,
    gpu_context: Option<&dyn GpuContext>,
) -> bool {
    // GPU is beneficial for larger problems and when GPU context is available
    gpu_context.is_some() && matrix_elements > threshold
}

/// Auto-select between CPU and GPU based on problem characteristics
pub trait AutoGpuSelector<T>
where
    T: Float + NumAssign + Zero + Send + Sync + Debug + 'static,
{
    /// Automatically choose the best implementation for matrix-vector multiplication
    fn auto_matvec(
        &self,
        a: &ArrayView2<T>,
        x: &ArrayView1<T>,
        gpu_context: Option<&dyn GpuContext>,
    ) -> LinalgResult<Array1<T>>;

    /// Automatically choose the best implementation for matrix-matrix multiplication
    fn auto_matmul(
        &self,
        a: &ArrayView2<T>,
        b: &ArrayView2<T>,
        gpu_context: Option<&dyn GpuContext>,
    ) -> LinalgResult<Array2<T>>;
}

/// Default thresholds for GPU usage (number of elements)
pub const DEFAULT_GPU_THRESHOLD_MATVEC: usize = 10_000;
pub const DEFAULT_GPU_THRESHOLD_MATMUL: usize = 100_000;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_manager_creation() {
        let manager = GpuManager::new();
        assert_eq!(manager.available_backends().len(), 0);
    }

    #[test]
    fn test_should_use_gpu_threshold() {
        // Below threshold should not use GPU
        assert!(!should_use_gpu(100, 1000, None));

        // Above threshold but no GPU context
        assert!(!should_use_gpu(2000, 1000, None));

        // Would use GPU if context was available
        // (We can't test with actual context without GPU backends)
    }

    #[test]
    fn test_gpu_device_type_equality() {
        assert_eq!(GpuDeviceType::Cuda, GpuDeviceType::Cuda);
        assert_ne!(GpuDeviceType::Cuda, GpuDeviceType::OpenCl);
    }
}
