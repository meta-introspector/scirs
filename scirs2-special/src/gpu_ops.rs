//! GPU-accelerated implementations of special functions
//!
//! This module provides GPU-accelerated versions of special functions that can
//! be used for large array computations when GPU hardware is available.

use crate::error::{SpecialError, SpecialResult};
use ndarray::{ArrayView1, ArrayViewMut1};
use scirs2_core::gpu::kernels::{DataType, KernelParams, KernelMetadata, OperationType};
use scirs2_core::gpu::{GpuContext, GpuError, GpuBackend};
use std::sync::Arc;
use std::collections::HashMap;

#[cfg(feature = "gpu")]
use scirs2_core::simd_ops::AutoOptimizer;

/// Helper function to execute GPU kernels
#[cfg(feature = "gpu")]
fn execute_gpu_kernel<F>(
    context: &GpuContext,
    kernel_name: &str,
    _params: &KernelParams,
    _input: &ArrayView1<F>,
    _output: &mut ArrayViewMut1<F>,
) -> Result<(), GpuError> {
    // Get the kernel from registry
    let registry = SpecialFunctionKernelRegistry::new();
    let kernel = registry.get_kernel(kernel_name)
        .ok_or_else(|| GpuError::KernelNotFound(format!("Kernel '{}' not found", kernel_name)))?;
    
    // Get kernel source for the current backend
    let backend = context.backend();
    let _source = kernel.source_for_backend(backend)?;
    
    // In a full implementation, this would:
    // 1. Compile the shader if needed (using wgpu, CUDA, etc.)
    // 2. Transfer data to GPU
    // 3. Execute the kernel with proper workgroup size
    // 4. Transfer results back
    
    // For now, since the full GPU infrastructure isn't complete,
    // we return an error to trigger CPU fallback
    Err(GpuError::KernelNotFound(
        "GPU kernel execution infrastructure in development".to_string(),
    ))
}

/// GPU-accelerated gamma function for arrays
///
/// This function automatically selects between GPU and CPU execution based on:
/// - Array size (GPU is beneficial for arrays > 10,000 elements)
/// - Available hardware (falls back to CPU if no GPU available)
/// - Data type (optimized for f32 and f64)
///
/// # Arguments
///
/// * `input` - Input array view
/// * `output` - Mutable output array view (must be same size as input)
///
/// # Returns
///
/// * `SpecialResult<()>` - Ok if successful, error otherwise
///
/// # Examples
///
/// ```no_run
/// use ndarray::Array1;
/// use scirs2_special::gpu_ops::gamma_gpu;
///
/// let input = Array1::linspace(0.1, 10.0, 1000);
/// let mut output = Array1::zeros(1000);
///
/// gamma_gpu(&input.view(), &mut output.view_mut()).unwrap();
/// ```
#[cfg(feature = "gpu")]
pub fn gamma_gpu<F>(input: &ArrayView1<F>, output: &mut ArrayViewMut1<F>) -> SpecialResult<()>
where
    F: num_traits::Float
        + num_traits::FromPrimitive
        + std::fmt::Debug
        + std::ops::AddAssign
        + Send
        + Sync
        + 'static,
{
    // Validate input dimensions
    if input.len() != output.len() {
        return Err(SpecialError::ValueError(
            "Input and output arrays must have the same length".to_string(),
        ));
    }

    // Use the AutoOptimizer to decide whether to use GPU
    let optimizer = AutoOptimizer::new();
    let array_size = input.len();

    if !optimizer.should_use_gpu(array_size) {
        // Fall back to CPU implementation
        return gamma_cpu_fallback(input, output);
    }

    // Initialize GPU context
    let gpu_context = match GpuContext::new(scirs2_core::gpu::GpuBackend::default()) {
        Ok(ctx) => ctx,
        Err(_) => {
            // Fall back to CPU if GPU initialization fails
            return gamma_cpu_fallback(input, output);
        }
    };

    // Prepare kernel parameters
    let data_type = if std::mem::size_of::<F>() == 4 {
        DataType::Float32
    } else if std::mem::size_of::<F>() == 8 {
        DataType::Float64
    } else {
        return Err(SpecialError::ComputationError(
            "Unsupported data type for GPU acceleration".to_string(),
        ));
    };

    let kernel_params = KernelParams {
        data_type,
        input_dims: vec![array_size],
        output_dims: vec![array_size],
        numeric_params: Default::default(),
        string_params: Default::default(),
    };

    // Execute the gamma kernel
    // Note: execute_kernel is a placeholder - actual implementation would use GPU compiler
    match execute_gpu_kernel(&gpu_context, "gamma", &kernel_params, input, output) {
        Ok(_) => Ok(()),
        Err(_) => {
            // Fall back to CPU on kernel execution error
            gamma_cpu_fallback(input, output)
        }
    }
}

/// CPU fallback implementation for gamma function
#[cfg(feature = "gpu")]
fn gamma_cpu_fallback<F>(input: &ArrayView1<F>, output: &mut ArrayViewMut1<F>) -> SpecialResult<()>
where
    F: num_traits::Float
        + num_traits::FromPrimitive
        + std::fmt::Debug
        + std::ops::AddAssign
        + Send
        + Sync,
{
    use crate::gamma::gamma;
    // Use parallel processing for large arrays if available
    #[cfg(feature = "parallel")]
    {
        use scirs2_core::parallel_ops::*;
        if is_parallel_enabled() && input.len() > 1000 {
            use scirs2_core::parallel_ops::IntoParallelRefIterator;
            use scirs2_core::parallel_ops::IntoParallelRefMutIterator;

            input
                .as_slice()
                .unwrap()
                .par_iter()
                .zip(output.as_slice_mut().unwrap().par_iter_mut())
                .for_each(|(inp, out)| {
                    *out = gamma(*inp);
                });
            return Ok(());
        }
    }
    
    // Sequential processing as fallback or for small arrays
    for (inp, out) in input.iter().zip(output.iter_mut()) {
        *out = gamma(*inp);
    }

    Ok(())
}

/// GPU-accelerated Bessel J0 function for arrays
#[cfg(feature = "gpu")]
pub fn j0_gpu<F>(input: &ArrayView1<F>, output: &mut ArrayViewMut1<F>) -> SpecialResult<()>
where
    F: num_traits::Float + num_traits::FromPrimitive + std::fmt::Debug + Send + Sync + 'static,
{
    // Similar implementation pattern as gamma_gpu
    if input.len() != output.len() {
        return Err(SpecialError::ValueError(
            "Input and output arrays must have the same length".to_string(),
        ));
    }

    let optimizer = AutoOptimizer::new();
    if !optimizer.should_use_gpu(input.len()) {
        return j0_cpu_fallback(input, output);
    }

    let gpu_context = match GpuContext::new(scirs2_core::gpu::GpuBackend::default()) {
        Ok(ctx) => ctx,
        Err(_) => return j0_cpu_fallback(input, output),
    };

    let data_type = if std::mem::size_of::<F>() == 4 {
        DataType::Float32
    } else if std::mem::size_of::<F>() == 8 {
        DataType::Float64
    } else {
        return Err(SpecialError::ComputationError(
            "Unsupported data type for GPU acceleration".to_string(),
        ));
    };

    let kernel_params = KernelParams {
        data_type,
        input_dims: vec![input.len()],
        output_dims: vec![output.len()],
        numeric_params: Default::default(),
        string_params: Default::default(),
    };

    match execute_gpu_kernel(&gpu_context, "bessel_j0", &kernel_params, input, output) {
        Ok(_) => Ok(()),
        Err(_) => j0_cpu_fallback(input, output),
    }
}

/// CPU fallback implementation for Bessel J0 function
#[cfg(feature = "gpu")]
fn j0_cpu_fallback<F>(input: &ArrayView1<F>, output: &mut ArrayViewMut1<F>) -> SpecialResult<()>
where
    F: num_traits::Float + num_traits::FromPrimitive + std::fmt::Debug + Send + Sync,
{
    use crate::bessel::j0;
    #[cfg(feature = "parallel")]
    {
        use scirs2_core::parallel_ops::*;
        if is_parallel_enabled() && input.len() > 1000 {
            use scirs2_core::parallel_ops::IntoParallelRefIterator;
            use scirs2_core::parallel_ops::IntoParallelRefMutIterator;

            input
                .as_slice()
                .unwrap()
                .par_iter()
                .zip(output.as_slice_mut().unwrap().par_iter_mut())
                .for_each(|(inp, out)| {
                    *out = j0(*inp);
                });
            return Ok(());
        }
    }
    
    for (inp, out) in input.iter().zip(output.iter_mut()) {
        *out = j0(*inp);
    }

    Ok(())
}

/// GPU-accelerated error function (erf) for arrays
#[cfg(feature = "gpu")]
pub fn erf_gpu<F>(input: &ArrayView1<F>, output: &mut ArrayViewMut1<F>) -> SpecialResult<()>
where
    F: num_traits::Float + num_traits::FromPrimitive + Send + Sync + 'static,
{
    if input.len() != output.len() {
        return Err(SpecialError::ValueError(
            "Input and output arrays must have the same length".to_string(),
        ));
    }

    let optimizer = AutoOptimizer::new();
    if !optimizer.should_use_gpu(input.len()) {
        return erf_cpu_fallback(input, output);
    }

    let gpu_context = match GpuContext::new(scirs2_core::gpu::GpuBackend::default()) {
        Ok(ctx) => ctx,
        Err(_) => return erf_cpu_fallback(input, output),
    };

    let data_type = if std::mem::size_of::<F>() == 4 {
        DataType::Float32
    } else if std::mem::size_of::<F>() == 8 {
        DataType::Float64
    } else {
        return Err(SpecialError::ComputationError(
            "Unsupported data type for GPU acceleration".to_string(),
        ));
    };

    let kernel_params = KernelParams {
        data_type,
        input_dims: vec![input.len()],
        output_dims: vec![output.len()],
        numeric_params: Default::default(),
        string_params: Default::default(),
    };

    match execute_gpu_kernel(&gpu_context, "erf", &kernel_params, input, output) {
        Ok(_) => Ok(()),
        Err(_) => erf_cpu_fallback(input, output),
    }
}

/// CPU fallback implementation for error function
#[cfg(feature = "gpu")]
fn erf_cpu_fallback<F>(input: &ArrayView1<F>, output: &mut ArrayViewMut1<F>) -> SpecialResult<()>
where
    F: num_traits::Float + num_traits::FromPrimitive + Send + Sync,
{
    use crate::erf::erf;
    #[cfg(feature = "parallel")]
    {
        use scirs2_core::parallel_ops::*;
        if is_parallel_enabled() && input.len() > 1000 {
            use scirs2_core::parallel_ops::IntoParallelRefIterator;
            use scirs2_core::parallel_ops::IntoParallelRefMutIterator;

            input
                .as_slice()
                .unwrap()
                .par_iter()
                .zip(output.as_slice_mut().unwrap().par_iter_mut())
                .for_each(|(inp, out)| {
                    *out = erf(*inp);
                });
            return Ok(());
        }
    }
    
    for (inp, out) in input.iter().zip(output.iter_mut()) {
        *out = erf(*inp);
    }

    Ok(())
}

/// GPU-accelerated digamma function for arrays
#[cfg(feature = "gpu")]
pub fn digamma_gpu<F>(input: &ArrayView1<F>, output: &mut ArrayViewMut1<F>) -> SpecialResult<()>
where
    F: num_traits::Float + num_traits::FromPrimitive + Send + Sync + 'static + std::fmt::Debug
        + std::ops::AddAssign + std::ops::SubAssign + std::ops::MulAssign + std::ops::DivAssign,
{
    if input.len() != output.len() {
        return Err(SpecialError::ValueError(
            "Input and output arrays must have the same length".to_string(),
        ));
    }

    let optimizer = AutoOptimizer::new();
    if !optimizer.should_use_gpu(input.len()) {
        return digamma_cpu_fallback(input, output);
    }

    let gpu_context = match GpuContext::new(scirs2_core::gpu::GpuBackend::default()) {
        Ok(ctx) => ctx,
        Err(_) => return digamma_cpu_fallback(input, output),
    };

    let data_type = if std::mem::size_of::<F>() == 4 {
        DataType::Float32
    } else if std::mem::size_of::<F>() == 8 {
        DataType::Float64
    } else {
        return Err(SpecialError::ComputationError(
            "Unsupported data type for GPU acceleration".to_string(),
        ));
    };

    let kernel_params = KernelParams {
        data_type,
        input_dims: vec![input.len()],
        output_dims: vec![output.len()],
        numeric_params: Default::default(),
        string_params: Default::default(),
    };

    match execute_gpu_kernel(&gpu_context, "digamma", &kernel_params, input, output) {
        Ok(_) => Ok(()),
        Err(_) => digamma_cpu_fallback(input, output),
    }
}

/// CPU fallback for digamma function
#[cfg(feature = "gpu")]
fn digamma_cpu_fallback<F>(input: &ArrayView1<F>, output: &mut ArrayViewMut1<F>) -> SpecialResult<()>
where
    F: num_traits::Float + num_traits::FromPrimitive + Send + Sync + std::fmt::Debug
        + std::ops::AddAssign + std::ops::SubAssign + std::ops::MulAssign + std::ops::DivAssign,
{
    use crate::gamma::digamma;
    #[cfg(feature = "parallel")]
    {
        use scirs2_core::parallel_ops::*;
        if is_parallel_enabled() && input.len() > 1000 {
            use scirs2_core::parallel_ops::IntoParallelRefIterator;
            use scirs2_core::parallel_ops::IntoParallelRefMutIterator;

            input
                .as_slice()
                .unwrap()
                .par_iter()
                .zip(output.as_slice_mut().unwrap().par_iter_mut())
                .for_each(|(inp, out)| {
                    *out = digamma(*inp);
                });
            return Ok(());
        }
    }
    
    for (inp, out) in input.iter().zip(output.iter_mut()) {
        *out = digamma(*inp);
    }

    Ok(())
}

/// GPU-accelerated log gamma function for arrays
#[cfg(feature = "gpu")]
pub fn log_gamma_gpu<F>(input: &ArrayView1<F>, output: &mut ArrayViewMut1<F>) -> SpecialResult<()>
where
    F: num_traits::Float + num_traits::FromPrimitive + Send + Sync + 'static + std::fmt::Debug
        + std::ops::AddAssign,
{
    if input.len() != output.len() {
        return Err(SpecialError::ValueError(
            "Input and output arrays must have the same length".to_string(),
        ));
    }

    let optimizer = AutoOptimizer::new();
    if !optimizer.should_use_gpu(input.len()) {
        return log_gamma_cpu_fallback(input, output);
    }

    let gpu_context = match GpuContext::new(scirs2_core::gpu::GpuBackend::default()) {
        Ok(ctx) => ctx,
        Err(_) => return log_gamma_cpu_fallback(input, output),
    };

    let data_type = if std::mem::size_of::<F>() == 4 {
        DataType::Float32
    } else if std::mem::size_of::<F>() == 8 {
        DataType::Float64
    } else {
        return Err(SpecialError::ComputationError(
            "Unsupported data type for GPU acceleration".to_string(),
        ));
    };

    let kernel_params = KernelParams {
        data_type,
        input_dims: vec![input.len()],
        output_dims: vec![output.len()],
        numeric_params: Default::default(),
        string_params: Default::default(),
    };

    match execute_gpu_kernel(&gpu_context, "log_gamma", &kernel_params, input, output) {
        Ok(_) => Ok(()),
        Err(_) => log_gamma_cpu_fallback(input, output),
    }
}

/// CPU fallback for log gamma function
#[cfg(feature = "gpu")]
fn log_gamma_cpu_fallback<F>(input: &ArrayView1<F>, output: &mut ArrayViewMut1<F>) -> SpecialResult<()>
where
    F: num_traits::Float + num_traits::FromPrimitive + Send + Sync + std::fmt::Debug
        + std::ops::AddAssign,
{
    use crate::gamma::loggamma;
    #[cfg(feature = "parallel")]
    {
        use scirs2_core::parallel_ops::*;
        if is_parallel_enabled() && input.len() > 1000 {
            use scirs2_core::parallel_ops::IntoParallelRefIterator;
            use scirs2_core::parallel_ops::IntoParallelRefMutIterator;

            input
                .as_slice()
                .unwrap()
                .par_iter()
                .zip(output.as_slice_mut().unwrap().par_iter_mut())
                .for_each(|(inp, out)| {
                    *out = loggamma(*inp);
                });
            return Ok(());
        }
    }
    
    for (inp, out) in input.iter().zip(output.iter_mut()) {
        *out = loggamma(*inp);
    }

    Ok(())
}

/// GPU kernel registry for special functions
///
/// This registry manages GPU kernels for various special functions and provides
/// a unified interface for kernel discovery and execution.
#[cfg(feature = "gpu")]
pub struct SpecialFunctionKernelRegistry {
    kernels: std::collections::HashMap<String, Arc<dyn GpuKernel>>,
}

#[cfg(feature = "gpu")]
impl SpecialFunctionKernelRegistry {
    /// Create a new kernel registry with default kernels
    pub fn new() -> Self {
        let mut registry = Self {
            kernels: std::collections::HashMap::new(),
        };

        // Register default kernels
        registry.register_default_kernels();
        registry
    }

    /// Register default GPU kernels for special functions
    fn register_default_kernels(&mut self) {
        // These would be actual kernel implementations in a full system
        // For now, they're placeholders that demonstrate the structure

        // Gamma function kernel
        self.kernels
            .insert("gamma".to_string(), Arc::new(GammaKernel::new()));

        // Bessel function kernels
        self.kernels
            .insert("bessel_j0".to_string(), Arc::new(BesselJ0Kernel::new()));

        // Error function kernel
        self.kernels
            .insert("erf".to_string(), Arc::new(ErfKernel::new()));
            
        // Digamma function kernel
        self.kernels
            .insert("digamma".to_string(), Arc::new(DigammaKernel::new()));
            
        // Log gamma function kernel
        self.kernels
            .insert("log_gamma".to_string(), Arc::new(LogGammaKernel::new()));
            
        // Spherical Bessel j0 kernel
        self.kernels
            .insert("spherical_j0".to_string(), Arc::new(SphericalJ0Kernel::new()));
    }

    /// Get a kernel by name
    pub fn get_kernel(&self, name: &str) -> Option<Arc<dyn GpuKernel>> {
        self.kernels.get(name).cloned()
    }
}

/// Trait for GPU kernels
#[cfg(feature = "gpu")]
trait GpuKernel: Send + Sync {
    /// Get the kernel name
    fn name(&self) -> &str;

    /// Get supported data types
    fn supported_types(&self) -> Vec<DataType>;
    
    /// Get kernel source for the specified backend
    fn source_for_backend(&self, backend: GpuBackend) -> Result<String, GpuError>;
    
    /// Get kernel metadata
    fn metadata(&self) -> KernelMetadata;

    /// Execute the kernel
    fn execute(
        &self,
        context: &GpuContext,
        params: &KernelParams,
        input: &[u8],
        output: &mut [u8],
    ) -> Result<(), GpuError>;
}

/// Placeholder kernel implementations
#[cfg(feature = "gpu")]
struct GammaKernel;

#[cfg(feature = "gpu")]
impl GammaKernel {
    fn new() -> Self {
        Self
    }
}

#[cfg(feature = "gpu")]
impl GpuKernel for GammaKernel {
    fn name(&self) -> &str {
        "gamma"
    }

    fn supported_types(&self) -> Vec<DataType> {
        vec![DataType::Float32, DataType::Float64]
    }
    
    fn source_for_backend(&self, backend: GpuBackend) -> Result<String, GpuError> {
        match backend {
            GpuBackend::Wgpu => {
                // Load WGSL shader
                Ok(include_str!("../shaders/gamma_compute.wgsl").to_string())
            }
            _ => Err(GpuError::BackendNotSupported(backend))
        }
    }
    
    fn metadata(&self) -> KernelMetadata {
        KernelMetadata {
            workgroup_size: [256, 1, 1],
            local_memory_usage: 0,
            supports_tensor_cores: false,
            operation_type: OperationType::ComputeIntensive,
            backend_metadata: HashMap::new(),
        }
    }

    fn execute(
        &self,
        _context: &GpuContext,
        _params: &KernelParams,
        _input: &[u8],
        _output: &mut [u8],
    ) -> Result<(), GpuError> {
        // In a full implementation, this would:
        // 1. Get the compiled shader from context
        // 2. Create GPU buffers for input/output
        // 3. Set up compute pipeline
        // 4. Dispatch the kernel
        // 5. Wait for completion and copy results back
        Ok(())
    }
}

#[cfg(feature = "gpu")]
struct BesselJ0Kernel;

#[cfg(feature = "gpu")]
impl BesselJ0Kernel {
    fn new() -> Self {
        Self
    }
}

#[cfg(feature = "gpu")]
impl GpuKernel for BesselJ0Kernel {
    fn name(&self) -> &str {
        "bessel_j0"
    }

    fn supported_types(&self) -> Vec<DataType> {
        vec![DataType::Float32, DataType::Float64]
    }
    
    fn source_for_backend(&self, backend: GpuBackend) -> Result<String, GpuError> {
        match backend {
            GpuBackend::Wgpu => {
                // Load WGSL shader
                Ok(include_str!("../shaders/bessel_j0_compute.wgsl").to_string())
            }
            _ => Err(GpuError::BackendNotSupported(backend))
        }
    }
    
    fn metadata(&self) -> KernelMetadata {
        KernelMetadata {
            workgroup_size: [256, 1, 1],
            local_memory_usage: 0,
            supports_tensor_cores: false,
            operation_type: OperationType::ComputeIntensive,
            backend_metadata: HashMap::new(),
        }
    }

    fn execute(
        &self,
        _context: &GpuContext,
        _params: &KernelParams,
        _input: &[u8],
        _output: &mut [u8],
    ) -> Result<(), GpuError> {
        Ok(())
    }
}

#[cfg(feature = "gpu")]
struct ErfKernel;

#[cfg(feature = "gpu")]
impl ErfKernel {
    fn new() -> Self {
        Self
    }
}

#[cfg(feature = "gpu")]
impl GpuKernel for ErfKernel {
    fn name(&self) -> &str {
        "erf"
    }

    fn supported_types(&self) -> Vec<DataType> {
        vec![DataType::Float32, DataType::Float64]
    }
    
    fn source_for_backend(&self, backend: GpuBackend) -> Result<String, GpuError> {
        match backend {
            GpuBackend::Wgpu => {
                // Load WGSL shader
                Ok(include_str!("../shaders/erf_compute.wgsl").to_string())
            }
            _ => Err(GpuError::BackendNotSupported(backend))
        }
    }
    
    fn metadata(&self) -> KernelMetadata {
        KernelMetadata {
            workgroup_size: [256, 1, 1],
            local_memory_usage: 0,
            supports_tensor_cores: false,
            operation_type: OperationType::ComputeIntensive,
            backend_metadata: HashMap::new(),
        }
    }

    fn execute(
        &self,
        _context: &GpuContext,
        _params: &KernelParams,
        _input: &[u8],
        _output: &mut [u8],
    ) -> Result<(), GpuError> {
        Ok(())
    }
}

// Additional kernel implementations

#[cfg(feature = "gpu")]
struct DigammaKernel;

#[cfg(feature = "gpu")]
impl DigammaKernel {
    fn new() -> Self {
        Self
    }
}

#[cfg(feature = "gpu")]
impl GpuKernel for DigammaKernel {
    fn name(&self) -> &str {
        "digamma"
    }

    fn supported_types(&self) -> Vec<DataType> {
        vec![DataType::Float32, DataType::Float64]
    }
    
    fn source_for_backend(&self, backend: GpuBackend) -> Result<String, GpuError> {
        match backend {
            GpuBackend::Wgpu => {
                Ok(include_str!("../shaders/digamma_compute.wgsl").to_string())
            }
            _ => Err(GpuError::BackendNotSupported(backend))
        }
    }
    
    fn metadata(&self) -> KernelMetadata {
        KernelMetadata {
            workgroup_size: [256, 1, 1],
            local_memory_usage: 0,
            supports_tensor_cores: false,
            operation_type: OperationType::ComputeIntensive,
            backend_metadata: HashMap::new(),
        }
    }

    fn execute(
        &self,
        _context: &GpuContext,
        _params: &KernelParams,
        _input: &[u8],
        _output: &mut [u8],
    ) -> Result<(), GpuError> {
        Ok(())
    }
}

#[cfg(feature = "gpu")]
struct LogGammaKernel;

#[cfg(feature = "gpu")]
impl LogGammaKernel {
    fn new() -> Self {
        Self
    }
}

#[cfg(feature = "gpu")]
impl GpuKernel for LogGammaKernel {
    fn name(&self) -> &str {
        "log_gamma"
    }

    fn supported_types(&self) -> Vec<DataType> {
        vec![DataType::Float32, DataType::Float64]
    }
    
    fn source_for_backend(&self, backend: GpuBackend) -> Result<String, GpuError> {
        match backend {
            GpuBackend::Wgpu => {
                Ok(include_str!("../shaders/log_gamma_compute.wgsl").to_string())
            }
            _ => Err(GpuError::BackendNotSupported(backend))
        }
    }
    
    fn metadata(&self) -> KernelMetadata {
        KernelMetadata {
            workgroup_size: [256, 1, 1],
            local_memory_usage: 0,
            supports_tensor_cores: false,
            operation_type: OperationType::ComputeIntensive,
            backend_metadata: HashMap::new(),
        }
    }

    fn execute(
        &self,
        _context: &GpuContext,
        _params: &KernelParams,
        _input: &[u8],
        _output: &mut [u8],
        ) -> Result<(), GpuError> {
        Ok(())
    }
}

#[cfg(feature = "gpu")]
struct SphericalJ0Kernel;

#[cfg(feature = "gpu")]
impl SphericalJ0Kernel {
    fn new() -> Self {
        Self
    }
}

#[cfg(feature = "gpu")]
impl GpuKernel for SphericalJ0Kernel {
    fn name(&self) -> &str {
        "spherical_j0"
    }

    fn supported_types(&self) -> Vec<DataType> {
        vec![DataType::Float32, DataType::Float64]
    }
    
    fn source_for_backend(&self, backend: GpuBackend) -> Result<String, GpuError> {
        match backend {
            GpuBackend::Wgpu => {
                Ok(include_str!("../shaders/spherical_j0_compute.wgsl").to_string())
            }
            _ => Err(GpuError::BackendNotSupported(backend))
        }
    }
    
    fn metadata(&self) -> KernelMetadata {
        KernelMetadata {
            workgroup_size: [256, 1, 1],
            local_memory_usage: 0,
            supports_tensor_cores: false,
            operation_type: OperationType::ComputeIntensive,
            backend_metadata: HashMap::new(),
        }
    }

    fn execute(
        &self,
        _context: &GpuContext,
        _params: &KernelParams,
        _input: &[u8],
        _output: &mut [u8],
    ) -> Result<(), GpuError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    #[cfg(feature = "gpu")]
    fn test_gamma_gpu_fallback() {
        // Test that CPU fallback works correctly
        let input = Array1::linspace(0.1, 5.0, 10);
        let mut output = Array1::zeros(10);

        gamma_gpu(&input.view(), &mut output.view_mut()).unwrap();

        // Verify some known values
        use crate::gamma::gamma;
        for i in 0..10 {
            let expected = gamma(input[i]);
            assert!((output[i] - expected).abs() < 1e-10);
        }
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn test_kernel_registry() {
        let registry = SpecialFunctionKernelRegistry::new();

        // Check that default kernels are registered
        assert!(registry.get_kernel("gamma").is_some());
        assert!(registry.get_kernel("bessel_j0").is_some());
        assert!(registry.get_kernel("erf").is_some());
    }
}
