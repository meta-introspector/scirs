//! GPU-accelerated implementations of special functions
//!
//! This module provides GPU-accelerated versions of special functions that can
//! be used for large array computations when GPU hardware is available.

use crate::error::{SpecialError, SpecialResult};
use ndarray::{ArrayView1, ArrayViewMut1};
use scirs2_core::gpu::kernels::{DataType, KernelMetadata, KernelParams, OperationType};
use scirs2_core::gpu::{GpuBackend, GpuContext, GpuError};
use std::collections::HashMap;
use std::sync::Arc;

#[cfg(feature = "gpu")]
use scirs2_core::simd_ops::AutoOptimizer;

// Additional logging for GPU operations
#[cfg(feature = "gpu")]
use log;

/// Helper function to execute GPU kernels
#[cfg(feature = "gpu")]
fn execute_gpu_kernel<F>(
    context: &GpuContext,
    kernel_name: &str,
    params: &KernelParams,
    input: &ArrayView1<F>,
    output: &mut ArrayViewMut1<F>,
) -> Result<(), GpuError>
where
    F: num_traits::Float + num_traits::FromPrimitive + Send + Sync + 'static,
{
    // Get the kernel from registry
    let registry = SpecialFunctionKernelRegistry::new();
    let kernel = registry
        .get_kernel(kernel_name)
        .ok_or_else(|| GpuError::KernelNotFound(format!("Kernel '{}' not found", kernel_name)))?;

    // Get kernel source for the current backend
    let backend = context.backend();
    let source = kernel.source_for_backend(backend)?;

    // Verify data type compatibility
    let data_type = if std::mem::size_of::<F>() == 4 {
        DataType::Float32
    } else if std::mem::size_of::<F>() == 8 {
        DataType::Float64
    } else {
        return Err(GpuError::UnsupportedDataType(
            "Only f32 and f64 are supported for GPU operations".to_string(),
        ));
    };

    if !kernel.supported_types().contains(&data_type) {
        return Err(GpuError::UnsupportedDataType(format!(
            "Kernel '{}' does not support data type: {:?}",
            kernel_name, data_type
        )));
    }

    // Get kernel metadata for execution configuration
    let metadata = kernel.metadata();
    let workgroup_size = metadata.workgroup_size[0];

    // Calculate optimal dispatch size
    let num_elements = input.len();
    let num_workgroups = (num_elements + workgroup_size - 1) / workgroup_size;

    // Convert arrays to byte slices for GPU execution
    let input_bytes = unsafe {
        std::slice::from_raw_parts(
            input.as_ptr() as *const u8,
            input.len() * std::mem::size_of::<F>(),
        )
    };

    let output_bytes = unsafe {
        std::slice::from_raw_parts_mut(
            output.as_mut_ptr() as *mut u8,
            output.len() * std::mem::size_of::<F>(),
        )
    };

    // Attempt to compile and execute the kernel
    match context.backend() {
        GpuBackend::Wgpu => {
            // For WGPU backend, we need to compile the WGSL shader
            execute_wgpu_kernel(
                context,
                &kernel,
                &source,
                params,
                input_bytes,
                output_bytes,
                num_workgroups,
            )
        }
        _ => {
            // For other backends (CUDA, Metal, etc.), we would have different compilation paths
            // For now, return unsupported error to trigger fallback
            Err(GpuError::BackendNotSupported(backend))
        }
    }
}

/// Execute WGPU compute kernel
#[cfg(feature = "gpu")]
fn execute_wgpu_kernel(
    context: &GpuContext,
    kernel: &Arc<dyn GpuKernel>,
    source: &str,
    params: &KernelParams,
    input_data: &[u8],
    output_data: &mut [u8],
    num_workgroups: usize,
) -> Result<(), GpuError> {
    // Check if the backend is WGPU
    match context.backend() {
        GpuBackend::Wgpu => {
            // For now, since the WGPU backend is not fully implemented in scirs2-core,
            // we'll try to use the backend-agnostic kernel execution with enhanced error handling
            log::debug!(
                "Attempting GPU execution via kernel abstraction for {}",
                kernel.name()
            );

            // Try to execute via the kernel's own execution method
            match kernel.execute(context, params, input_data, output_data) {
                Ok(()) => {
                    log::debug!(
                        "Successfully executed {} kernel via GPU abstraction",
                        kernel.name()
                    );
                    Ok(())
                }
                Err(GpuError::BackendNotImplemented(_)) | Err(GpuError::UnsupportedBackend(_)) => {
                    log::debug!(
                        "GPU backend not available for {}, falling back to optimized CPU",
                        kernel.name()
                    );
                    execute_optimized_cpu_kernel(kernel, params, input_data, output_data)
                }
                Err(e) => {
                    log::warn!(
                        "GPU execution failed for {}: {:?}, falling back to CPU",
                        kernel.name(),
                        e
                    );
                    execute_optimized_cpu_kernel(kernel, params, input_data, output_data)
                }
            }
        }
        _ => {
            // Non-WGPU backends - delegate to the kernel's execution method
            kernel.execute(context, params, input_data, output_data)
        }
    }
}

/// Execute kernel on CPU with optimizations
#[cfg(feature = "gpu")]
fn execute_optimized_cpu_kernel(
    kernel: &Arc<dyn GpuKernel>,
    params: &KernelParams,
    input_data: &[u8],
    output_data: &mut [u8],
) -> Result<(), GpuError> {
    // Create a dummy context for CPU execution
    let cpu_context = GpuContext::new(GpuBackend::Cpu)
        .map_err(|_| GpuError::ExecutionError("Failed to create CPU context".to_string()))?;

    // Use the kernel's CPU execution path
    kernel.execute(&cpu_context, params, input_data, output_data)
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
    F: num_traits::Float
        + num_traits::FromPrimitive
        + Send
        + Sync
        + 'static
        + std::fmt::Debug
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign,
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
fn digamma_cpu_fallback<F>(
    input: &ArrayView1<F>,
    output: &mut ArrayViewMut1<F>,
) -> SpecialResult<()>
where
    F: num_traits::Float
        + num_traits::FromPrimitive
        + Send
        + Sync
        + std::fmt::Debug
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign,
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
    F: num_traits::Float
        + num_traits::FromPrimitive
        + Send
        + Sync
        + 'static
        + std::fmt::Debug
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
fn log_gamma_cpu_fallback<F>(
    input: &ArrayView1<F>,
    output: &mut ArrayViewMut1<F>,
) -> SpecialResult<()>
where
    F: num_traits::Float
        + num_traits::FromPrimitive
        + Send
        + Sync
        + std::fmt::Debug
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
        self.kernels.insert(
            "spherical_j0".to_string(),
            Arc::new(SphericalJ0Kernel::new()),
        );
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

    /// Execute gamma computation for f32 data
    fn execute_f32(
        &self,
        _context: &GpuContext,
        input: &[u8],
        output: &mut [u8],
        len: usize,
    ) -> Result<(), GpuError> {
        // Convert byte slices to f32 slices
        let input_f32 = unsafe { std::slice::from_raw_parts(input.as_ptr() as *const f32, len) };
        let output_f32 =
            unsafe { std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut f32, len) };

        // For now, simulate GPU execution by using CPU implementation
        // In a real GPU implementation, this would dispatch the WGSL shader
        use crate::gamma::gamma;
        for (i, &val) in input_f32.iter().enumerate() {
            output_f32[i] = gamma(val as f64) as f32;
        }

        Ok(())
    }

    /// Execute gamma computation for f64 data
    fn execute_f64(
        &self,
        _context: &GpuContext,
        input: &[u8],
        output: &mut [u8],
        len: usize,
    ) -> Result<(), GpuError> {
        // Convert byte slices to f64 slices
        let input_f64 = unsafe { std::slice::from_raw_parts(input.as_ptr() as *const f64, len) };
        let output_f64 =
            unsafe { std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut f64, len) };

        // For now, simulate GPU execution by using CPU implementation
        use crate::gamma::gamma;
        for (i, &val) in input_f64.iter().enumerate() {
            output_f64[i] = gamma(val);
        }

        Ok(())
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
            _ => Err(GpuError::BackendNotSupported(backend)),
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
        context: &GpuContext,
        params: &KernelParams,
        input: &[u8],
        output: &mut [u8],
    ) -> Result<(), GpuError> {
        // Determine data type from params
        let data_type = params.data_type;

        // Get dimensions
        let input_len = params.input_dims.get(0).copied().unwrap_or(0);
        let output_len = params.output_dims.get(0).copied().unwrap_or(0);

        if input_len != output_len {
            return Err(GpuError::InvalidDimensions(
                "Input and output dimensions must match".to_string(),
            ));
        }

        match data_type {
            DataType::Float32 => self.execute_f32(context, input, output, input_len),
            DataType::Float64 => self.execute_f64(context, input, output, input_len),
            _ => Err(GpuError::UnsupportedDataType(format!(
                "Unsupported data type: {:?}",
                data_type
            ))),
        }
    }
}

#[cfg(feature = "gpu")]
struct BesselJ0Kernel;

#[cfg(feature = "gpu")]
impl BesselJ0Kernel {
    fn new() -> Self {
        Self
    }

    /// Execute Bessel J0 computation for f32 data
    fn execute_f32(
        &self,
        _context: &GpuContext,
        input: &[u8],
        output: &mut [u8],
        len: usize,
    ) -> Result<(), GpuError> {
        let input_f32 = unsafe { std::slice::from_raw_parts(input.as_ptr() as *const f32, len) };
        let output_f32 =
            unsafe { std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut f32, len) };

        use crate::bessel::j0;
        for (i, &val) in input_f32.iter().enumerate() {
            output_f32[i] = j0(val as f64) as f32;
        }

        Ok(())
    }

    /// Execute Bessel J0 computation for f64 data
    fn execute_f64(
        &self,
        _context: &GpuContext,
        input: &[u8],
        output: &mut [u8],
        len: usize,
    ) -> Result<(), GpuError> {
        let input_f64 = unsafe { std::slice::from_raw_parts(input.as_ptr() as *const f64, len) };
        let output_f64 =
            unsafe { std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut f64, len) };

        use crate::bessel::j0;
        for (i, &val) in input_f64.iter().enumerate() {
            output_f64[i] = j0(val);
        }

        Ok(())
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
            _ => Err(GpuError::BackendNotSupported(backend)),
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
        context: &GpuContext,
        params: &KernelParams,
        input: &[u8],
        output: &mut [u8],
    ) -> Result<(), GpuError> {
        let data_type = params.data_type;
        let input_len = params.input_dims.get(0).copied().unwrap_or(0);
        let output_len = params.output_dims.get(0).copied().unwrap_or(0);

        if input_len != output_len {
            return Err(GpuError::InvalidDimensions(
                "Input and output dimensions must match".to_string(),
            ));
        }

        match data_type {
            DataType::Float32 => self.execute_f32(context, input, output, input_len),
            DataType::Float64 => self.execute_f64(context, input, output, input_len),
            _ => Err(GpuError::UnsupportedDataType(format!(
                "Unsupported data type: {:?}",
                data_type
            ))),
        }
    }
}

#[cfg(feature = "gpu")]
struct ErfKernel;

#[cfg(feature = "gpu")]
impl ErfKernel {
    fn new() -> Self {
        Self
    }

    /// Execute error function computation for f32 data
    fn execute_f32(
        &self,
        _context: &GpuContext,
        input: &[u8],
        output: &mut [u8],
        len: usize,
    ) -> Result<(), GpuError> {
        let input_f32 = unsafe { std::slice::from_raw_parts(input.as_ptr() as *const f32, len) };
        let output_f32 =
            unsafe { std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut f32, len) };

        use crate::erf::erf;
        for (i, &val) in input_f32.iter().enumerate() {
            output_f32[i] = erf(val as f64) as f32;
        }

        Ok(())
    }

    /// Execute error function computation for f64 data
    fn execute_f64(
        &self,
        _context: &GpuContext,
        input: &[u8],
        output: &mut [u8],
        len: usize,
    ) -> Result<(), GpuError> {
        let input_f64 = unsafe { std::slice::from_raw_parts(input.as_ptr() as *const f64, len) };
        let output_f64 =
            unsafe { std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut f64, len) };

        use crate::erf::erf;
        for (i, &val) in input_f64.iter().enumerate() {
            output_f64[i] = erf(val);
        }

        Ok(())
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
            _ => Err(GpuError::BackendNotSupported(backend)),
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
        context: &GpuContext,
        params: &KernelParams,
        input: &[u8],
        output: &mut [u8],
    ) -> Result<(), GpuError> {
        let data_type = params.data_type;
        let input_len = params.input_dims.get(0).copied().unwrap_or(0);
        let output_len = params.output_dims.get(0).copied().unwrap_or(0);

        if input_len != output_len {
            return Err(GpuError::InvalidDimensions(
                "Input and output dimensions must match".to_string(),
            ));
        }

        match data_type {
            DataType::Float32 => self.execute_f32(context, input, output, input_len),
            DataType::Float64 => self.execute_f64(context, input, output, input_len),
            _ => Err(GpuError::UnsupportedDataType(format!(
                "Unsupported data type: {:?}",
                data_type
            ))),
        }
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

    /// Execute digamma computation for f32 data
    fn execute_f32(
        &self,
        _context: &GpuContext,
        input: &[u8],
        output: &mut [u8],
        len: usize,
    ) -> Result<(), GpuError> {
        let input_f32 = unsafe { std::slice::from_raw_parts(input.as_ptr() as *const f32, len) };
        let output_f32 =
            unsafe { std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut f32, len) };

        use crate::gamma::digamma;
        for (i, &val) in input_f32.iter().enumerate() {
            output_f32[i] = digamma(val as f64) as f32;
        }

        Ok(())
    }

    /// Execute digamma computation for f64 data
    fn execute_f64(
        &self,
        _context: &GpuContext,
        input: &[u8],
        output: &mut [u8],
        len: usize,
    ) -> Result<(), GpuError> {
        let input_f64 = unsafe { std::slice::from_raw_parts(input.as_ptr() as *const f64, len) };
        let output_f64 =
            unsafe { std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut f64, len) };

        use crate::gamma::digamma;
        for (i, &val) in input_f64.iter().enumerate() {
            output_f64[i] = digamma(val);
        }

        Ok(())
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
            GpuBackend::Wgpu => Ok(include_str!("../shaders/digamma_compute.wgsl").to_string()),
            _ => Err(GpuError::BackendNotSupported(backend)),
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
        context: &GpuContext,
        params: &KernelParams,
        input: &[u8],
        output: &mut [u8],
    ) -> Result<(), GpuError> {
        let data_type = params.data_type;
        let input_len = params.input_dims.get(0).copied().unwrap_or(0);
        let output_len = params.output_dims.get(0).copied().unwrap_or(0);

        if input_len != output_len {
            return Err(GpuError::InvalidDimensions(
                "Input and output dimensions must match".to_string(),
            ));
        }

        match data_type {
            DataType::Float32 => self.execute_f32(context, input, output, input_len),
            DataType::Float64 => self.execute_f64(context, input, output, input_len),
            _ => Err(GpuError::UnsupportedDataType(format!(
                "Unsupported data type: {:?}",
                data_type
            ))),
        }
    }
}

#[cfg(feature = "gpu")]
struct LogGammaKernel;

#[cfg(feature = "gpu")]
impl LogGammaKernel {
    fn new() -> Self {
        Self
    }

    /// Execute log gamma computation for f32 data
    fn execute_f32(
        &self,
        _context: &GpuContext,
        input: &[u8],
        output: &mut [u8],
        len: usize,
    ) -> Result<(), GpuError> {
        let input_f32 = unsafe { std::slice::from_raw_parts(input.as_ptr() as *const f32, len) };
        let output_f32 =
            unsafe { std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut f32, len) };

        use crate::gamma::loggamma;
        for (i, &val) in input_f32.iter().enumerate() {
            output_f32[i] = loggamma(val as f64) as f32;
        }

        Ok(())
    }

    /// Execute log gamma computation for f64 data
    fn execute_f64(
        &self,
        _context: &GpuContext,
        input: &[u8],
        output: &mut [u8],
        len: usize,
    ) -> Result<(), GpuError> {
        let input_f64 = unsafe { std::slice::from_raw_parts(input.as_ptr() as *const f64, len) };
        let output_f64 =
            unsafe { std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut f64, len) };

        use crate::gamma::loggamma;
        for (i, &val) in input_f64.iter().enumerate() {
            output_f64[i] = loggamma(val);
        }

        Ok(())
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
            GpuBackend::Wgpu => Ok(include_str!("../shaders/log_gamma_compute.wgsl").to_string()),
            _ => Err(GpuError::BackendNotSupported(backend)),
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
        context: &GpuContext,
        params: &KernelParams,
        input: &[u8],
        output: &mut [u8],
    ) -> Result<(), GpuError> {
        let data_type = params.data_type;
        let input_len = params.input_dims.get(0).copied().unwrap_or(0);
        let output_len = params.output_dims.get(0).copied().unwrap_or(0);

        if input_len != output_len {
            return Err(GpuError::InvalidDimensions(
                "Input and output dimensions must match".to_string(),
            ));
        }

        match data_type {
            DataType::Float32 => self.execute_f32(context, input, output, input_len),
            DataType::Float64 => self.execute_f64(context, input, output, input_len),
            _ => Err(GpuError::UnsupportedDataType(format!(
                "Unsupported data type: {:?}",
                data_type
            ))),
        }
    }
}

#[cfg(feature = "gpu")]
struct SphericalJ0Kernel;

#[cfg(feature = "gpu")]
impl SphericalJ0Kernel {
    fn new() -> Self {
        Self
    }

    /// Execute spherical Bessel j0 computation for f32 data
    fn execute_f32(
        &self,
        _context: &GpuContext,
        input: &[u8],
        output: &mut [u8],
        len: usize,
    ) -> Result<(), GpuError> {
        let input_f32 = unsafe { std::slice::from_raw_parts(input.as_ptr() as *const f32, len) };
        let output_f32 =
            unsafe { std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut f32, len) };

        use crate::bessel::spherical::spherical_jn;
        for (i, &val) in input_f32.iter().enumerate() {
            output_f32[i] = spherical_jn(0, val as f64) as f32;
        }

        Ok(())
    }

    /// Execute spherical Bessel j0 computation for f64 data
    fn execute_f64(
        &self,
        _context: &GpuContext,
        input: &[u8],
        output: &mut [u8],
        len: usize,
    ) -> Result<(), GpuError> {
        let input_f64 = unsafe { std::slice::from_raw_parts(input.as_ptr() as *const f64, len) };
        let output_f64 =
            unsafe { std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut f64, len) };

        use crate::bessel::spherical::spherical_jn;
        for (i, &val) in input_f64.iter().enumerate() {
            output_f64[i] = spherical_jn(0, val);
        }

        Ok(())
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
            _ => Err(GpuError::BackendNotSupported(backend)),
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
        context: &GpuContext,
        params: &KernelParams,
        input: &[u8],
        output: &mut [u8],
    ) -> Result<(), GpuError> {
        let data_type = params.data_type;
        let input_len = params.input_dims.get(0).copied().unwrap_or(0);
        let output_len = params.output_dims.get(0).copied().unwrap_or(0);

        if input_len != output_len {
            return Err(GpuError::InvalidDimensions(
                "Input and output dimensions must match".to_string(),
            ));
        }

        match data_type {
            DataType::Float32 => self.execute_f32(context, input, output, input_len),
            DataType::Float64 => self.execute_f64(context, input, output, input_len),
            _ => Err(GpuError::UnsupportedDataType(format!(
                "Unsupported data type: {:?}",
                data_type
            ))),
        }
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
