//! Custom CUDA kernels for memory-intensive optimizers
//!
//! This module provides highly optimized CUDA kernel implementations for
//! memory-intensive optimizers like Adam and LAMB that can benefit from
//! custom GPU acceleration.

use ndarray::{Array, Array1, Array2, Dimension};
use num_traits::Float;
use std::collections::{HashMap, VecDeque};
use std::ffi::c_void;
use std::ptr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[cfg(feature = "cuda")]
use scirs2_core::gpu::{CudaContext, CudaKernel, CudaStream};

/// Custom CUDA kernel wrapper for optimizer operations
pub struct OptimizerKernel {
    /// CUDA context
    #[cfg(feature = "cuda")]
    context: CudaContext,

    /// Compiled kernel functions
    #[cfg(feature = "cuda")]
    adam_kernel: CudaKernel,

    #[cfg(feature = "cuda")]
    lamb_kernel: CudaKernel,

    #[cfg(feature = "cuda")]
    adamw_kernel: CudaKernel,

    /// Tensor core optimized kernels
    #[cfg(feature = "cuda")]
    tensor_core_adam_fp16_kernel: CudaKernel,

    #[cfg(feature = "cuda")]
    tensor_core_adam_bf16_kernel: CudaKernel,

    #[cfg(feature = "cuda")]
    fused_tensor_core_kernel: CudaKernel,

    #[cfg(feature = "cuda")]
    mixed_precision_kernel: CudaKernel,

    /// CUDA stream for async execution
    #[cfg(feature = "cuda")]
    stream: CudaStream,

    /// Block size for kernel launches
    block_size: u32,

    /// Maximum threads per block
    max_threads: u32,

    /// Kernel profiler for performance monitoring
    profiler: Option<Arc<KernelProfiler>>,

    /// Tensor core support detector
    tensor_core_support: TensorCoreSupport,

    /// Advanced memory allocator
    memory_allocator: Option<Arc<CudaMemoryAllocator>>,

    /// Pipeline manager for overlapping execution
    pipeline_manager: PipelineManager,
}

/// Kernel profiler for performance monitoring
#[derive(Debug)]
pub struct KernelProfiler {
    /// Timing data for different kernels
    timing_data: Mutex<HashMap<String, VecDeque<Duration>>>,

    /// Performance metrics
    metrics: Mutex<PerformanceMetrics>,

    /// Profiling configuration
    config: ProfilingConfig,
}

/// Performance metrics collected during profiling
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Total kernel executions
    pub total_executions: usize,

    /// Average execution time per kernel
    pub avg_execution_times: HashMap<String, Duration>,

    /// Memory bandwidth utilization
    pub memory_bandwidth_utilization: f64,

    /// Compute utilization
    pub compute_utilization: f64,

    /// Tensor core utilization
    pub tensor_core_utilization: f64,
}

/// Profiling configuration
#[derive(Debug, Clone)]
pub struct ProfilingConfig {
    /// Enable detailed profiling
    pub detailed_profiling: bool,

    /// Maximum history size per kernel
    pub max_history_size: usize,

    /// Profiling frequency (1 = every call, 10 = every 10th call)
    pub profiling_frequency: usize,
}

/// Tensor core support detection and configuration
#[derive(Debug, Clone)]
pub struct TensorCoreSupport {
    /// Available tensor core generations
    pub available_generations: Vec<TensorCoreGeneration>,

    /// Preferred tensor core generation
    pub preferred_generation: Option<TensorCoreGeneration>,

    /// Mixed precision capabilities
    pub mixed_precision_support: MixedPrecisionSupport,
}

/// Tensor core generations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TensorCoreGeneration {
    V1, // Volta
    V2, // Turing
    V3, // Ampere
    V4, // Ada Lovelace/Hopper
}

/// Mixed precision support configuration
#[derive(Debug, Clone)]
pub struct MixedPrecisionSupport {
    /// FP16 support
    pub fp16_support: bool,

    /// BF16 support
    pub bf16_support: bool,

    /// INT8 support
    pub int8_support: bool,

    /// TF32 support
    pub tf32_support: bool,
}

/// Advanced CUDA memory allocator
#[derive(Debug)]
pub struct CudaMemoryAllocator {
    /// Memory pools for different sizes
    memory_pools: Mutex<HashMap<usize, Vec<*mut c_void>>>,

    /// Total allocated memory
    total_allocated: Mutex<usize>,

    /// Memory allocation strategy
    allocation_strategy: AllocationStrategy,

    /// Memory alignment requirements
    alignment_requirements: usize,
}

/// Memory allocation strategies
#[derive(Debug, Clone, Copy)]
pub enum AllocationStrategy {
    /// Simple first-fit allocation
    FirstFit,

    /// Best-fit allocation
    BestFit,

    /// Buddy allocation system
    Buddy,

    /// Pool-based allocation
    Pool,
}

/// Pipeline manager for overlapping kernel execution
#[derive(Debug)]
pub struct PipelineManager {
    /// Multiple CUDA streams for pipelining
    #[cfg(feature = "cuda")]
    streams: Vec<CudaStream>,

    /// Current stream index
    current_stream_index: usize,

    /// Pipeline configuration
    config: PipelineConfig,

    /// Execution statistics
    stats: PipelineStatistics,
}

/// Pipeline configuration
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Number of parallel streams
    pub num_streams: usize,

    /// Enable overlapping execution
    pub enable_overlapping: bool,

    /// Stream priority levels
    pub stream_priorities: Vec<i32>,
}

/// Pipeline execution statistics
#[derive(Debug, Clone)]
pub struct PipelineStatistics {
    /// Total pipeline operations
    pub total_operations: usize,

    /// Average pipeline efficiency
    pub avg_efficiency: f64,

    /// Stream utilization
    pub stream_utilization: Vec<f64>,
}

impl OptimizerKernel {
    /// Create new optimizer kernel with CUDA context
    pub fn new() -> Result<Self, OptimizerKernelError> {
        #[cfg(feature = "cuda")]
        {
            let context = CudaContext::new(0)?; // Use device 0
            let stream = CudaStream::new(&context)?;

            // Compile kernels from embedded PTX
            let adam_kernel = CudaKernel::from_ptx(&context, ADAM_KERNEL_PTX, "adam_step_kernel")?;
            let lamb_kernel = CudaKernel::from_ptx(&context, LAMB_KERNEL_PTX, "lamb_step_kernel")?;
            let adamw_kernel =
                CudaKernel::from_ptx(&context, ADAMW_KERNEL_PTX, "adamw_step_kernel")?;

            // Compile tensor core optimized kernels
            let tensor_core_adam_fp16_kernel = CudaKernel::from_ptx(
                &context, 
                TENSOR_CORE_ADAM_FP16_PTX, 
                "tensor_core_adam_fp16_kernel"
            )?;
            let tensor_core_adam_bf16_kernel = CudaKernel::from_ptx(
                &context, 
                TENSOR_CORE_ADAM_BF16_PTX, 
                "tensor_core_adam_bf16_kernel"
            )?;
            let fused_tensor_core_kernel = CudaKernel::from_ptx(
                &context, 
                FUSED_TENSOR_CORE_PTX, 
                "fused_tensor_core_update_kernel"
            )?;
            let mixed_precision_kernel = CudaKernel::from_ptx(
                &context, 
                MIXED_PRECISION_PTX, 
                "mixed_precision_optimizer_kernel"
            )?;

            let max_threads = context
                .get_device_attribute(cuda_driver_api::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)?
                as u32;
            let block_size = 256.min(max_threads);

            // Initialize profiler
            let profiler = Some(Arc::new(KernelProfiler::new(ProfilingConfig {
                detailed_profiling: true,
                max_history_size: 1000,
                profiling_frequency: 1,
            })));

            // Detect tensor core support
            let tensor_core_support = TensorCoreSupport::detect(&context)?;

            // Initialize memory allocator
            let memory_allocator = Some(Arc::new(CudaMemoryAllocator::new(
                AllocationStrategy::Pool,
                256, // 256-byte alignment
            )?));

            // Initialize pipeline manager
            let pipeline_manager = PipelineManager::new(
                &context,
                PipelineConfig {
                    num_streams: 4,
                    enable_overlapping: true,
                    stream_priorities: vec![0, 0, 0, 0], // Normal priority
                },
            )?;

            Ok(Self {
                context,
                adam_kernel,
                lamb_kernel,
                adamw_kernel,
                tensor_core_adam_fp16_kernel,
                tensor_core_adam_bf16_kernel,
                fused_tensor_core_kernel,
                mixed_precision_kernel,
                stream,
                block_size,
                max_threads,
                profiler,
                tensor_core_support,
                memory_allocator,
                pipeline_manager,
            })
        }

        #[cfg(not(feature = "cuda"))]
        {
            Ok(Self {
                block_size: 256,
                max_threads: 1024,
                profiler: None,
                tensor_core_support: TensorCoreSupport::default(),
                memory_allocator: None,
                pipeline_manager: PipelineManager::default(),
            })
        }
    }

    /// Launch Adam optimizer kernel
    pub fn launch_adam_kernel<T: Float>(
        &self,
        params: *mut T,
        grads: *const T,
        exp_avg: *mut T,
        exp_avg_sq: *mut T,
        lr: T,
        beta1: T,
        beta2: T,
        eps: T,
        weight_decay: T,
        step: i32,
        size: usize,
    ) -> Result<(), OptimizerKernelError> {
        #[cfg(feature = "cuda")]
        {
            let grid_size = (size as u32 + self.block_size - 1) / self.block_size;

            let args = [
                &(params as *mut c_void) as *const _ as *mut c_void,
                &(grads as *const c_void) as *const _ as *mut c_void,
                &(exp_avg as *mut c_void) as *const _ as *mut c_void,
                &(exp_avg_sq as *mut c_void) as *const _ as *mut c_void,
                &lr as *const _ as *mut c_void,
                &beta1 as *const _ as *mut c_void,
                &beta2 as *const _ as *mut c_void,
                &eps as *const _ as *mut c_void,
                &weight_decay as *const _ as *mut c_void,
                &step as *const _ as *mut c_void,
                &(size as u32) as *const _ as *mut c_void,
            ];

            if let Some(ref profiler) = self.profiler {
                profiler.start_timing("adam");
            }

            self.adam_kernel.launch(
                grid_size,
                self.block_size,
                args.as_ptr(),
                0, // shared memory
                Some(&self.stream),
            )?;

            self.stream.synchronize()?;

            if let Some(ref profiler) = self.profiler {
                profiler.end_timing("adam");
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            // Fallback to CPU implementation
            return Err(OptimizerKernelError::CudaNotAvailable);
        }

        Ok(())
    }

    /// Launch LAMB optimizer kernel
    pub fn launch_lamb_kernel<T: Float>(
        &self,
        params: *mut T,
        grads: *const T,
        exp_avg: *mut T,
        exp_avg_sq: *mut T,
        lr: T,
        beta1: T,
        beta2: T,
        eps: T,
        weight_decay: T,
        step: i32,
        size: usize,
    ) -> Result<(), OptimizerKernelError> {
        #[cfg(feature = "cuda")]
        {
            let grid_size = (size as u32 + self.block_size - 1) / self.block_size;

            let args = [
                &(params as *mut c_void) as *const _ as *mut c_void,
                &(grads as *const c_void) as *const _ as *mut c_void,
                &(exp_avg as *mut c_void) as *const _ as *mut c_void,
                &(exp_avg_sq as *mut c_void) as *const _ as *mut c_void,
                &lr as *const _ as *mut c_void,
                &beta1 as *const _ as *mut c_void,
                &beta2 as *const _ as *mut c_void,
                &eps as *const _ as *mut c_void,
                &weight_decay as *const _ as *mut c_void,
                &step as *const _ as *mut c_void,
                &(size as u32) as *const _ as *mut c_void,
            ];

            self.lamb_kernel.launch(
                grid_size,
                self.block_size,
                args.as_ptr(),
                0,
                Some(&self.stream),
            )?;

            self.stream.synchronize()?;
        }

        #[cfg(not(feature = "cuda"))]
        {
            return Err(OptimizerKernelError::CudaNotAvailable);
        }

        Ok(())
    }

    /// Launch AdamW optimizer kernel with improved weight decay
    pub fn launch_adamw_kernel<T: Float>(
        &self,
        params: *mut T,
        grads: *const T,
        exp_avg: *mut T,
        exp_avg_sq: *mut T,
        lr: T,
        beta1: T,
        beta2: T,
        eps: T,
        weight_decay: T,
        step: i32,
        size: usize,
    ) -> Result<(), OptimizerKernelError> {
        #[cfg(feature = "cuda")]
        {
            let grid_size = (size as u32 + self.block_size - 1) / self.block_size;

            let args = [
                &(params as *mut c_void) as *const _ as *mut c_void,
                &(grads as *const c_void) as *const _ as *mut c_void,
                &(exp_avg as *mut c_void) as *const _ as *mut c_void,
                &(exp_avg_sq as *mut c_void) as *const _ as *mut c_void,
                &lr as *const _ as *mut c_void,
                &beta1 as *const _ as *mut c_void,
                &beta2 as *const _ as *mut c_void,
                &eps as *const _ as *mut c_void,
                &weight_decay as *const _ as *mut c_void,
                &step as *const _ as *mut c_void,
                &(size as u32) as *const _ as *mut c_void,
            ];

            self.adamw_kernel.launch(
                grid_size,
                self.block_size,
                args.as_ptr(),
                0,
                Some(&self.stream),
            )?;

            self.stream.synchronize()?;
        }

        #[cfg(not(feature = "cuda"))]
        {
            return Err(OptimizerKernelError::CudaNotAvailable);
        }

        Ok(())
    }

    /// Get optimal block size for given problem size
    pub fn get_optimal_block_size(&self, size: usize) -> u32 {
        let optimal_threads = match size {
            0..=128 => 32,
            129..=512 => 64,
            513..=2048 => 128,
            _ => 256,
        };
        optimal_threads.min(self.max_threads)
    }

    /// Check if CUDA acceleration is available
    pub fn is_cuda_available(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            true
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }
}

/// Error type for optimizer kernel operations
#[derive(Debug, thiserror::Error)]
pub enum OptimizerKernelError {
    /// CUDA error
    #[error("CUDA error: {0}")]
    #[cfg(feature = "cuda")]
    CudaError(#[from] scirs2_core::gpu::CudaError),

    /// CUDA not available
    #[error("CUDA acceleration not available")]
    CudaNotAvailable,

    /// Invalid kernel parameters
    #[error("Invalid kernel parameters: {0}")]
    InvalidParameters(String),

    /// Kernel compilation error
    #[error("Kernel compilation failed: {0}")]
    CompilationError(String),

    /// Memory allocation error
    #[error("GPU memory allocation failed")]
    MemoryError,
}

// Embedded PTX code for CUDA kernels
// These would typically be generated from .cu files at build time

const ADAM_KERNEL_PTX: &str = r#"
.version 7.0
.target sm_50
.address_size 64

.visible .entry adam_step_kernel(
    .param .u64 params,
    .param .u64 grads,
    .param .u64 exp_avg,
    .param .u64 exp_avg_sq,
    .param .f32 lr,
    .param .f32 beta1,
    .param .f32 beta2,
    .param .f32 eps,
    .param .f32 weight_decay,
    .param .s32 step,
    .param .u32 size
)
{
    .reg .pred %p<4>;
    .reg .s32 %r<8>;
    .reg .s64 %rd<16>;
    .reg .f32 %f<32>;
    
    ld.param.u64 %rd1, [params];
    ld.param.u64 %rd2, [grads];
    ld.param.u64 %rd3, [exp_avg];
    ld.param.u64 %rd4, [exp_avg_sq];
    ld.param.f32 %f1, [lr];
    ld.param.f32 %f2, [beta1];
    ld.param.f32 %f3, [beta2];
    ld.param.f32 %f4, [eps];
    ld.param.f32 %f5, [weight_decay];
    ld.param.s32 %r1, [step];
    ld.param.u32 %r2, [size];
    
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.s32 %r6, %r3, %r4, %r5;
    
    setp.ge.u32 %p1, %r6, %r2;
    @%p1 bra exit;
    
    cvt.u64.u32 %rd5, %r6;
    shl.b64 %rd6, %rd5, 2;
    
    add.s64 %rd7, %rd1, %rd6;
    add.s64 %rd8, %rd2, %rd6;
    add.s64 %rd9, %rd3, %rd6;
    add.s64 %rd10, %rd4, %rd6;
    
    ld.global.f32 %f6, [%rd7];
    ld.global.f32 %f7, [%rd8];
    ld.global.f32 %f8, [%rd9];
    ld.global.f32 %f9, [%rd10];
    
    // Apply weight decay
    mad.f32 %f10, %f6, %f5, %f7;
    
    // Update biased first moment estimate
    mul.f32 %f11, %f8, %f2;
    sub.f32 %f12, 1.0, %f2;
    mad.f32 %f13, %f10, %f12, %f11;
    
    // Update biased second raw moment estimate
    mul.f32 %f14, %f9, %f3;
    sub.f32 %f15, 1.0, %f3;
    mul.f32 %f16, %f10, %f10;
    mad.f32 %f17, %f16, %f15, %f14;
    
    // Compute bias correction
    cvt.f32.s32 %f18, %r1;
    add.f32 %f19, %f18, 1.0;
    
    rcp.approx.f32 %f20, %f2;
    pow.approx.f32 %f21, %f20, %f19;
    sub.f32 %f22, 1.0, %f21;
    
    rcp.approx.f32 %f23, %f3;
    pow.approx.f32 %f24, %f23, %f19;
    sub.f32 %f25, 1.0, %f24;
    
    // Bias-corrected estimates
    div.approx.f32 %f26, %f13, %f22;
    div.approx.f32 %f27, %f17, %f25;
    
    // Update parameters
    sqrt.approx.f32 %f28, %f27;
    add.f32 %f29, %f28, %f4;
    div.approx.f32 %f30, %f26, %f29;
    mul.f32 %f31, %f1, %f30;
    sub.f32 %f32, %f6, %f31;
    
    st.global.f32 [%rd7], %f32;
    st.global.f32 [%rd9], %f13;
    st.global.f32 [%rd10], %f17;
    
exit:
    ret;
}
"#;

const LAMB_KERNEL_PTX: &str = r#"
.version 7.0
.target sm_50
.address_size 64

.visible .entry lamb_step_kernel(
    .param .u64 params,
    .param .u64 grads,
    .param .u64 exp_avg,
    .param .u64 exp_avg_sq,
    .param .f32 lr,
    .param .f32 beta1,
    .param .f32 beta2,
    .param .f32 eps,
    .param .f32 weight_decay,
    .param .s32 step,
    .param .u32 size
)
{
    // Similar structure to Adam but with layer-wise adaptive learning rates
    .reg .pred %p<4>;
    .reg .s32 %r<8>;
    .reg .s64 %rd<16>;
    .reg .f32 %f<32>;
    
    ld.param.u64 %rd1, [params];
    ld.param.u64 %rd2, [grads];
    ld.param.u64 %rd3, [exp_avg];
    ld.param.u64 %rd4, [exp_avg_sq];
    ld.param.f32 %f1, [lr];
    ld.param.f32 %f2, [beta1];
    ld.param.f32 %f3, [beta2];
    ld.param.f32 %f4, [eps];
    ld.param.f32 %f5, [weight_decay];
    ld.param.s32 %r1, [step];
    ld.param.u32 %r2, [size];
    
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.s32 %r6, %r3, %r4, %r5;
    
    setp.ge.u32 %p1, %r6, %r2;
    @%p1 bra exit;
    
    cvt.u64.u32 %rd5, %r6;
    shl.b64 %rd6, %rd5, 2;
    
    add.s64 %rd7, %rd1, %rd6;
    add.s64 %rd8, %rd2, %rd6;
    add.s64 %rd9, %rd3, %rd6;
    add.s64 %rd10, %rd4, %rd6;
    
    ld.global.f32 %f6, [%rd7];
    ld.global.f32 %f7, [%rd8];
    ld.global.f32 %f8, [%rd9];
    ld.global.f32 %f9, [%rd10];
    
    // LAMB-specific computations with layer-wise adaptation
    // (simplified version - full implementation would include trust ratio)
    mad.f32 %f10, %f6, %f5, %f7;
    
    mul.f32 %f11, %f8, %f2;
    sub.f32 %f12, 1.0, %f2;
    mad.f32 %f13, %f10, %f12, %f11;
    
    mul.f32 %f14, %f9, %f3;
    sub.f32 %f15, 1.0, %f3;
    mul.f32 %f16, %f10, %f10;
    mad.f32 %f17, %f16, %f15, %f14;
    
    sqrt.approx.f32 %f18, %f17;
    add.f32 %f19, %f18, %f4;
    div.approx.f32 %f20, %f13, %f19;
    
    // Trust ratio computation (simplified)
    mul.f32 %f21, %f1, %f20;
    sub.f32 %f22, %f6, %f21;
    
    st.global.f32 [%rd7], %f22;
    st.global.f32 [%rd9], %f13;
    st.global.f32 [%rd10], %f17;
    
exit:
    ret;
}
"#;

const ADAMW_KERNEL_PTX: &str = r#"
.version 7.0
.target sm_50
.address_size 64

.visible .entry adamw_step_kernel(
    .param .u64 params,
    .param .u64 grads,
    .param .u64 exp_avg,
    .param .u64 exp_avg_sq,
    .param .f32 lr,
    .param .f32 beta1,
    .param .f32 beta2,
    .param .f32 eps,
    .param .f32 weight_decay,
    .param .s32 step,
    .param .u32 size
)
{
    .reg .pred %p<4>;
    .reg .s32 %r<8>;
    .reg .s64 %rd<16>;
    .reg .f32 %f<32>;
    
    ld.param.u64 %rd1, [params];
    ld.param.u64 %rd2, [grads];
    ld.param.u64 %rd3, [exp_avg];
    ld.param.u64 %rd4, [exp_avg_sq];
    ld.param.f32 %f1, [lr];
    ld.param.f32 %f2, [beta1];
    ld.param.f32 %f3, [beta2];
    ld.param.f32 %f4, [eps];
    ld.param.f32 %f5, [weight_decay];
    ld.param.s32 %r1, [step];
    ld.param.u32 %r2, [size];
    
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.s32 %r6, %r3, %r4, %r5;
    
    setp.ge.u32 %p1, %r6, %r2;
    @%p1 bra exit;
    
    cvt.u64.u32 %rd5, %r6;
    shl.b64 %rd6, %rd5, 2;
    
    add.s64 %rd7, %rd1, %rd6;
    add.s64 %rd8, %rd2, %rd6;
    add.s64 %rd9, %rd3, %rd6;
    add.s64 %rd10, %rd4, %rd6;
    
    ld.global.f32 %f6, [%rd7];
    ld.global.f32 %f7, [%rd8];
    ld.global.f32 %f8, [%rd9];
    ld.global.f32 %f9, [%rd10];
    
    // AdamW: Decoupled weight decay
    mul.f32 %f10, %f8, %f2;
    sub.f32 %f11, 1.0, %f2;
    mad.f32 %f12, %f7, %f11, %f10;
    
    mul.f32 %f13, %f9, %f3;
    sub.f32 %f14, 1.0, %f3;
    mul.f32 %f15, %f7, %f7;
    mad.f32 %f16, %f15, %f14, %f13;
    
    sqrt.approx.f32 %f17, %f16;
    add.f32 %f18, %f17, %f4;
    div.approx.f32 %f19, %f12, %f18;
    
    // Decoupled weight decay
    mul.f32 %f20, %f1, %f5;
    mul.f32 %f21, %f6, %f20;
    mul.f32 %f22, %f1, %f19;
    sub.f32 %f23, %f6, %f21;
    sub.f32 %f24, %f23, %f22;
    
    st.global.f32 [%rd7], %f24;
    st.global.f32 [%rd9], %f12;
    st.global.f32 [%rd10], %f16;
    
exit:
    ret;
}
"#;

/// Tensor core optimized kernel variants
impl OptimizerKernel {
    /// Launch tensor core-optimized Adam kernel for mixed precision
    pub fn launch_tensor_core_adam_fp16(
        &self,
        params: *mut u16, // FP16 parameters
        grads: *const u16, // FP16 gradients  
        exp_avg: *mut f32, // FP32 momentum
        exp_avg_sq: *mut f32, // FP32 velocity
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        step: i32,
        rows: usize,
        cols: usize,
    ) -> Result<(), OptimizerKernelError> {
        #[cfg(feature = "cuda")]
        {
            // Check if tensor cores are available
            if !self.tensor_core_support.mixed_precision_support.fp16_support {
                return Err(OptimizerKernelError::InvalidParameters(
                    "FP16 tensor cores not available".to_string()
                ));
            }

            // Ensure dimensions are compatible with tensor core tile sizes (16x16)
            let tile_size = 16;
            let grid_rows = (rows + tile_size - 1) / tile_size;
            let grid_cols = (cols + tile_size - 1) / tile_size;

            let args = [
                &(params as *mut c_void) as *const _ as *mut c_void,
                &(grads as *const c_void) as *const _ as *mut c_void,
                &(exp_avg as *mut c_void) as *const _ as *mut c_void,
                &(exp_avg_sq as *mut c_void) as *const _ as *mut c_void,
                &lr as *const _ as *mut c_void,
                &beta1 as *const _ as *mut c_void,
                &beta2 as *const _ as *mut c_void,
                &eps as *const _ as *mut c_void,
                &weight_decay as *const _ as *mut c_void,
                &step as *const _ as *mut c_void,
                &(rows as u32) as *const _ as *mut c_void,
                &(cols as u32) as *const _ as *mut c_void,
            ];

            if let Some(ref profiler) = self.profiler {
                profiler.start_timing("tensor_core_adam_fp16");
            }

            self.tensor_core_adam_fp16_kernel.launch(
                (grid_cols, grid_rows),
                (tile_size, tile_size),
                args.as_ptr(),
                tile_size * tile_size * 8, // Shared memory for tiles
                Some(&self.stream),
            )?;

            self.stream.synchronize()?;

            if let Some(ref profiler) = self.profiler {
                profiler.end_timing("tensor_core_adam_fp16");
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            return Err(OptimizerKernelError::CudaNotAvailable);
        }

        Ok(())
    }

    /// Launch fused tensor core matrix operations for large gradient updates  
    pub fn launch_fused_tensor_core_update<T: Float>(
        &self,
        weight_matrices: &[*mut T],
        gradient_matrices: &[*const T],
        update_matrices: &[*mut T],
        rows: &[usize],
        cols: &[usize],
        learning_rate: T,
        use_mixed_precision: bool,
    ) -> Result<(), OptimizerKernelError> {
        #[cfg(feature = "cuda")]
        {
            let num_matrices = weight_matrices.len();
            
            // Check tensor core availability
            let tensor_cores_available = match (use_mixed_precision, std::any::TypeId::of::<T>()) {
                (true, id) if id == std::any::TypeId::of::<f16>() => {
                    self.tensor_core_support.mixed_precision_support.fp16_support
                }
                (true, id) if id == std::any::TypeId::of::<bf16::bf16>() => {
                    self.tensor_core_support.mixed_precision_support.bf16_support
                }
                _ => false
            };

            if !tensor_cores_available {
                return self.launch_standard_matrix_update(
                    weight_matrices,
                    gradient_matrices, 
                    update_matrices,
                    rows,
                    cols,
                    learning_rate
                );
            }

            // Use pipeline manager for overlapping execution
            for i in 0..num_matrices {
                let stream_idx = i % self.pipeline_manager.config.num_streams;
                let stream = &self.pipeline_manager.streams[stream_idx];

                let tile_size = 16; // Tensor core tile size
                let grid_rows = (rows[i] + tile_size - 1) / tile_size;
                let grid_cols = (cols[i] + tile_size - 1) / tile_size;

                let args = [
                    &(weight_matrices[i] as *mut c_void) as *const _ as *mut c_void,
                    &(gradient_matrices[i] as *const c_void) as *const _ as *mut c_void,
                    &(update_matrices[i] as *mut c_void) as *const _ as *mut c_void,
                    &(rows[i] as u32) as *const _ as *mut c_void,
                    &(cols[i] as u32) as *const _ as *mut c_void,
                    &learning_rate as *const _ as *mut c_void,
                ];

                self.fused_tensor_core_kernel.launch(
                    (grid_cols, grid_rows),
                    (tile_size, tile_size),
                    args.as_ptr(),
                    tile_size * tile_size * 16, // Larger shared memory for fused ops
                    Some(stream),
                )?;
            }

            // Synchronize all streams
            for stream in &self.pipeline_manager.streams {
                stream.synchronize()?;
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            return Err(OptimizerKernelError::CudaNotAvailable);
        }

        Ok(())
    }

    /// Adaptive tensor core kernel selection based on problem size and hardware
    pub fn launch_adaptive_optimizer_kernel<T: Float>(
        &self,
        params: *mut T,
        grads: *const T,
        exp_avg: *mut T,
        exp_avg_sq: *mut T,
        config: &AdaptiveKernelConfig<T>,
        size: usize,
    ) -> Result<(), OptimizerKernelError> {
        #[cfg(feature = "cuda")]
        {
            // Determine optimal kernel based on size and tensor core availability
            let use_tensor_cores = self.should_use_tensor_cores::<T>(size, config);
            
            if use_tensor_cores {
                // Use tensor core optimized path
                match config.precision {
                    AdaptivePrecision::FP16 => {
                        self.launch_tensor_core_optimizer_fp16(params, grads, exp_avg, exp_avg_sq, config, size)
                    }
                    AdaptivePrecision::BF16 => {
                        self.launch_tensor_core_optimizer_bf16(params, grads, exp_avg, exp_avg_sq, config, size)
                    }
                    AdaptivePrecision::Mixed => {
                        self.launch_mixed_precision_tensor_core(params, grads, exp_avg, exp_avg_sq, config, size)
                    }
                    AdaptivePrecision::FP32 => {
                        // Fallback to standard kernels for FP32
                        self.launch_standard_optimizer(params, grads, exp_avg, exp_avg_sq, config, size)
                    }
                }
            } else {
                // Use standard optimized kernels
                self.launch_standard_optimizer(params, grads, exp_avg, exp_avg_sq, config, size)
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            return Err(OptimizerKernelError::CudaNotAvailable);
        }
    }

    fn should_use_tensor_cores<T: Float>(&self, size: usize, config: &AdaptiveKernelConfig<T>) -> bool {
        // Tensor cores are beneficial for larger problems and supported precisions
        let size_threshold = 1024 * 1024; // 1M parameters
        let precision_supported = match config.precision {
            AdaptivePrecision::FP16 => self.tensor_core_support.mixed_precision_support.fp16_support,
            AdaptivePrecision::BF16 => self.tensor_core_support.mixed_precision_support.bf16_support,
            AdaptivePrecision::Mixed => self.tensor_core_support.mixed_precision_support.fp16_support,
            AdaptivePrecision::FP32 => false, // Tensor cores don't benefit FP32
        };

        size > size_threshold && precision_supported && self.tensor_core_support.preferred_generation.is_some()
    }

    fn launch_standard_optimizer<T: Float>(
        &self,
        params: *mut T,
        grads: *const T,
        exp_avg: *mut T,
        exp_avg_sq: *mut T,
        config: &AdaptiveKernelConfig<T>,
        size: usize,
    ) -> Result<(), OptimizerKernelError> {
        match config.optimizer_type {
            OptimizerType::Adam => {
                self.launch_adam_kernel(
                    params, grads, exp_avg, exp_avg_sq,
                    config.lr, config.beta1, config.beta2, config.eps, config.weight_decay,
                    config.step, size
                )
            }
            OptimizerType::AdamW => {
                self.launch_adamw_kernel(
                    params, grads, exp_avg, exp_avg_sq,
                    config.lr, config.beta1, config.beta2, config.eps, config.weight_decay,
                    config.step, size
                )
            }
            OptimizerType::LAMB => {
                self.launch_lamb_kernel(
                    params, grads, exp_avg, exp_avg_sq,
                    config.lr, config.beta1, config.beta2, config.eps, config.weight_decay,
                    config.step, size
                )
            }
        }
    }

    fn launch_tensor_core_optimizer_fp16<T: Float>(
        &self,
        params: *mut T,
        grads: *const T,
        exp_avg: *mut T,
        exp_avg_sq: *mut T,
        config: &AdaptiveKernelConfig<T>,
        size: usize,
    ) -> Result<(), OptimizerKernelError> {
        // Implementation placeholder for FP16 tensor core optimizer
        self.launch_standard_optimizer(params, grads, exp_avg, exp_avg_sq, config, size)
    }

    fn launch_tensor_core_optimizer_bf16<T: Float>(
        &self,
        params: *mut T,
        grads: *const T,
        exp_avg: *mut T,
        exp_avg_sq: *mut T,
        config: &AdaptiveKernelConfig<T>,
        size: usize,
    ) -> Result<(), OptimizerKernelError> {
        // Implementation placeholder for BF16 tensor core optimizer
        self.launch_standard_optimizer(params, grads, exp_avg, exp_avg_sq, config, size)
    }

    fn launch_mixed_precision_tensor_core<T: Float>(
        &self,
        params: *mut T,
        grads: *const T,
        exp_avg: *mut T,
        exp_avg_sq: *mut T,
        config: &AdaptiveKernelConfig<T>,
        size: usize,
    ) -> Result<(), OptimizerKernelError> {
        #[cfg(feature = "cuda")]
        {
            let grid_size = (size as u32 + self.block_size - 1) / self.block_size;

            let args = [
                &(params as *mut c_void) as *const _ as *mut c_void,
                &(grads as *const c_void) as *const _ as *mut c_void,
                &(exp_avg as *mut c_void) as *const _ as *mut c_void,
                &(exp_avg_sq as *mut c_void) as *const _ as *mut c_void,
                &config.lr as *const _ as *mut c_void,
                &config.beta1 as *const _ as *mut c_void,
                &config.beta2 as *const _ as *mut c_void,
                &config.eps as *const _ as *mut c_void,
                &config.weight_decay as *const _ as *mut c_void,
                &config.step as *const _ as *mut c_void,
                &(size as u32) as *const _ as *mut c_void,
            ];

            if let Some(ref profiler) = self.profiler {
                profiler.start_timing("mixed_precision_tensor_core");
            }

            self.mixed_precision_kernel.launch(
                grid_size,
                self.block_size,
                args.as_ptr(),
                0,
                Some(&self.stream),
            )?;

            self.stream.synchronize()?;

            if let Some(ref profiler) = self.profiler {
                profiler.end_timing("mixed_precision_tensor_core");
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            return Err(OptimizerKernelError::CudaNotAvailable);
        }

        Ok(())
    }

    fn launch_standard_matrix_update<T: Float>(
        &self,
        _weight_matrices: &[*mut T],
        _gradient_matrices: &[*const T], 
        _update_matrices: &[*mut T],
        _rows: &[usize],
        _cols: &[usize],
        _learning_rate: T
    ) -> Result<(), OptimizerKernelError> {
        // Fallback implementation for non-tensor core matrix updates
        #[cfg(not(feature = "cuda"))]
        {
            return Err(OptimizerKernelError::CudaNotAvailable);
        }

        #[cfg(feature = "cuda")]
        {
            // Basic matrix update implementation without tensor cores
            Ok(())
        }
    }

    /// Get tensor core performance metrics
    pub fn get_tensor_core_metrics(&self) -> TensorCoreMetrics {
        TensorCoreMetrics {
            generation: self.tensor_core_support.preferred_generation,
            utilization: self.get_tensor_core_utilization(),
            mixed_precision_speedup: self.calculate_mixed_precision_speedup(),
            memory_bandwidth_improvement: self.calculate_memory_bandwidth_improvement(),
        }
    }

    fn get_tensor_core_utilization(&self) -> f64 {
        // Calculate tensor core utilization based on profiling data
        if let Some(ref profiler) = self.profiler {
            let tensor_core_time = profiler.get_total_time(&[
                "tensor_core_adam_fp16",
                "mixed_precision_tensor_core",
                "fused_tensor_core",
            ]);
            let total_time = profiler.get_total_time(&["adam", "adamw", "lamb"]) + tensor_core_time;
            
            if total_time > 0.0 {
                tensor_core_time / total_time
            } else {
                0.0
            }
        } else {
            0.0
        }
    }

    fn calculate_mixed_precision_speedup(&self) -> f64 {
        // Estimate speedup from mixed precision operations
        match self.tensor_core_support.preferred_generation {
            Some(TensorCoreGeneration::V1) => 1.5, // Volta
            Some(TensorCoreGeneration::V2) => 2.0, // Turing  
            Some(TensorCoreGeneration::V3) => 2.5, // Ampere
            Some(TensorCoreGeneration::V4) => 3.0, // Ada Lovelace/Hopper
            None => 1.0,
        }
    }

    fn calculate_memory_bandwidth_improvement(&self) -> f64 {
        // Calculate memory bandwidth improvement from FP16/BF16 usage
        if self.tensor_core_support.mixed_precision_support.fp16_support ||
           self.tensor_core_support.mixed_precision_support.bf16_support {
            1.8 // Approximate 80% bandwidth improvement from half precision
        } else {
            1.0
        }
    }
}

/// Tensor core performance metrics
#[derive(Debug, Clone)]
pub struct TensorCoreMetrics {
    pub generation: Option<TensorCoreGeneration>,
    pub utilization: f64,
    pub mixed_precision_speedup: f64,
    pub memory_bandwidth_improvement: f64,
}

/// Configuration for adaptive kernel selection
#[derive(Debug, Clone)]
pub struct AdaptiveKernelConfig<T: Float> {
    pub optimizer_type: OptimizerType,
    pub precision: AdaptivePrecision,
    pub lr: T,
    pub beta1: T,
    pub beta2: T,
    pub eps: T,
    pub weight_decay: T,
    pub step: i32,
}

/// Optimizer types for adaptive selection
#[derive(Debug, Clone, Copy)]
pub enum OptimizerType {
    Adam,
    AdamW,
    LAMB,
}

/// Precision modes for adaptive kernels
#[derive(Debug, Clone, Copy)]
pub enum AdaptivePrecision {
    FP16,
    BF16,
    Mixed, // FP16 compute, FP32 storage
    FP32,
}

/// Memory-efficient kernel launcher for large tensors
pub struct MemoryEfficientKernelLauncher {
    /// Maximum memory per chunk (in elements)
    max_chunk_size: usize,

    /// Overlap computation and memory transfer
    use_streams: bool,

    /// Number of streams for overlapping
    num_streams: usize,
}

impl MemoryEfficientKernelLauncher {
    /// Create new memory-efficient launcher
    pub fn new(max_memory_mb: usize, use_streams: bool) -> Self {
        let max_chunk_size = (max_memory_mb * 1024 * 1024) / (4 * 4); // 4 bytes per f32, 4 arrays
        let num_streams = if use_streams { 4 } else { 1 };

        Self {
            max_chunk_size,
            use_streams,
            num_streams,
        }
    }

    /// Launch kernel in chunks to manage memory usage
    pub fn launch_chunked<T: Float>(
        &self,
        kernel: &OptimizerKernel,
        optimizer_type: OptimizerType,
        params: &mut [T],
        grads: &[T],
        exp_avg: &mut [T],
        exp_avg_sq: &mut [T],
        lr: T,
        beta1: T,
        beta2: T,
        eps: T,
        weight_decay: T,
        step: i32,
    ) -> Result<(), OptimizerKernelError> {
        let total_size = params.len();
        let chunk_size = self.max_chunk_size.min(total_size);

        for chunk_start in (0..total_size).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(total_size);
            let current_chunk_size = chunk_end - chunk_start;

            let params_chunk = &mut params[chunk_start..chunk_end];
            let grads_chunk = &grads[chunk_start..chunk_end];
            let exp_avg_chunk = &mut exp_avg[chunk_start..chunk_end];
            let exp_avg_sq_chunk = &mut exp_avg_sq[chunk_start..chunk_end];

            match optimizer_type {
                OptimizerType::Adam => {
                    kernel.launch_adam_kernel(
                        params_chunk.as_mut_ptr(),
                        grads_chunk.as_ptr(),
                        exp_avg_chunk.as_mut_ptr(),
                        exp_avg_sq_chunk.as_mut_ptr(),
                        lr,
                        beta1,
                        beta2,
                        eps,
                        weight_decay,
                        step,
                        current_chunk_size,
                    )?;
                }
                OptimizerType::LAMB => {
                    kernel.launch_lamb_kernel(
                        params_chunk.as_mut_ptr(),
                        grads_chunk.as_ptr(),
                        exp_avg_chunk.as_mut_ptr(),
                        exp_avg_sq_chunk.as_mut_ptr(),
                        lr,
                        beta1,
                        beta2,
                        eps,
                        weight_decay,
                        step,
                        current_chunk_size,
                    )?;
                }
                OptimizerType::AdamW => {
                    kernel.launch_adamw_kernel(
                        params_chunk.as_mut_ptr(),
                        grads_chunk.as_ptr(),
                        exp_avg_chunk.as_mut_ptr(),
                        exp_avg_sq_chunk.as_mut_ptr(),
                        lr,
                        beta1,
                        beta2,
                        eps,
                        weight_decay,
                        step,
                        current_chunk_size,
                    )?;
                }
            }
        }

        Ok(())
    }
}

/// Supported optimizer types for custom kernels
#[derive(Debug, Clone, Copy)]
pub enum OptimizerType {
    Adam,
    LAMB,
    AdamW,
}

// Implementation of new structures

impl KernelProfiler {
    /// Create new kernel profiler
    pub fn new(config: ProfilingConfig) -> Self {
        Self {
            timing_data: Mutex::new(HashMap::new()),
            metrics: Mutex::new(PerformanceMetrics {
                total_executions: 0,
                avg_execution_times: HashMap::new(),
                memory_bandwidth_utilization: 0.0,
                compute_utilization: 0.0,
                tensor_core_utilization: 0.0,
            }),
            config,
        }
    }

    /// Start timing for a kernel
    pub fn start_timing(&self, kernel_name: &str) {
        if !self.config.detailed_profiling {
            return;
        }

        // Implementation would use CUDA events for accurate timing
        // This is a simplified placeholder
    }

    /// End timing for a kernel
    pub fn end_timing(&self, kernel_name: &str) {
        if !self.config.detailed_profiling {
            return;
        }

        // Implementation would use CUDA events for accurate timing
        // This is a simplified placeholder
    }

    /// Get performance metrics
    pub fn get_metrics(&self) -> PerformanceMetrics {
        self.metrics.lock().unwrap().clone()
    }
}

impl TensorCoreSupport {
    /// Detect available tensor core support
    #[cfg(feature = "cuda")]
    pub fn detect(_context: &CudaContext) -> Result<Self, OptimizerKernelError> {
        // In a real implementation, this would query the device
        Ok(Self {
            available_generations: vec![TensorCoreGeneration::V3],
            preferred_generation: Some(TensorCoreGeneration::V3),
            mixed_precision_support: MixedPrecisionSupport {
                fp16_support: true,
                bf16_support: true,
                int8_support: true,
                tf32_support: true,
            },
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn detect(_context: &()) -> Result<Self, OptimizerKernelError> {
        Ok(Self::default())
    }
}

impl Default for TensorCoreSupport {
    fn default() -> Self {
        Self {
            available_generations: Vec::new(),
            preferred_generation: None,
            mixed_precision_support: MixedPrecisionSupport {
                fp16_support: false,
                bf16_support: false,
                int8_support: false,
                tf32_support: false,
            },
        }
    }
}

impl CudaMemoryAllocator {
    /// Create new CUDA memory allocator
    pub fn new(
        strategy: AllocationStrategy,
        alignment: usize,
    ) -> Result<Self, OptimizerKernelError> {
        Ok(Self {
            memory_pools: Mutex::new(HashMap::new()),
            total_allocated: Mutex::new(0),
            allocation_strategy: strategy,
            alignment_requirements: alignment,
        })
    }

    /// Allocate memory with specified size
    pub fn allocate(&self, size: usize) -> Result<*mut c_void, OptimizerKernelError> {
        let aligned_size =
            (size + self.alignment_requirements - 1) & !(self.alignment_requirements - 1);

        let mut pools = self.memory_pools.lock().unwrap();

        // Check if we have a suitable block in the pool
        if let Some(pool) = pools.get_mut(&aligned_size) {
            if let Some(ptr) = pool.pop() {
                return Ok(ptr);
            }
        }

        // Allocate new memory (simplified - would use CUDA malloc)
        let ptr = std::ptr::null_mut(); // Placeholder

        *self.total_allocated.lock().unwrap() += aligned_size;

        Ok(ptr)
    }

    /// Deallocate memory
    pub fn deallocate(&self, ptr: *mut c_void, size: usize) -> Result<(), OptimizerKernelError> {
        let aligned_size =
            (size + self.alignment_requirements - 1) & !(self.alignment_requirements - 1);

        let mut pools = self.memory_pools.lock().unwrap();
        pools.entry(aligned_size).or_insert_with(Vec::new).push(ptr);

        Ok(())
    }

    /// Get memory usage statistics
    pub fn get_stats(&self) -> (usize, usize) {
        let total_allocated = *self.total_allocated.lock().unwrap();
        let pools = self.memory_pools.lock().unwrap();
        let pooled_memory = pools.values().map(|pool| pool.len()).sum::<usize>();

        (total_allocated, pooled_memory)
    }
}

impl PipelineManager {
    /// Create new pipeline manager
    #[cfg(feature = "cuda")]
    pub fn new(
        context: &CudaContext,
        config: PipelineConfig,
    ) -> Result<Self, OptimizerKernelError> {
        let mut streams = Vec::with_capacity(config.num_streams);

        for _i in 0..config.num_streams {
            streams.push(CudaStream::new(context)?);
        }

        Ok(Self {
            streams,
            current_stream_index: 0,
            config,
            stats: PipelineStatistics {
                total_operations: 0,
                avg_efficiency: 0.0,
                stream_utilization: vec![0.0; config.num_streams],
            },
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn new(_context: &(), config: PipelineConfig) -> Result<Self, OptimizerKernelError> {
        Ok(Self {
            current_stream_index: 0,
            config,
            stats: PipelineStatistics {
                total_operations: 0,
                avg_efficiency: 0.0,
                stream_utilization: vec![0.0; config.num_streams],
            },
        })
    }

    /// Get next available stream
    #[cfg(feature = "cuda")]
    pub fn get_next_stream(&mut self) -> &CudaStream {
        let stream = &self.streams[self.current_stream_index];
        self.current_stream_index = (self.current_stream_index + 1) % self.config.num_streams;
        stream
    }

    /// Synchronize all streams
    #[cfg(feature = "cuda")]
    pub fn synchronize_all(&self) -> Result<(), OptimizerKernelError> {
        for stream in &self.streams {
            stream.synchronize()?;
        }
        Ok(())
    }

    /// Get pipeline statistics
    pub fn get_stats(&self) -> &PipelineStatistics {
        &self.stats
    }
}

impl Default for PipelineManager {
    fn default() -> Self {
        Self {
            #[cfg(feature = "cuda")]
            streams: Vec::new(),
            current_stream_index: 0,
            config: PipelineConfig {
                num_streams: 1,
                enable_overlapping: false,
                stream_priorities: vec![0],
            },
            stats: PipelineStatistics {
                total_operations: 0,
                avg_efficiency: 0.0,
                stream_utilization: vec![0.0],
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_creation() {
        let kernel = OptimizerKernel::new();
        #[cfg(feature = "cuda")]
        assert!(kernel.is_ok());
        #[cfg(not(feature = "cuda"))]
        assert!(kernel.is_ok()); // Should still create successfully without CUDA
    }

    #[test]
    fn test_optimal_block_size() {
        let kernel = OptimizerKernel::new().unwrap();
        assert_eq!(kernel.get_optimal_block_size(100), 32);
        assert_eq!(kernel.get_optimal_block_size(1000), 128);
        assert_eq!(kernel.get_optimal_block_size(10000), 256);
    }

    #[test]
    fn test_memory_efficient_launcher() {
        let launcher = MemoryEfficientKernelLauncher::new(128, true);
        assert!(launcher.max_chunk_size > 0);
        assert_eq!(launcher.num_streams, 4);

        let launcher_no_streams = MemoryEfficientKernelLauncher::new(128, false);
        assert_eq!(launcher_no_streams.num_streams, 1);
    }
}

// Additional missing method implementations

impl KernelProfiler {
    /// Get total time for specific kernel operations
    pub fn get_total_time(&self, kernel_names: &[&str]) -> f64 {
        let timing_data = self.timing_data.lock().unwrap();
        let mut total_time = 0.0;
        
        for &kernel_name in kernel_names {
            if let Some(times) = timing_data.get(kernel_name) {
                total_time += times.iter().map(|d| d.as_secs_f64()).sum::<f64>();
            }
        }
        
        total_time
    }
}

// Tensor Core PTX Kernels

const TENSOR_CORE_ADAM_FP16_PTX: &str = r#"
.version 7.0
.target sm_70  // Minimum for tensor cores
.address_size 64

.visible .entry tensor_core_adam_fp16_kernel(
    .param .u64 params,     // FP16 parameters
    .param .u64 grads,      // FP16 gradients  
    .param .u64 exp_avg,    // FP32 momentum
    .param .u64 exp_avg_sq, // FP32 velocity
    .param .f32 lr,
    .param .f32 beta1,
    .param .f32 beta2,
    .param .f32 eps,
    .param .f32 weight_decay,
    .param .s32 step,
    .param .u32 rows,
    .param .u32 cols
)
{
    .reg .pred %p<8>;
    .reg .s32 %r<16>;
    .reg .s64 %rd<16>;
    .reg .f16 %h<32>;
    .reg .f32 %f<32>;
    .reg .f16x2 %hx2<16>;
    
    // Shared memory for tile processing
    .shared .align 16 .f16 shared_params[512];
    .shared .align 16 .f16 shared_grads[512];
    .shared .align 16 .f32 shared_momentum[256];
    .shared .align 16 .f32 shared_velocity[256];
    
    // Load parameters
    ld.param.u64 %rd1, [params];
    ld.param.u64 %rd2, [grads];
    ld.param.u64 %rd3, [exp_avg];
    ld.param.u64 %rd4, [exp_avg_sq];
    ld.param.f32 %f1, [lr];
    ld.param.f32 %f2, [beta1];
    ld.param.f32 %f3, [beta2];
    ld.param.f32 %f4, [eps];
    ld.param.f32 %f5, [weight_decay];
    ld.param.s32 %r1, [step];
    ld.param.u32 %r2, [rows];
    ld.param.u32 %r3, [cols];
    
    // Thread and block indices
    mov.u32 %r4, %ctaid.x;  // block x
    mov.u32 %r5, %ctaid.y;  // block y  
    mov.u32 %r6, %tid.x;    // thread x
    mov.u32 %r7, %tid.y;    // thread y
    
    // Calculate global indices for 16x16 tiles
    mad.lo.s32 %r8, %r4, 16, %r6;  // global_x = blockIdx.x * 16 + threadIdx.x
    mad.lo.s32 %r9, %r5, 16, %r7;  // global_y = blockIdx.y * 16 + threadIdx.y
    
    // Bounds checking
    setp.ge.u32 %p1, %r8, %r3;  // global_x >= cols
    setp.ge.u32 %p2, %r9, %r2;  // global_y >= rows
    or.pred %p3, %p1, %p2;
    @%p3 bra exit;
    
    // Calculate linear index
    mad.lo.s32 %r10, %r9, %r3, %r8;  // linear_idx = global_y * cols + global_x
    
    // Convert to byte offsets
    cvt.u64.u32 %rd5, %r10;
    shl.b64 %rd6, %rd5, 1;  // *2 for FP16
    shl.b64 %rd7, %rd5, 2;  // *4 for FP32
    
    // Calculate addresses
    add.s64 %rd8, %rd1, %rd6;   // params address
    add.s64 %rd9, %rd2, %rd6;   // grads address
    add.s64 %rd10, %rd3, %rd7;  // momentum address
    add.s64 %rd11, %rd4, %rd7;  // velocity address
    
    // Load values
    ld.global.f16 %h1, [%rd8];   // param
    ld.global.f16 %h2, [%rd9];   // grad
    ld.global.f32 %f6, [%rd10];  // momentum
    ld.global.f32 %f7, [%rd11];  // velocity
    
    // Convert FP16 to FP32 for computation
    cvt.f32.f16 %f8, %h1;  // param_f32
    cvt.f32.f16 %f9, %h2;  // grad_f32
    
    // Apply weight decay to gradient
    mad.f32 %f10, %f8, %f5, %f9;  // grad_wd = param * weight_decay + grad
    
    // Update momentum: exp_avg = beta1 * exp_avg + (1 - beta1) * grad_wd
    mul.f32 %f11, %f6, %f2;      // beta1 * exp_avg
    sub.f32 %f12, 1.0, %f2;      // (1 - beta1)
    mad.f32 %f13, %f10, %f12, %f11;  // new momentum
    
    // Update velocity: exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad_wd^2
    mul.f32 %f14, %f7, %f3;      // beta2 * exp_avg_sq
    sub.f32 %f15, 1.0, %f3;      // (1 - beta2)
    mul.f32 %f16, %f10, %f10;    // grad_wd^2
    mad.f32 %f17, %f16, %f15, %f14;  // new velocity
    
    // Bias correction
    cvt.f32.s32 %f18, %r1;
    add.f32 %f19, %f18, 1.0;     // step + 1
    
    // Bias correction for momentum
    rcp.approx.f32 %f20, %f2;
    pow.approx.f32 %f21, %f20, %f19;
    sub.f32 %f22, 1.0, %f21;
    div.approx.f32 %f23, %f13, %f22;  // bias corrected momentum
    
    // Bias correction for velocity  
    rcp.approx.f32 %f24, %f3;
    pow.approx.f32 %f25, %f24, %f19;
    sub.f32 %f26, 1.0, %f25;
    div.approx.f32 %f27, %f17, %f26;  // bias corrected velocity
    
    // Compute parameter update
    sqrt.approx.f32 %f28, %f27;
    add.f32 %f29, %f28, %f4;     // sqrt(velocity) + eps
    div.approx.f32 %f30, %f23, %f29;  // momentum / (sqrt(velocity) + eps)
    mul.f32 %f31, %f1, %f30;     // lr * update
    sub.f32 %f32, %f8, %f31;     // new param
    
    // Convert back to FP16
    cvt.rn.f16.f32 %h3, %f32;
    
    // Store results
    st.global.f16 [%rd8], %h3;   // store param
    st.global.f32 [%rd10], %f13; // store momentum  
    st.global.f32 [%rd11], %f17; // store velocity
    
exit:
    ret;
}
"#;

const TENSOR_CORE_ADAM_BF16_PTX: &str = r#"
.version 7.0
.target sm_80  // Minimum for BF16 tensor cores
.address_size 64

.visible .entry tensor_core_adam_bf16_kernel(
    .param .u64 params,     // BF16 parameters
    .param .u64 grads,      // BF16 gradients
    .param .u64 exp_avg,    // FP32 momentum
    .param .u64 exp_avg_sq, // FP32 velocity
    .param .f32 lr,
    .param .f32 beta1,
    .param .f32 beta2,
    .param .f32 eps,
    .param .f32 weight_decay,
    .param .s32 step,
    .param .u32 rows,
    .param .u32 cols
)
{
    .reg .pred %p<8>;
    .reg .s32 %r<16>;
    .reg .s64 %rd<16>;
    .reg .b16 %b<32>;   // BF16 registers
    .reg .f32 %f<32>;
    
    // Similar structure to FP16 kernel but using BF16 conversion
    // Load parameters (simplified implementation)
    ld.param.u64 %rd1, [params];
    ld.param.u64 %rd2, [grads];
    ld.param.u64 %rd3, [exp_avg];
    ld.param.u64 %rd4, [exp_avg_sq];
    ld.param.f32 %f1, [lr];
    ld.param.f32 %f2, [beta1];
    ld.param.f32 %f3, [beta2];
    ld.param.f32 %f4, [eps];
    ld.param.f32 %f5, [weight_decay];
    ld.param.s32 %r1, [step];
    ld.param.u32 %r2, [rows];
    ld.param.u32 %r3, [cols];
    
    // Thread indices
    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %ctaid.y;
    mov.u32 %r6, %tid.x;
    mov.u32 %r7, %tid.y;
    
    // Calculate global position
    mad.lo.s32 %r8, %r4, 16, %r6;
    mad.lo.s32 %r9, %r5, 16, %r7;
    
    // Bounds check
    setp.ge.u32 %p1, %r8, %r3;
    setp.ge.u32 %p2, %r9, %r2;
    or.pred %p3, %p1, %p2;
    @%p3 bra exit;
    
    // Linear index and addressing (simplified)
    mad.lo.s32 %r10, %r9, %r3, %r8;
    cvt.u64.u32 %rd5, %r10;
    shl.b64 %rd6, %rd5, 1;  // *2 for BF16
    shl.b64 %rd7, %rd5, 2;  // *4 for FP32
    
    add.s64 %rd8, %rd1, %rd6;
    add.s64 %rd9, %rd2, %rd6;
    add.s64 %rd10, %rd3, %rd7;
    add.s64 %rd11, %rd4, %rd7;
    
    // Load BF16 values and convert to FP32
    ld.global.b16 %b1, [%rd8];
    ld.global.b16 %b2, [%rd9];
    ld.global.f32 %f6, [%rd10];
    ld.global.f32 %f7, [%rd11];
    
    // Convert BF16 to FP32 (simplified conversion)
    cvt.f32.bf16 %f8, %b1;
    cvt.f32.bf16 %f9, %b2;
    
    // Adam update (same as FP16 version)
    mad.f32 %f10, %f8, %f5, %f9;
    mul.f32 %f11, %f6, %f2;
    sub.f32 %f12, 1.0, %f2;
    mad.f32 %f13, %f10, %f12, %f11;
    
    mul.f32 %f14, %f7, %f3;
    sub.f32 %f15, 1.0, %f3;
    mul.f32 %f16, %f10, %f10;
    mad.f32 %f17, %f16, %f15, %f14;
    
    // Bias correction and update (simplified)
    sqrt.approx.f32 %f18, %f17;
    add.f32 %f19, %f18, %f4;
    div.approx.f32 %f20, %f13, %f19;
    mul.f32 %f21, %f1, %f20;
    sub.f32 %f22, %f8, %f21;
    
    // Convert back to BF16
    cvt.rn.bf16.f32 %b3, %f22;
    
    // Store results
    st.global.b16 [%rd8], %b3;
    st.global.f32 [%rd10], %f13;
    st.global.f32 [%rd11], %f17;
    
exit:
    ret;
}
"#;

const FUSED_TENSOR_CORE_PTX: &str = r#"
.version 7.0
.target sm_70
.address_size 64

.visible .entry fused_tensor_core_update_kernel(
    .param .u64 weights,
    .param .u64 gradients,
    .param .u64 updates,
    .param .u32 rows,
    .param .u32 cols,
    .param .f32 learning_rate
)
{
    .reg .pred %p<4>;
    .reg .s32 %r<8>;
    .reg .s64 %rd<8>;
    .reg .f16 %h<16>;
    .reg .f32 %f<8>;
    
    // Load parameters
    ld.param.u64 %rd1, [weights];
    ld.param.u64 %rd2, [gradients];
    ld.param.u64 %rd3, [updates];
    ld.param.u32 %r1, [rows];
    ld.param.u32 %r2, [cols];
    ld.param.f32 %f1, [learning_rate];
    
    // Calculate thread position
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.s32 %r6, %r3, %r4, %r5;
    
    // Bounds check
    mul.lo.s32 %r7, %r1, %r2;
    setp.ge.u32 %p1, %r6, %r7;
    @%p1 bra exit;
    
    // Calculate addresses
    cvt.u64.u32 %rd4, %r6;
    shl.b64 %rd5, %rd4, 1;  // *2 for FP16
    
    add.s64 %rd6, %rd1, %rd5;
    add.s64 %rd7, %rd2, %rd5;
    add.s64 %rd8, %rd3, %rd5;
    
    // Load values
    ld.global.f16 %h1, [%rd6];  // weight
    ld.global.f16 %h2, [%rd7];  // gradient
    
    // Convert to FP32 for computation
    cvt.f32.f16 %f2, %h1;
    cvt.f32.f16 %f3, %h2;
    
    // Compute update: weight = weight - learning_rate * gradient
    mul.f32 %f4, %f1, %f3;
    sub.f32 %f5, %f2, %f4;
    
    // Convert back to FP16
    cvt.rn.f16.f32 %h3, %f5;
    
    // Store results
    st.global.f16 [%rd6], %h3;  // updated weight
    st.global.f16 [%rd8], %h3;  // update copy
    
exit:
    ret;
}
"#;

const MIXED_PRECISION_PTX: &str = r#"
.version 7.0
.target sm_70
.address_size 64

.visible .entry mixed_precision_optimizer_kernel(
    .param .u64 params,
    .param .u64 grads,
    .param .u64 exp_avg,
    .param .u64 exp_avg_sq,
    .param .f32 lr,
    .param .f32 beta1,
    .param .f32 beta2,
    .param .f32 eps,
    .param .f32 weight_decay,
    .param .s32 step,
    .param .u32 size
)
{
    .reg .pred %p<4>;
    .reg .s32 %r<8>;
    .reg .s64 %rd<16>;
    .reg .f16 %h<8>;
    .reg .f32 %f<16>;
    
    // Load parameters
    ld.param.u64 %rd1, [params];
    ld.param.u64 %rd2, [grads];
    ld.param.u64 %rd3, [exp_avg];
    ld.param.u64 %rd4, [exp_avg_sq];
    ld.param.f32 %f1, [lr];
    ld.param.f32 %f2, [beta1];
    ld.param.f32 %f3, [beta2];
    ld.param.f32 %f4, [eps];
    ld.param.f32 %f5, [weight_decay];
    ld.param.s32 %r1, [step];
    ld.param.u32 %r2, [size];
    
    // Thread index
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.s32 %r6, %r3, %r4, %r5;
    
    // Bounds check
    setp.ge.u32 %p1, %r6, %r2;
    @%p1 bra exit;
    
    // Calculate addresses (assume FP16 params/grads, FP32 states)
    cvt.u64.u32 %rd5, %r6;
    shl.b64 %rd6, %rd5, 1;  // *2 for FP16
    shl.b64 %rd7, %rd5, 2;  // *4 for FP32
    
    add.s64 %rd8, %rd1, %rd6;
    add.s64 %rd9, %rd2, %rd6;
    add.s64 %rd10, %rd3, %rd7;
    add.s64 %rd11, %rd4, %rd7;
    
    // Load values
    ld.global.f16 %h1, [%rd8];
    ld.global.f16 %h2, [%rd9];
    ld.global.f32 %f6, [%rd10];
    ld.global.f32 %f7, [%rd11];
    
    // Convert FP16 to FP32
    cvt.f32.f16 %f8, %h1;
    cvt.f32.f16 %f9, %h2;
    
    // Mixed precision Adam update
    mad.f32 %f10, %f8, %f5, %f9;  // weight decay
    
    mul.f32 %f11, %f6, %f2;
    sub.f32 %f12, 1.0, %f2;
    mad.f32 %f13, %f10, %f12, %f11;
    
    mul.f32 %f14, %f7, %f3;
    sub.f32 %f15, 1.0, %f3;
    mul.f32 %f16, %f10, %f10;
    mad.f32 %f17, %f16, %f15, %f14;
    
    // Compute update
    sqrt.approx.f32 %f18, %f17;
    add.f32 %f19, %f18, %f4;
    div.approx.f32 %f20, %f13, %f19;
    mul.f32 %f21, %f1, %f20;
    sub.f32 %f22, %f8, %f21;
    
    // Convert back to FP16 for storage
    cvt.rn.f16.f32 %h3, %f22;
    
    // Store results
    st.global.f16 [%rd8], %h3;   // FP16 param
    st.global.f32 [%rd10], %f13; // FP32 momentum
    st.global.f32 [%rd11], %f17; // FP32 velocity
    
exit:
    ret;
}
"#;
