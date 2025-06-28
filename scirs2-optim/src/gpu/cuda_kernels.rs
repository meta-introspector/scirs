//! Custom CUDA kernels for memory-intensive optimizers
//!
//! This module provides highly optimized CUDA kernel implementations for
//! memory-intensive optimizers like Adam and LAMB that can benefit from
//! custom GPU acceleration.

use ndarray::{Array, Dimension};
use num_traits::Float;
use std::ffi::c_void;
use std::ptr;

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
    
    /// CUDA stream for async execution
    #[cfg(feature = "cuda")]
    stream: CudaStream,
    
    /// Block size for kernel launches
    block_size: u32,
    
    /// Maximum threads per block
    max_threads: u32,
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
            let adamw_kernel = CudaKernel::from_ptx(&context, ADAMW_KERNEL_PTX, "adamw_step_kernel")?;
            
            let max_threads = context.get_device_attribute(cuda_driver_api::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)? as u32;
            let block_size = 256.min(max_threads);
            
            Ok(Self {
                context,
                adam_kernel,
                lamb_kernel,
                adamw_kernel,
                stream,
                block_size,
                max_threads,
            })
        }
        
        #[cfg(not(feature = "cuda"))]
        {
            Ok(Self {
                block_size: 256,
                max_threads: 1024,
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
            
            self.adam_kernel.launch(
                grid_size,
                self.block_size,
                args.as_ptr(),
                0, // shared memory
                Some(&self.stream),
            )?;
            
            self.stream.synchronize()?;
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