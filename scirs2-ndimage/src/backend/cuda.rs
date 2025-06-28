//! CUDA backend implementation for GPU acceleration
//!
//! This module provides CUDA-specific implementations for GPU-accelerated
//! image processing operations.

use crate::backend::{GpuBuffer, GpuContext, GpuKernelExecutor, KernelInfo};
use crate::error::{NdimageError, NdimageResult};
use ndarray::{Array, ArrayView, ArrayView2, Dimension};
use num_traits::{Float, FromPrimitive, Zero};
use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr, CString};
use std::fmt::Debug;
use std::ptr;
use std::sync::{Arc, Mutex};

// CUDA FFI bindings
#[link(name = "cuda")]
#[link(name = "cudart")]
#[link(name = "nvrtc")]
extern "C" {
    // CUDA Runtime API
    fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(devPtr: *mut c_void) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    fn cudaMemcpyAsync(dst: *mut c_void, src: *const c_void, count: usize, kind: i32, stream: *mut c_void) -> i32;
    fn cudaGetDeviceCount(count: *mut i32) -> i32;
    fn cudaSetDevice(device: i32) -> i32;
    fn cudaGetDevice(device: *mut i32) -> i32;
    fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
    fn cudaStreamCreate(stream: *mut *mut c_void) -> i32;
    fn cudaStreamDestroy(stream: *mut c_void) -> i32;
    fn cudaStreamSynchronize(stream: *mut c_void) -> i32;
    fn cudaDeviceSynchronize() -> i32;
    fn cudaGetLastError() -> i32;
    fn cudaGetErrorString(error: i32) -> *const c_char;
    
    // CUDA Driver API for kernel launch
    fn cuModuleLoadData(module: *mut *mut c_void, image: *const c_void) -> i32;
    fn cuModuleGetFunction(hfunc: *mut *mut c_void, hmod: *mut c_void, name: *const c_char) -> i32;
    fn cuLaunchKernel(
        f: *mut c_void,
        grid_dim_x: u32, grid_dim_y: u32, grid_dim_z: u32,
        block_dim_x: u32, block_dim_y: u32, block_dim_z: u32,
        shared_mem_bytes: u32,
        stream: *mut c_void,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> i32;
    
    // NVRTC API for runtime compilation
    fn nvrtcCreateProgram(
        prog: *mut *mut c_void,
        src: *const c_char,
        name: *const c_char,
        num_headers: i32,
        headers: *const *const c_char,
        include_names: *const *const c_char,
    ) -> i32;
    fn nvrtcDestroyProgram(prog: *mut *mut c_void) -> i32;
    fn nvrtcCompileProgram(prog: *mut c_void, num_options: i32, options: *const *const c_char) -> i32;
    fn nvrtcGetPTXSize(prog: *mut c_void, ptx_size: *mut usize) -> i32;
    fn nvrtcGetPTX(prog: *mut c_void, ptx: *mut c_char) -> i32;
    fn nvrtcGetProgramLogSize(prog: *mut c_void, log_size: *mut usize) -> i32;
    fn nvrtcGetProgramLog(prog: *mut c_void, log: *mut c_char) -> i32;
}

// CUDA memory copy kinds
const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;
const CUDA_MEMCPY_DEVICE_TO_DEVICE: i32 = 3;

// CUDA error codes
const CUDA_SUCCESS: i32 = 0;
const NVRTC_SUCCESS: i32 = 0;

// Helper function to get CUDA error string
fn cuda_error_string(error: i32) -> String {
    unsafe {
        let error_ptr = cudaGetErrorString(error);
        if error_ptr.is_null() {
            format!("Unknown CUDA error: {}", error)
        } else {
            CStr::from_ptr(error_ptr)
                .to_string_lossy()
                .into_owned()
        }
    }
}

/// CUDA-specific GPU buffer implementation
pub struct CudaBuffer<T> {
    device_ptr: *mut c_void,
    size: usize,
    phantom: std::marker::PhantomData<T>,
}

impl<T> CudaBuffer<T> {
    pub fn new(size: usize) -> NdimageResult<Self> {
        let mut device_ptr: *mut c_void = ptr::null_mut();
        let byte_size = size * std::mem::size_of::<T>();
        
        unsafe {
            let result = cudaMalloc(&mut device_ptr, byte_size);
            if result != CUDA_SUCCESS {
                return Err(NdimageError::ComputationError(
                    format!("CUDA malloc failed with error code: {}", result),
                ));
            }
        }
        
        Ok(Self {
            device_ptr,
            size,
            phantom: std::marker::PhantomData,
        })
    }
    
    pub fn from_host_data(data: &[T]) -> NdimageResult<Self> {
        let buffer = Self::new(data.len())?;
        buffer.copy_from_host(data)?;
        Ok(buffer)
    }
}

impl<T> Drop for CudaBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            if !self.device_ptr.is_null() {
                cudaFree(self.device_ptr);
            }
        }
    }
}

impl<T> GpuBuffer<T> for CudaBuffer<T> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    
    fn size(&self) -> usize {
        self.size
    }
    
    fn copy_from_host(&mut self, data: &[T]) -> NdimageResult<()> {
        if data.len() != self.size {
            return Err(NdimageError::InvalidInput(
                "Data size mismatch".to_string(),
            ));
        }
        
        let byte_size = self.size * std::mem::size_of::<T>();
        unsafe {
            let result = cudaMemcpy(
                self.device_ptr,
                data.as_ptr() as *const c_void,
                byte_size,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            );
            
            if result != CUDA_SUCCESS {
                return Err(NdimageError::ComputationError(
                    format!("CUDA memcpy failed with error code: {}", result),
                ));
            }
        }
        
        Ok(())
    }
    
    fn copy_to_host(&self, data: &mut [T]) -> NdimageResult<()> {
        if data.len() != self.size {
            return Err(NdimageError::InvalidInput(
                "Data size mismatch".to_string(),
            ));
        }
        
        let byte_size = self.size * std::mem::size_of::<T>();
        unsafe {
            let result = cudaMemcpy(
                data.as_mut_ptr() as *mut c_void,
                self.device_ptr,
                byte_size,
                CUDA_MEMCPY_DEVICE_TO_HOST,
            );
            
            if result != CUDA_SUCCESS {
                return Err(NdimageError::ComputationError(
                    format!("CUDA memcpy failed with error code: {}", result),
                ));
            }
        }
        
        Ok(())
    }
}

/// CUDA context implementation
pub struct CudaContext {
    device_id: i32,
    compute_capability: (i32, i32),
    max_threads_per_block: i32,
    max_shared_memory: usize,
}

impl CudaContext {
    pub fn new(device_id: Option<usize>) -> NdimageResult<Self> {
        let device_id = device_id.unwrap_or(0) as i32;
        
        // Check if device exists
        let mut device_count: i32 = 0;
        unsafe {
            let result = cudaGetDeviceCount(&mut device_count);
            if result != CUDA_SUCCESS {
                return Err(NdimageError::ComputationError(
                    format!("Failed to get CUDA device count: {}", result),
                ));
            }
            
            if device_id >= device_count {
                return Err(NdimageError::InvalidInput(
                    format!("CUDA device {} not found. Only {} devices available", device_id, device_count),
                ));
            }
            
            // Set the device
            let result = cudaSetDevice(device_id);
            if result != CUDA_SUCCESS {
                return Err(NdimageError::ComputationError(
                    format!("Failed to set CUDA device: {}", result),
                ));
            }
        }
        
        // Get device properties (simplified for now)
        Ok(Self {
            device_id,
            compute_capability: (7, 5), // Default to compute capability 7.5
            max_threads_per_block: 1024,
            max_shared_memory: 49152, // 48KB
        })
    }
    
    pub fn compile_kernel(&self, source: &str, kernel_name: &str) -> NdimageResult<CudaKernel> {
        // Check cache first
        {
            let cache = KERNEL_CACHE.lock().unwrap();
            if let Some(kernel) = cache.get(kernel_name) {
                return Ok(CudaKernel {
                    name: kernel.name.clone(),
                    module: kernel.module,
                    function: kernel.function,
                    ptx_code: kernel.ptx_code.clone(),
                });
            }
        }
        
        // Convert OpenCL-style kernel to CUDA
        let cuda_source = convert_opencl_to_cuda(source);
        let c_source = CString::new(cuda_source).map_err(|_| {
            NdimageError::ComputationError("Failed to create C string for kernel source".into())
        })?;
        let c_name = CString::new(kernel_name).map_err(|_| {
            NdimageError::ComputationError("Failed to create C string for kernel name".into())
        })?;
        
        unsafe {
            // Create NVRTC program
            let mut prog: *mut c_void = ptr::null_mut();
            let result = nvrtcCreateProgram(
                &mut prog,
                c_source.as_ptr(),
                c_name.as_ptr(),
                0,
                ptr::null(),
                ptr::null(),
            );
            
            if result != NVRTC_SUCCESS {
                return Err(NdimageError::ComputationError(
                    format!("Failed to create NVRTC program: {}", result),
                ));
            }
            
            // Compile program with appropriate options
            let options = vec![
                CString::new("--gpu-architecture=compute_70").unwrap(),
                CString::new("--fmad=true").unwrap(),
            ];
            let option_ptrs: Vec<*const c_char> = options.iter().map(|s| s.as_ptr()).collect();
            
            let compile_result = nvrtcCompileProgram(
                prog,
                option_ptrs.len() as i32,
                option_ptrs.as_ptr(),
            );
            
            // Get compilation log
            if compile_result != NVRTC_SUCCESS {
                let mut log_size: usize = 0;
                nvrtcGetProgramLogSize(prog, &mut log_size);
                
                if log_size > 0 {
                    let mut log = vec![0u8; log_size];
                    nvrtcGetProgramLog(prog, log.as_mut_ptr() as *mut c_char);
                    let log_str = String::from_utf8_lossy(&log[..log_size - 1]);
                    
                    nvrtcDestroyProgram(&mut prog);
                    return Err(NdimageError::ComputationError(
                        format!("CUDA compilation failed:\n{}", log_str),
                    ));
                }
            }
            
            // Get PTX code
            let mut ptx_size: usize = 0;
            nvrtcGetPTXSize(prog, &mut ptx_size);
            
            let mut ptx_code = vec![0u8; ptx_size];
            nvrtcGetPTX(prog, ptx_code.as_mut_ptr() as *mut c_char);
            
            // Clean up NVRTC program
            nvrtcDestroyProgram(&mut prog);
            
            // Load PTX module
            let mut module: *mut c_void = ptr::null_mut();
            let load_result = cuModuleLoadData(&mut module, ptx_code.as_ptr() as *const c_void);
            
            if load_result != CUDA_SUCCESS {
                return Err(NdimageError::ComputationError(
                    format!("Failed to load CUDA module: {}", cuda_error_string(load_result)),
                ));
            }
            
            // Get function from module
            let mut function: *mut c_void = ptr::null_mut();
            let func_result = cuModuleGetFunction(&mut function, module, c_name.as_ptr());
            
            if func_result != CUDA_SUCCESS {
                return Err(NdimageError::ComputationError(
                    format!("Failed to get CUDA function: {}", cuda_error_string(func_result)),
                ));
            }
            
            let kernel = CudaKernel {
                name: kernel_name.to_string(),
                module,
                function,
                ptx_code: ptx_code[..ptx_size - 1].to_vec(), // Remove null terminator
            };
            
            // Cache the compiled kernel
            {
                let mut cache = KERNEL_CACHE.lock().unwrap();
                cache.insert(kernel_name.to_string(), CudaKernel {
                    name: kernel.name.clone(),
                    module: kernel.module,
                    function: kernel.function,
                    ptx_code: kernel.ptx_code.clone(),
                });
            }
            
            Ok(kernel)
        }
    }
}

impl GpuContext for CudaContext {
    fn name(&self) -> &str {
        "CUDA"
    }
    
    fn device_count(&self) -> usize {
        let mut count: i32 = 0;
        unsafe {
            cudaGetDeviceCount(&mut count);
        }
        count as usize
    }
    
    fn current_device(&self) -> usize {
        self.device_id as usize
    }
    
    fn memory_info(&self) -> (usize, usize) {
        let mut free: usize = 0;
        let mut total: usize = 0;
        unsafe {
            cudaMemGetInfo(&mut free, &mut total);
        }
        (total - free, total)
    }
}

/// CUDA kernel handle
pub struct CudaKernel {
    name: String,
    module: *mut c_void,
    function: *mut c_void,
    ptx_code: Vec<u8>,
}

/// Kernel cache to avoid recompilation
lazy_static::lazy_static! {
    static ref KERNEL_CACHE: Arc<Mutex<HashMap<String, CudaKernel>>> = Arc::new(Mutex::new(HashMap::new()));
}

/// CUDA kernel executor implementation
pub struct CudaExecutor {
    context: Arc<CudaContext>,
    stream: *mut c_void,
}

impl CudaExecutor {
    pub fn new(context: Arc<CudaContext>) -> NdimageResult<Self> {
        let mut stream: *mut c_void = ptr::null_mut();
        unsafe {
            let result = cudaStreamCreate(&mut stream);
            if result != CUDA_SUCCESS {
                return Err(NdimageError::ComputationError(
                    format!("Failed to create CUDA stream: {}", result),
                ));
            }
        }
        
        Ok(Self { context, stream })
    }
}

impl Drop for CudaExecutor {
    fn drop(&mut self) {
        unsafe {
            if !self.stream.is_null() {
                cudaStreamDestroy(self.stream);
            }
        }
    }
}

impl<T> GpuKernelExecutor<T> for CudaExecutor
where
    T: Float + FromPrimitive + Debug + Clone,
{
    fn execute_kernel(
        &self,
        kernel: &KernelInfo,
        inputs: &[&dyn GpuBuffer<T>],
        outputs: &[&mut dyn GpuBuffer<T>],
        work_size: &[usize],
        params: &[T],
    ) -> NdimageResult<()> {
        // Compile kernel
        let cuda_kernel = self.context.compile_kernel(&kernel.source, &kernel.entry_point)?;
        
        // Calculate grid and block dimensions
        let (grid_dim, block_dim) = calculate_launch_config(work_size, kernel.work_dimensions);
        
        // Prepare kernel arguments
        let mut kernel_args: Vec<*mut c_void> = Vec::new();
        
        // Add input buffers
        for input in inputs {
            let cuda_buf = input.as_any().downcast_ref::<CudaBuffer<T>>()
                .ok_or_else(|| NdimageError::InvalidInput("Expected CUDA buffer".into()))?;
            kernel_args.push(&cuda_buf.device_ptr as *const _ as *mut c_void);
        }
        
        // Add output buffers
        for output in outputs {
            let cuda_buf = output.as_any().downcast_ref::<CudaBuffer<T>>()
                .ok_or_else(|| NdimageError::InvalidInput("Expected CUDA buffer".into()))?;
            kernel_args.push(&cuda_buf.device_ptr as *const _ as *mut c_void);
        }
        
        // Add scalar parameters
        let mut param_storage: Vec<T> = params.to_vec();
        for param in &mut param_storage {
            kernel_args.push(param as *mut T as *mut c_void);
        }
        
        // Launch kernel
        unsafe {
            let result = cuLaunchKernel(
                cuda_kernel.function,
                grid_dim.0, grid_dim.1, grid_dim.2,
                block_dim.0, block_dim.1, block_dim.2,
                0, // shared memory
                self.stream,
                kernel_args.as_mut_ptr(),
                ptr::null_mut(),
            );
            
            if result != CUDA_SUCCESS {
                return Err(NdimageError::ComputationError(
                    format!("CUDA kernel launch failed: {}", cuda_error_string(result)),
                ));
            }
            
            // Synchronize stream
            let sync_result = cudaStreamSynchronize(self.stream);
            if sync_result != CUDA_SUCCESS {
                return Err(NdimageError::ComputationError(
                    format!("CUDA stream sync failed: {}", cuda_error_string(sync_result)),
                ));
            }
            
            // Check for kernel errors
            let error = cudaGetLastError();
            if error != CUDA_SUCCESS {
                return Err(NdimageError::ComputationError(
                    format!("CUDA kernel execution error: {}", cuda_error_string(error)),
                ));
            }
        }
        
        Ok(())
    }
}

/// High-level CUDA operations
pub struct CudaOperations {
    context: Arc<CudaContext>,
    executor: CudaExecutor,
}

impl CudaOperations {
    pub fn new(device_id: Option<usize>) -> NdimageResult<Self> {
        let context = Arc::new(CudaContext::new(device_id)?);
        let executor = CudaExecutor::new(context.clone())?;
        
        Ok(Self { context, executor })
    }
    
    /// GPU-accelerated Gaussian filter
    pub fn gaussian_filter_2d<T>(
        &self,
        input: &ArrayView2<T>,
        sigma: [T; 2],
    ) -> NdimageResult<Array<T, ndarray::Ix2>>
    where
        T: Float + FromPrimitive + Debug + Clone + Send + Sync + 'static,
    {
        crate::backend::kernels::gpu_gaussian_filter_2d(input, sigma, &self.executor)
    }
    
    /// GPU-accelerated convolution
    pub fn convolve_2d<T>(
        &self,
        input: &ArrayView2<T>,
        kernel: &ArrayView2<T>,
    ) -> NdimageResult<Array<T, ndarray::Ix2>>
    where
        T: Float + FromPrimitive + Debug + Clone + Send + Sync + 'static,
    {
        crate::backend::kernels::gpu_convolve_2d(input, kernel, &self.executor)
    }
    
    /// GPU-accelerated median filter
    pub fn median_filter_2d<T>(
        &self,
        input: &ArrayView2<T>,
        size: [usize; 2],
    ) -> NdimageResult<Array<T, ndarray::Ix2>>
    where
        T: Float + FromPrimitive + Debug + Clone + Send + Sync + 'static,
    {
        crate::backend::kernels::gpu_median_filter_2d(input, size, &self.executor)
    }
    
    /// GPU-accelerated morphological erosion
    pub fn erosion_2d<T>(
        &self,
        input: &ArrayView2<T>,
        structure: &ArrayView2<bool>,
    ) -> NdimageResult<Array<T, ndarray::Ix2>>
    where
        T: Float + FromPrimitive + Debug + Clone + Send + Sync + 'static,
    {
        crate::backend::kernels::gpu_erosion_2d(input, structure, &self.executor)
    }
}

/// Helper function to allocate GPU buffer
pub fn allocate_gpu_buffer<T>(data: &[T]) -> NdimageResult<Box<dyn GpuBuffer<T>>>
where
    T: 'static,
{
    Ok(Box::new(CudaBuffer::from_host_data(data)?))
}

/// Helper function to allocate empty GPU buffer
pub fn allocate_gpu_buffer_empty<T>(size: usize) -> NdimageResult<Box<dyn GpuBuffer<T>>>
where
    T: 'static,
{
    Ok(Box::new(CudaBuffer::<T>::new(size)?))
}

/// Convert OpenCL-style kernel to CUDA syntax
fn convert_opencl_to_cuda(source: &str) -> String {
    source
        .replace("__kernel", "extern \"C\" __global__")
        .replace("__global", "")
        .replace("get_global_id(0)", "blockIdx.x * blockDim.x + threadIdx.x")
        .replace("get_global_id(1)", "blockIdx.y * blockDim.y + threadIdx.y")
        .replace("get_global_id(2)", "blockIdx.z * blockDim.z + threadIdx.z")
        .replace("clamp(", "min(max(")
}

/// Calculate optimal grid and block dimensions for kernel launch
fn calculate_launch_config(work_size: &[usize], dimensions: usize) -> ((u32, u32, u32), (u32, u32, u32)) {
    let block_size = match dimensions {
        1 => (256, 1, 1),
        2 => (16, 16, 1),
        3 => (8, 8, 4),
        _ => (256, 1, 1),
    };
    
    let grid_size = match dimensions {
        1 => {
            let blocks = (work_size[0] + block_size.0 as usize - 1) / block_size.0 as usize;
            (blocks as u32, 1, 1)
        }
        2 => {
            let blocks_x = (work_size[0] + block_size.0 as usize - 1) / block_size.0 as usize;
            let blocks_y = (work_size[1] + block_size.1 as usize - 1) / block_size.1 as usize;
            (blocks_x as u32, blocks_y as u32, 1)
        }
        3 => {
            let blocks_x = (work_size[0] + block_size.0 as usize - 1) / block_size.0 as usize;
            let blocks_y = (work_size[1] + block_size.1 as usize - 1) / block_size.1 as usize;
            let blocks_z = (work_size[2] + block_size.2 as usize - 1) / block_size.2 as usize;
            (blocks_x as u32, blocks_y as u32, blocks_z as u32)
        }
        _ => {
            let blocks = (work_size[0] + block_size.0 as usize - 1) / block_size.0 as usize;
            (blocks as u32, 1, 1)
        }
    };
    
    (grid_size, block_size)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    #[ignore] // Ignore by default as it requires CUDA
    fn test_cuda_context_creation() {
        let context = CudaContext::new(None);
        assert!(context.is_ok());
        
        if let Ok(ctx) = context {
            assert_eq!(ctx.device_id, 0);
            assert!(ctx.device_count() > 0);
        }
    }
    
    #[test]
    #[ignore] // Ignore by default as it requires CUDA
    fn test_cuda_buffer_allocation() {
        let buffer: Result<CudaBuffer<f32>, _> = CudaBuffer::new(1024);
        assert!(buffer.is_ok());
        
        if let Ok(buf) = buffer {
            assert_eq!(buf.size(), 1024);
        }
    }
}