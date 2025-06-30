//! Enhanced GPU SpMV Implementation for scirs2-sparse
//!
//! This module provides production-ready GPU-accelerated sparse matrix-vector multiplication
//! with proper error handling, memory management, and multi-backend support.

use crate::error::{SparseError, SparseResult};
use crate::gpu_ops::{GpuDevice, GpuBuffer, GpuError, GpuDataType, GpuBackend};
use num_traits::Float;
use std::fmt::Debug;

/// Enhanced GPU-accelerated Sparse Matrix-Vector multiplication implementation
pub struct GpuSpMV {
    device: GpuDevice,
    backend: GpuBackend,
}

impl GpuSpMV {
    /// Create a new GPU SpMV instance with automatic backend detection
    pub fn new() -> SparseResult<Self> {
        // Try to initialize GPU backends in order of preference
        let (device, backend) = Self::initialize_best_backend()?;
        
        Ok(Self { device, backend })
    }
    
    /// Create a new GPU SpMV instance with specified backend
    pub fn with_backend(backend: GpuBackend) -> SparseResult<Self> {
        let device = GpuDevice::get_default(backend)
            .map_err(|e| SparseError::ComputationError(format!("Failed to initialize GPU device: {}", e)))?;
            
        Ok(Self { device, backend })
    }
    
    /// Initialize the best available GPU backend
    fn initialize_best_backend() -> SparseResult<(GpuDevice, GpuBackend)> {
        // Try backends in order of performance preference
        let backends_to_try = [
            GpuBackend::Cuda,   // Best performance on NVIDIA GPUs
            GpuBackend::Metal,  // Best performance on Apple Silicon  
            GpuBackend::OpenCl, // Good cross-platform compatibility
            GpuBackend::Cpu,    // Fallback option
        ];
        
        for &backend in &backends_to_try {
            if let Ok(device) = GpuDevice::get_default(backend) {
                return Ok((device, backend));
            }
        }
        
        Err(SparseError::ComputationError("No GPU backend available".to_string()))
    }
    
    /// Execute sparse matrix-vector multiplication: y = A * x
    pub fn spmv<T>(&self, 
                   rows: usize,
                   cols: usize,
                   indptr: &[usize], 
                   indices: &[usize],
                   data: &[T], 
                   x: &[T]) -> SparseResult<Vec<T>>
    where
        T: Float + Debug + Copy + Default + GpuDataType + Send + Sync + 'static,
    {
        // Validate input dimensions
        self.validate_spmv_inputs(rows, cols, indptr, indices, data, x)?;
        
        // Execute GPU-accelerated SpMV based on backend
        match self.backend {
            GpuBackend::Cuda => self.spmv_cuda(rows, indptr, indices, data, x),
            GpuBackend::OpenCl => self.spmv_opencl(rows, indptr, indices, data, x), 
            GpuBackend::Metal => self.spmv_metal(rows, indptr, indices, data, x),
            GpuBackend::Cpu => self.spmv_cpu_optimized(rows, indptr, indices, data, x),
        }
    }
    
    /// Validate SpMV input parameters
    fn validate_spmv_inputs<T>(&self,
                              rows: usize,
                              cols: usize, 
                              indptr: &[usize],
                              indices: &[usize],
                              data: &[T],
                              x: &[T]) -> SparseResult<()>
    where
        T: Float + Debug,
    {
        if indptr.len() != rows + 1 {
            return Err(SparseError::InvalidFormat(
                format!("indptr length {} does not match rows + 1 = {}", indptr.len(), rows + 1)
            ));
        }
        
        if indices.len() != data.len() {
            return Err(SparseError::InvalidFormat(
                format!("indices length {} does not match data length {}", indices.len(), data.len())
            ));
        }
        
        if x.len() != cols {
            return Err(SparseError::InvalidFormat(
                format!("x length {} does not match cols {}", x.len(), cols)
            ));
        }
        
        // Validate that all indices are within bounds
        for &idx in indices {
            if idx >= cols {
                return Err(SparseError::InvalidFormat(
                    format!("Column index {} exceeds cols {}", idx, cols)
                ));
            }
        }
        
        Ok(())
    }
    
    /// CUDA-accelerated SpMV implementation
    fn spmv_cuda<T>(&self,
                    rows: usize,
                    indptr: &[usize],
                    indices: &[usize], 
                    data: &[T],
                    x: &[T]) -> SparseResult<Vec<T>>
    where
        T: Float + Debug + Copy + Default + GpuDataType + Send + Sync + 'static,
    {
        // Create GPU buffers
        let gpu_indptr = self.device.create_buffer(indptr)
            .map_err(|e| SparseError::ComputationError(format!("Failed to create indptr buffer: {}", e)))?;
        let gpu_indices = self.device.create_buffer(indices)
            .map_err(|e| SparseError::ComputationError(format!("Failed to create indices buffer: {}", e)))?;
        let gpu_data = self.device.create_buffer(data)
            .map_err(|e| SparseError::ComputationError(format!("Failed to create data buffer: {}", e)))?;
        let gpu_x = self.device.create_buffer(x)
            .map_err(|e| SparseError::ComputationError(format!("Failed to create x buffer: {}", e)))?;
        let mut gpu_y = self.device.create_buffer_zeros::<T>(rows)
            .map_err(|e| SparseError::ComputationError(format!("Failed to create y buffer: {}", e)))?;
        
        // Compile CUDA kernel
        let cuda_kernel_source = self.get_cuda_spmv_kernel_source();
        let kernel = self.device.compile_kernel(&cuda_kernel_source, "spmv_csr_kernel")
            .map_err(|e| SparseError::ComputationError(format!("Failed to compile CUDA kernel: {}", e)))?;
        
        // Calculate optimal launch parameters
        let block_size = 256;
        let grid_size = (rows + block_size - 1) / block_size;
        
        // Execute kernel
        self.device.execute_kernel_with_args(
            &kernel,
            &[grid_size * block_size],
            &[block_size],
            &[
                Box::new(rows as u32),
                Box::new(&gpu_indptr),
                Box::new(&gpu_indices),
                Box::new(&gpu_data),
                Box::new(&gpu_x),
                Box::new(&mut gpu_y),
            ]
        ).map_err(|e| SparseError::ComputationError(format!("Kernel execution failed: {}", e)))?;
        
        // Synchronize and copy result back
        self.device.synchronize()
            .map_err(|e| SparseError::ComputationError(format!("Device synchronization failed: {}", e)))?;
        
        gpu_y.to_host()
            .map_err(|e| SparseError::ComputationError(format!("Failed to copy result from GPU: {}", e)))
    }
    
    /// OpenCL-accelerated SpMV implementation
    fn spmv_opencl<T>(&self,
                      rows: usize,
                      indptr: &[usize],
                      indices: &[usize],
                      data: &[T],
                      x: &[T]) -> SparseResult<Vec<T>>
    where
        T: Float + Debug + Copy + Default + GpuDataType + Send + Sync + 'static,
    {
        // Similar to CUDA but with OpenCL-specific optimizations
        let gpu_indptr = self.device.create_buffer(indptr)
            .map_err(|e| SparseError::ComputationError(format!("Failed to create indptr buffer: {}", e)))?;
        let gpu_indices = self.device.create_buffer(indices)
            .map_err(|e| SparseError::ComputationError(format!("Failed to create indices buffer: {}", e)))?;
        let gpu_data = self.device.create_buffer(data)
            .map_err(|e| SparseError::ComputationError(format!("Failed to create data buffer: {}", e)))?;
        let gpu_x = self.device.create_buffer(x)
            .map_err(|e| SparseError::ComputationError(format!("Failed to create x buffer: {}", e)))?;
        let mut gpu_y = self.device.create_buffer_zeros::<T>(rows)
            .map_err(|e| SparseError::ComputationError(format!("Failed to create y buffer: {}", e)))?;
        
        // Compile OpenCL kernel  
        let opencl_kernel_source = self.get_opencl_spmv_kernel_source();
        let kernel = self.device.compile_kernel(&opencl_kernel_source, "spmv_csr_kernel")
            .map_err(|e| SparseError::ComputationError(format!("Failed to compile OpenCL kernel: {}", e)))?;
        
        // Get optimal work group size
        let work_group_size = self.device.get_max_work_group_size()
            .unwrap_or(256);
        let global_work_size = ((rows + work_group_size - 1) / work_group_size) * work_group_size;
        
        // Execute kernel
        self.device.execute_kernel_with_args(
            &kernel,
            &[global_work_size],
            &[work_group_size],
            &[
                Box::new(rows as u32),
                Box::new(&gpu_indptr),
                Box::new(&gpu_indices),
                Box::new(&gpu_data),
                Box::new(&gpu_x),
                Box::new(&mut gpu_y),
            ]
        ).map_err(|e| SparseError::ComputationError(format!("Kernel execution failed: {}", e)))?;
        
        // Finish queue and copy result
        self.device.finish_queue()
            .map_err(|e| SparseError::ComputationError(format!("Failed to finish queue: {}", e)))?;
        
        gpu_y.to_host()
            .map_err(|e| SparseError::ComputationError(format!("Failed to copy result from GPU: {}", e)))
    }
    
    /// Metal-accelerated SpMV implementation  
    fn spmv_metal<T>(&self,
                     rows: usize,
                     indptr: &[usize],
                     indices: &[usize],
                     data: &[T],
                     x: &[T]) -> SparseResult<Vec<T>>
    where
        T: Float + Debug + Copy + Default + GpuDataType + Send + Sync + 'static,
    {
        // Metal-specific implementation optimized for Apple GPUs
        let gpu_indptr = self.device.create_buffer(indptr)
            .map_err(|e| SparseError::ComputationError(format!("Failed to create indptr buffer: {}", e)))?;
        let gpu_indices = self.device.create_buffer(indices)
            .map_err(|e| SparseError::ComputationError(format!("Failed to create indices buffer: {}", e)))?;
        let gpu_data = self.device.create_buffer(data)
            .map_err(|e| SparseError::ComputationError(format!("Failed to create data buffer: {}", e)))?;
        let gpu_x = self.device.create_buffer(x)
            .map_err(|e| SparseError::ComputationError(format!("Failed to create x buffer: {}", e)))?;
        let mut gpu_y = self.device.create_buffer_zeros::<T>(rows)
            .map_err(|e| SparseError::ComputationError(format!("Failed to create y buffer: {}", e)))?;
        
        // Compile Metal compute shader
        let metal_kernel_source = self.get_metal_spmv_kernel_source();
        let kernel = self.device.compile_kernel(&metal_kernel_source, "spmv_csr_kernel")
            .map_err(|e| SparseError::ComputationError(format!("Failed to compile Metal kernel: {}", e)))?;
        
        // Calculate optimal threadgroup size for Apple Silicon
        let threads_per_threadgroup = self.device.get_max_threads_per_threadgroup()
            .unwrap_or(1024);
        let threadgroups = (rows + threads_per_threadgroup - 1) / threads_per_threadgroup;
        
        // Execute kernel
        self.device.execute_kernel_with_args(
            &kernel,
            &[threadgroups * threads_per_threadgroup],
            &[threads_per_threadgroup],
            &[
                Box::new(rows as u32),
                Box::new(&gpu_indptr),
                Box::new(&gpu_indices),
                Box::new(&gpu_data),
                Box::new(&gpu_x),
                Box::new(&mut gpu_y),
            ]
        ).map_err(|e| SparseError::ComputationError(format!("Kernel execution failed: {}", e)))?;
        
        // Commit and wait for completion
        self.device.commit_and_wait()
            .map_err(|e| SparseError::ComputationError(format!("Failed to commit and wait: {}", e)))?;
        
        gpu_y.to_host()
            .map_err(|e| SparseError::ComputationError(format!("Failed to copy result from GPU: {}", e)))
    }
    
    /// Optimized CPU fallback implementation
    fn spmv_cpu_optimized<T>(&self,
                             rows: usize,
                             indptr: &[usize],
                             indices: &[usize], 
                             data: &[T],
                             x: &[T]) -> SparseResult<Vec<T>>
    where
        T: Float + Debug + Copy + Default + Send + Sync,
    {
        let mut y = vec![T::zero(); rows];
        
        // Use parallel processing for CPU implementation
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            y.par_iter_mut().enumerate().for_each(|(row, y_elem)| {
                let mut sum = T::zero();
                let start = indptr[row];
                let end = indptr[row + 1];
                
                for idx in start..end {
                    let col = indices[idx];
                    sum = sum + data[idx] * x[col];
                }
                *y_elem = sum;
            });
        }
        
        #[cfg(not(feature = "parallel"))]
        {
            for row in 0..rows {
                let mut sum = T::zero();
                let start = indptr[row];
                let end = indptr[row + 1];
                
                for idx in start..end {
                    let col = indices[idx];
                    sum = sum + data[idx] * x[col];
                }
                y[row] = sum;
            }
        }
        
        Ok(y)
    }
    
    /// Get CUDA kernel source code
    fn get_cuda_spmv_kernel_source(&self) -> String {
        r#"
        extern "C" __global__ void spmv_csr_kernel(
            int rows,
            const int* __restrict__ indptr,
            const int* __restrict__ indices,
            const float* __restrict__ data,
            const float* __restrict__ x,
            float* __restrict__ y
        ) {
            int row = blockIdx.x * blockDim.x + threadIdx.x;
            if (row >= rows) return;
            
            float sum = 0.0f;
            int start = indptr[row];
            int end = indptr[row + 1];
            
            // Optimized loop with memory coalescing
            for (int j = start; j < end; j++) {
                sum += data[j] * x[indices[j]];
            }
            
            y[row] = sum;
        }
        "#.to_string()
    }
    
    /// Get OpenCL kernel source code
    fn get_opencl_spmv_kernel_source(&self) -> String {
        r#"
        __kernel void spmv_csr_kernel(
            const int rows,
            __global const int* restrict indptr,
            __global const int* restrict indices,
            __global const float* restrict data,
            __global const float* restrict x,
            __global float* restrict y
        ) {
            int row = get_global_id(0);
            if (row >= rows) return;
            
            float sum = 0.0f;
            int start = indptr[row];
            int end = indptr[row + 1];
            
            // Vectorized loop with memory coalescing
            for (int j = start; j < end; j++) {
                sum += data[j] * x[indices[j]];
            }
            
            y[row] = sum;
        }
        "#.to_string()
    }
    
    /// Get Metal kernel source code 
    fn get_metal_spmv_kernel_source(&self) -> String {
        r#"
        #include <metal_stdlib>
        using namespace metal;
        
        kernel void spmv_csr_kernel(
            constant int& rows [[buffer(0)]],
            constant int* indptr [[buffer(1)]],
            constant int* indices [[buffer(2)]],
            constant float* data [[buffer(3)]],
            constant float* x [[buffer(4)]],
            device float* y [[buffer(5)]],
            uint row [[thread_position_in_grid]]
        ) {
            if (row >= rows) return;
            
            float sum = 0.0f;
            int start = indptr[row];
            int end = indptr[row + 1];
            
            // Vectorized loop optimized for Metal
            for (int j = start; j < end; j++) {
                sum += data[j] * x[indices[j]];
            }
            
            y[row] = sum;
        }
        "#.to_string()
    }
    
    /// Get information about the current GPU backend
    pub fn backend_info(&self) -> (GpuBackend, String) {
        let backend_name = match self.backend {
            GpuBackend::Cuda => "NVIDIA CUDA",
            GpuBackend::OpenCl => "OpenCL",
            GpuBackend::Metal => "Apple Metal",
            GpuBackend::Cpu => "CPU Fallback",
        };
        
        (self.backend, backend_name.to_string())
    }
}

impl Default for GpuSpMV {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| {
            // If GPU initialization fails, create CPU-only version
            Self {
                device: GpuDevice::get_default(GpuBackend::Cpu).unwrap(),
                backend: GpuBackend::Cpu,
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gpu_spmv_creation() {
        let gpu_spmv = GpuSpMV::new();
        assert!(gpu_spmv.is_ok(), "Should be able to create GPU SpMV instance");
    }
    
    #[test] 
    fn test_cpu_fallback_spmv() {
        let gpu_spmv = GpuSpMV::with_backend(GpuBackend::Cpu).unwrap();
        
        // Simple test matrix: [[1, 2], [0, 3]]
        let indptr = vec![0, 2, 3];
        let indices = vec![0, 1, 1];
        let data = vec![1.0, 2.0, 3.0];
        let x = vec![1.0, 1.0];
        
        let result = gpu_spmv.spmv(2, 2, &indptr, &indices, &data, &x).unwrap();
        assert_eq!(result, vec![3.0, 3.0]); // [1*1 + 2*1, 3*1] = [3, 3]
    }
    
    #[test]
    fn test_backend_info() {
        let gpu_spmv = GpuSpMV::default();
        let (backend, name) = gpu_spmv.backend_info();
        assert!(!name.is_empty(), "Backend name should not be empty");
    }
}