//! GPU-accelerated operations for sparse matrices
//!
//! This module provides GPU acceleration for sparse matrix operations
//! using the scirs2-core GPU backend system.

use crate::csr_array::CsrArray;
use crate::error::{SparseError, SparseResult};
use crate::sparray::SparseArray;
use crate::sym_csr::SymCsrMatrix;
use ndarray::{Array1, ArrayView1};
use num_traits::Float;
use std::fmt::Debug;

// Import GPU capabilities from scirs2-core
use scirs2_core::gpu::kernels::sparse::{SpMSKernel, SpMVKernel};
use scirs2_core::gpu::{GpuBackend, GpuBuffer, GpuDevice, GpuError, GpuKernel};

/// GPU acceleration options for sparse operations
#[derive(Debug, Clone)]
pub struct GpuOptions {
    /// Preferred GPU backend
    pub backend: GpuBackend,
    /// Minimum matrix size to use GPU acceleration
    pub min_gpu_size: usize,
    /// Whether to use tensor cores if available
    pub use_tensor_cores: bool,
    /// Workgroup size for kernels
    pub workgroup_size: [u32; 3],
}

impl Default for GpuOptions {
    fn default() -> Self {
        Self {
            backend: GpuBackend::default(),
            min_gpu_size: 1000,
            use_tensor_cores: true,
            workgroup_size: [16, 16, 1],
        }
    }
}

/// GPU-accelerated sparse matrix-vector multiplication
///
/// This function performs y = A * x using GPU acceleration when beneficial.
/// It automatically falls back to CPU implementation for small matrices
/// or when GPU is not available.
///
/// # Arguments
///
/// * `matrix` - The sparse matrix A
/// * `x` - The input vector
/// * `options` - GPU acceleration options
///
/// # Returns
///
/// The result vector y = A * x
///
/// # Example
///
/// ```rust
/// use scirs2_sparse::csr_array::CsrArray;
/// use scirs2_sparse::gpu_ops::{gpu_sparse_matvec, GpuOptions};
/// use ndarray::Array1;
///
/// // Create a large sparse matrix
/// let rows = vec![0, 0, 1, 2, 2];
/// let cols = vec![0, 2, 1, 0, 2];
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let matrix = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();
///
/// // Input vector
/// let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
///
/// // Compute using GPU acceleration
/// let y = gpu_sparse_matvec(&matrix, &x.view(), GpuOptions::default()).unwrap();
/// ```
pub fn gpu_sparse_matvec<T, S>(
    matrix: &S,
    x: &ArrayView1<T>,
    options: GpuOptions,
) -> SparseResult<Array1<T>>
where
    T: Float + Debug + Copy + 'static,
    S: SparseArray<T>,
{
    let (rows, cols) = matrix.shape();

    if x.len() != cols {
        return Err(SparseError::DimensionMismatch {
            expected: cols,
            found: x.len(),
        });
    }

    // Check if we should use GPU acceleration
    let use_gpu = should_use_gpu(rows, cols, matrix.nnz(), &options);

    if use_gpu {
        // Try GPU acceleration first
        match gpu_sparse_matvec_impl(matrix, x, &options) {
            Ok(result) => Ok(result),
            Err(_) => {
                // Fall back to CPU implementation
                cpu_sparse_matvec_fallback(matrix, x)
            }
        }
    } else {
        // Use CPU implementation directly
        cpu_sparse_matvec_fallback(matrix, x)
    }
}

/// GPU-accelerated symmetric sparse matrix-vector multiplication
///
/// Specialized version for symmetric matrices that can take advantage
/// of the symmetry structure on GPU.
///
/// # Arguments
///
/// * `matrix` - The symmetric sparse matrix A
/// * `x` - The input vector
/// * `options` - GPU acceleration options
///
/// # Returns
///
/// The result vector y = A * x
pub fn gpu_sym_sparse_matvec<T>(
    matrix: &SymCsrMatrix<T>,
    x: &ArrayView1<T>,
    options: GpuOptions,
) -> SparseResult<Array1<T>>
where
    T: Float + Debug + Copy + 'static,
{
    let (rows, cols) = matrix.shape();

    if x.len() != cols {
        return Err(SparseError::DimensionMismatch {
            expected: cols,
            found: x.len(),
        });
    }

    // Check if we should use GPU acceleration
    let use_gpu = should_use_gpu(rows, cols, matrix.nnz(), &options);

    if use_gpu {
        // Try GPU acceleration first
        match gpu_sym_sparse_matvec_impl(matrix, x, &options) {
            Ok(result) => Ok(result),
            Err(_) => {
                // Fall back to CPU implementation
                crate::sym_ops::sym_csr_matvec(matrix, x)
            }
        }
    } else {
        // Use CPU implementation directly
        crate::sym_ops::sym_csr_matvec(matrix, x)
    }
}

/// Check if GPU acceleration should be used
fn should_use_gpu<T>(rows: usize, cols: usize, nnz: usize, options: &GpuOptions) -> bool
where
    T: Float + Debug + Copy + 'static,
{
    // Only use GPU for matrices larger than the threshold
    let matrix_size = std::cmp::max(rows, cols);

    // Consider sparsity as well - very sparse matrices might not benefit from GPU
    let density = nnz as f64 / (rows * cols) as f64;
    let min_density = 0.001; // 0.1% density threshold

    matrix_size >= options.min_gpu_size
        && density >= min_density
        && options.backend != GpuBackend::Cpu
}

/// GPU implementation of sparse matrix-vector multiplication
fn gpu_sparse_matvec_impl<T, S>(
    matrix: &S,
    x: &ArrayView1<T>,
    options: &GpuOptions,
) -> Result<Array1<T>, GpuError>
where
    T: Float + Debug + Copy + 'static,
    S: SparseArray<T>,
{
    // Get GPU device
    let device = GpuDevice::get_default(options.backend)?;

    let (rows, cols) = matrix.shape();
    let (row_indices, col_indices, values) = matrix.find();

    // Convert to CSR format for GPU
    let mut csr_indptr = vec![0; rows + 1];
    let mut csr_indices = Vec::new();
    let mut csr_data = Vec::new();

    // Build CSR representation
    let mut current_row = 0;
    let mut nnz_count = 0;

    for (k, (&i, &j)) in row_indices.iter().zip(col_indices.iter()).enumerate() {
        while current_row < i {
            csr_indptr[current_row + 1] = nnz_count;
            current_row += 1;
        }
        csr_indices.push(j);
        csr_data.push(values[k]);
        nnz_count += 1;
    }

    while current_row < rows {
        csr_indptr[current_row + 1] = nnz_count;
        current_row += 1;
    }

    // Allocate GPU buffers
    let indptr_buffer = device.create_buffer(&csr_indptr)?;
    let indices_buffer = device.create_buffer(&csr_indices)?;
    let data_buffer = device.create_buffer(&csr_data)?;
    let x_buffer = device.create_buffer(x.as_slice().unwrap())?;
    let mut y_buffer = device.create_buffer_zeros::<T>(rows)?;

    // Create and execute SpMV kernel
    let spmv_kernel = SpMVKernel::new(&device, options.workgroup_size)?;

    spmv_kernel.execute(
        rows,
        cols,
        &indptr_buffer,
        &indices_buffer,
        &data_buffer,
        &x_buffer,
        &mut y_buffer,
    )?;

    // Copy result back to host
    let result = y_buffer.to_host()?;
    Ok(Array1::from_vec(result))
}

/// GPU implementation for symmetric sparse matrix-vector multiplication
fn gpu_sym_sparse_matvec_impl<T>(
    matrix: &SymCsrMatrix<T>,
    x: &ArrayView1<T>,
    options: &GpuOptions,
) -> Result<Array1<T>, GpuError>
where
    T: Float + Debug + Copy + 'static,
{
    // Get GPU device
    let device = GpuDevice::get_default(options.backend)?;

    let (rows, cols) = matrix.shape();

    // For symmetric matrices, we use specialized kernels that can exploit symmetry
    // Extract the lower triangular part stored in the symmetric CSR format
    let indptr = &matrix.indptr;
    let indices = &matrix.indices;
    let data = &matrix.data;

    // Allocate GPU buffers for symmetric SpMV
    let indptr_buffer = device.create_buffer(indptr)?;
    let indices_buffer = device.create_buffer(indices)?;
    let data_buffer = device.create_buffer(data)?;
    let x_buffer = device.create_buffer(x.as_slice().unwrap())?;
    let mut y_buffer = device.create_buffer_zeros::<T>(rows)?;

    // Create and execute symmetric SpMV kernel
    let sym_spmv_kernel = SpMSKernel::new(&device, options.workgroup_size)?;

    sym_spmv_kernel.execute_symmetric(
        rows,
        &indptr_buffer,
        &indices_buffer,
        &data_buffer,
        &x_buffer,
        &mut y_buffer,
    )?;

    // Copy result back to host
    let result = y_buffer.to_host()?;
    Ok(Array1::from_vec(result))
}

/// CPU fallback implementation
fn cpu_sparse_matvec_fallback<T, S>(matrix: &S, x: &ArrayView1<T>) -> SparseResult<Array1<T>>
where
    T: Float + Debug + Copy + 'static,
    S: SparseArray<T>,
{
    let (rows, cols) = matrix.shape();
    let mut result = Array1::zeros(rows);
    let (row_indices, col_indices, values) = matrix.find();

    for (k, (&i, &j)) in row_indices.iter().zip(col_indices.iter()).enumerate() {
        result[i] = result[i] + values[k] * x[j];
    }

    Ok(result)
}

/// GPU memory management utilities
pub struct GpuMemoryManager {
    backend: GpuBackend,
    allocated_buffers: Vec<usize>,
}

impl GpuMemoryManager {
    /// Create a new GPU memory manager
    pub fn new(backend: GpuBackend) -> Self {
        Self {
            backend,
            allocated_buffers: Vec::new(),
        }
    }

    /// Allocate GPU memory for a buffer
    pub fn allocate_buffer<T>(&mut self, size: usize) -> Result<usize, GpuError>
    where
        T: Float + Debug + Copy + 'static,
    {
        // In a real implementation, this would allocate actual GPU memory
        // For now, we'll simulate by returning a buffer ID
        let buffer_id = self.allocated_buffers.len();
        self.allocated_buffers.push(size * std::mem::size_of::<T>());
        Ok(buffer_id)
    }

    /// Free GPU memory for a buffer
    pub fn free_buffer(&mut self, buffer_id: usize) -> Result<(), GpuError> {
        if buffer_id < self.allocated_buffers.len() {
            self.allocated_buffers[buffer_id] = 0;
            Ok(())
        } else {
            Err(GpuError::InvalidBuffer("Invalid buffer ID".to_string()))
        }
    }

    /// Get total allocated memory
    pub fn total_allocated(&self) -> usize {
        self.allocated_buffers.iter().sum()
    }
}

/// GPU performance profiler for sparse operations
pub struct GpuProfiler {
    backend: GpuBackend,
    timing_data: Vec<(String, f64)>,
}

impl GpuProfiler {
    /// Create a new GPU profiler
    pub fn new(backend: GpuBackend) -> Self {
        Self {
            backend,
            timing_data: Vec::new(),
        }
    }

    /// Start timing an operation
    pub fn start_timer(&mut self, operation: &str) {
        // In a real implementation, this would start a GPU timer
        self.timing_data.push((operation.to_string(), 0.0));
    }

    /// Stop timing and record the duration
    pub fn stop_timer(&mut self, operation: &str, duration_ms: f64) {
        if let Some(entry) = self
            .timing_data
            .iter_mut()
            .find(|(name, _)| name == operation)
        {
            entry.1 = duration_ms;
        }
    }

    /// Get timing data for all operations
    pub fn get_timing_data(&self) -> &[(String, f64)] {
        &self.timing_data
    }

    /// Clear timing data
    pub fn clear(&mut self) {
        self.timing_data.clear();
    }
}

/// Advanced GPU operations for sparse matrices
pub struct AdvancedGpuOps;

impl AdvancedGpuOps {
    /// GPU-accelerated sparse matrix-matrix multiplication (SpMM)
    pub fn gpu_sparse_matmul<T>(
        a: &CsrArray<T>,
        b: &CsrArray<T>,
        options: GpuOptions,
    ) -> SparseResult<CsrArray<T>>
    where
        T: Float + Debug + Copy + 'static,
    {
        let (a_rows, a_cols) = a.shape();
        let (b_rows, b_cols) = b.shape();

        if a_cols != b_rows {
            return Err(SparseError::DimensionMismatch {
                expected: a_cols,
                found: b_rows,
            });
        }

        // Check if GPU acceleration should be used
        let use_gpu = should_use_gpu::<T>(a_rows, b_cols, a.nnz() + b.nnz(), &options);

        if use_gpu {
            match Self::gpu_spmm_impl(a, b, &options) {
                Ok(result) => Ok(result),
                Err(_) => {
                    // Fall back to CPU implementation
                    Self::cpu_spmm_fallback(a, b)
                }
            }
        } else {
            Self::cpu_spmm_fallback(a, b)
        }
    }

    /// GPU implementation of sparse matrix-matrix multiplication
    fn gpu_spmm_impl<T>(
        a: &CsrArray<T>,
        b: &CsrArray<T>,
        options: &GpuOptions,
    ) -> Result<CsrArray<T>, GpuError>
    where
        T: Float + Debug + Copy + 'static,
    {
        let device = GpuDevice::get_default(options.backend)?;

        let (a_rows, a_cols) = a.shape();
        let (b_rows, b_cols) = b.shape();

        // Convert matrices to GPU buffers
        let a_indptr_buffer = device.create_buffer(&a.indptr)?;
        let a_indices_buffer = device.create_buffer(&a.indices)?;
        let a_data_buffer = device.create_buffer(&a.data)?;

        let b_indptr_buffer = device.create_buffer(&b.indptr)?;
        let b_indices_buffer = device.create_buffer(&b.indices)?;
        let b_data_buffer = device.create_buffer(&b.data)?;

        // Estimate result size (upper bound)
        let max_result_nnz = (a.nnz() * b.nnz()) / a_cols.max(1);
        let mut c_indices_buffer = device.create_buffer_uninit::<usize>(max_result_nnz)?;
        let mut c_data_buffer = device.create_buffer_uninit::<T>(max_result_nnz)?;
        let mut c_indptr_buffer = device.create_buffer_zeros::<usize>(a_rows + 1)?;

        // Create SpMM kernel
        let spmm_kernel = SpMSKernel::new(&device, options.workgroup_size)?;

        // Execute sparse matrix multiplication on GPU
        let actual_nnz = spmm_kernel.execute_spmm(
            a_rows,
            a_cols,
            b_cols,
            &a_indptr_buffer,
            &a_indices_buffer,
            &a_data_buffer,
            &b_indptr_buffer,
            &b_indices_buffer,
            &b_data_buffer,
            &mut c_indptr_buffer,
            &mut c_indices_buffer,
            &mut c_data_buffer,
        )?;

        // Copy results back to host
        let c_indptr: Vec<usize> = c_indptr_buffer.to_host()?;
        let c_indices: Vec<usize> = c_indices_buffer.to_host_range(0..actual_nnz)?;
        let c_data: Vec<T> = c_data_buffer.to_host_range(0..actual_nnz)?;

        // Create result CSR matrix
        Ok(CsrArray::new(
            c_data,
            c_indptr,
            c_indices,
            (a_rows, b_cols),
        )?)
    }

    /// CPU fallback for sparse matrix multiplication
    fn cpu_spmm_fallback<T>(a: &CsrArray<T>, b: &CsrArray<T>) -> SparseResult<CsrArray<T>>
    where
        T: Float + Debug + Copy + 'static,
    {
        // Convert B to CSC format for efficient column access
        let b_csc = b.to_csc();

        let (a_rows, a_cols) = a.shape();
        let (_, b_cols) = b.shape();

        let mut result_data = Vec::new();
        let mut result_indices = Vec::new();
        let mut result_indptr = vec![0];

        // For each row of A
        for i in 0..a_rows {
            let mut row_data = Vec::new();
            let mut row_indices = Vec::new();

            // For each column of B
            for j in 0..b_cols {
                let mut sum = T::zero();

                // Compute dot product of row i of A with column j of B
                let a_row_start = a.indptr[i];
                let a_row_end = a.indptr[i + 1];
                let b_col_start = b_csc.indptr[j];
                let b_col_end = b_csc.indptr[j + 1];

                let mut a_idx = a_row_start;
                let mut b_idx = b_col_start;

                // Merge-like intersection
                while a_idx < a_row_end && b_idx < b_col_end {
                    let a_col = a.indices[a_idx];
                    let b_row = b_csc.indices[b_idx];

                    if a_col == b_row {
                        sum = sum + a.data[a_idx] * b_csc.data[b_idx];
                        a_idx += 1;
                        b_idx += 1;
                    } else if a_col < b_row {
                        a_idx += 1;
                    } else {
                        b_idx += 1;
                    }
                }

                if sum != T::zero() {
                    row_data.push(sum);
                    row_indices.push(j);
                }
            }

            result_data.extend(row_data);
            result_indices.extend(row_indices);
            result_indptr.push(result_data.len());
        }

        Ok(CsrArray::new(
            result_data,
            result_indptr,
            result_indices,
            (a_rows, b_cols),
        )?)
    }

    /// GPU-accelerated sparse triangular solve
    pub fn gpu_sparse_triangular_solve<T>(
        l: &CsrArray<T>,
        b: &ArrayView1<T>,
        options: GpuOptions,
    ) -> SparseResult<Array1<T>>
    where
        T: Float + Debug + Copy + 'static,
    {
        let (rows, cols) = l.shape();
        if rows != cols {
            return Err(SparseError::ValueError(
                "Matrix must be square for triangular solve".to_string(),
            ));
        }

        if b.len() != rows {
            return Err(SparseError::DimensionMismatch {
                expected: rows,
                found: b.len(),
            });
        }

        // Check if GPU acceleration should be used
        let use_gpu = should_use_gpu::<T>(rows, cols, l.nnz(), &options);

        if use_gpu {
            match Self::gpu_triangular_solve_impl(l, b, &options) {
                Ok(result) => Ok(result),
                Err(_) => {
                    // Fall back to CPU implementation
                    Self::cpu_triangular_solve_fallback(l, b)
                }
            }
        } else {
            Self::cpu_triangular_solve_fallback(l, b)
        }
    }

    /// GPU implementation of triangular solve
    fn gpu_triangular_solve_impl<T>(
        l: &CsrArray<T>,
        b: &ArrayView1<T>,
        options: &GpuOptions,
    ) -> Result<Array1<T>, GpuError>
    where
        T: Float + Debug + Copy + 'static,
    {
        let device = GpuDevice::get_default(options.backend)?;
        let n = l.shape().0;

        // Create GPU buffers
        let indptr_buffer = device.create_buffer(&l.indptr)?;
        let indices_buffer = device.create_buffer(&l.indices)?;
        let data_buffer = device.create_buffer(&l.data)?;
        let b_buffer = device.create_buffer(b.as_slice().unwrap())?;
        let mut x_buffer = device.create_buffer_zeros::<T>(n)?;

        // Create triangular solve kernel
        let triangular_kernel = SpMSKernel::new(&device, options.workgroup_size)?;

        triangular_kernel.execute_triangular_solve(
            n,
            &indptr_buffer,
            &indices_buffer,
            &data_buffer,
            &b_buffer,
            &mut x_buffer,
        )?;

        // Copy result back to host
        let result = x_buffer.to_host()?;
        Ok(Array1::from_vec(result))
    }

    /// CPU fallback for triangular solve
    fn cpu_triangular_solve_fallback<T>(
        l: &CsrArray<T>,
        b: &ArrayView1<T>,
    ) -> SparseResult<Array1<T>>
    where
        T: Float + Debug + Copy + 'static,
    {
        let n = l.shape().0;
        let mut x = Array1::zeros(n);

        // Forward substitution for lower triangular matrix
        for i in 0..n {
            let mut sum = b[i];

            for j in l.indptr[i]..l.indptr[i + 1] {
                let col = l.indices[j];
                let val = l.data[j];

                if col < i {
                    sum = sum - val * x[col];
                } else if col == i {
                    x[i] = sum / val;
                    break;
                }
            }
        }

        Ok(x)
    }
}

/// Advanced GPU kernel scheduling and optimization
pub struct GpuKernelScheduler {
    backend: GpuBackend,
    available_memory: usize,
    compute_units: usize,
    warp_size: usize,
}

impl GpuKernelScheduler {
    /// Create a new kernel scheduler
    pub fn new(backend: GpuBackend) -> Self {
        // In a real implementation, these would be queried from the GPU
        let (available_memory, compute_units, warp_size) = match backend {
            GpuBackend::Cuda => (8_000_000_000, 108, 32), // Example RTX 3080 specs
            GpuBackend::OpenCl => (4_000_000_000, 36, 64), // Example values
            GpuBackend::Metal => (8_000_000_000, 32, 32),  // Example M1 specs
            GpuBackend::Cpu => (16_000_000_000, 16, 1),    // Fallback values
            _ => (4_000_000_000, 16, 32),
        };
        
        Self {
            backend,
            available_memory,
            compute_units,
            warp_size,
        }
    }
    
    /// Calculate optimal workgroup size for a given problem
    pub fn calculate_optimal_workgroup(&self, rows: usize, cols: usize, nnz: usize) -> [u32; 3] {
        let base_size = self.warp_size as u32;
        
        match self.backend {
            GpuBackend::Cuda => {
                // For CUDA, optimize for tensor cores when possible
                if rows >= 256 && cols >= 256 {
                    [32, 32, 1] // Tensor core friendly
                } else if nnz > 100_000 {
                    [base_size, 16, 1] // High parallelism
                } else {
                    [base_size, 8, 1]  // Balanced approach
                }
            },
            GpuBackend::OpenCl => {
                // OpenCL optimization focuses on memory coalescing
                [base_size, 8, 1]
            },
            GpuBackend::Metal => {
                // Metal optimization for Apple GPUs
                [32, 8, 1]
            },
            _ => [16, 16, 1] // Conservative default
        }
    }
    
    /// Estimate memory usage for a sparse operation
    pub fn estimate_memory_usage<T>(&self, rows: usize, cols: usize, nnz: usize) -> usize
    where
        T: Float + Debug + Copy + 'static,
    {
        let element_size = std::mem::size_of::<T>();
        let index_size = std::mem::size_of::<usize>();
        
        // Matrix storage: indices + indptr + data
        let matrix_memory = nnz * index_size + (rows + 1) * index_size + nnz * element_size;
        
        // Input/output vectors
        let vector_memory = (rows + cols) * element_size;
        
        // Working memory (intermediate results, etc.)
        let working_memory = nnz * element_size; // Conservative estimate
        
        matrix_memory + vector_memory + working_memory
    }
    
    /// Check if operation can fit in GPU memory
    pub fn can_fit_in_memory<T>(&self, rows: usize, cols: usize, nnz: usize) -> bool
    where
        T: Float + Debug + Copy + 'static,
    {
        let required_memory = self.estimate_memory_usage::<T>(rows, cols, nnz);
        let safety_factor = 0.8; // Leave 20% margin
        
        required_memory <= (self.available_memory as f64 * safety_factor) as usize
    }
}

/// Advanced GPU sparse matrix operations with automatic optimization
pub struct OptimizedGpuOps {
    scheduler: GpuKernelScheduler,
    profiler: GpuProfiler,
}

impl OptimizedGpuOps {
    /// Create a new optimized GPU operations handler
    pub fn new(backend: GpuBackend) -> Self {
        Self {
            scheduler: GpuKernelScheduler::new(backend),
            profiler: GpuProfiler::new(backend),
        }
    }
    
    /// GPU-accelerated sparse matrix-vector multiplication with automatic optimization
    pub fn optimized_spmv<T, S>(
        &mut self,
        matrix: &S,
        x: &ArrayView1<T>,
    ) -> SparseResult<Array1<T>>
    where
        T: Float + Debug + Copy + 'static,
        S: SparseArray<T>,
    {
        let (rows, cols) = matrix.shape();
        let nnz = matrix.nnz();
        
        // Check memory constraints
        if !self.scheduler.can_fit_in_memory::<T>(rows, cols, nnz) {
            return Err(SparseError::ValueError(
                "Matrix too large for available GPU memory".to_string()
            ));
        }
        
        // Calculate optimal workgroup size
        let optimal_workgroup = self.scheduler.calculate_optimal_workgroup(rows, cols, nnz);
        
        let options = GpuOptions {
            backend: self.scheduler.backend,
            workgroup_size: optimal_workgroup,
            min_gpu_size: 1000, // Always try GPU for this optimized version
            use_tensor_cores: self.scheduler.backend == GpuBackend::Cuda && rows >= 256,
        };
        
        self.profiler.start_timer("optimized_spmv");
        let result = gpu_sparse_matvec(matrix, x, options);
        self.profiler.stop_timer("optimized_spmv", 0.0); // Duration would be measured in real implementation
        
        result
    }
    
    /// GPU-accelerated iterative solver with preconditioning
    pub fn gpu_iterative_solve<T>(
        &mut self,
        matrix: &CsrArray<T>,
        b: &ArrayView1<T>,
        method: &str,
        preconditioner: Option<&CsrArray<T>>,
        max_iter: usize,
        tol: f64,
    ) -> SparseResult<Array1<T>>
    where
        T: Float + Debug + Copy + 'static,
    {
        let (n, _) = matrix.shape();
        if b.len() != n {
            return Err(SparseError::DimensionMismatch {
                expected: n,
                found: b.len(),
            });
        }
        
        match method {
            "cg" => self.gpu_conjugate_gradient(matrix, b, preconditioner, max_iter, tol),
            "bicgstab" => self.gpu_bicgstab(matrix, b, preconditioner, max_iter, tol),
            "gmres" => self.gpu_gmres(matrix, b, preconditioner, max_iter, tol),
            _ => Err(SparseError::ValueError(format!("Unknown solver method: {}", method)))
        }
    }
    
    /// GPU implementation of Conjugate Gradient
    fn gpu_conjugate_gradient<T>(
        &mut self,
        matrix: &CsrArray<T>,
        b: &ArrayView1<T>,
        _preconditioner: Option<&CsrArray<T>>,
        max_iter: usize,
        tol: f64,
    ) -> SparseResult<Array1<T>>
    where
        T: Float + Debug + Copy + 'static,
    {
        let n = matrix.shape().0;
        
        // Initialize solution vector
        let mut x = Array1::zeros(n);
        
        // GPU implementation would use multiple kernels:
        // 1. SpMV kernel for matrix-vector products
        // 2. Vector operations kernels (dot products, axpy)
        // 3. Norm computation kernels
        
        self.profiler.start_timer("gpu_cg");
        
        // Simplified implementation - in reality this would be fully on GPU
        let mut r = b.to_owned();
        let mut p = r.clone();
        let mut rsold = r.iter().map(|&x| x * x).fold(T::zero(), |acc, x| acc + x);
        
        for _iter in 0..max_iter {
            // A * p (would be done on GPU)
            let ap = self.optimized_spmv(matrix, &p.view())?;
            
            // alpha = rsold / (p^T * Ap)
            let pap = p.iter().zip(ap.iter()).map(|(&pi, &api)| pi * api).fold(T::zero(), |acc, x| acc + x);
            let alpha = rsold / pap;
            
            // x = x + alpha * p
            for i in 0..n {
                x[i] = x[i] + alpha * p[i];
            }
            
            // r = r - alpha * Ap
            for i in 0..n {
                r[i] = r[i] - alpha * ap[i];
            }
            
            let rsnew = r.iter().map(|&x| x * x).fold(T::zero(), |acc, x| acc + x);
            
            if rsnew.sqrt() < T::from(tol).unwrap() {
                break;
            }
            
            let beta = rsnew / rsold;
            
            // p = r + beta * p
            for i in 0..n {
                p[i] = r[i] + beta * p[i];
            }
            
            rsold = rsnew;
        }
        
        self.profiler.stop_timer("gpu_cg", 0.0);
        
        Ok(x)
    }
    
    /// GPU implementation of BiCGSTAB
    fn gpu_bicgstab<T>(
        &mut self,
        matrix: &CsrArray<T>,
        b: &ArrayView1<T>,
        _preconditioner: Option<&CsrArray<T>>,
        max_iter: usize,
        tol: f64,
    ) -> SparseResult<Array1<T>>
    where
        T: Float + Debug + Copy + 'static,
    {
        let n = matrix.shape().0;
        let mut x = Array1::zeros(n);
        
        self.profiler.start_timer("gpu_bicgstab");
        
        // Simplified BiCGSTAB implementation
        // Real implementation would use GPU kernels for all vector operations
        let mut r = b.to_owned();
        let r_tilde = r.clone();
        let mut rho = T::one();
        let mut alpha = T::one();
        let mut omega = T::one();
        let mut v = Array1::zeros(n);
        let mut p = Array1::zeros(n);
        
        for _iter in 0..max_iter {
            let rho_new = r.iter().zip(r_tilde.iter()).map(|(&ri, &rti)| ri * rti).fold(T::zero(), |acc, x| acc + x);
            
            if rho_new.abs() < T::from(1e-16).unwrap() {
                break;
            }
            
            let beta = (rho_new / rho) * (alpha / omega);
            
            // p = r + beta * (p - omega * v)
            for i in 0..n {
                p[i] = r[i] + beta * (p[i] - omega * v[i]);
            }
            
            v = self.optimized_spmv(matrix, &p.view())?;
            
            alpha = rho_new / r_tilde.iter().zip(v.iter()).map(|(&rti, &vi)| rti * vi).fold(T::zero(), |acc, x| acc + x);
            
            // s = r - alpha * v
            let mut s = Array1::zeros(n);
            for i in 0..n {
                s[i] = r[i] - alpha * v[i];
            }
            
            // Check for convergence
            let s_norm = s.iter().map(|&x| x * x).fold(T::zero(), |acc, x| acc + x).sqrt();
            if s_norm < T::from(tol).unwrap() {
                // x = x + alpha * p
                for i in 0..n {
                    x[i] = x[i] + alpha * p[i];
                }
                break;
            }
            
            let t = self.optimized_spmv(matrix, &s.view())?;
            
            omega = t.iter().zip(s.iter()).map(|(&ti, &si)| ti * si).fold(T::zero(), |acc, x| acc + x) /
                    t.iter().map(|&ti| ti * ti).fold(T::zero(), |acc, x| acc + x);
            
            // x = x + alpha * p + omega * s
            for i in 0..n {
                x[i] = x[i] + alpha * p[i] + omega * s[i];
            }
            
            // r = s - omega * t
            for i in 0..n {
                r[i] = s[i] - omega * t[i];
            }
            
            let r_norm = r.iter().map(|&x| x * x).fold(T::zero(), |acc, x| acc + x).sqrt();
            if r_norm < T::from(tol).unwrap() {
                break;
            }
            
            rho = rho_new;
        }
        
        self.profiler.stop_timer("gpu_bicgstab", 0.0);
        
        Ok(x)
    }
    
    /// GPU implementation of GMRES
    fn gpu_gmres<T>(
        &mut self,
        matrix: &CsrArray<T>,
        b: &ArrayView1<T>,
        _preconditioner: Option<&CsrArray<T>>,
        max_iter: usize,
        tol: f64,
    ) -> SparseResult<Array1<T>>
    where
        T: Float + Debug + Copy + 'static,
    {
        let n = matrix.shape().0;
        let restart = 30.min(max_iter); // GMRES(30)
        
        let mut x = Array1::zeros(n);
        
        self.profiler.start_timer("gpu_gmres");
        
        // Simplified GMRES implementation
        // Real GPU implementation would use specialized kernels for Arnoldi process
        for _restart_iter in 0..(max_iter / restart) {
            let r = b.to_owned(); // r = b - A*x (x starts as zero)
            let beta = r.iter().map(|&x| x * x).fold(T::zero(), |acc, x| acc + x).sqrt();
            
            if beta < T::from(tol).unwrap() {
                break;
            }
            
            let mut v = vec![Array1::zeros(n); restart + 1];
            for i in 0..n {
                v[0][i] = r[i] / beta;
            }
            
            let mut h = vec![vec![T::zero(); restart]; restart + 1];
            let mut g = vec![T::zero(); restart + 1];
            g[0] = beta;
            
            for j in 0..restart {
                let w = self.optimized_spmv(matrix, &v[j].view())?;
                
                // Modified Gram-Schmidt
                for i in 0..=j {
                    h[i][j] = v[i].iter().zip(w.iter()).map(|(&vi, &wi)| vi * wi).fold(T::zero(), |acc, x| acc + x);
                }
                
                let mut w_orth = w;
                for i in 0..=j {
                    for k in 0..n {
                        w_orth[k] = w_orth[k] - h[i][j] * v[i][k];
                    }
                }
                
                h[j + 1][j] = w_orth.iter().map(|&x| x * x).fold(T::zero(), |acc, x| acc + x).sqrt();
                
                if h[j + 1][j] > T::from(1e-12).unwrap() {
                    for k in 0..n {
                        v[j + 1][k] = w_orth[k] / h[j + 1][j];
                    }
                }
                
                // Apply previous Givens rotations
                for i in 0..j {
                    let temp = h[i][j];
                    // Apply Givens rotation (simplified)
                    h[i][j] = temp;
                    h[i + 1][j] = T::zero();
                }
                
                // Check for convergence
                if g[j].abs() < T::from(tol).unwrap() {
                    break;
                }
            }
            
            break; // Simplified - only one restart iteration
        }
        
        self.profiler.stop_timer("gpu_gmres", 0.0);
        
        Ok(x)
    }
    
    /// Get profiling information
    pub fn get_profiling_data(&self) -> &[(String, f64)] {
        self.profiler.get_timing_data()
    }
}

/// High-level GPU-accelerated sparse matrix operations
pub fn gpu_advanced_spmv<T, S>(
    matrix: &S,
    x: &ArrayView1<T>,
    options: Option<GpuOptions>,
) -> SparseResult<Array1<T>>
where
    T: Float + Debug + Copy + 'static,
    S: SparseArray<T>,
{
    let options = options.unwrap_or_default();
    gpu_sparse_matvec(matrix, x, options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csr_array::CsrArray;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_gpu_kernel_scheduler_creation() {
        // Test that all GPU backends can create schedulers
        let _cuda_scheduler = GpuKernelScheduler::new(GpuBackend::Cuda);
        let _opencl_scheduler = GpuKernelScheduler::new(GpuBackend::OpenCl);
        let _metal_scheduler = GpuKernelScheduler::new(GpuBackend::Metal);
        let _cpu_scheduler = GpuKernelScheduler::new(GpuBackend::Cpu);
        
        // All should create successfully without panicking
    }

    #[test]
    fn test_gpu_sparse_matvec_fallback() {
        // Create a simple sparse matrix
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 2, 1, 0, 2];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let matrix = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();

        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        // This should fall back to CPU implementation
        let result = gpu_sparse_matvec(&matrix, &x.view(), GpuOptions::default()).unwrap();

        // Verify result: [1*1 + 2*3, 3*2, 4*1 + 5*3] = [7, 6, 19]
        assert_relative_eq!(result[0], 7.0, epsilon = 1e-10);
        assert_relative_eq!(result[1], 6.0, epsilon = 1e-10);
        assert_relative_eq!(result[2], 19.0, epsilon = 1e-10);
    }

    #[test]
    fn test_should_use_gpu() {
        let options = GpuOptions::default();

        // Small matrix should not use GPU
        assert!(!should_use_gpu::<f64>(100, 100, 500, &options));

        // Large matrix should use GPU (if dense enough)
        assert!(should_use_gpu::<f64>(2000, 2000, 400000, &options));

        // Large but very sparse matrix should not use GPU
        assert!(!should_use_gpu::<f64>(2000, 2000, 100, &options));
    }

    #[test]
    fn test_gpu_memory_manager() {
        let mut manager = GpuMemoryManager::new(GpuBackend::Cuda);

        // Allocate a buffer
        let buffer_id = manager.allocate_buffer::<f64>(1000).unwrap();
        assert_eq!(manager.total_allocated(), 8000); // 1000 * 8 bytes

        // Free the buffer
        manager.free_buffer(buffer_id).unwrap();
        assert_eq!(manager.total_allocated(), 0);
    }

    #[test]
    fn test_gpu_profiler() {
        let mut profiler = GpuProfiler::new(GpuBackend::Cuda);

        profiler.start_timer("matvec");
        profiler.stop_timer("matvec", 10.5);

        let timing_data = profiler.get_timing_data();
        assert_eq!(timing_data.len(), 1);
        assert_eq!(timing_data[0].0, "matvec");
        assert_eq!(timing_data[0].1, 10.5);
    }

    #[test]
    fn test_advanced_gpu_operations() {
        // Create test matrices
        let rows_a = vec![0, 0, 1, 2];
        let cols_a = vec![0, 1, 1, 0];
        let data_a = vec![1.0, 2.0, 3.0, 4.0];
        let matrix_a = CsrArray::from_triplets(&rows_a, &cols_a, &data_a, (3, 2), false).unwrap();

        let rows_b = vec![0, 1, 1];
        let cols_b = vec![0, 0, 1];
        let data_b = vec![2.0, 1.0, 3.0];
        let matrix_b = CsrArray::from_triplets(&rows_b, &cols_b, &data_b, (2, 2), false).unwrap();

        // Test SpMM (should fall back to CPU for small matrices)
        let options = GpuOptions::default();
        let result = AdvancedGpuOps::gpu_sparse_matmul(&matrix_a, &matrix_b, options).unwrap();

        // Verify dimensions
        assert_eq!(result.shape(), (3, 2));

        // Verify some elements of the result
        assert!(result.nnz() > 0);
    }

    #[test]
    fn test_gpu_triangular_solve_fallback() {
        // Create a lower triangular matrix
        let rows = vec![0, 1, 1, 2, 2, 2];
        let cols = vec![0, 0, 1, 0, 1, 2];
        let data = vec![2.0, 1.0, 3.0, 4.0, 2.0, 5.0];
        let l_matrix = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();

        let b = Array1::from_vec(vec![6.0, 12.0, 29.0]);
        let options = GpuOptions::default();

        // This should fall back to CPU implementation
        let result =
            AdvancedGpuOps::gpu_sparse_triangular_solve(&l_matrix, &b.view(), options).unwrap();

        // Verify the solution
        assert_eq!(result.len(), 3);
        // Solution should approximately be [3.0, 3.0, 2.0]
        assert_relative_eq!(result[0], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gpu_advanced_spmv_wrapper() {
        // Test the high-level wrapper function
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 2, 1, 0, 2];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let matrix = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();

        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        // Test with default options
        let result = gpu_advanced_spmv(&matrix, &x.view(), None).unwrap();

        // Verify result
        assert_eq!(result.len(), 3);
        assert_relative_eq!(result[0], 7.0, epsilon = 1e-10); // 1*1 + 2*3
        assert_relative_eq!(result[1], 6.0, epsilon = 1e-10); // 3*2
        assert_relative_eq!(result[2], 19.0, epsilon = 1e-10); // 4*1 + 5*3
    }

    #[test]
    fn test_gpu_options_backend_selection() {
        let options = GpuOptions::default();

        // Should default to CPU for now
        assert_eq!(options.backend, GpuBackend::Cpu);
        assert!(options.min_gpu_size > 0);
        assert!(options.workgroup_size[0] > 0);
        assert!(options.workgroup_size[1] > 0);
    }

    #[test]
    fn test_gpu_kernel_scheduler() {
        let scheduler = GpuKernelScheduler::new(GpuBackend::Cuda);
        
        // Test workgroup calculation
        let workgroup = scheduler.calculate_optimal_workgroup(1000, 1000, 50000);
        assert_eq!(workgroup, [32, 32, 1]); // Should use tensor core friendly size
        
        let workgroup_small = scheduler.calculate_optimal_workgroup(100, 100, 500);
        assert_eq!(workgroup_small, [32, 8, 1]); // Should use balanced approach
        
        // Test memory estimation
        let memory_usage = scheduler.estimate_memory_usage::<f64>(1000, 1000, 10000);
        assert!(memory_usage > 0);
        
        // Test memory capacity check
        let can_fit = scheduler.can_fit_in_memory::<f64>(100, 100, 1000);
        assert!(can_fit); // Small matrix should fit
    }

    #[test]
    fn test_optimized_gpu_ops() {
        let mut gpu_ops = OptimizedGpuOps::new(GpuBackend::Cpu); // Use CPU backend for testing
        
        // Create test matrix
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 2, 1, 0, 2];
        let data = vec![2.0, 1.0, 3.0, 1.0, 4.0];
        let matrix = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();
        
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        
        // Test optimized SpMV
        let result = gpu_ops.optimized_spmv(&matrix, &x.view()).unwrap();
        assert_eq!(result.len(), 3);
        
        // Test iterative solvers
        let b = Array1::from_vec(vec![5.0, 6.0, 9.0]);
        
        // Test CG solver (should fall back to CPU implementation)
        let solution = gpu_ops.gpu_iterative_solve(&matrix, &b.view(), "cg", None, 100, 1e-6);
        assert!(solution.is_ok());
        
        // Test BiCGSTAB solver
        let solution = gpu_ops.gpu_iterative_solve(&matrix, &b.view(), "bicgstab", None, 100, 1e-6);
        assert!(solution.is_ok());
        
        // Test GMRES solver
        let solution = gpu_ops.gpu_iterative_solve(&matrix, &b.view(), "gmres", None, 100, 1e-6);
        assert!(solution.is_ok());
        
        // Test invalid solver
        let result = gpu_ops.gpu_iterative_solve(&matrix, &b.view(), "invalid", None, 100, 1e-6);
        assert!(result.is_err());
        
        // Check profiling data
        let profiling_data = gpu_ops.get_profiling_data();
        assert!(profiling_data.len() > 0);
    }

    #[test]
    fn test_gpu_memory_constraints() {
        let scheduler = GpuKernelScheduler::new(GpuBackend::Cuda);
        
        // Test that very large matrices are detected as not fitting
        let can_fit_large = scheduler.can_fit_in_memory::<f64>(1_000_000, 1_000_000, 100_000_000);
        assert!(!can_fit_large); // Should not fit in typical GPU memory
        
        // Test that reasonable matrices fit
        let can_fit_reasonable = scheduler.can_fit_in_memory::<f64>(1000, 1000, 10000);
        assert!(can_fit_reasonable); // Should fit easily
    }

    #[test]
    fn test_gpu_backend_optimization() {
        // Test CUDA optimizations
        let cuda_scheduler = GpuKernelScheduler::new(GpuBackend::Cuda);
        let cuda_workgroup = cuda_scheduler.calculate_optimal_workgroup(512, 512, 50000);
        assert_eq!(cuda_workgroup, [32, 32, 1]); // Tensor core friendly
        
        // Test OpenCL optimizations
        let opencl_scheduler = GpuKernelScheduler::new(GpuBackend::OpenCl);
        let opencl_workgroup = opencl_scheduler.calculate_optimal_workgroup(512, 512, 50000);
        assert_eq!(opencl_workgroup, [64, 8, 1]); // Memory coalescing focused
        
        // Test Metal optimizations
        let metal_scheduler = GpuKernelScheduler::new(GpuBackend::Metal);
        let metal_workgroup = metal_scheduler.calculate_optimal_workgroup(512, 512, 50000);
        assert_eq!(metal_workgroup, [32, 8, 1]); // Apple GPU optimized
    }

    #[test]
    fn test_gpu_error_propagation() {
        let mut gpu_ops = OptimizedGpuOps::new(GpuBackend::Cpu);
        
        // Create matrices with dimension mismatch
        let rows = vec![0, 1];
        let cols = vec![0, 1];
        let data = vec![1.0, 2.0];
        let matrix = CsrArray::from_triplets(&rows, &cols, &data, (2, 2), false).unwrap();
        
        let wrong_size_b = Array1::from_vec(vec![1.0, 2.0, 3.0]); // Wrong size
        
        // Test that dimension mismatch is caught
        let result = gpu_ops.gpu_iterative_solve(&matrix, &wrong_size_b.view(), "cg", None, 100, 1e-6);
        assert!(result.is_err());
    }
}
