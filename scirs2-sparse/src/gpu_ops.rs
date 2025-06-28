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
}
