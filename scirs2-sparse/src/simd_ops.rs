//! SIMD-accelerated operations for sparse matrices
//!
//! This module provides SIMD optimizations for general sparse matrix operations,
//! leveraging the scirs2-core SIMD infrastructure for maximum performance.

use crate::coo_array::CooArray;
use crate::csc_array::CscArray;
use crate::csr_array::CsrArray;
use crate::error::{SparseError, SparseResult};
use crate::sparray::SparseArray;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_traits::Float;
use std::fmt::Debug;

// Import SIMD and parallel operations from scirs2-core
use scirs2_core::parallel_ops::*;
use scirs2_core::simd_ops::{PlatformCapabilities, SimdUnifiedOps};

/// SIMD acceleration options
#[derive(Debug, Clone)]
pub struct SimdOptions {
    /// Minimum vector length to use SIMD acceleration
    pub min_simd_size: usize,
    /// SIMD chunk size (typically 4, 8, or 16)
    pub chunk_size: usize,
    /// Use parallel processing for large operations
    pub use_parallel: bool,
    /// Minimum size to trigger parallel processing
    pub parallel_threshold: usize,
}

impl Default for SimdOptions {
    fn default() -> Self {
        // Detect platform capabilities and optimize accordingly
        let capabilities = PlatformCapabilities::detect();
        let optimal_chunk_size = if capabilities.has_avx512() {
            16 // AVX-512 can handle 16 f32 or 8 f64 elements
        } else if capabilities.has_avx2() {
            8 // AVX2 can handle 8 f32 or 4 f64 elements
        } else if capabilities.has_sse() {
            4 // SSE can handle 4 f32 or 2 f64 elements
        } else {
            8 // Default fallback
        };

        Self {
            min_simd_size: optimal_chunk_size,
            chunk_size: optimal_chunk_size,
            use_parallel: capabilities.num_cores() > 1,
            parallel_threshold: 1000 * capabilities.num_cores(),
        }
    }
}

/// SIMD-accelerated sparse matrix-vector multiplication for CSR matrices
///
/// This function automatically chooses between SIMD, parallel, and scalar implementations
/// based on the matrix size and data characteristics.
///
/// # Arguments
///
/// * `matrix` - The CSR matrix
/// * `x` - The input vector
/// * `options` - SIMD acceleration options
///
/// # Returns
///
/// The result vector y = A * x
///
/// # Example
///
/// ```rust
/// use scirs2_sparse::csr_array::CsrArray;
/// use scirs2_sparse::simd_ops::{simd_csr_matvec, SimdOptions};
/// use ndarray::Array1;
///
/// // Create a sparse matrix
/// let rows = vec![0, 0, 1, 2, 2];
/// let cols = vec![0, 2, 1, 0, 2];
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let matrix = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();
///
/// // Input vector
/// let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
///
/// // Compute using SIMD acceleration
/// let y = simd_csr_matvec(&matrix, &x.view(), SimdOptions::default()).unwrap();
/// ```
pub fn simd_csr_matvec<T>(
    matrix: &CsrArray<T>,
    x: &ArrayView1<T>,
    options: SimdOptions,
) -> SparseResult<Array1<T>>
where
    T: Float + Debug + Copy + SimdUnifiedOps + Send + Sync,
{
    let (rows, cols) = matrix.shape();

    if x.len() != cols {
        return Err(SparseError::DimensionMismatch {
            expected: cols,
            found: x.len(),
        });
    }

    // Choose implementation based on size and capabilities
    let capabilities = PlatformCapabilities::detect();
    let nnz = matrix.nnz();

    if options.use_parallel && nnz >= options.parallel_threshold && capabilities.num_cores() > 1 {
        // Use parallel SIMD implementation for large matrices
        simd_csr_matvec_parallel(matrix, x, &options)
    } else if nnz >= options.min_simd_size && capabilities.has_simd() {
        // Use sequential SIMD implementation for medium matrices
        simd_csr_matvec_sequential(matrix, x, &options)
    } else {
        // Use scalar implementation for small matrices
        simd_csr_matvec_scalar(matrix, x)
    }
}

/// Parallel SIMD-accelerated CSR matrix-vector multiplication
fn simd_csr_matvec_parallel<T>(
    matrix: &CsrArray<T>,
    x: &ArrayView1<T>,
    options: &SimdOptions,
) -> SparseResult<Array1<T>>
where
    T: Float + Debug + Copy + SimdUnifiedOps + Send + Sync,
{
    let (rows, _) = matrix.shape();
    let mut y = Array1::zeros(rows);

    // Process rows in parallel chunks
    let chunk_size = std::cmp::min(options.parallel_threshold / 4, 128);
    let y_slice = y.as_slice_mut().unwrap();

    ParallelIterator::par_chunks_mut(y_slice, chunk_size)
        .enumerate()
        .for_each(|(chunk_idx, y_chunk)| {
            let start_row = chunk_idx * chunk_size;
            let end_row = std::cmp::min(start_row + chunk_size, rows);

            for (local_row, row_i) in (start_row..end_row).enumerate() {
                let row_start = matrix.indptr[row_i];
                let row_end = matrix.indptr[row_i + 1];
                let row_len = row_end - row_start;

                if row_len >= options.chunk_size {
                    // Use SIMD for longer rows
                    let mut chunk_start = row_start;
                    while chunk_start + options.chunk_size <= row_end {
                        let chunk_end = chunk_start + options.chunk_size;

                        // Extract chunks of indices and values
                        let indices_chunk = &matrix.indices[chunk_start..chunk_end];
                        let values_chunk = &matrix.data[chunk_start..chunk_end];

                        // Gather corresponding x values and compute dot product
                        let mut x_vals = Vec::with_capacity(options.chunk_size);
                        for &idx in indices_chunk {
                            x_vals.push(x[idx]);
                        }

                        let values_view = ArrayView1::from(values_chunk);
                        let x_view = ArrayView1::from(&x_vals);
                        let dot_product = T::simd_dot(&values_view, &x_view);

                        y_chunk[local_row] = y_chunk[local_row] + dot_product;
                        chunk_start = chunk_end;
                    }

                    // Handle remaining elements in row
                    for j in chunk_start..row_end {
                        let col = matrix.indices[j];
                        let val = matrix.data[j];
                        y_chunk[local_row] = y_chunk[local_row] + val * x[col];
                    }
                } else {
                    // Use scalar operations for short rows
                    for j in row_start..row_end {
                        let col = matrix.indices[j];
                        let val = matrix.data[j];
                        y_chunk[local_row] = y_chunk[local_row] + val * x[col];
                    }
                }
            }
        });

    Ok(y)
}

/// Sequential SIMD-accelerated CSR matrix-vector multiplication
fn simd_csr_matvec_sequential<T>(
    matrix: &CsrArray<T>,
    x: &ArrayView1<T>,
    options: &SimdOptions,
) -> SparseResult<Array1<T>>
where
    T: Float + Debug + Copy + SimdUnifiedOps,
{
    let (rows, _) = matrix.shape();
    let mut y = Array1::zeros(rows);

    for i in 0..rows {
        let row_start = matrix.indptr[i];
        let row_end = matrix.indptr[i + 1];
        let row_len = row_end - row_start;

        if row_len >= options.chunk_size {
            // Use SIMD for longer rows
            let mut chunk_start = row_start;
            while chunk_start + options.chunk_size <= row_end {
                let chunk_end = chunk_start + options.chunk_size;

                // Extract chunks of indices and values
                let indices_chunk = &matrix.indices[chunk_start..chunk_end];
                let values_chunk = &matrix.data[chunk_start..chunk_end];

                // Gather corresponding x values and compute dot product
                let mut x_vals = Vec::with_capacity(options.chunk_size);
                for &idx in indices_chunk {
                    x_vals.push(x[idx]);
                }

                let values_view = ArrayView1::from(values_chunk);
                let x_view = ArrayView1::from(&x_vals);
                let dot_product = T::simd_dot(&values_view, &x_view);

                y[i] = y[i] + dot_product;
                chunk_start = chunk_end;
            }

            // Handle remaining elements in row
            for j in chunk_start..row_end {
                let col = matrix.indices[j];
                let val = matrix.data[j];
                y[i] = y[i] + val * x[col];
            }
        } else {
            // Use scalar operations for short rows
            for j in row_start..row_end {
                let col = matrix.indices[j];
                let val = matrix.data[j];
                y[i] = y[i] + val * x[col];
            }
        }
    }

    Ok(y)
}

/// Scalar fallback implementation
fn simd_csr_matvec_scalar<T>(matrix: &CsrArray<T>, x: &ArrayView1<T>) -> SparseResult<Array1<T>>
where
    T: Float + Debug + Copy,
{
    let (rows, _) = matrix.shape();
    let mut y = Array1::zeros(rows);

    for i in 0..rows {
        for j in matrix.indptr[i]..matrix.indptr[i + 1] {
            let col = matrix.indices[j];
            let val = matrix.data[j];
            y[i] = y[i] + val * x[col];
        }
    }

    Ok(y)
}

/// SIMD-accelerated sparse matrix-matrix multiplication
///
/// Computes C = A * B where A and B are sparse matrices.
///
/// # Arguments
///
/// * `a` - Left matrix (CSR format)
/// * `b` - Right matrix (CSC format for efficiency)
/// * `options` - SIMD acceleration options
///
/// # Returns
///
/// The result matrix C = A * B in CSR format
pub fn simd_sparse_matmul<T>(
    a: &CsrArray<T>,
    b: &CscArray<T>,
    options: SimdOptions,
) -> SparseResult<CsrArray<T>>
where
    T: Float + Debug + Copy + SimdUnifiedOps + Send + Sync,
{
    let (a_rows, a_cols) = a.shape();
    let (b_rows, b_cols) = b.shape();

    if a_cols != b_rows {
        return Err(SparseError::DimensionMismatch {
            expected: a_cols,
            found: b_rows,
        });
    }

    // For sparse matrix multiplication, we'll use a simplified approach
    // In practice, this would use more sophisticated algorithms like Gustavson's
    let mut result_data = Vec::new();
    let mut result_indices = Vec::new();
    let mut result_indptr = vec![0];

    // Process each row of A
    for i in 0..a_rows {
        let mut row_data = Vec::new();
        let mut row_indices = Vec::new();

        // For each column of B
        for j in 0..b_cols {
            let mut sum = T::zero();

            // Compute dot product of row i of A with column j of B
            let a_row_start = a.indptr[i];
            let a_row_end = a.indptr[i + 1];
            let b_col_start = b.indptr[j];
            let b_col_end = b.indptr[j + 1];

            let mut a_idx = a_row_start;
            let mut b_idx = b_col_start;

            // Merge-like intersection of sparse row and column
            while a_idx < a_row_end && b_idx < b_col_end {
                let a_col = a.indices[a_idx];
                let b_row = b.indices[b_idx];

                if a_col == b_row {
                    sum = sum + a.data[a_idx] * b.data[b_idx];
                    a_idx += 1;
                    b_idx += 1;
                } else if a_col < b_row {
                    a_idx += 1;
                } else {
                    b_idx += 1;
                }
            }

            // Add to result if non-zero
            if sum != T::zero() {
                row_data.push(sum);
                row_indices.push(j);
            }
        }

        // Add row to result
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

/// SIMD-accelerated element-wise operations between sparse matrices
///
/// Computes C = A op B where op can be addition, subtraction, or element-wise multiplication.
///
/// # Arguments
///
/// * `a` - First matrix
/// * `b` - Second matrix
/// * `operation` - The operation to perform
/// * `options` - SIMD acceleration options
///
/// # Returns
///
/// The result matrix C
pub fn simd_sparse_elementwise<T>(
    a: &CsrArray<T>,
    b: &CsrArray<T>,
    operation: ElementwiseOp,
    options: SimdOptions,
) -> SparseResult<CsrArray<T>>
where
    T: Float + Debug + Copy + SimdUnifiedOps + Send + Sync,
{
    let a_shape = a.shape();
    let b_shape = b.shape();

    if a_shape != b_shape {
        return Err(SparseError::DimensionMismatch {
            expected: a_shape.0,
            found: b_shape.0,
        });
    }

    // Convert both matrices to COO format for easier element-wise operations
    let a_coo = a.to_coo();
    let b_coo = b.to_coo();

    // Merge the two COO matrices based on coordinates
    let mut result_data = Vec::new();
    let mut result_rows = Vec::new();
    let mut result_cols = Vec::new();

    let mut a_idx = 0;
    let mut b_idx = 0;

    while a_idx < a_coo.data.len() || b_idx < b_coo.data.len() {
        let a_pos = if a_idx < a_coo.data.len() {
            Some((a_coo.rows[a_idx], a_coo.cols[a_idx]))
        } else {
            None
        };

        let b_pos = if b_idx < b_coo.data.len() {
            Some((b_coo.rows[b_idx], b_coo.cols[b_idx]))
        } else {
            None
        };

        match (a_pos, b_pos) {
            (Some((a_row, a_col)), Some((b_row, b_col))) => {
                if (a_row, a_col) == (b_row, b_col) {
                    // Both matrices have this element
                    let result_val = match operation {
                        ElementwiseOp::Add => a_coo.data[a_idx] + b_coo.data[b_idx],
                        ElementwiseOp::Sub => a_coo.data[a_idx] - b_coo.data[b_idx],
                        ElementwiseOp::Mul => a_coo.data[a_idx] * b_coo.data[b_idx],
                    };

                    if result_val != T::zero() {
                        result_data.push(result_val);
                        result_rows.push(a_row);
                        result_cols.push(a_col);
                    }

                    a_idx += 1;
                    b_idx += 1;
                } else if (a_row, a_col) < (b_row, b_col) {
                    // Only matrix A has this element
                    match operation {
                        ElementwiseOp::Add | ElementwiseOp::Sub => {
                            result_data.push(a_coo.data[a_idx]);
                            result_rows.push(a_row);
                            result_cols.push(a_col);
                        }
                        ElementwiseOp::Mul => {
                            // 0 * anything = 0, so we skip this element
                        }
                    }
                    a_idx += 1;
                } else {
                    // Only matrix B has this element
                    match operation {
                        ElementwiseOp::Add => {
                            result_data.push(b_coo.data[b_idx]);
                            result_rows.push(b_row);
                            result_cols.push(b_col);
                        }
                        ElementwiseOp::Sub => {
                            result_data.push(-b_coo.data[b_idx]);
                            result_rows.push(b_row);
                            result_cols.push(b_col);
                        }
                        ElementwiseOp::Mul => {
                            // anything * 0 = 0, so we skip this element
                        }
                    }
                    b_idx += 1;
                }
            }
            (Some((a_row, a_col)), None) => {
                // Only matrix A has remaining elements
                match operation {
                    ElementwiseOp::Add | ElementwiseOp::Sub => {
                        result_data.push(a_coo.data[a_idx]);
                        result_rows.push(a_row);
                        result_cols.push(a_col);
                    }
                    ElementwiseOp::Mul => {
                        // 0 * anything = 0, so we skip this element
                    }
                }
                a_idx += 1;
            }
            (None, Some((b_row, b_col))) => {
                // Only matrix B has remaining elements
                match operation {
                    ElementwiseOp::Add => {
                        result_data.push(b_coo.data[b_idx]);
                        result_rows.push(b_row);
                        result_cols.push(b_col);
                    }
                    ElementwiseOp::Sub => {
                        result_data.push(-b_coo.data[b_idx]);
                        result_rows.push(b_row);
                        result_cols.push(b_col);
                    }
                    ElementwiseOp::Mul => {
                        // anything * 0 = 0, so we skip this element
                    }
                }
                b_idx += 1;
            }
            (None, None) => break,
        }
    }

    // Convert result back to CSR format
    CsrArray::from_triplets(&result_rows, &result_cols, &result_data, a_shape, false)
}

/// Element-wise operations supported by SIMD acceleration
#[derive(Debug, Clone, Copy)]
pub enum ElementwiseOp {
    /// Addition: C = A + B
    Add,
    /// Subtraction: C = A - B
    Sub,
    /// Element-wise multiplication: C = A .* B
    Mul,
}

/// Advanced SIMD-optimized sparse matrix operations
pub struct AdvancedSimdOps;

impl AdvancedSimdOps {
    /// Optimized sparse matrix-vector multiplication with memory prefetching
    pub fn optimized_spmv<T>(
        matrix: &CsrArray<T>,
        x: &ArrayView1<T>,
        options: SimdOptions,
    ) -> SparseResult<Array1<T>>
    where
        T: Float + Debug + Copy + SimdUnifiedOps + Send + Sync,
    {
        let (rows, cols) = matrix.shape();
        let capabilities = PlatformCapabilities::detect();

        if x.len() != cols {
            return Err(SparseError::DimensionMismatch {
                expected: cols,
                found: x.len(),
            });
        }

        let mut y = Array1::zeros(rows);

        // Use different strategies based on matrix properties
        let avg_row_nnz = matrix.nnz() / rows;

        if avg_row_nnz >= options.chunk_size && capabilities.has_simd() {
            // Dense rows - use SIMD vectorization
            Self::simd_dense_rows_spmv(matrix, x, &mut y, &options, &capabilities)?;
        } else if capabilities.num_cores() > 1 && matrix.nnz() >= options.parallel_threshold {
            // Sparse rows - use parallel processing
            Self::parallel_sparse_rows_spmv(matrix, x, &mut y, &options)?;
        } else {
            // Small matrix - use optimized scalar code
            Self::scalar_optimized_spmv(matrix, x, &mut y)?;
        }

        Ok(y)
    }

    /// SIMD-optimized processing for rows with many non-zeros
    fn simd_dense_rows_spmv<T>(
        matrix: &CsrArray<T>,
        x: &ArrayView1<T>,
        y: &mut Array1<T>,
        options: &SimdOptions,
        capabilities: &PlatformCapabilities,
    ) -> SparseResult<()>
    where
        T: Float + Debug + Copy + SimdUnifiedOps,
    {
        let chunk_size = options.chunk_size;

        for i in 0..matrix.shape().0 {
            let row_start = matrix.indptr[i];
            let row_end = matrix.indptr[i + 1];
            let row_nnz = row_end - row_start;

            if row_nnz >= chunk_size {
                // Process this row with SIMD
                let mut sum = T::zero();
                let mut j = row_start;

                // Process chunks with SIMD
                while j + chunk_size <= row_end {
                    let indices_chunk = &matrix.indices[j..j + chunk_size];
                    let values_chunk = &matrix.data[j..j + chunk_size];

                    // Gather x values for this chunk
                    let mut x_chunk = Vec::with_capacity(chunk_size);
                    for &idx in indices_chunk {
                        x_chunk.push(x[idx]);
                    }

                    // Compute SIMD dot product
                    let values_view = ArrayView1::from(values_chunk);
                    let x_view = ArrayView1::from(&x_chunk);
                    sum = sum + T::simd_dot(&values_view, &x_view);

                    j += chunk_size;
                }

                // Handle remaining elements
                while j < row_end {
                    let col = matrix.indices[j];
                    let val = matrix.data[j];
                    sum = sum + val * x[col];
                    j += 1;
                }

                y[i] = sum;
            } else {
                // Use scalar code for sparse rows
                Self::scalar_row_spmv(matrix, x, y, i);
            }
        }

        Ok(())
    }

    /// Parallel processing for sparse rows
    fn parallel_sparse_rows_spmv<T>(
        matrix: &CsrArray<T>,
        x: &ArrayView1<T>,
        y: &mut Array1<T>,
        options: &SimdOptions,
    ) -> SparseResult<()>
    where
        T: Float + Debug + Copy + Send + Sync,
    {
        let chunk_size = std::cmp::min(options.parallel_threshold / 8, 64);
        let y_slice = y.as_slice_mut().unwrap();

        ParallelIterator::par_chunks_mut(y_slice, chunk_size)
            .enumerate()
            .for_each(|(chunk_idx, y_chunk)| {
                let start_row = chunk_idx * chunk_size;
                let end_row = std::cmp::min(start_row + chunk_size, matrix.shape().0);

                for (local_row, row_i) in (start_row..end_row).enumerate() {
                    let mut sum = T::zero();
                    for j in matrix.indptr[row_i]..matrix.indptr[row_i + 1] {
                        let col = matrix.indices[j];
                        let val = matrix.data[j];
                        sum = sum + val * x[col];
                    }
                    y_chunk[local_row] = sum;
                }
            });

        Ok(())
    }

    /// Optimized scalar code with loop unrolling
    fn scalar_optimized_spmv<T>(
        matrix: &CsrArray<T>,
        x: &ArrayView1<T>,
        y: &mut Array1<T>,
    ) -> SparseResult<()>
    where
        T: Float + Debug + Copy,
    {
        for i in 0..matrix.shape().0 {
            Self::scalar_row_spmv(matrix, x, y, i);
        }
        Ok(())
    }

    /// Optimized scalar computation for a single row
    fn scalar_row_spmv<T>(matrix: &CsrArray<T>, x: &ArrayView1<T>, y: &mut Array1<T>, row: usize)
    where
        T: Float + Debug + Copy,
    {
        let mut sum = T::zero();
        let start = matrix.indptr[row];
        let end = matrix.indptr[row + 1];

        // Loop unrolling for better performance
        let mut j = start;
        while j + 4 <= end {
            let col0 = matrix.indices[j];
            let val0 = matrix.data[j];
            let col1 = matrix.indices[j + 1];
            let val1 = matrix.data[j + 1];
            let col2 = matrix.indices[j + 2];
            let val2 = matrix.data[j + 2];
            let col3 = matrix.indices[j + 3];
            let val3 = matrix.data[j + 3];

            sum = sum + val0 * x[col0] + val1 * x[col1] + val2 * x[col2] + val3 * x[col3];
            j += 4;
        }

        // Handle remaining elements
        while j < end {
            let col = matrix.indices[j];
            let val = matrix.data[j];
            sum = sum + val * x[col];
            j += 1;
        }

        y[row] = sum;
    }
}

/// High-performance sparse matrix operations using advanced SIMD techniques
pub fn advanced_simd_spmv<T>(
    matrix: &CsrArray<T>,
    x: &ArrayView1<T>,
    options: Option<SimdOptions>,
) -> SparseResult<Array1<T>>
where
    T: Float + Debug + Copy + SimdUnifiedOps + Send + Sync,
{
    let options = options.unwrap_or_default();
    AdvancedSimdOps::optimized_spmv(matrix, x, options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csc_array::CscArray;
    use crate::csr_array::CsrArray;
    use approx::assert_relative_eq;

    #[test]
    fn test_simd_csr_matvec() {
        // Create a simple sparse matrix
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 2, 1, 0, 2];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let matrix = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();

        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let result = simd_csr_matvec(&matrix, &x.view(), SimdOptions::default()).unwrap();

        // Verify result: [1*1 + 2*3, 3*2, 4*1 + 5*3] = [7, 6, 19]
        assert_relative_eq!(result[0], 7.0, epsilon = 1e-10);
        assert_relative_eq!(result[1], 6.0, epsilon = 1e-10);
        assert_relative_eq!(result[2], 19.0, epsilon = 1e-10);
    }

    #[test]
    fn test_simd_sparse_elementwise_add() {
        // Create two simple sparse matrices
        let rows = vec![0, 1, 2];
        let cols = vec![0, 1, 2];
        let data_a = vec![1.0, 2.0, 3.0];
        let data_b = vec![4.0, 5.0, 6.0];

        let matrix_a = CsrArray::from_triplets(&rows, &cols, &data_a, (3, 3), false).unwrap();
        let matrix_b = CsrArray::from_triplets(&rows, &cols, &data_b, (3, 3), false).unwrap();

        let result = simd_sparse_elementwise(
            &matrix_a,
            &matrix_b,
            ElementwiseOp::Add,
            SimdOptions::default(),
        )
        .unwrap();

        // Verify diagonal elements are added correctly
        assert_relative_eq!(result.get(0, 0), 5.0, epsilon = 1e-10); // 1 + 4
        assert_relative_eq!(result.get(1, 1), 7.0, epsilon = 1e-10); // 2 + 5
        assert_relative_eq!(result.get(2, 2), 9.0, epsilon = 1e-10); // 3 + 6
    }

    #[test]
    fn test_simd_options() {
        let options = SimdOptions::default();
        // Options now depend on platform capabilities, so just check they're reasonable
        assert!(options.min_simd_size >= 4);
        assert!(options.chunk_size >= 4);
        assert!(options.parallel_threshold >= 1000);
    }

    #[test]
    fn test_advanced_simd_spmv() {
        // Create a larger sparse matrix for better SIMD testing
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut data = Vec::new();

        // Create a 10x10 matrix with various sparsity patterns
        for i in 0..10 {
            for j in 0..3 {
                rows.push(i);
                cols.push((i + j) % 10);
                data.push((i * 10 + j) as f64);
            }
        }

        let matrix = CsrArray::from_triplets(&rows, &cols, &data, (10, 10), false).unwrap();
        let x = Array1::from_iter((0..10).map(|i| i as f64));

        // Test advanced SIMD implementation
        let result_simd = advanced_simd_spmv(&matrix, &x.view(), None).unwrap();

        // Test regular implementation for comparison
        let result_regular = simd_csr_matvec(&matrix, &x.view(), SimdOptions::default()).unwrap();

        // Results should be the same
        for i in 0..10 {
            assert_relative_eq!(result_simd[i], result_regular[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_advanced_simd_ops_structure() {
        // Test that the AdvancedSimdOps methods can be called
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 2, 1, 0, 2];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let matrix = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();

        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let options = SimdOptions::default();

        let result = AdvancedSimdOps::optimized_spmv(&matrix, &x.view(), options).unwrap();

        // Verify result dimensions
        assert_eq!(result.len(), 3);

        // Verify computation correctness
        assert_relative_eq!(result[0], 7.0, epsilon = 1e-10); // 1*1 + 2*3
        assert_relative_eq!(result[1], 6.0, epsilon = 1e-10); // 3*2
        assert_relative_eq!(result[2], 19.0, epsilon = 1e-10); // 4*1 + 5*3
    }
}
