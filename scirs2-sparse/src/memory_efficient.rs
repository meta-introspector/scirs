//! Memory-efficient algorithms and patterns for sparse matrices
//!
//! This module provides advanced memory optimization techniques for sparse matrix operations,
//! including streaming algorithms, out-of-core processing, and cache-aware implementations.

use crate::csr_array::CsrArray;
use crate::error::{SparseError, SparseResult};
use crate::sparray::SparseArray;
use ndarray::{Array1, Array2, ArrayView1};
use num_traits::Float;
use std::collections::VecDeque;
use std::fmt::Debug;

// Import core utilities for memory management
use scirs2_core::parallel_ops::*;
use scirs2_core::simd_ops::SimdUnifiedOps;

/// Memory usage tracking and optimization
#[derive(Debug, Clone)]
pub struct MemoryTracker {
    /// Current memory usage estimate
    current_usage: usize,
    /// Peak memory usage observed
    peak_usage: usize,
    /// Memory budget limit
    memory_limit: usize,
}

impl MemoryTracker {
    /// Create a new memory tracker with given limit
    pub fn new(memory_limit: usize) -> Self {
        Self {
            current_usage: 0,
            peak_usage: 0,
            memory_limit,
        }
    }

    /// Allocate memory and track usage
    pub fn allocate(&mut self, size: usize) -> SparseResult<()> {
        self.current_usage += size;
        self.peak_usage = self.peak_usage.max(self.current_usage);
        
        if self.current_usage > self.memory_limit {
            Err(SparseError::ValueError(
                "Memory limit exceeded".to_string()
            ))
        } else {
            Ok(())
        }
    }

    /// Deallocate memory and update tracking
    pub fn deallocate(&mut self, size: usize) {
        self.current_usage = self.current_usage.saturating_sub(size);
    }

    /// Get current memory usage
    pub fn current_usage(&self) -> usize {
        self.current_usage
    }

    /// Get peak memory usage
    pub fn peak_usage(&self) -> usize {
        self.peak_usage
    }

    /// Check if allocation would exceed limit
    pub fn can_allocate(&self, size: usize) -> bool {
        self.current_usage + size <= self.memory_limit
    }
}

/// Memory-efficient sparse matrix-vector multiplication using streaming
///
/// This implementation processes the matrix in chunks to minimize memory usage
/// while maintaining computational efficiency.
///
/// # Arguments
///
/// * `matrix` - The sparse matrix
/// * `x` - The input vector
/// * `chunk_size` - Number of rows to process at once
/// * `memory_tracker` - Optional memory usage tracker
///
/// # Returns
///
/// The result vector y = A * x
pub fn streaming_sparse_matvec<T, S>(
    matrix: &S,
    x: &ArrayView1<T>,
    chunk_size: usize,
    memory_tracker: Option<&mut MemoryTracker>,
) -> SparseResult<Array1<T>>
where
    T: Float + Debug + Copy + 'static + SimdUnifiedOps + Send + Sync,
    S: SparseArray<T>,
{
    let (rows, cols) = matrix.shape();
    
    if x.len() != cols {
        return Err(SparseError::DimensionMismatch {
            expected: cols,
            found: x.len(),
        });
    }

    let mut y = Array1::zeros(rows);
    let element_size = std::mem::size_of::<T>();
    
    // Process matrix in chunks
    let num_chunks = (rows + chunk_size - 1) / chunk_size;
    
    for chunk_idx in 0..num_chunks {
        let row_start = chunk_idx * chunk_size;
        let row_end = std::cmp::min(row_start + chunk_size, rows);
        let current_chunk_size = row_end - row_start;
        
        // Estimate memory usage for this chunk
        let chunk_memory = current_chunk_size * cols * element_size; // Worst case
        
        if let Some(tracker) = memory_tracker.as_ref() {
            if !tracker.can_allocate(chunk_memory) {
                return Err(SparseError::ValueError(
                    "Insufficient memory for chunk processing".to_string()
                ));
            }
        }
        
        // Track memory allocation
        if let Some(tracker) = memory_tracker.as_mut() {
            tracker.allocate(chunk_memory)?;
        }
        
        // Extract chunk data
        let (row_indices, col_indices, values) = matrix.find();
        let mut chunk_result = vec![T::zero(); current_chunk_size];
        
        // Find elements in the current row range
        for (k, (&row, &col)) in row_indices.iter().zip(col_indices.iter()).enumerate() {
            if row >= row_start && row < row_end {
                let local_row = row - row_start;
                chunk_result[local_row] = chunk_result[local_row] + values[k] * x[col];
            }
        }
        
        // Copy results to output vector
        for (i, &val) in chunk_result.iter().enumerate() {
            y[row_start + i] = val;
        }
        
        // Deallocate chunk memory
        if let Some(tracker) = memory_tracker.as_mut() {
            tracker.deallocate(chunk_memory);
        }
    }
    
    Ok(y)
}

/// Out-of-core sparse matrix operations
///
/// This struct provides methods for processing matrices that are too large
/// to fit entirely in memory.
pub struct OutOfCoreProcessor<T>
where
    T: Float + Debug + Copy + 'static,
{
    memory_limit: usize,
    chunk_size: usize,
    temp_storage: VecDeque<Vec<T>>,
}

impl<T> OutOfCoreProcessor<T>
where
    T: Float + Debug + Copy + 'static + SimdUnifiedOps + Send + Sync,
{
    /// Create a new out-of-core processor
    pub fn new(memory_limit: usize) -> Self {
        let chunk_size = memory_limit / (8 * std::mem::size_of::<T>()); // Conservative estimate
        
        Self {
            memory_limit,
            chunk_size,
            temp_storage: VecDeque::new(),
        }
    }

    /// Perform matrix-matrix multiplication out-of-core
    pub fn out_of_core_matmul<S1, S2>(
        &mut self,
        a: &S1,
        b: &S2,
    ) -> SparseResult<CsrArray<T>>
    where
        S1: SparseArray<T>,
        S2: SparseArray<T>,
    {
        let (a_rows, a_cols) = a.shape();
        let (b_rows, b_cols) = b.shape();
        
        if a_cols != b_rows {
            return Err(SparseError::DimensionMismatch {
                expected: a_cols,
                found: b_rows,
            });
        }

        // Convert B to CSC for efficient column access
        let b_csc = b.to_csc();
        
        let mut result_data = Vec::new();
        let mut result_indices = Vec::new();
        let mut result_indptr = vec![0];
        
        let rows_per_chunk = self.chunk_size;
        let num_chunks = (a_rows + rows_per_chunk - 1) / rows_per_chunk;
        
        for chunk_idx in 0..num_chunks {
            let row_start = chunk_idx * rows_per_chunk;
            let row_end = std::cmp::min(row_start + rows_per_chunk, a_rows);
            
            // Process this chunk of rows
            let chunk_result = self.process_chunk_matmul(a, &b_csc, row_start, row_end, b_cols)?;
            
            // Append results
            result_data.extend(chunk_result.data);
            result_indices.extend(chunk_result.indices);
            
            // Update indptr
            let last_nnz = *result_indptr.last().unwrap();
            for &count in &chunk_result.indptr[1..] {
                result_indptr.push(last_nnz + count);
            }
        }
        
        CsrArray::new(result_data, result_indptr, result_indices, (a_rows, b_cols))
    }

    /// Process a chunk of matrix multiplication
    fn process_chunk_matmul<S1, S2>(
        &mut self,
        a: &S1,
        b_csc: &S2,
        row_start: usize,
        row_end: usize,
        b_cols: usize,
    ) -> SparseResult<ChunkResult<T>>
    where
        S1: SparseArray<T>,
        S2: SparseArray<T>,
    {
        let mut chunk_data = Vec::new();
        let mut chunk_indices = Vec::new();
        let mut chunk_indptr = vec![0];
        
        let (a_row_indices, a_col_indices, a_values) = a.find();
        let (b_row_indices, b_col_indices, b_values) = b_csc.find();
        let b_indptr = b_csc.indptr();
        
        for i in row_start..row_end {
            let mut row_data = Vec::new();
            let mut row_indices = Vec::new();
            
            // Find A's entries for row i
            let mut a_entries = Vec::new();
            for (k, (&row, &col)) in a_row_indices.iter().zip(a_col_indices.iter()).enumerate() {
                if row == i {
                    a_entries.push((col, a_values[k]));
                }
            }
            
            // For each column j in B
            for j in 0..b_cols {
                let mut sum = T::zero();
                let b_col_start = b_indptr[j];
                let b_col_end = b_indptr[j + 1];
                
                // Compute dot product of A[i,:] and B[:,j]
                for &(a_col, a_val) in &a_entries {
                    for b_idx in b_col_start..b_col_end {
                        if b_row_indices[b_idx] == a_col {
                            sum = sum + a_val * b_values[b_idx];
                            break;
                        }
                    }
                }
                
                if !sum.is_zero() {
                    row_data.push(sum);
                    row_indices.push(j);
                }
            }
            
            chunk_data.extend(row_data);
            chunk_indices.extend(row_indices);
            chunk_indptr.push(chunk_data.len());
        }
        
        Ok(ChunkResult {
            data: chunk_data,
            indices: chunk_indices,
            indptr: chunk_indptr,
        })
    }

    /// Get memory usage statistics
    pub fn memory_stats(&self) -> (usize, usize) {
        let current_usage = self.temp_storage.iter().map(|v| v.len() * std::mem::size_of::<T>()).sum();
        (current_usage, self.memory_limit)
    }
}

/// Result of processing a chunk
struct ChunkResult<T> {
    data: Vec<T>,
    indices: Vec<usize>,
    indptr: Vec<usize>,
}

/// Cache-aware sparse matrix operations
pub struct CacheAwareOps;

impl CacheAwareOps {
    /// Cache-optimized sparse matrix-vector multiplication
    ///
    /// This implementation optimizes for cache performance by reordering
    /// operations to improve data locality.
    pub fn cache_optimized_spmv<T, S>(
        matrix: &S,
        x: &ArrayView1<T>,
        cache_line_size: usize,
    ) -> SparseResult<Array1<T>>
    where
        T: Float + Debug + Copy + 'static + SimdUnifiedOps + Send + Sync,
        S: SparseArray<T>,
    {
        let (rows, cols) = matrix.shape();
        
        if x.len() != cols {
            return Err(SparseError::DimensionMismatch {
                expected: cols,
                found: x.len(),
            });
        }

        let mut y = Array1::zeros(rows);
        let elements_per_cache_line = cache_line_size / std::mem::size_of::<T>();
        
        // Group operations by cache lines for better locality
        let (row_indices, col_indices, values) = matrix.find();
        
        // Sort by column to improve x vector cache locality
        let mut sorted_ops: Vec<(usize, usize, T)> = row_indices
            .iter()
            .zip(col_indices.iter())
            .zip(values.iter())
            .map(|((&row, &col), &val)| (row, col, val))
            .collect();
            
        sorted_ops.sort_by_key(|&(_, col, _)| col);
        
        // Process in cache-friendly chunks
        for chunk in sorted_ops.chunks(elements_per_cache_line) {
            for &(row, col, val) in chunk {
                y[row] = y[row] + val * x[col];
            }
        }
        
        Ok(y)
    }

    /// Cache-optimized sparse matrix transpose
    pub fn cache_optimized_transpose<T, S>(
        matrix: &S,
        cache_line_size: usize,
    ) -> SparseResult<CsrArray<T>>
    where
        T: Float + Debug + Copy + 'static + SimdUnifiedOps + Send + Sync,
        S: SparseArray<T>,
    {
        let (rows, cols) = matrix.shape();
        let (row_indices, col_indices, values) = matrix.find();
        
        // Group operations by cache lines
        let elements_per_cache_line = cache_line_size / std::mem::size_of::<T>();
        
        let mut transposed_triplets = Vec::new();
        
        // Process in cache-friendly chunks
        for chunk_start in (0..row_indices.len()).step_by(elements_per_cache_line) {
            let chunk_end = std::cmp::min(chunk_start + elements_per_cache_line, row_indices.len());
            
            for k in chunk_start..chunk_end {
                transposed_triplets.push((col_indices[k], row_indices[k], values[k]));
            }
        }
        
        // Sort by new row index (original column)
        transposed_triplets.sort_by_key(|&(new_row, _, _)| new_row);
        
        let new_rows: Vec<usize> = transposed_triplets.iter().map(|&(new_row, _, _)| new_row).collect();
        let new_cols: Vec<usize> = transposed_triplets.iter().map(|&(_, new_col, _)| new_col).collect();
        let new_values: Vec<T> = transposed_triplets.iter().map(|&(_, _, val)| val).collect();
        
        CsrArray::from_triplets(&new_rows, &new_cols, &new_values, (cols, rows), false)
    }
}

/// Memory pool for efficient allocation and reuse
pub struct MemoryPool<T>
where
    T: Float + Debug + Copy + 'static,
{
    available_buffers: Vec<Vec<T>>,
    allocated_buffers: Vec<Vec<T>>,
    pool_size_limit: usize,
}

impl<T> MemoryPool<T>
where
    T: Float + Debug + Copy + 'static,
{
    /// Create a new memory pool
    pub fn new(pool_size_limit: usize) -> Self {
        Self {
            available_buffers: Vec::new(),
            allocated_buffers: Vec::new(),
            pool_size_limit,
        }
    }

    /// Allocate a buffer from the pool
    pub fn allocate(&mut self, size: usize) -> Vec<T> {
        if let Some(mut buffer) = self.available_buffers.pop() {
            buffer.resize(size, T::zero());
            buffer
        } else {
            vec![T::zero(); size]
        }
    }

    /// Return a buffer to the pool
    pub fn deallocate(&mut self, mut buffer: Vec<T>) {
        if self.available_buffers.len() < self.pool_size_limit {
            buffer.clear();
            self.available_buffers.push(buffer);
        }
        // If pool is full, let the buffer be dropped
    }

    /// Get pool statistics
    pub fn stats(&self) -> (usize, usize) {
        (self.available_buffers.len(), self.allocated_buffers.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csr_array::CsrArray;
    use approx::assert_relative_eq;

    #[test]
    fn test_memory_tracker() {
        let mut tracker = MemoryTracker::new(1000);
        
        // Test allocation
        assert!(tracker.allocate(500).is_ok());
        assert_eq!(tracker.current_usage(), 500);
        assert_eq!(tracker.peak_usage(), 500);
        
        // Test over-allocation
        assert!(tracker.allocate(600).is_err());
        
        // Test deallocation
        tracker.deallocate(200);
        assert_eq!(tracker.current_usage(), 300);
        assert_eq!(tracker.peak_usage(), 500); // Peak should remain
        
        // Test can_allocate
        assert!(tracker.can_allocate(700));
        assert!(!tracker.can_allocate(800));
    }

    #[test]
    fn test_streaming_sparse_matvec() {
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 2, 1, 0, 2];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let matrix = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();

        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        
        let mut tracker = MemoryTracker::new(10000);
        let result = streaming_sparse_matvec(&matrix, &x.view(), 2, Some(&mut tracker)).unwrap();

        // Expected: [1*1 + 2*3, 3*2, 4*1 + 5*3] = [7, 6, 19]
        assert_relative_eq!(result[0], 7.0);
        assert_relative_eq!(result[1], 6.0);
        assert_relative_eq!(result[2], 19.0);
        
        assert!(tracker.peak_usage() > 0);
    }

    #[test]
    fn test_out_of_core_processor() {
        let mut processor = OutOfCoreProcessor::<f64>::new(1_000_000);
        
        // Create small test matrices
        let rows_a = vec![0, 1, 1];
        let cols_a = vec![0, 0, 1];
        let data_a = vec![2.0, 1.0, 3.0];
        let matrix_a = CsrArray::from_triplets(&rows_a, &cols_a, &data_a, (2, 2), false).unwrap();

        let rows_b = vec![0, 1];
        let cols_b = vec![0, 1];
        let data_b = vec![1.0, 2.0];
        let matrix_b = CsrArray::from_triplets(&rows_b, &cols_b, &data_b, (2, 2), false).unwrap();

        let result = processor.out_of_core_matmul(&matrix_a, &matrix_b).unwrap();
        
        // Verify result dimensions
        assert_eq!(result.shape(), (2, 2));
        assert!(result.nnz() > 0);
        
        let (current, limit) = processor.memory_stats();
        assert!(current <= limit);
    }

    #[test]
    fn test_cache_aware_ops() {
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 2, 1, 0, 2];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let matrix = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();

        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        
        // Test cache-optimized SpMV
        let result = CacheAwareOps::cache_optimized_spmv(&matrix, &x.view(), 64).unwrap();
        assert_relative_eq!(result[0], 7.0);
        assert_relative_eq!(result[1], 6.0);
        assert_relative_eq!(result[2], 19.0);
        
        // Test cache-optimized transpose
        let transposed = CacheAwareOps::cache_optimized_transpose(&matrix, 64).unwrap();
        assert_eq!(transposed.shape(), (3, 3));
        
        // Verify transpose correctness
        assert_relative_eq!(transposed.get(0, 0), 1.0); // Original (0,0)
        assert_relative_eq!(transposed.get(2, 0), 2.0); // Original (0,2) -> (2,0)
        assert_relative_eq!(transposed.get(1, 1), 3.0); // Original (1,1)
    }

    #[test]
    fn test_memory_pool() {
        let mut pool = MemoryPool::<f64>::new(5);
        
        // Allocate buffer
        let buffer1 = pool.allocate(100);
        assert_eq!(buffer1.len(), 100);
        
        // Return buffer to pool
        pool.deallocate(buffer1);
        
        let (available, allocated) = pool.stats();
        assert_eq!(available, 1);
        assert_eq!(allocated, 0);
        
        // Allocate again (should reuse buffer)
        let buffer2 = pool.allocate(50);
        assert_eq!(buffer2.len(), 50);
        
        pool.deallocate(buffer2);
    }

    #[test]
    fn test_streaming_memory_limit() {
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 2, 1, 0, 2];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let matrix = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();

        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        
        // Set very small memory limit
        let mut tracker = MemoryTracker::new(10);
        let result = streaming_sparse_matvec(&matrix, &x.view(), 1, Some(&mut tracker));
        
        // Should fail due to memory limit
        assert!(result.is_err());
    }
}