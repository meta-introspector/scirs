//! Sparse matrix representations for memory-efficient text processing
//!
//! This module provides sparse matrix implementations optimized for text data
//! where most values are zero (common in TF-IDF and count vectorization).

use crate::error::{Result, TextError};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Compressed Sparse Row (CSR) matrix representation
#[derive(Debug, Clone)]
pub struct CsrMatrix {
    /// Non-zero values
    values: Vec<f64>,
    /// Column indices for each value
    col_indices: Vec<usize>,
    /// Row pointers (cumulative sum of non-zeros per row)
    row_ptrs: Vec<usize>,
    /// Number of rows
    n_rows: usize,
    /// Number of columns
    n_cols: usize,
}

impl CsrMatrix {
    /// Create a new CSR matrix from dense representation
    pub fn from_dense(dense: &Array2<f64>) -> Self {
        let (n_rows, n_cols) = dense.dim();
        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_ptrs = vec![0];
        
        for row in dense.rows() {
            for (col_idx, &value) in row.iter().enumerate() {
                if value != 0.0 {
                    values.push(value);
                    col_indices.push(col_idx);
                }
            }
            row_ptrs.push(values.len());
        }
        
        Self {
            values,
            col_indices,
            row_ptrs,
            n_rows,
            n_cols,
        }
    }
    
    /// Create an empty CSR matrix
    pub fn zeros(n_rows: usize, n_cols: usize) -> Self {
        Self {
            values: Vec::new(),
            col_indices: Vec::new(),
            row_ptrs: vec![0; n_rows + 1],
            n_rows,
            n_cols,
        }
    }
    
    /// Get the shape of the matrix
    pub fn shape(&self) -> (usize, usize) {
        (self.n_rows, self.n_cols)
    }
    
    /// Get the number of non-zero elements
    pub fn nnz(&self) -> usize {
        self.values.len()
    }
    
    /// Get memory usage in bytes (approximate)
    pub fn memory_usage(&self) -> usize {
        self.values.len() * std::mem::size_of::<f64>() +
        self.col_indices.len() * std::mem::size_of::<usize>() +
        self.row_ptrs.len() * std::mem::size_of::<usize>()
    }
    
    /// Convert to dense representation
    pub fn to_dense(&self) -> Array2<f64> {
        let mut dense = Array2::zeros((self.n_rows, self.n_cols));
        
        for row_idx in 0..self.n_rows {
            let start = self.row_ptrs[row_idx];
            let end = self.row_ptrs[row_idx + 1];
            
            for i in start..end {
                let col_idx = self.col_indices[i];
                let value = self.values[i];
                dense[[row_idx, col_idx]] = value;
            }
        }
        
        dense
    }
    
    /// Get a specific row as a sparse vector
    pub fn get_row(&self, row_idx: usize) -> Result<SparseVector> {
        if row_idx >= self.n_rows {
            return Err(TextError::InvalidInput(
                format!("Row index {} out of bounds for matrix with {} rows", row_idx, self.n_rows)
            ));
        }
        
        let start = self.row_ptrs[row_idx];
        let end = self.row_ptrs[row_idx + 1];
        
        let indices: Vec<usize> = self.col_indices[start..end].to_vec();
        let values: Vec<f64> = self.values[start..end].to_vec();
        
        Ok(SparseVector {
            indices,
            values,
            size: self.n_cols,
        })
    }
    
    /// Multiply by a dense vector
    pub fn dot(&self, vector: &Array1<f64>) -> Result<Array1<f64>> {
        if vector.len() != self.n_cols {
            return Err(TextError::InvalidInput(
                format!("Vector dimension {} doesn't match matrix columns {}", 
                       vector.len(), self.n_cols)
            ));
        }
        
        let mut result = Array1::zeros(self.n_rows);
        
        for row_idx in 0..self.n_rows {
            let start = self.row_ptrs[row_idx];
            let end = self.row_ptrs[row_idx + 1];
            
            let mut sum = 0.0;
            for i in start..end {
                let col_idx = self.col_indices[i];
                sum += self.values[i] * vector[col_idx];
            }
            result[row_idx] = sum;
        }
        
        Ok(result)
    }
}

/// Sparse vector representation using index-value pairs
#[derive(Debug, Clone)]
pub struct SparseVector {
    /// Indices of non-zero elements
    indices: Vec<usize>,
    /// Values of non-zero elements
    values: Vec<f64>,
    /// Total size of the vector
    size: usize,
}

impl SparseVector {
    /// Create a new sparse vector
    pub fn new(size: usize) -> Self {
        Self {
            indices: Vec::new(),
            values: Vec::new(),
            size,
        }
    }
    
    /// Create from a dense vector
    pub fn from_dense(dense: &Array1<f64>) -> Self {
        let mut indices = Vec::new();
        let mut values = Vec::new();
        
        for (idx, &value) in dense.iter().enumerate() {
            if value != 0.0 {
                indices.push(idx);
                values.push(value);
            }
        }
        
        Self {
            indices,
            values,
            size: dense.len(),
        }
    }
    
    /// Create from indices and values
    pub fn from_indices_values(indices: Vec<usize>, values: Vec<f64>, size: usize) -> Self {
        assert_eq!(indices.len(), values.len(), "Indices and values must have the same length");
        Self {
            indices,
            values,
            size,
        }
    }
    
    /// Convert to dense representation
    pub fn to_dense(&self) -> Array1<f64> {
        let mut dense = Array1::zeros(self.size);
        
        for (&idx, &value) in self.indices.iter().zip(self.values.iter()) {
            dense[idx] = value;
        }
        
        dense
    }
    
    /// Get the number of non-zero elements
    pub fn nnz(&self) -> usize {
        self.values.len()
    }
    
    /// Get the size of the vector
    pub fn size(&self) -> usize {
        self.size
    }
    
    /// Compute dot product with another sparse vector
    pub fn dot_sparse(&self, other: &SparseVector) -> Result<f64> {
        if self.size != other.size {
            return Err(TextError::InvalidInput(
                format!("Vector dimensions don't match: {} vs {}", self.size, other.size)
            ));
        }
        
        let mut result = 0.0;
        let mut i = 0;
        let mut j = 0;
        
        // Merge-like algorithm for sorted indices
        while i < self.indices.len() && j < other.indices.len() {
            if self.indices[i] == other.indices[j] {
                result += self.values[i] * other.values[j];
                i += 1;
                j += 1;
            } else if self.indices[i] < other.indices[j] {
                i += 1;
            } else {
                j += 1;
            }
        }
        
        Ok(result)
    }
    
    /// Compute L2 norm
    pub fn norm(&self) -> f64 {
        self.values.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }
    
    /// Scale by a constant
    pub fn scale(&mut self, scalar: f64) {
        for value in &mut self.values {
            *value *= scalar;
        }
    }
    
    /// Get the indices of non-zero elements
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }
    
    /// Get the values of non-zero elements
    pub fn values(&self) -> &[f64] {
        &self.values
    }
    
    /// Get mutable reference to values
    pub fn values_mut(&mut self) -> &mut [f64] {
        &mut self.values
    }
}

/// Dictionary of Keys (DOK) format for building sparse matrices
#[derive(Debug, Clone)]
pub struct DokMatrix {
    data: HashMap<(usize, usize), f64>,
    n_rows: usize,
    n_cols: usize,
}

impl DokMatrix {
    /// Create a new DOK matrix
    pub fn new(n_rows: usize, n_cols: usize) -> Self {
        Self {
            data: HashMap::new(),
            n_rows,
            n_cols,
        }
    }
    
    /// Set a value at the given position
    pub fn set(&mut self, row: usize, col: usize, value: f64) -> Result<()> {
        if row >= self.n_rows || col >= self.n_cols {
            return Err(TextError::InvalidInput(
                format!("Index ({}, {}) out of bounds for matrix shape ({}, {})", 
                       row, col, self.n_rows, self.n_cols)
            ));
        }
        
        if value != 0.0 {
            self.data.insert((row, col), value);
        } else {
            self.data.remove(&(row, col));
        }
        
        Ok(())
    }
    
    /// Get a value at the given position
    pub fn get(&self, row: usize, col: usize) -> f64 {
        *self.data.get(&(row, col)).unwrap_or(&0.0)
    }
    
    /// Convert to CSR format
    pub fn to_csr(&self) -> CsrMatrix {
        let mut entries: Vec<((usize, usize), f64)> = self.data.iter()
            .map(|(&k, &v)| (k, v))
            .collect();
        
        // Sort by row, then by column
        entries.sort_by_key(|&((r, c), _)| (r, c));
        
        let mut values = Vec::with_capacity(entries.len());
        let mut col_indices = Vec::with_capacity(entries.len());
        let mut row_ptrs = vec![0];
        
        let mut current_row = 0;
        
        for ((row, col), value) in entries {
            while current_row < row {
                row_ptrs.push(values.len());
                current_row += 1;
            }
            
            values.push(value);
            col_indices.push(col);
        }
        
        while current_row < self.n_rows {
            row_ptrs.push(values.len());
            current_row += 1;
        }
        
        CsrMatrix {
            values,
            col_indices,
            row_ptrs,
            n_rows: self.n_rows,
            n_cols: self.n_cols,
        }
    }
    
    /// Get the number of non-zero elements
    pub fn nnz(&self) -> usize {
        self.data.len()
    }
}

/// Builder for efficiently constructing sparse matrices row by row
pub struct SparseMatrixBuilder {
    rows: Vec<SparseVector>,
    n_cols: usize,
}

impl SparseMatrixBuilder {
    /// Create a new builder
    pub fn new(n_cols: usize) -> Self {
        Self {
            rows: Vec::new(),
            n_cols,
        }
    }
    
    /// Add a row to the matrix
    pub fn add_row(&mut self, row: SparseVector) -> Result<()> {
        if row.size() != self.n_cols {
            return Err(TextError::InvalidInput(
                format!("Row size {} doesn't match expected columns {}", 
                       row.size(), self.n_cols)
            ));
        }
        
        self.rows.push(row);
        Ok(())
    }
    
    /// Build the final CSR matrix
    pub fn build(self) -> CsrMatrix {
        let n_rows = self.rows.len();
        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_ptrs = vec![0];
        
        for row in self.rows {
            values.extend(row.values);
            col_indices.extend(row.indices);
            row_ptrs.push(values.len());
        }
        
        CsrMatrix {
            values,
            col_indices,
            row_ptrs,
            n_rows,
            n_cols: self.n_cols,
        }
    }
}

/// COO (Coordinate) sparse matrix format for efficient construction
#[derive(Debug, Clone)]
pub struct CooMatrix {
    /// Row indices
    row_indices: Vec<usize>,
    /// Column indices
    col_indices: Vec<usize>,
    /// Values
    values: Vec<f64>,
    /// Number of rows
    n_rows: usize,
    /// Number of columns
    n_cols: usize,
}

impl CooMatrix {
    /// Create a new COO matrix
    pub fn new(n_rows: usize, n_cols: usize) -> Self {
        Self {
            row_indices: Vec::new(),
            col_indices: Vec::new(),
            values: Vec::new(),
            n_rows,
            n_cols,
        }
    }
    
    /// Add a value to the matrix
    pub fn push(&mut self, row: usize, col: usize, value: f64) -> Result<()> {
        if row >= self.n_rows || col >= self.n_cols {
            return Err(TextError::InvalidInput(
                format!("Index ({}, {}) out of bounds for matrix shape ({}, {})", 
                       row, col, self.n_rows, self.n_cols)
            ));
        }
        
        if value != 0.0 {
            self.row_indices.push(row);
            self.col_indices.push(col);
            self.values.push(value);
        }
        
        Ok(())
    }
    
    /// Convert to CSR format (efficient for row operations)
    pub fn to_csr(&self) -> CsrMatrix {
        // Sort by row then column
        let mut indices: Vec<usize> = (0..self.values.len()).collect();
        indices.sort_by_key(|&i| (self.row_indices[i], self.col_indices[i]));
        
        let mut values = Vec::with_capacity(self.values.len());
        let mut col_indices = Vec::with_capacity(self.values.len());
        let mut row_ptrs = vec![0];
        
        let mut current_row = 0;
        for &idx in &indices {
            let row = self.row_indices[idx];
            
            // Fill row pointers for empty rows
            while current_row < row {
                row_ptrs.push(values.len());
                current_row += 1;
            }
            
            values.push(self.values[idx]);
            col_indices.push(self.col_indices[idx]);
        }
        
        // Fill remaining row pointers
        while row_ptrs.len() <= self.n_rows {
            row_ptrs.push(values.len());
        }
        
        CsrMatrix {
            values,
            col_indices,
            row_ptrs,
            n_rows: self.n_rows,
            n_cols: self.n_cols,
        }
    }
    
    /// Get number of non-zero elements
    pub fn nnz(&self) -> usize {
        self.values.len()
    }
}

/// CSC (Compressed Sparse Column) format for efficient column operations
#[derive(Debug, Clone)]
pub struct CscMatrix {
    /// Non-zero values
    values: Vec<f64>,
    /// Row indices for each value
    row_indices: Vec<usize>,
    /// Column pointers (cumulative sum of non-zeros per column)
    col_ptrs: Vec<usize>,
    /// Number of rows
    n_rows: usize,
    /// Number of columns
    n_cols: usize,
}

impl CscMatrix {
    /// Create from COO format
    pub fn from_coo(coo: &CooMatrix) -> Self {
        // Sort by column then row
        let mut indices: Vec<usize> = (0..coo.values.len()).collect();
        indices.sort_by_key(|&i| (coo.col_indices[i], coo.row_indices[i]));
        
        let mut values = Vec::with_capacity(coo.values.len());
        let mut row_indices = Vec::with_capacity(coo.values.len());
        let mut col_ptrs = vec![0];
        
        let mut current_col = 0;
        for &idx in &indices {
            let col = coo.col_indices[idx];
            
            // Fill column pointers for empty columns
            while current_col < col {
                col_ptrs.push(values.len());
                current_col += 1;
            }
            
            values.push(coo.values[idx]);
            row_indices.push(coo.row_indices[idx]);
        }
        
        // Fill remaining column pointers
        while col_ptrs.len() <= coo.n_cols {
            col_ptrs.push(values.len());
        }
        
        Self {
            values,
            row_indices,
            col_ptrs,
            n_rows: coo.n_rows,
            n_cols: coo.n_cols,
        }
    }
    
    /// Get a column as a sparse vector
    pub fn get_col(&self, col_idx: usize) -> Result<SparseVector> {
        if col_idx >= self.n_cols {
            return Err(TextError::InvalidInput(
                format!("Column index {} out of bounds for matrix with {} columns", col_idx, self.n_cols)
            ));
        }
        
        let start = self.col_ptrs[col_idx];
        let end = self.col_ptrs[col_idx + 1];
        
        let indices: Vec<usize> = self.row_indices[start..end].to_vec();
        let values: Vec<f64> = self.values[start..end].to_vec();
        
        Ok(SparseVector::from_indices_values(indices, values, self.n_rows))
    }
}

/// Block sparse matrix for better cache efficiency
#[derive(Debug, Clone)]
pub struct BlockSparseMatrix {
    /// Block size (square blocks)
    block_size: usize,
    /// Non-empty blocks stored as dense matrices
    blocks: HashMap<(usize, usize), Array2<f64>>,
    /// Number of rows
    n_rows: usize,
    /// Number of columns
    n_cols: usize,
}

impl BlockSparseMatrix {
    /// Create a new block sparse matrix
    pub fn new(n_rows: usize, n_cols: usize, block_size: usize) -> Self {
        Self {
            block_size,
            blocks: HashMap::new(),
            n_rows,
            n_cols,
        }
    }
    
    /// Set a value in the matrix
    pub fn set(&mut self, row: usize, col: usize, value: f64) -> Result<()> {
        if row >= self.n_rows || col >= self.n_cols {
            return Err(TextError::InvalidInput(
                format!("Index ({}, {}) out of bounds", row, col)
            ));
        }
        
        let block_row = row / self.block_size;
        let block_col = col / self.block_size;
        let local_row = row % self.block_size;
        let local_col = col % self.block_size;
        
        let block = self.blocks
            .entry((block_row, block_col))
            .or_insert_with(|| Array2::zeros((self.block_size, self.block_size)));
        
        block[[local_row, local_col]] = value;
        
        Ok(())
    }
    
    /// Get a value from the matrix
    pub fn get(&self, row: usize, col: usize) -> f64 {
        if row >= self.n_rows || col >= self.n_cols {
            return 0.0;
        }
        
        let block_row = row / self.block_size;
        let block_col = col / self.block_size;
        let local_row = row % self.block_size;
        let local_col = col % self.block_size;
        
        self.blocks
            .get(&(block_row, block_col))
            .map(|block| block[[local_row, local_col]])
            .unwrap_or(0.0)
    }
    
    /// Get memory usage
    pub fn memory_usage(&self) -> usize {
        self.blocks.len() * self.block_size * self.block_size * std::mem::size_of::<f64>()
    }
}

/// Quantized sparse vector for reduced memory usage
#[derive(Debug, Clone)]
pub struct QuantizedSparseVector {
    /// Indices of non-zero elements
    indices: Vec<u32>,
    /// Quantized values (8-bit)
    values: Vec<i8>,
    /// Scale factor for dequantization
    scale: f32,
    /// Zero point for dequantization
    zero_point: i8,
    /// Total size
    size: usize,
}

impl QuantizedSparseVector {
    /// Quantize a sparse vector to 8-bit representation
    pub fn from_sparse(sparse: &SparseVector) -> Self {
        if sparse.values.is_empty() {
            return Self {
                indices: Vec::new(),
                values: Vec::new(),
                scale: 1.0,
                zero_point: 0,
                size: sparse.size,
            };
        }
        
        // Find min and max values
        let min_val = sparse.values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = sparse.values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        // Calculate quantization parameters
        let range = max_val - min_val;
        let scale = (range / 255.0) as f32;
        let zero_point = (-min_val / range * 255.0).round() as i8;
        
        // Quantize values
        let indices: Vec<u32> = sparse.indices.iter().map(|&i| i as u32).collect();
        let values: Vec<i8> = sparse.values.iter()
            .map(|&v| {
                let scaled = ((v - min_val) / range * 255.0).round();
                (scaled.max(0.0).min(255.0) as i16 - 128) as i8  // Center around 0
            })
            .collect();
        
        Self {
            indices,
            values,
            scale,
            zero_point,
            size: sparse.size,
        }
    }
    
    /// Dequantize back to a regular sparse vector
    pub fn to_sparse(&self) -> SparseVector {
        let indices: Vec<usize> = self.indices.iter().map(|&i| i as usize).collect();
        let values: Vec<f64> = self.values.iter()
            .map(|&v| {
                let dequantized = (v as f32 + 128.0 - self.zero_point as f32) * self.scale;
                dequantized as f64
            })
            .collect();
        
        SparseVector::from_indices_values(indices, values, self.size)
    }
    
    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.indices.len() * std::mem::size_of::<u32>() +
        self.values.len() * std::mem::size_of::<i8>() +
        std::mem::size_of::<f32>() * 2 +
        std::mem::size_of::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_csr_from_dense() {
        let dense = arr2(&[
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 0.0],
            [4.0, 0.0, 5.0],
        ]);
        
        let csr = CsrMatrix::from_dense(&dense);
        
        assert_eq!(csr.shape(), (3, 3));
        assert_eq!(csr.nnz(), 5);
        
        let reconstructed = csr.to_dense();
        assert_eq!(reconstructed, dense);
    }

    #[test]
    fn test_sparse_vector() {
        let dense = arr1(&[0.0, 1.0, 0.0, 2.0, 0.0]);
        let sparse = SparseVector::from_dense(&dense);
        
        assert_eq!(sparse.nnz(), 2);
        assert_eq!(sparse.size(), 5);
        
        let reconstructed = sparse.to_dense();
        assert_eq!(reconstructed, dense);
    }

    #[test]
    fn test_sparse_dot_product() {
        let v1 = SparseVector::from_dense(&arr1(&[1.0, 0.0, 2.0, 0.0]));
        let v2 = SparseVector::from_dense(&arr1(&[0.0, 3.0, 2.0, 0.0]));
        
        let dot = v1.dot_sparse(&v2).unwrap();
        assert_eq!(dot, 4.0); // 1*0 + 0*3 + 2*2 + 0*0 = 4
    }

    #[test]
    fn test_dok_matrix() {
        let mut dok = DokMatrix::new(3, 3);
        
        dok.set(0, 0, 1.0).unwrap();
        dok.set(1, 1, 2.0).unwrap();
        dok.set(2, 0, 3.0).unwrap();
        
        assert_eq!(dok.get(0, 0), 1.0);
        assert_eq!(dok.get(0, 1), 0.0);
        assert_eq!(dok.nnz(), 3);
        
        let csr = dok.to_csr();
        assert_eq!(csr.nnz(), 3);
    }

    #[test]
    fn test_matrix_builder() {
        let mut builder = SparseMatrixBuilder::new(4);
        
        builder.add_row(SparseVector::from_dense(&arr1(&[1.0, 0.0, 2.0, 0.0]))).unwrap();
        builder.add_row(SparseVector::from_dense(&arr1(&[0.0, 3.0, 0.0, 4.0]))).unwrap();
        
        let matrix = builder.build();
        assert_eq!(matrix.shape(), (2, 4));
        assert_eq!(matrix.nnz(), 4);
    }

    #[test]
    fn test_memory_efficiency() {
        // Create a large sparse matrix (1000x1000 with 1% density)
        let n = 1000;
        let mut dense = Array2::zeros((n, n));
        
        // Add some random non-zero values
        for i in 0..n/10 {
            for j in 0..n/10 {
                dense[[i, j]] = i as f64 + j as f64;
            }
        }
        
        let sparse = CsrMatrix::from_dense(&dense);
        
        // Calculate memory savings
        let dense_memory = n * n * std::mem::size_of::<f64>();
        let sparse_memory = sparse.memory_usage();
        
        println!("Dense memory: {} bytes", dense_memory);
        println!("Sparse memory: {} bytes", sparse_memory);
        println!("Memory savings: {:.1}%", 
                (1.0 - sparse_memory as f64 / dense_memory as f64) * 100.0);
        
        assert!(sparse_memory < dense_memory / 10); // Should use less than 10% of dense memory
    }
    
    #[test]
    fn test_coo_matrix() {
        let mut coo = CooMatrix::new(3, 3);
        
        coo.push(0, 0, 1.0).unwrap();
        coo.push(1, 1, 2.0).unwrap();
        coo.push(2, 0, 3.0).unwrap();
        coo.push(1, 2, 4.0).unwrap();
        
        assert_eq!(coo.nnz(), 4);
        
        // Convert to CSR
        let csr = coo.to_csr();
        assert_eq!(csr.nnz(), 4);
        assert_eq!(csr.shape(), (3, 3));
        
        // Verify values
        let dense = csr.to_dense();
        assert_eq!(dense[[0, 0]], 1.0);
        assert_eq!(dense[[1, 1]], 2.0);
        assert_eq!(dense[[2, 0]], 3.0);
        assert_eq!(dense[[1, 2]], 4.0);
    }
    
    #[test]
    fn test_csc_matrix() {
        let mut coo = CooMatrix::new(3, 3);
        coo.push(0, 0, 1.0).unwrap();
        coo.push(1, 1, 2.0).unwrap();
        coo.push(2, 0, 3.0).unwrap();
        
        let csc = CscMatrix::from_coo(&coo);
        
        // Get column 0
        let col0 = csc.get_col(0).unwrap();
        assert_eq!(col0.nnz(), 2);
        let dense_col0 = col0.to_dense();
        assert_eq!(dense_col0[0], 1.0);
        assert_eq!(dense_col0[2], 3.0);
        
        // Get column 1
        let col1 = csc.get_col(1).unwrap();
        assert_eq!(col1.nnz(), 1);
        let dense_col1 = col1.to_dense();
        assert_eq!(dense_col1[1], 2.0);
    }
    
    #[test]
    fn test_block_sparse_matrix() {
        let mut block_sparse = BlockSparseMatrix::new(10, 10, 3);
        
        // Set some values
        block_sparse.set(0, 0, 1.0).unwrap();
        block_sparse.set(5, 5, 2.0).unwrap();
        block_sparse.set(9, 9, 3.0).unwrap();
        
        // Get values
        assert_eq!(block_sparse.get(0, 0), 1.0);
        assert_eq!(block_sparse.get(5, 5), 2.0);
        assert_eq!(block_sparse.get(9, 9), 3.0);
        assert_eq!(block_sparse.get(3, 3), 0.0);
        
        // Check memory usage
        let memory = block_sparse.memory_usage();
        assert!(memory > 0);
    }
    
    #[test]
    fn test_quantized_sparse_vector() {
        let mut sparse = SparseVector::new(10);
        sparse.indices = vec![1, 3, 7];
        sparse.values = vec![10.5, -5.2, 100.0];
        
        // Quantize
        let quantized = QuantizedSparseVector::from_sparse(&sparse);
        assert_eq!(quantized.indices.len(), 3);
        assert_eq!(quantized.values.len(), 3);
        
        // Dequantize
        let dequantized = quantized.to_sparse();
        assert_eq!(dequantized.nnz(), 3);
        
        // Check values are approximately preserved
        let orig_values = sparse.values();
        let deq_values = dequantized.values();
        for i in 0..3 {
            let error = (orig_values[i] - deq_values[i]).abs();
            let relative_error = error / orig_values[i].abs();
            assert!(relative_error < 0.02); // Less than 2% error
        }
        
        // Check memory savings
        let orig_memory = sparse.indices.len() * std::mem::size_of::<usize>() +
                         sparse.values.len() * std::mem::size_of::<f64>();
        let quantized_memory = quantized.memory_usage();
        assert!(quantized_memory < orig_memory);
    }
}