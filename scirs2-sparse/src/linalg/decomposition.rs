//! Matrix decomposition algorithms for sparse matrices
//!
//! This module provides various matrix decomposition algorithms optimized
//! for sparse matrices, including LU, QR, Cholesky, and incomplete variants.

use crate::error::{SparseError, SparseResult};
use crate::sparray::SparseArray;
use crate::csr_array::CsrArray;
use ndarray::{Array1, Array2};
use num_traits::Float;
use std::collections::HashMap;
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Sub};

/// LU decomposition result
#[derive(Debug, Clone)]
pub struct LUResult<T>
where
    T: Float + Debug + Copy + 'static,
{
    /// Lower triangular factor
    pub l: CsrArray<T>,
    /// Upper triangular factor
    pub u: CsrArray<T>,
    /// Permutation matrix (as permutation vector)
    pub p: Array1<usize>,
    /// Whether decomposition was successful
    pub success: bool,
}

/// QR decomposition result
#[derive(Debug, Clone)]
pub struct QRResult<T>
where
    T: Float + Debug + Copy + 'static,
{
    /// Orthogonal factor Q
    pub q: CsrArray<T>,
    /// Upper triangular factor R
    pub r: CsrArray<T>,
    /// Whether decomposition was successful
    pub success: bool,
}

/// Cholesky decomposition result
#[derive(Debug, Clone)]
pub struct CholeskyResult<T>
where
    T: Float + Debug + Copy + 'static,
{
    /// Lower triangular Cholesky factor
    pub l: CsrArray<T>,
    /// Whether decomposition was successful
    pub success: bool,
}

/// Options for incomplete LU decomposition
#[derive(Debug, Clone)]
pub struct ILUOptions {
    /// Drop tolerance for numerical stability
    pub drop_tol: f64,
    /// Fill factor (maximum fill-in ratio)
    pub fill_factor: f64,
    /// Maximum number of fill-in entries per row
    pub max_fill_per_row: usize,
    /// Pivoting strategy
    pub use_pivoting: bool,
}

impl Default for ILUOptions {
    fn default() -> Self {
        Self {
            drop_tol: 1e-4,
            fill_factor: 2.0,
            max_fill_per_row: 20,
            use_pivoting: true,
        }
    }
}

/// Options for incomplete Cholesky decomposition
#[derive(Debug, Clone)]
pub struct ICOptions {
    /// Drop tolerance for numerical stability
    pub drop_tol: f64,
    /// Fill factor (maximum fill-in ratio)
    pub fill_factor: f64,
    /// Maximum number of fill-in entries per row
    pub max_fill_per_row: usize,
}

impl Default for ICOptions {
    fn default() -> Self {
        Self {
            drop_tol: 1e-4,
            fill_factor: 2.0,
            max_fill_per_row: 20,
        }
    }
}

/// Compute sparse LU decomposition with partial pivoting
///
/// Computes the LU decomposition of a sparse matrix A such that P*A = L*U,
/// where P is a permutation matrix, L is lower triangular, and U is upper triangular.
///
/// # Arguments
///
/// * `matrix` - The sparse matrix to decompose
/// * `pivot_threshold` - Pivoting threshold for numerical stability (0.0 to 1.0)
///
/// # Returns
///
/// LU decomposition result
///
/// # Examples
///
/// ```
/// use scirs2_sparse::linalg::lu_decomposition;
/// use scirs2_sparse::csr_array::CsrArray;
///
/// // Create a sparse matrix
/// let rows = vec![0, 0, 1, 2];
/// let cols = vec![0, 1, 1, 2];
/// let data = vec![2.0, 1.0, 3.0, 4.0];
/// let matrix = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();
///
/// let lu_result = lu_decomposition(&matrix, 0.1).unwrap();
/// ```
pub fn lu_decomposition<T, S>(
    matrix: &S,
    pivot_threshold: f64,
) -> SparseResult<LUResult<T>>
where
    T: Float + Debug + Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
    S: SparseArray<T>,
{
    let (n, m) = matrix.shape();
    if n != m {
        return Err(SparseError::ValueError(
            "Matrix must be square for LU decomposition".to_string(),
        ));
    }
    
    // Convert to working format (simplified sparse representation)
    let (row_indices, col_indices, values) = matrix.find();
    let mut working_matrix = SparseWorkingMatrix::from_triplets(row_indices.as_slice().unwrap(), col_indices.as_slice().unwrap(), values.as_slice().unwrap(), n);
    
    // Initialize permutation
    let mut p: Vec<usize> = (0..n).collect();
    
    // Gaussian elimination with partial pivoting
    for k in 0..n - 1 {
        // Find pivot
        let pivot_row = find_pivot(&working_matrix, k, &p, pivot_threshold)?;
        
        // Swap rows in permutation
        if pivot_row != k {
            p.swap(k, pivot_row);
        }
        
        let actual_pivot_row = p[k];
        let pivot_value = working_matrix.get(actual_pivot_row, k);
        
        if pivot_value.abs() < T::from(1e-14).unwrap() {
            return Ok(LUResult {
                l: CsrArray::from_triplets(&[], &[], &[], (n, n), false)?,
                u: CsrArray::from_triplets(&[], &[], &[], (n, n), false)?,
                p: Array1::from_vec(p),
                success: false,
            });
        }
        
        // Eliminate below pivot
        for i in (k + 1)..n {
            let actual_row_i = p[i];
            let factor = working_matrix.get(actual_row_i, k) / pivot_value;
            
            if !factor.is_zero() {
                // Store multiplier in L
                working_matrix.set(actual_row_i, k, factor);
                
                // Update row i
                let pivot_row_data = working_matrix.get_row(actual_pivot_row);
                for (col, &value) in &pivot_row_data {
                    if *col > k {
                        let old_val = working_matrix.get(actual_row_i, *col);
                        working_matrix.set(actual_row_i, *col, old_val - factor * value);
                    }
                }
            }
        }
    }
    
    // Extract L and U matrices
    let (l_rows, l_cols, l_vals, u_rows, u_cols, u_vals) = extract_lu_factors(&working_matrix, &p, n);
    
    let l = CsrArray::from_triplets(&l_rows, &l_cols, &l_vals, (n, n), false)?;
    let u = CsrArray::from_triplets(&u_rows, &u_cols, &u_vals, (n, n), false)?;
    
    Ok(LUResult {
        l,
        u,
        p: Array1::from_vec(p),
        success: true,
    })
}

/// Compute sparse QR decomposition using Givens rotations
///
/// Computes the QR decomposition of a sparse matrix A = Q*R,
/// where Q is orthogonal and R is upper triangular.
///
/// # Arguments
///
/// * `matrix` - The sparse matrix to decompose
///
/// # Returns
///
/// QR decomposition result
///
/// # Examples
///
/// ```
/// use scirs2_sparse::linalg::qr_decomposition;
/// use scirs2_sparse::csr_array::CsrArray;
///
/// let rows = vec![0, 1, 2];
/// let cols = vec![0, 0, 1];
/// let data = vec![1.0, 2.0, 3.0];
/// let matrix = CsrArray::from_triplets(&rows, &cols, &data, (3, 2), false).unwrap();
///
/// let qr_result = qr_decomposition(&matrix).unwrap();
/// ```
pub fn qr_decomposition<T, S>(matrix: &S) -> SparseResult<QRResult<T>>
where
    T: Float + Debug + Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
    S: SparseArray<T>,
{
    let (m, n) = matrix.shape();
    
    // Convert to dense for QR (sparse QR is complex)
    let dense_matrix = matrix.to_array();
    
    // Simple Gram-Schmidt QR decomposition
    let mut q = Array2::zeros((m, n));
    let mut r = Array2::zeros((n, n));
    
    for j in 0..n {
        // Copy column j
        for i in 0..m {
            q[[i, j]] = dense_matrix[[i, j]];
        }
        
        // Orthogonalize against previous columns
        for k in 0..j {
            let mut dot = T::zero();
            for i in 0..m {
                dot = dot + q[[i, k]] * dense_matrix[[i, j]];
            }
            r[[k, j]] = dot;
            
            for i in 0..m {
                q[[i, j]] = q[[i, j]] - dot * q[[i, k]];
            }
        }
        
        // Normalize
        let mut norm = T::zero();
        for i in 0..m {
            norm = norm + q[[i, j]] * q[[i, j]];
        }
        norm = norm.sqrt();
        r[[j, j]] = norm;
        
        if !norm.is_zero() {
            for i in 0..m {
                q[[i, j]] = q[[i, j]] / norm;
            }
        }
    }
    
    // Convert back to sparse
    let q_sparse = dense_to_sparse(&q)?;
    let r_sparse = dense_to_sparse(&r)?;
    
    Ok(QRResult {
        q: q_sparse,
        r: r_sparse,
        success: true,
    })
}

/// Compute sparse Cholesky decomposition
///
/// Computes the Cholesky decomposition of a symmetric positive definite matrix A = L*L^T,
/// where L is lower triangular.
///
/// # Arguments
///
/// * `matrix` - The symmetric positive definite sparse matrix
///
/// # Returns
///
/// Cholesky decomposition result
///
/// # Examples
///
/// ```
/// use scirs2_sparse::linalg::cholesky_decomposition;
/// use scirs2_sparse::csr_array::CsrArray;
///
/// // Create a simple SPD matrix
/// let rows = vec![0, 1, 1, 2, 2, 2];
/// let cols = vec![0, 0, 1, 0, 1, 2];
/// let data = vec![4.0, 2.0, 5.0, 1.0, 3.0, 6.0];
/// let matrix = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();
///
/// let chol_result = cholesky_decomposition(&matrix).unwrap();
/// ```
pub fn cholesky_decomposition<T, S>(matrix: &S) -> SparseResult<CholeskyResult<T>>
where
    T: Float + Debug + Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
    S: SparseArray<T>,
{
    let (n, m) = matrix.shape();
    if n != m {
        return Err(SparseError::ValueError(
            "Matrix must be square for Cholesky decomposition".to_string(),
        ));
    }
    
    // Convert to working format
    let (row_indices, col_indices, values) = matrix.find();
    let mut working_matrix = SparseWorkingMatrix::from_triplets(row_indices.as_slice().unwrap(), col_indices.as_slice().unwrap(), values.as_slice().unwrap(), n);
    
    // Cholesky decomposition algorithm
    for k in 0..n {
        // Compute diagonal element
        let mut sum = T::zero();
        for j in 0..k {
            let l_kj = working_matrix.get(k, j);
            sum = sum + l_kj * l_kj;
        }
        
        let a_kk = working_matrix.get(k, k);
        let diag_val = a_kk - sum;
        
        if diag_val <= T::zero() {
            return Ok(CholeskyResult {
                l: CsrArray::from_triplets(&[], &[], &[], (n, n), false)?,
                success: false,
            });
        }
        
        let l_kk = diag_val.sqrt();
        working_matrix.set(k, k, l_kk);
        
        // Compute below-diagonal elements
        for i in (k + 1)..n {
            let mut sum = T::zero();
            for j in 0..k {
                sum = sum + working_matrix.get(i, j) * working_matrix.get(k, j);
            }
            
            let a_ik = working_matrix.get(i, k);
            let l_ik = (a_ik - sum) / l_kk;
            working_matrix.set(i, k, l_ik);
        }
    }
    
    // Extract lower triangular matrix
    let (l_rows, l_cols, l_vals) = extract_lower_triangular(&working_matrix, n);
    let l = CsrArray::from_triplets(&l_rows, &l_cols, &l_vals, (n, n), false)?;
    
    Ok(CholeskyResult {
        l,
        success: true,
    })
}

/// Compute incomplete LU decomposition (ILU)
///
/// Computes an approximate LU decomposition with controlled fill-in
/// for use as a preconditioner in iterative methods.
///
/// # Arguments
///
/// * `matrix` - The sparse matrix to decompose
/// * `options` - ILU options controlling fill-in and dropping
///
/// # Returns
///
/// Incomplete LU decomposition result
pub fn incomplete_lu<T, S>(
    matrix: &S,
    options: Option<ILUOptions>,
) -> SparseResult<LUResult<T>>
where
    T: Float + Debug + Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
    S: SparseArray<T>,
{
    let opts = options.unwrap_or_default();
    let (n, m) = matrix.shape();
    
    if n != m {
        return Err(SparseError::ValueError(
            "Matrix must be square for ILU decomposition".to_string(),
        ));
    }
    
    // Convert to working format
    let (row_indices, col_indices, values) = matrix.find();
    let mut working_matrix = SparseWorkingMatrix::from_triplets(row_indices.as_slice().unwrap(), col_indices.as_slice().unwrap(), values.as_slice().unwrap(), n);
    
    // ILU(0) algorithm - no fill-in beyond original sparsity pattern
    for k in 0..n - 1 {
        let pivot_val = working_matrix.get(k, k);
        
        if pivot_val.abs() < T::from(1e-14).unwrap() {
            continue; // Skip singular pivot
        }
        
        // Get all non-zero entries in column k below diagonal
        let col_k_entries = working_matrix.get_column_below_diagonal(k);
        
        for &row_i in &col_k_entries {
            let factor = working_matrix.get(row_i, k) / pivot_val;
            
            // Drop small factors
            if factor.abs() < T::from(opts.drop_tol).unwrap() {
                working_matrix.set(row_i, k, T::zero());
                continue;
            }
            
            working_matrix.set(row_i, k, factor);
            
            // Update row i (only existing non-zeros)
            let row_k_entries = working_matrix.get_row_after_column(k, k);
            for (col_j, &val_kj) in &row_k_entries {
                if working_matrix.has_entry(row_i, *col_j) {
                    let old_val = working_matrix.get(row_i, *col_j);
                    let new_val = old_val - factor * val_kj;
                    
                    // Drop small values
                    if new_val.abs() < T::from(opts.drop_tol).unwrap() {
                        working_matrix.set(row_i, *col_j, T::zero());
                    } else {
                        working_matrix.set(row_i, *col_j, new_val);
                    }
                }
            }
        }
    }
    
    // Extract L and U factors
    let identity_p: Vec<usize> = (0..n).collect();
    let (l_rows, l_cols, l_vals, u_rows, u_cols, u_vals) = extract_lu_factors(&working_matrix, &identity_p, n);
    
    let l = CsrArray::from_triplets(&l_rows, &l_cols, &l_vals, (n, n), false)?;
    let u = CsrArray::from_triplets(&u_rows, &u_cols, &u_vals, (n, n), false)?;
    
    Ok(LUResult {
        l,
        u,
        p: Array1::from_vec(identity_p),
        success: true,
    })
}

/// Compute incomplete Cholesky decomposition (IC)
///
/// Computes an approximate Cholesky decomposition with controlled fill-in
/// for use as a preconditioner in iterative methods.
///
/// # Arguments
///
/// * `matrix` - The symmetric positive definite sparse matrix
/// * `options` - IC options controlling fill-in and dropping
///
/// # Returns
///
/// Incomplete Cholesky decomposition result
pub fn incomplete_cholesky<T, S>(
    matrix: &S,
    options: Option<ICOptions>,
) -> SparseResult<CholeskyResult<T>>
where
    T: Float + Debug + Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
    S: SparseArray<T>,
{
    let opts = options.unwrap_or_default();
    let (n, m) = matrix.shape();
    
    if n != m {
        return Err(SparseError::ValueError(
            "Matrix must be square for IC decomposition".to_string(),
        ));
    }
    
    // Convert to working format
    let (row_indices, col_indices, values) = matrix.find();
    let mut working_matrix = SparseWorkingMatrix::from_triplets(row_indices.as_slice().unwrap(), col_indices.as_slice().unwrap(), values.as_slice().unwrap(), n);
    
    // IC(0) algorithm - no fill-in beyond original sparsity pattern
    for k in 0..n {
        // Compute diagonal element
        let mut sum = T::zero();
        let row_k_before_k = working_matrix.get_row_before_column(k, k);
        for (col_j, &val_kj) in &row_k_before_k {
            sum = sum + val_kj * val_kj;
        }
        
        let a_kk = working_matrix.get(k, k);
        let diag_val = a_kk - sum;
        
        if diag_val <= T::zero() {
            return Ok(CholeskyResult {
                l: CsrArray::from_triplets(&[], &[], &[], (n, n), false)?,
                success: false,
            });
        }
        
        let l_kk = diag_val.sqrt();
        working_matrix.set(k, k, l_kk);
        
        // Compute below-diagonal elements (only existing entries)
        let col_k_below = working_matrix.get_column_below_diagonal(k);
        for &row_i in &col_k_below {
            let mut sum = T::zero();
            let row_i_before_k = working_matrix.get_row_before_column(row_i, k);
            let row_k_before_k = working_matrix.get_row_before_column(k, k);
            
            // Compute dot product of L[i, :k] and L[k, :k]
            for (col_j, &val_ij) in &row_i_before_k {
                if let Some(&val_kj) = row_k_before_k.get(col_j) {
                    sum = sum + val_ij * val_kj;
                }
            }
            
            let a_ik = working_matrix.get(row_i, k);
            let l_ik = (a_ik - sum) / l_kk;
            
            // Drop small values
            if l_ik.abs() < T::from(opts.drop_tol).unwrap() {
                working_matrix.set(row_i, k, T::zero());
            } else {
                working_matrix.set(row_i, k, l_ik);
            }
        }
    }
    
    // Extract lower triangular matrix
    let (l_rows, l_cols, l_vals) = extract_lower_triangular(&working_matrix, n);
    let l = CsrArray::from_triplets(&l_rows, &l_cols, &l_vals, (n, n), false)?;
    
    Ok(CholeskyResult {
        l,
        success: true,
    })
}

/// Simple sparse working matrix for decomposition algorithms
struct SparseWorkingMatrix<T>
where
    T: Float + Debug + Copy,
{
    data: HashMap<(usize, usize), T>,
    n: usize,
}

impl<T> SparseWorkingMatrix<T>
where
    T: Float + Debug + Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
{
    fn from_triplets(rows: &[usize], cols: &[usize], values: &[T], n: usize) -> Self {
        let mut data = HashMap::new();
        
        for (i, (&row, &col)) in rows.iter().zip(cols.iter()).enumerate() {
            data.insert((row, col), values[i]);
        }
        
        Self { data, n }
    }
    
    fn get(&self, row: usize, col: usize) -> T {
        self.data.get(&(row, col)).copied().unwrap_or(T::zero())
    }
    
    fn set(&mut self, row: usize, col: usize, value: T) {
        if value.is_zero() {
            self.data.remove(&(row, col));
        } else {
            self.data.insert((row, col), value);
        }
    }
    
    fn has_entry(&self, row: usize, col: usize) -> bool {
        self.data.contains_key(&(row, col))
    }
    
    fn get_row(&self, row: usize) -> HashMap<usize, T> {
        let mut result = HashMap::new();
        for (&(r, c), &value) in &self.data {
            if r == row {
                result.insert(c, value);
            }
        }
        result
    }
    
    fn get_row_after_column(&self, row: usize, col: usize) -> HashMap<usize, T> {
        let mut result = HashMap::new();
        for (&(r, c), &value) in &self.data {
            if r == row && c > col {
                result.insert(c, value);
            }
        }
        result
    }
    
    fn get_row_before_column(&self, row: usize, col: usize) -> HashMap<usize, T> {
        let mut result = HashMap::new();
        for (&(r, c), &value) in &self.data {
            if r == row && c < col {
                result.insert(c, value);
            }
        }
        result
    }
    
    fn get_column_below_diagonal(&self, col: usize) -> Vec<usize> {
        let mut result = Vec::new();
        for (&(r, c), _) in &self.data {
            if c == col && r > col {
                result.push(r);
            }
        }
        result.sort();
        result
    }
}

/// Find pivot for LU decomposition
fn find_pivot<T>(
    matrix: &SparseWorkingMatrix<T>,
    k: usize,
    p: &[usize],
    threshold: f64,
) -> SparseResult<usize>
where
    T: Float + Debug + Copy,
{
    let mut max_val = T::zero();
    let mut pivot_row = k;
    
    for i in k..matrix.n {
        let actual_row = p[i];
        let val = matrix.get(actual_row, k).abs();
        if val > max_val {
            max_val = val;
            pivot_row = i;
        }
    }
    
    Ok(pivot_row)
}

/// Extract L and U factors from working matrix
fn extract_lu_factors<T>(
    matrix: &SparseWorkingMatrix<T>,
    p: &[usize],
    n: usize,
) -> (Vec<usize>, Vec<usize>, Vec<T>, Vec<usize>, Vec<usize>, Vec<T>)
where
    T: Float + Debug + Copy,
{
    let mut l_rows = Vec::new();
    let mut l_cols = Vec::new();
    let mut l_vals = Vec::new();
    let mut u_rows = Vec::new();
    let mut u_cols = Vec::new();
    let mut u_vals = Vec::new();
    
    for i in 0..n {
        let actual_row = p[i];
        
        // Add diagonal 1 to L
        l_rows.push(i);
        l_cols.push(i);
        l_vals.push(T::one());
        
        for j in 0..n {
            let val = matrix.get(actual_row, j);
            if !val.is_zero() {
                if j < i {
                    // Below diagonal - goes to L
                    l_rows.push(i);
                    l_cols.push(j);
                    l_vals.push(val);
                } else {
                    // On or above diagonal - goes to U
                    u_rows.push(i);
                    u_cols.push(j);
                    u_vals.push(val);
                }
            }
        }
    }
    
    (l_rows, l_cols, l_vals, u_rows, u_cols, u_vals)
}

/// Extract lower triangular matrix
fn extract_lower_triangular<T>(
    matrix: &SparseWorkingMatrix<T>,
    n: usize,
) -> (Vec<usize>, Vec<usize>, Vec<T>)
where
    T: Float + Debug + Copy,
{
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    
    for i in 0..n {
        for j in 0..=i {
            let val = matrix.get(i, j);
            if !val.is_zero() {
                rows.push(i);
                cols.push(j);
                vals.push(val);
            }
        }
    }
    
    (rows, cols, vals)
}

/// Convert dense matrix to sparse
fn dense_to_sparse<T>(matrix: &Array2<T>) -> SparseResult<CsrArray<T>>
where
    T: Float + Debug + Copy,
{
    let (m, n) = matrix.dim();
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    
    for i in 0..m {
        for j in 0..n {
            let val = matrix[[i, j]];
            if !val.is_zero() {
                rows.push(i);
                cols.push(j);
                vals.push(val);
            }
        }
    }
    
    CsrArray::from_triplets(&rows, &cols, &vals, (m, n), false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csr_array::CsrArray;
    use approx::assert_relative_eq;

    fn create_test_matrix() -> CsrArray<f64> {
        // Create a simple test matrix
        let rows = vec![0, 0, 1, 1, 2, 2];
        let cols = vec![0, 1, 0, 1, 1, 2];
        let data = vec![2.0, 1.0, 1.0, 3.0, 2.0, 4.0];
        
        CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap()
    }

    fn create_spd_matrix() -> CsrArray<f64> {
        // Create a symmetric positive definite matrix
        let rows = vec![0, 1, 1, 2, 2, 2];
        let cols = vec![0, 0, 1, 0, 1, 2];
        let data = vec![4.0, 2.0, 5.0, 1.0, 3.0, 6.0];
        
        CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap()
    }

    #[test]
    fn test_lu_decomposition() {
        let matrix = create_test_matrix();
        let lu_result = lu_decomposition(&matrix, 0.1).unwrap();
        
        assert!(lu_result.success);
        assert_eq!(lu_result.l.shape(), (3, 3));
        assert_eq!(lu_result.u.shape(), (3, 3));
        assert_eq!(lu_result.p.len(), 3);
    }

    #[test]
    fn test_qr_decomposition() {
        let rows = vec![0, 1, 2];
        let cols = vec![0, 0, 1];
        let data = vec![1.0, 2.0, 3.0];
        let matrix = CsrArray::from_triplets(&rows, &cols, &data, (3, 2), false).unwrap();
        
        let qr_result = qr_decomposition(&matrix).unwrap();
        
        assert!(qr_result.success);
        assert_eq!(qr_result.q.shape(), (3, 2));
        assert_eq!(qr_result.r.shape(), (2, 2));
    }

    #[test]
    fn test_cholesky_decomposition() {
        let matrix = create_spd_matrix();
        let chol_result = cholesky_decomposition(&matrix).unwrap();
        
        assert!(chol_result.success);
        assert_eq!(chol_result.l.shape(), (3, 3));
    }

    #[test]
    fn test_incomplete_lu() {
        let matrix = create_test_matrix();
        let options = ILUOptions {
            drop_tol: 1e-6,
            ..Default::default()
        };
        
        let ilu_result = incomplete_lu(&matrix, Some(options)).unwrap();
        
        assert!(ilu_result.success);
        assert_eq!(ilu_result.l.shape(), (3, 3));
        assert_eq!(ilu_result.u.shape(), (3, 3));
    }

    #[test]
    fn test_incomplete_cholesky() {
        let matrix = create_spd_matrix();
        let options = ICOptions {
            drop_tol: 1e-6,
            ..Default::default()
        };
        
        let ic_result = incomplete_cholesky(&matrix, Some(options)).unwrap();
        
        assert!(ic_result.success);
        assert_eq!(ic_result.l.shape(), (3, 3));
    }

    #[test]
    fn test_sparse_working_matrix() {
        let rows = vec![0, 1, 2];
        let cols = vec![0, 1, 2];
        let vals = vec![1.0, 2.0, 3.0];
        
        let mut matrix = SparseWorkingMatrix::from_triplets(&rows, &cols, &vals, 3);
        
        assert_eq!(matrix.get(0, 0), 1.0);
        assert_eq!(matrix.get(1, 1), 2.0);
        assert_eq!(matrix.get(2, 2), 3.0);
        assert_eq!(matrix.get(0, 1), 0.0);
        
        matrix.set(0, 1, 5.0);
        assert_eq!(matrix.get(0, 1), 5.0);
        
        matrix.set(0, 1, 0.0);
        assert_eq!(matrix.get(0, 1), 0.0);
        assert!(!matrix.has_entry(0, 1));
    }

    #[test]
    fn test_dense_to_sparse_conversion() {
        let dense = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 2.0, 3.0]).unwrap();
        let sparse = dense_to_sparse(&dense).unwrap();
        
        assert_eq!(sparse.nnz(), 3);
        assert_eq!(sparse.get(0, 0), 1.0);
        assert_eq!(sparse.get(0, 1), 0.0);
        assert_eq!(sparse.get(1, 0), 2.0);
        assert_eq!(sparse.get(1, 1), 3.0);
    }
}