//! Sparse Jacobian optimization
//!
//! This module provides functionality for detecting and exploiting sparsity
//! patterns in Jacobian matrices to improve computational efficiency.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use crate::common::IntegrateFloat;
use crate::error::{IntegrateError, IntegrateResult};
use std::collections::{HashSet, HashMap};

/// Represents a sparsity pattern
#[derive(Debug, Clone)]
pub struct SparsePattern {
    /// Non-zero entries as (row, col) pairs
    pub entries: Vec<(usize, usize)>,
    /// Number of rows
    pub n_rows: usize,
    /// Number of columns
    pub n_cols: usize,
    /// Row-wise non-zero indices
    pub row_indices: Vec<Vec<usize>>,
    /// Column-wise non-zero indices
    pub col_indices: Vec<Vec<usize>>,
}

impl SparsePattern {
    /// Create a new sparse pattern
    pub fn new(n_rows: usize, n_cols: usize) -> Self {
        SparsePattern {
            entries: Vec::new(),
            n_rows,
            n_cols,
            row_indices: vec![Vec::new(); n_rows],
            col_indices: vec![Vec::new(); n_cols],
        }
    }

    /// Add a non-zero entry
    pub fn add_entry(&mut self, row: usize, col: usize) {
        if row < self.n_rows && col < self.n_cols {
            self.entries.push((row, col));
            self.row_indices[row].push(col);
            self.col_indices[col].push(row);
        }
    }

    /// Get the number of non-zeros
    pub fn nnz(&self) -> usize {
        self.entries.len()
    }

    /// Get the sparsity ratio (fraction of zeros)
    pub fn sparsity(&self) -> f64 {
        let total = (self.n_rows * self.n_cols) as f64;
        if total > 0.0 {
            1.0 - (self.nnz() as f64 / total)
        } else {
            0.0
        }
    }

    /// Check if pattern is sparse enough to benefit from sparse methods
    pub fn is_sparse(&self, threshold: f64) -> bool {
        self.sparsity() > threshold
    }

    /// Compute coloring for efficient Jacobian computation
    pub fn compute_coloring(&self) -> ColGrouping {
        // Use greedy graph coloring algorithm
        let mut colors: HashMap<usize, usize> = HashMap::new();
        let mut max_color = 0;

        for col in 0..self.n_cols {
            let mut used_colors = HashSet::new();

            // Find colors used by adjacent columns
            for &row in &self.col_indices[col] {
                for &other_col in &self.row_indices[row] {
                    if other_col != col {
                        if let Some(&color) = colors.get(&other_col) {
                            used_colors.insert(color);
                        }
                    }
                }
            }

            // Find first available color
            let mut color = 0;
            while used_colors.contains(&color) {
                color += 1;
            }

            colors.insert(col, color);
            max_color = max_color.max(color);
        }

        // Group columns by color
        let mut groups = vec![Vec::new(); max_color + 1];
        for (col, &color) in &colors {
            groups[color].push(*col);
        }

        ColGrouping { groups }
    }
}

/// Column grouping for efficient Jacobian computation
pub struct ColGrouping {
    /// Groups of columns that can be computed together
    pub groups: Vec<Vec<usize>>,
}

impl ColGrouping {
    /// Get the number of groups
    pub fn n_groups(&self) -> usize {
        self.groups.len()
    }
}

/// Sparse Jacobian representation
pub struct SparseJacobian<F: IntegrateFloat> {
    /// The sparsity pattern
    pub pattern: SparsePattern,
    /// Non-zero values in row-major order
    pub values: Vec<F>,
    /// Mapping from (row, col) to value index
    pub index_map: HashMap<(usize, usize), usize>,
}

impl<F: IntegrateFloat> SparseJacobian<F> {
    /// Create a new sparse Jacobian from pattern
    pub fn from_pattern(pattern: SparsePattern) -> Self {
        let mut index_map = HashMap::new();
        for (idx, &(row, col)) in pattern.entries.iter().enumerate() {
            index_map.insert((row, col), idx);
        }

        SparseJacobian {
            values: vec![F::zero(); pattern.entries.len()],
            pattern,
            index_map,
        }
    }

    /// Set a value
    pub fn set(&mut self, row: usize, col: usize, value: F) {
        if let Some(&idx) = self.index_map.get(&(row, col)) {
            self.values[idx] = value;
        }
    }

    /// Get a value
    pub fn get(&self, row: usize, col: usize) -> F {
        if let Some(&idx) = self.index_map.get(&(row, col)) {
            self.values[idx]
        } else {
            F::zero()
        }
    }

    /// Convert to dense matrix
    pub fn to_dense(&self) -> Array2<F> {
        let mut dense = Array2::zeros((self.pattern.n_rows, self.pattern.n_cols));
        for (idx, &(row, col)) in self.pattern.entries.iter().enumerate() {
            dense[[row, col]] = self.values[idx];
        }
        dense
    }

    /// Apply to vector (matrix-vector multiplication)
    pub fn apply(&self, x: ArrayView1<F>) -> IntegrateResult<Array1<F>> {
        if x.len() != self.pattern.n_cols {
            return Err(IntegrateError::DimensionMismatch(
                format!("Expected {} columns, got {}", self.pattern.n_cols, x.len())
            ));
        }

        let mut result = Array1::zeros(self.pattern.n_rows);
        for (idx, &(row, col)) in self.pattern.entries.iter().enumerate() {
            result[row] += self.values[idx] * x[col];
        }

        Ok(result)
    }
}

/// Detect sparsity pattern by probing with finite differences
pub fn detect_sparsity<F, Func>(
    f: Func,
    x: ArrayView1<F>,
    eps: F,
) -> IntegrateResult<SparsePattern>
where
    F: IntegrateFloat,
    Func: Fn(ArrayView1<F>) -> IntegrateResult<Array1<F>>,
{
    let n = x.len();
    let f0 = f(x)?;
    let m = f0.len();

    let mut pattern = SparsePattern::new(m, n);

    // Probe each variable
    for j in 0..n {
        let mut x_pert = x.to_owned();
        x_pert[j] += eps;
        let f_pert = f(x_pert.view())?;

        // Check which outputs changed
        for i in 0..m {
            if (f_pert[i] - f0[i]).abs() > F::epsilon() {
                pattern.add_entry(i, j);
            }
        }
    }

    Ok(pattern)
}

/// Compress a dense Jacobian using a sparsity pattern
pub fn compress_jacobian<F: IntegrateFloat>(
    dense: ArrayView2<F>,
    pattern: &SparsePattern,
) -> SparseJacobian<F> {
    let mut sparse = SparseJacobian::from_pattern(pattern.clone());
    
    for (idx, &(row, col)) in pattern.entries.iter().enumerate() {
        sparse.values[idx] = dense[[row, col]];
    }
    
    sparse
}

/// Compute sparse Jacobian using coloring
pub fn colored_jacobian<F, Func>(
    f: Func,
    x: ArrayView1<F>,
    pattern: &SparsePattern,
    eps: F,
) -> IntegrateResult<SparseJacobian<F>>
where
    F: IntegrateFloat,
    Func: Fn(ArrayView1<F>) -> IntegrateResult<Array1<F>>,
{
    let coloring = pattern.compute_coloring();
    let f0 = f(x)?;
    let mut jacobian = SparseJacobian::from_pattern(pattern.clone());

    // Compute Jacobian using column groups
    for group in &coloring.groups {
        let mut x_pert = x.to_owned();
        
        // Perturb all columns in this group
        for &col in group {
            x_pert[col] += eps;
        }
        
        let f_pert = f(x_pert.view())?;
        
        // Extract derivatives for this group
        for &col in group {
            for &row in &pattern.col_indices[col] {
                let deriv = (f_pert[row] - f0[row]) / eps;
                jacobian.set(row, col, deriv);
            }
        }
    }

    Ok(jacobian)
}

/// Example: Create a tridiagonal sparsity pattern
pub fn example_tridiagonal_pattern(n: usize) -> SparsePattern {
    let mut pattern = SparsePattern::new(n, n);
    
    for i in 0..n {
        pattern.add_entry(i, i); // Diagonal
        if i > 0 {
            pattern.add_entry(i, i - 1); // Sub-diagonal
        }
        if i < n - 1 {
            pattern.add_entry(i, i + 1); // Super-diagonal
        }
    }
    
    pattern
}

/// Sparse Jacobian updater for quasi-Newton methods
pub struct SparseJacobianUpdater<F: IntegrateFloat> {
    pattern: SparsePattern,
    threshold: F,
}

impl<F: IntegrateFloat> SparseJacobianUpdater<F> {
    /// Create a new updater
    pub fn new(pattern: SparsePattern, threshold: F) -> Self {
        SparseJacobianUpdater { pattern, threshold }
    }

    /// Update sparse Jacobian using Broyden's method
    pub fn broyden_update(
        &self,
        jac: &mut SparseJacobian<F>,
        dx: ArrayView1<F>,
        df: ArrayView1<F>,
    ) -> IntegrateResult<()> {
        let jdx = jac.apply(dx)?;
        let dy = &df - &jdx;
        
        let dx_norm_sq = dx.dot(&dx);
        if dx_norm_sq < self.threshold {
            return Ok(());
        }

        // Update only non-zero entries
        for (idx, &(i, j)) in self.pattern.entries.iter().enumerate() {
            jac.values[idx] += dy[i] * dx[j] / dx_norm_sq;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_pattern() {
        let mut pattern = SparsePattern::new(3, 3);
        pattern.add_entry(0, 0);
        pattern.add_entry(1, 1);
        pattern.add_entry(2, 2);
        pattern.add_entry(0, 1);
        
        assert_eq!(pattern.nnz(), 4);
        assert!(pattern.sparsity() > 0.5);
    }

    #[test]
    fn test_coloring() {
        let pattern = example_tridiagonal_pattern(5);
        let coloring = pattern.compute_coloring();
        
        // Tridiagonal matrix should need at most 3 colors
        assert!(coloring.n_groups() <= 3);
    }

    #[test]
    fn test_sparse_jacobian() {
        let pattern = example_tridiagonal_pattern(3);
        let mut jac = SparseJacobian::from_pattern(pattern);
        
        // Set some values
        jac.set(0, 0, 2.0);
        jac.set(0, 1, -1.0);
        jac.set(1, 0, -1.0);
        jac.set(1, 1, 2.0);
        jac.set(1, 2, -1.0);
        jac.set(2, 1, -1.0);
        jac.set(2, 2, 2.0);
        
        // Test matrix-vector multiplication
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let y = jac.apply(x.view()).unwrap();
        
        // Should compute [2*1 - 1*2, -1*1 + 2*2 - 1*3, -1*2 + 2*3]
        assert!((y[0] - 0.0).abs() < 1e-10);
        assert!((y[1] - 0.0).abs() < 1e-10);
        assert!((y[2] - 4.0).abs() < 1e-10);
    }
}