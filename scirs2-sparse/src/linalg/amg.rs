//! Algebraic Multigrid (AMG) preconditioner for sparse linear systems
//!
//! AMG is a powerful preconditioner for solving large sparse linear systems,
//! particularly effective for systems arising from discretizations of
//! elliptic PDEs and other problems with nice geometric structure.

use crate::csr_array::CsrArray;
use crate::error::{SparseError, SparseResult};
use crate::sparray::SparseArray;
use ndarray::{Array1, ArrayView1};
use num_traits::Float;
use std::collections::HashMap;
use std::fmt::Debug;

/// Options for the AMG preconditioner
#[derive(Debug, Clone)]
pub struct AMGOptions {
    /// Maximum number of levels in the multigrid hierarchy
    pub max_levels: usize,
    /// Strong connection threshold for coarsening (typically 0.25-0.5)
    pub theta: f64,
    /// Maximum size of coarse grid before switching to direct solver
    pub max_coarse_size: usize,
    /// Interpolation method
    pub interpolation: InterpolationType,
    /// Smoother type
    pub smoother: SmootherType,
    /// Number of pre-smoothing steps
    pub pre_smooth_steps: usize,
    /// Number of post-smoothing steps
    pub post_smooth_steps: usize,
    /// Cycle type (V-cycle, W-cycle, etc.)
    pub cycle_type: CycleType,
}

impl Default for AMGOptions {
    fn default() -> Self {
        Self {
            max_levels: 10,
            theta: 0.25,
            max_coarse_size: 50,
            interpolation: InterpolationType::Classical,
            smoother: SmootherType::GaussSeidel,
            pre_smooth_steps: 1,
            post_smooth_steps: 1,
            cycle_type: CycleType::V,
        }
    }
}

/// Interpolation methods for AMG
#[derive(Debug, Clone, Copy)]
pub enum InterpolationType {
    /// Classical Ruge-Stuben interpolation
    Classical,
    /// Direct interpolation
    Direct,
    /// Standard interpolation
    Standard,
}

/// Smoother types for AMG
#[derive(Debug, Clone, Copy)]
pub enum SmootherType {
    /// Gauss-Seidel smoother
    GaussSeidel,
    /// Jacobi smoother
    Jacobi,
    /// SOR smoother
    SOR,
}

/// Cycle types for AMG
#[derive(Debug, Clone, Copy)]
pub enum CycleType {
    /// V-cycle
    V,
    /// W-cycle
    W,
    /// F-cycle
    F,
}

/// AMG preconditioner implementation
#[derive(Debug)]
pub struct AMGPreconditioner<T>
where
    T: Float + Debug + Copy + 'static,
{
    /// Matrices at each level
    operators: Vec<CsrArray<T>>,
    /// Prolongation operators (coarse to fine)
    prolongations: Vec<CsrArray<T>>,
    /// Restriction operators (fine to coarse)
    restrictions: Vec<CsrArray<T>>,
    /// AMG options
    options: AMGOptions,
    /// Number of levels in the hierarchy
    num_levels: usize,
}

impl<T> AMGPreconditioner<T>
where
    T: Float + Debug + Copy + 'static,
{
    /// Create a new AMG preconditioner from a sparse matrix
    ///
    /// # Arguments
    ///
    /// * `matrix` - The coefficient matrix
    /// * `options` - AMG options
    ///
    /// # Returns
    ///
    /// A new AMG preconditioner
    ///
    /// # Example
    ///
    /// ```rust
    /// use scirs2_sparse::csr_array::CsrArray;
    /// use scirs2_sparse::linalg::amg::{AMGPreconditioner, AMGOptions};
    ///
    /// // Create a simple matrix
    /// let rows = vec![0, 0, 1, 1, 2, 2];
    /// let cols = vec![0, 1, 0, 1, 1, 2];
    /// let data = vec![2.0, -1.0, -1.0, 2.0, -1.0, 2.0];
    /// let matrix = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();
    ///
    /// // Create AMG preconditioner
    /// let amg = AMGPreconditioner::new(&matrix, AMGOptions::default()).unwrap();
    /// ```
    pub fn new(matrix: &CsrArray<T>, options: AMGOptions) -> SparseResult<Self> {
        let mut amg = AMGPreconditioner {
            operators: vec![matrix.clone()],
            prolongations: Vec::new(),
            restrictions: Vec::new(),
            options,
            num_levels: 1,
        };

        // Build the multigrid hierarchy
        amg.build_hierarchy()?;

        Ok(amg)
    }

    /// Build the multigrid hierarchy
    fn build_hierarchy(&mut self) -> SparseResult<()> {
        let mut level = 0;

        while level < self.options.max_levels - 1 {
            let current_matrix = &self.operators[level];
            let (rows, _) = current_matrix.shape();

            // Stop if matrix is small enough
            if rows <= self.options.max_coarse_size {
                break;
            }

            // Coarsen the matrix
            let (coarse_matrix, prolongation, restriction) = self.coarsen_level(current_matrix)?;

            // Check if coarsening was successful
            let (coarse_rows, _) = coarse_matrix.shape();
            if coarse_rows >= rows {
                // Coarsening didn't reduce the problem size significantly
                break;
            }

            self.operators.push(coarse_matrix);
            self.prolongations.push(prolongation);
            self.restrictions.push(restriction);
            self.num_levels += 1;
            level += 1;
        }

        Ok(())
    }

    /// Coarsen a single level
    fn coarsen_level(
        &self,
        matrix: &CsrArray<T>,
    ) -> SparseResult<(CsrArray<T>, CsrArray<T>, CsrArray<T>)> {
        // Classical AMG coarsening
        let (n, _) = matrix.shape();

        // For simplicity, use geometric coarsening (every other point)
        // In practice, you'd use sophisticated algebraic coarsening
        let coarse_size = (n + 1) / 2;
        let mut coarse_indices = Vec::new();
        let mut fine_to_coarse = HashMap::new();

        // Select coarse points (simplified - use every other point)
        for i in (0..n).step_by(2) {
            fine_to_coarse.insert(i, coarse_indices.len());
            coarse_indices.push(i);
        }

        // Build prolongation operator (interpolation)
        let prolongation = self.build_prolongation(matrix, &fine_to_coarse, coarse_size)?;

        // Build restriction operator (typically transpose of prolongation)
        let restriction = prolongation.transpose()?;

        // Build coarse matrix: A_coarse = R * A * P
        let temp = restriction.dot(matrix)?;
        let coarse_matrix = temp.dot(&prolongation)?;

        Ok((coarse_matrix, prolongation, restriction))
    }

    /// Build prolongation (interpolation) operator
    fn build_prolongation(
        &self,
        matrix: &CsrArray<T>,
        fine_to_coarse: &HashMap<usize, usize>,
        coarse_size: usize,
    ) -> SparseResult<CsrArray<T>> {
        let (n, _) = matrix.shape();
        let mut prolongation_data = Vec::new();
        let mut prolongation_indices = Vec::new();
        let mut prolongation_indptr = vec![0];

        for i in 0..n {
            if let Some(&coarse_idx) = fine_to_coarse.get(&i) {
                // Direct injection for coarse points
                prolongation_data.push(T::one());
                prolongation_indices.push(coarse_idx);
            } else {
                // Interpolation for fine points
                // Simplified: use nearest coarse neighbors
                let nearest_coarse = self.find_nearest_coarse_point(i, fine_to_coarse);
                if let Some(coarse_idx) = nearest_coarse {
                    prolongation_data.push(T::one());
                    prolongation_indices.push(coarse_idx);
                } else {
                    // No coarse neighbors found, use direct injection to first coarse point
                    prolongation_data.push(T::one());
                    prolongation_indices.push(0);
                }
            }
            prolongation_indptr.push(prolongation_data.len());
        }

        CsrArray::new(
            prolongation_data,
            prolongation_indptr,
            prolongation_indices,
            (n, coarse_size),
        )
    }

    /// Find nearest coarse point (simplified implementation)
    fn find_nearest_coarse_point(
        &self,
        fine_point: usize,
        fine_to_coarse: &HashMap<usize, usize>,
    ) -> Option<usize> {
        // Simple approach: find the closest coarse point by index
        let mut best_dist = usize::MAX;
        let mut best_coarse = None;

        for (&coarse_fine_idx, &coarse_idx) in fine_to_coarse.iter() {
            let dist = if coarse_fine_idx > fine_point {
                coarse_fine_idx - fine_point
            } else {
                fine_point - coarse_fine_idx
            };

            if dist < best_dist {
                best_dist = dist;
                best_coarse = Some(coarse_idx);
            }
        }

        best_coarse
    }

    /// Apply the AMG preconditioner
    ///
    /// Solves the system M * x = b approximately, where M is the preconditioner
    ///
    /// # Arguments
    ///
    /// * `b` - Right-hand side vector
    ///
    /// # Returns
    ///
    /// Approximate solution x
    pub fn apply(&self, b: &ArrayView1<T>) -> SparseResult<Array1<T>> {
        let (n, _) = self.operators[0].shape();
        if b.len() != n {
            return Err(SparseError::DimensionMismatch {
                expected: n,
                found: b.len(),
            });
        }

        let mut x = Array1::zeros(n);
        self.mg_cycle(&mut x, b, 0)?;
        Ok(x)
    }

    /// Perform one multigrid cycle
    fn mg_cycle(&self, x: &mut Array1<T>, b: &ArrayView1<T>, level: usize) -> SparseResult<()> {
        if level == self.num_levels - 1 {
            // Coarsest level - solve directly (simplified)
            self.coarse_solve(x, b, level)?;
            return Ok(());
        }

        let matrix = &self.operators[level];

        // Pre-smoothing
        for _ in 0..self.options.pre_smooth_steps {
            self.smooth(x, b, matrix)?;
        }

        // Compute residual
        let ax = matrix_vector_multiply(matrix, &x.view())?;
        let residual = b - &ax;

        // Restrict residual to coarse grid
        let restriction = &self.restrictions[level];
        let coarse_residual = matrix_vector_multiply(restriction, &residual.view())?;

        // Solve on coarse grid
        let coarse_size = coarse_residual.len();
        let mut coarse_correction = Array1::zeros(coarse_size);

        match self.options.cycle_type {
            CycleType::V => {
                self.mg_cycle(&mut coarse_correction, &coarse_residual.view(), level + 1)?;
            }
            CycleType::W => {
                // Two recursive calls for W-cycle
                self.mg_cycle(&mut coarse_correction, &coarse_residual.view(), level + 1)?;
                self.mg_cycle(&mut coarse_correction, &coarse_residual.view(), level + 1)?;
            }
            CycleType::F => {
                // Full multigrid - not implemented here
                self.mg_cycle(&mut coarse_correction, &coarse_residual.view(), level + 1)?;
            }
        }

        // Prolongate correction to fine grid
        let prolongation = &self.prolongations[level];
        let fine_correction = matrix_vector_multiply(prolongation, &coarse_correction.view())?;

        // Add correction
        for i in 0..x.len() {
            x[i] = x[i] + fine_correction[i];
        }

        // Post-smoothing
        for _ in 0..self.options.post_smooth_steps {
            self.smooth(x, b, matrix)?;
        }

        Ok(())
    }

    /// Apply smoother
    fn smooth(
        &self,
        x: &mut Array1<T>,
        b: &ArrayView1<T>,
        matrix: &CsrArray<T>,
    ) -> SparseResult<()> {
        match self.options.smoother {
            SmootherType::GaussSeidel => self.gauss_seidel_smooth(x, b, matrix),
            SmootherType::Jacobi => self.jacobi_smooth(x, b, matrix),
            SmootherType::SOR => self.sor_smooth(x, b, matrix, T::from(1.2).unwrap()),
        }
    }

    /// Gauss-Seidel smoother
    fn gauss_seidel_smooth(
        &self,
        x: &mut Array1<T>,
        b: &ArrayView1<T>,
        matrix: &CsrArray<T>,
    ) -> SparseResult<()> {
        let n = x.len();

        for i in 0..n {
            let row_start = matrix.indptr[i];
            let row_end = matrix.indptr[i + 1];

            let mut sum = T::zero();
            let mut diag_val = T::zero();

            for j in row_start..row_end {
                let col = matrix.indices[j];
                let val = matrix.data[j];

                if col == i {
                    diag_val = val;
                } else {
                    sum = sum + val * x[col];
                }
            }

            if !diag_val.is_zero() {
                x[i] = (b[i] - sum) / diag_val;
            }
        }

        Ok(())
    }

    /// Jacobi smoother
    fn jacobi_smooth(
        &self,
        x: &mut Array1<T>,
        b: &ArrayView1<T>,
        matrix: &CsrArray<T>,
    ) -> SparseResult<()> {
        let n = x.len();
        let mut x_new = x.clone();

        for i in 0..n {
            let row_start = matrix.indptr[i];
            let row_end = matrix.indptr[i + 1];

            let mut sum = T::zero();
            let mut diag_val = T::zero();

            for j in row_start..row_end {
                let col = matrix.indices[j];
                let val = matrix.data[j];

                if col == i {
                    diag_val = val;
                } else {
                    sum = sum + val * x[col];
                }
            }

            if !diag_val.is_zero() {
                x_new[i] = (b[i] - sum) / diag_val;
            }
        }

        *x = x_new;
        Ok(())
    }

    /// SOR smoother
    fn sor_smooth(
        &self,
        x: &mut Array1<T>,
        b: &ArrayView1<T>,
        matrix: &CsrArray<T>,
        omega: T,
    ) -> SparseResult<()> {
        let n = x.len();

        for i in 0..n {
            let row_start = matrix.indptr[i];
            let row_end = matrix.indptr[i + 1];

            let mut sum = T::zero();
            let mut diag_val = T::zero();

            for j in row_start..row_end {
                let col = matrix.indices[j];
                let val = matrix.data[j];

                if col == i {
                    diag_val = val;
                } else {
                    sum = sum + val * x[col];
                }
            }

            if !diag_val.is_zero() {
                let x_gs = (b[i] - sum) / diag_val;
                x[i] = (T::one() - omega) * x[i] + omega * x_gs;
            }
        }

        Ok(())
    }

    /// Coarse grid solver (simplified direct method)
    fn coarse_solve(&self, x: &mut Array1<T>, b: &ArrayView1<T>, level: usize) -> SparseResult<()> {
        // For now, just use a few iterations of Gauss-Seidel
        let matrix = &self.operators[level];

        for _ in 0..10 {
            self.gauss_seidel_smooth(x, b, matrix)?;
        }

        Ok(())
    }

    /// Get the number of levels in the hierarchy
    pub fn num_levels(&self) -> usize {
        self.num_levels
    }

    /// Get the size of the matrix at a given level
    pub fn level_size(&self, level: usize) -> Option<(usize, usize)> {
        if level < self.num_levels {
            Some(self.operators[level].shape())
        } else {
            None
        }
    }
}

/// Helper function for matrix-vector multiplication
fn matrix_vector_multiply<T>(matrix: &CsrArray<T>, x: &ArrayView1<T>) -> SparseResult<Array1<T>>
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

    let mut result = Array1::zeros(rows);

    for i in 0..rows {
        for j in matrix.indptr[i]..matrix.indptr[i + 1] {
            let col = matrix.indices[j];
            let val = matrix.data[j];
            result[i] = result[i] + val * x[col];
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csr_array::CsrArray;
    use approx::assert_relative_eq;

    #[test]
    fn test_amg_preconditioner_creation() {
        // Create a simple 3x3 matrix
        let rows = vec![0, 0, 1, 1, 2, 2];
        let cols = vec![0, 1, 0, 1, 1, 2];
        let data = vec![2.0, -1.0, -1.0, 2.0, -1.0, 2.0];
        let matrix = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();

        let amg = AMGPreconditioner::new(&matrix, AMGOptions::default()).unwrap();

        assert!(amg.num_levels() >= 1);
        assert_eq!(amg.level_size(0), Some((3, 3)));
    }

    #[test]
    fn test_amg_apply() {
        // Create a diagonal system (easy test case)
        let rows = vec![0, 1, 2];
        let cols = vec![0, 1, 2];
        let data = vec![2.0, 3.0, 4.0];
        let matrix = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();

        let amg = AMGPreconditioner::new(&matrix, AMGOptions::default()).unwrap();

        let b = Array1::from_vec(vec![2.0, 3.0, 4.0]);
        let x = amg.apply(&b.view()).unwrap();

        // For a diagonal system, AMG should get close to the exact solution [1, 1, 1]
        assert!(x[0] > 0.5 && x[0] < 1.5);
        assert!(x[1] > 0.5 && x[1] < 1.5);
        assert!(x[2] > 0.5 && x[2] < 1.5);
    }

    #[test]
    fn test_amg_options() {
        let mut options = AMGOptions::default();
        options.max_levels = 5;
        options.theta = 0.5;
        options.smoother = SmootherType::Jacobi;
        options.cycle_type = CycleType::W;

        assert_eq!(options.max_levels, 5);
        assert_eq!(options.theta, 0.5);
        assert!(matches!(options.smoother, SmootherType::Jacobi));
        assert!(matches!(options.cycle_type, CycleType::W));
    }

    #[test]
    fn test_gauss_seidel_smoother() {
        let rows = vec![0, 0, 1, 1, 2, 2];
        let cols = vec![0, 1, 0, 1, 1, 2];
        let data = vec![2.0, -1.0, -1.0, 2.0, -1.0, 2.0];
        let matrix = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();

        let amg = AMGPreconditioner::new(&matrix, AMGOptions::default()).unwrap();

        let mut x = Array1::from_vec(vec![0.0, 0.0, 0.0]);
        let b = Array1::from_vec(vec![1.0, 1.0, 1.0]);

        // Apply one Gauss-Seidel iteration
        amg.gauss_seidel_smooth(&mut x, &b.view(), &matrix).unwrap();

        // Solution should improve (move away from zero)
        assert!(x.iter().any(|&val| val.abs() > 1e-10));
    }
}
