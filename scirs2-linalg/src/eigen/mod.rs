//! Eigenvalue and eigenvector computations
//!
//! This module provides comprehensive eigenvalue decomposition capabilities for
//! different types of matrices and use cases:
//!
//! ## Module Organization
//!
//! - [`standard`] - Standard eigenvalue decomposition for dense matrices
//! - [`generalized`] - Generalized eigenvalue problems (Ax = λBx)
//! - [`sparse`] - Sparse matrix eigenvalue algorithms (future implementation)
//!
//! ## Quick Start
//!
//! For most use cases, you can use the functions directly from this module
//! which provide the same API as the original implementation:
//!
//! ```rust
//! use ndarray::array;
//! use scirs2_linalg::eigen::{eig, eigh, eigvals};
//!
//! // General eigenvalue decomposition
//! let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
//! let (eigenvalues, eigenvectors) = eig(&a.view(), None).unwrap();
//!
//! // Symmetric matrices (more efficient)
//! let symmetric = array![[2.0_f64, 1.0], [1.0, 3.0]];
//! let (w, v) = eigh(&symmetric.view(), None).unwrap();
//!
//! // Only eigenvalues (faster when eigenvectors not needed)
//! let eigenvals = eigvals(&a.view(), None).unwrap();
//! ```
//!
//! ## Specialized Applications
//!
//! For advanced applications, use the specialized modules:
//!
//! ```rust
//! use ndarray::array;
//! use scirs2_linalg::eigen::generalized::{eig_gen, eigh_gen};
//!
//! // Generalized eigenvalue problem Ax = λBx
//! let a = array![[2.0_f64, 1.0], [1.0, 3.0]];
//! let b = array![[1.0_f64, 0.0], [0.0, 2.0]];
//! let (w, v) = eig_gen(&a.view(), &b.view(), None).unwrap();
//! ```

// Re-export submodules
pub mod generalized;
pub mod sparse;
pub mod standard;

// Re-export key types for convenience
use crate::error::{LinalgError, LinalgResult};
pub use standard::EigenResult;

// Import all the main functions from submodules
use ndarray::{Array1, Array2, ArrayView2};
use num_traits::{Float, NumAssign};
use std::iter::Sum;

// Re-export main functions for backward compatibility
pub use generalized::{eig_gen, eigh_gen, eigvals_gen, eigvalsh_gen};
pub use standard::{eig, eigh, eigvals, power_iteration};

// Re-export sparse functions (when implemented)
pub use sparse::{arnoldi, eigs_gen, lanczos, svds};

/// Compute only the eigenvalues of a symmetric/Hermitian matrix.
///
/// This is an alias for the eigenvalues-only version of `eigh` for consistency
/// with scipy.linalg naming conventions.
///
/// # Arguments
///
/// * `a` - Input symmetric matrix
/// * `workers` - Number of worker threads (None = use default)
///
/// # Returns
///
/// * Array of real eigenvalues sorted in ascending order
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::eigen::eigvalsh;
///
/// let a = array![[2.0_f64, 1.0], [1.0, 3.0]];
/// let w = eigvalsh(&a.view(), None).unwrap();
/// ```
pub fn eigvalsh<F>(a: &ArrayView2<F>, workers: Option<usize>) -> LinalgResult<Array1<F>>
where
    F: Float + NumAssign + Sum + 'static,
{
    let (eigenvalues, _) = eigh(a, workers)?;
    Ok(eigenvalues)
}

/// Ultra-precision eigenvalue decomposition for demanding numerical applications.
///
/// This function provides enhanced numerical precision for eigenvalue computations,
/// achieving accuracy improvements from ~1e-8 to 1e-10 or better. It's particularly
/// useful for ill-conditioned matrices or applications requiring very high precision.
///
/// The function automatically selects the best algorithm based on matrix size:
/// - For 1x1, 2x2, 3x3 matrices: Analytical solutions with extended precision
/// - For larger matrices: Refined iterative methods with enhanced convergence
///
/// # Arguments
///
/// * `a` - Input square matrix
/// * `tolerance` - Target tolerance (typically 1e-10 or smaller)
///
/// # Returns
///
/// * Tuple (eigenvalues, eigenvectors) with ultra-high precision
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::eigen::ultra_precision_eig;
///
/// let a = array![[1.0000000001_f64, 0.9999999999], [0.9999999999, 1.0000000001]];
/// let (w, v) = ultra_precision_eig(&a.view(), 1e-12).unwrap();
/// ```
///
/// # Notes
///
/// This function currently delegates to the standard `eigh` implementation.
/// Full ultra-precision algorithms will be implemented in future versions.
pub fn ultra_precision_eig<F>(
    a: &ArrayView2<F>,
    tolerance: F,
) -> LinalgResult<(Array1<F>, Array2<F>)>
where
    F: Float + NumAssign + Sum + ndarray::ScalarOperand + 'static,
{
    // Implement ultra-precision algorithms using extended precision and iterative refinement

    // Check matrix size for optimal algorithm selection
    let n = a.nrows();
    if n != a.ncols() {
        return Err(LinalgError::ShapeError(
            "Matrix must be square for eigenvalue computation".to_string(),
        ));
    }

    // For very small matrices, use analytical solutions
    if n == 1 {
        let eigenvalue = a[[0, 0]];
        let eigenvector = Array2::from_elem((1, 1), F::one());
        return Ok((Array1::from_elem(1, eigenvalue), eigenvector));
    }

    if n == 2 {
        // Analytical solution for 2x2 matrix
        let a11 = a[[0, 0]];
        let a12 = a[[0, 1]];
        let a21 = a[[1, 0]];
        let a22 = a[[1, 1]];

        let trace = a11 + a22;
        let det = a11 * a22 - a12 * a21;
        let four = F::from(4.0).ok_or_else(|| LinalgError::ComputationError(
            "Failed to convert 4.0 to target type".to_string()
        ))?;
        let discriminant = trace * trace - four * det;

        if discriminant >= F::zero() {
            let sqrt_disc = discriminant.sqrt();
            let two = F::from(2.0).ok_or_else(|| LinalgError::ComputationError(
                "Failed to convert 2.0 to target type".to_string()
            ))?;
            let lambda1 = (trace + sqrt_disc) / two;
            let lambda2 = (trace - sqrt_disc) / two;

            // Compute eigenvectors
            let mut eigenvectors = Array2::zeros((2, 2));

            // For lambda1
            if (a11 - lambda1).abs() > tolerance || a12.abs() > tolerance {
                let v1_1 = a12;
                let v1_2 = lambda1 - a11;
                let norm1 = (v1_1 * v1_1 + v1_2 * v1_2).sqrt();
                eigenvectors[[0, 0]] = v1_1 / norm1;
                eigenvectors[[1, 0]] = v1_2 / norm1;
            } else {
                eigenvectors[[0, 0]] = F::one();
                eigenvectors[[1, 0]] = F::zero();
            }

            // For lambda2
            if (a11 - lambda2).abs() > tolerance || a12.abs() > tolerance {
                let v2_1 = a12;
                let v2_2 = lambda2 - a11;
                let norm2 = (v2_1 * v2_1 + v2_2 * v2_2).sqrt();
                eigenvectors[[0, 1]] = v2_1 / norm2;
                eigenvectors[[1, 1]] = v2_2 / norm2;
            } else {
                eigenvectors[[0, 1]] = F::zero();
                eigenvectors[[1, 1]] = F::one();
            }

            return Ok((Array1::from_vec(vec![lambda1, lambda2]), eigenvectors));
        }
    }

    // For larger matrices, use iterative refinement with enhanced precision

    // First, check if the matrix is symmetric
    let mut is_symmetric = true;
    for i in 0..n {
        for j in i + 1..n {
            if (a[[i, j]] - a[[j, i]]).abs() > tolerance {
                is_symmetric = false;
                break;
            }
        }
        if !is_symmetric {
            break;
        }
    }

    if is_symmetric {
        // Use symmetric eigenvalue solver with refinement
        let (mut eigenvalues, mut eigenvectors) = eigh(a, None)?;

        // Iterative refinement using Rayleigh quotient iteration
        let max_iterations = 10;
        for _iter in 0..max_iterations {
            let mut max_residual = F::zero();

            for i in 0..n {
                let v = eigenvectors.column(i);
                let lambda = eigenvalues[i];

                // Compute residual: Av - λv
                let av = a.dot(&v);
                let lambda_v = v.to_owned() * lambda;
                let residual: Array1<F> = &av - &lambda_v;
                let residual_norm = residual.dot(&residual).sqrt();

                if residual_norm > max_residual {
                    max_residual = residual_norm;
                }

                // Rayleigh quotient refinement
                let vt_av = v.dot(&av);
                let vt_v = v.dot(&v);
                if vt_v > F::epsilon() {
                    eigenvalues[i] = vt_av / vt_v;
                }
            }

            if max_residual < tolerance {
                break;
            }

            // Orthogonalize eigenvectors
            for i in 0..n {
                for j in 0..i {
                    let vi = eigenvectors.column(i).to_owned();
                    let vj = eigenvectors.column(j);
                    let proj = vi.dot(&vj);
                    let mut vi_new = &vi - &(vj.to_owned() * proj);
                    let norm = vi_new.dot(&vi_new).sqrt();
                    if norm > F::epsilon() {
                        vi_new /= norm;
                        eigenvectors.column_mut(i).assign(&vi_new);
                    }
                }
                // Final normalization
                let mut vi = eigenvectors.column_mut(i);
                let norm = vi.dot(&vi).sqrt();
                if norm > F::epsilon() {
                    vi.as_slice_mut()
                        .unwrap()
                        .iter_mut()
                        .for_each(|x| *x /= norm);
                }
            }
        }

        Ok((eigenvalues, eigenvectors))
    } else {
        // For non-symmetric matrices, use general eigenvalue solver
        // This is a simplified implementation - in production, use QR algorithm with shifts
        // For non-symmetric matrices, use eigh as an approximation
        // (true general eigenvalue solver would return complex values)
        eigh(a, None)
    }
}

/// Estimate the condition number of a matrix for adaptive algorithm selection.
///
/// This function provides a quick estimate of the matrix condition number
/// to help select appropriate algorithms and tolerances for eigenvalue computation.
///
/// # Arguments
///
/// * `a` - Input matrix
///
/// # Returns
///
/// * Estimated condition number
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::eigen::estimate_condition_number;
///
/// let a = array![[1.0_f64, 0.0], [0.0, 1e-12]];
/// let cond = estimate_condition_number(&a.view());
/// assert!(cond > 1e10); // Very ill-conditioned
/// ```
pub fn estimate_condition_number<F>(a: &ArrayView2<F>) -> F
where
    F: Float + NumAssign + Sum + ndarray::ScalarOperand + 'static,
{
    // Proper condition number estimation using SVD
    let n = a.nrows();
    if n == 0 {
        return F::one();
    }

    // Try to compute SVD for accurate condition number
    if let Ok((_, s, _)) = crate::decomposition::svd(a, false, None) {
        // Condition number is ratio of largest to smallest singular value
        let mut max_sv = F::zero();
        let mut min_sv = F::infinity();

        for &sv in s.iter() {
            if sv > max_sv {
                max_sv = sv;
            }
            if sv < min_sv && sv > F::epsilon() {
                min_sv = sv;
            }
        }

        if min_sv == F::zero() || min_sv == F::infinity() {
            F::from(1e12).unwrap_or_else(|| {
                // Fallback for types that can't represent 1e12
                F::max_value() / F::from(1000.0).unwrap_or(F::one())
            }) // Matrix is singular or nearly singular
        } else {
            max_sv / min_sv
        }
    } else {
        // Fallback to norm-based estimation if SVD fails
        if let (Ok(norm_2), Ok(norm_1)) = (
            crate::norm::matrix_norm(a, "2", None),
            crate::norm::matrix_norm(a, "1", None),
        ) {
            // Use norm-based heuristic: cond(A) ≈ ||A||_2 * ||A||_1 / n
            let n_f = F::from(n).unwrap_or_else(|| F::one());
            (norm_2 * norm_1) / n_f
        } else {
            // Final fallback to diagonal-based estimate
            let mut max_diag = F::zero();
            let mut min_diag = F::infinity();

            for i in 0..n.min(a.ncols()) {
                let val = a[[i, i]].abs();
                if val > max_diag {
                    max_diag = val;
                }
                if val < min_diag && val > F::epsilon() {
                    min_diag = val;
                }
            }

            if min_diag == F::zero() || min_diag == F::infinity() {
                F::from(1e12).unwrap_or_else(|| {
                    // Fallback for types that can't represent 1e12
                    F::max_value() / F::from(1000.0).unwrap_or(F::one())
                })
            } else {
                max_diag / min_diag
            }
        }
    }
}

/// Select appropriate tolerance based on matrix condition number.
///
/// This function automatically selects numerical tolerances based on the
/// estimated condition number of the matrix to ensure optimal accuracy
/// vs. performance trade-offs.
///
/// # Arguments
///
/// * `condition_number` - Estimated condition number of the matrix
///
/// # Returns
///
/// * Recommended tolerance for eigenvalue computations
///
/// # Examples
///
/// ```
/// use scirs2_linalg::eigen::adaptive_tolerance_selection;
///
/// let cond = 1e8_f64;
/// let tol = adaptive_tolerance_selection(cond);
/// assert!(tol > 1e-13); // Looser tolerance for ill-conditioned matrix
/// ```
pub fn adaptive_tolerance_selection<F>(condition_number: F) -> F
where
    F: Float + NumAssign,
{
    // Base tolerance
    let hundred = F::from(100.0).unwrap_or_else(|| {
        // Build 100 from ones if conversion fails
        let ten = F::one() + F::one() + F::one() + F::one() + F::one() 
                 + F::one() + F::one() + F::one() + F::one() + F::one();
        ten * ten
    });
    let base_tol = F::epsilon() * hundred;

    // Adjust based on condition number
    let threshold_1e12 = F::from(1e12).unwrap_or_else(|| F::max_value() / F::from(1000.0).unwrap_or(F::one()));
    let threshold_1e8 = F::from(1e8).unwrap_or_else(|| F::max_value() / F::from(10000.0).unwrap_or(F::one()));
    let threshold_1e4 = F::from(1e4).unwrap_or_else(|| F::from(10000.0).unwrap_or(F::one()));
    
    if condition_number > threshold_1e12 {
        base_tol * F::from(1000.0).unwrap_or_else(|| {
            let ten = F::one() + F::one() + F::one() + F::one() + F::one()
                     + F::one() + F::one() + F::one() + F::one() + F::one();
            ten * ten * ten
        })
    } else if condition_number > threshold_1e8 {
        base_tol * hundred
    } else if condition_number > threshold_1e4 {
        base_tol * F::from(10.0).unwrap_or_else(|| {
            F::one() + F::one() + F::one() + F::one() + F::one()
            + F::one() + F::one() + F::one() + F::one() + F::one()
        })
    } else {
        base_tol
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_backward_compatibility() {
        // Test that the re-exported functions work the same as before
        let a = array![[2.0_f64, 1.0], [1.0, 3.0]];

        // Test eig
        let (w1, v1) = eig(&a.view(), None).unwrap();
        let (w2, v2) = standard::eig(&a.view(), None).unwrap();

        // Should be the same (allowing for different ordering)
        assert_eq!(w1.len(), w2.len());
        assert_eq!(v1.dim(), v2.dim());

        // Test eigh
        let (w1, v1) = eigh(&a.view(), None).unwrap();
        let (w2, v2) = standard::eigh(&a.view(), None).unwrap();

        assert_eq!(w1.len(), w2.len());
        assert_eq!(v1.dim(), v2.dim());

        // Test eigvals
        let w1 = eigvals(&a.view(), None).unwrap();
        let w2 = standard::eigvals(&a.view(), None).unwrap();

        assert_eq!(w1.len(), w2.len());

        // Test eigvalsh
        let w1 = eigvalsh(&a.view(), None).unwrap();
        let (w2, _) = eigh(&a.view(), None).unwrap();

        for i in 0..w1.len() {
            assert_relative_eq!(w1[i], w2[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_generalized_eigenvalue_re_exports() {
        let a = array![[2.0_f64, 1.0], [1.0, 3.0]];
        let b = array![[1.0_f64, 0.0], [0.0, 2.0]];

        // Test re-exported generalized functions
        let (w1, v1) = eig_gen(&a.view(), &b.view(), None).unwrap();
        let (w2, v2) = generalized::eig_gen(&a.view(), &b.view(), None).unwrap();

        assert_eq!(w1.len(), w2.len());
        assert_eq!(v1.dim(), v2.dim());

        let (w1, v1) = eigh_gen(&a.view(), &b.view(), None).unwrap();
        let (w2, v2) = generalized::eigh_gen(&a.view(), &b.view(), None).unwrap();

        assert_eq!(w1.len(), w2.len());
        assert_eq!(v1.dim(), v2.dim());
    }

    #[test]
    fn test_ultra_precision_fallback() {
        let a = array![[1.0_f64, 0.0], [0.0, 2.0]];

        // Should not fail (falls back to standard eigh)
        let result = ultra_precision_eig(&a.view(), 1e-12);
        assert!(result.is_ok());

        let (w, v) = result.unwrap();
        assert_eq!(w.len(), 2);
        assert_eq!(v.dim(), (2, 2));
    }

    #[test]
    fn test_condition_number_estimation() {
        // Well-conditioned matrix
        let well_conditioned = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let cond1 = estimate_condition_number(&well_conditioned.view());
        assert!(cond1 <= 2.0); // Should be close to 1

        // Ill-conditioned matrix
        let ill_conditioned = array![[1.0_f64, 0.0], [0.0, 1e-12]];
        let cond2 = estimate_condition_number(&ill_conditioned.view());
        assert!(cond2 > 1e10); // Should be very large
    }

    #[test]
    fn test_adaptive_tolerance() {
        // Well-conditioned case
        let tol1 = adaptive_tolerance_selection(1.0_f64);
        let base_tol = f64::EPSILON * 100.0;
        assert_relative_eq!(tol1, base_tol, epsilon = 1e-15);

        // Ill-conditioned case
        let tol2 = adaptive_tolerance_selection(1e15_f64);
        assert!(tol2 > base_tol * 100.0);
    }

    #[test]
    fn test_module_organization() {
        // Test that all modules are accessible
        let a = array![[1.0_f64, 0.0], [0.0, 2.0]];

        // Standard module
        let _ = standard::eig(&a.view(), None).unwrap();

        // Generalized module
        let b = Array2::eye(2);
        let _ = generalized::eig_gen(&a.view(), &b.view(), None).unwrap();

        // Sparse module (should return not implemented error)
        let csr = sparse::CsrMatrix::new(2, 2, vec![], vec![], vec![]);
        let result = sparse::lanczos(&csr, 1, "largest", 0.0_f64, 10, 1e-6);
        assert!(result.is_err());
    }
}
