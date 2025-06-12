//! Matrix-variate distributions
//!
//! This module provides implementations of probability distributions that operate
//! on matrices, including the Wishart distribution, matrix normal distribution,
//! and inverse Wishart distribution.

use ndarray::{Array2, ArrayView2};
use num_traits::{Float, One, Zero};
use std::f64::consts::PI;

use crate::basic::{det, inv};
use crate::decomposition::cholesky;
use crate::error::{LinalgError, LinalgResult};
use crate::random::random_normal_matrix;

/// Parameters for a matrix normal distribution
///
/// A matrix normal distribution N(M, U, V) has mean matrix M,
/// row covariance U, and column covariance V.
#[derive(Debug, Clone)]
pub struct MatrixNormalParams<F: Float> {
    /// Mean matrix
    pub mean: Array2<F>,
    /// Row covariance matrix
    pub row_cov: Array2<F>,
    /// Column covariance matrix  
    pub col_cov: Array2<F>,
}

impl<F: Float + Zero + One + Copy + std::fmt::Debug + std::fmt::Display> MatrixNormalParams<F> {
    /// Create new matrix normal parameters
    ///
    /// # Arguments
    ///
    /// * `mean` - Mean matrix of shape (m, n)
    /// * `row_cov` - Row covariance matrix of shape (m, m)
    /// * `col_cov` - Column covariance matrix of shape (n, n)
    ///
    /// # Returns
    ///
    /// * Matrix normal parameters
    pub fn new(mean: Array2<F>, row_cov: Array2<F>, col_cov: Array2<F>) -> LinalgResult<Self> {
        let (m, n) = mean.dim();

        if row_cov.dim() != (m, m) {
            return Err(LinalgError::ShapeError(format!(
                "Row covariance must be {}x{}, got {:?}",
                m,
                m,
                row_cov.dim()
            )));
        }

        if col_cov.dim() != (n, n) {
            return Err(LinalgError::ShapeError(format!(
                "Column covariance must be {}x{}, got {:?}",
                n,
                n,
                col_cov.dim()
            )));
        }

        Ok(Self {
            mean,
            row_cov,
            col_cov,
        })
    }
}

/// Parameters for a Wishart distribution
///
/// A Wishart distribution W(V, n) has scale matrix V and degrees of freedom n.
#[derive(Debug, Clone)]
pub struct WishartParams<F: Float> {
    /// Scale matrix (positive definite)
    pub scale: Array2<F>,
    /// Degrees of freedom
    pub dof: F,
}

impl<F: Float + Zero + One + Copy + std::fmt::Debug + std::fmt::Display> WishartParams<F> {
    /// Create new Wishart parameters
    ///
    /// # Arguments
    ///
    /// * `scale` - Scale matrix (must be positive definite)
    /// * `dof` - Degrees of freedom (must be > p-1 where p is matrix dimension)
    ///
    /// # Returns
    ///
    /// * Wishart parameters
    pub fn new(scale: Array2<F>, dof: F) -> LinalgResult<Self> {
        let p = scale.nrows();

        if scale.nrows() != scale.ncols() {
            return Err(LinalgError::ShapeError(
                "Scale matrix must be square".to_string(),
            ));
        }

        let min_dof = F::from(p).unwrap() - F::one();
        if dof <= min_dof {
            return Err(LinalgError::InvalidInputError(format!(
                "Degrees of freedom must be > {}, got {:?}",
                min_dof, dof
            )));
        }

        Ok(Self { scale, dof })
    }
}

/// Compute the log probability density function for a matrix normal distribution
///
/// # Arguments
///
/// * `x` - Matrix to evaluate
/// * `params` - Matrix normal distribution parameters
///
/// # Returns
///
/// * Log probability density
pub fn matrix_normal_logpdf<F>(x: &ArrayView2<F>, params: &MatrixNormalParams<F>) -> LinalgResult<F>
where
    F: Float
        + Zero
        + One
        + Copy
        + std::fmt::Debug
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive
        + num_traits::NumAssign
        + std::iter::Sum,
{
    let (m, n) = x.dim();

    if params.mean.dim() != (m, n) {
        return Err(LinalgError::ShapeError(format!(
            "Matrix dimensions don't match: x is {}x{}, mean is {:?}",
            m,
            n,
            params.mean.dim()
        )));
    }

    // Compute the centered matrix: X - M
    let centered = x - &params.mean;

    // Compute log determinants
    let log_det_u = det(&params.row_cov.view(), None)?.ln();
    let log_det_v = det(&params.col_cov.view(), None)?.ln();

    // Compute inverse matrices
    let u_inv = inv(&params.row_cov.view(), None)?;
    let v_inv = inv(&params.col_cov.view(), None)?;

    // Compute the quadratic form: tr(V^{-1} * X^T * U^{-1} * X)
    let temp1 = centered.t().dot(&u_inv);
    let temp2 = temp1.dot(&centered);
    let quad_form = v_inv.dot(&temp2).diag().sum();

    // Compute the normalizing constant
    let log_2pi = F::from(2.0 * PI).unwrap().ln();
    let normalizer = -F::from(m * n).unwrap() * F::from(0.5).unwrap() * log_2pi
        - F::from(n).unwrap() * F::from(0.5).unwrap() * log_det_u
        - F::from(m).unwrap() * F::from(0.5).unwrap() * log_det_v;

    Ok(normalizer - F::from(0.5).unwrap() * quad_form)
}

/// Compute the log probability density function for a Wishart distribution
///
/// # Arguments
///
/// * `x` - Matrix to evaluate (must be positive definite)
/// * `params` - Wishart distribution parameters
///
/// # Returns
///
/// * Log probability density
pub fn wishart_logpdf<F>(x: &ArrayView2<F>, params: &WishartParams<F>) -> LinalgResult<F>
where
    F: Float
        + Zero
        + One
        + Copy
        + std::fmt::Debug
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive
        + num_traits::NumAssign
        + std::iter::Sum,
{
    let p = x.nrows();

    if x.nrows() != x.ncols() {
        return Err(LinalgError::ShapeError(
            "Matrix must be square for Wishart distribution".to_string(),
        ));
    }

    if params.scale.dim() != (p, p) {
        return Err(LinalgError::ShapeError(format!(
            "Scale matrix dimension mismatch: expected {}x{}, got {:?}",
            p,
            p,
            params.scale.dim()
        )));
    }

    // Compute log determinants
    let log_det_x = det(x, None)?.ln();
    let log_det_v = det(&params.scale.view(), None)?.ln();

    // Compute the trace term: tr(V^{-1} * X)
    let v_inv = inv(&params.scale.view(), None)?;
    let trace_term = v_inv.dot(x).diag().sum();

    // Compute the log normalizing constant
    let log_gamma_p = multivariate_log_gamma(params.dof, p)?;
    let log_2 = F::from(2.0).unwrap().ln();

    let log_normalizer = params.dof * F::from(p).unwrap() * F::from(0.5).unwrap() * log_2
        + F::from(0.25).unwrap() * F::from(p * (p - 1)).unwrap() * F::from(PI).unwrap().ln()
        + log_gamma_p
        + params.dof * F::from(0.5).unwrap() * log_det_v;

    // Compute the main term
    let main_term = (params.dof - F::from(p + 1).unwrap()) * F::from(0.5).unwrap() * log_det_x
        - F::from(0.5).unwrap() * trace_term;

    Ok(main_term - log_normalizer)
}

/// Sample from a matrix normal distribution
///
/// # Arguments
///
/// * `params` - Matrix normal distribution parameters
/// * `rng_seed` - Optional random seed
///
/// # Returns
///
/// * Random matrix sample
pub fn sample_matrix_normal<F>(
    params: &MatrixNormalParams<F>,
    rng_seed: Option<u64>,
) -> LinalgResult<Array2<F>>
where
    F: Float
        + Zero
        + One
        + Copy
        + std::fmt::Debug
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive
        + num_traits::NumAssign
        + std::iter::Sum,
{
    let (m, n) = params.mean.dim();

    // Generate standard normal matrix
    let z = random_normal_matrix((m, n), rng_seed)?;

    // Compute Cholesky factorizations
    let l_u = cholesky(&params.row_cov.view(), None)?;
    let l_v = cholesky(&params.col_cov.view(), None)?;

    // Transform: X = M + L_U * Z * L_V^T
    let temp = l_u.dot(&z);
    let sample = &params.mean + &temp.dot(&l_v.t());

    Ok(sample)
}

/// Sample from a Wishart distribution using the Bartlett decomposition
///
/// # Arguments
///
/// * `params` - Wishart distribution parameters
/// * `rng_seed` - Optional random seed
///
/// # Returns
///
/// * Random positive definite matrix sample
pub fn sample_wishart<F>(
    params: &WishartParams<F>,
    rng_seed: Option<u64>,
) -> LinalgResult<Array2<F>>
where
    F: Float
        + Zero
        + One
        + Copy
        + std::fmt::Debug
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive
        + num_traits::NumAssign
        + std::iter::Sum,
{
    let p = params.scale.nrows();

    // Use Bartlett decomposition method
    // Generate lower triangular matrix A where:
    // - A[i,i] ~ Chi(nu - i) for i = 0..p-1
    // - A[i,j] ~ N(0,1) for i > j
    // - A[i,j] = 0 for i < j

    let mut a = Array2::zeros((p, p));

    // Fill lower triangular part with random values
    // This is a simplified version - in practice you'd use proper random distributions
    let z = random_normal_matrix::<F>((p, p), rng_seed)?;

    for i in 0..p {
        for j in 0..=i {
            if i == j {
                // Diagonal: Chi-distributed (approximated as |N(0,1)| * sqrt(nu-i))
                let chi_approx = z[[i, j]].abs() * (params.dof - F::from(i).unwrap()).sqrt();
                a[[i, j]] = chi_approx;
            } else {
                // Off-diagonal: standard normal
                a[[i, j]] = z[[i, j]];
            }
        }
    }

    // Compute Cholesky factor of scale matrix
    let l = cholesky(&params.scale.view(), None)?;

    // Compute the Wishart sample: L * A * A^T * L^T
    let temp = l.dot(&a);
    let sample = temp.dot(&temp.t());

    Ok(sample)
}

/// Compute the multivariate log gamma function
///
/// Γ_p(x) = π^{p(p-1)/4} * ∏_{j=1}^p Γ(x + (1-j)/2)
fn multivariate_log_gamma<F>(x: F, p: usize) -> LinalgResult<F>
where
    F: Float + Zero + One + Copy + std::fmt::Debug + num_traits::FromPrimitive,
{
    let log_pi = F::from(PI).unwrap().ln();
    let mut result = F::from(p * (p - 1)).unwrap() * F::from(0.25).unwrap() * log_pi;

    for j in 1..=p {
        let arg = x + (F::one() - F::from(j).unwrap()) * F::from(0.5).unwrap();
        // Use log-gamma approximation (Stirling's formula for large values)
        let log_gamma_approx = if arg > F::one() {
            (arg - F::from(0.5).unwrap()) * arg.ln() - arg
                + F::from(0.5).unwrap() * F::from(2.0 * PI).unwrap().ln()
        } else {
            F::zero() // Simplified for small values
        };
        result = result + log_gamma_approx;
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_matrix_normal_params() {
        let mean = array![[1.0, 2.0], [3.0, 4.0]];
        let row_cov = array![[1.0, 0.0], [0.0, 1.0]];
        let col_cov = array![[2.0, 0.0], [0.0, 2.0]];

        let params = MatrixNormalParams::new(mean, row_cov, col_cov).unwrap();
        assert_eq!(params.mean.dim(), (2, 2));
        assert_eq!(params.row_cov.dim(), (2, 2));
        assert_eq!(params.col_cov.dim(), (2, 2));
    }

    #[test]
    fn test_wishart_params() {
        let scale = array![[2.0, 0.0], [0.0, 2.0]];
        let dof = 3.0;

        let params = WishartParams::new(scale, dof).unwrap();
        assert_abs_diff_eq!(params.dof, 3.0, epsilon = 1e-10);
        assert_eq!(params.scale.dim(), (2, 2));
    }

    #[test]
    fn test_matrix_normal_logpdf() {
        let x = array![[1.0, 0.0], [0.0, 1.0]];
        let mean = array![[0.0, 0.0], [0.0, 0.0]];
        let row_cov = array![[1.0, 0.0], [0.0, 1.0]];
        let col_cov = array![[1.0, 0.0], [0.0, 1.0]];

        let params = MatrixNormalParams::new(mean, row_cov, col_cov).unwrap();
        let logpdf = matrix_normal_logpdf(&x.view(), &params).unwrap();

        // Should be a finite value for valid inputs
        assert!(logpdf.is_finite());
    }

    #[test]
    fn test_sample_matrix_normal() {
        let mean = array![[0.0, 0.0], [0.0, 0.0]];
        let row_cov = array![[1.0, 0.0], [0.0, 1.0]];
        let col_cov = array![[1.0, 0.0], [0.0, 1.0]];

        let params = MatrixNormalParams::new(mean, row_cov, col_cov).unwrap();
        let sample = sample_matrix_normal(&params, Some(42)).unwrap();

        assert_eq!(sample.dim(), (2, 2));
        assert!(sample.iter().all(|&x| x.is_finite()));
    }
}
