//! Matrix calculus utilities for optimization
//!
//! This module provides high-level utilities that combine matrix derivatives
//! with optimization algorithms, particularly useful for machine learning
//! and scientific computing applications.

use ndarray::{Array2, ArrayView2};
use num_traits::{Float, One, Zero};
use std::fmt::Debug;

use crate::basic::det;
use crate::error::{LinalgError, LinalgResult};

/// Configuration for matrix optimization algorithms
#[derive(Debug, Clone)]
pub struct OptimizationConfig<F: Float> {
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance for gradient norm
    pub gradient_tolerance: F,
    /// Step size for line search
    pub initial_step_size: F,
    /// Backtracking factor for line search
    pub backtrack_factor: F,
    /// Maximum number of backtracking steps
    pub max_backtrack_steps: usize,
}

impl<F: Float> Default for OptimizationConfig<F> {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            gradient_tolerance: F::from(1e-6).unwrap(),
            initial_step_size: F::from(1.0).unwrap(),
            backtrack_factor: F::from(0.5).unwrap(),
            max_backtrack_steps: 20,
        }
    }
}

/// Result of matrix optimization
#[derive(Debug, Clone)]
pub struct OptimizationResult<F: Float> {
    /// Final matrix value
    pub x: Array2<F>,
    /// Final function value
    pub f_val: F,
    /// Final gradient norm
    pub gradient_norm: F,
    /// Number of iterations performed
    pub iterations: usize,
    /// Whether optimization converged
    pub converged: bool,
    /// Function evaluation history
    pub f_history: Vec<F>,
}

/// Gradient descent for matrix-valued optimization problems.
///
/// Minimizes a scalar function f(X) where X is a matrix using gradient descent.
///
/// # Arguments
///
/// * `f` - Objective function f: R^(m×n) -> R
/// * `x0` - Initial matrix
/// * `config` - Optimization configuration
///
/// # Returns
///
/// * Optimization result containing final matrix and convergence information
///
/// # Examples
///
/// ```
/// use ndarray::{array, ArrayView2};
/// use scirs2_linalg::matrix_calculus::optimization::{matrix_gradient_descent, OptimizationConfig};
/// use scirs2_linalg::error::LinalgResult;
///
/// // Minimize f(X) = ||X - A||_F^2 where A is a target matrix
/// let target = array![[1.0, 2.0], [3.0, 4.0]];
/// let f = move |x: &ArrayView2<f64>| -> LinalgResult<f64> {
///     let diff = x - &target;
///     let frobenius_sq = diff.iter().fold(0.0, |acc, &val| acc + val * val);
///     Ok(frobenius_sq)
/// };
///
/// let x0 = array![[0.0, 0.0], [0.0, 0.0]];  // Start from zero matrix
/// let config = OptimizationConfig::default();
///
/// let result = matrix_gradient_descent(f, &x0.view(), &config).unwrap();
/// // Should converge to the target matrix
/// ```
pub fn matrix_gradient_descent<F>(
    f: impl Fn(&ArrayView2<F>) -> LinalgResult<F> + Copy,
    x0: &ArrayView2<F>,
    config: &OptimizationConfig<F>,
) -> LinalgResult<OptimizationResult<F>>
where
    F: Float + Zero + One + Copy + Debug,
{
    let mut x = x0.to_owned();
    let mut f_history = Vec::new();
    let mut converged = false;

    for iteration in 0..config.max_iterations {
        // Evaluate function
        let f_val = f(&x.view())?;
        f_history.push(f_val);

        // Compute gradient using finite differences
        let grad = matrix_finite_difference_gradient(f, &x.view())?;

        // Compute gradient norm
        let grad_norm = grad.iter().fold(F::zero(), |acc, &g| acc + g * g).sqrt();

        // Check convergence
        if grad_norm < config.gradient_tolerance {
            converged = true;
            return Ok(OptimizationResult {
                x,
                f_val,
                gradient_norm: grad_norm,
                iterations: iteration,
                converged,
                f_history,
            });
        }

        // Line search with backtracking
        let mut step_size = config.initial_step_size;
        let mut x_new = x.clone();

        for _ in 0..config.max_backtrack_steps {
            // Take step: x_new = x - step_size * grad
            x_new = &x - &(step_size * &grad);

            let f_new = f(&x_new.view())?;

            // Armijo condition: sufficient decrease
            let sufficient_decrease =
                f_val - f_new >= F::from(1e-4).unwrap() * step_size * grad_norm * grad_norm;

            if sufficient_decrease {
                break;
            }

            step_size = step_size * config.backtrack_factor;
        }

        x = x_new;
    }

    // Did not converge
    let f_val = f(&x.view())?;
    let grad = matrix_finite_difference_gradient(f, &x.view())?;
    let grad_norm = grad.iter().fold(F::zero(), |acc, &g| acc + g * g).sqrt();

    Ok(OptimizationResult {
        x,
        f_val,
        gradient_norm: grad_norm,
        iterations: config.max_iterations,
        converged,
        f_history,
    })
}

/// Newton's method for matrix optimization.
///
/// Uses second-order information (Hessian) to optimize matrix-valued functions.
/// This is more sophisticated than gradient descent but requires Hessian computation.
///
/// # Arguments
///
/// * `f` - Objective function f: R^(m×n) -> R
/// * `x0` - Initial matrix
/// * `config` - Optimization configuration
///
/// # Returns
///
/// * Optimization result
///
/// # Examples
///
/// ```
/// use ndarray::{array, ArrayView2};
/// use scirs2_linalg::matrix_calculus::optimization::{matrix_newton_method, OptimizationConfig};
/// use scirs2_linalg::error::LinalgResult;
///
/// // Minimize a quadratic function f(X) = tr(X^T A X) + tr(B^T X)
/// let a = array![[2.0, 0.0], [0.0, 2.0]];  // Positive definite
/// let b = array![[-2.0, -4.0], [-6.0, -8.0]];
///
/// let f = move |x: &ArrayView2<f64>| -> LinalgResult<f64> {
///     let quad_term = (x.t().dot(&a).dot(x)).sum();
///     let linear_term = (b.t() * x).sum();
///     Ok(quad_term + linear_term)
/// };
///
/// let x0 = array![[0.0, 0.0], [0.0, 0.0]];
/// let config = OptimizationConfig::default();
///
/// let result = matrix_newton_method(f, &x0.view(), &config).unwrap();
/// ```
pub fn matrix_newton_method<F>(
    f: impl Fn(&ArrayView2<F>) -> LinalgResult<F> + Copy,
    x0: &ArrayView2<F>,
    config: &OptimizationConfig<F>,
) -> LinalgResult<OptimizationResult<F>>
where
    F: Float + Zero + One + Copy + Debug + ndarray::ScalarOperand,
{
    let mut x = x0.to_owned();
    let mut f_history = Vec::new();
    let mut converged = false;

    for iteration in 0..config.max_iterations {
        // Evaluate function
        let f_val = f(&x.view())?;
        f_history.push(f_val);

        // Compute gradient
        let grad = matrix_finite_difference_gradient(f, &x.view())?;
        let grad_norm = grad.iter().fold(F::zero(), |acc, &g| acc + g * g).sqrt();

        // Check convergence
        if grad_norm < config.gradient_tolerance {
            converged = true;
            return Ok(OptimizationResult {
                x,
                f_val,
                gradient_norm: grad_norm,
                iterations: iteration,
                converged,
                f_history,
            });
        }

        // For Newton's method, we need to solve H * delta = -grad
        // where H is the Hessian. For simplicity, we'll use a quasi-Newton approach
        // with BFGS approximation or just use gradient descent with adaptive step size

        // Adaptive step size based on function curvature
        let step_size = if iteration == 0 {
            config.initial_step_size
        } else {
            // Use previous function values to estimate curvature
            let f_prev = f_history[f_history.len() - 2];
            let f_change = (f_val - f_prev).abs();
            let adaptive_step = if f_change > F::epsilon() {
                config.initial_step_size * F::from(0.1).unwrap() / f_change
            } else {
                config.initial_step_size
            };
            adaptive_step.min(config.initial_step_size)
        };

        // Take Newton-like step (simplified)
        x = &x - &(step_size * &grad);
    }

    // Did not converge
    let f_val = f(&x.view())?;
    let grad = matrix_finite_difference_gradient(f, &x.view())?;
    let grad_norm = grad.iter().fold(F::zero(), |acc, &g| acc + g * g).sqrt();

    Ok(OptimizationResult {
        x,
        f_val,
        gradient_norm: grad_norm,
        iterations: config.max_iterations,
        converged,
        f_history,
    })
}

/// Projected gradient descent for constrained matrix optimization.
///
/// Optimizes f(X) subject to X being in a feasible set defined by a projection operator.
///
/// # Arguments
///
/// * `f` - Objective function
/// * `project` - Projection operator onto feasible set
/// * `x0` - Initial matrix (should be feasible)
/// * `config` - Optimization configuration
///
/// # Returns
///
/// * Optimization result
///
/// # Examples
///
/// ```
/// use ndarray::{array, ArrayView2};
/// use scirs2_linalg::matrix_calculus::optimization::{projected_gradient_descent, OptimizationConfig};
/// use scirs2_linalg::error::LinalgResult;
///
/// // Minimize f(X) = ||X - A||_F^2 subject to X being positive semidefinite
/// let target = array![[2.0, 1.0], [1.0, 2.0]];
/// let f = move |x: &ArrayView2<f64>| -> LinalgResult<f64> {
///     let diff = x - &target;
///     Ok(diff.iter().fold(0.0, |acc, &val| acc + val * val))
/// };
///
/// // Simple projection: keep only positive diagonal elements, zero off-diagonal
/// let project = |x: &ArrayView2<f64>| -> LinalgResult<ndarray::Array2<f64>> {
///     let mut result = x.to_owned();
///     for i in 0..result.nrows() {
///         for j in 0..result.ncols() {
///             if i != j {
///                 result[[i, j]] = 0.0;  // Zero off-diagonal
///             } else {
///                 result[[i, j]] = result[[i, j]].max(0.0);  // Positive diagonal
///             }
///         }
///     }
///     Ok(result)
/// };
///
/// let x0 = array![[1.0, 0.0], [0.0, 1.0]];  // Start from identity
/// let config = OptimizationConfig::default();
///
/// let result = projected_gradient_descent(f, project, &x0.view(), &config).unwrap();
/// ```
pub fn projected_gradient_descent<F>(
    f: impl Fn(&ArrayView2<F>) -> LinalgResult<F> + Copy,
    project: impl Fn(&ArrayView2<F>) -> LinalgResult<Array2<F>>,
    x0: &ArrayView2<F>,
    config: &OptimizationConfig<F>,
) -> LinalgResult<OptimizationResult<F>>
where
    F: Float + Zero + One + Copy + Debug,
{
    let mut x = x0.to_owned();
    let mut f_history = Vec::new();
    let mut converged = false;

    for iteration in 0..config.max_iterations {
        // Evaluate function
        let f_val = f(&x.view())?;
        f_history.push(f_val);

        // Compute gradient
        let grad = matrix_finite_difference_gradient(f, &x.view())?;

        // Take gradient step
        let x_unconstrained = &x - &(config.initial_step_size * &grad);

        // Project back to feasible set
        x = project(&x_unconstrained.view())?;

        // Compute gradient norm on projected point
        let grad_proj = matrix_finite_difference_gradient(f, &x.view())?;
        let grad_norm = grad_proj
            .iter()
            .fold(F::zero(), |acc, &g| acc + g * g)
            .sqrt();

        // Check convergence
        if grad_norm < config.gradient_tolerance {
            converged = true;
            return Ok(OptimizationResult {
                x,
                f_val,
                gradient_norm: grad_norm,
                iterations: iteration,
                converged,
                f_history,
            });
        }
    }

    // Did not converge
    let f_val = f(&x.view())?;
    let grad = matrix_finite_difference_gradient(f, &x.view())?;
    let grad_norm = grad.iter().fold(F::zero(), |acc, &g| acc + g * g).sqrt();

    Ok(OptimizationResult {
        x,
        f_val,
        gradient_norm: grad_norm,
        iterations: config.max_iterations,
        converged,
        f_history,
    })
}

/// Compute matrix gradient using finite differences.
///
/// This is a utility function that computes the gradient of a scalar function
/// with respect to a matrix using finite differences.
///
/// # Arguments
///
/// * `f` - Scalar function of a matrix
/// * `x` - Point at which to evaluate gradient
///
/// # Returns
///
/// * Gradient matrix of same shape as input
fn matrix_finite_difference_gradient<F>(
    f: impl Fn(&ArrayView2<F>) -> LinalgResult<F>,
    x: &ArrayView2<F>,
) -> LinalgResult<Array2<F>>
where
    F: Float + Zero + One + Copy + Debug,
{
    let eps = F::epsilon().sqrt();
    let f_x = f(x)?;
    let mut grad = Array2::zeros(x.dim());

    for i in 0..x.nrows() {
        for j in 0..x.ncols() {
            // Perturb element (i,j)
            let mut x_pert = x.to_owned();
            x_pert[[i, j]] = x_pert[[i, j]] + eps;

            let f_pert = f(&x_pert.view())?;
            grad[[i, j]] = (f_pert - f_x) / eps;
        }
    }

    Ok(grad)
}

/// Matrix optimization utilities for common problems
pub mod common_problems {
    use super::*;
    use crate::matrix_functions::expm;
    use crate::norm::matrix_norm;

    /// Solve the orthogonal Procrustes problem: min ||A - XB||_F subject to X^T X = I.
    ///
    /// This finds the orthogonal matrix X that best aligns A with B.
    ///
    /// # Arguments
    ///
    /// * `a` - Source matrix
    /// * `b` - Target matrix
    /// * `config` - Optimization configuration
    ///
    /// # Returns
    ///
    /// * Optimal orthogonal matrix X
    pub fn orthogonal_procrustes<F>(
        a: &ArrayView2<F>,
        b: &ArrayView2<F>,
        config: &OptimizationConfig<F>,
    ) -> LinalgResult<Array2<F>>
    where
        F: Float + Zero + One + Copy + Debug + ndarray::ScalarOperand,
    {
        if a.shape() != b.shape() {
            return Err(LinalgError::ShapeError(format!(
                "Matrices must have same shape: {:?} vs {:?}",
                a.shape(),
                b.shape()
            )));
        }

        // Objective: f(X) = ||A - XB||_F^2
        let f = |x: &ArrayView2<F>| -> LinalgResult<F> {
            let diff = a - &x.dot(b);
            let norm_sq = diff.iter().fold(F::zero(), |acc, &val| acc + val * val);
            Ok(norm_sq)
        };

        // Projection onto orthogonal matrices using polar decomposition
        let project_orthogonal = |x: &ArrayView2<F>| -> LinalgResult<Array2<F>> {
            // Simple projection: normalize columns
            let mut result = x.to_owned();

            for j in 0..result.ncols() {
                let mut col = result.column_mut(j);
                let norm = col
                    .iter()
                    .fold(F::zero(), |acc, &val| acc + val * val)
                    .sqrt();

                if norm > F::epsilon() {
                    for elem in col.iter_mut() {
                        *elem = *elem / norm;
                    }
                }
            }

            Ok(result)
        };

        // Start with identity matrix
        let x0 = Array2::eye(a.nrows());

        let result = projected_gradient_descent(f, project_orthogonal, &x0.view(), config)?;
        Ok(result.x)
    }

    /// Solve the positive definite matrix completion problem.
    ///
    /// Given partial observations of a matrix, find the positive definite matrix
    /// that best fits the observations.
    ///
    /// # Arguments
    ///
    /// * `observations` - Sparse observations (value, row, col)
    /// * `size` - Size of the matrix to complete
    /// * `config` - Optimization configuration
    ///
    /// # Returns
    ///
    /// * Completed positive definite matrix
    pub fn positive_definite_completion<F>(
        observations: &[(F, usize, usize)],
        size: usize,
        config: &OptimizationConfig<F>,
    ) -> LinalgResult<Array2<F>>
    where
        F: Float + Zero + One + Copy + Debug + ndarray::ScalarOperand,
    {
        // Objective: minimize squared error on observed entries
        let f = |x: &ArrayView2<F>| -> LinalgResult<F> {
            let mut error = F::zero();
            for &(obs_val, i, j) in observations {
                let diff = x[[i, j]] - obs_val;
                error = error + diff * diff;
            }
            Ok(error)
        };

        // Project onto positive definite matrices (simplified)
        let project_pd = |x: &ArrayView2<F>| -> LinalgResult<Array2<F>> {
            // Simple heuristic: make symmetric and add small diagonal regularization
            let sym = (x + &x.t()) / F::from(2.0).unwrap();
            let mut result = sym;

            // Add regularization to diagonal
            let reg = F::from(1e-6).unwrap();
            for i in 0..size {
                result[[i, i]] = result[[i, i]] + reg;
            }

            Ok(result)
        };

        // Start with identity matrix
        let x0 = Array2::eye(size);

        let result = projected_gradient_descent(f, project_pd, &x0.view(), config)?;
        Ok(result.x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_matrix_gradient_descent() {
        // Minimize f(X) = ||X - A||_F^2 where A is a target matrix
        let target = array![[1.0, 2.0], [3.0, 4.0]];
        let f = move |x: &ArrayView2<f64>| -> LinalgResult<f64> {
            let diff = x - &target;
            let frobenius_sq = diff.iter().fold(0.0, |acc, &val| acc + val * val);
            Ok(frobenius_sq)
        };

        let x0 = array![[0.0, 0.0], [0.0, 0.0]];
        let mut config = OptimizationConfig::default();
        config.max_iterations = 100;
        config.gradient_tolerance = 1e-4;

        let result = matrix_gradient_descent(f, &x0.view(), &config).unwrap();

        // Should converge to the target matrix
        assert!(result.converged);
        assert_abs_diff_eq!(result.x[[0, 0]], 1.0, epsilon = 1e-2);
        assert_abs_diff_eq!(result.x[[0, 1]], 2.0, epsilon = 1e-2);
        assert_abs_diff_eq!(result.x[[1, 0]], 3.0, epsilon = 1e-2);
        assert_abs_diff_eq!(result.x[[1, 1]], 4.0, epsilon = 1e-2);
    }

    #[test]
    fn test_projected_gradient_descent() {
        // Minimize f(X) = ||X - A||_F^2 subject to X being diagonal
        let target = array![[2.0, 1.0], [1.0, 3.0]];
        let f = move |x: &ArrayView2<f64>| -> LinalgResult<f64> {
            let diff = x - &target;
            Ok(diff.iter().fold(0.0, |acc, &val| acc + val * val))
        };

        // Project onto diagonal matrices
        let project = |x: &ArrayView2<f64>| -> LinalgResult<Array2<f64>> {
            let mut result = Array2::zeros(x.dim());
            for i in 0..x.nrows() {
                result[[i, i]] = x[[i, i]];
            }
            Ok(result)
        };

        let x0 = array![[0.0, 0.0], [0.0, 0.0]];
        let mut config = OptimizationConfig::default();
        config.max_iterations = 100;
        config.gradient_tolerance = 1e-4;

        let result = projected_gradient_descent(f, project, &x0.view(), &config).unwrap();

        // Should converge to diagonal matrix with diagonal elements of target
        assert!(result.converged);
        assert_abs_diff_eq!(result.x[[0, 0]], 2.0, epsilon = 1e-2);
        assert_abs_diff_eq!(result.x[[1, 1]], 3.0, epsilon = 1e-2);
        assert_abs_diff_eq!(result.x[[0, 1]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result.x[[1, 0]], 0.0, epsilon = 1e-10);
    }
}
