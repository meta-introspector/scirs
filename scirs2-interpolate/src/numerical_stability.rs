//! Numerical Stability Monitoring and Condition Assessment
//!
//! This module provides utilities for monitoring numerical stability in interpolation
//! algorithms, particularly for matrix operations, linear solvers, and condition
//! number estimation.
//!
//! # Overview
//!
//! Numerical stability is critical for reliable interpolation, especially when:
//! - Solving linear systems with potentially ill-conditioned matrices
//! - Computing matrix factorizations and eigenvalue decompositions
//! - Performing division operations that might involve small numbers
//! - Working with matrices that approach singularity
//!
//! This module provides tools to:
//! - Estimate condition numbers efficiently
//! - Classify stability levels based on condition numbers
//! - Suggest appropriate regularization parameters
//! - Monitor for numerical issues during computation
//!
//! # Examples
//!
//! ```rust
//! use ndarray::Array2;
//! use scirs2_interpolate::numerical_stability::{assess_matrix_condition, StabilityLevel};
//!
//! // Assess the condition of a matrix
//! let matrix = Array2::<f64>::eye(3);
//! let report = assess_matrix_condition(&matrix.view()).unwrap();
//!
//! match report.stability_level {
//!     StabilityLevel::Excellent => println!("Matrix is well-conditioned"),
//!     StabilityLevel::Poor => println!("Consider regularization: {:?}",
//!                                     report.recommended_regularization),
//!     _ => println!("Condition number: {:.2e}", report.condition_number),
//! }
//! ```

use crate::error::{InterpolateError, InterpolateResult};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_traits::{Float, FromPrimitive};
use std::fmt::{Debug, Display};
use std::ops::{AddAssign, SubAssign};

/// Condition number and stability assessment report
#[derive(Debug, Clone)]
pub struct ConditionReport<F>
where
    F: Float + FromPrimitive + Debug + Display + std::ops::AddAssign + std::ops::SubAssign,
{
    /// Estimated condition number of the matrix
    pub condition_number: F,

    /// Whether the matrix is considered well-conditioned
    pub is_well_conditioned: bool,

    /// Suggested regularization parameter if needed
    pub recommended_regularization: Option<F>,

    /// Overall stability classification
    pub stability_level: StabilityLevel,

    /// Additional diagnostic information
    pub diagnostics: StabilityDiagnostics<F>,
}

/// Classification of numerical stability levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum StabilityLevel {
    /// Excellent stability (condition number < 1e12)
    Excellent,
    /// Good stability (condition number < 1e14)
    Good,
    /// Marginal stability (condition number < 1e16)
    Marginal,
    /// Poor stability (condition number >= 1e16)
    Poor,
}

/// Additional diagnostic information about matrix stability
#[derive(Debug, Clone)]
pub struct StabilityDiagnostics<F>
where
    F: Float + FromPrimitive + Debug + Display + std::ops::AddAssign + std::ops::SubAssign,
{
    /// Smallest singular value (if computed)
    pub min_singular_value: Option<F>,

    /// Largest singular value (if computed)
    pub max_singular_value: Option<F>,

    /// Matrix rank estimate
    pub estimated_rank: Option<usize>,

    /// Whether the matrix appears to be symmetric
    pub is_symmetric: bool,

    /// Whether the matrix appears to be positive definite
    pub is_positive_definite: Option<bool>,

    /// Machine epsilon for the floating point type
    pub machine_epsilon: F,
}

impl<F> Default for StabilityDiagnostics<F>
where
    F: Float + FromPrimitive + Debug + Display + std::ops::AddAssign + std::ops::SubAssign,
{
    fn default() -> Self {
        Self {
            min_singular_value: None,
            max_singular_value: None,
            estimated_rank: None,
            is_symmetric: false,
            is_positive_definite: None,
            machine_epsilon: machine_epsilon::<F>(),
        }
    }
}

/// Get machine epsilon for floating point type
pub fn machine_epsilon<F: Float + FromPrimitive>() -> F {
    match std::mem::size_of::<F>() {
        4 => F::from_f64(f32::EPSILON as f64).unwrap_or_else(|| {
            F::from(f32::EPSILON as f32)
                .unwrap_or_else(|| F::from_f64(1.19e-7).unwrap_or(F::from(1.19e-7)))
        }), // f32
        8 => F::from_f64(f64::EPSILON).unwrap_or_else(|| {
            F::from(f64::EPSILON)
                .unwrap_or_else(|| F::from_f64(2.22e-16).unwrap_or(F::from(2.22e-16)))
        }), // f64
        _ => F::from_f64(2.22e-16).unwrap_or_else(|| F::from(2.22e-16)), // Default to f64 epsilon
    }
}

/// Assess the numerical condition of a matrix
pub fn assess_matrix_condition<F>(matrix: &ArrayView2<F>) -> InterpolateResult<ConditionReport<F>>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign + 'static,
{
    if matrix.nrows() != matrix.ncols() {
        return Err(InterpolateError::ShapeMismatch {
            expected: "square matrix".to_string(),
            actual: format!("{}x{}", matrix.nrows(), matrix.ncols()),
            object: "condition assessment".to_string(),
        });
    }

    let n = matrix.nrows();
    if n == 0 {
        return Err(InterpolateError::InvalidInput {
            message: "Cannot assess condition of empty matrix".to_string(),
        });
    }

    let mut diagnostics = StabilityDiagnostics {
        is_symmetric: check_symmetry(matrix),
        ..Default::default()
    };

    // Estimate condition number
    let condition_number = estimate_condition_number(matrix, &mut diagnostics)?;

    // Classify stability level
    let stability_level = classify_stability(condition_number);

    // Determine if well-conditioned
    let is_well_conditioned = matches!(
        stability_level,
        StabilityLevel::Excellent | StabilityLevel::Good
    );

    // Suggest regularization if needed
    let recommended_regularization = if !is_well_conditioned {
        Some(suggest_regularization(condition_number, &diagnostics))
    } else {
        None
    };

    Ok(ConditionReport {
        condition_number,
        is_well_conditioned,
        recommended_regularization,
        stability_level,
        diagnostics,
    })
}

/// Check if a matrix is symmetric within numerical tolerance
fn check_symmetry<F>(matrix: &ArrayView2<F>) -> bool
where
    F: Float + FromPrimitive + Debug + Display + std::ops::AddAssign + std::ops::SubAssign,
{
    let n = matrix.nrows();
    let tol = F::from_f64(1e-12).unwrap_or_else(|| {
        machine_epsilon::<F>() * F::from(1e6).unwrap_or_else(|| F::from(1000000))
    });

    for i in 0..n {
        for j in 0..i {
            let diff = (matrix[[i, j]] - matrix[[j, i]]).abs();
            if diff > tol {
                return false;
            }
        }
    }
    true
}

/// Estimate condition number using different methods based on availability
fn estimate_condition_number<F>(
    matrix: &ArrayView2<F>,
    #[cfg_attr(not(feature = "linalg"), allow(unused_variables))]
    diagnostics: &mut StabilityDiagnostics<F>,
) -> InterpolateResult<F>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign + 'static,
{
    // Try SVD-based condition number first (most accurate)
    #[cfg(feature = "linalg")]
    {
        match estimate_condition_svd(matrix, diagnostics) {
            Ok(cond) => return Ok(cond),
            Err(_) => {
                // Fall back to other methods if SVD fails
            }
        }
    }

    // Fall back to norm-based estimation
    estimate_condition_norm_based(matrix)
}

/// Estimate condition number using SVD (requires linalg feature)
#[cfg(feature = "linalg")]
fn estimate_condition_svd<F>(
    matrix: &ArrayView2<F>,
    diagnostics: &mut StabilityDiagnostics<F>,
) -> InterpolateResult<F>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign + 'static,
{
    #[cfg(feature = "linalg")]
    {
        use ndarray_linalg::SVD;

        // Convert to f64 for SVD computation
        let matrix_f64 = matrix.mapv(|x| x.to_f64().unwrap_or(0.0));

        match matrix_f64.svd(false, false) {
            Ok((_, singular_values, _)) => {
                if singular_values.is_empty() {
                    return Err(InterpolateError::ComputationError(
                        "SVD returned empty singular values".to_string(),
                    ));
                }

                let max_sv = singular_values[0];
                let min_sv = singular_values[singular_values.len() - 1];

                // Update diagnostics
                diagnostics.max_singular_value = Some(
                    F::from_f64(max_sv)
                        .unwrap_or_else(|| F::from(max_sv as f32).unwrap_or(F::zero())),
                );
                diagnostics.min_singular_value = Some(
                    F::from_f64(min_sv)
                        .unwrap_or_else(|| F::from(min_sv as f32).unwrap_or(F::zero())),
                );

                // Estimate rank
                let eps = f64::EPSILON * max_sv * (matrix.nrows() as f64).sqrt();
                let rank = singular_values.iter().filter(|&&sv| sv > eps).count();
                diagnostics.estimated_rank = Some(rank);

                // Compute condition number
                if min_sv > f64::EPSILON {
                    Ok(F::from_f64(max_sv / min_sv).unwrap_or_else(|| F::infinity()))
                } else {
                    Ok(F::infinity()) // Singular matrix
                }
            }
            Err(_) => Err(InterpolateError::ComputationError(
                "SVD computation failed".to_string(),
            )),
        }
    }

    #[cfg(not(feature = "linalg"))]
    {
        // Fallback to norm-based estimation when linalg feature is not available
        estimate_condition_norm_based(matrix)
    }
}

/// Fallback condition number estimation using matrix norms
fn estimate_condition_norm_based<F>(matrix: &ArrayView2<F>) -> InterpolateResult<F>
where
    F: Float + FromPrimitive + Debug + Display + std::ops::AddAssign + std::ops::SubAssign,
{
    // Estimate condition number using maximum and minimum eigenvalue estimates
    let n = matrix.nrows();

    // Estimate largest and smallest eigenvalues using Gershgorin circles
    let mut min_gershgorin = F::infinity();
    let mut max_gershgorin = F::neg_infinity();

    for i in 0..n {
        let diagonal = matrix[[i, i]];
        let off_diagonal_sum = (0..n)
            .filter(|&j| j != i)
            .map(|j| matrix[[i, j]].abs())
            .fold(F::zero(), |a, b| a + b);

        let center = diagonal.abs();
        let radius = off_diagonal_sum;

        // Lower and upper bounds of Gershgorin disk
        let lower_bound = center - radius;
        let upper_bound = center + radius;

        if lower_bound > F::zero() && lower_bound < min_gershgorin {
            min_gershgorin = lower_bound;
        }

        if upper_bound > max_gershgorin {
            max_gershgorin = upper_bound;
        }
    }

    // Estimate condition number as ratio of max to min eigenvalue estimates
    if min_gershgorin.is_finite() && min_gershgorin > F::zero() && max_gershgorin.is_finite() {
        Ok(max_gershgorin / min_gershgorin)
    } else {
        // Conservative estimate for potentially ill-conditioned matrices
        Ok(F::from_f64(1e16)
            .unwrap_or_else(|| F::from(1e16 as f32).unwrap_or_else(|| F::infinity())))
    }
}

/// Classify stability level based on condition number
fn classify_stability<F>(condition_number: F) -> StabilityLevel
where
    F: Float + FromPrimitive,
{
    let threshold_1e12 = F::from_f64(1e12)
        .unwrap_or_else(|| F::from(1e12 as f32).unwrap_or_else(|| F::from(1000000000000)));
    let threshold_1e14 = F::from_f64(1e14)
        .unwrap_or_else(|| F::from(1e14 as f32).unwrap_or_else(|| F::from(100000000000000)));
    let threshold_1e16 = F::from_f64(1e16)
        .unwrap_or_else(|| F::from(1e16 as f32).unwrap_or_else(|| F::from(10000000000000000)));

    if condition_number < threshold_1e12 {
        StabilityLevel::Excellent
    } else if condition_number < threshold_1e14 {
        StabilityLevel::Good
    } else if condition_number < threshold_1e16 {
        StabilityLevel::Marginal
    } else {
        StabilityLevel::Poor
    }
}

/// Suggest regularization parameter based on condition number and diagnostics
fn suggest_regularization<F>(condition_number: F, diagnostics: &StabilityDiagnostics<F>) -> F
where
    F: Float + FromPrimitive + Debug + Display + std::ops::AddAssign + std::ops::SubAssign,
{
    let machine_eps = diagnostics.machine_epsilon;

    // Base regularization on condition number and machine epsilon
    let base_reg = machine_eps * condition_number.sqrt();

    // Adjust based on minimum singular value if available
    if let Some(min_sv) = diagnostics.min_singular_value {
        if min_sv < machine_eps {
            // Very small singular value, need stronger regularization
            base_reg
                * F::from_f64(100.0)
                    .unwrap_or_else(|| F::from(100.0 as f32).unwrap_or_else(|| F::from(100)))
        } else {
            // Moderate regularization
            base_reg
                * F::from_f64(10.0)
                    .unwrap_or_else(|| F::from(10.0 as f32).unwrap_or_else(|| F::from(10)))
        }
    } else {
        // Conservative regularization when no singular value information
        base_reg
            * F::from_f64(1000.0)
                .unwrap_or_else(|| F::from(1000.0 as f32).unwrap_or_else(|| F::from(1000)))
    }
}

/// Check if a division operation is numerically safe
pub fn check_safe_division<F>(numerator: F, denominator: F) -> InterpolateResult<F>
where
    F: Float
        + FromPrimitive
        + Debug
        + Display
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::fmt::LowerExp,
{
    let eps = machine_epsilon::<F>();
    let safe_threshold = eps
        * F::from_f64(1e6)
            .unwrap_or_else(|| F::from(1e6 as f32).unwrap_or_else(|| F::from(1000000)));

    if denominator.abs() < safe_threshold {
        Err(InterpolateError::NumericalError(format!(
            "Division by near-zero value: {} / {} (threshold: {:.2e})",
            numerator, denominator, safe_threshold
        )))
    } else {
        Ok(numerator / denominator)
    }
}

/// Check if reciprocal operation is numerically safe
pub fn safe_reciprocal<F>(value: F) -> InterpolateResult<F>
where
    F: Float
        + FromPrimitive
        + Debug
        + Display
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::fmt::LowerExp,
{
    check_safe_division(F::one(), value)
}

/// Apply Tikhonov regularization to a matrix
pub fn apply_tikhonov_regularization<F>(
    matrix: &mut Array2<F>,
    regularization: F,
) -> InterpolateResult<()>
where
    F: Float + FromPrimitive + Debug + Display + std::ops::AddAssign + std::ops::SubAssign,
{
    let n = matrix.nrows();

    if matrix.ncols() != n {
        return Err(InterpolateError::ShapeMismatch {
            expected: "square matrix".to_string(),
            actual: format!("{}x{}", n, matrix.ncols()),
            object: "Tikhonov regularization".to_string(),
        });
    }

    // Add regularization to diagonal
    for i in 0..n {
        matrix[[i, i]] += regularization;
    }

    Ok(())
}

/// Monitor and report numerical issues during matrix solve
pub fn solve_with_stability_monitoring<F>(
    matrix: &Array2<F>,
    rhs: &Array1<F>,
) -> InterpolateResult<(Array1<F>, ConditionReport<F>)>
where
    F: Float
        + FromPrimitive
        + Debug
        + Display
        + AddAssign
        + SubAssign
        + std::fmt::LowerExp
        + 'static,
{
    // Assess matrix condition first
    let condition_report = assess_matrix_condition(&matrix.view())?;

    // Warn about poor conditioning
    if matches!(condition_report.stability_level, StabilityLevel::Poor) {
        eprintln!(
            "Warning: Matrix is poorly conditioned (condition number: {:.2e}). \
             Consider regularization parameter: {:?}",
            condition_report.condition_number, condition_report.recommended_regularization
        );
    }

    // Attempt solve with regularization if recommended
    let solution = if let Some(reg) = condition_report.recommended_regularization {
        let mut regularized_matrix = matrix.clone();
        apply_tikhonov_regularization(&mut regularized_matrix, reg)?;

        // Try solve with regularized matrix
        solve_system(&regularized_matrix, rhs).or_else(|_| {
            eprintln!("Warning: Regularized solve failed, falling back to original matrix");
            solve_system(matrix, rhs)
        })?
    } else {
        solve_system(matrix, rhs)?
    };

    Ok((solution, condition_report))
}

/// Internal function to solve linear system
fn solve_system<F>(matrix: &Array2<F>, rhs: &Array1<F>) -> InterpolateResult<Array1<F>>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign + 'static,
{
    #[cfg(feature = "linalg")]
    {
        use ndarray_linalg::Solve;

        // Convert to f64 for solve
        let matrix_f64 = matrix.mapv(|x| x.to_f64().unwrap_or(0.0));
        let rhs_f64 = rhs.mapv(|x| x.to_f64().unwrap_or(0.0));

        match matrix_f64.solve(&rhs_f64) {
            Ok(solution_f64) => {
                let solution = solution_f64.mapv(|x| {
                    F::from_f64(x).unwrap_or_else(|| F::from(x as f32).unwrap_or(F::zero()))
                });
                Ok(solution)
            }
            Err(_) => Err(InterpolateError::ComputationError(
                "Linear system solve failed".to_string(),
            )),
        }
    }

    #[cfg(not(feature = "linalg"))]
    {
        // Fallback: simple Gaussian elimination for small systems
        if matrix.nrows() <= 3 {
            gaussian_elimination_small(matrix, rhs)
        } else {
            Err(InterpolateError::ComputationError(
                "Linear algebra solve requires 'linalg' feature for large systems".to_string(),
            ))
        }
    }
}

/// Simple Gaussian elimination for small systems (fallback)
#[cfg(not(feature = "linalg"))]
fn gaussian_elimination_small<F>(
    matrix: &Array2<F>,
    rhs: &Array1<F>,
) -> InterpolateResult<Array1<F>>
where
    F: Float + FromPrimitive + Debug + Display + std::ops::AddAssign + std::ops::SubAssign,
{
    let n = matrix.nrows();
    let mut a = matrix.clone();
    let mut b = rhs.clone();

    // Forward elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..n {
            if a[[k, i]].abs() > a[[max_row, i]].abs() {
                max_row = k;
            }
        }

        // Swap rows
        if max_row != i {
            for j in 0..n {
                let temp = a[[i, j]];
                a[[i, j]] = a[[max_row, j]];
                a[[max_row, j]] = temp;
            }
            let temp = b[i];
            b[i] = b[max_row];
            b[max_row] = temp;
        }

        // Check for near-zero pivot
        if a[[i, i]].abs()
            < machine_epsilon::<F>()
                * F::from_f64(1e6)
                    .unwrap_or_else(|| F::from(1e6 as f32).unwrap_or_else(|| F::from(1000000)))
        {
            return Err(InterpolateError::ComputationError(
                "Matrix is singular or nearly singular".to_string(),
            ));
        }

        // Eliminate column
        for k in (i + 1)..n {
            let factor = a[[k, i]] / a[[i, i]];
            for j in i..n {
                let a_ij = a[[i, j]];
                a[[k, j]] -= factor * a_ij;
            }
            let b_i = b[i];
            b[k] -= factor * b_i;
        }
    }

    // Back substitution
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = F::zero();
        for j in (i + 1)..n {
            sum += a[[i, j]] * x[j];
        }
        x[i] = (b[i] - sum) / a[[i, i]];
    }

    Ok(x)
}

/// Advanced edge case analysis for interpolation data
pub struct EdgeCaseAnalysis<F>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    /// Whether the data points have near-linear dependencies
    pub has_near_linear_dependencies: bool,

    /// Minimum distance between data points
    pub min_point_distance: F,

    /// Maximum distance between data points  
    pub max_point_distance: F,

    /// Ratio of max to min distance (conditioning indicator)
    pub distance_ratio: F,

    /// Number of points that are too close together
    pub clustered_points: usize,

    /// Whether data is suitable for interpolation without regularization
    pub interpolation_feasible: bool,

    /// Recommended minimum regularization based on data characteristics
    pub recommended_min_regularization: Option<F>,

    /// Analysis of boundary effects
    pub boundary_analysis: BoundaryAnalysis<F>,
}

/// Analysis of boundary effects and extrapolation stability
#[derive(Debug, Clone)]
pub struct BoundaryAnalysis<F>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    /// Whether boundary points are well-distributed
    pub boundary_well_distributed: bool,

    /// Distance to nearest boundary for domain center
    pub min_boundary_distance: F,

    /// Whether extrapolation beyond boundaries is likely to be stable
    pub extrapolation_stable: bool,

    /// Convex hull volume (if computable)
    pub convex_hull_volume: Option<F>,
}

/// Analyze interpolation data for potential edge cases and numerical issues
pub fn analyze_interpolation_edge_cases<F>(
    points: &ArrayView2<F>,
    _values: &ArrayView1<F>,
) -> InterpolateResult<EdgeCaseAnalysis<F>>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign + 'static,
{
    let n_points = points.nrows();
    let _dim = points.ncols();

    if n_points == 0 {
        return Err(InterpolateError::InvalidInput {
            message: "Cannot analyze empty dataset".to_string(),
        });
    }

    // Analyze point distances
    let (min_dist, max_dist, clustered_count) = analyze_point_distances(points)?;
    let distance_ratio = if min_dist > F::zero() {
        max_dist / min_dist
    } else {
        F::infinity()
    };

    // Check for near-linear dependencies
    let has_linear_deps = check_near_linear_dependencies(points)?;

    // Determine interpolation feasibility
    let feasible = !has_linear_deps
        && min_dist
            > machine_epsilon::<F>()
                * F::from_f64(1e6)
                    .unwrap_or_else(|| F::from(1e6 as f32).unwrap_or_else(|| F::from(1000000)));

    // Recommend regularization based on data characteristics
    let recommended_reg = if !feasible
        || distance_ratio
            > F::from_f64(1e12)
                .unwrap_or_else(|| F::from(1e12 as f32).unwrap_or_else(|| F::from(1000000000000)))
    {
        Some(suggest_data_based_regularization(min_dist, distance_ratio))
    } else {
        None
    };

    // Analyze boundary effects
    let boundary_analysis = analyze_boundary_effects(points)?;

    Ok(EdgeCaseAnalysis {
        has_near_linear_dependencies: has_linear_deps,
        min_point_distance: min_dist,
        max_point_distance: max_dist,
        distance_ratio,
        clustered_points: clustered_count,
        interpolation_feasible: feasible,
        recommended_min_regularization: recommended_reg,
        boundary_analysis,
    })
}

/// Analyze distances between data points to detect clustering issues
fn analyze_point_distances<F>(points: &ArrayView2<F>) -> InterpolateResult<(F, F, usize)>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    let n_points = points.nrows();
    let mut min_dist = F::infinity();
    let mut max_dist = F::zero();
    let mut clustered_count = 0;

    let cluster_threshold = machine_epsilon::<F>()
        * F::from_f64(1e6)
            .unwrap_or_else(|| F::from(1e6 as f32).unwrap_or_else(|| F::from(1000000)));

    for i in 0..n_points {
        for j in (i + 1)..n_points {
            // Compute Euclidean distance
            let mut dist_sq = F::zero();
            for k in 0..points.ncols() {
                let diff = points[[i, k]] - points[[j, k]];
                dist_sq += diff * diff;
            }
            let dist = dist_sq.sqrt();

            if dist < min_dist {
                min_dist = dist;
            }
            if dist > max_dist {
                max_dist = dist;
            }

            if dist < cluster_threshold {
                clustered_count += 1;
            }
        }
    }

    Ok((min_dist, max_dist, clustered_count))
}

/// Check for near-linear dependencies in data points
fn check_near_linear_dependencies<F>(points: &ArrayView2<F>) -> InterpolateResult<bool>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign + 'static,
{
    let n_points = points.nrows();
    let dim = points.ncols();

    // For fewer points than dimensions + 1, no dependencies are possible
    if n_points <= dim {
        return Ok(false);
    }

    // Check rank of the point matrix (after centering)
    let mut centered_points = points.to_owned();

    // Center the points
    for j in 0..dim {
        let mean = points.column(j).mean().unwrap_or(F::zero());
        for i in 0..n_points {
            centered_points[[i, j]] -= mean;
        }
    }

    // Try to assess rank using condition number
    if dim <= 10 && n_points <= 100 {
        // For small matrices, we can check more carefully
        let gram_matrix = compute_gram_matrix(&centered_points.view());
        let condition_report = assess_matrix_condition(&gram_matrix.view())?;

        // If condition number is very high, likely have linear dependencies
        Ok(condition_report.condition_number
            > F::from_f64(1e14).unwrap_or_else(|| {
                F::from(1e14 as f32).unwrap_or_else(|| F::from(100000000000000))
            }))
    } else {
        // For larger matrices, use a simpler heuristic
        Ok(false) // Conservative: assume no dependencies for large datasets
    }
}

/// Compute Gram matrix A^T A for rank analysis
fn compute_gram_matrix<F>(points: &ArrayView2<F>) -> Array2<F>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    let n_points = points.nrows();
    let dim = points.ncols();
    let mut gram = Array2::zeros((dim, dim));

    for i in 0..dim {
        for j in 0..dim {
            let mut sum = F::zero();
            for k in 0..n_points {
                sum += points[[k, i]] * points[[k, j]];
            }
            gram[[i, j]] = sum;
        }
    }

    gram
}

/// Suggest regularization parameter based on data characteristics
fn suggest_data_based_regularization<F>(min_distance: F, distance_ratio: F) -> F
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    let machine_eps = machine_epsilon::<F>();

    // Base regularization on minimum distance
    let distance_based = min_distance * machine_eps.sqrt();

    // Scale by distance ratio (more ill-conditioned data needs more regularization)
    let ratio_factor = if distance_ratio
        > F::from_f64(1e12)
            .unwrap_or_else(|| F::from(1e12 as f32).unwrap_or_else(|| F::from(1000000000000)))
    {
        F::from_f64(1000.0)
            .unwrap_or_else(|| F::from(1000.0 as f32).unwrap_or_else(|| F::from(1000)))
    } else if distance_ratio
        > F::from_f64(1e8)
            .unwrap_or_else(|| F::from(1e8 as f32).unwrap_or_else(|| F::from(100000000)))
    {
        F::from_f64(100.0).unwrap_or_else(|| F::from(100.0 as f32).unwrap_or_else(|| F::from(100)))
    } else {
        F::from_f64(10.0).unwrap_or_else(|| F::from(10.0 as f32).unwrap_or_else(|| F::from(10)))
    };

    distance_based * ratio_factor
}

/// Analyze boundary effects for interpolation stability  
fn analyze_boundary_effects<F>(points: &ArrayView2<F>) -> InterpolateResult<BoundaryAnalysis<F>>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    let dim = points.ncols();
    let n_points = points.nrows();

    if n_points == 0 {
        return Ok(BoundaryAnalysis {
            boundary_well_distributed: false,
            min_boundary_distance: F::zero(),
            extrapolation_stable: false,
            convex_hull_volume: None,
        });
    }

    // Find bounding box
    let mut min_coords = vec![F::infinity(); dim];
    let mut max_coords = vec![F::neg_infinity(); dim];

    for i in 0..n_points {
        for j in 0..dim {
            let coord = points[[i, j]];
            if coord < min_coords[j] {
                min_coords[j] = coord;
            }
            if coord > max_coords[j] {
                max_coords[j] = coord;
            }
        }
    }

    // Compute minimum distance to boundary
    let mut min_boundary_dist = F::infinity();
    for i in 0..n_points {
        for j in 0..dim {
            let coord = points[[i, j]];
            let dist_to_min = coord - min_coords[j];
            let dist_to_max = max_coords[j] - coord;
            let boundary_dist = dist_to_min.min(dist_to_max);
            if boundary_dist < min_boundary_dist {
                min_boundary_dist = boundary_dist;
            }
        }
    }

    // Simple heuristic for boundary distribution
    let domain_sizes: Vec<F> = (0..dim).map(|j| max_coords[j] - min_coords[j]).collect();
    let avg_domain_size = domain_sizes.iter().fold(F::zero(), |a, &b| a + b)
        / F::from_usize(dim).unwrap_or_else(|| F::from(dim as f32).unwrap_or_else(|| F::from(dim)));
    let well_distributed = min_boundary_dist
        > avg_domain_size
            / F::from_f64(10.0)
                .unwrap_or_else(|| F::from(10.0 as f32).unwrap_or_else(|| F::from(10)));

    // Extrapolation stability heuristic
    let extrapolation_stable = well_distributed && n_points >= 2 * dim + 1;

    Ok(BoundaryAnalysis {
        boundary_well_distributed: well_distributed,
        min_boundary_distance: min_boundary_dist,
        extrapolation_stable,
        convex_hull_volume: None, // Could implement convex hull volume calculation
    })
}

/// Detect potential numerical issues early in interpolation setup
pub fn early_numerical_warning_system<F>(
    points: &ArrayView2<F>,
    values: &ArrayView1<F>,
    method_name: &str,
) -> InterpolateResult<Vec<String>>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign + 'static,
{
    let mut warnings = Vec::new();

    // Analyze edge cases
    let edge_analysis = analyze_interpolation_edge_cases(points, values)?;

    if edge_analysis.has_near_linear_dependencies {
        warnings.push(format!(
            "{}: Data points have near-linear dependencies, which may cause numerical instability",
            method_name
        ));
    }

    if edge_analysis.clustered_points > 0 {
        warnings.push(format!(
            "{}: {} pairs of points are very close together (< machine epsilon * 1e6), consider removing duplicates",
            method_name, edge_analysis.clustered_points
        ));
    }

    if edge_analysis.distance_ratio
        > F::from_f64(1e12)
            .unwrap_or_else(|| F::from(1e12 as f32).unwrap_or_else(|| F::from(1000000000000)))
    {
        warnings.push(format!(
            "{}: Large variation in point distances (ratio: {:.2e}), consider data normalization",
            method_name, edge_analysis.distance_ratio
        ));
    }

    if !edge_analysis.interpolation_feasible {
        warnings.push(format!(
            "{}: Interpolation may not be feasible without regularization due to data characteristics",
            method_name
        ));

        if let Some(reg) = edge_analysis.recommended_min_regularization {
            warnings.push(format!(
                "{}: Recommended minimum regularization parameter: {:.2e}",
                method_name, reg
            ));
        }
    }

    if !edge_analysis.boundary_analysis.extrapolation_stable {
        warnings.push(format!(
            "{}: Extrapolation beyond data boundaries may be unstable",
            method_name
        ));
    }

    // Check for non-finite values
    for (i, &val) in values.iter().enumerate() {
        if !val.is_finite() {
            warnings.push(format!(
                "{}: Non-finite value found at index {}: {}",
                method_name, i, val
            ));
        }
    }

    Ok(warnings)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array2;

    #[test]
    fn test_machine_epsilon() {
        let eps_f32: f32 = machine_epsilon();
        let eps_f64: f64 = machine_epsilon();

        assert!(eps_f32 > 0.0);
        assert!(eps_f64 > 0.0);
        assert!(eps_f32 > eps_f64 as f32); // f32 has larger epsilon
    }

    #[test]
    fn test_condition_assessment_identity() {
        let matrix = Array2::<f64>::eye(3);
        let report =
            assess_matrix_condition(&matrix.view()).expect("Failed to assess matrix condition");

        assert_relative_eq!(report.condition_number, 1.0, epsilon = 1e-10);
        assert!(report.is_well_conditioned);
        assert_eq!(report.stability_level, StabilityLevel::Excellent);
        assert!(report.recommended_regularization.is_none());
    }

    #[test]
    fn test_condition_assessment_ill_conditioned() {
        let mut matrix = Array2::eye(3);
        matrix[[2, 2]] = 1e-16; // Make it ill-conditioned

        let report = assess_matrix_condition(&matrix.view()).unwrap();

        assert!(report.condition_number > 1e12);
        assert!(!report.is_well_conditioned);
        assert!(matches!(report.stability_level, StabilityLevel::Poor));
        assert!(report.recommended_regularization.is_some());
    }

    #[test]
    fn test_symmetry_check() {
        let symmetric = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 2.0, 3.0])
            .expect("Failed to create symmetric matrix");
        let asymmetric = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0])
            .expect("Failed to create asymmetric matrix");

        assert!(check_symmetry(&symmetric.view()));
        assert!(!check_symmetry(&asymmetric.view()));
    }

    #[test]
    fn test_safe_division() {
        assert!(check_safe_division(1.0, 2.0).is_ok());
        assert!(check_safe_division(1.0, 1e-20).is_err());

        assert_eq!(check_safe_division(6.0, 2.0).unwrap(), 3.0);
    }

    #[test]
    fn test_safe_reciprocal() {
        assert!(safe_reciprocal(2.0).is_ok());
        assert!(safe_reciprocal(1e-20).is_err());

        assert_relative_eq!(safe_reciprocal(4.0).unwrap(), 0.25, epsilon = 1e-10);
    }

    #[test]
    fn test_tikhonov_regularization() {
        let mut matrix = Array2::zeros((3, 3));
        matrix[[0, 0]] = 1.0;
        matrix[[1, 1]] = 2.0;
        matrix[[2, 2]] = 3.0;

        apply_tikhonov_regularization(&mut matrix, 0.1)
            .expect("Failed to apply Tikhonov regularization");

        assert_relative_eq!(matrix[[0, 0]], 1.1, epsilon = 1e-10);
        assert_relative_eq!(matrix[[1, 1]], 2.1, epsilon = 1e-10);
        assert_relative_eq!(matrix[[2, 2]], 3.1, epsilon = 1e-10);
    }

    #[test]
    #[cfg(feature = "linalg")]
    fn test_solve_with_monitoring() {
        let matrix = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 1.0, 1.0])
            .expect("Failed to create matrix");
        let rhs = Array1::from_vec(vec![3.0, 2.0]);

        let (solution, report) = solve_with_stability_monitoring(&matrix, &rhs)
            .expect("Failed to solve with stability monitoring");

        assert_relative_eq!(solution[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(solution[1], 1.0, epsilon = 1e-10);
        assert!(report.is_well_conditioned);
    }

    #[test]
    fn test_stability_classification() {
        assert_eq!(classify_stability(1e10_f64), StabilityLevel::Excellent);
        assert_eq!(classify_stability(1e13_f64), StabilityLevel::Good);
        assert_eq!(classify_stability(1e15_f64), StabilityLevel::Marginal);
        assert_eq!(classify_stability(1e17_f64), StabilityLevel::Poor);
    }
}
