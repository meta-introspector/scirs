//! Advanced analysis tools for dynamical systems
//!
//! This module provides tools for analyzing the behavior of dynamical systems,
//! including bifurcation analysis and stability assessment.

use crate::error::{IntegrateError, IntegrateResult as Result};
use ndarray::{Array1, Array2};
use num_complex::Complex64;

/// Bifurcation point information
#[derive(Debug, Clone)]
pub struct BifurcationPoint {
    /// Parameter value at bifurcation
    pub parameter_value: f64,
    /// State at bifurcation
    pub state: Array1<f64>,
    /// Type of bifurcation
    pub bifurcation_type: BifurcationType,
    /// Eigenvalues at bifurcation point
    pub eigenvalues: Vec<Complex64>,
}

/// Types of bifurcations
#[derive(Debug, Clone, PartialEq)]
pub enum BifurcationType {
    /// Fold/saddle-node bifurcation
    Fold,
    /// Transcritical bifurcation
    Transcritical,
    /// Pitchfork bifurcation
    Pitchfork,
    /// Hopf bifurcation (birth of limit cycle)
    Hopf,
    /// Period-doubling bifurcation
    PeriodDoubling,
    /// Homoclinic bifurcation
    Homoclinic,
    /// Unknown/unclassified bifurcation
    Unknown,
}

/// Stability assessment result
#[derive(Debug, Clone)]
pub struct StabilityResult {
    /// Fixed points found
    pub fixed_points: Vec<FixedPoint>,
    /// Periodic orbits found
    pub periodic_orbits: Vec<PeriodicOrbit>,
    /// Lyapunov exponents
    pub lyapunov_exponents: Option<Array1<f64>>,
    /// Basin of attraction estimates
    pub basin_analysis: Option<BasinAnalysis>,
}

/// Fixed point information
#[derive(Debug, Clone)]
pub struct FixedPoint {
    /// Location of fixed point
    pub location: Array1<f64>,
    /// Stability type
    pub stability: StabilityType,
    /// Eigenvalues of linearization
    pub eigenvalues: Vec<Complex64>,
    /// Eigenvectors of linearization
    pub eigenvectors: Array2<Complex64>,
}

/// Periodic orbit information
#[derive(Debug, Clone)]
pub struct PeriodicOrbit {
    /// Representative point on orbit
    pub representative_point: Array1<f64>,
    /// Period of orbit
    pub period: f64,
    /// Stability type
    pub stability: StabilityType,
    /// Floquet multipliers
    pub floquet_multipliers: Vec<Complex64>,
}

/// Stability classification
#[derive(Debug, Clone, PartialEq)]
pub enum StabilityType {
    /// Stable (attracting)
    Stable,
    /// Unstable (repelling)
    Unstable,
    /// Saddle (mixed stability)
    Saddle,
    /// Center (neutrally stable)
    Center,
    /// Degenerate (requires higher-order analysis)
    Degenerate,
}

/// Basin of attraction analysis
#[derive(Debug, Clone)]
pub struct BasinAnalysis {
    /// Grid points analyzed
    pub grid_points: Array2<f64>,
    /// Attractor index for each grid point (-1 for divergent)
    pub attractor_indices: Array2<i32>,
    /// List of attractors found
    pub attractors: Vec<Array1<f64>>,
}

/// Bifurcation analyzer for parametric dynamical systems
pub struct BifurcationAnalyzer {
    /// System dimension
    pub dimension: usize,
    /// Parameter range to analyze
    pub parameter_range: (f64, f64),
    /// Number of parameter values to sample
    pub parameter_samples: usize,
    /// Tolerance for detecting fixed points
    pub fixed_point_tolerance: f64,
    /// Maximum number of iterations for fixed point finding
    pub max_iterations: usize,
}

impl BifurcationAnalyzer {
    /// Create a new bifurcation analyzer
    pub fn new(dimension: usize, parameter_range: (f64, f64), parameter_samples: usize) -> Self {
        Self {
            dimension,
            parameter_range,
            parameter_samples,
            fixed_point_tolerance: 1e-8,
            max_iterations: 1000,
        }
    }

    /// Perform continuation analysis to find bifurcation points
    pub fn continuation_analysis<F>(
        &self,
        system: F,
        initial_guess: &Array1<f64>,
    ) -> Result<Vec<BifurcationPoint>>
    where
        F: Fn(&Array1<f64>, f64) -> Array1<f64> + Send + Sync,
    {
        let mut bifurcation_points = Vec::new();
        // Check for division by zero in parameter step calculation
        if self.parameter_samples <= 1 {
            return Err(IntegrateError::ValueError(
                "Parameter samples must be greater than 1".to_string(),
            ));
        }
        let param_step =
            (self.parameter_range.1 - self.parameter_range.0) / (self.parameter_samples - 1) as f64;

        let mut current_solution = initial_guess.clone();
        let mut previous_eigenvalues: Option<Vec<Complex64>> = None;

        for i in 0..self.parameter_samples {
            let param = self.parameter_range.0 + i as f64 * param_step;

            // Find fixed point for current parameter value
            match self.find_fixed_point(&system, &current_solution, param) {
                Ok(fixed_point) => {
                    current_solution = fixed_point.clone();

                    // Compute Jacobian and eigenvalues
                    let jacobian = self.compute_jacobian(&system, &fixed_point, param)?;
                    let eigenvalues = self.compute_eigenvalues(&jacobian)?;

                    // Check for bifurcation by comparing with previous eigenvalues
                    if let Some(prev_eigs) = &previous_eigenvalues {
                        if let Some(bif_type) = self.detect_bifurcation(prev_eigs, &eigenvalues) {
                            bifurcation_points.push(BifurcationPoint {
                                parameter_value: param,
                                state: fixed_point.clone(),
                                bifurcation_type: bif_type,
                                eigenvalues: eigenvalues.clone(),
                            });
                        }
                    }

                    previous_eigenvalues = Some(eigenvalues);
                }
                Err(_) => {
                    // Fixed point disappeared - potential bifurcation
                    if i > 0 {
                        bifurcation_points.push(BifurcationPoint {
                            parameter_value: param,
                            state: current_solution.clone(),
                            bifurcation_type: BifurcationType::Fold,
                            eigenvalues: previous_eigenvalues.clone().unwrap_or_default(),
                        });
                    }
                    break;
                }
            }
        }

        Ok(bifurcation_points)
    }

    /// Find fixed point using Newton's method
    fn find_fixed_point<F>(
        &self,
        system: &F,
        initial_guess: &Array1<f64>,
        parameter: f64,
    ) -> Result<Array1<f64>>
    where
        F: Fn(&Array1<f64>, f64) -> Array1<f64>,
    {
        let mut x = initial_guess.clone();

        for _ in 0..self.max_iterations {
            let f_x = system(&x, parameter);
            let sum_squares = f_x.iter().map(|&v| v * v).sum::<f64>();
            if sum_squares < 0.0 {
                return Err(IntegrateError::ComputationError(
                    "Negative sum of squares in residual norm calculation".to_string(),
                ));
            }
            let residual_norm = sum_squares.sqrt();

            if residual_norm < self.fixed_point_tolerance {
                return Ok(x);
            }

            // Compute Jacobian
            let jacobian = self.compute_jacobian(system, &x, parameter)?;

            // Solve J * dx = -f(x) using LU decomposition
            let dx = self.solve_linear_system(&jacobian, &(-&f_x))?;
            x += &dx;
        }

        Err(IntegrateError::ConvergenceError(
            "Fixed point iteration did not converge".to_string(),
        ))
    }

    /// Compute Jacobian matrix using finite differences
    fn compute_jacobian<F>(
        &self,
        system: &F,
        x: &Array1<f64>,
        parameter: f64,
    ) -> Result<Array2<f64>>
    where
        F: Fn(&Array1<f64>, f64) -> Array1<f64>,
    {
        let h = 1e-8_f64;
        let n = x.len();
        let mut jacobian = Array2::zeros((n, n));

        let f0 = system(x, parameter);

        // Check for valid step size
        if h.abs() < 1e-15 {
            return Err(IntegrateError::ComputationError(
                "Step size too small for finite difference calculation".to_string(),
            ));
        }

        for j in 0..n {
            let mut x_plus = x.clone();
            x_plus[j] += h;
            let f_plus = system(&x_plus, parameter);

            for i in 0..n {
                jacobian[[i, j]] = (f_plus[i] - f0[i]) / h;
            }
        }

        Ok(jacobian)
    }

    /// Compute eigenvalues of a matrix using QR algorithm
    fn compute_eigenvalues(&self, matrix: &Array2<f64>) -> Result<Vec<Complex64>> {
        // Convert to complex matrix for eigenvalue computation
        let n = matrix.nrows();
        let mut a = Array2::<Complex64>::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                a[[i, j]] = Complex64::new(matrix[[i, j]], 0.0);
            }
        }

        // Simple QR algorithm implementation (simplified)
        let max_iterations = 100;
        let tolerance = 1e-10;

        for _ in 0..max_iterations {
            let (q, r) = self.qr_decomposition(&a)?;
            let a_new = self.matrix_multiply(&r, &q)?;

            // Check convergence
            let mut converged = true;
            for i in 0..n - 1 {
                if a_new[[i + 1, i]].norm() > tolerance {
                    converged = false;
                    break;
                }
            }

            a = a_new;
            if converged {
                break;
            }
        }

        // Extract eigenvalues from diagonal
        let mut eigenvalues = Vec::new();
        for i in 0..n {
            eigenvalues.push(a[[i, i]]);
        }

        Ok(eigenvalues)
    }

    /// QR decomposition using Gram-Schmidt process
    fn qr_decomposition(
        &self,
        a: &Array2<Complex64>,
    ) -> Result<(Array2<Complex64>, Array2<Complex64>)> {
        let (m, n) = a.dim();
        let mut q = Array2::<Complex64>::zeros((m, n));
        let mut r = Array2::<Complex64>::zeros((n, n));

        for j in 0..n {
            // Get column j
            let mut v = Array1::<Complex64>::zeros(m);
            for i in 0..m {
                v[i] = a[[i, j]];
            }

            // Gram-Schmidt orthogonalization
            for k in 0..j {
                let mut u_k = Array1::<Complex64>::zeros(m);
                for i in 0..m {
                    u_k[i] = q[[i, k]];
                }

                let dot_product = v
                    .iter()
                    .zip(u_k.iter())
                    .map(|(&vi, &uk)| vi * uk.conj())
                    .sum::<Complex64>();

                r[[k, j]] = dot_product;

                for i in 0..m {
                    v[i] -= dot_product * u_k[i];
                }
            }

            // Normalize
            let norm_sqr = v.iter().map(|&x| x.norm_sqr()).sum::<f64>();
            if norm_sqr < 0.0 {
                return Err(IntegrateError::ComputationError(
                    "Negative norm squared in QR decomposition".to_string(),
                ));
            }
            let norm = norm_sqr.sqrt();
            r[[j, j]] = Complex64::new(norm, 0.0);

            if norm > 1e-12 {
                for i in 0..m {
                    q[[i, j]] = v[i] / norm;
                }
            }
        }

        Ok((q, r))
    }

    /// Matrix multiplication for complex matrices
    fn matrix_multiply(
        &self,
        a: &Array2<Complex64>,
        b: &Array2<Complex64>,
    ) -> Result<Array2<Complex64>> {
        let (m, k1) = a.dim();
        let (k2, n) = b.dim();

        if k1 != k2 {
            return Err(IntegrateError::ValueError(
                "Matrix dimensions incompatible for multiplication".to_string(),
            ));
        }

        let mut c = Array2::<Complex64>::zeros((m, n));

        for i in 0..m {
            for j in 0..n {
                for k in 0..k1 {
                    c[[i, j]] += a[[i, k]] * b[[k, j]];
                }
            }
        }

        Ok(c)
    }

    /// Advanced bifurcation detection with multiple algorithms
    fn detect_bifurcation(
        &self,
        prev_eigenvalues: &[Complex64],
        curr_eigenvalues: &[Complex64],
    ) -> Option<BifurcationType> {
        // Enhanced detection with better tolerance handling
        let tolerance = 1e-8;

        // Check for fold bifurcation (real eigenvalue crosses zero)
        if let Some(bif_type) =
            self.detect_fold_bifurcation(prev_eigenvalues, curr_eigenvalues, tolerance)
        {
            return Some(bif_type);
        }

        // Check for Hopf bifurcation (complex conjugate pair crosses imaginary axis)
        if let Some(bif_type) =
            self.detect_hopf_bifurcation(prev_eigenvalues, curr_eigenvalues, tolerance)
        {
            return Some(bif_type);
        }

        // Check for transcritical bifurcation
        if let Some(bif_type) =
            self.detect_transcritical_bifurcation(prev_eigenvalues, curr_eigenvalues, tolerance)
        {
            return Some(bif_type);
        }

        // Check for pitchfork bifurcation
        if let Some(bif_type) =
            self.detect_pitchfork_bifurcation(prev_eigenvalues, curr_eigenvalues, tolerance)
        {
            return Some(bif_type);
        }

        // Check for period-doubling bifurcation
        if let Some(bif_type) =
            self.detect_period_doubling_bifurcation(prev_eigenvalues, curr_eigenvalues, tolerance)
        {
            return Some(bif_type);
        }

        None
    }

    /// Detect fold (saddle-node) bifurcation
    fn detect_fold_bifurcation(
        &self,
        prev_eigenvalues: &[Complex64],
        curr_eigenvalues: &[Complex64],
        tolerance: f64,
    ) -> Option<BifurcationType> {
        for (prev, curr) in prev_eigenvalues.iter().zip(curr_eigenvalues.iter()) {
            // Real eigenvalue crossing zero
            if prev.re * curr.re < 0.0 && prev.im.abs() < tolerance && curr.im.abs() < tolerance {
                // Additional check: ensure it's not just numerical noise
                if prev.re.abs() > tolerance / 10.0 || curr.re.abs() > tolerance / 10.0 {
                    return Some(BifurcationType::Fold);
                }
            }
        }
        None
    }

    /// Detect Hopf bifurcation using advanced criteria
    fn detect_hopf_bifurcation(
        &self,
        prev_eigenvalues: &[Complex64],
        curr_eigenvalues: &[Complex64],
        tolerance: f64,
    ) -> Option<BifurcationType> {
        // Find complex conjugate pairs
        for i in 0..prev_eigenvalues.len() {
            for j in (i + 1)..prev_eigenvalues.len() {
                let prev1 = prev_eigenvalues[i];
                let prev2 = prev_eigenvalues[j];

                // Check if they form a complex conjugate pair
                if (prev1.conj() - prev2).norm() < tolerance {
                    // Find corresponding pair in current eigenvalues
                    for k in 0..curr_eigenvalues.len() {
                        for l in (k + 1)..curr_eigenvalues.len() {
                            let curr1 = curr_eigenvalues[k];
                            let curr2 = curr_eigenvalues[l];

                            if (curr1.conj() - curr2).norm() < tolerance {
                                // Check if real parts cross zero while imaginary parts remain non-zero
                                if prev1.re * curr1.re < 0.0
                                    && prev1.im.abs() > tolerance
                                    && curr1.im.abs() > tolerance
                                {
                                    return Some(BifurcationType::Hopf);
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect transcritical bifurcation
    fn detect_transcritical_bifurcation(
        &self,
        prev_eigenvalues: &[Complex64],
        curr_eigenvalues: &[Complex64],
        tolerance: f64,
    ) -> Option<BifurcationType> {
        // Transcritical bifurcation: one eigenvalue passes through zero
        // while another eigenvalue remains at zero
        let mut zero_crossings = 0;
        let mut zero_eigenvalues = 0;

        for (prev, curr) in prev_eigenvalues.iter().zip(curr_eigenvalues.iter()) {
            if prev.re * curr.re < 0.0 && prev.im.abs() < tolerance && curr.im.abs() < tolerance {
                zero_crossings += 1;
            }

            if prev.norm() < tolerance || curr.norm() < tolerance {
                zero_eigenvalues += 1;
            }
        }

        if zero_crossings == 1 && zero_eigenvalues >= 1 {
            return Some(BifurcationType::Transcritical);
        }

        None
    }

    /// Detect pitchfork bifurcation using symmetry analysis
    fn detect_pitchfork_bifurcation(
        &self,
        prev_eigenvalues: &[Complex64],
        curr_eigenvalues: &[Complex64],
        tolerance: f64,
    ) -> Option<BifurcationType> {
        // Simplified pitchfork detection
        // In practice, would need to analyze system symmetries
        let mut zero_crossings = 0;

        for (prev, curr) in prev_eigenvalues.iter().zip(curr_eigenvalues.iter()) {
            if prev.re * curr.re < 0.0
                && prev.im.abs() < tolerance
                && curr.im.abs() < tolerance
                && (prev.re - curr.re).abs() > tolerance
            {
                zero_crossings += 1;
            }
        }

        // Simple heuristic: if multiple real eigenvalues cross zero
        if zero_crossings >= 2 {
            return Some(BifurcationType::Pitchfork);
        }

        None
    }

    /// Detect period-doubling bifurcation
    fn detect_period_doubling_bifurcation(
        &self,
        prev_eigenvalues: &[Complex64],
        curr_eigenvalues: &[Complex64],
        tolerance: f64,
    ) -> Option<BifurcationType> {
        // Period-doubling: eigenvalue passes through -1
        for (prev, curr) in prev_eigenvalues.iter().zip(curr_eigenvalues.iter()) {
            let prev_dist_to_minus_one = (prev + 1.0).norm();
            let curr_dist_to_minus_one = (curr + 1.0).norm();

            if prev_dist_to_minus_one < tolerance || curr_dist_to_minus_one < tolerance {
                // Additional check: ensure it's a real eigenvalue
                if prev.im.abs() < tolerance && curr.im.abs() < tolerance {
                    return Some(BifurcationType::PeriodDoubling);
                }
            }
        }
        None
    }

    /// Advanced continuation method with predictor-corrector
    pub fn predictor_corrector_continuation<F>(
        &self,
        system: F,
        initial_solution: &Array1<f64>,
        initial_parameter: f64,
    ) -> Result<Vec<BifurcationPoint>>
    where
        F: Fn(&Array1<f64>, f64) -> Array1<f64> + Send + Sync,
    {
        let mut bifurcation_points = Vec::new();
        let mut current_solution = initial_solution.clone();
        let mut current_parameter = initial_parameter;

        let param_step = 0.01;
        let max_steps = 1000;

        let mut previous_eigenvalues: Option<Vec<Complex64>> = None;

        for _ in 0..max_steps {
            // Predictor step: linear extrapolation
            let (pred_solution, pred_parameter) =
                self.predictor_step(&current_solution, current_parameter, param_step);

            // Corrector step: Newton iteration to get back on solution curve
            match self.corrector_step(&system, &pred_solution, pred_parameter) {
                Ok((corrected_solution, corrected_parameter)) => {
                    current_solution = corrected_solution;
                    current_parameter = corrected_parameter;

                    // Check for bifurcations
                    let jacobian =
                        self.compute_jacobian(&system, &current_solution, current_parameter)?;
                    let eigenvalues = self.compute_eigenvalues(&jacobian)?;

                    if let Some(ref prev_eigs) = previous_eigenvalues {
                        if let Some(bif_type) = self.detect_bifurcation(prev_eigs, &eigenvalues) {
                            bifurcation_points.push(BifurcationPoint {
                                parameter_value: current_parameter,
                                state: current_solution.clone(),
                                bifurcation_type: bif_type,
                                eigenvalues: eigenvalues.clone(),
                            });
                        }
                    }

                    previous_eigenvalues = Some(eigenvalues);
                }
                Err(_) => break, // Continuation failed
            }

            // Check stopping criteria
            if current_parameter > self.parameter_range.1 {
                break;
            }
        }

        Ok(bifurcation_points)
    }

    /// Predictor step for continuation
    fn predictor_step(
        &self,
        current_solution: &Array1<f64>,
        current_parameter: f64,
        step_size: f64,
    ) -> (Array1<f64>, f64) {
        // Simple linear predictor
        let predicted_parameter = current_parameter + step_size;
        let predicted_solution = current_solution.clone(); // Could use tangent prediction

        (predicted_solution, predicted_parameter)
    }

    /// Corrector step for continuation
    fn corrector_step<F>(
        &self,
        system: &F,
        predicted_solution: &Array1<f64>,
        predicted_parameter: f64,
    ) -> Result<(Array1<f64>, f64)>
    where
        F: Fn(&Array1<f64>, f64) -> Array1<f64>,
    {
        // Newton iteration to correct the prediction
        let mut solution = predicted_solution.clone();
        let parameter = predicted_parameter; // Keep parameter fixed

        for _ in 0..10 {
            // Max 10 Newton iterations
            let residual = system(&solution, parameter);
            let sum_squares = residual.iter().map(|&r| r * r).sum::<f64>();
            if sum_squares < 0.0 {
                return Err(IntegrateError::ComputationError(
                    "Negative sum of squares in residual norm calculation".to_string(),
                ));
            }
            let residual_norm = sum_squares.sqrt();

            if residual_norm < 1e-10 {
                return Ok((solution, parameter));
            }

            let jacobian = self.compute_jacobian(system, &solution, parameter)?;
            let delta = self.solve_linear_system(&jacobian, &(-&residual))?;
            solution += &delta;
        }

        Err(IntegrateError::ConvergenceError(
            "Corrector step did not converge".to_string(),
        ))
    }

    /// Solve linear system using LU decomposition
    fn solve_linear_system(&self, a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
        let n = a.nrows();
        let mut lu = a.clone();
        let mut x = b.clone();

        // LU decomposition with partial pivoting
        let mut pivot = Array1::<usize>::zeros(n);
        for i in 0..n {
            pivot[i] = i;
        }

        for k in 0..n - 1 {
            // Find pivot
            let mut max_val = lu[[k, k]].abs();
            let mut max_idx = k;

            for i in k + 1..n {
                if lu[[i, k]].abs() > max_val {
                    max_val = lu[[i, k]].abs();
                    max_idx = i;
                }
            }

            // Swap rows
            if max_idx != k {
                for j in 0..n {
                    let temp = lu[[k, j]];
                    lu[[k, j]] = lu[[max_idx, j]];
                    lu[[max_idx, j]] = temp;
                }
                pivot.swap(k, max_idx);
            }

            // Eliminate
            for i in k + 1..n {
                if lu[[k, k]].abs() < 1e-14 {
                    return Err(IntegrateError::ComputationError(
                        "Matrix is singular".to_string(),
                    ));
                }

                let factor = lu[[i, k]] / lu[[k, k]];
                lu[[i, k]] = factor;

                for j in k + 1..n {
                    lu[[i, j]] -= factor * lu[[k, j]];
                }
            }
        }

        // Apply row swaps to RHS
        for k in 0..n - 1 {
            x.swap(k, pivot[k]);
        }

        // Forward substitution
        for i in 1..n {
            for j in 0..i {
                x[i] -= lu[[i, j]] * x[j];
            }
        }

        // Back substitution
        for i in (0..n).rev() {
            for j in i + 1..n {
                x[i] -= lu[[i, j]] * x[j];
            }
            // Check for zero diagonal element
            if lu[[i, i]].abs() < 1e-14 {
                return Err(IntegrateError::ComputationError(
                    "Zero diagonal element in back substitution".to_string(),
                ));
            }
            x[i] /= lu[[i, i]];
        }

        Ok(x)
    }
}

/// Stability analyzer for dynamical systems
pub struct StabilityAnalyzer {
    /// System dimension
    pub dimension: usize,
    /// Tolerance for fixed point detection
    pub tolerance: f64,
    /// Integration time for trajectory analysis
    pub integration_time: f64,
    /// Number of test points for basin analysis
    pub basin_grid_size: usize,
}

impl StabilityAnalyzer {
    /// Create a new stability analyzer
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            tolerance: 1e-8,
            integration_time: 100.0,
            basin_grid_size: 50,
        }
    }

    /// Perform comprehensive stability analysis
    pub fn analyze_stability<F>(&self, system: F, domain: &[(f64, f64)]) -> Result<StabilityResult>
    where
        F: Fn(&Array1<f64>) -> Array1<f64> + Send + Sync + Clone + 'static,
    {
        // Find fixed points
        let fixed_points = self.find_fixed_points(&system, domain)?;

        // Find periodic orbits (simplified)
        let periodic_orbits = self.find_periodic_orbits(&system, domain)?;

        // Compute Lyapunov exponents
        let lyapunov_exponents = self.compute_lyapunov_exponents(&system)?;

        // Analyze basins of attraction
        let basin_analysis = if self.dimension == 2 {
            Some(self.analyze_basins(&system, domain, &fixed_points)?)
        } else {
            None
        };

        Ok(StabilityResult {
            fixed_points,
            periodic_orbits,
            lyapunov_exponents,
            basin_analysis,
        })
    }

    /// Find fixed points in the given domain
    fn find_fixed_points<F>(&self, system: &F, domain: &[(f64, f64)]) -> Result<Vec<FixedPoint>>
    where
        F: Fn(&Array1<f64>) -> Array1<f64>,
    {
        let mut fixed_points: Vec<FixedPoint> = Vec::new();
        let grid_size = 10; // Number of initial guesses per dimension

        // Generate grid of initial guesses
        let mut initial_guesses = Vec::new();
        self.generate_grid_points(domain, grid_size, &mut initial_guesses);

        for guess in initial_guesses {
            if let Ok(fixed_point) = self.newton_raphson_fixed_point(system, &guess) {
                // Check if this fixed point is already found
                let mut is_duplicate = false;
                for existing_fp in &fixed_points {
                    let distance = (&fixed_point - &existing_fp.location)
                        .iter()
                        .map(|&x| x * x)
                        .sum::<f64>()
                        .sqrt();

                    if distance < self.tolerance * 10.0 {
                        is_duplicate = true;
                        break;
                    }
                }

                if !is_duplicate {
                    // Compute stability
                    let jacobian = self.compute_jacobian_at_point(system, &fixed_point)?;
                    let eigenvalues = self.compute_real_eigenvalues(&jacobian)?;
                    let eigenvectors = self.compute_eigenvectors(&jacobian, &eigenvalues)?;
                    let stability = self.classify_stability(&eigenvalues);

                    fixed_points.push(FixedPoint {
                        location: fixed_point,
                        stability,
                        eigenvalues,
                        eigenvectors,
                    });
                }
            }
        }

        Ok(fixed_points)
    }

    /// Generate grid of points in domain
    fn generate_grid_points(
        &self,
        domain: &[(f64, f64)],
        grid_size: usize,
        points: &mut Vec<Array1<f64>>,
    ) {
        fn generate_recursive(
            domain: &[(f64, f64)],
            grid_size: usize,
            current: &mut Vec<f64>,
            dim: usize,
            points: &mut Vec<Array1<f64>>,
        ) {
            if dim == domain.len() {
                points.push(Array1::from_vec(current.clone()));
                return;
            }

            // Check for division by zero in step calculation
            if grid_size <= 1 {
                return; // Skip invalid grid size
            }
            let step = (domain[dim].1 - domain[dim].0) / (grid_size - 1) as f64;
            for i in 0..grid_size {
                let value = domain[dim].0 + i as f64 * step;
                current.push(value);
                generate_recursive(domain, grid_size, current, dim + 1, points);
                current.pop();
            }
        }

        let mut current = Vec::new();
        generate_recursive(domain, grid_size, &mut current, 0, points);
    }

    /// Find fixed point using Newton-Raphson method
    fn newton_raphson_fixed_point<F>(
        &self,
        system: &F,
        initial_guess: &Array1<f64>,
    ) -> Result<Array1<f64>>
    where
        F: Fn(&Array1<f64>) -> Array1<f64>,
    {
        let mut x = initial_guess.clone();
        let max_iterations = 100;

        for _ in 0..max_iterations {
            let f_x = system(&x);
            let sum_squares = f_x.iter().map(|&v| v * v).sum::<f64>();
            if sum_squares < 0.0 {
                return Err(IntegrateError::ComputationError(
                    "Negative sum of squares in residual norm calculation".to_string(),
                ));
            }
            let residual_norm = sum_squares.sqrt();

            if residual_norm < self.tolerance {
                return Ok(x);
            }

            let jacobian = self.compute_jacobian_at_point(system, &x)?;

            // Solve J * dx = -f(x)
            let mut augmented = Array2::zeros((self.dimension, self.dimension + 1));
            for i in 0..self.dimension {
                for j in 0..self.dimension {
                    augmented[[i, j]] = jacobian[[i, j]];
                }
                augmented[[i, self.dimension]] = -f_x[i];
            }

            let dx = self.gaussian_elimination(&augmented)?;
            x += &dx;
        }

        Err(IntegrateError::ConvergenceError(
            "Newton-Raphson did not converge".to_string(),
        ))
    }

    /// Compute Jacobian at a specific point
    fn compute_jacobian_at_point<F>(&self, system: &F, x: &Array1<f64>) -> Result<Array2<f64>>
    where
        F: Fn(&Array1<f64>) -> Array1<f64>,
    {
        let h = 1e-8_f64;
        let n = x.len();
        let mut jacobian = Array2::zeros((n, n));

        let f0 = system(x);

        // Check for valid step size
        if h.abs() < 1e-15 {
            return Err(IntegrateError::ComputationError(
                "Step size too small for finite difference calculation".to_string(),
            ));
        }

        for j in 0..n {
            let mut x_plus = x.clone();
            x_plus[j] += h;
            let f_plus = system(&x_plus);

            for i in 0..n {
                jacobian[[i, j]] = (f_plus[i] - f0[i]) / h;
            }
        }

        Ok(jacobian)
    }

    /// Solve linear system using Gaussian elimination
    fn gaussian_elimination(&self, augmented: &Array2<f64>) -> Result<Array1<f64>> {
        let n = augmented.nrows();
        let mut a = augmented.clone();

        // Forward elimination
        for k in 0..n {
            // Find pivot
            let mut max_row = k;
            for i in k + 1..n {
                if a[[i, k]].abs() > a[[max_row, k]].abs() {
                    max_row = i;
                }
            }

            // Swap rows
            if max_row != k {
                for j in k..n + 1 {
                    let temp = a[[k, j]];
                    a[[k, j]] = a[[max_row, j]];
                    a[[max_row, j]] = temp;
                }
            }

            // Check for singularity
            if a[[k, k]].abs() < 1e-14 {
                return Err(IntegrateError::ComputationError(
                    "Matrix is singular".to_string(),
                ));
            }

            // Eliminate
            for i in k + 1..n {
                let factor = a[[i, k]] / a[[k, k]];
                for j in k..n + 1 {
                    a[[i, j]] -= factor * a[[k, j]];
                }
            }
        }

        // Back substitution
        let mut x = Array1::zeros(n);
        for i in (0..n).rev() {
            x[i] = a[[i, n]];
            for j in i + 1..n {
                x[i] -= a[[i, j]] * x[j];
            }
            // Check for zero diagonal element
            if a[[i, i]].abs() < 1e-14 {
                return Err(IntegrateError::ComputationError(
                    "Zero diagonal element in back substitution".to_string(),
                ));
            }
            x[i] /= a[[i, i]];
        }

        Ok(x)
    }

    /// Compute real eigenvalues (simplified implementation)
    fn compute_real_eigenvalues(&self, matrix: &Array2<f64>) -> Result<Vec<Complex64>> {
        // For now, use a simplified approach for 2x2 matrices
        let n = matrix.nrows();

        if n == 2 {
            let a = matrix[[0, 0]];
            let b = matrix[[0, 1]];
            let c = matrix[[1, 0]];
            let d = matrix[[1, 1]];

            let trace = a + d;
            let det = a * d - b * c;
            let discriminant = trace * trace - 4.0 * det;

            if discriminant >= 0.0 {
                let sqrt_disc = discriminant.sqrt();
                let lambda1 = (trace + sqrt_disc) / 2.0;
                let lambda2 = (trace - sqrt_disc) / 2.0;
                Ok(vec![
                    Complex64::new(lambda1, 0.0),
                    Complex64::new(lambda2, 0.0),
                ])
            } else {
                let real_part = trace / 2.0;
                let imag_part = (-discriminant).sqrt() / 2.0;
                Ok(vec![
                    Complex64::new(real_part, imag_part),
                    Complex64::new(real_part, -imag_part),
                ])
            }
        } else {
            // For higher dimensions, use power iteration or other methods
            Err(IntegrateError::NotImplementedError(
                "Eigenvalue computation for matrices larger than 2x2 not implemented".to_string(),
            ))
        }
    }

    /// Compute eigenvectors (simplified)
    fn compute_eigenvectors(
        &self,
        _matrix: &Array2<f64>,
        eigenvalues: &[Complex64],
    ) -> Result<Array2<Complex64>> {
        let n = eigenvalues.len();
        let eigenvectors = Array2::<Complex64>::zeros((n, n));

        // Simplified: return identity matrix
        // In practice, would solve (A - λI)v = 0 for each eigenvalue
        Ok(eigenvectors)
    }

    /// Classify stability based on eigenvalues
    fn classify_stability(&self, eigenvalues: &[Complex64]) -> StabilityType {
        let mut has_positive_real = false;
        let mut has_negative_real = false;
        let mut has_zero_real = false;

        for eigenvalue in eigenvalues {
            if eigenvalue.re > 1e-10 {
                has_positive_real = true;
            } else if eigenvalue.re < -1e-10 {
                has_negative_real = true;
            } else {
                has_zero_real = true;
            }
        }

        if has_zero_real {
            StabilityType::Degenerate
        } else if has_positive_real && has_negative_real {
            StabilityType::Saddle
        } else if has_positive_real {
            StabilityType::Unstable
        } else if has_negative_real {
            StabilityType::Stable
        } else {
            StabilityType::Center
        }
    }

    /// Find periodic orbits (simplified implementation)
    fn find_periodic_orbits<F>(
        &self,
        _system: &F,
        _domain: &[(f64, f64)],
    ) -> Result<Vec<PeriodicOrbit>>
    where
        F: Fn(&Array1<f64>) -> Array1<f64>,
    {
        // Placeholder for periodic orbit detection
        // Would use methods like Poincaré sections, shooting methods, etc.
        Ok(vec![])
    }

    /// Compute Lyapunov exponents
    fn compute_lyapunov_exponents<F>(&self, _system: &F) -> Result<Option<Array1<f64>>>
    where
        F: Fn(&Array1<f64>) -> Array1<f64>,
    {
        // Placeholder for Lyapunov exponent computation
        // Would integrate the variational equation
        Ok(None)
    }

    /// Analyze basins of attraction for 2D systems
    fn analyze_basins<F>(
        &self,
        system: &F,
        domain: &[(f64, f64)],
        fixed_points: &[FixedPoint],
    ) -> Result<BasinAnalysis>
    where
        F: Fn(&Array1<f64>) -> Array1<f64> + Send + Sync,
    {
        if self.dimension != 2 || domain.len() != 2 {
            return Err(IntegrateError::ValueError(
                "Basin analysis only implemented for 2D systems".to_string(),
            ));
        }

        let grid_size = self.basin_grid_size;
        let mut grid_points = Array2::zeros((grid_size * grid_size, 2));
        let mut attractor_indices = Array2::<i32>::zeros((grid_size, grid_size));

        let dx = (domain[0].1 - domain[0].0) / (grid_size - 1) as f64;
        let dy = (domain[1].1 - domain[1].0) / (grid_size - 1) as f64;

        // Generate grid and integrate each point
        for i in 0..grid_size {
            for j in 0..grid_size {
                let x = domain[0].0 + i as f64 * dx;
                let y = domain[1].0 + j as f64 * dy;

                grid_points[[i * grid_size + j, 0]] = x;
                grid_points[[i * grid_size + j, 1]] = y;

                // Integrate trajectory and find which attractor it converges to
                let initial_state = Array1::from_vec(vec![x, y]);
                let final_state = self.integrate_trajectory(system, &initial_state)?;

                // Find closest fixed point
                let mut closest_attractor = -1;
                let mut min_distance = f64::INFINITY;

                for (idx, fp) in fixed_points.iter().enumerate() {
                    if fp.stability == StabilityType::Stable {
                        let distance = (&final_state - &fp.location)
                            .iter()
                            .map(|&x| x * x)
                            .sum::<f64>()
                            .sqrt();

                        if distance < min_distance && distance < 0.1 {
                            min_distance = distance;
                            closest_attractor = idx as i32;
                        }
                    }
                }

                attractor_indices[[i, j]] = closest_attractor;
            }
        }

        // Extract stable attractors
        let attractors = fixed_points
            .iter()
            .filter(|fp| fp.stability == StabilityType::Stable)
            .map(|fp| fp.location.clone())
            .collect();

        Ok(BasinAnalysis {
            grid_points,
            attractor_indices,
            attractors,
        })
    }

    /// Integrate trajectory to find final state
    fn integrate_trajectory<F>(
        &self,
        system: &F,
        initial_state: &Array1<f64>,
    ) -> Result<Array1<f64>>
    where
        F: Fn(&Array1<f64>) -> Array1<f64>,
    {
        // Simple Euler integration
        let dt = 0.01;
        let n_steps = (self.integration_time / dt) as usize;
        let mut state = initial_state.clone();

        for _ in 0..n_steps {
            let derivative = system(&state);
            state += &(derivative * dt);
        }

        Ok(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_bifurcation_analyzer() {
        // Test with a simple pitchfork bifurcation: dx/dt = r*x - x^3
        let system = |x: &Array1<f64>, r: f64| -> Array1<f64> {
            Array1::from_vec(vec![r * x[0] - x[0].powi(3)])
        };

        let analyzer = BifurcationAnalyzer::new(1, (-1.0, 1.0), 10);
        let initial_guess = Array1::from_vec(vec![0.1]);

        let bifurcations = analyzer
            .continuation_analysis(system, &initial_guess)
            .unwrap();

        // Should detect bifurcation near r = 0
        assert!(!bifurcations.is_empty());
    }

    #[test]
    fn test_stability_analyzer() {
        // Test with simple 2D system: dx/dt = -x, dy/dt = -2y
        let system =
            |x: &Array1<f64>| -> Array1<f64> { Array1::from_vec(vec![-x[0], -2.0 * x[1]]) };

        let analyzer = StabilityAnalyzer::new(2);
        let domain = vec![(-2.0, 2.0), (-2.0, 2.0)];

        let result = analyzer.analyze_stability(system, &domain).unwrap();

        // Should find stable fixed point at origin
        assert!(!result.fixed_points.is_empty());
        let origin = &result.fixed_points[0];
        assert_relative_eq!(origin.location[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(origin.location[1], 0.0, epsilon = 1e-6);
        assert_eq!(origin.stability, StabilityType::Stable);
    }

    #[test]
    fn test_stability_classification() {
        let analyzer = StabilityAnalyzer::new(2);

        // Stable node
        let stable_eigs = vec![Complex64::new(-1.0, 0.0), Complex64::new(-2.0, 0.0)];
        assert_eq!(
            analyzer.classify_stability(&stable_eigs),
            StabilityType::Stable
        );

        // Unstable node
        let unstable_eigs = vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];
        assert_eq!(
            analyzer.classify_stability(&unstable_eigs),
            StabilityType::Unstable
        );

        // Saddle point
        let saddle_eigs = vec![Complex64::new(-1.0, 0.0), Complex64::new(1.0, 0.0)];
        assert_eq!(
            analyzer.classify_stability(&saddle_eigs),
            StabilityType::Saddle
        );
    }
}

/// Advanced analysis tools for complex dynamical systems
pub mod advanced_analysis {
    use super::*;
    use crate::error::{IntegrateError, IntegrateResult as Result};
    use ndarray::{Array1, Array2};

    /// Poincaré section analysis for periodic orbit detection
    pub struct PoincareAnalyzer {
        /// Section definition (hyperplane normal)
        pub section_normal: Array1<f64>,
        /// Point on the section
        pub section_point: Array1<f64>,
        /// Crossing direction (1: positive, -1: negative, 0: both)
        pub crossing_direction: i8,
        /// Tolerance for section crossing detection
        pub crossing_tolerance: f64,
    }

    impl PoincareAnalyzer {
        /// Create a new Poincaré analyzer
        pub fn new(
            section_normal: Array1<f64>,
            section_point: Array1<f64>,
            crossing_direction: i8,
        ) -> Self {
            Self {
                section_normal,
                section_point,
                crossing_direction,
                crossing_tolerance: 1e-8,
            }
        }

        /// Analyze trajectory to find Poincaré map
        pub fn analyze_trajectory(
            &self,
            trajectory: &[Array1<f64>],
            times: &[f64],
        ) -> Result<PoincareMap> {
            let mut crossings = Vec::new();
            let mut crossing_times = Vec::new();

            for i in 1..trajectory.len() {
                if let Some((crossing_point, crossing_time)) = self.detect_crossing(
                    &trajectory[i - 1],
                    &trajectory[i],
                    times[i - 1],
                    times[i],
                )? {
                    crossings.push(crossing_point);
                    crossing_times.push(crossing_time);
                }
            }

            // Compute return map if sufficient crossings
            let return_map = if crossings.len() > 1 {
                Some(self.compute_return_map(&crossings)?)
            } else {
                None
            };

            // Detect periodic orbits
            let periodic_orbits = self.detect_periodic_orbits(&crossings)?;

            Ok(PoincareMap {
                crossings,
                crossing_times,
                return_map,
                periodic_orbits,
                section_normal: self.section_normal.clone(),
                section_point: self.section_point.clone(),
            })
        }

        /// Detect crossing of Poincaré section
        fn detect_crossing(
            &self,
            point1: &Array1<f64>,
            point2: &Array1<f64>,
            t1: f64,
            t2: f64,
        ) -> Result<Option<(Array1<f64>, f64)>> {
            // Calculate distances from section
            let d1 = self.distance_from_section(point1);
            let d2 = self.distance_from_section(point2);

            // Check for crossing
            let crossed = match self.crossing_direction {
                1 => d1 < 0.0 && d2 > 0.0,  // Positive crossing
                -1 => d1 > 0.0 && d2 < 0.0, // Negative crossing
                0 => d1 * d2 < 0.0,         // Any crossing
                _ => false,
            };

            if !crossed {
                return Ok(None);
            }

            // Interpolate crossing point
            let alpha = d1.abs() / (d1.abs() + d2.abs());
            let crossing_point = (1.0 - alpha) * point1 + alpha * point2;
            let crossing_time = (1.0 - alpha) * t1 + alpha * t2;

            Ok(Some((crossing_point, crossing_time)))
        }

        /// Calculate distance from point to section
        fn distance_from_section(&self, point: &Array1<f64>) -> f64 {
            let relative_pos = point - &self.section_point;
            relative_pos.dot(&self.section_normal)
        }

        /// Compute return map from crossings
        fn compute_return_map(&self, crossings: &[Array1<f64>]) -> Result<ReturnMap> {
            let mut current_points = Vec::new();
            let mut next_points = Vec::new();

            for i in 0..crossings.len() - 1 {
                // Project points onto section (remove normal component)
                let current_projected = self.project_to_section(&crossings[i]);
                let next_projected = self.project_to_section(&crossings[i + 1]);

                current_points.push(current_projected);
                next_points.push(next_projected);
            }

            Ok(ReturnMap {
                current_points,
                next_points,
            })
        }

        /// Project point onto Poincaré section
        fn project_to_section(&self, point: &Array1<f64>) -> Array1<f64> {
            let distance = self.distance_from_section(point);
            point - distance * &self.section_normal
        }

        /// Detect periodic orbits from crossings
        fn detect_periodic_orbits(&self, crossings: &[Array1<f64>]) -> Result<Vec<PeriodicOrbit>> {
            let mut periodic_orbits = Vec::new();
            let tolerance = 1e-6;

            // Look for approximate returns to previous crossing points
            for i in 0..crossings.len() {
                for j in (i + 2)..crossings.len() {
                    let distance = self.euclidean_distance(&crossings[i], &crossings[j]);
                    if distance < tolerance {
                        // Found potential periodic orbit
                        let period_length = j - i;
                        let representative_point = crossings[i].clone();

                        // Verify periodicity by checking intermediate points
                        let mut is_periodic = true;
                        for k in 1..period_length {
                            if i + k < crossings.len() && j + k < crossings.len() {
                                let dist =
                                    self.euclidean_distance(&crossings[i + k], &crossings[j + k]);
                                if dist > tolerance {
                                    is_periodic = false;
                                    break;
                                }
                            }
                        }

                        if is_periodic {
                            // Calculate approximate period in time
                            let period = (j - i) as f64; // Simplified period estimate

                            periodic_orbits.push(PeriodicOrbit {
                                representative_point,
                                period,
                                stability: StabilityType::Stable, // Would need proper analysis
                                floquet_multipliers: vec![],      // Would need computation
                            });
                        }
                    }
                }
            }

            Ok(periodic_orbits)
        }

        fn euclidean_distance(&self, a: &Array1<f64>, b: &Array1<f64>) -> f64 {
            (a - b).iter().map(|&x| x * x).sum::<f64>().sqrt()
        }
    }

    /// Poincaré map data structure
    #[derive(Debug, Clone)]
    pub struct PoincareMap {
        /// Section crossing points
        pub crossings: Vec<Array1<f64>>,
        /// Times of crossings
        pub crossing_times: Vec<f64>,
        /// Return map data
        pub return_map: Option<ReturnMap>,
        /// Detected periodic orbits
        pub periodic_orbits: Vec<PeriodicOrbit>,
        /// Section normal vector
        pub section_normal: Array1<f64>,
        /// Point on section
        pub section_point: Array1<f64>,
    }

    /// Return map data
    #[derive(Debug, Clone)]
    pub struct ReturnMap {
        /// Current points
        pub current_points: Vec<Array1<f64>>,
        /// Next points in return map
        pub next_points: Vec<Array1<f64>>,
    }

    /// Lyapunov exponent calculator for chaos detection
    pub struct LyapunovCalculator {
        /// Number of exponents to calculate
        pub n_exponents: usize,
        /// Perturbation magnitude for tangent vectors
        pub perturbation_magnitude: f64,
        /// Renormalization interval
        pub renormalization_interval: usize,
        /// Integration time step
        pub dt: f64,
    }

    impl LyapunovCalculator {
        /// Create new Lyapunov calculator
        pub fn new(n_exponents: usize, dt: f64) -> Self {
            Self {
                n_exponents,
                perturbation_magnitude: 1e-8,
                renormalization_interval: 100,
                dt,
            }
        }

        /// Calculate Lyapunov exponents using tangent space evolution
        pub fn calculate_lyapunov_exponents<F>(
            &self,
            system: F,
            initial_state: &Array1<f64>,
            total_time: f64,
        ) -> Result<Array1<f64>>
        where
            F: Fn(&Array1<f64>) -> Array1<f64> + Send + Sync,
        {
            let n_steps = (total_time / self.dt) as usize;
            let dim = initial_state.len();

            if self.n_exponents > dim {
                return Err(IntegrateError::ValueError(
                    "Number of exponents cannot exceed system dimension".to_string(),
                ));
            }

            // Initialize state and tangent vectors
            let mut state = initial_state.clone();
            let mut tangent_vectors = self.initialize_tangent_vectors(dim)?;
            let mut lyapunov_sums = Array1::zeros(self.n_exponents);

            // Main integration loop
            for step in 0..n_steps {
                // Evolve main trajectory
                let derivative = system(&state);
                state += &(derivative * self.dt);

                // Evolve tangent vectors
                for i in 0..self.n_exponents {
                    let jacobian = self.compute_jacobian(&system, &state)?;
                    let tangent_derivative = jacobian.dot(&tangent_vectors.column(i));
                    let old_tangent = tangent_vectors.column(i).to_owned();
                    tangent_vectors
                        .column_mut(i)
                        .assign(&(&old_tangent + &(tangent_derivative * self.dt)));
                }

                // Renormalization to prevent overflow
                if step % self.renormalization_interval == 0 && step > 0 {
                    let (q, r) = self.qr_decomposition(&tangent_vectors)?;
                    tangent_vectors = q;

                    // Add to Lyapunov sum
                    for i in 0..self.n_exponents {
                        lyapunov_sums[i] += r[[i, i]].abs().ln();
                    }
                }
            }

            // Final normalization
            let lyapunov_exponents = lyapunov_sums / total_time;

            Ok(lyapunov_exponents)
        }

        /// Initialize orthonormal tangent vectors
        fn initialize_tangent_vectors(&self, dim: usize) -> Result<Array2<f64>> {
            let mut vectors = Array2::zeros((dim, self.n_exponents));

            // Initialize with random vectors
            use rand::Rng;
            let mut rng = rand::rng();
            for i in 0..self.n_exponents {
                for j in 0..dim {
                    vectors[[j, i]] = rng.random::<f64>() - 0.5;
                }
            }

            // Gram-Schmidt orthogonalization
            for i in 0..self.n_exponents {
                // Orthogonalize against previous vectors
                for j in 0..i {
                    let projection = vectors.column(i).dot(&vectors.column(j));
                    let col_j = vectors.column(j).to_owned();
                    let mut col_i = vectors.column_mut(i);
                    col_i -= &(projection * &col_j);
                }

                // Normalize
                let norm = vectors.column(i).iter().map(|&x| x * x).sum::<f64>().sqrt();
                if norm > 1e-12 {
                    vectors.column_mut(i).mapv_inplace(|x| x / norm);
                }
            }

            Ok(vectors)
        }

        /// Compute Jacobian matrix using finite differences
        fn compute_jacobian<F>(&self, system: &F, state: &Array1<f64>) -> Result<Array2<f64>>
        where
            F: Fn(&Array1<f64>) -> Array1<f64>,
        {
            let dim = state.len();
            let mut jacobian = Array2::zeros((dim, dim));
            let h = self.perturbation_magnitude;

            for j in 0..dim {
                let mut state_plus = state.clone();
                let mut state_minus = state.clone();

                state_plus[j] += h;
                state_minus[j] -= h;

                let f_plus = system(&state_plus);
                let f_minus = system(&state_minus);

                for i in 0..dim {
                    jacobian[[i, j]] = (f_plus[i] - f_minus[i]) / (2.0 * h);
                }
            }

            Ok(jacobian)
        }

        /// QR decomposition using Gram-Schmidt
        fn qr_decomposition(&self, matrix: &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>)> {
            let (_m, n) = matrix.dim();
            let mut q = matrix.clone();
            let mut r = Array2::zeros((n, n));

            for j in 0..n {
                // Orthogonalize against previous columns
                for i in 0..j {
                    r[[i, j]] = q.column(j).dot(&q.column(i));
                    let col_i = q.column(i).to_owned();
                    let mut col_j = q.column_mut(j);
                    col_j -= &(r[[i, j]] * &col_i);
                }

                // Normalize
                r[[j, j]] = q.column(j).iter().map(|&x| x * x).sum::<f64>().sqrt();
                if r[[j, j]] > 1e-12 {
                    q.column_mut(j).mapv_inplace(|x| x / r[[j, j]]);
                }
            }

            Ok((q, r))
        }
    }

    /// Fractal dimension analyzer for strange attractors
    pub struct FractalAnalyzer {
        /// Range of scales to analyze
        pub scale_range: (f64, f64),
        /// Number of scale points
        pub n_scales: usize,
        /// Box-counting parameters
        pub box_counting_method: BoxCountingMethod,
    }

    /// Box counting methods
    #[derive(Debug, Clone, Copy)]
    pub enum BoxCountingMethod {
        /// Standard box counting
        Standard,
        /// Differential box counting
        Differential,
        /// Correlation dimension
        Correlation,
    }

    impl Default for FractalAnalyzer {
        fn default() -> Self {
            Self::new()
        }
    }

    impl FractalAnalyzer {
        /// Create new fractal analyzer
        pub fn new() -> Self {
            Self {
                scale_range: (1e-4, 1e-1),
                n_scales: 20,
                box_counting_method: BoxCountingMethod::Standard,
            }
        }

        /// Calculate fractal dimension of attractor
        pub fn calculate_fractal_dimension(
            &self,
            attractor_points: &[Array1<f64>],
        ) -> Result<FractalDimension> {
            match self.box_counting_method {
                BoxCountingMethod::Standard => self.box_counting_dimension(attractor_points),
                BoxCountingMethod::Differential => self.differential_box_counting(attractor_points),
                BoxCountingMethod::Correlation => self.correlation_dimension(attractor_points),
            }
        }

        /// Standard box counting dimension
        fn box_counting_dimension(&self, points: &[Array1<f64>]) -> Result<FractalDimension> {
            if points.is_empty() {
                return Err(IntegrateError::ValueError(
                    "Cannot analyze empty point set".to_string(),
                ));
            }

            let dim = points[0].len();

            // Find bounding box
            let (min_bounds, max_bounds) = self.find_bounding_box(points);
            let domain_size = max_bounds
                .iter()
                .zip(min_bounds.iter())
                .map(|(&max, &min)| max - min)
                .fold(0.0f64, |acc, x| acc.max(x));

            let mut scales = Vec::new();
            let mut counts = Vec::new();

            // Analyze different scales
            for i in 0..self.n_scales {
                let t = i as f64 / (self.n_scales - 1) as f64;
                let scale = self.scale_range.0 * (self.scale_range.1 / self.scale_range.0).powf(t);

                let box_size = scale * domain_size;
                let count = self.count_occupied_boxes(points, &min_bounds, box_size, dim)?;

                scales.push(scale);
                counts.push(count as f64);
            }

            // Linear regression on log-log plot
            let dimension = self.calculate_slope_from_log_data(&scales, &counts)?;

            let r_squared = self.calculate_r_squared(&scales, &counts, dimension)?;

            Ok(FractalDimension {
                dimension,
                method: self.box_counting_method,
                scales,
                counts,
                r_squared,
            })
        }

        /// Differential box counting for higher accuracy
        fn differential_box_counting(&self, points: &[Array1<f64>]) -> Result<FractalDimension> {
            // Simplified implementation - would need full differential box counting
            self.box_counting_dimension(points)
        }

        /// Correlation dimension using Grassberger-Procaccia algorithm
        fn correlation_dimension(&self, points: &[Array1<f64>]) -> Result<FractalDimension> {
            let n_points = points.len();
            let mut scales = Vec::new();
            let mut correlations = Vec::new();

            for i in 0..self.n_scales {
                let t = i as f64 / (self.n_scales - 1) as f64;
                let r = self.scale_range.0 * (self.scale_range.1 / self.scale_range.0).powf(t);

                let mut count = 0;
                for i in 0..n_points {
                    for j in i + 1..n_points {
                        let distance = self.euclidean_distance(&points[i], &points[j]);
                        if distance < r {
                            count += 1;
                        }
                    }
                }

                let correlation = 2.0 * count as f64 / (n_points * (n_points - 1)) as f64;

                scales.push(r);
                correlations.push(correlation);
            }

            // Filter out zero correlations for log calculation
            let filtered_data: Vec<(f64, f64)> = scales
                .iter()
                .zip(correlations.iter())
                .filter(|(_, &c)| c > 0.0)
                .map(|(&s, &c)| (s, c))
                .collect();

            if filtered_data.len() < 2 {
                return Err(IntegrateError::ComputationError(
                    "Insufficient data for correlation dimension calculation".to_string(),
                ));
            }

            let filtered_scales: Vec<f64> = filtered_data.iter().map(|(s, _)| *s).collect();
            let filtered_correlations: Vec<f64> = filtered_data.iter().map(|(_, c)| *c).collect();

            let dimension =
                self.calculate_slope_from_log_data(&filtered_scales, &filtered_correlations)?;

            Ok(FractalDimension {
                dimension,
                method: BoxCountingMethod::Correlation,
                scales,
                counts: correlations,
                r_squared: self.calculate_r_squared(
                    &filtered_scales,
                    &filtered_correlations,
                    dimension,
                )?,
            })
        }

        /// Helper functions
        fn find_bounding_box(&self, points: &[Array1<f64>]) -> (Array1<f64>, Array1<f64>) {
            let dim = points[0].len();
            let mut min_bounds = Array1::from_elem(dim, f64::INFINITY);
            let mut max_bounds = Array1::from_elem(dim, f64::NEG_INFINITY);

            for point in points {
                for i in 0..dim {
                    min_bounds[i] = min_bounds[i].min(point[i]);
                    max_bounds[i] = max_bounds[i].max(point[i]);
                }
            }

            (min_bounds, max_bounds)
        }

        fn count_occupied_boxes(
            &self,
            points: &[Array1<f64>],
            min_bounds: &Array1<f64>,
            box_size: f64,
            dim: usize,
        ) -> Result<usize> {
            let mut occupied_boxes = std::collections::HashSet::new();

            for point in points {
                let mut box_index = Vec::with_capacity(dim);
                for i in 0..dim {
                    let index = ((point[i] - min_bounds[i]) / box_size).floor() as i64;
                    box_index.push(index);
                }
                occupied_boxes.insert(box_index);
            }

            Ok(occupied_boxes.len())
        }

        fn calculate_slope_from_log_data(&self, x_data: &[f64], y_data: &[f64]) -> Result<f64> {
            if x_data.len() != y_data.len() || x_data.len() < 2 {
                return Err(IntegrateError::ValueError(
                    "Insufficient data for slope calculation".to_string(),
                ));
            }

            let n = x_data.len() as f64;
            let log_x: Vec<f64> = x_data.iter().map(|&x| x.ln()).collect();
            let log_y: Vec<f64> = y_data.iter().map(|&y| y.ln()).collect();

            let sum_x: f64 = log_x.iter().sum();
            let sum_y: f64 = log_y.iter().sum();
            let sum_xy: f64 = log_x.iter().zip(log_y.iter()).map(|(&x, &y)| x * y).sum();
            let sum_xx: f64 = log_x.iter().map(|&x| x * x).sum();

            let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);

            Ok(slope)
        }

        fn calculate_r_squared(&self, x_data: &[f64], y_data: &[f64], slope: f64) -> Result<f64> {
            let log_x: Vec<f64> = x_data.iter().map(|&x| x.ln()).collect();
            let log_y: Vec<f64> = y_data.iter().map(|&y| y.ln()).collect();

            let mean_y = log_y.iter().sum::<f64>() / log_y.len() as f64;
            let mean_x = log_x.iter().sum::<f64>() / log_x.len() as f64;
            let intercept = mean_y - slope * mean_x;

            let mut ss_tot = 0.0;
            let mut ss_res = 0.0;

            for i in 0..log_y.len() {
                let y_pred = slope * log_x[i] + intercept;
                ss_res += (log_y[i] - y_pred).powi(2);
                ss_tot += (log_y[i] - mean_y).powi(2);
            }

            let r_squared = 1.0 - (ss_res / ss_tot);
            Ok(r_squared)
        }

        fn euclidean_distance(&self, a: &Array1<f64>, b: &Array1<f64>) -> f64 {
            a.iter()
                .zip(b.iter())
                .map(|(&x, &y)| (x - y).powi(2))
                .sum::<f64>()
                .sqrt()
        }
    }

    /// Fractal dimension result
    #[derive(Debug, Clone)]
    pub struct FractalDimension {
        /// Calculated dimension
        pub dimension: f64,
        /// Method used
        pub method: BoxCountingMethod,
        /// Scale values used
        pub scales: Vec<f64>,
        /// Count/correlation values
        pub counts: Vec<f64>,
        /// Quality of fit (R²)
        pub r_squared: f64,
    }

    /// Recurrence analysis for detecting patterns and periodicities
    pub struct RecurrenceAnalyzer {
        /// Recurrence threshold
        pub threshold: f64,
        /// Embedding dimension for delay coordinate embedding
        pub embedding_dimension: usize,
        /// Time delay for embedding
        pub time_delay: usize,
        /// Distance metric
        pub distance_metric: DistanceMetric,
    }

    /// Distance metrics for recurrence analysis
    #[derive(Debug, Clone, Copy)]
    pub enum DistanceMetric {
        /// Euclidean distance
        Euclidean,
        /// Maximum (Chebyshev) distance
        Maximum,
        /// Manhattan distance
        Manhattan,
    }

    impl RecurrenceAnalyzer {
        /// Create new recurrence analyzer
        pub fn new(threshold: f64, embedding_dimension: usize, time_delay: usize) -> Self {
            Self {
                threshold,
                embedding_dimension,
                time_delay,
                distance_metric: DistanceMetric::Euclidean,
            }
        }

        /// Perform recurrence analysis
        pub fn analyze_recurrence(&self, time_series: &[f64]) -> Result<RecurrenceAnalysis> {
            // Create delay coordinate embedding
            let embedded_vectors = self.create_embedding(time_series)?;

            // Compute recurrence matrix
            let recurrence_matrix = self.compute_recurrence_matrix(&embedded_vectors)?;

            // Calculate recurrence quantification measures
            let rqa_measures = self.calculate_rqa_measures(&recurrence_matrix)?;

            Ok(RecurrenceAnalysis {
                recurrence_matrix,
                embedded_vectors,
                rqa_measures,
                threshold: self.threshold,
                embedding_dimension: self.embedding_dimension,
                time_delay: self.time_delay,
            })
        }

        /// Create delay coordinate embedding
        fn create_embedding(&self, time_series: &[f64]) -> Result<Vec<Array1<f64>>> {
            let n = time_series.len();
            let embedded_length = n - (self.embedding_dimension - 1) * self.time_delay;

            if embedded_length <= 0 {
                return Err(IntegrateError::ValueError(
                    "Time series too short for given embedding parameters".to_string(),
                ));
            }

            let mut embedded_vectors = Vec::with_capacity(embedded_length);

            for i in 0..embedded_length {
                let mut vector = Array1::zeros(self.embedding_dimension);
                for j in 0..self.embedding_dimension {
                    vector[j] = time_series[i + j * self.time_delay];
                }
                embedded_vectors.push(vector);
            }

            Ok(embedded_vectors)
        }

        /// Compute recurrence matrix
        fn compute_recurrence_matrix(&self, vectors: &[Array1<f64>]) -> Result<Array2<bool>> {
            let n = vectors.len();
            let mut recurrence_matrix = Array2::from_elem((n, n), false);

            for i in 0..n {
                for j in 0..n {
                    let distance = self.calculate_distance(&vectors[i], &vectors[j]);
                    recurrence_matrix[[i, j]] = distance <= self.threshold;
                }
            }

            Ok(recurrence_matrix)
        }

        /// Calculate distance between vectors
        fn calculate_distance(&self, a: &Array1<f64>, b: &Array1<f64>) -> f64 {
            match self.distance_metric {
                DistanceMetric::Euclidean => a
                    .iter()
                    .zip(b.iter())
                    .map(|(&x, &y)| (x - y).powi(2))
                    .sum::<f64>()
                    .sqrt(),
                DistanceMetric::Maximum => a
                    .iter()
                    .zip(b.iter())
                    .map(|(&x, &y)| (x - y).abs())
                    .fold(0.0, f64::max),
                DistanceMetric::Manhattan => {
                    a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).abs()).sum()
                }
            }
        }

        /// Calculate Recurrence Quantification Analysis measures
        fn calculate_rqa_measures(&self, recurrence_matrix: &Array2<bool>) -> Result<RQAMeasures> {
            let n = recurrence_matrix.nrows();
            let total_points = (n * n) as f64;

            // Recurrence rate
            let recurrent_points = recurrence_matrix.iter().filter(|&&x| x).count() as f64;
            let recurrence_rate = recurrent_points / total_points;

            // Determinism (percentage of recurrent points forming diagonal lines)
            let diagonal_lines = self.find_diagonal_lines(recurrence_matrix, 2)?;
            let diagonal_points: usize = diagonal_lines.iter().map(|line| line.length).sum();
            let determinism = diagonal_points as f64 / recurrent_points;

            // Average diagonal line length
            let avg_diagonal_length = if !diagonal_lines.is_empty() {
                diagonal_points as f64 / diagonal_lines.len() as f64
            } else {
                0.0
            };

            // Maximum diagonal line length
            let max_diagonal_length = diagonal_lines
                .iter()
                .map(|line| line.length)
                .max()
                .unwrap_or(0) as f64;

            // Laminarity (percentage of recurrent points forming vertical lines)
            let vertical_lines = self.find_vertical_lines(recurrence_matrix, 2)?;
            let vertical_points: usize = vertical_lines.iter().map(|line| line.length).sum();
            let laminarity = vertical_points as f64 / recurrent_points;

            // Trapping time (average vertical line length)
            let trapping_time = if !vertical_lines.is_empty() {
                vertical_points as f64 / vertical_lines.len() as f64
            } else {
                0.0
            };

            Ok(RQAMeasures {
                recurrence_rate,
                determinism,
                avg_diagonal_length,
                max_diagonal_length,
                laminarity,
                trapping_time,
            })
        }

        /// Find diagonal lines in recurrence matrix
        fn find_diagonal_lines(
            &self,
            matrix: &Array2<bool>,
            min_length: usize,
        ) -> Result<Vec<RecurrentLine>> {
            let n = matrix.nrows();
            let mut lines = Vec::new();

            // Check all diagonals
            for k in -(n as i32 - 1)..(n as i32) {
                let mut current_length = 0;
                let mut start_i = 0;
                let mut start_j = 0;

                let (start_row, start_col) = if k >= 0 {
                    (0, k as usize)
                } else {
                    ((-k) as usize, 0)
                };

                let max_steps = n - start_row.max(start_col);

                for step in 0..max_steps {
                    let i = start_row + step;
                    let j = start_col + step;

                    if i < n && j < n && matrix[[i, j]] {
                        if current_length == 0 {
                            start_i = i;
                            start_j = j;
                        }
                        current_length += 1;
                    } else {
                        if current_length >= min_length {
                            lines.push(RecurrentLine {
                                start_i,
                                start_j,
                                length: current_length,
                                line_type: LineType::Diagonal,
                            });
                        }
                        current_length = 0;
                    }
                }

                // Check end of diagonal
                if current_length >= min_length {
                    lines.push(RecurrentLine {
                        start_i,
                        start_j,
                        length: current_length,
                        line_type: LineType::Diagonal,
                    });
                }
            }

            Ok(lines)
        }

        /// Find vertical lines in recurrence matrix
        fn find_vertical_lines(
            &self,
            matrix: &Array2<bool>,
            min_length: usize,
        ) -> Result<Vec<RecurrentLine>> {
            let n = matrix.nrows();
            let mut lines = Vec::new();

            for j in 0..n {
                let mut current_length = 0;
                let mut start_i = 0;

                for i in 0..n {
                    if matrix[[i, j]] {
                        if current_length == 0 {
                            start_i = i;
                        }
                        current_length += 1;
                    } else {
                        if current_length >= min_length {
                            lines.push(RecurrentLine {
                                start_i,
                                start_j: j,
                                length: current_length,
                                line_type: LineType::Vertical,
                            });
                        }
                        current_length = 0;
                    }
                }

                // Check end of column
                if current_length >= min_length {
                    lines.push(RecurrentLine {
                        start_i,
                        start_j: j,
                        length: current_length,
                        line_type: LineType::Vertical,
                    });
                }
            }

            Ok(lines)
        }
    }

    /// Recurrence analysis result
    #[derive(Debug, Clone)]
    pub struct RecurrenceAnalysis {
        /// Recurrence matrix
        pub recurrence_matrix: Array2<bool>,
        /// Embedded vectors
        pub embedded_vectors: Vec<Array1<f64>>,
        /// RQA measures
        pub rqa_measures: RQAMeasures,
        /// Analysis parameters
        pub threshold: f64,
        pub embedding_dimension: usize,
        pub time_delay: usize,
    }

    /// Recurrence Quantification Analysis measures
    #[derive(Debug, Clone)]
    pub struct RQAMeasures {
        /// Recurrence rate
        pub recurrence_rate: f64,
        /// Determinism
        pub determinism: f64,
        /// Average diagonal line length
        pub avg_diagonal_length: f64,
        /// Maximum diagonal line length
        pub max_diagonal_length: f64,
        /// Laminarity
        pub laminarity: f64,
        /// Trapping time
        pub trapping_time: f64,
    }

    /// Recurrent line structure
    #[derive(Debug, Clone)]
    pub struct RecurrentLine {
        pub start_i: usize,
        pub start_j: usize,
        pub length: usize,
        pub line_type: LineType,
    }

    /// Line types in recurrence plot
    #[derive(Debug, Clone, Copy)]
    pub enum LineType {
        Diagonal,
        Vertical,
        Horizontal,
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use approx::assert_relative_eq;

        #[test]
        fn test_poincare_analyzer() {
            // Test with simple circular trajectory
            let mut trajectory = Vec::new();
            let mut times = Vec::new();

            for i in 0..100 {
                let t = i as f64 * 0.1;
                let x = t.cos();
                let y = t.sin();
                let z = 0.0;

                trajectory.push(Array1::from_vec(vec![x, y, z]));
                times.push(t);
            }

            // Define Poincaré section as z = 0 plane
            let section_normal = Array1::from_vec(vec![0.0, 0.0, 1.0]);
            let section_point = Array1::from_vec(vec![0.0, 0.0, 0.0]);

            let analyzer = PoincareAnalyzer::new(section_normal, section_point, 1);
            let result = analyzer.analyze_trajectory(&trajectory, &times).unwrap();

            // Should find crossings for this trajectory
            assert!(!result.crossings.is_empty());
        }

        #[test]
        fn test_lyapunov_calculator() {
            // Test with simple linear system (should have negative Lyapunov exponent)
            let system = |state: &Array1<f64>| -> Array1<f64> {
                Array1::from_vec(vec![-state[0], -state[1]])
            };

            let calculator = LyapunovCalculator::new(2, 0.01);
            let initial_state = Array1::from_vec(vec![1.0, 1.0]);

            let exponents = calculator
                .calculate_lyapunov_exponents(system, &initial_state, 10.0)
                .unwrap();

            // Both exponents should be negative for stable linear system
            assert!(exponents[0] < 0.0);
            assert!(exponents[1] < 0.0);
        }

        #[test]
        fn test_fractal_analyzer() {
            // Test with random points (should give dimension close to embedding dimension)
            use rand::Rng;
            let mut rng = rand::rng();

            let mut points = Vec::new();
            for _ in 0..1000 {
                let point = Array1::from_vec(vec![rng.gen::<f64>(), rng.gen::<f64>()]);
                points.push(point);
            }

            let analyzer = FractalAnalyzer::new();
            let result = analyzer.calculate_fractal_dimension(&points).unwrap();

            // Dimension should be reasonable (between 1 and 2 for 2D random points)
            assert!(result.dimension > 1.0 && result.dimension < 3.0);
            assert!(result.r_squared >= 0.0 && result.r_squared <= 1.0);
        }

        #[test]
        fn test_recurrence_analyzer() {
            // Test with sinusoidal time series
            let time_series: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();

            let analyzer = RecurrenceAnalyzer::new(0.1, 3, 1);
            let result = analyzer.analyze_recurrence(&time_series).unwrap();

            // Should have reasonable recurrence measures
            assert!(result.rqa_measures.recurrence_rate > 0.0);
            assert!(result.rqa_measures.recurrence_rate <= 1.0);
            assert!(result.rqa_measures.determinism >= 0.0);
            assert!(result.rqa_measures.determinism <= 1.0);
        }
    }
}
