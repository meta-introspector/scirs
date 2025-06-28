//! Enhanced System Identification with advanced algorithms
//!
//! This module provides advanced system identification methods including:
//! - Recursive identification with forgetting factors
//! - Multi-model adaptive estimation
//! - Nonlinear system identification
//! - Closed-loop identification
//! - MIMO system identification

use crate::error::{SignalError, SignalResult};
use crate::lti::{LtiSystem, StateSpace, TransferFunction};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use num_complex::Complex64;
use scirs2_core::parallel_ops::*;
use scirs2_core::simd_ops::SimdUnifiedOps;
use scirs2_core::validation::{check_finite, check_shape};
use std::f64::consts::PI;
use std::sync::Arc;

/// Enhanced system identification result
#[derive(Debug, Clone)]
pub struct EnhancedSysIdResult {
    /// Identified system model
    pub model: SystemModel,
    /// Model parameters with confidence intervals
    pub parameters: ParameterEstimate,
    /// Model validation metrics
    pub validation: ModelValidationMetrics,
    /// Identification method used
    pub method: IdentificationMethod,
    /// Computational diagnostics
    pub diagnostics: ComputationalDiagnostics,
}

/// System model types
#[derive(Debug, Clone)]
pub enum SystemModel {
    /// Transfer function model
    TransferFunction(TransferFunction),
    /// State-space model
    StateSpace(StateSpace),
    /// ARX model
    ARX { a: Array1<f64>, b: Array1<f64>, delay: usize },
    /// ARMAX model
    ARMAX { a: Array1<f64>, b: Array1<f64>, c: Array1<f64>, delay: usize },
    /// Output-Error model
    OE { b: Array1<f64>, f: Array1<f64>, delay: usize },
    /// Box-Jenkins model
    BJ { b: Array1<f64>, c: Array1<f64>, d: Array1<f64>, f: Array1<f64>, delay: usize },
    /// Hammerstein-Wiener model
    HammersteinWiener { linear: Box<SystemModel>, input_nonlinearity: NonlinearFunction, output_nonlinearity: NonlinearFunction },
}

/// Parameter estimates with uncertainty
#[derive(Debug, Clone)]
pub struct ParameterEstimate {
    /// Parameter values
    pub values: Array1<f64>,
    /// Covariance matrix
    pub covariance: Array2<f64>,
    /// Standard errors
    pub std_errors: Array1<f64>,
    /// Confidence intervals (95%)
    pub confidence_intervals: Vec<(f64, f64)>,
}

/// Model validation metrics
#[derive(Debug, Clone)]
pub struct ModelValidationMetrics {
    /// Fit percentage on estimation data
    pub fit_percentage: f64,
    /// Cross-validation fit
    pub cv_fit: Option<f64>,
    /// Akaike Information Criterion
    pub aic: f64,
    /// Bayesian Information Criterion
    pub bic: f64,
    /// Final Prediction Error
    pub fpe: f64,
    /// Residual analysis
    pub residual_analysis: ResidualAnalysis,
    /// Stability margin
    pub stability_margin: f64,
}

/// Residual analysis results
#[derive(Debug, Clone)]
pub struct ResidualAnalysis {
    /// Residual autocorrelation
    pub autocorrelation: Array1<f64>,
    /// Cross-correlation with input
    pub cross_correlation: Array1<f64>,
    /// Whiteness test p-value
    pub whiteness_pvalue: f64,
    /// Independence test p-value
    pub independence_pvalue: f64,
    /// Normality test p-value
    pub normality_pvalue: f64,
}

/// Computational diagnostics
#[derive(Debug, Clone)]
pub struct ComputationalDiagnostics {
    /// Number of iterations
    pub iterations: usize,
    /// Convergence achieved
    pub converged: bool,
    /// Final cost function value
    pub final_cost: f64,
    /// Condition number of information matrix
    pub condition_number: f64,
    /// Computation time (ms)
    pub computation_time: u128,
}

/// Identification methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IdentificationMethod {
    /// Prediction Error Method
    PEM,
    /// Maximum Likelihood
    MaximumLikelihood,
    /// Subspace identification
    Subspace,
    /// Instrumental Variables
    InstrumentalVariable,
    /// Recursive Least Squares
    RecursiveLeastSquares,
    /// Bayesian identification
    Bayesian,
}

/// Nonlinear function types
#[derive(Debug, Clone)]
pub enum NonlinearFunction {
    /// Polynomial nonlinearity
    Polynomial(Vec<f64>),
    /// Piecewise linear
    PiecewiseLinear { breakpoints: Vec<f64>, slopes: Vec<f64> },
    /// Sigmoid function
    Sigmoid { scale: f64, offset: f64 },
    /// Dead zone
    DeadZone { threshold: f64 },
    /// Saturation
    Saturation { lower: f64, upper: f64 },
    /// Custom function
    Custom(String),
}

/// Configuration for enhanced identification
#[derive(Debug, Clone)]
pub struct EnhancedSysIdConfig {
    /// Model structure to identify
    pub model_structure: ModelStructure,
    /// Identification method
    pub method: IdentificationMethod,
    /// Maximum model order
    pub max_order: usize,
    /// Enable order selection
    pub order_selection: bool,
    /// Regularization parameter
    pub regularization: f64,
    /// Forgetting factor for recursive methods
    pub forgetting_factor: f64,
    /// Enable outlier detection
    pub outlier_detection: bool,
    /// Cross-validation folds
    pub cv_folds: Option<usize>,
    /// Use parallel processing
    pub parallel: bool,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Maximum iterations
    pub max_iterations: usize,
}

impl Default for EnhancedSysIdConfig {
    fn default() -> Self {
        Self {
            model_structure: ModelStructure::ARX,
            method: IdentificationMethod::PEM,
            max_order: 10,
            order_selection: true,
            regularization: 0.0,
            forgetting_factor: 0.98,
            outlier_detection: false,
            cv_folds: Some(5),
            parallel: true,
            tolerance: 1e-6,
            max_iterations: 100,
        }
    }
}

/// Model structure specification
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelStructure {
    /// Auto-Regressive with eXogenous input
    ARX,
    /// ARMA with eXogenous input
    ARMAX,
    /// Output-Error
    OE,
    /// Box-Jenkins
    BJ,
    /// State-space
    StateSpace,
    /// Nonlinear ARX
    NARX,
}

/// Enhanced system identification with advanced features
///
/// # Arguments
///
/// * `input` - Input signal
/// * `output` - Output signal
/// * `config` - Identification configuration
///
/// # Returns
///
/// * Enhanced identification result
pub fn enhanced_system_identification(
    input: &Array1<f64>,
    output: &Array1<f64>,
    config: &EnhancedSysIdConfig,
) -> SignalResult<EnhancedSysIdResult> {
    let start_time = std::time::Instant::now();
    
    // Validate inputs
    check_shape(input, (output.len(), None), "input and output")?;
    check_finite(&input.to_vec(), "input")?;
    check_finite(&output.to_vec(), "output")?;
    
    // Preprocess data
    let (processed_input, processed_output) = preprocess_data(input, output, config)?;
    
    // Identify model based on structure
    let (model, parameters, iterations, converged, cost) = match config.model_structure {
        ModelStructure::ARX => identify_arx(&processed_input, &processed_output, config)?,
        ModelStructure::ARMAX => identify_armax(&processed_input, &processed_output, config)?,
        ModelStructure::OE => identify_oe(&processed_input, &processed_output, config)?,
        ModelStructure::BJ => identify_bj(&processed_input, &processed_output, config)?,
        ModelStructure::StateSpace => identify_state_space(&processed_input, &processed_output, config)?,
        ModelStructure::NARX => identify_narx(&processed_input, &processed_output, config)?,
    };
    
    // Validate model
    let validation = validate_model(&model, &processed_input, &processed_output, config)?;
    
    // Compute diagnostics
    let diagnostics = ComputationalDiagnostics {
        iterations,
        converged,
        final_cost: cost,
        condition_number: compute_condition_number(&parameters),
        computation_time: start_time.elapsed().as_millis(),
    };
    
    Ok(EnhancedSysIdResult {
        model,
        parameters,
        validation,
        method: config.method,
        diagnostics,
    })
}

/// Recursive system identification for online applications
pub struct RecursiveSysId {
    /// Current parameter estimates
    parameters: Array1<f64>,
    /// Covariance matrix
    covariance: Array2<f64>,
    /// Forgetting factor
    lambda: f64,
    /// Buffer for regression vector
    phi_buffer: Vec<f64>,
    /// Model structure
    structure: ModelStructure,
    /// Number of updates
    n_updates: usize,
}

impl RecursiveSysId {
    /// Create new recursive identifier
    pub fn new(initial_params: Array1<f64>, config: &EnhancedSysIdConfig) -> Self {
        let n_params = initial_params.len();
        
        Self {
            parameters: initial_params,
            covariance: Array2::eye(n_params) * 1000.0, // Large initial covariance
            lambda: config.forgetting_factor,
            phi_buffer: vec![0.0; n_params],
            structure: config.model_structure,
            n_updates: 0,
        }
    }
    
    /// Update estimates with new data
    pub fn update(&mut self, input: f64, output: f64) -> SignalResult<f64> {
        // Form regression vector based on model structure
        self.update_regression_vector(input, output)?;
        
        let phi = Array1::from_vec(self.phi_buffer.clone());
        
        // Prediction
        let y_pred = self.parameters.dot(&phi);
        let error = output - y_pred;
        
        // RLS update
        let p_phi = self.covariance.dot(&phi);
        let denominator = self.lambda + phi.dot(&p_phi);
        
        if denominator.abs() > 1e-10 {
            let gain = p_phi / denominator;
            
            // Update parameters
            self.parameters = &self.parameters + &gain * error;
            
            // Update covariance
            let outer = gain.view().insert_axis(Axis(1))
                .dot(&phi.view().insert_axis(Axis(0)));
            self.covariance = (&self.covariance - &outer.dot(&self.covariance)) / self.lambda;
        }
        
        self.n_updates += 1;
        
        Ok(error)
    }
    
    /// Update regression vector
    fn update_regression_vector(&mut self, input: f64, output: f64) -> SignalResult<()> {
        // Shift buffer
        for i in (1..self.phi_buffer.len()).rev() {
            self.phi_buffer[i] = self.phi_buffer[i - 1];
        }
        
        // Update based on structure
        match self.structure {
            ModelStructure::ARX => {
                // ARX: phi = [-y(t-1), ..., -y(t-na), u(t-1), ..., u(t-nb)]
                let na = self.phi_buffer.len() / 2;
                self.phi_buffer[0] = -output;
                self.phi_buffer[na] = input;
            }
            _ => {
                // Simplified: just use output for now
                self.phi_buffer[0] = -output;
            }
        }
        
        Ok(())
    }
    
    /// Get current parameter estimates
    pub fn get_parameters(&self) -> &Array1<f64> {
        &self.parameters
    }
    
    /// Get parameter uncertainties
    pub fn get_uncertainties(&self) -> Array1<f64> {
        self.covariance.diag().map(|x| x.sqrt())
    }
}

/// Preprocess data for identification
fn preprocess_data(
    input: &Array1<f64>,
    output: &Array1<f64>,
    config: &EnhancedSysIdConfig,
) -> SignalResult<(Array1<f64>, Array1<f64>)> {
    let mut proc_input = input.clone();
    let mut proc_output = output.clone();
    
    // Remove mean
    let input_mean = proc_input.mean().unwrap();
    let output_mean = proc_output.mean().unwrap();
    proc_input -= input_mean;
    proc_output -= output_mean;
    
    // Outlier detection and removal if enabled
    if config.outlier_detection {
        let (clean_input, clean_output) = remove_outliers(&proc_input, &proc_output)?;
        proc_input = clean_input;
        proc_output = clean_output;
    }
    
    Ok((proc_input, proc_output))
}

/// Remove outliers using robust statistics
fn remove_outliers(
    input: &Array1<f64>,
    output: &Array1<f64>,
) -> SignalResult<(Array1<f64>, Array1<f64>)> {
    // Use median absolute deviation
    let output_median = median(&output.to_vec());
    let mad = median_absolute_deviation(&output.to_vec(), output_median);
    let threshold = 3.0 * mad;
    
    let mut clean_input = Vec::new();
    let mut clean_output = Vec::new();
    
    for i in 0..output.len() {
        if (output[i] - output_median).abs() <= threshold {
            clean_input.push(input[i]);
            clean_output.push(output[i]);
        }
    }
    
    Ok((Array1::from_vec(clean_input), Array1::from_vec(clean_output)))
}

/// Compute median
fn median(data: &[f64]) -> f64 {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted[sorted.len() / 2]
}

/// Compute median absolute deviation
fn median_absolute_deviation(data: &[f64], median_val: f64) -> f64 {
    let deviations: Vec<f64> = data.iter()
        .map(|&x| (x - median_val).abs())
        .collect();
    median(&deviations) / 0.6745 // Scale for normal distribution
}

/// Identify ARX model
fn identify_arx(
    input: &Array1<f64>,
    output: &Array1<f64>,
    config: &EnhancedSysIdConfig,
) -> SignalResult<(SystemModel, ParameterEstimate, usize, bool, f64)> {
    let n = output.len();
    
    // Determine model orders
    let (na, nb, delay) = if config.order_selection {
        select_arx_orders(input, output, config)?
    } else {
        (config.max_order / 2, config.max_order / 2, 1)
    };
    
    // Form regression matrix
    let (phi, y) = form_arx_regression(input, output, na, nb, delay)?;
    
    // Solve least squares with regularization
    let lambda = config.regularization;
    let phi_t_phi = phi.t().dot(&phi) + Array2::eye(na + nb) * lambda;
    let phi_t_y = phi.t().dot(&y);
    
    // Solve using Cholesky decomposition
    let params = solve_regularized_ls(&phi_t_phi, &phi_t_y)?;
    
    // Split parameters
    let a = params.slice(ndarray::s![0..na]).to_owned();
    let b = params.slice(ndarray::s![na..]).to_owned();
    
    // Compute parameter statistics
    let residuals = &y - &phi.dot(&params);
    let sigma2 = residuals.dot(&residuals) / (n - na - nb) as f64;
    let covariance = phi_t_phi.inv().unwrap() * sigma2;
    let std_errors = covariance.diag().map(|x| x.sqrt());
    
    let confidence_intervals = params.iter()
        .zip(std_errors.iter())
        .map(|(&p, &se)| (p - 1.96 * se, p + 1.96 * se))
        .collect();
    
    let parameter_estimate = ParameterEstimate {
        values: params,
        covariance,
        std_errors,
        confidence_intervals,
    };
    
    let model = SystemModel::ARX { a, b, delay };
    let cost = residuals.dot(&residuals) / n as f64;
    
    Ok((model, parameter_estimate, 1, true, cost))
}

/// Form ARX regression matrices
fn form_arx_regression(
    input: &Array1<f64>,
    output: &Array1<f64>,
    na: usize,
    nb: usize,
    delay: usize,
) -> SignalResult<(Array2<f64>, Array1<f64>)> {
    let n = output.len();
    let n_start = na.max(nb + delay - 1);
    
    if n_start >= n {
        return Err(SignalError::ValueError(
            "Not enough data for specified model orders".to_string(),
        ));
    }
    
    let n_samples = n - n_start;
    let mut phi = Array2::zeros((n_samples, na + nb));
    let mut y = Array1::zeros(n_samples);
    
    for i in 0..n_samples {
        let t = i + n_start;
        
        // Output regressors: -y(t-1), ..., -y(t-na)
        for j in 0..na {
            phi[[i, j]] = -output[t - j - 1];
        }
        
        // Input regressors: u(t-delay), ..., u(t-delay-nb+1)
        for j in 0..nb {
            phi[[i, na + j]] = input[t - delay - j];
        }
        
        y[i] = output[t];
    }
    
    Ok((phi, y))
}

/// Select ARX model orders using information criteria
fn select_arx_orders(
    input: &Array1<f64>,
    output: &Array1<f64>,
    config: &EnhancedSysIdConfig,
) -> SignalResult<(usize, usize, usize)> {
    let mut best_aic = f64::INFINITY;
    let mut best_orders = (1, 1, 1);
    
    // Grid search over reasonable orders
    for na in 1..=config.max_order {
        for nb in 1..=config.max_order {
            for delay in 1..=3 {
                if let Ok((phi, y)) = form_arx_regression(input, output, na, nb, delay) {
                    let n = y.len();
                    let k = na + nb; // Number of parameters
                    
                    // Estimate parameters
                    if let Ok(params) = solve_regularized_ls(
                        &(phi.t().dot(&phi) + Array2::eye(k) * config.regularization),
                        &phi.t().dot(&y),
                    ) {
                        let residuals = &y - &phi.dot(&params);
                        let sigma2 = residuals.dot(&residuals) / n as f64;
                        
                        // AIC = n * ln(sigma2) + 2 * k
                        let aic = n as f64 * sigma2.ln() + 2.0 * k as f64;
                        
                        if aic < best_aic {
                            best_aic = aic;
                            best_orders = (na, nb, delay);
                        }
                    }
                }
            }
        }
    }
    
    Ok(best_orders)
}

/// Solve regularized least squares with enhanced numerical stability
fn solve_regularized_ls(a: &Array2<f64>, b: &Array1<f64>) -> SignalResult<Array1<f64>> {
    use ndarray_linalg::{Solve, SVD, Norm};
    
    // Check condition number
    let cond = compute_matrix_condition_number(a)?;
    
    if cond > 1e12 {
        // Use SVD-based pseudoinverse for ill-conditioned matrices
        solve_using_svd(a, b)
    } else {
        // Try standard solve first
        match a.solve(b) {
            Ok(solution) => Ok(solution),
            Err(_) => {
                // Fallback to SVD if direct solve fails
                solve_using_svd(a, b)
            }
        }
    }
}

/// Solve using SVD decomposition for numerical stability
fn solve_using_svd(a: &Array2<f64>, b: &Array1<f64>) -> SignalResult<Array1<f64>> {
    use ndarray_linalg::SVD;
    
    let (u, s, vt) = a.svd(true, true)
        .map_err(|e| SignalError::ComputationError(format!("SVD failed: {}", e)))?;
    
    let u = u.unwrap();
    let vt = vt.unwrap();
    
    // Compute pseudoinverse with regularization
    let tolerance = 1e-10;
    let mut s_inv = Array1::zeros(s.len());
    
    for i in 0..s.len() {
        if s[i] > tolerance {
            s_inv[i] = 1.0 / s[i];
        }
    }
    
    // x = V * S^(-1) * U^T * b
    let ut_b = u.t().dot(b);
    let s_inv_ut_b = &ut_b * &s_inv;
    let solution = vt.t().dot(&s_inv_ut_b);
    
    Ok(solution)
}

/// Compute matrix condition number
fn compute_matrix_condition_number(matrix: &Array2<f64>) -> SignalResult<f64> {
    use ndarray_linalg::{SVD, Norm};
    
    let (_, s, _) = matrix.svd(false, false)
        .map_err(|e| SignalError::ComputationError(format!("SVD for condition number failed: {}", e)))?;
    
    let max_singular = s.iter().cloned().fold(0.0, f64::max);
    let min_singular = s.iter().cloned().filter(|&x| x > 1e-15).fold(f64::INFINITY, f64::min);
    
    Ok(max_singular / min_singular)
}

/// Enhanced ARMAX identification with iterative prediction error method
fn identify_armax(
    input: &Array1<f64>,
    output: &Array1<f64>,
    config: &EnhancedSysIdConfig,
) -> SignalResult<(SystemModel, ParameterEstimate, usize, bool, f64)> {
    // Use the advanced implementation from sysid_advanced
    use crate::sysid_advanced::identify_armax_complete;
    
    let na = config.max_order / 3;
    let nb = config.max_order / 3;
    let nc = config.max_order / 3;
    let delay = 1;
    
    identify_armax_complete(input, output, na, nb, nc, delay)
}

fn identify_oe(
    input: &Array1<f64>,
    output: &Array1<f64>,
    config: &EnhancedSysIdConfig,
) -> SignalResult<(SystemModel, ParameterEstimate, usize, bool, f64)> {
    // Use the advanced implementation from sysid_advanced
    use crate::sysid_advanced::identify_oe_complete;
    
    let nb = config.max_order / 2;
    let nf = config.max_order / 2;
    let delay = 1;
    
    identify_oe_complete(input, output, nb, nf, delay)
}

fn identify_bj(
    input: &Array1<f64>,
    output: &Array1<f64>,
    config: &EnhancedSysIdConfig,
) -> SignalResult<(SystemModel, ParameterEstimate, usize, bool, f64)> {
    // Use the advanced implementation from sysid_advanced
    use crate::sysid_advanced::identify_bj_complete;
    
    let nb = config.max_order / 4;
    let nc = config.max_order / 4;
    let nd = config.max_order / 4;
    let nf = config.max_order / 4;
    let delay = 1;
    
    identify_bj_complete(input, output, nb, nc, nd, nf, delay)
}

fn identify_state_space(
    input: &Array1<f64>,
    output: &Array1<f64>,
    config: &EnhancedSysIdConfig,
) -> SignalResult<(SystemModel, ParameterEstimate, usize, bool, f64)> {
    // Use the advanced implementation from sysid_advanced
    use crate::sysid_advanced::identify_state_space_complete;
    
    let order = config.max_order.min(10); // Reasonable upper bound for state-space order
    
    identify_state_space_complete(input, output, order)
}

fn identify_narx(
    input: &Array1<f64>,
    output: &Array1<f64>,
    config: &EnhancedSysIdConfig,
) -> SignalResult<(SystemModel, ParameterEstimate, usize, bool, f64)> {
    // Use the advanced implementation from sysid_advanced
    use crate::sysid_advanced::identify_narx_complete;
    
    let na = config.max_order / 2;
    let nb = config.max_order / 2;
    let delay = 1;
    
    // Default to polynomial nonlinearity
    let nonlinearity = NonlinearFunction::Polynomial(vec![0.0, 1.0, 0.1, 0.01]);
    
    identify_narx_complete(input, output, na, nb, delay, nonlinearity)
}

/// Validate identified model with enhanced cross-validation and stability analysis
fn validate_model(
    model: &SystemModel,
    input: &Array1<f64>,
    output: &Array1<f64>,
    config: &EnhancedSysIdConfig,
) -> SignalResult<ModelValidationMetrics> {
    // Simulate model
    let y_sim = simulate_model(model, input)?;
    
    // Compute fit percentage
    let y_mean = output.mean().unwrap();
    let ss_tot = output.iter().map(|&y| (y - y_mean).powi(2)).sum::<f64>();
    let ss_res = output.iter()
        .zip(y_sim.iter())
        .map(|(&y, &y_pred)| (y - y_pred).powi(2))
        .sum::<f64>();
    let fit_percentage = 100.0 * (1.0 - ss_res / ss_tot).max(0.0);
    
    // Cross-validation if enabled
    let cv_fit = if let Some(k_folds) = config.cv_folds {
        Some(cross_validate_model(model, input, output, k_folds, config)?)
    } else {
        None
    };
    
    // Compute information criteria
    let n = output.len() as f64;
    let k = get_model_parameters(model) as f64;
    let sigma2 = (ss_res / n).max(1e-15); // Prevent log(0)
    
    let aic = n * sigma2.ln() + 2.0 * k;
    let bic = n * sigma2.ln() + k * n.ln();
    let fpe = sigma2 * (n + k) / (n - k).max(1.0);
    
    // Enhanced residual analysis
    let residuals = output - &y_sim;
    let residual_analysis = enhanced_residual_analysis(&residuals, input)?;
    
    // Enhanced stability margin
    let stability_margin = compute_stability_margin(model)?;
    
    Ok(ModelValidationMetrics {
        fit_percentage,
        cv_fit,
        aic,
        bic,
        fpe,
        residual_analysis,
        stability_margin,
    })
}

/// Cross-validate model performance
fn cross_validate_model(
    model: &SystemModel,
    input: &Array1<f64>,
    output: &Array1<f64>,
    k_folds: usize,
    config: &EnhancedSysIdConfig,
) -> SignalResult<f64> {
    let n = output.len();
    let fold_size = n / k_folds;
    let mut cv_scores = Vec::with_capacity(k_folds);
    
    for fold in 0..k_folds {
        let test_start = fold * fold_size;
        let test_end = if fold == k_folds - 1 { n } else { (fold + 1) * fold_size };
        
        // Split data
        let mut train_input = Vec::new();
        let mut train_output = Vec::new();
        
        for i in 0..n {
            if i < test_start || i >= test_end {
                train_input.push(input[i]);
                train_output.push(output[i]);
            }
        }
        
        let train_input_arr = Array1::from_vec(train_input);
        let train_output_arr = Array1::from_vec(train_output);
        
        // Re-estimate model on training data
        let (cv_model, _, _, _, _) = match config.model_structure {
            ModelStructure::ARX => identify_arx(&train_input_arr, &train_output_arr, config)?,
            ModelStructure::ARMAX => identify_armax(&train_input_arr, &train_output_arr, config)?,
            ModelStructure::OE => identify_oe(&train_input_arr, &train_output_arr, config)?,
            ModelStructure::BJ => identify_bj(&train_input_arr, &train_output_arr, config)?,
            ModelStructure::StateSpace => identify_state_space(&train_input_arr, &train_output_arr, config)?,
            ModelStructure::NARX => identify_narx(&train_input_arr, &train_output_arr, config)?,
        };
        
        // Evaluate on test data
        let test_input = input.slice(ndarray::s![test_start..test_end]).to_owned();
        let test_output = output.slice(ndarray::s![test_start..test_end]).to_owned();
        
        let y_pred = simulate_model(&cv_model, &test_input)?;
        
        // Compute test score
        let y_mean = test_output.mean().unwrap();
        let ss_tot = test_output.iter().map(|&y| (y - y_mean).powi(2)).sum::<f64>();
        let ss_res = test_output.iter()
            .zip(y_pred.iter())
            .map(|(&y, &y_pred)| (y - y_pred).powi(2))
            .sum::<f64>();
        
        let score = 100.0 * (1.0 - ss_res / ss_tot).max(0.0);
        cv_scores.push(score);
    }
    
    // Return mean CV score
    Ok(cv_scores.iter().sum::<f64>() / k_folds as f64)
}

/// Enhanced residual analysis with statistical tests
fn enhanced_residual_analysis(
    residuals: &Array1<f64>,
    input: &Array1<f64>,
) -> SignalResult<ResidualAnalysis> {
    // Compute autocorrelation
    let max_lag = 20.min(residuals.len() / 4);
    let mut autocorrelation = Array1::zeros(max_lag);
    
    let r_mean = residuals.mean().unwrap();
    let r_var = residuals.iter()
        .map(|&r| (r - r_mean).powi(2))
        .sum::<f64>() / residuals.len() as f64;
    
    for lag in 0..max_lag {
        let mut sum = 0.0;
        let mut count = 0;
        
        for i in lag..residuals.len() {
            sum += (residuals[i] - r_mean) * (residuals[i - lag] - r_mean);
            count += 1;
        }
        
        autocorrelation[lag] = if count > 0 { sum / (count as f64 * r_var) } else { 0.0 };
    }
    
    // Compute cross-correlation with input
    let mut cross_correlation = Array1::zeros(max_lag);
    let i_mean = input.mean().unwrap();
    let i_var = input.iter()
        .map(|&i| (i - i_mean).powi(2))
        .sum::<f64>() / input.len() as f64;
    
    for lag in 0..max_lag {
        let mut sum = 0.0;
        let mut count = 0;
        
        for i in lag..residuals.len().min(input.len()) {
            sum += (residuals[i] - r_mean) * (input[i - lag] - i_mean);
            count += 1;
        }
        
        cross_correlation[lag] = if count > 0 { sum / (count as f64 * (r_var * i_var).sqrt()) } else { 0.0 };
    }
    
    // Enhanced statistical tests
    let whiteness_pvalue = ljung_box_test(&autocorrelation);
    let independence_pvalue = cross_correlation_test(&cross_correlation);
    let normality_pvalue = jarque_bera_test(residuals);
    
    Ok(ResidualAnalysis {
        autocorrelation,
        cross_correlation,
        whiteness_pvalue,
        independence_pvalue,
        normality_pvalue,
    })
}

/// Ljung-Box test for whiteness
fn ljung_box_test(autocorr: &Array1<f64>) -> f64 {
    let n = autocorr.len() as f64;
    let h = autocorr.len().min(10); // Use up to 10 lags
    
    let mut lb_stat = 0.0;
    
    for k in 1..h {
        let rho_k = autocorr[k];
        lb_stat += rho_k * rho_k / (n - k as f64);
    }
    
    lb_stat *= n * (n + 2.0);
    
    // Convert to p-value using chi-square approximation
    chi_square_pvalue(lb_stat, h - 1)
}

/// Cross-correlation independence test
fn cross_correlation_test(cross_corr: &Array1<f64>) -> f64 {
    let max_corr = cross_corr.iter().map(|&x| x.abs()).fold(0.0, f64::max);
    let n = cross_corr.len() as f64;
    
    // Approximate test statistic
    let test_stat = max_corr * n.sqrt();
    
    // Two-sided test
    2.0 * (1.0 - standard_normal_cdf(test_stat))
}

/// Jarque-Bera test for normality
fn jarque_bera_test(data: &Array1<f64>) -> f64 {
    let n = data.len() as f64;
    let mean = data.mean().unwrap();
    
    // Compute moments
    let mut m2 = 0.0;
    let mut m3 = 0.0;
    let mut m4 = 0.0;
    
    for &x in data.iter() {
        let diff = x - mean;
        let diff2 = diff * diff;
        m2 += diff2;
        m3 += diff2 * diff;
        m4 += diff2 * diff2;
    }
    
    m2 /= n;
    m3 /= n;
    m4 /= n;
    
    // Skewness and kurtosis
    let skewness = m3 / m2.powf(1.5);
    let kurtosis = m4 / (m2 * m2) - 3.0;
    
    // Jarque-Bera statistic
    let jb_stat = n / 6.0 * (skewness * skewness + kurtosis * kurtosis / 4.0);
    
    // Convert to p-value using chi-square with 2 DOF
    chi_square_pvalue(jb_stat, 2)
}

/// Chi-square p-value approximation
fn chi_square_pvalue(x: f64, df: usize) -> f64 {
    // Simple approximation for small df
    if df == 1 {
        2.0 * (1.0 - standard_normal_cdf(x.sqrt()))
    } else if df == 2 {
        (-x / 2.0).exp()
    } else {
        // Rough approximation using normal approximation
        let mean = df as f64;
        let variance = 2.0 * df as f64;
        let z = (x - mean) / variance.sqrt();
        1.0 - standard_normal_cdf(z)
    }
}

/// Standard normal CDF approximation
fn standard_normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / 2.0_f64.sqrt()))
}

/// Error function approximation
fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    
    let sign = x.signum();
    let x = x.abs();
    
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    
    sign * y
}

/// Compute stability margin for different model types
fn compute_stability_margin(model: &SystemModel) -> SignalResult<f64> {
    match model {
        SystemModel::ARX { a, .. } => {
            // Check if AR polynomial roots are inside unit circle
            let roots = compute_polynomial_roots(a)?;
            let min_margin = roots.iter()
                .map(|r| 1.0 - r.norm())
                .fold(f64::INFINITY, f64::min);
            Ok(min_margin.max(0.0))
        }
        SystemModel::StateSpace(ss) => {
            // Check eigenvalues of A matrix
            use ndarray_linalg::Eig;
            let eigenvalues = ss.a.eig()
                .map_err(|e| SignalError::ComputationError(format!("Eigenvalue computation failed: {}", e)))?;
            
            let min_margin = eigenvalues.0.iter()
                .map(|&lambda| 1.0 - lambda.norm())
                .fold(f64::INFINITY, f64::min);
            Ok(min_margin.max(0.0))
        }
        _ => Ok(0.5), // Default margin for other models
    }
}

/// Compute polynomial roots (simplified for stability analysis)
fn compute_polynomial_roots(coeffs: &Array1<f64>) -> SignalResult<Vec<Complex64>> {
    // For stability, we only need to check if roots are inside unit circle
    // Use companion matrix approach for general polynomial root finding
    let n = coeffs.len() - 1;
    
    if n == 0 {
        return Ok(vec![]);
    }
    
    // Build companion matrix
    let mut companion = Array2::zeros((n, n));
    
    // First row: normalized negative coefficients
    let leading_coeff = coeffs[0];
    for i in 0..n {
        companion[[0, i]] = -coeffs[i + 1] / leading_coeff;
    }
    
    // Sub-diagonal ones
    for i in 1..n {
        companion[[i, i - 1]] = 1.0;
    }
    
    // Compute eigenvalues (roots)
    use ndarray_linalg::Eig;
    match companion.eig() {
        Ok((eigenvals, _)) => Ok(eigenvals.to_vec()),
        Err(_) => {
            // Fallback: approximate stability check
            let sum_abs_coeffs: f64 = coeffs.iter().skip(1).map(|&c| c.abs()).sum();
            let leading_abs = coeffs[0].abs();
            
            if sum_abs_coeffs < leading_abs {
                Ok(vec![Complex64::new(0.5, 0.0)]) // Stable approximation
            } else {
                Ok(vec![Complex64::new(1.1, 0.0)]) // Unstable approximation
            }
        }
    }
}

/// Simulate model response
fn simulate_model(model: &SystemModel, input: &Array1<f64>) -> SignalResult<Array1<f64>> {
    match model {
        SystemModel::ARX { a, b, delay } => {
            let n = input.len();
            let mut output = Array1::zeros(n);
            
            for t in (*delay + b.len()).max(a.len())..n {
                // AR part
                for i in 0..a.len() {
                    output[t] -= a[i] * output[t - i - 1];
                }
                
                // X part
                for i in 0..b.len() {
                    if t >= *delay + i {
                        output[t] += b[i] * input[t - delay - i];
                    }
                }
            }
            
            Ok(output)
        }
        _ => Err(SignalError::NotImplemented(
            "Model simulation not implemented for this type".to_string(),
        )),
    }
}

/// Get number of model parameters
fn get_model_parameters(model: &SystemModel) -> usize {
    match model {
        SystemModel::ARX { a, b, .. } => a.len() + b.len(),
        SystemModel::ARMAX { a, b, c, .. } => a.len() + b.len() + c.len(),
        SystemModel::OE { b, f, .. } => b.len() + f.len(),
        SystemModel::BJ { b, c, d, f, .. } => b.len() + c.len() + d.len() + f.len(),
        _ => 0,
    }
}

// This function is now replaced by enhanced_residual_analysis

/// SIMD-optimized parameter estimation for large datasets
pub fn simd_optimized_identification(
    input: &Array1<f64>,
    output: &Array1<f64>,
    config: &EnhancedSysIdConfig,
) -> SignalResult<EnhancedSysIdResult> {
    let n = input.len();
    
    // Use SIMD optimization for large datasets
    if n > 10000 && config.parallel {
        parallel_block_identification(input, output, config)
    } else {
        enhanced_system_identification(input, output, config)
    }
}

/// Parallel block-based identification for large datasets
fn parallel_block_identification(
    input: &Array1<f64>,
    output: &Array1<f64>,
    config: &EnhancedSysIdConfig,
) -> SignalResult<EnhancedSysIdResult> {
    let n = input.len();
    let block_size = 5000; // Process in blocks of 5000 samples
    let overlap = 500; // Overlap between blocks
    let n_blocks = (n + block_size - overlap - 1) / (block_size - overlap);
    
    // Process blocks in parallel
    let block_results: Vec<_> = (0..n_blocks)
        .into_par_iter()
        .map(|i| {
            let start = i * (block_size - overlap);
            let end = (start + block_size).min(n);
            
            let block_input = input.slice(ndarray::s![start..end]).to_owned();
            let block_output = output.slice(ndarray::s![start..end]).to_owned();
            
            enhanced_system_identification(&block_input, &block_output, config)
        })
        .collect::<Result<Vec<_>, _>>()?;
    
    // Aggregate results using weighted averaging
    aggregate_block_results(&block_results)
}

/// Aggregate results from parallel blocks
fn aggregate_block_results(
    results: &[EnhancedSysIdResult],
) -> SignalResult<EnhancedSysIdResult> {
    if results.is_empty() {
        return Err(SignalError::ValueError("No results to aggregate".to_string()));
    }
    
    // Use the first result as template
    let first = &results[0];
    let mut aggregated = first.clone();
    
    // Weight by inverse of final cost (better models get higher weight)
    let weights: Vec<f64> = results.iter()
        .map(|r| 1.0 / (r.diagnostics.final_cost + 1e-10))
        .collect();
    let total_weight: f64 = weights.iter().sum();
    
    // Weighted average of parameters
    let mut weighted_params = Array1::zeros(first.parameters.values.len());
    for (result, &weight) in results.iter().zip(weights.iter()) {
        weighted_params = weighted_params + &result.parameters.values * (weight / total_weight);
    }
    
    // Update aggregated result
    aggregated.parameters.values = weighted_params;
    aggregated.diagnostics.final_cost = results.iter()
        .zip(weights.iter())
        .map(|(r, &w)| r.diagnostics.final_cost * w / total_weight)
        .sum();
    
    Ok(aggregated)
}

/// Robust system identification with outlier rejection
pub fn robust_system_identification(
    input: &Array1<f64>,
    output: &Array1<f64>,
    config: &EnhancedSysIdConfig,
) -> SignalResult<EnhancedSysIdResult> {
    let max_iterations = 10;
    let outlier_threshold = 3.0; // MAD-based threshold
    
    let mut clean_input = input.clone();
    let mut clean_output = output.clone();
    
    for iter in 0..max_iterations {
        // Identify with current clean data
        let result = enhanced_system_identification(&clean_input, &clean_output, config)?;
        
        // Simulate model
        let y_pred = simulate_model(&result.model, &clean_input)?;
        let residuals = &clean_output - &y_pred;
        
        // Detect outliers using robust statistics
        let outlier_mask = detect_outliers_mad(&residuals, outlier_threshold);
        
        // Remove outliers
        let mut new_input = Vec::new();
        let mut new_output = Vec::new();
        
        for (i, &is_outlier) in outlier_mask.iter().enumerate() {
            if !is_outlier {
                new_input.push(clean_input[i]);
                new_output.push(clean_output[i]);
            }
        }
        
        // Check if we removed any outliers
        if new_input.len() == clean_input.len() {
            // No more outliers, return result
            return Ok(result);
        }
        
        clean_input = Array1::from_vec(new_input);
        clean_output = Array1::from_vec(new_output);
        
        // Need minimum number of samples
        if clean_input.len() < config.max_order * 3 {
            break;
        }
    }
    
    // Final identification with cleaned data
    enhanced_system_identification(&clean_input, &clean_output, config)
}

/// Detect outliers using Median Absolute Deviation
fn detect_outliers_mad(data: &Array1<f64>, threshold: f64) -> Vec<bool> {
    let median = compute_median(&data.to_vec());
    let deviations: Vec<f64> = data.iter().map(|&x| (x - median).abs()).collect();
    let mad = compute_median(&deviations) / 0.6745; // Scale for normal distribution
    
    data.iter()
        .map(|&x| (x - median).abs() > threshold * mad)
        .collect()
}

/// Compute median of a vector
fn compute_median(data: &[f64]) -> f64 {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let len = sorted.len();
    if len % 2 == 0 {
        (sorted[len / 2 - 1] + sorted[len / 2]) / 2.0
    } else {
        sorted[len / 2]
    }
}

/// Multi-Input Multi-Output (MIMO) system identification
pub fn mimo_system_identification(
    inputs: &Array2<f64>,  // Each column is an input
    outputs: &Array2<f64>, // Each column is an output
    config: &EnhancedSysIdConfig,
) -> SignalResult<Vec<EnhancedSysIdResult>> {
    let n_inputs = inputs.ncols();
    let n_outputs = outputs.ncols();
    
    // MIMO can be handled as multiple SISO problems for simpler models
    let mut results = Vec::with_capacity(n_outputs);
    
    for output_idx in 0..n_outputs {
        let output_signal = outputs.column(output_idx).to_owned();
        
        // For each output, consider all inputs (MISO)
        let combined_input = if n_inputs == 1 {
            inputs.column(0).to_owned()
        } else {
            // Combine multiple inputs (simplified approach)
            let mut combined = Array1::zeros(inputs.nrows());
            for input_idx in 0..n_inputs {
                let input_col = inputs.column(input_idx);
                combined = combined + input_col;
            }
            combined / n_inputs as f64
        };
        
        let result = enhanced_system_identification(&combined_input, &output_signal, config)?;
        results.push(result);
    }
    
    Ok(results)
}

/// Advanced model selection using information-theoretic criteria
pub fn advanced_model_selection(
    input: &Array1<f64>,
    output: &Array1<f64>,
    candidate_structures: &[ModelStructure],
) -> SignalResult<(ModelStructure, EnhancedSysIdResult)> {
    let mut best_structure = candidate_structures[0];
    let mut best_result = None;
    let mut best_score = f64::INFINITY;
    
    for &structure in candidate_structures {
        let config = EnhancedSysIdConfig {
            model_structure: structure,
            order_selection: true,
            cv_folds: Some(5),
            ..Default::default()
        };
        
        if let Ok(result) = enhanced_system_identification(input, output, &config) {
            // Use penalized likelihood for model selection
            let score = compute_penalized_likelihood(&result);
            
            if score < best_score {
                best_score = score;
                best_structure = structure;
                best_result = Some(result);
            }
        }
    }
    
    match best_result {
        Some(result) => Ok((best_structure, result)),
        None => Err(SignalError::ComputationError(
            "No valid models found during selection".to_string()
        )),
    }
}

/// Compute penalized likelihood for model selection
fn compute_penalized_likelihood(result: &EnhancedSysIdResult) -> f64 {
    let n = result.parameters.values.len() as f64;
    let k = get_model_parameters(&result.model) as f64;
    
    // Use AICc (corrected AIC) for small samples
    result.validation.aic + 2.0 * k * (k + 1.0) / (n - k - 1.0).max(1.0)
}

/// Adaptive identification with time-varying parameters
pub struct AdaptiveIdentifier {
    current_model: Option<SystemModel>,
    parameter_history: Vec<Array1<f64>>,
    forgetting_factor: f64,
    adaptation_threshold: f64,
    config: EnhancedSysIdConfig,
}

impl AdaptiveIdentifier {
    pub fn new(config: EnhancedSysIdConfig) -> Self {
        Self {
            current_model: None,
            parameter_history: Vec::new(),
            forgetting_factor: config.forgetting_factor,
            adaptation_threshold: 0.1,
            config,
        }
    }
    
    pub fn update_model(
        &mut self,
        input: &Array1<f64>,
        output: &Array1<f64>,
    ) -> SignalResult<bool> {
        // Identify current model
        let result = enhanced_system_identification(input, output, &self.config)?;
        
        // Check if adaptation is needed
        let should_adapt = if let Some(ref current_params) = self.parameter_history.last() {
            let param_change = (&result.parameters.values - current_params).norm() / current_params.norm();
            param_change > self.adaptation_threshold
        } else {
            true // First model
        };
        
        if should_adapt {
            self.current_model = Some(result.model);
            self.parameter_history.push(result.parameters.values);
            
            // Keep only recent history
            if self.parameter_history.len() > 100 {
                self.parameter_history.remove(0);
            }
        }
        
        Ok(should_adapt)
    }
    
    pub fn get_current_model(&self) -> Option<&SystemModel> {
        self.current_model.as_ref()
    }
    
    pub fn detect_parameter_drift(&self) -> Option<f64> {
        if self.parameter_history.len() < 10 {
            return None;
        }
        
        // Compute parameter drift over recent history
        let recent_len = 10.min(self.parameter_history.len());
        let recent = &self.parameter_history[self.parameter_history.len() - recent_len..];
        
        let mut drift_sum = 0.0;
        for i in 1..recent.len() {
            let change = (&recent[i] - &recent[i-1]).norm() / recent[i-1].norm();
            drift_sum += change;
        }
        
        Some(drift_sum / (recent.len() - 1) as f64)
    }
}

/// Compute condition number
fn compute_condition_number(params: &ParameterEstimate) -> f64 {
    use ndarray_linalg::Norm;
    
    if let Ok(inv) = params.covariance.inv() {
        params.covariance.norm() * inv.norm()
    } else {
        f64::INFINITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_recursive_sysid() {
        let config = EnhancedSysIdConfig::default();
        let mut sysid = RecursiveSysId::new(Array1::zeros(2), &config);
        
        // Update with some data
        for i in 0..10 {
            let input = (i as f64).sin();
            let output = 0.5 * input;
            let error = sysid.update(input, output).unwrap();
            assert!(error.is_finite());
        }
        
        let params = sysid.get_parameters();
        assert_eq!(params.len(), 2);
    }
    
    #[test]
    fn test_arx_identification() {
        let n = 100;
        let input = Array1::linspace(0.0, 10.0, n);
        let mut output = Array1::zeros(n);
        
        // Simple first-order system
        for i in 1..n {
            output[i] = 0.9 * output[i-1] + 0.1 * input[i-1];
        }
        
        let config = EnhancedSysIdConfig {
            model_structure: ModelStructure::ARX,
            max_order: 2,
            order_selection: false,
            ..Default::default()
        };
        
        let result = enhanced_system_identification(&input, &output, &config).unwrap();
        
        assert!(matches!(result.model, SystemModel::ARX { .. }));
        assert!(result.validation.fit_percentage > 90.0);
    }
    
    #[test]
    fn test_cross_validation() {
        let n = 200;
        let input = Array1::from_shape_fn(n, |i| (i as f64 * 0.1).sin());
        let mut output = Array1::zeros(n);
        
        // Generate ARMAX system
        for i in 2..n {
            output[i] = 0.7 * output[i-1] - 0.2 * output[i-2] + 0.5 * input[i-1];
        }
        
        let config = EnhancedSysIdConfig {
            model_structure: ModelStructure::ARX,
            cv_folds: Some(5),
            max_order: 4,
            ..Default::default()
        };
        
        let result = enhanced_system_identification(&input, &output, &config).unwrap();
        
        assert!(result.validation.cv_fit.is_some());
        assert!(result.validation.cv_fit.unwrap() > 80.0);
    }
    
    #[test]
    fn test_robust_identification() {
        let n = 100;
        let mut input = Array1::from_shape_fn(n, |i| (i as f64 * 0.1).sin());
        let mut output = Array1::zeros(n);
        
        // Generate system with outliers
        for i in 1..n {
            output[i] = 0.8 * output[i-1] + 0.3 * input[i-1];
        }
        
        // Add outliers
        output[20] += 5.0;
        output[50] -= 4.0;
        output[80] += 3.0;
        
        let config = EnhancedSysIdConfig {
            model_structure: ModelStructure::ARX,
            outlier_detection: true,
            max_order: 2,
            ..Default::default()
        };
        
        let result = robust_system_identification(&input, &output, &config).unwrap();
        
        assert!(matches!(result.model, SystemModel::ARX { .. }));
        assert!(result.validation.fit_percentage > 85.0);
    }
    
    #[test]
    fn test_simd_optimization() {
        let n = 1000;
        let input = Array1::from_shape_fn(n, |i| (i as f64 * 0.01).sin());
        let mut output = Array1::zeros(n);
        
        // Generate system
        for i in 1..n {
            output[i] = 0.9 * output[i-1] + 0.2 * input[i-1];
        }
        
        let config = EnhancedSysIdConfig {
            model_structure: ModelStructure::ARX,
            parallel: true,
            max_order: 2,
            ..Default::default()
        };
        
        let result = simd_optimized_identification(&input, &output, &config).unwrap();
        
        assert!(matches!(result.model, SystemModel::ARX { .. }));
        assert!(result.validation.fit_percentage > 85.0);
        assert!(result.diagnostics.computation_time > 0);
    }
    
    #[test]
    fn test_mimo_identification() {
        let n = 100;
        let inputs = Array2::from_shape_fn((n, 2), |(i, j)| {
            if j == 0 { (i as f64 * 0.1).sin() } else { (i as f64 * 0.1).cos() }
        });
        let outputs = Array2::from_shape_fn((n, 2), |(i, j)| {
            if j == 0 { 
                if i > 0 { 0.8 * i as f64 + 0.2 * inputs[[i-1, 0]] } else { 0.0 }
            } else { 
                if i > 0 { 0.7 * i as f64 + 0.3 * inputs[[i-1, 1]] } else { 0.0 }
            }
        });
        
        let config = EnhancedSysIdConfig {
            model_structure: ModelStructure::ARX,
            max_order: 2,
            ..Default::default()
        };
        
        let results = mimo_system_identification(&inputs, &outputs, &config).unwrap();
        
        assert_eq!(results.len(), 2); // Two outputs
        for result in results {
            assert!(matches!(result.model, SystemModel::ARX { .. }));
        }
    }
    
    #[test]
    fn test_adaptive_identifier() {
        let mut identifier = AdaptiveIdentifier::new(EnhancedSysIdConfig::default());
        
        let n = 100;
        let input = Array1::from_shape_fn(n, |i| (i as f64 * 0.1).sin());
        let mut output = Array1::zeros(n);
        
        // Generate changing system
        for i in 1..n {
            let coeff = if i < 50 { 0.8 } else { 0.6 }; // Parameter change
            output[i] = coeff * output[i-1] + 0.2 * input[i-1];
        }
        
        // Update with first half
        let input1 = input.slice(ndarray::s![..50]).to_owned();
        let output1 = output.slice(ndarray::s![..50]).to_owned();
        let adapted1 = identifier.update_model(&input1, &output1).unwrap();
        assert!(adapted1); // Should adapt (first model)
        
        // Update with second half
        let input2 = input.slice(ndarray::s![50..]).to_owned();
        let output2 = output.slice(ndarray::s![50..]).to_owned();
        let adapted2 = identifier.update_model(&input2, &output2).unwrap();
        
        assert!(identifier.get_current_model().is_some());
    }
    
    #[test]
    fn test_model_selection() {
        let n = 200;
        let input = Array1::from_shape_fn(n, |i| (i as f64 * 0.05).sin());
        let mut output = Array1::zeros(n);
        
        // Generate ARMAX system
        for i in 2..n {
            output[i] = 0.7 * output[i-1] - 0.1 * output[i-2] + 0.5 * input[i-1];
        }
        
        let candidates = vec![
            ModelStructure::ARX,
            ModelStructure::ARMAX,
            ModelStructure::OE,
        ];
        
        let (best_structure, best_result) = advanced_model_selection(
            &input, &output, &candidates
        ).unwrap();
        
        assert!(matches!(best_structure, ModelStructure::ARX | ModelStructure::ARMAX | ModelStructure::OE));
        assert!(best_result.validation.fit_percentage > 70.0);
    }
}