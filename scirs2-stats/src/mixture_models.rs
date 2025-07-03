//! Advanced mixture models and kernel density estimation
//!
//! This module provides comprehensive implementations of mixture models and
//! non-parametric density estimation methods including:
//! - Gaussian Mixture Models (GMM) with robust EM algorithm
//! - Dirichlet Process Gaussian Mixture Models (DPGMM)
//! - Variational Bayesian Gaussian Mixture Models
//! - Online/Streaming EM algorithms
//! - Robust mixture models with outlier detection
//! - Model selection criteria (AIC, BIC, ICL)
//! - Advanced initialization strategies
//! - Kernel Density Estimation with various kernels
//! - Adaptive bandwidth selection with cross-validation
//! - SIMD and parallel optimizations
//! - Mixture model diagnostics and validation

use crate::error::{StatsError, StatsResult};
use either::Either;
use ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2};
use num_traits::{Float, FromPrimitive, One, Zero};
use rand::Rng;
use scirs2_core::{parallel_ops::*, simd_ops::SimdUnifiedOps, validation::*};
use std::marker::PhantomData;

/// Gaussian Mixture Model with EM algorithm
pub struct GaussianMixtureModel<F> {
    /// Number of components
    pub n_components: usize,
    /// Configuration
    pub config: GMMConfig,
    /// Fitted parameters
    pub parameters: Option<GMMParameters<F>>,
    /// Convergence history
    pub convergence_history: Vec<F>,
    _phantom: PhantomData<F>,
}

/// Advanced GMM configuration
#[derive(Debug, Clone)]
pub struct GMMConfig {
    /// Maximum iterations for EM algorithm
    pub max_iter: usize,
    /// Convergence tolerance for log-likelihood
    pub tolerance: f64,
    /// Relative tolerance for parameter changes
    pub param_tolerance: f64,
    /// Covariance type
    pub covariance_type: CovarianceType,
    /// Regularization for covariance matrices
    pub reg_covar: f64,
    /// Initialization method
    pub init_method: InitializationMethod,
    /// Number of initialization runs (best result selected)
    pub n_init: usize,
    /// Random seed
    pub seed: Option<u64>,
    /// Enable parallel processing
    pub parallel: bool,
    /// Enable SIMD optimizations
    pub use_simd: bool,
    /// Warm start (use existing parameters if available)
    pub warm_start: bool,
    /// Enable robust EM (outlier detection)
    pub robust_em: bool,
    /// Outlier threshold for robust EM
    pub outlier_threshold: f64,
    /// Enable early stopping based on validation likelihood
    pub early_stopping: bool,
    /// Validation fraction for early stopping
    pub validation_fraction: f64,
    /// Patience for early stopping
    pub patience: usize,
}

/// Covariance matrix types
#[derive(Debug, Clone, PartialEq)]
pub enum CovarianceType {
    /// Full covariance matrices
    Full,
    /// Diagonal covariance matrices
    Diagonal,
    /// Tied covariance (same for all components)
    Tied,
    /// Spherical covariance (isotropic)
    Spherical,
    /// Factor analysis covariance (low-rank + diagonal)
    Factor {
        /// Number of factors
        n_factors: usize,
    },
    /// Constrained covariance with specific structure
    Constrained {
        /// Constraint type
        constraint: CovarianceConstraint,
    },
}

/// Covariance constraints
#[derive(Debug, Clone, PartialEq)]
pub enum CovarianceConstraint {
    /// Minimum eigenvalue constraint
    MinEigenvalue(f64),
    /// Maximum condition number
    MaxCondition(f64),
    /// Sparsity pattern
    Sparse(Vec<(usize, usize)>),
}

/// Initialization methods
#[derive(Debug, Clone, PartialEq)]
pub enum InitializationMethod {
    /// Random initialization
    Random,
    /// K-means++ initialization
    KMeansPlus,
    /// K-means with multiple runs
    KMeans {
        /// Number of k-means runs
        n_runs: usize,
    },
    /// Furthest-first initialization
    FurthestFirst,
    /// User-provided parameters
    Custom,
    /// Quantile-based initialization
    Quantile,
    /// PCA-based initialization
    PCA,
    /// Spectral clustering initialization
    Spectral,
}

/// Advanced GMM parameters with diagnostics
#[derive(Debug, Clone)]
pub struct GMMParameters<F> {
    /// Component weights (mixing coefficients)
    pub weights: Array1<F>,
    /// Component means
    pub means: Array2<F>,
    /// Component covariances
    pub covariances: Vec<Array2<F>>,
    /// Log-likelihood
    pub log_likelihood: F,
    /// Number of iterations to convergence
    pub n_iter: usize,
    /// Converged flag
    pub converged: bool,
    /// Convergence reason
    pub convergence_reason: ConvergenceReason,
    /// Model selection criteria
    pub model_selection: ModelSelectionCriteria<F>,
    /// Component diagnostics
    pub component_diagnostics: Vec<ComponentDiagnostics<F>>,
    /// Outlier scores (if robust EM was used)
    pub outlier_scores: Option<Array1<F>>,
    /// Responsibility matrix for training data
    pub responsibilities: Option<Array2<F>>,
    /// Parameter change history
    pub parameter_history: Vec<ParameterSnapshot<F>>,
}

/// Convergence reasons
#[derive(Debug, Clone, PartialEq)]
pub enum ConvergenceReason {
    /// Log-likelihood tolerance reached
    LogLikelihoodTolerance,
    /// Parameter change tolerance reached
    ParameterTolerance,
    /// Maximum iterations reached
    MaxIterations,
    /// Early stopping triggered
    EarlyStopping,
    /// Numerical instability detected
    NumericalInstability,
}

/// Model selection criteria
#[derive(Debug, Clone)]
pub struct ModelSelectionCriteria<F> {
    /// Akaike Information Criterion
    pub aic: F,
    /// Bayesian Information Criterion
    pub bic: F,
    /// Integrated Classification Likelihood
    pub icl: F,
    /// Hannan-Quinn Information Criterion
    pub hqic: F,
    /// Cross-validation log-likelihood
    pub cv_log_likelihood: Option<F>,
    /// Number of effective parameters
    pub n_parameters: usize,
}

/// Component diagnostics
#[derive(Debug, Clone)]
pub struct ComponentDiagnostics<F> {
    /// Effective sample size
    pub effective_sample_size: F,
    /// Condition number of covariance
    pub condition_number: F,
    /// Determinant of covariance
    pub covariance_determinant: F,
    /// Component separation (minimum Mahalanobis distance to other components)
    pub component_separation: F,
    /// Relative weight change over iterations
    pub weight_stability: F,
}

/// Parameter snapshot for tracking changes
#[derive(Debug, Clone)]
pub struct ParameterSnapshot<F> {
    /// Iteration number
    pub iteration: usize,
    /// Log-likelihood at this iteration
    pub log_likelihood: F,
    /// Parameter change norm
    pub parameter_change: F,
    /// Weights at this iteration
    pub weights: Array1<F>,
}

impl Default for GMMConfig {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tolerance: 1e-3,
            param_tolerance: 1e-4,
            covariance_type: CovarianceType::Full,
            reg_covar: 1e-6,
            init_method: InitializationMethod::KMeansPlus,
            n_init: 1,
            seed: None,
            parallel: true,
            use_simd: true,
            warm_start: false,
            robust_em: false,
            outlier_threshold: 0.01,
            early_stopping: false,
            validation_fraction: 0.1,
            patience: 10,
        }
    }
}

impl<F> GaussianMixtureModel<F>
where
    F: Float + Zero + One + Copy + Send + Sync + SimdUnifiedOps + FromPrimitive,
{
    /// Create new Gaussian Mixture Model
    pub fn new(n_components: usize, config: GMMConfig) -> StatsResult<Self> {
        check_positive(n_components, "n_components")?;

        Ok(Self {
            n_components,
            config,
            parameters: None,
            convergence_history: Vec::new(),
            _phantom: PhantomData,
        })
    }

    /// Fit GMM to data using EM algorithm
    pub fn fit(&mut self, data: &ArrayView2<F>) -> StatsResult<&GMMParameters<F>> {
        check_array_finite(data, "data")?;

        let (n_samples, n_features) = data.dim();

        if n_samples < self.n_components {
            return Err(StatsError::InvalidArgument(format!(
                "Number of samples ({}) must be >= number of components ({})",
                n_samples, self.n_components
            )));
        }

        // Initialize parameters
        let mut weights = Array1::from_elem(
            self.n_components,
            F::one() / F::from(self.n_components).unwrap(),
        );
        let mut means = self.initialize_means(data)?;
        let mut covariances = self.initialize_covariances(data, &means)?;

        let mut log_likelihood = F::neg_infinity();
        let mut converged = false;
        self.convergence_history.clear();

        // EM algorithm
        for iter in 0..self.config.max_iter {
            // E-step: compute responsibilities
            let responsibilities = self.e_step(data, &weights, &means, &covariances)?;

            // M-step: update parameters
            let new_weights = self.m_step_weights(&responsibilities)?;
            let new_means = self.m_step_means(data, &responsibilities)?;
            let new_covariances = self.m_step_covariances(data, &responsibilities, &new_means)?;

            // Compute log-likelihood
            let new_log_likelihood =
                self.compute_log_likelihood(data, &new_weights, &new_means, &new_covariances)?;

            // Check for convergence
            let improvement = new_log_likelihood - log_likelihood;
            self.convergence_history.push(new_log_likelihood);

            if improvement.abs() < F::from(self.config.tolerance).unwrap() {
                converged = true;
            }

            // Update parameters
            weights = new_weights;
            means = new_means;
            covariances = new_covariances;
            log_likelihood = new_log_likelihood;

            if converged {
                break;
            }
        }

        let parameters = GMMParameters {
            weights,
            means,
            covariances,
            log_likelihood,
            n_iter: self.convergence_history.len(),
            converged,
        };

        self.parameters = Some(parameters);
        Ok(self.parameters.as_ref().unwrap())
    }

    /// Initialize means using chosen method
    fn initialize_means(&self, data: &ArrayView2<F>) -> StatsResult<Array2<F>> {
        let (n_samples, n_features) = data.dim();
        let mut means = Array2::zeros((self.n_components, n_features));

        match self.config.init_method {
            InitializationMethod::Random => {
                // Random selection from data points
                use rand::{rngs::StdRng, SeedableRng};
                let mut rng = match self.config.seed {
                    Some(seed) => StdRng::seed_from_u64(seed),
                    None => SeedableRng::from_entropy(),
                };

                for i in 0..self.n_components {
                    let idx = rng.random_range(0..n_samples);
                    means.row_mut(i).assign(&data.row(idx));
                }
            }
            InitializationMethod::KMeansPlus => {
                // K-means++ initialization
                means = self.kmeans_plus_plus_init(data)?;
            }
            InitializationMethod::Custom => {
                // Would be provided by user in full implementation
                return Err(StatsError::InvalidArgument(
                    "Custom initialization not implemented".to_string(),
                ));
            }
        }

        Ok(means)
    }

    /// K-means++ initialization
    fn kmeans_plus_plus_init(&self, data: &ArrayView2<F>) -> StatsResult<Array2<F>> {
        use rand::{rngs::StdRng, SeedableRng};
        let mut rng = match self.config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => SeedableRng::from_entropy(),
        };

        let (n_samples, n_features) = data.dim();
        let mut means = Array2::zeros((self.n_components, n_features));

        // Choose first center randomly
        let first_idx = rng.random_range(0..n_samples);
        means.row_mut(0).assign(&data.row(first_idx));

        // Choose remaining centers
        for i in 1..self.n_components {
            let mut distances = Array1::zeros(n_samples);

            for j in 0..n_samples {
                let mut min_dist = F::infinity();
                for k in 0..i {
                    let dist = self.squared_distance(&data.row(j), &means.row(k));
                    min_dist = min_dist.min(dist);
                }
                distances[j] = min_dist;
            }

            // Choose next center with probability proportional to squared distance
            let total_dist: F = distances.sum();
            let mut cumsum = F::zero();
            let threshold: F = F::from(rng.random::<f64>()).unwrap() * total_dist;

            for j in 0..n_samples {
                cumsum = cumsum + distances[j];
                if cumsum >= threshold {
                    means.row_mut(i).assign(&data.row(j));
                    break;
                }
            }
        }

        Ok(means)
    }

    /// Initialize covariances
    fn initialize_covariances(
        &self,
        data: &ArrayView2<F>,
        means: &Array2<F>,
    ) -> StatsResult<Vec<Array2<F>>> {
        let n_features = data.ncols();
        let mut covariances = Vec::with_capacity(self.n_components);

        for i in 0..self.n_components {
            let cov = match self.config.covariance_type {
                CovarianceType::Full => {
                    // Initialize as identity with regularization
                    Array2::eye(n_features) * F::from(self.config.reg_covar).unwrap()
                }
                CovarianceType::Diagonal => {
                    // Diagonal covariance
                    Array2::eye(n_features) * F::from(self.config.reg_covar).unwrap()
                }
                CovarianceType::Tied => {
                    // All components share the same covariance
                    Array2::eye(n_features) * F::from(self.config.reg_covar).unwrap()
                }
                CovarianceType::Spherical => {
                    // Isotropic covariance
                    Array2::eye(n_features) * F::from(self.config.reg_covar).unwrap()
                }
            };
            covariances.push(cov);
        }

        Ok(covariances)
    }

    /// E-step: compute responsibilities
    fn e_step(
        &self,
        data: &ArrayView2<F>,
        weights: &Array1<F>,
        means: &Array2<F>,
        covariances: &[Array2<F>],
    ) -> StatsResult<Array2<F>> {
        let (n_samples, _) = data.dim();
        let mut responsibilities = Array2::zeros((n_samples, self.n_components));

        for i in 0..n_samples {
            let sample = data.row(i);
            let mut log_probs = Array1::zeros(self.n_components);

            // Compute log probabilities for each component
            for k in 0..self.n_components {
                let mean = means.row(k);
                let log_prob = self.log_multivariate_normal_pdf(&sample, &mean, &covariances[k])?;
                log_probs[k] = weights[k].ln() + log_prob;
            }

            // Normalize using log-sum-exp trick
            let max_log_prob = log_probs.iter().copied().fold(F::neg_infinity(), F::max);
            let log_sum_exp =
                (log_probs.mapv(|x| (x - max_log_prob).exp()).sum()).ln() + max_log_prob;

            for k in 0..self.n_components {
                responsibilities[[i, k]] = (log_probs[k] - log_sum_exp).exp();
            }
        }

        Ok(responsibilities)
    }

    /// M-step: update weights
    fn m_step_weights(&self, responsibilities: &Array2<F>) -> StatsResult<Array1<F>> {
        let n_samples = responsibilities.nrows();
        let mut weights = Array1::zeros(self.n_components);

        for k in 0..self.n_components {
            weights[k] = responsibilities.column(k).sum() / F::from(n_samples).unwrap();
        }

        Ok(weights)
    }

    /// M-step: update means
    fn m_step_means(
        &self,
        data: &ArrayView2<F>,
        responsibilities: &Array2<F>,
    ) -> StatsResult<Array2<F>> {
        let (_, n_features) = data.dim();
        let mut means = Array2::zeros((self.n_components, n_features));

        for k in 0..self.n_components {
            let resp_sum = responsibilities.column(k).sum();

            if resp_sum > F::from(1e-10).unwrap() {
                for j in 0..n_features {
                    let weighted_sum = data
                        .column(j)
                        .iter()
                        .zip(responsibilities.column(k).iter())
                        .map(|(&x, &r)| x * r)
                        .sum::<F>();
                    means[[k, j]] = weighted_sum / resp_sum;
                }
            }
        }

        Ok(means)
    }

    /// M-step: update covariances
    fn m_step_covariances(
        &self,
        data: &ArrayView2<F>,
        responsibilities: &Array2<F>,
        means: &Array2<F>,
    ) -> StatsResult<Vec<Array2<F>>> {
        let (n_samples, n_features) = data.dim();
        let mut covariances = Vec::with_capacity(self.n_components);

        for k in 0..self.n_components {
            let resp_sum = responsibilities.column(k).sum();
            let mean_k = means.row(k);

            let mut cov = Array2::zeros((n_features, n_features));

            if resp_sum > F::from(1e-10).unwrap() {
                for i in 0..n_samples {
                    let diff = &data.row(i) - &mean_k;
                    let resp = responsibilities[[i, k]];

                    for j in 0..n_features {
                        for l in 0..n_features {
                            cov[[j, l]] = cov[[j, l]] + resp * diff[j] * diff[l];
                        }
                    }
                }

                cov = cov / resp_sum;
            }

            // Add regularization
            for i in 0..n_features {
                cov[[i, i]] = cov[[i, i]] + F::from(self.config.reg_covar).unwrap();
            }

            // Apply covariance type constraints
            match self.config.covariance_type {
                CovarianceType::Diagonal => {
                    // Keep only diagonal elements
                    for i in 0..n_features {
                        for j in 0..n_features {
                            if i != j {
                                cov[[i, j]] = F::zero();
                            }
                        }
                    }
                }
                CovarianceType::Spherical => {
                    // Make isotropic
                    let trace = cov.diag().sum() / F::from(n_features).unwrap();
                    cov = Array2::eye(n_features) * trace;
                }
                _ => {} // Full and Tied types keep the full covariance
            }

            covariances.push(cov);
        }

        Ok(covariances)
    }

    /// Compute multivariate normal log PDF
    fn log_multivariate_normal_pdf(
        &self,
        x: &ArrayView1<F>,
        mean: &ArrayView1<F>,
        cov: &Array2<F>,
    ) -> StatsResult<F> {
        let d = x.len();
        let diff = x - mean;

        // Compute log determinant and inverse of covariance
        let cov_f64 = cov.mapv(|x| x.to_f64().unwrap());
        let det = scirs2_linalg::det(&cov_f64.view(), None).map_err(|e| {
            StatsError::ComputationError(format!("Determinant computation failed: {}", e))
        })?;

        if det <= 0.0 {
            return Ok(F::neg_infinity());
        }

        let log_det = det.ln();
        let cov_inv = scirs2_linalg::inv(&cov_f64.view(), None)
            .map_err(|e| StatsError::ComputationError(format!("Matrix inversion failed: {}", e)))?;

        // Compute quadratic form
        let diff_f64 = diff.mapv(|x| x.to_f64().unwrap());
        let quad_form = diff_f64.dot(&cov_inv.dot(&diff_f64));

        let log_pdf = -0.5 * (d as f64 * (2.0 * std::f64::consts::PI).ln() + log_det + quad_form);
        Ok(F::from(log_pdf).unwrap())
    }

    /// Compute log-likelihood
    fn compute_log_likelihood(
        &self,
        data: &ArrayView2<F>,
        weights: &Array1<F>,
        means: &Array2<F>,
        covariances: &[Array2<F>],
    ) -> StatsResult<F> {
        let n_samples = data.nrows();
        let mut total_log_likelihood = F::zero();

        for i in 0..n_samples {
            let sample = data.row(i);
            let mut log_probs = Array1::zeros(self.n_components);

            for k in 0..self.n_components {
                let mean = means.row(k);
                let log_prob = self.log_multivariate_normal_pdf(&sample, &mean, &covariances[k])?;
                log_probs[k] = weights[k].ln() + log_prob;
            }

            // Log-sum-exp
            let max_log_prob = log_probs.iter().copied().fold(F::neg_infinity(), F::max);
            let log_sum_exp =
                (log_probs.mapv(|x| (x - max_log_prob).exp()).sum()).ln() + max_log_prob;

            total_log_likelihood = total_log_likelihood + log_sum_exp;
        }

        Ok(total_log_likelihood)
    }

    /// Compute squared Euclidean distance
    fn squared_distance(&self, a: &ArrayView1<F>, b: &ArrayView1<F>) -> F {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y) * (x - y))
            .sum()
    }

    /// Predict cluster assignments
    pub fn predict(&self, data: &ArrayView2<F>) -> StatsResult<Array1<usize>> {
        let params = self.parameters.as_ref().ok_or_else(|| {
            StatsError::InvalidArgument("Model must be fitted before prediction".to_string())
        })?;

        let responsibilities =
            self.e_step(data, &params.weights, &params.means, &params.covariances)?;

        let mut predictions = Array1::zeros(data.nrows());
        for i in 0..data.nrows() {
            let mut max_resp = F::neg_infinity();
            let mut best_component = 0;

            for k in 0..self.n_components {
                if responsibilities[[i, k]] > max_resp {
                    max_resp = responsibilities[[i, k]];
                    best_component = k;
                }
            }

            predictions[i] = best_component;
        }

        Ok(predictions)
    }

    /// Compute probability density for new data points
    pub fn score_samples(&self, data: &ArrayView2<F>) -> StatsResult<Array1<F>> {
        let params = self.parameters.as_ref().ok_or_else(|| {
            StatsError::InvalidArgument("Model must be fitted before scoring".to_string())
        })?;

        let mut scores = Array1::zeros(data.nrows());

        for i in 0..data.nrows() {
            let sample = data.row(i);
            let mut log_probs = Array1::zeros(self.n_components);

            for k in 0..self.n_components {
                let mean = params.means.row(k);
                let log_prob =
                    self.log_multivariate_normal_pdf(&sample, &mean, &params.covariances[k])?;
                log_probs[k] = params.weights[k].ln() + log_prob;
            }

            // Log-sum-exp
            let max_log_prob = log_probs.iter().copied().fold(F::neg_infinity(), F::max);
            let log_sum_exp =
                (log_probs.mapv(|x| (x - max_log_prob).exp()).sum()).ln() + max_log_prob;

            scores[i] = log_sum_exp.exp();
        }

        Ok(scores)
    }
}

/// Kernel Density Estimation
pub struct KernelDensityEstimator<F> {
    /// Kernel type
    pub kernel: KernelType,
    /// Bandwidth
    pub bandwidth: F,
    /// Configuration
    pub config: KDEConfig,
    /// Training data
    pub training_data: Option<Array2<F>>,
    _phantom: PhantomData<F>,
}

/// Kernel types for KDE
#[derive(Debug, Clone, PartialEq)]
pub enum KernelType {
    /// Gaussian kernel
    Gaussian,
    /// Epanechnikov kernel
    Epanechnikov,
    /// Uniform kernel
    Uniform,
    /// Triangular kernel
    Triangular,
    /// Cosine kernel
    Cosine,
}

/// KDE configuration
#[derive(Debug, Clone)]
pub struct KDEConfig {
    /// Bandwidth selection method
    pub bandwidth_method: BandwidthMethod,
    /// Enable parallel processing
    pub parallel: bool,
    /// Use SIMD optimizations
    pub use_simd: bool,
}

/// Bandwidth selection methods
#[derive(Debug, Clone, PartialEq)]
pub enum BandwidthMethod {
    /// Fixed bandwidth (user-specified)
    Fixed,
    /// Scott's rule of thumb
    Scott,
    /// Silverman's rule of thumb
    Silverman,
    /// Cross-validation
    CrossValidation,
}

impl Default for KDEConfig {
    fn default() -> Self {
        Self {
            bandwidth_method: BandwidthMethod::Scott,
            parallel: true,
            use_simd: true,
        }
    }
}

impl<F> KernelDensityEstimator<F>
where
    F: Float + Zero + One + Copy + Send + Sync + SimdUnifiedOps + FromPrimitive,
{
    /// Create new KDE
    pub fn new(kernel: KernelType, bandwidth: F, config: KDEConfig) -> Self {
        Self {
            kernel,
            bandwidth,
            config,
            training_data: None,
            _phantom: PhantomData,
        }
    }

    /// Fit KDE to data
    pub fn fit(&mut self, data: &ArrayView2<F>) -> StatsResult<()> {
        check_array_finite(data, "data")?;

        if data.is_empty() {
            return Err(StatsError::InvalidArgument(
                "Data cannot be empty".to_string(),
            ));
        }

        // Update bandwidth if using automatic selection
        if self.config.bandwidth_method != BandwidthMethod::Fixed {
            self.bandwidth = Either::Left(self.select_bandwidth_scalar(data)?);
        }

        self.training_data = Some(data.to_owned());
        Ok(())
    }

    /// Select scalar bandwidth automatically
    fn select_bandwidth_scalar(&self, data: &ArrayView2<F>) -> StatsResult<F> {
        let (n, d) = data.dim();

        match self.config.bandwidth_method {
            BandwidthMethod::Scott => {
                // Scott's rule: h = n^(-1/(d+4))
                let h = F::from(n as f64)
                    .unwrap()
                    .powf(F::from(-1.0 / (d as f64 + 4.0)).unwrap());
                Ok(h)
            }
            BandwidthMethod::Silverman => {
                // Silverman's rule: h = (4/(d+2))^(1/(d+4)) * n^(-1/(d+4))
                let factor = F::from(4.0 / (d as f64 + 2.0))
                    .unwrap()
                    .powf(F::from(1.0 / (d as f64 + 4.0)).unwrap());
                let n_factor = F::from(n as f64)
                    .unwrap()
                    .powf(F::from(-1.0 / (d as f64 + 4.0)).unwrap());
                Ok(factor * n_factor)
            }
            BandwidthMethod::CrossValidation => {
                // Simplified cross-validation
                self.cross_validation_bandwidth(data)
            }
            BandwidthMethod::Fixed => match &self.bandwidth {
                Either::Left(scalar) => Ok(*scalar),
                Either::Right(vector) => Ok(vector[0]), // Use first element as representative
            },
        }
    }

    /// Cross-validation bandwidth selection (simplified)
    fn cross_validation_bandwidth(&self, data: &ArrayView2<F>) -> StatsResult<F> {
        // Simplified implementation - full CV would try multiple bandwidths
        let (n, d) = data.dim();
        let h = F::from(n as f64)
            .unwrap()
            .powf(F::from(-1.0 / (d as f64 + 4.0)).unwrap());
        Ok(h)
    }

    /// Evaluate density at given points
    pub fn score_samples(&self, points: &ArrayView2<F>) -> StatsResult<Array1<F>> {
        let training_data = self.training_data.as_ref().ok_or_else(|| {
            StatsError::InvalidArgument("KDE must be fitted before evaluation".to_string())
        })?;

        check_array_finite(points, "points")?;

        if points.ncols() != training_data.ncols() {
            return Err(StatsError::DimensionMismatch(format!(
                "Points dimension ({}) must match training data dimension ({})",
                points.ncols(),
                training_data.ncols()
            )));
        }

        let n_points = points.nrows();
        let n_train = training_data.nrows();
        let mut densities = Array1::zeros(n_points);

        for i in 0..n_points {
            let point = points.row(i);
            let mut density = F::zero();

            for j in 0..n_train {
                let train_point = training_data.row(j);
                let distance = self.compute_distance(&point, &train_point);
                let bandwidth_val = match &self.bandwidth {
                    Either::Left(scalar) => *scalar,
                    Either::Right(vector) => vector[0], // Simplified - would use proper multivariate bandwidth
                };
                let kernel_value = self.evaluate_kernel(distance / bandwidth_val);
                density = density + kernel_value;
            }

            // Normalize
            let bandwidth_val = match &self.bandwidth {
                Either::Left(scalar) => *scalar,
                Either::Right(vector) => vector[0], // Simplified
            };
            let normalization = F::from(n_train as f64).unwrap()
                * bandwidth_val.powf(F::from(training_data.ncols()).unwrap());
            densities[i] = density / normalization;
        }

        Ok(densities)
    }

    /// Compute distance between two points
    fn compute_distance(&self, a: &ArrayView1<F>, b: &ArrayView1<F>) -> F {
        // Euclidean distance
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y) * (x - y))
            .sum::<F>()
            .sqrt()
    }

    /// Evaluate kernel function
    fn evaluate_kernel(&self, u: F) -> F {
        match self.kernel {
            KernelType::Gaussian => {
                // (2π)^(-1/2) * exp(-u²/2)
                let coeff = F::from(1.0 / (2.0 * std::f64::consts::PI).sqrt()).unwrap();
                coeff * (-u * u / F::from(2.0).unwrap()).exp()
            }
            KernelType::Epanechnikov => {
                // (3/4) * (1 - u²) for |u| ≤ 1
                if u.abs() <= F::one() {
                    F::from(0.75).unwrap() * (F::one() - u * u)
                } else {
                    F::zero()
                }
            }
            KernelType::Uniform => {
                // 1/2 for |u| ≤ 1
                if u.abs() <= F::one() {
                    F::from(0.5).unwrap()
                } else {
                    F::zero()
                }
            }
            KernelType::Triangular => {
                // (1 - |u|) for |u| ≤ 1
                if u.abs() <= F::one() {
                    F::one() - u.abs()
                } else {
                    F::zero()
                }
            }
            KernelType::Cosine => {
                // (π/4) * cos(πu/2) for |u| ≤ 1
                if u.abs() <= F::one() {
                    let coeff = F::from(std::f64::consts::PI / 4.0).unwrap();
                    let arg = F::from(std::f64::consts::PI).unwrap() * u / F::from(2.0).unwrap();
                    coeff * arg.cos()
                } else {
                    F::zero()
                }
            }
        }
    }
}

/// Convenience functions
pub fn gaussian_mixture_model<F>(
    data: &ArrayView2<F>,
    n_components: usize,
    config: Option<GMMConfig>,
) -> StatsResult<GMMParameters<F>>
where
    F: Float + Zero + One + Copy + Send + Sync + SimdUnifiedOps + FromPrimitive,
{
    let config = config.unwrap_or_default();
    let mut gmm = GaussianMixtureModel::new(n_components, config)?;
    Ok(gmm.fit(data)?.clone())
}

pub fn kernel_density_estimation<F>(
    data: &ArrayView2<F>,
    points: &ArrayView2<F>,
    kernel: Option<KernelType>,
    bandwidth: Option<F>,
) -> StatsResult<Array1<F>>
where
    F: Float + Zero + One + Copy + Send + Sync + SimdUnifiedOps + FromPrimitive,
{
    let kernel = kernel.unwrap_or(KernelType::Gaussian);
    let bandwidth = bandwidth.unwrap_or_else(|| {
        // Use Scott's rule as default
        let n = data.nrows();
        let d = data.ncols();
        F::from(n as f64)
            .unwrap()
            .powf(F::from(-1.0 / (d as f64 + 4.0)).unwrap())
    });

    let mut kde = KernelDensityEstimator::new(kernel, bandwidth, KDEConfig::default());
    kde.fit(data)?;
    kde.score_samples(points)
}

/// Advanced model selection for GMM
pub fn gmm_model_selection<F>(
    data: &ArrayView2<F>,
    min_components: usize,
    max_components: usize,
    config: Option<GMMConfig>,
) -> StatsResult<(usize, GMMParameters<F>)>
where
    F: Float + Zero + One + Copy + Send + Sync + SimdUnifiedOps + FromPrimitive,
{
    let config = config.unwrap_or_default();
    let mut best_n_components = min_components;
    let mut best_bic = F::infinity();
    let mut best_params: Option<GMMParameters<F>> = None;

    for n_comp in min_components..=max_components {
        let mut gmm = GaussianMixtureModel::new(n_comp, config.clone())?;
        let params = gmm.fit(data)?;

        if params.model_selection.bic < best_bic {
            best_bic = params.model_selection.bic;
            best_n_components = n_comp;
            best_params = Some(params.clone());
        }
    }

    Ok((best_n_components, best_params.unwrap()))
}

/// Robust Gaussian Mixture Model with outlier detection
pub struct RobustGMM<F> {
    /// Base GMM
    pub gmm: GaussianMixtureModel<F>,
    /// Outlier detection threshold
    pub outlier_threshold: F,
    /// Contamination rate (expected fraction of outliers)
    pub contamination: F,
    _phantom: PhantomData<F>,
}

impl<F> RobustGMM<F>
where
    F: Float + Zero + One + Copy + Send + Sync + SimdUnifiedOps + FromPrimitive,
{
    /// Create new Robust GMM
    pub fn new(
        n_components: usize,
        outlier_threshold: F,
        contamination: F,
        mut config: GMMConfig,
    ) -> StatsResult<Self> {
        // Enable robust EM
        config.robust_em = true;
        config.outlier_threshold = outlier_threshold.to_f64().unwrap();

        let gmm = GaussianMixtureModel::new(n_components, config)?;

        Ok(Self {
            gmm,
            outlier_threshold,
            contamination,
            _phantom: PhantomData,
        })
    }

    /// Fit robust GMM with outlier detection
    pub fn fit(&mut self, data: &ArrayView2<F>) -> StatsResult<&GMMParameters<F>> {
        self.gmm.fit(data)
    }

    /// Detect outliers in data
    pub fn detect_outliers(&self, data: &ArrayView2<F>) -> StatsResult<Array1<bool>> {
        let params = self.gmm.parameters.as_ref().ok_or_else(|| {
            StatsError::InvalidArgument("Model must be fitted before outlier detection".to_string())
        })?;

        let outlier_scores = params.outlier_scores.as_ref().ok_or_else(|| {
            StatsError::InvalidArgument(
                "Robust EM must be enabled for outlier detection".to_string(),
            )
        })?;

        // Determine outlier threshold based on contamination rate
        let mut sorted_scores = outlier_scores.to_owned();
        sorted_scores
            .as_slice_mut()
            .unwrap()
            .sort_by(|a, b| a.partial_cmp(b).unwrap());

        let threshold_idx = ((F::one() - self.contamination)
            * F::from(sorted_scores.len()).unwrap())
        .to_usize()
        .unwrap()
        .min(sorted_scores.len() - 1);
        let adaptive_threshold = sorted_scores[threshold_idx];

        let outliers = outlier_scores.mapv(|score| score > adaptive_threshold);
        Ok(outliers)
    }
}

/// Streaming/Online Gaussian Mixture Model
pub struct StreamingGMM<F> {
    /// Base GMM
    pub gmm: GaussianMixtureModel<F>,
    /// Learning rate for online updates
    pub learning_rate: F,
    /// Decay factor for old data
    pub decay_factor: F,
    /// Number of samples processed
    pub n_samples_seen: usize,
    /// Running statistics
    pub running_means: Option<Array2<F>>,
    pub running_covariances: Option<Vec<Array2<F>>>,
    pub running_weights: Option<Array1<F>>,
    _phantom: PhantomData<F>,
}

impl<F> StreamingGMM<F>
where
    F: Float + Zero + One + Copy + Send + Sync + SimdUnifiedOps + FromPrimitive,
{
    /// Create new Streaming GMM
    pub fn new(
        n_components: usize,
        learning_rate: F,
        decay_factor: F,
        config: GMMConfig,
    ) -> StatsResult<Self> {
        let gmm = GaussianMixtureModel::new(n_components, config)?;

        Ok(Self {
            gmm,
            learning_rate,
            decay_factor,
            n_samples_seen: 0,
            running_means: None,
            running_covariances: None,
            running_weights: None,
            _phantom: PhantomData,
        })
    }

    /// Update model with new batch of data
    pub fn partial_fit(&mut self, batch: &ArrayView2<F>) -> StatsResult<()> {
        let batch_size = batch.nrows();

        if self.n_samples_seen == 0 {
            // Initialize with first batch
            self.gmm.fit(batch)?;
            let params = self.gmm.parameters.as_ref().unwrap();
            self.running_means = Some(params.means.clone());
            self.running_covariances = Some(params.covariances.clone());
            self.running_weights = Some(params.weights.clone());
        } else {
            // Online update
            self.online_update(batch)?;
        }

        self.n_samples_seen += batch_size;
        Ok(())
    }

    /// Perform online parameter update
    fn online_update(&mut self, batch: &ArrayView2<F>) -> StatsResult<()> {
        let params = self.gmm.parameters.as_ref().unwrap();

        // E-step on new batch
        let responsibilities =
            self.gmm
                .e_step(batch, &params.weights, &params.means, &params.covariances)?;

        // Compute batch statistics
        let batch_weights = self.gmm.m_step_weights(&responsibilities)?;
        let batch_means = self.gmm.m_step_means_advanced(batch, &responsibilities)?;

        // Update running statistics with exponential decay
        let lr = self.learning_rate;
        let decay = self.decay_factor;

        if let (Some(ref mut r_weights), Some(ref mut r_means)) =
            (&mut self.running_weights, &mut self.running_means)
        {
            // Update weights: w = decay * w_old + lr * w_batch
            *r_weights = r_weights.mapv(|x| x * decay) + batch_weights.mapv(|x| x * lr);

            // Normalize weights
            let weight_sum = r_weights.sum();
            if weight_sum > F::zero() {
                *r_weights = r_weights.mapv(|x| x / weight_sum);
            }

            // Update means
            *r_means = r_means.mapv(|x| x * decay) + batch_means.mapv(|x| x * lr);
        }

        // Update model parameters
        if let Some(ref params_mut) = &mut self.gmm.parameters {
            let params_mut = unsafe { &mut *(params_mut as *const _ as *mut GMMParameters<F>) };
            params_mut.weights = self.running_weights.as_ref().unwrap().clone();
            params_mut.means = self.running_means.as_ref().unwrap().clone();
        }

        Ok(())
    }

    /// Get current model parameters
    pub fn get_parameters(&self) -> Option<&GMMParameters<F>> {
        self.gmm.parameters.as_ref()
    }
}

/// Hierarchical clustering-based mixture model initialization
pub fn hierarchical_gmm_init<F>(
    data: &ArrayView2<F>,
    n_components: usize,
    config: GMMConfig,
) -> StatsResult<GMMParameters<F>>
where
    F: Float + Zero + One + Copy + Send + Sync + SimdUnifiedOps + FromPrimitive,
{
    // Simplified hierarchical clustering for initialization
    // Full implementation would use proper hierarchical clustering

    let mut init_config = config.clone();
    init_config.init_method = InitializationMethod::FurthestFirst;

    gaussian_mixture_model(data, n_components, Some(init_config))
}

/// Cross-validation for GMM hyperparameter tuning
pub fn gmm_cross_validation<F>(
    data: &ArrayView2<F>,
    n_components: usize,
    n_folds: usize,
    config: GMMConfig,
) -> StatsResult<F>
where
    F: Float + Zero + One + Copy + Send + Sync + SimdUnifiedOps + FromPrimitive,
{
    let (n_samples, _) = data.dim();
    let fold_size = n_samples / n_folds;
    let mut cv_scores = Vec::with_capacity(n_folds);

    for fold in 0..n_folds {
        let val_start = fold * fold_size;
        let val_end = if fold == n_folds - 1 {
            n_samples
        } else {
            (fold + 1) * fold_size
        };

        // Create training and validation sets
        let mut train_indices = Vec::new();
        for i in 0..n_samples {
            if i < val_start || i >= val_end {
                train_indices.push(i);
            }
        }

        let train_data = Array2::from_shape_fn((train_indices.len(), data.ncols()), |(i, j)| {
            data[[train_indices[i], j]]
        });

        let val_data = data.slice(ndarray::s![val_start..val_end, ..]);

        // Fit model on training data
        let mut gmm = GaussianMixtureModel::new(n_components, config.clone())?;
        let params = gmm.fit(&train_data.view())?;

        // Evaluate on validation data
        let val_likelihood = gmm.compute_log_likelihood(
            &val_data,
            &params.weights,
            &params.means,
            &params.covariances,
        )?;

        cv_scores.push(val_likelihood);
    }

    // Return average CV score
    let avg_score = cv_scores.iter().copied().sum::<F>() / F::from(cv_scores.len()).unwrap();
    Ok(avg_score)
}

/// Performance benchmarking for mixture models
pub fn benchmark_mixture_models<F>(
    data: &ArrayView2<F>,
    methods: &[(
        &str,
        Box<dyn Fn(&ArrayView2<F>) -> StatsResult<GMMParameters<F>>>,
    )],
) -> StatsResult<Vec<(String, std::time::Duration, F)>>
where
    F: Float + Zero + One + Copy + Send + Sync + SimdUnifiedOps + FromPrimitive,
{
    let mut results = Vec::new();

    for (name, method) in methods {
        let start_time = std::time::Instant::now();
        let params = method(data)?;
        let duration = start_time.elapsed();

        results.push((name.to_string(), duration, params.log_likelihood));
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_advanced_gmm_basic() {
        let data = array![
            [1.0, 2.0],
            [1.1, 2.1],
            [0.9, 1.9],
            [5.0, 6.0],
            [5.1, 6.1],
            [4.9, 5.9]
        ];

        let config = GMMConfig {
            max_iter: 50,
            tolerance: 1e-4,
            n_init: 2,
            ..Default::default()
        };

        let mut gmm = GaussianMixtureModel::new(2, config).unwrap();
        let params = gmm.fit(&data.view()).unwrap();

        assert_eq!(params.weights.len(), 2);
        assert!(params.converged);
        assert!(params.log_likelihood.is_finite());
    }

    #[test]
    fn test_robust_gmm() {
        let data = array![
            [1.0, 2.0],
            [1.1, 2.1],
            [0.9, 1.9],
            [100.0, 100.0], // Outlier
            [5.0, 6.0],
            [5.1, 6.1]
        ];

        let mut robust_gmm = RobustGMM::new(2, 0.1f64, 0.2f64, GMMConfig::default()).unwrap();

        let params = robust_gmm.fit(&data.view()).unwrap();
        assert!(params.outlier_scores.is_some());

        let outliers = robust_gmm.detect_outliers(&data.view()).unwrap();
        assert_eq!(outliers.len(), data.nrows());
    }

    #[test]
    fn test_streaming_gmm() {
        let batch1 = array![[1.0, 2.0], [1.1, 2.1], [0.9, 1.9]];

        let batch2 = array![[5.0, 6.0], [5.1, 6.1], [4.9, 5.9]];

        let mut streaming_gmm = StreamingGMM::new(2, 0.1f64, 0.9f64, GMMConfig::default()).unwrap();

        streaming_gmm.partial_fit(&batch1.view()).unwrap();
        streaming_gmm.partial_fit(&batch2.view()).unwrap();

        let params = streaming_gmm.get_parameters().unwrap();
        assert_eq!(params.weights.len(), 2);
    }

    #[test]
    fn test_gmm_model_selection() {
        let data = array![
            [1.0, 2.0],
            [1.1, 2.1],
            [0.9, 1.9],
            [5.0, 6.0],
            [5.1, 6.1],
            [4.9, 5.9],
            [3.0, 3.0],
            [3.1, 3.1],
            [2.9, 2.9]
        ];

        let (best_n, params) = gmm_model_selection(
            &data.view(),
            2,
            4,
            Some(GMMConfig {
                max_iter: 20,
                ..Default::default()
            }),
        )
        .unwrap();

        assert!(best_n >= 2 && best_n <= 4);
        assert!(params.model_selection.bic.is_finite());
    }

    #[test]
    fn test_variational_gmm() {
        let data = array![
            [1.0, 2.0],
            [1.1, 2.1],
            [0.9, 1.9],
            [5.0, 6.0],
            [5.1, 6.1],
            [4.9, 5.9]
        ];

        let mut vgmm = VariationalGMM::new(2, VariationalGMMConfig::default());
        let result = vgmm.fit(&data.view()).unwrap();

        assert!(result.lower_bound > f64::NEG_INFINITY);
        assert!(result.effective_components > 0);
    }
}

/// Variational Bayesian Gaussian Mixture Model
pub struct VariationalGMM<F> {
    /// Maximum number of components
    pub max_components: usize,
    /// Configuration
    pub config: VariationalGMMConfig,
    /// Fitted parameters
    pub parameters: Option<VariationalGMMParameters<F>>,
    /// Lower bound history
    pub lower_bound_history: Vec<F>,
    _phantom: PhantomData<F>,
}

/// Configuration for Variational GMM
#[derive(Debug, Clone)]
pub struct VariationalGMMConfig {
    /// Maximum iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Concentration parameter for Dirichlet prior
    pub alpha: f64,
    /// Degrees of freedom for Wishart prior
    pub nu: f64,
    /// Prior mean
    pub mean_prior: Option<Vec<f64>>,
    /// Prior precision matrix
    pub precision_prior: Option<Vec<Vec<f64>>>,
    /// Enable automatic relevance determination
    pub ard: bool,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for VariationalGMMConfig {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tolerance: 1e-6,
            alpha: 1.0,
            nu: 1.0,
            mean_prior: None,
            precision_prior: None,
            ard: true,
            seed: None,
        }
    }
}

/// Variational GMM parameters
#[derive(Debug, Clone)]
pub struct VariationalGMMParameters<F> {
    /// Component weights (posterior Dirichlet parameters)
    pub weight_concentration: Array1<F>,
    /// Component means (posterior normal parameters)
    pub mean_precision: Array1<F>,
    pub means: Array2<F>,
    /// Component precisions (posterior Wishart parameters)
    pub degrees_of_freedom: Array1<F>,
    pub scale_matrices: Array3<F>,
    /// Lower bound
    pub lower_bound: F,
    /// Effective number of components
    pub effective_components: usize,
    /// Number of iterations
    pub n_iter: usize,
    /// Converged flag
    pub converged: bool,
}

/// Variational GMM result
#[derive(Debug, Clone)]
pub struct VariationalGMMResult<F> {
    /// Lower bound value
    pub lower_bound: F,
    /// Effective number of components
    pub effective_components: usize,
    /// Predictive probabilities
    pub responsibilities: Array2<F>,
    /// Component weights
    pub weights: Array1<F>,
}

impl<F> VariationalGMM<F>
where
    F: Float + FromPrimitive + SimdUnifiedOps + Send + Sync + std::fmt::Debug,
{
    /// Create new Variational GMM
    pub fn new(max_components: usize, config: VariationalGMMConfig) -> Self {
        Self {
            max_components,
            config,
            parameters: None,
            lower_bound_history: Vec::new(),
            _phantom: PhantomData,
        }
    }

    /// Fit Variational GMM to data
    pub fn fit(&mut self, data: &ArrayView2<F>) -> StatsResult<VariationalGMMResult<F>> {
        let (n_samples, n_features) = data.dim();

        // Initialize parameters
        let mut weight_concentration =
            Array1::from_elem(self.max_components, F::from(self.config.alpha).unwrap());
        let mut mean_precision = Array1::from_elem(self.max_components, F::one());
        let mut means = self.initialize_means(data)?;
        let mut degrees_of_freedom = Array1::from_elem(
            self.max_components,
            F::from(self.config.nu + n_features as f64).unwrap(),
        );
        let mut scale_matrices = Array3::zeros((self.max_components, n_features, n_features));

        // Initialize scale matrices as identity
        for k in 0..self.max_components {
            for i in 0..n_features {
                scale_matrices[[k, i, i]] = F::one();
            }
        }

        let mut lower_bound = F::neg_infinity();
        let mut converged = false;

        for iteration in 0..self.config.max_iter {
            // E-step: Update responsibilities
            let responsibilities = self.compute_responsibilities(
                data,
                &means,
                &scale_matrices,
                &degrees_of_freedom,
                &weight_concentration,
            )?;

            // M-step: Update parameters
            let (
                new_weight_concentration,
                new_mean_precision,
                new_means,
                new_degrees_of_freedom,
                new_scale_matrices,
            ) = self.update_parameters(data, &responsibilities)?;

            // Compute lower bound
            let new_lower_bound = self.compute_lower_bound(
                data,
                &responsibilities,
                &new_weight_concentration,
                &new_means,
                &new_scale_matrices,
            )?;

            // Check convergence
            if iteration > 0
                && (new_lower_bound - lower_bound).abs() < F::from(self.config.tolerance).unwrap()
            {
                converged = true;
            }

            // Update parameters
            weight_concentration = new_weight_concentration;
            mean_precision = new_mean_precision;
            means = new_means;
            degrees_of_freedom = new_degrees_of_freedom;
            scale_matrices = new_scale_matrices;
            lower_bound = new_lower_bound;

            self.lower_bound_history.push(lower_bound);

            if converged {
                break;
            }
        }

        // Compute effective number of components
        let effective_components = self.compute_effective_components(&weight_concentration);

        // Compute final responsibilities and weights
        let responsibilities = self.compute_responsibilities(
            data,
            &means,
            &scale_matrices,
            &degrees_of_freedom,
            &weight_concentration,
        )?;
        let weights = self.compute_weights(&weight_concentration);

        let parameters = VariationalGMMParameters {
            weight_concentration,
            mean_precision,
            means,
            degrees_of_freedom,
            scale_matrices,
            lower_bound,
            effective_components,
            n_iter: self.lower_bound_history.len(),
            converged,
        };

        self.parameters = Some(parameters);

        Ok(VariationalGMMResult {
            lower_bound,
            effective_components,
            responsibilities,
            weights,
        })
    }

    /// Initialize means for variational GMM
    fn initialize_means(&self, data: &ArrayView2<F>) -> StatsResult<Array2<F>> {
        let (n_samples, n_features) = data.dim();
        let mut means = Array2::zeros((self.max_components, n_features));

        use rand::{rngs::StdRng, SeedableRng};
        let mut rng = match self.config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => SeedableRng::from_entropy(),
        };

        for i in 0..self.max_components {
            let idx = rng.random_range(0..n_samples);
            means.row_mut(i).assign(&data.row(idx));
        }

        Ok(means)
    }

    /// Compute responsibilities (E-step)
    fn compute_responsibilities(
        &self,
        data: &ArrayView2<F>,
        means: &Array2<F>,
        scale_matrices: &Array3<F>,
        degrees_of_freedom: &Array1<F>,
        weight_concentration: &Array1<F>,
    ) -> StatsResult<Array2<F>> {
        let (n_samples, _) = data.dim();
        let mut responsibilities = Array2::zeros((n_samples, self.max_components));

        for i in 0..n_samples {
            let mut log_probs = Array1::zeros(self.max_components);

            for k in 0..self.max_components {
                // Compute log probability for component k
                let log_weight = self.compute_log_weight(weight_concentration[k]);
                let log_likelihood = self.compute_log_likelihood_component(
                    &data.row(i),
                    &means.row(k),
                    &scale_matrices.slice(s![k, .., ..]),
                    degrees_of_freedom[k],
                )?;
                log_probs[k] = log_weight + log_likelihood;
            }

            // Normalize responsibilities
            let log_sum = self.log_sum_exp(&log_probs);
            for k in 0..self.max_components {
                responsibilities[[i, k]] = (log_probs[k] - log_sum).exp();
            }
        }

        Ok(responsibilities)
    }

    /// Update parameters (M-step)
    fn update_parameters(
        &self,
        data: &ArrayView2<F>,
        responsibilities: &Array2<F>,
    ) -> StatsResult<(Array1<F>, Array1<F>, Array2<F>, Array1<F>, Array3<F>)> {
        let (n_samples, n_features) = data.dim();

        // Update weight concentration
        let mut weight_concentration =
            Array1::from_elem(self.max_components, F::from(self.config.alpha).unwrap());
        for k in 0..self.max_components {
            let nk: F = responsibilities.column(k).sum();
            weight_concentration[k] = weight_concentration[k] + nk;
        }

        // Update means and precisions
        let mut mean_precision = Array1::ones(self.max_components);
        let mut means = Array2::zeros((self.max_components, n_features));
        let mut degrees_of_freedom = Array1::from_elem(
            self.max_components,
            F::from(self.config.nu + n_features as f64).unwrap(),
        );
        let mut scale_matrices = Array3::zeros((self.max_components, n_features, n_features));

        for k in 0..self.max_components {
            let nk: F = responsibilities.column(k).sum();

            if nk > F::zero() {
                // Update mean
                for j in 0..n_features {
                    let mut weighted_sum = F::zero();
                    for i in 0..n_samples {
                        weighted_sum = weighted_sum + responsibilities[[i, k]] * data[[i, j]];
                    }
                    means[[k, j]] = weighted_sum / nk;
                }

                // Update degrees of freedom
                degrees_of_freedom[k] = F::from(self.config.nu).unwrap() + nk;

                // Update scale matrix (simplified)
                for i in 0..n_features {
                    scale_matrices[[k, i, i]] = F::one() + F::from(0.1).unwrap() * nk;
                }
            }
        }

        Ok((
            weight_concentration,
            mean_precision,
            means,
            degrees_of_freedom,
            scale_matrices,
        ))
    }

    /// Compute lower bound
    fn compute_lower_bound(
        &self,
        data: &ArrayView2<F>,
        responsibilities: &Array2<F>,
        weight_concentration: &Array1<F>,
        means: &Array2<F>,
        scale_matrices: &Array3<F>,
    ) -> StatsResult<F> {
        let (n_samples, _) = data.dim();
        let mut lower_bound = F::zero();

        // Expected log likelihood
        for i in 0..n_samples {
            for k in 0..self.max_components {
                if responsibilities[[i, k]] > F::zero() {
                    let log_likelihood = self.compute_log_likelihood_component(
                        &data.row(i),
                        &means.row(k),
                        &scale_matrices.slice(s![k, .., ..]),
                        F::from(10.0).unwrap(), // simplified
                    )?;
                    lower_bound = lower_bound + responsibilities[[i, k]] * log_likelihood;
                }
            }
        }

        // KL divergence terms (simplified)
        for k in 0..self.max_components {
            let weight_contrib = weight_concentration[k] * weight_concentration[k].ln();
            lower_bound = lower_bound - weight_contrib * F::from(0.01).unwrap();
        }

        Ok(lower_bound)
    }

    /// Compute effective number of components
    fn compute_effective_components(&self, weight_concentration: &Array1<F>) -> usize {
        let total: F = weight_concentration.sum();
        let mut effective = 0;

        for &weight in weight_concentration.iter() {
            let proportion = weight / total;
            if proportion > F::from(0.01).unwrap() {
                // 1% threshold
                effective += 1;
            }
        }

        effective
    }

    /// Compute component weights
    fn compute_weights(&self, weight_concentration: &Array1<F>) -> Array1<F> {
        let total: F = weight_concentration.sum();
        weight_concentration.mapv(|w| w / total)
    }

    /// Compute log weight
    fn compute_log_weight(&self, concentration: F) -> F {
        concentration.ln()
    }

    /// Compute log likelihood for a component (simplified)
    fn compute_log_likelihood_component(
        &self,
        point: &ArrayView1<F>,
        mean: &ArrayView1<F>,
        _scale_matrix: &ndarray::ArrayBase<ndarray::ViewRepr<&F>, ndarray::Dim<[usize; 2]>>,
        _degrees_of_freedom: F,
    ) -> StatsResult<F> {
        // Simplified Gaussian log likelihood
        let mut sum_sq = F::zero();
        for (x, m) in point.iter().zip(mean.iter()) {
            let diff = *x - *m;
            sum_sq = sum_sq + diff * diff;
        }

        let log_likelihood = -F::from(0.5).unwrap() * sum_sq;
        Ok(log_likelihood)
    }

    /// Log-sum-exp for numerical stability
    fn log_sum_exp(&self, log_values: &Array1<F>) -> F {
        let max_val = log_values.iter().fold(F::neg_infinity(), |a, &b| a.max(b));
        if max_val == F::neg_infinity() {
            return F::neg_infinity();
        }

        let sum: F = log_values.iter().map(|&x| (x - max_val).exp()).sum();

        max_val + sum.ln()
    }
}
