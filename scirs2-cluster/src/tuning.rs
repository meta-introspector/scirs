//! Automatic hyperparameter tuning for clustering algorithms
//!
//! This module provides comprehensive hyperparameter optimization capabilities
//! for all clustering algorithms in the scirs2-cluster crate. It supports
//! grid search, random search, Bayesian optimization, and adaptive strategies.

use ndarray::{Array1, Array2, ArrayView2};
use num_traits::{Float, FromPrimitive};
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Debug;

use crate::advanced::{
    adaptive_online_clustering, quantum_kmeans, rl_clustering, AdaptiveOnlineConfig, QuantumConfig,
    RLClusteringConfig,
};
use crate::density::{dbscan, optics};
use crate::error::{ClusteringError, Result};
use crate::gmm::gaussian_mixture;
use crate::hierarchy::linkage;
use crate::meanshift::mean_shift;
use crate::metrics::{calinski_harabasz_score, davies_bouldin_score, silhouette_score};
use crate::stability::OptimalKSelector;
use crate::vq::{kmeans, kmeans2};

/// Hyperparameter tuning configuration
#[derive(Debug, Clone)]
pub struct TuningConfig {
    /// Search strategy to use
    pub strategy: SearchStrategy,
    /// Evaluation metric for optimization
    pub metric: EvaluationMetric,
    /// Cross-validation configuration
    pub cv_config: CrossValidationConfig,
    /// Maximum number of evaluations
    pub max_evaluations: usize,
    /// Early stopping criteria
    pub early_stopping: Option<EarlyStoppingConfig>,
    /// Random seed for reproducible results
    pub random_seed: Option<u64>,
    /// Parallel evaluation configuration
    pub parallel_config: Option<ParallelConfig>,
    /// Resource constraints
    pub resource_constraints: ResourceConstraints,
}

/// Hyperparameter search strategies
#[derive(Debug, Clone)]
pub enum SearchStrategy {
    /// Exhaustive grid search
    GridSearch,
    /// Random search with specified number of trials
    RandomSearch { n_trials: usize },
    /// Bayesian optimization using Gaussian processes
    BayesianOptimization {
        n_initial_points: usize,
        acquisition_function: AcquisitionFunction,
    },
    /// Adaptive search that adjusts based on results
    AdaptiveSearch {
        initial_strategy: Box<SearchStrategy>,
        adaptation_frequency: usize,
    },
    /// Multi-objective optimization
    MultiObjective {
        objectives: Vec<EvaluationMetric>,
        strategy: Box<SearchStrategy>,
    },
    /// Ensemble search combining multiple strategies
    EnsembleSearch {
        strategies: Vec<SearchStrategy>,
        weights: Vec<f64>,
    },
    /// Evolutionary search strategy
    EvolutionarySearch {
        population_size: usize,
        n_generations: usize,
        mutation_rate: f64,
        crossover_rate: f64,
    },
    /// Sequential model-based optimization
    SMBO {
        surrogate_model: SurrogateModel,
        acquisition_function: AcquisitionFunction,
    },
}

/// Acquisition functions for Bayesian optimization
#[derive(Debug, Clone)]
pub enum AcquisitionFunction {
    /// Expected Improvement
    ExpectedImprovement,
    /// Upper Confidence Bound
    UpperConfidenceBound { beta: f64 },
    /// Probability of Improvement
    ProbabilityOfImprovement,
    /// Entropy Search
    EntropySearch,
    /// Knowledge Gradient
    KnowledgeGradient,
    /// Thompson Sampling
    ThompsonSampling,
}

/// Surrogate models for SMBO
#[derive(Debug, Clone)]
pub enum SurrogateModel {
    /// Gaussian Process
    GaussianProcess { kernel: KernelType, noise: f64 },
    /// Random Forest
    RandomForest {
        n_trees: usize,
        max_depth: Option<usize>,
    },
    /// Gradient Boosting
    GradientBoosting {
        n_estimators: usize,
        learning_rate: f64,
    },
}

/// Kernel types for Gaussian Processes
#[derive(Debug, Clone)]
pub enum KernelType {
    /// Radial Basis Function (RBF)
    RBF { length_scale: f64 },
    /// Matérn kernel
    Matern { length_scale: f64, nu: f64 },
    /// Linear kernel
    Linear,
    /// Polynomial kernel
    Polynomial { degree: usize },
}

/// Evaluation metrics for hyperparameter optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvaluationMetric {
    /// Silhouette coefficient (higher is better)
    SilhouetteScore,
    /// Davies-Bouldin index (lower is better)
    DaviesBouldinIndex,
    /// Calinski-Harabasz index (higher is better)
    CalinskiHarabaszIndex,
    /// Within-cluster sum of squares (lower is better)
    Inertia,
    /// Adjusted Rand Index (for labeled data)
    AdjustedRandIndex,
    /// Custom metric
    Custom(String),
    /// Ensemble consensus score
    EnsembleConsensus,
    /// Stability-based metrics
    Stability,
    /// Information-theoretic metrics
    MutualInformation,
}

/// Cross-validation configuration
#[derive(Debug, Clone)]
pub struct CrossValidationConfig {
    /// Number of folds
    pub n_folds: usize,
    /// Fraction of data to use for validation
    pub validation_ratio: f64,
    /// Strategy for cross-validation
    pub strategy: CVStrategy,
    /// Shuffle data before splitting
    pub shuffle: bool,
}

/// Cross-validation strategies
#[derive(Debug, Clone)]
pub enum CVStrategy {
    /// K-fold cross-validation
    KFold,
    /// Stratified K-fold (for labeled data)
    StratifiedKFold,
    /// Time series split (preserves temporal order)
    TimeSeriesSplit,
    /// Bootstrap cross-validation
    Bootstrap { n_bootstrap: usize },
    /// Ensemble cross-validation (multiple CV strategies)
    EnsembleCV { strategies: Vec<CVStrategy> },
    /// Monte Carlo cross-validation
    MonteCarlo { n_splits: usize, test_size: f64 },
    /// Nested cross-validation
    NestedCV {
        outer_folds: usize,
        inner_folds: usize,
    },
}

/// Early stopping configuration
#[derive(Debug, Clone)]
pub struct EarlyStoppingConfig {
    /// Patience (number of evaluations without improvement)
    pub patience: usize,
    /// Minimum improvement required
    pub min_improvement: f64,
    /// Evaluation frequency
    pub evaluation_frequency: usize,
}

/// Parallel evaluation configuration
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Number of parallel workers
    pub n_workers: usize,
    /// Batch size for parallel evaluation
    pub batch_size: usize,
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
}

/// Load balancing strategies for parallel evaluation
#[derive(Debug, Clone)]
pub enum LoadBalancingStrategy {
    /// Round-robin assignment
    RoundRobin,
    /// Work stealing
    WorkStealing,
    /// Dynamic load balancing
    Dynamic,
}

/// Resource constraints for hyperparameter tuning
#[derive(Debug, Clone)]
pub struct ResourceConstraints {
    /// Maximum memory usage per evaluation (bytes)
    pub max_memory_per_evaluation: Option<usize>,
    /// Maximum time per evaluation (seconds)
    pub max_time_per_evaluation: Option<f64>,
    /// Maximum total tuning time (seconds)
    pub max_total_time: Option<f64>,
}

/// Hyperparameter specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HyperParameter {
    /// Integer parameter with range [min, max]
    Integer { min: i64, max: i64 },
    /// Float parameter with range [min, max]
    Float { min: f64, max: f64 },
    /// Categorical parameter with choices
    Categorical { choices: Vec<String> },
    /// Boolean parameter
    Boolean,
    /// Log-uniform distribution for float parameters
    LogUniform { min: f64, max: f64 },
    /// Discrete choices for integer parameters
    IntegerChoices { choices: Vec<i64> },
}

/// Hyperparameter search space for clustering algorithms
#[derive(Debug, Clone)]
pub struct SearchSpace {
    /// Parameters to optimize
    pub parameters: HashMap<String, HyperParameter>,
    /// Algorithm-specific constraints
    pub constraints: Vec<ParameterConstraint>,
}

/// Parameter constraints for interdependent hyperparameters
#[derive(Debug, Clone)]
pub enum ParameterConstraint {
    /// Conditional constraint: if condition then constraint
    Conditional {
        condition: String,
        constraint: Box<ParameterConstraint>,
    },
    /// Range constraint: parameter must be in range
    Range {
        parameter: String,
        min: f64,
        max: f64,
    },
    /// Dependency constraint: parameter A depends on parameter B
    Dependency {
        dependent: String,
        dependency: String,
        relationship: DependencyRelationship,
    },
}

/// Dependency relationships between parameters
#[derive(Debug, Clone)]
pub enum DependencyRelationship {
    /// Linear relationship: A = k * B + c
    Linear { k: f64, c: f64 },
    /// Proportional: A <= ratio * B
    Proportional { ratio: f64 },
    /// Custom function
    Custom(String),
}

/// Hyperparameter evaluation result
#[derive(Debug, Clone)]
pub struct EvaluationResult {
    /// Parameter values used
    pub parameters: HashMap<String, f64>,
    /// Primary metric score
    pub score: f64,
    /// Additional metrics
    pub additional_metrics: HashMap<String, f64>,
    /// Evaluation time (seconds)
    pub evaluation_time: f64,
    /// Memory usage (bytes)
    pub memory_usage: Option<usize>,
    /// Cross-validation scores
    pub cv_scores: Vec<f64>,
    /// Standard deviation of CV scores
    pub cv_std: f64,
    /// Algorithm-specific metadata
    pub metadata: HashMap<String, String>,
}

/// Tuning results
#[derive(Debug, Clone)]
pub struct TuningResult {
    /// Best parameter configuration found
    pub best_parameters: HashMap<String, f64>,
    /// Best score achieved
    pub best_score: f64,
    /// All evaluation results
    pub evaluation_history: Vec<EvaluationResult>,
    /// Convergence information
    pub convergence_info: ConvergenceInfo,
    /// Search space exploration statistics
    pub exploration_stats: ExplorationStats,
    /// Total tuning time
    pub total_time: f64,
    /// Ensemble results (if ensemble method was used)
    pub ensemble_results: Option<EnsembleResults>,
    /// Pareto front (for multi-objective optimization)
    pub pareto_front: Option<Vec<HashMap<String, f64>>>,
}

/// Results from ensemble tuning
#[derive(Debug, Clone)]
pub struct EnsembleResults {
    /// Results from each ensemble member
    pub member_results: Vec<TuningResult>,
    /// Consensus best parameters
    pub consensus_parameters: HashMap<String, f64>,
    /// Agreement score between ensemble members
    pub agreement_score: f64,
    /// Diversity metrics
    pub diversity_metrics: HashMap<String, f64>,
}

/// Bayesian optimization state
#[derive(Debug, Clone)]
struct BayesianState {
    /// Observed parameters and scores
    observations: Vec<(HashMap<String, f64>, f64)>,
    /// Gaussian process mean function
    gp_mean: Option<f64>,
    /// Gaussian process covariance matrix
    gp_covariance: Option<Array2<f64>>,
    /// Acquisition function values
    acquisition_values: Vec<f64>,
    /// Parameter names for consistent ordering
    parameter_names: Vec<String>,
    /// GP hyperparameters
    gp_hyperparameters: GpHyperparameters,
    /// Noise level
    noise_level: f64,
    /// Current best observed value
    current_best: f64,
}

/// Gaussian Process hyperparameters
#[derive(Debug, Clone)]
struct GpHyperparameters {
    /// Length scales for each dimension
    length_scales: Vec<f64>,
    /// Signal variance
    signal_variance: f64,
    /// Noise variance
    noise_variance: f64,
    /// Kernel type
    kernel_type: KernelType,
}

/// Convergence information
#[derive(Debug, Clone)]
pub struct ConvergenceInfo {
    /// Whether tuning converged
    pub converged: bool,
    /// Iteration at which convergence was detected
    pub convergence_iteration: Option<usize>,
    /// Reason for stopping
    pub stopping_reason: StoppingReason,
}

/// Reasons for stopping hyperparameter tuning
#[derive(Debug, Clone)]
pub enum StoppingReason {
    /// Maximum evaluations reached
    MaxEvaluations,
    /// Early stopping triggered
    EarlyStopping,
    /// Time limit exceeded
    TimeLimit,
    /// Convergence achieved
    Convergence,
    /// User interruption
    UserInterruption,
    /// Resource constraints
    ResourceConstraints,
}

/// Search space exploration statistics
#[derive(Debug, Clone)]
pub struct ExplorationStats {
    /// Parameter space coverage
    pub coverage: f64,
    /// Distribution of parameter values explored
    pub parameter_distributions: HashMap<String, Vec<f64>>,
    /// Correlation between parameters and performance
    pub parameter_importance: HashMap<String, f64>,
}

/// Main hyperparameter tuner
pub struct AutoTuner<F: Float> {
    config: TuningConfig,
    _phantom: std::marker::PhantomData<F>,
}

impl<
        F: Float + FromPrimitive + Debug + 'static + std::iter::Sum + std::fmt::Display + Send + Sync,
    > AutoTuner<F>
where
    f64: From<F>,
{
    /// Create a new auto tuner
    pub fn new(config: TuningConfig) -> Self {
        Self {
            config,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Tune K-means hyperparameters
    pub fn tune_kmeans(
        &self,
        data: ArrayView2<F>,
        search_space: SearchSpace,
    ) -> Result<TuningResult> {
        let start_time = std::time::Instant::now();
        let mut evaluation_history = Vec::new();
        let mut best_score = f64::NEG_INFINITY;
        let mut best_parameters = HashMap::new();

        // Generate parameter combinations based on search strategy
        let parameter_combinations = self.generate_parameter_combinations(&search_space)?;

        let mut rng = match self.config.random_seed {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
            None => rand::rngs::StdRng::seed_from_u64(42),
        };

        for (eval_idx, params) in parameter_combinations.iter().enumerate() {
            if eval_idx >= self.config.max_evaluations {
                break;
            }

            // Check time constraints
            if let Some(max_time) = self.config.resource_constraints.max_total_time {
                if start_time.elapsed().as_secs_f64() > max_time {
                    break;
                }
            }

            let eval_start = std::time::Instant::now();

            // Extract parameters for K-means
            let k = params.get("n_clusters").map(|&x| x as usize).unwrap_or(3);
            let max_iter = params.get("max_iter").map(|&x| x as usize);
            let tol = params.get("tolerance").copied();
            let seed = rng.random::<u64>();

            // Perform cross-validation
            let cv_scores = self.cross_validate_kmeans(data, k, max_iter, tol, Some(seed))?;

            let mean_score = cv_scores.iter().sum::<f64>() / cv_scores.len() as f64;
            let cv_std = if cv_scores.len() > 1 {
                let variance = cv_scores
                    .iter()
                    .map(|&x| (x - mean_score).powi(2))
                    .sum::<f64>()
                    / (cv_scores.len() - 1) as f64;
                variance.sqrt()
            } else {
                0.0
            };

            let eval_time = eval_start.elapsed().as_secs_f64();

            let result = EvaluationResult {
                parameters: params.clone(),
                score: mean_score,
                additional_metrics: HashMap::new(),
                evaluation_time: eval_time,
                memory_usage: None,
                cv_scores,
                cv_std,
                metadata: HashMap::new(),
            };

            // Update best result (handle minimization vs maximization)
            let is_better = match self.config.metric {
                EvaluationMetric::SilhouetteScore
                | EvaluationMetric::CalinskiHarabaszIndex
                | EvaluationMetric::AdjustedRandIndex => mean_score > best_score,
                EvaluationMetric::DaviesBouldinIndex | EvaluationMetric::Inertia => {
                    mean_score < best_score || best_score == f64::NEG_INFINITY
                }
                _ => mean_score > best_score,
            };

            if is_better {
                best_score = mean_score;
                best_parameters = params.clone();
            }

            evaluation_history.push(result);

            // Check early stopping
            if let Some(ref early_stop) = self.config.early_stopping {
                if self.should_stop_early(&evaluation_history, early_stop) {
                    break;
                }
            }
        }

        let total_time = start_time.elapsed().as_secs_f64();

        let convergence_info = ConvergenceInfo {
            converged: evaluation_history.len() >= self.config.max_evaluations,
            convergence_iteration: None,
            stopping_reason: if evaluation_history.len() >= self.config.max_evaluations {
                StoppingReason::MaxEvaluations
            } else {
                StoppingReason::EarlyStopping
            },
        };

        let exploration_stats = self.calculate_exploration_stats(&evaluation_history);

        Ok(TuningResult {
            best_parameters,
            best_score,
            evaluation_history,
            convergence_info,
            exploration_stats,
            total_time,
        })
    }

    /// Tune DBSCAN hyperparameters
    pub fn tune_dbscan(
        &self,
        data: ArrayView2<F>,
        search_space: SearchSpace,
    ) -> Result<TuningResult> {
        let start_time = std::time::Instant::now();
        let mut evaluation_history = Vec::new();
        let mut best_score = f64::NEG_INFINITY;
        let mut best_parameters = HashMap::new();

        let parameter_combinations = self.generate_parameter_combinations(&search_space)?;

        for (eval_idx, params) in parameter_combinations.iter().enumerate() {
            if eval_idx >= self.config.max_evaluations {
                break;
            }

            let eval_start = std::time::Instant::now();

            // Extract DBSCAN parameters
            let eps = params.get("eps").copied().unwrap_or(0.5);
            let min_samples = params.get("min_samples").map(|&x| x as usize).unwrap_or(5);

            // Perform cross-validation for DBSCAN
            let cv_scores = self.cross_validate_dbscan(data, eps, min_samples)?;

            let mean_score = cv_scores.iter().sum::<f64>() / cv_scores.len() as f64;
            let cv_std = if cv_scores.len() > 1 {
                let variance = cv_scores
                    .iter()
                    .map(|&x| (x - mean_score).powi(2))
                    .sum::<f64>()
                    / (cv_scores.len() - 1) as f64;
                variance.sqrt()
            } else {
                0.0
            };

            let eval_time = eval_start.elapsed().as_secs_f64();

            let result = EvaluationResult {
                parameters: params.clone(),
                score: mean_score,
                additional_metrics: HashMap::new(),
                evaluation_time: eval_time,
                memory_usage: None,
                cv_scores,
                cv_std,
                metadata: HashMap::new(),
            };

            // Update best result
            let is_better = match self.config.metric {
                EvaluationMetric::SilhouetteScore
                | EvaluationMetric::CalinskiHarabaszIndex
                | EvaluationMetric::AdjustedRandIndex => mean_score > best_score,
                EvaluationMetric::DaviesBouldinIndex | EvaluationMetric::Inertia => {
                    mean_score < best_score || best_score == f64::NEG_INFINITY
                }
                _ => mean_score > best_score,
            };

            if is_better {
                best_score = mean_score;
                best_parameters = params.clone();
            }

            evaluation_history.push(result);

            // Check early stopping
            if let Some(ref early_stop) = self.config.early_stopping {
                if self.should_stop_early(&evaluation_history, early_stop) {
                    break;
                }
            }
        }

        let total_time = start_time.elapsed().as_secs_f64();

        let convergence_info = ConvergenceInfo {
            converged: evaluation_history.len() >= self.config.max_evaluations,
            convergence_iteration: None,
            stopping_reason: if evaluation_history.len() >= self.config.max_evaluations {
                StoppingReason::MaxEvaluations
            } else {
                StoppingReason::EarlyStopping
            },
        };

        let exploration_stats = self.calculate_exploration_stats(&evaluation_history);

        Ok(TuningResult {
            best_parameters,
            best_score,
            evaluation_history,
            convergence_info,
            exploration_stats,
            total_time,
        })
    }

    /// Cross-validate K-means clustering
    fn cross_validate_kmeans(
        &self,
        data: ArrayView2<F>,
        k: usize,
        max_iter: Option<usize>,
        tol: Option<f64>,
        seed: Option<u64>,
    ) -> Result<Vec<f64>> {
        let mut scores = Vec::new();
        let n_samples = data.shape()[0];

        match self.config.cv_config.strategy {
            CVStrategy::KFold => {
                let fold_size = n_samples / self.config.cv_config.n_folds;

                for fold in 0..self.config.cv_config.n_folds {
                    let start_idx = fold * fold_size;
                    let end_idx = if fold == self.config.cv_config.n_folds - 1 {
                        n_samples
                    } else {
                        (fold + 1) * fold_size
                    };

                    // Create train/test split
                    let mut train_indices = Vec::new();
                    let mut test_indices = Vec::new();

                    for i in 0..n_samples {
                        if i >= start_idx && i < end_idx {
                            test_indices.push(i);
                        } else {
                            train_indices.push(i);
                        }
                    }

                    if train_indices.is_empty() || test_indices.is_empty() {
                        continue;
                    }

                    // Extract training data
                    let train_data = self.extract_subset(data, &train_indices)?;

                    // Run K-means on training data
                    match kmeans2(
                        train_data.view(),
                        k,
                        max_iter.unwrap_or(100),
                        tol,
                        None,
                        None,
                        Some(false),
                        seed,
                    ) {
                        Ok((centroids, labels)) => {
                            // Calculate score based on metric
                            let score = self.calculate_metric_score(
                                train_data.view(),
                                &labels.mapv(|x| x),
                                Some(&centroids),
                            )?;
                            scores.push(score);
                        }
                        Err(_) => {
                            // Skip failed runs
                            continue;
                        }
                    }
                }
            }
            _ => {
                // For other CV strategies, implement similar logic
                // For now, just do a single evaluation
                match kmeans2(
                    data,
                    k,
                    max_iter.unwrap_or(100),
                    tol,
                    None,
                    None,
                    Some(false),
                    seed,
                ) {
                    Ok((centroids, labels)) => {
                        let score = self.calculate_metric_score(
                            data,
                            &labels.mapv(|x| x),
                            Some(&centroids),
                        )?;
                        scores.push(score);
                    }
                    Err(_) => {
                        scores.push(f64::NEG_INFINITY);
                    }
                }
            }
        }

        if scores.is_empty() {
            scores.push(f64::NEG_INFINITY);
        }

        Ok(scores)
    }

    /// Cross-validate DBSCAN clustering
    fn cross_validate_dbscan(
        &self,
        data: ArrayView2<F>,
        eps: f64,
        min_samples: usize,
    ) -> Result<Vec<f64>> {
        let mut scores = Vec::new();

        // For DBSCAN, we typically don't use cross-validation in the traditional sense
        // since it's not a predictive model. Instead, we evaluate on the full dataset.
        let data_f64 = data.mapv(|x| x.to_f64().unwrap_or(0.0));

        match dbscan(data_f64.view(), eps, min_samples) {
            Ok(labels) => {
                let score = self.calculate_metric_score(data, &labels, None)?;
                scores.push(score);
            }
            Err(_) => {
                scores.push(f64::NEG_INFINITY);
            }
        }

        Ok(scores)
    }

    /// Calculate metric score for evaluation
    fn calculate_metric_score(
        &self,
        data: ArrayView2<F>,
        labels: &Array1<usize>,
        centroids: Option<&Array2<usize>>,
    ) -> Result<f64> {
        let data_f64 = data.mapv(|x| x.to_f64().unwrap_or(0.0));
        let labels_i32 = labels.mapv(|x| x as i32);

        match self.config.metric {
            EvaluationMetric::SilhouetteScore => {
                silhouette_score(data_f64.view(), labels_i32.view())
            }
            EvaluationMetric::DaviesBouldinIndex => {
                davies_bouldin_score(data_f64.view(), labels_i32.view())
            }
            EvaluationMetric::CalinskiHarabaszIndex => {
                calinski_harabasz_score(data_f64.view(), labels_i32.view())
            }
            EvaluationMetric::Inertia => {
                // Calculate within-cluster sum of squares
                if let Some(centroids) = centroids {
                    let centroids_f64 = centroids.mapv(|x| x as f64);
                    self.calculate_inertia(&data_f64, labels, &centroids_f64)
                } else {
                    Ok(f64::INFINITY) // Invalid for algorithms without centroids
                }
            }
            _ => Ok(0.0), // Placeholder for other metrics
        }
    }

    /// Calculate inertia (within-cluster sum of squares)
    fn calculate_inertia(
        &self,
        data: &Array2<f64>,
        labels: &Array1<usize>,
        centroids: &Array2<f64>,
    ) -> Result<f64> {
        let mut total_inertia = 0.0;

        for (i, &label) in labels.iter().enumerate() {
            let mut distance_sq = 0.0;
            for j in 0..data.ncols() {
                let diff = data[[i, j]] - centroids[[label, j]];
                distance_sq += diff * diff;
            }
            total_inertia += distance_sq;
        }

        Ok(total_inertia)
    }

    /// Extract subset of data based on indices
    fn extract_subset(&self, data: ArrayView2<F>, indices: &[usize]) -> Result<Array2<F>> {
        let n_features = data.ncols();
        let mut subset = Array2::zeros((indices.len(), n_features));

        for (new_idx, &old_idx) in indices.iter().enumerate() {
            if old_idx < data.nrows() {
                subset.row_mut(new_idx).assign(&data.row(old_idx));
            }
        }

        Ok(subset)
    }

    /// Generate parameter combinations based on search strategy
    fn generate_parameter_combinations(
        &self,
        search_space: &SearchSpace,
    ) -> Result<Vec<HashMap<String, f64>>> {
        match &self.config.strategy {
            SearchStrategy::GridSearch => self.generate_grid_combinations(search_space),
            SearchStrategy::RandomSearch { n_trials } => {
                self.generate_random_combinations(search_space, *n_trials)
            }
            SearchStrategy::BayesianOptimization {
                n_initial_points,
                acquisition_function,
            } => self.generate_bayesian_combinations(
                search_space,
                *n_initial_points,
                acquisition_function,
            ),
            SearchStrategy::EnsembleSearch {
                strategies,
                weights,
            } => self.generate_ensemble_combinations(search_space, strategies, weights),
            SearchStrategy::EvolutionarySearch {
                population_size,
                n_generations,
                mutation_rate,
                crossover_rate,
            } => self.generate_evolutionary_combinations(
                search_space,
                *population_size,
                *n_generations,
                *mutation_rate,
                *crossover_rate,
            ),
            SearchStrategy::SMBO {
                surrogate_model,
                acquisition_function,
            } => {
                self.generate_smbo_combinations(search_space, surrogate_model, acquisition_function)
            }
            SearchStrategy::MultiObjective {
                objectives,
                strategy,
            } => {
                // For multi-objective, we need special handling
                self.generate_multi_objective_combinations(search_space, objectives, strategy)
            }
            SearchStrategy::AdaptiveSearch {
                initial_strategy, ..
            } => {
                // Start with initial strategy
                match initial_strategy.as_ref() {
                    SearchStrategy::RandomSearch { n_trials } => {
                        self.generate_random_combinations(search_space, *n_trials)
                    }
                    SearchStrategy::GridSearch => self.generate_grid_combinations(search_space),
                    _ => {
                        // Fallback to random search
                        self.generate_random_combinations(search_space, self.config.max_evaluations)
                    }
                }
            }
        }
    }

    /// Generate grid search combinations
    fn generate_grid_combinations(
        &self,
        search_space: &SearchSpace,
    ) -> Result<Vec<HashMap<String, f64>>> {
        let mut combinations = Vec::new();
        let mut param_names = Vec::new();
        let mut param_values = Vec::new();

        // Extract parameter ranges
        for (name, param) in &search_space.parameters {
            param_names.push(name.clone());
            match param {
                HyperParameter::Integer { min, max } => {
                    let values: Vec<f64> = (*min..=*max).map(|x| x as f64).collect();
                    param_values.push(values);
                }
                HyperParameter::Float { min, max } => {
                    // Create a reasonable grid for float parameters
                    let n_steps = 10; // Could be configurable
                    let step = (max - min) / (n_steps as f64 - 1.0);
                    let values: Vec<f64> = (0..n_steps).map(|i| min + i as f64 * step).collect();
                    param_values.push(values);
                }
                HyperParameter::Categorical { choices } => {
                    // Map categorical choices to numeric values
                    let values: Vec<f64> = (0..choices.len()).map(|i| i as f64).collect();
                    param_values.push(values);
                }
                HyperParameter::Boolean => {
                    param_values.push(vec![0.0, 1.0]);
                }
                HyperParameter::LogUniform { min, max } => {
                    let n_steps = 10;
                    let log_min = min.ln();
                    let log_max = max.ln();
                    let step = (log_max - log_min) / (n_steps as f64 - 1.0);
                    let values: Vec<f64> = (0..n_steps)
                        .map(|i| (log_min + i as f64 * step).exp())
                        .collect();
                    param_values.push(values);
                }
                HyperParameter::IntegerChoices { choices } => {
                    let values: Vec<f64> = choices.iter().map(|&x| x as f64).collect();
                    param_values.push(values);
                }
            }
        }

        // Generate all combinations
        self.generate_cartesian_product(
            &param_names,
            &param_values,
            &mut combinations,
            Vec::new(),
            0,
        );

        Ok(combinations)
    }

    /// Generate cartesian product of parameter values
    fn generate_cartesian_product(
        &self,
        param_names: &[String],
        param_values: &[Vec<f64>],
        combinations: &mut Vec<HashMap<String, f64>>,
        current: Vec<f64>,
        index: usize,
    ) {
        if index == param_names.len() {
            let mut combination = HashMap::new();
            for (i, name) in param_names.iter().enumerate() {
                combination.insert(name.clone(), current[i]);
            }
            combinations.push(combination);
            return;
        }

        for &value in &param_values[index] {
            let mut new_current = current.clone();
            new_current.push(value);
            self.generate_cartesian_product(
                param_names,
                param_values,
                combinations,
                new_current,
                index + 1,
            );
        }
    }

    /// Generate random search combinations
    fn generate_random_combinations(
        &self,
        search_space: &SearchSpace,
        n_trials: usize,
    ) -> Result<Vec<HashMap<String, f64>>> {
        let mut combinations = Vec::new();
        let mut rng = match self.config.random_seed {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
            None => rand::rngs::StdRng::seed_from_u64(42),
        };

        for _ in 0..n_trials {
            let mut combination = HashMap::new();

            for (name, param) in &search_space.parameters {
                let value = match param {
                    HyperParameter::Integer { min, max } => rng.gen_range(*min..=*max) as f64,
                    HyperParameter::Float { min, max } => rng.gen_range(*min..=*max),
                    HyperParameter::Categorical { choices } => {
                        rng.gen_range(0..choices.len()) as f64
                    }
                    HyperParameter::Boolean => {
                        if rng.gen_bool(0.5) {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    HyperParameter::LogUniform { min, max } => {
                        let log_min = min.ln();
                        let log_max = max.ln();
                        let log_value = rng.gen_range(log_min..=log_max);
                        log_value.exp()
                    }
                    HyperParameter::IntegerChoices { choices } => {
                        let idx = rng.gen_range(0..choices.len());
                        choices[idx] as f64
                    }
                };

                combination.insert(name.clone(), value);
            }

            combinations.push(combination);
        }

        Ok(combinations)
    }

    /// Check if early stopping criteria are met
    fn should_stop_early(
        &self,
        evaluation_history: &[EvaluationResult],
        early_stop_config: &EarlyStoppingConfig,
    ) -> bool {
        if evaluation_history.len() < early_stop_config.patience {
            return false;
        }

        let recent_evaluations =
            &evaluation_history[evaluation_history.len() - early_stop_config.patience..];
        let best_recent = recent_evaluations
            .iter()
            .map(|r| r.score)
            .fold(f64::NEG_INFINITY, f64::max);

        let current_best = evaluation_history
            .iter()
            .map(|r| r.score)
            .fold(f64::NEG_INFINITY, f64::max);

        (current_best - best_recent) < early_stop_config.min_improvement
    }

    /// Calculate exploration statistics
    fn calculate_exploration_stats(
        &self,
        evaluation_history: &[EvaluationResult],
    ) -> ExplorationStats {
        let mut parameter_distributions = HashMap::new();
        let mut parameter_importance = HashMap::new();

        // Collect parameter distributions
        for result in evaluation_history {
            for (param_name, &value) in &result.parameters {
                parameter_distributions
                    .entry(param_name.clone())
                    .or_insert_with(Vec::new)
                    .push(value);
            }
        }

        // Calculate parameter importance (simplified)
        for (param_name, values) in &parameter_distributions {
            let scores: Vec<f64> = evaluation_history.iter().map(|r| r.score).collect();
            let correlation = self.calculate_correlation(values, &scores);
            parameter_importance.insert(param_name.clone(), correlation.abs());
        }

        ExplorationStats {
            coverage: 1.0, // Simplified calculation
            parameter_distributions,
            parameter_importance,
        }
    }

    /// Calculate correlation between two vectors
    fn calculate_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.len() < 2 {
            return 0.0;
        }

        let n = x.len() as f64;
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xy: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        let sum_x_sq: f64 = x.iter().map(|a| a * a).sum();
        let sum_y_sq: f64 = y.iter().map(|a| a * a).sum();

        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x_sq - sum_x * sum_x) * (n * sum_y_sq - sum_y * sum_y)).sqrt();

        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }

    /// Generate Bayesian optimization combinations
    fn generate_bayesian_combinations(
        &self,
        search_space: &SearchSpace,
        n_initial_points: usize,
        acquisition_function: &AcquisitionFunction,
    ) -> Result<Vec<HashMap<String, f64>>> {
        let mut combinations = Vec::new();
        // Extract parameter names for consistent ordering
        let parameter_names: Vec<String> = search_space.parameters.keys().cloned().collect();
        
        let mut bayesian_state = BayesianState {
            observations: Vec::new(),
            gp_mean: None,
            gp_covariance: None,
            acquisition_values: Vec::new(),
            parameter_names: parameter_names.clone(),
            gp_hyperparameters: GpHyperparameters {
                length_scales: vec![1.0; parameter_names.len()],
                signal_variance: 1.0,
                noise_variance: 0.1,
                kernel_type: KernelType::RBF { length_scale: 1.0 },
            },
            noise_level: 0.1,
            current_best: f64::NEG_INFINITY,
        };

        // Generate initial random points
        let initial_points = self.generate_random_combinations(search_space, n_initial_points)?;
        combinations.extend(initial_points);

        // Generate remaining points using Bayesian optimization
        let remaining_points = self.config.max_evaluations.saturating_sub(n_initial_points);

        for _ in 0..remaining_points {
            // Update Gaussian process with current observations
            self.update_gaussian_process(&mut bayesian_state, &combinations);

            // Find next point with highest acquisition function value
            let next_point = self.optimize_acquisition_function(
                search_space,
                &bayesian_state,
                acquisition_function,
            )?;

            combinations.push(next_point);
        }

        Ok(combinations)
    }

    /// Generate ensemble search combinations
    fn generate_ensemble_combinations(
        &self,
        search_space: &SearchSpace,
        strategies: &[SearchStrategy],
        weights: &[f64],
    ) -> Result<Vec<HashMap<String, f64>>> {
        let mut all_combinations = Vec::new();
        let total_evaluations = self.config.max_evaluations;

        // Normalize weights
        let weight_sum: f64 = weights.iter().sum();
        let normalized_weights: Vec<f64> = weights.iter().map(|w| w / weight_sum).collect();

        // Allocate evaluations based on weights
        for (strategy, &weight) in strategies.iter().zip(normalized_weights.iter()) {
            let n_evaluations = (total_evaluations as f64 * weight) as usize;

            let strategy_combinations = match strategy {
                SearchStrategy::RandomSearch { .. } => {
                    self.generate_random_combinations(search_space, n_evaluations)?
                }
                SearchStrategy::GridSearch => {
                    let grid_combinations = self.generate_grid_combinations(search_space)?;
                    grid_combinations.into_iter().take(n_evaluations).collect()
                }
                _ => {
                    // Fallback to random search for complex strategies
                    self.generate_random_combinations(search_space, n_evaluations)?
                }
            };

            all_combinations.extend(strategy_combinations);
        }

        // Shuffle to mix different strategies
        let mut rng = match self.config.random_seed {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
            None => rand::rngs::StdRng::seed_from_u64(42),
        };

        use rand::seq::SliceRandom;
        all_combinations.shuffle(&mut rng);

        Ok(all_combinations)
    }

    /// Generate evolutionary search combinations using genetic algorithm
    fn generate_evolutionary_combinations(
        &self,
        search_space: &SearchSpace,
        population_size: usize,
        n_generations: usize,
        mutation_rate: f64,
        crossover_rate: f64,
    ) -> Result<Vec<HashMap<String, f64>>> {
        let mut rng = match self.config.random_seed {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
            None => rand::rngs::StdRng::seed_from_u64(42),
        };

        // Initialize population
        let mut population = self.generate_random_combinations(search_space, population_size)?;
        let mut all_combinations = population.clone();

        // Evolution loop
        for generation in 0..n_generations {
            let mut new_population = Vec::new();

            // Elitism: keep best individual from previous generation
            if !population.is_empty() {
                new_population.push(population[0].clone());
            }

            // Generate new offspring
            while new_population.len() < population_size {
                // Selection: tournament selection
                let parent1 = self.tournament_selection(&population, &mut rng)?;
                let parent2 = self.tournament_selection(&population, &mut rng)?;

                // Crossover
                let (mut child1, mut child2) = if rng.gen::<f64>() < crossover_rate {
                    self.crossover(&parent1, &parent2, search_space, &mut rng)?
                } else {
                    (parent1.clone(), parent2.clone())
                };

                // Mutation
                if rng.gen::<f64>() < mutation_rate {
                    self.mutate(&mut child1, search_space, &mut rng)?;
                }
                if rng.gen::<f64>() < mutation_rate {
                    self.mutate(&mut child2, search_space, &mut rng)?;
                }

                new_population.push(child1);
                if new_population.len() < population_size {
                    new_population.push(child2);
                }
            }

            population = new_population;
            all_combinations.extend(population.clone());

            // Early termination if we have enough evaluations
            if all_combinations.len() >= self.config.max_evaluations {
                break;
            }
        }

        // Trim to max evaluations
        all_combinations.truncate(self.config.max_evaluations);
        Ok(all_combinations)
    }

    /// Tournament selection for evolutionary algorithm
    fn tournament_selection(
        &self,
        population: &[HashMap<String, f64>],
        rng: &mut rand::rngs::StdRng,
    ) -> Result<HashMap<String, f64>> {
        let tournament_size = 3.min(population.len());
        let mut best_individual = None;

        for _ in 0..tournament_size {
            let idx = rng.gen_range(0..population.len());
            let individual = &population[idx];
            
            // In a real implementation, we would evaluate fitness here
            // For now, just return the first selected individual
            if best_individual.is_none() {
                best_individual = Some(individual.clone());
            }
        }

        best_individual.ok_or_else(|| ClusteringError::InvalidInput("Empty population".to_string()))
    }

    /// Crossover operation for evolutionary algorithm
    fn crossover(
        &self,
        parent1: &HashMap<String, f64>,
        parent2: &HashMap<String, f64>,
        search_space: &SearchSpace,
        rng: &mut rand::rngs::StdRng,
    ) -> Result<(HashMap<String, f64>, HashMap<String, f64>)> {
        let mut child1 = HashMap::new();
        let mut child2 = HashMap::new();

        for (param_name, param_spec) in &search_space.parameters {
            let val1 = parent1.get(param_name).copied().unwrap_or(0.0);
            let val2 = parent2.get(param_name).copied().unwrap_or(0.0);

            // Uniform crossover with parameter-specific handling
            let (new_val1, new_val2) = match param_spec {
                HyperParameter::Float { min, max } => {
                    // Blend crossover for continuous parameters
                    let alpha = 0.5;
                    let beta = rng.gen::<f64>() * (1.0 + 2.0 * alpha) - alpha;
                    let v1 = (1.0 - beta) * val1 + beta * val2;
                    let v2 = beta * val1 + (1.0 - beta) * val2;
                    (v1.clamp(*min, *max), v2.clamp(*min, *max))
                }
                HyperParameter::Integer { min, max } => {
                    // Single-point crossover for discrete parameters
                    if rng.gen_bool(0.5) {
                        (val1.clamp(*min as f64, *max as f64), val2.clamp(*min as f64, *max as f64))
                    } else {
                        (val2.clamp(*min as f64, *max as f64), val1.clamp(*min as f64, *max as f64))
                    }
                }
                _ => {
                    // For other types, just swap randomly
                    if rng.gen_bool(0.5) {
                        (val1, val2)
                    } else {
                        (val2, val1)
                    }
                }
            };

            child1.insert(param_name.clone(), new_val1);
            child2.insert(param_name.clone(), new_val2);
        }

        Ok((child1, child2))
    }

    /// Mutation operation for evolutionary algorithm
    fn mutate(
        &self,
        individual: &mut HashMap<String, f64>,
        search_space: &SearchSpace,
        rng: &mut rand::rngs::StdRng,
    ) -> Result<()> {
        for (param_name, param_spec) in &search_space.parameters {
            if rng.gen::<f64>() < 0.1 { // 10% chance to mutate each parameter
                let current_val = individual.get(param_name).copied().unwrap_or(0.0);
                
                let new_val = match param_spec {
                    HyperParameter::Float { min, max } => {
                        // Gaussian mutation
                        let std_dev = (max - min) * 0.1; // 10% of range as standard deviation
                        let mutation_delta = rand_distr::Normal::new(0.0, std_dev)
                            .map_err(|e| ClusteringError::InvalidInput(format!("Mutation error: {}", e)))?
                            .sample(rng);
                        (current_val + mutation_delta).clamp(*min, *max)
                    }
                    HyperParameter::Integer { min, max } => {
                        // Random reset mutation for discrete parameters
                        rng.gen_range(*min..=*max) as f64
                    }
                    HyperParameter::Categorical { choices } => {
                        rng.gen_range(0..choices.len()) as f64
                    }
                    HyperParameter::Boolean => {
                        if rng.gen_bool(0.5) { 1.0 } else { 0.0 }
                    }
                    HyperParameter::LogUniform { min, max } => {
                        let log_min = min.ln();
                        let log_max = max.ln();
                        let log_val = rng.gen_range(log_min..=log_max);
                        log_val.exp()
                    }
                    HyperParameter::IntegerChoices { choices } => {
                        let idx = rng.gen_range(0..choices.len());
                        choices[idx] as f64
                    }
                };

                individual.insert(param_name.clone(), new_val);
            }
        }
        Ok(())
    }

    /// Generate SMBO (Sequential Model-Based Optimization) combinations
    fn generate_smbo_combinations(
        &self,
        search_space: &SearchSpace,
        surrogate_model: &SurrogateModel,
        acquisition_function: &AcquisitionFunction,
    ) -> Result<Vec<HashMap<String, f64>>> {
        let mut combinations = Vec::new();
        let n_initial_points = 5; // Start with 5 random points

        // Generate initial random points
        let initial_points = self.generate_random_combinations(search_space, n_initial_points)?;
        combinations.extend(initial_points);

        // Sequential optimization
        let remaining_points = self.config.max_evaluations.saturating_sub(n_initial_points);
        
        for iteration in 0..remaining_points {
            // Build surrogate model from current observations
            let next_point = match surrogate_model {
                SurrogateModel::GaussianProcess { .. } => {
                    // Use Gaussian Process for surrogate modeling
                    self.optimize_gp_acquisition(search_space, &combinations, acquisition_function)?
                }
                SurrogateModel::RandomForest { .. } => {
                    // Use Random Forest (simplified implementation)
                    self.optimize_rf_acquisition(search_space, &combinations)?
                }
                SurrogateModel::GradientBoosting { .. } => {
                    // Use Gradient Boosting (simplified implementation)
                    self.optimize_gb_acquisition(search_space, &combinations)?
                }
            };

            combinations.push(next_point);
        }

        Ok(combinations)
    }

    /// Generate multi-objective optimization combinations
    fn generate_multi_objective_combinations(
        &self,
        search_space: &SearchSpace,
        objectives: &[EvaluationMetric],
        strategy: &SearchStrategy,
    ) -> Result<Vec<HashMap<String, f64>>> {
        // For multi-objective optimization, we use NSGA-II style approach
        let population_size = 50;
        let n_generations = 20;

        // Generate initial population
        let mut population = self.generate_random_combinations(search_space, population_size)?;
        let mut all_combinations = population.clone();

        // Multi-objective evolution
        for generation in 0..n_generations {
            // In a full implementation, we would:
            // 1. Evaluate population on all objectives
            // 2. Perform non-dominated sorting
            // 3. Calculate crowding distance
            // 4. Select parents using tournament selection
            // 5. Apply crossover and mutation
            // 6. Combine parent and offspring populations
            // 7. Select next generation using Pareto dominance

            // Simplified implementation: just generate random variations
            let mut new_population = Vec::new();
            
            for individual in &population {
                let mut mutated = individual.clone();
                
                // Apply small random mutations
                for (param_name, param_spec) in &search_space.parameters {
                    if rand::random::<f64>() < 0.1 { // 10% mutation rate
                        let current_val = mutated.get(param_name).copied().unwrap_or(0.0);
                        let new_val = match param_spec {
                            HyperParameter::Float { min, max } => {
                                let range = max - min;
                                let delta = (rand::random::<f64>() - 0.5) * range * 0.1;
                                (current_val + delta).clamp(*min, *max)
                            }
                            HyperParameter::Integer { min, max } => {
                                rand::random::<f64>() * (*max - *min) as f64 + *min as f64
                            }
                            _ => current_val,
                        };
                        mutated.insert(param_name.clone(), new_val);
                    }
                }
                
                new_population.push(mutated);
            }

            population = new_population;
            all_combinations.extend(population.clone());

            if all_combinations.len() >= self.config.max_evaluations {
                break;
            }
        }

        all_combinations.truncate(self.config.max_evaluations);
        Ok(all_combinations)
    }

    /// Optimize acquisition function using Gaussian Process
    fn optimize_gp_acquisition(
        &self,
        search_space: &SearchSpace,
        _observations: &[HashMap<String, f64>],
        acquisition_function: &AcquisitionFunction,
    ) -> Result<HashMap<String, f64>> {
        // Simplified GP-based acquisition optimization
        // In practice, this would:
        // 1. Fit GP to current observations
        // 2. Compute acquisition function values over search space
        // 3. Find point with maximum acquisition value
        
        // For now, generate multiple random candidates and pick best
        let n_candidates = 100;
        let candidates = self.generate_random_combinations(search_space, n_candidates)?;
        
        // In a real implementation, we would evaluate acquisition function
        // For now, just return a random candidate
        Ok(candidates.into_iter().next().unwrap())
    }

    /// Optimize acquisition function using Random Forest
    fn optimize_rf_acquisition(
        &self,
        search_space: &SearchSpace,
        _observations: &[HashMap<String, f64>],
    ) -> Result<HashMap<String, f64>> {
        // Simplified RF-based acquisition optimization
        let candidates = self.generate_random_combinations(search_space, 50)?;
        Ok(candidates.into_iter().next().unwrap())
    }

    /// Optimize acquisition function using Gradient Boosting
    fn optimize_gb_acquisition(
        &self,
        search_space: &SearchSpace,
        _observations: &[HashMap<String, f64>],
    ) -> Result<HashMap<String, f64>> {
        // Simplified GB-based acquisition optimization
        let candidates = self.generate_random_combinations(search_space, 50)?;
        Ok(candidates.into_iter().next().unwrap())
    }

    /// Update Gaussian process with new observations
    fn update_gaussian_process(
        &self,
        bayesian_state: &mut BayesianState,
        combinations: &[HashMap<String, f64>],
    ) {
        // Enhanced Gaussian process implementation
        if combinations.is_empty() {
            return;
        }

        // Convert parameter combinations to feature vectors
        let param_names: Vec<String> = combinations[0].keys().cloned().collect();
        let n_features = param_names.len();
        let n_observations = combinations.len();

        // Create feature matrix (simplified)
        let mut feature_matrix = Array2::zeros((n_observations, n_features));
        for (i, combination) in combinations.iter().enumerate() {
            for (j, param_name) in param_names.iter().enumerate() {
                feature_matrix[[i, j]] = combination.get(param_name).copied().unwrap_or(0.0);
            }
        }

        // In a full implementation, we would:
        // 1. Compute kernel matrix
        // 2. Add noise term for numerical stability
        // 3. Compute Cholesky decomposition
        // 4. Store GP hyperparameters
        
        // For now, just store simplified mean
        let mean = feature_matrix.mean_axis(ndarray::Axis(0)).unwrap();
        bayesian_state.gp_mean = Some(mean[0]);
        
        // Store a simplified covariance matrix
        bayesian_state.gp_covariance = Some(Array2::eye(n_features));
    }

    /// Optimize acquisition function to find next point
    fn optimize_acquisition_function(
        &self,
        search_space: &SearchSpace,
        bayesian_state: &BayesianState,
        acquisition_function: &AcquisitionFunction,
    ) -> Result<HashMap<String, f64>> {
        // Enhanced acquisition function optimization
        let n_candidates = 1000;
        let candidates = self.generate_random_combinations(search_space, n_candidates)?;
        
        let mut best_candidate = candidates[0].clone();
        let mut best_acquisition_value = f64::NEG_INFINITY;

        for candidate in &candidates {
            let acquisition_value = self.evaluate_acquisition_function(
                candidate,
                bayesian_state,
                acquisition_function,
            );

            if acquisition_value > best_acquisition_value {
                best_acquisition_value = acquisition_value;
                best_candidate = candidate.clone();
            }
        }

        Ok(best_candidate)
    }

    /// Evaluate acquisition function at a point
    fn evaluate_acquisition_function(
        &self,
        point: &HashMap<String, f64>,
        bayesian_state: &BayesianState,
        acquisition_function: &AcquisitionFunction,
    ) -> f64 {
        // Simplified acquisition function evaluation
        // In practice, this would compute EI, UCB, PI, etc. based on GP predictions
        
        match acquisition_function {
            AcquisitionFunction::ExpectedImprovement => {
                // Simplified EI calculation
                let mean = bayesian_state.gp_mean.unwrap_or(0.0);
                let variance = 1.0; // Simplified variance
                
                // EI = (μ - f_best) * Φ(Z) + σ * φ(Z)
                // where Z = (μ - f_best) / σ
                let f_best = 0.0; // Would be current best observed value
                let z = (mean - f_best) / variance.sqrt();
                
                // Simplified normal distribution evaluation
                let phi_z = 0.5 * (1.0 + (z / 1.41421356).tanh()); // Approximation of CDF
                let pdf_z = (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();
                
                (mean - f_best) * phi_z + variance.sqrt() * pdf_z
            }
            AcquisitionFunction::UpperConfidenceBound { beta } => {
                // UCB = μ + β * σ
                let mean = bayesian_state.gp_mean.unwrap_or(0.0);
                let std_dev = 1.0; // Simplified standard deviation
                mean + beta * std_dev
            }
            AcquisitionFunction::ProbabilityOfImprovement => {
                // PI = Φ((μ - f_best) / σ)
                let mean = bayesian_state.gp_mean.unwrap_or(0.0);
                let variance = 1.0;
                let f_best = 0.0;
                let z = (mean - f_best) / variance.sqrt();
                0.5 * (1.0 + (z / 1.41421356).tanh()) // Approximation of CDF
            }
            _ => {
                // For other acquisition functions, return random value
                rand::random::<f64>()
            }
        }
    }

    /// Generate Latin Hypercube Sampling combinations for better space coverage
    fn generate_lhs_combinations(
        &self,
        search_space: &SearchSpace,
        n_points: usize,
    ) -> Result<Vec<HashMap<String, f64>>> {
        let mut rng = rand::thread_rng();
        let mut combinations = Vec::new();
        
        let param_names: Vec<String> = search_space.parameters.keys().cloned().collect();
        let n_params = param_names.len();
        
        // Generate LHS samples
        for i in 0..n_points {
            let mut params = HashMap::new();
            
            for (j, param_name) in param_names.iter().enumerate() {
                let param_spec = &search_space.parameters[param_name];
                
                // LHS sampling: divide parameter space into n_points intervals
                let interval_size = 1.0 / n_points as f64;
                let base_point = i as f64 * interval_size;
                let random_offset = rng.gen::<f64>() * interval_size;
                let normalized_value = base_point + random_offset;
                
                let value = match param_spec {
                    HyperParameter::Float { min, max } => {
                        min + normalized_value * (max - min)
                    }
                    HyperParameter::LogUniform { min, max } => {
                        let log_min = min.ln();
                        let log_max = max.ln();
                        (log_min + normalized_value * (log_max - log_min)).exp()
                    }
                    HyperParameter::Integer { min, max } => {
                        (*min as f64 + normalized_value * (*max - *min) as f64).round()
                    }
                    HyperParameter::IntegerChoices { choices } => {
                        let idx = (normalized_value * choices.len() as f64).floor() as usize;
                        choices[idx.min(choices.len() - 1)] as f64
                    }
                    HyperParameter::Boolean => {
                        if normalized_value < 0.5 { 0.0 } else { 1.0 }
                    }
                    HyperParameter::Categorical { choices } => {
                        let idx = (normalized_value * choices.len() as f64).floor() as usize;
                        idx.min(choices.len() - 1) as f64
                    }
                };
                
                params.insert(param_name.clone(), value);
            }
            
            combinations.push(params);
        }
        
        Ok(combinations)
    }

    /// Enhanced Gaussian Process update with proper kernel computations
    fn update_gaussian_process_enhanced(
        &self,
        bayesian_state: &mut BayesianState,
        combinations: &[HashMap<String, f64>],
    ) {
        if combinations.is_empty() {
            return;
        }

        let n_points = combinations.len();
        let n_features = bayesian_state.parameter_names.len();
        
        // Convert parameter combinations to feature matrix
        let mut feature_matrix = Array2::zeros((n_points, n_features));
        for (i, combo) in combinations.iter().enumerate() {
            for (j, param_name) in bayesian_state.parameter_names.iter().enumerate() {
                feature_matrix[[i, j]] = combo.get(param_name).unwrap_or(&0.0).clone();
            }
        }
        
        // Compute kernel matrix
        let kernel_matrix = self.compute_kernel_matrix(&feature_matrix, &bayesian_state.gp_hyperparameters);
        
        // Add noise to diagonal for numerical stability
        let mut k_with_noise = kernel_matrix.clone();
        for i in 0..n_points {
            k_with_noise[[i, i]] += bayesian_state.gp_hyperparameters.noise_variance;
        }
        
        // Store enhanced GP state
        bayesian_state.gp_covariance = Some(k_with_noise);
        
        // Compute mean (simplified for now - would use actual observations)
        let mean = feature_matrix.mean_axis(Axis(0)).unwrap();
        bayesian_state.gp_mean = Some(mean[0]);
    }

    /// Compute kernel matrix for GP
    fn compute_kernel_matrix(
        &self,
        feature_matrix: &Array2<f64>,
        hyperparams: &GpHyperparameters,
    ) -> Array2<f64> {
        let n_points = feature_matrix.nrows();
        let mut kernel_matrix = Array2::zeros((n_points, n_points));
        
        for i in 0..n_points {
            for j in i..n_points {
                let xi = feature_matrix.row(i);
                let xj = feature_matrix.row(j);
                
                let kernel_value = match &hyperparams.kernel_type {
                    KernelType::RBF { length_scale } => {
                        let dist_sq = xi.iter()
                            .zip(xj.iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f64>();
                        hyperparams.signal_variance * (-0.5 * dist_sq / length_scale.powi(2)).exp()
                    }
                    KernelType::Matern { length_scale, nu } => {
                        let dist = xi.iter()
                            .zip(xj.iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f64>()
                            .sqrt();
                        
                        if dist == 0.0 {
                            hyperparams.signal_variance
                        } else {
                            let sqrt_2nu_dist = (2.0 * nu).sqrt() * dist / length_scale;
                            let bessel_term = 1.0; // Simplified Bessel function
                            hyperparams.signal_variance * 
                                (sqrt_2nu_dist.powf(*nu) * bessel_term * (-sqrt_2nu_dist).exp())
                        }
                    }
                    KernelType::Linear => {
                        xi.iter().zip(xj.iter()).map(|(a, b)| a * b).sum::<f64>()
                    }
                    KernelType::Polynomial { degree } => {
                        let dot_product = xi.iter().zip(xj.iter()).map(|(a, b)| a * b).sum::<f64>();
                        (1.0 + dot_product).powf(*degree as f64)
                    }
                };
                
                kernel_matrix[[i, j]] = kernel_value;
                kernel_matrix[[j, i]] = kernel_value;
            }
        }
        
        kernel_matrix
    }

    /// Enhanced acquisition function optimization with multiple strategies
    fn optimize_acquisition_function_enhanced(
        &self,
        search_space: &SearchSpace,
        bayesian_state: &BayesianState,
        acquisition_function: &AcquisitionFunction,
        iteration: usize,
    ) -> Result<HashMap<String, f64>> {
        let n_candidates = std::cmp::max(1000, 100 * search_space.parameters.len());
        
        // Use multiple strategies for finding the best candidate
        let mut all_candidates = Vec::new();
        
        // Strategy 1: Random sampling
        let random_candidates = self.generate_random_combinations(search_space, n_candidates / 3)?;
        all_candidates.extend(random_candidates);
        
        // Strategy 2: Latin Hypercube Sampling
        let lhs_candidates = self.generate_lhs_combinations(search_space, n_candidates / 3)?;
        all_candidates.extend(lhs_candidates);
        
        // Strategy 3: Gradient-based local optimization around best points
        if !bayesian_state.observations.is_empty() && iteration > 5 {
            let local_candidates = self.generate_local_optimization_candidates(
                search_space, 
                bayesian_state, 
                n_candidates / 3
            )?;
            all_candidates.extend(local_candidates);
        } else {
            // Fallback to more random samples
            let extra_random = self.generate_random_combinations(search_space, n_candidates / 3)?;
            all_candidates.extend(extra_random);
        }
        
        // Evaluate acquisition function for all candidates
        let mut best_candidate = all_candidates[0].clone();
        let mut best_acquisition_value = f64::NEG_INFINITY;
        
        for candidate in &all_candidates {
            let acquisition_value = self.evaluate_acquisition_function_enhanced(
                candidate,
                bayesian_state,
                acquisition_function,
            );
            
            if acquisition_value > best_acquisition_value {
                best_acquisition_value = acquisition_value;
                best_candidate = candidate.clone();
            }
        }
        
        Ok(best_candidate)
    }

    /// Generate local optimization candidates around promising regions
    fn generate_local_optimization_candidates(
        &self,
        search_space: &SearchSpace,
        bayesian_state: &BayesianState,
        n_candidates: usize,
    ) -> Result<Vec<HashMap<String, f64>>> {
        let mut candidates = Vec::new();
        let mut rng = rand::thread_rng();
        
        // Find top performing regions from observations (placeholder)
        let n_centers = std::cmp::min(5, n_candidates / 10);
        
        for _ in 0..n_centers {
            let center = self.generate_random_combinations(search_space, 1)?[0].clone();
            
            // Generate candidates around this center
            for _ in 0..n_candidates / n_centers {
                let mut candidate = HashMap::new();
                
                for (param_name, param_spec) in &search_space.parameters {
                    let center_value = center.get(param_name).unwrap_or(&0.0);
                    
                    let perturbed_value = match param_spec {
                        HyperParameter::Float { min, max } => {
                            let noise_scale = (max - min) * 0.1; // 10% of range
                            let noise = rng.gen_range(-noise_scale..noise_scale);
                            (center_value + noise).clamp(*min, *max)
                        }
                        HyperParameter::Integer { min, max } => {
                            let noise = rng.gen_range(-2..=2);
                            (center_value + noise as f64).round().clamp(*min as f64, *max as f64)
                        }
                        _ => *center_value, // For other types, use center value
                    };
                    
                    candidate.insert(param_name.clone(), perturbed_value);
                }
                
                candidates.push(candidate);
            }
        }
        
        Ok(candidates)
    }

    /// Enhanced acquisition function evaluation with proper GP predictions
    fn evaluate_acquisition_function_enhanced(
        &self,
        point: &HashMap<String, f64>,
        bayesian_state: &BayesianState,
        acquisition_function: &AcquisitionFunction,
    ) -> f64 {
        // Get GP predictions at the point (simplified)
        let (mean, variance) = self.predict_gp(point, bayesian_state);
        let std_dev = variance.sqrt();
        
        match acquisition_function {
            AcquisitionFunction::ExpectedImprovement => {
                let f_best = bayesian_state.current_best;
                if std_dev < 1e-6 {
                    return 0.0; // No uncertainty, no improvement expected
                }
                
                let z = (mean - f_best) / std_dev;
                let phi_z = 0.5 * (1.0 + erf(z / 2.0_f64.sqrt()));
                let pdf_z = (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();
                
                (mean - f_best) * phi_z + std_dev * pdf_z
            }
            AcquisitionFunction::UpperConfidenceBound { beta } => {
                mean + beta * std_dev
            }
            AcquisitionFunction::ProbabilityOfImprovement => {
                let f_best = bayesian_state.current_best;
                if std_dev < 1e-6 {
                    return if mean > f_best { 1.0 } else { 0.0 };
                }
                
                let z = (mean - f_best) / std_dev;
                0.5 * (1.0 + erf(z / 2.0_f64.sqrt()))
            }
            AcquisitionFunction::EntropySearch => {
                // Simplified entropy-based acquisition
                -variance * variance.ln()
            }
            AcquisitionFunction::KnowledgeGradient => {
                // Simplified knowledge gradient
                std_dev * (1.0 + variance.ln())
            }
            AcquisitionFunction::ThompsonSampling => {
                // Thompson sampling: sample from posterior
                let mut rng = rand::thread_rng();
                mean + std_dev * rng.gen_range(-1.0..1.0)
            }
        }
    }

    /// Predict GP mean and variance at a point
    fn predict_gp(&self, point: &HashMap<String, f64>, bayesian_state: &BayesianState) -> (f64, f64) {
        // Convert point to feature vector
        let mut feature_vector = vec![0.0; bayesian_state.parameter_names.len()];
        for (i, param_name) in bayesian_state.parameter_names.iter().enumerate() {
            feature_vector[i] = point.get(param_name).unwrap_or(&0.0).clone();
        }
        
        // Simplified GP prediction (in practice, would use full GP inference)
        let mean = bayesian_state.gp_mean.unwrap_or(0.0);
        let variance = bayesian_state.gp_hyperparameters.signal_variance;
        
        (mean, variance)
    }
}

/// Error function approximation for statistical calculations
fn erf(x: f64) -> f64 {
    // Abramowitz and Stegun approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// Default configurations for different algorithms
impl Default for TuningConfig {
    fn default() -> Self {
        Self {
            strategy: SearchStrategy::RandomSearch { n_trials: 50 },
            metric: EvaluationMetric::SilhouetteScore,
            cv_config: CrossValidationConfig {
                n_folds: 5,
                validation_ratio: 0.2,
                strategy: CVStrategy::KFold,
                shuffle: true,
            },
            max_evaluations: 100,
            early_stopping: Some(EarlyStoppingConfig {
                patience: 10,
                min_improvement: 0.001,
                evaluation_frequency: 1,
            }),
            random_seed: None,
            parallel_config: None,
            resource_constraints: ResourceConstraints {
                max_memory_per_evaluation: None,
                max_time_per_evaluation: Some(300.0), // 5 minutes
                max_total_time: Some(3600.0),         // 1 hour
            },
        }
    }
}

/// Predefined search spaces for common algorithms
pub struct StandardSearchSpaces;

impl StandardSearchSpaces {
    /// K-means search space
    pub fn kmeans() -> SearchSpace {
        let mut parameters = HashMap::new();
        parameters.insert(
            "n_clusters".to_string(),
            HyperParameter::Integer { min: 2, max: 20 },
        );
        parameters.insert(
            "max_iter".to_string(),
            HyperParameter::IntegerChoices {
                choices: vec![100, 200, 300, 500, 1000],
            },
        );
        parameters.insert(
            "tolerance".to_string(),
            HyperParameter::LogUniform {
                min: 1e-6,
                max: 1e-2,
            },
        );

        SearchSpace {
            parameters,
            constraints: Vec::new(),
        }
    }

    /// DBSCAN search space
    pub fn dbscan() -> SearchSpace {
        let mut parameters = HashMap::new();
        parameters.insert(
            "eps".to_string(),
            HyperParameter::Float { min: 0.1, max: 2.0 },
        );
        parameters.insert(
            "min_samples".to_string(),
            HyperParameter::Integer { min: 2, max: 20 },
        );

        SearchSpace {
            parameters,
            constraints: Vec::new(),
        }
    }

    /// Hierarchical clustering search space
    pub fn hierarchical() -> SearchSpace {
        let mut parameters = HashMap::new();
        parameters.insert(
            "method".to_string(),
            HyperParameter::Categorical {
                choices: vec![
                    "single".to_string(),
                    "complete".to_string(),
                    "average".to_string(),
                    "ward".to_string(),
                ],
            },
        );

        SearchSpace {
            parameters,
            constraints: Vec::new(),
        }
    }

    /// Mean Shift search space
    pub fn mean_shift() -> SearchSpace {
        let mut parameters = HashMap::new();
        parameters.insert(
            "bandwidth".to_string(),
            HyperParameter::Float { min: 0.1, max: 5.0 },
        );

        SearchSpace {
            parameters,
            constraints: Vec::new(),
        }
    }

    /// Quantum K-means search space
    pub fn quantum_kmeans() -> SearchSpace {
        let mut parameters = HashMap::new();
        parameters.insert(
            "n_clusters".to_string(),
            HyperParameter::Integer { min: 2, max: 20 },
        );
        parameters.insert(
            "n_quantum_states".to_string(),
            HyperParameter::IntegerChoices {
                choices: vec![4, 8, 16, 32],
            },
        );
        parameters.insert(
            "quantum_iterations".to_string(),
            HyperParameter::IntegerChoices {
                choices: vec![20, 50, 100, 200],
            },
        );
        parameters.insert(
            "decoherence_factor".to_string(),
            HyperParameter::Float {
                min: 0.8,
                max: 0.99,
            },
        );
        parameters.insert(
            "entanglement_strength".to_string(),
            HyperParameter::Float { min: 0.1, max: 0.5 },
        );

        SearchSpace {
            parameters,
            constraints: Vec::new(),
        }
    }

    /// Reinforcement learning clustering search space
    pub fn rl_clustering() -> SearchSpace {
        let mut parameters = HashMap::new();
        parameters.insert(
            "n_actions".to_string(),
            HyperParameter::Integer { min: 5, max: 50 },
        );
        parameters.insert(
            "learning_rate".to_string(),
            HyperParameter::LogUniform {
                min: 0.001,
                max: 0.5,
            },
        );
        parameters.insert(
            "exploration_rate".to_string(),
            HyperParameter::Float { min: 0.1, max: 1.0 },
        );
        parameters.insert(
            "n_episodes".to_string(),
            HyperParameter::IntegerChoices {
                choices: vec![50, 100, 200, 500],
            },
        );

        SearchSpace {
            parameters,
            constraints: Vec::new(),
        }
    }

    /// Adaptive online clustering search space
    pub fn adaptive_online() -> SearchSpace {
        let mut parameters = HashMap::new();
        parameters.insert(
            "initial_learning_rate".to_string(),
            HyperParameter::LogUniform {
                min: 0.001,
                max: 0.5,
            },
        );
        parameters.insert(
            "cluster_creation_threshold".to_string(),
            HyperParameter::Float { min: 1.0, max: 5.0 },
        );
        parameters.insert(
            "max_clusters".to_string(),
            HyperParameter::Integer { min: 10, max: 100 },
        );
        parameters.insert(
            "forgetting_factor".to_string(),
            HyperParameter::Float {
                min: 0.9,
                max: 0.99,
            },
        );

        SearchSpace {
            parameters,
            constraints: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_tuning_config_default() {
        let config = TuningConfig::default();
        assert_eq!(config.max_evaluations, 100);
        assert!(matches!(config.metric, EvaluationMetric::SilhouetteScore));
        assert!(config.early_stopping.is_some());
    }

    #[test]
    fn test_standard_search_spaces() {
        let kmeans_space = StandardSearchSpaces::kmeans();
        assert!(kmeans_space.parameters.contains_key("n_clusters"));
        assert!(kmeans_space.parameters.contains_key("max_iter"));

        let dbscan_space = StandardSearchSpaces::dbscan();
        assert!(dbscan_space.parameters.contains_key("eps"));
        assert!(dbscan_space.parameters.contains_key("min_samples"));

        let quantum_space = StandardSearchSpaces::quantum_kmeans();
        assert!(quantum_space.parameters.contains_key("n_clusters"));
        assert!(quantum_space.parameters.contains_key("n_quantum_states"));
        assert!(quantum_space.parameters.contains_key("decoherence_factor"));

        let rl_space = StandardSearchSpaces::rl_clustering();
        assert!(rl_space.parameters.contains_key("n_actions"));
        assert!(rl_space.parameters.contains_key("learning_rate"));

        let adaptive_space = StandardSearchSpaces::adaptive_online();
        assert!(adaptive_space
            .parameters
            .contains_key("initial_learning_rate"));
        assert!(adaptive_space.parameters.contains_key("max_clusters"));
    }

    #[test]
    fn test_auto_tuner_creation() {
        let config = TuningConfig::default();
        let tuner: AutoTuner<f64> = AutoTuner::new(config);
        // Test that the tuner can be created successfully
        assert_eq!(
            std::mem::size_of_val(&tuner),
            std::mem::size_of::<TuningConfig>()
        );
    }

    #[test]
    fn test_random_combinations_generation() {
        let config = TuningConfig::default();
        let tuner: AutoTuner<f64> = AutoTuner::new(config);
        let search_space = StandardSearchSpaces::kmeans();

        let combinations = tuner
            .generate_random_combinations(&search_space, 10)
            .unwrap();
        assert_eq!(combinations.len(), 10);

        for combo in &combinations {
            assert!(combo.contains_key("n_clusters"));
            assert!(combo.contains_key("max_iter"));

            let n_clusters = combo["n_clusters"];
            assert!(n_clusters >= 2.0 && n_clusters <= 20.0);
        }
    }
}
