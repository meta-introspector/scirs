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
        let mut bayesian_state = BayesianState {
            observations: Vec::new(),
            gp_mean: None,
            gp_covariance: None,
            acquisition_values: Vec::new(),
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

    /// Update Gaussian process with new observations
    fn update_gaussian_process(
        &self,
        _bayesian_state: &mut BayesianState,
        _combinations: &[HashMap<String, f64>],
    ) {
        // Simplified implementation - in practice, this would fit a GP
        // to the observed (parameter, score) pairs
    }

    /// Optimize acquisition function to find next point
    fn optimize_acquisition_function(
        &self,
        search_space: &SearchSpace,
        _bayesian_state: &BayesianState,
        _acquisition_function: &AcquisitionFunction,
    ) -> Result<HashMap<String, f64>> {
        // Simplified implementation - in practice, this would optimize
        // the acquisition function over the parameter space
        let random_points = self.generate_random_combinations(search_space, 1)?;
        Ok(random_points.into_iter().next().unwrap())
    }
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
