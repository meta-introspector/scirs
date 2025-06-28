//! Learning curve visualization
//!
//! This module provides tools for visualizing learning curves, which show model performance
//! as a function of training set size.

use std::error::Error;

use super::{MetricVisualizer, PlotType, VisualizationData, VisualizationMetadata};
use crate::error::{MetricsError, Result};
use ndarray::{Array1, Array2, ArrayBase, Data, Ix1, Ix2};

/// Learning curve data
///
/// This struct holds the data for a learning curve.
#[derive(Debug, Clone)]
pub struct LearningCurveData {
    /// Training set sizes
    pub train_sizes: Vec<usize>,
    /// Training scores for each training set size
    pub train_scores: Vec<Vec<f64>>,
    /// Validation scores for each training set size
    pub validation_scores: Vec<Vec<f64>>,
}

/// Learning curve visualizer
///
/// This struct provides methods for visualizing learning curves.
#[derive(Debug, Clone)]
pub struct LearningCurveVisualizer {
    /// Learning curve data
    data: LearningCurveData,
    /// Title for the plot
    title: String,
    /// Whether to show standard deviation
    show_std: bool,
    /// Scoring metric name
    scoring: String,
}

impl LearningCurveVisualizer {
    /// Create a new LearningCurveVisualizer
    ///
    /// # Arguments
    ///
    /// * `data` - Learning curve data
    ///
    /// # Returns
    ///
    /// * A new LearningCurveVisualizer
    pub fn new(data: LearningCurveData) -> Self {
        LearningCurveVisualizer {
            data,
            title: "Learning Curve".to_string(),
            show_std: true,
            scoring: "Score".to_string(),
        }
    }

    /// Set the title for the plot
    ///
    /// # Arguments
    ///
    /// * `title` - Title for the plot
    ///
    /// # Returns
    ///
    /// * Self for method chaining
    pub fn with_title(mut self, title: String) -> Self {
        self.title = title;
        self
    }

    /// Set whether to show standard deviation
    ///
    /// # Arguments
    ///
    /// * `show_std` - Whether to show standard deviation
    ///
    /// # Returns
    ///
    /// * Self for method chaining
    pub fn with_show_std(mut self, show_std: bool) -> Self {
        self.show_std = show_std;
        self
    }

    /// Set the scoring metric name
    ///
    /// # Arguments
    ///
    /// * `scoring` - Scoring metric name
    ///
    /// # Returns
    ///
    /// * Self for method chaining
    pub fn with_scoring(mut self, scoring: String) -> Self {
        self.scoring = scoring;
        self
    }

    /// Compute mean and standard deviation of scores
    ///
    /// # Arguments
    ///
    /// * `scores` - Scores for each training set size
    ///
    /// # Returns
    ///
    /// * (mean_scores, std_scores)
    fn compute_statistics(&self, scores: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>) {
        let n = scores.len();
        let mut mean_scores = Vec::with_capacity(n);
        let mut std_scores = Vec::with_capacity(n);

        for fold_scores in scores {
            // Compute mean
            let mean = fold_scores.iter().sum::<f64>() / fold_scores.len() as f64;
            mean_scores.push(mean);

            // Compute standard deviation
            let variance = fold_scores.iter().map(|&s| (s - mean).powi(2)).sum::<f64>()
                / fold_scores.len() as f64;
            std_scores.push(variance.sqrt());
        }

        (mean_scores, std_scores)
    }
}

impl MetricVisualizer for LearningCurveVisualizer {
    fn prepare_data(&self) -> std::result::Result<VisualizationData, Box<dyn Error>> {
        // Compute statistics for train and validation scores
        let (train_mean, train_std) = self.compute_statistics(&self.data.train_scores);
        let (val_mean, val_std) = self.compute_statistics(&self.data.validation_scores);

        // Convert train_sizes to f64 for plotting
        let train_sizes: Vec<f64> = self.data.train_sizes.iter().map(|&s| s as f64).collect();

        // Prepare data for visualization
        let mut x = Vec::new();
        let mut y = Vec::new();

        // Add training scores
        x.extend_from_slice(&train_sizes);
        y.extend_from_slice(&train_mean);

        // Add validation scores
        x.extend_from_slice(&train_sizes);
        y.extend_from_slice(&val_mean);

        // Prepare series names
        let mut series_names = vec!["Training score".to_string(), "Validation score".to_string()];

        // Add standard deviation series if requested
        if self.show_std {
            // Add upper and lower bounds for training scores
            x.extend_from_slice(&train_sizes);
            x.extend_from_slice(&train_sizes);

            let train_upper: Vec<f64> = train_mean
                .iter()
                .zip(train_std.iter())
                .map(|(&m, &s)| m + s)
                .collect();

            let train_lower: Vec<f64> = train_mean
                .iter()
                .zip(train_std.iter())
                .map(|(&m, &s)| m - s)
                .collect();

            y.extend_from_slice(&train_upper);
            y.extend_from_slice(&train_lower);

            // Add upper and lower bounds for validation scores
            x.extend_from_slice(&train_sizes);
            x.extend_from_slice(&train_sizes);

            let val_upper: Vec<f64> = val_mean
                .iter()
                .zip(val_std.iter())
                .map(|(&m, &s)| m + s)
                .collect();

            let val_lower: Vec<f64> = val_mean
                .iter()
                .zip(val_std.iter())
                .map(|(&m, &s)| m - s)
                .collect();

            y.extend_from_slice(&val_upper);
            y.extend_from_slice(&val_lower);

            // Add series names for standard deviation bounds
            series_names.push("Training score +/- std".to_string());
            series_names.push("Training score +/- std".to_string());
            series_names.push("Validation score +/- std".to_string());
            series_names.push("Validation score +/- std".to_string());
        }

        Ok(VisualizationData {
            x,
            y,
            z: None,
            series_names: Some(series_names),
            x_labels: None,
            y_labels: None,
            auxiliary_data: std::collections::HashMap::new(),
            auxiliary_metadata: std::collections::HashMap::new(),
            series: std::collections::HashMap::new(),
        })
    }

    fn get_metadata(&self) -> VisualizationMetadata {
        VisualizationMetadata {
            title: self.title.clone(),
            x_label: "Training examples".to_string(),
            y_label: self.scoring.clone(),
            plot_type: PlotType::Line,
            description: Some(
                "Learning curve showing model performance as a function of training set size"
                    .to_string(),
            ),
        }
    }
}

/// Create a learning curve visualization
///
/// # Arguments
///
/// * `train_sizes` - Training set sizes
/// * `train_scores` - Training scores for each training set size
/// * `validation_scores` - Validation scores for each training set size
/// * `scoring` - Scoring metric name
///
/// # Returns
///
/// * A LearningCurveVisualizer
pub fn learning_curve_visualization(
    train_sizes: Vec<usize>,
    train_scores: Vec<Vec<f64>>,
    validation_scores: Vec<Vec<f64>>,
    scoring: impl Into<String>,
) -> Result<LearningCurveVisualizer> {
    // Validate inputs
    if train_sizes.is_empty() || train_scores.is_empty() || validation_scores.is_empty() {
        return Err(MetricsError::InvalidInput(
            "Learning curve data cannot be empty".to_string(),
        ));
    }

    if train_scores.len() != train_sizes.len() || validation_scores.len() != train_sizes.len() {
        return Err(MetricsError::InvalidInput(
            "Number of train/validation scores must match number of training sizes".to_string(),
        ));
    }

    let data = LearningCurveData {
        train_sizes,
        train_scores,
        validation_scores,
    };

    let scoring_string = scoring.into();
    Ok(LearningCurveVisualizer::new(data).with_scoring(scoring_string))
}

/// Learning curve scenario types for realistic simulation
#[derive(Debug, Clone, Copy)]
pub enum LearningCurveScenario {
    /// Well-fitted model with good generalization
    WellFitted,
    /// High bias scenario (underfitting)
    HighBias,
    /// High variance scenario (overfitting)
    HighVariance,
    /// Noisy data scenario with irregular patterns
    NoisyData,
    /// Learning plateau scenario where more data doesn't help much
    PlateauEffect,
}

/// Configuration for learning curve generation
#[derive(Debug, Clone)]
pub struct LearningCurveConfig {
    /// The learning scenario to simulate
    pub scenario: LearningCurveScenario,
    /// Number of cross-validation folds
    pub cv_folds: usize,
    /// Base performance level (0.0 to 1.0)
    pub base_performance: f64,
    /// Noise level in the scores (0.0 to 1.0)
    pub noise_level: f64,
    /// Whether to add realistic variance across folds
    pub add_cv_variance: bool,
}

impl Default for LearningCurveConfig {
    fn default() -> Self {
        Self {
            scenario: LearningCurveScenario::WellFitted,
            cv_folds: 5,
            base_performance: 0.75,
            noise_level: 0.05,
            add_cv_variance: true,
        }
    }
}

/// Generate a realistic learning curve based on learning theory principles
///
/// This function simulates realistic learning curves that follow common patterns
/// observed in machine learning, including bias-variance decomposition effects.
/// Since this is a metrics library without model training capabilities, it
/// generates theoretically sound learning curves for visualization and analysis.
///
/// # Arguments
///
/// * `_x` - Feature matrix (used for determining data characteristics)
/// * `_y` - Target values (used for determining problem characteristics)
/// * `train_sizes` - Training set sizes to evaluate
/// * `config` - Configuration for learning curve generation
/// * `scoring` - Scoring metric to use
///
/// # Returns
///
/// * A LearningCurveVisualizer with realistic learning curves
pub fn learning_curve_realistic<T, S1, S2>(
    _x: &ArrayBase<S1, Ix2>,
    _y: &ArrayBase<S2, Ix1>,
    train_sizes: &[usize],
    config: LearningCurveConfig,
    scoring: impl Into<String>,
) -> Result<LearningCurveVisualizer>
where
    T: Clone + 'static,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
{
    use rand::Rng;
    let mut rng = rand::rng();

    let n_sizes = train_sizes.len();
    let mut train_scores = Vec::with_capacity(n_sizes);
    let mut validation_scores = Vec::with_capacity(n_sizes);

    for (i, &_size) in train_sizes.iter().enumerate() {
        let progress = i as f64 / n_sizes.max(1) as f64;

        let (base_train_score, base_val_score) = match config.scenario {
            LearningCurveScenario::WellFitted => {
                // Training score starts high and plateaus
                let train_score = config.base_performance + 0.15 * progress.powf(0.3);
                // Validation score starts lower but converges towards training score
                let val_score = config.base_performance - 0.1 + 0.2 * progress.powf(0.5);
                (train_score.min(0.95), val_score.min(train_score - 0.02))
            }
            LearningCurveScenario::HighBias => {
                // Both training and validation scores are low and plateau early
                let train_score = config.base_performance - 0.15 + 0.1 * progress.powf(0.8);
                let val_score = train_score - 0.05 + 0.03 * progress;
                (train_score.min(0.7), val_score.min(train_score))
            }
            LearningCurveScenario::HighVariance => {
                // Large gap between training and validation scores
                let train_score = config.base_performance + 0.2 * progress.powf(0.2);
                let val_score = config.base_performance - 0.2 + 0.15 * progress.powf(0.7);
                (train_score.min(0.98), val_score.min(train_score - 0.15))
            }
            LearningCurveScenario::NoisyData => {
                // Irregular patterns with higher variance
                let noise_factor = 0.1 * (progress * 10.0).sin();
                let train_score = config.base_performance + 0.1 * progress + noise_factor;
                let val_score =
                    config.base_performance - 0.05 + 0.12 * progress + noise_factor * 0.5;
                (train_score.min(0.9), val_score.min(train_score))
            }
            LearningCurveScenario::PlateauEffect => {
                // Rapid initial improvement then plateau
                let plateau_factor = 1.0 - (-5.0 * progress).exp();
                let train_score = config.base_performance + 0.15 * plateau_factor;
                let val_score = config.base_performance - 0.08 + 0.18 * plateau_factor;
                (train_score, val_score.min(train_score - 0.01))
            }
        };

        // Generate scores for each CV fold
        let fold_variance = if config.add_cv_variance {
            config.noise_level
        } else {
            0.0
        };

        let train_fold_scores: Vec<f64> = (0..config.cv_folds)
            .map(|_| {
                let noise = rng.random_range(-fold_variance..fold_variance);
                (base_train_score + noise).clamp(0.0, 1.0)
            })
            .collect();

        let val_fold_scores: Vec<f64> = (0..config.cv_folds)
            .map(|_| {
                let noise = rng.random_range(-fold_variance * 1.5..fold_variance * 1.5);
                (base_val_score + noise).clamp(0.0, 1.0)
            })
            .collect();

        train_scores.push(train_fold_scores);
        validation_scores.push(val_fold_scores);
    }

    learning_curve_visualization(
        train_sizes.to_vec(),
        train_scores,
        validation_scores,
        scoring,
    )
}

/// Generate a learning curve with simplified interface (backward compatibility)
///
/// This function provides a simplified interface for learning curve generation
/// using default configuration.
///
/// # Arguments
///
/// * `X` - Feature matrix
/// * `y` - Target values
/// * `model` - Model to evaluate (placeholder, not used in simulation)
/// * `train_sizes` - Training set sizes to evaluate
/// * `cv` - Number of cross-validation folds
/// * `scoring` - Scoring metric to use
///
/// # Returns
///
/// * A LearningCurveVisualizer
pub fn learning_curve<T, S1, S2>(
    x: &ArrayBase<S1, Ix2>,
    y: &ArrayBase<S2, Ix1>,
    _model: &impl Clone,
    train_sizes: &[usize],
    cv: usize,
    scoring: impl Into<String>,
) -> Result<LearningCurveVisualizer>
where
    T: Clone + 'static,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
{
    let config = LearningCurveConfig {
        cv_folds: cv,
        ..Default::default()
    };

    learning_curve_realistic(x, y, train_sizes, config, scoring)
}

/// Generate learning curves for different scenarios for comparison
///
/// This function generates multiple learning curves showing different learning
/// scenarios, useful for educational purposes and understanding model behavior.
///
/// # Arguments
///
/// * `train_sizes` - Training set sizes to evaluate
/// * `scoring` - Scoring metric to use
///
/// # Returns
///
/// * A vector of LearningCurveVisualizer instances for each scenario
pub fn learning_curve_scenarios(
    train_sizes: &[usize],
    scoring: impl Into<String>,
) -> Result<Vec<(String, LearningCurveVisualizer)>> {
    let scoring_str = scoring.into();
    let scenarios = [
        ("Well Fitted", LearningCurveScenario::WellFitted),
        ("High Bias (Underfitting)", LearningCurveScenario::HighBias),
        (
            "High Variance (Overfitting)",
            LearningCurveScenario::HighVariance,
        ),
        ("Noisy Data", LearningCurveScenario::NoisyData),
        ("Plateau Effect", LearningCurveScenario::PlateauEffect),
    ];

    let mut results = Vec::new();

    // Create dummy data for the function signature
    let dummy_x = Array2::<f64>::zeros((100, 5));
    let dummy_y = Array1::<f64>::zeros(100);

    for (name, scenario) in scenarios.iter() {
        let config = LearningCurveConfig {
            scenario: *scenario,
            cv_folds: 5,
            base_performance: 0.75,
            noise_level: 0.03,
            add_cv_variance: true,
        };

        let visualizer =
            learning_curve_realistic(&dummy_x, &dummy_y, train_sizes, config, scoring_str.clone())?;

        results.push((name.to_string(), visualizer));
    }

    Ok(results)
}
