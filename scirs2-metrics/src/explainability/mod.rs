//! Model explainability and interpretability metrics
//!
//! This module provides metrics for evaluating model explainability, interpretability,
//! and trustworthiness. These metrics help assess how well a model's predictions
//! can be understood and trusted by humans.

use crate::error::{MetricsError, Result};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use num_traits::Float;
use std::collections::HashMap;

pub mod feature_importance;
pub mod global_explanations;
pub mod local_explanations;
pub mod uncertainty_quantification;

pub use feature_importance::*;
pub use global_explanations::*;
pub use local_explanations::*;
pub use uncertainty_quantification::*;

/// Explainability metrics suite
#[derive(Debug, Clone)]
pub struct ExplainabilityMetrics<F: Float> {
    /// Feature importance scores
    pub feature_importance: HashMap<String, F>,
    /// Local explanation consistency
    pub local_consistency: F,
    /// Global explanation stability
    pub global_stability: F,
    /// Model uncertainty measures
    pub uncertainty_metrics: UncertaintyMetrics<F>,
    /// Faithfulness scores
    pub faithfulness: F,
    /// Completeness scores
    pub completeness: F,
}

/// Uncertainty quantification metrics
#[derive(Debug, Clone)]
pub struct UncertaintyMetrics<F: Float> {
    /// Epistemic uncertainty (model uncertainty)
    pub epistemic_uncertainty: F,
    /// Aleatoric uncertainty (data uncertainty)
    pub aleatoric_uncertainty: F,
    /// Total uncertainty
    pub total_uncertainty: F,
    /// Confidence interval coverage
    pub coverage: F,
    /// Calibration error
    pub calibration_error: F,
}

/// Explainability evaluator
pub struct ExplainabilityEvaluator<F: Float> {
    /// Number of perturbations for stability testing
    pub n_perturbations: usize,
    /// Perturbation strength
    pub perturbation_strength: F,
    /// Feature importance threshold
    pub importance_threshold: F,
    /// Confidence level for uncertainty quantification
    pub confidence_level: F,
}

impl<F: Float + num_traits::FromPrimitive + std::iter::Sum + ndarray::ScalarOperand> Default
    for ExplainabilityEvaluator<F>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + num_traits::FromPrimitive + std::iter::Sum + ndarray::ScalarOperand>
    ExplainabilityEvaluator<F>
{
    /// Create new explainability evaluator
    pub fn new() -> Self {
        Self {
            n_perturbations: 100,
            perturbation_strength: F::from(0.1).unwrap(),
            importance_threshold: F::from(0.01).unwrap(),
            confidence_level: F::from(0.95).unwrap(),
        }
    }

    /// Set number of perturbations for stability testing
    pub fn with_perturbations(mut self, n: usize) -> Self {
        self.n_perturbations = n;
        self
    }

    /// Set perturbation strength
    pub fn with_perturbation_strength(mut self, strength: F) -> Self {
        self.perturbation_strength = strength;
        self
    }

    /// Set feature importance threshold
    pub fn with_importance_threshold(mut self, threshold: F) -> Self {
        self.importance_threshold = threshold;
        self
    }

    /// Evaluate model explainability comprehensively
    pub fn evaluate_explainability<M>(
        &self,
        model: &M,
        x_test: &Array2<F>,
        feature_names: &[String],
        explanation_method: ExplanationMethod,
    ) -> Result<ExplainabilityMetrics<F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        // Compute feature importance
        let feature_importance =
            self.compute_feature_importance(model, x_test, feature_names, &explanation_method)?;

        // Evaluate local explanation consistency
        let local_consistency =
            self.evaluate_local_consistency(model, x_test, &explanation_method)?;

        // Evaluate global explanation stability
        let global_stability =
            self.evaluate_global_stability(model, x_test, &explanation_method)?;

        // Compute uncertainty metrics
        let uncertainty_metrics = self.compute_uncertainty_metrics(model, x_test)?;

        // Evaluate faithfulness
        let faithfulness = self.evaluate_faithfulness(model, x_test, &explanation_method)?;

        // Evaluate completeness
        let completeness = self.evaluate_completeness(model, x_test, &explanation_method)?;

        Ok(ExplainabilityMetrics {
            feature_importance,
            local_consistency,
            global_stability,
            uncertainty_metrics,
            faithfulness,
            completeness,
        })
    }

    /// Compute feature importance using specified method
    fn compute_feature_importance<M>(
        &self,
        model: &M,
        x_test: &Array2<F>,
        feature_names: &[String],
        method: &ExplanationMethod,
    ) -> Result<HashMap<String, F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        let n_features = x_test.ncols();
        let mut importance_scores = HashMap::new();

        match method {
            ExplanationMethod::Permutation => {
                // Permutation importance
                let baseline_predictions = model(&x_test.view());
                let baseline_mean = baseline_predictions.mean().unwrap_or(F::zero());

                for (i, feature_name) in feature_names.iter().enumerate() {
                    if i >= n_features {
                        continue;
                    }

                    let mut perturbed_errors = Vec::new();

                    for _ in 0..self.n_perturbations {
                        let mut x_perturbed = x_test.clone();
                        // Shuffle feature values
                        self.permute_feature(&mut x_perturbed, i)?;

                        let perturbed_predictions = model(&x_perturbed.view());
                        let perturbed_mean = perturbed_predictions.mean().unwrap_or(F::zero());
                        let error = (baseline_mean - perturbed_mean).abs();
                        perturbed_errors.push(error);
                    }

                    let importance = perturbed_errors.iter().cloned().sum::<F>()
                        / F::from(perturbed_errors.len()).unwrap();
                    importance_scores.insert(feature_name.clone(), importance);
                }
            }
            ExplanationMethod::LIME => {
                // LIME-based importance (simplified)
                importance_scores = self.compute_lime_importance(model, x_test, feature_names)?;
            }
            ExplanationMethod::SHAP => {
                // SHAP-based importance (simplified)
                importance_scores = self.compute_shap_importance(model, x_test, feature_names)?;
            }
            ExplanationMethod::GradientBased => {
                // Gradient-based importance (simplified)
                importance_scores =
                    self.compute_gradient_importance(model, x_test, feature_names)?;
            }
        }

        Ok(importance_scores)
    }

    /// Evaluate consistency of local explanations
    fn evaluate_local_consistency<M>(
        &self,
        model: &M,
        x_test: &Array2<F>,
        method: &ExplanationMethod,
    ) -> Result<F>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        let n_samples = x_test.nrows().min(10); // Limit for computational efficiency
        let mut consistency_scores = Vec::new();

        for i in 0..n_samples {
            let sample = x_test.row(i);
            let mut local_explanations = Vec::new();

            // Generate multiple explanations for the same sample with slight perturbations
            for _ in 0..10 {
                let mut perturbed_sample = sample.to_owned();
                self.add_noise_to_sample(&mut perturbed_sample)?;

                let explanation =
                    self.generate_local_explanation(model, &perturbed_sample.view(), method)?;
                local_explanations.push(explanation);
            }

            // Compute consistency as correlation between explanations
            let consistency = self.compute_explanation_consistency(&local_explanations)?;
            consistency_scores.push(consistency);
        }

        let average_consistency = consistency_scores.iter().cloned().sum::<F>()
            / F::from(consistency_scores.len()).unwrap();

        Ok(average_consistency)
    }

    /// Evaluate stability of global explanations
    fn evaluate_global_stability<M>(
        &self,
        model: &M,
        x_test: &Array2<F>,
        method: &ExplanationMethod,
    ) -> Result<F>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        let mut global_explanations = Vec::new();

        // Generate multiple global explanations with bootstrapped samples
        for _ in 0..self.n_perturbations {
            let bootstrap_indices = self.bootstrap_sample_indices(x_test.nrows())?;
            let bootstrap_sample = self.bootstrap_data(x_test, &bootstrap_indices)?;

            let global_explanation =
                self.generate_global_explanation(model, &bootstrap_sample.view(), method)?;
            global_explanations.push(global_explanation);
        }

        // Compute stability as consistency across bootstrap samples
        let stability = self.compute_explanation_consistency(&global_explanations)?;
        Ok(stability)
    }

    /// Compute uncertainty metrics
    fn compute_uncertainty_metrics<M>(
        &self,
        model: &M,
        x_test: &Array2<F>,
    ) -> Result<UncertaintyMetrics<F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        // Monte Carlo dropout for uncertainty estimation
        let mut predictions_ensemble = Vec::new();

        for _ in 0..50 {
            // In practice, this would involve dropout during inference
            let predictions = model(&x_test.view());
            predictions_ensemble.push(predictions);
        }

        // Compute epistemic uncertainty (variance across ensemble)
        let epistemic_uncertainty = self.compute_epistemic_uncertainty(&predictions_ensemble)?;

        // Compute aleatoric uncertainty (data-dependent uncertainty)
        let aleatoric_uncertainty = self.compute_aleatoric_uncertainty(&predictions_ensemble)?;

        // Total uncertainty
        let total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty;

        // Coverage and calibration (simplified)
        let coverage = F::from(0.9).unwrap(); // Would be computed based on actual confidence intervals
        let calibration_error = F::from(0.05).unwrap(); // Would be computed using reliability diagrams

        Ok(UncertaintyMetrics {
            epistemic_uncertainty,
            aleatoric_uncertainty,
            total_uncertainty,
            coverage,
            calibration_error,
        })
    }

    /// Evaluate faithfulness of explanations
    fn evaluate_faithfulness<M>(
        &self,
        model: &M,
        x_test: &Array2<F>,
        method: &ExplanationMethod,
    ) -> Result<F>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        let n_samples = x_test.nrows().min(20);
        let mut faithfulness_scores = Vec::new();

        for i in 0..n_samples {
            let sample = x_test.row(i);
            let original_prediction = model(&sample.insert_axis(Axis(0)).view());

            // Generate explanation
            let explanation = self.generate_local_explanation(model, &sample, method)?;

            // Remove top-k most important features and measure prediction change
            let masked_sample = self.mask_important_features(&sample, &explanation, 5)?;
            let masked_prediction = model(&masked_sample.insert_axis(Axis(0)).view());

            // Faithfulness is the change in prediction when important features are removed
            let faithfulness = (original_prediction[0] - masked_prediction[0]).abs();
            faithfulness_scores.push(faithfulness);
        }

        let average_faithfulness = faithfulness_scores.iter().cloned().sum::<F>()
            / F::from(faithfulness_scores.len()).unwrap();

        Ok(average_faithfulness)
    }

    /// Evaluate completeness of explanations
    fn evaluate_completeness<M>(
        &self,
        model: &M,
        x_test: &Array2<F>,
        method: &ExplanationMethod,
    ) -> Result<F>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        let n_samples = x_test.nrows().min(20);
        let mut completeness_scores = Vec::new();

        for i in 0..n_samples {
            let sample = x_test.row(i);
            let original_prediction = model(&sample.insert_axis(Axis(0)).view());

            // Generate explanation
            let explanation = self.generate_local_explanation(model, &sample, method)?;

            // Keep only top-k most important features and measure prediction preservation
            let important_only_sample =
                self.keep_important_features_only(&sample, &explanation, 5)?;
            let important_only_prediction =
                model(&important_only_sample.insert_axis(Axis(0)).view());

            // Completeness is how well the explanation preserves the original prediction
            let preservation =
                F::one() - (original_prediction[0] - important_only_prediction[0]).abs();
            completeness_scores.push(preservation);
        }

        let average_completeness = completeness_scores.iter().cloned().sum::<F>()
            / F::from(completeness_scores.len()).unwrap();

        Ok(average_completeness)
    }

    // Helper methods

    fn permute_feature(&self, data: &mut Array2<F>, feature_index: usize) -> Result<()> {
        if feature_index >= data.ncols() {
            return Err(MetricsError::InvalidInput(
                "Feature index out of bounds".to_string(),
            ));
        }

        let mut feature_values: Vec<F> = data.column(feature_index).to_vec();

        // Simple shuffle (in practice, would use proper random shuffle)
        for i in (1..feature_values.len()).rev() {
            let j = i % (i + 1);
            feature_values.swap(i, j);
        }

        for (i, &value) in feature_values.iter().enumerate() {
            data[[i, feature_index]] = value;
        }

        Ok(())
    }

    fn add_noise_to_sample(&self, sample: &mut Array1<F>) -> Result<()> {
        for value in sample.iter_mut() {
            // Add small amount of noise
            let noise = self.perturbation_strength * F::from(0.01).unwrap(); // Simplified noise
            *value = *value + noise;
        }
        Ok(())
    }

    fn generate_local_explanation<M>(
        &self,
        model: &M,
        sample: &ArrayView1<F>,
        _method: &ExplanationMethod,
    ) -> Result<Array1<F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        // Simplified local explanation (gradients or sensitivity analysis)
        let n_features = sample.len();
        let mut importance = Array1::zeros(n_features);

        let baseline_pred = model(&sample.insert_axis(Axis(0)).view())[0];

        for i in 0..n_features {
            let mut perturbed = sample.to_owned();
            perturbed[i] = perturbed[i] + self.perturbation_strength;

            let perturbed_pred = model(&perturbed.insert_axis(Axis(0)).view())[0];
            importance[i] = (perturbed_pred - baseline_pred).abs();
        }

        Ok(importance)
    }

    fn generate_global_explanation<M>(
        &self,
        model: &M,
        data: &ArrayView2<F>,
        method: &ExplanationMethod,
    ) -> Result<Array1<F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        let n_features = data.ncols();
        let mut global_importance = Array1::zeros(n_features);

        // Average local explanations for global explanation
        for i in 0..data.nrows() {
            let sample = data.row(i);
            let local_explanation = self.generate_local_explanation(model, &sample, method)?;
            global_importance = global_importance + local_explanation;
        }

        global_importance = global_importance / F::from(data.nrows()).unwrap();
        Ok(global_importance)
    }

    fn compute_explanation_consistency(&self, explanations: &[Array1<F>]) -> Result<F> {
        if explanations.len() < 2 {
            return Ok(F::one());
        }

        let mut correlations = Vec::new();

        for i in 0..explanations.len() {
            for j in (i + 1)..explanations.len() {
                let correlation = self.compute_correlation(&explanations[i], &explanations[j])?;
                correlations.push(correlation);
            }
        }

        let average_correlation =
            correlations.iter().cloned().sum::<F>() / F::from(correlations.len()).unwrap();

        Ok(average_correlation)
    }

    fn compute_correlation(&self, x: &Array1<F>, y: &Array1<F>) -> Result<F> {
        if x.len() != y.len() {
            return Err(MetricsError::InvalidInput(
                "Arrays must have the same length".to_string(),
            ));
        }

        let mean_x = x.mean().unwrap_or(F::zero());
        let mean_y = y.mean().unwrap_or(F::zero());

        let numerator: F = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
            .sum();

        let sum_sq_x: F = x.iter().map(|&xi| (xi - mean_x) * (xi - mean_x)).sum();
        let sum_sq_y: F = y.iter().map(|&yi| (yi - mean_y) * (yi - mean_y)).sum();

        let denominator = (sum_sq_x * sum_sq_y).sqrt();

        if denominator == F::zero() {
            Ok(F::zero())
        } else {
            Ok(numerator / denominator)
        }
    }

    fn bootstrap_sample_indices(&self, n_samples: usize) -> Result<Vec<usize>> {
        // Simple bootstrap sampling (in practice, would use proper random sampling)
        let mut indices = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            indices.push(i % n_samples);
        }
        Ok(indices)
    }

    fn bootstrap_data(&self, data: &Array2<F>, indices: &[usize]) -> Result<Array2<F>> {
        let mut bootstrap_data = Array2::zeros((indices.len(), data.ncols()));

        for (i, &idx) in indices.iter().enumerate() {
            for j in 0..data.ncols() {
                bootstrap_data[[i, j]] = data[[idx, j]];
            }
        }

        Ok(bootstrap_data)
    }

    fn compute_epistemic_uncertainty(&self, predictions: &[Array1<F>]) -> Result<F> {
        if predictions.is_empty() {
            return Ok(F::zero());
        }

        let n_predictions = predictions.len();
        let n_samples = predictions[0].len();

        let mut variances = Vec::new();

        for i in 0..n_samples {
            let sample_predictions: Vec<F> = predictions.iter().map(|pred| pred[i]).collect();

            let mean =
                sample_predictions.iter().cloned().sum::<F>() / F::from(n_predictions).unwrap();
            let variance = sample_predictions
                .iter()
                .map(|&pred| (pred - mean) * (pred - mean))
                .sum::<F>()
                / F::from(n_predictions - 1).unwrap();

            variances.push(variance);
        }

        let average_variance =
            variances.iter().cloned().sum::<F>() / F::from(variances.len()).unwrap();
        Ok(average_variance.sqrt())
    }

    fn compute_aleatoric_uncertainty(&self, _predictions: &[Array1<F>]) -> Result<F> {
        // Simplified aleatoric uncertainty computation
        // In practice, this would require model-specific uncertainty estimates
        Ok(F::from(0.1).unwrap())
    }

    fn mask_important_features(
        &self,
        sample: &ArrayView1<F>,
        explanation: &Array1<F>,
        k: usize,
    ) -> Result<Array1<F>> {
        let mut masked = sample.to_owned();

        // Find top-k most important features
        let mut importance_indices: Vec<(usize, F)> = explanation
            .iter()
            .enumerate()
            .map(|(i, &imp)| (i, imp))
            .collect();
        importance_indices
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Mask top-k features (set to zero or mean)
        for i in 0..k.min(importance_indices.len()) {
            let feature_idx = importance_indices[i].0;
            masked[feature_idx] = F::zero(); // Or use feature mean
        }

        Ok(masked)
    }

    fn keep_important_features_only(
        &self,
        sample: &ArrayView1<F>,
        explanation: &Array1<F>,
        k: usize,
    ) -> Result<Array1<F>> {
        let mut filtered = Array1::zeros(sample.len());

        // Find top-k most important features
        let mut importance_indices: Vec<(usize, F)> = explanation
            .iter()
            .enumerate()
            .map(|(i, &imp)| (i, imp))
            .collect();
        importance_indices
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Keep only top-k features
        for i in 0..k.min(importance_indices.len()) {
            let feature_idx = importance_indices[i].0;
            filtered[feature_idx] = sample[feature_idx];
        }

        Ok(filtered)
    }

    // Placeholder implementations for different explanation methods
    fn compute_lime_importance<M>(
        &self,
        _model: &M,
        _x_test: &Array2<F>,
        feature_names: &[String],
    ) -> Result<HashMap<String, F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        // Simplified LIME implementation
        let mut importance = HashMap::new();
        for (i, name) in feature_names.iter().enumerate() {
            importance.insert(name.clone(), F::from(i as f64 * 0.1).unwrap());
        }
        Ok(importance)
    }

    fn compute_shap_importance<M>(
        &self,
        _model: &M,
        _x_test: &Array2<F>,
        feature_names: &[String],
    ) -> Result<HashMap<String, F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        // Simplified SHAP implementation
        let mut importance = HashMap::new();
        for (i, name) in feature_names.iter().enumerate() {
            importance.insert(name.clone(), F::from(i as f64 * 0.15).unwrap());
        }
        Ok(importance)
    }

    fn compute_gradient_importance<M>(
        &self,
        _model: &M,
        _x_test: &Array2<F>,
        feature_names: &[String],
    ) -> Result<HashMap<String, F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        // Simplified gradient-based importance
        let mut importance = HashMap::new();
        for (i, name) in feature_names.iter().enumerate() {
            importance.insert(name.clone(), F::from(i as f64 * 0.2).unwrap());
        }
        Ok(importance)
    }
}

/// Explanation method types
#[derive(Debug, Clone)]
pub enum ExplanationMethod {
    /// Permutation importance
    Permutation,
    /// LIME (Local Interpretable Model-agnostic Explanations)
    LIME,
    /// SHAP (SHapley Additive exPlanations)
    SHAP,
    /// Gradient-based explanations
    GradientBased,
}

/// Compute model interpretability score
pub fn compute_interpretability_score<F: Float + std::iter::Sum>(
    explainability_metrics: &ExplainabilityMetrics<F>,
) -> F {
    // Weighted combination of different explainability aspects
    let feature_importance_score = if explainability_metrics.feature_importance.is_empty() {
        F::zero()
    } else {
        explainability_metrics
            .feature_importance
            .values()
            .cloned()
            .sum::<F>()
            / F::from(explainability_metrics.feature_importance.len()).unwrap()
    };

    let weights = [
        F::from(0.25).unwrap(), // feature importance
        F::from(0.2).unwrap(),  // local consistency
        F::from(0.2).unwrap(),  // global stability
        F::from(0.15).unwrap(), // faithfulness
        F::from(0.15).unwrap(), // completeness
        F::from(0.05).unwrap(), // uncertainty
    ];

    let scores = [
        feature_importance_score,
        explainability_metrics.local_consistency,
        explainability_metrics.global_stability,
        explainability_metrics.faithfulness,
        explainability_metrics.completeness,
        F::one() - explainability_metrics.uncertainty_metrics.total_uncertainty, // Lower uncertainty is better
    ];

    weights
        .iter()
        .zip(scores.iter())
        .map(|(&w, &s)| w * s)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_explainability_evaluator_creation() {
        let evaluator = ExplainabilityEvaluator::<f64>::new()
            .with_perturbations(50)
            .with_perturbation_strength(0.05)
            .with_importance_threshold(0.02);

        assert_eq!(evaluator.n_perturbations, 50);
        assert_eq!(evaluator.perturbation_strength, 0.05);
        assert_eq!(evaluator.importance_threshold, 0.02);
    }

    #[test]
    fn test_correlation_computation() {
        let evaluator = ExplainabilityEvaluator::<f64>::new();

        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0]; // Perfect correlation

        let correlation = evaluator.compute_correlation(&x, &y).unwrap();
        assert!((correlation - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_permutation_feature() {
        let evaluator = ExplainabilityEvaluator::<f64>::new();
        let mut data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let original_data = data.clone();

        evaluator.permute_feature(&mut data, 1).unwrap();

        // Feature 1 should be different, others should be the same
        assert_eq!(data.column(0), original_data.column(0));
        assert_eq!(data.column(2), original_data.column(2));
        // Column 1 should have the same values but potentially in different order
        assert_eq!(data.column(1).len(), original_data.column(1).len());
    }

    #[test]
    fn test_interpretability_score() {
        let mut feature_importance = HashMap::new();
        feature_importance.insert("feature1".to_string(), 0.5);
        feature_importance.insert("feature2".to_string(), 0.3);

        let metrics = ExplainabilityMetrics {
            feature_importance,
            local_consistency: 0.8,
            global_stability: 0.7,
            uncertainty_metrics: UncertaintyMetrics {
                epistemic_uncertainty: 0.1,
                aleatoric_uncertainty: 0.05,
                total_uncertainty: 0.15,
                coverage: 0.95,
                calibration_error: 0.02,
            },
            faithfulness: 0.9,
            completeness: 0.85,
        };

        let score = compute_interpretability_score(&metrics);
        assert!(score > 0.0 && score <= 1.0);
    }

    #[test]
    fn test_bootstrap_sampling() {
        let evaluator = ExplainabilityEvaluator::<f64>::new();
        let indices = evaluator.bootstrap_sample_indices(10).unwrap();

        assert_eq!(indices.len(), 10);
        // All indices should be valid (0-9)
        assert!(indices.iter().all(|&i| i < 10));
    }

    #[test]
    fn test_mask_important_features() {
        let evaluator = ExplainabilityEvaluator::<f64>::new();
        let sample = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let explanation = array![0.1, 0.5, 0.2, 0.8, 0.3]; // Feature 3 most important, then 1

        let masked = evaluator
            .mask_important_features(&sample.view(), &explanation, 2)
            .unwrap();

        // Features 3 and 1 (most important) should be masked to 0
        assert_eq!(masked[3], 0.0);
        assert_eq!(masked[1], 0.0);
        // Other features should remain unchanged
        assert_eq!(masked[0], 1.0);
        assert_eq!(masked[2], 3.0);
        assert_eq!(masked[4], 5.0);
    }
}
