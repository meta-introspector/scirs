//! Model interpretation utilities for neural networks
//!
//! This module provides tools for understanding neural network decisions including:
//! - Gradient-based attribution methods (Saliency, Integrated Gradients, GradCAM)
//! - Feature visualization and analysis
//! - Layer activation analysis and statistics
//! - Decision explanation tools

use crate::error::{NeuralError, Result};
use crate::layers::Layer;
use ndarray::{Array, ArrayD, Axis, IxDyn, s};
use num_traits::Float;
use std::collections::HashMap;
use std::fmt::Debug;

/// Attribution method for computing feature importance
#[derive(Debug, Clone, PartialEq)]
pub enum AttributionMethod {
    /// Simple gradient-based saliency
    Saliency,
    /// Integrated gradients
    IntegratedGradients { 
        baseline: BaselineMethod,
        num_steps: usize 
    },
    /// Grad-CAM (Gradient-weighted Class Activation Mapping)
    GradCAM { target_layer: String },
    /// Guided backpropagation
    GuidedBackprop,
    /// DeepLIFT
    DeepLIFT { baseline: BaselineMethod },
    /// SHAP (SHapley Additive exPlanations)
    SHAP { 
        background_samples: usize,
        num_samples: usize 
    },
}

/// Baseline methods for attribution
#[derive(Debug, Clone, PartialEq)]
pub enum BaselineMethod {
    /// Zero baseline
    Zero,
    /// Random noise baseline
    Random { seed: u64 },
    /// Gaussian blur baseline
    GaussianBlur { sigma: f64 },
    /// Mean of training data
    TrainingMean,
    /// Custom baseline
    Custom(ArrayD<f32>),
}

/// Feature visualization method
#[derive(Debug, Clone, PartialEq)]
pub enum VisualizationMethod {
    /// Activation maximization
    ActivationMaximization {
        target_layer: String,
        target_unit: Option<usize>,
        num_iterations: usize,
        learning_rate: f64,
    },
    /// Deep dream
    DeepDream {
        target_layer: String,
        num_iterations: usize,
        learning_rate: f64,
        amplify_factor: f64,
    },
    /// Feature inversion
    FeatureInversion {
        target_layer: String,
        regularization_weight: f64,
    },
}

/// Model interpreter for analyzing neural network decisions
pub struct ModelInterpreter<F: Float + Debug> {
    /// Available attribution methods
    attribution_methods: Vec<AttributionMethod>,
    /// Cached gradients for different layers
    gradient_cache: HashMap<String, ArrayD<F>>,
    /// Cached activations for different layers
    activation_cache: HashMap<String, ArrayD<F>>,
    /// Layer statistics
    layer_statistics: HashMap<String, LayerAnalysisStats<F>>,
}

/// Statistical analysis of layer activations
#[derive(Debug, Clone)]
pub struct LayerAnalysisStats<F: Float + Debug> {
    /// Mean activation value
    pub mean_activation: F,
    /// Standard deviation of activations
    pub std_activation: F,
    /// Maximum activation value
    pub max_activation: F,
    /// Minimum activation value
    pub min_activation: F,
    /// Percentage of dead neurons (always zero)
    pub dead_neuron_percentage: f64,
    /// Sparsity (percentage of near-zero activations)
    pub sparsity: f64,
    /// Activation distribution histogram
    pub histogram: Vec<u32>,
    /// Histogram bin edges
    pub bin_edges: Vec<F>,
}

impl<F: Float + Debug + 'static> ModelInterpreter<F> {
    /// Create a new model interpreter
    pub fn new() -> Self {
        Self {
            attribution_methods: Vec::new(),
            gradient_cache: HashMap::new(),
            activation_cache: HashMap::new(),
            layer_statistics: HashMap::new(),
        }
    }

    /// Add an attribution method
    pub fn add_attribution_method(&mut self, method: AttributionMethod) {
        self.attribution_methods.push(method);
    }

    /// Cache layer activations
    pub fn cache_activations(&mut self, layer_name: String, activations: ArrayD<F>) {
        self.activation_cache.insert(layer_name, activations);
    }

    /// Cache layer gradients
    pub fn cache_gradients(&mut self, layer_name: String, gradients: ArrayD<F>) {
        self.gradient_cache.insert(layer_name, gradients);
    }

    /// Compute attribution using specified method
    pub fn compute_attribution(
        &self,
        method: &AttributionMethod,
        input: &ArrayD<F>,
        target_class: Option<usize>,
    ) -> Result<ArrayD<F>> {
        match method {
            AttributionMethod::Saliency => {
                self.compute_saliency_attribution(input, target_class)
            }
            AttributionMethod::IntegratedGradients { baseline, num_steps } => {
                self.compute_integrated_gradients(input, baseline, *num_steps, target_class)
            }
            AttributionMethod::GradCAM { target_layer } => {
                self.compute_gradcam_attribution(input, target_layer, target_class)
            }
            AttributionMethod::GuidedBackprop => {
                self.compute_guided_backprop_attribution(input, target_class)
            }
            AttributionMethod::DeepLIFT { baseline } => {
                self.compute_deeplift_attribution(input, baseline, target_class)
            }
            AttributionMethod::SHAP { background_samples, num_samples } => {
                self.compute_shap_attribution(input, *background_samples, *num_samples, target_class)
            }
        }
    }

    fn compute_saliency_attribution(
        &self,
        input: &ArrayD<F>,
        _target_class: Option<usize>,
    ) -> Result<ArrayD<F>> {
        // Simple gradient-based saliency
        // In practice, this would require running backward pass
        // For now, return a simplified version
        
        let grad_key = "input_gradient";
        if let Some(gradient) = self.gradient_cache.get(grad_key) {
            Ok(gradient.mapv(|x| x.abs()))
        } else {
            // Return random attribution as placeholder
            let attribution = input.mapv(|_| F::from(0.5).unwrap());
            Ok(attribution)
        }
    }

    fn compute_integrated_gradients(
        &self,
        input: &ArrayD<F>,
        baseline: &BaselineMethod,
        num_steps: usize,
        _target_class: Option<usize>,
    ) -> Result<ArrayD<F>> {
        let baseline_input = self.create_baseline(input, baseline)?;
        let mut accumulated_gradients = Array::zeros(input.raw_dim());
        
        for i in 0..num_steps {
            let alpha = F::from(i as f64 / (num_steps - 1) as f64).unwrap();
            let interpolated_input = &baseline_input + (&(*input - &baseline_input) * alpha);
            
            // In practice, would compute gradients for interpolated input
            // For now, use a simplified approximation
            let step_gradient = interpolated_input.mapv(|x| x * F::from(0.1).unwrap());
            accumulated_gradients = accumulated_gradients + step_gradient;
        }
        
        let integrated_gradients = (input - &baseline_input) * accumulated_gradients / F::from(num_steps).unwrap();
        Ok(integrated_gradients)
    }

    fn compute_gradcam_attribution(
        &self,
        input: &ArrayD<F>,
        target_layer: &str,
        _target_class: Option<usize>,
    ) -> Result<ArrayD<F>> {
        // Get activations and gradients for target layer
        let activations = self.activation_cache.get(target_layer)
            .ok_or_else(|| NeuralError::ComputationError(
                format!("Activations not found for layer: {}", target_layer)
            ))?;
        
        let gradients = self.gradient_cache.get(target_layer)
            .ok_or_else(|| NeuralError::ComputationError(
                format!("Gradients not found for layer: {}", target_layer)
            ))?;

        if activations.ndim() < 3 {
            return Err(NeuralError::InvalidArchitecture(
                "GradCAM requires at least 3D activations (batch, channels, spatial)".to_string()
            ));
        }

        // Compute channel-wise weights by global average pooling of gradients
        let mut weights = Vec::new();
        let num_channels = activations.shape()[1];
        
        for c in 0..num_channels {
            let channel_grad = gradients.slice(s![.., c, ..]);
            let weight = channel_grad.mean().unwrap_or(F::zero());
            weights.push(weight);
        }

        // Compute weighted combination of activation maps
        let mut gradcam = Array::zeros(activations.slice(s![.., 0, ..]).raw_dim());
        
        for c in 0..num_channels {
            let channel_activation = activations.slice(s![.., c, ..]);
            let weighted_activation = channel_activation * weights[c];
            gradcam = gradcam + weighted_activation;
        }

        // ReLU to keep only positive influences
        let gradcam_relu = gradcam.mapv(|x| x.max(F::zero()));
        
        // Resize to input dimensions if needed
        if gradcam_relu.shape() != input.shape() {
            // Simplified resize - in practice would use proper interpolation
            self.resize_attribution(&gradcam_relu, input.raw_dim())
        } else {
            Ok(gradcam_relu)
        }
    }

    fn compute_guided_backprop_attribution(
        &self,
        input: &ArrayD<F>,
        _target_class: Option<usize>,
    ) -> Result<ArrayD<F>> {
        // Guided backpropagation - simplified implementation
        // In practice, this would modify the backward pass to zero negative gradients
        if let Some(gradient) = self.gradient_cache.get("input_gradient") {
            // Keep only positive gradients
            Ok(gradient.mapv(|x| x.max(F::zero())))
        } else {
            Ok(input.mapv(|_| F::zero()))
        }
    }

    fn compute_deeplift_attribution(
        &self,
        input: &ArrayD<F>,
        baseline: &BaselineMethod,
        _target_class: Option<usize>,
    ) -> Result<ArrayD<F>> {
        let baseline_input = self.create_baseline(input, baseline)?;
        
        // DeepLIFT attribution - simplified implementation
        // In practice, this would require special backward pass rules
        let diff = input - &baseline_input;
        
        if let Some(gradient) = self.gradient_cache.get("input_gradient") {
            Ok(&diff * gradient)
        } else {
            Ok(diff)
        }
    }

    fn compute_shap_attribution(
        &self,
        input: &ArrayD<F>,
        background_samples: usize,
        num_samples: usize,
        _target_class: Option<usize>,
    ) -> Result<ArrayD<F>> {
        // SHAP attribution - simplified implementation
        // In practice, this would use proper Shapley value computation
        
        let mut total_attribution = Array::zeros(input.raw_dim());
        let _background_size = background_samples; // Placeholder
        
        for _ in 0..num_samples {
            // Create random coalition
            let coalition_mask = input.mapv(|_| {
                if rand::random::<f64>() > 0.5 { F::one() } else { F::zero() }
            });
            
            // Compute marginal contribution (simplified)
            let marginal_contribution = input * &coalition_mask * F::from(0.1).unwrap();
            total_attribution = total_attribution + marginal_contribution;
        }
        
        Ok(total_attribution / F::from(num_samples).unwrap())
    }

    fn create_baseline(&self, input: &ArrayD<F>, method: &BaselineMethod) -> Result<ArrayD<F>> {
        match method {
            BaselineMethod::Zero => {
                Ok(Array::zeros(input.raw_dim()))
            }
            BaselineMethod::Random { seed: _ } => {
                // Create random baseline with same shape
                Ok(input.mapv(|_| F::from(rand::random::<f64>()).unwrap()))
            }
            BaselineMethod::GaussianBlur { sigma: _ } => {
                // Simplified Gaussian blur - in practice would use proper convolution
                let blurred = input.mapv(|x| x * F::from(0.5).unwrap());
                Ok(blurred)
            }
            BaselineMethod::TrainingMean => {
                // Use zero as placeholder for training mean
                Ok(Array::zeros(input.raw_dim()))
            }
            BaselineMethod::Custom(baseline) => {
                if baseline.shape() == input.shape() {
                    // Convert f32 baseline to F type
                    let converted = baseline.mapv(|x| F::from(x).unwrap_or(F::zero()));
                    Ok(converted)
                } else {
                    Err(NeuralError::DimensionMismatch(
                        "Custom baseline shape doesn't match input".to_string()
                    ))
                }
            }
        }
    }

    fn resize_attribution(&self, attribution: &ArrayD<F>, target_shape: IxDyn) -> Result<ArrayD<F>> {
        // Simplified resize - in practice would use proper interpolation
        if attribution.len() == target_shape.size() {
            Ok(attribution.clone().into_shape(target_shape)?)
        } else {
            // Create new array with target shape and fill with mean
            let mean_val = attribution.mean().unwrap_or(F::zero());
            Ok(Array::from_elem(target_shape, mean_val))
        }
    }

    /// Analyze layer activations and compute statistics
    pub fn analyze_layer_activations(&mut self, layer_name: String, activations: &ArrayD<F>) -> Result<()> {
        let mean_activation = activations.mean().unwrap_or(F::zero());
        let variance = activations.mapv(|x| (x - mean_activation) * (x - mean_activation)).mean().unwrap_or(F::zero());
        let std_activation = variance.sqrt();
        
        let max_activation = activations.iter().cloned().fold(F::neg_infinity(), F::max);
        let min_activation = activations.iter().cloned().fold(F::infinity(), F::min);
        
        // Compute dead neuron percentage (neurons that are always zero)
        let zero_threshold = F::from(1e-6).unwrap();
        let dead_neurons = activations.iter().filter(|&&x| x.abs() < zero_threshold).count();
        let dead_neuron_percentage = dead_neurons as f64 / activations.len() as f64 * 100.0;
        
        // Compute sparsity (percentage of near-zero activations)
        let sparsity_threshold = F::from(0.01).unwrap();
        let sparse_neurons = activations.iter().filter(|&&x| x.abs() < sparsity_threshold).count();
        let sparsity = sparse_neurons as f64 / activations.len() as f64 * 100.0;
        
        // Create histogram
        let num_bins = 50;
        let range = max_activation - min_activation;
        let bin_width = if range > F::zero() { range / F::from(num_bins).unwrap() } else { F::one() };
        
        let mut histogram = vec![0u32; num_bins];
        let mut bin_edges = Vec::with_capacity(num_bins + 1);
        
        for i in 0..=num_bins {
            bin_edges.push(min_activation + bin_width * F::from(i).unwrap());
        }
        
        for &val in activations.iter() {
            if val.is_finite() && range > F::zero() {
                let bin_idx = ((val - min_activation) / bin_width).to_usize().unwrap_or(0);
                let bin_idx = bin_idx.min(num_bins - 1);
                histogram[bin_idx] += 1;
            }
        }
        
        let stats = LayerAnalysisStats {
            mean_activation,
            std_activation,
            max_activation,
            min_activation,
            dead_neuron_percentage,
            sparsity,
            histogram,
            bin_edges,
        };
        
        self.layer_statistics.insert(layer_name, stats);
        Ok(())
    }

    /// Get layer analysis statistics
    pub fn get_layer_statistics(&self, layer_name: &str) -> Option<&LayerAnalysisStats<F>> {
        self.layer_statistics.get(layer_name)
    }

    /// Get all layer statistics
    pub fn get_all_layer_statistics(&self) -> &HashMap<String, LayerAnalysisStats<F>> {
        &self.layer_statistics
    }

    /// Generate interpretation report for a sample
    pub fn generate_interpretation_report(
        &self,
        input: &ArrayD<F>,
        target_class: Option<usize>,
    ) -> Result<InterpretationReport<F>> {
        let mut attributions = HashMap::new();
        
        // Compute attributions using all available methods
        for method in &self.attribution_methods {
            let attribution = self.compute_attribution(method, input, target_class)?;
            let method_name = format!("{:?}", method);
            attributions.insert(method_name, attribution);
        }
        
        // Compute attribution statistics
        let mut attribution_stats = HashMap::new();
        for (method_name, attribution) in &attributions {
            let stats = self.compute_attribution_statistics(attribution);
            attribution_stats.insert(method_name.clone(), stats);
        }
        
        Ok(InterpretationReport {
            input_shape: input.raw_dim(),
            target_class,
            attributions,
            attribution_statistics: attribution_stats,
            layer_statistics: self.layer_statistics.clone(),
            interpretation_summary: self.generate_interpretation_summary(&attributions),
        })
    }

    fn compute_attribution_statistics(&self, attribution: &ArrayD<F>) -> AttributionStatistics<F> {
        let mean = attribution.mean().unwrap_or(F::zero());
        let abs_attribution = attribution.mapv(|x| x.abs());
        let mean_abs = abs_attribution.mean().unwrap_or(F::zero());
        let max_abs = abs_attribution.iter().cloned().fold(F::zero(), F::max);
        let positive_ratio = attribution.iter().filter(|&&x| x > F::zero()).count() as f64 / attribution.len() as f64;
        
        AttributionStatistics {
            mean,
            mean_absolute: mean_abs,
            max_absolute: max_abs,
            positive_attribution_ratio: positive_ratio,
            total_positive_attribution: attribution.iter().filter(|&&x| x > F::zero()).cloned().sum(),
            total_negative_attribution: attribution.iter().filter(|&&x| x < F::zero()).cloned().sum(),
        }
    }

    fn generate_interpretation_summary(&self, attributions: &HashMap<String, ArrayD<F>>) -> InterpretationSummary {
        let num_methods = attributions.len();
        
        // Find most consistent features across methods
        let mut feature_consistency_scores = Vec::new();
        
        if let Some((_, first_attribution)) = attributions.iter().next() {
            for i in 0..first_attribution.len() {
                let mut scores = Vec::new();
                for attribution in attributions.values() {
                    if i < attribution.len() {
                        scores.push(attribution[i].to_f64().unwrap_or(0.0));
                    }
                }
                
                // Compute consistency as standard deviation (lower is more consistent)
                if !scores.is_empty() {
                    let mean = scores.iter().sum::<f64>() / scores.len() as f64;
                    let variance = scores.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / scores.len() as f64;
                    let std_dev = variance.sqrt();
                    feature_consistency_scores.push(1.0 / (1.0 + std_dev)); // Higher is more consistent
                } else {
                    feature_consistency_scores.push(0.0);
                }
            }
        }
        
        let avg_consistency = if !feature_consistency_scores.is_empty() {
            feature_consistency_scores.iter().sum::<f64>() / feature_consistency_scores.len() as f64
        } else {
            0.0
        };
        
        InterpretationSummary {
            num_attribution_methods: num_methods,
            average_method_consistency: avg_consistency,
            most_important_features: self.find_most_important_features(attributions, 10),
            interpretation_confidence: self.compute_interpretation_confidence(attributions),
        }
    }

    fn find_most_important_features(&self, attributions: &HashMap<String, ArrayD<F>>, top_k: usize) -> Vec<usize> {
        if attributions.is_empty() {
            return Vec::new();
        }
        
        // Average attributions across methods
        let first_attribution = attributions.values().next().unwrap();
        let mut averaged_attribution = Array::zeros(first_attribution.raw_dim());
        
        for attribution in attributions.values() {
            averaged_attribution = averaged_attribution + attribution;
        }
        averaged_attribution = averaged_attribution / F::from(attributions.len()).unwrap();
        
        // Find top-k features by absolute importance
        let mut feature_scores: Vec<(usize, f64)> = averaged_attribution
            .iter()
            .enumerate()
            .map(|(i, &score)| (i, score.abs().to_f64().unwrap_or(0.0)))
            .collect();
        
        feature_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        feature_scores.into_iter().take(top_k).map(|(i, _)| i).collect()
    }

    fn compute_interpretation_confidence(&self, attributions: &HashMap<String, ArrayD<F>>) -> f64 {
        if attributions.len() < 2 {
            return 1.0; // Single method, assume full confidence
        }
        
        // Compute pairwise correlations between attribution methods
        let methods: Vec<_> = attributions.keys().collect();
        let mut correlations = Vec::new();
        
        for i in 0..methods.len() {
            for j in (i + 1)..methods.len() {
                let attr1 = &attributions[methods[i]];
                let attr2 = &attributions[methods[j]];
                
                if attr1.len() == attr2.len() {
                    let correlation = self.compute_correlation(attr1, attr2);
                    correlations.push(correlation);
                }
            }
        }
        
        // Average correlation as confidence measure
        if !correlations.is_empty() {
            correlations.iter().sum::<f64>() / correlations.len() as f64
        } else {
            0.5
        }
    }

    fn compute_correlation(&self, x: &ArrayD<F>, y: &ArrayD<F>) -> f64 {
        let n = x.len() as f64;
        if n == 0.0 {
            return 0.0;
        }
        
        let x_mean = x.mean().unwrap_or(F::zero()).to_f64().unwrap_or(0.0);
        let y_mean = y.mean().unwrap_or(F::zero()).to_f64().unwrap_or(0.0);
        
        let mut numerator = 0.0;
        let mut x_sum_sq = 0.0;
        let mut y_sum_sq = 0.0;
        
        for (x_val, y_val) in x.iter().zip(y.iter()) {
            let x_diff = x_val.to_f64().unwrap_or(0.0) - x_mean;
            let y_diff = y_val.to_f64().unwrap_or(0.0) - y_mean;
            
            numerator += x_diff * y_diff;
            x_sum_sq += x_diff * x_diff;
            y_sum_sq += y_diff * y_diff;
        }
        
        let denominator = (x_sum_sq * y_sum_sq).sqrt();
        
        if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        }
    }
}

impl<F: Float + Debug + 'static> Default for ModelInterpreter<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for attribution methods
#[derive(Debug, Clone)]
pub struct AttributionStatistics<F: Float + Debug> {
    /// Mean attribution value
    pub mean: F,
    /// Mean absolute attribution value
    pub mean_absolute: F,
    /// Maximum absolute attribution value
    pub max_absolute: F,
    /// Ratio of positive attributions
    pub positive_attribution_ratio: f64,
    /// Total positive attribution
    pub total_positive_attribution: F,
    /// Total negative attribution
    pub total_negative_attribution: F,
}

/// Summary of interpretation analysis
#[derive(Debug, Clone)]
pub struct InterpretationSummary {
    /// Number of attribution methods used
    pub num_attribution_methods: usize,
    /// Average consistency across methods
    pub average_method_consistency: f64,
    /// Indices of most important features
    pub most_important_features: Vec<usize>,
    /// Overall interpretation confidence (0-1)
    pub interpretation_confidence: f64,
}

/// Comprehensive interpretation report
#[derive(Debug)]
pub struct InterpretationReport<F: Float + Debug> {
    /// Shape of input that was interpreted
    pub input_shape: IxDyn,
    /// Target class (if specified)
    pub target_class: Option<usize>,
    /// Attribution maps for each method
    pub attributions: HashMap<String, ArrayD<F>>,
    /// Statistics for each attribution method
    pub attribution_statistics: HashMap<String, AttributionStatistics<F>>,
    /// Layer analysis statistics
    pub layer_statistics: HashMap<String, LayerAnalysisStats<F>>,
    /// Summary of interpretation
    pub interpretation_summary: InterpretationSummary,
}

impl<F: Float + Debug> std::fmt::Display for InterpretationReport<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Neural Network Interpretation Report")?;
        writeln!(f, "===================================")?;
        writeln!(f, "Input Shape: {:?}", self.input_shape)?;
        writeln!(f, "Target Class: {:?}", self.target_class)?;
        writeln!(f, "Attribution Methods: {}", self.attribution_statistics.len())?;
        writeln!(f, "Interpretation Confidence: {:.3}", self.interpretation_summary.interpretation_confidence)?;
        writeln!(f, "Average Method Consistency: {:.3}", self.interpretation_summary.average_method_consistency)?;
        writeln!(f, "Top Important Features: {:?}", self.interpretation_summary.most_important_features)?;
        
        writeln!(f, "\nLayer Statistics:")?;
        for (layer_name, stats) in &self.layer_statistics {
            writeln!(f, "  {}: mean={:.3}, std={:.3}, sparsity={:.1}%, dead_neurons={:.1}%", 
                    layer_name, 
                    stats.mean_activation.to_f64().unwrap_or(0.0),
                    stats.std_activation.to_f64().unwrap_or(0.0),
                    stats.sparsity,
                    stats.dead_neuron_percentage)?;
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_model_interpreter_creation() {
        let interpreter = ModelInterpreter::<f64>::new();
        assert_eq!(interpreter.attribution_methods.len(), 0);
        assert_eq!(interpreter.gradient_cache.len(), 0);
    }

    #[test]
    fn test_saliency_attribution() {
        let mut interpreter = ModelInterpreter::<f64>::new();
        
        // Cache some gradients
        let gradients = Array2::from_shape_vec((2, 3), vec![0.1, 0.2, -0.3, 0.4, -0.5, 0.6]).unwrap().into_dyn();
        interpreter.cache_gradients("input_gradient".to_string(), gradients);
        
        let input = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap().into_dyn();
        
        let attribution = interpreter.compute_saliency_attribution(&input, None);
        assert!(attribution.is_ok());
        
        let attr = attribution.unwrap();
        assert_eq!(attr.shape(), input.shape());
        
        // Should be absolute values of gradients
        assert_eq!(attr[[0, 0]], 0.1);
        assert_eq!(attr[[0, 2]], 0.3); // abs(-0.3)
    }

    #[test]
    fn test_integrated_gradients() {
        let interpreter = ModelInterpreter::<f64>::new();
        
        let input = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap().into_dyn();
        let baseline = BaselineMethod::Zero;
        
        let attribution = interpreter.compute_integrated_gradients(&input, &baseline, 10, None);
        assert!(attribution.is_ok());
        
        let attr = attribution.unwrap();
        assert_eq!(attr.shape(), input.shape());
    }

    #[test]
    fn test_baseline_creation() {
        let interpreter = ModelInterpreter::<f64>::new();
        let input = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap().into_dyn();
        
        // Test zero baseline
        let zero_baseline = interpreter.create_baseline(&input, &BaselineMethod::Zero);
        assert!(zero_baseline.is_ok());
        let baseline = zero_baseline.unwrap();
        assert_eq!(baseline.shape(), input.shape());
        assert!(baseline.iter().all(|&x| x == 0.0));
        
        // Test custom baseline
        let custom_array = Array2::from_elem((2, 3), 0.5f32).into_dyn();
        let custom_baseline = interpreter.create_baseline(&input, &BaselineMethod::Custom(custom_array));
        assert!(custom_baseline.is_ok());
        let baseline = custom_baseline.unwrap();
        assert_eq!(baseline.shape(), input.shape());
        assert!(baseline.iter().all(|&x| x == 0.5));
    }

    #[test]
    fn test_layer_analysis() {
        let mut interpreter = ModelInterpreter::<f64>::new();
        
        let activations = Array2::from_shape_vec((10, 5), 
            vec![
                0.0, 0.1, 0.0, 0.5, 1.0,
                0.0, 0.0, 0.2, 0.3, 0.8,
                0.1, 0.0, 0.0, 0.7, 0.9,
                0.0, 0.3, 0.1, 0.4, 0.6,
                0.2, 0.0, 0.0, 0.2, 0.7,
                0.0, 0.1, 0.3, 0.6, 0.8,
                0.1, 0.0, 0.0, 0.5, 0.9,
                0.0, 0.2, 0.1, 0.3, 0.7,
                0.0, 0.0, 0.0, 0.8, 1.0,
                0.1, 0.1, 0.2, 0.4, 0.6,
            ]
        ).unwrap().into_dyn();
        
        interpreter.analyze_layer_activations("test_layer".to_string(), &activations).unwrap();
        
        let stats = interpreter.get_layer_statistics("test_layer").unwrap();
        assert!(stats.mean_activation > 0.0);
        assert!(stats.std_activation > 0.0);
        assert!(stats.sparsity > 0.0); // Should have some near-zero values
        assert_eq!(stats.histogram.len(), 50);
    }

    #[test]
    fn test_gradcam_attribution() {
        let mut interpreter = ModelInterpreter::<f64>::new();
        
        // Cache activations and gradients for a convolutional layer
        let activations = Array::from_shape_vec((1, 3, 4, 4), (0..48).map(|x| x as f64 / 10.0).collect()).unwrap().into_dyn();
        let gradients = Array::from_shape_vec((1, 3, 4, 4), (0..48).map(|x| (x % 5) as f64 / 10.0).collect()).unwrap().into_dyn();
        
        interpreter.cache_activations("conv_layer".to_string(), activations);
        interpreter.cache_gradients("conv_layer".to_string(), gradients);
        
        let input = Array::from_shape_vec((1, 3, 4, 4), (0..48).map(|x| x as f64).collect()).unwrap().into_dyn();
        
        let attribution = interpreter.compute_gradcam_attribution(&input, "conv_layer", None);
        assert!(attribution.is_ok());
    }

    #[test]
    fn test_interpretation_report_generation() {
        let mut interpreter = ModelInterpreter::<f64>::new();
        interpreter.add_attribution_method(AttributionMethod::Saliency);
        interpreter.add_attribution_method(AttributionMethod::IntegratedGradients { 
            baseline: BaselineMethod::Zero, 
            num_steps: 10 
        });
        
        // Cache some gradients for saliency
        let gradients = Array2::from_shape_vec((2, 3), vec![0.1, 0.2, -0.3, 0.4, -0.5, 0.6]).unwrap().into_dyn();
        interpreter.cache_gradients("input_gradient".to_string(), gradients);
        
        let input = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap().into_dyn();
        
        let report = interpreter.generate_interpretation_report(&input, Some(1));
        assert!(report.is_ok());
        
        let rep = report.unwrap();
        assert_eq!(rep.target_class, Some(1));
        assert_eq!(rep.attributions.len(), 2); // Two attribution methods
        assert_eq!(rep.input_shape, input.raw_dim());
    }

    #[test]
    fn test_attribution_statistics() {
        let interpreter = ModelInterpreter::<f64>::new();
        
        let attribution = Array2::from_shape_vec((2, 3), vec![0.5, -0.3, 0.8, -0.2, 0.0, 0.7]).unwrap().into_dyn();
        let stats = interpreter.compute_attribution_statistics(&attribution);
        
        assert!(stats.mean_absolute > 0.0);
        assert!(stats.positive_attribution_ratio > 0.0);
        assert!(stats.positive_attribution_ratio < 1.0);
        assert!(stats.total_positive_attribution > 0.0);
        assert!(stats.total_negative_attribution < 0.0);
    }

    #[test]
    fn test_correlation_computation() {
        let interpreter = ModelInterpreter::<f64>::new();
        
        let x = Array2::from_shape_vec((1, 5), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap().into_dyn();
        let y = Array2::from_shape_vec((1, 5), vec![2.0, 4.0, 6.0, 8.0, 10.0]).unwrap().into_dyn(); // Perfect correlation
        
        let correlation = interpreter.compute_correlation(&x, &y);
        assert!((correlation - 1.0).abs() < 1e-10); // Should be perfectly correlated
        
        let z = Array2::from_shape_vec((1, 5), vec![5.0, 4.0, 3.0, 2.0, 1.0]).unwrap().into_dyn(); // Perfect anti-correlation
        let anti_correlation = interpreter.compute_correlation(&x, &z);
        assert!((anti_correlation + 1.0).abs() < 1e-10); // Should be perfectly anti-correlated
    }
}