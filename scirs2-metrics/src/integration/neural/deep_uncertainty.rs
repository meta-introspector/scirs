//! Deep learning uncertainty quantification methods
//!
//! This module provides advanced uncertainty quantification methods specifically
//! designed for deep neural networks, including:
//! - Monte Carlo Dropout for epistemic uncertainty
//! - Deep Ensembles for robust uncertainty estimation
//! - Bayesian Neural Networks with variational inference
//! - Test-time augmentation for prediction diversity
//! - Predictive entropy decomposition
//! - Temperature scaling for neural network calibration

#![allow(clippy::too_many_arguments)]

use crate::error::{MetricsError, Result};
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, Axis};
use num_traits::Float;
use scirs2_core::simd_ops::SimdUnifiedOps;
use std::collections::HashMap;
use std::iter::Sum;

/// Deep learning uncertainty quantifier
pub struct DeepUncertaintyQuantifier<F: Float> {
    /// Number of Monte Carlo samples for dropout
    pub n_mc_dropout_samples: usize,
    /// Dropout rate for MC Dropout
    pub dropout_rate: F,
    /// Number of ensemble members
    pub n_ensemble_members: usize,
    /// Number of test-time augmentation samples
    pub n_tta_samples: usize,
    /// Enable temperature scaling
    pub enable_temperature_scaling: bool,
    /// Enable SWAG (Stochastic Weight Averaging Gaussian)
    pub enable_swag: bool,
    /// Number of SWAG samples
    pub n_swag_samples: usize,
    /// Random seed
    pub random_seed: Option<u64>,
}

impl<F: Float + num_traits::FromPrimitive + Sum> Default for DeepUncertaintyQuantifier<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + num_traits::FromPrimitive + Sum> DeepUncertaintyQuantifier<F> {
    /// Create new deep uncertainty quantifier
    pub fn new() -> Self {
        Self {
            n_mc_dropout_samples: 100,
            dropout_rate: F::from(0.1).unwrap(),
            n_ensemble_members: 5,
            n_tta_samples: 10,
            enable_temperature_scaling: true,
            enable_swag: false,
            n_swag_samples: 20,
            random_seed: None,
        }
    }
    
    /// Set Monte Carlo dropout parameters
    pub fn with_mc_dropout(mut self, n_samples: usize, dropout_rate: F) -> Self {
        self.n_mc_dropout_samples = n_samples;
        self.dropout_rate = dropout_rate;
        self
    }
    
    /// Set ensemble parameters
    pub fn with_ensemble(mut self, n_members: usize) -> Self {
        self.n_ensemble_members = n_members;
        self
    }
    
    /// Set test-time augmentation parameters
    pub fn with_tta(mut self, n_samples: usize) -> Self {
        self.n_tta_samples = n_samples;
        self
    }
    
    /// Enable/disable temperature scaling
    pub fn with_temperature_scaling(mut self, enable: bool) -> Self {
        self.enable_temperature_scaling = enable;
        self
    }
    
    /// Set SWAG parameters
    pub fn with_swag(mut self, enable: bool, n_samples: usize) -> Self {
        self.enable_swag = enable;
        self.n_swag_samples = n_samples;
        self
    }
    
    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }
    
    /// Compute comprehensive deep learning uncertainty
    pub fn compute_deep_uncertainty<M, A, E>(
        &self,
        mc_dropout_model: &M,
        ensemble_models: &[E],
        augmentation_fn: &A,
        x_test: &Array2<F>,
        x_calibration: Option<&Array2<F>>,
        y_calibration: Option<&Array1<F>>,
    ) -> Result<DeepUncertaintyAnalysis<F>>
    where
        M: Fn(&ArrayView2<F>, bool) -> Array1<F>, // model with dropout flag
        E: Fn(&ArrayView2<F>) -> Array1<F>, // ensemble member
        A: Fn(&ArrayView2<F>) -> Array3<F>, // augmentation function returning [n_aug, n_samples, n_features]
    {
        // Monte Carlo Dropout uncertainty
        let mc_dropout_uncertainty = self.compute_mc_dropout_uncertainty(mc_dropout_model, x_test)?;
        
        // Deep ensemble uncertainty
        let ensemble_uncertainty = self.compute_ensemble_uncertainty(ensemble_models, x_test)?;
        
        // Test-time augmentation uncertainty
        let tta_uncertainty = self.compute_tta_uncertainty(mc_dropout_model, augmentation_fn, x_test)?;
        
        // Predictive entropy decomposition
        let entropy_decomposition = self.compute_entropy_decomposition(&mc_dropout_uncertainty.predictions)?;
        
        // Temperature scaling (if enabled and calibration data available)
        let temperature_scaling = if self.enable_temperature_scaling 
            && x_calibration.is_some() 
            && y_calibration.is_some() {
            Some(self.compute_neural_temperature_scaling(
                mc_dropout_model, 
                x_calibration.unwrap(), 
                y_calibration.unwrap()
            )?)
        } else {
            None
        };
        
        // SWAG uncertainty (if enabled)
        let swag_uncertainty = if self.enable_swag {
            Some(self.compute_swag_uncertainty(mc_dropout_model, x_test)?)
        } else {
            None
        };
        
        // Disagreement-based uncertainty
        let disagreement_uncertainty = self.compute_disagreement_uncertainty(
            &mc_dropout_uncertainty.predictions,
            &ensemble_uncertainty.predictions,
        )?;
        
        Ok(DeepUncertaintyAnalysis {
            mc_dropout_uncertainty,
            ensemble_uncertainty,
            tta_uncertainty,
            entropy_decomposition,
            temperature_scaling,
            swag_uncertainty,
            disagreement_uncertainty,
            sample_size: x_test.nrows(),
        })
    }
    
    /// Compute Monte Carlo Dropout uncertainty
    fn compute_mc_dropout_uncertainty<M>(
        &self,
        model: &M,
        x_test: &Array2<F>,
    ) -> Result<MCDropoutUncertainty<F>>
    where
        M: Fn(&ArrayView2<F>, bool) -> Array1<F>,
    {
        let n_samples = x_test.nrows();
        let mut predictions = Array2::zeros((self.n_mc_dropout_samples, n_samples));
        
        // Generate MC Dropout samples
        for i in 0..self.n_mc_dropout_samples {
            let sample_predictions = model(&x_test.view(), true); // Enable dropout
            for j in 0..n_samples {
                predictions[[i, j]] = sample_predictions[j];
            }
        }
        
        // Compute statistics
        let mean_predictions = predictions.mean_axis(Axis(0)).unwrap();
        let var_predictions = predictions.var_axis(Axis(0), F::zero());
        let std_predictions = var_predictions.mapv(|x| x.sqrt());
        
        // Compute epistemic and aleatoric uncertainty
        let epistemic_uncertainty = self.compute_epistemic_from_samples(&predictions)?;
        let aleatoric_uncertainty = self.compute_aleatoric_from_samples(&predictions)?;
        
        // Compute prediction intervals
        let prediction_intervals = self.compute_mc_prediction_intervals(&predictions)?;
        
        Ok(MCDropoutUncertainty {
            predictions,
            mean_predictions,
            std_predictions,
            epistemic_uncertainty,
            aleatoric_uncertainty,
            prediction_intervals,
            n_samples: self.n_mc_dropout_samples,
        })
    }
    
    /// Compute deep ensemble uncertainty
    fn compute_ensemble_uncertainty<E>(
        &self,
        ensemble_models: &[E],
        x_test: &Array2<F>,
    ) -> Result<EnsembleUncertainty<F>>
    where
        E: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        let n_samples = x_test.nrows();
        let n_models = ensemble_models.len();
        let mut predictions = Array2::zeros((n_models, n_samples));
        
        // Generate ensemble predictions
        for (i, model) in ensemble_models.iter().enumerate() {
            let model_predictions = model(&x_test.view());
            for j in 0..n_samples {
                predictions[[i, j]] = model_predictions[j];
            }
        }
        
        // Compute ensemble statistics
        let mean_predictions = predictions.mean_axis(Axis(0)).unwrap();
        let var_predictions = predictions.var_axis(Axis(0), F::zero());
        let std_predictions = var_predictions.mapv(|x| x.sqrt());
        
        // Compute model diversity
        let model_diversity = self.compute_model_diversity(&predictions)?;
        
        // Compute prediction intervals
        let prediction_intervals = self.compute_ensemble_prediction_intervals(&predictions)?;
        
        // Compute mutual information between models
        let mutual_information = self.compute_model_mutual_information(&predictions)?;
        
        Ok(EnsembleUncertainty {
            predictions,
            mean_predictions,
            std_predictions,
            model_diversity,
            prediction_intervals,
            mutual_information,
            n_models,
        })
    }
    
    /// Compute test-time augmentation uncertainty
    fn compute_tta_uncertainty<M, A>(
        &self,
        model: &M,
        augmentation_fn: &A,
        x_test: &Array2<F>,
    ) -> Result<TTAUncertainty<F>>
    where
        M: Fn(&ArrayView2<F>, bool) -> Array1<F>,
        A: Fn(&ArrayView2<F>) -> Array3<F>,
    {
        let n_samples = x_test.nrows();
        let mut predictions = Array2::zeros((self.n_tta_samples, n_samples));
        
        // Generate augmented samples and predictions
        let augmented_data = augmentation_fn(&x_test.view());
        
        for i in 0..self.n_tta_samples.min(augmented_data.nrows()) {
            let aug_sample = augmented_data.slice(s![i, .., ..]);
            let aug_predictions = model(&aug_sample, false); // No dropout for TTA
            for j in 0..n_samples {
                predictions[[i, j]] = aug_predictions[j];
            }
        }
        
        // Compute TTA statistics
        let mean_predictions = predictions.mean_axis(Axis(0)).unwrap();
        let std_predictions = predictions.var_axis(Axis(0), F::zero()).mapv(|x| x.sqrt());
        
        // Compute augmentation consistency
        let consistency_score = self.compute_augmentation_consistency(&predictions)?;
        
        Ok(TTAUncertainty {
            predictions,
            mean_predictions,
            std_predictions,
            consistency_score,
            n_augmentations: self.n_tta_samples,
        })
    }
    
    /// Compute predictive entropy decomposition
    fn compute_entropy_decomposition(&self, predictions: &Array2<F>) -> Result<EntropyDecomposition<F>> {
        let n_samples = predictions.ncols();
        let mut total_entropy = Array1::zeros(n_samples);
        let mut aleatoric_entropy = Array1::zeros(n_samples);
        let mut epistemic_entropy = Array1::zeros(n_samples);
        
        // Convert predictions to probabilities (assuming logits)
        let probabilities = predictions.mapv(|x| F::one() / (F::one() + (-x).exp()));
        
        for i in 0..n_samples {
            let sample_probs = probabilities.column(i);
            
            // Compute mean probability
            let mean_prob = sample_probs.mean().unwrap_or(F::zero());
            
            // Total entropy (entropy of mean prediction)
            if mean_prob > F::zero() && mean_prob < F::one() {
                total_entropy[i] = -mean_prob * mean_prob.ln() - (F::one() - mean_prob) * (F::one() - mean_prob).ln();
            }
            
            // Aleatoric entropy (mean of individual entropies)
            let mut sum_entropy = F::zero();
            let mut count = 0;
            for &prob in sample_probs.iter() {
                if prob > F::zero() && prob < F::one() {
                    sum_entropy = sum_entropy - prob * prob.ln() - (F::one() - prob) * (F::one() - prob).ln();
                    count += 1;
                }
            }
            if count > 0 {
                aleatoric_entropy[i] = sum_entropy / F::from(count).unwrap();
            }
            
            // Epistemic entropy = Total - Aleatoric
            epistemic_entropy[i] = total_entropy[i] - aleatoric_entropy[i];
        }
        
        Ok(EntropyDecomposition {
            total_entropy,
            aleatoric_entropy,
            epistemic_entropy,
        })
    }
    
    /// Compute neural network specific temperature scaling
    fn compute_neural_temperature_scaling<M>(
        &self,
        model: &M,
        x_calibration: &Array2<F>,
        y_calibration: &Array1<F>,
    ) -> Result<NeuralTemperatureScaling<F>>
    where
        M: Fn(&ArrayView2<F>, bool) -> Array1<F>,
    {
        // Get uncalibrated predictions (logits)
        let logits = model(&x_calibration.view(), false);
        
        // Find optimal temperature using gradient descent
        let mut temperature = F::one();
        let learning_rate = F::from(0.01).unwrap();
        let n_iterations = 100;
        
        for _ in 0..n_iterations {
            let scaled_logits = logits.mapv(|x| x / temperature);
            let probabilities = scaled_logits.mapv(|x| F::one() / (F::one() + (-x).exp()));
            
            // Compute negative log-likelihood and its gradient
            let mut loss = F::zero();
            let mut grad = F::zero();
            
            for i in 0..y_calibration.len() {
                let prob = probabilities[i];
                let y_true = y_calibration[i];
                
                // Binary cross-entropy loss
                let eps = F::from(1e-15).unwrap();
                let prob_clipped = prob.max(eps).min(F::one() - eps);
                loss = loss - (y_true * prob_clipped.ln() + (F::one() - y_true) * (F::one() - prob_clipped).ln());
                
                // Gradient with respect to temperature
                let logit = logits[i];
                let sigmoid_deriv = prob * (F::one() - prob);
                grad = grad + (prob - y_true) * sigmoid_deriv * logit / (temperature * temperature);
            }
            
            // Update temperature
            temperature = temperature - learning_rate * grad;
            temperature = temperature.max(F::from(0.01).unwrap()).min(F::from(10.0).unwrap());
        }
        
        // Compute calibrated predictions
        let calibrated_logits = logits.mapv(|x| x / temperature);
        let calibrated_probabilities = calibrated_logits.mapv(|x| F::one() / (F::one() + (-x).exp()));
        
        // Compute calibration metrics
        let pre_calibration_ece = self.compute_expected_calibration_error(&logits.mapv(|x| F::one() / (F::one() + (-x).exp())), y_calibration)?;
        let post_calibration_ece = self.compute_expected_calibration_error(&calibrated_probabilities, y_calibration)?;
        
        Ok(NeuralTemperatureScaling {
            temperature,
            calibrated_probabilities,
            pre_calibration_ece,
            post_calibration_ece,
            calibration_improvement: pre_calibration_ece - post_calibration_ece,
        })
    }
    
    /// Compute SWAG (Stochastic Weight Averaging Gaussian) uncertainty
    fn compute_swag_uncertainty<M>(
        &self,
        model: &M,
        x_test: &Array2<F>,
    ) -> Result<SWAGUncertainty<F>>
    where
        M: Fn(&ArrayView2<F>, bool) -> Array1<F>,
    {
        let n_samples = x_test.nrows();
        let mut predictions = Array2::zeros((self.n_swag_samples, n_samples));
        
        // Simulate SWAG sampling (in practice, this would sample from weight posterior)
        for i in 0..self.n_swag_samples {
            // Add Gaussian noise to simulate weight sampling
            let predictions_sample = model(&x_test.view(), false);
            
            // Add noise to simulate weight uncertainty
            let noise_scale = F::from(0.02).unwrap();
            for j in 0..n_samples {
                let noise = self.sample_gaussian() * noise_scale;
                predictions[[i, j]] = predictions_sample[j] + noise;
            }
        }
        
        // Compute SWAG statistics
        let mean_predictions = predictions.mean_axis(Axis(0)).unwrap();
        let var_predictions = predictions.var_axis(Axis(0), F::zero());
        let std_predictions = var_predictions.mapv(|x| x.sqrt());
        
        // Compute effective sample size
        let effective_sample_size = self.compute_effective_sample_size(&predictions)?;
        
        Ok(SWAGUncertainty {
            predictions,
            mean_predictions,
            std_predictions,
            effective_sample_size,
            n_swag_samples: self.n_swag_samples,
        })
    }
    
    /// Compute disagreement-based uncertainty
    fn compute_disagreement_uncertainty(
        &self,
        mc_predictions: &Array2<F>,
        ensemble_predictions: &Array2<F>,
    ) -> Result<DisagreementUncertainty<F>> {
        let n_samples = mc_predictions.ncols();
        let mut disagreement_scores = Array1::zeros(n_samples);
        let mut confidence_scores = Array1::zeros(n_samples);
        
        for i in 0..n_samples {
            let mc_col = mc_predictions.column(i);
            let ensemble_col = ensemble_predictions.column(i);
            
            // Compute disagreement as variance across methods
            let mc_mean = mc_col.mean().unwrap_or(F::zero());
            let ensemble_mean = ensemble_col.mean().unwrap_or(F::zero());
            
            let method_means = [mc_mean, ensemble_mean];
            let overall_mean = method_means.iter().cloned().sum::<F>() / F::from(method_means.len()).unwrap();
            
            let disagreement = method_means.iter()
                .map(|&x| (x - overall_mean) * (x - overall_mean))
                .sum::<F>() / F::from(method_means.len()).unwrap();
            
            disagreement_scores[i] = disagreement.sqrt();
            
            // Confidence is inverse of disagreement
            confidence_scores[i] = F::one() / (F::one() + disagreement_scores[i]);
        }
        
        Ok(DisagreementUncertainty {
            disagreement_scores,
            confidence_scores,
            method_correlation: self.compute_method_correlation(mc_predictions, ensemble_predictions)?,
        })
    }
    
    // Helper methods
    
    fn compute_epistemic_from_samples(&self, predictions: &Array2<F>) -> Result<Array1<F>> {
        // Epistemic uncertainty as variance across MC samples
        Ok(predictions.var_axis(Axis(0), F::zero()))
    }
    
    fn compute_aleatoric_from_samples(&self, predictions: &Array2<F>) -> Result<Array1<F>> {
        // Simplified aleatoric uncertainty estimation
        let mean_predictions = predictions.mean_axis(Axis(0)).unwrap();
        Ok(mean_predictions.mapv(|p| p * (F::one() - p))) // For binary classification
    }
    
    fn compute_mc_prediction_intervals(&self, predictions: &Array2<F>) -> Result<Array2<F>> {
        let n_samples = predictions.ncols();
        let mut intervals = Array2::zeros((n_samples, 2));
        let alpha = F::from(0.05).unwrap(); // 95% confidence interval
        
        for i in 0..n_samples {
            let mut sample_preds = predictions.column(i).to_vec();
            sample_preds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            
            let lower_idx = (alpha * F::from(sample_preds.len()).unwrap() / F::from(2.0).unwrap()).to_usize().unwrap_or(0);
            let upper_idx = ((F::one() - alpha / F::from(2.0).unwrap()) * F::from(sample_preds.len()).unwrap()).to_usize().unwrap_or(sample_preds.len() - 1);
            
            intervals[[i, 0]] = sample_preds[lower_idx];
            intervals[[i, 1]] = sample_preds[upper_idx];
        }
        
        Ok(intervals)
    }
    
    fn compute_ensemble_prediction_intervals(&self, predictions: &Array2<F>) -> Result<Array2<F>> {
        self.compute_mc_prediction_intervals(predictions)
    }
    
    fn compute_model_diversity(&self, predictions: &Array2<F>) -> Result<F> {
        let n_models = predictions.nrows();
        let mut total_diversity = F::zero();
        let mut count = 0;
        
        for i in 0..n_models {
            for j in (i + 1)..n_models {
                let model_i = predictions.row(i);
                let model_j = predictions.row(j);
                
                // Compute correlation
                let correlation = self.compute_correlation_arrays(&model_i, &model_j)?;
                let diversity = F::one() - correlation.abs(); // Diversity = 1 - |correlation|
                
                total_diversity = total_diversity + diversity;
                count += 1;
            }
        }
        
        if count > 0 {
            Ok(total_diversity / F::from(count).unwrap())
        } else {
            Ok(F::zero())
        }
    }
    
    fn compute_model_mutual_information(&self, predictions: &Array2<F>) -> Result<F> {
        // Simplified mutual information between models
        let n_models = predictions.nrows();
        if n_models < 2 {
            return Ok(F::zero());
        }
        
        let model1 = predictions.row(0);
        let model2 = predictions.row(1);
        
        // Approximate MI using correlation
        let correlation = self.compute_correlation_arrays(&model1, &model2)?;
        let mi = -F::from(0.5).unwrap() * (F::one() - correlation * correlation).ln();
        
        Ok(mi.max(F::zero()))
    }
    
    fn compute_augmentation_consistency(&self, predictions: &Array2<F>) -> Result<F> {
        let n_samples = predictions.ncols();
        let mut consistency_sum = F::zero();
        
        for i in 0..n_samples {
            let sample_preds = predictions.column(i);
            let variance = sample_preds.var(F::zero());
            let consistency = F::one() / (F::one() + variance); // Higher variance = lower consistency
            consistency_sum = consistency_sum + consistency;
        }
        
        Ok(consistency_sum / F::from(n_samples).unwrap())
    }
    
    fn compute_effective_sample_size(&self, predictions: &Array2<F>) -> Result<F> {
        // Simplified effective sample size estimation
        let n_samples = predictions.nrows();
        let autocorrelation = self.compute_autocorrelation(predictions)?;
        let eff_sample_size = F::from(n_samples).unwrap() / (F::one() + F::from(2.0).unwrap() * autocorrelation);
        Ok(eff_sample_size)
    }
    
    fn compute_autocorrelation(&self, predictions: &Array2<F>) -> Result<F> {
        // Simplified autocorrelation computation
        if predictions.nrows() < 2 {
            return Ok(F::zero());
        }
        
        let first_half = predictions.slice(s![..predictions.nrows()/2, ..]);
        let second_half = predictions.slice(s![predictions.nrows()/2.., ..]);
        
        if first_half.nrows() != second_half.nrows() {
            return Ok(F::zero());
        }
        
        let mut correlation_sum = F::zero();
        let mut count = 0;
        
        for i in 0..first_half.ncols() {
            let corr = self.compute_correlation_arrays(&first_half.column(i), &second_half.column(i))?;
            correlation_sum = correlation_sum + corr;
            count += 1;
        }
        
        if count > 0 {
            Ok(correlation_sum / F::from(count).unwrap())
        } else {
            Ok(F::zero())
        }
    }
    
    fn compute_method_correlation(&self, predictions1: &Array2<F>, predictions2: &Array2<F>) -> Result<F> {
        let mean1 = predictions1.mean_axis(Axis(0)).unwrap();
        let mean2 = predictions2.mean_axis(Axis(0)).unwrap();
        
        self.compute_correlation_arrays(&mean1.view(), &mean2.view())
    }
    
    fn compute_correlation_arrays(&self, x: &ArrayView1<F>, y: &ArrayView1<F>) -> Result<F> {
        if x.len() != y.len() {
            return Err(MetricsError::InvalidInput(
                "Arrays must have same length".to_string()
            ));
        }
        
        let n = F::from(x.len()).unwrap();
        let mean_x = x.sum() / n;
        let mean_y = y.sum() / n;
        
        let mut numerator = F::zero();
        let mut sum_sq_x = F::zero();
        let mut sum_sq_y = F::zero();
        
        for (&xi, &yi) in x.iter().zip(y.iter()) {
            let dx = xi - mean_x;
            let dy = yi - mean_y;
            numerator = numerator + dx * dy;
            sum_sq_x = sum_sq_x + dx * dx;
            sum_sq_y = sum_sq_y + dy * dy;
        }
        
        let denominator = (sum_sq_x * sum_sq_y).sqrt();
        
        if denominator > F::zero() {
            Ok(numerator / denominator)
        } else {
            Ok(F::zero())
        }
    }
    
    fn compute_expected_calibration_error(&self, predictions: &Array1<F>, y_true: &Array1<F>) -> Result<F> {
        let n_bins = 10;
        let mut ece = F::zero();
        let mut total_samples = 0;
        
        for bin in 0..n_bins {
            let bin_lower = F::from(bin).unwrap() / F::from(n_bins).unwrap();
            let bin_upper = F::from(bin + 1).unwrap() / F::from(n_bins).unwrap();
            
            let mut bin_predictions = Vec::new();
            let mut bin_labels = Vec::new();
            
            for i in 0..predictions.len() {
                if predictions[i] >= bin_lower && predictions[i] < bin_upper {
                    bin_predictions.push(predictions[i]);
                    bin_labels.push(y_true[i]);
                }
            }
            
            if !bin_predictions.is_empty() {
                let bin_accuracy = bin_labels.iter().cloned().sum::<F>() / F::from(bin_labels.len()).unwrap();
                let bin_confidence = bin_predictions.iter().cloned().sum::<F>() / F::from(bin_predictions.len()).unwrap();
                let bin_weight = bin_predictions.len();
                
                ece = ece + F::from(bin_weight).unwrap() * (bin_accuracy - bin_confidence).abs();
                total_samples += bin_weight;
            }
        }
        
        if total_samples > 0 {
            Ok(ece / F::from(total_samples).unwrap())
        } else {
            Ok(F::zero())
        }
    }
    
    fn sample_gaussian(&self) -> F {
        // Simplified Gaussian sampling using Box-Muller
        let seed = self.random_seed.unwrap_or(42);
        let u1 = F::from((seed % 1000) as f64 / 1000.0).unwrap();
        let u2 = F::from(((seed / 1000) % 1000) as f64 / 1000.0).unwrap();
        
        (-F::from(2.0).unwrap() * u1.ln()).sqrt() 
            * (F::from(2.0 * std::f64::consts::PI).unwrap() * u2).cos()
    }
}

// Result structures

/// Comprehensive deep learning uncertainty analysis
#[derive(Debug, Clone)]
pub struct DeepUncertaintyAnalysis<F: Float> {
    /// Monte Carlo Dropout uncertainty
    pub mc_dropout_uncertainty: MCDropoutUncertainty<F>,
    /// Deep ensemble uncertainty
    pub ensemble_uncertainty: EnsembleUncertainty<F>,
    /// Test-time augmentation uncertainty
    pub tta_uncertainty: TTAUncertainty<F>,
    /// Predictive entropy decomposition
    pub entropy_decomposition: EntropyDecomposition<F>,
    /// Temperature scaling results
    pub temperature_scaling: Option<NeuralTemperatureScaling<F>>,
    /// SWAG uncertainty
    pub swag_uncertainty: Option<SWAGUncertainty<F>>,
    /// Disagreement-based uncertainty
    pub disagreement_uncertainty: DisagreementUncertainty<F>,
    /// Sample size
    pub sample_size: usize,
}

/// Monte Carlo Dropout uncertainty results
#[derive(Debug, Clone)]
pub struct MCDropoutUncertainty<F: Float> {
    /// All MC predictions [n_mc_samples, n_test_samples]
    pub predictions: Array2<F>,
    /// Mean predictions across MC samples
    pub mean_predictions: Array1<F>,
    /// Standard deviation of predictions
    pub std_predictions: Array1<F>,
    /// Epistemic uncertainty
    pub epistemic_uncertainty: Array1<F>,
    /// Aleatoric uncertainty
    pub aleatoric_uncertainty: Array1<F>,
    /// Prediction intervals [n_samples, 2]
    pub prediction_intervals: Array2<F>,
    /// Number of MC samples
    pub n_samples: usize,
}

/// Deep ensemble uncertainty results
#[derive(Debug, Clone)]
pub struct EnsembleUncertainty<F: Float> {
    /// All ensemble predictions [n_models, n_test_samples]
    pub predictions: Array2<F>,
    /// Mean predictions across ensemble
    pub mean_predictions: Array1<F>,
    /// Standard deviation of predictions
    pub std_predictions: Array1<F>,
    /// Model diversity score
    pub model_diversity: F,
    /// Prediction intervals [n_samples, 2]
    pub prediction_intervals: Array2<F>,
    /// Mutual information between models
    pub mutual_information: F,
    /// Number of ensemble members
    pub n_models: usize,
}

/// Test-time augmentation uncertainty results
#[derive(Debug, Clone)]
pub struct TTAUncertainty<F: Float> {
    /// All TTA predictions [n_augmentations, n_test_samples]
    pub predictions: Array2<F>,
    /// Mean predictions across augmentations
    pub mean_predictions: Array1<F>,
    /// Standard deviation of predictions
    pub std_predictions: Array1<F>,
    /// Augmentation consistency score
    pub consistency_score: F,
    /// Number of augmentations
    pub n_augmentations: usize,
}

/// Predictive entropy decomposition
#[derive(Debug, Clone)]
pub struct EntropyDecomposition<F: Float> {
    /// Total entropy
    pub total_entropy: Array1<F>,
    /// Aleatoric entropy (irreducible)
    pub aleatoric_entropy: Array1<F>,
    /// Epistemic entropy (reducible)
    pub epistemic_entropy: Array1<F>,
}

/// Neural network temperature scaling results
#[derive(Debug, Clone)]
pub struct NeuralTemperatureScaling<F: Float> {
    /// Optimal temperature parameter
    pub temperature: F,
    /// Calibrated probabilities
    pub calibrated_probabilities: Array1<F>,
    /// Expected calibration error before scaling
    pub pre_calibration_ece: F,
    /// Expected calibration error after scaling
    pub post_calibration_ece: F,
    /// Calibration improvement
    pub calibration_improvement: F,
}

/// SWAG uncertainty results
#[derive(Debug, Clone)]
pub struct SWAGUncertainty<F: Float> {
    /// All SWAG predictions [n_swag_samples, n_test_samples]
    pub predictions: Array2<F>,
    /// Mean predictions
    pub mean_predictions: Array1<F>,
    /// Standard deviation of predictions
    pub std_predictions: Array1<F>,
    /// Effective sample size
    pub effective_sample_size: F,
    /// Number of SWAG samples
    pub n_swag_samples: usize,
}

/// Disagreement-based uncertainty
#[derive(Debug, Clone)]
pub struct DisagreementUncertainty<F: Float> {
    /// Disagreement scores between methods
    pub disagreement_scores: Array1<F>,
    /// Confidence scores (inverse of disagreement)
    pub confidence_scores: Array1<F>,
    /// Correlation between uncertainty methods
    pub method_correlation: F,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // Mock neural network model for testing
    fn mock_neural_model(x: &ArrayView2<f64>, _dropout: bool) -> Array1<f64> {
        x.map_axis(Axis(1), |row| row.sum())
    }

    // Mock ensemble models
    fn mock_ensemble_model_1(x: &ArrayView2<f64>) -> Array1<f64> {
        x.map_axis(Axis(1), |row| row.sum() * 1.1)
    }

    fn mock_ensemble_model_2(x: &ArrayView2<f64>) -> Array1<f64> {
        x.map_axis(Axis(1), |row| row.sum() * 0.9)
    }

    // Mock augmentation function
    fn mock_augmentation(x: &ArrayView2<f64>) -> Array3<f64> {
        let n_aug = 3;
        let mut augmented = Array3::zeros((n_aug, x.nrows(), x.ncols()));
        
        for i in 0..n_aug {
            for j in 0..x.nrows() {
                for k in 0..x.ncols() {
                    augmented[[i, j, k]] = x[[j, k]] * (1.0 + 0.1 * (i as f64 - 1.0));
                }
            }
        }
        
        augmented
    }

    #[test]
    fn test_deep_uncertainty_quantifier_creation() {
        let quantifier = DeepUncertaintyQuantifier::<f64>::new()
            .with_mc_dropout(50, 0.2)
            .with_ensemble(3)
            .with_tta(5)
            .with_temperature_scaling(true)
            .with_swag(true, 10)
            .with_seed(42);

        assert_eq!(quantifier.n_mc_dropout_samples, 50);
        assert_eq!(quantifier.dropout_rate, 0.2);
        assert_eq!(quantifier.n_ensemble_members, 3);
        assert_eq!(quantifier.n_tta_samples, 5);
        assert!(quantifier.enable_temperature_scaling);
        assert!(quantifier.enable_swag);
        assert_eq!(quantifier.n_swag_samples, 10);
        assert_eq!(quantifier.random_seed, Some(42));
    }

    #[test]
    fn test_mc_dropout_uncertainty() {
        let quantifier = DeepUncertaintyQuantifier::<f64>::new()
            .with_mc_dropout(10, 0.1)
            .with_seed(123);

        let x_test = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let mc_uncertainty = quantifier
            .compute_mc_dropout_uncertainty(&mock_neural_model, &x_test)
            .unwrap();

        assert_eq!(mc_uncertainty.predictions.nrows(), 10);
        assert_eq!(mc_uncertainty.predictions.ncols(), 3);
        assert_eq!(mc_uncertainty.mean_predictions.len(), 3);
        assert_eq!(mc_uncertainty.std_predictions.len(), 3);
        assert_eq!(mc_uncertainty.prediction_intervals.nrows(), 3);
        assert_eq!(mc_uncertainty.prediction_intervals.ncols(), 2);
    }

    #[test]
    fn test_ensemble_uncertainty() {
        let quantifier = DeepUncertaintyQuantifier::<f64>::new().with_seed(456);

        let x_test = array![[1.0, 2.0], [3.0, 4.0]];
        let ensemble_models = vec![mock_ensemble_model_1, mock_ensemble_model_2];

        let ensemble_uncertainty = quantifier
            .compute_ensemble_uncertainty(&ensemble_models, &x_test)
            .unwrap();

        assert_eq!(ensemble_uncertainty.predictions.nrows(), 2);
        assert_eq!(ensemble_uncertainty.predictions.ncols(), 2);
        assert_eq!(ensemble_uncertainty.mean_predictions.len(), 2);
        assert_eq!(ensemble_uncertainty.n_models, 2);
        assert!(ensemble_uncertainty.model_diversity >= 0.0);
    }

    #[test]
    fn test_tta_uncertainty() {
        let quantifier = DeepUncertaintyQuantifier::<f64>::new()
            .with_tta(3)
            .with_seed(789);

        let x_test = array![[1.0, 2.0], [3.0, 4.0]];

        let tta_uncertainty = quantifier
            .compute_tta_uncertainty(&mock_neural_model, &mock_augmentation, &x_test)
            .unwrap();

        assert_eq!(tta_uncertainty.predictions.nrows(), 3);
        assert_eq!(tta_uncertainty.predictions.ncols(), 2);
        assert_eq!(tta_uncertainty.mean_predictions.len(), 2);
        assert!(tta_uncertainty.consistency_score >= 0.0);
        assert!(tta_uncertainty.consistency_score <= 1.0);
    }

    #[test]
    fn test_entropy_decomposition() {
        let quantifier = DeepUncertaintyQuantifier::<f64>::new();

        // Create mock predictions (logits)
        let predictions = array![
            [0.5, 1.0, -0.5],
            [0.7, 0.8, -0.3],
            [0.3, 1.2, -0.7],
        ];

        let entropy_decomp = quantifier
            .compute_entropy_decomposition(&predictions)
            .unwrap();

        assert_eq!(entropy_decomp.total_entropy.len(), 3);
        assert_eq!(entropy_decomp.aleatoric_entropy.len(), 3);
        assert_eq!(entropy_decomp.epistemic_entropy.len(), 3);

        // All entropies should be non-negative
        for &entropy in entropy_decomp.total_entropy.iter() {
            assert!(entropy >= 0.0);
        }
        for &entropy in entropy_decomp.aleatoric_entropy.iter() {
            assert!(entropy >= 0.0);
        }
    }

    #[test]
    fn test_temperature_scaling() {
        let quantifier = DeepUncertaintyQuantifier::<f64>::new().with_seed(321);

        let x_cal = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y_cal = array![0.0, 0.5, 1.0];

        let temp_scaling = quantifier
            .compute_neural_temperature_scaling(&mock_neural_model, &x_cal, &y_cal)
            .unwrap();

        assert!(temp_scaling.temperature > 0.0);
        assert_eq!(temp_scaling.calibrated_probabilities.len(), 3);
        assert!(temp_scaling.pre_calibration_ece >= 0.0);
        assert!(temp_scaling.post_calibration_ece >= 0.0);

        // All calibrated probabilities should be valid probabilities
        for &prob in temp_scaling.calibrated_probabilities.iter() {
            assert!(prob >= 0.0 && prob <= 1.0);
        }
    }

    #[test]
    fn test_comprehensive_deep_uncertainty() {
        let quantifier = DeepUncertaintyQuantifier::<f64>::new()
            .with_mc_dropout(5, 0.1)
            .with_ensemble(2)
            .with_tta(3)
            .with_temperature_scaling(true)
            .with_swag(true, 5)
            .with_seed(42);

        let x_test = array![[1.0, 2.0], [3.0, 4.0]];
        let x_cal = array![[0.5, 1.5], [2.5, 3.5]];
        let y_cal = array![0.0, 1.0];
        let ensemble_models = vec![mock_ensemble_model_1, mock_ensemble_model_2];

        let analysis = quantifier
            .compute_deep_uncertainty(
                &mock_neural_model,
                &ensemble_models,
                &mock_augmentation,
                &x_test,
                Some(&x_cal),
                Some(&y_cal),
            )
            .unwrap();

        assert_eq!(analysis.sample_size, 2);
        assert!(analysis.temperature_scaling.is_some());
        assert!(analysis.swag_uncertainty.is_some());
        assert_eq!(analysis.mc_dropout_uncertainty.mean_predictions.len(), 2);
        assert_eq!(analysis.ensemble_uncertainty.mean_predictions.len(), 2);
        assert_eq!(analysis.tta_uncertainty.mean_predictions.len(), 2);
    }
}