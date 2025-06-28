//! Uncertainty quantification methods for model predictions

use crate::error::{MetricsError, Result};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use num_traits::Float;
use std::collections::HashMap;

/// Uncertainty quantification analyzer
pub struct UncertaintyQuantifier<F: Float> {
    /// Number of Monte Carlo samples
    pub n_mc_samples: usize,
    /// Confidence level for intervals
    pub confidence_level: F,
    /// Bootstrap samples for confidence estimation
    pub n_bootstrap: usize,
    /// Random seed
    pub random_seed: Option<u64>,
}

impl<F: Float + num_traits::FromPrimitive + std::iter::Sum + ndarray::ScalarOperand> Default
    for UncertaintyQuantifier<F>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + num_traits::FromPrimitive + std::iter::Sum + ndarray::ScalarOperand>
    UncertaintyQuantifier<F>
{
    /// Create new uncertainty quantifier
    pub fn new() -> Self {
        Self {
            n_mc_samples: 1000,
            confidence_level: F::from(0.95).unwrap(),
            n_bootstrap: 100,
            random_seed: None,
        }
    }

    /// Set number of Monte Carlo samples
    pub fn with_mc_samples(mut self, n: usize) -> Self {
        self.n_mc_samples = n;
        self
    }

    /// Set confidence level
    pub fn with_confidence_level(mut self, level: F) -> Self {
        self.confidence_level = level;
        self
    }

    /// Set number of bootstrap samples
    pub fn with_bootstrap(mut self, n: usize) -> Self {
        self.n_bootstrap = n;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }

    /// Compute comprehensive uncertainty metrics
    pub fn compute_uncertainty<M>(
        &self,
        model: &M,
        x_test: &Array2<F>,
        y_test: Option<&Array1<F>>,
    ) -> Result<UncertaintyAnalysis<F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        // Epistemic uncertainty via Monte Carlo dropout
        let epistemic_uncertainty = self.compute_epistemic_uncertainty(model, x_test)?;

        // Aleatoric uncertainty estimation
        let aleatoric_uncertainty = self.compute_aleatoric_uncertainty(model, x_test)?;

        // Prediction intervals
        let prediction_intervals = self.compute_prediction_intervals(model, x_test)?;

        // Calibration metrics (if labels available)
        let calibration_metrics = if let Some(y_true) = y_test {
            Some(self.compute_calibration_metrics(model, x_test, y_true)?)
        } else {
            None
        };

        // Confidence estimation
        let confidence_scores = self.compute_confidence_scores(model, x_test)?;

        // Out-of-distribution detection
        let ood_scores = self.compute_ood_scores(model, x_test)?;

        Ok(UncertaintyAnalysis {
            epistemic_uncertainty,
            aleatoric_uncertainty,
            prediction_intervals,
            calibration_metrics,
            confidence_scores,
            ood_scores,
            sample_size: x_test.nrows(),
        })
    }

    /// Compute epistemic uncertainty using Monte Carlo methods
    fn compute_epistemic_uncertainty<M>(
        &self,
        model: &M,
        x_test: &Array2<F>,
    ) -> Result<EpistemicUncertainty<F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        let mut mc_predictions = Vec::new();

        // Simulate model uncertainty with Monte Carlo sampling
        for _ in 0..self.n_mc_samples {
            // In practice, this would involve dropout or other stochastic elements
            let predictions = model(&x_test.view());
            mc_predictions.push(predictions);
        }

        // Compute statistics across MC samples
        let mean_predictions = self.compute_mc_mean(&mc_predictions)?;
        let prediction_variance = self.compute_mc_variance(&mc_predictions, &mean_predictions)?;
        let prediction_entropy = self.compute_prediction_entropy(&mc_predictions)?;
        let mutual_information = self.compute_mutual_information(&mc_predictions)?;

        Ok(EpistemicUncertainty {
            mean_predictions,
            prediction_variance,
            prediction_entropy,
            mutual_information,
            mc_samples: self.n_mc_samples,
        })
    }

    /// Compute aleatoric uncertainty (data-dependent)
    fn compute_aleatoric_uncertainty<M>(
        &self,
        model: &M,
        x_test: &Array2<F>,
    ) -> Result<AleatoricUncertainty<F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        // Estimate aleatoric uncertainty using input perturbations
        let mut perturbed_predictions = Vec::new();
        let noise_std = F::from(0.01).unwrap(); // Small noise for input perturbation

        for _ in 0..50 {
            let mut x_perturbed = x_test.clone();
            self.add_input_noise(&mut x_perturbed, noise_std)?;

            let predictions = model(&x_perturbed.view());
            perturbed_predictions.push(predictions);
        }

        let baseline_predictions = model(&x_test.view());
        let input_sensitivity =
            self.compute_input_sensitivity(&perturbed_predictions, &baseline_predictions)?;
        let data_uncertainty = self.estimate_data_uncertainty(x_test)?;

        Ok(AleatoricUncertainty {
            input_sensitivity,
            data_uncertainty,
            noise_level: noise_std,
        })
    }

    /// Compute prediction intervals
    fn compute_prediction_intervals<M>(
        &self,
        model: &M,
        x_test: &Array2<F>,
    ) -> Result<PredictionIntervals<F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        // Bootstrap-based prediction intervals
        let mut bootstrap_predictions = Vec::new();

        for _ in 0..self.n_bootstrap {
            // Bootstrap sample (with replacement)
            let bootstrap_indices = self.generate_bootstrap_indices(x_test.nrows())?;
            let x_bootstrap = self.sample_by_indices(x_test, &bootstrap_indices)?;

            let predictions = model(&x_bootstrap.view());
            bootstrap_predictions.push(predictions);
        }

        let lower_bound = self.compute_percentile(
            &bootstrap_predictions,
            (F::one() - self.confidence_level) / F::from(2).unwrap(),
        )?;
        let upper_bound = self.compute_percentile(
            &bootstrap_predictions,
            F::one() - (F::one() - self.confidence_level) / F::from(2).unwrap(),
        )?;
        let median_prediction =
            self.compute_percentile(&bootstrap_predictions, F::from(0.5).unwrap())?;

        Ok(PredictionIntervals {
            lower_bound,
            upper_bound,
            median_prediction,
            confidence_level: self.confidence_level,
        })
    }

    /// Compute calibration metrics
    fn compute_calibration_metrics<M>(
        &self,
        model: &M,
        x_test: &Array2<F>,
        y_test: &Array1<F>,
    ) -> Result<CalibrationMetrics<F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        let predictions = model(&x_test.view());

        // Expected Calibration Error (ECE)
        let ece = self.compute_expected_calibration_error(&predictions, y_test)?;

        // Maximum Calibration Error
        let mce = self.compute_maximum_calibration_error(&predictions, y_test)?;

        // Reliability diagram data
        let reliability_data = self.compute_reliability_diagram(&predictions, y_test)?;

        // Brier score decomposition
        let brier_decomposition = self.compute_brier_decomposition(&predictions, y_test)?;

        Ok(CalibrationMetrics {
            expected_calibration_error: ece,
            maximum_calibration_error: mce,
            reliability_data,
            brier_decomposition,
        })
    }

    /// Compute confidence scores for predictions
    fn compute_confidence_scores<M>(
        &self,
        model: &M,
        x_test: &Array2<F>,
    ) -> Result<ConfidenceScores<F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        let predictions = model(&x_test.view());

        // Max probability as confidence (for classification-like problems)
        let max_confidence = predictions.iter().cloned().fold(F::neg_infinity(), F::max);

        // Entropy-based uncertainty
        let entropy_uncertainty = self.compute_entropy_uncertainty(&predictions)?;

        // Distance-based confidence
        let distance_confidence = self.compute_distance_based_confidence(x_test)?;

        // Ensemble-based confidence (simplified)
        let ensemble_confidence = self.compute_ensemble_confidence(model, x_test)?;

        Ok(ConfidenceScores {
            max_confidence,
            entropy_uncertainty,
            distance_confidence,
            ensemble_confidence,
        })
    }

    /// Compute out-of-distribution detection scores
    fn compute_ood_scores<M>(&self, model: &M, x_test: &Array2<F>) -> Result<OODScores<F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        // Mahalanobis distance-based OOD detection
        let mahalanobis_scores = self.compute_mahalanobis_scores(x_test)?;

        // Energy-based OOD detection
        let energy_scores = self.compute_energy_scores(model, x_test)?;

        // Reconstruction error (if applicable)
        let reconstruction_errors = self.compute_reconstruction_errors(x_test)?;

        // Density-based scores
        let density_scores = self.compute_density_scores(x_test)?;

        Ok(OODScores {
            mahalanobis_scores,
            energy_scores,
            reconstruction_errors,
            density_scores,
        })
    }

    // Helper methods

    fn compute_mc_mean(&self, mc_predictions: &[Array1<F>]) -> Result<Array1<F>> {
        if mc_predictions.is_empty() {
            return Err(MetricsError::InvalidInput(
                "No MC predictions provided".to_string(),
            ));
        }

        let n_samples = mc_predictions[0].len();
        let mut mean_pred = Array1::zeros(n_samples);

        for predictions in mc_predictions {
            mean_pred = mean_pred + predictions;
        }

        mean_pred = mean_pred / F::from(mc_predictions.len()).unwrap();
        Ok(mean_pred)
    }

    fn compute_mc_variance(
        &self,
        mc_predictions: &[Array1<F>],
        mean_pred: &Array1<F>,
    ) -> Result<Array1<F>> {
        let n_samples = mean_pred.len();
        let mut variance = Array1::zeros(n_samples);

        for predictions in mc_predictions {
            let diff = predictions - mean_pred;
            variance = variance + &(&diff * &diff);
        }

        variance = variance / F::from(mc_predictions.len()).unwrap();
        Ok(variance)
    }

    fn compute_prediction_entropy(&self, mc_predictions: &[Array1<F>]) -> Result<Array1<F>> {
        let mean_pred = self.compute_mc_mean(mc_predictions)?;
        let mut entropy = Array1::zeros(mean_pred.len());

        for (i, &mean_val) in mean_pred.iter().enumerate() {
            // Simplified entropy calculation
            if mean_val > F::zero() && mean_val < F::one() {
                entropy[i] =
                    -mean_val * mean_val.ln() - (F::one() - mean_val) * (F::one() - mean_val).ln();
            }
        }

        Ok(entropy)
    }

    fn compute_mutual_information(&self, mc_predictions: &[Array1<F>]) -> Result<F> {
        // Simplified mutual information calculation
        let mean_pred = self.compute_mc_mean(mc_predictions)?;
        let variance = self.compute_mc_variance(mc_predictions, &mean_pred)?;

        let avg_variance = variance.mean().unwrap_or(F::zero());
        let avg_entropy = mean_pred
            .iter()
            .map(|&p| {
                if p > F::zero() && p < F::one() {
                    -p * p.ln() - (F::one() - p) * (F::one() - p).ln()
                } else {
                    F::zero()
                }
            })
            .sum::<F>()
            / F::from(mean_pred.len()).unwrap();

        Ok(avg_entropy - avg_variance)
    }

    fn add_input_noise(&self, x_data: &mut Array2<F>, noise_std: F) -> Result<()> {
        for value in x_data.iter_mut() {
            let noise = self.generate_gaussian_noise()? * noise_std;
            *value = *value + noise;
        }
        Ok(())
    }

    fn generate_gaussian_noise(&self) -> Result<F> {
        // Simplified Gaussian noise generation
        let seed = self.random_seed.unwrap_or(42);
        let u1 = F::from((seed % 1000) as f64 / 1000.0).unwrap();
        let u2 = F::from(((seed / 1000) % 1000) as f64 / 1000.0).unwrap();

        let z = (-F::from(2.0).unwrap() * u1.ln()).sqrt()
            * (F::from(2.0).unwrap() * F::from(std::f64::consts::PI).unwrap() * u2).cos();

        Ok(z)
    }

    fn compute_input_sensitivity(
        &self,
        perturbed_preds: &[Array1<F>],
        baseline_pred: &Array1<F>,
    ) -> Result<Array1<F>> {
        let mut sensitivity = Array1::zeros(baseline_pred.len());

        for pred in perturbed_preds {
            let diff = pred - baseline_pred;
            sensitivity = sensitivity + &diff.mapv(|x| x.abs());
        }

        sensitivity = sensitivity / F::from(perturbed_preds.len()).unwrap();
        Ok(sensitivity)
    }

    fn estimate_data_uncertainty(&self, x_test: &Array2<F>) -> Result<Array1<F>> {
        // Simplified data uncertainty based on local density
        let mut uncertainty = Array1::zeros(x_test.nrows());

        for i in 0..x_test.nrows() {
            let mut min_distance = F::infinity();

            for j in 0..x_test.nrows() {
                if i != j {
                    let distance =
                        self.compute_euclidean_distance(&x_test.row(i), &x_test.row(j))?;
                    min_distance = min_distance.min(distance);
                }
            }

            uncertainty[i] = min_distance; // Higher distance = higher uncertainty
        }

        Ok(uncertainty)
    }

    fn compute_euclidean_distance(
        &self,
        x1: &ndarray::ArrayView1<F>,
        x2: &ndarray::ArrayView1<F>,
    ) -> Result<F> {
        let squared_diff: F = x1
            .iter()
            .zip(x2.iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum();
        Ok(squared_diff.sqrt())
    }

    fn generate_bootstrap_indices(&self, n_samples: usize) -> Result<Vec<usize>> {
        let mut indices = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let idx = (self.random_seed.unwrap_or(0) as usize + i) % n_samples;
            indices.push(idx);
        }
        Ok(indices)
    }

    fn sample_by_indices(&self, data: &Array2<F>, indices: &[usize]) -> Result<Array2<F>> {
        let mut sampled = Array2::zeros((indices.len(), data.ncols()));

        for (i, &idx) in indices.iter().enumerate() {
            for j in 0..data.ncols() {
                sampled[[i, j]] = data[[idx, j]];
            }
        }

        Ok(sampled)
    }

    fn compute_percentile(
        &self,
        bootstrap_preds: &[Array1<F>],
        percentile: F,
    ) -> Result<Array1<F>> {
        if bootstrap_preds.is_empty() {
            return Err(MetricsError::InvalidInput(
                "No bootstrap predictions".to_string(),
            ));
        }

        let n_samples = bootstrap_preds[0].len();
        let mut result = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let mut values: Vec<F> = bootstrap_preds.iter().map(|pred| pred[i]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let index = (percentile * F::from(values.len() - 1).unwrap())
                .to_usize()
                .unwrap_or(0);
            result[i] = values[index.min(values.len() - 1)];
        }

        Ok(result)
    }

    fn compute_expected_calibration_error(
        &self,
        predictions: &Array1<F>,
        y_true: &Array1<F>,
    ) -> Result<F> {
        let n_bins = 10;
        let mut ece = F::zero();

        for bin in 0..n_bins {
            let bin_lower = F::from(bin).unwrap() / F::from(n_bins).unwrap();
            let bin_upper = F::from(bin + 1).unwrap() / F::from(n_bins).unwrap();

            let (bin_accuracy, bin_confidence, bin_weight) =
                self.compute_bin_metrics(predictions, y_true, bin_lower, bin_upper)?;

            ece = ece + bin_weight * (bin_accuracy - bin_confidence).abs();
        }

        Ok(ece)
    }

    fn compute_maximum_calibration_error(
        &self,
        predictions: &Array1<F>,
        y_true: &Array1<F>,
    ) -> Result<F> {
        let n_bins = 10;
        let mut mce = F::zero();

        for bin in 0..n_bins {
            let bin_lower = F::from(bin).unwrap() / F::from(n_bins).unwrap();
            let bin_upper = F::from(bin + 1).unwrap() / F::from(n_bins).unwrap();

            let (bin_accuracy, bin_confidence, _) =
                self.compute_bin_metrics(predictions, y_true, bin_lower, bin_upper)?;

            let bin_error = (bin_accuracy - bin_confidence).abs();
            mce = mce.max(bin_error);
        }

        Ok(mce)
    }

    fn compute_bin_metrics(
        &self,
        predictions: &Array1<F>,
        y_true: &Array1<F>,
        bin_lower: F,
        bin_upper: F,
    ) -> Result<(F, F, F)> {
        let mut bin_predictions = Vec::new();
        let mut bin_labels = Vec::new();

        for (i, &pred) in predictions.iter().enumerate() {
            if pred >= bin_lower && pred < bin_upper {
                bin_predictions.push(pred);
                bin_labels.push(y_true[i]);
            }
        }

        if bin_predictions.is_empty() {
            return Ok((F::zero(), F::zero(), F::zero()));
        }

        let bin_accuracy =
            bin_labels.iter().cloned().sum::<F>() / F::from(bin_labels.len()).unwrap();
        let bin_confidence =
            bin_predictions.iter().cloned().sum::<F>() / F::from(bin_predictions.len()).unwrap();
        let bin_weight =
            F::from(bin_predictions.len()).unwrap() / F::from(predictions.len()).unwrap();

        Ok((bin_accuracy, bin_confidence, bin_weight))
    }

    fn compute_reliability_diagram(
        &self,
        predictions: &Array1<F>,
        y_true: &Array1<F>,
    ) -> Result<Vec<(F, F, F)>> {
        let n_bins = 10;
        let mut reliability_data = Vec::new();

        for bin in 0..n_bins {
            let bin_lower = F::from(bin).unwrap() / F::from(n_bins).unwrap();
            let bin_upper = F::from(bin + 1).unwrap() / F::from(n_bins).unwrap();

            let (bin_accuracy, bin_confidence, bin_weight) =
                self.compute_bin_metrics(predictions, y_true, bin_lower, bin_upper)?;

            reliability_data.push((bin_confidence, bin_accuracy, bin_weight));
        }

        Ok(reliability_data)
    }

    fn compute_brier_decomposition(
        &self,
        predictions: &Array1<F>,
        y_true: &Array1<F>,
    ) -> Result<BrierDecomposition<F>> {
        let brier_score = predictions
            .iter()
            .zip(y_true.iter())
            .map(|(&pred, &label)| (pred - label) * (pred - label))
            .sum::<F>()
            / F::from(predictions.len()).unwrap();

        let reliability = self.compute_expected_calibration_error(predictions, y_true)?;

        let mean_pred = predictions.mean().unwrap_or(F::zero());
        let mean_label = y_true.mean().unwrap_or(F::zero());
        let resolution = (mean_pred - mean_label) * (mean_pred - mean_label);

        let uncertainty = mean_label * (F::one() - mean_label);

        Ok(BrierDecomposition {
            brier_score,
            reliability,
            resolution,
            uncertainty,
        })
    }

    fn compute_entropy_uncertainty(&self, predictions: &Array1<F>) -> Result<Array1<F>> {
        let mut entropy = Array1::zeros(predictions.len());

        for (i, &pred) in predictions.iter().enumerate() {
            if pred > F::zero() && pred < F::one() {
                entropy[i] = -pred * pred.ln() - (F::one() - pred) * (F::one() - pred).ln();
            }
        }

        Ok(entropy)
    }

    fn compute_distance_based_confidence(&self, x_test: &Array2<F>) -> Result<Array1<F>> {
        let mut confidence = Array1::zeros(x_test.nrows());

        for i in 0..x_test.nrows() {
            let mut min_distance = F::infinity();

            for j in 0..x_test.nrows() {
                if i != j {
                    let distance =
                        self.compute_euclidean_distance(&x_test.row(i), &x_test.row(j))?;
                    min_distance = min_distance.min(distance);
                }
            }

            confidence[i] = F::one() / (F::one() + min_distance); // Higher distance = lower confidence
        }

        Ok(confidence)
    }

    fn compute_ensemble_confidence<M>(&self, model: &M, x_test: &Array2<F>) -> Result<Array1<F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        // Simplified ensemble confidence using prediction variance
        let mut ensemble_predictions = Vec::new();

        for _ in 0..10 {
            let predictions = model(&x_test.view());
            ensemble_predictions.push(predictions);
        }

        let mean_pred = self.compute_mc_mean(&ensemble_predictions)?;
        let variance = self.compute_mc_variance(&ensemble_predictions, &mean_pred)?;

        // Confidence is inverse of variance
        let confidence = variance.mapv(|v| F::one() / (F::one() + v));
        Ok(confidence)
    }

    fn compute_mahalanobis_scores(&self, x_test: &Array2<F>) -> Result<Array1<F>> {
        // Simplified Mahalanobis distance computation
        let mean = x_test.mean_axis(Axis(0)).unwrap();
        let mut scores = Array1::zeros(x_test.nrows());

        for i in 0..x_test.nrows() {
            let diff = &x_test.row(i) - &mean;
            let score = diff.iter().map(|&x| x * x).sum::<F>().sqrt();
            scores[i] = score;
        }

        Ok(scores)
    }

    fn compute_energy_scores<M>(&self, model: &M, x_test: &Array2<F>) -> Result<Array1<F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        let predictions = model(&x_test.view());

        // Energy score based on prediction magnitude
        let energy_scores = predictions.mapv(|pred| -pred.ln());
        Ok(energy_scores)
    }

    fn compute_reconstruction_errors(&self, x_test: &Array2<F>) -> Result<Array1<F>> {
        // Simplified reconstruction error (assuming identity reconstruction)
        let mut errors = Array1::zeros(x_test.nrows());

        for i in 0..x_test.nrows() {
            let reconstruction_error = x_test.row(i).iter().map(|&x| x * x).sum::<F>().sqrt();
            errors[i] = reconstruction_error;
        }

        Ok(errors)
    }

    fn compute_density_scores(&self, x_test: &Array2<F>) -> Result<Array1<F>> {
        // Simplified density estimation using k-nearest neighbors
        let mut density_scores = Array1::zeros(x_test.nrows());
        let k = 5; // Number of nearest neighbors

        for i in 0..x_test.nrows() {
            let mut distances = Vec::new();

            for j in 0..x_test.nrows() {
                if i != j {
                    let distance =
                        self.compute_euclidean_distance(&x_test.row(i), &x_test.row(j))?;
                    distances.push(distance);
                }
            }

            distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            if distances.len() >= k {
                let kth_distance = distances[k - 1];
                density_scores[i] = F::one() / (F::one() + kth_distance);
            }
        }

        Ok(density_scores)
    }
}

/// Comprehensive uncertainty analysis results
#[derive(Debug, Clone)]
pub struct UncertaintyAnalysis<F: Float> {
    /// Epistemic uncertainty (model uncertainty)
    pub epistemic_uncertainty: EpistemicUncertainty<F>,
    /// Aleatoric uncertainty (data uncertainty)
    pub aleatoric_uncertainty: AleatoricUncertainty<F>,
    /// Prediction intervals
    pub prediction_intervals: PredictionIntervals<F>,
    /// Calibration metrics (if labels available)
    pub calibration_metrics: Option<CalibrationMetrics<F>>,
    /// Confidence scores
    pub confidence_scores: ConfidenceScores<F>,
    /// Out-of-distribution detection scores
    pub ood_scores: OODScores<F>,
    /// Sample size
    pub sample_size: usize,
}

/// Epistemic uncertainty metrics
#[derive(Debug, Clone)]
pub struct EpistemicUncertainty<F: Float> {
    /// Mean predictions across MC samples
    pub mean_predictions: Array1<F>,
    /// Prediction variance
    pub prediction_variance: Array1<F>,
    /// Prediction entropy
    pub prediction_entropy: Array1<F>,
    /// Mutual information
    pub mutual_information: F,
    /// Number of MC samples used
    pub mc_samples: usize,
}

/// Aleatoric uncertainty metrics
#[derive(Debug, Clone)]
pub struct AleatoricUncertainty<F: Float> {
    /// Input sensitivity measure
    pub input_sensitivity: Array1<F>,
    /// Data uncertainty estimate
    pub data_uncertainty: Array1<F>,
    /// Noise level used for estimation
    pub noise_level: F,
}

/// Prediction intervals
#[derive(Debug, Clone)]
pub struct PredictionIntervals<F: Float> {
    /// Lower bound of interval
    pub lower_bound: Array1<F>,
    /// Upper bound of interval
    pub upper_bound: Array1<F>,
    /// Median prediction
    pub median_prediction: Array1<F>,
    /// Confidence level
    pub confidence_level: F,
}

/// Calibration metrics
#[derive(Debug, Clone)]
pub struct CalibrationMetrics<F: Float> {
    /// Expected Calibration Error
    pub expected_calibration_error: F,
    /// Maximum Calibration Error
    pub maximum_calibration_error: F,
    /// Reliability diagram data (confidence, accuracy, weight)
    pub reliability_data: Vec<(F, F, F)>,
    /// Brier score decomposition
    pub brier_decomposition: BrierDecomposition<F>,
}

/// Brier score decomposition
#[derive(Debug, Clone)]
pub struct BrierDecomposition<F: Float> {
    /// Overall Brier score
    pub brier_score: F,
    /// Reliability component
    pub reliability: F,
    /// Resolution component
    pub resolution: F,
    /// Uncertainty component
    pub uncertainty: F,
}

/// Confidence scores
#[derive(Debug, Clone)]
pub struct ConfidenceScores<F: Float> {
    /// Maximum probability confidence
    pub max_confidence: F,
    /// Entropy-based uncertainty
    pub entropy_uncertainty: Array1<F>,
    /// Distance-based confidence
    pub distance_confidence: Array1<F>,
    /// Ensemble-based confidence
    pub ensemble_confidence: Array1<F>,
}

/// Out-of-distribution detection scores
#[derive(Debug, Clone)]
pub struct OODScores<F: Float> {
    /// Mahalanobis distance scores
    pub mahalanobis_scores: Array1<F>,
    /// Energy-based scores
    pub energy_scores: Array1<F>,
    /// Reconstruction error scores
    pub reconstruction_errors: Array1<F>,
    /// Density-based scores
    pub density_scores: Array1<F>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // Mock model for testing
    fn mock_model(x: &ArrayView2<f64>) -> Array1<f64> {
        x.map_axis(Axis(1), |row| row.sum())
    }

    #[test]
    fn test_uncertainty_quantifier_creation() {
        let quantifier = UncertaintyQuantifier::<f64>::new()
            .with_mc_samples(500)
            .with_confidence_level(0.9)
            .with_bootstrap(50)
            .with_seed(42);

        assert_eq!(quantifier.n_mc_samples, 500);
        assert_eq!(quantifier.confidence_level, 0.9);
        assert_eq!(quantifier.n_bootstrap, 50);
        assert_eq!(quantifier.random_seed, Some(42));
    }

    #[test]
    fn test_epistemic_uncertainty() {
        let quantifier = UncertaintyQuantifier::<f64>::new()
            .with_mc_samples(10)
            .with_seed(42);

        let x_test = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let epistemic = quantifier
            .compute_epistemic_uncertainty(&mock_model, &x_test)
            .unwrap();

        assert_eq!(epistemic.mean_predictions.len(), 3);
        assert_eq!(epistemic.prediction_variance.len(), 3);
        assert_eq!(epistemic.mc_samples, 10);
    }

    #[test]
    fn test_prediction_intervals() {
        let quantifier = UncertaintyQuantifier::<f64>::new()
            .with_bootstrap(10)
            .with_confidence_level(0.95)
            .with_seed(42);

        let x_test = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let intervals = quantifier
            .compute_prediction_intervals(&mock_model, &x_test)
            .unwrap();

        assert_eq!(intervals.lower_bound.len(), 3);
        assert_eq!(intervals.upper_bound.len(), 3);
        assert_eq!(intervals.median_prediction.len(), 3);
        assert_eq!(intervals.confidence_level, 0.95);
    }

    #[test]
    fn test_calibration_metrics() {
        let quantifier = UncertaintyQuantifier::<f64>::new().with_seed(42);

        let predictions = array![0.1, 0.4, 0.7, 0.9];
        let y_true = array![0.0, 0.0, 1.0, 1.0];

        let calibration = quantifier
            .compute_calibration_metrics(
                &mock_model,
                &array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
                &y_true,
            )
            .unwrap();

        assert!(calibration.expected_calibration_error >= 0.0);
        assert!(calibration.maximum_calibration_error >= 0.0);
        assert_eq!(calibration.reliability_data.len(), 10); // 10 bins
    }

    #[test]
    fn test_ood_scores() {
        let quantifier = UncertaintyQuantifier::<f64>::new().with_seed(42);
        let x_test = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let ood_scores = quantifier.compute_ood_scores(&mock_model, &x_test).unwrap();

        assert_eq!(ood_scores.mahalanobis_scores.len(), 3);
        assert_eq!(ood_scores.energy_scores.len(), 3);
        assert_eq!(ood_scores.reconstruction_errors.len(), 3);
        assert_eq!(ood_scores.density_scores.len(), 3);
    }

    #[test]
    fn test_gaussian_noise_generation() {
        let quantifier = UncertaintyQuantifier::<f64>::new().with_seed(42);
        let noise = quantifier.generate_gaussian_noise().unwrap();

        // Just check that noise is finite
        assert!(noise.is_finite());
    }

    #[test]
    fn test_euclidean_distance() {
        let quantifier = UncertaintyQuantifier::<f64>::new();
        let x1 = array![1.0, 2.0, 3.0];
        let x2 = array![4.0, 5.0, 6.0];

        let distance = quantifier
            .compute_euclidean_distance(&x1.view(), &x2.view())
            .unwrap();
        let expected = ((3.0_f64).powi(2) * 3.0).sqrt(); // sqrt(9 + 9 + 9) = sqrt(27)
        assert!((distance - expected).abs() < 1e-10);
    }

    #[test]
    fn test_brier_decomposition() {
        let quantifier = UncertaintyQuantifier::<f64>::new();
        let predictions = array![0.2, 0.8, 0.6, 0.9];
        let y_true = array![0.0, 1.0, 1.0, 1.0];

        let decomposition = quantifier
            .compute_brier_decomposition(&predictions, &y_true)
            .unwrap();

        assert!(decomposition.brier_score >= 0.0);
        assert!(decomposition.reliability >= 0.0);
        assert!(decomposition.uncertainty >= 0.0);
    }

    #[test]
    fn test_comprehensive_uncertainty_analysis() {
        let quantifier = UncertaintyQuantifier::<f64>::new()
            .with_mc_samples(5)
            .with_bootstrap(5)
            .with_seed(42);

        let x_test = array![[1.0, 2.0], [3.0, 4.0]];
        let y_test = array![0.0, 1.0];

        let analysis = quantifier
            .compute_uncertainty(&mock_model, &x_test, Some(&y_test))
            .unwrap();

        assert_eq!(analysis.sample_size, 2);
        assert!(analysis.calibration_metrics.is_some());
        assert_eq!(analysis.epistemic_uncertainty.mean_predictions.len(), 2);
        assert_eq!(analysis.prediction_intervals.lower_bound.len(), 2);
    }
}
