//! Uncertainty quantification methods for model predictions

use crate::error::{MetricsError, Result};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use num_traits::Float;
use std::collections::HashMap;
use std::f64::consts::PI;

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

/// Random number generator types
#[derive(Debug, Clone)]
pub enum RandomNumberGenerator {
    /// Linear Congruential Generator (fast, basic quality)
    Lcg,
    /// Xorshift (good balance of speed and quality)
    Xorshift,
    /// Permuted Congruential Generator (high quality)
    Pcg,
    /// ChaCha (cryptographically secure)
    ChaCha,
}

/// Trait for random number generators
pub trait RandomNumberGeneratorTrait {
    fn uniform_01<F: Float + num_traits::FromPrimitive>(&mut self) -> F;
    fn normal<F: Float + num_traits::FromPrimitive>(&mut self) -> F;
    fn seed(&mut self, seed: u64);
}

/// Linear Congruential Generator implementation
pub struct LcgRng {
    state: u64,
}

impl LcgRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }
}

impl RandomNumberGeneratorTrait for LcgRng {
    fn uniform_01<F: Float + num_traits::FromPrimitive>(&mut self) -> F {
        self.state = self.state.wrapping_mul(1103515245).wrapping_add(12345);
        F::from((self.state >> 16) as f64 / (1u64 << 32) as f64).unwrap()
    }
    
    fn normal<F: Float + num_traits::FromPrimitive>(&mut self) -> F {
        // Box-Muller transform
        let u1 = self.uniform_01::<F>();
        let u2 = self.uniform_01::<F>();
        
        (-F::from(2.0).unwrap() * u1.ln()).sqrt() 
            * (F::from(2.0 * PI).unwrap() * u2).cos()
    }
    
    fn seed(&mut self, seed: u64) {
        self.state = seed;
    }
}

/// Xorshift random number generator
pub struct XorshiftRng {
    state: u64,
}

impl XorshiftRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed.max(1) } // Ensure non-zero state
    }
}

impl RandomNumberGeneratorTrait for XorshiftRng {
    fn uniform_01<F: Float + num_traits::FromPrimitive>(&mut self) -> F {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        F::from(self.state as f64 / u64::MAX as f64).unwrap()
    }
    
    fn normal<F: Float + num_traits::FromPrimitive>(&mut self) -> F {
        let u1 = self.uniform_01::<F>();
        let u2 = self.uniform_01::<F>();
        
        (-F::from(2.0).unwrap() * u1.ln()).sqrt() 
            * (F::from(2.0 * PI).unwrap() * u2).cos()
    }
    
    fn seed(&mut self, seed: u64) {
        self.state = seed.max(1);
    }
}

/// PCG random number generator
pub struct PcgRng {
    state: u64,
    inc: u64,
}

impl PcgRng {
    pub fn new(seed: u64) -> Self {
        Self { 
            state: seed,
            inc: 721347520444481703u64,
        }
    }
}

impl RandomNumberGeneratorTrait for PcgRng {
    fn uniform_01<F: Float + num_traits::FromPrimitive>(&mut self) -> F {
        let oldstate = self.state;
        self.state = oldstate.wrapping_mul(6364136223846793005u64).wrapping_add(self.inc);
        let xorshifted = ((oldstate >> 18) ^ oldstate) >> 27;
        let rot = oldstate >> 59;
        let result = (xorshifted >> rot) | (xorshifted << ((-rot as i32) & 31));
        F::from(result as f64 / u32::MAX as f64).unwrap()
    }
    
    fn normal<F: Float + num_traits::FromPrimitive>(&mut self) -> F {
        let u1 = self.uniform_01::<F>();
        let u2 = self.uniform_01::<F>();
        
        (-F::from(2.0).unwrap() * u1.ln()).sqrt() 
            * (F::from(2.0 * PI).unwrap() * u2).cos()
    }
    
    fn seed(&mut self, seed: u64) {
        self.state = seed;
    }
}

/// ChaCha random number generator (simplified)
pub struct ChaChaRng {
    state: [u32; 16],
    counter: u64,
}

impl ChaChaRng {
    pub fn new(seed: u64) -> Self {
        let mut state = [0u32; 16];
        state[0] = seed as u32;
        state[1] = (seed >> 32) as u32;
        Self { state, counter: 0 }
    }
}

impl RandomNumberGeneratorTrait for ChaChaRng {
    fn uniform_01<F: Float + num_traits::FromPrimitive>(&mut self) -> F {
        // Simplified ChaCha implementation
        self.counter = self.counter.wrapping_add(1);
        let mut x = self.state[0].wrapping_add(self.counter as u32);
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state[0] = x;
        
        F::from(x as f64 / u32::MAX as f64).unwrap()
    }
    
    fn normal<F: Float + num_traits::FromPrimitive>(&mut self) -> F {
        let u1 = self.uniform_01::<F>();
        let u2 = self.uniform_01::<F>();
        
        (-F::from(2.0).unwrap() * u1.ln()).sqrt() 
            * (F::from(2.0 * PI).unwrap() * u2).cos()
    }
    
    fn seed(&mut self, seed: u64) {
        self.state[0] = seed as u32;
        self.state[1] = (seed >> 32) as u32;
        self.counter = 0;
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

/// Advanced uncertainty analysis results
#[derive(Debug, Clone)]
pub struct AdvancedUncertaintyAnalysis<F: Float> {
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
    /// Conformal prediction results
    pub conformal_prediction: Option<ConformalPrediction<F>>,
    /// Bayesian uncertainty estimation
    pub bayesian_uncertainty: Option<BayesianUncertainty<F>>,
    /// Temperature scaling results
    pub temperature_scaling: Option<TemperatureScaling<F>>,
    /// Deep ensemble uncertainty
    pub deep_ensemble_uncertainty: DeepEnsembleUncertainty<F>,
    /// Sample size
    pub sample_size: usize,
}

/// Conformal prediction results
#[derive(Debug, Clone)]
pub struct ConformalPrediction<F: Float> {
    /// Prediction sets for each test sample
    pub prediction_sets: Vec<PredictionSet<F>>,
    /// Coverage probability
    pub coverage_probability: F,
    /// Average set size
    pub average_set_size: F,
    /// Conditional coverage by groups
    pub conditional_coverage: HashMap<String, F>,
}

/// Prediction set for conformal prediction
#[derive(Debug, Clone)]
pub struct PredictionSet<F: Float> {
    /// Lower bound
    pub lower: F,
    /// Upper bound
    pub upper: F,
    /// Set size
    pub size: F,
    /// Contains true value (if known)
    pub contains_truth: Option<bool>,
}

/// Bayesian uncertainty estimation
#[derive(Debug, Clone)]
pub struct BayesianUncertainty<F: Float> {
    /// Posterior mean predictions
    pub posterior_mean: Array1<F>,
    /// Posterior variance
    pub posterior_variance: Array1<F>,
    /// Credible intervals
    pub credible_intervals: Array2<F>, // [n_samples, 2] for lower/upper bounds
    /// Model evidence (marginal likelihood)
    pub model_evidence: F,
    /// MCMC effective sample size
    pub effective_sample_size: F,
    /// R-hat convergence diagnostic
    pub r_hat: F,
}

/// Temperature scaling results
#[derive(Debug, Clone)]
pub struct TemperatureScaling<F: Float> {
    /// Optimal temperature parameter
    pub temperature: F,
    /// Calibrated predictions
    pub calibrated_predictions: Array1<F>,
    /// Before calibration error
    pub before_calibration_error: F,
    /// After calibration error
    pub after_calibration_error: F,
    /// Calibration improvement
    pub improvement: F,
}

/// Deep ensemble uncertainty
#[derive(Debug, Clone)]
pub struct DeepEnsembleUncertainty<F: Float> {
    /// Individual model predictions
    pub individual_predictions: Array2<F>, // [n_models, n_samples]
    /// Ensemble mean
    pub ensemble_mean: Array1<F>,
    /// Ensemble variance
    pub ensemble_variance: Array1<F>,
    /// Model disagreement
    pub model_disagreement: Array1<F>,
    /// Diversity scores
    pub diversity_scores: Array1<F>,
}

// Additional convenience functions for uncertainty quantification

/// Compute entropy of probability distribution
pub fn compute_entropy<F: Float + num_traits::FromPrimitive>(probabilities: &Array1<F>) -> F {
    let mut entropy = F::zero();
    let eps = F::from(1e-15).unwrap();
    
    for &p in probabilities.iter() {
        if p > eps {
            entropy = entropy - p * p.ln();
        }
    }
    
    entropy
}

/// Compute KL divergence between two distributions
pub fn compute_kl_divergence<F: Float + num_traits::FromPrimitive>(
    p: &Array1<F>, 
    q: &Array1<F>
) -> Result<F> {
    if p.len() != q.len() {
        return Err(MetricsError::InvalidInput(
            "Distributions must have same length".to_string()
        ));
    }
    
    let mut kl_div = F::zero();
    let eps = F::from(1e-15).unwrap();
    
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        if pi > eps && qi > eps {
            kl_div = kl_div + pi * (pi / qi).ln();
        }
    }
    
    Ok(kl_div)
}

/// Compute Jensen-Shannon divergence
pub fn compute_js_divergence<F: Float + num_traits::FromPrimitive>(
    p: &Array1<F>, 
    q: &Array1<F>
) -> Result<F> {
    let m = (p + q) / F::from(2.0).unwrap();
    let kl_pm = compute_kl_divergence(p, &m)?;
    let kl_qm = compute_kl_divergence(q, &m)?;
    
    Ok((kl_pm + kl_qm) / F::from(2.0).unwrap())
}

/// Compute Wasserstein distance (simplified 1D version)
pub fn compute_wasserstein_distance<F: Float + num_traits::FromPrimitive>(
    samples1: &Array1<F>,
    samples2: &Array1<F>,
) -> F {
    let mut s1 = samples1.to_vec();
    let mut s2 = samples2.to_vec();
    
    s1.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    s2.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    
    let min_len = s1.len().min(s2.len());
    let mut distance = F::zero();
    
    for i in 0..min_len {
        distance = distance + (s1[i] - s2[i]).abs();
    }
    
    distance / F::from(min_len).unwrap()
}

/// Compute maximum mean discrepancy (simplified)
pub fn compute_mmd<F: Float + num_traits::FromPrimitive>(
    samples1: &Array2<F>,
    samples2: &Array2<F>,
    gamma: F,
) -> Result<F> {
    if samples1.ncols() != samples2.ncols() {
        return Err(MetricsError::InvalidInput(
            "Samples must have same dimensionality".to_string()
        ));
    }
    
    let n1 = samples1.nrows();
    let n2 = samples2.nrows();
    
    // Compute kernel means
    let mut k11 = F::zero();
    let mut k22 = F::zero();
    let mut k12 = F::zero();
    
    // K(X, X)
    for i in 0..n1 {
        for j in 0..n1 {
            let dist_sq = samples1.row(i).iter()
                .zip(samples1.row(j).iter())
                .map(|(&xi, &xj)| (xi - xj) * (xi - xj))
                .sum::<F>();
            k11 = k11 + (-gamma * dist_sq).exp();
        }
    }
    k11 = k11 / F::from(n1 * n1).unwrap();
    
    // K(Y, Y)
    for i in 0..n2 {
        for j in 0..n2 {
            let dist_sq = samples2.row(i).iter()
                .zip(samples2.row(j).iter())
                .map(|(&yi, &yj)| (yi - yj) * (yi - yj))
                .sum::<F>();
            k22 = k22 + (-gamma * dist_sq).exp();
        }
    }
    k22 = k22 / F::from(n2 * n2).unwrap();
    
    // K(X, Y)
    for i in 0..n1 {
        for j in 0..n2 {
            let dist_sq = samples1.row(i).iter()
                .zip(samples2.row(j).iter())
                .map(|(&xi, &yj)| (xi - yj) * (xi - yj))
                .sum::<F>();
            k12 = k12 + (-gamma * dist_sq).exp();
        }
    }
    k12 = k12 / F::from(n1 * n2).unwrap();
    
    Ok(k11 + k22 - F::from(2.0).unwrap() * k12)
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
    
    #[test]
    fn test_random_number_generators() {
        let mut lcg = LcgRng::new(42);
        let mut xorshift = XorshiftRng::new(42);
        let mut pcg = PcgRng::new(42);
        let mut chacha = ChaChaRng::new(42);
        
        // Test uniform generation
        let lcg_val = lcg.uniform_01::<f64>();
        let xorshift_val = xorshift.uniform_01::<f64>();
        let pcg_val = pcg.uniform_01::<f64>();
        let chacha_val = chacha.uniform_01::<f64>();
        
        assert!(lcg_val >= 0.0 && lcg_val <= 1.0);
        assert!(xorshift_val >= 0.0 && xorshift_val <= 1.0);
        assert!(pcg_val >= 0.0 && pcg_val <= 1.0);
        assert!(chacha_val >= 0.0 && chacha_val <= 1.0);
        
        // Test normal generation
        let lcg_normal = lcg.normal::<f64>();
        let xorshift_normal = xorshift.normal::<f64>();
        let pcg_normal = pcg.normal::<f64>();
        let chacha_normal = chacha.normal::<f64>();
        
        assert!(lcg_normal.is_finite());
        assert!(xorshift_normal.is_finite());
        assert!(pcg_normal.is_finite());
        assert!(chacha_normal.is_finite());
    }
    
    #[test]
    fn test_advanced_uncertainty_quantifier() {
        let quantifier = UncertaintyQuantifier::<f64>::new()
            .with_rng_type(RandomNumberGenerator::Pcg)
            .with_conformal_calibration(50)
            .with_bayesian(true)
            .with_mcmc(100, 20)
            .with_temperature_scaling(true)
            .with_simd(true);
        
        assert_eq!(quantifier.n_conformal_calibration, 50);
        assert!(quantifier.enable_bayesian);
        assert_eq!(quantifier.n_mcmc_samples, 100);
        assert_eq!(quantifier.mcmc_burn_in, 20);
        assert!(quantifier.enable_temperature_scaling);
        assert!(quantifier.enable_simd);
    }
    
    #[test]
    fn test_entropy_computation() {
        let probs = array![0.1, 0.2, 0.3, 0.4];
        let entropy = compute_entropy(&probs);
        assert!(entropy > 0.0);
        
        // Uniform distribution should have maximum entropy
        let uniform_probs = array![0.25, 0.25, 0.25, 0.25];
        let uniform_entropy = compute_entropy(&uniform_probs);
        assert!(uniform_entropy > entropy);
    }
    
    #[test]
    fn test_kl_divergence() {
        let p = array![0.5, 0.3, 0.2];
        let q = array![0.4, 0.4, 0.2];
        
        let kl_div = compute_kl_divergence(&p, &q).unwrap();
        assert!(kl_div >= 0.0);
        
        // KL divergence should be 0 for identical distributions
        let kl_self = compute_kl_divergence(&p, &p).unwrap();
        assert!(kl_self.abs() < 1e-10);
    }
    
    #[test]
    fn test_js_divergence() {
        let p = array![0.5, 0.3, 0.2];
        let q = array![0.4, 0.4, 0.2];
        
        let js_div = compute_js_divergence(&p, &q).unwrap();
        assert!(js_div >= 0.0);
        
        // JS divergence should be 0 for identical distributions
        let js_self = compute_js_divergence(&p, &p).unwrap();
        assert!(js_self.abs() < 1e-10);
    }
    
    #[test]
    fn test_wasserstein_distance() {
        let samples1 = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let samples2 = array![1.1, 2.1, 3.1, 4.1, 5.1];
        
        let wasserstein = compute_wasserstein_distance(&samples1, &samples2);
        assert!(wasserstein >= 0.0);
        assert!((wasserstein - 0.1).abs() < 1e-10);
    }
}
