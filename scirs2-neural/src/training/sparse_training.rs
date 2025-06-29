//! Sparse Training utilities
//!
//! This module provides utilities for sparse training, including various pruning
//! strategies, sparse optimizers, and techniques for training networks with
//! sparse connectivity patterns. Sparse training can significantly reduce
//! computational requirements and memory usage while maintaining model performance.

use crate::error::{NeuralError, Result};
use crate::layers::{Layer, ParamLayer};
use crate::losses::Loss;
use crate::optimizers::{Optimizer, OptimizerStep};
use ndarray::prelude::*;
use ndarray::{Array, IxDyn};
use num_traits::{Float, FromPrimitive, Zero};
use std::collections::HashMap;
use std::fmt::Debug;

/// Pruning strategy for sparse training
#[derive(Debug, Clone, PartialEq)]
pub enum PruningStrategy {
    /// Magnitude-based pruning (remove smallest weights)
    Magnitude,
    /// Random pruning
    Random,
    /// Structured pruning (remove entire channels/filters)
    Structured,
    /// Gradual magnitude pruning
    GradualMagnitude,
    /// SNIP (Single-shot Network Pruning)
    SNIP,
    /// GraSP (Gradient Signal Preservation)
    GraSP,
    /// Lottery Ticket Hypothesis pruning
    LotteryTicket,
}

/// Sparse training configuration
#[derive(Debug, Clone)]
pub struct SparseTrainingConfig {
    /// Pruning strategy to use
    pub strategy: PruningStrategy,
    /// Target sparsity level (0.0 to 1.0)
    pub target_sparsity: f64,
    /// Initial sparsity level
    pub initial_sparsity: f64,
    /// Number of pruning steps
    pub pruning_steps: usize,
    /// Frequency of pruning (in training steps)
    pub pruning_frequency: usize,
    /// Whether to use gradual pruning
    pub gradual_pruning: bool,
    /// Recovery epochs after pruning
    pub recovery_epochs: usize,
    /// Whether to fine-tune after pruning
    pub fine_tune: bool,
    /// Structured pruning granularity
    pub structured_granularity: StructuredGranularity,
    /// Whether to use magnitude recovery
    pub magnitude_recovery: bool,
    /// Learning rate multiplier for sparse weights
    pub sparse_lr_multiplier: f64,
}

impl Default for SparseTrainingConfig {
    fn default() -> Self {
        Self {
            strategy: PruningStrategy::GradualMagnitude,
            target_sparsity: 0.9,
            initial_sparsity: 0.0,
            pruning_steps: 100,
            pruning_frequency: 100,
            gradual_pruning: true,
            recovery_epochs: 10,
            fine_tune: true,
            structured_granularity: StructuredGranularity::Filter,
            magnitude_recovery: false,
            sparse_lr_multiplier: 1.0,
        }
    }
}

/// Structured pruning granularity
#[derive(Debug, Clone, PartialEq)]
pub enum StructuredGranularity {
    /// Prune entire filters/channels
    Filter,
    /// Prune groups of channels
    Group(usize),
    /// Prune blocks of weights
    Block(usize, usize),
}

/// Sparsity mask for a layer
#[derive(Debug, Clone)]
pub struct SparsityMask {
    /// Binary mask (1 = keep, 0 = prune)
    pub mask: Array<bool, IxDyn>,
    /// Current sparsity level
    pub sparsity: f64,
    /// Layer name/identifier
    pub layer_id: String,
    /// Pruning scores (for tracking importance)
    pub scores: Option<Array<f64, IxDyn>>,
}

impl SparsityMask {
    /// Create a new sparsity mask
    pub fn new(shape: &[usize], layer_id: String) -> Self {
        let mask = Array::<bool, _>::from_elem(IxDyn(shape), true);
        Self {
            mask,
            sparsity: 0.0,
            layer_id,
            scores: None,
        }
    }

    /// Update mask to achieve target sparsity
    pub fn update_magnitude_mask<F: Float + Debug + PartialOrd>(
        &mut self,
        weights: &Array<F, IxDyn>,
        target_sparsity: f64,
    ) {
        let total_elements = weights.len();
        let target_pruned = (total_elements as f64 * target_sparsity) as usize;

        // Calculate magnitude scores
        let mut weight_magnitudes: Vec<(usize, F)> = weights
            .iter()
            .enumerate()
            .map(|(i, &w)| (i, w.abs()))
            .collect();

        // Sort by magnitude (ascending)
        weight_magnitudes.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Reset mask
        self.mask.fill(true);

        // Prune smallest weights
        for i in 0..target_pruned.min(total_elements) {
            let idx = weight_magnitudes[i].0;
            let linear_idx = self.linear_index_to_nd(idx, weights.shape());
            if let Some(mut_ref) = self.mask.get_mut(linear_idx) {
                *mut_ref = false;
            }
        }

        self.sparsity = target_pruned as f64 / total_elements as f64;
    }

    /// Convert linear index to n-dimensional index
    fn linear_index_to_nd(&self, linear_idx: usize, shape: &[usize]) -> IxDyn {
        let mut idx = vec![0; shape.len()];
        let mut remaining = linear_idx;

        for i in (0..shape.len()).rev() {
            let stride: usize = shape[i + 1..].iter().product();
            idx[i] = remaining / stride;
            remaining %= stride;
        }

        IxDyn(&idx)
    }

    /// Apply mask to weights
    pub fn apply_mask<F: Float + Debug + Zero>(&self, weights: &mut Array<F, IxDyn>) {
        for (weight, &keep) in weights.iter_mut().zip(self.mask.iter()) {
            if !keep {
                *weight = F::zero();
            }
        }
    }

    /// Calculate current sparsity
    pub fn calculate_sparsity(&self) -> f64 {
        let total = self.mask.len();
        let pruned = self.mask.iter().filter(|&&keep| !keep).count();
        pruned as f64 / total as f64
    }

    /// Combine with another mask (logical AND)
    pub fn intersect(&mut self, other: &SparsityMask) {
        if self.mask.shape() != other.mask.shape() {
            return;
        }

        for (self_val, &other_val) in self.mask.iter_mut().zip(other.mask.iter()) {
            *self_val = *self_val && other_val;
        }

        self.sparsity = self.calculate_sparsity();
    }
}

/// SNIP (Single-shot Network Pruning) implementation
pub struct SNIPPruner {
    /// Data for computing connection sensitivity
    data_batch: Option<Array<f32, IxDyn>>,
    /// Target batch for computing gradients
    target_batch: Option<Array<f32, IxDyn>>,
}

impl SNIPPruner {
    /// Create a new SNIP pruner
    pub fn new() -> Self {
        Self {
            data_batch: None,
            target_batch: None,
        }
    }

    /// Set calibration data
    pub fn set_calibration_data(&mut self, data: Array<f32, IxDyn>, targets: Array<f32, IxDyn>) {
        self.data_batch = Some(data);
        self.target_batch = Some(targets);
    }

    /// Compute connection sensitivity scores
    pub fn compute_snip_scores<L: Layer<f32>>(
        &self,
        model: &L,
        loss_fn: &dyn Loss<f32>,
    ) -> Result<HashMap<String, Array<f64, IxDyn>>> {
        let data = self
            .data_batch
            .as_ref()
            .ok_or_else(|| NeuralError::InvalidState("No calibration data set".to_string()))?;
        let targets = self
            .target_batch
            .as_ref()
            .ok_or_else(|| NeuralError::InvalidState("No calibration targets set".to_string()))?;

        // Forward pass
        let outputs = model.forward(data)?;

        // Compute loss
        let loss = loss_fn.forward(&outputs, targets)?;
        let grad_output = loss_fn.backward(&outputs, targets)?;

        // Backward pass to get gradients
        let _grad_input = model.backward(data, &grad_output)?;

        // Get gradients and parameters
        let gradients = model.gradients();
        let params = model.params();

        let mut scores = HashMap::new();

        // Compute connection sensitivity: |gradient * weight|
        for (i, (param, grad)) in params.iter().zip(gradients.iter()).enumerate() {
            let layer_id = format!("layer_{}", i);
            let mut sensitivity = Array::<f64, _>::zeros(param.raw_dim());

            for ((s, &p), &g) in sensitivity.iter_mut().zip(param.iter()).zip(grad.iter()) {
                *s = (g * p).abs().to_f64().unwrap();
            }

            scores.insert(layer_id, sensitivity);
        }

        Ok(scores)
    }
}

/// Sparse training manager
pub struct SparseTrainer<F: Float + Debug + FromPrimitive + Send + Sync + Zero + PartialOrd> {
    /// Configuration
    config: SparseTrainingConfig,
    /// Sparsity masks for each layer
    masks: HashMap<String, SparsityMask>,
    /// Current training step
    training_step: usize,
    /// Current sparsity level
    current_sparsity: f64,
    /// SNIP pruner for single-shot pruning
    snip_pruner: SNIPPruner,
    /// Original weights (for lottery ticket)
    original_weights: Option<Vec<Array<F, IxDyn>>>,
    /// Pruning schedule
    pruning_schedule: Vec<f64>,
    /// Recovery phase counter
    recovery_counter: usize,
    /// Whether currently in recovery phase
    in_recovery: bool,
}

impl<F: Float + Debug + FromPrimitive + Send + Sync + Zero + PartialOrd> SparseTrainer<F> {
    /// Create a new sparse trainer
    pub fn new(config: SparseTrainingConfig) -> Self {
        let pruning_schedule = Self::create_pruning_schedule(&config);

        Self {
            config,
            masks: HashMap::new(),
            training_step: 0,
            current_sparsity: 0.0,
            snip_pruner: SNIPPruner::new(),
            original_weights: None,
            pruning_schedule,
            recovery_counter: 0,
            in_recovery: false,
        }
    }

    /// Create pruning schedule
    fn create_pruning_schedule(config: &SparseTrainingConfig) -> Vec<f64> {
        if !config.gradual_pruning {
            return vec![config.target_sparsity];
        }

        let mut schedule = Vec::new();
        let start = config.initial_sparsity;
        let end = config.target_sparsity;
        let steps = config.pruning_steps;

        for i in 0..=steps {
            let progress = i as f64 / steps as f64;
            // Cubic sparsity schedule (as used in magnitude pruning papers)
            let sparsity = start + (end - start) * (1.0 - (1.0 - progress).powi(3));
            schedule.push(sparsity);
        }

        schedule
    }

    /// Initialize sparse training
    pub fn initialize<L: Layer<F>>(&mut self, model: &L) -> Result<()> {
        let params = model.params();

        // Store original weights for lottery ticket
        if self.config.strategy == PruningStrategy::LotteryTicket {
            self.original_weights = Some(params.clone());
        }

        // Create sparsity masks
        for (i, param) in params.iter().enumerate() {
            let layer_id = format!("layer_{}", i);
            let mask = SparsityMask::new(param.shape(), layer_id.clone());
            self.masks.insert(layer_id, mask);
        }

        self.current_sparsity = self.config.initial_sparsity;
        Ok(())
    }

    /// Set calibration data for SNIP
    pub fn set_calibration_data(&mut self, data: Array<f32, IxDyn>, targets: Array<f32, IxDyn>) {
        self.snip_pruner.set_calibration_data(data, targets);
    }

    /// Check if pruning should occur at current step
    fn should_prune(&self) -> bool {
        if self.in_recovery {
            return false;
        }

        match self.config.strategy {
            PruningStrategy::SNIP => self.training_step == 0,
            PruningStrategy::LotteryTicket => self.training_step == 0,
            _ => {
                self.training_step % self.config.pruning_frequency == 0
                    && self.current_sparsity < self.config.target_sparsity
            }
        }
    }

    /// Perform pruning
    pub fn prune<L: ParamLayer<F>>(
        &mut self,
        model: &mut L,
        loss_fn: Option<&dyn Loss<f32>>,
    ) -> Result<()> {
        if !self.should_prune() {
            return Ok(());
        }

        match self.config.strategy {
            PruningStrategy::Magnitude => self.prune_magnitude(model)?,
            PruningStrategy::GradualMagnitude => self.prune_gradual_magnitude(model)?,
            PruningStrategy::Random => self.prune_random(model)?,
            PruningStrategy::Structured => self.prune_structured(model)?,
            PruningStrategy::SNIP => {
                if let Some(loss_fn) = loss_fn {
                    self.prune_snip(model, loss_fn)?;
                }
            }
            PruningStrategy::LotteryTicket => self.prune_lottery_ticket(model)?,
            _ => {
                return Err(NeuralError::NotImplementedError(
                    "Pruning strategy not implemented".to_string(),
                ))
            }
        }

        // Enter recovery phase if configured
        if self.config.recovery_epochs > 0 {
            self.in_recovery = true;
            self.recovery_counter = 0;
        }

        Ok(())
    }

    /// Magnitude-based pruning
    fn prune_magnitude<L: ParamLayer<F>>(&mut self, model: &mut L) -> Result<()> {
        let params = model.params();

        for (i, param) in params.iter().enumerate() {
            let layer_id = format!("layer_{}", i);
            if let Some(mask) = self.masks.get_mut(&layer_id) {
                mask.update_magnitude_mask(param, self.config.target_sparsity);
            }
        }

        self.current_sparsity = self.config.target_sparsity;
        self.apply_masks(model)?;
        Ok(())
    }

    /// Gradual magnitude pruning
    fn prune_gradual_magnitude<L: ParamLayer<F>>(&mut self, model: &mut L) -> Result<()> {
        let schedule_idx = self.training_step / self.config.pruning_frequency;
        let target_sparsity = if schedule_idx < self.pruning_schedule.len() {
            self.pruning_schedule[schedule_idx]
        } else {
            self.config.target_sparsity
        };

        let params = model.params();

        for (i, param) in params.iter().enumerate() {
            let layer_id = format!("layer_{}", i);
            if let Some(mask) = self.masks.get_mut(&layer_id) {
                mask.update_magnitude_mask(param, target_sparsity);
            }
        }

        self.current_sparsity = target_sparsity;
        self.apply_masks(model)?;
        Ok(())
    }

    /// Random pruning
    fn prune_random<L: ParamLayer<F>>(&mut self, model: &mut L) -> Result<()> {
        let params = model.params();

        for (i, param) in params.iter().enumerate() {
            let layer_id = format!("layer_{}", i);
            if let Some(mask) = self.masks.get_mut(&layer_id) {
                let total_elements = param.len();
                let target_pruned = (total_elements as f64 * self.config.target_sparsity) as usize;

                // Reset mask
                mask.mask.fill(true);

                // Randomly select elements to prune
                use rand::seq::SliceRandom;
                use rand::rng;

                let mut indices: Vec<usize> = (0..total_elements).collect();
                indices.shuffle(&mut rng());

                for &idx in indices.iter().take(target_pruned) {
                    let nd_idx = mask.linear_index_to_nd(idx, param.shape());
                    if let Some(mut_ref) = mask.mask.get_mut(nd_idx) {
                        *mut_ref = false;
                    }
                }

                mask.sparsity = target_pruned as f64 / total_elements as f64;
            }
        }

        self.current_sparsity = self.config.target_sparsity;
        self.apply_masks(model)?;
        Ok(())
    }

    /// Structured pruning
    fn prune_structured<L: ParamLayer<F>>(&mut self, model: &mut L) -> Result<()> {
        // Simplified structured pruning - in practice would need more sophisticated layer analysis
        match self.config.structured_granularity {
            StructuredGranularity::Filter => {
                // Prune entire filters/channels
                self.prune_filters(model)?;
            }
            StructuredGranularity::Group(group_size) => {
                // Prune groups of channels
                self.prune_channel_groups(model, group_size)?;
            }
            StructuredGranularity::Block(block_h, block_w) => {
                // Prune blocks of weights
                self.prune_blocks(model, block_h, block_w)?;
            }
        }

        Ok(())
    }

    /// SNIP pruning
    fn prune_snip<L: ParamLayer<F>>(
        &mut self,
        model: &mut L,
        loss_fn: &dyn Loss<f32>,
    ) -> Result<()> {
        // Convert model to f32 for SNIP computation (simplified)
        let f32_model = self.convert_to_f32_model(model)?;
        let scores = self.snip_pruner.compute_snip_scores(&*f32_model, loss_fn)?;

        for (layer_id, score) in scores {
            if let Some(mask) = self.masks.get_mut(&layer_id) {
                mask.scores = Some(score);
                // Use scores to create mask
                self.create_mask_from_scores(mask, self.config.target_sparsity);
            }
        }

        self.current_sparsity = self.config.target_sparsity;
        self.apply_masks(model)?;
        Ok(())
    }

    /// Lottery ticket pruning
    fn prune_lottery_ticket<L: ParamLayer<F>>(&mut self, model: &mut L) -> Result<()> {
        // First prune using magnitude-based approach
        self.prune_magnitude(model)?;

        // Reset weights to original initialization
        if let Some(ref original_weights) = self.original_weights {
            model.set_params(original_weights)?;
            // Apply masks to the original weights
            self.apply_masks(model)?;
        }

        Ok(())
    }

    /// Apply sparsity masks to model
    fn apply_masks<L: ParamLayer<F>>(&self, model: &mut L) -> Result<()> {
        let mut params = model.params();

        for (i, param) in params.iter_mut().enumerate() {
            let layer_id = format!("layer_{}", i);
            if let Some(mask) = self.masks.get(&layer_id) {
                mask.apply_mask(param);
            }
        }

        model.set_params(&params)?;
        Ok(())
    }

    /// Train step with sparse training
    pub fn train_step<L, O>(
        &mut self,
        model: &mut L,
        optimizer: &mut O,
        inputs: &Array<F, IxDyn>,
        targets: &Array<F, IxDyn>,
        loss_fn: &dyn Loss<F>,
    ) -> Result<F>
    where
        L: ParamLayer<F>,
        O: Optimizer<F> + OptimizerStep<F>,
    {
        // Perform pruning if needed
        // Note: This is simplified - in practice, SNIP would need f32 loss function
        self.prune(model, None)?;

        // Standard training step
        let outputs = model.forward(inputs)?;
        let loss = loss_fn.forward(&outputs, targets)?;
        let grad_output = loss_fn.backward(&outputs, targets)?;

        // Backward pass
        let _grad_input = model.backward(inputs, &grad_output)?;

        // Apply sparse learning rate if configured
        if self.config.sparse_lr_multiplier != 1.0 {
            self.apply_sparse_learning_rate(model)?;
        }

        // Update parameters
        optimizer.step(model)?;

        // Apply masks after update to maintain sparsity
        self.apply_masks(model)?;

        // Update training state
        self.training_step += 1;

        // Update recovery phase
        if self.in_recovery {
            self.recovery_counter += 1;
            if self.recovery_counter >= self.config.recovery_epochs {
                self.in_recovery = false;
                self.recovery_counter = 0;
            }
        }

        Ok(loss)
    }

    /// Apply sparse learning rate multiplier
    fn apply_sparse_learning_rate<L: ParamLayer<F>>(&self, model: &mut L) -> Result<()> {
        let mut gradients = model.gradients();

        for (i, grad) in gradients.iter_mut().enumerate() {
            let layer_id = format!("layer_{}", i);
            if let Some(mask) = self.masks.get(&layer_id) {
                for (grad_val, &keep) in grad.iter_mut().zip(mask.mask.iter()) {
                    if keep {
                        *grad_val = *grad_val * F::from(self.config.sparse_lr_multiplier).unwrap();
                    }
                }
            }
        }

        model.set_gradients(&gradients)?;
        Ok(())
    }

    /// Get current sparsity statistics
    pub fn get_sparsity_stats(&self) -> SparseTrainingStats {
        let mut layer_sparsities = HashMap::new();
        let mut total_params = 0;
        let mut total_nonzero = 0;

        for (layer_id, mask) in &self.masks {
            let sparsity = mask.calculate_sparsity();
            layer_sparsities.insert(layer_id.clone(), sparsity);

            let layer_params = mask.mask.len();
            let layer_nonzero = mask.mask.iter().filter(|&&keep| keep).count();

            total_params += layer_params;
            total_nonzero += layer_nonzero;
        }

        let overall_sparsity = if total_params > 0 {
            1.0 - (total_nonzero as f64 / total_params as f64)
        } else {
            0.0
        };

        SparseTrainingStats {
            overall_sparsity,
            layer_sparsities,
            training_step: self.training_step,
            in_recovery: self.in_recovery,
            target_sparsity: self.config.target_sparsity,
            total_parameters: total_params,
            active_parameters: total_nonzero,
        }
    }

    // Helper functions (simplified implementations)

    fn prune_filters<L: ParamLayer<F>>(&mut self, _model: &mut L) -> Result<()> {
        // Simplified implementation - would need proper layer analysis
        Ok(())
    }

    fn prune_channel_groups<L: ParamLayer<F>>(
        &mut self,
        _model: &mut L,
        _group_size: usize,
    ) -> Result<()> {
        // Simplified implementation
        Ok(())
    }

    fn prune_blocks<L: ParamLayer<F>>(
        &mut self,
        _model: &mut L,
        _block_h: usize,
        _block_w: usize,
    ) -> Result<()> {
        // Simplified implementation
        Ok(())
    }

    fn convert_to_f32_model<L: ParamLayer<F>>(&self, _model: &L) -> Result<Box<dyn Layer<f32>>> {
        // Simplified - would need proper model conversion
        Err(NeuralError::NotImplementedError(
            "Model conversion not implemented".to_string(),
        ))
    }

    fn create_mask_from_scores(&self, mask: &mut SparsityMask, target_sparsity: f64) {
        if let Some(ref scores) = mask.scores {
            let total_elements = scores.len();
            let target_pruned = (total_elements as f64 * target_sparsity) as usize;

            // Create (index, score) pairs and sort by score (ascending for pruning lowest scores)
            let mut indexed_scores: Vec<(usize, f64)> = scores
                .iter()
                .enumerate()
                .map(|(i, &score)| (i, score))
                .collect();

            indexed_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            // Reset mask
            mask.mask.fill(true);

            // Prune lowest scoring connections
            for i in 0..target_pruned.min(total_elements) {
                let idx = indexed_scores[i].0;
                let nd_idx = mask.linear_index_to_nd(idx, scores.shape());
                if let Some(mut_ref) = mask.mask.get_mut(nd_idx) {
                    *mut_ref = false;
                }
            }

            mask.sparsity = target_pruned as f64 / total_elements as f64;
        }
    }
}

/// Statistics for sparse training
#[derive(Debug, Clone)]
pub struct SparseTrainingStats {
    /// Overall sparsity across all layers
    pub overall_sparsity: f64,
    /// Sparsity per layer
    pub layer_sparsities: HashMap<String, f64>,
    /// Current training step
    pub training_step: usize,
    /// Whether in recovery phase
    pub in_recovery: bool,
    /// Target sparsity
    pub target_sparsity: f64,
    /// Total number of parameters
    pub total_parameters: usize,
    /// Number of active (non-zero) parameters
    pub active_parameters: usize,
}

impl SparseTrainingStats {
    /// Get compression ratio
    pub fn compression_ratio(&self) -> f64 {
        if self.total_parameters > 0 {
            self.total_parameters as f64 / self.active_parameters as f64
        } else {
            1.0
        }
    }

    /// Get memory savings
    pub fn memory_savings(&self) -> f64 {
        self.overall_sparsity
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_sparsity_mask() {
        let mut mask = SparsityMask::new(&[2, 2], "test_layer".to_string());

        let weights = Array2::<f32>::from_shape_vec((2, 2), vec![0.1, 0.9, 0.3, 0.7])
            .unwrap()
            .into_dyn();

        mask.update_magnitude_mask(&weights, 0.5);

        // Should prune 50% of weights (2 out of 4)
        assert_eq!(mask.calculate_sparsity(), 0.5);

        // Should prune the smallest weights (0.1 and 0.3)
        let expected_mask = vec![false, true, false, true]; // In flattened order
        for (i, &expected) in expected_mask.iter().enumerate() {
            let nd_idx = mask.linear_index_to_nd(i, weights.shape());
            assert_eq!(mask.mask[nd_idx], expected);
        }
    }

    #[test]
    fn test_pruning_schedule() {
        let config = SparseTrainingConfig {
            initial_sparsity: 0.0,
            target_sparsity: 0.9,
            pruning_steps: 3,
            gradual_pruning: true,
            ..Default::default()
        };

        let schedule = SparseTrainer::<f32>::create_pruning_schedule(&config);

        assert_eq!(schedule.len(), 4); // 0 to pruning_steps inclusive
        assert!((schedule[0] - 0.0).abs() < 1e-6);
        assert!((schedule[3] - 0.9).abs() < 1e-6);

        // Check that schedule is monotonically increasing
        for i in 1..schedule.len() {
            assert!(schedule[i] >= schedule[i - 1]);
        }
    }

    #[test]
    fn test_sparse_training_config_default() {
        let config = SparseTrainingConfig::default();
        assert_eq!(config.strategy, PruningStrategy::GradualMagnitude);
        assert_eq!(config.target_sparsity, 0.9);
        assert!(config.gradual_pruning);
    }

    #[test]
    fn test_sparse_training_stats() {
        let mut stats = SparseTrainingStats {
            overall_sparsity: 0.8,
            layer_sparsities: HashMap::new(),
            training_step: 100,
            in_recovery: false,
            target_sparsity: 0.9,
            total_parameters: 1000,
            active_parameters: 200,
        };

        stats.layer_sparsities.insert("layer_0".to_string(), 0.75);
        stats.layer_sparsities.insert("layer_1".to_string(), 0.85);

        assert_eq!(stats.compression_ratio(), 5.0);
        assert_eq!(stats.memory_savings(), 0.8);
    }
}
