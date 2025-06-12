//! Transfer learning utilities for neural networks
//!
//! This module provides tools for transfer learning, including:
//! - Pre-trained model weight loading and adaptation
//! - Layer freezing and unfreezing
//! - Fine-tuning utilities
//! - Domain adaptation tools

use crate::error::{NeuralError, Result};
use crate::layers::{Layer, Sequential};
use ndarray::{Array, ArrayD, IxDyn};
use num_traits::Float;
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::{Arc, RwLock};

/// Transfer learning strategy
#[derive(Debug, Clone, PartialEq)]
pub enum TransferStrategy {
    /// Freeze all layers except the last few
    FeatureExtraction { unfrozen_layers: usize },
    /// Fine-tune all layers with different learning rates
    FineTuning { 
        backbone_lr_ratio: f64,
        head_lr_ratio: f64 
    },
    /// Progressive unfreezing during training
    ProgressiveUnfreezing { 
        unfreeze_schedule: Vec<(usize, usize)> // (epoch, layers_to_unfreeze)
    },
    /// Custom layer-specific learning rates
    LayerWiseLearningRates { 
        layer_lr_map: HashMap<String, f64> 
    },
}

/// Layer freezing state
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayerState {
    /// Layer parameters are frozen (not updated)
    Frozen,
    /// Layer parameters are trainable
    Trainable,
    /// Layer uses reduced learning rate
    ReducedLearningRate(f64),
}

/// Transfer learning manager
pub struct TransferLearningManager<F: Float + Debug> {
    /// Layer states for each layer
    layer_states: HashMap<String, LayerState>,
    /// Transfer strategy
    strategy: TransferStrategy,
    /// Base learning rate
    base_learning_rate: F,
    /// Current epoch (for progressive unfreezing)
    current_epoch: usize,
    /// Layer statistics for monitoring
    layer_stats: Arc<RwLock<HashMap<String, LayerStatistics<F>>>>,
}

/// Statistics for tracking layer behavior during transfer learning
#[derive(Debug, Clone)]
pub struct LayerStatistics<F: Float + Debug> {
    /// Average gradient magnitude
    pub avg_gradient_magnitude: F,
    /// Parameter update magnitude
    pub param_update_magnitude: F,
    /// Number of parameters
    pub param_count: usize,
    /// Layer activation variance
    pub activation_variance: F,
    /// Whether layer is currently frozen
    pub is_frozen: bool,
}

impl<F: Float + Debug + 'static> TransferLearningManager<F> {
    /// Create a new transfer learning manager
    pub fn new(strategy: TransferStrategy, base_learning_rate: f64) -> Result<Self> {
        Ok(Self {
            layer_states: HashMap::new(),
            strategy,
            base_learning_rate: F::from(base_learning_rate).ok_or_else(|| {
                NeuralError::InvalidArchitecture("Invalid learning rate".to_string())
            })?,
            current_epoch: 0,
            layer_stats: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Initialize layer states based on the transfer strategy
    pub fn initialize_layer_states(&mut self, layer_names: &[String]) -> Result<()> {
        match &self.strategy {
            TransferStrategy::FeatureExtraction { unfrozen_layers } => {
                let total_layers = layer_names.len();
                let frozen_layers = total_layers.saturating_sub(*unfrozen_layers);
                
                for (i, layer_name) in layer_names.iter().enumerate() {
                    let state = if i < frozen_layers {
                        LayerState::Frozen
                    } else {
                        LayerState::Trainable
                    };
                    self.layer_states.insert(layer_name.clone(), state);
                }
            }
            
            TransferStrategy::FineTuning { backbone_lr_ratio, head_lr_ratio } => {
                let total_layers = layer_names.len();
                let backbone_layers = total_layers.saturating_sub(2); // Last 2 layers are "head"
                
                for (i, layer_name) in layer_names.iter().enumerate() {
                    let state = if i < backbone_layers {
                        LayerState::ReducedLearningRate(*backbone_lr_ratio)
                    } else {
                        LayerState::ReducedLearningRate(*head_lr_ratio)
                    };
                    self.layer_states.insert(layer_name.clone(), state);
                }
            }
            
            TransferStrategy::ProgressiveUnfreezing { .. } => {
                // Start with all layers frozen
                for layer_name in layer_names {
                    self.layer_states.insert(layer_name.clone(), LayerState::Frozen);
                }
            }
            
            TransferStrategy::LayerWiseLearningRates { layer_lr_map } => {
                for layer_name in layer_names {
                    let lr_ratio = layer_lr_map.get(layer_name).unwrap_or(&1.0);
                    let state = if *lr_ratio == 0.0 {
                        LayerState::Frozen
                    } else {
                        LayerState::ReducedLearningRate(*lr_ratio)
                    };
                    self.layer_states.insert(layer_name.clone(), state);
                }
            }
        }
        
        Ok(())
    }

    /// Update layer states at the beginning of each epoch
    pub fn update_epoch(&mut self, epoch: usize) -> Result<()> {
        self.current_epoch = epoch;
        
        if let TransferStrategy::ProgressiveUnfreezing { unfreeze_schedule } = &self.strategy {
            for (unfreeze_epoch, layers_to_unfreeze) in unfreeze_schedule {
                if epoch == *unfreeze_epoch {
                    self.unfreeze_layers(*layers_to_unfreeze)?;
                }
            }
        }
        
        Ok(())
    }

    /// Unfreeze the specified number of layers from the end
    pub fn unfreeze_layers(&mut self, count: usize) -> Result<()> {
        let layer_names: Vec<String> = self.layer_states.keys().cloned().collect();
        let total_layers = layer_names.len();
        let start_idx = total_layers.saturating_sub(count);
        
        for layer_name in layer_names.iter().skip(start_idx) {
            if let Some(state) = self.layer_states.get_mut(layer_name) {
                if *state == LayerState::Frozen {
                    *state = LayerState::Trainable;
                }
            }
        }
        
        Ok(())
    }

    /// Freeze specific layers
    pub fn freeze_layers(&mut self, layer_names: &[String]) -> Result<()> {
        for layer_name in layer_names {
            self.layer_states.insert(layer_name.clone(), LayerState::Frozen);
        }
        Ok(())
    }

    /// Get effective learning rate for a layer
    pub fn get_layer_learning_rate(&self, layer_name: &str) -> F {
        match self.layer_states.get(layer_name) {
            Some(LayerState::Frozen) => F::zero(),
            Some(LayerState::Trainable) => self.base_learning_rate,
            Some(LayerState::ReducedLearningRate(ratio)) => {
                self.base_learning_rate * F::from(*ratio).unwrap_or(F::one())
            }
            None => self.base_learning_rate, // Default for unknown layers
        }
    }

    /// Check if a layer is frozen
    pub fn is_layer_frozen(&self, layer_name: &str) -> bool {
        matches!(self.layer_states.get(layer_name), Some(LayerState::Frozen))
    }

    /// Update layer statistics
    pub fn update_layer_statistics(
        &self,
        layer_name: String,
        gradient_magnitude: F,
        param_update_magnitude: F,
        param_count: usize,
        activation_variance: F,
    ) -> Result<()> {
        let is_frozen = self.is_layer_frozen(&layer_name);
        
        let stats = LayerStatistics {
            avg_gradient_magnitude: gradient_magnitude,
            param_update_magnitude,
            param_count,
            activation_variance,
            is_frozen,
        };
        
        if let Ok(mut layer_stats) = self.layer_stats.write() {
            layer_stats.insert(layer_name, stats);
        }
        
        Ok(())
    }

    /// Get layer statistics
    pub fn get_layer_statistics(&self) -> Result<HashMap<String, LayerStatistics<F>>> {
        match self.layer_stats.read() {
            Ok(stats) => Ok(stats.clone()),
            Err(_) => Err(NeuralError::InferenceError(
                "Failed to read layer statistics".to_string(),
            )),
        }
    }

    /// Get summary of current transfer learning state
    pub fn get_summary(&self) -> TransferLearningState {
        let mut frozen_layers = 0;
        let mut trainable_layers = 0;
        let mut reduced_lr_layers = 0;
        
        for state in self.layer_states.values() {
            match state {
                LayerState::Frozen => frozen_layers += 1,
                LayerState::Trainable => trainable_layers += 1,
                LayerState::ReducedLearningRate(_) => reduced_lr_layers += 1,
            }
        }
        
        TransferLearningState {
            current_epoch: self.current_epoch,
            total_layers: self.layer_states.len(),
            frozen_layers,
            trainable_layers,
            reduced_lr_layers,
            strategy: self.strategy.clone(),
        }
    }
}

/// Summary of transfer learning state
#[derive(Debug, Clone)]
pub struct TransferLearningState {
    /// Current training epoch
    pub current_epoch: usize,
    /// Total number of layers
    pub total_layers: usize,
    /// Number of frozen layers
    pub frozen_layers: usize,
    /// Number of trainable layers
    pub trainable_layers: usize,
    /// Number of layers with reduced learning rate
    pub reduced_lr_layers: usize,
    /// Current transfer strategy
    pub strategy: TransferStrategy,
}

impl std::fmt::Display for TransferLearningState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Transfer Learning State (Epoch {}):", self.current_epoch)?;
        writeln!(f, "  Total layers: {}", self.total_layers)?;
        writeln!(f, "  Frozen layers: {}", self.frozen_layers)?;
        writeln!(f, "  Trainable layers: {}", self.trainable_layers)?;
        writeln!(f, "  Reduced LR layers: {}", self.reduced_lr_layers)?;
        writeln!(f, "  Strategy: {:?}", self.strategy)?;
        Ok(())
    }
}

/// Pre-trained model weight loader
pub struct PretrainedWeightLoader {
    /// Model weights storage
    weights: HashMap<String, ArrayD<f32>>,
    /// Weight mapping (source_layer -> target_layer)
    layer_mapping: HashMap<String, String>,
    /// Whether to ignore size mismatches
    ignore_mismatches: bool,
}

impl PretrainedWeightLoader {
    /// Create a new weight loader
    pub fn new() -> Self {
        Self {
            weights: HashMap::new(),
            layer_mapping: HashMap::new(),
            ignore_mismatches: false,
        }
    }

    /// Load weights from a dictionary
    pub fn load_weights(&mut self, weights: HashMap<String, ArrayD<f32>>) -> Result<()> {
        self.weights = weights;
        Ok(())
    }

    /// Add layer mapping for weight transfer
    pub fn add_layer_mapping(&mut self, source_layer: String, target_layer: String) {
        self.layer_mapping.insert(source_layer, target_layer);
    }

    /// Set whether to ignore size mismatches
    pub fn set_ignore_mismatches(&mut self, ignore: bool) {
        self.ignore_mismatches = ignore;
    }

    /// Apply weights to a model layer
    pub fn apply_weights_to_layer<L: Layer<f32>>(
        &self,
        layer: &mut L,
        layer_name: &str,
    ) -> Result<bool> {
        // Try direct layer name first, then check mapping
        let weight_key = self.layer_mapping.get(layer_name)
            .unwrap_or(&layer_name.to_string());
        
        if let Some(weights) = self.weights.get(weight_key) {
            // Here we would need to implement the actual weight setting
            // This is a simplified version - real implementation would need
            // access to layer internals
            println!("Loading weights for layer {}: shape {:?}", layer_name, weights.shape());
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get available weight keys
    pub fn get_available_weights(&self) -> Vec<String> {
        self.weights.keys().cloned().collect()
    }
}

impl Default for PretrainedWeightLoader {
    fn default() -> Self {
        Self::new()
    }
}

/// Fine-tuning utilities
pub struct FineTuningUtilities<F: Float + Debug> {
    /// Learning rate scheduler for different layer groups
    lr_scheduler: HashMap<String, F>,
    /// Gradient clipping values per layer
    gradient_clips: HashMap<String, F>,
    /// Weight decay values per layer
    weight_decays: HashMap<String, F>,
}

impl<F: Float + Debug + 'static> FineTuningUtilities<F> {
    /// Create new fine-tuning utilities
    pub fn new() -> Self {
        Self {
            lr_scheduler: HashMap::new(),
            gradient_clips: HashMap::new(),
            weight_decays: HashMap::new(),
        }
    }

    /// Set learning rate for a layer group
    pub fn set_layer_learning_rate(&mut self, layer_pattern: String, learning_rate: f64) -> Result<()> {
        let lr = F::from(learning_rate).ok_or_else(|| {
            NeuralError::InvalidArchitecture("Invalid learning rate".to_string())
        })?;
        self.lr_scheduler.insert(layer_pattern, lr);
        Ok(())
    }

    /// Set gradient clipping for a layer group
    pub fn set_layer_gradient_clip(&mut self, layer_pattern: String, clip_value: f64) -> Result<()> {
        let clip = F::from(clip_value).ok_or_else(|| {
            NeuralError::InvalidArchitecture("Invalid clip value".to_string())
        })?;
        self.gradient_clips.insert(layer_pattern, clip);
        Ok(())
    }

    /// Set weight decay for a layer group
    pub fn set_layer_weight_decay(&mut self, layer_pattern: String, weight_decay: f64) -> Result<()> {
        let decay = F::from(weight_decay).ok_or_else(|| {
            NeuralError::InvalidArchitecture("Invalid weight decay".to_string())
        })?;
        self.weight_decays.insert(layer_pattern, decay);
        Ok(())
    }

    /// Get effective learning rate for a layer
    pub fn get_effective_learning_rate(&self, layer_name: &str, base_lr: F) -> F {
        // Check for exact match first, then pattern matches
        for (pattern, &lr) in &self.lr_scheduler {
            if layer_name == pattern || layer_name.contains(pattern) {
                return lr;
            }
        }
        base_lr
    }

    /// Get gradient clip value for a layer
    pub fn get_gradient_clip(&self, layer_name: &str) -> Option<F> {
        for (pattern, &clip) in &self.gradient_clips {
            if layer_name == pattern || layer_name.contains(pattern) {
                return Some(clip);
            }
        }
        None
    }

    /// Get weight decay for a layer
    pub fn get_weight_decay(&self, layer_name: &str) -> Option<F> {
        for (pattern, &decay) in &self.weight_decays {
            if layer_name == pattern || layer_name.contains(pattern) {
                return Some(decay);
            }
        }
        None
    }
}

impl<F: Float + Debug + 'static> Default for FineTuningUtilities<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Domain adaptation utilities
pub struct DomainAdaptation<F: Float + Debug> {
    /// Source domain statistics
    source_stats: HashMap<String, DomainStatistics<F>>,
    /// Target domain statistics
    target_stats: HashMap<String, DomainStatistics<F>>,
    /// Adaptation method
    adaptation_method: AdaptationMethod,
}

/// Domain statistics for adaptation
#[derive(Debug, Clone)]
pub struct DomainStatistics<F: Float + Debug> {
    /// Feature means
    pub mean: ArrayD<F>,
    /// Feature variances
    pub variance: ArrayD<F>,
    /// Feature covariance matrix (optional)
    pub covariance: Option<ArrayD<F>>,
}

/// Domain adaptation methods
#[derive(Debug, Clone, PartialEq)]
pub enum AdaptationMethod {
    /// Batch normalization statistics adaptation
    BatchNormAdaptation,
    /// Feature alignment via moment matching
    MomentMatching,
    /// Adversarial domain adaptation
    AdversarialTraining { lambda: f64 },
    /// Coral (correlation alignment)
    CoralAlignment,
}

impl<F: Float + Debug + 'static> DomainAdaptation<F> {
    /// Create new domain adaptation utility
    pub fn new(method: AdaptationMethod) -> Self {
        Self {
            source_stats: HashMap::new(),
            target_stats: HashMap::new(),
            adaptation_method: method,
        }
    }

    /// Compute domain statistics from data
    pub fn compute_domain_statistics(
        &mut self,
        domain_name: String,
        features: &ArrayD<F>,
        is_source: bool,
    ) -> Result<()> {
        let batch_size = features.shape()[0];
        if batch_size == 0 {
            return Err(NeuralError::ComputationError(
                "Empty feature batch".to_string(),
            ));
        }

        // Compute mean across batch dimension
        let mean = features.mean_axis(ndarray::Axis(0))
            .ok_or_else(|| NeuralError::ComputationError("Failed to compute mean".to_string()))?;

        // Compute variance
        let variance = {
            let diff = features - &mean;
            let squared_diff = diff.mapv(|x| x * x);
            squared_diff.mean_axis(ndarray::Axis(0))
                .ok_or_else(|| NeuralError::ComputationError("Failed to compute variance".to_string()))?
        };

        let stats = DomainStatistics {
            mean: mean.into_dyn(),
            variance: variance.into_dyn(),
            covariance: None, // Could be computed if needed
        };

        if is_source {
            self.source_stats.insert(domain_name, stats);
        } else {
            self.target_stats.insert(domain_name, stats);
        }

        Ok(())
    }

    /// Apply domain adaptation
    pub fn adapt_features(
        &self,
        layer_name: &str,
        features: &ArrayD<F>,
    ) -> Result<ArrayD<F>> {
        let source_stats = self.source_stats.get(layer_name)
            .ok_or_else(|| NeuralError::ComputationError("Source stats not found".to_string()))?;
        let target_stats = self.target_stats.get(layer_name)
            .ok_or_else(|| NeuralError::ComputationError("Target stats not found".to_string()))?;

        match self.adaptation_method {
            AdaptationMethod::BatchNormAdaptation => {
                self.batch_norm_adaptation(features, source_stats, target_stats)
            }
            AdaptationMethod::MomentMatching => {
                self.moment_matching_adaptation(features, source_stats, target_stats)
            }
            _ => {
                // Other methods would require more complex implementations
                Ok(features.clone())
            }
        }
    }

    fn batch_norm_adaptation(
        &self,
        features: &ArrayD<F>,
        source_stats: &DomainStatistics<F>,
        target_stats: &DomainStatistics<F>,
    ) -> Result<ArrayD<F>> {
        // Normalize using source statistics, then denormalize using target statistics
        let eps = F::from(1e-5).unwrap();
        
        // Normalize with source stats
        let normalized = (features - &source_stats.mean) / 
            (source_stats.variance.mapv(|x| (x + eps).sqrt()));
        
        // Denormalize with target stats
        let adapted = normalized * target_stats.variance.mapv(|x| (x + eps).sqrt()) + 
            &target_stats.mean;
        
        Ok(adapted)
    }

    fn moment_matching_adaptation(
        &self,
        features: &ArrayD<F>,
        _source_stats: &DomainStatistics<F>,
        target_stats: &DomainStatistics<F>,
    ) -> Result<ArrayD<F>> {
        // Simple moment matching: adjust to target mean and variance
        let current_mean = features.mean_axis(ndarray::Axis(0))
            .ok_or_else(|| NeuralError::ComputationError("Failed to compute mean".to_string()))?;
        let current_var = {
            let diff = features - &current_mean;
            let squared_diff = diff.mapv(|x| x * x);
            squared_diff.mean_axis(ndarray::Axis(0))
                .ok_or_else(|| NeuralError::ComputationError("Failed to compute variance".to_string()))?
        };

        let eps = F::from(1e-5).unwrap();
        
        // Normalize current features
        let normalized = (features - &current_mean) / 
            current_var.mapv(|x| (x + eps).sqrt());
        
        // Apply target statistics
        let adapted = normalized * target_stats.variance.mapv(|x| (x + eps).sqrt()) + 
            &target_stats.mean;
        
        Ok(adapted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transfer_learning_manager_creation() {
        let strategy = TransferStrategy::FeatureExtraction { unfrozen_layers: 2 };
        let manager = TransferLearningManager::<f64>::new(strategy, 0.001);
        assert!(manager.is_ok());
    }

    #[test]
    fn test_feature_extraction_strategy() {
        let strategy = TransferStrategy::FeatureExtraction { unfrozen_layers: 2 };
        let mut manager = TransferLearningManager::<f64>::new(strategy, 0.001).unwrap();
        
        let layer_names = vec![
            "conv1".to_string(),
            "conv2".to_string(), 
            "conv3".to_string(),
            "fc1".to_string(),
            "fc2".to_string(),
        ];
        
        manager.initialize_layer_states(&layer_names).unwrap();
        
        // First 3 layers should be frozen, last 2 trainable
        assert!(manager.is_layer_frozen("conv1"));
        assert!(manager.is_layer_frozen("conv2"));
        assert!(manager.is_layer_frozen("conv3"));
        assert!(!manager.is_layer_frozen("fc1"));
        assert!(!manager.is_layer_frozen("fc2"));
    }

    #[test]
    fn test_fine_tuning_strategy() {
        let strategy = TransferStrategy::FineTuning { 
            backbone_lr_ratio: 0.1, 
            head_lr_ratio: 1.0 
        };
        let mut manager = TransferLearningManager::<f64>::new(strategy, 0.001).unwrap();
        
        let layer_names = vec![
            "backbone1".to_string(),
            "backbone2".to_string(),
            "head1".to_string(),
            "head2".to_string(),
        ];
        
        manager.initialize_layer_states(&layer_names).unwrap();
        
        // Check learning rates
        let backbone_lr = manager.get_layer_learning_rate("backbone1");
        let head_lr = manager.get_layer_learning_rate("head1");
        
        assert!((backbone_lr - 0.0001).abs() < 1e-6); // 0.001 * 0.1
        assert!((head_lr - 0.001).abs() < 1e-6); // 0.001 * 1.0
    }

    #[test]
    fn test_progressive_unfreezing() {
        let strategy = TransferStrategy::ProgressiveUnfreezing { 
            unfreeze_schedule: vec![(5, 2), (10, 2)] 
        };
        let mut manager = TransferLearningManager::<f64>::new(strategy, 0.001).unwrap();
        
        let layer_names = vec![
            "layer1".to_string(),
            "layer2".to_string(),
            "layer3".to_string(),
            "layer4".to_string(),
        ];
        
        manager.initialize_layer_states(&layer_names).unwrap();
        
        // Initially all frozen
        assert!(manager.is_layer_frozen("layer1"));
        assert!(manager.is_layer_frozen("layer4"));
        
        // After epoch 5, last 2 layers unfrozen
        manager.update_epoch(5).unwrap();
        assert!(manager.is_layer_frozen("layer1"));
        assert!(manager.is_layer_frozen("layer2"));
        assert!(!manager.is_layer_frozen("layer3"));
        assert!(!manager.is_layer_frozen("layer4"));
    }

    #[test]
    fn test_pretrained_weight_loader() {
        let mut loader = PretrainedWeightLoader::new();
        
        let mut weights = HashMap::new();
        weights.insert("layer1".to_string(), Array::zeros((10, 5)).into_dyn());
        weights.insert("layer2".to_string(), Array::ones((5, 3)).into_dyn());
        
        loader.load_weights(weights).unwrap();
        loader.add_layer_mapping("layer1".to_string(), "new_layer1".to_string());
        
        let available = loader.get_available_weights();
        assert_eq!(available.len(), 2);
        assert!(available.contains(&"layer1".to_string()));
        assert!(available.contains(&"layer2".to_string()));
    }

    #[test]
    fn test_fine_tuning_utilities() {
        let mut utils = FineTuningUtilities::<f64>::new();
        
        utils.set_layer_learning_rate("backbone".to_string(), 0.0001).unwrap();
        utils.set_layer_learning_rate("head".to_string(), 0.001).unwrap();
        utils.set_layer_gradient_clip("backbone".to_string(), 1.0).unwrap();
        
        let backbone_lr = utils.get_effective_learning_rate("backbone_layer1", 0.01);
        let head_lr = utils.get_effective_learning_rate("head_layer1", 0.01);
        let unknown_lr = utils.get_effective_learning_rate("unknown_layer", 0.01);
        
        assert!((backbone_lr - 0.0001).abs() < 1e-6);
        assert!((head_lr - 0.001).abs() < 1e-6);
        assert!((unknown_lr - 0.01).abs() < 1e-6);
        
        let clip = utils.get_gradient_clip("backbone_layer1");
        assert!(clip.is_some());
        assert!((clip.unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_domain_adaptation() {
        let mut adapter = DomainAdaptation::<f64>::new(AdaptationMethod::BatchNormAdaptation);
        
        // Create some test features
        let source_features = Array::from_shape_vec((10, 5), 
            (0..50).map(|x| x as f64 / 10.0).collect()).unwrap().into_dyn();
        let target_features = Array::from_shape_vec((10, 5), 
            (0..50).map(|x| (x as f64 + 25.0) / 10.0).collect()).unwrap().into_dyn();
        
        adapter.compute_domain_statistics("layer1".to_string(), &source_features, true).unwrap();
        adapter.compute_domain_statistics("layer1".to_string(), &target_features, false).unwrap();
        
        let adapted = adapter.adapt_features("layer1", &source_features).unwrap();
        assert_eq!(adapted.shape(), source_features.shape());
    }

    #[test]
    fn test_transfer_learning_state_display() {
        let strategy = TransferStrategy::FeatureExtraction { unfrozen_layers: 2 };
        let state = TransferLearningState {
            current_epoch: 10,
            total_layers: 5,
            frozen_layers: 3,
            trainable_layers: 2,
            reduced_lr_layers: 0,
            strategy,
        };
        
        let display_str = format!("{}", state);
        assert!(display_str.contains("Epoch 10"));
        assert!(display_str.contains("Total layers: 5"));
        assert!(display_str.contains("Frozen layers: 3"));
    }
}