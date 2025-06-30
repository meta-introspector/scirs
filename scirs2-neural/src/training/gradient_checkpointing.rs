//! Gradient Checkpointing for Memory-Efficient Training
//!
//! This module implements gradient checkpointing (also known as gradient accumulation),
//! a technique that trades computation for memory by recomputing activations during
//! the backward pass instead of storing them during the forward pass.

use crate::error::{NeuralError, Result};
use crate::layers::{Layer, ParamLayer};
use crate::losses::Loss;
use crate::optimizers::{Optimizer, OptimizerStep};
use ndarray::{Array, IxDyn};
use num_traits::{Float, FromPrimitive};
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::{Arc, Mutex};

/// Gradient checkpointing configuration
#[derive(Debug, Clone)]
pub struct GradientCheckpointingConfig {
    /// Number of segments to divide the model into for checkpointing
    pub num_segments: usize,
    /// Memory budget in bytes (if Some, automatically determine segments)
    pub memory_budget: Option<usize>,
    /// Whether to use adaptive checkpointing based on memory usage
    pub adaptive: bool,
    /// Whether to preserve RNG state for dropout layers
    pub preserve_rng_state: bool,
    /// Recomputation strategy
    pub strategy: CheckpointingStrategy,
}

impl Default for GradientCheckpointingConfig {
    fn default() -> Self {
        Self {
            num_segments: 4,
            memory_budget: None,
            adaptive: true,
            preserve_rng_state: true,
            strategy: CheckpointingStrategy::Uniform,
        }
    }
}

/// Checkpointing strategies
#[derive(Debug, Clone, PartialEq)]
pub enum CheckpointingStrategy {
    /// Uniform segmentation - divide model into equal segments
    Uniform,
    /// Memory-aware - segment based on memory usage
    MemoryAware,
    /// Computation-aware - segment based on computational cost
    ComputationAware,
    /// Custom segments defined by user
    Custom(Vec<usize>),
}

/// Checkpoint segment information
#[derive(Debug, Clone)]
pub struct CheckpointSegment {
    /// Segment ID
    pub id: usize,
    /// Start layer index
    pub start_layer: usize,
    /// End layer index (exclusive)
    pub end_layer: usize,
    /// Memory usage estimate in bytes
    pub memory_usage: usize,
    /// Computation cost estimate (arbitrary units)
    pub computation_cost: f64,
    /// Whether this segment should be checkpointed
    pub checkpoint: bool,
}

/// Activation checkpoint for storing intermediate activations
#[derive(Debug, Clone)]
pub struct ActivationCheckpoint<F: Float + Debug + Clone> {
    /// Checkpointed activations
    pub activations: Array<F, IxDyn>,
    /// Layer index where checkpoint was taken
    pub layer_index: usize,
    /// RNG state at checkpoint (if preserved)
    pub rng_state: Option<Vec<u8>>,
    /// Metadata for the checkpoint
    pub metadata: CheckpointMetadata,
}

/// Metadata for activation checkpoints
#[derive(Debug, Clone)]
pub struct CheckpointMetadata {
    /// Timestamp when checkpoint was created
    pub timestamp: std::time::SystemTime,
    /// Memory size of the checkpoint in bytes
    pub memory_size: usize,
    /// Computation cost to recreate from this checkpoint
    pub recomputation_cost: f64,
}

/// Gradient checkpointing manager
pub struct GradientCheckpointingManager<F: Float + Debug + Clone + Send + Sync + FromPrimitive> {
    /// Configuration
    config: GradientCheckpointingConfig,
    /// Segments for checkpointing
    segments: Vec<CheckpointSegment>,
    /// Active checkpoints
    checkpoints: Arc<Mutex<HashMap<usize, ActivationCheckpoint<F>>>>,
    /// Memory usage tracker
    memory_tracker: MemoryTracker,
    /// Model layer information
    layer_info: Vec<LayerInfo>,
    /// Current forward pass state
    forward_state: ForwardState<F>,
    /// Statistics
    stats: CheckpointingStats,
}

impl<F: Float + Debug + Clone + Send + Sync + FromPrimitive + ndarray::ScalarOperand>
    GradientCheckpointingManager<F>
{
    /// Create a new gradient checkpointing manager
    pub fn new(config: GradientCheckpointingConfig) -> Self {
        Self {
            config,
            segments: Vec::new(),
            checkpoints: Arc::new(Mutex::new(HashMap::new())),
            memory_tracker: MemoryTracker::new(),
            layer_info: Vec::new(),
            forward_state: ForwardState::new(),
            stats: CheckpointingStats::default(),
        }
    }

    /// Initialize checkpointing for a model
    pub fn initialize<L: Layer<F> + ?Sized>(&mut self, model: &L) -> Result<()> {
        // Analyze model structure
        self.analyze_model(model)?;

        // Create segments based on strategy
        self.create_segments()?;

        // Initialize memory tracker
        self.memory_tracker.reset();

        Ok(())
    }

    /// Analyze model structure to determine optimal checkpointing strategy
    fn analyze_model<L: Layer<F> + ?Sized>(&mut self, model: &L) -> Result<()> {
        // Get model parameters to estimate layer sizes
        let params = model.params();

        // Create layer info for each parameter group (simplified)
        self.layer_info = params
            .iter()
            .enumerate()
            .map(|(i, param)| {
                let memory_usage = param.len() * std::mem::size_of::<F>();
                let computation_cost = param.len() as f64; // Simplified estimate

                LayerInfo {
                    layer_index: i,
                    memory_usage,
                    computation_cost,
                    layer_type: LayerType::Unknown,
                    input_shape: param.shape().to_vec(),
                    output_shape: param.shape().to_vec(),
                }
            })
            .collect();

        Ok(())
    }

    /// Create checkpointing segments based on the configured strategy
    fn create_segments(&mut self) -> Result<()> {
        if self.layer_info.is_empty() {
            return Err(NeuralError::InvalidState(
                "No layer information available".to_string(),
            ));
        }

        match &self.config.strategy {
            CheckpointingStrategy::Uniform => self.create_uniform_segments(),
            CheckpointingStrategy::MemoryAware => self.create_memory_aware_segments(),
            CheckpointingStrategy::ComputationAware => self.create_computation_aware_segments(),
            CheckpointingStrategy::Custom(segments) => self.create_custom_segments(segments),
        }
    }

    /// Create uniform segments
    fn create_uniform_segments(&mut self) -> Result<()> {
        let num_layers = self.layer_info.len();
        let segment_size = (num_layers + self.config.num_segments - 1) / self.config.num_segments;

        self.segments.clear();

        for i in 0..self.config.num_segments {
            let start = i * segment_size;
            let end = ((i + 1) * segment_size).min(num_layers);

            if start < end {
                let memory_usage: usize = self.layer_info[start..end]
                    .iter()
                    .map(|info| info.memory_usage)
                    .sum();

                let computation_cost: f64 = self.layer_info[start..end]
                    .iter()
                    .map(|info| info.computation_cost)
                    .sum();

                self.segments.push(CheckpointSegment {
                    id: i,
                    start_layer: start,
                    end_layer: end,
                    memory_usage,
                    computation_cost,
                    checkpoint: i > 0, // Don't checkpoint the first segment
                });
            }
        }

        Ok(())
    }

    /// Create memory-aware segments
    fn create_memory_aware_segments(&mut self) -> Result<()> {
        let total_memory: usize = self.layer_info.iter().map(|info| info.memory_usage).sum();

        let target_memory_per_segment = if let Some(budget) = self.config.memory_budget {
            budget / self.config.num_segments
        } else {
            total_memory / self.config.num_segments
        };

        self.segments.clear();

        let mut current_segment_memory = 0;
        let mut segment_start = 0;
        let mut segment_id = 0;

        for (i, layer_info) in self.layer_info.iter().enumerate() {
            current_segment_memory += layer_info.memory_usage;

            // If we've exceeded the target memory or reached the end
            if current_segment_memory >= target_memory_per_segment || i == self.layer_info.len() - 1
            {
                let computation_cost: f64 = self.layer_info[segment_start..=i]
                    .iter()
                    .map(|info| info.computation_cost)
                    .sum();

                self.segments.push(CheckpointSegment {
                    id: segment_id,
                    start_layer: segment_start,
                    end_layer: i + 1,
                    memory_usage: current_segment_memory,
                    computation_cost,
                    checkpoint: segment_id > 0,
                });

                // Reset for next segment
                current_segment_memory = 0;
                segment_start = i + 1;
                segment_id += 1;
            }
        }

        Ok(())
    }

    /// Create computation-aware segments
    fn create_computation_aware_segments(&mut self) -> Result<()> {
        let total_computation: f64 = self
            .layer_info
            .iter()
            .map(|info| info.computation_cost)
            .sum();

        let target_computation_per_segment = total_computation / self.config.num_segments as f64;

        self.segments.clear();

        let mut current_segment_computation = 0.0;
        let mut segment_start = 0;
        let mut segment_id = 0;

        for (i, layer_info) in self.layer_info.iter().enumerate() {
            current_segment_computation += layer_info.computation_cost;

            // If we've exceeded the target computation or reached the end
            if current_segment_computation >= target_computation_per_segment
                || i == self.layer_info.len() - 1
            {
                let memory_usage: usize = self.layer_info[segment_start..=i]
                    .iter()
                    .map(|info| info.memory_usage)
                    .sum();

                self.segments.push(CheckpointSegment {
                    id: segment_id,
                    start_layer: segment_start,
                    end_layer: i + 1,
                    memory_usage,
                    computation_cost: current_segment_computation,
                    checkpoint: segment_id > 0,
                });

                // Reset for next segment
                current_segment_computation = 0.0;
                segment_start = i + 1;
                segment_id += 1;
            }
        }

        Ok(())
    }

    /// Create custom segments
    fn create_custom_segments(&mut self, segment_boundaries: &[usize]) -> Result<()> {
        if segment_boundaries.is_empty() {
            return Err(NeuralError::InvalidArgument(
                "Custom segments cannot be empty".to_string(),
            ));
        }

        self.segments.clear();
        let mut start = 0;

        for (i, &end) in segment_boundaries.iter().enumerate() {
            if end > self.layer_info.len() {
                return Err(NeuralError::InvalidArgument(format!(
                    "Segment boundary {} exceeds number of layers {}",
                    end,
                    self.layer_info.len()
                )));
            }

            let memory_usage: usize = self.layer_info[start..end]
                .iter()
                .map(|info| info.memory_usage)
                .sum();

            let computation_cost: f64 = self.layer_info[start..end]
                .iter()
                .map(|info| info.computation_cost)
                .sum();

            self.segments.push(CheckpointSegment {
                id: i,
                start_layer: start,
                end_layer: end,
                memory_usage,
                computation_cost,
                checkpoint: i > 0,
            });

            start = end;
        }

        Ok(())
    }

    /// Forward pass with checkpointing
    pub fn forward_with_checkpointing<L: Layer<F> + ?Sized>(
        &mut self,
        model: &L,
        input: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        // Reset forward state
        self.forward_state.reset();

        let mut current_input = input.clone();
        let start_time = std::time::Instant::now();

        // Process each segment
        for segment in &self.segments {
            if segment.checkpoint {
                // Save checkpoint before this segment
                self.save_checkpoint(segment.id, &current_input, segment.start_layer)?;
            }

            // Forward through this segment (simplified - in practice would need layer-by-layer)
            current_input = model.forward(&current_input)?;

            // Track memory usage
            self.memory_tracker
                .add_allocation(current_input.len() * std::mem::size_of::<F>());
        }

        // Update statistics
        self.stats.forward_passes += 1;
        self.stats.total_forward_time += start_time.elapsed();

        Ok(current_input)
    }

    /// Backward pass with recomputation
    pub fn backward_with_recomputation<L: Layer<F> + ?Sized>(
        &mut self,
        model: &L,
        input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        let start_time = std::time::Instant::now();

        // Process segments in reverse order
        let mut current_grad = grad_output.clone();

        for segment in self.segments.iter().rev() {
            if segment.checkpoint {
                // Recompute activations from checkpoint
                let checkpoint_input = self.load_checkpoint(segment.id)?;

                // Recompute forward pass for this segment
                let recomputed_output = model.forward(&checkpoint_input)?;
                self.stats.recomputations += 1;

                // Backward pass through this segment
                current_grad = model.backward(&checkpoint_input, &current_grad)?;
            } else {
                // Use stored activations (first segment typically)
                current_grad = model.backward(input, &current_grad)?;
            }
        }

        // Update statistics
        self.stats.backward_passes += 1;
        self.stats.total_backward_time += start_time.elapsed();

        Ok(current_grad)
    }

    /// Save an activation checkpoint
    fn save_checkpoint(
        &mut self,
        segment_id: usize,
        activations: &Array<F, IxDyn>,
        layer_index: usize,
    ) -> Result<()> {
        let memory_size = activations.len() * std::mem::size_of::<F>();

        let checkpoint = ActivationCheckpoint {
            activations: activations.clone(),
            layer_index,
            rng_state: if self.config.preserve_rng_state {
                Some(self.capture_rng_state())
            } else {
                None
            },
            metadata: CheckpointMetadata {
                timestamp: std::time::SystemTime::now(),
                memory_size,
                recomputation_cost: self.estimate_recomputation_cost(segment_id),
            },
        };

        let mut checkpoints = self.checkpoints.lock().unwrap();
        checkpoints.insert(segment_id, checkpoint);

        // Update memory tracker
        self.memory_tracker.add_checkpoint(memory_size);
        self.stats.checkpoints_saved += 1;

        Ok(())
    }

    /// Load an activation checkpoint
    fn load_checkpoint(&self, segment_id: usize) -> Result<Array<F, IxDyn>> {
        let checkpoints = self.checkpoints.lock().unwrap();

        if let Some(checkpoint) = checkpoints.get(&segment_id) {
            // Restore RNG state if preserved
            if let Some(ref rng_state) = checkpoint.rng_state {
                self.restore_rng_state(rng_state);
            }

            Ok(checkpoint.activations.clone())
        } else {
            Err(NeuralError::InvalidState(format!(
                "Checkpoint {} not found",
                segment_id
            )))
        }
    }

    /// Estimate recomputation cost for a segment
    fn estimate_recomputation_cost(&self, segment_id: usize) -> f64 {
        if let Some(segment) = self.segments.get(segment_id) {
            segment.computation_cost
        } else {
            0.0
        }
    }

    /// Capture RNG state (simplified)
    fn capture_rng_state(&self) -> Vec<u8> {
        // In practice, would capture the actual RNG state
        // For now, return empty vector
        Vec::new()
    }

    /// Restore RNG state (simplified)
    fn restore_rng_state(&self, _rng_state: &[u8]) {
        // In practice, would restore the actual RNG state
    }

    /// Clear all checkpoints
    pub fn clear_checkpoints(&mut self) {
        let mut checkpoints = self.checkpoints.lock().unwrap();
        checkpoints.clear();
        self.memory_tracker.clear_checkpoints();
    }

    /// Get memory usage statistics
    pub fn get_memory_stats(&self) -> MemoryStats {
        self.memory_tracker.get_stats()
    }

    /// Get checkpointing statistics
    pub fn get_checkpointing_stats(&self) -> &CheckpointingStats {
        &self.stats
    }

    /// Get configuration
    #[allow(dead_code)]
    pub fn get_config(&self) -> &GradientCheckpointingConfig {
        &self.config
    }

    /// Get number of checkpoints currently stored
    #[allow(dead_code)]
    pub fn num_checkpoints(&self) -> usize {
        self.checkpoints.lock().unwrap().len()
    }

    /// Get current segments
    pub fn get_segments(&self) -> &[CheckpointSegment] {
        &self.segments
    }

    /// Train step with gradient checkpointing
    pub fn train_step_with_checkpointing<L, O>(
        &mut self,
        model: &mut L,
        optimizer: &mut O,
        input: &Array<F, IxDyn>,
        target: &Array<F, IxDyn>,
        loss_fn: &dyn Loss<F>,
    ) -> Result<F>
    where
        L: ParamLayer<F> + ?Sized,
        O: Optimizer<F> + OptimizerStep<F> + ?Sized,
    {
        // Forward pass with checkpointing
        let output = self.forward_with_checkpointing(model, input)?;

        // Compute loss
        let loss = loss_fn.forward(&output, target)?;
        let grad_output = loss_fn.backward(&output, target)?;

        // Backward pass with recomputation
        let _grad_input = self.backward_with_recomputation(model, input, &grad_output)?;

        // Update parameters
        optimizer.step(model)?;

        // Clear checkpoints for next iteration
        self.clear_checkpoints();

        Ok(loss)
    }
}

/// Layer information for checkpointing analysis
#[derive(Debug, Clone)]
struct LayerInfo {
    layer_index: usize,
    memory_usage: usize,
    computation_cost: f64,
    layer_type: LayerType,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
}

/// Layer types for analysis
#[derive(Debug, Clone, PartialEq)]
enum LayerType {
    Dense,
    Convolutional,
    Recurrent,
    Attention,
    Normalization,
    Activation,
    Unknown,
}

/// Memory usage tracker
#[derive(Debug)]
struct MemoryTracker {
    total_allocated: usize,
    checkpoint_memory: usize,
    peak_usage: usize,
    allocations: Vec<usize>,
}

impl MemoryTracker {
    fn new() -> Self {
        Self {
            total_allocated: 0,
            checkpoint_memory: 0,
            peak_usage: 0,
            allocations: Vec::new(),
        }
    }

    fn add_allocation(&mut self, size: usize) {
        self.total_allocated += size;
        self.allocations.push(size);
        self.peak_usage = self
            .peak_usage
            .max(self.total_allocated + self.checkpoint_memory);
    }

    fn add_checkpoint(&mut self, size: usize) {
        self.checkpoint_memory += size;
        self.peak_usage = self
            .peak_usage
            .max(self.total_allocated + self.checkpoint_memory);
    }

    fn clear_checkpoints(&mut self) {
        self.checkpoint_memory = 0;
    }

    fn reset(&mut self) {
        self.total_allocated = 0;
        self.checkpoint_memory = 0;
        self.allocations.clear();
    }

    fn get_stats(&self) -> MemoryStats {
        MemoryStats {
            total_allocated: self.total_allocated,
            checkpoint_memory: self.checkpoint_memory,
            peak_usage: self.peak_usage,
            current_usage: self.total_allocated + self.checkpoint_memory,
        }
    }
}

/// Forward pass state
#[derive(Debug)]
struct ForwardState<F: Float + Debug + Clone> {
    intermediate_activations: Vec<Array<F, IxDyn>>,
    current_layer: usize,
}

impl<F: Float + Debug + Clone> ForwardState<F> {
    fn new() -> Self {
        Self {
            intermediate_activations: Vec::new(),
            current_layer: 0,
        }
    }

    fn reset(&mut self) {
        self.intermediate_activations.clear();
        self.current_layer = 0;
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Total memory allocated for activations
    pub total_allocated: usize,
    /// Memory used for checkpoints
    pub checkpoint_memory: usize,
    /// Peak memory usage
    pub peak_usage: usize,
    /// Current memory usage
    pub current_usage: usize,
}

/// Checkpointing statistics
#[derive(Debug, Clone, Default)]
pub struct CheckpointingStats {
    /// Number of forward passes
    pub forward_passes: u64,
    /// Number of backward passes
    pub backward_passes: u64,
    /// Number of recomputations performed
    pub recomputations: u64,
    /// Number of checkpoints saved
    pub checkpoints_saved: u64,
    /// Total time spent in forward passes
    pub total_forward_time: std::time::Duration,
    /// Total time spent in backward passes
    pub total_backward_time: std::time::Duration,
}

impl CheckpointingStats {
    /// Calculate memory savings ratio
    pub fn memory_savings_ratio(&self, total_memory: usize, checkpoint_memory: usize) -> f32 {
        if total_memory > 0 {
            1.0 - (checkpoint_memory as f32 / total_memory as f32)
        } else {
            0.0
        }
    }

    /// Calculate average recomputations per backward pass
    pub fn avg_recomputations_per_backward(&self) -> f64 {
        if self.backward_passes > 0 {
            self.recomputations as f64 / self.backward_passes as f64
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_gradient_checkpointing_config() {
        let config = GradientCheckpointingConfig::default();
        assert_eq!(config.num_segments, 4);
        assert_eq!(config.strategy, CheckpointingStrategy::Uniform);
        assert!(config.adaptive);
    }

    #[test]
    fn test_memory_tracker() {
        let mut tracker = MemoryTracker::new();

        tracker.add_allocation(1000);
        tracker.add_checkpoint(500);

        let stats = tracker.get_stats();
        assert_eq!(stats.total_allocated, 1000);
        assert_eq!(stats.checkpoint_memory, 500);
        assert_eq!(stats.current_usage, 1500);
        assert_eq!(stats.peak_usage, 1500);
    }

    #[test]
    fn test_checkpoint_segment() {
        let segment = CheckpointSegment {
            id: 0,
            start_layer: 0,
            end_layer: 5,
            memory_usage: 1024,
            computation_cost: 100.0,
            checkpoint: true,
        };

        assert_eq!(segment.id, 0);
        assert_eq!(segment.memory_usage, 1024);
        assert!(segment.checkpoint);
    }

    #[test]
    fn test_checkpointing_stats() {
        let mut stats = CheckpointingStats::default();
        stats.forward_passes = 10;
        stats.backward_passes = 10;
        stats.recomputations = 20;

        assert_eq!(stats.avg_recomputations_per_backward(), 2.0);
        assert_eq!(stats.memory_savings_ratio(1000, 200), 0.8);
    }
}
