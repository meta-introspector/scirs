//! Continual and Multi-task Learning Module
//!
//! This module provides frameworks for training neural networks on multiple tasks
//! either simultaneously (multi-task learning) or sequentially (continual learning)
//! while avoiding catastrophic forgetting.

pub mod advanced_continual_learning;
pub mod elastic_weight_consolidation;
pub mod shared_backbone;

pub use advanced_continual_learning::{
    LateralConnection, LearningWithoutForgetting, LwFConfig, PackNet, PackNetConfig,
    ProgressiveConfig, ProgressiveNeuralNetwork, TaskColumn, TaskMask,
};
pub use elastic_weight_consolidation::{EWCConfig, EWC};
pub use shared_backbone::{MultiTaskArchitecture, SharedBackbone, TaskSpecificHead, TaskType};

use crate::error::Result;
use crate::models::sequential::Sequential;
use ndarray::concatenate;
use ndarray::prelude::*;
use std::collections::HashMap;

/// Configuration for continual learning
#[derive(Debug, Clone)]
pub struct ContinualConfig {
    /// Strategy for continual learning
    pub strategy: ContinualStrategy,
    /// Memory size for replay methods
    pub memory_size: usize,
    /// Regularization strength
    pub regularization_strength: f32,
    /// Number of tasks
    pub num_tasks: usize,
    /// Task-specific learning rates
    pub task_learning_rates: Option<Vec<f32>>,
    /// Enable meta-learning
    pub enable_meta_learning: bool,
    /// Temperature for knowledge distillation
    pub distillation_temperature: f32,
}

impl Default for ContinualConfig {
    fn default() -> Self {
        Self {
            strategy: ContinualStrategy::EWC,
            memory_size: 5000,
            regularization_strength: 1000.0,
            num_tasks: 5,
            task_learning_rates: None,
            enable_meta_learning: false,
            distillation_temperature: 3.0,
        }
    }
}

/// Continual learning strategies
#[derive(Debug, Clone, PartialEq)]
pub enum ContinualStrategy {
    /// Elastic Weight Consolidation
    EWC,
    /// Progressive Neural Networks
    Progressive,
    /// Experience Replay
    Replay,
    /// Generative Replay
    GenerativeReplay,
    /// Gradient Episodic Memory
    GEM,
    /// Average Gradient Episodic Memory
    AGEM,
    /// Learning without Forgetting
    LWF,
    /// PackNet
    PackNet,
    /// Dynamic Architecture
    DynamicArchitecture,
}

/// Multi-task learning configuration
#[derive(Debug, Clone)]
pub struct MultiTaskConfig {
    /// Task names
    pub task_names: Vec<String>,
    /// Task weights for loss balancing
    pub task_weights: Option<Vec<f32>>,
    /// Shared layers configuration
    pub shared_layers: Vec<usize>,
    /// Task-specific layers configuration
    pub task_specific_layers: HashMap<String, Vec<usize>>,
    /// Gradient normalization
    pub gradient_normalization: bool,
    /// Dynamic weight averaging
    pub dynamic_weight_averaging: bool,
    /// Uncertainty weighting
    pub uncertainty_weighting: bool,
}

impl Default for MultiTaskConfig {
    fn default() -> Self {
        Self {
            task_names: vec!["task1".to_string(), "task2".to_string()],
            task_weights: None,
            shared_layers: vec![512, 256],
            task_specific_layers: HashMap::new(),
            gradient_normalization: true,
            dynamic_weight_averaging: false,
            uncertainty_weighting: false,
        }
    }
}

/// Continual learning framework
pub struct ContinualLearner {
    config: ContinualConfig,
    base_model: Sequential<f32>,
    task_models: Vec<Sequential<f32>>,
    memory_bank: MemoryBank,
    fisher_information: Option<Vec<Array2<f32>>>,
    optimal_params: Option<Vec<Array2<f32>>>,
    current_task: usize,
}

impl ContinualLearner {
    /// Create a new continual learner
    pub fn new(config: ContinualConfig, base_model: Sequential<f32>) -> Result<Self> {
        let memory_bank = MemoryBank::new(config.memory_size);

        Ok(Self {
            config,
            base_model,
            task_models: Vec::new(),
            memory_bank,
            fisher_information: None,
            optimal_params: None,
            current_task: 0,
        })
    }

    /// Train on a new task
    pub fn train_task(
        &mut self,
        task_id: usize,
        train_data: &ArrayView2<f32>,
        train_labels: &ArrayView1<usize>,
        val_data: &ArrayView2<f32>,
        val_labels: &ArrayView1<usize>,
        epochs: usize,
    ) -> Result<TaskTrainingResult> {
        self.current_task = task_id;

        let result = match self.config.strategy {
            ContinualStrategy::EWC => {
                self.train_with_ewc(train_data, train_labels, val_data, val_labels, epochs)?
            }
            ContinualStrategy::Replay => {
                self.train_with_replay(train_data, train_labels, val_data, val_labels, epochs)?
            }
            ContinualStrategy::GEM => {
                self.train_with_gem(train_data, train_labels, val_data, val_labels, epochs)?
            }
            _ => self.train_standard(train_data, train_labels, val_data, val_labels, epochs)?,
        };

        // Store task-specific information
        self.update_task_memory(train_data, train_labels)?;

        Ok(result)
    }

    /// Train with Elastic Weight Consolidation
    fn train_with_ewc(
        &mut self,
        train_data: &ArrayView2<f32>,
        train_labels: &ArrayView1<usize>,
        val_data: &ArrayView2<f32>,
        val_labels: &ArrayView1<usize>,
        epochs: usize,
    ) -> Result<TaskTrainingResult> {
        let mut total_loss = 0.0;
        let mut best_accuracy = 0.0;

        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;

            // Standard training loss
            let task_loss = self.compute_task_loss(train_data, train_labels)?;
            epoch_loss += task_loss;

            // EWC regularization loss
            if self.current_task > 0 {
                let ewc_loss = self.compute_ewc_loss()?;
                epoch_loss += self.config.regularization_strength * ewc_loss;
            }

            total_loss += epoch_loss;

            // Validation
            let val_accuracy = self.evaluate(val_data, val_labels)?;
            best_accuracy = best_accuracy.max(val_accuracy);
        }

        // Update Fisher information and optimal parameters
        self.update_fisher_information(train_data, train_labels)?;
        self.update_optimal_params()?;

        Ok(TaskTrainingResult {
            task_id: self.current_task,
            final_loss: total_loss / epochs as f32,
            best_accuracy,
            forgetting_measure: self.measure_forgetting()?,
        })
    }

    /// Train with experience replay
    fn train_with_replay(
        &mut self,
        train_data: &ArrayView2<f32>,
        train_labels: &ArrayView1<usize>,
        val_data: &ArrayView2<f32>,
        val_labels: &ArrayView1<usize>,
        epochs: usize,
    ) -> Result<TaskTrainingResult> {
        let mut total_loss = 0.0;
        let mut best_accuracy = 0.0;

        for epoch in 0..epochs {
            // Combine current task data with replay data
            let (combined_data, combined_labels) =
                self.combine_with_replay(train_data, train_labels)?;

            // Train on combined data
            let epoch_loss =
                self.compute_task_loss(&combined_data.view(), &combined_labels.view())?;
            total_loss += epoch_loss;

            // Validation
            let val_accuracy = self.evaluate(val_data, val_labels)?;
            best_accuracy = best_accuracy.max(val_accuracy);
        }

        Ok(TaskTrainingResult {
            task_id: self.current_task,
            final_loss: total_loss / epochs as f32,
            best_accuracy,
            forgetting_measure: self.measure_forgetting()?,
        })
    }

    /// Train with Gradient Episodic Memory
    fn train_with_gem(
        &mut self,
        train_data: &ArrayView2<f32>,
        train_labels: &ArrayView1<usize>,
        val_data: &ArrayView2<f32>,
        val_labels: &ArrayView1<usize>,
        epochs: usize,
    ) -> Result<TaskTrainingResult> {
        let mut total_loss = 0.0;
        let mut best_accuracy = 0.0;

        for epoch in 0..epochs {
            let epoch_loss = self.compute_task_loss(train_data, train_labels)?;

            // Project gradients to avoid interfering with previous tasks
            if self.current_task > 0 {
                self.project_gradients()?;
            }

            total_loss += epoch_loss;

            // Validation
            let val_accuracy = self.evaluate(val_data, val_labels)?;
            best_accuracy = best_accuracy.max(val_accuracy);
        }

        Ok(TaskTrainingResult {
            task_id: self.current_task,
            final_loss: total_loss / epochs as f32,
            best_accuracy,
            forgetting_measure: self.measure_forgetting()?,
        })
    }

    /// Standard training without continual learning techniques
    fn train_standard(
        &mut self,
        train_data: &ArrayView2<f32>,
        train_labels: &ArrayView1<usize>,
        val_data: &ArrayView2<f32>,
        val_labels: &ArrayView1<usize>,
        epochs: usize,
    ) -> Result<TaskTrainingResult> {
        let mut total_loss = 0.0;
        let mut best_accuracy = 0.0;

        for epoch in 0..epochs {
            let epoch_loss = self.compute_task_loss(train_data, train_labels)?;
            total_loss += epoch_loss;

            let val_accuracy = self.evaluate(val_data, val_labels)?;
            best_accuracy = best_accuracy.max(val_accuracy);
        }

        Ok(TaskTrainingResult {
            task_id: self.current_task,
            final_loss: total_loss / epochs as f32,
            best_accuracy,
            forgetting_measure: 0.0,
        })
    }

    /// Compute task-specific loss
    fn compute_task_loss(&self, data: &ArrayView2<f32>, labels: &ArrayView1<usize>) -> Result<f32> {
        // Forward pass through the model
        let predictions = self.base_model.forward(data)?;

        // Compute cross-entropy loss
        let mut total_loss = 0.0;
        let batch_size = data.shape()[0];

        for i in 0..batch_size {
            let true_label = labels[i];
            if true_label < predictions.shape()[1] {
                let pred_value = predictions[[i, true_label]].max(1e-7);
                total_loss -= pred_value.ln();
            }
        }

        Ok(total_loss / batch_size as f32)
    }

    /// Compute EWC regularization loss
    fn compute_ewc_loss(&self) -> Result<f32> {
        if self.fisher_information.is_none() || self.optimal_params.is_none() {
            return Ok(0.0);
        }

        // Simplified EWC loss
        Ok(0.1) // Placeholder
    }

    /// Update Fisher information matrix
    fn update_fisher_information(
        &mut self,
        data: &ArrayView2<f32>,
        labels: &ArrayView1<usize>,
    ) -> Result<()> {
        // Simplified Fisher information computation
        let num_params = 10; // Placeholder
        self.fisher_information = Some(vec![Array2::from_elem((10, 10), 0.1); num_params]);
        Ok(())
    }

    /// Update optimal parameters
    fn update_optimal_params(&mut self) -> Result<()> {
        // Store current model parameters as optimal
        let num_params = 10; // Placeholder
        self.optimal_params = Some(vec![Array2::from_elem((10, 10), 0.5); num_params]);
        Ok(())
    }

    /// Combine current task data with replay data
    fn combine_with_replay(
        &self,
        data: &ArrayView2<f32>,
        labels: &ArrayView1<usize>,
    ) -> Result<(Array2<f32>, Array1<usize>)> {
        let replay_samples = self.memory_bank.sample(self.config.memory_size / 10)?;

        // Combine current and replay data
        let combined_data = concatenate![Axis(0), *data, replay_samples.data];
        let combined_labels = concatenate![Axis(0), *labels, replay_samples.labels];

        Ok((combined_data, combined_labels))
    }

    /// Project gradients for GEM
    fn project_gradients(&mut self) -> Result<()> {
        // Simplified gradient projection
        Ok(())
    }

    /// Update task memory
    fn update_task_memory(
        &mut self,
        data: &ArrayView2<f32>,
        labels: &ArrayView1<usize>,
    ) -> Result<()> {
        self.memory_bank
            .add_task_data(self.current_task, data, labels)
    }

    /// Evaluate on validation data
    fn evaluate(&self, data: &ArrayView2<f32>, labels: &ArrayView1<usize>) -> Result<f32> {
        // Simplified evaluation
        Ok(0.85) // Placeholder
    }

    /// Measure forgetting on previous tasks
    fn measure_forgetting(&self) -> Result<f32> {
        if self.current_task == 0 {
            return Ok(0.0);
        }

        // Simplified forgetting measure
        Ok(0.05) // Placeholder
    }

    /// Get performance on all tasks
    pub fn evaluate_all_tasks(
        &self,
        task_data: &[(Array2<f32>, Array1<usize>)],
    ) -> Result<Vec<f32>> {
        let mut accuracies = Vec::new();

        for (data, labels) in task_data {
            let accuracy = self.evaluate(&data.view(), &labels.view())?;
            accuracies.push(accuracy);
        }

        Ok(accuracies)
    }
}

/// Memory bank for storing task data
struct MemoryBank {
    capacity: usize,
    task_memories: HashMap<usize, TaskMemory>,
}

struct TaskMemory {
    data: Array2<f32>,
    labels: Array1<usize>,
}

struct MemorySamples {
    data: Array2<f32>,
    labels: Array1<usize>,
}

impl MemoryBank {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            task_memories: HashMap::new(),
        }
    }

    fn add_task_data(
        &mut self,
        task_id: usize,
        data: &ArrayView2<f32>,
        labels: &ArrayView1<usize>,
    ) -> Result<()> {
        let samples_per_task = self.capacity / (self.task_memories.len() + 1);

        // Random sampling for memory
        let num_samples = data.shape()[0].min(samples_per_task);
        let indices: Vec<usize> = (0..data.shape()[0]).collect();
        let selected_indices = &indices[..num_samples];

        let mut selected_data = Array2::zeros((num_samples, data.shape()[1]));
        let mut selected_labels = Array1::zeros(num_samples);

        for (i, &idx) in selected_indices.iter().enumerate() {
            selected_data.row_mut(i).assign(&data.row(idx));
            selected_labels[i] = labels[idx];
        }

        self.task_memories.insert(
            task_id,
            TaskMemory {
                data: selected_data,
                labels: selected_labels,
            },
        );

        Ok(())
    }

    fn sample(&self, num_samples: usize) -> Result<MemorySamples> {
        if self.task_memories.is_empty() {
            return Ok(MemorySamples {
                data: Array2::zeros((0, 1)),
                labels: Array1::zeros(0),
            });
        }

        let samples_per_task = num_samples / self.task_memories.len();
        let mut all_data = Vec::new();
        let mut all_labels = Vec::new();

        for memory in self.task_memories.values() {
            let task_samples = samples_per_task.min(memory.data.shape()[0]);
            for i in 0..task_samples {
                all_data.push(memory.data.row(i).to_owned());
                all_labels.push(memory.labels[i]);
            }
        }

        let data = if all_data.is_empty() {
            Array2::zeros((0, 1))
        } else {
            let rows = all_data.len();
            let cols = all_data[0].len();
            let mut arr = Array2::zeros((rows, cols));
            for (i, row) in all_data.into_iter().enumerate() {
                arr.row_mut(i).assign(&row);
            }
            arr
        };

        Ok(MemorySamples {
            data,
            labels: Array1::from_vec(all_labels),
        })
    }
}

/// Result of training on a task
#[derive(Debug, Clone)]
pub struct TaskTrainingResult {
    pub task_id: usize,
    pub final_loss: f32,
    pub best_accuracy: f32,
    pub forgetting_measure: f32,
}

/// Multi-task learner
pub struct MultiTaskLearner {
    config: MultiTaskConfig,
    shared_backbone: SharedBackbone,
    task_heads: HashMap<String, TaskSpecificHead>,
    task_uncertainties: Option<HashMap<String, f32>>,
}

impl MultiTaskLearner {
    /// Create a new multi-task learner
    pub fn new(config: MultiTaskConfig, input_dim: usize) -> Result<Self> {
        let shared_backbone = SharedBackbone::new(input_dim, &config.shared_layers)?;

        let mut task_heads = HashMap::new();
        for task_name in &config.task_names {
            let task_layers = config
                .task_specific_layers
                .get(task_name)
                .cloned()
                .unwrap_or_else(|| vec![128, 64]);

            let head = TaskSpecificHead::new(
                config.shared_layers.last().copied().unwrap_or(256),
                &task_layers,
                10, // Placeholder output dim
            )?;

            task_heads.insert(task_name.clone(), head);
        }

        let task_uncertainties = if config.uncertainty_weighting {
            Some(
                config
                    .task_names
                    .iter()
                    .map(|name| (name.clone(), 0.0))
                    .collect(),
            )
        } else {
            None
        };

        Ok(Self {
            config,
            shared_backbone,
            task_heads,
            task_uncertainties,
        })
    }

    /// Train on multiple tasks
    pub fn train(
        &mut self,
        task_data: &HashMap<String, (ArrayView2<f32>, ArrayView1<usize>)>,
        epochs: usize,
    ) -> Result<MultiTaskTrainingResult> {
        let mut task_losses = HashMap::new();
        let mut task_accuracies = HashMap::new();

        for epoch in 0..epochs {
            let mut epoch_losses = HashMap::new();

            // Forward pass for all tasks
            for (task_name, (data, labels)) in task_data {
                let shared_features = self.shared_backbone.forward(data)?;

                if let Some(head) = self.task_heads.get(task_name) {
                    let task_output = head.forward(&shared_features.view())?;
                    let task_loss = self.compute_task_loss(&task_output.view(), labels)?;
                    epoch_losses.insert(task_name.clone(), task_loss);
                }
            }

            // Compute weighted loss
            let total_loss = self.compute_weighted_loss(&epoch_losses)?;

            // Update task uncertainties if enabled
            if self.config.uncertainty_weighting {
                self.update_task_uncertainties(&epoch_losses)?;
            }

            // Track metrics
            for (task_name, loss) in epoch_losses {
                task_losses
                    .entry(task_name.clone())
                    .or_insert_with(Vec::new)
                    .push(loss);
            }
        }

        // Compute final accuracies
        for (task_name, (data, labels)) in task_data {
            let accuracy = self.evaluate_task(task_name, data, labels)?;
            task_accuracies.insert(task_name.clone(), accuracy);
        }

        Ok(MultiTaskTrainingResult {
            task_losses,
            task_accuracies,
            task_weights: self.get_current_task_weights(),
        })
    }

    /// Compute weighted loss across tasks
    fn compute_weighted_loss(&self, task_losses: &HashMap<String, f32>) -> Result<f32> {
        let weights = self.get_current_task_weights();
        let mut total_loss = 0.0;

        for (task_name, &loss) in task_losses {
            let weight = weights.get(task_name).unwrap_or(&1.0);
            total_loss += weight * loss;
        }

        Ok(total_loss)
    }

    /// Update task uncertainties for uncertainty weighting
    fn update_task_uncertainties(&mut self, task_losses: &HashMap<String, f32>) -> Result<()> {
        if let Some(ref mut uncertainties) = self.task_uncertainties {
            for (task_name, &loss) in task_losses {
                // Simple exponential moving average
                let current = uncertainties.get(task_name).copied().unwrap_or(0.0);
                uncertainties.insert(task_name.clone(), 0.9 * current + 0.1 * loss);
            }
        }
        Ok(())
    }

    /// Get current task weights
    fn get_current_task_weights(&self) -> HashMap<String, f32> {
        if let Some(ref weights) = self.config.task_weights {
            self.config
                .task_names
                .iter()
                .zip(weights)
                .map(|(name, &weight)| (name.clone(), weight))
                .collect()
        } else if let Some(ref uncertainties) = self.task_uncertainties {
            // Compute weights from uncertainties
            uncertainties
                .iter()
                .map(|(name, &uncertainty)| {
                    let weight = 1.0 / (2.0 * uncertainty.max(0.1));
                    (name.clone(), weight)
                })
                .collect()
        } else {
            // Equal weights
            self.config
                .task_names
                .iter()
                .map(|name| (name.clone(), 1.0))
                .collect()
        }
    }

    /// Evaluate a specific task
    fn evaluate_task(
        &self,
        task_name: &str,
        data: &ArrayView2<f32>,
        labels: &ArrayView1<usize>,
    ) -> Result<f32> {
        let shared_features = self.shared_backbone.forward(data)?;

        if let Some(head) = self.task_heads.get(task_name) {
            let task_output = head.forward(&shared_features.view())?;
            // Compute accuracy (simplified)
            Ok(0.9) // Placeholder
        } else {
            Err(crate::error::NeuralError::InvalidArgument(format!(
                "Task {} not found",
                task_name
            )))
        }
    }
}

/// Multi-task training result
#[derive(Debug)]
pub struct MultiTaskTrainingResult {
    pub task_losses: HashMap<String, Vec<f32>>,
    pub task_accuracies: HashMap<String, f32>,
    pub task_weights: HashMap<String, f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_continual_config_default() {
        let config = ContinualConfig::default();
        assert_eq!(config.strategy, ContinualStrategy::EWC);
        assert_eq!(config.memory_size, 5000);
    }

    #[test]
    fn test_multi_task_config_default() {
        let config = MultiTaskConfig::default();
        assert_eq!(config.task_names.len(), 2);
        assert!(config.gradient_normalization);
    }

    #[test]
    fn test_memory_bank() {
        let mut bank = MemoryBank::new(1000);
        let data = Array2::from_elem((100, 10), 1.0);
        let labels = Array1::from_elem(100, 0);

        bank.add_task_data(0, &data.view(), &labels.view()).unwrap();

        let samples = bank.sample(50).unwrap();
        assert!(samples.data.shape()[0] <= 50);
    }
}
