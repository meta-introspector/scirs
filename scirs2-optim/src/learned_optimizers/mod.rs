//! Learned optimizers using neural networks
//!
//! This module implements learned optimizers that use neural networks (particularly LSTMs)
//! to learn optimization strategies, enabling meta-learning for automated optimizer design.

use ndarray::{
    s, Array, Array1, Array2, Array3, ArrayBase, Data, DataMut, Dimension, Ix1, Ix2, Ix3,
};
use num_traits::Float;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

pub mod lstm_optimizer;
pub mod meta_learning;
pub mod neural_architecture_search;
pub mod transformer_optimizer;

use crate::error::{OptimError, OptimizerError};
use crate::optimizers::Optimizer;

/// Configuration for learned optimizers
#[derive(Debug, Clone)]
pub struct LearnedOptimizerConfig {
    /// Type of neural optimizer
    pub optimizer_type: NeuralOptimizerType,

    /// Hidden state size for LSTM-based optimizers
    pub hidden_size: usize,

    /// Number of layers in the neural network
    pub num_layers: usize,

    /// Input feature dimensions
    pub input_features: usize,

    /// Output dimensions (typically parameter updates)
    pub output_features: usize,

    /// Learning rate for meta-learning
    pub meta_learning_rate: f64,

    /// Window size for gradient history
    pub gradient_history_size: usize,

    /// Enable attention mechanism
    pub use_attention: bool,

    /// Attention head count
    pub attention_heads: usize,

    /// Enable recurrent connections
    pub use_recurrent: bool,

    /// Dropout rate for regularization
    pub dropout_rate: f64,

    /// Enable learned learning rate schedules
    pub learned_lr_schedule: bool,

    /// Meta-optimization strategy
    pub meta_strategy: MetaOptimizationStrategy,

    /// Pre-training dataset size
    pub pretraining_dataset_size: usize,

    /// Enable transfer learning
    pub enable_transfer_learning: bool,
}

impl Default for LearnedOptimizerConfig {
    fn default() -> Self {
        Self {
            optimizer_type: NeuralOptimizerType::LSTM,
            hidden_size: 256,
            num_layers: 2,
            input_features: 64,
            output_features: 1,
            meta_learning_rate: 0.001,
            gradient_history_size: 10,
            use_attention: true,
            attention_heads: 8,
            use_recurrent: true,
            dropout_rate: 0.1,
            learned_lr_schedule: true,
            meta_strategy: MetaOptimizationStrategy::MAML,
            pretraining_dataset_size: 10000,
            enable_transfer_learning: true,
        }
    }
}

/// Types of neural optimizers
#[derive(Debug, Clone, Copy)]
pub enum NeuralOptimizerType {
    /// LSTM-based optimizer
    LSTM,
    /// Transformer-based optimizer
    Transformer,
    /// Convolutional optimizer
    Convolutional,
    /// Graph Neural Network optimizer
    GraphNN,
    /// Hybrid architecture
    Hybrid,
}

/// Meta-optimization strategies
#[derive(Debug, Clone, Copy)]
pub enum MetaOptimizationStrategy {
    /// Model-Agnostic Meta-Learning
    MAML,
    /// Reptile meta-learning
    Reptile,
    /// Learning to Learn by Gradient Descent
    L2L,
    /// Meta-SGD
    MetaSGD,
    /// Learned optimizer (from scratch)
    LearnedOptimizer,
}

/// LSTM-based neural optimizer
pub struct LSTMOptimizer<A: Float> {
    /// Configuration
    config: LearnedOptimizerConfig,

    /// LSTM cell state
    cell_state: LSTMState<A>,

    /// Learned parameters
    parameters: LSTMParameters<A>,

    /// Gradient history for context
    gradient_history: VecDeque<Array1<A>>,

    /// Parameter history
    parameter_history: VecDeque<Array1<A>>,

    /// Loss history
    loss_history: VecDeque<A>,

    /// Meta-optimizer for learning the optimizer
    meta_optimizer: Box<dyn Optimizer<A> + Send + Sync>,

    /// Training state
    training_state: MetaTrainingState<A>,

    /// Performance metrics
    metrics: LearnedOptimizerMetrics,

    /// Step count
    step_count: usize,
}

/// LSTM state for neural optimizer
#[derive(Debug, Clone)]
struct LSTMState<A: Float> {
    /// Hidden states for each layer
    hidden_states: Vec<Array1<A>>,

    /// Cell states for each layer
    cell_states: Vec<Array1<A>>,

    /// Attention weights (if using attention)
    attention_weights: Option<Array2<A>>,

    /// Context vector
    context_vector: Option<Array1<A>>,
}

/// LSTM parameters (weights and biases)
#[derive(Debug, Clone)]
struct LSTMParameters<A: Float> {
    /// Input-to-hidden weights
    weight_ih: Vec<Array2<A>>,

    /// Hidden-to-hidden weights
    weight_hh: Vec<Array2<A>>,

    /// Input-to-hidden biases
    bias_ih: Vec<Array1<A>>,

    /// Hidden-to-hidden biases
    bias_hh: Vec<Array1<A>>,

    /// Output projection weights
    output_weights: Array2<A>,

    /// Output biases
    output_bias: Array1<A>,

    /// Attention parameters (if using attention)
    attention_params: Option<AttentionParameters<A>>,

    /// Learned learning rate parameters
    lr_params: Option<LearningRateParameters<A>>,
}

/// Attention mechanism parameters
#[derive(Debug, Clone)]
struct AttentionParameters<A: Float> {
    /// Query projection weights
    query_weights: Array2<A>,

    /// Key projection weights
    key_weights: Array2<A>,

    /// Value projection weights
    value_weights: Array2<A>,

    /// Output projection weights
    output_weights: Array2<A>,

    /// Multi-head attention parameters
    head_weights: Vec<Array2<A>>,
}

/// Learned learning rate parameters
#[derive(Debug, Clone)]
struct LearningRateParameters<A: Float> {
    /// Base learning rate
    base_lr: A,

    /// Adaptive factors
    adaptive_factors: Array1<A>,

    /// Schedule parameters
    schedule_params: Array1<A>,

    /// Decay parameters
    decay_params: Array1<A>,
}

/// Meta-training state
#[derive(Debug, Clone)]
struct MetaTrainingState<A: Float> {
    /// Meta-training step count
    meta_step: usize,

    /// Meta-gradients accumulator
    meta_gradients: HashMap<String, Array1<A>>,

    /// Task performance history
    task_performance: VecDeque<TaskPerformance<A>>,

    /// Current meta-batch
    current_meta_batch: Vec<MetaTask<A>>,

    /// Meta-validation metrics
    meta_validation: MetaValidationMetrics<A>,

    /// Transfer learning state
    transfer_state: Option<TransferLearningState<A>>,
}

/// Task performance tracking
#[derive(Debug, Clone)]
struct TaskPerformance<A: Float> {
    /// Task identifier
    task_id: String,

    /// Initial loss
    initial_loss: A,

    /// Final loss
    final_loss: A,

    /// Convergence steps
    convergence_steps: usize,

    /// Learning curve
    learning_curve: Vec<A>,

    /// Task metadata
    metadata: HashMap<String, String>,
}

/// Meta-learning task
#[derive(Debug, Clone)]
struct MetaTask<A: Float> {
    /// Task identifier
    id: String,

    /// Training data
    train_data: Vec<(Array1<A>, A)>,

    /// Validation data
    val_data: Vec<(Array1<A>, A)>,

    /// Task-specific parameters
    task_params: HashMap<String, A>,

    /// Expected performance
    target_performance: Option<A>,
}

/// Meta-validation metrics
#[derive(Debug, Clone)]
struct MetaValidationMetrics<A: Float> {
    /// Average task performance
    avg_task_performance: A,

    /// Performance variance
    performance_variance: A,

    /// Generalization error
    generalization_error: A,

    /// Adaptation speed
    adaptation_speed: A,

    /// Task diversity score
    task_diversity: A,
}

/// Transfer learning state
#[derive(Debug, Clone)]
struct TransferLearningState<A: Float> {
    /// Source domain performance
    source_performance: A,

    /// Target domain performance
    target_performance: A,

    /// Transfer efficiency
    transfer_efficiency: A,

    /// Adapted parameters
    adapted_params: HashMap<String, Array1<A>>,

    /// Fine-tuning steps
    finetuning_steps: usize,
}

/// Performance metrics for learned optimizers
#[derive(Debug, Clone)]
pub struct LearnedOptimizerMetrics {
    /// Meta-training loss
    pub meta_training_loss: f64,

    /// Average task convergence speed
    pub avg_convergence_speed: f64,

    /// Generalization performance
    pub generalization_performance: f64,

    /// Parameter efficiency
    pub parameter_efficiency: f64,

    /// Transfer learning success rate
    pub transfer_success_rate: f64,

    /// Computational overhead
    pub computational_overhead: f64,

    /// Memory usage
    pub memory_usage_mb: f64,

    /// Meta-gradient norm
    pub meta_gradient_norm: f64,
}

impl<A> LSTMOptimizer<A>
where
    A: Float + Default + Clone + Send + Sync + std::fmt::Debug,
{
    /// Create a new LSTM optimizer
    pub fn new(
        config: LearnedOptimizerConfig,
        meta_optimizer: Box<dyn Optimizer<A> + Send + Sync>,
    ) -> Result<Self, OptimizerError> {
        let cell_state = LSTMState::new(&config)?;
        let parameters = LSTMParameters::new(&config)?;

        let gradient_history = VecDeque::with_capacity(config.gradient_history_size);
        let parameter_history = VecDeque::with_capacity(config.gradient_history_size);
        let loss_history = VecDeque::with_capacity(1000);

        let training_state = MetaTrainingState::new(&config)?;
        let metrics = LearnedOptimizerMetrics::default();

        Ok(Self {
            config,
            cell_state,
            parameters,
            gradient_history,
            parameter_history,
            loss_history,
            meta_optimizer,
            training_state,
            metrics,
            step_count: 0,
        })
    }

    /// Perform learned optimization step
    pub fn learned_step<S, D>(
        &mut self,
        params: &ArrayBase<S, D>,
        gradients: &ArrayBase<S, D>,
        loss: Option<A>,
    ) -> Result<Array<A, D>, OptimizerError>
    where
        S: Data<Elem = A>,
        D: Dimension + Clone,
    {
        // Convert inputs to 1D for processing
        let flat_params = self.flatten_array(params)?;
        let flat_gradients = self.flatten_array(gradients)?;

        // Update history
        self.update_history(&flat_params, &flat_gradients, loss);

        // Prepare input features for LSTM
        let input_features = self.prepare_input_features(&flat_gradients)?;

        // Forward pass through LSTM
        let update_direction = self.lstm_forward(&input_features)?;

        // Compute learned learning rate
        let learned_lr = self.compute_learned_lr(&flat_gradients)?;

        // Apply update
        let mut flat_updated = flat_params.clone();
        for i in 0..flat_updated.len() {
            flat_updated[i] = flat_updated[i] - learned_lr * update_direction[i];
        }

        // Reshape back to original dimensions
        let updated_params = self.reshape_array(&flat_updated, params.raw_dim())?;

        self.step_count += 1;

        // Update metrics
        self.update_metrics(&flat_gradients, &update_direction, learned_lr);

        Ok(updated_params)
    }

    /// Meta-training step
    pub fn meta_train_step(&mut self, meta_batch: Vec<MetaTask<A>>) -> Result<A, OptimizerError> {
        let mut total_meta_loss = A::zero();
        let batch_size = A::from(meta_batch.len()).unwrap();

        for task in &meta_batch {
            let task_loss = self.train_on_task(task)?;
            total_meta_loss = total_meta_loss + task_loss;
        }

        let avg_meta_loss = total_meta_loss / batch_size;

        // Compute meta-gradients
        let meta_gradients = self.compute_meta_gradients(&meta_batch)?;

        // Update meta-parameters
        self.update_meta_parameters(&meta_gradients)?;

        self.training_state.meta_step += 1;
        self.metrics.meta_training_loss = avg_meta_loss.to_f64().unwrap_or(0.0);

        Ok(avg_meta_loss)
    }

    fn update_history(&mut self, params: &Array1<A>, gradients: &Array1<A>, loss: Option<A>) {
        self.parameter_history.push_back(params.clone());
        self.gradient_history.push_back(gradients.clone());

        if let Some(l) = loss {
            self.loss_history.push_back(l);
        }

        // Maintain window size
        if self.parameter_history.len() > self.config.gradient_history_size {
            self.parameter_history.pop_front();
        }
        if self.gradient_history.len() > self.config.gradient_history_size {
            self.gradient_history.pop_front();
        }
        if self.loss_history.len() > 1000 {
            self.loss_history.pop_front();
        }
    }

    fn prepare_input_features(&self, gradients: &Array1<A>) -> Result<Array1<A>, OptimizerError> {
        let mut features = Vec::new();

        // Current gradient
        features.extend_from_slice(gradients.as_slice().unwrap());

        // Gradient history features
        if let Some(prev_grad) = self.gradient_history.back() {
            // Gradient difference
            let grad_diff: Vec<A> = gradients
                .iter()
                .zip(prev_grad.iter())
                .map(|(&g1, &g2)| g1 - g2)
                .collect();
            features.extend_from_slice(&grad_diff);

            // Gradient ratio
            let grad_ratio: Vec<A> = gradients
                .iter()
                .zip(prev_grad.iter())
                .map(|(&g1, &g2)| {
                    if g2.abs() > A::from(1e-8).unwrap() {
                        g1 / g2
                    } else {
                        A::one()
                    }
                })
                .collect();
            features.extend_from_slice(&grad_ratio);
        }

        // Gradient statistics
        let grad_norm = gradients.iter().map(|&g| g * g).sum::<A>().sqrt();
        let grad_mean = gradients.iter().cloned().sum::<A>() / A::from(gradients.len()).unwrap();
        let grad_std = {
            let variance = gradients
                .iter()
                .map(|&g| (g - grad_mean) * (g - grad_mean))
                .sum::<A>()
                / A::from(gradients.len()).unwrap();
            variance.sqrt()
        };

        features.push(grad_norm);
        features.push(grad_mean);
        features.push(grad_std);

        // Loss history features (if available)
        if self.loss_history.len() >= 2 {
            let current_loss = *self.loss_history.back().unwrap();
            let prev_loss = self.loss_history[self.loss_history.len() - 2];
            let loss_change = current_loss - prev_loss;
            let loss_ratio = if prev_loss.abs() > A::from(1e-8).unwrap() {
                current_loss / prev_loss
            } else {
                A::one()
            };

            features.push(loss_change);
            features.push(loss_ratio);
        }

        // Pad or truncate to input_features size
        features.resize(self.config.input_features, A::zero());

        Ok(Array1::from_vec(features))
    }

    fn lstm_forward(&mut self, input: &Array1<A>) -> Result<Array1<A>, OptimizerError> {
        let mut current_input = input.clone();

        // Forward pass through LSTM layers
        for layer in 0..self.config.num_layers {
            let (hidden_output, cell_output) = self.lstm_cell_forward(
                &current_input,
                &self.cell_state.hidden_states[layer],
                &self.cell_state.cell_states[layer],
                layer,
            )?;

            self.cell_state.hidden_states[layer] = hidden_output.clone();
            self.cell_state.cell_states[layer] = cell_output;
            current_input = hidden_output;
        }

        // Apply attention if enabled
        if self.config.use_attention {
            current_input = self.apply_attention(&current_input)?;
        }

        // Final output projection
        let output =
            self.parameters.output_weights.dot(&current_input) + &self.parameters.output_bias;

        Ok(output)
    }

    fn lstm_cell_forward(
        &self,
        input: &Array1<A>,
        hidden: &Array1<A>,
        cell: &Array1<A>,
        layer: usize,
    ) -> Result<(Array1<A>, Array1<A>), OptimizerError> {
        // LSTM cell computation: i_t, f_t, g_t, o_t = σ(W_ih @ x_t + W_hh @ h_{t-1} + b)
        let ih_linear =
            self.parameters.weight_ih[layer].dot(input) + &self.parameters.bias_ih[layer];
        let hh_linear =
            self.parameters.weight_hh[layer].dot(hidden) + &self.parameters.bias_hh[layer];
        let combined = ih_linear + hh_linear;

        let hidden_size = self.config.hidden_size;

        // Split into gates
        let input_gate = self.sigmoid(&combined.slice(s![0..hidden_size]).to_owned());
        let forget_gate =
            self.sigmoid(&combined.slice(s![hidden_size..2 * hidden_size]).to_owned());
        let cell_gate = self.tanh(
            &combined
                .slice(s![2 * hidden_size..3 * hidden_size])
                .to_owned(),
        );
        let output_gate = self.sigmoid(
            &combined
                .slice(s![3 * hidden_size..4 * hidden_size])
                .to_owned(),
        );

        // Update cell state
        let new_cell = &forget_gate * cell + &input_gate * &cell_gate;

        // Update hidden state
        let new_hidden = &output_gate * &self.tanh(&new_cell);

        Ok((new_hidden, new_cell))
    }

    fn apply_attention(&mut self, hidden: &Array1<A>) -> Result<Array1<A>, OptimizerError> {
        if let Some(ref attention_params) = self.parameters.attention_params {
            // Simplified attention mechanism
            // In practice, this would implement multi-head attention
            let query = attention_params.query_weights.dot(hidden);
            let key = attention_params.key_weights.dot(hidden);
            let value = attention_params.value_weights.dot(hidden);

            // Compute attention weights (simplified)
            let attention_score = query.dot(&key);
            let attention_weight = self.softmax(&Array1::from_vec(vec![attention_score]))[0];

            let attended_output = value * attention_weight;
            self.cell_state.context_vector = Some(attended_output.clone());

            Ok(attended_output)
        } else {
            Ok(hidden.clone())
        }
    }

    fn compute_learned_lr(&self, gradients: &Array1<A>) -> Result<A, OptimizerError> {
        if let Some(ref lr_params) = self.parameters.lr_params {
            // Adaptive learning rate based on gradient statistics
            let grad_norm = gradients.iter().map(|&g| g * g).sum::<A>().sqrt();
            let adaptive_factor =
                lr_params.adaptive_factors[0] / (grad_norm + A::from(1e-8).unwrap());

            let learned_lr = lr_params.base_lr * adaptive_factor;
            Ok(learned_lr)
        } else {
            // Fallback to base learning rate
            Ok(A::from(self.config.meta_learning_rate).unwrap())
        }
    }

    fn train_on_task(&mut self, task: &MetaTask<A>) -> Result<A, OptimizerError> {
        // Simplified task training
        let mut task_loss = A::zero();
        let data_size = A::from(task.train_data.len()).unwrap();

        for (features, target) in &task.train_data {
            // Compute prediction (simplified)
            let prediction = features.iter().sum::<A>() / A::from(features.len()).unwrap();
            let loss = (prediction - *target) * (prediction - *target);
            task_loss = task_loss + loss;
        }

        Ok(task_loss / data_size)
    }

    fn compute_meta_gradients(
        &self,
        _meta_batch: &[MetaTask<A>],
    ) -> Result<HashMap<String, Array1<A>>, OptimizerError> {
        // Simplified meta-gradient computation
        // In practice, this would compute gradients of the meta-objective
        let mut meta_grads = HashMap::new();

        // Placeholder meta-gradients
        meta_grads.insert(
            "weight_ih_0".to_string(),
            Array1::zeros(self.config.hidden_size * 4),
        );
        meta_grads.insert(
            "weight_hh_0".to_string(),
            Array1::zeros(self.config.hidden_size * 4),
        );

        Ok(meta_grads)
    }

    fn update_meta_parameters(
        &mut self,
        _meta_gradients: &HashMap<String, Array1<A>>,
    ) -> Result<(), OptimizerError> {
        // Update meta-parameters using meta-optimizer
        // This is simplified - in practice would update all LSTM parameters
        Ok(())
    }

    fn update_metrics(&mut self, gradients: &Array1<A>, updates: &Array1<A>, lr: A) {
        let grad_norm = gradients.iter().map(|&g| g * g).sum::<A>().sqrt();
        let update_norm = updates.iter().map(|&u| u * u).sum::<A>().sqrt();

        self.metrics.meta_gradient_norm = grad_norm.to_f64().unwrap_or(0.0);
        self.metrics.parameter_efficiency = (update_norm / grad_norm).to_f64().unwrap_or(1.0);

        // Update other metrics based on current performance
        if self.step_count > 0 {
            self.metrics.avg_convergence_speed = 1.0 / self.step_count as f64;
        }
    }

    // Utility functions
    fn flatten_array<S, D>(&self, array: &ArrayBase<S, D>) -> Result<Array1<A>, OptimizerError>
    where
        S: Data<Elem = A>,
        D: Dimension,
    {
        Ok(Array1::from_vec(array.iter().cloned().collect()))
    }

    fn reshape_array<D>(
        &self,
        flat_array: &Array1<A>,
        shape: D,
    ) -> Result<Array<A, D>, OptimizerError>
    where
        D: Dimension + Clone,
    {
        // Simplified reshape - in practice would handle arbitrary dimensions
        Array::from_shape_vec(shape, flat_array.to_vec())
            .map_err(|_| OptimizerError::InvalidConfig("Reshape error".to_string()))
    }

    fn sigmoid(&self, x: &Array1<A>) -> Array1<A> {
        x.mapv(|xi| A::one() / (A::one() + (-xi).exp()))
    }

    fn tanh(&self, x: &Array1<A>) -> Array1<A> {
        x.mapv(|xi| xi.tanh())
    }

    fn softmax(&self, x: &Array1<A>) -> Array1<A> {
        let exp_x = x.mapv(|xi| xi.exp());
        let sum_exp = exp_x.sum();
        exp_x / sum_exp
    }

    /// Get current performance metrics
    pub fn get_metrics(&self) -> &LearnedOptimizerMetrics {
        &self.metrics
    }

    /// Save learned optimizer state
    pub fn save_state(&self) -> LearnedOptimizerState<A> {
        LearnedOptimizerState {
            parameters: self.parameters.clone(),
            cell_state: self.cell_state.clone(),
            training_state: self.training_state.clone(),
            step_count: self.step_count,
        }
    }

    /// Load learned optimizer state
    pub fn load_state(&mut self, state: LearnedOptimizerState<A>) -> Result<(), OptimizerError> {
        self.parameters = state.parameters;
        self.cell_state = state.cell_state;
        self.training_state = state.training_state;
        self.step_count = state.step_count;
        Ok(())
    }

    /// Transfer learning to new domain
    pub fn transfer_to_domain(
        &mut self,
        target_tasks: &[MetaTask<A>],
    ) -> Result<TransferResults<A>, OptimizerError> {
        let initial_performance = self.evaluate_on_tasks(target_tasks)?;

        // Fine-tune on target domain
        let mut transfer_state = TransferLearningState {
            source_performance: self.metrics.generalization_performance,
            target_performance: initial_performance,
            transfer_efficiency: A::zero(),
            adapted_params: HashMap::new(),
            finetuning_steps: 0,
        };

        // Simplified fine-tuning process
        for _ in 0..100 {
            // 100 fine-tuning steps
            let batch: Vec<_> = target_tasks.iter().take(5).cloned().collect();
            self.meta_train_step(batch)?;
            transfer_state.finetuning_steps += 1;
        }

        let final_performance = self.evaluate_on_tasks(target_tasks)?;
        transfer_state.target_performance = final_performance;
        transfer_state.transfer_efficiency = final_performance / initial_performance;

        self.training_state.transfer_state = Some(transfer_state.clone());

        Ok(TransferResults {
            initial_performance,
            final_performance,
            transfer_state,
            adaptation_steps: 100,
        })
    }

    fn evaluate_on_tasks(&self, tasks: &[MetaTask<A>]) -> Result<A, OptimizerError> {
        // Simplified evaluation
        let mut total_performance = A::zero();
        for task in tasks {
            let task_perf = self.evaluate_single_task(task)?;
            total_performance = total_performance + task_perf;
        }
        Ok(total_performance / A::from(tasks.len()).unwrap())
    }

    fn evaluate_single_task(&self, task: &MetaTask<A>) -> Result<A, OptimizerError> {
        // Simplified single task evaluation
        let mut loss = A::zero();
        for (features, target) in &task.val_data {
            let prediction = features.iter().sum::<A>() / A::from(features.len()).unwrap();
            loss = loss + (prediction - *target) * (prediction - *target);
        }
        Ok(loss / A::from(task.val_data.len()).unwrap())
    }
}

/// Saved state for learned optimizers
#[derive(Debug, Clone)]
pub struct LearnedOptimizerState<A: Float> {
    pub parameters: LSTMParameters<A>,
    pub cell_state: LSTMState<A>,
    pub training_state: MetaTrainingState<A>,
    pub step_count: usize,
}

/// Transfer learning results
#[derive(Debug, Clone)]
pub struct TransferResults<A: Float> {
    pub initial_performance: A,
    pub final_performance: A,
    pub transfer_state: TransferLearningState<A>,
    pub adaptation_steps: usize,
}

// Implementation of initialization functions

impl<A: Float + Default + Clone> LSTMState<A> {
    fn new(config: &LearnedOptimizerConfig) -> Result<Self, OptimizerError> {
        let mut hidden_states = Vec::new();
        let mut cell_states = Vec::new();

        for _ in 0..config.num_layers {
            hidden_states.push(Array1::zeros(config.hidden_size));
            cell_states.push(Array1::zeros(config.hidden_size));
        }

        Ok(Self {
            hidden_states,
            cell_states,
            attention_weights: if config.use_attention {
                Some(Array2::zeros((config.attention_heads, config.hidden_size)))
            } else {
                None
            },
            context_vector: None,
        })
    }
}

impl<A: Float + Default + Clone> LSTMParameters<A> {
    fn new(config: &LearnedOptimizerConfig) -> Result<Self, OptimizerError> {
        let mut weight_ih = Vec::new();
        let mut weight_hh = Vec::new();
        let mut bias_ih = Vec::new();
        let mut bias_hh = Vec::new();

        for layer in 0..config.num_layers {
            let input_size = if layer == 0 {
                config.input_features
            } else {
                config.hidden_size
            };
            let hidden_size = config.hidden_size;

            // Xavier initialization
            let fan_in = input_size as f64;
            let fan_out = hidden_size as f64;
            let scale = (6.0 / (fan_in + fan_out)).sqrt();

            weight_ih.push(Self::random_array_2d(4 * hidden_size, input_size, scale));
            weight_hh.push(Self::random_array_2d(4 * hidden_size, hidden_size, scale));
            bias_ih.push(Array1::zeros(4 * hidden_size));
            bias_hh.push(Array1::zeros(4 * hidden_size));
        }

        let output_weights = Self::random_array_2d(config.output_features, config.hidden_size, 0.1);
        let output_bias = Array1::zeros(config.output_features);

        let attention_params = if config.use_attention {
            Some(AttentionParameters::new(config)?)
        } else {
            None
        };

        let lr_params = if config.learned_lr_schedule {
            Some(LearningRateParameters::new(config)?)
        } else {
            None
        };

        Ok(Self {
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            output_weights,
            output_bias,
            attention_params,
            lr_params,
        })
    }

    fn random_array_2d(rows: usize, cols: usize, scale: f64) -> Array2<A> {
        // Simplified random initialization
        Array2::zeros((rows, cols))
            .mapv(|_| A::from(scale * (rand::random::<f64>() - 0.5)).unwrap())
    }
}

impl<A: Float + Default + Clone> AttentionParameters<A> {
    fn new(config: &LearnedOptimizerConfig) -> Result<Self, OptimizerError> {
        let hidden_size = config.hidden_size;
        let scale = 0.1;

        Ok(Self {
            query_weights: LSTMParameters::random_array_2d(hidden_size, hidden_size, scale),
            key_weights: LSTMParameters::random_array_2d(hidden_size, hidden_size, scale),
            value_weights: LSTMParameters::random_array_2d(hidden_size, hidden_size, scale),
            output_weights: LSTMParameters::random_array_2d(hidden_size, hidden_size, scale),
            head_weights: (0..config.attention_heads)
                .map(|_| LSTMParameters::random_array_2d(hidden_size, hidden_size, scale))
                .collect(),
        })
    }
}

impl<A: Float + Default + Clone> LearningRateParameters<A> {
    fn new(config: &LearnedOptimizerConfig) -> Result<Self, OptimizerError> {
        Ok(Self {
            base_lr: A::from(config.meta_learning_rate).unwrap(),
            adaptive_factors: Array1::ones(config.output_features),
            schedule_params: Array1::zeros(4), // Parameters for schedule
            decay_params: Array1::zeros(2),    // Decay parameters
        })
    }
}

impl<A: Float + Default + Clone> MetaTrainingState<A> {
    fn new(_config: &LearnedOptimizerConfig) -> Result<Self, OptimizerError> {
        Ok(Self {
            meta_step: 0,
            meta_gradients: HashMap::new(),
            task_performance: VecDeque::with_capacity(1000),
            current_meta_batch: Vec::new(),
            meta_validation: MetaValidationMetrics {
                avg_task_performance: A::zero(),
                performance_variance: A::zero(),
                generalization_error: A::zero(),
                adaptation_speed: A::zero(),
                task_diversity: A::zero(),
            },
            transfer_state: None,
        })
    }
}

impl Default for LearnedOptimizerMetrics {
    fn default() -> Self {
        Self {
            meta_training_loss: 0.0,
            avg_convergence_speed: 0.0,
            generalization_performance: 0.0,
            parameter_efficiency: 1.0,
            transfer_success_rate: 0.0,
            computational_overhead: 1.0,
            memory_usage_mb: 0.0,
            meta_gradient_norm: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizers::SGD;

    #[test]
    fn test_learned_optimizer_config_default() {
        let config = LearnedOptimizerConfig::default();
        assert_eq!(config.hidden_size, 256);
        assert_eq!(config.num_layers, 2);
        assert!(config.use_attention);
        assert!(matches!(config.optimizer_type, NeuralOptimizerType::LSTM));
    }

    #[test]
    fn test_lstm_state_creation() {
        let config = LearnedOptimizerConfig::default();
        let state = LSTMState::<f64>::new(&config);
        assert!(state.is_ok());

        let state = state.unwrap();
        assert_eq!(state.hidden_states.len(), config.num_layers);
        assert_eq!(state.cell_states.len(), config.num_layers);
        assert!(state.attention_weights.is_some());
    }

    #[test]
    fn test_lstm_parameters_creation() {
        let config = LearnedOptimizerConfig::default();
        let params = LSTMParameters::<f64>::new(&config);
        assert!(params.is_ok());

        let params = params.unwrap();
        assert_eq!(params.weight_ih.len(), config.num_layers);
        assert_eq!(params.weight_hh.len(), config.num_layers);
        assert!(params.attention_params.is_some());
        assert!(params.lr_params.is_some());
    }

    #[test]
    fn test_lstm_optimizer_creation() {
        let config = LearnedOptimizerConfig::default();
        let meta_optimizer = Box::new(SGD::new(0.001));

        let optimizer = LSTMOptimizer::<f64>::new(config, meta_optimizer);
        assert!(optimizer.is_ok());

        let optimizer = optimizer.unwrap();
        assert_eq!(optimizer.step_count, 0);
        assert!(optimizer.gradient_history.is_empty());
    }

    #[test]
    fn test_metrics_default() {
        let metrics = LearnedOptimizerMetrics::default();
        assert_eq!(metrics.meta_training_loss, 0.0);
        assert_eq!(metrics.parameter_efficiency, 1.0);
        assert_eq!(metrics.transfer_success_rate, 0.0);
    }
}
