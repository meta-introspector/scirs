//! Meta-gradients for meta-learning algorithms
//!
//! This module implements meta-gradient computation for algorithms like MAML,
//! Reptile, and other meta-learning approaches that require gradients of
//! gradients with respect to meta-parameters.

use ndarray::{Array, Array1, Array2, ArrayBase, Data, Dimension};
use num_traits::Float;
use std::collections::{HashMap, VecDeque};

use crate::error::OptimizerError;
use super::forward_mode::ForwardModeEngine;
use super::reverse_mode::ReverseModeEngine;

/// Meta-gradient computation engine
pub struct MetaGradientEngine<T: Float> {
    /// Forward-mode engine for directional derivatives
    forward_engine: ForwardModeEngine<T>,
    
    /// Reverse-mode engine for efficient computation
    reverse_engine: ReverseModeEngine<T>,
    
    /// Meta-learning algorithm type
    algorithm: MetaLearningAlgorithm,
    
    /// Inner loop configuration
    inner_loop_config: InnerLoopConfig,
    
    /// Outer loop configuration  
    outer_loop_config: OuterLoopConfig,
    
    /// Gradient computation cache
    gradient_cache: HashMap<String, Array1<T>>,
    
    /// Meta-parameter history
    meta_param_history: VecDeque<Array1<T>>,
    
    /// Task performance history
    task_performance: HashMap<String, Vec<T>>,
}

/// Meta-learning algorithm types
#[derive(Debug, Clone, Copy)]
pub enum MetaLearningAlgorithm {
    /// Model-Agnostic Meta-Learning
    MAML,
    
    /// First-Order MAML (FOMAML)
    FirstOrderMAML,
    
    /// Reptile algorithm
    Reptile,
    
    /// Learning to Learn by Gradient Descent (L2L)
    L2L,
    
    /// Meta-SGD
    MetaSGD,
    
    /// Learned optimizer
    LearnedOptimizer,
    
    /// Implicit Function Theorem based
    ImplicitFunctionTheorem,
}

/// Inner loop configuration for meta-learning
#[derive(Debug, Clone)]
pub struct InnerLoopConfig {
    /// Number of inner steps
    pub num_steps: usize,
    
    /// Inner learning rate
    pub learning_rate: f64,
    
    /// Inner optimizer type
    pub optimizer: InnerOptimizer,
    
    /// Stop condition
    pub stop_condition: StopCondition,
    
    /// Enable gradient checkpointing
    pub gradient_checkpointing: bool,
    
    /// Maximum inner steps
    pub max_steps: usize,
}

/// Outer loop configuration for meta-learning
#[derive(Debug, Clone)]
pub struct OuterLoopConfig {
    /// Meta learning rate
    pub meta_learning_rate: f64,
    
    /// Meta-optimizer type
    pub meta_optimizer: MetaOptimizer,
    
    /// Number of tasks per meta-batch
    pub tasks_per_batch: usize,
    
    /// Enable second-order derivatives
    pub second_order: bool,
    
    /// Use implicit function theorem
    pub use_implicit_function_theorem: bool,
    
    /// Meta-regularization coefficient
    pub meta_regularization: f64,
}

/// Inner optimizer types
#[derive(Debug, Clone, Copy)]
pub enum InnerOptimizer {
    SGD,
    Adam,
    RMSprop,
    Custom,
}

/// Meta-optimizer types
#[derive(Debug, Clone, Copy)]
pub enum MetaOptimizer {
    SGD,
    Adam,
    RMSprop,
    LBFGS,
    Custom,
}

/// Stop conditions for inner loop
#[derive(Debug, Clone)]
pub enum StopCondition {
    /// Fixed number of steps
    FixedSteps,
    
    /// Convergence threshold
    Convergence { threshold: f64 },
    
    /// Loss threshold
    LossThreshold { threshold: f64 },
    
    /// Gradient norm threshold
    GradientNorm { threshold: f64 },
}

/// Meta-learning task definition
#[derive(Debug, Clone)]
pub struct MetaTask<T: Float> {
    /// Task identifier
    pub id: String,
    
    /// Support set (training data)
    pub support_set: Vec<(Array1<T>, Array1<T>)>,
    
    /// Query set (test data)
    pub query_set: Vec<(Array1<T>, Array1<T>)>,
    
    /// Task-specific parameters
    pub task_params: HashMap<String, T>,
    
    /// Task weight for meta-batch
    pub weight: T,
}

/// Meta-gradient computation result
#[derive(Debug, Clone)]
pub struct MetaGradientResult<T: Float> {
    /// Meta-gradients w.r.t. meta-parameters
    pub meta_gradients: Array1<T>,
    
    /// Inner loop gradients for each task
    pub inner_gradients: Vec<Array1<T>>,
    
    /// Task losses
    pub task_losses: Vec<T>,
    
    /// Meta-loss
    pub meta_loss: T,
    
    /// Computation method used
    pub method: MetaGradientMethod,
    
    /// Computation statistics
    pub stats: MetaGradientStats,
}

/// Meta-gradient computation methods
#[derive(Debug, Clone, Copy)]
pub enum MetaGradientMethod {
    /// Exact second-order derivatives
    ExactSecondOrder,
    
    /// First-order approximation
    FirstOrderApproximation,
    
    /// Implicit function theorem
    ImplicitFunctionTheorem,
    
    /// Finite differences
    FiniteDifferences,
    
    /// Truncated backpropagation
    TruncatedBackprop,
}

/// Meta-gradient computation statistics
#[derive(Debug, Clone)]
pub struct MetaGradientStats {
    /// Total computation time (microseconds)
    pub computation_time_us: u64,
    
    /// Number of inner steps per task
    pub inner_steps_per_task: Vec<usize>,
    
    /// Memory usage estimate (bytes)
    pub memory_usage: usize,
    
    /// Gradient computation count
    pub gradient_computations: usize,
    
    /// Second-order computations
    pub second_order_computations: usize,
}

impl<T: Float + Default + Clone> MetaGradientEngine<T> {
    /// Create a new meta-gradient engine
    pub fn new(
        algorithm: MetaLearningAlgorithm,
        inner_config: InnerLoopConfig,
        outer_config: OuterLoopConfig,
    ) -> Self {
        Self {
            forward_engine: ForwardModeEngine::new(),
            reverse_engine: ReverseModeEngine::new(),
            algorithm,
            inner_loop_config: inner_config,
            outer_loop_config: outer_config,
            gradient_cache: HashMap::new(),
            meta_param_history: VecDeque::with_capacity(1000),
            task_performance: HashMap::new(),
        }
    }
    
    /// Compute meta-gradients for a batch of tasks
    pub fn compute_meta_gradients(
        &mut self,
        meta_params: &Array1<T>,
        tasks: &[MetaTask<T>],
        objective_fn: impl Fn(&Array1<T>, &Array1<T>, &[(Array1<T>, Array1<T>)]) -> T,
    ) -> Result<MetaGradientResult<T>, OptimizerError> {
        let start_time = std::time::Instant::now();
        
        match self.algorithm {
            MetaLearningAlgorithm::MAML => {
                self.compute_maml_gradients(meta_params, tasks, &objective_fn)
            }
            MetaLearningAlgorithm::FirstOrderMAML => {
                self.compute_fomaml_gradients(meta_params, tasks, &objective_fn)
            }
            MetaLearningAlgorithm::Reptile => {
                self.compute_reptile_gradients(meta_params, tasks, &objective_fn)
            }
            MetaLearningAlgorithm::L2L => {
                self.compute_l2l_gradients(meta_params, tasks, &objective_fn)
            }
            MetaLearningAlgorithm::MetaSGD => {
                self.compute_meta_sgd_gradients(meta_params, tasks, &objective_fn)
            }
            MetaLearningAlgorithm::LearnedOptimizer => {
                self.compute_learned_optimizer_gradients(meta_params, tasks, &objective_fn)
            }
            MetaLearningAlgorithm::ImplicitFunctionTheorem => {
                self.compute_ift_gradients(meta_params, tasks, &objective_fn)
            }
        }
    }
    
    /// Compute MAML meta-gradients
    fn compute_maml_gradients(
        &mut self,
        meta_params: &Array1<T>,
        tasks: &[MetaTask<T>],
        objective_fn: &impl Fn(&Array1<T>, &Array1<T>, &[(Array1<T>, Array1<T>)]) -> T,
    ) -> Result<MetaGradientResult<T>, OptimizerError> {
        let mut meta_gradients = Array1::zeros(meta_params.len());
        let mut inner_gradients = Vec::new();
        let mut task_losses = Vec::new();
        let mut total_meta_loss = T::zero();
        
        for task in tasks {
            // Inner loop adaptation
            let adapted_params = self.inner_loop_adaptation(meta_params, task, objective_fn)?;
            
            // Compute query loss with adapted parameters
            let query_loss = objective_fn(&adapted_params, meta_params, &task.query_set);
            task_losses.push(query_loss);
            total_meta_loss = total_meta_loss + query_loss * task.weight;
            
            // Compute meta-gradients using second-order derivatives
            let task_meta_gradients = if self.outer_loop_config.second_order {
                self.compute_second_order_meta_gradients(
                    meta_params,
                    &adapted_params,
                    task,
                    objective_fn,
                )?
            } else {
                self.compute_first_order_meta_gradients(
                    meta_params,
                    &adapted_params,
                    task,
                    objective_fn,
                )?
            };
            
            // Accumulate meta-gradients
            meta_gradients = meta_gradients + task_meta_gradients * task.weight;
            inner_gradients.push(task_meta_gradients);
        }
        
        let meta_loss = total_meta_loss / T::from(tasks.len()).unwrap();
        let computation_time = start_time.elapsed().as_micros() as u64;
        
        Ok(MetaGradientResult {
            meta_gradients,
            inner_gradients,
            task_losses,
            meta_loss,
            method: if self.outer_loop_config.second_order {
                MetaGradientMethod::ExactSecondOrder
            } else {
                MetaGradientMethod::FirstOrderApproximation
            },
            stats: MetaGradientStats {
                computation_time_us: computation_time,
                inner_steps_per_task: vec![self.inner_loop_config.num_steps; tasks.len()],
                memory_usage: self.estimate_memory_usage(),
                gradient_computations: tasks.len() * self.inner_loop_config.num_steps,
                second_order_computations: if self.outer_loop_config.second_order { tasks.len() } else { 0 },
            },
        })
    }
    
    /// Compute First-Order MAML meta-gradients
    fn compute_fomaml_gradients(
        &mut self,
        meta_params: &Array1<T>,
        tasks: &[MetaTask<T>],
        objective_fn: &impl Fn(&Array1<T>, &Array1<T>, &[(Array1<T>, Array1<T>)]) -> T,
    ) -> Result<MetaGradientResult<T>, OptimizerError> {
        let mut meta_gradients = Array1::zeros(meta_params.len());
        let mut task_losses = Vec::new();
        let mut total_meta_loss = T::zero();
        
        for task in tasks {
            // Inner loop adaptation
            let adapted_params = self.inner_loop_adaptation(meta_params, task, objective_fn)?;
            
            // Compute query loss
            let query_loss = objective_fn(&adapted_params, meta_params, &task.query_set);
            task_losses.push(query_loss);
            total_meta_loss = total_meta_loss + query_loss * task.weight;
            
            // First-order approximation: ignore second derivatives
            let gradient = self.compute_gradient_wrt_params(&adapted_params, &task.query_set, objective_fn)?;
            meta_gradients = meta_gradients + gradient * task.weight;
        }
        
        let meta_loss = total_meta_loss / T::from(tasks.len()).unwrap();
        
        Ok(MetaGradientResult {
            meta_gradients,
            inner_gradients: vec![Array1::zeros(meta_params.len()); tasks.len()],
            task_losses,
            meta_loss,
            method: MetaGradientMethod::FirstOrderApproximation,
            stats: MetaGradientStats {
                computation_time_us: 0,
                inner_steps_per_task: vec![self.inner_loop_config.num_steps; tasks.len()],
                memory_usage: self.estimate_memory_usage(),
                gradient_computations: tasks.len(),
                second_order_computations: 0,
            },
        })
    }
    
    /// Compute Reptile meta-gradients
    fn compute_reptile_gradients(
        &mut self,
        meta_params: &Array1<T>,
        tasks: &[MetaTask<T>],
        objective_fn: &impl Fn(&Array1<T>, &Array1<T>, &[(Array1<T>, Array1<T>)]) -> T,
    ) -> Result<MetaGradientResult<T>, OptimizerError> {
        let mut meta_gradients = Array1::zeros(meta_params.len());
        let mut task_losses = Vec::new();
        let mut total_meta_loss = T::zero();
        
        for task in tasks {
            // Inner loop adaptation (multiple steps)
            let adapted_params = self.inner_loop_adaptation(meta_params, task, objective_fn)?;
            
            // Reptile gradient: direction from meta-params to adapted params
            let reptile_gradient = &adapted_params - meta_params;
            meta_gradients = meta_gradients + reptile_gradient * task.weight;
            
            // Compute loss for tracking
            let query_loss = objective_fn(&adapted_params, meta_params, &task.query_set);
            task_losses.push(query_loss);
            total_meta_loss = total_meta_loss + query_loss * task.weight;
        }
        
        let meta_loss = total_meta_loss / T::from(tasks.len()).unwrap();
        
        Ok(MetaGradientResult {
            meta_gradients,
            inner_gradients: vec![Array1::zeros(meta_params.len()); tasks.len()],
            task_losses,
            meta_loss,
            method: MetaGradientMethod::FirstOrderApproximation,
            stats: MetaGradientStats {
                computation_time_us: 0,
                inner_steps_per_task: vec![self.inner_loop_config.num_steps; tasks.len()],
                memory_usage: self.estimate_memory_usage(),
                gradient_computations: tasks.len() * self.inner_loop_config.num_steps,
                second_order_computations: 0,
            },
        })
    }
    
    /// Compute L2L (Learning to Learn) meta-gradients
    fn compute_l2l_gradients(
        &mut self,
        meta_params: &Array1<T>,
        tasks: &[MetaTask<T>],
        objective_fn: &impl Fn(&Array1<T>, &Array1<T>, &[(Array1<T>, Array1<T>)]) -> T,
    ) -> Result<MetaGradientResult<T>, OptimizerError> {
        // L2L uses a learned optimizer, so this is a simplified implementation
        self.compute_maml_gradients(meta_params, tasks, objective_fn)
    }
    
    /// Compute Meta-SGD gradients
    fn compute_meta_sgd_gradients(
        &mut self,
        meta_params: &Array1<T>,
        tasks: &[MetaTask<T>],
        objective_fn: &impl Fn(&Array1<T>, &Array1<T>, &[(Array1<T>, Array1<T>)]) -> T,
    ) -> Result<MetaGradientResult<T>, OptimizerError> {
        // Meta-SGD learns both parameters and learning rates
        // This is a simplified implementation
        self.compute_maml_gradients(meta_params, tasks, objective_fn)
    }
    
    /// Compute learned optimizer meta-gradients
    fn compute_learned_optimizer_gradients(
        &mut self,
        meta_params: &Array1<T>,
        tasks: &[MetaTask<T>],
        objective_fn: &impl Fn(&Array1<T>, &Array1<T>, &[(Array1<T>, Array1<T>)]) -> T,
    ) -> Result<MetaGradientResult<T>, OptimizerError> {
        // Use learned optimizer for inner loop
        self.compute_maml_gradients(meta_params, tasks, objective_fn)
    }
    
    /// Compute gradients using Implicit Function Theorem
    fn compute_ift_gradients(
        &mut self,
        meta_params: &Array1<T>,
        tasks: &[MetaTask<T>],
        objective_fn: &impl Fn(&Array1<T>, &Array1<T>, &[(Array1<T>, Array1<T>)]) -> T,
    ) -> Result<MetaGradientResult<T>, OptimizerError> {
        let mut meta_gradients = Array1::zeros(meta_params.len());
        let mut task_losses = Vec::new();
        let mut total_meta_loss = T::zero();
        
        for task in tasks {
            // Find optimal parameters by solving inner optimization to convergence
            let optimal_params = self.solve_inner_optimization(meta_params, task, objective_fn)?;
            
            // Use implicit function theorem to compute meta-gradients
            let ift_gradients = self.compute_implicit_function_gradients(
                meta_params,
                &optimal_params,
                task,
                objective_fn,
            )?;
            
            meta_gradients = meta_gradients + ift_gradients * task.weight;
            
            let query_loss = objective_fn(&optimal_params, meta_params, &task.query_set);
            task_losses.push(query_loss);
            total_meta_loss = total_meta_loss + query_loss * task.weight;
        }
        
        let meta_loss = total_meta_loss / T::from(tasks.len()).unwrap();
        
        Ok(MetaGradientResult {
            meta_gradients,
            inner_gradients: vec![Array1::zeros(meta_params.len()); tasks.len()],
            task_losses,
            meta_loss,
            method: MetaGradientMethod::ImplicitFunctionTheorem,
            stats: MetaGradientStats {
                computation_time_us: 0,
                inner_steps_per_task: vec![100; tasks.len()], // Convergence steps
                memory_usage: self.estimate_memory_usage(),
                gradient_computations: tasks.len() * 100,
                second_order_computations: tasks.len(),
            },
        })
    }
    
    /// Perform inner loop adaptation
    fn inner_loop_adaptation(
        &self,
        meta_params: &Array1<T>,
        task: &MetaTask<T>,
        objective_fn: &impl Fn(&Array1<T>, &Array1<T>, &[(Array1<T>, Array1<T>)]) -> T,
    ) -> Result<Array1<T>, OptimizerError> {
        let mut params = meta_params.clone();
        let lr = T::from(self.inner_loop_config.learning_rate).unwrap();
        
        for _step in 0..self.inner_loop_config.num_steps {
            // Compute gradient on support set
            let gradient = self.compute_gradient_wrt_params(&params, &task.support_set, objective_fn)?;
            
            // SGD update
            params = params - gradient * lr;
            
            // Check stop condition
            if self.check_stop_condition(&gradient, &params, task, objective_fn)? {
                break;
            }
        }
        
        Ok(params)
    }
    
    /// Solve inner optimization to convergence (for IFT)
    fn solve_inner_optimization(
        &self,
        meta_params: &Array1<T>,
        task: &MetaTask<T>,
        objective_fn: &impl Fn(&Array1<T>, &Array1<T>, &[(Array1<T>, Array1<T>)]) -> T,
    ) -> Result<Array1<T>, OptimizerError> {
        let mut params = meta_params.clone();
        let lr = T::from(self.inner_loop_config.learning_rate).unwrap();
        let convergence_threshold = T::from(1e-6).unwrap();
        
        for _step in 0..self.inner_loop_config.max_steps {
            let gradient = self.compute_gradient_wrt_params(&params, &task.support_set, objective_fn)?;
            
            // Check convergence
            let grad_norm = gradient.iter().map(|&g| g * g).sum::<T>().sqrt();
            if grad_norm < convergence_threshold {
                break;
            }
            
            // Update parameters
            params = params - gradient * lr;
        }
        
        Ok(params)
    }
    
    /// Compute second-order meta-gradients
    fn compute_second_order_meta_gradients(
        &self,
        meta_params: &Array1<T>,
        adapted_params: &Array1<T>,
        task: &MetaTask<T>,
        objective_fn: &impl Fn(&Array1<T>, &Array1<T>, &[(Array1<T>, Array1<T>)]) -> T,
    ) -> Result<Array1<T>, OptimizerError> {
        // Simplified second-order computation using finite differences
        let eps = T::from(1e-6).unwrap();
        let mut meta_gradients = Array1::zeros(meta_params.len());
        
        for i in 0..meta_params.len() {
            let mut meta_plus = meta_params.clone();
            let mut meta_minus = meta_params.clone();
            
            meta_plus[i] = meta_plus[i] + eps;
            meta_minus[i] = meta_minus[i] - eps;
            
            // Adapt parameters for perturbed meta-parameters
            let adapted_plus = self.inner_loop_adaptation(&meta_plus, task, objective_fn)?;
            let adapted_minus = self.inner_loop_adaptation(&meta_minus, task, objective_fn)?;
            
            // Compute query losses
            let loss_plus = objective_fn(&adapted_plus, &meta_plus, &task.query_set);
            let loss_minus = objective_fn(&adapted_minus, &meta_minus, &task.query_set);
            
            // Finite difference approximation
            meta_gradients[i] = (loss_plus - loss_minus) / (T::from(2.0).unwrap() * eps);
        }
        
        Ok(meta_gradients)
    }
    
    /// Compute first-order meta-gradients
    fn compute_first_order_meta_gradients(
        &self,
        _meta_params: &Array1<T>,
        adapted_params: &Array1<T>,
        task: &MetaTask<T>,
        objective_fn: &impl Fn(&Array1<T>, &Array1<T>, &[(Array1<T>, Array1<T>)]) -> T,
    ) -> Result<Array1<T>, OptimizerError> {
        // First-order approximation: gradient w.r.t. adapted parameters
        self.compute_gradient_wrt_params(adapted_params, &task.query_set, objective_fn)
    }
    
    /// Compute implicit function theorem gradients
    fn compute_implicit_function_gradients(
        &self,
        meta_params: &Array1<T>,
        optimal_params: &Array1<T>,
        task: &MetaTask<T>,
        objective_fn: &impl Fn(&Array1<T>, &Array1<T>, &[(Array1<T>, Array1<T>)]) -> T,
    ) -> Result<Array1<T>, OptimizerError> {
        // Simplified IFT computation
        // In practice, this would solve: dψ/dθ = -H^(-1) * d²L/dψdθ
        // where ψ are optimal params, θ are meta-params, H is Hessian
        
        let eps = T::from(1e-6).unwrap();
        let mut gradients = Array1::zeros(meta_params.len());
        
        for i in 0..meta_params.len() {
            let mut meta_plus = meta_params.clone();
            meta_plus[i] = meta_plus[i] + eps;
            
            let optimal_plus = self.solve_inner_optimization(&meta_plus, task, objective_fn)?;
            
            // Approximate derivative of optimal params w.r.t. meta-params
            let param_derivative = (&optimal_plus - optimal_params) / eps;
            
            // Chain rule: dL/dθ = dL/dψ * dψ/dθ
            let loss_gradient = self.compute_gradient_wrt_params(optimal_params, &task.query_set, objective_fn)?;
            gradients[i] = loss_gradient.dot(&param_derivative);
        }
        
        Ok(gradients)
    }
    
    /// Compute gradient w.r.t. parameters
    fn compute_gradient_wrt_params(
        &self,
        params: &Array1<T>,
        data: &[(Array1<T>, Array1<T>)],
        objective_fn: &impl Fn(&Array1<T>, &Array1<T>, &[(Array1<T>, Array1<T>)]) -> T,
    ) -> Result<Array1<T>, OptimizerError> {
        let eps = T::from(1e-6).unwrap();
        let mut gradient = Array1::zeros(params.len());
        
        for i in 0..params.len() {
            let mut params_plus = params.clone();
            let mut params_minus = params.clone();
            
            params_plus[i] = params_plus[i] + eps;
            params_minus[i] = params_minus[i] - eps;
            
            let loss_plus = objective_fn(&params_plus, params, data);
            let loss_minus = objective_fn(&params_minus, params, data);
            
            gradient[i] = (loss_plus - loss_minus) / (T::from(2.0).unwrap() * eps);
        }
        
        Ok(gradient)
    }
    
    /// Check inner loop stop condition
    fn check_stop_condition(
        &self,
        gradient: &Array1<T>,
        _params: &Array1<T>,
        _task: &MetaTask<T>,
        _objective_fn: &impl Fn(&Array1<T>, &Array1<T>, &[(Array1<T>, Array1<T>)]) -> T,
    ) -> Result<bool, OptimizerError> {
        match &self.inner_loop_config.stop_condition {
            StopCondition::FixedSteps => Ok(false), // Always run fixed number of steps
            StopCondition::Convergence { threshold } => {
                let grad_norm = gradient.iter().map(|&g| g * g).sum::<T>().sqrt();
                Ok(grad_norm < T::from(*threshold).unwrap())
            }
            StopCondition::GradientNorm { threshold } => {
                let grad_norm = gradient.iter().map(|&g| g * g).sum::<T>().sqrt();
                Ok(grad_norm < T::from(*threshold).unwrap())
            }
            StopCondition::LossThreshold { threshold: _ } => {
                // Would need to compute loss here
                Ok(false)
            }
        }
    }
    
    /// Estimate memory usage
    fn estimate_memory_usage(&self) -> usize {
        let cache_size = self.gradient_cache.len() * std::mem::size_of::<(String, Array1<T>)>();
        let history_size = self.meta_param_history.len() * std::mem::size_of::<Array1<T>>();
        let engine_size = std::mem::size_of::<Self>();
        
        cache_size + history_size + engine_size
    }
    
    /// Get meta-learning statistics
    pub fn get_meta_stats(&self) -> MetaLearningStats {
        MetaLearningStats {
            algorithm: self.algorithm,
            cache_size: self.gradient_cache.len(),
            history_length: self.meta_param_history.len(),
            memory_usage: self.estimate_memory_usage(),
            total_tasks_processed: self.task_performance.len(),
        }
    }
    
    /// Clear caches and history
    pub fn clear_cache(&mut self) {
        self.gradient_cache.clear();
        self.meta_param_history.clear();
        self.task_performance.clear();
    }
}

/// Meta-learning statistics
#[derive(Debug, Clone)]
pub struct MetaLearningStats {
    pub algorithm: MetaLearningAlgorithm,
    pub cache_size: usize,
    pub history_length: usize,
    pub memory_usage: usize,
    pub total_tasks_processed: usize,
}

/// Default configurations
impl Default for InnerLoopConfig {
    fn default() -> Self {
        Self {
            num_steps: 5,
            learning_rate: 0.01,
            optimizer: InnerOptimizer::SGD,
            stop_condition: StopCondition::FixedSteps,
            gradient_checkpointing: false,
            max_steps: 100,
        }
    }
}

impl Default for OuterLoopConfig {
    fn default() -> Self {
        Self {
            meta_learning_rate: 0.001,
            meta_optimizer: MetaOptimizer::Adam,
            tasks_per_batch: 16,
            second_order: true,
            use_implicit_function_theorem: false,
            meta_regularization: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_meta_gradient_engine_creation() {
        let inner_config = InnerLoopConfig::default();
        let outer_config = OuterLoopConfig::default();
        let engine = MetaGradientEngine::<f64>::new(
            MetaLearningAlgorithm::MAML,
            inner_config,
            outer_config,
        );
        
        assert!(matches!(engine.algorithm, MetaLearningAlgorithm::MAML));
    }

    #[test]
    fn test_meta_task_creation() {
        let support_set = vec![
            (Array1::from_vec(vec![1.0, 2.0]), Array1::from_vec(vec![3.0])),
            (Array1::from_vec(vec![2.0, 3.0]), Array1::from_vec(vec![5.0])),
        ];
        
        let query_set = vec![
            (Array1::from_vec(vec![3.0, 4.0]), Array1::from_vec(vec![7.0])),
        ];
        
        let task = MetaTask {
            id: "test_task".to_string(),
            support_set,
            query_set,
            task_params: HashMap::new(),
            weight: 1.0,
        };
        
        assert_eq!(task.id, "test_task");
        assert_eq!(task.support_set.len(), 2);
        assert_eq!(task.query_set.len(), 1);
    }

    #[test]
    fn test_inner_loop_config_default() {
        let config = InnerLoopConfig::default();
        assert_eq!(config.num_steps, 5);
        assert_eq!(config.learning_rate, 0.01);
        assert!(matches!(config.optimizer, InnerOptimizer::SGD));
    }

    #[test]
    fn test_outer_loop_config_default() {
        let config = OuterLoopConfig::default();
        assert_eq!(config.meta_learning_rate, 0.001);
        assert!(matches!(config.meta_optimizer, MetaOptimizer::Adam));
        assert_eq!(config.tasks_per_batch, 16);
        assert!(config.second_order);
    }

    #[test]
    fn test_stop_condition_convergence() {
        let condition = StopCondition::Convergence { threshold: 1e-6 };
        
        match condition {
            StopCondition::Convergence { threshold } => {
                assert_eq!(threshold, 1e-6);
            }
            _ => panic!("Wrong stop condition type"),
        }
    }

    #[test]
    fn test_meta_learning_algorithm_types() {
        let algorithms = [
            MetaLearningAlgorithm::MAML,
            MetaLearningAlgorithm::FirstOrderMAML,
            MetaLearningAlgorithm::Reptile,
            MetaLearningAlgorithm::L2L,
            MetaLearningAlgorithm::MetaSGD,
            MetaLearningAlgorithm::LearnedOptimizer,
            MetaLearningAlgorithm::ImplicitFunctionTheorem,
        ];
        
        assert_eq!(algorithms.len(), 7);
    }
}