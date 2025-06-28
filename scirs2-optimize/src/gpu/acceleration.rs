//! High-level GPU acceleration interface for optimization algorithms
//!
//! This module provides convenient interfaces for accelerating various optimization
//! algorithms using GPU computation, hiding the complexity of low-level GPU operations.

use crate::error::{ScirsError, ScirsResult};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::collections::HashMap;
use std::sync::Arc;

use super::{
    cuda_kernels::{
        DifferentialEvolutionKernel, FunctionEvaluationKernel, GradientKernel, ParticleSwarmKernel,
    },
    memory_management::GpuMemoryPool,
    tensor_core_optimization::{AMPManager, TensorCoreOptimizationConfig, TensorCoreOptimizer},
    GpuFunction, GpuOptimizationConfig, GpuOptimizationContext, GpuPrecision,
};
use crate::result::OptimizeResults;

/// GPU acceleration strategies
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AccelerationStrategy {
    /// Use basic GPU parallelization
    Basic,
    /// Use Tensor Cores when available
    TensorCore,
    /// Use mixed precision computation
    MixedPrecision,
    /// Adaptive strategy based on problem characteristics
    Adaptive,
}

/// Configuration for GPU acceleration
#[derive(Debug, Clone)]
pub struct AccelerationConfig {
    /// Strategy to use for acceleration
    pub strategy: AccelerationStrategy,
    /// GPU optimization configuration
    pub gpu_config: GpuOptimizationConfig,
    /// Tensor Core configuration (if using Tensor Cores)
    pub tensor_config: Option<TensorCoreOptimizationConfig>,
    /// Minimum problem size to enable GPU acceleration
    pub min_problem_size: usize,
    /// Maximum batch size for function evaluations
    pub max_batch_size: usize,
    /// Whether to use asynchronous execution
    pub async_execution: bool,
}

impl Default for AccelerationConfig {
    fn default() -> Self {
        Self {
            strategy: AccelerationStrategy::Adaptive,
            gpu_config: GpuOptimizationConfig::default(),
            tensor_config: Some(TensorCoreOptimizationConfig::default()),
            min_problem_size: 100,
            max_batch_size: 10000,
            async_execution: true,
        }
    }
}

/// Main GPU acceleration manager
pub struct AccelerationManager {
    config: AccelerationConfig,
    context: GpuOptimizationContext,
    tensor_optimizer: Option<TensorCoreOptimizer>,
    amp_manager: Option<AMPManager>,
    kernels: KernelCache,
    performance_stats: PerformanceStats,
}

impl AccelerationManager {
    /// Create a new acceleration manager
    pub fn new(config: AccelerationConfig) -> ScirsResult<Self> {
        let context = GpuOptimizationContext::new(config.gpu_config.clone())?;

        let tensor_optimizer = if matches!(
            config.strategy,
            AccelerationStrategy::TensorCore | AccelerationStrategy::MixedPrecision
        ) {
            if let Some(tensor_config) = &config.tensor_config {
                Some(TensorCoreOptimizer::new(
                    Arc::clone(context.context()),
                    tensor_config.clone(),
                )?)
            } else {
                None
            }
        } else {
            None
        };

        let amp_manager = if matches!(config.strategy, AccelerationStrategy::MixedPrecision) {
            Some(AMPManager::new())
        } else {
            None
        };

        let kernels = KernelCache::new(Arc::clone(context.context()))?;

        Ok(Self {
            config,
            context,
            tensor_optimizer,
            amp_manager,
            kernels,
            performance_stats: PerformanceStats::new(),
        })
    }

    /// Determine if GPU acceleration should be used for a given problem
    pub fn should_accelerate(&self, problem_size: usize, batch_size: usize) -> bool {
        if problem_size < self.config.min_problem_size {
            return false;
        }

        match self.config.strategy {
            AccelerationStrategy::Adaptive => {
                // Use heuristics to determine if GPU acceleration is beneficial
                let total_ops = problem_size * batch_size;
                total_ops > 10000 || batch_size > 100
            }
            _ => true,
        }
    }

    /// Select optimal acceleration strategy based on problem characteristics
    pub fn select_strategy(
        &self,
        problem_size: usize,
        has_gradient: bool,
        is_matrix_intensive: bool,
    ) -> AccelerationStrategy {
        match self.config.strategy {
            AccelerationStrategy::Adaptive => {
                if is_matrix_intensive && self.tensor_optimizer.is_some() {
                    if self
                        .context
                        .context()
                        .supports_tensor_cores()
                        .unwrap_or(false)
                    {
                        AccelerationStrategy::TensorCore
                    } else {
                        AccelerationStrategy::MixedPrecision
                    }
                } else if problem_size > 1000 {
                    AccelerationStrategy::Basic
                } else {
                    AccelerationStrategy::Basic
                }
            }
            strategy => strategy,
        }
    }

    /// Accelerate function evaluation batch
    pub fn accelerate_function_batch<F>(
        &mut self,
        function: &F,
        points: &Array2<f64>,
        strategy: Option<AccelerationStrategy>,
    ) -> ScirsResult<Array1<f64>>
    where
        F: GpuFunction,
    {
        let strategy = strategy.unwrap_or(self.config.strategy);
        self.performance_stats.start_timer("function_evaluation");

        let result = match strategy {
            AccelerationStrategy::Basic => self.accelerate_basic_function_batch(function, points),
            AccelerationStrategy::TensorCore => {
                if let Some(ref tensor_optimizer) = self.tensor_optimizer {
                    self.accelerate_tensor_function_batch(function, points, tensor_optimizer)
                } else {
                    self.accelerate_basic_function_batch(function, points)
                }
            }
            AccelerationStrategy::MixedPrecision => {
                self.accelerate_mixed_precision_function_batch(function, points)
            }
            AccelerationStrategy::Adaptive => {
                let selected = self.select_strategy(points.ncols(), false, false);
                self.accelerate_function_batch(function, points, Some(selected))
            }
        };

        self.performance_stats.end_timer("function_evaluation");
        result
    }

    /// Accelerate gradient computation batch
    pub fn accelerate_gradient_batch<F>(
        &mut self,
        function: &F,
        points: &Array2<f64>,
        strategy: Option<AccelerationStrategy>,
    ) -> ScirsResult<Array2<f64>>
    where
        F: GpuFunction,
    {
        let strategy = strategy.unwrap_or(self.config.strategy);
        self.performance_stats.start_timer("gradient_computation");

        let result = match strategy {
            AccelerationStrategy::Basic => self.accelerate_basic_gradient_batch(function, points),
            AccelerationStrategy::TensorCore => {
                if let Some(ref tensor_optimizer) = self.tensor_optimizer {
                    self.accelerate_tensor_gradient_batch(function, points, tensor_optimizer)
                } else {
                    self.accelerate_basic_gradient_batch(function, points)
                }
            }
            AccelerationStrategy::MixedPrecision => {
                self.accelerate_mixed_precision_gradient_batch(function, points)
            }
            AccelerationStrategy::Adaptive => {
                let selected = self.select_strategy(points.ncols(), true, true);
                self.accelerate_gradient_batch(function, points, Some(selected))
            }
        };

        self.performance_stats.end_timer("gradient_computation");
        result
    }

    /// Accelerate differential evolution optimization
    pub fn accelerate_differential_evolution<F>(
        &mut self,
        function: &F,
        bounds: &[(f64, f64)],
        population_size: usize,
        max_iterations: usize,
        f_scale: f64,
        crossover_rate: f64,
    ) -> ScirsResult<OptimizeResults>
    where
        F: GpuFunction,
    {
        self.performance_stats.start_timer("differential_evolution");

        let mut algorithm = super::algorithms::GpuDifferentialEvolution::new(
            self.context.clone(), // This would need to be implemented
            population_size,
            max_iterations,
        )
        .with_f_scale(f_scale)
        .with_crossover_rate(crossover_rate);

        let result = algorithm.optimize(function, bounds);

        self.performance_stats.end_timer("differential_evolution");
        result
    }

    /// Accelerate particle swarm optimization
    pub fn accelerate_particle_swarm<F>(
        &mut self,
        function: &F,
        bounds: &[(f64, f64)],
        swarm_size: usize,
        max_iterations: usize,
        w: f64,
        c1: f64,
        c2: f64,
    ) -> ScirsResult<OptimizeResults>
    where
        F: GpuFunction,
    {
        self.performance_stats.start_timer("particle_swarm");

        let mut algorithm = super::algorithms::GpuParticleSwarm::new(
            self.context.clone(), // This would need to be implemented
            swarm_size,
            max_iterations,
        )
        .with_parameters(w, c1, c2);

        let result = algorithm.optimize(function, bounds);

        self.performance_stats.end_timer("particle_swarm");
        result
    }

    /// Get performance statistics
    pub fn performance_stats(&self) -> &PerformanceStats {
        &self.performance_stats
    }

    /// Generate performance report
    pub fn generate_performance_report(&self) -> String {
        self.performance_stats.generate_report()
    }

    /// Reset performance statistics
    pub fn reset_stats(&mut self) {
        self.performance_stats.reset();
    }

    // Private implementation methods
    fn accelerate_basic_function_batch<F>(
        &self,
        function: &F,
        points: &Array2<f64>,
    ) -> ScirsResult<Array1<f64>>
    where
        F: GpuFunction,
    {
        self.context.evaluate_function_batch(function, points)
    }

    fn accelerate_tensor_function_batch<F>(
        &self,
        function: &F,
        points: &Array2<f64>,
        tensor_optimizer: &TensorCoreOptimizer,
    ) -> ScirsResult<Array1<f64>>
    where
        F: GpuFunction,
    {
        // For tensor core acceleration, we need matrix-intensive operations
        // This would be implemented based on the specific function structure
        self.accelerate_basic_function_batch(function, points)
    }

    fn accelerate_mixed_precision_function_batch<F>(
        &mut self,
        function: &F,
        points: &Array2<f64>,
    ) -> ScirsResult<Array1<f64>>
    where
        F: GpuFunction,
    {
        // Implement mixed precision evaluation
        if let Some(ref mut amp_manager) = self.amp_manager {
            // Check for overflow and adjust loss scale
            let result = self.accelerate_basic_function_batch(function, points)?;

            // Check for overflow (simplified)
            let has_overflow = result.iter().any(|&x| !x.is_finite());
            amp_manager.update(has_overflow);

            Ok(result)
        } else {
            self.accelerate_basic_function_batch(function, points)
        }
    }

    fn accelerate_basic_gradient_batch<F>(
        &self,
        function: &F,
        points: &Array2<f64>,
    ) -> ScirsResult<Array2<f64>>
    where
        F: GpuFunction,
    {
        self.context.evaluate_gradient_batch(function, points)
    }

    fn accelerate_tensor_gradient_batch<F>(
        &self,
        function: &F,
        points: &Array2<f64>,
        tensor_optimizer: &TensorCoreOptimizer,
    ) -> ScirsResult<Array2<f64>>
    where
        F: GpuFunction,
    {
        // Use Tensor Cores for gradient computation when applicable
        self.accelerate_basic_gradient_batch(function, points)
    }

    fn accelerate_mixed_precision_gradient_batch<F>(
        &mut self,
        function: &F,
        points: &Array2<f64>,
    ) -> ScirsResult<Array2<f64>>
    where
        F: GpuFunction,
    {
        // Implement mixed precision gradient computation
        if let Some(ref mut amp_manager) = self.amp_manager {
            let result = self.accelerate_basic_gradient_batch(function, points)?;

            // Check for overflow in gradients
            let has_overflow = result.iter().any(|&x| !x.is_finite());
            amp_manager.update(has_overflow);

            Ok(result)
        } else {
            self.accelerate_basic_gradient_batch(function, points)
        }
    }
}

/// Cache for compiled GPU kernels
struct KernelCache {
    function_kernel: FunctionEvaluationKernel,
    gradient_kernel: GradientKernel,
    de_kernel: DifferentialEvolutionKernel,
    pso_kernel: ParticleSwarmKernel,
}

impl KernelCache {
    fn new(context: Arc<scirs2_core::gpu::GpuContext>) -> ScirsResult<Self> {
        Ok(Self {
            function_kernel: FunctionEvaluationKernel::new(Arc::clone(&context))?,
            gradient_kernel: GradientKernel::new(Arc::clone(&context))?,
            de_kernel: DifferentialEvolutionKernel::new(Arc::clone(&context))?,
            pso_kernel: ParticleSwarmKernel::new(context)?,
        })
    }
}

/// Performance statistics tracking
#[derive(Debug, Clone)]
pub struct PerformanceStats {
    timers: HashMap<String, f64>,
    counters: HashMap<String, u64>,
    start_times: HashMap<String, std::time::Instant>,
}

impl PerformanceStats {
    pub fn new() -> Self {
        Self {
            timers: HashMap::new(),
            counters: HashMap::new(),
            start_times: HashMap::new(),
        }
    }

    fn start_timer(&mut self, name: &str) {
        self.start_times
            .insert(name.to_string(), std::time::Instant::now());
    }

    fn end_timer(&mut self, name: &str) {
        if let Some(start_time) = self.start_times.remove(name) {
            let elapsed = start_time.elapsed().as_secs_f64();
            *self.timers.entry(name.to_string()).or_insert(0.0) += elapsed;
        }
    }

    fn increment_counter(&mut self, name: &str) {
        *self.counters.entry(name.to_string()).or_insert(0) += 1;
    }

    /// Get timer value
    pub fn get_timer(&self, name: &str) -> Option<f64> {
        self.timers.get(name).copied()
    }

    /// Get counter value
    pub fn get_counter(&self, name: &str) -> Option<u64> {
        self.counters.get(name).copied()
    }

    /// Get all timers
    pub fn timers(&self) -> &HashMap<String, f64> {
        &self.timers
    }

    /// Get all counters
    pub fn counters(&self) -> &HashMap<String, u64> {
        &self.counters
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        self.timers.clear();
        self.counters.clear();
        self.start_times.clear();
    }

    /// Generate performance report
    pub fn generate_report(&self) -> String {
        let mut report = String::from("GPU Acceleration Performance Report\n");
        report.push_str("=====================================\n\n");

        if !self.timers.is_empty() {
            report.push_str("Timers:\n");
            for (name, time) in &self.timers {
                report.push_str(&format!("  {}: {:.6}s\n", name, time));
            }
            report.push('\n');
        }

        if !self.counters.is_empty() {
            report.push_str("Counters:\n");
            for (name, count) in &self.counters {
                report.push_str(&format!("  {}: {}\n", name, count));
            }
            report.push('\n');
        }

        // Calculate throughput metrics
        if let (Some(func_time), Some(func_count)) = (
            self.timers.get("function_evaluation"),
            self.counters.get("function_evaluations"),
        ) {
            let throughput = *func_count as f64 / func_time;
            report.push_str(&format!(
                "Function Evaluation Throughput: {:.2} evals/s\n",
                throughput
            ));
        }

        report
    }
}

/// Utility functions for GPU acceleration
pub mod utils {
    use super::*;

    /// Estimate optimal batch size for GPU evaluation
    pub fn estimate_batch_size(
        problem_dims: usize,
        available_memory: usize,
        precision: GpuPrecision,
    ) -> usize {
        let element_size = match precision {
            GpuPrecision::F32 => 4,
            GpuPrecision::F64 => 8,
            GpuPrecision::Mixed => 6,
        };

        let memory_per_point = problem_dims * element_size * 4; // Input, output, gradient, temp
        let batch_size = available_memory / memory_per_point / 2; // Use half available memory

        batch_size.max(1).min(10000)
    }

    /// Check if GPU acceleration is beneficial
    pub fn is_gpu_beneficial(
        problem_size: usize,
        batch_size: usize,
        cpu_time_per_eval: f64,
        gpu_overhead: f64,
    ) -> bool {
        let total_cpu_time = batch_size as f64 * cpu_time_per_eval;
        let estimated_gpu_time = gpu_overhead + (batch_size as f64 * cpu_time_per_eval * 0.1);

        estimated_gpu_time < total_cpu_time
    }

    /// Create acceleration configuration for specific problem type
    pub fn create_config_for_problem(
        problem_type: ProblemType,
        problem_size: usize,
    ) -> AccelerationConfig {
        match problem_type {
            ProblemType::GlobalOptimization => AccelerationConfig {
                strategy: AccelerationStrategy::Basic,
                max_batch_size: 10000,
                ..Default::default()
            },
            ProblemType::LeastSquares => AccelerationConfig {
                strategy: AccelerationStrategy::TensorCore,
                max_batch_size: 5000,
                ..Default::default()
            },
            ProblemType::NeuralNetworkTraining => AccelerationConfig {
                strategy: AccelerationStrategy::MixedPrecision,
                max_batch_size: 2000,
                ..Default::default()
            },
            ProblemType::MatrixOptimization => AccelerationConfig {
                strategy: AccelerationStrategy::TensorCore,
                max_batch_size: 1000,
                ..Default::default()
            },
        }
    }

    /// Problem types for acceleration configuration
    pub enum ProblemType {
        GlobalOptimization,
        LeastSquares,
        NeuralNetworkTraining,
        MatrixOptimization,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_acceleration_config() {
        let config = AccelerationConfig::default();
        assert_eq!(config.strategy, AccelerationStrategy::Adaptive);
        assert_eq!(config.min_problem_size, 100);
        assert!(config.async_execution);
    }

    #[test]
    fn test_performance_stats() {
        let mut stats = PerformanceStats::new();

        stats.start_timer("test");
        std::thread::sleep(std::time::Duration::from_millis(10));
        stats.end_timer("test");

        stats.increment_counter("test_count");
        stats.increment_counter("test_count");

        assert!(stats.get_timer("test").unwrap() > 0.0);
        assert_eq!(stats.get_counter("test_count").unwrap(), 2);
    }

    #[test]
    fn test_batch_size_estimation() {
        let batch_size = utils::estimate_batch_size(
            10,          // 10-dimensional problem
            1024 * 1024, // 1MB memory
            GpuPrecision::F64,
        );
        assert!(batch_size > 0);
        assert!(batch_size <= 10000);
    }

    #[test]
    fn test_gpu_beneficial_heuristic() {
        assert!(utils::is_gpu_beneficial(100, 1000, 0.001, 0.1));
        assert!(!utils::is_gpu_beneficial(10, 10, 0.001, 1.0));
    }
}
