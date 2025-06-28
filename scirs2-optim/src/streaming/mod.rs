//! Streaming optimization for real-time learning
//!
//! This module provides streaming gradient descent and other online optimization
//! algorithms designed for real-time data processing and low-latency inference.

use ndarray::{Array, Array1, Array2, ArrayBase, Data, DataMut, Dimension};
use num_traits::Float;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub mod adaptive_streaming;
pub mod concept_drift;
pub mod low_latency;
pub mod streaming_metrics;

// Re-export key types for convenience
pub use concept_drift::{ConceptDriftDetector, DriftDetectorConfig, DriftEvent, DriftStatus};
pub use low_latency::{LowLatencyConfig, LowLatencyMetrics, LowLatencyOptimizer};
pub use streaming_metrics::{MetricsSample, MetricsSummary, StreamingMetricsCollector};

use crate::error::{OptimError, OptimizerError};
use crate::optimizers::Optimizer;

/// Streaming optimization configuration
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Buffer size for mini-batches
    pub buffer_size: usize,

    /// Maximum latency budget (milliseconds)
    pub latency_budget_ms: u64,

    /// Enable adaptive learning rates
    pub adaptive_learning_rate: bool,

    /// Concept drift detection threshold
    pub drift_threshold: f64,

    /// Window size for drift detection
    pub drift_window_size: usize,

    /// Enable gradient compression
    pub gradient_compression: bool,

    /// Compression ratio (0.0 to 1.0)
    pub compression_ratio: f64,

    /// Enable asynchronous updates
    pub async_updates: bool,

    /// Maximum staleness for asynchronous updates
    pub max_staleness: usize,

    /// Enable memory-efficient processing
    pub memory_efficient: bool,

    /// Target memory usage (MB)
    pub memory_budget_mb: usize,

    /// Learning rate adaptation strategy
    pub lr_adaptation: LearningRateAdaptation,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            buffer_size: 32,
            latency_budget_ms: 10,
            adaptive_learning_rate: true,
            drift_threshold: 0.1,
            drift_window_size: 1000,
            gradient_compression: false,
            compression_ratio: 0.5,
            async_updates: false,
            max_staleness: 10,
            memory_efficient: true,
            memory_budget_mb: 100,
            lr_adaptation: LearningRateAdaptation::Adagrad,
        }
    }
}

/// Learning rate adaptation strategies for streaming
#[derive(Debug, Clone, Copy)]
pub enum LearningRateAdaptation {
    /// Fixed learning rate
    Fixed,
    /// AdaGrad-style adaptation
    Adagrad,
    /// RMSprop-style adaptation
    RMSprop,
    /// Performance-based adaptation
    PerformanceBased,
    /// Concept drift aware adaptation
    DriftAware,
}

/// Streaming gradient descent optimizer
pub struct StreamingOptimizer<O, A>
where
    A: Float,
    O: Optimizer<A>,
{
    /// Base optimizer
    base_optimizer: O,

    /// Configuration
    config: StreamingConfig,

    /// Data buffer for mini-batches
    data_buffer: VecDeque<StreamingDataPoint<A>>,

    /// Gradient buffer
    gradient_buffer: Option<Array1<A>>,

    /// Learning rate adaptation state
    lr_adaptation_state: LearningRateAdaptationState<A>,

    /// Concept drift detector
    drift_detector: StreamingDriftDetector<A>,

    /// Performance metrics
    metrics: StreamingMetrics,

    /// Timing information
    timing: TimingTracker,

    /// Memory usage tracker
    memory_tracker: MemoryTracker,

    /// Asynchronous update state
    async_state: Option<AsyncUpdateState<A>>,

    /// Current step count
    step_count: usize,
}

/// Streaming data point
#[derive(Debug, Clone)]
pub struct StreamingDataPoint<A: Float> {
    /// Feature vector
    pub features: Array1<A>,

    /// Target value (for supervised learning)
    pub target: Option<A>,

    /// Timestamp
    pub timestamp: Instant,

    /// Sample weight
    pub weight: A,

    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Learning rate adaptation state
#[derive(Debug, Clone)]
struct LearningRateAdaptationState<A: Float> {
    /// Current learning rate
    current_lr: A,

    /// Accumulated squared gradients (for AdaGrad)
    accumulated_gradients: Option<Array1<A>>,

    /// Exponential moving average of squared gradients (for RMSprop)
    ema_squared_gradients: Option<Array1<A>>,

    /// Performance history
    performance_history: VecDeque<A>,

    /// Last adaptation time
    last_adaptation: Instant,

    /// Adaptation frequency
    adaptation_frequency: Duration,
}

/// Streaming concept drift detection
#[derive(Debug, Clone)]
struct StreamingDriftDetector<A: Float> {
    /// Window of recent losses
    loss_window: VecDeque<A>,

    /// Historical loss statistics
    historical_mean: A,
    historical_std: A,

    /// Drift detection threshold
    threshold: A,

    /// Last drift detection time
    last_drift: Option<Instant>,

    /// Drift count
    drift_count: usize,
}

/// Streaming performance metrics
#[derive(Debug, Clone)]
pub struct StreamingMetrics {
    /// Total samples processed
    pub samples_processed: usize,

    /// Current processing rate (samples/second)
    pub processing_rate: f64,

    /// Average latency per sample (milliseconds)
    pub avg_latency_ms: f64,

    /// 95th percentile latency (milliseconds)
    pub p95_latency_ms: f64,

    /// Memory usage (MB)
    pub memory_usage_mb: f64,

    /// Concept drifts detected
    pub drift_count: usize,

    /// Current loss
    pub current_loss: f64,

    /// Learning rate
    pub current_learning_rate: f64,

    /// Throughput violations (exceeded latency budget)
    pub throughput_violations: usize,
}

/// Timing tracker for performance monitoring
#[derive(Debug)]
struct TimingTracker {
    /// Latency samples
    latency_samples: VecDeque<Duration>,

    /// Last processing start time
    last_start: Option<Instant>,

    /// Processing start time for current batch
    batch_start: Option<Instant>,

    /// Maximum samples to keep
    max_samples: usize,
}

/// Memory usage tracker
#[derive(Debug)]
struct MemoryTracker {
    /// Current estimated usage (bytes)
    current_usage: usize,

    /// Peak usage
    peak_usage: usize,

    /// Memory budget (bytes)
    budget: usize,

    /// Usage history
    usage_history: VecDeque<usize>,
}

/// Asynchronous update state
#[derive(Debug)]
struct AsyncUpdateState<A: Float> {
    /// Pending gradients
    pending_gradients: Vec<Array1<A>>,

    /// Update queue
    update_queue: VecDeque<AsyncUpdate<A>>,

    /// Staleness counter
    staleness_counter: HashMap<usize, usize>,

    /// Update thread handle
    update_thread: Option<std::thread::JoinHandle<()>>,
}

/// Asynchronous update entry
#[derive(Debug, Clone)]
struct AsyncUpdate<A: Float> {
    /// Parameter update
    update: Array1<A>,

    /// Timestamp
    timestamp: Instant,

    /// Priority
    priority: UpdatePriority,

    /// Staleness
    staleness: usize,
}

/// Update priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum UpdatePriority {
    Low,
    Normal,
    High,
    Critical,
}

impl<O, A> StreamingOptimizer<O, A>
where
    A: Float + Default + Clone + Send + Sync + std::fmt::Debug,
    O: Optimizer<A> + Send + Sync,
{
    /// Create a new streaming optimizer
    pub fn new(base_optimizer: O, config: StreamingConfig) -> Self {
        let lr_adaptation_state = LearningRateAdaptationState {
            current_lr: A::from(0.01).unwrap(), // Default learning rate
            accumulated_gradients: None,
            ema_squared_gradients: None,
            performance_history: VecDeque::with_capacity(100),
            last_adaptation: Instant::now(),
            adaptation_frequency: Duration::from_millis(1000),
        };

        let drift_detector = StreamingDriftDetector {
            loss_window: VecDeque::with_capacity(config.drift_window_size),
            historical_mean: A::zero(),
            historical_std: A::one(),
            threshold: A::from(config.drift_threshold).unwrap(),
            last_drift: None,
            drift_count: 0,
        };

        let timing = TimingTracker {
            latency_samples: VecDeque::with_capacity(1000),
            last_start: None,
            batch_start: None,
            max_samples: 1000,
        };

        let memory_tracker = MemoryTracker {
            current_usage: 0,
            peak_usage: 0,
            budget: config.memory_budget_mb * 1024 * 1024,
            usage_history: VecDeque::with_capacity(100),
        };

        let async_state = if config.async_updates {
            Some(AsyncUpdateState {
                pending_gradients: Vec::new(),
                update_queue: VecDeque::new(),
                staleness_counter: HashMap::new(),
                update_thread: None,
            })
        } else {
            None
        };

        Self {
            base_optimizer,
            config,
            data_buffer: VecDeque::with_capacity(config.buffer_size),
            gradient_buffer: None,
            lr_adaptation_state,
            drift_detector,
            metrics: StreamingMetrics::default(),
            timing,
            memory_tracker,
            async_state,
            step_count: 0,
        }
    }

    /// Process a single streaming data point
    pub fn process_sample(
        &mut self,
        data_point: StreamingDataPoint<A>,
    ) -> Result<Option<Array1<A>>, OptimizerError> {
        let start_time = Instant::now();
        self.timing.batch_start = Some(start_time);

        // Add to buffer
        self.data_buffer.push_back(data_point);
        self.update_memory_usage();

        // Check if buffer is full or latency budget is approaching
        let should_update = self.data_buffer.len() >= self.config.buffer_size
            || self.should_force_update(start_time);

        if should_update {
            let result = self.process_buffer()?;

            // Update timing metrics
            let latency = start_time.elapsed();
            self.update_timing_metrics(latency);

            // Check for concept drift
            if let Some(ref update) = result {
                self.check_concept_drift(update)?;
            }

            Ok(result)
        } else {
            Ok(None)
        }
    }

    fn should_force_update(&self, start_time: Instant) -> bool {
        if let Some(batch_start) = self.timing.batch_start {
            let elapsed = start_time.duration_since(batch_start);
            elapsed.as_millis() as u64 >= self.config.latency_budget_ms / 2
        } else {
            false
        }
    }

    fn process_buffer(&mut self) -> Result<Option<Array1<A>>, OptimizerError> {
        if self.data_buffer.is_empty() {
            return Ok(None);
        }

        // Compute mini-batch gradient
        let gradient = self.compute_mini_batch_gradient()?;

        // Apply gradient compression if enabled
        let compressed_gradient = if self.config.gradient_compression {
            self.compress_gradient(&gradient)?
        } else {
            gradient
        };

        // Adapt learning rate
        self.adapt_learning_rate(&compressed_gradient)?;

        // Apply optimizer step
        let current_params = self.get_current_parameters()?;
        let updated_params = if self.config.async_updates {
            self.async_update(&current_params, &compressed_gradient)?
        } else {
            self.sync_update(&current_params, &compressed_gradient)?
        };

        // Clear buffer
        self.data_buffer.clear();
        self.step_count += 1;

        // Update metrics
        self.update_metrics();

        Ok(Some(updated_params))
    }

    fn compute_mini_batch_gradient(&self) -> Result<Array1<A>, OptimizerError> {
        if self.data_buffer.is_empty() {
            return Err(OptimizerError::InvalidConfig(
                "Empty data buffer".to_string(),
            ));
        }

        let batch_size = self.data_buffer.len();
        let feature_dim = self.data_buffer[0].features.len();
        let mut gradient = Array1::zeros(feature_dim);

        // Simplified gradient computation (would depend on loss function)
        for data_point in &self.data_buffer {
            // For demonstration: compute a simple linear regression gradient
            if let Some(target) = data_point.target {
                let prediction = A::zero(); // Would compute actual prediction
                let error = prediction - target;

                for (i, &feature) in data_point.features.iter().enumerate() {
                    gradient[i] = gradient[i] + error * feature * data_point.weight;
                }
            }
        }

        // Average over batch
        let batch_size_a = A::from(batch_size).unwrap();
        gradient.mapv_inplace(|g| g / batch_size_a);

        Ok(gradient)
    }

    fn compress_gradient(&self, gradient: &Array1<A>) -> Result<Array1<A>, OptimizerError> {
        let k = (gradient.len() as f64 * self.config.compression_ratio) as usize;
        let mut compressed = gradient.clone();

        // Top-k sparsification
        let mut abs_values: Vec<(usize, A)> = gradient
            .iter()
            .enumerate()
            .map(|(i, &g)| (i, g.abs()))
            .collect();

        abs_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Zero out all but top-k elements
        for (i, _) in abs_values.iter().skip(k) {
            compressed[*i] = A::zero();
        }

        Ok(compressed)
    }

    fn adapt_learning_rate(&mut self, gradient: &Array1<A>) -> Result<(), OptimizerError> {
        if !self.config.adaptive_learning_rate {
            return Ok(());
        }

        match self.config.lr_adaptation {
            LearningRateAdaptation::Fixed => {
                // Do nothing
            }
            LearningRateAdaptation::Adagrad => {
                self.adapt_adagrad(gradient)?;
            }
            LearningRateAdaptation::RMSprop => {
                self.adapt_rmsprop(gradient)?;
            }
            LearningRateAdaptation::PerformanceBased => {
                self.adapt_performance_based()?;
            }
            LearningRateAdaptation::DriftAware => {
                self.adapt_drift_aware()?;
            }
        }

        Ok(())
    }

    fn adapt_adagrad(&mut self, gradient: &Array1<A>) -> Result<(), OptimizerError> {
        if self.lr_adaptation_state.accumulated_gradients.is_none() {
            self.lr_adaptation_state.accumulated_gradients = Some(Array1::zeros(gradient.len()));
        }

        let acc_grads = self
            .lr_adaptation_state
            .accumulated_gradients
            .as_mut()
            .unwrap();

        // Update accumulated gradients
        for i in 0..gradient.len() {
            acc_grads[i] = acc_grads[i] + gradient[i] * gradient[i];
        }

        // Compute adaptive learning rate (simplified)
        let base_lr = A::from(0.01).unwrap();
        let eps = A::from(1e-8).unwrap();
        let norm_sum = acc_grads.iter().map(|&g| g).sum::<A>();
        let adaptive_factor = (norm_sum + eps).sqrt();

        self.lr_adaptation_state.current_lr = base_lr / adaptive_factor;

        Ok(())
    }

    fn adapt_rmsprop(&mut self, gradient: &Array1<A>) -> Result<(), OptimizerError> {
        if self.lr_adaptation_state.ema_squared_gradients.is_none() {
            self.lr_adaptation_state.ema_squared_gradients = Some(Array1::zeros(gradient.len()));
        }

        let ema_grads = self
            .lr_adaptation_state
            .ema_squared_gradients
            .as_mut()
            .unwrap();
        let decay = A::from(0.9).unwrap();
        let one_minus_decay = A::one() - decay;

        // Update exponential moving average
        for i in 0..gradient.len() {
            ema_grads[i] = decay * ema_grads[i] + one_minus_decay * gradient[i] * gradient[i];
        }

        // Compute adaptive learning rate
        let base_lr = A::from(0.01).unwrap();
        let eps = A::from(1e-8).unwrap();
        let rms = ema_grads.iter().map(|&g| g).sum::<A>().sqrt();

        self.lr_adaptation_state.current_lr = base_lr / (rms + eps);

        Ok(())
    }

    fn adapt_performance_based(&mut self) -> Result<(), OptimizerError> {
        // Adapt based on recent performance
        if self.lr_adaptation_state.performance_history.len() < 2 {
            return Ok(());
        }

        let recent_perf = self.lr_adaptation_state.performance_history.back().unwrap();
        let prev_perf = self
            .lr_adaptation_state
            .performance_history
            .get(self.lr_adaptation_state.performance_history.len() - 2)
            .unwrap();

        let improvement = *prev_perf - *recent_perf; // Assuming lower is better

        if improvement > A::zero() {
            // Performance improved, slightly increase LR
            self.lr_adaptation_state.current_lr =
                self.lr_adaptation_state.current_lr * A::from(1.01).unwrap();
        } else {
            // Performance degraded, decrease LR
            self.lr_adaptation_state.current_lr =
                self.lr_adaptation_state.current_lr * A::from(0.99).unwrap();
        }

        Ok(())
    }

    fn adapt_drift_aware(&mut self) -> Result<(), OptimizerError> {
        // Increase learning rate if drift was recently detected
        if let Some(last_drift) = self.drift_detector.last_drift {
            let time_since_drift = last_drift.elapsed();
            if time_since_drift < Duration::from_secs(60) {
                // Recent drift detected, increase learning rate
                self.lr_adaptation_state.current_lr =
                    self.lr_adaptation_state.current_lr * A::from(1.5).unwrap();
            }
        }

        Ok(())
    }

    fn check_concept_drift(&mut self, _update: &Array1<A>) -> Result<(), OptimizerError> {
        // Simplified concept drift detection based on loss
        let current_loss = A::from(self.metrics.current_loss).unwrap();

        self.drift_detector.loss_window.push_back(current_loss);
        if self.drift_detector.loss_window.len() > self.config.drift_window_size {
            self.drift_detector.loss_window.pop_front();
        }

        if self.drift_detector.loss_window.len() >= 10 {
            // Compute statistics
            let mean = self.drift_detector.loss_window.iter().cloned().sum::<A>()
                / A::from(self.drift_detector.loss_window.len()).unwrap();

            let variance = self
                .drift_detector
                .loss_window
                .iter()
                .map(|&loss| {
                    let diff = loss - mean;
                    diff * diff
                })
                .sum::<A>()
                / A::from(self.drift_detector.loss_window.len()).unwrap();

            let std = variance.sqrt();

            // Check for drift (simplified statistical test)
            let z_score = (current_loss - self.drift_detector.historical_mean).abs()
                / (self.drift_detector.historical_std + A::from(1e-8).unwrap());

            if z_score > self.drift_detector.threshold {
                // Drift detected
                self.drift_detector.last_drift = Some(Instant::now());
                self.drift_detector.drift_count += 1;
                self.metrics.drift_count = self.drift_detector.drift_count;

                // Update historical statistics
                self.drift_detector.historical_mean = mean;
                self.drift_detector.historical_std = std;

                // Trigger learning rate adaptation
                if matches!(
                    self.config.lr_adaptation,
                    LearningRateAdaptation::DriftAware
                ) {
                    self.adapt_drift_aware()?;
                }
            }
        }

        Ok(())
    }

    fn get_current_parameters(&self) -> Result<Array1<A>, OptimizerError> {
        // Placeholder - would get actual parameters from model
        Ok(Array1::zeros(10))
    }

    fn sync_update(
        &mut self,
        params: &Array1<A>,
        gradient: &Array1<A>,
    ) -> Result<Array1<A>, OptimizerError> {
        // Apply gradient update synchronously
        self.base_optimizer.step(params, gradient)
    }

    fn async_update(
        &mut self,
        _params: &Array1<A>,
        gradient: &Array1<A>,
    ) -> Result<Array1<A>, OptimizerError> {
        if let Some(ref mut async_state) = self.async_state {
            // Add to update queue
            let update = AsyncUpdate {
                update: gradient.clone(),
                timestamp: Instant::now(),
                priority: UpdatePriority::Normal,
                staleness: 0,
            };

            async_state.update_queue.push_back(update);

            // Process updates if queue is full or max staleness reached
            if async_state.update_queue.len() >= self.config.buffer_size
                || self.max_staleness_reached()
            {
                return self.process_async_updates();
            }
        }

        // Return current parameters for now
        self.get_current_parameters()
    }

    fn max_staleness_reached(&self) -> bool {
        if let Some(ref async_state) = self.async_state {
            async_state
                .update_queue
                .iter()
                .any(|update| update.staleness >= self.config.max_staleness)
        } else {
            false
        }
    }

    fn process_async_updates(&mut self) -> Result<Array1<A>, OptimizerError> {
        // Simplified async update processing
        if let Some(ref mut async_state) = self.async_state {
            if let Some(update) = async_state.update_queue.pop_front() {
                let current_params = self.get_current_parameters()?;
                return self.base_optimizer.step(&current_params, &update.update);
            }
        }

        self.get_current_parameters()
    }

    fn update_timing_metrics(&mut self, latency: Duration) {
        self.timing.latency_samples.push_back(latency);
        if self.timing.latency_samples.len() > self.timing.max_samples {
            self.timing.latency_samples.pop_front();
        }

        // Check for throughput violations
        if latency.as_millis() as u64 > self.config.latency_budget_ms {
            self.metrics.throughput_violations += 1;
        }
    }

    fn update_memory_usage(&mut self) {
        // Estimate memory usage
        let buffer_size = self.data_buffer.len() * std::mem::size_of::<StreamingDataPoint<A>>();
        let gradient_size = self
            .gradient_buffer
            .as_ref()
            .map(|g| g.len() * std::mem::size_of::<A>())
            .unwrap_or(0);

        self.memory_tracker.current_usage = buffer_size + gradient_size;
        self.memory_tracker.peak_usage = self
            .memory_tracker
            .peak_usage
            .max(self.memory_tracker.current_usage);

        self.memory_tracker
            .usage_history
            .push_back(self.memory_tracker.current_usage);
        if self.memory_tracker.usage_history.len() > 100 {
            self.memory_tracker.usage_history.pop_front();
        }
    }

    fn update_metrics(&mut self) {
        self.metrics.samples_processed += self.data_buffer.len();

        // Update processing rate
        if let Some(batch_start) = self.timing.batch_start {
            let elapsed = batch_start.elapsed().as_secs_f64();
            if elapsed > 0.0 {
                self.metrics.processing_rate = self.data_buffer.len() as f64 / elapsed;
            }
        }

        // Update latency metrics
        if !self.timing.latency_samples.is_empty() {
            let sum: Duration = self.timing.latency_samples.iter().sum();
            self.metrics.avg_latency_ms =
                sum.as_millis() as f64 / self.timing.latency_samples.len() as f64;

            // Compute 95th percentile
            let mut sorted_latencies: Vec<_> = self.timing.latency_samples.iter().collect();
            sorted_latencies.sort();
            let p95_index = (0.95 * sorted_latencies.len() as f64) as usize;
            if p95_index < sorted_latencies.len() {
                self.metrics.p95_latency_ms = sorted_latencies[p95_index].as_millis() as f64;
            }
        }

        // Update memory metrics
        self.metrics.memory_usage_mb = self.memory_tracker.current_usage as f64 / (1024.0 * 1024.0);

        // Update learning rate
        self.metrics.current_learning_rate =
            self.lr_adaptation_state.current_lr.to_f64().unwrap_or(0.0);
    }

    /// Get current streaming metrics
    pub fn get_metrics(&self) -> &StreamingMetrics {
        &self.metrics
    }

    /// Check if streaming optimizer is healthy (within budgets)
    pub fn is_healthy(&self) -> StreamingHealthStatus {
        let mut warnings = Vec::new();
        let mut is_healthy = true;

        // Check latency budget
        if self.metrics.avg_latency_ms > self.config.latency_budget_ms as f64 {
            warnings.push("Average latency exceeds budget".to_string());
            is_healthy = false;
        }

        // Check memory budget
        if self.memory_tracker.current_usage > self.memory_tracker.budget {
            warnings.push("Memory usage exceeds budget".to_string());
            is_healthy = false;
        }

        // Check for frequent concept drift
        if self.metrics.drift_count > 10 && self.step_count > 0 {
            let drift_rate = self.metrics.drift_count as f64 / self.step_count as f64;
            if drift_rate > 0.1 {
                warnings.push("High concept drift rate detected".to_string());
            }
        }

        StreamingHealthStatus {
            is_healthy,
            warnings,
            metrics: self.metrics.clone(),
        }
    }

    /// Force processing of current buffer
    pub fn flush(&mut self) -> Result<Option<Array1<A>>, OptimizerError> {
        if !self.data_buffer.is_empty() {
            self.process_buffer()
        } else {
            Ok(None)
        }
    }
}

impl Default for StreamingMetrics {
    fn default() -> Self {
        Self {
            samples_processed: 0,
            processing_rate: 0.0,
            avg_latency_ms: 0.0,
            p95_latency_ms: 0.0,
            memory_usage_mb: 0.0,
            drift_count: 0,
            current_loss: 0.0,
            current_learning_rate: 0.01,
            throughput_violations: 0,
        }
    }
}

/// Health status of streaming optimizer
#[derive(Debug, Clone)]
pub struct StreamingHealthStatus {
    pub is_healthy: bool,
    pub warnings: Vec<String>,
    pub metrics: StreamingMetrics,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizers::SGD;

    #[test]
    fn test_streaming_config_default() {
        let config = StreamingConfig::default();
        assert_eq!(config.buffer_size, 32);
        assert_eq!(config.latency_budget_ms, 10);
        assert!(config.adaptive_learning_rate);
    }

    #[test]
    fn test_streaming_optimizer_creation() {
        let sgd = SGD::new(0.01);
        let config = StreamingConfig::default();
        let optimizer = StreamingOptimizer::new(sgd, config);

        assert_eq!(optimizer.step_count, 0);
        assert!(optimizer.data_buffer.is_empty());
    }

    #[test]
    fn test_data_point_creation() {
        let features = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let data_point = StreamingDataPoint {
            features,
            target: Some(0.5),
            timestamp: Instant::now(),
            weight: 1.0,
            metadata: HashMap::new(),
        };

        assert_eq!(data_point.features.len(), 3);
        assert_eq!(data_point.target, Some(0.5));
        assert_eq!(data_point.weight, 1.0);
    }

    #[test]
    fn test_streaming_metrics_default() {
        let metrics = StreamingMetrics::default();
        assert_eq!(metrics.samples_processed, 0);
        assert_eq!(metrics.processing_rate, 0.0);
        assert_eq!(metrics.drift_count, 0);
    }
}
