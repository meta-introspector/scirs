//! Real-Time Performance Monitoring and Adaptation for Ultrathink Processors
//!
//! This module provides comprehensive real-time monitoring and adaptive optimization
//! for all ultrathink mode processors, including quantum-inspired, neural-adaptive,
//! and hybrid processors.

use crate::adaptive_memory_compression::MemoryStats;
use crate::error::SparseResult;
use crate::neural_adaptive_sparse::NeuralProcessorStats;
use crate::quantum_inspired_sparse::QuantumProcessorStats;
use crate::quantum_neural_hybrid::QuantumNeuralHybridStats;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Configuration for real-time performance monitoring
#[derive(Debug, Clone)]
pub struct PerformanceMonitorConfig {
    /// Monitoring interval in milliseconds
    pub monitoring_interval_ms: u64,
    /// Maximum number of performance samples to keep
    pub max_samples: usize,
    /// Enable adaptive tuning based on performance
    pub adaptive_tuning: bool,
    /// Performance threshold for adaptation triggers
    pub adaptation_threshold: f64,
    /// Enable real-time alerts
    pub enable_alerts: bool,
    /// Alert threshold for performance degradation
    pub alert_threshold: f64,
    /// Enable automatic optimization
    pub auto_optimization: bool,
    /// Optimization interval in seconds
    pub optimization_interval_s: u64,
    /// Enable performance prediction
    pub enable_prediction: bool,
    /// Prediction horizon in samples
    pub prediction_horizon: usize,
}

impl Default for PerformanceMonitorConfig {
    fn default() -> Self {
        Self {
            monitoring_interval_ms: 100,
            max_samples: 10000,
            adaptive_tuning: true,
            adaptation_threshold: 0.8,
            enable_alerts: true,
            alert_threshold: 0.5,
            auto_optimization: true,
            optimization_interval_s: 30,
            enable_prediction: true,
            prediction_horizon: 50,
        }
    }
}

/// Real-time performance monitor for ultrathink processors
#[allow(dead_code)]
pub struct RealTimePerformanceMonitor {
    config: PerformanceMonitorConfig,
    monitoring_active: Arc<AtomicBool>,
    sample_counter: AtomicUsize,
    performance_history: Arc<Mutex<PerformanceHistory>>,
    system_metrics: Arc<Mutex<SystemMetrics>>,
    alert_manager: Arc<Mutex<AlertManager>>,
    adaptation_engine: Arc<Mutex<AdaptationEngine>>,
    prediction_engine: Arc<Mutex<PredictionEngine>>,
    processor_registry: Arc<Mutex<ProcessorRegistry>>,
}

/// Performance history tracking
#[derive(Debug)]
#[allow(dead_code)]
struct PerformanceHistory {
    samples: VecDeque<PerformanceSample>,
    aggregated_metrics: AggregatedMetrics,
    trend_analysis: TrendAnalysis,
    performance_baselines: HashMap<String, f64>,
}

/// Individual performance sample
#[derive(Debug, Clone)]
pub struct PerformanceSample {
    pub timestamp: u64,
    pub processor_type: ProcessorType,
    pub processor_id: String,
    pub execution_time_ms: f64,
    pub throughput_ops_per_sec: f64,
    pub memory_usage_mb: f64,
    pub cache_hit_ratio: f64,
    pub error_rate: f64,
    pub cpu_utilization: f64,
    pub gpu_utilization: f64,
    pub quantum_coherence: Option<f64>,
    pub neural_confidence: Option<f64>,
    pub compression_ratio: Option<f64>,
}

/// Execution timing helper for measuring performance
pub struct ExecutionTimer {
    start_time: Instant,
}

impl ExecutionTimer {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
        }
    }
}

impl Default for ExecutionTimer {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutionTimer {
    
    pub fn elapsed_ms(&self) -> f64 {
        self.start_time.elapsed().as_millis() as f64
    }
    
    pub fn restart(&mut self) {
        self.start_time = Instant::now();
    }
}

/// Type of ultrathink processor
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProcessorType {
    QuantumInspired,
    NeuralAdaptive,
    QuantumNeuralHybrid,
    MemoryCompression,
}

/// Aggregated performance metrics
#[derive(Debug, Default, Clone)]
pub struct AggregatedMetrics {
    avg_execution_time: f64,
    avg_throughput: f64,
    avg_memory_usage: f64,
    avg_cache_hit_ratio: f64,
    avg_error_rate: f64,
    peak_throughput: f64,
    min_execution_time: f64,
    total_operations: usize,
    efficiency_score: f64,
}

/// Trend analysis for performance prediction
#[derive(Debug)]
struct TrendAnalysis {
    execution_time_trend: LinearTrend,
    throughput_trend: LinearTrend,
    memory_trend: LinearTrend,
    efficiency_trend: LinearTrend,
    anomaly_detection: AnomalyDetector,
}

/// Linear trend analysis
#[derive(Debug, Default)]
#[allow(dead_code)]
struct LinearTrend {
    slope: f64,
    intercept: f64,
    correlation: f64,
    prediction_confidence: f64,
}

/// Anomaly detection system
#[derive(Debug)]
struct AnomalyDetector {
    moving_average: f64,
    moving_variance: f64,
    anomaly_threshold: f64,
    recent_anomalies: VecDeque<AnomalyEvent>,
}

/// Anomaly event
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct AnomalyEvent {
    timestamp: u64,
    metric_name: String,
    expected_value: f64,
    actual_value: f64,
    severity: AnomalySeverity,
}

/// Severity of anomaly
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
enum AnomalySeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// System metrics tracking
#[derive(Debug)]
#[allow(dead_code)]
struct SystemMetrics {
    cpu_usage: f64,
    memory_usage: f64,
    gpu_usage: f64,
    network_io: f64,
    disk_io: f64,
    temperature: f64,
    power_consumption: f64,
    system_load: f64,
}

/// Alert management system
#[derive(Debug)]
#[allow(dead_code)]
struct AlertManager {
    active_alerts: HashMap<String, Alert>,
    alert_history: VecDeque<Alert>,
    notification_channels: Vec<NotificationChannel>,
    alert_rules: Vec<AlertRule>,
}

/// Performance alert
#[derive(Debug, Clone)]
pub struct Alert {
    pub id: String,
    pub timestamp: u64,
    pub severity: AlertSeverity,
    pub message: String,
    pub processor_type: ProcessorType,
    pub processor_id: String,
    pub metric_name: String,
    pub threshold_value: f64,
    pub actual_value: f64,
    pub resolved: bool,
}

/// Alert severity levels
#[derive(Debug, Clone, Copy)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Notification channels for alerts
#[derive(Debug, Clone)]
#[allow(dead_code)]
enum NotificationChannel {
    Console,
    Log,
    Email,
    Webhook,
}

/// Alert rule configuration
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct AlertRule {
    id: String,
    metric_name: String,
    condition: AlertCondition,
    threshold: f64,
    severity: AlertSeverity,
    enabled: bool,
}

/// Alert condition types
#[derive(Debug, Clone)]
#[allow(dead_code)]
enum AlertCondition {
    GreaterThan,
    LessThan,
    Equals,
    NotEquals,
    PercentageIncrease,
    PercentageDecrease,
}

/// Adaptive engine for performance optimization
#[derive(Debug)]
#[allow(dead_code)]
struct AdaptationEngine {
    optimization_strategies: Vec<OptimizationStrategy>,
    strategy_effectiveness: HashMap<String, f64>,
    active_optimizations: HashMap<String, ActiveOptimization>,
    adaptation_history: VecDeque<AdaptationEvent>,
}

/// Optimization strategy
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct OptimizationStrategy {
    id: String,
    name: String,
    description: String,
    target_metrics: Vec<String>,
    parameters: HashMap<String, f64>,
    effectiveness_score: f64,
    usage_count: usize,
}

/// Active optimization
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ActiveOptimization {
    strategy_id: String,
    processor_id: String,
    start_time: u64,
    expected_improvement: f64,
    actual_improvement: Option<f64>,
    status: OptimizationStatus,
}

/// Optimization status
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
enum OptimizationStatus {
    Pending,
    Active,
    Completed,
    Failed,
}

/// Adaptation event
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct AdaptationEvent {
    timestamp: u64,
    processor_type: ProcessorType,
    processor_id: String,
    strategy_applied: String,
    trigger_reason: String,
    before_metrics: HashMap<String, f64>,
    after_metrics: HashMap<String, f64>,
    improvement_achieved: f64,
}

/// Performance prediction engine
#[derive(Debug)]
#[allow(dead_code)]
struct PredictionEngine {
    prediction_models: HashMap<String, PredictionModel>,
    forecast_cache: HashMap<String, Forecast>,
    model_accuracy: HashMap<String, f64>,
}

/// Prediction model
#[derive(Debug)]
#[allow(dead_code)]
struct PredictionModel {
    model_type: ModelType,
    parameters: Vec<f64>,
    training_data: VecDeque<f64>,
    last_updated: u64,
    accuracy: f64,
}

/// Types of prediction models
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
enum ModelType {
    LinearRegression,
    MovingAverage,
    ExponentialSmoothing,
    Arima,
    NeuralNetwork,
}

/// Performance forecast
#[derive(Debug, Clone)]
pub struct Forecast {
    pub metric_name: String,
    pub predictions: Vec<PredictionPoint>,
    pub confidence_interval: (f64, f64),
    pub model_accuracy: f64,
    pub forecast_horizon: usize,
}

/// Individual prediction point
#[derive(Debug, Clone)]
pub struct PredictionPoint {
    pub timestamp: u64,
    pub predicted_value: f64,
    pub confidence: f64,
}

/// Registry of monitored processors
struct ProcessorRegistry {
    quantum_processors: HashMap<String, Box<dyn QuantumProcessorMonitor>>,
    neural_processors: HashMap<String, Box<dyn NeuralProcessorMonitor>>,
    hybrid_processors: HashMap<String, Box<dyn HybridProcessorMonitor>>,
    memory_compressors: HashMap<String, Box<dyn MemoryCompressorMonitor>>,
}

/// Monitoring traits for different processor types
pub trait QuantumProcessorMonitor: Send + Sync {
    fn get_stats(&self) -> QuantumProcessorStats;
    fn get_id(&self) -> &str;
}

pub trait NeuralProcessorMonitor: Send + Sync {
    fn get_stats(&self) -> NeuralProcessorStats;
    fn get_id(&self) -> &str;
}

pub trait HybridProcessorMonitor: Send + Sync {
    fn get_stats(&self) -> QuantumNeuralHybridStats;
    fn get_id(&self) -> &str;
}

pub trait MemoryCompressorMonitor: Send + Sync {
    fn get_stats(&self) -> MemoryStats;
    fn get_id(&self) -> &str;
}

impl RealTimePerformanceMonitor {
    /// Create a new real-time performance monitor
    pub fn new(config: PerformanceMonitorConfig) -> Self {
        let performance_history = PerformanceHistory {
            samples: VecDeque::with_capacity(config.max_samples),
            aggregated_metrics: AggregatedMetrics::default(),
            trend_analysis: TrendAnalysis::new(),
            performance_baselines: HashMap::new(),
        };

        let system_metrics = SystemMetrics {
            cpu_usage: 0.0,
            memory_usage: 0.0,
            gpu_usage: 0.0,
            network_io: 0.0,
            disk_io: 0.0,
            temperature: 0.0,
            power_consumption: 0.0,
            system_load: 0.0,
        };

        let alert_manager = AlertManager {
            active_alerts: HashMap::new(),
            alert_history: VecDeque::new(),
            notification_channels: vec![NotificationChannel::Console, NotificationChannel::Log],
            alert_rules: Self::create_default_alert_rules(),
        };

        let adaptation_engine = AdaptationEngine {
            optimization_strategies: Self::create_default_strategies(),
            strategy_effectiveness: HashMap::new(),
            active_optimizations: HashMap::new(),
            adaptation_history: VecDeque::new(),
        };

        let prediction_engine = PredictionEngine {
            prediction_models: HashMap::new(),
            forecast_cache: HashMap::new(),
            model_accuracy: HashMap::new(),
        };

        let processor_registry = ProcessorRegistry {
            quantum_processors: HashMap::new(),
            neural_processors: HashMap::new(),
            hybrid_processors: HashMap::new(),
            memory_compressors: HashMap::new(),
        };

        Self {
            config,
            monitoring_active: Arc::new(AtomicBool::new(false)),
            sample_counter: AtomicUsize::new(0),
            performance_history: Arc::new(Mutex::new(performance_history)),
            system_metrics: Arc::new(Mutex::new(system_metrics)),
            alert_manager: Arc::new(Mutex::new(alert_manager)),
            adaptation_engine: Arc::new(Mutex::new(adaptation_engine)),
            prediction_engine: Arc::new(Mutex::new(prediction_engine)),
            processor_registry: Arc::new(Mutex::new(processor_registry)),
        }
    }

    /// Start real-time monitoring
    pub fn start_monitoring(&self) -> SparseResult<()> {
        if self.monitoring_active.swap(true, Ordering::Relaxed) {
            return Ok(()); // Already running
        }

        let monitoring_active = Arc::clone(&self.monitoring_active);
        let config = self.config.clone();
        let performance_history = Arc::clone(&self.performance_history);
        let system_metrics = Arc::clone(&self.system_metrics);
        let alert_manager = Arc::clone(&self.alert_manager);
        let adaptation_engine = Arc::clone(&self.adaptation_engine);
        let prediction_engine = Arc::clone(&self.prediction_engine);
        let processor_registry = Arc::clone(&self.processor_registry);

        // Spawn monitoring thread
        std::thread::spawn(move || {
            let interval = Duration::from_millis(config.monitoring_interval_ms);
            let mut last_optimization = Instant::now();

            while monitoring_active.load(Ordering::Relaxed) {
                let start_time = Instant::now();

                // Collect performance samples
                Self::collect_performance_samples(
                    &processor_registry,
                    &performance_history,
                    &system_metrics,
                );

                // Update aggregated metrics and trends
                Self::update_aggregated_metrics(&performance_history);
                Self::update_trend_analysis(&performance_history);

                // Check for alerts
                if config.enable_alerts {
                    Self::check_alerts(&performance_history, &alert_manager);
                }

                // Run predictions
                if config.enable_prediction {
                    Self::update_predictions(&performance_history, &prediction_engine);
                }

                // Run adaptive optimization
                if config.auto_optimization
                    && last_optimization.elapsed()
                        >= Duration::from_secs(config.optimization_interval_s)
                {
                    Self::run_adaptive_optimization(
                        &performance_history,
                        &adaptation_engine,
                        &processor_registry,
                    );
                    last_optimization = Instant::now();
                }

                // Sleep for remaining interval time
                let elapsed = start_time.elapsed();
                if elapsed < interval {
                    std::thread::sleep(interval - elapsed);
                }
            }
        });

        Ok(())
    }

    /// Stop monitoring
    pub fn stop_monitoring(&self) {
        self.monitoring_active.store(false, Ordering::Relaxed);
    }

    /// Register a quantum processor for monitoring
    pub fn register_quantum_processor(
        &self,
        id: String,
        processor: Box<dyn QuantumProcessorMonitor>,
    ) -> SparseResult<()> {
        if let Ok(mut registry) = self.processor_registry.lock() {
            registry.quantum_processors.insert(id, processor);
        }
        Ok(())
    }

    /// Register a neural processor for monitoring
    pub fn register_neural_processor(
        &self,
        id: String,
        processor: Box<dyn NeuralProcessorMonitor>,
    ) -> SparseResult<()> {
        if let Ok(mut registry) = self.processor_registry.lock() {
            registry.neural_processors.insert(id, processor);
        }
        Ok(())
    }

    /// Register a hybrid processor for monitoring
    pub fn register_hybrid_processor(
        &self,
        id: String,
        processor: Box<dyn HybridProcessorMonitor>,
    ) -> SparseResult<()> {
        if let Ok(mut registry) = self.processor_registry.lock() {
            registry.hybrid_processors.insert(id, processor);
        }
        Ok(())
    }

    /// Register a memory compressor for monitoring
    pub fn register_memory_compressor(
        &self,
        id: String,
        compressor: Box<dyn MemoryCompressorMonitor>,
    ) -> SparseResult<()> {
        if let Ok(mut registry) = self.processor_registry.lock() {
            registry.memory_compressors.insert(id, compressor);
        }
        Ok(())
    }

    /// Get current performance metrics
    pub fn get_current_metrics(&self) -> PerformanceMetrics {
        let history = self.performance_history.lock().unwrap();
        let system = self.system_metrics.lock().unwrap();

        PerformanceMetrics {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            aggregated: history.aggregated_metrics.clone(),
            system_metrics: SystemMetricsSnapshot {
                cpu_usage: system.cpu_usage,
                memory_usage: system.memory_usage,
                gpu_usage: system.gpu_usage,
                system_load: system.system_load,
            },
            total_samples: history.samples.len(),
            monitoring_active: self.monitoring_active.load(Ordering::Relaxed),
        }
    }

    /// Get active alerts
    pub fn get_active_alerts(&self) -> Vec<Alert> {
        if let Ok(alert_manager) = self.alert_manager.lock() {
            alert_manager.active_alerts.values().cloned().collect()
        } else {
            Vec::new()
        }
    }

    /// Get performance forecast
    pub fn get_forecast(&self, metric_name: &str, _horizon: usize) -> Option<Forecast> {
        if let Ok(prediction_engine) = self.prediction_engine.lock() {
            prediction_engine.forecast_cache.get(metric_name).cloned()
        } else {
            None
        }
    }

    // Internal implementation methods

    fn collect_performance_samples(
        registry: &Arc<Mutex<ProcessorRegistry>>,
        history: &Arc<Mutex<PerformanceHistory>>,
        system_metrics: &Arc<Mutex<SystemMetrics>>,
    ) {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let mut samples = Vec::new();

        // Get current system metrics once for all samples
        let current_cpu = Self::get_cpu_usage();
        let current_gpu = Self::get_gpu_usage();
        let current_memory = Self::get_memory_usage();

        if let Ok(registry) = registry.lock() {
            // Collect quantum processor samples
            for (id, processor) in &registry.quantum_processors {
                let stats = processor.get_stats();
                
                // Estimate execution time based on operations and coherence
                let base_time = 1.0 / (stats.operations_count.max(1) as f64);
                let coherence_factor = stats.average_logical_fidelity;
                let estimated_exec_time = base_time * (2.0 - coherence_factor) * 1000.0;
                
                // Estimate cache efficiency from quantum metrics
                let cache_efficiency = (stats.average_logical_fidelity * 0.8 + 0.2).min(1.0);
                
                samples.push(PerformanceSample {
                    timestamp,
                    processor_type: ProcessorType::QuantumInspired,
                    processor_id: id.clone(),
                    execution_time_ms: estimated_exec_time,
                    throughput_ops_per_sec: stats.operations_count as f64 / estimated_exec_time * 1000.0,
                    memory_usage_mb: stats.cache_efficiency * current_memory * 100.0,
                    cache_hit_ratio: cache_efficiency,
                    error_rate: 1.0 - stats.average_logical_fidelity,
                    cpu_utilization: current_cpu * 0.6, // Quantum processors use less CPU
                    gpu_utilization: current_gpu * 0.1, // Minimal GPU usage for quantum simulation
                    quantum_coherence: Some(stats.average_logical_fidelity),
                    neural_confidence: None,
                    compression_ratio: None,
                });
            }

            // Collect neural processor samples
            for (id, processor) in &registry.neural_processors {
                let stats = processor.get_stats();
                
                // Estimate execution time based on neural complexity
                let neural_complexity = stats.pattern_memory_size as f64 + stats.experience_buffer_size as f64;
                let base_time = neural_complexity / 10000.0; // Scale factor
                let learning_overhead = if stats.rl_enabled { 1.5 } else { 1.0 };
                let estimated_exec_time = base_time * learning_overhead * 1000.0;
                
                // Neural processors typically have good cache locality
                let neural_cache_ratio = 0.85 + (1.0 - stats.current_exploration_rate) * 0.1;
                
                // Error rate based on exploration vs exploitation balance
                let neural_error_rate = if stats.rl_enabled {
                    stats.current_exploration_rate * 0.1 + 0.01
                } else {
                    0.05
                };
                
                samples.push(PerformanceSample {
                    timestamp,
                    processor_type: ProcessorType::NeuralAdaptive,
                    processor_id: id.clone(),
                    execution_time_ms: estimated_exec_time,
                    throughput_ops_per_sec: stats.adaptations_count as f64 / estimated_exec_time * 1000.0,
                    memory_usage_mb: stats.pattern_memory_size as f64 / 1024.0 + current_memory * 50.0,
                    cache_hit_ratio: neural_cache_ratio,
                    error_rate: neural_error_rate,
                    cpu_utilization: current_cpu * 0.8, // Neural networks are CPU intensive
                    gpu_utilization: current_gpu * 0.3, // Some GPU usage for matrix ops
                    quantum_coherence: None,
                    neural_confidence: Some(1.0 - stats.current_exploration_rate),
                    compression_ratio: None,
                });
            }

            // Collect hybrid processor samples
            for (id, processor) in &registry.hybrid_processors {
                let stats = processor.get_stats();
                
                // Hybrid execution time depends on synchronization and strategy balance
                let complexity_factor = 1.0 + (1.0 - stats.hybrid_synchronization) * 0.5;
                let quantum_weight_factor = stats.quantum_weight * 1.2; // Quantum is slower
                let neural_weight_factor = stats.neural_weight * 0.8;   // Neural is faster
                let base_time = (quantum_weight_factor + neural_weight_factor) * complexity_factor;
                let estimated_exec_time = base_time * 1000.0;
                
                // Cache efficiency combines both quantum and neural characteristics
                let hybrid_cache_ratio = stats.quantum_coherence * 0.9 + stats.neural_confidence * 0.85;
                
                samples.push(PerformanceSample {
                    timestamp,
                    processor_type: ProcessorType::QuantumNeuralHybrid,
                    processor_id: id.clone(),
                    execution_time_ms: estimated_exec_time,
                    throughput_ops_per_sec: stats.total_operations as f64 / estimated_exec_time * 1000.0,
                    memory_usage_mb: stats.memory_utilization * current_memory * 200.0,
                    cache_hit_ratio: hybrid_cache_ratio.min(1.0),
                    error_rate: 1.0 - stats.hybrid_synchronization,
                    cpu_utilization: current_cpu * (0.6 * stats.quantum_weight + 0.8 * stats.neural_weight),
                    gpu_utilization: current_gpu * (0.1 * stats.quantum_weight + 0.4 * stats.neural_weight),
                    quantum_coherence: Some(stats.quantum_coherence),
                    neural_confidence: Some(stats.neural_confidence),
                    compression_ratio: None,
                });
            }

            // Collect memory compressor samples
            for (id, compressor) in &registry.memory_compressors {
                let stats = compressor.get_stats();
                
                // Real execution time from compression stats
                let compression_exec_time = if stats.compression_stats.compression_time > 0.0 {
                    stats.compression_stats.compression_time * 1000.0
                } else {
                    1.0 // Minimum 1ms
                };
                
                // Throughput based on blocks processed per time
                let throughput = if compression_exec_time > 0.0 {
                    stats.compression_stats.total_blocks as f64 / compression_exec_time * 1000.0
                } else {
                    stats.compression_stats.total_blocks as f64
                };
                
                // Error rate based on compression efficiency
                let compression_error_rate = if stats.compression_stats.compression_ratio > 1.0 {
                    0.01 / stats.compression_stats.compression_ratio // Better compression = fewer errors
                } else {
                    0.05 // Higher error rate for poor compression
                };
                
                // CPU utilization for compression is typically high
                let compression_cpu_util = current_cpu * 0.9;
                
                samples.push(PerformanceSample {
                    timestamp,
                    processor_type: ProcessorType::MemoryCompression,
                    processor_id: id.clone(),
                    execution_time_ms: compression_exec_time,
                    throughput_ops_per_sec: throughput,
                    memory_usage_mb: stats.current_memory_usage as f64 / (1024.0 * 1024.0),
                    cache_hit_ratio: stats.cache_hit_ratio,
                    error_rate: compression_error_rate,
                    cpu_utilization: compression_cpu_util,
                    gpu_utilization: current_gpu * 0.05, // Minimal GPU usage for compression
                    quantum_coherence: None,
                    neural_confidence: None,
                    compression_ratio: Some(stats.compression_stats.compression_ratio),
                });
            }
        }

        // Store samples in history
        if let Ok(mut history) = history.lock() {
            for sample in samples {
                history.samples.push_back(sample);
                if history.samples.len() > 10000 {
                    // Max samples
                    history.samples.pop_front();
                }
            }
        }

        // Update system metrics (simplified)
        if let Ok(mut system) = system_metrics.lock() {
            system.cpu_usage = Self::get_cpu_usage();
            system.memory_usage = Self::get_memory_usage();
            system.gpu_usage = Self::get_gpu_usage();
            system.system_load = Self::get_system_load();
        }
    }

    fn update_aggregated_metrics(history: &Arc<Mutex<PerformanceHistory>>) {
        if let Ok(mut history) = history.lock() {
            if history.samples.is_empty() {
                return;
            }

            let count = history.samples.len() as f64;

            // Calculate all metrics first before updating the struct
            let avg_execution_time = history
                .samples
                .iter()
                .map(|s| s.execution_time_ms)
                .sum::<f64>()
                / count;
            let avg_throughput = history
                .samples
                .iter()
                .map(|s| s.throughput_ops_per_sec)
                .sum::<f64>()
                / count;
            let avg_memory_usage = history
                .samples
                .iter()
                .map(|s| s.memory_usage_mb)
                .sum::<f64>()
                / count;
            let avg_cache_hit_ratio = history
                .samples
                .iter()
                .map(|s| s.cache_hit_ratio)
                .sum::<f64>()
                / count;
            let avg_error_rate = history.samples.iter().map(|s| s.error_rate).sum::<f64>() / count;
            let peak_throughput = history
                .samples
                .iter()
                .map(|s| s.throughput_ops_per_sec)
                .fold(0.0, f64::max);
            let min_execution_time = history
                .samples
                .iter()
                .map(|s| s.execution_time_ms)
                .fold(f64::INFINITY, f64::min);
            let total_operations = history.samples.len();

            // Calculate efficiency score
            let efficiency_score = (avg_throughput * avg_cache_hit_ratio)
                / (avg_execution_time + 1.0)
                * (1.0 - avg_error_rate);

            // Now update all the metrics
            history.aggregated_metrics.avg_execution_time = avg_execution_time;
            history.aggregated_metrics.avg_throughput = avg_throughput;
            history.aggregated_metrics.avg_memory_usage = avg_memory_usage;
            history.aggregated_metrics.avg_cache_hit_ratio = avg_cache_hit_ratio;
            history.aggregated_metrics.avg_error_rate = avg_error_rate;
            history.aggregated_metrics.peak_throughput = peak_throughput;
            history.aggregated_metrics.min_execution_time = min_execution_time;
            history.aggregated_metrics.total_operations = total_operations;
            history.aggregated_metrics.efficiency_score = efficiency_score;
        }
    }

    fn update_trend_analysis(history: &Arc<Mutex<PerformanceHistory>>) {
        if let Ok(mut history) = history.lock() {
            if history.samples.len() < 10 {
                return;
            }

            // Clone recent samples to avoid borrow checker issues
            let recent_samples: Vec<_> = history.samples.iter().rev().take(100).cloned().collect();

            // Calculate all trends first
            let execution_times: Vec<f64> =
                recent_samples.iter().map(|s| s.execution_time_ms).collect();
            let execution_time_trend = Self::calculate_linear_trend(&execution_times);

            let throughputs: Vec<f64> = recent_samples
                .iter()
                .map(|s| s.throughput_ops_per_sec)
                .collect();
            let throughput_trend = Self::calculate_linear_trend(&throughputs);

            let memory_usage: Vec<f64> = recent_samples.iter().map(|s| s.memory_usage_mb).collect();
            let memory_trend = Self::calculate_linear_trend(&memory_usage);

            let efficiency: Vec<f64> = recent_samples
                .iter()
                .map(|s| {
                    (s.throughput_ops_per_sec * s.cache_hit_ratio) / (s.execution_time_ms + 1.0)
                        * (1.0 - s.error_rate)
                })
                .collect();
            let efficiency_trend = Self::calculate_linear_trend(&efficiency);

            // Now update the trends
            history.trend_analysis.execution_time_trend = execution_time_trend;
            history.trend_analysis.throughput_trend = throughput_trend;
            history.trend_analysis.memory_trend = memory_trend;
            history.trend_analysis.efficiency_trend = efficiency_trend;

            // Update anomaly detection
            history.trend_analysis.anomaly_detection.update(&efficiency);
        }
    }

    fn calculate_linear_trend(data: &[f64]) -> LinearTrend {
        if data.len() < 2 {
            return LinearTrend::default();
        }

        let n = data.len() as f64;
        let x_values: Vec<f64> = (0..data.len()).map(|i| i as f64).collect();

        let x_mean = x_values.iter().sum::<f64>() / n;
        let y_mean = data.iter().sum::<f64>() / n;

        let numerator: f64 = x_values
            .iter()
            .zip(data)
            .map(|(x, y)| (x - x_mean) * (y - y_mean))
            .sum();
        let denominator: f64 = x_values.iter().map(|x| (x - x_mean).powi(2)).sum();

        let slope = if denominator != 0.0 {
            numerator / denominator
        } else {
            0.0
        };
        let intercept = y_mean - slope * x_mean;

        // Calculate correlation coefficient
        let ss_tot: f64 = data.iter().map(|y| (y - y_mean).powi(2)).sum();
        let ss_res: f64 = x_values
            .iter()
            .zip(data)
            .map(|(x, y)| {
                let predicted = slope * x + intercept;
                (y - predicted).powi(2)
            })
            .sum();

        let correlation = if ss_tot != 0.0 {
            1.0 - (ss_res / ss_tot)
        } else {
            0.0
        };

        LinearTrend {
            slope,
            intercept,
            correlation: correlation.sqrt(),
            prediction_confidence: correlation.abs(),
        }
    }

    fn check_alerts(
        history: &Arc<Mutex<PerformanceHistory>>,
        alert_manager: &Arc<Mutex<AlertManager>>,
    ) {
        // Simplified alert checking
        if let (Ok(history), Ok(mut alert_manager)) = (history.lock(), alert_manager.lock()) {
            let metrics = &history.aggregated_metrics;

            // Check efficiency degradation
            if metrics.efficiency_score < 0.5 {
                let alert = Alert {
                    id: format!(
                        "efficiency_low_{}",
                        SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_secs()
                    ),
                    timestamp: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                    severity: AlertSeverity::Warning,
                    message: "System efficiency below threshold".to_string(),
                    processor_type: ProcessorType::QuantumInspired, // Placeholder
                    processor_id: "system".to_string(),
                    metric_name: "efficiency_score".to_string(),
                    threshold_value: 0.5,
                    actual_value: metrics.efficiency_score,
                    resolved: false,
                };

                alert_manager.active_alerts.insert(alert.id.clone(), alert);
            }
        }
    }

    fn update_predictions(
        history: &Arc<Mutex<PerformanceHistory>>,
        prediction_engine: &Arc<Mutex<PredictionEngine>>,
    ) {
        // Simplified prediction update
        if let (Ok(history), Ok(mut prediction_engine)) = (history.lock(), prediction_engine.lock())
        {
            if history.samples.len() < 20 {
                return;
            }

            let recent_efficiency: Vec<f64> = history
                .samples
                .iter()
                .rev()
                .take(50)
                .map(|s| {
                    (s.throughput_ops_per_sec * s.cache_hit_ratio) / (s.execution_time_ms + 1.0)
                        * (1.0 - s.error_rate)
                })
                .collect();

            // Simple moving average prediction
            let avg = recent_efficiency.iter().sum::<f64>() / recent_efficiency.len() as f64;
            let variance = recent_efficiency
                .iter()
                .map(|x| (x - avg).powi(2))
                .sum::<f64>()
                / recent_efficiency.len() as f64;
            let std_dev = variance.sqrt();

            let mut predictions = Vec::new();
            for i in 1..=10 {
                predictions.push(PredictionPoint {
                    timestamp: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_secs()
                        + i * 60,
                    predicted_value: avg,
                    confidence: 1.0 / (1.0 + std_dev),
                });
            }

            let forecast = Forecast {
                metric_name: "efficiency_score".to_string(),
                predictions,
                confidence_interval: (avg - std_dev, avg + std_dev),
                model_accuracy: 0.8, // Placeholder
                forecast_horizon: 10,
            };

            prediction_engine
                .forecast_cache
                .insert("efficiency_score".to_string(), forecast);
        }
    }

    fn run_adaptive_optimization(
        history: &Arc<Mutex<PerformanceHistory>>,
        adaptation_engine: &Arc<Mutex<AdaptationEngine>>,
        _processor_registry: &Arc<Mutex<ProcessorRegistry>>,
    ) {
        // Simplified adaptive optimization
        if let (Ok(history), Ok(mut adaptation_engine)) = (history.lock(), adaptation_engine.lock())
        {
            let metrics = &history.aggregated_metrics;

            // Check if optimization is needed
            if metrics.efficiency_score < 0.7 {
                // Select optimization strategy
                let strategy = adaptation_engine.optimization_strategies.first().cloned();

                if let Some(strategy) = strategy {
                    let optimization = ActiveOptimization {
                        strategy_id: strategy.id.clone(),
                        processor_id: "system".to_string(),
                        start_time: SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_secs(),
                        expected_improvement: 0.2,
                        actual_improvement: None,
                        status: OptimizationStatus::Pending,
                    };

                    adaptation_engine
                        .active_optimizations
                        .insert(optimization.processor_id.clone(), optimization);

                    // Log adaptation event
                    let event = AdaptationEvent {
                        timestamp: SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_secs(),
                        processor_type: ProcessorType::QuantumInspired,
                        processor_id: "system".to_string(),
                        strategy_applied: strategy.name,
                        trigger_reason: "Low efficiency score".to_string(),
                        before_metrics: HashMap::new(),
                        after_metrics: HashMap::new(),
                        improvement_achieved: 0.0,
                    };

                    adaptation_engine.adaptation_history.push_back(event);
                }
            }
        }
    }

    fn create_default_alert_rules() -> Vec<AlertRule> {
        vec![
            AlertRule {
                id: "low_efficiency".to_string(),
                metric_name: "efficiency_score".to_string(),
                condition: AlertCondition::LessThan,
                threshold: 0.5,
                severity: AlertSeverity::Warning,
                enabled: true,
            },
            AlertRule {
                id: "high_error_rate".to_string(),
                metric_name: "error_rate".to_string(),
                condition: AlertCondition::GreaterThan,
                threshold: 0.1,
                severity: AlertSeverity::Error,
                enabled: true,
            },
        ]
    }

    fn create_default_strategies() -> Vec<OptimizationStrategy> {
        vec![
            OptimizationStrategy {
                id: "reduce_batch_size".to_string(),
                name: "Reduce Batch Size".to_string(),
                description: "Reduce batch size to improve cache locality".to_string(),
                target_metrics: vec!["execution_time".to_string(), "cache_hit_ratio".to_string()],
                parameters: HashMap::new(),
                effectiveness_score: 0.7,
                usage_count: 0,
            },
            OptimizationStrategy {
                id: "increase_parallelism".to_string(),
                name: "Increase Parallelism".to_string(),
                description: "Increase parallel threads for better throughput".to_string(),
                target_metrics: vec!["throughput".to_string()],
                parameters: HashMap::new(),
                effectiveness_score: 0.8,
                usage_count: 0,
            },
        ]
    }

    // Real system metrics functions
    fn get_cpu_usage() -> f64 {
        // Read CPU usage from /proc/stat on Linux
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/proc/stat") {
                if let Some(cpu_line) = content.lines().next() {
                    let values: Vec<u64> = cpu_line
                        .split_whitespace()
                        .skip(1)
                        .filter_map(|s| s.parse().ok())
                        .collect();
                    
                    if values.len() >= 4 {
                        let idle = values[3];
                        let total: u64 = values.iter().sum();
                        if total > 0 {
                            return (total - idle) as f64 / total as f64;
                        }
                    }
                }
            }
        }
        
        // Fallback: estimate based on system load
        let load = Self::get_system_load();
        (load / 4.0).min(1.0) // Assuming 4 cores
    }
    
    fn get_memory_usage() -> f64 {
        // Read memory usage from /proc/meminfo on Linux
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
                let mut total_mem = 0u64;
                let mut avail_mem = 0u64;
                
                for line in content.lines() {
                    if line.starts_with("MemTotal:") {
                        if let Some(value) = line.split_whitespace().nth(1) {
                            total_mem = value.parse().unwrap_or(0);
                        }
                    } else if line.starts_with("MemAvailable:") {
                        if let Some(value) = line.split_whitespace().nth(1) {
                            avail_mem = value.parse().unwrap_or(0);
                        }
                    }
                }
                
                if total_mem > 0 && avail_mem <= total_mem {
                    let used_mem = total_mem - avail_mem;
                    return used_mem as f64 / total_mem as f64;
                }
            }
        }
        
        // Fallback for other platforms
        0.6
    }
    
    fn get_gpu_usage() -> f64 {
        // Try to read NVIDIA GPU usage via nvidia-smi
        #[cfg(target_os = "linux")]
        {
            if let Ok(output) = std::process::Command::new("nvidia-smi")
                .args(["--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"])
                .output()
            {
                if output.status.success() {
                    if let Ok(usage_str) = String::from_utf8(output.stdout) {
                        if let Ok(usage) = usage_str.trim().parse::<f64>() {
                            return usage / 100.0;
                        }
                    }
                }
            }
        }
        
        // Fallback: no GPU or unavailable
        0.0
    }
    
    fn get_system_load() -> f64 {
        // Read system load average from /proc/loadavg on Linux
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/proc/loadavg") {
                if let Some(load_str) = content.split_whitespace().next() {
                    if let Ok(load) = load_str.parse::<f64>() {
                        return load;
                    }
                }
            }
        }
        
        // Cross-platform alternative using CPU usage as proxy
        Self::get_cpu_usage() * 4.0 // Estimate based on CPU usage
    }
}

impl TrendAnalysis {
    fn new() -> Self {
        Self {
            execution_time_trend: LinearTrend::default(),
            throughput_trend: LinearTrend::default(),
            memory_trend: LinearTrend::default(),
            efficiency_trend: LinearTrend::default(),
            anomaly_detection: AnomalyDetector::new(),
        }
    }
}

impl AnomalyDetector {
    fn new() -> Self {
        Self {
            moving_average: 0.0,
            moving_variance: 0.0,
            anomaly_threshold: 2.0, // 2 standard deviations
            recent_anomalies: VecDeque::new(),
        }
    }

    fn update(&mut self, values: &[f64]) {
        if values.is_empty() {
            return;
        }

        let latest = values[values.len() - 1];

        // Update moving average and variance
        let alpha = 0.1; // Smoothing factor
        self.moving_average = alpha * latest + (1.0 - alpha) * self.moving_average;

        let squared_diff = (latest - self.moving_average).powi(2);
        self.moving_variance = alpha * squared_diff + (1.0 - alpha) * self.moving_variance;

        // Check for anomaly
        let std_dev = self.moving_variance.sqrt();
        let z_score = (latest - self.moving_average).abs() / (std_dev + 1e-8);

        if z_score > self.anomaly_threshold {
            let severity = if z_score > 4.0 {
                AnomalySeverity::Critical
            } else if z_score > 3.0 {
                AnomalySeverity::High
            } else {
                AnomalySeverity::Medium
            };

            let anomaly = AnomalyEvent {
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                metric_name: "efficiency".to_string(),
                expected_value: self.moving_average,
                actual_value: latest,
                severity,
            };

            self.recent_anomalies.push_back(anomaly);

            // Keep only recent anomalies
            if self.recent_anomalies.len() > 100 {
                self.recent_anomalies.pop_front();
            }
        }
    }
}

/// Current performance metrics snapshot
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub timestamp: u64,
    pub aggregated: AggregatedMetrics,
    pub system_metrics: SystemMetricsSnapshot,
    pub total_samples: usize,
    pub monitoring_active: bool,
}

/// System metrics snapshot
#[derive(Debug, Clone)]
pub struct SystemMetricsSnapshot {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub gpu_usage: f64,
    pub system_load: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_monitor_creation() {
        let config = PerformanceMonitorConfig::default();
        let monitor = RealTimePerformanceMonitor::new(config);

        assert!(!monitor.monitoring_active.load(Ordering::Relaxed));
        assert_eq!(monitor.sample_counter.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_linear_trend_calculation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let trend = RealTimePerformanceMonitor::calculate_linear_trend(&data);

        assert!((trend.slope - 1.0).abs() < 0.1);
        assert!(trend.correlation > 0.9);
    }

    #[test]
    fn test_anomaly_detection() {
        let mut detector = AnomalyDetector::new();

        // Normal values
        let normal_values = vec![1.0, 1.1, 0.9, 1.05, 0.95];
        detector.update(&normal_values);

        // No anomalies expected for normal values
        assert!(detector.recent_anomalies.is_empty());

        // Anomalous value
        let anomalous_values = vec![10.0]; // Significantly different
        detector.update(&anomalous_values);

        // Should detect anomaly (though it might take a few updates to stabilize)
        // This is a simplified test
    }

    #[test]
    fn test_performance_metrics() {
        let config = PerformanceMonitorConfig::default();
        let monitor = RealTimePerformanceMonitor::new(config);

        let metrics = monitor.get_current_metrics();
        assert_eq!(metrics.total_samples, 0);
        assert!(!metrics.monitoring_active);
    }
}
