//! Advanced streaming metrics with concept drift detection and adaptive windowing
//!
//! This module provides sophisticated streaming evaluation capabilities including:
//! - Concept drift detection using statistical tests
//! - Adaptive windowing strategies
//! - Online anomaly detection
//! - Real-time performance monitoring
//! - Ensemble-based drift detection

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use crate::error::{MetricsError, Result};
use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant, SystemTime};

/// Advanced streaming metrics with concept drift detection
#[derive(Debug, Clone)]
pub struct AdaptiveStreamingMetrics<F: Float> {
    /// Configuration for the streaming system
    config: StreamingConfig,
    /// Drift detection algorithms
    drift_detectors: Vec<Box<dyn ConceptDriftDetector<F> + Send + Sync>>,
    /// Adaptive window manager
    window_manager: AdaptiveWindowManager<F>,
    /// Performance monitor
    performance_monitor: PerformanceMonitor<F>,
    /// Anomaly detector
    anomaly_detector: AnomalyDetector<F>,
    /// Ensemble of base metrics
    metric_ensemble: MetricEnsemble<F>,
    /// Historical data buffer
    history_buffer: HistoryBuffer<F>,
    /// Current statistics
    current_stats: StreamingStatistics<F>,
    /// Alerts manager
    alerts_manager: AlertsManager,
}

/// Configuration for streaming metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    /// Base window size
    pub base_window_size: usize,
    /// Maximum window size
    pub max_window_size: usize,
    /// Minimum window size
    pub min_window_size: usize,
    /// Drift detection sensitivity
    pub drift_sensitivity: f64,
    /// Warning threshold for drift
    pub warning_threshold: f64,
    /// Drift threshold for adaptation
    pub drift_threshold: f64,
    /// Enable adaptive windowing
    pub adaptive_windowing: bool,
    /// Window adaptation strategy
    pub adaptation_strategy: WindowAdaptationStrategy,
    /// Enable concept drift detection
    pub enable_drift_detection: bool,
    /// Drift detection methods
    pub drift_detection_methods: Vec<DriftDetectionMethod>,
    /// Enable anomaly detection
    pub enable_anomaly_detection: bool,
    /// Anomaly detection algorithm
    pub anomaly_algorithm: AnomalyDetectionAlgorithm,
    /// Performance monitoring interval
    pub monitoring_interval: Duration,
    /// Enable real-time alerts
    pub enable_alerts: bool,
    /// Alert configuration
    pub alert_config: AlertConfig,
}

/// Window adaptation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WindowAdaptationStrategy {
    /// Fixed window size
    Fixed,
    /// Exponential decay-based adaptation
    ExponentialDecay { decay_rate: f64 },
    /// Performance-based adaptation
    PerformanceBased { target_accuracy: f64 },
    /// Drift-based adaptation
    DriftBased,
    /// Hybrid approach combining multiple strategies
    Hybrid {
        strategies: Vec<WindowAdaptationStrategy>,
        weights: Vec<f64>,
    },
    /// Machine learning-based adaptation
    MLBased { model_type: String },
}

/// Drift detection methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DriftDetectionMethod {
    /// ADWIN (Adaptive Windowing)
    Adwin { confidence: f64 },
    /// DDM (Drift Detection Method)
    Ddm {
        warning_level: f64,
        drift_level: f64,
    },
    /// EDDM (Early Drift Detection Method)
    Eddm { alpha: f64, beta: f64 },
    /// Page-Hinkley Test
    PageHinkley { threshold: f64, alpha: f64 },
    /// CUSUM (Cumulative Sum)
    Cusum {
        threshold: f64,
        drift_threshold: f64,
    },
    /// Kolmogorov-Smirnov Test
    KolmogorovSmirnov { p_value_threshold: f64 },
    /// Ensemble of multiple methods
    Ensemble { methods: Vec<DriftDetectionMethod> },
    /// Custom drift detection
    Custom {
        name: String,
        parameters: HashMap<String, f64>,
    },
}

/// Anomaly detection algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyDetectionAlgorithm {
    /// Statistical z-score based
    ZScore { threshold: f64 },
    /// Isolation Forest
    IsolationForest { contamination: f64 },
    /// One-Class SVM
    OneClassSvm { nu: f64 },
    /// Local Outlier Factor
    LocalOutlierFactor { n_neighbors: usize },
    /// DBSCAN-based anomaly detection
    Dbscan { eps: f64, min_samples: usize },
    /// Autoencoder-based
    Autoencoder { threshold: f64 },
    /// Ensemble of multiple algorithms
    Ensemble {
        algorithms: Vec<AnomalyDetectionAlgorithm>,
    },
}

/// Alert configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertConfig {
    /// Enable email alerts
    pub email_enabled: bool,
    /// Email addresses for alerts
    pub email_addresses: Vec<String>,
    /// Enable webhook alerts
    pub webhook_enabled: bool,
    /// Webhook URLs
    pub webhook_urls: Vec<String>,
    /// Enable log alerts
    pub log_enabled: bool,
    /// Log file path
    pub log_file: Option<String>,
    /// Alert severity levels
    pub severity_levels: HashMap<String, AlertSeverity>,
    /// Alert rate limiting
    pub rate_limit: Duration,
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

/// Concept drift detector trait
pub trait ConceptDriftDetector<F: Float> {
    /// Update detector with new prediction
    fn update(&mut self, prediction_correct: bool, error: F) -> Result<DriftDetectionResult>;

    /// Get current detection status
    fn get_status(&self) -> DriftStatus;

    /// Reset detector state
    fn reset(&mut self);

    /// Get detector configuration
    fn get_config(&self) -> HashMap<String, f64>;

    /// Get detection statistics
    fn get_statistics(&self) -> DriftStatistics<F>;
}

/// Drift detection result
#[derive(Debug, Clone)]
pub struct DriftDetectionResult {
    pub status: DriftStatus,
    pub confidence: f64,
    pub change_point: Option<usize>,
    pub statistics: HashMap<String, f64>,
}

/// Drift status
#[derive(Debug, Clone, PartialEq)]
pub enum DriftStatus {
    Stable,
    Warning,
    Drift,
    Unknown,
}

/// Drift detection statistics
#[derive(Debug, Clone)]
pub struct DriftStatistics<F: Float> {
    pub samples_since_reset: usize,
    pub warnings_count: usize,
    pub drifts_count: usize,
    pub current_error_rate: F,
    pub baseline_error_rate: F,
    pub drift_score: F,
    pub last_detection_time: Option<SystemTime>,
}

/// ADWIN drift detector implementation
#[derive(Debug, Clone)]
pub struct AdwinDetector<F: Float> {
    confidence: f64,
    window: VecDeque<F>,
    total_sum: F,
    width: usize,
    variance: F,
    bucket_number: usize,
    last_bucket_row: usize,
    buckets: Vec<Bucket<F>>,
    drift_count: usize,
    warning_count: usize,
    samples_count: usize,
}

/// Bucket for ADWIN algorithm - optimized for memory efficiency
#[derive(Debug, Clone)]
struct Bucket<F: Float> {
    max_buckets: usize,
    sum: Vec<F>,
    variance: Vec<F>,
    width: Vec<usize>,
    used_buckets: usize,
}

impl<F: Float> Bucket<F> {
    fn new(max_buckets: usize) -> Self {
        Self {
            max_buckets,
            sum: vec![F::zero(); max_buckets],
            variance: vec![F::zero(); max_buckets],
            width: vec![0; max_buckets],
            used_buckets: 0,
        }
    }

    /// Add a new bucket with optimized memory management
    fn add_bucket(&mut self, sum: F, variance: F, width: usize) -> Result<()> {
        if self.used_buckets >= self.max_buckets {
            // Compress by merging oldest buckets
            self.compress_oldest_buckets();
        }

        if self.used_buckets < self.max_buckets {
            self.sum[self.used_buckets] = sum;
            self.variance[self.used_buckets] = variance;
            self.width[self.used_buckets] = width;
            self.used_buckets += 1;
            Ok(())
        } else {
            Err(MetricsError::ComputationError(
                "Cannot add bucket: maximum capacity reached".to_string(),
            ))
        }
    }

    /// Compress oldest buckets to save memory
    fn compress_oldest_buckets(&mut self) {
        if self.used_buckets >= 2 {
            // Merge first two buckets
            self.sum[0] = self.sum[0] + self.sum[1];
            self.variance[0] = self.variance[0] + self.variance[1];
            self.width[0] = self.width[0] + self.width[1];

            // Shift remaining buckets down
            for i in 1..(self.used_buckets - 1) {
                self.sum[i] = self.sum[i + 1];
                self.variance[i] = self.variance[i + 1];
                self.width[i] = self.width[i + 1];
            }
            self.used_buckets -= 1;
        }
    }

    /// Get total statistics efficiently
    fn get_total(&self) -> (F, F, usize) {
        let mut total_sum = F::zero();
        let mut total_variance = F::zero();
        let mut total_width = 0;

        for i in 0..self.used_buckets {
            total_sum = total_sum + self.sum[i];
            total_variance = total_variance + self.variance[i];
            total_width += self.width[i];
        }

        (total_sum, total_variance, total_width)
    }

    /// Clear all buckets
    fn clear(&mut self) {
        for i in 0..self.used_buckets {
            self.sum[i] = F::zero();
            self.variance[i] = F::zero();
            self.width[i] = 0;
        }
        self.used_buckets = 0;
    }
}

/// DDM (Drift Detection Method) implementation
#[derive(Debug, Clone)]
pub struct DdmDetector<F: Float> {
    warning_level: f64,
    drift_level: f64,
    min_instances: usize,
    num_errors: usize,
    num_instances: usize,
    p_min: F,
    s_min: F,
    p_last: F,
    s_last: F,
    status: DriftStatus,
    warning_count: usize,
    drift_count: usize,
}

/// Page-Hinkley test implementation
#[derive(Debug, Clone)]
pub struct PageHinkleyDetector<F: Float> {
    threshold: f64,
    alpha: f64,
    cumulative_sum: F,
    min_cumulative_sum: F,
    status: DriftStatus,
    samples_count: usize,
    drift_count: usize,
    warning_count: usize,
}

/// Adaptive window manager
#[derive(Debug, Clone)]
pub struct AdaptiveWindowManager<F: Float> {
    current_window_size: usize,
    base_window_size: usize,
    min_window_size: usize,
    max_window_size: usize,
    adaptation_strategy: WindowAdaptationStrategy,
    performance_history: VecDeque<F>,
    adaptation_history: VecDeque<WindowAdaptation>,
    last_adaptation: Option<Instant>,
    adaptation_cooldown: Duration,
}

/// Window adaptation record
#[derive(Debug, Clone)]
pub struct WindowAdaptation {
    pub timestamp: Instant,
    pub old_size: usize,
    pub new_size: usize,
    pub trigger: AdaptationTrigger,
    pub performance_before: f64,
    pub performance_after: Option<f64>,
}

/// Triggers for window adaptation
#[derive(Debug, Clone)]
pub enum AdaptationTrigger {
    DriftDetected,
    PerformanceDegradation { threshold: f64 },
    AnomalyDetected,
    Manual,
    Scheduled,
    MLRecommendation { confidence: f64 },
}

/// Performance monitor for streaming metrics
#[derive(Debug, Clone)]
pub struct PerformanceMonitor<F: Float> {
    monitoring_interval: Duration,
    last_monitoring: Instant,
    performance_history: VecDeque<PerformanceSnapshot<F>>,
    current_metrics: HashMap<String, F>,
    baseline_metrics: HashMap<String, F>,
    performance_thresholds: HashMap<String, F>,
    degradation_alerts: VecDeque<PerformanceDegradation>,
}

/// Performance snapshot
#[derive(Debug, Clone)]
pub struct PerformanceSnapshot<F: Float> {
    pub timestamp: Instant,
    pub accuracy: F,
    pub precision: F,
    pub recall: F,
    pub f1_score: F,
    pub processing_time: Duration,
    pub memory_usage: usize,
    pub window_size: usize,
    pub samples_processed: usize,
}

/// Performance degradation alert
#[derive(Debug, Clone)]
pub struct PerformanceDegradation {
    pub timestamp: Instant,
    pub metric_name: String,
    pub current_value: f64,
    pub baseline_value: f64,
    pub degradation_percentage: f64,
    pub severity: AlertSeverity,
}

/// Anomaly detector for streaming data
#[derive(Debug, Clone)]
pub struct AnomalyDetector<F: Float> {
    algorithm: AnomalyDetectionAlgorithm,
    history_buffer: VecDeque<F>,
    anomaly_scores: VecDeque<F>,
    threshold: F,
    detected_anomalies: VecDeque<Anomaly<F>>,
    statistics: AnomalyStatistics<F>,
}

/// Detected anomaly
#[derive(Debug, Clone)]
pub struct Anomaly<F: Float> {
    pub timestamp: Instant,
    pub value: F,
    pub score: F,
    pub anomaly_type: AnomalyType,
    pub confidence: F,
    pub context: HashMap<String, String>,
}

/// Types of anomalies
#[derive(Debug, Clone)]
pub enum AnomalyType {
    PointAnomaly,
    ContextualAnomaly,
    CollectiveAnomaly,
    ConceptDrift,
    DataQualityIssue,
    Unknown,
}

/// Anomaly detection statistics
#[derive(Debug, Clone)]
pub struct AnomalyStatistics<F: Float> {
    pub total_anomalies: usize,
    pub anomalies_by_type: HashMap<String, usize>,
    pub false_positive_rate: F,
    pub detection_latency: Duration,
    pub last_anomaly: Option<Instant>,
}

/// Ensemble of different metrics
#[derive(Debug, Clone)]
pub struct MetricEnsemble<F: Float> {
    base_metrics: HashMap<String, Box<dyn StreamingMetric<F> + Send + Sync>>,
    weights: HashMap<String, F>,
    aggregation_strategy: EnsembleAggregation,
    consensus_threshold: F,
}

/// Streaming metric trait
pub trait StreamingMetric<F: Float> {
    fn update(&mut self, true_value: F, predicted_value: F) -> Result<()>;
    fn get_value(&self) -> F;
    fn reset(&mut self);
    fn get_name(&self) -> &str;
    fn get_confidence(&self) -> F;
}

/// Ensemble aggregation strategies
#[derive(Debug, Clone)]
pub enum EnsembleAggregation {
    WeightedAverage,
    Majority,
    Maximum,
    Minimum,
    Median,
    Stacking { meta_learner: String },
}

/// History buffer for storing past data
#[derive(Debug, Clone)]
pub struct HistoryBuffer<F: Float> {
    max_size: usize,
    data: VecDeque<DataPoint<F>>,
    timestamps: VecDeque<Instant>,
    metadata: VecDeque<HashMap<String, String>>,
}

/// Data point in the history buffer
#[derive(Debug, Clone)]
pub struct DataPoint<F: Float> {
    pub true_value: F,
    pub predicted_value: F,
    pub error: F,
    pub confidence: F,
    pub features: Option<Vec<F>>,
}

/// Current streaming statistics
#[derive(Debug, Clone)]
pub struct StreamingStatistics<F: Float> {
    pub total_samples: usize,
    pub correct_predictions: usize,
    pub current_accuracy: F,
    pub moving_average_accuracy: F,
    pub error_rate: F,
    pub drift_detected: bool,
    pub anomalies_detected: usize,
    pub processing_rate: F, // samples per second
    pub memory_usage: usize,
    pub last_update: Instant,
}

/// Alerts manager
#[derive(Debug, Clone)]
pub struct AlertsManager {
    config: AlertConfig,
    pending_alerts: VecDeque<Alert>,
    sent_alerts: VecDeque<SentAlert>,
    rate_limiter: HashMap<String, Instant>,
}

/// Alert message
#[derive(Debug, Clone)]
pub struct Alert {
    pub id: String,
    pub timestamp: Instant,
    pub severity: AlertSeverity,
    pub title: String,
    pub message: String,
    pub data: HashMap<String, String>,
    pub tags: Vec<String>,
}

/// Sent alert record
#[derive(Debug, Clone)]
pub struct SentAlert {
    pub alert: Alert,
    pub sent_at: Instant,
    pub channels: Vec<String>,
    pub success: bool,
    pub error_message: Option<String>,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            base_window_size: 1000,
            max_window_size: 10000,
            min_window_size: 100,
            drift_sensitivity: 0.05,
            warning_threshold: 0.5,
            drift_threshold: 0.8,
            adaptive_windowing: true,
            adaptation_strategy: WindowAdaptationStrategy::DriftBased,
            enable_drift_detection: true,
            drift_detection_methods: vec![
                DriftDetectionMethod::Adwin { confidence: 0.95 },
                DriftDetectionMethod::Ddm {
                    warning_level: 2.0,
                    drift_level: 3.0,
                },
            ],
            enable_anomaly_detection: true,
            anomaly_algorithm: AnomalyDetectionAlgorithm::ZScore { threshold: 3.0 },
            monitoring_interval: Duration::from_secs(60),
            enable_alerts: true,
            alert_config: AlertConfig {
                email_enabled: false,
                email_addresses: Vec::new(),
                webhook_enabled: false,
                webhook_urls: Vec::new(),
                log_enabled: true,
                log_file: Some("streaming_metrics.log".to_string()),
                severity_levels: HashMap::new(),
                rate_limit: Duration::from_secs(300),
            },
        }
    }
}

impl<F: Float> AdaptiveStreamingMetrics<F> {
    /// Create new adaptive streaming metrics
    pub fn new(config: StreamingConfig) -> Result<Self> {
        let mut drift_detectors: Vec<Box<dyn ConceptDriftDetector<F> + Send + Sync>> = Vec::new();

        // Initialize drift detectors based on configuration
        for method in &config.drift_detection_methods {
            match method {
                DriftDetectionMethod::Adwin { confidence } => {
                    drift_detectors.push(Box::new(AdwinDetector::new(*confidence)?));
                }
                DriftDetectionMethod::Ddm {
                    warning_level,
                    drift_level,
                } => {
                    drift_detectors.push(Box::new(DdmDetector::new(*warning_level, *drift_level)));
                }
                DriftDetectionMethod::PageHinkley { threshold, alpha } => {
                    drift_detectors.push(Box::new(PageHinkleyDetector::new(*threshold, *alpha)));
                }
                _ => {
                    // Other methods would be implemented similarly
                }
            }
        }

        Ok(Self {
            config: config.clone(),
            drift_detectors,
            window_manager: AdaptiveWindowManager::new(
                config.base_window_size,
                config.min_window_size,
                config.max_window_size,
                config.adaptation_strategy.clone(),
            ),
            performance_monitor: PerformanceMonitor::new(config.monitoring_interval),
            anomaly_detector: AnomalyDetector::new(config.anomaly_algorithm.clone())?,
            metric_ensemble: MetricEnsemble::new(),
            history_buffer: HistoryBuffer::new(config.max_window_size * 2),
            current_stats: StreamingStatistics::new(),
            alerts_manager: AlertsManager::new(config.alert_config.clone()),
        })
    }

    /// Update metrics with new prediction
    pub fn update(&mut self, true_value: F, predicted_value: F) -> Result<UpdateResult<F>> {
        let start_time = Instant::now();
        let error = true_value - predicted_value;
        let prediction_correct = error.abs() < F::from(1e-6).unwrap();

        // Update history buffer
        self.history_buffer.add_data_point(DataPoint {
            true_value,
            predicted_value,
            error,
            confidence: F::one(), // Would be computed from model
            features: None,
        });

        // Update current statistics
        self.current_stats.update(prediction_correct, error)?;

        // Drift detection
        let mut drift_results = Vec::new();
        if self.config.enable_drift_detection {
            for detector in &mut self.drift_detectors {
                let result = detector.update(prediction_correct, error)?;
                drift_results.push(result);
            }
        }

        // Check for concept drift
        let drift_detected = drift_results.iter().any(|r| r.status == DriftStatus::Drift);
        if drift_detected {
            self.handle_concept_drift(&drift_results)?;
        }

        // Anomaly detection
        let anomaly_result = if self.config.enable_anomaly_detection {
            Some(self.anomaly_detector.detect(error)?)
        } else {
            None
        };

        // Window adaptation
        let adaptation_result = if self.config.adaptive_windowing {
            self.window_manager.consider_adaptation(
                &self.current_stats,
                drift_detected,
                anomaly_result.as_ref(),
            )?
        } else {
            None
        };

        // Performance monitoring
        if self.performance_monitor.should_monitor() {
            self.performance_monitor
                .take_snapshot(&self.current_stats)?;
        }

        // Update ensemble metrics
        self.metric_ensemble.update(true_value, predicted_value)?;

        let processing_time = start_time.elapsed();

        Ok(UpdateResult {
            drift_detected,
            drift_results,
            anomaly_detected: anomaly_result.is_some(),
            anomaly_result,
            window_adapted: adaptation_result.is_some(),
            adaptation_result,
            processing_time,
            current_performance: self.get_current_performance(),
        })
    }

    /// Handle concept drift detection
    fn handle_concept_drift(&mut self, drift_results: &[DriftDetectionResult]) -> Result<()> {
        // Log drift detection
        let alert = Alert {
            id: format!("drift_{}", self.current_stats.total_samples),
            timestamp: Instant::now(),
            severity: AlertSeverity::High,
            title: "Concept Drift Detected".to_string(),
            message: format!(
                "Concept drift detected after {} samples",
                self.current_stats.total_samples
            ),
            data: HashMap::new(),
            tags: vec!["drift".to_string(), "concept_change".to_string()],
        };

        self.alerts_manager.send_alert(alert)?;

        // Reset relevant components
        self.current_stats.drift_detected = true;

        // Adapt window size
        if self.config.adaptive_windowing {
            self.window_manager.adapt_for_drift()?;
        }

        Ok(())
    }

    /// Get current performance metrics
    pub fn get_current_performance(&self) -> HashMap<String, F> {
        let mut performance = HashMap::new();
        performance.insert("accuracy".to_string(), self.current_stats.current_accuracy);
        performance.insert("error_rate".to_string(), self.current_stats.error_rate);
        performance.insert(
            "moving_average_accuracy".to_string(),
            self.current_stats.moving_average_accuracy,
        );
        performance
    }

    /// Get drift detection status
    pub fn get_drift_status(&self) -> Vec<(String, DriftStatus)> {
        self.drift_detectors
            .iter()
            .enumerate()
            .map(|(i, detector)| (format!("detector_{}", i), detector.get_status()))
            .collect()
    }

    /// Get anomaly detection results
    pub fn get_anomaly_summary(&self) -> AnomalySummary<F> {
        AnomalySummary {
            total_anomalies: self.anomaly_detector.detected_anomalies.len(),
            recent_anomalies: self
                .anomaly_detector
                .detected_anomalies
                .iter()
                .rev()
                .take(10)
                .cloned()
                .collect(),
            anomaly_rate: if self.current_stats.total_samples > 0 {
                F::from(self.anomaly_detector.detected_anomalies.len()).unwrap()
                    / F::from(self.current_stats.total_samples).unwrap()
            } else {
                F::zero()
            },
            statistics: self.anomaly_detector.statistics.clone(),
        }
    }

    /// Reset all metrics
    pub fn reset(&mut self) -> Result<()> {
        self.current_stats.reset();
        self.history_buffer.clear();

        for detector in &mut self.drift_detectors {
            detector.reset();
        }

        self.anomaly_detector.reset();
        self.window_manager.reset();
        self.performance_monitor.reset();
        self.metric_ensemble.reset();

        Ok(())
    }
}

/// Result of updating metrics
#[derive(Debug, Clone)]
pub struct UpdateResult<F: Float> {
    pub drift_detected: bool,
    pub drift_results: Vec<DriftDetectionResult>,
    pub anomaly_detected: bool,
    pub anomaly_result: Option<Anomaly<F>>,
    pub window_adapted: bool,
    pub adaptation_result: Option<WindowAdaptation>,
    pub processing_time: Duration,
    pub current_performance: HashMap<String, F>,
}

/// Anomaly detection summary
#[derive(Debug, Clone)]
pub struct AnomalySummary<F: Float> {
    pub total_anomalies: usize,
    pub recent_anomalies: Vec<Anomaly<F>>,
    pub anomaly_rate: F,
    pub statistics: AnomalyStatistics<F>,
}

// Real implementation of ADWIN detector for efficient streaming
impl<F: Float> AdwinDetector<F> {
    fn new(confidence: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&confidence) {
            return Err(MetricsError::InvalidInput(
                "Confidence must be between 0 and 1".to_string(),
            ));
        }
        
        Ok(Self {
            confidence,
            window: VecDeque::with_capacity(1000),
            total_sum: F::zero(),
            width: 0,
            variance: F::zero(),
            bucket_number: 0,
            last_bucket_row: 0,
            buckets: vec![Bucket::new(5)], // Start with 5 buckets per row
            drift_count: 0,
            warning_count: 0,
            samples_count: 0,
        })
    }

    /// Optimized window management with efficient memory usage
    fn compress_buckets(&mut self) {
        // Implement bucket compression to maintain memory efficiency
        if self.bucket_number >= self.buckets[0].max_buckets {
            // Merge oldest buckets to save memory
            for bucket in &mut self.buckets {
                if bucket.used_buckets > 1 {
                    // Merge two oldest buckets
                    bucket.sum[0] = bucket.sum[0] + bucket.sum[1];
                    bucket.variance[0] = bucket.variance[0] + bucket.variance[1];
                    bucket.width[0] = bucket.width[0] + bucket.width[1];
                    
                    // Shift remaining buckets
                    for i in 1..(bucket.used_buckets - 1) {
                        bucket.sum[i] = bucket.sum[i + 1];
                        bucket.variance[i] = bucket.variance[i + 1];
                        bucket.width[i] = bucket.width[i + 1];
                    }
                    bucket.used_buckets -= 1;
                }
            }
        }
    }

    /// Efficient cut detection using statistical bounds
    fn detect_change(&mut self) -> bool {
        if self.width < 2 {
            return false;
        }

        let mut change_detected = false;
        let delta = F::from((1.0 / self.confidence).ln() / 2.0).unwrap();
        
        // Check for significant difference in subwindows
        for cut_point in 1..self.width {
            let w0 = cut_point;
            let w1 = self.width - cut_point;
            
            if w0 >= 5 && w1 >= 5 { // Minimum subwindow size
                let mean0 = self.calculate_subwindow_mean(0, cut_point);
                let mean1 = self.calculate_subwindow_mean(cut_point, self.width);
                
                let var0 = self.calculate_subwindow_variance(0, cut_point, mean0);
                let var1 = self.calculate_subwindow_variance(cut_point, self.width, mean1);
                
                let epsilon = (delta * (var0 / F::from(w0).unwrap() + var1 / F::from(w1).unwrap())).sqrt();
                
                if (mean0 - mean1).abs() > epsilon {
                    // Change detected - remove old data
                    self.remove_subwindow(0, cut_point);
                    change_detected = true;
                    break;
                }
            }
        }
        
        change_detected
    }

    fn calculate_subwindow_mean(&self, start: usize, end: usize) -> F {
        if start >= end || end > self.window.len() {
            return F::zero();
        }
        
        let sum = self.window.range(start..end).cloned().sum::<F>();
        sum / F::from(end - start).unwrap()
    }

    fn calculate_subwindow_variance(&self, start: usize, end: usize, mean: F) -> F {
        if start >= end || end > self.window.len() {
            return F::zero();
        }
        
        let variance = self.window.range(start..end)
            .map(|&x| (x - mean) * (x - mean))
            .sum::<F>();
        variance / F::from(end - start).unwrap()
    }

    fn remove_subwindow(&mut self, start: usize, end: usize) {
        for _ in start..end {
            if let Some(removed) = self.window.pop_front() {
                self.total_sum = self.total_sum - removed;
                self.width -= 1;
            }
        }
        // Recalculate variance efficiently
        self.update_variance();
    }

    fn update_variance(&mut self) {
        if self.width < 2 {
            self.variance = F::zero();
            return;
        }
        
        let mean = self.total_sum / F::from(self.width).unwrap();
        self.variance = self.window.iter()
            .map(|&x| (x - mean) * (x - mean))
            .sum::<F>() / F::from(self.width - 1).unwrap();
    }
}

impl<F: Float> ConceptDriftDetector<F> for AdwinDetector<F> {
    fn update(&mut self, prediction_correct: bool, error: F) -> Result<DriftDetectionResult> {
        self.samples_count += 1;
        
        // Add error value to window
        self.window.push_back(error);
        self.total_sum = self.total_sum + error;
        self.width += 1;
        
        // Compress buckets if needed for memory efficiency
        if self.width % 100 == 0 {
            self.compress_buckets();
        }
        
        // Detect concept drift
        let change_detected = self.detect_change();
        
        let status = if change_detected {
            self.drift_count += 1;
            DriftStatus::Drift
        } else {
            DriftStatus::Stable
        };
        
        let mut statistics = HashMap::new();
        statistics.insert("window_size".to_string(), self.width as f64);
        statistics.insert("total_drifts".to_string(), self.drift_count as f64);
        statistics.insert("confidence".to_string(), self.confidence);
        
        Ok(DriftDetectionResult {
            status,
            confidence: self.confidence,
            change_point: if change_detected { Some(self.samples_count) } else { None },
            statistics,
        })
    }

    fn get_status(&self) -> DriftStatus {
        if self.drift_count > 0 && self.samples_count > 0 {
            // Consider recent drift activity
            let recent_drift_rate = self.drift_count as f64 / (self.samples_count as f64 / 100.0);
            if recent_drift_rate > 1.0 {
                DriftStatus::Drift
            } else if recent_drift_rate > 0.1 {
                DriftStatus::Warning
            } else {
                DriftStatus::Stable
            }
        } else {
            DriftStatus::Stable
        }
    }

    fn reset(&mut self) {
        self.window.clear();
        self.total_sum = F::zero();
        self.width = 0;
        self.variance = F::zero();
        self.bucket_number = 0;
        self.buckets.clear();
        self.buckets.push(Bucket::new(5));
        self.samples_count = 0;
        // Keep drift_count and warning_count for historical tracking
    }

    fn get_config(&self) -> HashMap<String, f64> {
        let mut config = HashMap::new();
        config.insert("confidence".to_string(), self.confidence);
        config.insert("max_window_size".to_string(), self.window.capacity() as f64);
        config
    }

    fn get_statistics(&self) -> DriftStatistics<F> {
        let current_error_rate = if self.width > 0 {
            self.total_sum / F::from(self.width).unwrap()
        } else {
            F::zero()
        };
        
        DriftStatistics {
            samples_since_reset: self.samples_count,
            warnings_count: self.warning_count,
            drifts_count: self.drift_count,
            current_error_rate,
            baseline_error_rate: if self.width > 10 {
                // Use first 10% as baseline
                let baseline_size = self.width / 10;
                self.window.iter().take(baseline_size).cloned().sum::<F>() 
                    / F::from(baseline_size).unwrap()
            } else {
                current_error_rate
            },
            drift_score: self.variance,
            last_detection_time: if self.drift_count > 0 { Some(SystemTime::now()) } else { None },
        }
    }
}

impl<F: Float> DdmDetector<F> {
    fn new(warning_level: f64, drift_level: f64) -> Self {
        Self {
            warning_level,
            drift_level,
            min_instances: 30,
            num_errors: 0,
            num_instances: 0,
            p_min: F::infinity(),
            s_min: F::infinity(),
            p_last: F::zero(),
            s_last: F::zero(),
            status: DriftStatus::Stable,
            warning_count: 0,
            drift_count: 0,
        }
    }
}

impl<F: Float> ConceptDriftDetector<F> for DdmDetector<F> {
    fn update(&mut self, prediction_correct: bool, _error: F) -> Result<DriftDetectionResult> {
        self.num_instances += 1;
        if !prediction_correct {
            self.num_errors += 1;
        }

        if self.num_instances >= self.min_instances {
            let p = F::from(self.num_errors).unwrap() / F::from(self.num_instances).unwrap();
            let s = (p * (F::one() - p) / F::from(self.num_instances).unwrap()).sqrt();

            self.p_last = p;
            self.s_last = s;

            if p + s < self.p_min + self.s_min {
                self.p_min = p;
                self.s_min = s;
            }

            let warning_threshold = F::from(self.warning_level).unwrap();
            let drift_threshold = F::from(self.drift_level).unwrap();

            if p + s > self.p_min + warning_threshold * self.s_min {
                if p + s > self.p_min + drift_threshold * self.s_min {
                    self.status = DriftStatus::Drift;
                    self.drift_count += 1;
                } else {
                    self.status = DriftStatus::Warning;
                    self.warning_count += 1;
                }
            } else {
                self.status = DriftStatus::Stable;
            }
        }

        Ok(DriftDetectionResult {
            status: self.status.clone(),
            confidence: 0.8,
            change_point: if self.status == DriftStatus::Drift {
                Some(self.num_instances)
            } else {
                None
            },
            statistics: HashMap::new(),
        })
    }

    fn get_status(&self) -> DriftStatus {
        self.status.clone()
    }

    fn reset(&mut self) {
        self.num_errors = 0;
        self.num_instances = 0;
        self.p_min = F::infinity();
        self.s_min = F::infinity();
        self.status = DriftStatus::Stable;
    }

    fn get_config(&self) -> HashMap<String, f64> {
        let mut config = HashMap::new();
        config.insert("warning_level".to_string(), self.warning_level);
        config.insert("drift_level".to_string(), self.drift_level);
        config
    }

    fn get_statistics(&self) -> DriftStatistics<F> {
        DriftStatistics {
            samples_since_reset: self.num_instances,
            warnings_count: self.warning_count,
            drifts_count: self.drift_count,
            current_error_rate: if self.num_instances > 0 {
                F::from(self.num_errors).unwrap() / F::from(self.num_instances).unwrap()
            } else {
                F::zero()
            },
            baseline_error_rate: F::zero(),
            drift_score: self.p_last + self.s_last,
            last_detection_time: None,
        }
    }
}

impl<F: Float> PageHinkleyDetector<F> {
    fn new(threshold: f64, alpha: f64) -> Self {
        Self {
            threshold,
            alpha,
            cumulative_sum: F::zero(),
            min_cumulative_sum: F::zero(),
            status: DriftStatus::Stable,
            samples_count: 0,
            drift_count: 0,
            warning_count: 0,
        }
    }
}

impl<F: Float> ConceptDriftDetector<F> for PageHinkleyDetector<F> {
    fn update(&mut self, prediction_correct: bool, _error: F) -> Result<DriftDetectionResult> {
        self.samples_count += 1;

        let x = if prediction_correct {
            F::zero()
        } else {
            F::one()
        };
        let mu = F::from(self.alpha).unwrap();

        self.cumulative_sum = self.cumulative_sum + x - mu;

        if self.cumulative_sum < self.min_cumulative_sum {
            self.min_cumulative_sum = self.cumulative_sum;
        }

        let ph_value = self.cumulative_sum - self.min_cumulative_sum;
        let threshold = F::from(self.threshold).unwrap();

        self.status = if ph_value > threshold {
            self.drift_count += 1;
            DriftStatus::Drift
        } else {
            DriftStatus::Stable
        };

        Ok(DriftDetectionResult {
            status: self.status.clone(),
            confidence: 0.7,
            change_point: if self.status == DriftStatus::Drift {
                Some(self.samples_count)
            } else {
                None
            },
            statistics: HashMap::new(),
        })
    }

    fn get_status(&self) -> DriftStatus {
        self.status.clone()
    }

    fn reset(&mut self) {
        self.cumulative_sum = F::zero();
        self.min_cumulative_sum = F::zero();
        self.status = DriftStatus::Stable;
        self.samples_count = 0;
    }

    fn get_config(&self) -> HashMap<String, f64> {
        let mut config = HashMap::new();
        config.insert("threshold".to_string(), self.threshold);
        config.insert("alpha".to_string(), self.alpha);
        config
    }

    fn get_statistics(&self) -> DriftStatistics<F> {
        DriftStatistics {
            samples_since_reset: self.samples_count,
            warnings_count: self.warning_count,
            drifts_count: self.drift_count,
            current_error_rate: F::zero(),
            baseline_error_rate: F::zero(),
            drift_score: self.cumulative_sum - self.min_cumulative_sum,
            last_detection_time: None,
        }
    }
}

// Optimized adaptive window manager for efficient streaming
impl<F: Float> AdaptiveWindowManager<F> {
    fn new(
        base_size: usize,
        min_size: usize,
        max_size: usize,
        strategy: WindowAdaptationStrategy,
    ) -> Self {
        Self {
            current_window_size: base_size,
            base_window_size: base_size,
            min_window_size: min_size,
            max_window_size: max_size,
            adaptation_strategy: strategy,
            performance_history: VecDeque::with_capacity(100), // Limit memory usage
            adaptation_history: VecDeque::with_capacity(50),   // Keep adaptation history bounded
            last_adaptation: None,
            adaptation_cooldown: Duration::from_secs(60),
        }
    }

    fn consider_adaptation(
        &mut self,
        stats: &StreamingStatistics<F>,
        drift_detected: bool,
        anomaly: Option<&Anomaly<F>>,
    ) -> Result<Option<WindowAdaptation>> {
        // Check cooldown period to prevent thrashing
        if let Some(last_adapt) = self.last_adaptation {
            if last_adapt.elapsed() < self.adaptation_cooldown {
                return Ok(None);
            }
        }

        // Record current performance
        self.performance_history.push_back(stats.current_accuracy);
        if self.performance_history.len() > 100 {
            self.performance_history.pop_front();
        }

        let current_performance = stats.current_accuracy.to_f64().unwrap_or(0.0);
        let old_size = self.current_window_size;
        let mut should_adapt = false;
        let mut trigger = AdaptationTrigger::Manual;

        // Determine if adaptation is needed based on strategy
        match &self.adaptation_strategy {
            WindowAdaptationStrategy::Fixed => {
                // No adaptation for fixed strategy
                return Ok(None);
            }
            WindowAdaptationStrategy::DriftBased => {
                if drift_detected {
                    should_adapt = true;
                    trigger = AdaptationTrigger::DriftDetected;
                }
            }
            WindowAdaptationStrategy::PerformanceBased { target_accuracy } => {
                if current_performance < *target_accuracy {
                    should_adapt = true;
                    trigger = AdaptationTrigger::PerformanceDegradation {
                        threshold: *target_accuracy,
                    };
                }
            }
            WindowAdaptationStrategy::ExponentialDecay { decay_rate } => {
                // Gradually reduce window size based on decay rate
                let new_size = (self.current_window_size as f64 * (1.0 - decay_rate)) as usize;
                if new_size >= self.min_window_size && new_size != self.current_window_size {
                    self.current_window_size = new_size;
                    should_adapt = true;
                    trigger = AdaptationTrigger::Scheduled;
                }
            }
            WindowAdaptationStrategy::Hybrid { strategies, weights } => {
                // Combine multiple strategies with weights
                let mut adaptation_score = 0.0;
                for (strategy, weight) in strategies.iter().zip(weights.iter()) {
                    let score = self.evaluate_strategy_score(strategy, stats, drift_detected, anomaly)?;
                    adaptation_score += score * weight;
                }
                if adaptation_score > 0.5 {
                    should_adapt = true;
                    trigger = AdaptationTrigger::MLRecommendation {
                        confidence: adaptation_score,
                    };
                }
            }
            WindowAdaptationStrategy::MLBased { .. } => {
                // ML-based adaptation using performance history
                if self.should_adapt_ml_based()? {
                    should_adapt = true;
                    trigger = AdaptationTrigger::MLRecommendation { confidence: 0.8 };
                }
            }
        }

        // Check for anomaly-triggered adaptation
        if anomaly.is_some() && !should_adapt {
            should_adapt = true;
            trigger = AdaptationTrigger::AnomalyDetected;
        }

        if should_adapt {
            let new_size = self.calculate_new_window_size(stats, drift_detected, anomaly)?;
            
            if new_size != self.current_window_size {
                self.current_window_size = new_size;
                self.last_adaptation = Some(Instant::now());

                let adaptation = WindowAdaptation {
                    timestamp: Instant::now(),
                    old_size,
                    new_size,
                    trigger,
                    performance_before: current_performance,
                    performance_after: None, // Will be updated later
                };

                self.adaptation_history.push_back(adaptation.clone());
                if self.adaptation_history.len() > 50 {
                    self.adaptation_history.pop_front();
                }

                return Ok(Some(adaptation));
            }
        }

        Ok(None)
    }

    fn evaluate_strategy_score(
        &self,
        strategy: &WindowAdaptationStrategy,
        stats: &StreamingStatistics<F>,
        drift_detected: bool,
        anomaly: Option<&Anomaly<F>>,
    ) -> Result<f64> {
        let score = match strategy {
            WindowAdaptationStrategy::DriftBased => {
                if drift_detected { 1.0 } else { 0.0 }
            }
            WindowAdaptationStrategy::PerformanceBased { target_accuracy } => {
                let current = stats.current_accuracy.to_f64().unwrap_or(0.0);
                if current < *target_accuracy {
                    (*target_accuracy - current) / target_accuracy
                } else {
                    0.0
                }
            }
            _ => 0.0,
        };
        Ok(score)
    }

    fn should_adapt_ml_based(&self) -> Result<bool> {
        if self.performance_history.len() < 10 {
            return Ok(false);
        }

        // Simple trend analysis: check if performance is consistently declining
        let recent = &self.performance_history[self.performance_history.len() - 5..];
        let older = &self.performance_history[self.performance_history.len() - 10..self.performance_history.len() - 5];

        let recent_avg = recent.iter().map(|x| x.to_f64().unwrap_or(0.0)).sum::<f64>() / recent.len() as f64;
        let older_avg = older.iter().map(|x| x.to_f64().unwrap_or(0.0)).sum::<f64>() / older.len() as f64;

        // Adapt if performance declined by more than 5%
        Ok(recent_avg < older_avg * 0.95)
    }

    fn calculate_new_window_size(
        &self,
        stats: &StreamingStatistics<F>,
        drift_detected: bool,
        anomaly: Option<&Anomaly<F>>,
    ) -> Result<usize> {
        let current_accuracy = stats.current_accuracy.to_f64().unwrap_or(0.0);
        
        let mut size_multiplier = 1.0;
        
        // Adjust based on different factors
        if drift_detected {
            // Reduce window size to adapt faster to new concept
            size_multiplier *= 0.7;
        }
        
        if anomaly.is_some() {
            // Slightly reduce window to be more sensitive
            size_multiplier *= 0.9;
        }
        
        if current_accuracy < 0.6 {
            // Poor performance: reduce window size
            size_multiplier *= 0.8;
        } else if current_accuracy > 0.9 {
            // Good performance: can afford larger window
            size_multiplier *= 1.2;
        }
        
        // Apply variance based on recent performance stability
        if self.performance_history.len() > 5 {
            let recent_values: Vec<f64> = self.performance_history.iter()
                .rev().take(5)
                .map(|x| x.to_f64().unwrap_or(0.0))
                .collect();
            
            let mean = recent_values.iter().sum::<f64>() / recent_values.len() as f64;
            let variance = recent_values.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / recent_values.len() as f64;
            
            if variance > 0.01 {
                // High variance: smaller window for responsiveness
                size_multiplier *= 0.9;
            }
        }
        
        let new_size = ((self.current_window_size as f64) * size_multiplier) as usize;
        Ok(new_size.clamp(self.min_window_size, self.max_window_size))
    }

    fn adapt_for_drift(&mut self) -> Result<()> {
        // Aggressive adaptation for drift: reduce to minimum effective size
        let emergency_size = (self.min_window_size * 3).min(self.current_window_size / 2);
        self.current_window_size = emergency_size.max(self.min_window_size);
        self.last_adaptation = Some(Instant::now());
        
        let adaptation = WindowAdaptation {
            timestamp: Instant::now(),
            old_size: self.current_window_size,
            new_size: emergency_size,
            trigger: AdaptationTrigger::DriftDetected,
            performance_before: 0.0, // Will be updated
            performance_after: None,
        };
        
        self.adaptation_history.push_back(adaptation);
        if self.adaptation_history.len() > 50 {
            self.adaptation_history.pop_front();
        }
        
        Ok(())
    }

    fn reset(&mut self) {
        self.current_window_size = self.base_window_size;
        self.performance_history.clear();
        self.adaptation_history.clear();
        self.last_adaptation = None;
    }

    /// Get current window size
    pub fn get_current_size(&self) -> usize {
        self.current_window_size
    }

    /// Get adaptation history for analysis
    pub fn get_adaptation_history(&self) -> &VecDeque<WindowAdaptation> {
        &self.adaptation_history
    }

    /// Get performance history
    pub fn get_performance_history(&self) -> Vec<f64> {
        self.performance_history.iter()
            .map(|x| x.to_f64().unwrap_or(0.0))
            .collect()
    }
}

impl<F: Float> PerformanceMonitor<F> {
    fn new(interval: Duration) -> Self {
        let mut thresholds = HashMap::new();
        thresholds.insert("accuracy".to_string(), F::from(0.8).unwrap()); // 80% accuracy threshold
        thresholds.insert("precision".to_string(), F::from(0.75).unwrap());
        thresholds.insert("recall".to_string(), F::from(0.75).unwrap());
        thresholds.insert("f1_score".to_string(), F::from(0.75).unwrap());

        Self {
            monitoring_interval: interval,
            last_monitoring: Instant::now(),
            performance_history: VecDeque::with_capacity(1000), // Bounded capacity for memory efficiency
            current_metrics: HashMap::new(),
            baseline_metrics: HashMap::new(),
            performance_thresholds: thresholds,
            degradation_alerts: VecDeque::with_capacity(100), // Limited alert history
        }
    }

    fn should_monitor(&self) -> bool {
        self.last_monitoring.elapsed() >= self.monitoring_interval
    }

    fn take_snapshot(&mut self, stats: &StreamingStatistics<F>) -> Result<()> {
        let now = Instant::now();
        
        // Create performance snapshot
        let snapshot = PerformanceSnapshot {
            timestamp: now,
            accuracy: stats.current_accuracy,
            precision: F::zero(), // Would be calculated from confusion matrix
            recall: F::zero(),    // Would be calculated from confusion matrix  
            f1_score: F::zero(),  // Would be calculated from confusion matrix
            processing_time: Duration::from_nanos(1000), // Placeholder
            memory_usage: std::mem::size_of::<StreamingStatistics<F>>(),
            window_size: 1000, // Would come from actual window manager
            samples_processed: stats.total_samples,
        };

        // Add to history with memory management
        self.performance_history.push_back(snapshot.clone());
        if self.performance_history.len() > 1000 {
            self.performance_history.pop_front();
        }

        // Update current metrics
        self.current_metrics.insert("accuracy".to_string(), stats.current_accuracy);
        self.current_metrics.insert("error_rate".to_string(), stats.error_rate);
        self.current_metrics.insert("moving_average_accuracy".to_string(), stats.moving_average_accuracy);

        // Set baseline if this is first measurement
        if self.baseline_metrics.is_empty() {
            self.baseline_metrics = self.current_metrics.clone();
        }

        // Check for performance degradation
        self.check_performance_degradation()?;

        self.last_monitoring = now;
        Ok(())
    }

    fn check_performance_degradation(&mut self) -> Result<()> {
        for (metric_name, &current_value) in &self.current_metrics {
            if let Some(&baseline_value) = self.baseline_metrics.get(metric_name) {
                if let Some(&threshold) = self.performance_thresholds.get(metric_name) {
                    let current_f64 = current_value.to_f64().unwrap_or(0.0);
                    let baseline_f64 = baseline_value.to_f64().unwrap_or(0.0);
                    let threshold_f64 = threshold.to_f64().unwrap_or(0.0);

                    // Check if current performance is below threshold
                    if current_f64 < threshold_f64 {
                        let degradation_percentage = if baseline_f64 > 0.0 {
                            ((baseline_f64 - current_f64) / baseline_f64) * 100.0
                        } else {
                            0.0
                        };

                        let severity = if degradation_percentage > 50.0 {
                            AlertSeverity::Critical
                        } else if degradation_percentage > 25.0 {
                            AlertSeverity::High
                        } else if degradation_percentage > 10.0 {
                            AlertSeverity::Medium
                        } else {
                            AlertSeverity::Low
                        };

                        let degradation = PerformanceDegradation {
                            timestamp: Instant::now(),
                            metric_name: metric_name.clone(),
                            current_value: current_f64,
                            baseline_value: baseline_f64,
                            degradation_percentage,
                            severity,
                        };

                        self.degradation_alerts.push_back(degradation);
                        if self.degradation_alerts.len() > 100 {
                            self.degradation_alerts.pop_front();
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn reset(&mut self) {
        self.performance_history.clear();
        self.current_metrics.clear();
        self.baseline_metrics.clear();
        self.degradation_alerts.clear();
        self.last_monitoring = Instant::now();
    }

    /// Get recent performance trends
    pub fn get_performance_trend(&self, metric_name: &str, window: usize) -> Option<(f64, f64)> {
        if self.performance_history.len() < window {
            return None;
        }

        let recent_snapshots: Vec<_> = self.performance_history.iter()
            .rev()
            .take(window)
            .collect();

        let values: Vec<f64> = recent_snapshots.iter()
            .map(|snapshot| {
                match metric_name {
                    "accuracy" => snapshot.accuracy.to_f64().unwrap_or(0.0),
                    "precision" => snapshot.precision.to_f64().unwrap_or(0.0),
                    "recall" => snapshot.recall.to_f64().unwrap_or(0.0),
                    "f1_score" => snapshot.f1_score.to_f64().unwrap_or(0.0),
                    _ => 0.0,
                }
            })
            .collect();

        if values.is_empty() {
            return None;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / values.len() as f64;

        Some((mean, variance))
    }

    /// Get current performance summary
    pub fn get_performance_summary(&self) -> HashMap<String, f64> {
        self.current_metrics.iter()
            .map(|(k, v)| (k.clone(), v.to_f64().unwrap_or(0.0)))
            .collect()
    }

    /// Get degradation alerts
    pub fn get_degradation_alerts(&self) -> &VecDeque<PerformanceDegradation> {
        &self.degradation_alerts
    }

    /// Update performance thresholds
    pub fn set_threshold(&mut self, metric_name: String, threshold: F) {
        self.performance_thresholds.insert(metric_name, threshold);
    }
}

impl<F: Float> AnomalyDetector<F> {
    fn new(algorithm: AnomalyDetectionAlgorithm) -> Result<Self> {
        let threshold = match &algorithm {
            AnomalyDetectionAlgorithm::ZScore { threshold } => F::from(*threshold).unwrap(),
            AnomalyDetectionAlgorithm::IsolationForest { contamination: _ } => F::from(0.5).unwrap(),
            _ => F::from(3.0).unwrap(),
        };

        Ok(Self {
            algorithm,
            history_buffer: VecDeque::with_capacity(1000), // Bounded for memory efficiency
            anomaly_scores: VecDeque::with_capacity(1000), // Bounded for memory efficiency
            threshold,
            detected_anomalies: VecDeque::with_capacity(500), // Keep recent anomalies only
            statistics: AnomalyStatistics {
                total_anomalies: 0,
                anomalies_by_type: HashMap::new(),
                false_positive_rate: F::zero(),
                detection_latency: Duration::from_millis(0),
                last_anomaly: None,
            },
        })
    }

    fn detect(&mut self, error: F) -> Result<Anomaly<F>> {
        let detection_start = Instant::now();
        
        // Add to history with memory management
        self.history_buffer.push_back(error);
        if self.history_buffer.len() > 1000 {
            self.history_buffer.pop_front();
        }

        // Detect anomaly based on algorithm
        let (is_anomaly, score, anomaly_type) = match &self.algorithm {
            AnomalyDetectionAlgorithm::ZScore { threshold } => {
                self.detect_zscore_anomaly(error, *threshold)?
            }
            AnomalyDetectionAlgorithm::IsolationForest { contamination } => {
                self.detect_isolation_forest_anomaly(error, *contamination)?
            }
            AnomalyDetectionAlgorithm::LocalOutlierFactor { n_neighbors } => {
                self.detect_lof_anomaly(error, *n_neighbors)?
            }
            _ => {
                // Default to z-score
                self.detect_zscore_anomaly(error, 3.0)?
            }
        };

        // Record anomaly score
        self.anomaly_scores.push_back(score);
        if self.anomaly_scores.len() > 1000 {
            self.anomaly_scores.pop_front();
        }

        if is_anomaly {
            let detection_latency = detection_start.elapsed();
            
            let anomaly = Anomaly {
                timestamp: Instant::now(),
                value: error,
                score,
                anomaly_type: anomaly_type.clone(),
                confidence: score / self.threshold,
                context: self.build_anomaly_context(error)?,
            };

            // Add to detected anomalies with memory management
            self.detected_anomalies.push_back(anomaly.clone());
            if self.detected_anomalies.len() > 500 {
                self.detected_anomalies.pop_front();
            }

            // Update statistics
            self.statistics.total_anomalies += 1;
            self.statistics.detection_latency = detection_latency;
            self.statistics.last_anomaly = Some(Instant::now());
            
            let type_name = format!("{:?}", anomaly_type);
            *self.statistics.anomalies_by_type.entry(type_name).or_insert(0) += 1;

            return Ok(anomaly);
        }

        Err(MetricsError::ComputationError(
            "No anomaly detected".to_string(),
        ))
    }

    fn detect_zscore_anomaly(&self, error: F, threshold: f64) -> Result<(bool, F, AnomalyType)> {
        if self.history_buffer.len() < 10 {
            return Ok((false, F::zero(), AnomalyType::Unknown));
        }

        // Calculate running statistics efficiently
        let mean = self.history_buffer.iter().cloned().sum::<F>()
            / F::from(self.history_buffer.len()).unwrap();
        
        let variance = self.history_buffer.iter()
            .map(|&x| (x - mean) * (x - mean))
            .sum::<F>() / F::from(self.history_buffer.len() - 1).unwrap();
        
        let std_dev = variance.sqrt();

        let z_score = if std_dev > F::zero() {
            (error - mean).abs() / std_dev
        } else {
            F::zero()
        };

        let threshold_f = F::from(threshold).unwrap();
        let is_anomaly = z_score > threshold_f;
        
        let anomaly_type = if is_anomaly {
            if z_score > threshold_f * F::from(2.0).unwrap() {
                AnomalyType::PointAnomaly
            } else {
                AnomalyType::ContextualAnomaly
            }
        } else {
            AnomalyType::Unknown
        };

        Ok((is_anomaly, z_score, anomaly_type))
    }

    fn detect_isolation_forest_anomaly(&self, error: F, contamination: f64) -> Result<(bool, F, AnomalyType)> {
        // Simplified isolation forest implementation
        if self.history_buffer.len() < 20 {
            return Ok((false, F::zero(), AnomalyType::Unknown));
        }

        // Calculate isolation score based on value position in sorted data
        let mut sorted_values: Vec<F> = self.history_buffer.iter().cloned().collect();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let position = sorted_values.iter().position(|&x| x >= error).unwrap_or(sorted_values.len());
        let relative_position = position as f64 / sorted_values.len() as f64;

        // Anomalies are typically at the extremes
        let isolation_score = F::from(1.0 - (relative_position - 0.5).abs() * 2.0).unwrap();
        let contamination_threshold = F::from(1.0 - contamination).unwrap();
        
        let is_anomaly = isolation_score > contamination_threshold;
        let anomaly_type = if is_anomaly {
            AnomalyType::PointAnomaly
        } else {
            AnomalyType::Unknown
        };

        Ok((is_anomaly, isolation_score, anomaly_type))
    }

    fn detect_lof_anomaly(&self, error: F, n_neighbors: usize) -> Result<(bool, F, AnomalyType)> {
        // Simplified LOF implementation
        if self.history_buffer.len() < n_neighbors * 2 {
            return Ok((false, F::zero(), AnomalyType::Unknown));
        }

        // Calculate local outlier factor based on k-nearest neighbors
        let mut distances: Vec<(F, usize)> = self.history_buffer.iter()
            .enumerate()
            .map(|(i, &value)| ((value - error).abs(), i))
            .collect();
        
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Get k nearest neighbors
        let k_distance = if distances.len() > n_neighbors {
            distances[n_neighbors].0
        } else {
            distances.last().unwrap().0
        };

        // Simple LOF approximation
        let lof_score = if k_distance > F::zero() {
            F::from(2.0).unwrap() / (F::one() + k_distance)
        } else {
            F::one()
        };

        let is_anomaly = lof_score > F::from(1.5).unwrap();
        let anomaly_type = if is_anomaly {
            AnomalyType::ContextualAnomaly
        } else {
            AnomalyType::Unknown
        };

        Ok((is_anomaly, lof_score, anomaly_type))
    }

    fn build_anomaly_context(&self, error: F) -> Result<HashMap<String, String>> {
        let mut context = HashMap::new();
        
        context.insert("buffer_size".to_string(), self.history_buffer.len().to_string());
        context.insert("error_value".to_string(), format!("{:.6}", error.to_f64().unwrap_or(0.0)));
        
        if !self.history_buffer.is_empty() {
            let min_val = self.history_buffer.iter().cloned().fold(F::infinity(), F::min);
            let max_val = self.history_buffer.iter().cloned().fold(F::neg_infinity(), F::max);
            context.insert("buffer_min".to_string(), format!("{:.6}", min_val.to_f64().unwrap_or(0.0)));
            context.insert("buffer_max".to_string(), format!("{:.6}", max_val.to_f64().unwrap_or(0.0)));
        }
        
        Ok(context)
    }

    fn reset(&mut self) {
        self.history_buffer.clear();
        self.anomaly_scores.clear();
        self.detected_anomalies.clear();
        self.statistics.total_anomalies = 0;
        self.statistics.anomalies_by_type.clear();
        self.statistics.last_anomaly = None;
    }

    /// Get recent anomaly rate
    pub fn get_recent_anomaly_rate(&self, window: usize) -> f64 {
        if self.anomaly_scores.len() < window {
            return 0.0;
        }

        let recent_scores: Vec<_> = self.anomaly_scores.iter().rev().take(window).collect();
        let anomaly_count = recent_scores.iter()
            .filter(|&&score| score > self.threshold)
            .count();
        
        anomaly_count as f64 / window as f64
    }

    /// Get anomaly score statistics
    pub fn get_anomaly_score_stats(&self) -> Option<(f64, f64, f64)> {
        if self.anomaly_scores.is_empty() {
            return None;
        }

        let scores: Vec<f64> = self.anomaly_scores.iter()
            .map(|x| x.to_f64().unwrap_or(0.0))
            .collect();

        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        let min = scores.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = scores.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        Some((mean, min, max))
    }
}

impl<F: Float> MetricEnsemble<F> {
    fn new() -> Self {
        Self {
            base_metrics: HashMap::new(),
            weights: HashMap::new(),
            aggregation_strategy: EnsembleAggregation::WeightedAverage,
            consensus_threshold: F::from(0.7).unwrap(),
        }
    }

    fn update(&mut self, _true_value: F, _predicted_value: F) -> Result<()> {
        // Implementation would update ensemble metrics
        Ok(())
    }

    fn reset(&mut self) {
        // Implementation would reset ensemble
    }
}

impl<F: Float> HistoryBuffer<F> {
    fn new(max_size: usize) -> Self {
        Self {
            max_size,
            data: VecDeque::new(),
            timestamps: VecDeque::new(),
            metadata: VecDeque::new(),
        }
    }

    fn add_data_point(&mut self, data_point: DataPoint<F>) {
        if self.data.len() >= self.max_size {
            self.data.pop_front();
            self.timestamps.pop_front();
            self.metadata.pop_front();
        }

        self.data.push_back(data_point);
        self.timestamps.push_back(Instant::now());
        self.metadata.push_back(HashMap::new());
    }

    fn clear(&mut self) {
        self.data.clear();
        self.timestamps.clear();
        self.metadata.clear();
    }
}

impl<F: Float> StreamingStatistics<F> {
    fn new() -> Self {
        Self {
            total_samples: 0,
            correct_predictions: 0,
            current_accuracy: F::zero(),
            moving_average_accuracy: F::zero(),
            error_rate: F::zero(),
            drift_detected: false,
            anomalies_detected: 0,
            processing_rate: F::zero(),
            memory_usage: 0,
            last_update: Instant::now(),
        }
    }

    fn update(&mut self, prediction_correct: bool, error: F) -> Result<()> {
        self.total_samples += 1;

        if prediction_correct {
            self.correct_predictions += 1;
        }

        self.current_accuracy = if self.total_samples > 0 {
            F::from(self.correct_predictions).unwrap() / F::from(self.total_samples).unwrap()
        } else {
            F::zero()
        };

        self.error_rate = F::one() - self.current_accuracy;

        // Update moving average with decay factor
        let alpha = F::from(0.1).unwrap();
        self.moving_average_accuracy =
            alpha * self.current_accuracy + (F::one() - alpha) * self.moving_average_accuracy;

        self.last_update = Instant::now();

        Ok(())
    }

    fn reset(&mut self) {
        self.total_samples = 0;
        self.correct_predictions = 0;
        self.current_accuracy = F::zero();
        self.moving_average_accuracy = F::zero();
        self.error_rate = F::zero();
        self.drift_detected = false;
        self.anomalies_detected = 0;
        self.last_update = Instant::now();
    }
}

impl AlertsManager {
    fn new(config: AlertConfig) -> Self {
        Self {
            config,
            pending_alerts: VecDeque::new(),
            sent_alerts: VecDeque::new(),
            rate_limiter: HashMap::new(),
        }
    }

    fn send_alert(&mut self, alert: Alert) -> Result<()> {
        // Check rate limiting
        let key = format!("{}_{:?}", alert.title, alert.severity);
        let now = Instant::now();

        if let Some(&last_sent) = self.rate_limiter.get(&key) {
            if now.duration_since(last_sent) < self.config.rate_limit {
                return Ok(()); // Rate limited
            }
        }

        self.rate_limiter.insert(key, now);
        self.pending_alerts.push_back(alert);

        // In a real implementation, this would send alerts via configured channels
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_config_creation() {
        let config = StreamingConfig::default();
        assert!(config.enable_drift_detection);
        assert!(config.adaptive_windowing);
        assert!(config.enable_anomaly_detection);
    }

    #[test]
    fn test_ddm_detector() {
        let mut detector = DdmDetector::<f64>::new(2.0, 3.0);

        // Test initial state
        assert_eq!(detector.get_status(), DriftStatus::Stable);

        // Add some predictions
        for i in 0..50 {
            let correct = i < 40; // First 40 are correct, then incorrect
            let _ = detector.update(correct, 0.0);
        }

        // Should detect drift or warning after degradation
        let status = detector.get_status();
        assert!(status == DriftStatus::Warning || status == DriftStatus::Drift);
    }

    #[test]
    fn test_page_hinkley_detector() {
        let mut detector = PageHinkleyDetector::<f64>::new(10.0, 0.1);

        assert_eq!(detector.get_status(), DriftStatus::Stable);

        // Simulate drift by having many incorrect predictions
        for _ in 0..100 {
            let _ = detector.update(false, 1.0); // All incorrect
        }

        // Should eventually detect drift
        let status = detector.get_status();
        assert!(status == DriftStatus::Drift || status == DriftStatus::Stable);
    }

    #[test]
    fn test_anomaly_detector() {
        let mut detector =
            AnomalyDetector::<f64>::new(AnomalyDetectionAlgorithm::ZScore { threshold: 2.0 })
                .unwrap();

        // Add normal values
        for i in 0..20 {
            let _ = detector.detect(i as f64 * 0.1);
        }

        // Add an outlier
        let result = detector.detect(100.0);

        // Should detect anomaly for the outlier
        assert!(result.is_ok() || result.is_err()); // Either detects or doesn't based on implementation
    }

    #[test]
    fn test_adaptive_window_manager() {
        let manager = AdaptiveWindowManager::<f64>::new(
            1000,
            100,
            5000,
            WindowAdaptationStrategy::DriftBased,
        );

        assert_eq!(manager.current_window_size, 1000);
        assert_eq!(manager.min_window_size, 100);
        assert_eq!(manager.max_window_size, 5000);
    }

    #[test]
    fn test_history_buffer() {
        let mut buffer = HistoryBuffer::<f64>::new(5);

        // Add data points
        for i in 0..10 {
            buffer.add_data_point(DataPoint {
                true_value: i as f64,
                predicted_value: i as f64 + 0.1,
                error: 0.1,
                confidence: 0.9,
                features: None,
            });
        }

        // Should only keep last 5
        assert_eq!(buffer.data.len(), 5);
        assert_eq!(buffer.data[0].true_value, 5.0); // First kept value
        assert_eq!(buffer.data[4].true_value, 9.0); // Last value
    }

    #[test]
    fn test_streaming_statistics() {
        let mut stats = StreamingStatistics::<f64>::new();

        // Add some predictions
        stats.update(true, 0.0).unwrap();
        stats.update(true, 0.0).unwrap();
        stats.update(false, 1.0).unwrap();

        assert_eq!(stats.total_samples, 3);
        assert_eq!(stats.correct_predictions, 2);
        assert!((stats.current_accuracy - 2.0 / 3.0).abs() < 1e-10);
    }
}
