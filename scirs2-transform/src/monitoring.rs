//! Production monitoring with drift detection and model degradation alerts
//!
//! This module provides comprehensive monitoring capabilities for transformation
//! pipelines in production environments, including data drift detection,
//! performance monitoring, and automated alerting.

use crate::error::{Result, TransformError};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[cfg(feature = "monitoring")]
use prometheus::{Counter, Gauge, Histogram, Registry};
#[cfg(feature = "monitoring")]
use serde_json::Value;

/// Drift detection methods
#[derive(Debug, Clone, PartialEq)]
pub enum DriftMethod {
    /// Kolmogorov-Smirnov test for continuous features
    KolmogorovSmirnov,
    /// Chi-square test for categorical features
    ChiSquare,
    /// Population Stability Index (PSI)
    PopulationStabilityIndex,
    /// Maximum Mean Discrepancy (MMD)
    MaximumMeanDiscrepancy,
    /// Wasserstein distance
    WassersteinDistance,
}

/// Data drift detection result
#[derive(Debug, Clone)]
pub struct DriftDetectionResult {
    /// Feature name or index
    pub feature_name: String,
    /// Drift detection method used
    pub method: DriftMethod,
    /// Test statistic value
    pub statistic: f64,
    /// P-value (if applicable)
    pub p_value: Option<f64>,
    /// Whether drift is detected
    pub is_drift_detected: bool,
    /// Severity level (0.0 = no drift, 1.0 = severe drift)
    pub severity: f64,
    /// Timestamp of detection
    pub timestamp: u64,
}

/// Performance degradation metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Processing time in milliseconds
    pub processing_time_ms: f64,
    /// Memory usage in MB
    pub memory_usage_mb: f64,
    /// Error rate (0.0 to 1.0)
    pub error_rate: f64,
    /// Throughput (samples per second)
    pub throughput: f64,
    /// Data quality score (0.0 to 1.0)
    pub data_quality_score: f64,
    /// Timestamp
    pub timestamp: u64,
}

/// Alert configuration
#[derive(Debug, Clone)]
pub struct AlertConfig {
    /// Drift detection threshold
    pub drift_threshold: f64,
    /// Performance degradation threshold
    pub performance_threshold: f64,
    /// Error rate threshold
    pub error_rate_threshold: f64,
    /// Memory usage threshold in MB
    pub memory_threshold_mb: f64,
    /// Alert cooldown period in seconds
    pub cooldown_seconds: u64,
}

impl Default for AlertConfig {
    fn default() -> Self {
        AlertConfig {
            drift_threshold: 0.05,
            performance_threshold: 2.0, // 2x baseline
            error_rate_threshold: 0.05, // 5%
            memory_threshold_mb: 1000.0, // 1GB
            cooldown_seconds: 300, // 5 minutes
        }
    }
}

/// Alert types
#[derive(Debug, Clone)]
pub enum AlertType {
    DataDrift { feature: String, severity: f64 },
    PerformanceDegradation { metric: String, value: f64 },
    HighErrorRate { rate: f64 },
    MemoryExhaustion { usage_mb: f64 },
    DataQualityIssue { score: f64 },
}

/// Production monitoring system
pub struct TransformationMonitor {
    /// Reference data for drift detection
    reference_data: Option<Array2<f64>>,
    /// Feature names
    feature_names: Vec<String>,
    /// Drift detection methods per feature
    drift_methods: HashMap<String, DriftMethod>,
    /// Historical performance metrics
    performance_history: VecDeque<PerformanceMetrics>,
    /// Historical drift results
    drift_history: VecDeque<DriftDetectionResult>,
    /// Alert configuration
    alert_config: AlertConfig,
    /// Last alert timestamps (for cooldown)
    last_alert_times: HashMap<String, u64>,
    /// Baseline performance metrics
    baseline_metrics: Option<PerformanceMetrics>,
    /// Prometheus metrics registry
    #[cfg(feature = "monitoring")]
    metrics_registry: Registry,
    /// Prometheus counters and gauges
    #[cfg(feature = "monitoring")]
    prometheus_metrics: PrometheusMetrics,
}

#[cfg(feature = "monitoring")]
struct PrometheusMetrics {
    drift_detections: Counter,
    processing_time: Histogram,
    memory_usage: Gauge,
    error_rate: Gauge,
    throughput: Gauge,
    data_quality: Gauge,
}

impl TransformationMonitor {
    /// Create a new transformation monitor
    pub fn new() -> Result<Self> {
        #[cfg(feature = "monitoring")]
        let metrics_registry = Registry::new();
        
        #[cfg(feature = "monitoring")]
        let prometheus_metrics = PrometheusMetrics {
            drift_detections: Counter::new("transform_drift_detections_total", "Total number of drift detections")
                .map_err(|e| TransformError::ComputationError(format!("Failed to create counter: {}", e)))?,
            processing_time: Histogram::new("transform_processing_time_seconds", "Processing time in seconds")
                .map_err(|e| TransformError::ComputationError(format!("Failed to create histogram: {}", e)))?,
            memory_usage: Gauge::new("transform_memory_usage_mb", "Memory usage in MB")
                .map_err(|e| TransformError::ComputationError(format!("Failed to create gauge: {}", e)))?,
            error_rate: Gauge::new("transform_error_rate", "Error rate")
                .map_err(|e| TransformError::ComputationError(format!("Failed to create gauge: {}", e)))?,
            throughput: Gauge::new("transform_throughput_samples_per_second", "Throughput in samples per second")
                .map_err(|e| TransformError::ComputationError(format!("Failed to create gauge: {}", e)))?,
            data_quality: Gauge::new("transform_data_quality_score", "Data quality score")
                .map_err(|e| TransformError::ComputationError(format!("Failed to create gauge: {}", e)))?,
        };

        #[cfg(feature = "monitoring")]
        {
            metrics_registry.register(Box::new(prometheus_metrics.drift_detections.clone()))?;
            metrics_registry.register(Box::new(prometheus_metrics.processing_time.clone()))?;
            metrics_registry.register(Box::new(prometheus_metrics.memory_usage.clone()))?;
            metrics_registry.register(Box::new(prometheus_metrics.error_rate.clone()))?;
            metrics_registry.register(Box::new(prometheus_metrics.throughput.clone()))?;
            metrics_registry.register(Box::new(prometheus_metrics.data_quality.clone()))?;
        }

        Ok(TransformationMonitor {
            reference_data: None,
            feature_names: Vec::new(),
            drift_methods: HashMap::new(),
            performance_history: VecDeque::with_capacity(1000),
            drift_history: VecDeque::with_capacity(1000),
            alert_config: AlertConfig::default(),
            last_alert_times: HashMap::new(),
            baseline_metrics: None,
            #[cfg(feature = "monitoring")]
            metrics_registry,
            #[cfg(feature = "monitoring")]
            prometheus_metrics,
        })
    }

    /// Set reference data for drift detection
    pub fn set_reference_data(&mut self, data: Array2<f64>, feature_names: Option<Vec<String>>) -> Result<()> {
        self.reference_data = Some(data.clone());
        
        if let Some(names) = feature_names {
            if names.len() != data.ncols() {
                return Err(TransformError::InvalidInput(
                    "Number of feature names must match number of columns".to_string(),
                ));
            }
            self.feature_names = names;
        } else {
            self.feature_names = (0..data.ncols())
                .map(|i| format!("feature_{}", i))
                .collect();
        }

        // Set default drift detection methods
        for feature_name in &self.feature_names {
            self.drift_methods.insert(
                feature_name.clone(),
                DriftMethod::KolmogorovSmirnov,
            );
        }

        Ok(())
    }

    /// Configure drift detection method for a specific feature
    pub fn set_drift_method(&mut self, feature_name: &str, method: DriftMethod) -> Result<()> {
        if !self.feature_names.contains(&feature_name.to_string()) {
            return Err(TransformError::InvalidInput(
                format!("Unknown feature name: {}", feature_name),
            ));
        }

        self.drift_methods.insert(feature_name.to_string(), method);
        Ok(())
    }

    /// Set alert configuration
    pub fn set_alert_config(&mut self, config: AlertConfig) {
        self.alert_config = config;
    }

    /// Set baseline performance metrics
    pub fn set_baseline_metrics(&mut self, metrics: PerformanceMetrics) {
        self.baseline_metrics = Some(metrics);
    }

    /// Detect data drift in new data
    pub fn detect_drift(&mut self, new_data: &ArrayView2<f64>) -> Result<Vec<DriftDetectionResult>> {
        let reference_data = self.reference_data.as_ref()
            .ok_or_else(|| TransformError::InvalidInput("Reference data not set".to_string()))?;

        if new_data.ncols() != reference_data.ncols() {
            return Err(TransformError::InvalidInput(
                "New data must have same number of features as reference data".to_string(),
            ));
        }

        let mut results = Vec::new();
        let timestamp = current_timestamp();

        for (i, feature_name) in self.feature_names.iter().enumerate() {
            let method = self.drift_methods.get(feature_name)
                .unwrap_or(&DriftMethod::KolmogorovSmirnov);

            let reference_feature = reference_data.column(i);
            let new_feature = new_data.column(i);

            let result = self.detect_feature_drift(
                &reference_feature,
                &new_feature,
                feature_name,
                method,
                timestamp,
            )?;

            results.push(result.clone());
            self.drift_history.push_back(result);

            // Keep only recent history
            if self.drift_history.len() > 1000 {
                self.drift_history.pop_front();
            }
        }

        // Update Prometheus metrics
        #[cfg(feature = "monitoring")]
        {
            let drift_count = results.iter().filter(|r| r.is_drift_detected).count();
            self.prometheus_metrics.drift_detections.inc_by(drift_count as u64);
        }

        Ok(results)
    }

    /// Record performance metrics
    pub fn record_metrics(&mut self, metrics: PerformanceMetrics) -> Result<Vec<AlertType>> {
        self.performance_history.push_back(metrics.clone());

        // Keep only recent history
        if self.performance_history.len() > 1000 {
            self.performance_history.pop_front();
        }

        // Update Prometheus metrics
        #[cfg(feature = "monitoring")]
        {
            self.prometheus_metrics.processing_time
                .observe(metrics.processing_time_ms / 1000.0);
            self.prometheus_metrics.memory_usage.set(metrics.memory_usage_mb);
            self.prometheus_metrics.error_rate.set(metrics.error_rate);
            self.prometheus_metrics.throughput.set(metrics.throughput);
            self.prometheus_metrics.data_quality.set(metrics.data_quality_score);
        }

        // Check for alerts
        self.check_performance_alerts(&metrics)
    }

    /// Get drift detection summary
    pub fn get_drift_summary(&self, lookback_hours: u64) -> Result<HashMap<String, f64>> {
        let cutoff_time = current_timestamp() - (lookback_hours * 3600);
        let mut summary = HashMap::new();

        for feature_name in &self.feature_names {
            let recent_detections: Vec<_> = self.drift_history.iter()
                .filter(|r| r.timestamp >= cutoff_time && r.feature_name == *feature_name)
                .collect();

            let drift_rate = if recent_detections.is_empty() {
                0.0
            } else {
                recent_detections.iter()
                    .filter(|r| r.is_drift_detected)
                    .count() as f64 / recent_detections.len() as f64
            };

            summary.insert(feature_name.clone(), drift_rate);
        }

        Ok(summary)
    }

    /// Get performance trends
    pub fn get_performance_trends(&self, lookback_hours: u64) -> Result<HashMap<String, f64>> {
        let cutoff_time = current_timestamp() - (lookback_hours * 3600);
        let recent_metrics: Vec<_> = self.performance_history.iter()
            .filter(|m| m.timestamp >= cutoff_time)
            .collect();

        if recent_metrics.is_empty() {
            return Ok(HashMap::new());
        }

        let mut trends = HashMap::new();

        // Calculate trends (change from first to last measurement)
        if recent_metrics.len() >= 2 {
            let first = recent_metrics.first().unwrap();
            let last = recent_metrics.last().unwrap();

            trends.insert(
                "processing_time_trend".to_string(),
                (last.processing_time_ms - first.processing_time_ms) / first.processing_time_ms,
            );
            trends.insert(
                "memory_usage_trend".to_string(),
                (last.memory_usage_mb - first.memory_usage_mb) / first.memory_usage_mb,
            );
            trends.insert(
                "error_rate_trend".to_string(),
                last.error_rate - first.error_rate,
            );
            trends.insert(
                "throughput_trend".to_string(),
                (last.throughput - first.throughput) / first.throughput,
            );
        }

        Ok(trends)
    }

    fn detect_feature_drift(
        &self,
        reference: &ArrayView1<f64>,
        new_data: &ArrayView1<f64>,
        feature_name: &str,
        method: &DriftMethod,
        timestamp: u64,
    ) -> Result<DriftDetectionResult> {
        let (statistic, p_value, is_drift) = match method {
            DriftMethod::KolmogorovSmirnov => {
                let (stat, p_val) = self.kolmogorov_smirnov_test(reference, new_data)?;
                (stat, Some(p_val), p_val < self.alert_config.drift_threshold)
            },
            DriftMethod::PopulationStabilityIndex => {
                let psi = self.population_stability_index(reference, new_data)?;
                (psi, None, psi > 0.1) // PSI > 0.1 indicates drift
            },
            DriftMethod::WassersteinDistance => {
                let distance = self.wasserstein_distance(reference, new_data)?;
                (distance, None, distance > self.alert_config.drift_threshold)
            },
            _ => {
                // Fallback to KS test for other methods
                let (stat, p_val) = self.kolmogorov_smirnov_test(reference, new_data)?;
                (stat, Some(p_val), p_val < self.alert_config.drift_threshold)
            },
        };

        let severity = if let Some(p_val) = p_value {
            1.0 - p_val // Lower p-value = higher severity
        } else {
            statistic.min(1.0) // Cap at 1.0
        };

        Ok(DriftDetectionResult {
            feature_name: feature_name.to_string(),
            method: method.clone(),
            statistic,
            p_value,
            is_drift_detected: is_drift,
            severity,
            timestamp,
        })
    }

    fn kolmogorov_smirnov_test(&self, x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> Result<(f64, f64)> {
        // Simplified KS test implementation
        let mut x_sorted = x.to_vec();
        let mut y_sorted = y.to_vec();
        x_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        y_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n1 = x_sorted.len() as f64;
        let n2 = y_sorted.len() as f64;

        let mut max_diff = 0.0;
        let mut i = 0;
        let mut j = 0;

        while i < x_sorted.len() && j < y_sorted.len() {
            let cdf1 = (i + 1) as f64 / n1;
            let cdf2 = (j + 1) as f64 / n2;
            max_diff = max_diff.max((cdf1 - cdf2).abs());

            if x_sorted[i] <= y_sorted[j] {
                i += 1;
            } else {
                j += 1;
            }
        }

        let statistic = max_diff;
        let sqrt_term = ((n1 + n2) / (n1 * n2)).sqrt();
        let critical_value = 1.36 * sqrt_term; // For α = 0.05
        let p_value = 2.0 * (-2.0 * statistic.powi(2) * n1 * n2 / (n1 + n2)).exp();

        Ok((statistic, p_value.max(0.0).min(1.0)))
    }

    fn population_stability_index(&self, reference: &ArrayView1<f64>, new_data: &ArrayView1<f64>) -> Result<f64> {
        // Create bins based on reference data
        let mut ref_sorted = reference.to_vec();
        ref_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let n_bins = 10;
        let mut bins = Vec::new();
        for i in 0..=n_bins {
            let percentile = (i as f64) / (n_bins as f64);
            let index = ((ref_sorted.len() - 1) as f64 * percentile) as usize;
            bins.push(ref_sorted[index]);
        }

        // Calculate frequencies
        let ref_freq = self.calculate_bin_frequencies(reference, &bins);
        let new_freq = self.calculate_bin_frequencies(new_data, &bins);

        // Calculate PSI
        let mut psi = 0.0;
        for i in 0..n_bins {
            let ref_pct = ref_freq[i];
            let new_pct = new_freq[i];
            
            if ref_pct > 0.0 && new_pct > 0.0 {
                psi += (new_pct - ref_pct) * (new_pct / ref_pct).ln();
            }
        }

        Ok(psi)
    }

    fn calculate_bin_frequencies(&self, data: &ArrayView1<f64>, bins: &[f64]) -> Vec<f64> {
        let mut frequencies = vec![0; bins.len() - 1];
        
        for &value in data.iter() {
            for i in 0..bins.len() - 1 {
                if value >= bins[i] && value < bins[i + 1] {
                    frequencies[i] += 1;
                    break;
                }
            }
        }

        let total = data.len() as f64;
        frequencies.iter().map(|&f| f as f64 / total).collect()
    }

    fn wasserstein_distance(&self, x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> Result<f64> {
        // Simplified 1D Wasserstein distance (Earth Mover's Distance)
        let mut x_sorted = x.to_vec();
        let mut y_sorted = y.to_vec();
        x_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        y_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n1 = x_sorted.len();
        let n2 = y_sorted.len();
        let max_len = n1.max(n2);

        let mut distance = 0.0;
        for i in 0..max_len {
            let x_val = if i < n1 { x_sorted[i] } else { x_sorted[n1 - 1] };
            let y_val = if i < n2 { y_sorted[i] } else { y_sorted[n2 - 1] };
            distance += (x_val - y_val).abs();
        }

        Ok(distance / max_len as f64)
    }

    fn check_performance_alerts(&mut self, metrics: &PerformanceMetrics) -> Result<Vec<AlertType>> {
        let mut alerts = Vec::new();
        let current_time = current_timestamp();

        // Check if we're in cooldown period
        let cooldown_key = "performance";
        if let Some(&last_alert_time) = self.last_alert_times.get(cooldown_key) {
            if current_time - last_alert_time < self.alert_config.cooldown_seconds {
                return Ok(alerts);
            }
        }

        // Check performance degradation
        if let Some(ref baseline) = self.baseline_metrics {
            let degradation_ratio = metrics.processing_time_ms / baseline.processing_time_ms;
            if degradation_ratio > self.alert_config.performance_threshold {
                alerts.push(AlertType::PerformanceDegradation {
                    metric: "processing_time".to_string(),
                    value: degradation_ratio,
                });
            }
        }

        // Check error rate
        if metrics.error_rate > self.alert_config.error_rate_threshold {
            alerts.push(AlertType::HighErrorRate {
                rate: metrics.error_rate,
            });
        }

        // Check memory usage
        if metrics.memory_usage_mb > self.alert_config.memory_threshold_mb {
            alerts.push(AlertType::MemoryExhaustion {
                usage_mb: metrics.memory_usage_mb,
            });
        }

        // Check data quality
        if metrics.data_quality_score < 0.8 {
            alerts.push(AlertType::DataQualityIssue {
                score: metrics.data_quality_score,
            });
        }

        if !alerts.is_empty() {
            self.last_alert_times.insert(cooldown_key.to_string(), current_time);
        }

        Ok(alerts)
    }

    /// Export metrics in Prometheus format
    #[cfg(feature = "monitoring")]
    pub fn export_prometheus_metrics(&self) -> Result<String> {
        use prometheus::Encoder;
        let encoder = prometheus::TextEncoder::new();
        let metric_families = self.metrics_registry.gather();
        encoder.encode_to_string(&metric_families)
            .map_err(|e| TransformError::ComputationError(format!("Failed to encode metrics: {}", e)))
    }
}

fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}