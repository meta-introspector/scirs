//! Production monitoring with drift detection and model degradation alerts
//!
//! This module provides comprehensive monitoring capabilities for transformation
//! pipelines in production environments, including data drift detection,
//! performance monitoring, and automated alerting.

use crate::error::{Result, TransformError};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::validation::{check_finite, check_not_empty, check_positive};
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
            performance_threshold: 2.0,  // 2x baseline
            error_rate_threshold: 0.05,  // 5%
            memory_threshold_mb: 1000.0, // 1GB
            cooldown_seconds: 300,       // 5 minutes
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
            drift_detections: Counter::new(
                "transform_drift_detections_total",
                "Total number of drift detections",
            )
            .map_err(|e| {
                TransformError::ComputationError(format!("Failed to create counter: {}", e))
            })?,
            processing_time: Histogram::new(
                "transform_processing_time_seconds",
                "Processing time in seconds",
            )
            .map_err(|e| {
                TransformError::ComputationError(format!("Failed to create histogram: {}", e))
            })?,
            memory_usage: Gauge::new("transform_memory_usage_mb", "Memory usage in MB").map_err(
                |e| TransformError::ComputationError(format!("Failed to create gauge: {}", e)),
            )?,
            error_rate: Gauge::new("transform_error_rate", "Error rate").map_err(|e| {
                TransformError::ComputationError(format!("Failed to create gauge: {}", e))
            })?,
            throughput: Gauge::new(
                "transform_throughput_samples_per_second",
                "Throughput in samples per second",
            )
            .map_err(|e| {
                TransformError::ComputationError(format!("Failed to create gauge: {}", e))
            })?,
            data_quality: Gauge::new("transform_data_quality_score", "Data quality score")
                .map_err(|e| {
                    TransformError::ComputationError(format!("Failed to create gauge: {}", e))
                })?,
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
    pub fn set_reference_data(
        &mut self,
        data: Array2<f64>,
        feature_names: Option<Vec<String>>,
    ) -> Result<()> {
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
            self.drift_methods
                .insert(feature_name.clone(), DriftMethod::KolmogorovSmirnov);
        }

        Ok(())
    }

    /// Configure drift detection method for a specific feature
    pub fn set_drift_method(&mut self, feature_name: &str, method: DriftMethod) -> Result<()> {
        if !self.feature_names.contains(&feature_name.to_string()) {
            return Err(TransformError::InvalidInput(format!(
                "Unknown feature name: {}",
                feature_name
            )));
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
    pub fn detect_drift(
        &mut self,
        new_data: &ArrayView2<f64>,
    ) -> Result<Vec<DriftDetectionResult>> {
        let reference_data = self
            .reference_data
            .as_ref()
            .ok_or_else(|| TransformError::InvalidInput("Reference data not set".to_string()))?;

        if new_data.ncols() != reference_data.ncols() {
            return Err(TransformError::InvalidInput(
                "New data must have same number of features as reference data".to_string(),
            ));
        }

        let mut results = Vec::new();
        let timestamp = current_timestamp();

        for (i, feature_name) in self.feature_names.iter().enumerate() {
            let method = self
                .drift_methods
                .get(feature_name)
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
            self.prometheus_metrics
                .drift_detections
                .inc_by(drift_count as u64);
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
            self.prometheus_metrics
                .processing_time
                .observe(metrics.processing_time_ms / 1000.0);
            self.prometheus_metrics
                .memory_usage
                .set(metrics.memory_usage_mb);
            self.prometheus_metrics.error_rate.set(metrics.error_rate);
            self.prometheus_metrics.throughput.set(metrics.throughput);
            self.prometheus_metrics
                .data_quality
                .set(metrics.data_quality_score);
        }

        // Check for alerts
        self.check_performance_alerts(&metrics)
    }

    /// Get drift detection summary
    pub fn get_drift_summary(&self, lookback_hours: u64) -> Result<HashMap<String, f64>> {
        let cutoff_time = current_timestamp() - (lookback_hours * 3600);
        let mut summary = HashMap::new();

        for feature_name in &self.feature_names {
            let recent_detections: Vec<_> = self
                .drift_history
                .iter()
                .filter(|r| r.timestamp >= cutoff_time && r.feature_name == *feature_name)
                .collect();

            let drift_rate = if recent_detections.is_empty() {
                0.0
            } else {
                recent_detections
                    .iter()
                    .filter(|r| r.is_drift_detected)
                    .count() as f64
                    / recent_detections.len() as f64
            };

            summary.insert(feature_name.clone(), drift_rate);
        }

        Ok(summary)
    }

    /// Get performance trends
    pub fn get_performance_trends(&self, lookback_hours: u64) -> Result<HashMap<String, f64>> {
        let cutoff_time = current_timestamp() - (lookback_hours * 3600);
        let recent_metrics: Vec<_> = self
            .performance_history
            .iter()
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
        check_not_empty(reference, "reference")?;
        check_not_empty(new_data, "new_data")?;
        check_finite(reference, "reference")?;
        check_finite(new_data, "new_data")?;

        let (statistic, p_value, is_drift) = match method {
            DriftMethod::KolmogorovSmirnov => {
                let (stat, p_val) = self.kolmogorov_smirnov_test(reference, new_data)?;
                (stat, Some(p_val), p_val < self.alert_config.drift_threshold)
            }
            DriftMethod::ChiSquare => {
                let (stat, p_val) = self.chi_square_test(reference, new_data)?;
                (stat, Some(p_val), p_val < self.alert_config.drift_threshold)
            }
            DriftMethod::PopulationStabilityIndex => {
                let psi = self.population_stability_index(reference, new_data)?;
                (psi, None, psi > 0.1) // PSI > 0.1 indicates drift
            }
            DriftMethod::MaximumMeanDiscrepancy => {
                let mmd = self.maximum_mean_discrepancy(reference, new_data)?;
                (mmd, None, mmd > self.alert_config.drift_threshold)
            }
            DriftMethod::WassersteinDistance => {
                let distance = self.wasserstein_distance(reference, new_data)?;
                (distance, None, distance > self.alert_config.drift_threshold)
            }
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

    fn kolmogorov_smirnov_test(
        &self,
        x: &ArrayView1<f64>,
        y: &ArrayView1<f64>,
    ) -> Result<(f64, f64)> {
        let mut x_sorted = x.to_vec();
        let mut y_sorted = y.to_vec();
        x_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        y_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n1 = x_sorted.len() as f64;
        let n2 = y_sorted.len() as f64;

        // Create combined sorted array for precise CDF calculation
        let mut combined: Vec<(f64, i32)> = Vec::new();
        for val in &x_sorted {
            combined.push((*val, 1)); // Mark as from first sample
        }
        for val in &y_sorted {
            combined.push((*val, 2)); // Mark as from second sample
        }
        combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut cdf1 = 0.0;
        let mut cdf2 = 0.0;
        let mut max_diff = 0.0;

        for (_, sample_id) in combined {
            if sample_id == 1 {
                cdf1 += 1.0 / n1;
            } else {
                cdf2 += 1.0 / n2;
            }
            max_diff = max_diff.max((cdf1 - cdf2).abs());
        }

        let statistic = max_diff;
        
        // More accurate p-value calculation using the asymptotic distribution
        let effective_n = (n1 * n2) / (n1 + n2);
        let lambda = statistic * effective_n.sqrt();
        
        // Kolmogorov distribution approximation for p-value
        let p_value = if lambda < 0.27 {
            1.0
        } else if lambda < 1.0 {
            2.0 * (-2.0 * lambda * lambda).exp()
        } else {
            // Series expansion for large lambda
            let mut sum = 0.0;
            for k in 1..=10 {
                let k_f = k as f64;
                sum += (-1.0_f64).powi(k - 1) * (-2.0 * k_f * k_f * lambda * lambda).exp();
            }
            2.0 * sum
        };

        Ok((statistic, p_value.max(0.0).min(1.0)))
    }

    fn population_stability_index(
        &self,
        reference: &ArrayView1<f64>,
        new_data: &ArrayView1<f64>,
    ) -> Result<f64> {
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
        if bins.len() < 2 {
            return vec![];
        }

        let mut frequencies = vec![0; bins.len() - 1];

        for &value in data.iter() {
            if !value.is_finite() {
                continue;
            }

            // Find appropriate bin for this value
            let mut placed = false;
            for i in 0..bins.len() - 1 {
                if i == bins.len() - 2 {
                    // Last bin includes upper bound
                    if value >= bins[i] && value <= bins[i + 1] {
                        frequencies[i] += 1;
                        placed = true;
                        break;
                    }
                } else if value >= bins[i] && value < bins[i + 1] {
                    frequencies[i] += 1;
                    placed = true;
                    break;
                }
            }

            // Handle values outside the range
            if !placed {
                if value < bins[0] {
                    frequencies[0] += 1;
                } else if value > bins[bins.len() - 1] {
                    frequencies[frequencies.len() - 1] += 1;
                }
            }
        }

        let total = data.iter().filter(|&&v| v.is_finite()).count() as f64;
        if total == 0.0 {
            vec![0.0; frequencies.len()]
        } else {
            frequencies.iter().map(|&f| f as f64 / total).collect()
        }
    }

    fn wasserstein_distance(&self, x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> Result<f64> {
        // Simplified 1D Wasserstein distance (Earth Mover's Distance)
        let mut x_sorted: Vec<f64> = x.iter().filter(|&&v| v.is_finite()).copied().collect();
        let mut y_sorted: Vec<f64> = y.iter().filter(|&&v| v.is_finite()).copied().collect();

        if x_sorted.is_empty() || y_sorted.is_empty() {
            return Ok(0.0);
        }

        x_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        y_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n1 = x_sorted.len();
        let n2 = y_sorted.len();
        let max_len = n1.max(n2);

        let mut distance = 0.0;
        for i in 0..max_len {
            let x_val = if i < n1 {
                x_sorted[i]
            } else {
                x_sorted[n1 - 1]
            };
            let y_val = if i < n2 {
                y_sorted[i]
            } else {
                y_sorted[n2 - 1]
            };
            distance += (x_val - y_val).abs();
        }

        Ok(distance / max_len as f64)
    }

    /// Chi-square test for categorical data drift detection
    fn chi_square_test(
        &self,
        reference: &ArrayView1<f64>,
        new_data: &ArrayView1<f64>,
    ) -> Result<(f64, f64)> {
        // For continuous data, we'll bin it first and then apply chi-square test
        let n_bins = 10;

        // Combine data to determine common bins
        let mut combined_data: Vec<f64> = reference
            .iter()
            .chain(new_data.iter())
            .filter(|&&v| v.is_finite())
            .copied()
            .collect();

        if combined_data.len() < n_bins {
            return Ok((0.0, 1.0)); // Not enough data for meaningful test
        }

        combined_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Create bins based on quantiles
        let mut bins = Vec::new();
        for i in 0..=n_bins {
            let percentile = i as f64 / n_bins as f64;
            let index = ((combined_data.len() - 1) as f64 * percentile) as usize;
            bins.push(combined_data[index]);
        }

        // Remove duplicate bin edges
        bins.dedup_by(|a, b| (a - b).abs() < f64::EPSILON);

        if bins.len() < 2 {
            return Ok((0.0, 1.0));
        }

        let ref_freq = self.calculate_bin_frequencies(reference, &bins);
        let new_freq = self.calculate_bin_frequencies(new_data, &bins);

        let ref_total = reference.iter().filter(|&&v| v.is_finite()).count() as f64;
        let new_total = new_data.iter().filter(|&&v| v.is_finite()).count() as f64;

        if ref_total == 0.0 || new_total == 0.0 {
            return Ok((0.0, 1.0));
        }

        // Calculate chi-square statistic
        let mut chi_square = 0.0;
        let mut degrees_of_freedom = 0;

        for i in 0..ref_freq.len() {
            let observed_ref = ref_freq[i] * ref_total;
            let observed_new = new_freq[i] * new_total;

            // Calculate expected frequencies under null hypothesis
            let total_in_bin = observed_ref + observed_new;
            let expected_ref_null = total_in_bin * ref_total / (ref_total + new_total);
            let expected_new_null = total_in_bin * new_total / (ref_total + new_total);

            if expected_ref_null > 5.0 && expected_new_null > 5.0 {
                chi_square += (observed_ref - expected_ref_null).powi(2) / expected_ref_null;
                chi_square += (observed_new - expected_new_null).powi(2) / expected_new_null;
                degrees_of_freedom += 1;
            }
        }

        // Approximate p-value using chi-square distribution
        let p_value = if degrees_of_freedom > 0 {
            self.chi_square_cdf_complement(chi_square, degrees_of_freedom as f64)
        } else {
            1.0
        };

        Ok((chi_square, p_value))
    }

    /// Maximum Mean Discrepancy (MMD) test for distribution comparison
    fn maximum_mean_discrepancy(&self, x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> Result<f64> {
        let x_clean: Vec<f64> = x.iter().filter(|&&v| v.is_finite()).copied().collect();
        let y_clean: Vec<f64> = y.iter().filter(|&&v| v.is_finite()).copied().collect();

        if x_clean.is_empty() || y_clean.is_empty() {
            return Ok(0.0);
        }

        let n = x_clean.len();
        let m = y_clean.len();

        // Use RBF kernel with adaptive bandwidth
        let all_data: Vec<f64> = x_clean.iter().chain(y_clean.iter()).copied().collect();
        let mut sorted_data = all_data;
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Use median absolute deviation for bandwidth selection
        let median = sorted_data[sorted_data.len() / 2];
        let mad: f64 =
            sorted_data.iter().map(|&x| (x - median).abs()).sum::<f64>() / sorted_data.len() as f64;
        let bandwidth = mad.max(1.0); // Ensure reasonable bandwidth

        // Calculate MMD^2 = E[k(X,X')] + E[k(Y,Y')] - 2*E[k(X,Y)]
        let mut kxx = 0.0;
        let mut kyy = 0.0;
        let mut kxy = 0.0;

        // E[k(X,X')] - sample without replacement
        if n > 1 {
            for i in 0..n {
                for j in (i + 1)..n {
                    kxx += self.rbf_kernel(x_clean[i], x_clean[j], bandwidth);
                }
            }
            kxx = 2.0 * kxx / (n * (n - 1)) as f64;
        }

        // E[k(Y,Y')] - sample without replacement
        if m > 1 {
            for i in 0..m {
                for j in (i + 1)..m {
                    kyy += self.rbf_kernel(y_clean[i], y_clean[j], bandwidth);
                }
            }
            kyy = 2.0 * kyy / (m * (m - 1)) as f64;
        }

        // E[k(X,Y)]
        for i in 0..n {
            for j in 0..m {
                kxy += self.rbf_kernel(x_clean[i], y_clean[j], bandwidth);
            }
        }
        kxy /= (n * m) as f64;

        let mmd_squared = kxx + kyy - 2.0 * kxy;
        Ok(mmd_squared.max(0.0).sqrt()) // Take square root and ensure non-negative
    }

    /// RBF (Gaussian) kernel function
    fn rbf_kernel(&self, x: f64, y: f64, bandwidth: f64) -> f64 {
        let diff = x - y;
        (-diff * diff / (2.0 * bandwidth * bandwidth)).exp()
    }

    /// Complement of chi-square CDF using improved approximations
    fn chi_square_cdf_complement(&self, x: f64, df: f64) -> f64 {
        if x <= 0.0 {
            return 1.0;
        }
        if df <= 0.0 {
            return 0.0;
        }

        // For large df, use Wilson-Hilferty transformation (normal approximation)
        if df >= 30.0 {
            let h = 2.0 / (9.0 * df);
            let z = ((x / df).powf(1.0 / 3.0) - (1.0 - h)) / h.sqrt();
            return 0.5 * (1.0 - self.erf(z / 2.0_f64.sqrt()));
        }

        // For moderate df, use incomplete gamma function approximation
        // P(X > x) = 1 - P(X <= x) = 1 - gamma_cdf(x/2, df/2)
        let alpha = df / 2.0;
        let x_half = x / 2.0;
        
        // Use series expansion for gamma CDF
        if x_half < alpha + 1.0 {
            // Use series when x is relatively small compared to alpha
            let mut term = x_half.powf(alpha) * (-x_half).exp();
            let mut sum = term;
            
            for k in 1..=50 {
                term *= x_half / (alpha + k as f64);
                sum += term;
                if term / sum < 1e-10 {
                    break;
                }
            }
            
            let gamma_cdf = sum / self.gamma(alpha);
            1.0 - gamma_cdf.min(1.0)
        } else {
            // Use continued fraction when x is large
            let a = alpha;
            let b = x_half + 1.0 - a;
            let c = 1e30;
            let mut d = 1.0 / b;
            let mut h = d;
            
            for i in 1..=100 {
                let an = -i as f64 * (i as f64 - a);
                let b = b + 2.0;
                d = an * d + b;
                if d.abs() < 1e-30 { d = 1e-30; }
                let c = b + an / c;
                if c.abs() < 1e-30 { c = 1e-30; }
                d = 1.0 / d;
                let del = d * c;
                h *= del;
                if (del - 1.0).abs() < 1e-10 {
                    break;
                }
            }
            
            let gamma_cf = (-x_half).exp() * x_half.powf(a) * h / self.gamma(a);
            gamma_cf.max(0.0).min(1.0)
        }
    }

    /// Error function approximation
    fn erf(&self, x: f64) -> f64 {
        // Abramowitz and Stegun approximation
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if x >= 0.0 { 1.0 } else { -1.0 };
        let x = x.abs();

        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        sign * y
    }

    /// Gamma function using Lanczos approximation
    fn gamma(&self, z: f64) -> f64 {
        if z < 0.5 {
            // Use reflection formula: Γ(z)Γ(1-z) = π/sin(πz)
            std::f64::consts::PI / (std::f64::consts::PI * z).sin() / self.gamma(1.0 - z)
        } else {
            // Lanczos approximation coefficients
            let g = 7.0;
            let c = [
                0.99999999999980993,
                676.5203681218851,
                -1259.1392167224028,
                771.32342877765313,
                -176.61502916214059,
                12.507343278686905,
                -0.13857109526572012,
                9.9843695780195716e-6,
                1.5056327351493116e-7,
            ];

            let z = z - 1.0;
            let mut x = c[0];
            for i in 1..c.len() {
                x += c[i] / (z + i as f64);
            }

            let t = z + g + 0.5;
            (2.0 * std::f64::consts::PI).sqrt() * t.powf(z + 0.5) * (-t).exp() * x
        }
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
            self.last_alert_times
                .insert(cooldown_key.to_string(), current_time);
        }

        Ok(alerts)
    }

    /// Export metrics in Prometheus format
    #[cfg(feature = "monitoring")]
    pub fn export_prometheus_metrics(&self) -> Result<String> {
        use prometheus::Encoder;
        let encoder = prometheus::TextEncoder::new();
        let metric_families = self.metrics_registry.gather();
        encoder.encode_to_string(&metric_families).map_err(|e| {
            TransformError::ComputationError(format!("Failed to encode metrics: {}", e))
        })
    }
}

fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0))
        .as_secs()
}
