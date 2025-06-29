//! Real-time streaming time series analysis
//!
//! This module provides capabilities for analyzing time series data in real-time,
//! including online learning algorithms, streaming forecasting, and incremental statistics.

use ndarray::{Array1, ArrayView1};
use num_traits::Float;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::error::{Result, TimeSeriesError};

/// Configuration for streaming analysis
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Maximum window size for online calculations
    pub window_size: usize,
    /// Minimum number of observations before starting analysis
    pub min_observations: usize,
    /// Update frequency for model parameters
    pub update_frequency: usize,
    /// Memory threshold for automatic cleanup
    pub memory_threshold: usize,
    /// Enable adaptive windowing
    pub adaptive_windowing: bool,
    /// Detection threshold for change points
    pub change_detection_threshold: f64,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            window_size: 1000,
            min_observations: 10,
            update_frequency: 10,
            memory_threshold: 10000,
            adaptive_windowing: false,
            change_detection_threshold: 3.0,
        }
    }
}

/// Real-time change point detection result
#[derive(Debug, Clone)]
pub struct ChangePoint {
    /// Index of the change point
    pub index: usize,
    /// Timestamp of the change point
    pub timestamp: Option<Instant>,
    /// Confidence score (higher = more confident)
    pub confidence: f64,
    /// Type of change detected
    pub change_type: ChangeType,
}

/// Types of changes that can be detected
#[derive(Debug, Clone)]
pub enum ChangeType {
    /// Change in mean
    MeanShift,
    /// Change in variance
    VarianceShift,
    /// Change in trend
    TrendChange,
    /// Change in seasonality
    SeasonalityChange,
    /// General structural break
    StructuralBreak,
}

/// Online statistics tracker
#[derive(Debug, Clone)]
pub struct OnlineStats<F: Float> {
    count: usize,
    mean: F,
    m2: F, // For variance calculation
    min_val: F,
    max_val: F,
    sum: F,
    sum_squares: F,
}

impl<F: Float + Debug> OnlineStats<F> {
    /// Create new online statistics tracker
    pub fn new() -> Self {
        Self {
            count: 0,
            mean: F::zero(),
            m2: F::zero(),
            min_val: F::infinity(),
            max_val: F::neg_infinity(),
            sum: F::zero(),
            sum_squares: F::zero(),
        }
    }

    /// Update statistics with new observation
    pub fn update(&mut self, value: F) {
        self.count += 1;
        self.sum = self.sum + value;
        self.sum_squares = self.sum_squares + value * value;
        
        if value < self.min_val {
            self.min_val = value;
        }
        if value > self.max_val {
            self.max_val = value;
        }

        // Welford's online algorithm for mean and variance
        let delta = value - self.mean;
        self.mean = self.mean + delta / F::from(self.count).unwrap();
        let delta2 = value - self.mean;
        self.m2 = self.m2 + delta * delta2;
    }

    /// Get current mean
    pub fn mean(&self) -> F {
        self.mean
    }

    /// Get current variance
    pub fn variance(&self) -> F {
        if self.count < 2 {
            F::zero()
        } else {
            self.m2 / F::from(self.count - 1).unwrap()
        }
    }

    /// Get current standard deviation
    pub fn std_dev(&self) -> F {
        self.variance().sqrt()
    }

    /// Get current minimum
    pub fn min(&self) -> F {
        self.min_val
    }

    /// Get current maximum
    pub fn max(&self) -> F {
        self.max_val
    }

    /// Get current count
    pub fn count(&self) -> usize {
        self.count
    }

    /// Get current sum
    pub fn sum(&self) -> F {
        self.sum
    }
}

impl<F: Float + Debug> Default for OnlineStats<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Exponentially Weighted Moving Average (EWMA) tracker
#[derive(Debug, Clone)]
pub struct EWMA<F: Float> {
    alpha: F,
    current_value: Option<F>,
    variance: Option<F>,
}

impl<F: Float + Debug> EWMA<F> {
    /// Create new EWMA tracker
    pub fn new(alpha: F) -> Result<Self> {
        if alpha <= F::zero() || alpha > F::one() {
            return Err(TimeSeriesError::InvalidParameter {
                name: "alpha".to_string(),
                message: "Alpha must be between 0 and 1".to_string(),
            });
        }

        Ok(Self {
            alpha,
            current_value: None,
            variance: None,
        })
    }

    /// Update EWMA with new observation
    pub fn update(&mut self, value: F) {
        match self.current_value {
            None => {
                self.current_value = Some(value);
                self.variance = Some(F::zero());
            }
            Some(prev) => {
                let new_value = self.alpha * value + (F::one() - self.alpha) * prev;
                self.current_value = Some(new_value);
                
                // Update variance estimate
                let error = value - new_value;
                let new_variance = self.alpha * error * error + 
                    (F::one() - self.alpha) * self.variance.unwrap_or(F::zero());
                self.variance = Some(new_variance);
            }
        }
    }

    /// Get current EWMA value
    pub fn value(&self) -> Option<F> {
        self.current_value
    }

    /// Get current variance estimate
    pub fn variance(&self) -> Option<F> {
        self.variance
    }

    /// Check if value is an outlier based on EWMA
    pub fn is_outlier(&self, value: F, threshold: F) -> bool {
        if let (Some(ewma), Some(var)) = (self.current_value, self.variance) {
            let std_dev = var.sqrt();
            let z_score = (value - ewma).abs() / std_dev;
            z_score > threshold
        } else {
            false
        }
    }
}

/// Cumulative Sum (CUSUM) change point detector
#[derive(Debug, Clone)]
pub struct CusumDetector<F: Float> {
    mean_estimate: F,
    threshold: F,
    cusum_pos: F,
    cusum_neg: F,
    count: usize,
    drift: F,
}

impl<F: Float + Debug> CusumDetector<F> {
    /// Create new CUSUM detector
    pub fn new(threshold: F, drift: F) -> Self {
        Self {
            mean_estimate: F::zero(),
            threshold,
            cusum_pos: F::zero(),
            cusum_neg: F::zero(),
            count: 0,
            drift,
        }
    }

    /// Update CUSUM with new observation
    pub fn update(&mut self, value: F) -> Option<ChangePoint> {
        self.count += 1;
        
        // Update mean estimate
        let delta = value - self.mean_estimate;
        self.mean_estimate = self.mean_estimate + delta / F::from(self.count).unwrap();
        
        // Update CUSUM statistics
        let diff = value - self.mean_estimate;
        self.cusum_pos = F::max(F::zero(), self.cusum_pos + diff - self.drift);
        self.cusum_neg = F::max(F::zero(), self.cusum_neg - diff - self.drift);
        
        // Check for change point
        if self.cusum_pos > self.threshold {
            self.reset();
            Some(ChangePoint {
                index: self.count,
                timestamp: Some(Instant::now()),
                confidence: self.cusum_pos.to_f64().unwrap_or(0.0),
                change_type: ChangeType::MeanShift,
            })
        } else if self.cusum_neg > self.threshold {
            self.reset();
            Some(ChangePoint {
                index: self.count,
                timestamp: Some(Instant::now()),
                confidence: self.cusum_neg.to_f64().unwrap_or(0.0),
                change_type: ChangeType::MeanShift,
            })
        } else {
            None
        }
    }

    fn reset(&mut self) {
        self.cusum_pos = F::zero();
        self.cusum_neg = F::zero();
    }
}

/// Streaming time series analyzer
#[derive(Debug)]
pub struct StreamingAnalyzer<F: Float + Debug> {
    config: StreamConfig,
    buffer: VecDeque<F>,
    timestamps: VecDeque<Instant>,
    stats: OnlineStats<F>,
    ewma: EWMA<F>,
    cusum: CusumDetector<F>,
    change_points: Vec<ChangePoint>,
    forecasts: VecDeque<F>,
    last_update: Instant,
    observation_count: usize,
}

impl<F: Float + Debug> StreamingAnalyzer<F> {
    /// Create new streaming analyzer
    pub fn new(config: StreamConfig) -> Result<Self> {
        let ewma = EWMA::new(F::from(0.1).unwrap())?;
        let cusum = CusumDetector::new(
            F::from(config.change_detection_threshold).unwrap(),
            F::from(0.5).unwrap(),
        );

        Ok(Self {
            config,
            buffer: VecDeque::new(),
            timestamps: VecDeque::new(),
            stats: OnlineStats::new(),
            ewma,
            cusum,
            change_points: Vec::new(),
            forecasts: VecDeque::new(),
            last_update: Instant::now(),
            observation_count: 0,
        })
    }

    /// Add new observation to the stream
    pub fn add_observation(&mut self, value: F) -> Result<()> {
        let now = Instant::now();
        
        // Add to buffer with window management
        self.buffer.push_back(value);
        self.timestamps.push_back(now);
        self.observation_count += 1;

        // Maintain window size
        if self.buffer.len() > self.config.window_size {
            self.buffer.pop_front();
            self.timestamps.pop_front();
        }

        // Update statistics
        self.stats.update(value);
        self.ewma.update(value);

        // Check for change points
        if let Some(change_point) = self.cusum.update(value) {
            self.change_points.push(change_point);
        }

        // Periodic model updates
        if self.observation_count % self.config.update_frequency == 0 {
            self.update_models()?;
        }

        // Memory management
        if self.change_points.len() > self.config.memory_threshold {
            self.cleanup_old_data();
        }

        self.last_update = now;
        Ok(())
    }

    /// Get current statistics
    pub fn get_stats(&self) -> &OnlineStats<F> {
        &self.stats
    }

    /// Get detected change points
    pub fn get_change_points(&self) -> &[ChangePoint] {
        &self.change_points
    }

    /// Get current EWMA value
    pub fn get_ewma(&self) -> Option<F> {
        self.ewma.value()
    }

    /// Check if a value is an outlier
    pub fn is_outlier(&self, value: F) -> bool {
        self.ewma.is_outlier(value, F::from(self.config.change_detection_threshold).unwrap())
    }

    /// Get streaming forecast for next n steps
    pub fn forecast(&self, steps: usize) -> Result<Array1<F>> {
        if self.buffer.len() < self.config.min_observations {
            return Err(TimeSeriesError::InsufficientData {
                message: "Not enough observations for forecasting".to_string(),
                required: self.config.min_observations,
                actual: self.buffer.len(),
            });
        }

        let mut forecasts = Array1::zeros(steps);
        
        // Simple forecasting using EWMA and linear trend
        let current_value = self.ewma.value().unwrap_or(F::zero());
        let trend = self.estimate_trend();
        
        for i in 0..steps {
            let step_f = F::from(i + 1).unwrap();
            forecasts[i] = current_value + trend * step_f;
        }

        Ok(forecasts)
    }

    /// Estimate current trend from recent observations
    fn estimate_trend(&self) -> F {
        if self.buffer.len() < 2 {
            return F::zero();
        }

        let n = std::cmp::min(20, self.buffer.len()); // Use last 20 observations
        let recent: Vec<F> = self.buffer.iter().rev().take(n).cloned().collect();
        
        if recent.len() < 2 {
            return F::zero();
        }

        // Simple linear regression for trend
        let n_f = F::from(recent.len()).unwrap();
        let mut sum_x = F::zero();
        let mut sum_y = F::zero();
        let mut sum_xy = F::zero();
        let mut sum_x2 = F::zero();

        for (i, &y) in recent.iter().enumerate() {
            let x = F::from(i).unwrap();
            sum_x = sum_x + x;
            sum_y = sum_y + y;
            sum_xy = sum_xy + x * y;
            sum_x2 = sum_x2 + x * x;
        }

        let denominator = n_f * sum_x2 - sum_x * sum_x;
        if denominator.abs() < F::epsilon() {
            F::zero()
        } else {
            (n_f * sum_xy - sum_x * sum_y) / denominator
        }
    }

    /// Update internal models
    fn update_models(&mut self) -> Result<()> {
        // Here we could update more sophisticated models
        // For now, just update forecasts buffer
        if self.buffer.len() >= self.config.min_observations {
            let next_forecast = self.forecast(1)?;
            self.forecasts.push_back(next_forecast[0]);
            
            // Keep forecasts buffer reasonable size
            if self.forecasts.len() > 100 {
                self.forecasts.pop_front();
            }
        }
        Ok(())
    }

    /// Clean up old data to manage memory
    fn cleanup_old_data(&mut self) {
        // Keep only recent change points
        let keep_count = self.config.memory_threshold / 2;
        if self.change_points.len() > keep_count {
            self.change_points.drain(0..self.change_points.len() - keep_count);
        }

        // Clean up forecasts buffer
        if self.forecasts.len() > 50 {
            self.forecasts.drain(0..self.forecasts.len() - 50);
        }
    }

    /// Get time since last update
    pub fn time_since_last_update(&self) -> Duration {
        Instant::now().duration_since(self.last_update)
    }

    /// Get total observation count
    pub fn observation_count(&self) -> usize {
        self.observation_count
    }

    /// Get current buffer size
    pub fn buffer_size(&self) -> usize {
        self.buffer.len()
    }

    /// Get current buffer as array view
    pub fn get_buffer(&self) -> Vec<F> {
        self.buffer.iter().cloned().collect()
    }

    /// Reset the analyzer
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.timestamps.clear();
        self.stats = OnlineStats::new();
        self.ewma = EWMA::new(F::from(0.1).unwrap()).unwrap();
        self.cusum = CusumDetector::new(
            F::from(self.config.change_detection_threshold).unwrap(),
            F::from(0.5).unwrap(),
        );
        self.change_points.clear();
        self.forecasts.clear();
        self.observation_count = 0;
        self.last_update = Instant::now();
    }
}

/// Multi-series streaming analyzer for handling multiple time series simultaneously
#[derive(Debug)]
pub struct MultiSeriesAnalyzer<F: Float + Debug> {
    analyzers: HashMap<String, StreamingAnalyzer<F>>,
    config: StreamConfig,
}

impl<F: Float + Debug> MultiSeriesAnalyzer<F> {
    /// Create new multi-series analyzer
    pub fn new(config: StreamConfig) -> Self {
        Self {
            analyzers: HashMap::new(),
            config,
        }
    }

    /// Add new time series to track
    pub fn add_series(&mut self, series_id: String) -> Result<()> {
        let analyzer = StreamingAnalyzer::new(self.config.clone())?;
        self.analyzers.insert(series_id, analyzer);
        Ok(())
    }

    /// Add observation to specific series
    pub fn add_observation(&mut self, series_id: &str, value: F) -> Result<()> {
        if let Some(analyzer) = self.analyzers.get_mut(series_id) {
            analyzer.add_observation(value)
        } else {
            Err(TimeSeriesError::InvalidInput(format!(
                "Series '{}' not found",
                series_id
            )))
        }
    }

    /// Get analyzer for specific series
    pub fn get_analyzer(&self, series_id: &str) -> Option<&StreamingAnalyzer<F>> {
        self.analyzers.get(series_id)
    }

    /// Get mutable analyzer for specific series
    pub fn get_analyzer_mut(&mut self, series_id: &str) -> Option<&mut StreamingAnalyzer<F>> {
        self.analyzers.get_mut(series_id)
    }

    /// Get all series IDs
    pub fn get_series_ids(&self) -> Vec<String> {
        self.analyzers.keys().cloned().collect()
    }

    /// Remove series
    pub fn remove_series(&mut self, series_id: &str) -> bool {
        self.analyzers.remove(series_id).is_some()
    }

    /// Get cross-series correlation (simplified)
    pub fn get_correlation(&self, series1: &str, series2: &str) -> Result<F> {
        let analyzer1 = self.analyzers.get(series1).ok_or_else(|| {
            TimeSeriesError::InvalidInput(format!("Series '{}' not found", series1))
        })?;
        
        let analyzer2 = self.analyzers.get(series2).ok_or_else(|| {
            TimeSeriesError::InvalidInput(format!("Series '{}' not found", series2))
        })?;

        let buffer1 = analyzer1.get_buffer();
        let buffer2 = analyzer2.get_buffer();
        
        let min_len = std::cmp::min(buffer1.len(), buffer2.len());
        if min_len < 2 {
            return Ok(F::zero());
        }

        // Calculate Pearson correlation
        let mean1 = buffer1.iter().take(min_len).cloned().fold(F::zero(), |acc, x| acc + x) / F::from(min_len).unwrap();
        let mean2 = buffer2.iter().take(min_len).cloned().fold(F::zero(), |acc, x| acc + x) / F::from(min_len).unwrap();

        let mut numerator = F::zero();
        let mut sum1_sq = F::zero();
        let mut sum2_sq = F::zero();

        for i in 0..min_len {
            let diff1 = buffer1[i] - mean1;
            let diff2 = buffer2[i] - mean2;
            numerator = numerator + diff1 * diff2;
            sum1_sq = sum1_sq + diff1 * diff1;
            sum2_sq = sum2_sq + diff2 * diff2;
        }

        let denominator = (sum1_sq * sum2_sq).sqrt();
        if denominator > F::epsilon() {
            Ok(numerator / denominator)
        } else {
            Ok(F::zero())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_online_stats() {
        let mut stats = OnlineStats::<f64>::new();
        
        // Add some data points
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        for &val in &data {
            stats.update(val);
        }

        assert_eq!(stats.count(), 5);
        assert_abs_diff_eq!(stats.mean(), 3.0);
        assert_abs_diff_eq!(stats.min(), 1.0);
        assert_abs_diff_eq!(stats.max(), 5.0);
        assert!(stats.variance() > 0.0);
    }

    #[test]
    fn test_ewma() {
        let mut ewma = EWMA::<f64>::new(0.3).unwrap();
        
        ewma.update(10.0);
        assert_abs_diff_eq!(ewma.value().unwrap(), 10.0);
        
        ewma.update(20.0);
        let expected = 0.3 * 20.0 + 0.7 * 10.0;
        assert_abs_diff_eq!(ewma.value().unwrap(), expected);
    }

    #[test]
    fn test_cusum_detector() {
        let mut cusum = CusumDetector::<f64>::new(5.0, 0.5);
        
        // Add normal data
        for i in 0..10 {
            let change = cusum.update(i as f64);
            assert!(change.is_none());
        }
        
        // Add data with mean shift
        for i in 0..10 {
            let change = cusum.update(10.0 + i as f64);
            if change.is_some() {
                assert!(matches!(change.unwrap().change_type, ChangeType::MeanShift));
                break;
            }
        }
    }

    #[test]
    fn test_streaming_analyzer() {
        let config = StreamConfig::default();
        let mut analyzer = StreamingAnalyzer::<f64>::new(config).unwrap();
        
        // Add some observations
        for i in 0..50 {
            let value = (i as f64).sin();
            analyzer.add_observation(value).unwrap();
        }

        assert!(analyzer.observation_count() > 0);
        assert!(analyzer.get_stats().count() > 0);
        
        // Test forecasting
        let forecast = analyzer.forecast(5);
        assert!(forecast.is_ok());
        assert_eq!(forecast.unwrap().len(), 5);
    }

    #[test]
    fn test_multi_series_analyzer() {
        let config = StreamConfig::default();
        let mut multi_analyzer = MultiSeriesAnalyzer::<f64>::new(config);
        
        // Add two series
        multi_analyzer.add_series("series1".to_string()).unwrap();
        multi_analyzer.add_series("series2".to_string()).unwrap();
        
        // Add data to both series
        for i in 0..20 {
            multi_analyzer.add_observation("series1", i as f64).unwrap();
            multi_analyzer.add_observation("series2", (i as f64) * 2.0).unwrap();
        }

        // Check correlation
        let correlation = multi_analyzer.get_correlation("series1", "series2").unwrap();
        assert!(correlation > 0.5); // Should be highly correlated
    }

    #[test]
    fn test_outlier_detection() {
        let config = StreamConfig::default();
        let mut analyzer = StreamingAnalyzer::<f64>::new(config).unwrap();
        
        // Add normal data
        for i in 0..20 {
            analyzer.add_observation(i as f64).unwrap();
        }
        
        // Check if a clear outlier is detected
        assert!(analyzer.is_outlier(1000.0));
        assert!(!analyzer.is_outlier(20.0)); // Should be normal
    }
}