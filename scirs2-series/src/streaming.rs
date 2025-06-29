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

/// Advanced streaming time series capabilities
pub mod advanced {
    use super::*;
    use ndarray::Array1;
    use std::collections::VecDeque;

    /// Real-time forecasting with online model updates
    #[derive(Debug)]
    pub struct StreamingForecaster<F: Float + Debug> {
        /// Exponential smoothing parameter
        alpha: F,
        /// Trend parameter
        beta: Option<F>,
        /// Seasonal parameter
        gamma: Option<F>,
        /// Seasonal period
        seasonal_period: Option<usize>,
        /// Current level
        level: Option<F>,
        /// Current trend
        trend: Option<F>,
        /// Seasonal components
        seasonal: VecDeque<F>,
        /// Recent observations buffer
        buffer: VecDeque<F>,
        /// Maximum buffer size
        max_buffer_size: usize,
        /// Number of observations processed
        observation_count: usize,
    }

    impl<F: Float + Debug + Clone> StreamingForecaster<F> {
        /// Create new streaming forecaster
        pub fn new(
            alpha: F,
            beta: Option<F>,
            gamma: Option<F>,
            seasonal_period: Option<usize>,
            max_buffer_size: usize,
        ) -> Result<Self> {
            if alpha <= F::zero() || alpha > F::one() {
                return Err(TimeSeriesError::InvalidParameter {
                    name: "alpha".to_string(),
                    message: "Alpha must be between 0 and 1".to_string(),
                });
            }

            let seasonal = if let Some(period) = seasonal_period {
                VecDeque::with_capacity(period)
            } else {
                VecDeque::new()
            };

            Ok(Self {
                alpha,
                beta,
                gamma,
                seasonal_period,
                level: None,
                trend: None,
                seasonal,
                buffer: VecDeque::with_capacity(max_buffer_size),
                max_buffer_size,
                observation_count: 0,
            })
        }

        /// Add new observation and update model
        pub fn update(&mut self, value: F) -> Result<()> {
            self.observation_count += 1;

            // Add to buffer
            if self.buffer.len() >= self.max_buffer_size {
                self.buffer.pop_front();
            }
            self.buffer.push_back(value);

            // Initialize components
            if self.level.is_none() {
                self.level = Some(value);
                if self.beta.is_some() {
                    self.trend = Some(F::zero());
                }
                if let Some(period) = self.seasonal_period {
                    for _ in 0..period {
                        self.seasonal.push_back(F::zero());
                    }
                }
                return Ok(());
            }

            let current_level = self.level.unwrap();
            let mut new_level = value;

            // Handle seasonality
            let seasonal_component = if let Some(period) = self.seasonal_period {
                if self.seasonal.len() >= period {
                    let seasonal_idx = (self.observation_count - 1) % period;
                    let seasonal_val = self.seasonal[seasonal_idx];
                    new_level = new_level - seasonal_val;
                    seasonal_val
                } else {
                    F::zero()
                }
            } else {
                F::zero()
            };

            // Update level
            self.level = Some(self.alpha * new_level + (F::one() - self.alpha) * current_level);

            // Update trend if enabled
            if let Some(beta) = self.beta {
                if let Some(current_trend) = self.trend {
                    let new_trend = beta * (self.level.unwrap() - current_level) 
                                  + (F::one() - beta) * current_trend;
                    self.trend = Some(new_trend);
                }
            }

            // Update seasonal component if enabled
            if let (Some(gamma), Some(period)) = (self.gamma, self.seasonal_period) {
                if self.seasonal.len() >= period {
                    let seasonal_idx = (self.observation_count - 1) % period;
                    let current_seasonal = self.seasonal[seasonal_idx];
                    let new_seasonal = gamma * (value - self.level.unwrap()) 
                                     + (F::one() - gamma) * current_seasonal;
                    self.seasonal[seasonal_idx] = new_seasonal;
                }
            }

            Ok(())
        }

        /// Generate forecast for next h steps
        pub fn forecast(&self, steps: usize) -> Result<Array1<F>> {
            if self.level.is_none() {
                return Err(TimeSeriesError::InvalidModel(
                    "Model not initialized with any data".to_string(),
                ));
            }

            let mut forecasts = Array1::zeros(steps);
            let level = self.level.unwrap();
            let trend = self.trend.unwrap_or(F::zero());

            for h in 0..steps {
                let h_f = F::from(h + 1).unwrap();
                let mut forecast = level + trend * h_f;

                // Add seasonal component if available
                if let Some(period) = self.seasonal_period {
                    if !self.seasonal.is_empty() {
                        let seasonal_idx = (self.observation_count + h) % period;
                        if seasonal_idx < self.seasonal.len() {
                            forecast = forecast + self.seasonal[seasonal_idx];
                        }
                    }
                }

                forecasts[h] = forecast;
            }

            Ok(forecasts)
        }

        /// Get current model state summary
        pub fn get_state(&self) -> ModelState<F> {
            ModelState {
                level: self.level,
                trend: self.trend,
                seasonal_components: self.seasonal.iter().cloned().collect(),
                observation_count: self.observation_count,
                buffer_size: self.buffer.len(),
            }
        }
    }

    /// Model state summary
    #[derive(Debug, Clone)]
    pub struct ModelState<F: Float> {
        pub level: Option<F>,
        pub trend: Option<F>,
        pub seasonal_components: Vec<F>,
        pub observation_count: usize,
        pub buffer_size: usize,
    }

    /// Online anomaly detection using Isolation Forest-like approach
    #[derive(Debug)]
    pub struct StreamingAnomalyDetector<F: Float + Debug> {
        /// Recent feature vectors for comparison
        feature_buffer: VecDeque<Vec<F>>,
        /// Maximum buffer size
        max_buffer_size: usize,
        /// Anomaly threshold
        threshold: F,
        /// Feature extractors
        window_size: usize,
        /// Number of features to extract
        num_features: usize,
    }

    impl<F: Float + Debug + Clone> StreamingAnomalyDetector<F> {
        /// Create new anomaly detector
        pub fn new(
            max_buffer_size: usize,
            threshold: F,
            window_size: usize,
            num_features: usize,
        ) -> Self {
            Self {
                feature_buffer: VecDeque::with_capacity(max_buffer_size),
                max_buffer_size,
                threshold,
                window_size,
                num_features,
            }
        }

        /// Extract features from a time series window
        fn extract_features(&self, window: &[F]) -> Vec<F> {
            if window.is_empty() {
                return vec![F::zero(); self.num_features];
            }

            let mut features = Vec::with_capacity(self.num_features);
            let n = F::from(window.len()).unwrap();

            // Feature 1: Mean
            let mean = window.iter().fold(F::zero(), |acc, &x| acc + x) / n;
            features.push(mean);

            // Feature 2: Standard deviation
            let variance = window.iter()
                .map(|&x| (x - mean) * (x - mean))
                .fold(F::zero(), |acc, x| acc + x) / n;
            features.push(variance.sqrt());

            // Feature 3: Skewness (simplified)
            let skewness = window.iter()
                .map(|&x| {
                    let normalized = (x - mean) / variance.sqrt();
                    normalized * normalized * normalized
                })
                .fold(F::zero(), |acc, x| acc + x) / n;
            features.push(skewness);

            // Feature 4: Range
            let min_val = window.iter().fold(F::infinity(), |acc, &x| acc.min(x));
            let max_val = window.iter().fold(F::neg_infinity(), |acc, &x| acc.max(x));
            features.push(max_val - min_val);

            // Feature 5: Trend (slope of linear regression)
            if window.len() > 1 {
                let x_mean = F::from(window.len() - 1).unwrap() / F::from(2).unwrap();
                let mut num = F::zero();
                let mut den = F::zero();
                
                for (i, &y) in window.iter().enumerate() {
                    let x = F::from(i).unwrap();
                    num = num + (x - x_mean) * (y - mean);
                    den = den + (x - x_mean) * (x - x_mean);
                }
                
                let slope = if den > F::zero() { num / den } else { F::zero() };
                features.push(slope);
            } else {
                features.push(F::zero());
            }

            features
        }

        /// Update detector with new window and check for anomalies
        pub fn update(&mut self, window: &[F]) -> Result<bool> {
            if window.len() < self.window_size {
                return Ok(false); // Not enough data
            }

            let features = self.extract_features(&window[window.len() - self.window_size..]);

            if self.feature_buffer.is_empty() {
                // First observation - just store
                if self.feature_buffer.len() >= self.max_buffer_size {
                    self.feature_buffer.pop_front();
                }
                self.feature_buffer.push_back(features);
                return Ok(false);
            }

            // Calculate isolation score (simplified)
            let mut min_distance = F::infinity();
            for stored_features in &self.feature_buffer {
                let distance = features.iter()
                    .zip(stored_features.iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .fold(F::zero(), |acc, x| acc + x)
                    .sqrt();
                min_distance = min_distance.min(distance);
            }

            // Add current features to buffer
            if self.feature_buffer.len() >= self.max_buffer_size {
                self.feature_buffer.pop_front();
            }
            self.feature_buffer.push_back(features);

            // Check if anomaly (isolated point)
            Ok(min_distance > self.threshold)
        }

        /// Update threshold based on recent observations
        pub fn adapt_threshold(&mut self, factor: F) {
            if self.feature_buffer.len() > 2 {
                // Calculate average distance between recent features
                let mut total_distance = F::zero();
                let mut count = 0;

                for i in 0..self.feature_buffer.len() {
                    for j in i + 1..self.feature_buffer.len() {
                        let distance = self.feature_buffer[i].iter()
                            .zip(self.feature_buffer[j].iter())
                            .map(|(&a, &b)| (a - b) * (a - b))
                            .fold(F::zero(), |acc, x| acc + x)
                            .sqrt();
                        total_distance = total_distance + distance;
                        count += 1;
                    }
                }

                if count > 0 {
                    let avg_distance = total_distance / F::from(count).unwrap();
                    self.threshold = avg_distance * factor;
                }
            }
        }
    }

    /// Online pattern matching for streaming time series
    #[derive(Debug)]
    pub struct StreamingPatternMatcher<F: Float + Debug> {
        /// Template patterns to match against
        patterns: Vec<Vec<F>>,
        /// Pattern names
        pattern_names: Vec<String>,
        /// Recent data buffer for pattern matching
        buffer: VecDeque<F>,
        /// Maximum buffer size
        max_buffer_size: usize,
        /// Matching threshold (normalized correlation)
        threshold: F,
    }

    impl<F: Float + Debug + Clone> StreamingPatternMatcher<F> {
        /// Create new pattern matcher
        pub fn new(max_buffer_size: usize, threshold: F) -> Self {
            Self {
                patterns: Vec::new(),
                pattern_names: Vec::new(),
                buffer: VecDeque::with_capacity(max_buffer_size),
                max_buffer_size,
                threshold,
            }
        }

        /// Add a pattern to match against
        pub fn add_pattern(&mut self, pattern: Vec<F>, name: String) -> Result<()> {
            if pattern.is_empty() {
                return Err(TimeSeriesError::InvalidInput(
                    "Pattern cannot be empty".to_string(),
                ));
            }
            self.patterns.push(pattern);
            self.pattern_names.push(name);
            Ok(())
        }

        /// Update buffer and check for pattern matches
        pub fn update(&mut self, value: F) -> Vec<PatternMatch> {
            // Add to buffer
            if self.buffer.len() >= self.max_buffer_size {
                self.buffer.pop_front();
            }
            self.buffer.push_back(value);

            let mut matches = Vec::new();

            // Check each pattern
            for (i, pattern) in self.patterns.iter().enumerate() {
                if self.buffer.len() >= pattern.len() {
                    let recent_data: Vec<F> = self.buffer.iter()
                        .rev()
                        .take(pattern.len())
                        .rev()
                        .cloned()
                        .collect();

                    if let Ok(correlation) = self.normalized_correlation(&recent_data, pattern) {
                        if correlation >= self.threshold {
                            matches.push(PatternMatch {
                                pattern_name: self.pattern_names[i].clone(),
                                correlation,
                                start_index: self.buffer.len() - pattern.len(),
                                pattern_length: pattern.len(),
                            });
                        }
                    }
                }
            }

            matches
        }

        /// Calculate normalized correlation between two sequences
        fn normalized_correlation(&self, a: &[F], b: &[F]) -> Result<F> {
            if a.len() != b.len() || a.is_empty() {
                return Err(TimeSeriesError::InvalidInput(
                    "Sequences must have the same non-zero length".to_string(),
                ));
            }

            let n = F::from(a.len()).unwrap();

            // Calculate means
            let mean_a = a.iter().fold(F::zero(), |acc, &x| acc + x) / n;
            let mean_b = b.iter().fold(F::zero(), |acc, &x| acc + x) / n;

            // Calculate correlation components
            let mut num = F::zero();
            let mut den_a = F::zero();
            let mut den_b = F::zero();

            for (i, (&val_a, &val_b)) in a.iter().zip(b.iter()).enumerate() {
                let diff_a = val_a - mean_a;
                let diff_b = val_b - mean_b;

                num = num + diff_a * diff_b;
                den_a = den_a + diff_a * diff_a;
                den_b = den_b + diff_b * diff_b;
            }

            let denominator = (den_a * den_b).sqrt();
            if denominator > F::zero() {
                Ok(num / denominator)
            } else {
                Ok(F::zero())
            }
        }
    }

    /// Pattern match result
    #[derive(Debug, Clone)]
    pub struct PatternMatch {
        pub pattern_name: String,
        pub correlation: f64,
        pub start_index: usize,
        pub pattern_length: usize,
    }

    /// Memory-efficient circular buffer for streaming data
    #[derive(Debug)]
    pub struct CircularBuffer<F: Float> {
        /// Internal buffer
        buffer: Vec<F>,
        /// Current write position
        position: usize,
        /// Maximum capacity
        capacity: usize,
        /// Whether buffer is full
        is_full: bool,
    }

    impl<F: Float + Debug + Clone + Default> CircularBuffer<F> {
        /// Create new circular buffer
        pub fn new(capacity: usize) -> Self {
            Self {
                buffer: vec![F::default(); capacity],
                position: 0,
                capacity,
                is_full: false,
            }
        }

        /// Add new value to buffer
        pub fn push(&mut self, value: F) {
            self.buffer[self.position] = value;
            self.position = (self.position + 1) % self.capacity;
            
            if self.position == 0 {
                self.is_full = true;
            }
        }

        /// Get current size of buffer
        pub fn len(&self) -> usize {
            if self.is_full {
                self.capacity
            } else {
                self.position
            }
        }

        /// Check if buffer is empty
        pub fn is_empty(&self) -> bool {
            !self.is_full && self.position == 0
        }

        /// Get slice of recent n values
        pub fn recent(&self, n: usize) -> Vec<F> {
            let available = self.len();
            let take = n.min(available);
            let mut result = Vec::with_capacity(take);

            if self.is_full {
                // Buffer is full, need to handle wrap-around
                let start_pos = (self.position + self.capacity - take) % self.capacity;
                
                if start_pos + take <= self.capacity {
                    // No wrap-around needed
                    result.extend_from_slice(&self.buffer[start_pos..start_pos + take]);
                } else {
                    // Need to handle wrap-around
                    let first_part = self.capacity - start_pos;
                    result.extend_from_slice(&self.buffer[start_pos..]);
                    result.extend_from_slice(&self.buffer[..take - first_part]);
                }
            } else {
                // Buffer not full, simple case
                let start = self.position.saturating_sub(take);
                result.extend_from_slice(&self.buffer[start..self.position]);
            }

            result
        }

        /// Get all values in chronological order
        pub fn to_vec(&self) -> Vec<F> {
            self.recent(self.len())
        }

        /// Calculate statistics over recent window
        pub fn window_stats(&self, window_size: usize) -> OnlineStats<F> {
            let recent_data = self.recent(window_size);
            let mut stats = OnlineStats::new();
            
            for value in recent_data {
                stats.update(value);
            }
            
            stats
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_streaming_forecaster() {
            let mut forecaster = StreamingForecaster::new(
                0.3, Some(0.1), None, None, 100
            ).unwrap();

            // Add trend data
            for i in 1..=20 {
                forecaster.update(i as f64).unwrap();
            }

            let forecast = forecaster.forecast(5).unwrap();
            assert_eq!(forecast.len(), 5);
            
            // Should forecast increasing trend
            assert!(forecast[1] > forecast[0]);
            assert!(forecast[2] > forecast[1]);
        }

        #[test]
        fn test_anomaly_detector() {
            let mut detector = StreamingAnomalyDetector::new(100, 2.0, 10, 5);
            
            // Add normal data
            let normal_data: Vec<f64> = (0..20).map(|x| x as f64).collect();
            
            for window in normal_data.windows(10) {
                let is_anomaly = detector.update(window).unwrap();
                assert!(!is_anomaly, "Normal data should not be anomalous");
            }

            // Add anomalous data
            let mut anomalous_data = normal_data.clone();
            anomalous_data.extend(vec![1000.0; 10]); // Clear anomaly
            
            let result = detector.update(&anomalous_data[anomalous_data.len()-10..]).unwrap();
            assert!(result, "Clear anomaly should be detected");
        }

        #[test]
        fn test_pattern_matcher() {
            let mut matcher = StreamingPatternMatcher::new(100, 0.8);
            
            // Add a simple pattern
            let pattern = vec![1.0, 2.0, 3.0, 2.0, 1.0];
            matcher.add_pattern(pattern.clone(), "triangle".to_string()).unwrap();

            // Add matching data
            for &value in &pattern {
                let matches = matcher.update(value);
                if !matches.is_empty() {
                    assert_eq!(matches[0].pattern_name, "triangle");
                    assert!(matches[0].correlation >= 0.8);
                }
            }
        }

        #[test]
        fn test_circular_buffer() {
            let mut buffer = CircularBuffer::new(5);
            
            // Add data
            for i in 1..=3 {
                buffer.push(i as f64);
            }
            
            assert_eq!(buffer.len(), 3);
            assert_eq!(buffer.recent(2), vec![2.0, 3.0]);
            
            // Fill buffer completely
            for i in 4..=7 {
                buffer.push(i as f64);
            }
            
            assert_eq!(buffer.len(), 5);
            assert_eq!(buffer.to_vec(), vec![3.0, 4.0, 5.0, 6.0, 7.0]);
        }
    }
}