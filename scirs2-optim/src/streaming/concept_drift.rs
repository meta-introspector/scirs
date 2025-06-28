//! Concept drift detection and adaptation for streaming optimization
//!
//! This module provides various algorithms for detecting when the underlying
//! data distribution changes (concept drift) and adapting the optimizer accordingly.

use ndarray::{Array1, Array2};
use num_traits::Float;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

use crate::error::OptimizerError;

/// Types of concept drift detection algorithms
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DriftDetectionMethod {
    /// Page-Hinkley test for change detection
    PageHinkley,
    /// ADWIN (Adaptive Windowing) algorithm
    Adwin,
    /// Drift Detection Method (DDM)
    DriftDetectionMethod,
    /// Early Drift Detection Method (EDDM)
    EarlyDriftDetection,
    /// Statistical test-based detection
    StatisticalTest,
    /// Ensemble-based detection
    Ensemble,
}

/// Concept drift detector configuration
#[derive(Debug, Clone)]
pub struct DriftDetectorConfig {
    /// Detection method to use
    pub method: DriftDetectionMethod,
    /// Minimum samples before detection
    pub min_samples: usize,
    /// Detection threshold
    pub threshold: f64,
    /// Window size for statistical methods
    pub window_size: usize,
    /// Alpha value for statistical tests
    pub alpha: f64,
    /// Warning threshold (before drift)
    pub warning_threshold: f64,
    /// Enable ensemble detection
    pub enable_ensemble: bool,
}

impl Default for DriftDetectorConfig {
    fn default() -> Self {
        Self {
            method: DriftDetectionMethod::PageHinkley,
            min_samples: 30,
            threshold: 3.0,
            window_size: 100,
            alpha: 0.005,
            warning_threshold: 2.0,
            enable_ensemble: false,
        }
    }
}

/// Concept drift detection result
#[derive(Debug, Clone, PartialEq)]
pub enum DriftStatus {
    /// No drift detected
    Stable,
    /// Warning level - potential drift
    Warning,
    /// Drift detected
    Drift,
}

/// Drift detection event
#[derive(Debug, Clone)]
pub struct DriftEvent<A: Float> {
    /// Timestamp of detection
    pub timestamp: Instant,
    /// Detection confidence (0.0 to 1.0)
    pub confidence: A,
    /// Type of drift detected
    pub drift_type: DriftType,
    /// Recommendation for adaptation
    pub adaptation_recommendation: AdaptationRecommendation,
}

/// Types of concept drift
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DriftType {
    /// Sudden/abrupt drift
    Sudden,
    /// Gradual drift
    Gradual,
    /// Incremental drift
    Incremental,
    /// Recurring drift
    Recurring,
    /// Blip (temporary change)
    Blip,
}

/// Recommendations for adapting to drift
#[derive(Debug, Clone)]
pub enum AdaptationRecommendation {
    /// Reset optimizer state
    Reset,
    /// Increase learning rate
    IncreaseLearningRate { factor: f64 },
    /// Decrease learning rate
    DecreaseLearningRate { factor: f64 },
    /// Use different optimizer
    SwitchOptimizer { new_optimizer: String },
    /// Adjust window size
    AdjustWindow { new_size: usize },
    /// No adaptation needed
    NoAction,
}

/// Page-Hinkley drift detector
#[derive(Debug, Clone)]
pub struct PageHinkleyDetector<A: Float> {
    /// Cumulative sum
    sum: A,
    /// Minimum cumulative sum seen
    min_sum: A,
    /// Detection threshold
    threshold: A,
    /// Warning threshold
    warning_threshold: A,
    /// Sample count
    sample_count: usize,
    /// Last drift time
    last_drift: Option<Instant>,
}

impl<A: Float> PageHinkleyDetector<A> {
    /// Create a new Page-Hinkley detector
    pub fn new(threshold: A, warning_threshold: A) -> Self {
        Self {
            sum: A::zero(),
            min_sum: A::zero(),
            threshold,
            warning_threshold,
            sample_count: 0,
            last_drift: None,
        }
    }
    
    /// Update detector with new loss value
    pub fn update(&mut self, loss: A) -> DriftStatus {
        self.sample_count += 1;
        
        // Update cumulative sum (assuming we want to detect increases in loss)
        let mean_loss = A::from(0.1).unwrap(); // Estimated mean under H0
        self.sum = self.sum + loss - mean_loss;
        
        // Update minimum
        if self.sum < self.min_sum {
            self.min_sum = self.sum;
        }
        
        // Compute test statistic
        let test_stat = self.sum - self.min_sum;
        
        if test_stat > self.threshold {
            self.last_drift = Some(Instant::now());
            self.reset();
            DriftStatus::Drift
        } else if test_stat > self.warning_threshold {
            DriftStatus::Warning
        } else {
            DriftStatus::Stable
        }
    }
    
    /// Reset detector state
    pub fn reset(&mut self) {
        self.sum = A::zero();
        self.min_sum = A::zero();
        self.sample_count = 0;
    }
}

/// ADWIN (Adaptive Windowing) drift detector
#[derive(Debug, Clone)]
pub struct AdwinDetector<A: Float> {
    /// Window of recent values
    window: VecDeque<A>,
    /// Maximum window size
    max_window_size: usize,
    /// Detection confidence level
    delta: A,
    /// Minimum window size for detection
    min_window_size: usize,
}

impl<A: Float> AdwinDetector<A> {
    /// Create a new ADWIN detector
    pub fn new(delta: A, max_window_size: usize) -> Self {
        Self {
            window: VecDeque::new(),
            max_window_size,
            delta,
            min_window_size: 10,
        }
    }
    
    /// Update detector with new value
    pub fn update(&mut self, value: A) -> DriftStatus {
        self.window.push_back(value);
        
        // Maintain window size
        if self.window.len() > self.max_window_size {
            self.window.pop_front();
        }
        
        // Check for drift
        if self.window.len() >= self.min_window_size {
            if self.detect_change() {
                self.shrink_window();
                DriftStatus::Drift
            } else {
                DriftStatus::Stable
            }
        } else {
            DriftStatus::Stable
        }
    }
    
    /// Detect change using ADWIN algorithm
    fn detect_change(&self) -> bool {
        let n = self.window.len();
        if n < 2 {
            return false;
        }
        
        // Simplified ADWIN: check for significant difference between halves
        let mid = n / 2;
        
        let first_half: Vec<_> = self.window.iter().take(mid).cloned().collect();
        let second_half: Vec<_> = self.window.iter().skip(mid).cloned().collect();
        
        let mean1 = first_half.iter().cloned().sum::<A>() / A::from(first_half.len()).unwrap();
        let mean2 = second_half.iter().cloned().sum::<A>() / A::from(second_half.len()).unwrap();
        
        // Compute variance
        let var1 = first_half.iter()
            .map(|&x| {
                let diff = x - mean1;
                diff * diff
            })
            .sum::<A>() / A::from(first_half.len()).unwrap();
            
        let var2 = second_half.iter()
            .map(|&x| {
                let diff = x - mean2;
                diff * diff
            })
            .sum::<A>() / A::from(second_half.len()).unwrap();
        
        // Simplified change detection
        let diff = (mean1 - mean2).abs();
        let threshold = (var1 + var2 + A::from(0.01).unwrap()).sqrt();
        
        diff > threshold
    }
    
    /// Shrink window after drift detection
    fn shrink_window(&mut self) {
        let new_size = self.window.len() / 2;
        while self.window.len() > new_size {
            self.window.pop_front();
        }
    }
}

/// DDM (Drift Detection Method) detector
#[derive(Debug, Clone)]
pub struct DdmDetector<A: Float> {
    /// Error rate
    error_rate: A,
    /// Standard deviation of error rate
    error_std: A,
    /// Minimum error rate + 2*std
    min_error_plus_2std: A,
    /// Minimum error rate + 3*std
    min_error_plus_3std: A,
    /// Sample count
    sample_count: usize,
    /// Error count
    error_count: usize,
}

impl<A: Float> DdmDetector<A> {
    /// Create a new DDM detector
    pub fn new() -> Self {
        Self {
            error_rate: A::zero(),
            error_std: A::one(),
            min_error_plus_2std: A::from(f64::MAX).unwrap(),
            min_error_plus_3std: A::from(f64::MAX).unwrap(),
            sample_count: 0,
            error_count: 0,
        }
    }
    
    /// Update with prediction result
    pub fn update(&mut self, is_error: bool) -> DriftStatus {
        self.sample_count += 1;
        if is_error {
            self.error_count += 1;
        }
        
        if self.sample_count < 30 {
            return DriftStatus::Stable;
        }
        
        // Update error rate and standard deviation
        self.error_rate = A::from(self.error_count as f64 / self.sample_count as f64).unwrap();
        let p = self.error_rate;
        let n = A::from(self.sample_count as f64).unwrap();
        self.error_std = (p * (A::one() - p) / n).sqrt();
        
        let current_level = self.error_rate + A::from(2.0).unwrap() * self.error_std;
        
        // Update minimums
        if current_level < self.min_error_plus_2std {
            self.min_error_plus_2std = current_level;
            self.min_error_plus_3std = self.error_rate + A::from(3.0).unwrap() * self.error_std;
        }
        
        // Check for drift
        if current_level > self.min_error_plus_3std {
            self.reset();
            DriftStatus::Drift
        } else if current_level > self.min_error_plus_2std {
            DriftStatus::Warning
        } else {
            DriftStatus::Stable
        }
    }
    
    /// Reset detector state
    pub fn reset(&mut self) {
        self.sample_count = 0;
        self.error_count = 0;
        self.error_rate = A::zero();
        self.error_std = A::one();
        self.min_error_plus_2std = A::from(f64::MAX).unwrap();
        self.min_error_plus_3std = A::from(f64::MAX).unwrap();
    }
}

/// Comprehensive concept drift detector
pub struct ConceptDriftDetector<A: Float> {
    /// Configuration
    config: DriftDetectorConfig,
    
    /// Page-Hinkley detector
    ph_detector: PageHinkleyDetector<A>,
    
    /// ADWIN detector
    adwin_detector: AdwinDetector<A>,
    
    /// DDM detector
    ddm_detector: DdmDetector<A>,
    
    /// Ensemble voting history
    ensemble_history: VecDeque<DriftStatus>,
    
    /// Drift events history
    drift_events: Vec<DriftEvent<A>>,
    
    /// Performance before/after drift
    performance_tracker: PerformanceDriftTracker<A>,
}

impl<A: Float + std::fmt::Debug> ConceptDriftDetector<A> {
    /// Create a new concept drift detector
    pub fn new(config: DriftDetectorConfig) -> Self {
        let threshold = A::from(config.threshold).unwrap();
        let warning_threshold = A::from(config.warning_threshold).unwrap();
        let delta = A::from(config.alpha).unwrap();
        
        Self {
            ph_detector: PageHinkleyDetector::new(threshold, warning_threshold),
            adwin_detector: AdwinDetector::new(delta, config.window_size),
            ddm_detector: DdmDetector::new(),
            ensemble_history: VecDeque::with_capacity(10),
            drift_events: Vec::new(),
            performance_tracker: PerformanceDriftTracker::new(),
            config,
        }
    }
    
    /// Update detector with new loss and prediction error
    pub fn update(&mut self, loss: A, is_prediction_error: bool) -> Result<DriftStatus, OptimizerError> {
        let ph_status = self.ph_detector.update(loss);
        let adwin_status = self.adwin_detector.update(loss);
        let ddm_status = self.ddm_detector.update(is_prediction_error);
        
        let final_status = if self.config.enable_ensemble {
            self.ensemble_vote(ph_status, adwin_status, ddm_status)
        } else {
            match self.config.method {
                DriftDetectionMethod::PageHinkley => ph_status,
                DriftDetectionMethod::Adwin => adwin_status,
                DriftDetectionMethod::DriftDetectionMethod => ddm_status,
                _ => ph_status, // Default fallback
            }
        };
        
        // Record drift event if detected
        if final_status == DriftStatus::Drift {
            let event = DriftEvent {
                timestamp: Instant::now(),
                confidence: A::from(0.8).unwrap(), // Simplified confidence
                drift_type: self.classify_drift_type(),
                adaptation_recommendation: self.generate_adaptation_recommendation(),
            };
            self.drift_events.push(event);
        }
        
        // Update performance tracking
        self.performance_tracker.update(loss, final_status.clone());
        
        Ok(final_status)
    }
    
    /// Ensemble voting among detectors
    fn ensemble_vote(&mut self, ph: DriftStatus, adwin: DriftStatus, ddm: DriftStatus) -> DriftStatus {
        let votes = vec![ph, adwin, ddm];
        
        // Count votes
        let drift_votes = votes.iter().filter(|&&s| s == DriftStatus::Drift).count();
        let warning_votes = votes.iter().filter(|&&s| s == DriftStatus::Warning).count();
        
        if drift_votes >= 2 {
            DriftStatus::Drift
        } else if warning_votes >= 2 || drift_votes >= 1 {
            DriftStatus::Warning
        } else {
            DriftStatus::Stable
        }
    }
    
    /// Classify the type of drift based on recent history
    fn classify_drift_type(&self) -> DriftType {
        // Simplified classification based on recent drift events
        if self.drift_events.len() < 2 {
            return DriftType::Sudden;
        }
        
        let recent_events = self.drift_events.iter().rev().take(5);
        let time_intervals: Vec<_> = recent_events
            .map(|event| event.timestamp)
            .collect::<Vec<_>>()
            .windows(2)
            .map(|window| window[0].duration_since(window[1]))
            .collect();
        
        if time_intervals.iter().all(|&d| d < Duration::from_secs(60)) {
            DriftType::Sudden
        } else if time_intervals.len() > 2 {
            DriftType::Gradual
        } else {
            DriftType::Incremental
        }
    }
    
    /// Generate adaptation recommendation based on drift characteristics
    fn generate_adaptation_recommendation(&self) -> AdaptationRecommendation {
        let recent_performance = self.performance_tracker.get_recent_performance_change();
        
        if recent_performance > 0.5 {
            // Significant performance degradation
            AdaptationRecommendation::Reset
        } else if recent_performance > 0.2 {
            // Moderate degradation
            AdaptationRecommendation::IncreaseLearningRate { factor: 1.5 }
        } else if recent_performance < -0.1 {
            // Performance improved (suspicious)
            AdaptationRecommendation::DecreaseLearningRate { factor: 0.8 }
        } else {
            AdaptationRecommendation::NoAction
        }
    }
    
    /// Get drift detection statistics
    pub fn get_statistics(&self) -> DriftStatistics<A> {
        DriftStatistics {
            total_drifts: self.drift_events.len(),
            recent_drift_rate: self.calculate_recent_drift_rate(),
            average_drift_confidence: self.calculate_average_confidence(),
            drift_types_distribution: self.calculate_drift_type_distribution(),
            time_since_last_drift: self.time_since_last_drift(),
        }
    }
    
    fn calculate_recent_drift_rate(&self) -> f64 {
        // Calculate drift rate in the last hour
        let one_hour_ago = Instant::now() - Duration::from_secs(3600);
        let recent_drifts = self.drift_events.iter()
            .filter(|event| event.timestamp > one_hour_ago)
            .count();
        recent_drifts as f64 / 3600.0 // Drifts per second
    }
    
    fn calculate_average_confidence(&self) -> Option<A> {
        if self.drift_events.is_empty() {
            None
        } else {
            let sum = self.drift_events.iter()
                .map(|event| event.confidence)
                .sum::<A>();
            Some(sum / A::from(self.drift_events.len()).unwrap())
        }
    }
    
    fn calculate_drift_type_distribution(&self) -> std::collections::HashMap<DriftType, usize> {
        let mut distribution = std::collections::HashMap::new();
        for event in &self.drift_events {
            *distribution.entry(event.drift_type).or_insert(0) += 1;
        }
        distribution
    }
    
    fn time_since_last_drift(&self) -> Option<Duration> {
        self.drift_events.last()
            .map(|event| event.timestamp.elapsed())
    }
}

/// Performance tracker for drift impact analysis
#[derive(Debug, Clone)]
struct PerformanceDriftTracker<A: Float> {
    /// Performance history with drift annotations
    performance_history: VecDeque<(A, DriftStatus, Instant)>,
    /// Window size for analysis
    window_size: usize,
}

impl<A: Float> PerformanceDriftTracker<A> {
    fn new() -> Self {
        Self {
            performance_history: VecDeque::new(),
            window_size: 100,
        }
    }
    
    fn update(&mut self, performance: A, drift_status: DriftStatus) {
        self.performance_history.push_back((performance, drift_status, Instant::now()));
        
        // Maintain window size
        if self.performance_history.len() > self.window_size {
            self.performance_history.pop_front();
        }
    }
    
    /// Get recent performance change (positive = degradation, negative = improvement)
    fn get_recent_performance_change(&self) -> f64 {
        if self.performance_history.len() < 10 {
            return 0.0;
        }
        
        let recent: Vec<_> = self.performance_history.iter().rev().take(10).collect();
        let older: Vec<_> = self.performance_history.iter().rev().skip(10).take(10).collect();
        
        if older.is_empty() {
            return 0.0;
        }
        
        let recent_avg = recent.iter().map(|(p, _, _)| *p).sum::<A>() / A::from(recent.len()).unwrap();
        let older_avg = older.iter().map(|(p, _, _)| *p).sum::<A>() / A::from(older.len()).unwrap();
        
        (recent_avg - older_avg).to_f64().unwrap_or(0.0)
    }
}

/// Drift detection statistics
#[derive(Debug, Clone)]
pub struct DriftStatistics<A: Float> {
    /// Total number of drifts detected
    pub total_drifts: usize,
    /// Recent drift rate (drifts per second)
    pub recent_drift_rate: f64,
    /// Average confidence of drift detections
    pub average_drift_confidence: Option<A>,
    /// Distribution of drift types
    pub drift_types_distribution: std::collections::HashMap<DriftType, usize>,
    /// Time since last drift
    pub time_since_last_drift: Option<Duration>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_page_hinkley_detector() {
        let mut detector = PageHinkleyDetector::new(3.0f64, 2.0f64);
        
        // Stable period
        for _ in 0..10 {
            let status = detector.update(0.1);
            assert_eq!(status, DriftStatus::Stable);
        }
        
        // Drift period
        for _ in 0..5 {
            let status = detector.update(0.5); // Higher loss
            if status == DriftStatus::Drift {
                break;
            }
        }
    }
    
    #[test]
    fn test_adwin_detector() {
        let mut detector = AdwinDetector::new(0.005f64, 100);
        
        // Add stable values
        for i in 0..20 {
            let value = 0.1 + (i as f64) * 0.001; // Slight trend
            detector.update(value);
        }
        
        // Add drift values
        for i in 0..10 {
            let value = 0.5 + (i as f64) * 0.01; // Clear change
            let status = detector.update(value);
            if status == DriftStatus::Drift {
                break;
            }
        }
    }
    
    #[test]
    fn test_ddm_detector() {
        let mut detector = DdmDetector::new();
        
        // Stable period with low error rate
        for i in 0..50 {
            let is_error = i % 10 == 0; // 10% error rate
            detector.update(is_error);
        }
        
        // Period with high error rate
        for i in 0..20 {
            let is_error = i % 2 == 0; // 50% error rate
            let status = detector.update(is_error);
            if status == DriftStatus::Drift {
                break;
            }
        }
    }
    
    #[test]
    fn test_concept_drift_detector() {
        let config = DriftDetectorConfig::default();
        let mut detector = ConceptDriftDetector::new(config);
        
        // Simulate stable period
        for i in 0..30 {
            let loss = 0.1 + (i as f64) * 0.001;
            let is_error = i % 10 == 0;
            let status = detector.update(loss, is_error).unwrap();
            assert_ne!(status, DriftStatus::Drift); // Should be stable
        }
        
        // Simulate drift
        for i in 0..20 {
            let loss = 0.5 + (i as f64) * 0.01; // Much higher loss
            let is_error = i % 2 == 0; // Higher error rate
            let _status = detector.update(loss, is_error).unwrap();
        }
        
        let stats = detector.get_statistics();
        assert!(stats.total_drifts > 0 || stats.recent_drift_rate > 0.0);
    }
    
    #[test]
    fn test_drift_event() {
        let event = DriftEvent {
            timestamp: Instant::now(),
            confidence: 0.85f64,
            drift_type: DriftType::Sudden,
            adaptation_recommendation: AdaptationRecommendation::Reset,
        };
        
        assert_eq!(event.drift_type, DriftType::Sudden);
        assert!(event.confidence > 0.8);
    }
}