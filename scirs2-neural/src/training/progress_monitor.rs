//! Training Progress Monitoring System
//!
//! This module provides comprehensive training progress monitoring, including
//! metrics tracking, visualization callbacks, and early stopping mechanisms.

use crate::error::{NeuralError, Result};
use ndarray::{Array1, Array2};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Training progress monitor
#[derive(Debug)]
pub struct TrainingProgressMonitor {
    /// Training metrics history
    metrics_history: HashMap<String, VecDeque<f32>>,
    /// Validation metrics history
    validation_history: HashMap<String, VecDeque<f32>>,
    /// Training time per epoch
    epoch_times: VecDeque<Duration>,
    /// Current epoch
    current_epoch: usize,
    /// Total epochs planned
    total_epochs: usize,
    /// Configuration
    config: ProgressMonitorConfig,
    /// Early stopping state
    early_stopping: EarlyStoppingState,
    /// Best metrics tracker
    best_metrics: HashMap<String, f32>,
    /// Start time of training
    training_start_time: Option<Instant>,
}

/// Configuration for progress monitoring
#[derive(Debug, Clone)]
pub struct ProgressMonitorConfig {
    /// Maximum history length to keep
    pub max_history_length: usize,
    /// Metrics to track for early stopping
    pub early_stopping_metric: String,
    /// Early stopping patience (epochs)
    pub early_stopping_patience: usize,
    /// Minimum improvement threshold for early stopping
    pub min_improvement_threshold: f32,
    /// Whether to maximize or minimize the early stopping metric
    pub maximize_metric: bool,
    /// Print progress every N epochs
    pub print_every: usize,
    /// Save checkpoint every N epochs
    pub checkpoint_every: usize,
    /// Validation frequency (epochs)
    pub validation_frequency: usize,
}

impl Default for ProgressMonitorConfig {
    fn default() -> Self {
        Self {
            max_history_length: 1000,
            early_stopping_metric: "val_loss".to_string(),
            early_stopping_patience: 10,
            min_improvement_threshold: 1e-4,
            maximize_metric: false,
            print_every: 1,
            checkpoint_every: 10,
            validation_frequency: 1,
        }
    }
}

/// Early stopping state
#[derive(Debug, Clone)]
struct EarlyStoppingState {
    /// Best metric value seen so far
    best_value: f32,
    /// Epochs without improvement
    epochs_without_improvement: usize,
    /// Whether early stopping should trigger
    should_stop: bool,
    /// Epoch where best value was achieved
    best_epoch: usize,
}

impl Default for EarlyStoppingState {
    fn default() -> Self {
        Self {
            best_value: f32::NEG_INFINITY,
            epochs_without_improvement: 0,
            should_stop: false,
            best_epoch: 0,
        }
    }
}

/// Training step information
#[derive(Debug, Clone)]
pub struct TrainingStepInfo {
    /// Current epoch
    pub epoch: usize,
    /// Current batch within epoch
    pub batch: usize,
    /// Total batches in epoch
    pub total_batches: usize,
    /// Current loss value
    pub loss: f32,
    /// Current learning rate
    pub learning_rate: f32,
    /// Batch processing time
    pub batch_time: Duration,
    /// Additional metrics
    pub metrics: HashMap<String, f32>,
}

/// Validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Validation loss
    pub loss: f32,
    /// Validation accuracy (if applicable)
    pub accuracy: Option<f32>,
    /// Additional validation metrics
    pub metrics: HashMap<String, f32>,
    /// Time taken for validation
    pub validation_time: Duration,
}

impl TrainingProgressMonitor {
    /// Create a new training progress monitor
    pub fn new(total_epochs: usize, config: ProgressMonitorConfig) -> Self {
        let mut early_stopping = EarlyStoppingState::default();
        
        // Initialize best value based on whether we're maximizing or minimizing
        early_stopping.best_value = if config.maximize_metric {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        };

        Self {
            metrics_history: HashMap::new(),
            validation_history: HashMap::new(),
            epoch_times: VecDeque::new(),
            current_epoch: 0,
            total_epochs,
            config,
            early_stopping,
            best_metrics: HashMap::new(),
            training_start_time: None,
        }
    }

    /// Start training monitoring
    pub fn start_training(&mut self) {
        self.training_start_time = Some(Instant::now());
        self.current_epoch = 0;
        println!("🚀 Starting training for {} epochs", self.total_epochs);
    }

    /// Record metrics for the current epoch
    pub fn record_epoch_metrics(&mut self, metrics: HashMap<String, f32>, epoch_time: Duration) {
        self.epoch_times.push_back(epoch_time);
        
        // Limit history size
        if self.epoch_times.len() > self.config.max_history_length {
            self.epoch_times.pop_front();
        }

        // Record each metric
        for (name, value) in metrics {
            let history = self.metrics_history
                .entry(name.clone())
                .or_insert_with(VecDeque::new);
            
            history.push_back(value);
            
            // Limit history size
            if history.len() > self.config.max_history_length {
                history.pop_front();
            }

            // Update best metrics
            self.update_best_metric(&name, value);
        }

        self.current_epoch += 1;

        // Print progress if needed
        if self.current_epoch % self.config.print_every == 0 {
            self.print_progress();
        }
    }

    /// Record validation metrics
    pub fn record_validation_metrics(&mut self, result: ValidationResult) -> Result<bool> {
        // Record validation metrics
        let mut all_metrics = result.metrics.clone();
        all_metrics.insert("val_loss".to_string(), result.loss);
        
        if let Some(accuracy) = result.accuracy {
            all_metrics.insert("val_accuracy".to_string(), accuracy);
        }

        for (name, value) in all_metrics {
            let history = self.validation_history
                .entry(name.clone())
                .or_insert_with(VecDeque::new);
            
            history.push_back(value);
            
            // Limit history size
            if history.len() > self.config.max_history_length {
                history.pop_front();
            }

            // Update best metrics
            self.update_best_metric(&name, value);
        }

        // Check early stopping
        let should_stop = self.check_early_stopping(&result)?;
        
        if should_stop {
            println!("⏹️  Early stopping triggered at epoch {}", self.current_epoch);
            println!("   Best {} achieved at epoch {}: {:.6}", 
                    self.config.early_stopping_metric,
                    self.early_stopping.best_epoch,
                    self.early_stopping.best_value);
        }

        Ok(should_stop)
    }

    /// Update best metric tracking
    fn update_best_metric(&mut self, name: &str, value: f32) {
        let current_best = self.best_metrics.get(name).copied().unwrap_or(
            if self.is_metric_maximized(name) {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            }
        );

        let is_better = if self.is_metric_maximized(name) {
            value > current_best
        } else {
            value < current_best
        };

        if is_better {
            self.best_metrics.insert(name.to_string(), value);
        }
    }

    /// Check if a metric should be maximized (e.g., accuracy) or minimized (e.g., loss)
    fn is_metric_maximized(&self, metric_name: &str) -> bool {
        metric_name.contains("accuracy") || 
        metric_name.contains("precision") || 
        metric_name.contains("recall") || 
        metric_name.contains("f1") ||
        metric_name.contains("auc")
    }

    /// Check early stopping conditions
    fn check_early_stopping(&mut self, result: &ValidationResult) -> Result<bool> {
        // Get the metric value for early stopping
        let metric_value = if self.config.early_stopping_metric == "val_loss" {
            result.loss
        } else {
            result.metrics.get(&self.config.early_stopping_metric)
                .copied()
                .ok_or_else(|| NeuralError::InvalidArgument(
                    format!("Early stopping metric '{}' not found in validation results", 
                           self.config.early_stopping_metric)
                ))?
        };

        let is_improvement = if self.config.maximize_metric {
            metric_value > self.early_stopping.best_value + self.config.min_improvement_threshold
        } else {
            metric_value < self.early_stopping.best_value - self.config.min_improvement_threshold
        };

        if is_improvement {
            self.early_stopping.best_value = metric_value;
            self.early_stopping.epochs_without_improvement = 0;
            self.early_stopping.best_epoch = self.current_epoch;
        } else {
            self.early_stopping.epochs_without_improvement += 1;
        }

        if self.early_stopping.epochs_without_improvement >= self.config.early_stopping_patience {
            self.early_stopping.should_stop = true;
        }

        Ok(self.early_stopping.should_stop)
    }

    /// Print current training progress
    pub fn print_progress(&self) {
        let elapsed = self.training_start_time
            .map(|start| start.elapsed())
            .unwrap_or_default();
        
        let avg_epoch_time = if !self.epoch_times.is_empty() {
            self.epoch_times.iter().sum::<Duration>() / self.epoch_times.len() as u32
        } else {
            Duration::from_secs(0)
        };

        let eta = avg_epoch_time * (self.total_epochs - self.current_epoch) as u32;

        println!("📊 Epoch {}/{} | Elapsed: {:?} | ETA: {:?}", 
                self.current_epoch, self.total_epochs, elapsed, eta);

        // Print recent metrics
        if let Some(recent_loss) = self.get_recent_metric("loss") {
            print!("   Loss: {:.6}", recent_loss);
        }

        if let Some(recent_acc) = self.get_recent_metric("accuracy") {
            print!(" | Acc: {:.4}", recent_acc);
        }

        if let Some(recent_val_loss) = self.get_recent_metric("val_loss") {
            print!(" | Val Loss: {:.6}", recent_val_loss);
        }

        if let Some(recent_val_acc) = self.get_recent_metric("val_accuracy") {
            print!(" | Val Acc: {:.4}", recent_val_acc);
        }

        println!();

        // Print best metrics achieved so far
        if !self.best_metrics.is_empty() && self.current_epoch % (self.config.print_every * 5) == 0 {
            println!("🏆 Best metrics so far:");
            for (name, value) in &self.best_metrics {
                println!("   {}: {:.6}", name, value);
            }
        }
    }

    /// Get the most recent value for a metric
    fn get_recent_metric(&self, name: &str) -> Option<f32> {
        self.metrics_history.get(name)
            .and_then(|history| history.back().copied())
            .or_else(|| {
                self.validation_history.get(name)
                    .and_then(|history| history.back().copied())
            })
    }

    /// Get metric history
    pub fn get_metric_history(&self, name: &str) -> Option<Vec<f32>> {
        self.metrics_history.get(name)
            .or_else(|| self.validation_history.get(name))
            .map(|history| history.iter().copied().collect())
    }

    /// Get training summary
    pub fn get_training_summary(&self) -> TrainingSummary {
        let total_time = self.training_start_time
            .map(|start| start.elapsed())
            .unwrap_or_default();

        let avg_epoch_time = if !self.epoch_times.is_empty() {
            self.epoch_times.iter().sum::<Duration>() / self.epoch_times.len() as u32
        } else {
            Duration::from_secs(0)
        };

        TrainingSummary {
            total_epochs: self.current_epoch,
            total_time,
            avg_epoch_time,
            best_metrics: self.best_metrics.clone(),
            final_metrics: self.get_final_metrics(),
            early_stopped: self.early_stopping.should_stop,
            best_epoch: self.early_stopping.best_epoch,
        }
    }

    /// Get final metrics (last recorded values)
    fn get_final_metrics(&self) -> HashMap<String, f32> {
        let mut final_metrics = HashMap::new();

        for (name, history) in &self.metrics_history {
            if let Some(&value) = history.back() {
                final_metrics.insert(name.clone(), value);
            }
        }

        for (name, history) in &self.validation_history {
            if let Some(&value) = history.back() {
                final_metrics.insert(name.clone(), value);
            }
        }

        final_metrics
    }

    /// Should training continue?
    pub fn should_continue(&self) -> bool {
        !self.early_stopping.should_stop && self.current_epoch < self.total_epochs
    }

    /// Get current epoch
    pub fn current_epoch(&self) -> usize {
        self.current_epoch
    }

    /// Generate training curves data for plotting
    #[allow(dead_code)]
    pub fn get_training_curves(&self) -> HashMap<String, (Vec<usize>, Vec<f32>)> {
        let mut curves = HashMap::new();

        for (name, history) in &self.metrics_history {
            let epochs: Vec<usize> = (1..=history.len()).collect();
            let values: Vec<f32> = history.iter().copied().collect();
            curves.insert(name.clone(), (epochs, values));
        }

        for (name, history) in &self.validation_history {
            let epochs: Vec<usize> = (1..=history.len()).collect();
            let values: Vec<f32> = history.iter().copied().collect();
            curves.insert(name.clone(), (epochs, values));
        }

        curves
    }
}

/// Training summary after completion
#[derive(Debug, Clone)]
pub struct TrainingSummary {
    /// Total epochs completed
    pub total_epochs: usize,
    /// Total training time
    pub total_time: Duration,
    /// Average time per epoch
    pub avg_epoch_time: Duration,
    /// Best metrics achieved
    pub best_metrics: HashMap<String, f32>,
    /// Final metrics
    pub final_metrics: HashMap<String, f32>,
    /// Whether training was early stopped
    pub early_stopped: bool,
    /// Epoch where best metric was achieved
    pub best_epoch: usize,
}

impl TrainingSummary {
    /// Print a comprehensive training summary
    pub fn print_summary(&self) {
        println!("\n🎯 Training Summary");
        println!("================");
        println!("Total epochs: {}", self.total_epochs);
        println!("Total time: {:?}", self.total_time);
        println!("Average epoch time: {:?}", self.avg_epoch_time);
        
        if self.early_stopped {
            println!("Early stopped: Yes (best at epoch {})", self.best_epoch);
        } else {
            println!("Early stopped: No");
        }

        println!("\n🏆 Best Metrics:");
        for (name, value) in &self.best_metrics {
            println!("  {}: {:.6}", name, value);
        }

        println!("\n📈 Final Metrics:");
        for (name, value) in &self.final_metrics {
            println!("  {}: {:.6}", name, value);
        }
        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progress_monitor_creation() {
        let config = ProgressMonitorConfig::default();
        let monitor = TrainingProgressMonitor::new(100, config);
        
        assert_eq!(monitor.total_epochs, 100);
        assert_eq!(monitor.current_epoch, 0);
    }

    #[test]
    fn test_metric_recording() {
        let config = ProgressMonitorConfig::default();
        let mut monitor = TrainingProgressMonitor::new(10, config);
        
        let mut metrics = HashMap::new();
        metrics.insert("loss".to_string(), 0.5);
        metrics.insert("accuracy".to_string(), 0.85);
        
        monitor.record_epoch_metrics(metrics, Duration::from_secs(30));
        
        assert_eq!(monitor.current_epoch, 1);
        assert_eq!(monitor.get_recent_metric("loss"), Some(0.5));
        assert_eq!(monitor.get_recent_metric("accuracy"), Some(0.85));
    }

    #[test]
    fn test_early_stopping() {
        let mut config = ProgressMonitorConfig::default();
        config.early_stopping_patience = 2;
        config.min_improvement_threshold = 0.01;
        
        let mut monitor = TrainingProgressMonitor::new(10, config);
        
        // First validation - baseline
        let result1 = ValidationResult {
            loss: 1.0,
            accuracy: Some(0.5),
            metrics: HashMap::new(),
            validation_time: Duration::from_secs(5),
        };
        
        let should_stop = monitor.record_validation_metrics(result1).unwrap();
        assert!(!should_stop);
        
        // Second validation - no improvement
        let result2 = ValidationResult {
            loss: 1.1,
            accuracy: Some(0.45),
            metrics: HashMap::new(),
            validation_time: Duration::from_secs(5),
        };
        
        let should_stop = monitor.record_validation_metrics(result2).unwrap();
        assert!(!should_stop);
        
        // Third validation - still no improvement, should trigger early stopping
        let result3 = ValidationResult {
            loss: 1.2,
            accuracy: Some(0.4),
            metrics: HashMap::new(),
            validation_time: Duration::from_secs(5),
        };
        
        let should_stop = monitor.record_validation_metrics(result3).unwrap();
        assert!(should_stop);
    }

    #[test]
    fn test_best_metrics_tracking() {
        let config = ProgressMonitorConfig::default();
        let mut monitor = TrainingProgressMonitor::new(10, config);
        
        // Record some metrics
        let mut metrics1 = HashMap::new();
        metrics1.insert("loss".to_string(), 0.8);
        metrics1.insert("accuracy".to_string(), 0.6);
        monitor.record_epoch_metrics(metrics1, Duration::from_secs(30));
        
        let mut metrics2 = HashMap::new();
        metrics2.insert("loss".to_string(), 0.5);  // Better (lower)
        metrics2.insert("accuracy".to_string(), 0.85);  // Better (higher)
        monitor.record_epoch_metrics(metrics2, Duration::from_secs(30));
        
        let summary = monitor.get_training_summary();
        assert_eq!(summary.best_metrics.get("loss"), Some(&0.5));
        assert_eq!(summary.best_metrics.get("accuracy"), Some(&0.85));
    }
}