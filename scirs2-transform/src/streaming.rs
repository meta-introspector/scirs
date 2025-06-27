//! Streaming transformations for continuous data processing
//!
//! This module provides utilities for processing data streams in real-time,
//! maintaining running statistics and transforming data incrementally.

use ndarray::{Array1, Array2, ArrayBase, Data, Ix1, Ix2};
use num_traits::{Float, NumCast};
use std::collections::VecDeque;

use crate::error::{Result, TransformError};

/// Trait for transformers that support streaming/incremental updates
pub trait StreamingTransformer: Send + Sync {
    /// Update the transformer with a new batch of data
    fn partial_fit(&mut self, x: &Array2<f64>) -> Result<()>;
    
    /// Transform a batch of data using current statistics
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>>;
    
    /// Reset the transformer to initial state
    fn reset(&mut self);
    
    /// Get the number of samples seen so far
    fn n_samples_seen(&self) -> usize;
}

/// Streaming standard scaler that maintains running statistics
pub struct StreamingStandardScaler {
    /// Running mean for each feature
    mean: Array1<f64>,
    /// Running variance for each feature
    variance: Array1<f64>,
    /// Number of samples seen
    n_samples: usize,
    /// Whether to center the data
    with_mean: bool,
    /// Whether to scale to unit variance
    with_std: bool,
    /// Epsilon for numerical stability
    epsilon: f64,
}

impl StreamingStandardScaler {
    /// Create a new streaming standard scaler
    pub fn new(n_features: usize, with_mean: bool, with_std: bool) -> Self {
        StreamingStandardScaler {
            mean: Array1::zeros(n_features),
            variance: Array1::zeros(n_features),
            n_samples: 0,
            with_mean,
            with_std,
            epsilon: 1e-8,
        }
    }
    
    /// Update statistics using Welford's online algorithm
    fn update_statistics(&mut self, x: &Array2<f64>) {
        let batch_size = x.shape()[0];
        let n_features = x.shape()[1];
        
        for i in 0..batch_size {
            self.n_samples += 1;
            let n = self.n_samples as f64;
            
            for j in 0..n_features {
                let value = x[[i, j]];
                let delta = value - self.mean[j];
                self.mean[j] += delta / n;
                
                if self.with_std {
                    let delta2 = value - self.mean[j];
                    self.variance[j] += delta * delta2;
                }
            }
        }
    }
    
    /// Get the current standard deviation
    fn get_std(&self) -> Array1<f64> {
        if self.n_samples <= 1 {
            Array1::ones(self.mean.len())
        } else {
            self.variance.mapv(|v| (v / (self.n_samples - 1) as f64).sqrt().max(self.epsilon))
        }
    }
}

impl StreamingTransformer for StreamingStandardScaler {
    fn partial_fit(&mut self, x: &Array2<f64>) -> Result<()> {
        if x.shape()[1] != self.mean.len() {
            return Err(TransformError::InvalidInput(
                format!("Expected {} features, got {}", self.mean.len(), x.shape()[1])
            ));
        }
        
        self.update_statistics(x);
        Ok(())
    }
    
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        if x.shape()[1] != self.mean.len() {
            return Err(TransformError::InvalidInput(
                format!("Expected {} features, got {}", self.mean.len(), x.shape()[1])
            ));
        }
        
        let mut result = x.to_owned();
        
        if self.with_mean {
            for i in 0..result.shape()[0] {
                for j in 0..result.shape()[1] {
                    result[[i, j]] -= self.mean[j];
                }
            }
        }
        
        if self.with_std {
            let std = self.get_std();
            for i in 0..result.shape()[0] {
                for j in 0..result.shape()[1] {
                    result[[i, j]] /= std[j];
                }
            }
        }
        
        Ok(result)
    }
    
    fn reset(&mut self) {
        self.mean.fill(0.0);
        self.variance.fill(0.0);
        self.n_samples = 0;
    }
    
    fn n_samples_seen(&self) -> usize {
        self.n_samples
    }
}

/// Streaming min-max scaler that tracks min and max values
pub struct StreamingMinMaxScaler {
    /// Minimum values for each feature
    min: Array1<f64>,
    /// Maximum values for each feature
    max: Array1<f64>,
    /// Target range
    feature_range: (f64, f64),
    /// Number of samples seen
    n_samples: usize,
}

impl StreamingMinMaxScaler {
    /// Create a new streaming min-max scaler
    pub fn new(n_features: usize, feature_range: (f64, f64)) -> Self {
        StreamingMinMaxScaler {
            min: Array1::from_elem(n_features, f64::INFINITY),
            max: Array1::from_elem(n_features, f64::NEG_INFINITY),
            feature_range,
            n_samples: 0,
        }
    }
    
    /// Update min and max values
    fn update_bounds(&mut self, x: &Array2<f64>) {
        for i in 0..x.shape()[0] {
            for j in 0..x.shape()[1] {
                let value = x[[i, j]];
                self.min[j] = self.min[j].min(value);
                self.max[j] = self.max[j].max(value);
            }
            self.n_samples += 1;
        }
    }
}

impl StreamingTransformer for StreamingMinMaxScaler {
    fn partial_fit(&mut self, x: &Array2<f64>) -> Result<()> {
        if x.shape()[1] != self.min.len() {
            return Err(TransformError::InvalidInput(
                format!("Expected {} features, got {}", self.min.len(), x.shape()[1])
            ));
        }
        
        self.update_bounds(x);
        Ok(())
    }
    
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        if x.shape()[1] != self.min.len() {
            return Err(TransformError::InvalidInput(
                format!("Expected {} features, got {}", self.min.len(), x.shape()[1])
            ));
        }
        
        let mut result = Array2::zeros(x.shape());
        let (min_val, max_val) = self.feature_range;
        let scale = max_val - min_val;
        
        for i in 0..x.shape()[0] {
            for j in 0..x.shape()[1] {
                let range = self.max[j] - self.min[j];
                if range > 1e-10 {
                    result[[i, j]] = (x[[i, j]] - self.min[j]) / range * scale + min_val;
                } else {
                    result[[i, j]] = (min_val + max_val) / 2.0;
                }
            }
        }
        
        Ok(result)
    }
    
    fn reset(&mut self) {
        self.min.fill(f64::INFINITY);
        self.max.fill(f64::NEG_INFINITY);
        self.n_samples = 0;
    }
    
    fn n_samples_seen(&self) -> usize {
        self.n_samples
    }
}

/// Streaming quantile tracker using P² algorithm
pub struct StreamingQuantileTracker {
    /// Quantiles to track
    quantiles: Vec<f64>,
    /// P² algorithm state for each feature and quantile
    p2_states: Vec<Vec<P2State>>,
    /// Number of features
    n_features: usize,
}

/// P² algorithm state for a single quantile
struct P2State {
    /// Marker positions
    n: [f64; 5],
    /// Marker values
    q: [f64; 5],
    /// Desired marker positions
    n_prime: [f64; 5],
    /// Number of observations
    count: usize,
    /// Target quantile
    p: f64,
}

impl P2State {
    fn new(p: f64) -> Self {
        P2State {
            n: [1.0, 2.0, 3.0, 4.0, 5.0],
            q: [0.0; 5],
            n_prime: [1.0, 1.0 + 2.0 * p, 1.0 + 4.0 * p, 3.0 + 2.0 * p, 5.0],
            count: 0,
            p,
        }
    }
    
    fn update(&mut self, value: f64) {
        if self.count < 5 {
            self.q[self.count] = value;
            self.count += 1;
            
            if self.count == 5 {
                // Sort initial observations
                self.q.sort_by(|a, b| a.partial_cmp(b).unwrap());
            }
            return;
        }
        
        // Find cell k such that q[k] <= value < q[k+1]
        let mut k = 0;
        for i in 1..5 {
            if value < self.q[i] {
                k = i - 1;
                break;
            }
        }
        if value >= self.q[4] {
            k = 3;
        }
        
        // Update marker positions
        for i in (k + 1)..5 {
            self.n[i] += 1.0;
        }
        
        // Update desired marker positions
        for i in 0..5 {
            self.n_prime[i] += match i {
                0 => 0.0,
                1 => self.p / 2.0,
                2 => self.p,
                3 => (1.0 + self.p) / 2.0,
                4 => 1.0,
                _ => unreachable!(),
            };
        }
        
        // Adjust marker values
        for i in 1..4 {
            let d = self.n_prime[i] - self.n[i];
            
            if (d >= 1.0 && self.n[i + 1] - self.n[i] > 1.0) ||
               (d <= -1.0 && self.n[i - 1] - self.n[i] < -1.0) {
                
                let d_sign = d.signum();
                
                // Try parabolic interpolation
                let qi = self.parabolic_interpolation(i, d_sign);
                
                if self.q[i - 1] < qi && qi < self.q[i + 1] {
                    self.q[i] = qi;
                } else {
                    // Fall back to linear interpolation
                    self.q[i] = self.linear_interpolation(i, d_sign);
                }
                
                self.n[i] += d_sign;
            }
        }
    }
    
    fn parabolic_interpolation(&self, i: usize, d: f64) -> f64 {
        let qi = self.q[i];
        let qim1 = self.q[i - 1];
        let qip1 = self.q[i + 1];
        let ni = self.n[i];
        let nim1 = self.n[i - 1];
        let nip1 = self.n[i + 1];
        
        qi + d / (nip1 - nim1) * (
            (ni - nim1 + d) * (qip1 - qi) / (nip1 - ni) +
            (nip1 - ni - d) * (qi - qim1) / (ni - nim1)
        )
    }
    
    fn linear_interpolation(&self, i: usize, d: f64) -> f64 {
        let j = if d > 0.0 { i + 1 } else { i - 1 };
        self.q[i] + d * (self.q[j] - self.q[i]) / (self.n[j] - self.n[i])
    }
    
    fn quantile(&self) -> f64 {
        if self.count < 5 {
            // Not enough data, return median of available values
            let mut sorted = self.q[..self.count].to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted[sorted.len() / 2]
        } else {
            self.q[2] // Middle marker estimates the quantile
        }
    }
}

impl StreamingQuantileTracker {
    /// Create a new streaming quantile tracker
    pub fn new(n_features: usize, quantiles: Vec<f64>) -> Result<Self> {
        // Validate quantiles
        for &q in &quantiles {
            if q < 0.0 || q > 1.0 {
                return Err(TransformError::InvalidInput(
                    format!("Quantile {} must be between 0 and 1", q)
                ));
            }
        }
        
        let mut p2_states = Vec::with_capacity(n_features);
        for _ in 0..n_features {
            let feature_states: Vec<P2State> = quantiles
                .iter()
                .map(|&q| P2State::new(q))
                .collect();
            p2_states.push(feature_states);
        }
        
        Ok(StreamingQuantileTracker {
            quantiles,
            p2_states,
            n_features,
        })
    }
    
    /// Update quantile estimates with new data
    pub fn update(&mut self, x: &Array2<f64>) -> Result<()> {
        if x.shape()[1] != self.n_features {
            return Err(TransformError::InvalidInput(
                format!("Expected {} features, got {}", self.n_features, x.shape()[1])
            ));
        }
        
        for i in 0..x.shape()[0] {
            for j in 0..x.shape()[1] {
                let value = x[[i, j]];
                for k in 0..self.quantiles.len() {
                    self.p2_states[j][k].update(value);
                }
            }
        }
        
        Ok(())
    }
    
    /// Get current quantile estimates
    pub fn get_quantiles(&self) -> Array2<f64> {
        let mut result = Array2::zeros((self.n_features, self.quantiles.len()));
        
        for j in 0..self.n_features {
            for k in 0..self.quantiles.len() {
                result[[j, k]] = self.p2_states[j][k].quantile();
            }
        }
        
        result
    }
}

/// Window-based streaming transformer that maintains a sliding window
pub struct WindowedStreamingTransformer<T: StreamingTransformer> {
    /// Underlying transformer
    transformer: T,
    /// Sliding window of recent data
    window: VecDeque<Array2<f64>>,
    /// Maximum window size
    window_size: usize,
    /// Current number of samples in window
    current_size: usize,
}

impl<T: StreamingTransformer> WindowedStreamingTransformer<T> {
    /// Create a new windowed streaming transformer
    pub fn new(transformer: T, window_size: usize) -> Self {
        WindowedStreamingTransformer {
            transformer,
            window: VecDeque::with_capacity(window_size),
            window_size,
            current_size: 0,
        }
    }
    
    /// Update the transformer with new data
    pub fn update(&mut self, x: &Array2<f64>) -> Result<()> {
        // Add new data to window
        self.window.push_back(x.to_owned());
        self.current_size += x.shape()[0];
        
        // Remove old data if window is full
        while self.current_size > self.window_size && !self.window.is_empty() {
            if let Some(old_data) = self.window.pop_front() {
                self.current_size -= old_data.shape()[0];
            }
        }
        
        // Refit transformer on window data
        self.transformer.reset();
        for data in &self.window {
            self.transformer.partial_fit(data)?;
        }
        
        Ok(())
    }
    
    /// Transform data using the windowed statistics
    pub fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.transformer.transform(x)
    }
}