//! Neural network-based forecasting models
//!
//! This module provides neural network architectures for time series forecasting,
//! including LSTM, Transformer, and N-BEATS models.

use ndarray::Array1;
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use crate::error::{Result, TimeSeriesError};
use crate::forecasting::ForecastResult;

/// Configuration for neural forecasting models
#[derive(Debug, Clone)]
pub struct NeuralConfig {
    /// Number of past time steps to use as input (lookback window)
    pub lookback_window: usize,
    /// Number of future time steps to forecast
    pub forecast_horizon: usize,
    /// Number of training epochs
    pub epochs: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Batch size for training
    pub batch_size: usize,
    /// Validation split ratio
    pub validation_split: f64,
    /// Early stopping patience
    pub early_stopping_patience: Option<usize>,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl Default for NeuralConfig {
    fn default() -> Self {
        Self {
            lookback_window: 24,
            forecast_horizon: 1,
            epochs: 100,
            learning_rate: 0.001,
            batch_size: 32,
            validation_split: 0.2,
            early_stopping_patience: Some(10),
            random_seed: Some(42),
        }
    }
}

/// LSTM network configuration
#[derive(Debug, Clone)]
pub struct LSTMConfig {
    /// Base neural network configuration
    pub base: NeuralConfig,
    /// Number of LSTM layers
    pub num_layers: usize,
    /// Hidden size for LSTM layers
    pub hidden_size: usize,
    /// Dropout rate
    pub dropout: f64,
    /// Whether to use bidirectional LSTM
    pub bidirectional: bool,
}

impl Default for LSTMConfig {
    fn default() -> Self {
        Self {
            base: NeuralConfig::default(),
            num_layers: 2,
            hidden_size: 64,
            dropout: 0.2,
            bidirectional: false,
        }
    }
}

/// Transformer network configuration
#[derive(Debug, Clone)]
pub struct TransformerConfig {
    /// Base neural network configuration
    pub base: NeuralConfig,
    /// Model dimension
    pub d_model: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of encoder layers
    pub num_encoder_layers: usize,
    /// Number of decoder layers
    pub num_decoder_layers: usize,
    /// Feedforward dimension
    pub d_ff: usize,
    /// Dropout rate
    pub dropout: f64,
    /// Use positional encoding
    pub use_positional_encoding: bool,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            base: NeuralConfig::default(),
            d_model: 64,
            num_heads: 8,
            num_encoder_layers: 3,
            num_decoder_layers: 3,
            d_ff: 256,
            dropout: 0.1,
            use_positional_encoding: true,
        }
    }
}

/// N-BEATS network configuration
#[derive(Debug, Clone)]
pub struct NBeatsConfig {
    /// Base neural network configuration
    pub base: NeuralConfig,
    /// Number of stacks
    pub num_stacks: usize,
    /// Number of blocks per stack
    pub num_blocks_per_stack: usize,
    /// Number of layers per block
    pub num_layers_per_block: usize,
    /// Layer width
    pub layer_width: usize,
    /// Expansion coefficient dimensions
    pub expansion_coefficient_dim: usize,
    /// Share weights in each stack
    pub share_weights_in_stack: bool,
    /// Generic architecture (if false, uses interpretable architecture)
    pub generic_architecture: bool,
}

impl Default for NBeatsConfig {
    fn default() -> Self {
        Self {
            base: NeuralConfig::default(),
            num_stacks: 30,
            num_blocks_per_stack: 1,
            num_layers_per_block: 4,
            layer_width: 512,
            expansion_coefficient_dim: 5,
            share_weights_in_stack: false,
            generic_architecture: true,
        }
    }
}

/// Neural forecasting model trait
pub trait NeuralForecaster<F: Float + Debug + FromPrimitive> {
    /// Train the model on the given time series data
    fn fit(&mut self, data: &Array1<F>) -> Result<()>;

    /// Make forecasts for the specified number of steps
    fn predict(&self, steps: usize) -> Result<ForecastResult<F>>;

    /// Make forecasts with confidence intervals
    fn predict_with_uncertainty(
        &self,
        steps: usize,
        confidence_level: f64,
    ) -> Result<ForecastResult<F>>;

    /// Get model configuration
    fn get_config(&self) -> &dyn std::any::Any;

    /// Get training loss history
    fn get_loss_history(&self) -> Option<&[F]>;
}

/// LSTM-based forecasting model
#[derive(Debug)]
pub struct LSTMForecaster<F: Float + Debug + FromPrimitive> {
    config: LSTMConfig,
    trained: bool,
    loss_history: Vec<F>,
    // Neural network model will be added when dependencies are available
    #[allow(dead_code)]
    model_weights: Option<Vec<F>>,
}

impl<F: Float + Debug + FromPrimitive> LSTMForecaster<F> {
    /// Create a new LSTM forecaster
    pub fn new(config: LSTMConfig) -> Self {
        Self {
            config,
            trained: false,
            loss_history: Vec::new(),
            model_weights: None,
        }
    }

    /// Create with default configuration
    pub fn with_default_config() -> Self {
        Self::new(LSTMConfig::default())
    }
}

impl<F: Float + Debug + FromPrimitive> NeuralForecaster<F> for LSTMForecaster<F> {
    fn fit(&mut self, _data: &Array1<F>) -> Result<()> {
        // TODO: Implement LSTM training when neural network dependencies are added
        // For now, return an error indicating the feature is not yet available
        Err(TimeSeriesError::NotImplemented(
            "LSTM forecasting requires neural network dependencies (candle or tch). \
             This feature will be available in the next release."
                .to_string(),
        ))
    }

    fn predict(&self, _steps: usize) -> Result<ForecastResult<F>> {
        if !self.trained {
            return Err(TimeSeriesError::InvalidModel(
                "Model has not been trained".to_string(),
            ));
        }

        Err(TimeSeriesError::NotImplemented(
            "LSTM forecasting requires neural network dependencies".to_string(),
        ))
    }

    fn predict_with_uncertainty(
        &self,
        steps: usize,
        _confidence_level: f64,
    ) -> Result<ForecastResult<F>> {
        // For neural networks, uncertainty can be estimated using:
        // 1. Monte Carlo dropout
        // 2. Ensemble methods
        // 3. Bayesian neural networks
        self.predict(steps)
    }

    fn get_config(&self) -> &dyn std::any::Any {
        &self.config
    }

    fn get_loss_history(&self) -> Option<&[F]> {
        if self.loss_history.is_empty() {
            None
        } else {
            Some(&self.loss_history)
        }
    }
}

/// Transformer-based forecasting model
#[derive(Debug)]
pub struct TransformerForecaster<F: Float + Debug + FromPrimitive> {
    config: TransformerConfig,
    trained: bool,
    loss_history: Vec<F>,
    #[allow(dead_code)]
    model_weights: Option<Vec<F>>,
}

impl<F: Float + Debug + FromPrimitive> TransformerForecaster<F> {
    /// Create a new Transformer forecaster
    pub fn new(config: TransformerConfig) -> Self {
        Self {
            config,
            trained: false,
            loss_history: Vec::new(),
            model_weights: None,
        }
    }

    /// Create with default configuration
    pub fn with_default_config() -> Self {
        Self::new(TransformerConfig::default())
    }
}

impl<F: Float + Debug + FromPrimitive> NeuralForecaster<F> for TransformerForecaster<F> {
    fn fit(&mut self, _data: &Array1<F>) -> Result<()> {
        Err(TimeSeriesError::NotImplemented(
            "Transformer forecasting requires neural network dependencies (candle or tch). \
             This feature will be available in the next release."
                .to_string(),
        ))
    }

    fn predict(&self, _steps: usize) -> Result<ForecastResult<F>> {
        if !self.trained {
            return Err(TimeSeriesError::InvalidModel(
                "Model has not been trained".to_string(),
            ));
        }

        Err(TimeSeriesError::NotImplemented(
            "Transformer forecasting requires neural network dependencies".to_string(),
        ))
    }

    fn predict_with_uncertainty(
        &self,
        steps: usize,
        _confidence_level: f64,
    ) -> Result<ForecastResult<F>> {
        self.predict(steps)
    }

    fn get_config(&self) -> &dyn std::any::Any {
        &self.config
    }

    fn get_loss_history(&self) -> Option<&[F]> {
        if self.loss_history.is_empty() {
            None
        } else {
            Some(&self.loss_history)
        }
    }
}

/// N-BEATS forecasting model
#[derive(Debug)]
pub struct NBeatsForecaster<F: Float + Debug + FromPrimitive> {
    config: NBeatsConfig,
    trained: bool,
    loss_history: Vec<F>,
    #[allow(dead_code)]
    model_weights: Option<Vec<F>>,
}

impl<F: Float + Debug + FromPrimitive> NBeatsForecaster<F> {
    /// Create a new N-BEATS forecaster
    pub fn new(config: NBeatsConfig) -> Self {
        Self {
            config,
            trained: false,
            loss_history: Vec::new(),
            model_weights: None,
        }
    }

    /// Create with default configuration
    pub fn with_default_config() -> Self {
        Self::new(NBeatsConfig::default())
    }
}

impl<F: Float + Debug + FromPrimitive> NeuralForecaster<F> for NBeatsForecaster<F> {
    fn fit(&mut self, _data: &Array1<F>) -> Result<()> {
        Err(TimeSeriesError::NotImplemented(
            "N-BEATS forecasting requires neural network dependencies (candle or tch). \
             This feature will be available in the next release."
                .to_string(),
        ))
    }

    fn predict(&self, _steps: usize) -> Result<ForecastResult<F>> {
        if !self.trained {
            return Err(TimeSeriesError::InvalidModel(
                "Model has not been trained".to_string(),
            ));
        }

        Err(TimeSeriesError::NotImplemented(
            "N-BEATS forecasting requires neural network dependencies".to_string(),
        ))
    }

    fn predict_with_uncertainty(
        &self,
        steps: usize,
        _confidence_level: f64,
    ) -> Result<ForecastResult<F>> {
        self.predict(steps)
    }

    fn get_config(&self) -> &dyn std::any::Any {
        &self.config
    }

    fn get_loss_history(&self) -> Option<&[F]> {
        if self.loss_history.is_empty() {
            None
        } else {
            Some(&self.loss_history)
        }
    }
}

/// Utility functions for neural forecasting
pub mod utils {
    use super::*;
    use ndarray::{Array2, Axis};

    /// Create sliding windows for time series data
    pub fn create_sliding_windows<F: Float + Clone>(
        data: &Array1<F>,
        window_size: usize,
        horizon: usize,
    ) -> Result<(Array2<F>, Array2<F>)> {
        if window_size == 0 || horizon == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "Window size and horizon must be positive".to_string(),
            ));
        }

        if data.len() < window_size + horizon {
            return Err(TimeSeriesError::InvalidInput(
                "Data length is too short for the specified window size and horizon".to_string(),
            ));
        }

        let num_samples = data.len() - window_size - horizon + 1;
        let mut x = Array2::zeros((num_samples, window_size));
        let mut y = Array2::zeros((num_samples, horizon));

        for i in 0..num_samples {
            for j in 0..window_size {
                x[[i, j]] = data[i + j];
            }
            for j in 0..horizon {
                y[[i, j]] = data[i + window_size + j];
            }
        }

        Ok((x, y))
    }

    /// Normalize data for neural network training
    pub fn normalize_data<F: Float + FromPrimitive>(data: &Array1<F>) -> Result<(Array1<F>, F, F)> {
        if data.is_empty() {
            return Err(TimeSeriesError::InvalidInput(
                "Data cannot be empty".to_string(),
            ));
        }

        let min_val = data.iter().cloned().fold(data[0], F::min);
        let max_val = data.iter().cloned().fold(data[0], F::max);

        if min_val == max_val {
            return Err(TimeSeriesError::InvalidInput(
                "Data has no variance, cannot normalize".to_string(),
            ));
        }

        let range = max_val - min_val;
        let normalized = data.mapv(|x| (x - min_val) / range);

        Ok((normalized, min_val, max_val))
    }

    /// Denormalize predictions back to original scale
    pub fn denormalize_data<F: Float>(
        normalized_data: &Array1<F>,
        min_val: F,
        max_val: F,
    ) -> Array1<F> {
        let range = max_val - min_val;
        normalized_data.mapv(|x| x * range + min_val)
    }

    /// Type alias for train-validation split result
    pub type TrainValSplit<F> = (Array2<F>, Array2<F>, Array2<F>, Array2<F>);

    /// Split data into training and validation sets
    pub fn train_val_split<F: Float + Clone>(
        x: &Array2<F>,
        y: &Array2<F>,
        val_ratio: f64,
    ) -> Result<TrainValSplit<F>> {
        if !(0.0..1.0).contains(&val_ratio) {
            return Err(TimeSeriesError::InvalidInput(
                "Validation ratio must be between 0 and 1".to_string(),
            ));
        }

        let n_samples = x.nrows();
        let n_val = (n_samples as f64 * val_ratio) as usize;
        let n_train = n_samples - n_val;

        let x_train = x.slice_axis(Axis(0), (0..n_train).into()).to_owned();
        let x_val = x
            .slice_axis(Axis(0), (n_train..n_samples).into())
            .to_owned();
        let y_train = y.slice_axis(Axis(0), (0..n_train).into()).to_owned();
        let y_val = y
            .slice_axis(Axis(0), (n_train..n_samples).into())
            .to_owned();

        Ok((x_train, x_val, y_train, y_val))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_sliding_windows() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let (x, y) = utils::create_sliding_windows(&data, 3, 2).unwrap();

        assert_eq!(x.nrows(), 6);
        assert_eq!(x.ncols(), 3);
        assert_eq!(y.nrows(), 6);
        assert_eq!(y.ncols(), 2);

        // Check first window
        assert_abs_diff_eq!(x[[0, 0]], 1.0);
        assert_abs_diff_eq!(x[[0, 1]], 2.0);
        assert_abs_diff_eq!(x[[0, 2]], 3.0);
        assert_abs_diff_eq!(y[[0, 0]], 4.0);
        assert_abs_diff_eq!(y[[0, 1]], 5.0);
    }

    #[test]
    fn test_normalize_data() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let (normalized, min_val, max_val) = utils::normalize_data(&data).unwrap();

        assert_abs_diff_eq!(min_val, 1.0);
        assert_abs_diff_eq!(max_val, 5.0);
        assert_abs_diff_eq!(normalized[0], 0.0);
        assert_abs_diff_eq!(normalized[4], 1.0);

        // Test denormalization
        let denormalized = utils::denormalize_data(&normalized, min_val, max_val);
        for i in 0..data.len() {
            assert_abs_diff_eq!(denormalized[i], data[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_train_val_split() {
        let x = Array2::from_shape_vec((10, 3), (0..30).map(|i| i as f64).collect()).unwrap();
        let y = Array2::from_shape_vec((10, 2), (0..20).map(|i| i as f64).collect()).unwrap();

        let (x_train, x_val, y_train, y_val) = utils::train_val_split(&x, &y, 0.2).unwrap();

        assert_eq!(x_train.nrows(), 8);
        assert_eq!(x_val.nrows(), 2);
        assert_eq!(y_train.nrows(), 8);
        assert_eq!(y_val.nrows(), 2);
    }

    #[test]
    fn test_neural_config_defaults() {
        let config = NeuralConfig::default();
        assert_eq!(config.lookback_window, 24);
        assert_eq!(config.forecast_horizon, 1);
        assert_eq!(config.epochs, 100);
        assert_eq!(config.batch_size, 32);
    }

    #[test]
    fn test_lstm_forecaster_creation() {
        let forecaster = LSTMForecaster::<f64>::with_default_config();
        assert!(!forecaster.trained);
        assert!(forecaster.loss_history.is_empty());
    }

    #[test]
    fn test_transformer_forecaster_creation() {
        let forecaster = TransformerForecaster::<f64>::with_default_config();
        assert!(!forecaster.trained);
        assert!(forecaster.loss_history.is_empty());
    }

    #[test]
    fn test_nbeats_forecaster_creation() {
        let forecaster = NBeatsForecaster::<f64>::with_default_config();
        assert!(!forecaster.trained);
        assert!(forecaster.loss_history.is_empty());
    }
}
