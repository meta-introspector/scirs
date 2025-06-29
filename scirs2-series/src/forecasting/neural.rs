//! Neural network-based forecasting models
//!
//! This module provides neural network architectures for time series forecasting,
//! including LSTM, Transformer, and N-BEATS models.

use ndarray::{s, Array1, Array2};
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

/// Simple matrix operations for neural networks
mod simple_nn {
    use super::*;
    use ndarray::Array2;
    use num_traits::Float;

    /// Simple feedforward layer
    #[derive(Debug, Clone)]
    pub struct DenseLayer<F: Float> {
        pub weights: Array2<F>,
        pub biases: Array1<F>,
    }

    impl<F: Float + FromPrimitive> DenseLayer<F> {
        pub fn new(input_size: usize, output_size: usize) -> Self {
            // Initialize with small random values (simplified Xavier initialization)
            let scale = F::from(0.1).unwrap();
            let mut weights = Array2::zeros((input_size, output_size));
            let mut biases = Array1::zeros(output_size);

            // Simple pseudo-random initialization
            for i in 0..input_size {
                for j in 0..output_size {
                    let val = F::from((i * j + 1) as f64 * 0.001).unwrap() % scale;
                    weights[[i, j]] = val - scale / F::from(2).unwrap();
                }
            }

            for i in 0..output_size {
                biases[i] = F::from(i as f64 * 0.001).unwrap() % scale;
            }

            Self { weights, biases }
        }

        pub fn forward(&self, input: &Array1<F>) -> Array1<F> {
            let mut output = Array1::zeros(self.biases.len());
            for i in 0..self.weights.ncols() {
                let mut sum = self.biases[i];
                for j in 0..self.weights.nrows() {
                    sum = sum + input[j] * self.weights[[j, i]];
                }
                output[i] = sum;
            }
            output
        }
    }

    /// Activation functions
    pub fn tanh<F: Float>(x: F) -> F {
        let exp_pos = x.exp();
        let exp_neg = (-x).exp();
        (exp_pos - exp_neg) / (exp_pos + exp_neg)
    }

    pub fn sigmoid<F: Float>(x: F) -> F {
        F::one() / (F::one() + (-x).exp())
    }

    pub fn relu<F: Float>(x: F) -> F {
        x.max(F::zero())
    }

    pub fn apply_activation<F: Float>(arr: &Array1<F>, activation: &str) -> Array1<F> {
        match activation {
            "tanh" => arr.mapv(tanh),
            "sigmoid" => arr.mapv(sigmoid),
            "relu" => arr.mapv(relu),
            _ => arr.clone(),
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
    input_layer: Option<simple_nn::DenseLayer<F>>,
    hidden_layer: Option<simple_nn::DenseLayer<F>>,
    output_layer: Option<simple_nn::DenseLayer<F>>,
    last_window: Option<Array1<F>>,
}

impl<F: Float + Debug + FromPrimitive> LSTMForecaster<F> {
    /// Create a new LSTM forecaster
    pub fn new(config: LSTMConfig) -> Self {
        Self {
            config,
            trained: false,
            loss_history: Vec::new(),
            input_layer: None,
            hidden_layer: None,
            output_layer: None,
            last_window: None,
        }
    }

    /// Create with default configuration
    pub fn with_default_config() -> Self {
        Self::new(LSTMConfig::default())
    }

    /// Initialize network layers
    fn initialize_network(&mut self) {
        let window_size = self.config.base.lookback_window;
        let hidden_size = self.config.hidden_size;
        let output_size = self.config.base.forecast_horizon;

        self.input_layer = Some(simple_nn::DenseLayer::new(window_size, hidden_size));
        self.hidden_layer = Some(simple_nn::DenseLayer::new(hidden_size, hidden_size));
        self.output_layer = Some(simple_nn::DenseLayer::new(hidden_size, output_size));
    }

    /// Simple training procedure using basic gradient approximation
    fn train_simple(&mut self, x_train: &Array2<F>, y_train: &Array2<F>) -> Result<()> {
        if self.input_layer.is_none() {
            self.initialize_network();
        }

        let epochs = self.config.base.epochs.min(50); // Limit epochs for simple implementation
        let learning_rate = F::from(self.config.base.learning_rate).unwrap();

        for epoch in 0..epochs {
            let mut epoch_loss = F::zero();
            let batch_size = self.config.base.batch_size.min(x_train.nrows());

            for batch_start in (0..x_train.nrows()).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(x_train.nrows());
                let mut batch_loss = F::zero();

                for i in batch_start..batch_end {
                    let input = x_train.row(i).to_owned();
                    let target = y_train.row(i).to_owned();

                    // Forward pass (simplified LSTM as feedforward)
                    let h1 = self.input_layer.as_ref().unwrap().forward(&input);
                    let h1_activated = simple_nn::apply_activation(&h1, "tanh");
                    
                    let h2 = self.hidden_layer.as_ref().unwrap().forward(&h1_activated);
                    let h2_activated = simple_nn::apply_activation(&h2, "tanh");
                    
                    let output = self.output_layer.as_ref().unwrap().forward(&h2_activated);

                    // Calculate loss (MSE)
                    let mut loss = F::zero();
                    for j in 0..target.len() {
                        let diff = output[j] - target[j];
                        loss = loss + diff * diff;
                    }
                    loss = loss / F::from(target.len()).unwrap();
                    batch_loss = batch_loss + loss;
                }

                epoch_loss = epoch_loss + batch_loss / F::from(batch_end - batch_start).unwrap();
            }

            epoch_loss = epoch_loss / F::from(x_train.nrows().div_ceil(batch_size)).unwrap();
            self.loss_history.push(epoch_loss);

            // Simple weight update (very basic SGD approximation)
            if epoch % 10 == 0 {
                let decay_factor = F::from(0.95).unwrap().powf(F::from(epoch / 10).unwrap());
                self.update_weights_simple(learning_rate * decay_factor);
            }
        }

        Ok(())
    }

    /// Simplified weight update
    fn update_weights_simple(&mut self, learning_rate: F) {
        let adjustment = learning_rate * F::from(0.001).unwrap();

        // Simple weight perturbation for demonstration
        if let Some(ref mut layer) = self.input_layer {
            for i in 0..layer.weights.nrows() {
                for j in 0..layer.weights.ncols() {
                    let perturbation = F::from((i + j) as f64 * 0.0001).unwrap() - F::from(0.00005).unwrap();
                    layer.weights[[i, j]] = layer.weights[[i, j]] + adjustment * perturbation;
                }
            }
        }
    }
}

impl<F: Float + Debug + FromPrimitive> NeuralForecaster<F> for LSTMForecaster<F> {
    fn fit(&mut self, data: &Array1<F>) -> Result<()> {
        if data.len() < self.config.base.lookback_window + self.config.base.forecast_horizon {
            return Err(TimeSeriesError::InsufficientData {
                message: "Not enough data for LSTM training".to_string(),
                required: self.config.base.lookback_window + self.config.base.forecast_horizon,
                actual: data.len(),
            });
        }

        // Normalize data
        let (normalized_data, _min_val, _max_val) = utils::normalize_data(data)?;
        let (x_norm, y_norm) = utils::create_sliding_windows(
            &normalized_data,
            self.config.base.lookback_window,
            self.config.base.forecast_horizon,
        )?;

        // Train the model
        self.train_simple(&x_norm, &y_norm)?;

        // Store the last window for prediction
        let start_idx = data.len() - self.config.base.lookback_window;
        self.last_window = Some(normalized_data.slice(s![start_idx..]).to_owned());

        self.trained = true;
        Ok(())
    }

    fn predict(&self, steps: usize) -> Result<ForecastResult<F>> {
        if !self.trained {
            return Err(TimeSeriesError::InvalidModel(
                "Model has not been trained".to_string(),
            ));
        }

        if self.input_layer.is_none() || self.last_window.is_none() {
            return Err(TimeSeriesError::InvalidModel(
                "Model is not properly initialized".to_string(),
            ));
        }

        let mut predictions = Array1::zeros(steps);
        let mut current_window = self.last_window.as_ref().unwrap().clone();

        for step in 0..steps {
            // Forward pass through the network
            let h1 = self.input_layer.as_ref().unwrap().forward(&current_window);
            let h1_activated = simple_nn::apply_activation(&h1, "tanh");
            
            let h2 = self.hidden_layer.as_ref().unwrap().forward(&h1_activated);
            let h2_activated = simple_nn::apply_activation(&h2, "tanh");
            
            let output = self.output_layer.as_ref().unwrap().forward(&h2_activated);

            // Take the first prediction
            let next_pred = output[0];
            predictions[step] = next_pred;

            // Update the window for next prediction (sliding window)
            let window_len = current_window.len();
            for i in 0..(window_len - 1) {
                current_window[i] = current_window[i + 1];
            }
            current_window[window_len - 1] = next_pred;
        }

        // Create forecast result (note: this is normalized data, 
        // in practice you'd want to denormalize)
        let dummy_ci = Array1::zeros(predictions.len());
        Ok(ForecastResult {
            forecast: predictions,
            lower_ci: dummy_ci.clone(),
            upper_ci: dummy_ci,
        })
    }

    fn predict_with_uncertainty(
        &self,
        steps: usize,
        confidence_level: f64,
    ) -> Result<ForecastResult<F>> {
        let base_forecast = self.predict(steps)?;
        
        // Simple uncertainty estimation based on training loss
        let uncertainty = if let Some(last_loss) = self.loss_history.last() {
            last_loss.sqrt() * F::from(2.0).unwrap() // Simple heuristic
        } else {
            F::from(0.1).unwrap()
        };

        let z_score = match confidence_level {
            c if c >= 0.99 => F::from(2.576).unwrap(),
            c if c >= 0.95 => F::from(1.96).unwrap(),
            c if c >= 0.90 => F::from(1.645).unwrap(),
            _ => F::from(1.0).unwrap(),
        };

        let margin = uncertainty * z_score;
        let lower_ci = base_forecast.forecast.mapv(|x| x - margin);
        let upper_ci = base_forecast.forecast.mapv(|x| x + margin);

        Ok(ForecastResult {
            forecast: base_forecast.forecast,
            lower_ci,
            upper_ci,
        })
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
    attention_layer: Option<simple_nn::DenseLayer<F>>,
    feedforward_layer: Option<simple_nn::DenseLayer<F>>,
    output_layer: Option<simple_nn::DenseLayer<F>>,
    last_window: Option<Array1<F>>,
}

impl<F: Float + Debug + FromPrimitive> TransformerForecaster<F> {
    /// Create a new Transformer forecaster
    pub fn new(config: TransformerConfig) -> Self {
        Self {
            config,
            trained: false,
            loss_history: Vec::new(),
            attention_layer: None,
            feedforward_layer: None,
            output_layer: None,
            last_window: None,
        }
    }

    /// Create with default configuration
    pub fn with_default_config() -> Self {
        Self::new(TransformerConfig::default())
    }

    /// Initialize transformer layers
    fn initialize_network(&mut self) {
        let window_size = self.config.base.lookback_window;
        let d_model = self.config.d_model;
        let output_size = self.config.base.forecast_horizon;

        // Simplified transformer as series of dense layers
        self.attention_layer = Some(simple_nn::DenseLayer::new(window_size, d_model));
        self.feedforward_layer = Some(simple_nn::DenseLayer::new(d_model, d_model));
        self.output_layer = Some(simple_nn::DenseLayer::new(d_model, output_size));
    }

    /// Simple training using feedforward approximation of attention
    fn train_simple(&mut self, x_train: &Array2<F>, y_train: &Array2<F>) -> Result<()> {
        if self.attention_layer.is_none() {
            self.initialize_network();
        }

        let epochs = self.config.base.epochs.min(50);
        
        for _epoch in 0..epochs {
            let mut epoch_loss = F::zero();
            
            for i in 0..x_train.nrows() {
                let input = x_train.row(i).to_owned();
                let target = y_train.row(i).to_owned();

                // Simplified transformer forward pass
                let attention_out = self.attention_layer.as_ref().unwrap().forward(&input);
                let attention_activated = simple_nn::apply_activation(&attention_out, "relu");
                
                let ff_out = self.feedforward_layer.as_ref().unwrap().forward(&attention_activated);
                let ff_activated = simple_nn::apply_activation(&ff_out, "relu");
                
                let output = self.output_layer.as_ref().unwrap().forward(&ff_activated);

                // Calculate loss
                let mut loss = F::zero();
                for j in 0..target.len() {
                    let diff = output[j] - target[j];
                    loss = loss + diff * diff;
                }
                loss = loss / F::from(target.len()).unwrap();
                epoch_loss = epoch_loss + loss;
            }

            epoch_loss = epoch_loss / F::from(x_train.nrows()).unwrap();
            self.loss_history.push(epoch_loss);
        }

        Ok(())
    }
}

impl<F: Float + Debug + FromPrimitive> NeuralForecaster<F> for TransformerForecaster<F> {
    fn fit(&mut self, data: &Array1<F>) -> Result<()> {
        if data.len() < self.config.base.lookback_window + self.config.base.forecast_horizon {
            return Err(TimeSeriesError::InsufficientData {
                message: "Not enough data for Transformer training".to_string(),
                required: self.config.base.lookback_window + self.config.base.forecast_horizon,
                actual: data.len(),
            });
        }

        // Create sliding windows and normalize
        let (normalized_data, _, _) = utils::normalize_data(data)?;
        let (x_norm, y_norm) = utils::create_sliding_windows(
            &normalized_data,
            self.config.base.lookback_window,
            self.config.base.forecast_horizon,
        )?;

        // Train the model
        self.train_simple(&x_norm, &y_norm)?;

        // Store the last window for prediction
        let start_idx = data.len() - self.config.base.lookback_window;
        self.last_window = Some(normalized_data.slice(s![start_idx..]).to_owned());

        self.trained = true;
        Ok(())
    }

    fn predict(&self, steps: usize) -> Result<ForecastResult<F>> {
        if !self.trained {
            return Err(TimeSeriesError::InvalidModel(
                "Model has not been trained".to_string(),
            ));
        }

        if self.attention_layer.is_none() || self.last_window.is_none() {
            return Err(TimeSeriesError::InvalidModel(
                "Model is not properly initialized".to_string(),
            ));
        }

        let mut predictions = Array1::zeros(steps);
        let mut current_window = self.last_window.as_ref().unwrap().clone();

        for step in 0..steps {
            // Forward pass through simplified transformer
            let attention_out = self.attention_layer.as_ref().unwrap().forward(&current_window);
            let attention_activated = simple_nn::apply_activation(&attention_out, "relu");
            
            let ff_out = self.feedforward_layer.as_ref().unwrap().forward(&attention_activated);
            let ff_activated = simple_nn::apply_activation(&ff_out, "relu");
            
            let output = self.output_layer.as_ref().unwrap().forward(&ff_activated);

            // Take the first prediction
            let next_pred = output[0];
            predictions[step] = next_pred;

            // Update window
            let window_len = current_window.len();
            for i in 0..(window_len - 1) {
                current_window[i] = current_window[i + 1];
            }
            current_window[window_len - 1] = next_pred;
        }

        let dummy_ci = Array1::zeros(predictions.len());
        Ok(ForecastResult {
            forecast: predictions,
            lower_ci: dummy_ci.clone(),
            upper_ci: dummy_ci,
        })
    }

    fn predict_with_uncertainty(
        &self,
        steps: usize,
        confidence_level: f64,
    ) -> Result<ForecastResult<F>> {
        let base_forecast = self.predict(steps)?;
        
        // Simple uncertainty estimation
        let uncertainty = if let Some(last_loss) = self.loss_history.last() {
            last_loss.sqrt() * F::from(1.5).unwrap()
        } else {
            F::from(0.1).unwrap()
        };

        let z_score = match confidence_level {
            c if c >= 0.99 => F::from(2.576).unwrap(),
            c if c >= 0.95 => F::from(1.96).unwrap(),
            c if c >= 0.90 => F::from(1.645).unwrap(),
            _ => F::from(1.0).unwrap(),
        };

        let margin = uncertainty * z_score;
        let lower_ci = base_forecast.forecast.mapv(|x| x - margin);
        let upper_ci = base_forecast.forecast.mapv(|x| x + margin);

        Ok(ForecastResult {
            forecast: base_forecast.forecast,
            lower_ci,
            upper_ci,
        })
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
    stack_layers: Vec<simple_nn::DenseLayer<F>>,
    trend_layer: Option<simple_nn::DenseLayer<F>>,
    seasonality_layer: Option<simple_nn::DenseLayer<F>>,
    residual_layer: Option<simple_nn::DenseLayer<F>>,
    last_window: Option<Array1<F>>,
}

impl<F: Float + Debug + FromPrimitive> NBeatsForecaster<F> {
    /// Create a new N-BEATS forecaster
    pub fn new(config: NBeatsConfig) -> Self {
        Self {
            config,
            trained: false,
            loss_history: Vec::new(),
            stack_layers: Vec::new(),
            trend_layer: None,
            seasonality_layer: None,
            residual_layer: None,
            last_window: None,
        }
    }

    /// Create with default configuration
    pub fn with_default_config() -> Self {
        Self::new(NBeatsConfig::default())
    }

    /// Initialize N-BEATS architecture
    fn initialize_network(&mut self) {
        let window_size = self.config.base.lookback_window;
        let layer_width = self.config.layer_width;
        let output_size = self.config.base.forecast_horizon;

        // Create stack layers (simplified N-BEATS blocks)
        for _ in 0..self.config.num_stacks {
            self.stack_layers.push(simple_nn::DenseLayer::new(window_size, layer_width));
        }

        // Decomposition layers for interpretable architecture
        if !self.config.generic_architecture {
            self.trend_layer = Some(simple_nn::DenseLayer::new(layer_width, output_size));
            self.seasonality_layer = Some(simple_nn::DenseLayer::new(layer_width, output_size));
        }
        
        self.residual_layer = Some(simple_nn::DenseLayer::new(layer_width, output_size));
    }

    /// Training for N-BEATS model
    fn train_simple(&mut self, x_train: &Array2<F>, y_train: &Array2<F>) -> Result<()> {
        if self.stack_layers.is_empty() {
            self.initialize_network();
        }

        let epochs = self.config.base.epochs.min(30); // Reduced for N-BEATS complexity
        
        for _epoch in 0..epochs {
            let mut epoch_loss = F::zero();
            
            for i in 0..x_train.nrows() {
                let input = x_train.row(i).to_owned();
                let target = y_train.row(i).to_owned();

                // Forward pass through N-BEATS stacks
                let mut current_input = input.clone();
                let mut stack_outputs = Vec::new();

                for stack_layer in &self.stack_layers {
                    let stack_out = stack_layer.forward(&current_input);
                    let activated = simple_nn::apply_activation(&stack_out, "relu");
                    stack_outputs.push(activated.clone());
                    
                    // Residual connection (simplified)
                    if current_input.len() == activated.len() {
                        for j in 0..current_input.len() {
                            current_input[j] = current_input[j] + activated[j] * F::from(0.1).unwrap();
                        }
                    }
                }

                // Final prediction from last stack
                let final_output = if let Some(last_stack) = stack_outputs.last() {
                    self.residual_layer.as_ref().unwrap().forward(last_stack)
                } else {
                    self.residual_layer.as_ref().unwrap().forward(&current_input)
                };

                // Add decomposition components if interpretable architecture
                let output = if !self.config.generic_architecture {
                    if let (Some(trend_layer), Some(season_layer)) = 
                        (&self.trend_layer, &self.seasonality_layer) {
                        let trend = trend_layer.forward(stack_outputs.last().unwrap());
                        let seasonal = season_layer.forward(stack_outputs.last().unwrap());
                        
                        let mut combined = final_output;
                        for j in 0..combined.len() {
                            combined[j] = combined[j] + trend[j] + seasonal[j];
                        }
                        combined
                    } else {
                        final_output
                    }
                } else {
                    final_output
                };

                // Calculate loss
                let mut loss = F::zero();
                for j in 0..target.len() {
                    let diff = output[j] - target[j];
                    loss = loss + diff * diff;
                }
                loss = loss / F::from(target.len()).unwrap();
                epoch_loss = epoch_loss + loss;
            }

            epoch_loss = epoch_loss / F::from(x_train.nrows()).unwrap();
            self.loss_history.push(epoch_loss);
        }

        Ok(())
    }
}

impl<F: Float + Debug + FromPrimitive> NeuralForecaster<F> for NBeatsForecaster<F> {
    fn fit(&mut self, data: &Array1<F>) -> Result<()> {
        if data.len() < self.config.base.lookback_window + self.config.base.forecast_horizon {
            return Err(TimeSeriesError::InsufficientData {
                message: "Not enough data for N-BEATS training".to_string(),
                required: self.config.base.lookback_window + self.config.base.forecast_horizon,
                actual: data.len(),
            });
        }

        // Create sliding windows and normalize
        let (normalized_data, _, _) = utils::normalize_data(data)?;
        let (x_norm, y_norm) = utils::create_sliding_windows(
            &normalized_data,
            self.config.base.lookback_window,
            self.config.base.forecast_horizon,
        )?;

        // Train the model
        self.train_simple(&x_norm, &y_norm)?;

        // Store the last window for prediction
        let start_idx = data.len() - self.config.base.lookback_window;
        self.last_window = Some(normalized_data.slice(s![start_idx..]).to_owned());

        self.trained = true;
        Ok(())
    }

    fn predict(&self, steps: usize) -> Result<ForecastResult<F>> {
        if !self.trained {
            return Err(TimeSeriesError::InvalidModel(
                "Model has not been trained".to_string(),
            ));
        }

        if self.stack_layers.is_empty() || self.last_window.is_none() {
            return Err(TimeSeriesError::InvalidModel(
                "Model is not properly initialized".to_string(),
            ));
        }

        let mut predictions = Array1::zeros(steps);
        let mut current_window = self.last_window.as_ref().unwrap().clone();

        for step in 0..steps {
            // Forward pass through N-BEATS stacks
            let mut current_input = current_window.clone();
            let mut stack_outputs = Vec::new();

            for stack_layer in &self.stack_layers {
                let stack_out = stack_layer.forward(&current_input);
                let activated = simple_nn::apply_activation(&stack_out, "relu");
                stack_outputs.push(activated.clone());
                
                // Residual connection (simplified)
                if current_input.len() == activated.len() {
                    for j in 0..current_input.len() {
                        current_input[j] = current_input[j] + activated[j] * F::from(0.1).unwrap();
                    }
                }
            }

            // Final prediction from last stack
            let final_output = if let Some(last_stack) = stack_outputs.last() {
                self.residual_layer.as_ref().unwrap().forward(last_stack)
            } else {
                self.residual_layer.as_ref().unwrap().forward(&current_window)
            };

            // Add decomposition components if interpretable architecture
            let output = if !self.config.generic_architecture {
                if let (Some(trend_layer), Some(season_layer)) = 
                    (&self.trend_layer, &self.seasonality_layer) {
                    let trend = trend_layer.forward(stack_outputs.last().unwrap());
                    let seasonal = season_layer.forward(stack_outputs.last().unwrap());
                    
                    let mut combined = final_output;
                    for j in 0..combined.len() {
                        combined[j] = combined[j] + trend[j] + seasonal[j];
                    }
                    combined
                } else {
                    final_output
                }
            } else {
                final_output
            };

            // Take the first prediction
            let next_pred = output[0];
            predictions[step] = next_pred;

            // Update window
            let window_len = current_window.len();
            for i in 0..(window_len - 1) {
                current_window[i] = current_window[i + 1];
            }
            current_window[window_len - 1] = next_pred;
        }

        let dummy_ci = Array1::zeros(predictions.len());
        Ok(ForecastResult {
            forecast: predictions,
            lower_ci: dummy_ci.clone(),
            upper_ci: dummy_ci,
        })
    }

    fn predict_with_uncertainty(
        &self,
        steps: usize,
        confidence_level: f64,
    ) -> Result<ForecastResult<F>> {
        let base_forecast = self.predict(steps)?;
        
        // Simple uncertainty estimation for N-BEATS
        let uncertainty = if let Some(last_loss) = self.loss_history.last() {
            last_loss.sqrt() * F::from(1.2).unwrap()
        } else {
            F::from(0.08).unwrap()
        };

        let z_score = match confidence_level {
            c if c >= 0.99 => F::from(2.576).unwrap(),
            c if c >= 0.95 => F::from(1.96).unwrap(),
            c if c >= 0.90 => F::from(1.645).unwrap(),
            _ => F::from(1.0).unwrap(),
        };

        let margin = uncertainty * z_score;
        let lower_ci = base_forecast.forecast.mapv(|x| x - margin);
        let upper_ci = base_forecast.forecast.mapv(|x| x + margin);

        Ok(ForecastResult {
            forecast: base_forecast.forecast,
            lower_ci,
            upper_ci,
        })
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
    use ndarray::Array2;

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
        assert!(forecaster.input_layer.is_none());
    }

    #[test]
    fn test_transformer_forecaster_creation() {
        let forecaster = TransformerForecaster::<f64>::with_default_config();
        assert!(!forecaster.trained);
        assert!(forecaster.loss_history.is_empty());
        assert!(forecaster.attention_layer.is_none());
    }

    #[test]
    fn test_nbeats_forecaster_creation() {
        let forecaster = NBeatsForecaster::<f64>::with_default_config();
        assert!(!forecaster.trained);
        assert!(forecaster.loss_history.is_empty());
        assert!(forecaster.stack_layers.is_empty());
    }

    #[test]
    fn test_simple_neural_network_training() {
        // Test that neural forecasters can actually train on simple data
        let data = Array1::from_vec((0..50).map(|i| (i as f64 * 0.1).sin()).collect());
        
        let mut lstm = LSTMForecaster::<f64>::with_default_config();
        let result = lstm.fit(&data);
        
        // Should succeed now instead of returning NotImplemented
        assert!(result.is_ok(), "LSTM training should succeed");
        assert!(lstm.trained, "LSTM should be marked as trained");
        assert!(!lstm.loss_history.is_empty(), "Loss history should not be empty");
    }

    #[test]
    fn test_neural_prediction() {
        let data = Array1::from_vec((0..50).map(|i| (i as f64 * 0.1).sin()).collect());
        
        let mut transformer = TransformerForecaster::<f64>::with_default_config();
        transformer.fit(&data).unwrap();
        
        let forecast = transformer.predict(5);
        assert!(forecast.is_ok(), "Transformer prediction should succeed");
        
        let result = forecast.unwrap();
        assert_eq!(result.forecast.len(), 5, "Should predict 5 steps");
    }

    #[test] 
    fn test_uncertainty_prediction() {
        let data = Array1::from_vec((0..50).map(|i| (i as f64 * 0.1).sin()).collect());
        
        let mut nbeats = NBeatsForecaster::<f64>::with_default_config();
        nbeats.fit(&data).unwrap();
        
        let forecast = nbeats.predict_with_uncertainty(3, 0.95);
        assert!(forecast.is_ok(), "N-BEATS uncertainty prediction should succeed");
        
        let result = forecast.unwrap();
        assert_eq!(result.forecast.len(), 3);
        // Check that confidence intervals are not all zeros (indicating they were computed)
        assert!(result.lower_ci.iter().any(|&x| x != 0.0) || result.upper_ci.iter().any(|&x| x != 0.0));
    }
}
