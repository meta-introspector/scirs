//! Neural Forecasting Models for Time Series
//!
//! This module provides foundation implementations for neural network-based
//! time series forecasting, including LSTM, GRU, and Transformer architectures.
//! These implementations focus on core algorithmic components and can be
//! extended with actual neural network frameworks.

use ndarray::{Array1, Array2, Array3};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use crate::error::{Result, TimeSeriesError};

/// Activation functions for neural networks
#[derive(Debug, Clone)]
pub enum ActivationFunction {
    /// Sigmoid activation
    Sigmoid,
    /// Hyperbolic tangent
    Tanh,
    /// Rectified Linear Unit
    ReLU,
    /// Gaussian Error Linear Unit
    GELU,
    /// Swish activation
    Swish,
    /// Linear activation (identity)
    Linear,
}

impl ActivationFunction {
    /// Apply activation function
    pub fn apply<F: Float>(&self, x: F) -> F {
        match self {
            ActivationFunction::Sigmoid => {
                let one = F::one();
                one / (one + (-x).exp())
            }
            ActivationFunction::Tanh => x.tanh(),
            ActivationFunction::ReLU => x.max(F::zero()),
            ActivationFunction::GELU => {
                // Approximation of GELU
                let half = F::from(0.5).unwrap();
                let one = F::one();
                let sqrt_2_pi = F::from(0.7978845608).unwrap(); // sqrt(2/π)
                let coeff = F::from(0.044715).unwrap();
                
                half * x * (one + (sqrt_2_pi * (x + coeff * x * x * x)).tanh())
            }
            ActivationFunction::Swish => {
                let sigmoid = F::one() / (F::one() + (-x).exp());
                x * sigmoid
            }
            ActivationFunction::Linear => x,
        }
    }

    /// Apply derivative of activation function
    pub fn derivative<F: Float>(&self, x: F) -> F {
        match self {
            ActivationFunction::Sigmoid => {
                let sigmoid = self.apply(x);
                sigmoid * (F::one() - sigmoid)
            }
            ActivationFunction::Tanh => {
                let tanh_x = x.tanh();
                F::one() - tanh_x * tanh_x
            }
            ActivationFunction::ReLU => {
                if x > F::zero() { F::one() } else { F::zero() }
            }
            ActivationFunction::GELU => {
                // Simplified derivative approximation
                let sigmoid = F::one() / (F::one() + (-x).exp());
                sigmoid
            }
            ActivationFunction::Swish => {
                let sigmoid = F::one() / (F::one() + (-x).exp());
                sigmoid * (F::one() + x * (F::one() - sigmoid))
            }
            ActivationFunction::Linear => F::one(),
        }
    }
}

/// LSTM cell state and hidden state
#[derive(Debug, Clone)]
pub struct LSTMState<F: Float> {
    /// Hidden state
    pub hidden: Array1<F>,
    /// Cell state
    pub cell: Array1<F>,
}

/// LSTM cell implementation
#[derive(Debug)]
pub struct LSTMCell<F: Float + Debug> {
    /// Input size
    input_size: usize,
    /// Hidden size
    hidden_size: usize,
    /// Forget gate weights
    w_forget: Array2<F>,
    /// Input gate weights
    w_input: Array2<F>,
    /// Candidate gate weights
    w_candidate: Array2<F>,
    /// Output gate weights
    w_output: Array2<F>,
    /// Bias terms
    bias: Array1<F>,
}

impl<F: Float + Debug + Clone + FromPrimitive> LSTMCell<F> {
    /// Create new LSTM cell with random initialization
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let total_input_size = input_size + hidden_size;
        
        // Initialize weights with Xavier/Glorot initialization
        let scale = F::from(2.0).unwrap() / F::from(total_input_size).unwrap();
        let std_dev = scale.sqrt();
        
        Self {
            input_size,
            hidden_size,
            w_forget: Self::random_matrix(hidden_size, total_input_size, std_dev),
            w_input: Self::random_matrix(hidden_size, total_input_size, std_dev),
            w_candidate: Self::random_matrix(hidden_size, total_input_size, std_dev),
            w_output: Self::random_matrix(hidden_size, total_input_size, std_dev),
            bias: Array1::zeros(4 * hidden_size), // Bias for all gates
        }
    }

    /// Initialize random matrix with given standard deviation
    fn random_matrix(rows: usize, cols: usize, std_dev: F) -> Array2<F> {
        let mut matrix = Array2::zeros((rows, cols));
        
        // Simple pseudo-random initialization (for production, use proper RNG)
        let mut seed = 12345;
        for i in 0..rows {
            for j in 0..cols {
                // Linear congruential generator
                seed = (seed.wrapping_mul(1103515245).wrapping_add(12345)) & 0x7fffffff;
                let rand_val = F::from(seed as f64 / 2147483647.0).unwrap();
                let normalized = (rand_val - F::from(0.5).unwrap()) * F::from(2.0).unwrap();
                matrix[[i, j]] = normalized * std_dev;
            }
        }
        
        matrix
    }

    /// Forward pass through LSTM cell
    pub fn forward(&self, input: &Array1<F>, prev_state: &LSTMState<F>) -> Result<LSTMState<F>> {
        if input.len() != self.input_size {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: self.input_size,
                actual: input.len(),
            });
        }

        if prev_state.hidden.len() != self.hidden_size || prev_state.cell.len() != self.hidden_size {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: self.hidden_size,
                actual: prev_state.hidden.len(),
            });
        }

        // Concatenate input and previous hidden state
        let mut combined_input = Array1::zeros(self.input_size + self.hidden_size);
        for (i, &val) in input.iter().enumerate() {
            combined_input[i] = val;
        }
        for (i, &val) in prev_state.hidden.iter().enumerate() {
            combined_input[self.input_size + i] = val;
        }

        // Compute gate values
        let forget_gate = self.compute_gate(&self.w_forget, &combined_input, 0);
        let input_gate = self.compute_gate(&self.w_input, &combined_input, self.hidden_size);
        let candidate_gate = self.compute_gate(&self.w_candidate, &combined_input, 2 * self.hidden_size);
        let output_gate = self.compute_gate(&self.w_output, &combined_input, 3 * self.hidden_size);

        // Apply activations
        let forget_activated = forget_gate.mapv(|x| ActivationFunction::Sigmoid.apply(x));
        let input_activated = input_gate.mapv(|x| ActivationFunction::Sigmoid.apply(x));
        let candidate_activated = candidate_gate.mapv(|x| ActivationFunction::Tanh.apply(x));
        let output_activated = output_gate.mapv(|x| ActivationFunction::Sigmoid.apply(x));

        // Update cell state
        let mut new_cell = Array1::zeros(self.hidden_size);
        for i in 0..self.hidden_size {
            new_cell[i] = forget_activated[i] * prev_state.cell[i] + 
                         input_activated[i] * candidate_activated[i];
        }

        // Update hidden state
        let cell_tanh = new_cell.mapv(|x| x.tanh());
        let mut new_hidden = Array1::zeros(self.hidden_size);
        for i in 0..self.hidden_size {
            new_hidden[i] = output_activated[i] * cell_tanh[i];
        }

        Ok(LSTMState {
            hidden: new_hidden,
            cell: new_cell,
        })
    }

    /// Compute gate output (linear transformation)
    fn compute_gate(&self, weights: &Array2<F>, input: &Array1<F>, bias_offset: usize) -> Array1<F> {
        let mut output = Array1::zeros(self.hidden_size);
        
        for i in 0..self.hidden_size {
            let mut sum = self.bias[bias_offset + i];
            for j in 0..input.len() {
                sum = sum + weights[[i, j]] * input[j];
            }
            output[i] = sum;
        }
        
        output
    }

    /// Initialize zero state
    pub fn init_state(&self) -> LSTMState<F> {
        LSTMState {
            hidden: Array1::zeros(self.hidden_size),
            cell: Array1::zeros(self.hidden_size),
        }
    }
}

/// Multi-layer LSTM network
#[derive(Debug)]
pub struct LSTMNetwork<F: Float + Debug> {
    /// LSTM layers
    layers: Vec<LSTMCell<F>>,
    /// Output projection layer
    output_layer: Array2<F>,
    /// Output bias
    output_bias: Array1<F>,
    /// Dropout probability
    dropout_prob: F,
}

impl<F: Float + Debug + Clone + FromPrimitive> LSTMNetwork<F> {
    /// Create new multi-layer LSTM network
    pub fn new(
        input_size: usize,
        hidden_sizes: Vec<usize>,
        output_size: usize,
        dropout_prob: F,
    ) -> Self {
        let mut layers = Vec::new();
        
        // First layer
        if !hidden_sizes.is_empty() {
            layers.push(LSTMCell::new(input_size, hidden_sizes[0]));
            
            // Additional layers
            for i in 1..hidden_sizes.len() {
                layers.push(LSTMCell::new(hidden_sizes[i-1], hidden_sizes[i]));
            }
        }

        let final_hidden_size = hidden_sizes.last().copied().unwrap_or(input_size);
        
        // Output layer initialization
        let output_scale = F::from(2.0).unwrap() / F::from(final_hidden_size).unwrap();
        let output_std = output_scale.sqrt();
        let output_layer = LSTMCell::random_matrix(output_size, final_hidden_size, output_std);
        
        Self {
            layers,
            output_layer,
            output_bias: Array1::zeros(output_size),
            dropout_prob,
        }
    }

    /// Forward pass through the network
    pub fn forward(&self, input_sequence: &Array2<F>) -> Result<Array2<F>> {
        let (seq_len, input_size) = input_sequence.dim();
        
        if self.layers.is_empty() {
            return Err(TimeSeriesError::InvalidModel(
                "No LSTM layers defined".to_string(),
            ));
        }

        let output_size = self.output_layer.nrows();
        let mut outputs = Array2::zeros((seq_len, output_size));

        // Initialize states for all layers
        let mut states: Vec<LSTMState<F>> = self.layers.iter()
            .map(|layer| layer.init_state())
            .collect();

        // Process each time step
        for t in 0..seq_len {
            let mut layer_input = input_sequence.row(t).to_owned();

            // Forward through LSTM layers
            for (i, layer) in self.layers.iter().enumerate() {
                let new_state = layer.forward(&layer_input, &states[i])?;
                layer_input = new_state.hidden.clone();
                states[i] = new_state;
            }

            // Apply dropout (simplified - just scaling)
            if self.dropout_prob > F::zero() {
                let keep_prob = F::one() - self.dropout_prob;
                layer_input = layer_input.mapv(|x| x * keep_prob);
            }

            // Output projection
            let output = self.compute_output(&layer_input);
            for (j, &val) in output.iter().enumerate() {
                outputs[[t, j]] = val;
            }
        }

        Ok(outputs)
    }

    /// Compute final output projection
    fn compute_output(&self, hidden: &Array1<F>) -> Array1<F> {
        let mut output = self.output_bias.clone();
        
        for i in 0..self.output_layer.nrows() {
            for j in 0..self.output_layer.ncols() {
                output[i] = output[i] + self.output_layer[[i, j]] * hidden[j];
            }
        }
        
        output
    }

    /// Generate forecast for multiple steps
    pub fn forecast(&self, input_sequence: &Array2<F>, forecast_steps: usize) -> Result<Array1<F>> {
        let (seq_len, _) = input_sequence.dim();
        
        // Get the last hidden states from input sequence
        let _ = self.forward(input_sequence)?;
        
        // Initialize states for forecasting
        let mut states: Vec<LSTMState<F>> = self.layers.iter()
            .map(|layer| layer.init_state())
            .collect();

        // Re-run forward pass to get final states
        for t in 0..seq_len {
            let mut layer_input = input_sequence.row(t).to_owned();
            for (i, layer) in self.layers.iter().enumerate() {
                let new_state = layer.forward(&layer_input, &states[i])?;
                layer_input = new_state.hidden.clone();
                states[i] = new_state;
            }
        }

        let mut forecasts = Array1::zeros(forecast_steps);
        let mut last_output = input_sequence.row(seq_len - 1).to_owned();

        // Generate forecasts step by step
        for step in 0..forecast_steps {
            let mut layer_input = last_output.clone();

            // Forward through LSTM layers
            for (i, layer) in self.layers.iter().enumerate() {
                let new_state = layer.forward(&layer_input, &states[i])?;
                layer_input = new_state.hidden.clone();
                states[i] = new_state;
            }

            // Compute output
            let output = self.compute_output(&layer_input);
            forecasts[step] = output[0]; // Assuming single output for forecasting
            
            // Use forecast as input for next step (assuming univariate)
            if last_output.len() == 1 {
                last_output[0] = output[0];
            } else {
                // For multivariate, use the forecast as the first feature
                last_output[0] = output[0];
            }
        }

        Ok(forecasts)
    }
}

/// Self-Attention mechanism for Transformer
#[derive(Debug)]
pub struct MultiHeadAttention<F: Float + Debug> {
    /// Number of attention heads
    num_heads: usize,
    /// Model dimension
    model_dim: usize,
    /// Head dimension
    head_dim: usize,
    /// Query projection weights
    w_query: Array2<F>,
    /// Key projection weights
    w_key: Array2<F>,
    /// Value projection weights
    w_value: Array2<F>,
    /// Output projection weights
    w_output: Array2<F>,
}

impl<F: Float + Debug + Clone + FromPrimitive> MultiHeadAttention<F> {
    /// Create new multi-head attention layer
    pub fn new(model_dim: usize, num_heads: usize) -> Result<Self> {
        if model_dim % num_heads != 0 {
            return Err(TimeSeriesError::InvalidInput(
                "Model dimension must be divisible by number of heads".to_string(),
            ));
        }

        let head_dim = model_dim / num_heads;
        let scale = F::from(2.0).unwrap() / F::from(model_dim).unwrap();
        let std_dev = scale.sqrt();

        Ok(Self {
            num_heads,
            model_dim,
            head_dim,
            w_query: LSTMCell::random_matrix(model_dim, model_dim, std_dev),
            w_key: LSTMCell::random_matrix(model_dim, model_dim, std_dev),
            w_value: LSTMCell::random_matrix(model_dim, model_dim, std_dev),
            w_output: LSTMCell::random_matrix(model_dim, model_dim, std_dev),
        })
    }

    /// Forward pass through multi-head attention
    pub fn forward(&self, input: &Array2<F>) -> Result<Array2<F>> {
        let (seq_len, model_dim) = input.dim();
        
        if model_dim != self.model_dim {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: self.model_dim,
                actual: model_dim,
            });
        }

        // Project to query, key, value
        let queries = self.linear_transform(input, &self.w_query);
        let keys = self.linear_transform(input, &self.w_key);
        let values = self.linear_transform(input, &self.w_value);

        // Reshape for multi-head attention
        let queries_reshaped = self.reshape_for_attention(&queries, seq_len);
        let keys_reshaped = self.reshape_for_attention(&keys, seq_len);
        let values_reshaped = self.reshape_for_attention(&values, seq_len);

        // Compute scaled dot-product attention for each head
        let mut attention_outputs = Vec::new();
        for head in 0..self.num_heads {
            let q_head = self.get_head(&queries_reshaped, head, seq_len);
            let k_head = self.get_head(&keys_reshaped, head, seq_len);
            let v_head = self.get_head(&values_reshaped, head, seq_len);

            let attention_output = self.scaled_dot_product_attention(&q_head, &k_head, &v_head)?;
            attention_outputs.push(attention_output);
        }

        // Concatenate heads
        let concatenated = self.concatenate_heads(&attention_outputs, seq_len);

        // Final output projection
        let output = self.linear_transform(&concatenated, &self.w_output);

        Ok(output)
    }

    /// Linear transformation (matrix multiplication)
    fn linear_transform(&self, input: &Array2<F>, weights: &Array2<F>) -> Array2<F> {
        let (seq_len, input_dim) = input.dim();
        let output_dim = weights.nrows();
        let mut output = Array2::zeros((seq_len, output_dim));

        for i in 0..seq_len {
            for j in 0..output_dim {
                let mut sum = F::zero();
                for k in 0..input_dim {
                    sum = sum + input[[i, k]] * weights[[j, k]];
                }
                output[[i, j]] = sum;
            }
        }

        output
    }

    /// Reshape tensor for multi-head attention
    fn reshape_for_attention(&self, tensor: &Array2<F>, seq_len: usize) -> Array3<F> {
        let mut reshaped = Array3::zeros((self.num_heads, seq_len, self.head_dim));
        
        for head in 0..self.num_heads {
            for seq in 0..seq_len {
                for dim in 0..self.head_dim {
                    let original_dim = head * self.head_dim + dim;
                    reshaped[[head, seq, dim]] = tensor[[seq, original_dim]];
                }
            }
        }

        reshaped
    }

    /// Get specific attention head
    fn get_head(&self, tensor: &Array3<F>, head: usize, seq_len: usize) -> Array2<F> {
        let mut head_tensor = Array2::zeros((seq_len, self.head_dim));
        
        for seq in 0..seq_len {
            for dim in 0..self.head_dim {
                head_tensor[[seq, dim]] = tensor[[head, seq, dim]];
            }
        }

        head_tensor
    }

    /// Scaled dot-product attention
    fn scaled_dot_product_attention(
        &self,
        query: &Array2<F>,
        key: &Array2<F>,
        value: &Array2<F>,
    ) -> Result<Array2<F>> {
        let (seq_len, head_dim) = query.dim();
        let scale = F::one() / F::from(head_dim as f64).unwrap().sqrt();

        // Compute attention scores: Q * K^T
        let mut scores = Array2::zeros((seq_len, seq_len));
        for i in 0..seq_len {
            for j in 0..seq_len {
                let mut dot_product = F::zero();
                for k in 0..head_dim {
                    dot_product = dot_product + query[[i, k]] * key[[j, k]];
                }
                scores[[i, j]] = dot_product * scale;
            }
        }

        // Apply causal mask for autoregressive generation
        for i in 0..seq_len {
            for j in i + 1..seq_len {
                scores[[i, j]] = F::neg_infinity();
            }
        }

        // Apply softmax to get attention weights
        let attention_weights = self.softmax(&scores);

        // Apply attention to values
        let mut output = Array2::zeros((seq_len, head_dim));
        for i in 0..seq_len {
            for k in 0..head_dim {
                let mut weighted_sum = F::zero();
                for j in 0..seq_len {
                    weighted_sum = weighted_sum + attention_weights[[i, j]] * value[[j, k]];
                }
                output[[i, k]] = weighted_sum;
            }
        }

        Ok(output)
    }

    /// Softmax activation
    fn softmax(&self, input: &Array2<F>) -> Array2<F> {
        let (rows, cols) = input.dim();
        let mut output = Array2::zeros((rows, cols));

        for i in 0..rows {
            // Find maximum for numerical stability
            let mut max_val = F::neg_infinity();
            for j in 0..cols {
                if input[[i, j]] > max_val {
                    max_val = input[[i, j]];
                }
            }

            // Compute exponentials and sum
            let mut sum = F::zero();
            for j in 0..cols {
                let exp_val = (input[[i, j]] - max_val).exp();
                output[[i, j]] = exp_val;
                sum = sum + exp_val;
            }

            // Normalize
            for j in 0..cols {
                output[[i, j]] = output[[i, j]] / sum;
            }
        }

        output
    }

    /// Concatenate attention heads
    fn concatenate_heads(&self, heads: &[Array2<F>], seq_len: usize) -> Array2<F> {
        let mut concatenated = Array2::zeros((seq_len, self.model_dim));

        for (head_idx, head) in heads.iter().enumerate() {
            for seq in 0..seq_len {
                for dim in 0..self.head_dim {
                    let output_dim = head_idx * self.head_dim + dim;
                    concatenated[[seq, output_dim]] = head[[seq, dim]];
                }
            }
        }

        concatenated
    }
}

/// Feed-forward network for Transformer
#[derive(Debug)]
pub struct FeedForwardNetwork<F: Float + Debug> {
    /// First linear layer
    w1: Array2<F>,
    /// Second linear layer
    w2: Array2<F>,
    /// Bias terms
    bias1: Array1<F>,
    bias2: Array1<F>,
    /// Activation function
    activation: ActivationFunction,
}

impl<F: Float + Debug + Clone + FromPrimitive> FeedForwardNetwork<F> {
    /// Create new feed-forward network
    pub fn new(model_dim: usize, hidden_dim: usize, activation: ActivationFunction) -> Self {
        let scale1 = F::from(2.0).unwrap() / F::from(model_dim).unwrap();
        let std1 = scale1.sqrt();
        let scale2 = F::from(2.0).unwrap() / F::from(hidden_dim).unwrap();
        let std2 = scale2.sqrt();

        Self {
            w1: LSTMCell::random_matrix(hidden_dim, model_dim, std1),
            w2: LSTMCell::random_matrix(model_dim, hidden_dim, std2),
            bias1: Array1::zeros(hidden_dim),
            bias2: Array1::zeros(model_dim),
            activation,
        }
    }

    /// Forward pass
    pub fn forward(&self, input: &Array2<F>) -> Array2<F> {
        let (seq_len, model_dim) = input.dim();
        let hidden_dim = self.w1.nrows();

        // First linear layer
        let mut hidden = Array2::zeros((seq_len, hidden_dim));
        for i in 0..seq_len {
            for j in 0..hidden_dim {
                let mut sum = self.bias1[j];
                for k in 0..model_dim {
                    sum = sum + self.w1[[j, k]] * input[[i, k]];
                }
                hidden[[i, j]] = self.activation.apply(sum);
            }
        }

        // Second linear layer
        let mut output = Array2::zeros((seq_len, model_dim));
        for i in 0..seq_len {
            for j in 0..model_dim {
                let mut sum = self.bias2[j];
                for k in 0..hidden_dim {
                    sum = sum + self.w2[[j, k]] * hidden[[i, k]];
                }
                output[[i, j]] = sum;
            }
        }

        output
    }
}

/// Transformer block
#[derive(Debug)]
pub struct TransformerBlock<F: Float + Debug> {
    /// Multi-head attention
    attention: MultiHeadAttention<F>,
    /// Feed-forward network
    ffn: FeedForwardNetwork<F>,
    /// Layer normalization parameters
    ln1_gamma: Array1<F>,
    ln1_beta: Array1<F>,
    ln2_gamma: Array1<F>,
    ln2_beta: Array1<F>,
    /// Model dimension
    model_dim: usize,
}

impl<F: Float + Debug + Clone + FromPrimitive> TransformerBlock<F> {
    /// Create new transformer block
    pub fn new(model_dim: usize, num_heads: usize, ffn_hidden_dim: usize) -> Result<Self> {
        let attention = MultiHeadAttention::new(model_dim, num_heads)?;
        let ffn = FeedForwardNetwork::new(model_dim, ffn_hidden_dim, ActivationFunction::ReLU);

        Ok(Self {
            attention,
            ffn,
            ln1_gamma: Array1::ones(model_dim),
            ln1_beta: Array1::zeros(model_dim),
            ln2_gamma: Array1::ones(model_dim),
            ln2_beta: Array1::zeros(model_dim),
            model_dim,
        })
    }

    /// Forward pass through transformer block
    pub fn forward(&self, input: &Array2<F>) -> Result<Array2<F>> {
        // Self-attention with residual connection
        let attention_output = self.attention.forward(input)?;
        let attention_residual = self.add_tensors(input, &attention_output);
        let attention_norm = self.layer_norm(&attention_residual, &self.ln1_gamma, &self.ln1_beta);

        // Feed-forward with residual connection
        let ffn_output = self.ffn.forward(&attention_norm);
        let ffn_residual = self.add_tensors(&attention_norm, &ffn_output);
        let ffn_norm = self.layer_norm(&ffn_residual, &self.ln2_gamma, &self.ln2_beta);

        Ok(ffn_norm)
    }

    /// Add two tensors (residual connection)
    fn add_tensors(&self, a: &Array2<F>, b: &Array2<F>) -> Array2<F> {
        let mut result = a.clone();
        for ((i, j), val) in b.indexed_iter() {
            result[[i, j]] = result[[i, j]] + *val;
        }
        result
    }

    /// Layer normalization
    fn layer_norm(&self, input: &Array2<F>, gamma: &Array1<F>, beta: &Array1<F>) -> Array2<F> {
        let (seq_len, model_dim) = input.dim();
        let mut output = Array2::zeros((seq_len, model_dim));
        let eps = F::from(1e-6).unwrap();

        for i in 0..seq_len {
            // Compute mean and variance
            let mut mean = F::zero();
            for j in 0..model_dim {
                mean = mean + input[[i, j]];
            }
            mean = mean / F::from(model_dim).unwrap();

            let mut variance = F::zero();
            for j in 0..model_dim {
                let diff = input[[i, j]] - mean;
                variance = variance + diff * diff;
            }
            variance = variance / F::from(model_dim).unwrap();
            let std_dev = (variance + eps).sqrt();

            // Normalize and scale
            for j in 0..model_dim {
                let normalized = (input[[i, j]] - mean) / std_dev;
                output[[i, j]] = gamma[j] * normalized + beta[j];
            }
        }

        output
    }
}

/// Simple Transformer model for time series forecasting
#[derive(Debug)]
pub struct TransformerForecaster<F: Float + Debug> {
    /// Input embedding
    input_embedding: Array2<F>,
    /// Positional encoding
    positional_encoding: Array2<F>,
    /// Transformer blocks
    blocks: Vec<TransformerBlock<F>>,
    /// Output projection
    output_projection: Array2<F>,
    /// Model parameters
    model_dim: usize,
    max_seq_len: usize,
}

impl<F: Float + Debug + Clone + FromPrimitive> TransformerForecaster<F> {
    /// Create new transformer forecaster
    pub fn new(
        input_dim: usize,
        model_dim: usize,
        num_layers: usize,
        num_heads: usize,
        ffn_hidden_dim: usize,
        max_seq_len: usize,
        output_dim: usize,
    ) -> Result<Self> {
        let input_scale = F::from(2.0).unwrap() / F::from(input_dim).unwrap();
        let input_embedding = LSTMCell::random_matrix(model_dim, input_dim, input_scale.sqrt());

        let output_scale = F::from(2.0).unwrap() / F::from(model_dim).unwrap();
        let output_projection = LSTMCell::random_matrix(output_dim, model_dim, output_scale.sqrt());

        // Create positional encoding
        let positional_encoding = Self::create_positional_encoding(max_seq_len, model_dim);

        // Create transformer blocks
        let mut blocks = Vec::new();
        for _ in 0..num_layers {
            blocks.push(TransformerBlock::new(model_dim, num_heads, ffn_hidden_dim)?);
        }

        Ok(Self {
            input_embedding,
            positional_encoding,
            blocks,
            output_projection,
            model_dim,
            max_seq_len,
        })
    }

    /// Create sinusoidal positional encoding
    fn create_positional_encoding(max_seq_len: usize, model_dim: usize) -> Array2<F> {
        let mut pe = Array2::zeros((max_seq_len, model_dim));
        
        for pos in 0..max_seq_len {
            for i in 0..model_dim / 2 {
                let angle = F::from(pos as f64).unwrap() / 
                    F::from(10000.0_f64.powf(2.0 * i as f64 / model_dim as f64)).unwrap();
                
                pe[[pos, 2 * i]] = angle.sin();
                if 2 * i + 1 < model_dim {
                    pe[[pos, 2 * i + 1]] = angle.cos();
                }
            }
        }

        pe
    }

    /// Forward pass through transformer
    pub fn forward(&self, input: &Array2<F>) -> Result<Array2<F>> {
        let (seq_len, input_dim) = input.dim();
        
        if seq_len > self.max_seq_len {
            return Err(TimeSeriesError::InvalidInput(
                "Sequence length exceeds maximum".to_string(),
            ));
        }

        // Input embedding
        let mut embedded = Array2::zeros((seq_len, self.model_dim));
        for i in 0..seq_len {
            for j in 0..self.model_dim {
                let mut sum = F::zero();
                for k in 0..input_dim {
                    sum = sum + self.input_embedding[[j, k]] * input[[i, k]];
                }
                embedded[[i, j]] = sum;
            }
        }

        // Add positional encoding
        for i in 0..seq_len {
            for j in 0..self.model_dim {
                embedded[[i, j]] = embedded[[i, j]] + self.positional_encoding[[i, j]];
            }
        }

        // Pass through transformer blocks
        let mut x = embedded;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }

        // Output projection
        let output_dim = self.output_projection.nrows();
        let mut output = Array2::zeros((seq_len, output_dim));
        for i in 0..seq_len {
            for j in 0..output_dim {
                let mut sum = F::zero();
                for k in 0..self.model_dim {
                    sum = sum + self.output_projection[[j, k]] * x[[i, k]];
                }
                output[[i, j]] = sum;
            }
        }

        Ok(output)
    }

    /// Generate forecast using the transformer model
    pub fn forecast(&self, input_sequence: &Array2<F>, forecast_steps: usize) -> Result<Array1<F>> {
        let (seq_len, input_dim) = input_sequence.dim();
        let mut extended_sequence = input_sequence.clone();
        let mut forecasts = Array1::zeros(forecast_steps);

        for step in 0..forecast_steps {
            // Forward pass with current sequence
            let output = self.forward(&extended_sequence)?;
            
            // Get the last prediction
            let last_prediction = output[[seq_len + step - 1, 0]];
            forecasts[step] = last_prediction;

            // Extend sequence with the prediction
            if step < forecast_steps - 1 {
                let mut new_row = Array1::zeros(input_dim);
                new_row[0] = last_prediction; // Assuming univariate for simplicity
                
                // Create new extended sequence
                let new_len = extended_sequence.nrows() + 1;
                let mut new_sequence = Array2::zeros((new_len, input_dim));
                
                // Copy existing data
                for i in 0..extended_sequence.nrows() {
                    for j in 0..input_dim {
                        new_sequence[[i, j]] = extended_sequence[[i, j]];
                    }
                }
                
                // Add new prediction
                for j in 0..input_dim {
                    new_sequence[[new_len - 1, j]] = new_row[j];
                }
                
                extended_sequence = new_sequence;
            }
        }

        Ok(forecasts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_activation_functions() {
        let x = 0.5;
        
        let sigmoid = ActivationFunction::Sigmoid.apply(x);
        assert!(sigmoid > 0.0 && sigmoid < 1.0);
        
        let tanh = ActivationFunction::Tanh.apply(x);
        assert!(tanh > 0.0 && tanh < 1.0);
        
        let relu = ActivationFunction::ReLU.apply(-0.5);
        assert_abs_diff_eq!(relu, 0.0);
        
        let relu_positive = ActivationFunction::ReLU.apply(0.5);
        assert_abs_diff_eq!(relu_positive, 0.5);
    }

    #[test]
    fn test_lstm_cell() {
        let lstm = LSTMCell::<f64>::new(10, 20);
        let input = Array1::from_vec((0..10).map(|i| i as f64 * 0.1).collect());
        let initial_state = lstm.init_state();
        
        let new_state = lstm.forward(&input, &initial_state).unwrap();
        
        assert_eq!(new_state.hidden.len(), 20);
        assert_eq!(new_state.cell.len(), 20);
        
        // Check that states are updated (not zero)
        let hidden_sum: f64 = new_state.hidden.sum();
        let cell_sum: f64 = new_state.cell.sum();
        assert!(hidden_sum.abs() > 1e-10);
        assert!(cell_sum.abs() > 1e-10);
    }

    #[test]
    fn test_lstm_network() {
        let network = LSTMNetwork::<f64>::new(5, vec![10, 8], 1, 0.1);
        let input_sequence = Array2::from_shape_vec((4, 5), 
            (0..20).map(|i| i as f64 * 0.1).collect()).unwrap();
        
        let output = network.forward(&input_sequence).unwrap();
        assert_eq!(output.dim(), (4, 1));
        
        // Test forecasting
        let forecasts = network.forecast(&input_sequence, 3).unwrap();
        assert_eq!(forecasts.len(), 3);
    }

    #[test]
    fn test_multi_head_attention() {
        let attention = MultiHeadAttention::<f64>::new(64, 8).unwrap();
        let input = Array2::from_shape_vec((10, 64), 
            (0..640).map(|i| i as f64 * 0.01).collect()).unwrap();
        
        let output = attention.forward(&input).unwrap();
        assert_eq!(output.dim(), (10, 64));
    }

    #[test]
    fn test_transformer_block() {
        let block = TransformerBlock::<f64>::new(64, 8, 256).unwrap();
        let input = Array2::from_shape_vec((10, 64), 
            (0..640).map(|i| i as f64 * 0.01).collect()).unwrap();
        
        let output = block.forward(&input).unwrap();
        assert_eq!(output.dim(), (10, 64));
    }

    #[test]
    fn test_transformer_forecaster() {
        let model = TransformerForecaster::<f64>::new(
            1,  // input_dim
            64, // model_dim
            2,  // num_layers
            4,  // num_heads
            128, // ffn_hidden_dim
            50,  // max_seq_len
            1,   // output_dim
        ).unwrap();
        
        let input_sequence = Array2::from_shape_vec((10, 1), 
            (0..10).map(|i| i as f64 * 0.1).collect()).unwrap();
        
        let output = model.forward(&input_sequence).unwrap();
        assert_eq!(output.dim(), (10, 1));
        
        // Test forecasting
        let forecasts = model.forecast(&input_sequence, 5).unwrap();
        assert_eq!(forecasts.len(), 5);
    }

    #[test]
    fn test_feed_forward_network() {
        let ffn = FeedForwardNetwork::<f64>::new(64, 256, ActivationFunction::ReLU);
        let input = Array2::from_shape_vec((10, 64), 
            (0..640).map(|i| i as f64 * 0.01).collect()).unwrap();
        
        let output = ffn.forward(&input);
        assert_eq!(output.dim(), (10, 64));
    }
}