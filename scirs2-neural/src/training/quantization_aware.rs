//! Quantization-Aware Training (QAT) utilities
//!
//! This module provides utilities for quantization-aware training, which enables
//! training neural networks with quantized weights and activations. This approach
//! allows models to maintain accuracy while significantly reducing memory usage
//! and computational requirements.

use crate::error::{NeuralError, Result};
use crate::layers::{Layer, ParamLayer};
use crate::losses::Loss;
use crate::optimizers::{Optimizer, OptimizerStep};
use ndarray::prelude::*;
use ndarray::{Array, IxDyn};
use num_traits::{Float, FromPrimitive};
use std::collections::HashMap;
use std::fmt::Debug;
/// Quantization scheme for weights and activations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantizationScheme {
    /// Symmetric quantization (zero-point = 0)
    Symmetric,
    /// Asymmetric quantization (arbitrary zero-point)
    Asymmetric,
    /// Dynamic quantization (per-channel statistics)
    Dynamic,
    /// Static quantization (fixed statistics)
    Static,
}
/// Quantization bit-width configuration
pub enum BitWidth {
    /// 8-bit quantization
    Int8,
    /// 4-bit quantization
    Int4,
    /// 2-bit quantization
    Int2,
    /// 1-bit quantization (binary)
    Int1,
    /// Custom bit-width
    Custom(u8),
impl BitWidth {
    /// Get the number of bits for this configuration
    pub fn bits(&self) -> u8 {
        match self {
            BitWidth::Int8 => 8,
            BitWidth::Int4 => 4,
            BitWidth::Int2 => 2,
            BitWidth::Int1 => 1,
            BitWidth::Custom(bits) => *bits,
        }
    }
    /// Get the quantization range
    pub fn range(&self) -> (i32, i32) {
        let bits = self.bits() as i32;
        if bits == 1 {
            (0, 1)
        } else {
            let max_val = (1 << (bits - 1)) - 1;
            (-max_val - 1, max_val)
/// Quantization configuration for QAT
#[derive(Debug, Clone)]
pub struct QuantizationConfig {
    /// Quantization scheme to use
    pub scheme: QuantizationScheme,
    /// Bit-width for weights
    pub weight_bits: BitWidth,
    /// Bit-width for activations
    pub activation_bits: BitWidth,
    /// Whether to fake quantize during training
    pub fake_quantize: bool,
    /// Number of calibration batches for statistics collection
    pub calibration_batches: usize,
    /// Quantization observer update frequency
    pub observer_update_freq: usize,
    /// Whether to quantize bias terms
    pub quantize_bias: bool,
    /// Learning rate for quantization parameters
    pub quantization_lr: f64,
    /// Symmetric quantization for weights
    pub symmetric_weights: bool,
    /// Symmetric quantization for activations
    pub symmetric_activations: bool,
impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            scheme: QuantizationScheme::Symmetric,
            weight_bits: BitWidth::Int8,
            activation_bits: BitWidth::Int8,
            fake_quantize: true,
            calibration_batches: 100,
            observer_update_freq: 1,
            quantize_bias: false,
            quantization_lr: 0.01,
            symmetric_weights: true,
            symmetric_activations: false,
/// Quantization parameters for a tensor
pub struct QuantizationParams {
    /// Scale factor
    pub scale: f64,
    /// Zero point
    pub zero_point: i32,
    /// Minimum value observed
    pub min_val: f64,
    /// Maximum value observed
    pub max_val: f64,
    /// Bit-width
    pub bits: BitWidth,
    /// Whether symmetric quantization is used
    pub symmetric: bool,
impl QuantizationParams {
    /// Create new quantization parameters
    pub fn new(min_val: f64, max_val: f64, bits: BitWidth, symmetric: bool) -> Self {
        let (qmin, qmax) = bits.range();
        let (scale, zero_point) = if symmetric {
            let max_abs = min_val.abs().max(max_val.abs());
            let scale = (2.0 * max_abs) / (qmax - qmin) as f64;
            (scale, 0)
            let scale = (max_val - min_val) / (qmax - qmin) as f64;
            let zero_point = (qmin as f64 - min_val / scale).round() as i32;
            (scale, zero_point.max(qmin).min(qmax))
        };
            scale,
            zero_point,
            min_val,
            max_val,
            bits,
            symmetric,
    /// Update parameters with new observations
    pub fn update(&mut self, min_val: f64, max_val: f64) {
        self.min_val = self.min_val.min(min_val);
        self.max_val = self.max_val.max(max_val);
        let (qmin, qmax) = self.bits.range();
        let (scale, zero_point) = if self.symmetric {
            let max_abs = self.min_val.abs().max(self.max_val.abs());
            let scale = (self.max_val - self.min_val) / (qmax - qmin) as f64;
            let zero_point = (qmin as f64 - self.min_val / scale).round() as i32;
        self.scale = scale;
        self.zero_point = zero_point;
/// Quantization observer for collecting statistics
#[derive(Debug)]
pub struct QuantizationObserver {
    /// Current quantization parameters
    pub params: QuantizationParams,
    /// Number of observations
    pub num_observations: usize,
    /// Moving average factor
    pub momentum: f64,
    /// Whether the observer is enabled
    pub enabled: bool,
impl QuantizationObserver {
    /// Create a new observer
    pub fn new(bits: BitWidth, symmetric: bool, momentum: f64) -> Self {
            params: QuantizationParams::new(0.0, 0.0, bits, symmetric),
            num_observations: 0,
            momentum,
            enabled: true,
    /// Update observer with new tensor
    pub fn update<F: Float + Debug>(&mut self, tensor: &Array<F, IxDyn>) {
        if !self.enabled {
            return;
        let min_val = tensor
            .iter()
            .fold(F::infinity(), |a, &b| a.min(b))
            .to_f64()
            .unwrap();
        let max_val = tensor
            .fold(F::neg_infinity(), |a, &b| a.max(b))
        if self.num_observations == 0 {
            self.params =
                QuantizationParams::new(min_val, max_val, self.params.bits, self.params.symmetric);
            // Exponential moving average
            let current_min = self.momentum * self.params.min_val + (1.0 - self.momentum) * min_val;
            let current_max = self.momentum * self.params.max_val + (1.0 - self.momentum) * max_val;
            self.params.update(current_min, current_max);
        self.num_observations += 1;
    /// Get current quantization parameters
    pub fn get_params(&self) -> &QuantizationParams {
        &self.params
    /// Enable or disable the observer
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
/// Fake quantization function for training
pub fn fake_quantize<F: Float + Debug + FromPrimitive>(
    tensor: &Array<F, IxDyn>,
    params: &QuantizationParams,
) -> Result<Array<F, IxDyn>> {
    let (qmin, qmax) = params.bits.range();
    let scale = F::from(params.scale).unwrap();
    let zero_point = F::from(params.zero_point).unwrap();
    let mut result = tensor.clone();
    for value in result.iter_mut() {
        // Quantize: x_q = round(x / scale) + zero_point
        let quantized = (*value / scale + zero_point).round();
        let clamped = quantized
            .max(F::from(qmin).unwrap())
            .min(F::from(qmax).unwrap());
        // Dequantize: x = (x_q - zero_point) * scale
        *value = (clamped - zero_point) * scale;
    Ok(result)
/// Quantization-aware training manager
pub struct QuantizationAwareTrainer<F: Float + Debug + FromPrimitive + Send + Sync> {
    /// Configuration
    config: QuantizationConfig,
    /// Weight observers
    weight_observers: HashMap<String, QuantizationObserver>,
    /// Activation observers
    activation_observers: HashMap<String, QuantizationObserver>,
    /// Current training step
    training_step: usize,
    /// Whether in calibration mode
    calibration_mode: bool,
    /// Calibration batch count
    calibration_batches_processed: usize,
    /// Quantized model cache
    quantized_params: HashMap<String, Array<F, IxDyn>>,
impl<F: Float + Debug + FromPrimitive + Send + Sync + ndarray::ScalarOperand>
    QuantizationAwareTrainer<F>
{
    /// Create a new QAT trainer
    pub fn new(config: QuantizationConfig) -> Self {
            config,
            weight_observers: HashMap::new(),
            activation_observers: HashMap::new(),
            training_step: 0,
            calibration_mode: true,
            calibration_batches_processed: 0,
            quantized_params: HashMap::new(),
    /// Initialize observers for a model
    pub fn initialize_observers<L: Layer<F>>(&mut self, model: &L) -> Result<()> {
        let params = model.params();
        // Create weight observers
        for (i, _param) in params.iter().enumerate() {
            let observer_name = format!("weight_{}", i);
            let observer = QuantizationObserver::new(
                self.config.weight_bits,
                self.config.symmetric_weights,
                0.01, // momentum
            );
            self.weight_observers.insert(observer_name, observer);
        // Create activation observers (would be initialized during forward pass)
        // For now, create a default one
        let activation_observer = QuantizationObserver::new(
            self.config.activation_bits,
            self.config.symmetric_activations,
            0.01,
        );
        self.activation_observers
            .insert("default_activation".to_string(), activation_observer);
        Ok(())
    /// Update observers with current model parameters
    pub fn update_weight_observers<L: Layer<F>>(&mut self, model: &L) -> Result<()> {
        if !self.calibration_mode && self.training_step % self.config.observer_update_freq != 0 {
            return Ok(());
        for (i, param) in params.iter().enumerate() {
            if let Some(observer) = self.weight_observers.get_mut(&observer_name) {
                observer.update(param);
            }
    /// Update activation observers
    pub fn update_activation_observer(&mut self, name: &str, activation: &Array<F, IxDyn>) {
        if let Some(observer) = self.activation_observers.get_mut(name) {
            observer.update(activation);
    /// Apply fake quantization to model parameters
    pub fn apply_fake_quantization<L: ParamLayer<F>>(&mut self, model: &mut L) -> Result<()> {
        if !self.config.fake_quantize {
        let mut quantized_params = Vec::new();
            if let Some(observer) = self.weight_observers.get(&observer_name) {
                let quantized = fake_quantize(param, observer.get_params())?;
                quantized_params.push(quantized);
            } else {
                quantized_params.push(param.clone());
        model.set_params(&quantized_params)?;
    /// Train step with quantization-aware training
    pub fn train_step<L, O>(
        &mut self,
        model: &mut L,
        optimizer: &mut O,
        inputs: &Array<F, IxDyn>,
        targets: &Array<F, IxDyn>,
        loss_fn: &dyn Loss<F>,
    ) -> Result<F>
    where
        L: ParamLayer<F>,
        O: Optimizer<F> + OptimizerStep<F>,
    {
        // Update weight observers
        self.update_weight_observers(model)?;
        // Apply fake quantization to weights
        self.apply_fake_quantization(model)?;
        // Forward pass
        let outputs = model.forward(inputs)?;
        // Update activation observer (simplified - in practice would instrument forward pass)
        self.update_activation_observer("default_activation", &outputs);
        // Fake quantize activations if needed
        let quantized_outputs = if self.config.fake_quantize {
            if let Some(observer) = self.activation_observers.get("default_activation") {
                fake_quantize(&outputs, observer.get_params())?
                outputs
            outputs
        // Compute loss
        let loss = loss_fn.forward(&quantized_outputs, targets)?;
        let grad_output = loss_fn.backward(&quantized_outputs, targets)?;
        // Backward pass
        let _grad_input = model.backward(inputs, &grad_output)?;
        // Update parameters
        optimizer.step(model)?;
        // Update training state
        self.training_step += 1;
        // Check if calibration phase is complete
        if self.calibration_mode {
            self.calibration_batches_processed += 1;
            if self.calibration_batches_processed >= self.config.calibration_batches {
                self.calibration_mode = false;
                self.finalize_calibration();
        Ok(loss)
    /// Finalize calibration and prepare for training
    fn finalize_calibration(&mut self) {
        // Disable observers after calibration
        for observer in self.weight_observers.values_mut() {
            observer.set_enabled(false);
        for observer in self.activation_observers.values_mut() {
        println!("Calibration complete. Quantization parameters finalized.");
    /// Get quantization parameters for deployment
    pub fn get_quantization_parameters(&self) -> HashMap<String, QuantizationParams> {
        let mut params = HashMap::new();
        for (name, observer) in &self.weight_observers {
            params.insert(name.clone(), observer.get_params().clone());
        for (name, observer) in &self.activation_observers {
        params
    /// Convert model to integer quantized format
    pub fn quantize_model<L: ParamLayer<F>>(&self, model: &L) -> Result<QuantizedModel<F>> {
        let mut quantized_weights = Vec::new();
        let mut weight_params = Vec::new();
                let qparams = observer.get_params();
                let quantized = self.quantize_tensor(param, qparams)?;
                quantized_weights.push(quantized);
                weight_params.push(qparams.clone());
                return Err(NeuralError::InvalidState(format!(
                    "No observer found for weight {}",
                    i
                )));
        let activation_params: Vec<_> = self
            .activation_observers
            .values()
            .map(|obs| obs.get_params().clone())
            .collect();
        Ok(QuantizedModel {
            quantized_weights,
            weight_params,
            activation_params,
            config: self.config.clone(),
            _phantom: std::marker::PhantomData,
        })
    /// Quantize a tensor to integer values
    fn quantize_tensor(
        &self,
        tensor: &Array<F, IxDyn>,
        params: &QuantizationParams,
    ) -> Result<Array<i32, IxDyn>> {
        let (qmin, qmax) = params.bits.range();
        let scale = params.scale;
        let zero_point = params.zero_point;
        let mut result = Array::<i32, _>::zeros(tensor.raw_dim());
        for (output, &input) in result.iter_mut().zip(tensor.iter()) {
            let input_f64 = input.to_f64().unwrap();
            let quantized = (input_f64 / scale).round() as i32 + zero_point;
            *output = quantized.max(qmin).min(qmax);
        Ok(result)
    /// Dequantize a tensor from integer values
    pub fn dequantize_tensor(
        quantized: &Array<i32, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        let scale = F::from(params.scale).unwrap();
        let zero_point = F::from(params.zero_point).unwrap();
        let mut result = Array::<F, _>::zeros(quantized.raw_dim());
        for (output, &input) in result.iter_mut().zip(quantized.iter()) {
            let input_f = F::from(input).unwrap();
            *output = (input_f - zero_point) * scale;
    /// Get current training statistics
    pub fn get_statistics(&self) -> QuantizationStatistics {
        let mut weight_ranges = Vec::new();
        let mut activation_ranges = Vec::new();
        for observer in self.weight_observers.values() {
            let params = observer.get_params();
            weight_ranges.push((params.min_val, params.max_val));
        for observer in self.activation_observers.values() {
            activation_ranges.push((params.min_val, params.max_val));
        QuantizationStatistics {
            training_step: self.training_step,
            calibration_mode: self.calibration_mode,
            calibration_progress: if self.config.calibration_batches > 0 {
                self.calibration_batches_processed as f64 / self.config.calibration_batches as f64
                1.0
            },
            weight_ranges,
            activation_ranges,
/// Quantized model representation
pub struct QuantizedModel<F: Float + Debug> {
    /// Quantized weight tensors
    pub quantized_weights: Vec<Array<i32, IxDyn>>,
    /// Weight quantization parameters
    pub weight_params: Vec<QuantizationParams>,
    /// Activation quantization parameters
    pub activation_params: Vec<QuantizationParams>,
    /// Original configuration
    pub config: QuantizationConfig,
    /// Phantom data to maintain F parameter
    _phantom: std::marker::PhantomData<F>,
impl<F: Float + Debug + FromPrimitive> QuantizedModel<F> {
    /// Perform quantized inference
    pub fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // This is a simplified implementation
        // In practice, would need to implement quantized operations
        // For now, dequantize weights and perform regular computation
        let mut current_input = input.clone();
        for (i, (quantized_weight, weight_params)) in self
            .quantized_weights
            .zip(self.weight_params.iter())
            .enumerate()
        {
            // Dequantize weight
            let weight =
                QuantizationAwareTrainer::<F>::dequantize_tensor(quantized_weight, weight_params)?;
            // Simplified linear operation (would need proper layer implementation)
            if current_input.ndim() == 2 && weight.ndim() == 2 {
                current_input = current_input
                    .dot(&weight.view().into_dimensionality::<Ix2>().unwrap())
                    .into_dyn();
            // Apply activation quantization if not the last layer
            if i < self.quantized_weights.len() - 1 {
                if let Some(activation_params) = self.activation_params.get(0) {
                    current_input = fake_quantize(&current_input, activation_params)?;
                }
        Ok(current_input)
    /// Get model size in bytes
    pub fn model_size(&self) -> usize {
        let mut size = 0;
        for quantized_weight in &self.quantized_weights {
            size += quantized_weight.len() * (self.config.weight_bits.bits() as usize / 8).max(1);
        size
    /// Get compression ratio compared to FP32
    pub fn compression_ratio(&self) -> f32 {
        let fp32_bits = 32;
        let quantized_bits = self.config.weight_bits.bits() as f32;
        fp32_bits / quantized_bits
/// Training statistics for quantization
pub struct QuantizationStatistics {
    pub training_step: usize,
    pub calibration_mode: bool,
    /// Calibration progress (0.0 to 1.0)
    pub calibration_progress: f64,
    /// Weight value ranges (min, max)
    pub weight_ranges: Vec<(f64, f64)>,
    /// Activation value ranges (min, max)
    pub activation_ranges: Vec<(f64, f64)>,
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    #[test]
    fn test_bit_width() {
        assert_eq!(BitWidth::Int8.bits(), 8);
        assert_eq!(BitWidth::Int4.bits(), 4);
        assert_eq!(BitWidth::Custom(6).bits(), 6);
        let (min, max) = BitWidth::Int8.range();
        assert_eq!(min, -128);
        assert_eq!(max, 127);
    fn test_quantization_params() {
        let params = QuantizationParams::new(-1.0, 1.0, BitWidth::Int8, true);
        assert!(params.symmetric);
        assert_eq!(params.zero_point, 0);
        assert!((params.scale - (2.0 / 255.0)).abs() < 1e-6);
    fn test_quantization_observer() {
        let mut observer = QuantizationObserver::new(BitWidth::Int8, true, 0.1);
        let tensor = Array2::<f32>::from_shape_vec((2, 2), vec![-0.5, 0.0, 0.5, 1.0])
            .unwrap()
            .into_dyn();
        observer.update(&tensor);
        let params = observer.get_params();
        assert!(params.min_val <= -0.5);
        assert!(params.max_val >= 1.0);
    fn test_fake_quantize() {
        let tensor = Array2::<f32>::from_shape_vec((2, 2), vec![-1.0, -0.5, 0.5, 1.0])
        let quantized = fake_quantize(&tensor, &params).unwrap();
        // Values should be quantized but still in floating point
        assert_eq!(quantized.shape(), tensor.shape());
        // Check that values are quantized (not exactly the original values)
        for (&original, &quantized_val) in tensor.iter().zip(quantized.iter()) {
            let diff = (original - quantized_val).abs();
            assert!(diff <= params.scale as f32); // Should be within quantization error
    fn test_quantization_config_default() {
        let config = QuantizationConfig::default();
        assert_eq!(config.scheme, QuantizationScheme::Symmetric);
        assert_eq!(config.weight_bits, BitWidth::Int8);
        assert_eq!(config.activation_bits, BitWidth::Int8);
        assert!(config.fake_quantize);
