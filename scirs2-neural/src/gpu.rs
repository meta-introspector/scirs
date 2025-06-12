//! GPU acceleration for neural network operations
//!
//! This module provides GPU-accelerated implementations of neural network primitives.
//! Currently provides CPU fallback implementations with framework for future GPU integration.

use crate::error::{Error, Result};
use ndarray::{s, Array, Array1, Array2, ArrayD};

/// Neural operations accelerator (CPU implementation with GPU framework ready)
pub struct NeuralOps {
    /// Backend identifier for future GPU support
    backend_type: String,
}

impl NeuralOps {
    /// Create new neural operations context
    pub fn new() -> Result<Self> {
        Ok(Self {
            backend_type: "CPU".to_string(),
        })
    }

    /// Create with specified backend preference (for future GPU support)
    pub fn with_backend(backend: &str) -> Result<Self> {
        // TODO: Add proper logging when log crate is available
        println!("Using {} backend for neural operations", backend);
        Ok(Self {
            backend_type: backend.to_string(),
        })
    }

    /// Optimized matrix multiplication
    pub fn matrix_multiply(&self, a: &Array2<f32>, b: &Array2<f32>) -> Result<Array2<f32>> {
        let (m, k) = a.dim();
        let (k2, n) = b.dim();

        if k != k2 {
            return Err(Error::DimensionMismatch(format!(
                "Matrix dimensions don't match for multiplication: {}x{} * {}x{}",
                m, k, k2, n
            )));
        }

        // Use ndarray's optimized BLAS implementation
        Ok(a.dot(b))
    }

    /// Batch matrix multiplication for neural network layers
    pub fn batch_matrix_multiply(&self, a: &ArrayD<f32>, b: &ArrayD<f32>) -> Result<ArrayD<f32>> {
        let a_shape = a.shape();
        let b_shape = b.shape();

        if a_shape.len() != 3 || b_shape.len() != 3 {
            return Err(Error::DimensionMismatch(
                "Batch matrix multiply requires 3D arrays (batch, rows, cols)".to_string(),
            ));
        }

        let batch_size = a_shape[0];
        let m = a_shape[1];
        let _k = a_shape[2];
        let n = b_shape[2];

        if a_shape[0] != b_shape[0] || a_shape[2] != b_shape[1] {
            return Err(Error::DimensionMismatch(format!(
                "Batch matrix dimensions don't match: {:?} * {:?}",
                a_shape, b_shape
            )));
        }

        let mut result = Array::zeros((batch_size, m, n));

        // Process each batch
        for i in 0..batch_size {
            let a_slice = a.slice(s![i, .., ..]);
            let b_slice = b.slice(s![i, .., ..]);
            let mut result_slice = result.slice_mut(s![i, .., ..]);

            // Convert to 2D for matrix multiplication
            let a_2d = a_slice
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|e| Error::ComputationError(format!("Failed to convert to 2D: {}", e)))?;
            let b_2d = b_slice
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|e| Error::ComputationError(format!("Failed to convert to 2D: {}", e)))?;

            result_slice.assign(&a_2d.dot(&b_2d));
        }

        Ok(result.into_dyn())
    }

    /// ReLU activation function
    pub fn relu_forward(&self, input: &ArrayD<f32>) -> Result<ArrayD<f32>> {
        Ok(input.mapv(|x| x.max(0.0)))
    }

    /// ReLU derivative for backpropagation
    pub fn relu_backward(
        &self,
        input: &ArrayD<f32>,
        grad_output: &ArrayD<f32>,
    ) -> Result<ArrayD<f32>> {
        if input.shape() != grad_output.shape() {
            return Err(Error::DimensionMismatch(
                "Input and gradient shapes must match for ReLU backward".to_string(),
            ));
        }

        Ok(ndarray::Zip::from(input)
            .and(grad_output)
            .map_collect(|&x, &grad| if x > 0.0 { grad } else { 0.0 }))
    }

    /// Sigmoid activation function
    pub fn sigmoid_forward(&self, input: &ArrayD<f32>) -> Result<ArrayD<f32>> {
        Ok(input.mapv(|x| 1.0 / (1.0 + (-x).exp())))
    }

    /// Sigmoid derivative
    pub fn sigmoid_backward(
        &self,
        output: &ArrayD<f32>,
        grad_output: &ArrayD<f32>,
    ) -> Result<ArrayD<f32>> {
        if output.shape() != grad_output.shape() {
            return Err(Error::DimensionMismatch(
                "Output and gradient shapes must match for sigmoid backward".to_string(),
            ));
        }

        Ok(ndarray::Zip::from(output)
            .and(grad_output)
            .map_collect(|&sigmoid_out, &grad| grad * sigmoid_out * (1.0 - sigmoid_out)))
    }

    /// Batch normalization forward pass
    pub fn batch_normalize(
        &self,
        input: &ArrayD<f32>,
        mean: &Array1<f32>,
        var: &Array1<f32>,
        gamma: &Array1<f32>,
        beta: &Array1<f32>,
        epsilon: f32,
    ) -> Result<ArrayD<f32>> {
        let input_shape = input.shape();
        let channels = mean.len();

        // Check that all parameter arrays have the same length
        if var.len() != channels || gamma.len() != channels || beta.len() != channels {
            return Err(Error::DimensionMismatch(
                "All batch norm parameters must have the same length".to_string(),
            ));
        }

        // Assume channel-last format (NHWC) - last dimension is channels
        if input_shape[input_shape.len() - 1] != channels {
            return Err(Error::DimensionMismatch(
                "Channel dimension mismatch in batch normalization".to_string(),
            ));
        }

        let mut normalized = input.clone();

        // Apply normalization per channel
        for c in 0..channels {
            let channel_mean = mean[c];
            let channel_var = var[c];
            let channel_gamma = gamma[c];
            let channel_beta = beta[c];

            let std_dev = (channel_var + epsilon).sqrt();

            // Create a slice for the current channel across all other dimensions
            let mut channel_slice = normalized.slice_mut(s![.., c]);
            channel_slice
                .mapv_inplace(|x| (x - channel_mean) / std_dev * channel_gamma + channel_beta);
        }

        Ok(normalized)
    }

    /// Softmax activation function
    pub fn softmax_forward(&self, input: &ArrayD<f32>) -> Result<ArrayD<f32>> {
        let input_shape = input.shape();

        if input_shape.len() < 2 {
            return Err(Error::DimensionMismatch(
                "Softmax requires at least 2D input (batch_size, features)".to_string(),
            ));
        }

        let mut output = input.clone();
        let _last_axis = input_shape.len() - 1;

        // Apply softmax along the last axis (features)
        for mut row in output.axis_iter_mut(ndarray::Axis(0)) {
            // Find max for numerical stability
            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            // Subtract max and compute exp
            row.mapv_inplace(|x| (x - max_val).exp());

            // Compute sum and normalize
            let sum: f32 = row.sum();
            row.mapv_inplace(|x| x / sum);
        }

        Ok(output)
    }

    /// Convolution forward pass (simplified 2D implementation)
    pub fn conv2d_forward(
        &self,
        input: &ArrayD<f32>,
        kernel: &ArrayD<f32>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<ArrayD<f32>> {
        let input_shape = input.shape();
        let kernel_shape = kernel.shape();

        // Check input format: (batch, channels, height, width)
        if input_shape.len() != 4 || kernel_shape.len() != 4 {
            return Err(Error::DimensionMismatch(
                "Conv2D requires 4D input and kernel (batch, channels, height, width)".to_string(),
            ));
        }

        let (batch_size, in_channels, in_height, in_width) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );
        let (out_channels, kernel_in_channels, kernel_height, kernel_width) = (
            kernel_shape[0],
            kernel_shape[1],
            kernel_shape[2],
            kernel_shape[3],
        );

        if in_channels != kernel_in_channels {
            return Err(Error::DimensionMismatch(
                "Input and kernel channel dimensions must match".to_string(),
            ));
        }

        // Calculate output dimensions
        let out_height = (in_height + 2 * padding.0 - kernel_height) / stride.0 + 1;
        let out_width = (in_width + 2 * padding.1 - kernel_width) / stride.1 + 1;

        let mut output = Array::zeros((batch_size, out_channels, out_height, out_width));

        // Simplified convolution (for demonstration - real implementation would be optimized)
        for b in 0..batch_size {
            for out_c in 0..out_channels {
                for out_h in 0..out_height {
                    for out_w in 0..out_width {
                        let mut sum = 0.0;

                        for in_c in 0..in_channels {
                            for k_h in 0..kernel_height {
                                for k_w in 0..kernel_width {
                                    let in_h = out_h * stride.0 + k_h;
                                    let in_w = out_w * stride.1 + k_w;

                                    // Apply padding
                                    if in_h >= padding.0
                                        && in_w >= padding.1
                                        && in_h < in_height + padding.0
                                        && in_w < in_width + padding.1
                                    {
                                        let actual_h = in_h - padding.0;
                                        let actual_w = in_w - padding.1;

                                        if actual_h < in_height && actual_w < in_width {
                                            sum += input[[b, in_c, actual_h, actual_w]]
                                                * kernel[[out_c, in_c, k_h, k_w]];
                                        }
                                    }
                                }
                            }
                        }

                        output[[b, out_c, out_h, out_w]] = sum;
                    }
                }
            }
        }

        Ok(output.into_dyn())
    }

    /// Get backend information
    pub fn backend_info(&self) -> String {
        format!("Neural operations running on: {}", self.backend_type)
    }
}

impl Default for NeuralOps {
    fn default() -> Self {
        Self::new().expect("Failed to create default NeuralOps")
    }
}

/// Helper function to create neural operations with automatic backend detection
pub fn create_neural_ops() -> Result<NeuralOps> {
    // For now, always use CPU. Future versions will detect GPU availability
    NeuralOps::new()
}

/// Helper function to create neural operations with preferred backend
pub fn create_neural_ops_with_backend(backend: &str) -> Result<NeuralOps> {
    NeuralOps::with_backend(backend)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_matrix_multiply() {
        let ops = create_neural_ops().unwrap();

        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];

        let result = ops.matrix_multiply(&a, &b).unwrap();
        let expected = array![[19.0, 22.0], [43.0, 50.0]];

        assert_eq!(result, expected);
    }

    #[test]
    fn test_relu_forward() {
        let ops = create_neural_ops().unwrap();

        let input = array![[-1.0, 0.0, 1.0, 2.0]].into_dyn();
        let result = ops.relu_forward(&input).unwrap();
        let expected = array![[0.0, 0.0, 1.0, 2.0]].into_dyn();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_sigmoid_forward() {
        let ops = create_neural_ops().unwrap();

        let input = array![[0.0, 1.0, -1.0]].into_dyn();
        let result = ops.sigmoid_forward(&input).unwrap();

        // Check that outputs are in valid sigmoid range (0, 1)
        for &val in result.iter() {
            assert!(val > 0.0 && val < 1.0);
        }

        // Check that sigmoid(0) ≈ 0.5
        assert!((result[[0, 0]] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_batch_normalize() {
        let ops = create_neural_ops().unwrap();

        let input = array![[1.0, 2.0], [3.0, 4.0]].into_dyn();
        let mean = array![2.0, 3.0];
        let var = array![1.0, 1.0];
        let gamma = array![1.0, 1.0];
        let beta = array![0.0, 0.0];

        let result = ops
            .batch_normalize(&input, &mean, &var, &gamma, &beta, 1e-5)
            .unwrap();

        // Result should be normalized
        assert!(result.shape() == input.shape());
    }

    #[test]
    fn test_softmax_forward() {
        let ops = create_neural_ops().unwrap();

        let input = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn();
        let result = ops.softmax_forward(&input).unwrap();

        // Check that each row sums to 1
        for row in result.axis_iter(ndarray::Axis(0)) {
            let sum: f32 = row.sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }

        // Check that all values are positive
        for &val in result.iter() {
            assert!(val > 0.0);
        }
    }
}
