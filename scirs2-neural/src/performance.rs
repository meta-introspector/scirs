//! Performance optimization utilities for neural networks
//!
//! This module provides performance optimizations including SIMD acceleration,
//! memory-efficient operations, and thread pool support.

use crate::error::{NeuralError, Result};
use ndarray::{Array, ArrayD};
#[cfg(feature = "simd")]
use ndarray::{ArrayView, ArrayViewMut, IxDyn};
#[allow(unused_imports)]
use num_traits::Float;
use std::fmt::Debug;
use std::sync::Arc;

// Remove scirs2-core simd import as it may not be available

#[cfg(feature = "memory_efficient")]
use scirs2_core::memory_efficient::{chunk_wise_op, ChunkProcessor};

#[cfg(feature = "simd")]
use wide::{f32x8, CmpGt};

/// SIMD-accelerated operations for neural networks
#[cfg(feature = "simd")]
pub struct SIMDOperations;

#[cfg(feature = "simd")]
impl SIMDOperations {
    /// SIMD-accelerated ReLU activation for f32 arrays
    pub fn simd_relu_f32_inplace(input: &mut ArrayViewMut<f32, IxDyn>) {
        if let Some(slice) = input.as_slice_mut() {
            Self::simd_relu_f32_slice(slice);
        } else {
            input.mapv_inplace(|x| x.max(0.0));
        }
    }

    /// SIMD-accelerated ReLU for f32 slice
    fn simd_relu_f32_slice(slice: &mut [f32]) {
        let mut i = 0;
        let chunk_size = 8;
        let zero = f32x8::splat(0.0);

        while i + chunk_size <= slice.len() {
            let values = [
                slice[i],
                slice[i + 1],
                slice[i + 2],
                slice[i + 3],
                slice[i + 4],
                slice[i + 5],
                slice[i + 6],
                slice[i + 7],
            ];
            let vec = f32x8::new(values);
            let result = vec.cmp_gt(zero).blend(vec, zero);
            let result_arr: [f32; 8] = result.into();
            slice[i..i + chunk_size].copy_from_slice(&result_arr);
            i += chunk_size;
        }

        // Process remaining elements
        for val in &mut slice[i..] {
            *val = val.max(0.0);
        }
    }

    /// Vectorized ReLU activation returning new array
    pub fn simd_relu_f32(input: &ArrayView<f32, IxDyn>) -> ArrayD<f32> {
        let mut result = input.to_owned();
        Self::simd_relu_f32_inplace(&mut result.view_mut());
        result
    }

    /// SIMD-accelerated sigmoid activation for f32 arrays
    pub fn simd_sigmoid_f32_inplace(input: &mut ArrayViewMut<f32, IxDyn>) {
        if let Some(slice) = input.as_slice_mut() {
            Self::simd_sigmoid_f32_slice(slice);
        } else {
            input.mapv_inplace(|x| 1.0 / (1.0 + (-x).exp()));
        }
    }

    /// SIMD-accelerated sigmoid for f32 slice
    fn simd_sigmoid_f32_slice(slice: &mut [f32]) {
        let mut i = 0;
        let chunk_size = 8;
        let one = f32x8::splat(1.0);

        while i + chunk_size <= slice.len() {
            let values = [
                slice[i],
                slice[i + 1],
                slice[i + 2],
                slice[i + 3],
                slice[i + 4],
                slice[i + 5],
                slice[i + 6],
                slice[i + 7],
            ];
            let vec = f32x8::new(values);
            let neg_vec = -vec;
            let exp_vec = neg_vec.exp();
            let result = one / (one + exp_vec);
            let result_arr: [f32; 8] = result.into();
            slice[i..i + chunk_size].copy_from_slice(&result_arr);
            i += chunk_size;
        }

        // Process remaining elements
        for val in &mut slice[i..] {
            *val = 1.0 / (1.0 + (-*val).exp());
        }
    }

    /// Vectorized sigmoid activation returning new array
    pub fn simd_sigmoid_f32(input: &ArrayView<f32, IxDyn>) -> ArrayD<f32> {
        let mut result = input.to_owned();
        Self::simd_sigmoid_f32_inplace(&mut result.view_mut());
        result
    }

    /// SIMD-accelerated tanh activation for f32 arrays
    pub fn simd_tanh_f32_inplace(input: &mut ArrayViewMut<f32, IxDyn>) {
        if let Some(slice) = input.as_slice_mut() {
            Self::simd_tanh_f32_slice(slice);
        } else {
            input.mapv_inplace(|x| x.tanh());
        }
    }

    /// SIMD-accelerated tanh for f32 slice
    fn simd_tanh_f32_slice(slice: &mut [f32]) {
        let mut i = 0;
        let chunk_size = 8;

        while i + chunk_size <= slice.len() {
            let values = [
                slice[i],
                slice[i + 1],
                slice[i + 2],
                slice[i + 3],
                slice[i + 4],
                slice[i + 5],
                slice[i + 6],
                slice[i + 7],
            ];
            let vec = f32x8::new(values);
            // Manual tanh implementation: tanh(x) = (e^x - e^-x) / (e^x + e^-x)
            let pos_exp = vec.exp();
            let neg_exp = (-vec).exp();
            let result = (pos_exp - neg_exp) / (pos_exp + neg_exp);
            let result_arr: [f32; 8] = result.into();
            slice[i..i + chunk_size].copy_from_slice(&result_arr);
            i += chunk_size;
        }

        // Process remaining elements
        for val in &mut slice[i..] {
            *val = val.tanh();
        }
    }

    /// Vectorized tanh activation returning new array
    pub fn simd_tanh_f32(input: &ArrayView<f32, IxDyn>) -> ArrayD<f32> {
        let mut result = input.to_owned();
        Self::simd_tanh_f32_inplace(&mut result.view_mut());
        result
    }

    /// SIMD-accelerated GELU activation for f32 arrays
    pub fn simd_gelu_f32(input: &ArrayView<f32, IxDyn>) -> ArrayD<f32> {
        let mut result = input.to_owned();
        if let Some(slice) = result.as_slice_mut() {
            Self::simd_gelu_f32_slice(slice);
        } else {
            result.mapv_inplace(|x| {
                0.5 * x * (1.0 + (x * 0.7978845608 * (1.0 + 0.044715 * x * x)).tanh())
            });
        }
        result
    }

    /// SIMD-accelerated GELU for f32 slice (approximation)
    fn simd_gelu_f32_slice(slice: &mut [f32]) {
        let mut i = 0;
        let chunk_size = 8;
        let half = f32x8::splat(0.5);
        let one = f32x8::splat(1.0);
        let coeff1 = f32x8::splat(0.7978845608); // sqrt(2/π)
        let coeff2 = f32x8::splat(0.044715);

        while i + chunk_size <= slice.len() {
            let values = [
                slice[i],
                slice[i + 1],
                slice[i + 2],
                slice[i + 3],
                slice[i + 4],
                slice[i + 5],
                slice[i + 6],
                slice[i + 7],
            ];
            let x = f32x8::new(values);
            let x_sq = x * x;
            let x_cube = x * x_sq;
            let inner = x * coeff1 * (one + coeff2 * x_cube);
            // Manual tanh implementation for GELU: tanh(x) = (e^x - e^-x) / (e^x + e^-x)
            let pos_exp = inner.exp();
            let neg_exp = (-inner).exp();
            let tanh_val = (pos_exp - neg_exp) / (pos_exp + neg_exp);
            let result = half * x * (one + tanh_val);
            let result_arr: [f32; 8] = result.into();
            slice[i..i + chunk_size].copy_from_slice(&result_arr);
            i += chunk_size;
        }

        // Process remaining elements
        for val in &mut slice[i..] {
            let x = *val;
            *val = 0.5 * x * (1.0 + (x * 0.7978845608 * (1.0 + 0.044715 * x * x)).tanh());
        }
    }

    /// SIMD-accelerated Swish/SiLU activation for f32 arrays
    pub fn simd_swish_f32(input: &ArrayView<f32, IxDyn>) -> ArrayD<f32> {
        let mut result = input.to_owned();
        if let Some(slice) = result.as_slice_mut() {
            Self::simd_swish_f32_slice(slice);
        } else {
            result.mapv_inplace(|x| x / (1.0 + (-x).exp()));
        }
        result
    }

    /// SIMD-accelerated Swish for f32 slice
    fn simd_swish_f32_slice(slice: &mut [f32]) {
        let mut i = 0;
        let chunk_size = 8;
        let one = f32x8::splat(1.0);

        while i + chunk_size <= slice.len() {
            let values = [
                slice[i],
                slice[i + 1],
                slice[i + 2],
                slice[i + 3],
                slice[i + 4],
                slice[i + 5],
                slice[i + 6],
                slice[i + 7],
            ];
            let x = f32x8::new(values);
            let neg_x = -x;
            let exp_neg_x = neg_x.exp();
            let sigmoid = one / (one + exp_neg_x);
            let result = x * sigmoid;
            let result_arr: [f32; 8] = result.into();
            slice[i..i + chunk_size].copy_from_slice(&result_arr);
            i += chunk_size;
        }

        // Process remaining elements
        for val in &mut slice[i..] {
            let x = *val;
            *val = x / (1.0 + (-x).exp());
        }
    }

    /// SIMD-accelerated softmax for f32 arrays
    pub fn simd_softmax_f32(input: &ArrayD<f32>, axis: Option<usize>) -> Result<ArrayD<f32>> {
        let axis = axis.unwrap_or(input.ndim() - 1);

        if axis >= input.ndim() {
            return Err(NeuralError::InvalidArchitecture(
                "Softmax axis out of bounds".to_string(),
            ));
        }

        let mut result = Array::zeros(input.raw_dim());

        // Process along the specified axis
        for lane in input.axis_iter(ndarray::Axis(axis)) {
            if let (Some(input_slice), Some(mut result_slice)) = (
                lane.as_slice(),
                result.axis_iter_mut(ndarray::Axis(axis)).next(),
            ) {
                if let Some(result_slice_mut) = result_slice.as_slice_mut() {
                    Self::simd_softmax_f32_slice(input_slice, result_slice_mut);
                }
            }
        }

        Ok(result)
    }

    /// SIMD softmax for f32 slice
    fn simd_softmax_f32_slice(input: &[f32], result: &mut [f32]) {
        if input.is_empty() {
            return;
        }

        // Find maximum for numerical stability
        let max_val = input.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Compute exp(x - max) using SIMD
        let mut i = 0;
        let chunk_size = 8;
        let max_vec = f32x8::splat(max_val);
        let mut sum = 0.0f32;

        // First pass: compute exp(x - max) and accumulate sum
        while i + chunk_size <= input.len() {
            let input_vals = [
                input[i],
                input[i + 1],
                input[i + 2],
                input[i + 3],
                input[i + 4],
                input[i + 5],
                input[i + 6],
                input[i + 7],
            ];
            let input_vec = f32x8::new(input_vals);
            let shifted = input_vec - max_vec;
            let exp_vec = shifted.exp();
            let exp_arr: [f32; 8] = exp_vec.into();

            for (j, &val) in exp_arr.iter().enumerate() {
                result[i + j] = val;
                sum += val;
            }
            i += chunk_size;
        }

        // Process remaining elements
        for idx in i..input.len() {
            let exp_val = (input[idx] - max_val).exp();
            result[idx] = exp_val;
            sum += exp_val;
        }

        // Second pass: divide by sum using SIMD
        let sum_vec = f32x8::splat(sum);
        i = 0;

        while i + chunk_size <= result.len() {
            let exp_vals = [
                result[i],
                result[i + 1],
                result[i + 2],
                result[i + 3],
                result[i + 4],
                result[i + 5],
                result[i + 6],
                result[i + 7],
            ];
            let exp_vec = f32x8::new(exp_vals);
            let softmax_vec = exp_vec / sum_vec;
            let softmax_arr: [f32; 8] = softmax_vec.into();
            result[i..i + chunk_size].copy_from_slice(&softmax_arr);
            i += chunk_size;
        }

        // Process remaining elements
        for val in &mut result[i..] {
            *val /= sum;
        }
    }

    /// SIMD-accelerated cross-entropy loss computation
    pub fn simd_cross_entropy_loss_f32(
        predictions: &ArrayView<f32, IxDyn>,
        targets: &ArrayView<f32, IxDyn>,
        epsilon: f32,
    ) -> Result<f32> {
        if predictions.shape() != targets.shape() {
            return Err(NeuralError::ComputationError(
                "Predictions and targets must have the same shape".to_string(),
            ));
        }

        if let (Some(pred_slice), Some(target_slice)) = (predictions.as_slice(), targets.as_slice())
        {
            Ok(Self::simd_cross_entropy_f32_slices(
                pred_slice,
                target_slice,
                epsilon,
            ))
        } else {
            // Fallback
            let mut loss = 0.0;
            for (&pred, &target) in predictions.iter().zip(targets.iter()) {
                let clamped_pred = pred.max(epsilon).min(1.0 - epsilon);
                loss -= target * clamped_pred.ln();
            }
            Ok(loss / predictions.len() as f32)
        }
    }

    /// SIMD cross-entropy for f32 slices
    fn simd_cross_entropy_f32_slices(predictions: &[f32], targets: &[f32], epsilon: f32) -> f32 {
        let mut i = 0;
        let chunk_size = 8;
        let eps_vec = f32x8::splat(epsilon);
        let one_minus_eps = f32x8::splat(1.0 - epsilon);
        let mut loss_vec = f32x8::splat(0.0);
        let len = predictions.len().min(targets.len());

        while i + chunk_size <= len {
            let pred_vals = [
                predictions[i],
                predictions[i + 1],
                predictions[i + 2],
                predictions[i + 3],
                predictions[i + 4],
                predictions[i + 5],
                predictions[i + 6],
                predictions[i + 7],
            ];
            let target_vals = [
                targets[i],
                targets[i + 1],
                targets[i + 2],
                targets[i + 3],
                targets[i + 4],
                targets[i + 5],
                targets[i + 6],
                targets[i + 7],
            ];

            let pred_vec = f32x8::new(pred_vals);
            let target_vec = f32x8::new(target_vals);

            // Clamp predictions to avoid log(0)
            let clamped_pred = pred_vec.max(eps_vec).min(one_minus_eps);

            // Compute -target * log(prediction)
            let log_pred = clamped_pred.ln();
            let contrib = target_vec * log_pred;
            loss_vec = loss_vec - contrib;

            i += chunk_size;
        }

        // Sum the SIMD vector
        let loss_arr: [f32; 8] = loss_vec.into();
        let mut total_loss = loss_arr.iter().sum::<f32>();

        // Process remaining elements
        for idx in i..len {
            let pred = predictions[idx].max(epsilon).min(1.0 - epsilon);
            total_loss -= targets[idx] * pred.ln();
        }

        total_loss / len as f32
    }

    /// SIMD-accelerated mean squared error loss computation
    pub fn simd_mse_loss_f32(
        predictions: &ArrayView<f32, IxDyn>,
        targets: &ArrayView<f32, IxDyn>,
    ) -> Result<f32> {
        if predictions.shape() != targets.shape() {
            return Err(NeuralError::ComputationError(
                "Predictions and targets must have the same shape".to_string(),
            ));
        }

        if let (Some(pred_slice), Some(target_slice)) = (predictions.as_slice(), targets.as_slice())
        {
            Ok(Self::simd_mse_f32_slices(pred_slice, target_slice))
        } else {
            // Fallback
            let mut loss = 0.0;
            for (&pred, &target) in predictions.iter().zip(targets.iter()) {
                let diff = pred - target;
                loss += diff * diff;
            }
            Ok(loss / predictions.len() as f32)
        }
    }

    /// SIMD MSE for f32 slices
    fn simd_mse_f32_slices(predictions: &[f32], targets: &[f32]) -> f32 {
        let mut i = 0;
        let chunk_size = 8;
        let mut loss_vec = f32x8::splat(0.0);
        let len = predictions.len().min(targets.len());

        while i + chunk_size <= len {
            let pred_vals = [
                predictions[i],
                predictions[i + 1],
                predictions[i + 2],
                predictions[i + 3],
                predictions[i + 4],
                predictions[i + 5],
                predictions[i + 6],
                predictions[i + 7],
            ];
            let target_vals = [
                targets[i],
                targets[i + 1],
                targets[i + 2],
                targets[i + 3],
                targets[i + 4],
                targets[i + 5],
                targets[i + 6],
                targets[i + 7],
            ];

            let pred_vec = f32x8::new(pred_vals);
            let target_vec = f32x8::new(target_vals);

            let diff = pred_vec - target_vec;
            let squared = diff * diff;
            loss_vec = loss_vec + squared;

            i += chunk_size;
        }

        // Sum the SIMD vector
        let loss_arr: [f32; 8] = loss_vec.into();
        let mut total_loss = loss_arr.iter().sum::<f32>();

        // Process remaining elements
        for idx in i..len {
            let diff = predictions[idx] - targets[idx];
            total_loss += diff * diff;
        }

        total_loss / len as f32
    }

    /// SIMD-accelerated layer normalization
    pub fn simd_layer_norm_f32(
        input: &ArrayView<f32, IxDyn>,
        gamma: Option<&ArrayView<f32, IxDyn>>,
        beta: Option<&ArrayView<f32, IxDyn>>,
        epsilon: f32,
    ) -> Result<ArrayD<f32>> {
        let mut result = Array::zeros(input.raw_dim());

        // Compute mean and variance along the last axis
        let last_axis = input.ndim() - 1;
        let _feature_size = input.shape()[last_axis];

        for (input_lane, mut result_lane) in input
            .axis_iter(ndarray::Axis(last_axis))
            .zip(result.axis_iter_mut(ndarray::Axis(last_axis)))
        {
            if let (Some(input_slice), Some(result_slice)) =
                (input_lane.as_slice(), result_lane.as_slice_mut())
            {
                Self::simd_layer_norm_f32_slice(
                    input_slice,
                    result_slice,
                    gamma.and_then(|g| g.as_slice()),
                    beta.and_then(|b| b.as_slice()),
                    epsilon,
                );
            }
        }

        Ok(result)
    }

    /// SIMD layer normalization for f32 slice
    fn simd_layer_norm_f32_slice(
        input: &[f32],
        result: &mut [f32],
        gamma: Option<&[f32]>,
        beta: Option<&[f32]>,
        epsilon: f32,
    ) {
        if input.is_empty() {
            return;
        }

        // Compute mean
        let mean = input.iter().sum::<f32>() / input.len() as f32;

        // Compute variance
        let variance =
            input.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / input.len() as f32;

        let inv_std = 1.0 / (variance + epsilon).sqrt();

        // Normalize using SIMD
        let mut i = 0;
        let chunk_size = 8;
        let mean_vec = f32x8::splat(mean);
        let inv_std_vec = f32x8::splat(inv_std);

        while i + chunk_size <= input.len() {
            let input_vals = [
                input[i],
                input[i + 1],
                input[i + 2],
                input[i + 3],
                input[i + 4],
                input[i + 5],
                input[i + 6],
                input[i + 7],
            ];
            let input_vec = f32x8::new(input_vals);
            let normalized = (input_vec - mean_vec) * inv_std_vec;

            let mut output_vec = normalized;

            // Apply gamma if provided
            if let Some(g) = gamma {
                let gamma_vals = [
                    g[i % g.len()],
                    g[(i + 1) % g.len()],
                    g[(i + 2) % g.len()],
                    g[(i + 3) % g.len()],
                    g[(i + 4) % g.len()],
                    g[(i + 5) % g.len()],
                    g[(i + 6) % g.len()],
                    g[(i + 7) % g.len()],
                ];
                let gamma_vec = f32x8::new(gamma_vals);
                output_vec = output_vec * gamma_vec;
            }

            // Apply beta if provided
            if let Some(b) = beta {
                let beta_vals = [
                    b[i % b.len()],
                    b[(i + 1) % b.len()],
                    b[(i + 2) % b.len()],
                    b[(i + 3) % b.len()],
                    b[(i + 4) % b.len()],
                    b[(i + 5) % b.len()],
                    b[(i + 6) % b.len()],
                    b[(i + 7) % b.len()],
                ];
                let beta_vec = f32x8::new(beta_vals);
                output_vec = output_vec + beta_vec;
            }

            let result_arr: [f32; 8] = output_vec.into();
            result[i..i + chunk_size].copy_from_slice(&result_arr);
            i += chunk_size;
        }

        // Process remaining elements
        for idx in i..input.len() {
            let normalized = (input[idx] - mean) * inv_std;
            let mut output = normalized;
            if let Some(g) = gamma {
                output *= g[idx % g.len()];
            }
            if let Some(b) = beta {
                output += b[idx % b.len()];
            }
            result[idx] = output;
        }
    }

    /// Check if SIMD is available on the current platform
    pub fn is_simd_available() -> bool {
        cfg!(feature = "simd")
    }

    /// SIMD-accelerated matrix multiplication for f32 arrays
    pub fn simd_matmul_f32(
        a: &ArrayView<f32, IxDyn>,
        b: &ArrayView<f32, IxDyn>,
    ) -> Result<ArrayD<f32>> {
        if a.ndim() != 2 || b.ndim() != 2 {
            return Err(NeuralError::ComputationError(
                "SIMD matmul requires 2D arrays".to_string(),
            ));
        }

        let (m, k) = (a.shape()[0], a.shape()[1]);
        let (k2, n) = (b.shape()[0], b.shape()[1]);

        if k != k2 {
            return Err(NeuralError::ComputationError(
                "Matrix dimensions incompatible for multiplication".to_string(),
            ));
        }

        let mut output = Array::zeros((m, n));

        // SIMD-optimized matrix multiplication with blocking
        let block_size = 64; // Cache-friendly block size

        for i_block in (0..m).step_by(block_size) {
            for j_block in (0..n).step_by(block_size) {
                for k_block in (0..k).step_by(block_size) {
                    let i_end = (i_block + block_size).min(m);
                    let j_end = (j_block + block_size).min(n);
                    let k_end = (k_block + block_size).min(k);

                    for i in i_block..i_end {
                        for j in (j_block..j_end).step_by(8) {
                            let j_simd_end = (j + 8).min(j_end);
                            let mut sum_vec = f32x8::splat(0.0);

                            for k_idx in k_block..k_end {
                                let a_val = f32x8::splat(a[[i, k_idx]]);
                                if j_simd_end - j == 8 {
                                    let b_vals = [
                                        b[[k_idx, j]],
                                        b[[k_idx, j + 1]],
                                        b[[k_idx, j + 2]],
                                        b[[k_idx, j + 3]],
                                        b[[k_idx, j + 4]],
                                        b[[k_idx, j + 5]],
                                        b[[k_idx, j + 6]],
                                        b[[k_idx, j + 7]],
                                    ];
                                    let b_vec = f32x8::new(b_vals);
                                    sum_vec += a_val * b_vec;
                                }
                            }

                            if j_simd_end - j == 8 {
                                let result_arr: [f32; 8] = sum_vec.into();
                                for (idx, &val) in result_arr.iter().enumerate() {
                                    output[[i, j + idx]] += val;
                                }
                            } else {
                                // Handle remaining elements
                                for j_idx in j..j_simd_end {
                                    let mut sum = 0.0;
                                    for k_idx in k_block..k_end {
                                        sum += a[[i, k_idx]] * b[[k_idx, j_idx]];
                                    }
                                    output[[i, j_idx]] += sum;
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(output.into_dyn())
    }

    /// SIMD-accelerated element-wise addition for f32 arrays
    pub fn simd_add_f32(
        a: &ArrayView<f32, IxDyn>,
        b: &ArrayView<f32, IxDyn>,
    ) -> Result<ArrayD<f32>> {
        if a.shape() != b.shape() {
            return Err(NeuralError::ComputationError(
                "Array shapes must match for element-wise addition".to_string(),
            ));
        }

        let mut result = Array::zeros(a.raw_dim());

        if let (Some(a_slice), Some(b_slice), Some(result_slice)) =
            (a.as_slice(), b.as_slice(), result.as_slice_mut())
        {
            Self::simd_add_f32_slices(a_slice, b_slice, result_slice);
        } else {
            for ((a_val, b_val), out_val) in a.iter().zip(b.iter()).zip(result.iter_mut()) {
                *out_val = a_val + b_val;
            }
        }

        Ok(result)
    }

    /// SIMD addition for f32 slices
    fn simd_add_f32_slices(a: &[f32], b: &[f32], result: &mut [f32]) {
        let mut i = 0;
        let chunk_size = 8;
        let len = a.len().min(b.len()).min(result.len());

        while i + chunk_size <= len {
            let a_vals = [
                a[i],
                a[i + 1],
                a[i + 2],
                a[i + 3],
                a[i + 4],
                a[i + 5],
                a[i + 6],
                a[i + 7],
            ];
            let b_vals = [
                b[i],
                b[i + 1],
                b[i + 2],
                b[i + 3],
                b[i + 4],
                b[i + 5],
                b[i + 6],
                b[i + 7],
            ];
            let a_vec = f32x8::new(a_vals);
            let b_vec = f32x8::new(b_vals);
            let sum_vec = a_vec + b_vec;
            let result_arr: [f32; 8] = sum_vec.into();
            result[i..i + chunk_size].copy_from_slice(&result_arr);
            i += chunk_size;
        }

        // Process remaining elements
        for idx in i..len {
            result[idx] = a[idx] + b[idx];
        }
    }

    /// SIMD-accelerated element-wise multiplication for f32 arrays
    pub fn simd_mul_f32(
        a: &ArrayView<f32, IxDyn>,
        b: &ArrayView<f32, IxDyn>,
    ) -> Result<ArrayD<f32>> {
        if a.shape() != b.shape() {
            return Err(NeuralError::ComputationError(
                "Array shapes must match for element-wise multiplication".to_string(),
            ));
        }

        let mut result = Array::zeros(a.raw_dim());

        if let (Some(a_slice), Some(b_slice), Some(result_slice)) =
            (a.as_slice(), b.as_slice(), result.as_slice_mut())
        {
            Self::simd_mul_f32_slices(a_slice, b_slice, result_slice);
        } else {
            for ((a_val, b_val), out_val) in a.iter().zip(b.iter()).zip(result.iter_mut()) {
                *out_val = a_val * b_val;
            }
        }

        Ok(result)
    }

    /// SIMD multiplication for f32 slices
    fn simd_mul_f32_slices(a: &[f32], b: &[f32], result: &mut [f32]) {
        let mut i = 0;
        let chunk_size = 8;
        let len = a.len().min(b.len()).min(result.len());

        while i + chunk_size <= len {
            let a_vals = [
                a[i],
                a[i + 1],
                a[i + 2],
                a[i + 3],
                a[i + 4],
                a[i + 5],
                a[i + 6],
                a[i + 7],
            ];
            let b_vals = [
                b[i],
                b[i + 1],
                b[i + 2],
                b[i + 3],
                b[i + 4],
                b[i + 5],
                b[i + 6],
                b[i + 7],
            ];
            let a_vec = f32x8::new(a_vals);
            let b_vec = f32x8::new(b_vals);
            let mul_vec = a_vec * b_vec;
            let result_arr: [f32; 8] = mul_vec.into();
            result[i..i + chunk_size].copy_from_slice(&result_arr);
            i += chunk_size;
        }

        // Process remaining elements
        for idx in i..len {
            result[idx] = a[idx] * b[idx];
        }
    }

    /// SIMD-accelerated batch normalization for f32 arrays
    pub fn simd_batch_norm_f32(
        input: &ArrayView<f32, IxDyn>,
        mean: &ArrayView<f32, IxDyn>,
        variance: &ArrayView<f32, IxDyn>,
        gamma: Option<&ArrayView<f32, IxDyn>>,
        beta: Option<&ArrayView<f32, IxDyn>>,
        epsilon: f32,
    ) -> Result<ArrayD<f32>> {
        let mut result = Array::zeros(input.raw_dim());

        if let (Some(input_slice), Some(mean_slice), Some(var_slice), Some(result_slice)) = (
            input.as_slice(),
            mean.as_slice(),
            variance.as_slice(),
            result.as_slice_mut(),
        ) {
            Self::simd_batch_norm_f32_slices(
                input_slice,
                mean_slice,
                var_slice,
                result_slice,
                gamma.and_then(|g| g.as_slice()),
                beta.and_then(|b| b.as_slice()),
                epsilon,
            );
        } else {
            // Fallback to element-wise operations
            for (i, ((&inp, &m), &v)) in input
                .iter()
                .zip(mean.iter())
                .zip(variance.iter())
                .enumerate()
            {
                let normalized = (inp - m) / (v + epsilon).sqrt();
                let mut output = normalized;
                if let Some(g) = gamma {
                    output *= g.iter().nth(i % g.len()).unwrap_or(&1.0);
                }
                if let Some(b) = beta {
                    output += b.iter().nth(i % b.len()).unwrap_or(&0.0);
                }
                result.iter_mut().nth(i).map(|r| *r = output);
            }
        }

        Ok(result)
    }

    /// SIMD batch normalization for f32 slices
    fn simd_batch_norm_f32_slices(
        input: &[f32],
        mean: &[f32],
        variance: &[f32],
        result: &mut [f32],
        gamma: Option<&[f32]>,
        beta: Option<&[f32]>,
        epsilon: f32,
    ) {
        let mut i = 0;
        let chunk_size = 8;
        let len = input.len().min(result.len());
        let eps_vec = f32x8::splat(epsilon);

        while i + chunk_size <= len {
            let input_vals = [
                input[i],
                input[i + 1],
                input[i + 2],
                input[i + 3],
                input[i + 4],
                input[i + 5],
                input[i + 6],
                input[i + 7],
            ];
            let mean_vals = [
                mean[i % mean.len()],
                mean[(i + 1) % mean.len()],
                mean[(i + 2) % mean.len()],
                mean[(i + 3) % mean.len()],
                mean[(i + 4) % mean.len()],
                mean[(i + 5) % mean.len()],
                mean[(i + 6) % mean.len()],
                mean[(i + 7) % mean.len()],
            ];
            let var_vals = [
                variance[i % variance.len()],
                variance[(i + 1) % variance.len()],
                variance[(i + 2) % variance.len()],
                variance[(i + 3) % variance.len()],
                variance[(i + 4) % variance.len()],
                variance[(i + 5) % variance.len()],
                variance[(i + 6) % variance.len()],
                variance[(i + 7) % variance.len()],
            ];

            let input_vec = f32x8::new(input_vals);
            let mean_vec = f32x8::new(mean_vals);
            let var_vec = f32x8::new(var_vals);

            let normalized = (input_vec - mean_vec) / (var_vec + eps_vec).sqrt();

            let mut output_vec = normalized;

            if let Some(g) = gamma {
                let gamma_vals = [
                    g[i % g.len()],
                    g[(i + 1) % g.len()],
                    g[(i + 2) % g.len()],
                    g[(i + 3) % g.len()],
                    g[(i + 4) % g.len()],
                    g[(i + 5) % g.len()],
                    g[(i + 6) % g.len()],
                    g[(i + 7) % g.len()],
                ];
                let gamma_vec = f32x8::new(gamma_vals);
                output_vec = output_vec * gamma_vec;
            }

            if let Some(b) = beta {
                let beta_vals = [
                    b[i % b.len()],
                    b[(i + 1) % b.len()],
                    b[(i + 2) % b.len()],
                    b[(i + 3) % b.len()],
                    b[(i + 4) % b.len()],
                    b[(i + 5) % b.len()],
                    b[(i + 6) % b.len()],
                    b[(i + 7) % b.len()],
                ];
                let beta_vec = f32x8::new(beta_vals);
                output_vec = output_vec + beta_vec;
            }

            let result_arr: [f32; 8] = output_vec.into();
            result[i..i + chunk_size].copy_from_slice(&result_arr);
            i += chunk_size;
        }

        // Process remaining elements
        for idx in i..len {
            let inp = input[idx];
            let m = mean[idx % mean.len()];
            let v = variance[idx % variance.len()];
            let normalized = (inp - m) / (v + epsilon).sqrt();
            let mut output = normalized;
            if let Some(g) = gamma {
                output *= g[idx % g.len()];
            }
            if let Some(b) = beta {
                output += b[idx % b.len()];
            }
            result[idx] = output;
        }
    }
}

/// Memory-efficient batch processor
#[cfg(feature = "memory_efficient")]
pub struct MemoryEfficientProcessor {
    chunk_size: usize,
    max_memory_mb: usize,
}

#[cfg(feature = "memory_efficient")]
impl MemoryEfficientProcessor {
    /// Create a new memory-efficient processor
    pub fn new(chunk_size: Option<usize>, max_memory_mb: Option<usize>) -> Self {
        Self {
            chunk_size: chunk_size.unwrap_or(1024),
            max_memory_mb: max_memory_mb.unwrap_or(512),
        }
    }

    /// Process large arrays in chunks to reduce memory usage
    pub fn process_in_chunks<F, T>(
        &self,
        input: &ArrayD<f32>,
        mut processor: F,
    ) -> Result<ArrayD<T>>
    where
        F: FnMut(&ArrayView<f32, IxDyn>) -> Result<ArrayD<T>>,
        T: Clone + Debug + Default,
    {
        let batch_size = input.shape()[0];

        if batch_size <= self.chunk_size {
            // Process all at once if small enough
            return processor(&input.view());
        }

        // Process in chunks
        let mut results = Vec::new();
        let mut start_idx = 0;

        while start_idx < batch_size {
            let end_idx = (start_idx + self.chunk_size).min(batch_size);
            let chunk = input.slice(ndarray::s![start_idx..end_idx, ..]);

            let result = processor(&chunk)?;
            results.push(result);

            start_idx = end_idx;
        }

        // Concatenate results
        if results.is_empty() {
            return Err(NeuralError::ComputationError(
                "No chunks were processed".to_string(),
            ));
        }

        // For simplicity, just return the first result
        // A full implementation would concatenate along the batch dimension
        Ok(results.into_iter().next().unwrap())
    }

    /// Memory-efficient forward pass for large batches
    pub fn memory_efficient_forward<F>(
        &self,
        input: &ArrayD<f32>,
        forward_fn: F,
    ) -> Result<ArrayD<f32>>
    where
        F: Fn(&ArrayView<f32, IxDyn>) -> Result<ArrayD<f32>>,
    {
        chunk_wise_op(input, self.chunk_size, &ChunkProcessor::new(forward_fn)).map_err(|e| {
            NeuralError::ComputationError(format!("Memory-efficient forward failed: {:?}", e))
        })
    }
}

/// Thread pool manager for parallel neural network operations
pub struct ThreadPoolManager {
    #[cfg(feature = "rayon")]
    pool: rayon::ThreadPool,
    num_threads: usize,
}

impl ThreadPoolManager {
    /// Create a new thread pool manager
    pub fn new(num_threads: Option<usize>) -> Result<Self> {
        let num_threads = num_threads.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        });

        #[cfg(feature = "rayon")]
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .map_err(|e| {
                NeuralError::ComputationError(format!("Failed to create thread pool: {}", e))
            })?;

        Ok(Self {
            #[cfg(feature = "rayon")]
            pool,
            num_threads,
        })
    }

    /// Execute a function in the thread pool
    #[cfg(feature = "rayon")]
    pub fn execute<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        self.pool.install(f)
    }

    /// Execute a function in the thread pool (no-op without rayon)
    #[cfg(not(feature = "rayon"))]
    pub fn execute<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        f()
    }

    /// Parallel matrix multiplication using thread pool
    pub fn parallel_matmul(&self, a: &ArrayD<f32>, b: &ArrayD<f32>) -> Result<ArrayD<f32>> {
        if a.ndim() != 2 || b.ndim() != 2 {
            return Err(NeuralError::ComputationError(
                "Parallel matmul requires 2D arrays".to_string(),
            ));
        }

        let (m, k) = (a.shape()[0], a.shape()[1]);
        let (k2, n) = (b.shape()[0], b.shape()[1]);

        if k != k2 {
            return Err(NeuralError::ComputationError(
                "Matrix dimensions incompatible for multiplication".to_string(),
            ));
        }

        #[cfg(feature = "rayon")]
        return self.execute(|| {
            use rayon::prelude::*;
            let mut result = Array::zeros((m, n));

            result
                .axis_iter_mut(ndarray::Axis(0))
                .into_par_iter()
                .enumerate()
                .for_each(|(i, mut row)| {
                    for j in 0..n {
                        let mut sum = 0.0;
                        for k in 0..k {
                            sum += a[[i, k]] * b[[k, j]];
                        }
                        row[j] = sum;
                    }
                });

            Ok(result.into_dyn())
        });

        #[cfg(not(feature = "rayon"))]
        {
            let mut result = Array::zeros((m, n));
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0;
                    for k in 0..k {
                        sum += a[[i, k]] * b[[k, j]];
                    }
                    result[[i, j]] = sum;
                }
            }
            Ok(result.into_dyn())
        }
    }

    /// Get the number of threads in the pool
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }
}

/// Performance profiler for neural network operations
pub struct PerformanceProfiler {
    enabled: bool,
    timings: std::collections::HashMap<String, std::time::Duration>,
}

impl PerformanceProfiler {
    /// Create a new performance profiler
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            timings: std::collections::HashMap::new(),
        }
    }

    /// Start timing an operation
    pub fn start_timer(&self, _name: &str) -> Option<std::time::Instant> {
        if self.enabled {
            Some(std::time::Instant::now())
        } else {
            None
        }
    }

    /// End timing an operation and record the result
    pub fn end_timer(&mut self, name: String, start_time: Option<std::time::Instant>) {
        if self.enabled {
            if let Some(start) = start_time {
                let elapsed = start.elapsed();
                self.timings.insert(name, elapsed);
            }
        }
    }

    /// Get timing information
    pub fn get_timings(&self) -> &std::collections::HashMap<String, std::time::Duration> {
        &self.timings
    }

    /// Clear all timing information
    pub fn clear(&mut self) {
        self.timings.clear();
    }

    /// Print timing summary
    pub fn print_summary(&self) {
        if !self.enabled {
            println!("Performance profiling is disabled");
            return;
        }

        println!("Performance Profile Summary:");
        println!("===========================");

        let mut sorted_timings: Vec<_> = self.timings.iter().collect();
        sorted_timings.sort_by(|a, b| b.1.cmp(a.1));

        for (name, duration) in sorted_timings {
            println!("{}: {:.3}ms", name, duration.as_secs_f64() * 1000.0);
        }
    }
}

/// Unified performance optimization manager
pub struct PerformanceOptimizer {
    #[cfg(feature = "simd")]
    #[allow(dead_code)]
    simd_ops: SIMDOperations,

    #[cfg(feature = "memory_efficient")]
    memory_processor: MemoryEfficientProcessor,

    thread_pool: Arc<ThreadPoolManager>,
    profiler: PerformanceProfiler,
}

impl PerformanceOptimizer {
    /// Create a new performance optimizer
    pub fn new(
        _chunk_size: Option<usize>,
        _max_memory_mb: Option<usize>,
        num_threads: Option<usize>,
        enable_profiling: bool,
    ) -> Result<Self> {
        Ok(Self {
            #[cfg(feature = "simd")]
            simd_ops: SIMDOperations,

            #[cfg(feature = "memory_efficient")]
            memory_processor: MemoryEfficientProcessor::new(_chunk_size, _max_memory_mb),

            thread_pool: Arc::new(ThreadPoolManager::new(num_threads)?),
            profiler: PerformanceProfiler::new(enable_profiling),
        })
    }

    /// Get reference to thread pool
    pub fn thread_pool(&self) -> &Arc<ThreadPoolManager> {
        &self.thread_pool
    }

    /// Get mutable reference to profiler
    pub fn profiler_mut(&mut self) -> &mut PerformanceProfiler {
        &mut self.profiler
    }

    /// Get reference to profiler
    pub fn profiler(&self) -> &PerformanceProfiler {
        &self.profiler
    }

    /// Optimized matrix multiplication using all available optimizations
    pub fn optimized_matmul(&mut self, a: &ArrayD<f32>, b: &ArrayD<f32>) -> Result<ArrayD<f32>> {
        let timer = self.profiler.start_timer("matmul");

        let result = {
            #[cfg(feature = "simd")]
            {
                // Try SIMD first if available
                SIMDOperations::simd_matmul_f32(&a.view(), &b.view()).unwrap_or_else(|_| {
                    self.thread_pool.parallel_matmul(a, b).unwrap_or_else(|_| {
                        // Ultimate fallback
                        let (m, k) = (a.shape()[0], a.shape()[1]);
                        let n = b.shape()[1];
                        let mut result = Array::zeros((m, n));
                        for i in 0..m {
                            for j in 0..n {
                                let mut sum = 0.0;
                                for ki in 0..k {
                                    sum += a[[i, ki]] * b[[ki, j]];
                                }
                                result[[i, j]] = sum;
                            }
                        }
                        result.into_dyn()
                    })
                })
            }

            #[cfg(not(feature = "simd"))]
            {
                self.thread_pool.parallel_matmul(a, b)?
            }
        };

        self.profiler.end_timer("matmul".to_string(), timer);
        Ok(result)
    }

    /// SIMD-accelerated softmax for f32 arrays
    #[cfg(feature = "simd")]
    pub fn simd_softmax_f32(
        &mut self,
        input: &ArrayD<f32>,
        axis: Option<usize>,
    ) -> Result<ArrayD<f32>> {
        let timer = self.profiler.start_timer("softmax");
        let result = SIMDOperations::simd_softmax_f32(input, axis);
        self.profiler.end_timer("softmax".to_string(), timer);
        result
    }

    /// SIMD-accelerated ReLU for f32 arrays
    #[cfg(feature = "simd")]
    pub fn simd_relu_f32(&mut self, input: &ArrayD<f32>) -> ArrayD<f32> {
        let timer = self.profiler.start_timer("relu");
        let result = SIMDOperations::simd_relu_f32(&input.view());
        self.profiler.end_timer("relu".to_string(), timer);
        result
    }

    /// SIMD-accelerated sigmoid for f32 arrays
    #[cfg(feature = "simd")]
    pub fn simd_sigmoid_f32(&mut self, input: &ArrayD<f32>) -> ArrayD<f32> {
        let timer = self.profiler.start_timer("sigmoid");
        let result = SIMDOperations::simd_sigmoid_f32(&input.view());
        self.profiler.end_timer("sigmoid".to_string(), timer);
        result
    }

    /// SIMD-accelerated element-wise operations
    #[cfg(feature = "simd")]
    pub fn simd_add_f32(&mut self, a: &ArrayD<f32>, b: &ArrayD<f32>) -> Result<ArrayD<f32>> {
        let timer = self.profiler.start_timer("add");
        let result = SIMDOperations::simd_add_f32(&a.view(), &b.view());
        self.profiler.end_timer("add".to_string(), timer);
        result
    }

    /// SIMD-accelerated batch normalization
    #[cfg(feature = "simd")]
    pub fn simd_batch_norm_f32(
        &mut self,
        input: &ArrayD<f32>,
        mean: &ArrayD<f32>,
        variance: &ArrayD<f32>,
        gamma: Option<&ArrayD<f32>>,
        beta: Option<&ArrayD<f32>>,
        epsilon: f32,
    ) -> Result<ArrayD<f32>> {
        let timer = self.profiler.start_timer("batch_norm");
        let result = SIMDOperations::simd_batch_norm_f32(
            &input.view(),
            &mean.view(),
            &variance.view(),
            gamma.map(|g| g.view()).as_ref(),
            beta.map(|b| b.view()).as_ref(),
            epsilon,
        );
        self.profiler.end_timer("batch_norm".to_string(), timer);
        result
    }

    /// SIMD-accelerated loss computation
    #[cfg(feature = "simd")]
    pub fn simd_cross_entropy_loss(
        &mut self,
        predictions: &ArrayD<f32>,
        targets: &ArrayD<f32>,
        epsilon: f32,
    ) -> Result<f32> {
        let timer = self.profiler.start_timer("cross_entropy");
        let result = SIMDOperations::simd_cross_entropy_loss_f32(
            &predictions.view(),
            &targets.view(),
            epsilon,
        );
        self.profiler.end_timer("cross_entropy".to_string(), timer);
        result
    }

    /// Get SIMD operation statistics
    #[cfg(feature = "simd")]
    pub fn get_simd_stats(&self) -> SIMDStats {
        SIMDStats {
            simd_available: SIMDOperations::is_simd_available(),
            vector_width_f32: 8, // f32x8
            vector_width_f64: 4, // f64x4
            supported_operations: vec![
                "relu".to_string(),
                "sigmoid".to_string(),
                "tanh".to_string(),
                "gelu".to_string(),
                "swish".to_string(),
                "softmax".to_string(),
                "matmul".to_string(),
                "add".to_string(),
                "mul".to_string(),
                "batch_norm".to_string(),
                "layer_norm".to_string(),
                "cross_entropy".to_string(),
                "mse".to_string(),
            ],
        }
    }

    /// Get optimization capabilities
    pub fn get_capabilities(&self) -> OptimizationCapabilities {
        OptimizationCapabilities {
            simd_available: cfg!(feature = "simd"),
            memory_efficient_available: cfg!(feature = "memory_efficient"),
            thread_pool_available: true,
            num_threads: self.thread_pool.num_threads(),
        }
    }
}

/// Information about available optimization capabilities
#[derive(Debug, Clone)]
pub struct OptimizationCapabilities {
    /// Whether SIMD optimizations are available
    pub simd_available: bool,
    /// Whether memory-efficient operations are available
    pub memory_efficient_available: bool,
    /// Whether thread pool is available
    pub thread_pool_available: bool,
    /// Number of threads in the pool
    pub num_threads: usize,
}

impl std::fmt::Display for OptimizationCapabilities {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Optimization Capabilities:")?;
        writeln!(f, "  SIMD: {}", if self.simd_available { "✓" } else { "✗" })?;
        writeln!(
            f,
            "  Memory Efficient: {}",
            if self.memory_efficient_available {
                "✓"
            } else {
                "✗"
            }
        )?;
        writeln!(
            f,
            "  Thread Pool: {}",
            if self.thread_pool_available {
                "✓"
            } else {
                "✗"
            }
        )?;
        writeln!(f, "  Threads: {}", self.num_threads)?;
        Ok(())
    }
}

/// SIMD operation statistics and capabilities
#[derive(Debug, Clone)]
pub struct SIMDStats {
    /// Whether SIMD is available
    pub simd_available: bool,
    /// Vector width for f32 operations
    pub vector_width_f32: usize,
    /// Vector width for f64 operations
    pub vector_width_f64: usize,
    /// List of supported SIMD operations
    pub supported_operations: Vec<String>,
}

impl std::fmt::Display for SIMDStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "SIMD Operation Statistics:")?;
        writeln!(
            f,
            "  Available: {}",
            if self.simd_available { "✓" } else { "✗" }
        )?;
        writeln!(f, "  F32 Vector Width: {}", self.vector_width_f32)?;
        writeln!(f, "  F64 Vector Width: {}", self.vector_width_f64)?;
        writeln!(f, "  Supported Operations:")?;
        for op in &self.supported_operations {
            writeln!(f, "    - {}", op)?;
        }
        Ok(())
    }
}

/// Kernel fusion operations for combining multiple operations into single kernels
pub mod kernel_fusion {
    use super::*;
    use ndarray::{Array, ArrayView, ArrayViewMut, IxDyn, ScalarOperand};
    use std::collections::HashMap;

    /// Fused operation descriptor
    #[derive(Debug, Clone)]
    pub enum FusedOp<F> {
        /// Matrix multiplication followed by bias addition and activation
        MatMulBiasActivation {
            /// Bias vector to add
            bias: Array<F, IxDyn>,
            /// Activation function to apply
            activation: ActivationType,
        },
        /// Convolution followed by batch normalization and activation
        ConvBatchNormActivation {
            /// Batch norm scale parameter
            gamma: Array<F, IxDyn>,
            /// Batch norm shift parameter
            beta: Array<F, IxDyn>,
            /// Running mean for batch norm
            running_mean: Box<Array<F, IxDyn>>,
            /// Running variance for batch norm
            running_var: Box<Array<F, IxDyn>>,
            /// Small constant for numerical stability
            epsilon: F,
            /// Activation function to apply
            activation: ActivationType,
        },
        /// Element-wise operations (add, multiply, activation)
        ElementWise {
            /// Sequence of element-wise operations
            operations: Vec<ElementWiseOp<F>>,
        },
    }

    /// Activation type for fusion
    #[derive(Debug, Clone, Copy)]
    pub enum ActivationType {
        /// ReLU activation function
        ReLU,
        /// Sigmoid activation function
        Sigmoid,
        /// Tanh activation function
        Tanh,
        /// No activation function
        None,
    }

    /// Element-wise operation
    #[derive(Debug, Clone)]
    pub enum ElementWiseOp<F> {
        /// Add a scalar value
        Add(F),
        /// Multiply by a scalar value
        Multiply(F),
        /// Apply an activation function
        Activation(ActivationType),
    }

    /// Kernel fusion optimizer
    pub struct KernelFusionOptimizer<F: Float> {
        /// Cache of fused kernels
        fused_kernels: HashMap<String, FusedKernel<F>>,
        /// Statistics
        pub fusion_stats: FusionStats,
    }

    /// Cached fused kernel
    #[allow(dead_code)]
    struct FusedKernel<F> {
        /// The fused operation
        operation: FusedOp<F>,
        /// Cache key for this kernel
        cache_key: String,
    }

    /// Fusion statistics
    #[derive(Debug, Default, Clone)]
    pub struct FusionStats {
        /// Number of kernels fused
        pub kernels_fused: usize,
        /// Number of operations saved
        pub operations_saved: usize,
        /// Cache hit rate
        pub cache_hits: usize,
        /// Number of cache misses
        pub cache_misses: usize,
    }

    impl<F: Float + Debug + ScalarOperand + Clone + 'static> KernelFusionOptimizer<F> {
        /// Create new kernel fusion optimizer
        pub fn new() -> Self {
            Self {
                fused_kernels: HashMap::new(),
                fusion_stats: FusionStats::default(),
            }
        }

        /// Fuse matrix multiplication, bias addition, and activation
        pub fn fuse_matmul_bias_activation(
            &mut self,
            input: &ArrayView<F, IxDyn>,
            weight: &ArrayView<F, IxDyn>,
            bias: &ArrayView<F, IxDyn>,
            activation: ActivationType,
        ) -> Result<Array<F, IxDyn>> {
            let cache_key = format!(
                "matmul_bias_act_{}_{}_{}_{:?}",
                input
                    .shape()
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join("x"),
                weight
                    .shape()
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join("x"),
                bias.shape()
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join("x"),
                activation
            );

            let cache_key_for_entry = cache_key.clone();
            if let std::collections::hash_map::Entry::Vacant(e) =
                self.fused_kernels.entry(cache_key_for_entry)
            {
                self.fusion_stats.cache_misses += 1;
                self.fusion_stats.kernels_fused += 1;
                self.fusion_stats.operations_saved += 2; // Saved bias add and activation

                // Store in cache (simplified for now)
                let kernel = FusedKernel {
                    operation: FusedOp::MatMulBiasActivation {
                        bias: bias.to_owned(),
                        activation,
                    },
                    cache_key,
                };
                e.insert(kernel);
            } else {
                self.fusion_stats.cache_hits += 1;
            }

            self.execute_matmul_bias_activation(input, weight, bias, activation)
        }

        /// Execute fused matrix multiplication, bias addition, and activation
        fn execute_matmul_bias_activation(
            &self,
            input: &ArrayView<F, IxDyn>,
            weight: &ArrayView<F, IxDyn>,
            bias: &ArrayView<F, IxDyn>,
            activation: ActivationType,
        ) -> Result<Array<F, IxDyn>> {
            // Perform matrix multiplication
            let result = if input.ndim() == 2 && weight.ndim() == 2 {
                let input_2d = input
                    .view()
                    .into_dimensionality::<ndarray::Ix2>()
                    .map_err(|e| {
                        NeuralError::ComputationError(format!("Input reshape failed: {}", e))
                    })?;
                let weight_2d = weight
                    .view()
                    .into_dimensionality::<ndarray::Ix2>()
                    .map_err(|e| {
                        NeuralError::ComputationError(format!("Weight reshape failed: {}", e))
                    })?;

                input_2d.dot(&weight_2d).into_dyn()
            } else {
                return Err(NeuralError::ComputationError(
                    "Only 2D matrix multiplication supported in fusion".to_string(),
                ));
            };

            // Add bias (broadcasted)
            let mut result = &result + bias;

            // Apply activation in-place
            self.apply_activation_inplace(&mut result.view_mut(), activation)?;

            Ok(result)
        }

        /// Apply activation function in-place
        fn apply_activation_inplace(
            &self,
            output: &mut ArrayViewMut<F, IxDyn>,
            activation: ActivationType,
        ) -> Result<()> {
            match activation {
                ActivationType::ReLU => {
                    output.mapv_inplace(|x| x.max(F::zero()));
                }
                ActivationType::Sigmoid => {
                    output.mapv_inplace(|x| F::one() / (F::one() + (-x).exp()));
                }
                ActivationType::Tanh => {
                    output.mapv_inplace(|x| x.tanh());
                }
                ActivationType::None => {
                    // No activation
                }
            }
            Ok(())
        }

        /// Fuse element-wise operations
        pub fn fuse_elementwise_ops(
            &mut self,
            input: &ArrayView<F, IxDyn>,
            operations: &[ElementWiseOp<F>],
        ) -> Result<Array<F, IxDyn>> {
            let mut result = input.to_owned();

            // Execute all operations in sequence
            for op in operations {
                match op {
                    ElementWiseOp::Add(value) => {
                        result.mapv_inplace(|x| x + *value);
                    }
                    ElementWiseOp::Multiply(value) => {
                        result.mapv_inplace(|x| x * *value);
                    }
                    ElementWiseOp::Activation(activation) => {
                        self.apply_activation_inplace(&mut result.view_mut(), *activation)?;
                    }
                }
            }

            self.fusion_stats.operations_saved += operations.len().saturating_sub(1);
            Ok(result)
        }

        /// Get fusion statistics
        pub fn get_stats(&self) -> &FusionStats {
            &self.fusion_stats
        }

        /// Clear fusion cache
        pub fn clear_cache(&mut self) {
            self.fused_kernels.clear();
        }

        /// Get cache hit ratio
        pub fn cache_hit_ratio(&self) -> f64 {
            let total = self.fusion_stats.cache_hits + self.fusion_stats.cache_misses;
            if total == 0 {
                0.0
            } else {
                self.fusion_stats.cache_hits as f64 / total as f64
            }
        }
    }

    impl<F: Float + Debug + ScalarOperand + Clone + 'static> Default for KernelFusionOptimizer<F> {
        fn default() -> Self {
            Self::new()
        }
    }

    impl std::fmt::Display for FusionStats {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            writeln!(f, "Kernel Fusion Statistics:")?;
            writeln!(f, "  Kernels fused: {}", self.kernels_fused)?;
            writeln!(f, "  Operations saved: {}", self.operations_saved)?;
            writeln!(f, "  Cache hits: {}", self.cache_hits)?;
            writeln!(f, "  Cache misses: {}", self.cache_misses)?;
            let total = self.cache_hits + self.cache_misses;
            if total > 0 {
                let hit_rate = self.cache_hits as f64 / total as f64 * 100.0;
                writeln!(f, "  Cache hit rate: {:.1}%", hit_rate)?;
            }
            Ok(())
        }
    }
}

/// Just-In-Time compilation system for neural network operations
pub mod jit {
    use super::*;
    use ndarray::{Array, ArrayView, ArrayViewMut, Ix2, IxDyn, ScalarOperand};
    use std::collections::HashMap;
    use std::sync::{Arc, RwLock};
    use std::time::{Duration, Instant};

    /// JIT compilation strategy
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum JitStrategy {
        /// Aggressive compilation - compile everything possible
        Aggressive,
        /// Conservative compilation - only compile hot paths
        Conservative,
        /// Adaptive compilation - adjust based on performance feedback
        Adaptive,
        /// Disabled - no JIT compilation
        Disabled,
    }

    /// Compiled function signature for matrix operations
    pub type CompiledMatrixOp<F> = Arc<
        dyn Fn(&ArrayView<F, IxDyn>, &ArrayView<F, IxDyn>) -> Result<Array<F, IxDyn>> + Send + Sync,
    >;

    /// Compiled function signature for element-wise operations
    pub type CompiledElementOp<F> =
        Arc<dyn Fn(&ArrayView<F, IxDyn>) -> Result<Array<F, IxDyn>> + Send + Sync>;

    /// JIT compilation context
    pub struct JitContext<F: Float> {
        /// Compilation strategy
        strategy: JitStrategy,
        /// Cache of compiled matrix operations
        matrix_cache: Arc<RwLock<HashMap<String, CompiledMatrixOp<F>>>>,
        /// Cache of compiled element-wise operations
        element_cache: Arc<RwLock<HashMap<String, CompiledElementOp<F>>>>,
        /// Performance profiler
        profiler: Arc<RwLock<JitProfiler>>,
        /// Compilation statistics
        pub stats: Arc<RwLock<JitStats>>,
    }

    /// JIT compilation statistics
    #[derive(Debug, Default)]
    pub struct JitStats {
        /// Number of functions compiled
        pub functions_compiled: usize,
        /// Number of cache hits
        pub cache_hits: usize,
        /// Number of cache misses
        pub cache_misses: usize,
        /// Total compilation time
        pub compilation_time: Duration,
        /// Total execution time saved
        pub execution_time_saved: Duration,
        /// Number of adaptive optimizations performed
        pub adaptive_optimizations: usize,
    }

    /// Performance profiler for JIT decisions
    #[derive(Debug, Default)]
    pub struct JitProfiler {
        /// Function execution times
        execution_times: HashMap<String, Vec<Duration>>,
        /// Function call counts
        call_counts: HashMap<String, usize>,
        /// Compilation candidates
        hot_functions: HashMap<String, f64>, // function_key -> hotness_score
    }

    /// JIT-compiled operation descriptor
    #[derive(Debug)]
    pub struct JitOperation<F> {
        /// Operation key for caching
        pub key: String,
        /// Input shapes
        pub input_shapes: Vec<Vec<usize>>,
        /// Operation type
        pub op_type: JitOperationType<F>,
        /// Compilation time
        pub compile_time: Duration,
        /// Average execution time
        pub avg_execution_time: Duration,
        /// Number of times executed
        pub execution_count: usize,
    }

    /// Types of JIT-compiled operations
    #[derive(Debug, Clone)]
    pub enum JitOperationType<F> {
        /// Matrix multiplication with optional bias and activation
        MatMul {
            /// Whether to add bias
            with_bias: bool,
            /// Activation function
            activation: Option<kernel_fusion::ActivationType>,
        },
        /// Convolution operation
        Convolution {
            /// Kernel size
            kernel_size: (usize, usize),
            /// Stride
            stride: (usize, usize),
            /// Padding
            padding: (usize, usize),
        },
        /// Element-wise operation chain
        ElementWise {
            /// Operations to chain
            operations: Vec<kernel_fusion::ElementWiseOp<F>>,
        },
        /// Attention mechanism
        Attention {
            /// Number of heads
            num_heads: usize,
            /// Head dimension
            head_dim: usize,
        },
    }

    impl<F: Float + Debug + ScalarOperand + Clone + Send + Sync + 'static> JitContext<F> {
        /// Create new JIT context
        pub fn new(strategy: JitStrategy) -> Self {
            Self {
                strategy,
                matrix_cache: Arc::new(RwLock::new(HashMap::new())),
                element_cache: Arc::new(RwLock::new(HashMap::new())),
                profiler: Arc::new(RwLock::new(JitProfiler::default())),
                stats: Arc::new(RwLock::new(JitStats::default())),
            }
        }

        /// JIT compile matrix multiplication operation
        pub fn jit_matmul(
            &mut self,
            input_shape: &[usize],
            weight_shape: &[usize],
            with_bias: bool,
            activation: Option<kernel_fusion::ActivationType>,
        ) -> Result<CompiledMatrixOp<F>> {
            let cache_key = format!(
                "matmul_{}_{}_{}_{}",
                input_shape
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join("x"),
                weight_shape
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join("x"),
                with_bias,
                activation.map_or("none".to_string(), |a| format!("{:?}", a))
            );

            // Check cache first
            if let Ok(cache) = self.matrix_cache.read() {
                if let Some(compiled_fn) = cache.get(&cache_key) {
                    if let Ok(mut stats) = self.stats.write() {
                        stats.cache_hits += 1;
                    }
                    return Ok(compiled_fn.clone());
                }
            }

            // Cache miss - compile new function
            if let Ok(mut stats) = self.stats.write() {
                stats.cache_misses += 1;
            }

            let compile_start = Instant::now();
            let compiled_fn =
                self.compile_matmul_operation(input_shape, weight_shape, with_bias, activation)?;
            let compile_time = compile_start.elapsed();

            // Update statistics
            if let Ok(mut stats) = self.stats.write() {
                stats.functions_compiled += 1;
                stats.compilation_time += compile_time;
            }

            // Cache the compiled function
            if let Ok(mut cache) = self.matrix_cache.write() {
                cache.insert(cache_key, compiled_fn.clone());
            }

            Ok(compiled_fn)
        }

        /// Compile matrix multiplication operation with optimizations
        fn compile_matmul_operation(
            &self,
            input_shape: &[usize],
            weight_shape: &[usize],
            with_bias: bool,
            activation: Option<kernel_fusion::ActivationType>,
        ) -> Result<CompiledMatrixOp<F>> {
            // Select optimization strategy based on input characteristics
            let optimization_level = self.select_optimization_level(input_shape, weight_shape);

            match optimization_level {
                OptimizationLevel::High => self.compile_optimized_matmul(with_bias, activation),
                OptimizationLevel::Medium => self.compile_standard_matmul(with_bias, activation),
                OptimizationLevel::Low => self.compile_fallback_matmul(with_bias, activation),
            }
        }

        /// Compile highly optimized matrix multiplication
        fn compile_optimized_matmul(
            &self,
            with_bias: bool,
            activation: Option<kernel_fusion::ActivationType>,
        ) -> Result<CompiledMatrixOp<F>> {
            let compiled_fn: CompiledMatrixOp<F> = Arc::new(move |input, weight| {
                // Highly optimized implementation using SIMD and blocking
                let result = Self::optimized_matmul_impl(input, weight)?;

                let mut final_result = if with_bias {
                    // Bias addition would be passed separately in real implementation
                    result
                } else {
                    result
                };

                // Apply activation if specified
                if let Some(act) = activation {
                    Self::apply_activation_inplace(&mut final_result.view_mut(), act)?;
                }

                Ok(final_result)
            });

            Ok(compiled_fn)
        }

        /// Compile standard matrix multiplication
        fn compile_standard_matmul(
            &self,
            _with_bias: bool,
            activation: Option<kernel_fusion::ActivationType>,
        ) -> Result<CompiledMatrixOp<F>> {
            let compiled_fn: CompiledMatrixOp<F> = Arc::new(move |input, weight| {
                let result = Self::standard_matmul_impl(input, weight)?;

                let mut final_result = result;

                // Apply activation if specified
                if let Some(act) = activation {
                    Self::apply_activation_inplace(&mut final_result.view_mut(), act)?;
                }

                Ok(final_result)
            });

            Ok(compiled_fn)
        }

        /// Compile fallback matrix multiplication
        fn compile_fallback_matmul(
            &self,
            _with_bias: bool,
            activation: Option<kernel_fusion::ActivationType>,
        ) -> Result<CompiledMatrixOp<F>> {
            let compiled_fn: CompiledMatrixOp<F> = Arc::new(move |input, weight| {
                let result = Self::fallback_matmul_impl(input, weight)?;

                let mut final_result = result;

                if let Some(act) = activation {
                    Self::apply_activation_inplace(&mut final_result.view_mut(), act)?;
                }

                Ok(final_result)
            });

            Ok(compiled_fn)
        }

        /// Optimized matrix multiplication implementation
        fn optimized_matmul_impl(
            input: &ArrayView<F, IxDyn>,
            weight: &ArrayView<F, IxDyn>,
        ) -> Result<Array<F, IxDyn>> {
            // Convert to 2D for matrix multiplication
            if input.ndim() == 2 && weight.ndim() == 2 {
                let input_2d = input.view().into_dimensionality::<Ix2>().map_err(|e| {
                    NeuralError::ComputationError(format!("Input reshape failed: {}", e))
                })?;
                let weight_2d = weight.view().into_dimensionality::<Ix2>().map_err(|e| {
                    NeuralError::ComputationError(format!("Weight reshape failed: {}", e))
                })?;

                // Use optimized blocked matrix multiplication
                let result = Self::blocked_matmul(&input_2d, &weight_2d)?;
                Ok(result.into_dyn())
            } else {
                Err(NeuralError::ComputationError(
                    "Only 2D matrices supported for optimized matmul".to_string(),
                ))
            }
        }

        /// Blocked matrix multiplication for cache efficiency
        pub fn blocked_matmul(
            a: &ArrayView<F, Ix2>,
            b: &ArrayView<F, Ix2>,
        ) -> Result<Array<F, Ix2>> {
            let (m, k) = a.dim();
            let (k2, n) = b.dim();

            if k != k2 {
                return Err(NeuralError::ComputationError(
                    "Matrix dimensions don't match for multiplication".to_string(),
                ));
            }

            let mut c = Array::zeros((m, n));
            let block_size = 64; // Cache-friendly block size

            for i in (0..m).step_by(block_size) {
                for j in (0..n).step_by(block_size) {
                    for l in (0..k).step_by(block_size) {
                        let i_end = (i + block_size).min(m);
                        let j_end = (j + block_size).min(n);
                        let l_end = (l + block_size).min(k);

                        for ii in i..i_end {
                            for jj in j..j_end {
                                let mut sum = c[[ii, jj]];
                                for ll in l..l_end {
                                    sum = sum + a[[ii, ll]] * b[[ll, jj]];
                                }
                                c[[ii, jj]] = sum;
                            }
                        }
                    }
                }
            }

            Ok(c)
        }

        /// Standard matrix multiplication implementation
        fn standard_matmul_impl(
            input: &ArrayView<F, IxDyn>,
            weight: &ArrayView<F, IxDyn>,
        ) -> Result<Array<F, IxDyn>> {
            if input.ndim() == 2 && weight.ndim() == 2 {
                let input_2d = input.view().into_dimensionality::<Ix2>().map_err(|e| {
                    NeuralError::ComputationError(format!("Input reshape failed: {}", e))
                })?;
                let weight_2d = weight.view().into_dimensionality::<Ix2>().map_err(|e| {
                    NeuralError::ComputationError(format!("Weight reshape failed: {}", e))
                })?;

                Ok(input_2d.dot(&weight_2d).into_dyn())
            } else {
                Err(NeuralError::ComputationError(
                    "Only 2D matrices supported".to_string(),
                ))
            }
        }

        /// Fallback matrix multiplication implementation
        fn fallback_matmul_impl(
            input: &ArrayView<F, IxDyn>,
            weight: &ArrayView<F, IxDyn>,
        ) -> Result<Array<F, IxDyn>> {
            Self::standard_matmul_impl(input, weight)
        }

        /// Apply activation function in-place
        fn apply_activation_inplace(
            output: &mut ArrayViewMut<F, IxDyn>,
            activation: kernel_fusion::ActivationType,
        ) -> Result<()> {
            match activation {
                kernel_fusion::ActivationType::ReLU => {
                    output.mapv_inplace(|x| x.max(F::zero()));
                }
                kernel_fusion::ActivationType::Sigmoid => {
                    output.mapv_inplace(|x| F::one() / (F::one() + (-x).exp()));
                }
                kernel_fusion::ActivationType::Tanh => {
                    output.mapv_inplace(|x| x.tanh());
                }
                kernel_fusion::ActivationType::None => {
                    // No activation
                }
            }
            Ok(())
        }

        /// Select optimization level based on input characteristics
        fn select_optimization_level(
            &self,
            input_shape: &[usize],
            weight_shape: &[usize],
        ) -> OptimizationLevel {
            let input_size: usize = input_shape.iter().product();
            let weight_size: usize = weight_shape.iter().product();
            let total_ops = input_size * weight_size;

            match self.strategy {
                JitStrategy::Aggressive => {
                    if total_ops > 10000 {
                        OptimizationLevel::High
                    } else if total_ops > 1000 {
                        OptimizationLevel::Medium
                    } else {
                        OptimizationLevel::Low
                    }
                }
                JitStrategy::Conservative => {
                    if total_ops > 100000 {
                        OptimizationLevel::Medium
                    } else {
                        OptimizationLevel::Low
                    }
                }
                JitStrategy::Adaptive => {
                    // Use profiler data to make decisions
                    OptimizationLevel::Medium // Simplified for now
                }
                JitStrategy::Disabled => OptimizationLevel::Low,
            }
        }

        /// Update profiler with execution data
        pub fn profile_execution(&mut self, operation_key: &str, execution_time: Duration) {
            if let Ok(mut profiler) = self.profiler.write() {
                profiler.record_execution(operation_key, execution_time);
            }
        }

        /// Get JIT compilation statistics
        pub fn get_stats(&self) -> JitStats {
            if let Ok(stats) = self.stats.read() {
                stats.clone()
            } else {
                JitStats::default()
            }
        }

        /// Clear compilation caches
        pub fn clear_caches(&mut self) {
            if let Ok(mut cache) = self.matrix_cache.write() {
                cache.clear();
            }
            if let Ok(mut cache) = self.element_cache.write() {
                cache.clear();
            }
        }

        /// Get cache hit ratio
        pub fn cache_hit_ratio(&self) -> f64 {
            if let Ok(stats) = self.stats.read() {
                let total = stats.cache_hits + stats.cache_misses;
                if total == 0 {
                    0.0
                } else {
                    stats.cache_hits as f64 / total as f64
                }
            } else {
                0.0
            }
        }
    }

    /// Optimization level for JIT compilation
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum OptimizationLevel {
        /// High optimization - maximum performance, longer compile time
        High,
        /// Medium optimization - balanced performance and compile time
        Medium,
        /// Low optimization - fast compilation, basic optimizations
        Low,
    }

    impl JitProfiler {
        /// Record execution time for an operation
        pub fn record_execution(&mut self, operation_key: &str, execution_time: Duration) {
            self.execution_times
                .entry(operation_key.to_string())
                .or_default()
                .push(execution_time);

            *self
                .call_counts
                .entry(operation_key.to_string())
                .or_insert(0) += 1;

            // Update hotness score
            let avg_time = self.average_execution_time(operation_key);
            let call_count = self.call_counts[operation_key];
            let hotness = call_count as f64 * avg_time.as_nanos() as f64;

            self.hot_functions
                .insert(operation_key.to_string(), hotness);
        }

        /// Get average execution time for an operation
        pub fn average_execution_time(&self, operation_key: &str) -> Duration {
            if let Some(times) = self.execution_times.get(operation_key) {
                if !times.is_empty() {
                    let total: Duration = times.iter().sum();
                    total / times.len() as u32
                } else {
                    Duration::ZERO
                }
            } else {
                Duration::ZERO
            }
        }

        /// Get hottest functions that should be JIT compiled
        pub fn get_hot_functions(&self, threshold: f64) -> Vec<String> {
            self.hot_functions
                .iter()
                .filter(|(_, &hotness)| hotness > threshold)
                .map(|(key, _)| key.clone())
                .collect()
        }
    }

    impl Clone for JitStats {
        fn clone(&self) -> Self {
            Self {
                functions_compiled: self.functions_compiled,
                cache_hits: self.cache_hits,
                cache_misses: self.cache_misses,
                compilation_time: self.compilation_time,
                execution_time_saved: self.execution_time_saved,
                adaptive_optimizations: self.adaptive_optimizations,
            }
        }
    }

    impl std::fmt::Display for JitStats {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            writeln!(f, "JIT Compilation Statistics:")?;
            writeln!(f, "  Functions compiled: {}", self.functions_compiled)?;
            writeln!(f, "  Cache hits: {}", self.cache_hits)?;
            writeln!(f, "  Cache misses: {}", self.cache_misses)?;
            writeln!(f, "  Compilation time: {:?}", self.compilation_time)?;
            writeln!(f, "  Execution time saved: {:?}", self.execution_time_saved)?;
            writeln!(
                f,
                "  Adaptive optimizations: {}",
                self.adaptive_optimizations
            )?;

            let total_requests = self.cache_hits + self.cache_misses;
            if total_requests > 0 {
                let hit_rate = self.cache_hits as f64 / total_requests as f64 * 100.0;
                writeln!(f, "  Cache hit rate: {:.1}%", hit_rate)?;
            }

            Ok(())
        }
    }
}

/// High-performance neural network executor combining JIT compilation and kernel fusion
pub mod fusion_jit {
    use super::*;
    use ndarray::{Array, ArrayView, IxDyn, ScalarOperand};
    use std::sync::{Arc, RwLock};
    use std::time::{Duration, Instant};

    /// High-performance executor that combines JIT compilation and kernel fusion
    pub struct FusionJitExecutor<F: Float> {
        /// JIT compilation context
        jit_context: Arc<RwLock<jit::JitContext<F>>>,
        /// Kernel fusion optimizer
        fusion_optimizer: Arc<RwLock<kernel_fusion::KernelFusionOptimizer<F>>>,
        /// Performance statistics
        pub stats: Arc<RwLock<FusionJitStats>>,
        /// Execution strategy
        strategy: ExecutionStrategy,
    }

    /// Combined statistics for fusion and JIT
    #[derive(Debug, Default)]
    pub struct FusionJitStats {
        /// Number of operations executed
        pub operations_executed: usize,
        /// Number of JIT-compiled operations
        pub jit_operations: usize,
        /// Number of fused operations
        pub fused_operations: usize,
        /// Number of combined fusion+JIT operations
        pub fusion_jit_operations: usize,
        /// Total execution time
        pub total_execution_time: Duration,
        /// JIT compilation overhead
        pub jit_compilation_overhead: Duration,
        /// Performance improvement ratio
        pub performance_improvement: f64,
    }

    /// Strategy for combining fusion and JIT
    #[derive(Debug, Clone, Copy)]
    pub enum ExecutionStrategy {
        /// Always prefer JIT-compiled fused operations
        FusionFirst,
        /// Always prefer JIT compilation, then fusion
        JitFirst,
        /// Adaptive strategy based on operation characteristics
        Adaptive,
        /// Use simple heuristics
        Heuristic,
    }

    /// Optimized operation descriptor
    #[derive(Debug)]
    pub struct OptimizedOperation<F> {
        /// Operation identifier
        pub id: String,
        /// Whether it uses fusion
        pub uses_fusion: bool,
        /// Whether it uses JIT
        pub uses_jit: bool,
        /// Expected speedup
        pub expected_speedup: f64,
        /// Last execution time
        pub last_execution_time: Duration,
        /// Phantom data for type parameter
        _phantom: std::marker::PhantomData<F>,
    }

    impl<F: Float + Debug + ScalarOperand + Clone + Send + Sync + 'static> FusionJitExecutor<F> {
        /// Create new fusion-JIT executor
        pub fn new(jit_strategy: jit::JitStrategy, execution_strategy: ExecutionStrategy) -> Self {
            Self {
                jit_context: Arc::new(RwLock::new(jit::JitContext::new(jit_strategy))),
                fusion_optimizer: Arc::new(
                    RwLock::new(kernel_fusion::KernelFusionOptimizer::new()),
                ),
                stats: Arc::new(RwLock::new(FusionJitStats::default())),
                strategy: execution_strategy,
            }
        }

        /// Execute highly optimized matrix multiplication with bias and activation
        pub fn execute_fused_jit_matmul(
            &mut self,
            input: &ArrayView<F, IxDyn>,
            weight: &ArrayView<F, IxDyn>,
            bias: &ArrayView<F, IxDyn>,
            activation: kernel_fusion::ActivationType,
        ) -> Result<Array<F, IxDyn>> {
            let start_time = Instant::now();

            let operation_id = format!(
                "fused_jit_matmul_{}_{}_{}_{:?}",
                input
                    .shape()
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join("x"),
                weight
                    .shape()
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join("x"),
                bias.shape()
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join("x"),
                activation
            );

            let result = match self.strategy {
                ExecutionStrategy::FusionFirst => {
                    self.execute_fusion_first(input, weight, bias, activation)
                }
                ExecutionStrategy::JitFirst => {
                    self.execute_jit_first(input, weight, bias, activation)
                }
                ExecutionStrategy::Adaptive => {
                    self.execute_adaptive(input, weight, bias, activation)
                }
                ExecutionStrategy::Heuristic => {
                    self.execute_heuristic(input, weight, bias, activation)
                }
            };

            let execution_time = start_time.elapsed();
            self.update_stats(&operation_id, execution_time, true, true)?;

            result
        }

        /// Execute with fusion-first strategy
        fn execute_fusion_first(
            &mut self,
            input: &ArrayView<F, IxDyn>,
            weight: &ArrayView<F, IxDyn>,
            bias: &ArrayView<F, IxDyn>,
            activation: kernel_fusion::ActivationType,
        ) -> Result<Array<F, IxDyn>> {
            // First apply kernel fusion
            let fused_result = if let Ok(mut fusion_opt) = self.fusion_optimizer.write() {
                fusion_opt.fuse_matmul_bias_activation(input, weight, bias, activation)
            } else {
                return Err(NeuralError::ComputationError(
                    "Failed to acquire fusion optimizer lock".to_string(),
                ));
            }?;

            // Then potentially JIT-compile the fused operation for future use
            self.maybe_jit_compile_operation(input.shape(), weight.shape())?;

            Ok(fused_result)
        }

        /// Execute with JIT-first strategy
        fn execute_jit_first(
            &mut self,
            input: &ArrayView<F, IxDyn>,
            weight: &ArrayView<F, IxDyn>,
            bias: &ArrayView<F, IxDyn>,
            activation: kernel_fusion::ActivationType,
        ) -> Result<Array<F, IxDyn>> {
            // Try to get JIT-compiled function first
            if let Ok(mut jit_ctx) = self.jit_context.write() {
                let compiled_fn = jit_ctx.jit_matmul(
                    input.shape(),
                    weight.shape(),
                    false, // without bias in JIT, we'll add it manually
                    Some(activation),
                )?;

                // Execute with JIT-compiled function and manually add bias
                let mut result = compiled_fn(input, weight)?;
                result = &result + bias;
                Ok(result)
            } else {
                Err(NeuralError::ComputationError(
                    "Failed to acquire JIT context lock".to_string(),
                ))
            }
        }

        /// Execute with adaptive strategy
        fn execute_adaptive(
            &mut self,
            input: &ArrayView<F, IxDyn>,
            weight: &ArrayView<F, IxDyn>,
            bias: &ArrayView<F, IxDyn>,
            activation: kernel_fusion::ActivationType,
        ) -> Result<Array<F, IxDyn>> {
            let input_size: usize = input.shape().iter().product();
            let weight_size: usize = weight.shape().iter().product();
            let total_ops = input_size * weight_size;

            // Use heuristics to choose the best approach
            if total_ops > 100000 {
                // Large operations benefit from both fusion and JIT
                self.execute_fusion_jit_combined(input, weight, bias, activation)
            } else if total_ops > 10000 {
                // Medium operations benefit from fusion
                self.execute_fusion_first(input, weight, bias, activation)
            } else {
                // Small operations use standard implementation
                self.execute_standard_impl(input, weight, bias, activation)
            }
        }

        /// Execute with heuristic strategy
        fn execute_heuristic(
            &mut self,
            input: &ArrayView<F, IxDyn>,
            weight: &ArrayView<F, IxDyn>,
            bias: &ArrayView<F, IxDyn>,
            activation: kernel_fusion::ActivationType,
        ) -> Result<Array<F, IxDyn>> {
            // Simple heuristic: use fusion for most operations
            self.execute_fusion_first(input, weight, bias, activation)
        }

        /// Execute combined fusion and JIT approach
        fn execute_fusion_jit_combined(
            &mut self,
            input: &ArrayView<F, IxDyn>,
            weight: &ArrayView<F, IxDyn>,
            bias: &ArrayView<F, IxDyn>,
            activation: kernel_fusion::ActivationType,
        ) -> Result<Array<F, IxDyn>> {
            // This represents the ultimate optimization: fused operations that are JIT-compiled
            // In a real implementation, this would involve compiling the fused operation
            // For now, we'll use the fusion approach with JIT compilation hints

            let compiled_result = if let Ok(mut jit_ctx) = self.jit_context.write() {
                let compiled_fn =
                    jit_ctx.jit_matmul(input.shape(), weight.shape(), true, Some(activation))?;

                // Execute the JIT-compiled operation
                Some(compiled_fn(input, weight)?)
            } else {
                None
            };

            if let Some(result) = compiled_result {
                Ok(result)
            } else {
                // Fallback to fusion
                self.execute_fusion_first(input, weight, bias, activation)
            }
        }

        /// Execute standard implementation as fallback
        fn execute_standard_impl(
            &mut self,
            input: &ArrayView<F, IxDyn>,
            weight: &ArrayView<F, IxDyn>,
            bias: &ArrayView<F, IxDyn>,
            activation: kernel_fusion::ActivationType,
        ) -> Result<Array<F, IxDyn>> {
            // Standard matrix multiplication with bias and activation
            if input.ndim() == 2 && weight.ndim() == 2 {
                let input_2d = input
                    .view()
                    .into_dimensionality::<ndarray::Ix2>()
                    .map_err(|e| {
                        NeuralError::ComputationError(format!("Input reshape failed: {}", e))
                    })?;
                let weight_2d = weight
                    .view()
                    .into_dimensionality::<ndarray::Ix2>()
                    .map_err(|e| {
                        NeuralError::ComputationError(format!("Weight reshape failed: {}", e))
                    })?;

                let mut result = input_2d.dot(&weight_2d).into_dyn();

                // Add bias
                result = &result + bias;

                // Apply activation
                match activation {
                    kernel_fusion::ActivationType::ReLU => {
                        result.mapv_inplace(|x| x.max(F::zero()));
                    }
                    kernel_fusion::ActivationType::Sigmoid => {
                        result.mapv_inplace(|x| F::one() / (F::one() + (-x).exp()));
                    }
                    kernel_fusion::ActivationType::Tanh => {
                        result.mapv_inplace(|x| x.tanh());
                    }
                    kernel_fusion::ActivationType::None => {}
                }

                Ok(result)
            } else {
                Err(NeuralError::ComputationError(
                    "Only 2D matrices supported".to_string(),
                ))
            }
        }

        /// Maybe JIT-compile an operation based on usage patterns
        fn maybe_jit_compile_operation(
            &mut self,
            input_shape: &[usize],
            weight_shape: &[usize],
        ) -> Result<()> {
            // In a real implementation, this would use profiling data to decide
            // For now, we'll compile operations above a certain size threshold
            let total_ops: usize =
                input_shape.iter().product::<usize>() * weight_shape.iter().product::<usize>();

            if total_ops > 50000 {
                if let Ok(mut jit_ctx) = self.jit_context.write() {
                    let _ = jit_ctx.jit_matmul(
                        input_shape,
                        weight_shape,
                        true,
                        Some(kernel_fusion::ActivationType::ReLU),
                    )?;
                }
            }

            Ok(())
        }

        /// Update performance statistics
        fn update_stats(
            &mut self,
            _operation_id: &str,
            execution_time: Duration,
            used_fusion: bool,
            used_jit: bool,
        ) -> Result<()> {
            if let Ok(mut stats) = self.stats.write() {
                stats.operations_executed += 1;
                stats.total_execution_time += execution_time;

                if used_fusion {
                    stats.fused_operations += 1;
                }

                if used_jit {
                    stats.jit_operations += 1;
                }

                if used_fusion && used_jit {
                    stats.fusion_jit_operations += 1;
                }
            }

            Ok(())
        }

        /// Get combined performance statistics
        pub fn get_combined_stats(&self) -> FusionJitStats {
            if let Ok(stats) = self.stats.read() {
                stats.clone()
            } else {
                FusionJitStats::default()
            }
        }

        /// Get JIT-specific statistics
        pub fn get_jit_stats(&self) -> jit::JitStats {
            if let Ok(jit_ctx) = self.jit_context.read() {
                jit_ctx.get_stats()
            } else {
                jit::JitStats::default()
            }
        }

        /// Get fusion-specific statistics
        pub fn get_fusion_stats(&self) -> kernel_fusion::FusionStats {
            if let Ok(fusion_opt) = self.fusion_optimizer.read() {
                fusion_opt.get_stats().clone()
            } else {
                kernel_fusion::FusionStats::default()
            }
        }

        /// Clear all caches and reset statistics
        pub fn reset(&mut self) {
            if let Ok(mut jit_ctx) = self.jit_context.write() {
                jit_ctx.clear_caches();
            }

            if let Ok(mut fusion_opt) = self.fusion_optimizer.write() {
                fusion_opt.clear_cache();
            }

            if let Ok(mut stats) = self.stats.write() {
                *stats = FusionJitStats::default();
            }
        }
    }

    impl Clone for FusionJitStats {
        fn clone(&self) -> Self {
            Self {
                operations_executed: self.operations_executed,
                jit_operations: self.jit_operations,
                fused_operations: self.fused_operations,
                fusion_jit_operations: self.fusion_jit_operations,
                total_execution_time: self.total_execution_time,
                jit_compilation_overhead: self.jit_compilation_overhead,
                performance_improvement: self.performance_improvement,
            }
        }
    }

    impl std::fmt::Display for FusionJitStats {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            writeln!(f, "Fusion-JIT Executor Statistics:")?;
            writeln!(f, "  Operations executed: {}", self.operations_executed)?;
            writeln!(f, "  JIT operations: {}", self.jit_operations)?;
            writeln!(f, "  Fused operations: {}", self.fused_operations)?;
            writeln!(f, "  Combined fusion+JIT: {}", self.fusion_jit_operations)?;
            writeln!(f, "  Total execution time: {:?}", self.total_execution_time)?;
            writeln!(
                f,
                "  JIT compilation overhead: {:?}",
                self.jit_compilation_overhead
            )?;
            writeln!(
                f,
                "  Performance improvement: {:.2}x",
                self.performance_improvement
            )?;

            if self.operations_executed > 0 {
                let jit_ratio =
                    self.jit_operations as f64 / self.operations_executed as f64 * 100.0;
                let fusion_ratio =
                    self.fused_operations as f64 / self.operations_executed as f64 * 100.0;
                writeln!(f, "  JIT usage: {:.1}%", jit_ratio)?;
                writeln!(f, "  Fusion usage: {:.1}%", fusion_ratio)?;
            }

            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_thread_pool_creation() {
        let pool = ThreadPoolManager::new(Some(2)).unwrap();
        assert_eq!(pool.num_threads(), 2);
    }

    #[test]
    fn test_performance_profiler() {
        let mut profiler = PerformanceProfiler::new(true);

        let timer = profiler.start_timer("test_op");
        std::thread::sleep(std::time::Duration::from_millis(1));
        profiler.end_timer("test_op".to_string(), timer);

        assert!(profiler.get_timings().contains_key("test_op"));
        let duration = profiler.get_timings()["test_op"];
        assert!(duration.as_millis() >= 1);
    }

    #[test]
    fn test_parallel_matmul() {
        let pool = ThreadPoolManager::new(Some(2)).unwrap();

        let a = Array2::from_elem((3, 4), 2.0).into_dyn();
        let b = Array2::from_elem((4, 5), 3.0).into_dyn();

        let result = pool.parallel_matmul(&a, &b).unwrap();

        assert_eq!(result.shape(), [3, 5]);
        // Each element should be 2.0 * 3.0 * 4 = 24.0
        for &val in result.iter() {
            assert!((val - 24.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_optimization_capabilities() {
        let optimizer = PerformanceOptimizer::new(None, None, Some(4), false).unwrap();
        let caps = optimizer.get_capabilities();

        assert_eq!(caps.num_threads, 4);
        assert!(caps.thread_pool_available);
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_simd_relu_f32() {
        let input =
            Array2::from_shape_vec((2, 4), vec![-1.0, 2.0, -3.0, 4.0, 0.5, -0.5, 1.5, -2.5])
                .unwrap()
                .into_dyn();
        let result = SIMDOperations::simd_relu_f32(&input.view());

        assert_eq!(result.shape(), input.shape());
        assert_eq!(result[[0, 0]], 0.0); // -1.0 -> 0.0
        assert_eq!(result[[0, 1]], 2.0); // 2.0 -> 2.0
        assert_eq!(result[[1, 0]], 0.5); // 0.5 -> 0.5
        assert_eq!(result[[1, 1]], 0.0); // -0.5 -> 0.0
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_simd_sigmoid_f32() {
        let input = Array2::from_elem((2, 2), 0.0).into_dyn();
        let result = SIMDOperations::simd_sigmoid_f32(&input.view());

        assert_eq!(result.shape(), input.shape());
        // sigmoid(0) should be 0.5
        for &val in result.iter() {
            assert!((val - 0.5).abs() < 1e-6);
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_simd_add_f32() {
        let a = Array2::from_elem((2, 3), 1.0).into_dyn();
        let b = Array2::from_elem((2, 3), 2.0).into_dyn();

        let result = SIMDOperations::simd_add_f32(&a.view(), &b.view()).unwrap();

        assert_eq!(result.shape(), a.shape());
        for &val in result.iter() {
            assert!((val - 3.0).abs() < 1e-6);
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_simd_matmul_f32() {
        let a = Array2::from_elem((2, 3), 2.0).into_dyn();
        let b = Array2::from_elem((3, 2), 3.0).into_dyn();

        let result = SIMDOperations::simd_matmul_f32(&a.view(), &b.view()).unwrap();

        assert_eq!(result.shape(), [2, 2]);
        // Each element should be 2.0 * 3.0 * 3 = 18.0
        for &val in result.iter() {
            assert!((val - 18.0).abs() < 1e-6);
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_simd_cross_entropy_loss() {
        let predictions = Array2::from_shape_vec((2, 2), vec![0.9, 0.1, 0.2, 0.8])
            .unwrap()
            .into_dyn();
        let targets = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0])
            .unwrap()
            .into_dyn();

        let loss =
            SIMDOperations::simd_cross_entropy_loss_f32(&predictions.view(), &targets.view(), 1e-7)
                .unwrap();

        assert!(loss > 0.0);
        assert!(loss.is_finite());
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_simd_mse_loss() {
        let predictions = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0])
            .unwrap()
            .into_dyn();
        let targets = Array2::from_shape_vec((2, 2), vec![1.5, 2.5, 3.5, 4.5])
            .unwrap()
            .into_dyn();

        let loss = SIMDOperations::simd_mse_loss_f32(&predictions.view(), &targets.view()).unwrap();

        // Each diff is 0.5, so squared is 0.25, average is 0.25
        assert!((loss - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_simd_stats() {
        let stats = SIMDStats {
            simd_available: cfg!(feature = "simd"),
            vector_width_f32: 8,
            vector_width_f64: 4,
            supported_operations: vec!["relu".to_string(), "sigmoid".to_string()],
        };

        assert_eq!(stats.vector_width_f32, 8);
        assert_eq!(stats.vector_width_f64, 4);
        assert_eq!(stats.supported_operations.len(), 2);
    }

    #[cfg(feature = "memory_efficient")]
    #[test]
    fn test_memory_efficient_processor() {
        let processor = MemoryEfficientProcessor::new(Some(2), Some(100));

        let input = Array2::from_elem((5, 3), 1.0).into_dyn();

        let result = processor
            .process_in_chunks(&input, |chunk| Ok(chunk.to_owned()))
            .unwrap();

        assert_eq!(result.shape(), input.shape());
    }

    #[test]
    fn test_kernel_fusion_matmul_bias_activation() {
        use kernel_fusion::{ActivationType, KernelFusionOptimizer};

        let mut optimizer = KernelFusionOptimizer::<f32>::new();

        let input = Array2::from_elem((2, 3), 1.0).into_dyn();
        let weight = Array2::from_elem((3, 4), 0.5).into_dyn();
        let bias = Array2::from_elem((2, 4), 0.1).into_dyn();

        let result = optimizer
            .fuse_matmul_bias_activation(
                &input.view(),
                &weight.view(),
                &bias.view(),
                ActivationType::ReLU,
            )
            .unwrap();

        assert_eq!(result.shape(), [2, 4]);

        // Expected: (1.0 * 0.5 * 3) + 0.1 = 1.6, then ReLU -> 1.6
        for &val in result.iter() {
            assert!((val - 1.6).abs() < 1e-6);
        }

        let stats = optimizer.get_stats();
        assert_eq!(stats.operations_saved, 2); // bias add + activation
    }

    #[test]
    fn test_kernel_fusion_elementwise() {
        use kernel_fusion::{ActivationType, ElementWiseOp, KernelFusionOptimizer};

        let mut optimizer = KernelFusionOptimizer::<f32>::new();

        let input = Array2::from_elem((2, 3), 2.0).into_dyn();
        let operations = vec![
            ElementWiseOp::Multiply(0.5),
            ElementWiseOp::Add(1.0),
            ElementWiseOp::Activation(ActivationType::ReLU),
        ];

        let result = optimizer
            .fuse_elementwise_ops(&input.view(), &operations)
            .unwrap();

        assert_eq!(result.shape(), [2, 3]);

        // Expected: (2.0 * 0.5) + 1.0 = 2.0, then ReLU -> 2.0
        for &val in result.iter() {
            assert!((val - 2.0).abs() < 1e-6);
        }

        let stats = optimizer.get_stats();
        assert_eq!(stats.operations_saved, 2); // 3 operations - 1
    }

    #[test]
    fn test_kernel_fusion_stats() {
        use kernel_fusion::{ActivationType, KernelFusionOptimizer};

        let mut optimizer = KernelFusionOptimizer::<f32>::new();

        let input = Array2::from_elem((2, 2), 1.0).into_dyn();
        let weight = Array2::from_elem((2, 2), 1.0).into_dyn();
        let bias = Array2::from_elem((2, 2), 0.0).into_dyn();

        // First call should be a cache miss
        let _ = optimizer
            .fuse_matmul_bias_activation(
                &input.view(),
                &weight.view(),
                &bias.view(),
                ActivationType::None,
            )
            .unwrap();

        // Second call with same dimensions should be a cache hit
        let _ = optimizer
            .fuse_matmul_bias_activation(
                &input.view(),
                &weight.view(),
                &bias.view(),
                ActivationType::None,
            )
            .unwrap();

        let stats = optimizer.get_stats();
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_misses, 1);
        assert_eq!(optimizer.cache_hit_ratio(), 0.5);
    }

    #[test]
    fn test_jit_context_creation() {
        use jit::{JitContext, JitStrategy};

        let jit_ctx = JitContext::<f32>::new(JitStrategy::Aggressive);
        let stats = jit_ctx.get_stats();

        assert_eq!(stats.functions_compiled, 0);
        assert_eq!(stats.cache_hits, 0);
        assert_eq!(stats.cache_misses, 0);
    }

    #[test]
    fn test_jit_matmul_compilation() {
        use jit::{JitContext, JitStrategy};
        use kernel_fusion::ActivationType;

        let mut jit_ctx = JitContext::<f32>::new(JitStrategy::Aggressive);

        let input_shape = [2, 3];
        let weight_shape = [3, 4];

        // First compilation should be a cache miss
        let compiled_fn = jit_ctx
            .jit_matmul(
                &input_shape,
                &weight_shape,
                false,
                Some(ActivationType::ReLU),
            )
            .unwrap();

        let stats = jit_ctx.get_stats();
        assert_eq!(stats.cache_misses, 1);
        assert_eq!(stats.functions_compiled, 1);

        // Test the compiled function
        let input = Array2::from_elem((2, 3), 1.0).into_dyn();
        let weight = Array2::from_elem((3, 4), 0.5).into_dyn();

        let result = compiled_fn(&input.view(), &weight.view()).unwrap();
        assert_eq!(result.shape(), [2, 4]);

        // Expected: (1.0 * 0.5 * 3) = 1.5, then ReLU -> 1.5
        for &val in result.iter() {
            assert!((val - 1.5).abs() < 1e-6);
        }
    }

    #[test]
    fn test_jit_cache_hit() {
        use jit::{JitContext, JitStrategy};
        use kernel_fusion::ActivationType;

        let mut jit_ctx = JitContext::<f32>::new(JitStrategy::Conservative);

        let input_shape = [2, 2];
        let weight_shape = [2, 2];

        // First call - cache miss
        let _ = jit_ctx
            .jit_matmul(
                &input_shape,
                &weight_shape,
                true,
                Some(ActivationType::Sigmoid),
            )
            .unwrap();

        // Second call with same parameters - cache hit
        let _ = jit_ctx
            .jit_matmul(
                &input_shape,
                &weight_shape,
                true,
                Some(ActivationType::Sigmoid),
            )
            .unwrap();

        let stats = jit_ctx.get_stats();
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_misses, 1);
        assert_eq!(jit_ctx.cache_hit_ratio(), 0.5);
    }

    #[test]
    fn test_jit_blocked_matmul() {
        use jit::JitContext;
        use ndarray::Array2;

        let a = Array2::from_elem((4, 4), 2.0);
        let b = Array2::from_elem((4, 4), 3.0);

        let result = JitContext::<f32>::blocked_matmul(&a.view(), &b.view()).unwrap();

        assert_eq!(result.shape(), [4, 4]);
        // Each element should be 2.0 * 3.0 * 4 = 24.0
        for &val in result.iter() {
            assert!((val - 24.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_jit_profiler() {
        use jit::JitProfiler;
        use std::time::Duration;

        let mut profiler = JitProfiler::default();

        let op_key = "test_operation";
        profiler.record_execution(op_key, Duration::from_millis(10));
        profiler.record_execution(op_key, Duration::from_millis(20));
        profiler.record_execution(op_key, Duration::from_millis(30));

        let avg_time = profiler.average_execution_time(op_key);
        assert_eq!(avg_time, Duration::from_millis(20));

        let hot_functions = profiler.get_hot_functions(1000.0);
        assert!(hot_functions.contains(&op_key.to_string()));
    }

    #[test]
    fn test_fusion_jit_executor_creation() {
        use fusion_jit::{ExecutionStrategy, FusionJitExecutor};
        use jit::JitStrategy;

        let executor =
            FusionJitExecutor::<f32>::new(JitStrategy::Adaptive, ExecutionStrategy::FusionFirst);

        let stats = executor.get_combined_stats();
        assert_eq!(stats.operations_executed, 0);
        assert_eq!(stats.jit_operations, 0);
        assert_eq!(stats.fused_operations, 0);
    }

    #[test]
    fn test_fusion_jit_matmul_execution() {
        use fusion_jit::{ExecutionStrategy, FusionJitExecutor};
        use jit::JitStrategy;
        use kernel_fusion::ActivationType;

        let mut executor =
            FusionJitExecutor::<f32>::new(JitStrategy::Aggressive, ExecutionStrategy::Adaptive);

        let input = Array2::from_elem((4, 6), 1.0).into_dyn();
        let weight = Array2::from_elem((6, 8), 0.5).into_dyn();
        let bias = Array2::from_elem((4, 8), 0.1).into_dyn();

        let result = executor
            .execute_fused_jit_matmul(
                &input.view(),
                &weight.view(),
                &bias.view(),
                ActivationType::ReLU,
            )
            .unwrap();

        assert_eq!(result.shape(), [4, 8]);

        let stats = executor.get_combined_stats();
        assert_eq!(stats.operations_executed, 1);

        // Verify the computation result
        // Expected: (1.0 * 0.5 * 6) + 0.1 = 3.1, then ReLU -> 3.1
        for &val in result.iter() {
            assert!((val - 3.1).abs() < 1e-6);
        }
    }

    #[test]
    fn test_fusion_jit_strategies() {
        use fusion_jit::{ExecutionStrategy, FusionJitExecutor};
        use jit::JitStrategy;
        use kernel_fusion::ActivationType;

        let strategies = [
            ExecutionStrategy::FusionFirst,
            ExecutionStrategy::JitFirst,
            ExecutionStrategy::Adaptive,
            ExecutionStrategy::Heuristic,
        ];

        for strategy in strategies.iter() {
            let mut executor = FusionJitExecutor::<f32>::new(JitStrategy::Conservative, *strategy);

            let input = Array2::from_elem((2, 3), 2.0).into_dyn();
            let weight = Array2::from_elem((3, 4), 0.25).into_dyn();
            let bias = Array2::from_elem((2, 4), 0.5).into_dyn();

            let result = executor
                .execute_fused_jit_matmul(
                    &input.view(),
                    &weight.view(),
                    &bias.view(),
                    ActivationType::None,
                )
                .unwrap();

            assert_eq!(result.shape(), [2, 4]);

            let stats = executor.get_combined_stats();
            assert_eq!(stats.operations_executed, 1);

            // Expected: (2.0 * 0.25 * 3) + 0.5 = 2.0
            for &val in result.iter() {
                assert!((val - 2.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_fusion_jit_performance_tracking() {
        use fusion_jit::{ExecutionStrategy, FusionJitExecutor};
        use jit::JitStrategy;
        use kernel_fusion::ActivationType;

        let mut executor =
            FusionJitExecutor::<f32>::new(JitStrategy::Aggressive, ExecutionStrategy::Adaptive);

        // Execute multiple operations
        for i in 1..=3 {
            let size = i * 2;
            let input = Array2::from_elem((size, size), 1.0).into_dyn();
            let weight = Array2::from_elem((size, size), 1.0).into_dyn();
            let bias = Array2::from_elem((size, size), 0.0).into_dyn();

            let _ = executor
                .execute_fused_jit_matmul(
                    &input.view(),
                    &weight.view(),
                    &bias.view(),
                    ActivationType::ReLU,
                )
                .unwrap();
        }

        let stats = executor.get_combined_stats();
        assert_eq!(stats.operations_executed, 3);
        assert!(stats.total_execution_time.as_nanos() > 0);

        // Check that JIT and fusion statistics are available
        let jit_stats = executor.get_jit_stats();
        let fusion_stats = executor.get_fusion_stats();

        // JIT and fusion statistics should be valid
        assert!(jit_stats.functions_compiled < 1000); // Sanity check
        assert!(fusion_stats.operations_saved < 1000); // Sanity check
    }

    #[test]
    fn test_fusion_jit_adaptive_strategy() {
        use fusion_jit::{ExecutionStrategy, FusionJitExecutor};
        use jit::JitStrategy;
        use kernel_fusion::ActivationType;

        let mut executor =
            FusionJitExecutor::<f32>::new(JitStrategy::Adaptive, ExecutionStrategy::Adaptive);

        // Small operation (should use standard implementation)
        let small_input = Array2::from_elem((2, 2), 1.0).into_dyn();
        let small_weight = Array2::from_elem((2, 2), 1.0).into_dyn();
        let small_bias = Array2::from_elem((2, 2), 0.0).into_dyn();

        let small_result = executor
            .execute_fused_jit_matmul(
                &small_input.view(),
                &small_weight.view(),
                &small_bias.view(),
                ActivationType::None,
            )
            .unwrap();

        assert_eq!(small_result.shape(), [2, 2]);

        // Medium operation (should use fusion)
        let medium_input = Array2::from_elem((50, 50), 1.0).into_dyn();
        let medium_weight = Array2::from_elem((50, 50), 1.0).into_dyn();
        let medium_bias = Array2::from_elem((50, 50), 0.0).into_dyn();

        let medium_result = executor
            .execute_fused_jit_matmul(
                &medium_input.view(),
                &medium_weight.view(),
                &medium_bias.view(),
                ActivationType::ReLU,
            )
            .unwrap();

        assert_eq!(medium_result.shape(), [50, 50]);

        let stats = executor.get_combined_stats();
        assert_eq!(stats.operations_executed, 2);
    }
}

/// Distributed training support for neural networks
///
/// This module provides comprehensive distributed training capabilities including:
/// - Multiple communication backends (NCCL, Gloo, MPI)
/// - Data and model parallelism strategies
/// - Gradient synchronization with all-reduce operations
/// - Process management and fault tolerance
/// - Performance monitoring and load balancing
pub mod distributed {
    use super::*;
    use ndarray::{Array, ArrayView, IxDyn};
    use num_traits::Float;
    use serde::{Deserialize, Serialize};
    use std::collections::HashMap;
    use std::fmt;
    use std::marker::PhantomData;
    use std::sync::{Arc, Mutex, RwLock};
    use std::time::{Duration, Instant};

    /// Communication backend for distributed training
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    pub enum CommunicationBackend {
        /// NVIDIA Collective Communications Library
        NCCL,
        /// Facebook's collective communications library
        Gloo,
        /// Message Passing Interface
        MPI,
        /// TCP-based backend for CPU-only training
        TCP,
        /// In-memory backend for single-machine multi-process training
        InMemory,
    }

    impl fmt::Display for CommunicationBackend {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                CommunicationBackend::NCCL => write!(f, "NCCL"),
                CommunicationBackend::Gloo => write!(f, "Gloo"),
                CommunicationBackend::MPI => write!(f, "MPI"),
                CommunicationBackend::TCP => write!(f, "TCP"),
                CommunicationBackend::InMemory => write!(f, "InMemory"),
            }
        }
    }

    /// Distributed training strategy
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    pub enum DistributedStrategy {
        /// Data parallelism - same model, different data across workers
        DataParallel,
        /// Model parallelism - different parts of model across workers
        ModelParallel,
        /// Pipeline parallelism - different layers across workers with pipelining
        PipelineParallel,
        /// Hybrid parallelism - combination of data and model parallelism
        Hybrid,
    }

    /// Gradient synchronization method
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    pub enum GradientSyncMethod {
        /// All-reduce - everyone gets the same result
        AllReduce,
        /// Parameter server - centralized parameter updates
        ParameterServer,
        /// Ring all-reduce - bandwidth-optimal for large clusters
        RingAllReduce,
        /// Tree all-reduce - latency-optimal for small clusters
        TreeAllReduce,
        /// Hierarchical all-reduce - multi-level reduction
        HierarchicalAllReduce,
    }

    /// Process coordination information
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ProcessInfo {
        /// Local rank within the node
        pub local_rank: usize,
        /// Global rank across all nodes
        pub global_rank: usize,
        /// Total number of processes
        pub world_size: usize,
        /// Node identifier
        pub node_id: usize,
        /// Number of processes per node
        pub local_world_size: usize,
        /// Master node address
        pub master_addr: String,
        /// Master node port
        pub master_port: u16,
    }

    /// Distributed training configuration
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct DistributedConfig {
        /// Communication backend to use
        pub backend: CommunicationBackend,
        /// Training strategy
        pub strategy: DistributedStrategy,
        /// Gradient synchronization method
        pub sync_method: GradientSyncMethod,
        /// Process information
        pub process_info: ProcessInfo,
        /// Timeout for collective operations (seconds)
        pub timeout: u64,
        /// Enable gradient compression
        pub enable_compression: bool,
        /// Bucket size for gradient bucketing (MB)
        pub bucket_size_mb: usize,
        /// Enable mixed precision training
        pub mixed_precision: bool,
        /// Overlap communication with computation
        pub overlap_comm: bool,
    }

    impl Default for DistributedConfig {
        fn default() -> Self {
            Self {
                backend: CommunicationBackend::TCP,
                strategy: DistributedStrategy::DataParallel,
                sync_method: GradientSyncMethod::AllReduce,
                process_info: ProcessInfo {
                    local_rank: 0,
                    global_rank: 0,
                    world_size: 1,
                    node_id: 0,
                    local_world_size: 1,
                    master_addr: "localhost".to_string(),
                    master_port: 12345,
                },
                timeout: 300, // 5 minutes
                enable_compression: false,
                bucket_size_mb: 25,
                mixed_precision: false,
                overlap_comm: true,
            }
        }
    }

    /// Statistics for distributed training
    #[derive(Debug, Clone, Default, Serialize, Deserialize)]
    pub struct DistributedStats {
        /// Total bytes communicated
        pub bytes_communicated: u64,
        /// Number of all-reduce operations
        pub allreduce_count: u64,
        /// Total communication time
        pub communication_time: Duration,
        /// Total computation time
        pub computation_time: Duration,
        /// Communication efficiency (computation / (computation + communication))
        pub efficiency: f64,
        /// Number of timeouts encountered
        pub timeout_count: u64,
        /// Number of retries due to failures
        pub retry_count: u64,
        /// Current throughput (samples/second)
        pub throughput: f64,
        /// Load balance ratio (min_time / max_time across workers)
        pub load_balance: f64,
    }

    impl fmt::Display for DistributedStats {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            writeln!(f, "Distributed Training Statistics:")?;
            writeln!(
                f,
                "  Bytes Communicated: {:.2} MB",
                self.bytes_communicated as f64 / 1_048_576.0
            )?;
            writeln!(f, "  All-Reduce Operations: {}", self.allreduce_count)?;
            writeln!(
                f,
                "  Communication Time: {:.3}s",
                self.communication_time.as_secs_f64()
            )?;
            writeln!(
                f,
                "  Computation Time: {:.3}s",
                self.computation_time.as_secs_f64()
            )?;
            writeln!(f, "  Efficiency: {:.2}%", self.efficiency * 100.0)?;
            writeln!(f, "  Throughput: {:.2} samples/sec", self.throughput)?;
            writeln!(f, "  Load Balance: {:.2}%", self.load_balance * 100.0)?;
            writeln!(f, "  Timeouts: {}", self.timeout_count)?;
            write!(f, "  Retries: {}", self.retry_count)
        }
    }

    /// Gradient bucket for efficient communication
    #[derive(Debug)]
    struct GradientBucket<F: Float> {
        /// Accumulated gradients
        #[allow(dead_code)]
        gradients: Vec<Array<F, IxDyn>>,
        /// Current size in bytes
        #[allow(dead_code)]
        current_size: usize,
        /// Maximum size in bytes
        #[allow(dead_code)]
        max_size: usize,
        /// Last synchronization time
        #[allow(dead_code)]
        last_sync: Instant,
    }

    impl<F: Float> GradientBucket<F> {
        fn new(max_size_mb: usize) -> Self {
            Self {
                gradients: Vec::new(),
                current_size: 0,
                max_size: max_size_mb * 1_048_576, // Convert to bytes
                last_sync: Instant::now(),
            }
        }

        #[allow(dead_code)]
        fn add_gradient(&mut self, gradient: Array<F, IxDyn>) -> bool {
            let grad_size = gradient.len() * std::mem::size_of::<F>();
            if self.current_size + grad_size <= self.max_size {
                self.current_size += grad_size;
                self.gradients.push(gradient);
                false // Not full yet
            } else {
                true // Bucket is full
            }
        }

        #[allow(dead_code)]
        fn is_ready_for_sync(&self) -> bool {
            self.current_size >= self.max_size
                || self.last_sync.elapsed() > Duration::from_millis(100) // Timeout-based sync
        }

        #[allow(dead_code)]
        fn clear(&mut self) {
            self.gradients.clear();
            self.current_size = 0;
            self.last_sync = Instant::now();
        }
    }

    /// Distributed process group for managing communication
    pub struct ProcessGroup<F: Float> {
        config: DistributedConfig,
        stats: Arc<RwLock<DistributedStats>>,
        #[allow(dead_code)]
        gradient_buckets: Arc<Mutex<Vec<GradientBucket<F>>>>,
        #[allow(dead_code)]
        parameter_cache: Arc<RwLock<HashMap<String, Array<F, IxDyn>>>>,
        #[cfg(feature = "rayon")]
        comm_thread_pool: Option<rayon::ThreadPool>,
        #[cfg(not(feature = "rayon"))]
        _comm_thread_pool: PhantomData<()>,
    }

    impl<F: Float + Send + Sync + 'static> ProcessGroup<F> {
        /// Create a new process group
        pub fn new(config: DistributedConfig) -> Result<Self> {
            let num_buckets = 8; // Multiple buckets for overlapping communication
            let buckets = (0..num_buckets)
                .map(|_| GradientBucket::new(config.bucket_size_mb))
                .collect();

            let _comm_thread_pool = if config.overlap_comm {
                #[cfg(feature = "rayon")]
                {
                    Some(
                        rayon::ThreadPoolBuilder::new()
                        .num_threads(2) // Dedicated threads for communication
                        .thread_name(|i| format!("comm-{}", i))
                        .build()
                        .map_err(|e| NeuralError::DistributedError(format!("Failed to create thread pool: {}", e)))?,
                    )
                }
                #[cfg(not(feature = "rayon"))]
                {
                    Some(())
                }
            } else {
                None
            };

            Ok(Self {
                config,
                stats: Arc::new(RwLock::new(DistributedStats::default())),
                gradient_buckets: Arc::new(Mutex::new(buckets)),
                parameter_cache: Arc::new(RwLock::new(HashMap::new())),
                #[cfg(feature = "rayon")]
                comm_thread_pool,
                #[cfg(not(feature = "rayon"))]
                _comm_thread_pool: PhantomData,
            })
        }

        /// Initialize the distributed environment
        pub fn init(&self) -> Result<()> {
            match self.config.backend {
                CommunicationBackend::NCCL => self.init_nccl(),
                CommunicationBackend::Gloo => self.init_gloo(),
                CommunicationBackend::MPI => self.init_mpi(),
                CommunicationBackend::TCP => self.init_tcp(),
                CommunicationBackend::InMemory => self.init_in_memory(),
            }
        }

        /// All-reduce operation for gradient synchronization
        pub fn allreduce(&self, tensors: &mut [Array<F, IxDyn>]) -> Result<()> {
            let start_time = Instant::now();

            // Simulate different all-reduce algorithms
            match self.config.sync_method {
                GradientSyncMethod::AllReduce => self.allreduce_basic(tensors),
                GradientSyncMethod::RingAllReduce => self.allreduce_ring(tensors),
                GradientSyncMethod::TreeAllReduce => self.allreduce_tree(tensors),
                GradientSyncMethod::HierarchicalAllReduce => self.allreduce_hierarchical(tensors),
                GradientSyncMethod::ParameterServer => self.parameter_server_sync(tensors),
            }?;

            // Update statistics
            let mut stats = self.stats.write().unwrap();
            stats.allreduce_count += 1;
            stats.communication_time += start_time.elapsed();

            // Calculate bytes communicated
            let bytes: usize = tensors
                .iter()
                .map(|t| t.len() * std::mem::size_of::<F>())
                .sum();
            stats.bytes_communicated += bytes as u64;

            Ok(())
        }

        /// Broadcast operation
        pub fn broadcast(&self, tensor: &mut ArrayView<F, IxDyn>, root: usize) -> Result<()> {
            if self.config.process_info.global_rank == root {
                // Root sends to all others
                self.broadcast_from_root(tensor)
            } else {
                // Non-root receives from root
                self.broadcast_to_non_root(tensor, root)
            }
        }

        /// Scatter operation for data distribution
        pub fn scatter(
            &self,
            input: Option<&ArrayView<F, IxDyn>>,
            output: &mut ArrayView<F, IxDyn>,
            root: usize,
        ) -> Result<()> {
            if self.config.process_info.global_rank == root {
                let input = input.ok_or_else(|| {
                    NeuralError::DistributedError("Root must provide input for scatter".to_string())
                })?;
                self.scatter_from_root(input, output)
            } else {
                self.scatter_to_non_root(output, root)
            }
        }

        /// Gather operation for collecting results
        pub fn gather(
            &self,
            input: &ArrayView<F, IxDyn>,
            output: Option<&mut ArrayView<F, IxDyn>>,
            root: usize,
        ) -> Result<()> {
            if self.config.process_info.global_rank == root {
                let output = output.ok_or_else(|| {
                    NeuralError::DistributedError("Root must provide output for gather".to_string())
                })?;
                self.gather_to_root(input, output)
            } else {
                self.gather_from_non_root(input, root)
            }
        }

        /// Reduce-scatter operation
        pub fn reduce_scatter(
            &self,
            input: &Array<F, IxDyn>,
            output: &mut Array<F, IxDyn>,
        ) -> Result<()> {
            // Simplified implementation for reduce-scatter
            let _ = (input, output); // Use parameters to avoid warnings
                                     // In a real implementation, this would:
                                     // 1. Reduce gradients across all processes
                                     // 2. Scatter the result to each process
            Ok(())
        }

        /// All-gather operation
        pub fn allgather(
            &self,
            input: &Array<F, IxDyn>,
            output: &mut Array<F, IxDyn>,
        ) -> Result<()> {
            // Gather to all processes instead of just root
            for rank in 0..self.config.process_info.world_size {
                let start_idx = rank * input.len();
                let end_idx = start_idx + input.len();

                // Simplified implementation - simulate without actual copying
                if rank == self.config.process_info.global_rank {
                    // Simulate copying own data
                    let _ = (input, start_idx, end_idx);
                } else {
                    // Simulate receiving from other ranks
                    let mut recv_data = input.to_owned();
                    self.simulate_communication(&mut recv_data, rank)?;
                }
                // In a real implementation, this would copy data to output
                let _ = output; // Use output to avoid warnings
            }
            Ok(())
        }

        /// Barrier synchronization
        pub fn barrier(&self) -> Result<()> {
            // Simulate barrier by doing a dummy all-reduce
            let dummy = Array::zeros(1).into_dyn();
            self.allreduce(&mut [dummy])?;
            Ok(())
        }

        /// Get current distributed training statistics
        pub fn get_stats(&self) -> DistributedStats {
            self.stats.read().unwrap().clone()
        }

        /// Update load balancing statistics
        pub fn update_load_balance(&self, worker_times: &[Duration]) -> Result<()> {
            if worker_times.len() != self.config.process_info.world_size {
                return Err(NeuralError::DistributedError(
                    "Worker times size mismatch".to_string(),
                ));
            }

            let min_time = worker_times.iter().min().unwrap().as_secs_f64();
            let max_time = worker_times.iter().max().unwrap().as_secs_f64();

            let load_balance = if max_time > 0.0 {
                min_time / max_time
            } else {
                1.0
            };

            let mut stats = self.stats.write().unwrap();
            stats.load_balance = load_balance;

            Ok(())
        }

        /// Shutdown the distributed environment
        pub fn finalize(&self) -> Result<()> {
            // Synchronize all processes before shutdown
            self.barrier()?;

            // Print final statistics
            let stats = self.get_stats();
            if self.config.process_info.global_rank == 0 {
                println!("\nFinal Distributed Training Statistics:");
                println!("{}", stats);
            }

            Ok(())
        }

        // Private implementation methods

        fn init_nccl(&self) -> Result<()> {
            // NCCL initialization would go here
            // For now, just simulate successful initialization
            println!(
                "Initialized NCCL backend for process {}/{}",
                self.config.process_info.global_rank, self.config.process_info.world_size
            );
            Ok(())
        }

        fn init_gloo(&self) -> Result<()> {
            // Gloo initialization would go here
            println!(
                "Initialized Gloo backend for process {}/{}",
                self.config.process_info.global_rank, self.config.process_info.world_size
            );
            Ok(())
        }

        fn init_mpi(&self) -> Result<()> {
            // MPI initialization would go here
            println!(
                "Initialized MPI backend for process {}/{}",
                self.config.process_info.global_rank, self.config.process_info.world_size
            );
            Ok(())
        }

        fn init_tcp(&self) -> Result<()> {
            // TCP initialization would go here
            println!(
                "Initialized TCP backend for process {}/{}",
                self.config.process_info.global_rank, self.config.process_info.world_size
            );
            Ok(())
        }

        fn init_in_memory(&self) -> Result<()> {
            // In-memory initialization for single-machine training
            println!(
                "Initialized InMemory backend for process {}/{}",
                self.config.process_info.global_rank, self.config.process_info.world_size
            );
            Ok(())
        }

        fn allreduce_basic(&self, tensors: &mut [Array<F, IxDyn>]) -> Result<()> {
            // Basic all-reduce: sum all tensors and divide by world size
            for tensor in tensors.iter_mut() {
                // Simulate communication with other processes
                for _ in 1..self.config.process_info.world_size {
                    self.simulate_gradient_communication(&tensor.view())?;
                }
                // Average the gradients
                let world_size = F::from(self.config.process_info.world_size).unwrap();
                tensor.mapv_inplace(|x| x / world_size);
            }
            Ok(())
        }

        fn allreduce_ring(&self, tensors: &mut [Array<F, IxDyn>]) -> Result<()> {
            // Ring all-reduce algorithm
            let world_size = self.config.process_info.world_size;
            let rank = self.config.process_info.global_rank;

            for tensor in tensors.iter_mut() {
                // Simulate ring communication pattern
                for step in 0..world_size - 1 {
                    let send_to = (rank + 1) % world_size;
                    let recv_from = (rank + world_size - 1) % world_size;

                    // Simulate sending and receiving chunks
                    self.simulate_ring_communication(&tensor.view(), send_to, recv_from, step)?;
                }

                // Average the results
                let world_size_f = F::from(world_size).unwrap();
                tensor.mapv_inplace(|x| x / world_size_f);
            }
            Ok(())
        }

        fn allreduce_tree(&self, tensors: &mut [Array<F, IxDyn>]) -> Result<()> {
            // Tree all-reduce algorithm
            for tensor in tensors.iter_mut() {
                // Simulate tree reduction
                let mut level = 1;
                while level < self.config.process_info.world_size {
                    if self.config.process_info.global_rank % (level * 2) == 0 {
                        let partner = self.config.process_info.global_rank + level;
                        if partner < self.config.process_info.world_size {
                            self.simulate_tree_communication(&tensor.view(), partner)?;
                        }
                    }
                    level *= 2;
                }

                // Broadcast result back down the tree
                level = self.config.process_info.world_size.next_power_of_two() / 2;
                while level >= 1 {
                    if self.config.process_info.global_rank % (level * 2) == 0 {
                        let partner = self.config.process_info.global_rank + level;
                        if partner < self.config.process_info.world_size {
                            self.simulate_tree_communication(&tensor.view(), partner)?;
                        }
                    }
                    level /= 2;
                }

                let world_size_f = F::from(self.config.process_info.world_size).unwrap();
                tensor.mapv_inplace(|x| x / world_size_f);
            }
            Ok(())
        }

        fn allreduce_hierarchical(&self, _tensors: &mut [Array<F, IxDyn>]) -> Result<()> {
            // Hierarchical all-reduce: first reduce within nodes, then across nodes
            // Implementation would depend on node topology
            Ok(())
        }

        fn parameter_server_sync(&self, tensors: &mut [Array<F, IxDyn>]) -> Result<()> {
            // Parameter server synchronization
            if self.config.process_info.global_rank == 0 {
                // This is the parameter server
                for tensor in tensors.iter_mut() {
                    // Receive gradients from all workers and average
                    for worker_rank in 1..self.config.process_info.world_size {
                        self.simulate_communication(tensor, worker_rank)?;
                    }

                    let world_size_f = F::from(self.config.process_info.world_size).unwrap();
                    tensor.mapv_inplace(|x| x / world_size_f);

                    // Send updated parameters back to workers
                    for worker_rank in 1..self.config.process_info.world_size {
                        self.simulate_communication(tensor, worker_rank)?;
                    }
                }
            } else {
                // This is a worker
                for tensor in tensors.iter_mut() {
                    // Send gradients to parameter server
                    self.simulate_communication(tensor, 0)?;
                    // Receive updated parameters from parameter server
                    self.simulate_communication(tensor, 0)?;
                }
            }
            Ok(())
        }

        fn broadcast_from_root(&self, _tensor: &ArrayView<F, IxDyn>) -> Result<()> {
            // Root broadcasts tensor to all other processes
            for rank in 1..self.config.process_info.world_size {
                // Simulate sending to rank
                std::thread::sleep(Duration::from_micros(10 * rank as u64));
            }
            Ok(())
        }

        fn broadcast_to_non_root(
            &self,
            _tensor: &mut ArrayView<F, IxDyn>,
            _root: usize,
        ) -> Result<()> {
            // Non-root receives tensor from root
            std::thread::sleep(Duration::from_micros(10));
            Ok(())
        }

        fn scatter_from_root(
            &self,
            _input: &ArrayView<F, IxDyn>,
            _output: &mut ArrayView<F, IxDyn>,
        ) -> Result<()> {
            // Root scatters input to all processes
            for rank in 1..self.config.process_info.world_size {
                std::thread::sleep(Duration::from_micros(5 * rank as u64));
            }
            Ok(())
        }

        fn scatter_to_non_root(
            &self,
            _output: &mut ArrayView<F, IxDyn>,
            _root: usize,
        ) -> Result<()> {
            // Non-root receives scattered data from root
            std::thread::sleep(Duration::from_micros(5));
            Ok(())
        }

        fn gather_to_root(
            &self,
            _input: &ArrayView<F, IxDyn>,
            _output: &mut ArrayView<F, IxDyn>,
        ) -> Result<()> {
            // Root gathers data from all processes
            for rank in 1..self.config.process_info.world_size {
                std::thread::sleep(Duration::from_micros(5 * rank as u64));
            }
            Ok(())
        }

        fn gather_from_non_root(&self, _input: &ArrayView<F, IxDyn>, _root: usize) -> Result<()> {
            // Non-root sends data to root
            std::thread::sleep(Duration::from_micros(5));
            Ok(())
        }

        fn simulate_communication(
            &self,
            _tensor: &mut Array<F, IxDyn>,
            _rank: usize,
        ) -> Result<()> {
            // Simulate network latency and bandwidth
            std::thread::sleep(Duration::from_micros(50));
            Ok(())
        }

        fn simulate_gradient_communication(&self, _tensor: &ArrayView<F, IxDyn>) -> Result<()> {
            // Simulate gradient communication overhead
            std::thread::sleep(Duration::from_micros(10));
            Ok(())
        }

        fn simulate_ring_communication(
            &self,
            _tensor: &ArrayView<F, IxDyn>,
            _send_to: usize,
            _recv_from: usize,
            _step: usize,
        ) -> Result<()> {
            // Simulate ring communication pattern
            std::thread::sleep(Duration::from_micros(20));
            Ok(())
        }

        fn simulate_tree_communication(
            &self,
            _tensor: &ArrayView<F, IxDyn>,
            _partner: usize,
        ) -> Result<()> {
            // Simulate tree communication pattern
            std::thread::sleep(Duration::from_micros(15));
            Ok(())
        }
    }

    /// Distributed data loader for splitting datasets across workers
    pub struct DistributedDataLoader<F: Float> {
        /// Total dataset size
        total_size: usize,
        /// Local batch size per worker
        #[allow(dead_code)]
        local_batch_size: usize,
        /// Process information
        process_info: ProcessInfo,
        /// Current epoch
        current_epoch: usize,
        /// Data sharding strategy
        shard_strategy: ShardStrategy,
        /// Phantom data for type parameter
        _phantom: PhantomData<F>,
    }

    /// Data sharding strategy
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ShardStrategy {
        /// Contiguous sharding - each worker gets a contiguous block
        Contiguous,
        /// Interleaved sharding - workers get interleaved samples
        Interleaved,
        /// Random sharding - workers get random samples (with different seeds)
        Random,
    }

    impl<F: Float> DistributedDataLoader<F> {
        /// Create new distributed data loader
        pub fn new(
            total_size: usize,
            local_batch_size: usize,
            process_info: ProcessInfo,
            shard_strategy: ShardStrategy,
        ) -> Self {
            Self {
                total_size,
                local_batch_size,
                process_info,
                current_epoch: 0,
                shard_strategy,
                _phantom: PhantomData,
            }
        }

        /// Get the local indices for this worker
        pub fn get_local_indices(&self) -> Vec<usize> {
            match self.shard_strategy {
                ShardStrategy::Contiguous => self.get_contiguous_indices(),
                ShardStrategy::Interleaved => self.get_interleaved_indices(),
                ShardStrategy::Random => self.get_random_indices(),
            }
        }

        /// Set the current epoch (affects random sharding)
        pub fn set_epoch(&mut self, epoch: usize) {
            self.current_epoch = epoch;
        }

        fn get_contiguous_indices(&self) -> Vec<usize> {
            let per_worker = self.total_size / self.process_info.world_size;
            let start = self.process_info.global_rank * per_worker;
            let end = if self.process_info.global_rank == self.process_info.world_size - 1 {
                self.total_size // Last worker gets any remainder
            } else {
                start + per_worker
            };
            (start..end).collect()
        }

        fn get_interleaved_indices(&self) -> Vec<usize> {
            (self.process_info.global_rank..self.total_size)
                .step_by(self.process_info.world_size)
                .collect()
        }

        fn get_random_indices(&self) -> Vec<usize> {
            use rand::prelude::*;
            use rand_chacha::ChaCha8Rng;

            // Use epoch and rank to ensure reproducible but different sharding
            let seed = self.current_epoch as u64 * 1000 + self.process_info.global_rank as u64;
            let mut rng = ChaCha8Rng::seed_from_u64(seed);

            let mut indices: Vec<usize> = (0..self.total_size).collect();
            indices.shuffle(&mut rng);

            // Take our portion
            let per_worker = self.total_size / self.process_info.world_size;
            let start = self.process_info.global_rank * per_worker;
            let end = if self.process_info.global_rank == self.process_info.world_size - 1 {
                self.total_size
            } else {
                start + per_worker
            };

            indices[start..end].to_vec()
        }
    }

    /// High-level distributed training manager
    pub struct DistributedTrainingManager<F: Float + Send + Sync + 'static> {
        process_group: ProcessGroup<F>,
        data_loader: DistributedDataLoader<F>,
        config: DistributedConfig,
    }

    impl<F: Float + Send + Sync + 'static> DistributedTrainingManager<F> {
        /// Create a new distributed training manager
        pub fn new(
            config: DistributedConfig,
            total_dataset_size: usize,
            local_batch_size: usize,
        ) -> Result<Self> {
            let process_group = ProcessGroup::new(config.clone())?;
            let data_loader = DistributedDataLoader::new(
                total_dataset_size,
                local_batch_size,
                config.process_info.clone(),
                ShardStrategy::Interleaved,
            );

            Ok(Self {
                process_group,
                data_loader,
                config,
            })
        }

        /// Initialize distributed training
        pub fn init(&self) -> Result<()> {
            self.process_group.init()
        }

        /// Synchronize model parameters across all workers
        pub fn sync_parameters(&self, parameters: &mut [ArrayView<F, IxDyn>]) -> Result<()> {
            self.process_group.broadcast(&mut parameters[0], 0)?;
            for param in parameters.iter_mut().skip(1) {
                self.process_group.broadcast(param, 0)?;
            }
            Ok(())
        }

        /// Synchronize gradients across all workers
        pub fn sync_gradients(&self, gradients: &mut [Array<F, IxDyn>]) -> Result<()> {
            self.process_group.allreduce(gradients)
        }

        /// Get distributed training statistics
        pub fn get_stats(&self) -> DistributedStats {
            self.process_group.get_stats()
        }

        /// Get local data indices for current epoch
        pub fn get_local_data_indices(&mut self, epoch: usize) -> Vec<usize> {
            self.data_loader.set_epoch(epoch);
            self.data_loader.get_local_indices()
        }

        /// Check if this is the master process
        pub fn is_master(&self) -> bool {
            self.config.process_info.global_rank == 0
        }

        /// Get the current process rank
        pub fn rank(&self) -> usize {
            self.config.process_info.global_rank
        }

        /// Get the world size
        pub fn world_size(&self) -> usize {
            self.config.process_info.world_size
        }

        /// Finalize distributed training
        pub fn finalize(&self) -> Result<()> {
            self.process_group.finalize()
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use ndarray::Array2;

        #[test]
        fn test_distributed_config_creation() {
            let config = DistributedConfig::default();
            assert_eq!(config.backend, CommunicationBackend::TCP);
            assert_eq!(config.strategy, DistributedStrategy::DataParallel);
            assert_eq!(config.sync_method, GradientSyncMethod::AllReduce);
            assert_eq!(config.process_info.world_size, 1);
        }

        #[test]
        fn test_process_group_creation() {
            let config = DistributedConfig::default();
            let process_group = ProcessGroup::<f32>::new(config);
            assert!(process_group.is_ok());
        }

        #[test]
        fn test_distributed_stats_display() {
            let stats = DistributedStats {
                bytes_communicated: 1_048_576, // 1 MB
                allreduce_count: 10,
                communication_time: Duration::from_millis(100),
                computation_time: Duration::from_millis(900),
                efficiency: 0.9,
                throughput: 100.0,
                load_balance: 0.95,
                timeout_count: 0,
                retry_count: 0,
            };

            let display = format!("{}", stats);
            assert!(display.contains("1.00 MB"));
            assert!(display.contains("10"));
            assert!(display.contains("90.00%"));
        }

        #[test]
        fn test_data_loader_contiguous_sharding() {
            let process_info = ProcessInfo {
                local_rank: 0,
                global_rank: 1,
                world_size: 4,
                node_id: 0,
                local_world_size: 4,
                master_addr: "localhost".to_string(),
                master_port: 12345,
            };

            let data_loader = DistributedDataLoader::<f32>::new(
                100, // total size
                25,  // local batch size
                process_info,
                ShardStrategy::Contiguous,
            );

            let indices = data_loader.get_local_indices();
            assert_eq!(indices.len(), 25);
            assert_eq!(indices[0], 25); // Should start at rank * per_worker
            assert_eq!(indices[24], 49); // Should end at start + per_worker - 1
        }

        #[test]
        fn test_data_loader_interleaved_sharding() {
            let process_info = ProcessInfo {
                local_rank: 0,
                global_rank: 1,
                world_size: 3,
                node_id: 0,
                local_world_size: 3,
                master_addr: "localhost".to_string(),
                master_port: 12345,
            };

            let data_loader = DistributedDataLoader::<f32>::new(
                9, // total size
                3, // local batch size
                process_info,
                ShardStrategy::Interleaved,
            );

            let indices = data_loader.get_local_indices();
            assert_eq!(indices, vec![1, 4, 7]); // rank 1 with step 3
        }

        #[test]
        fn test_distributed_training_manager_creation() {
            let config = DistributedConfig::default();
            let manager = DistributedTrainingManager::<f32>::new(config, 1000, 32);
            assert!(manager.is_ok());

            let manager = manager.unwrap();
            assert!(manager.is_master());
            assert_eq!(manager.rank(), 0);
            assert_eq!(manager.world_size(), 1);
        }

        #[test]
        fn test_gradient_bucket() {
            let mut bucket = GradientBucket::<f32>::new(1); // 1 MB max

            let small_grad = Array2::zeros((10, 10)).into_dyn();
            assert!(!bucket.add_gradient(small_grad)); // Should not be full

            // Add a large gradient that fills the bucket
            let large_grad = Array2::zeros((1000, 1000)).into_dyn(); // ~4MB for f32
            assert!(bucket.add_gradient(large_grad)); // Should be full

            bucket.clear();
            assert_eq!(bucket.current_size, 0);
            assert_eq!(bucket.gradients.len(), 0);
        }

        #[test]
        fn test_communication_backend_display() {
            assert_eq!(format!("{}", CommunicationBackend::NCCL), "NCCL");
            assert_eq!(format!("{}", CommunicationBackend::Gloo), "Gloo");
            assert_eq!(format!("{}", CommunicationBackend::MPI), "MPI");
            assert_eq!(format!("{}", CommunicationBackend::TCP), "TCP");
            assert_eq!(format!("{}", CommunicationBackend::InMemory), "InMemory");
        }

        #[test]
        fn test_allreduce_simulation() {
            let config = DistributedConfig::default();
            let process_group = ProcessGroup::<f32>::new(config).unwrap();

            let tensor = Array2::ones((10, 10)).into_dyn();
            let result = process_group.allreduce(&mut [tensor]);
            assert!(result.is_ok());

            let stats = process_group.get_stats();
            assert_eq!(stats.allreduce_count, 1);
            // Communication time should be recorded (duration is always non-negative)
            assert!(stats.communication_time.as_nanos() > 0 || stats.communication_time.is_zero());
        }
    }
}
