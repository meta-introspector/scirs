//! Performance optimization utilities for neural networks
//!
//! This module provides performance optimizations including SIMD acceleration,
//! memory-efficient operations, and thread pool support.

use crate::error::{NeuralError, Result};
use ndarray::{Array, ArrayD};
#[allow(unused_imports)]
use num_traits::Float;
use std::fmt::Debug;
use std::sync::Arc;

#[cfg(feature = "simd")]
use scirs2_core::simd::*;

#[cfg(feature = "memory_efficient")]
use scirs2_core::memory_efficient::{chunk_wise_op, ChunkProcessor};

#[cfg(feature = "simd")]
use wide::{f32x4, f32x8, f64x2, f64x4, CmpGt};

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
            let result = vec.tanh();
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
            let tanh_val = inner.tanh();
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
        let feature_size = input.shape()[last_axis];

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
            for (i, (&inp, &m, &v)) in input
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
        &self,
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
}
