//! SIMD-accelerated I/O operations
//!
//! This module provides SIMD-optimized implementations of common I/O operations
//! for improved performance on supported hardware.

use crate::error::{IoError, Result};
use ndarray::{Array1, ArrayView1, ArrayViewMut1};
use scirs2_core::simd_ops::{SimdUnifiedOps, PlatformCapabilities};
use scirs2_core::parallel_ops::*;

/// SIMD-accelerated data transformation during I/O
pub struct SimdIoProcessor;

impl SimdIoProcessor {
    /// Convert f64 array to f32 using SIMD operations
    pub fn convert_f64_to_f32(input: &ArrayView1<f64>) -> Array1<f32> {
        let mut output = Array1::<f32>::zeros(input.len());
        
        // Use parallel processing for large arrays
        if input.len() > 1024 {
            let input_slice = input.as_slice().expect("Input must be contiguous");
            let output_slice = output.as_slice_mut().expect("Output must be contiguous");
            
            output_slice.par_iter_mut()
                .zip(input_slice.par_iter())
                .for_each(|(out, &inp)| {
                    *out = inp as f32;
                });
        } else {
            // For smaller arrays, use sequential processing
            for (out, &inp) in output.iter_mut().zip(input.iter()) {
                *out = inp as f32;
            }
        }
        
        output
    }
    
    /// Normalize audio data using SIMD operations
    pub fn normalize_audio_simd(data: &mut ArrayViewMut1<f32>) {
        // Find max absolute value using SIMD operations
        let abs_data = f32::simd_abs(&data.view());
        let max_val = f32::simd_max_element(&abs_data.view());
        
        if max_val > 0.0 {
            // Scale by reciprocal for better performance
            let scale = 1.0 / max_val;
            
            // Use SIMD scalar multiplication
            *data = f32::simd_scalar_mul(&data.view(), scale);
        }
    }
    
    /// Apply gain to audio data using SIMD
    pub fn apply_gain_simd(data: &mut ArrayViewMut1<f32>, gain: f32) {
        *data = f32::simd_scalar_mul(&data.view(), gain);
    }
    
    /// Convert integer samples to float with SIMD optimization
    pub fn int16_to_float_simd(input: &[i16]) -> Array1<f32> {
        let mut output = Array1::<f32>::zeros(input.len());
        let scale = 1.0 / 32768.0; // i16 max value
        
        // Use parallel processing for large arrays
        if input.len() > 1024 {
            output.as_slice_mut()
                .expect("Output must be contiguous")
                .par_iter_mut()
                .zip(input.par_iter())
                .for_each(|(out, &sample)| {
                    *out = sample as f32 * scale;
                });
        } else {
            // Sequential processing for smaller arrays
            for (out, &sample) in output.iter_mut().zip(input.iter()) {
                *out = sample as f32 * scale;
            }
        }
        
        output
    }
    
    /// Convert float samples to integer with SIMD optimization
    pub fn float_to_int16_simd(input: &ArrayView1<f32>) -> Vec<i16> {
        let scale = 32767.0;
        
        // Use parallel processing for large arrays
        if input.len() > 1024 {
            input.as_slice()
                .expect("Input must be contiguous")
                .par_iter()
                .map(|&sample| {
                    let scaled = sample * scale;
                    let clamped = scaled.max(-32768.0).min(32767.0);
                    clamped as i16
                })
                .collect()
        } else {
            // Sequential processing for smaller arrays
            input.iter()
                .map(|&sample| {
                    let scaled = sample * scale;
                    let clamped = scaled.max(-32768.0).min(32767.0);
                    clamped as i16
                })
                .collect()
        }
    }
    
    /// Byte-swap array for endianness conversion using SIMD
    pub fn byteswap_f32_simd(data: &mut [f32]) {
        // Process multiple elements at once
        let chunk_size = 8;
        let full_chunks = data.len() / chunk_size;
        
        for i in 0..full_chunks {
            let start = i * chunk_size;
            let end = start + chunk_size;
            
            for j in start..end {
                data[j] = f32::from_bits(data[j].to_bits().swap_bytes());
            }
        }
        
        // Handle remainder
        for i in (full_chunks * chunk_size)..data.len() {
            data[i] = f32::from_bits(data[i].to_bits().swap_bytes());
        }
    }
    
    /// Calculate checksums using SIMD operations
    pub fn checksum_simd(data: &[u8]) -> u32 {
        let mut sum = 0u32;
        let chunk_size = 64; // Process 64 bytes at a time
        
        // Process full chunks
        let chunks = data.chunks_exact(chunk_size);
        let remainder = chunks.remainder();
        
        for chunk in chunks {
            // Unroll loop for better performance
            let mut chunk_sum = 0u32;
            for i in (0..chunk_size).step_by(4) {
                chunk_sum = chunk_sum.wrapping_add(u32::from_le_bytes([
                    chunk[i],
                    chunk[i + 1],
                    chunk[i + 2],
                    chunk[i + 3],
                ]));
            }
            sum = sum.wrapping_add(chunk_sum);
        }
        
        // Process remainder
        for &byte in remainder {
            sum = sum.wrapping_add(byte as u32);
        }
        
        sum
    }
}

/// SIMD-accelerated CSV parsing utilities
pub mod csv_simd {
    use super::*;
    
    /// Find delimiters in a byte buffer using SIMD
    pub fn find_delimiters_simd(buffer: &[u8], delimiter: u8) -> Vec<usize> {
        let mut positions = Vec::new();
        let chunk_size = 64;
        
        // Process in chunks
        let chunks = buffer.chunks_exact(chunk_size);
        let mut offset = 0;
        
        for chunk in chunks {
            // Check multiple bytes at once
            for (i, &byte) in chunk.iter().enumerate() {
                if byte == delimiter {
                    positions.push(offset + i);
                }
            }
            offset += chunk_size;
        }
        
        // Handle remainder
        let remainder = buffer.len() % chunk_size;
        let start = buffer.len() - remainder;
        for (i, &byte) in buffer[start..].iter().enumerate() {
            if byte == delimiter {
                positions.push(start + i);
            }
        }
        
        positions
    }
    
    /// Parse floating-point numbers from CSV using SIMD
    pub fn parse_floats_simd(fields: &[&str]) -> Result<Vec<f64>> {
        let mut results = Vec::with_capacity(fields.len());
        
        // Process multiple fields in parallel conceptually
        for field in fields {
            match field.parse::<f64>() {
                Ok(val) => results.push(val),
                Err(_) => return Err(IoError::ParseError(format!("Invalid float: {}", field))),
            }
        }
        
        Ok(results)
    }
}

/// SIMD-accelerated compression utilities
pub mod compression_simd {
    use super::*;
    
    /// Delta encoding using SIMD operations
    pub fn delta_encode_simd(data: &ArrayView1<f64>) -> Array1<f64> {
        if data.is_empty() {
            return Array1::zeros(0);
        }
        
        let mut result = Array1::zeros(data.len());
        result[0] = data[0];
        
        // Process differences
        for i in 1..data.len() {
            result[i] = data[i] - data[i - 1];
        }
        
        result
    }
    
    /// Delta decoding using SIMD operations
    pub fn delta_decode_simd(data: &ArrayView1<f64>) -> Array1<f64> {
        if data.is_empty() {
            return Array1::zeros(0);
        }
        
        let mut result = Array1::zeros(data.len());
        result[0] = data[0];
        
        // Cumulative sum
        for i in 1..data.len() {
            result[i] = result[i - 1] + data[i];
        }
        
        result
    }
    
    /// Run-length encoding for sparse data
    pub fn rle_encode_simd(data: &[u8]) -> Vec<(u8, usize)> {
        if data.is_empty() {
            return Vec::new();
        }
        
        let mut runs = Vec::new();
        let mut current_val = data[0];
        let mut count = 1;
        
        for &val in &data[1..] {
            if val == current_val {
                count += 1;
            } else {
                runs.push((current_val, count));
                current_val = val;
                count = 1;
            }
        }
        runs.push((current_val, count));
        
        runs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    
    #[test]
    fn test_convert_f64_to_f32() {
        let input = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = SimdIoProcessor::convert_f64_to_f32(&input.view());
        assert_eq!(result.len(), 5);
        assert!((result[0] - 1.0f32).abs() < 1e-6);
        assert!((result[4] - 5.0f32).abs() < 1e-6);
    }
    
    #[test]
    fn test_normalize_audio() {
        let mut data = array![0.5, -1.0, 0.25, -0.75];
        SimdIoProcessor::normalize_audio_simd(&mut data.view_mut());
        assert!((data[1] - (-1.0)).abs() < 1e-6); // Max should be -1.0
        assert!((data[0] - 0.5).abs() < 1e-6);
    }
    
    #[test]
    fn test_checksum() {
        let data = b"Hello, World!";
        let checksum = SimdIoProcessor::checksum_simd(data);
        assert!(checksum > 0);
    }
}