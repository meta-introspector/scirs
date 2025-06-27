//! Validation utilities for Wavelet Packet Transform
//!
//! This module provides comprehensive validation functions for WPT implementations,
//! including energy conservation checks, reconstruction accuracy, and numerical stability.

use crate::error::{SignalError, SignalResult};
use crate::wpt::{wpt_decompose, wpt_reconstruct, WaveletPacketTree};
use crate::dwt::Wavelet;
use ndarray::{Array1, ArrayView1};
use scirs2_core::simd_ops::SimdUnifiedOps;
use scirs2_core::validation::{check_finite, check_positive};
use std::collections::HashMap;

/// Validation result for WPT
#[derive(Debug, Clone)]
pub struct WptValidationResult {
    /// Energy conservation ratio (should be close to 1.0)
    pub energy_ratio: f64,
    /// Maximum reconstruction error
    pub max_reconstruction_error: f64,
    /// Mean reconstruction error
    pub mean_reconstruction_error: f64,
    /// Signal-to-noise ratio of reconstruction
    pub reconstruction_snr: f64,
    /// Parseval frame ratio (should be close to 1.0)
    pub parseval_ratio: f64,
    /// Numerical stability score (0-1)
    pub stability_score: f64,
    /// Issues found during validation
    pub issues: Vec<String>,
}

/// Validate WPT decomposition and reconstruction
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `wavelet` - Wavelet to use
/// * `max_level` - Maximum decomposition level
/// * `tolerance` - Tolerance for numerical comparisons
///
/// # Returns
///
/// * Validation result with detailed metrics
pub fn validate_wpt<T>(
    signal: &[T],
    wavelet: Wavelet,
    max_level: usize,
    tolerance: f64,
) -> SignalResult<WptValidationResult>
where
    T: Float + NumCast,
{
    // Convert to f64
    let signal_f64: Vec<f64> = signal
        .iter()
        .map(|&x| NumCast::from(x).unwrap())
        .collect();
    
    check_finite(&signal_f64, "signal")?;
    check_positive(max_level, "max_level")?;
    
    let mut issues = Vec::new();
    
    // Perform decomposition
    let tree = wpt_decompose(&signal_f64, wavelet, max_level, None)?;
    
    // Test 1: Energy conservation
    let input_energy = compute_energy(&signal_f64);
    let tree_energy = compute_tree_energy(&tree)?;
    let energy_ratio = tree_energy / input_energy;
    
    if (energy_ratio - 1.0).abs() > tolerance {
        issues.push(format!(
            "Energy not conserved: ratio = {:.6} (expected ≈ 1.0)",
            energy_ratio
        ));
    }
    
    // Test 2: Perfect reconstruction
    let reconstructed = wpt_reconstruct(&tree)?;
    let reconstruction_errors = validate_reconstruction(&signal_f64, &reconstructed, tolerance)?;
    
    if reconstruction_errors.max_error > tolerance {
        issues.push(format!(
            "Reconstruction error exceeds tolerance: {:.2e} > {:.2e}",
            reconstruction_errors.max_error, tolerance
        ));
    }
    
    // Test 3: Parseval frame property
    let parseval_ratio = validate_parseval_frame(&tree, &signal_f64)?;
    
    if (parseval_ratio - 1.0).abs() > tolerance * 10.0 {
        issues.push(format!(
            "Parseval frame property violated: ratio = {:.6}",
            parseval_ratio
        ));
    }
    
    // Test 4: Numerical stability
    let stability_score = test_numerical_stability(&signal_f64, wavelet, max_level)?;
    
    if stability_score < 0.9 {
        issues.push(format!(
            "Numerical stability concerns: score = {:.2}",
            stability_score
        ));
    }
    
    Ok(WptValidationResult {
        energy_ratio,
        max_reconstruction_error: reconstruction_errors.max_error,
        mean_reconstruction_error: reconstruction_errors.mean_error,
        reconstruction_snr: reconstruction_errors.snr,
        parseval_ratio,
        stability_score,
        issues,
    })
}

/// Compute signal energy
fn compute_energy(signal: &[f64]) -> f64 {
    let signal_view = ArrayView1::from(signal);
    f64::simd_dot(&signal_view, &signal_view)
}

/// Compute total energy in wavelet packet tree
fn compute_tree_energy(tree: &WaveletPacketTree) -> SignalResult<f64> {
    let mut total_energy = 0.0;
    
    // Get all leaf nodes (terminal nodes)
    let leaf_nodes = tree.get_leaf_nodes();
    
    for (level, position) in leaf_nodes {
        if let Some(packet) = tree.get_node(level, position) {
            let packet_energy = compute_energy(&packet.data);
            total_energy += packet_energy;
        }
    }
    
    Ok(total_energy)
}

/// Reconstruction error metrics
struct ReconstructionErrors {
    max_error: f64,
    mean_error: f64,
    snr: f64,
}

/// Validate reconstruction accuracy
fn validate_reconstruction(
    original: &[f64],
    reconstructed: &[f64],
    tolerance: f64,
) -> SignalResult<ReconstructionErrors> {
    if original.len() != reconstructed.len() {
        return Err(SignalError::ShapeMismatch(
            "Original and reconstructed signals have different lengths".to_string(),
        ));
    }
    
    let n = original.len();
    let mut errors = vec![0.0; n];
    
    // Compute errors using SIMD
    let orig_view = ArrayView1::from(original);
    let recon_view = ArrayView1::from(reconstructed);
    let error_view = ArrayView1::from_shape(n, &mut errors).unwrap();
    
    f64::simd_sub(&orig_view, &recon_view, &error_view);
    
    // Compute error metrics
    let max_error = errors.iter().map(|&e| e.abs()).fold(0.0, f64::max);
    let mean_error = errors.iter().map(|&e| e.abs()).sum::<f64>() / n as f64;
    
    // Compute SNR
    let signal_power = compute_energy(original) / n as f64;
    let noise_power = compute_energy(&errors) / n as f64;
    let snr = if noise_power > tolerance * tolerance {
        10.0 * (signal_power / noise_power).log10()
    } else {
        f64::INFINITY
    };
    
    Ok(ReconstructionErrors {
        max_error,
        mean_error,
        snr,
    })
}

/// Validate Parseval frame property
fn validate_parseval_frame(tree: &WaveletPacketTree, signal: &[f64]) -> SignalResult<f64> {
    // For a Parseval frame, sum of squared coefficients equals signal energy
    let signal_energy = compute_energy(signal);
    let coeffs_energy = compute_tree_energy(tree)?;
    
    Ok(coeffs_energy / signal_energy)
}

/// Test numerical stability with edge cases
fn test_numerical_stability(
    signal: &[f64],
    wavelet: Wavelet,
    max_level: usize,
) -> SignalResult<f64> {
    let mut passed_tests = 0;
    let total_tests = 5;
    
    // Test 1: Zero signal
    let zero_signal = vec![0.0; signal.len()];
    match wpt_decompose(&zero_signal, wavelet, max_level, None) {
        Ok(tree) => {
            let reconstructed = wpt_reconstruct(&tree)?;
            if reconstructed.iter().all(|&x| x.abs() < 1e-10) {
                passed_tests += 1;
            }
        }
        Err(_) => {}
    }
    
    // Test 2: Constant signal
    let const_signal = vec![1.0; signal.len()];
    match wpt_decompose(&const_signal, wavelet, max_level, None) {
        Ok(tree) => {
            let reconstructed = wpt_reconstruct(&tree)?;
            let mean = reconstructed.iter().sum::<f64>() / reconstructed.len() as f64;
            if (mean - 1.0).abs() < 1e-10 {
                passed_tests += 1;
            }
        }
        Err(_) => {}
    }
    
    // Test 3: Impulse signal
    let mut impulse = vec![0.0; signal.len()];
    impulse[signal.len() / 2] = 1.0;
    match wpt_decompose(&impulse, wavelet, max_level, None) {
        Ok(tree) => {
            let reconstructed = wpt_reconstruct(&tree)?;
            let energy_preserved = compute_energy(&reconstructed).abs() - 1.0;
            if energy_preserved.abs() < 1e-10 {
                passed_tests += 1;
            }
        }
        Err(_) => {}
    }
    
    // Test 4: Very small values
    let small_signal: Vec<f64> = signal.iter().map(|&x| x * 1e-10).collect();
    match wpt_decompose(&small_signal, wavelet, max_level, None) {
        Ok(tree) => {
            if tree.nodes.values().all(|node| {
                node.data.iter().all(|&x| x.is_finite())
            }) {
                passed_tests += 1;
            }
        }
        Err(_) => {}
    }
    
    // Test 5: Large values
    let large_signal: Vec<f64> = signal.iter().map(|&x| x * 1e10).collect();
    match wpt_decompose(&large_signal, wavelet, max_level, None) {
        Ok(tree) => {
            if tree.nodes.values().all(|node| {
                node.data.iter().all(|&x| x.is_finite())
            }) {
                passed_tests += 1;
            }
        }
        Err(_) => {}
    }
    
    Ok(passed_tests as f64 / total_tests as f64)
}

/// Validate best basis selection
///
/// # Arguments
///
/// * `tree` - Wavelet packet tree
/// * `cost_function` - Cost function name ("shannon", "norm", "threshold")
///
/// # Returns
///
/// * Whether the best basis selection is valid
pub fn validate_best_basis(
    tree: &WaveletPacketTree,
    cost_function: &str,
) -> SignalResult<bool> {
    // Get the selected best basis
    let best_basis = tree.get_best_basis(cost_function)?;
    
    // Verify that selected nodes don't overlap
    let mut covered_samples = HashMap::new();
    
    for (level, position) in &best_basis {
        let node = tree.get_node(*level, *position)
            .ok_or_else(|| SignalError::ValueError(
                format!("Best basis contains invalid node ({}, {})", level, position)
            ))?;
        
        // Calculate sample range covered by this node
        let node_size = node.data.len();
        let start = position * node_size;
        let end = start + node_size;
        
        // Check for overlaps
        for i in start..end {
            if covered_samples.contains_key(&i) {
                return Ok(false); // Overlap detected
            }
            covered_samples.insert(i, (*level, *position));
        }
    }
    
    // Verify complete coverage
    let total_samples = tree.get_node(0, 0)
        .map(|n| n.data.len())
        .unwrap_or(0);
    
    Ok(covered_samples.len() == total_samples)
}

/// Compare WPT with DWT for consistency
pub fn compare_with_dwt(
    signal: &[f64],
    wavelet: Wavelet,
    level: usize,
) -> SignalResult<f64> {
    use crate::dwt::wavedec;
    
    // Perform WPT decomposition
    let wpt_tree = wpt_decompose(signal, wavelet, level, None)?;
    
    // Perform DWT decomposition
    let dwt_coeffs = wavedec(signal, wavelet, level, None)?;
    
    // Compare approximation path in WPT with DWT approximation
    let mut max_diff = 0.0;
    let mut current_pos = 0;
    
    for l in 0..=level {
        if let Some(wpt_node) = wpt_tree.get_node(l, current_pos) {
            // For DWT path, we always take the approximation (left child)
            current_pos *= 2; // Move to left child for next level
            
            // Compare with corresponding DWT coefficients
            // This is a simplified comparison - actual implementation would need
            // proper mapping between WPT and DWT coefficient structures
            let node_energy = compute_energy(&wpt_node.data);
            
            // Check that energy is reasonable
            if node_energy.is_finite() && node_energy >= 0.0 {
                // Energy-based comparison
                let dwt_energy = if l == level {
                    compute_energy(&dwt_coeffs.approx)
                } else {
                    // Would need to extract appropriate detail coefficients
                    node_energy // Placeholder
                };
                
                let diff = (node_energy - dwt_energy).abs() / node_energy.max(1e-10);
                max_diff = max_diff.max(diff);
            }
        }
    }
    
    Ok(max_diff)
}

/// Validate entropy-based cost functions
pub fn validate_entropy_computation(
    tree: &WaveletPacketTree,
) -> SignalResult<bool> {
    let entropy_types = ["shannon", "norm", "threshold"];
    
    for entropy_type in &entropy_types {
        let costs = tree.compute_all_costs(entropy_type)?;
        
        // Verify all costs are non-negative
        for cost in costs.values() {
            if *cost < 0.0 || !cost.is_finite() {
                return Ok(false);
            }
        }
        
        // Verify parent cost >= sum of children costs (for additive costs)
        for ((level, position), _) in &costs {
            if *level < tree.max_level {
                let left_child = (*level + 1, position * 2);
                let right_child = (*level + 1, position * 2 + 1);
                
                if let (Some(&parent_cost), Some(&left_cost), Some(&right_cost)) = 
                    (costs.get(&(*level, *position)), 
                     costs.get(&left_child),
                     costs.get(&right_child)) {
                    
                    // For Shannon entropy, parent cost should generally be >= children sum
                    if entropy_type == "shannon" && parent_cost < left_cost + right_cost - 1e-10 {
                        return Ok(false);
                    }
                }
            }
        }
    }
    
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dwt::Wavelet;
    
    #[test]
    fn test_wpt_validation_basic() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = validate_wpt(&signal, Wavelet::Haar, 2, 1e-10).unwrap();
        
        assert!((result.energy_ratio - 1.0).abs() < 1e-10);
        assert!(result.max_reconstruction_error < 1e-10);
        assert!(result.reconstruction_snr > 100.0);
    }
    
    #[test]
    fn test_parseval_frame() {
        let signal = vec![1.0; 64];
        let tree = wpt_decompose(&signal, Wavelet::DB(4), 3, None).unwrap();
        let ratio = validate_parseval_frame(&tree, &signal).unwrap();
        
        assert!((ratio - 1.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_numerical_stability() {
        let signal = vec![1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0];
        let score = test_numerical_stability(&signal, Wavelet::Haar, 2).unwrap();
        
        assert!(score >= 0.8);
    }
}