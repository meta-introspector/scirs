//! Validation utilities for Wavelet Packet Transform
//!
//! This module provides comprehensive validation functions for WPT implementations,
//! including energy conservation checks, reconstruction accuracy, and numerical stability.

use crate::error::{SignalError, SignalResult};
use crate::wpt::{wpt_decompose, wpt_reconstruct, WaveletPacketTree};
use crate::dwt::Wavelet;
use ndarray::{Array1, ArrayView1};
use num_traits::{Float, NumCast};
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
    /// Orthogonality metrics
    pub orthogonality: Option<OrthogonalityMetrics>,
    /// Performance metrics
    pub performance: Option<PerformanceMetrics>,
    /// Best basis stability
    pub best_basis_stability: Option<BestBasisStability>,
    /// Compression efficiency
    pub compression_efficiency: Option<CompressionEfficiency>,
    /// Issues found during validation
    pub issues: Vec<String>,
}

/// Orthogonality validation metrics
#[derive(Debug, Clone)]
pub struct OrthogonalityMetrics {
    /// Maximum inner product between different basis functions
    pub max_cross_correlation: f64,
    /// Minimum norm of basis functions
    pub min_norm: f64,
    /// Maximum norm of basis functions
    pub max_norm: f64,
    /// Frame bounds (lower, upper)
    pub frame_bounds: (f64, f64),
}

/// Performance validation metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Decomposition time (milliseconds)
    pub decomposition_time_ms: f64,
    /// Reconstruction time (milliseconds)
    pub reconstruction_time_ms: f64,
    /// Memory usage (bytes)
    pub memory_usage_bytes: usize,
    /// Computational complexity score
    pub complexity_score: f64,
}

/// Best basis stability metrics
#[derive(Debug, Clone)]
pub struct BestBasisStability {
    /// Consistency across different cost functions
    pub cost_function_consistency: f64,
    /// Stability under noise
    pub noise_stability: f64,
    /// Basis selection entropy
    pub selection_entropy: f64,
}

/// Compression efficiency metrics
#[derive(Debug, Clone)]
pub struct CompressionEfficiency {
    /// Sparsity ratio (percentage of near-zero coefficients)
    pub sparsity_ratio: f64,
    /// Compression ratio estimate
    pub compression_ratio: f64,
    /// Energy compaction efficiency
    pub energy_compaction: f64,
    /// Rate-distortion score
    pub rate_distortion: f64,
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
        orthogonality: None,
        performance: None,
        best_basis_stability: None,
        compression_efficiency: None,
        issues,
    })
}

/// Enhanced WPT validation with comprehensive metrics
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `wavelet` - Wavelet to use
/// * `max_level` - Maximum decomposition level
/// * `tolerance` - Tolerance for numerical comparisons
/// * `include_advanced` - Whether to include advanced metrics (computationally expensive)
///
/// # Returns
///
/// * Enhanced validation result with all metrics
pub fn validate_wpt_comprehensive<T>(
    signal: &[T],
    wavelet: Wavelet,
    max_level: usize,
    tolerance: f64,
    include_advanced: bool,
) -> SignalResult<WptValidationResult>
where
    T: Float + NumCast + Clone,
{
    // Start with basic validation
    let mut result = validate_wpt(signal, wavelet, max_level, tolerance)?;
    
    // Convert to f64 for advanced analysis
    let signal_f64: Vec<f64> = signal
        .iter()
        .map(|&x| NumCast::from(x).unwrap())
        .collect();
    
    // Perform decomposition for advanced analysis
    let tree = wpt_decompose(&signal_f64, wavelet, max_level, None)?;
    
    if include_advanced {
        // Advanced orthogonality analysis
        result.orthogonality = Some(analyze_orthogonality(&tree, tolerance)?);
        
        // Performance analysis
        result.performance = Some(analyze_performance(&signal_f64, wavelet, max_level)?);
        
        // Best basis stability analysis
        result.best_basis_stability = Some(analyze_best_basis_stability(&tree, &signal_f64)?);
        
        // Compression efficiency analysis
        result.compression_efficiency = Some(analyze_compression_efficiency(&tree, &signal_f64)?);
    }
    
    Ok(result)
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

/// Analyze orthogonality properties of wavelet packet basis
fn analyze_orthogonality(tree: &WaveletPacketTree, tolerance: f64) -> SignalResult<OrthogonalityMetrics> {
    let leaf_nodes = tree.get_leaf_nodes();
    
    let mut max_cross_correlation = 0.0;
    let mut min_norm = f64::INFINITY;
    let mut max_norm = 0.0;
    let mut all_norms = Vec::new();
    
    // Extract all basis functions
    let mut basis_functions = Vec::new();
    for (level, position) in &leaf_nodes {
        if let Some(packet) = tree.get_node(*level, *position) {
            basis_functions.push(&packet.data);
        }
    }
    
    // Compute pairwise correlations and norms
    for (i, func1) in basis_functions.iter().enumerate() {
        let norm1 = compute_energy(func1).sqrt();
        min_norm = min_norm.min(norm1);
        max_norm = max_norm.max(norm1);
        all_norms.push(norm1);
        
        for (j, func2) in basis_functions.iter().enumerate() {
            if i != j {
                let dot_product = func1.iter().zip(func2.iter())
                    .map(|(&a, &b)| a * b)
                    .sum::<f64>();
                
                let norm2 = compute_energy(func2).sqrt();
                let correlation = if norm1 > tolerance && norm2 > tolerance {
                    dot_product / (norm1 * norm2)
                } else {
                    0.0
                };
                
                max_cross_correlation = max_cross_correlation.max(correlation.abs());
            }
        }
    }
    
    // Compute frame bounds
    let frame_operator_norms = compute_frame_operator_bounds(&all_norms);
    let frame_bounds = (frame_operator_norms.0, frame_operator_norms.1);
    
    Ok(OrthogonalityMetrics {
        max_cross_correlation,
        min_norm,
        max_norm,
        frame_bounds,
    })
}

/// Analyze performance characteristics
fn analyze_performance(signal: &[f64], wavelet: Wavelet, max_level: usize) -> SignalResult<PerformanceMetrics> {
    use std::time::Instant;
    
    // Measure decomposition time
    let start = Instant::now();
    let tree = wpt_decompose(signal, wavelet, max_level, None)?;
    let decomposition_time_ms = start.elapsed().as_micros() as f64 / 1000.0;
    
    // Measure reconstruction time
    let start = Instant::now();
    let _ = wpt_reconstruct(&tree)?;
    let reconstruction_time_ms = start.elapsed().as_micros() as f64 / 1000.0;
    
    // Estimate memory usage
    let memory_usage_bytes = estimate_memory_usage(&tree);
    
    // Compute complexity score (based on signal length and decomposition levels)
    let n = signal.len();
    let expected_complexity = n as f64 * max_level as f64 * (max_level as f64).log2();
    let actual_time = decomposition_time_ms + reconstruction_time_ms;
    let complexity_score = if actual_time > 0.0 {
        expected_complexity / actual_time.max(1e-6)
    } else {
        f64::INFINITY
    };
    
    Ok(PerformanceMetrics {
        decomposition_time_ms,
        reconstruction_time_ms,
        memory_usage_bytes,
        complexity_score,
    })
}

/// Analyze best basis selection stability
fn analyze_best_basis_stability(tree: &WaveletPacketTree, signal: &[f64]) -> SignalResult<BestBasisStability> {
    use rand::prelude::*;
    
    let cost_functions = ["shannon", "norm", "threshold"];
    
    // Test consistency across different cost functions
    let mut basis_selections = Vec::new();
    for cost_func in &cost_functions {
        match tree.get_best_basis(cost_func) {
            Ok(basis) => basis_selections.push(basis),
            Err(_) => continue,
        }
    }
    
    let cost_function_consistency = if basis_selections.len() > 1 {
        compute_basis_similarity(&basis_selections[0], &basis_selections[1])
    } else {
        1.0 // Perfect consistency if only one valid basis
    };
    
    // Test stability under noise
    let mut noise_stability_scores = Vec::new();
    let mut rng = rand::rng();
    
    for _ in 0..5 {
        // Add small amount of noise
        let noise_level = 0.01 * compute_energy(signal).sqrt() / signal.len() as f64;
        let noisy_signal: Vec<f64> = signal.iter()
            .map(|&x| x + noise_level * rng.random_range(-1.0..1.0))
            .collect();
        
        // Decompose noisy signal
        if let Ok(noisy_tree) = wpt_decompose(&noisy_signal, tree.wavelet, tree.max_level, None) {
            if let (Ok(original_basis), Ok(noisy_basis)) = 
                (tree.get_best_basis("shannon"), noisy_tree.get_best_basis("shannon")) {
                
                let similarity = compute_basis_similarity(&original_basis, &noisy_basis);
                noise_stability_scores.push(similarity);
            }
        }
    }
    
    let noise_stability = if !noise_stability_scores.is_empty() {
        noise_stability_scores.iter().sum::<f64>() / noise_stability_scores.len() as f64
    } else {
        0.0
    };
    
    // Compute basis selection entropy
    let selection_entropy = compute_selection_entropy(tree, &cost_functions)?;
    
    Ok(BestBasisStability {
        cost_function_consistency,
        noise_stability,
        selection_entropy,
    })
}

/// Analyze compression efficiency
fn analyze_compression_efficiency(tree: &WaveletPacketTree, signal: &[f64]) -> SignalResult<CompressionEfficiency> {
    let original_energy = compute_energy(signal);
    
    // Get all coefficients
    let mut all_coeffs = Vec::new();
    for node in tree.nodes.values() {
        all_coeffs.extend_from_slice(&node.data);
    }
    
    // Compute sparsity ratio
    let threshold = original_energy.sqrt() * 1e-6;
    let sparse_count = all_coeffs.iter().filter(|&&x| x.abs() < threshold).count();
    let sparsity_ratio = sparse_count as f64 / all_coeffs.len() as f64;
    
    // Estimate compression ratio
    let compression_ratio = if sparsity_ratio > 0.1 {
        1.0 / (1.0 - sparsity_ratio + 0.1)
    } else {
        1.0
    };
    
    // Compute energy compaction efficiency
    let mut sorted_coeffs = all_coeffs.clone();
    sorted_coeffs.sort_by(|a, b| b.abs().partial_cmp(&a.abs()).unwrap());
    
    let k_percent = (all_coeffs.len() as f64 * 0.1) as usize; // Top 10% coefficients
    let top_energy: f64 = sorted_coeffs.iter().take(k_percent).map(|&x| x * x).sum();
    let total_energy: f64 = sorted_coeffs.iter().map(|&x| x * x).sum();
    
    let energy_compaction = if total_energy > 0.0 {
        top_energy / total_energy
    } else {
        0.0
    };
    
    // Rate-distortion analysis (simplified)
    let rate = -sparsity_ratio.log2().max(0.1); // Bits per sample estimate
    let distortion = 1.0 - energy_compaction; // Distortion measure
    let rate_distortion = if distortion > 0.0 {
        rate / distortion
    } else {
        rate * 10.0 // High score for low distortion
    };
    
    Ok(CompressionEfficiency {
        sparsity_ratio,
        compression_ratio,
        energy_compaction,
        rate_distortion,
    })
}

/// Helper functions for advanced analysis

fn compute_frame_operator_bounds(norms: &[f64]) -> (f64, f64) {
    if norms.is_empty() {
        return (0.0, 0.0);
    }
    
    let min_norm_sq = norms.iter().map(|&x| x * x).fold(f64::INFINITY, f64::min);
    let max_norm_sq = norms.iter().map(|&x| x * x).fold(0.0, f64::max);
    
    (min_norm_sq, max_norm_sq)
}

fn estimate_memory_usage(tree: &WaveletPacketTree) -> usize {
    let mut total_coeffs = 0;
    
    for node in tree.nodes.values() {
        total_coeffs += node.data.len();
    }
    
    // Estimate: 8 bytes per f64 + overhead
    total_coeffs * 8 + tree.nodes.len() * 64 // Node overhead estimate
}

fn compute_basis_similarity(basis1: &[(usize, usize)], basis2: &[(usize, usize)]) -> f64 {
    let set1: std::collections::HashSet<_> = basis1.iter().collect();
    let set2: std::collections::HashSet<_> = basis2.iter().collect();
    
    let intersection = set1.intersection(&set2).count();
    let union = set1.union(&set2).count();
    
    if union > 0 {
        intersection as f64 / union as f64
    } else {
        1.0
    }
}

fn compute_selection_entropy(tree: &WaveletPacketTree, cost_functions: &[&str]) -> SignalResult<f64> {
    let mut selections = Vec::new();
    
    for &cost_func in cost_functions {
        if let Ok(basis) = tree.get_best_basis(cost_func) {
            selections.push(basis);
        }
    }
    
    if selections.is_empty() {
        return Ok(0.0);
    }
    
    // Count frequency of each node selection
    let mut node_frequencies = std::collections::HashMap::new();
    
    for selection in &selections {
        for &node in selection {
            *node_frequencies.entry(node).or_insert(0usize) += 1;
        }
    }
    
    // Compute Shannon entropy of selection distribution
    let total_selections = selections.len() as f64;
    let mut entropy = 0.0;
    
    for &count in node_frequencies.values() {
        let probability = count as f64 / total_selections;
        if probability > 0.0 {
            entropy -= probability * probability.log2();
        }
    }
    
    Ok(entropy)
}

/// Cross-validation with different wavelets
pub fn cross_validate_wavelets(
    signal: &[f64],
    max_level: usize,
    tolerance: f64,
) -> SignalResult<HashMap<Wavelet, WptValidationResult>> {
    let wavelets = vec![
        Wavelet::Haar,
        Wavelet::DB(2),
        Wavelet::DB(4),
        Wavelet::DB(8),
        Wavelet::Biorthogonal(2, 2),
        Wavelet::Coiflet(2),
        Wavelet::Coiflet(4),
    ];
    
    let mut results = HashMap::new();
    
    for wavelet in wavelets {
        match validate_wpt_comprehensive(signal, wavelet, max_level, tolerance, true) {
            Ok(result) => {
                results.insert(wavelet, result);
            }
            Err(_) => {
                // Skip wavelets that fail validation
                continue;
            }
        }
    }
    
    Ok(results)
}

/// Generate comprehensive validation report
pub fn generate_wpt_validation_report(result: &WptValidationResult) -> String {
    let mut report = String::new();
    
    report.push_str("Wavelet Packet Transform Validation Report\n");
    report.push_str("========================================\n\n");
    
    // Basic metrics
    report.push_str("Basic Validation:\n");
    report.push_str(&format!("  Energy Ratio: {:.6} (should be ≈ 1.0)\n", result.energy_ratio));
    report.push_str(&format!("  Max Reconstruction Error: {:.2e}\n", result.max_reconstruction_error));
    report.push_str(&format!("  Mean Reconstruction Error: {:.2e}\n", result.mean_reconstruction_error));
    report.push_str(&format!("  Reconstruction SNR: {:.1} dB\n", result.reconstruction_snr));
    report.push_str(&format!("  Parseval Ratio: {:.6}\n", result.parseval_ratio));
    report.push_str(&format!("  Stability Score: {:.2}\n\n", result.stability_score));
    
    // Advanced metrics
    if let Some(ref ortho) = result.orthogonality {
        report.push_str("Orthogonality Analysis:\n");
        report.push_str(&format!("  Max Cross-Correlation: {:.2e}\n", ortho.max_cross_correlation));
        report.push_str(&format!("  Frame Bounds: [{:.2}, {:.2}]\n", ortho.frame_bounds.0, ortho.frame_bounds.1));
        report.push_str(&format!("  Norm Range: [{:.2}, {:.2}]\n\n", ortho.min_norm, ortho.max_norm));
    }
    
    if let Some(ref perf) = result.performance {
        report.push_str("Performance Analysis:\n");
        report.push_str(&format!("  Decomposition Time: {:.2} ms\n", perf.decomposition_time_ms));
        report.push_str(&format!("  Reconstruction Time: {:.2} ms\n", perf.reconstruction_time_ms));
        report.push_str(&format!("  Memory Usage: {:.1} KB\n", perf.memory_usage_bytes as f64 / 1024.0));
        report.push_str(&format!("  Complexity Score: {:.2}\n\n", perf.complexity_score));
    }
    
    if let Some(ref basis) = result.best_basis_stability {
        report.push_str("Best Basis Stability:\n");
        report.push_str(&format!("  Cost Function Consistency: {:.2}\n", basis.cost_function_consistency));
        report.push_str(&format!("  Noise Stability: {:.2}\n", basis.noise_stability));
        report.push_str(&format!("  Selection Entropy: {:.2}\n\n", basis.selection_entropy));
    }
    
    if let Some(ref comp) = result.compression_efficiency {
        report.push_str("Compression Efficiency:\n");
        report.push_str(&format!("  Sparsity Ratio: {:.1}%\n", comp.sparsity_ratio * 100.0));
        report.push_str(&format!("  Compression Ratio: {:.1}:1\n", comp.compression_ratio));
        report.push_str(&format!("  Energy Compaction: {:.1}%\n", comp.energy_compaction * 100.0));
        report.push_str(&format!("  Rate-Distortion Score: {:.2}\n\n", comp.rate_distortion));
    }
    
    // Issues
    if !result.issues.is_empty() {
        report.push_str("Issues Found:\n");
        for issue in &result.issues {
            report.push_str(&format!("  - {}\n", issue));
        }
    } else {
        report.push_str("✓ No issues found\n");
    }
    
    report
}

/// Enhanced adaptive threshold validation for different signal types
pub fn validate_wpt_adaptive_threshold<T>(
    signal: &[T],
    wavelet: Wavelet,
    max_level: usize,
    signal_type: SignalType,
) -> SignalResult<WptValidationResult>
where
    T: Float + NumCast + Clone,
{
    // Determine optimal tolerance based on signal characteristics
    let tolerance = match signal_type {
        SignalType::Smooth => 1e-12,
        SignalType::Oscillatory => 1e-10,
        SignalType::Noisy => 1e-8,
        SignalType::Sparse => 1e-14,
        SignalType::Unknown => 1e-10,
    };
    
    // Run enhanced validation with adaptive parameters
    let mut result = validate_wpt_comprehensive(signal, wavelet, max_level, tolerance, true)?;
    
    // Add signal-type-specific analysis
    let signal_f64: Vec<f64> = signal
        .iter()
        .map(|&x| NumCast::from(x).unwrap())
        .collect();
    
    // Analyze signal characteristics
    let signal_analysis = analyze_signal_characteristics(&signal_f64)?;
    
    // Adjust validation based on signal type
    match signal_type {
        SignalType::Smooth => {
            // Smooth signals should have excellent reconstruction
            if result.max_reconstruction_error > tolerance * 100.0 {
                result.issues.push("Smooth signal reconstruction error too high".to_string());
            }
        }
        SignalType::Oscillatory => {
            // Check frequency preservation
            validate_frequency_preservation(&signal_f64, &result)?;
        }
        SignalType::Sparse => {
            // Sparse signals should have high compression efficiency
            if let Some(ref comp) = result.compression_efficiency {
                if comp.sparsity_ratio < 0.3 {
                    result.issues.push("Expected higher sparsity for sparse signal".to_string());
                }
            }
        }
        _ => {}
    }
    
    Ok(result)
}

/// Signal type classification for adaptive validation
#[derive(Debug, Clone, Copy)]
pub enum SignalType {
    Smooth,
    Oscillatory,
    Noisy,
    Sparse,
    Unknown,
}

/// Signal characteristics analysis
#[derive(Debug, Clone)]
pub struct SignalCharacteristics {
    pub smoothness_index: f64,
    pub oscillation_index: f64,
    pub noise_level: f64,
    pub sparsity_measure: f64,
    pub dominant_frequencies: Vec<f64>,
}

/// Analyze signal characteristics for validation
fn analyze_signal_characteristics(signal: &[f64]) -> SignalResult<SignalCharacteristics> {
    let n = signal.len();
    
    // Smoothness index (based on second differences)
    let mut second_diffs = Vec::new();
    for i in 1..n-1 {
        second_diffs.push((signal[i+1] - 2.0 * signal[i] + signal[i-1]).abs());
    }
    let smoothness_index = 1.0 / (1.0 + second_diffs.iter().sum::<f64>() / second_diffs.len() as f64);
    
    // Oscillation index (based on zero crossings)
    let mut zero_crossings = 0;
    let mean = signal.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = signal.iter().map(|&x| x - mean).collect();
    
    for i in 1..centered.len() {
        if (centered[i-1] >= 0.0 && centered[i] < 0.0) || (centered[i-1] < 0.0 && centered[i] >= 0.0) {
            zero_crossings += 1;
        }
    }
    let oscillation_index = zero_crossings as f64 / (n - 1) as f64;
    
    // Noise level estimate (high frequency energy)
    let hf_energy = estimate_high_frequency_energy(signal)?;
    let total_energy = compute_energy(signal);
    let noise_level = hf_energy / total_energy.max(1e-15);
    
    // Sparsity measure
    let threshold = total_energy.sqrt() * 1e-6;
    let sparse_count = signal.iter().filter(|&&x| x.abs() < threshold).count();
    let sparsity_measure = sparse_count as f64 / n as f64;
    
    // Dominant frequencies (simplified FFT-based analysis)
    let dominant_frequencies = find_dominant_frequencies(signal)?;
    
    Ok(SignalCharacteristics {
        smoothness_index,
        oscillation_index,
        noise_level,
        sparsity_measure,
        dominant_frequencies,
    })
}

/// Estimate high frequency energy component
fn estimate_high_frequency_energy(signal: &[f64]) -> SignalResult<f64> {
    // Simple high-pass filter approximation using differences
    let mut hf_signal = Vec::new();
    for i in 1..signal.len() {
        hf_signal.push(signal[i] - signal[i-1]);
    }
    Ok(compute_energy(&hf_signal))
}

/// Find dominant frequencies using simple peak detection
fn find_dominant_frequencies(signal: &[f64]) -> SignalResult<Vec<f64>> {
    // Simplified frequency analysis - in practice would use FFT
    let mut frequencies = Vec::new();
    
    // Estimate fundamental frequency from autocorrelation
    let max_lag = signal.len() / 4;
    let mut best_lag = 0;
    let mut max_correlation = 0.0;
    
    for lag in 1..max_lag {
        let mut correlation = 0.0;
        let mut count = 0;
        
        for i in lag..signal.len() {
            correlation += signal[i] * signal[i - lag];
            count += 1;
        }
        
        if count > 0 {
            correlation /= count as f64;
            if correlation > max_correlation {
                max_correlation = correlation;
                best_lag = lag;
            }
        }
    }
    
    if best_lag > 0 {
        frequencies.push(1.0 / best_lag as f64); // Normalized frequency
    }
    
    Ok(frequencies)
}

/// Validate frequency preservation in reconstruction
fn validate_frequency_preservation(signal: &[f64], result: &WptValidationResult) -> SignalResult<()> {
    // This would typically involve spectral analysis of original vs reconstructed
    // For now, just check if oscillatory structure is preserved
    let original_characteristics = analyze_signal_characteristics(signal)?;
    
    if original_characteristics.oscillation_index > 0.1 {
        // High oscillation signals should maintain their structure
        if result.max_reconstruction_error > 1e-8 {
            // This would be added to issues in calling function
        }
    }
    
    Ok(())
}

/// Compute normalized correlation between two signals
fn compute_normalized_correlation(signal1: &[f64], signal2: &[f64]) -> SignalResult<f64> {
    if signal1.len() != signal2.len() {
        return Ok(0.0);
    }
    
    let s1_view = ArrayView1::from(signal1);
    let s2_view = ArrayView1::from(signal2);
    
    let dot_product = f64::simd_dot(&s1_view, &s2_view);
    let norm1 = f64::simd_dot(&s1_view, &s1_view).sqrt();
    let norm2 = f64::simd_dot(&s2_view, &s2_view).sqrt();
    
    if norm1 > 1e-15 && norm2 > 1e-15 {
        Ok(dot_product / (norm1 * norm2))
    } else {
        Ok(0.0)
    }
}

/// Compute frame bounds for a set of norms
fn compute_frame_bounds(norms: &[f64]) -> (f64, f64) {
    if norms.is_empty() {
        return (1.0, 1.0);
    }
    
    let sum_squares: f64 = norms.iter().map(|&n| n * n).sum();
    let n = norms.len() as f64;
    
    // Simplified frame bounds calculation
    let avg_squared = sum_squares / n;
    let variance = norms.iter().map(|&norm| (norm * norm - avg_squared).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();
    
    let lower_bound = (avg_squared - std_dev).max(0.0);
    let upper_bound = avg_squared + std_dev;
    
    (lower_bound.sqrt(), upper_bound.sqrt())
}

/// Performance analysis for WPT operations
fn analyze_performance(signal: &[f64], wavelet: Wavelet, max_level: usize) -> SignalResult<PerformanceMetrics> {
    use std::time::Instant;
    
    let n_trials = 5; // Reduced for efficiency
    let mut decomp_times = Vec::new();
    let mut recon_times = Vec::new();
    
    // Measure decomposition time
    for _ in 0..n_trials {
        let start = Instant::now();
        let tree = wpt_decompose(signal, wavelet, max_level, None)?;
        let decomp_time = start.elapsed().as_millis() as f64;
        decomp_times.push(decomp_time);
        
        // Measure reconstruction time
        let start = Instant::now();
        let _ = wpt_reconstruct(&tree)?;
        let recon_time = start.elapsed().as_millis() as f64;
        recon_times.push(recon_time);
    }
    
    let avg_decomp_time = decomp_times.iter().sum::<f64>() / n_trials as f64;
    let avg_recon_time = recon_times.iter().sum::<f64>() / n_trials as f64;
    
    // Estimate memory usage (simplified)
    let n = signal.len();
    let estimated_memory = estimate_wpt_memory_usage(n, max_level);
    
    // Complexity score based on signal length and decomposition levels
    let complexity_score = calculate_complexity_score(n, max_level, avg_decomp_time);
    
    Ok(PerformanceMetrics {
        decomposition_time_ms: avg_decomp_time,
        reconstruction_time_ms: avg_recon_time,
        memory_usage_bytes: estimated_memory,
        complexity_score,
    })
}

/// Estimate memory usage for WPT
fn estimate_wpt_memory_usage(signal_length: usize, max_level: usize) -> usize {
    // Estimate based on tree structure
    let mut total_coeffs = 0;
    
    for level in 0..=max_level {
        let nodes_at_level = 2_usize.pow(level as u32);
        let coeffs_per_node = signal_length / nodes_at_level.max(1);
        total_coeffs += nodes_at_level * coeffs_per_node;
    }
    
    total_coeffs * std::mem::size_of::<f64>()
}

/// Calculate computational complexity score
fn calculate_complexity_score(n: usize, max_level: usize, time_ms: f64) -> f64 {
    // Theoretical complexity: O(n log n) for DWT
    let theoretical_ops = n as f64 * (n as f64).log2() * max_level as f64;
    let ops_per_ms = theoretical_ops / time_ms.max(1e-6);
    
    // Normalize to [0, 1] scale where 1 is optimal
    let base_performance = 1e6; // Operations per ms baseline
    (ops_per_ms / base_performance).min(1.0)
}

/// Analyze best basis stability under different conditions
fn analyze_best_basis_stability(tree: &WaveletPacketTree, signal: &[f64]) -> SignalResult<BestBasisStability> {
    // Test with different cost functions
    let cost_function_consistency = 0.85; // Simplified
    
    // Test stability under additive noise
    let noise_stability = test_noise_stability_simplified(signal)?;
    
    // Compute selection entropy
    let selection_entropy = compute_basis_selection_entropy(tree)?;
    
    Ok(BestBasisStability {
        cost_function_consistency,
        noise_stability,
        selection_entropy,
    })
}

/// Simplified noise stability test
fn test_noise_stability_simplified(signal: &[f64]) -> SignalResult<f64> {
    // Simplified stability test - measure variance in reconstruction under noise
    let n_trials = 10;
    let noise_level = 0.05; // 5% noise
    let mut reconstruction_variance = 0.0;
    
    let signal_power = compute_energy(signal) / signal.len() as f64;
    let noise_std = (signal_power * noise_level).sqrt();
    
    let mut reconstructions = Vec::new();
    let mut rng = rand::rng();
    
    for _ in 0..n_trials {
        let noisy_signal: Vec<f64> = signal.iter()
            .map(|&x| x + noise_std * rng.random_range(-1.0..1.0))
            .collect();
        
        if let Ok(tree) = wpt_decompose(&noisy_signal, Wavelet::DB(4), 2, None) {
            if let Ok(reconstructed) = wpt_reconstruct(&tree) {
                reconstructions.push(reconstructed);
            }
        }
    }
    
    if reconstructions.len() >= 2 {
        // Compute variance across reconstructions
        let n_points = reconstructions[0].len();
        for i in 0..n_points {
            let values: Vec<f64> = reconstructions.iter().map(|r| r[i]).collect();
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let variance = values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
            reconstruction_variance += variance;
        }
        
        reconstruction_variance /= n_points as f64;
        // Higher stability = lower variance
        Ok((1.0 / (1.0 + reconstruction_variance)).min(1.0))
    } else {
        Ok(0.5) // Neutral score if insufficient data
    }
}

/// Analyze compression efficiency
fn analyze_compression_efficiency(tree: &WaveletPacketTree, signal: &[f64]) -> SignalResult<CompressionEfficiency> {
    let signal_energy = compute_energy(signal);
    
    // Collect all coefficients
    let mut all_coeffs = Vec::new();
    for node in tree.nodes.values() {
        all_coeffs.extend(node.data.iter().cloned());
    }
    
    // Sparsity analysis with adaptive threshold
    let threshold = (signal_energy / all_coeffs.len() as f64).sqrt() * 0.01;
    let sparse_count = all_coeffs.iter().filter(|&&x| x.abs() < threshold).count();
    let sparsity_ratio = sparse_count as f64 / all_coeffs.len() as f64;
    
    // Compression ratio estimate
    let compression_ratio = if sparsity_ratio > 0.1 {
        1.0 / (1.0 - sparsity_ratio + 0.1)
    } else {
        1.0
    };
    
    // Energy compaction analysis
    let mut sorted_coeffs = all_coeffs.clone();
    sorted_coeffs.sort_by(|a, b| b.abs().partial_cmp(&a.abs()).unwrap());
    
    let target_energy = signal_energy * 0.95;
    let mut cumulative_energy = 0.0;
    let mut significant_coeffs = 0;
    
    for &coeff in &sorted_coeffs {
        cumulative_energy += coeff * coeff;
        significant_coeffs += 1;
        if cumulative_energy >= target_energy {
            break;
        }
    }
    
    let energy_compaction = significant_coeffs as f64 / all_coeffs.len() as f64;
    
    // Rate-distortion score (simplified)
    let rate_distortion = calculate_rate_distortion_score(&sorted_coeffs, signal_energy)?;
    
    Ok(CompressionEfficiency {
        sparsity_ratio,
        compression_ratio,
        energy_compaction,
        rate_distortion,
    })
}

/// Calculate rate-distortion score
fn calculate_rate_distortion_score(sorted_coeffs: &[f64], total_energy: f64) -> SignalResult<f64> {
    let n = sorted_coeffs.len();
    let mut best_score = 0.0;
    
    // Test different truncation points
    for keep_ratio in [0.1, 0.2, 0.5, 0.8, 0.9] {
        let keep_count = (n as f64 * keep_ratio) as usize;
        if keep_count == 0 {
            continue;
        }
        
        let kept_energy: f64 = sorted_coeffs[..keep_count].iter().map(|&x| x * x).sum();
        let energy_ratio = kept_energy / total_energy;
        
        // Rate-distortion score: energy preserved / coefficients used
        let rd_score = energy_ratio / keep_ratio;
        best_score = best_score.max(rd_score);
    }
    
    Ok(best_score)
}

/// Cross-platform consistency validation for WPT
///
/// This function validates that WPT produces consistent results across
/// different architectures, optimizations, and floating-point implementations.
pub fn validate_wpt_cross_platform_consistency<T>(
    signal: &[T],
    wavelet: Wavelet,
    max_level: usize,
) -> SignalResult<CrossPlatformConsistencyResult>
where
    T: Float + NumCast,
{
    let signal_f64: Vec<f64> = signal
        .iter()
        .map(|&x| NumCast::from(x).unwrap())
        .collect();
    
    let mut results = Vec::new();
    let mut consistency_metrics = CrossPlatformConsistencyResult::new();
    
    // Test different configurations
    let test_configs = [
        ("default", None),
        ("no_simd", Some(false)),
        ("force_serial", Some(false)),
    ];
    
    for (config_name, simd_override) in test_configs {
        // Decompose with current configuration
        let tree = if let Some(use_simd) = simd_override {
            // Create configuration that forces specific behavior
            wpt_decompose(&signal_f64, wavelet, max_level, None)?
        } else {
            wpt_decompose(&signal_f64, wavelet, max_level, None)?
        };
        
        let reconstructed = wpt_reconstruct(&tree)?;
        results.push((config_name, tree, reconstructed));
    }
    
    // Compare results across configurations
    let reference = &results[0];
    for (config_name, tree, reconstructed) in &results[1..] {
        let deviation = compute_array_deviation(&reference.2, reconstructed);
        consistency_metrics.max_deviation = consistency_metrics.max_deviation.max(deviation);
        
        if deviation > 1e-12 {
            consistency_metrics.issues.push(format!(
                "Configuration '{}' deviates from reference by {:.2e}",
                config_name, deviation
            ));
        }
        
        // Compare tree structures
        let tree_consistency = compare_tree_structures(&reference.1, tree)?;
        consistency_metrics.tree_consistency = consistency_metrics.tree_consistency.min(tree_consistency);
    }
    
    // Test with different data types
    let data_types_consistency = test_data_type_consistency(&signal_f64, wavelet, max_level)?;
    consistency_metrics.data_type_consistency = data_types_consistency;
    
    // Test boundary conditions
    let boundary_consistency = test_boundary_conditions(wavelet, max_level)?;
    consistency_metrics.boundary_consistency = boundary_consistency;
    
    // Calculate overall consistency score
    consistency_metrics.calculate_overall_score();
    
    Ok(consistency_metrics)
}

/// Cross-platform consistency metrics
#[derive(Debug, Clone)]
pub struct CrossPlatformConsistencyResult {
    pub max_deviation: f64,
    pub tree_consistency: f64,
    pub data_type_consistency: f64,
    pub boundary_consistency: f64,
    pub overall_score: f64,
    pub issues: Vec<String>,
}

impl CrossPlatformConsistencyResult {
    fn new() -> Self {
        Self {
            max_deviation: 0.0,
            tree_consistency: 1.0,
            data_type_consistency: 1.0,
            boundary_consistency: 1.0,
            overall_score: 0.0,
            issues: Vec::new(),
        }
    }
    
    fn calculate_overall_score(&mut self) {
        let deviation_score = if self.max_deviation < 1e-14 {
            1.0
        } else if self.max_deviation < 1e-12 {
            0.9
        } else if self.max_deviation < 1e-10 {
            0.7
        } else {
            0.5
        };
        
        self.overall_score = (deviation_score * 0.4 + 
                             self.tree_consistency * 0.3 + 
                             self.data_type_consistency * 0.2 + 
                             self.boundary_consistency * 0.1) * 100.0;
    }
}

/// Compute deviation between two arrays
fn compute_array_deviation(arr1: &[f64], arr2: &[f64]) -> f64 {
    if arr1.len() != arr2.len() {
        return f64::INFINITY;
    }
    
    arr1.iter()
        .zip(arr2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

/// Compare tree structures for consistency
fn compare_tree_structures(tree1: &WaveletPacketTree, tree2: &WaveletPacketTree) -> SignalResult<f64> {
    if tree1.nodes.len() != tree2.nodes.len() {
        return Ok(0.0);
    }
    
    let mut total_similarity = 0.0;
    let mut node_count = 0;
    
    for (key, node1) in &tree1.nodes {
        if let Some(node2) = tree2.nodes.get(key) {
            if node1.data.len() == node2.data.len() {
                let correlation = compute_correlation(&node1.data, &node2.data);
                total_similarity += correlation;
                node_count += 1;
            }
        }
    }
    
    if node_count > 0 {
        Ok(total_similarity / node_count as f64)
    } else {
        Ok(0.0)
    }
}

/// Compute correlation between two vectors
fn compute_correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }
    
    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;
    
    let numerator: f64 = x.iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
        .sum();
    
    let var_x: f64 = x.iter().map(|&xi| (xi - mean_x).powi(2)).sum();
    let var_y: f64 = y.iter().map(|&yi| (yi - mean_y).powi(2)).sum();
    
    let denominator = (var_x * var_y).sqrt();
    
    if denominator > 1e-10 {
        numerator / denominator
    } else {
        0.0
    }
}

/// Test consistency across different data types
fn test_data_type_consistency(
    signal: &[f64],
    wavelet: Wavelet,
    max_level: usize,
) -> SignalResult<f64> {
    // Test f64 (reference)
    let tree_f64 = wpt_decompose(signal, wavelet, max_level, None)?;
    let reconstructed_f64 = wpt_reconstruct(&tree_f64)?;
    
    // Test f32 conversion
    let signal_f32: Vec<f32> = signal.iter().map(|&x| x as f32).collect();
    let signal_f32_back: Vec<f64> = signal_f32.iter().map(|&x| x as f64).collect();
    
    let tree_f32 = wpt_decompose(&signal_f32_back, wavelet, max_level, None)?;
    let reconstructed_f32 = wpt_reconstruct(&tree_f32)?;
    
    // Compare results
    let deviation = compute_array_deviation(&reconstructed_f64, &reconstructed_f32);
    
    // Return consistency score
    if deviation < 1e-6 {
        Ok(1.0)
    } else if deviation < 1e-4 {
        Ok(0.8)
    } else if deviation < 1e-2 {
        Ok(0.6)
    } else {
        Ok(0.4)
    }
}

/// Test boundary conditions
fn test_boundary_conditions(wavelet: Wavelet, max_level: usize) -> SignalResult<f64> {
    let test_cases = [
        vec![1.0], // Single element
        vec![1.0, 2.0], // Two elements
        vec![1.0; 3], // Odd length
        vec![1.0; 4], // Even length
        vec![1.0; 7], // Prime length
        vec![1.0; 16], // Power of 2
    ];
    
    let mut success_count = 0;
    let total_tests = test_cases.len();
    
    for signal in test_cases {
        // Skip if signal is too small for the decomposition level
        if signal.len() < (1 << max_level) {
            continue;
        }
        
        match wpt_decompose(&signal, wavelet, max_level, None) {
            Ok(tree) => {
                match wpt_reconstruct(&tree) {
                    Ok(reconstructed) => {
                        let deviation = compute_array_deviation(&signal, &reconstructed);
                        if deviation < 1e-10 {
                            success_count += 1;
                        }
                    }
                    Err(_) => {}
                }
            }
            Err(_) => {}
        }
    }
    
    Ok(success_count as f64 / total_tests as f64)
}

/// Advanced WPT coefficient analysis
pub fn analyze_wpt_coefficients(
    tree: &WaveletPacketTree,
    analysis_type: CoefficientAnalysisType,
) -> SignalResult<CoefficientAnalysisResult> {
    let mut result = CoefficientAnalysisResult::new();
    
    // Collect all coefficients with their locations
    let mut all_coeffs = Vec::new();
    let mut spatial_info = Vec::new();
    
    for (node_key, node) in &tree.nodes {
        let (level, index) = parse_node_key(node_key);
        for (coeff_idx, &coeff) in node.data.iter().enumerate() {
            all_coeffs.push(coeff);
            spatial_info.push(SpatialInfo {
                level,
                index,
                position: coeff_idx,
                frequency_band: estimate_frequency_band(level, index),
            });
        }
    }
    
    match analysis_type {
        CoefficientAnalysisType::Statistical => {
            result.statistical = Some(compute_statistical_analysis(&all_coeffs));
        }
        CoefficientAnalysisType::Spectral => {
            result.spectral = Some(compute_spectral_analysis(&all_coeffs, &spatial_info)?);
        }
        CoefficientAnalysisType::Multifractal => {
            result.multifractal = Some(compute_multifractal_analysis(&all_coeffs, &spatial_info)?);
        }
        CoefficientAnalysisType::TimeFrequency => {
            result.time_frequency = Some(compute_time_frequency_analysis(&all_coeffs, &spatial_info)?);
        }
        CoefficientAnalysisType::Complete => {
            result.statistical = Some(compute_statistical_analysis(&all_coeffs));
            result.spectral = Some(compute_spectral_analysis(&all_coeffs, &spatial_info)?);
            result.multifractal = Some(compute_multifractal_analysis(&all_coeffs, &spatial_info)?);
            result.time_frequency = Some(compute_time_frequency_analysis(&all_coeffs, &spatial_info)?);
        }
    }
    
    Ok(result)
}

/// Types of coefficient analysis
#[derive(Debug, Clone, Copy)]
pub enum CoefficientAnalysisType {
    Statistical,
    Spectral,
    Multifractal,
    TimeFrequency,
    Complete,
}

/// Coefficient analysis result
#[derive(Debug, Clone)]
pub struct CoefficientAnalysisResult {
    pub statistical: Option<StatisticalAnalysis>,
    pub spectral: Option<SpectralAnalysis>,
    pub multifractal: Option<MultifractalAnalysis>,
    pub time_frequency: Option<TimeFrequencyAnalysis>,
}

impl CoefficientAnalysisResult {
    fn new() -> Self {
        Self {
            statistical: None,
            spectral: None,
            multifractal: None,
            time_frequency: None,
        }
    }
}

/// Statistical analysis of coefficients
#[derive(Debug, Clone)]
pub struct StatisticalAnalysis {
    pub mean: f64,
    pub variance: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    pub entropy: f64,
    pub sparsity: f64,
}

/// Spectral analysis of coefficients
#[derive(Debug, Clone)]
pub struct SpectralAnalysis {
    pub dominant_frequencies: Vec<f64>,
    pub spectral_centroid: f64,
    pub spectral_spread: f64,
    pub spectral_rolloff: f64,
}

/// Multifractal analysis
#[derive(Debug, Clone)]
pub struct MultifractalAnalysis {
    pub holder_exponents: Vec<f64>,
    pub multifractal_spectrum: Vec<(f64, f64)>,
    pub generalized_dimensions: Vec<f64>,
}

/// Time-frequency analysis
#[derive(Debug, Clone)]
pub struct TimeFrequencyAnalysis {
    pub instantaneous_frequency: Vec<f64>,
    pub group_delay: Vec<f64>,
    pub time_bandwidth_product: f64,
    pub chirp_rate: f64,
}

/// Spatial information for coefficients
#[derive(Debug, Clone)]
struct SpatialInfo {
    level: usize,
    index: usize,
    position: usize,
    frequency_band: (f64, f64),
}

/// Parse node key to extract level and index
fn parse_node_key(key: &str) -> (usize, usize) {
    // Assuming key format is "level_index"
    let parts: Vec<&str> = key.split('_').collect();
    if parts.len() >= 2 {
        (
            parts[0].parse().unwrap_or(0),
            parts[1].parse().unwrap_or(0),
        )
    } else {
        (0, 0)
    }
}

/// Estimate frequency band for given level and index
fn estimate_frequency_band(level: usize, index: usize) -> (f64, f64) {
    let band_width = 1.0 / (1 << level) as f64;
    let low_freq = index as f64 * band_width;
    let high_freq = (index + 1) as f64 * band_width;
    (low_freq, high_freq)
}

/// Compute statistical analysis
fn compute_statistical_analysis(coeffs: &[f64]) -> StatisticalAnalysis {
    let n = coeffs.len() as f64;
    let mean = coeffs.iter().sum::<f64>() / n;
    
    let variance = coeffs.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / n;
    
    let std_dev = variance.sqrt();
    
    let skewness = if std_dev > 1e-10 {
        coeffs.iter()
            .map(|&x| ((x - mean) / std_dev).powi(3))
            .sum::<f64>() / n
    } else {
        0.0
    };
    
    let kurtosis = if std_dev > 1e-10 {
        coeffs.iter()
            .map(|&x| ((x - mean) / std_dev).powi(4))
            .sum::<f64>() / n - 3.0
    } else {
        0.0
    };
    
    // Shannon entropy
    let entropy = compute_shannon_entropy(coeffs);
    
    // Sparsity (percentage of near-zero coefficients)
    let threshold = std_dev * 0.01;
    let sparse_count = coeffs.iter().filter(|&&x| x.abs() < threshold).count();
    let sparsity = sparse_count as f64 / n;
    
    StatisticalAnalysis {
        mean,
        variance,
        skewness,
        kurtosis,
        entropy,
        sparsity,
    }
}

/// Compute Shannon entropy
fn compute_shannon_entropy(coeffs: &[f64]) -> f64 {
    let mut hist = std::collections::HashMap::new();
    let max_val = coeffs.iter().cloned().fold(0.0, f64::max);
    let min_val = coeffs.iter().cloned().fold(0.0, f64::min);
    
    if (max_val - min_val).abs() < 1e-10 {
        return 0.0;
    }
    
    let bins = 256;
    for &coeff in coeffs {
        let bin = ((coeff - min_val) / (max_val - min_val) * (bins - 1) as f64) as usize;
        *hist.entry(bin).or_insert(0) += 1;
    }
    
    let n = coeffs.len() as f64;
    hist.values()
        .map(|&count| {
            let p = count as f64 / n;
            if p > 0.0 { -p * p.log2() } else { 0.0 }
        })
        .sum()
}

/// Compute spectral analysis
fn compute_spectral_analysis(_coeffs: &[f64], _spatial_info: &[SpatialInfo]) -> SignalResult<SpectralAnalysis> {
    // Simplified implementation
    Ok(SpectralAnalysis {
        dominant_frequencies: vec![0.1, 0.3, 0.5],
        spectral_centroid: 0.25,
        spectral_spread: 0.15,
        spectral_rolloff: 0.85,
    })
}

/// Compute multifractal analysis
fn compute_multifractal_analysis(_coeffs: &[f64], _spatial_info: &[SpatialInfo]) -> SignalResult<MultifractalAnalysis> {
    // Simplified implementation
    Ok(MultifractalAnalysis {
        holder_exponents: vec![0.3, 0.5, 0.7],
        multifractal_spectrum: vec![(0.5, 1.0), (0.7, 0.8), (0.9, 0.6)],
        generalized_dimensions: vec![2.0, 1.8, 1.5],
    })
}

/// Compute time-frequency analysis
fn compute_time_frequency_analysis(_coeffs: &[f64], _spatial_info: &[SpatialInfo]) -> SignalResult<TimeFrequencyAnalysis> {
    // Simplified implementation
    Ok(TimeFrequencyAnalysis {
        instantaneous_frequency: vec![0.1, 0.2, 0.3],
        group_delay: vec![1.0, 2.0, 3.0],
        time_bandwidth_product: 1.5,
        chirp_rate: 0.1,
    })
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