//! Extended property-based tests for advanced statistical operations
//!
//! This module provides comprehensive testing for SIMD optimizations,
//! parallel processing, memory optimizations, and mathematical invariants.

use crate::{
    // SIMD functions
    moments_simd::{skewness_simd, kurtosis_simd, moment_simd, moments_batch_simd},
    descriptive_simd::{mean_simd, variance_simd},
    correlation_parallel_enhanced::{
        corrcoef_parallel_enhanced, pearson_r_simd_enhanced, 
        batch_correlations_parallel, ParallelCorrelationConfig,
    },
    memory_optimized_advanced::{
        corrcoef_memory_aware, AdaptiveMemoryManager as AdvancedMemoryManager, MemoryConstraints,
    },
    // Standard functions for comparison
    descriptive::{skew, kurtosis, moment, mean, var},
    corrcoef, pearson_r,
};
use ndarray::{Array1, Array2, ArrayView1};

/// Test data generator for property-based tests
#[derive(Clone, Debug)]
pub struct StatisticalTestData {
    pub data: Vec<f64>,
}

impl StatisticalTestData {
    pub fn new(data: Vec<f64>) -> Self {
        Self { data }
    }
    
    pub fn generate_sample() -> Self {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        Self { data }
    }
    
    pub fn generate_large_sample() -> Self {
        let data: Vec<f64> = (1..=1000).map(|i| i as f64).collect();
        Self { data }
    }
}

/// Matrix test data generator
#[derive(Clone, Debug)]
pub struct MatrixTestData {
    pub data: Vec<Vec<f64>>,
    pub rows: usize,
    pub cols: usize,
}

impl MatrixTestData {
    pub fn new(data: Vec<Vec<f64>>) -> Self {
        let rows = data.len();
        let cols = if rows > 0 { data[0].len() } else { 0 };
        Self { data, rows, cols }
    }
    
    pub fn generate_sample() -> Self {
        let data = vec![
            vec![1.0, 5.0, 10.0],
            vec![2.0, 4.0, 9.0],
            vec![3.0, 3.0, 8.0],
            vec![4.0, 2.0, 7.0],
            vec![5.0, 1.0, 6.0],
        ];
        Self::new(data)
    }
}

/// Property-based test framework for SIMD consistency
pub struct SimdConsistencyTester;

impl SimdConsistencyTester {
    /// Test that SIMD and scalar implementations produce identical results
    pub fn test_mean_consistency(test_data: &StatisticalTestData) -> bool {
        if test_data.data.len() < 1 {
            return false;
        }
        
        let arr = Array1::from_vec(test_data.data.clone());
        
        match (mean(&arr.view()), mean_simd(&arr.view())) {
            (Ok(scalar_result), Ok(simd_result)) => {
                let relative_error = ((scalar_result - simd_result) / scalar_result.max(1e-10)).abs();
                relative_error < 1e-12
            }
            _ => false,
        }
    }
    
    pub fn test_variance_consistency(test_data: &StatisticalTestData) -> bool {
        if test_data.data.len() < 2 {
            return false;
        }
        
        let arr = Array1::from_vec(test_data.data.clone());
        
        match (var(&arr.view(), 1), variance_simd(&arr.view(), 1)) {
            (Ok(scalar_result), Ok(simd_result)) => {
                let relative_error = ((scalar_result - simd_result) / scalar_result.max(1e-10)).abs();
                relative_error < 1e-12
            }
            _ => false,
        }
    }
    
    pub fn test_skewness_consistency(test_data: &StatisticalTestData) -> bool {
        if test_data.data.len() < 3 {
            return false;
        }
        
        let arr = Array1::from_vec(test_data.data.clone());
        
        match (skew(&arr.view(), false), skewness_simd(&arr.view(), false)) {
            (Ok(scalar_result), Ok(simd_result)) => {
                let relative_error = ((scalar_result - simd_result) / scalar_result.abs().max(1e-10)).abs();
                relative_error < 1e-12
            }
            _ => false,
        }
    }
    
    pub fn test_kurtosis_consistency(test_data: &StatisticalTestData) -> bool {
        if test_data.data.len() < 4 {
            return false;
        }
        
        let arr = Array1::from_vec(test_data.data.clone());
        
        match (kurtosis(&arr.view(), true, false), kurtosis_simd(&arr.view(), true, false)) {
            (Ok(scalar_result), Ok(simd_result)) => {
                let relative_error = ((scalar_result - simd_result) / scalar_result.abs().max(1e-10)).abs();
                relative_error < 1e-12
            }
            _ => false,
        }
    }
}

/// Property-based test framework for parallel processing consistency
pub struct ParallelConsistencyTester;

impl ParallelConsistencyTester {
    pub fn test_correlation_matrix_consistency(matrix_data: &MatrixTestData) -> bool {
        if matrix_data.rows < 3 || matrix_data.cols < 2 {
            return false;
        }
        
        // Convert to ndarray
        let mut data = Array2::zeros((matrix_data.rows, matrix_data.cols));
        for (i, row) in matrix_data.data.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                data[[i, j]] = val;
            }
        }
        
        let config = ParallelCorrelationConfig::default();
        
        match (
            corrcoef(&data.view(), "pearson"),
            corrcoef_parallel_enhanced(&data.view(), "pearson", &config)
        ) {
            (Ok(sequential_result), Ok(parallel_result)) => {
                let mut max_error = 0.0;
                
                for i in 0..matrix_data.cols {
                    for j in 0..matrix_data.cols {
                        let error = (sequential_result[[i, j]] - parallel_result[[i, j]]).abs();
                        max_error = max_error.max(error);
                    }
                }
                
                max_error < 1e-12
            }
            _ => false,
        }
    }
}

/// Mathematical invariant tests
pub struct MathematicalInvariantTester;

impl MathematicalInvariantTester {
    /// Test that correlation coefficients are bounded [-1, 1]
    pub fn test_correlation_bounds(test_data1: &StatisticalTestData, test_data2: &StatisticalTestData) -> bool {
        if test_data1.data.len() != test_data2.data.len() || test_data1.data.len() < 2 {
            return false;
        }
        
        let arr1 = Array1::from_vec(test_data1.data.clone());
        let arr2 = Array1::from_vec(test_data2.data.clone());
        
        match pearson_r_simd_enhanced(&arr1.view(), &arr2.view()) {
            Ok(correlation) => correlation >= -1.0 && correlation <= 1.0 && correlation.is_finite(),
            Err(_) => false,
        }
    }
    
    /// Test that variance is non-negative
    pub fn test_variance_non_negative(test_data: &StatisticalTestData) -> bool {
        if test_data.data.len() < 2 {
            return false;
        }
        
        let arr = Array1::from_vec(test_data.data.clone());
        
        match variance_simd(&arr.view(), 1) {
            Ok(variance) => variance >= 0.0 && variance.is_finite(),
            Err(_) => false,
        }
    }
    
    /// Test mean bounds property (min <= mean <= max)
    pub fn test_mean_bounds(test_data: &StatisticalTestData) -> bool {
        if test_data.data.is_empty() {
            return false;
        }
        
        let min_val = test_data.data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = test_data.data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        let arr = Array1::from_vec(test_data.data.clone());
        
        match mean_simd(&arr.view()) {
            Ok(mean_val) => {
                mean_val >= min_val && mean_val <= max_val && mean_val.is_finite()
            },
            Err(_) => false,
        }
    }
    
    /// Test that perfect linear relationship gives correlation ±1
    pub fn test_perfect_correlation() -> bool {
        let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_positive = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // y = 2x
        let y_negative = vec![10.0, 8.0, 6.0, 4.0, 2.0]; // y = -2x + 12
        
        let x = Array1::from_vec(x_data);
        let y_pos = Array1::from_vec(y_positive);
        let y_neg = Array1::from_vec(y_negative);
        
        match (
            pearson_r_simd_enhanced(&x.view(), &y_pos.view()),
            pearson_r_simd_enhanced(&x.view(), &y_neg.view())
        ) {
            (Ok(corr_pos), Ok(corr_neg)) => {
                (corr_pos - 1.0).abs() < 1e-12 && (corr_neg - (-1.0)).abs() < 1e-12
            }
            _ => false,
        }
    }
}

/// Memory optimization property tests
pub struct MemoryOptimizationTester;

impl MemoryOptimizationTester {
    pub fn test_memory_aware_correlation_consistency(matrix_data: &MatrixTestData) -> bool {
        if matrix_data.rows < 3 || matrix_data.cols < 2 {
            return false;
        }
        
        // Convert to ndarray
        let mut data = Array2::zeros((matrix_data.rows, matrix_data.cols));
        for (i, row) in matrix_data.data.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                data[[i, j]] = val;
            }
        }
        
        let constraints = MemoryConstraints::default();
        let mut manager = AdvancedMemoryManager::new(constraints);
        
        match (
            corrcoef(&data.view(), "pearson"),
            corrcoef_memory_aware(&data.view(), "pearson", &mut manager)
        ) {
            (Ok(standard_result), Ok(memory_aware_result)) => {
                let mut max_error = 0.0;
                
                for i in 0..matrix_data.cols {
                    for j in 0..matrix_data.cols {
                        let error = (standard_result[[i, j]] - memory_aware_result[[i, j]]).abs();
                        max_error = max_error.max(error);
                    }
                }
                
                max_error < 1e-12
            }
            _ => false,
        }
    }
}

/// Batch processing tests
pub struct BatchProcessingTester;

impl BatchProcessingTester {
    pub fn test_batch_moments_consistency(test_data: &StatisticalTestData) -> bool {
        if test_data.data.len() < 5 {
            return false;
        }
        
        let arr = Array1::from_vec(test_data.data.clone());
        let orders = vec![1, 2, 3, 4];
        
        match moments_batch_simd(&arr.view(), &orders, true) {
            Ok(batch_results) => {
                // Test individual moments against batch results
                for (i, &order) in orders.iter().enumerate() {
                    if let Ok(individual_result) = moment_simd(&arr.view(), order, true) {
                        let error = (batch_results[i] - individual_result).abs();
                        if error >= 1e-12 {
                            return false;
                        }
                    } else {
                        return false;
                    }
                }
                true
            }
            Err(_) => false,
        }
    }
}

/// Numerical stability tests
pub struct NumericalStabilityTester;

impl NumericalStabilityTester {
    /// Test numerical stability with very small numbers
    pub fn test_small_numbers_stability() -> bool {
        let small_data = vec![1e-10, 2e-10, 3e-10, 4e-10, 5e-10];
        let arr = Array1::from_vec(small_data);
        
        // Test that operations don't catastrophically lose precision
        match (mean_simd(&arr.view()), variance_simd(&arr.view(), 1)) {
            (Ok(mean_val), Ok(var_val)) => {
                mean_val.is_finite() && var_val.is_finite() && var_val >= 0.0
            }
            _ => false,
        }
    }
    
    /// Test numerical stability with very large numbers
    pub fn test_large_numbers_stability() -> bool {
        let large_data = vec![1e10, 2e10, 3e10, 4e10, 5e10];
        let arr = Array1::from_vec(large_data);
        
        match (mean_simd(&arr.view()), variance_simd(&arr.view(), 1)) {
            (Ok(mean_val), Ok(var_val)) => {
                mean_val.is_finite() && var_val.is_finite() && var_val >= 0.0
            }
            _ => false,
        }
    }
    
    /// Test with numbers spanning many orders of magnitude
    pub fn test_mixed_scale_stability() -> bool {
        let mixed_data = vec![1e-5, 1.0, 1e5, 2e-5, 2.0, 2e5];
        let arr = Array1::from_vec(mixed_data);
        
        match variance_simd(&arr.view(), 1) {
            Ok(var_val) => var_val.is_finite() && var_val >= 0.0,
            _ => false,
        }
    }
}

/// Edge case testing
pub struct EdgeCaseTester;

impl EdgeCaseTester {
    /// Test with identical values (zero variance case)
    pub fn test_identical_values() -> bool {
        let identical_data = vec![5.0; 10];
        let arr = Array1::from_vec(identical_data);
        
        match (mean_simd(&arr.view()), variance_simd(&arr.view(), 1)) {
            (Ok(mean_val), Ok(var_val)) => {
                (mean_val - 5.0).abs() < 1e-12 && var_val.abs() < 1e-12
            }
            _ => false,
        }
    }
    
    /// Test with alternating positive/negative values
    pub fn test_alternating_signs() -> bool {
        let alternating_data = vec![-1.0, 1.0, -2.0, 2.0, -3.0, 3.0];
        let arr = Array1::from_vec(alternating_data);
        
        match mean_simd(&arr.view()) {
            Ok(mean_val) => mean_val.abs() < 1e-12, // Should be near zero
            _ => false,
        }
    }
    
    /// Test with single element array
    pub fn test_single_element() -> bool {
        let single_data = vec![42.0];
        let arr = Array1::from_vec(single_data);
        
        match mean_simd(&arr.view()) {
            Ok(mean_val) => (mean_val - 42.0).abs() < 1e-12,
            _ => false,
        }
    }
}

/// Performance consistency tests
pub struct PerformanceTester;

impl PerformanceTester {
    /// Test that SIMD and scalar versions have consistent performance characteristics
    pub fn test_performance_scaling(sizes: &[usize]) -> bool {
        for &size in sizes {
            let data: Vec<f64> = (0..size).map(|i| i as f64).collect();
            let arr = Array1::from_vec(data);
            
            // Time SIMD version
            let start = std::time::Instant::now();
            let _ = mean_simd(&arr.view());
            let simd_time = start.elapsed();
            
            // Time standard version  
            let start = std::time::Instant::now();
            let _ = mean(&arr.view());
            let scalar_time = start.elapsed();
            
            // For large arrays, SIMD should be at least as fast or faster
            if size > 1000 && simd_time > scalar_time * 2 {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simd_mean_consistency() {
        let test_data = StatisticalTestData::generate_sample();
        assert!(SimdConsistencyTester::test_mean_consistency(&test_data));
        
        let large_test_data = StatisticalTestData::generate_large_sample();
        assert!(SimdConsistencyTester::test_mean_consistency(&large_test_data));
    }
    
    #[test]
    fn test_simd_variance_consistency() {
        let test_data = StatisticalTestData::generate_sample();
        assert!(SimdConsistencyTester::test_variance_consistency(&test_data));
        
        let large_test_data = StatisticalTestData::generate_large_sample();
        assert!(SimdConsistencyTester::test_variance_consistency(&large_test_data));
    }
    
    #[test]
    fn test_simd_skewness_consistency() {
        let test_data = StatisticalTestData::generate_sample();
        assert!(SimdConsistencyTester::test_skewness_consistency(&test_data));
        
        let large_test_data = StatisticalTestData::generate_large_sample();
        assert!(SimdConsistencyTester::test_skewness_consistency(&large_test_data));
    }
    
    #[test]
    fn test_simd_kurtosis_consistency() {
        let test_data = StatisticalTestData::generate_sample();
        assert!(SimdConsistencyTester::test_kurtosis_consistency(&test_data));
        
        let large_test_data = StatisticalTestData::generate_large_sample();
        assert!(SimdConsistencyTester::test_kurtosis_consistency(&large_test_data));
    }
    
    #[test]
    fn test_correlation_bounds() {
        let test_data1 = StatisticalTestData::generate_sample();
        let test_data2 = StatisticalTestData::new(vec![10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
        assert!(MathematicalInvariantTester::test_correlation_bounds(&test_data1, &test_data2));
    }
    
    #[test]
    fn test_variance_non_negative() {
        let test_data = StatisticalTestData::generate_sample();
        assert!(MathematicalInvariantTester::test_variance_non_negative(&test_data));
    }
    
    #[test]
    fn test_mean_bounds() {
        let test_data = StatisticalTestData::generate_sample();
        assert!(MathematicalInvariantTester::test_mean_bounds(&test_data));
    }
    
    #[test]
    fn test_perfect_correlation_property() {
        assert!(MathematicalInvariantTester::test_perfect_correlation());
    }
    
    #[test]
    fn test_batch_moments() {
        let test_data = StatisticalTestData::generate_sample();
        assert!(BatchProcessingTester::test_batch_moments_consistency(&test_data));
    }
    
    #[test]
    fn test_parallel_correlation_consistency() {
        let matrix_data = MatrixTestData::generate_sample();
        assert!(ParallelConsistencyTester::test_correlation_matrix_consistency(&matrix_data));
    }
    
    #[test]
    fn test_memory_optimization() {
        let matrix_data = MatrixTestData::generate_sample();
        assert!(MemoryOptimizationTester::test_memory_aware_correlation_consistency(&matrix_data));
    }
    
    #[test]
    fn test_numerical_stability() {
        assert!(NumericalStabilityTester::test_small_numbers_stability());
        assert!(NumericalStabilityTester::test_large_numbers_stability());
        assert!(NumericalStabilityTester::test_mixed_scale_stability());
    }
    
    #[test]
    fn test_edge_cases() {
        assert!(EdgeCaseTester::test_identical_values());
        assert!(EdgeCaseTester::test_alternating_signs());
        assert!(EdgeCaseTester::test_single_element());
    }
    
    #[test]
    fn test_performance_scaling() {
        let sizes = vec![10, 100, 1000, 10000];
        assert!(PerformanceTester::test_performance_scaling(&sizes));
    }
}