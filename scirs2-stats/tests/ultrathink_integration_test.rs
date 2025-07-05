//! Integration tests for Advanced mode functionality

use ndarray::Array1;
use scirs2_stats::advanced_error_enhancements::{AdvancedContextBuilder, AdvancedErrorMessages};
use scirs2_stats::advanced_numerical_stability::{
    AdvancedNumericalStabilityAnalyzer, NumericalStabilityConfig,
};
use scirs2_stats::advanced_property_tests::AdvancedPropertyTester;
use scirs2_stats::advanced_simd_optimizations::{advanced_batch_statistics, AdvancedSimdConfig};
use scirs2_stats::parallel_enhancements::AdvancedParallelConfig;
use scirs2_stats::{mean, var};

#[test]
#[allow(dead_code)]
fn test_advanced_error_context_creation() {
    let context = AdvancedContextBuilder::new(1000)
        .memory_usage(256.0)
        .simd_enabled(true)
        .parallel_enabled(false)
        .build();

    assert_eq!(context.data_size, 1000);
    assert_eq!(context.memory_usage_mb, 256.0);
    assert!(context.simd_enabled);
    assert!(!context.parallel_enabled);
}

#[test]
#[allow(dead_code)]
fn test_advanced_error_messages() {
    // Test memory exhaustion message
    let error = AdvancedErrorMessages::memory_exhaustion(2000.0, 1000.0, 50000);
    let error_str = format!("{}", error);
    assert!(error_str.contains("Memory exhaustion"));
    assert!(error_str.contains("2000.0MB"));
    assert!(error_str.contains("1000.0MB"));
}

#[test]
#[allow(dead_code)]
fn test_advanced_numerical_stability_config() {
    let config = NumericalStabilityConfig::default();
    assert_eq!(config.relative_tolerance, 1e-12);
    assert_eq!(config.absolute_tolerance, 1e-15);
    assert!(config.detect_cancellation);
    assert!(config.detect_overflow);
    assert!(config.test_extreme_values);
}

#[test]
#[allow(dead_code)]
fn test_advanced_parallel_config() {
    let config = AdvancedParallelConfig::default();
    assert!(config.max_threads > 0);
    assert_eq!(config.min_chunk_size, 1000);
    assert!(config.enable_work_stealing);
    assert!(config.adaptive_scaling);
}

#[test]
#[allow(dead_code)]
fn test_advanced_simd_config() {
    let config = AdvancedSimdConfig::default();
    assert_eq!(config.min_simd_size, 64);
    assert_eq!(config.chunk_size, 8192);
    assert!(config.adaptive_vectorization);
    assert!(config.enable_prefetch);
    assert_eq!(config.memory_threshold_mb, 1024.0);
}

#[test]
#[allow(dead_code)]
fn test_advanced_batch_statistics() {
    let data = Array1::from_vec((1..=1000).map(|x| x as f64).collect());
    let config = AdvancedSimdConfig::default();

    let result = advanced_batch_statistics(&data.view(), &config);
    assert!(result.is_ok());

    let stats = result.unwrap();
    assert!(stats.mean > 0.0);
    assert!(stats.variance > 0.0);
    assert!(stats.min > 0.0);
    assert!(stats.max > 0.0);

    // Verify against known values for 1..1000
    let expected_mean = 500.5; // (1 + 1000) / 2
    assert!((stats.mean - expected_mean).abs() < 1e-10);
}

#[test]
#[allow(dead_code)]
fn test_advanced_property_tester_creation() {
    let simd_config = AdvancedSimdConfig::default();
    let parallel_config = AdvancedParallelConfig::default();

    let tester = AdvancedPropertyTester::new(simd_config, parallel_config);

    // Just verify it can be created without panicking
    assert_eq!(tester.numerical_tolerance, 1e-12);
    assert_eq!(tester.performance_tolerance, 2.0);
}

#[test]
#[allow(dead_code)]
fn test_numerical_stability_analyzer() {
    let config = NumericalStabilityConfig::default();
    let analyzer = AdvancedNumericalStabilityAnalyzer::new(config);

    // Test that analyzer can be created and used
    let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

    let result = analyzer.test_function_stability("mean", &data, |test_data| {
        mean(&test_data.view()).map_err(|e| format!("{}", e))
    });

    assert!(result.is_ok());
}

#[test]
#[allow(dead_code)]
fn test_advanced_integration_basic_stats() {
    // Test that Advanced features work with basic statistical functions
    let data = Array1::from_vec((1..=100).map(|x| x as f64).collect());

    // Test mean calculation
    let mean_result = mean(&data.view());
    assert!(mean_result.is_ok());
    assert_eq!(mean_result.unwrap(), 50.5);

    // Test variance calculation
    let var_result = var(&data.view(), 1);
    assert!(var_result.is_ok());
    assert!(var_result.unwrap() > 0.0);

    // Test Advanced batch statistics
    let config = AdvancedSimdConfig::default();
    let batch_result = advanced_batch_statistics(&data.view(), &config);
    assert!(batch_result.is_ok());

    let batch_stats = batch_result.unwrap();
    assert!((batch_stats.mean - 50.5).abs() < 1e-10);
    assert!(batch_stats.variance > 0.0);
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use quickcheck::TestResult;

    #[quickcheck::quickcheck]
    fn test_advanced_error_context_properties(data_size: usize, memory_mb: f64) -> TestResult {
        if data_size == 0 || memory_mb < 0.0 || memory_mb > 1e9 {
            return TestResult::discard();
        }

        let context = AdvancedContextBuilder::new(data_size)
            .memory_usage(memory_mb)
            .build();

        TestResult::from_bool(
            context.data_size == data_size && context.memory_usage_mb == memory_mb,
        )
    }
}
