//! Advanced Mode Integration Tests
//!
//! Comprehensive integration tests to validate the Advanced mode functionality
//! including SIMD optimizations, parallel processing, and numerical stability.

use ndarray::{array, Array1, Array2, Axis};
use scirs2__stats::{
    create_advanced_processor, create_numerical_stability_analyzer,
    unified_processor::{AdvancedMatrixOperation, AdvancedTimeSeriesOperation},
    AdvancedProcessorConfig, OptimizationMode, ProcessingStrategy,
};
use std::time::Instant;

#[test]
#[allow(dead_code)]
fn test_advanced_basic_functionality() {
    let mut processor = create_advanced_processor();
    let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

    let result = processor
        .process_comprehensive_statistics(&data.view())
        .unwrap();

    // Verify basic statistics
    assert!((result.statistics.mean - 5.5).abs() < 1e-10);
    assert_eq!(result.statistics.count, 10);
    assert_eq!(result.statistics.min, 1.0);
    assert_eq!(result.statistics.max, 10.0);

    // Verify processing metrics
    assert!(result.processing_metrics.data_size == 10);
    assert!(result.processing_metrics.processing_time.as_millis() < 1000); // Should be fast
}

#[test]
#[allow(dead_code)]
fn test_advanced_large_dataset_performance() {
    let mut processor = create_advanced_processor();

    // Generate a larger dataset to trigger optimizations
    let large_data: Array1<f64> = (0..10000).map(|i| (i as f64 / 1000.0).sin()).collect();

    let start_time = Instant::now();
    let result = processor
        .process_comprehensive_statistics(&large_data.view())
        .unwrap();
    let processing_time = start_time.elapsed();

    // Verify the computation is reasonable
    assert!(result.statistics.mean.abs() < 0.1); // Should be close to 0 for sine wave
    assert_eq!(result.statistics.count, 10000);

    // Should use optimizations for large datasets
    assert!(result.processing_metrics.simd_enabled || result.processing_metrics.parallel_enabled);

    // Performance should be reasonable
    assert!(processing_time.as_millis() < 100); // Should complete quickly
}

#[test]
#[allow(dead_code)]
fn test_optimization_mode_selection() {
    let test_data = array![1.0, 2.0, 3.0, 4.0, 5.0];

    let optimization_modes = vec![
        OptimizationMode::Performance,
        OptimizationMode::Accuracy,
        OptimizationMode::Balanced,
        OptimizationMode::Adaptive,
    ];

    for mode in optimization_modes {
        let config = AdvancedProcessorConfig {
            optimization_mode: mode,
            ..Default::default()
        };

        let mut processor = scirs2_stats::unified, _processor::AdvancedUnifiedProcessor::new(config);
        let result = processor
            .process_comprehensive_statistics(&test_data.view())
            .unwrap();

        // All modes should produce the same mathematical result
        assert!((result.statistics.mean - 3.0).abs() < 1e-10);
        assert_eq!(result.statistics.count, 5);

        // But may use different processing strategies
        match mode {
            OptimizationMode::Performance => {
                // Performance mode might use more aggressive optimizations
            }
            OptimizationMode::Accuracy => {
                // Accuracy mode should enable stability testing
                assert!(result.processing_metrics.stability_tested);
            }
            _ => {
                // Other modes should work normally
            }
        }
    }
}

#[test]
#[allow(dead_code)]
fn test_numerical_stability_integration() {
    let mut processor =
        scirs2_stats::unified, _processor::AdvancedUnifiedProcessor::new(AdvancedProcessorConfig {
            enable_stability_testing: true,
            ..Default::default()
        });

    // Test with well-conditioned data
    let normal_data = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = processor
        .process_comprehensive_statistics(&normal_data.view())
        .unwrap();

    assert!(result.stability_report.is_some());
    let stability_report = result.stability_report.unwrap();

    // Normal data should pass most stability tests
    assert!(stability_report.descriptive_stats.passed);

    // Test with problematic data
    let problematic_data = array![1e-15, 2e-15, 1e15, 2e15]; // Large dynamic range
    let result2 = processor
        .process_comprehensive_statistics(&problematic_data.view())
        .unwrap();

    if let Some(ref stability_report2) = result2.stability_report {
        // This might detect issues due to large dynamic range
        let issues = stability_report2.get_all_issues();
        let warnings = stability_report2.get_all_warnings();

        // Either issues or warnings should be detected for this problematic data
        assert!(issues.len() > 0 || warnings.len() > 0);
    }
}

#[test]
#[allow(dead_code)]
fn test_matrix_operations_integration() {
    let mut processor = create_advanced_processor();

    // Create a small test matrix
    let matrix = Array2::from_shape_fn((5, 3), |(i, j)| (i as f64 + 1.0) * (j as f64 + 1.0));

    // Test covariance matrix computation
    let covariance_result = processor
        .process_matrix_operations(&matrix.view(), AdvancedMatrixOperation::Covariance)
        .unwrap();

    // Result should be a 3x3 matrix (features x features)
    assert_eq!(covariance_result.matrix.dim(), (3, 3));

    // Covariance matrix should be symmetric
    for i in 0..3 {
        for j in 0..3 {
            let diff = (covariance_result.matrix[[i, j]] - covariance_result.matrix[[j, i]]).abs();
            assert!(diff < 1e-10, "Covariance matrix should be symmetric");
        }
    }

    // Test correlation matrix computation
    let correlation_result = processor
        .process_matrix_operations(&matrix.view(), AdvancedMatrixOperation::Correlation)
        .unwrap();

    // Result should be a 3x3 matrix
    assert_eq!(correlation_result.matrix.dim(), (3, 3));

    // Diagonal should be 1.0 (perfect self-correlation)
    for i in 0..3 {
        assert!((correlation_result.matrix[[i, i]] - 1.0).abs() < 1e-10);
    }

    // Correlation values should be between -1 and 1
    for i in 0..3 {
        for j in 0..3 {
            let corr = correlation_result.matrix[[i, j]];
            assert!(
                corr >= -1.0 && corr <= 1.0,
                "Correlation should be in [-1, 1], got {}",
                corr
            );
        }
    }
}

#[test]
#[allow(dead_code)]
fn test_time_series_processing() {
    let mut processor = create_advanced_processor();

    // Generate a simple time series
    let time_series: Array1<f64> = (0..100).map(|i| (i as f64 / 10.0).sin()).collect();

    let operations = vec![AdvancedTimeSeriesOperation::MovingWindow];
    let window_size = 10;

    let result = processor
        .process_time_series(&time_series.view(), window_size, &operations)
        .unwrap();

    assert_eq!(result.window_size, window_size);
    assert_eq!(result.operations, operations);
    assert!(!result.results.is_empty());

    // Should produce the right number of windows
    let expected_windows = time_series.len() - window_size + 1;
    if !result.results.is_empty() && !result.results[0].means.is_empty() {
        assert_eq!(result.results[0].means.len(), expected_windows);
    }
}

#[test]
#[allow(dead_code)]
fn test_performance_monitoring() {
    let mut processor =
        scirs2_stats::unified, _processor::AdvancedUnifiedProcessor::new(AdvancedProcessorConfig {
            enable_performance_monitoring: true,
            ..Default::default()
        });

    // Perform several operations to build performance history
    let test_cases = vec![
        array![1.0, 2.0, 3.0],
        array![1.0, 2.0, 3.0, 4.0, 5.0],
        (0..1000).map(|i| i as f64).collect::<Array1<f64>>(),
    ];

    for data in test_cases {
        let _result = processor
            .process_comprehensive_statistics(&data.view())
            .unwrap();
    }

    // Get performance analytics
    let analytics = processor.get_performance_analytics();

    assert!(analytics.total_operations >= 3);
    assert!(analytics.average_processing_time_ms >= 0.0);
    assert!(analytics.simd_usage_rate >= 0.0 && analytics.simd_usage_rate <= 1.0);
    assert!(analytics.parallel_usage_rate >= 0.0 && analytics.parallel_usage_rate <= 1.0);
    assert!(
        analytics.optimization_effectiveness >= 0.0 && analytics.optimization_effectiveness <= 1.0
    );
}

#[test]
#[allow(dead_code)]
fn test_error_handling() {
    let mut processor = create_advanced_processor();

    // Test with empty array
    let empty_data = Array1::<f64>::zeros(0);
    let result = processor.process_comprehensive_statistics(&empty_data.view());
    assert!(result.is_err(), "Should fail with empty data");

    // Test matrix operations with empty matrix
    let empty_matrix = Array2::<f64>::zeros((0, 0));
    let matrix_result = processor
        .process_matrix_operations(&empty_matrix.view(), AdvancedMatrixOperation::Covariance);
    assert!(matrix_result.is_err(), "Should fail with empty matrix");

    // Test time series with invalid window size
    let valid_data = array![1.0, 2.0, 3.0];
    let ts_result = processor.process_time_series(
        &valid_data.view(),
        0, // Invalid window size
        &[AdvancedTimeSeriesOperation::MovingWindow],
    );
    assert!(ts_result.is_err(), "Should fail with zero window size");
}

#[test]
#[allow(dead_code)]
fn test_stability_analyzer_standalone() {
    let mut analyzer = create_numerical_stability_analyzer();

    // Test with normal data
    let normal_data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let report = analyzer.analyze_statistical_stability(&normal_data.view());

    // Should pass most tests with normal data
    assert!(report.descriptive_stats.passed);
    assert!(report.variance_computation.passed);

    // Test with edge case data
    let constant_data = array![5.0, 5.0, 5.0, 5.0, 5.0];
    let constant_report = analyzer.analyze_statistical_stability(&constant_data.view());

    // Should handle constant data properly
    assert!(constant_report.edge_cases.passed);
}

#[test]
#[allow(dead_code)]
fn test_processing_strategy_effectiveness() {
    let strategies = vec![
        ProcessingStrategy::Standard,
        ProcessingStrategy::SimdOnly,
        ProcessingStrategy::ParallelOnly,
        ProcessingStrategy::SimdParallel,
    ];

    // Test that strategy properties work correctly
    for strategy in strategies {
        match strategy {
            ProcessingStrategy::Standard => {
                assert!(!strategy.uses_simd());
                assert!(!strategy.uses_parallel());
            }
            ProcessingStrategy::SimdOnly => {
                assert!(strategy.uses_simd());
                assert!(!strategy.uses_parallel());
            }
            ProcessingStrategy::ParallelOnly => {
                assert!(!strategy.uses_simd());
                assert!(strategy.uses_parallel());
            }
            ProcessingStrategy::SimdParallel => {
                assert!(strategy.uses_simd());
                assert!(strategy.uses_parallel());
            }
        }
    }
}

#[test]
#[allow(dead_code)]
fn test_comprehensive_workflow() {
    // This test demonstrates a complete Advanced workflow
    let mut processor =
        scirs2_stats::unified, _processor::AdvancedUnifiedProcessor::new(AdvancedProcessorConfig {
            optimization_mode: OptimizationMode::Adaptive,
            enable_stability_testing: true,
            enable_performance_monitoring: true,
            ..Default::default()
        });

    // Generate test data with different characteristics
    let datasets = vec![
        // Small dataset
        (0..50).map(|i| i as f64).collect::<Array1<f64>>(),
        // Medium dataset with pattern
        (0..5000)
            .map(|i| (i as f64 / 100.0).sin())
            .collect::<Array1<f64>>(),
        // Large dataset
        (0..50000)
            .map(|i| i as f64 % 1000.0)
            .collect::<Array1<f64>>(),
    ];

    let mut all_results = Vec::new();

    for (i, data) in datasets.iter().enumerate() {
        println!("Processing dataset {} with {} elements", i + 1, data.len());

        let result = processor
            .process_comprehensive_statistics(&data.view())
            .unwrap();

        // Verify mathematical correctness
        assert_eq!(result.statistics.count, data.len());
        assert!(result.statistics.mean.is_finite());
        assert!(result.statistics.std_dev.is_finite());
        assert!(result.statistics.variance >= 0.0);

        // Verify processing completed
        assert!(result.processing_metrics.processing_time.as_millis() < 10000); // Should complete in reasonable time

        all_results.push(result);
    }

    // Get final performance analytics
    let analytics = processor.get_performance_analytics();
    assert_eq!(analytics.total_operations, datasets.len());

    // Adaptive mode should have learned and potentially optimized
    if analytics.total_operations > 2 {
        assert!(analytics.optimization_effectiveness >= 0.0);
    }

    println!("Comprehensive workflow completed successfully!");
    println!("  Total operations: {}", analytics.total_operations);
    println!(
        "  Average processing time: {:.2} ms",
        analytics.average_processing_time_ms
    );
    println!(
        "  SIMD usage rate: {:.1}%",
        analytics.simd_usage_rate * 100.0
    );
    println!(
        "  Parallel usage rate: {:.1}%",
        analytics.parallel_usage_rate * 100.0
    );
}
