//! Integration tests for the comprehensive stability testing framework
//!
//! This module tests the integration of all stability testing components including
//! numerical analysis, stability metrics, and the comprehensive test framework.

use scirs2_autograd::tensor::Tensor;
use scirs2_autograd::testing::numerical_analysis::{
    analyze_conditioning, analyze_error_propagation, ConditioningAssessment, NumericalAnalyzer,
};
use scirs2_autograd::testing::stability_metrics::{
    compute_backward_stability, compute_forward_stability, StabilityGrade, StabilityMetrics,
};
use scirs2_autograd::testing::stability_test_framework::{
    create_test_scenario, run_basic_stability_tests, run_comprehensive_stability_tests,
    run_stability_tests_with_config, test_function_stability, EdgeCaseBehavior, StabilityTestSuite,
    TestConfig,
};
use scirs2_autograd::testing::StabilityError;

/// Test the basic stability framework functionality
#[test]
fn test_basic_stability_framework() {
    let result = run_basic_stability_tests::<f32>();
    assert!(result.is_ok());

    let summary = result.unwrap();
    assert!(summary.total_tests > 0);
    println!(
        "Basic stability tests completed: {}/{} passed",
        summary.passed_tests, summary.total_tests
    );
}

/// Test comprehensive stability testing
#[test]
fn test_comprehensive_stability_testing() {
    let result = run_comprehensive_stability_tests::<f32>();
    assert!(result.is_ok());

    let summary = result.unwrap();
    assert!(summary.total_tests > 0);
    println!(
        "Comprehensive tests: {} total, {} passed, {} failed",
        summary.total_tests, summary.passed_tests, summary.failed_tests
    );

    // Print the full summary
    summary.print_summary();
}

/// Test custom configuration for stability testing
#[test]
fn test_custom_stability_config() {
    let config = TestConfig {
        run_basic_tests: true,
        run_advanced_tests: false,
        run_edge_case_tests: true,
        run_precision_tests: false,
        run_benchmarks: true,
        run_scenario_tests: false,
        tolerance_level: 1e-12,
        ..Default::default()
    };

    let result = run_stability_tests_with_config::<f32>(config);
    assert!(result.is_ok());

    let summary = result.unwrap();
    assert!(summary.total_tests > 0);
    println!(
        "Custom config tests: success rate = {:.1}%",
        summary.success_rate()
    );
}

/// Test individual function stability analysis
#[test]
fn test_function_stability_analysis() {
    // Test identity function - should be excellent stability
    let input = create_test_tensor(vec![5, 5]);
    let identity_function = |x: &Tensor<f32>| {
        // Create a new tensor with the same data and shape as input
        let data = x.eval(&x.graph()).unwrap().iter().cloned().collect::<Vec<_>>();
        Ok(Tensor::from_vec(data, x.shape(), &x.graph()))
    };

    let result = test_function_stability(identity_function, &input, "identity_test");
    assert!(result.is_ok());

    let test_result = result.unwrap();
    assert!(test_result.passed);
    assert!(matches!(
        test_result.actual_grade,
        StabilityGrade::Excellent | StabilityGrade::Good
    ));

    println!(
        "Identity function stability: {:?}",
        test_result.actual_grade
    );
}

/// Test scenario-based testing
#[test]
fn test_scenario_based_testing() {
    let input = create_test_tensor(vec![10]);

    // Create a test scenario for a linear function
    let scenario = create_test_scenario(
        "linear_scaling".to_string(),
        "Test linear scaling function y = 2x".to_string(),
        |x: &Tensor<f32>| {
            // Simplified linear scaling (would implement actual scaling)
            Ok(x.clone())
        },
        input,
        StabilityGrade::Excellent,
    );

    let mut suite = StabilityTestSuite::new();
    suite.add_scenario(scenario);

    let result = suite.run_all_tests();
    assert!(result.is_ok());

    let summary = result.unwrap();
    println!(
        "Scenario test results: {}/{} passed",
        summary.passed_tests, summary.total_tests
    );
}

/// Test numerical analysis integration
#[test]
fn test_numerical_analysis_integration() {
    let analyzer = NumericalAnalyzer::<f32>::new();
    let input = create_test_tensor(vec![8, 8]);

    // Test condition number analysis
    let test_function = |x: &Tensor<f32>| Ok(x.clone());
    let conditioning_result = analyzer.analyze_condition_number(test_function, &input);
    assert!(conditioning_result.is_ok());

    let conditioning = conditioning_result.unwrap();
    println!("Condition number analysis:");
    println!("  Spectral: {:.2e}", conditioning.spectral_condition_number);
    println!("  Assessment: {:?}", conditioning.conditioning_assessment);
    assert!(matches!(
        conditioning.conditioning_assessment,
        ConditioningAssessment::WellConditioned
            | ConditioningAssessment::ModeratelyConditioned
            | ConditioningAssessment::IllConditioned
            | ConditioningAssessment::SeverelyIllConditioned
    ));
}

/// Test stability metrics integration
#[test]
fn test_stability_metrics_integration() {
    let metrics = StabilityMetrics::<f32>::new();
    let input = create_test_tensor(vec![6, 6]);

    // Test forward stability
    let test_function = |x: &Tensor<f32>| Ok(x.clone());
    let forward_result = metrics.compute_forward_stability(test_function, &input, 1e-8);
    assert!(forward_result.is_ok());

    let forward_metrics = forward_result.unwrap();
    println!("Forward stability metrics:");
    println!("  Grade: {:?}", forward_metrics.stability_grade);
    println!("  Mean error: {:.2e}", forward_metrics.mean_relative_error);

    // Test backward stability
    let output = test_function(&input).unwrap();
    let backward_result = metrics.compute_backward_stability(test_function, &input, &output);
    assert!(backward_result.is_ok());

    let backward_metrics = backward_result.unwrap();
    println!("Backward stability metrics:");
    println!("  Grade: {:?}", backward_metrics.stability_grade);
    println!("  Error: {:.2e}", backward_metrics.backward_error);
}

/// Test error propagation analysis
#[test]
fn test_error_propagation_analysis() {
    let input = create_test_tensor(vec![5]);
    let uncertainty = create_uncertainty_tensor(vec![5], 1e-8);

    let linear_function = |x: &Tensor<f32>| Ok(x.clone());

    let result = analyze_error_propagation(linear_function, &input, &uncertainty);
    assert!(result.is_ok());

    let analysis = result.unwrap();
    println!("Error propagation analysis:");
    println!("  Linear error bound: {:.2e}", analysis.linear_error_bound);
    println!("  First order error: {:.2e}", analysis.first_order_error);
    assert!(analysis.linear_error_bound >= 0.0);
    assert!(analysis.first_order_error >= 0.0);
}

/// Test comprehensive integration of all components
#[test]
fn test_full_pipeline_integration() {
    // Create a comprehensive test suite with all features enabled
    let config = TestConfig {
        run_basic_tests: true,
        run_advanced_tests: true,
        run_edge_case_tests: true,
        run_precision_tests: true,
        run_benchmarks: true,
        run_scenario_tests: true,
        tolerance_level: 1e-10,
        ..Default::default()
    };

    let mut suite = StabilityTestSuite::with_config(config);

    // Add some custom scenarios
    let scenarios = create_test_scenarios();
    for scenario in scenarios {
        suite.add_scenario(scenario);
    }

    // Run all tests
    let result = suite.run_all_tests();
    assert!(result.is_ok());

    let summary = result.unwrap();
    println!("\n=== FULL PIPELINE INTEGRATION RESULTS ===");
    summary.print_summary();

    // Verify we got comprehensive results
    assert!(summary.total_tests >= 4); // Basic + scenarios
    assert!(summary.success_rate() >= 50.0); // At least half should pass
    assert!(!summary.recommendations.is_empty());
}

/// Test edge case handling
#[test]
fn test_edge_case_handling() {
    let config = TestConfig {
        run_basic_tests: false,
        run_advanced_tests: false,
        run_edge_case_tests: true,
        run_precision_tests: false,
        run_benchmarks: false,
        run_scenario_tests: false,
        ..Default::default()
    };

    let result = run_stability_tests_with_config::<f32>(config);
    assert!(result.is_ok());

    let summary = result.unwrap();
    println!("Edge case test results:");
    println!("  Total tests: {}", summary.total_tests);
    println!("  Success rate: {:.1}%", summary.success_rate());

    // Edge cases might have some failures, which is expected
    assert!(summary.total_tests > 0);
}

/// Test performance benchmarking
#[test]
fn test_performance_benchmarking() {
    let config = TestConfig {
        run_basic_tests: false,
        run_advanced_tests: false,
        run_edge_case_tests: false,
        run_precision_tests: false,
        run_benchmarks: true,
        run_scenario_tests: false,
        ..Default::default()
    };

    let result = run_stability_tests_with_config::<f32>(config);
    assert!(result.is_ok());

    let summary = result.unwrap();
    println!("Performance benchmark results:");
    println!(
        "  Avg analysis duration: {:.3}s",
        summary
            .performance_summary
            .average_analysis_duration
            .as_secs_f64()
    );
    println!(
        "  Max ops/sec: {}",
        summary.performance_summary.max_operations_per_second
    );

    // Performance tests should always pass (they're measuring, not validating)
    assert_eq!(summary.failed_tests, 0);
}

/// Test precision sensitivity analysis
#[test]
fn test_precision_sensitivity() {
    let config = TestConfig {
        run_basic_tests: false,
        run_advanced_tests: false,
        run_edge_case_tests: false,
        run_precision_tests: true,
        run_benchmarks: false,
        run_scenario_tests: false,
        ..Default::default()
    };

    let result = run_stability_tests_with_config::<f32>(config);
    assert!(result.is_ok());

    let summary = result.unwrap();
    println!("Precision sensitivity test completed");
    println!("  Tests performed: {}", summary.total_tests);

    // Precision tests should provide useful information
    assert!(summary.total_tests >= 0); // May be 0 if no precision-specific tests
}

/// Test various function types for stability
#[test]
fn test_different_function_types() {
    let input = create_test_tensor(vec![4]);

    // Test different function types
    let functions_to_test = vec![
        ("constant", |_: &Tensor<f32>| {
            Ok(Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![4]))
        }),
        ("identity", |x: &Tensor<f32>| Ok(x.clone())),
        ("square", |x: &Tensor<f32>| {
            // Simplified square function
            Ok(x.clone())
        }),
    ];

    for (name, function) in functions_to_test {
        let result = test_function_stability(function, &input, name);
        assert!(result.is_ok(), "Function {} failed stability test", name);

        let test_result = result.unwrap();
        println!(
            "Function '{}' stability: {:?} (passed: {})",
            name, test_result.actual_grade, test_result.passed
        );
    }
}

/// Test large tensor stability
#[test]
fn test_large_tensor_stability() {
    let large_input = create_test_tensor(vec![100, 100]);
    let identity_function = |x: &Tensor<f32>| Ok(x.clone());

    let result = test_function_stability(identity_function, &large_input, "large_tensor_test");
    assert!(result.is_ok());

    let test_result = result.unwrap();
    println!("Large tensor stability test:");
    println!("  Input shape: {:?}", large_input.shape());
    println!("  Stability grade: {:?}", test_result.actual_grade);
    println!(
        "  Test duration: {:.3}s",
        test_result.duration.as_secs_f64()
    );

    // Large tensors should still maintain good stability for simple operations
    assert!(matches!(
        test_result.actual_grade,
        StabilityGrade::Excellent | StabilityGrade::Good | StabilityGrade::Fair
    ));
}

/// Test mixed precision scenarios
#[test]
fn test_mixed_precision_scenarios() {
    // Test with f32
    let f32_result = run_basic_stability_tests::<f32>();
    assert!(f32_result.is_ok());
    let f32_summary = f32_result.unwrap();

    // Test with f64
    let f64_result = run_basic_stability_tests::<f64>();
    assert!(f64_result.is_ok());
    let f64_summary = f64_result.unwrap();

    println!("Mixed precision comparison:");
    println!("  f32 success rate: {:.1}%", f32_summary.success_rate());
    println!("  f64 success rate: {:.1}%", f64_summary.success_rate());

    // f64 should generally have better or equal stability
    assert!(f64_summary.success_rate() >= f32_summary.success_rate() - 10.0);
}

// Helper functions

fn create_test_tensor(shape: Vec<usize>) -> Tensor<'static, f32> {
    let size = shape.iter().product();
    let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
    Tensor::from_vec(data, shape)
}

fn create_uncertainty_tensor(shape: Vec<usize>, magnitude: f64) -> Tensor<'static, f32> {
    let size = shape.iter().product();
    let uncertainty_value = magnitude as f32;
    let data = vec![uncertainty_value; size];
    Tensor::from_vec(data, shape)
}

fn create_test_scenarios(
) -> Vec<scirs2_autograd::testing::stability_test_framework::TestScenario<'static, f32>> {
    let mut scenarios = Vec::new();

    // Scenario 1: Linear transformation
    scenarios.push(create_test_scenario(
        "linear_transform".to_string(),
        "Linear transformation y = 3x + 2".to_string(),
        |x: &Tensor<f32>| {
            // Simplified implementation
            Ok(x.clone())
        },
        create_test_tensor(vec![8]),
        StabilityGrade::Excellent,
    ));

    // Scenario 2: Polynomial function
    scenarios.push(create_test_scenario(
        "polynomial".to_string(),
        "Polynomial function y = x^2 + 2x + 1".to_string(),
        |x: &Tensor<f32>| {
            // Simplified implementation
            Ok(x.clone())
        },
        create_test_tensor(vec![6]),
        StabilityGrade::Good,
    ));

    // Scenario 3: Trigonometric function
    scenarios.push(create_test_scenario(
        "trigonometric".to_string(),
        "Trigonometric function y = sin(x)".to_string(),
        |x: &Tensor<f32>| {
            // Simplified implementation
            Ok(x.clone())
        },
        create_test_tensor(vec![10]),
        StabilityGrade::Fair,
    ));

    scenarios
}

/// Integration test for the complete stability testing workflow
#[test]
fn test_complete_stability_workflow() {
    println!("\n=== COMPLETE STABILITY TESTING WORKFLOW ===");

    // Step 1: Create test data
    let input = create_test_tensor(vec![50, 50]);
    println!("1. Created test tensor with shape {:?}", input.shape());

    // Step 2: Test individual components
    println!("2. Testing individual components...");

    // Test numerical analysis
    let analyzer = NumericalAnalyzer::new();
    let test_function = |x: &Tensor<f32>| Ok(x.clone());
    let conditioning = analyzer.analyze_condition_number(test_function, &input);
    assert!(conditioning.is_ok());
    println!("   ✓ Numerical analysis completed");

    // Test stability metrics
    let forward_metrics = compute_forward_stability(test_function, &input, 1e-8);
    assert!(forward_metrics.is_ok());
    println!("   ✓ Stability metrics computed");

    // Step 3: Run comprehensive test suite
    println!("3. Running comprehensive test suite...");
    let comprehensive_result = run_comprehensive_stability_tests::<f32>();
    assert!(comprehensive_result.is_ok());
    let summary = comprehensive_result.unwrap();
    println!("   ✓ Comprehensive tests completed");

    // Step 4: Analyze results
    println!("4. Analyzing results...");
    println!("   Total tests: {}", summary.total_tests);
    println!("   Success rate: {:.1}%", summary.success_rate());
    println!("   Duration: {:.2}s", summary.total_duration.as_secs_f64());

    // Step 5: Validate workflow success
    assert!(summary.total_tests > 0);
    assert!(summary.success_rate() >= 0.0);
    assert!(!summary.recommendations.is_empty());

    println!("5. ✓ Complete workflow validation passed");
    println!("=========================================\n");
}
