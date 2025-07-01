//! Comprehensive numerical accuracy validation for ultrathink mode
//!
//! This module tests the numerical accuracy of graph algorithms when
//! run through ultrathink optimizations compared to reference implementations.

use scirs2_graph::base::Graph;
use scirs2_graph::ultrathink::{
    create_adaptive_ultrathink_processor, create_enhanced_ultrathink_processor,
    create_memory_efficient_ultrathink_processor, create_performance_ultrathink_processor,
    execute_with_enhanced_ultrathink, UltrathinkProcessor, UltrathinkStats,
};
use std::collections::HashMap;
use std::f64;

const EPSILON: f64 = 1e-10;
const RELATIVE_TOLERANCE: f64 = 1e-6;

/// Test data structure for numerical validation
#[derive(Debug, Clone)]
struct ValidationTestCase {
    name: String,
    graph: Graph<usize, f64>,
    expected_results: HashMap<String, f64>,
    tolerance: f64,
}

/// Generate a set of reference test graphs with known properties
fn generate_reference_test_graphs() -> Vec<ValidationTestCase> {
    let mut test_cases = Vec::new();

    // Test Case 1: Simple linear graph
    let mut linear_graph = Graph::new();
    linear_graph.add_edge(0, 1, 1.0).unwrap();
    linear_graph.add_edge(1, 2, 2.0).unwrap();
    linear_graph.add_edge(2, 3, 3.0).unwrap();

    let mut linear_expected = HashMap::new();
    linear_expected.insert("node_count".to_string(), 4.0);
    linear_expected.insert("edge_count".to_string(), 3.0);
    linear_expected.insert("total_weight".to_string(), 6.0);

    test_cases.push(ValidationTestCase {
        name: "linear_graph".to_string(),
        graph: linear_graph,
        expected_results: linear_expected,
        tolerance: EPSILON,
    });

    // Test Case 2: Complete graph K4
    let mut complete_graph = Graph::new();
    for i in 0..4 {
        for j in (i + 1)..4 {
            complete_graph.add_edge(i, j, 1.0).unwrap();
        }
    }

    let mut complete_expected = HashMap::new();
    complete_expected.insert("node_count".to_string(), 4.0);
    complete_expected.insert("edge_count".to_string(), 6.0);
    complete_expected.insert("total_weight".to_string(), 6.0);
    complete_expected.insert("density".to_string(), 1.0); // Complete graph

    test_cases.push(ValidationTestCase {
        name: "complete_k4".to_string(),
        graph: complete_graph,
        expected_results: complete_expected,
        tolerance: EPSILON,
    });

    // Test Case 3: Star graph
    let mut star_graph = Graph::new();
    for i in 1..6 {
        star_graph.add_edge(0, i, i as f64).unwrap();
    }

    let mut star_expected = HashMap::new();
    star_expected.insert("node_count".to_string(), 6.0);
    star_expected.insert("edge_count".to_string(), 5.0);
    star_expected.insert("total_weight".to_string(), 15.0); // 1+2+3+4+5
    star_expected.insert("center_degree".to_string(), 5.0);

    test_cases.push(ValidationTestCase {
        name: "star_graph".to_string(),
        graph: star_graph,
        expected_results: star_expected,
        tolerance: EPSILON,
    });

    // Test Case 4: Cycle graph
    let mut cycle_graph = Graph::new();
    for i in 0..5 {
        cycle_graph.add_edge(i, (i + 1) % 5, 1.0).unwrap();
    }

    let mut cycle_expected = HashMap::new();
    cycle_expected.insert("node_count".to_string(), 5.0);
    cycle_expected.insert("edge_count".to_string(), 5.0);
    cycle_expected.insert("total_weight".to_string(), 5.0);
    cycle_expected.insert("is_cyclic".to_string(), 1.0); // True

    test_cases.push(ValidationTestCase {
        name: "cycle_graph".to_string(),
        graph: cycle_graph,
        expected_results: cycle_expected,
        tolerance: EPSILON,
    });

    test_cases
}

/// Calculate reference graph properties directly (without optimization)
fn calculate_reference_properties(graph: &Graph<usize, f64>) -> HashMap<String, f64> {
    let mut properties = HashMap::new();

    // Basic properties
    properties.insert("node_count".to_string(), graph.node_count() as f64);
    properties.insert("edge_count".to_string(), graph.edge_count() as f64);

    // Calculate total weight
    let total_weight: f64 = graph.edges().map(|edge| *edge.weight()).sum();
    properties.insert("total_weight".to_string(), total_weight);

    // Calculate density
    let n = graph.node_count() as f64;
    if n > 1.0 {
        let density = (graph.edge_count() as f64) / (n * (n - 1.0) / 2.0);
        properties.insert("density".to_string(), density);
    }

    // Calculate degree-related properties
    let nodes: Vec<_> = graph.nodes().collect();
    if !nodes.is_empty() {
        let degrees: Vec<usize> = nodes.iter().map(|&node| graph.degree(node)).collect();
        let max_degree = *degrees.iter().max().unwrap_or(&0);
        let min_degree = *degrees.iter().min().unwrap_or(&0);
        let avg_degree = degrees.iter().sum::<usize>() as f64 / degrees.len() as f64;

        properties.insert("max_degree".to_string(), max_degree as f64);
        properties.insert("min_degree".to_string(), min_degree as f64);
        properties.insert("avg_degree".to_string(), avg_degree);

        // Special case: center degree for star graphs
        if max_degree > 0 && degrees.iter().filter(|&&d| d == max_degree).count() == 1 {
            properties.insert("center_degree".to_string(), max_degree as f64);
        }
    }

    // Check for cycles (simplified - just check if any node has degree > 2)
    let has_high_degree = graph.nodes().any(|node| graph.degree(node) > 2);
    properties.insert(
        "is_cyclic".to_string(),
        if has_high_degree { 1.0 } else { 0.0 },
    );

    properties
}

/// Test numerical accuracy of a single test case
fn test_numerical_accuracy(
    test_case: &ValidationTestCase,
    processor: &mut UltrathinkProcessor,
    processor_name: &str,
) -> ValidationResult {
    let mut results = ValidationResult::new(test_case.name.clone(), processor_name.to_string());

    // Calculate reference properties
    let reference_props = calculate_reference_properties(&test_case.graph);

    // Calculate properties using ultrathink optimization
    let optimized_props = execute_with_enhanced_ultrathink(
        processor,
        &test_case.graph,
        &format!("validation_{}", test_case.name),
        |graph| Ok(calculate_reference_properties(graph)),
    )
    .unwrap_or_else(|_| HashMap::new());

    // Compare results
    for (property, &expected) in &test_case.expected_results {
        let reference_val = reference_props.get(property).copied().unwrap_or(0.0);
        let optimized_val = optimized_props.get(property).copied().unwrap_or(0.0);

        let accuracy =
            calculate_accuracy(expected, reference_val, optimized_val, test_case.tolerance);
        results.add_property_result(
            property.clone(),
            expected,
            reference_val,
            optimized_val,
            accuracy,
        );
    }

    // Also compare additional computed properties
    for (property, &reference_val) in &reference_props {
        if !test_case.expected_results.contains_key(property) {
            let optimized_val = optimized_props.get(property).copied().unwrap_or(0.0);
            let accuracy = calculate_accuracy(
                reference_val,
                reference_val,
                optimized_val,
                RELATIVE_TOLERANCE,
            );
            results.add_property_result(
                property.clone(),
                reference_val,
                reference_val,
                optimized_val,
                accuracy,
            );
        }
    }

    results
}

/// Calculate numerical accuracy between expected, reference, and optimized values
fn calculate_accuracy(expected: f64, reference: f64, optimized: f64, tolerance: f64) -> f64 {
    if (expected - reference).abs() < tolerance && (reference - optimized).abs() < tolerance {
        1.0 // Perfect accuracy
    } else if expected.is_finite() && reference.is_finite() && optimized.is_finite() {
        let ref_error = (expected - reference).abs() / expected.abs().max(1.0);
        let opt_error = (reference - optimized).abs() / reference.abs().max(1.0);
        let total_error = ref_error + opt_error;

        if total_error < tolerance {
            1.0 - total_error
        } else {
            (1.0 / (1.0 + total_error)).max(0.0)
        }
    } else {
        0.0 // NaN or infinite values
    }
}

/// Results of a validation test
#[derive(Debug, Clone)]
struct ValidationResult {
    test_name: String,
    processor_name: String,
    property_results: HashMap<String, PropertyResult>,
    overall_accuracy: f64,
}

#[derive(Debug, Clone)]
struct PropertyResult {
    expected: f64,
    reference: f64,
    optimized: f64,
    accuracy: f64,
}

impl ValidationResult {
    fn new(test_name: String, processor_name: String) -> Self {
        Self {
            test_name,
            processor_name,
            property_results: HashMap::new(),
            overall_accuracy: 0.0,
        }
    }

    fn add_property_result(
        &mut self,
        property: String,
        expected: f64,
        reference: f64,
        optimized: f64,
        accuracy: f64,
    ) {
        self.property_results.insert(
            property,
            PropertyResult {
                expected,
                reference,
                optimized,
                accuracy,
            },
        );

        // Update overall accuracy
        if !self.property_results.is_empty() {
            self.overall_accuracy = self
                .property_results
                .values()
                .map(|r| r.accuracy)
                .sum::<f64>()
                / self.property_results.len() as f64;
        }
    }

    fn is_passing(&self, threshold: f64) -> bool {
        self.overall_accuracy >= threshold
    }

    fn report(&self) -> String {
        let mut report = format!(
            "Validation Result: {} with {}\n",
            self.test_name, self.processor_name
        );
        report.push_str(&format!("Overall Accuracy: {:.6}\n", self.overall_accuracy));

        for (property, result) in &self.property_results {
            report.push_str(&format!(
                "  {}: Expected={:.6}, Reference={:.6}, Optimized={:.6}, Accuracy={:.6}\n",
                property, result.expected, result.reference, result.optimized, result.accuracy
            ));
        }

        report
    }
}

/// Comprehensive numerical validation suite
fn run_comprehensive_validation() -> Vec<ValidationResult> {
    let test_cases = generate_reference_test_graphs();
    let mut all_results = Vec::new();

    // Test different processor configurations
    let mut processors = vec![
        ("enhanced", create_enhanced_ultrathink_processor()),
        ("performance", create_performance_ultrathink_processor()),
        (
            "memory_efficient",
            create_memory_efficient_ultrathink_processor(),
        ),
        ("adaptive", create_adaptive_ultrathink_processor()),
    ];

    for test_case in &test_cases {
        for (processor_name, processor) in &mut processors {
            let result = test_numerical_accuracy(test_case, processor, processor_name);
            all_results.push(result);
        }
    }

    all_results
}

/// Generate validation report
fn generate_validation_report(results: &[ValidationResult]) -> String {
    let mut report = String::new();
    report.push_str("=== Ultrathink Numerical Validation Report ===\n\n");

    // Overall statistics
    let total_tests = results.len();
    let passing_tests = results.iter().filter(|r| r.is_passing(0.95)).count();
    let avg_accuracy = results.iter().map(|r| r.overall_accuracy).sum::<f64>() / total_tests as f64;

    report.push_str(&format!("Total Tests: {}\n", total_tests));
    report.push_str(&format!(
        "Passing Tests (>95% accuracy): {}\n",
        passing_tests
    ));
    report.push_str(&format!(
        "Pass Rate: {:.2}%\n",
        (passing_tests as f64 / total_tests as f64) * 100.0
    ));
    report.push_str(&format!("Average Accuracy: {:.6}\n\n", avg_accuracy));

    // Group results by processor
    let mut processor_stats: HashMap<String, Vec<&ValidationResult>> = HashMap::new();
    for result in results {
        processor_stats
            .entry(result.processor_name.clone())
            .or_insert_with(Vec::new)
            .push(result);
    }

    report.push_str("=== Results by Processor ===\n");
    for (processor, processor_results) in &processor_stats {
        let processor_avg = processor_results
            .iter()
            .map(|r| r.overall_accuracy)
            .sum::<f64>()
            / processor_results.len() as f64;
        let processor_passing = processor_results
            .iter()
            .filter(|r| r.is_passing(0.95))
            .count();

        report.push_str(&format!(
            "\n{}: {:.6} avg accuracy, {}/{} passing\n",
            processor,
            processor_avg,
            processor_passing,
            processor_results.len()
        ));
    }

    // Detailed results
    report.push_str("\n=== Detailed Results ===\n");
    for result in results {
        report.push_str(&result.report());
        report.push('\n');
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reference_graph_generation() {
        let test_cases = generate_reference_test_graphs();
        assert!(!test_cases.is_empty());

        for test_case in &test_cases {
            assert!(!test_case.name.is_empty());
            assert!(test_case.graph.node_count() > 0);
            assert!(!test_case.expected_results.is_empty());
        }
    }

    #[test]
    fn test_reference_property_calculation() {
        let mut graph = Graph::new();
        graph.add_edge(0, 1, 2.0).unwrap();
        graph.add_edge(1, 2, 3.0).unwrap();

        let props = calculate_reference_properties(&graph);

        assert_eq!(props.get("node_count"), Some(&3.0));
        assert_eq!(props.get("edge_count"), Some(&2.0));
        assert_eq!(props.get("total_weight"), Some(&5.0));
    }

    #[test]
    fn test_accuracy_calculation() {
        // Perfect accuracy
        assert_eq!(calculate_accuracy(1.0, 1.0, 1.0, 1e-10), 1.0);

        // Small error within tolerance
        assert!(calculate_accuracy(1.0, 1.0, 1.000001, 1e-3) > 0.99);

        // Large error
        assert!(calculate_accuracy(1.0, 1.0, 2.0, 1e-10) < 0.5);

        // Handle edge cases
        assert_eq!(calculate_accuracy(f64::NAN, 1.0, 1.0, 1e-10), 0.0);
        assert_eq!(calculate_accuracy(1.0, f64::INFINITY, 1.0, 1e-10), 0.0);
    }

    #[test]
    fn test_single_validation() {
        let test_cases = generate_reference_test_graphs();
        let test_case = &test_cases[0]; // Linear graph

        let mut processor = create_enhanced_ultrathink_processor();
        let result = test_numerical_accuracy(test_case, &mut processor, "test");

        assert!(!result.property_results.is_empty());
        assert!(result.overall_accuracy >= 0.0);
        assert!(result.overall_accuracy <= 1.0);
    }

    #[test]
    fn test_comprehensive_validation() {
        let results = run_comprehensive_validation();
        assert!(!results.is_empty());

        // Check that we have results for all processor types
        let processor_names: std::collections::HashSet<_> =
            results.iter().map(|r| &r.processor_name).collect();
        assert!(processor_names.contains("enhanced"));
        assert!(processor_names.contains("performance"));
        assert!(processor_names.contains("memory_efficient"));
        assert!(processor_names.contains("adaptive"));

        // Generate report (should not crash)
        let report = generate_validation_report(&results);
        assert!(!report.is_empty());
        println!("Validation Report:\n{}", report);

        // Check that most tests pass
        let passing_rate =
            results.iter().filter(|r| r.is_passing(0.90)).count() as f64 / results.len() as f64;
        assert!(
            passing_rate >= 0.8,
            "Pass rate too low: {:.2}",
            passing_rate
        );
    }

    #[test]
    fn test_validation_result_reporting() {
        let mut result = ValidationResult::new("test".to_string(), "processor".to_string());
        result.add_property_result("prop1".to_string(), 1.0, 1.0, 1.0, 1.0);
        result.add_property_result("prop2".to_string(), 2.0, 2.0, 2.1, 0.95);

        assert!(result.is_passing(0.9));
        assert!(!result.is_passing(0.99));

        let report = result.report();
        assert!(report.contains("test"));
        assert!(report.contains("processor"));
        assert!(report.contains("prop1"));
        assert!(report.contains("prop2"));
    }
}

/// Integration test that can be run manually
#[test]
fn integration_test_numerical_validation() {
    let results = run_comprehensive_validation();
    let report = generate_validation_report(&results);

    // Print report for manual inspection
    println!("\n{}", report);

    // Assert minimum quality standards
    let avg_accuracy =
        results.iter().map(|r| r.overall_accuracy).sum::<f64>() / results.len() as f64;
    assert!(
        avg_accuracy >= 0.95,
        "Average accuracy too low: {:.6}",
        avg_accuracy
    );

    let passing_rate =
        results.iter().filter(|r| r.is_passing(0.95)).count() as f64 / results.len() as f64;
    assert!(
        passing_rate >= 0.8,
        "Pass rate too low: {:.2}",
        passing_rate
    );
}
