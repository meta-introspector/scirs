//! Comprehensive cross-framework benchmarking example
//!
//! This example demonstrates how to use the cross-framework benchmarking capabilities
//! to compare SciRS2 optimizers against PyTorch and TensorFlow implementations.

use ndarray::{Array1, Array2};
use scirs2_optim::{
    adam::{Adam, AdamConfig},
    benchmarking::{cross_framework::*, BenchmarkResult, TestFunction},
    error::Result,
    optimizers::Optimizer,
    sgd::{SGDConfig, SGD},
};
use std::collections::HashMap;

fn main() -> Result<()> {
    println!("🚀 SciRS2 Comprehensive Cross-Framework Benchmark Suite");
    println!("=======================================================\n");

    // Setup cross-framework benchmark configuration
    let config = CrossFrameworkConfig {
        enable_pytorch: true,
        enable_tensorflow: true,
        python_path: "python3".to_string(),
        temp_dir: "/tmp/scirs2_benchmark_example".to_string(),
        precision: Precision::F64,
        max_iterations: 1000,
        tolerance: 1e-6,
        random_seed: 42,
        batch_sizes: vec![1, 32, 128],
        problem_dimensions: vec![10, 100, 1000],
        num_runs: 5,
    };

    // Create benchmark suite
    let mut benchmark = CrossFrameworkBenchmark::new(config)?;

    // Add standard optimization test functions
    benchmark.add_standard_test_functions();

    // Add custom test functions for more comprehensive evaluation
    add_custom_test_functions(&mut benchmark);

    // Prepare SciRS2 optimizers for benchmarking
    let scirs2_optimizers = prepare_scirs2_optimizers();

    println!("📊 Running comprehensive benchmark suite...");
    println!("This may take several minutes depending on your system.\n");

    // Run the benchmark
    let results = benchmark.run_comprehensive_benchmark(scirs2_optimizers)?;

    // Generate and display comprehensive report
    let report = benchmark.generate_comprehensive_report();
    println!("{}", report);

    // Save detailed results to file
    save_detailed_results(&results)?;

    // Generate visualization data
    generate_visualization_data(&results)?;

    // Run performance analysis
    run_performance_analysis(&results)?;

    println!("\n✅ Benchmark completed successfully!");
    println!("📄 Detailed results saved to: /tmp/scirs2_benchmark_results.json");
    println!("📈 Visualization data saved to: /tmp/scirs2_benchmark_plots.json");

    Ok(())
}

/// Add custom test functions for comprehensive evaluation
fn add_custom_test_functions(benchmark: &mut CrossFrameworkBenchmark<f64>) {
    // Himmelblau's function
    benchmark.add_test_function(TestFunction {
        name: "Himmelblau".to_string(),
        dimension: 2,
        function: Box::new(|x: &Array1<f64>| {
            let x1 = x[0];
            let x2 = x[1];
            (x1 * x1 + x2 - 11.0).powi(2) + (x1 + x2 * x2 - 7.0).powi(2)
        }),
        gradient: Box::new(|x: &Array1<f64>| {
            let x1 = x[0];
            let x2 = x[1];
            let dx1 = 4.0 * x1 * (x1 * x1 + x2 - 11.0) + 2.0 * (x1 + x2 * x2 - 7.0);
            let dx2 = 2.0 * (x1 * x1 + x2 - 11.0) + 4.0 * x2 * (x1 + x2 * x2 - 7.0);
            Array1::from_vec(vec![dx1, dx2])
        }),
        optimal_value: Some(0.0),
        optimal_point: Some(Array1::from_vec(vec![3.0, 2.0])),
    });

    // Ackley function (high-dimensional non-convex)
    benchmark.add_test_function(TestFunction {
        name: "Ackley".to_string(),
        dimension: 10,
        function: Box::new(|x: &Array1<f64>| {
            let n = x.len() as f64;
            let sum_sq = x.iter().map(|&xi| xi * xi).sum::<f64>();
            let sum_cos = x
                .iter()
                .map(|&xi| (2.0 * std::f64::consts::PI * xi).cos())
                .sum::<f64>();

            -20.0 * (-0.2 * (sum_sq / n).sqrt()).exp() - (sum_cos / n).exp()
                + 20.0
                + std::f64::consts::E
        }),
        gradient: Box::new(|x: &Array1<f64>| {
            let n = x.len() as f64;
            let sum_sq = x.iter().map(|&xi| xi * xi).sum::<f64>();
            let sum_cos = x
                .iter()
                .map(|&xi| (2.0 * std::f64::consts::PI * xi).cos())
                .sum::<f64>();

            let sqrt_term = (sum_sq / n).sqrt();
            let exp1 = (-0.2 * sqrt_term).exp();
            let exp2 = (sum_cos / n).exp();

            x.iter()
                .map(|&xi| {
                    let grad_term1 = 4.0 * exp1 * xi / (sqrt_term * n);
                    let grad_term2 =
                        (2.0 * std::f64::consts::PI * xi).sin() * 2.0 * std::f64::consts::PI * exp2
                            / n;
                    grad_term1 + grad_term2
                })
                .collect::<Array1<f64>>()
        }),
        optimal_value: Some(0.0),
        optimal_point: Some(Array1::zeros(10)),
    });

    // High-dimensional quadratic with condition number
    benchmark.add_test_function(TestFunction {
        name: "IllConditionedQuadratic".to_string(),
        dimension: 50,
        function: Box::new(|x: &Array1<f64>| {
            // Create ill-conditioned quadratic: sum_i (i * x_i^2)
            x.iter()
                .enumerate()
                .map(|(i, &xi)| (i + 1) as f64 * xi * xi)
                .sum()
        }),
        gradient: Box::new(|x: &Array1<f64>| {
            x.iter()
                .enumerate()
                .map(|(i, &xi)| 2.0 * (i + 1) as f64 * xi)
                .collect::<Array1<f64>>()
        }),
        optimal_value: Some(0.0),
        optimal_point: Some(Array1::zeros(50)),
    });
}

/// Prepare SciRS2 optimizers with various configurations
fn prepare_scirs2_optimizers() -> Vec<(
    String,
    Box<dyn Fn(&Array1<f64>, &Array1<f64>) -> Array1<f64>>,
)> {
    let mut optimizers: Vec<(
        String,
        Box<dyn Fn(&Array1<f64>, &Array1<f64>) -> Array1<f64>>,
    )> = Vec::new();

    // Adam optimizer with default settings
    {
        let config = AdamConfig {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            amsgrad: false,
        };
        let mut adam = Adam::new(config);

        optimizers.push((
            "Adam_default".to_string(),
            Box::new(move |params: &Array1<f64>, grads: &Array1<f64>| {
                adam.step(params, grads).unwrap_or_else(|_| params.clone())
            }),
        ));
    }

    // Adam with higher learning rate
    {
        let config = AdamConfig {
            learning_rate: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            amsgrad: false,
        };
        let mut adam_high_lr = Adam::new(config);

        optimizers.push((
            "Adam_high_lr".to_string(),
            Box::new(move |params: &Array1<f64>, grads: &Array1<f64>| {
                adam_high_lr
                    .step(params, grads)
                    .unwrap_or_else(|_| params.clone())
            }),
        ));
    }

    // Adam with weight decay (AdamW-style)
    {
        let config = AdamConfig {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            amsgrad: false,
        };
        let mut adam_wd = Adam::new(config);

        optimizers.push((
            "Adam_weight_decay".to_string(),
            Box::new(move |params: &Array1<f64>, grads: &Array1<f64>| {
                adam_wd
                    .step(params, grads)
                    .unwrap_or_else(|_| params.clone())
            }),
        ));
    }

    // AMSGrad variant
    {
        let config = AdamConfig {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            amsgrad: true,
        };
        let mut amsgrad = Adam::new(config);

        optimizers.push((
            "AMSGrad".to_string(),
            Box::new(move |params: &Array1<f64>, grads: &Array1<f64>| {
                amsgrad
                    .step(params, grads)
                    .unwrap_or_else(|_| params.clone())
            }),
        ));
    }

    // SGD with momentum
    {
        let config = SGDConfig {
            learning_rate: 0.01,
            momentum: 0.9,
            dampening: 0.0,
            weight_decay: 0.0,
            nesterov: false,
        };
        let mut sgd_momentum = SGD::new(config);

        optimizers.push((
            "SGD_momentum".to_string(),
            Box::new(move |params: &Array1<f64>, grads: &Array1<f64>| {
                sgd_momentum
                    .step(params, grads)
                    .unwrap_or_else(|_| params.clone())
            }),
        ));
    }

    // Nesterov SGD
    {
        let config = SGDConfig {
            learning_rate: 0.01,
            momentum: 0.9,
            dampening: 0.0,
            weight_decay: 0.0,
            nesterov: true,
        };
        let mut nesterov = SGD::new(config);

        optimizers.push((
            "Nesterov_SGD".to_string(),
            Box::new(move |params: &Array1<f64>, grads: &Array1<f64>| {
                nesterov
                    .step(params, grads)
                    .unwrap_or_else(|_| params.clone())
            }),
        ));
    }

    optimizers
}

/// Save detailed results to JSON file
fn save_detailed_results(results: &[CrossFrameworkBenchmarkResult<f64>]) -> Result<()> {
    use std::fs::File;
    use std::io::Write;

    // Simplified serialization for now - would use serde_json in full implementation
    let json_data = format!("{:#?}", results);

    let mut file = File::create("/tmp/scirs2_benchmark_results.json").map_err(|e| {
        scirs2_optim::error::OptimError::InvalidConfig(format!("File creation failed: {}", e))
    })?;

    file.write_all(json_data.as_bytes()).map_err(|e| {
        scirs2_optim::error::OptimError::InvalidConfig(format!("File write failed: {}", e))
    })?;

    println!("💾 Detailed results saved to /tmp/scirs2_benchmark_results.json");
    Ok(())
}

/// Generate visualization data for plotting
fn generate_visualization_data(results: &[CrossFrameworkBenchmarkResult<f64>]) -> Result<()> {
    use std::fs::File;
    use std::io::Write;

    // Simplified visualization data generation for now
    let mut plot_data = Vec::new();

    for result in results {
        let mut convergence_summary = Vec::new();

        for (optimizer_id, summary) in &result.optimizer_results {
            convergence_summary.push(format!(
                "optimizer: {}, framework: {}, success_rate: {:.2}, avg_iterations: {:.1}",
                optimizer_id.name,
                optimizer_id.framework,
                summary.success_rate,
                summary.mean_iterations
            ));
        }

        plot_data.push(format!(
            "test_function: {}, problem_dim: {}, batch_size: {}, optimizers: {:?}, p_value: {:.6}",
            result.function_name,
            result.problem_dim,
            result.batch_size,
            convergence_summary,
            result.statistical_comparison.anova_results.p_value
        ));
    }

    // Simplified serialization - would use proper JSON in full implementation
    let json_data = format!("[\n{}\n]", plot_data.join(",\n"));

    let mut file = File::create("/tmp/scirs2_benchmark_plots.json").map_err(|e| {
        scirs2_optim::error::OptimError::InvalidConfig(format!("File creation failed: {}", e))
    })?;

    file.write_all(json_data.as_bytes()).map_err(|e| {
        scirs2_optim::error::OptimError::InvalidConfig(format!("File write failed: {}", e))
    })?;

    println!("📈 Visualization data saved to /tmp/scirs2_benchmark_plots.json");
    Ok(())
}

/// Run detailed performance analysis
fn run_performance_analysis(results: &[CrossFrameworkBenchmarkResult<f64>]) -> Result<()> {
    println!("\n🔍 DETAILED PERFORMANCE ANALYSIS");
    println!("==================================");

    let mut framework_wins = HashMap::new();
    let mut optimizer_performance = HashMap::new();

    for result in results {
        println!(
            "\n📊 Analysis for {} ({}D, batch={})",
            result.function_name, result.problem_dim, result.batch_size
        );

        // Find best performing optimizer
        if let Some((best_optimizer, best_score)) = result.performance_ranking.first() {
            println!(
                "🏆 Best performer: {} (score: {:.6})",
                best_optimizer, best_score
            );

            // Track framework wins
            *framework_wins
                .entry(best_optimizer.framework.clone())
                .or_insert(0) += 1;

            // Track optimizer performance
            optimizer_performance
                .entry(best_optimizer.clone())
                .or_insert(Vec::new())
                .push(*best_score);
        }

        // Analyze convergence characteristics
        analyze_convergence_characteristics(result);

        // Check statistical significance
        let anova = &result.statistical_comparison.anova_results;
        if anova.p_value < 0.05 {
            println!(
                "📈 Statistically significant differences found (p={:.6})",
                anova.p_value
            );
            analyze_pairwise_comparisons(result);
        } else {
            println!(
                "📊 No statistically significant differences (p={:.6})",
                anova.p_value
            );
        }
    }

    // Overall framework comparison
    println!("\n🌟 OVERALL FRAMEWORK COMPARISON");
    println!("=================================");

    for (framework, wins) in &framework_wins {
        let win_rate = *wins as f64 / results.len() as f64 * 100.0;
        println!("{}: {} wins ({:.1}%)", framework, wins, win_rate);
    }

    // Top performing optimizers
    println!("\n🚀 TOP PERFORMING OPTIMIZERS");
    println!("===============================");

    let mut avg_scores: Vec<_> = optimizer_performance
        .iter()
        .map(|(optimizer, scores)| {
            let avg_score = scores.iter().sum::<f64>() / scores.len() as f64;
            (optimizer.clone(), avg_score, scores.len())
        })
        .collect();

    avg_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    for (i, (optimizer, avg_score, test_count)) in avg_scores.iter().take(10).enumerate() {
        println!(
            "{}. {} - Avg Score: {:.6} ({} tests)",
            i + 1,
            optimizer,
            avg_score,
            test_count
        );
    }

    Ok(())
}

/// Analyze convergence characteristics for a specific test
fn analyze_convergence_characteristics(result: &CrossFrameworkBenchmarkResult<f64>) {
    println!("  📉 Convergence Analysis:");

    for (optimizer_id, summary) in &result.optimizer_results {
        if summary.successful_runs > 0 {
            let success_rate = summary.success_rate * 100.0;
            let avg_time_ms = summary.mean_convergence_time.as_millis();

            println!(
                "    {} - Success: {:.1}%, Avg Time: {}ms, Avg Iters: {:.1}",
                optimizer_id.name, success_rate, avg_time_ms, summary.mean_iterations
            );

            // Analyze convergence speed
            if !summary.convergence_curves.is_empty() {
                let avg_curve_length = summary
                    .convergence_curves
                    .iter()
                    .map(|curve| curve.len())
                    .sum::<usize>() as f64
                    / summary.convergence_curves.len() as f64;

                if avg_curve_length < 100.0 {
                    println!("      ⚡ Fast convergence");
                } else if avg_curve_length < 500.0 {
                    println!("      🐢 Moderate convergence");
                } else {
                    println!("      🐌 Slow convergence");
                }
            }
        } else {
            println!("    {} - ❌ Failed to converge", optimizer_id.name);
        }
    }
}

/// Analyze pairwise statistical comparisons
fn analyze_pairwise_comparisons(result: &CrossFrameworkBenchmarkResult<f64>) {
    println!("  🔬 Pairwise Comparisons:");

    let mut significant_pairs = Vec::new();

    for ((opt1, opt2), test_result) in &result.statistical_comparison.final_value_tests {
        if test_result.is_significant {
            significant_pairs.push((opt1, opt2, test_result.p_value));
        }
    }

    if significant_pairs.is_empty() {
        println!("    No significant pairwise differences found");
    } else {
        significant_pairs
            .sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        for (opt1, opt2, p_value) in significant_pairs.iter().take(5) {
            println!("    {} vs {} - p={:.6}", opt1.name, opt2.name, p_value);
        }

        if significant_pairs.len() > 5 {
            println!(
                "    ... and {} more significant comparisons",
                significant_pairs.len() - 5
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_setup() {
        let config = CrossFrameworkConfig::default();
        let benchmark = CrossFrameworkBenchmark::new(config);
        assert!(benchmark.is_ok());
    }

    #[test]
    fn test_optimizer_preparation() {
        let optimizers = prepare_scirs2_optimizers();
        assert!(!optimizers.is_empty());
        assert!(optimizers.len() >= 5); // Should have at least 5 optimizer variants
    }

    #[test]
    fn test_custom_test_functions() {
        let mut benchmark = CrossFrameworkBenchmark::new(CrossFrameworkConfig::default()).unwrap();
        add_custom_test_functions(&mut benchmark);
        // This should not panic and should add the custom functions
    }
}
