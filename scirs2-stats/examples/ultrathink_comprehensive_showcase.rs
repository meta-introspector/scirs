//! Comprehensive Ultrathink Mode Showcase
//!
//! This example demonstrates the full capabilities of the ultrathink mode in scirs2-stats,
//! including SIMD optimizations, parallel processing, numerical stability testing,
//! and the unified processing framework.

use ndarray::{array, Array1, Array2};
use scirs2_stats::{
    create_ultrathink_processor, OptimizationMode, UltrathinkProcessorConfig,
    UltrathinkSimdConfig, UltrathinkParallelConfig, create_numerical_stability_analyzer,
    NumericalStabilityConfig, ultrathink_parallel_enhancements::MatrixOperationType,
    ultrathink_unified_processor::{UltrathinkMatrixOperation, UltrathinkTimeSeriesOperation},
};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Ultrathink Mode Comprehensive Showcase");
    println!("==========================================\n");

    // Demonstrate basic ultrathink processing
    demonstrate_basic_ultrathink()?;
    
    // Demonstrate advanced configurations
    demonstrate_advanced_configurations()?;
    
    // Demonstrate numerical stability testing
    demonstrate_stability_testing()?;
    
    // Demonstrate matrix operations
    demonstrate_matrix_operations()?;
    
    // Demonstrate time series processing
    demonstrate_time_series_processing()?;
    
    // Demonstrate performance analytics
    demonstrate_performance_analytics()?;
    
    println!("\n✅ Ultrathink mode showcase completed successfully!");

    Ok(())
}

fn demonstrate_basic_ultrathink() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Basic Ultrathink Processing");
    println!("------------------------------");

    // Create a unified ultrathink processor with default settings
    let mut processor = create_ultrathink_processor();

    // Generate sample data
    let data = generate_sample_data(10000);
    println!("Generated {} data points", data.len());

    let start_time = Instant::now();
    
    // Process comprehensive statistics using ultrathink mode
    let result = processor.process_comprehensive_statistics(&data.view())?;
    
    let processing_time = start_time.elapsed();
    
    println!("Processing completed in {:?}", processing_time);
    println!("Statistics computed:");
    println!("  Mean: {:.6}", result.statistics.mean);
    println!("  Std Dev: {:.6}", result.statistics.std_dev);
    println!("  Min: {:.6}", result.statistics.min);
    println!("  Max: {:.6}", result.statistics.max);
    println!("  Skewness: {:.6}", result.statistics.skewness);
    println!("  Kurtosis: {:.6}", result.statistics.kurtosis);
    
    println!("Processing metrics:");
    println!("  Strategy used: {:?}", result.processing_metrics.strategy_used);
    println!("  SIMD enabled: {}", result.processing_metrics.simd_enabled);
    println!("  Parallel enabled: {}", result.processing_metrics.parallel_enabled);
    println!("  Memory usage: {:.2} MB", result.processing_metrics.memory_usage_mb);
    
    if !result.recommendations.is_empty() {
        println!("Recommendations:");
        for recommendation in &result.recommendations {
            println!("  • {}", recommendation);
        }
    }

    println!();
    Ok(())
}

fn demonstrate_advanced_configurations() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚙️  Advanced Configuration Options");
    println!("----------------------------------");

    // Create different processor configurations
    let configs = vec![
        ("Performance Optimized", UltrathinkProcessorConfig {
            optimization_mode: OptimizationMode::Performance,
            enable_stability_testing: false,
            simd_config: UltrathinkSimdConfig {
                min_simd_size: 32,
                adaptive_vectorization: true,
                enable_prefetch: true,
                ..Default::default()
            },
            ..Default::default()
        }),
        ("Accuracy Focused", UltrathinkProcessorConfig {
            optimization_mode: OptimizationMode::Accuracy,
            enable_stability_testing: true,
            stability_config: NumericalStabilityConfig {
                relative_tolerance: 1e-15,
                detect_cancellation: true,
                detect_overflow: true,
                ..Default::default()
            },
            ..Default::default()
        }),
        ("Balanced", UltrathinkProcessorConfig {
            optimization_mode: OptimizationMode::Balanced,
            enable_stability_testing: true,
            enable_performance_monitoring: true,
            ..Default::default()
        }),
        ("Adaptive", UltrathinkProcessorConfig {
            optimization_mode: OptimizationMode::Adaptive,
            enable_stability_testing: true,
            enable_performance_monitoring: true,
            ..Default::default()
        }),
    ];

    let data = generate_sample_data(5000);

    for (name, config) in configs {
        let mut processor = scirs2_stats::ultrathink_unified_processor::UltrathinkUnifiedProcessor::new(config);
        
        let start_time = Instant::now();
        let result = processor.process_comprehensive_statistics(&data.view())?;
        let processing_time = start_time.elapsed();
        
        println!("{} Configuration:", name);
        println!("  Processing time: {:?}", processing_time);
        println!("  Strategy: {:?}", result.processing_metrics.strategy_used);
        println!("  Stability tested: {}", result.processing_metrics.stability_tested);
        
        if let Some(ref stability_report) = result.stability_report {
            println!("  Stability: {}", if stability_report.all_passed() { "✅ All tests passed" } else { "⚠️  Issues detected" });
        }
        
        println!();
    }

    Ok(())
}

fn demonstrate_stability_testing() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 Numerical Stability Testing");
    println!("------------------------------");

    let mut stability_analyzer = create_numerical_stability_analyzer();

    // Test with different data characteristics
    let test_cases = vec![
        ("Normal distribution", generate_normal_data(1000)),
        ("Data with outliers", generate_data_with_outliers(1000)),
        ("Very small values", generate_small_values_data(1000)),
        ("Large dynamic range", generate_large_range_data(1000)),
    ];

    for (name, data) in test_cases {
        println!("Testing: {}", name);
        
        let report = stability_analyzer.analyze_statistical_stability(&data.view());
        
        println!("  Overall stability: {}", if report.all_passed() { "✅ PASS" } else { "❌ FAIL" });
        
        let all_issues = report.get_all_issues();
        if !all_issues.is_empty() {
            println!("  Issues found:");
            for issue in all_issues.iter().take(3) { // Show first 3 issues
                println!("    • {}", issue);
            }
            if all_issues.len() > 3 {
                println!("    ... and {} more", all_issues.len() - 3);
            }
        }
        
        let all_warnings = report.get_all_warnings();
        if !all_warnings.is_empty() {
            println!("  Warnings ({}):", all_warnings.len());
            for warning in all_warnings.iter().take(2) { // Show first 2 warnings
                println!("    • {}", warning);
            }
        }
        
        println!();
    }

    Ok(())
}

fn demonstrate_matrix_operations() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔢 Matrix Operations with Ultrathink");
    println!("-------------------------------------");

    let mut processor = create_ultrathink_processor();

    // Generate sample matrix data
    let matrix_data = generate_sample_matrix(100, 20);
    println!("Generated {}x{} matrix", matrix_data.dim().0, matrix_data.dim().1);

    // Test different matrix operations
    let operations = vec![
        ("Covariance Matrix", UltrathinkMatrixOperation::Covariance),
        ("Correlation Matrix", UltrathinkMatrixOperation::Correlation),
    ];

    for (name, operation) in operations {
        println!("\nComputing {}:", name);
        
        let start_time = Instant::now();
        let result = processor.process_matrix_operations(&matrix_data.view(), operation)?;
        let processing_time = start_time.elapsed();
        
        println!("  Processing time: {:?}", processing_time);
        println!("  Result matrix shape: {:?}", result.matrix.dim());
        println!("  Strategy used: {:?}", result.processing_metrics.strategy_used);
        println!("  Memory usage: {:.2} MB", result.processing_metrics.memory_usage_mb);
        
        // Show a sample of the result matrix (top-left 3x3)
        let (rows, cols) = result.matrix.dim();
        println!("  Sample (top-left 3x3):");
        for i in 0..3.min(rows) {
            print!("    [");
            for j in 0..3.min(cols) {
                print!(" {:8.4}", result.matrix[[i, j]]);
            }
            println!(" ]");
        }
    }

    println!();
    Ok(())
}

fn demonstrate_time_series_processing() -> Result<(), Box<dyn std::error::Error>> {
    println!("📈 Time Series Processing");
    println!("-------------------------");

    let mut processor = create_ultrathink_processor();

    // Generate time series data
    let time_series = generate_time_series_data(5000);
    println!("Generated time series with {} points", time_series.len());

    let window_sizes = vec![50, 100, 200];
    let operations = vec![UltrathinkTimeSeriesOperation::MovingWindow];

    for window_size in window_sizes {
        println!("\nMoving window analysis (window size: {}):", window_size);
        
        let start_time = Instant::now();
        let result = processor.process_time_series(&time_series.view(), window_size, &operations)?;
        let processing_time = start_time.elapsed();
        
        println!("  Processing time: {:?}", processing_time);
        println!("  Strategy used: {:?}", result.processing_metrics.strategy_used);
        println!("  Number of results: {}", result.results.len());
        
        if !result.results.is_empty() {
            let first_result = &result.results[0];
            println!("  Moving statistics computed: {} windows", first_result.means.len());
            
            if !first_result.means.is_empty() {
                println!("  First few moving averages:");
                for (i, mean) in first_result.means.iter().take(5).enumerate() {
                    println!("    Window {}: {:.4}", i + 1, mean);
                }
            }
        }
    }

    println!();
    Ok(())
}

fn demonstrate_performance_analytics() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Performance Analytics");
    println!("------------------------");

    let mut processor = create_ultrathink_processor();

    // Perform several operations to build performance history
    let data_sizes = vec![1000, 5000, 10000, 50000];
    
    println!("Running benchmark operations...");
    for (i, &size) in data_sizes.iter().enumerate() {
        let data = generate_sample_data(size);
        let _result = processor.process_comprehensive_statistics(&data.view())?;
        println!("  Completed operation {} of {} (size: {})", i + 1, data_sizes.len(), size);
    }

    // Get performance analytics
    let analytics = processor.get_performance_analytics();
    
    println!("\nPerformance Analytics:");
    println!("  Total operations: {}", analytics.total_operations);
    println!("  Average processing time: {:.2} ms", analytics.average_processing_time_ms);
    println!("  SIMD usage rate: {:.1}%", analytics.simd_usage_rate * 100.0);
    println!("  Parallel usage rate: {:.1}%", analytics.parallel_usage_rate * 100.0);
    println!("  Average data size: {:.0} elements", analytics.average_data_size);
    println!("  Optimization effectiveness: {:.1}%", analytics.optimization_effectiveness * 100.0);
    
    if !analytics.recommendations.is_empty() {
        println!("  Recommendations:");
        for recommendation in &analytics.recommendations {
            println!("    • {}", recommendation);
        }
    }

    println!();
    Ok(())
}

// Helper functions for generating test data

fn generate_sample_data(n: usize) -> Array1<f64> {
    use std::f64::consts::PI;
    
    (0..n)
        .map(|i| {
            let x = i as f64 / n as f64;
            // Mix of deterministic and pseudo-random components
            (x * 2.0 * PI).sin() + 0.1 * (x * 13.0 * PI).cos() + 0.05 * (i as f64 * 0.1).sin()
        })
        .collect()
}

fn generate_normal_data(n: usize) -> Array1<f64> {
    // Simple Box-Muller approximation for normal distribution
    (0..n)
        .map(|i| {
            let u1 = (i as f64 + 1.0) / (n as f64 + 2.0);
            let u2 = ((i * 7) % n) as f64 / n as f64;
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        })
        .collect()
}

fn generate_data_with_outliers(n: usize) -> Array1<f64> {
    let mut data = generate_normal_data(n);
    // Add some outliers
    for i in (0..n).step_by(n / 10) {
        data[i] *= 10.0; // Make every 10th value an outlier
    }
    data
}

fn generate_small_values_data(n: usize) -> Array1<f64> {
    generate_normal_data(n).mapv(|x| x * 1e-12) // Very small values
}

fn generate_large_range_data(n: usize) -> Array1<f64> {
    (0..n)
        .map(|i| {
            if i % 100 == 0 {
                1e12 // Very large values occasionally
            } else {
                (i as f64).sin() * 1e-6 // Very small values mostly
            }
        })
        .collect()
}

fn generate_sample_matrix(rows: usize, cols: usize) -> Array2<f64> {
    use std::f64::consts::PI;
    
    Array2::from_shape_fn((rows, cols), |(i, j)| {
        let x = i as f64 / rows as f64;
        let y = j as f64 / cols as f64;
        (x * 2.0 * PI).sin() * (y * 2.0 * PI).cos() + 0.1 * ((i + j) as f64).sin()
    })
}

fn generate_time_series_data(n: usize) -> Array1<f64> {
    use std::f64::consts::PI;
    
    (0..n)
        .map(|i| {
            let t = i as f64 / 100.0; // Time scale
            // Trend + seasonal + noise
            0.1 * t + (t * 2.0 * PI / 365.0).sin() + 0.2 * (t * 2.0 * PI / 7.0).sin() + 0.05 * (t * 17.0).sin()
        })
        .collect()
}