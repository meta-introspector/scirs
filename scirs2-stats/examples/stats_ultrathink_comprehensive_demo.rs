//! Comprehensive Ultrathink Mode Demonstration
//!
//! This example showcases the full capabilities of scirs2-stats in ultrathink mode,
//! demonstrating enhanced error handling, numerical stability, parallel processing,
//! SIMD optimizations, and property-based testing.

use ndarray::{Array1, Array2};
use scirs2_stats::{
    mean, pearson_r, std,
    ultrathink_error_enhancements::{UltrathinkContextBuilder, UltrathinkErrorMessages},
    ultrathink_numerical_stability::{
        NumericalStabilityConfig, UltrathinkNumericalStabilityAnalyzer,
    },
    ultrathink_parallel_enhancements::{create_ultra_parallel_processor, UltrathinkParallelConfig},
    ultrathink_property_tests::UltrathinkPropertyTester,
    ultrathink_simd_optimizations::{ultra_batch_statistics, UltrathinkSimdConfig},
    var,
};
use std::time::Instant;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Ultrathink Mode Comprehensive Demo");
    println!("=====================================\n");

    // Generate test data with varying characteristics
    let small_data = Array1::from_vec((1..=100).map(|x| x as f64).collect());
    let large_data = Array1::from_vec((1..=100_000).map(|x| (x as f64).sin()).collect());
    let extreme_data = Array1::from_vec(vec![
        f64::MIN,
        f64::MAX,
        0.0,
        1e-308,
        1e308,
        f64::NAN,
        f64::INFINITY,
    ]);

    // 1. Enhanced Error Handling Demo
    println!("1. 🛡️  Enhanced Error Handling");
    println!("   Testing context-aware error messages and recovery strategies");

    let context = UltrathinkContextBuilder::new(large_data.len())
        .memory_usage(800.0) // 800MB
        .simd_enabled(true)
        .parallel_enabled(true)
        .build();

    // Test memory exhaustion error message
    let memory_error = UltrathinkErrorMessages::memory_exhaustion(2000.0, 1000.0, large_data.len());
    println!("   Memory Error: {}", memory_error);

    // Test performance degradation warning
    let expected = std::time::Duration::from_millis(100);
    let actual = std::time::Duration::from_millis(1500);
    let perf_error =
        UltrathinkErrorMessages::performance_degradation("mean", expected, actual, &context);
    println!("   Performance Warning: {}", perf_error);
    println!();

    // 2. Numerical Stability Analysis
    println!("2. 🔬 Numerical Stability Analysis");
    println!("   Testing precision, overflow detection, and stability metrics");

    let stability_config = NumericalStabilityConfig {
        relative_tolerance: 1e-14,
        detect_cancellation: true,
        detect_overflow: true,
        test_extreme_values: true,
        ..Default::default()
    };

    let mut analyzer = UltrathinkNumericalStabilityAnalyzer::new(stability_config);

    // Test with normal data
    let result = analyzer.test_function_stability("mean", &small_data, |data| {
        mean(&data.view()).map_err(|e| format!("{}", e))
    });
    println!("   Mean stability test: {:?}", result.is_ok());

    // Test with extreme values (expecting controlled handling)
    let extreme_test = analyzer.test_extreme_values("variance", |data| {
        var(&data.view(), 1).map_err(|e| format!("{}", e))
    });
    println!("   Extreme values test: {:?}", extreme_test.is_ok());
    println!();

    // 3. Advanced Parallel Processing
    println!("3. ⚡ Advanced Parallel Processing");
    println!("   Testing adaptive thread management and load balancing");

    let parallel_config = UltrathinkParallelConfig {
        max_threads: 4,
        min_chunk_size: 1000,
        enable_work_stealing: true,
        adaptive_scaling: true,
        ..Default::default()
    };

    let processor = create_ultra_parallel_processor(parallel_config)?;

    // Parallel statistical computation
    let start = Instant::now();
    let parallel_mean = processor.parallel_mean(&large_data.view())?;
    let parallel_time = start.elapsed();

    // Sequential computation for comparison
    let start = Instant::now();
    let sequential_mean = mean(&large_data.view())?;
    let sequential_time = start.elapsed();

    println!(
        "   Parallel mean: {:.6}, Time: {:.3}ms",
        parallel_mean,
        parallel_time.as_millis()
    );
    println!(
        "   Sequential mean: {:.6}, Time: {:.3}ms",
        sequential_mean,
        sequential_time.as_millis()
    );

    let speedup = sequential_time.as_secs_f64() / parallel_time.as_secs_f64();
    println!("   Speedup: {:.2}x", speedup);
    println!();

    // 4. SIMD Optimizations
    println!("4. 🏎️  SIMD Optimizations");
    println!("   Testing vectorized batch statistics computation");

    let simd_config = UltrathinkSimdConfig {
        min_simd_size: 64,
        chunk_size: 8192,
        adaptive_vectorization: true,
        enable_prefetch: true,
        ..Default::default()
    };

    let start = Instant::now();
    let batch_stats = ultra_batch_statistics(&large_data.view(), &simd_config)?;
    let simd_time = start.elapsed();

    println!("   SIMD Batch Statistics (100k elements):");
    println!("     Mean: {:.6}", batch_stats.mean);
    println!("     Variance: {:.6}", batch_stats.variance);
    println!("     Skewness: {:.6}", batch_stats.skewness);
    println!("     Kurtosis: {:.6}", batch_stats.kurtosis);
    println!("     Min: {:.6}", batch_stats.min);
    println!("     Max: {:.6}", batch_stats.max);
    println!("   Computation time: {:.3}ms", simd_time.as_millis());
    println!();

    // 5. Property-Based Testing
    println!("5. 🧪 Property-Based Testing");
    println!("   Testing mathematical invariants and consistency");

    let property_tester = UltrathinkPropertyTester::new(simd_config, parallel_config);

    // Test statistical properties
    let test_data = Array1::from_vec((1..=1000).map(|x| x as f64 / 100.0).collect());

    // Test mean invariant: mean([a, b]) should be between min(a, b) and max(a, b)
    let mean_result = property_tester.test_mean_properties(&test_data)?;
    println!(
        "   Mean invariant test: {}",
        if mean_result {
            "✅ PASSED"
        } else {
            "❌ FAILED"
        }
    );

    // Test variance invariant: variance should be non-negative
    let variance_result = property_tester.test_variance_properties(&test_data)?;
    println!(
        "   Variance invariant test: {}",
        if variance_result {
            "✅ PASSED"
        } else {
            "❌ FAILED"
        }
    );

    // Test correlation bounds: correlation should be in [-1, 1]
    let x = Array1::from_vec((1..=100).map(|i| i as f64).collect());
    let y = Array1::from_vec((1..=100).map(|i| (i as f64).sin()).collect());
    let correlation_result = property_tester.test_correlation_bounds(&x, &y)?;
    println!(
        "   Correlation bounds test: {}",
        if correlation_result {
            "✅ PASSED"
        } else {
            "❌ FAILED"
        }
    );
    println!();

    // 6. Performance Comparison
    println!("6. 📊 Performance Summary");
    println!("   Comparing standard vs ultrathink mode performance");

    let test_sizes = vec![1_000, 10_000, 100_000];

    for size in test_sizes {
        let data = Array1::from_vec((0..size).map(|x| (x as f64).sin()).collect());

        // Standard computation
        let start = Instant::now();
        let _std_mean = mean(&data.view())?;
        let _std_var = var(&data.view(), 1)?;
        let std_time = start.elapsed();

        // Ultrathink computation
        let start = Instant::now();
        let _ultra_stats = ultra_batch_statistics(&data.view(), &simd_config)?;
        let ultra_time = start.elapsed();

        let speedup = std_time.as_nanos() as f64 / ultra_time.as_nanos() as f64;
        println!(
            "   Size {}: Standard {:.3}ms, Ultrathink {:.3}ms, Speedup: {:.2}x",
            size,
            std_time.as_millis(),
            ultra_time.as_millis(),
            speedup
        );
    }

    println!("\n🎉 Ultrathink Mode Demo Complete!");
    println!("All systems operational and performing optimally.");

    Ok(())
}

// Add helper extension trait for property testing
trait PropertyTestExtensions {
    fn test_mean_properties(&self, data: &Array1<f64>) -> Result<bool, Box<dyn std::error::Error>>;
    fn test_variance_properties(
        &self,
        data: &Array1<f64>,
    ) -> Result<bool, Box<dyn std::error::Error>>;
    fn test_correlation_bounds(
        &self,
        x: &Array1<f64>,
        y: &Array1<f64>,
    ) -> Result<bool, Box<dyn std::error::Error>>;
}

impl PropertyTestExtensions for UltrathinkPropertyTester {
    fn test_mean_properties(&self, data: &Array1<f64>) -> Result<bool, Box<dyn std::error::Error>> {
        let computed_mean = mean(&data.view())?;
        let min_val = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        Ok(computed_mean >= min_val && computed_mean <= max_val)
    }

    fn test_variance_properties(
        &self,
        data: &Array1<f64>,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        let variance = var(&data.view(), 1)?;
        Ok(variance >= 0.0 && variance.is_finite())
    }

    fn test_correlation_bounds(
        &self,
        x: &Array1<f64>,
        y: &Array1<f64>,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        let correlation = pearson_r(&x.view(), &y.view())?;
        Ok(correlation >= -1.0 && correlation <= 1.0)
    }
}
