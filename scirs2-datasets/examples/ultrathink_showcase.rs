//! Ultrathink Mode Showcase
//!
//! This example demonstrates the ultra-sophisticated enhancements added to scirs2-datasets,
//! including advanced analytics, GPU optimization, and adaptive streaming processing.

use ndarray::{Array1, Array2};
use scirs2_datasets::{
    // Ultra-advanced analytics
    analyze_dataset_ultra,
    benchmark_ultra_performance,
    // Adaptive streaming
    create_adaptive_engine,
    create_adaptive_engine_with_config,
    // Ultra-GPU optimization
    generate_ultra_matrix,
    // Core functionality
    make_classification,
    quick_quality_assessment,
    AdaptiveStreamConfig,
    AdaptiveStreamingEngine,
    ChunkMetadata,
    DataCharacteristics,
    Dataset,
    GpuBackend,
    GpuContext,
    StatisticalMoments,
    StreamChunk,
    TrendDirection,
    TrendIndicators,
    UltraDatasetAnalyzer,
    UltraGpuOptimizer,
};
use std::time::{Duration, Instant};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 SciRS2-Datasets Ultrathink Mode Showcase");
    println!("===========================================\n");

    // Create a sample dataset for demonstration
    let dataset = create_sample_dataset()?;
    println!(
        "📊 Created sample dataset: {} samples, {} features",
        dataset.n_samples(),
        dataset.n_features()
    );

    // Demonstrate ultra-advanced analytics
    demonstrate_ultra_analytics(&dataset)?;

    // Demonstrate ultra-GPU optimization
    demonstrate_ultra_gpu_optimization()?;

    // Demonstrate adaptive streaming
    demonstrate_adaptive_streaming(&dataset)?;

    println!("\n✅ Ultrathink mode demonstration completed successfully!");
    Ok(())
}

/// Create a sample dataset for demonstration
#[allow(dead_code)]
fn create_sample_dataset() -> Result<Dataset, Box<dyn std::error::Error>> {
    println!("🔧 Generating sample classification dataset...");

    let dataset = make_classification(
        1000,     // n_samples
        10,       // n_features
        3,        // n_classes
        2,        // n_clusters_per_class
        5,        // n_informative
        Some(42), // random_state
    )?;

    Ok(dataset)
}

/// Demonstrate ultra-advanced analytics capabilities
#[allow(dead_code)]
fn demonstrate_ultra_analytics(dataset: &Dataset) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🧠 Ultra-Advanced Analytics Demonstration");
    println!("==========================================");

    // Quick quality assessment
    println!("📈 Running quick quality assessment...");
    let quick_quality = quick_quality_assessment(dataset)?;
    println!("   Quality Score: {:.3}", quick_quality);

    // Comprehensive ultra-analysis
    println!("🔬 Running comprehensive ultra-analysis...");
    let start_time = Instant::now();

    let analyzer = UltraDatasetAnalyzer::new()
        .with_gpu(true)
        .with_ultra_precision(true)
        .with_significance_threshold(0.01);

    let metrics = analyzer.analyze_dataset_quality(dataset)?;
    let analysis_time = start_time.elapsed();

    println!("   Analysis completed in: {:?}", analysis_time);
    println!("   Complexity Score: {:.3}", metrics.complexity_score);
    println!("   Entropy: {:.3}", metrics.entropy);
    println!("   Outlier Score: {:.3}", metrics.outlier_score);
    println!("   ML Quality Score: {:.3}", metrics.ml_quality_score);

    // Display normality assessment
    println!("   Normality Assessment:");
    println!(
        "     Overall Normality: {:.3}",
        metrics.normality_assessment.overall_normality
    );
    println!(
        "     Shapiro-Wilk (avg): {:.3}",
        metrics
            .normality_assessment
            .shapiro_wilk_scores
            .mean()
            .unwrap_or(0.0)
    );

    // Display correlation insights
    println!("   Correlation Insights:");
    println!(
        "     Feature Importance (top 3): {:?}",
        metrics
            .correlation_insights
            .feature_importance
            .iter()
            .take(3)
            .map(|&x| format!("{:.3}", x))
            .collect::<Vec<_>>()
    );

    Ok(())
}

/// Demonstrate ultra-GPU optimization capabilities
#[allow(dead_code)]
fn demonstrate_ultra_gpu_optimization() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n⚡ Ultra-GPU Optimization Demonstration");
    println!("=====================================");

    // Create GPU context (falls back to CPU if no GPU available)
    println!("🔧 Initializing GPU context...");
    let gpu_context = GpuContext::new(GpuBackend::Cpu)?; // Using CPU backend for demo
    println!("   Backend: {:?}", gpu_context.backend());

    // Create ultra-GPU optimizer
    let optimizer = UltraGpuOptimizer::new()
        .with_adaptive_kernels(true)
        .with_memory_prefetch(true)
        .with_multi_gpu(false) // Single GPU for demo
        .with_auto_tuning(true);

    // Generate ultra-optimized matrix
    println!("🔥 Generating ultra-optimized matrix...");
    let start_time = Instant::now();
    let matrix = optimizer.generate_ultra_optimized_matrix(
        &gpu_context,
        500,      // rows
        200,      // cols
        "normal", // distribution
    )?;
    let generation_time = start_time.elapsed();

    println!(
        "   Generated {}x{} matrix in: {:?}",
        matrix.nrows(),
        matrix.ncols(),
        generation_time
    );
    println!(
        "   Matrix stats: mean={:.3}, std={:.3}",
        matrix.mean().unwrap_or(0.0),
        matrix.var(1.0).sqrt()
    );

    // Benchmark performance
    println!("📊 Running performance benchmarks...");
    let data_shapes = vec![(100, 50), (500, 200), (1000, 500)];
    let benchmark_results =
        optimizer.benchmark_performance(&gpu_context, "matrix_generation", &data_shapes)?;

    println!("   Benchmark Results:");
    println!(
        "     Best Speedup: {:.2}x",
        benchmark_results.best_speedup()
    );
    println!(
        "     Average Speedup: {:.2}x",
        benchmark_results.average_speedup()
    );
    println!(
        "     Total Memory Usage: {:.1} MB",
        benchmark_results.total_memory_usage()
    );

    Ok(())
}

/// Demonstrate adaptive streaming capabilities
#[allow(dead_code)]
fn demonstrate_adaptive_streaming(dataset: &Dataset) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🌊 Adaptive Streaming Demonstration");
    println!("===================================");

    // Configure streaming engine
    let config = AdaptiveStreamConfig {
        max_buffer_size: 10 * 1024 * 1024, // 10MB
        batch_size: 100,
        adaptive_threshold: 0.8,
        ml_optimization: true,
        quality_check_interval: Duration::from_secs(5),
    };

    println!("🔧 Initializing adaptive streaming engine...");
    let mut engine = create_adaptive_engine_with_config(config);

    // Simulate streaming data
    println!("📡 Simulating data stream...");
    let data = &dataset.data;
    let chunk_size = 20;
    let num_chunks = (data.nrows() / chunk_size).min(10); // Limit for demo

    let mut total_processed = 0;
    let start_time = Instant::now();

    for i in 0..num_chunks {
        let start_row = i * chunk_size;
        let end_row = (start_row + chunk_size).min(data.nrows());

        // Create chunk from dataset slice
        let chunk_data = data.slice(ndarray::s![start_row..end_row, ..]).to_owned();

        let chunk = StreamChunk {
            data: chunk_data,
            timestamp: Instant::now(),
            metadata: ChunkMetadata {
                source_id: format!("demo_source_{}", i),
                sequence_number: i as u64,
                characteristics: DataCharacteristics {
                    moments: StatisticalMoments {
                        mean: 0.0,
                        variance: 1.0,
                        skewness: 0.0,
                        kurtosis: 0.0,
                    },
                    entropy: 1.0,
                    trend: TrendIndicators {
                        linear_slope: 0.1,
                        trend_strength: 0.5,
                        direction: TrendDirection::Increasing,
                        seasonality: 0.2,
                    },
                    anomaly_score: 0.1,
                },
            },
            quality_score: 0.9,
        };

        // Process chunk
        let results = engine.process_stream(chunk)?;
        total_processed += results.len();

        if !results.is_empty() {
            println!(
                "   Processed batch {}: {} datasets generated",
                i + 1,
                results.len()
            );
        }
    }

    let streaming_time = start_time.elapsed();

    println!("   Streaming completed in: {:?}", streaming_time);
    println!("   Total datasets processed: {}", total_processed);

    // Get performance metrics
    println!("📈 Getting performance metrics...");
    let perf_metrics = engine.get_performance_metrics()?;
    println!("   Processing Latency: {:?}", perf_metrics.latency);
    println!("   Throughput: {:.1} chunks/sec", perf_metrics.throughput);
    println!(
        "   Memory Efficiency: {:.1}%",
        perf_metrics.memory_efficiency * 100.0
    );

    // Get quality metrics
    let quality_metrics = engine.get_quality_metrics()?;
    println!("   Quality Metrics:");
    println!(
        "     Integrity: {:.1}%",
        quality_metrics.integrity_score * 100.0
    );
    println!(
        "     Completeness: {:.1}%",
        quality_metrics.completeness_score * 100.0
    );
    println!(
        "     Overall Quality: {:.1}%",
        quality_metrics.overall_score * 100.0
    );

    // Get buffer statistics
    let buffer_stats = engine.get_buffer_statistics()?;
    println!("   Buffer Statistics:");
    println!("     Utilization: {:.1}%", buffer_stats.utilization * 100.0);
    println!("     Memory Usage: {} bytes", buffer_stats.memory_usage);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_dataset_creation() {
        let result = create_sample_dataset();
        assert!(result.is_ok());
        let dataset = result.unwrap();
        assert_eq!(dataset.n_samples(), 1000);
        assert_eq!(dataset.n_features(), 10);
    }

    #[test]
    fn test_ultra_analytics_integration() {
        let dataset = create_sample_dataset().unwrap();
        let result = demonstrate_ultra_analytics(&dataset);
        assert!(result.is_ok());
    }

    #[test]
    fn test_gpu_optimization_integration() {
        let result = demonstrate_ultra_gpu_optimization();
        assert!(result.is_ok());
    }

    #[test]
    fn test_adaptive_streaming_integration() {
        let dataset = create_sample_dataset().unwrap();
        let result = demonstrate_adaptive_streaming(&dataset);
        assert!(result.is_ok());
    }
}
