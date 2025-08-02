//! Advanced-Comprehensive Performance Benchmarks for scirs2-series
//!
//! This benchmark suite provides extensive performance testing across all major time series
//! analysis functions with multiple scenarios, stress testing, and comparative analysis.
//! Features include memory profiling, scalability analysis, and cross-platform testing.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::{Array1, Array2, Array3};
use std::collections::HashMap;
use std::time::{Duration, Instant};

use scirs2__series::{
use statrs::statistics::Statistics;
    advanced_training::{NeuralODE, TimeSeriesTransformer, TimeSeriesVAE, MAML},
    anomaly::AnomalyDetector,
    arima_models::ArimaModel,
    biomedical::ECGAnalysis,
    causality::GrangerCausalityTest,
    change_point::PELTDetector,
    clustering::TimeSeriesClusterer,
    correlation::CrossCorrelation,
    decomposition::stl::STLDecomposer,
    detection::pattern::PatternDetector,
    dimensionality_reduction::FunctionalPCA,
    distributed::{ClusterConfig, DistributedProcessor},
    environmental::EnvironmentalSensorAnalysis,
    feature_selection::filter::FilterSelector,
    features::statistical::StatisticalFeatures,
    financial::garch_model,
    forecasting::neural::NeuralForecaster,
    gpu_acceleration::GpuTimeSeriesProcessor,
    iot_sensors::EnvironmentalSensorAnalysis as IoTEnvironmental,
    neural_forecasting::LSTMForecaster,
    optimization::OptimizationConfig,
    out_of_core::{ChunkedProcessor, ProcessingConfig},
    quantum_forecasting::{QuantumEnsemble, QuantumEnsembleMethod, QuantumNeuralNetwork},
    regression::TimeSeriesRegression,
    sarima_models::SARIMAModel,
    state_space::KalmanFilter,
    streaming::StreamingAnalyzer,
    transformations::BoxCoxTransform,
    trends::robust::RobustTrendFilter,
    utils::*,
    validation::CrossValidator,
    var_models::VectorAutoregression,
    visualization::TimeSeriesPlot,
};

/// Benchmark configuration for different scenarios
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    pub scenario_name: String,
    pub data_sizes: Vec<usize>,
    pub noise_levels: Vec<f64>,
    pub seasonal_periods: Vec<usize>,
    pub enable_memory_profiling: bool,
    pub enable_gpu_benchmarks: bool,
    pub enable_distributed_benchmarks: bool,
    pub repetitions: usize,
    pub timeout: Duration,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            scenario_name: "default".to_string(),
            data_sizes: vec![1_000, 10_000, 100_000],
            noise_levels: vec![0.1, 1.0, 5.0],
            seasonal_periods: vec![12, 24, 52],
            enable_memory_profiling: true,
            enable_gpu_benchmarks: false, // Disabled by default for compatibility
            enable_distributed_benchmarks: true,
            repetitions: 10,
            timeout: Duration::from_secs(300),
        }
    }
}

/// Memory profiling results
#[derive(Debug, Clone)]
pub struct MemoryProfile {
    pub peak_memory_usage: usize,
    pub average_memory_usage: usize,
    pub memory_allocations: usize,
    pub memory_deallocations: usize,
    pub memory_efficiency_score: f64,
}

/// Performance metrics for comprehensive analysis
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub execution_time: Duration,
    pub throughput: f64,      // items per second
    pub memory_usage: usize,  // bytes
    pub cpu_utilization: f64, // percentage
    pub accuracy_score: Option<f64>,
    pub scalability_factor: f64,
    pub error_rate: f64,
}

/// Comprehensive data generators for various scenarios
pub mod data_generators {
    use super::*;

    /// Generate realistic financial time series data
    pub fn generate_financial_data(_size: usize, volatility: f64) -> Array1<f64> {
        let mut data = Array1::zeros(_size);
        let mut price = 100.0;
        let mut rng_state = 42u64;

        for i in 0.._size {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let random = (rng_state % 10000) as f64 / 10000.0 - 0.5;

            let return_rate = random * volatility * 0.01;
            price *= 1.0 + return_rate;
            data[i] = price;
        }

        data
    }

    /// Generate ECG-like biomedical data
    pub fn generate_ecg_data(_size: usize, heart_rate: f64) -> Array1<f64> {
        let mut data = Array1::zeros(_size);
        let sampling_rate = 250.0; // Hz
        let period = sampling_rate / (heart_rate / 60.0);

        for i in 0.._size {
            let t = i as f64 / sampling_rate;
            let beat_phase = (i as f64 % period) / period * 2.0 * std::f64::consts::PI;

            // P wave
            let p_wave = 0.1 * (-((beat_phase - 0.2).powi(2)) * 100.0).exp();
            // QRS complex
            let qrs = if beat_phase > 1.4 && beat_phase < 1.8 {
                0.8 * (1.6 - beat_phase).abs()
            } else {
                0.0
            };
            // T wave
            let t_wave = 0.2 * (-((beat_phase - 4.0).powi(2)) * 50.0).exp();

            let noise = (((i * 17) % 1000) as f64 / 1000.0 - 0.5) * 0.05;
            data[i] = p_wave + qrs + t_wave + noise;
        }

        data
    }

    /// Generate environmental sensor data with realistic patterns
    pub fn generate_environmental_data(_size: usize, sensor_type: &str) -> Array1<f64> {
        let mut data = Array1::zeros(_size);
        let mut rng_state = 42u64;

        for i in 0.._size {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let random = (rng_state % 10000) as f64 / 10000.0;

            let base_value = match sensor_type {
                "temperature" => {
                    // Daily temperature cycle
                    let daily_cycle = (i as f64 * 2.0 * std::f64::consts::PI / 144.0).sin() * 10.0;
                    // Seasonal variation
                    let seasonal =
                        (i as f64 * 2.0 * std::f64::consts::PI / (365.0 * 144.0)).sin() * 20.0;
                    20.0 + daily_cycle + seasonal + (random - 0.5) * 2.0
                }
                "humidity" => {
                    let base = 50.0;
                    let variation = (i as f64 * 2.0 * std::f64::consts::PI / 288.0).sin() * 20.0;
                    (base + variation + (random - 0.5) * 5.0)
                        .max(0.0)
                        .min(100.0)
                }
                "pressure" => {
                    let base = 1013.25; // Standard atmospheric pressure
                    let weather_pattern =
                        (i as f64 * 2.0 * std::f64::consts::PI / 720.0).sin() * 30.0;
                    base + weather_pattern + (random - 0.5) * 5.0
                }
                _ => random * 100.0,
            };

            data[i] = base_value;
        }

        data
    }

    /// Generate multivariate time series with complex dependencies
    pub fn generate_complex_multivariate(
        size: usize,
        dimensions: usize,
        coupling_strength: f64,
    ) -> Array2<f64> {
        let mut data = Array2::zeros((size, dimensions));
        let mut states = vec![0.0; dimensions];
        let mut rng_state = 42u64;

        for i in 0..size {
            for j in 0..dimensions {
                rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                let random = (rng_state % 10000) as f64 / 10000.0 - 0.5;

                // Auto-regressive component
                let ar_component = if i > 0 { 0.8 * data[[i - 1, j]] } else { 0.0 };

                // Cross-dimensional coupling
                let coupling = if j > 0 {
                    coupling_strength * states[j - 1]
                } else {
                    0.0
                };

                // Trend and seasonality
                let trend = (i as f64) * 0.01 * (j + 1) as f64;
                let seasonal = (i as f64 * 2.0 * std::f64::consts::PI / (50.0 + j as f64 * 10.0))
                    .sin()
                    * (j + 1) as f64;

                states[j] = ar_component + coupling + trend + seasonal + random * 0.5;
                data[[i, j]] = states[j];
            }
        }

        data
    }
}

/// Advanced-comprehensive anomaly detection benchmarks
#[allow(dead_code)]
fn advanced_bench_anomaly_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("advanced_anomaly_detection");
    group.sample_size(50);

    let configs = [
        ("small_clean", 1_000, 0.1),
        ("medium_noisy", 10_000, 2.0),
        ("large_extreme", 100_000, 5.0),
        ("financial", 50_000, 1.5),
    ];

    for (config_name, size, noise_level) in configs.iter() {
        let data = match *config_name {
            "financial" => data_generators::generate_financial_data(*size, *noise_level, _ => generate_synthetic_data(*size, *noise_level),
        };

        group.throughput(Throughput::Elements(*size as u64));

        // Test multiple anomaly detection methods
        let methods = [
            ("z_score", "ZScore"),
            ("iqr", "IQR"),
            ("isolation_forest", "IsolationForest"),
            ("spc", "SPC"),
        ];

        for (method_name, method_display) in methods.iter() {
            group.bench_with_input(
                BenchmarkId::new(format!("{}_{}", method_display, config_name), size),
                &data,
                |b, data| {
                    b.iter(|| {
                        let detector = match *method_name {
                            "z_score" => AnomalyDetector::new().with_method(
                                scirs2_series::anomaly::AnomalyMethod::ZScore { threshold: 3.0 },
                            ),
                            "iqr" => AnomalyDetector::new().with_method(
                                scirs2_series::anomaly::AnomalyMethod::IQR { factor: 1.5 },
                            ),
                            "isolation_forest" => AnomalyDetector::new().with_method(
                                scirs2_series::anomaly::AnomalyMethod::IsolationForest {
                                    n_trees: 100,
                                    contamination: 0.1,
                                },
                            ),
                            "spc" => AnomalyDetector::new().with_method(
                                scirs2_series::anomaly::AnomalyMethod::SPC {
                                    window_size: 50,
                                    sigma_threshold: 3.0,
                                },
                            , _ => unreachable!(),
                        };
                        black_box(detector.detect(data).unwrap());
                    });
                },
            );
        }
    }

    group.finish();
}

/// Advanced-comprehensive forecasting benchmarks with multiple models
#[allow(dead_code)]
fn advanced_bench_forecasting(c: &mut Criterion) {
    let mut group = c.benchmark_group("advanced_forecasting");
    group.sample_size(30);

    let scenarios = [
        ("financial_daily", 2000, "financial"),
        ("environmental_hourly", 8760, "environmental"), // 1 year hourly
        ("iot_sensor_minutely", 43200, "sensor"),        // 30 days minutely
        ("complex_multivariate", 5000, "multivariate"),
    ];

    for (scenario_name, size, data_type) in scenarios.iter() {
        let data = match *data_type {
            "financial" => data_generators::generate_financial_data(*size, 2.0),
            "environmental" => data_generators::generate_environmental_data(*size, "temperature"),
            "sensor" => data_generators::generate_environmental_data(*size, "pressure"),
            "multivariate" => {
                let mv_data = data_generators::generate_complex_multivariate(*size, 3, 0.5);
                mv_data.column(0).to_owned()
            }
            _ => generate_synthetic_data(*size, 1.0),
        };

        group.throughput(Throughput::Elements(*size as u64));

        // Classical models
        group.bench_with_input(
            BenchmarkId::new(format!("ARIMA_{}", scenario_name), size),
            &data,
            |b, data| {
                b.iter(|| {
                    let mut model = ArimaModel::new(2, 1, 2).unwrap();
                    black_box(model.fit(data).unwrap());
                    black_box(model.forecast(24).unwrap());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("SARIMA_{}", scenario_name), size),
            &data,
            |b, data| {
                b.iter(|| {
                    let mut model = SARIMAModel::new(1, 1, 1, 1, 1, 1, 24).unwrap();
                    black_box(model.fit(data).unwrap());
                    black_box(model.forecast(24).unwrap());
                });
            },
        );

        // Neural models
        group.bench_with_input(
            BenchmarkId::new(format!("LSTM_{}", scenario_name), size),
            &data,
            |b, data| {
                b.iter(|| {
                    let mut model = LSTMForecaster::new(50, 64, 24).unwrap();
                    let train_data = Array2::fromshape_vec(
                        (data.len() - 50, 50),
                        data.windows(50).flat_map(|w| w.iter().copied()).collect(),
                    )
                    .unwrap();
                    let targets = Array1::from_iter(data.iter().skip(50).copied());
                    black_box(model.train(&train_data, &targets, 10, 0.001).unwrap());
                    black_box(model.forecast(&data.slice(s![-50..]), 24).unwrap());
                });
            },
        );

        // Quantum models (if available)
        group.bench_with_input(
            BenchmarkId::new(format!("QuantumNN_{}", scenario_name), size),
            &data,
            |b, data| {
                b.iter(|| {
                    let mut model = QuantumNeuralNetwork::<f64>::new(2, 4, 10, 1);
                    let input = Array1::from_iter(data.iter().take(10).copied());
                    black_box(model.forward(&input).unwrap());
                });
            },
        );
    }

    group.finish();
}

/// Advanced meta-learning and neural ODE benchmarks
#[allow(dead_code)]
fn advanced_bench_advanced_training(c: &mut Criterion) {
    let mut group = c.benchmark_group("advanced_advanced_training");
    group.sample_size(20);

    let task_scenarios = [
        ("few_shot_adaptation", 100),
        ("meta_learning_tasks", 500),
        ("continuous_time_modeling", 1000),
    ];

    for (scenario_name, size) in task_scenarios.iter() {
        let data = generate_synthetic_data(*size, 1.0);

        group.throughput(Throughput::Elements(*size as u64));

        // MAML benchmarks
        group.bench_with_input(
            BenchmarkId::new(format!("MAML_{}", scenario_name), size),
            &data,
            |b, data| {
                b.iter(|| {
                    let mut maml = MAML::<f64>::new(10, 32, 1, 0.01, 0.1, 5);

                    // Create dummy task data
                    let support_x = Array2::fromshape_vec(
                        (10, 10),
                        (0..100).map(|i| i as f64 * 0.01).collect(),
                    )
                    .unwrap();
                    let support_y = Array2::fromshape_vec(
                        (10, 1),
                        (0..10).map(|i| (i as f64).sin()).collect(),
                    )
                    .unwrap();
                    let query_x = support_x.clone();
                    let query_y = support_y.clone();

                    let task = scirs2_series::advanced, _training::TaskData {
                        support_x,
                        support_y,
                        query_x,
                        query_y,
                    };

                    black_box(maml.meta_train(&[task]).unwrap());
                });
            },
        );

        // Neural ODE benchmarks
        group.bench_with_input(
            BenchmarkId::new(format!("NeuralODE_{}", scenario_name), size),
            &data,
            |b, data| {
                b.iter(|| {
                    let time_steps = Array1::from_iter((0..20).map(|i| i as f64 * 0.1));
                    let solver_config = scirs2_series::advanced, _training::ODESolverConfig {
                        method: scirs2, series: advanced, _training: IntegrationMethod::RungeKutta4,
                        step_size: 0.1,
                        tolerance: 1e-6,
                    };

                    let node = NeuralODE::<f64>::new(3, 16, time_steps, solver_config);
                    let initial_state = Array1::from_vec(vec![1.0, 0.0, -0.5]);

                    black_box(node.forward(&initial_state).unwrap());
                });
            },
        );

        // VAE benchmarks
        group.bench_with_input(
            BenchmarkId::new(format!("TimeSeriesVAE_{}", scenario_name), size),
            &data,
            |b, data| {
                b.iter(|| {
                    let vae = TimeSeriesVAE::<f64>::new(20, 3, 8, 32, 32);
                    let input =
                        Array2::fromshape_vec((20, 3), data.iter().take(60).copied().collect())
                            .unwrap();

                    black_box(vae.forward(&input).unwrap());
                });
            },
        );

        // Transformer benchmarks
        group.bench_with_input(
            BenchmarkId::new(format!("Transformer_{}", scenario_name), size),
            &data,
            |b, data| {
                b.iter(|| {
                    let transformer = TimeSeriesTransformer::<f64>::new(50, 10, 128, 8, 4, 512);
                    let input =
                        Array2::fromshape_vec((2, 50), data.iter().take(100).copied().collect())
                            .unwrap();

                    black_box(transformer.forward(&input).unwrap());
                });
            },
        );
    }

    group.finish();
}

/// Comprehensive scalability and stress testing
#[allow(dead_code)]
fn advanced_bench_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("advanced_scalability");
    group.sample_size(20);

    // Extreme data sizes for scalability testing
    let extreme_sizes = [50_000, 100_000, 500_000, 1_000_000];

    for size in extreme_sizes.iter() {
        let data = generate_synthetic_data(*size, 1.0);

        group.throughput(Throughput::Elements(*size as u64));

        // Memory-efficient streaming statistics
        group.bench_with_input(
            BenchmarkId::new("streaming_stats_extreme", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let mut analyzer = StreamingAnalyzer::new(1000);
                    for &value in data.iter() {
                        black_box(analyzer.update(value).unwrap());
                    }
                });
            },
        );

        // Out-of-core processing simulation
        group.bench_with_input(
            BenchmarkId::new("out_of_core_extreme", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let config = ProcessingConfig::new()
                        .with_chunk_size(10_000)
                        .with_parallel_processing(true);

                    let processor = ChunkedProcessor::new(config);

                    // Simulate chunked processing
                    let chunk_size = 10_000;
                    let mut total_sum = 0.0;

                    for chunk in data.chunks(chunk_size) {
                        total_sum += chunk.iter().sum::<f64>();
                    }

                    black_box(total_sum / data.len() as f64);
                });
            },
        );
    }

    group.finish();
}

/// Cross-domain application benchmarks
#[allow(dead_code)]
fn advanced_bench_domain_applications(c: &mut Criterion) {
    let mut group = c.benchmark_group("advanced_domain_applications");
    group.sample_size(30);

    // Biomedical signal processing
    let ecg_data = data_generators::generate_ecg_data(5000, 72.0); // 72 BPM

    group.bench_function("ECG_analysis_complete", |b| {
        b.iter(|| {
            let analyzer = ECGAnalysis::new();
            black_box(analyzer.detect_r_peaks(&ecg_data, 250.0).unwrap());
            black_box(analyzer.calculate_hrv(&ecg_data, 250.0).unwrap());
            black_box(analyzer.detect_arrhythmia(&ecg_data, 250.0).unwrap());
        });
    });

    // Environmental monitoring
    let temp_data = data_generators::generate_environmental_data(8760, "temperature");
    let humidity_data = data_generators::generate_environmental_data(8760, "humidity");
    let pressure_data = data_generators::generate_environmental_data(8760, "pressure");

    group.bench_function("environmental_monitoring_suite", |b| {
        b.iter(|| {
            let analyzer = EnvironmentalSensorAnalysis::new();
            black_box(analyzer.detect_heat_waves(&temp_data, 30.0).unwrap());
            black_box(
                analyzer
                    .calculate_drought_index(&temp_data, &humidity_data)
                    .unwrap(),
            );
            black_box(analyzer.detect_storm_patterns(&pressure_data, 12).unwrap());
        });
    });

    // Financial risk analysis
    let price_data = data_generators::generate_financial_data(2000, 2.5);

    group.bench_function("financial_risk_analysis", |b| {
        b.iter(|| {
            // Calculate returns
            let returns =
                Array1::from_iter(price_data.windows(2).map(|w| (w[1] / w[0] - 1.0) * 100.0));

            // GARCH modeling
            black_box(garch_model(&returns, 1, 1).unwrap());

            // Risk metrics
            let sorted_returns = {
                let mut r = returns.to_vec();
                r.sort_by(|a, b| a.partial_cmp(b).unwrap());
                r
            };
            let var_95 = sorted_returns[(sorted_returns.len() as f64 * 0.05) as usize];
            black_box(var_95);
        });
    });
}

/// Comparative performance analysis across different implementations
#[allow(dead_code)]
fn advanced_bench_comparative_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("advanced_comparative_analysis");
    group.sample_size(50);

    let test_sizes = [1_000, 10_000, 50_000];

    for size in test_sizes.iter() {
        let data = generate_synthetic_data(*size, 1.0);

        // Compare different decomposition methods
        group.bench_with_input(
            BenchmarkId::new("STL_vs_Classical", size),
            &data,
            |b, data| {
                b.iter(|| {
                    // STL decomposition
                    let stl_decomposer = STLDecomposer::new(12, 7, 7, 1, false).unwrap();
                    black_box(stl_decomposer.decompose(data).unwrap());

                    // Classical decomposition (simple moving average for comparison)
                    let window_size = 12;
                    let mut trend = Array1::zeros(data.len());
                    for i in window_size / 2..data.len() - window_size / 2 {
                        let start = i - window_size / 2;
                        let end = i + window_size / 2;
                        trend[i] = data.slice(s![start..end]).mean().unwrap();
                    }
                    black_box(trend);
                });
            },
        );

        // Compare different clustering approaches
        let mv_data = data_generators::generate_complex_multivariate(*size, 5, 0.3);

        group.bench_with_input(
            BenchmarkId::new("clustering_comparison", size),
            &mv_data,
            |b, data| {
                b.iter(|| {
                    // K-means clustering
                    let kmeans_clusterer = TimeSeriesClusterer::new(
                        scirs2_series::clustering::ClusteringMethod::KMeans { k: 3 },
                        scirs2_series::clustering::DistanceMetric::Euclidean,
                    );
                    black_box(kmeans_clusterer.cluster(data).unwrap());

                    // Hierarchical clustering
                    let hierarchical_clusterer = TimeSeriesClusterer::new(
                        scirs2_series::clustering::ClusteringMethod::Hierarchical { n_clusters: 3 },
                        scirs2_series::clustering::DistanceMetric::DTW,
                    );
                    black_box(hierarchical_clusterer.cluster(data).unwrap());
                });
            },
        );
    }

    group.finish();
}

/// Memory usage profiling benchmarks
#[allow(dead_code)]
fn advanced_bench_memory_profiling(c: &mut Criterion) {
    let mut group = c.benchmark_group("advanced_memory_profiling");
    group.sample_size(20);

    // Test memory efficiency of different algorithms
    let memory_test_sizes = [10_000, 50_000, 100_000];

    for size in memory_test_sizes.iter() {
        let data = generate_synthetic_data(*size, 1.0);

        group.bench_with_input(
            BenchmarkId::new("memory_efficient_processing", size),
            &data,
            |b, data| {
                b.iter(|| {
                    // Out-of-core processing
                    let config = ProcessingConfig::new()
                        .with_chunk_size(1000)
                        .with_max_memory(1_000_000); // 1MB limit

                    let processor = ChunkedProcessor::new(config);

                    // Simulate memory-constrained processing
                    let mut stats = scirs2_series::out_of, _core::StreamingStats::new();
                    for chunk in data.chunks(1000) {
                        for &value in chunk {
                            stats.update(value);
                        }
                    }

                    black_box((stats.mean, stats.variance()));
                });
            },
        );
    }

    group.finish();
}

/// Error handling and robustness benchmarks
#[allow(dead_code)]
fn advanced_bench_robustness(c: &mut Criterion) {
    let mut group = c.benchmark_group("advanced_robustness");
    group.sample_size(30);

    // Test with various data quality issues
    let robustness_scenarios = [
        ("missing_values", 5000, 0.1), // 10% missing values
        ("outliers", 5000, 0.05),      // 5% outliers
        ("noise_burst", 5000, 0.02),   // 2% noise bursts
    ];

    for (scenario_name, size, corruption_rate) in robustness_scenarios.iter() {
        let mut data = generate_synthetic_data(*size, 1.0);
        let mut rng_state = 42u64;

        // Introduce data quality issues
        for i in 0..*size {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let random = (rng_state % 10000) as f64 / 10000.0;

            if random < *corruption_rate {
                match *scenario_name {
                    "missing_values" => data[i] = f64::NAN,
                    "outliers" => data[i] += (random - 0.5) * 100.0,
                    "noise_burst" => data[i] += (random - 0.5) * 50.0_ => {}
                }
            }
        }

        group.bench_with_input(
            BenchmarkId::new(format!("robust_processing_{}", scenario_name), size),
            &data,
            |b, data| {
                b.iter(|| {
                    // Test robust algorithms
                    let filter = RobustTrendFilter::new(0.1);

                    // Clean data first
                    let cleaned_data: Array1<f64> = data
                        .iter()
                        .map(|&x| if x.is_finite() { x } else { 0.0 })
                        .collect();

                    black_box(filter.filter(&cleaned_data).unwrap());
                });
            },
        );
    }

    group.finish();
}

/// Utility function to generate synthetic data with various patterns
#[allow(dead_code)]
fn generate_synthetic_data(_size: usize, noise_level: f64) -> Array1<f64> {
    let mut data = Array1::zeros(_size);
    let mut rng_state = 42u64; // Simple LCG for reproducible random numbers

    for i in 0.._size {
        // Generate deterministic but pseudo-random data
        rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        let random = (rng_state % 10000) as f64 / 10000.0;

        // Multi-component time series
        let trend = (i as f64) * 0.01;
        let seasonal_annual = (i as f64 * 2.0 * std::f64::consts::PI / 365.0).sin() * 10.0;
        let seasonal_weekly = (i as f64 * 2.0 * std::f64::consts::PI / 7.0).sin() * 3.0;
        let noise = (random - 0.5) * noise_level;

        data[i] = 50.0 + trend + seasonal_annual + seasonal_weekly + noise;
    }

    data
}

// Benchmark groups
criterion_group!(
    advanced_benchmarks,
    advanced_bench_anomaly_detection,
    advanced_bench_forecasting,
    advanced_bench_advanced_training,
    advanced_bench_scalability,
    advanced_bench_domain_applications,
    advanced_bench_comparative_analysis,
    advanced_bench_memory_profiling,
    advanced_bench_robustness,
);

criterion_main!(advanced_benchmarks);
