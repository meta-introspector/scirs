//! Comprehensive performance benchmarks for scirs2-series modules
//!
//! This benchmark suite measures the performance of various time series analysis
//! functions across different data sizes and configurations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::{Array1, Array2};
use scirs2_series::{
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

/// Generate synthetic time series data for benchmarking
#[allow(dead_code)]
fn generate_synthetic_data(size: usize, noise_level: f64) -> Array1<f64> {
    let mut data = Array1::zeros(size);
    let mut rng_state = 42u64; // Simple LCG for reproducible random numbers

    for i in 0..size {
        // Generate deterministic but pseudo-random data
        rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        let random = (rng_state % 10000) as f64 / 10000.0;

        // Trend + seasonality + noise
        let trend = (i as f64) * 0.01;
        let seasonal = (i as f64 * 2.0 * std::f64::consts::PI / 100.0).sin() * 5.0;
        let noise = (random - 0.5) * noise_level;

        data[i] = 50.0 + trend + seasonal + noise;
    }

    data
}

/// Generate multivariate time series data
#[allow(dead_code)]
fn generate_multivariate_data(size: usize, dimensions: usize) -> Array2<f64> {
    let mut data = Array2::zeros((size, dimensions));
    let mut rng_state = 42u64;

    for i in 0..size {
        for j in 0..dimensions {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let random = (rng_state % 10000) as f64 / 10000.0;

            let trend = (i as f64) * 0.01 * (j + 1) as f64;
            let seasonal = (i as f64 * 2.0 * std::f64::consts::PI / (50.0 + j as f64 * 20.0)).sin()
                * (j + 1) as f64;
            let noise = (random - 0.5) * 2.0;

            data[[i, j]] = 50.0 + trend + seasonal + noise;
        }
    }

    data
}

/// Benchmark anomaly detection algorithms
#[allow(dead_code)]
fn bench_anomaly_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("anomaly_detection");

    for size in [1_000, 10_000, 100_000].iter() {
        let data = generate_synthetic_data(*size, 2.0);

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("z_score", size), size, |b, _| {
            b.iter(|| {
                let detector = AnomalyDetector::new()
                    .with_method(scirs2_series::anomaly::AnomalyMethod::ZScore { threshold: 3.0 });
                black_box(detector.detect(&data).unwrap());
            });
        });

        group.bench_with_input(BenchmarkId::new("iqr", size), size, |b, _| {
            b.iter(|| {
                let detector = AnomalyDetector::new()
                    .with_method(scirs2_series::anomaly::AnomalyMethod::IQR { factor: 1.5 });
                black_box(detector.detect(&data).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark time series decomposition methods
#[allow(dead_code)]
fn bench_decomposition(c: &mut Criterion) {
    let mut group = c.benchmark_group("decomposition");

    for size in [1_000, 5_000, 20_000].iter() {
        let data = generate_synthetic_data(*size, 1.0);

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("stl", size), size, |b, _| {
            b.iter(|| {
                let decomposer = STLDecomposer::new(12, 7, 7, 1, false).unwrap();
                black_box(decomposer.decompose(&data).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark forecasting algorithms
#[allow(dead_code)]
fn bench_forecasting(c: &mut Criterion) {
    let mut group = c.benchmark_group("forecasting");

    for size in [500, 2_000, 10_000].iter() {
        let data = generate_synthetic_data(*size, 1.0);

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("arima", size), size, |b, _| {
            b.iter(|| {
                let mut model = ArimaModel::new(1, 1, 1).unwrap();
                black_box(model.fit(&data).unwrap());
                black_box(model.forecast(10).unwrap());
            });
        });

        group.bench_with_input(BenchmarkId::new("sarima", size), size, |b, _| {
            b.iter(|| {
                let mut model = SARIMAModel::new(1, 1, 1, 1, 1, 1, 12).unwrap();
                black_box(model.fit(&data).unwrap());
                black_box(model.forecast(10).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark change point detection
#[allow(dead_code)]
fn bench_change_point_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("change_point_detection");

    for size in [1_000, 5_000, 20_000].iter() {
        let data = generate_synthetic_data(*size, 1.0);

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("pelt", size), size, |b, _| {
            b.iter(|| {
                let detector = PELTDetector::new(5.0);
                black_box(detector.detect(&data).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark correlation analysis
#[allow(dead_code)]
fn bench_correlation(c: &mut Criterion) {
    let mut group = c.benchmark_group("correlation");

    for size in [1_000, 5_000, 20_000].iter() {
        let data1 = generate_synthetic_data(*size, 1.0);
        let data2 = generate_synthetic_data(*size, 1.0);

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("cross_correlation", size), size, |b, _| {
            b.iter(|| {
                let correlator = CrossCorrelation::new();
                black_box(correlator.cross_correlate(&data1, &data2, 50).unwrap());
            });
        });

        group.bench_with_input(BenchmarkId::new("autocorrelation", size), size, |b, _| {
            b.iter(|| {
                let correlator = CrossCorrelation::new();
                black_box(correlator.autocorrelation(&data1, 50).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark feature extraction
#[allow(dead_code)]
fn bench_feature_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature_extraction");

    for size in [1_000, 5_000, 20_000].iter() {
        let data = generate_synthetic_data(*size, 1.0);

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(
            BenchmarkId::new("statistical_features", size),
            size,
            |b, _| {
                b.iter(|| {
                    let extractor = StatisticalFeatures::new();
                    black_box(extractor.extract(&data).unwrap());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark clustering algorithms
#[allow(dead_code)]
fn bench_clustering(c: &mut Criterion) {
    let mut group = c.benchmark_group("clustering");

    for (n_series, length) in [(50, 500), (100, 1000), (200, 500)].iter() {
        let data = generate_multivariate_data(*length, *n_series);

        group.throughput(Throughput::Elements((n_series * length) as u64));

        group.bench_with_input(
            BenchmarkId::new("kmeans", format!("{}x{}", n_series, length)),
            &(*n_series, *length),
            |b, _| {
                b.iter(|| {
                    let clusterer = TimeSeriesClusterer::new(
                        scirs2_series::clustering::ClusteringMethod::KMeans { k: 3 },
                        scirs2_series::clustering::DistanceMetric::Euclidean,
                    );
                    black_box(clusterer.cluster(&data).unwrap());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark state space models
#[allow(dead_code)]
fn bench_state_space(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_space");

    for size in [500, 2_000, 10_000].iter() {
        let observations = generate_synthetic_data(*size, 2.0);

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("kalman_filter", size), size, |b, _| {
            b.iter(|| {
                let initial_state = scirs2_series::state_space::StateVector {
                    state: ndarray::arr1(&[0.0, 0.0]),
                    covariance: ndarray::arr2(&[[1.0, 0.0], [0.0, 1.0]]),
                };
                let transition = scirs2_series::state_space::StateTransition {
                    transition_matrix: ndarray::arr2(&[[1.0, 1.0], [0.0, 1.0]]),
                    process_noise: ndarray::arr2(&[[0.1, 0.0], [0.0, 0.1]]),
                };
                let observation = scirs2_series::state_space::ObservationModel {
                    observation_matrix: ndarray::arr2(&[[1.0, 0.0]]),
                    observation_noise: ndarray::arr2(&[[1.0]]),
                };
                let mut filter = KalmanFilter::new(initial_state, transition, observation);

                for &obs in observations.iter() {
                    black_box(filter.update(&ndarray::arr1(&[obs])).unwrap());
                }
            });
        });
    }

    group.finish();
}

/// Benchmark financial models
#[allow(dead_code)]
fn bench_financial(c: &mut Criterion) {
    let mut group = c.benchmark_group("financial");

    for size in [500, 2_000, 5_000].iter() {
        let returns = generate_synthetic_data(*size, 0.1);

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("garch", size), size, |b, _| {
            b.iter(|| {
                black_box(garch_model(&returns, 1, 1).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark streaming analysis
#[allow(dead_code)]
fn bench_streaming(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming");

    for window_size in [100, 500, 1000].iter() {
        let data = generate_synthetic_data(10000, 1.0);

        group.throughput(Throughput::Elements(data.len() as u64));

        group.bench_with_input(
            BenchmarkId::new("streaming_stats", window_size),
            window_size,
            |b, &window_size| {
                b.iter(|| {
                    let config = scirs2_series::streaming::StreamConfig {
                        window_size,
                        ..Default::default()
                    };
                    let mut analyzer = StreamingAnalyzer::new(config).unwrap();
                    for &value in data.iter() {
                        black_box(analyzer.add_observation(value).unwrap());
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark distributed processing
#[allow(dead_code)]
fn bench_distributed(c: &mut Criterion) {
    let mut group = c.benchmark_group("distributed");

    for size in [5_000, 20_000, 50_000].iter() {
        let data = generate_synthetic_data(*size, 1.0);

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(
            BenchmarkId::new("distributed_forecasting", size),
            size,
            |b, _| {
                b.iter(|| {
                    let config = ClusterConfig::default();
                    let mut processor = DistributedProcessor::new(config);
                    black_box(processor.distributed_forecast(&data, 10, "arima").unwrap());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark out-of-core processing
#[allow(dead_code)]
fn bench_out_of_core(c: &mut Criterion) {
    let mut group = c.benchmark_group("out_of_core");

    for chunk_size in [1_000, 5_000, 10_000].iter() {
        group.bench_with_input(
            BenchmarkId::new("chunked_processing", chunk_size),
            chunk_size,
            |b, &chunk_size| {
                b.iter(|| {
                    let config = ProcessingConfig::new()
                        .with_chunk_size(chunk_size)
                        .with_parallel_processing(false);

                    let processor = ChunkedProcessor::new(config);

                    // Simulate processing chunks
                    let data = generate_synthetic_data(chunk_size * 5, 1.0);
                    let chunks: Vec<_> = data.windows(chunk_size).collect();

                    for chunk in chunks {
                        black_box(chunk.iter().sum::<f64>() / chunk.len() as f64);
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark trend analysis
#[allow(dead_code)]
fn bench_trend_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("trend_analysis");

    for size in [1_000, 5_000, 20_000].iter() {
        let data = generate_synthetic_data(*size, 2.0);

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("robust_trend", size), size, |b, _| {
            b.iter(|| {
                let filter = RobustTrendFilter::new(0.1);
                black_box(filter.filter(&data).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark validation and cross-validation
#[allow(dead_code)]
fn bench_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("validation");

    for size in [500, 2_000, 5_000].iter() {
        let data = generate_synthetic_data(*size, 1.0);

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("cross_validation", size), size, |b, _| {
            b.iter(|| {
                let validator = CrossValidator::new(5, 0.2); // 5-fold CV with 20% test
                black_box(validator.time_series_split(&data).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark data transformations
#[allow(dead_code)]
fn bench_transformations(c: &mut Criterion) {
    let mut group = c.benchmark_group("transformations");

    for size in [1_000, 10_000, 100_000].iter() {
        let data = generate_synthetic_data(*size, 1.0).mapv(|x| x.abs() + 1.0); // Ensure positive values

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("box_cox", size), size, |b, _| {
            b.iter(|| {
                let result = scirs2_series::transformations::box_cox_transform(&data, None);
                black_box(result.unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark causality analysis
#[allow(dead_code)]
fn bench_causality(c: &mut Criterion) {
    let mut group = c.benchmark_group("causality");

    for size in [500, 1_000, 2_000].iter() {
        let data1 = generate_synthetic_data(*size, 1.0);
        let data2 = generate_synthetic_data(*size, 1.0);

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("granger_causality", size), size, |b, _| {
            b.iter(|| {
                let test = GrangerCausalityTest::new(5); // lag = 5
                black_box(test.test(&data1, &data2).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark dimensionality reduction
#[allow(dead_code)]
fn bench_dimensionality_reduction(c: &mut Criterion) {
    let mut group = c.benchmark_group("dimensionality_reduction");

    for (n_series, length) in [(20, 1000), (50, 500), (100, 200)].iter() {
        let data = generate_multivariate_data(*length, *n_series);

        group.throughput(Throughput::Elements((n_series * length) as u64));

        group.bench_with_input(
            BenchmarkId::new("functional_pca", format!("{}x{}", n_series, length)),
            &(*n_series, *length),
            |b, _| {
                b.iter(|| {
                    let fpca = FunctionalPCA::new(5); // 5 components
                    black_box(fpca.fit_transform(&data).unwrap());
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_anomaly_detection,
    bench_decomposition,
    bench_forecasting,
    bench_change_point_detection,
    bench_correlation,
    bench_feature_extraction,
    bench_clustering,
    bench_state_space,
    bench_financial,
    bench_streaming,
    bench_distributed,
    bench_out_of_core,
    bench_trend_analysis,
    bench_validation,
    bench_transformations,
    bench_causality,
    bench_dimensionality_reduction,
);

criterion_main!(benches);
