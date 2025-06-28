//! Performance benchmarks with SciPy-style reference implementations
//!
//! This benchmark suite provides performance comparisons against baseline implementations
//! that would be comparable to SciPy's ndimage performance characteristics.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array1, Array2, Array3, Array4};
use scirs2_ndimage::filters::*;
use scirs2_ndimage::interpolation::*;
use scirs2_ndimage::measurements::*;
use scirs2_ndimage::morphology::*;
use std::time::Duration;

/// Benchmark filters against baseline implementations
fn bench_filter_performance_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("filter_performance_comparison");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(10);

    // Test multiple array sizes to understand scaling behavior
    let sizes = vec![
        (32, 32),     // Small
        (128, 128),   // Medium
        (512, 512),   // Large
        (1024, 1024), // Very large
    ];

    for (rows, cols) in sizes {
        let input = Array2::from_shape_fn((rows, cols), |(i, j)| {
            ((i as f64 * 0.1).sin() * (j as f64 * 0.1).cos() * 100.0)
        });

        // Benchmark Gaussian filter at different sigma values
        for sigma in [0.5, 1.0, 2.0, 4.0] {
            group.bench_with_input(
                BenchmarkId::new(
                    "gaussian_filter",
                    format!("{}x{}_sigma{}", rows, cols, sigma),
                ),
                &input,
                |b, input| {
                    b.iter(|| {
                        gaussian_filter(black_box(input), black_box(sigma), None, None).unwrap()
                    })
                },
            );
        }

        // Benchmark median filter with different kernel sizes
        for kernel_size in [3, 5, 7, 9] {
            group.bench_with_input(
                BenchmarkId::new(
                    "median_filter",
                    format!("{}x{}_k{}", rows, cols, kernel_size),
                ),
                &input,
                |b, input| {
                    b.iter(|| {
                        median_filter(
                            black_box(input),
                            black_box(&[kernel_size, kernel_size]),
                            None,
                        )
                        .unwrap()
                    })
                },
            );
        }

        // Benchmark uniform filter
        group.bench_with_input(
            BenchmarkId::new("uniform_filter", format!("{}x{}", rows, cols)),
            &input,
            |b, input| {
                b.iter(|| uniform_filter(black_box(input), black_box(&[5, 5]), None, None).unwrap())
            },
        );
    }

    group.finish();
}

/// Benchmark morphological operations scaling
fn bench_morphology_performance_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("morphology_performance_scaling");
    group.measurement_time(Duration::from_secs(10));

    let sizes = vec![(64, 64), (128, 128), (256, 256), (512, 512)];

    for (rows, cols) in sizes {
        // Create binary test image with complex structure
        let binary_input = Array2::from_shape_fn((rows, cols), |(i, j)| {
            let x = i as f64 / rows as f64;
            let y = j as f64 / cols as f64;
            ((x * 10.0).sin() + (y * 10.0).cos()) > 0.0
        });

        // Benchmark binary morphological operations
        group.bench_with_input(
            BenchmarkId::new("binary_erosion", format!("{}x{}", rows, cols)),
            &binary_input,
            |b, input| {
                b.iter(|| {
                    binary_erosion(black_box(input), None, None, None, None, None, None).unwrap()
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("binary_dilation", format!("{}x{}", rows, cols)),
            &binary_input,
            |b, input| {
                b.iter(|| {
                    binary_dilation(black_box(input), None, None, None, None, None, None).unwrap()
                })
            },
        );

        // Test opening (erosion + dilation)
        group.bench_with_input(
            BenchmarkId::new("binary_opening", format!("{}x{}", rows, cols)),
            &binary_input,
            |b, input| {
                b.iter(|| {
                    binary_opening(black_box(input), None, None, None, None, None, None).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark measurement operations on labeled arrays
fn bench_measurements_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("measurements_performance");
    group.measurement_time(Duration::from_secs(8));

    let sizes = vec![(100, 100), (200, 200), (400, 400)];

    for (rows, cols) in sizes {
        // Create test data with multiple regions
        let values = Array2::from_shape_fn((rows, cols), |(i, j)| {
            (i as f64 * j as f64).sqrt() + (i + j) as f64 * 0.1
        });

        let labels = Array2::from_shape_fn((rows, cols), |(i, j)| {
            ((i / 20) * 10 + (j / 20)) + 1 // Create grid of labeled regions
        });

        group.bench_with_input(
            BenchmarkId::new("sum_labels", format!("{}x{}", rows, cols)),
            &(&values, &labels),
            |b, (values, labels)| {
                b.iter(|| sum_labels(black_box(values), black_box(labels), None).unwrap())
            },
        );

        group.bench_with_input(
            BenchmarkId::new("mean_labels", format!("{}x{}", rows, cols)),
            &(&values, &labels),
            |b, (values, labels)| {
                b.iter(|| mean_labels(black_box(values), black_box(labels), None).unwrap())
            },
        );

        group.bench_with_input(
            BenchmarkId::new("variance_labels", format!("{}x{}", rows, cols)),
            &(&values, &labels),
            |b, (values, labels)| {
                b.iter(|| variance_labels(black_box(values), black_box(labels), None).unwrap())
            },
        );

        group.bench_with_input(
            BenchmarkId::new("center_of_mass", format!("{}x{}", rows, cols)),
            &values,
            |b, values| b.iter(|| center_of_mass(black_box(values)).unwrap()),
        );
    }

    group.finish();
}

/// Benchmark interpolation operations at different scales
fn bench_interpolation_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("interpolation_performance");
    group.measurement_time(Duration::from_secs(10));

    let base_sizes = vec![(64, 64), (128, 128), (256, 256)];

    for (rows, cols) in base_sizes {
        let input = Array2::from_shape_fn((rows, cols), |(i, j)| {
            ((i as f64 / 10.0).sin() * (j as f64 / 10.0).cos())
        });

        // Benchmark zoom at different factors
        for zoom_factor in [0.5, 1.5, 2.0, 3.0] {
            group.bench_with_input(
                BenchmarkId::new("zoom", format!("{}x{}_factor{}", rows, cols, zoom_factor)),
                &input,
                |b, input| {
                    b.iter(|| {
                        zoom(
                            black_box(input),
                            black_box(&[zoom_factor, zoom_factor]),
                            None,
                            None,
                            None,
                            None,
                        )
                        .unwrap()
                    })
                },
            );
        }

        // Benchmark rotation
        group.bench_with_input(
            BenchmarkId::new("rotate", format!("{}x{}", rows, cols)),
            &input,
            |b, input| {
                b.iter(|| {
                    rotate(
                        black_box(input),
                        black_box(std::f64::consts::PI / 4.0), // 45 degrees
                        None,
                        None,
                        None,
                        None,
                        None,
                    )
                    .unwrap()
                })
            },
        );

        // Benchmark affine transformation
        let transform_matrix = ndarray::array![[0.8, -0.2], [0.2, 0.9]]; // Scale + rotate
        group.bench_with_input(
            BenchmarkId::new("affine_transform", format!("{}x{}", rows, cols)),
            &input,
            |b, input| {
                b.iter(|| {
                    affine_transform(
                        black_box(input),
                        black_box(&transform_matrix),
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                    )
                    .unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark 3D operations to test multi-dimensional scaling
fn bench_3d_operations_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("3d_operations_performance");
    group.measurement_time(Duration::from_secs(12));
    group.sample_size(5); // Fewer samples for 3D due to longer execution time

    let sizes = vec![(32, 32, 32), (64, 64, 64), (100, 100, 100)];

    for (depth, rows, cols) in sizes {
        let volume = Array3::from_shape_fn((depth, rows, cols), |(d, i, j)| {
            ((d + i + j) as f64 / 10.0).sin() * 100.0
        });

        // 3D Gaussian filter
        group.bench_with_input(
            BenchmarkId::new("gaussian_3d", format!("{}x{}x{}", depth, rows, cols)),
            &volume,
            |b, volume| {
                b.iter(|| gaussian_filter(black_box(volume), black_box(1.0), None, None).unwrap())
            },
        );

        // 3D uniform filter
        group.bench_with_input(
            BenchmarkId::new("uniform_3d", format!("{}x{}x{}", depth, rows, cols)),
            &volume,
            |b, volume| {
                b.iter(|| {
                    uniform_filter(black_box(volume), black_box(&[3, 3, 3]), None, None).unwrap()
                })
            },
        );

        // 3D center of mass calculation
        group.bench_with_input(
            BenchmarkId::new("center_of_mass_3d", format!("{}x{}x{}", depth, rows, cols)),
            &volume,
            |b, volume| b.iter(|| center_of_mass(black_box(volume)).unwrap()),
        );
    }

    group.finish();
}

/// Benchmark memory efficiency with different border modes
fn bench_border_mode_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("border_mode_performance");
    group.measurement_time(Duration::from_secs(8));

    let input = Array2::from_shape_fn((200, 200), |(i, j)| (i + j) as f64);
    let kernel_size = [9, 9]; // Larger kernel to emphasize border effects

    let border_modes = [
        BorderMode::Constant,
        BorderMode::Reflect,
        BorderMode::Mirror,
        BorderMode::Wrap,
        BorderMode::Nearest,
    ];

    for mode in border_modes {
        group.bench_with_input(
            BenchmarkId::new("gaussian_border", format!("{:?}", mode)),
            &input,
            |b, input| {
                b.iter(|| {
                    gaussian_filter(black_box(input), 2.0, Some(black_box(mode)), None).unwrap()
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("uniform_border", format!("{:?}", mode)),
            &input,
            |b, input| {
                b.iter(|| {
                    uniform_filter(
                        black_box(input),
                        black_box(&kernel_size),
                        Some(black_box(mode)),
                        None,
                    )
                    .unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark high-dimensional array operations (4D+)
fn bench_high_dimensional_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("high_dimensional_performance");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(5);

    // 4D array (e.g., video data: time x height x width x channels)
    let shape_4d = (10, 32, 32, 3);
    let array_4d = Array4::from_shape_fn(shape_4d, |(t, i, j, c)| (t + i + j + c) as f64);

    group.bench_with_input(
        BenchmarkId::new(
            "gaussian_4d",
            format!(
                "{}x{}x{}x{}",
                shape_4d.0, shape_4d.1, shape_4d.2, shape_4d.3
            ),
        ),
        &array_4d,
        |b, array| {
            b.iter(|| gaussian_filter(black_box(array), black_box(1.0), None, None).unwrap())
        },
    );

    group.bench_with_input(
        BenchmarkId::new(
            "center_of_mass_4d",
            format!(
                "{}x{}x{}x{}",
                shape_4d.0, shape_4d.1, shape_4d.2, shape_4d.3
            ),
        ),
        &array_4d,
        |b, array| b.iter(|| center_of_mass(black_box(array)).unwrap()),
    );

    group.finish();
}

criterion_group!(
    scipy_performance_benches,
    bench_filter_performance_comparison,
    bench_morphology_performance_scaling,
    bench_measurements_performance,
    bench_interpolation_performance,
    bench_3d_operations_performance,
    bench_border_mode_performance,
    bench_high_dimensional_performance
);

criterion_main!(scipy_performance_benches);
