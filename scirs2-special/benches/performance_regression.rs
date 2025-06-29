//! Performance regression benchmarks for CI/CD integration
//!
//! This benchmark suite is designed to detect performance regressions in
//! special function implementations. It covers critical paths and common
//! usage patterns that must maintain stable performance.

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use ndarray::{Array1, Array2};
use num_complex::Complex64;
use scirs2_special::*;
use std::time::Duration;

// Benchmark configuration
const SMALL_ARRAY_SIZE: usize = 100;
const MEDIUM_ARRAY_SIZE: usize = 1_000;
const LARGE_ARRAY_SIZE: usize = 10_000;
const XLARGE_ARRAY_SIZE: usize = 100_000;

/// Core single-value benchmarks for critical functions
fn core_single_value_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("core_single_value");
    group.measurement_time(Duration::from_secs(5));
    
    // Gamma function - most critical
    group.bench_function("gamma_small", |b| {
        b.iter(|| gamma(black_box(1.5)))
    });
    
    group.bench_function("gamma_medium", |b| {
        b.iter(|| gamma(black_box(10.5)))
    });
    
    group.bench_function("gamma_large", |b| {
        b.iter(|| gamma(black_box(50.5)))
    });
    
    // Error functions - high usage
    group.bench_function("erf_small", |b| {
        b.iter(|| erf(black_box(0.5)))
    });
    
    group.bench_function("erf_medium", |b| {
        b.iter(|| erf(black_box(2.0)))
    });
    
    group.bench_function("erf_large", |b| {
        b.iter(|| erf(black_box(5.0)))
    });
    
    // Bessel functions - computationally intensive
    group.bench_function("bessel_j0_small", |b| {
        b.iter(|| bessel::j0(black_box(1.0)))
    });
    
    group.bench_function("bessel_j0_medium", |b| {
        b.iter(|| bessel::j0(black_box(10.0)))
    });
    
    group.bench_function("bessel_j0_large", |b| {
        b.iter(|| bessel::j0(black_box(50.0)))
    });
    
    group.bench_function("bessel_j1_medium", |b| {
        b.iter(|| bessel::j1(black_box(10.0)))
    });
    
    // Complex functions
    group.bench_function("gamma_complex", |b| {
        b.iter(|| gamma_complex(black_box(Complex64::new(2.0, 1.0))))
    });
    
    group.finish();
}

/// Array operation benchmarks - critical for SIMD/vectorization
fn array_operation_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_operations");
    group.measurement_time(Duration::from_secs(10));
    
    // Test different array sizes
    for &size in &[SMALL_ARRAY_SIZE, MEDIUM_ARRAY_SIZE, LARGE_ARRAY_SIZE] {
        let data: Vec<f64> = (0..size).map(|i| 1.0 + i as f64 * 0.01).collect();
        let array = Array1::from(data.clone());
        
        group.throughput(Throughput::Elements(size as u64));
        
        // Gamma function array operations
        group.bench_with_input(
            BenchmarkId::new("gamma_array_scalar", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let mut results = Vec::with_capacity(data.len());
                    for &x in data {
                        results.push(gamma(black_box(x)));
                    }
                    black_box(results)
                })
            },
        );
        
        // Error function array operations
        group.bench_with_input(
            BenchmarkId::new("erf_array_scalar", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let mut results = Vec::with_capacity(data.len());
                    for &x in data {
                        results.push(erf(black_box(x)));
                    }
                    black_box(results)
                })
            },
        );
        
        // Bessel function array operations
        group.bench_with_input(
            BenchmarkId::new("bessel_j0_array_scalar", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let mut results = Vec::with_capacity(data.len());
                    for &x in data {
                        results.push(bessel::j0(black_box(x)));
                    }
                    black_box(results)
                })
            },
        );
    }
    
    group.finish();
}

/// Memory allocation and large array benchmarks
fn memory_intensive_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_intensive");
    group.measurement_time(Duration::from_secs(15));
    
    // Large array allocations
    group.bench_function("large_array_gamma", |b| {
        b.iter(|| {
            let data: Vec<f64> = (0..XLARGE_ARRAY_SIZE)
                .map(|i| 1.0 + i as f64 * 0.0001)
                .collect();
            
            let mut results = Vec::with_capacity(XLARGE_ARRAY_SIZE);
            for x in data {
                results.push(gamma(black_box(x)));
            }
            black_box(results)
        })
    });
    
    // Matrix operations
    group.bench_function("matrix_gamma_operations", |b| {
        b.iter(|| {
            let matrix = Array2::from_shape_fn((100, 100), |(i, j)| {
                1.0 + (i * 100 + j) as f64 * 0.01
            });
            
            let mut result = Array2::zeros((100, 100));
            for ((i, j), &val) in matrix.indexed_iter() {
                result[[i, j]] = gamma(black_box(val));
            }
            black_box(result)
        })
    });
    
    group.finish();
}

/// Algorithm switching benchmarks - test boundary conditions
fn algorithm_switching_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("algorithm_switching");
    group.measurement_time(Duration::from_secs(8));
    
    // Test around algorithm switching points for gamma function
    let gamma_test_points = vec![
        ("very_small", 0.1),
        ("small_boundary", 0.9),
        ("around_one", 1.1),
        ("medium_boundary", 7.9),
        ("medium_start", 8.1),
        ("large_boundary", 99.9),
        ("large_start", 100.1),
    ];
    
    for (name, value) in gamma_test_points {
        group.bench_function(&format!("gamma_{}", name), |b| {
            b.iter(|| gamma(black_box(value)))
        });
    }
    
    // Test around algorithm switching points for Bessel functions
    let bessel_test_points = vec![
        ("small_series", 1.0),
        ("series_boundary", 4.9),
        ("cf_start", 5.1),
        ("cf_boundary", 49.9),
        ("asymptotic_start", 50.1),
        ("large_asymptotic", 100.0),
    ];
    
    for (name, value) in bessel_test_points {
        group.bench_function(&format!("bessel_j0_{}", name), |b| {
            b.iter(|| bessel::j0(black_box(value)))
        });
    }
    
    group.finish();
}

/// Edge case and extreme value benchmarks
fn edge_case_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("edge_cases");
    group.measurement_time(Duration::from_secs(5));
    
    // Near-zero values
    group.bench_function("gamma_near_zero", |b| {
        b.iter(|| gamma(black_box(1e-10)))
    });
    
    group.bench_function("erf_near_zero", |b| {
        b.iter(|| erf(black_box(1e-10)))
    });
    
    // Large values (near overflow)
    group.bench_function("gamma_large_safe", |b| {
        b.iter(|| gamma(black_box(170.0)))
    });
    
    group.bench_function("erf_large", |b| {
        b.iter(|| erf(black_box(10.0)))
    });
    
    // Values near zeros of Bessel functions
    group.bench_function("bessel_j0_near_zero", |b| {
        b.iter(|| bessel::j0(black_box(2.4048255576957727686)))
    });
    
    group.bench_function("bessel_j1_near_zero", |b| {
        b.iter(|| bessel::j1(black_box(3.8317059702075123156)))
    });
    
    // Complex values with different argument ranges
    group.bench_function("gamma_complex_first_quad", |b| {
        b.iter(|| gamma_complex(black_box(Complex64::new(2.0, 1.0))))
    });
    
    group.bench_function("gamma_complex_imag_axis", |b| {
        b.iter(|| gamma_complex(black_box(Complex64::new(0.0, 5.0))))
    });
    
    group.finish();
}

/// Critical path benchmarks - functions used in hot paths
fn critical_path_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("critical_paths");
    group.measurement_time(Duration::from_secs(8));
    
    // Simulated scientific computing workload
    group.bench_function("scientific_workload", |b| {
        b.iter(|| {
            let mut result = 0.0;
            for i in 0..1000 {
                let x = 1.0 + i as f64 * 0.01;
                result += gamma(black_box(x)) * erf(black_box(x * 0.5));
            }
            black_box(result)
        })
    });
    
    // Machine learning workload simulation
    group.bench_function("ml_activation_functions", |b| {
        b.iter(|| {
            let mut results = Vec::with_capacity(1000);
            for i in 0..1000 {
                let x = -5.0 + i as f64 * 0.01;
                // Simulated activation function using erf
                results.push(0.5 * (1.0 + erf(black_box(x / std::f64::consts::SQRT_2))));
            }
            black_box(results)
        })
    });
    
    // Statistical computing workload
    group.bench_function("statistical_workload", |b| {
        b.iter(|| {
            let mut log_likelihood = 0.0;
            for i in 0..1000 {
                let alpha = 1.0 + i as f64 * 0.001;
                let beta = 2.0 + i as f64 * 0.0005;
                // Simulated Beta function log-likelihood
                log_likelihood += gammaln(black_box(alpha)) + gammaln(black_box(beta))
                    - gammaln(black_box(alpha + beta));
            }
            black_box(log_likelihood)
        })
    });
    
    group.finish();
}

/// Regression-prone areas - functions that historically had issues
fn regression_prone_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("regression_prone");
    group.measurement_time(Duration::from_secs(6));
    
    // Functions that have shown performance variations
    group.bench_function("gamma_cancellation_region", |b| {
        // Around gamma function poles where cancellation occurs
        b.iter(|| {
            let mut sum = 0.0;
            for i in 1..=10 {
                let x = -0.1 + i as f64 * 0.01; // Near negative integers
                sum += gamma(black_box(x)).abs();
            }
            black_box(sum)
        })
    });
    
    group.bench_function("bessel_oscillatory_region", |b| {
        // Large arguments where Bessel functions oscillate rapidly
        b.iter(|| {
            let mut sum = 0.0;
            for i in 0..100 {
                let x = 50.0 + i as f64 * 0.1;
                sum += bessel::j0(black_box(x)).abs();
            }
            black_box(sum)
        })
    });
    
    group.bench_function("hypergeometric_near_singularity", |b| {
        // Near z = 1 where hypergeometric functions have issues
        b.iter(|| {
            let mut sum = 0.0;
            for i in 0..100 {
                let z = 0.99 + i as f64 * 1e-5;
                // Simulated 2F1(1,1;2;z) which equals -ln(1-z)/z
                sum += -(1.0 - z).ln() / z;
            }
            black_box(sum)
        })
    });
    
    group.finish();
}

// Helper functions for complex benchmarks
fn gamma_complex(z: Complex64) -> Complex64 {
    // Placeholder implementation - would use actual complex gamma
    Complex64::new(gamma(z.re), 0.0)
}

// Benchmark groups
criterion_group!(
    regression_benchmarks,
    core_single_value_benchmarks,
    array_operation_benchmarks,
    memory_intensive_benchmarks,
    algorithm_switching_benchmarks,
    edge_case_benchmarks,
    critical_path_benchmarks,
    regression_prone_benchmarks,
);

criterion_main!(regression_benchmarks);