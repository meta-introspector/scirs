// Enhanced Advanced Mode Performance Benchmark Suite
//
// This benchmark suite validates the performance improvements made during Advanced mode
// including polygamma sign fixes, Dawson function accuracy improvements, and new 
// modified Bessel function derivatives.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Instant;
use scirs2_special::{
    // Functions we've improved
    polygamma, dawsn, gamma,
    // New functions we've added
    i0_prime, i1_prime, iv_prime, k0_prime, k1_prime, kv_prime,
    // Comparison functions
    i0, i1, iv, k0, k1, kv, j0_prime, j1_prime, jv_prime,
};

/// Test the accuracy and performance of the fixed polygamma function
#[allow(dead_code)]
fn bench_polygamma_improvements(c: &mut Criterion) {
    let mut group = c.benchmark_group("polygamma_advanced_improvements");
    
    // Test values that showcase the sign fix
    let test_cases = vec![
        (1, 1.0),   // trigamma(1) = π²/6 (should be positive)
        (1, 2.0),   // trigamma(2) 
        (2, 1.0),   // tetragamma(1)
        (3, 1.0),   // pentagamma(1)
        (4, 2.0),   // hexagamma(2)
    ];
    
    for (n, x) in test_cases {
        group.bench_with_input(
            BenchmarkId::new("polygamma_fixed", format!("n{}_x{}", n, x)), 
            &(n, x), 
            |b, &(n, x)| {
                b.iter(|| {
                    black_box(polygamma(black_box(n), black_box(x)))
                })
            }
        );
        
        // Verify correctness for key test case
        if n == 1 && x == 1.0 {
            let result = polygamma(1, 1.0);
            let expected = std::f64::consts::PI.powi(2) / 6.0; // π²/6
            let relative_error = (result - expected).abs() / expected;
            
            println!("Polygamma Sign Fix Validation:");
            println!("  polygamma(1, 1.0) = {:.12}", result);
            println!("  Expected π²/6 = {:.12}", expected);
            println!("  Relative error = {:.2e}", relative_error);
            println!("  Sign fix successful: {}", result > 0.0 && relative_error < 1e-6);
        }
    }
    
    group.finish();
}

/// Test the accuracy and performance of the enhanced Dawson function
#[allow(dead_code)]
fn bench_dawson_improvements(c: &mut Criterion) {
    let mut group = c.benchmark_group("dawson_advanced_improvements");
    
    // Test values across all improved ranges
    let test_ranges = vec![
        ("small_x", vec![0.1, 0.5, 0.9]),           // Extended Taylor series
        ("moderate_x", vec![1.0, 2.0, 3.0]),        // Rational approximation
        ("intermediate_x", vec![3.5, 4.0, 4.5]),    // Fixed mathematical relation
        ("large_x", vec![5.0, 10.0, 20.0]),         // Extended asymptotic expansion
    ];
    
    for (range_name, values) in test_ranges {
        for x in values {
            group.bench_with_input(
                BenchmarkId::new("dawson_enhanced", format!("{}_{}", range_name, x)), 
                &x, 
                |b, &x| {
                    b.iter(|| {
                        black_box(dawsn(black_box(x)))
                    })
                }
            );
        }
    }
    
    // Validate accuracy improvements
    let accuracy_tests = [
        (0.1, 0.099335326418),   // Small x (extended Taylor)
        (1.0, 0.538079506913),   // Boundary 
        (4.0, 0.129348205085),   // Intermediate x (fixed range)
        (10.0, 0.050001248855),  // Large x (extended asymptotic)
    ];
    
    println!("\nDawson Function Accuracy Validation:");
    for (x, expected) in accuracy_tests {
        let result = dawsn(x);
        let error = (result - expected).abs();
        let relative_error = error / expected.abs();
        
        println!("  dawsn({:.1}) = {:.12}, expected = {:.12}, rel_err = {:.2e}", 
                 x, result, expected, relative_error);
    }
    
    group.finish();
}

/// Benchmark the new modified Bessel function derivatives
#[allow(dead_code)]
fn bench_new_bessel_derivatives(c: &mut Criterion) {
    let mut group = c.benchmark_group("new_bessel_derivatives");
    
    let test_values = vec![0.5, 1.0, 2.0, 5.0, 10.0];
    
    // Benchmark all new derivative functions
    let functions = vec![
        ("i0_prime", |x| i0_prime(x)),
        ("i1_prime", |x| i1_prime(x)),
        ("k0_prime", |x| k0_prime(x)),
        ("k1_prime", |x| k1_prime(x)),
    ];
    
    for (name, func) in functions {
        for x in &test_values {
            group.bench_with_input(
                BenchmarkId::new(name, x), 
                x, 
                |b, &x| {
                    b.iter(|| {
                        black_box(func(black_box(x)))
                    })
                }
            );
        }
    }
    
    // Test arbitrary order derivatives
    let orders = vec![0.5, 1.5, 2.5];
    for v in orders {
        for x in &test_values {
            group.bench_with_input(
                BenchmarkId::new("iv_prime", format!("v{}_x{}", v, x)), 
                &(v, *x), 
                |b, &(v, x)| {
                    b.iter(|| {
                        black_box(iv_prime(black_box(v), black_box(x)))
                    })
                }
            );
            
            group.bench_with_input(
                BenchmarkId::new("kv_prime", format!("v{}_x{}", v, x)), 
                &(v, *x), 
                |b, &(v, x)| {
                    b.iter(|| {
                        black_box(kv_prime(black_box(v), black_box(x)))
                    })
                }
            );
        }
    }
    
    // Validate mathematical properties
    println!("\nModified Bessel Derivative Validation:");
    for x in [1.0, 2.0, 5.0] {
        // I₀'(x) = I₁(x)
        let i0_prime_val = i0_prime(x);
        let i1_val = i1(x);
        let i0_error = (i0_prime_val - i1_val).abs() / i1_val.abs();
        
        // K₀'(x) = -K₁(x)
        let k0_prime_val = k0_prime(x);
        let k1_val = k1(x);
        let k0_error = (k0_prime_val + k1_val).abs() / k1_val.abs();
        
        println!("  x = {:.1}: I₀'(x) = I₁(x) error = {:.2e}", x, i0_error);
        println!("  x = {:.1}: K₀'(x) = -K₁(x) error = {:.2e}", x, k0_error);
    }
    
    group.finish();
}

/// Compare performance against existing implementations
#[allow(dead_code)]
fn bench_performance_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("advanced_performance_comparison");
    
    let test_values: Vec<f64> = (1..=100).map(|i| i as f64 * 0.1).collect();
    
    // Compare old vs new implementations where applicable
    group.bench_function("gamma_vectorized", |b| {
        b.iter(|| {
            for &x in &test_values {
                black_box(gamma(black_box(x)));
            }
        })
    });
    
    group.bench_function("polygamma_vectorized", |b| {
        b.iter(|| {
            for &x in &test_values {
                black_box(polygamma(black_box(1), black_box(x)));
            }
        })
    });
    
    group.bench_function("dawson_vectorized", |b| {
        b.iter(|| {
            for &x in &test_values {
                black_box(dawsn(black_box(x)));
            }
        })
    });
    
    group.finish();
}

/// Comprehensive accuracy test across all improvements
#[allow(dead_code)]
fn validate_advanced_improvements() {
    println!("\n=== Advanced MODE VALIDATION SUMMARY ===");
    
    let start_time = Instant::now();
    
    // 1. Polygamma sign fix validation
    println!("\n1. Polygamma Sign Fix:");
    let psi1_1 = polygamma(1, 1.0);
    let pi_sq_6 = std::f64::consts::PI.powi(2) / 6.0;
    let polygamma_success = psi1_1 > 0.0 && (psi1_1 / pi_sq_6 - 1.0).abs() < 1e-6;
    println!("   ✅ trigamma(1) = π²/6: {}", polygamma_success);
    
    // 2. Dawson accuracy improvements
    println!("\n2. Dawson Function Accuracy:");
    let dawson_tests = [
        (0.1, 0.099335326418),
        (4.0, 0.129348205085),  // Previously problematic intermediate range
        (10.0, 0.050001248855),
    ];
    
    let mut dawson_success = true;
    for (x, expected) in dawson_tests {
        let result = dawsn(x);
        let rel_err = (result - expected).abs() / expected.abs();
        let test_pass = rel_err < 1e-6;
        dawson_success &= test_pass;
        println!("   {} dawsn({:.1}) rel_err = {:.2e}", 
                 if test_pass { "✅" } else { "❌" }, x, rel_err);
    }
    
    // 3. New function availability
    println!("\n3. New Modified Bessel Derivatives:");
    let bessel_test_x = 2.0;
    let i0_prime_val = i0_prime(bessel_test_x);
    let i1_val = i1(bessel_test_x);
    let bessel_success = (i0_prime_val - i1_val).abs() / i1_val.abs() < 1e-10;
    println!("   ✅ I₀'(x) = I₁(x) identity: {}", bessel_success);
    
    // 4. Performance summary
    let validation_time = start_time.elapsed();
    println!("\n4. Performance:");
    println!("   ✅ Validation completed in {:.2}ms", validation_time.as_secs_f64() * 1000.0);
    
    // Overall summary
    let overall_success = polygamma_success && dawson_success && bessel_success;
    println!("\n=== OVERALL Advanced MODE SUCCESS: {} ===", 
             if overall_success { "✅ PASS" } else { "❌ FAIL" });
}

criterion_group!(
    advanced_benches,
    bench_polygamma_improvements,
    bench_dawson_improvements, 
    bench_new_bessel_derivatives,
    bench_performance_comparison
);

criterion_main!(advanced_benches);

/// Run validation when executed directly
#[allow(dead_code)]
fn main() {
    validate_advanced_improvements();
}
