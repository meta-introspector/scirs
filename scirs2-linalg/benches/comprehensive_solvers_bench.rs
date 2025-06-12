//! Comprehensive benchmarks for linear system solvers
//!
//! This benchmark suite covers direct and iterative solvers for various
//! matrix types and system configurations, including specialized solvers
//! for structured matrices and advanced iterative methods.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::{Array1, Array2};
use scirs2_linalg::prelude::*;
use std::time::Duration;

/// Create a well-conditioned test matrix
fn create_test_matrix(n: usize) -> Array2<f64> {
    let mut matrix = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            if i == j {
                matrix[[i, j]] = (i + 1) as f64; // Diagonal dominance
            } else {
                matrix[[i, j]] = 0.1 * ((i * n + j) as f64 * 0.01).sin();
            }
        }
    }
    matrix
}

/// Create a symmetric positive definite matrix
fn create_spd_matrix(n: usize) -> Array2<f64> {
    let a = Array2::from_shape_fn((n, n), |(i, j)| ((i + j + 1) as f64 * 0.1).sin());
    a.t().dot(&a) + Array2::<f64>::eye(n) * (n as f64)
}

/// Create a tridiagonal matrix
fn create_tridiagonal_matrix(n: usize) -> Array2<f64> {
    let mut matrix = Array2::zeros((n, n));
    for i in 0..n {
        matrix[[i, i]] = 2.0; // Main diagonal
        if i > 0 {
            matrix[[i, i - 1]] = -1.0; // Lower diagonal
        }
        if i < n - 1 {
            matrix[[i, i + 1]] = -1.0; // Upper diagonal
        }
    }
    matrix
}

/// Create a banded matrix with specified bandwidth
fn create_banded_matrix(n: usize, bandwidth: usize) -> Array2<f64> {
    let mut matrix = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            if (i as isize - j as isize).abs() <= bandwidth as isize {
                if i == j {
                    matrix[[i, j]] = (i + 1) as f64;
                } else {
                    matrix[[i, j]] = 0.1 * ((i + j + 1) as f64 * 0.1).sin();
                }
            }
        }
    }
    matrix
}

/// Create an ill-conditioned matrix with specified condition number
fn create_ill_conditioned_matrix(n: usize, condition_number: f64) -> Array2<f64> {
    let mut matrix = Array2::zeros((n, n));
    for i in 0..n {
        matrix[[i, i]] = if i == 0 {
            condition_number
        } else {
            1.0 + (i as f64 * 0.1)
        };
    }
    // Apply random orthogonal transformation to mix eigenvalues
    let q = create_orthogonal_matrix(n);
    q.t().dot(&matrix).dot(&q)
}

/// Create an orthogonal matrix for transformations
fn create_orthogonal_matrix(n: usize) -> Array2<f64> {
    let a = Array2::from_shape_fn((n, n), |(i, j)| ((i + j + 1) as f64 * 0.1).sin());
    let (q, _) = qr(&a.view()).unwrap();
    q
}

/// Create a test vector
fn create_test_vector(n: usize) -> Array1<f64> {
    Array1::from_shape_fn(n, |i| ((i + 1) as f64 * 0.1).sin())
}

/// Create a rectangular matrix for overdetermined/underdetermined systems
fn create_rect_matrix(m: usize, n: usize) -> Array2<f64> {
    Array2::from_shape_fn((m, n), |(i, j)| {
        ((i + j + 1) as f64 * 0.1).sin() + 0.01 * (i as f64)
    })
}

/// Benchmark direct solvers for general matrices
fn bench_direct_solvers_general(c: &mut Criterion) {
    let mut group = c.benchmark_group("direct_solvers_general");
    group.sample_size(20);

    for &size in &[50, 100, 200, 500] {
        let matrix = create_test_matrix(size);
        let rhs = create_test_vector(size);

        group.throughput(Throughput::Elements(size as u64 * size as u64));

        // General linear solver (LU-based)
        group.bench_with_input(
            BenchmarkId::new("solve_lu", size),
            &(&matrix, &rhs),
            |b, (m, r)| b.iter(|| solve_lu(black_box(&m.view()), black_box(&r.view())).unwrap()),
        );

        // General linear solver with partial pivoting
        group.bench_with_input(
            BenchmarkId::new("solve_lu_partial_pivot", size),
            &(&matrix, &rhs),
            |b, (m, r)| {
                b.iter(|| {
                    solve_lu_partial_pivot(black_box(&m.view()), black_box(&r.view())).unwrap()
                })
            },
        );

        // General linear solver with full pivoting
        if size <= 200 {
            group.bench_with_input(
                BenchmarkId::new("solve_lu_full_pivot", size),
                &(&matrix, &rhs),
                |b, (m, r)| {
                    b.iter(|| {
                        solve_lu_full_pivot(black_box(&m.view()), black_box(&r.view())).unwrap()
                    })
                },
            );
        }

        // QR-based solver
        group.bench_with_input(
            BenchmarkId::new("solve_qr", size),
            &(&matrix, &rhs),
            |b, (m, r)| b.iter(|| solve_qr(black_box(&m.view()), black_box(&r.view())).unwrap()),
        );

        // SVD-based solver (most robust but slowest)
        if size <= 200 {
            group.bench_with_input(
                BenchmarkId::new("solve_svd", size),
                &(&matrix, &rhs),
                |b, (m, r)| {
                    b.iter(|| solve_svd(black_box(&m.view()), black_box(&r.view())).unwrap())
                },
            );
        }
    }

    group.finish();
}

/// Benchmark direct solvers for symmetric positive definite matrices
fn bench_direct_solvers_spd(c: &mut Criterion) {
    let mut group = c.benchmark_group("direct_solvers_spd");
    group.sample_size(25);

    for &size in &[50, 100, 200, 500] {
        let spd_matrix = create_spd_matrix(size);
        let rhs = create_test_vector(size);

        group.throughput(Throughput::Elements(size as u64 * size as u64));

        // Cholesky-based solver
        group.bench_with_input(
            BenchmarkId::new("solve_cholesky", size),
            &(&spd_matrix, &rhs),
            |b, (m, r)| {
                b.iter(|| solve_cholesky(black_box(&m.view()), black_box(&r.view())).unwrap())
            },
        );

        // LDLT-based solver
        group.bench_with_input(
            BenchmarkId::new("solve_ldlt", size),
            &(&spd_matrix, &rhs),
            |b, (m, r)| b.iter(|| solve_ldlt(black_box(&m.view()), black_box(&r.view())).unwrap()),
        );

        // Symmetric indefinite solver (even though matrix is positive definite)
        group.bench_with_input(
            BenchmarkId::new("solve_symmetric", size),
            &(&spd_matrix, &rhs),
            |b, (m, r)| {
                b.iter(|| solve_symmetric(black_box(&m.view()), black_box(&r.view())).unwrap())
            },
        );

        // Bunch-Kaufman solver
        group.bench_with_input(
            BenchmarkId::new("solve_bunch_kaufman", size),
            &(&spd_matrix, &rhs),
            |b, (m, r)| {
                b.iter(|| solve_bunch_kaufman(black_box(&m.view()), black_box(&r.view())).unwrap())
            },
        );
    }

    group.finish();
}

/// Benchmark triangular solvers
fn bench_triangular_solvers(c: &mut Criterion) {
    let mut group = c.benchmark_group("triangular_solvers");
    group.sample_size(30);

    for &size in &[100, 500, 1000, 2000] {
        let matrix = create_test_matrix(size);
        let rhs = create_test_vector(size);

        // Create triangular matrices
        let mut lower_triangular = matrix.clone();
        let mut upper_triangular = matrix.clone();

        for i in 0..size {
            for j in 0..size {
                if i < j {
                    lower_triangular[[i, j]] = 0.0;
                }
                if i > j {
                    upper_triangular[[i, j]] = 0.0;
                }
            }
        }

        group.throughput(Throughput::Elements(size as u64 * size as u64));

        // Lower triangular solver (forward substitution)
        group.bench_with_input(
            BenchmarkId::new("solve_triangular_lower", size),
            &(&lower_triangular, &rhs),
            |b, (m, r)| {
                b.iter(|| {
                    solve_triangular_lower(black_box(&m.view()), black_box(&r.view())).unwrap()
                })
            },
        );

        // Upper triangular solver (back substitution)
        group.bench_with_input(
            BenchmarkId::new("solve_triangular_upper", size),
            &(&upper_triangular, &rhs),
            |b, (m, r)| {
                b.iter(|| {
                    solve_triangular_upper(black_box(&m.view()), black_box(&r.view())).unwrap()
                })
            },
        );

        // Unit lower triangular solver
        let mut unit_lower = lower_triangular.clone();
        for i in 0..size {
            unit_lower[[i, i]] = 1.0;
        }
        group.bench_with_input(
            BenchmarkId::new("solve_unit_triangular_lower", size),
            &(&unit_lower, &rhs),
            |b, (m, r)| {
                b.iter(|| {
                    solve_unit_triangular_lower(black_box(&m.view()), black_box(&r.view())).unwrap()
                })
            },
        );

        // Multiple RHS triangular solve
        let rhs_multiple =
            Array2::from_shape_fn((size, 5), |(i, j)| ((i + j + 1) as f64 * 0.1).sin());
        group.bench_with_input(
            BenchmarkId::new("solve_triangular_multiple_rhs", size),
            &(&lower_triangular, &rhs_multiple),
            |b, (m, r)| {
                b.iter(|| {
                    solve_triangular_multiple_rhs(black_box(&m.view()), black_box(&r.view()))
                        .unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark banded matrix solvers
fn bench_banded_solvers(c: &mut Criterion) {
    let mut group = c.benchmark_group("banded_solvers");
    group.sample_size(25);

    for &size in &[100, 500, 1000] {
        for &bandwidth in &[3, 7, 15] {
            let banded_matrix = create_banded_matrix(size, bandwidth);
            let rhs = create_test_vector(size);

            group.throughput(Throughput::Elements(size as u64 * bandwidth as u64));

            // General banded solver
            group.bench_with_input(
                BenchmarkId::new(format!("solve_banded_bw_{}", bandwidth), size),
                &(&banded_matrix, &rhs),
                |b, (m, r)| {
                    b.iter(|| {
                        solve_banded(black_box(&m.view()), black_box(&r.view()), bandwidth).unwrap()
                    })
                },
            );

            // Symmetric banded solver
            group.bench_with_input(
                BenchmarkId::new(format!("solve_symmetric_banded_bw_{}", bandwidth), size),
                &(&banded_matrix, &rhs),
                |b, (m, r)| {
                    b.iter(|| {
                        solve_symmetric_banded(
                            black_box(&m.view()),
                            black_box(&r.view()),
                            bandwidth,
                        )
                        .unwrap()
                    })
                },
            );
        }

        // Tridiagonal solver (special case of banded)
        let tridiagonal = create_tridiagonal_matrix(size);
        let rhs = create_test_vector(size);

        group.bench_with_input(
            BenchmarkId::new("solve_tridiagonal", size),
            &(&tridiagonal, &rhs),
            |b, (m, r)| {
                b.iter(|| solve_tridiagonal(black_box(&m.view()), black_box(&r.view())).unwrap())
            },
        );

        // Tridiagonal solver (Thomas algorithm)
        group.bench_with_input(
            BenchmarkId::new("solve_tridiagonal_thomas", size),
            &(&tridiagonal, &rhs),
            |b, (m, r)| {
                b.iter(|| {
                    solve_tridiagonal_thomas(black_box(&m.view()), black_box(&r.view())).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark least squares solvers
fn bench_least_squares_solvers(c: &mut Criterion) {
    let mut group = c.benchmark_group("least_squares_solvers");
    group.sample_size(20);

    for &m in &[100, 200, 500] {
        for &n in &[50, 100, 200] {
            if m <= n {
                continue;
            } // Only overdetermined systems

            let overdetermined = create_rect_matrix(m, n);
            let rhs = create_test_vector(m);

            group.throughput(Throughput::Elements(m as u64 * n as u64));

            // Normal equations solver
            group.bench_with_input(
                BenchmarkId::new(format!("lstsq_normal_equations_{}x{}", m, n), n),
                &(&overdetermined, &rhs),
                |b, (a, r)| {
                    b.iter(|| {
                        lstsq_normal_equations(black_box(&a.view()), black_box(&r.view())).unwrap()
                    })
                },
            );

            // QR-based least squares
            group.bench_with_input(
                BenchmarkId::new(format!("lstsq_qr_{}x{}", m, n), n),
                &(&overdetermined, &rhs),
                |b, (a, r)| {
                    b.iter(|| lstsq_qr(black_box(&a.view()), black_box(&r.view())).unwrap())
                },
            );

            // SVD-based least squares
            if n <= 100 {
                group.bench_with_input(
                    BenchmarkId::new(format!("lstsq_svd_{}x{}", m, n), n),
                    &(&overdetermined, &rhs),
                    |b, (a, r)| {
                        b.iter(|| lstsq_svd(black_box(&a.view()), black_box(&r.view())).unwrap())
                    },
                );
            }

            // Iterative least squares (LSQR)
            group.bench_with_input(
                BenchmarkId::new(format!("lstsq_lsqr_{}x{}", m, n), n),
                &(&overdetermined, &rhs),
                |b, (a, r)| {
                    b.iter(|| lsqr(black_box(&a.view()), black_box(&r.view()), 100, 1e-10).unwrap())
                },
            );

            // Iterative least squares (LSMR)
            group.bench_with_input(
                BenchmarkId::new(format!("lstsq_lsmr_{}x{}", m, n), n),
                &(&overdetermined, &rhs),
                |b, (a, r)| {
                    b.iter(|| lsmr(black_box(&a.view()), black_box(&r.view()), 100, 1e-10).unwrap())
                },
            );
        }
    }

    group.finish();
}

/// Benchmark Krylov subspace iterative solvers
fn bench_krylov_solvers(c: &mut Criterion) {
    let mut group = c.benchmark_group("krylov_solvers");
    group.sample_size(15);
    group.measurement_time(Duration::from_secs(30));

    for &size in &[100, 500, 1000] {
        let spd_matrix = create_spd_matrix(size);
        let general_matrix = create_test_matrix(size);
        let rhs = create_test_vector(size);

        group.throughput(Throughput::Elements(size as u64 * size as u64));

        // Conjugate Gradient (SPD matrices)
        group.bench_with_input(
            BenchmarkId::new("cg", size),
            &(&spd_matrix, &rhs),
            |b, (m, r)| {
                b.iter(|| {
                    conjugate_gradient(black_box(&m.view()), black_box(&r.view()), 200, 1e-10)
                        .unwrap()
                })
            },
        );

        // Preconditioned Conjugate Gradient
        group.bench_with_input(
            BenchmarkId::new("pcg_jacobi", size),
            &(&spd_matrix, &rhs),
            |b, (m, r)| {
                b.iter(|| {
                    pcg_jacobi(black_box(&m.view()), black_box(&r.view()), 200, 1e-10).unwrap()
                })
            },
        );

        // GMRES (general matrices)
        group.bench_with_input(
            BenchmarkId::new("gmres", size),
            &(&general_matrix, &rhs),
            |b, (m, r)| {
                b.iter(|| {
                    gmres(black_box(&m.view()), black_box(&r.view()), 50, 200, 1e-10).unwrap()
                })
            },
        );

        // BiCGSTAB (general matrices)
        group.bench_with_input(
            BenchmarkId::new("bicgstab", size),
            &(&general_matrix, &rhs),
            |b, (m, r)| {
                b.iter(|| bicgstab(black_box(&m.view()), black_box(&r.view()), 200, 1e-10).unwrap())
            },
        );

        // QMR (Quasi-Minimal Residual)
        group.bench_with_input(
            BenchmarkId::new("qmr", size),
            &(&general_matrix, &rhs),
            |b, (m, r)| {
                b.iter(|| qmr(black_box(&m.view()), black_box(&r.view()), 200, 1e-10).unwrap())
            },
        );

        // MINRES (symmetric indefinite)
        group.bench_with_input(
            BenchmarkId::new("minres", size),
            &(&spd_matrix, &rhs),
            |b, (m, r)| {
                b.iter(|| minres(black_box(&m.view()), black_box(&r.view()), 200, 1e-10).unwrap())
            },
        );

        // Flexible GMRES
        group.bench_with_input(
            BenchmarkId::new("fgmres", size),
            &(&general_matrix, &rhs),
            |b, (m, r)| {
                b.iter(|| {
                    fgmres(black_box(&m.view()), black_box(&r.view()), 50, 200, 1e-10).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark stationary iterative solvers
fn bench_stationary_solvers(c: &mut Criterion) {
    let mut group = c.benchmark_group("stationary_solvers");
    group.sample_size(20);

    for &size in &[100, 500, 1000] {
        let matrix = create_test_matrix(size);
        let rhs = create_test_vector(size);

        group.throughput(Throughput::Elements(size as u64 * size as u64));

        // Jacobi method
        group.bench_with_input(
            BenchmarkId::new("jacobi", size),
            &(&matrix, &rhs),
            |b, (m, r)| {
                b.iter(|| {
                    jacobi_method(black_box(&m.view()), black_box(&r.view()), 100, 1e-10).unwrap()
                })
            },
        );

        // Gauss-Seidel method
        group.bench_with_input(
            BenchmarkId::new("gauss_seidel", size),
            &(&matrix, &rhs),
            |b, (m, r)| {
                b.iter(|| {
                    gauss_seidel(black_box(&m.view()), black_box(&r.view()), 100, 1e-10).unwrap()
                })
            },
        );

        // Successive Over-Relaxation (SOR)
        group.bench_with_input(
            BenchmarkId::new("sor_1.2", size),
            &(&matrix, &rhs),
            |b, (m, r)| {
                b.iter(|| sor(black_box(&m.view()), black_box(&r.view()), 1.2, 100, 1e-10).unwrap())
            },
        );

        group.bench_with_input(
            BenchmarkId::new("sor_1.8", size),
            &(&matrix, &rhs),
            |b, (m, r)| {
                b.iter(|| sor(black_box(&m.view()), black_box(&r.view()), 1.8, 100, 1e-10).unwrap())
            },
        );

        // Symmetric SOR (SSOR)
        group.bench_with_input(
            BenchmarkId::new("ssor", size),
            &(&matrix, &rhs),
            |b, (m, r)| {
                b.iter(|| {
                    ssor(black_box(&m.view()), black_box(&r.view()), 1.2, 100, 1e-10).unwrap()
                })
            },
        );

        // Richardson iteration
        group.bench_with_input(
            BenchmarkId::new("richardson", size),
            &(&matrix, &rhs),
            |b, (m, r)| {
                b.iter(|| {
                    richardson(black_box(&m.view()), black_box(&r.view()), 0.1, 100, 1e-10).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark multigrid solvers
fn bench_multigrid_solvers(c: &mut Criterion) {
    let mut group = c.benchmark_group("multigrid_solvers");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(40));

    for &size in &[127, 255, 511] {
        // Use sizes that are 2^n - 1 for easy coarsening
        let matrix = create_test_matrix(size);
        let rhs = create_test_vector(size);

        group.throughput(Throughput::Elements(size as u64 * size as u64));

        // Geometric multigrid (V-cycle)
        group.bench_with_input(
            BenchmarkId::new("multigrid_v_cycle", size),
            &(&matrix, &rhs),
            |b, (m, r)| {
                b.iter(|| {
                    multigrid_v_cycle(black_box(&m.view()), black_box(&r.view()), 10, 1e-10)
                        .unwrap()
                })
            },
        );

        // Geometric multigrid (W-cycle)
        group.bench_with_input(
            BenchmarkId::new("multigrid_w_cycle", size),
            &(&matrix, &rhs),
            |b, (m, r)| {
                b.iter(|| {
                    multigrid_w_cycle(black_box(&m.view()), black_box(&r.view()), 10, 1e-10)
                        .unwrap()
                })
            },
        );

        // Algebraic multigrid
        group.bench_with_input(
            BenchmarkId::new("algebraic_multigrid", size),
            &(&matrix, &rhs),
            |b, (m, r)| {
                b.iter(|| {
                    algebraic_multigrid(black_box(&m.view()), black_box(&r.view()), 10, 1e-10)
                        .unwrap()
                })
            },
        );

        // Full Multigrid (FMG)
        group.bench_with_input(
            BenchmarkId::new("full_multigrid", size),
            &(&matrix, &rhs),
            |b, (m, r)| {
                b.iter(|| {
                    full_multigrid(black_box(&m.view()), black_box(&r.view()), 10, 1e-10).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark iterative refinement and conditioning
fn bench_iterative_refinement(c: &mut Criterion) {
    let mut group = c.benchmark_group("iterative_refinement");
    group.sample_size(20);

    for &size in &[100, 200, 500] {
        let well_conditioned = create_test_matrix(size);
        let ill_conditioned = create_ill_conditioned_matrix(size, 1e8);
        let rhs = create_test_vector(size);

        group.throughput(Throughput::Elements(size as u64 * size as u64));

        // Standard LU solve (well-conditioned)
        group.bench_with_input(
            BenchmarkId::new("solve_lu_well_conditioned", size),
            &(&well_conditioned, &rhs),
            |b, (m, r)| b.iter(|| solve_lu(black_box(&m.view()), black_box(&r.view())).unwrap()),
        );

        // LU solve with iterative refinement (well-conditioned)
        group.bench_with_input(
            BenchmarkId::new("solve_lu_with_refinement_well", size),
            &(&well_conditioned, &rhs),
            |b, (m, r)| {
                b.iter(|| {
                    solve_lu_with_refinement(black_box(&m.view()), black_box(&r.view()), 3).unwrap()
                })
            },
        );

        // Standard LU solve (ill-conditioned)
        if size <= 200 {
            group.bench_with_input(
                BenchmarkId::new("solve_lu_ill_conditioned", size),
                &(&ill_conditioned, &rhs),
                |b, (m, r)| {
                    b.iter(|| solve_lu(black_box(&m.view()), black_box(&r.view())).unwrap())
                },
            );

            // LU solve with iterative refinement (ill-conditioned)
            group.bench_with_input(
                BenchmarkId::new("solve_lu_with_refinement_ill", size),
                &(&ill_conditioned, &rhs),
                |b, (m, r)| {
                    b.iter(|| {
                        solve_lu_with_refinement(black_box(&m.view()), black_box(&r.view()), 5)
                            .unwrap()
                    })
                },
            );

            // Mixed precision iterative refinement
            group.bench_with_input(
                BenchmarkId::new("solve_mixed_precision_refinement", size),
                &(&ill_conditioned, &rhs),
                |b, (m, r)| {
                    b.iter(|| {
                        solve_mixed_precision_refinement(
                            black_box(&m.view()),
                            black_box(&r.view()),
                            5,
                        )
                        .unwrap()
                    })
                },
            );
        }
    }

    group.finish();
}

/// Benchmark solver performance with different matrix types
fn bench_solver_matrix_types(c: &mut Criterion) {
    let mut group = c.benchmark_group("solver_matrix_types");
    group.sample_size(20);

    let size = 200;
    let rhs = create_test_vector(size);

    // Different matrix types
    let well_conditioned = create_test_matrix(size);
    let spd_matrix = create_spd_matrix(size);
    let tridiagonal = create_tridiagonal_matrix(size);
    let banded = create_banded_matrix(size, 5);
    let ill_conditioned = create_ill_conditioned_matrix(size, 1e6);

    group.throughput(Throughput::Elements(size as u64 * size as u64));

    // Compare solvers on different matrix types
    for (matrix_type, matrix) in [
        ("well_conditioned", &well_conditioned),
        ("spd", &spd_matrix),
        ("tridiagonal", &tridiagonal),
        ("banded", &banded),
        ("ill_conditioned", &ill_conditioned),
    ] {
        // LU solver
        group.bench_with_input(
            BenchmarkId::new(format!("lu_{}", matrix_type), size),
            &(matrix, &rhs),
            |b, (m, r)| b.iter(|| solve_lu(black_box(&m.view()), black_box(&r.view())).unwrap()),
        );

        // QR solver
        group.bench_with_input(
            BenchmarkId::new(format!("qr_{}", matrix_type), size),
            &(matrix, &rhs),
            |b, (m, r)| b.iter(|| solve_qr(black_box(&m.view()), black_box(&r.view())).unwrap()),
        );

        // Iterative solver (GMRES)
        group.bench_with_input(
            BenchmarkId::new(format!("gmres_{}", matrix_type), size),
            &(matrix, &rhs),
            |b, (m, r)| {
                b.iter(|| {
                    gmres(black_box(&m.view()), black_box(&r.view()), 20, 100, 1e-10).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark parallel solver implementations
fn bench_parallel_solvers(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_solvers");
    group.sample_size(15);

    for &size in &[200, 500, 1000] {
        let matrix = create_test_matrix(size);
        let rhs = create_test_vector(size);

        group.throughput(Throughput::Elements(size as u64 * size as u64));

        // Sequential vs parallel LU
        group.bench_with_input(
            BenchmarkId::new("lu_sequential", size),
            &(&matrix, &rhs),
            |b, (m, r)| {
                b.iter(|| solve_lu_sequential(black_box(&m.view()), black_box(&r.view())).unwrap())
            },
        );

        group.bench_with_input(
            BenchmarkId::new("lu_parallel", size),
            &(&matrix, &rhs),
            |b, (m, r)| {
                b.iter(|| solve_lu_parallel(black_box(&m.view()), black_box(&r.view())).unwrap())
            },
        );

        // Parallel iterative solvers
        group.bench_with_input(
            BenchmarkId::new("gmres_parallel", size),
            &(&matrix, &rhs),
            |b, (m, r)| {
                b.iter(|| {
                    gmres_parallel(black_box(&m.view()), black_box(&r.view()), 20, 100, 1e-10)
                        .unwrap()
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("cg_parallel", size),
            &(&matrix, &rhs),
            |b, (m, r)| {
                b.iter(|| {
                    conjugate_gradient_parallel(
                        black_box(&m.view()),
                        black_box(&r.view()),
                        100,
                        1e-10,
                    )
                    .unwrap()
                })
            },
        );
    }

    group.finish();
}

// Group all benchmarks
criterion_group!(
    benches,
    bench_direct_solvers_general,
    bench_direct_solvers_spd,
    bench_triangular_solvers,
    bench_banded_solvers,
    bench_least_squares_solvers,
    bench_krylov_solvers,
    bench_stationary_solvers,
    bench_multigrid_solvers,
    bench_iterative_refinement,
    bench_solver_matrix_types,
    bench_parallel_solvers
);

criterion_main!(benches);
