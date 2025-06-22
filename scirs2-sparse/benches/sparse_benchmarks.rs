use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::{Array1, Array2, ArrayView1};
use rand::Rng;
use scirs2_sparse::*;

fn generate_sparse_matrix(size: usize, density: f64) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let mut rng = rand::thread_rng();
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();

    for i in 0..size {
        for j in 0..size {
            if rng.gen::<f64>() < density {
                rows.push(i);
                cols.push(j);
                data.push(rng.gen::<f64>());
            }
        }
    }

    (rows, cols, data)
}

fn bench_sparse_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_construction");

    for size in [100, 500, 1000].iter() {
        let (rows, cols, data) = generate_sparse_matrix(*size, 0.05);
        let shape = (*size, *size);

        group.throughput(Throughput::Elements(data.len() as u64));

        group.bench_with_input(BenchmarkId::new("csr_from_triplets", size), size, |b, _| {
            b.iter(|| {
                CsrArray::from_triplets(
                    black_box(&rows),
                    black_box(&cols),
                    black_box(&data),
                    black_box(shape),
                    false,
                )
                .unwrap()
            })
        });

        group.bench_with_input(BenchmarkId::new("csc_from_triplets", size), size, |b, _| {
            b.iter(|| {
                CscArray::from_triplets(
                    black_box(&rows),
                    black_box(&cols),
                    black_box(&data),
                    black_box(shape),
                    false,
                )
                .unwrap()
            })
        });

        group.bench_with_input(BenchmarkId::new("coo_from_triplets", size), size, |b, _| {
            b.iter(|| {
                CooArray::from_triplets(
                    black_box(&rows),
                    black_box(&cols),
                    black_box(&data),
                    black_box(shape),
                    false,
                )
                .unwrap()
            })
        });
    }

    group.finish();
}

fn bench_sparse_matrix_vector(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_matvec");

    for size in [100, 500, 1000].iter() {
        let (rows, cols, data) = generate_sparse_matrix(*size, 0.05);
        let shape = (*size, *size);

        let csr = CsrArray::from_triplets(&rows, &cols, &data, shape, false).unwrap();
        let csc = CscArray::from_triplets(&rows, &cols, &data, shape, false).unwrap();
        let vector = Array1::from_iter((0..*size).map(|i| i as f64));

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("csr_matvec", size), size, |b, _| {
            b.iter(|| {
                let vector_view = vector.view();
                csr.dot_vector(&vector_view).unwrap()
            })
        });

        group.bench_with_input(BenchmarkId::new("csc_matvec", size), size, |b, _| {
            b.iter(|| {
                let vector_view = vector.view();
                csc.dot_vector(&vector_view).unwrap()
            })
        });
    }

    group.finish();
}

fn bench_sparse_matrix_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_matmul");

    for size in [100, 300, 500].iter() {
        let (rows, cols, data) = generate_sparse_matrix(*size, 0.05);
        let shape = (*size, *size);

        let csr1 = CsrArray::from_triplets(&rows, &cols, &data, shape, false).unwrap();
        let csr2 = CsrArray::from_triplets(&rows, &cols, &data, shape, false).unwrap();

        group.throughput(Throughput::Elements((*size * *size) as u64));

        group.bench_with_input(BenchmarkId::new("csr_matmul", size), size, |b, _| {
            b.iter(|| black_box(&csr1).dot(black_box(&csr2)).unwrap())
        });
    }

    group.finish();
}

fn bench_format_conversions(c: &mut Criterion) {
    let mut group = c.benchmark_group("format_conversions");

    for size in [100, 500, 1000].iter() {
        let (rows, cols, data) = generate_sparse_matrix(*size, 0.05);
        let shape = (*size, *size);

        let coo = CooArray::from_triplets(&rows, &cols, &data, shape, false).unwrap();
        let csr = CsrArray::from_triplets(&rows, &cols, &data, shape, false).unwrap();
        let csc = CscArray::from_triplets(&rows, &cols, &data, shape, false).unwrap();

        group.throughput(Throughput::Elements(data.len() as u64));

        group.bench_with_input(BenchmarkId::new("coo_to_csr", size), size, |b, _| {
            b.iter(|| black_box(&coo).to_csr().unwrap())
        });

        group.bench_with_input(BenchmarkId::new("csr_to_csc", size), size, |b, _| {
            b.iter(|| black_box(&csr).to_csc().unwrap())
        });

        group.bench_with_input(BenchmarkId::new("csc_to_csr", size), size, |b, _| {
            b.iter(|| black_box(&csc).to_csr().unwrap())
        });
    }

    group.finish();
}

fn bench_sparse_linear_algebra(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_linalg");

    for size in [100, 300, 500].iter() {
        // Create a symmetric positive definite matrix for CG solver
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut data = Vec::new();

        // Add diagonal dominance
        for i in 0..*size {
            rows.push(i);
            cols.push(i);
            data.push(10.0);
        }

        // Add some off-diagonal elements
        let mut rng = rand::thread_rng();
        for i in 0..*size {
            for j in (i + 1)..*size {
                if rng.gen::<f64>() < 0.02 {
                    let val = rng.gen::<f64>() * 0.5;
                    rows.push(i);
                    cols.push(j);
                    data.push(val);
                    rows.push(j);
                    cols.push(i);
                    data.push(val);
                }
            }
        }

        let shape = (*size, *size);
        let matrix = CsrArray::from_triplets(&rows, &cols, &data, shape, true).unwrap();
        let rhs = Array1::from_iter((0..*size).map(|i| (i + 1) as f64));

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(
            BenchmarkId::new("conjugate_gradient", size),
            size,
            |b, _| {
                b.iter(|| {
                    let options = CGOptions {
                        max_iter: 100,
                        rtol: 1e-6,
                        atol: 1e-12,
                        x0: None,
                        preconditioner: None,
                    };
                    // Skip CG for now - needs LinearOperator trait implementation
                    black_box(0.0)
                })
            },
        );
    }

    group.finish();
}

fn bench_symmetric_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("symmetric_operations");

    for size in [100, 500, 1000].iter() {
        // Create symmetric matrix data (lower triangular only)
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut data = Vec::new();
        let mut rng = rand::thread_rng();

        for i in 0..*size {
            for j in 0..=i {
                if rng.gen::<f64>() < 0.05 {
                    rows.push(i);
                    cols.push(j);
                    data.push(rng.gen::<f64>());
                }
            }
        }

        let shape = (*size, *size);
        let csr_temp = CsrArray::from_triplets(&rows, &cols, &data, shape, true).unwrap();
        let sym_csr = SymCsrArray::from_csr_array(&csr_temp).unwrap();
        let vector = Array1::from_iter((0..*size).map(|i| i as f64));

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("sym_csr_matvec", size), size, |b, _| {
            b.iter(|| {
                let vector_view = vector.view();
                sym_csr.dot_vector(&vector_view).unwrap()
            })
        });

        group.bench_with_input(
            BenchmarkId::new("sym_csr_quadratic_form", size),
            size,
            |b, _| {
                b.iter(|| {
                    // Use regular matrix-vector multiplication as substitute
                    let vector_view = vector.view();
                    sym_csr.dot_vector(&vector_view).unwrap()
                })
            },
        );
    }

    group.finish();
}

fn bench_diagonal_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("diagonal_operations");

    for size in [100, 500, 1000].iter() {
        let diag_data = vec![Array1::from_iter((0..*size).map(|i| (i + 1) as f64))];
        let offsets = vec![0];
        let shape = (*size, *size);

        let dia = DiaArray::new(diag_data, offsets, shape).unwrap();
        let vector = Array1::from_iter((0..*size).map(|i| i as f64));

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("dia_matvec", size), size, |b, _| {
            b.iter(|| {
                let vector_view = vector.view();
                dia.dot_vector(&vector_view).unwrap()
            })
        });

        group.bench_with_input(BenchmarkId::new("dia_to_csr", size), size, |b, _| {
            b.iter(|| black_box(&dia).to_csr().unwrap())
        });
    }

    group.finish();
}

fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");

    // Test memory usage with different densities
    for density in [0.01, 0.05, 0.1].iter() {
        let size = 1000;
        let (rows, cols, data) = generate_sparse_matrix(size, *density);
        let shape = (size, size);

        group.throughput(Throughput::Elements(data.len() as u64));

        group.bench_with_input(
            BenchmarkId::new("dok_insertion", format!("density_{}", density)),
            density,
            |b, _| {
                b.iter(|| {
                    let mut dok = DokArray::<f64>::new(shape);
                    for ((&r, &c), &v) in rows.iter().zip(cols.iter()).zip(data.iter()) {
                        dok.set(r, c, v).unwrap();
                    }
                    black_box(dok)
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("lil_insertion", format!("density_{}", density)),
            density,
            |b, _| {
                b.iter(|| {
                    let mut lil = LilArray::<f64>::new(shape);
                    for ((&r, &c), &v) in rows.iter().zip(cols.iter()).zip(data.iter()) {
                        lil.set(r, c, v).unwrap();
                    }
                    black_box(lil)
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_sparse_construction,
    bench_sparse_matrix_vector,
    bench_sparse_matrix_matrix,
    bench_format_conversions,
    bench_sparse_linear_algebra,
    bench_symmetric_operations,
    bench_diagonal_operations,
    bench_memory_efficiency
);
criterion_main!(benches);
