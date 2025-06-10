//! Sparse least squares optimization example
//!
//! This example demonstrates the use of sparsity-aware algorithms for large-scale
//! least squares problems, including L1-regularized (LASSO) regression and
//! sparse matrix operations.

use ndarray::{array, Array1, Array2, ArrayView1};
use scirs2_optimize::least_squares::{sparse_least_squares, SparseMatrix, SparseOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Sparse Least Squares Optimization Examples\n");

    // Example 1: Basic sparse least squares
    sparse_least_squares_example()?;

    // Example 2: L1-regularized regression (LASSO)
    lasso_regression_example()?;

    // Example 3: Large sparse matrix operations
    sparse_matrix_operations_example()?;

    Ok(())
}

/// Example 1: Basic sparse least squares problem
fn sparse_least_squares_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Example 1: Basic Sparse Least Squares ===");

    // Define a sparse linear system: Ax = b
    // A is a 100x50 sparse matrix with only ~5% non-zero elements
    let m = 100; // Number of observations
    let n = 50; // Number of parameters

    // Create a sparse system where only a few parameters are active
    let active_params = vec![1, 5, 10, 15, 25, 30, 40]; // 7 out of 50 parameters are non-zero
    let true_x = {
        let mut x = Array1::zeros(n);
        for &i in &active_params {
            x[i] = (i as f64 + 1.0) * 0.1; // Small but non-zero values
        }
        x
    };

    // Create sparse observation matrix and observations
    let (a_matrix, observations) = create_sparse_system(m, n, &active_params, &true_x);

    // Define the residual function
    let fun = move |x: &ArrayView1<f64>| {
        let predicted = a_matrix.dot(x);
        &predicted - &observations
    };

    // Set up sparse options
    let options = SparseOptions {
        max_iter: 1000,
        tol: 1e-8,
        sparsity_threshold: 1e-12,
        lambda: 0.0, // No L1 regularization for this example
        use_coordinate_descent: false,
        ..Default::default()
    };

    // Solve the sparse least squares problem
    let x0 = Array1::zeros(n);
    let result = sparse_least_squares(
        fun,
        None::<fn(&ArrayView1<f64>) -> Array2<f64>>,
        x0,
        Some(options),
    )?;

    println!("Sparse least squares completed:");
    println!("  Success: {}", result.success);
    println!("  Iterations: {}", result.nit);
    println!("  Function evaluations: {}", result.nfev);
    println!("  Final cost: {:.6}", result.cost);
    println!(
        "  Jacobian sparsity: {:.1}%",
        result.sparsity_info.jacobian_sparsity * 100.0
    );
    println!(
        "  Memory usage: {:.2} MB",
        result.sparsity_info.memory_usage_mb
    );

    // Compare with true solution
    let error = (&result.x - &true_x).mapv(|e| e.abs()).sum();
    println!("  Solution error (L1 norm): {:.6}", error);

    // Show which parameters were recovered as non-zero
    let recovered_params: Vec<usize> = result
        .x
        .iter()
        .enumerate()
        .filter(|(_, &val)| val.abs() > 1e-6)
        .map(|(i, _)| i)
        .collect();
    println!("  Active parameters (true): {:?}", active_params);
    println!("  Active parameters (recovered): {:?}", recovered_params);

    println!();
    Ok(())
}

/// Example 2: L1-regularized regression (LASSO)
fn lasso_regression_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Example 2: LASSO Regression ===");

    let m = 80; // Number of observations
    let n = 100; // Number of features (more features than observations)

    // Create a problem where only a few features are truly relevant
    let true_active = vec![10, 25, 50, 75, 90]; // 5 out of 100 features are active
    let true_coefficients = array![2.5, -1.8, 3.2, -2.1, 1.5]; // True coefficients

    // Generate design matrix with some correlation between features
    let mut design_matrix = Array2::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            // Add some structure to make the problem more realistic
            design_matrix[[i, j]] = 0.1 * (i as f64 * j as f64).sin() + 0.05 * (i + j) as f64;
        }
    }

    // Create true signal and add noise
    let mut true_signal = Array1::<f64>::zeros(m);
    for (idx, &feature_idx) in true_active.iter().enumerate() {
        for i in 0..m {
            true_signal[i] += true_coefficients[idx] * design_matrix[[i, feature_idx]];
        }
    }

    // Add noise
    let noise_level = 0.1;
    let observations = true_signal.mapv(|s: f64| s + noise_level * (rand::random::<f64>() - 0.5));

    // Try different L1 regularization strengths
    let lambda_values = vec![0.01, 0.1, 0.5, 1.0];

    for &lambda in &lambda_values {
        println!("LASSO with λ = {:.2}:", lambda);

        // Define the residual function (recreate for each iteration)
        let fun = {
            let design_matrix_clone = design_matrix.clone();
            let observations_clone = observations.clone();
            move |x: &ArrayView1<f64>| {
                let predicted = design_matrix_clone.dot(x);
                &predicted - &observations_clone
            }
        };

        let options = SparseOptions {
            max_iter: 1000,
            tol: 1e-6,
            lambda,
            use_coordinate_descent: true,
            ..Default::default()
        };

        let x0 = Array1::zeros(n);
        let result = sparse_least_squares(
            fun,
            None::<fn(&ArrayView1<f64>) -> Array2<f64>>,
            x0,
            Some(options),
        )?;

        // Count non-zero coefficients
        let num_nonzero = result.x.iter().filter(|&&val| val.abs() > 1e-6).count();
        let l1_norm = result.x.mapv(|x| x.abs()).sum();

        println!("  Non-zero coefficients: {}/", num_nonzero);
        println!("  L1 norm of solution: {:.4}", l1_norm);
        println!("  Final cost: {:.6}", result.cost);

        // Show which features were selected
        let selected_features: Vec<usize> = result
            .x
            .iter()
            .enumerate()
            .filter(|(_, &val)| val.abs() > 1e-6)
            .map(|(i, _)| i)
            .collect();
        println!("  Selected features: {:?}", selected_features);

        // Calculate overlap with true active features
        let overlap = selected_features
            .iter()
            .filter(|&&f| true_active.contains(&f))
            .count();
        println!(
            "  Correct feature selection: {}/{}",
            overlap,
            true_active.len()
        );
        println!();
    }

    Ok(())
}

/// Example 3: Sparse matrix operations
fn sparse_matrix_operations_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Example 3: Sparse Matrix Operations ===");

    // Create a large sparse matrix
    let size = 1000;
    let sparsity = 0.05; // 5% non-zero elements

    let mut dense_matrix = Array2::zeros((size, size));
    let mut nnz_count = 0;

    // Fill matrix with sparse pattern
    for i in 0..size {
        for j in 0..size {
            if rand::random::<f64>() < sparsity {
                dense_matrix[[i, j]] = rand::random::<f64>() * 2.0 - 1.0;
                nnz_count += 1;
            }
        }
        // Ensure diagonal dominance for numerical stability
        dense_matrix[[i, i]] += 1.0;
    }

    println!(
        "Dense matrix: {}x{} with {} non-zeros ({:.1}%)",
        size,
        size,
        nnz_count,
        (nnz_count as f64) / (size * size) as f64 * 100.0
    );

    // Convert to sparse format
    let sparse_matrix = SparseMatrix::from_dense(&dense_matrix.view(), 1e-12);

    println!("Sparse matrix conversion:");
    println!("  Stored non-zeros: {}", sparse_matrix.values.len());
    println!("  Sparsity ratio: {:.4}", sparse_matrix.sparsity_ratio());
    println!(
        "  Memory usage: {:.2} MB",
        estimate_sparse_memory_usage(&sparse_matrix)
    );

    // Test matrix-vector operations
    let x = Array1::from_shape_fn(size, |i| (i as f64 + 1.0).sin());

    // Dense operation
    let start_time = std::time::Instant::now();
    let y_dense = dense_matrix.dot(&x);
    let dense_time = start_time.elapsed();

    // Sparse operation
    let start_time = std::time::Instant::now();
    let y_sparse = sparse_matrix.matvec(&x.view());
    let sparse_time = start_time.elapsed();

    // Check accuracy
    let error = (&y_dense - &y_sparse).mapv(|e| e.abs()).sum();

    println!("Matrix-vector multiplication:");
    println!("  Dense operation time: {:.2} ms", dense_time.as_millis());
    println!("  Sparse operation time: {:.2} ms", sparse_time.as_millis());
    println!(
        "  Speedup: {:.1}x",
        dense_time.as_millis() as f64 / sparse_time.as_millis() as f64
    );
    println!("  Accuracy (L1 error): {:.2e}", error);

    // Test transpose operations
    let y = Array1::from_shape_fn(size, |i| (i as f64).cos());

    let start_time = std::time::Instant::now();
    let z_dense = dense_matrix.t().dot(&y);
    let dense_transpose_time = start_time.elapsed();

    let start_time = std::time::Instant::now();
    let z_sparse = sparse_matrix.transpose_matvec(&y.view());
    let sparse_transpose_time = start_time.elapsed();

    let transpose_error = (&z_dense - &z_sparse).mapv(|e| e.abs()).sum();

    println!("Transpose matrix-vector multiplication:");
    println!(
        "  Dense operation time: {:.2} ms",
        dense_transpose_time.as_millis()
    );
    println!(
        "  Sparse operation time: {:.2} ms",
        sparse_transpose_time.as_millis()
    );
    println!(
        "  Speedup: {:.1}x",
        dense_transpose_time.as_millis() as f64 / sparse_transpose_time.as_millis() as f64
    );
    println!("  Accuracy (L1 error): {:.2e}", transpose_error);

    Ok(())
}

/// Helper function to create a sparse system for testing
fn create_sparse_system(
    m: usize,
    n: usize,
    active_params: &[usize],
    true_x: &Array1<f64>,
) -> (Array2<f64>, Array1<f64>) {
    let mut a_matrix = Array2::zeros((m, n));

    // Create a sparse design matrix where only active parameters have non-zero columns
    for i in 0..m {
        for &j in active_params {
            // Add some structure to the design matrix
            a_matrix[[i, j]] = (i as f64 * j as f64).sin() + 0.1 * (i + j) as f64;
        }
        // Add small amounts to other columns to make it slightly less sparse
        for j in 0..n {
            if !active_params.contains(&j) && rand::random::<f64>() < 0.02 {
                a_matrix[[i, j]] = 0.01 * rand::random::<f64>();
            }
        }
    }

    // Create observations with some noise
    let clean_observations = a_matrix.dot(true_x);
    let noise_level = 0.01;
    let observations =
        clean_observations.mapv(|obs| obs + noise_level * (rand::random::<f64>() - 0.5));

    (a_matrix, observations)
}

/// Estimate memory usage of sparse matrix representation
fn estimate_sparse_memory_usage(sparse_matrix: &SparseMatrix) -> f64 {
    let nnz = sparse_matrix.values.len();
    let nrows = sparse_matrix.nrows;

    // Memory for values (f64), column indices (usize), and row pointers (usize)
    let memory_bytes = nnz * std::mem::size_of::<f64>()
        + nnz * std::mem::size_of::<usize>()
        + (nrows + 1) * std::mem::size_of::<usize>();

    memory_bytes as f64 / (1024.0 * 1024.0)
}
