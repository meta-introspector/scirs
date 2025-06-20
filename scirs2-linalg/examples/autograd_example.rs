//! Automatic differentiation example for linear algebra operations
//!
//! This example demonstrates the use of automatic differentiation
//! with various linear algebra operations in scirs2-linalg.
//!
//! NOTE: The autograd API has been simplified. Many operations that were
//! previously available directly (like det, inv, svd, etc.) are not yet
//! implemented in scirs2-autograd. This example shows what is currently
//! available. For the full planned API, see src/autograd/mod.rs.

#[cfg(feature = "autograd")]
use scirs2_autograd as ag;
#[cfg(feature = "autograd")]
use ag::tensor_ops as T;

#[cfg(not(feature = "autograd"))]
fn main() {
    println!("This example requires the 'autograd' feature. Run with:");
    println!("cargo run --example autograd_example --features=autograd");
}

#[cfg(feature = "autograd")]
fn main() {
    println!("SciRS2 Linear Algebra Automatic Differentiation Example");
    println!("====================================================\n");

    // Example 1: Matrix multiplication with gradient computation
    demo_matrix_multiplication();

    // Example 2: Matrix operations with trace
    demo_matrix_trace();

    // Example 3: Matrix norms
    demo_matrix_norms();

    // Example 4: Composite operations
    demo_composite_operations();

    println!("\nNOTE: Many linear algebra operations (det, inv, svd, eig, etc.)");
    println!("are not yet available in scirs2-autograd. See src/autograd/mod.rs");
    println!("for the list of planned features.");
}

#[cfg(feature = "autograd")]
fn demo_matrix_multiplication() {
    println!("1. Matrix Multiplication with Gradients");
    println!("-------------------------------------");

    ag::run(|ctx: &mut ag::Context<f64>| {
        // Create matrix placeholders
        let a = ctx.placeholder("a", &[2, 2]);
        let b = ctx.placeholder("b", &[2, 2]);

        // Perform matrix multiplication
        let c = T::matmul(a, b);

        // Compute loss (sum of all elements)
        let loss = T::reduce_sum(&c, &[], false);

        // Compute gradients
        let grads = T::grad(&[loss], &[a, b]);

        // Feed concrete values
        let a_val = ag::ndarray::arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let b_val = ag::ndarray::arr2(&[[5.0, 6.0], [7.0, 8.0]]);

        // Evaluate
        let result = ctx
            .evaluator()
            .push(&c)
            .extend(&grads)
            .feed(a, a_val.view().into_dyn())
            .feed(b, b_val.view().into_dyn())
            .run();

        println!("Matrix C = A * B:\n{:?}", result[0]);
        println!("\nGradient w.r.t. A:\n{:?}", result[1]);
        println!("\nGradient w.r.t. B:\n{:?}", result[2]);
    });

    println!();
}

#[cfg(feature = "autograd")]
fn demo_matrix_trace() {
    println!("2. Matrix Trace with Gradients");
    println!("------------------------------");

    ag::run(|ctx: &mut ag::Context<f64>| {
        // Create a matrix placeholder
        let a = ctx.placeholder("a", &[3, 3]);

        // Compute trace using diagonal extraction
        let diag = T::extract_diag(a);
        let trace = T::reduce_sum(&diag, &[], false);

        // Compute gradient
        let grad = T::grad(&[trace], &[a])[0];

        // Feed concrete value
        let a_val = ag::ndarray::arr2(&[[3.0, 1.0, 2.0], [2.0, 4.0, 1.0], [1.0, 2.0, 5.0]]);

        // Evaluate
        let results = ctx
            .evaluator()
            .push(&trace)
            .push(&grad)
            .feed(a, a_val.view().into_dyn())
            .run();

        println!("Matrix A:\n{:?}", a_val);
        println!("\nTrace(A) = {:?}", results[0]);
        println!("\nGradient of trace w.r.t. A:\n{:?}", results[1]);
    });

    println!();
}

#[cfg(feature = "autograd")]
fn demo_matrix_norms() {
    println!("3. Matrix Norms with Gradients");
    println!("------------------------------");

    ag::run(|ctx: &mut ag::Context<f64>| {
        // Create a matrix placeholder
        let a = ctx.placeholder("a", &[2, 3]);

        // Frobenius norm (element-wise L2 norm)
        let a_squared = a * a;
        let sum_squared = T::reduce_sum(&a_squared, &[], false);
        let frobenius_norm = T::sqrt(&sum_squared);

        // Compute gradient
        let grad = T::grad(&[frobenius_norm], &[a])[0];

        // Feed concrete value
        let a_val = ag::ndarray::arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

        // Evaluate
        let results = ctx
            .evaluator()
            .push(&frobenius_norm)
            .push(&grad)
            .feed(a, a_val.view().into_dyn())
            .run();

        println!("Matrix A:\n{:?}", a_val);
        println!("\nFrobenius norm of A = {:?}", results[0]);
        println!("\nGradient w.r.t. A:\n{:?}", results[1]);
    });

    println!();
}

#[cfg(feature = "autograd")]
fn demo_composite_operations() {
    println!("4. Composite Operations");
    println!("----------------------");

    ag::run(|ctx: &mut ag::Context<f64>| {
        // Create matrix placeholders
        let a = ctx.placeholder("a", &[2, 2]);
        let x = ctx.placeholder("x", &[2]);

        // Composite operation: f(A, x) = ||A*x||^2 + trace(A)
        let ax = T::matmul(a, T::expand_dims(x, 1));
        let ax_squeezed = T::squeeze(ax, &[1]);
        let norm_squared = T::reduce_sum(&(ax_squeezed * ax_squeezed), &[], false);
        
        let diag = T::extract_diag(a);
        let trace = T::reduce_sum(&diag, &[], false);
        
        let f = norm_squared + trace;

        // Compute gradients
        let grads = T::grad(&[f], &[a, x]);

        // Feed concrete values
        let a_val = ag::ndarray::arr2(&[[2.0, 1.0], [1.0, 3.0]]);
        let x_val = ag::ndarray::arr1(&[1.0, 2.0]);

        // Evaluate
        let results = ctx
            .evaluator()
            .push(&f)
            .extend(&grads)
            .feed(a, a_val.view().into_dyn())
            .feed(x, x_val.view().into_dyn())
            .run();

        println!("Matrix A:\n{:?}", a_val);
        println!("Vector x: {:?}", x_val);
        println!("\nf(A, x) = ||A*x||² + trace(A) = {:?}", results[0]);
        println!("\nGradient w.r.t. A:\n{:?}", results[1]);
        println!("\nGradient w.r.t. x:\n{:?}", results[2]);
    });
}