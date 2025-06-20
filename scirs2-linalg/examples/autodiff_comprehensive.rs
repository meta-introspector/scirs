//! Comprehensive example of linear algebra operations with autodiff support
//!
//! NOTE: This example has been simplified due to API changes in scirs2-autograd.
//! Many linear algebra operations that were planned for autodiff support
//! (inv, det, svd, eig, pinv, sqrtm, logm, etc.) are not yet implemented
//! in the scirs2-autograd crate.
//!
//! This example demonstrates what is currently available and provides
//! workarounds for some common operations.

#[cfg(feature = "autograd")]
use ag::tensor_ops as T;
#[cfg(feature = "autograd")]
use scirs2_autograd as ag;

#[cfg(not(feature = "autograd"))]
fn main() {
    println!("This example requires the 'autograd' feature. Run with:");
    println!("cargo run --example autodiff_comprehensive --features=autograd");
}

#[cfg(feature = "autograd")]
fn main() {
    println!("Comprehensive Linear Algebra Operations with Automatic Differentiation");
    println!("===================================================================");
    println!("\nNOTE: Many operations are not yet available in scirs2-autograd.");
    println!("See src/autograd/mod.rs for the list of planned features.\n");

    // 1. Available Operations
    demo_available_operations();

    // 2. Matrix Calculus
    demo_matrix_calculus();

    // 3. Workarounds for Common Operations
    demo_workarounds();

    // 4. Performance Considerations
    demo_performance_tips();
}

#[cfg(feature = "autograd")]
fn demo_available_operations() {
    println!("1. Currently Available Operations");
    println!("--------------------------------");

    ag::run(|ctx: &mut ag::Context<f64>| {
        // Create matrix and vector placeholders
        let a = ctx.placeholder("a", &[2, 2]);
        let b = ctx.placeholder("b", &[2, 2]);
        let x = ctx.placeholder("x", &[2]);

        println!("Available operations:");
        println!("- Matrix multiplication: matmul(A, B)");
        let c = T::matmul(a, b);

        println!("- Transpose: transpose(A)");
        let a_t = T::transpose(a);

        println!("- Element-wise operations: A + B, A * B, etc.");
        let sum = a + b;
        let prod = a * b;

        println!("- Reductions: sum, mean, etc.");
        let total = T::reduce_sum(&c, &[], false);

        println!("- Diagonal extraction: extract_diag(A)");
        let diag = T::extract_diag(a);

        println!("- Matrix-vector operations (via matmul with expand_dims)");
        let x_col = T::expand_dims(x, 1);
        let y = T::matmul(a, x_col);
        let y_vec = T::squeeze(y, &[1]);

        // Feed values and evaluate
        let a_val = ag::ndarray::arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let b_val = ag::ndarray::arr2(&[[5.0, 6.0], [7.0, 8.0]]);
        let x_val = ag::ndarray::arr1(&[1.0, 2.0]);

        let results = ctx
            .evaluator()
            .push(&c)
            .push(&total)
            .push(&y_vec)
            .feed(a, a_val.view().into_dyn())
            .feed(b, b_val.view().into_dyn())
            .feed(x, x_val.view().into_dyn())
            .run();

        println!("\nResults:");
        println!("C = A @ B = \n{:?}", results[0]);
        println!("sum(C) = {:?}", results[1]);
        println!("y = A @ x = {:?}", results[2]);
    });

    println!();
}

#[cfg(feature = "autograd")]
fn demo_matrix_calculus() {
    println!("2. Matrix Calculus");
    println!("-----------------");

    ag::run(|ctx: &mut ag::Context<f64>| {
        // Demonstrate gradient computation for matrix functions
        let a = ctx.placeholder("a", &[2, 2]);

        // Define a scalar function of a matrix: f(A) = trace(A^T @ A)
        let a_t = T::transpose(a);
        let ata = T::matmul(a_t, a);
        let diag = T::extract_diag(ata);
        let f = T::reduce_sum(&diag, &[], false);

        // Compute gradient
        let grad_f = T::grad(&[f], &[a])[0];

        // Feed value
        let a_val = ag::ndarray::arr2(&[[1.0, 2.0], [3.0, 4.0]]);

        let results = ctx
            .evaluator()
            .push(&f)
            .push(&grad_f)
            .feed(a, a_val.view().into_dyn())
            .run();

        println!("A = \n{:?}", a_val);
        println!("f(A) = trace(A^T @ A) = {:?}", results[0]);
        println!("∇f(A) = 2A = \n{:?}", results[1]);

        // Demonstrate Jacobian computation
        let x = ctx.placeholder("x", &[3]);

        // Define vector function: g(x) = [x₁², x₁*x₂, x₂*x₃]
        let x1 = T::gather(x, &T::constant(vec![0i32], &[1], ctx), 0);
        let x2 = T::gather(x, &T::constant(vec![1i32], &[1], ctx), 0);
        let x3 = T::gather(x, &T::constant(vec![2i32], &[1], ctx), 0);

        let g1 = x1 * x1;
        let g2 = x1 * x2;
        let g3 = x2 * x3;
        let g = T::concat(&[g1, g2, g3], 0);

        // Compute Jacobian
        let jacobian = T::jacobian(&[g], &[x]);

        let x_val = ag::ndarray::arr1(&[2.0, 3.0, 4.0]);

        let jac_result = ctx
            .evaluator()
            .push(&jacobian[0])
            .feed(x, x_val.view().into_dyn())
            .run();

        println!("\nVector function g(x) = [x₁², x₁*x₂, x₂*x₃]");
        println!("x = {:?}", x_val);
        println!("Jacobian = \n{:?}", jac_result[0]);
    });

    println!();
}

#[cfg(feature = "autograd")]
fn demo_workarounds() {
    println!("3. Workarounds for Missing Operations");
    println!("------------------------------------");

    ag::run(|ctx: &mut ag::Context<f64>| {
        println!("Trace (using diagonal extraction):");
        let a = ctx.placeholder("a", &[3, 3]);
        let diag = T::extract_diag(a);
        let trace = T::reduce_sum(&diag, &[], false);

        println!("\nFrobenius norm (using element-wise operations):");
        let frob_squared = T::reduce_sum(&(a * a), &[], false);
        let frob_norm = T::sqrt(&frob_squared);

        println!("\nMatrix power (for small powers, use repeated multiplication):");
        let a2 = T::matmul(a, a); // A²
        let a3 = T::matmul(a2, a); // A³

        println!("\nIdentity matrix (manual construction):");
        let eye_data = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let eye = T::constant(eye_data, &[3, 3], ctx);

        // Feed and evaluate
        let a_val = ag::ndarray::arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);

        let results = ctx
            .evaluator()
            .push(&trace)
            .push(&frob_norm)
            .push(&a3)
            .feed(a, a_val.view().into_dyn())
            .run();

        println!("\nA = \n{:?}", a_val);
        println!("trace(A) = {:?}", results[0]);
        println!("||A||_F = {:?}", results[1]);
        println!("A³ = \n{:?}", results[2]);
    });

    println!();
}

#[cfg(feature = "autograd")]
fn demo_performance_tips() {
    println!("4. Performance Considerations");
    println!("----------------------------");

    println!("Tips for efficient autodiff with linear algebra:");
    println!("1. Batch operations when possible (reduces graph size)");
    println!("2. Use in-place operations where gradients aren't needed");
    println!("3. Consider manual gradient implementation for complex operations");
    println!("4. Profile memory usage for large matrices");
    println!("5. Use sparse matrices when appropriate (future feature)");

    ag::run(|ctx: &mut ag::Context<f64>| {
        // Example: Efficient batch matrix multiplication
        let batch_a = ctx.placeholder("batch_a", &[10, 3, 3]);
        let batch_b = ctx.placeholder("batch_b", &[10, 3, 3]);

        // This is more efficient than 10 separate matmuls
        let batch_c = T::matmul(batch_a, batch_b);

        println!("\nBatch operations example:");
        println!("- Input: 10 3x3 matrices");
        println!("- Single batched matmul is more efficient than loop");

        // For actual use, you would feed real batch data
    });

    println!("\nFor more examples, see the autograd module documentation.");
}
