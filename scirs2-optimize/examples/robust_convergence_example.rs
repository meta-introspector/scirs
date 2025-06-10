//! Robust convergence criteria example
//!
//! This example demonstrates the robust convergence detection system that provides:
//! - Multiple stopping criteria with early stopping
//! - Noise-robust convergence for difficult functions
//! - Progress-based and plateau detection
//! - Adaptive tolerance selection
//! - Problem-specific convergence configuration

use ndarray::{Array1, ArrayView1};
use scirs2_optimize::unconstrained::{
    create_robust_options_for_problem, RobustConvergenceOptions, RobustConvergenceState,
};
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Robust Convergence Criteria Examples\n");

    // Example 1: Easy well-conditioned problem
    well_conditioned_example()?;

    // Example 2: Difficult noisy function
    noisy_function_example()?;

    // Example 3: Plateau detection
    plateau_detection_example()?;

    // Example 4: Early stopping for slow convergence
    early_stopping_example()?;

    // Example 5: Multiple criteria requirement
    multiple_criteria_example()?;

    // Example 6: Custom convergence configuration
    custom_convergence_example()?;

    Ok(())
}

/// Example 1: Well-conditioned quadratic function
fn well_conditioned_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Well-Conditioned Quadratic Function ===");

    let n = 50;
    let fun = |x: &ArrayView1<f64>| -> f64 { x.mapv(|xi| xi.powi(2)).sum() };

    let grad = |x: &ArrayView1<f64>| -> Array1<f64> { x.mapv(|xi| 2.0 * xi) };

    // Use problem-specific robust options
    let options = create_robust_options_for_problem("well_conditioned", n, "easy");
    let mut convergence_state = RobustConvergenceState::new(options, n);
    convergence_state.start_timing();

    let mut x = Array1::ones(n) * 0.5; // Start away from optimum
    let mut iteration = 0;

    println!("Configuration:");
    println!(
        "  Early stopping: {}",
        convergence_state.options.enable_early_stopping
    );
    println!(
        "  Noise robust: {}",
        convergence_state.options.enable_noise_robust
    );
    println!(
        "  Multiple criteria: {}",
        convergence_state.options.require_multiple_criteria
    );

    let start = Instant::now();

    // Simple gradient descent optimization
    loop {
        let f_val = fun(&x.view());
        let grad_val = grad(&x.view());
        let grad_norm = grad_val.mapv(|g| g.abs()).sum();

        // Simple step
        let step_size = 0.01;
        let step = &grad_val * (-step_size);
        let step_norm = step.mapv(|s| s.abs()).sum();

        x = &x + &step;

        // Check convergence
        let convergence_result = convergence_state.update_and_check_convergence(
            f_val,
            grad_norm,
            step_norm,
            iteration,
            Some(&x.view()),
        )?;

        if convergence_result.converged || iteration >= 1000 {
            println!("\nResults:");
            println!("  Converged: {}", convergence_result.converged);
            println!("  Iterations: {}", iteration);
            println!("  Final function value: {:.6e}", f_val);
            println!("  Final gradient norm: {:.6e}", grad_norm);
            println!("  Message: {}", convergence_result.get_message());
            println!("  Criteria met: {}", convergence_result.criteria_met);

            if !convergence_result.convergence_reasons.is_empty() {
                println!(
                    "  Convergence reasons: {}",
                    convergence_result.convergence_reasons.join(", ")
                );
            }

            println!("  Optimization time: {:.3} ms", start.elapsed().as_millis());
            break;
        }

        iteration += 1;
    }

    println!();
    Ok(())
}

/// Example 2: Noisy function optimization
fn noisy_function_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Noisy Function Optimization ===");

    let n = 20;

    // Function with added noise
    let fun = |x: &ArrayView1<f64>| -> f64 {
        let clean_value = x.mapv(|xi| xi.powi(2)).sum();
        let noise = (iteration_counter() as f64 * 0.123).sin() * 1e-6; // Deterministic "noise"
        clean_value + noise
    };

    let grad = |x: &ArrayView1<f64>| -> Array1<f64> {
        let clean_grad = x.mapv(|xi| 2.0 * xi);
        // Add some noise to gradient too
        let noise_factor = (iteration_counter() as f64 * 0.456).cos() * 1e-8;
        clean_grad.mapv(|g| g + noise_factor)
    };

    // Use robust options for noisy problems
    let options = create_robust_options_for_problem("noisy", n, "moderate");
    let mut convergence_state = RobustConvergenceState::new(options, n);
    convergence_state.start_timing();

    let mut x = Array1::ones(n) * 0.3;
    let mut iteration = 0;

    println!("Configuration:");
    println!(
        "  Noise robust: {}",
        convergence_state.options.enable_noise_robust
    );
    println!("  Noise window: {}", convergence_state.options.noise_window);
    println!(
        "  Noise confidence: {:.1}%",
        convergence_state.options.noise_confidence * 100.0
    );

    // Reset iteration counter
    reset_iteration_counter();

    loop {
        increment_iteration_counter();

        let f_val = fun(&x.view());
        let grad_val = grad(&x.view());
        let grad_norm = grad_val.mapv(|g| g.abs()).sum();

        let step_size = 0.01;
        let step = &grad_val * (-step_size);
        let step_norm = step.mapv(|s| s.abs()).sum();

        x = &x + &step;

        let convergence_result = convergence_state.update_and_check_convergence(
            f_val,
            grad_norm,
            step_norm,
            iteration,
            Some(&x.view()),
        )?;

        if convergence_result.converged || iteration >= 2000 {
            println!("\nResults:");
            println!("  Converged: {}", convergence_result.converged);
            println!("  Iterations: {}", iteration);
            println!("  Final function value: {:.6e}", f_val);
            println!(
                "  Noise confidence: {:.1}%",
                convergence_result.noise_robust_confidence * 100.0
            );

            if convergence_result.converged {
                println!(
                    "  Convergence reasons: {}",
                    convergence_result.convergence_reasons.join(", ")
                );
            }

            if !convergence_result.warning_flags.is_empty() {
                println!(
                    "  Warnings: {}",
                    convergence_result.warning_flags.join(", ")
                );
            }

            break;
        }

        iteration += 1;
    }

    println!();
    Ok(())
}

/// Example 3: Function with plateau
fn plateau_detection_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Plateau Detection Example ===");

    let fun = |x: &ArrayView1<f64>| -> f64 {
        // Function that plateaus around x = 0.1
        if x[0].abs() < 0.15 {
            1e-8 // Very flat region
        } else {
            x[0].powi(2)
        }
    };

    let grad = |x: &ArrayView1<f64>| -> Array1<f64> {
        if x[0].abs() < 0.15 {
            Array1::zeros(1) // Zero gradient in plateau
        } else {
            Array1::from_vec(vec![2.0 * x[0]])
        }
    };

    let mut options = RobustConvergenceOptions::default();
    options.enable_plateau_detection = true;
    options.plateau_window = 8;
    options.plateau_tolerance = 1e-10;
    options.enable_early_stopping = true;
    options.early_stopping_patience = 15;

    let mut convergence_state = RobustConvergenceState::new(options, 1);
    convergence_state.start_timing();

    let mut x = Array1::from_vec(vec![0.5]); // Start outside plateau
    let mut iteration = 0;

    println!("Configuration:");
    println!(
        "  Plateau detection: {}",
        convergence_state.options.enable_plateau_detection
    );
    println!(
        "  Plateau window: {}",
        convergence_state.options.plateau_window
    );
    println!(
        "  Plateau tolerance: {:.1e}",
        convergence_state.options.plateau_tolerance
    );

    loop {
        let f_val = fun(&x.view());
        let grad_val = grad(&x.view());
        let grad_norm = grad_val.mapv(|g| g.abs()).sum();

        // Aggressive step to reach plateau quickly
        let step_size = 0.1;
        let step = &grad_val * (-step_size);
        let step_norm = step.mapv(|s| s.abs()).sum();

        x = &x + &step;

        let convergence_result = convergence_state.update_and_check_convergence(
            f_val,
            grad_norm,
            step_norm,
            iteration,
            Some(&x.view()),
        )?;

        // Print progress when plateau is detected
        if convergence_result.plateau_detected && iteration % 5 == 0 {
            println!(
                "  Iteration {}: Plateau detected, f = {:.3e}, x = {:.3f}",
                iteration, f_val, x[0]
            );
        }

        if convergence_result.converged || iteration >= 100 {
            println!("\nResults:");
            println!("  Converged: {}", convergence_result.converged);
            println!("  Iterations: {}", iteration);
            println!("  Final position: x = {:.6f}", x[0]);
            println!("  Final function value: {:.6e}", f_val);
            println!(
                "  Plateau detected: {}",
                convergence_result.plateau_detected
            );

            if convergence_result.converged {
                println!(
                    "  Convergence reasons: {}",
                    convergence_result.convergence_reasons.join(", ")
                );
            }

            break;
        }

        iteration += 1;
    }

    println!();
    Ok(())
}

/// Example 4: Early stopping for slow convergence
fn early_stopping_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Early Stopping Example ===");

    // Poorly conditioned function (very slow convergence)
    let fun = |x: &ArrayView1<f64>| -> f64 {
        1000.0 * x[0].powi(2) + x[1].powi(2) // Ill-conditioned
    };

    let grad =
        |x: &ArrayView1<f64>| -> Array1<f64> { Array1::from_vec(vec![2000.0 * x[0], 2.0 * x[1]]) };

    let mut options = RobustConvergenceOptions::default();
    options.enable_early_stopping = true;
    options.early_stopping_patience = 20;
    options.early_stopping_min_delta = 1e-8;
    options.enable_progress_based = true;
    options.min_progress_rate = 1e-6;

    let mut convergence_state = RobustConvergenceState::new(options, 2);
    convergence_state.start_timing();

    let mut x = Array1::from_vec(vec![1.0, 1.0]);
    let mut iteration = 0;

    println!("Configuration:");
    println!(
        "  Early stopping patience: {}",
        convergence_state.options.early_stopping_patience
    );
    println!(
        "  Min improvement delta: {:.1e}",
        convergence_state.options.early_stopping_min_delta
    );
    println!(
        "  Min progress rate: {:.1e}",
        convergence_state.options.min_progress_rate
    );

    loop {
        let f_val = fun(&x.view());
        let grad_val = grad(&x.view());
        let grad_norm = grad_val.mapv(|g| g.abs()).sum();

        // Use a very small step size to simulate slow convergence
        let step_size = 1e-5;
        let step = &grad_val * (-step_size);
        let step_norm = step.mapv(|s| s.abs()).sum();

        x = &x + &step;

        let convergence_result = convergence_state.update_and_check_convergence(
            f_val,
            grad_norm,
            step_norm,
            iteration,
            Some(&x.view()),
        )?;

        // Print progress periodically
        if iteration % 10 == 0 {
            println!(
                "  Iteration {}: f = {:.3e}, progress = {:.3e}, early_stop = {}",
                iteration,
                f_val,
                convergence_result.progress_rate,
                convergence_result.early_stopping_iterations
            );
        }

        if convergence_result.converged || iteration >= 200 {
            println!("\nResults:");
            println!("  Converged: {}", convergence_result.converged);
            println!("  Iterations: {}", iteration);
            println!("  Final function value: {:.6e}", f_val);
            println!(
                "  Early stopping iterations: {}",
                convergence_result.early_stopping_iterations
            );
            println!(
                "  Final progress rate: {:.3e}",
                convergence_result.progress_rate
            );

            if convergence_result.converged {
                println!(
                    "  Convergence reasons: {}",
                    convergence_result.convergence_reasons.join(", ")
                );
            }

            break;
        }

        iteration += 1;
    }

    println!();
    Ok(())
}

/// Example 5: Multiple criteria requirement
fn multiple_criteria_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Multiple Criteria Requirement ===");

    let fun = |x: &ArrayView1<f64>| -> f64 { x.mapv(|xi| xi.powi(2)).sum() };

    let grad = |x: &ArrayView1<f64>| -> Array1<f64> { x.mapv(|xi| 2.0 * xi) };

    let mut options = RobustConvergenceOptions::default();
    options.require_multiple_criteria = true;
    options.min_criteria_count = 2;
    options.enable_progress_based = true;

    // Set strict tolerances to make it harder to satisfy multiple criteria
    options.adaptive_tolerance.initial_ftol = 1e-10;
    options.adaptive_tolerance.initial_gtol = 1e-8;
    options.adaptive_tolerance.initial_xtol = 1e-10;

    let mut convergence_state = RobustConvergenceState::new(options, 3);
    convergence_state.start_timing();

    let mut x = Array1::from_vec(vec![0.1, 0.1, 0.1]);
    let mut iteration = 0;

    println!("Configuration:");
    println!(
        "  Require multiple criteria: {}",
        convergence_state.options.require_multiple_criteria
    );
    println!(
        "  Min criteria count: {}",
        convergence_state.options.min_criteria_count
    );
    println!(
        "  Function tolerance: {:.1e}",
        convergence_state.options.adaptive_tolerance.initial_ftol
    );
    println!(
        "  Gradient tolerance: {:.1e}",
        convergence_state.options.adaptive_tolerance.initial_gtol
    );

    loop {
        let f_val = fun(&x.view());
        let grad_val = grad(&x.view());
        let grad_norm = grad_val.mapv(|g| g.abs()).sum();

        let step_size = 0.1;
        let step = &grad_val * (-step_size);
        let step_norm = step.mapv(|s| s.abs()).sum();

        x = &x + &step;

        let convergence_result = convergence_state.update_and_check_convergence(
            f_val,
            grad_norm,
            step_norm,
            iteration,
            Some(&x.view()),
        )?;

        // Print criteria status
        if iteration % 5 == 0 {
            println!(
                "  Iteration {}: f = {:.3e}, grad = {:.3e}, criteria = {}",
                iteration, f_val, grad_norm, convergence_result.criteria_met
            );
        }

        if convergence_result.converged || iteration >= 100 {
            println!("\nResults:");
            println!("  Converged: {}", convergence_result.converged);
            println!("  Iterations: {}", iteration);
            println!("  Final function value: {:.6e}", f_val);
            println!("  Final gradient norm: {:.6e}", grad_norm);
            println!("  Criteria met: {}", convergence_result.criteria_met);
            println!(
                "  Required criteria: {}",
                convergence_state.options.min_criteria_count
            );

            if convergence_result.converged {
                println!(
                    "  Convergence reasons: {}",
                    convergence_result.convergence_reasons.join(", ")
                );
            }

            break;
        }

        iteration += 1;
    }

    println!();
    Ok(())
}

/// Example 6: Custom convergence configuration
fn custom_convergence_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Custom Convergence Configuration ===");

    let fun = |x: &ArrayView1<f64>| -> f64 {
        // Rosenbrock function (moderately difficult)
        let mut sum = 0.0;
        for i in 0..x.len() - 1 {
            sum += 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2);
        }
        sum
    };

    let grad = |x: &ArrayView1<f64>| -> Array1<f64> {
        let mut g = Array1::zeros(x.len());
        for i in 0..x.len() - 1 {
            if i > 0 {
                g[i] += 200.0 * (x[i] - x[i - 1].powi(2));
            }
            g[i] += -400.0 * x[i] * (x[i + 1] - x[i].powi(2)) - 2.0 * (1.0 - x[i]);
            if i < x.len() - 1 {
                g[i + 1] += 200.0 * (x[i + 1] - x[i].powi(2));
            }
        }
        g
    };

    // Custom robust configuration for Rosenbrock
    let mut options = RobustConvergenceOptions::default();
    options.enable_early_stopping = true;
    options.early_stopping_patience = 30;
    options.enable_progress_based = true;
    options.enable_noise_robust = false; // Not needed for Rosenbrock
    options.enable_plateau_detection = true;
    options.enable_time_limit = true;
    options.max_time = Duration::from_secs(5); // 5 second limit
    options.adaptive_tolerance.initial_gtol = 1e-4; // Relaxed for difficult function

    let mut convergence_state = RobustConvergenceState::new(options, 4);
    convergence_state.start_timing();

    let mut x = Array1::from_vec(vec![-1.0, 1.0, -1.0, 1.0]); // Challenging start
    let mut iteration = 0;

    println!("Rosenbrock function with custom configuration:");
    println!(
        "  Time limit: {} seconds",
        convergence_state.options.max_time.as_secs()
    );
    println!(
        "  Early stopping patience: {}",
        convergence_state.options.early_stopping_patience
    );
    println!(
        "  Plateau detection: {}",
        convergence_state.options.enable_plateau_detection
    );

    let start = Instant::now();

    loop {
        let f_val = fun(&x.view());
        let grad_val = grad(&x.view());
        let grad_norm = grad_val.mapv(|g| g * g).sum().sqrt();

        // Adaptive step size
        let step_size = if grad_norm > 10.0 { 0.001 } else { 0.01 };
        let step = &grad_val * (-step_size);
        let step_norm = step.mapv(|s| s * s).sum().sqrt();

        x = &x + &step;

        let convergence_result = convergence_state.update_and_check_convergence(
            f_val,
            grad_norm,
            step_norm,
            iteration,
            Some(&x.view()),
        )?;

        if iteration % 10 == 0 {
            println!(
                "  Iteration {}: f = {:.3e}, |g| = {:.3e}, time = {:.1}s",
                iteration,
                f_val,
                grad_norm,
                start.elapsed().as_secs_f64()
            );
        }

        if convergence_result.converged || iteration >= 500 {
            println!("\nResults:");
            println!("  Converged: {}", convergence_result.converged);
            println!("  Iterations: {}", iteration);
            println!("  Final function value: {:.6e}", f_val);
            println!("  Final gradient norm: {:.6e}", grad_norm);

            if let Some(time_elapsed) = convergence_result.time_elapsed {
                println!("  Time elapsed: {:.3} seconds", time_elapsed.as_secs_f64());
            }

            println!(
                "  Final solution: x = [{:.4}, {:.4}, {:.4}, {:.4}]",
                x[0], x[1], x[2], x[3]
            );

            if convergence_result.converged {
                println!(
                    "  Convergence reasons: {}",
                    convergence_result.convergence_reasons.join(", ")
                );
            }

            if !convergence_result.warning_flags.is_empty() {
                println!(
                    "  Warnings: {}",
                    convergence_result.warning_flags.join(", ")
                );
            }

            // Print detailed report
            println!("\nDetailed Report:");
            print!("{}", convergence_result.get_detailed_report());

            break;
        }

        iteration += 1;
    }

    println!();
    Ok(())
}

// Global counter for simulating deterministic noise
static mut ITERATION_COUNTER: usize = 0;

fn increment_iteration_counter() {
    unsafe {
        ITERATION_COUNTER += 1;
    }
}

fn iteration_counter() -> usize {
    unsafe { ITERATION_COUNTER }
}

fn reset_iteration_counter() {
    unsafe {
        ITERATION_COUNTER = 0;
    }
}
