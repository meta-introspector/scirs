//! Interior point method example
//!
//! This example demonstrates how to use the interior point method
//! for constrained optimization problems.

use ndarray::{array, Array1};
use scirs2_optimize::constrained::{minimize_constrained, Constraint, Method};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Interior Point Method Examples\n");

    // Example 1: Simple quadratic with inequality constraint
    simple_inequality_example()?;

    // Example 2: Quadratic with equality constraint
    equality_constraint_example()?;

    // Example 3: Multi-constraint problem
    multi_constraint_example()?;

    Ok(())
}

/// Example 1: Minimize x^2 + y^2 subject to x + y >= 1
fn simple_inequality_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Simple Inequality Constraint Example ===");

    // Objective function: minimize x^2 + y^2
    let objective = |x: &[f64]| -> f64 { x[0].powi(2) + x[1].powi(2) };

    // Inequality constraint: x + y >= 1 (reformulated as 1 - x - y <= 0)
    let inequality_constraint = |x: &[f64]| -> f64 {
        x[0] + x[1] - 1.0 // Should be >= 0 for constraint satisfaction
    };

    let initial_point = array![2.0, 2.0];
    let constraints = vec![Constraint::new(
        inequality_constraint,
        Constraint::INEQUALITY,
    )];

    println!("Objective: minimize x^2 + y^2");
    println!("Constraint: x + y >= 1");
    println!(
        "Initial point: [{:.3}, {:.3}]",
        initial_point[0], initial_point[1]
    );

    // Note: Interior point method integration is still being developed
    // For now, we'll use SLSQP as a fallback to demonstrate the constraint system
    let result = minimize_constrained(
        objective,
        &initial_point,
        &constraints,
        Method::SLSQP, // Method::InteriorPoint once fully integrated
        None,
    )?;

    println!("Results:");
    println!("  Success: {}", result.success);
    println!("  Solution: [{:.6}, {:.6}]", result.x[0], result.x[1]);
    println!("  Function value: {:.6}", result.fun);
    println!("  Iterations: {}", result.nit);

    // The optimal solution should be approximately [0.5, 0.5] with f = 0.5
    let constraint_value = inequality_constraint(&[result.x[0], result.x[1]]);
    println!(
        "  Constraint value (should be ≈ 0): {:.6}",
        constraint_value
    );
    println!();

    Ok(())
}

/// Example 2: Minimize x^2 + y^2 subject to x + y = 2
fn equality_constraint_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Equality Constraint Example ===");

    // Objective function: minimize x^2 + y^2
    let objective = |x: &[f64]| -> f64 { x[0].powi(2) + x[1].powi(2) };

    // Equality constraint: x + y = 2 (reformulated as x + y - 2 = 0)
    let equality_constraint = |x: &[f64]| -> f64 {
        x[0] + x[1] - 2.0 // Should be = 0
    };

    let initial_point = array![0.0, 0.0];
    let constraints = vec![Constraint::new(equality_constraint, Constraint::EQUALITY)];

    println!("Objective: minimize x^2 + y^2");
    println!("Constraint: x + y = 2");
    println!(
        "Initial point: [{:.3}, {:.3}]",
        initial_point[0], initial_point[1]
    );

    let result = minimize_constrained(
        objective,
        &initial_point,
        &constraints,
        Method::SLSQP, // Method::InteriorPoint once fully integrated
        None,
    )?;

    println!("Results:");
    println!("  Success: {}", result.success);
    println!("  Solution: [{:.6}, {:.6}]", result.x[0], result.x[1]);
    println!("  Function value: {:.6}", result.fun);
    println!("  Iterations: {}", result.nit);

    // The optimal solution should be approximately [1.0, 1.0] with f = 2.0
    let constraint_value = equality_constraint(&[result.x[0], result.x[1]]);
    println!(
        "  Constraint value (should be ≈ 0): {:.6}",
        constraint_value
    );
    println!();

    Ok(())
}

/// Example 3: Multi-constraint problem
fn multi_constraint_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Multi-Constraint Example ===");

    // Objective function: minimize (x-1)^2 + (y-2)^2
    let objective = |x: &[f64]| -> f64 { (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2) };

    // Constraint 1: x + y >= 2
    let constraint1 = |x: &[f64]| -> f64 {
        x[0] + x[1] - 2.0 // Should be >= 0
    };

    // Constraint 2: x >= 0
    let constraint2 = |x: &[f64]| -> f64 {
        x[0] // Should be >= 0
    };

    // Constraint 3: y >= 0
    let constraint3 = |x: &[f64]| -> f64 {
        x[1] // Should be >= 0
    };

    let initial_point = array![0.5, 1.5];
    let constraints = vec![
        Constraint::new(constraint1, Constraint::INEQUALITY),
        Constraint::new(constraint2, Constraint::INEQUALITY),
        Constraint::new(constraint3, Constraint::INEQUALITY),
    ];

    println!("Objective: minimize (x-1)^2 + (y-2)^2");
    println!("Constraints:");
    println!("  x + y >= 2");
    println!("  x >= 0");
    println!("  y >= 0");
    println!(
        "Initial point: [{:.3}, {:.3}]",
        initial_point[0], initial_point[1]
    );

    let result = minimize_constrained(
        objective,
        &initial_point,
        &constraints,
        Method::SLSQP, // Method::InteriorPoint once fully integrated
        None,
    )?;

    println!("Results:");
    println!("  Success: {}", result.success);
    println!("  Solution: [{:.6}, {:.6}]", result.x[0], result.x[1]);
    println!("  Function value: {:.6}", result.fun);
    println!("  Iterations: {}", result.nit);

    // Check constraint satisfaction
    let c1_value = constraint1(&[result.x[0], result.x[1]]);
    let c2_value = constraint2(&[result.x[0], result.x[1]]);
    let c3_value = constraint3(&[result.x[0], result.x[1]]);

    println!("  Constraint 1 value (should be ≥ 0): {:.6}", c1_value);
    println!("  Constraint 2 value (should be ≥ 0): {:.6}", c2_value);
    println!("  Constraint 3 value (should be ≥ 0): {:.6}", c3_value);
    println!();

    Ok(())
}

/// Example demonstrating potential Interior Point method usage
/// (Currently commented out due to integration work in progress)
#[allow(dead_code)]
fn interior_point_direct_example() -> Result<(), Box<dyn std::error::Error>> {
    use scirs2_optimize::constrained::{minimize_interior_point_constrained, InteriorPointOptions};

    println!("=== Direct Interior Point Method Example ===");

    // Objective function: minimize x^2 + y^2
    let objective = |x: &[f64]| -> f64 { x[0].powi(2) + x[1].powi(2) };

    // Inequality constraint: x + y >= 1
    let constraint = |x: &[f64]| -> f64 { x[0] + x[1] - 1.0 };

    let initial_point = Array1::from_vec(vec![2.0, 2.0]);
    let constraints = vec![Constraint::new(constraint, Constraint::INEQUALITY)];

    let options = InteriorPointOptions {
        max_iter: 100,
        tol: 1e-6,
        feas_tol: 1e-6,
        ..Default::default()
    };

    println!("Using Interior Point method directly...");

    let result =
        minimize_interior_point_constrained(objective, initial_point, &constraints, Some(options))?;

    println!("Interior Point Results:");
    println!("  Success: {}", result.success);
    println!("  Solution: [{:.6}, {:.6}]", result.x[0], result.x[1]);
    println!("  Function value: {:.6}", result.fun);
    println!("  Iterations: {}", result.nit);
    println!("  Function evaluations: {}", result.nfev);

    Ok(())
}
