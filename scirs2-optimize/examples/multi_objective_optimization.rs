//! Multi-objective optimization examples
//!
//! This example demonstrates how to use NSGA-II and NSGA-III algorithms
//! for solving multi-objective optimization problems.

use ndarray::{array, s, Array1, ArrayView1};
use scirs2_optimize::multi_objective::{scalarization, MultiObjectiveConfig, NSGAII, NSGAIII};
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Multi-objective Optimization Examples\n");

    // Example 1: Bi-objective optimization with NSGA-II
    bi_objective_nsga2_demo()?;

    // Example 2: Many-objective optimization with NSGA-III
    many_objective_nsga3_demo()?;

    // Example 3: Scalarization methods
    scalarization_demo()?;

    // Example 4: Constrained multi-objective optimization
    constrained_multi_objective_demo()?;

    // Example 5: Engineering design problem
    engineering_design_demo()?;

    Ok(())
}

/// Demonstrate NSGA-II on a bi-objective problem
fn bi_objective_nsga2_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== NSGA-II Bi-objective Optimization ===");

    // ZDT1 problem: minimize (f1(x), f2(x))
    // f1(x) = x1
    // f2(x) = g(x) * h(f1, g)
    // g(x) = 1 + 9 * sum(x_i for i=2..n) / (n-1)
    // h(f1, g) = 1 - sqrt(f1/g)
    let zdt1_objectives = |x: &ArrayView1<f64>| {
        let f1 = x[0];
        let g = 1.0 + 9.0 * x.slice(s![1..]).sum() / (x.len() - 1) as f64;
        let h = 1.0 - (f1 / g).sqrt();
        let f2 = g * h;
        array![f1, f2]
    };

    let config = MultiObjectiveConfig {
        population_size: 100,
        max_generations: 100,
        crossover_probability: 0.9,
        mutation_probability: 0.1,
        ..Default::default()
    };

    let mut optimizer =
        NSGAII::new(30, 2, Some(config)).with_bounds(Array1::zeros(30), Array1::ones(30))?;

    let result = optimizer.optimize(zdt1_objectives, None::<fn(&ArrayView1<f64>) -> f64>)?;

    println!("NSGA-II Results:");
    println!("  Success: {}", result.success);
    println!("  Generations: {}", result.n_generations);
    println!("  Function evaluations: {}", result.n_evaluations);
    println!("  Pareto front size: {}", result.pareto_front.len());

    // Display some solutions from the Pareto front
    println!("  Sample Pareto solutions (f1, f2):");
    for (i, solution) in result.pareto_front.iter().take(5).enumerate() {
        println!(
            "    Solution {}: ({:.4}, {:.4})",
            i + 1,
            solution.objectives[0],
            solution.objectives[1]
        );
    }

    // Check Pareto front properties
    let mut f1_values: Vec<f64> = result
        .pareto_front
        .iter()
        .map(|sol| sol.objectives[0])
        .collect();
    f1_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

    println!(
        "  f1 range: [{:.4}, {:.4}]",
        f1_values[0],
        f1_values[f1_values.len() - 1]
    );
    println!();

    Ok(())
}

/// Demonstrate NSGA-III on a many-objective problem
fn many_objective_nsga3_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== NSGA-III Many-objective Optimization ===");

    // DTLZ2 problem with 5 objectives
    let n_objectives = 5;
    let n_variables = n_objectives + 9; // 14 variables total

    let dtlz2_objectives = move |x: &ArrayView1<f64>| {
        let n = x.len();
        let m = n_objectives;

        // Calculate g(x)
        let g = x
            .slice(s![m - 1..])
            .iter()
            .map(|&xi| (xi - 0.5).powi(2))
            .sum::<f64>();

        let mut objectives = Array1::zeros(m);

        for i in 0..m {
            let mut prod = 1.0 + g;

            // Product of cosines
            for j in 0..m - 1 - i {
                prod *= (x[j] * PI / 2.0).cos();
            }

            // Final sine term (if not the last objective)
            if i < m - 1 {
                prod *= (x[m - 1 - i] * PI / 2.0).sin();
            }

            objectives[i] = prod;
        }

        objectives
    };

    let config = MultiObjectiveConfig {
        population_size: 150, // Larger population for many objectives
        max_generations: 200,
        crossover_probability: 0.9,
        mutation_probability: 0.1,
        reference_point: Some(Array1::from_elem(n_objectives, 2.0)), // For hypervolume
        ..Default::default()
    };

    let mut optimizer = NSGAIII::new(n_variables, n_objectives, Some(config))
        .with_bounds(Array1::zeros(n_variables), Array1::ones(n_variables))?;

    let result = optimizer.optimize(dtlz2_objectives, None::<fn(&ArrayView1<f64>) -> f64>)?;

    println!("NSGA-III Results:");
    println!("  Success: {}", result.success);
    println!("  Generations: {}", result.n_generations);
    println!("  Function evaluations: {}", result.n_evaluations);
    println!("  Pareto front size: {}", result.pareto_front.len());

    if let Some(hv) = result.hypervolume {
        println!("  Hypervolume: {:.6}", hv);
    }

    // Display some solutions from the Pareto front
    println!("  Sample Pareto solutions:");
    for (i, solution) in result.pareto_front.iter().take(5).enumerate() {
        print!("    Solution {}: [", i + 1);
        for (j, &obj) in solution.objectives.iter().enumerate() {
            if j > 0 {
                print!(", ");
            }
            print!("{:.4}", obj);
        }
        println!("]");
    }

    // Check convergence to known Pareto front
    let distances_to_unit_sphere: Vec<f64> = result
        .pareto_front
        .iter()
        .map(|sol| {
            let norm_sq = sol.objectives.iter().map(|&x| x.powi(2)).sum::<f64>();
            (norm_sq.sqrt() - 1.0).abs() // DTLZ2 Pareto front lies on unit sphere
        })
        .collect();

    let avg_distance =
        distances_to_unit_sphere.iter().sum::<f64>() / distances_to_unit_sphere.len() as f64;
    println!("  Average distance to unit sphere: {:.6}", avg_distance);
    println!();

    Ok(())
}

/// Demonstrate scalarization methods
fn scalarization_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Scalarization Methods Demo ===");

    // Simple bi-objective problem: minimize (x^2, (x-1)^2)
    let bi_objective_fn = |x: &ArrayView1<f64>| array![x[0].powi(2), (x[0] - 1.0).powi(2)];

    let x_test = array![0.3];
    let objectives = bi_objective_fn(&x_test.view());

    println!(
        "Test point x = {:.2}, objectives = [{:.4}, {:.4}]",
        x_test[0], objectives[0], objectives[1]
    );

    // Weighted sum method
    let weights = array![0.5, 0.5];
    let weighted_result = scalarization::weighted_sum(bi_objective_fn, &weights, &x_test.view());
    println!("Weighted sum (w=[0.5, 0.5]): {:.4}", weighted_result);

    // Weighted Tchebycheff method
    let ideal_point = array![0.0, 0.0];
    let tcheby_result = scalarization::weighted_tchebycheff(
        bi_objective_fn,
        &weights,
        &ideal_point,
        &x_test.view(),
    );
    println!("Weighted Tchebycheff: {:.4}", tcheby_result);

    // Achievement Scalarizing Function
    let reference_point = array![0.2, 0.2];
    let asf_result = scalarization::achievement_scalarizing(
        bi_objective_fn,
        &weights,
        &reference_point,
        &x_test.view(),
    );
    println!("Achievement Scalarizing Function: {:.4}", asf_result);

    // ε-constraint method
    let epsilon_constraint = scalarization::EpsilonConstraint::new(0, array![0.5]); // Minimize f1, f2 ≤ 0.5
    let constraint_fn = epsilon_constraint.scalarize(bi_objective_fn, 1000.0);
    let epsilon_result = constraint_fn(&x_test.view());
    println!(
        "ε-constraint (minimize f1, f2 ≤ 0.5): {:.4}",
        epsilon_result
    );

    println!();
    Ok(())
}

/// Demonstrate constrained multi-objective optimization
fn constrained_multi_objective_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Constrained Multi-objective Optimization ===");

    // Constrained test problem: minimize (f1, f2) subject to g(x) ≤ 0
    let constrained_objectives = |x: &ArrayView1<f64>| {
        let f1 = x[0];
        let f2 = (1.0 + x[1]) / x[0];
        array![f1, f2]
    };

    // Constraint: x[0] + x[1] - 1 ≤ 0 (i.e., x[0] + x[1] ≤ 1)
    let constraint_fn = |x: &ArrayView1<f64>| x[0] + x[1] - 1.0;

    let config = MultiObjectiveConfig {
        population_size: 100,
        max_generations: 150,
        crossover_probability: 0.9,
        mutation_probability: 0.15,
        ..Default::default()
    };

    let mut optimizer = NSGAII::new(2, 2, Some(config)).with_bounds(
        array![0.1, 0.0], // x[0] > 0 to avoid division by zero
        array![1.0, 1.0],
    )?;

    let result = optimizer.optimize(constrained_objectives, Some(constraint_fn))?;

    println!("Constrained optimization results:");
    println!("  Success: {}", result.success);
    println!("  Pareto front size: {}", result.pareto_front.len());

    // Check constraint satisfaction
    let feasible_solutions = result
        .pareto_front
        .iter()
        .filter(|sol| sol.constraint_violation == 0.0)
        .count();

    println!(
        "  Feasible solutions: {}/{}",
        feasible_solutions,
        result.pareto_front.len()
    );

    // Display feasible solutions
    println!("  Sample feasible Pareto solutions:");
    for (i, solution) in result
        .pareto_front
        .iter()
        .filter(|sol| sol.constraint_violation == 0.0)
        .take(5)
        .enumerate()
    {
        println!(
            "    Solution {}: x=[{:.4}, {:.4}], f=[{:.4}, {:.4}], constraint={:.6}",
            i + 1,
            solution.variables[0],
            solution.variables[1],
            solution.objectives[0],
            solution.objectives[1],
            solution.constraint_violation
        );
    }

    println!();
    Ok(())
}

/// Engineering design optimization example
fn engineering_design_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Engineering Design Optimization ===");

    // Welded beam design problem (simplified)
    // Minimize: (Volume, Deflection)
    // Variables: [h, l, t, b] (height, length, thickness, width)
    let welded_beam_objectives = |x: &ArrayView1<f64>| {
        let h = x[0]; // beam height
        let l = x[1]; // beam length
        let t = x[2]; // thickness
        let b = x[3]; // width

        // Constants
        let p = 6000.0; // applied load (N)
        let e = 2.0e5; // elastic modulus (MPa)

        // Volume (to minimize)
        let volume = h * l * t * b;

        // Deflection (to minimize)
        let i = t * h.powi(3) / 12.0; // moment of inertia
        let deflection = (p * l.powi(3)) / (3.0 * e * i);

        array![volume, deflection]
    };

    // Constraint: maximum stress ≤ allowable stress
    let stress_constraint = |x: &ArrayView1<f64>| {
        let h = x[0];
        let t = x[2];
        let b = x[3];

        let p = 6000.0;
        let sigma_allow = 250.0; // allowable stress (MPa)

        // Maximum bending stress
        let i = t * h.powi(3) / 12.0;
        let sigma_max = (p * h) / (2.0 * i);

        sigma_max - sigma_allow // ≤ 0 for feasibility
    };

    let config = MultiObjectiveConfig {
        population_size: 80,
        max_generations: 100,
        crossover_probability: 0.8,
        mutation_probability: 0.2,
        ..Default::default()
    };

    let mut optimizer = NSGAII::new(4, 2, Some(config)).with_bounds(
        array![10.0, 10.0, 0.1, 0.1],   // minimum sizes
        array![80.0, 80.0, 10.0, 10.0], // maximum sizes
    )?;

    let result = optimizer.optimize(welded_beam_objectives, Some(stress_constraint))?;

    println!("Welded beam design results:");
    println!("  Success: {}", result.success);
    println!("  Generations: {}", result.n_generations);
    println!("  Function evaluations: {}", result.n_evaluations);

    // Find best solutions for each objective
    let mut best_volume_solution = &result.pareto_front[0];
    let mut best_deflection_solution = &result.pareto_front[0];

    for solution in &result.pareto_front {
        if solution.constraint_violation == 0.0 {
            if solution.objectives[0] < best_volume_solution.objectives[0] {
                best_volume_solution = solution;
            }
            if solution.objectives[1] < best_deflection_solution.objectives[1] {
                best_deflection_solution = solution;
            }
        }
    }

    println!("  Best volume design:");
    println!(
        "    Variables: h={:.2}, l={:.2}, t={:.2}, b={:.2}",
        best_volume_solution.variables[0],
        best_volume_solution.variables[1],
        best_volume_solution.variables[2],
        best_volume_solution.variables[3]
    );
    println!(
        "    Volume: {:.2}, Deflection: {:.6}",
        best_volume_solution.objectives[0], best_volume_solution.objectives[1]
    );

    println!("  Best deflection design:");
    println!(
        "    Variables: h={:.2}, l={:.2}, t={:.2}, b={:.2}",
        best_deflection_solution.variables[0],
        best_deflection_solution.variables[1],
        best_deflection_solution.variables[2],
        best_deflection_solution.variables[3]
    );
    println!(
        "    Volume: {:.2}, Deflection: {:.6}",
        best_deflection_solution.objectives[0], best_deflection_solution.objectives[1]
    );

    println!();
    Ok(())
}

/// Helper function to create test problems
#[allow(dead_code)]
fn create_zdt_problem(problem_num: usize) -> impl Fn(&ArrayView1<f64>) -> Array1<f64> {
    move |x: &ArrayView1<f64>| {
        match problem_num {
            1 => {
                // ZDT1
                let f1 = x[0];
                let g = 1.0 + 9.0 * x.slice(s![1..]).sum() / (x.len() - 1) as f64;
                let h = 1.0 - (f1 / g).sqrt();
                let f2 = g * h;
                array![f1, f2]
            }
            2 => {
                // ZDT2
                let f1 = x[0];
                let g = 1.0 + 9.0 * x.slice(s![1..]).sum() / (x.len() - 1) as f64;
                let h = 1.0 - (f1 / g).powi(2);
                let f2 = g * h;
                array![f1, f2]
            }
            3 => {
                // ZDT3
                let f1 = x[0];
                let g = 1.0 + 9.0 * x.slice(s![1..]).sum() / (x.len() - 1) as f64;
                let h = 1.0 - (f1 / g).sqrt() - (f1 / g) * (10.0 * PI * f1).sin();
                let f2 = g * h;
                array![f1, f2]
            }
            _ => {
                // Default to ZDT1
                let f1 = x[0];
                let g = 1.0 + 9.0 * x.slice(s![1..]).sum() / (x.len() - 1) as f64;
                let h = 1.0 - (f1 / g).sqrt();
                let f2 = g * h;
                array![f1, f2]
            }
        }
    }
}

/// Helper function to create DTLZ test problems
#[allow(dead_code)]
fn create_dtlz_problem(
    problem_num: usize,
    n_objectives: usize,
) -> impl Fn(&ArrayView1<f64>) -> Array1<f64> {
    move |x: &ArrayView1<f64>| {
        let m = n_objectives;

        match problem_num {
            1 => {
                // DTLZ1
                let k = x.len() - m + 1;
                let g = 100.0
                    * (k as f64
                        + x.slice(s![m - 1..])
                            .iter()
                            .map(|&xi| (xi - 0.5).powi(2) - (20.0 * PI * (xi - 0.5)).cos())
                            .sum::<f64>());

                let mut objectives = Array1::zeros(m);
                for i in 0..m {
                    let mut prod = 0.5 * (1.0 + g);
                    for j in 0..m - 1 - i {
                        prod *= x[j];
                    }
                    if i > 0 {
                        prod *= 1.0 - x[m - 1 - i];
                    }
                    objectives[i] = prod;
                }
                objectives
            }
            2 => {
                // DTLZ2 (already implemented above)
                let g = x
                    .slice(s![m - 1..])
                    .iter()
                    .map(|&xi| (xi - 0.5).powi(2))
                    .sum::<f64>();

                let mut objectives = Array1::zeros(m);
                for i in 0..m {
                    let mut prod = 1.0 + g;
                    for j in 0..m - 1 - i {
                        prod *= (x[j] * PI / 2.0).cos();
                    }
                    if i < m - 1 {
                        prod *= (x[m - 1 - i] * PI / 2.0).sin();
                    }
                    objectives[i] = prod;
                }
                objectives
            }
            _ => {
                // Default to DTLZ2
                let g = x
                    .slice(s![m - 1..])
                    .iter()
                    .map(|&xi| (xi - 0.5).powi(2))
                    .sum::<f64>();

                let mut objectives = Array1::zeros(m);
                for i in 0..m {
                    let mut prod = 1.0 + g;
                    for j in 0..m - 1 - i {
                        prod *= (x[j] * PI / 2.0).cos();
                    }
                    if i < m - 1 {
                        prod *= (x[m - 1 - i] * PI / 2.0).sin();
                    }
                    objectives[i] = prod;
                }
                objectives
            }
        }
    }
}
