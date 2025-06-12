//! Debug script for Radau mass matrix issue

use ndarray::{array, Array2};
use scirs2_integrate::ode::{solve_ivp, MassMatrix, ODEMethod, ODEOptions};

fn main() -> scirs2_integrate::error::IntegrateResult<()> {
    // Simple test case: 2D oscillator with mass matrix
    // M·[x', v']^T = [v, -x]^T where M = [2 0; 0 1]
    
    let f = |_t: f64, y: ndarray::ArrayView1<f64>| array![y[1], -y[0]];
    let y0 = array![1.0, 0.0];
    let t_span = [0.0, 0.1]; // Very short time span
    
    println!("Testing Radau without mass matrix (should work):");
    let opts_no_mass = ODEOptions {
        method: ODEMethod::Radau,
        rtol: 1e-6,
        atol: 1e-8,
        h0: Some(0.01),
        ..Default::default()
    };
    
    match solve_ivp(f, t_span, y0.clone(), Some(opts_no_mass)) {
        Ok(result) => {
            println!("  Success! Final state: {:?}", result.y.last().unwrap());
            println!("  Steps: {}, Function evals: {}", result.n_steps, result.n_eval);
        }
        Err(e) => println!("  Failed: {:?}", e),
    }
    
    println!("\nTesting Radau with identity mass matrix (should work):");
    let identity_mass = MassMatrix::identity();
    let opts_identity = ODEOptions {
        method: ODEMethod::Radau,
        rtol: 1e-6,
        atol: 1e-8,
        h0: Some(0.01),
        mass_matrix: Some(identity_mass),
        ..Default::default()
    };
    
    match solve_ivp(f, t_span, y0.clone(), Some(opts_identity)) {
        Ok(result) => {
            println!("  Success! Final state: {:?}", result.y.last().unwrap());
            println!("  Steps: {}, Function evals: {}", result.n_steps, result.n_eval);
        }
        Err(e) => println!("  Failed: {:?}", e),
    }
    
    println!("\nTesting Radau with non-identity mass matrix (currently fails):");
    let mut mass_matrix = Array2::<f64>::eye(2);
    mass_matrix[[0, 0]] = 2.0;
    let mass = MassMatrix::constant(mass_matrix);
    
    let opts_mass = ODEOptions {
        method: ODEMethod::Radau,
        rtol: 1e-6,
        atol: 1e-8,
        h0: Some(0.01),
        mass_matrix: Some(mass),
        ..Default::default()
    };
    
    match solve_ivp(f, t_span, y0.clone(), Some(opts_mass)) {
        Ok(result) => {
            println!("  Success! Final state: {:?}", result.y.last().unwrap());
            println!("  Steps: {}, Function evals: {}", result.n_steps, result.n_eval);
        }
        Err(e) => println!("  Failed: {:?}", e),
    }
    
    Ok(())
}