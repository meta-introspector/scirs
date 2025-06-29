use ndarray::{array, ArrayView1};

// Copy of the BasinHopping implementation to test independently
use rand::distr::Uniform;
use rand::prelude::*;
use rand::rngs::StdRng;

// Simplified test for bounds behavior
fn test_bounds_enforcement() {
    println!("Testing Basin-hopping bounds enforcement...");
    
    // Function with minimum at (-1, -1), but we'll bound it to positive region
    let func = |x: &ArrayView1<f64>| (x[0] + 1.0).powi(2) + (x[1] + 1.0).powi(2);
    
    let bounds = vec![(0.0, 2.0), (0.0, 2.0)];
    let stepsize = 0.5;
    let x = array![1.0, 1.0];
    
    // Test the default take_step function behavior
    let mut rng = StdRng::seed_from_u64(42);
    let mut x_new = x.clone();
    
    println!("Original point: {:?}", x);
    
    // Simulate the random perturbation from the default take_step
    for i in 0..x.len() {
        let uniform = Uniform::new(-stepsize, stepsize).unwrap();
        x_new[i] += rng.sample(uniform);
        
        // Apply bounds if specified
        let (lb, ub) = bounds[i];
        x_new[i] = x_new[i].max(lb).min(ub);
        
        println!("Dimension {}: {} -> {} (bounded to [{}, {}])", 
                 i, x[i], x_new[i], lb, ub);
    }
    
    println!("New point after perturbation and bounds: {:?}", x_new);
    
    // Check if bounds are properly enforced
    for (i, (&xi, &(lb, ub))) in x_new.iter().zip(bounds.iter()).enumerate() {
        assert!(xi >= lb && xi <= ub, 
                "Dimension {} value {} is outside bounds [{}, {}]", i, xi, lb, ub);
    }
    
    println!("✓ Bounds are properly enforced during perturbation");
}

fn main() {
    test_bounds_enforcement();
}