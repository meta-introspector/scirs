//! Integration tests for scirs2-optimize
//!
//! These tests validate the key functionality of the optimization library
//! across different algorithm categories.

use scirs2_optimize::{
    stochastic::{
        minimize_adam, minimize_sgd, AdamOptions, InMemoryDataProvider, SGDOptions, 
        StochasticGradientFunction,
    },
    unconstrained::{minimize_bfgs, BfgsOptions},
    OptimizeResult,
};
use ndarray::{Array1, ArrayView1};

/// Simple quadratic function for testing
struct QuadraticFunction;

impl StochasticGradientFunction for QuadraticFunction {
    fn compute_gradient(&mut self, x: &ArrayView1<f64>, _batch_data: &[f64]) -> Array1<f64> {
        x.mapv(|xi| 2.0 * xi)
    }

    fn compute_value(&mut self, x: &ArrayView1<f64>, _batch_data: &[f64]) -> f64 {
        x.mapv(|xi| xi * xi).sum()
    }
}

#[test]
fn test_stochastic_optimization_integration() {
    // Test SGD
    let grad_func = QuadraticFunction;
    let x0 = Array1::from_vec(vec![1.0, -1.0]);
    let data_provider = Box::new(InMemoryDataProvider::new(vec![1.0; 50]));
    
    let options = SGDOptions {
        learning_rate: 0.1,
        max_iter: 100,
        tol: 1e-4,
        ..Default::default()
    };

    let result = minimize_sgd(grad_func, x0.clone(), data_provider, options);
    assert!(result.is_ok());
    let result = result.unwrap();
    
    // Should converge toward zero
    assert!(result.fun < 1e-2);
    println!("SGD converged to f = {:.2e} in {} iterations", result.fun, result.iterations);
}

#[test]
fn test_adam_optimization_integration() {
    // Test Adam optimizer
    let grad_func = QuadraticFunction;
    let x0 = Array1::from_vec(vec![2.0, -1.5]);
    let data_provider = Box::new(InMemoryDataProvider::new(vec![1.0; 100]));
    
    let options = AdamOptions {
        learning_rate: 0.1,
        max_iter: 200,
        tol: 1e-6,
        ..Default::default()
    };

    let result = minimize_adam(grad_func, x0, data_provider, options);
    assert!(result.is_ok());
    let result = result.unwrap();
    
    // Adam should converge efficiently
    assert!(result.fun < 1e-3);
    println!("Adam converged to f = {:.2e} in {} iterations", result.fun, result.iterations);
}

#[test]
fn test_bfgs_optimization_integration() {
    // Test BFGS on a simple function
    let mut func = |x: &ArrayView1<f64>| -> f64 {
        x[0] * x[0] + x[1] * x[1]
    };
    
    let mut grad = |x: &ArrayView1<f64>| -> Array1<f64> {
        Array1::from_vec(vec![2.0 * x[0], 2.0 * x[1]])
    };
    
    let x0 = Array1::from_vec(vec![3.0, -2.0]);
    let options = BfgsOptions::default();
    
    let result = minimize_bfgs(&mut func, &mut grad, x0, options);
    assert!(result.is_ok());
    let result = result.unwrap();
    
    // BFGS should converge quickly for quadratic functions
    assert!(result.success);
    assert!(result.fun < 1e-8);
    println!("BFGS converged to f = {:.2e} in {} iterations", result.fun, result.iterations);
}

#[test]
fn test_optimization_library_capabilities() {
    println!("\n🔬 scirs2-optimize Library Capabilities Test");
    println!("============================================");
    
    // Test that we can create different optimizers
    let _sgd_options = SGDOptions::default();
    let _adam_options = AdamOptions::default();
    let _bfgs_options = BfgsOptions::default();
    
    println!("✅ All optimizer option structs created successfully");
    
    // Test data provider
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let provider = InMemoryDataProvider::new(data.clone());
    assert_eq!(provider.num_samples(), 5);
    assert_eq!(provider.get_full_data(), data);
    
    println!("✅ Data provider functionality verified");
    
    // Test gradient function trait
    let mut grad_func = QuadraticFunction;
    let x = Array1::from_vec(vec![1.0, 2.0]);
    let batch_data = vec![1.0];
    
    let gradient = grad_func.compute_gradient(&x.view(), &batch_data);
    let expected = Array1::from_vec(vec![2.0, 4.0]);
    assert_eq!(gradient, expected);
    
    let value = grad_func.compute_value(&x.view(), &batch_data);
    assert_eq!(value, 5.0); // 1^2 + 2^2 = 5
    
    println!("✅ Stochastic gradient function trait verified");
    println!("✅ All core library capabilities working correctly!");
}