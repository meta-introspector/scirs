//! Simple test for stochastic optimization functionality

use ndarray::Array1;
use scirs2_optimize::stochastic::{
    minimize_sgd, InMemoryDataProvider, SGDOptions, StochasticGradientFunction,
};

// Simple quadratic function for testing
struct QuadraticFunction;

impl StochasticGradientFunction for QuadraticFunction {
    fn compute_gradient(
        &mut self,
        x: &ndarray::ArrayView1<f64>,
        _batch_data: &[f64],
    ) -> Array1<f64> {
        x.mapv(|xi| 2.0 * xi)
    }

    fn compute_value(&mut self, x: &ndarray::ArrayView1<f64>, _batch_data: &[f64]) -> f64 {
        x.mapv(|xi| xi * xi).sum()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing simple stochastic optimization");

    let grad_func = QuadraticFunction;
    let x0 = Array1::from_vec(vec![1.0, 2.0]);
    let data_provider = Box::new(InMemoryDataProvider::new(vec![1.0; 100]));

    let options = SGDOptions {
        learning_rate: 0.1,
        max_iter: 100,
        tol: 1e-6,
        ..Default::default()
    };

    let result = minimize_sgd(grad_func, x0, data_provider, options)?;

    println!("Final solution: [{:.6}, {:.6}]", result.x[0], result.x[1]);
    println!("Function value: {:.2e}", result.fun);
    println!("Success: {}", result.success);

    Ok(())
}
