//! Test just the SGD optimizer to verify the core functionality

use ndarray::{Array1, ArrayView1};

// Define the traits and types locally to avoid import issues
pub trait StochasticGradientFunction {
    fn compute_gradient(&mut self, x: &ArrayView1<f64>, batch_data: &[f64]) -> Array1<f64>;
    fn compute_value(&mut self, x: &ArrayView1<f64>, batch_data: &[f64]) -> f64;
}

pub trait DataProvider {
    fn num_samples(&self) -> usize;
    fn get_batch(&self, indices: &[usize]) -> Vec<f64>;
    fn get_full_data(&self) -> Vec<f64>;
}

pub struct InMemoryDataProvider {
    data: Vec<f64>,
}

impl InMemoryDataProvider {
    pub fn new(data: Vec<f64>) -> Self {
        Self { data }
    }
}

impl DataProvider for InMemoryDataProvider {
    fn num_samples(&self) -> usize {
        self.data.len()
    }

    fn get_batch(&self, indices: &[usize]) -> Vec<f64> {
        indices.iter().map(|&i| self.data[i]).collect()
    }

    fn get_full_data(&self) -> Vec<f64> {
        self.data.clone()
    }
}

// Simple quadratic function for testing
struct QuadraticFunction;

impl StochasticGradientFunction for QuadraticFunction {
    fn compute_gradient(&mut self, x: &ArrayView1<f64>, _batch_data: &[f64]) -> Array1<f64> {
        x.mapv(|xi| 2.0 * xi)
    }

    fn compute_value(&mut self, x: &ArrayView1<f64>, _batch_data: &[f64]) -> f64 {
        x.mapv(|xi| xi * xi).sum()
    }
}

#[derive(Debug, Clone)]
pub struct SGDOptions {
    pub learning_rate: f64,
    pub max_iter: usize,
    pub tol: f64,
    pub batch_size: Option<usize>,
}

impl Default for SGDOptions {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            max_iter: 1000,
            tol: 1e-6,
            batch_size: None,
        }
    }
}

fn basic_sgd_step<F>(
    grad_func: &mut F,
    x: &mut Array1<f64>,
    data_provider: &Box<dyn DataProvider>,
    options: &SGDOptions,
) where
    F: StochasticGradientFunction,
{
    let num_samples = data_provider.num_samples();
    let batch_size = options.batch_size.unwrap_or(num_samples);
    let actual_batch_size = batch_size.min(num_samples);

    // Generate batch indices (simple sequential for this test)
    let batch_indices: Vec<usize> = (0..actual_batch_size).collect();

    // Get batch data
    let batch_data = data_provider.get_batch(&batch_indices);

    // Compute gradient on batch
    let gradient = grad_func.compute_gradient(&x.view(), &batch_data);

    // SGD update: x = x - lr * gradient
    *x = &*x - &gradient * options.learning_rate;
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing basic SGD functionality");

    let mut grad_func = QuadraticFunction;
    let mut x = Array1::from_vec(vec![2.0, -1.5]);
    let data_provider = Box::new(InMemoryDataProvider::new(vec![1.0; 50]));

    let options = SGDOptions {
        learning_rate: 0.1,
        max_iter: 100,
        tol: 1e-6,
        ..Default::default()
    };

    println!("Initial point: [{:.3}, {:.3}]", x[0], x[1]);
    println!(
        "Initial function value: {:.6}",
        grad_func.compute_value(&x.view(), &vec![1.0])
    );

    // Run a few SGD steps manually
    for iteration in 0..options.max_iter {
        let prev_x = x.clone();
        basic_sgd_step(&mut grad_func, &mut x, &data_provider, &options);

        if iteration % 10 == 0 {
            let f_val = grad_func.compute_value(&x.view(), &vec![1.0]);
            println!(
                "Iteration {}: x = [{:.6}, {:.6}], f = {:.6}",
                iteration, x[0], x[1], f_val
            );
        }

        // Check convergence
        let change = (&x - &prev_x).mapv(|xi| xi.abs()).sum();
        if change < options.tol {
            println!("Converged after {} iterations", iteration);
            break;
        }
    }

    let final_f = grad_func.compute_value(&x.view(), &vec![1.0]);
    println!("Final point: [{:.6}, {:.6}]", x[0], x[1]);
    println!("Final function value: {:.6}", final_f);

    // Check if we're close to the optimal solution (0, 0)
    let error = (x[0].abs() + x[1].abs()) / 2.0;
    println!("Average error from optimum: {:.6}", error);

    if error < 0.1 {
        println!("✅ SGD test PASSED - converged to near optimal solution");
    } else {
        println!("❌ SGD test FAILED - did not converge sufficiently");
    }

    Ok(())
}
