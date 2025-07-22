//! Optimization algorithms for registration

use crate::error::Result;
use ndarray::Array1;

/// Optimization result
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Optimized parameter values
    pub parameters: Array1<f64>,
    /// Final cost function value
    pub final_cost: f64,
    /// Number of iterations performed
    pub iterations: usize,
    /// Whether optimization converged
    pub converged: bool,
}

/// Gradient descent optimization
#[allow(dead_code)]
pub fn gradient_descent_optimize(
    initial_params: &Array1<f64>,
    cost_function: &dyn Fn(&Array1<f64>) -> Result<f64>,
    gradient_function: &dyn Fn(&Array1<f64>) -> Result<Array1<f64>>,
    learning_rate: f64,
    max_iterations: usize,
    tolerance: f64,
) -> Result<OptimizationResult> {
    use ndarray::Zip;

    let mut _params = initial_params.clone();
    let mut prev_cost = cost_function(&_params)?;
    let mut converged = false;
    let mut _iterations = 0;

    for i in 0..max_iterations {
        _iterations = i + 1;

        // Compute gradient
        let gradient = gradient_function(&_params)?;

        // Update parameters: _params = _params - learning_rate * gradient
        Zip::from(&mut _params)
            .and(&gradient)
            .for_each(|p, &g| *p -= learning_rate * g);

        // Compute new cost
        let current_cost = cost_function(&_params)?;

        // Check convergence
        if (prev_cost - current_cost).abs() < tolerance {
            converged = true;
            break;
        }

        // Check if cost is increasing (diverging)
        if current_cost > prev_cost {
            // Optionally implement adaptive learning _rate here
            // For now, just continue with fixed learning _rate
        }

        prev_cost = current_cost;
    }

    Ok(OptimizationResult {
        parameters: _params,
        final_cost: prev_cost,
        _iterations,
        converged,
    })
}

/// Powell's method optimization
#[allow(dead_code)]
pub fn powell_optimize(
    initial_params: &Array1<f64>,
    cost_function: &dyn Fn(&Array1<f64>) -> Result<f64>,
    max_iterations: usize,
    tolerance: f64,
) -> Result<OptimizationResult> {
    let n = initial_params.len();
    let mut _params = initial_params.clone();
    let mut directions = ndarray::Array2::eye(n);
    let mut converged = false;
    let mut _iterations = 0;
    let mut prev_cost = cost_function(&_params)?;

    for iter in 0..max_iterations {
        _iterations = iter + 1;
        let start_params = _params.clone();
        let mut biggest_decrease = 0.0;
        let mut biggest_decrease_idx = 0;

        // Line search along each direction
        for i in 0..n {
            let old_cost = cost_function(&_params)?;
            let direction = directions.row(i).to_owned();

            // Perform line search along this direction
            let (new_params, new_cost) = line_search(&_params, &direction, cost_function, 1e-6)?;
            _params = new_params;

            let decrease = old_cost - new_cost;
            if decrease > biggest_decrease {
                biggest_decrease = decrease;
                biggest_decrease_idx = i;
            }
        }

        // Check convergence
        let current_cost = cost_function(&_params)?;
        if (prev_cost - current_cost).abs() < tolerance {
            converged = true;
            break;
        }

        // Update search directions
        if iter > 0 && iter % n == 0 {
            // Calculate new direction
            let new_direction = &_params - &start_params;
            let new_dir_norm = new_direction.dot(&new_direction).sqrt();

            if new_dir_norm > 1e-10 {
                // Replace the direction that gave the biggest decrease
                let normalized_dir = &new_direction / new_dir_norm;
                directions
                    .row_mut(biggest_decrease_idx)
                    .assign(&normalized_dir);

                // Perform line search along the new direction
                let (new_params_) = line_search(&_params, &normalized_dir, cost_function, 1e-6)?;
                _params = new_params;
            }
        }

        prev_cost = current_cost;
    }

    Ok(OptimizationResult {
        parameters: _params,
        final_cost: prev_cost,
        _iterations,
        converged,
    })
}

/// Perform line search along a direction
#[allow(dead_code)]
fn line_search(
    start_point: &Array1<f64>,
    direction: &Array1<f64>,
    cost_function: &dyn Fn(&Array1<f64>) -> Result<f64>,
    tolerance: f64,
) -> Result<(Array1<f64>, f64)> {
    // Golden section search
    const GOLDEN_RATIO: f64 = 0.618033988749895;

    // Bracket the minimum
    let mut a = 0.0;
    let mut b = 1.0;
    let mut c = a + GOLDEN_RATIO * (b - a);
    let mut d = b - GOLDEN_RATIO * (b - a);

    // Evaluate at initial points
    let mut fc = cost_function(&(start_point + c * direction))?;
    let mut fd = cost_function(&(start_point + d * direction))?;

    // Golden section search
    while (b - a).abs() > tolerance {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = a + GOLDEN_RATIO * (b - a);
            fc = cost_function(&(start_point + c * direction))?;
        } else {
            a = c;
            c = d;
            fc = fd;
            d = b - GOLDEN_RATIO * (b - a);
            fd = cost_function(&(start_point + d * direction))?;
        }
    }

    // Return the optimal _point
    let alpha = (a + b) / 2.0;
    let optimal_point = start_point + alpha * direction;
    let optimal_cost = cost_function(&optimal_point)?;

    Ok((optimal_point, optimal_cost))
}
