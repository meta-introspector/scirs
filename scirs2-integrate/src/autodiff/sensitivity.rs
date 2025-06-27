//! Sensitivity analysis tools
//!
//! This module provides tools for analyzing how solutions depend on parameters,
//! including local sensitivity analysis and global sensitivity indices.

use ndarray::{Array1, Array2, ArrayView1};
use crate::common::IntegrateFloat;
use crate::error::{IntegrateError, IntegrateResult};
use crate::ode::{solve_ivp, ODEOptions};
use std::collections::HashMap;

/// Parameter sensitivity information
#[derive(Clone)]
pub struct ParameterSensitivity<F: IntegrateFloat> {
    /// Parameter name
    pub name: String,
    /// Parameter index
    pub index: usize,
    /// Nominal value
    pub nominal_value: F,
    /// Sensitivity matrix (∂y/∂p)
    pub sensitivity: Array2<F>,
    /// Time points
    pub t_eval: Array1<F>,
}

/// Sensitivity analysis results
pub struct SensitivityAnalysis<F: IntegrateFloat> {
    /// Solution at nominal parameters
    pub nominal_solution: Array2<F>,
    /// Time points
    pub t_eval: Array1<F>,
    /// Parameter sensitivities
    pub sensitivities: Vec<ParameterSensitivity<F>>,
    /// First-order sensitivity indices (if computed)
    pub first_order_indices: Option<HashMap<String, Array1<F>>>,
    /// Total sensitivity indices (if computed)
    pub total_indices: Option<HashMap<String, Array1<F>>>,
}

impl<F: IntegrateFloat> SensitivityAnalysis<F> {
    /// Get sensitivity for a specific parameter
    pub fn get_sensitivity(&self, param_name: &str) -> Option<&ParameterSensitivity<F>> {
        self.sensitivities.iter().find(|s| s.name == param_name)
    }

    /// Compute relative sensitivities
    pub fn relative_sensitivities(&self) -> IntegrateResult<HashMap<String, Array2<F>>> {
        let mut result = HashMap::new();
        
        for sens in &self.sensitivities {
            let mut rel_sens = sens.sensitivity.clone();
            
            // Compute S_ij = (p_j / y_i) * (∂y_i/∂p_j)
            for i in 0..rel_sens.nrows() {
                for j in 0..rel_sens.ncols() {
                    let y_nominal = self.nominal_solution[[i, j]];
                    if y_nominal.abs() > F::epsilon() {
                        rel_sens[[i, j]] *= sens.nominal_value / y_nominal;
                    }
                }
            }
            
            result.insert(sens.name.clone(), rel_sens);
        }
        
        Ok(result)
    }

    /// Compute time-averaged sensitivities
    pub fn time_averaged_sensitivities(&self) -> HashMap<String, Array1<F>> {
        let mut result = HashMap::new();
        let n_time = self.t_eval.len();
        
        for sens in &self.sensitivities {
            let n_states = sens.sensitivity.ncols();
            let mut avg_sens = Array1::zeros(n_states);
            
            // Compute time average for each state variable
            for j in 0..n_states {
                let mut sum = F::zero();
                for i in 0..n_time {
                    sum += sens.sensitivity[[i, j]].abs();
                }
                avg_sens[j] = sum / F::from(n_time).unwrap();
            }
            
            result.insert(sens.name.clone(), avg_sens);
        }
        
        result
    }
}

/// Compute sensitivities using forward sensitivity analysis
pub fn compute_sensitivities<F, SysFunc, ParamFunc>(
    system: SysFunc,
    _parameters: ParamFunc,
    param_names: Vec<String>,
    nominal_params: ArrayView1<F>,
    y0: ArrayView1<F>,
    t_span: (F, F),
    _t_eval: Option<ArrayView1<F>>,
    options: Option<ODEOptions<F>>,
) -> IntegrateResult<SensitivityAnalysis<F>>
where
    F: IntegrateFloat,
    SysFunc: Fn(F, ArrayView1<F>, ArrayView1<F>) -> Array1<F> + Clone,
    ParamFunc: Fn(usize) -> Array1<F>,
{
    let n_states = y0.len();
    let n_params = nominal_params.len();
    
    if param_names.len() != n_params {
        return Err(IntegrateError::ValueError(
            "Number of parameter names must match number of parameters".to_string()
        ));
    }

    // Solve nominal system
    let opts = options.clone().unwrap_or_default();
    
    let nominal_result = solve_ivp(
        |t, y| system(t, y, nominal_params),
        [t_span.0, t_span.1],
        y0.to_owned(),
        Some(opts),
    )?;

    let t_points = nominal_result.t.clone();
    
    // Compute sensitivities for each parameter
    let mut sensitivities = Vec::new();
    
    for (param_idx, param_name) in param_names.iter().enumerate() {
        // Create augmented system for sensitivity equations
        let augmented_dim = n_states * (1 + 1); // States + sensitivity matrix
        let mut y0_aug = Array1::zeros(augmented_dim);
        
        // Initial conditions: y(0) and S(0) = 0
        y0_aug.slice_mut(ndarray::s![0..n_states]).assign(&y0);
        
        let system_clone = system.clone();
        let params = nominal_params.to_owned();
        
        // Augmented system: [dy/dt; dS/dt]
        let augmented_system = move |t: F, y_aug: ArrayView1<F>| -> Array1<F> {
            let y = y_aug.slice(ndarray::s![0..n_states]);
            let s = y_aug.slice(ndarray::s![n_states..]).to_owned()
                .into_shape_with_order((n_states,)).unwrap();
            
            // Compute f(t, y, p)
            let f = system_clone(t, y, params.view());
            
            // Compute ∂f/∂y using finite differences
            let eps = F::from(1e-8).unwrap();
            let mut df_dy = Array2::zeros((n_states, n_states));
            
            for j in 0..n_states {
                let mut y_pert = y.to_owned();
                y_pert[j] += eps;
                let f_pert = system_clone(t, y_pert.view(), params.view());
                
                for i in 0..n_states {
                    df_dy[[i, j]] = (f_pert[i] - f[i]) / eps;
                }
            }
            
            // Compute ∂f/∂p for the current parameter
            let mut params_pert = params.to_owned();
            params_pert[param_idx] += eps;
            let f_pert = system_clone(t, y, params_pert.view());
            let df_dp = (f_pert - &f) / eps;
            
            // dS/dt = ∂f/∂y * S + ∂f/∂p
            let ds_dt = df_dy.dot(&s) + df_dp;
            
            // Combine derivatives
            let mut result = Array1::zeros(augmented_dim);
            result.slice_mut(ndarray::s![0..n_states]).assign(&f);
            result.slice_mut(ndarray::s![n_states..]).assign(&ds_dt);
            
            result
        };
        
        // Solve augmented system
        let aug_opts = options.clone().unwrap_or_default();
        
        let aug_result = solve_ivp(
            augmented_system,
            [t_span.0, t_span.1],
            y0_aug,
            Some(aug_opts),
        )?;
        
        // Extract sensitivity matrix
        let aug_time = aug_result.t.len();
        let mut sensitivity = Array2::zeros((aug_time, n_states));
        for (i, sol) in aug_result.y.iter().enumerate() {
            let s = sol.slice(ndarray::s![n_states..]);
            sensitivity.row_mut(i).assign(&s);
        }
        
        sensitivities.push(ParameterSensitivity {
            name: param_name.clone(),
            index: param_idx,
            nominal_value: nominal_params[param_idx],
            sensitivity,
            t_eval: Array1::from_vec(aug_result.t.clone()),
        });
    }

    // Convert Vec<Array1<F>> to Array2<F>
    let n_points = nominal_result.t.len();
    let mut nominal_solution = Array2::zeros((n_points, n_states));
    for (i, sol) in nominal_result.y.iter().enumerate() {
        nominal_solution.row_mut(i).assign(sol);
    }
    
    Ok(SensitivityAnalysis {
        nominal_solution,
        t_eval: Array1::from_vec(t_points),
        sensitivities,
        first_order_indices: None,
        total_indices: None,
    })
}

/// Compute local sensitivity indices at a specific time
pub fn local_sensitivity_indices<F: IntegrateFloat>(
    analysis: &SensitivityAnalysis<F>,
    time_index: usize,
) -> IntegrateResult<HashMap<String, Array1<F>>> {
    let n_states = analysis.nominal_solution.ncols();
    let mut indices = HashMap::new();
    
    for sens in &analysis.sensitivities {
        let mut param_indices = Array1::zeros(n_states);
        
        for j in 0..n_states {
            let y_nominal = analysis.nominal_solution[[time_index, j]];
            let s_ij = sens.sensitivity[[time_index, j]];
            
            if y_nominal.abs() > F::epsilon() {
                // Normalized sensitivity index
                param_indices[j] = (s_ij * sens.nominal_value / y_nominal).abs();
            }
        }
        
        indices.insert(sens.name.clone(), param_indices);
    }
    
    Ok(indices)
}

/// Morris screening method for global sensitivity
pub struct MorrisScreening<F: IntegrateFloat> {
    /// Number of trajectories
    n_trajectories: usize,
    /// Parameter bounds
    param_bounds: Vec<(F, F)>,
    /// Grid levels
    grid_levels: usize,
}

impl<F: IntegrateFloat> MorrisScreening<F> {
    /// Create a new Morris screening analysis
    pub fn new(n_trajectories: usize, param_bounds: Vec<(F, F)>) -> Self {
        MorrisScreening {
            n_trajectories,
            param_bounds,
            grid_levels: 4,
        }
    }

    /// Set number of grid levels
    pub fn with_grid_levels(mut self, levels: usize) -> Self {
        self.grid_levels = levels;
        self
    }

    /// Compute elementary effects
    pub fn compute_effects<Func>(
        &self,
        model: Func,
        param_names: Vec<String>,
    ) -> IntegrateResult<HashMap<String, (F, F)>>
    where
        Func: Fn(ArrayView1<F>) -> IntegrateResult<F>,
    {
        let n_params = self.param_bounds.len();
        if param_names.len() != n_params {
            return Err(IntegrateError::ValueError(
                "Number of parameter names must match bounds".to_string()
            ));
        }

        let mut effects = HashMap::new();
        for name in &param_names {
            effects.insert(name.clone(), (F::zero(), F::zero()));
        }

        // Generate trajectories and compute elementary effects
        for _ in 0..self.n_trajectories {
            let trajectory = self.generate_trajectory(n_params);
            
            for i in 0..n_params {
                let p1 = trajectory[i].view();
                let p2 = trajectory[i + 1].view();
                
                let y1 = model(p1)?;
                let y2 = model(p2)?;
                
                // Find which parameter changed
                let mut changed_param = None;
                for j in 0..n_params {
                    if (p1[j] - p2[j]).abs() > F::epsilon() {
                        changed_param = Some(j);
                        break;
                    }
                }
                
                if let Some(j) = changed_param {
                    let delta = p2[j] - p1[j];
                    let ee = (y2 - y1) / delta;
                    
                    let name = &param_names[j];
                    let (sum, sum_sq) = effects.get_mut(name).unwrap();
                    *sum += ee;
                    *sum_sq += ee * ee;
                }
            }
        }

        // Compute mean and standard deviation
        let n_traj = F::from(self.n_trajectories).unwrap();
        let mut results = HashMap::new();
        
        for (name, (sum, sum_sq)) in effects {
            let mu = sum / n_traj;
            let sigma = ((sum_sq / n_traj) - mu * mu).sqrt();
            results.insert(name, (mu.abs(), sigma));
        }

        Ok(results)
    }

    /// Generate a Morris trajectory
    fn generate_trajectory(&self, n_params: usize) -> Vec<Array1<F>> {
        // Simplified trajectory generation
        let mut trajectory = Vec::new();
        let mut current = Array1::zeros(n_params);
        
        // Random starting point
        for i in 0..n_params {
            let (low, high) = self.param_bounds[i];
            current[i] = low + (high - low) * F::from(0.5).unwrap();
        }
        trajectory.push(current.clone());
        
        // Change one parameter at a time
        for i in 0..n_params {
            let (low, high) = self.param_bounds[i];
            let delta = (high - low) / F::from(self.grid_levels - 1).unwrap();
            current[i] += delta;
            trajectory.push(current.clone());
        }
        
        trajectory
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_sensitivity() {
        // Simple linear ODE: dy/dt = -a*y
        let system = |_t: f64, y: ArrayView1<f64>, p: ArrayView1<f64>| {
            Array1::from_vec(vec![-p[0] * y[0]])
        };
        
        let param_names = vec!["a".to_string()];
        let nominal_params = Array1::from_vec(vec![1.0]);
        let y0 = Array1::from_vec(vec![1.0]);
        let t_span = (0.0, 1.0);
        
        let analysis = compute_sensitivities(
            system,
            |_| Array1::from_vec(vec![1.0]),
            param_names,
            nominal_params.view(),
            y0.view(),
            t_span,
            None,
            None,
        );
        
        // Should complete without errors
        assert!(analysis.is_ok());
    }
}