//! Interactive parameter exploration tools
//!
//! This module provides interactive tools for exploring parameter spaces
//! and generating bifurcation diagrams for dynamical systems.

use super::types::*;
use crate::analysis::BifurcationPoint;
use crate::error::{IntegrateError, IntegrateResult};
use ndarray::Array1;
use rand::Rng;

/// Interactive parameter space explorer
#[derive(Debug, Clone)]
pub struct InteractiveParameterExplorer {
    /// Parameter space dimensions
    pub param_dimensions: usize,
    /// Parameter bounds
    pub param_bounds: Vec<(f64, f64)>,
    /// Number of samples per dimension
    pub samples_per_dim: usize,
    /// Current parameter values
    pub current_params: Array1<f64>,
    /// Parameter history
    pub param_history: Vec<Array1<f64>>,
    /// System response cache
    pub response_cache: std::collections::HashMap<String, Array1<f64>>,
}

impl InteractiveParameterExplorer {
    /// Create new parameter explorer
    pub fn new(
        param_dimensions: usize,
        param_bounds: Vec<(f64, f64)>,
        samples_per_dim: usize,
    ) -> Self {
        let current_params = Array1::from_vec(
            param_bounds
                .iter()
                .map(|(min, max)| (min + max) / 2.0)
                .collect(),
        );

        Self {
            param_dimensions,
            param_bounds,
            samples_per_dim,
            current_params,
            param_history: Vec::new(),
            response_cache: std::collections::HashMap::new(),
        }
    }

    /// Explore parameter space with interactive visualization
    pub fn explore_parameter_space<F>(
        &mut self,
        system_function: F,
        exploration_method: ExplorationMethod,
    ) -> IntegrateResult<ParameterExplorationResult>
    where
        F: Fn(&Array1<f64>) -> IntegrateResult<Array1<f64>>,
    {
        match exploration_method {
            ExplorationMethod::GridScan => self.grid_scan_exploration(&system_function),
            ExplorationMethod::RandomSampling => self.random_sampling_exploration(&system_function),
            ExplorationMethod::AdaptiveSampling => {
                self.adaptive_sampling_exploration(&system_function)
            }
            ExplorationMethod::GradientGuided => self.gradient_guided_exploration(&system_function),
        }
    }

    /// Grid-based parameter exploration
    fn grid_scan_exploration<F>(
        &mut self,
        system_function: &F,
    ) -> IntegrateResult<ParameterExplorationResult>
    where
        F: Fn(&Array1<f64>) -> IntegrateResult<Array1<f64>>,
    {
        let mut exploration_points = Vec::new();
        let mut response_values = Vec::new();
        let mut parameter_grid = Vec::new();

        // Generate grid points
        let grid_indices = self.generate_grid_indices();

        for indices in grid_indices {
            let params = self.indices_to_parameters(&indices);
            parameter_grid.push(params.clone());

            // Evaluate system at this parameter point
            match system_function(&params) {
                Ok(response) => {
                    exploration_points.push(params.clone());
                    response_values.push(response.clone());

                    // Cache result
                    let key = self.params_to_cache_key(&params);
                    self.response_cache.insert(key, response);
                }
                Err(_) => {
                    // Skip invalid parameter combinations
                    continue;
                }
            }
        }

        Ok(ParameterExplorationResult {
            exploration_points,
            response_values: response_values.clone(),
            parameter_grid,
            convergence_history: self.param_history.clone(),
            exploration_method: ExplorationMethod::GridScan,
            optimization_metrics: self.compute_exploration_metrics(&response_values)?,
        })
    }

    /// Random sampling exploration
    fn random_sampling_exploration<F>(
        &mut self,
        system_function: &F,
    ) -> IntegrateResult<ParameterExplorationResult>
    where
        F: Fn(&Array1<f64>) -> IntegrateResult<Array1<f64>>,
    {
        use rand::Rng;
        let mut rng = rand::rng();

        let mut exploration_points = Vec::new();
        let mut response_values = Vec::new();
        let mut parameter_grid = Vec::new();

        let total_samples = self.samples_per_dim.pow(self.param_dimensions as u32);

        for _ in 0..total_samples {
            let mut params = Array1::zeros(self.param_dimensions);

            for i in 0..self.param_dimensions {
                let (min, max) = self.param_bounds[i];
                params[i] = rng.random::<f64>() * (max - min) + min;
            }

            parameter_grid.push(params.clone());

            match system_function(&params) {
                Ok(response) => {
                    exploration_points.push(params.clone());
                    response_values.push(response.clone());

                    let key = self.params_to_cache_key(&params);
                    self.response_cache.insert(key, response);
                }
                Err(_) => continue,
            }
        }

        Ok(ParameterExplorationResult {
            exploration_points,
            response_values: response_values.clone(),
            parameter_grid,
            convergence_history: self.param_history.clone(),
            exploration_method: ExplorationMethod::RandomSampling,
            optimization_metrics: self.compute_exploration_metrics(&response_values)?,
        })
    }

    /// Adaptive sampling based on response characteristics
    fn adaptive_sampling_exploration<F>(
        &mut self,
        system_function: &F,
    ) -> IntegrateResult<ParameterExplorationResult>
    where
        F: Fn(&Array1<f64>) -> IntegrateResult<Array1<f64>>,
    {
        let mut exploration_points = Vec::new();
        let mut response_values = Vec::new();
        let mut parameter_grid = Vec::new();

        // Start with a coarse grid
        let coarse_samples = (self.samples_per_dim as f64).sqrt() as usize;
        let mut current_resolution = coarse_samples;

        // Initial coarse sampling
        let initial_grid = self.generate_coarse_grid(coarse_samples);

        for params in &initial_grid {
            parameter_grid.push(params.clone());

            match system_function(params) {
                Ok(response) => {
                    exploration_points.push(params.clone());
                    response_values.push(response);
                }
                Err(_) => continue,
            }
        }

        // Adaptive refinement
        while current_resolution < self.samples_per_dim {
            let refinement_candidates =
                self.identify_refinement_regions(&exploration_points, &response_values)?;

            for region in refinement_candidates {
                let refined_points = self.refine_region(&region, current_resolution * 2);

                for params in refined_points {
                    parameter_grid.push(params.clone());

                    match system_function(&params) {
                        Ok(response) => {
                            exploration_points.push(params.clone());
                            response_values.push(response);
                        }
                        Err(_) => continue,
                    }
                }
            }

            current_resolution *= 2;
        }

        Ok(ParameterExplorationResult {
            exploration_points,
            response_values: response_values.clone(),
            parameter_grid,
            convergence_history: self.param_history.clone(),
            exploration_method: ExplorationMethod::AdaptiveSampling,
            optimization_metrics: self.compute_exploration_metrics(&response_values)?,
        })
    }

    /// Gradient-guided exploration
    fn gradient_guided_exploration<F>(
        &mut self,
        system_function: &F,
    ) -> IntegrateResult<ParameterExplorationResult>
    where
        F: Fn(&Array1<f64>) -> IntegrateResult<Array1<f64>>,
    {
        let mut exploration_points = Vec::new();
        let mut response_values = Vec::new();
        let mut parameter_grid = Vec::new();

        // Start from current parameters
        let mut current_params = self.current_params.clone();
        exploration_points.push(current_params.clone());
        parameter_grid.push(current_params.clone());

        let initial_response = system_function(&current_params)?;
        response_values.push(initial_response.clone());

        let learning_rate = 0.01;
        let max_iterations = 100;
        let tolerance = 1e-6;

        for iteration in 0..max_iterations {
            // Compute numerical gradient
            let gradient = self.compute_numerical_gradient(system_function, &current_params)?;

            // Update parameters along gradient
            let gradient_norm = gradient.iter().map(|&g| g * g).sum::<f64>().sqrt();

            if gradient_norm < tolerance {
                break;
            }

            // Gradient ascent step (maximize response)
            for i in 0..self.param_dimensions {
                current_params[i] += learning_rate * gradient[i] / gradient_norm;

                // Clamp to bounds
                let (min, max) = self.param_bounds[i];
                current_params[i] = current_params[i].max(min).min(max);
            }

            // Evaluate at new point
            match system_function(&current_params) {
                Ok(response) => {
                    exploration_points.push(current_params.clone());
                    parameter_grid.push(current_params.clone());
                    response_values.push(response);
                    self.param_history.push(current_params.clone());
                }
                Err(_) => {
                    // If evaluation fails, take smaller step
                    current_params = exploration_points.last().unwrap().clone();
                    break;
                }
            }

            // Convergence check
            if iteration > 0 {
                let current_response = response_values.last().unwrap();
                let prev_response = &response_values[response_values.len() - 2];

                let response_change = current_response
                    .iter()
                    .zip(prev_response.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max);

                if response_change < tolerance {
                    break;
                }
            }
        }

        self.current_params = current_params;

        Ok(ParameterExplorationResult {
            exploration_points,
            response_values: response_values.clone(),
            parameter_grid,
            convergence_history: self.param_history.clone(),
            exploration_method: ExplorationMethod::GradientGuided,
            optimization_metrics: self.compute_exploration_metrics(&response_values)?,
        })
    }

    // Helper methods
    fn generate_grid_indices(&self) -> Vec<Vec<usize>> {
        let mut indices = Vec::new();
        let mut current_index = vec![0; self.param_dimensions];

        loop {
            indices.push(current_index.clone());

            // Increment counter
            let mut carry = 1;
            for i in (0..self.param_dimensions).rev() {
                current_index[i] += carry;
                if current_index[i] < self.samples_per_dim {
                    carry = 0;
                    break;
                } else {
                    current_index[i] = 0;
                }
            }

            if carry == 1 {
                break;
            }
        }

        indices
    }

    fn indices_to_parameters(&self, _indices: &[usize]) -> Array1<f64> {
        let mut params = Array1::zeros(self.param_dimensions);

        for i in 0..self.param_dimensions {
            let (min, max) = self.param_bounds[i];
            let fraction = _indices[i] as f64 / (self.samples_per_dim - 1) as f64;
            params[i] = min + fraction * (max - min);
        }

        params
    }

    fn params_to_cache_key(&self, _params: &Array1<f64>) -> String {
        _params
            .iter()
            .map(|&p| format!("{p:.6}"))
            .collect::<Vec<_>>()
            .join(",")
    }

    fn generate_coarse_grid(&self, _resolution: usize) -> Vec<Array1<f64>> {
        let mut grid = Vec::new();
        let mut indices = vec![0; self.param_dimensions];

        loop {
            let mut params = Array1::zeros(self.param_dimensions);

            for i in 0..self.param_dimensions {
                let (min, max) = self.param_bounds[i];
                let fraction = indices[i] as f64 / (_resolution - 1) as f64;
                params[i] = min + fraction * (max - min);
            }

            grid.push(params);

            // Increment
            let mut carry = 1;
            for i in (0..self.param_dimensions).rev() {
                indices[i] += carry;
                if indices[i] < _resolution {
                    carry = 0;
                    break;
                } else {
                    indices[i] = 0;
                }
            }

            if carry == 1 {
                break;
            }
        }

        grid
    }

    fn identify_refinement_regions(
        &self,
        _points: &[Array1<f64>],
        responses: &[Array1<f64>],
    ) -> IntegrateResult<Vec<ParameterRegion>> {
        let mut regions = Vec::new();

        // Simple heuristic: identify regions with high response variance
        if responses.len() < 2 {
            return Ok(regions);
        }

        // Find best response point for refinement
        let best_idx = responses
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                let a_norm = a.iter().map(|&x| x * x).sum::<f64>().sqrt();
                let b_norm = b.iter().map(|&x| x * x).sum::<f64>().sqrt();
                a_norm
                    .partial_cmp(&b_norm)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        let center = _points[best_idx].clone();
        let radius = 0.1; // 10% of parameter range

        regions.push(ParameterRegion { center, radius });

        Ok(regions)
    }

    fn refine_region(&self, _region: &ParameterRegion, resolution: usize) -> Vec<Array1<f64>> {
        let mut refined_points = Vec::new();

        // Generate points within the _region
        for _ in 0..resolution {
            let mut rng = rand::rng();
            let mut point = _region.center.clone();

            for i in 0..self.param_dimensions {
                let (min, max) = self.param_bounds[i];
                let range = (max - min) * _region.radius;
                let offset = (rng.random::<f64>() - 0.5) * 2.0 * range;
                point[i] = (point[i] + offset).max(min).min(max);
            }

            refined_points.push(point);
        }

        refined_points
    }

    fn compute_numerical_gradient<F>(
        &self,
        system_function: &F,
        params: &Array1<f64>,
    ) -> IntegrateResult<Array1<f64>>
    where
        F: Fn(&Array1<f64>) -> IntegrateResult<Array1<f64>>,
    {
        let epsilon = 1e-6;
        let mut gradient = Array1::zeros(self.param_dimensions);

        let base_response = system_function(params)?;
        let _base_norm = base_response.iter().map(|&x| x * x).sum::<f64>().sqrt();

        for i in 0..self.param_dimensions {
            let mut params_plus = params.clone();
            let mut params_minus = params.clone();

            params_plus[i] += epsilon;
            params_minus[i] -= epsilon;

            let response_plus = system_function(&params_plus)?;
            let response_minus = system_function(&params_minus)?;

            let norm_plus = response_plus.iter().map(|&x| x * x).sum::<f64>().sqrt();
            let norm_minus = response_minus.iter().map(|&x| x * x).sum::<f64>().sqrt();

            gradient[i] = (norm_plus - norm_minus) / (2.0 * epsilon);
        }

        Ok(gradient)
    }

    fn compute_exploration_metrics(
        &self,
        _responses: &[Array1<f64>],
    ) -> IntegrateResult<ExplorationMetrics> {
        if _responses.is_empty() {
            return Ok(ExplorationMetrics {
                max_response_norm: 0.0,
                min_response_norm: 0.0,
                mean_response_norm: 0.0,
                response_variance: 0.0,
                coverage_efficiency: 0.0,
            });
        }

        let norms: Vec<f64> = _responses
            .iter()
            .map(|r| r.iter().map(|&x| x * x).sum::<f64>().sqrt())
            .collect();

        let max_norm = norms.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_norm = norms.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let mean_norm = norms.iter().sum::<f64>() / norms.len() as f64;

        let variance =
            norms.iter().map(|&x| (x - mean_norm).powi(2)).sum::<f64>() / norms.len() as f64;

        // Simple coverage efficiency based on response diversity
        let coverage_efficiency = if max_norm > min_norm {
            variance / (max_norm - min_norm).powi(2)
        } else {
            1.0
        };

        Ok(ExplorationMetrics {
            max_response_norm: max_norm,
            min_response_norm: min_norm,
            mean_response_norm: mean_norm,
            response_variance: variance,
            coverage_efficiency,
        })
    }
}

/// Advanced Bifurcation Diagram Generator
#[derive(Debug, Clone)]
pub struct BifurcationDiagramGenerator {
    /// Parameter range for bifurcation analysis
    pub parameter_range: (f64, f64),
    /// Number of parameter samples
    pub n_parameter_samples: usize,
    /// Number of initial transient steps to skip
    pub transient_steps: usize,
    /// Number of sampling steps after transients
    pub sampling_steps: usize,
    /// Tolerance for detecting fixed points
    pub fixed_point_tolerance: f64,
    /// Tolerance for detecting periodic orbits
    pub period_tolerance: f64,
}

impl BifurcationDiagramGenerator {
    /// Create new bifurcation diagram generator
    pub fn new(_parameter_range: (f64, f64), n_parameter_samples: usize) -> Self {
        Self {
            parameter_range: _parameter_range,
            n_parameter_samples,
            transient_steps: 1000,
            sampling_steps: 500,
            fixed_point_tolerance: 1e-8,
            period_tolerance: 1e-6,
        }
    }

    /// Generate bifurcation diagram for 1D map
    pub fn generate_1d_map_diagram<F>(
        &self,
        map_function: F,
        initial_condition: f64,
    ) -> IntegrateResult<BifurcationDiagram>
    where
        F: Fn(f64, f64) -> f64, // (x, parameter) -> x_next
    {
        let mut parameter_values = Vec::new();
        let mut state_values = Vec::new();
        let mut stability_flags = Vec::new();
        let mut bifurcation_points = Vec::new();

        let param_step = (self.parameter_range.1 - self.parameter_range.0)
            / (self.n_parameter_samples - 1) as f64;

        for i in 0..self.n_parameter_samples {
            let param = self.parameter_range.0 + i as f64 * param_step;

            // Run transients
            let mut x = initial_condition;
            for _ in 0..self.transient_steps {
                x = map_function(x, param);
            }

            // Sample attractor
            let mut attractor_states = Vec::new();
            for _ in 0..self.sampling_steps {
                x = map_function(x, param);
                attractor_states.push(x);
            }

            // Analyze attractor structure
            let attractor_info = self.analyze_1d_attractor(&attractor_states)?;

            // Store results
            for &state in &attractor_info.representative_states {
                parameter_values.push(param);
                state_values.push(state);
                stability_flags.push(attractor_info.is_stable);
            }

            // Detect bifurcations
            if i > 0 {
                let prev_param = parameter_values
                    [parameter_values.len() - attractor_info.representative_states.len() - 1];
                if let Some(bif_point) =
                    self.detect_bifurcation_1d(prev_param, param, &map_function, initial_condition)?
                {
                    bifurcation_points.push(bif_point);
                }
            }
        }

        Ok(BifurcationDiagram {
            parameters: parameter_values,
            states: vec![state_values],
            stability: stability_flags,
            bifurcation_points,
        })
    }

    /// Analyze 1D attractor structure
    fn analyze_1d_attractor(&self, _states: &[f64]) -> IntegrateResult<AttractorInfo> {
        // Detect fixed points
        let mut representative_states = Vec::new();
        let mut unique_states = _states.to_vec();
        unique_states.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        unique_states.dedup_by(|a, b| (*a - *b).abs() < self.fixed_point_tolerance);

        if unique_states.len() == 1 {
            // Fixed point
            representative_states.push(unique_states[0]);
        } else {
            // Periodic orbit or chaos
            let period = self.detect_period(_states)?;

            if period > 0 && period <= self.sampling_steps / 10 {
                // Periodic orbit - sample one period
                for i in 0..period {
                    representative_states.push(_states[_states.len() - period + i]);
                }
            } else {
                // Chaotic - sample representative points
                let sample_rate = _states.len() / 20;
                for i in (0.._states.len()).step_by(sample_rate) {
                    representative_states.push(_states[i]);
                }
            }
        }

        // Simple stability analysis
        let is_stable = unique_states.len() <= 2; // Fixed point or period-2

        Ok(AttractorInfo {
            representative_states,
            is_stable,
            period: if unique_states.len() <= 2 {
                unique_states.len()
            } else {
                0
            },
        })
    }

    /// Detect period of oscillation
    fn detect_period(&self, _states: &[f64]) -> IntegrateResult<usize> {
        let max_period = (self.sampling_steps / 10).min(50);

        for period in 1..=max_period {
            let mut is_periodic = true;

            for i in 0..(_states.len() - period) {
                if (_states[i] - _states[i + period]).abs() > self.period_tolerance {
                    is_periodic = false;
                    break;
                }
            }

            if is_periodic {
                return Ok(period);
            }
        }

        Ok(0) // Not periodic within tolerance
    }

    /// Detect bifurcation between two parameter values
    fn detect_bifurcation_1d<F>(
        &self,
        param1: f64,
        param2: f64,
        map_function: &F,
        initial_condition: f64,
    ) -> IntegrateResult<Option<BifurcationPoint>>
    where
        F: Fn(f64, f64) -> f64,
    {
        // Simple bifurcation detection based on attractor dimension change
        let attractor1 = self.sample_attractor_1d(map_function, param1, initial_condition)?;
        let attractor2 = self.sample_attractor_1d(map_function, param2, initial_condition)?;

        let info1 = self.analyze_1d_attractor(&attractor1)?;
        let info2 = self.analyze_1d_attractor(&attractor2)?;

        if info1.representative_states.len() != info2.representative_states.len() {
            // Detected a change in attractor dimension
            let bif_param = (param1 + param2) / 2.0;
            let bif_state = Array1::from_vec(vec![info1.representative_states[0]]);

            let bif_type = match (
                info1.representative_states.len(),
                info2.representative_states.len(),
            ) {
                (1, 2) => crate::analysis::BifurcationType::PeriodDoubling,
                (1, _) => crate::analysis::BifurcationType::PeriodDoubling,
                (_, 1) => crate::analysis::BifurcationType::Fold,
                _ => crate::analysis::BifurcationType::Unknown,
            };

            return Ok(Some(BifurcationPoint {
                parameter_value: bif_param,
                state: bif_state,
                bifurcation_type: bif_type,
                eigenvalues: vec![], // Not computed for maps
            }));
        }

        Ok(None)
    }

    /// Sample attractor for given parameter
    fn sample_attractor_1d<F>(
        &self,
        map_function: &F,
        param: f64,
        initial_condition: f64,
    ) -> IntegrateResult<Vec<f64>>
    where
        F: Fn(f64, f64) -> f64,
    {
        let mut x = initial_condition;

        // Skip transients
        for _ in 0..self.transient_steps {
            x = map_function(x, param);
        }

        // Sample attractor
        let mut states = Vec::new();
        for _ in 0..self.sampling_steps {
            x = map_function(x, param);
            states.push(x);
        }

        Ok(states)
    }
}
