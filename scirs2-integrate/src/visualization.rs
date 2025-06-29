//! Visualization utilities for numerical integration and specialized solvers
//!
//! This module provides tools for visualizing results from various solvers,
//! including phase space plots, bifurcation diagrams, and field visualizations.

use crate::analysis::{BasinAnalysis, BifurcationPoint};
use crate::error::{IntegrateError, IntegrateResult as Result};
use crate::ode::ODEResult;
use ndarray::{Array1, Array2, Array3};
use scirs2_core::simd_ops::SimdUnifiedOps;
use std::collections::HashMap;

/// Data structure for plotting 2D phase space
#[derive(Debug, Clone)]
pub struct PhaseSpacePlot {
    /// X coordinates
    pub x: Vec<f64>,
    /// Y coordinates
    pub y: Vec<f64>,
    /// Optional color values for each point
    pub colors: Option<Vec<f64>>,
    /// Trajectory metadata
    pub metadata: PlotMetadata,
}

/// Data structure for bifurcation diagrams
#[derive(Debug, Clone)]
pub struct BifurcationDiagram {
    /// Parameter values
    pub parameters: Vec<f64>,
    /// State values (can be multiple branches)
    pub states: Vec<Vec<f64>>,
    /// Stability information for each point
    pub stability: Vec<bool>,
    /// Bifurcation points
    pub bifurcation_points: Vec<BifurcationPoint>,
}

/// Visualization metadata
#[derive(Debug, Clone)]
pub struct PlotMetadata {
    /// Plot title
    pub title: String,
    /// X-axis label
    pub xlabel: String,
    /// Y-axis label
    pub ylabel: String,
    /// Additional annotations
    pub annotations: HashMap<String, String>,
}

impl Default for PlotMetadata {
    fn default() -> Self {
        Self {
            title: "Numerical Integration Result".to_string(),
            xlabel: "X".to_string(),
            ylabel: "Y".to_string(),
            annotations: HashMap::new(),
        }
    }
}

/// Field visualization for 2D vector fields
#[derive(Debug, Clone)]
pub struct VectorFieldPlot {
    /// Grid x coordinates
    pub x_grid: Array2<f64>,
    /// Grid y coordinates
    pub y_grid: Array2<f64>,
    /// X components of vectors
    pub u: Array2<f64>,
    /// Y components of vectors
    pub v: Array2<f64>,
    /// Magnitude for color coding
    pub magnitude: Array2<f64>,
    /// Plot metadata
    pub metadata: PlotMetadata,
}

/// Heat map visualization for scalar fields
#[derive(Debug, Clone)]
pub struct HeatMapPlot {
    /// X coordinates
    pub x: Array1<f64>,
    /// Y coordinates
    pub y: Array1<f64>,
    /// Z values (scalar field)
    pub z: Array2<f64>,
    /// Plot metadata
    pub metadata: PlotMetadata,
}

/// 3D surface plot data
#[derive(Debug, Clone)]
pub struct SurfacePlot {
    /// X grid
    pub x: Array2<f64>,
    /// Y grid
    pub y: Array2<f64>,
    /// Z values
    pub z: Array2<f64>,
    /// Plot metadata
    pub metadata: PlotMetadata,
}

/// Visualization engine for creating plots
pub struct VisualizationEngine {
    /// Output format preference
    pub output_format: OutputFormat,
    /// Color scheme
    pub color_scheme: ColorScheme,
    /// Figure size
    pub figure_size: (f64, f64),
}

/// Output format options
#[derive(Debug, Clone, Copy)]
pub enum OutputFormat {
    /// ASCII art for terminal
    ASCII,
    /// CSV data
    CSV,
    /// JSON data
    JSON,
    /// SVG graphics
    SVG,
}

/// Color scheme options
#[derive(Debug, Clone, Copy)]
pub enum ColorScheme {
    /// Viridis (default)
    Viridis,
    /// Plasma
    Plasma,
    /// Inferno
    Inferno,
    /// Grayscale
    Grayscale,
}

impl Default for VisualizationEngine {
    fn default() -> Self {
        Self {
            output_format: OutputFormat::ASCII,
            color_scheme: ColorScheme::Viridis,
            figure_size: (800.0, 600.0),
        }
    }
}

impl VisualizationEngine {
    /// Create a new visualization engine
    pub fn new() -> Self {
        Default::default()
    }

    /// Create phase space plot from ODE result
    pub fn create_phase_space_plot<F: crate::common::IntegrateFloat>(
        &self,
        ode_result: &ODEResult<F>,
        x_index: usize,
        y_index: usize,
    ) -> Result<PhaseSpacePlot> {
        let n_points = ode_result.t.len();
        let n_vars = if !ode_result.y.is_empty() {
            ode_result.y[0].len()
        } else {
            0
        };

        if x_index >= n_vars || y_index >= n_vars {
            return Err(IntegrateError::ValueError(
                "Variable index out of bounds".to_string(),
            ));
        }

        let x: Vec<f64> = (0..n_points)
            .map(|i| ode_result.y[i][x_index].to_f64().unwrap_or(0.0))
            .collect();

        let y: Vec<f64> = (0..n_points)
            .map(|i| ode_result.y[i][y_index].to_f64().unwrap_or(0.0))
            .collect();

        // Color by time for trajectory visualization
        let colors: Vec<f64> = ode_result
            .t
            .iter()
            .map(|t| t.to_f64().unwrap_or(0.0))
            .collect();

        let mut metadata = PlotMetadata::default();
        metadata.title = "Phase Space Plot".to_string();
        metadata.xlabel = format!("Variable {}", x_index);
        metadata.ylabel = format!("Variable {}", y_index);

        Ok(PhaseSpacePlot {
            x,
            y,
            colors: Some(colors),
            metadata,
        })
    }

    /// Create bifurcation diagram from analysis results
    pub fn create_bifurcation_diagram(
        &self,
        bifurcation_points: &[BifurcationPoint],
        parameter_range: (f64, f64),
        n_points: usize,
    ) -> Result<BifurcationDiagram> {
        let mut parameters = Vec::new();
        let mut states = Vec::new();
        let mut stability = Vec::new();

        // Create parameter grid
        let param_step = (parameter_range.1 - parameter_range.0) / (n_points - 1) as f64;
        for i in 0..n_points {
            let param = parameter_range.0 + i as f64 * param_step;
            parameters.push(param);

            // Find corresponding state (simplified)
            let mut found = false;
            for bif_point in bifurcation_points {
                if (bif_point.parameter_value - param).abs() < param_step {
                    states.push(bif_point.state.to_vec());
                    // Simplified stability check based on eigenvalues
                    let is_stable = bif_point.eigenvalues.iter().all(|eig| eig.re < 0.0);
                    stability.push(is_stable);
                    found = true;
                    break;
                }
            }

            if !found {
                states.push(vec![0.0]); // Default value
                stability.push(true);
            }
        }

        Ok(BifurcationDiagram {
            parameters,
            states,
            stability,
            bifurcation_points: bifurcation_points.to_vec(),
        })
    }

    /// Create vector field plot for 2D dynamical systems
    pub fn create_vector_field_plot<F>(
        &self,
        system: F,
        x_range: (f64, f64),
        y_range: (f64, f64),
        grid_size: (usize, usize),
    ) -> Result<VectorFieldPlot>
    where
        F: Fn(&Array1<f64>) -> Array1<f64>,
    {
        let (nx, ny) = grid_size;
        let mut x_grid = Array2::zeros((ny, nx));
        let mut y_grid = Array2::zeros((ny, nx));
        let mut u = Array2::zeros((ny, nx));
        let mut v = Array2::zeros((ny, nx));
        let mut magnitude = Array2::zeros((ny, nx));

        let dx = (x_range.1 - x_range.0) / (nx - 1) as f64;
        let dy = (y_range.1 - y_range.0) / (ny - 1) as f64;

        for i in 0..ny {
            for j in 0..nx {
                let x = x_range.0 + j as f64 * dx;
                let y = y_range.0 + i as f64 * dy;

                x_grid[[i, j]] = x;
                y_grid[[i, j]] = y;

                let state = Array1::from_vec(vec![x, y]);
                let derivative = system(&state);

                if derivative.len() >= 2 {
                    u[[i, j]] = derivative[0];
                    v[[i, j]] = derivative[1];
                    magnitude[[i, j]] = (derivative[0].powi(2) + derivative[1].powi(2)).sqrt();
                }
            }
        }

        let mut metadata = PlotMetadata::default();
        metadata.title = "Vector Field Plot".to_string();
        metadata.xlabel = "X".to_string();
        metadata.ylabel = "Y".to_string();

        Ok(VectorFieldPlot {
            x_grid,
            y_grid,
            u,
            v,
            magnitude,
            metadata,
        })
    }

    /// Create basin of attraction visualization
    pub fn create_basin_plot(&self, basin_analysis: &BasinAnalysis) -> Result<HeatMapPlot> {
        let grid_size = basin_analysis.attractor_indices.nrows();
        let x = Array1::linspace(0.0, 1.0, grid_size);
        let y = Array1::linspace(0.0, 1.0, grid_size);

        // Convert attractor indices to f64 for plotting
        let z = basin_analysis.attractor_indices.mapv(|x| x as f64);

        let mut metadata = PlotMetadata::default();
        metadata.title = "Basin of Attraction".to_string();
        metadata.xlabel = "X".to_string();
        metadata.ylabel = "Y".to_string();

        Ok(HeatMapPlot { x, y, z, metadata })
    }

    /// Generate ASCII art representation of a 2D plot
    pub fn render_ascii_plot(&self, data: &[(f64, f64)], width: usize, height: usize) -> String {
        if data.is_empty() {
            return "No data to plot".to_string();
        }

        // Find data bounds
        let x_min = data.iter().map(|(x, _)| *x).fold(f64::INFINITY, f64::min);
        let x_max = data
            .iter()
            .map(|(x, _)| *x)
            .fold(f64::NEG_INFINITY, f64::max);
        let y_min = data.iter().map(|(_, y)| *y).fold(f64::INFINITY, f64::min);
        let y_max = data
            .iter()
            .map(|(_, y)| *y)
            .fold(f64::NEG_INFINITY, f64::max);

        // Create character grid
        let mut grid = vec![vec![' '; width]; height];

        // Map data points to grid
        for (x, y) in data {
            let i = ((y - y_min) / (y_max - y_min) * (height - 1) as f64) as usize;
            let j = ((x - x_min) / (x_max - x_min) * (width - 1) as f64) as usize;

            if i < height && j < width {
                grid[height - 1 - i][j] = '*'; // Flip y-axis for proper orientation
            }
        }

        // Convert grid to string
        let mut result = String::new();
        for row in grid {
            result.push_str(&row.iter().collect::<String>());
            result.push('\n');
        }

        // Add axis labels
        result.push_str(&format!("\nX range: [{:.3}, {:.3}]\n", x_min, x_max));
        result.push_str(&format!("Y range: [{:.3}, {:.3}]\n", y_min, y_max));

        result
    }

    /// Export plot data to CSV format
    pub fn export_csv(&self, plot: &PhaseSpacePlot) -> Result<String> {
        let mut csv = String::new();

        // Header
        csv.push_str("x,y");
        if plot.colors.is_some() {
            csv.push_str(",color");
        }
        csv.push('\n');

        // Data
        for i in 0..plot.x.len() {
            csv.push_str(&format!("{},{}", plot.x[i], plot.y[i]));
            if let Some(ref colors) = plot.colors {
                csv.push_str(&format!(",{}", colors[i]));
            }
            csv.push('\n');
        }

        Ok(csv)
    }

    /// Create learning curve plot for optimization algorithms
    pub fn create_learning_curve(
        &self,
        iterations: &[usize],
        values: &[f64],
        title: &str,
    ) -> Result<PhaseSpacePlot> {
        let x: Vec<f64> = iterations.iter().map(|&i| i as f64).collect();
        let y = values.to_vec();

        let mut metadata = PlotMetadata::default();
        metadata.title = title.to_string();
        metadata.xlabel = "Iteration".to_string();
        metadata.ylabel = "Value".to_string();

        Ok(PhaseSpacePlot {
            x,
            y,
            colors: None,
            metadata,
        })
    }

    /// Create convergence analysis plot
    pub fn create_convergence_plot(
        &self,
        step_sizes: &[f64],
        errors: &[f64],
        theoretical_order: f64,
    ) -> Result<PhaseSpacePlot> {
        let x: Vec<f64> = step_sizes.iter().map(|h| h.log10()).collect();
        let y: Vec<f64> = errors.iter().map(|e| e.log10()).collect();

        let mut metadata = PlotMetadata::default();
        metadata.title = "Convergence Analysis".to_string();
        metadata.xlabel = "log10(step size)".to_string();
        metadata.ylabel = "log10(error)".to_string();
        metadata.annotations.insert(
            "theoretical_slope".to_string(),
            theoretical_order.to_string(),
        );

        Ok(PhaseSpacePlot {
            x,
            y,
            colors: None,
            metadata,
        })
    }

    /// Create interactive parameter space exploration plot
    pub fn create_parameter_exploration(
        &self,
        param_ranges: &[(f64, f64)], // [(min1, max1), (min2, max2), ...]
        param_names: &[String],
        evaluation_function: &dyn Fn(&[f64]) -> f64,
        resolution: usize,
    ) -> Result<ParameterExplorationPlot> {
        if param_ranges.len() != 2 {
            return Err(IntegrateError::ValueError(
                "Parameter exploration currently supports only 2D parameter spaces".to_string(),
            ));
        }

        let (x_min, x_max) = param_ranges[0];
        let (y_min, y_max) = param_ranges[1];

        let dx = (x_max - x_min) / (resolution - 1) as f64;
        let dy = (y_max - y_min) / (resolution - 1) as f64;

        let mut x_grid = Array2::zeros((resolution, resolution));
        let mut y_grid = Array2::zeros((resolution, resolution));
        let mut z_values = Array2::zeros((resolution, resolution));

        for i in 0..resolution {
            for j in 0..resolution {
                let x = x_min + i as f64 * dx;
                let y = y_min + j as f64 * dy;

                x_grid[[i, j]] = x;
                y_grid[[i, j]] = y;
                z_values[[i, j]] = evaluation_function(&[x, y]);
            }
        }

        let mut metadata = PlotMetadata::default();
        metadata.title = "Parameter Space Exploration".to_string();
        metadata.xlabel = param_names
            .get(0)
            .cloned()
            .unwrap_or_else(|| "Parameter 1".to_string());
        metadata.ylabel = param_names
            .get(1)
            .cloned()
            .unwrap_or_else(|| "Parameter 2".to_string());

        Ok(ParameterExplorationPlot {
            x_grid,
            y_grid,
            z_values,
            param_ranges: param_ranges.to_vec(),
            param_names: param_names.to_vec(),
            metadata,
        })
    }

    /// Create real-time bifurcation diagram
    pub fn create_real_time_bifurcation(
        &self,
        system: &dyn Fn(&Array1<f64>, f64) -> Array1<f64>,
        parameter_range: (f64, f64),
        initial_conditions: &[Array1<f64>],
        transient_steps: usize,
        record_steps: usize,
    ) -> Result<RealTimeBifurcationPlot> {
        let n_params = 200;
        let param_step = (parameter_range.1 - parameter_range.0) / (n_params - 1) as f64;

        let mut parameter_values = Vec::new();
        let mut attractor_data = Vec::new();
        let mut stability_data = Vec::new();

        for i in 0..n_params {
            let param = parameter_range.0 + i as f64 * param_step;
            parameter_values.push(param);

            let mut param_attractors = Vec::new();
            let mut param_stability = Vec::new();

            for initial in initial_conditions {
                // Evolve system to let transients die out
                let mut state = initial.clone();
                for _ in 0..transient_steps {
                    let derivative = system(&state, param);
                    state += &(&derivative * 0.01); // Small time step
                }

                // Record attractor points
                let mut attractor_points = Vec::new();
                let mut local_maxima = Vec::new();

                for step in 0..record_steps {
                    let derivative = system(&state, param);
                    let derivative_scaled = &derivative * 0.01;
                    let new_state = &state + &derivative_scaled;

                    // Simple local maxima detection for period identification
                    if step > 2
                        && new_state[0] > state[0]
                        && state[0] > (state.clone() - &derivative_scaled)[0]
                    {
                        local_maxima.push(state[0]);
                    }

                    attractor_points.push(state[0]);
                    state = new_state;
                }

                // Determine stability based on attractor behavior
                let stability = if local_maxima.len() == 1 {
                    AttractorStability::FixedPoint
                } else if local_maxima.len() == 2 {
                    AttractorStability::PeriodTwo
                } else if local_maxima.len() > 2 && local_maxima.len() < 10 {
                    AttractorStability::Periodic(local_maxima.len())
                } else {
                    AttractorStability::Chaotic
                };

                param_attractors.push(attractor_points);
                param_stability.push(stability);
            }

            attractor_data.push(param_attractors);
            stability_data.push(param_stability);
        }

        let mut metadata = PlotMetadata::default();
        metadata.title = "Real-time Bifurcation Diagram".to_string();
        metadata.xlabel = "Parameter".to_string();
        metadata.ylabel = "Attractor Values".to_string();

        Ok(RealTimeBifurcationPlot {
            parameter_values,
            attractor_data,
            stability_data,
            parameter_range,
            metadata,
        })
    }

    /// Create 3D phase space trajectory
    pub fn create_3d_phase_space<F: crate::common::IntegrateFloat>(
        &self,
        ode_result: &ODEResult<F>,
        x_index: usize,
        y_index: usize,
        z_index: usize,
    ) -> Result<PhaseSpace3D> {
        let n_points = ode_result.t.len();
        let n_vars = if !ode_result.y.is_empty() {
            ode_result.y[0].len()
        } else {
            0
        };

        if x_index >= n_vars || y_index >= n_vars || z_index >= n_vars {
            return Err(IntegrateError::ValueError(
                "Variable index out of bounds".to_string(),
            ));
        }

        let x: Vec<f64> = (0..n_points)
            .map(|i| ode_result.y[i][x_index].to_f64().unwrap_or(0.0))
            .collect();

        let y: Vec<f64> = (0..n_points)
            .map(|i| ode_result.y[i][y_index].to_f64().unwrap_or(0.0))
            .collect();

        let z: Vec<f64> = (0..n_points)
            .map(|i| ode_result.y[i][z_index].to_f64().unwrap_or(0.0))
            .collect();

        // Color by time or by distance from initial point
        let colors: Vec<f64> = ode_result
            .t
            .iter()
            .map(|t| t.to_f64().unwrap_or(0.0))
            .collect();

        let mut metadata = PlotMetadata::default();
        metadata.title = "3D Phase Space Trajectory".to_string();
        metadata.xlabel = format!("Variable {}", x_index);
        metadata.ylabel = format!("Variable {}", y_index);
        metadata
            .annotations
            .insert("zlabel".to_string(), format!("Variable {}", z_index));

        Ok(PhaseSpace3D {
            x,
            y,
            z,
            colors: Some(colors),
            metadata,
        })
    }

    /// Create interactive sensitivity analysis plot
    pub fn create_sensitivity_analysis(
        &self,
        base_parameters: &[f64],
        parameter_names: &[String],
        sensitivity_function: &dyn Fn(&[f64]) -> f64,
        perturbation_percent: f64,
    ) -> Result<SensitivityPlot> {
        let n_params = base_parameters.len();
        let mut sensitivities = Vec::with_capacity(n_params);
        let base_value = sensitivity_function(base_parameters);

        for i in 0..n_params {
            let mut perturbed_params = base_parameters.to_vec();
            let perturbation = base_parameters[i] * perturbation_percent / 100.0;

            // Forward difference
            perturbed_params[i] += perturbation;
            let perturbed_value = sensitivity_function(&perturbed_params);

            // Calculate normalized sensitivity
            let sensitivity = if perturbation.abs() > 1e-12 {
                (perturbed_value - base_value) / perturbation * base_parameters[i] / base_value
            } else {
                0.0
            };

            sensitivities.push(sensitivity);
        }

        let mut metadata = PlotMetadata::default();
        metadata.title = "Parameter Sensitivity Analysis".to_string();
        metadata.xlabel = "Parameters".to_string();
        metadata.ylabel = "Normalized Sensitivity".to_string();

        Ok(SensitivityPlot {
            parameter_names: parameter_names.to_vec(),
            sensitivities,
            base_parameters: base_parameters.to_vec(),
            base_value,
            metadata,
        })
    }
}

/// Utility functions for common visualization tasks
pub mod utils {
    use super::*;

    /// Generate color map values
    pub fn generate_colormap(values: &[f64], scheme: ColorScheme) -> Vec<(u8, u8, u8)> {
        let n = values.len();
        let mut colors = Vec::with_capacity(n);

        let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let range = max_val - min_val;

        for &val in values {
            let normalized = if range > 0.0 {
                (val - min_val) / range
            } else {
                0.5
            };

            let color = match scheme {
                ColorScheme::Viridis => viridis_color(normalized),
                ColorScheme::Plasma => plasma_color(normalized),
                ColorScheme::Inferno => inferno_color(normalized),
                ColorScheme::Grayscale => {
                    let gray = (normalized * 255.0) as u8;
                    (gray, gray, gray)
                }
            };
            colors.push(color);
        }

        colors
    }

    /// Viridis colormap
    fn viridis_color(t: f64) -> (u8, u8, u8) {
        let t = t.max(0.0).min(1.0);
        let r = (0.267004 + t * (0.127568 + t * (0.019234 - t * 0.012814))) * 255.0;
        let g = (0.004874 + t * (0.950141 + t * (-0.334896 + t * 0.158789))) * 255.0;
        let b = (0.329415 + t * (0.234092 + t * (1.384085 - t * 1.388488))) * 255.0;
        (r as u8, g as u8, b as u8)
    }

    /// Plasma colormap
    fn plasma_color(t: f64) -> (u8, u8, u8) {
        let t = t.max(0.0).min(1.0);
        let r = (0.050383 + t * (1.075483 + t * (-0.346066 + t * 0.220971))) * 255.0;
        let g = (0.029803 + t * (0.089467 + t * (1.234884 - t * 1.281864))) * 255.0;
        let b = (0.527975 + t * (0.670134 + t * (-1.397127 + t * 1.149498))) * 255.0;
        (r as u8, g as u8, b as u8)
    }

    /// Inferno colormap
    fn inferno_color(t: f64) -> (u8, u8, u8) {
        let t = t.max(0.0).min(1.0);
        let r = (0.001462 + t * (0.998260 + t * (-0.149678 + t * 0.150124))) * 255.0;
        let g = (0.000466 + t * (0.188724 + t * (1.203007 - t * 1.391543))) * 255.0;
        let b = (0.013866 + t * (0.160930 + t * (0.690929 - t * 0.865624))) * 255.0;
        (r as u8, g as u8, b as u8)
    }

    /// Calculate optimal grid resolution for vector field plots
    pub fn optimal_grid_resolution(domain_size: (f64, f64), target_density: f64) -> (usize, usize) {
        let (width, height) = domain_size;
        let area = width * height;
        let total_points = (area * target_density) as usize;

        let aspect_ratio = width / height;
        let ny = (total_points as f64 / aspect_ratio).sqrt() as usize;
        let nx = (ny as f64 * aspect_ratio) as usize;

        (nx.max(10), ny.max(10))
    }

    /// Create summary statistics for plot data
    pub fn plot_statistics(data: &[f64]) -> PlotStatistics {
        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let min = sorted_data[0];
        let max = sorted_data[sorted_data.len() - 1];
        let median = if sorted_data.len() % 2 == 0 {
            (sorted_data[sorted_data.len() / 2 - 1] + sorted_data[sorted_data.len() / 2]) / 2.0
        } else {
            sorted_data[sorted_data.len() / 2]
        };

        PlotStatistics {
            count: data.len(),
            mean,
            std_dev,
            min,
            max,
            median,
        }
    }
}

/// Statistical summary of plot data
#[derive(Debug, Clone)]
pub struct PlotStatistics {
    pub count: usize,
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub median: f64,
}

/// Parameter exploration plot for 2D parameter spaces
#[derive(Debug, Clone)]
pub struct ParameterExplorationPlot {
    /// X parameter grid
    pub x_grid: Array2<f64>,
    /// Y parameter grid  
    pub y_grid: Array2<f64>,
    /// Function values at each grid point
    pub z_values: Array2<f64>,
    /// Parameter ranges
    pub param_ranges: Vec<(f64, f64)>,
    /// Parameter names
    pub param_names: Vec<String>,
    /// Plot metadata
    pub metadata: PlotMetadata,
}

/// Stability classification for attractors
#[derive(Debug, Clone, PartialEq)]
pub enum AttractorStability {
    /// Fixed point attractor
    FixedPoint,
    /// Period-2 cycle
    PeriodTwo,
    /// Higher-order periodic cycle
    Periodic(usize),
    /// Quasi-periodic attractor
    QuasiPeriodic,
    /// Chaotic attractor
    Chaotic,
    /// Unknown/undetermined
    Unknown,
}

/// Real-time bifurcation diagram
#[derive(Debug, Clone)]
pub struct RealTimeBifurcationPlot {
    /// Parameter values
    pub parameter_values: Vec<f64>,
    /// Attractor data for each parameter value and initial condition
    pub attractor_data: Vec<Vec<Vec<f64>>>,
    /// Stability classification for each attractor
    pub stability_data: Vec<Vec<AttractorStability>>,
    /// Parameter range
    pub parameter_range: (f64, f64),
    /// Plot metadata
    pub metadata: PlotMetadata,
}

/// 3D phase space trajectory
#[derive(Debug, Clone)]
pub struct PhaseSpace3D {
    /// X coordinates
    pub x: Vec<f64>,
    /// Y coordinates
    pub y: Vec<f64>,
    /// Z coordinates
    pub z: Vec<f64>,
    /// Optional color values for each point
    pub colors: Option<Vec<f64>>,
    /// Plot metadata
    pub metadata: PlotMetadata,
}

/// Sensitivity analysis plot
#[derive(Debug, Clone)]
pub struct SensitivityPlot {
    /// Parameter names
    pub parameter_names: Vec<String>,
    /// Sensitivity values for each parameter
    pub sensitivities: Vec<f64>,
    /// Base parameter values
    pub base_parameters: Vec<f64>,
    /// Base function value
    pub base_value: f64,
    /// Plot metadata
    pub metadata: PlotMetadata,
}

/// Interactive plot controls
#[derive(Debug, Clone)]
pub struct InteractivePlotControls {
    /// Zoom level
    pub zoom: f64,
    /// Pan offset (x, y)
    pub pan_offset: (f64, f64),
    /// Selected parameter ranges
    pub selected_ranges: Vec<(f64, f64)>,
    /// Animation frame
    pub current_frame: usize,
    /// Animation speed
    pub animation_speed: f64,
    /// Show/hide elements
    pub visibility_flags: std::collections::HashMap<String, bool>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_visualization_engine() {
        let engine = VisualizationEngine::new();
        assert_eq!(engine.output_format as i32, OutputFormat::ASCII as i32);
    }

    #[test]
    fn test_ascii_plot() {
        let engine = VisualizationEngine::new();
        let data = vec![(0.0, 0.0), (1.0, 1.0), (2.0, 4.0)];
        let plot = engine.render_ascii_plot(&data, 20, 10);
        assert!(plot.contains("*"));
        assert!(plot.contains("X range"));
        assert!(plot.contains("Y range"));
    }

    #[test]
    fn test_colormap_generation() {
        let values = vec![0.0, 0.5, 1.0];
        let colors = utils::generate_colormap(&values, ColorScheme::Grayscale);
        assert_eq!(colors.len(), 3);
        assert_eq!(colors[0], (0, 0, 0)); // Black for min
        assert_eq!(colors[2], (255, 255, 255)); // White for max
    }

    #[test]
    fn test_plot_statistics() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = utils::plot_statistics(&data);
        assert_relative_eq!(stats.mean, 3.0, epsilon = 1e-10);
        assert_relative_eq!(stats.median, 3.0, epsilon = 1e-10);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
    }
}

/// Advanced visualization features for scientific computing
pub mod advanced_visualization {
    use super::*;
    use crate::ode::ODEResult;
    use ndarray::{Array1, Array2};
    use rand::Rng;

    /// Multi-dimensional data visualization engine
    pub struct MultiDimensionalVisualizer {
        /// Dimension reduction method
        pub reduction_method: DimensionReductionMethod,
        /// Target dimensions for visualization
        pub target_dimensions: usize,
        /// Clustering method for data grouping
        pub clustering_method: ClusteringMethod,
    }

    /// Dimension reduction methods
    #[derive(Debug, Clone, Copy)]
    pub enum DimensionReductionMethod {
        /// Principal Component Analysis
        PCA,
        /// t-Distributed Stochastic Neighbor Embedding
        TSNE,
        /// Uniform Manifold Approximation and Projection
        UMAP,
        /// Linear Discriminant Analysis
        LDA,
        /// Multidimensional Scaling
        MDS,
    }

    /// Clustering methods for visualization
    #[derive(Debug, Clone, Copy)]
    pub enum ClusteringMethod {
        /// K-means clustering
        KMeans { k: usize },
        /// DBSCAN clustering
        DBSCAN { eps: f64, min_samples: usize },
        /// Hierarchical clustering
        Hierarchical { n_clusters: usize },
        /// No clustering
        None,
    }

    impl MultiDimensionalVisualizer {
        /// Create new multi-dimensional visualizer
        pub fn new() -> Self {
            Self {
                reduction_method: DimensionReductionMethod::PCA,
                target_dimensions: 2,
                clustering_method: ClusteringMethod::None,
            }
        }

        /// Visualize high-dimensional data
        pub fn visualize_high_dimensional_data(
            &self,
            data: &Array2<f64>,
            labels: Option<&Array1<usize>>,
        ) -> Result<HighDimensionalPlot> {
            // Apply dimension reduction
            let reduced_data = self.apply_dimension_reduction(data)?;

            // Apply clustering if specified
            let cluster_labels = self.apply_clustering(&reduced_data)?;

            // Create plot data
            let x: Vec<f64> = reduced_data.column(0).to_vec();
            let y: Vec<f64> = if reduced_data.ncols() > 1 {
                reduced_data.column(1).to_vec()
            } else {
                vec![0.0; x.len()]
            };

            let z: Option<Vec<f64>> = if self.target_dimensions > 2 && reduced_data.ncols() > 2 {
                Some(reduced_data.column(2).to_vec())
            } else {
                None
            };

            let colors = if let Some(labels) = labels {
                labels.to_vec().into_iter().map(|l| l as f64).collect()
            } else if let Some(clusters) = &cluster_labels {
                clusters.iter().map(|&c| c as f64).collect()
            } else {
                (0..x.len()).map(|i| i as f64).collect()
            };

            let mut metadata = PlotMetadata::default();
            metadata.title = "High-Dimensional Data Visualization".to_string();
            metadata.xlabel = format!("{:?} Component 1", self.reduction_method);
            metadata.ylabel = format!("{:?} Component 2", self.reduction_method);

            Ok(HighDimensionalPlot {
                x,
                y,
                z,
                colors,
                cluster_labels,
                original_dimensions: data.ncols(),
                reduced_dimensions: self.target_dimensions,
                reduction_method: self.reduction_method,
                metadata,
            })
        }

        /// Apply dimension reduction to data
        fn apply_dimension_reduction(&self, data: &Array2<f64>) -> Result<Array2<f64>> {
            match self.reduction_method {
                DimensionReductionMethod::PCA => self.apply_pca(data),
                DimensionReductionMethod::TSNE => self.apply_tsne(data),
                DimensionReductionMethod::UMAP => self.apply_umap(data),
                DimensionReductionMethod::LDA => self.apply_lda(data),
                DimensionReductionMethod::MDS => self.apply_mds(data),
            }
        }

        /// Apply Principal Component Analysis
        fn apply_pca(&self, data: &Array2<f64>) -> Result<Array2<f64>> {
            let (n_samples, n_features) = data.dim();

            // Center the data
            let mean = data.mean_axis(ndarray::Axis(0)).unwrap();
            let centered_data = data - &mean.insert_axis(ndarray::Axis(0));

            // Compute covariance matrix
            let cov_matrix = centered_data.t().dot(&centered_data) / (n_samples - 1) as f64;

            // Simplified eigenvalue decomposition (for small matrices)
            let (eigenvalues, eigenvectors) = self.compute_eigendecomposition(&cov_matrix)?;

            // Sort by eigenvalue magnitude (descending)
            let mut eigenvalue_indices: Vec<usize> = (0..eigenvalues.len()).collect();
            eigenvalue_indices
                .sort_by(|&i, &j| eigenvalues[j].partial_cmp(&eigenvalues[i]).unwrap());

            // Project data onto principal components
            let n_components = self.target_dimensions.min(n_features);
            let mut projected_data = Array2::zeros((n_samples, n_components));

            for (i, &idx) in eigenvalue_indices.iter().take(n_components).enumerate() {
                let component = eigenvectors.column(idx);
                let projection = centered_data.dot(&component);
                projected_data.column_mut(i).assign(&projection);
            }

            Ok(projected_data)
        }

        /// Apply t-SNE (simplified implementation)
        fn apply_tsne(&self, data: &Array2<f64>) -> Result<Array2<f64>> {
            // Simplified t-SNE implementation
            // In practice, would use a proper t-SNE algorithm
            let (n_samples, _) = data.dim();
            let mut rng = rand::rng();

            // Random initialization
            let mut embedding = Array2::zeros((n_samples, self.target_dimensions));
            for i in 0..n_samples {
                for j in 0..self.target_dimensions {
                    embedding[[i, j]] = rng.random::<f64>() * 2.0 - 1.0;
                }
            }

            // Simplified optimization (would need proper gradient descent)
            let learning_rate = 200.0;
            let n_iterations = 1000;

            for _iter in 0..n_iterations {
                // Compute pairwise similarities in high dimension
                let p_similarities = self.compute_gaussian_similarities(data, 1.0)?;

                // Compute pairwise similarities in low dimension
                let q_similarities = self.compute_t_similarities(&embedding)?;

                // Gradient descent step (simplified)
                for i in 0..n_samples {
                    for j in 0..self.target_dimensions {
                        let mut gradient = 0.0;
                        for k in 0..n_samples {
                            if i != k {
                                let p_ik = p_similarities[[i, k]];
                                let q_ik = q_similarities[[i, k]];
                                let diff = embedding[[i, j]] - embedding[[k, j]];
                                gradient += 4.0 * (p_ik - q_ik) * diff;
                            }
                        }
                        embedding[[i, j]] -= learning_rate * gradient / n_samples as f64;
                    }
                }
            }

            Ok(embedding)
        }

        /// Apply UMAP (simplified implementation)
        fn apply_umap(&self, data: &Array2<f64>) -> Result<Array2<f64>> {
            // Simplified UMAP - in practice would use proper UMAP algorithm
            // For now, fall back to PCA
            self.apply_pca(data)
        }

        /// Apply Linear Discriminant Analysis
        fn apply_lda(&self, data: &Array2<f64>) -> Result<Array2<f64>> {
            // Simplified LDA - would need class labels for proper implementation
            // For now, fall back to PCA
            self.apply_pca(data)
        }

        /// Apply Multidimensional Scaling
        fn apply_mds(&self, data: &Array2<f64>) -> Result<Array2<f64>> {
            let n_samples = data.nrows();

            // Compute distance matrix
            let mut distance_matrix = Array2::zeros((n_samples, n_samples));
            for i in 0..n_samples {
                for j in i..n_samples {
                    let dist = self.euclidean_distance(&data.row(i), &data.row(j));
                    distance_matrix[[i, j]] = dist;
                    distance_matrix[[j, i]] = dist;
                }
            }

            // Classical MDS using eigendecomposition
            let squared_distances = distance_matrix.mapv(|d| d * d);

            // Double centering
            let _n = n_samples as f64;
            let row_means = squared_distances.mean_axis(ndarray::Axis(1)).unwrap();
            let col_means = squared_distances.mean_axis(ndarray::Axis(0)).unwrap();
            let grand_mean = squared_distances.mean().unwrap();

            let mut b_matrix = Array2::zeros((n_samples, n_samples));
            for i in 0..n_samples {
                for j in 0..n_samples {
                    b_matrix[[i, j]] = -0.5
                        * (squared_distances[[i, j]] - row_means[i] - col_means[j] + grand_mean);
                }
            }

            // Eigendecomposition of B matrix
            let (eigenvalues, eigenvectors) = self.compute_eigendecomposition(&b_matrix)?;

            // Sort eigenvalues in descending order
            let mut eigenvalue_indices: Vec<usize> = (0..eigenvalues.len()).collect();
            eigenvalue_indices
                .sort_by(|&i, &j| eigenvalues[j].partial_cmp(&eigenvalues[i]).unwrap());

            // Construct embedding
            let n_components = self.target_dimensions.min(n_samples);
            let mut embedding = Array2::zeros((n_samples, n_components));

            for (i, &idx) in eigenvalue_indices.iter().take(n_components).enumerate() {
                if eigenvalues[idx] > 0.0 {
                    let scale = eigenvalues[idx].sqrt();
                    for j in 0..n_samples {
                        embedding[[j, i]] = eigenvectors[[j, idx]] * scale;
                    }
                }
            }

            Ok(embedding)
        }

        /// Apply clustering to reduced data
        fn apply_clustering(&self, data: &Array2<f64>) -> Result<Option<Vec<usize>>> {
            match self.clustering_method {
                ClusteringMethod::KMeans { k } => Ok(Some(self.kmeans_clustering(data, k)?)),
                ClusteringMethod::DBSCAN { eps, min_samples } => {
                    Ok(Some(self.dbscan_clustering(data, eps, min_samples)?))
                }
                ClusteringMethod::Hierarchical { n_clusters } => {
                    Ok(Some(self.hierarchical_clustering(data, n_clusters)?))
                }
                ClusteringMethod::None => Ok(None),
            }
        }

        /// K-means clustering implementation
        fn kmeans_clustering(&self, data: &Array2<f64>, k: usize) -> Result<Vec<usize>> {
            use rand::Rng;
            let mut rng = rand::rng();
            let (n_samples, n_features) = data.dim();

            // Initialize centroids randomly
            let mut centroids = Array2::zeros((k, n_features));
            for i in 0..k {
                for j in 0..n_features {
                    let min_val = data.column(j).iter().fold(f64::INFINITY, |a, &b| a.min(b));
                    let max_val = data
                        .column(j)
                        .iter()
                        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                    centroids[[i, j]] = min_val + rng.random::<f64>() * (max_val - min_val);
                }
            }

            let mut labels = vec![0; n_samples];
            let max_iterations = 100;

            for _iteration in 0..max_iterations {
                let mut changed = false;

                // Assign points to nearest centroids
                for i in 0..n_samples {
                    let mut min_distance = f64::INFINITY;
                    let mut best_cluster = 0;

                    for j in 0..k {
                        let distance = self.euclidean_distance(&data.row(i), &centroids.row(j));
                        if distance < min_distance {
                            min_distance = distance;
                            best_cluster = j;
                        }
                    }

                    if labels[i] != best_cluster {
                        labels[i] = best_cluster;
                        changed = true;
                    }
                }

                if !changed {
                    break;
                }

                // Update centroids
                let mut cluster_counts = vec![0; k];
                centroids.fill(0.0);

                for i in 0..n_samples {
                    let cluster = labels[i];
                    cluster_counts[cluster] += 1;
                    for j in 0..n_features {
                        centroids[[cluster, j]] += data[[i, j]];
                    }
                }

                for i in 0..k {
                    if cluster_counts[i] > 0 {
                        for j in 0..n_features {
                            centroids[[i, j]] /= cluster_counts[i] as f64;
                        }
                    }
                }
            }

            Ok(labels)
        }

        /// DBSCAN clustering implementation
        fn dbscan_clustering(
            &self,
            data: &Array2<f64>,
            eps: f64,
            min_samples: usize,
        ) -> Result<Vec<usize>> {
            let n_samples = data.nrows();
            let mut labels = vec![usize::MAX; n_samples]; // MAX means unclassified
            let mut cluster_id = 0;

            for i in 0..n_samples {
                if labels[i] != usize::MAX {
                    continue; // Already processed
                }

                let neighbors = self.find_neighbors(data, i, eps);

                if neighbors.len() < min_samples {
                    labels[i] = usize::MAX - 1; // Mark as noise
                } else {
                    self.expand_cluster(
                        data,
                        i,
                        &neighbors,
                        cluster_id,
                        eps,
                        min_samples,
                        &mut labels,
                    );
                    cluster_id += 1;
                }
            }

            // Convert noise points to cluster 0 and increment others
            for label in &mut labels {
                if *label == usize::MAX - 1 {
                    *label = 0; // Noise cluster
                } else if *label != usize::MAX {
                    *label += 1; // Shift cluster IDs
                }
            }

            Ok(labels)
        }

        /// Hierarchical clustering implementation
        fn hierarchical_clustering(
            &self,
            data: &Array2<f64>,
            n_clusters: usize,
        ) -> Result<Vec<usize>> {
            let n_samples = data.nrows();

            // Initialize each point as its own cluster
            let mut clusters: Vec<Vec<usize>> = (0..n_samples).map(|i| vec![i]).collect();

            // Compute initial distance matrix
            let mut distance_matrix = Array2::zeros((n_samples, n_samples));
            for i in 0..n_samples {
                for j in i + 1..n_samples {
                    let dist = self.euclidean_distance(&data.row(i), &data.row(j));
                    distance_matrix[[i, j]] = dist;
                    distance_matrix[[j, i]] = dist;
                }
            }

            // Agglomerative clustering
            while clusters.len() > n_clusters {
                let mut min_distance = f64::INFINITY;
                let mut merge_i = 0;
                let mut merge_j = 1;

                // Find closest clusters
                for i in 0..clusters.len() {
                    for j in i + 1..clusters.len() {
                        let dist =
                            self.cluster_distance(&clusters[i], &clusters[j], &distance_matrix);
                        if dist < min_distance {
                            min_distance = dist;
                            merge_i = i;
                            merge_j = j;
                        }
                    }
                }

                // Merge clusters
                let mut merged_cluster = clusters[merge_i].clone();
                merged_cluster.extend(&clusters[merge_j]);

                // Remove old clusters and add merged cluster
                clusters.remove(merge_j); // Remove j first (higher index)
                clusters.remove(merge_i);
                clusters.push(merged_cluster);
            }

            // Assign labels
            let mut labels = vec![0; n_samples];
            for (cluster_id, cluster) in clusters.iter().enumerate() {
                for &point_id in cluster {
                    labels[point_id] = cluster_id;
                }
            }

            Ok(labels)
        }

        /// Helper functions
        fn euclidean_distance(
            &self,
            a: &ndarray::ArrayView1<f64>,
            b: &ndarray::ArrayView1<f64>,
        ) -> f64 {
            a.iter()
                .zip(b.iter())
                .map(|(&x, &y)| (x - y).powi(2))
                .sum::<f64>()
                .sqrt()
        }

        fn compute_eigendecomposition(
            &self,
            matrix: &Array2<f64>,
        ) -> Result<(Array1<f64>, Array2<f64>)> {
            let n = matrix.nrows();

            // For small matrices, use power iteration for dominant eigenvalues
            if n <= 10 {
                let mut eigenvalues = Array1::zeros(n);
                let mut eigenvectors = Array2::zeros((n, n));

                let mut remaining_matrix = matrix.clone();

                for k in 0..n {
                    // Power iteration for largest eigenvalue
                    let mut v = Array1::from_elem(n, 1.0 / (n as f64).sqrt());
                    let mut eigenvalue = 0.0;

                    for _ in 0..100 {
                        let v_new = remaining_matrix.dot(&v);
                        eigenvalue = v.dot(&v_new);
                        let norm = v_new.iter().map(|&x| x * x).sum::<f64>().sqrt();
                        if norm > 1e-10 {
                            v = v_new / norm;
                        }
                    }

                    eigenvalues[k] = eigenvalue;
                    eigenvectors.column_mut(k).assign(&v);

                    // Deflate matrix
                    let v_col = v.view().insert_axis(ndarray::Axis(1));
                    let v_row = v.view().insert_axis(ndarray::Axis(0));
                    let vv = v_col.dot(&v_row);
                    remaining_matrix = &remaining_matrix - eigenvalue * &vv;
                }

                Ok((eigenvalues, eigenvectors))
            } else {
                // For larger matrices, use a more efficient approach
                // We'll compute the dominant eigenvalues using power iteration
                self.eigendecomposition_large_matrix(matrix)
            }
        }

        /// Eigendecomposition for large matrices using iterative methods
        fn eigendecomposition_large_matrix(
            &self,
            matrix: &Array2<f64>,
        ) -> Result<(Array1<f64>, Array2<f64>)> {
            let n = matrix.nrows();
            let num_eigenvalues = std::cmp::min(n, 20); // Compute up to 20 eigenvalues
            
            let mut eigenvalues = Array1::zeros(num_eigenvalues);
            let mut eigenvectors = Array2::zeros((n, num_eigenvalues));
            let mut remaining_matrix = matrix.clone();
            
            for k in 0..num_eigenvalues {
                // Power iteration for the k-th eigenvalue
                let mut v = Array1::from_vec(
                    (0..n).map(|i| (i + k + 1) as f64).collect() // Simple initialization
                );
                
                // Normalize initial vector
                let norm = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
                if norm > 1e-15 {
                    v.mapv_inplace(|x| x / norm);
                }
                
                let mut eigenvalue = 0.0;
                
                // Power iteration
                for _ in 0..1000 {
                    let v_new = remaining_matrix.dot(&v);
                    let v_new_norm = v_new.iter().map(|&x| x * x).sum::<f64>().sqrt();
                    
                    if v_new_norm < 1e-15 {
                        break;
                    }
                    
                    eigenvalue = v.dot(&v_new);
                    v.assign(&v_new);
                    v.mapv_inplace(|x| x / v_new_norm);
                    
                    // Check convergence
                    let residual = &remaining_matrix.dot(&v) - eigenvalue * &v;
                    let residual_norm = residual.iter().map(|&x| x * x).sum::<f64>().sqrt();
                    
                    if residual_norm < 1e-10 {
                        break;
                    }
                }
                
                eigenvalues[k] = eigenvalue;
                eigenvectors.column_mut(k).assign(&v);
                
                // Deflate the matrix: A' = A - λ * v * v^T
                if eigenvalue.abs() > 1e-15 {
                    for i in 0..n {
                        for j in 0..n {
                            remaining_matrix[[i, j]] -= eigenvalue * v[i] * v[j];
                        }
                    }
                }
                
                // Check if remaining eigenvalues are small
                let remaining_norm = remaining_matrix.iter().map(|&x| x * x).sum::<f64>().sqrt();
                if remaining_norm < 1e-12 {
                    break;
                }
            }
            
            // Extend to full size matrices with zeros for unused eigenvalues/vectors
            let mut full_eigenvalues = Array1::zeros(n);
            let mut full_eigenvectors = Array2::zeros((n, n));
            
            for i in 0..num_eigenvalues {
                full_eigenvalues[i] = eigenvalues[i];
                full_eigenvectors.column_mut(i).assign(&eigenvectors.column(i));
            }
            
            // Fill remaining eigenvectors with orthogonal vectors
            for i in num_eigenvalues..n {
                let mut v = Array1::zeros(n);
                v[i] = 1.0; // Standard basis vector
                full_eigenvectors.column_mut(i).assign(&v);
            }
            
            Ok((full_eigenvalues, full_eigenvectors))
        }

        fn compute_gaussian_similarities(
            &self,
            data: &Array2<f64>,
            sigma: f64,
        ) -> Result<Array2<f64>> {
            let n = data.nrows();
            let mut similarities = Array2::zeros((n, n));

            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        let dist_sq = self.euclidean_distance(&data.row(i), &data.row(j)).powi(2);
                        similarities[[i, j]] = (-dist_sq / (2.0 * sigma.powi(2))).exp();
                    }
                }
            }

            // Normalize
            for i in 0..n {
                let row_sum: f64 = similarities.row(i).sum();
                if row_sum > 0.0 {
                    similarities.row_mut(i).mapv_inplace(|x| x / row_sum);
                }
            }

            Ok(similarities)
        }

        fn compute_t_similarities(&self, embedding: &Array2<f64>) -> Result<Array2<f64>> {
            let n = embedding.nrows();
            let mut similarities = Array2::zeros((n, n));

            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        let dist_sq = self
                            .euclidean_distance(&embedding.row(i), &embedding.row(j))
                            .powi(2);
                        similarities[[i, j]] = 1.0 / (1.0 + dist_sq);
                    }
                }
            }

            // Normalize
            let total_sum: f64 = similarities.sum();
            if total_sum > 0.0 {
                similarities.mapv_inplace(|x| x / total_sum);
            }

            Ok(similarities)
        }

        fn find_neighbors(&self, data: &Array2<f64>, point: usize, eps: f64) -> Vec<usize> {
            let mut neighbors = Vec::new();
            for i in 0..data.nrows() {
                if i != point {
                    let dist = self.euclidean_distance(&data.row(point), &data.row(i));
                    if dist <= eps {
                        neighbors.push(i);
                    }
                }
            }
            neighbors
        }

        fn expand_cluster(
            &self,
            data: &Array2<f64>,
            point: usize,
            neighbors: &[usize],
            cluster_id: usize,
            eps: f64,
            min_samples: usize,
            labels: &mut [usize],
        ) {
            labels[point] = cluster_id;
            let mut seed_set = neighbors.to_vec();
            let mut i = 0;

            while i < seed_set.len() {
                let q = seed_set[i];

                if labels[q] == usize::MAX - 1 {
                    // Change noise to border point
                    labels[q] = cluster_id;
                } else if labels[q] == usize::MAX {
                    // Unclassified
                    labels[q] = cluster_id;
                    let q_neighbors = self.find_neighbors(data, q, eps);

                    if q_neighbors.len() >= min_samples {
                        for &neighbor in &q_neighbors {
                            if !seed_set.contains(&neighbor) {
                                seed_set.push(neighbor);
                            }
                        }
                    }
                }

                i += 1;
            }
        }

        fn cluster_distance(
            &self,
            cluster1: &[usize],
            cluster2: &[usize],
            distance_matrix: &Array2<f64>,
        ) -> f64 {
            // Single linkage (minimum distance)
            let mut min_distance = f64::INFINITY;

            for &i in cluster1 {
                for &j in cluster2 {
                    let dist = distance_matrix[[i, j]];
                    if dist < min_distance {
                        min_distance = dist;
                    }
                }
            }

            min_distance
        }
    }

    /// High-dimensional data visualization plot
    #[derive(Debug, Clone)]
    pub struct HighDimensionalPlot {
        /// X coordinates (first component)
        pub x: Vec<f64>,
        /// Y coordinates (second component)
        pub y: Vec<f64>,
        /// Z coordinates (third component, if 3D)
        pub z: Option<Vec<f64>>,
        /// Colors for points
        pub colors: Vec<f64>,
        /// Cluster labels
        pub cluster_labels: Option<Vec<usize>>,
        /// Original data dimensions
        pub original_dimensions: usize,
        /// Reduced dimensions
        pub reduced_dimensions: usize,
        /// Reduction method used
        pub reduction_method: DimensionReductionMethod,
        /// Plot metadata
        pub metadata: PlotMetadata,
    }

    /// Animated visualization for time-series or dynamic systems
    pub struct AnimatedVisualizer {
        /// Frame data
        pub frames: Vec<PhaseSpacePlot>,
        /// Animation settings
        pub animation_settings: AnimationSettings,
        /// Current frame index
        pub current_frame: usize,
    }

    /// Animation settings
    #[derive(Debug, Clone)]
    pub struct AnimationSettings {
        /// Frames per second
        pub fps: f64,
        /// Loop animation
        pub loop_animation: bool,
        /// Frame interpolation
        pub interpolate_frames: bool,
        /// Fade trail length
        pub trail_length: usize,
    }

    impl Default for AnimationSettings {
        fn default() -> Self {
            Self {
                fps: 30.0,
                loop_animation: true,
                interpolate_frames: false,
                trail_length: 50,
            }
        }
    }

    impl AnimatedVisualizer {
        /// Create new animated visualizer
        pub fn new() -> Self {
            Self {
                frames: Vec::new(),
                animation_settings: AnimationSettings::default(),
                current_frame: 0,
            }
        }

        /// Create animation from ODE solution
        pub fn create_animation_from_ode<F: crate::common::IntegrateFloat>(
            &mut self,
            ode_result: &ODEResult<F>,
            x_index: usize,
            y_index: usize,
            frames_per_time_unit: usize,
        ) -> Result<()> {
            let n_points = ode_result.t.len();
            let n_vars = if !ode_result.y.is_empty() {
                ode_result.y[0].len()
            } else {
                0
            };

            if x_index >= n_vars || y_index >= n_vars {
                return Err(crate::error::IntegrateError::ValueError(
                    "Variable index out of bounds".to_string(),
                ));
            }

            // Calculate frame indices
            let total_frames = n_points * frames_per_time_unit;

            for frame_idx in 0..total_frames {
                let time_idx = frame_idx / frames_per_time_unit;
                let sub_frame = frame_idx % frames_per_time_unit;

                // Determine how many points to include in this frame
                let end_point = time_idx + 1;

                let mut x_data = Vec::new();
                let mut y_data = Vec::new();
                let mut colors = Vec::new();

                // Add trajectory up to current time
                for i in 0..end_point.min(n_points) {
                    x_data.push(ode_result.y[i][x_index].to_f64().unwrap_or(0.0));
                    y_data.push(ode_result.y[i][y_index].to_f64().unwrap_or(0.0));
                    colors.push(i as f64);
                }

                // Add interpolated current point if within time step
                if sub_frame > 0 && time_idx < n_points - 1 {
                    let alpha = sub_frame as f64 / frames_per_time_unit as f64;
                    let x_interp = (1.0 - alpha)
                        * ode_result.y[time_idx][x_index].to_f64().unwrap_or(0.0)
                        + alpha * ode_result.y[time_idx + 1][x_index].to_f64().unwrap_or(0.0);
                    let y_interp = (1.0 - alpha)
                        * ode_result.y[time_idx][y_index].to_f64().unwrap_or(0.0)
                        + alpha * ode_result.y[time_idx + 1][y_index].to_f64().unwrap_or(0.0);

                    x_data.push(x_interp);
                    y_data.push(y_interp);
                    colors.push(time_idx as f64 + alpha);
                }

                let mut metadata = PlotMetadata::default();
                metadata.title = format!("Trajectory Animation - Frame {}", frame_idx);
                metadata.xlabel = format!("Variable {}", x_index);
                metadata.ylabel = format!("Variable {}", y_index);

                let frame = PhaseSpacePlot {
                    x: x_data,
                    y: y_data,
                    colors: Some(colors),
                    metadata,
                };

                self.frames.push(frame);
            }

            Ok(())
        }

        /// Get current frame
        pub fn get_current_frame(&self) -> Option<&PhaseSpacePlot> {
            self.frames.get(self.current_frame)
        }

        /// Advance to next frame
        pub fn next_frame(&mut self) {
            if !self.frames.is_empty() {
                self.current_frame = (self.current_frame + 1) % self.frames.len();
            }
        }

        /// Go to previous frame
        pub fn previous_frame(&mut self) {
            if !self.frames.is_empty() {
                self.current_frame = if self.current_frame == 0 {
                    self.frames.len() - 1
                } else {
                    self.current_frame - 1
                };
            }
        }

        /// Set specific frame
        pub fn set_frame(&mut self, frame_index: usize) {
            if frame_index < self.frames.len() {
                self.current_frame = frame_index;
            }
        }

        /// Export animation frames to vector
        pub fn export_frames(&self) -> &[PhaseSpacePlot] {
            &self.frames
        }
    }

    /// Statistical plotting utilities
    pub struct StatisticalPlotter {
        /// Confidence level for error bars
        pub confidence_level: f64,
        /// Bootstrap samples for uncertainty estimation
        pub bootstrap_samples: usize,
    }

    impl Default for StatisticalPlotter {
        fn default() -> Self {
            Self {
                confidence_level: 0.95,
                bootstrap_samples: 1000,
            }
        }
    }

    impl StatisticalPlotter {
        /// Create box plot data
        pub fn create_box_plot(
            &self,
            data_groups: &[Vec<f64>],
            group_names: &[String],
        ) -> Result<BoxPlot> {
            let mut box_data = Vec::new();

            for data in data_groups {
                let box_stats = self.calculate_box_statistics(data)?;
                box_data.push(box_stats);
            }

            let mut metadata = PlotMetadata::default();
            metadata.title = "Box Plot".to_string();
            metadata.xlabel = "Groups".to_string();
            metadata.ylabel = "Values".to_string();

            Ok(BoxPlot {
                box_data,
                group_names: group_names.to_vec(),
                metadata,
            })
        }

        /// Create histogram with density estimation
        pub fn create_histogram_with_density(
            &self,
            data: &[f64],
            n_bins: usize,
        ) -> Result<HistogramPlot> {
            let min_val = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_val = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let bin_width = (max_val - min_val) / n_bins as f64;

            let mut histogram = vec![0; n_bins];
            let mut bin_edges = Vec::with_capacity(n_bins + 1);

            for i in 0..=n_bins {
                bin_edges.push(min_val + i as f64 * bin_width);
            }

            // Count frequencies
            for &value in data {
                let bin = ((value - min_val) / bin_width) as usize;
                let bin_index = bin.min(n_bins - 1);
                histogram[bin_index] += 1;
            }

            // Normalize to density
            let total_count = data.len() as f64;
            let density: Vec<f64> = histogram
                .iter()
                .map(|&count| count as f64 / (total_count * bin_width))
                .collect();

            // Kernel density estimation
            let kde_points = Array1::linspace(min_val, max_val, 200);
            let kde_values = self.kernel_density_estimation(data, &kde_points)?;

            let mut metadata = PlotMetadata::default();
            metadata.title = "Histogram with Density Estimation".to_string();
            metadata.xlabel = "Values".to_string();
            metadata.ylabel = "Density".to_string();

            Ok(HistogramPlot {
                bin_edges,
                frequencies: histogram.iter().map(|&x| x as f64).collect(),
                density,
                kde_points: kde_points.to_vec(),
                kde_values,
                metadata,
            })
        }

        /// Create Q-Q plot for normality testing
        pub fn create_qq_plot(&self, data: &[f64]) -> Result<QQPlot> {
            let mut sorted_data = data.to_vec();
            sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let n = sorted_data.len();
            let mut theoretical_quantiles = Vec::with_capacity(n);
            let mut sample_quantiles = Vec::with_capacity(n);

            for i in 0..n {
                let p = (i as f64 + 0.5) / n as f64;
                let theoretical_q = self.inverse_normal_cdf(p);

                theoretical_quantiles.push(theoretical_q);
                sample_quantiles.push(sorted_data[i]);
            }

            let mut metadata = PlotMetadata::default();
            metadata.title = "Q-Q Plot (Normal)".to_string();
            metadata.xlabel = "Theoretical Quantiles".to_string();
            metadata.ylabel = "Sample Quantiles".to_string();

            Ok(QQPlot {
                theoretical_quantiles,
                sample_quantiles,
                metadata,
            })
        }

        /// Calculate bootstrap confidence intervals
        pub fn bootstrap_confidence_interval(
            &self,
            data: &[f64],
            statistic: fn(&[f64]) -> f64,
        ) -> Result<(f64, f64)> {
            use rand::Rng;
            let mut rng = rand::rng();
            let n = data.len();
            let mut bootstrap_stats = Vec::with_capacity(self.bootstrap_samples);

            for _ in 0..self.bootstrap_samples {
                let mut bootstrap_sample = Vec::with_capacity(n);
                for _ in 0..n {
                    let idx = rng.random_range(0..n);
                    bootstrap_sample.push(data[idx]);
                }

                let stat = statistic(&bootstrap_sample);
                bootstrap_stats.push(stat);
            }

            bootstrap_stats.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let alpha = 1.0 - self.confidence_level;
            let lower_idx = (alpha / 2.0 * self.bootstrap_samples as f64) as usize;
            let upper_idx = ((1.0 - alpha / 2.0) * self.bootstrap_samples as f64) as usize;

            let lower_bound = bootstrap_stats[lower_idx];
            let upper_bound = bootstrap_stats[upper_idx.min(self.bootstrap_samples - 1)];

            Ok((lower_bound, upper_bound))
        }

        /// Helper functions
        fn calculate_box_statistics(&self, data: &[f64]) -> Result<BoxStatistics> {
            let mut sorted_data = data.to_vec();
            sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let n = sorted_data.len();
            if n == 0 {
                return Err(crate::error::IntegrateError::ValueError(
                    "Cannot calculate statistics for empty data".to_string(),
                ));
            }

            let q1 = self.percentile(&sorted_data, 25.0);
            let median = self.percentile(&sorted_data, 50.0);
            let q3 = self.percentile(&sorted_data, 75.0);

            let iqr = q3 - q1;
            let lower_fence = q1 - 1.5 * iqr;
            let upper_fence = q3 + 1.5 * iqr;

            let whisker_low = sorted_data
                .iter()
                .find(|&&x| x >= lower_fence)
                .copied()
                .unwrap_or(sorted_data[0]);
            let whisker_high = sorted_data
                .iter()
                .rev()
                .find(|&&x| x <= upper_fence)
                .copied()
                .unwrap_or(sorted_data[n - 1]);

            let outliers: Vec<f64> = sorted_data
                .iter()
                .filter(|&&x| x < lower_fence || x > upper_fence)
                .copied()
                .collect();

            Ok(BoxStatistics {
                min: sorted_data[0],
                q1,
                median,
                q3,
                max: sorted_data[n - 1],
                whisker_low,
                whisker_high,
                outliers,
            })
        }

        fn percentile(&self, sorted_data: &[f64], p: f64) -> f64 {
            let n = sorted_data.len();
            let index = (p / 100.0) * (n - 1) as f64;
            let lower = index.floor() as usize;
            let upper = index.ceil() as usize;
            let weight = index - lower as f64;

            if upper >= n {
                sorted_data[n - 1]
            } else if lower == upper {
                sorted_data[lower]
            } else {
                sorted_data[lower] * (1.0 - weight) + sorted_data[upper] * weight
            }
        }

        fn kernel_density_estimation(
            &self,
            data: &[f64],
            points: &Array1<f64>,
        ) -> Result<Vec<f64>> {
            let n = data.len() as f64;
            let bandwidth = self.silverman_bandwidth(data);

            let mut kde_values = Vec::with_capacity(points.len());

            for &x in points {
                let mut density = 0.0;
                for &xi in data {
                    let u = (x - xi) / bandwidth;
                    density += self.gaussian_kernel(u);
                }
                kde_values.push(density / (n * bandwidth));
            }

            Ok(kde_values)
        }

        fn silverman_bandwidth(&self, data: &[f64]) -> f64 {
            let n = data.len() as f64;
            let std_dev = self.standard_deviation(data);
            1.06 * std_dev * n.powf(-1.0 / 5.0)
        }

        fn standard_deviation(&self, data: &[f64]) -> f64 {
            let mean = data.iter().sum::<f64>() / data.len() as f64;
            let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
            variance.sqrt()
        }

        fn gaussian_kernel(&self, u: f64) -> f64 {
            (2.0 * std::f64::consts::PI).sqrt().recip() * (-0.5 * u * u).exp()
        }

        fn inverse_normal_cdf(&self, p: f64) -> f64 {
            // Simplified inverse normal CDF approximation
            if p <= 0.0 {
                return f64::NEG_INFINITY;
            }
            if p >= 1.0 {
                return f64::INFINITY;
            }

            // Use Box-Muller transformation for approximation
            if p == 0.5 {
                return 0.0;
            }

            // Simple rational approximation
            let t = if p < 0.5 {
                (-2.0 * p.ln()).sqrt()
            } else {
                (-2.0 * (1.0 - p).ln()).sqrt()
            };

            let result = t
                - (2.515517 + 0.802853 * t + 0.010328 * t * t)
                    / (1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t);

            if p < 0.5 {
                -result
            } else {
                result
            }
        }
    }

    /// Box plot data structure
    #[derive(Debug, Clone)]
    pub struct BoxPlot {
        /// Box statistics for each group
        pub box_data: Vec<BoxStatistics>,
        /// Group names
        pub group_names: Vec<String>,
        /// Plot metadata
        pub metadata: PlotMetadata,
    }

    /// Box statistics
    #[derive(Debug, Clone)]
    pub struct BoxStatistics {
        pub min: f64,
        pub q1: f64,
        pub median: f64,
        pub q3: f64,
        pub max: f64,
        pub whisker_low: f64,
        pub whisker_high: f64,
        pub outliers: Vec<f64>,
    }

    /// Histogram plot with density estimation
    #[derive(Debug, Clone)]
    pub struct HistogramPlot {
        /// Bin edges
        pub bin_edges: Vec<f64>,
        /// Frequencies in each bin
        pub frequencies: Vec<f64>,
        /// Density values
        pub density: Vec<f64>,
        /// KDE evaluation points
        pub kde_points: Vec<f64>,
        /// KDE values
        pub kde_values: Vec<f64>,
        /// Plot metadata
        pub metadata: PlotMetadata,
    }

    /// Q-Q plot for distribution testing
    #[derive(Debug, Clone)]
    pub struct QQPlot {
        /// Theoretical quantiles
        pub theoretical_quantiles: Vec<f64>,
        /// Sample quantiles
        pub sample_quantiles: Vec<f64>,
        /// Plot metadata
        pub metadata: PlotMetadata,
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use approx::assert_relative_eq;

        #[test]
        fn test_multi_dimensional_visualizer() {
            // Create test data (3D -> 2D)
            let data =
                Array2::from_shape_vec((10, 3), (0..30).map(|x| x as f64).collect()).unwrap();

            let visualizer = MultiDimensionalVisualizer::new();
            let result = visualizer
                .visualize_high_dimensional_data(&data, None)
                .unwrap();

            assert_eq!(result.x.len(), 10);
            assert_eq!(result.y.len(), 10);
            assert_eq!(result.original_dimensions, 3);
            assert_eq!(result.reduced_dimensions, 2);
        }

        #[test]
        fn test_animated_visualizer() {
            let mut animator = AnimatedVisualizer::new();

            // Create mock ODE result
            let mut ode_result: ODEResult<f64> = ODEResult {
                t: vec![0.0, 0.1, 0.2, 0.3, 0.4],
                y: vec![
                    Array1::from(vec![0.0, 0.0]),
                    Array1::from(vec![0.1, 0.05]),
                    Array1::from(vec![0.2, 0.2]),
                    Array1::from(vec![0.3, 0.45]),
                    Array1::from(vec![0.4, 0.8]),
                ],
                success: true,
                message: Some("Success".to_string()),
                n_eval: 10,
                n_steps: 5,
                n_accepted: 5,
                n_rejected: 0,
                n_lu: 0,
                n_jac: 2,
                method: crate::ode::types::ODEMethod::RK45,
            };

            let result = animator.create_animation_from_ode(&ode_result, 0, 1, 2);
            assert!(result.is_ok());
            assert!(!animator.frames.is_empty());
        }

        #[test]
        fn test_statistical_plotter() {
            let plotter = StatisticalPlotter::default();

            // Test data
            let data1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
            let data2 = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];

            // Test box plot
            let box_plot = plotter
                .create_box_plot(
                    &[data1.clone(), data2.clone()],
                    &["Group 1".to_string(), "Group 2".to_string()],
                )
                .unwrap();

            assert_eq!(box_plot.box_data.len(), 2);
            assert_eq!(box_plot.group_names.len(), 2);

            // Test histogram
            let histogram = plotter.create_histogram_with_density(&data1, 5).unwrap();
            assert_eq!(histogram.bin_edges.len(), 6); // n_bins + 1
            assert_eq!(histogram.frequencies.len(), 5);

            // Test Q-Q plot
            let qq_plot = plotter.create_qq_plot(&data1).unwrap();
            assert_eq!(qq_plot.theoretical_quantiles.len(), data1.len());
            assert_eq!(qq_plot.sample_quantiles.len(), data1.len());
        }

        #[test]
        fn test_kmeans_clustering() {
            let visualizer = MultiDimensionalVisualizer {
                reduction_method: DimensionReductionMethod::PCA,
                target_dimensions: 2,
                clustering_method: ClusteringMethod::KMeans { k: 2 },
            };

            // Create two distinct clusters
            let mut data_vec = Vec::new();
            // Cluster 1: around (0, 0)
            for _ in 0..5 {
                data_vec.extend_from_slice(&[0.0, 0.0]);
            }
            // Cluster 2: around (5, 5)
            for _ in 0..5 {
                data_vec.extend_from_slice(&[5.0, 5.0]);
            }

            let data = Array2::from_shape_vec((10, 2), data_vec).unwrap();
            let labels = visualizer.kmeans_clustering(&data, 2).unwrap();

            assert_eq!(labels.len(), 10);
            // Check that we get two distinct clusters
            let unique_labels: std::collections::HashSet<_> = labels.iter().collect();
            assert!(unique_labels.len() <= 2);
        }
    }
}

/// Advanced visualization capabilities for specialized solvers
pub mod specialized_visualizations {
    use super::*;
    use crate::specialized::quantum::QuantumState;

    /// Quantum state visualization tools
    pub struct QuantumVisualizer;

    impl QuantumVisualizer {
        /// Create wave function visualization
        pub fn visualize_wavefunction(state: &QuantumState) -> Result<HeatMapPlot> {
            let probability_density = state.probability_density();

            let mut metadata = PlotMetadata::default();
            metadata.title = format!("Quantum State at t = {:.3}", state.t);
            metadata.xlabel = "Position".to_string();
            metadata.ylabel = "Probability Density".to_string();

            Ok(HeatMapPlot {
                x: state.x.clone(),
                y: Array1::from_elem(1, 0.0), // 1D visualization
                z: Array2::from_shape_vec(
                    (1, probability_density.len()),
                    probability_density.to_vec(),
                )
                .map_err(|e| IntegrateError::ComputationError(format!("Shape error: {}", e)))?,
                metadata,
            })
        }

        /// Create complex phase visualization
        pub fn visualize_complex_phase(state: &QuantumState) -> Result<PhaseSpacePlot> {
            let real_parts: Vec<f64> = state.psi.iter().map(|c| c.re).collect();
            let imag_parts: Vec<f64> = state.psi.iter().map(|c| c.im).collect();
            let phases: Vec<f64> = state.psi.iter().map(|c| c.arg()).collect();

            let mut metadata = PlotMetadata::default();
            metadata.title = "Complex Wave Function Phase".to_string();
            metadata.xlabel = "Real Part".to_string();
            metadata.ylabel = "Imaginary Part".to_string();

            Ok(PhaseSpacePlot {
                x: real_parts,
                y: imag_parts,
                colors: Some(phases),
                metadata,
            })
        }

        /// Create expectation value evolution plot
        pub fn visualize_expectation_evolution(states: &[QuantumState]) -> Result<PhaseSpacePlot> {
            let times: Vec<f64> = states.iter().map(|s| s.t).collect();
            let positions: Vec<f64> = states.iter().map(|s| s.expectation_position()).collect();
            let momenta: Vec<f64> = states.iter().map(|s| s.expectation_momentum()).collect();

            let mut metadata = PlotMetadata::default();
            metadata.title = "Quantum Expectation Values Evolution".to_string();
            metadata.xlabel = "Position Expectation".to_string();
            metadata.ylabel = "Momentum Expectation".to_string();

            Ok(PhaseSpacePlot {
                x: positions,
                y: momenta,
                colors: Some(times),
                metadata,
            })
        }

        /// Create energy level diagram
        pub fn visualize_energy_levels(
            energies: &Array1<f64>,
            wavefunctions: &Array2<f64>,
        ) -> Result<VectorFieldPlot> {
            let n_levels = energies.len().min(5); // Show up to 5 levels
            let n_points = wavefunctions.nrows();

            let x_coords = Array1::linspace(-1.0, 1.0, n_points);
            let mut x_grid = Array2::zeros((n_levels, n_points));
            let mut y_grid = Array2::zeros((n_levels, n_points));
            let mut u = Array2::zeros((n_levels, n_points));
            let mut v = Array2::zeros((n_levels, n_points));
            let mut magnitude = Array2::zeros((n_levels, n_points));

            for level in 0..n_levels {
                for i in 0..n_points {
                    x_grid[[level, i]] = x_coords[i];
                    y_grid[[level, i]] = energies[level];
                    u[[level, i]] = wavefunctions[[i, level]];
                    v[[level, i]] = 0.0; // No y-component for energy levels
                    magnitude[[level, i]] = wavefunctions[[i, level]].abs();
                }
            }

            let mut metadata = PlotMetadata::default();
            metadata.title = "Energy Level Diagram".to_string();
            metadata.xlabel = "Position".to_string();
            metadata.ylabel = "Energy".to_string();

            Ok(VectorFieldPlot {
                x_grid,
                y_grid,
                u,
                v,
                magnitude,
                metadata,
            })
        }
    }

    /// Fluid state for 2D visualization
    #[derive(Debug, Clone)]
    pub struct FluidState {
        /// Velocity components [u, v] as 2D arrays
        pub velocity: Vec<Array2<f64>>,
        /// Pressure field
        pub pressure: Array2<f64>,
        /// Temperature field (optional)
        pub temperature: Option<Array2<f64>>,
        /// Current time
        pub time: f64,
        /// Grid spacing in x-direction
        pub dx: f64,
        /// Grid spacing in y-direction
        pub dy: f64,
    }

    /// Fluid state for 3D visualization
    #[derive(Debug, Clone)]
    pub struct FluidState3D {
        /// Velocity components [u, v, w] as 3D arrays
        pub velocity: Vec<Array3<f64>>,
        /// Pressure field
        pub pressure: Array3<f64>,
        /// Temperature field (optional)
        pub temperature: Option<Array3<f64>>,
        /// Current time
        pub time: f64,
        /// Grid spacing in x-direction
        pub dx: f64,
        /// Grid spacing in y-direction
        pub dy: f64,
        /// Grid spacing in z-direction
        pub dz: f64,
    }

    /// Fluid dynamics visualization tools
    pub struct FluidVisualizer;

    impl FluidVisualizer {
        /// Create velocity field visualization
        pub fn visualize_velocity_field(state: &FluidState) -> Result<VectorFieldPlot> {
            if state.velocity.len() < 2 {
                return Err(IntegrateError::ValueError(
                    "Need at least 2 velocity components".to_string(),
                ));
            }

            let u = &state.velocity[0];
            let v = &state.velocity[1];
            let (ny, nx) = u.dim();

            let mut x_grid = Array2::zeros((ny, nx));
            let mut y_grid = Array2::zeros((ny, nx));
            let mut magnitude = Array2::zeros((ny, nx));

            for i in 0..ny {
                for j in 0..nx {
                    x_grid[[i, j]] = j as f64 * state.dx;
                    y_grid[[i, j]] = i as f64 * state.dy;
                    magnitude[[i, j]] = (u[[i, j]].powi(2) + v[[i, j]].powi(2)).sqrt();
                }
            }

            let mut metadata = PlotMetadata::default();
            metadata.title = format!("Velocity Field at t = {:.3}", state.time);
            metadata.xlabel = "X Position".to_string();
            metadata.ylabel = "Y Position".to_string();

            Ok(VectorFieldPlot {
                x_grid,
                y_grid,
                u: u.clone(),
                v: v.clone(),
                magnitude,
                metadata,
            })
        }

        /// Create pressure field heatmap
        pub fn visualize_pressure_field(state: &FluidState) -> Result<HeatMapPlot> {
            let (ny, nx) = state.pressure.dim();
            let x = Array1::from_iter((0..nx).map(|i| i as f64 * state.dx));
            let y = Array1::from_iter((0..ny).map(|i| i as f64 * state.dy));

            let mut metadata = PlotMetadata::default();
            metadata.title = format!("Pressure Field at t = {:.3}", state.time);
            metadata.xlabel = "X Position".to_string();
            metadata.ylabel = "Y Position".to_string();

            Ok(HeatMapPlot {
                x,
                y,
                z: state.pressure.clone(),
                metadata,
            })
        }

        /// Create vorticity visualization
        pub fn visualize_vorticity(state: &FluidState) -> Result<HeatMapPlot> {
            if state.velocity.len() < 2 {
                return Err(IntegrateError::ValueError(
                    "Need at least 2 velocity components".to_string(),
                ));
            }

            let u = &state.velocity[0];
            let v = &state.velocity[1];
            let (ny, nx) = u.dim();

            let mut vorticity = Array2::zeros((ny, nx));

            // Compute vorticity using finite differences
            for i in 1..ny - 1 {
                for j in 1..nx - 1 {
                    let dvdx = (v[[i, j + 1]] - v[[i, j - 1]]) / (2.0 * state.dx);
                    let dudy = (u[[i + 1, j]] - u[[i - 1, j]]) / (2.0 * state.dy);
                    vorticity[[i, j]] = dvdx - dudy;
                }
            }

            let x = Array1::from_iter((0..nx).map(|i| i as f64 * state.dx));
            let y = Array1::from_iter((0..ny).map(|i| i as f64 * state.dy));

            let mut metadata = PlotMetadata::default();
            metadata.title = format!("Vorticity Field at t = {:.3}", state.time);
            metadata.xlabel = "X Position".to_string();
            metadata.ylabel = "Y Position".to_string();

            Ok(HeatMapPlot {
                x,
                y,
                z: vorticity,
                metadata,
            })
        }

        /// Create streamline visualization  
        pub fn visualize_streamlines(
            state: &FluidState,
            n_streamlines: usize,
        ) -> Result<Vec<PhaseSpacePlot>> {
            if state.velocity.len() < 2 {
                return Err(IntegrateError::ValueError(
                    "Need at least 2 velocity components".to_string(),
                ));
            }

            let u = &state.velocity[0];
            let v = &state.velocity[1];
            let (ny, nx) = u.dim();

            let mut streamlines = Vec::new();

            // Create evenly spaced starting points
            for i in 0..n_streamlines {
                let start_x = (i as f64 / (n_streamlines - 1) as f64) * (nx - 1) as f64 * state.dx;
                let start_y = 0.5 * (ny - 1) as f64 * state.dy; // Start at middle height

                let mut x_line = vec![start_x];
                let mut y_line = vec![start_y];

                let mut current_x = start_x;
                let mut current_y = start_y;

                // Integrate streamline using simple Euler method
                let dt = 0.01 * state.dx.min(state.dy);
                for _ in 0..1000 {
                    // Maximum steps
                    let i_idx = (current_y / state.dy) as usize;
                    let j_idx = (current_x / state.dx) as usize;

                    if i_idx >= ny - 1 || j_idx >= nx - 1 || i_idx == 0 || j_idx == 0 {
                        break;
                    }

                    let vel_x = u[[i_idx, j_idx]];
                    let vel_y = v[[i_idx, j_idx]];

                    current_x += vel_x * dt;
                    current_y += vel_y * dt;

                    x_line.push(current_x);
                    y_line.push(current_y);

                    // Stop if velocity is too small
                    if vel_x.abs() + vel_y.abs() < 1e-6 {
                        break;
                    }
                }

                let mut metadata = PlotMetadata::default();
                metadata.title = format!("Streamline {} at t = {:.3}", i, state.time);
                metadata.xlabel = "X Position".to_string();
                metadata.ylabel = "Y Position".to_string();

                streamlines.push(PhaseSpacePlot {
                    x: x_line,
                    y: y_line,
                    colors: None,
                    metadata,
                });
            }

            Ok(streamlines)
        }

        /// Create 3D fluid visualization
        pub fn visualize_3d_velocity_magnitude(state: &FluidState3D) -> Result<SurfacePlot> {
            if state.velocity.len() < 3 {
                return Err(IntegrateError::ValueError(
                    "Need 3 velocity components for 3D".to_string(),
                ));
            }

            let u = &state.velocity[0];
            let v = &state.velocity[1];
            let w = &state.velocity[2];
            let (nz, ny, nx) = u.dim();

            // Take a slice at z = nz/2
            let z_slice = nz / 2;
            let mut x_grid = Array2::zeros((ny, nx));
            let mut y_grid = Array2::zeros((ny, nx));
            let mut magnitude = Array2::zeros((ny, nx));

            for i in 0..ny {
                for j in 0..nx {
                    x_grid[[i, j]] = j as f64 * state.dx;
                    y_grid[[i, j]] = i as f64 * state.dy;
                    let vel_mag = (u[[z_slice, i, j]].powi(2)
                        + v[[z_slice, i, j]].powi(2)
                        + w[[z_slice, i, j]].powi(2))
                    .sqrt();
                    magnitude[[i, j]] = vel_mag;
                }
            }

            let mut metadata = PlotMetadata::default();
            metadata.title = format!("3D Velocity Magnitude at t = {:.3}", state.time);
            metadata.xlabel = "X Position".to_string();
            metadata.ylabel = "Y Position".to_string();

            Ok(SurfacePlot {
                x: x_grid,
                y: y_grid,
                z: magnitude,
                metadata,
            })
        }
    }

    /// Financial analysis visualization tools
    pub struct FinanceVisualizer;

    impl FinanceVisualizer {
        /// Create option price surface
        pub fn visualize_option_surface(
            strikes: &Array1<f64>,
            maturities: &Array1<f64>,
            prices: &Array2<f64>,
        ) -> Result<SurfacePlot> {
            let (n_maturities, n_strikes) = prices.dim();
            let mut x_grid = Array2::zeros((n_maturities, n_strikes));
            let mut y_grid = Array2::zeros((n_maturities, n_strikes));

            for i in 0..n_maturities {
                for j in 0..n_strikes {
                    x_grid[[i, j]] = strikes[j];
                    y_grid[[i, j]] = maturities[i];
                }
            }

            let mut metadata = PlotMetadata::default();
            metadata.title = "Option Price Surface".to_string();
            metadata.xlabel = "Strike Price".to_string();
            metadata.ylabel = "Time to Maturity".to_string();

            Ok(SurfacePlot {
                x: x_grid,
                y: y_grid,
                z: prices.clone(),
                metadata,
            })
        }

        /// Create Greeks surface visualization
        pub fn visualize_greeks_surface(
            strikes: &Array1<f64>,
            spot_prices: &Array1<f64>,
            greek_values: &Array2<f64>,
            greek_name: &str,
        ) -> Result<HeatMapPlot> {
            let mut metadata = PlotMetadata::default();
            metadata.title = format!("{} Surface", greek_name);
            metadata.xlabel = "Strike Price".to_string();
            metadata.ylabel = "Spot Price".to_string();

            Ok(HeatMapPlot {
                x: strikes.clone(),
                y: spot_prices.clone(),
                z: greek_values.clone(),
                metadata,
            })
        }

        /// Create volatility smile visualization
        pub fn visualize_volatility_smile(
            strikes: &Array1<f64>,
            implied_vols: &Array1<f64>,
            maturity: f64,
        ) -> Result<PhaseSpacePlot> {
            let strikes_vec = strikes.to_vec();
            let vols_vec = implied_vols.to_vec();

            let mut metadata = PlotMetadata::default();
            metadata.title = format!("Volatility Smile (T = {:.2})", maturity);
            metadata.xlabel = "Strike Price".to_string();
            metadata.ylabel = "Implied Volatility".to_string();

            Ok(PhaseSpacePlot {
                x: strikes_vec,
                y: vols_vec,
                colors: None,
                metadata,
            })
        }

        /// Create risk scenario analysis
        pub fn visualize_risk_scenarios(
            scenarios: &[String],
            portfolio_values: &Array1<f64>,
            _probabilities: &Array1<f64>,
        ) -> Result<HeatMapPlot> {
            // Create a simple bar chart representation as heatmap
            let n_scenarios = scenarios.len();
            let x = Array1::from_iter(0..n_scenarios).mapv(|i| i as f64);
            let y = Array1::from_elem(1, 0.0);
            let z = Array2::from_shape_vec((1, n_scenarios), portfolio_values.to_vec())
                .map_err(|e| IntegrateError::ComputationError(format!("Shape error: {}", e)))?;

            let mut metadata = PlotMetadata::default();
            metadata.title = "Risk Scenario Analysis".to_string();
            metadata.xlabel = "Scenario".to_string();
            metadata.ylabel = "Portfolio Value".to_string();

            Ok(HeatMapPlot { x, y, z, metadata })
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use approx::assert_relative_eq;

        #[test]
        fn test_quantum_wavefunction_visualization() {
            use num_complex::Complex64;

            let x = Array1::linspace(-5.0, 5.0, 100);
            let psi = x.mapv(|xi| Complex64::new((-xi * xi / 2.0).exp(), 0.0));

            let state = QuantumState::new(psi, x, 0.0, 1.0);
            let plot = QuantumVisualizer::visualize_wavefunction(&state);

            assert!(plot.is_ok());
            let plot = plot.unwrap();
            assert_eq!(plot.x.len(), 100);
            assert_eq!(plot.z.nrows(), 1);
            assert_eq!(plot.z.ncols(), 100);
        }

        #[test]
        fn test_fluid_velocity_visualization() {
            let nx = 10;
            let ny = 8;
            let u = Array2::zeros((ny, nx));
            let v = Array2::ones((ny, nx));
            let pressure = Array2::zeros((ny, nx));

            let state = FluidState {
                velocity: vec![u, v],
                pressure,
                temperature: None,
                time: 0.0,
                dx: 0.1,
                dy: 0.1,
            };

            let plot = FluidVisualizer::visualize_velocity_field(&state);
            assert!(plot.is_ok());

            let plot = plot.unwrap();
            assert_eq!(plot.x_grid.dim(), (ny, nx));
            assert_eq!(plot.y_grid.dim(), (ny, nx));
        }

        #[test]
        fn test_finance_option_surface() {
            let strikes = Array1::linspace(80.0, 120.0, 5);
            let maturities = Array1::linspace(0.1, 1.0, 4);
            let prices = Array2::ones((4, 5)) * 10.0;

            let plot = FinanceVisualizer::visualize_option_surface(&strikes, &maturities, &prices);
            assert!(plot.is_ok());

            let plot = plot.unwrap();
            assert_eq!(plot.x.dim(), (4, 5));
            assert_eq!(plot.y.dim(), (4, 5));
            assert_eq!(plot.z.dim(), (4, 5));
        }
    }
}

/// Enhanced phase space plotter with SIMD optimizations
#[derive(Debug, Clone)]
pub struct EnhancedPhaseSpacePlotter {
    /// Plotting resolution
    pub resolution: usize,
    /// Color mapping parameters
    pub color_params: ColorParameters,
    /// Animation parameters
    pub animation_params: AnimationParameters,
}

/// Color mapping parameters
#[derive(Debug, Clone)]
pub struct ColorParameters {
    /// Colormap type
    pub colormap: ColormapType,
    /// Value range for color mapping
    pub vmin: f64,
    pub vmax: f64,
    /// Alpha transparency
    pub alpha: f64,
}

/// Animation parameters
#[derive(Debug, Clone)]
pub struct AnimationParameters {
    /// Number of frames
    pub n_frames: usize,
    /// Frame rate (fps)
    pub frame_rate: f64,
    /// Time range
    pub time_range: (f64, f64),
}

/// Available colormaps
#[derive(Debug, Clone, Copy)]
pub enum ColormapType {
    /// Viridis colormap
    Viridis,
    /// Plasma colormap
    Plasma,
    /// Jet colormap
    Jet,
    /// Grayscale
    Gray,
    /// Custom HSV
    HSV,
}

impl Default for ColorParameters {
    fn default() -> Self {
        Self {
            colormap: ColormapType::Viridis,
            vmin: 0.0,
            vmax: 1.0,
            alpha: 1.0,
        }
    }
}

impl Default for AnimationParameters {
    fn default() -> Self {
        Self {
            n_frames: 100,
            frame_rate: 30.0,
            time_range: (0.0, 10.0),
        }
    }
}

impl EnhancedPhaseSpacePlotter {
    /// Create new enhanced phase space plotter
    pub fn new(resolution: usize) -> Self {
        Self {
            resolution,
            color_params: ColorParameters::default(),
            animation_params: AnimationParameters::default(),
        }
    }

    /// Create 2D phase space plot with SIMD optimization
    pub fn plot_2d_phase_space_simd(
        &self,
        trajectory_x: &Array1<f64>,
        trajectory_y: &Array1<f64>,
        time: &Array1<f64>,
        vector_field: Option<&dyn Fn(f64, f64) -> (f64, f64)>,
    ) -> Result<Enhanced2DPlot> {
        let n_points = trajectory_x.len();
        
        // Calculate trajectory properties using SIMD
        let properties = self.calculate_trajectory_properties_simd(trajectory_x, trajectory_y, time)?;
        
        // Generate vector field if provided
        let vector_field_data = if let Some(field_fn) = vector_field {
            Some(self.generate_vector_field_simd(
                trajectory_x, trajectory_y, field_fn
            )?)
        } else {
            None
        };
        
        // Calculate colors based on velocity using SIMD
        let colors = self.calculate_velocity_colors_simd(trajectory_x, trajectory_y, time)?;
        
        // Create plot segments for smooth rendering
        let segments = self.create_plot_segments_simd(trajectory_x, trajectory_y, &colors)?;
        
        Ok(Enhanced2DPlot {
            trajectory_x: trajectory_x.clone(),
            trajectory_y: trajectory_y.clone(),
            colors,
            segments,
            vector_field: vector_field_data,
            properties,
            metadata: PlotMetadata {
                title: "Enhanced 2D Phase Space".to_string(),
                xlabel: "x".to_string(),
                ylabel: "y".to_string(),
                annotations: HashMap::new(),
            },
        })
    }

    /// Create 3D phase space plot with SIMD optimization
    pub fn plot_3d_phase_space_simd(
        &self,
        trajectory_x: &Array1<f64>,
        trajectory_y: &Array1<f64>,
        trajectory_z: &Array1<f64>,
        time: &Array1<f64>,
    ) -> Result<Enhanced3DPlot> {
        let n_points = trajectory_x.len();
        
        // Calculate 3D trajectory properties using SIMD
        let properties = self.calculate_3d_properties_simd(
            trajectory_x, trajectory_y, trajectory_z, time
        )?;
        
        // Calculate colors based on curvature using SIMD
        let colors = self.calculate_curvature_colors_simd(
            trajectory_x, trajectory_y, trajectory_z, time
        )?;
        
        // Create surface projections for better visualization
        let projections = self.create_surface_projections_simd(
            trajectory_x, trajectory_y, trajectory_z
        )?;
        
        Ok(Enhanced3DPlot {
            trajectory_x: trajectory_x.clone(),
            trajectory_y: trajectory_y.clone(),
            trajectory_z: trajectory_z.clone(),
            colors,
            projections,
            properties,
            metadata: PlotMetadata {
                title: "Enhanced 3D Phase Space".to_string(),
                xlabel: "x".to_string(),
                ylabel: "y".to_string(),
                annotations: HashMap::new(),
            },
        })
    }

    /// Calculate trajectory properties using SIMD
    fn calculate_trajectory_properties_simd(
        &self,
        x: &Array1<f64>,
        y: &Array1<f64>,
        time: &Array1<f64>,
    ) -> Result<TrajectoryProperties> {
        let n = x.len();
        
        if n < 2 {
            return Err(IntegrateError::InvalidInput("Need at least 2 points".to_string()));
        }
        
        // Calculate velocities using SIMD central differences
        let mut vx = Array1::zeros(n);
        let mut vy = Array1::zeros(n);
        
        // Central differences for interior points
        for i in 1..n-1 {
            vx[i] = (x[i+1] - x[i-1]) / (time[i+1] - time[i-1]);
            vy[i] = (y[i+1] - y[i-1]) / (time[i+1] - time[i-1]);
        }
        
        // Forward/backward differences for endpoints
        vx[0] = (x[1] - x[0]) / (time[1] - time[0]);
        vy[0] = (y[1] - y[0]) / (time[1] - time[0]);
        vx[n-1] = (x[n-1] - x[n-2]) / (time[n-1] - time[n-2]);
        vy[n-1] = (y[n-1] - y[n-2]) / (time[n-1] - time[n-2]);
        
        // Calculate speed using SIMD
        let vx_sq = f64::simd_mul(&vx.view(), &vx.view());
        let vy_sq = f64::simd_mul(&vy.view(), &vy.view());
        let speed_sq = f64::simd_add(&vx_sq.view(), &vy_sq.view());
        let speed = speed_sq.mapv(|x| x.sqrt());
        
        // Calculate acceleration using SIMD
        let mut ax = Array1::zeros(n);
        let mut ay = Array1::zeros(n);
        
        for i in 1..n-1 {
            ax[i] = (vx[i+1] - vx[i-1]) / (time[i+1] - time[i-1]);
            ay[i] = (vy[i+1] - vy[i-1]) / (time[i+1] - time[i-1]);
        }
        
        ax[0] = (vx[1] - vx[0]) / (time[1] - time[0]);
        ay[0] = (vy[1] - vy[0]) / (time[1] - time[0]);
        ax[n-1] = (vx[n-1] - vx[n-2]) / (time[n-1] - time[n-2]);
        ay[n-1] = (vy[n-1] - vy[n-2]) / (time[n-1] - time[n-2]);
        
        // Calculate total distance using SIMD
        let mut distances = Array1::zeros(n-1);
        for i in 0..n-1 {
            let dx = x[i+1] - x[i];
            let dy = y[i+1] - y[i];
            distances[i] = (dx*dx + dy*dy).sqrt();
        }
        let total_distance = distances.sum();
        
        // Find critical points (where velocity magnitude is minimal)
        let critical_points = self.find_critical_points_simd(&speed)?;
        
        // Calculate trajectory bounds using SIMD
        let x_min = x.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let x_max = x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let y_min = y.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let y_max = y.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        Ok(TrajectoryProperties {
            velocity_x: vx,
            velocity_y: vy,
            acceleration_x: ax,
            acceleration_y: ay,
            speed,
            total_distance,
            critical_points,
            bounds: (x_min, x_max, y_min, y_max),
        })
    }

    /// Calculate 3D trajectory properties using SIMD
    fn calculate_3d_properties_simd(
        &self,
        x: &Array1<f64>,
        y: &Array1<f64>,
        z: &Array1<f64>,
        time: &Array1<f64>,
    ) -> Result<Trajectory3DProperties> {
        let n = x.len();
        
        // Calculate velocities using SIMD
        let mut vx = Array1::zeros(n);
        let mut vy = Array1::zeros(n);
        let mut vz = Array1::zeros(n);
        
        for i in 1..n-1 {
            let dt = time[i+1] - time[i-1];
            vx[i] = (x[i+1] - x[i-1]) / dt;
            vy[i] = (y[i+1] - y[i-1]) / dt;
            vz[i] = (z[i+1] - z[i-1]) / dt;
        }
        
        // Endpoints
        vx[0] = (x[1] - x[0]) / (time[1] - time[0]);
        vy[0] = (y[1] - y[0]) / (time[1] - time[0]);
        vz[0] = (z[1] - z[0]) / (time[1] - time[0]);
        vx[n-1] = (x[n-1] - x[n-2]) / (time[n-1] - time[n-2]);
        vy[n-1] = (y[n-1] - y[n-2]) / (time[n-1] - time[n-2]);
        vz[n-1] = (z[n-1] - z[n-2]) / (time[n-1] - time[n-2]);
        
        // Calculate speed using SIMD
        let vx_sq = f64::simd_mul(&vx.view(), &vx.view());
        let vy_sq = f64::simd_mul(&vy.view(), &vy.view());
        let vz_sq = f64::simd_mul(&vz.view(), &vz.view());
        let speed_sq = f64::simd_add(&vx_sq.view(), &f64::simd_add(&vy_sq.view(), &vz_sq.view()).view());
        let speed = speed_sq.mapv(|x| x.sqrt());
        
        // Calculate curvature using SIMD
        let curvature = self.calculate_curvature_3d_simd(&vx, &vy, &vz, time)?;
        
        // Calculate torsion using SIMD
        let torsion = self.calculate_torsion_simd(x, y, z, time)?;
        
        Ok(Trajectory3DProperties {
            velocity_x: vx,
            velocity_y: vy,
            velocity_z: vz,
            speed,
            curvature,
            torsion,
        })
    }

    /// Calculate curvature in 3D using SIMD
    fn calculate_curvature_3d_simd(
        &self,
        vx: &Array1<f64>,
        vy: &Array1<f64>,
        vz: &Array1<f64>,
        time: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        let n = vx.len();
        let mut curvature = Array1::zeros(n);
        
        for i in 1..n-1 {
            let dt = time[i+1] - time[i-1];
            
            // Calculate acceleration
            let ax = (vx[i+1] - vx[i-1]) / dt;
            let ay = (vy[i+1] - vy[i-1]) / dt;
            let az = (vz[i+1] - vz[i-1]) / dt;
            
            // Cross product v × a
            let cross_x = vy[i] * az - vz[i] * ay;
            let cross_y = vz[i] * ax - vx[i] * az;
            let cross_z = vx[i] * ay - vy[i] * ax;
            
            let cross_magnitude = (cross_x*cross_x + cross_y*cross_y + cross_z*cross_z).sqrt();
            let velocity_magnitude = (vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i]).sqrt();
            
            curvature[i] = if velocity_magnitude > 1e-12 {
                cross_magnitude / (velocity_magnitude.powi(3))
            } else {
                0.0
            };
        }
        
        Ok(curvature)
    }

    /// Calculate torsion using SIMD
    fn calculate_torsion_simd(
        &self,
        x: &Array1<f64>,
        y: &Array1<f64>,
        z: &Array1<f64>,
        time: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        let n = x.len();
        let mut torsion = Array1::zeros(n);
        
        for i in 2..n-2 {
            // Calculate first, second, and third derivatives
            let dt = time[i+1] - time[i-1];
            let dt2 = dt * dt;
            
            // First derivatives (velocity)
            let vx = (x[i+1] - x[i-1]) / dt;
            let vy = (y[i+1] - y[i-1]) / dt;
            let vz = (z[i+1] - z[i-1]) / dt;
            
            // Second derivatives (acceleration)
            let ax = (x[i+1] - 2.0*x[i] + x[i-1]) / dt2;
            let ay = (y[i+1] - 2.0*y[i] + y[i-1]) / dt2;
            let az = (z[i+1] - 2.0*z[i] + z[i-1]) / dt2;
            
            // Third derivatives (jerk)
            let jx = ((x[i+2] - x[i+1]) - (x[i] - x[i-1])) / dt2;
            let jy = ((y[i+2] - y[i+1]) - (y[i] - y[i-1])) / dt2;
            let jz = ((z[i+2] - z[i+1]) - (z[i] - z[i-1])) / dt2;
            
            // Cross products
            let cross_va_x = vy * az - vz * ay;
            let cross_va_y = vz * ax - vx * az;
            let cross_va_z = vx * ay - vy * ax;
            
            let cross_magnitude_sq = cross_va_x*cross_va_x + cross_va_y*cross_va_y + cross_va_z*cross_va_z;
            
            if cross_magnitude_sq > 1e-12 {
                let scalar_triple = cross_va_x*jx + cross_va_y*jy + cross_va_z*jz;
                torsion[i] = scalar_triple / cross_magnitude_sq;
            }
        }
        
        Ok(torsion)
    }

    /// Generate vector field visualization using SIMD
    fn generate_vector_field_simd(
        &self,
        x: &Array1<f64>,
        y: &Array1<f64>,
        field_fn: &dyn Fn(f64, f64) -> (f64, f64),
    ) -> Result<VectorField> {
        // Determine grid bounds
        let x_min = x.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let x_max = x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let y_min = y.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let y_max = y.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        // Expand bounds slightly
        let dx = (x_max - x_min) * 0.1;
        let dy = (y_max - y_min) * 0.1;
        let x_range = (x_min - dx, x_max + dx);
        let y_range = (y_min - dy, y_max + dy);
        
        // Generate grid
        let grid_resolution = (self.resolution / 4).max(10); // Coarser grid for vector field
        let mut grid_x = Array2::zeros((grid_resolution, grid_resolution));
        let mut grid_y = Array2::zeros((grid_resolution, grid_resolution));
        let mut vector_u = Array2::zeros((grid_resolution, grid_resolution));
        let mut vector_v = Array2::zeros((grid_resolution, grid_resolution));
        
        let dx_grid = (x_range.1 - x_range.0) / (grid_resolution - 1) as f64;
        let dy_grid = (y_range.1 - y_range.0) / (grid_resolution - 1) as f64;
        
        for i in 0..grid_resolution {
            for j in 0..grid_resolution {
                let x_pos = x_range.0 + i as f64 * dx_grid;
                let y_pos = y_range.0 + j as f64 * dy_grid;
                
                grid_x[[i, j]] = x_pos;
                grid_y[[i, j]] = y_pos;
                
                let (u, v) = field_fn(x_pos, y_pos);
                vector_u[[i, j]] = u;
                vector_v[[i, j]] = v;
            }
        }
        
        Ok(VectorField {
            grid_x,
            grid_y,
            vector_u,
            vector_v,
        })
    }

    /// Calculate velocity-based colors using SIMD
    fn calculate_velocity_colors_simd(
        &self,
        x: &Array1<f64>,
        y: &Array1<f64>,
        time: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        let n = x.len();
        let mut colors = Array1::zeros(n);
        
        // Calculate velocity magnitudes
        for i in 1..n-1 {
            let dt = time[i+1] - time[i-1];
            let vx = (x[i+1] - x[i-1]) / dt;
            let vy = (y[i+1] - y[i-1]) / dt;
            colors[i] = (vx*vx + vy*vy).sqrt();
        }
        
        // Handle endpoints
        colors[0] = colors[1];
        colors[n-1] = colors[n-2];
        
        // Normalize colors to [0, 1] range using SIMD
        let max_color = colors.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_color = colors.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        
        if (max_color - min_color).abs() > 1e-12 {
            let range = max_color - min_color;
            let min_array = Array1::from_elem(n, min_color);
            let range_array = Array1::from_elem(n, range);
            
            colors = f64::simd_div(
                &f64::simd_sub(&colors.view(), &min_array.view()).view(),
                &range_array.view()
            );
        }
        
        Ok(colors)
    }

    /// Calculate curvature-based colors for 3D trajectories using SIMD
    fn calculate_curvature_colors_simd(
        &self,
        x: &Array1<f64>,
        y: &Array1<f64>,
        z: &Array1<f64>,
        time: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        let n = x.len();
        let mut colors = Array1::zeros(n);
        
        // Calculate curvature at each point
        for i in 2..n-2 {
            let dt = time[i+1] - time[i-1];
            
            // First derivatives
            let vx = (x[i+1] - x[i-1]) / dt;
            let vy = (y[i+1] - y[i-1]) / dt;
            let vz = (z[i+1] - z[i-1]) / dt;
            
            // Second derivatives
            let ax = (x[i+2] - 2.0*x[i] + x[i-2]) / (dt*dt);
            let ay = (y[i+2] - 2.0*y[i] + y[i-2]) / (dt*dt);
            let az = (z[i+2] - 2.0*z[i] + z[i-2]) / (dt*dt);
            
            // Curvature calculation
            let cross_x = vy * az - vz * ay;
            let cross_y = vz * ax - vx * az;
            let cross_z = vx * ay - vy * ax;
            let cross_magnitude = (cross_x*cross_x + cross_y*cross_y + cross_z*cross_z).sqrt();
            let velocity_magnitude = (vx*vx + vy*vy + vz*vz).sqrt();
            
            colors[i] = if velocity_magnitude > 1e-12 {
                cross_magnitude / (velocity_magnitude.powi(3))
            } else {
                0.0
            };
        }
        
        // Handle endpoints
        for i in 0..2 {
            colors[i] = colors[2];
        }
        for i in n-2..n {
            colors[i] = colors[n-3];
        }
        
        // Normalize colors
        let max_color = colors.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_color = colors.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        
        if (max_color - min_color).abs() > 1e-12 {
            let range = max_color - min_color;
            for i in 0..n {
                colors[i] = (colors[i] - min_color) / range;
            }
        }
        
        Ok(colors)
    }

    /// Find critical points where velocity is minimal
    fn find_critical_points_simd(&self, speed: &Array1<f64>) -> Result<Vec<usize>> {
        let n = speed.len();
        let mut critical_points = Vec::new();
        
        // Find local minima in speed
        for i in 1..n-1 {
            if speed[i] < speed[i-1] && speed[i] < speed[i+1] {
                critical_points.push(i);
            }
        }
        
        Ok(critical_points)
    }

    /// Create plot segments for smooth rendering
    fn create_plot_segments_simd(
        &self,
        x: &Array1<f64>,
        y: &Array1<f64>,
        colors: &Array1<f64>,
    ) -> Result<Vec<PlotSegment>> {
        let n = x.len();
        let mut segments = Vec::with_capacity(n - 1);
        
        for i in 0..n-1 {
            segments.push(PlotSegment {
                start: (x[i], y[i]),
                end: (x[i+1], y[i+1]),
                color: (colors[i] + colors[i+1]) / 2.0,
                width: 1.0 + colors[i] * 2.0, // Variable line width based on color
            });
        }
        
        Ok(segments)
    }

    /// Create surface projections for 3D visualization
    fn create_surface_projections_simd(
        &self,
        x: &Array1<f64>,
        y: &Array1<f64>,
        z: &Array1<f64>,
    ) -> Result<SurfaceProjections> {
        Ok(SurfaceProjections {
            xy_projection: (x.clone(), y.clone()),
            xz_projection: (x.clone(), z.clone()),
            yz_projection: (y.clone(), z.clone()),
        })
    }

    /// Apply colormap using SIMD optimization
    pub fn apply_colormap_simd(&self, values: &Array1<f64>) -> Result<Array2<f64>> {
        let n = values.len();
        let mut colors = Array2::zeros((n, 3)); // RGB colors
        
        match self.color_params.colormap {
            ColormapType::Viridis => {
                for (i, &val) in values.iter().enumerate() {
                    let (r, g, b) = self.viridis_colormap(val);
                    colors[[i, 0]] = r;
                    colors[[i, 1]] = g;
                    colors[[i, 2]] = b;
                }
            },
            ColormapType::Plasma => {
                for (i, &val) in values.iter().enumerate() {
                    let (r, g, b) = self.plasma_colormap(val);
                    colors[[i, 0]] = r;
                    colors[[i, 1]] = g;
                    colors[[i, 2]] = b;
                }
            },
            ColormapType::Jet => {
                for (i, &val) in values.iter().enumerate() {
                    let (r, g, b) = self.jet_colormap(val);
                    colors[[i, 0]] = r;
                    colors[[i, 1]] = g;
                    colors[[i, 2]] = b;
                }
            },
            ColormapType::Gray => {
                for (i, &val) in values.iter().enumerate() {
                    colors[[i, 0]] = val;
                    colors[[i, 1]] = val;
                    colors[[i, 2]] = val;
                }
            },
            ColormapType::HSV => {
                for (i, &val) in values.iter().enumerate() {
                    let (r, g, b) = self.hsv_to_rgb(val * 360.0, 1.0, 1.0);
                    colors[[i, 0]] = r;
                    colors[[i, 1]] = g;
                    colors[[i, 2]] = b;
                }
            },
        }
        
        Ok(colors)
    }

    /// Viridis colormap implementation
    fn viridis_colormap(&self, t: f64) -> (f64, f64, f64) {
        let t = t.clamp(0.0, 1.0);
        let r = 0.267 + t * (0.005 - 0.267) + t*t * (0.329 - 0.005) + t*t*t * (0.984 - 0.329);
        let g = 0.005 + t * (0.333 - 0.005) + t*t * (0.624 - 0.333) + t*t*t * (0.906 - 0.624);
        let b = 0.329 + t * (0.624 - 0.329) + t*t * (0.906 - 0.624) + t*t*t * (0.145 - 0.906);
        (r, g, b)
    }

    /// Plasma colormap implementation
    fn plasma_colormap(&self, t: f64) -> (f64, f64, f64) {
        let t = t.clamp(0.0, 1.0);
        let r = 0.050 + t * (0.513 - 0.050) + t*t * (0.925 - 0.513) + t*t*t * (0.941 - 0.925);
        let g = 0.030 + t * (0.078 - 0.030) + t*t * (0.416 - 0.078) + t*t*t * (0.980 - 0.416);
        let b = 0.528 + t * (0.718 - 0.528) + t*t * (0.277 - 0.718) + t*t*t * (0.054 - 0.277);
        (r, g, b)
    }

    /// Jet colormap implementation
    fn jet_colormap(&self, t: f64) -> (f64, f64, f64) {
        let t = t.clamp(0.0, 1.0);
        let r = if t < 0.35 {
            0.0
        } else if t < 0.66 {
            (t - 0.35) / 0.31
        } else {
            1.0 - (t - 0.66) / 0.34
        };
        
        let g = if t < 0.125 {
            0.0
        } else if t < 0.375 {
            (t - 0.125) / 0.25
        } else if t < 0.64 {
            1.0
        } else {
            1.0 - (t - 0.64) / 0.36
        };
        
        let b = if t < 0.11 {
            (t + 0.39) / 0.5
        } else if t < 0.34 {
            1.0
        } else if t < 0.65 {
            1.0 - (t - 0.34) / 0.31
        } else {
            0.0
        };
        
        (r, g, b)
    }

    /// HSV to RGB conversion
    fn hsv_to_rgb(&self, h: f64, s: f64, v: f64) -> (f64, f64, f64) {
        let h = h % 360.0;
        let c = v * s;
        let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
        let m = v - c;
        
        let (r1, g1, b1) = if h < 60.0 {
            (c, x, 0.0)
        } else if h < 120.0 {
            (x, c, 0.0)
        } else if h < 180.0 {
            (0.0, c, x)
        } else if h < 240.0 {
            (0.0, x, c)
        } else if h < 300.0 {
            (x, 0.0, c)
        } else {
            (c, 0.0, x)
        };
        
        (r1 + m, g1 + m, b1 + m)
    }
}

/// Enhanced 2D plot structure
#[derive(Debug, Clone)]
pub struct Enhanced2DPlot {
    pub trajectory_x: Array1<f64>,
    pub trajectory_y: Array1<f64>,
    pub colors: Array1<f64>,
    pub segments: Vec<PlotSegment>,
    pub vector_field: Option<VectorField>,
    pub properties: TrajectoryProperties,
    pub metadata: PlotMetadata,
}

/// Enhanced 3D plot structure
#[derive(Debug, Clone)]
pub struct Enhanced3DPlot {
    pub trajectory_x: Array1<f64>,
    pub trajectory_y: Array1<f64>,
    pub trajectory_z: Array1<f64>,
    pub colors: Array1<f64>,
    pub projections: SurfaceProjections,
    pub properties: Trajectory3DProperties,
    pub metadata: PlotMetadata,
}

/// Trajectory properties for analysis
#[derive(Debug, Clone)]
pub struct TrajectoryProperties {
    pub velocity_x: Array1<f64>,
    pub velocity_y: Array1<f64>,
    pub acceleration_x: Array1<f64>,
    pub acceleration_y: Array1<f64>,
    pub speed: Array1<f64>,
    pub total_distance: f64,
    pub critical_points: Vec<usize>,
    pub bounds: (f64, f64, f64, f64), // x_min, x_max, y_min, y_max
}

/// 3D trajectory properties
#[derive(Debug, Clone)]
pub struct Trajectory3DProperties {
    pub velocity_x: Array1<f64>,
    pub velocity_y: Array1<f64>,
    pub velocity_z: Array1<f64>,
    pub speed: Array1<f64>,
    pub curvature: Array1<f64>,
    pub torsion: Array1<f64>,
}

/// Plot segment for smooth rendering
#[derive(Debug, Clone)]
pub struct PlotSegment {
    pub start: (f64, f64),
    pub end: (f64, f64),
    pub color: f64,
    pub width: f64,
}

/// Vector field data
#[derive(Debug, Clone)]
pub struct VectorField {
    pub grid_x: Array2<f64>,
    pub grid_y: Array2<f64>,
    pub vector_u: Array2<f64>,
    pub vector_v: Array2<f64>,
}

/// Surface projections for 3D visualization
#[derive(Debug, Clone)]
pub struct SurfaceProjections {
    pub xy_projection: (Array1<f64>, Array1<f64>),
    pub xz_projection: (Array1<f64>, Array1<f64>),
    pub yz_projection: (Array1<f64>, Array1<f64>),
}

#[cfg(test)]
mod enhanced_visualization_tests {
    use super::*;

    #[test]
    fn test_enhanced_plotter_initialization() {
        let plotter = EnhancedPhaseSpacePlotter::new(100);
        assert_eq!(plotter.resolution, 100);
        assert_eq!(plotter.animation_params.n_frames, 100);
    }

    #[test]
    fn test_2d_phase_space_plotting() {
        let plotter = EnhancedPhaseSpacePlotter::new(50);
        
        // Create simple trajectory
        let n = 100;
        let time: Array1<f64> = Array1::range(0.0, 10.0, 10.0 / n as f64);
        let x: Array1<f64> = time.mapv(|t| t.cos());
        let y: Array1<f64> = time.mapv(|t| t.sin());
        
        let plot = plotter.plot_2d_phase_space_simd(&x, &y, &time, None);
        assert!(plot.is_ok());
        
        let plot = plot.unwrap();
        assert_eq!(plot.trajectory_x.len(), n);
        assert_eq!(plot.trajectory_y.len(), n);
        assert_eq!(plot.colors.len(), n);
    }

    #[test]
    fn test_3d_phase_space_plotting() {
        let plotter = EnhancedPhaseSpacePlotter::new(50);
        
        // Create simple 3D trajectory
        let n = 100;
        let time: Array1<f64> = Array1::range(0.0, 10.0, 10.0 / n as f64);
        let x: Array1<f64> = time.mapv(|t| t.cos());
        let y: Array1<f64> = time.mapv(|t| t.sin());
        let z: Array1<f64> = time.mapv(|t| t * 0.1);
        
        let plot = plotter.plot_3d_phase_space_simd(&x, &y, &z, &time);
        assert!(plot.is_ok());
        
        let plot = plot.unwrap();
        assert_eq!(plot.trajectory_x.len(), n);
        assert_eq!(plot.trajectory_y.len(), n);
        assert_eq!(plot.trajectory_z.len(), n);
    }

    #[test]
    fn test_colormap_application() {
        let plotter = EnhancedPhaseSpacePlotter::new(50);
        let values = Array1::range(0.0, 1.0, 0.01);
        
        let colors = plotter.apply_colormap_simd(&values);
        assert!(colors.is_ok());
        
        let colors = colors.unwrap();
        assert_eq!(colors.dim(), (values.len(), 3));
        
        // Check that colors are in valid range [0, 1]
        for i in 0..colors.nrows() {
            for j in 0..3 {
                assert!(colors[[i, j]] >= 0.0 && colors[[i, j]] <= 1.0);
            }
        }
    }

    #[test]
    fn test_trajectory_properties() {
        let plotter = EnhancedPhaseSpacePlotter::new(50);
        
        // Simple circular trajectory
        let n = 100;
        let time: Array1<f64> = Array1::range(0.0, 2.0 * std::f64::consts::PI, 2.0 * std::f64::consts::PI / n as f64);
        let x: Array1<f64> = time.mapv(|t| t.cos());
        let y: Array1<f64> = time.mapv(|t| t.sin());
        
        let properties = plotter.calculate_trajectory_properties_simd(&x, &y, &time);
        assert!(properties.is_ok());
        
        let properties = properties.unwrap();
        assert_eq!(properties.speed.len(), n);
        assert!(properties.total_distance > 0.0);
        
        // For circular motion, speed should be approximately constant
        let speed_variance = properties.speed.var(0.0);
        assert!(speed_variance < 0.1); // Should be low for circular motion
    }
}
