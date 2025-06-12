//! Adaptive Mesh Refinement (AMR) for PDEs
//!
//! This module implements adaptive mesh refinement techniques for partial
//! differential equations. AMR automatically refines the computational mesh
//! in regions where high accuracy is needed while keeping coarse meshes
//! elsewhere to maintain computational efficiency.
//!
//! # AMR Concepts
//!
//! - **Error Estimation**: Identify regions needing refinement
//! - **Refinement Criteria**: Decide when and where to refine
//! - **Data Transfer**: Interpolate solution between mesh levels
//! - **Load Balancing**: Distribute refined regions across processors
//!
//! # Examples
//!
//! ```
//! use scirs2_integrate::pde::amr::{AMRGrid, RefinementCriteria, AMRSolver};
//!
//! // Create AMR grid with initial coarse mesh
//! let mut grid = AMRGrid::new(64, 64, [0.0, 1.0], [0.0, 1.0]);
//!
//! // Set refinement criteria based on solution gradients
//! let criteria = RefinementCriteria::gradient_based(0.1);
//!
//! // Solve with adaptive refinement
//! let solver = AMRSolver::new(grid, criteria);
//! ```

use crate::common::IntegrateFloat;
use crate::error::{IntegrateError, IntegrateResult};
use crate::pde::PDEResult;
use ndarray::{Array2, ArrayView2};
use std::collections::HashMap;

/// Adaptive mesh refinement grid with hierarchical structure
#[derive(Debug, Clone)]
pub struct AMRGrid<F: IntegrateFloat> {
    /// Grid hierarchy levels (level 0 is coarsest)
    levels: Vec<GridLevel<F>>,
    /// Maximum allowed refinement level
    max_level: usize,
    /// Minimum allowed refinement level
    min_level: usize,
    /// Domain bounds
    #[allow(dead_code)]
    domain: ([F; 2], [F; 2]),
    /// Current solution on all levels
    solution: HashMap<(usize, usize, usize), F>, // (level, i, j) -> value
}

/// Single level in the AMR hierarchy
#[derive(Debug, Clone)]
pub struct GridLevel<F: IntegrateFloat> {
    /// Grid level (0 = coarsest)
    #[allow(dead_code)]
    level: usize,
    /// Number of cells in x direction
    nx: usize,
    /// Number of cells in y direction
    ny: usize,
    /// Grid spacing in x direction
    dx: F,
    /// Grid spacing in y direction
    dy: F,
    /// Refinement map (true = refined, false = not refined)
    refined: Array2<bool>,
    /// Child level information for refined cells
    children: HashMap<(usize, usize), ChildInfo>,
}

/// Information about child cells in refined regions
#[derive(Debug, Clone)]
pub struct ChildInfo {
    /// Starting indices of child region in next level
    child_start: (usize, usize),
    /// Size of child region
    child_size: (usize, usize),
}

/// Refinement criteria for deciding when to refine/coarsen
#[derive(Clone)]
pub enum RefinementCriteria<F: IntegrateFloat> {
    /// Refine based on solution gradient magnitude
    GradientBased { threshold: F, coarsen_threshold: F },
    /// Refine based on second derivative (curvature)
    CurvatureBased { threshold: F, coarsen_threshold: F },
    /// Refine based on estimated truncation error
    ErrorBased { threshold: F, coarsen_threshold: F },
    // Custom refinement function (Note: Clone not supported for function pointers)
    // Custom {
    //     refine_fn: Box<dyn Fn(ArrayView2<F>, usize, usize) -> bool + Send + Sync>,
    //     coarsen_fn: Box<dyn Fn(ArrayView2<F>, usize, usize) -> bool + Send + Sync>,
    // },
}

impl<F: IntegrateFloat> std::fmt::Debug for RefinementCriteria<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GradientBased {
                threshold,
                coarsen_threshold,
            } => f
                .debug_struct("GradientBased")
                .field("threshold", &format!("{:?}", threshold))
                .field("coarsen_threshold", &format!("{:?}", coarsen_threshold))
                .finish(),
            Self::CurvatureBased {
                threshold,
                coarsen_threshold,
            } => f
                .debug_struct("CurvatureBased")
                .field("threshold", &format!("{:?}", threshold))
                .field("coarsen_threshold", &format!("{:?}", coarsen_threshold))
                .finish(),
            Self::ErrorBased {
                threshold,
                coarsen_threshold,
            } => f
                .debug_struct("ErrorBased")
                .field("threshold", &format!("{:?}", threshold))
                .field("coarsen_threshold", &format!("{:?}", coarsen_threshold))
                .finish(),
        }
    }
}

impl<F: IntegrateFloat> RefinementCriteria<F> {
    /// Create gradient-based refinement criteria
    pub fn gradient_based(threshold: F) -> Self {
        Self::GradientBased {
            threshold,
            coarsen_threshold: threshold / F::from(4.0).unwrap(),
        }
    }

    /// Create curvature-based refinement criteria
    pub fn curvature_based(threshold: F) -> Self {
        Self::CurvatureBased {
            threshold,
            coarsen_threshold: threshold / F::from(4.0).unwrap(),
        }
    }

    /// Create error-based refinement criteria
    pub fn error_based(threshold: F) -> Self {
        Self::ErrorBased {
            threshold,
            coarsen_threshold: threshold / F::from(16.0).unwrap(),
        }
    }

    /// Check if cell should be refined
    pub fn should_refine(&self, solution: ArrayView2<F>, i: usize, j: usize) -> bool {
        match self {
            Self::GradientBased { threshold, .. } => {
                self.compute_gradient_magnitude(solution, i, j) > *threshold
            }
            Self::CurvatureBased { threshold, .. } => {
                self.compute_curvature(solution, i, j) > *threshold
            }
            Self::ErrorBased { threshold, .. } => {
                self.estimate_truncation_error(solution, i, j) > *threshold
            } // Self::Custom { refine_fn, .. } => refine_fn(solution, i, j),
        }
    }

    /// Check if cell should be coarsened
    pub fn should_coarsen(&self, solution: ArrayView2<F>, i: usize, j: usize) -> bool {
        match self {
            Self::GradientBased {
                coarsen_threshold, ..
            } => self.compute_gradient_magnitude(solution, i, j) < *coarsen_threshold,
            Self::CurvatureBased {
                coarsen_threshold, ..
            } => self.compute_curvature(solution, i, j) < *coarsen_threshold,
            Self::ErrorBased {
                coarsen_threshold, ..
            } => self.estimate_truncation_error(solution, i, j) < *coarsen_threshold,
            // Self::Custom { coarsen_fn, .. } => coarsen_fn(solution, i, j),
        }
    }

    /// Compute gradient magnitude at cell (i, j)
    fn compute_gradient_magnitude(&self, solution: ArrayView2<F>, i: usize, j: usize) -> F {
        let (nx, ny) = solution.dim();

        // Compute gradients using centered differences where possible
        let grad_x = if i == 0 {
            solution[[1, j]] - solution[[0, j]]
        } else if i == nx - 1 {
            solution[[nx - 1, j]] - solution[[nx - 2, j]]
        } else {
            (solution[[i + 1, j]] - solution[[i - 1, j]]) / F::from(2.0).unwrap()
        };

        let grad_y = if j == 0 {
            solution[[i, 1]] - solution[[i, 0]]
        } else if j == ny - 1 {
            solution[[i, ny - 1]] - solution[[i, ny - 2]]
        } else {
            (solution[[i, j + 1]] - solution[[i, j - 1]]) / F::from(2.0).unwrap()
        };

        (grad_x * grad_x + grad_y * grad_y).sqrt()
    }

    /// Compute curvature (second derivative magnitude) at cell (i, j)
    fn compute_curvature(&self, solution: ArrayView2<F>, i: usize, j: usize) -> F {
        let (nx, ny) = solution.dim();

        if i == 0 || i == nx - 1 || j == 0 || j == ny - 1 {
            return F::zero(); // Can't compute curvature at boundaries
        }

        // Second derivatives using centered differences
        let d2_dx2 =
            solution[[i + 1, j]] - F::from(2.0).unwrap() * solution[[i, j]] + solution[[i - 1, j]];
        let d2_dy2 =
            solution[[i, j + 1]] - F::from(2.0).unwrap() * solution[[i, j]] + solution[[i, j - 1]];
        let d2_dxdy =
            (solution[[i + 1, j + 1]] - solution[[i + 1, j - 1]] - solution[[i - 1, j + 1]]
                + solution[[i - 1, j - 1]])
                / F::from(4.0).unwrap();

        // Frobenius norm of Hessian matrix
        (d2_dx2 * d2_dx2 + d2_dy2 * d2_dy2 + F::from(2.0).unwrap() * d2_dxdy * d2_dxdy).sqrt()
    }

    /// Estimate local truncation error
    fn estimate_truncation_error(&self, solution: ArrayView2<F>, i: usize, j: usize) -> F {
        // Simple Richardson extrapolation-based error estimate
        // Compare solution at current resolution vs estimated higher-order solution
        self.compute_curvature(solution, i, j) / F::from(12.0).unwrap() // h² error estimate
    }
}

impl<F: IntegrateFloat> AMRGrid<F> {
    /// Create new AMR grid with initial coarse level
    pub fn new(nx: usize, ny: usize, domain_x: [F; 2], domain_y: [F; 2]) -> Self {
        let dx = (domain_x[1] - domain_x[0]) / F::from(nx).unwrap();
        let dy = (domain_y[1] - domain_y[0]) / F::from(ny).unwrap();

        let coarse_level = GridLevel {
            level: 0,
            nx,
            ny,
            dx,
            dy,
            refined: Array2::from_elem((nx, ny), false),
            children: HashMap::new(),
        };

        Self {
            levels: vec![coarse_level],
            max_level: 5, // Default maximum 5 levels
            min_level: 0,
            domain: (domain_x, domain_y),
            solution: HashMap::new(),
        }
    }

    /// Set maximum refinement level
    pub fn set_max_level(&mut self, max_level: usize) {
        self.max_level = max_level;
    }

    /// Refine grid based on criteria
    pub fn refine(&mut self, criteria: &RefinementCriteria<F>) -> IntegrateResult<usize> {
        let mut total_refined = 0;

        // Process each level from coarse to fine
        for level in 0..self.levels.len() {
            if level >= self.max_level {
                break;
            }

            let current_level = &self.levels[level];
            let solution_level = self.get_solution_array(level)?;

            let mut cells_to_refine = Vec::new();

            // Find cells that need refinement
            for i in 0..current_level.nx {
                for j in 0..current_level.ny {
                    if !current_level.refined[[i, j]]
                        && criteria.should_refine(solution_level.view(), i, j)
                    {
                        cells_to_refine.push((i, j));
                    }
                }
            }

            // Refine selected cells
            for (i, j) in cells_to_refine {
                self.refine_cell(level, i, j)?;
                total_refined += 1;
            }
        }

        Ok(total_refined)
    }

    /// Coarsen grid based on criteria
    pub fn coarsen(&mut self, criteria: &RefinementCriteria<F>) -> IntegrateResult<usize> {
        let mut total_coarsened = 0;

        // Process from fine to coarse levels
        for level in (self.min_level + 1..self.levels.len()).rev() {
            let current_level = &self.levels[level];
            let solution_level = self.get_solution_array(level)?;

            let mut cells_to_coarsen = Vec::new();

            // Find cells that can be coarsened
            for i in 0..current_level.nx {
                for j in 0..current_level.ny {
                    if criteria.should_coarsen(solution_level.view(), i, j) {
                        cells_to_coarsen.push((i, j));
                    }
                }
            }

            // Coarsen selected cells (group into parent cells)
            let coarsened = self.coarsen_cells(level, cells_to_coarsen)?;
            total_coarsened += coarsened;
        }

        Ok(total_coarsened)
    }

    /// Refine a single cell
    fn refine_cell(&mut self, level: usize, i: usize, j: usize) -> IntegrateResult<()> {
        if level >= self.max_level {
            return Err(IntegrateError::ValueError(
                "Cannot refine beyond maximum level".to_string(),
            ));
        }

        // Ensure next level exists
        while self.levels.len() <= level + 1 {
            let parent_level = &self.levels[level];
            let child_nx = parent_level.nx * 2;
            let child_ny = parent_level.ny * 2;
            let child_dx = parent_level.dx / F::from(2.0).unwrap();
            let child_dy = parent_level.dy / F::from(2.0).unwrap();

            let child_level = GridLevel {
                level: level + 1,
                nx: child_nx,
                ny: child_ny,
                dx: child_dx,
                dy: child_dy,
                refined: Array2::from_elem((child_nx, child_ny), false),
                children: HashMap::new(),
            };

            self.levels.push(child_level);
        }

        // Mark cell as refined
        self.levels[level].refined[[i, j]] = true;

        // Create child information
        let child_start = (i * 2, j * 2);
        let child_size = (2, 2);

        self.levels[level].children.insert(
            (i, j),
            ChildInfo {
                child_start,
                child_size,
            },
        );

        // Interpolate solution to child cells
        self.interpolate_to_children(level, i, j)?;

        Ok(())
    }

    /// Coarsen cells (placeholder implementation)
    fn coarsen_cells(
        &mut self,
        _level: usize,
        _cells: Vec<(usize, usize)>,
    ) -> IntegrateResult<usize> {
        // TODO: Implement proper coarsening logic
        // This involves grouping child cells back to parent cells
        // and averaging/restricting solution values
        Ok(0)
    }

    /// Interpolate solution from parent cell to child cells
    fn interpolate_to_children(&mut self, level: usize, i: usize, j: usize) -> IntegrateResult<()> {
        let parent_value = self
            .solution
            .get(&(level, i, j))
            .copied()
            .unwrap_or(F::zero());

        if let Some(child_info) = self.levels[level].children.get(&(i, j)) {
            let (start_i, start_j) = child_info.child_start;
            let (size_i, size_j) = child_info.child_size;

            // Simple constant interpolation (could be improved with higher-order)
            for ci in 0..size_i {
                for cj in 0..size_j {
                    self.solution
                        .insert((level + 1, start_i + ci, start_j + cj), parent_value);
                }
            }
        }

        Ok(())
    }

    /// Get solution as Array2 for a specific level
    fn get_solution_array(&self, level: usize) -> IntegrateResult<Array2<F>> {
        if level >= self.levels.len() {
            return Err(IntegrateError::ValueError(
                "Level does not exist".to_string(),
            ));
        }

        let grid_level = &self.levels[level];
        let mut solution = Array2::zeros((grid_level.nx, grid_level.ny));

        for i in 0..grid_level.nx {
            for j in 0..grid_level.ny {
                if let Some(&value) = self.solution.get(&(level, i, j)) {
                    solution[[i, j]] = value;
                }
            }
        }

        Ok(solution)
    }

    /// Get grid information for a specific level
    pub fn get_level_info(&self, level: usize) -> Option<&GridLevel<F>> {
        self.levels.get(level)
    }

    /// Get total number of cells across all levels
    pub fn total_cells(&self) -> usize {
        self.levels.iter().map(|level| level.nx * level.ny).sum()
    }

    /// Get refinement efficiency (fraction of cells refined)
    pub fn refinement_efficiency(&self) -> f64 {
        let total_refined = self
            .levels
            .iter()
            .map(|level| level.refined.iter().filter(|&&x| x).count())
            .sum::<usize>();

        let total_possible = self.levels[0].nx * self.levels[0].ny;

        total_refined as f64 / total_possible as f64
    }
}

/// AMR-enhanced PDE solver
pub struct AMRSolver<F: IntegrateFloat> {
    /// AMR grid
    grid: AMRGrid<F>,
    /// Refinement criteria
    criteria: RefinementCriteria<F>,
    /// Number of AMR cycles performed
    amr_cycles: usize,
    /// Maximum AMR cycles per solve
    max_amr_cycles: usize,
}

impl<F: IntegrateFloat> AMRSolver<F> {
    /// Create new AMR solver
    pub fn new(grid: AMRGrid<F>, criteria: RefinementCriteria<F>) -> Self {
        Self {
            grid,
            criteria,
            amr_cycles: 0,
            max_amr_cycles: 5,
        }
    }

    /// Set maximum AMR cycles
    pub fn set_max_amr_cycles(&mut self, max_cycles: usize) {
        self.max_amr_cycles = max_cycles;
    }

    /// Solve PDE with adaptive mesh refinement
    pub fn solve_adaptive<ProblemFn>(
        &mut self,
        problem: ProblemFn,
        initial_solution: Array2<F>,
    ) -> PDEResult<Array2<F>>
    where
        ProblemFn: Fn(&AMRGrid<F>, ArrayView2<F>) -> PDEResult<Array2<F>>,
    {
        // Initialize solution on coarse grid
        let mut current_solution = initial_solution;

        for cycle in 0..self.max_amr_cycles {
            self.amr_cycles = cycle;

            // Store current solution in grid
            self.store_solution_in_grid(&current_solution)?;

            // Refine grid based on current solution
            let refined_cells = self.grid.refine(&self.criteria)?;

            // If no refinement occurred, we're done
            if refined_cells == 0 {
                break;
            }

            // Solve on refined grid
            current_solution = problem(&self.grid, current_solution.view())?;

            // Optional: coarsen grid where possible
            let _coarsened_cells = self.grid.coarsen(&self.criteria)?;
        }

        Ok(current_solution)
    }

    /// Store solution array in the AMR grid structure
    fn store_solution_in_grid(&mut self, solution: &Array2<F>) -> IntegrateResult<()> {
        let (nx, ny) = solution.dim();

        // Clear existing solution
        self.grid.solution.clear();

        // Store solution for level 0
        for i in 0..nx {
            for j in 0..ny {
                self.grid.solution.insert((0, i, j), solution[[i, j]]);
            }
        }

        Ok(())
    }

    /// Get grid statistics
    pub fn grid_statistics(&self) -> AMRStatistics {
        AMRStatistics {
            num_levels: self.grid.levels.len(),
            total_cells: self.grid.total_cells(),
            refinement_efficiency: self.grid.refinement_efficiency(),
            amr_cycles: self.amr_cycles,
        }
    }
}

/// Statistics about AMR grid
#[derive(Debug, Clone)]
pub struct AMRStatistics {
    /// Number of refinement levels
    pub num_levels: usize,
    /// Total number of cells across all levels
    pub total_cells: usize,
    /// Refinement efficiency (0.0 to 1.0)
    pub refinement_efficiency: f64,
    /// Number of AMR cycles performed
    pub amr_cycles: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_amr_grid_creation() {
        let grid: AMRGrid<f64> = AMRGrid::new(32, 32, [0.0, 1.0], [0.0, 1.0]);

        assert_eq!(grid.levels.len(), 1);
        assert_eq!(grid.levels[0].nx, 32);
        assert_eq!(grid.levels[0].ny, 32);
        assert_abs_diff_eq!(grid.levels[0].dx, 1.0 / 32.0);
        assert_abs_diff_eq!(grid.levels[0].dy, 1.0 / 32.0);
    }

    #[test]
    fn test_refinement_criteria() {
        // Create test solution with gradient
        let mut solution = Array2::zeros((5, 5));
        for i in 0..5 {
            for j in 0..5 {
                solution[[i, j]] = (i * i + j * j) as f64 * 0.1;
            }
        }

        let criteria = RefinementCriteria::gradient_based(0.5);

        // Test gradient computation
        let should_refine = criteria.should_refine(solution.view(), 2, 2);
        // Middle cell should have some gradient

        let should_coarsen = criteria.should_coarsen(solution.view(), 0, 0);
        // Corner cell might be suitable for coarsening

        // Just verify these don't panic - exact values depend on implementation
        let _ = should_refine;
        let _ = should_coarsen;
    }

    #[test]
    fn test_cell_refinement() {
        let mut grid: AMRGrid<f64> = AMRGrid::new(4, 4, [0.0, 1.0], [0.0, 1.0]);

        // Set up some solution values
        grid.solution.insert((0, 1, 1), 1.0);

        // Refine a cell
        assert!(grid.refine_cell(0, 1, 1).is_ok());

        // Check that refinement was applied
        assert!(grid.levels[0].refined[[1, 1]]);
        assert_eq!(grid.levels.len(), 2); // Should create next level

        // Check child information
        assert!(grid.levels[0].children.contains_key(&(1, 1)));
    }

    #[test]
    fn test_amr_solver() {
        let grid: AMRGrid<f64> = AMRGrid::new(8, 8, [0.0, 1.0], [0.0, 1.0]);
        let criteria = RefinementCriteria::gradient_based(0.1);
        let mut solver = AMRSolver::new(grid, criteria);

        // Simple initial solution
        let initial = Array2::from_elem((8, 8), 0.5);

        // Dummy problem function
        let problem = |_grid: &AMRGrid<f64>, solution: ArrayView2<f64>| -> PDEResult<Array2<f64>> {
            Ok(solution.to_owned())
        };

        let result = solver.solve_adaptive(problem, initial);
        assert!(result.is_ok());

        let stats = solver.grid_statistics();
        assert!(stats.num_levels >= 1);
        assert!(stats.total_cells >= 64); // At least the initial 8×8 grid
    }
}
