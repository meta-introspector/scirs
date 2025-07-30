//! Quantum mechanics solvers for the Schrödinger equation
//!
//! This module provides specialized solvers for quantum mechanical systems,
//! including time-dependent and time-independent Schrödinger equations.
//! The implementation has been modularized for better organization and maintainability.
//!
//! ## Modules
//!
//! - [`core`] - Core quantum state representations and basic solvers
//! - [`algorithms`] - Advanced quantum algorithms (annealing, VQE, error correction)
//! - [`entanglement`] - Multi-particle entanglement and Bell states
//! - [`basis_sets`] - Advanced basis sets for quantum calculations  
//! - [`gpu`] - GPU-accelerated quantum computations

// Import all modular components
mod quantum;

// Re-export all components for backward compatibility
pub use quantum::*;

use crate::error::{IntegrateError, IntegrateResult as Result};
use ndarray::{Array1, Array2};
use num_complex::Complex64;
use scirs2_core::constants::PI;

/// High-level quantum system coordinator
#[derive(Debug)]
pub struct QuantumSystemCoordinator {
    /// Core Schrödinger solver
    pub schrodinger_solver: Option<quantum::SchrodingerSolver>,
    /// Quantum algorithms handler
    pub algorithms: QuantumAlgorithmsHandler,
    /// Entanglement system
    pub entanglement: Option<quantum::MultiParticleEntanglement>,
    /// Basis set manager
    pub basis_sets: Option<quantum::AdvancedBasisSets>,
    /// GPU acceleration
    pub gpu_solver: Option<quantum::GPUQuantumSolver>,
}

impl QuantumSystemCoordinator {
    /// Create new quantum system coordinator
    pub fn new() -> Self {
        Self {
            schrodinger_solver: None,
            algorithms: QuantumAlgorithmsHandler::new(),
            entanglement: None,
            basis_sets: None,
            gpu_solver: None,
        }
    }

    /// Initialize Schrödinger solver
    pub fn with_schrodinger_solver(
        mut self,
        n_points: usize,
        dt: f64,
        potential: Box<dyn quantum::QuantumPotential>,
        method: quantum::SchrodingerMethod,
    ) -> Self {
        self.schrodinger_solver = Some(quantum::SchrodingerSolver::new(n_points, dt, potential, method));
        self
    }

    /// Initialize entanglement system
    pub fn with_entanglement_system(mut self, n_particles: usize, masses: Array1<f64>) -> Self {
        self.entanglement = Some(quantum::MultiParticleEntanglement::new(n_particles, masses));
        self
    }

    /// Initialize basis sets
    pub fn with_basis_sets(mut self, n_basis: usize, basis_type: quantum::BasisSetType) -> Self {
        self.basis_sets = Some(quantum::AdvancedBasisSets::new(n_basis, basis_type));
        self
    }

    /// Initialize GPU solver
    pub fn with_gpu_solver(mut self, device_id: usize, max_qubits: usize) -> Result<Self> {
        self.gpu_solver = Some(quantum::GPUQuantumSolver::new(device_id, max_qubits)?);
        Ok(self)
    }

    /// Solve quantum evolution problem
    pub fn solve_evolution(
        &self,
        initial_state: &quantum::QuantumState,
        t_final: f64,
    ) -> Result<Vec<quantum::QuantumState>> {
        if let Some(ref solver) = self.schrodinger_solver {
            solver.solve_time_dependent(initial_state, t_final)
        } else {
            Err(IntegrateError::InvalidInput(
                "Schrödinger solver not initialized".to_string(),
            ))
        }
    }

    /// Solve ground state problem
    pub fn solve_ground_state(
        &self,
        x_min: f64,
        x_max: f64,
        n_states: usize,
    ) -> Result<(Array1<f64>, Array2<f64>)> {
        if let Some(ref solver) = self.schrodinger_solver {
            solver.solve_time_independent(x_min, x_max, n_states)
        } else {
            Err(IntegrateError::InvalidInput(
                "Schrödinger solver not initialized".to_string(),
            ))
        }
    }

    /// Run quantum algorithm
    pub fn run_quantum_algorithm(&mut self, algorithm_type: QuantumAlgorithmType) -> Result<QuantumAlgorithmResult> {
        self.algorithms.run_algorithm(algorithm_type)
    }

    /// Create and manipulate entangled states
    pub fn create_entangled_state(&mut self, entanglement_type: EntanglementType) -> Result<()> {
        if let Some(ref mut entanglement) = self.entanglement {
            match entanglement_type {
                EntanglementType::Bell(bell_type) => entanglement.create_bell_state(bell_type),
                EntanglementType::GHZ => entanglement.create_ghz_state(),
                EntanglementType::W => entanglement.create_w_state(),
            }
        } else {
            Err(IntegrateError::InvalidInput(
                "Entanglement system not initialized".to_string(),
            ))
        }
    }
}

impl Default for QuantumSystemCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

/// Quantum algorithms handler
#[derive(Debug)]
pub struct QuantumAlgorithmsHandler {
    /// Available algorithms
    algorithms: Vec<String>,
}

impl QuantumAlgorithmsHandler {
    /// Create new algorithms handler
    pub fn new() -> Self {
        Self {
            algorithms: vec![
                "quantum_annealing".to_string(),
                "vqe".to_string(),
                "error_correction".to_string(),
            ],
        }
    }

    /// Run specified algorithm
    pub fn run_algorithm(&mut self, algorithm_type: QuantumAlgorithmType) -> Result<QuantumAlgorithmResult> {
        match algorithm_type {
            QuantumAlgorithmType::QuantumAnnealing { n_qubits, annealing_time } => {
                let annealer = quantum::QuantumAnnealer::new(n_qubits, annealing_time, 100);
                let j_matrix = Array2::zeros((n_qubits, n_qubits));
                let h_fields = Array1::zeros(n_qubits);
                let (spins, energy) = annealer.solve_ising(&j_matrix, &h_fields)?;
                Ok(QuantumAlgorithmResult::AnnealingResult { spins, energy })
            }
            QuantumAlgorithmType::VQE { n_qubits, circuit_depth } => {
                let vqe = quantum::VariationalQuantumEigensolver::new(n_qubits, circuit_depth);
                let hamiltonian = Array2::eye(1 << n_qubits);
                let (energy, params) = vqe.find_ground_state(&hamiltonian)?;
                Ok(QuantumAlgorithmResult::VQEResult { energy, parameters: params })
            }
            QuantumAlgorithmType::ErrorCorrection { n_logical_qubits, code } => {
                let qec = quantum::QuantumErrorCorrection::new(n_logical_qubits, code);
                let error_rate = qec.estimate_logical_error_rate();
                Ok(QuantumAlgorithmResult::ErrorCorrectionResult { logical_error_rate: error_rate })
            }
        }
    }

    /// Get available algorithms
    pub fn available_algorithms(&self) -> &[String] {
        &self.algorithms
    }
}

impl Default for QuantumAlgorithmsHandler {
    fn default() -> Self {
        Self::new()
    }
}

/// Types of quantum algorithms
#[derive(Debug, Clone)]
pub enum QuantumAlgorithmType {
    /// Quantum annealing optimization
    QuantumAnnealing {
        n_qubits: usize,
        annealing_time: f64,
    },
    /// Variational quantum eigensolver
    VQE {
        n_qubits: usize,
        circuit_depth: usize,
    },
    /// Quantum error correction
    ErrorCorrection {
        n_logical_qubits: usize,
        code: quantum::ErrorCorrectionCode,
    },
}

/// Results from quantum algorithms
#[derive(Debug, Clone)]
pub enum QuantumAlgorithmResult {
    /// Quantum annealing result
    AnnealingResult {
        spins: Array1<i8>,
        energy: f64,
    },
    /// VQE result
    VQEResult {
        energy: f64,
        parameters: Array1<f64>,
    },
    /// Error correction result
    ErrorCorrectionResult {
        logical_error_rate: f64,
    },
}

/// Types of entanglement
#[derive(Debug, Clone)]
pub enum EntanglementType {
    /// Bell state entanglement
    Bell(quantum::BellState),
    /// GHZ state
    GHZ,
    /// W state
    W,
}

/// Convenience functions for common quantum operations
pub mod convenience {
    use super::*;

    /// Create a simple harmonic oscillator system
    pub fn harmonic_oscillator_system(
        k: f64,
        x0: f64,
        n_points: usize,
        dt: f64,
    ) -> QuantumSystemCoordinator {
        let potential = Box::new(quantum::HarmonicOscillator { k, x0 });
        
        QuantumSystemCoordinator::new()
            .with_schrodinger_solver(n_points, dt, potential, quantum::SchrodingerMethod::SplitOperator)
    }

    /// Create a particle in a box system
    pub fn particle_in_box_system(
        left: f64,
        right: f64,
        barrier_height: f64,
        n_points: usize,
        dt: f64,
    ) -> QuantumSystemCoordinator {
        let potential = Box::new(quantum::ParticleInBox {
            left,
            right,
            barrier_height,
        });
        
        QuantumSystemCoordinator::new()
            .with_schrodinger_solver(n_points, dt, potential, quantum::SchrodingerMethod::CrankNicolson)
    }

    /// Create a hydrogen atom system
    pub fn hydrogen_atom_system(
        z: f64,
        e2_4pi_eps0: f64,
        n_points: usize,
        dt: f64,
    ) -> QuantumSystemCoordinator {
        let potential = Box::new(quantum::HydrogenAtom { z, e2_4pi_eps0 });
        
        QuantumSystemCoordinator::new()
            .with_schrodinger_solver(n_points, dt, potential, quantum::SchrodingerMethod::RungeKutta4)
    }

    /// Create entangled Bell pair
    pub fn create_bell_pair(bell_type: quantum::BellState) -> Result<quantum::MultiParticleEntanglement> {
        let masses = Array1::from_vec(vec![1.0, 1.0]);
        let mut system = quantum::MultiParticleEntanglement::new(2, masses);
        system.create_bell_state(bell_type)?;
        Ok(system)
    }

    /// Create GHZ state for n particles
    pub fn create_ghz_state(n_particles: usize) -> Result<quantum::MultiParticleEntanglement> {
        let masses = Array1::ones(n_particles);
        let mut system = quantum::MultiParticleEntanglement::new(n_particles, masses);
        system.create_ghz_state()?;
        Ok(system)
    }

    /// Run simple quantum annealing optimization
    pub fn run_quantum_annealing(
        n_qubits: usize,
        j_matrix: &Array2<f64>,
        h_fields: &Array1<f64>,
    ) -> Result<(Array1<i8>, f64)> {
        let annealer = quantum::QuantumAnnealer::new(n_qubits, 10.0, 100);
        annealer.solve_ising(j_matrix, h_fields)
    }

    /// Run VQE for ground state finding
    pub fn run_vqe_ground_state(
        hamiltonian: &Array2<Complex64>,
        n_qubits: usize,
        circuit_depth: usize,
    ) -> Result<(f64, Array1<f64>)> {
        let vqe = quantum::VariationalQuantumEigensolver::new(n_qubits, circuit_depth);
        vqe.find_ground_state(hamiltonian)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_quantum_system_coordinator() {
        let coordinator = QuantumSystemCoordinator::new();
        assert!(coordinator.schrodinger_solver.is_none());
        assert!(coordinator.entanglement.is_none());
    }

    #[test]
    fn test_convenience_functions() {
        let system = convenience::harmonic_oscillator_system(1.0, 0.0, 100, 0.01);
        assert!(system.schrodinger_solver.is_some());

        let bell_pair = convenience::create_bell_pair(quantum::BellState::PhiPlus);
        assert!(bell_pair.is_ok());
    }

    #[test]
    fn test_algorithms_handler() {
        let mut handler = QuantumAlgorithmsHandler::new();
        assert!(!handler.available_algorithms().is_empty());

        let algorithm = QuantumAlgorithmType::QuantumAnnealing {
            n_qubits: 4,
            annealing_time: 10.0,
        };

        let result = handler.run_algorithm(algorithm);
        assert!(result.is_ok());
    }
}