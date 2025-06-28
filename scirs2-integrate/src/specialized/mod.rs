//! Specialized solvers for domain-specific problems
//!
//! This module provides optimized solvers for specific scientific domains:
//! - Quantum mechanics (Schrödinger equation)
//! - Fluid dynamics (Navier-Stokes)
//! - Financial modeling (stochastic PDEs)

pub mod quantum;
pub mod fluid_dynamics;
pub mod finance;

pub use quantum::{
    SchrodingerSolver, QuantumState, QuantumPotential, HarmonicOscillator, 
    ParticleInBox, HydrogenAtom, SchrodingerMethod
};
pub use quantum::quantum_algorithms::{
    QuantumAnnealer, VariationalQuantumEigensolver, MultiBodyQuantumSolver
};
pub use fluid_dynamics::{
    NavierStokesSolver, FluidState, FluidBoundaryCondition, NavierStokesParams
};
pub use fluid_dynamics::turbulence_models::{
    LESolver, RANSSolver, FluidState3D, RANSState, SGSModel, RANSModel
};
pub use finance::{
    StochasticPDESolver, FinancialOption, VolatilityModel, OptionType, 
    OptionStyle, FinanceMethod, JumpProcess, Greeks
};