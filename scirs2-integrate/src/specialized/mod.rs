//! Specialized solvers for domain-specific problems
//!
//! This module provides optimized solvers for specific scientific domains:
//! - Quantum mechanics (Schrödinger equation)
//! - Fluid dynamics (Navier-Stokes)
//! - Financial modeling (stochastic PDEs)

pub mod finance;
pub mod fluid_dynamics;
pub mod quantum;

pub use finance::{
    FinanceMethod, FinancialOption, Greeks, JumpProcess, OptionStyle, OptionType,
    StochasticPDESolver, VolatilityModel,
};
pub use fluid_dynamics::turbulence_models::{
    FluidState3D, LESolver, RANSModel, RANSSolver, RANSState, SGSModel,
};
pub use fluid_dynamics::{
    FluidBoundaryCondition, FluidState, NavierStokesParams, NavierStokesSolver,
};
pub use quantum::quantum_algorithms::{
    MultiBodyQuantumSolver, QuantumAnnealer, VariationalQuantumEigensolver,
};
pub use quantum::{
    HarmonicOscillator, HydrogenAtom, ParticleInBox, QuantumPotential, QuantumState,
    SchrodingerMethod, SchrodingerSolver,
};
