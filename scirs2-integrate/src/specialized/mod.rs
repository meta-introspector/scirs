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
pub use fluid_dynamics::{
    NavierStokesSolver, FluidState, FluidBoundaryCondition, NavierStokesParams
};
pub use finance::{
    StochasticPDESolver, FinancialOption, VolatilityModel, OptionType, 
    OptionStyle, FinanceMethod, JumpProcess, Greeks
};