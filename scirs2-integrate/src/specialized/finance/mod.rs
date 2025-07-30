//! Financial modeling solvers for stochastic PDEs
//!
//! This module provides specialized solvers for quantitative finance applications,
//! including Black-Scholes, stochastic volatility models, and jump-diffusion processes.

pub mod types;
pub mod models;
pub mod pricing;
pub mod derivatives;
pub mod risk;
pub mod solvers;
pub mod ml;
pub mod utils;

// Re-export commonly used types
pub use types::*;
pub use models::*;
pub use solvers::stochastic_pde::StochasticPDESolver;
pub use risk::greeks::Greeks;

// Re-export pricing methods
pub use pricing::{
    black_scholes::*,
    monte_carlo::*,
    finite_difference::*,
};

// Re-export main solver for backwards compatibility
pub use solvers::StochasticPDESolver as FinancialPDESolver;