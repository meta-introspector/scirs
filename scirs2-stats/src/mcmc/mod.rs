//! Markov Chain Monte Carlo (MCMC) methods
//!
//! This module provides implementations of MCMC algorithms for sampling from
//! complex probability distributions including:
//! - Metropolis-Hastings
//! - Gibbs sampling
//! - Hamiltonian Monte Carlo
//! - Advanced methods (Multiple-try Metropolis, Parallel Tempering, Slice Sampling, Ensemble Methods)

mod metropolis;
mod gibbs;
mod hamiltonian;
mod advanced;

pub use metropolis::*;
pub use gibbs::*;
pub use hamiltonian::*;
pub use advanced::*;

#[allow(unused_imports)]
use crate::error::StatsResult as Result;
#[allow(unused_imports)]
use scirs2_core::{simd_ops::SimdUnifiedOps, validation::*};