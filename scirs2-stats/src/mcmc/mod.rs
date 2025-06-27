//! Markov Chain Monte Carlo (MCMC) methods
//!
//! This module provides implementations of MCMC algorithms for sampling from
//! complex probability distributions including:
//! - Metropolis-Hastings
//! - Gibbs sampling
//! - Hamiltonian Monte Carlo

mod metropolis;

pub use metropolis::*;

#[allow(unused_imports)]
use crate::error::StatsResult as Result;
#[allow(unused_imports)]
use scirs2_core::{simd_ops::SimdUnifiedOps, validation::*};