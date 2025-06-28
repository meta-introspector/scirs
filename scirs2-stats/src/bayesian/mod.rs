//! Bayesian statistical methods
//!
//! This module provides implementations of Bayesian statistical techniques including:
//! - Conjugate priors
//! - Bayesian linear regression
//! - Hierarchical models
//! - Variational inference

mod conjugate;
mod hierarchical;
mod regression;
mod variational;

pub use conjugate::*;
pub use hierarchical::*;
pub use regression::*;
pub use variational::*;

#[allow(unused_imports)]
use crate::error::StatsResult as Result;
#[allow(unused_imports)]
use scirs2_core::{simd_ops::SimdUnifiedOps, validation::*};
