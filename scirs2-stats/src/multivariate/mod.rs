//! Multivariate statistical analysis methods
//!
//! This module provides implementations of multivariate analysis techniques including:
//! - Principal Component Analysis (PCA)
//! - Factor Analysis
//! - Discriminant Analysis

mod pca;

pub use pca::*;

#[allow(unused_imports)]
use crate::error::StatsResult as Result;
#[allow(unused_imports)]
use scirs2_core::{simd_ops::SimdUnifiedOps, validation::*};