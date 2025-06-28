//! Multivariate statistical analysis methods
//!
//! This module provides implementations of multivariate analysis techniques including:
//! - Principal Component Analysis (PCA)
//! - Factor Analysis (EFA with EM algorithm, varimax/promax rotation)
//! - Linear and Quadratic Discriminant Analysis (LDA/QDA)
//! - Canonical Correlation Analysis (CCA)
//! - Partial Least Squares (PLS)

mod canonical_correlation;
mod discriminant_analysis;
mod factor_analysis;
mod pca;

pub use canonical_correlation::*;
pub use discriminant_analysis::*;
pub use factor_analysis::*;
pub use pca::*;

#[allow(unused_imports)]
use crate::error::StatsResult as Result;
#[allow(unused_imports)]
use scirs2_core::{simd_ops::SimdUnifiedOps, validation::*};
