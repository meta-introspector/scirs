//! Rigid registration algorithms
//!
//! This module provides rigid transformation registration (translation + rotation only).

use crate::error::Result;
use crate::registration::{RegistrationResult, RegistrationParams};

/// Rigid registration using point matches
pub fn register_rigid_points(
    _source_points: &[(f64, f64)],
    _target_points: &[(f64, f64)],
    _params: &RegistrationParams,
) -> Result<RegistrationResult> {
    // TODO: Implement rigid point registration
    todo!("Rigid point registration not yet implemented")
}