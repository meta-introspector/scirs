//! Affine registration algorithms

use crate::error::Result;
use crate::registration::{RegistrationParams, RegistrationResult};

/// Affine registration placeholder
pub fn register_affine_points(
    _source_points: &[(f64, f64)],
    _target_points: &[(f64, f64)],
    _params: &RegistrationParams,
) -> Result<RegistrationResult> {
    todo!("Affine registration not yet implemented")
}
