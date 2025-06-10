//! Non-rigid (deformable) registration algorithms

use crate::error::Result;
use crate::registration::{RegistrationResult, RegistrationParams};

/// Non-rigid registration placeholder
pub fn register_non_rigid_points(
    _source_points: &[(f64, f64)],
    _target_points: &[(f64, f64)],
    _params: &RegistrationParams,
) -> Result<RegistrationResult> {
    todo!("Non-rigid registration not yet implemented")
}