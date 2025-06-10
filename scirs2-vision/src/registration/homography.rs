//! Homography registration algorithms

use crate::error::Result;
use crate::registration::{RegistrationParams, RegistrationResult};

/// Homography registration placeholder
pub fn register_homography_points(
    _source_points: &[(f64, f64)],
    _target_points: &[(f64, f64)],
    _params: &RegistrationParams,
) -> Result<RegistrationResult> {
    todo!("Homography registration not yet implemented")
}
