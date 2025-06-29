//! Ecosystem management and validation for SciRS2
//!
//! This module provides comprehensive ecosystem management capabilities
//! including validation, compatibility checking, and ecosystem health
//! monitoring for production 1.0 deployments.

pub mod validation;

pub use validation::*;

/// Initialize the ecosystem management system
pub fn initialize_ecosystem() -> crate::error::CoreResult<()> {
    validation::initialize_ecosystem_validation()?;
    Ok(())
}

/// Quick ecosystem health check
pub fn quick_health_check() -> crate::error::CoreResult<validation::EcosystemHealth> {
    let validator = validation::EcosystemValidator::global()?;
    validator.get_ecosystem_health()
}

/// Get ecosystem validation summary
pub fn get_validation_summary() -> crate::error::CoreResult<validation::EcosystemValidationResult> {
    let validator = validation::EcosystemValidator::global()?;
    validator.validate_ecosystem()
}