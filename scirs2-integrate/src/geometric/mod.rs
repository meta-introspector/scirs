//! Geometric integration methods
//!
//! This module provides structure-preserving numerical integrators for systems
//! with geometric properties such as:
//! - Lie group structure
//! - Volume preservation (divergence-free flows)
//! - Energy conservation (Hamiltonian systems)
//! - Momentum conservation (Lagrangian systems)

pub mod lie_group;
pub mod volume_preserving;
pub mod structure_preserving;

pub use lie_group::{LieGroupIntegrator, LieAlgebra, ExponentialMap, SO3Integrator, SE3Integrator};
pub use volume_preserving::{VolumePreservingIntegrator, DivergenceFreeFlow, IncompressibleFlow};
pub use structure_preserving::{
    StructurePreservingIntegrator, ConservationChecker, GeometricInvariant,
    EnergyPreservingMethod, MomentumPreservingMethod
};