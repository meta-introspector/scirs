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

pub use lie_group::{
    LieGroupIntegrator, LieAlgebra, ExponentialMap, LieGroupMethod,
    SO3, SO3Integrator, So3, 
    SE3, SE3Integrator, Se3,
    SLn, Sln, GLn, Gln, Sp2n, 
    HeisenbergGroup, HeisenbergAlgebra
};
pub use volume_preserving::{
    VolumePreservingIntegrator, VolumePreservingMethod, DivergenceFreeFlow, IncompressibleFlow,
    CircularFlow2D, ABCFlow, DoubleGyre, VolumeChecker,
    StreamFunction, StuartVortex, TaylorGreenVortex, HamiltonianFlow,
    ModifiedMidpointIntegrator, VariationalIntegrator, DiscreteGradientIntegrator
};
pub use structure_preserving::{
    StructurePreservingIntegrator, StructurePreservingMethod, ConservationChecker, GeometricInvariant,
    EnergyPreservingMethod, MomentumPreservingMethod,
    SplittingIntegrator, EnergyMomentumIntegrator, ConstrainedIntegrator, MultiSymplecticIntegrator,
    invariants::{EnergyInvariant, LinearMomentumInvariant, AngularMomentumInvariant2D}
};