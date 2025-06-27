//! Enhanced automatic differentiation for numerical integration
//!
//! This module provides advanced automatic differentiation capabilities including:
//! - Forward mode AD for efficient gradient computation
//! - Reverse mode AD for efficient Jacobian computation
//! - Sparse Jacobian optimization
//! - Sensitivity analysis tools

pub mod forward;
pub mod reverse;
pub mod sparse;
pub mod sensitivity;
pub mod dual;

// Re-export main types and functions
pub use forward::{ForwardAD, forward_gradient, forward_jacobian, VectorizedForwardAD, ForwardODEJacobian};
pub use reverse::{ReverseAD, reverse_gradient, reverse_jacobian, TapeNode, Tape, CheckpointStrategy};
pub use sparse::{SparsePattern, SparseJacobian, detect_sparsity, compress_jacobian, 
                 CSRJacobian, CSCJacobian, ColGrouping, colored_jacobian, detect_sparsity_adaptive,
                 SparseJacobianUpdater, BlockPattern, HybridJacobian};
pub use sensitivity::{SensitivityAnalysis, ParameterSensitivity, compute_sensitivities,
                      SobolSensitivity, MorrisScreening, EFAST};
pub use dual::{Dual, DualVector};