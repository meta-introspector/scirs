//! Symbolic integration support for enhanced numerical methods
//!
//! This module provides symbolic manipulation capabilities that enhance
//! the numerical integration methods, including:
//! - Automatic Jacobian generation using symbolic differentiation
//! - Higher-order ODE to first-order system conversion
//! - Conservation law detection and enforcement
//! - Symbolic simplification for performance optimization

pub mod jacobian;
pub mod conversion;
pub mod conservation;
pub mod expression;

// Re-export main types and functions
pub use jacobian::{SymbolicJacobian, generate_jacobian};
pub use conversion::{higher_order_to_first_order, HigherOrderODE, FirstOrderSystem};
pub use conservation::{ConservationLaw, detect_conservation_laws, ConservationEnforcer};
pub use expression::{SymbolicExpression, Variable, simplify};