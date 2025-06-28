//! Linear algebra operations for sparse matrices
//!
//! This module provides linear algebra operations for sparse matrices,
//! including solvers, eigenvalue computations, and matrix functions.

mod cgs;
mod eigen;
mod expm;
mod ic;
mod interface;
mod iterative;
mod lgmres;
mod matfuncs;
mod minres;
mod preconditioners;
mod qmr;
mod qmr_simple;
mod solvers;
mod spai;
mod svd;
mod decomposition;

pub use cgs::{cgs, CGSOptions, CGSResult};
pub use eigen::{
    power_iteration, lanczos, eigs, eigsh, PowerIterationOptions, LanczosOptions, EigenResult,
    EigenvalueMethod, ArpackOptions,
};
pub use expm::expm;
pub use ic::IC0Preconditioner;
pub use interface::{
    AsLinearOperator, DiagonalOperator, IdentityOperator, LinearOperator, ScaledIdentityOperator,
};
pub use iterative::{
    bicg, bicgstab, cg, gmres, BiCGOptions, BiCGSTABOptions, BiCGSTABResult, CGOptions,
    GMRESOptions, IterationResult, IterativeSolver,
};
pub use lgmres::{lgmres, LGMRESOptions, LGMRESResult};
pub use matfuncs::{expm_multiply, onenormest};
pub use minres::{minres, MINRESOptions, MINRESResult};
pub use preconditioners::{ILU0Preconditioner, JacobiPreconditioner, SSORPreconditioner};
pub use qmr::{qmr, QMROptions, QMRResult};
pub use solvers::{
    add, diag_matrix, eye, inv, matmul, matrix_power, multiply, norm, sparse_direct_solve,
    sparse_lstsq, spsolve,
};
pub use spai::{SpaiOptions, SpaiPreconditioner};
pub use svd::{svds, SVDOptions, SVDResult, svd_truncated};
pub use decomposition::{
    lu_decomposition, qr_decomposition, cholesky_decomposition, LUResult, QRResult, CholeskyResult,
    incomplete_lu, incomplete_cholesky, ILUOptions, ICOptions,
};
