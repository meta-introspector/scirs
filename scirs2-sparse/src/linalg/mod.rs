//! Linear algebra operations for sparse matrices
//!
//! This module provides linear algebra operations for sparse matrices,
//! including solvers, eigenvalue computations, and matrix functions.

mod amg;
mod cgs;
mod decomposition;
mod eigen;
mod enhanced_operators;
mod expm;
mod gcrot;
mod ic;
mod interface;
mod iterative;
mod lgmres;
mod lsmr;
mod lsqr;
mod matfuncs;
mod minres;
mod preconditioners;
mod qmr;
mod qmr_simple;
mod solvers;
mod spai;
mod svd;
mod tfqmr;

pub use amg::{AMGOptions, AMGPreconditioner, CycleType, InterpolationType, SmootherType};
pub use cgs::{cgs, CGSOptions, CGSResult};
pub use decomposition::{
    cholesky_decomposition, incomplete_cholesky, incomplete_lu, lu_decomposition, qr_decomposition,
    CholeskyResult, ICOptions, ILUOptions, LUResult, QRResult,
};
pub use eigen::{
    eigs, eigsh, lanczos, power_iteration, ArpackOptions, EigenResult, EigenvalueMethod,
    LanczosOptions, PowerIterationOptions,
};
pub use enhanced_operators::{
    convolution_operator, enhanced_add, enhanced_diagonal, enhanced_scale, enhanced_subtract,
    finite_difference_operator, BoundaryCondition, ConvolutionMode, ConvolutionOperator,
    EnhancedDiagonalOperator, EnhancedDifferenceOperator, EnhancedOperatorOptions,
    EnhancedScaledOperator, EnhancedSumOperator, FiniteDifferenceOperator,
};
pub use expm::expm;
pub use gcrot::{gcrot, GCROTOptions, GCROTResult};
pub use ic::IC0Preconditioner;
pub use interface::{
    AsLinearOperator, DiagonalOperator, IdentityOperator, LinearOperator, ScaledIdentityOperator,
};
pub use iterative::{
    bicg, bicgstab, cg, gmres, BiCGOptions, BiCGSTABOptions, BiCGSTABResult, CGOptions,
    GMRESOptions, IterationResult, IterativeSolver,
};
pub use lgmres::{lgmres, LGMRESOptions, LGMRESResult};
pub use lsmr::{lsmr, LSMROptions, LSMRResult};
pub use lsqr::{lsqr, LSQROptions, LSQRResult};
pub use matfuncs::{expm_multiply, onenormest};
pub use minres::{minres, MINRESOptions, MINRESResult};
pub use preconditioners::{ILU0Preconditioner, JacobiPreconditioner, SSORPreconditioner};
pub use qmr::{qmr, QMROptions, QMRResult};
pub use solvers::{
    add, diag_matrix, eye, inv, matmul, matrix_power, multiply, norm, sparse_direct_solve,
    sparse_lstsq, spsolve,
};
pub use spai::{SpaiOptions, SpaiPreconditioner};
pub use svd::{svd_truncated, svds, SVDOptions, SVDResult};
pub use tfqmr::{tfqmr, TFQMROptions, TFQMRResult};
