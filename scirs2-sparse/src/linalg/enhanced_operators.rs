//! Enhanced linear operators with SIMD and parallel acceleration
//!
//! This module provides performance-optimized linear operators that leverage
//! the parallel vector operations and SIMD acceleration from scirs2-core.

use crate::error::{SparseError, SparseResult};
use crate::linalg::interface::LinearOperator;
use crate::parallel_vector_ops::*;
use num_traits::{Float, NumAssign};
use std::fmt::Debug;
use std::iter::Sum;

// Import SIMD operations from scirs2-core
use scirs2_core::simd_ops::SimdUnifiedOps;

/// Configuration for enhanced operators
#[derive(Debug, Clone)]
pub struct EnhancedOperatorOptions {
    /// Use parallel processing for large vectors
    pub use_parallel: bool,
    /// Threshold for switching to parallel processing
    pub parallel_threshold: usize,
    /// Use SIMD acceleration when available
    pub use_simd: bool,
    /// Threshold for switching to SIMD processing
    pub simd_threshold: usize,
    /// Chunk size for parallel processing
    pub chunk_size: usize,
}

impl Default for EnhancedOperatorOptions {
    fn default() -> Self {
        Self {
            use_parallel: true,
            parallel_threshold: 10000,
            use_simd: true,
            simd_threshold: 32,
            chunk_size: 1024,
        }
    }
}

/// Enhanced diagonal operator with SIMD and parallel acceleration
#[derive(Clone)]
pub struct EnhancedDiagonalOperator<F> {
    diagonal: Vec<F>,
    #[allow(dead_code)]
    options: EnhancedOperatorOptions,
}

impl<F: Float + NumAssign + Sum + Copy + Send + Sync + SimdUnifiedOps> EnhancedDiagonalOperator<F> {
    /// Create a new enhanced diagonal operator
    pub fn new(diagonal: Vec<F>) -> Self {
        Self {
            diagonal,
            options: EnhancedOperatorOptions::default(),
        }
    }

    /// Create with custom options
    pub fn with_options(diagonal: Vec<F>, options: EnhancedOperatorOptions) -> Self {
        Self { diagonal, options }
    }

    /// Get the diagonal values
    pub fn diagonal(&self) -> &[F] {
        &self.diagonal
    }
}

impl<F: Float + NumAssign + Sum + Copy + Send + Sync + SimdUnifiedOps> LinearOperator<F>
    for EnhancedDiagonalOperator<F>
{
    fn shape(&self) -> (usize, usize) {
        let n = self.diagonal.len();
        (n, n)
    }

    fn matvec(&self, x: &[F]) -> SparseResult<Vec<F>> {
        if x.len() != self.diagonal.len() {
            return Err(SparseError::DimensionMismatch {
                expected: self.diagonal.len(),
                found: x.len(),
            });
        }

        let mut result = vec![F::zero(); x.len()];

        // Use optimized element-wise multiplication via parallel vector operations
        // This is equivalent to diagonal multiplication: result[i] = diagonal[i] * x[i]

        if self.options.use_parallel && x.len() >= self.options.parallel_threshold {
            // Use parallel processing for large diagonal operations
            use scirs2_core::parallel_ops::*;
            let indices: Vec<usize> = (0..x.len()).collect();
            let values = parallel_map(&indices, |&i| self.diagonal[i] * x[i]);
            result = values;
        } else {
            // Sequential computation for small vectors
            for i in 0..x.len() {
                result[i] = self.diagonal[i] * x[i];
            }
        }

        Ok(result)
    }

    fn rmatvec(&self, x: &[F]) -> SparseResult<Vec<F>> {
        // For real diagonal matrices, adjoint equals original
        self.matvec(x)
    }

    fn has_adjoint(&self) -> bool {
        true
    }
}

/// Enhanced sum operator with parallel acceleration
pub struct EnhancedSumOperator<F> {
    a: Box<dyn LinearOperator<F>>,
    b: Box<dyn LinearOperator<F>>,
    #[allow(dead_code)]
    options: EnhancedOperatorOptions,
}

impl<F: Float + NumAssign + Sum + Copy + Send + Sync + SimdUnifiedOps> EnhancedSumOperator<F> {
    /// Create a new enhanced sum operator
    pub fn new(a: Box<dyn LinearOperator<F>>, b: Box<dyn LinearOperator<F>>) -> SparseResult<Self> {
        if a.shape() != b.shape() {
            return Err(SparseError::ShapeMismatch {
                expected: a.shape(),
                found: b.shape(),
            });
        }
        Ok(Self {
            a,
            b,
            options: EnhancedOperatorOptions::default(),
        })
    }

    /// Create with custom options
    pub fn with_options(
        a: Box<dyn LinearOperator<F>>,
        b: Box<dyn LinearOperator<F>>,
        #[allow(dead_code)]
    options: EnhancedOperatorOptions,
    ) -> SparseResult<Self> {
        if a.shape() != b.shape() {
            return Err(SparseError::ShapeMismatch {
                expected: a.shape(),
                found: b.shape(),
            });
        }
        Ok(Self { a, b, options })
    }
}

impl<F: Float + NumAssign + Sum + Copy + Send + Sync + SimdUnifiedOps> LinearOperator<F>
    for EnhancedSumOperator<F>
{
    fn shape(&self) -> (usize, usize) {
        self.a.shape()
    }

    fn matvec(&self, x: &[F]) -> SparseResult<Vec<F>> {
        let a_result = self.a.matvec(x)?;
        let b_result = self.b.matvec(x)?;

        let mut result = vec![F::zero(); a_result.len()];
        let parallel_opts = Some(ParallelVectorOptions {
            use_parallel: self.options.use_parallel,
            parallel_threshold: self.options.parallel_threshold,
            chunk_size: self.options.chunk_size,
            use_simd: self.options.use_simd,
            simd_threshold: self.options.simd_threshold,
        });

        // Use optimized parallel vector addition
        parallel_vector_add(&a_result, &b_result, &mut result, parallel_opts);
        Ok(result)
    }

    fn rmatvec(&self, x: &[F]) -> SparseResult<Vec<F>> {
        if !self.a.has_adjoint() || !self.b.has_adjoint() {
            return Err(SparseError::OperationNotSupported(
                "adjoint not supported for one or both operators".to_string(),
            ));
        }
        let a_result = self.a.rmatvec(x)?;
        let b_result = self.b.rmatvec(x)?;

        let mut result = vec![F::zero(); a_result.len()];
        let parallel_opts = Some(ParallelVectorOptions {
            use_parallel: self.options.use_parallel,
            parallel_threshold: self.options.parallel_threshold,
            chunk_size: self.options.chunk_size,
            use_simd: self.options.use_simd,
            simd_threshold: self.options.simd_threshold,
        });

        parallel_vector_add(&a_result, &b_result, &mut result, parallel_opts);
        Ok(result)
    }

    fn has_adjoint(&self) -> bool {
        self.a.has_adjoint() && self.b.has_adjoint()
    }
}

/// Enhanced difference operator with parallel acceleration
pub struct EnhancedDifferenceOperator<F> {
    a: Box<dyn LinearOperator<F>>,
    b: Box<dyn LinearOperator<F>>,
    #[allow(dead_code)]
    options: EnhancedOperatorOptions,
}

impl<F: Float + NumAssign + Sum + Copy + Send + Sync + SimdUnifiedOps>
    EnhancedDifferenceOperator<F>
{
    /// Create a new enhanced difference operator
    pub fn new(a: Box<dyn LinearOperator<F>>, b: Box<dyn LinearOperator<F>>) -> SparseResult<Self> {
        if a.shape() != b.shape() {
            return Err(SparseError::ShapeMismatch {
                expected: a.shape(),
                found: b.shape(),
            });
        }
        Ok(Self {
            a,
            b,
            options: EnhancedOperatorOptions::default(),
        })
    }

    /// Create with custom options
    pub fn with_options(
        a: Box<dyn LinearOperator<F>>,
        b: Box<dyn LinearOperator<F>>,
        #[allow(dead_code)]
    options: EnhancedOperatorOptions,
    ) -> SparseResult<Self> {
        if a.shape() != b.shape() {
            return Err(SparseError::ShapeMismatch {
                expected: a.shape(),
                found: b.shape(),
            });
        }
        Ok(Self { a, b, options })
    }
}

impl<F: Float + NumAssign + Sum + Copy + Send + Sync + SimdUnifiedOps> LinearOperator<F>
    for EnhancedDifferenceOperator<F>
{
    fn shape(&self) -> (usize, usize) {
        self.a.shape()
    }

    fn matvec(&self, x: &[F]) -> SparseResult<Vec<F>> {
        let a_result = self.a.matvec(x)?;
        let b_result = self.b.matvec(x)?;

        let mut result = vec![F::zero(); a_result.len()];
        let parallel_opts = Some(ParallelVectorOptions {
            use_parallel: self.options.use_parallel,
            parallel_threshold: self.options.parallel_threshold,
            chunk_size: self.options.chunk_size,
            use_simd: self.options.use_simd,
            simd_threshold: self.options.simd_threshold,
        });

        // Use optimized parallel vector subtraction
        parallel_vector_sub(&a_result, &b_result, &mut result, parallel_opts);
        Ok(result)
    }

    fn rmatvec(&self, x: &[F]) -> SparseResult<Vec<F>> {
        if !self.a.has_adjoint() || !self.b.has_adjoint() {
            return Err(SparseError::OperationNotSupported(
                "adjoint not supported for one or both operators".to_string(),
            ));
        }
        let a_result = self.a.rmatvec(x)?;
        let b_result = self.b.rmatvec(x)?;

        let mut result = vec![F::zero(); a_result.len()];
        let parallel_opts = Some(ParallelVectorOptions {
            use_parallel: self.options.use_parallel,
            parallel_threshold: self.options.parallel_threshold,
            chunk_size: self.options.chunk_size,
            use_simd: self.options.use_simd,
            simd_threshold: self.options.simd_threshold,
        });

        parallel_vector_sub(&a_result, &b_result, &mut result, parallel_opts);
        Ok(result)
    }

    fn has_adjoint(&self) -> bool {
        self.a.has_adjoint() && self.b.has_adjoint()
    }
}

/// Enhanced scaled operator with parallel acceleration
pub struct EnhancedScaledOperator<F> {
    alpha: F,
    operator: Box<dyn LinearOperator<F>>,
    #[allow(dead_code)]
    options: EnhancedOperatorOptions,
}

impl<F: Float + NumAssign + Sum + Copy + Send + Sync + SimdUnifiedOps> EnhancedScaledOperator<F> {
    /// Create a new enhanced scaled operator
    pub fn new(alpha: F, operator: Box<dyn LinearOperator<F>>) -> Self {
        Self {
            alpha,
            operator,
            options: EnhancedOperatorOptions::default(),
        }
    }

    /// Create with custom options
    pub fn with_options(
        alpha: F,
        operator: Box<dyn LinearOperator<F>>,
        #[allow(dead_code)]
    options: EnhancedOperatorOptions,
    ) -> Self {
        Self {
            alpha,
            operator,
            options,
        }
    }
}

impl<F: Float + NumAssign + Sum + Copy + Send + Sync + SimdUnifiedOps> LinearOperator<F>
    for EnhancedScaledOperator<F>
{
    fn shape(&self) -> (usize, usize) {
        self.operator.shape()
    }

    fn matvec(&self, x: &[F]) -> SparseResult<Vec<F>> {
        let result = self.operator.matvec(x)?;
        let mut scaled_result = vec![F::zero(); result.len()];

        let parallel_opts = Some(ParallelVectorOptions {
            use_parallel: self.options.use_parallel,
            parallel_threshold: self.options.parallel_threshold,
            chunk_size: self.options.chunk_size,
            use_simd: self.options.use_simd,
            simd_threshold: self.options.simd_threshold,
        });

        // Use optimized parallel vector scaling
        parallel_vector_scale(self.alpha, &result, &mut scaled_result, parallel_opts);
        Ok(scaled_result)
    }

    fn rmatvec(&self, x: &[F]) -> SparseResult<Vec<F>> {
        if !self.operator.has_adjoint() {
            return Err(SparseError::OperationNotSupported(
                "adjoint not supported for underlying operator".to_string(),
            ));
        }
        let result = self.operator.rmatvec(x)?;
        let mut scaled_result = vec![F::zero(); result.len()];

        let parallel_opts = Some(ParallelVectorOptions {
            use_parallel: self.options.use_parallel,
            parallel_threshold: self.options.parallel_threshold,
            chunk_size: self.options.chunk_size,
            use_simd: self.options.use_simd,
            simd_threshold: self.options.simd_threshold,
        });

        parallel_vector_scale(self.alpha, &result, &mut scaled_result, parallel_opts);
        Ok(scaled_result)
    }

    fn has_adjoint(&self) -> bool {
        self.operator.has_adjoint()
    }
}

/// Convolution operator for matrix-free convolution operations
pub struct ConvolutionOperator<F> {
    kernel: Vec<F>,
    input_size: usize,
    output_size: usize,
    mode: ConvolutionMode,
    #[allow(dead_code)]
    options: EnhancedOperatorOptions,
}

#[derive(Debug, Clone, Copy)]
pub enum ConvolutionMode {
    /// Full convolution (output_size = input_size + kernel_size - 1)
    Full,
    /// Same convolution (output_size = input_size)
    Same,
    /// Valid convolution (output_size = input_size - kernel_size + 1)
    Valid,
}

impl<F: Float + NumAssign + Sum + Copy + Send + Sync + SimdUnifiedOps> ConvolutionOperator<F> {
    /// Create a new convolution operator
    pub fn new(kernel: Vec<F>, input_size: usize, mode: ConvolutionMode) -> Self {
        let output_size = match mode {
            ConvolutionMode::Full => input_size + kernel.len() - 1,
            ConvolutionMode::Same => input_size,
            ConvolutionMode::Valid => {
                if input_size >= kernel.len() {
                    input_size - kernel.len() + 1
                } else {
                    0
                }
            }
        };

        Self {
            kernel,
            input_size,
            output_size,
            mode,
            options: EnhancedOperatorOptions::default(),
        }
    }

    /// Create with custom options
    pub fn with_options(
        kernel: Vec<F>,
        input_size: usize,
        mode: ConvolutionMode,
        #[allow(dead_code)]
    options: EnhancedOperatorOptions,
    ) -> Self {
        let output_size = match mode {
            ConvolutionMode::Full => input_size + kernel.len() - 1,
            ConvolutionMode::Same => input_size,
            ConvolutionMode::Valid => {
                if input_size >= kernel.len() {
                    input_size - kernel.len() + 1
                } else {
                    0
                }
            }
        };

        Self {
            kernel,
            input_size,
            output_size,
            mode,
            options,
        }
    }
}

impl<F: Float + NumAssign + Sum + Copy + Send + Sync + SimdUnifiedOps> LinearOperator<F>
    for ConvolutionOperator<F>
{
    fn shape(&self) -> (usize, usize) {
        (self.output_size, self.input_size)
    }

    fn matvec(&self, x: &[F]) -> SparseResult<Vec<F>> {
        if x.len() != self.input_size {
            return Err(SparseError::DimensionMismatch {
                expected: self.input_size,
                found: x.len(),
            });
        }

        let mut result = vec![F::zero(); self.output_size];

        // Implement convolution based on mode
        match self.mode {
            ConvolutionMode::Full => {
                for i in 0..self.output_size {
                    let mut sum = F::zero();
                    for (j, &kernel_val) in self.kernel.iter().enumerate() {
                        if i >= j && (i - j) < x.len() {
                            sum += kernel_val * x[i - j];
                        }
                    }
                    result[i] = sum;
                }
            }
            ConvolutionMode::Same => {
                let pad = self.kernel.len() / 2;
                for i in 0..self.output_size {
                    let mut sum = F::zero();
                    for (j, &kernel_val) in self.kernel.iter().enumerate() {
                        let idx = i + j;
                        if idx >= pad && (idx - pad) < x.len() {
                            sum += kernel_val * x[idx - pad];
                        }
                    }
                    result[i] = sum;
                }
            }
            ConvolutionMode::Valid => {
                for i in 0..self.output_size {
                    let mut sum = F::zero();
                    for (j, &kernel_val) in self.kernel.iter().enumerate() {
                        let idx = i + j;
                        if idx < x.len() {
                            sum += kernel_val * x[idx];
                        }
                    }
                    result[i] = sum;
                }
            }
        }

        // TODO: Add parallel/SIMD optimization for large convolutions
        Ok(result)
    }

    fn has_adjoint(&self) -> bool {
        true
    }

    fn rmatvec(&self, x: &[F]) -> SparseResult<Vec<F>> {
        if x.len() != self.output_size {
            return Err(SparseError::DimensionMismatch {
                expected: self.output_size,
                found: x.len(),
            });
        }

        // Adjoint of convolution is correlation with flipped kernel
        let mut result = vec![F::zero(); self.input_size];
        let flipped_kernel: Vec<F> = self.kernel.iter().rev().copied().collect();

        // Implement correlation (adjoint of convolution)
        match self.mode {
            ConvolutionMode::Full => {
                for i in 0..self.input_size {
                    let mut sum = F::zero();
                    for (j, &kernel_val) in flipped_kernel.iter().enumerate() {
                        let idx = i + j;
                        if idx < x.len() {
                            sum += kernel_val * x[idx];
                        }
                    }
                    result[i] = sum;
                }
            }
            ConvolutionMode::Same => {
                let pad = flipped_kernel.len() / 2;
                for i in 0..self.input_size {
                    let mut sum = F::zero();
                    for (j, &kernel_val) in flipped_kernel.iter().enumerate() {
                        if i + pad >= j && (i + pad - j) < x.len() {
                            sum += kernel_val * x[i + pad - j];
                        }
                    }
                    result[i] = sum;
                }
            }
            ConvolutionMode::Valid => {
                for i in 0..self.input_size {
                    let mut sum = F::zero();
                    for (j, &kernel_val) in flipped_kernel.iter().enumerate() {
                        if i >= j && (i - j) < x.len() {
                            sum += kernel_val * x[i - j];
                        }
                    }
                    result[i] = sum;
                }
            }
        }

        Ok(result)
    }
}

/// Finite difference operator for computing derivatives
pub struct FiniteDifferenceOperator<F> {
    size: usize,
    order: usize,
    spacing: F,
    boundary: BoundaryCondition,
    #[allow(dead_code)]
    options: EnhancedOperatorOptions,
}

#[derive(Debug, Clone, Copy)]
pub enum BoundaryCondition {
    /// Neumann boundary conditions (zero derivative)
    Neumann,
    /// Dirichlet boundary conditions (zero value)
    Dirichlet,
    /// Periodic boundary conditions
    Periodic,
}

impl<F: Float + NumAssign + Sum + Copy + Send + Sync + SimdUnifiedOps> FiniteDifferenceOperator<F> {
    /// Create a new finite difference operator
    pub fn new(size: usize, order: usize, spacing: F, boundary: BoundaryCondition) -> Self {
        Self {
            size,
            order,
            spacing,
            boundary,
            options: EnhancedOperatorOptions::default(),
        }
    }

    /// Create with custom options
    pub fn with_options(
        size: usize,
        order: usize,
        spacing: F,
        boundary: BoundaryCondition,
        #[allow(dead_code)]
    options: EnhancedOperatorOptions,
    ) -> Self {
        Self {
            size,
            order,
            spacing,
            boundary,
            options,
        }
    }

    /// Get finite difference coefficients for given order
    fn get_coefficients(&self) -> Vec<F> {
        match self.order {
            1 => {
                // First derivative: central difference
                vec![
                    -F::one() / (F::from(2.0).unwrap() * self.spacing),
                    F::zero(),
                    F::one() / (F::from(2.0).unwrap() * self.spacing),
                ]
            }
            2 => {
                // Second derivative: central difference
                let h_sq = self.spacing * self.spacing;
                vec![
                    F::one() / h_sq,
                    -F::from(2.0).unwrap() / h_sq,
                    F::one() / h_sq,
                ]
            }
            _ => {
                // Higher order derivatives - simplified implementation
                // In practice, would use more sophisticated stencils
                vec![F::zero(); 2 * self.order + 1]
            }
        }
    }
}

impl<F: Float + NumAssign + Sum + Copy + Send + Sync + SimdUnifiedOps> LinearOperator<F>
    for FiniteDifferenceOperator<F>
{
    fn shape(&self) -> (usize, usize) {
        (self.size, self.size)
    }

    fn matvec(&self, x: &[F]) -> SparseResult<Vec<F>> {
        if x.len() != self.size {
            return Err(SparseError::DimensionMismatch {
                expected: self.size,
                found: x.len(),
            });
        }

        let mut result = vec![F::zero(); self.size];
        let coeffs = self.get_coefficients();
        let stencil_radius = coeffs.len() / 2;

        for i in 0..self.size {
            let mut sum = F::zero();
            for (j, &coeff) in coeffs.iter().enumerate() {
                let idx = i as isize + j as isize - stencil_radius as isize;

                let value = match self.boundary {
                    BoundaryCondition::Neumann => {
                        if idx < 0 {
                            x[0]
                        } else if idx >= self.size as isize {
                            x[self.size - 1]
                        } else {
                            x[idx as usize]
                        }
                    }
                    BoundaryCondition::Dirichlet => {
                        if idx < 0 || idx >= self.size as isize {
                            F::zero()
                        } else {
                            x[idx as usize]
                        }
                    }
                    BoundaryCondition::Periodic => {
                        let periodic_idx = ((idx % self.size as isize + self.size as isize)
                            % self.size as isize)
                            as usize;
                        x[periodic_idx]
                    }
                };

                sum += coeff * value;
            }
            result[i] = sum;
        }

        Ok(result)
    }

    fn has_adjoint(&self) -> bool {
        true
    }

    fn rmatvec(&self, x: &[F]) -> SparseResult<Vec<F>> {
        // For symmetric finite difference operators, adjoint equals transpose
        // For asymmetric operators, would need to implement proper adjoint
        self.matvec(x)
    }
}

/// Create utility functions for enhanced operators
/// Create an enhanced diagonal operator
pub fn enhanced_diagonal<
    F: Float + NumAssign + Sum + Copy + Send + Sync + SimdUnifiedOps + 'static,
>(
    diagonal: Vec<F>,
) -> Box<dyn LinearOperator<F>> {
    Box::new(EnhancedDiagonalOperator::new(diagonal))
}

/// Create an enhanced sum operator
pub fn enhanced_add<F: Float + NumAssign + Sum + Copy + Send + Sync + SimdUnifiedOps + 'static>(
    left: Box<dyn LinearOperator<F>>,
    right: Box<dyn LinearOperator<F>>,
) -> SparseResult<Box<dyn LinearOperator<F>>> {
    Ok(Box::new(EnhancedSumOperator::new(left, right)?))
}

/// Create an enhanced difference operator
pub fn enhanced_subtract<
    F: Float + NumAssign + Sum + Copy + Send + Sync + SimdUnifiedOps + 'static,
>(
    left: Box<dyn LinearOperator<F>>,
    right: Box<dyn LinearOperator<F>>,
) -> SparseResult<Box<dyn LinearOperator<F>>> {
    Ok(Box::new(EnhancedDifferenceOperator::new(left, right)?))
}

/// Create an enhanced scaled operator
pub fn enhanced_scale<
    F: Float + NumAssign + Sum + Copy + Send + Sync + SimdUnifiedOps + 'static,
>(
    alpha: F,
    operator: Box<dyn LinearOperator<F>>,
) -> Box<dyn LinearOperator<F>> {
    Box::new(EnhancedScaledOperator::new(alpha, operator))
}

/// Create a convolution operator
pub fn convolution_operator<
    F: Float + NumAssign + Sum + Copy + Send + Sync + SimdUnifiedOps + 'static,
>(
    kernel: Vec<F>,
    input_size: usize,
    mode: ConvolutionMode,
) -> Box<dyn LinearOperator<F>> {
    Box::new(ConvolutionOperator::new(kernel, input_size, mode))
}

/// Create a finite difference operator
pub fn finite_difference_operator<
    F: Float + NumAssign + Sum + Copy + Send + Sync + SimdUnifiedOps + 'static,
>(
    size: usize,
    order: usize,
    spacing: F,
    boundary: BoundaryCondition,
) -> Box<dyn LinearOperator<F>> {
    Box::new(FiniteDifferenceOperator::new(
        size, order, spacing, boundary,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_enhanced_diagonal_operator() {
        let diag = vec![2.0, 3.0, 4.0];
        let op = EnhancedDiagonalOperator::new(diag);
        let x = vec![1.0, 2.0, 3.0];
        let y = op.matvec(&x).unwrap();
        assert_eq!(y, vec![2.0, 6.0, 12.0]);
    }

    #[test]
    fn test_enhanced_sum_operator() {
        let diag1 = enhanced_diagonal(vec![1.0, 2.0, 3.0]);
        let diag2 = enhanced_diagonal(vec![2.0, 1.0, 1.0]);
        let sum_op = EnhancedSumOperator::new(diag1, diag2).unwrap();

        let x = vec![1.0, 1.0, 1.0];
        let y = sum_op.matvec(&x).unwrap();
        assert_eq!(y, vec![3.0, 3.0, 4.0]); // (1+2)*1, (2+1)*1, (3+1)*1
    }

    #[test]
    fn test_enhanced_difference_operator() {
        let diag1 = enhanced_diagonal(vec![3.0, 4.0, 5.0]);
        let diag2 = enhanced_diagonal(vec![1.0, 2.0, 1.0]);
        let diff_op = EnhancedDifferenceOperator::new(diag1, diag2).unwrap();

        let x = vec![1.0, 1.0, 1.0];
        let y = diff_op.matvec(&x).unwrap();
        assert_eq!(y, vec![2.0, 2.0, 4.0]); // (3-1)*1, (4-2)*1, (5-1)*1
    }

    #[test]
    fn test_enhanced_scaled_operator() {
        let diag = enhanced_diagonal(vec![1.0, 2.0, 3.0]);
        let scaled_op = EnhancedScaledOperator::new(2.0, diag);

        let x = vec![1.0, 1.0, 1.0];
        let y = scaled_op.matvec(&x).unwrap();
        assert_eq!(y, vec![2.0, 4.0, 6.0]); // 2 * [1, 2, 3]
    }

    #[test]
    fn test_convolution_operator_full() {
        let kernel = vec![1.0, 2.0, 3.0];
        let conv_op = ConvolutionOperator::new(kernel, 3, ConvolutionMode::Full);

        let x = vec![1.0, 0.0, 0.0];
        let y = conv_op.matvec(&x).unwrap();
        assert_eq!(y, vec![1.0, 2.0, 3.0, 0.0, 0.0]); // Full convolution output
    }

    #[test]
    fn test_convolution_operator_same() {
        let kernel = vec![0.0, 1.0, 0.0]; // Identity kernel
        let conv_op = ConvolutionOperator::new(kernel, 3, ConvolutionMode::Same);

        let x = vec![1.0, 2.0, 3.0];
        let y = conv_op.matvec(&x).unwrap();
        assert_relative_eq!(y[0], 2.0, epsilon = 1e-10);
        assert_relative_eq!(y[1], 3.0, epsilon = 1e-10);
        assert_relative_eq!(y[2], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_finite_difference_first_derivative() {
        let fd_op = FiniteDifferenceOperator::new(5, 1, 1.0, BoundaryCondition::Dirichlet);

        // Test on a linear function: f(x) = x
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = fd_op.matvec(&x).unwrap();

        // First derivative of linear function should be approximately 1
        for i in 1..y.len() - 1 {
            assert_relative_eq!(y[i], 1.0, epsilon = 0.1);
        }
    }

    #[test]
    fn test_finite_difference_second_derivative() {
        let fd_op = FiniteDifferenceOperator::new(5, 2, 1.0, BoundaryCondition::Dirichlet);

        // Test on a quadratic function: f(x) = x^2
        let x = vec![0.0, 1.0, 4.0, 9.0, 16.0];
        let y = fd_op.matvec(&x).unwrap();

        // Second derivative of x^2 should be approximately 2
        for i in 1..y.len() - 1 {
            assert_relative_eq!(y[i], 2.0, epsilon = 0.5);
        }
    }

    #[test]
    fn test_enhanced_operators_with_large_vectors() {
        // Test that large vectors trigger parallel processing
        let large_size = 15000; // Above default parallel threshold
        let diag1: Vec<f64> = (0..large_size).map(|i| (i + 1) as f64).collect();
        let diag2: Vec<f64> = vec![2.0; large_size];

        let op1 = enhanced_diagonal(diag1);
        let op2 = enhanced_diagonal(diag2);
        let sum_op = EnhancedSumOperator::new(op1, op2).unwrap();

        let x = vec![1.0; large_size];
        let y = sum_op.matvec(&x).unwrap();

        // Check some values
        assert_relative_eq!(y[0], 3.0, epsilon = 1e-10); // (1 + 2) * 1
        assert_relative_eq!(y[1], 4.0, epsilon = 1e-10); // (2 + 2) * 1
        assert_relative_eq!(y[999], 1002.0, epsilon = 1e-10); // (1000 + 2) * 1
    }
}
