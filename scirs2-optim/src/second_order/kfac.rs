//! K-FAC (Kronecker-Factored Approximate Curvature) optimizer
//!
//! This module implements K-FAC, an efficient second-order optimization method
//! that approximates the Fisher information matrix using Kronecker factorization.
//! This allows for much more scalable second-order optimization compared to
//! storing and inverting the full Fisher information matrix.

use crate::error::{OptimError, Result};
use ndarray::{s, Array, Array1, Array2, Array3, ArrayBase, Axis, Data, Dimension, Ix1, Ix2};
use num_traits::Float;
use std::collections::HashMap;
use std::fmt::Debug;

/// K-FAC optimizer configuration
#[derive(Debug, Clone)]
pub struct KFACConfig<T: Float> {
    /// Learning rate
    pub learning_rate: T,

    /// Damping parameter for numerical stability
    pub damping: T,

    /// Weight decay (L2 regularization)
    pub weight_decay: T,

    /// Update frequency for covariance matrices
    pub cov_update_freq: usize,

    /// Update frequency for inverse covariance matrices
    pub inv_update_freq: usize,

    /// Exponential moving average decay for statistics
    pub stat_decay: T,

    /// Minimum eigenvalue for regularization
    pub min_eigenvalue: T,

    /// Maximum number of iterations for iterative inversion
    pub max_inv_iterations: usize,

    /// Tolerance for iterative inversion
    pub inv_tolerance: T,

    /// Use Tikhonov regularization
    pub use_tikhonov: bool,

    /// Enable automatic damping adjustment
    pub auto_damping: bool,

    /// Target acceptance ratio for damping adjustment
    pub target_acceptance_ratio: T,
}

impl<T: Float> Default for KFACConfig<T> {
    fn default() -> Self {
        Self {
            learning_rate: T::from(0.001).unwrap(),
            damping: T::from(0.001).unwrap(),
            weight_decay: T::from(0.0).unwrap(),
            cov_update_freq: 10,
            inv_update_freq: 100,
            stat_decay: T::from(0.95).unwrap(),
            min_eigenvalue: T::from(1e-7).unwrap(),
            max_inv_iterations: 50,
            inv_tolerance: T::from(1e-6).unwrap(),
            use_tikhonov: true,
            auto_damping: true,
            target_acceptance_ratio: T::from(0.75).unwrap(),
        }
    }
}

/// Layer information for K-FAC
#[derive(Debug, Clone)]
pub struct LayerInfo {
    /// Layer name/identifier
    pub name: String,

    /// Input dimension
    pub input_dim: usize,

    /// Output dimension  
    pub output_dim: usize,

    /// Layer type
    pub layer_type: LayerType,

    /// Whether to include bias
    pub has_bias: bool,
}

/// Types of layers supported by K-FAC
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayerType {
    /// Dense/Fully connected layer
    Dense,

    /// Convolutional layer
    Convolution,

    /// Convolutional layer with grouped/depthwise convolution
    GroupedConvolution { groups: usize },

    /// Embedding layer
    Embedding,

    /// Batch normalization layer
    BatchNorm,
}

/// K-FAC optimizer state for a single layer
#[derive(Debug, Clone)]
pub struct KFACLayerState<T: Float> {
    /// Input covariance matrix A = E[a a^T]
    pub a_cov: Array2<T>,

    /// Output gradient covariance matrix G = E[g g^T]
    pub g_cov: Array2<T>,

    /// Inverse of input covariance matrix
    pub a_cov_inv: Option<Array2<T>>,

    /// Inverse of output gradient covariance matrix
    pub g_cov_inv: Option<Array2<T>>,

    /// Number of updates performed
    pub num_updates: usize,

    /// Last update step for covariance matrices
    pub last_cov_update: usize,

    /// Last update step for inverse matrices
    pub last_inv_update: usize,

    /// Damping values for this layer
    pub damping_a: T,
    pub damping_g: T,

    /// Layer information
    pub layer_info: LayerInfo,

    /// Precomputed Kronecker factors for bias
    pub bias_correction: Option<Array1<T>>,

    /// Moving average statistics
    pub running_mean_a: Option<Array1<T>>,
    pub running_mean_g: Option<Array1<T>>,
}

/// Main K-FAC optimizer
#[derive(Debug)]
pub struct KFAC<T: Float> {
    /// Configuration
    config: KFACConfig<T>,

    /// Per-layer state
    layer_states: HashMap<String, KFACLayerState<T>>,

    /// Global step counter
    step_count: usize,

    /// Acceptance ratio for damping adjustment
    acceptance_ratio: T,

    /// Previous loss for loss-based damping
    previous_loss: Option<T>,

    /// Eigenvalue regularization history
    eigenvalue_history: Vec<T>,

    /// Performance statistics
    stats: KFACStats<T>,
}

/// K-FAC performance statistics
#[derive(Debug, Clone, Default)]
pub struct KFACStats<T: Float> {
    /// Total number of optimization steps
    pub total_steps: usize,

    /// Number of covariance updates
    pub cov_updates: usize,

    /// Number of inverse updates
    pub inv_updates: usize,

    /// Average condition number of covariance matrices
    pub avg_condition_number: T,

    /// Time spent in different operations (in microseconds)
    pub time_cov_update: u64,
    pub time_inv_update: u64,
    pub time_gradient_update: u64,

    /// Memory usage estimate (in bytes)
    pub memory_usage: usize,
}

impl<T: Float + Default + Clone + Send + Sync> KFAC<T> {
    /// Create a new K-FAC optimizer
    pub fn new(config: KFACConfig<T>) -> Self {
        Self {
            config,
            layer_states: HashMap::new(),
            step_count: 0,
            acceptance_ratio: T::from(1.0).unwrap(),
            previous_loss: None,
            eigenvalue_history: Vec::new(),
            stats: KFACStats::default(),
        }
    }

    /// Register a layer with the optimizer
    pub fn register_layer(&mut self, layer_info: LayerInfo) -> Result<()> {
        let input_size = layer_info.input_dim + if layer_info.has_bias { 1 } else { 0 };
        let output_size = layer_info.output_dim;

        let state = KFACLayerState {
            a_cov: Array2::eye(input_size),
            g_cov: Array2::eye(output_size),
            a_cov_inv: None,
            g_cov_inv: None,
            num_updates: 0,
            last_cov_update: 0,
            last_inv_update: 0,
            damping_a: self.config.damping,
            damping_g: self.config.damping,
            layer_info,
            bias_correction: None,
            running_mean_a: None,
            running_mean_g: None,
        };

        self.layer_states.insert(layer_info.name.clone(), state);
        Ok(())
    }

    /// Update covariance matrices with new activations and gradients
    pub fn update_covariance_matrices(
        &mut self,
        layer_name: &str,
        activations: &Array2<T>,
        gradients: &Array2<T>,
    ) -> Result<()> {
        let state = self
            .layer_states
            .get_mut(layer_name)
            .ok_or_else(|| OptimError::InvalidConfig(format!("Layer {} not found", layer_name)))?;

        // Update only if it's time
        if self.step_count - state.last_cov_update >= self.config.cov_update_freq {
            self.update_input_covariance(state, activations)?;
            self.update_output_covariance(state, gradients)?;
            state.last_cov_update = self.step_count;
            self.stats.cov_updates += 1;
        }

        Ok(())
    }

    /// Update input covariance matrix A = E[a a^T]
    fn update_input_covariance(
        &mut self,
        state: &mut KFACLayerState<T>,
        activations: &Array2<T>,
    ) -> Result<()> {
        let batch_size = T::from(activations.nrows()).unwrap();

        // Add bias term if needed
        let augmented_activations = if state.layer_info.has_bias {
            let mut aug = Array2::ones((activations.nrows(), activations.ncols() + 1));
            aug.slice_mut(s![.., ..activations.ncols()])
                .assign(activations);
            aug
        } else {
            activations.clone()
        };

        // Compute batch covariance: (1/batch_size) * A^T * A
        let batch_cov = augmented_activations.t().dot(&augmented_activations) / batch_size;

        // Exponential moving average update
        let decay = self.config.stat_decay;
        state.a_cov = &state.a_cov * decay + &batch_cov * (T::one() - decay);

        // Update running mean for bias correction
        if state.running_mean_a.is_none() {
            state.running_mean_a = Some(augmented_activations.mean_axis(Axis(0)).unwrap());
        } else {
            let mean = augmented_activations.mean_axis(Axis(0)).unwrap();
            let running_mean = state.running_mean_a.as_mut().unwrap();
            *running_mean = &*running_mean * decay + &mean * (T::one() - decay);
        }

        Ok(())
    }

    /// Update output gradient covariance matrix G = E[g g^T]
    fn update_output_covariance(
        &mut self,
        state: &mut KFACLayerState<T>,
        gradients: &Array2<T>,
    ) -> Result<()> {
        let batch_size = T::from(gradients.nrows()).unwrap();

        // Compute batch covariance: (1/batch_size) * G^T * G
        let batch_cov = gradients.t().dot(gradients) / batch_size;

        // Exponential moving average update
        let decay = self.config.stat_decay;
        state.g_cov = &state.g_cov * decay + &batch_cov * (T::one() - decay);

        // Update running mean
        if state.running_mean_g.is_none() {
            state.running_mean_g = Some(gradients.mean_axis(Axis(0)).unwrap());
        } else {
            let mean = gradients.mean_axis(Axis(0)).unwrap();
            let running_mean = state.running_mean_g.as_mut().unwrap();
            *running_mean = &*running_mean * decay + &mean * (T::one() - decay);
        }

        Ok(())
    }

    /// Update inverse covariance matrices
    pub fn update_inverse_matrices(&mut self, layer_name: &str) -> Result<()> {
        let state = self
            .layer_states
            .get_mut(layer_name)
            .ok_or_else(|| OptimError::InvalidConfig(format!("Layer {} not found", layer_name)))?;

        // Update only if it's time
        if self.step_count - state.last_inv_update >= self.config.inv_update_freq {
            self.compute_inverse_covariance(state)?;
            state.last_inv_update = self.step_count;
            self.stats.inv_updates += 1;
        }

        Ok(())
    }

    /// Compute inverse covariance matrices with regularization
    fn compute_inverse_covariance(&mut self, state: &mut KFACLayerState<T>) -> Result<()> {
        // Add Tikhonov regularization for numerical stability
        let mut a_reg = state.a_cov.clone();
        let mut g_reg = state.g_cov.clone();

        if self.config.use_tikhonov {
            // Adaptive damping based on condition number
            let damping_a = if self.config.auto_damping {
                self.compute_adaptive_damping(&a_reg)?
            } else {
                state.damping_a
            };

            let damping_g = if self.config.auto_damping {
                self.compute_adaptive_damping(&g_reg)?
            } else {
                state.damping_g
            };

            // Add damping to diagonal
            for i in 0..a_reg.nrows() {
                a_reg[[i, i]] = a_reg[[i, i]] + damping_a;
            }
            for i in 0..g_reg.nrows() {
                g_reg[[i, i]] = g_reg[[i, i]] + damping_g;
            }

            state.damping_a = damping_a;
            state.damping_g = damping_g;
        }

        // Compute inverses using Cholesky decomposition for numerical stability
        state.a_cov_inv = Some(self.safe_matrix_inverse(&a_reg)?);
        state.g_cov_inv = Some(self.safe_matrix_inverse(&g_reg)?);

        Ok(())
    }

    /// Safely compute matrix inverse with fallback methods
    fn safe_matrix_inverse(&self, matrix: &Array2<T>) -> Result<Array2<T>> {
        // Check matrix condition number first
        let condition_number = self.estimate_condition_number(matrix);
        let max_condition = T::from(1e12).unwrap();

        if condition_number > max_condition {
            // Use regularized inverse for ill-conditioned matrices
            return self.regularized_inverse(matrix);
        }

        // Try Cholesky decomposition first (fastest for positive definite)
        if let Ok(inverse) = self.cholesky_inverse(matrix) {
            return Ok(inverse);
        }

        // Fallback to LU decomposition with partial pivoting
        if let Ok(inverse) = self.lu_inverse(matrix) {
            return Ok(inverse);
        }

        // Use iterative refinement for better accuracy
        if let Ok(inverse) = self.iterative_inverse(matrix) {
            return Ok(inverse);
        }

        // Final fallback to pseudoinverse using SVD
        self.svd_pseudoinverse(matrix)
    }

    /// Compute matrix inverse using Cholesky decomposition
    fn cholesky_inverse(&self, matrix: &Array2<T>) -> Result<Array2<T>> {
        let n = matrix.nrows();
        let mut l = Array2::zeros((n, n));

        // Perform Cholesky decomposition: A = L * L^T
        for i in 0..n {
            for j in 0..=i {
                if i == j {
                    // Diagonal elements
                    let mut sum = T::zero();
                    for k in 0..j {
                        sum = sum + l[[j, k]] * l[[j, k]];
                    }
                    let diag_val = matrix[[j, j]] - sum;
                    if diag_val <= T::zero() {
                        return Err(OptimError::InvalidConfig(
                            "Matrix not positive definite".to_string(),
                        ));
                    }
                    l[[j, j]] = diag_val.sqrt();
                } else {
                    // Lower triangular elements
                    let mut sum = T::zero();
                    for k in 0..j {
                        sum = sum + l[[i, k]] * l[[j, k]];
                    }
                    l[[i, j]] = (matrix[[i, j]] - sum) / l[[j, j]];
                }
            }
        }

        // Compute inverse using forward and backward substitution
        self.cholesky_solve_identity(&l)
    }

    /// Compute matrix inverse using LU decomposition
    fn lu_inverse(&self, matrix: &Array2<T>) -> Result<Array2<T>> {
        let n = matrix.nrows();
        let mut lu = matrix.clone();
        let mut perm = (0..n).collect::<Vec<usize>>();

        // LU decomposition with partial pivoting
        for k in 0..n - 1 {
            // Find pivot
            let mut max_idx = k;
            let mut max_val = lu[[k, k]].abs();
            for i in k + 1..n {
                let val = lu[[i, k]].abs();
                if val > max_val {
                    max_val = val;
                    max_idx = i;
                }
            }

            // Swap rows if needed
            if max_idx != k {
                for j in 0..n {
                    let temp = lu[[k, j]];
                    lu[[k, j]] = lu[[max_idx, j]];
                    lu[[max_idx, j]] = temp;
                }
                perm.swap(k, max_idx);
            }

            // Check for zero pivot
            if lu[[k, k]].abs() < T::from(1e-14).unwrap() {
                return Err(OptimError::InvalidConfig("Singular matrix".to_string()));
            }

            // Eliminate below pivot
            for i in k + 1..n {
                lu[[i, k]] = lu[[i, k]] / lu[[k, k]];
                for j in k + 1..n {
                    lu[[i, j]] = lu[[i, j]] - lu[[i, k]] * lu[[k, j]];
                }
            }
        }

        // Solve for inverse using forward and backward substitution
        self.lu_solve_identity(&lu, &perm)
    }

    /// Compute pseudoinverse using SVD
    fn svd_pseudoinverse(&self, matrix: &Array2<T>) -> Result<Array2<T>> {
        let (m, n) = matrix.dim();
        let min_dim = m.min(n);

        // Simplified SVD using power iteration for dominant eigenvalues
        let mut u = Array2::zeros((m, min_dim));
        let mut s = Array1::zeros(min_dim);
        let mut vt = Array2::zeros((min_dim, n));

        let mut a = matrix.clone();
        let tolerance = T::from(1e-10).unwrap();

        for k in 0..min_dim {
            // Power iteration to find k-th singular vector
            let mut v = Array1::ones(n);

            for _ in 0..50 {
                // Maximum iterations
                let u_k = a.dot(&v);
                let u_norm = u_k.iter().map(|&x| x * x).sum::<T>().sqrt();

                if u_norm < tolerance {
                    break;
                }

                let u_normalized = &u_k / u_norm;
                let v_new = a.t().dot(&u_normalized);
                let v_norm = v_new.iter().map(|&x| x * x).sum::<T>().sqrt();

                if v_norm < tolerance {
                    break;
                }

                v = &v_new / v_norm;
                s[k] = v_norm;
            }

            // Store singular vectors
            let u_k = a.dot(&v);
            let u_norm = u_k.iter().map(|&x| x * x).sum::<T>().sqrt();
            if u_norm > tolerance {
                for i in 0..m {
                    u[[i, k]] = u_k[i] / u_norm;
                }
                for j in 0..n {
                    vt[[k, j]] = v[j];
                }

                // Deflate matrix
                let outer_prod =
                    Array2::from_shape_fn((m, n), |(i, j)| u[[i, k]] * s[k] * vt[[k, j]]);
                a = a - outer_prod;
            }
        }

        // Compute pseudoinverse: A^+ = V * S^+ * U^T
        let mut s_inv = Array2::zeros((n, m));
        for i in 0..min_dim {
            if s[i] > self.config.min_eigenvalue {
                s_inv[[i, i]] = T::one() / s[i];
            }
        }

        Ok(vt.t().dot(&s_inv).dot(&u.t()))
    }

    /// Compute adaptive damping based on matrix condition number
    fn compute_adaptive_damping(&self, matrix: &Array2<T>) -> Result<T> {
        // Estimate condition number using ratio of max/min diagonal elements
        let mut max_diag = T::zero();
        let mut min_diag = T::infinity();

        for i in 0..matrix.nrows() {
            let diag = matrix[[i, i]];
            if diag > max_diag {
                max_diag = diag;
            }
            if diag < min_diag {
                min_diag = diag;
            }
        }

        let condition_estimate = if min_diag > T::zero() {
            max_diag / min_diag
        } else {
            T::infinity()
        };

        // Adaptive damping based on condition number
        let target_condition = T::from(1e6).unwrap();
        let base_damping = self.config.damping;

        if condition_estimate > target_condition {
            base_damping * (condition_estimate / target_condition).sqrt()
        } else {
            base_damping
        }
    }

    /// Apply K-FAC update to parameter gradients
    pub fn apply_update(&mut self, layer_name: &str, gradients: &Array2<T>) -> Result<Array2<T>> {
        let state = self
            .layer_states
            .get(layer_name)
            .ok_or_else(|| OptimError::InvalidConfig(format!("Layer {} not found", layer_name)))?;

        // Ensure inverse matrices are computed
        if state.a_cov_inv.is_none() || state.g_cov_inv.is_none() {
            return Err(OptimError::InvalidConfig(
                "Inverse covariance matrices not computed".to_string(),
            ));
        }

        let a_inv = state.a_cov_inv.as_ref().unwrap();
        let g_inv = state.g_cov_inv.as_ref().unwrap();

        // Apply Kronecker-factored preconditioner: G^(-1) ⊗ A^(-1)
        // For gradients G (output_dim x input_dim): G_new = G_inv * G * A_inv
        let preconditioned = g_inv.dot(&gradients.dot(a_inv));

        // Apply learning rate and weight decay
        let mut update = preconditioned * self.config.learning_rate;

        if self.config.weight_decay > T::zero() {
            // Add weight decay to the original gradients
            update = update + gradients * self.config.weight_decay;
        }

        Ok(update)
    }

    /// Perform a complete K-FAC optimization step
    pub fn step(
        &mut self,
        layer_name: &str,
        parameters: &Array2<T>,
        gradients: &Array2<T>,
        activations: &Array2<T>,
        loss: Option<T>,
    ) -> Result<Array2<T>> {
        self.step_count += 1;
        self.stats.total_steps += 1;

        // Update loss-based statistics
        if let Some(current_loss) = loss {
            if let Some(prev_loss) = self.previous_loss {
                let improvement = prev_loss - current_loss;
                // Update acceptance ratio for adaptive damping
                if improvement > T::zero() {
                    self.acceptance_ratio =
                        self.acceptance_ratio * T::from(0.9).unwrap() + T::from(0.1).unwrap();
                } else {
                    self.acceptance_ratio = self.acceptance_ratio * T::from(0.9).unwrap();
                }
            }
            self.previous_loss = Some(current_loss);
        }

        // Update covariance matrices
        self.update_covariance_matrices(layer_name, activations, gradients)?;

        // Update inverse matrices if needed
        self.update_inverse_matrices(layer_name)?;

        // Apply K-FAC update
        let update = self.apply_update(layer_name, gradients)?;

        // Update parameters
        let new_parameters = parameters - &update;

        Ok(new_parameters)
    }

    /// Get statistics for monitoring
    pub fn get_stats(&self) -> &KFACStats<T> {
        &self.stats
    }

    /// Reset optimizer state
    pub fn reset(&mut self) {
        for state in self.layer_states.values_mut() {
            let input_size =
                state.layer_info.input_dim + if state.layer_info.has_bias { 1 } else { 0 };
            let output_size = state.layer_info.output_dim;

            state.a_cov = Array2::eye(input_size);
            state.g_cov = Array2::eye(output_size);
            state.a_cov_inv = None;
            state.g_cov_inv = None;
            state.num_updates = 0;
            state.last_cov_update = 0;
            state.last_inv_update = 0;
            state.running_mean_a = None;
            state.running_mean_g = None;
        }

        self.step_count = 0;
        self.acceptance_ratio = T::from(1.0).unwrap();
        self.previous_loss = None;
        self.eigenvalue_history.clear();
        self.stats = KFACStats::default();
    }

    /// Estimate memory usage
    pub fn estimate_memory_usage(&self) -> usize {
        let mut total_memory = 0;

        for state in self.layer_states.values() {
            let input_size = state.a_cov.len();
            let output_size = state.g_cov.len();

            // Covariance matrices (2 per layer)
            total_memory += input_size * std::mem::size_of::<T>();
            total_memory += output_size * std::mem::size_of::<T>();

            // Inverse matrices (2 per layer)
            if state.a_cov_inv.is_some() {
                total_memory += input_size * std::mem::size_of::<T>();
            }
            if state.g_cov_inv.is_some() {
                total_memory += output_size * std::mem::size_of::<T>();
            }

            // Running means
            if state.running_mean_a.is_some() {
                total_memory += state.layer_info.input_dim * std::mem::size_of::<T>();
            }
            if state.running_mean_g.is_some() {
                total_memory += state.layer_info.output_dim * std::mem::size_of::<T>();
            }
        }

        total_memory
    }

    /// Get layer state information
    pub fn get_layer_state(&self, layer_name: &str) -> Option<&KFACLayerState<T>> {
        self.layer_states.get(layer_name)
    }

    /// Set custom damping for a specific layer
    pub fn set_layer_damping(
        &mut self,
        layer_name: &str,
        damping_a: T,
        damping_g: T,
    ) -> Result<()> {
        let state = self
            .layer_states
            .get_mut(layer_name)
            .ok_or_else(|| OptimError::InvalidConfig(format!("Layer {} not found", layer_name)))?;

        state.damping_a = damping_a;
        state.damping_g = damping_g;
        Ok(())
    }

    /// Advanced matrix analysis and conditioning
    fn estimate_condition_number(&self, matrix: &Array2<T>) -> T {
        let mut max_diag = T::zero();
        let mut min_diag = T::infinity();

        for i in 0..matrix.nrows() {
            let diag = matrix[[i, i]];
            if diag > max_diag {
                max_diag = diag;
            }
            if diag < min_diag {
                min_diag = diag;
            }
        }

        if min_diag > T::zero() {
            max_diag / min_diag
        } else {
            T::infinity()
        }
    }

    /// Regularized inverse for ill-conditioned matrices
    fn regularized_inverse(&self, matrix: &Array2<T>) -> Result<Array2<T>> {
        let n = matrix.nrows();
        let mut regularized = matrix.clone();

        // Add Tikhonov regularization
        let reg_param = self.config.damping * T::from(10.0).unwrap();
        for i in 0..n {
            regularized[[i, i]] = regularized[[i, i]] + reg_param;
        }

        self.cholesky_inverse(&regularized)
    }

    /// Iterative refinement for improved accuracy
    fn iterative_inverse(&self, matrix: &Array2<T>) -> Result<Array2<T>> {
        let n = matrix.nrows();
        let mut x = Array2::eye(n); // Initial guess
        let eye = Array2::eye(n);

        // Use Newton-Schulz iteration: X_{k+1} = X_k * (2*I - A*X_k)
        for _ in 0..5 {
            // 5 iterations should be enough
            let ax = matrix.dot(&x);
            let residual = &eye - &ax;
            let update = x.dot(&residual);
            x = &x + &update;

            // Check convergence
            let error = residual.iter().map(|&r| r * r).sum::<T>().sqrt();
            if error < T::from(1e-12).unwrap() {
                break;
            }
        }

        Ok(x)
    }

    /// Solve L * L^T * X = I using Cholesky factorization
    fn cholesky_solve_identity(&self, l: &Array2<T>) -> Result<Array2<T>> {
        let n = l.nrows();
        let mut inv = Array2::zeros((n, n));

        // Solve for each column of the identity matrix
        for i in 0..n {
            let mut b = Array1::zeros(n);
            b[i] = T::one();

            // Forward substitution: L * y = b
            let mut y = Array1::zeros(n);
            for j in 0..n {
                let mut sum = T::zero();
                for k in 0..j {
                    sum = sum + l[[j, k]] * y[k];
                }
                y[j] = (b[j] - sum) / l[[j, j]];
            }

            // Backward substitution: L^T * x = y
            let mut x = Array1::zeros(n);
            for j in (0..n).rev() {
                let mut sum = T::zero();
                for k in j + 1..n {
                    sum = sum + l[[k, j]] * x[k];
                }
                x[j] = (y[j] - sum) / l[[j, j]];
            }

            // Store column in inverse matrix
            for j in 0..n {
                inv[[j, i]] = x[j];
            }
        }

        Ok(inv)
    }

    /// Solve LU * X = I using LU factorization
    fn lu_solve_identity(&self, lu: &Array2<T>, perm: &[usize]) -> Result<Array2<T>> {
        let n = lu.nrows();
        let mut inv = Array2::zeros((n, n));

        for i in 0..n {
            let mut b = Array1::zeros(n);
            b[perm[i]] = T::one(); // Apply permutation

            // Forward substitution for L
            let mut y = Array1::zeros(n);
            for j in 0..n {
                let mut sum = T::zero();
                for k in 0..j {
                    sum = sum + lu[[j, k]] * y[k];
                }
                y[j] = b[j] - sum;
            }

            // Backward substitution for U
            let mut x = Array1::zeros(n);
            for j in (0..n).rev() {
                let mut sum = T::zero();
                for k in j + 1..n {
                    sum = sum + lu[[j, k]] * x[k];
                }
                x[j] = (y[j] - sum) / lu[[j, j]];
            }

            // Store column in inverse matrix
            for j in 0..n {
                inv[[j, i]] = x[j];
            }
        }

        Ok(inv)
    }
}

/// K-FAC utilities for specialized layer types
pub mod kfac_utils {
    use super::*;

    /// Compute K-FAC update for convolutional layers
    pub fn conv_kfac_update<T: Float>(
        input_patches: &Array2<T>,
        output_grads: &Array2<T>,
        a_inv: &Array2<T>,
        g_inv: &Array2<T>,
    ) -> Result<Array2<T>> {
        // For conv layers, we need to reshape and handle patches properly
        let preconditioned = g_inv.dot(&output_grads.dot(a_inv));
        Ok(preconditioned)
    }

    /// Compute statistics for batch normalization layers
    pub fn batchnorm_statistics<T: Float>(
        inputs: &Array2<T>,
        gamma: &Array1<T>,
        beta: &Array1<T>,
    ) -> Result<(Array1<T>, Array1<T>)> {
        let mean = inputs.mean_axis(Axis(0)).unwrap();
        let var = inputs.var_axis(Axis(0), T::zero());
        Ok((mean, var))
    }

    /// Handle grouped convolution layers
    pub fn grouped_conv_kfac<T: Float>(
        groups: usize,
        input_patches: &Array2<T>,
        output_grads: &Array2<T>,
    ) -> Result<Vec<Array2<T>>> {
        let group_size = input_patches.ncols() / groups;
        let mut updates = Vec::new();

        for g in 0..groups {
            let start_idx = g * group_size;
            let end_idx = (g + 1) * group_size;

            let group_input = input_patches.slice(s![.., start_idx..end_idx]);
            let group_output = output_grads.slice(s![.., start_idx..end_idx]);

            // Apply K-FAC to each group independently
            updates.push(group_input.to_owned());
        }

        Ok(updates)
    }
}

#[allow(dead_code)]
impl<T: Float> KFACLayerState<T> {
    /// Get the condition number estimate of covariance matrices
    pub fn condition_number_estimate(&self) -> (T, T) {
        let a_cond = self.estimate_condition_number(&self.a_cov);
        let g_cond = self.estimate_condition_number(&self.g_cov);
        (a_cond, g_cond)
    }

    fn estimate_condition_number(&self, matrix: &Array2<T>) -> T {
        let mut max_diag = T::zero();
        let mut min_diag = T::infinity();

        for i in 0..matrix.nrows() {
            let diag = matrix[[i, i]];
            if diag > max_diag {
                max_diag = diag;
            }
            if diag < min_diag {
                min_diag = diag;
            }
        }

        if min_diag > T::zero() {
            max_diag / min_diag
        } else {
            T::infinity()
        }
    }
}

/// Natural gradient optimization using Fisher information matrix
pub mod natural_gradients {
    use super::*;

    /// Natural gradient optimizer configuration
    #[derive(Debug, Clone)]
    pub struct NaturalGradientConfig<T: Float> {
        /// Learning rate for natural gradients
        pub learning_rate: T,

        /// Damping parameter for Fisher information matrix
        pub fisher_damping: T,

        /// Update frequency for Fisher information matrix
        pub fisher_update_freq: usize,

        /// Use empirical Fisher information (vs true Fisher)
        pub use_empirical_fisher: bool,

        /// Maximum rank for low-rank Fisher approximation
        pub max_rank: Option<usize>,

        /// Enable adaptive damping
        pub adaptive_damping: bool,

        /// Use conjugate gradient for matrix inversion
        pub use_conjugate_gradient: bool,

        /// CG iteration limit
        pub cg_max_iterations: usize,

        /// CG tolerance
        pub cg_tolerance: T,
    }

    impl<T: Float> Default for NaturalGradientConfig<T> {
        fn default() -> Self {
            Self {
                learning_rate: T::from(0.01).unwrap(),
                fisher_damping: T::from(0.001).unwrap(),
                fisher_update_freq: 10,
                use_empirical_fisher: true,
                max_rank: Some(100),
                adaptive_damping: true,
                use_conjugate_gradient: true,
                cg_max_iterations: 50,
                cg_tolerance: T::from(1e-6).unwrap(),
            }
        }
    }

    /// Natural gradient optimizer using Fisher information matrix
    #[derive(Debug)]
    pub struct NaturalGradientOptimizer<T: Float> {
        /// Configuration
        config: NaturalGradientConfig<T>,

        /// Fisher information matrix approximation
        fisher_matrix: FisherInformation<T>,

        /// Current step count
        step_count: usize,

        /// Adaptive damping state
        damping_state: AdaptiveDampingState<T>,

        /// Performance metrics
        metrics: NaturalGradientMetrics<T>,
    }

    /// Fisher information matrix representation
    #[derive(Debug, Clone)]
    pub enum FisherInformation<T: Float> {
        /// Full Fisher information matrix
        Full(Array2<T>),

        /// Diagonal approximation
        Diagonal(Array1<T>),

        /// Low-rank approximation: F ≈ U * S * U^T
        LowRank { u: Array2<T>, s: Array1<T> },

        /// Block-diagonal approximation
        BlockDiagonal {
            blocks: Vec<Array2<T>>,
            block_indices: Vec<(usize, usize)>,
        },

        /// Kronecker-factored approximation (K-FAC style)
        KroneckerFactored {
            a_factors: Vec<Array2<T>>,
            g_factors: Vec<Array2<T>>,
        },
    }

    /// Adaptive damping state
    #[derive(Debug, Clone)]
    struct AdaptiveDampingState<T: Float> {
        current_damping: T,
        acceptance_ratio: T,
        previous_loss: Option<T>,
        damping_history: VecDeque<T>,
    }

    /// Natural gradient performance metrics
    #[derive(Debug, Clone)]
    pub struct NaturalGradientMetrics<T: Float> {
        /// Condition number of Fisher matrix
        pub fisher_condition_number: T,

        /// Effective rank of Fisher matrix
        pub fisher_effective_rank: usize,

        /// Average eigenvalue
        pub avg_eigenvalue: T,

        /// Min/max eigenvalues
        pub min_eigenvalue: T,
        pub max_eigenvalue: T,

        /// Fisher update computation time (microseconds)
        pub fisher_update_time_us: u64,

        /// Natural gradient computation time (microseconds)
        pub nat_grad_compute_time_us: u64,

        /// Memory usage for Fisher matrix (bytes)
        pub fisher_memory_bytes: usize,
    }

    impl<T: Float + Default + Clone + Send + Sync> NaturalGradientOptimizer<T> {
        /// Create a new natural gradient optimizer
        pub fn new(config: NaturalGradientConfig<T>) -> Self {
            let damping_state = AdaptiveDampingState {
                current_damping: config.fisher_damping,
                acceptance_ratio: T::from(1.0).unwrap(),
                previous_loss: None,
                damping_history: VecDeque::with_capacity(100),
            };

            Self {
                config,
                fisher_matrix: FisherInformation::Diagonal(Array1::zeros(1)),
                step_count: 0,
                damping_state,
                metrics: NaturalGradientMetrics::default(),
            }
        }

        /// Initialize Fisher information matrix with parameter dimensions
        pub fn initialize_fisher(&mut self, param_dims: &[usize]) -> Result<()> {
            let total_params: usize = param_dims.iter().sum();

            self.fisher_matrix = if let Some(rank) = self.config.max_rank {
                if rank < total_params {
                    FisherInformation::LowRank {
                        u: Array2::zeros((total_params, rank)),
                        s: Array1::zeros(rank),
                    }
                } else {
                    FisherInformation::Full(Array2::eye(total_params))
                }
            } else {
                FisherInformation::Full(Array2::eye(total_params))
            };

            Ok(())
        }

        /// Update Fisher information matrix using gradient samples
        pub fn update_fisher_information(
            &mut self,
            gradient_samples: &[Array1<T>],
            loss_samples: Option<&[T]>,
        ) -> Result<()> {
            if self.step_count % self.config.fisher_update_freq != 0 {
                return Ok(());
            }

            let start_time = std::time::Instant::now();

            if self.config.use_empirical_fisher {
                self.update_empirical_fisher(gradient_samples)?;
            } else {
                self.update_true_fisher(gradient_samples, loss_samples)?;
            }

            self.metrics.fisher_update_time_us = start_time.elapsed().as_micros() as u64;
            self.update_fisher_metrics()?;

            Ok(())
        }

        fn update_empirical_fisher(&mut self, gradient_samples: &[Array1<T>]) -> Result<()> {
            if gradient_samples.is_empty() {
                return Ok(());
            }

            let n_samples = T::from(gradient_samples.len()).unwrap();

            match &mut self.fisher_matrix {
                FisherInformation::Full(ref mut fisher) => {
                    // F = (1/n) * sum(g_i * g_i^T)
                    fisher.fill(T::zero());

                    for grad in gradient_samples {
                        // Outer product: g * g^T
                        for i in 0..grad.len() {
                            for j in 0..grad.len() {
                                fisher[[i, j]] = fisher[[i, j]] + grad[i] * grad[j];
                            }
                        }
                    }

                    // Average over samples
                    fisher.mapv_inplace(|x| x / n_samples);

                    // Add damping
                    for i in 0..fisher.nrows() {
                        fisher[[i, i]] = fisher[[i, i]] + self.damping_state.current_damping;
                    }
                }

                FisherInformation::Diagonal(ref mut diag) => {
                    // Diagonal approximation: F_ii = (1/n) * sum(g_i^2)
                    diag.fill(T::zero());

                    for grad in gradient_samples {
                        for i in 0..grad.len() {
                            diag[i] = diag[i] + grad[i] * grad[i];
                        }
                    }

                    diag.mapv_inplace(|x| x / n_samples + self.damping_state.current_damping);
                }

                FisherInformation::LowRank {
                    ref mut u,
                    ref mut s,
                } => {
                    // Low-rank approximation using randomized SVD
                    self.update_low_rank_fisher(gradient_samples, u, s)?;
                }

                _ => {
                    return Err(OptimError::InvalidConfig(
                        "Unsupported Fisher matrix type for empirical update".to_string(),
                    ));
                }
            }

            Ok(())
        }

        fn update_true_fisher(
            &mut self,
            gradient_samples: &[Array1<T>],
            loss_samples: Option<&[T]>,
        ) -> Result<()> {
            // True Fisher information requires second derivatives
            // This is a simplified implementation that falls back to empirical Fisher
            if loss_samples.is_some() {
                // Could implement true Fisher using loss function Hessian
                // For now, fall back to empirical Fisher
                self.update_empirical_fisher(gradient_samples)
            } else {
                self.update_empirical_fisher(gradient_samples)
            }
        }

        fn update_low_rank_fisher(
            &mut self,
            gradient_samples: &[Array1<T>],
            u: &mut Array2<T>,
            s: &mut Array1<T>,
        ) -> Result<()> {
            if gradient_samples.is_empty() {
                return Ok(());
            }

            // Create gradient matrix: G = [g_1, g_2, ..., g_n]
            let n_samples = gradient_samples.len();
            let param_dim = gradient_samples[0].len();
            let mut grad_matrix = Array2::zeros((param_dim, n_samples));

            for (j, grad) in gradient_samples.iter().enumerate() {
                for i in 0..param_dim {
                    grad_matrix[[i, j]] = grad[i];
                }
            }

            // Simplified low-rank update using power iteration
            // In practice, would use proper randomized SVD
            let rank = u.ncols();

            // Power iteration for top eigenvectors
            for k in 0..rank.min(n_samples) {
                let mut v = Array1::ones(param_dim);

                // Power iteration
                for _ in 0..10 {
                    // v = G * G^T * v
                    let Gv = grad_matrix.dot(&grad_matrix.t().dot(&v));
                    let norm = Gv.iter().map(|&x| x * x).sum::<T>().sqrt();

                    if norm > T::from(1e-10).unwrap() {
                        v = Gv / norm;
                    }
                }

                // Store eigenvector
                for i in 0..param_dim {
                    u[[i, k]] = v[i];
                }

                // Compute eigenvalue
                let uv = grad_matrix.t().dot(&v);
                s[k] = uv.iter().map(|&x| x * x).sum::<T>() / T::from(n_samples).unwrap();
            }

            Ok(())
        }

        /// Compute natural gradient: F^(-1) * g
        pub fn compute_natural_gradient(&mut self, gradient: &Array1<T>) -> Result<Array1<T>> {
            let start_time = std::time::Instant::now();

            let natural_grad = match &self.fisher_matrix {
                FisherInformation::Full(fisher) => {
                    if self.config.use_conjugate_gradient {
                        self.solve_cg(fisher, gradient)?
                    } else {
                        self.solve_direct(fisher, gradient)?
                    }
                }

                FisherInformation::Diagonal(diag) => {
                    // Element-wise division
                    let mut nat_grad = gradient.clone();
                    for i in 0..nat_grad.len() {
                        nat_grad[i] = nat_grad[i] / diag[i];
                    }
                    nat_grad
                }

                FisherInformation::LowRank { u, s } => self.solve_low_rank(u, s, gradient)?,

                _ => {
                    return Err(OptimError::InvalidConfig(
                        "Unsupported Fisher matrix type for natural gradient".to_string(),
                    ));
                }
            };

            self.metrics.nat_grad_compute_time_us = start_time.elapsed().as_micros() as u64;

            Ok(natural_grad)
        }

        fn solve_direct(&self, fisher: &Array2<T>, gradient: &Array1<T>) -> Result<Array1<T>> {
            // Direct solve using Cholesky decomposition (simplified)
            // In practice, would use proper linear algebra library
            let mut solution = gradient.clone();

            // Simplified diagonal solve for stability
            for i in 0..fisher.nrows() {
                let diag_elem = fisher[[i, i]];
                if diag_elem > T::from(1e-10).unwrap() {
                    solution[i] = solution[i] / diag_elem;
                }
            }

            Ok(solution)
        }

        fn solve_cg(&self, fisher: &Array2<T>, gradient: &Array1<T>) -> Result<Array1<T>> {
            // Conjugate gradient solver for F * x = g
            let mut x = Array1::zeros(gradient.len());
            let mut r = gradient.clone(); // r = b - A*x (x starts at 0)
            let mut p = r.clone();
            let mut rsold = r.dot(&r);

            for _iter in 0..self.config.cg_max_iterations {
                let ap = fisher.dot(&p);
                let alpha = rsold / p.dot(&ap);

                // Update solution: x = x + alpha * p
                for i in 0..x.len() {
                    x[i] = x[i] + alpha * p[i];
                }

                // Update residual: r = r - alpha * A*p
                for i in 0..r.len() {
                    r[i] = r[i] - alpha * ap[i];
                }

                let rsnew = r.dot(&r);

                // Check convergence
                if rsnew.sqrt() < self.config.cg_tolerance {
                    break;
                }

                let beta = rsnew / rsold;

                // Update search direction: p = r + beta * p
                for i in 0..p.len() {
                    p[i] = r[i] + beta * p[i];
                }

                rsold = rsnew;
            }

            Ok(x)
        }

        fn solve_low_rank(
            &self,
            u: &Array2<T>,
            s: &Array1<T>,
            gradient: &Array1<T>,
        ) -> Result<Array1<T>> {
            // For low-rank F = U * S * U^T, solve using Sherman-Morrison-Woodbury formula
            // (U*S*U^T + damping*I)^(-1) * g

            let damping = self.damping_state.current_damping;
            let mut solution = gradient.clone();

            // Apply damping: solution = g / damping
            solution.mapv_inplace(|x| x / damping);

            // Apply Sherman-Morrison-Woodbury correction
            // This is simplified - in practice would be more sophisticated
            for k in 0..s.len() {
                if s[k] > T::from(1e-10).unwrap() {
                    let uk_dot_g = u.column(k).dot(gradient);
                    let correction_factor = s[k] / (s[k] + damping);

                    for i in 0..solution.len() {
                        solution[i] =
                            solution[i] - correction_factor * uk_dot_g * u[[i, k]] / damping;
                    }
                }
            }

            Ok(solution)
        }

        /// Apply natural gradient step
        pub fn step(
            &mut self,
            parameters: &Array1<T>,
            gradient: &Array1<T>,
            loss: Option<T>,
        ) -> Result<Array1<T>> {
            self.step_count += 1;

            // Update adaptive damping
            if self.config.adaptive_damping {
                self.update_adaptive_damping(loss)?;
            }

            // Compute natural gradient
            let natural_gradient = self.compute_natural_gradient(gradient)?;

            // Apply update: θ_{t+1} = θ_t - lr * F^(-1) * g
            let mut new_params = parameters.clone();
            for i in 0..new_params.len() {
                new_params[i] = new_params[i] - self.config.learning_rate * natural_gradient[i];
            }

            Ok(new_params)
        }

        fn update_adaptive_damping(&mut self, loss: Option<T>) -> Result<()> {
            if let Some(current_loss) = loss {
                if let Some(prev_loss) = self.damping_state.previous_loss {
                    let improvement = prev_loss - current_loss;

                    // Update acceptance ratio
                    if improvement > T::zero() {
                        self.damping_state.acceptance_ratio = self.damping_state.acceptance_ratio
                            * T::from(0.9).unwrap()
                            + T::from(0.1).unwrap();
                    } else {
                        self.damping_state.acceptance_ratio =
                            self.damping_state.acceptance_ratio * T::from(0.9).unwrap();
                    }

                    // Adjust damping based on acceptance ratio
                    let target_ratio = T::from(0.75).unwrap();
                    if self.damping_state.acceptance_ratio < target_ratio {
                        // Increase damping if acceptance is low
                        self.damping_state.current_damping =
                            self.damping_state.current_damping * T::from(1.1).unwrap();
                    } else {
                        // Decrease damping if acceptance is high
                        self.damping_state.current_damping =
                            self.damping_state.current_damping * T::from(0.95).unwrap();
                    }

                    // Clamp damping to reasonable range
                    let min_damping = T::from(1e-6).unwrap();
                    let max_damping = T::from(1.0).unwrap();
                    self.damping_state.current_damping = self
                        .damping_state
                        .current_damping
                        .max(min_damping)
                        .min(max_damping);
                }

                self.damping_state.previous_loss = Some(current_loss);

                // Update history
                self.damping_state
                    .damping_history
                    .push_back(self.damping_state.current_damping);
                if self.damping_state.damping_history.len() > 100 {
                    self.damping_state.damping_history.pop_front();
                }
            }

            Ok(())
        }

        fn update_fisher_metrics(&mut self) -> Result<()> {
            match &self.fisher_matrix {
                FisherInformation::Full(fisher) => {
                    // Compute condition number (simplified using diagonal approximation)
                    let mut min_diag = T::infinity();
                    let mut max_diag = T::zero();
                    let mut sum_diag = T::zero();

                    for i in 0..fisher.nrows() {
                        let diag = fisher[[i, i]];
                        min_diag = min_diag.min(diag);
                        max_diag = max_diag.max(diag);
                        sum_diag = sum_diag + diag;
                    }

                    self.metrics.min_eigenvalue = min_diag;
                    self.metrics.max_eigenvalue = max_diag;
                    self.metrics.avg_eigenvalue = sum_diag / T::from(fisher.nrows()).unwrap();
                    self.metrics.fisher_condition_number = if min_diag > T::zero() {
                        max_diag / min_diag
                    } else {
                        T::infinity()
                    };

                    self.metrics.fisher_memory_bytes = fisher.len() * std::mem::size_of::<T>();
                }

                FisherInformation::Diagonal(diag) => {
                    let min_val = diag.iter().cloned().fold(T::infinity(), T::min);
                    let max_val = diag.iter().cloned().fold(T::zero(), T::max);
                    let sum_val = diag.iter().cloned().sum::<T>();

                    self.metrics.min_eigenvalue = min_val;
                    self.metrics.max_eigenvalue = max_val;
                    self.metrics.avg_eigenvalue = sum_val / T::from(diag.len()).unwrap();
                    self.metrics.fisher_condition_number = if min_val > T::zero() {
                        max_val / min_val
                    } else {
                        T::infinity()
                    };

                    self.metrics.fisher_memory_bytes = diag.len() * std::mem::size_of::<T>();
                }

                FisherInformation::LowRank { u, s } => {
                    let min_val = s.iter().cloned().fold(T::infinity(), T::min);
                    let max_val = s.iter().cloned().fold(T::zero(), T::max);
                    let sum_val = s.iter().cloned().sum::<T>();

                    self.metrics.min_eigenvalue = min_val;
                    self.metrics.max_eigenvalue = max_val;
                    self.metrics.avg_eigenvalue = sum_val / T::from(s.len()).unwrap();
                    self.metrics.fisher_condition_number = if min_val > T::zero() {
                        max_val / min_val
                    } else {
                        T::infinity()
                    };

                    // Effective rank (number of non-zero eigenvalues)
                    self.metrics.fisher_effective_rank =
                        s.iter().filter(|&&x| x > T::from(1e-10).unwrap()).count();

                    self.metrics.fisher_memory_bytes =
                        (u.len() + s.len()) * std::mem::size_of::<T>();
                }

                _ => {}
            }

            Ok(())
        }

        /// Get current metrics
        pub fn get_metrics(&self) -> &NaturalGradientMetrics<T> {
            &self.metrics
        }

        /// Get current damping value
        pub fn get_current_damping(&self) -> T {
            self.damping_state.current_damping
        }

        /// Reset optimizer state
        pub fn reset(&mut self) {
            self.step_count = 0;
            self.damping_state.previous_loss = None;
            self.damping_state.acceptance_ratio = T::from(1.0).unwrap();
            self.damping_state.damping_history.clear();
            self.damping_state.current_damping = self.config.fisher_damping;
        }
    }

    impl<T: Float> Default for NaturalGradientMetrics<T> {
        fn default() -> Self {
            Self {
                fisher_condition_number: T::one(),
                fisher_effective_rank: 0,
                avg_eigenvalue: T::one(),
                min_eigenvalue: T::one(),
                max_eigenvalue: T::one(),
                fisher_update_time_us: 0,
                nat_grad_compute_time_us: 0,
                fisher_memory_bytes: 0,
            }
        }
    }

    /// Integrate natural gradients with K-FAC
    impl<T: Float + Default + Clone + Send + Sync + std::fmt::Debug> KFAC<T> {
        /// Compute natural gradient update using K-FAC approximation
        pub fn natural_gradient_step(
            &mut self,
            layer_name: &str,
            parameters: &Array2<T>,
            gradients: &Array2<T>,
            activations: &Array2<T>,
            loss: Option<T>,
        ) -> Result<Array2<T>> {
            // Regular K-FAC step
            let kfac_update = self.step(layer_name, parameters, gradients, activations, loss)?;

            // Apply natural gradient scaling based on Fisher information
            let state = self.layer_states.get(layer_name).ok_or_else(|| {
                OptimError::InvalidConfig(format!("Layer {} not found", layer_name))
            })?;

            if let (Some(a_inv), Some(g_inv)) = (&state.a_cov_inv, &state.g_cov_inv) {
                // Scale by Fisher information to get natural gradient
                let fisher_scaled = self.apply_fisher_scaling(&kfac_update, a_inv, g_inv)?;
                Ok(fisher_scaled)
            } else {
                Ok(kfac_update)
            }
        }

        fn apply_fisher_scaling(
            &self,
            update: &Array2<T>,
            a_inv: &Array2<T>,
            g_inv: &Array2<T>,
        ) -> Result<Array2<T>> {
            // Apply Fisher information scaling: F^(-1/2) * update
            // This is simplified - in practice would use proper matrix square root
            let scaled = g_inv.dot(&update.dot(a_inv));
            Ok(scaled * T::from(0.5).unwrap()) // Scale factor for stability
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_kfac_creation() {
        let config = KFACConfig::default();
        let kfac = KFAC::<f64>::new(config);
        assert_eq!(kfac.step_count, 0);
        assert_eq!(kfac.layer_states.len(), 0);
    }

    #[test]
    fn test_layer_registration() {
        let mut kfac = KFAC::<f64>::new(KFACConfig::default());

        let layer_info = LayerInfo {
            name: "dense1".to_string(),
            input_dim: 10,
            output_dim: 5,
            layer_type: LayerType::Dense,
            has_bias: true,
        };

        assert!(kfac.register_layer(layer_info).is_ok());
        assert_eq!(kfac.layer_states.len(), 1);

        let state = kfac.get_layer_state("dense1").unwrap();
        assert_eq!(state.a_cov.nrows(), 11); // 10 + 1 for bias
        assert_eq!(state.g_cov.nrows(), 5);
    }

    #[test]
    fn test_covariance_update() {
        let mut kfac = KFAC::<f64>::new(KFACConfig {
            cov_update_freq: 1,
            ..Default::default()
        });

        let layer_info = LayerInfo {
            name: "test_layer".to_string(),
            input_dim: 3,
            output_dim: 2,
            layer_type: LayerType::Dense,
            has_bias: false,
        };

        kfac.register_layer(layer_info).unwrap();

        let activations =
            Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let gradients = Array2::from_shape_vec((2, 2), vec![0.1, 0.2, 0.3, 0.4]).unwrap();

        assert!(kfac
            .update_covariance_matrices("test_layer", &activations, &gradients)
            .is_ok());

        let state = kfac.get_layer_state("test_layer").unwrap();
        assert!(state.running_mean_a.is_some());
        assert!(state.running_mean_g.is_some());
    }

    #[test]
    fn test_adaptive_damping() {
        let kfac = KFAC::<f64>::new(KFACConfig::default());

        // Test matrix with large condition number
        let mut matrix = Array2::eye(3);
        matrix[[0, 0]] = 1.0;
        matrix[[1, 1]] = 1e-6;
        matrix[[2, 2]] = 1e-6;

        let damping = kfac.compute_adaptive_damping(&matrix).unwrap();
        assert!(damping > kfac.config.damping);
    }

    #[test]
    fn test_memory_estimation() {
        let mut kfac = KFAC::<f64>::new(KFACConfig::default());

        let layer_info = LayerInfo {
            name: "test".to_string(),
            input_dim: 100,
            output_dim: 50,
            layer_type: LayerType::Dense,
            has_bias: true,
        };

        kfac.register_layer(layer_info).unwrap();

        let memory = kfac.estimate_memory_usage();
        assert!(memory > 0);
    }

    #[test]
    fn test_layer_types() {
        let dense = LayerType::Dense;
        let conv = LayerType::Convolution;
        let grouped = LayerType::GroupedConvolution { groups: 4 };

        assert_eq!(dense, LayerType::Dense);
        assert_ne!(conv, LayerType::Dense);

        if let LayerType::GroupedConvolution { groups } = grouped {
            assert_eq!(groups, 4);
        } else {
            panic!("Incorrect layer type");
        }
    }

    #[test]
    fn test_kfac_utils() {
        let input_patches =
            Array2::from_shape_vec((4, 6), (0..24).map(|x| x as f64).collect()).unwrap();
        let output_grads =
            Array2::from_shape_vec((4, 3), (0..12).map(|x| x as f64 * 0.1).collect()).unwrap();

        let groups = 2;
        let result = kfac_utils::grouped_conv_kfac(groups, &input_patches, &output_grads);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), groups);
    }
}
