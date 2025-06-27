//! Lie group integrators
//!
//! This module provides numerical integrators for differential equations on Lie groups,
//! preserving the group structure throughout the integration.

use ndarray::{Array1, Array2, ArrayView1};
use crate::error::IntegrateResult as Result;
use scirs2_core::constants::PI;

/// Trait for Lie algebra operations
pub trait LieAlgebra: Clone {
    /// Dimension of the Lie algebra
    fn dim() -> usize;
    
    /// Lie bracket [X, Y] = XY - YX
    fn bracket(&self, other: &Self) -> Self;
    
    /// Convert from vector representation to algebra element
    fn from_vector(v: &ArrayView1<f64>) -> Self;
    
    /// Convert to vector representation
    fn to_vector(&self) -> Array1<f64>;
    
    /// Norm of the algebra element
    fn norm(&self) -> f64;
}

/// Trait for exponential map on Lie groups
pub trait ExponentialMap: Sized {
    /// Associated Lie algebra type
    type Algebra: LieAlgebra;
    
    /// Exponential map from Lie algebra to Lie group
    fn exp(algebra: &Self::Algebra) -> Self;
    
    /// Logarithm map from Lie group to Lie algebra
    fn log(&self) -> Self::Algebra;
    
    /// Group multiplication
    fn multiply(&self, other: &Self) -> Self;
    
    /// Group inverse
    fn inverse(&self) -> Self;
    
    /// Identity element
    fn identity() -> Self;
}

/// General Lie group integrator
pub struct LieGroupIntegrator<G: ExponentialMap> {
    /// Time step
    pub dt: f64,
    /// Integration method
    pub method: LieGroupMethod,
    /// Phantom data for group type
    _phantom: std::marker::PhantomData<G>,
}

/// Available Lie group integration methods
#[derive(Debug, Clone, Copy)]
pub enum LieGroupMethod {
    /// Lie-Euler method (first order)
    LieEuler,
    /// Lie-Midpoint method (second order)
    LieMidpoint,
    /// Lie-Trapezoidal method (second order)
    LieTrapezoidal,
    /// Runge-Kutta-Munthe-Kaas method (fourth order)
    RKMK4,
    /// Crouch-Grossman method
    CrouchGrossman,
}

impl<G: ExponentialMap> LieGroupIntegrator<G> {
    /// Create a new Lie group integrator
    pub fn new(dt: f64, method: LieGroupMethod) -> Self {
        Self {
            dt,
            method,
            _phantom: std::marker::PhantomData,
        }
    }
    
    /// Integrate one step
    pub fn step<F>(&self, g: &G, f: F) -> Result<G>
    where
        F: Fn(&G) -> G::Algebra,
    {
        match self.method {
            LieGroupMethod::LieEuler => self.lie_euler_step(g, f),
            LieGroupMethod::LieMidpoint => self.lie_midpoint_step(g, f),
            LieGroupMethod::LieTrapezoidal => self.lie_trapezoidal_step(g, f),
            LieGroupMethod::RKMK4 => self.rkmk4_step(g, f),
            LieGroupMethod::CrouchGrossman => self.crouch_grossman_step(g, f),
        }
    }
    
    /// Lie-Euler method: g_{n+1} = g_n * exp(dt * f(g_n))
    fn lie_euler_step<F>(&self, g: &G, f: F) -> Result<G>
    where
        F: Fn(&G) -> G::Algebra,
    {
        let xi = f(g);
        let mut scaled_xi = xi.to_vector();
        scaled_xi *= self.dt;
        let scaled_algebra = G::Algebra::from_vector(&scaled_xi.view());
        Ok(g.multiply(&G::exp(&scaled_algebra)))
    }
    
    /// Lie-Midpoint method
    fn lie_midpoint_step<F>(&self, g: &G, f: F) -> Result<G>
    where
        F: Fn(&G) -> G::Algebra,
    {
        let xi1 = f(g);
        let mut half_xi = xi1.to_vector();
        half_xi *= self.dt / 2.0;
        let half_algebra = G::Algebra::from_vector(&half_xi.view());
        let g_mid = g.multiply(&G::exp(&half_algebra));
        
        let xi_mid = f(&g_mid);
        let mut full_xi = xi_mid.to_vector();
        full_xi *= self.dt;
        let full_algebra = G::Algebra::from_vector(&full_xi.view());
        Ok(g.multiply(&G::exp(&full_algebra)))
    }
    
    /// Lie-Trapezoidal method
    fn lie_trapezoidal_step<F>(&self, g: &G, f: F) -> Result<G>
    where
        F: Fn(&G) -> G::Algebra,
    {
        let xi1 = f(g);
        let mut full_xi = xi1.to_vector();
        full_xi *= self.dt;
        let full_algebra = G::Algebra::from_vector(&full_xi.view());
        let g_euler = g.multiply(&G::exp(&full_algebra));
        
        let xi2 = f(&g_euler);
        let mut avg_xi = (xi1.to_vector() + xi2.to_vector()) / 2.0;
        avg_xi *= self.dt;
        let avg_algebra = G::Algebra::from_vector(&avg_xi.view());
        Ok(g.multiply(&G::exp(&avg_algebra)))
    }
    
    /// Runge-Kutta-Munthe-Kaas 4th order method
    fn rkmk4_step<F>(&self, g: &G, f: F) -> Result<G>
    where
        F: Fn(&G) -> G::Algebra,
    {
        // k1
        let k1 = f(g);
        
        // k2
        let mut exp_arg = k1.to_vector() * (self.dt / 2.0);
        let exp_k1_2 = G::exp(&G::Algebra::from_vector(&exp_arg.view()));
        let g2 = g.multiply(&exp_k1_2);
        let k2 = f(&g2);
        
        // k3
        exp_arg = k2.to_vector() * (self.dt / 2.0);
        let exp_k2_2 = G::exp(&G::Algebra::from_vector(&exp_arg.view()));
        let g3 = g.multiply(&exp_k2_2);
        let k3 = f(&g3);
        
        // k4
        exp_arg = k3.to_vector() * self.dt;
        let exp_k3 = G::exp(&G::Algebra::from_vector(&exp_arg.view()));
        let g4 = g.multiply(&exp_k3);
        let k4 = f(&g4);
        
        // Combine using BCH formula approximation
        let combined = (k1.to_vector() + k2.to_vector() * 2.0 + k3.to_vector() * 2.0 + k4.to_vector()) / 6.0;
        let mut final_xi = combined * self.dt;
        
        // Second-order BCH correction
        let comm = k1.bracket(&k2);
        final_xi = final_xi + comm.to_vector() * (self.dt * self.dt / 12.0);
        
        let final_algebra = G::Algebra::from_vector(&final_xi.view());
        Ok(g.multiply(&G::exp(&final_algebra)))
    }
    
    /// Crouch-Grossman method
    fn crouch_grossman_step<F>(&self, g: &G, f: F) -> Result<G>
    where
        F: Fn(&G) -> G::Algebra,
    {
        let c2 = 0.5;
        let c3 = 1.0;
        let b1 = 1.0 / 6.0;
        let b2 = 2.0 / 3.0;
        let b3 = 1.0 / 6.0;
        
        // Stage 1
        let k1 = f(g);
        
        // Stage 2
        let mut exp_arg = k1.to_vector() * (c2 * self.dt);
        let y2 = g.multiply(&G::exp(&G::Algebra::from_vector(&exp_arg.view())));
        let k2 = f(&y2);
        
        // Stage 3
        exp_arg = k2.to_vector() * (c3 * self.dt);
        let y3 = g.multiply(&G::exp(&G::Algebra::from_vector(&exp_arg.view())));
        let k3 = f(&y3);
        
        // Final update
        let combined = k1.to_vector() * b1 + k2.to_vector() * b2 + k3.to_vector() * b3;
        exp_arg = combined * self.dt;
        Ok(g.multiply(&G::exp(&G::Algebra::from_vector(&exp_arg.view()))))
    }
}

/// SO(3) Lie algebra (skew-symmetric 3x3 matrices)
#[derive(Debug, Clone)]
pub struct So3 {
    /// Components [ω_x, ω_y, ω_z] representing angular velocity
    pub omega: Array1<f64>,
}

impl LieAlgebra for So3 {
    fn dim() -> usize {
        3
    }
    
    fn bracket(&self, other: &Self) -> Self {
        // [ω1, ω2] = ω1 × ω2 (cross product)
        let omega = Array1::from_vec(vec![
            self.omega[1] * other.omega[2] - self.omega[2] * other.omega[1],
            self.omega[2] * other.omega[0] - self.omega[0] * other.omega[2],
            self.omega[0] * other.omega[1] - self.omega[1] * other.omega[0],
        ]);
        So3 { omega }
    }
    
    fn from_vector(v: &ArrayView1<f64>) -> Self {
        So3 {
            omega: v.to_owned(),
        }
    }
    
    fn to_vector(&self) -> Array1<f64> {
        self.omega.clone()
    }
    
    fn norm(&self) -> f64 {
        self.omega.dot(&self.omega).sqrt()
    }
}

impl So3 {
    /// Convert to 3x3 skew-symmetric matrix
    pub fn to_matrix(&self) -> Array2<f64> {
        let mut mat = Array2::zeros((3, 3));
        mat[[0, 1]] = -self.omega[2];
        mat[[0, 2]] = self.omega[1];
        mat[[1, 0]] = self.omega[2];
        mat[[1, 2]] = -self.omega[0];
        mat[[2, 0]] = -self.omega[1];
        mat[[2, 1]] = self.omega[0];
        mat
    }
}

/// SO(3) Lie group (3D rotation matrices)
#[derive(Debug, Clone)]
pub struct SO3 {
    /// Rotation matrix
    pub matrix: Array2<f64>,
}

impl ExponentialMap for SO3 {
    type Algebra = So3;
    
    fn exp(algebra: &Self::Algebra) -> Self {
        let theta = algebra.norm();
        
        if theta < 1e-10 {
            // Small angle approximation
            SO3 {
                matrix: Array2::eye(3) + algebra.to_matrix(),
            }
        } else {
            // Rodrigues' formula
            let k = algebra.to_matrix() / theta;
            let k2 = k.dot(&k);
            
            SO3 {
                matrix: Array2::eye(3) + k * theta.sin() + k2 * (1.0 - theta.cos()),
            }
        }
    }
    
    fn log(&self) -> Self::Algebra {
        // Extract axis-angle from rotation matrix
        let trace = self.matrix[[0, 0]] + self.matrix[[1, 1]] + self.matrix[[2, 2]];
        let theta = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0).acos();
        
        if theta.abs() < 1e-10 {
            // Small rotation
            So3 {
                omega: Array1::from_vec(vec![
                    self.matrix[[2, 1]] - self.matrix[[1, 2]],
                    self.matrix[[0, 2]] - self.matrix[[2, 0]],
                    self.matrix[[1, 0]] - self.matrix[[0, 1]],
                ]) / 2.0,
            }
        } else {
            // General case
            let factor = theta / (2.0 * theta.sin());
            So3 {
                omega: Array1::from_vec(vec![
                    self.matrix[[2, 1]] - self.matrix[[1, 2]],
                    self.matrix[[0, 2]] - self.matrix[[2, 0]],
                    self.matrix[[1, 0]] - self.matrix[[0, 1]],
                ]) * factor,
            }
        }
    }
    
    fn multiply(&self, other: &Self) -> Self {
        SO3 {
            matrix: self.matrix.dot(&other.matrix),
        }
    }
    
    fn inverse(&self) -> Self {
        SO3 {
            matrix: self.matrix.t().to_owned(),
        }
    }
    
    fn identity() -> Self {
        SO3 {
            matrix: Array2::eye(3),
        }
    }
}

/// Integrator specifically for SO(3)
pub struct SO3Integrator {
    /// Base Lie group integrator
    base: LieGroupIntegrator<SO3>,
}

impl SO3Integrator {
    /// Create a new SO(3) integrator
    pub fn new(dt: f64, method: LieGroupMethod) -> Self {
        Self {
            base: LieGroupIntegrator::new(dt, method),
        }
    }
    
    /// Integrate rigid body dynamics
    pub fn integrate_rigid_body(
        &self,
        orientation: &SO3,
        angular_velocity: &Array1<f64>,
        inertia_tensor: &Array2<f64>,
        external_torque: &Array1<f64>,
        n_steps: usize,
    ) -> Result<Vec<(SO3, Array1<f64>)>> {
        let mut states = vec![(orientation.clone(), angular_velocity.clone())];
        let mut current_orientation = orientation.clone();
        let mut current_omega = angular_velocity.clone();
        
        let inertia_inv = self.invert_inertia(inertia_tensor)?;
        
        for _ in 0..n_steps {
            // Compute angular acceleration
            let omega_cross_I_omega = self.cross_product(&current_omega, &inertia_tensor.dot(&current_omega));
            let angular_accel = inertia_inv.dot(&(external_torque - &omega_cross_I_omega));
            
            // Update angular velocity
            current_omega = &current_omega + &angular_accel * self.base.dt;
            
            // Update orientation
            let omega_algebra = So3 { omega: current_omega.clone() };
            current_orientation = self.base.step(&current_orientation, |_| omega_algebra.clone())?;
            
            states.push((current_orientation.clone(), current_omega.clone()));
        }
        
        Ok(states)
    }
    
    /// Cross product for 3D vectors
    fn cross_product(&self, a: &Array1<f64>, b: &Array1<f64>) -> Array1<f64> {
        Array1::from_vec(vec![
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ])
    }
    
    /// Invert 3x3 matrix (for inertia tensor)
    fn invert_inertia(&self, inertia: &Array2<f64>) -> Result<Array2<f64>> {
        // Simple 3x3 matrix inversion
        let det = inertia[[0, 0]] * (inertia[[1, 1]] * inertia[[2, 2]] - inertia[[1, 2]] * inertia[[2, 1]])
                - inertia[[0, 1]] * (inertia[[1, 0]] * inertia[[2, 2]] - inertia[[1, 2]] * inertia[[2, 0]])
                + inertia[[0, 2]] * (inertia[[1, 0]] * inertia[[2, 1]] - inertia[[1, 1]] * inertia[[2, 0]]);
        
        if det.abs() < 1e-10 {
            return Err(crate::error::IntegrateError::ValueError(
                "Singular inertia tensor".to_string()
            ));
        }
        
        let mut inv = Array2::zeros((3, 3));
        inv[[0, 0]] = (inertia[[1, 1]] * inertia[[2, 2]] - inertia[[1, 2]] * inertia[[2, 1]]) / det;
        inv[[0, 1]] = (inertia[[0, 2]] * inertia[[2, 1]] - inertia[[0, 1]] * inertia[[2, 2]]) / det;
        inv[[0, 2]] = (inertia[[0, 1]] * inertia[[1, 2]] - inertia[[0, 2]] * inertia[[1, 1]]) / det;
        inv[[1, 0]] = (inertia[[1, 2]] * inertia[[2, 0]] - inertia[[1, 0]] * inertia[[2, 2]]) / det;
        inv[[1, 1]] = (inertia[[0, 0]] * inertia[[2, 2]] - inertia[[0, 2]] * inertia[[2, 0]]) / det;
        inv[[1, 2]] = (inertia[[0, 2]] * inertia[[1, 0]] - inertia[[0, 0]] * inertia[[1, 2]]) / det;
        inv[[2, 0]] = (inertia[[1, 0]] * inertia[[2, 1]] - inertia[[1, 1]] * inertia[[2, 0]]) / det;
        inv[[2, 1]] = (inertia[[0, 1]] * inertia[[2, 0]] - inertia[[0, 0]] * inertia[[2, 1]]) / det;
        inv[[2, 2]] = (inertia[[0, 0]] * inertia[[1, 1]] - inertia[[0, 1]] * inertia[[1, 0]]) / det;
        
        Ok(inv)
    }
}

/// SE(3) Lie algebra (rigid body motions)
#[derive(Debug, Clone)]
pub struct Se3 {
    /// Linear velocity
    pub v: Array1<f64>,
    /// Angular velocity
    pub omega: Array1<f64>,
}

impl LieAlgebra for Se3 {
    fn dim() -> usize {
        6
    }
    
    fn bracket(&self, other: &Self) -> Self {
        // [ξ1, ξ2] = ad_ξ1(ξ2)
        let omega_bracket = So3 { omega: self.omega.clone() }
            .bracket(&So3 { omega: other.omega.clone() });
        
        let v_bracket = self.cross_3d(&self.omega, &other.v) 
                      - self.cross_3d(&other.omega, &self.v);
        
        Se3 {
            v: v_bracket,
            omega: omega_bracket.omega,
        }
    }
    
    fn from_vector(v: &ArrayView1<f64>) -> Self {
        Se3 {
            v: v.slice(ndarray::s![0..3]).to_owned(),
            omega: v.slice(ndarray::s![3..6]).to_owned(),
        }
    }
    
    fn to_vector(&self) -> Array1<f64> {
        let mut vec = Array1::zeros(6);
        vec.slice_mut(ndarray::s![0..3]).assign(&self.v);
        vec.slice_mut(ndarray::s![3..6]).assign(&self.omega);
        vec
    }
    
    fn norm(&self) -> f64 {
        (self.v.dot(&self.v) + self.omega.dot(&self.omega)).sqrt()
    }
}

impl Se3 {
    fn cross_3d(&self, a: &Array1<f64>, b: &Array1<f64>) -> Array1<f64> {
        Array1::from_vec(vec![
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ])
    }
}

/// SE(3) Lie group (rigid body transformations)
#[derive(Debug, Clone)]
pub struct SE3 {
    /// Rotation part (SO(3))
    pub rotation: SO3,
    /// Translation part
    pub translation: Array1<f64>,
}

impl ExponentialMap for SE3 {
    type Algebra = Se3;
    
    fn exp(algebra: &Self::Algebra) -> Self {
        let omega_norm = algebra.omega.dot(&algebra.omega).sqrt();
        let rotation = SO3::exp(&So3 { omega: algebra.omega.clone() });
        
        let translation = if omega_norm < 1e-10 {
            // Small angle: V ≈ I
            algebra.v.clone()
        } else {
            // General case: compute V matrix
            let axis = &algebra.omega / omega_norm;
            let axis_cross = So3 { omega: axis.clone() }.to_matrix();
            let axis_cross2 = axis_cross.dot(&axis_cross);
            
            let v_matrix = Array2::eye(3) 
                        + axis_cross * ((1.0 - omega_norm.cos()) / omega_norm)
                        + axis_cross2 * ((omega_norm - omega_norm.sin()) / omega_norm);
            
            v_matrix.dot(&algebra.v)
        };
        
        SE3 {
            rotation,
            translation,
        }
    }
    
    fn log(&self) -> Self::Algebra {
        let omega_algebra = self.rotation.log();
        let omega = &omega_algebra.omega;
        let omega_norm = omega.dot(omega).sqrt();
        
        let v = if omega_norm < 1e-10 {
            // Small rotation: V^(-1) ≈ I
            self.translation.clone()
        } else {
            // General case: compute V^(-1)
            let axis = omega / omega_norm;
            let axis_cross = So3 { omega: axis.clone() }.to_matrix();
            let axis_cross2 = axis_cross.dot(&axis_cross);
            
            let cot_half = 1.0 / (omega_norm / 2.0).tan();
            let v_inv = Array2::eye(3) 
                      - axis_cross / 2.0
                      + axis_cross2 * (1.0 / omega_norm.powi(2)) * (1.0 - omega_norm / 2.0 * cot_half);
            
            v_inv.dot(&self.translation)
        };
        
        Se3 {
            v,
            omega: omega_algebra.omega,
        }
    }
    
    fn multiply(&self, other: &Self) -> Self {
        SE3 {
            rotation: self.rotation.multiply(&other.rotation),
            translation: &self.translation + &self.rotation.matrix.dot(&other.translation),
        }
    }
    
    fn inverse(&self) -> Self {
        let rotation_inv = self.rotation.inverse();
        SE3 {
            rotation: rotation_inv.clone(),
            translation: -rotation_inv.matrix.dot(&self.translation),
        }
    }
    
    fn identity() -> Self {
        SE3 {
            rotation: SO3::identity(),
            translation: Array1::zeros(3),
        }
    }
}

/// Integrator specifically for SE(3)
pub struct SE3Integrator {
    /// Base Lie group integrator
    base: LieGroupIntegrator<SE3>,
}

impl SE3Integrator {
    /// Create a new SE(3) integrator
    pub fn new(dt: f64, method: LieGroupMethod) -> Self {
        Self {
            base: LieGroupIntegrator::new(dt, method),
        }
    }
    
    /// Integrate rigid body motion with forces and torques
    pub fn integrate_rigid_body_6dof(
        &self,
        pose: &SE3,
        velocity: &Se3,
        mass: f64,
        inertia: &Array2<f64>,
        forces: &Array1<f64>,
        torques: &Array1<f64>,
        n_steps: usize,
    ) -> Result<Vec<(SE3, Se3)>> {
        let mut states = vec![(pose.clone(), velocity.clone())];
        let mut current_pose = pose.clone();
        let mut current_velocity = velocity.clone();
        
        for _ in 0..n_steps {
            // Update velocities with Newton-Euler equations
            let linear_accel = forces / mass;
            let angular_accel = self.compute_angular_acceleration(
                &current_velocity.omega,
                inertia,
                torques
            )?;
            
            // Update velocity
            current_velocity.v = &current_velocity.v + &linear_accel * self.base.dt;
            current_velocity.omega = &current_velocity.omega + &angular_accel * self.base.dt;
            
            // Update pose
            current_pose = self.base.step(&current_pose, |_| current_velocity.clone())?;
            
            states.push((current_pose.clone(), current_velocity.clone()));
        }
        
        Ok(states)
    }
    
    fn compute_angular_acceleration(
        &self,
        omega: &Array1<f64>,
        inertia: &Array2<f64>,
        torque: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        // τ = Iα + ω × (Iω)
        // α = I^(-1)(τ - ω × (Iω))
        let I_omega = inertia.dot(omega);
        let omega_cross_I_omega = Array1::from_vec(vec![
            omega[1] * I_omega[2] - omega[2] * I_omega[1],
            omega[2] * I_omega[0] - omega[0] * I_omega[2],
            omega[0] * I_omega[1] - omega[1] * I_omega[0],
        ]);
        
        let inertia_inv = SO3Integrator::new(self.base.dt, LieGroupMethod::LieEuler)
            .invert_inertia(inertia)?;
        
        Ok(inertia_inv.dot(&(torque - &omega_cross_I_omega)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_So3_exp_log() {
        let omega = Array1::from_vec(vec![0.1, 0.2, 0.3]);
        let algebra = So3 { omega };
        
        let group = SO3::exp(&algebra);
        let algebra_recovered = group.log();
        
        for i in 0..3 {
            assert_relative_eq!(algebra.omega[i], algebra_recovered.omega[i], epsilon = 1e-10);
        }
    }
    
    #[test]
    fn test_rigid_body_energy_conservation() {
        let dt = 0.01;
        let integrator = SO3Integrator::new(dt, LieGroupMethod::RKMK4);
        
        // Initial conditions
        let orientation = SO3::identity();
        let angular_velocity = Array1::from_vec(vec![0.1, 0.5, 0.3]);
        let inertia = Array2::from_shape_vec((3, 3), vec![
            2.0, 0.0, 0.0,
            0.0, 3.0, 0.0,
            0.0, 0.0, 4.0,
        ]).unwrap();
        let external_torque = Array1::zeros(3);
        
        // Initial energy
        let initial_energy = 0.5 * angular_velocity.dot(&inertia.dot(&angular_velocity));
        
        // Integrate
        let states = integrator.integrate_rigid_body(
            &orientation,
            &angular_velocity,
            &inertia,
            &external_torque,
            100
        ).unwrap();
        
        // Check energy conservation
        let (_, final_omega) = &states.last().unwrap();
        let final_energy = 0.5 * final_omega.dot(&inertia.dot(final_omega));
        
        assert_relative_eq!(initial_energy, final_energy, epsilon = 1e-8);
    }
}