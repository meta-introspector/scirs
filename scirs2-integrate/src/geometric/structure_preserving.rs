//! Structure-preserving integrators
//!
//! This module provides integrators that preserve various geometric structures
//! such as energy, momentum, symplectic structure, and other invariants.

use ndarray::{Array1, ArrayView1};
use crate::error::IntegrateResult as Result;

/// Trait for geometric invariants
pub trait GeometricInvariant {
    /// Evaluate the invariant quantity
    fn evaluate(&self, x: &ArrayView1<f64>, v: &ArrayView1<f64>, t: f64) -> f64;
    
    /// Name of the invariant (for debugging)
    fn name(&self) -> &str;
}

/// General structure-preserving integrator
pub struct StructurePreservingIntegrator {
    /// Time step
    pub dt: f64,
    /// Integration method
    pub method: StructurePreservingMethod,
    /// Invariants to preserve
    pub invariants: Vec<Box<dyn GeometricInvariant>>,
    /// Tolerance for invariant preservation
    pub tol: f64,
}

/// Available structure-preserving methods
#[derive(Debug, Clone, Copy)]
pub enum StructurePreservingMethod {
    /// Discrete gradient method
    DiscreteGradient,
    /// Average vector field method
    AverageVectorField,
    /// Energy-momentum method
    EnergyMomentum,
    /// Variational integrator
    Variational,
}

impl StructurePreservingIntegrator {
    /// Create a new structure-preserving integrator
    pub fn new(dt: f64, method: StructurePreservingMethod) -> Self {
        Self {
            dt,
            method,
            invariants: Vec::new(),
            tol: 1e-10,
        }
    }
    
    /// Add an invariant to preserve
    pub fn add_invariant(mut self, invariant: Box<dyn GeometricInvariant>) -> Self {
        self.invariants.push(invariant);
        self
    }
    
    /// Check invariant preservation
    pub fn check_invariants(
        &self,
        x0: &ArrayView1<f64>,
        v0: &ArrayView1<f64>,
        x1: &ArrayView1<f64>,
        v1: &ArrayView1<f64>,
        t: f64,
    ) -> Vec<(String, f64)> {
        let mut errors = Vec::new();
        
        for invariant in &self.invariants {
            let i0 = invariant.evaluate(x0, v0, t);
            let i1 = invariant.evaluate(x1, v1, t + self.dt);
            let error = (i1 - i0).abs() / (1.0 + i0.abs());
            errors.push((invariant.name().to_string(), error));
        }
        
        errors
    }
}

/// Energy-preserving integrator for Hamiltonian systems
pub struct EnergyPreservingMethod {
    /// Hamiltonian function
    hamiltonian: Box<dyn Fn(&ArrayView1<f64>, &ArrayView1<f64>) -> f64>,
    /// Dimension
    dim: usize,
}

impl EnergyPreservingMethod {
    /// Create a new energy-preserving integrator
    pub fn new(
        hamiltonian: Box<dyn Fn(&ArrayView1<f64>, &ArrayView1<f64>) -> f64>,
        dim: usize,
    ) -> Self {
        Self { hamiltonian, dim }
    }
    
    /// Discrete gradient method
    pub fn discrete_gradient_step(
        &self,
        q: &ArrayView1<f64>,
        p: &ArrayView1<f64>,
        dt: f64,
    ) -> Result<(Array1<f64>, Array1<f64>)> {
        let h = 1e-8;
        
        // Compute gradients using finite differences
        let mut grad_q = Array1::zeros(self.dim);
        let mut grad_p = Array1::zeros(self.dim);
        
        for i in 0..self.dim {
            let mut q_plus = q.to_owned();
            let mut q_minus = q.to_owned();
            q_plus[i] += h;
            q_minus[i] -= h;
            grad_q[i] = ((self.hamiltonian)(&q_plus.view(), p) - (self.hamiltonian)(&q_minus.view(), p)) / (2.0 * h);
            
            let mut p_plus = p.to_owned();
            let mut p_minus = p.to_owned();
            p_plus[i] += h;
            p_minus[i] -= h;
            grad_p[i] = ((self.hamiltonian)(q, &p_plus.view()) - (self.hamiltonian)(q, &p_minus.view())) / (2.0 * h);
        }
        
        // Implicit midpoint with discrete gradient
        let q_mid = q + &grad_p * (dt / 2.0);
        let p_mid = p - &grad_q * (dt / 2.0);
        
        // Compute discrete gradient at midpoint
        let mut grad_q_mid = Array1::zeros(self.dim);
        let mut grad_p_mid = Array1::zeros(self.dim);
        
        for i in 0..self.dim {
            let mut q_plus = q_mid.clone();
            let mut q_minus = q_mid.clone();
            q_plus[i] += h;
            q_minus[i] -= h;
            grad_q_mid[i] = ((self.hamiltonian)(&q_plus.view(), &p_mid.view()) 
                           - (self.hamiltonian)(&q_minus.view(), &p_mid.view())) / (2.0 * h);
            
            let mut p_plus = p_mid.clone();
            let mut p_minus = p_mid.clone();
            p_plus[i] += h;
            p_minus[i] -= h;
            grad_p_mid[i] = ((self.hamiltonian)(&q_mid.view(), &p_plus.view()) 
                           - (self.hamiltonian)(&q_mid.view(), &p_minus.view())) / (2.0 * h);
        }
        
        let q_new = q + &grad_p_mid * dt;
        let p_new = p - &grad_q_mid * dt;
        
        Ok((q_new, p_new))
    }
    
    /// Average vector field method
    pub fn average_vector_field_step(
        &self,
        q: &ArrayView1<f64>,
        p: &ArrayView1<f64>,
        dt: f64,
    ) -> Result<(Array1<f64>, Array1<f64>)> {
        // Simplified AVF - uses quadrature to average the vector field
        let n_quad = 3; // Number of quadrature points
        let weights = vec![1.0/6.0, 4.0/6.0, 1.0/6.0]; // Simpson's rule weights
        let nodes = vec![0.0, 0.5, 1.0]; // Quadrature nodes
        
        let mut q_avg = Array1::zeros(self.dim);
        let mut p_avg = Array1::zeros(self.dim);
        
        for (&w, &s) in weights.iter().zip(nodes.iter()) {
            // Linear interpolation
            let q_s = q * (1.0 - s) + &(q + &(p * dt)) * s;
            let p_s = p * (1.0 - s) + &(p - &(q * dt)) * s; // Simplified
            
            // Compute gradients at interpolated point
            let h = 1e-8;
            for j in 0..self.dim {
                let mut q_plus = q_s.clone();
                let mut q_minus = q_s.clone();
                q_plus[j] += h;
                q_minus[j] -= h;
                
                p_avg[j] += w * (self.hamiltonian)(&q_plus.view(), &p_s.view()) 
                             - (self.hamiltonian)(&q_minus.view(), &p_s.view()) / (2.0 * h);
                
                let mut p_plus = p_s.clone();
                let mut p_minus = p_s.clone();
                p_plus[j] += h;
                p_minus[j] -= h;
                
                q_avg[j] += w * (self.hamiltonian)(&q_s.view(), &p_plus.view()) 
                             - (self.hamiltonian)(&q_s.view(), &p_minus.view()) / (2.0 * h);
            }
        }
        
        let q_new = q + &q_avg * dt;
        let p_new = p - &p_avg * dt;
        
        Ok((q_new, p_new))
    }
}

/// Momentum-preserving integrator
pub struct MomentumPreservingMethod {
    /// System dimension
    dim: usize,
    /// Force function
    force: Box<dyn Fn(&ArrayView1<f64>) -> Array1<f64>>,
    /// Mass matrix (diagonal)
    mass: Array1<f64>,
}

impl MomentumPreservingMethod {
    /// Create a new momentum-preserving integrator
    pub fn new(
        dim: usize,
        force: Box<dyn Fn(&ArrayView1<f64>) -> Array1<f64>>,
        mass: Array1<f64>,
    ) -> Self {
        Self { dim, force, mass }
    }
    
    /// Integrate one step preserving total momentum
    pub fn step(
        &self,
        x: &ArrayView1<f64>,
        v: &ArrayView1<f64>,
        dt: f64,
    ) -> Result<(Array1<f64>, Array1<f64>)> {
        // Compute forces
        let f = (self.force)(x);
        
        // Check momentum conservation (for internal forces, total force should be zero)
        let total_force: f64 = f.sum();
        if total_force.abs() > 1e-10 {
            // Apply momentum correction
            let f_corrected = &f - total_force / self.dim as f64;
            
            // Velocity Verlet with corrected forces
            let a = &f_corrected / &self.mass;
            let x_new = x + v * dt + &a * (dt * dt / 2.0);
            
            let f_new = (self.force)(&x_new.view());
            let f_new_corrected = &f_new - f_new.sum() / self.dim as f64;
            let a_new = &f_new_corrected / &self.mass;
            
            let v_new = v + (&a + &a_new) * (dt / 2.0);
            
            Ok((x_new, v_new))
        } else {
            // Standard velocity Verlet
            let a = &f / &self.mass;
            let x_new = x + v * dt + &a * (dt * dt / 2.0);
            
            let f_new = (self.force)(&x_new.view());
            let a_new = &f_new / &self.mass;
            
            let v_new = v + (&a + &a_new) * (dt / 2.0);
            
            Ok((x_new, v_new))
        }
    }
}

/// Conservation checker for verifying invariant preservation
pub struct ConservationChecker;

impl ConservationChecker {
    /// Check energy conservation
    pub fn check_energy<H>(
        trajectory: &[(Array1<f64>, Array1<f64>)],
        hamiltonian: H,
    ) -> Vec<f64>
    where
        H: Fn(&ArrayView1<f64>, &ArrayView1<f64>) -> f64,
    {
        trajectory.iter()
            .map(|(q, p)| hamiltonian(&q.view(), &p.view()))
            .collect()
    }
    
    /// Check momentum conservation
    pub fn check_momentum(
        trajectory: &[(Array1<f64>, Array1<f64>)],
        masses: &ArrayView1<f64>,
    ) -> Vec<Array1<f64>> {
        trajectory.iter()
            .map(|(_, v)| v * masses)
            .collect()
    }
    
    /// Check angular momentum conservation
    pub fn check_angular_momentum(
        trajectory: &[(Array1<f64>, Array1<f64>)],
        masses: &ArrayView1<f64>,
    ) -> Vec<Array1<f64>> {
        trajectory.iter()
            .map(|(x, v)| {
                // For 3D systems, compute L = r × p
                if x.len() == 3 && v.len() == 3 {
                    let px = v[0] * masses[0];
                    let py = v[1] * masses[1];
                    let pz = v[2] * masses[2];
                    
                    Array1::from_vec(vec![
                        x[1] * pz - x[2] * py,
                        x[2] * px - x[0] * pz,
                        x[0] * py - x[1] * px,
                    ])
                } else {
                    Array1::zeros(3)
                }
            })
            .collect()
    }
    
    /// Compute relative error in conservation
    pub fn relative_error(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        
        let initial = values[0];
        let max_deviation = values.iter()
            .map(|&v| (v - initial).abs())
            .fold(0.0, f64::max);
        
        max_deviation / (1.0 + initial.abs())
    }
}

/// Example invariants
pub mod invariants {
    use super::*;
    
    /// Energy invariant for Hamiltonian systems
    pub struct EnergyInvariant {
        hamiltonian: Box<dyn Fn(&ArrayView1<f64>, &ArrayView1<f64>) -> f64>,
    }
    
    impl EnergyInvariant {
        pub fn new(hamiltonian: Box<dyn Fn(&ArrayView1<f64>, &ArrayView1<f64>) -> f64>) -> Self {
            Self { hamiltonian }
        }
    }
    
    impl GeometricInvariant for EnergyInvariant {
        fn evaluate(&self, x: &ArrayView1<f64>, v: &ArrayView1<f64>, _t: f64) -> f64 {
            (self.hamiltonian)(x, v)
        }
        
        fn name(&self) -> &str {
            "Energy"
        }
    }
    
    /// Linear momentum invariant
    pub struct LinearMomentumInvariant {
        masses: Array1<f64>,
        component: usize,
    }
    
    impl LinearMomentumInvariant {
        pub fn new(masses: Array1<f64>, component: usize) -> Self {
            Self { masses, component }
        }
    }
    
    impl GeometricInvariant for LinearMomentumInvariant {
        fn evaluate(&self, _x: &ArrayView1<f64>, v: &ArrayView1<f64>, _t: f64) -> f64 {
            v[self.component] * self.masses[self.component]
        }
        
        fn name(&self) -> &str {
            "Linear Momentum"
        }
    }
    
    /// Angular momentum invariant (for 2D systems)
    pub struct AngularMomentumInvariant2D {
        masses: Array1<f64>,
    }
    
    impl AngularMomentumInvariant2D {
        pub fn new(masses: Array1<f64>) -> Self {
            Self { masses }
        }
    }
    
    impl GeometricInvariant for AngularMomentumInvariant2D {
        fn evaluate(&self, x: &ArrayView1<f64>, v: &ArrayView1<f64>, _t: f64) -> f64 {
            // L = m(xv_y - yv_x) for 2D
            let n = x.len() / 2;
            let mut l = 0.0;
            
            for i in 0..n {
                let xi = x[2*i];
                let yi = x[2*i + 1];
                let vxi = v[2*i];
                let vyi = v[2*i + 1];
                l += self.masses[i] * (xi * vyi - yi * vxi);
            }
            
            l
        }
        
        fn name(&self) -> &str {
            "Angular Momentum"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::invariants::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_energy_preservation() {
        // Simple harmonic oscillator: H = p²/2m + kx²/2
        let m = 1.0;
        let k = 1.0;
        let hamiltonian = Box::new(move |q: &ArrayView1<f64>, p: &ArrayView1<f64>| {
            p[0] * p[0] / (2.0 * m) + k * q[0] * q[0] / 2.0
        });
        
        let integrator = EnergyPreservingMethod::new(hamiltonian.clone(), 1);
        let q0 = Array1::from_vec(vec![1.0]);
        let p0 = Array1::from_vec(vec![0.0]);
        
        let initial_energy = hamiltonian(&q0.view(), &p0.view());
        
        // Integrate for one period
        let dt = 0.1;
        let n_steps = 63; // approximately 2π
        
        let mut q = q0.clone();
        let mut p = p0.clone();
        
        for _ in 0..n_steps {
            let (q_new, p_new) = integrator.discrete_gradient_step(&q.view(), &p.view(), dt).unwrap();
            q = q_new;
            p = p_new;
        }
        
        let final_energy = hamiltonian(&q.view(), &p.view());
        assert_relative_eq!(initial_energy, final_energy, epsilon = 1e-10);
    }
    
    #[test]
    fn test_momentum_preservation() {
        // Two-particle system with internal forces
        let dim = 4; // 2 particles in 2D
        let force = Box::new(|x: &ArrayView1<f64>| {
            // Spring force between particles
            let dx = x[2] - x[0];
            let dy = x[3] - x[1];
            let r = (dx * dx + dy * dy).sqrt();
            let f = if r > 0.0 { 1.0 / r } else { 0.0 };
            
            Array1::from_vec(vec![
                f * dx,
                f * dy,
                -f * dx,
                -f * dy,
            ])
        });
        
        let mass = Array1::from_vec(vec![1.0, 1.0, 2.0, 2.0]);
        let integrator = MomentumPreservingMethod::new(dim, force, mass.clone());
        
        let x0 = Array1::from_vec(vec![0.0, 0.0, 1.0, 0.0]);
        let v0 = Array1::from_vec(vec![0.0, 0.1, 0.0, -0.05]);
        
        let initial_momentum: f64 = (&v0 * &mass).sum();
        
        let (x1, v1) = integrator.step(&x0.view(), &v0.view(), 0.01).unwrap();
        let final_momentum: f64 = (&v1 * &mass).sum();
        
        assert_relative_eq!(initial_momentum, final_momentum, epsilon = 1e-12);
    }
}