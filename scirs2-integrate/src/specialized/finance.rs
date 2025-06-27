//! Financial modeling solvers for stochastic PDEs
//!
//! This module provides specialized solvers for quantitative finance applications,
//! including Black-Scholes, stochastic volatility models, and jump-diffusion processes.

use ndarray::{Array1, Array2};
use scirs2_core::constants::PI;
use crate::error::{IntegrateError, IntegrateResult as Result};
use rand::prelude::*;
use rand_distr::{Normal, StandardNormal};
use std::f64::consts::SQRT_2;

/// Option type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptionType {
    /// Call option (right to buy)
    Call,
    /// Put option (right to sell)
    Put,
}

/// Option style
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptionStyle {
    /// European option (exercise only at maturity)
    European,
    /// American option (exercise any time before maturity)
    American,
    /// Asian option (payoff depends on average price)
    Asian,
    /// Barrier option (activated/deactivated by price level)
    Barrier { barrier: f64, is_up: bool, is_knock_in: bool },
}

/// Financial option specification
#[derive(Debug, Clone)]
pub struct FinancialOption {
    /// Option type (call/put)
    pub option_type: OptionType,
    /// Option style (European/American/etc)
    pub option_style: OptionStyle,
    /// Strike price
    pub strike: f64,
    /// Time to maturity
    pub maturity: f64,
    /// Initial asset price
    pub spot: f64,
    /// Risk-free rate
    pub risk_free_rate: f64,
    /// Dividend yield
    pub dividend_yield: f64,
}

/// Volatility model
pub enum VolatilityModel {
    /// Constant volatility (Black-Scholes)
    Constant(f64),
    /// Heston stochastic volatility model
    Heston {
        /// Initial volatility
        v0: f64,
        /// Long-term variance
        theta: f64,
        /// Mean reversion speed
        kappa: f64,
        /// Volatility of volatility
        sigma: f64,
        /// Correlation between asset and volatility
        rho: f64,
    },
    /// SABR (Stochastic Alpha Beta Rho) model
    SABR {
        /// Initial volatility
        alpha: f64,
        /// CEV exponent
        beta: f64,
        /// Volatility of volatility
        nu: f64,
        /// Correlation
        rho: f64,
    },
    /// Local volatility surface
    LocalVolatility(Box<dyn Fn(f64, f64) -> f64 + Send + Sync>),
}

/// Jump process specification
#[derive(Debug, Clone)]
pub struct JumpProcess {
    /// Jump intensity (average number of jumps per year)
    pub lambda: f64,
    /// Mean jump size
    pub mu_jump: f64,
    /// Jump size standard deviation
    pub sigma_jump: f64,
}

/// Solver for stochastic PDEs in finance
pub struct StochasticPDESolver {
    /// Grid points in asset dimension
    pub n_asset: usize,
    /// Grid points in time dimension
    pub n_time: usize,
    /// Grid points in volatility dimension (for stochastic vol)
    pub n_vol: Option<usize>,
    /// Volatility model
    pub volatility_model: VolatilityModel,
    /// Jump process (optional)
    pub jump_process: Option<JumpProcess>,
    /// Solver method
    pub method: FinanceMethod,
}

/// Available methods for solving financial PDEs
#[derive(Debug, Clone, Copy)]
pub enum FinanceMethod {
    /// Finite difference method
    FiniteDifference,
    /// Monte Carlo simulation
    MonteCarlo { n_paths: usize, antithetic: bool },
    /// Fourier transform methods
    FourierTransform,
    /// Tree methods (binomial/trinomial)
    Tree { n_steps: usize },
}

impl StochasticPDESolver {
    /// Create a new stochastic PDE solver
    pub fn new(
        n_asset: usize,
        n_time: usize,
        volatility_model: VolatilityModel,
        method: FinanceMethod,
    ) -> Self {
        let n_vol = match &volatility_model {
            VolatilityModel::Heston { .. } | VolatilityModel::SABR { .. } => Some(50),
            _ => None,
        };
        
        Self {
            n_asset,
            n_time,
            n_vol,
            volatility_model,
            jump_process: None,
            method,
        }
    }
    
    /// Add jump process
    pub fn with_jumps(mut self, jump_process: JumpProcess) -> Self {
        self.jump_process = Some(jump_process);
        self
    }
    
    /// Price option using specified method
    pub fn price_option(&self, option: &FinancialOption) -> Result<f64> {
        match self.method {
            FinanceMethod::FiniteDifference => self.price_finite_difference(option),
            FinanceMethod::MonteCarlo { n_paths, antithetic } => {
                self.price_monte_carlo(option, n_paths, antithetic)
            }
            FinanceMethod::FourierTransform => self.price_fourier_transform(option),
            FinanceMethod::Tree { n_steps } => self.price_tree(option, n_steps),
        }
    }
    
    /// Finite difference method for option pricing
    fn price_finite_difference(&self, option: &FinancialOption) -> Result<f64> {
        match &self.volatility_model {
            VolatilityModel::Constant(sigma) => {
                self.black_scholes_finite_difference(option, *sigma)
            }
            VolatilityModel::Heston { v0, theta, kappa, sigma, rho } => {
                self.heston_finite_difference(option, *v0, *theta, *kappa, *sigma, *rho)
            }
            _ => Err(IntegrateError::NotImplementedError(
                "Finite difference not implemented for this volatility model".into()
            ))
        }
    }
    
    /// Black-Scholes finite difference solver
    fn black_scholes_finite_difference(
        &self,
        option: &FinancialOption,
        sigma: f64,
    ) -> Result<f64> {
        let dt = option.maturity / (self.n_time - 1) as f64;
        let s_max = option.spot * 3.0;
        let ds = s_max / (self.n_asset - 1) as f64;
        
        // Initialize grid
        let mut v = Array2::zeros((self.n_time, self.n_asset));
        
        // Terminal condition
        for i in 0..self.n_asset {
            let s = i as f64 * ds;
            v[[self.n_time - 1, i]] = self.payoff(option, s);
        }
        
        // Boundary conditions
        match option.option_type {
            OptionType::Call => {
                // At S = 0, call value is 0
                // At S = S_max, call value is S - K*exp(-r*t)
                for t_idx in 0..self.n_time {
                    let t = (self.n_time - 1 - t_idx) as f64 * dt;
                    v[[t_idx, 0]] = 0.0;
                    v[[t_idx, self.n_asset - 1]] = s_max - option.strike * 
                        (-option.risk_free_rate * t).exp();
                }
            }
            OptionType::Put => {
                // At S = 0, put value is K*exp(-r*t)
                // At S = S_max, put value is 0
                for t_idx in 0..self.n_time {
                    let t = (self.n_time - 1 - t_idx) as f64 * dt;
                    v[[t_idx, 0]] = option.strike * (-option.risk_free_rate * t).exp();
                    v[[t_idx, self.n_asset - 1]] = 0.0;
                }
            }
        }
        
        // Backward induction
        for t_idx in (0..self.n_time - 1).rev() {
            for i in 1..self.n_asset - 1 {
                let s = i as f64 * ds;
                
                // Finite difference coefficients
                let a = 0.5 * dt * (sigma * sigma * (i as f64) * (i as f64) - option.risk_free_rate * i as f64);
                let b = 1.0 - dt * (sigma * sigma * (i as f64) * (i as f64) + option.risk_free_rate);
                let c = 0.5 * dt * (sigma * sigma * (i as f64) * (i as f64) + option.risk_free_rate * i as f64);
                
                // Explicit scheme (can be made implicit for better stability)
                let v_new = a * v[[t_idx + 1, i - 1]] + b * v[[t_idx + 1, i]] + 
                           c * v[[t_idx + 1, i + 1]];
                
                // For American options, apply early exercise condition
                if option.option_style == OptionStyle::American {
                    v[[t_idx, i]] = v_new.max(self.payoff(option, s));
                } else {
                    v[[t_idx, i]] = v_new;
                }
            }
        }
        
        // Interpolate to get price at spot
        let spot_idx = (option.spot / ds) as usize;
        let alpha = option.spot / ds - spot_idx as f64;
        
        if spot_idx < self.n_asset - 1 {
            Ok(v[[0, spot_idx]] * (1.0 - alpha) + v[[0, spot_idx + 1]] * alpha)
        } else {
            Ok(v[[0, self.n_asset - 1]])
        }
    }
    
    /// Heston model finite difference solver (ADI method)
    fn heston_finite_difference(
        &self,
        option: &FinancialOption,
        v0: f64,
        theta: f64,
        kappa: f64,
        sigma: f64,
        _rho: f64,
    ) -> Result<f64> {
        let n_vol = self.n_vol.unwrap_or(50);
        let dt = option.maturity / (self.n_time - 1) as f64;
        
        // Asset grid
        let s_max = option.spot * 3.0;
        let ds = s_max / (self.n_asset - 1) as f64;
        
        // Variance grid
        let v_max = 1.0; // Maximum variance
        let dv = v_max / (n_vol - 1) as f64;
        
        // Initialize solution grid
        let mut u = Array2::zeros((self.n_asset * n_vol, self.n_time));
        
        // Terminal condition
        for i in 0..self.n_asset {
            for j in 0..n_vol {
                let s = i as f64 * ds;
                let idx = i * n_vol + j;
                u[[idx, self.n_time - 1]] = self.payoff(option, s);
            }
        }
        
        // ADI (Alternating Direction Implicit) scheme
        for t_idx in (0..self.n_time - 1).rev() {
            // Step 1: Implicit in S direction, explicit in V direction
            let mut intermediate = Array1::zeros(self.n_asset * n_vol);
            
            for j in 1..n_vol - 1 {
                // Tridiagonal system for each variance level
                let mut a = vec![0.0; self.n_asset];
                let mut b = vec![0.0; self.n_asset];
                let mut c = vec![0.0; self.n_asset];
                let mut d = vec![0.0; self.n_asset];
                
                for i in 1..self.n_asset - 1 {
                    let s = i as f64 * ds;
                    let v = j as f64 * dv;
                    let idx = i * n_vol + j;
                    
                    // Coefficients for S-direction
                    let alpha_s = 0.5 * v * s * s / (ds * ds);
                    let beta_s = option.risk_free_rate * s / (2.0 * ds);
                    
                    a[i] = -0.5 * dt * (alpha_s - beta_s);
                    b[i] = 1.0 + dt * (alpha_s + 0.5 * option.risk_free_rate);
                    c[i] = -0.5 * dt * (alpha_s + beta_s);
                    
                    // RHS includes explicit V-direction terms
                    let alpha_v = 0.5 * sigma * sigma * v / (dv * dv);
                    let beta_v = kappa * (theta - v) / (2.0 * dv);
                    
                    d[i] = u[[idx, t_idx + 1]] 
                         + 0.5 * dt * alpha_v * (u[[idx - 1, t_idx + 1]] - 2.0 * u[[idx, t_idx + 1]] + u[[idx + 1, t_idx + 1]])
                         + 0.5 * dt * beta_v * (u[[idx + 1, t_idx + 1]] - u[[idx - 1, t_idx + 1]]);
                }
                
                // Solve tridiagonal system
                let solution = self.solve_tridiagonal_system(&a, &b, &c, &d)?;
                for i in 0..self.n_asset {
                    intermediate[i * n_vol + j] = solution[i];
                }
            }
            
            // Step 2: Implicit in V direction, using intermediate values
            for i in 1..self.n_asset - 1 {
                // Tridiagonal system for each asset level
                let mut a = vec![0.0; n_vol];
                let mut b = vec![0.0; n_vol];
                let mut c = vec![0.0; n_vol];
                let mut d = vec![0.0; n_vol];
                
                for j in 1..n_vol - 1 {
                    let v = j as f64 * dv;
                    let idx = i * n_vol + j;
                    
                    // Coefficients for V-direction
                    let alpha_v = 0.5 * sigma * sigma * v / (dv * dv);
                    let beta_v = kappa * (theta - v) / (2.0 * dv);
                    
                    a[j] = -0.5 * dt * (alpha_v - beta_v);
                    b[j] = 1.0 + dt * alpha_v;
                    c[j] = -0.5 * dt * (alpha_v + beta_v);
                    
                    d[j] = intermediate[idx];
                }
                
                // Solve tridiagonal system
                let solution = self.solve_tridiagonal_system(&a, &b, &c, &d)?;
                for j in 0..n_vol {
                    u[[i * n_vol + j, t_idx]] = solution[j];
                }
            }
            
            // Apply boundary conditions
            self.apply_heston_boundary_conditions(&mut u, t_idx, option, n_vol)?;
        }
        
        // Interpolate to get price at (S0, v0)
        let s_idx = (option.spot / ds).min(self.n_asset as f64 - 1.0) as usize;
        let v_idx = (v0 / dv).min(n_vol as f64 - 1.0) as usize;
        let idx = s_idx * n_vol + v_idx;
        
        Ok(u[[idx, 0]])
    }
    
    /// Monte Carlo pricing
    fn price_monte_carlo(
        &self,
        option: &FinancialOption,
        n_paths: usize,
        antithetic: bool,
    ) -> Result<f64> {
        let mut rng = rand::rng();
        let _normal = StandardNormal;
        
        let dt = option.maturity / 100.0; // 100 time steps
        let n_steps = (option.maturity / dt) as usize;
        
        let mut payoffs = Vec::with_capacity(n_paths);
        
        for _ in 0..n_paths {
            let paths = match &self.volatility_model {
                VolatilityModel::Constant(sigma) => {
                    self.simulate_gbm_path(option, *sigma, n_steps, dt, &mut rng)?
                }
                VolatilityModel::Heston { v0, theta, kappa, sigma, rho } => {
                    self.simulate_heston_path(
                        option, *v0, *theta, *kappa, *sigma, *rho, 
                        n_steps, dt, &mut rng
                    )?
                }
                _ => return Err(IntegrateError::NotImplementedError(
                    "Monte Carlo not implemented for this volatility model".into()
                )),
            };
            
            // Calculate payoff based on option style
            let payoff = match option.option_style {
                OptionStyle::European => {
                    self.payoff(option, paths.0[paths.0.len() - 1])
                }
                OptionStyle::American => {
                    // American option requires optimal exercise
                    // Using Longstaff-Schwartz method would be more accurate
                    let mut max_payoff: f64 = 0.0;
                    for (i, &s) in paths.0.iter().enumerate() {
                        let t = i as f64 * dt;
                        let discount = (-option.risk_free_rate * (option.maturity - t)).exp();
                        max_payoff = max_payoff.max(self.payoff(option, s) * discount);
                    }
                    max_payoff
                }
                OptionStyle::Asian => {
                    let avg_price = paths.0.iter().sum::<f64>() / paths.0.len() as f64;
                    match option.option_type {
                        OptionType::Call => (avg_price - option.strike).max(0.0),
                        OptionType::Put => (option.strike - avg_price).max(0.0),
                    }
                }
                _ => self.payoff(option, paths.0[paths.0.len() - 1]),
            };
            
            payoffs.push(payoff);
            
            // Antithetic variates
            if antithetic {
                let anti_paths = match &self.volatility_model {
                    VolatilityModel::Constant(sigma) => {
                        self.simulate_gbm_path_antithetic(option, *sigma, n_steps, dt, &paths)?
                    }
                    _ => paths.clone(), // Simplified for now
                };
                
                let anti_payoff = match option.option_style {
                    OptionStyle::European => {
                        self.payoff(option, anti_paths.0[anti_paths.0.len() - 1])
                    }
                    _ => self.payoff(option, anti_paths.0[anti_paths.0.len() - 1]),
                };
                
                payoffs.push(anti_payoff);
            }
        }
        
        // Discount and average
        let avg_payoff = payoffs.iter().sum::<f64>() / payoffs.len() as f64;
        Ok(avg_payoff * (-option.risk_free_rate * option.maturity).exp())
    }
    
    /// Simulate GBM path
    fn simulate_gbm_path(
        &self,
        option: &FinancialOption,
        sigma: f64,
        n_steps: usize,
        dt: f64,
        rng: &mut ThreadRng,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let mut s_path = vec![option.spot];
        let mut v_path = vec![sigma * sigma]; // Constant variance
        
        let drift = option.risk_free_rate - option.dividend_yield - 0.5 * sigma * sigma;
        
        for _ in 0..n_steps {
            let z: f64 = rng.sample(StandardNormal);
            let s_new = s_path.last().unwrap() * 
                (drift * dt + sigma * dt.sqrt() * z).exp();
            
            // Add jump if specified
            if let Some(ref jump) = self.jump_process {
                if rng.random::<f64>() < jump.lambda * dt {
                    let jump_size = 1.0 + rng.sample(Normal::new(jump.mu_jump, jump.sigma_jump).unwrap());
                    s_path.push(s_new * jump_size);
                } else {
                    s_path.push(s_new);
                }
            } else {
                s_path.push(s_new);
            }
            
            v_path.push(sigma * sigma);
        }
        
        Ok((s_path, v_path))
    }
    
    /// Simulate GBM path with antithetic variates
    fn simulate_gbm_path_antithetic(
        &self,
        option: &FinancialOption,
        sigma: f64,
        n_steps: usize,
        dt: f64,
        original_path: &(Vec<f64>, Vec<f64>),
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let mut s_path = vec![option.spot];
        let mut v_path = vec![sigma * sigma];
        
        let drift = option.risk_free_rate - option.dividend_yield - 0.5 * sigma * sigma;
        
        // Use negative of original random numbers
        for i in 0..n_steps {
            let original_return = (original_path.0[i + 1] / original_path.0[i]).ln();
            let z = -(original_return - drift * dt) / (sigma * dt.sqrt());
            
            let s_new = s_path.last().unwrap() * 
                (drift * dt + sigma * dt.sqrt() * z).exp();
            
            s_path.push(s_new);
            v_path.push(sigma * sigma);
        }
        
        Ok((s_path, v_path))
    }
    
    /// Simulate Heston model path
    fn simulate_heston_path(
        &self,
        option: &FinancialOption,
        v0: f64,
        theta: f64,
        kappa: f64,
        sigma: f64,
        rho: f64,
        n_steps: usize,
        dt: f64,
        rng: &mut ThreadRng,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let mut s_path = vec![option.spot];
        let mut v_path = vec![v0];
        
        for _ in 0..n_steps {
            let z1: f64 = rng.sample(StandardNormal);
            let z2: f64 = rng.sample(StandardNormal);
            
            // Correlated Brownian motions
            let w1 = z1;
            let w2 = rho * z1 + (1.0 - rho * rho).sqrt() * z2;
            
            // Variance process (using full truncation scheme)
            let v_curr = v_path.last().unwrap().max(0.0);
            let v_new = v_curr + kappa * (theta - v_curr) * dt 
                      + sigma * v_curr.sqrt() * dt.sqrt() * w2;
            let v_new = v_new.max(0.0); // Ensure non-negative
            
            // Asset process
            let s_curr = s_path.last().unwrap();
            let drift = option.risk_free_rate - option.dividend_yield - 0.5 * v_curr;
            let s_new = s_curr * (drift * dt + v_curr.sqrt() * dt.sqrt() * w1).exp();
            
            s_path.push(s_new);
            v_path.push(v_new);
        }
        
        Ok((s_path, v_path))
    }
    
    /// Fourier transform pricing (for European options)
    fn price_fourier_transform(&self, option: &FinancialOption) -> Result<f64> {
        match &self.volatility_model {
            VolatilityModel::Constant(sigma) => {
                // Use Black-Scholes closed form
                self.black_scholes_formula(option, *sigma)
            }
            VolatilityModel::Heston { v0, theta, kappa, sigma, rho } => {
                // Heston characteristic function method
                self.heston_fourier(option, *v0, *theta, *kappa, *sigma, *rho)
            }
            _ => Err(IntegrateError::NotImplementedError(
                "Fourier transform not implemented for this model".into()
            ))
        }
    }
    
    /// Black-Scholes closed-form formula
    fn black_scholes_formula(&self, option: &FinancialOption, sigma: f64) -> Result<f64> {
        let s = option.spot;
        let k = option.strike;
        let r = option.risk_free_rate;
        let q = option.dividend_yield;
        let t = option.maturity;
        
        let d1 = ((s / k).ln() + (r - q + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());
        let d2 = d1 - sigma * t.sqrt();
        
        let price = match option.option_type {
            OptionType::Call => {
                s * (-q * t).exp() * self.normal_cdf(d1) 
                - k * (-r * t).exp() * self.normal_cdf(d2)
            }
            OptionType::Put => {
                k * (-r * t).exp() * self.normal_cdf(-d2) 
                - s * (-q * t).exp() * self.normal_cdf(-d1)
            }
        };
        
        Ok(price)
    }
    
    /// Heston model Fourier pricing
    fn heston_fourier(
        &self,
        option: &FinancialOption,
        v0: f64,
        theta: f64,
        kappa: f64,
        sigma: f64,
        rho: f64,
    ) -> Result<f64> {
        // Simplified implementation using characteristic function
        // Full implementation would use FFT for efficiency
        
        let s = option.spot;
        let k = option.strike;
        let r = option.risk_free_rate;
        let t = option.maturity;
        
        // For simplicity, use numerical integration
        // In practice, would use FFT
        let n_points = 1000;
        let du = 0.01;
        let mut integral = 0.0;
        
        for i in 1..n_points {
            let u = i as f64 * du;
            let phi = self.heston_characteristic_function(
                u - 0.5, t, v0, theta, kappa, sigma, rho, r
            );
            
            let integrand = (phi * (-u * (s / k).ln()).exp()).re / (u * u + 0.25);
            integral += integrand * du;
        }
        
        let price = s - k.sqrt() * s.sqrt() * integral / PI;
        
        match option.option_type {
            OptionType::Call => Ok(price),
            OptionType::Put => Ok(price + k * (-r * t).exp() - s), // Put-call parity
        }
    }
    
    /// Heston characteristic function
    fn heston_characteristic_function(
        &self,
        u: f64,
        t: f64,
        v0: f64,
        theta: f64,
        kappa: f64,
        sigma: f64,
        rho: f64,
        r: f64,
    ) -> num_complex::Complex64 {
        use num_complex::Complex64;
        
        let i = Complex64::new(0.0, 1.0);
        let d = ((rho * sigma * u * i - kappa).powi(2) 
               + sigma * sigma * (u * i + u * u)).sqrt();
        
        let g = (kappa - rho * sigma * u * i - d) / (kappa - rho * sigma * u * i + d);
        
        let c = kappa * theta / (sigma * sigma) * (
            (kappa - rho * sigma * u * i - d) * t 
            - 2.0 * ((1.0 - g * (-d * t).exp()) / (1.0 - g)).ln()
        );
        
        let d_term = (kappa - rho * sigma * u * i - d) / (sigma * sigma) 
                   * (1.0 - (-d * t).exp()) / (1.0 - g * (-d * t).exp());
        
        (i * u * r * t).exp() * (c + d_term * v0).exp()
    }
    
    /// Tree method pricing
    fn price_tree(&self, option: &FinancialOption, n_steps: usize) -> Result<f64> {
        match &self.volatility_model {
            VolatilityModel::Constant(sigma) => {
                self.binomial_tree(option, *sigma, n_steps)
            }
            _ => Err(IntegrateError::NotImplementedError(
                "Tree methods only implemented for constant volatility".into()
            ))
        }
    }
    
    /// Binomial tree pricing
    fn binomial_tree(&self, option: &FinancialOption, sigma: f64, n_steps: usize) -> Result<f64> {
        let dt = option.maturity / n_steps as f64;
        let u = (sigma * dt.sqrt()).exp();
        let d = 1.0 / u;
        let p = ((option.risk_free_rate - option.dividend_yield) * dt - d) / (u - d);
        
        // Build final payoffs
        let mut prices = vec![0.0; n_steps + 1];
        for i in 0..=n_steps {
            let s_final = option.spot * u.powi(i as i32) * d.powi((n_steps - i) as i32);
            prices[i] = self.payoff(option, s_final);
        }
        
        // Backward induction
        let discount = (-option.risk_free_rate * dt).exp();
        for step in (0..n_steps).rev() {
            for i in 0..=step {
                let continuation_value = discount * (p * prices[i + 1] + (1.0 - p) * prices[i]);
                
                if option.option_style == OptionStyle::American {
                    let s = option.spot * u.powi(i as i32) * d.powi((step - i) as i32);
                    prices[i] = continuation_value.max(self.payoff(option, s));
                } else {
                    prices[i] = continuation_value;
                }
            }
        }
        
        Ok(prices[0])
    }
    
    /// Calculate option payoff
    fn payoff(&self, option: &FinancialOption, s: f64) -> f64 {
        match option.option_type {
            OptionType::Call => (s - option.strike).max(0.0),
            OptionType::Put => (option.strike - s).max(0.0),
        }
    }
    
    /// Normal CDF
    fn normal_cdf(&self, x: f64) -> f64 {
        0.5 * (1.0 + self.erf(x / SQRT_2))
    }
    
    /// Error function approximation
    fn erf(&self, x: f64) -> f64 {
        // Abramowitz and Stegun approximation
        let a1 =  0.254829592;
        let a2 = -0.284496736;
        let a3 =  1.421413741;
        let a4 = -1.453152027;
        let a5 =  1.061405429;
        let p  =  0.3275911;
        
        let sign = if x >= 0.0 { 1.0 } else { -1.0 };
        let x = x.abs();
        
        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
        
        sign * y
    }
    
    /// Solve tridiagonal system
    fn solve_tridiagonal_system(
        &self,
        a: &[f64],
        b: &[f64],
        c: &[f64],
        d: &[f64],
    ) -> Result<Vec<f64>> {
        let n = b.len();
        let mut c_star = vec![0.0; n];
        let mut d_star = vec![0.0; n];
        let mut x = vec![0.0; n];
        
        // Forward sweep
        c_star[0] = c[0] / b[0];
        d_star[0] = d[0] / b[0];
        
        for i in 1..n-1 {
            let m = b[i] - a[i] * c_star[i-1];
            c_star[i] = c[i] / m;
            d_star[i] = (d[i] - a[i] * d_star[i-1]) / m;
        }
        
        let m = b[n-1] - a[n-1] * c_star[n-2];
        d_star[n-1] = (d[n-1] - a[n-1] * d_star[n-2]) / m;
        
        // Back substitution
        x[n-1] = d_star[n-1];
        for i in (0..n-1).rev() {
            x[i] = d_star[i] - c_star[i] * x[i+1];
        }
        
        Ok(x)
    }
    
    /// Apply boundary conditions for Heston PDE
    fn apply_heston_boundary_conditions(
        &self,
        _u: &mut Array2<f64>,
        _t_idx: usize,
        _option: &FinancialOption,
        _n_vol: usize,
    ) -> Result<()> {
        // Boundary conditions at S = 0 and S = S_max
        // At v = 0 (deterministic case)
        // At v = v_max
        
        // Implementation depends on specific boundary conditions
        // This is a simplified version
        
        Ok(())
    }
    
    /// Calculate Greeks (sensitivities)
    pub fn calculate_greeks(&self, option: &FinancialOption) -> Result<Greeks> {
        let base_price = self.price_option(option)?;
        let h = 0.01; // Finite difference step
        
        // Delta: ∂V/∂S
        let mut option_up = option.clone();
        option_up.spot *= 1.0 + h;
        let price_up = self.price_option(&option_up)?;
        
        let mut option_down = option.clone();
        option_down.spot *= 1.0 - h;
        let price_down = self.price_option(&option_down)?;
        
        let delta = (price_up - price_down) / (2.0 * h * option.spot);
        
        // Gamma: ∂²V/∂S²
        let gamma = (price_up - 2.0 * base_price + price_down) / (h * option.spot).powi(2);
        
        // Theta: ∂V/∂t (negative of time derivative)
        let mut option_later = option.clone();
        option_later.maturity *= 1.0 - h;
        let price_later = self.price_option(&option_later)?;
        let theta = -(base_price - price_later) / (h * option.maturity);
        
        // Rho: ∂V/∂r
        let mut option_high_rate = option.clone();
        option_high_rate.risk_free_rate += h;
        let price_high_rate = self.price_option(&option_high_rate)?;
        let rho = (price_high_rate - base_price) / h;
        
        // Vega: ∂V/∂σ (only for constant volatility)
        let vega = match &self.volatility_model {
            VolatilityModel::Constant(sigma) => {
                let sigma_up = sigma + h;
                let solver_up = StochasticPDESolver::new(
                    self.n_asset,
                    self.n_time,
                    VolatilityModel::Constant(sigma_up),
                    self.method,
                );
                let price_vol_up = solver_up.price_option(option)?;
                (price_vol_up - base_price) / h
            }
            _ => 0.0, // More complex for stochastic vol models
        };
        
        Ok(Greeks {
            delta,
            gamma,
            theta,
            rho,
            vega,
        })
    }
}

/// Option Greeks (sensitivities)
#[derive(Debug, Clone)]
pub struct Greeks {
    /// Delta: ∂V/∂S
    pub delta: f64,
    /// Gamma: ∂²V/∂S²
    pub gamma: f64,
    /// Theta: ∂V/∂t
    pub theta: f64,
    /// Rho: ∂V/∂r
    pub rho: f64,
    /// Vega: ∂V/∂σ
    pub vega: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_black_scholes_call_put_parity() {
        let call = FinancialOption {
            option_type: OptionType::Call,
            option_style: OptionStyle::European,
            strike: 100.0,
            maturity: 1.0,
            spot: 100.0,
            risk_free_rate: 0.05,
            dividend_yield: 0.0,
        };
        
        let put = FinancialOption {
            option_type: OptionType::Put,
            ..call
        };
        
        let solver = StochasticPDESolver::new(
            100,
            50,
            VolatilityModel::Constant(0.2),
            FinanceMethod::FiniteDifference,
        );
        
        let call_price = solver.price_option(&call).unwrap();
        let put_price = solver.price_option(&put).unwrap();
        
        // Put-call parity: C - P = S - K*exp(-r*T)
        let parity = call_price - put_price;
        let theoretical = call.spot - call.strike * (-call.risk_free_rate * call.maturity).exp();
        
        assert_relative_eq!(parity, theoretical, epsilon = 0.01);
    }
    
    #[test]
    fn test_option_greeks() {
        let option = FinancialOption {
            option_type: OptionType::Call,
            option_style: OptionStyle::European,
            strike: 100.0,
            maturity: 1.0,
            spot: 100.0,
            risk_free_rate: 0.05,
            dividend_yield: 0.0,
        };
        
        let solver = StochasticPDESolver::new(
            100,
            50,
            VolatilityModel::Constant(0.2),
            FinanceMethod::FiniteDifference,
        );
        
        let greeks = solver.calculate_greeks(&option).unwrap();
        
        // Delta should be around 0.5 for ATM option
        assert!(greeks.delta > 0.4 && greeks.delta < 0.7);
        
        // Gamma should be positive
        assert!(greeks.gamma > 0.0);
        
        // Theta should be negative for long option
        assert!(greeks.theta < 0.0);
        
        // Rho should be positive for call
        assert!(greeks.rho > 0.0);
        
        // Vega should be positive
        assert!(greeks.vega > 0.0);
    }
}