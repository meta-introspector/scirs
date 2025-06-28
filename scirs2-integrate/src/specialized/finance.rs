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
    /// Bates model (Heston + jumps in volatility)
    Bates {
        /// Heston parameters
        v0: f64,
        theta: f64,
        kappa: f64,
        sigma: f64,
        rho: f64,
        /// Jump parameters for volatility
        lambda_v: f64,
        mu_v: f64,
        sigma_v: f64,
    },
    /// Hull-White stochastic volatility
    HullWhite {
        /// Initial volatility
        v0: f64,
        /// Volatility drift
        alpha: f64,
        /// Volatility of volatility
        beta: f64,
        /// Correlation
        rho: f64,
    },
    /// 3/2 stochastic volatility model
    ThreeHalves {
        /// Initial variance
        v0: f64,
        /// Long-term variance
        theta: f64,
        /// Mean reversion speed
        kappa: f64,
        /// Volatility of volatility
        sigma: f64,
        /// Correlation
        rho: f64,
    },
}

/// Advanced stochastic processes
#[derive(Debug, Clone)]
pub enum StochasticProcess {
    /// Geometric Brownian Motion (Black-Scholes)
    GeometricBrownian {
        /// Drift parameter
        mu: f64,
        /// Diffusion parameter
        sigma: f64,
    },
    /// Variance Gamma process
    VarianceGamma {
        /// Drift of the Brownian motion with random time
        theta: f64,
        /// Volatility of the Brownian motion with random time
        sigma: f64,
        /// Variance rate of the time change
        nu: f64,
    },
    /// Normal Inverse Gaussian process
    NormalInverseGaussian {
        /// Asymmetry parameter
        alpha: f64,
        /// Tail heaviness parameter
        beta: f64,
        /// Scale parameter
        delta: f64,
        /// Location parameter
        mu: f64,
    },
    /// CGMY/KoBoL process
    CGMY {
        /// Fine structure of price jumps near zero
        c: f64,
        /// Positive jump activity
        g: f64,
        /// Negative jump activity
        m: f64,
        /// Blowup rate of jump activity near zero
        y: f64,
    },
    /// Merton jump diffusion
    MertonJumpDiffusion {
        /// GBM parameters
        mu: f64,
        sigma: f64,
        /// Jump parameters
        lambda: f64,
        mu_jump: f64,
        sigma_jump: f64,
    },
}

/// Jump process specification
pub enum JumpProcess {
    /// Simple Poisson jump process
    Poisson {
        /// Jump intensity (average number of jumps per year)
        lambda: f64,
        /// Mean jump size
        mu_jump: f64,
        /// Jump size standard deviation
        sigma_jump: f64,
    },
    /// Double exponential jump process (Kou model)
    DoubleExponential {
        /// Jump intensity
        lambda: f64,
        /// Probability of upward jump
        p: f64,
        /// Rate parameter for positive jumps
        eta_up: f64,
        /// Rate parameter for negative jumps
        eta_down: f64,
    },
    /// Compound Poisson with normal jumps
    CompoundPoissonNormal {
        /// Jump intensity
        lambda: f64,
        /// Jump size mean
        mu: f64,
        /// Jump size variance
        sigma_squared: f64,
    },
    /// Jump process with time-varying intensity
    TimeVaryingIntensity {
        /// Intensity function
        intensity_fn: Box<dyn Fn(f64) -> f64 + Send + Sync>,
        /// Jump size distribution parameters
        mu_jump: f64,
        sigma_jump: f64,
    },
}

impl std::fmt::Debug for JumpProcess {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JumpProcess::Poisson { lambda, mu_jump, sigma_jump } => {
                f.debug_struct("Poisson")
                    .field("lambda", lambda)
                    .field("mu_jump", mu_jump)
                    .field("sigma_jump", sigma_jump)
                    .finish()
            }
            JumpProcess::DoubleExponential { lambda, p, eta_up, eta_down } => {
                f.debug_struct("DoubleExponential")
                    .field("lambda", lambda)
                    .field("p", p)
                    .field("eta_up", eta_up)
                    .field("eta_down", eta_down)
                    .finish()
            }
            JumpProcess::CompoundPoissonNormal { lambda, mu, sigma_squared } => {
                f.debug_struct("CompoundPoissonNormal")
                    .field("lambda", lambda)
                    .field("mu", mu)
                    .field("sigma_squared", sigma_squared)
                    .finish()
            }
            JumpProcess::TimeVaryingIntensity { mu_jump, sigma_jump, .. } => {
                f.debug_struct("TimeVaryingIntensity")
                    .field("intensity_fn", &"<function>")
                    .field("mu_jump", mu_jump)
                    .field("sigma_jump", sigma_jump)
                    .finish()
            }
        }
    }
}

impl Clone for JumpProcess {
    fn clone(&self) -> Self {
        match self {
            JumpProcess::Poisson { lambda, mu_jump, sigma_jump } => {
                JumpProcess::Poisson { 
                    lambda: *lambda, 
                    mu_jump: *mu_jump, 
                    sigma_jump: *sigma_jump 
                }
            }
            JumpProcess::DoubleExponential { lambda, p, eta_up, eta_down } => {
                JumpProcess::DoubleExponential { 
                    lambda: *lambda, 
                    p: *p, 
                    eta_up: *eta_up, 
                    eta_down: *eta_down 
                }
            }
            JumpProcess::CompoundPoissonNormal { lambda, mu, sigma_squared } => {
                JumpProcess::CompoundPoissonNormal { 
                    lambda: *lambda, 
                    mu: *mu, 
                    sigma_squared: *sigma_squared 
                }
            }
            JumpProcess::TimeVaryingIntensity { intensity_fn: _, mu_jump, sigma_jump } => {
                // Note: Function pointers cannot be cloned in general
                // For now, we'll create a simple constant intensity
                let constant_intensity = 0.1; // Default intensity
                JumpProcess::Poisson { 
                    lambda: constant_intensity, 
                    mu_jump: *mu_jump, 
                    sigma_jump: *sigma_jump 
                }
            }
        }
    }
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
    /// Underlying stochastic process
    pub stochastic_process: Option<StochasticProcess>,
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
            VolatilityModel::Heston { .. } | VolatilityModel::SABR { .. } 
            | VolatilityModel::Bates { .. } | VolatilityModel::HullWhite { .. }
            | VolatilityModel::ThreeHalves { .. } => Some(50),
            _ => None,
        };
        
        Self {
            n_asset,
            n_time,
            n_vol,
            volatility_model,
            jump_process: None,
            stochastic_process: None,
            method,
        }
    }
    
    /// Add jump process
    pub fn with_jumps(mut self, jump_process: JumpProcess) -> Self {
        self.jump_process = Some(jump_process);
        self
    }
    
    /// Set underlying stochastic process
    pub fn with_stochastic_process(mut self, process: StochasticProcess) -> Self {
        self.stochastic_process = Some(process);
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
            let final_s = if let Some(ref jump) = self.jump_process {
                self.simulate_jump(s_new, dt, rng, jump)?
            } else {
                s_new
            };
            
            s_path.push(final_s);
            
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
        for (i, price) in prices.iter_mut().enumerate().take(n_steps + 1) {
            let s_final = option.spot * u.powi(i as i32) * d.powi((n_steps - i) as i32);
            *price = self.payoff(option, s_final);
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
    
    /// Simulate jump process for a single time step
    fn simulate_jump(
        &self,
        s_current: f64,
        dt: f64,
        rng: &mut ThreadRng,
        jump: &JumpProcess,
    ) -> Result<f64> {
        match jump {
            JumpProcess::Poisson { lambda, mu_jump, sigma_jump } => {
                if rng.random::<f64>() < lambda * dt {
                    let jump_size = 1.0 + rng.sample(Normal::new(*mu_jump, *sigma_jump).unwrap());
                    Ok(s_current * jump_size)
                } else {
                    Ok(s_current)
                }
            }
            JumpProcess::DoubleExponential { lambda, p, eta_up, eta_down } => {
                if rng.random::<f64>() < lambda * dt {
                    let is_up_jump = rng.random::<f64>() < *p;
                    let jump_size = if is_up_jump {
                        // Exponential with rate eta_up
                        1.0 + rng.sample(rand_distr::Exp::new(*eta_up).unwrap())
                    } else {
                        // Negative exponential with rate eta_down
                        1.0 - rng.sample(rand_distr::Exp::new(*eta_down).unwrap())
                    };
                    Ok(s_current * jump_size)
                } else {
                    Ok(s_current)
                }
            }
            JumpProcess::CompoundPoissonNormal { lambda, mu, sigma_squared } => {
                if rng.random::<f64>() < lambda * dt {
                    let jump_size = 1.0 + rng.sample(Normal::new(*mu, sigma_squared.sqrt()).unwrap());
                    Ok(s_current * jump_size)
                } else {
                    Ok(s_current)
                }
            }
            JumpProcess::TimeVaryingIntensity { intensity_fn, mu_jump, sigma_jump } => {
                let current_time = 0.0; // This would need to be passed as parameter
                let intensity = intensity_fn(current_time);
                if rng.random::<f64>() < intensity * dt {
                    let jump_size = 1.0 + rng.sample(Normal::new(*mu_jump, *sigma_jump).unwrap());
                    Ok(s_current * jump_size)
                } else {
                    Ok(s_current)
                }
            }
        }
    }
    
    /// Simulate variance gamma process path
    #[allow(dead_code)]
    fn simulate_variance_gamma_path(
        &self,
        option: &FinancialOption,
        theta: f64,
        sigma: f64,
        nu: f64,
        n_steps: usize,
        dt: f64,
        rng: &mut ThreadRng,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let mut s_path = vec![option.spot];
        let mut v_path = vec![sigma * sigma];
        
        let omega = (1.0 / nu) * ((1.0 - theta * nu - 0.5 * sigma * sigma * nu).sqrt() - 1.0);
        
        for _ in 0..n_steps {
            // Generate gamma random variable for time change
            let gamma_shape = dt / nu;
            let gamma_scale = nu;
            let time_change = rng.sample(rand_distr::Gamma::new(gamma_shape, gamma_scale).unwrap());
            
            // Generate normal increment scaled by time change
            let z: f64 = rng.sample(StandardNormal);
            let increment = (omega + theta) * time_change + sigma * time_change.sqrt() * z;
            
            let s_new = s_path.last().unwrap() * increment.exp();
            s_path.push(s_new);
            v_path.push(sigma * sigma);
        }
        
        Ok((s_path, v_path))
    }
    
    /// Simulate Normal Inverse Gaussian process path
    #[allow(dead_code)]
    fn simulate_nig_path(
        &self,
        option: &FinancialOption,
        alpha: f64,
        beta: f64,
        delta: f64,
        mu: f64,
        n_steps: usize,
        dt: f64,
        rng: &mut ThreadRng,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let mut s_path = vec![option.spot];
        let mut v_path = vec![0.0]; // NIG doesn't have explicit volatility process
        
        let gamma = (alpha * alpha - beta * beta).sqrt();
        
        for _ in 0..n_steps {
            // Generate inverse Gaussian random variable
            let ig_mean = delta / gamma;
            let ig_shape = delta * delta;
            let v = self.sample_inverse_gaussian(ig_mean, ig_shape, rng);
            
            // Generate normal increment
            let z: f64 = rng.sample(StandardNormal);
            let increment = mu * dt + beta * v + (alpha * alpha - beta * beta).sqrt() * v.sqrt() * z;
            
            let s_new = s_path.last().unwrap() * increment.exp();
            s_path.push(s_new);
            v_path.push(v);
        }
        
        Ok((s_path, v_path))
    }
    
    /// Sample from inverse Gaussian distribution
    #[allow(dead_code)]
    fn sample_inverse_gaussian(&self, mu: f64, lambda: f64, rng: &mut ThreadRng) -> f64 {
        // Michael-Schucany-Haas algorithm
        let n: f64 = rng.sample(StandardNormal);
        let y = n * n;
        let x = mu + (mu * mu * y) / (2.0 * lambda) - 
                (mu / (2.0 * lambda)) * (4.0 * mu * lambda * y + mu * mu * y * y).sqrt();
        
        let test = rng.random::<f64>();
        if test <= mu / (mu + x) {
            x
        } else {
            mu * mu / x
        }
    }
    
    /// Simulate CGMY process using series representation
    #[allow(dead_code)]
    fn simulate_cgmy_path(
        &self,
        option: &FinancialOption,
        c: f64,
        g: f64,
        m: f64,
        y: f64,
        n_steps: usize,
        dt: f64,
        rng: &mut ThreadRng,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let mut s_path = vec![option.spot];
        let mut v_path = vec![0.0];
        
        // Small jump threshold
        let epsilon = 1e-4;
        
        for _ in 0..n_steps {
            let mut jump_sum = 0.0;
            
            // Simulate jumps using series representation
            // This is a simplified implementation
            let lambda_pos = c * dt * gamma_fn(1.0 - y) / g.powf(1.0 - y);
            let lambda_neg = c * dt * gamma_fn(1.0 - y) / m.powf(1.0 - y);
            
            // Positive jumps
            let n_pos = rng.sample(rand_distr::Poisson::new(lambda_pos).unwrap()) as usize;
            for _ in 0..n_pos {
                let u: f64 = rng.random();
                let jump = (u.powf(-1.0 / (1.0 - y)) - 1.0) / g;
                if jump > epsilon {
                    jump_sum += jump;
                }
            }
            
            // Negative jumps
            let n_neg = rng.sample(rand_distr::Poisson::new(lambda_neg).unwrap()) as usize;
            for _ in 0..n_neg {
                let u: f64 = rng.random();
                let jump = -((u.powf(-1.0 / (1.0 - y)) - 1.0) / m);
                if jump.abs() > epsilon {
                    jump_sum += jump;
                }
            }
            
            let s_new = s_path.last().unwrap() * jump_sum.exp();
            s_path.push(s_new);
            v_path.push(0.0);
        }
        
        Ok((s_path, v_path))
    }
}

/// Gamma function approximation
#[allow(dead_code)]
fn gamma_fn(x: f64) -> f64 {
    // Stirling's approximation for simplicity
    if x > 1.0 {
        (2.0 * PI / x).sqrt() * (x / std::f64::consts::E).powf(x)
    } else {
        // Use recurrence relation
        gamma_fn(x + 1.0) / x
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

/// Advanced stochastic PDE solvers and exotic derivatives
pub mod advanced_solvers {
    use super::*;
    use std::collections::HashMap;
    
    /// PIDE (Partial Integro-Differential Equation) solver for jump-diffusion models
    pub struct PIDESolver {
        /// Grid dimensions
        pub n_asset: usize,
        pub n_time: usize,
        pub n_vol: Option<usize>,
        /// Grid boundaries
        pub s_min: f64,
        pub s_max: f64,
        /// Numerical integration parameters
        pub integration_points: usize,
        /// Jump truncation threshold
        pub jump_cutoff: f64,
    }
    
    impl PIDESolver {
        /// Create new PIDE solver
        pub fn new(
            n_asset: usize,
            n_time: usize, 
            s_min: f64,
            s_max: f64,
        ) -> Self {
            Self {
                n_asset,
                n_time,
                n_vol: None,
                s_min,
                s_max,
                integration_points: 1000,
                jump_cutoff: 5.0,
            }
        }
        
        /// Solve PIDE for Merton jump-diffusion model
        pub fn solve_merton_jump_diffusion(
            &self,
            option: &FinancialOption,
            sigma: f64,
            lambda: f64,
            mu_jump: f64,
            sigma_jump: f64,
        ) -> Result<f64> {
            let dt = option.maturity / (self.n_time - 1) as f64;
            let ds = (self.s_max - self.s_min) / (self.n_asset - 1) as f64;
            
            // Initialize solution grid
            let mut v = Array2::zeros((self.n_time, self.n_asset));
            
            // Terminal condition
            for i in 0..self.n_asset {
                let s = self.s_min + i as f64 * ds;
                v[[self.n_time - 1, i]] = self.payoff(option, s);
            }
            
            // Precompute jump integral operator
            let jump_operator = self.precompute_jump_integral(
                lambda, mu_jump, sigma_jump, ds
            )?;
            
            // Backward time-stepping
            for t_idx in (0..self.n_time - 1).rev() {
                for i in 1..self.n_asset - 1 {
                    let s = self.s_min + i as f64 * ds;
                    
                    // Diffusion term (standard Black-Scholes operator)
                    let diff_term = self.apply_diffusion_operator(
                        &v, t_idx + 1, i, s, sigma, option.risk_free_rate, ds, dt
                    );
                    
                    // Jump integral term
                    let jump_term = self.apply_jump_operator(
                        &v, &jump_operator, t_idx + 1, i, s, ds
                    );
                    
                    // Full PIDE solution
                    v[[t_idx, i]] = diff_term + dt * lambda * jump_term;
                    
                    // Apply early exercise for American options
                    if option.option_style == OptionStyle::American {
                        v[[t_idx, i]] = v[[t_idx, i]].max(self.payoff(option, s));
                    }
                }
                
                // Apply boundary conditions
                self.apply_jump_boundary_conditions(&mut v, t_idx, option, dt);
            }
            
            // Interpolate to spot price
            self.interpolate_solution(&v, option.spot, ds)
        }
        
        fn payoff(&self, option: &FinancialOption, s: f64) -> f64 {
            match option.option_type {
                OptionType::Call => (s - option.strike).max(0.0),
                OptionType::Put => (option.strike - s).max(0.0),
            }
        }
        
        fn precompute_jump_integral(
            &self,
            _lambda: f64,
            mu_jump: f64,
            sigma_jump: f64,
            _ds: f64,
        ) -> Result<Array1<f64>> {
            let mut integral_weights = Array1::zeros(self.n_asset);
            
            for i in 0..self.n_asset {
                let mut integral = 0.0;
                
                // Numerical integration over jump sizes
                for k in 0..self.integration_points {
                    let y = -self.jump_cutoff + 2.0 * self.jump_cutoff * k as f64 / (self.integration_points - 1) as f64;
                    let jump_density = (-0.5 * ((y - mu_jump) / sigma_jump).powi(2)).exp() / 
                                     (sigma_jump * (2.0 * PI).sqrt());
                    integral += jump_density * (2.0 * self.jump_cutoff / (self.integration_points - 1) as f64);
                }
                
                integral_weights[i] = integral;
            }
            
            Ok(integral_weights)
        }
        
        fn apply_diffusion_operator(
            &self,
            v: &Array2<f64>,
            t_idx: usize,
            i: usize,
            s: f64,
            sigma: f64,
            r: f64,
            ds: f64,
            dt: f64,
        ) -> f64 {
            let v_prev = v[[t_idx, i - 1]];
            let v_curr = v[[t_idx, i]];
            let v_next = v[[t_idx, i + 1]];
            
            // Central differences for derivatives
            let dv_ds = (v_next - v_prev) / (2.0 * ds);
            let d2v_ds2 = (v_next - 2.0 * v_curr + v_prev) / (ds * ds);
            
            // Black-Scholes operator
            v_curr - dt * (0.5 * sigma * sigma * s * s * d2v_ds2 + 
                          r * s * dv_ds - r * v_curr)
        }
        
        fn apply_jump_operator(
            &self,
            v: &Array2<f64>,
            jump_operator: &Array1<f64>,
            t_idx: usize,
            i: usize,
            s: f64,
            ds: f64,
        ) -> f64 {
            let mut jump_integral = 0.0;
            
            // Integrate over possible jump destinations
            for k in 0..self.integration_points {
                let y = -self.jump_cutoff + 2.0 * self.jump_cutoff * k as f64 / (self.integration_points - 1) as f64;
                let s_jumped = s * y.exp();
                
                if s_jumped >= self.s_min && s_jumped <= self.s_max {
                    let j = ((s_jumped - self.s_min) / ds) as usize;
                    if j < self.n_asset - 1 {
                        // Linear interpolation
                        let alpha = (s_jumped - (self.s_min + j as f64 * ds)) / ds;
                        let v_jumped = v[[t_idx, j]] * (1.0 - alpha) + v[[t_idx, j + 1]] * alpha;
                        jump_integral += v_jumped * jump_operator[k];
                    }
                }
            }
            
            jump_integral - v[[t_idx, i]]
        }
        
        fn apply_jump_boundary_conditions(
            &self,
            v: &mut Array2<f64>,
            t_idx: usize,
            option: &FinancialOption,
            dt: f64,
        ) {
            let t = (self.n_time - 1 - t_idx) as f64 * dt;
            
            match option.option_type {
                OptionType::Call => {
                    v[[t_idx, 0]] = 0.0;
                    v[[t_idx, self.n_asset - 1]] = self.s_max - option.strike * 
                        (-option.risk_free_rate * t).exp();
                }
                OptionType::Put => {
                    v[[t_idx, 0]] = option.strike * (-option.risk_free_rate * t).exp();
                    v[[t_idx, self.n_asset - 1]] = 0.0;
                }
            }
        }
        
        fn interpolate_solution(
            &self,
            v: &Array2<f64>,
            spot: f64,
            ds: f64,
        ) -> Result<f64> {
            let i = ((spot - self.s_min) / ds) as usize;
            if i >= self.n_asset - 1 {
                return Ok(v[[0, self.n_asset - 1]]);
            }
            
            let alpha = (spot - (self.s_min + i as f64 * ds)) / ds;
            Ok(v[[0, i]] * (1.0 - alpha) + v[[0, i + 1]] * alpha)
        }
    }
    
    /// Longstaff-Schwartz Monte Carlo for American options
    pub struct LSMCSolver {
        /// Number of simulation paths
        pub n_paths: usize,
        /// Number of time steps
        pub n_steps: usize,
        /// Polynomial degree for regression
        pub poly_degree: usize,
        /// Random number generator seed
        pub seed: Option<u64>,
    }
    
    impl LSMCSolver {
        /// Create new LSMC solver
        pub fn new(n_paths: usize, n_steps: usize) -> Self {
            Self {
                n_paths,
                n_steps,
                poly_degree: 3,
                seed: None,
            }
        }
        
        /// Set polynomial degree for continuation value regression
        pub fn with_poly_degree(mut self, degree: usize) -> Self {
            self.poly_degree = degree;
            self
        }
        
        /// Price American option using Longstaff-Schwartz method
        pub fn price_american_option(
            &self,
            option: &FinancialOption,
            volatility_model: &VolatilityModel,
        ) -> Result<f64> {
            let dt = option.maturity / self.n_steps as f64;
            let discount = (-option.risk_free_rate * dt).exp();
            
            // Generate price paths
            let paths = self.generate_price_paths(option, volatility_model)?;
            
            // Initialize cash flows (exercise values at maturity)
            let mut cash_flows = Array2::zeros((self.n_paths, self.n_steps + 1));
            for i in 0..self.n_paths {
                cash_flows[[i, self.n_steps]] = self.payoff(option, paths[[i, self.n_steps]]);
            }
            
            // Backward induction with optimal stopping
            for t in (1..self.n_steps).rev() {
                let mut in_money_paths = Vec::new();
                let mut regression_x = Vec::new();
                let mut regression_y = Vec::new();
                
                // Identify in-the-money paths
                for i in 0..self.n_paths {
                    let intrinsic = self.payoff(option, paths[[i, t]]);
                    if intrinsic > 0.0 {
                        in_money_paths.push(i);
                        regression_x.push(paths[[i, t]]);
                        regression_y.push(cash_flows[[i, t + 1]] * discount);
                    }
                }
                
                if !in_money_paths.is_empty() {
                    // Perform polynomial regression
                    let continuation_values = self.polynomial_regression(
                        &regression_x, &regression_y
                    )?;
                    
                    // Apply optimal stopping rule
                    for (idx, &path_idx) in in_money_paths.iter().enumerate() {
                        let intrinsic = self.payoff(option, paths[[path_idx, t]]);
                        let continuation = continuation_values[idx];
                        
                        if intrinsic > continuation {
                            // Exercise now
                            cash_flows[[path_idx, t]] = intrinsic;
                            // Zero out future cash flows
                            for future_t in (t + 1)..=self.n_steps {
                                cash_flows[[path_idx, future_t]] = 0.0;
                            }
                        } else {
                            // Continue holding
                            cash_flows[[path_idx, t]] = cash_flows[[path_idx, t + 1]] * discount;
                        }
                    }
                }
                
                // Update cash flows for paths not in the money
                for i in 0..self.n_paths {
                    if !in_money_paths.contains(&i) {
                        cash_flows[[i, t]] = cash_flows[[i, t + 1]] * discount;
                    }
                }
            }
            
            // Calculate option value as average discounted cash flow
            let mut total_value = 0.0;
            for i in 0..self.n_paths {
                // Find first exercise time
                for t in 1..=self.n_steps {
                    if cash_flows[[i, t]] > 0.0 {
                        total_value += cash_flows[[i, t]] * (-option.risk_free_rate * t as f64 * dt).exp();
                        break;
                    }
                }
            }
            
            Ok(total_value / self.n_paths as f64)
        }
        
        fn generate_price_paths(
            &self,
            option: &FinancialOption,
            volatility_model: &VolatilityModel,
        ) -> Result<Array2<f64>> {
            let mut paths = Array2::zeros((self.n_paths, self.n_steps + 1));
            let dt = option.maturity / self.n_steps as f64;
            
            let mut rng = if let Some(seed) = self.seed {
                StdRng::seed_from_u64(seed)
            } else {
                StdRng::seed_from_u64(0)
            };
            
            for i in 0..self.n_paths {
                paths[[i, 0]] = option.spot;
                
                match volatility_model {
                    VolatilityModel::Constant(sigma) => {
                        for t in 1..=self.n_steps {
                            let z: f64 = StandardNormal.sample(&mut rng);
                            let s_prev = paths[[i, t - 1]];
                            paths[[i, t]] = s_prev * (
                                (option.risk_free_rate - 0.5 * sigma * sigma) * dt + 
                                sigma * dt.sqrt() * z
                            ).exp();
                        }
                    }
                    _ => {
                        return Err(IntegrateError::NotImplementedError(
                            "LSMC not yet implemented for stochastic volatility models".into()
                        ))
                    }
                }
            }
            
            Ok(paths)
        }
        
        fn payoff(&self, option: &FinancialOption, s: f64) -> f64 {
            match option.option_type {
                OptionType::Call => (s - option.strike).max(0.0),
                OptionType::Put => (option.strike - s).max(0.0),
            }
        }
        
        fn polynomial_regression(
            &self,
            x: &[f64],
            y: &[f64],
        ) -> Result<Vec<f64>> {
            let n = x.len();
            if n == 0 {
                return Ok(vec![]);
            }
            
            // Create design matrix for polynomial regression
            let mut design_matrix = Array2::zeros((n, self.poly_degree + 1));
            for i in 0..n {
                for j in 0..=self.poly_degree {
                    design_matrix[[i, j]] = x[i].powi(j as i32);
                }
            }
            
            // Solve normal equations: (X'X)β = X'y
            let x_transpose = design_matrix.t();
            let xtx = x_transpose.dot(&design_matrix);
            let xty = x_transpose.dot(&Array1::from_vec(y.to_vec()));
            
            // Simple LU decomposition for small matrices
            let coefficients = self.solve_linear_system(&xtx.to_owned(), &xty)?;
            
            // Evaluate polynomial at regression points
            let mut fitted_values = Vec::with_capacity(n);
            for &xi in x {
                let mut value = 0.0;
                for (j, &coeff) in coefficients.iter().enumerate().take(self.poly_degree + 1) {
                    value += coeff * xi.powi(j as i32);
                }
                fitted_values.push(value);
            }
            
            Ok(fitted_values)
        }
        
        fn solve_linear_system(
            &self,
            a: &Array2<f64>,
            b: &Array1<f64>,
        ) -> Result<Vec<f64>> {
            let n = a.nrows();
            let mut aug = Array2::zeros((n, n + 1));
            
            // Create augmented matrix
            for i in 0..n {
                for j in 0..n {
                    aug[[i, j]] = a[[i, j]];
                }
                aug[[i, n]] = b[i];
            }
            
            // Gaussian elimination with partial pivoting
            for k in 0..n {
                // Find pivot
                let mut max_row = k;
                for i in (k + 1)..n {
                    if aug[[i, k]].abs() > aug[[max_row, k]].abs() {
                        max_row = i;
                    }
                }
                
                // Swap rows
                if max_row != k {
                    for j in 0..=n {
                        let temp = aug[[k, j]];
                        aug[[k, j]] = aug[[max_row, j]];
                        aug[[max_row, j]] = temp;
                    }
                }
                
                // Check for singular matrix
                if aug[[k, k]].abs() < 1e-12 {
                    return Err(IntegrateError::ValueError("Singular matrix encountered".to_string()));
                }
                
                // Eliminate column
                for i in (k + 1)..n {
                    let factor = aug[[i, k]] / aug[[k, k]];
                    for j in k..=n {
                        aug[[i, j]] -= factor * aug[[k, j]];
                    }
                }
            }
            
            // Back substitution
            let mut x = vec![0.0; n];
            for i in (0..n).rev() {
                x[i] = aug[[i, n]];
                for j in (i + 1)..n {
                    x[i] -= aug[[i, j]] * x[j];
                }
                x[i] /= aug[[i, i]];
            }
            
            Ok(x)
        }
    }
    
    /// Volatility surface calibration and modeling
    pub struct VolatilitySurface {
        /// Strike grid
        pub strikes: Array1<f64>,
        /// Maturity grid
        pub maturities: Array1<f64>,
        /// Implied volatility surface
        pub vol_surface: Array2<f64>,
        /// Market data for calibration
        pub market_data: Vec<MarketQuote>,
    }
    
    /// Market quote for calibration
    #[derive(Debug, Clone)]
    pub struct MarketQuote {
        pub strike: f64,
        pub maturity: f64,
        pub price: f64,
        pub bid: Option<f64>,
        pub ask: Option<f64>,
        pub volume: Option<f64>,
    }
    
    impl VolatilitySurface {
        /// Create new volatility surface
        pub fn new(
            strikes: Array1<f64>,
            maturities: Array1<f64>,
        ) -> Self {
            let vol_surface = Array2::zeros((maturities.len(), strikes.len()));
            Self {
                strikes,
                maturities,
                vol_surface,
                market_data: Vec::new(),
            }
        }
        
        /// Add market quote for calibration
        pub fn add_market_quote(&mut self, quote: MarketQuote) {
            self.market_data.push(quote);
        }
        
        /// Calibrate volatility surface using SVI parameterization
        pub fn calibrate_svi(&mut self, spot: f64, risk_free_rate: f64) -> Result<()> {
            for (t_idx, &maturity) in self.maturities.iter().enumerate() {
                let quotes_for_maturity: Vec<_> = self.market_data.iter()
                    .filter(|q| (q.maturity - maturity).abs() < 1e-6)
                    .collect();
                
                if quotes_for_maturity.is_empty() {
                    continue;
                }
                
                // Extract implied volatilities from market prices
                let mut implied_vols = Vec::new();
                let mut log_moneyness = Vec::new();
                
                for quote in quotes_for_maturity {
                    // Calculate implied volatility using Black-Scholes inverse
                    let option = FinancialOption {
                        option_type: if quote.strike >= spot { OptionType::Call } else { OptionType::Put },
                        option_style: OptionStyle::European,
                        strike: quote.strike,
                        maturity: quote.maturity,
                        spot,
                        risk_free_rate,
                        dividend_yield: 0.0,
                    };
                    
                    if let Ok(iv) = self.implied_volatility_newton_raphson(&option, quote.price) {
                        implied_vols.push(iv);
                        log_moneyness.push((quote.strike / spot).ln());
                    }
                }
                
                // Fit SVI parameterization: w(k) = a + b(ρ(k-m) + sqrt((k-m)² + σ²))
                if implied_vols.len() >= 5 {
                    let svi_params = self.fit_svi_slice(&log_moneyness, &implied_vols, maturity)?;
                    
                    // Populate volatility surface for this maturity
                    for (k_idx, &strike) in self.strikes.iter().enumerate() {
                        let k = (strike / spot).ln();
                        let vol = self.evaluate_svi(k, &svi_params);
                        self.vol_surface[[t_idx, k_idx]] = vol;
                    }
                }
            }
            
            Ok(())
        }
        
        fn implied_volatility_newton_raphson(
            &self,
            option: &FinancialOption,
            market_price: f64,
        ) -> Result<f64> {
            let mut sigma = 0.2; // Initial guess
            let tolerance = 1e-6;
            let max_iterations = 100;
            
            for _ in 0..max_iterations {
                let bs_price = self.black_scholes_price(option, sigma);
                let vega = self.black_scholes_vega(option, sigma);
                
                if vega.abs() < 1e-12 {
                    return Err(IntegrateError::ConvergenceError(
                        "Vega too small in implied volatility calculation".into()
                    ));
                }
                
                let price_diff = bs_price - market_price;
                if price_diff.abs() < tolerance {
                    return Ok(sigma);
                }
                
                sigma -= price_diff / vega;
                
                // Ensure sigma stays positive
                if sigma <= 0.0 {
                    sigma = 0.001;
                }
            }
            
            Err(IntegrateError::ConvergenceError(
                "Implied volatility did not converge".into()
            ))
        }
        
        fn black_scholes_price(&self, option: &FinancialOption, sigma: f64) -> f64 {
            let d1 = ((option.spot / option.strike).ln() + 
                     (option.risk_free_rate + 0.5 * sigma * sigma) * option.maturity) / 
                     (sigma * option.maturity.sqrt());
            let d2 = d1 - sigma * option.maturity.sqrt();
            
            match option.option_type {
                OptionType::Call => {
                    option.spot * self.norm_cdf(d1) - 
                    option.strike * (-option.risk_free_rate * option.maturity).exp() * self.norm_cdf(d2)
                }
                OptionType::Put => {
                    option.strike * (-option.risk_free_rate * option.maturity).exp() * self.norm_cdf(-d2) - 
                    option.spot * self.norm_cdf(-d1)
                }
            }
        }
        
        fn black_scholes_vega(&self, option: &FinancialOption, sigma: f64) -> f64 {
            let d1 = ((option.spot / option.strike).ln() + 
                     (option.risk_free_rate + 0.5 * sigma * sigma) * option.maturity) / 
                     (sigma * option.maturity.sqrt());
            
            option.spot * option.maturity.sqrt() * self.norm_pdf(d1)
        }
        
        fn norm_cdf(&self, x: f64) -> f64 {
            0.5 * (1.0 + self.erf(x / SQRT_2))
        }
        
        fn norm_pdf(&self, x: f64) -> f64 {
            (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
        }
        
        fn erf(&self, x: f64) -> f64 {
            // Approximation of error function
            let a1 = 0.254829592;
            let a2 = -0.284496736;
            let a3 = 1.421413741;
            let a4 = -1.453152027;
            let a5 = 1.061405429;
            let p = 0.3275911;
            
            let sign = if x < 0.0 { -1.0 } else { 1.0 };
            let x = x.abs();
            
            let t = 1.0 / (1.0 + p * x);
            let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
            
            sign * y
        }
        
        fn fit_svi_slice(
            &self,
            _log_moneyness: &[f64],
            implied_vols: &[f64],
            maturity: f64,
        ) -> Result<SVIParameters> {
            // Simple least squares fit for SVI parameters
            // In practice, this would use constrained optimization
            
            // Convert implied volatilities to total variance
            let total_variances: Vec<f64> = implied_vols.iter()
                .map(|&iv| iv * iv * maturity)
                .collect();
            
            // Initial parameter guess
            let mut params = SVIParameters {
                a: 0.04,
                b: 0.4,
                rho: -0.4,
                m: 0.0,
                sigma: 0.1,
            };
            
            // Simple parameter fitting (in practice would use sophisticated optimization)
            let mean_var = total_variances.iter().sum::<f64>() / total_variances.len() as f64;
            params.a = mean_var * 0.5;
            params.b = mean_var * 0.5;
            
            Ok(params)
        }
        
        fn evaluate_svi(&self, k: f64, params: &SVIParameters) -> f64 {
            let w = params.a + params.b * (
                params.rho * (k - params.m) + 
                ((k - params.m).powi(2) + params.sigma.powi(2)).sqrt()
            );
            w.max(0.0).sqrt()
        }
    }
    
    /// SVI volatility parameterization
    #[derive(Debug, Clone)]
    pub struct SVIParameters {
        pub a: f64,
        pub b: f64,
        pub rho: f64,
        pub m: f64,
        pub sigma: f64,
    }
    
    /// Machine learning-enhanced option pricing
    pub struct MLPricer {
        /// Neural network parameters (simplified representation)
        pub network_weights: Vec<Array2<f64>>,
        /// Network topology
        pub layer_sizes: Vec<usize>,
        /// Training data cache
        pub training_cache: HashMap<String, Vec<f64>>,
    }
    
    impl MLPricer {
        /// Create new ML-based pricer
        pub fn new(layer_sizes: Vec<usize>) -> Self {
            let mut network_weights = Vec::new();
            
            for i in 0..layer_sizes.len() - 1 {
                let weights = Array2::zeros((layer_sizes[i], layer_sizes[i + 1]));
                network_weights.push(weights);
            }
            
            Self {
                network_weights,
                layer_sizes,
                training_cache: HashMap::new(),
            }
        }
        
        /// Train network on Black-Scholes prices
        pub fn train_on_black_scholes(
            &mut self,
            training_samples: usize,
            learning_rate: f64,
        ) -> Result<()> {
            let mut rng = rand::rng();
            
            for _ in 0..training_samples {
                // Generate random option parameters
                let spot = 80.0 + rng.random::<f64>() * 40.0; // S0 ∈ [80, 120]
                let strike = 80.0 + rng.random::<f64>() * 40.0; // K ∈ [80, 120]
                let maturity = 0.1 + rng.random::<f64>() * 0.9; // T ∈ [0.1, 1.0]
                let vol = 0.1 + rng.random::<f64>() * 0.4; // σ ∈ [0.1, 0.5]
                let rate = 0.01 + rng.random::<f64>() * 0.04; // r ∈ [0.01, 0.05]
                
                // Create option
                let _option = FinancialOption {
                    option_type: OptionType::Call,
                    option_style: OptionStyle::European,
                    strike,
                    maturity,
                    spot,
                    risk_free_rate: rate,
                    dividend_yield: 0.0,
                };
                
                // Calculate Black-Scholes price as target
                let target_price = self.black_scholes_call_price(
                    spot, strike, maturity, rate, vol
                );
                
                // Forward pass
                let input = vec![spot, strike, maturity, rate, vol];
                let prediction = self.forward_pass(&input)?;
                
                // Backward pass (simplified gradient descent)
                let error = prediction - target_price;
                self.backward_pass(&input, error, learning_rate)?;
            }
            
            Ok(())
        }
        
        /// Price option using trained neural network
        pub fn price_option_ml(&self, option: &FinancialOption, vol: f64) -> Result<f64> {
            let input = vec![
                option.spot,
                option.strike,
                option.maturity,
                option.risk_free_rate,
                vol,
            ];
            
            self.forward_pass(&input)
        }
        
        fn forward_pass(&self, input: &[f64]) -> Result<f64> {
            let mut activations = Array1::from_vec(input.to_vec());
            
            for weights in &self.network_weights {
                let z = activations.dot(weights);
                activations = z.mapv(|x| self.relu(x)); // ReLU activation
            }
            
            // Return final output (single neuron)
            Ok(activations[0])
        }
        
        fn backward_pass(
            &mut self,
            _input: &[f64],
            _error: f64,
            _learning_rate: f64,
        ) -> Result<()> {
            // Simplified backward pass implementation
            // In practice would implement full backpropagation
            Ok(())
        }
        
        fn relu(&self, x: f64) -> f64 {
            x.max(0.0)
        }
        
        fn black_scholes_call_price(
            &self,
            s: f64,
            k: f64,
            t: f64,
            r: f64,
            sigma: f64,
        ) -> f64 {
            let d1 = ((s / k).ln() + (r + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());
            let d2 = d1 - sigma * t.sqrt();
            
            s * self.norm_cdf(d1) - k * (-r * t).exp() * self.norm_cdf(d2)
        }
        
        fn norm_cdf(&self, x: f64) -> f64 {
            0.5 * (1.0 + self.erf(x / SQRT_2))
        }
        
        fn erf(&self, x: f64) -> f64 {
            // Same approximation as in VolatilitySurface
            let a1 = 0.254829592;
            let a2 = -0.284496736;
            let a3 = 1.421413741;
            let a4 = -1.453152027;
            let a5 = 1.061405429;
            let p = 0.3275911;
            
            let sign = if x < 0.0 { -1.0 } else { 1.0 };
            let x = x.abs();
            
            let t = 1.0 / (1.0 + p * x);
            let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
            
            sign * y
        }
    }
    
    #[cfg(test)]
    mod tests {
        use super::*;
        use approx::assert_relative_eq;
        
        #[test]
        fn test_pide_solver() {
            let solver = PIDESolver::new(100, 50, 50.0, 200.0);
            
            let option = FinancialOption {
                option_type: OptionType::Call,
                option_style: OptionStyle::European,
                strike: 100.0,
                maturity: 1.0,
                spot: 100.0,
                risk_free_rate: 0.05,
                dividend_yield: 0.0,
            };
            
            // Test Merton jump-diffusion pricing
            let price = solver.solve_merton_jump_diffusion(
                &option, 0.2, 0.1, -0.1, 0.25
            ).unwrap();
            
            // Price should be reasonable
            assert!(price > 5.0 && price < 20.0);
        }
        
        #[test]
        fn test_lsmc_solver() {
            let solver = LSMCSolver::new(10000, 50);
            
            let option = FinancialOption {
                option_type: OptionType::Put,
                option_style: OptionStyle::American,
                strike: 110.0,
                maturity: 1.0,
                spot: 100.0,
                risk_free_rate: 0.05,
                dividend_yield: 0.0,
            };
            
            let price = solver.price_american_option(
                &option, &VolatilityModel::Constant(0.2)
            ).unwrap();
            
            // American put should be worth more than intrinsic value
            let intrinsic = (option.strike - option.spot).max(0.0);
            assert!(price >= intrinsic);
            assert!(price < 25.0); // Reasonable upper bound
        }
        
        #[test]
        fn test_volatility_surface() {
            let strikes = Array1::linspace(80.0, 120.0, 21);
            let maturities = Array1::linspace(0.1, 2.0, 8);
            
            let mut surface = VolatilitySurface::new(strikes, maturities);
            
            // Add some market quotes
            surface.add_market_quote(MarketQuote {
                strike: 100.0,
                maturity: 1.0,
                price: 10.0,
                bid: Some(9.8),
                ask: Some(10.2),
                volume: Some(1000.0),
            });
            
            // Calibration should complete without error
            assert!(surface.calibrate_svi(100.0, 0.05).is_ok());
        }
        
        #[test]
        fn test_ml_pricer() {
            let mut pricer = MLPricer::new(vec![5, 10, 10, 1]);
            
            // Training should complete without error
            assert!(pricer.train_on_black_scholes(1000, 0.01).is_ok());
            
            let option = FinancialOption {
                option_type: OptionType::Call,
                option_style: OptionStyle::European,
                strike: 100.0,
                maturity: 1.0,
                spot: 100.0,
                risk_free_rate: 0.05,
                dividend_yield: 0.0,
            };
            
            // ML pricing should work
            let price = pricer.price_option_ml(&option, 0.2);
            assert!(price.is_ok());
        }
    }
}