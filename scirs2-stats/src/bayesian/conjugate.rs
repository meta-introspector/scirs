//! Conjugate prior distributions for Bayesian inference
//!
//! This module implements conjugate prior-posterior relationships for efficient Bayesian updates.

use ndarray::{Array1, ArrayView1};
use crate::error::{StatsResult as Result, StatsError};
use scirs2_core::validation::*;

/// Beta-Binomial conjugate pair
///
/// Prior: Beta(α, β)
/// Likelihood: Binomial(n, p)
/// Posterior: Beta(α + successes, β + failures)
#[derive(Debug, Clone)]
pub struct BetaBinomial {
    /// Alpha parameter of the Beta prior
    pub alpha: f64,
    /// Beta parameter of the Beta prior  
    pub beta: f64,
}

impl BetaBinomial {
    /// Create a new Beta-Binomial conjugate prior
    pub fn new(alpha: f64, beta: f64) -> Result<Self> {
        check_positive(alpha, "alpha")?;
        check_positive(beta, "beta")?;
        Ok(Self { alpha, beta })
    }

    /// Update the prior with observed data
    ///
    /// # Arguments
    /// * `successes` - Number of successes observed
    /// * `failures` - Number of failures observed
    ///
    /// # Returns
    /// Updated BetaBinomial with posterior parameters
    pub fn update(&self, successes: usize, failures: usize) -> Self {
        Self {
            alpha: self.alpha + successes as f64,
            beta: self.beta + failures as f64,
        }
    }

    /// Compute the posterior mean
    pub fn posterior_mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    /// Compute the posterior variance
    pub fn posterior_variance(&self) -> f64 {
        let total = self.alpha + self.beta;
        (self.alpha * self.beta) / (total * total * (total + 1.0))
    }

    /// Compute the posterior mode (MAP estimate)
    pub fn posterior_mode(&self) -> Option<f64> {
        if self.alpha > 1.0 && self.beta > 1.0 {
            Some((self.alpha - 1.0) / (self.alpha + self.beta - 2.0))
        } else {
            None
        }
    }

    /// Compute credible interval
    pub fn credible_interval(&self, confidence: f64) -> Result<(f64, f64)> {
        check_probability(confidence, "confidence")?;
        
        // Use beta distribution quantiles
        use crate::distributions::beta::Beta;
        let dist = Beta::new(self.alpha, self.beta, 0.0, 1.0)?;
        
        let alpha_level = (1.0 - confidence) / 2.0;
        Ok((dist.ppf(alpha_level)?, dist.ppf(1.0 - alpha_level)?))
    }
}

/// Gamma-Poisson (Gamma-Negative Binomial) conjugate pair
///
/// Prior: Gamma(α, β)
/// Likelihood: Poisson(λ)
/// Posterior: Gamma(α + sum(data), β + n)
#[derive(Debug, Clone)]
pub struct GammaPoisson {
    /// Shape parameter of the Gamma prior
    pub alpha: f64,
    /// Rate parameter of the Gamma prior
    pub beta: f64,
}

impl GammaPoisson {
    /// Create a new Gamma-Poisson conjugate prior
    pub fn new(alpha: f64, beta: f64) -> Result<Self> {
        check_positive(alpha, "alpha")?;
        check_positive(beta, "beta")?;
        Ok(Self { alpha, beta })
    }

    /// Update the prior with observed count data
    ///
    /// # Arguments
    /// * `data` - Array of observed counts
    ///
    /// # Returns
    /// Updated GammaPoisson with posterior parameters
    pub fn update(&self, data: ArrayView1<f64>) -> Result<Self> {
        check_array_finite(&data, "data")?;
        let sum: f64 = data.sum();
        let n = data.len() as f64;
        
        Ok(Self {
            alpha: self.alpha + sum,
            beta: self.beta + n,
        })
    }

    /// Compute the posterior mean
    pub fn posterior_mean(&self) -> f64 {
        self.alpha / self.beta
    }

    /// Compute the posterior variance
    pub fn posterior_variance(&self) -> f64 {
        self.alpha / (self.beta * self.beta)
    }

    /// Compute the posterior mode (MAP estimate)
    pub fn posterior_mode(&self) -> Option<f64> {
        if self.alpha >= 1.0 {
            Some((self.alpha - 1.0) / self.beta)
        } else {
            None
        }
    }

    /// Compute credible interval
    pub fn credible_interval(&self, confidence: f64) -> Result<(f64, f64)> {
        check_probability(confidence, "confidence")?;
        
        // Use gamma distribution quantiles
        use crate::distributions::gamma::Gamma;
        let dist = Gamma::new(self.alpha, 1.0 / self.beta, 0.0)?; // Note: using scale parameterization
        
        let alpha_level = (1.0 - confidence) / 2.0;
        Ok((dist.ppf(alpha_level)?, dist.ppf(1.0 - alpha_level)?))
    }
}

/// Normal-Normal conjugate pair with known variance
///
/// Prior: Normal(μ₀, σ₀²)
/// Likelihood: Normal(μ, σ²) with known σ²
/// Posterior: Normal(μₙ, σₙ²)
#[derive(Debug, Clone)]
pub struct NormalKnownVariance {
    /// Prior mean
    pub prior_mean: f64,
    /// Prior variance
    pub prior_variance: f64,
    /// Known data variance
    pub data_variance: f64,
}

impl NormalKnownVariance {
    /// Create a new Normal conjugate prior with known data variance
    pub fn new(prior_mean: f64, prior_variance: f64, data_variance: f64) -> Result<Self> {
        check_positive(prior_variance, "prior_variance")?;
        check_positive(data_variance, "data_variance")?;
        Ok(Self {
            prior_mean,
            prior_variance,
            data_variance,
        })
    }

    /// Update the prior with observed data
    ///
    /// # Arguments
    /// * `data` - Array of observed values
    ///
    /// # Returns
    /// Updated NormalKnownVariance with posterior parameters
    pub fn update(&self, data: ArrayView1<f64>) -> Result<Self> {
        check_array_finite(&data, "data")?;
        let n = data.len() as f64;
        let data_mean = data.mean().unwrap_or(0.0);
        
        let precision_prior = 1.0 / self.prior_variance;
        let precision_data = n / self.data_variance;
        let precision_posterior = precision_prior + precision_data;
        
        let posterior_variance = 1.0 / precision_posterior;
        let posterior_mean = (precision_prior * self.prior_mean + precision_data * data_mean) / precision_posterior;
        
        Ok(Self {
            prior_mean: posterior_mean,
            prior_variance: posterior_variance,
            data_variance: self.data_variance,
        })
    }

    /// Compute the posterior mean
    pub fn posterior_mean(&self) -> f64 {
        self.prior_mean
    }

    /// Compute the posterior variance
    pub fn posterior_variance(&self) -> f64 {
        self.prior_variance
    }

    /// Compute credible interval
    pub fn credible_interval(&self, confidence: f64) -> Result<(f64, f64)> {
        check_probability(confidence, "confidence")?;
        
        // Use normal distribution quantiles
        use crate::distributions::normal::Normal;
        let dist = Normal::new(self.prior_mean, self.prior_variance.sqrt())?;
        
        let alpha_level = (1.0 - confidence) / 2.0;
        Ok((dist.ppf(alpha_level)?, dist.ppf(1.0 - alpha_level)?))
    }

    /// Compute the predictive distribution parameters
    pub fn predictive_params(&self) -> (f64, f64) {
        (self.prior_mean, self.prior_variance + self.data_variance)
    }
}

/// Dirichlet-Multinomial conjugate pair
///
/// Prior: Dirichlet(α)
/// Likelihood: Multinomial(n, p)
/// Posterior: Dirichlet(α + counts)
#[derive(Debug, Clone)]
pub struct DirichletMultinomial {
    /// Concentration parameters of the Dirichlet prior
    pub alpha: Array1<f64>,
}

impl DirichletMultinomial {
    /// Create a new Dirichlet-Multinomial conjugate prior
    pub fn new(alpha: Array1<f64>) -> Result<Self> {
        check_array_finite(&alpha, "alpha")?;
        for &a in alpha.iter() {
            check_positive(a, "alpha element")?;
        }
        Ok(Self { alpha })
    }

    /// Create uniform prior with given dimension
    pub fn uniform(k: usize) -> Result<Self> {
        check_positive(k, "k")?;
        Ok(Self {
            alpha: Array1::from_elem(k, 1.0),
        })
    }

    /// Update the prior with observed count data
    ///
    /// # Arguments
    /// * `counts` - Array of observed counts for each category
    ///
    /// # Returns
    /// Updated DirichletMultinomial with posterior parameters
    pub fn update(&self, counts: ArrayView1<f64>) -> Result<Self> {
        if counts.len() != self.alpha.len() {
            return Err(StatsError::DimensionMismatch(
                format!("counts length ({}) must match alpha length ({})", counts.len(), self.alpha.len())
            ));
        }
        check_array_finite(&counts, "counts")?;
        
        Ok(Self {
            alpha: &self.alpha + &counts,
        })
    }

    /// Compute the posterior mean
    pub fn posterior_mean(&self) -> Array1<f64> {
        let sum = self.alpha.sum();
        &self.alpha / sum
    }

    /// Compute the posterior mode (MAP estimate)
    pub fn posterior_mode(&self) -> Option<Array1<f64>> {
        let k = self.alpha.len() as f64;
        if self.alpha.iter().all(|&a| a > 1.0) {
            let sum = self.alpha.sum();
            Some((&self.alpha - 1.0) / (sum - k))
        } else {
            None
        }
    }

    /// Compute the marginal variance for each component
    pub fn posterior_variance(&self) -> Array1<f64> {
        let sum = self.alpha.sum();
        let mean = self.posterior_mean();
        mean.mapv(|p| p * (1.0 - p) / (sum + 1.0))
    }
}

/// Normal-Inverse-Gamma conjugate pair for unknown mean and variance
///
/// Prior: NIG(μ₀, λ, α, β)
/// Likelihood: Normal(μ, σ²) with both unknown
/// Posterior: NIG(μₙ, λₙ, αₙ, βₙ)
#[derive(Debug, Clone)]
pub struct NormalInverseGamma {
    /// Prior mean
    pub mu0: f64,
    /// Prior precision factor
    pub lambda: f64,
    /// Shape parameter
    pub alpha: f64,
    /// Scale parameter
    pub beta: f64,
}

impl NormalInverseGamma {
    /// Create a new Normal-Inverse-Gamma conjugate prior
    pub fn new(mu0: f64, lambda: f64, alpha: f64, beta: f64) -> Result<Self> {
        check_positive(lambda, "lambda")?;
        check_positive(alpha, "alpha")?;
        check_positive(beta, "beta")?;
        Ok(Self { mu0, lambda, alpha, beta })
    }

    /// Update the prior with observed data
    pub fn update(&self, data: ArrayView1<f64>) -> Result<Self> {
        check_array_finite(&data, "data")?;
        let n = data.len() as f64;
        let data_mean = data.mean().unwrap_or(0.0);
        
        // Compute sum of squares
        let ss = data.mapv(|x| (x - data_mean).powi(2)).sum();
        
        // Update parameters
        let lambda_n = self.lambda + n;
        let mu_n = (self.lambda * self.mu0 + n * data_mean) / lambda_n;
        let alpha_n = self.alpha + n / 2.0;
        let beta_n = self.beta + 0.5 * ss + 0.5 * self.lambda * n * (data_mean - self.mu0).powi(2) / lambda_n;
        
        Ok(Self {
            mu0: mu_n,
            lambda: lambda_n,
            alpha: alpha_n,
            beta: beta_n,
        })
    }

    /// Compute the posterior mean of μ
    pub fn posterior_mean_mu(&self) -> f64 {
        self.mu0
    }

    /// Compute the posterior mean of σ²
    pub fn posterior_mean_variance(&self) -> f64 {
        self.beta / (self.alpha - 1.0)
    }

    /// Compute the marginal posterior variance of μ
    pub fn posterior_variance_mu(&self) -> f64 {
        self.beta / (self.lambda * (self.alpha - 1.0))
    }
}