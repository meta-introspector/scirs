//! Statistical distribution traits
//!
//! This module defines the core traits for statistical distributions,
//! including standard distributions and specialized circular distributions.

use crate::error::StatsResult;
use ndarray::Array1;
use num_traits::Float;

/// Base trait for all statistical distributions
pub trait Distribution<F: Float> {
    /// Mean (expected value) of the distribution
    fn mean(&self) -> F;

    /// Variance of the distribution
    fn var(&self) -> F;

    /// Standard deviation of the distribution
    fn std(&self) -> F;

    /// Generate random samples from the distribution
    ///
    /// # Arguments
    ///
    /// * `size` - Number of samples to generate
    ///
    /// # Returns
    ///
    /// An array of random samples from the distribution
    fn rvs(&self, size: usize) -> StatsResult<Array1<F>>;

    /// Entropy of the distribution
    fn entropy(&self) -> F;
}

/// Trait for continuous distributions
pub trait ContinuousDistribution<F: Float>: Distribution<F> {
    /// Probability density function (PDF)
    ///
    /// # Arguments
    ///
    /// * `x` - Point at which to evaluate the PDF
    ///
    /// # Returns
    ///
    /// The probability density at x
    fn pdf(&self, x: F) -> F;

    /// Cumulative distribution function (CDF)
    ///
    /// # Arguments
    ///
    /// * `x` - Point at which to evaluate the CDF
    ///
    /// # Returns
    ///
    /// The cumulative probability up to x
    fn cdf(&self, x: F) -> F;

    /// Percent point function (inverse CDF)
    ///
    /// # Arguments
    ///
    /// * `q` - Quantile (probability) in [0, 1]
    ///
    /// # Returns
    ///
    /// The value x such that CDF(x) = q
    fn ppf(&self, q: F) -> StatsResult<F> {
        // Default implementation using binary search
        // This can be overridden for distributions with analytical ppf
        if q < F::zero() || q > F::one() {
            return Err(crate::error::StatsError::InvalidArgument(
                "Quantile must be in [0, 1]".to_string(),
            ));
        }

        // Use binary search to find the inverse
        let mut low = F::from(-10.0).unwrap();
        let mut high = F::from(10.0).unwrap();
        let eps = F::from(1e-12).unwrap();

        // Find a reasonable search range
        while self.cdf(low) > q {
            low = low * F::from(2.0).unwrap();
        }
        while self.cdf(high) < q {
            high = high * F::from(2.0).unwrap();
        }

        // Binary search
        for _ in 0..100 {
            let mid = (low + high) / F::from(2.0).unwrap();
            let cdf_mid = self.cdf(mid);

            if (cdf_mid - q).abs() < eps {
                return Ok(mid);
            }

            if cdf_mid < q {
                low = mid;
            } else {
                high = mid;
            }
        }

        Ok((low + high) / F::from(2.0).unwrap())
    }
}

/// Trait for discrete distributions
pub trait DiscreteDistribution<F: Float>: Distribution<F> {
    /// Probability mass function (PMF)
    ///
    /// # Arguments
    ///
    /// * `k` - Point at which to evaluate the PMF
    ///
    /// # Returns
    ///
    /// The probability mass at k
    fn pmf(&self, k: F) -> F;

    /// Cumulative distribution function (CDF)
    ///
    /// # Arguments
    ///
    /// * `k` - Point at which to evaluate the CDF
    ///
    /// # Returns
    ///
    /// The cumulative probability up to k
    fn cdf(&self, k: F) -> F;

    /// Support of the distribution (range of possible values)
    fn support(&self) -> (Option<F>, Option<F>) {
        (None, None) // Default: unbounded support
    }

    /// Percent point function (inverse CDF)
    fn ppf(&self, _p: F) -> StatsResult<F> {
        Err(crate::error::StatsError::NotImplementedError(
            "PPF not implemented for this discrete distribution".to_string(),
        ))
    }

    /// Log probability mass function
    fn logpmf(&self, x: F) -> F {
        self.pmf(x).ln()
    }
}

/// Trait for circular distributions (distributions on the unit circle)
pub trait CircularDistribution<F: Float>: Distribution<F> {
    /// Probability density function for circular distributions
    ///
    /// # Arguments
    ///
    /// * `x` - Angle in radians
    ///
    /// # Returns
    ///
    /// The probability density at angle x
    fn pdf(&self, x: F) -> F;

    /// Cumulative distribution function for circular distributions
    ///
    /// # Arguments
    ///
    /// * `x` - Angle in radians
    ///
    /// # Returns
    ///
    /// The cumulative probability up to angle x
    fn cdf(&self, x: F) -> F;

    /// Generate a single random sample
    ///
    /// # Returns
    ///
    /// A single random sample from the distribution
    fn rvs_single(&self) -> StatsResult<F>;

    /// Circular mean (mean direction)
    ///
    /// # Returns
    ///
    /// The mean direction in radians
    fn circular_mean(&self) -> F;

    /// Circular variance
    ///
    /// # Returns
    ///
    /// The circular variance (1 - mean resultant length)
    fn circular_variance(&self) -> F;

    /// Circular standard deviation
    ///
    /// # Returns
    ///
    /// The circular standard deviation in radians
    fn circular_std(&self) -> F;

    /// Mean resultant length
    ///
    /// # Returns
    ///
    /// The mean resultant length (measure of concentration)
    fn mean_resultant_length(&self) -> F;

    /// Concentration parameter
    ///
    /// # Returns
    ///
    /// The concentration parameter of the distribution
    fn concentration(&self) -> F;
}

/// Trait for multivariate distributions
pub trait MultivariateDistribution<F: Float> {
    /// Probability density function for multivariate distributions
    ///
    /// # Arguments
    ///
    /// * `x` - Point at which to evaluate the PDF
    ///
    /// # Returns
    ///
    /// The probability density at x
    fn pdf(&self, x: &Array1<F>) -> F;

    /// Generate random samples from the multivariate distribution
    ///
    /// # Arguments
    ///
    /// * `size` - Number of samples to generate
    ///
    /// # Returns
    ///
    /// A matrix where each row is a sample
    fn rvs(&self, size: usize) -> StatsResult<ndarray::Array2<F>>;

    /// Mean vector of the distribution
    fn mean(&self) -> Array1<F>;

    /// Covariance matrix of the distribution
    fn cov(&self) -> ndarray::Array2<F>;

    /// Dimensionality of the distribution
    fn dim(&self) -> usize;

    /// Log probability density function for multivariate distributions
    fn logpdf(&self, x: &Array1<F>) -> F {
        self.pdf(x).ln()
    }

    /// Generate a single random sample from the multivariate distribution
    fn rvs_single(&self) -> StatsResult<Vec<F>> {
        let samples = self.rvs(1)?;
        Ok(samples.row(0).to_vec())
    }
}

/// Trait for distributions that support fitting to data
pub trait Fittable<F: Float> {
    /// Fit the distribution to observed data
    ///
    /// # Arguments
    ///
    /// * `data` - Observed data points
    ///
    /// # Returns
    ///
    /// A fitted distribution instance
    fn fit(data: &Array1<F>) -> StatsResult<Self>
    where
        Self: Sized;

    /// Maximum likelihood estimation of parameters
    ///
    /// # Arguments
    ///
    /// * `data` - Observed data points
    ///
    /// # Returns
    ///
    /// A tuple of estimated parameters
    fn mle(data: &Array1<F>) -> StatsResult<Vec<F>>;
}

/// Trait for distributions that can be truncated
pub trait Truncatable<F: Float>: Distribution<F> {
    /// Create a truncated version of the distribution
    ///
    /// # Arguments
    ///
    /// * `lower` - Lower bound (None for no lower bound)
    /// * `upper` - Upper bound (None for no upper bound)
    ///
    /// # Returns
    ///
    /// A truncated version of the distribution
    fn truncate(&self, lower: Option<F>, upper: Option<F>)
        -> StatsResult<Box<dyn Distribution<F>>>;
}
