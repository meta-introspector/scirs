//! Financial time series analysis toolkit
//!
//! This module provides specialized functionality for financial time series analysis,
//! including GARCH models, volatility modeling, and technical indicators.

use ndarray::{s, Array1, Array2};
use num_traits::Float;
use std::fmt::Debug;

use crate::error::{Result, TimeSeriesError};

/// GARCH model configuration
#[derive(Debug, Clone)]
pub struct GarchConfig {
    /// GARCH order (p)
    pub p: usize,
    /// ARCH order (q)
    pub q: usize,
    /// Mean model type
    pub mean_model: MeanModel,
    /// Distribution for residuals
    pub distribution: Distribution,
    /// Maximum iterations for optimization
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Use numerical derivatives
    pub use_numerical_derivatives: bool,
}

impl Default for GarchConfig {
    fn default() -> Self {
        Self {
            p: 1,
            q: 1,
            mean_model: MeanModel::Constant,
            distribution: Distribution::Normal,
            max_iterations: 1000,
            tolerance: 1e-6,
            use_numerical_derivatives: false,
        }
    }
}

/// Mean model specification for GARCH
#[derive(Debug, Clone)]
pub enum MeanModel {
    /// Constant mean
    Constant,
    /// Zero mean
    Zero,
    /// AR(p) mean model
    AR {
        /// Autoregressive order
        order: usize,
    },
    /// ARMA(p,q) mean model  
    ARMA {
        /// Autoregressive order
        ar_order: usize,
        /// Moving average order
        ma_order: usize,
    },
}

/// Distribution for GARCH residuals
#[derive(Debug, Clone)]
pub enum Distribution {
    /// Normal distribution
    Normal,
    /// Student's t-distribution
    StudentT,
    /// Skewed Student's t-distribution
    SkewedStudentT,
    /// Generalized Error Distribution
    GED,
}

/// GARCH model results
#[derive(Debug, Clone)]
pub struct GarchResult<F: Float> {
    /// Model parameters
    pub parameters: GarchParameters<F>,
    /// Conditional variance (volatility squared)
    pub conditional_variance: Array1<F>,
    /// Standardized residuals
    pub standardized_residuals: Array1<F>,
    /// Log-likelihood
    pub log_likelihood: F,
    /// Information criteria
    pub aic: F,
    /// Bayesian Information Criterion
    pub bic: F,
    /// Convergence status
    pub converged: bool,
    /// Number of iterations
    pub iterations: usize,
}

/// GARCH model parameters
#[derive(Debug, Clone)]
pub struct GarchParameters<F: Float> {
    /// Mean equation parameters
    pub mean_params: Array1<F>,
    /// GARCH parameters (omega, alpha_i, beta_j)
    pub garch_params: Array1<F>,
    /// Distribution parameters (if applicable)
    pub dist_params: Option<Array1<F>>,
}

/// GARCH model implementation
#[derive(Debug)]
pub struct GarchModel<F: Float + Debug> {
    #[allow(dead_code)]
    config: GarchConfig,
    fitted: bool,
    parameters: Option<GarchParameters<F>>,
    #[allow(dead_code)]
    conditional_variance: Option<Array1<F>>,
}

impl<F: Float + Debug> GarchModel<F> {
    /// Create a new GARCH model
    pub fn new(config: GarchConfig) -> Self {
        Self {
            config,
            fitted: false,
            parameters: None,
            conditional_variance: None,
        }
    }

    /// Create GARCH(1,1) model with default settings
    pub fn garch_11() -> Self {
        Self::new(GarchConfig::default())
    }

    /// Fit the GARCH model to data using simplified method of moments
    pub fn fit(&mut self, data: &Array1<F>) -> Result<GarchResult<F>> {
        if data.len() < 20 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Need at least 20 observations for GARCH estimation".to_string(),
                required: 20,
                actual: data.len(),
            });
        }

        // For GARCH(1,1), use simplified method of moments estimation
        if self.config.p == 1 && self.config.q == 1 {
            self.fit_garch_11_mom(data)
        } else {
            Err(TimeSeriesError::NotImplemented(
                "Only GARCH(1,1) with method of moments is currently implemented. \
                 Full MLE estimation will be available in the next release."
                    .to_string(),
            ))
        }
    }

    /// Fit GARCH(1,1) using method of moments
    fn fit_garch_11_mom(&mut self, data: &Array1<F>) -> Result<GarchResult<F>> {
        // Calculate returns if data represents prices
        let returns = if data.iter().all(|&x| x > F::zero()) {
            // Assume prices, calculate log returns
            let mut ret = Array1::zeros(data.len() - 1);
            for i in 1..data.len() {
                ret[i - 1] = (data[i] / data[i - 1]).ln();
            }
            ret
        } else {
            // Assume already returns
            data.clone()
        };

        let n = returns.len();
        let n_f = F::from(n).unwrap();

        // Calculate sample moments
        let mean = returns.sum() / n_f;
        let centered_returns: Array1<F> = returns.mapv(|r| r - mean);

        // Sample variance
        let sample_var = centered_returns.mapv(|r| r.powi(2)).sum() / (n_f - F::one());

        // Sample skewness and kurtosis for moment matching
        let _sample_skew = centered_returns.mapv(|r| r.powi(3)).sum()
            / ((n_f - F::one()) * sample_var.powf(F::from(1.5).unwrap()));
        let sample_kurt =
            centered_returns.mapv(|r| r.powi(4)).sum() / ((n_f - F::one()) * sample_var.powi(2));

        // Method of moments for GARCH(1,1)
        // Using theoretical moments of GARCH(1,1) process

        // For GARCH(1,1): E[r^2] = omega / (1 - alpha - beta)
        // E[r^4] / (E[r^2])^2 = 3(1 - (alpha + beta)^2) / (1 - (alpha + beta)^2 - 2*alpha^2)

        // Simplified parameter estimation
        let alpha_beta_sum = F::one() - F::from(3.0).unwrap() / sample_kurt;
        let alpha_beta_sum = alpha_beta_sum
            .max(F::from(0.1).unwrap())
            .min(F::from(0.99).unwrap());

        // Split alpha and beta based on typical GARCH patterns
        let alpha = alpha_beta_sum * F::from(0.1).unwrap(); // Typically alpha < beta
        let beta = alpha_beta_sum - alpha;
        let omega = sample_var * (F::one() - alpha - beta);

        // Ensure parameters are positive and sum to less than 1
        let omega = omega.max(F::from(1e-6).unwrap());
        let alpha = alpha.max(F::from(0.01).unwrap()).min(F::from(0.3).unwrap());
        let beta = beta.max(F::from(0.01).unwrap()).min(F::from(0.95).unwrap());

        // Adjust if sum exceeds 1
        let sum_ab = alpha + beta;
        let (alpha, beta) = if sum_ab >= F::one() {
            let scale = F::from(0.99).unwrap() / sum_ab;
            (alpha * scale, beta * scale)
        } else {
            (alpha, beta)
        };

        // Calculate conditional variance recursively
        let mut conditional_variance = Array1::zeros(n);
        conditional_variance[0] = sample_var; // Initialize with unconditional variance

        for i in 1..n {
            conditional_variance[i] = omega
                + alpha * centered_returns[i - 1].powi(2)
                + beta * conditional_variance[i - 1];
        }

        // Calculate standardized residuals
        let standardized_residuals: Array1<F> = centered_returns
            .iter()
            .zip(conditional_variance.iter())
            .map(|(&r, &v)| r / v.sqrt())
            .collect();

        // Calculate log-likelihood (simplified)
        let mut log_likelihood = F::zero();
        for i in 0..n {
            let variance = conditional_variance[i];
            if variance > F::zero() {
                log_likelihood = log_likelihood
                    - F::from(0.5).unwrap()
                        * (variance.ln() + centered_returns[i].powi(2) / variance);
            }
        }

        // Information criteria
        let k = F::from(3).unwrap(); // Number of parameters (omega, alpha, beta)
        let aic = -F::from(2.0).unwrap() * log_likelihood + F::from(2.0).unwrap() * k;
        let bic = -F::from(2.0).unwrap() * log_likelihood + k * n_f.ln();

        // Create parameter structure
        let mean_params = Array1::from_vec(vec![mean]);
        let garch_params = Array1::from_vec(vec![omega, alpha, beta]);

        let parameters = GarchParameters {
            mean_params,
            garch_params,
            dist_params: None,
        };

        // Update model state
        self.fitted = true;
        self.parameters = Some(parameters.clone());
        self.conditional_variance = Some(conditional_variance.clone());

        Ok(GarchResult {
            parameters,
            conditional_variance,
            standardized_residuals,
            log_likelihood,
            aic,
            bic,
            converged: true,
            iterations: 1, // Method of moments is direct
        })
    }

    /// Forecast conditional variance
    pub fn forecast_variance(&self, steps: usize) -> Result<Array1<F>> {
        if !self.fitted {
            return Err(TimeSeriesError::InvalidModel(
                "Model has not been fitted".to_string(),
            ));
        }

        let parameters = self.parameters.as_ref().unwrap();
        let conditional_variance = self.conditional_variance.as_ref().unwrap();

        if parameters.garch_params.len() < 3 {
            return Err(TimeSeriesError::InvalidModel(
                "Invalid GARCH parameters".to_string(),
            ));
        }

        let omega = parameters.garch_params[0];
        let alpha = parameters.garch_params[1];
        let beta = parameters.garch_params[2];

        let mut forecasts = Array1::zeros(steps);

        // Initialize with last conditional variance
        let mut current_variance = conditional_variance[conditional_variance.len() - 1];

        // Calculate unconditional variance for long-term forecast
        let unconditional_variance = omega / (F::one() - alpha - beta);

        for i in 0..steps {
            if i == 0 {
                // One-step ahead forecast
                // Since we don't know the future shock, we use expected value (zero)
                forecasts[i] = omega + beta * current_variance;
            } else {
                // Multi-step ahead forecast converges to unconditional variance
                // h-step ahead variance: omega + (alpha + beta)^(h-1) * (1-step variance - unconditional)
                let decay_factor = (alpha + beta).powf(F::from(i).unwrap());
                forecasts[i] =
                    unconditional_variance + decay_factor * (forecasts[0] - unconditional_variance);
            }
            current_variance = forecasts[i];
        }

        Ok(forecasts)
    }

    /// Get model parameters
    pub fn get_parameters(&self) -> Option<&GarchParameters<F>> {
        self.parameters.as_ref()
    }

    /// Check if model is fitted
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }
}

/// Technical indicators for financial time series
pub mod technical_indicators {
    use super::*;

    /// Simple Moving Average
    pub fn sma<F: Float + Clone>(data: &Array1<F>, window: usize) -> Result<Array1<F>> {
        if window == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "Window size must be positive".to_string(),
            ));
        }

        if data.len() < window {
            return Err(TimeSeriesError::InsufficientData {
                message: "Not enough data for SMA calculation".to_string(),
                required: window,
                actual: data.len(),
            });
        }

        let mut result = Array1::zeros(data.len() - window + 1);

        for i in 0..result.len() {
            let sum = data.slice(s![i..i + window]).sum();
            let window_f = F::from(window).unwrap();
            result[i] = sum / window_f;
        }

        Ok(result)
    }

    /// Exponential Moving Average
    pub fn ema<F: Float + Clone>(data: &Array1<F>, alpha: F) -> Result<Array1<F>> {
        if data.is_empty() {
            return Err(TimeSeriesError::InvalidInput(
                "Data cannot be empty".to_string(),
            ));
        }

        let zero = F::zero();
        let one = F::one();

        if alpha <= zero || alpha > one {
            return Err(TimeSeriesError::InvalidParameter {
                name: "alpha".to_string(),
                message: "Alpha must be between 0 and 1".to_string(),
            });
        }

        let mut result = Array1::zeros(data.len());
        result[0] = data[0];

        let one_minus_alpha = one - alpha;

        for i in 1..data.len() {
            result[i] = alpha * data[i] + one_minus_alpha * result[i - 1];
        }

        Ok(result)
    }

    /// Bollinger Bands
    pub fn bollinger_bands<F: Float + Clone>(
        data: &Array1<F>,
        window: usize,
        num_std: F,
    ) -> Result<(Array1<F>, Array1<F>, Array1<F>)> {
        let sma_values = sma(data, window)?;
        let mut upper = Array1::zeros(sma_values.len());
        let mut lower = Array1::zeros(sma_values.len());

        for i in 0..sma_values.len() {
            let slice = data.slice(s![i..i + window]);
            let mean = sma_values[i];

            // Calculate standard deviation
            let variance = slice
                .mapv(|x| {
                    let diff = x - mean;
                    diff * diff
                })
                .sum()
                / F::from(window).unwrap();

            let std_dev = variance.sqrt();

            upper[i] = mean + num_std * std_dev;
            lower[i] = mean - num_std * std_dev;
        }

        Ok((upper, sma_values, lower))
    }

    /// Relative Strength Index (RSI)
    pub fn rsi<F: Float + Clone>(data: &Array1<F>, period: usize) -> Result<Array1<F>> {
        if period == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "Period must be positive".to_string(),
            ));
        }

        if data.len() < period + 1 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Not enough data for RSI calculation".to_string(),
                required: period + 1,
                actual: data.len(),
            });
        }

        // Calculate price changes
        let mut changes = Array1::zeros(data.len() - 1);
        for i in 0..changes.len() {
            changes[i] = data[i + 1] - data[i];
        }

        // Separate gains and losses
        let gains = changes.mapv(|x| if x > F::zero() { x } else { F::zero() });
        let losses = changes.mapv(|x| if x < F::zero() { -x } else { F::zero() });

        // Calculate average gains and losses
        let avg_gain = sma(&gains, period)?;
        let avg_loss = sma(&losses, period)?;

        // Calculate RSI
        let mut rsi = Array1::zeros(avg_gain.len());
        let hundred = F::from(100).unwrap();

        for i in 0..rsi.len() {
            if avg_loss[i] == F::zero() {
                rsi[i] = hundred;
            } else {
                let rs = avg_gain[i] / avg_loss[i];
                rsi[i] = hundred - (hundred / (F::one() + rs));
            }
        }

        Ok(rsi)
    }

    /// MACD (Moving Average Convergence Divergence)
    pub fn macd<F: Float + Clone>(
        data: &Array1<F>,
        fast_period: usize,
        slow_period: usize,
        signal_period: usize,
    ) -> Result<(Array1<F>, Array1<F>, Array1<F>)> {
        if fast_period >= slow_period {
            return Err(TimeSeriesError::InvalidInput(
                "Fast period must be less than slow period".to_string(),
            ));
        }

        let fast_alpha = F::from(2.0).unwrap() / F::from(fast_period + 1).unwrap();
        let slow_alpha = F::from(2.0).unwrap() / F::from(slow_period + 1).unwrap();
        let signal_alpha = F::from(2.0).unwrap() / F::from(signal_period + 1).unwrap();

        let fast_ema = ema(data, fast_alpha)?;
        let slow_ema = ema(data, slow_alpha)?;

        // Calculate MACD line
        let macd_line = &fast_ema - &slow_ema;

        // Calculate signal line
        let signal_line = ema(&macd_line, signal_alpha)?;

        // Calculate histogram
        let histogram = &macd_line - &signal_line;

        Ok((macd_line, signal_line, histogram))
    }

    /// Stochastic Oscillator
    pub fn stochastic<F: Float + Clone>(
        high: &Array1<F>,
        low: &Array1<F>,
        close: &Array1<F>,
        k_period: usize,
        d_period: usize,
    ) -> Result<(Array1<F>, Array1<F>)> {
        if high.len() != low.len() || low.len() != close.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: high.len(),
                actual: close.len(),
            });
        }

        if high.len() < k_period {
            return Err(TimeSeriesError::InsufficientData {
                message: "Not enough data for stochastic calculation".to_string(),
                required: k_period,
                actual: high.len(),
            });
        }

        let mut k_percent = Array1::zeros(high.len() - k_period + 1);
        let hundred = F::from(100).unwrap();

        for i in 0..k_percent.len() {
            let period_high = high
                .slice(s![i..i + k_period])
                .iter()
                .cloned()
                .fold(F::neg_infinity(), F::max);
            let period_low = low
                .slice(s![i..i + k_period])
                .iter()
                .cloned()
                .fold(F::infinity(), F::min);

            let current_close = close[i + k_period - 1];

            if period_high == period_low {
                k_percent[i] = hundred;
            } else {
                k_percent[i] = hundred * (current_close - period_low) / (period_high - period_low);
            }
        }

        let d_percent = sma(&k_percent, d_period)?;

        Ok((k_percent, d_percent))
    }

    /// Average True Range (ATR)
    pub fn atr<F: Float + Clone>(
        high: &Array1<F>,
        low: &Array1<F>,
        close: &Array1<F>,
        period: usize,
    ) -> Result<Array1<F>> {
        if high.len() != low.len() || low.len() != close.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: high.len(),
                actual: close.len(),
            });
        }

        if high.len() < period + 1 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Not enough data for ATR calculation".to_string(),
                required: period + 1,
                actual: high.len(),
            });
        }

        let mut true_ranges = Array1::zeros(high.len() - 1);

        for i in 1..high.len() {
            let tr1 = high[i] - low[i];
            let tr2 = (high[i] - close[i - 1]).abs();
            let tr3 = (low[i] - close[i - 1]).abs();

            true_ranges[i - 1] = tr1.max(tr2).max(tr3);
        }

        sma(&true_ranges, period)
    }

    /// Williams %R oscillator
    pub fn williams_r<F: Float + Clone>(
        high: &Array1<F>,
        low: &Array1<F>,
        close: &Array1<F>,
        period: usize,
    ) -> Result<Array1<F>> {
        if high.len() != low.len() || low.len() != close.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: high.len(),
                actual: close.len(),
            });
        }

        if high.len() < period {
            return Err(TimeSeriesError::InsufficientData {
                message: "Not enough data for Williams %R calculation".to_string(),
                required: period,
                actual: high.len(),
            });
        }

        let mut williams_r = Array1::zeros(high.len() - period + 1);
        let hundred = F::from(100).unwrap();

        for i in 0..williams_r.len() {
            let period_high = high
                .slice(s![i..i + period])
                .iter()
                .cloned()
                .fold(F::neg_infinity(), F::max);
            let period_low = low
                .slice(s![i..i + period])
                .iter()
                .cloned()
                .fold(F::infinity(), F::min);

            let current_close = close[i + period - 1];

            if period_high == period_low {
                williams_r[i] = F::zero();
            } else {
                williams_r[i] =
                    ((period_high - current_close) / (period_high - period_low)) * (-hundred);
            }
        }

        Ok(williams_r)
    }

    /// Commodity Channel Index (CCI)
    pub fn cci<F: Float + Clone>(
        high: &Array1<F>,
        low: &Array1<F>,
        close: &Array1<F>,
        period: usize,
    ) -> Result<Array1<F>> {
        if high.len() != low.len() || low.len() != close.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: high.len(),
                actual: close.len(),
            });
        }

        if high.len() < period {
            return Err(TimeSeriesError::InsufficientData {
                message: "Not enough data for CCI calculation".to_string(),
                required: period,
                actual: high.len(),
            });
        }

        // Calculate Typical Price
        let mut typical_price = Array1::zeros(high.len());
        let three = F::from(3).unwrap();

        for i in 0..high.len() {
            typical_price[i] = (high[i] + low[i] + close[i]) / three;
        }

        // Calculate SMA of typical price
        let sma_tp = sma(&typical_price, period)?;

        // Calculate mean deviation
        let mut cci = Array1::zeros(sma_tp.len());
        let constant = F::from(0.015).unwrap();

        for i in 0..cci.len() {
            let slice = typical_price.slice(s![i..i + period]);
            let mean = sma_tp[i];

            let mean_deviation = slice.mapv(|x| (x - mean).abs()).sum() / F::from(period).unwrap();

            if mean_deviation != F::zero() {
                cci[i] = (typical_price[i + period - 1] - mean) / (constant * mean_deviation);
            }
        }

        Ok(cci)
    }

    /// On-Balance Volume (OBV)
    pub fn obv<F: Float + Clone>(close: &Array1<F>, volume: &Array1<F>) -> Result<Array1<F>> {
        if close.len() != volume.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: close.len(),
                actual: volume.len(),
            });
        }

        if close.len() < 2 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Need at least 2 data points for OBV".to_string(),
                required: 2,
                actual: close.len(),
            });
        }

        let mut obv = Array1::zeros(close.len());
        obv[0] = volume[0];

        for i in 1..close.len() {
            if close[i] > close[i - 1] {
                obv[i] = obv[i - 1] + volume[i];
            } else if close[i] < close[i - 1] {
                obv[i] = obv[i - 1] - volume[i];
            } else {
                obv[i] = obv[i - 1];
            }
        }

        Ok(obv)
    }

    /// Money Flow Index (MFI)
    pub fn mfi<F: Float + Clone>(
        high: &Array1<F>,
        low: &Array1<F>,
        close: &Array1<F>,
        volume: &Array1<F>,
        period: usize,
    ) -> Result<Array1<F>> {
        if high.len() != low.len() || low.len() != close.len() || close.len() != volume.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: high.len(),
                actual: volume.len(),
            });
        }

        if high.len() < period + 1 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Not enough data for MFI calculation".to_string(),
                required: period + 1,
                actual: high.len(),
            });
        }

        // Calculate typical price and raw money flow
        let mut typical_price = Array1::zeros(high.len());
        let mut raw_money_flow = Array1::zeros(high.len());
        let three = F::from(3).unwrap();

        for i in 0..high.len() {
            typical_price[i] = (high[i] + low[i] + close[i]) / three;
            raw_money_flow[i] = typical_price[i] * volume[i];
        }

        let mut mfi = Array1::zeros(high.len() - period);
        let hundred = F::from(100).unwrap();

        for i in 0..mfi.len() {
            let mut positive_flow = F::zero();
            let mut negative_flow = F::zero();

            for j in 1..=period {
                let current_idx = i + j;
                let prev_idx = i + j - 1;

                if typical_price[current_idx] > typical_price[prev_idx] {
                    positive_flow = positive_flow + raw_money_flow[current_idx];
                } else if typical_price[current_idx] < typical_price[prev_idx] {
                    negative_flow = negative_flow + raw_money_flow[current_idx];
                }
            }

            if negative_flow == F::zero() {
                mfi[i] = hundred;
            } else {
                let money_ratio = positive_flow / negative_flow;
                mfi[i] = hundred - (hundred / (F::one() + money_ratio));
            }
        }

        Ok(mfi)
    }

    /// Parabolic SAR (Stop and Reverse)
    pub fn parabolic_sar<F: Float + Clone>(
        high: &Array1<F>,
        low: &Array1<F>,
        acceleration: F,
        maximum: F,
    ) -> Result<Array1<F>> {
        if high.len() != low.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: high.len(),
                actual: low.len(),
            });
        }

        if high.len() < 2 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Need at least 2 data points for Parabolic SAR".to_string(),
                required: 2,
                actual: high.len(),
            });
        }

        let mut sar = Array1::zeros(high.len());
        let mut ep = high[0]; // Extreme Point
        let mut af = acceleration; // Acceleration Factor
        let mut up_trend = true;

        sar[0] = low[0];

        for i in 1..high.len() {
            if up_trend {
                sar[i] = sar[i - 1] + af * (ep - sar[i - 1]);

                if high[i] > ep {
                    ep = high[i];
                    af = (af + acceleration).min(maximum);
                }

                if low[i] <= sar[i] {
                    up_trend = false;
                    sar[i] = ep;
                    ep = low[i];
                    af = acceleration;
                }
            } else {
                sar[i] = sar[i - 1] + af * (ep - sar[i - 1]);

                if low[i] < ep {
                    ep = low[i];
                    af = (af + acceleration).min(maximum);
                }

                if high[i] >= sar[i] {
                    up_trend = true;
                    sar[i] = ep;
                    ep = high[i];
                    af = acceleration;
                }
            }
        }

        Ok(sar)
    }

    /// Aroon indicator
    pub fn aroon<F: Float + Clone>(
        high: &Array1<F>,
        low: &Array1<F>,
        period: usize,
    ) -> Result<(Array1<F>, Array1<F>)> {
        if high.len() != low.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: high.len(),
                actual: low.len(),
            });
        }

        if high.len() < period {
            return Err(TimeSeriesError::InsufficientData {
                message: "Not enough data for Aroon calculation".to_string(),
                required: period,
                actual: high.len(),
            });
        }

        let mut aroon_up = Array1::zeros(high.len() - period + 1);
        let mut aroon_down = Array1::zeros(high.len() - period + 1);
        let hundred = F::from(100).unwrap();
        let period_f = F::from(period).unwrap();

        for i in 0..aroon_up.len() {
            let mut highest_idx = 0;
            let mut lowest_idx = 0;
            let mut highest_val = high[i];
            let mut lowest_val = low[i];

            for j in 1..period {
                if high[i + j] > highest_val {
                    highest_val = high[i + j];
                    highest_idx = j;
                }
                if low[i + j] < lowest_val {
                    lowest_val = low[i + j];
                    lowest_idx = j;
                }
            }

            aroon_up[i] =
                hundred * (period_f - F::from(period - 1 - highest_idx).unwrap()) / period_f;
            aroon_down[i] =
                hundred * (period_f - F::from(period - 1 - lowest_idx).unwrap()) / period_f;
        }

        Ok((aroon_up, aroon_down))
    }

    /// Directional Movement Index (DMI) and Average Directional Index (ADX)
    pub fn adx<F: Float + Clone>(
        high: &Array1<F>,
        low: &Array1<F>,
        close: &Array1<F>,
        period: usize,
    ) -> Result<(Array1<F>, Array1<F>, Array1<F>)> {
        if high.len() != low.len() || low.len() != close.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: high.len(),
                actual: close.len(),
            });
        }

        if high.len() < period + 1 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Not enough data for ADX calculation".to_string(),
                required: period + 1,
                actual: high.len(),
            });
        }

        let atr_values = atr(high, low, close, period)?;

        // Calculate Directional Movement
        let mut dm_plus = Array1::zeros(high.len() - 1);
        let mut dm_minus = Array1::zeros(high.len() - 1);

        for i in 1..high.len() {
            let up_move = high[i] - high[i - 1];
            let down_move = low[i - 1] - low[i];

            if up_move > down_move && up_move > F::zero() {
                dm_plus[i - 1] = up_move;
            }
            if down_move > up_move && down_move > F::zero() {
                dm_minus[i - 1] = down_move;
            }
        }

        // Calculate smoothed DM values
        let smoothed_dm_plus = ema(
            &dm_plus,
            F::from(2.0).unwrap() / F::from(period + 1).unwrap(),
        )?;
        let smoothed_dm_minus = ema(
            &dm_minus,
            F::from(2.0).unwrap() / F::from(period + 1).unwrap(),
        )?;

        // Calculate DI+ and DI-
        let mut di_plus = Array1::zeros(atr_values.len());
        let mut di_minus = Array1::zeros(atr_values.len());
        let hundred = F::from(100).unwrap();

        for i in 0..di_plus.len() {
            if atr_values[i] != F::zero() {
                di_plus[i] = hundred * smoothed_dm_plus[i] / atr_values[i];
                di_minus[i] = hundred * smoothed_dm_minus[i] / atr_values[i];
            }
        }

        // Calculate DX and ADX
        let mut dx = Array1::zeros(di_plus.len());
        for i in 0..dx.len() {
            let di_sum = di_plus[i] + di_minus[i];
            if di_sum != F::zero() {
                dx[i] = hundred * (di_plus[i] - di_minus[i]).abs() / di_sum;
            }
        }

        let adx_values = ema(&dx, F::from(2.0).unwrap() / F::from(period + 1).unwrap())?;

        Ok((di_plus, di_minus, adx_values))
    }
}

/// Volatility modeling functions
pub mod volatility {
    use super::*;

    /// Calculate realized volatility from high-frequency returns
    pub fn realized_volatility<F: Float>(returns: &Array1<F>) -> F {
        returns.mapv(|x| x * x).sum()
    }

    /// Garman-Klass volatility estimator
    pub fn garman_klass_volatility<F: Float + Clone>(
        high: &Array1<F>,
        low: &Array1<F>,
        close: &Array1<F>,
        open: &Array1<F>,
    ) -> Result<Array1<F>> {
        if high.len() != low.len() || low.len() != close.len() || close.len() != open.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: high.len(),
                actual: open.len(),
            });
        }

        let mut gk_vol = Array1::zeros(high.len());
        let half = F::from(0.5).unwrap();
        let ln_2_minus_1 = F::from(2.0 * (2.0_f64).ln() - 1.0).unwrap();

        for i in 0..gk_vol.len() {
            let log_hl = (high[i] / low[i]).ln();
            let log_co = (close[i] / open[i]).ln();

            gk_vol[i] = half * log_hl * log_hl - ln_2_minus_1 * log_co * log_co;
        }

        Ok(gk_vol)
    }

    /// Parkinson volatility estimator
    pub fn parkinson_volatility<F: Float + Clone>(
        high: &Array1<F>,
        low: &Array1<F>,
    ) -> Result<Array1<F>> {
        if high.len() != low.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: high.len(),
                actual: low.len(),
            });
        }

        let mut park_vol = Array1::zeros(high.len());
        let four_ln_2 = F::from(4.0 * (2.0_f64).ln()).unwrap();

        for i in 0..park_vol.len() {
            let log_hl = (high[i] / low[i]).ln();
            park_vol[i] = log_hl * log_hl / four_ln_2;
        }

        Ok(park_vol)
    }

    /// Rogers-Satchell volatility estimator
    pub fn rogers_satchell_volatility<F: Float + Clone>(
        high: &Array1<F>,
        low: &Array1<F>,
        close: &Array1<F>,
        open: &Array1<F>,
    ) -> Result<Array1<F>> {
        if high.len() != low.len() || low.len() != close.len() || close.len() != open.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: high.len(),
                actual: open.len(),
            });
        }

        let mut rs_vol = Array1::zeros(high.len());

        for i in 0..rs_vol.len() {
            let log_ho = (high[i] / open[i]).ln();
            let log_co = (close[i] / open[i]).ln();
            let log_lo = (low[i] / open[i]).ln();

            rs_vol[i] = log_ho * log_co + log_lo * log_co;
        }

        Ok(rs_vol)
    }

    /// Yang-Zhang volatility estimator
    pub fn yang_zhang_volatility<F: Float + Clone>(
        high: &Array1<F>,
        low: &Array1<F>,
        close: &Array1<F>,
        open: &Array1<F>,
        k: F,
    ) -> Result<Array1<F>> {
        if high.len() != low.len() || low.len() != close.len() || close.len() != open.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: high.len(),
                actual: open.len(),
            });
        }

        if high.len() < 2 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Need at least 2 data points for Yang-Zhang volatility".to_string(),
                required: 2,
                actual: high.len(),
            });
        }

        let mut yz_vol = Array1::zeros(high.len() - 1);

        for i in 1..high.len() {
            // Overnight return
            let overnight = (open[i] / close[i - 1]).ln();

            // Open-to-close return
            let open_close = (close[i] / open[i]).ln();

            // Rogers-Satchell component
            let log_ho = (high[i] / open[i]).ln();
            let log_co = (close[i] / open[i]).ln();
            let log_lo = (low[i] / open[i]).ln();
            let rs = log_ho * log_co + log_lo * log_co;

            yz_vol[i - 1] = overnight * overnight + k * open_close * open_close + rs;
        }

        Ok(yz_vol)
    }

    /// GARCH(1,1) volatility estimation using simple method of moments
    pub fn garch_volatility_estimate<F: Float + Clone>(
        returns: &Array1<F>,
        window: usize,
    ) -> Result<Array1<F>> {
        if returns.len() < window + 1 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Not enough data for GARCH volatility estimation".to_string(),
                required: window + 1,
                actual: returns.len(),
            });
        }

        let mut volatilities = Array1::zeros(returns.len() - window + 1);

        // Simple GARCH(1,1) parameters (typical values)
        let omega = F::from(0.000001).unwrap();
        let alpha = F::from(0.1).unwrap();
        let beta = F::from(0.85).unwrap();

        for i in 0..volatilities.len() {
            let window_returns = returns.slice(s![i..i + window]);

            // Initialize with sample variance
            let mean = window_returns.sum() / F::from(window).unwrap();
            let mut variance =
                window_returns.mapv(|x| (x - mean).powi(2)).sum() / F::from(window - 1).unwrap();

            // Apply GARCH updating for last few observations
            for j in 1..std::cmp::min(window, 10) {
                let return_sq = window_returns[window - j].powi(2);
                variance = omega + alpha * return_sq + beta * variance;
            }

            volatilities[i] = variance.sqrt();
        }

        Ok(volatilities)
    }

    /// Exponentially Weighted Moving Average (EWMA) volatility
    pub fn ewma_volatility<F: Float + Clone>(returns: &Array1<F>, lambda: F) -> Result<Array1<F>> {
        if returns.is_empty() {
            return Err(TimeSeriesError::InvalidInput(
                "Returns cannot be empty".to_string(),
            ));
        }

        if lambda <= F::zero() || lambda >= F::one() {
            return Err(TimeSeriesError::InvalidParameter {
                name: "lambda".to_string(),
                message: "Lambda must be between 0 and 1 (exclusive)".to_string(),
            });
        }

        let mut ewma_var = Array1::zeros(returns.len());

        // Initialize with first squared return
        ewma_var[0] = returns[0].powi(2);

        let one_minus_lambda = F::one() - lambda;

        for i in 1..returns.len() {
            ewma_var[i] = lambda * ewma_var[i - 1] + one_minus_lambda * returns[i].powi(2);
        }

        Ok(ewma_var.mapv(|x| x.sqrt()))
    }

    /// Range-based volatility using high-low range
    pub fn range_volatility<F: Float + Clone>(
        high: &Array1<F>,
        low: &Array1<F>,
        period: usize,
    ) -> Result<Array1<F>> {
        if high.len() != low.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: high.len(),
                actual: low.len(),
            });
        }

        if high.len() < period {
            return Err(TimeSeriesError::InsufficientData {
                message: "Not enough data for range volatility calculation".to_string(),
                required: period,
                actual: high.len(),
            });
        }

        let mut range_vol = Array1::zeros(high.len() - period + 1);
        let scaling_factor = F::from(1.0 / (4.0 * (2.0_f64).ln())).unwrap();

        for i in 0..range_vol.len() {
            let mut sum_log_range_sq = F::zero();

            for j in 0..period {
                let log_range = (high[i + j] / low[i + j]).ln();
                sum_log_range_sq = sum_log_range_sq + log_range.powi(2);
            }

            range_vol[i] = (scaling_factor * sum_log_range_sq / F::from(period).unwrap()).sqrt();
        }

        Ok(range_vol)
    }

    /// Intraday volatility estimation using tick data concept
    pub fn intraday_volatility<F: Float + Clone>(
        prices: &Array1<F>,
        sampling_frequency: usize,
    ) -> Result<F> {
        if prices.len() < sampling_frequency + 1 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Not enough data for intraday volatility calculation".to_string(),
                required: sampling_frequency + 1,
                actual: prices.len(),
            });
        }

        let mut squared_returns = F::zero();
        let mut count = 0;

        for i in sampling_frequency..prices.len() {
            let log_return = (prices[i] / prices[i - sampling_frequency]).ln();
            squared_returns = squared_returns + log_return.powi(2);
            count += 1;
        }

        if count == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "No valid returns calculated".to_string(),
            ));
        }

        Ok((squared_returns / F::from(count).unwrap()).sqrt())
    }
}

/// Risk management utilities
pub mod risk {
    use super::*;

    /// Calculate Value at Risk (VaR) using historical simulation
    pub fn var_historical<F: Float + Clone>(returns: &Array1<F>, confidence: f64) -> Result<F> {
        if returns.is_empty() {
            return Err(TimeSeriesError::InvalidInput(
                "Returns cannot be empty".to_string(),
            ));
        }

        if confidence <= 0.0 || confidence >= 1.0 {
            return Err(TimeSeriesError::InvalidParameter {
                name: "confidence".to_string(),
                message: "Confidence must be between 0 and 1".to_string(),
            });
        }

        let mut sorted_returns = returns.to_vec();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let index = ((1.0 - confidence) * sorted_returns.len() as f64) as usize;
        let index = index.min(sorted_returns.len() - 1);

        Ok(sorted_returns[index])
    }

    /// Calculate Expected Shortfall (Conditional VaR)
    pub fn expected_shortfall<F: Float + Clone + std::iter::Sum>(
        returns: &Array1<F>,
        confidence: f64,
    ) -> Result<F> {
        let var = var_historical(returns, confidence)?;

        let tail_returns: Vec<F> = returns.iter().filter(|&&x| x <= var).cloned().collect();

        if tail_returns.is_empty() {
            return Ok(var);
        }

        let sum = tail_returns.iter().fold(F::zero(), |acc, &x| acc + x);
        Ok(sum / F::from(tail_returns.len()).unwrap())
    }

    /// Calculate maximum drawdown
    pub fn max_drawdown<F: Float + Clone>(prices: &Array1<F>) -> Result<F> {
        if prices.is_empty() {
            return Err(TimeSeriesError::InvalidInput(
                "Prices cannot be empty".to_string(),
            ));
        }

        let mut max_price = prices[0];
        let mut max_dd = F::zero();

        for &price in prices.iter() {
            if price > max_price {
                max_price = price;
            }

            let drawdown = (max_price - price) / max_price;
            if drawdown > max_dd {
                max_dd = drawdown;
            }
        }

        Ok(max_dd)
    }

    /// Calmar ratio (annual return / maximum drawdown)
    pub fn calmar_ratio<F: Float + Clone>(
        returns: &Array1<F>,
        prices: &Array1<F>,
        periods_per_year: usize,
    ) -> Result<F> {
        if returns.is_empty() || prices.is_empty() {
            return Err(TimeSeriesError::InvalidInput(
                "Returns and prices cannot be empty".to_string(),
            ));
        }

        // Calculate annualized return
        let total_return = (prices[prices.len() - 1] / prices[0]) - F::one();
        let years = F::from(returns.len()).unwrap() / F::from(periods_per_year).unwrap();
        let annualized_return = (F::one() + total_return).powf(F::one() / years) - F::one();

        // Calculate maximum drawdown
        let mdd = max_drawdown(prices)?;

        if mdd == F::zero() {
            Ok(F::infinity())
        } else {
            Ok(annualized_return / mdd)
        }
    }

    /// Sortino ratio (excess return / downside deviation)
    pub fn sortino_ratio<F: Float + Clone>(
        returns: &Array1<F>,
        risk_free_rate: F,
        periods_per_year: usize,
    ) -> Result<F> {
        if returns.is_empty() {
            return Err(TimeSeriesError::InvalidInput(
                "Returns cannot be empty".to_string(),
            ));
        }

        // Calculate excess returns
        let annualized_rf = risk_free_rate / F::from(periods_per_year).unwrap();
        let excess_returns: Array1<F> = returns.mapv(|r| r - annualized_rf);

        // Calculate mean excess return
        let mean_excess = excess_returns.sum() / F::from(returns.len()).unwrap();

        // Calculate downside deviation (only negative excess returns)
        let downside_returns: Vec<F> = excess_returns
            .iter()
            .filter(|&&r| r < F::zero())
            .cloned()
            .collect();

        if downside_returns.is_empty() {
            return Ok(F::infinity());
        }

        let downside_variance = downside_returns
            .iter()
            .map(|&r| r.powi(2))
            .fold(F::zero(), |acc, x| acc + x)
            / F::from(downside_returns.len()).unwrap();

        let downside_deviation = downside_variance.sqrt();

        if downside_deviation == F::zero() {
            Ok(F::infinity())
        } else {
            // Annualize the ratio
            let annualized_excess = mean_excess * F::from(periods_per_year).unwrap();
            let annualized_downside =
                downside_deviation * F::from(periods_per_year).unwrap().sqrt();
            Ok(annualized_excess / annualized_downside)
        }
    }

    /// Sharpe ratio (excess return / volatility)
    pub fn sharpe_ratio<F: Float + Clone>(
        returns: &Array1<F>,
        risk_free_rate: F,
        periods_per_year: usize,
    ) -> Result<F> {
        if returns.is_empty() {
            return Err(TimeSeriesError::InvalidInput(
                "Returns cannot be empty".to_string(),
            ));
        }

        // Calculate excess returns
        let annualized_rf = risk_free_rate / F::from(periods_per_year).unwrap();
        let excess_returns: Array1<F> = returns.mapv(|r| r - annualized_rf);

        // Calculate mean excess return
        let mean_excess = excess_returns.sum() / F::from(returns.len()).unwrap();

        // Calculate standard deviation of excess returns
        let variance = excess_returns.mapv(|r| (r - mean_excess).powi(2)).sum()
            / F::from(returns.len() - 1).unwrap();

        let std_dev = variance.sqrt();

        if std_dev == F::zero() {
            Ok(F::infinity())
        } else {
            // Annualize the ratio
            let annualized_excess = mean_excess * F::from(periods_per_year).unwrap();
            let annualized_std = std_dev * F::from(periods_per_year).unwrap().sqrt();
            Ok(annualized_excess / annualized_std)
        }
    }

    /// Information ratio (active return / tracking error)
    pub fn information_ratio<F: Float + Clone>(
        portfolio_returns: &Array1<F>,
        benchmark_returns: &Array1<F>,
    ) -> Result<F> {
        if portfolio_returns.len() != benchmark_returns.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: portfolio_returns.len(),
                actual: benchmark_returns.len(),
            });
        }

        if portfolio_returns.is_empty() {
            return Err(TimeSeriesError::InvalidInput(
                "Returns cannot be empty".to_string(),
            ));
        }

        // Calculate active returns (portfolio - benchmark)
        let active_returns: Array1<F> = portfolio_returns
            .iter()
            .zip(benchmark_returns.iter())
            .map(|(&p, &b)| p - b)
            .collect();

        // Calculate mean active return
        let mean_active = active_returns.sum() / F::from(active_returns.len()).unwrap();

        // Calculate tracking error (standard deviation of active returns)
        let variance = active_returns.mapv(|r| (r - mean_active).powi(2)).sum()
            / F::from(active_returns.len() - 1).unwrap();

        let tracking_error = variance.sqrt();

        if tracking_error == F::zero() {
            Ok(F::infinity())
        } else {
            Ok(mean_active / tracking_error)
        }
    }

    /// Beta coefficient (systematic risk measure)
    pub fn beta<F: Float + Clone>(
        asset_returns: &Array1<F>,
        market_returns: &Array1<F>,
    ) -> Result<F> {
        if asset_returns.len() != market_returns.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: asset_returns.len(),
                actual: market_returns.len(),
            });
        }

        if asset_returns.len() < 2 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Need at least 2 observations for beta calculation".to_string(),
                required: 2,
                actual: asset_returns.len(),
            });
        }

        // Calculate means
        let asset_mean = asset_returns.sum() / F::from(asset_returns.len()).unwrap();
        let market_mean = market_returns.sum() / F::from(market_returns.len()).unwrap();

        // Calculate covariance and market variance
        let mut covariance = F::zero();
        let mut market_variance = F::zero();

        for i in 0..asset_returns.len() {
            let asset_dev = asset_returns[i] - asset_mean;
            let market_dev = market_returns[i] - market_mean;

            covariance = covariance + asset_dev * market_dev;
            market_variance = market_variance + market_dev.powi(2);
        }

        let n = F::from(asset_returns.len() - 1).unwrap();
        covariance = covariance / n;
        market_variance = market_variance / n;

        if market_variance == F::zero() {
            Err(TimeSeriesError::InvalidInput(
                "Market returns have zero variance".to_string(),
            ))
        } else {
            Ok(covariance / market_variance)
        }
    }

    /// Treynor ratio (excess return / beta)
    pub fn treynor_ratio<F: Float + Clone>(
        returns: &Array1<F>,
        market_returns: &Array1<F>,
        risk_free_rate: F,
        periods_per_year: usize,
    ) -> Result<F> {
        // Calculate portfolio beta
        let portfolio_beta = beta(returns, market_returns)?;

        if portfolio_beta == F::zero() {
            return Ok(F::infinity());
        }

        // Calculate annualized excess return
        let annualized_rf = risk_free_rate / F::from(periods_per_year).unwrap();
        let mean_return = returns.sum() / F::from(returns.len()).unwrap();
        let excess_return = mean_return - annualized_rf;
        let annualized_excess = excess_return * F::from(periods_per_year).unwrap();

        Ok(annualized_excess / portfolio_beta)
    }

    /// Jensen's alpha (risk-adjusted excess return)
    pub fn jensens_alpha<F: Float + Clone>(
        returns: &Array1<F>,
        market_returns: &Array1<F>,
        risk_free_rate: F,
        periods_per_year: usize,
    ) -> Result<F> {
        // Calculate portfolio beta
        let portfolio_beta = beta(returns, market_returns)?;

        // Calculate mean returns
        let annualized_rf = risk_free_rate / F::from(periods_per_year).unwrap();
        let mean_portfolio = returns.sum() / F::from(returns.len()).unwrap();
        let mean_market = market_returns.sum() / F::from(market_returns.len()).unwrap();

        // Calculate alpha using CAPM formula
        // Alpha = Portfolio Return - (Risk Free Rate + Beta * (Market Return - Risk Free Rate))
        let portfolio_excess = mean_portfolio - annualized_rf;
        let market_excess = mean_market - annualized_rf;
        let expected_excess = portfolio_beta * market_excess;

        Ok((portfolio_excess - expected_excess) * F::from(periods_per_year).unwrap())
    }

    /// Omega ratio (probability-weighted gains over losses)
    pub fn omega_ratio<F: Float + Clone>(returns: &Array1<F>, threshold: F) -> Result<F> {
        if returns.is_empty() {
            return Err(TimeSeriesError::InvalidInput(
                "Returns cannot be empty".to_string(),
            ));
        }

        let mut gains = F::zero();
        let mut losses = F::zero();

        for &ret in returns.iter() {
            let excess = ret - threshold;
            if excess > F::zero() {
                gains = gains + excess;
            } else {
                losses = losses - excess; // Make positive
            }
        }

        if losses == F::zero() {
            Ok(F::infinity())
        } else {
            Ok(gains / losses)
        }
    }

    /// Value at Risk using Monte Carlo simulation (simplified)
    pub fn var_monte_carlo<F: Float + Clone>(
        returns: &Array1<F>,
        confidence: f64,
        simulations: usize,
        horizon: usize,
    ) -> Result<F> {
        if returns.is_empty() {
            return Err(TimeSeriesError::InvalidInput(
                "Returns cannot be empty".to_string(),
            ));
        }

        if confidence <= 0.0 || confidence >= 1.0 {
            return Err(TimeSeriesError::InvalidParameter {
                name: "confidence".to_string(),
                message: "Confidence must be between 0 and 1 (exclusive)".to_string(),
            });
        }

        // Calculate mean and standard deviation
        let mean = returns.sum() / F::from(returns.len()).unwrap();
        let variance =
            returns.mapv(|r| (r - mean).powi(2)).sum() / F::from(returns.len() - 1).unwrap();
        let std_dev = variance.sqrt();

        // Simplified Monte Carlo: assume normal distribution
        // In practice, this would use proper random number generation
        let mut simulated_returns = Vec::with_capacity(simulations);

        for i in 0..simulations {
            // Simple pseudo-random generation (Box-Muller transform approximation)
            let u1 = F::from((i as f64 + 1.0) / (simulations as f64 + 1.0)).unwrap();
            let u2 = F::from(0.5).unwrap(); // Simplified

            // Convert to normal distribution (simplified)
            let z = (-F::from(2.0).unwrap() * u1.ln()).sqrt()
                * (F::from(2.0 * std::f64::consts::PI).unwrap() * u2).cos();

            let simulated_return = mean + std_dev * z;

            // Calculate portfolio value change over horizon
            let mut portfolio_return = F::one();
            for _ in 0..horizon {
                portfolio_return = portfolio_return * (F::one() + simulated_return);
            }

            simulated_returns.push(portfolio_return - F::one());
        }

        // Sort returns and find VaR
        simulated_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let var_index = ((1.0 - confidence) * simulations as f64) as usize;
        let var_index = var_index.min(simulations - 1);

        Ok(simulated_returns[var_index])
    }
}

/// Portfolio analysis and optimization utilities
pub mod portfolio {
    use super::*;

    /// Portfolio performance metrics
    #[derive(Debug, Clone)]
    pub struct PortfolioMetrics<F: Float> {
        /// Total return
        pub total_return: F,
        /// Annualized return
        pub annualized_return: F,
        /// Annualized volatility
        pub volatility: F,
        /// Sharpe ratio
        pub sharpe_ratio: F,
        /// Sortino ratio
        pub sortino_ratio: F,
        /// Maximum drawdown
        pub max_drawdown: F,
        /// Calmar ratio
        pub calmar_ratio: F,
        /// Value at Risk (95%)
        pub var_95: F,
        /// Expected Shortfall (95%)
        pub es_95: F,
    }

    /// Portfolio weights and holdings
    #[derive(Debug, Clone)]
    pub struct Portfolio<F: Float> {
        /// Asset weights (should sum to 1.0)
        pub weights: Array1<F>,
        /// Asset names/identifiers
        pub asset_names: Vec<String>,
        /// Rebalancing frequency (days)
        pub rebalance_frequency: Option<usize>,
    }

    impl<F: Float + Clone> Portfolio<F> {
        /// Create a new portfolio
        pub fn new(weights: Array1<F>, asset_names: Vec<String>) -> Result<Self> {
            if weights.len() != asset_names.len() {
                return Err(TimeSeriesError::DimensionMismatch {
                    expected: weights.len(),
                    actual: asset_names.len(),
                });
            }

            let weight_sum = weights.sum();
            let tolerance = F::from(0.01).unwrap();
            if (weight_sum - F::one()).abs() > tolerance {
                return Err(TimeSeriesError::InvalidInput(
                    "Portfolio weights must sum to approximately 1.0".to_string(),
                ));
            }

            Ok(Self {
                weights,
                asset_names,
                rebalance_frequency: None,
            })
        }

        /// Create equally weighted portfolio
        pub fn equal_weight(n_assets: usize, asset_names: Vec<String>) -> Result<Self> {
            if n_assets == 0 {
                return Err(TimeSeriesError::InvalidInput(
                    "Number of assets must be positive".to_string(),
                ));
            }

            let weight = F::one() / F::from(n_assets).unwrap();
            let weights = Array1::from_elem(n_assets, weight);

            Self::new(weights, asset_names)
        }

        /// Get portfolio weight for specific asset
        pub fn get_weight(&self, asset_name: &str) -> Option<F> {
            self.asset_names
                .iter()
                .position(|name| name == asset_name)
                .map(|idx| self.weights[idx])
        }
    }

    /// Calculate portfolio returns from asset returns and weights
    pub fn calculate_portfolio_returns<F: Float + Clone>(
        asset_returns: &Array2<F>, // rows: time, cols: assets
        weights: &Array1<F>,
    ) -> Result<Array1<F>> {
        if asset_returns.ncols() != weights.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: asset_returns.ncols(),
                actual: weights.len(),
            });
        }

        let mut portfolio_returns = Array1::zeros(asset_returns.nrows());

        for t in 0..asset_returns.nrows() {
            let mut return_sum = F::zero();
            for i in 0..weights.len() {
                return_sum = return_sum + weights[i] * asset_returns[[t, i]];
            }
            portfolio_returns[t] = return_sum;
        }

        Ok(portfolio_returns)
    }

    /// Calculate portfolio metrics
    pub fn calculate_portfolio_metrics<F: Float + Clone + std::iter::Sum>(
        returns: &Array1<F>,
        prices: &Array1<F>,
        risk_free_rate: F,
        periods_per_year: usize,
    ) -> Result<PortfolioMetrics<F>> {
        if returns.is_empty() || prices.is_empty() {
            return Err(TimeSeriesError::InvalidInput(
                "Returns and prices cannot be empty".to_string(),
            ));
        }

        // Total return
        let total_return = (prices[prices.len() - 1] / prices[0]) - F::one();

        // Annualized return
        let years = F::from(returns.len()).unwrap() / F::from(periods_per_year).unwrap();
        let annualized_return = (F::one() + total_return).powf(F::one() / years) - F::one();

        // Volatility (annualized)
        let mean_return = returns.sum() / F::from(returns.len()).unwrap();
        let variance =
            returns.mapv(|r| (r - mean_return).powi(2)).sum() / F::from(returns.len() - 1).unwrap();
        let volatility = variance.sqrt() * F::from(periods_per_year).unwrap().sqrt();

        // Risk metrics
        let sharpe = super::risk::sharpe_ratio(returns, risk_free_rate, periods_per_year)?;
        let sortino = super::risk::sortino_ratio(returns, risk_free_rate, periods_per_year)?;
        let max_dd = super::risk::max_drawdown(prices)?;
        let calmar = super::risk::calmar_ratio(returns, prices, periods_per_year)?;
        let var_95 = super::risk::var_historical(returns, 0.95)?;
        let es_95 = super::risk::expected_shortfall(returns, 0.95)?;

        Ok(PortfolioMetrics {
            total_return,
            annualized_return,
            volatility,
            sharpe_ratio: sharpe,
            sortino_ratio: sortino,
            max_drawdown: max_dd,
            calmar_ratio: calmar,
            var_95,
            es_95,
        })
    }

    /// Modern Portfolio Theory: Calculate efficient frontier point
    pub fn calculate_efficient_portfolio<F: Float + Clone>(
        expected_returns: &Array1<F>,
        covariance_matrix: &Array2<F>,
        target_return: F,
    ) -> Result<Array1<F>> {
        let n = expected_returns.len();

        if covariance_matrix.nrows() != n || covariance_matrix.ncols() != n {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: n,
                actual: covariance_matrix.nrows(),
            });
        }

        // This is a simplified implementation
        // In practice, you would use quadratic programming

        // Equal weight as starting point
        let mut weights = Array1::from_elem(n, F::one() / F::from(n).unwrap());

        // Simple iterative adjustment toward target return
        for _ in 0..100 {
            let current_return = weights
                .iter()
                .zip(expected_returns.iter())
                .map(|(&w, &r)| w * r)
                .fold(F::zero(), |acc, x| acc + x);

            let return_diff = target_return - current_return;

            if return_diff.abs() < F::from(1e-6).unwrap() {
                break;
            }

            // Adjust weights toward higher/lower return assets
            for i in 0..n {
                let adjustment = return_diff * F::from(0.01).unwrap();
                if expected_returns[i] > current_return {
                    weights[i] = weights[i] + adjustment;
                } else {
                    weights[i] = weights[i] - adjustment;
                }
                weights[i] = weights[i].max(F::zero());
            }

            // Normalize weights
            let weight_sum = weights.sum();
            if weight_sum > F::zero() {
                weights = weights.mapv(|w| w / weight_sum);
            }
        }

        Ok(weights)
    }

    /// Risk parity portfolio optimization
    pub fn risk_parity_portfolio<F: Float + Clone>(
        covariance_matrix: &Array2<F>,
    ) -> Result<Array1<F>> {
        let n = covariance_matrix.nrows();

        if covariance_matrix.ncols() != n {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: n,
                actual: covariance_matrix.ncols(),
            });
        }

        // Calculate inverse volatility weights as starting approximation
        let mut weights = Array1::zeros(n);

        for i in 0..n {
            let variance = covariance_matrix[[i, i]];
            if variance > F::zero() {
                weights[i] = F::one() / variance.sqrt();
            } else {
                weights[i] = F::one();
            }
        }

        // Normalize weights
        let weight_sum = weights.sum();
        if weight_sum > F::zero() {
            weights = weights.mapv(|w| w / weight_sum);
        } else {
            // Equal weights fallback
            weights = Array1::from_elem(n, F::one() / F::from(n).unwrap());
        }

        Ok(weights)
    }

    /// Minimum variance portfolio
    pub fn minimum_variance_portfolio<F: Float + Clone>(
        covariance_matrix: &Array2<F>,
    ) -> Result<Array1<F>> {
        let n = covariance_matrix.nrows();

        // Simplified implementation: inverse variance weighting
        let mut weights = Array1::zeros(n);

        for i in 0..n {
            let variance = covariance_matrix[[i, i]];
            if variance > F::zero() {
                weights[i] = F::one() / variance;
            } else {
                weights[i] = F::one();
            }
        }

        // Normalize weights
        let weight_sum = weights.sum();
        if weight_sum > F::zero() {
            weights = weights.mapv(|w| w / weight_sum);
        } else {
            return Err(TimeSeriesError::InvalidInput(
                "All assets have zero variance".to_string(),
            ));
        }

        Ok(weights)
    }

    /// Calculate portfolio Value at Risk using parametric method
    pub fn portfolio_var_parametric<F: Float + Clone>(
        portfolio_value: F,
        portfolio_return_mean: F,
        portfolio_return_std: F,
        confidence_level: f64,
        time_horizon: usize,
    ) -> Result<F> {
        if confidence_level <= 0.0 || confidence_level >= 1.0 {
            return Err(TimeSeriesError::InvalidParameter {
                name: "confidence_level".to_string(),
                message: "Confidence level must be between 0 and 1".to_string(),
            });
        }

        // Z-score for confidence level
        let z_score = match confidence_level {
            c if c >= 0.99 => F::from(-2.326).unwrap(), // 99% VaR
            c if c >= 0.95 => F::from(-1.645).unwrap(), // 95% VaR
            c if c >= 0.90 => F::from(-1.282).unwrap(), // 90% VaR
            _ => F::from(-1.0).unwrap(),
        };

        // Scale for time horizon
        let horizon_scaling = F::from(time_horizon).unwrap().sqrt();
        let horizon_mean = portfolio_return_mean * F::from(time_horizon).unwrap();
        let horizon_std = portfolio_return_std * horizon_scaling;

        // Calculate VaR
        let var_return = horizon_mean + z_score * horizon_std;
        let var_amount = portfolio_value * var_return.abs();

        Ok(var_amount)
    }

    /// Calculate correlation matrix from returns
    pub fn calculate_correlation_matrix<F: Float + Clone>(
        returns: &Array2<F>, // rows: time, cols: assets
    ) -> Result<Array2<F>> {
        let n_assets = returns.ncols();
        let n_periods = returns.nrows();

        if n_periods < 2 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Need at least 2 periods for correlation calculation".to_string(),
                required: 2,
                actual: n_periods,
            });
        }

        let mut correlation_matrix = Array2::zeros((n_assets, n_assets));

        // Calculate means
        let means: Array1<F> = (0..n_assets)
            .map(|i| {
                let col = returns.column(i);
                col.sum() / F::from(n_periods).unwrap()
            })
            .collect();

        // Calculate correlation coefficients
        for i in 0..n_assets {
            for j in 0..n_assets {
                if i == j {
                    correlation_matrix[[i, j]] = F::one();
                } else {
                    let col_i = returns.column(i);
                    let col_j = returns.column(j);

                    let mut numerator = F::zero();
                    let mut sum_sq_i = F::zero();
                    let mut sum_sq_j = F::zero();

                    for t in 0..n_periods {
                        let dev_i = col_i[t] - means[i];
                        let dev_j = col_j[t] - means[j];

                        numerator = numerator + dev_i * dev_j;
                        sum_sq_i = sum_sq_i + dev_i * dev_i;
                        sum_sq_j = sum_sq_j + dev_j * dev_j;
                    }

                    let denominator = (sum_sq_i * sum_sq_j).sqrt();
                    if denominator > F::zero() {
                        correlation_matrix[[i, j]] = numerator / denominator;
                    }
                }
            }
        }

        Ok(correlation_matrix)
    }
}

#[cfg(test)]
mod tests {
    use super::risk::*;
    use super::technical_indicators::*;
    use super::volatility::*;
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::s;

    #[test]
    fn test_garch_config_default() {
        let config = GarchConfig::default();
        assert_eq!(config.p, 1);
        assert_eq!(config.q, 1);
        assert!(matches!(config.mean_model, MeanModel::Constant));
        assert!(matches!(config.distribution, Distribution::Normal));
    }

    #[test]
    fn test_garch_model_creation() {
        let model = GarchModel::<f64>::garch_11();
        assert!(!model.is_fitted());
        assert!(model.get_parameters().is_none());
    }

    #[test]
    fn test_sma() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = sma(&data, 3).unwrap();

        assert_eq!(result.len(), 3);
        assert_abs_diff_eq!(result[0], 2.0);
        assert_abs_diff_eq!(result[1], 3.0);
        assert_abs_diff_eq!(result[2], 4.0);
    }

    #[test]
    fn test_ema() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = ema(&data, 0.5).unwrap();

        assert_eq!(result.len(), 5);
        assert_abs_diff_eq!(result[0], 1.0);
        assert_abs_diff_eq!(result[1], 1.5);
    }

    #[test]
    fn test_bollinger_bands() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
        let (upper, middle, lower) = bollinger_bands(&data, 3, 2.0).unwrap();

        assert_eq!(upper.len(), 7);
        assert_eq!(middle.len(), 7);
        assert_eq!(lower.len(), 7);

        // Middle band should be SMA
        let sma_result = sma(&data, 3).unwrap();
        for i in 0..middle.len() {
            assert_abs_diff_eq!(middle[i], sma_result[i]);
        }
    }

    #[test]
    fn test_rsi() {
        let data = Array1::from_vec(vec![
            44.0, 44.25, 44.5, 43.75, 44.5, 44.75, 47.0, 47.25, 46.5, 46.25, 47.75, 47.5, 47.25,
            47.75, 48.75, 48.5, 48.0, 48.25, 48.75, 48.5,
        ]);
        let result = rsi(&data, 14).unwrap();

        assert_eq!(result.len(), 5);
        // RSI should be between 0 and 100
        for &value in result.iter() {
            assert!(value >= 0.0 && value <= 100.0);
        }
    }

    #[test]
    fn test_realized_volatility() {
        let returns = Array1::from_vec(vec![0.01, -0.02, 0.015, -0.01, 0.005]);
        let vol = realized_volatility(&returns);

        let expected = 0.01_f64.powi(2)
            + 0.02_f64.powi(2)
            + 0.015_f64.powi(2)
            + 0.01_f64.powi(2)
            + 0.005_f64.powi(2);
        assert_abs_diff_eq!(vol, expected);
    }

    #[test]
    fn test_var_historical() {
        let returns = Array1::from_vec(vec![-0.05, -0.02, 0.01, 0.03, -0.01, 0.02, -0.03]);
        let var = var_historical(&returns, 0.95).unwrap();

        // At 95% confidence, VaR should be the 5th percentile (worst return)
        assert_abs_diff_eq!(var, -0.05);
    }

    #[test]
    fn test_max_drawdown() {
        let prices = Array1::from_vec(vec![100.0, 110.0, 105.0, 120.0, 90.0, 95.0]);
        let mdd = max_drawdown(&prices).unwrap();

        // Maximum drawdown should be (120 - 90) / 120 = 0.25
        assert_abs_diff_eq!(mdd, 0.25);
    }

    #[test]
    fn test_stochastic() {
        let high = Array1::from_vec(vec![15.0, 16.0, 17.0, 18.0, 19.0]);
        let low = Array1::from_vec(vec![13.0, 14.0, 15.0, 16.0, 17.0]);
        let close = Array1::from_vec(vec![14.0, 15.0, 16.0, 17.0, 18.0]);

        let (k, d) = stochastic(&high, &low, &close, 3, 2).unwrap();

        assert_eq!(k.len(), 3);
        assert_eq!(d.len(), 2);

        // Stochastic values should be between 0 and 100
        for &value in k.iter() {
            assert!(value >= 0.0 && value <= 100.0);
        }
        for &value in d.iter() {
            assert!(value >= 0.0 && value <= 100.0);
        }
    }
}
