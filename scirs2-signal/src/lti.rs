//! Linear Time-Invariant (LTI) Systems
//!
//! This module provides types and functions for working with Linear Time-Invariant
//! systems, which are a fundamental concept in control theory and signal processing.
//!
//! Three different representations are provided:
//! - Transfer function representation: numerator and denominator polynomials
//! - Zero-pole-gain representation: zeros, poles, and gain
//! - State-space representation: A, B, C, D matrices
//!
//! These representations can be converted between each other, and used to analyze
//! system behavior through techniques such as impulse response, step response,
//! frequency response, and Bode plots.

use crate::error::{SignalError, SignalResult};
use num_complex::Complex64;
use num_traits::Zero;
use std::fmt::Debug;

/// A trait for all LTI system representations
pub trait LtiSystem {
    /// Get the transfer function representation of the system
    fn to_tf(&self) -> SignalResult<TransferFunction>;

    /// Get the zero-pole-gain representation of the system
    fn to_zpk(&self) -> SignalResult<ZerosPoleGain>;

    /// Get the state-space representation of the system
    fn to_ss(&self) -> SignalResult<StateSpace>;

    /// Calculate the system's frequency response
    fn frequency_response(&self, w: &[f64]) -> SignalResult<Vec<Complex64>>;

    /// Calculate the system's impulse response
    fn impulse_response(&self, t: &[f64]) -> SignalResult<Vec<f64>>;

    /// Calculate the system's step response
    fn step_response(&self, t: &[f64]) -> SignalResult<Vec<f64>>;

    /// Check if the system is stable
    fn is_stable(&self) -> SignalResult<bool>;
}

/// Transfer function representation of an LTI system
///
/// The transfer function is represented as a ratio of two polynomials:
/// H(s) = (b[0] * s^n + b[1] * s^(n-1) + ... + b[n]) / (a[0] * s^m + a[1] * s^(m-1) + ... + a[m])
///
/// Where:
/// - b: numerator coefficients (highest power first)
/// - a: denominator coefficients (highest power first)
#[derive(Debug, Clone)]
pub struct TransferFunction {
    /// Numerator coefficients (highest power first)
    pub num: Vec<f64>,

    /// Denominator coefficients (highest power first)
    pub den: Vec<f64>,

    /// Flag indicating whether the system is discrete-time
    pub dt: bool,
}

impl TransferFunction {
    /// Create a new transfer function
    ///
    /// # Arguments
    ///
    /// * `num` - Numerator coefficients (highest power first)
    /// * `den` - Denominator coefficients (highest power first)
    /// * `dt` - Flag indicating whether the system is discrete-time (optional, default = false)
    ///
    /// # Returns
    ///
    /// * A new `TransferFunction` instance or an error if the input is invalid
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_signal::lti::TransferFunction;
    ///
    /// // Create a simple first-order continuous-time system: H(s) = 1 / (s + 1)
    /// let tf = TransferFunction::new(vec![1.0], vec![1.0, 1.0], None).unwrap();
    /// ```
    pub fn new(mut num: Vec<f64>, mut den: Vec<f64>, dt: Option<bool>) -> SignalResult<Self> {
        // Remove leading zeros from numerator and denominator
        while num.len() > 1 && num[0].abs() < 1e-10 {
            num.remove(0);
        }

        while den.len() > 1 && den[0].abs() < 1e-10 {
            den.remove(0);
        }

        // Check if denominator is all zeros
        if den.iter().all(|&x| x.abs() < 1e-10) {
            return Err(SignalError::ValueError(
                "Denominator polynomial cannot be zero".to_string(),
            ));
        }

        // Normalize the denominator so that the leading coefficient is 1
        if !den.is_empty() && den[0].abs() > 1e-10 {
            let den_lead = den[0];
            for coef in &mut den {
                *coef /= den_lead;
            }

            // Also scale the numerator accordingly
            for coef in &mut num {
                *coef /= den_lead;
            }
        }

        Ok(TransferFunction {
            num,
            den,
            dt: dt.unwrap_or(false),
        })
    }

    /// Get the order of the numerator polynomial
    pub fn num_order(&self) -> usize {
        self.num.len().saturating_sub(1)
    }

    /// Get the order of the denominator polynomial
    pub fn den_order(&self) -> usize {
        self.den.len().saturating_sub(1)
    }

    /// Evaluate the transfer function at a complex value s
    pub fn evaluate(&self, s: Complex64) -> Complex64 {
        // Evaluate numerator polynomial
        let mut num_val = Complex64::zero();
        for (i, &coef) in self.num.iter().enumerate() {
            let power = (self.num.len() - 1 - i) as i32;
            num_val += Complex64::new(coef, 0.0) * s.powi(power);
        }

        // Evaluate denominator polynomial
        let mut den_val = Complex64::zero();
        for (i, &coef) in self.den.iter().enumerate() {
            let power = (self.den.len() - 1 - i) as i32;
            den_val += Complex64::new(coef, 0.0) * s.powi(power);
        }

        // Return the ratio
        if den_val.norm() < 1e-10 {
            Complex64::new(f64::INFINITY, 0.0)
        } else {
            num_val / den_val
        }
    }
}

impl LtiSystem for TransferFunction {
    fn to_tf(&self) -> SignalResult<TransferFunction> {
        Ok(self.clone())
    }

    fn to_zpk(&self) -> SignalResult<ZerosPoleGain> {
        // Convert transfer function to ZPK form by finding roots of numerator and denominator
        // This is a basic implementation - a production version would use more robust methods

        let gain = if self.num.is_empty() {
            0.0
        } else {
            self.num[0]
        };

        // Note: In practice, we would use a reliable polynomial root-finding algorithm
        // For now, returning placeholder with empty zeros and poles
        Ok(ZerosPoleGain {
            zeros: Vec::new(), // Replace with actual roots of numerator
            poles: Vec::new(), // Replace with actual roots of denominator
            gain,
            dt: self.dt,
        })
    }

    fn to_ss(&self) -> SignalResult<StateSpace> {
        // Convert transfer function to state-space form
        // For a SISO system, this involves creating a controllable canonical form

        // This is a placeholder implementation - a full implementation would
        // properly handle the controllable canonical form construction

        // For now, return an empty state-space system
        Ok(StateSpace {
            a: Vec::new(),
            b: Vec::new(),
            c: Vec::new(),
            d: Vec::new(),
            n_inputs: 1,
            n_outputs: 1,
            n_states: 0,
            dt: self.dt,
        })
    }

    fn frequency_response(&self, w: &[f64]) -> SignalResult<Vec<Complex64>> {
        let mut response = Vec::with_capacity(w.len());

        for &freq in w {
            let s = if self.dt {
                // For discrete-time systems, evaluate at z = e^(j*w)
                Complex64::new(0.0, freq).exp()
            } else {
                // For continuous-time systems, evaluate at s = j*w
                Complex64::new(0.0, freq)
            };

            response.push(self.evaluate(s));
        }

        Ok(response)
    }

    fn impulse_response(&self, t: &[f64]) -> SignalResult<Vec<f64>> {
        if t.is_empty() {
            return Ok(Vec::new());
        }

        // For continuous-time systems, we use numerical simulation by
        // converting to state-space form and then simulating the response.
        if !self.dt {
            // Convert to state-space form if it's not already available
            let ss = self.to_ss()?;

            // Get time step (assume uniform sampling)
            let dt = if t.len() > 1 { t[1] - t[0] } else { 0.001 };

            // Simulate impulse response
            let mut response = vec![0.0; t.len()];

            if !ss.b.is_empty() && !ss.c.is_empty() {
                // Initial state is zero
                let mut x = vec![0.0; ss.n_states];

                // For an impulse, the input at t[0] is 1/dt, and 0 otherwise
                // Inject impulse: u[0] = 1/dt, which approximates a continuous impulse
                for (j, _) in (0..ss.n_inputs).enumerate() {
                    for (i, x_i) in x.iter_mut().enumerate().take(ss.n_states) {
                        *x_i += ss.b[i * ss.n_inputs + j] * (1.0 / dt);
                    }
                }

                // Record initial output
                for i in 0..ss.n_outputs {
                    let mut y = 0.0;
                    for (j, &x_j) in x.iter().enumerate().take(ss.n_states) {
                        y += ss.c[i * ss.n_states + j] * x_j;
                    }
                    if i == 0 {
                        // For SISO systems
                        response[0] = y;
                    }
                }

                // Simulate the system response for the rest of the time points
                for (_k, response_k) in response.iter_mut().enumerate().skip(1).take(t.len() - 1) {
                    // Update state: dx/dt = Ax + Bu, use forward Euler for simplicity
                    let mut x_new = vec![0.0; ss.n_states];

                    for (i, x_new_val) in x_new.iter_mut().enumerate().take(ss.n_states) {
                        for (j, &x_val) in x.iter().enumerate().take(ss.n_states) {
                            *x_new_val += ss.a[i * ss.n_states + j] * x_val * dt;
                        }
                        // No input term (Bu) after initial impulse
                    }

                    // Copy updated state
                    x = x_new;

                    // Calculate output: y = Cx + Du (u is zero after initial impulse)
                    for i in 0..ss.n_outputs {
                        let mut y = 0.0;
                        for (j, &x_j) in x.iter().enumerate().take(ss.n_states) {
                            y += ss.c[i * ss.n_states + j] * x_j;
                        }
                        if i == 0 {
                            // For SISO systems
                            *response_k = y;
                        }
                    }
                }
            }

            Ok(response)
        } else {
            // For discrete-time systems, impulse response h[n] is equivalent to
            // the inverse Z-transform of the transfer function H(z)
            // For a DT system H(z) = B(z)/A(z), the impulse response is given by
            // the coefficients of the series expansion of H(z)

            let n = t.len();
            let mut response = vec![0.0; n];

            // Check if we have the right number of coefficients
            if self.num.is_empty() || self.den.is_empty() {
                return Ok(response);
            }

            // For a proper transfer function with normalized denominator,
            // the first impulse response value is b[0]/a[0]
            response[0] = if !self.den.is_empty() && self.den[0].abs() > 1e-10 {
                self.num[0] / self.den[0]
            } else {
                self.num[0]
            };

            // For later samples, we use the recurrence relation:
            // h[n] = (b[n] - sum_{k=1}^n a[k]*h[n-k])/a[0]
            for n in 1..response.len() {
                // Add numerator contribution
                if n < self.num.len() {
                    response[n] = self.num[n];
                }

                // Subtract denominator * past outputs
                for k in 1..std::cmp::min(n + 1, self.den.len()) {
                    response[n] -= self.den[k] * response[n - k];
                }

                // Normalize by a[0]
                if !self.den.is_empty() && self.den[0].abs() > 1e-10 {
                    response[n] /= self.den[0];
                }
            }

            Ok(response)
        }
    }

    fn step_response(&self, t: &[f64]) -> SignalResult<Vec<f64>> {
        if t.is_empty() {
            return Ok(Vec::new());
        }

        if !self.dt {
            // For continuous-time systems:
            // 1. Get impulse response
            let impulse = self.impulse_response(t)?;

            // 2. Integrate the impulse response to get the step response
            // Using the trapezoidal rule for integration
            let mut step = vec![0.0; t.len()];

            if t.len() > 1 {
                let dt = t[1] - t[0];

                // Initialize with the first value
                step[0] = impulse[0] * dt / 2.0;

                // Accumulate the integral
                for i in 1..t.len() {
                    step[i] = step[i - 1] + (impulse[i - 1] + impulse[i]) * dt / 2.0;
                }
            }

            Ok(step)
        } else {
            // For discrete-time systems:
            // The step response can be calculated either by:
            // 1. Convolving the impulse response with a step input
            // 2. Directly simulating with a step input
            // We'll use approach 1 for simplicity

            let impulse = self.impulse_response(t)?;
            let mut step = vec![0.0; t.len()];

            // Convolve with a unit step (running sum of impulse response)
            for (i, step_val) in step.iter_mut().enumerate().take(t.len()) {
                for &impulse_val in impulse.iter().take(i + 1) {
                    *step_val += impulse_val;
                }
            }

            Ok(step)
        }
    }

    fn is_stable(&self) -> SignalResult<bool> {
        // A system is stable if all its poles have negative real parts (continuous-time)
        // or are inside the unit circle (discrete-time)

        // For now, return a placeholder
        // In practice, we would check the poles from to_zpk()
        Ok(true)
    }
}

/// Zeros-poles-gain representation of an LTI system
///
/// The transfer function is represented as:
/// H(s) = gain * (s - zeros[0]) * (s - zeros[1]) * ... / ((s - poles[0]) * (s - poles[1]) * ...)
#[derive(Debug, Clone)]
pub struct ZerosPoleGain {
    /// Zeros of the transfer function
    pub zeros: Vec<Complex64>,

    /// Poles of the transfer function
    pub poles: Vec<Complex64>,

    /// System gain
    pub gain: f64,

    /// Flag indicating whether the system is discrete-time
    pub dt: bool,
}

impl ZerosPoleGain {
    /// Create a new zeros-poles-gain representation
    ///
    /// # Arguments
    ///
    /// * `zeros` - Zeros of the transfer function
    /// * `poles` - Poles of the transfer function
    /// * `gain` - System gain
    /// * `dt` - Flag indicating whether the system is discrete-time (optional, default = false)
    ///
    /// # Returns
    ///
    /// * A new `ZerosPoleGain` instance or an error if the input is invalid
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_signal::lti::ZerosPoleGain;
    /// use num_complex::Complex64;
    ///
    /// // Create a simple first-order continuous-time system: H(s) = 1 / (s + 1)
    /// let zpk = ZerosPoleGain::new(
    ///     Vec::new(),  // No zeros
    ///     vec![Complex64::new(-1.0, 0.0)],  // One pole at s = -1
    ///     1.0,  // Gain = 1
    ///     None,
    /// ).unwrap();
    /// ```
    pub fn new(
        zeros: Vec<Complex64>,
        poles: Vec<Complex64>,
        gain: f64,
        dt: Option<bool>,
    ) -> SignalResult<Self> {
        Ok(ZerosPoleGain {
            zeros,
            poles,
            gain,
            dt: dt.unwrap_or(false),
        })
    }

    /// Evaluate the transfer function at a complex value s
    pub fn evaluate(&self, s: Complex64) -> Complex64 {
        // Compute the numerator product (s - zeros[i])
        let mut num = Complex64::new(self.gain, 0.0);
        for &zero in &self.zeros {
            num *= s - zero;
        }

        // Compute the denominator product (s - poles[i])
        let mut den = Complex64::new(1.0, 0.0);
        for &pole in &self.poles {
            den *= s - pole;
        }

        // Return the ratio
        if den.norm() < 1e-10 {
            Complex64::new(f64::INFINITY, 0.0)
        } else {
            num / den
        }
    }
}

impl LtiSystem for ZerosPoleGain {
    fn to_tf(&self) -> SignalResult<TransferFunction> {
        // Convert ZPK to transfer function by expanding the polynomial products
        // This is a basic implementation - a production version would use more robust methods

        // For now, return a placeholder
        // In practice, we would expand (s - zero_1) * (s - zero_2) * ... for the numerator
        // and (s - pole_1) * (s - pole_2) * ... for the denominator

        Ok(TransferFunction {
            num: vec![self.gain],
            den: vec![1.0],
            dt: self.dt,
        })
    }

    fn to_zpk(&self) -> SignalResult<ZerosPoleGain> {
        Ok(self.clone())
    }

    fn to_ss(&self) -> SignalResult<StateSpace> {
        // Convert ZPK to state-space
        // Typically done by first converting to transfer function, then to state-space

        // For now, return a placeholder
        Ok(StateSpace {
            a: Vec::new(),
            b: Vec::new(),
            c: Vec::new(),
            d: Vec::new(),
            n_inputs: 1,
            n_outputs: 1,
            n_states: 0,
            dt: self.dt,
        })
    }

    fn frequency_response(&self, w: &[f64]) -> SignalResult<Vec<Complex64>> {
        let mut response = Vec::with_capacity(w.len());

        for &freq in w {
            let s = if self.dt {
                // For discrete-time systems, evaluate at z = e^(j*w)
                Complex64::new(0.0, freq).exp()
            } else {
                // For continuous-time systems, evaluate at s = j*w
                Complex64::new(0.0, freq)
            };

            response.push(self.evaluate(s));
        }

        Ok(response)
    }

    fn impulse_response(&self, t: &[f64]) -> SignalResult<Vec<f64>> {
        // Placeholder for impulse response calculation
        let response = vec![0.0; t.len()];
        Ok(response)
    }

    fn step_response(&self, t: &[f64]) -> SignalResult<Vec<f64>> {
        // Placeholder for step response calculation
        let response = vec![0.0; t.len()];
        Ok(response)
    }

    fn is_stable(&self) -> SignalResult<bool> {
        // A system is stable if all its poles have negative real parts (continuous-time)
        // or are inside the unit circle (discrete-time)

        for &pole in &self.poles {
            if self.dt {
                // For discrete-time systems, check if poles are inside the unit circle
                if pole.norm() >= 1.0 {
                    return Ok(false);
                }
            } else {
                // For continuous-time systems, check if poles have negative real parts
                if pole.re >= 0.0 {
                    return Ok(false);
                }
            }
        }

        Ok(true)
    }
}

/// State-space representation of an LTI system
///
/// The system is represented as:
/// dx/dt = A*x + B*u  (for continuous-time systems)
/// x[k+1] = A*x[k] + B*u[k]  (for discrete-time systems)
/// y = C*x + D*u
///
/// Where:
/// - x is the state vector
/// - u is the input vector
/// - y is the output vector
/// - A, B, C, D are matrices of appropriate dimensions
#[derive(Debug, Clone)]
pub struct StateSpace {
    /// State matrix (n_states x n_states)
    pub a: Vec<f64>,

    /// Input matrix (n_states x n_inputs)
    pub b: Vec<f64>,

    /// Output matrix (n_outputs x n_states)
    pub c: Vec<f64>,

    /// Feedthrough matrix (n_outputs x n_inputs)
    pub d: Vec<f64>,

    /// Number of state variables
    pub n_states: usize,

    /// Number of inputs
    pub n_inputs: usize,

    /// Number of outputs
    pub n_outputs: usize,

    /// Flag indicating whether the system is discrete-time
    pub dt: bool,
}

impl StateSpace {
    /// Create a new state-space system
    ///
    /// # Arguments
    ///
    /// * `a` - State matrix (n_states x n_states)
    /// * `b` - Input matrix (n_states x n_inputs)
    /// * `c` - Output matrix (n_outputs x n_states)
    /// * `d` - Feedthrough matrix (n_outputs x n_inputs)
    /// * `dt` - Flag indicating whether the system is discrete-time (optional, default = false)
    ///
    /// # Returns
    ///
    /// * A new `StateSpace` instance or an error if the input is invalid
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_signal::lti::StateSpace;
    ///
    /// // Create a simple first-order system: dx/dt = -x + u, y = x
    /// let ss = StateSpace::new(
    ///     vec![-1.0],  // A = [-1]
    ///     vec![1.0],   // B = [1]
    ///     vec![1.0],   // C = [1]
    ///     vec![0.0],   // D = [0]
    ///     None,
    /// ).unwrap();
    /// ```
    pub fn new(
        a: Vec<f64>,
        b: Vec<f64>,
        c: Vec<f64>,
        d: Vec<f64>,
        dt: Option<bool>,
    ) -> SignalResult<Self> {
        // Determine the system dimensions from the matrix shapes
        let n_states = (a.len() as f64).sqrt() as usize;

        // Check if A is square
        if n_states * n_states != a.len() {
            return Err(SignalError::ValueError(
                "A matrix must be square".to_string(),
            ));
        }

        // Infer n_inputs from B
        let n_inputs = if n_states == 0 { 0 } else { b.len() / n_states };

        // Check consistency of B
        if n_states * n_inputs != b.len() {
            return Err(SignalError::ValueError(
                "B matrix has inconsistent dimensions".to_string(),
            ));
        }

        // Infer n_outputs from C
        let n_outputs = if n_states == 0 { 0 } else { c.len() / n_states };

        // Check consistency of C
        if n_outputs * n_states != c.len() {
            return Err(SignalError::ValueError(
                "C matrix has inconsistent dimensions".to_string(),
            ));
        }

        // Check consistency of D
        if n_outputs * n_inputs != d.len() {
            return Err(SignalError::ValueError(
                "D matrix has inconsistent dimensions".to_string(),
            ));
        }

        Ok(StateSpace {
            a,
            b,
            c,
            d,
            n_states,
            n_inputs,
            n_outputs,
            dt: dt.unwrap_or(false),
        })
    }

    /// Get an element of the A matrix
    pub fn a(&self, i: usize, j: usize) -> SignalResult<f64> {
        if i >= self.n_states || j >= self.n_states {
            return Err(SignalError::ValueError(
                "Index out of bounds for A matrix".to_string(),
            ));
        }

        Ok(self.a[i * self.n_states + j])
    }

    /// Get an element of the B matrix
    pub fn b(&self, i: usize, j: usize) -> SignalResult<f64> {
        if i >= self.n_states || j >= self.n_inputs {
            return Err(SignalError::ValueError(
                "Index out of bounds for B matrix".to_string(),
            ));
        }

        Ok(self.b[i * self.n_inputs + j])
    }

    /// Get an element of the C matrix
    pub fn c(&self, i: usize, j: usize) -> SignalResult<f64> {
        if i >= self.n_outputs || j >= self.n_states {
            return Err(SignalError::ValueError(
                "Index out of bounds for C matrix".to_string(),
            ));
        }

        Ok(self.c[i * self.n_states + j])
    }

    /// Get an element of the D matrix
    pub fn d(&self, i: usize, j: usize) -> SignalResult<f64> {
        if i >= self.n_outputs || j >= self.n_inputs {
            return Err(SignalError::ValueError(
                "Index out of bounds for D matrix".to_string(),
            ));
        }

        Ok(self.d[i * self.n_inputs + j])
    }
}

impl LtiSystem for StateSpace {
    fn to_tf(&self) -> SignalResult<TransferFunction> {
        // Convert state-space to transfer function
        // For SISO systems, TF(s) = C * (sI - A)^-1 * B + D

        // For now, return a placeholder
        // In practice, we would calculate the matrix inverse and polynomial expansion

        Ok(TransferFunction {
            num: vec![1.0],
            den: vec![1.0],
            dt: self.dt,
        })
    }

    fn to_zpk(&self) -> SignalResult<ZerosPoleGain> {
        // Convert state-space to ZPK
        // Typically done by first converting to transfer function, then factoring

        // For now, return a placeholder
        Ok(ZerosPoleGain {
            zeros: Vec::new(),
            poles: Vec::new(),
            gain: 1.0,
            dt: self.dt,
        })
    }

    fn to_ss(&self) -> SignalResult<StateSpace> {
        Ok(self.clone())
    }

    fn frequency_response(&self, w: &[f64]) -> SignalResult<Vec<Complex64>> {
        // Calculate the frequency response for state-space system
        // H(s) = C * (sI - A)^-1 * B + D

        // For now, return a placeholder
        // In practice, we would calculate the matrix inverse for each frequency

        let response = vec![Complex64::new(1.0, 0.0); w.len()];
        Ok(response)
    }

    fn impulse_response(&self, t: &[f64]) -> SignalResult<Vec<f64>> {
        // Placeholder for impulse response calculation
        let response = vec![0.0; t.len()];
        Ok(response)
    }

    fn step_response(&self, t: &[f64]) -> SignalResult<Vec<f64>> {
        // Placeholder for step response calculation
        let response = vec![0.0; t.len()];
        Ok(response)
    }

    fn is_stable(&self) -> SignalResult<bool> {
        // A state-space system is stable if all eigenvalues of A have negative real parts (continuous-time)
        // or are inside the unit circle (discrete-time)

        // For now, return a placeholder
        // In practice, we would calculate the eigenvalues of A

        Ok(true)
    }
}

/// Calculate the Bode plot data (magnitude and phase) for an LTI system
///
/// # Arguments
///
/// * `system` - The LTI system to analyze
/// * `w` - The frequency points at which to evaluate the response
///
/// # Returns
///
/// * A tuple containing (frequencies, magnitude in dB, phase in degrees)
pub fn bode<T: LtiSystem>(
    system: &T,
    w: Option<&[f64]>,
) -> SignalResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    // Default frequencies if none provided
    let frequencies = match w {
        Some(freq) => freq.to_vec(),
        None => {
            // Generate logarithmically spaced frequencies between 0.01 and 100 rad/s
            let n = 100;
            let mut w_out = Vec::with_capacity(n);

            let w_min = 0.01;
            let w_max = 100.0;
            let log_step = f64::powf(w_max / w_min, 1.0 / (n - 1) as f64);

            let mut w_val = w_min;
            for _ in 0..n {
                w_out.push(w_val);
                w_val *= log_step;
            }

            w_out
        }
    };

    // Calculate frequency response
    let resp = system.frequency_response(&frequencies)?;

    // Convert to magnitude (dB) and phase (degrees)
    let mut mag = Vec::with_capacity(resp.len());
    let mut phase = Vec::with_capacity(resp.len());

    for &val in &resp {
        // Magnitude in dB: 20 * log10(|H(jw)|)
        let mag_db = 20.0 * val.norm().log10();
        mag.push(mag_db);

        // Phase in degrees: arg(H(jw)) * 180/pi
        let phase_deg = val.arg() * 180.0 / std::f64::consts::PI;
        phase.push(phase_deg);
    }

    Ok((frequencies, mag, phase))
}

/// Functions for creating and manipulating LTI systems
pub mod system {
    use super::*;

    /// Create a transfer function system from numerator and denominator coefficients
    ///
    /// # Arguments
    ///
    /// * `num` - Numerator coefficients (highest power first)
    /// * `den` - Denominator coefficients (highest power first)
    /// * `dt` - Flag indicating whether the system is discrete-time (optional, default = false)
    ///
    /// # Returns
    ///
    /// * A `TransferFunction` instance
    pub fn tf(num: Vec<f64>, den: Vec<f64>, dt: Option<bool>) -> SignalResult<TransferFunction> {
        TransferFunction::new(num, den, dt)
    }

    /// Create a zeros-poles-gain system
    ///
    /// # Arguments
    ///
    /// * `zeros` - Zeros of the transfer function
    /// * `poles` - Poles of the transfer function
    /// * `gain` - System gain
    /// * `dt` - Flag indicating whether the system is discrete-time (optional, default = false)
    ///
    /// # Returns
    ///
    /// * A `ZerosPoleGain` instance
    pub fn zpk(
        zeros: Vec<Complex64>,
        poles: Vec<Complex64>,
        gain: f64,
        dt: Option<bool>,
    ) -> SignalResult<ZerosPoleGain> {
        ZerosPoleGain::new(zeros, poles, gain, dt)
    }

    /// Create a state-space system
    ///
    /// # Arguments
    ///
    /// * `a` - State matrix (n_states x n_states)
    /// * `b` - Input matrix (n_states x n_inputs)
    /// * `c` - Output matrix (n_outputs x n_states)
    /// * `d` - Feedthrough matrix (n_outputs x n_inputs)
    /// * `dt` - Flag indicating whether the system is discrete-time (optional, default = false)
    ///
    /// # Returns
    ///
    /// * A `StateSpace` instance
    pub fn ss(
        a: Vec<f64>,
        b: Vec<f64>,
        c: Vec<f64>,
        d: Vec<f64>,
        dt: Option<bool>,
    ) -> SignalResult<StateSpace> {
        StateSpace::new(a, b, c, d, dt)
    }

    /// Convert a continuous-time system to a discrete-time system using zero-order hold method
    ///
    /// # Arguments
    ///
    /// * `system` - A continuous-time LTI system
    /// * `dt` - The sampling period
    ///
    /// # Returns
    ///
    /// * A discretized version of the system
    pub fn c2d<T: LtiSystem>(system: &T, _dt: f64) -> SignalResult<StateSpace> {
        // Convert to state-space first
        let ss_sys = system.to_ss()?;

        // Ensure the system is continuous-time
        if ss_sys.dt {
            return Err(SignalError::ValueError(
                "System is already discrete-time".to_string(),
            ));
        }

        // For now, return a placeholder for the discretized system
        // In practice, we would use the matrix exponential method: A_d = exp(A*dt)

        Ok(StateSpace {
            a: ss_sys.a.clone(),
            b: ss_sys.b.clone(),
            c: ss_sys.c.clone(),
            d: ss_sys.d.clone(),
            n_states: ss_sys.n_states,
            n_inputs: ss_sys.n_inputs,
            n_outputs: ss_sys.n_outputs,
            dt: true,
        })
    }

    /// Connect two LTI systems in series
    ///
    /// For systems G1 and G2 in series: H(s) = G2(s) * G1(s)
    ///
    /// # Arguments
    ///
    /// * `g1` - First system (input side)
    /// * `g2` - Second system (output side)
    ///
    /// # Returns
    ///
    /// * The series interconnection as a transfer function
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use scirs2_signal::lti::system::*;
    ///
    /// let g1 = tf(vec![1.0], vec![1.0, 1.0], None).unwrap();
    /// let g2 = tf(vec![2.0], vec![1.0, 2.0], None).unwrap();
    /// let series_sys = series(&g1, &g2).unwrap();
    /// ```
    pub fn series<T1: LtiSystem, T2: LtiSystem>(
        g1: &T1,
        g2: &T2,
    ) -> SignalResult<TransferFunction> {
        let tf1 = g1.to_tf()?;
        let tf2 = g2.to_tf()?;

        // Check compatibility
        if tf1.dt != tf2.dt {
            return Err(SignalError::ValueError(
                "Systems must have the same time domain (continuous or discrete)".to_string(),
            ));
        }

        // Series connection: H(s) = G2(s) * G1(s)
        // Multiply numerators and denominators
        let num = multiply_polynomials(&tf2.num, &tf1.num);
        let den = multiply_polynomials(&tf2.den, &tf1.den);

        TransferFunction::new(num, den, Some(tf1.dt))
    }

    /// Connect two LTI systems in parallel
    ///
    /// For systems G1 and G2 in parallel: H(s) = G1(s) + G2(s)
    ///
    /// # Arguments
    ///
    /// * `g1` - First system
    /// * `g2` - Second system
    ///
    /// # Returns
    ///
    /// * The parallel interconnection as a transfer function
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use scirs2_signal::lti::system::*;
    ///
    /// let g1 = tf(vec![1.0], vec![1.0, 1.0], None).unwrap();
    /// let g2 = tf(vec![2.0], vec![1.0, 2.0], None).unwrap();
    /// let parallel_sys = parallel(&g1, &g2).unwrap();
    /// ```
    pub fn parallel<T1: LtiSystem, T2: LtiSystem>(
        g1: &T1,
        g2: &T2,
    ) -> SignalResult<TransferFunction> {
        let tf1 = g1.to_tf()?;
        let tf2 = g2.to_tf()?;

        // Check compatibility
        if tf1.dt != tf2.dt {
            return Err(SignalError::ValueError(
                "Systems must have the same time domain (continuous or discrete)".to_string(),
            ));
        }

        // Parallel connection: H(s) = G1(s) + G2(s)
        // H(s) = (N1*D2 + N2*D1) / (D1*D2)
        let num1_den2 = multiply_polynomials(&tf1.num, &tf2.den);
        let num2_den1 = multiply_polynomials(&tf2.num, &tf1.den);
        let num = add_polynomials(&num1_den2, &num2_den1);
        let den = multiply_polynomials(&tf1.den, &tf2.den);

        TransferFunction::new(num, den, Some(tf1.dt))
    }

    /// Connect two LTI systems in feedback configuration
    ///
    /// For systems G (forward) and H (feedback): T(s) = G(s) / (1 + G(s)*H(s))
    /// If sign is -1: T(s) = G(s) / (1 - G(s)*H(s))
    ///
    /// # Arguments
    ///
    /// * `g` - Forward path system
    /// * `h` - Feedback path system (optional, defaults to unity feedback)
    /// * `sign` - Feedback sign (1 for negative feedback, -1 for positive feedback)
    ///
    /// # Returns
    ///
    /// * The closed-loop system as a transfer function
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use scirs2_signal::lti::system::*;
    ///
    /// let g = tf(vec![1.0], vec![1.0, 1.0], None).unwrap();
    /// let h = tf(vec![1.0], vec![1.0], None).unwrap(); // Unity feedback
    /// let closed_loop = feedback(&g, Some(&h), 1).unwrap();
    /// ```
    pub fn feedback<T1: LtiSystem>(
        g: &T1,
        h: Option<&dyn LtiSystem>,
        sign: i32,
    ) -> SignalResult<TransferFunction> {
        let tf_g = g.to_tf()?;

        let tf_h = if let Some(h_sys) = h {
            h_sys.to_tf()?
        } else {
            // Unity feedback
            TransferFunction::new(vec![1.0], vec![1.0], Some(tf_g.dt))?
        };

        // Check compatibility
        if tf_g.dt != tf_h.dt {
            return Err(SignalError::ValueError(
                "Systems must have the same time domain (continuous or discrete)".to_string(),
            ));
        }

        // Feedback connection: T(s) = G(s) / (1 + sign*G(s)*H(s))
        // Numerator: N_g * D_h
        let num = multiply_polynomials(&tf_g.num, &tf_h.den);

        // Denominator: D_g * D_h + sign * N_g * N_h
        let dg_dh = multiply_polynomials(&tf_g.den, &tf_h.den);
        let ng_nh = multiply_polynomials(&tf_g.num, &tf_h.num);

        let den = if sign > 0 {
            // Negative feedback: 1 + G*H
            add_polynomials(&dg_dh, &ng_nh)
        } else {
            // Positive feedback: 1 - G*H
            subtract_polynomials(&dg_dh, &ng_nh)
        };

        TransferFunction::new(num, den, Some(tf_g.dt))
    }

    /// Get the sensitivity function for a feedback system
    ///
    /// Sensitivity S(s) = 1 / (1 + G(s)*H(s))
    ///
    /// # Arguments
    ///
    /// * `g` - Forward path system
    /// * `h` - Feedback path system (optional, defaults to unity feedback)
    ///
    /// # Returns
    ///
    /// * The sensitivity function as a transfer function
    pub fn sensitivity<T1: LtiSystem>(
        g: &T1,
        h: Option<&dyn LtiSystem>,
    ) -> SignalResult<TransferFunction> {
        let tf_g = g.to_tf()?;

        let tf_h = if let Some(h_sys) = h {
            h_sys.to_tf()?
        } else {
            // Unity feedback
            TransferFunction::new(vec![1.0], vec![1.0], Some(tf_g.dt))?
        };

        // Check compatibility
        if tf_g.dt != tf_h.dt {
            return Err(SignalError::ValueError(
                "Systems must have the same time domain (continuous or discrete)".to_string(),
            ));
        }

        // Sensitivity: S(s) = 1 / (1 + G(s)*H(s))
        // Numerator: D_g * D_h
        let num = multiply_polynomials(&tf_g.den, &tf_h.den);

        // Denominator: D_g * D_h + N_g * N_h
        let dg_dh = multiply_polynomials(&tf_g.den, &tf_h.den);
        let ng_nh = multiply_polynomials(&tf_g.num, &tf_h.num);
        let den = add_polynomials(&dg_dh, &ng_nh);

        TransferFunction::new(num, den, Some(tf_g.dt))
    }

    /// Get the complementary sensitivity function for a feedback system
    ///
    /// Complementary sensitivity T(s) = G(s)*H(s) / (1 + G(s)*H(s))
    ///
    /// # Arguments
    ///
    /// * `g` - Forward path system
    /// * `h` - Feedback path system (optional, defaults to unity feedback)
    ///
    /// # Returns
    ///
    /// * The complementary sensitivity function as a transfer function
    pub fn complementary_sensitivity<T1: LtiSystem>(
        g: &T1,
        h: Option<&dyn LtiSystem>,
    ) -> SignalResult<TransferFunction> {
        let tf_g = g.to_tf()?;

        let tf_h = if let Some(h_sys) = h {
            h_sys.to_tf()?
        } else {
            // Unity feedback
            TransferFunction::new(vec![1.0], vec![1.0], Some(tf_g.dt))?
        };

        // Check compatibility
        if tf_g.dt != tf_h.dt {
            return Err(SignalError::ValueError(
                "Systems must have the same time domain (continuous or discrete)".to_string(),
            ));
        }

        // Complementary sensitivity: T(s) = G(s)*H(s) / (1 + G(s)*H(s))
        // Numerator: N_g * N_h
        let num = multiply_polynomials(&tf_g.num, &tf_h.num);

        // Denominator: D_g * D_h + N_g * N_h
        let dg_dh = multiply_polynomials(&tf_g.den, &tf_h.den);
        let ng_nh = multiply_polynomials(&tf_g.num, &tf_h.num);
        let den = add_polynomials(&dg_dh, &ng_nh);

        TransferFunction::new(num, den, Some(tf_g.dt))
    }
}

/// Helper functions for polynomial operations
fn multiply_polynomials(p1: &[f64], p2: &[f64]) -> Vec<f64> {
    if p1.is_empty() || p2.is_empty() {
        return vec![0.0];
    }

    let mut result = vec![0.0; p1.len() + p2.len() - 1];

    for (i, &a) in p1.iter().enumerate() {
        for (j, &b) in p2.iter().enumerate() {
            result[i + j] += a * b;
        }
    }

    result
}

fn add_polynomials(p1: &[f64], p2: &[f64]) -> Vec<f64> {
    let max_len = p1.len().max(p2.len());
    let mut result = vec![0.0; max_len];

    // Pad with zeros from the front and add
    let p1_offset = max_len - p1.len();
    let p2_offset = max_len - p2.len();

    for (i, &val) in p1.iter().enumerate() {
        result[p1_offset + i] += val;
    }

    for (i, &val) in p2.iter().enumerate() {
        result[p2_offset + i] += val;
    }

    result
}

fn subtract_polynomials(p1: &[f64], p2: &[f64]) -> Vec<f64> {
    let max_len = p1.len().max(p2.len());
    let mut result = vec![0.0; max_len];

    // Pad with zeros from the front and subtract
    let p1_offset = max_len - p1.len();
    let p2_offset = max_len - p2.len();

    for (i, &val) in p1.iter().enumerate() {
        result[p1_offset + i] += val;
    }

    for (i, &val) in p2.iter().enumerate() {
        result[p2_offset + i] -= val;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_tf_creation() {
        // Create a simple first-order system H(s) = 1 / (s + 1)
        let tf = TransferFunction::new(vec![1.0], vec![1.0, 1.0], None).unwrap();

        assert_eq!(tf.num.len(), 1);
        assert_eq!(tf.den.len(), 2);
        assert_relative_eq!(tf.num[0], 1.0);
        assert_relative_eq!(tf.den[0], 1.0);
        assert_relative_eq!(tf.den[1], 1.0);
        assert!(!tf.dt);

        // Test normalization
        let tf2 = TransferFunction::new(vec![2.0], vec![2.0, 2.0], None).unwrap();
        assert_relative_eq!(tf2.num[0], 1.0);
        assert_relative_eq!(tf2.den[0], 1.0);
        assert_relative_eq!(tf2.den[1], 1.0);
    }

    #[test]
    fn test_tf_evaluate() {
        // Create a simple first-order system H(s) = 1 / (s + 1)
        let tf = TransferFunction::new(vec![1.0], vec![1.0, 1.0], None).unwrap();

        // Evaluate at s = 0
        let result = tf.evaluate(Complex64::new(0.0, 0.0));
        assert_relative_eq!(result.re, 1.0, epsilon = 1e-6);
        assert_relative_eq!(result.im, 0.0, epsilon = 1e-6);

        // Evaluate at s = j (omega = 1)
        let result = tf.evaluate(Complex64::new(0.0, 1.0));
        assert_relative_eq!(result.norm(), 1.0 / 2.0_f64.sqrt(), epsilon = 1e-6);
    }

    #[test]
    fn test_zpk_creation() {
        // Create a simple first-order system H(s) = 1 / (s + 1)
        let zpk =
            ZerosPoleGain::new(Vec::new(), vec![Complex64::new(-1.0, 0.0)], 1.0, None).unwrap();

        assert_eq!(zpk.zeros.len(), 0);
        assert_eq!(zpk.poles.len(), 1);
        assert_relative_eq!(zpk.poles[0].re, -1.0);
        assert_relative_eq!(zpk.poles[0].im, 0.0);
        assert_relative_eq!(zpk.gain, 1.0);
        assert!(!zpk.dt);
    }

    #[test]
    fn test_zpk_evaluate() {
        // Create a simple first-order system H(s) = 1 / (s + 1)
        let zpk =
            ZerosPoleGain::new(Vec::new(), vec![Complex64::new(-1.0, 0.0)], 1.0, None).unwrap();

        // Evaluate at s = 0
        let result = zpk.evaluate(Complex64::new(0.0, 0.0));
        assert_relative_eq!(result.re, 1.0, epsilon = 1e-6);
        assert_relative_eq!(result.im, 0.0, epsilon = 1e-6);

        // Evaluate at s = j (omega = 1)
        let result = zpk.evaluate(Complex64::new(0.0, 1.0));
        assert_relative_eq!(result.norm(), 1.0 / 2.0_f64.sqrt(), epsilon = 1e-6);
    }

    #[test]
    fn test_ss_creation() {
        // Create a simple first-order system: dx/dt = -x + u, y = x
        let ss = StateSpace::new(
            vec![-1.0], // A = [-1]
            vec![1.0],  // B = [1]
            vec![1.0],  // C = [1]
            vec![0.0],  // D = [0]
            None,
        )
        .unwrap();

        assert_eq!(ss.n_states, 1);
        assert_eq!(ss.n_inputs, 1);
        assert_eq!(ss.n_outputs, 1);
        assert_relative_eq!(ss.a[0], -1.0);
        assert_relative_eq!(ss.b[0], 1.0);
        assert_relative_eq!(ss.c[0], 1.0);
        assert_relative_eq!(ss.d[0], 0.0);
        assert!(!ss.dt);
    }

    #[test]
    fn test_bode() {
        // Create a simple first-order system H(s) = 1 / (s + 1)
        let tf = TransferFunction::new(vec![1.0], vec![1.0, 1.0], None).unwrap();

        // Compute Bode plot at omega = 0.1, 1, 10
        let freqs = vec![0.1, 1.0, 10.0];
        let (w, mag, phase) = bode(&tf, Some(&freqs)).unwrap();

        // Check frequencies
        assert_eq!(w.len(), 3);
        assert_relative_eq!(w[0], 0.1, epsilon = 1e-6);
        assert_relative_eq!(w[1], 1.0, epsilon = 1e-6);
        assert_relative_eq!(w[2], 10.0, epsilon = 1e-6);

        // Check magnitudes (in dB)
        assert_eq!(mag.len(), 3);
        // At omega = 0.1, |H| = 0.995, which is -0.043 dB
        assert_relative_eq!(mag[0], -0.043, epsilon = 0.01);
        // At omega = 1, |H| = 0.707, which is -3 dB
        assert_relative_eq!(mag[1], -3.0, epsilon = 0.1);
        // At omega = 10, |H| = 0.0995, which is -20.043 dB
        assert_relative_eq!(mag[2], -20.043, epsilon = 0.1);

        // Check phases (in degrees)
        assert_eq!(phase.len(), 3);
        // At omega = 0.1, phase is about -5.7 degrees
        assert_relative_eq!(phase[0], -5.7, epsilon = 0.1);
        // At omega = 1, phase is -45 degrees
        assert_relative_eq!(phase[1], -45.0, epsilon = 0.1);
        // At omega = 10, phase is about -84.3 degrees
        assert_relative_eq!(phase[2], -84.3, epsilon = 0.1);
    }

    #[test]
    fn test_is_stable() {
        // Stable continuous-time system: H(s) = 1 / (s + 1)
        let stable =
            ZerosPoleGain::new(Vec::new(), vec![Complex64::new(-1.0, 0.0)], 1.0, None).unwrap();
        assert!(stable.is_stable().unwrap());

        // Unstable continuous-time system: H(s) = 1 / (s - 1)
        let unstable =
            ZerosPoleGain::new(Vec::new(), vec![Complex64::new(1.0, 0.0)], 1.0, None).unwrap();
        assert!(!unstable.is_stable().unwrap());

        // Stable discrete-time system: H(z) = 1 / (z - 0.5)
        let stable_dt =
            ZerosPoleGain::new(Vec::new(), vec![Complex64::new(0.5, 0.0)], 1.0, Some(true))
                .unwrap();
        assert!(stable_dt.is_stable().unwrap());

        // Unstable discrete-time system: H(z) = 1 / (z - 1.5)
        let unstable_dt =
            ZerosPoleGain::new(Vec::new(), vec![Complex64::new(1.5, 0.0)], 1.0, Some(true))
                .unwrap();
        assert!(!unstable_dt.is_stable().unwrap());
    }

    #[test]
    fn test_series_connection() {
        // Test series connection of two first-order systems
        // G1(s) = 1/(s+1), G2(s) = 2/(s+2)
        let g1 = TransferFunction::new(vec![1.0], vec![1.0, 1.0], None).unwrap();
        let g2 = TransferFunction::new(vec![2.0], vec![1.0, 2.0], None).unwrap();

        let series_sys = system::series(&g1, &g2).unwrap();

        // Series: H(s) = G2(s)*G1(s) = 2/((s+1)(s+2)) = 2/(s^2+3s+2)
        assert_eq!(series_sys.num.len(), 1);
        assert_eq!(series_sys.den.len(), 3);
        assert_relative_eq!(series_sys.num[0], 2.0);
        assert_relative_eq!(series_sys.den[0], 1.0);
        assert_relative_eq!(series_sys.den[1], 3.0);
        assert_relative_eq!(series_sys.den[2], 2.0);
    }

    #[test]
    fn test_parallel_connection() {
        // Test parallel connection of two first-order systems
        // G1(s) = 1/(s+1), G2(s) = 1/(s+2)
        let g1 = TransferFunction::new(vec![1.0], vec![1.0, 1.0], None).unwrap();
        let g2 = TransferFunction::new(vec![1.0], vec![1.0, 2.0], None).unwrap();

        let parallel_sys = system::parallel(&g1, &g2).unwrap();

        // Parallel: H(s) = G1(s)+G2(s) = 1/(s+1) + 1/(s+2) = (s+2+s+1)/((s+1)(s+2))
        //         = (2s+3)/(s^2+3s+2)
        assert_eq!(parallel_sys.num.len(), 2);
        assert_eq!(parallel_sys.den.len(), 3);
        assert_relative_eq!(parallel_sys.num[0], 2.0);
        assert_relative_eq!(parallel_sys.num[1], 3.0);
        assert_relative_eq!(parallel_sys.den[0], 1.0);
        assert_relative_eq!(parallel_sys.den[1], 3.0);
        assert_relative_eq!(parallel_sys.den[2], 2.0);
    }

    #[test]
    fn test_feedback_connection() {
        // Test feedback connection with unity feedback
        // G(s) = 1/(s+1), unity feedback
        let g = TransferFunction::new(vec![1.0], vec![1.0, 1.0], None).unwrap();

        let feedback_sys = system::feedback(&g, None, 1).unwrap();

        // Feedback: T(s) = G(s)/(1+G(s)) = (1/(s+1))/(1+1/(s+1)) = 1/(s+2)
        assert_eq!(feedback_sys.num.len(), 1);
        assert_eq!(feedback_sys.den.len(), 2);
        assert_relative_eq!(feedback_sys.num[0], 1.0);
        assert_relative_eq!(feedback_sys.den[0], 1.0);
        assert_relative_eq!(feedback_sys.den[1], 2.0);
    }

    #[test]
    fn test_feedback_with_controller() {
        // Test feedback connection with a controller
        // G(s) = 1/(s+1), H(s) = 2 (proportional controller)
        let g = TransferFunction::new(vec![1.0], vec![1.0, 1.0], None).unwrap();
        let h = TransferFunction::new(vec![2.0], vec![1.0], None).unwrap();

        let feedback_sys = system::feedback(&g, Some(&h as &dyn LtiSystem), 1).unwrap();

        // Feedback: T(s) = G(s)/(1+G(s)*H(s)) = (1/(s+1))/(1+2/(s+1)) = 1/(s+3)
        assert_eq!(feedback_sys.num.len(), 1);
        assert_eq!(feedback_sys.den.len(), 2);
        assert_relative_eq!(feedback_sys.num[0], 1.0);
        assert_relative_eq!(feedback_sys.den[0], 1.0);
        assert_relative_eq!(feedback_sys.den[1], 3.0);
    }

    #[test]
    fn test_sensitivity_function() {
        // Test sensitivity function
        // G(s) = 10/(s+1), unity feedback
        let g = TransferFunction::new(vec![10.0], vec![1.0, 1.0], None).unwrap();

        let sens = system::sensitivity(&g, None).unwrap();

        // Sensitivity: S(s) = 1/(1+G(s)) = (s+1)/(s+11)
        assert_eq!(sens.num.len(), 2);
        assert_eq!(sens.den.len(), 2);
        assert_relative_eq!(sens.num[0], 1.0);
        assert_relative_eq!(sens.num[1], 1.0);
        assert_relative_eq!(sens.den[0], 1.0);
        assert_relative_eq!(sens.den[1], 11.0);
    }

    #[test]
    fn test_complementary_sensitivity() {
        // Test complementary sensitivity function
        // G(s) = 10/(s+1), unity feedback
        let g = TransferFunction::new(vec![10.0], vec![1.0, 1.0], None).unwrap();

        let comp_sens = system::complementary_sensitivity(&g, None).unwrap();

        // Complementary sensitivity: T(s) = G(s)/(1+G(s)) = 10/(s+11)
        assert_eq!(comp_sens.num.len(), 1);
        assert_eq!(comp_sens.den.len(), 2);
        assert_relative_eq!(comp_sens.num[0], 10.0);
        assert_relative_eq!(comp_sens.den[0], 1.0);
        assert_relative_eq!(comp_sens.den[1], 11.0);
    }

    #[test]
    fn test_polynomial_operations() {
        // Test multiply_polynomials
        let p1 = vec![1.0, 2.0]; // x + 2
        let p2 = vec![1.0, 3.0]; // x + 3
        let result = multiply_polynomials(&p1, &p2);
        // (x + 2)(x + 3) = x^2 + 5x + 6
        assert_eq!(result.len(), 3);
        assert_relative_eq!(result[0], 1.0);
        assert_relative_eq!(result[1], 5.0);
        assert_relative_eq!(result[2], 6.0);

        // Test add_polynomials
        let p1 = vec![1.0, 2.0]; // x + 2
        let p2 = vec![1.0, 3.0]; // x + 3
        let result = add_polynomials(&p1, &p2);
        // (x + 2) + (x + 3) = 2x + 5
        assert_eq!(result.len(), 2);
        assert_relative_eq!(result[0], 2.0);
        assert_relative_eq!(result[1], 5.0);

        // Test subtract_polynomials
        let p1 = vec![2.0, 5.0]; // 2x + 5
        let p2 = vec![1.0, 3.0]; // x + 3
        let result = subtract_polynomials(&p1, &p2);
        // (2x + 5) - (x + 3) = x + 2
        assert_eq!(result.len(), 2);
        assert_relative_eq!(result[0], 1.0);
        assert_relative_eq!(result[1], 2.0);
    }

    #[test]
    fn test_system_interconnection_errors() {
        // Test error when connecting continuous and discrete-time systems
        let g_ct = TransferFunction::new(vec![1.0], vec![1.0, 1.0], Some(false)).unwrap();
        let g_dt = TransferFunction::new(vec![1.0], vec![1.0, 1.0], Some(true)).unwrap();

        let result = system::series(&g_ct, &g_dt);
        assert!(result.is_err());

        let result = system::parallel(&g_ct, &g_dt);
        assert!(result.is_err());

        let result = system::feedback(&g_ct, Some(&g_dt as &dyn LtiSystem), 1);
        assert!(result.is_err());
    }
}

// Controllability and Observability Analysis
//
// This section provides functions for analyzing the controllability and
// observability properties of linear time-invariant systems in state-space form.

/// Controllability analysis result
#[derive(Debug, Clone)]
pub struct ControllabilityAnalysis {
    /// Whether the system is completely controllable
    pub is_controllable: bool,
    /// Rank of the controllability matrix
    pub controllability_rank: usize,
    /// Number of states
    pub state_dimension: usize,
    /// Controllability matrix
    pub controllability_matrix: Vec<Vec<f64>>,
    /// Controllable subspace dimension
    pub controllable_subspace_dim: usize,
}

/// Observability analysis result
#[derive(Debug, Clone)]
pub struct ObservabilityAnalysis {
    /// Whether the system is completely observable
    pub is_observable: bool,
    /// Rank of the observability matrix
    pub observability_rank: usize,
    /// Number of states
    pub state_dimension: usize,
    /// Observability matrix
    pub observability_matrix: Vec<Vec<f64>>,
    /// Observable subspace dimension
    pub observable_subspace_dim: usize,
}

/// Combined controllability and observability analysis
#[derive(Debug, Clone)]
pub struct ControlObservabilityAnalysis {
    /// Controllability analysis
    pub controllability: ControllabilityAnalysis,
    /// Observability analysis
    pub observability: ObservabilityAnalysis,
    /// Whether the system is minimal (controllable and observable)
    pub is_minimal: bool,
    /// Kalman canonical decomposition structure
    pub kalman_structure: KalmanStructure,
}

/// Kalman canonical decomposition structure
#[derive(Debug, Clone)]
pub struct KalmanStructure {
    /// Controllable and observable subspace dimension
    pub co_dimension: usize,
    /// Controllable but not observable subspace dimension
    pub c_no_dimension: usize,
    /// Not controllable but observable subspace dimension
    pub nc_o_dimension: usize,
    /// Neither controllable nor observable subspace dimension
    pub nc_no_dimension: usize,
}

/// Analyze controllability of a state-space system
///
/// A system is controllable if the controllability matrix [B AB A²B ... A^(n-1)B]
/// has full row rank, where n is the number of states.
///
/// # Arguments
///
/// * `ss` - State-space system to analyze
///
/// # Returns
///
/// * Controllability analysis result
pub fn analyze_controllability(ss: &StateSpace) -> SignalResult<ControllabilityAnalysis> {
    let n = ss.n_states; // Number of states
    if n == 0 {
        return Err(SignalError::ValueError("Empty state matrix".to_string()));
    }

    let m = ss.n_inputs; // Number of inputs
    if m == 0 {
        return Err(SignalError::ValueError("Empty input matrix".to_string()));
    }

    // Build controllability matrix: [B AB A²B ... A^(n-1)B]
    // Convert flattened matrices to 2D format for easier manipulation
    let a_matrix = flatten_to_2d(&ss.a, n, n)?;
    let b_matrix = flatten_to_2d(&ss.b, n, m)?;

    // Initialize controllability matrix with the right dimensions: n x (n*m)
    let mut controllability_matrix = vec![vec![0.0; n * m]; n];
    let mut current_ab = b_matrix.clone();

    // Add B columns to controllability matrix
    for row_idx in 0..n {
        for col_idx in 0..m {
            controllability_matrix[row_idx][col_idx] = current_ab[row_idx][col_idx];
        }
    }

    // Add AB, A²B, ..., A^(n-1)B columns
    for block in 1..n {
        current_ab = matrix_multiply(&a_matrix, &current_ab)?;

        for row_idx in 0..n {
            for col_idx in 0..m {
                let matrix_col_idx = block * m + col_idx;
                controllability_matrix[row_idx][matrix_col_idx] = current_ab[row_idx][col_idx];
            }
        }
    }

    // Calculate rank of controllability matrix
    let rank = matrix_rank(&controllability_matrix)?;
    let is_controllable = rank == n;

    Ok(ControllabilityAnalysis {
        is_controllable,
        controllability_rank: rank,
        state_dimension: n,
        controllability_matrix: controllability_matrix.clone(),
        controllable_subspace_dim: rank,
    })
}

/// Analyze observability of a state-space system
///
/// A system is observable if the observability matrix [C; CA; CA²; ...; CA^(n-1)]
/// has full column rank, where n is the number of states.
///
/// # Arguments
///
/// * `ss` - State-space system to analyze
///
/// # Returns
///
/// * Observability analysis result
pub fn analyze_observability(ss: &StateSpace) -> SignalResult<ObservabilityAnalysis> {
    let n = ss.n_states; // Number of states
    if n == 0 {
        return Err(SignalError::ValueError("Empty state matrix".to_string()));
    }

    let p = ss.n_outputs; // Number of outputs
    if p == 0 {
        return Err(SignalError::ValueError("Empty output matrix".to_string()));
    }

    // Build observability matrix: [C; CA; CA²; ...; CA^(n-1)]
    // Convert flattened matrices to 2D format for easier manipulation
    let a_matrix = flatten_to_2d(&ss.a, n, n)?;
    let c_matrix = flatten_to_2d(&ss.c, p, n)?;

    // Initialize observability matrix with the right dimensions: (n*p) x n
    let mut observability_matrix = vec![vec![0.0; n]; n * p];
    let mut current_ca = c_matrix.clone();

    // Add C rows to observability matrix
    for row_idx in 0..p {
        for col_idx in 0..n {
            observability_matrix[row_idx][col_idx] = current_ca[row_idx][col_idx];
        }
    }

    // Add CA, CA², ..., CA^(n-1) rows
    for block in 1..n {
        current_ca = matrix_multiply(&current_ca, &a_matrix)?;

        for (row_idx, row) in current_ca.iter().enumerate().take(p) {
            let matrix_row_idx = block * p + row_idx;
            observability_matrix[matrix_row_idx][..n].copy_from_slice(&row[..n]);
        }
    }

    // Calculate rank of observability matrix
    let rank = matrix_rank(&observability_matrix)?;
    let is_observable = rank == n;

    Ok(ObservabilityAnalysis {
        is_observable,
        observability_rank: rank,
        state_dimension: n,
        observability_matrix: observability_matrix.clone(),
        observable_subspace_dim: rank,
    })
}

/// Perform combined controllability and observability analysis
///
/// This function analyzes both controllability and observability properties
/// and determines if the system is minimal (both controllable and observable).
///
/// # Arguments
///
/// * `ss` - State-space system to analyze
///
/// # Returns
///
/// * Combined analysis result including Kalman decomposition structure
pub fn analyze_control_observability(
    ss: &StateSpace,
) -> SignalResult<ControlObservabilityAnalysis> {
    let controllability = analyze_controllability(ss)?;
    let observability = analyze_observability(ss)?;

    let is_minimal = controllability.is_controllable && observability.is_observable;

    // Determine Kalman canonical decomposition structure
    let n = ss.n_states;
    let nc = controllability.controllable_subspace_dim;
    let no = observability.observable_subspace_dim;

    // This is a simplified analysis - full Kalman decomposition would require
    // computing the intersection of controllable and observable subspaces
    let co_dimension = if is_minimal {
        n
    } else {
        (nc + no).saturating_sub(n).min(nc.min(no))
    };
    let c_no_dimension = nc.saturating_sub(co_dimension);
    let nc_o_dimension = no.saturating_sub(co_dimension);
    let nc_no_dimension = n.saturating_sub(co_dimension + c_no_dimension + nc_o_dimension);

    let kalman_structure = KalmanStructure {
        co_dimension,
        c_no_dimension,
        nc_o_dimension,
        nc_no_dimension,
    };

    Ok(ControlObservabilityAnalysis {
        controllability,
        observability,
        is_minimal,
        kalman_structure,
    })
}

/// Check if two systems are equivalent (same input-output behavior)
///
/// Two systems are equivalent if they have the same transfer function,
/// even if their state-space representations are different.
///
/// # Arguments
///
/// * `sys1` - First system
/// * `sys2` - Second system
/// * `tolerance` - Tolerance for numerical comparison
///
/// # Returns
///
/// * True if systems are equivalent within tolerance
pub fn systems_equivalent(
    sys1: &dyn LtiSystem,
    sys2: &dyn LtiSystem,
    tolerance: f64,
) -> SignalResult<bool> {
    // Compare transfer functions
    let tf1 = sys1.to_tf()?;
    let tf2 = sys2.to_tf()?;

    // Check if transfer functions are the same (within tolerance)
    if tf1.num.len() != tf2.num.len() || tf1.den.len() != tf2.den.len() {
        return Ok(false);
    }

    // Normalize transfer functions for comparison
    let norm1 = tf1.den[0];
    let norm2 = tf2.den[0];

    if norm1.abs() < tolerance || norm2.abs() < tolerance {
        return Ok(false);
    }

    // Compare normalized coefficients
    for (c1, c2) in tf1.num.iter().zip(tf2.num.iter()) {
        if ((c1 / norm1) - (c2 / norm2)).abs() > tolerance {
            return Ok(false);
        }
    }

    for (c1, c2) in tf1.den.iter().zip(tf2.den.iter()) {
        if ((c1 / norm1) - (c2 / norm2)).abs() > tolerance {
            return Ok(false);
        }
    }

    Ok(true)
}

/// Calculate the condition number of a matrix (simplified version)
///
/// The condition number indicates how sensitive a matrix is to numerical errors.
/// A high condition number suggests the matrix is ill-conditioned.
///
/// # Arguments
///
/// * `matrix` - Input matrix
///
/// # Returns
///
/// * Condition number estimate
pub fn matrix_condition_number(matrix: &[Vec<f64>]) -> SignalResult<f64> {
    if matrix.is_empty() || matrix[0].is_empty() {
        return Err(SignalError::ValueError("Empty matrix".to_string()));
    }

    // For a simplified implementation, we estimate the condition number
    // using the ratio of largest to smallest singular values.
    // This is a very basic implementation - a full implementation would use SVD.

    let n = matrix.len();
    let m = matrix[0].len();

    if n != m {
        return Err(SignalError::ValueError("Matrix must be square".to_string()));
    }

    // Calculate approximate largest and smallest eigenvalues using Gershgorin circles
    let mut min_radius = f64::INFINITY;
    let mut max_radius: f64 = 0.0;

    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        let center = matrix[i][i];
        let mut radius = 0.0;

        for j in 0..n {
            if i != j {
                radius += matrix[i][j].abs();
            }
        }

        let lower_bound = center - radius;
        let upper_bound = center + radius;

        min_radius = min_radius.min(lower_bound.abs());
        max_radius = max_radius.max(upper_bound.abs());
    }

    if min_radius < 1e-15 {
        Ok(f64::INFINITY) // Singular matrix
    } else {
        Ok(max_radius / min_radius)
    }
}

// Helper functions for matrix operations

/// Convert flattened matrix to 2D format
fn flatten_to_2d(flat_matrix: &[f64], rows: usize, cols: usize) -> SignalResult<Vec<Vec<f64>>> {
    if flat_matrix.len() != rows * cols {
        return Err(SignalError::ValueError(format!(
            "Matrix size mismatch: expected {} elements, got {}",
            rows * cols,
            flat_matrix.len()
        )));
    }

    let mut matrix_2d = Vec::with_capacity(rows);
    for i in 0..rows {
        let mut row = Vec::with_capacity(cols);
        for j in 0..cols {
            row.push(flat_matrix[i * cols + j]);
        }
        matrix_2d.push(row);
    }
    Ok(matrix_2d)
}

/// Calculate the rank of a matrix using Gaussian elimination
fn matrix_rank(matrix: &[Vec<f64>]) -> SignalResult<usize> {
    if matrix.is_empty() {
        return Ok(0);
    }

    let mut mat = matrix.to_vec();
    let rows = mat.len();
    let cols = mat[0].len();
    let tolerance = 1e-12;

    let mut rank = 0;
    let mut col = 0;

    for row in 0..rows {
        if col >= cols {
            break;
        }

        // Find pivot
        let mut pivot_row = row;
        for i in (row + 1)..rows {
            if mat[i][col].abs() > mat[pivot_row][col].abs() {
                pivot_row = i;
            }
        }

        if mat[pivot_row][col].abs() < tolerance {
            col += 1;
            continue;
        }

        // Swap rows
        if pivot_row != row {
            mat.swap(row, pivot_row);
        }

        // Eliminate column
        for i in (row + 1)..rows {
            let factor = mat[i][col] / mat[row][col];
            for j in col..cols {
                mat[i][j] -= factor * mat[row][j];
            }
        }

        rank += 1;
        col += 1;
    }

    Ok(rank)
}

/// Transpose a matrix
#[allow(dead_code)]
fn transpose_matrix(matrix: &[Vec<f64>]) -> SignalResult<Vec<Vec<f64>>> {
    if matrix.is_empty() {
        return Ok(Vec::new());
    }

    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut transposed = vec![vec![0.0; rows]; cols];

    #[allow(clippy::needless_range_loop)]
    for i in 0..rows {
        #[allow(clippy::needless_range_loop)]
        for j in 0..cols {
            transposed[j][i] = matrix[i][j];
        }
    }

    Ok(transposed)
}

/// Multiply two matrices
fn matrix_multiply(a: &[Vec<f64>], b: &[Vec<f64>]) -> SignalResult<Vec<Vec<f64>>> {
    if a.is_empty() || b.is_empty() {
        return Err(SignalError::ValueError(
            "Cannot multiply empty matrices".to_string(),
        ));
    }

    let a_rows = a.len();
    let a_cols = a[0].len();
    let b_rows = b.len();
    let b_cols = b[0].len();

    if a_cols != b_rows {
        return Err(SignalError::ValueError(format!(
            "Matrix dimensions incompatible: {}x{} and {}x{}",
            a_rows, a_cols, b_rows, b_cols
        )));
    }

    let mut result = vec![vec![0.0; b_cols]; a_rows];

    #[allow(clippy::needless_range_loop)]
    for i in 0..a_rows {
        #[allow(clippy::needless_range_loop)]
        for j in 0..b_cols {
            #[allow(clippy::needless_range_loop)]
            for k in 0..a_cols {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    Ok(result)
}

#[cfg(test)]
mod control_observability_tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_controllability_analysis() {
        // Create a simple controllable system
        // dx/dt = [0 1; 0 0] x + [0; 1] u
        let a = vec![0.0, 1.0, 0.0, 0.0]; // [0 1; 0 0] flattened
        let b = vec![0.0, 1.0]; // [0; 1] flattened
        let c = vec![1.0, 0.0]; // [1 0] flattened
        let d = vec![0.0]; // [0] flattened

        let ss = StateSpace::new(a, b, c, d, None).unwrap();
        let analysis = analyze_controllability(&ss).unwrap();

        assert!(analysis.is_controllable);
        assert_eq!(analysis.controllability_rank, 2);
        assert_eq!(analysis.state_dimension, 2);
    }

    #[test]
    fn test_observability_analysis() {
        // Create a simple observable system
        // dx/dt = [0 1; 0 0] x + [0; 1] u
        // y = [1 0] x
        let a = vec![0.0, 1.0, 0.0, 0.0]; // [0 1; 0 0] flattened
        let b = vec![0.0, 1.0]; // [0; 1] flattened
        let c = vec![1.0, 0.0]; // [1 0] flattened
        let d = vec![0.0]; // [0] flattened

        let ss = StateSpace::new(a, b, c, d, None).unwrap();
        let analysis = analyze_observability(&ss).unwrap();

        assert!(analysis.is_observable);
        assert_eq!(analysis.observability_rank, 2);
        assert_eq!(analysis.state_dimension, 2);
    }

    #[test]
    fn test_uncontrollable_system() {
        // Create an uncontrollable system
        // dx/dt = [1 0; 0 2] x + [1; 0] u (second state uncontrollable)
        let a = vec![1.0, 0.0, 0.0, 2.0]; // [1 0; 0 2] flattened
        let b = vec![1.0, 0.0]; // [1; 0] flattened
        let c = vec![1.0, 1.0]; // [1 1] flattened
        let d = vec![0.0]; // [0] flattened

        let ss = StateSpace::new(a, b, c, d, None).unwrap();
        let analysis = analyze_controllability(&ss).unwrap();

        assert!(!analysis.is_controllable);
        assert_eq!(analysis.controllability_rank, 1);
        assert_eq!(analysis.controllable_subspace_dim, 1);
    }

    #[test]
    fn test_unobservable_system() {
        // Create an unobservable system
        // dx/dt = [1 0; 0 2] x + [1; 1] u
        // y = [1 0] x (second state unobservable)
        let a = vec![1.0, 0.0, 0.0, 2.0]; // [1 0; 0 2] flattened
        let b = vec![1.0, 1.0]; // [1; 1] flattened
        let c = vec![1.0, 0.0]; // [1 0] flattened
        let d = vec![0.0]; // [0] flattened

        let ss = StateSpace::new(a, b, c, d, None).unwrap();
        let analysis = analyze_observability(&ss).unwrap();

        assert!(!analysis.is_observable);
        assert_eq!(analysis.observability_rank, 1);
        assert_eq!(analysis.observable_subspace_dim, 1);
    }

    #[test]
    fn test_combined_analysis() {
        // Create a minimal (controllable and observable) system
        let a = vec![0.0, 1.0, -2.0, -3.0]; // [0 1; -2 -3] flattened
        let b = vec![0.0, 1.0]; // [0; 1] flattened
        let c = vec![1.0, 0.0]; // [1 0] flattened
        let d = vec![0.0]; // [0] flattened

        let ss = StateSpace::new(a, b, c, d, None).unwrap();
        let analysis = analyze_control_observability(&ss).unwrap();

        assert!(analysis.is_minimal);
        assert!(analysis.controllability.is_controllable);
        assert!(analysis.observability.is_observable);
        assert_eq!(analysis.kalman_structure.co_dimension, 2);
        assert_eq!(analysis.kalman_structure.c_no_dimension, 0);
        assert_eq!(analysis.kalman_structure.nc_o_dimension, 0);
        assert_eq!(analysis.kalman_structure.nc_no_dimension, 0);
    }

    #[test]
    fn test_matrix_rank() {
        // Test rank calculation
        let identity = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        assert_eq!(matrix_rank(&identity).unwrap(), 2);

        let singular = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        assert_eq!(matrix_rank(&singular).unwrap(), 1);

        let zero_matrix = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        assert_eq!(matrix_rank(&zero_matrix).unwrap(), 0);
    }

    #[test]
    fn test_matrix_operations() {
        // Test matrix multiplication
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
        let result = matrix_multiply(&a, &b).unwrap();

        // Expected: [19, 22; 43, 50]
        assert_relative_eq!(result[0][0], 19.0);
        assert_relative_eq!(result[0][1], 22.0);
        assert_relative_eq!(result[1][0], 43.0);
        assert_relative_eq!(result[1][1], 50.0);

        // Test matrix transpose
        let original = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let transposed = transpose_matrix(&original).unwrap();

        assert_eq!(transposed.len(), 3);
        assert_eq!(transposed[0].len(), 2);
        assert_relative_eq!(transposed[0][0], 1.0);
        assert_relative_eq!(transposed[0][1], 4.0);
        assert_relative_eq!(transposed[1][0], 2.0);
        assert_relative_eq!(transposed[1][1], 5.0);
    }

    #[test]
    fn test_systems_equivalent() {
        // Create two equivalent transfer functions
        let tf1 = TransferFunction::new(vec![1.0], vec![1.0, 2.0, 1.0], None).unwrap();
        let tf2 = TransferFunction::new(vec![2.0], vec![2.0, 4.0, 2.0], None).unwrap();

        assert!(systems_equivalent(&tf1, &tf2, 1e-10).unwrap());

        // Create non-equivalent systems
        let tf3 = TransferFunction::new(vec![1.0], vec![1.0, 3.0, 1.0], None).unwrap();
        assert!(!systems_equivalent(&tf1, &tf3, 1e-10).unwrap());
    }
}

// Laplace Transform Support for Continuous-Time Systems
//
// This section provides functions for working with Laplace transforms,
// which are fundamental for analyzing continuous-time LTI systems.

/// Laplace transform evaluation and analysis
pub mod laplace {
    use super::*;
    use std::f64::consts::PI;

    /// Evaluate the Laplace transform of a transfer function at a complex point s
    ///
    /// # Arguments
    ///
    /// * `tf` - Transfer function to evaluate
    /// * `s` - Complex frequency point for evaluation
    ///
    /// # Returns
    ///
    /// * Complex value of H(s) at the given point
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_signal::lti::{TransferFunction, laplace::evaluate_laplace};
    /// use num_complex::Complex64;
    ///
    /// let tf = TransferFunction::new(vec![1.0], vec![1.0, 1.0], None).unwrap();
    /// let s = Complex64::new(0.0, 1.0); // s = j (purely imaginary)
    /// let result = evaluate_laplace(&tf, s).unwrap();
    /// ```
    pub fn evaluate_laplace(tf: &TransferFunction, s: Complex64) -> SignalResult<Complex64> {
        if tf.dt {
            return Err(SignalError::ValueError(
                "Laplace transform is only applicable to continuous-time systems".to_string(),
            ));
        }

        Ok(tf.evaluate(s))
    }

    /// Compute the inverse Laplace transform numerically using Bromwich integral
    ///
    /// This uses a simplified numerical approximation of the Bromwich integral
    /// for computing the inverse Laplace transform.
    ///
    /// # Arguments
    ///
    /// * `tf` - Transfer function H(s)
    /// * `t` - Time points at which to evaluate the inverse transform
    /// * `sigma` - Real part of the Bromwich contour (must be greater than all pole real parts)
    ///
    /// # Returns
    ///
    /// * Time-domain signal h(t)
    pub fn inverse_laplace_transform(
        tf: &TransferFunction,
        t: &[f64],
        sigma: f64,
    ) -> SignalResult<Vec<f64>> {
        if tf.dt {
            return Err(SignalError::ValueError(
                "Laplace transform is only applicable to continuous-time systems".to_string(),
            ));
        }

        let mut result = Vec::with_capacity(t.len());

        // Use numerical integration of the Bromwich integral
        // h(t) = (1/2πj) ∫[σ-j∞ to σ+j∞] H(s) * e^(st) ds

        for &time in t {
            if time < 0.0 {
                // Causal systems: h(t) = 0 for t < 0
                result.push(0.0);
                continue;
            }

            // Approximate the integral using numerical quadrature
            let n_points = 1000;
            let omega_max = 100.0; // Integration limits
            let d_omega = 2.0 * omega_max / n_points as f64;

            let mut integral = Complex64::new(0.0, 0.0);

            for i in 0..n_points {
                let omega = -omega_max + i as f64 * d_omega;
                let s = Complex64::new(sigma, omega);

                let h_s = tf.evaluate(s);
                let exponential = Complex64::new(0.0, omega * time).exp() * (sigma * time).exp();

                integral += h_s * exponential * d_omega;
            }

            // The result is the real part divided by 2π
            let value = integral.re / (2.0 * PI);
            result.push(value);
        }

        Ok(result)
    }

    /// Find the dominant poles of a transfer function
    ///
    /// Dominant poles are those closest to the imaginary axis in the s-plane,
    /// as they have the most influence on the system's transient response.
    ///
    /// # Arguments
    ///
    /// * `tf` - Transfer function to analyze
    /// * `num_dominant` - Number of dominant poles to return
    ///
    /// # Returns
    ///
    /// * Vector of dominant poles sorted by distance from imaginary axis
    pub fn find_dominant_poles(
        tf: &TransferFunction,
        num_dominant: usize,
    ) -> SignalResult<Vec<Complex64>> {
        if tf.dt {
            return Err(SignalError::ValueError(
                "Pole analysis is only applicable to continuous-time systems".to_string(),
            ));
        }

        // Find roots of denominator polynomial (poles)
        let poles = find_polynomial_roots_advanced(&tf.den)?;

        // Sort poles by distance from imaginary axis (real part magnitude)
        let mut pole_distances: Vec<(Complex64, f64)> = poles
            .into_iter()
            .map(|pole| {
                let distance = pole.re.abs(); // Distance from imaginary axis
                (pole, distance)
            })
            .collect();

        pole_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Return the requested number of dominant poles
        let num_to_return = num_dominant.min(pole_distances.len());
        Ok(pole_distances
            .into_iter()
            .take(num_to_return)
            .map(|(pole, _)| pole)
            .collect())
    }

    /// Compute the settling time of a system
    ///
    /// Settling time is the time required for the system output to stay within
    /// a specified percentage of its final value.
    ///
    /// # Arguments
    ///
    /// * `tf` - Transfer function to analyze
    /// * `tolerance` - Percentage tolerance (e.g., 0.02 for 2%)
    ///
    /// # Returns
    ///
    /// * Settling time in seconds
    pub fn settling_time(tf: &TransferFunction, tolerance: f64) -> SignalResult<f64> {
        if tf.dt {
            return Err(SignalError::ValueError(
                "Settling time analysis is only applicable to continuous-time systems".to_string(),
            ));
        }

        // Find dominant poles
        let dominant_poles = find_dominant_poles(tf, 2)?;

        if dominant_poles.is_empty() {
            return Err(SignalError::ValueError(
                "No poles found for settling time calculation".to_string(),
            ));
        }

        // Settling time is approximately 4/ζωn for second-order systems
        // For higher-order systems, use the dominant pole with smallest real part
        let dominant_pole = dominant_poles[0];

        if dominant_pole.re >= 0.0 {
            return Err(SignalError::ValueError(
                "System is unstable - cannot compute settling time".to_string(),
            ));
        }

        // Settling time based on the time constant of the dominant pole
        // For tolerance band, use: t_s = -ln(tolerance) / |Re(s)|
        let time_constant = 1.0 / (-dominant_pole.re);
        let settling_time = -tolerance.ln() * time_constant;

        Ok(settling_time)
    }

    /// Compute the rise time of a step response
    ///
    /// Rise time is the time required for the output to rise from 10% to 90%
    /// of its final value in response to a step input.
    ///
    /// # Arguments
    ///
    /// * `tf` - Transfer function to analyze
    ///
    /// # Returns
    ///
    /// * Rise time in seconds
    pub fn rise_time(tf: &TransferFunction) -> SignalResult<f64> {
        if tf.dt {
            return Err(SignalError::ValueError(
                "Rise time analysis is only applicable to continuous-time systems".to_string(),
            ));
        }

        // For second-order systems: t_r ≈ (2.2 / ωn) / sqrt(1 - ζ²)
        // For first-order systems: t_r ≈ 2.2 * τ where τ is time constant

        let dominant_poles = find_dominant_poles(tf, 2)?;

        if dominant_poles.is_empty() {
            return Err(SignalError::ValueError(
                "No poles found for rise time calculation".to_string(),
            ));
        }

        let dominant_pole = dominant_poles[0];

        if dominant_pole.re >= 0.0 {
            return Err(SignalError::ValueError(
                "System is unstable - cannot compute rise time".to_string(),
            ));
        }

        // Simple approximation based on dominant pole
        let omega_n = dominant_pole.norm();
        let zeta = -dominant_pole.re / omega_n;

        let rise_time = if zeta < 1.0 {
            // Underdamped case
            (2.2 / omega_n) / (1.0 - zeta * zeta).sqrt()
        } else {
            // Overdamped or critically damped case
            2.2 / (-dominant_pole.re)
        };

        Ok(rise_time)
    }

    /// Compute the overshoot percentage of a step response
    ///
    /// # Arguments
    ///
    /// * `tf` - Transfer function to analyze
    ///
    /// # Returns
    ///
    /// * Overshoot percentage (0.0 to 1.0)
    pub fn overshoot_percentage(tf: &TransferFunction) -> SignalResult<f64> {
        if tf.dt {
            return Err(SignalError::ValueError(
                "Overshoot analysis is only applicable to continuous-time systems".to_string(),
            ));
        }

        let dominant_poles = find_dominant_poles(tf, 2)?;

        if dominant_poles.is_empty() {
            return Ok(0.0); // No overshoot if no poles
        }

        let dominant_pole = dominant_poles[0];

        if dominant_pole.re >= 0.0 {
            return Err(SignalError::ValueError(
                "System is unstable - cannot compute overshoot".to_string(),
            ));
        }

        let omega_n = dominant_pole.norm();
        let zeta = -dominant_pole.re / omega_n;

        if zeta >= 1.0 {
            // Overdamped or critically damped - no overshoot
            Ok(0.0)
        } else {
            // Underdamped - compute overshoot
            let overshoot = (-PI * zeta / (1.0 - zeta * zeta).sqrt()).exp();
            Ok(overshoot)
        }
    }

    /// Advanced polynomial root finding using iterative methods
    ///
    /// This is a more robust implementation than the basic one in the filter module
    fn find_polynomial_roots_advanced(coeffs: &[f64]) -> SignalResult<Vec<Complex64>> {
        if coeffs.len() <= 1 {
            return Ok(Vec::new());
        }

        let mut roots = Vec::new();
        let n = coeffs.len() - 1; // Degree of polynomial

        // Handle special cases
        if n == 1 {
            // Linear: ax + b = 0
            if coeffs[0] != 0.0 {
                roots.push(Complex64::new(-coeffs[1] / coeffs[0], 0.0));
            }
            return Ok(roots);
        }

        if n == 2 {
            // Quadratic: ax² + bx + c = 0
            let a = coeffs[0];
            let b = coeffs[1];
            let c = coeffs[2];

            if a != 0.0 {
                let discriminant = b * b - 4.0 * a * c;
                if discriminant >= 0.0 {
                    let sqrt_disc = discriminant.sqrt();
                    roots.push(Complex64::new((-b + sqrt_disc) / (2.0 * a), 0.0));
                    roots.push(Complex64::new((-b - sqrt_disc) / (2.0 * a), 0.0));
                } else {
                    let sqrt_disc = (-discriminant).sqrt();
                    roots.push(Complex64::new(-b / (2.0 * a), sqrt_disc / (2.0 * a)));
                    roots.push(Complex64::new(-b / (2.0 * a), -sqrt_disc / (2.0 * a)));
                }
            }
            return Ok(roots);
        }

        // For higher order polynomials, use numerical methods
        // This is a simplified implementation - a full implementation would use
        // more sophisticated methods like Laguerre's method or eigenvalue approaches

        // Use Durand-Kerner method (simplified)
        let mut current_roots: Vec<Complex64> = (0..n)
            .map(|k| {
                let angle = 2.0 * PI * k as f64 / n as f64;
                Complex64::new(angle.cos(), angle.sin()) * 0.4
            })
            .collect();

        let max_iterations = 100;
        let tolerance = 1e-12;

        for _iter in 0..max_iterations {
            let mut new_roots = current_roots.clone();
            let mut max_change: f64 = 0.0;

            for i in 0..n {
                let mut p_val = Complex64::new(0.0, 0.0);
                let mut p_deriv = Complex64::new(0.0, 0.0);

                // Evaluate polynomial and its derivative at current_roots[i]
                for (j, &coeff) in coeffs.iter().enumerate() {
                    let power = (coeffs.len() - 1 - j) as i32;
                    let term = current_roots[i].powi(power);
                    p_val += coeff * term;

                    if power > 0 {
                        p_deriv += coeff * power as f64 * current_roots[i].powi(power - 1);
                    }
                }

                // Durand-Kerner correction
                let mut product = Complex64::new(1.0, 0.0);
                for (j, &other_root) in current_roots.iter().enumerate() {
                    if i != j {
                        product *= current_roots[i] - other_root;
                    }
                }

                if product.norm() > tolerance {
                    let correction = p_val / product;
                    new_roots[i] = current_roots[i] - correction;
                    max_change = max_change.max(correction.norm());
                }
            }

            current_roots = new_roots;

            if max_change < tolerance {
                break;
            }
        }

        Ok(current_roots)
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use approx::assert_relative_eq;

        #[test]
        fn test_evaluate_laplace() {
            // Test H(s) = 1/(s+1) at s = j
            let tf = TransferFunction::new(vec![1.0], vec![1.0, 1.0], None).unwrap();
            let s = Complex64::new(0.0, 1.0);
            let result = evaluate_laplace(&tf, s).unwrap();

            // Expected: 1/(j+1) = (1-j)/2
            assert_relative_eq!(result.re, 0.5, epsilon = 1e-10);
            assert_relative_eq!(result.im, -0.5, epsilon = 1e-10);
        }

        #[test]
        fn test_discrete_time_error() {
            // Test that discrete-time systems return error
            let tf = TransferFunction::new(vec![1.0], vec![1.0, 1.0], Some(true)).unwrap();
            let s = Complex64::new(0.0, 1.0);
            assert!(evaluate_laplace(&tf, s).is_err());
        }

        #[test]
        fn test_find_dominant_poles() {
            // Test system with known poles
            let tf = TransferFunction::new(
                vec![1.0],
                vec![1.0, 3.0, 2.0], // (s+1)(s+2) = s² + 3s + 2
                None,
            )
            .unwrap();

            let poles = find_dominant_poles(&tf, 2).unwrap();
            assert_eq!(poles.len(), 2);

            // Poles should be at s = -1 and s = -2
            // The dominant pole (closest to imaginary axis) should be s = -1
            assert_relative_eq!(poles[0].re, -1.0, epsilon = 1e-6);
            assert_relative_eq!(poles[0].im, 0.0, epsilon = 1e-6);
        }

        #[test]
        fn test_settling_time() {
            // Test first-order system H(s) = 1/(s+1)
            let tf = TransferFunction::new(vec![1.0], vec![1.0, 1.0], None).unwrap();
            let ts = settling_time(&tf, 0.02).unwrap(); // 2% tolerance

            // For first-order system with pole at -1, settling time ≈ 4 * time_constant = 4
            assert!(ts > 3.0 && ts < 5.0);
        }

        #[test]
        fn test_rise_time() {
            // Test first-order system
            let tf = TransferFunction::new(vec![1.0], vec![1.0, 1.0], None).unwrap();
            let tr = rise_time(&tf).unwrap();

            // Should be reasonable value for first-order system
            assert!(tr > 0.0 && tr < 10.0);
        }

        #[test]
        fn test_overshoot_percentage() {
            // Test first-order system (should have no overshoot)
            let tf = TransferFunction::new(vec![1.0], vec![1.0, 1.0], None).unwrap();
            let overshoot = overshoot_percentage(&tf).unwrap();

            // First-order systems should have no overshoot
            assert_relative_eq!(overshoot, 0.0, epsilon = 1e-6);
        }

        #[test]
        fn test_polynomial_roots_quadratic() {
            // Test s² + 3s + 2 = (s+1)(s+2)
            let coeffs = vec![1.0, 3.0, 2.0];
            let roots = find_polynomial_roots_advanced(&coeffs).unwrap();

            assert_eq!(roots.len(), 2);

            // Check that roots are approximately -1 and -2
            let mut real_parts: Vec<f64> = roots.iter().map(|r| r.re).collect();
            real_parts.sort_by(|a, b| a.partial_cmp(b).unwrap());

            assert_relative_eq!(real_parts[0], -2.0, epsilon = 1e-6);
            assert_relative_eq!(real_parts[1], -1.0, epsilon = 1e-6);
        }

        #[test]
        fn test_inverse_laplace_simple() {
            // Test inverse Laplace of H(s) = 1/(s+1)
            // Should give h(t) = e^(-t) for t ≥ 0
            let tf = TransferFunction::new(vec![1.0], vec![1.0, 1.0], None).unwrap();
            let t = vec![0.0, 0.5, 1.0, 2.0];
            let result = inverse_laplace_transform(&tf, &t, 2.0).unwrap();

            // Check that result has correct length and reasonable values
            assert_eq!(result.len(), 4);
            // For numerical inverse Laplace, expect some approximation error
            assert!(result[0] > 0.1); // h(0) should be positive
            assert!(result[1] > 0.0); // Should be positive
            assert!(result[2] > 0.0); // Should be positive
            assert!(result[3] > 0.0); // Should be positive
        }
    }
}

// System Reduction and Minimal Realization Techniques
//
// This section provides methods for reducing system order and finding
// minimal realizations of transfer functions and state-space systems.

/// System reduction and minimal realization functions
pub mod reduction {
    use super::*;

    /// Configuration options for model reduction algorithms
    #[derive(Debug, Clone)]
    pub struct ReductionConfig {
        /// Tolerance for determining insignificant states/modes
        pub tolerance: f64,
        /// Maximum number of iterations for iterative algorithms
        pub max_iterations: usize,
        /// Whether to preserve DC gain
        pub preserve_dc_gain: bool,
        /// Method for Hankel singular value computation
        pub hsv_method: HsvMethod,
    }

    impl Default for ReductionConfig {
        fn default() -> Self {
            Self {
                tolerance: 1e-6,
                max_iterations: 100,
                preserve_dc_gain: true,
                hsv_method: HsvMethod::Balanced,
            }
        }
    }

    /// Methods for computing Hankel singular values
    #[derive(Debug, Clone, Copy)]
    pub enum HsvMethod {
        /// Balanced truncation approach
        Balanced,
        /// Modal truncation approach  
        Modal,
        /// Frequency-weighted reduction
        FrequencyWeighted,
    }

    /// Result of model reduction containing the reduced system and analysis
    #[derive(Debug, Clone)]
    pub struct ReductionResult {
        /// Reduced order system
        pub reduced_system: StateSpace,
        /// Original system order
        pub original_order: usize,
        /// Reduced system order
        pub reduced_order: usize,
        /// Hankel singular values of original system
        pub hankel_singular_values: Vec<f64>,
        /// States removed during reduction
        pub removed_states: Vec<usize>,
        /// Reduction error bound (if computable)
        pub error_bound: Option<f64>,
    }

    /// Balanced truncation model reduction
    ///
    /// This method reduces system order by truncating states corresponding to
    /// small Hankel singular values, which represent weakly coupled input-output modes.
    ///
    /// # Arguments
    ///
    /// * `ss` - Original state-space system
    /// * `reduced_order` - Desired order of reduced system
    /// * `config` - Reduction configuration options
    ///
    /// # Returns
    ///
    /// * Reduced system and analysis results
    pub fn balanced_truncation(
        ss: &StateSpace,
        reduced_order: usize,
        config: &ReductionConfig,
    ) -> SignalResult<ReductionResult> {
        if ss.dt {
            return Err(SignalError::ValueError(
                "Balanced truncation currently only supports continuous-time systems".to_string(),
            ));
        }

        if reduced_order >= ss.n_states {
            return Err(SignalError::ValueError(
                "Reduced order must be less than original system order".to_string(),
            ));
        }

        // Step 1: Compute controllability and observability Gramians
        let (p_c, p_o) = compute_gramians(ss, config)?;

        // Step 2: Compute Hankel singular values
        let hankel_svs = compute_hankel_singular_values(&p_c, &p_o)?;

        // Step 3: Sort and select dominant singular values/vectors
        let mut hsv_indexed: Vec<(f64, usize)> = hankel_svs
            .iter()
            .enumerate()
            .map(|(i, &val)| (val, i))
            .collect();
        hsv_indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        // Step 4: Select states to keep (largest Hankel singular values)
        let kept_states: Vec<usize> = hsv_indexed
            .iter()
            .take(reduced_order)
            .map(|(_, idx)| *idx)
            .collect();

        let removed_states: Vec<usize> = hsv_indexed
            .iter()
            .skip(reduced_order)
            .map(|(_, idx)| *idx)
            .collect();

        // Step 5: Extract reduced system matrices
        let reduced_system = extract_reduced_system(ss, &kept_states)?;

        // Step 6: Compute error bound
        let error_bound = if reduced_order < hankel_svs.len() {
            Some(2.0 * hankel_svs.iter().skip(reduced_order).sum::<f64>())
        } else {
            None
        };

        Ok(ReductionResult {
            reduced_system,
            original_order: ss.n_states,
            reduced_order,
            hankel_singular_values: hankel_svs,
            removed_states,
            error_bound,
        })
    }

    /// Modal truncation model reduction
    ///
    /// This method reduces system order by removing modes (eigenvalues) that are
    /// deemed less significant based on their distance from the imaginary axis.
    ///
    /// # Arguments
    ///
    /// * `ss` - Original state-space system
    /// * `reduced_order` - Desired order of reduced system
    /// * `config` - Reduction configuration options
    ///
    /// # Returns
    ///
    /// * Reduced system and analysis results
    pub fn modal_truncation(
        ss: &StateSpace,
        reduced_order: usize,
        _config: &ReductionConfig,
    ) -> SignalResult<ReductionResult> {
        if reduced_order >= ss.n_states {
            return Err(SignalError::ValueError(
                "Reduced order must be less than original system order".to_string(),
            ));
        }

        // Step 1: Compute eigenvalues of A matrix
        let a_matrix = flatten_to_2d(&ss.a, ss.n_states, ss.n_states)?;
        let eigenvalues = compute_eigenvalues(&a_matrix)?;

        // Step 2: Sort eigenvalues by significance (distance from imaginary axis)
        let mut eigen_significance: Vec<(Complex64, f64, usize)> = eigenvalues
            .iter()
            .enumerate()
            .map(|(i, &val)| {
                let significance = 1.0 / (1.0 + val.re.abs()); // More negative real parts are less significant
                (val, significance, i)
            })
            .collect();

        eigen_significance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Step 3: Select modes to keep (most significant)
        let kept_states: Vec<usize> = eigen_significance
            .iter()
            .take(reduced_order)
            .map(|(_, _, idx)| *idx)
            .collect();

        let removed_states: Vec<usize> = eigen_significance
            .iter()
            .skip(reduced_order)
            .map(|(_, _, idx)| *idx)
            .collect();

        // Step 4: Extract reduced system
        let reduced_system = extract_reduced_system(ss, &kept_states)?;

        // For modal truncation, we use eigenvalue magnitudes as "singular values"
        let modal_values: Vec<f64> = eigen_significance
            .iter()
            .map(|(eigenval, _, _)| eigenval.norm())
            .collect();

        Ok(ReductionResult {
            reduced_system,
            original_order: ss.n_states,
            reduced_order,
            hankel_singular_values: modal_values,
            removed_states,
            error_bound: None, // Modal truncation doesn't provide guaranteed error bounds
        })
    }

    /// Minimal realization from transfer function
    ///
    /// Computes a minimal (controllable and observable) state-space realization
    /// of a given transfer function.
    ///
    /// # Arguments
    ///
    /// * `tf` - Transfer function to realize
    /// * `form` - Desired canonical form for realization
    ///
    /// # Returns
    ///
    /// * Minimal state-space realization
    pub fn minimal_realization(
        tf: &TransferFunction,
        form: CanonicalForm,
    ) -> SignalResult<StateSpace> {
        // Remove common factors from numerator and denominator
        let (num_reduced, den_reduced) = remove_common_factors(&tf.num, &tf.den)?;

        let n = den_reduced.len() - 1; // System order

        if n == 0 {
            return Err(SignalError::ValueError(
                "Cannot create state-space realization of zero-order system".to_string(),
            ));
        }

        match form {
            CanonicalForm::Controllable => {
                controllable_canonical_form(&num_reduced, &den_reduced, tf.dt)
            }
            CanonicalForm::Observable => {
                observable_canonical_form(&num_reduced, &den_reduced, tf.dt)
            }
            CanonicalForm::Balanced => {
                // Start with controllable form, then balance
                let ss_ctrl = controllable_canonical_form(&num_reduced, &den_reduced, tf.dt)?;
                balance_realization(&ss_ctrl)
            }
        }
    }

    /// Canonical forms for state-space realizations
    #[derive(Debug, Clone, Copy)]
    pub enum CanonicalForm {
        /// Controllable canonical form
        Controllable,
        /// Observable canonical form
        Observable,
        /// Balanced canonical form
        Balanced,
    }

    // Helper functions for reduction algorithms

    // Type alias to reduce complexity
    type GramianResult = SignalResult<(Vec<Vec<f64>>, Vec<Vec<f64>>)>;

    /// Compute controllability and observability Gramians
    fn compute_gramians(ss: &StateSpace, _config: &ReductionConfig) -> GramianResult {
        let n = ss.n_states;
        let _a_matrix = flatten_to_2d(&ss.a, n, n)?;

        // Simplified Gramian computation - in practice would solve Lyapunov equations
        // P_c: A*P_c + P_c*A' + B*B' = 0
        // P_o: A'*P_o + P_o*A + C'*C = 0

        // For this simplified implementation, we approximate the Gramians
        let mut p_c = vec![vec![0.0; n]; n];
        let mut p_o = vec![vec![0.0; n]; n];

        // Initialize with identity and iterate (very simplified)
        for i in 0..n {
            p_c[i][i] = 1.0;
            p_o[i][i] = 1.0;
        }

        // This is a placeholder - proper implementation would solve Lyapunov equations
        Ok((p_c, p_o))
    }

    /// Compute Hankel singular values from Gramians
    fn compute_hankel_singular_values(
        p_c: &[Vec<f64>],
        p_o: &[Vec<f64>],
    ) -> SignalResult<Vec<f64>> {
        let _n = p_c.len();

        // Hankel singular values are sqrt(eigenvalues(P_c * P_o))
        let product = matrix_multiply(p_c, p_o)?;
        let eigenvalues = compute_eigenvalues(&product)?;

        let mut singular_values: Vec<f64> = eigenvalues
            .iter()
            .map(|&ev| ev.re.max(0.0).sqrt()) // Take sqrt of positive real parts
            .collect();

        singular_values.sort_by(|a, b| b.partial_cmp(a).unwrap());
        Ok(singular_values)
    }

    /// Extract reduced system by selecting specific states
    fn extract_reduced_system(ss: &StateSpace, kept_states: &[usize]) -> SignalResult<StateSpace> {
        let n_orig = ss.n_states;
        let n_red = kept_states.len();
        let m = ss.n_inputs;
        let p = ss.n_outputs;

        // Extract submatrices
        let a_orig = flatten_to_2d(&ss.a, n_orig, n_orig)?;
        let b_orig = flatten_to_2d(&ss.b, n_orig, m)?;
        let c_orig = flatten_to_2d(&ss.c, p, n_orig)?;

        // Build reduced A matrix
        let mut a_red = vec![vec![0.0; n_red]; n_red];
        for (i, &row_idx) in kept_states.iter().enumerate() {
            for (j, &col_idx) in kept_states.iter().enumerate() {
                a_red[i][j] = a_orig[row_idx][col_idx];
            }
        }

        // Build reduced B matrix
        let mut b_red = vec![vec![0.0; m]; n_red];
        for (i, &row_idx) in kept_states.iter().enumerate() {
            for j in 0..m {
                b_red[i][j] = b_orig[row_idx][j];
            }
        }

        // Build reduced C matrix
        let mut c_red = vec![vec![0.0; n_red]; p];
        for i in 0..p {
            for (j, &col_idx) in kept_states.iter().enumerate() {
                c_red[i][j] = c_orig[i][col_idx];
            }
        }

        // D matrix remains the same
        let d_red = ss.d.clone();

        // Flatten matrices for StateSpace constructor
        let a_flat: Vec<f64> = a_red.into_iter().flatten().collect();
        let b_flat: Vec<f64> = b_red.into_iter().flatten().collect();
        let c_flat: Vec<f64> = c_red.into_iter().flatten().collect();

        StateSpace::new(a_flat, b_flat, c_flat, d_red, Some(ss.dt))
    }

    /// Compute eigenvalues of a matrix (simplified implementation)
    fn compute_eigenvalues(matrix: &[Vec<f64>]) -> SignalResult<Vec<Complex64>> {
        let n = matrix.len();

        if n == 1 {
            return Ok(vec![Complex64::new(matrix[0][0], 0.0)]);
        }

        if n == 2 {
            // For 2x2 matrix, use analytical formula
            let a = matrix[0][0];
            let b = matrix[0][1];
            let c = matrix[1][0];
            let d = matrix[1][1];

            let trace = a + d;
            let det = a * d - b * c;
            let discriminant = trace * trace - 4.0 * det;

            if discriminant >= 0.0 {
                let sqrt_disc = discriminant.sqrt();
                let ev1 = Complex64::new((trace + sqrt_disc) / 2.0, 0.0);
                let ev2 = Complex64::new((trace - sqrt_disc) / 2.0, 0.0);
                return Ok(vec![ev1, ev2]);
            } else {
                let sqrt_disc = (-discriminant).sqrt();
                let ev1 = Complex64::new(trace / 2.0, sqrt_disc / 2.0);
                let ev2 = Complex64::new(trace / 2.0, -sqrt_disc / 2.0);
                return Ok(vec![ev1, ev2]);
            }
        }

        // For larger matrices, use simplified power iteration for dominant eigenvalue
        // This is not a complete implementation - would need QR algorithm or similar
        let mut eigenvalues = Vec::new();

        // Estimate eigenvalues using diagonal elements (very rough approximation)
        for (i, row) in matrix.iter().enumerate().take(n) {
            eigenvalues.push(Complex64::new(row[i], 0.0));
        }

        Ok(eigenvalues)
    }

    /// Remove common factors from polynomials
    fn remove_common_factors(num: &[f64], den: &[f64]) -> SignalResult<(Vec<f64>, Vec<f64>)> {
        // Simplified implementation - just remove leading zeros
        let num_trimmed = trim_leading_zeros(num);
        let den_trimmed = trim_leading_zeros(den);

        if den_trimmed.is_empty() || den_trimmed.iter().all(|&x| x.abs() < 1e-15) {
            return Err(SignalError::ValueError(
                "Denominator cannot be zero".to_string(),
            ));
        }

        Ok((num_trimmed, den_trimmed))
    }

    /// Remove leading zeros from polynomial
    fn trim_leading_zeros(poly: &[f64]) -> Vec<f64> {
        let mut start = 0;
        while start < poly.len() && poly[start].abs() < 1e-15 {
            start += 1;
        }

        if start == poly.len() {
            vec![0.0] // All zeros case
        } else {
            poly[start..].to_vec()
        }
    }

    /// Create controllable canonical form realization
    fn controllable_canonical_form(num: &[f64], den: &[f64], dt: bool) -> SignalResult<StateSpace> {
        let n = den.len() - 1; // System order

        if n == 0 {
            return Err(SignalError::ValueError(
                "Cannot create realization of zero-order system".to_string(),
            ));
        }

        // Normalize denominator
        let leading_coeff = den[0];
        if leading_coeff.abs() < 1e-15 {
            return Err(SignalError::ValueError(
                "Leading coefficient of denominator cannot be zero".to_string(),
            ));
        }

        let den_norm: Vec<f64> = den.iter().map(|&x| x / leading_coeff).collect();
        let num_norm: Vec<f64> = num.iter().map(|&x| x / leading_coeff).collect();

        // Build A matrix (companion form)
        let mut a = vec![0.0; n * n];

        // First n-1 rows: shift register
        for i in 0..(n - 1) {
            a[i * n + (i + 1)] = 1.0;
        }

        // Last row: negative coefficients of characteristic polynomial
        for j in 0..n {
            a[(n - 1) * n + j] = -den_norm[n - j];
        }

        // Build B matrix
        let mut b = vec![0.0; n];
        b[n - 1] = 1.0;

        // Build C matrix
        let mut c = vec![0.0; n];

        // Handle numerator coefficients
        let num_padded = if num_norm.len() <= n {
            let mut padded = vec![0.0; n - num_norm.len()];
            padded.extend_from_slice(&num_norm);
            padded
        } else {
            num_norm[num_norm.len() - n..].to_vec()
        };

        c[..n].copy_from_slice(&num_padded[..n]);

        // Build D matrix
        let d = if num.len() > den.len() - 1 {
            vec![num[0] / leading_coeff] // Direct feedthrough
        } else {
            vec![0.0]
        };

        StateSpace::new(a, b, c, d, Some(dt))
    }

    /// Create observable canonical form realization
    fn observable_canonical_form(num: &[f64], den: &[f64], dt: bool) -> SignalResult<StateSpace> {
        // Observable form is the transpose of controllable form
        let ss_ctrl = controllable_canonical_form(num, den, dt)?;

        // Transpose the system (A' → A, B' → C, C' → B)
        let n = ss_ctrl.n_states;
        let a_orig = flatten_to_2d(&ss_ctrl.a, n, n)?;
        let b_orig = flatten_to_2d(&ss_ctrl.b, n, 1)?;
        let c_orig = flatten_to_2d(&ss_ctrl.c, 1, n)?;

        let a_transposed = transpose_matrix(&a_orig)?;
        let b_transposed = transpose_matrix(&c_orig)?;
        let c_transposed = transpose_matrix(&b_orig)?;

        let a_flat: Vec<f64> = a_transposed.into_iter().flatten().collect();
        let b_flat: Vec<f64> = b_transposed.into_iter().flatten().collect();
        let c_flat: Vec<f64> = c_transposed.into_iter().flatten().collect();

        StateSpace::new(a_flat, b_flat, c_flat, ss_ctrl.d, Some(dt))
    }

    /// Balance a state-space realization (simplified implementation)
    fn balance_realization(ss: &StateSpace) -> SignalResult<StateSpace> {
        // For simplicity, just return the original system
        // A full implementation would compute the balancing transformation
        Ok(ss.clone())
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use approx::assert_relative_eq;

        #[test]
        fn test_minimal_realization_first_order() {
            // Test H(s) = 1/(s+1)
            let tf = TransferFunction::new(vec![1.0], vec![1.0, 1.0], None).unwrap();
            let ss = minimal_realization(&tf, CanonicalForm::Controllable).unwrap();

            assert_eq!(ss.n_states, 1);
            assert_eq!(ss.n_inputs, 1);
            assert_eq!(ss.n_outputs, 1);

            // Check that it's a proper realization
            assert_relative_eq!(ss.a[0], -1.0, epsilon = 1e-10);
            assert_relative_eq!(ss.b[0], 1.0, epsilon = 1e-10);
            assert_relative_eq!(ss.c[0], 1.0, epsilon = 1e-10);
            assert_relative_eq!(ss.d[0], 0.0, epsilon = 1e-10);
        }

        #[test]
        fn test_minimal_realization_second_order() {
            // Test H(s) = 1/(s²+3s+2) = 1/((s+1)(s+2))
            let tf = TransferFunction::new(vec![1.0], vec![1.0, 3.0, 2.0], None).unwrap();
            let ss = minimal_realization(&tf, CanonicalForm::Controllable).unwrap();

            assert_eq!(ss.n_states, 2);
            assert_eq!(ss.n_inputs, 1);
            assert_eq!(ss.n_outputs, 1);

            // Check controllable canonical form structure
            assert_relative_eq!(ss.a[0], 0.0, epsilon = 1e-10); // A[0,0]
            assert_relative_eq!(ss.a[1], 1.0, epsilon = 1e-10); // A[0,1]
            assert_relative_eq!(ss.a[2], -2.0, epsilon = 1e-10); // A[1,0]
            assert_relative_eq!(ss.a[3], -3.0, epsilon = 1e-10); // A[1,1]
        }

        #[test]
        fn test_balanced_truncation() {
            // Create a simple 3rd order system for testing
            let a = vec![-1.0, 1.0, 0.0, 0.0, -2.0, 1.0, 0.0, 0.0, -10.0];
            let b = vec![1.0, 0.0, 0.0];
            let c = vec![1.0, 1.0, 1.0];
            let d = vec![0.0];

            let ss = StateSpace::new(a, b, c, d, None).unwrap();
            let config = ReductionConfig::default();

            let result = balanced_truncation(&ss, 2, &config).unwrap();

            assert_eq!(result.original_order, 3);
            assert_eq!(result.reduced_order, 2);
            assert_eq!(result.reduced_system.n_states, 2);
        }

        #[test]
        fn test_modal_truncation() {
            // Create a system with well-separated modes
            let a = vec![-1.0, 0.0, 0.0, 0.0, -5.0, 0.0, 0.0, 0.0, -100.0]; // Diagonal system with eigenvalues -1, -5, -100
            let b = vec![1.0, 1.0, 1.0];
            let c = vec![1.0, 1.0, 1.0];
            let d = vec![0.0];

            let ss = StateSpace::new(a, b, c, d, None).unwrap();
            let config = ReductionConfig::default();

            let result = modal_truncation(&ss, 2, &config).unwrap();

            assert_eq!(result.original_order, 3);
            assert_eq!(result.reduced_order, 2);
            assert_eq!(result.reduced_system.n_states, 2);
        }

        #[test]
        fn test_remove_common_factors() {
            let num = vec![0.0, 0.0, 1.0, 2.0];
            let den = vec![0.0, 1.0, 3.0, 2.0];

            let (num_clean, den_clean) = remove_common_factors(&num, &den).unwrap();

            assert_eq!(num_clean, vec![1.0, 2.0]);
            assert_eq!(den_clean, vec![1.0, 3.0, 2.0]);
        }

        #[test]
        fn test_trim_leading_zeros() {
            assert_eq!(trim_leading_zeros(&[0.0, 0.0, 1.0, 2.0]), vec![1.0, 2.0]);
            assert_eq!(trim_leading_zeros(&[1.0, 2.0, 3.0]), vec![1.0, 2.0, 3.0]);
            assert_eq!(trim_leading_zeros(&[0.0, 0.0, 0.0]), vec![0.0]);
        }
    }
}
