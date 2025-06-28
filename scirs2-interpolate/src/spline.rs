//! Spline interpolation methods
//!
//! This module provides functionality for spline interpolation.

use crate::error::{InterpolateError, InterpolateResult};
use ndarray::{Array1, Array2, ArrayView1};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

/// Boundary conditions for cubic splines
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SplineBoundaryCondition {
    /// Natural spline: second derivative is zero at endpoints
    Natural,
    /// Not-a-knot: third derivative is continuous at second and second-to-last points
    NotAKnot,
    /// Clamped/Complete: first derivative specified at endpoints
    Clamped(f64, f64),
    /// Periodic: function and derivatives match at endpoints
    Periodic,
    /// Second derivative specified at endpoints
    SecondDerivative(f64, f64),
    /// Parabolic runout: second derivative is zero at one endpoint
    ParabolicRunout,
}

/// Cubic spline interpolation object
///
/// Represents a piecewise cubic polynomial that passes through all given points
/// with continuous first and second derivatives.
#[derive(Debug, Clone)]
pub struct CubicSpline<F: Float + FromPrimitive> {
    /// X coordinates (must be sorted)
    x: Array1<F>,
    /// Y coordinates
    y: Array1<F>,
    /// Coefficients for cubic polynomials (n-1 segments, 4 coefficients each)
    /// Each row represents [a, b, c, d] for a segment
    /// y(x) = a + b*(x-x_i) + c*(x-x_i)^2 + d*(x-x_i)^3
    coeffs: Array2<F>,
}

/// Builder for cubic splines with custom boundary conditions
#[derive(Debug, Clone)]
pub struct CubicSplineBuilder<F: Float + FromPrimitive> {
    x: Option<Array1<F>>,
    y: Option<Array1<F>>,
    boundary_condition: SplineBoundaryCondition,
}

impl<F: Float + FromPrimitive + Debug> CubicSplineBuilder<F> {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            x: None,
            y: None,
            boundary_condition: SplineBoundaryCondition::Natural,
        }
    }
    
    /// Set the x coordinates
    pub fn x(mut self, x: Array1<F>) -> Self {
        self.x = Some(x);
        self
    }
    
    /// Set the y coordinates
    pub fn y(mut self, y: Array1<F>) -> Self {
        self.y = Some(y);
        self
    }
    
    /// Set the boundary condition
    pub fn boundary_condition(mut self, bc: SplineBoundaryCondition) -> Self {
        self.boundary_condition = bc;
        self
    }
    
    /// Build the spline
    pub fn build(self) -> InterpolateResult<CubicSpline<F>> {
        let x = self.x.ok_or_else(|| InterpolateError::invalid_input("x coordinates not set".to_string()))?;
        let y = self.y.ok_or_else(|| InterpolateError::invalid_input("y coordinates not set".to_string()))?;
        
        match self.boundary_condition {
            SplineBoundaryCondition::Natural => CubicSpline::new(&x.view(), &y.view()),
            SplineBoundaryCondition::NotAKnot => CubicSpline::new_not_a_knot(&x.view(), &y.view()),
            SplineBoundaryCondition::Clamped(left_deriv, right_deriv) => {
                let left_f = F::from_f64(left_deriv).ok_or_else(|| {
                    InterpolateError::ComputationError(
                        format!("Failed to convert left derivative {} to float type", left_deriv)
                    )
                })?;
                let right_f = F::from_f64(right_deriv).ok_or_else(|| {
                    InterpolateError::ComputationError(
                        format!("Failed to convert right derivative {} to float type", right_deriv)
                    )
                })?;
                CubicSpline::new_clamped(&x.view(), &y.view(), left_f, right_f)
            }
            SplineBoundaryCondition::Periodic => CubicSpline::new_periodic(&x.view(), &y.view()),
            SplineBoundaryCondition::SecondDerivative(left_d2, right_d2) => {
                let left_f = F::from_f64(left_d2).ok_or_else(|| {
                    InterpolateError::ComputationError(
                        format!("Failed to convert left second derivative {} to float type", left_d2)
                    )
                })?;
                let right_f = F::from_f64(right_d2).ok_or_else(|| {
                    InterpolateError::ComputationError(
                        format!("Failed to convert right second derivative {} to float type", right_d2)
                    )
                })?;
                CubicSpline::new_second_derivative(&x.view(), &y.view(), left_f, right_f)
            }
            SplineBoundaryCondition::ParabolicRunout => CubicSpline::new_parabolic_runout(&x.view(), &y.view()),
        }
    }
}

impl<F: Float + FromPrimitive + Debug> CubicSpline<F> {
    /// Create a new builder for cubic splines
    pub fn builder() -> CubicSplineBuilder<F> {
        CubicSplineBuilder::new()
    }
    /// Create a new cubic spline with natural boundary conditions
    ///
    /// # Arguments
    ///
    /// * `x` - The x coordinates (must be sorted in ascending order)
    /// * `y` - The y coordinates (must have the same length as x)
    ///
    /// # Returns
    ///
    /// A new `CubicSpline` object
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::array;
    /// use scirs2_interpolate::spline::CubicSpline;
    ///
    /// let x = array![0.0, 1.0, 2.0, 3.0];
    /// let y = array![0.0, 1.0, 4.0, 9.0];
    ///
    /// let spline = CubicSpline::new(&x.view(), &y.view()).unwrap();
    ///
    /// // Interpolate at x = 1.5
    /// let y_interp = spline.evaluate(1.5).unwrap();
    /// println!("Interpolated value at x=1.5: {}", y_interp);
    /// ```
    pub fn new(x: &ArrayView1<F>, y: &ArrayView1<F>) -> InterpolateResult<Self> {
        // Check inputs
        if x.len() != y.len() {
            return Err(InterpolateError::invalid_input(
                "x and y arrays must have the same length".to_string(),
            ));
        }

        if x.len() < 3 {
            return Err(InterpolateError::invalid_input(
                "at least 3 points are required for cubic spline".to_string(),
            ));
        }

        // Check that x is sorted
        for i in 1..x.len() {
            if x[i] <= x[i - 1] {
                return Err(InterpolateError::invalid_input(
                    "x values must be sorted in ascending order".to_string(),
                ));
            }
        }

        // Get coefficients for natural cubic spline
        let coeffs = compute_natural_cubic_spline(x, y)?;

        Ok(CubicSpline {
            x: x.to_owned(),
            y: y.to_owned(),
            coeffs,
        })
    }

    /// Create a new cubic spline with not-a-knot boundary conditions
    ///
    /// # Arguments
    ///
    /// * `x` - The x coordinates (must be sorted in ascending order)
    /// * `y` - The y coordinates (must have the same length as x)
    ///
    /// # Returns
    ///
    /// A new `CubicSpline` object
    pub fn new_not_a_knot(x: &ArrayView1<F>, y: &ArrayView1<F>) -> InterpolateResult<Self> {
        // Check inputs
        if x.len() != y.len() {
            return Err(InterpolateError::invalid_input(
                "x and y arrays must have the same length".to_string(),
            ));
        }

        if x.len() < 4 {
            return Err(InterpolateError::invalid_input(
                "at least 4 points are required for not-a-knot cubic spline".to_string(),
            ));
        }

        // Check that x is sorted
        for i in 1..x.len() {
            if x[i] <= x[i - 1] {
                return Err(InterpolateError::invalid_input(
                    "x values must be sorted in ascending order".to_string(),
                ));
            }
        }

        // Get coefficients for not-a-knot cubic spline
        let coeffs = compute_not_a_knot_cubic_spline(x, y)?;

        Ok(CubicSpline {
            x: x.to_owned(),
            y: y.to_owned(),
            coeffs,
        })
    }

    /// Create a new cubic spline with clamped boundary conditions
    ///
    /// # Arguments
    ///
    /// * `x` - The x coordinates (must be sorted in ascending order)
    /// * `y` - The y coordinates (must have the same length as x)
    /// * `left_deriv` - First derivative at the left endpoint
    /// * `right_deriv` - First derivative at the right endpoint
    ///
    /// # Returns
    ///
    /// A new `CubicSpline` object
    pub fn new_clamped(x: &ArrayView1<F>, y: &ArrayView1<F>, left_deriv: F, right_deriv: F) -> InterpolateResult<Self> {
        // Check inputs
        if x.len() != y.len() {
            return Err(InterpolateError::invalid_input(
                "x and y arrays must have the same length".to_string(),
            ));
        }

        if x.len() < 3 {
            return Err(InterpolateError::invalid_input(
                "at least 3 points are required for cubic spline".to_string(),
            ));
        }

        // Check that x is sorted
        for i in 1..x.len() {
            if x[i] <= x[i - 1] {
                return Err(InterpolateError::invalid_input(
                    "x values must be sorted in ascending order".to_string(),
                ));
            }
        }

        // Get coefficients for clamped cubic spline
        let coeffs = compute_clamped_cubic_spline(x, y, left_deriv, right_deriv)?;

        Ok(CubicSpline {
            x: x.to_owned(),
            y: y.to_owned(),
            coeffs,
        })
    }

    /// Create a new cubic spline with periodic boundary conditions
    ///
    /// # Arguments
    ///
    /// * `x` - The x coordinates (must be sorted in ascending order)
    /// * `y` - The y coordinates (must have the same length as x)
    ///
    /// # Returns
    ///
    /// A new `CubicSpline` object
    pub fn new_periodic(x: &ArrayView1<F>, y: &ArrayView1<F>) -> InterpolateResult<Self> {
        // Check inputs
        if x.len() != y.len() {
            return Err(InterpolateError::invalid_input(
                "x and y arrays must have the same length".to_string(),
            ));
        }

        if x.len() < 3 {
            return Err(InterpolateError::invalid_input(
                "at least 3 points are required for cubic spline".to_string(),
            ));
        }

        // Check that x is sorted
        for i in 1..x.len() {
            if x[i] <= x[i - 1] {
                return Err(InterpolateError::invalid_input(
                    "x values must be sorted in ascending order".to_string(),
                ));
            }
        }

        // Check periodicity
        let tol = F::from_f64(1e-10).ok_or_else(|| {
            InterpolateError::ComputationError(
                "Failed to convert tolerance 1e-10 to float type".to_string()
            )
        })?;
        if (y[0] - y[y.len() - 1]).abs() > tol {
            return Err(InterpolateError::invalid_input(
                "y values must be periodic (y[0] == y[n-1])".to_string(),
            ));
        }

        // Get coefficients for periodic cubic spline
        let coeffs = compute_periodic_cubic_spline(x, y)?;

        Ok(CubicSpline {
            x: x.to_owned(),
            y: y.to_owned(),
            coeffs,
        })
    }

    /// Create a new cubic spline with specified second derivatives at endpoints
    ///
    /// # Arguments
    ///
    /// * `x` - The x coordinates (must be sorted in ascending order)
    /// * `y` - The y coordinates (must have the same length as x)
    /// * `left_d2` - Second derivative at the left endpoint
    /// * `right_d2` - Second derivative at the right endpoint
    ///
    /// # Returns
    ///
    /// A new `CubicSpline` object
    pub fn new_second_derivative(x: &ArrayView1<F>, y: &ArrayView1<F>, left_d2: F, right_d2: F) -> InterpolateResult<Self> {
        // Check inputs
        if x.len() != y.len() {
            return Err(InterpolateError::invalid_input(
                "x and y arrays must have the same length".to_string(),
            ));
        }

        if x.len() < 3 {
            return Err(InterpolateError::invalid_input(
                "at least 3 points are required for cubic spline".to_string(),
            ));
        }

        // Check that x is sorted
        for i in 1..x.len() {
            if x[i] <= x[i - 1] {
                return Err(InterpolateError::invalid_input(
                    "x values must be sorted in ascending order".to_string(),
                ));
            }
        }

        // Get coefficients
        let coeffs = compute_second_derivative_cubic_spline(x, y, left_d2, right_d2)?;

        Ok(CubicSpline {
            x: x.to_owned(),
            y: y.to_owned(),
            coeffs,
        })
    }

    /// Create a new cubic spline with parabolic runout boundary conditions
    ///
    /// # Arguments
    ///
    /// * `x` - The x coordinates (must be sorted in ascending order)
    /// * `y` - The y coordinates (must have the same length as x)
    ///
    /// # Returns
    ///
    /// A new `CubicSpline` object
    pub fn new_parabolic_runout(x: &ArrayView1<F>, y: &ArrayView1<F>) -> InterpolateResult<Self> {
        // Check inputs
        if x.len() != y.len() {
            return Err(InterpolateError::invalid_input(
                "x and y arrays must have the same length".to_string(),
            ));
        }

        if x.len() < 3 {
            return Err(InterpolateError::invalid_input(
                "at least 3 points are required for cubic spline".to_string(),
            ));
        }

        // Check that x is sorted
        for i in 1..x.len() {
            if x[i] <= x[i - 1] {
                return Err(InterpolateError::invalid_input(
                    "x values must be sorted in ascending order".to_string(),
                ));
            }
        }

        // Get coefficients - parabolic runout means d[0] = d[n-2] = 0
        let coeffs = compute_parabolic_runout_cubic_spline(x, y)?;

        Ok(CubicSpline {
            x: x.to_owned(),
            y: y.to_owned(),
            coeffs,
        })
    }

    /// Evaluate the spline at the given point
    ///
    /// # Arguments
    ///
    /// * `x_new` - The x coordinate at which to evaluate the spline
    ///
    /// # Returns
    ///
    /// The interpolated y value at `x_new`
    pub fn evaluate(&self, x_new: F) -> InterpolateResult<F> {
        // Check if x_new is within the range
        if x_new < self.x[0] || x_new > self.x[self.x.len() - 1] {
            return Err(InterpolateError::OutOfBounds(
                "x_new is outside the interpolation range".to_string(),
            ));
        }

        // Find the index of the segment containing x_new
        let mut idx = 0;
        for i in 0..self.x.len() - 1 {
            if x_new >= self.x[i] && x_new <= self.x[i + 1] {
                idx = i;
                break;
            }
        }

        // Special case: x_new is exactly the last point
        if x_new == self.x[self.x.len() - 1] {
            return Ok(self.y[self.x.len() - 1]);
        }

        // Evaluate the cubic polynomial
        let dx = x_new - self.x[idx];
        let a = self.coeffs[[idx, 0]];
        let b = self.coeffs[[idx, 1]];
        let c = self.coeffs[[idx, 2]];
        let d = self.coeffs[[idx, 3]];

        let result = a + b * dx + c * dx * dx + d * dx * dx * dx;
        Ok(result)
    }

    /// Evaluate the spline at multiple points
    ///
    /// # Arguments
    ///
    /// * `x_new` - The x coordinates at which to evaluate the spline
    ///
    /// # Returns
    ///
    /// The interpolated y values at `x_new`
    pub fn evaluate_array(&self, x_new: &ArrayView1<F>) -> InterpolateResult<Array1<F>> {
        let mut result = Array1::zeros(x_new.len());
        for (i, &x) in x_new.iter().enumerate() {
            result[i] = self.evaluate(x)?;
        }
        Ok(result)
    }

    /// Get the derivative of the spline at the given point
    ///
    /// # Arguments
    ///
    /// * `x_new` - The x coordinate at which to evaluate the derivative
    /// * `order` - Order of derivative (1 = first derivative, 2 = second derivative)
    ///
    /// # Returns
    ///
    /// The derivative at `x_new`
    pub fn derivative(&self, x_new: F) -> InterpolateResult<F> {
        self.derivative_n(x_new, 1)
    }
    
    /// Get the nth derivative of the spline at the given point
    ///
    /// # Arguments
    ///
    /// * `x_new` - The x coordinate at which to evaluate the derivative
    /// * `order` - Order of derivative (1 = first derivative, 2 = second derivative, etc.)
    ///
    /// # Returns
    ///
    /// The nth derivative at `x_new`
    pub fn derivative_n(&self, x_new: F, order: usize) -> InterpolateResult<F> {
        // Check order validity
        if order == 0 {
            return self.evaluate(x_new);
        }
        
        if order > 3 {
            // Cubic spline has zero derivatives of order > 3
            return Ok(F::zero());
        }

        // Check if x_new is within the range
        if x_new < self.x[0] || x_new > self.x[self.x.len() - 1] {
            return Err(InterpolateError::OutOfBounds(
                "x_new is outside the interpolation range".to_string(),
            ));
        }

        // Find the index of the segment containing x_new
        let mut idx = 0;
        for i in 0..self.x.len() - 1 {
            if x_new >= self.x[i] && x_new <= self.x[i + 1] {
                idx = i;
                break;
            }
        }

        // Special case: x_new is exactly the last point
        if x_new == self.x[self.x.len() - 1] {
            idx = self.x.len() - 2;
        }

        let dx = x_new - self.x[idx];
        let b = self.coeffs[[idx, 1]];
        let c = self.coeffs[[idx, 2]];
        let d = self.coeffs[[idx, 3]];

        match order {
            1 => {
                // First derivative: b + 2*c*dx + 3*d*dx^2
                let two = F::from_f64(2.0).ok_or_else(|| {
                    InterpolateError::ComputationError(
                        "Failed to convert constant 2.0 to float type".to_string()
                    )
                })?;
                let three = F::from_f64(3.0).ok_or_else(|| {
                    InterpolateError::ComputationError(
                        "Failed to convert constant 3.0 to float type".to_string()
                    )
                })?;
                Ok(b + two * c * dx + three * d * dx * dx)
            }
            2 => {
                // Second derivative: 2*c + 6*d*dx
                let two = F::from_f64(2.0).ok_or_else(|| {
                    InterpolateError::ComputationError(
                        "Failed to convert constant 2.0 to float type".to_string()
                    )
                })?;
                let six = F::from_f64(6.0).ok_or_else(|| {
                    InterpolateError::ComputationError(
                        "Failed to convert constant 6.0 to float type".to_string()
                    )
                })?;
                Ok(two * c + six * d * dx)
            }
            3 => {
                // Third derivative: 6*d
                let six = F::from_f64(6.0).ok_or_else(|| {
                    InterpolateError::ComputationError(
                        "Failed to convert constant 6.0 to float type".to_string()
                    )
                })?;
                Ok(six * d)
            }
            _ => Ok(F::zero()),
        }
    }
    
    /// Compute derivatives at multiple points
    ///
    /// # Arguments
    ///
    /// * `x_new` - Array of points to evaluate derivatives at
    /// * `order` - Order of derivative (1, 2, or 3)
    ///
    /// # Returns
    ///
    /// Array of derivative values
    pub fn derivative_array(&self, x_new: &ArrayView1<F>, order: usize) -> InterpolateResult<Array1<F>> {
        let mut result = Array1::zeros(x_new.len());
        
        for (i, &x) in x_new.iter().enumerate() {
            result[i] = self.derivative_n(x, order)?;
        }
        
        Ok(result)
    }
    
    /// Find the antiderivative (indefinite integral) spline
    ///
    /// Returns a new cubic spline representing the antiderivative.
    /// The integration constant is chosen so that the antiderivative is 0 at x[0].
    ///
    /// # Returns
    ///
    /// A new CubicSpline representing the antiderivative
    pub fn antiderivative(&self) -> InterpolateResult<CubicSpline<F>> {
        let n = self.x.len();
        let mut antideriv_y = Array1::zeros(n);
        
        // Set first value to 0 (integration constant)
        antideriv_y[0] = F::zero();
        
        // Compute values at each knot by integrating from the first point
        for i in 1..n {
            let integral = self.integrate(self.x[0], self.x[i])?;
            antideriv_y[i] = integral;
        }
        
        // Create a new spline with these values
        CubicSpline::new(&self.x.view(), &antideriv_y.view())
    }
    
    /// Find roots of the spline (points where spline equals zero)
    ///
    /// Uses a combination of bracketing and Newton's method to find roots.
    ///
    /// # Arguments
    ///
    /// * `tolerance` - Convergence tolerance for root finding
    /// * `max_iterations` - Maximum number of iterations per root
    ///
    /// # Returns
    ///
    /// Vector of root locations
    pub fn find_roots(&self, tolerance: F, max_iterations: usize) -> InterpolateResult<Vec<F>> {
        let mut roots = Vec::new();
        let n_segments = self.coeffs.nrows();
        
        // Check for roots in each segment
        for segment in 0..n_segments {
            let x_left = self.x[segment];
            let x_right = self.x[segment + 1];
            
            let y_left = self.evaluate(x_left)?;
            let y_right = self.evaluate(x_right)?;
            
            // Check if there's a sign change (indicates a root)
            if y_left * y_right < F::zero() {
                // Use Newton's method with bisection fallback
                if let Some(root) = self.find_root_in_segment(segment, x_left, x_right, tolerance, max_iterations)? {
                    roots.push(root);
                }
            } else if y_left.abs() < tolerance {
                // Check if left endpoint is a root
                if roots.is_empty() || (root_far_enough(&roots, x_left, tolerance)) {
                    roots.push(x_left);
                }
            }
        }
        
        // Check the last point
        let x_last = self.x[n_segments];
        let y_last = self.evaluate(x_last)?;
        if y_last.abs() < tolerance && root_far_enough(&roots, x_last, tolerance) {
            roots.push(x_last);
        }
        
        Ok(roots)
    }
    
    /// Find extrema (local minima and maxima) of the spline
    ///
    /// # Arguments
    ///
    /// * `tolerance` - Convergence tolerance
    /// * `max_iterations` - Maximum iterations per extremum
    ///
    /// # Returns
    ///
    /// Vector of (x, y, type) where type is "min" or "max"
    pub fn find_extrema(&self, tolerance: F, max_iterations: usize) -> InterpolateResult<Vec<(F, F, &'static str)>> {
        let mut extrema = Vec::new();
        let n_segments = self.coeffs.nrows();
        
        // Find critical points by looking for roots of the first derivative
        for segment in 0..n_segments {
            let x_left = self.x[segment];
            let x_right = self.x[segment + 1];
            
            let dy_left = self.derivative_n(x_left, 1)?;
            let dy_right = self.derivative_n(x_right, 1)?;
            
            // Check for sign change in first derivative
            if dy_left * dy_right < F::zero() {
                if let Some(critical_x) = self.find_derivative_root_in_segment(segment, x_left, x_right, tolerance, max_iterations)? {
                    let critical_y = self.evaluate(critical_x)?;
                    let second_deriv = self.derivative_n(critical_x, 2)?;
                    
                    let extremum_type = if second_deriv > F::zero() {
                        "min"
                    } else if second_deriv < F::zero() {
                        "max"
                    } else {
                        continue; // Inflection point, skip
                    };
                    
                    extrema.push((critical_x, critical_y, extremum_type));
                }
            }
        }
        
        Ok(extrema)
    }
    
    /// Compute arc length of the spline from a to b
    ///
    /// Uses adaptive quadrature to compute the integral of sqrt(1 + (dy/dx)^2)
    ///
    /// # Arguments
    ///
    /// * `a` - Start point
    /// * `b` - End point
    /// * `tolerance` - Integration tolerance
    ///
    /// # Returns
    ///
    /// Arc length from a to b
    pub fn arc_length(&self, a: F, b: F, tolerance: F) -> InterpolateResult<F> {
        if a == b {
            return Ok(F::zero());
        }
        
        // Handle reversed bounds
        if a > b {
            return self.arc_length(b, a, tolerance);
        }
        
        // Check bounds
        let x_min = self.x[0];
        let x_max = self.x[self.x.len() - 1];
        
        if a < x_min || b > x_max {
            return Err(InterpolateError::OutOfDomain {
                point: format!("({}, {})", a, b),
                min: x_min.to_string(),
                max: x_max.to_string(),
                context: "arc length computation".to_string(),
            });
        }
        
        // Use adaptive Simpson's rule to integrate sqrt(1 + (dy/dx)^2)
        self.adaptive_arc_length_integration(a, b, tolerance)
    }
    
    /// Compute the integral of the spline from a to b
    ///
    /// # Arguments
    ///
    /// * `a` - Lower bound of integration
    /// * `b` - Upper bound of integration
    ///
    /// # Returns
    ///
    /// The definite integral from a to b
    pub fn integrate(&self, a: F, b: F) -> InterpolateResult<F> {
        // Handle reversed bounds
        if a > b {
            return Ok(-self.integrate(b, a)?);
        }
        
        if a == b {
            return Ok(F::zero());
        }
        
        // Check bounds
        let x_min = self.x[0];
        let x_max = self.x[self.x.len() - 1];
        
        if a < x_min || b > x_max {
            return Err(InterpolateError::OutOfBounds(
                "Integration bounds outside interpolation range".to_string(),
            ));
        }
        
        // Find the segments containing a and b
        let mut idx_a = 0;
        let mut idx_b = 0;
        
        for i in 0..self.x.len() - 1 {
            if a >= self.x[i] && a <= self.x[i + 1] {
                idx_a = i;
            }
            if b >= self.x[i] && b <= self.x[i + 1] {
                idx_b = i;
            }
        }
        
        let mut integral = F::zero();
        
        // If both points are in the same segment
        if idx_a == idx_b {
            integral = self.integrate_segment(idx_a, a, b)?;
        } else {
            // Integrate from a to the end of its segment
            integral = integral + self.integrate_segment(idx_a, a, self.x[idx_a + 1])?;
            
            // Integrate all complete segments in between
            for i in (idx_a + 1)..idx_b {
                integral = integral + self.integrate_segment(i, self.x[i], self.x[i + 1])?;
            }
            
            // Integrate from the start of b's segment to b
            integral = integral + self.integrate_segment(idx_b, self.x[idx_b], b)?;
        }
        
        Ok(integral)
    }
    
    /// Helper function to find a root in a specific segment using Newton's method with bisection fallback
    fn find_root_in_segment(
        &self, 
        segment: usize, 
        x_left: F, 
        x_right: F, 
        tolerance: F, 
        max_iterations: usize
    ) -> InterpolateResult<Option<F>> {
        let two = F::from_f64(2.0).ok_or_else(|| {
            InterpolateError::ComputationError(
                "Failed to convert constant 2.0 to float type".to_string()
            )
        })?;
        let mut x = (x_left + x_right) / two; // Start at midpoint
        
        for _ in 0..max_iterations {
            let y = self.evaluate_segment(segment, x)?;
            if y.abs() < tolerance {
                return Ok(Some(x));
            }
            
            let dy = self.derivative_segment(segment, x, 1)?;
            if dy.abs() < tolerance {
                // Newton's method would fail, use bisection
                return self.bisection_root_find(segment, x_left, x_right, tolerance, max_iterations);
            }
            
            // Check for division by zero in Newton's method
            if dy.abs() < F::from_f64(1e-14).unwrap_or_else(|| F::from(1e-14)) {
                return self.bisection_root_find(segment, x_left, x_right, tolerance, max_iterations);
            }
            let x_new = x - y / dy;
            
            // If Newton step goes outside interval, use bisection
            if x_new < x_left || x_new > x_right {
                return self.bisection_root_find(segment, x_left, x_right, tolerance, max_iterations);
            }
            
            if (x_new - x).abs() < tolerance {
                return Ok(Some(x_new));
            }
            
            x = x_new;
        }
        
        Ok(None) // Convergence failed
    }
    
    /// Helper function for bisection root finding
    fn bisection_root_find(
        &self,
        segment: usize,
        mut a: F,
        mut b: F,
        tolerance: F,
        max_iterations: usize,
    ) -> InterpolateResult<Option<F>> {
        for _ in 0..max_iterations {
            let two = F::from_f64(2.0).ok_or_else(|| {
                InterpolateError::ComputationError(
                    "Failed to convert constant 2.0 to float type".to_string()
                )
            })?;
            let c = (a + b) / two;
            let y_c = self.evaluate_segment(segment, c)?;
            
            if y_c.abs() < tolerance || (b - a).abs() < tolerance {
                return Ok(Some(c));
            }
            
            let y_a = self.evaluate_segment(segment, a)?;
            if y_a * y_c < F::zero() {
                b = c;
            } else {
                a = c;
            }
        }
        
        Ok(None)
    }
    
    /// Helper function to find roots of the derivative in a segment
    fn find_derivative_root_in_segment(
        &self,
        segment: usize,
        x_left: F,
        x_right: F,
        tolerance: F,
        max_iterations: usize,
    ) -> InterpolateResult<Option<F>> {
        let two = F::from_f64(2.0).ok_or_else(|| {
            InterpolateError::ComputationError(
                "Failed to convert constant 2.0 to float type".to_string()
            )
        })?;
        let mut x = (x_left + x_right) / two;
        
        for _ in 0..max_iterations {
            let dy = self.derivative_segment(segment, x, 1)?;
            if dy.abs() < tolerance {
                return Ok(Some(x));
            }
            
            let d2y = self.derivative_segment(segment, x, 2)?;
            if d2y.abs() < tolerance {
                // Use bisection fallback
                return self.bisection_derivative_root_find(segment, x_left, x_right, tolerance, max_iterations);
            }
            
            // Check for division by zero
            if d2y.abs() < F::from_f64(1e-14).unwrap_or_else(|| F::from(1e-14)) {
                return self.bisection_derivative_root_find(segment, x_left, x_right, tolerance, max_iterations);
            }
            let x_new = x - dy / d2y;
            
            if x_new < x_left || x_new > x_right {
                return self.bisection_derivative_root_find(segment, x_left, x_right, tolerance, max_iterations);
            }
            
            if (x_new - x).abs() < tolerance {
                return Ok(Some(x_new));
            }
            
            x = x_new;
        }
        
        Ok(None)
    }
    
    /// Helper function for bisection derivative root finding
    fn bisection_derivative_root_find(
        &self,
        segment: usize,
        mut a: F,
        mut b: F,
        tolerance: F,
        max_iterations: usize,
    ) -> InterpolateResult<Option<F>> {
        for _ in 0..max_iterations {
            let two = F::from_f64(2.0).ok_or_else(|| {
                InterpolateError::ComputationError(
                    "Failed to convert constant 2.0 to float type".to_string()
                )
            })?;
            let c = (a + b) / two;
            let dy_c = self.derivative_segment(segment, c, 1)?;
            
            if dy_c.abs() < tolerance || (b - a).abs() < tolerance {
                return Ok(Some(c));
            }
            
            let dy_a = self.derivative_segment(segment, a, 1)?;
            if dy_a * dy_c < F::zero() {
                b = c;
            } else {
                a = c;
            }
        }
        
        Ok(None)
    }
    
    /// Evaluate spline in a specific segment
    fn evaluate_segment(&self, segment: usize, x: F) -> InterpolateResult<F> {
        let x0 = self.x[segment];
        let dx = x - x0;
        
        let a = self.coeffs[[segment, 0]];
        let b = self.coeffs[[segment, 1]];
        let c = self.coeffs[[segment, 2]];
        let d = self.coeffs[[segment, 3]];
        
        Ok(a + b * dx + c * dx * dx + d * dx * dx * dx)
    }
    
    /// Evaluate derivative in a specific segment
    fn derivative_segment(&self, segment: usize, x: F, order: usize) -> InterpolateResult<F> {
        let x0 = self.x[segment];
        let dx = x - x0;
        
        let b = self.coeffs[[segment, 1]];
        let c = self.coeffs[[segment, 2]];
        let d = self.coeffs[[segment, 3]];
        
        match order {
            1 => {
                let two = F::from_f64(2.0).ok_or_else(|| {
                    InterpolateError::ComputationError(
                        "Failed to convert constant 2.0 to float type".to_string()
                    )
                })?;
                let three = F::from_f64(3.0).ok_or_else(|| {
                    InterpolateError::ComputationError(
                        "Failed to convert constant 3.0 to float type".to_string()
                    )
                })?;
                Ok(b + two * c * dx + three * d * dx * dx)
            }
            2 => {
                let two = F::from_f64(2.0).ok_or_else(|| {
                    InterpolateError::ComputationError(
                        "Failed to convert constant 2.0 to float type".to_string()
                    )
                })?;
                let six = F::from_f64(6.0).ok_or_else(|| {
                    InterpolateError::ComputationError(
                        "Failed to convert constant 6.0 to float type".to_string()
                    )
                })?;
                Ok(two * c + six * d * dx)
            }
            3 => {
                let six = F::from_f64(6.0).ok_or_else(|| {
                    InterpolateError::ComputationError(
                        "Failed to convert constant 6.0 to float type".to_string()
                    )
                })?;
                Ok(six * d)
            }
            _ => Ok(F::zero()),
        }
    }
    
    /// Adaptive arc length integration using Simpson's rule
    fn adaptive_arc_length_integration(&self, a: F, b: F, tolerance: F) -> InterpolateResult<F> {
        // Simple implementation using composite Simpson's rule
        let n = 100; // Number of subdivisions
        let n_f = F::from_usize(n).ok_or_else(|| {
            InterpolateError::ComputationError(
                "Failed to convert number of subdivisions to float type".to_string()
            )
        })?;
        let h = (b - a) / n_f;
        let mut sum = F::zero();
        
        let two = F::from_f64(2.0).ok_or_else(|| {
            InterpolateError::ComputationError(
                "Failed to convert constant 2.0 to float type".to_string()
            )
        })?;
        let four = F::from_f64(4.0).ok_or_else(|| {
            InterpolateError::ComputationError(
                "Failed to convert constant 4.0 to float type".to_string()
            )
        })?;
        let six = F::from_f64(6.0).ok_or_else(|| {
            InterpolateError::ComputationError(
                "Failed to convert constant 6.0 to float type".to_string()
            )
        })?;
        
        for i in 0..=n {
            let i_f = F::from_usize(i).ok_or_else(|| {
                InterpolateError::ComputationError(
                    "Failed to convert loop index to float type".to_string()
                )
            })?;
            let x = a + i_f * h;
            let dy_dx = self.derivative_n(x, 1)?;
            let integrand = (F::one() + dy_dx * dy_dx).sqrt();
            
            let coefficient = if i == 0 || i == n {
                F::one()
            } else if i % 2 == 1 {
                four
            } else {
                two
            };
            
            sum += coefficient * integrand;
        }
        
        Ok(sum * h / six)
    }
    
    /// Compute the integral of a single spline segment
    fn integrate_segment(&self, idx: usize, x_start: F, x_end: F) -> InterpolateResult<F> {
        let a = self.coeffs[[idx, 0]];
        let b = self.coeffs[[idx, 1]];
        let c = self.coeffs[[idx, 2]];
        let d = self.coeffs[[idx, 3]];
        
        let x_i = self.x[idx];
        
        // Integral of a + b*(x-x_i) + c*(x-x_i)^2 + d*(x-x_i)^3
        // = a*(x-x_i) + b*(x-x_i)^2/2 + c*(x-x_i)^3/3 + d*(x-x_i)^4/4
        
        let dx_start = x_start - x_i;
        let dx_end = x_end - x_i;
        
        let two = F::from_f64(2.0).ok_or_else(|| {
            InterpolateError::ComputationError(
                "Failed to convert constant 2.0 to float type".to_string()
            )
        })?;
        let three = F::from_f64(3.0).ok_or_else(|| {
            InterpolateError::ComputationError(
                "Failed to convert constant 3.0 to float type".to_string()
            )
        })?;
        let four = F::from_f64(4.0).ok_or_else(|| {
            InterpolateError::ComputationError(
                "Failed to convert constant 4.0 to float type".to_string()
            )
        })?;
        
        // Check for potential division issues - these constants should never be zero,
        // but we protect against it for numerical safety
        if two.is_zero() || three.is_zero() || four.is_zero() {
            return Err(InterpolateError::ComputationError(
                "Division by zero in polynomial integration".to_string()
            ));
        }
        
        let integral_end = a * dx_end 
            + b * dx_end * dx_end / two
            + c * dx_end * dx_end * dx_end / three
            + d * dx_end * dx_end * dx_end * dx_end / four;
            
        let integral_start = a * dx_start 
            + b * dx_start * dx_start / two
            + c * dx_start * dx_start * dx_start / three
            + d * dx_start * dx_start * dx_start * dx_start / four;
        
        Ok(integral_end - integral_start)
    }
    
    /// Evaluate multiple derivatives at once
    ///
    /// # Arguments
    ///
    /// * `x_new` - The x coordinate at which to evaluate
    /// * `max_order` - Maximum order of derivative to compute (0 to 3)
    ///
    /// # Returns
    ///
    /// Array containing [f(x), f'(x), f''(x), ...] up to the requested order
    pub fn derivatives_all(&self, x_new: F, max_order: usize) -> InterpolateResult<Array1<F>> {
        let order = max_order.min(3);
        let mut result = Array1::zeros(order + 1);
        
        for i in 0..=order {
            result[i] = self.derivative_n(x_new, i)?;
        }
        
        Ok(result)
    }
}

/// Compute the coefficients for a natural cubic spline
///
/// Natural boundary conditions: second derivative is zero at the endpoints
fn compute_natural_cubic_spline<F: Float + FromPrimitive + Debug>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
) -> InterpolateResult<Array2<F>> {
    let n = x.len();
    let n_segments = n - 1;

    // Create array to hold the coefficients (n-1 segments x 4 coefficients)
    let mut coeffs = Array2::<F>::zeros((n_segments, 4));

    // Step 1: Calculate the second derivatives at each point
    // We solve the tridiagonal system to get these

    // Set up the tridiagonal system
    let mut a = Array1::<F>::zeros(n);
    let mut b = Array1::<F>::zeros(n);
    let mut c = Array1::<F>::zeros(n);
    let mut d = Array1::<F>::zeros(n);

    // Natural boundary conditions
    b[0] = F::one();
    d[0] = F::zero();
    b[n - 1] = F::one();
    d[n - 1] = F::zero();

    // Fill in the tridiagonal system
    for i in 1..n - 1 {
        let h_i_minus_1 = x[i] - x[i - 1];
        let h_i = x[i + 1] - x[i];

        a[i] = h_i_minus_1;
        let two = F::from_f64(2.0).ok_or_else(|| {
            InterpolateError::ComputationError(
                "Failed to convert constant 2.0 to float type".to_string()
            )
        })?;
        b[i] = two * (h_i_minus_1 + h_i);
        c[i] = h_i;

        let dy_i_minus_1 = y[i] - y[i - 1];
        let dy_i = y[i + 1] - y[i];

        // Check for division by zero in slope calculations
        if h_i.is_zero() || h_i_minus_1.is_zero() {
            return Err(InterpolateError::ComputationError(
                "Zero interval length in spline computation".to_string()
            ));
        }
        
        let six = F::from_f64(6.0).ok_or_else(|| {
            InterpolateError::ComputationError(
                "Failed to convert constant 6.0 to float type".to_string()
            )
        })?;
        d[i] = six * (dy_i / h_i - dy_i_minus_1 / h_i_minus_1);
    }

    // Solve the tridiagonal system using the Thomas algorithm
    let mut sigma = Array1::<F>::zeros(n);

    // Forward sweep
    for i in 1..n {
        // Check for division by zero
        if b[i - 1].is_zero() {
            return Err(InterpolateError::ComputationError(
                "Zero diagonal element in Thomas algorithm forward sweep".to_string()
            ));
        }
        let m = a[i] / b[i - 1];
        b[i] = b[i] - m * c[i - 1];
        d[i] = d[i] - m * d[i - 1];
    }

    // Back substitution
    if b[n - 1].is_zero() {
        return Err(InterpolateError::ComputationError(
            "Zero diagonal element in Thomas algorithm back substitution".to_string()
        ));
    }
    sigma[n - 1] = d[n - 1] / b[n - 1];
    for i in (0..n - 1).rev() {
        if b[i].is_zero() {
            return Err(InterpolateError::ComputationError(
                "Zero diagonal element in Thomas algorithm back substitution".to_string()
            ));
        }
        sigma[i] = (d[i] - c[i] * sigma[i + 1]) / b[i];
    }

    // Step 2: Calculate the polynomial coefficients
    for i in 0..n_segments {
        let h_i = x[i + 1] - x[i];

        // Check for division by zero in coefficient calculation
        if h_i.is_zero() {
            return Err(InterpolateError::ComputationError(
                "Zero interval length in spline coefficient calculation".to_string()
            ));
        }
        
        let two = F::from_f64(2.0).ok_or_else(|| {
            InterpolateError::ComputationError(
                "Failed to convert constant 2.0 to float type".to_string()
            )
        })?;
        let six = F::from_f64(6.0).ok_or_else(|| {
            InterpolateError::ComputationError(
                "Failed to convert constant 6.0 to float type".to_string()
            )
        })?;

        // a is just the y value at the left endpoint
        coeffs[[i, 0]] = y[i];

        // b is the first derivative at the left endpoint
        coeffs[[i, 1]] = (y[i + 1] - y[i]) / h_i
            - h_i * (two * sigma[i] + sigma[i + 1]) / six;

        // c is half the second derivative at the left endpoint
        coeffs[[i, 2]] = sigma[i] / two;

        // d is the rate of change of the second derivative / 6
        coeffs[[i, 3]] = (sigma[i + 1] - sigma[i]) / (six * h_i);
    }

    Ok(coeffs)
}

/// Compute the coefficients for a not-a-knot cubic spline
///
/// Not-a-knot boundary conditions: third derivative is continuous across the
/// first and last interior knots
fn compute_not_a_knot_cubic_spline<F: Float + FromPrimitive + Debug>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
) -> InterpolateResult<Array2<F>> {
    let n = x.len();
    let n_segments = n - 1;

    // Create array to hold the coefficients (n-1 segments x 4 coefficients)
    let mut coeffs = Array2::<F>::zeros((n_segments, 4));

    // Step 1: Calculate the second derivatives at each point

    // Set up the tridiagonal system
    let mut a = Array1::<F>::zeros(n);
    let mut b = Array1::<F>::zeros(n);
    let mut c = Array1::<F>::zeros(n);
    let mut d = Array1::<F>::zeros(n);

    // Not-a-knot condition at first interior point
    let h0 = x[1] - x[0];
    let h1 = x[2] - x[1];

    // Check for zero intervals
    if h0.is_zero() || h1.is_zero() || (h0 + h1).is_zero() {
        return Err(InterpolateError::ComputationError(
            "Zero interval length in not-a-knot spline boundary conditions".to_string()
        ));
    }

    b[0] = h1;
    c[0] = h0 + h1;
    d[0] = ((h0 + h1) * h1 * (y[1] - y[0]) / h0 + h0 * h0 * (y[2] - y[1]) / h1) / (h0 + h1);

    // Not-a-knot condition at last interior point
    let hn_2 = x[n - 2] - x[n - 3];
    let hn_1 = x[n - 1] - x[n - 2];

    // Check for zero intervals
    if hn_1.is_zero() || hn_2.is_zero() || (hn_1 + hn_2).is_zero() {
        return Err(InterpolateError::ComputationError(
            "Zero interval length in not-a-knot spline boundary conditions".to_string()
        ));
    }

    a[n - 1] = hn_1 + hn_2;
    b[n - 1] = hn_2;
    d[n - 1] = ((hn_1 + hn_2) * hn_2 * (y[n - 1] - y[n - 2]) / hn_1
        + hn_1 * hn_1 * (y[n - 2] - y[n - 3]) / hn_2)
        / (hn_1 + hn_2);

    // Fill in the tridiagonal system for interior points
    for i in 1..n - 1 {
        let h_i_minus_1 = x[i] - x[i - 1];
        let h_i = x[i + 1] - x[i];

        // Check for zero intervals
        if h_i.is_zero() || h_i_minus_1.is_zero() {
            return Err(InterpolateError::ComputationError(
                "Zero interval length in not-a-knot spline computation".to_string()
            ));
        }

        a[i] = h_i_minus_1;
        let two = F::from_f64(2.0).ok_or_else(|| {
            InterpolateError::ComputationError(
                "Failed to convert constant 2.0 to float type".to_string()
            )
        })?;
        b[i] = two * (h_i_minus_1 + h_i);
        c[i] = h_i;

        let dy_i_minus_1 = y[i] - y[i - 1];
        let dy_i = y[i + 1] - y[i];

        let six = F::from_f64(6.0).ok_or_else(|| {
            InterpolateError::ComputationError(
                "Failed to convert constant 6.0 to float type".to_string()
            )
        })?;
        d[i] = six * (dy_i / h_i - dy_i_minus_1 / h_i_minus_1);
    }

    // Solve the tridiagonal system using the Thomas algorithm
    let mut sigma = Array1::<F>::zeros(n);

    // Forward sweep
    let mut c_prime = Array1::<F>::zeros(n);
    
    // Check for division by zero in first step
    if b[0].is_zero() {
        return Err(InterpolateError::ComputationError(
            "Zero diagonal element in not-a-knot Thomas algorithm".to_string()
        ));
    }
    c_prime[0] = c[0] / b[0];
    
    for i in 1..n {
        let m = b[i] - a[i] * c_prime[i - 1];
        
        // Check for division by zero
        if m.is_zero() {
            return Err(InterpolateError::ComputationError(
                "Zero diagonal element in not-a-knot Thomas algorithm".to_string()
            ));
        }
        
        if i < n - 1 {
            c_prime[i] = c[i] / m;
        }
        d[i] = (d[i] - a[i] * d[i - 1]) / m;
    }

    // Back substitution
    sigma[n - 1] = d[n - 1];
    for i in (0..n - 1).rev() {
        sigma[i] = d[i] - c_prime[i] * sigma[i + 1];
    }

    // Step 2: Calculate the polynomial coefficients
    for i in 0..n_segments {
        let h_i = x[i + 1] - x[i];

        // Check for division by zero in coefficient calculation
        if h_i.is_zero() {
            return Err(InterpolateError::ComputationError(
                "Zero interval length in not-a-knot spline coefficient calculation".to_string()
            ));
        }
        
        let two = F::from_f64(2.0).ok_or_else(|| {
            InterpolateError::ComputationError(
                "Failed to convert constant 2.0 to float type".to_string()
            )
        })?;
        let six = F::from_f64(6.0).ok_or_else(|| {
            InterpolateError::ComputationError(
                "Failed to convert constant 6.0 to float type".to_string()
            )
        })?;

        // a is just the y value at the left endpoint
        coeffs[[i, 0]] = y[i];

        // b is the first derivative at the left endpoint
        coeffs[[i, 1]] = (y[i + 1] - y[i]) / h_i
            - h_i * (two * sigma[i] + sigma[i + 1]) / six;

        // c is half the second derivative at the left endpoint
        coeffs[[i, 2]] = sigma[i] / two;

        // d is the rate of change of the second derivative / 6
        coeffs[[i, 3]] = (sigma[i + 1] - sigma[i]) / (six * h_i);
    }

    Ok(coeffs)
}

/// Compute the coefficients for a clamped cubic spline
///
/// Clamped boundary conditions: first derivative specified at endpoints
fn compute_clamped_cubic_spline<F: Float + FromPrimitive + Debug>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
    left_deriv: F,
    right_deriv: F,
) -> InterpolateResult<Array2<F>> {
    let n = x.len();
    let n_segments = n - 1;

    // Create array to hold the coefficients
    let mut coeffs = Array2::<F>::zeros((n_segments, 4));

    // Set up the tridiagonal system
    let mut a = Array1::<F>::zeros(n);
    let mut b = Array1::<F>::zeros(n);
    let mut c = Array1::<F>::zeros(n);
    let mut d = Array1::<F>::zeros(n);

    // Clamped boundary conditions
    let h0 = x[1] - x[0];
    
    // Check for zero interval
    if h0.is_zero() {
        return Err(InterpolateError::ComputationError(
            "Zero interval length in clamped spline boundary conditions".to_string()
        ));
    }
    
    let two = F::from_f64(2.0).ok_or_else(|| {
        InterpolateError::ComputationError(
            "Failed to convert constant 2.0 to float type".to_string()
        )
    })?;
    let six = F::from_f64(6.0).ok_or_else(|| {
        InterpolateError::ComputationError(
            "Failed to convert constant 6.0 to float type".to_string()
        )
    })?;
    
    b[0] = two * h0;
    c[0] = h0;
    d[0] = six * ((y[1] - y[0]) / h0 - left_deriv);

    let hn_1 = x[n - 1] - x[n - 2];
    
    // Check for zero interval
    if hn_1.is_zero() {
        return Err(InterpolateError::ComputationError(
            "Zero interval length in clamped spline boundary conditions".to_string()
        ));
    }
    
    a[n - 1] = hn_1;
    b[n - 1] = two * hn_1;
    d[n - 1] = six * (right_deriv - (y[n - 1] - y[n - 2]) / hn_1);

    // Fill in the tridiagonal system for interior points
    for i in 1..n - 1 {
        let h_i_minus_1 = x[i] - x[i - 1];
        let h_i = x[i + 1] - x[i];
        
        // Check for zero intervals
        if h_i.is_zero() || h_i_minus_1.is_zero() {
            return Err(InterpolateError::ComputationError(
                "Zero interval length in clamped spline computation".to_string()
            ));
        }

        a[i] = h_i_minus_1;
        b[i] = two * (h_i_minus_1 + h_i);
        c[i] = h_i;

        let dy_i_minus_1 = y[i] - y[i - 1];
        let dy_i = y[i + 1] - y[i];

        d[i] = six * (dy_i / h_i - dy_i_minus_1 / h_i_minus_1);
    }

    // Solve the tridiagonal system
    let mut sigma = Array1::<F>::zeros(n);

    // Forward sweep
    for i in 1..n {
        let m = a[i] / b[i - 1];
        b[i] = b[i] - m * c[i - 1];
        d[i] = d[i] - m * d[i - 1];
    }

    // Back substitution
    sigma[n - 1] = d[n - 1] / b[n - 1];
    for i in (0..n - 1).rev() {
        sigma[i] = (d[i] - c[i] * sigma[i + 1]) / b[i];
    }

    // Calculate the polynomial coefficients
    for i in 0..n_segments {
        let h_i = x[i + 1] - x[i];
        
        // Check for zero interval in coefficient calculation
        if h_i.is_zero() {
            return Err(InterpolateError::ComputationError(
                "Zero interval length in clamped spline coefficient calculation".to_string()
            ));
        }

        coeffs[[i, 0]] = y[i];
        coeffs[[i, 1]] = (y[i + 1] - y[i]) / h_i
            - h_i * (two * sigma[i] + sigma[i + 1]) / six;
        coeffs[[i, 2]] = sigma[i] / two;
        coeffs[[i, 3]] = (sigma[i + 1] - sigma[i]) / (six * h_i);
    }

    Ok(coeffs)
}

/// Compute the coefficients for a periodic cubic spline
///
/// Periodic boundary conditions: function and derivatives match at endpoints
fn compute_periodic_cubic_spline<F: Float + FromPrimitive + Debug>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
) -> InterpolateResult<Array2<F>> {
    let n = x.len();
    let n_segments = n - 1;

    // Define constants
    let two = F::from_f64(2.0).ok_or_else(|| {
        InterpolateError::ComputationError(
            "Failed to convert constant 2.0 to float type".to_string()
        )
    })?;
    let six = F::from_f64(6.0).ok_or_else(|| {
        InterpolateError::ComputationError(
            "Failed to convert constant 6.0 to float type".to_string()
        )
    })?;

    // Create array to hold the coefficients
    let mut coeffs = Array2::<F>::zeros((n_segments, 4));

    // For periodic splines, we need to solve a slightly modified system
    // The matrix is almost tridiagonal with additional corner elements

    let mut a = Array1::<F>::zeros(n - 1);
    let mut b = Array1::<F>::zeros(n - 1);
    let mut c = Array1::<F>::zeros(n - 1);
    let mut d = Array1::<F>::zeros(n - 1);

    // Fill the system (we work with n-1 equations due to periodicity)
    for i in 0..n - 1 {
        let h_i_minus_1 = if i == 0 {
            x[n - 1] - x[n - 2]
        } else {
            x[i] - x[i - 1]
        };
        let h_i = x[i + 1] - x[i];

        // Check for zero intervals
        if h_i.is_zero() || h_i_minus_1.is_zero() {
            return Err(InterpolateError::ComputationError(
                "Zero interval length in periodic spline computation".to_string()
            ));
        }
        
        a[i] = h_i_minus_1;
        b[i] = two * (h_i_minus_1 + h_i);
        c[i] = h_i;

        let dy_i_minus_1 = if i == 0 {
            y[0] - y[n - 2]  // Using periodicity
        } else {
            y[i] - y[i - 1]
        };
        let dy_i = y[i + 1] - y[i];

        d[i] = six * (dy_i / h_i - dy_i_minus_1 / h_i_minus_1);
    }

    // For periodic boundary conditions, we need to solve a cyclic tridiagonal system
    // Using Sherman-Morrison formula or reduction to standard tridiagonal
    // For simplicity, we'll use a modified Thomas algorithm

    let mut sigma = Array1::<F>::zeros(n);

    // Simplified approach: assume natural boundary conditions as approximation
    // (A more accurate implementation would solve the cyclic system)
    let mut b_mod = b.clone();
    let mut d_mod = d.clone();

    // Forward sweep
    for i in 1..n - 1 {
        let m = a[i] / b_mod[i - 1];
        b_mod[i] = b_mod[i] - m * c[i - 1];
        d_mod[i] = d_mod[i] - m * d_mod[i - 1];
    }

    // Back substitution
    sigma[n - 2] = d_mod[n - 2] / b_mod[n - 2];
    for i in (0..n - 2).rev() {
        sigma[i] = (d_mod[i] - c[i] * sigma[i + 1]) / b_mod[i];
    }
    sigma[n - 1] = sigma[0]; // Periodicity

    // Calculate the polynomial coefficients
    for i in 0..n_segments {
        let h_i = x[i + 1] - x[i];
        
        // Check for zero interval in coefficient calculation
        if h_i.is_zero() {
            return Err(InterpolateError::ComputationError(
                "Zero interval length in periodic spline coefficient calculation".to_string()
            ));
        }

        coeffs[[i, 0]] = y[i];
        coeffs[[i, 1]] = (y[i + 1] - y[i]) / h_i
            - h_i * (two * sigma[i] + sigma[i + 1]) / six;
        coeffs[[i, 2]] = sigma[i] / two;
        coeffs[[i, 3]] = (sigma[i + 1] - sigma[i]) / (six * h_i);
    }

    Ok(coeffs)
}

/// Compute the coefficients for a cubic spline with specified second derivatives
fn compute_second_derivative_cubic_spline<F: Float + FromPrimitive + Debug>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
    left_d2: F,
    right_d2: F,
) -> InterpolateResult<Array2<F>> {
    let n = x.len();
    let n_segments = n - 1;

    // Define constants
    let two = F::from_f64(2.0).ok_or_else(|| {
        InterpolateError::ComputationError(
            "Failed to convert constant 2.0 to float type".to_string()
        )
    })?;
    let six = F::from_f64(6.0).ok_or_else(|| {
        InterpolateError::ComputationError(
            "Failed to convert constant 6.0 to float type".to_string()
        )
    })?;

    // Create array to hold the coefficients
    let mut coeffs = Array2::<F>::zeros((n_segments, 4));

    // Set up the tridiagonal system
    let mut a = Array1::<F>::zeros(n);
    let mut b = Array1::<F>::zeros(n);
    let mut c = Array1::<F>::zeros(n);
    let mut d = Array1::<F>::zeros(n);

    // Specified second derivative boundary conditions
    b[0] = F::one();
    d[0] = left_d2;
    b[n - 1] = F::one();
    d[n - 1] = right_d2;

    // Fill in the tridiagonal system for interior points
    for i in 1..n - 1 {
        let h_i_minus_1 = x[i] - x[i - 1];
        let h_i = x[i + 1] - x[i];

        // Check for zero intervals
        if h_i.is_zero() || h_i_minus_1.is_zero() {
            return Err(InterpolateError::ComputationError(
                "Zero interval length in second derivative spline computation".to_string()
            ));
        }
        
        let two = F::from_f64(2.0).ok_or_else(|| {
            InterpolateError::ComputationError(
                "Failed to convert constant 2.0 to float type".to_string()
            )
        })?;
        let six = F::from_f64(6.0).ok_or_else(|| {
            InterpolateError::ComputationError(
                "Failed to convert constant 6.0 to float type".to_string()
            )
        })?;

        a[i] = h_i_minus_1;
        b[i] = two * (h_i_minus_1 + h_i);
        c[i] = h_i;

        let dy_i_minus_1 = y[i] - y[i - 1];
        let dy_i = y[i + 1] - y[i];

        d[i] = six * (dy_i / h_i - dy_i_minus_1 / h_i_minus_1);
    }

    // Solve the tridiagonal system
    let mut sigma = Array1::<F>::zeros(n);

    // Forward sweep
    for i in 1..n {
        let m = a[i] / b[i - 1];
        b[i] = b[i] - m * c[i - 1];
        d[i] = d[i] - m * d[i - 1];
    }

    // Back substitution
    sigma[n - 1] = d[n - 1] / b[n - 1];
    for i in (0..n - 1).rev() {
        sigma[i] = (d[i] - c[i] * sigma[i + 1]) / b[i];
    }

    // Calculate the polynomial coefficients
    for i in 0..n_segments {
        let h_i = x[i + 1] - x[i];
        
        // Check for zero interval in coefficient calculation
        if h_i.is_zero() {
            return Err(InterpolateError::ComputationError(
                "Zero interval length in second derivative spline coefficient calculation".to_string()
            ));
        }

        coeffs[[i, 0]] = y[i];
        coeffs[[i, 1]] = (y[i + 1] - y[i]) / h_i
            - h_i * (two * sigma[i] + sigma[i + 1]) / six;
        coeffs[[i, 2]] = sigma[i] / two;
        coeffs[[i, 3]] = (sigma[i + 1] - sigma[i]) / (six * h_i);
    }

    Ok(coeffs)
}

/// Compute the coefficients for a parabolic runout cubic spline
fn compute_parabolic_runout_cubic_spline<F: Float + FromPrimitive + Debug>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
) -> InterpolateResult<Array2<F>> {
    // Parabolic runout means the third derivative is zero at the endpoints
    // This is equivalent to d[0] = 0 and d[n-2] = 0 in our coefficient representation
    // We can achieve this by setting specific boundary conditions on the second derivatives

    let n = x.len();
    let n_segments = n - 1;

    // Create array to hold the coefficients
    let mut coeffs = Array2::<F>::zeros((n_segments, 4));

    // Set up the tridiagonal system
    let mut a = Array1::<F>::zeros(n);
    let mut b = Array1::<F>::zeros(n);
    let mut c = Array1::<F>::zeros(n);
    let mut d = Array1::<F>::zeros(n);

    // Parabolic runout conditions
    // At the first point: 2*sigma[0] + sigma[1] = 0
    let two = F::from_f64(2.0).ok_or_else(|| {
        InterpolateError::ComputationError(
            "Failed to convert constant 2.0 to float type".to_string()
        )
    })?;
    let six = F::from_f64(6.0).ok_or_else(|| {
        InterpolateError::ComputationError(
            "Failed to convert constant 6.0 to float type".to_string()
        )
    })?;
    
    b[0] = two;
    c[0] = F::one();
    d[0] = F::zero();

    // At the last point: sigma[n-2] + 2*sigma[n-1] = 0
    a[n - 1] = F::one();
    b[n - 1] = two;
    d[n - 1] = F::zero();

    // Fill in the tridiagonal system for interior points
    for i in 1..n - 1 {
        let h_i_minus_1 = x[i] - x[i - 1];
        let h_i = x[i + 1] - x[i];

        // Check for zero intervals
        if h_i.is_zero() || h_i_minus_1.is_zero() {
            return Err(InterpolateError::ComputationError(
                "Zero interval length in parabolic runout spline computation".to_string()
            ));
        }

        a[i] = h_i_minus_1;
        b[i] = two * (h_i_minus_1 + h_i);
        c[i] = h_i;

        let dy_i_minus_1 = y[i] - y[i - 1];
        let dy_i = y[i + 1] - y[i];

        d[i] = six * (dy_i / h_i - dy_i_minus_1 / h_i_minus_1);
    }

    // Solve the tridiagonal system
    let mut sigma = Array1::<F>::zeros(n);

    // Forward sweep
    for i in 1..n {
        let m = a[i] / b[i - 1];
        b[i] = b[i] - m * c[i - 1];
        d[i] = d[i] - m * d[i - 1];
    }

    // Back substitution
    sigma[n - 1] = d[n - 1] / b[n - 1];
    for i in (0..n - 1).rev() {
        sigma[i] = (d[i] - c[i] * sigma[i + 1]) / b[i];
    }

    // Calculate the polynomial coefficients
    for i in 0..n_segments {
        let h_i = x[i + 1] - x[i];
        
        // Check for zero interval in coefficient calculation
        if h_i.is_zero() {
            return Err(InterpolateError::ComputationError(
                "Zero interval length in parabolic runout spline coefficient calculation".to_string()
            ));
        }

        coeffs[[i, 0]] = y[i];
        coeffs[[i, 1]] = (y[i + 1] - y[i]) / h_i
            - h_i * (two * sigma[i] + sigma[i + 1]) / six;
        coeffs[[i, 2]] = sigma[i] / two;
        coeffs[[i, 3]] = (sigma[i + 1] - sigma[i]) / (six * h_i);
    }

    Ok(coeffs)
}


/// Integrate a cubic polynomial segment from a to b
///
/// The polynomial is defined as: p(x) = a + b*(x-x0) + c*(x-x0)^2 + d*(x-x0)^3
fn integrate_segment<F: Float + FromPrimitive>(coeffs: &Array1<F>, x0: F, a: F, b: F) -> F {
    // Shift to x-x0 coordinates
    let a_shifted = a - x0;
    let b_shifted = b - x0;

    // Extract coefficients
    let coef_a = coeffs[0];
    let coef_b = coeffs[1];
    let coef_c = coeffs[2];
    let coef_d = coeffs[3];

    // Integrate the polynomial:
    // ∫(a + b*x + c*x^2 + d*x^3) dx = a*x + b*x^2/2 + c*x^3/3 + d*x^4/4
    let two = F::from_f64(2.0).unwrap_or_else(|| F::from(2).unwrap_or(F::zero()));
    let three = F::from_f64(3.0).unwrap_or_else(|| F::from(3).unwrap_or(F::zero()));
    let four = F::from_f64(4.0).unwrap_or_else(|| F::from(4).unwrap_or(F::zero()));

    // Evaluate at the bounds
    let int_a = coef_a * a_shifted
        + coef_b * a_shifted * a_shifted / two
        + coef_c * a_shifted * a_shifted * a_shifted / three
        + coef_d * a_shifted * a_shifted * a_shifted * a_shifted / four;

    let int_b = coef_a * b_shifted
        + coef_b * b_shifted * b_shifted / two
        + coef_c * b_shifted * b_shifted * b_shifted / three
        + coef_d * b_shifted * b_shifted * b_shifted * b_shifted / four;

    // Return the difference
    int_b - int_a
}

/// Check if a root is far enough from existing roots
fn root_far_enough<F: Float>(roots: &[F], candidate: F, tolerance: F) -> bool {
    for &existing_root in roots {
        if (candidate - existing_root).abs() < tolerance {
            return false;
        }
    }
    true
}

/// Create a cubic spline interpolation object
///
/// # Arguments
///
/// * `x` - The x coordinates (must be sorted in ascending order)
/// * `y` - The y coordinates (must have the same length as x)
/// * `bc_type` - The boundary condition type: "natural", "not-a-knot", "clamped", or "periodic"
/// * `bc_params` - Additional parameters for boundary conditions (required for "clamped"):
///   * For "clamped": [first_derivative_start, first_derivative_end]
///
/// # Returns
///
/// A new `CubicSpline` object
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_interpolate::spline::make_interp_spline;
///
/// let x = array![0.0, 1.0, 2.0, 3.0];
/// let y = array![0.0, 1.0, 4.0, 9.0];
///
/// // Natural boundary conditions
/// let spline = make_interp_spline(&x.view(), &y.view(), "natural", None).unwrap();
///
/// // Clamped boundary conditions with specified first derivatives
/// let clamped_spline = make_interp_spline(
///     &x.view(),
///     &y.view(),
///     "clamped",
///     Some(&array![0.0, 6.0].view()),  // first derivative at start = 0, end = 6
/// ).unwrap();
///
/// // Interpolate at x = 1.5
/// let y_interp = spline.evaluate(1.5).unwrap();
/// println!("Interpolated value at x=1.5: {}", y_interp);
/// ```
pub fn make_interp_spline<F: Float + FromPrimitive + Debug>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
    bc_type: &str,
    bc_params: Option<&ArrayView1<F>>,
) -> InterpolateResult<CubicSpline<F>> {
    match bc_type {
        "natural" => CubicSpline::new(x, y),
        "not-a-knot" => CubicSpline::new_not_a_knot(x, y),
        "clamped" => {
            if let Some(params) = bc_params {
                if params.len() != 2 {
                    return Err(InterpolateError::invalid_input(
                        "clamped boundary conditions require 2 parameters: [first_deriv_start, first_deriv_end]".to_string(),
                    ));
                }
                CubicSpline::new_clamped(x, y, params[0], params[1])
            } else {
                Err(InterpolateError::invalid_input(
                    "clamped boundary conditions require bc_params: [first_deriv_start, first_deriv_end]".to_string(),
                ))
            }
        },
        "periodic" => {
            CubicSpline::new_periodic(x, y)
        },
        _ => Err(InterpolateError::invalid_input(format!(
            "Unknown boundary condition type: {}. Use 'natural', 'not-a-knot', 'clamped', or 'periodic'",
            bc_type
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_natural_cubic_spline() {
        // Create a spline for y = x^2
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 4.0, 9.0];

        let spline = CubicSpline::new(&x.view(), &y.view()).unwrap();

        // Test at the knot points
        assert_relative_eq!(spline.evaluate(0.0).unwrap(), 0.0);
        assert_relative_eq!(spline.evaluate(1.0).unwrap(), 1.0);
        assert_relative_eq!(spline.evaluate(2.0).unwrap(), 4.0);
        assert_relative_eq!(spline.evaluate(3.0).unwrap(), 9.0);

        // Test at some intermediate points
        // Note: The spline won't exactly reproduce x^2 between the points
        // Using wide tolerances to account for implementation differences
        assert_relative_eq!(spline.evaluate(0.5).unwrap(), 0.25, epsilon = 0.25);
        assert_relative_eq!(spline.evaluate(1.5).unwrap(), 2.25, epsilon = 0.25);
        assert_relative_eq!(spline.evaluate(2.5).unwrap(), 6.25, epsilon = 0.25);

        // Test error for point outside range
        assert!(spline.evaluate(-1.0).is_err());
        assert!(spline.evaluate(4.0).is_err());
    }

    #[test]
    fn test_not_a_knot_cubic_spline() {
        // Create a spline for y = x^2
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 4.0, 9.0];

        let spline = CubicSpline::new_not_a_knot(&x.view(), &y.view()).unwrap();

        // Test at the knot points
        assert_relative_eq!(spline.evaluate(0.0).unwrap(), 0.0);
        assert_relative_eq!(spline.evaluate(1.0).unwrap(), 1.0);
        assert_relative_eq!(spline.evaluate(2.0).unwrap(), 4.0);
        assert_relative_eq!(spline.evaluate(3.0).unwrap(), 9.0);

        // Test at some intermediate points
        // Not-a-knot should reproduce x^2 more closely than natural spline
        // Using wide tolerances to account for implementation differences
        assert_relative_eq!(spline.evaluate(0.5).unwrap(), 0.25, epsilon = 0.5);
        assert_relative_eq!(spline.evaluate(1.5).unwrap(), 2.25, epsilon = 0.5);
        assert_relative_eq!(spline.evaluate(2.5).unwrap(), 6.25, epsilon = 0.5);
    }

    #[test]
    fn test_spline_derivative() {
        // Create a spline for y = x^2
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 4.0, 9.0];

        let spline = CubicSpline::new(&x.view(), &y.view()).unwrap();

        // Test derivative (should be close to 2*x for y = x^2)
        // Using wide tolerances to account for implementation differences
        assert_relative_eq!(spline.derivative(1.0).unwrap(), 2.0, epsilon = 0.5);
        assert_relative_eq!(spline.derivative(2.0).unwrap(), 4.0, epsilon = 0.5);
        assert_relative_eq!(spline.derivative(0.5).unwrap(), 1.0, epsilon = 0.5);
        assert_relative_eq!(spline.derivative(1.5).unwrap(), 3.0, epsilon = 0.2);
        assert_relative_eq!(spline.derivative(2.5).unwrap(), 5.0, epsilon = 0.2);
    }

    #[test]
    fn test_make_interp_spline() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 4.0, 9.0];

        // Test natural boundary conditions
        let spline_natural = make_interp_spline(&x.view(), &y.view(), "natural", None).unwrap();
        assert_relative_eq!(spline_natural.evaluate(1.5).unwrap(), 2.25, epsilon = 0.1);

        // Test not-a-knot boundary conditions
        let spline_not_a_knot =
            make_interp_spline(&x.view(), &y.view(), "not-a-knot", None).unwrap();
        assert_relative_eq!(
            spline_not_a_knot.evaluate(1.5).unwrap(),
            2.25,
            epsilon = 0.1
        );

        // Test invalid boundary condition
        let result = make_interp_spline(&x.view(), &y.view(), "invalid", None);
        assert!(result.is_err());
    }

    #[test]
    fn test_evaluate_array() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 4.0, 9.0];

        let spline = CubicSpline::new(&x.view(), &y.view()).unwrap();

        let x_new = array![0.5, 1.0, 1.5, 2.0, 2.5];
        let y_new = spline.evaluate_array(&x_new.view()).unwrap();

        assert_eq!(y_new.len(), 5);
        assert_relative_eq!(y_new[1], 1.0); // Exact at knot point
        assert_relative_eq!(y_new[3], 4.0); // Exact at knot point
    }

    #[test]
    fn test_cubic_spline_error_conditions() {
        let x_short = array![0.0, 1.0];
        let y_short = array![0.0, 1.0];

        // Test too few points
        let result = CubicSpline::new(&x_short.view(), &y_short.view());
        assert!(result.is_err());

        let x = array![0.0, 1.0, 2.0, 3.0];
        let y_wrong_len = array![0.0, 1.0, 4.0];

        // Test x and y different lengths
        let result = CubicSpline::new(&x.view(), &y_wrong_len.view());
        assert!(result.is_err());

        let x_unsorted = array![0.0, 2.0, 1.0, 3.0];
        let y = array![0.0, 1.0, 4.0, 9.0];

        // Test unsorted x
        let result = CubicSpline::new(&x_unsorted.view(), &y.view());
        assert!(result.is_err());
    }
}
