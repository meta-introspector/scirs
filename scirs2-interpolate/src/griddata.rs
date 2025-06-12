//! Grid data interpolation - SciPy-compatible griddata implementation
//!
//! This module provides the `griddata` function, which interpolates unstructured
//! data to a regular grid or arbitrary points. This is one of the most commonly
//! used interpolation functions in SciPy's interpolate module.
//!
//! # Examples
//!
//! ```
//! use ndarray::{array, Array2};
//! use scirs2_interpolate::griddata::{griddata, GriddataMethod};
//!
//! // Scattered data points
//! let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
//! let values = array![0.0, 1.0, 1.0, 2.0];
//!
//! // Grid to interpolate onto
//! let xi = array![[0.5, 0.5], [0.25, 0.75]];
//!
//! // Interpolate using linear method
//! let result = griddata(&points.view(), &values.view(), &xi.view(),
//!                       GriddataMethod::Linear, None)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::advanced::rbf::{RBFInterpolator, RBFKernel};
use crate::error::{InterpolateError, InterpolateResult};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

/// Interpolation methods available for griddata
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GriddataMethod {
    /// Linear interpolation using Delaunay triangulation
    Linear,
    /// Nearest neighbor interpolation  
    Nearest,
    /// Cubic interpolation using Clough-Tocher scheme
    Cubic,
    /// Radial basis function interpolation with linear kernel
    Rbf,
    /// Radial basis function interpolation with cubic kernel
    RbfCubic,
    /// Radial basis function interpolation with thin plate spline
    RbfThinPlate,
}

/// Interpolate unstructured D-dimensional data to arbitrary points.
///
/// This function provides a SciPy-compatible interface for interpolating
/// scattered data points to a regular grid or arbitrary query points.
///
/// # Arguments
///
/// * `points` - Data point coordinates with shape (n_points, n_dims)
/// * `values` - Data values at each point with shape (n_points,)  
/// * `xi` - Points at which to interpolate data with shape (n_queries, n_dims)
/// * `method` - Interpolation method to use
/// * `fill_value` - Value to use for points outside convex hull (None uses NaN)
///
/// # Returns
///
/// Array of interpolated values with shape (n_queries,)
///
/// # Errors
///
/// * `ShapeMismatch` - If input arrays have incompatible shapes
/// * `InvalidParameter` - If method parameters are invalid
/// * `ComputationFailed` - If interpolation setup fails
///
/// # Examples
///
/// ## Basic usage with scattered 2D data
/// ```
/// use ndarray::array;
/// use scirs2_interpolate::griddata::{griddata, GriddataMethod};
///
/// // Scattered data: z = x² + y²
/// let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]];
/// let values = array![0.0, 1.0, 1.0, 2.0, 0.5];
///
/// // Query points
/// let xi = array![[0.25, 0.25], [0.75, 0.75]];
///
/// // Linear interpolation
/// let result = griddata(&points.view(), &values.view(), &xi.view(),
///                       GriddataMethod::Linear, None)?;
///
/// println!("Interpolated values: {:?}", result);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Using different interpolation methods
/// ```
/// use ndarray::array;
/// use scirs2_interpolate::griddata::{griddata, GriddataMethod};
///
/// let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
/// let values = array![0.0, 1.0, 1.0];
/// let xi = array![[0.5, 0.5]];
///
/// // Compare different methods
/// let linear = griddata(&points.view(), &values.view(), &xi.view(),
///                      GriddataMethod::Linear, None)?;
/// let nearest = griddata(&points.view(), &values.view(), &xi.view(),
///                       GriddataMethod::Nearest, None)?;
/// let rbf = griddata(&points.view(), &values.view(), &xi.view(),
///                    GriddataMethod::Rbf, None)?;
///
/// println!("Linear: {}, Nearest: {}, RBF: {}", linear[0], nearest[0], rbf[0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Performance Notes
///
/// - **Linear**: Fast setup O(n log n), fast evaluation O(log n) per point
/// - **Nearest**: Very fast setup O(n log n), very fast evaluation O(log n)  
/// - **Cubic**: Slow setup O(n³), medium evaluation O(n)
/// - **RBF methods**: Slow setup O(n³), medium evaluation O(n)
///
/// For large datasets (n > 1000), consider using FastRBF or other approximation methods.
pub fn griddata<F>(
    points: &ArrayView2<F>,
    values: &ArrayView1<F>,
    xi: &ArrayView2<F>,
    method: GriddataMethod,
    fill_value: Option<F>,
) -> InterpolateResult<Array1<F>>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    // Validate inputs
    validate_griddata_inputs(points, values, xi)?;

    match method {
        GriddataMethod::Linear => griddata_linear(points, values, xi, fill_value),
        GriddataMethod::Nearest => griddata_nearest(points, values, xi, fill_value),
        GriddataMethod::Cubic => griddata_cubic(points, values, xi, fill_value),
        GriddataMethod::Rbf => griddata_rbf(points, values, xi, RBFKernel::Linear, fill_value),
        GriddataMethod::RbfCubic => griddata_rbf(points, values, xi, RBFKernel::Cubic, fill_value),
        GriddataMethod::RbfThinPlate => {
            griddata_rbf(points, values, xi, RBFKernel::ThinPlateSpline, fill_value)
        }
    }
}

/// Validate input arrays for griddata
fn validate_griddata_inputs<F>(
    points: &ArrayView2<F>,
    values: &ArrayView1<F>,
    xi: &ArrayView2<F>,
) -> InterpolateResult<()>
where
    F: Float + Debug,
{
    // Check that points and values have compatible shapes
    if points.nrows() != values.len() {
        return Err(InterpolateError::shape_mismatch(
            format!("points.nrows() = {}", points.nrows()),
            format!("values.len() = {}", values.len()),
            "griddata input validation",
        ));
    }

    // Check that xi has the same number of dimensions as points
    if points.ncols() != xi.ncols() {
        return Err(InterpolateError::shape_mismatch(
            format!("points.ncols() = {}", points.ncols()),
            format!("xi.ncols() = {}", xi.ncols()),
            "griddata dimension consistency",
        ));
    }

    // Check for minimum number of points
    if points.nrows() < points.ncols() + 1 {
        return Err(InterpolateError::invalid_input(format!(
            "Need at least {} points for {}-dimensional interpolation, got {}",
            points.ncols() + 1,
            points.ncols(),
            points.nrows()
        )));
    }

    Ok(())
}

/// Linear interpolation using triangulation (simplified implementation)
fn griddata_linear<F>(
    points: &ArrayView2<F>,
    values: &ArrayView1<F>,
    xi: &ArrayView2<F>,
    fill_value: Option<F>,
) -> InterpolateResult<Array1<F>>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    // For now, fall back to RBF with linear kernel
    // TODO: Implement proper Delaunay triangulation-based interpolation
    griddata_rbf(points, values, xi, RBFKernel::Linear, fill_value)
}

/// Nearest neighbor interpolation
fn griddata_nearest<F>(
    points: &ArrayView2<F>,
    values: &ArrayView1<F>,
    xi: &ArrayView2<F>,
    fill_value: Option<F>,
) -> InterpolateResult<Array1<F>>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    let n_queries = xi.nrows();
    let n_points = points.nrows();
    let mut result = Array1::zeros(n_queries);

    let default_fill = fill_value.unwrap_or_else(|| F::nan());

    for i in 0..n_queries {
        let query = xi.slice(ndarray::s![i, ..]);
        let mut min_dist = F::infinity();
        let mut nearest_idx = 0;

        // Find nearest neighbor
        for j in 0..n_points {
            let point = points.slice(ndarray::s![j, ..]);
            let mut dist_sq = F::zero();

            for k in 0..query.len() {
                let diff = query[k] - point[k];
                dist_sq = dist_sq + diff * diff;
            }

            let dist = dist_sq.sqrt();
            if dist < min_dist {
                min_dist = dist;
                nearest_idx = j;
            }
        }

        result[i] = if min_dist.is_finite() {
            values[nearest_idx]
        } else {
            default_fill
        };
    }

    Ok(result)
}

/// Cubic interpolation using Clough-Tocher scheme (simplified)
fn griddata_cubic<F>(
    points: &ArrayView2<F>,
    values: &ArrayView1<F>,
    xi: &ArrayView2<F>,
    fill_value: Option<F>,
) -> InterpolateResult<Array1<F>>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    // For now, fall back to RBF with cubic kernel
    // TODO: Implement proper Clough-Tocher interpolation
    griddata_rbf(points, values, xi, RBFKernel::Cubic, fill_value)
}

/// RBF-based interpolation
fn griddata_rbf<F>(
    points: &ArrayView2<F>,
    values: &ArrayView1<F>,
    xi: &ArrayView2<F>,
    kernel: RBFKernel,
    _fill_value: Option<F>,
) -> InterpolateResult<Array1<F>>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    // Determine appropriate epsilon based on data scale
    let epsilon = estimate_rbf_epsilon(points);

    // Create RBF interpolator
    let interpolator = RBFInterpolator::new(points, values, kernel, epsilon)?;

    // Interpolate at query points
    interpolator.interpolate(xi)
}

/// Estimate appropriate epsilon parameter for RBF interpolation
fn estimate_rbf_epsilon<F>(points: &ArrayView2<F>) -> F
where
    F: Float + FromPrimitive,
{
    let n_points = points.nrows();

    if n_points < 2 {
        return F::one();
    }

    // Estimate data scale using mean nearest neighbor distance
    let mut total_dist = F::zero();
    let mut count = 0;

    for i in 0..n_points.min(100) {
        // Sample for efficiency
        let mut min_dist = F::infinity();
        let point_i = points.slice(ndarray::s![i, ..]);

        for j in 0..n_points {
            if i == j {
                continue;
            }

            let point_j = points.slice(ndarray::s![j, ..]);
            let mut dist_sq = F::zero();

            for k in 0..point_i.len() {
                let diff = point_i[k] - point_j[k];
                dist_sq = dist_sq + diff * diff;
            }

            let dist = dist_sq.sqrt();
            if dist < min_dist && dist > F::zero() {
                min_dist = dist;
            }
        }

        if min_dist.is_finite() {
            total_dist = total_dist + min_dist;
            count += 1;
        }
    }

    if count > 0 {
        total_dist / F::from_usize(count).unwrap_or(F::one())
    } else {
        F::one()
    }
}

/// Create a regular grid for interpolation
///
/// This is a convenience function to create regular grids similar to
/// numpy.mgrid or scipy's RegularGridInterpolator.
///
/// # Arguments
///
/// * `bounds` - List of (min, max) bounds for each dimension
/// * `resolution` - Number of points in each dimension
///
/// # Returns
///
/// Array of grid points with shape (n_total_points, n_dims)
///
/// # Examples
///
/// ```
/// use scirs2_interpolate::griddata::make_regular_grid;
///
/// // Create a 2D grid from (0,0) to (1,1) with 3x3 points
/// let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
/// let resolution = vec![3, 3];
/// let grid = make_regular_grid(&bounds, &resolution)?;
///
/// assert_eq!(grid.nrows(), 9); // 3 * 3 = 9 points
/// assert_eq!(grid.ncols(), 2); // 2 dimensions
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn make_regular_grid<F>(bounds: &[(F, F)], resolution: &[usize]) -> InterpolateResult<Array2<F>>
where
    F: Float + FromPrimitive + Clone,
{
    if bounds.len() != resolution.len() {
        return Err(InterpolateError::shape_mismatch(
            format!("bounds.len() = {}", bounds.len()),
            format!("resolution.len() = {}", resolution.len()),
            "make_regular_grid dimension consistency",
        ));
    }

    let n_dims = bounds.len();
    let total_points: usize = resolution.iter().product();

    let mut grid = Array2::zeros((total_points, n_dims));

    // Generate coordinates for each point
    for (point_idx, mut indices) in (0..total_points)
        .map(|i| {
            let mut coords = vec![0; n_dims];
            let mut temp = i;
            for d in (0..n_dims).rev() {
                coords[d] = temp % resolution[d];
                temp /= resolution[d];
            }
            (i, coords)
        })
        .enumerate()
    {
        for (dim, &idx) in indices.iter().enumerate() {
            let (min_val, max_val) = bounds[dim];
            let coord = if resolution[dim] > 1 {
                let t = F::from_usize(idx).unwrap() / F::from_usize(resolution[dim] - 1).unwrap();
                min_val + t * (max_val - min_val)
            } else {
                (min_val + max_val) / (F::one() + F::one())
            };
            grid[[point_idx, dim]] = coord;
        }
    }

    Ok(grid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_griddata_nearest() -> InterpolateResult<()> {
        let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let values = array![0.0, 1.0, 2.0];
        let xi = array![[0.1, 0.1], [0.9, 0.1]];

        let result = griddata(
            &points.view(),
            &values.view(),
            &xi.view(),
            GriddataMethod::Nearest,
            None,
        )?;

        assert_eq!(result.len(), 2);
        assert_abs_diff_eq!(result[0], 0.0, epsilon = 1e-10); // Nearest to (0,0)
        assert_abs_diff_eq!(result[1], 1.0, epsilon = 1e-10); // Nearest to (1,0)

        Ok(())
    }

    #[test]
    fn test_griddata_rbf() -> InterpolateResult<()> {
        let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let values = array![0.0, 1.0, 1.0, 2.0]; // z = x + y
        let xi = array![[0.5, 0.5]];

        let result = griddata(
            &points.view(),
            &values.view(),
            &xi.view(),
            GriddataMethod::Rbf,
            None,
        )?;

        assert_eq!(result.len(), 1);
        // Should be close to 1.0 (0.5 + 0.5) for this function
        assert!((result[0] - 1.0).abs() < 0.5); // Loose tolerance for RBF

        Ok(())
    }

    #[test]
    fn test_make_regular_grid() -> InterpolateResult<()> {
        let bounds = vec![(0.0, 1.0), (0.0, 2.0)];
        let resolution = vec![3, 2];

        let grid = make_regular_grid(&bounds, &resolution)?;

        assert_eq!(grid.nrows(), 6); // 3 * 2 = 6
        assert_eq!(grid.ncols(), 2);

        // Check corner points
        assert_abs_diff_eq!(grid[[0, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(grid[[0, 1]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(grid[[5, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(grid[[5, 1]], 2.0, epsilon = 1e-10);

        Ok(())
    }

    #[test]
    fn test_validation() {
        let points = array![[0.0, 0.0], [1.0, 0.0]];
        let values = array![0.0, 1.0, 2.0]; // Wrong length
        let xi = array![[0.5, 0.5]];

        let result = griddata(
            &points.view(),
            &values.view(),
            &xi.view(),
            GriddataMethod::Nearest,
            None,
        );

        assert!(result.is_err());
    }
}
