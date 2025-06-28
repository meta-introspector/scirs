//! Natural Neighbor interpolation methods
//!
//! This module implements Natural Neighbor interpolation, a spatial interpolation
//! technique based on Voronoi diagrams. This method is well-suited for irregularly
//! scattered data and produces a smooth interpolation that adapts to local data density.
//!
//! Natural Neighbor interpolation works by inserting the query point into the Voronoi
//! diagram of the data points and calculating how much the Voronoi cell of each data point
//! would be "stolen" by the query point. These proportions are used as weights for the
//! interpolation.
//!
//! The implementation uses the Sibson method for 2D interpolation, which calculates
//! the natural neighbor coordinates based on the areas of Voronoi cells.

use crate::delaunay::Delaunay;
use crate::error::{SpatialError, SpatialResult};
use crate::voronoi::Voronoi;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::collections::HashMap;
use std::fmt;

/// Natural Neighbor interpolator for scattered data
///
/// This interpolator uses the Sibson method to compute natural neighbor
/// coordinates based on Voronoi diagrams and Delaunay triangulation.
///
/// # Examples
///
/// ```
/// use scirs2_spatial::interpolate::NaturalNeighborInterpolator;
/// use ndarray::array;
///
/// // Create sample points and values
/// let points = array![
///     [0.0, 0.0],
///     [1.0, 0.0],
///     [0.0, 1.0],
///     [1.0, 1.0],
/// ];
/// let values = array![0.0, 1.0, 2.0, 3.0];
///
/// // Create interpolator
/// let interp = NaturalNeighborInterpolator::new(&points.view(), &values.view()).unwrap();
///
/// // Interpolate at a point
/// let query_point = array![0.5, 0.5];
/// let result = interp.interpolate(&query_point.view()).unwrap();
///
/// // Should be close to 1.5 (average of the 4 corners)
/// assert!((result - 1.5).abs() < 1e-10);
/// // Note: This test is currently ignored due to implementation issues
/// ```
pub struct NaturalNeighborInterpolator {
    /// Input points (N x D)
    points: Array2<f64>,
    /// Input values (N)
    values: Array1<f64>,
    /// Delaunay triangulation of the input points
    delaunay: Delaunay,
    /// Voronoi diagram of the input points
    voronoi: Voronoi,
    /// Dimensionality of the input points
    dim: usize,
    /// Number of input points
    n_points: usize,
}

impl Clone for NaturalNeighborInterpolator {
    fn clone(&self) -> Self {
        // We need to recreate the Delaunay and Voronoi structures
        let delaunay = Delaunay::new(&self.points).unwrap();
        let voronoi = Voronoi::new(&self.points.view(), false).unwrap();

        Self {
            points: self.points.clone(),
            values: self.values.clone(),
            delaunay,
            voronoi,
            dim: self.dim,
            n_points: self.n_points,
        }
    }
}

impl fmt::Debug for NaturalNeighborInterpolator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NaturalNeighborInterpolator")
            .field("dim", &self.dim)
            .field("n_points", &self.n_points)
            .field("points_shape", &self.points.shape())
            .field("values_len", &self.values.len())
            .finish()
    }
}

impl NaturalNeighborInterpolator {
    /// Create a new natural neighbor interpolator
    ///
    /// # Arguments
    ///
    /// * `points` - Input points with shape (n_samples, n_dims)
    /// * `values` - Input values with shape (n_samples,)
    ///
    /// # Returns
    ///
    /// A new NaturalNeighborInterpolator
    ///
    /// # Errors
    ///
    /// * If points and values have different lengths
    /// * If points are not 2D
    /// * If fewer than 3 points are provided
    /// * If the Delaunay triangulation fails
    pub fn new(points: &ArrayView2<f64>, values: &ArrayView1<f64>) -> SpatialResult<Self> {
        // Check input dimensions
        let n_points = points.nrows();
        let dim = points.ncols();

        if n_points != values.len() {
            return Err(SpatialError::DimensionError(format!(
                "Number of points ({}) must match number of values ({})",
                n_points,
                values.len()
            )));
        }

        if dim != 2 {
            return Err(SpatialError::DimensionError(format!(
                "Natural neighbor interpolation currently only supports 2D points, got {}D",
                dim
            )));
        }

        if n_points < 3 {
            return Err(SpatialError::ValueError(
                "Natural neighbor interpolation requires at least 3 points".to_string(),
            ));
        }

        // Create Delaunay triangulation
        let delaunay = Delaunay::new(&points.to_owned())?;

        // Create Voronoi diagram
        let voronoi = Voronoi::new(points, false)?;

        Ok(Self {
            points: points.to_owned(),
            values: values.to_owned(),
            delaunay,
            voronoi,
            dim,
            n_points,
        })
    }

    /// Interpolate at a single point
    ///
    /// # Arguments
    ///
    /// * `point` - Query point with shape (n_dims,)
    ///
    /// # Returns
    ///
    /// Interpolated value at the query point
    ///
    /// # Errors
    ///
    /// * If the point dimensions don't match the interpolator
    /// * If the point is outside the convex hull of the input points
    pub fn interpolate(&self, point: &ArrayView1<f64>) -> SpatialResult<f64> {
        // Check dimension
        if point.len() != self.dim {
            return Err(SpatialError::DimensionError(format!(
                "Query point has dimension {}, expected {}",
                point.len(),
                self.dim
            )));
        }

        // Find the simplex (triangle) containing the point
        let simplex_idx = self.delaunay.find_simplex(point.as_slice().unwrap());

        if simplex_idx.is_none() {
            return Err(SpatialError::ValueError(
                "Query point is outside the convex hull of the input points".to_string(),
            ));
        }

        // Get the natural neighbor coordinates
        let weights = self.natural_neighbor_weights(point)?;

        // Compute the weighted sum
        let mut result = 0.0;
        for (idx, weight) in weights {
            result += weight * self.values[idx];
        }

        Ok(result)
    }

    /// Interpolate at multiple points
    ///
    /// # Arguments
    ///
    /// * `points` - Query points with shape (n_queries, n_dims)
    ///
    /// # Returns
    ///
    /// Interpolated values with shape (n_queries,)
    ///
    /// # Errors
    ///
    /// * If the points dimensions don't match the interpolator
    pub fn interpolate_many(&self, points: &ArrayView2<f64>) -> SpatialResult<Array1<f64>> {
        // Check dimensions
        if points.ncols() != self.dim {
            return Err(SpatialError::DimensionError(format!(
                "Query points have dimension {}, expected {}",
                points.ncols(),
                self.dim
            )));
        }

        let n_queries = points.nrows();
        let mut results = Array1::zeros(n_queries);

        // Interpolate each point
        for i in 0..n_queries {
            let point = points.row(i);

            // Handle points outside the convex hull by returning NaN
            match self.interpolate(&point) {
                Ok(value) => results[i] = value,
                Err(_) => results[i] = f64::NAN,
            }
        }

        Ok(results)
    }

    /// Compute the natural neighbor weights for a query point
    ///
    /// # Arguments
    ///
    /// * `point` - Query point with shape (n_dims,)
    ///
    /// # Returns
    ///
    /// A HashMap mapping point indices to their natural neighbor weights
    ///
    /// # Errors
    ///
    /// * If the point is outside the convex hull of the input points
    /// * If the weights cannot be computed
    fn natural_neighbor_weights(
        &self,
        point: &ArrayView1<f64>,
    ) -> SpatialResult<HashMap<usize, f64>> {
        // This implementation uses the Sibson method which computes the natural
        // neighbor coordinates based on the "stolen area" when inserting the query point
        // into the Voronoi diagram.

        // First, find the triangle containing the query point
        let simplex_idx = self.delaunay.find_simplex(point.as_slice().unwrap());

        if simplex_idx.is_none() {
            return Err(SpatialError::ValueError(
                "Query point is outside the convex hull of the input points".to_string(),
            ));
        }

        let simplex_idx = simplex_idx.unwrap();
        let simplex = &self.delaunay.simplices()[simplex_idx];

        // For 2D interpolation, implement proper Sibson natural neighbor interpolation
        if self.dim == 2 {
            // Get the natural neighbors - points whose Voronoi cells will be affected
            let natural_neighbors = self.find_natural_neighbors(point, simplex)?;

            // Create augmented point set with the query point
            let mut augmented_points = self.points.clone();
            augmented_points.push_row(point.view()).map_err(|_| {
                SpatialError::ComputationError("Failed to add query point".to_string())
            })?;

            // Create new Voronoi diagram with the query point included
            let augmented_voronoi = match Voronoi::new(&augmented_points.view(), false) {
                Ok(v) => v,
                Err(_) => {
                    // Fall back to barycentric coordinates if Voronoi fails
                    return self.barycentric_weights_as_map(point, simplex_idx);
                }
            };

            // Compute the stolen areas
            let mut weights = HashMap::new();
            let mut total_weight = 0.0;

            // The query point is the last point in the augmented set
            let query_idx = augmented_points.nrows() - 1;
            let query_region = &augmented_voronoi.regions()[query_idx];

            // Get vertices of the query point's Voronoi cell
            let _query_vertices = match Self::get_voronoi_vertices(&augmented_voronoi, query_region)
            {
                Ok(v) => v,
                Err(_) => {
                    // Fall back to barycentric coordinates
                    return self.barycentric_weights_as_map(point, simplex_idx);
                }
            };

            // For each natural neighbor, compute the intersection area
            for &neighbor_idx in &natural_neighbors {
                // Get the original Voronoi cell area
                let orig_region = &self.voronoi.regions()[neighbor_idx];
                let orig_vertices = match Self::get_voronoi_vertices(&self.voronoi, orig_region) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let orig_area = match Self::polygon_area(&orig_vertices) {
                    Ok(a) => a,
                    Err(_) => continue,
                };

                // Get the new Voronoi cell area after adding query point
                let new_region = &augmented_voronoi.regions()[neighbor_idx];
                let new_vertices = match Self::get_voronoi_vertices(&augmented_voronoi, new_region)
                {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let new_area = match Self::polygon_area(&new_vertices) {
                    Ok(a) => a,
                    Err(_) => continue,
                };

                // The stolen area is the difference
                let stolen_area = (orig_area - new_area).abs();
                if stolen_area > 1e-10 {
                    weights.insert(neighbor_idx, stolen_area);
                    total_weight += stolen_area;
                }
            }

            // If we couldn't compute any weights, fall back to barycentric
            if weights.is_empty() || total_weight <= 1e-10 {
                return self.barycentric_weights_as_map(point, simplex_idx);
            }

            // Normalize the weights
            for weight in weights.values_mut() {
                *weight /= total_weight;
            }

            Ok(weights)
        } else {
            // For dimensions other than 2, use barycentric coordinates
            self.barycentric_weights_as_map(point, simplex_idx)
        }
    }

    /// Compute barycentric weights for a point in a simplex
    ///
    /// # Arguments
    ///
    /// * `point` - Query point
    /// * `simplex_idx` - Index of the simplex containing the point
    ///
    /// # Returns
    ///
    /// Barycentric weights for the simplex vertices
    ///
    /// # Errors
    ///
    /// * If the barycentric coordinates cannot be computed
    fn barycentric_weights(
        &self,
        point: &ArrayView1<f64>,
        simplex_idx: usize,
    ) -> SpatialResult<Vec<f64>> {
        let simplex = &self.delaunay.simplices()[simplex_idx];
        let mut vertices = Vec::new();

        for &idx in simplex {
            vertices.push(self.points.row(idx));
        }

        // For 2D, we have a triangle
        if vertices.len() != 3 {
            return Err(SpatialError::ValueError(format!(
                "Expected 3 vertices for 2D triangle, got {}",
                vertices.len()
            )));
        }

        // Compute barycentric coordinates
        let a = vertices[0];
        let b = vertices[1];
        let c = vertices[2];
        let p = point;

        let v0_x = b[0] - a[0];
        let v0_y = b[1] - a[1];
        let v1_x = c[0] - a[0];
        let v1_y = c[1] - a[1];
        let v2_x = p[0] - a[0];
        let v2_y = p[1] - a[1];

        let d00 = v0_x * v0_x + v0_y * v0_y;
        let d01 = v0_x * v1_x + v0_y * v1_y;
        let d11 = v1_x * v1_x + v1_y * v1_y;
        let d20 = v2_x * v0_x + v2_y * v0_y;
        let d21 = v2_x * v1_x + v2_y * v1_y;

        let denom = d00 * d11 - d01 * d01;
        if denom.abs() < 1e-10 {
            return Err(SpatialError::ValueError(
                "Degenerate triangle, cannot compute barycentric coordinates".to_string(),
            ));
        }

        let v = (d11 * d20 - d01 * d21) / denom;
        let w = (d00 * d21 - d01 * d20) / denom;
        let u = 1.0 - v - w;

        Ok(vec![u, v, w])
    }

    /// Get the vertices of a Voronoi region
    ///
    /// # Arguments
    ///
    /// * `voronoi` - Voronoi diagram
    /// * `region` - Indices of vertices in the region
    ///
    /// # Returns
    ///
    /// Array of vertex coordinates
    ///
    /// # Errors
    ///
    /// * If the region is empty
    fn get_voronoi_vertices(voronoi: &Voronoi, region: &[i64]) -> SpatialResult<Array2<f64>> {
        if region.is_empty() {
            return Err(SpatialError::ValueError("Empty Voronoi region".to_string()));
        }

        // Count valid vertices (ignoring -1)
        let valid_count = region.iter().filter(|&&idx| idx >= 0).count();

        if valid_count == 0 {
            return Err(SpatialError::ValueError(
                "All vertices are at infinity".to_string(),
            ));
        }

        let mut vertices = Array2::zeros((valid_count, 2));
        let mut j = 0;

        for &idx in region.iter() {
            if idx >= 0 {
                vertices
                    .row_mut(j)
                    .assign(&voronoi.vertices().row(idx as usize));
                j += 1;
            }
        }

        Ok(vertices)
    }

    /// Compute the area of a polygon
    ///
    /// # Arguments
    ///
    /// * `vertices` - Polygon vertices in counter-clockwise order
    ///
    /// # Returns
    ///
    /// Area of the polygon
    ///
    /// # Errors
    ///
    /// * If the polygon has fewer than 3 vertices
    fn polygon_area(vertices: &Array2<f64>) -> SpatialResult<f64> {
        let n = vertices.nrows();

        if n < 3 {
            return Err(SpatialError::ValueError(format!(
                "Polygon must have at least 3 vertices, got {}",
                n
            )));
        }

        let mut area = 0.0;

        for i in 0..n {
            let j = (i + 1) % n;
            area += vertices[[i, 0]] * vertices[[j, 1]] - vertices[[j, 0]] * vertices[[i, 1]];
        }

        Ok(area.abs() / 2.0)
    }

    /// Compute the Euclidean distance between two points
    ///
    /// # Arguments
    ///
    /// * `p1` - First point
    /// * `p2` - Second point
    ///
    /// # Returns
    ///
    /// Euclidean distance between the points
    fn euclidean_distance(p1: &ArrayView1<f64>, p2: &ArrayView1<f64>) -> f64 {
        let mut sum_sq = 0.0;
        for i in 0..p1.len().min(p2.len()) {
            let diff = p1[i] - p2[i];
            sum_sq += diff * diff;
        }
        sum_sq.sqrt()
    }

    /// Find the natural neighbors of a query point
    ///
    /// Natural neighbors are points whose Voronoi cells would be affected
    /// by inserting the query point into the diagram.
    fn find_natural_neighbors(
        &self,
        point: &ArrayView1<f64>,
        simplex: &[usize],
    ) -> SpatialResult<Vec<usize>> {
        let mut neighbors = Vec::new();

        // Add vertices of the containing simplex
        for &idx in simplex {
            neighbors.push(idx);
        }

        // Find additional neighbors by checking which Voronoi cells
        // the query point would intersect
        let circumradius = self.compute_circumradius(simplex)?;

        // Check points within a reasonable distance
        for i in 0..self.n_points {
            if !neighbors.contains(&i) {
                let dist = Self::euclidean_distance(point, &self.points.row(i));
                // Include points within 2x the circumradius as potential neighbors
                if dist <= 2.0 * circumradius {
                    neighbors.push(i);
                }
            }
        }

        Ok(neighbors)
    }

    /// Compute the circumradius of a simplex
    fn compute_circumradius(&self, simplex: &[usize]) -> SpatialResult<f64> {
        if simplex.len() != 3 || self.dim != 2 {
            return Err(SpatialError::ValueError(
                "Circumradius computation only supported for 2D triangles".to_string(),
            ));
        }

        let a = self.points.row(simplex[0]);
        let b = self.points.row(simplex[1]);
        let c = self.points.row(simplex[2]);

        // Compute side lengths
        let ab = Self::euclidean_distance(&a, &b);
        let bc = Self::euclidean_distance(&b, &c);
        let ca = Self::euclidean_distance(&c, &a);

        // Compute area using Heron's formula
        let s = (ab + bc + ca) / 2.0;
        let area = (s * (s - ab) * (s - bc) * (s - ca)).sqrt();

        if area < 1e-10 {
            return Err(SpatialError::ValueError("Degenerate triangle".to_string()));
        }

        // Circumradius = (abc) / (4 * Area)
        Ok((ab * bc * ca) / (4.0 * area))
    }

    /// Convert barycentric weights to a HashMap format
    fn barycentric_weights_as_map(
        &self,
        point: &ArrayView1<f64>,
        simplex_idx: usize,
    ) -> SpatialResult<HashMap<usize, f64>> {
        let simplex = &self.delaunay.simplices()[simplex_idx];
        let bary_weights = self.barycentric_weights(point, simplex_idx)?;

        let mut weights = HashMap::new();
        for (i, &idx) in simplex.iter().enumerate() {
            if bary_weights[i] > 1e-10 {
                weights.insert(idx, bary_weights[i]);
            }
        }

        Ok(weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_natural_neighbor_interpolator() {
        // Create sample points in a square
        let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],];
        let values = array![0.0, 1.0, 2.0, 3.0];

        // Create interpolator
        let interp = NaturalNeighborInterpolator::new(&points.view(), &values.view()).unwrap();

        // Test interpolation at center point
        let query_point = array![0.5, 0.5];
        let result = interp.interpolate(&query_point.view()).unwrap();

        // The center of the square should have equal weights from all corners
        // Expected value: (0 + 1 + 2 + 3) / 4 = 1.5
        assert!((result - 1.5).abs() < 0.1, "Expected ~1.5, got {}", result);

        // Test interpolation at a corner (should return exact value)
        let corner = array![0.0, 0.0];
        let corner_result = interp.interpolate(&corner.view()).unwrap();
        assert!(
            (corner_result - 0.0).abs() < 1e-6,
            "Expected 0.0 at corner, got {}",
            corner_result
        );
    }

    #[test]
    fn test_outside_convex_hull() {
        // Create triangle points
        let points = array![[0.0, 0.0], [1.0, 0.0], [0.5, 1.0],];
        let values = array![0.0, 1.0, 2.0];

        let interp = NaturalNeighborInterpolator::new(&points.view(), &values.view()).unwrap();

        // Test point outside convex hull
        let outside_point = array![2.0, 2.0];
        let result = interp.interpolate(&outside_point.view());

        assert!(
            result.is_err(),
            "Expected error for point outside convex hull"
        );

        // Test interpolate_many with mixed points
        let query_points = array![
            [0.5, 0.5],   // Inside
            [2.0, 2.0],   // Outside
            [0.25, 0.25], // Inside
        ];

        let results = interp.interpolate_many(&query_points.view()).unwrap();
        assert!(
            !results[0].is_nan(),
            "Inside point should have valid result"
        );
        assert!(results[1].is_nan(), "Outside point should return NaN");
        assert!(
            !results[2].is_nan(),
            "Inside point should have valid result"
        );
    }

    #[test]
    fn test_error_handling() {
        // Not enough points
        let points = array![[0.0, 0.0], [1.0, 1.0]];
        let values = array![0.0, 1.0];

        let result = NaturalNeighborInterpolator::new(&points.view(), &values.view());
        assert!(result.is_err());

        // Wrong dimensions
        let points_3d = array![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]];
        let values = array![0.0, 1.0, 2.0];

        let result = NaturalNeighborInterpolator::new(&points_3d.view(), &values.view());
        assert!(result.is_err());

        // Mismatched lengths
        let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let values = array![0.0, 1.0];

        let result = NaturalNeighborInterpolator::new(&points.view(), &values.view());
        assert!(result.is_err());
    }
}
