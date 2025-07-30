//! Automatic Mesh Generation Interfaces
//!
//! This module provides automatic mesh generation capabilities for common geometric domains.
//! It includes algorithms for generating structured and unstructured meshes for various
//! shapes and domains, with quality control and refinement options.

use crate::pde::finite_element::{ElementType, Point, Triangle, TriangularMesh};
use crate::pde::{PDEError, PDEResult};
use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;

/// Parameters for controlling automatic mesh generation
#[derive(Debug, Clone)]
pub struct MeshGenerationParams {
    /// Target element size (average edge length)
    pub element_size: f64,
    /// Minimum angle constraint for triangles (degrees)
    pub min_angle: f64,
    /// Maximum angle constraint for triangles (degrees)
    pub max_angle: f64,
    /// Quality improvement iterations
    pub quality_iterations: usize,
    /// Element type to generate
    pub element_type: ElementType,
    /// Maximum number of boundary refinement iterations
    pub boundary_refinement_iterations: usize,
}

impl Default for MeshGenerationParams {
    fn default() -> Self {
        Self {
            element_size: 0.1,
            min_angle: 20.0,
            max_angle: 140.0,
            quality_iterations: 5,
            element_type: ElementType::Linear,
            boundary_refinement_iterations: 3,
        }
    }
}

/// Geometric domain types for automatic mesh generation
#[derive(Debug, Clone)]
pub enum Domain {
    /// Rectangle: (x_min, y_min, x_max, y_max)
    Rectangle {
        x_min: f64,
        y_min: f64,
        x_max: f64,
        y_max: f64,
    },
    /// Circle: (center_x, center_y, radius)
    Circle {
        center_x: f64,
        center_y: f64,
        radius: f64,
    },
    /// Ellipse: (center_x, center_y, a, b, rotation_angle)
    Ellipse {
        center_x: f64,
        center_y: f64,
        a: f64,
        b: f64,
        rotation: f64,
    },
    /// L-shaped domain
    LShape {
        width: f64,
        height: f64,
        notch_width: f64,
        notch_height: f64,
    },
    /// Custom polygon defined by vertices
    Polygon { vertices: Vec<Point> },
    /// Annulus (ring): (center_x, center_y, inner_radius, outer_radius)
    Annulus {
        center_x: f64,
        center_y: f64,
        inner_radius: f64,
        outer_radius: f64,
    },
}

/// Boundary condition specification for domain boundaries
#[derive(Debug, Clone, Default)]
pub struct BoundarySpecification {
    /// Boundary markers for different boundary segments
    pub boundary_markers: HashMap<String, i32>,
    /// Point markers for specific points
    pub point_markers: HashMap<String, i32>,
}

/// Quality metrics for mesh assessment
#[derive(Debug, Clone)]
pub struct MeshQuality {
    /// Minimum angle in degrees
    pub min_angle: f64,
    /// Maximum angle in degrees  
    pub max_angle: f64,
    /// Average element size
    pub avg_element_size: f64,
    /// Aspect ratio statistics
    pub min_aspect_ratio: f64,
    /// Number of poor quality elements
    pub poor_quality_elements: usize,
    /// Overall quality score (0-1, higher is better)
    pub quality_score: f64,
}

/// Main automatic mesh generator
pub struct AutoMeshGenerator {
    params: MeshGenerationParams,
}

impl Default for AutoMeshGenerator {
    fn default() -> Self {
        Self::new(MeshGenerationParams::default())
    }
}

impl AutoMeshGenerator {
    /// Create a new mesh generator with specified parameters
    pub fn new(_params: MeshGenerationParams) -> Self {
        Self { _params }
    }

    /// Create a mesh generator with default parameters
    pub fn with_default_params(&self) -> Self {
        Self::default()
    }

    /// Generate mesh for a specified domain
    pub fn generate_mesh(
        &self,
        domain: &Domain,
        boundary_spec: &BoundarySpecification,
    ) -> PDEResult<TriangularMesh> {
        match domain {
            Domain::Rectangle {
                x_min,
                y_min,
                x_max,
                y_max,
            } => self.generate_rectangle_mesh(*x_min, *y_min, *x_max, *y_max, boundary_spec),
            Domain::Circle {
                center_x,
                center_y,
                radius,
            } => self.generate_circle_mesh(*center_x, *center_y, *radius, boundary_spec),
            Domain::Ellipse {
                center_x,
                center_y,
                a,
                b,
                rotation,
            } => self.generate_ellipse_mesh(*center_x, *center_y, *a, *b, *rotation, boundary_spec),
            Domain::LShape {
                width,
                height,
                notch_width,
                notch_height,
            } => self.generate_l_shape_mesh(
                *width,
                *height,
                *notch_width,
                *notch_height,
                boundary_spec,
            ),
            Domain::Polygon { vertices } => self.generate_polygon_mesh(vertices, boundary_spec),
            Domain::Annulus {
                center_x,
                center_y,
                inner_radius,
                outer_radius,
            } => self.generate_annulus_mesh(
                *center_x,
                *center_y,
                *inner_radius,
                *outer_radius,
                boundary_spec,
            ),
        }
    }

    /// Generate rectangular mesh using structured approach
    fn generate_rectangle_mesh(
        &self,
        x_min: f64,
        y_min: f64,
        x_max: f64,
        y_max: f64,
        boundary_spec: &BoundarySpecification,
    ) -> PDEResult<TriangularMesh> {
        let width = x_max - x_min;
        let height = y_max - y_min;

        // Calculate number of divisions
        let nx = ((width / self.params.element_size).ceil() as usize)._max(2);
        let ny = ((height / self.params.element_size).ceil() as usize)._max(2);

        let dx = width / (nx - 1) as f64;
        let dy = height / (ny - 1) as f64;

        // Generate points
        let mut points = Vec::new();
        for j in 0..ny {
            for i in 0..nx {
                let x = x_min + i as f64 * dx;
                let y = y_min + j as f64 * dy;
                points.push(Point::new(x, y));
            }
        }

        // Generate triangles (two triangles per rectangular cell)
        let mut triangles = Vec::new();
        for j in 0..ny - 1 {
            for i in 0..nx - 1 {
                let idx = |row: usize, col: usize| row * nx + col;

                let p0 = idx(j, i);
                let p1 = idx(j, i + 1);
                let p2 = idx(j + 1, i);
                let p3 = idx(j + 1, i + 1);

                // First triangle
                triangles.push(Triangle::new([p0, p1, p2], Some(1)));
                // Second triangle
                triangles.push(Triangle::new([p1, p3, p2], Some(1)));
            }
        }

        let mut mesh = TriangularMesh::new();
        mesh.points = points;
        mesh.elements = triangles;
        self.apply_boundary_markers(
            &mut mesh,
            boundary_spec,
            &Domain::Rectangle {
                x_min,
                y_min,
                x_max,
                y_max,
            },
        )?;
        self.improve_mesh_quality(&mut mesh)?;

        Ok(mesh)
    }

    /// Generate circular mesh using radial structured approach
    fn generate_circle_mesh(
        &self,
        center_x: f64,
        center_y: f64,
        radius: f64,
        boundary_spec: &BoundarySpecification,
    ) -> PDEResult<TriangularMesh> {
        // Estimate number of radial and angular divisions
        let circumference = 2.0 * PI * radius;
        let n_theta = ((circumference / self.params.element_size).ceil() as usize).max(8);
        let n_r = ((radius / self.params.element_size).ceil() as usize).max(2);

        let mut points = Vec::new();
        let mut triangles = Vec::new();

        // Add center point
        points.push(Point::new(center_x, center_y));

        // Generate points in radial layers
        for i in 1..=n_r {
            let r = radius * i as f64 / n_r as f64;
            for j in 0..n_theta {
                let theta = 2.0 * PI * j as f64 / n_theta as f64;
                let _x = center_x + r * theta.cos();
                let _y = center_y + r * theta.sin();
                points.push(Point::new(_x_y));
            }
        }

        // Generate triangles
        // Connect center to first ring
        for j in 0..n_theta {
            let p1 = 1 + j;
            let p2 = 1 + (j + 1) % n_theta;
            triangles.push(Triangle::new([0, p1, p2], Some(1)));
        }

        // Connect rings
        for i in 0..n_r - 1 {
            let ring1_start = 1 + i * n_theta;
            let ring2_start = 1 + (i + 1) * n_theta;

            for j in 0..n_theta {
                let p1 = ring1_start + j;
                let p2 = ring1_start + (j + 1) % n_theta;
                let p3 = ring2_start + j;
                let p4 = ring2_start + (j + 1) % n_theta;

                triangles.push(Triangle::new([p1, p2, p3], Some(1)));
                triangles.push(Triangle::new([p2, p4, p3], Some(1)));
            }
        }

        let mut mesh = TriangularMesh::new();
        mesh.points = points;
        mesh.elements = triangles;
        self.apply_boundary_markers(
            &mut mesh,
            boundary_spec,
            &Domain::Circle {
                center_x,
                center_y,
                radius,
            },
        )?;
        self.improve_mesh_quality(&mut mesh)?;

        Ok(mesh)
    }

    /// Generate ellipse mesh using transformation from circle
    fn generate_ellipse_mesh(
        &self,
        center_x: f64,
        center_y: f64,
        a: f64,
        b: f64,
        rotation: f64,
        boundary_spec: &BoundarySpecification,
    ) -> PDEResult<TriangularMesh> {
        // Generate circle mesh with radius = max(a, b)
        let max_radius = a.max(b);
        let mut mesh =
            self.generate_circle_mesh(0.0, 0.0, max_radius, &BoundarySpecification::default())?;

        // Transform points to ellipse
        let cos_rot = rotation.cos();
        let sin_rot = rotation.sin();

        for point in &mut mesh.points {
            // Scale to ellipse
            point._x *= a / max_radius;
            point._y *= b / max_radius;

            // Rotate
            let x_rot = point._x * cos_rot - point._y * sin_rot;
            let y_rot = point._x * sin_rot + point._y * cos_rot;

            // Translate
            point._x = center_x + x_rot;
            point._y = center_y + y_rot;
        }

        self.apply_boundary_markers(
            &mut mesh,
            boundary_spec,
            &Domain::Ellipse {
                center_x,
                center_y,
                a,
                b,
                rotation,
            },
        )?;
        self.improve_mesh_quality(&mut mesh)?;

        Ok(mesh)
    }

    /// Generate L-shaped domain mesh
    fn generate_l_shape_mesh(
        &self,
        width: f64,
        height: f64,
        notch_width: f64,
        notch_height: f64,
        boundary_spec: &BoundarySpecification,
    ) -> PDEResult<TriangularMesh> {
        // Create L-shape as combination of two rectangles
        let mesh1 = self.generate_rectangle_mesh(
            0.0,
            0.0,
            _width,
            _height - notch_height,
            &BoundarySpecification::default(),
        )?;
        let mesh2 = self.generate_rectangle_mesh(
            0.0,
            _height - notch_height,
            _width - notch_width,
            _height,
            &BoundarySpecification::default(),
        )?;

        // Combine meshes
        let combined_mesh = self.combine_meshes(&[mesh1, mesh2])?;
        let mut mesh = combined_mesh;

        self.apply_boundary_markers(
            &mut mesh,
            boundary_spec,
            &Domain::LShape {
                _width,
                _height,
                notch_width,
                notch_height,
            },
        )?;
        self.improve_mesh_quality(&mut mesh)?;

        Ok(mesh)
    }

    /// Generate mesh for arbitrary polygon using Delaunay triangulation
    fn generate_polygon_mesh(
        &self,
        vertices: &[Point],
        boundary_spec: &BoundarySpecification,
    ) -> PDEResult<TriangularMesh> {
        if vertices.len() < 3 {
            return Err(PDEError::FiniteElementError(
                "Polygon must have at least 3 vertices".to_string(),
            ));
        }

        // Simple implementation: triangulate using fan triangulation from first vertex
        // In practice, would use more sophisticated algorithms like Delaunay triangulation
        let mut points = vertices.to_vec();
        let mut triangles = Vec::new();

        // Add interior points if needed for refinement
        let (min_x, max_x) = vertices
            .iter()
            .map(|p| p.x)
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), x| {
                (min.min(x), max.max(x))
            });
        let (min_y, max_y) = vertices
            .iter()
            .map(|p| p.y)
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), y| {
                (min.min(y), max.max(y))
            });

        // Generate interior points in a grid
        let nx = ((max_x - min_x) / self.params.element_size) as usize;
        let ny = ((max_y - min_y) / self.params.element_size) as usize;

        for i in 1..nx {
            for j in 1..ny {
                let x = min_x + (max_x - min_x) * i as f64 / nx as f64;
                let y = min_y + (max_y - min_y) * j as f64 / ny as f64;
                let point = Point::new(x, y);

                // Check if point is inside polygon
                if self.point_in_polygon(&point, vertices) {
                    points.push(point);
                }
            }
        }

        // Simple triangulation (would be replaced with proper Delaunay triangulation)
        for i in 1..vertices.len() - 1 {
            triangles.push(Triangle::new([0, i, i + 1], Some(1)));
        }

        let mut mesh = TriangularMesh::new();
        mesh.points = points;
        mesh.elements = triangles;
        self.apply_boundary_markers(
            &mut mesh,
            boundary_spec,
            &Domain::Polygon {
                vertices: vertices.to_vec(),
            },
        )?;

        Ok(mesh)
    }

    /// Generate annulus (ring) mesh
    fn generate_annulus_mesh(
        &self,
        center_x: f64,
        center_y: f64,
        inner_radius: f64,
        outer_radius: f64,
        boundary_spec: &BoundarySpecification,
    ) -> PDEResult<TriangularMesh> {
        if inner_radius >= outer_radius {
            return Err(PDEError::FiniteElementError(
                "Inner _radius must be less than outer _radius".to_string(),
            ));
        }

        // Generate structured radial mesh
        let n_theta = ((2.0 * PI * outer_radius / self.params.element_size).ceil() as usize).max(8);
        let n_r =
            (((outer_radius - inner_radius) / self.params.element_size).ceil() as usize).max(2);

        let mut points = Vec::new();
        let mut triangles = Vec::new();

        // Generate points in radial layers
        for i in 0..=n_r {
            let r = inner_radius + (outer_radius - inner_radius) * i as f64 / n_r as f64;
            for j in 0..n_theta {
                let theta = 2.0 * PI * j as f64 / n_theta as f64;
                let _x = center_x + r * theta.cos();
                let _y = center_y + r * theta.sin();
                points.push(Point::new(_x_y));
            }
        }

        // Connect rings with triangles
        for i in 0..n_r {
            let ring1_start = i * n_theta;
            let ring2_start = (i + 1) * n_theta;

            for j in 0..n_theta {
                let p1 = ring1_start + j;
                let p2 = ring1_start + (j + 1) % n_theta;
                let p3 = ring2_start + j;
                let p4 = ring2_start + (j + 1) % n_theta;

                triangles.push(Triangle::new([p1, p2, p3], Some(1)));
                triangles.push(Triangle::new([p2, p4, p3], Some(1)));
            }
        }

        let mut mesh = TriangularMesh::new();
        mesh.points = points;
        mesh.elements = triangles;
        self.apply_boundary_markers(
            &mut mesh,
            boundary_spec,
            &Domain::Annulus {
                center_x,
                center_y,
                inner_radius,
                outer_radius,
            },
        )?;
        self.improve_mesh_quality(&mut mesh)?;

        Ok(mesh)
    }

    /// Apply boundary markers to mesh based on domain and specification
    fn apply_boundary_markers(
        &self,
        mesh: &mut TriangularMesh,
        boundary_spec: &BoundarySpecification, _domain: &Domain,
    ) -> PDEResult<()> {
        // This is a simplified implementation
        // In practice, would need sophisticated boundary detection

        // Apply default boundary markers if none specified
        if boundary_spec.boundary_markers.is_empty() {
            // Mark all boundary edges with marker 1
            let boundary_edges = self.find_boundary_edges(mesh);
            for (_p1_p2) in boundary_edges {
                // Mark boundary points
                // In a complete implementation, would track edge markers
            }
        }

        Ok(())
    }

    /// Find boundary edges in the mesh
    fn find_boundary_edges(_mesh: &TriangularMesh) -> Vec<(usize, usize)> {
        let mut edge_count = HashMap::new();

        // Count how many times each edge appears
        for element in &_mesh.elements {
            let [p1, p2, p3] = element.nodes;
            let edges = [
                (p1.min(p2), p1.max(p2)),
                (p2.min(p3), p2.max(p3)),
                (p3.min(p1), p3.max(p1)),
            ];

            for edge in &edges {
                *edge_count.entry(*edge).or_insert(0) += 1;
            }
        }

        // Boundary edges appear only once
        edge_count
            .into_iter()
            .filter(|(_, count)| *count == 1)
            .map(|(edge_)| edge)
            .collect()
    }

    /// Improve mesh quality through smoothing and refinement
    fn improve_mesh_quality(_mesh: &mut TriangularMesh) -> PDEResult<()> {
        for _ in 0..self.params.quality_iterations {
            // Laplacian smoothing
            self.laplacian_smoothing(_mesh)?;

            // Quality-based refinement
            self.quality_refinement(_mesh)?;
        }

        Ok(())
    }

    /// Apply Laplacian smoothing to improve mesh quality
    fn laplacian_smoothing(_mesh: &mut TriangularMesh) -> PDEResult<()> {
        let n_points = _mesh.points.len();
        let mut new_positions = vec![Point::new(0.0, 0.0); n_points];
        let mut neighbor_counts = vec![0; n_points];

        // Find neighbors for each point
        for element in &_mesh.elements {
            let [p1, p2, p3] = element.nodes;

            new_positions[p1].x += _mesh.points[p2].x + _mesh.points[p3].x;
            new_positions[p1].y += _mesh.points[p2].y + _mesh.points[p3].y;
            neighbor_counts[p1] += 2;

            new_positions[p2].x += _mesh.points[p1].x + _mesh.points[p3].x;
            new_positions[p2].y += _mesh.points[p1].y + _mesh.points[p3].y;
            neighbor_counts[p2] += 2;

            new_positions[p3].x += _mesh.points[p1].x + _mesh.points[p2].x;
            new_positions[p3].y += _mesh.points[p1].y + _mesh.points[p2].y;
            neighbor_counts[p3] += 2;
        }

        // Update positions (keep boundary points fixed)
        let boundary_edges = self.find_boundary_edges(_mesh);
        let boundary_points: HashSet<usize> = boundary_edges
            .iter()
            .flat_map(|(p1, p2)| vec![*p1, *p2])
            .collect();

        for i in 0..n_points {
            if !boundary_points.contains(&i) && neighbor_counts[i] > 0 {
                _mesh.points[i].x = new_positions[i].x / neighbor_counts[i] as f64;
                _mesh.points[i].y = new_positions[i].y / neighbor_counts[i] as f64;
            }
        }

        Ok(())
    }

    /// Refine elements with poor quality
    fn quality_refinement(_mesh: &mut TriangularMesh) -> PDEResult<()> {
        let mut elements_to_refine = Vec::new();

        // Identify poor quality elements
        for (i, element) in _mesh.elements.iter().enumerate() {
            let quality = self.element_quality(_mesh, element);
            if quality.min_angle < self.params.min_angle
                || quality.max_angle > self.params.max_angle
            {
                elements_to_refine.push(i);
            }
        }

        // For simplicity, we'll skip actual refinement here
        // In practice, would implement edge splitting and local refinement

        Ok(())
    }

    /// Calculate quality metrics for a single element
    fn element_quality(_mesh: &TriangularMesh, element: &Triangle) -> ElementQuality {
        let [p1, p2, p3] = element.nodes;
        let a = &_mesh.points[p1];
        let b = &_mesh.points[p2];
        let c = &_mesh.points[p3];

        // Calculate edge lengths
        let ab = ((b.x - a.x).powi(2) + (b.y - a.y).powi(2)).sqrt();
        let bc = ((c.x - b.x).powi(2) + (c.y - b.y).powi(2)).sqrt();
        let ca = ((a.x - c.x).powi(2) + (a.y - c.y).powi(2)).sqrt();

        // Calculate angles using law of cosines
        let angle_a =
            ((bc.powi(2) + ca.powi(2) - ab.powi(2)) / (2.0 * bc * ca)).acos() * 180.0 / PI;
        let angle_b =
            ((ca.powi(2) + ab.powi(2) - bc.powi(2)) / (2.0 * ca * ab)).acos() * 180.0 / PI;
        let angle_c = 180.0 - angle_a - angle_b;

        let min_angle = angle_a.min(angle_b).min(angle_c);
        let max_angle = angle_a.max(angle_b).max(angle_c);

        // Calculate area using cross product
        let area = 0.5 * ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)).abs();

        // Aspect ratio (radius ratio)
        let s = (ab + bc + ca) / 2.0;
        let inradius = area / s;
        let circumradius = (ab * bc * ca) / (4.0 * area);
        let aspect_ratio = circumradius / inradius;

        ElementQuality {
            min_angle,
            max_angle,
            area,
            aspect_ratio,
        }
    }

    /// Check if a point is inside a polygon using ray casting
    fn point_in_polygon(_point: &Point, polygon: &[Point]) -> bool {
        let mut inside = false;
        let mut j = polygon.len() - 1;

        for i in 0..polygon.len() {
            if ((polygon[i].y > _point.y) != (polygon[j].y > _point.y))
                && (_point.x
                    < (polygon[j].x - polygon[i].x) * (_point.y - polygon[i].y)
                        / (polygon[j].y - polygon[i].y)
                        + polygon[i].x)
            {
                inside = !inside;
            }
            j = i;
        }

        inside
    }

    /// Combine multiple meshes into one
    fn combine_meshes(_meshes: &[TriangularMesh]) -> PDEResult<TriangularMesh> {
        if _meshes.is_empty() {
            return Err(PDEError::FiniteElementError(
                "Cannot combine empty mesh list".to_string(),
            ));
        }

        let mut combined_points = Vec::new();
        let mut combined_elements = Vec::new();
        let mut point_offset = 0;

        for mesh in _meshes {
            // Add points
            combined_points.extend(mesh.points.iter().cloned());

            // Add elements with updated indices
            for element in &mesh.elements {
                let [p1, p2, p3] = element.nodes;
                combined_elements.push(Triangle::new(
                    [p1 + point_offset, p2 + point_offset, p3 + point_offset],
                    element.marker,
                ));
            }

            point_offset += mesh.points.len();
        }

        let mut combined_mesh = TriangularMesh::new();
        combined_mesh.points = combined_points;
        combined_mesh.elements = combined_elements;
        Ok(combined_mesh)
    }

    /// Calculate overall mesh quality metrics
    pub fn assess_mesh_quality(_mesh: &TriangularMesh) -> MeshQuality {
        let mut min_angle = f64::INFINITY;
        let mut max_angle: f64 = 0.0;
        let mut total_area = 0.0;
        let mut min_aspect_ratio = f64::INFINITY;
        let mut poor_quality_count = 0;

        for element in &_mesh.elements {
            let quality = self.element_quality(_mesh, element);

            min_angle = min_angle.min(quality.min_angle);
            max_angle = max_angle.max(quality.max_angle);
            total_area += quality.area;
            min_aspect_ratio = min_aspect_ratio.min(quality.aspect_ratio);

            if quality.min_angle < self.params.min_angle
                || quality.max_angle > self.params.max_angle
            {
                poor_quality_count += 1;
            }
        }

        let avg_element_size = (total_area / _mesh.elements.len() as f64).sqrt();
        let quality_score = 1.0 - (poor_quality_count as f64 / _mesh.elements.len() as f64);

        MeshQuality {
            min_angle,
            max_angle,
            avg_element_size,
            min_aspect_ratio,
            poor_quality_elements: poor_quality_count,
            quality_score,
        }
    }
}

/// Quality metrics for individual elements
#[derive(Debug, Clone)]
struct ElementQuality {
    min_angle: f64,
    max_angle: f64,
    area: f64,
    aspect_ratio: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rectangle_mesh_generation() {
        let generator = AutoMeshGenerator::default();
        let domain = Domain::Rectangle {
            x_min: 0.0,
            y_min: 0.0,
            x_max: 1.0,
            y_max: 1.0,
        };
        let boundary_spec = BoundarySpecification::default();

        let mesh = generator.generate_mesh(&domain, &boundary_spec).unwrap();

        assert!(!mesh.points.is_empty());
        assert!(!mesh.elements.is_empty());

        // Check that all points are within domain
        for point in &mesh.points {
            assert!(point.x >= 0.0 && point.x <= 1.0);
            assert!(point.y >= 0.0 && point.y <= 1.0);
        }
    }

    #[test]
    fn test_circle_mesh_generation() {
        let params = MeshGenerationParams {
            element_size: 0.2,
            ..Default::default()
        };
        let generator = AutoMeshGenerator::new(params);

        let domain = Domain::Circle {
            center_x: 0.0,
            center_y: 0.0,
            radius: 1.0,
        };
        let boundary_spec = BoundarySpecification::default();

        let mesh = generator.generate_mesh(&domain, &boundary_spec).unwrap();

        assert!(!mesh.points.is_empty());
        assert!(!mesh.elements.is_empty());

        // Check that all points are within or on circle boundary
        for point in &mesh.points {
            let distance = (point.x.powi(2) + point.y.powi(2)).sqrt();
            assert!(distance <= 1.01); // Allow small tolerance for numerical errors
        }
    }

    #[test]
    fn test_mesh_quality_assessment() {
        let generator = AutoMeshGenerator::default();
        let domain = Domain::Rectangle {
            x_min: 0.0,
            y_min: 0.0,
            x_max: 1.0,
            y_max: 1.0,
        };
        let boundary_spec = BoundarySpecification::default();

        let mesh = generator.generate_mesh(&domain, &boundary_spec).unwrap();
        let quality = generator.assess_mesh_quality(&mesh);

        assert!(quality.min_angle > 0.0);
        assert!(quality.max_angle < 180.0);
        assert!(quality.avg_element_size > 0.0);
        assert!(quality.quality_score >= 0.0 && quality.quality_score <= 1.0);
    }

    #[test]
    fn test_point_in_polygon() {
        let generator = AutoMeshGenerator::default();

        // Square polygon
        let polygon = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(0.0, 1.0),
        ];

        assert!(generator.point_in_polygon(&Point::new(0.5, 0.5), &polygon));
        assert!(!generator.point_in_polygon(&Point::new(1.5, 0.5), &polygon));
        assert!(!generator.point_in_polygon(&Point::new(-0.5, 0.5), &polygon));
    }

    #[test]
    fn test_annulus_mesh_generation() {
        let generator = AutoMeshGenerator::default();
        let domain = Domain::Annulus {
            center_x: 0.0,
            center_y: 0.0,
            inner_radius: 0.5,
            outer_radius: 1.0,
        };
        let boundary_spec = BoundarySpecification::default();

        let mesh = generator.generate_mesh(&domain, &boundary_spec).unwrap();

        assert!(!mesh.points.is_empty());
        assert!(!mesh.elements.is_empty());

        // Check that all points are within annulus
        for point in &mesh.points {
            let distance = (point.x.powi(2) + point.y.powi(2)).sqrt();
            assert!((0.49..=1.01).contains(&distance)); // Allow tolerance
        }
    }

    #[test]
    fn test_l_shape_mesh_generation() {
        let generator = AutoMeshGenerator::default();
        let domain = Domain::LShape {
            width: 2.0,
            height: 2.0,
            notch_width: 1.0,
            notch_height: 1.0,
        };
        let boundary_spec = BoundarySpecification::default();

        let mesh = generator.generate_mesh(&domain, &boundary_spec).unwrap();

        assert!(!mesh.points.is_empty());
        assert!(!mesh.elements.is_empty());
    }

    #[test]
    fn test_custom_mesh_parameters() {
        let params = MeshGenerationParams {
            element_size: 0.05,
            min_angle: 25.0,
            max_angle: 135.0,
            quality_iterations: 10,
            element_type: ElementType::Linear,
            boundary_refinement_iterations: 5,
        };

        let generator = AutoMeshGenerator::new(params);
        let domain = Domain::Rectangle {
            x_min: 0.0,
            y_min: 0.0,
            x_max: 1.0,
            y_max: 1.0,
        };
        let boundary_spec = BoundarySpecification::default();

        let mesh = generator.generate_mesh(&domain, &boundary_spec).unwrap();
        let quality = generator.assess_mesh_quality(&mesh);

        // With smaller element size, should have more elements
        assert!(mesh.elements.len() > 10);
        assert!(quality.avg_element_size < 0.2);
    }
}
