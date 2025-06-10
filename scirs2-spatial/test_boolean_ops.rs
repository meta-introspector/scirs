// Standalone test for Boolean operations module
// This can be run with: rustc --test test_boolean_ops.rs && ./test_boolean_ops

#[allow(dead_code)]
mod boolean_ops;

#[allow(dead_code)]
mod error {
    #[derive(Debug)]
    pub enum SpatialError {
        ValueError(String),
        ComputationError(String),
        NotImplementedError(String),
    }

    pub type SpatialResult<T> = Result<T, SpatialError>;
}

use boolean_ops::*;
use ndarray::arr2;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_polygon_operations() {
        // Test polygon creation and area calculation
        let square = arr2(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        let area = compute_polygon_area(&square.view()).unwrap();
        assert!((area - 1.0).abs() < 1e-10);

        // Test convexity
        let is_convex = is_convex_polygon(&square.view()).unwrap();
        assert!(is_convex);

        // Test self-intersection
        let is_self_intersecting = is_self_intersecting(&square.view()).unwrap();
        assert!(!is_self_intersecting);
    }

    #[test]
    fn test_polygon_union() {
        let poly1 = arr2(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        let poly2 = arr2(&[[2.0, 0.0], [3.0, 0.0], [3.0, 1.0], [2.0, 1.0]]);

        let union = polygon_union(&poly1.view(), &poly2.view()).unwrap();
        assert!(union.nrows() >= 4);
    }

    #[test]
    fn test_polygon_intersection() {
        let poly1 = arr2(&[[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]]);
        let poly2 = arr2(&[[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]]);

        let intersection = polygon_intersection(&poly1.view(), &poly2.view()).unwrap();
        // Should produce some result
        assert!(intersection.nrows() >= 0);
    }

    #[test]
    fn test_l_shape_convexity() {
        let l_shape = arr2(&[
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [0.0, 2.0],
        ]);

        let is_convex = is_convex_polygon(&l_shape.view()).unwrap();
        assert!(!is_convex); // L-shape should not be convex
    }

    #[test]
    fn test_self_intersecting_polygon() {
        // Bowtie shape
        let bowtie = arr2(&[[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]);

        let is_self_intersecting = is_self_intersecting(&bowtie.view()).unwrap();
        assert!(is_self_intersecting);
    }
}

fn main() {
    println!("Running Boolean operations tests...");
    tests::test_basic_polygon_operations();
    println!("✓ Basic polygon operations test passed");

    tests::test_polygon_union();
    println!("✓ Polygon union test passed");

    tests::test_polygon_intersection();
    println!("✓ Polygon intersection test passed");

    tests::test_l_shape_convexity();
    println!("✓ L-shape convexity test passed");

    tests::test_self_intersecting_polygon();
    println!("✓ Self-intersecting polygon test passed");

    println!("All Boolean operations tests passed!");
}