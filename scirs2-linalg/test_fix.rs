#[cfg(test)]
mod test_eigh_fix {
    use ndarray::array;
    use scirs2_linalg::compat;

    #[test]
    fn test_specific_matrix_from_failing_test() {
        // Test the exact matrix from the failing test
        let symmetric_matrix = array![[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]];
        
        // Test eigenvalues + eigenvectors
        let (eigenvals_with_vecs, eigenvecs_with_opt) = compat::eigh(
            &symmetric_matrix.view(),
            None,
            false,
            false, // Get eigenvectors too
            false,
            false,
            true,
            None,
            None,
            None,
            1,
        ).unwrap();
        
        // Check if eigenvalues are sorted
        for i in 1..eigenvals_with_vecs.len() {
            assert!(eigenvals_with_vecs[i-1] <= eigenvals_with_vecs[i], 
                "Eigenvalues not sorted: {} > {}", eigenvals_with_vecs[i-1], eigenvals_with_vecs[i]);
        }
        
        if let Some(eigenvecs) = eigenvecs_with_opt {
            // Test A*V = V*Λ
            let av = symmetric_matrix.dot(&eigenvecs);
            let vl = eigenvecs.dot(&array2::from_diag(&eigenvals_with_vecs));
            let max_diff = (&av - &vl).iter().map(|x| x.abs()).fold(0.0, f64::max);
            assert!(max_diff < 1e-10, "A*V != V*Λ, max diff: {:.2e}", max_diff);
        }
    }
}

fn main() {
    println!("Run with: cargo test test_eigh_fix");
}