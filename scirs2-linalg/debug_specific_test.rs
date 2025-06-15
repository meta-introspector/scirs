use ndarray::array;
use scirs2_linalg::compat;

fn main() {
    // Test the exact matrix from the failing test
    let symmetric_matrix = array![[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]];
    
    println!("Test matrix from failing test:\n{:?}\n", symmetric_matrix);
    
    // Test eigenvalues only
    match compat::eigh(
        &symmetric_matrix.view(),
        None,
        false,
        true, // eigenvalues only
        false,
        false,
        true,
        None,
        None,
        None,
        1,
    ) {
        Ok((eigenvals, eigenvecs_opt)) => {
            println!("Eigenvalues only: {:?}", eigenvals);
            assert!(eigenvecs_opt.is_none());
            
            // Check if eigenvalues are sorted
            for i in 1..eigenvals.len() {
                println!("λ[{}] = {:.6}, λ[{}] = {:.6}, sorted: {}", 
                    i-1, eigenvals[i-1], i, eigenvals[i], eigenvals[i-1] <= eigenvals[i]);
                assert!(eigenvals[i-1] <= eigenvals[i], "Eigenvalues not sorted in ascending order");
            }
            println!("✓ Eigenvalues are correctly sorted in ascending order");
        }
        Err(e) => {
            println!("Error with eigenvalues only: {:?}", e);
            return;
        }
    }
    
    // Test eigenvalues + eigenvectors
    match compat::eigh(
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
    ) {
        Ok((eigenvals_with_vecs, eigenvecs_with_opt)) => {
            println!("\nEigenvalues with vectors: {:?}", eigenvals_with_vecs);
            
            // Check if eigenvalues are sorted
            for i in 1..eigenvals_with_vecs.len() {
                assert!(eigenvals_with_vecs[i-1] <= eigenvals_with_vecs[i], "Eigenvalues not sorted in ascending order");
            }
            println!("✓ Eigenvalues with vectors are correctly sorted in ascending order");
            
            if let Some(eigenvecs) = eigenvecs_with_opt {
                println!("Eigenvectors:\n{:?}\n", eigenvecs);
                
                // Test A*V = V*Λ for each eigenvector
                for i in 0..eigenvals_with_vecs.len() {
                    let v = eigenvecs.column(i);
                    let av = symmetric_matrix.dot(&v);
                    let lv = &v * eigenvals_with_vecs[i];
                    
                    println!("Eigenvector {}: {:?}", i, v);
                    println!("A*v = {:?}", av);
                    println!("λ*v = {:?}", lv);
                    
                    let max_diff = av.iter().zip(lv.iter()).map(|(a, l)| (a - l).abs()).fold(0.0, f64::max);
                    println!("Max difference: {:.2e}", max_diff);
                    assert!(max_diff < 1e-10, "A*v != λ*v for eigenvector {}", i);
                }
                println!("✓ All eigenvectors satisfy A*v = λ*v");
                
                // Check orthogonality
                let vtv = eigenvecs.t().dot(&eigenvecs);
                let identity = ndarray::Array2::eye(eigenvecs.ncols());
                let ortho_diff = (&vtv - &identity).iter().map(|x| x.abs()).fold(0.0, f64::max);
                println!("Orthogonality check, max diff from identity: {:.2e}", ortho_diff);
                assert!(ortho_diff < 1e-10, "Eigenvectors are not orthogonal");
                println!("✓ Eigenvectors are orthogonal");
                
                // Test full A*V = V*Λ
                let av = symmetric_matrix.dot(&eigenvecs);
                let vl = eigenvecs.dot(&ndarray::Array2::from_diag(&eigenvals_with_vecs));
                let full_diff = (&av - &vl).iter().map(|x| x.abs()).fold(0.0, f64::max);
                println!("Full A*V = V*Λ check, max diff: {:.2e}", full_diff);
                assert!(full_diff < 1e-10, "A*V != V*Λ");
                println!("✓ Full A*V = V*Λ relationship satisfied");
            }
        }
        Err(e) => {
            println!("Error with eigenvalues + eigenvectors: {:?}", e);
        }
    }
    
    println!("\n✅ All tests passed!");
}