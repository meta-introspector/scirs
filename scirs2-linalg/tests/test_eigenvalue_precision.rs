use ndarray::array;
use scirs2_linalg::eigh;

#[test]
fn test_3x3_eigenvalue_precision() {
    let symmetric_matrix = array![[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]];

    let (eigenvals, eigenvecs) = eigh(&symmetric_matrix.view(), None).unwrap();

    // Check A * V = V * Λ with high precision
    let av = symmetric_matrix.dot(&eigenvecs);
    let vl = eigenvecs.dot(&ndarray::Array2::from_diag(&eigenvals));

    let mut max_error = 0.0f64;
    for i in 0..3 {
        for j in 0..3 {
            let error = (av[[i, j]] - vl[[i, j]]).abs();
            if error > max_error {
                max_error = error;
            }
        }
    }

    println!("Maximum error in A*V = V*Λ: {:.2e}", max_error);
    println!("Eigenvalues: {:?}", eigenvals);

    // Check orthogonality
    let vtv = eigenvecs.t().dot(&eigenvecs);
    let identity = ndarray::Array2::eye(3);

    let mut max_ortho_error = 0.0f64;
    for i in 0..3 {
        for j in 0..3 {
            let error = (vtv[[i, j]] - identity[[i, j]]).abs();
            if error > max_ortho_error {
                max_ortho_error = error;
            }
        }
    }

    println!("Maximum orthogonality error: {:.2e}", max_ortho_error);

    // The goal is to achieve 1e-10 precision
    assert!(
        max_error < 1e-10,
        "A*V = V*Λ error {:.2e} exceeds 1e-10 tolerance",
        max_error
    );
    assert!(
        max_ortho_error < 1e-10,
        "Orthogonality error {:.2e} exceeds 1e-10 tolerance",
        max_ortho_error
    );
}
