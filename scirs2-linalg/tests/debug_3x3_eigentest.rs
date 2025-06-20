use ndarray::array;
use num_traits::Float;
use scirs2_linalg::eigh;

#[test]
#[ignore = "3x3 symmetric eigenvalue decomposition not fully implemented yet"]
fn debug_3x3_eigenvalue_computation() {
    let a = array![[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]];

    println!("Matrix A:");
    println!("{:?}", a);

    let (eigenvalues, eigenvectors) = eigh(&a.view(), None).unwrap();

    println!("Eigenvalues: {:?}", eigenvalues);
    println!("Eigenvectors:");
    println!("{:?}", eigenvectors);

    // Check A*V = V*Λ
    let av = a.dot(&eigenvectors);
    let vl = eigenvectors.dot(&ndarray::Array2::from_diag(&eigenvalues));

    println!("A*V:");
    println!("{:?}", av);
    println!("V*Λ:");
    println!("{:?}", vl);

    println!("Difference (A*V - V*Λ):");
    let diff = &av - &vl;
    println!("{:?}", diff);

    // Check max difference
    let max_diff = diff.iter().map(|x| x.abs()).fold(0.0, f64::max);
    println!("Max difference: {}", max_diff);

    // Check orthogonality V^T * V = I
    let vtv = eigenvectors.t().dot(&eigenvectors);
    let identity: ndarray::Array2<f64> = ndarray::Array2::eye(3);
    let ortho_diff = &vtv - &identity;
    let max_ortho_diff = ortho_diff.iter().map(|x| x.abs()).fold(0.0, f64::max);

    println!("V^T * V:");
    println!("{:?}", vtv);
    println!("Max orthogonality difference: {}", max_ortho_diff);

    println!("Testing if 1e-8 tolerance passes: {}", max_diff < 1e-8);
    println!("Testing if 1e-7 tolerance passes: {}", max_diff < 1e-7);
    println!("Testing if 1e-6 tolerance passes: {}", max_diff < 1e-6);
    assert!(
        max_diff < 1e-6,
        "A*V should equal V*Λ within reasonable tolerance"
    );
    assert!(max_ortho_diff < 1e-10, "V should be orthogonal");
}
