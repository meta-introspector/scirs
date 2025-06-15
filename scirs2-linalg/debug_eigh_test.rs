use ndarray::array;
use scirs2_linalg::eigen::eigh;

fn main() {
    let symmetric_matrix = array![[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]];
    
    println!("Test matrix:\n{:?}\n", symmetric_matrix);
    
    match eigh(&symmetric_matrix.view()) {
        Ok((eigenvals, eigenvecs)) => {
            println!("Eigenvalues: {:?}", eigenvals);
            println!("Eigenvectors:\n{:?}\n", eigenvecs);
            
            // Check ordering
            for i in 1..eigenvals.len() {
                println!("λ[{}] = {:.6}, λ[{}] = {:.6}, sorted: {}", 
                    i-1, eigenvals[i-1], i, eigenvals[i], eigenvals[i-1] <= eigenvals[i]);
            }
            
            // Test A*V = V*Λ for each eigenvector
            for i in 0..eigenvals.len() {
                let v = eigenvecs.column(i);
                let av = symmetric_matrix.dot(&v);
                let lv = &v * eigenvals[i];
                
                println!("\nEigenvector {}: {:?}", i, v);
                println!("A*v = {:?}", av);
                println!("λ*v = {:?}", lv);
                
                let max_diff = av.iter().zip(lv.iter()).map(|(a, l)| (a - l).abs()).fold(0.0, f64::max);
                println!("Max difference: {:.2e}", max_diff);
            }
        }
        Err(e) => {
            println!("Error: {:?}", e);
        }
    }
}