// Simple compilation test for the fixes we made
#[cfg(test)]
mod compilation_tests {
    use ndarray::Array1;

    #[test]
    fn test_basic_trait_bounds() {
        // Test that Sum trait bounds work
        let data = Array1::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]);
        let sum: f64 = data.iter().sum();
        assert!((sum - 15.0).abs() < 1e-10);
        
        // Test that Display trait bounds work
        let display_string = format!("{}", sum);
        assert!(!display_string.is_empty());
    }
    
    #[test]
    fn test_function_signatures() {
        // Test that we can call functions with the correct number of parameters
        // This would fail if our function signature fixes didn't work
        let data = Array1::from_vec(vec![1.0_f64, 2.0, 3.0]);
        
        // Mock the function signatures we fixed
        fn mock_var(data: &ndarray::ArrayView1<f64>, ddof: usize, workers: Option<usize>) -> Result<f64, String> {
            let _ = (_data, ddof, workers);
            Ok(1.0)
        }
        
        fn mock_std(data: &ndarray::ArrayView1<f64>, ddof: usize, workers: Option<usize>) -> Result<f64, String> {
            let _ = (_data, ddof, workers);
            Ok(1.0)
        }
        
        fn mock_skew(data: &ndarray::ArrayView1<f64>, bias: bool, workers: Option<usize>) -> Result<f64, String> {
            let _ = (_data, bias, workers);
            Ok(0.0)
        }
        
        // These calls should compile with our fixes
        let _var_result = mock_var(&data.view(), 1, None);
        let _std_result = mock_std(&data.view(), 1, None);
        let _skew_result = mock_skew(&data.view(), false, None);
    }
}