#[cfg(test)]
mod tests {
    use crate::simd_ops::SimdPatternMatcher;

    #[test]
    fn debug_fuzzy_search() {
        let text = "hello world, helo rust, hallo there";
        let pattern = "hello";
        println!("Text: '{}'", text);
        println!("Pattern: '{}'", pattern);
        println!("Pattern length: {}", pattern.len());
        
        // Test different windows manually
        for start in 0..=5 {  // Check first few positions
            for len in 4..=7 {  // Check different window sizes
                if start + len <= text.len() {
                    let window = &text[start..start + len];
                    let distance = SimdPatternMatcher::edit_distance_simd(window, pattern);
                    println!("Position {}, len {}: '{}' -> distance {}", start, len, window, distance);
                }
            }
        }
        
        let matches = SimdPatternMatcher::fuzzy_search_vectorized(text, pattern, 1);
        println!("Matches found: {:?}", matches);
    }
}
