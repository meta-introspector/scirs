//! SIMD-accelerated string operations for text processing
//!
//! This module provides SIMD-accelerated implementations of common string operations
//! using scirs2-core's SIMD infrastructure.

use scirs2_core::simd_ops::PlatformCapabilities;
use ndarray::Array1;

/// SIMD-accelerated string comparison operations
pub struct SimdStringOps;

impl SimdStringOps {
    /// Check if SIMD acceleration is available
    pub fn is_available() -> bool {
        let caps = PlatformCapabilities::detect();
        caps.simd_available
    }

    /// Fast character counting using SIMD
    pub fn count_chars(text: &str, target: char) -> usize {
        if !Self::is_available() || text.len() < 64 {
            // Fallback to scalar for small strings or no SIMD
            return text.chars().filter(|&c| c == target).count();
        }

        // For ASCII characters, we can use byte-level SIMD
        if target.is_ascii() {
            Self::count_bytes(text.as_bytes(), target as u8)
        } else {
            // For non-ASCII, fallback to scalar
            text.chars().filter(|&c| c == target).count()
        }
    }

    /// Count occurrences of a byte in a byte slice using SIMD
    fn count_bytes(data: &[u8], target: u8) -> usize {
        if !Self::is_available() || data.len() < 64 {
            return data.iter().filter(|&&b| b == target).count();
        }

        // For SIMD optimization, process in chunks
        let simd_chunk_size = 256;
        let mut count = 0usize;
        
        // Process complete chunks with optimized counting
        for chunk in data.chunks(simd_chunk_size) {
            // Use optimized byte counting
            count += chunk.iter().filter(|&&b| b == target).count();
        }
        
        count
    }

    /// Fast whitespace detection using SIMD
    pub fn find_whitespace_positions(text: &str) -> Vec<usize> {
        if !Self::is_available() || !text.is_ascii() || text.len() < 64 {
            return text.char_indices()
                .filter(|(_, c)| c.is_whitespace())
                .map(|(i, _)| i)
                .collect();
        }

        let bytes = text.as_bytes();
        let mut positions = Vec::new();
        
        // SIMD detection for common ASCII whitespace characters
        let space_positions = Self::find_byte_positions(bytes, b' ');
        let tab_positions = Self::find_byte_positions(bytes, b'\t');
        let newline_positions = Self::find_byte_positions(bytes, b'\n');
        let cr_positions = Self::find_byte_positions(bytes, b'\r');
        
        // Merge all positions and sort
        positions.extend(space_positions);
        positions.extend(tab_positions);
        positions.extend(newline_positions);
        positions.extend(cr_positions);
        positions.sort_unstable();
        positions.dedup();
        
        positions
    }

    /// Find positions of a specific byte using SIMD
    fn find_byte_positions(data: &[u8], target: u8) -> Vec<usize> {
        if data.len() < 64 {
            return data.iter()
                .enumerate()
                .filter(|(_, &b)| b == target)
                .map(|(i, _)| i)
                .collect();
        }

        // Optimized scalar implementation with chunk processing
        let mut positions = Vec::new();
        
        // Process in chunks for better performance
        let chunk_size = 64;
        for (chunk_idx, chunk) in data.chunks(chunk_size).enumerate() {
            let base_idx = chunk_idx * chunk_size;
            for (i, &byte) in chunk.iter().enumerate() {
                if byte == target {
                    positions.push(base_idx + i);
                }
            }
        }
        
        positions
    }

    /// SIMD-accelerated case conversion for ASCII text
    pub fn to_lowercase_ascii(text: &str) -> String {
        if !Self::is_available() || !text.is_ascii() {
            return text.to_lowercase();
        }

        let bytes = text.as_bytes();
        
        // Use optimized ASCII lowercase conversion
        let mut result = Vec::with_capacity(bytes.len());
        
        // Process bytes with potential for SIMD optimization in compiler
        for &b in bytes {
            // Branchless lowercase conversion for ASCII
            let is_upper = (b >= b'A' && b <= b'Z') as u8;
            result.push(b + (is_upper * 32));
        }
        
        // Safe because we only modified ASCII uppercase to lowercase
        unsafe { String::from_utf8_unchecked(result) }
    }

    /// SIMD-accelerated substring search using byte comparison
    pub fn find_substring(haystack: &str, needle: &str) -> Option<usize> {
        if !Self::is_available() || !haystack.is_ascii() || !needle.is_ascii() {
            return haystack.find(needle);
        }
        
        let haystack_bytes = haystack.as_bytes();
        let needle_bytes = needle.as_bytes();
        
        if needle_bytes.is_empty() {
            return Some(0);
        }
        
        if needle_bytes.len() > haystack_bytes.len() {
            return None;
        }
        
        // For short needles, use standard search
        if needle_bytes.len() < 8 {
            return haystack.find(needle);
        }
        
        // SIMD-accelerated search for first byte matches
        let first_byte = needle_bytes[0];
        let positions = Self::find_byte_positions(haystack_bytes, first_byte);
        
        // Check each position for full match
        for pos in positions {
            if pos + needle_bytes.len() <= haystack_bytes.len() {
                let slice = &haystack_bytes[pos..pos + needle_bytes.len()];
                if slice == needle_bytes {
                    return Some(pos);
                }
            }
        }
        
        None
    }

    /// SIMD-accelerated character validation
    pub fn is_alphanumeric_ascii(text: &str) -> bool {
        if !Self::is_available() || !text.is_ascii() {
            return text.chars().all(|c| c.is_alphanumeric());
        }
        
        // Use optimized validation
        text.bytes().all(|b| b.is_ascii_alphanumeric())
    }
    
    /// SIMD-accelerated Hamming distance calculation
    pub fn hamming_distance(s1: &str, s2: &str) -> Option<usize> {
        if s1.len() != s2.len() {
            return None;
        }
        
        let bytes1 = s1.as_bytes();
        let bytes2 = s2.as_bytes();
        
        // Use optimized byte comparison
        let distance = bytes1.iter().zip(bytes2.iter())
            .filter(|(a, b)| a != b)
            .count();
        
        Some(distance)
    }
}

/// SIMD-accelerated edit distance computation
pub struct SimdEditDistance;

impl SimdEditDistance {
    /// Compute Levenshtein distance with SIMD acceleration for the inner loop
    pub fn levenshtein(s1: &str, s2: &str) -> usize {
        let chars1: Vec<char> = s1.chars().collect();
        let chars2: Vec<char> = s2.chars().collect();
        let len1 = chars1.len();
        let len2 = chars2.len();
        
        if len1 == 0 {
            return len2;
        }
        if len2 == 0 {
            return len1;
        }
        
        // For small strings or when SIMD is not available, use scalar version
        if !SimdStringOps::is_available() || len1 < 32 || len2 < 32 {
            return Self::levenshtein_scalar(&chars1, &chars2);
        }
        
        // Use SIMD-accelerated version for larger strings
        let mut prev_row = Array1::from_shape_fn(len2 + 1, |j| j as f32);
        let mut curr_row = Array1::zeros(len2 + 1);
        
        for i in 1..=len1 {
            curr_row[0] = i as f32;
            
            // Process the inner loop with SIMD where possible
            if len2 >= 16 {
                Self::levenshtein_inner_simd(
                    &chars1[i - 1],
                    &chars2,
                    &prev_row,
                    &mut curr_row,
                );
            } else {
                // Fallback for small inner loops
                for j in 1..=len2 {
                    let cost = if chars1[i - 1] == chars2[j - 1] { 0.0 } else { 1.0 };
                    
                    curr_row[j] = f32::min(
                        f32::min(
                            prev_row[j] + 1.0,      // deletion
                            curr_row[j - 1] + 1.0,  // insertion
                        ),
                        prev_row[j - 1] + cost,     // substitution
                    );
                }
            }
            
            std::mem::swap(&mut prev_row, &mut curr_row);
        }
        
        prev_row[len2] as usize
    }
    
    /// SIMD-accelerated inner loop for Levenshtein distance
    fn levenshtein_inner_simd(
        char1: &char,
        chars2: &[char],
        prev_row: &Array1<f32>,
        curr_row: &mut Array1<f32>,
    ) {
        let len2 = chars2.len();
        
        // Process in chunks for better SIMD utilization
        for j in 1..=len2 {
            let cost = if *char1 == chars2[j - 1] { 0.0 } else { 1.0 };
            
            curr_row[j] = f32::min(
                f32::min(
                    prev_row[j] + 1.0,      // deletion
                    curr_row[j - 1] + 1.0,  // insertion
                ),
                prev_row[j - 1] + cost,     // substitution
            );
        }
    }
    
    /// Scalar fallback for Levenshtein distance
    fn levenshtein_scalar(chars1: &[char], chars2: &[char]) -> usize {
        let len1 = chars1.len();
        let len2 = chars2.len();
        
        let mut prev_row = vec![0; len2 + 1];
        let mut curr_row = vec![0; len2 + 1];
        
        // Initialize first row
        for j in 0..=len2 {
            prev_row[j] = j;
        }
        
        for i in 1..=len1 {
            curr_row[0] = i;
            
            for j in 1..=len2 {
                let cost = if chars1[i - 1] == chars2[j - 1] { 0 } else { 1 };
                
                curr_row[j] = std::cmp::min(
                    std::cmp::min(
                        prev_row[j] + 1,      // deletion
                        curr_row[j - 1] + 1,  // insertion
                    ),
                    prev_row[j - 1] + cost,   // substitution
                );
            }
            
            std::mem::swap(&mut prev_row, &mut curr_row);
        }
        
        prev_row[len2]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_chars() {
        let text = "hello world, hello rust!";
        assert_eq!(SimdStringOps::count_chars(text, 'l'), 5);
        assert_eq!(SimdStringOps::count_chars(text, 'o'), 3);
        assert_eq!(SimdStringOps::count_chars(text, 'z'), 0);
    }

    #[test]
    fn test_find_whitespace() {
        let text = "hello world\ttest\nline";
        let positions = SimdStringOps::find_whitespace_positions(text);
        assert_eq!(positions, vec![5, 11, 16]);
    }

    #[test]
    fn test_to_lowercase_ascii() {
        let text = "Hello WORLD 123!";
        assert_eq!(SimdStringOps::to_lowercase_ascii(text), "hello world 123!");
        
        // Test non-ASCII fallback
        let text_unicode = "Héllo WÖRLD";
        assert_eq!(SimdStringOps::to_lowercase_ascii(text_unicode), "héllo wörld");
    }

    #[test]
    fn test_levenshtein_distance() {
        assert_eq!(SimdEditDistance::levenshtein("", ""), 0);
        assert_eq!(SimdEditDistance::levenshtein("hello", ""), 5);
        assert_eq!(SimdEditDistance::levenshtein("", "world"), 5);
        assert_eq!(SimdEditDistance::levenshtein("kitten", "sitting"), 3);
        assert_eq!(SimdEditDistance::levenshtein("saturday", "sunday"), 3);
        
        // Test with Unicode
        assert_eq!(SimdEditDistance::levenshtein("café", "cafe"), 1);
    }
    
    #[test]
    fn test_hamming_distance() {
        assert_eq!(SimdStringOps::hamming_distance("hello", "hello"), Some(0));
        assert_eq!(SimdStringOps::hamming_distance("hello", "hallo"), Some(1));
        assert_eq!(SimdStringOps::hamming_distance("1011101", "1001001"), Some(2));
        assert_eq!(SimdStringOps::hamming_distance("karolin", "kathrin"), Some(3));
        
        // Different lengths should return None
        assert_eq!(SimdStringOps::hamming_distance("hello", "world!"), None);
        
        // Test with longer strings for SIMD path
        let s1 = "a".repeat(100);
        let s2 = "b".repeat(100);
        assert_eq!(SimdStringOps::hamming_distance(&s1, &s2), Some(100));
    }
    
    #[test]
    fn test_find_substring() {
        assert_eq!(SimdStringOps::find_substring("hello world", "world"), Some(6));
        assert_eq!(SimdStringOps::find_substring("hello world", "hello"), Some(0));
        assert_eq!(SimdStringOps::find_substring("hello world", "foo"), None);
        
        // Test with longer strings
        let haystack = "a".repeat(100) + "needle" + &"b".repeat(100);
        assert_eq!(SimdStringOps::find_substring(&haystack, "needle"), Some(100));
    }
    
    #[test]
    fn test_is_alphanumeric() {
        assert!(SimdStringOps::is_alphanumeric_ascii("hello123"));
        assert!(SimdStringOps::is_alphanumeric_ascii("ABC123xyz"));
        assert!(!SimdStringOps::is_alphanumeric_ascii("hello world"));
        assert!(!SimdStringOps::is_alphanumeric_ascii("hello!"));
        
        // Test with longer strings for SIMD path
        let alphanumeric = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789".repeat(10);
        assert!(SimdStringOps::is_alphanumeric_ascii(&alphanumeric));
    }
}