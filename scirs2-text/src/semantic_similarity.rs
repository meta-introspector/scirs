//! Advanced semantic similarity measures for text analysis
//!
//! This module provides advanced similarity metrics that go beyond simple
//! lexical matching to capture semantic relationships between texts.

use crate::error::{Result, TextError};
use crate::embeddings::Word2Vec;
use crate::tokenize::Tokenizer;
use ndarray::{Array1, Array2, ArrayView1};
use std::collections::{HashMap, HashSet};
use std::cmp;

/// Word Mover's Distance (WMD) for measuring semantic distance between documents
pub struct WordMoversDistance {
    embeddings: HashMap<String, Array1<f64>>,
}

impl WordMoversDistance {
    /// Create a new WMD calculator from pre-computed embeddings
    pub fn from_embeddings(embeddings: HashMap<String, Array1<f64>>) -> Self {
        Self { embeddings }
    }

    /// Create from a trained Word2Vec model
    pub fn from_word2vec(model: &Word2Vec, vocabulary: &[String]) -> Result<Self> {
        let mut embeddings = HashMap::new();
        
        for word in vocabulary {
            if let Ok(vector) = model.get_word_vector(word) {
                embeddings.insert(word.clone(), vector);
            }
        }
        
        if embeddings.is_empty() {
            return Err(TextError::EmbeddingError(
                "No embeddings could be extracted from the model".into()
            ));
        }
        
        Ok(Self { embeddings })
    }

    /// Calculate Word Mover's Distance between two texts
    pub fn distance(&self, text1: &str, text2: &str, tokenizer: &dyn Tokenizer) -> Result<f64> {
        let tokens1 = tokenizer.tokenize(text1)?;
        let tokens2 = tokenizer.tokenize(text2)?;

        // Filter tokens to only those with embeddings
        let tokens1: Vec<&str> = tokens1.iter()
            .map(|s| s.as_str())
            .filter(|t| self.embeddings.contains_key(*t))
            .collect();
        
        let tokens2: Vec<&str> = tokens2.iter()
            .map(|s| s.as_str())
            .filter(|t| self.embeddings.contains_key(*t))
            .collect();

        if tokens1.is_empty() || tokens2.is_empty() {
            return Err(TextError::InvalidInput(
                "No tokens with embeddings found in one or both texts".into()
            ));
        }

        // Calculate normalized frequencies
        let freq1 = Self::calculate_frequencies(&tokens1);
        let freq2 = Self::calculate_frequencies(&tokens2);

        // Build distance matrix between all word pairs
        let n1 = freq1.len();
        let n2 = freq2.len();
        let mut distance_matrix = Array2::zeros((n1, n2));

        let words1: Vec<&String> = freq1.keys().collect();
        let words2: Vec<&String> = freq2.keys().collect();

        for (i, word1) in words1.iter().enumerate() {
            for (j, word2) in words2.iter().enumerate() {
                let embed1 = &self.embeddings[*word1];
                let embed2 = &self.embeddings[*word2];
                distance_matrix[[i, j]] = Self::euclidean_distance(embed1.view(), embed2.view());
            }
        }

        // Solve optimal transport problem (simplified greedy approach)
        // For a full implementation, you would use a linear programming solver
        self.solve_transport_greedy(&freq1, &freq2, &distance_matrix, &words1, &words2)
    }

    /// Calculate word frequencies (normalized)
    fn calculate_frequencies(tokens: &[&str]) -> HashMap<String, f64> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for &token in tokens {
            *counts.entry(token.to_string()).or_insert(0) += 1;
        }

        let total = tokens.len() as f64;
        counts.into_iter()
            .map(|(word, count)| (word, count as f64 / total))
            .collect()
    }

    /// Calculate Euclidean distance between two vectors
    fn euclidean_distance(v1: ArrayView1<f64>, v2: ArrayView1<f64>) -> f64 {
        v1.iter()
            .zip(v2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Greedy approximation of optimal transport
    fn solve_transport_greedy(
        &self,
        freq1: &HashMap<String, f64>,
        freq2: &HashMap<String, f64>,
        distances: &Array2<f64>,
        words1: &[&String],
        words2: &[&String],
    ) -> Result<f64> {
        let mut remaining1: HashMap<String, f64> = freq1.clone();
        let mut remaining2: HashMap<String, f64> = freq2.clone();
        let mut total_cost = 0.0;

        // Create a list of all edges sorted by distance
        let mut edges = Vec::new();
        for (i, word1) in words1.iter().enumerate() {
            for (j, word2) in words2.iter().enumerate() {
                edges.push((distances[[i, j]], word1.to_string(), word2.to_string()));
            }
        }
        edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Greedily assign mass
        for (distance, word1, word2) in edges {
            let mass1 = remaining1.get(&word1).copied().unwrap_or(0.0);
            let mass2 = remaining2.get(&word2).copied().unwrap_or(0.0);
            
            if mass1 > 0.0 && mass2 > 0.0 {
                let transported = mass1.min(mass2);
                total_cost += transported * distance;
                
                remaining1.insert(word1.clone(), mass1 - transported);
                remaining2.insert(word2.clone(), mass2 - transported);
            }
        }

        Ok(total_cost)
    }
}

/// Soft Cosine Similarity using word similarities
pub struct SoftCosineSimilarity {
    similarity_matrix: HashMap<(String, String), f64>,
}

impl SoftCosineSimilarity {
    /// Create from pre-computed word similarities
    pub fn new(similarity_matrix: HashMap<(String, String), f64>) -> Self {
        Self { similarity_matrix }
    }

    /// Create from word embeddings by computing cosine similarities
    pub fn from_embeddings(embeddings: &HashMap<String, Array1<f64>>) -> Self {
        let mut similarity_matrix = HashMap::new();
        
        let words: Vec<&String> = embeddings.keys().collect();
        for (i, word1) in words.iter().enumerate() {
            for word2 in words.iter().skip(i) {
                let sim = Self::cosine_similarity(
                    embeddings[*word1].view(),
                    embeddings[*word2].view()
                );
                similarity_matrix.insert(((*word1).clone(), (*word2).clone()), sim);
                if word1 != word2 {
                    similarity_matrix.insert(((*word2).clone(), (*word1).clone()), sim);
                }
            }
        }
        
        Self { similarity_matrix }
    }

    /// Calculate soft cosine similarity between two texts
    pub fn similarity(
        &self,
        text1: &str,
        text2: &str,
        tokenizer: &dyn Tokenizer,
    ) -> Result<f64> {
        let tokens1 = tokenizer.tokenize(text1)?;
        let tokens2 = tokenizer.tokenize(text2)?;

        // Calculate TF vectors
        let tf1 = Self::calculate_tf(&tokens1);
        let tf2 = Self::calculate_tf(&tokens2);

        // Get all unique words
        let mut all_words: HashSet<String> = HashSet::new();
        all_words.extend(tf1.keys().cloned());
        all_words.extend(tf2.keys().cloned());

        // Calculate soft cosine similarity
        let mut numerator = 0.0;
        let mut norm1 = 0.0;
        let mut norm2 = 0.0;

        for word1 in &all_words {
            let weight1 = tf1.get(word1).copied().unwrap_or(0.0);
            
            for word2 in &all_words {
                let weight2 = tf2.get(word2).copied().unwrap_or(0.0);
                let similarity = self.get_similarity(word1, word2);
                
                numerator += weight1 * weight2 * similarity;
                
                if text1 == text1 { // Computing norm1
                    norm1 += weight1 * weight1 * similarity;
                }
                if text2 == text2 { // Computing norm2
                    norm2 += weight2 * weight2 * similarity;
                }
            }
        }

        if norm1 > 0.0 && norm2 > 0.0 {
            Ok(numerator / (norm1.sqrt() * norm2.sqrt()))
        } else {
            Ok(0.0)
        }
    }

    /// Get similarity between two words
    fn get_similarity(&self, word1: &str, word2: &str) -> f64 {
        if word1 == word2 {
            return 1.0;
        }
        
        self.similarity_matrix
            .get(&(word1.to_string(), word2.to_string()))
            .copied()
            .unwrap_or(0.0)
    }

    /// Calculate term frequencies
    fn calculate_tf(tokens: &[String]) -> HashMap<String, f64> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for token in tokens {
            *counts.entry(token.clone()).or_insert(0) += 1;
        }
        
        let max_count = counts.values().max().copied().unwrap_or(1) as f64;
        counts.into_iter()
            .map(|(word, count)| (word, count as f64 / max_count))
            .collect()
    }

    /// Calculate cosine similarity between vectors
    fn cosine_similarity(v1: ArrayView1<f64>, v2: ArrayView1<f64>) -> f64 {
        let dot: f64 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
        let norm1 = v1.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm2 = v2.iter().map(|x| x * x).sum::<f64>().sqrt();
        
        if norm1 > 0.0 && norm2 > 0.0 {
            dot / (norm1 * norm2)
        } else {
            0.0
        }
    }
}

/// Weighted Jaccard similarity with custom term weights
pub struct WeightedJaccard {
    weights: HashMap<String, f64>,
}

impl WeightedJaccard {
    /// Create with uniform weights
    pub fn new() -> Self {
        Self {
            weights: HashMap::new(),
        }
    }

    /// Create with custom term weights (e.g., IDF weights)
    pub fn with_weights(weights: HashMap<String, f64>) -> Self {
        Self { weights }
    }

    /// Calculate weighted Jaccard similarity
    pub fn similarity(
        &self,
        text1: &str,
        text2: &str,
        tokenizer: &dyn Tokenizer,
    ) -> Result<f64> {
        let tokens1 = tokenizer.tokenize(text1)?;
        let tokens2 = tokenizer.tokenize(text2)?;

        let set1: HashSet<String> = tokens1.into_iter().collect();
        let set2: HashSet<String> = tokens2.into_iter().collect();

        let intersection: HashSet<&String> = set1.intersection(&set2).collect();
        let union: HashSet<&String> = set1.union(&set2).collect();

        if union.is_empty() {
            return Ok(1.0);
        }

        // Calculate weighted intersection and union
        let weighted_intersection: f64 = intersection.iter()
            .map(|term| self.get_weight(term))
            .sum();

        let weighted_union: f64 = union.iter()
            .map(|term| self.get_weight(term))
            .sum();

        Ok(weighted_intersection / weighted_union)
    }

    /// Get weight for a term
    fn get_weight(&self, term: &str) -> f64 {
        self.weights.get(term).copied().unwrap_or(1.0)
    }
}

/// Longest Common Subsequence (LCS) based similarity
pub struct LcsSimilarity;

impl LcsSimilarity {
    /// Calculate LCS-based similarity between two texts
    pub fn similarity(text1: &str, text2: &str, tokenizer: &dyn Tokenizer) -> Result<f64> {
        let tokens1 = tokenizer.tokenize(text1)?;
        let tokens2 = tokenizer.tokenize(text2)?;

        let lcs_length = Self::lcs_length(&tokens1, &tokens2);
        let max_length = cmp::max(tokens1.len(), tokens2.len()) as f64;

        if max_length == 0.0 {
            Ok(1.0)
        } else {
            Ok(lcs_length as f64 / max_length)
        }
    }

    /// Calculate length of longest common subsequence
    fn lcs_length(seq1: &[String], seq2: &[String]) -> usize {
        let m = seq1.len();
        let n = seq2.len();
        let mut dp = vec![vec![0; n + 1]; m + 1];

        for i in 1..=m {
            for j in 1..=n {
                if seq1[i - 1] == seq2[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = cmp::max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }

        dp[m][n]
    }
}

/// Semantic similarity ensemble that combines multiple metrics
pub struct SemanticSimilarityEnsemble {
    wmd: Option<WordMoversDistance>,
    soft_cosine: Option<SoftCosineSimilarity>,
    weighted_jaccard: Option<WeightedJaccard>,
    weights: HashMap<String, f64>,
}

impl SemanticSimilarityEnsemble {
    /// Create a new ensemble
    pub fn new() -> Self {
        let mut weights = HashMap::new();
        weights.insert("wmd".to_string(), 0.3);
        weights.insert("soft_cosine".to_string(), 0.4);
        weights.insert("weighted_jaccard".to_string(), 0.2);
        weights.insert("lcs".to_string(), 0.1);

        Self {
            wmd: None,
            soft_cosine: None,
            weighted_jaccard: None,
            weights,
        }
    }

    /// Set Word Mover's Distance component
    pub fn with_wmd(mut self, wmd: WordMoversDistance) -> Self {
        self.wmd = Some(wmd);
        self
    }

    /// Set Soft Cosine Similarity component
    pub fn with_soft_cosine(mut self, soft_cosine: SoftCosineSimilarity) -> Self {
        self.soft_cosine = Some(soft_cosine);
        self
    }

    /// Set Weighted Jaccard component
    pub fn with_weighted_jaccard(mut self, weighted_jaccard: WeightedJaccard) -> Self {
        self.weighted_jaccard = Some(weighted_jaccard);
        self
    }

    /// Set custom weights for components
    pub fn with_weights(mut self, weights: HashMap<String, f64>) -> Self {
        self.weights = weights;
        self
    }

    /// Calculate ensemble similarity
    pub fn similarity(
        &self,
        text1: &str,
        text2: &str,
        tokenizer: &dyn Tokenizer,
    ) -> Result<f64> {
        let mut scores = HashMap::new();
        let mut total_weight = 0.0;

        // Calculate WMD similarity (converted from distance)
        if let Some(ref wmd) = self.wmd {
            if let Ok(distance) = wmd.distance(text1, text2, tokenizer) {
                // Convert distance to similarity (simple inverse)
                let similarity = 1.0 / (1.0 + distance);
                scores.insert("wmd".to_string(), similarity);
                total_weight += self.weights.get("wmd").copied().unwrap_or(0.0);
            }
        }

        // Calculate soft cosine similarity
        if let Some(ref soft_cosine) = self.soft_cosine {
            if let Ok(similarity) = soft_cosine.similarity(text1, text2, tokenizer) {
                scores.insert("soft_cosine".to_string(), similarity);
                total_weight += self.weights.get("soft_cosine").copied().unwrap_or(0.0);
            }
        }

        // Calculate weighted Jaccard similarity
        if let Some(ref weighted_jaccard) = self.weighted_jaccard {
            if let Ok(similarity) = weighted_jaccard.similarity(text1, text2, tokenizer) {
                scores.insert("weighted_jaccard".to_string(), similarity);
                total_weight += self.weights.get("weighted_jaccard").copied().unwrap_or(0.0);
            }
        }

        // Calculate LCS similarity
        if let Ok(lcs_sim) = LcsSimilarity::similarity(text1, text2, tokenizer) {
            scores.insert("lcs".to_string(), lcs_sim);
            total_weight += self.weights.get("lcs").copied().unwrap_or(0.0);
        }

        if scores.is_empty() {
            return Err(TextError::InvalidInput(
                "No similarity metrics could be calculated".into()
            ));
        }

        // Calculate weighted average
        let weighted_sum: f64 = scores.iter()
            .map(|(name, &score)| score * self.weights.get(name).copied().unwrap_or(0.0))
            .sum();

        Ok(weighted_sum / total_weight)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lcs_similarity() {
        let tokenizer = WordTokenizer::default();
        
        let sim1 = LcsSimilarity::similarity(
            "the quick brown fox",
            "the fast brown fox",
            &tokenizer
        ).unwrap();
        assert!(sim1 > 0.5); // Should have high similarity
        
        let sim2 = LcsSimilarity::similarity(
            "hello world",
            "goodbye moon",
            &tokenizer
        ).unwrap();
        assert!(sim2 < 0.3); // Should have low similarity
    }

    #[test]
    fn test_weighted_jaccard() {
        let tokenizer = WordTokenizer::default();
        let mut weights = HashMap::new();
        weights.insert("important".to_string(), 5.0);
        weights.insert("the".to_string(), 0.1);
        
        let weighted_jaccard = WeightedJaccard::with_weights(weights);
        
        let sim = weighted_jaccard.similarity(
            "the important document",
            "the important paper",
            &tokenizer
        ).unwrap();
        
        // Should give high weight to "important" which is common
        assert!(sim > 0.7);
    }

    #[test]
    fn test_soft_cosine_similarity() {
        // Create mock embeddings
        let mut embeddings = HashMap::new();
        embeddings.insert("cat".to_string(), arr1(&[1.0, 0.0]));
        embeddings.insert("dog".to_string(), arr1(&[0.9, 0.1]));
        embeddings.insert("car".to_string(), arr1(&[0.0, 1.0]));
        
        let soft_cosine = SoftCosineSimilarity::from_embeddings(&embeddings);
        let tokenizer = WordTokenizer::default();
        
        let sim = soft_cosine.similarity(
            "cat dog",
            "dog cat",
            &tokenizer
        ).unwrap();
        
        // Should be very high as they contain the same words
        assert!(sim > 0.9);
    }

    fn arr1(data: &[f64]) -> Array1<f64> {
        Array1::from_vec(data.to_vec())
    }
}