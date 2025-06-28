//! Graph embedding algorithms and utilities
//!
//! This module provides graph embedding algorithms including Node2Vec, DeepWalk,
//! and other representation learning methods for graphs.

use crate::base::{DiGraph, EdgeWeight, Graph, Node};
use crate::error::{GraphError, Result};
use rand::prelude::*;
use rand::Rng;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use serde::{Deserialize, Serialize};
use scirs2_core::simd_ops::SimdUnifiedOps;
use scirs2_core::parallel_ops::*;

/// Configuration for Node2Vec embedding algorithm
#[derive(Debug, Clone)]
pub struct Node2VecConfig {
    /// Dimensions of the embedding vectors
    pub dimensions: usize,
    /// Length of each random walk
    pub walk_length: usize,
    /// Number of random walks per node
    pub num_walks: usize,
    /// Window size for skip-gram model
    pub window_size: usize,
    /// Return parameter p (likelihood of immediate revisiting)
    pub p: f64,
    /// In-out parameter q (exploration vs exploitation)
    pub q: f64,
    /// Number of training epochs
    pub epochs: usize,
    /// Learning rate for gradient descent
    pub learning_rate: f64,
    /// Number of negative samples for training
    pub negative_samples: usize,
}

impl Default for Node2VecConfig {
    fn default() -> Self {
        Node2VecConfig {
            dimensions: 128,
            walk_length: 80,
            num_walks: 10,
            window_size: 10,
            p: 1.0,
            q: 1.0,
            epochs: 1,
            learning_rate: 0.025,
            negative_samples: 5,
        }
    }
}

/// Configuration for DeepWalk embedding algorithm
#[derive(Debug, Clone)]
pub struct DeepWalkConfig {
    /// Dimensions of the embedding vectors
    pub dimensions: usize,
    /// Length of each random walk
    pub walk_length: usize,
    /// Number of random walks per node
    pub num_walks: usize,
    /// Window size for skip-gram model
    pub window_size: usize,
    /// Number of training epochs
    pub epochs: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Number of negative samples
    pub negative_samples: usize,
}

impl Default for DeepWalkConfig {
    fn default() -> Self {
        DeepWalkConfig {
            dimensions: 128,
            walk_length: 40,
            num_walks: 80,
            window_size: 5,
            epochs: 1,
            learning_rate: 0.025,
            negative_samples: 5,
        }
    }
}

/// A random walk on a graph
#[derive(Debug, Clone)]
pub struct RandomWalk<N: Node> {
    /// The sequence of nodes in the walk
    pub nodes: Vec<N>,
}

/// Skip-gram training context pair
#[derive(Debug, Clone)]
pub struct ContextPair<N: Node> {
    /// Target node
    pub target: N,
    /// Context node
    pub context: N,
}

/// Negative sampling configuration
#[derive(Debug, Clone)]
pub struct NegativeSampler<N: Node> {
    /// Vocabulary (all nodes)
    vocabulary: Vec<N>,
    /// Frequency distribution for sampling
    frequencies: Vec<f64>,
    /// Cumulative distribution for fast sampling
    cumulative: Vec<f64>,
}

impl<N: Node> NegativeSampler<N> {
    /// Create a new negative sampler from graph
    pub fn new<E, Ix>(graph: &Graph<N, E, Ix>) -> Self
    where
        N: Clone,
        E: EdgeWeight,
        Ix: petgraph::graph::IndexType,
    {
        let vocabulary: Vec<N> = graph.nodes().into_iter().cloned().collect();
        let node_degrees = vocabulary
            .iter()
            .map(|node| graph.degree(node).unwrap_or(0) as f64)
            .collect::<Vec<_>>();
        
        // Use subsampling with power 0.75 as in Word2Vec
        let total_degree: f64 = node_degrees.iter().sum();
        let frequencies: Vec<f64> = node_degrees
            .iter()
            .map(|d| (d / total_degree).powf(0.75))
            .collect();
        
        let total_freq: f64 = frequencies.iter().sum();
        let frequencies: Vec<f64> = frequencies
            .iter()
            .map(|f| f / total_freq)
            .collect();
        
        // Build cumulative distribution
        let mut cumulative = vec![0.0; frequencies.len()];
        cumulative[0] = frequencies[0];
        for i in 1..frequencies.len() {
            cumulative[i] = cumulative[i - 1] + frequencies[i];
        }
        
        NegativeSampler {
            vocabulary,
            frequencies,
            cumulative,
        }
    }
    
    /// Sample a negative node
    pub fn sample(&self, rng: &mut impl Rng) -> Option<&N> {
        if self.vocabulary.is_empty() {
            return None;
        }
        
        let r = rng.random::<f64>();
        for (i, &cum_freq) in self.cumulative.iter().enumerate() {
            if r <= cum_freq {
                return Some(&self.vocabulary[i]);
            }
        }
        
        self.vocabulary.last()
    }
    
    /// Sample multiple negative nodes excluding target and context
    pub fn sample_negatives(
        &self,
        count: usize,
        exclude: &HashSet<&N>,
        rng: &mut impl Rng,
    ) -> Vec<N> {
        let mut negatives = Vec::new();
        let mut attempts = 0;
        let max_attempts = count * 10; // Prevent infinite loops
        
        while negatives.len() < count && attempts < max_attempts {
            if let Some(candidate) = self.sample(rng) {
                if !exclude.contains(candidate) {
                    negatives.push(candidate.clone());
                }
            }
            attempts += 1;
        }
        
        negatives
    }
}

/// Node embedding vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    /// The embedding vector
    pub vector: Vec<f64>,
}

impl Embedding {
    /// Create a new embedding with given dimensions
    pub fn new(dimensions: usize) -> Self {
        Embedding {
            vector: vec![0.0; dimensions],
        }
    }

    /// Create a random embedding
    pub fn random(dimensions: usize, rng: &mut impl Rng) -> Self {
        let vector: Vec<f64> = (0..dimensions)
            .map(|_| rng.random_range(-0.5..0.5))
            .collect();
        Embedding { vector }
    }

    /// Get the dimensionality of the embedding
    pub fn dimensions(&self) -> usize {
        self.vector.len()
    }

    /// Calculate cosine similarity with another embedding (SIMD optimized)
    pub fn cosine_similarity(&self, other: &Embedding) -> Result<f64> {
        if self.vector.len() != other.vector.len() {
            return Err(GraphError::InvalidGraph(
                "Embeddings must have same dimensions".to_string(),
            ));
        }

        // Use SIMD optimized cosine similarity when available
        if self.vector.len() >= 4 {
            let similarity = f64::simd_cosine_similarity(&self.vector, &other.vector);
            Ok(similarity)
        } else {
            // Fallback for small vectors
            let dot_product: f64 = self
                .vector
                .iter()
                .zip(other.vector.iter())
                .map(|(a, b)| a * b)
                .sum();

            let norm_a = self.norm();
            let norm_b = other.norm();

            if norm_a == 0.0 || norm_b == 0.0 {
                Ok(0.0)
            } else {
                Ok(dot_product / (norm_a * norm_b))
            }
        }
    }

    /// Calculate L2 norm of the embedding (SIMD optimized)
    pub fn norm(&self) -> f64 {
        if self.vector.len() >= 4 {
            f64::simd_norm(&self.vector)
        } else {
            // Fallback for small vectors
            self.vector.iter().map(|x| x * x).sum::<f64>().sqrt()
        }
    }

    /// Normalize the embedding to unit length
    pub fn normalize(&mut self) {
        let norm = self.norm();
        if norm > 0.0 {
            for x in &mut self.vector {
                *x /= norm;
            }
        }
    }

    /// Add another embedding (element-wise)
    pub fn add(&mut self, other: &Embedding) -> Result<()> {
        if self.vector.len() != other.vector.len() {
            return Err(GraphError::InvalidGraph(
                "Embeddings must have same dimensions".to_string(),
            ));
        }

        for (a, b) in self.vector.iter_mut().zip(other.vector.iter()) {
            *a += b;
        }
        Ok(())
    }

    /// Scale the embedding by a scalar
    pub fn scale(&mut self, factor: f64) {
        for x in &mut self.vector {
            *x *= factor;
        }
    }
    
    /// Compute dot product with another embedding (SIMD optimized)
    pub fn dot_product(&self, other: &Embedding) -> Result<f64> {
        if self.vector.len() != other.vector.len() {
            return Err(GraphError::InvalidGraph(
                "Embeddings must have same dimensions".to_string(),
            ));
        }
        
        // Use SIMD optimization from scirs2-core when available
        if self.vector.len() >= 4 {
            let dot = f64::simd_dot_product(&self.vector, &other.vector);
            Ok(dot)
        } else {
            // Fallback for small vectors
            let dot: f64 = self.vector
                .iter()
                .zip(other.vector.iter())
                .map(|(a, b)| a * b)
                .sum();
            Ok(dot)
        }
    }
    
    /// Sigmoid activation function
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
    
    /// Update embedding using gradient (SIMD optimized)
    pub fn update_gradient(&mut self, gradient: &[f64], learning_rate: f64) {
        if self.vector.len() >= 4 && gradient.len() >= 4 {
            // Use SIMD optimized gradient update
            f64::simd_scaled_add_assign(&mut self.vector, gradient, -learning_rate);
        } else {
            // Fallback for small vectors
            for (emb, &grad) in self.vector.iter_mut().zip(gradient.iter()) {
                *emb -= learning_rate * grad;
            }
        }
    }
}

/// Graph embedding model
#[derive(Debug)]
pub struct EmbeddingModel<N: Node> {
    /// Node embeddings (input vectors)
    pub embeddings: HashMap<N, Embedding>,
    /// Context embeddings (output vectors) for skip-gram
    pub context_embeddings: HashMap<N, Embedding>,
    /// Dimensionality of embeddings
    pub dimensions: usize,
}

impl<N: Node> EmbeddingModel<N> {
    /// Create a new embedding model
    pub fn new(dimensions: usize) -> Self {
        EmbeddingModel {
            embeddings: HashMap::new(),
            context_embeddings: HashMap::new(),
            dimensions,
        }
    }

    /// Get embedding for a node
    pub fn get_embedding(&self, node: &N) -> Option<&Embedding> {
        self.embeddings.get(node)
    }

    /// Set embedding for a node
    pub fn set_embedding(&mut self, node: N, embedding: Embedding) -> Result<()> {
        if embedding.dimensions() != self.dimensions {
            return Err(GraphError::InvalidGraph(
                "Embedding dimensions don't match model".to_string(),
            ));
        }
        self.embeddings.insert(node, embedding);
        Ok(())
    }

    /// Initialize random embeddings for all nodes
    pub fn initialize_random<E, Ix>(&mut self, graph: &Graph<N, E, Ix>, rng: &mut impl Rng)
    where
        N: Clone,
        E: EdgeWeight,
        Ix: petgraph::graph::IndexType,
    {
        for node in graph.nodes() {
            let embedding = Embedding::random(self.dimensions, rng);
            let context_embedding = Embedding::random(self.dimensions, rng);
            self.embeddings.insert(node.clone(), embedding);
            self.context_embeddings.insert(node.clone(), context_embedding);
        }
    }

    /// Initialize random embeddings for directed graph
    pub fn initialize_random_digraph<E, Ix>(
        &mut self,
        graph: &DiGraph<N, E, Ix>,
        rng: &mut impl Rng,
    ) where
        N: Clone,
        E: EdgeWeight,
        Ix: petgraph::graph::IndexType,
    {
        for node in graph.nodes() {
            let embedding = Embedding::random(self.dimensions, rng);
            let context_embedding = Embedding::random(self.dimensions, rng);
            self.embeddings.insert(node.clone(), embedding);
            self.context_embeddings.insert(node.clone(), context_embedding);
        }
    }

    /// Find k most similar nodes to a given node
    pub fn most_similar(&self, node: &N, k: usize) -> Result<Vec<(N, f64)>>
    where
        N: Clone,
    {
        let target_embedding = self.embeddings.get(node).ok_or(GraphError::NodeNotFound)?;

        let mut similarities = Vec::new();

        for (other_node, other_embedding) in &self.embeddings {
            if other_node != node {
                let similarity = target_embedding.cosine_similarity(other_embedding)?;
                similarities.push((other_node.clone(), similarity));
            }
        }

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similarities.truncate(k);

        Ok(similarities)
    }
    
    /// Train skip-gram model on context pairs with negative sampling
    pub fn train_skip_gram(
        &mut self,
        pairs: &[ContextPair<N>],
        negative_sampler: &NegativeSampler<N>,
        learning_rate: f64,
        negative_samples: usize,
        rng: &mut impl Rng,
    ) -> Result<()> {
        for pair in pairs {
            // Get embeddings
            let target_emb = self.embeddings.get(&pair.target)
                .ok_or(GraphError::NodeNotFound)?
                .clone();
            let context_emb = self.context_embeddings.get(&pair.context)
                .ok_or(GraphError::NodeNotFound)?
                .clone();
            
            // Positive sample: maximize probability of context given target
            let positive_score = target_emb.dot_product(&context_emb)?;
            let positive_prob = Embedding::sigmoid(positive_score);
            
            // Compute gradients for positive sample
            let positive_error = 1.0 - positive_prob;
            let mut target_gradient = vec![0.0; self.dimensions];
            let mut context_gradient = vec![0.0; self.dimensions];
            
            for i in 0..self.dimensions {
                target_gradient[i] += positive_error * context_emb.vector[i];
                context_gradient[i] += positive_error * target_emb.vector[i];
            }
            
            // Negative samples: minimize probability of negative contexts
            let exclude_set: HashSet<&N> = [&pair.target, &pair.context].iter().cloned().collect();
            let negatives = negative_sampler.sample_negatives(negative_samples, &exclude_set, rng);
            
            for negative in &negatives {
                if let Some(neg_context_emb) = self.context_embeddings.get(negative) {
                    let negative_score = target_emb.dot_product(neg_context_emb)?;
                    let negative_prob = Embedding::sigmoid(negative_score);
                    
                    // Negative sample error
                    let negative_error = -negative_prob;
                    
                    for i in 0..self.dimensions {
                        target_gradient[i] += negative_error * neg_context_emb.vector[i];
                        // Update negative context embedding
                        let neg_context_grad = negative_error * target_emb.vector[i];
                        if let Some(neg_context_emb_mut) = self.context_embeddings.get_mut(negative) {
                            neg_context_emb_mut.vector[i] -= learning_rate * neg_context_grad;
                        }
                    }
                }
            }
            
            // Apply gradients
            if let Some(target_emb_mut) = self.embeddings.get_mut(&pair.target) {
                target_emb_mut.update_gradient(&target_gradient, learning_rate);
            }
            if let Some(context_emb_mut) = self.context_embeddings.get_mut(&pair.context) {
                context_emb_mut.update_gradient(&context_gradient, learning_rate);
            }
        }
        
        Ok(())
    }
    
    /// Parallel and SIMD-optimized skip-gram training
    pub fn train_skip_gram_parallel(
        &mut self,
        pairs: &[ContextPair<N>],
        negative_sampler: &NegativeSampler<N>,
        learning_rate: f64,
        negative_samples: usize,
        rng: &mut impl Rng,
    ) -> Result<()> {
        use std::sync::Mutex;
        
        // Split pairs into chunks for parallel processing
        let chunk_size = (pairs.len() / rayon::current_num_threads()).max(1);
        let embeddings_mutex = Arc::new(Mutex::new(&mut self.embeddings));
        let context_embeddings_mutex = Arc::new(Mutex::new(&mut self.context_embeddings));
        
        pairs
            .par_chunks(chunk_size)
            .try_for_each(|chunk| -> Result<()> {
                let mut local_rng = rand::rng();
                
                for pair in chunk {
                    // Get embeddings (with locking)
                    let (target_emb, context_emb) = {
                        let embeddings = embeddings_mutex.lock().unwrap();
                        let context_embeddings = context_embeddings_mutex.lock().unwrap();
                        
                        let target_emb = embeddings.get(&pair.target)
                            .ok_or(GraphError::NodeNotFound)?
                            .clone();
                        let context_emb = context_embeddings.get(&pair.context)
                            .ok_or(GraphError::NodeNotFound)?
                            .clone();
                        
                        (target_emb, context_emb)
                    };
                    
                    // Compute gradients (SIMD optimized)
                    let positive_score = target_emb.dot_product(&context_emb)?;
                    let positive_prob = Embedding::sigmoid(positive_score);
                    let positive_error = 1.0 - positive_prob;
                    
                    let mut target_gradient = vec![0.0; self.dimensions];
                    let mut context_gradient = vec![0.0; self.dimensions];
                    
                    // SIMD-optimized gradient computation
                    if self.dimensions >= 4 {
                        f64::simd_scaled_add(&mut target_gradient, &context_emb.vector, positive_error);
                        f64::simd_scaled_add(&mut context_gradient, &target_emb.vector, positive_error);
                    } else {
                        for i in 0..self.dimensions {
                            target_gradient[i] += positive_error * context_emb.vector[i];
                            context_gradient[i] += positive_error * target_emb.vector[i];
                        }
                    }
                    
                    // Negative sampling
                    let exclude_set: HashSet<&N> = [&pair.target, &pair.context].iter().cloned().collect();
                    let negatives = negative_sampler.sample_negatives(negative_samples, &exclude_set, &mut local_rng);
                    
                    for negative in &negatives {
                        let context_embeddings = context_embeddings_mutex.lock().unwrap();
                        if let Some(neg_context_emb) = context_embeddings.get(negative) {
                            let negative_score = target_emb.dot_product(neg_context_emb)?;
                            let negative_prob = Embedding::sigmoid(negative_score);
                            let negative_error = -negative_prob;
                            
                            if self.dimensions >= 4 {
                                f64::simd_scaled_add(&mut target_gradient, &neg_context_emb.vector, negative_error);
                            } else {
                                for i in 0..self.dimensions {
                                    target_gradient[i] += negative_error * neg_context_emb.vector[i];
                                }
                            }
                        }
                    }
                    
                    // Apply gradients with locking
                    {
                        let mut embeddings = embeddings_mutex.lock().unwrap();
                        let mut context_embeddings = context_embeddings_mutex.lock().unwrap();
                        
                        if let Some(target_emb_mut) = embeddings.get_mut(&pair.target) {
                            target_emb_mut.update_gradient(&target_gradient, learning_rate);
                        }
                        if let Some(context_emb_mut) = context_embeddings.get_mut(&pair.context) {
                            context_emb_mut.update_gradient(&context_gradient, learning_rate);
                        }
                    }
                }
                Ok(())
            })?;
        
        Ok(())
    }
    
    /// Generate context pairs from random walks
    pub fn generate_context_pairs(walks: &[RandomWalk<N>], window_size: usize) -> Vec<ContextPair<N>>
    where
        N: Clone,
    {
        let mut pairs = Vec::new();
        
        for walk in walks {
            for (i, target) in walk.nodes.iter().enumerate() {
                let start = i.saturating_sub(window_size);
                let end = (i + window_size + 1).min(walk.nodes.len());
                
                for j in start..end {
                    if i != j {
                        pairs.push(ContextPair {
                            target: target.clone(),
                            context: walk.nodes[j].clone(),
                        });
                    }
                }
            }
        }
        
        pairs
    }
    
    /// Evaluate embeddings using link prediction task
    /// Returns AUC score for predicting missing edges
    pub fn evaluate_link_prediction<E, Ix>(
        &self,
        graph: &Graph<N, E, Ix>,
        test_edges: &[(N, N)],
        negative_edges: &[(N, N)],
    ) -> Result<f64>
    where
        N: Clone,
        E: EdgeWeight,
        Ix: petgraph::graph::IndexType,
    {
        let mut scores = Vec::new();
        let mut labels = Vec::new();
        
        // Positive examples (existing edges)
        for (u, v) in test_edges {
            if let (Some(u_emb), Some(v_emb)) = (self.embeddings.get(u), self.embeddings.get(v)) {
                let similarity = u_emb.cosine_similarity(v_emb)?;
                scores.push(similarity);
                labels.push(1.0);
            }
        }
        
        // Negative examples (non-existing edges)
        for (u, v) in negative_edges {
            if let (Some(u_emb), Some(v_emb)) = (self.embeddings.get(u), self.embeddings.get(v)) {
                let similarity = u_emb.cosine_similarity(v_emb)?;
                scores.push(similarity);
                labels.push(0.0);
            }
        }
        
        // Calculate AUC using trapezoidal rule
        let auc = Self::calculate_auc(&scores, &labels)?;
        Ok(auc)
    }
    
    /// Calculate AUC (Area Under Curve) for binary classification
    fn calculate_auc(scores: &[f64], labels: &[f64]) -> Result<f64> {
        if scores.len() != labels.len() {
            return Err(GraphError::ComputationError(
                "Scores and labels must have same length".to_string(),
            ));
        }
        
        // Create sorted pairs (score, label)
        let mut pairs: Vec<_> = scores.iter().zip(labels.iter()).collect();
        pairs.sort_by(|a, b| b.0.partial_cmp(a.0).unwrap_or(std::cmp::Ordering::Equal));
        
        let mut tp = 0.0; // True positives
        let mut fp = 0.0; // False positives
        let total_pos = labels.iter().sum::<f64>();
        let total_neg = labels.len() as f64 - total_pos;
        
        if total_pos == 0.0 || total_neg == 0.0 {
            return Ok(0.5); // Random classifier performance
        }
        
        let mut auc = 0.0;
        let mut prev_fp_rate = 0.0;
        
        for (_, &label) in pairs {
            if label > 0.5 {
                tp += 1.0;
            } else {
                fp += 1.0;
            }
            
            let tp_rate = tp / total_pos;
            let fp_rate = fp / total_neg;
            
            // Add trapezoid area
            auc += tp_rate * (fp_rate - prev_fp_rate);
            prev_fp_rate = fp_rate;
        }
        
        Ok(auc)
    }
    
    /// Evaluate embeddings for node classification using k-NN
    pub fn evaluate_node_classification(
        &self,
        train_nodes: &[(N, i32)], // (node, class_label)
        test_nodes: &[N],
        k: usize,
    ) -> Result<HashMap<N, i32>>
    where
        N: Clone,
    {
        let mut predictions = HashMap::new();
        
        for test_node in test_nodes {
            if let Some(test_emb) = self.embeddings.get(test_node) {
                // Find k nearest neighbors
                let mut similarities = Vec::new();
                
                for (train_node, label) in train_nodes {
                    if let Some(train_emb) = self.embeddings.get(train_node) {
                        let sim = test_emb.cosine_similarity(train_emb)?;
                        similarities.push((sim, *label));
                    }
                }
                
                // Sort by similarity (descending)
                similarities.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                
                // Take top k and predict by majority vote
                let top_k = similarities.into_iter().take(k);
                let mut label_counts: HashMap<i32, usize> = HashMap::new();
                
                for (_, label) in top_k {
                    *label_counts.entry(label).or_insert(0) += 1;
                }
                
                // Find most frequent label
                if let Some((&predicted_label, _)) = label_counts.iter().max_by_key(|(_, &count)| count) {
                    predictions.insert(test_node.clone(), predicted_label);
                }
            }
        }
        
        Ok(predictions)
    }
    
    /// Calculate classification accuracy
    pub fn calculate_accuracy(
        predictions: &HashMap<N, i32>,
        ground_truth: &HashMap<N, i32>,
    ) -> f64 {
        let mut correct = 0;
        let mut total = 0;
        
        for (node, &true_label) in ground_truth {
            if let Some(&predicted_label) = predictions.get(node) {
                if predicted_label == true_label {
                    correct += 1;
                }
                total += 1;
            }
        }
        
        if total == 0 {
            0.0
        } else {
            correct as f64 / total as f64
        }
    }
    
    /// Generate negative edges for link prediction evaluation
    pub fn generate_negative_edges<E, Ix>(
        graph: &Graph<N, E, Ix>,
        count: usize,
        rng: &mut impl Rng,
    ) -> Vec<(N, N)>
    where
        N: Clone,
        E: EdgeWeight,
        Ix: petgraph::graph::IndexType,
    {
        let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
        let mut negative_edges = Vec::new();
        let mut attempts = 0;
        let max_attempts = count * 10;
        
        while negative_edges.len() < count && attempts < max_attempts {
            if let (Some(u), Some(v)) = (nodes.choose(rng), nodes.choose(rng)) {
                if u != v && !graph.has_edge(u, v) {
                    negative_edges.push((u.clone(), v.clone()));
                }
            }
            attempts += 1;
        }
        
        negative_edges
    }
    
    /// Save embeddings to a file in JSON format
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()>
    where
        N: Serialize,
    {
        let file = File::create(path).map_err(|e| {
            GraphError::ComputationError(format!("Failed to create file: {}", e))
        })?;
        let writer = BufWriter::new(file);
        
        // Create a serializable representation
        let serializable_data = SerializableEmbeddingModel {
            embeddings: self.embeddings.clone(),
            context_embeddings: self.context_embeddings.clone(),
            dimensions: self.dimensions,
        };
        
        serde_json::to_writer_pretty(writer, &serializable_data).map_err(|e| {
            GraphError::ComputationError(format!("Failed to serialize embeddings: {}", e))
        })?;
        
        Ok(())
    }
    
    /// Load embeddings from a file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<EmbeddingModel<N>>
    where
        N: for<'de> Deserialize<'de>,
    {
        let file = File::open(path).map_err(|e| {
            GraphError::ComputationError(format!("Failed to open file: {}", e))
        })?;
        let reader = BufReader::new(file);
        
        let serializable_data: SerializableEmbeddingModel<N> = 
            serde_json::from_reader(reader).map_err(|e| {
                GraphError::ComputationError(format!("Failed to deserialize embeddings: {}", e))
            })?;
        
        Ok(EmbeddingModel {
            embeddings: serializable_data.embeddings,
            context_embeddings: serializable_data.context_embeddings,
            dimensions: serializable_data.dimensions,
        })
    }
    
    /// Save embeddings in binary format for faster loading
    pub fn save_binary<P: AsRef<Path>>(&self, path: P) -> Result<()>
    where
        N: Serialize,
    {
        let file = File::create(path).map_err(|e| {
            GraphError::ComputationError(format!("Failed to create binary file: {}", e))
        })?;
        let writer = BufWriter::new(file);
        
        let serializable_data = SerializableEmbeddingModel {
            embeddings: self.embeddings.clone(),
            context_embeddings: self.context_embeddings.clone(),
            dimensions: self.dimensions,
        };
        
        bincode::serialize_into(writer, &serializable_data).map_err(|e| {
            GraphError::ComputationError(format!("Failed to serialize embeddings to binary: {}", e))
        })?;
        
        Ok(())
    }
    
    /// Load embeddings from binary format
    pub fn load_binary<P: AsRef<Path>>(path: P) -> Result<EmbeddingModel<N>>
    where
        N: for<'de> Deserialize<'de>,
    {
        let file = File::open(path).map_err(|e| {
            GraphError::ComputationError(format!("Failed to open binary file: {}", e))
        })?;
        let reader = BufReader::new(file);
        
        let serializable_data: SerializableEmbeddingModel<N> = 
            bincode::deserialize_from(reader).map_err(|e| {
                GraphError::ComputationError(format!("Failed to deserialize binary embeddings: {}", e))
            })?;
        
        Ok(EmbeddingModel {
            embeddings: serializable_data.embeddings,
            context_embeddings: serializable_data.context_embeddings,
            dimensions: serializable_data.dimensions,
        })
    }
    
    /// Export embeddings to CSV format for analysis
    pub fn export_csv<P: AsRef<Path>>(&self, path: P) -> Result<()>
    where
        N: std::fmt::Display,
    {
        let mut file = File::create(path).map_err(|e| {
            GraphError::ComputationError(format!("Failed to create CSV file: {}", e))
        })?;
        
        // Write header
        write!(file, "node")?;
        for i in 0..self.dimensions {
            write!(file, ",dim_{}", i)?;
        }
        writeln!(file)?;
        
        // Write embeddings
        for (node, embedding) in &self.embeddings {
            write!(file, "{}", node)?;
            for value in &embedding.vector {
                write!(file, ",{}", value)?;
            }
            writeln!(file)?;
        }
        
        Ok(())
    }
    
    /// Import embeddings from CSV format
    pub fn import_csv<P: AsRef<Path>>(path: P) -> Result<EmbeddingModel<String>> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            GraphError::ComputationError(format!("Failed to read CSV file: {}", e))
        })?;
        
        let lines: Vec<&str> = content.lines().collect();
        if lines.is_empty() {
            return Err(GraphError::ComputationError("Empty CSV file".to_string()));
        }
        
        // Parse header to get dimensions
        let header = lines[0];
        let header_parts: Vec<&str> = header.split(',').collect();
        let dimensions = header_parts.len() - 1; // Subtract 1 for the node column
        
        let mut embeddings = HashMap::new();
        
        for line in lines.iter().skip(1) {
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() != dimensions + 1 {
                continue; // Skip malformed lines
            }
            
            let node = parts[0].to_string();
            let vector: Result<Vec<f64>, _> = parts[1..]
                .iter()
                .map(|s| s.parse::<f64>())
                .collect();
                
            match vector {
                Ok(v) => {
                    embeddings.insert(node, Embedding { vector: v });
                }
                Err(_) => continue, // Skip lines with parsing errors
            }
        }
        
        Ok(EmbeddingModel {
            embeddings,
            context_embeddings: HashMap::new(), // CSV doesn't include context embeddings
            dimensions,
        })
    }
}

/// Serializable version of EmbeddingModel for persistence
#[derive(Serialize, Deserialize)]
struct SerializableEmbeddingModel<N: Node> {
    embeddings: HashMap<N, Embedding>,
    context_embeddings: HashMap<N, Embedding>,
    dimensions: usize,
}

/// Random walk generator for graphs
pub struct RandomWalkGenerator<N: Node> {
    /// Random number generator
    rng: rand::rngs::ThreadRng,
    /// Phantom marker for node type
    _phantom: std::marker::PhantomData<N>,
}

impl<N: Node> Default for RandomWalkGenerator<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<N: Node> RandomWalkGenerator<N> {
    /// Create a new random walk generator
    pub fn new() -> Self {
        RandomWalkGenerator {
            rng: rand::rng(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Generate a simple random walk from a starting node
    pub fn simple_random_walk<E, Ix>(
        &mut self,
        graph: &Graph<N, E, Ix>,
        start: &N,
        length: usize,
    ) -> Result<RandomWalk<N>>
    where
        N: Clone + std::fmt::Debug,
        E: EdgeWeight,
        Ix: petgraph::graph::IndexType,
    {
        if !graph.contains_node(start) {
            return Err(GraphError::NodeNotFound);
        }

        let mut walk = vec![start.clone()];
        let mut current = start.clone();

        for _ in 1..length {
            let neighbors = graph.neighbors(&current)?;
            if neighbors.is_empty() {
                break; // No outgoing edges, stop walk
            }

            current = neighbors
                .choose(&mut self.rng)
                .ok_or(GraphError::AlgorithmError(
                    "Failed to choose neighbor".to_string(),
                ))?
                .clone();
            walk.push(current.clone());
        }

        Ok(RandomWalk { nodes: walk })
    }

    /// Generate a Node2Vec biased random walk
    pub fn node2vec_walk<E, Ix>(
        &mut self,
        graph: &Graph<N, E, Ix>,
        start: &N,
        length: usize,
        p: f64,
        q: f64,
    ) -> Result<RandomWalk<N>>
    where
        N: Clone + std::fmt::Debug,
        E: EdgeWeight + Into<f64>,
        Ix: petgraph::graph::IndexType,
    {
        if !graph.contains_node(start) {
            return Err(GraphError::NodeNotFound);
        }

        let mut walk = vec![start.clone()];
        if length == 1 {
            return Ok(RandomWalk { nodes: walk });
        }

        // First step is unbiased
        let first_neighbors = graph.neighbors(start)?;
        if first_neighbors.is_empty() {
            return Ok(RandomWalk { nodes: walk });
        }

        let current = first_neighbors
            .choose(&mut self.rng)
            .ok_or(GraphError::AlgorithmError(
                "Failed to choose first neighbor".to_string(),
            ))?
            .clone();
        walk.push(current.clone());

        // Subsequent steps use biased sampling
        for _ in 2..length {
            let current_neighbors = graph.neighbors(&current)?;
            if current_neighbors.is_empty() {
                break;
            }

            let prev = &walk[walk.len() - 2];
            let mut weights = Vec::new();

            for neighbor in &current_neighbors {
                let weight = if neighbor == prev {
                    // Return to previous node
                    1.0 / p
                } else if graph.has_edge(prev, neighbor) {
                    // Neighbor is also connected to previous node
                    1.0
                } else {
                    // New exploration
                    1.0 / q
                };
                weights.push(weight);
            }

            // Weighted random selection
            let total_weight: f64 = weights.iter().sum();
            let mut random_value = self.rng.random::<f64>() * total_weight;
            let mut selected_index = 0;

            for (i, &weight) in weights.iter().enumerate() {
                random_value -= weight;
                if random_value <= 0.0 {
                    selected_index = i;
                    break;
                }
            }

            let next_node = current_neighbors[selected_index].clone();
            walk.push(next_node.clone());
            // Update current for next iteration
            let current = next_node;
        }

        Ok(RandomWalk { nodes: walk })
    }

    /// Generate multiple random walks from a starting node
    pub fn generate_walks<E, Ix>(
        &mut self,
        graph: &Graph<N, E, Ix>,
        start: &N,
        num_walks: usize,
        walk_length: usize,
    ) -> Result<Vec<RandomWalk<N>>>
    where
        N: Clone + std::fmt::Debug,
        E: EdgeWeight,
        Ix: petgraph::graph::IndexType,
    {
        let mut walks = Vec::new();
        for _ in 0..num_walks {
            let walk = self.simple_random_walk(graph, start, walk_length)?;
            walks.push(walk);
        }
        Ok(walks)
    }
}

/// Basic Node2Vec implementation foundation
pub struct Node2Vec<N: Node> {
    config: Node2VecConfig,
    model: EmbeddingModel<N>,
    walk_generator: RandomWalkGenerator<N>,
}

impl<N: Node> Node2Vec<N> {
    /// Create a new Node2Vec instance
    pub fn new(config: Node2VecConfig) -> Self {
        Node2Vec {
            model: EmbeddingModel::new(config.dimensions),
            config,
            walk_generator: RandomWalkGenerator::new(),
        }
    }

    /// Generate training data (random walks) for Node2Vec
    pub fn generate_walks<E, Ix>(&mut self, graph: &Graph<N, E, Ix>) -> Result<Vec<RandomWalk<N>>>
    where
        N: Clone + std::fmt::Debug,
        E: EdgeWeight + Into<f64>,
        Ix: petgraph::graph::IndexType,
    {
        let mut all_walks = Vec::new();

        for node in graph.nodes() {
            for _ in 0..self.config.num_walks {
                let walk = self.walk_generator.node2vec_walk(
                    graph,
                    node,
                    self.config.walk_length,
                    self.config.p,
                    self.config.q,
                )?;
                all_walks.push(walk);
            }
        }

        Ok(all_walks)
    }

    /// Train the Node2Vec model with complete skip-gram implementation
    pub fn train<E, Ix>(&mut self, graph: &Graph<N, E, Ix>) -> Result<()>
    where
        N: Clone + std::fmt::Debug,
        E: EdgeWeight + Into<f64>,
        Ix: petgraph::graph::IndexType,
    {
        // Initialize random embeddings
        self.model
            .initialize_random(graph, &mut self.walk_generator.rng);

        // Create negative sampler
        let negative_sampler = NegativeSampler::new(graph);
        
        // Training loop over epochs
        for epoch in 0..self.config.epochs {
            // Generate walks for this epoch
            let walks = self.generate_walks(graph)?;
            
            // Generate context pairs from walks
            let context_pairs = EmbeddingModel::generate_context_pairs(&walks, self.config.window_size);
            
            // Shuffle pairs for better training
            let mut shuffled_pairs = context_pairs;
            shuffled_pairs.shuffle(&mut self.walk_generator.rng);
            
            // Train skip-gram model with negative sampling
            let current_lr = self.config.learning_rate * (1.0 - epoch as f64 / self.config.epochs as f64);
            
            self.model.train_skip_gram(
                &shuffled_pairs,
                &negative_sampler,
                current_lr,
                self.config.negative_samples,
                &mut self.walk_generator.rng,
            )?;
            
            if epoch % 10 == 0 || epoch == self.config.epochs - 1 {
                println!(
                    "Node2Vec epoch {}/{}, generated {} walks, {} context pairs",
                    epoch + 1,
                    self.config.epochs,
                    walks.len(),
                    shuffled_pairs.len()
                );
            }
        }

        Ok(())
    }

    /// Get the trained model
    pub fn model(&self) -> &EmbeddingModel<N> {
        &self.model
    }

    /// Get mutable reference to the model
    pub fn model_mut(&mut self) -> &mut EmbeddingModel<N> {
        &mut self.model
    }
}

/// Basic DeepWalk implementation foundation
pub struct DeepWalk<N: Node> {
    config: DeepWalkConfig,
    model: EmbeddingModel<N>,
    walk_generator: RandomWalkGenerator<N>,
}

impl<N: Node> DeepWalk<N> {
    /// Create a new DeepWalk instance
    pub fn new(config: DeepWalkConfig) -> Self {
        DeepWalk {
            model: EmbeddingModel::new(config.dimensions),
            config,
            walk_generator: RandomWalkGenerator::new(),
        }
    }

    /// Generate training data (random walks) for DeepWalk
    pub fn generate_walks<E, Ix>(&mut self, graph: &Graph<N, E, Ix>) -> Result<Vec<RandomWalk<N>>>
    where
        N: Clone + std::fmt::Debug,
        E: EdgeWeight,
        Ix: petgraph::graph::IndexType,
    {
        let mut all_walks = Vec::new();

        for node in graph.nodes() {
            for _ in 0..self.config.num_walks {
                let walk =
                    self.walk_generator
                        .simple_random_walk(graph, node, self.config.walk_length)?;
                all_walks.push(walk);
            }
        }

        Ok(all_walks)
    }

    /// Train the DeepWalk model with complete skip-gram implementation
    pub fn train<E, Ix>(&mut self, graph: &Graph<N, E, Ix>) -> Result<()>
    where
        N: Clone + std::fmt::Debug,
        E: EdgeWeight,
        Ix: petgraph::graph::IndexType,
    {
        // Initialize random embeddings
        self.model
            .initialize_random(graph, &mut self.walk_generator.rng);

        // Create negative sampler
        let negative_sampler = NegativeSampler::new(graph);
        
        // Training loop over epochs
        for epoch in 0..self.config.epochs {
            // Generate walks for this epoch
            let walks = self.generate_walks(graph)?;
            
            // Generate context pairs from walks
            let context_pairs = EmbeddingModel::generate_context_pairs(&walks, self.config.window_size);
            
            // Shuffle pairs for better training
            let mut shuffled_pairs = context_pairs;
            shuffled_pairs.shuffle(&mut self.walk_generator.rng);
            
            // Train skip-gram model with negative sampling
            let current_lr = self.config.learning_rate * (1.0 - epoch as f64 / self.config.epochs as f64);
            
            self.model.train_skip_gram(
                &shuffled_pairs,
                &negative_sampler,
                current_lr,
                self.config.negative_samples,
                &mut self.walk_generator.rng,
            )?;
            
            if epoch % 10 == 0 || epoch == self.config.epochs - 1 {
                println!(
                    "DeepWalk epoch {}/{}, generated {} walks, {} context pairs",
                    epoch + 1,
                    self.config.epochs,
                    walks.len(),
                    shuffled_pairs.len()
                );
            }
        }

        Ok(())
    }

    /// Get the trained model
    pub fn model(&self) -> &EmbeddingModel<N> {
        &self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_creation() {
        let embedding = Embedding::new(64);
        assert_eq!(embedding.dimensions(), 64);
        assert_eq!(embedding.vector.len(), 64);
    }

    #[test]
    fn test_embedding_similarity() {
        let mut emb1 = Embedding {
            vector: vec![1.0, 0.0, 0.0],
        };
        let mut emb2 = Embedding {
            vector: vec![0.0, 1.0, 0.0],
        };
        let mut emb3 = Embedding {
            vector: vec![1.0, 0.0, 0.0],
        };

        emb1.normalize();
        emb2.normalize();
        emb3.normalize();

        // Test cosine similarity
        let sim_orthogonal = emb1.cosine_similarity(&emb2).unwrap();
        let sim_identical = emb1.cosine_similarity(&emb3).unwrap();

        assert!((sim_orthogonal - 0.0).abs() < 1e-10);
        assert!((sim_identical - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_random_walk_generation() {
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_edge(1, 2, 1.0).unwrap();
        graph.add_edge(2, 3, 1.0).unwrap();
        graph.add_edge(3, 4, 1.0).unwrap();
        graph.add_edge(4, 1, 1.0).unwrap(); // Create a cycle

        let mut generator = RandomWalkGenerator::new();
        let walk = generator.simple_random_walk(&graph, &1, 10).unwrap();

        assert!(!walk.nodes.is_empty());
        assert_eq!(walk.nodes[0], 1); // Should start with the specified node
        assert!(walk.nodes.len() <= 10); // Should not exceed requested length
    }

    #[test]
    fn test_embedding_model() {
        let mut model: EmbeddingModel<i32> = EmbeddingModel::new(32);
        let embedding = Embedding::random(32, &mut rand::rng());

        model.set_embedding(1, embedding).unwrap();
        assert!(model.get_embedding(&1).is_some());
        assert!(model.get_embedding(&2).is_none());
    }

    #[test]
    fn test_node2vec_config() {
        let config = Node2VecConfig::default();
        assert_eq!(config.dimensions, 128);
        assert_eq!(config.walk_length, 80);
        assert_eq!(config.p, 1.0);
        assert_eq!(config.q, 1.0);
    }

    #[test]
    fn test_deepwalk_initialization() {
        let config = DeepWalkConfig::default();
        let deepwalk: DeepWalk<i32> = DeepWalk::new(config);
        assert_eq!(deepwalk.model.dimensions, 128);
    }

    #[test]
    fn test_node2vec_walk_generation() {
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_edge(1, 2, 1.0).unwrap();
        graph.add_edge(2, 3, 1.0).unwrap();
        graph.add_edge(3, 1, 1.0).unwrap();

        let config = Node2VecConfig::default();
        let mut node2vec = Node2Vec::new(config);

        let walks = node2vec.generate_walks(&graph).unwrap();
        assert!(!walks.is_empty());

        // Each node should generate num_walks walks
        let expected_total_walks = graph.node_count() * node2vec.config.num_walks;
        assert_eq!(walks.len(), expected_total_walks);
    }
}
