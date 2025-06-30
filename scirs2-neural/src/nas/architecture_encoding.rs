//! Architecture encoding schemes for Neural Architecture Search

use crate::error::Result;
use crate::nas::search_space::{Architecture, LayerType};
use std::fmt;
use std::collections::HashMap;

/// Padding type for convolution layers
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Padding {
    Valid,
    Same,
    Custom(usize),
}

/// Activation type enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum ActivationType {
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    GELU,
    Swish,
    Mish,
}

/// Architectural patterns for intelligent crossover
#[derive(Debug, Clone)]
pub enum ArchitecturalPattern {
    /// Sequential pattern of layers
    Sequential {
        /// Sequence of layer types
        layers: Vec<LayerType>,
    },
    /// Skip connection pattern
    SkipConnection {
        /// Source node index
        from: usize,
        /// Target node index
        to: usize,
    },
}

/// Trait for architecture encoding schemes
pub trait ArchitectureEncoding: Send + Sync + fmt::Display {
    /// Convert to a vector representation
    fn to_vector(&self) -> Vec<f64>;

    /// Create from a vector representation
    fn from_vector(vec: &[f64]) -> Result<Self>
    where
        Self: Sized;

    /// Get the dimensionality of the encoding
    fn dimension(&self) -> usize;

    /// Mutate the architecture
    fn mutate(&self, mutation_rate: f32) -> Result<Box<dyn ArchitectureEncoding>>;

    /// Crossover with another architecture
    fn crossover(&self, other: &dyn ArchitectureEncoding) -> Result<Box<dyn ArchitectureEncoding>>;

    /// Convert to Architecture struct
    fn to_architecture(&self) -> Result<Architecture>;

    /// Clone as trait object
    fn clone_box(&self) -> Box<dyn ArchitectureEncoding>;
}

/// Graph-based encoding (for complex topologies)
#[derive(Debug, Clone)]
pub struct GraphEncoding {
    /// Node types (layer types)
    pub nodes: Vec<NodeType>,
    /// Adjacency matrix
    pub edges: Vec<Vec<bool>>,
    /// Node attributes
    pub node_attrs: Vec<NodeAttributes>,
}

#[derive(Debug, Clone)]
pub struct NodeType {
    pub layer_type: LayerType,
    pub is_input: bool,
    pub is_output: bool,
}

#[derive(Debug, Clone)]
pub struct NodeAttributes {
    pub name: String,
    pub operation_type: String,
    pub parameters: HashMap<String, f64>,
}

impl GraphEncoding {
    /// Create a new graph encoding
    pub fn new(nodes: Vec<NodeType>, edges: Vec<Vec<bool>>) -> Self {
        let node_attrs = nodes
            .iter()
            .enumerate()
            .map(|(i, _)| NodeAttributes {
                name: format!("node_{}", i),
                operation_type: "default".to_string(),
                parameters: HashMap::new(),
            })
            .collect();

        Self {
            nodes,
            edges,
            node_attrs,
        }
    }

    fn compute_complexity_factor(&self) -> f32 {
        let mut complexity = 0.0;
        
        // Count different layer types
        let mut layer_types = std::collections::HashSet::new();
        for node in &self.nodes {
            layer_types.insert(std::mem::discriminant(&node.layer_type));
        }
        complexity += layer_types.len() as f32 / self.nodes.len() as f32;
        
        // Count connections
        let mut connections = 0;
        for row in &self.edges {
            connections += row.iter().filter(|&&x| x).count();
        }
        complexity += connections as f32 / (self.nodes.len() * self.nodes.len()) as f32;
        
        complexity.min(1.0)
    }

    fn mutate_layer_types(&self, mutated: &mut GraphEncoding, rate: f32, rng: &mut impl rand::Rng) -> Result<()> {
        for node in &mut mutated.nodes {
            if !node.is_input && !node.is_output && rng.gen_bool(rate as f64) {
                node.layer_type = self.choose_random_layer_type(rng);
            }
        }
        Ok(())
    }

    fn mutate_layer_parameters(&self, mutated: &mut GraphEncoding, rate: f32, rng: &mut impl rand::Rng) -> Result<()> {
        for node in &mut mutated.nodes {
            if !node.is_input && !node.is_output && rng.gen_bool(rate as f64) {
                match &mut node.layer_type {
                    LayerType::Dense(ref mut units) => {
                        *units = rng.random_range(32..=512);
                    }
                    LayerType::Conv2D { ref mut filters, ref mut kernel_size, ref mut stride } => {
                        *filters = rng.random_range(16..=256);
                        *kernel_size = self.choose_kernel_size(rng);
                        *stride = self.choose_stride(rng);
                    }
                    LayerType::Dropout(ref mut rate) => {
                        *rate = rng.gen_range(0.1..0.5);
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }

    fn mutate_connections(&self, mutated: &mut GraphEncoding, rate: f32, rng: &mut impl rand::Rng) -> Result<()> {
        let num_nodes = mutated.nodes.len();
        for i in 0..num_nodes {
            for j in 0..num_nodes {
                if i != j && rng.gen_bool(rate as f64) {
                    let would_disconnect = self.would_disconnect_graph(&mutated.edges, i, j, num_nodes);
                    if !would_disconnect {
                        mutated.edges[i][j] = !mutated.edges[i][j];
                    }
                }
            }
        }
        Ok(())
    }

    fn mutate_architecture_structure(&self, mutated: &mut GraphEncoding, rate: f32, rng: &mut impl rand::Rng) -> Result<()> {
        if rng.gen_bool(rate as f64) && mutated.nodes.len() < 20 {
            self.add_node(mutated, rng)?;
        }
        Ok(())
    }

    fn mutate_hybrid(&self, mutated: &mut GraphEncoding, rate: f32, rng: &mut impl rand::Rng) -> Result<()> {
        self.mutate_layer_types(mutated, rate * 0.3, rng)?;
        self.mutate_layer_parameters(mutated, rate * 0.3, rng)?;
        self.mutate_connections(mutated, rate * 0.2, rng)?;
        self.mutate_architecture_structure(mutated, rate * 0.2, rng)?;
        Ok(())
    }

    fn choose_kernel_size(&self, rng: &mut impl rand::Rng) -> (usize, usize) {
        let sizes = [(1, 1), (3, 3), (5, 5), (7, 7)];
        let idx = rng.random_range(0..sizes.len());
        sizes[idx]
    }

    fn choose_stride(&self, rng: &mut impl rand::Rng) -> (usize, usize) {
        let strides = [(1, 1), (2, 2)];
        let idx = rng.random_range(0..strides.len());
        strides[idx]
    }

    fn choose_random_layer_type(&self, rng: &mut impl rand::Rng) -> LayerType {
        let layer_types = [
            LayerType::Dense(rng.random_range(32..=512)),
            LayerType::Conv2D {
                filters: rng.random_range(16..=256),
                kernel_size: self.choose_kernel_size(rng),
                stride: self.choose_stride(rng),
            },
            LayerType::Dropout(rng.gen_range(0.1..0.5)),
            LayerType::BatchNorm,
            LayerType::Activation("relu".to_string()),
        ];
        
        let idx = rng.random_range(0..layer_types.len());
        layer_types[idx].clone()
    }

    fn would_disconnect_graph(&self, edges: &[Vec<bool>], from: usize, to: usize, num_nodes: usize) -> bool {
        // Simple connectivity check
        let mut test_edges = edges.to_vec();
        test_edges[from][to] = !test_edges[from][to];
        
        // Check if all output nodes are still reachable from input nodes
        let mut reachable = vec![false; num_nodes];
        
        // Mark input nodes as reachable
        for (i, node) in self.nodes.iter().enumerate() {
            if node.is_input {
                reachable[i] = true;
            }
        }
        
        // Propagate reachability
        let mut changed = true;
        while changed {
            changed = false;
            for i in 0..num_nodes {
                if reachable[i] {
                    for j in 0..num_nodes {
                        if test_edges[i][j] && !reachable[j] {
                            reachable[j] = true;
                            changed = true;
                        }
                    }
                }
            }
        }
        
        // Check if any output node is unreachable
        for (i, node) in self.nodes.iter().enumerate() {
            if node.is_output && !reachable[i] {
                return true;
            }
        }
        
        false
    }

    fn compute_connection_utility(&self, from: usize, to: usize, arch: &GraphEncoding) -> f32 {
        // Simple utility based on layer types
        let from_layer = &arch.nodes[from].layer_type;
        let to_layer = &arch.nodes[to].layer_type;
        
        match (from_layer, to_layer) {
            (LayerType::Conv2D { .. }, LayerType::BatchNorm) => 0.9,
            (LayerType::BatchNorm, LayerType::Activation(_)) => 0.9,
            (LayerType::Dense(_), LayerType::Dropout(_)) => 0.8,
            _ => 0.5,
        }
    }

    fn add_node(&self, mutated: &mut GraphEncoding, rng: &mut impl rand::Rng) -> Result<()> {
        let new_layer_type = self.choose_random_layer_type(rng);
        let new_node = NodeType {
            layer_type: new_layer_type,
            is_input: false,
            is_output: false,
        };
        
        mutated.nodes.push(new_node);
        
        // Extend edges matrix
        let new_size = mutated.nodes.len();
        for row in &mut mutated.edges {
            row.push(false);
        }
        mutated.edges.push(vec![false; new_size]);
        
        // Add node attributes
        mutated.node_attrs.push(NodeAttributes {
            name: format!("node_{}", new_size - 1),
            operation_type: "default".to_string(),
            parameters: HashMap::new(),
        });
        
        // Connect the new node
        let from_idx = rng.random_range(0..new_size - 1);
        let to_idx = rng.random_range(0..new_size - 1);
        mutated.edges[from_idx][new_size - 1] = true;
        mutated.edges[new_size - 1][to_idx] = true;
        
        Ok(())
    }
}

impl ArchitectureEncoding for GraphEncoding {
    fn to_vector(&self) -> Vec<f64> {
        let mut vec = Vec::new();

        // First element: number of nodes
        vec.push(self.nodes.len() as f64);

        // Encode nodes
        for node in &self.nodes {
            vec.push(if node.is_input { 1.0 } else { 0.0 });
            vec.push(if node.is_output { 1.0 } else { 0.0 });

            // Encode layer type
            match &node.layer_type {
                LayerType::Dense(units) => {
                    vec.push(1.0);
                    vec.push(*units as f64);
                }
                LayerType::Conv2D { filters, .. } => {
                    vec.push(2.0);
                    vec.push(*filters as f64);
                }
                LayerType::Dropout(rate) => {
                    vec.push(3.0);
                    vec.push(*rate as f64);
                }
                LayerType::BatchNorm => {
                    vec.push(4.0);
                    vec.push(0.0);
                }
                LayerType::Activation(_) => {
                    vec.push(5.0);
                    vec.push(0.0);
                }
                _ => {
                    vec.push(0.0);
                    vec.push(0.0);
                }
            }
        }

        // Encode edges
        for row in &self.edges {
            for &edge in row {
                vec.push(if edge { 1.0 } else { 0.0 });
            }
        }

        vec
    }

    fn from_vector(vec: &[f64]) -> Result<Self> {
        if vec.is_empty() {
            return Err(crate::error::NeuralError::ConfigError(
                "Empty vector for GraphEncoding".to_string(),
            ));
        }

        // First element is the number of nodes
        let num_nodes = vec[0] as usize;
        if num_nodes == 0 {
            return Err(crate::error::NeuralError::ConfigError(
                "GraphEncoding must have at least one node".to_string(),
            ));
        }

        // Calculate expected vector size
        let expected_size = 1 + num_nodes * 4 + num_nodes * num_nodes;
        if vec.len() < expected_size {
            return Err(crate::error::NeuralError::ConfigError(
                format!("Vector too short: expected at least {}, got {}", expected_size, vec.len()),
            ));
        }

        let mut nodes = Vec::with_capacity(num_nodes);
        let mut edges = vec![vec![false; num_nodes]; num_nodes];
        let mut node_attrs = Vec::with_capacity(num_nodes);

        let mut idx = 1;

        // Decode nodes
        for _ in 0..num_nodes {
            let is_input = vec[idx] > 0.5;
            let is_output = vec[idx + 1] > 0.5;
            let layer_type_code = vec[idx + 2] as i32;
            let layer_param = vec[idx + 3];

            let layer_type = match layer_type_code {
                1 => LayerType::Dense(layer_param as usize),
                2 => LayerType::Conv2D {
                    filters: layer_param as usize,
                    kernel_size: (3, 3), // Default kernel size
                    stride: (1, 1),
                },
                3 => LayerType::Dropout(layer_param as f32),
                4 => LayerType::BatchNorm,
                5 => LayerType::Activation("relu".to_string()), // Default activation
                _ => LayerType::Dense(64), // Default fallback
            };

            nodes.push(NodeType {
                layer_type,
                is_input,
                is_output,
            });

            node_attrs.push(NodeAttributes {
                name: format!("node_{}", nodes.len() - 1),
                operation_type: "default".to_string(),
                parameters: std::collections::HashMap::new(),
            });

            idx += 4;
        }

        // Decode edges
        for i in 0..num_nodes {
            for j in 0..num_nodes {
                if idx < vec.len() {
                    edges[i][j] = vec[idx] > 0.5;
                    idx += 1;
                }
            }
        }

        Ok(GraphEncoding {
            nodes,
            edges,
            node_attrs,
        })
    }

    fn dimension(&self) -> usize {
        1 + self.nodes.len() * 4 + self.edges.len() * self.edges.len()
    }

    fn mutate(&self, mutation_rate: f32) -> Result<Box<dyn ArchitectureEncoding>> {
        use rand::prelude::*;
        let mut rng = rand::rng();
        let mut mutated = self.clone();

        // Adaptive mutation rate based on architecture complexity
        let complexity_factor = self.compute_complexity_factor();
        let adaptive_rate = mutation_rate * (1.0 + complexity_factor * 0.5);

        // Multi-type mutation strategy
        let mutation_type = rng.random_range(0..5);
        
        match mutation_type {
            0 => self.mutate_layer_types(&mut mutated, adaptive_rate, &mut rng)?,
            1 => self.mutate_layer_parameters(&mut mutated, adaptive_rate, &mut rng)?,
            2 => self.mutate_connections(&mut mutated, adaptive_rate, &mut rng)?,
            3 => self.mutate_architecture_structure(&mut mutated, adaptive_rate, &mut rng)?,
            _ => self.mutate_hybrid(&mut mutated, adaptive_rate, &mut rng)?,
        }

        Ok(Box::new(mutated))
    }

    fn crossover(&self, other: &dyn ArchitectureEncoding) -> Result<Box<dyn ArchitectureEncoding>> {
        // Simple crossover implementation - create a basic crossover
        // For now, just do a basic crossover by mixing the vector representations
        let self_vec = self.to_vector();
        let other_vec = other.to_vector();
        
        use rand::prelude::*;
        let mut rng = rand::rng();
        
        // Create a mixed vector
        let min_len = self_vec.len().min(other_vec.len());
        let mut mixed_vec = Vec::with_capacity(self_vec.len().max(other_vec.len()));
        
        // Mix the vectors randomly
        for i in 0..min_len {
            if rng.gen_bool(0.5) {
                mixed_vec.push(self_vec[i]);
            } else {
                mixed_vec.push(other_vec[i]);
            }
        }
        
        // Add remaining elements from the longer vector
        if self_vec.len() > min_len {
            mixed_vec.extend_from_slice(&self_vec[min_len..]);
        } else if other_vec.len() > min_len {
            mixed_vec.extend_from_slice(&other_vec[min_len..]);
        }
        
        // Create new GraphEncoding from mixed vector
        let result = GraphEncoding::from_vector(&mixed_vec)?;
        Ok(Box::new(result))
    }

    fn to_architecture(&self) -> Result<Architecture> {
        let mut layers = Vec::new();
        let mut connections = Vec::new();

        // Convert nodes to layers
        for node in &self.nodes {
            layers.push(node.layer_type.clone());
        }

        // Convert edges to connections
        for (i, row) in self.edges.iter().enumerate() {
            for (j, &connected) in row.iter().enumerate() {
                if connected {
                    connections.push((i, j));
                }
            }
        }

        Architecture::new(layers, connections)
    }

    fn clone_box(&self) -> Box<dyn ArchitectureEncoding> {
        Box::new(self.clone())
    }
}

impl fmt::Display for GraphEncoding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "GraphEncoding:")?;
        writeln!(f, "  Nodes: {}", self.nodes.len())?;
        for (i, node) in self.nodes.iter().enumerate() {
            write!(f, "    {}: {:?}", i, node.layer_type)?;
            if node.is_input {
                write!(f, " [INPUT]")?;
            }
            if node.is_output {
                write!(f, " [OUTPUT]")?;
            }
            writeln!(f)?;
        }
        writeln!(f, "  Edges:")?;
        for (i, row) in self.edges.iter().enumerate() {
            write!(f, "    {}: ", i)?;
            for (j, &connected) in row.iter().enumerate() {
                if connected {
                    write!(f, "{} ", j)?;
                }
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

/// Sequential encoding (for simple feed-forward networks)
#[derive(Debug, Clone)]
pub struct SequentialEncoding {
    pub layers: Vec<LayerType>,
}

impl SequentialEncoding {
    pub fn new(layers: Vec<LayerType>) -> Self {
        Self { layers }
    }

    pub fn random(rng: &mut impl rand::Rng) -> Result<Self> {
        let num_layers = rng.random_range(3..=10);
        let mut layers = Vec::with_capacity(num_layers);
        
        // Input layer
        layers.push(LayerType::Dense(rng.random_range(64..=512)));
        
        // Hidden layers
        for _ in 1..num_layers - 1 {
            let layer_type = match rng.random_range(0..4) {
                0 => LayerType::Dense(rng.random_range(32..=512)),
                1 => LayerType::Dropout(rng.gen_range(0.1..0.5)),
                2 => LayerType::BatchNorm,
                _ => LayerType::Activation("relu".to_string()),
            };
            layers.push(layer_type);
        }
        
        // Output layer
        layers.push(LayerType::Dense(rng.random_range(1..=10)));
        
        Ok(Self { layers })
    }
}

impl ArchitectureEncoding for SequentialEncoding {
    fn to_vector(&self) -> Vec<f64> {
        let mut vec = Vec::new();
        
        // First element: number of layers
        vec.push(self.layers.len() as f64);
        
        // Encode each layer
        for layer in &self.layers {
            match layer {
                LayerType::Dense(units) => {
                    vec.push(1.0);
                    vec.push(*units as f64);
                    vec.push(0.0); // padding
                }
                LayerType::Conv2D { filters, .. } => {
                    vec.push(2.0);
                    vec.push(*filters as f64);
                    vec.push(0.0); // padding
                }
                LayerType::Dropout(rate) => {
                    vec.push(3.0);
                    vec.push(*rate as f64);
                    vec.push(0.0); // padding
                }
                LayerType::BatchNorm => {
                    vec.push(4.0);
                    vec.push(0.0);
                    vec.push(0.0);
                }
                LayerType::Activation(_) => {
                    vec.push(5.0);
                    vec.push(0.0);
                    vec.push(0.0);
                }
                _ => {
                    vec.push(0.0);
                    vec.push(0.0);
                    vec.push(0.0);
                }
            }
        }
        
        vec
    }

    fn from_vector(vec: &[f64]) -> Result<Self> {
        if vec.is_empty() {
            return Err(crate::error::NeuralError::ConfigError(
                "Empty vector for SequentialEncoding".to_string(),
            ));
        }
        
        let num_layers = vec[0] as usize;
        if num_layers == 0 {
            return Err(crate::error::NeuralError::ConfigError(
                "SequentialEncoding must have at least one layer".to_string(),
            ));
        }
        
        let expected_size = 1 + num_layers * 3;
        if vec.len() < expected_size {
            return Err(crate::error::NeuralError::ConfigError(
                format!("Vector too short: expected {}, got {}", expected_size, vec.len()),
            ));
        }
        
        let mut layers = Vec::with_capacity(num_layers);
        let mut idx = 1;
        
        for _ in 0..num_layers {
            let layer_type_code = vec[idx] as i32;
            let param1 = vec[idx + 1];
            
            let layer_type = match layer_type_code {
                1 => LayerType::Dense(param1 as usize),
                2 => LayerType::Conv2D {
                    filters: param1 as usize,
                    kernel_size: (3, 3),
                    stride: (1, 1),
                },
                3 => LayerType::Dropout(param1 as f32),
                4 => LayerType::BatchNorm,
                5 => LayerType::Activation("relu".to_string()),
                _ => LayerType::Dense(64), // Default fallback
            };
            
            layers.push(layer_type);
            idx += 3;
        }
        
        Ok(Self { layers })
    }

    fn dimension(&self) -> usize {
        1 + self.layers.len() * 3
    }

    fn mutate(&self, mutation_rate: f32) -> Result<Box<dyn ArchitectureEncoding>> {
        use rand::prelude::*;
        let mut rng = rand::rng();
        let mut mutated = self.clone();
        
        // Mutate existing layers
        for layer in &mut mutated.layers {
            if rng.gen_bool(mutation_rate as f64) {
                match layer {
                    LayerType::Dense(ref mut units) => {
                        *units = rng.random_range(32..=512);
                    }
                    LayerType::Conv2D { ref mut filters, .. } => {
                        *filters = rng.random_range(16..=256);
                    }
                    LayerType::Dropout(ref mut rate) => {
                        *rate = rng.gen_range(0.1..0.5);
                    }
                    _ => {}
                }
            }
        }
        
        // Occasionally add or remove layers
        if rng.gen_bool(mutation_rate as f64 * 0.1) {
            if mutated.layers.len() < 15 && rng.gen_bool(0.7) {
                // Add layer
                let pos = rng.random_range(1..mutated.layers.len());
                let new_layer = match rng.random_range(0..4) {
                    0 => LayerType::Dense(rng.random_range(32..=512)),
                    1 => LayerType::Dropout(rng.gen_range(0.1..0.5)),
                    2 => LayerType::BatchNorm,
                    _ => LayerType::Activation("relu".to_string()),
                };
                mutated.layers.insert(pos, new_layer);
            } else if mutated.layers.len() > 3 {
                // Remove layer (but not first or last)
                let pos = rng.random_range(1..mutated.layers.len() - 1);
                mutated.layers.remove(pos);
            }
        }
        
        Ok(Box::new(mutated))
    }

    fn crossover(&self, other: &dyn ArchitectureEncoding) -> Result<Box<dyn ArchitectureEncoding>> {
        // Try to convert other to SequentialEncoding
        if other.to_string().contains("SequentialEncoding") {
            use rand::prelude::*;
            let mut rng = rand::rng();
            
            // Get the vector representations
            let self_vec = self.to_vector();
            let other_vec = other.to_vector();
            
            // Simple crossover point
            let crossover_point = rng.random_range(1..self_vec.len().min(other_vec.len()));
            
            let mut child_vec = Vec::new();
            child_vec.extend_from_slice(&self_vec[..crossover_point]);
            child_vec.extend_from_slice(&other_vec[crossover_point..]);
            
            let result = SequentialEncoding::from_vector(&child_vec)?;
            Ok(Box::new(result))
        } else {
            // Fallback to mutation
            self.mutate(0.1)
        }
    }

    fn to_architecture(&self) -> Result<Architecture> {
        let mut connections = Vec::new();
        
        // Create sequential connections
        for i in 0..self.layers.len().saturating_sub(1) {
            connections.push((i, i + 1));
        }
        
        Architecture::new(self.layers.clone(), connections)
    }

    fn clone_box(&self) -> Box<dyn ArchitectureEncoding> {
        Box::new(self.clone())
    }
}

impl fmt::Display for SequentialEncoding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "SequentialEncoding:")?;
        for (i, layer) in self.layers.iter().enumerate() {
            writeln!(f, "  {}: {:?}", i, layer)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    #[test]
    fn test_graph_encoding() {
        let nodes = vec![
            NodeType {
                layer_type: LayerType::Dense(64),
                is_input: true,
                is_output: false,
            },
            NodeType {
                layer_type: LayerType::Dense(32),
                is_input: false,
                is_output: false,
            },
            NodeType {
                layer_type: LayerType::Dense(10),
                is_input: false,
                is_output: true,
            },
        ];

        let edges = vec![
            vec![false, true, false],
            vec![false, false, true],
            vec![false, false, false],
        ];

        let encoding = GraphEncoding::new(nodes, edges);
        let vector = encoding.to_vector();
        let decoded = GraphEncoding::from_vector(&vector).unwrap();

        assert_eq!(vector[0], 3.0); // Number of nodes
        assert_eq!(decoded.nodes.len(), 3);
    }

    #[test]
    fn test_sequential_encoding() {
        let layers = vec![
            LayerType::Dense(128),
            LayerType::Activation("relu".to_string()),
            LayerType::Dense(64),
            LayerType::Dropout(0.2),
            LayerType::Dense(10),
        ];

        let encoding = SequentialEncoding::new(layers);
        let vector = encoding.to_vector();
        let decoded = SequentialEncoding::from_vector(&vector).unwrap();

        assert_eq!(vector[0], 5.0); // Number of layers
        assert_eq!(decoded.layers.len(), 5);
    }

    #[test]
    fn test_random_generation() {
        let mut rng = rand::rng();
        let seq_encoding = SequentialEncoding::random(&mut rng).unwrap();
        assert!(seq_encoding.layers.len() >= 3);
        assert!(seq_encoding.layers.len() <= 10);
    }
}