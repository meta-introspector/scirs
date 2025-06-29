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

    /// Create a random graph encoding
    pub fn random(rng: &mut impl rand::Rng) -> Result<Self> {
        use rand::prelude::*;

        let num_nodes = rng.random_range(5..20);
        let mut nodes = Vec::with_capacity(num_nodes);

        // Input node
        nodes.push(NodeType {
            layer_type: LayerType::Dense(128),
            is_input: true,
            is_output: false,
        });

        // Hidden nodes
        for _ in 1..num_nodes - 1 {
            let layer_type = match rng.random_range(0..6) {
                0 => LayerType::Dense(rng.random_range(1..8) * 64),
                1 => LayerType::Conv2D {
                    filters: rng.random_range(1..8) * 32,
                    kernel_size: (3, 3),
                    stride: (1, 1),
                },
                2 => LayerType::Dropout(rng.random_range(1..5) as f32 * 0.1),
                3 => LayerType::BatchNorm,
                4 => LayerType::Activation("relu".to_string()),
                _ => LayerType::MaxPool2D {
                    pool_size: (2, 2),
                    stride: (2, 2),
                },
            };

            nodes.push(NodeType {
                layer_type,
                is_input: false,
                is_output: false,
            });
        }

        // Output node
        nodes.push(NodeType {
            layer_type: LayerType::Dense(10),
            is_input: false,
            is_output: true,
        });

        // Generate edges (DAG)
        let mut edges = vec![vec![false; num_nodes]; num_nodes];
        for i in 0..num_nodes {
            for j in (i + 1)..num_nodes {
                if rng.gen_bool(0.3) || j == i + 1 {
                    edges[i][j] = true;
                }
            }
        }

        Ok(Self::new(nodes, edges))
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

        // Mutate nodes
        for node in &mut mutated.nodes {
            if !node.is_input && !node.is_output && rng.gen::<f32>() < mutation_rate {
                // Change layer type
                node.layer_type = match rng.random_range(0..5) {
                    0 => LayerType::Dense(rng.random_range(1..8) * 64),
                    1 => LayerType::Conv2D {
                        filters: rng.random_range(1..8) * 32,
                        kernel_size: (3, 3),
                        stride: (1, 1),
                    },
                    2 => LayerType::Dropout(rng.random_range(1..5) as f32 * 0.1),
                    3 => LayerType::BatchNorm,
                    _ => LayerType::Activation("relu".to_string()),
                };
            }
        }

        // Mutate edges
        for i in 0..mutated.edges.len() {
            for j in (i + 1)..mutated.edges[i].len() {
                if rng.gen::<f32>() < mutation_rate {
                    mutated.edges[i][j] = !mutated.edges[i][j];
                }
            }
        }

        Ok(Box::new(mutated))
    }

    fn crossover(&self, other: &dyn ArchitectureEncoding) -> Result<Box<dyn ArchitectureEncoding>> {
        // Try to downcast to GraphEncoding
        if let Some(other_graph) = other.to_string().contains("GraphEncoding").then_some(self) {
            use rand::prelude::*;
            let mut rng = rand::rng();

            let mut child = self.clone();

            // Crossover nodes
            for i in 0..child.nodes.len().min(other_graph.nodes.len()) {
                if rng.gen_bool(0.5) {
                    child.nodes[i] = other_graph.nodes[i].clone();
                }
            }

            // Crossover edges
            for i in 0..child.edges.len() {
                for j in 0..child.edges[i].len() {
                    if rng.gen_bool(0.5)
                        && i < other_graph.edges.len()
                        && j < other_graph.edges[i].len()
                    {
                        child.edges[i][j] = other_graph.edges[i][j];
                    }
                }
            }

            Ok(Box::new(child))
        } else {
            // Fallback to mutation
            self.mutate(0.1)
        }
    }

    fn to_architecture(&self) -> Result<Architecture> {
        let mut layers = Vec::new();
        let mut connections = Vec::new();

        // Convert nodes to layers
        for node in &self.nodes {
            if !node.is_input && !node.is_output {
                layers.push(node.layer_type.clone());
            }
        }

        // Convert edges to connections
        for i in 0..self.edges.len() {
            for j in 0..self.edges[i].len() {
                if self.edges[i][j] {
                    connections.push((i, j));
                }
            }
        }

        Ok(Architecture {
            layers,
            connections,
            width_multiplier: 1.0,
            depth_multiplier: 1.0,
        })
    }

    fn clone_box(&self) -> Box<dyn ArchitectureEncoding> {
        Box::new(self.clone())
    }

    /// Validate the encoding
    #[allow(dead_code)]
    fn validate(&self) -> Result<()> {
        if self.nodes.is_empty() {
            return Err(crate::error::NeuralError::InvalidArgument(
                "GraphEncoding must have at least one node".to_string(),
            ));
        }

        // Check for consistent adjacency matrix
        if self.edges.len() != self.nodes.len() {
            return Err(crate::error::NeuralError::InvalidShape(
                "Adjacency matrix size doesn't match number of nodes".to_string(),
            ));
        }

        for edge_row in &self.edges {
            if edge_row.len() != self.nodes.len() {
                return Err(crate::error::NeuralError::InvalidShape(
                    "Adjacency matrix is not square".to_string(),
                ));
            }
        }

        Ok(())
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
        for i in 0..self.edges.len() {
            for j in (i + 1)..self.edges[i].len() {
                if self.edges[i][j] {
                    writeln!(f, "    {} -> {}", i, j)?;
                }
            }
        }
        Ok(())
    }
}

/// Sequential encoding (for chain-like architectures)
#[derive(Debug, Clone)]
pub struct SequentialEncoding {
    /// Layer sequence
    pub layers: Vec<LayerType>,
    /// Skip connections
    pub skip_connections: Vec<(usize, usize)>,
}

impl SequentialEncoding {
    /// Create a new sequential encoding
    pub fn new(layers: Vec<LayerType>) -> Self {
        Self {
            layers,
            skip_connections: Vec::new(),
        }
    }

    /// Create a random sequential encoding
    pub fn random(rng: &mut impl rand::Rng) -> Result<Self> {
        use rand::prelude::*;

        let num_layers = rng.random_range(5..20);
        let mut layers = Vec::with_capacity(num_layers);

        for _ in 0..num_layers {
            let layer = match rng.random_range(0..7) {
                0 => LayerType::Dense(rng.random_range(1..8) * 64),
                1 => LayerType::Conv2D {
                    filters: rng.random_range(1..8) * 32,
                    kernel_size: (3, 3),
                    stride: (1, 1),
                },
                2 => LayerType::Dropout(rng.random_range(1..5) as f32 * 0.1),
                3 => LayerType::BatchNorm,
                4 => LayerType::Activation("relu".to_string()),
                5 => LayerType::MaxPool2D {
                    pool_size: (2, 2),
                    stride: (2, 2),
                },
                _ => LayerType::Flatten,
            };
            layers.push(layer);
        }

        // Add some skip connections
        let mut skip_connections = Vec::new();
        for i in 0..num_layers {
            for j in (i + 2)..num_layers {
                if rng.gen_bool(0.1) {
                    skip_connections.push((i, j));
                }
            }
        }

        Ok(Self {
            layers,
            skip_connections,
        })
    }
}

impl ArchitectureEncoding for SequentialEncoding {
    fn to_vector(&self) -> Vec<f64> {
        let mut vec = Vec::new();

        // Encode layers
        for layer in &self.layers {
            match layer {
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

        // Encode skip connections
        vec.push(self.skip_connections.len() as f64);
        for (i, j) in &self.skip_connections {
            vec.push(*i as f64);
            vec.push(*j as f64);
        }

        vec
    }

    fn from_vector(vec: &[f64]) -> Result<Self> {
        if vec.is_empty() {
            return Err(crate::error::NeuralError::ConfigError(
                "Empty vector for SequentialEncoding".to_string(),
            ));
        }

        let mut layers = Vec::new();
        let mut skip_connections = Vec::new();
        let mut idx = 0;

        // Decode layers (pairs of values: layer_type_code, parameter)
        while idx + 1 < vec.len() {
            let layer_type_code = vec[idx] as i32;
            let layer_param = vec[idx + 1];

            // Check if this is the skip connections section
            // We encode the number of skip connections as a special marker
            if layer_type_code == 0 && layer_param >= 0.0 {
                // This might be the start of skip connections section
                let num_skip_connections = layer_param as usize;
                idx += 2;

                // Decode skip connections
                for _ in 0..num_skip_connections {
                    if idx + 1 < vec.len() {
                        let from_idx = vec[idx] as usize;
                        let to_idx = vec[idx + 1] as usize;
                        skip_connections.push((from_idx, to_idx));
                        idx += 2;
                    } else {
                        break;
                    }
                }
                break;
            }

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
                _ => {
                    // If we hit an unknown layer type, treat it as end of layers
                    // and check if the remaining data represents skip connections
                    if layer_type_code == 0 {
                        let num_skip_connections = layer_param as usize;
                        idx += 2;

                        for _ in 0..num_skip_connections {
                            if idx + 1 < vec.len() {
                                let from_idx = vec[idx] as usize;
                                let to_idx = vec[idx + 1] as usize;
                                skip_connections.push((from_idx, to_idx));
                                idx += 2;
                            } else {
                                break;
                            }
                        }
                        break;
                    } else {
                        LayerType::Dense(64) // Default fallback
                    }
                }
            };

            if layer_type_code != 0 || layer_param < 0.0 {
                layers.push(layer_type);
            }
            idx += 2;
        }

        // Ensure we have at least one layer
        if layers.is_empty() {
            layers.push(LayerType::Dense(64)); // Default layer
        }

        Ok(SequentialEncoding {
            layers,
            skip_connections,
        })
    }

    fn dimension(&self) -> usize {
        self.layers.len() * 2 + 1 + self.skip_connections.len() * 2
    }

    fn mutate(&self, mutation_rate: f32) -> Result<Box<dyn ArchitectureEncoding>> {
        use rand::prelude::*;
        let mut rng = rand::rng();
        let mut mutated = self.clone();

        // Mutate layers
        for layer in &mut mutated.layers {
            if rng.gen::<f32>() < mutation_rate {
                *layer = match rng.random_range(0..7) {
                    0 => LayerType::Dense(rng.random_range(1..8) * 64),
                    1 => LayerType::Conv2D {
                        filters: rng.random_range(1..8) * 32,
                        kernel_size: (3, 3),
                        stride: (1, 1),
                    },
                    2 => LayerType::Dropout(rng.random_range(1..5) as f32 * 0.1),
                    3 => LayerType::BatchNorm,
                    4 => LayerType::Activation("relu".to_string()),
                    5 => LayerType::MaxPool2D {
                        pool_size: (2, 2),
                        stride: (2, 2),
                    },
                    _ => LayerType::Flatten,
                };
            }
        }

        // Add/remove layers
        if rng.gen::<f32>() < mutation_rate {
            if mutated.layers.len() > 3 && rng.gen_bool(0.5) {
                let idx = rng.random_range(0..mutated.layers.len());
                mutated.layers.remove(idx);
            } else if mutated.layers.len() < 30 {
                let idx = rng.random_range(0..=mutated.layers.len());
                let layer = match rng.random_range(0..5) {
                    0 => LayerType::Dense(rng.random_range(1..8) * 64),
                    1 => LayerType::Dropout(0.2),
                    2 => LayerType::BatchNorm,
                    3 => LayerType::Activation("relu".to_string()),
                    _ => LayerType::Dense(128),
                };
                mutated.layers.insert(idx, layer);
            }
        }

        Ok(Box::new(mutated))
    }

    fn crossover(&self, other: &dyn ArchitectureEncoding) -> Result<Box<dyn ArchitectureEncoding>> {
        // Try to work with any encoding by using vector representation
        use rand::prelude::*;
        let mut rng = rand::rng();

        let mut child = self.clone();

        // Simple crossover - take first half from parent1, second half from parent2
        if let Some(other_seq) = other
            .to_string()
            .contains("SequentialEncoding")
            .then_some(self)
        {
            let crossover_point = child.layers.len() / 2;

            if other_seq.layers.len() > crossover_point {
                child.layers.truncate(crossover_point);
                child
                    .layers
                    .extend_from_slice(&other_seq.layers[crossover_point..]);
            }
        }

        Ok(Box::new(child))
    }

    fn to_architecture(&self) -> Result<Architecture> {
        Ok(Architecture {
            layers: self.layers.clone(),
            connections: self.skip_connections.clone(),
            width_multiplier: 1.0,
            depth_multiplier: 1.0,
        })
    }

    fn clone_box(&self) -> Box<dyn ArchitectureEncoding> {
        Box::new(self.clone())
    }
}

impl fmt::Display for SequentialEncoding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "SequentialEncoding:")?;
        writeln!(f, "  Layers:")?;
        for (i, layer) in self.layers.iter().enumerate() {
            writeln!(f, "    {}: {:?}", i, layer)?;
        }
        if !self.skip_connections.is_empty() {
            writeln!(f, "  Skip connections:")?;
            for (i, j) in &self.skip_connections {
                writeln!(f, "    {} -> {}", i, j)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_encoding() {
        let nodes = vec![
            NodeType {
                layer_type: LayerType::Dense(128),
                is_input: true,
                is_output: false,
            },
            NodeType {
                layer_type: LayerType::Dense(64),
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
            vec![false, true, true],
            vec![false, false, true],
            vec![false, false, false],
        ];

        let encoding = GraphEncoding::new(nodes, edges);
        let vec = encoding.to_vector();
        assert!(vec.len() > 0);

        let arch = encoding.to_architecture().unwrap();
        assert_eq!(arch.layers.len(), 1); // Only hidden layer
    }

    #[test]
    fn test_sequential_encoding() {
        let layers = vec![
            LayerType::Dense(128),
            LayerType::Activation("relu".to_string()),
            LayerType::Dense(64),
            LayerType::Activation("relu".to_string()),
            LayerType::Dense(10),
        ];

        let encoding = SequentialEncoding::new(layers.clone());
        assert_eq!(encoding.layers.len(), 5);

        let vec = encoding.to_vector();
        assert!(vec.len() > 0);

        let arch = encoding.to_architecture().unwrap();
        assert_eq!(arch.layers, layers);
    }

    #[test]
    fn test_random_generation() {
        use rand::prelude::*;
        let mut rng = rand::rng();

        let graph = GraphEncoding::random(&mut rng).unwrap();
        assert!(graph.nodes.len() >= 5);

        let seq = SequentialEncoding::random(&mut rng).unwrap();
        assert!(seq.layers.len() >= 5);
    }
}
