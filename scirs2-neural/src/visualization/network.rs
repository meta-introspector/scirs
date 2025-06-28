//! Network architecture visualization for neural networks
//!
//! This module provides comprehensive tools for visualizing neural network architectures
//! including layout algorithms, rendering capabilities, and interactive features.

use super::config::{ImageFormat, VisualizationConfig};
use crate::error::{NeuralError, Result};
use crate::models::sequential::Sequential;

use ndarray;
use num_traits::Float;
use serde::Serialize;
use std::fmt::Debug;
use std::fs;
use std::path::PathBuf;

/// Network architecture visualizer
#[allow(dead_code)]
pub struct NetworkVisualizer<F: Float + Debug + ndarray::ScalarOperand> {
    /// Model to visualize
    model: Sequential<F>,
    /// Visualization configuration
    config: VisualizationConfig,
    /// Cached layout information
    layout_cache: Option<NetworkLayout>,
}

/// Network layout information
#[derive(Debug, Clone, Serialize)]
pub struct NetworkLayout {
    /// Layer positions
    pub layer_positions: Vec<LayerPosition>,
    /// Connection information
    pub connections: Vec<Connection>,
    /// Bounding box
    pub bounds: BoundingBox,
    /// Layout algorithm used
    pub algorithm: LayoutAlgorithm,
}

/// Layer position in the visualization
#[derive(Debug, Clone, Serialize)]
pub struct LayerPosition {
    /// Layer name/identifier
    pub name: String,
    /// Layer type
    pub layer_type: String,
    /// Position coordinates
    pub position: Point2D,
    /// Layer dimensions
    pub size: Size2D,
    /// Input/output information
    pub io_info: LayerIOInfo,
    /// Visual properties
    pub visual_props: LayerVisualProps,
}

/// Point in 2D space
#[derive(Debug, Clone, Serialize)]
pub struct Point2D {
    /// X coordinate
    pub x: f32,
    /// Y coordinate
    pub y: f32,
}

/// Size in 2D space
#[derive(Debug, Clone, Serialize)]
pub struct Size2D {
    /// Width
    pub width: f32,
    /// Height
    pub height: f32,
}

/// Layer input/output information
#[derive(Debug, Clone, Serialize)]
pub struct LayerIOInfo {
    /// Input shape
    pub input_shape: Vec<usize>,
    /// Output shape
    pub output_shape: Vec<usize>,
    /// Parameter count
    pub parameter_count: usize,
    /// Computation complexity (FLOPs)
    pub flops: u64,
}

/// Visual properties for layer rendering
#[derive(Debug, Clone, Serialize)]
pub struct LayerVisualProps {
    /// Fill color
    pub fill_color: String,
    /// Border color
    pub border_color: String,
    /// Border width
    pub border_width: f32,
    /// Opacity
    pub opacity: f32,
    /// Layer icon/symbol
    pub icon: Option<String>,
}

/// Connection between layers
#[derive(Debug, Clone, Serialize)]
pub struct Connection {
    /// Source layer index
    pub from_layer: usize,
    /// Target layer index
    pub to_layer: usize,
    /// Connection type
    pub connection_type: ConnectionType,
    /// Visual properties
    pub visual_props: ConnectionVisualProps,
    /// Data flow information
    pub data_flow: DataFlowInfo,
}

/// Type of connection between layers
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum ConnectionType {
    /// Standard forward connection
    Forward,
    /// Skip/residual connection
    Skip,
    /// Attention connection
    Attention,
    /// Recurrent connection
    Recurrent,
    /// Custom connection
    Custom(String),
}

/// Visual properties for connection rendering
#[derive(Debug, Clone, Serialize)]
pub struct ConnectionVisualProps {
    /// Line color
    pub color: String,
    /// Line width
    pub width: f32,
    /// Line style
    pub style: LineStyle,
    /// Arrow style
    pub arrow: ArrowStyle,
    /// Opacity
    pub opacity: f32,
}

/// Line style for connections
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum LineStyle {
    /// Solid line
    Solid,
    /// Dashed line
    Dashed,
    /// Dotted line
    Dotted,
    /// Dash-dot line
    DashDot,
}

/// Arrow style for connections
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum ArrowStyle {
    /// No arrow
    None,
    /// Simple arrow
    Simple,
    /// Block arrow
    Block,
    /// Curved arrow
    Curved,
}

/// Data flow information
#[derive(Debug, Clone, Serialize)]
pub struct DataFlowInfo {
    /// Tensor shape flowing through connection
    pub tensor_shape: Vec<usize>,
    /// Data type
    pub data_type: String,
    /// Estimated memory usage in bytes
    pub memory_usage: usize,
    /// Throughput information
    pub throughput: Option<ThroughputInfo>,
}

/// Throughput information for data flow
#[derive(Debug, Clone, Serialize)]
pub struct ThroughputInfo {
    /// Samples per second
    pub samples_per_second: f64,
    /// Bytes per second
    pub bytes_per_second: u64,
    /// Latency in milliseconds
    pub latency_ms: f64,
}

/// Bounding box for layout
#[derive(Debug, Clone, Serialize)]
pub struct BoundingBox {
    /// Minimum X coordinate
    pub min_x: f32,
    /// Minimum Y coordinate
    pub min_y: f32,
    /// Maximum X coordinate
    pub max_x: f32,
    /// Maximum Y coordinate
    pub max_y: f32,
}

/// Layout algorithm for network visualization
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum LayoutAlgorithm {
    /// Hierarchical layout (top-down)
    Hierarchical,
    /// Force-directed layout
    ForceDirected,
    /// Circular layout
    Circular,
    /// Grid layout
    Grid,
    /// Custom layout
    Custom(String),
}

/// Layer information for analysis
#[derive(Debug, Clone, Serialize)]
pub struct LayerInfo {
    /// Layer name
    pub layer_name: String,
    /// Layer index
    pub layer_index: usize,
    /// Layer type
    pub layer_type: String,
}

// Implementation for NetworkVisualizer

impl<
        F: Float + Debug + 'static + num_traits::FromPrimitive + ndarray::ScalarOperand + Send + Sync,
    > NetworkVisualizer<F>
{
    /// Create a new network visualizer
    pub fn new(model: Sequential<F>, config: VisualizationConfig) -> Self {
        Self {
            model,
            config,
            layout_cache: None,
        }
    }

    /// Generate network architecture visualization
    pub fn visualize_architecture(&mut self) -> Result<PathBuf> {
        // Compute network layout
        let layout = self.compute_layout()?;
        self.layout_cache = Some(layout.clone());

        // Generate visualization based on format
        match self.config.image_format {
            ImageFormat::SVG => self.generate_svg_visualization(&layout),
            ImageFormat::HTML => self.generate_html_visualization(&layout),
            ImageFormat::JSON => self.generate_json_visualization(&layout),
            _ => self.generate_svg_visualization(&layout), // Default to SVG
        }
    }

    /// Compute network layout using specified algorithm
    fn compute_layout(&self) -> Result<NetworkLayout> {
        // Analyze model structure
        let layer_info = self.analyze_model_structure()?;

        // Choose layout algorithm based on network complexity
        let algorithm = self.select_layout_algorithm(&layer_info);

        // Compute positions using selected algorithm
        let (positions, connections) = match algorithm {
            LayoutAlgorithm::Hierarchical => self.compute_hierarchical_layout(&layer_info)?,
            LayoutAlgorithm::ForceDirected => self.compute_force_directed_layout(&layer_info)?,
            LayoutAlgorithm::Circular => self.compute_circular_layout(&layer_info)?,
            LayoutAlgorithm::Grid => self.compute_grid_layout(&layer_info)?,
            LayoutAlgorithm::Custom(_) => self.compute_hierarchical_layout(&layer_info)?, // Fallback
        };

        // Compute bounding box
        let bounds = self.compute_bounds(&positions);

        Ok(NetworkLayout {
            layer_positions: positions,
            connections,
            bounds,
            algorithm,
        })
    }

    fn analyze_model_structure(&self) -> Result<Vec<LayerInfo>> {
        let mut layer_info = Vec::new();

        // For Sequential models, we can iterate through the layers
        let layers = self.model.layers();

        for (index, layer) in layers.iter().enumerate() {
            let layer_type = layer.layer_type().to_string();
            let layer_name = format!("{}_{}", layer_type, index);

            layer_info.push(LayerInfo {
                layer_name,
                layer_index: index,
                layer_type,
            });
        }

        // If no layers found, return error
        if layer_info.is_empty() {
            return Err(NeuralError::InvalidArgument(
                "Model has no layers".to_string(),
            ));
        }

        Ok(layer_info)
    }

    fn select_layout_algorithm(&self, _layer_info: &[LayerInfo]) -> LayoutAlgorithm {
        // For now, default to hierarchical layout
        // In a full implementation, this would analyze the network structure
        // and choose the most appropriate layout algorithm
        LayoutAlgorithm::Hierarchical
    }

    fn compute_hierarchical_layout(
        &self,
        layer_info: &[LayerInfo],
    ) -> Result<(Vec<LayerPosition>, Vec<Connection>)> {
        if layer_info.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        let mut positions = Vec::new();
        let mut connections = Vec::new();

        // Layout parameters
        let layer_width = 120.0;
        let layer_height = 60.0;
        let vertical_spacing = 100.0;
        let horizontal_spacing = 150.0;

        // Calculate total width and starting position
        let total_width = (layer_info.len() as f32 - 1.0) * horizontal_spacing + layer_width;
        let start_x = -total_width / 2.0 + layer_width / 2.0;
        let start_y = -(layer_info.len() as f32 - 1.0) * vertical_spacing / 2.0;

        // Create layer positions
        for (i, layer) in layer_info.iter().enumerate() {
            let x = start_x;
            let y = start_y + i as f32 * vertical_spacing;

            // Determine layer visual properties based on type
            let (fill_color, border_color, icon) = match layer.layer_type.as_str() {
                "Dense" => (
                    "#4CAF50".to_string(),
                    "#2E7D32".to_string(),
                    Some("◯".to_string()),
                ),
                "Conv2D" => (
                    "#2196F3".to_string(),
                    "#1565C0".to_string(),
                    Some("⬜".to_string()),
                ),
                "Conv1D" => (
                    "#03A9F4".to_string(),
                    "#0277BD".to_string(),
                    Some("▬".to_string()),
                ),
                "MaxPool2D" | "AvgPool2D" => (
                    "#FF9800".to_string(),
                    "#E65100".to_string(),
                    Some("▣".to_string()),
                ),
                "Dropout" => (
                    "#9C27B0".to_string(),
                    "#6A1B9A".to_string(),
                    Some("×".to_string()),
                ),
                "BatchNorm" => (
                    "#607D8B".to_string(),
                    "#37474F".to_string(),
                    Some("∼".to_string()),
                ),
                "Activation" => (
                    "#FFC107".to_string(),
                    "#F57C00".to_string(),
                    Some("∘".to_string()),
                ),
                "LSTM" => (
                    "#E91E63".to_string(),
                    "#AD1457".to_string(),
                    Some("⟲".to_string()),
                ),
                "GRU" => (
                    "#F44336".to_string(),
                    "#C62828".to_string(),
                    Some("⟳".to_string()),
                ),
                "Attention" => (
                    "#673AB7".to_string(),
                    "#4527A0".to_string(),
                    Some("◉".to_string()),
                ),
                _ => (
                    "#9E9E9E".to_string(),
                    "#424242".to_string(),
                    Some("?".to_string()),
                ),
            };

            // Estimate parameter count (simplified)
            let parameter_count = match layer.layer_type.as_str() {
                "Dense" => 10000, // Placeholder
                "Conv2D" => 5000,
                "Conv1D" => 3000,
                _ => 0,
            };

            // Estimate FLOPs (simplified)
            let flops = match layer.layer_type.as_str() {
                "Dense" => 100000,
                "Conv2D" => 500000,
                "Conv1D" => 200000,
                _ => 1000,
            };

            let position = LayerPosition {
                name: layer.layer_name.clone(),
                layer_type: layer.layer_type.clone(),
                position: Point2D { x, y },
                size: Size2D {
                    width: layer_width,
                    height: layer_height,
                },
                io_info: LayerIOInfo {
                    input_shape: vec![32, 32, 3],  // Placeholder
                    output_shape: vec![32, 32, 3], // Placeholder
                    parameter_count,
                    flops,
                },
                visual_props: LayerVisualProps {
                    fill_color,
                    border_color,
                    border_width: 2.0,
                    opacity: 0.9,
                    icon,
                },
            };

            positions.push(position);
        }

        // Create connections between adjacent layers
        for i in 0..(layer_info.len().saturating_sub(1)) {
            let connection = Connection {
                from_layer: i,
                to_layer: i + 1,
                connection_type: ConnectionType::Forward,
                visual_props: ConnectionVisualProps {
                    color: "#666666".to_string(),
                    width: 2.0,
                    style: LineStyle::Solid,
                    arrow: ArrowStyle::Simple,
                    opacity: 0.8,
                },
                data_flow: DataFlowInfo {
                    tensor_shape: vec![32, 32, 3], // Placeholder
                    data_type: "f32".to_string(),
                    memory_usage: 4096, // Placeholder
                    throughput: Some(ThroughputInfo {
                        samples_per_second: 1000.0,
                        bytes_per_second: 4096000,
                        latency_ms: 1.0,
                    }),
                },
            };

            connections.push(connection);
        }

        Ok((positions, connections))
    }

    fn compute_force_directed_layout(
        &self,
        _layer_info: &[LayerInfo],
    ) -> Result<(Vec<LayerPosition>, Vec<Connection>)> {
        // TODO: Implement force-directed layout algorithm
        Err(NeuralError::NotImplementedError(
            "Force-directed layout not yet implemented".to_string(),
        ))
    }

    fn compute_circular_layout(
        &self,
        _layer_info: &[LayerInfo],
    ) -> Result<(Vec<LayerPosition>, Vec<Connection>)> {
        // TODO: Implement circular layout algorithm
        Err(NeuralError::NotImplementedError(
            "Circular layout not yet implemented".to_string(),
        ))
    }

    fn compute_grid_layout(
        &self,
        _layer_info: &[LayerInfo],
    ) -> Result<(Vec<LayerPosition>, Vec<Connection>)> {
        // TODO: Implement grid layout algorithm
        Err(NeuralError::NotImplementedError(
            "Grid layout not yet implemented".to_string(),
        ))
    }

    fn compute_bounds(&self, positions: &[LayerPosition]) -> BoundingBox {
        if positions.is_empty() {
            return BoundingBox {
                min_x: 0.0,
                min_y: 0.0,
                max_x: 100.0,
                max_y: 100.0,
            };
        }

        let mut min_x = f32::INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut max_y = f32::NEG_INFINITY;

        for pos in positions {
            min_x = min_x.min(pos.position.x - pos.size.width / 2.0);
            min_y = min_y.min(pos.position.y - pos.size.height / 2.0);
            max_x = max_x.max(pos.position.x + pos.size.width / 2.0);
            max_y = max_y.max(pos.position.y + pos.size.height / 2.0);
        }

        BoundingBox {
            min_x,
            min_y,
            max_x,
            max_y,
        }
    }

    fn generate_svg_visualization(&self, layout: &NetworkLayout) -> Result<PathBuf> {
        let output_path = self.config.output_dir.join("network_architecture.svg");

        // Generate SVG content
        let svg_content = self.create_svg_content(layout)?;

        // Write to file
        fs::write(&output_path, svg_content)
            .map_err(|e| NeuralError::IOError(format!("Failed to write SVG file: {}", e)))?;

        Ok(output_path)
    }

    fn generate_html_visualization(&self, layout: &NetworkLayout) -> Result<PathBuf> {
        let output_path = self.config.output_dir.join("network_architecture.html");

        // Generate HTML content with interactive features
        let html_content = self.create_html_content(layout)?;

        // Write to file
        fs::write(&output_path, html_content)
            .map_err(|e| NeuralError::IOError(format!("Failed to write HTML file: {}", e)))?;

        Ok(output_path)
    }

    fn generate_json_visualization(&self, layout: &NetworkLayout) -> Result<PathBuf> {
        let output_path = self.config.output_dir.join("network_architecture.json");

        // Serialize layout to JSON
        let json_content = serde_json::to_string_pretty(&layout).map_err(|e| {
            NeuralError::SerializationError(format!("Failed to serialize layout: {}", e))
        })?;

        // Write to file
        fs::write(&output_path, json_content)
            .map_err(|e| NeuralError::IOError(format!("Failed to write JSON file: {}", e)))?;

        Ok(output_path)
    }

    fn create_svg_content(&self, layout: &NetworkLayout) -> Result<String> {
        let bounds = &layout.bounds;
        let margin = 50.0;

        // Calculate SVG dimensions
        let svg_width = (bounds.max_x - bounds.min_x + 2.0 * margin) as u32;
        let svg_height = (bounds.max_y - bounds.min_y + 2.0 * margin) as u32;

        // Calculate viewBox to center the network
        let viewbox_x = bounds.min_x - margin;
        let viewbox_y = bounds.min_y - margin;
        let viewbox_width = bounds.max_x - bounds.min_x + 2.0 * margin;
        let viewbox_height = bounds.max_y - bounds.min_y + 2.0 * margin;

        let mut svg = format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<svg width="{}" height="{}" viewBox="{} {} {} {}" xmlns="http://www.w3.org/2000/svg">
  <title>Neural Network Architecture</title>
  <defs>
    <style>
      .layer-rect {{ stroke-width: 2; }}
      .connection {{ fill: none; marker-end: url(#arrowhead); }}
      .layer-text {{ font-family: Arial, sans-serif; font-size: 11px; text-anchor: middle; fill: white; font-weight: bold; }}
      .layer-info {{ font-family: Arial, sans-serif; font-size: 9px; text-anchor: middle; fill: #333; }}
      .layer-icon {{ font-family: Arial, sans-serif; font-size: 16px; text-anchor: middle; fill: white; font-weight: bold; }}
    </style>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#666666"/>
    </marker>
  </defs>
  
  <!-- Background -->
  <rect x="{}" y="{}" width="{}" height="{}" fill="#{}" stroke="#{}"/>
  
"#,
            svg_width, svg_height, viewbox_x, viewbox_y, viewbox_width, viewbox_height,
            viewbox_x, viewbox_y, viewbox_width, viewbox_height, "f8f9fa", "dee2e6"
        );

        // Draw connections first (so they appear behind layers)
        for connection in &layout.connections {
            if connection.from_layer < layout.layer_positions.len()
                && connection.to_layer < layout.layer_positions.len()
            {
                let from_pos = &layout.layer_positions[connection.from_layer];
                let to_pos = &layout.layer_positions[connection.to_layer];

                // Calculate connection points (bottom of source to top of target)
                let x1 = from_pos.position.x;
                let y1 = from_pos.position.y + from_pos.size.height / 2.0;
                let x2 = to_pos.position.x;
                let y2 = to_pos.position.y - to_pos.size.height / 2.0;

                let stroke_width = connection.visual_props.width;
                let stroke_color = &connection.visual_props.color;
                let opacity = connection.visual_props.opacity;

                svg.push_str(&format!(
                    r#"  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="{}" opacity="{}" class="connection"/>
"#,
                    x1, y1, x2, y2, stroke_color, stroke_width, opacity
                ));
            }
        }

        // Draw layers
        for (i, layer_pos) in layout.layer_positions.iter().enumerate() {
            let x = layer_pos.position.x - layer_pos.size.width / 2.0;
            let y = layer_pos.position.y - layer_pos.size.height / 2.0;
            let width = layer_pos.size.width;
            let height = layer_pos.size.height;

            let fill_color = &layer_pos.visual_props.fill_color;
            let border_color = &layer_pos.visual_props.border_color;
            let border_width = layer_pos.visual_props.border_width;
            let opacity = layer_pos.visual_props.opacity;

            // Draw layer rectangle
            svg.push_str(&format!(
                r#"  <rect x="{}" y="{}" width="{}" height="{}" fill="{}" stroke="{}" stroke-width="{}" opacity="{}" rx="5" class="layer-rect"/>
"#,
                x, y, width, height, fill_color, border_color, border_width, opacity
            ));

            // Draw layer icon if available
            if let Some(ref icon) = layer_pos.visual_props.icon {
                svg.push_str(&format!(
                    r#"  <text x="{}" y="{}" class="layer-icon">{}</text>
"#,
                    layer_pos.position.x,
                    layer_pos.position.y - 5.0,
                    icon
                ));
            }

            // Draw layer name
            svg.push_str(&format!(
                r#"  <text x="{}" y="{}" class="layer-text">{}</text>
"#,
                layer_pos.position.x,
                layer_pos.position.y + 8.0,
                layer_pos.layer_type
            ));

            // Draw parameter info below the layer
            let param_text = if layer_pos.io_info.parameter_count > 0 {
                format!("{}K params", layer_pos.io_info.parameter_count / 1000)
            } else {
                "No params".to_string()
            };

            svg.push_str(&format!(
                r#"  <text x="{}" y="{}" class="layer-info">{}</text>
"#,
                layer_pos.position.x,
                y + height + 15.0,
                param_text
            ));

            // Draw layer index
            svg.push_str(&format!(
                r#"  <text x="{}" y="{}" class="layer-info">Layer {}</text>
"#,
                layer_pos.position.x,
                y - 10.0,
                i
            ));
        }

        // Add legend
        let legend_x = viewbox_x + 10.0;
        let legend_y = viewbox_y + viewbox_height - 100.0;

        svg.push_str(&format!(
            r#"  <!-- Legend -->
  <rect x="{}" y="{}" width="200" height="80" fill="white" stroke="#ccc" stroke-width="1" opacity="0.9" rx="5"/>
  <text x="{}" y="{}" font-family="Arial" font-size="12" font-weight="bold" fill="#333">Legend</text>
  <text x="{}" y="{}" font-family="Arial" font-size="10" fill="#666">◯ Dense Layer</text>
  <text x="{}" y="{}" font-family="Arial" font-size="10" fill="#666">⬜ Conv2D Layer</text>
  <text x="{}" y="{}" font-family="Arial" font-size="10" fill="#666">× Dropout Layer</text>
  <text x="{}" y="{}" font-family="Arial" font-size="10" fill="#666">∼ BatchNorm Layer</text>
"#,
            legend_x, legend_y,
            legend_x + 10.0, legend_y + 15.0,
            legend_x + 10.0, legend_y + 30.0,
            legend_x + 10.0, legend_y + 45.0,
            legend_x + 10.0, legend_y + 60.0,
            legend_x + 10.0, legend_y + 75.0
        ));

        svg.push_str("</svg>");

        Ok(svg)
    }

    fn create_html_content(&self, _layout: &NetworkLayout) -> Result<String> {
        // TODO: Implement interactive HTML generation
        let html_template = r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Neural Network Architecture</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
        #visualization { width: 100%; height: 600px; border: 1px solid #ccc; }
        .controls { margin-bottom: 20px; }
        button { padding: 8px 16px; margin: 4px; }
    </style>
</head>
<body>
    <h1>Neural Network Architecture Visualization</h1>
    <div class="controls">
        <button onclick="zoomIn()">Zoom In</button>
        <button onclick="zoomOut()">Zoom Out</button>
        <button onclick="resetView()">Reset View</button>
        <button onclick="toggleLabels()">Toggle Labels</button>
    </div>
    <div id="visualization">
        <p>Interactive network visualization not yet implemented</p>
    </div>
    
    <script>
        function zoomIn() { console.log('Zoom in'); }
        function zoomOut() { console.log('Zoom out'); }
        function resetView() { console.log('Reset view'); }
        function toggleLabels() { console.log('Toggle labels'); }
        
        // TODO: Implement interactive visualization logic
    </script>
</body>
</html>"#
            .to_string();

        Ok(html_template)
    }

    /// Get the cached layout if available
    pub fn get_cached_layout(&self) -> Option<&NetworkLayout> {
        self.layout_cache.as_ref()
    }

    /// Clear the layout cache
    pub fn clear_cache(&mut self) {
        self.layout_cache = None;
    }

    /// Update the visualization configuration
    pub fn update_config(&mut self, config: VisualizationConfig) {
        self.config = config;
        self.clear_cache(); // Clear cache when config changes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::Dense;
    use rand::SeedableRng;

    #[test]
    fn test_network_visualizer_creation() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut model = Sequential::<f32>::new();
        model.add_layer(Dense::new(10, 5, Some("relu"), &mut rng).unwrap());

        let config = VisualizationConfig::default();
        let visualizer = NetworkVisualizer::new(model, config);

        assert!(visualizer.layout_cache.is_none());
    }

    #[test]
    fn test_layout_algorithm_variants() {
        let hierarchical = LayoutAlgorithm::Hierarchical;
        let force_directed = LayoutAlgorithm::ForceDirected;
        let circular = LayoutAlgorithm::Circular;
        let grid = LayoutAlgorithm::Grid;

        assert_eq!(hierarchical, LayoutAlgorithm::Hierarchical);
        assert_eq!(force_directed, LayoutAlgorithm::ForceDirected);
        assert_eq!(circular, LayoutAlgorithm::Circular);
        assert_eq!(grid, LayoutAlgorithm::Grid);
    }

    #[test]
    fn test_connection_types() {
        let forward = ConnectionType::Forward;
        let skip = ConnectionType::Skip;
        let attention = ConnectionType::Attention;
        let recurrent = ConnectionType::Recurrent;
        let custom = ConnectionType::Custom("test".to_string());

        assert_eq!(forward, ConnectionType::Forward);
        assert_eq!(skip, ConnectionType::Skip);
        assert_eq!(attention, ConnectionType::Attention);
        assert_eq!(recurrent, ConnectionType::Recurrent);

        match custom {
            ConnectionType::Custom(name) => assert_eq!(name, "test"),
            _ => panic!("Expected custom connection type"),
        }
    }

    #[test]
    fn test_bounding_box_computation() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut model = Sequential::<f32>::new();
        model.add_layer(Dense::new(10, 5, Some("relu"), &mut rng).unwrap());

        let config = VisualizationConfig::default();
        let visualizer = NetworkVisualizer::new(model, config);

        // Test empty positions
        let empty_positions = vec![];
        let bounds = visualizer.compute_bounds(&empty_positions);
        assert_eq!(bounds.min_x, 0.0);
        assert_eq!(bounds.min_y, 0.0);
        assert_eq!(bounds.max_x, 100.0);
        assert_eq!(bounds.max_y, 100.0);
    }

    #[test]
    fn test_point_2d() {
        let point = Point2D { x: 10.0, y: 20.0 };
        assert_eq!(point.x, 10.0);
        assert_eq!(point.y, 20.0);
    }

    #[test]
    fn test_size_2d() {
        let size = Size2D {
            width: 100.0,
            height: 50.0,
        };
        assert_eq!(size.width, 100.0);
        assert_eq!(size.height, 50.0);
    }

    #[test]
    fn test_line_style_variants() {
        assert_eq!(LineStyle::Solid, LineStyle::Solid);
        assert_eq!(LineStyle::Dashed, LineStyle::Dashed);
        assert_eq!(LineStyle::Dotted, LineStyle::Dotted);
        assert_eq!(LineStyle::DashDot, LineStyle::DashDot);
    }

    #[test]
    fn test_arrow_style_variants() {
        assert_eq!(ArrowStyle::None, ArrowStyle::None);
        assert_eq!(ArrowStyle::Simple, ArrowStyle::Simple);
        assert_eq!(ArrowStyle::Block, ArrowStyle::Block);
        assert_eq!(ArrowStyle::Curved, ArrowStyle::Curved);
    }
}
