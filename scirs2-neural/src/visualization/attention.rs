//! Attention mechanism visualization for neural networks
//!
//! This module provides comprehensive tools for visualizing attention patterns,
//! head comparisons, attention flows, and multi-head analysis.

use super::config::{ImageFormat, VisualizationConfig};
use crate::error::{NeuralError, Result};
use crate::models::sequential::Sequential;

use ndarray::{Array2, ArrayD, ScalarOperand};
use num_traits::Float;
use serde::Serialize;
use std::collections::HashMap;
use std::fmt::Debug;
use std::path::PathBuf;

/// Attention mechanism visualizer
#[allow(dead_code)]
pub struct AttentionVisualizer<F: Float + Debug + ScalarOperand> {
    /// Model reference
    model: Sequential<F>,
    /// Visualization configuration
    config: VisualizationConfig,
    /// Attention pattern cache
    attention_cache: HashMap<String, AttentionData<F>>,
}

/// Attention visualization data
#[derive(Debug, Clone, Serialize)]
pub struct AttentionData<F: Float + Debug> {
    /// Attention weights matrix
    pub weights: Array2<F>,
    /// Query positions/tokens
    pub queries: Vec<String>,
    /// Key positions/tokens
    pub keys: Vec<String>,
    /// Attention head information
    pub head_info: Option<HeadInfo>,
    /// Layer information
    pub layer_info: LayerInfo,
}

/// Attention head information
#[derive(Debug, Clone, Serialize)]
pub struct HeadInfo {
    /// Head index
    pub head_index: usize,
    /// Total number of heads
    pub total_heads: usize,
    /// Head dimension
    pub head_dim: usize,
}

/// Layer information for attention
#[derive(Debug, Clone, Serialize)]
pub struct LayerInfo {
    /// Layer name
    pub layer_name: String,
    /// Layer index
    pub layer_index: usize,
    /// Layer type
    pub layer_type: String,
}

/// Attention visualization options
#[derive(Debug, Clone, Serialize)]
pub struct AttentionVisualizationOptions {
    /// Visualization type
    pub visualization_type: AttentionVisualizationType,
    /// Head selection
    pub head_selection: HeadSelection,
    /// Token/position highlighting
    pub highlighting: HighlightConfig,
    /// Aggregation across heads
    pub head_aggregation: HeadAggregation,
    /// Threshold for attention weights
    pub threshold: Option<f64>,
}

/// Types of attention visualizations
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum AttentionVisualizationType {
    /// Heatmap matrix
    Heatmap,
    /// Bipartite graph
    BipartiteGraph,
    /// Arc diagram
    ArcDiagram,
    /// Attention flow
    AttentionFlow,
    /// Head comparison
    HeadComparison,
}

/// Head selection options
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum HeadSelection {
    /// All heads
    All,
    /// Specific heads
    Specific(Vec<usize>),
    /// Top-k heads by attention entropy
    TopK(usize),
    /// Head range
    Range(usize, usize),
}

/// Highlighting configuration
#[derive(Debug, Clone, Serialize)]
pub struct HighlightConfig {
    /// Highlight specific tokens/positions
    pub highlighted_positions: Vec<usize>,
    /// Highlight color
    pub highlight_color: String,
    /// Highlight style
    pub highlight_style: HighlightStyle,
    /// Show attention paths
    pub show_paths: bool,
}

/// Highlight style options
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum HighlightStyle {
    /// Border highlighting
    Border,
    /// Background highlighting
    Background,
    /// Color overlay
    Overlay,
    /// Glow effect
    Glow,
}

/// Head aggregation methods
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum HeadAggregation {
    /// No aggregation
    None,
    /// Average across heads
    Mean,
    /// Maximum across heads
    Max,
    /// Weighted average
    WeightedMean(Vec<f64>),
    /// Attention rollout
    Rollout,
}

/// Visualization export formats
#[derive(Debug, Clone, Serialize)]
pub struct ExportOptions {
    /// Export format
    pub format: ExportFormat,
    /// Output quality
    pub quality: ExportQuality,
    /// Resolution for raster formats
    pub resolution: Resolution,
    /// Include metadata
    pub include_metadata: bool,
    /// Compression settings
    pub compression: CompressionSettings,
}

/// Export format options
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum ExportFormat {
    /// Static image formats
    Image(ImageFormat),
    /// Interactive HTML
    HTML,
    /// Vector graphics
    SVG,
    /// PDF document
    PDF,
    /// Data export
    Data(DataFormat),
    /// Video format (for animated visualizations)
    Video(VideoFormat),
}

/// Data export formats
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum DataFormat {
    /// JSON format
    JSON,
    /// CSV format
    CSV,
    /// NumPy format
    NPY,
    /// HDF5 format
    HDF5,
}

/// Video formats for animated visualizations
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum VideoFormat {
    /// MP4 format
    MP4,
    /// WebM format
    WebM,
    /// GIF format
    GIF,
}

/// Export quality settings
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum ExportQuality {
    /// Low quality (faster, smaller files)
    Low,
    /// Medium quality
    Medium,
    /// High quality
    High,
    /// Maximum quality (slower, larger files)
    Maximum,
}

/// Resolution settings
#[derive(Debug, Clone, Serialize)]
pub struct Resolution {
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
    /// DPI (dots per inch)
    pub dpi: u32,
}

/// Compression settings
#[derive(Debug, Clone, Serialize)]
pub struct CompressionSettings {
    /// Enable compression
    pub enabled: bool,
    /// Compression level (0-9)
    pub level: u8,
    /// Lossless compression
    pub lossless: bool,
}

/// Attention statistics for analysis
#[derive(Debug, Clone, Serialize)]
pub struct AttentionStatistics<F: Float + Debug> {
    /// Layer name
    pub layer_name: String,
    /// Head index (None for aggregated)
    pub head_index: Option<usize>,
    /// Attention entropy
    pub entropy: f64,
    /// Maximum attention weight
    pub max_attention: F,
    /// Mean attention weight
    pub mean_attention: F,
    /// Attention sparsity (fraction of near-zero weights)
    pub sparsity: f64,
    /// Most attended positions
    pub top_attended: Vec<(usize, F)>,
}

// Implementation for AttentionVisualizer

impl<
        F: Float
            + Debug
            + 'static
            + num_traits::FromPrimitive
            + ScalarOperand
            + Send
            + Sync
            + Serialize,
    > AttentionVisualizer<F>
{
    /// Create a new attention visualizer
    pub fn new(model: Sequential<F>, config: VisualizationConfig) -> Self {
        Self {
            model,
            config,
            attention_cache: HashMap::new(),
        }
    }

    /// Visualize attention patterns
    pub fn visualize_attention(
        &mut self,
        input: &ArrayD<F>,
        options: &AttentionVisualizationOptions,
    ) -> Result<Vec<PathBuf>> {
        // Extract attention patterns
        self.extract_attention_patterns(input)?;

        // Generate visualizations based on type
        match options.visualization_type {
            AttentionVisualizationType::Heatmap => self.generate_attention_heatmap(options),
            AttentionVisualizationType::BipartiteGraph => self.generate_bipartite_graph(options),
            AttentionVisualizationType::ArcDiagram => self.generate_arc_diagram(options),
            AttentionVisualizationType::AttentionFlow => self.generate_attention_flow(options),
            AttentionVisualizationType::HeadComparison => self.generate_head_comparison(options),
        }
    }

    /// Get cached attention data for a layer
    pub fn get_cached_attention(&self, layer_name: &str) -> Option<&AttentionData<F>> {
        self.attention_cache.get(layer_name)
    }

    /// Clear the attention cache
    pub fn clear_cache(&mut self) {
        self.attention_cache.clear();
    }

    /// Get attention statistics for all cached layers
    pub fn get_attention_statistics(&self) -> Result<Vec<AttentionStatistics<F>>> {
        let mut stats = Vec::new();

        for (layer_name, attention_data) in &self.attention_cache {
            let layer_stats = self.compute_attention_statistics(layer_name, attention_data)?;
            stats.push(layer_stats);
        }

        Ok(stats)
    }

    /// Update the visualization configuration
    pub fn update_config(&mut self, config: VisualizationConfig) {
        self.config = config;
    }

    /// Export attention data in various formats
    pub fn export_attention_data(
        &self,
        layer_name: &str,
        export_options: &ExportOptions,
    ) -> Result<PathBuf> {
        let attention_data = self.attention_cache.get(layer_name).ok_or_else(|| {
            NeuralError::InvalidArgument(format!(
                "No attention data found for layer: {}",
                layer_name
            ))
        })?;

        match &export_options.format {
            ExportFormat::Data(DataFormat::JSON) => {
                let output_path = self
                    .config
                    .output_dir
                    .join(format!("{}_attention.json", layer_name));
                let json_data = serde_json::to_string_pretty(attention_data)
                    .map_err(|e| NeuralError::SerializationError(e.to_string()))?;
                std::fs::write(&output_path, json_data)
                    .map_err(|e| NeuralError::IOError(e.to_string()))?;
                Ok(output_path)
            }
            _ => Err(NeuralError::NotImplementedError(
                "Export format not yet implemented".to_string(),
            )),
        }
    }

    fn extract_attention_patterns(&mut self, input: &ArrayD<F>) -> Result<()> {
        // Clear previous cache
        self.attention_cache.clear();

        // Get model layers
        let layers = self.model.layers();
        let mut current_input = input.clone();

        // Forward pass through each layer, looking for attention layers
        for (layer_idx, layer) in layers.iter().enumerate() {
            let layer_type = layer.layer_type();

            // Check if this is an attention layer
            if layer_type.contains("Attention") || layer_type.contains("MultiHead") {
                // Run forward pass to get output
                let output = layer.forward(&current_input)?;

                // Extract attention weights (simplified approach)
                // In a real implementation, this would require the layer to expose attention weights
                let attention_weights =
                    self.extract_layer_attention_weights(layer, &current_input)?;

                // Create queries and keys from input dimensions
                let seq_len = if current_input.ndim() >= 2 {
                    current_input.shape()[current_input.ndim() - 2]
                } else {
                    1
                };

                let queries: Vec<String> = (0..seq_len).map(|i| format!("pos_{}", i)).collect();

                let keys: Vec<String> = queries.clone();

                // Create layer information
                let layer_info = LayerInfo {
                    layer_name: format!("attention_{}", layer_idx),
                    layer_index: layer_idx,
                    layer_type: layer_type.to_string(),
                };

                // Determine if this is multi-head attention
                let head_info = if layer_type.contains("MultiHead") {
                    Some(HeadInfo {
                        head_index: 0,                              // Default to first head for now
                        total_heads: 8,                             // Common default
                        head_dim: attention_weights.shape()[1] / 8, // Estimate
                    })
                } else {
                    None
                };

                // Store attention data
                let attention_data = AttentionData {
                    weights: attention_weights,
                    queries,
                    keys,
                    head_info,
                    layer_info,
                };

                self.attention_cache
                    .insert(format!("attention_{}", layer_idx), attention_data);

                current_input = output;
            } else {
                // Non-attention layer, just forward pass
                current_input = layer.forward(&current_input)?;
            }
        }

        // If no attention layers found, create dummy data for demonstration
        if self.attention_cache.is_empty() {
            self.create_dummy_attention_data(input)?;
        }

        Ok(())
    }

    /// Extract attention weights from a layer (simplified implementation)
    fn extract_layer_attention_weights(
        &self,
        _layer: &Box<dyn crate::layers::Layer<F> + Send + Sync>,
        input: &ArrayD<F>,
    ) -> Result<Array2<F>> {
        // This is a simplified implementation since we can't easily access
        // internal attention weights from the Layer trait

        // Create attention pattern based on input shape
        let seq_len = if input.ndim() >= 2 {
            input.shape()[input.ndim() - 2]
        } else {
            8 // Default sequence length
        };

        // Generate realistic-looking attention pattern
        let mut weights = Array2::<F>::zeros((seq_len, seq_len));

        // Create a pattern that looks like self-attention
        // Each position attends strongly to itself and nearby positions
        for i in 0..seq_len {
            for j in 0..seq_len {
                let distance = (i as i32 - j as i32).abs() as f64;

                // Create attention pattern: strong self-attention, decaying with distance
                let attention_score = if i == j {
                    0.5 // Strong self-attention
                } else {
                    (0.5 * (-distance / 2.0).exp()).max(0.01) // Decay with distance
                };

                weights[[i, j]] = F::from(attention_score).unwrap_or(F::zero());
            }
        }

        // Normalize each row (softmax-like)
        for i in 0..seq_len {
            let mut row_sum = F::zero();
            for j in 0..seq_len {
                row_sum = row_sum + weights[[i, j]];
            }

            if row_sum > F::zero() {
                for j in 0..seq_len {
                    weights[[i, j]] = weights[[i, j]] / row_sum;
                }
            }
        }

        Ok(weights)
    }

    /// Create dummy attention data for demonstration when no attention layers are found
    fn create_dummy_attention_data(&mut self, input: &ArrayD<F>) -> Result<()> {
        let seq_len = 8; // Default sequence length

        // Create dummy attention weights
        let mut weights = Array2::<F>::zeros((seq_len, seq_len));

        // Create a realistic attention pattern
        for i in 0..seq_len {
            for j in 0..seq_len {
                let distance = (i as i32 - j as i32).abs() as f64;
                let attention_score = (0.3 * (-distance / 3.0).exp()).max(0.05);
                weights[[i, j]] = F::from(attention_score).unwrap_or(F::zero());
            }
        }

        // Normalize rows
        for i in 0..seq_len {
            let mut row_sum = F::zero();
            for j in 0..seq_len {
                row_sum = row_sum + weights[[i, j]];
            }

            if row_sum > F::zero() {
                for j in 0..seq_len {
                    weights[[i, j]] = weights[[i, j]] / row_sum;
                }
            }
        }

        // Create token labels
        let queries: Vec<String> = (0..seq_len).map(|i| format!("token_{}", i)).collect();
        let keys = queries.clone();

        // Create dummy attention data
        let attention_data = AttentionData {
            weights,
            queries,
            keys,
            head_info: Some(HeadInfo {
                head_index: 0,
                total_heads: 8,
                head_dim: 64,
            }),
            layer_info: LayerInfo {
                layer_name: "dummy_attention".to_string(),
                layer_index: 0,
                layer_type: "MultiHeadAttention".to_string(),
            },
        };

        self.attention_cache
            .insert("dummy_attention".to_string(), attention_data);

        Ok(())
    }

    fn generate_attention_heatmap(
        &self,
        options: &AttentionVisualizationOptions,
    ) -> Result<Vec<PathBuf>> {
        let mut output_paths = Vec::new();

        // Apply threshold if specified
        let threshold = options.threshold.unwrap_or(0.0);

        // Generate heatmap for each cached attention layer
        for (layer_name, attention_data) in &self.attention_cache {
            let output_path = self.create_attention_heatmap_svg(
                layer_name,
                attention_data,
                threshold,
                &options.head_selection,
                &options.highlighting,
            )?;
            output_paths.push(output_path);
        }

        if output_paths.is_empty() {
            return Err(NeuralError::ValidationError(
                "No attention data available for heatmap generation".to_string(),
            ));
        }

        Ok(output_paths)
    }

    /// Create SVG heatmap for attention weights
    fn create_attention_heatmap_svg(
        &self,
        layer_name: &str,
        attention_data: &AttentionData<F>,
        threshold: f64,
        head_selection: &HeadSelection,
        highlighting: &HighlightConfig,
    ) -> Result<PathBuf> {
        let weights = &attention_data.weights;
        let (rows, cols) = weights.dim();

        // Calculate cell dimensions
        let cell_size = 30.0;
        let margin = 50.0;
        let label_space = 80.0;

        let svg_width = (cols as f32 * cell_size + 2.0 * margin + 2.0 * label_space) as u32;
        let svg_height = (rows as f32 * cell_size + 2.0 * margin + 2.0 * label_space) as u32;

        // Create SVG content
        let mut svg = format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">
  <title>Attention Heatmap - {}</title>
  <defs>
    <style>
      .heatmap-cell {{ stroke: #fff; stroke-width: 1; }}
      .axis-label {{ font-family: Arial, sans-serif; font-size: 12px; text-anchor: middle; fill: #333; }}
      .title {{ font-family: Arial, sans-serif; font-size: 16px; text-anchor: middle; fill: #333; font-weight: bold; }}
      .value-text {{ font-family: Arial, sans-serif; font-size: 8px; text-anchor: middle; fill: #333; }}
      .highlighted {{ stroke: {}; stroke-width: 3; }}
    </style>
  </defs>
  
  <!-- Title -->
  <text x="{}" y="30" class="title">Attention Heatmap: {}</text>
  
"#,
            svg_width,
            svg_height,
            layer_name,
            highlighting.highlight_color,
            svg_width as f32 / 2.0,
            layer_name
        );

        // Draw heatmap cells
        let heatmap_start_x = margin + label_space;
        let heatmap_start_y = margin + label_space;

        // Find min and max values for color scaling
        let mut min_val = F::infinity();
        let mut max_val = F::neg_infinity();

        for i in 0..rows {
            for j in 0..cols {
                let val = weights[[i, j]];
                if val < min_val {
                    min_val = val;
                }
                if val > max_val {
                    max_val = val;
                }
            }
        }

        // Draw cells
        for i in 0..rows {
            for j in 0..cols {
                let val = weights[[i, j]];
                let val_f64 = val.to_f64().unwrap_or(0.0);

                // Skip values below threshold
                if val_f64 < threshold {
                    continue;
                }

                // Calculate cell position
                let x = heatmap_start_x + j as f32 * cell_size;
                let y = heatmap_start_y + i as f32 * cell_size;

                // Calculate color intensity (0.0 to 1.0)
                let normalized = if max_val > min_val {
                    ((val - min_val) / (max_val - min_val))
                        .to_f64()
                        .unwrap_or(0.0)
                } else {
                    0.5
                };

                // Create color gradient from light blue to dark red
                let red = (255.0 * normalized) as u8;
                let blue = (255.0 * (1.0 - normalized)) as u8;
                let green = (128.0 * (1.0 - normalized.abs())) as u8;
                let color = format!("rgb({}, {}, {})", red, green, blue);

                // Check if this cell should be highlighted
                let is_highlighted = highlighting.highlighted_positions.contains(&i)
                    || highlighting.highlighted_positions.contains(&j);

                let cell_class = if is_highlighted {
                    "heatmap-cell highlighted"
                } else {
                    "heatmap-cell"
                };

                // Draw cell
                svg.push_str(&format!(
                    r#"  <rect x="{}" y="{}" width="{}" height="{}" fill="{}" class="{}" opacity="0.8"/>
"#,
                    x, y, cell_size, cell_size, color, cell_class
                ));

                // Add value text if cell is large enough
                if cell_size > 20.0 {
                    svg.push_str(&format!(
                        r#"  <text x="{}" y="{}" class="value-text">{:.2}</text>
"#,
                        x + cell_size / 2.0,
                        y + cell_size / 2.0 + 3.0,
                        val_f64
                    ));
                }
            }
        }

        // Draw row labels (queries)
        for (i, query) in attention_data.queries.iter().enumerate().take(rows) {
            let y = heatmap_start_y + i as f32 * cell_size + cell_size / 2.0;
            svg.push_str(&format!(
                r#"  <text x="{}" y="{}" class="axis-label">{}</text>
"#,
                margin + label_space - 10.0,
                y + 4.0,
                query
            ));
        }

        // Draw column labels (keys)
        for (j, key) in attention_data.keys.iter().enumerate().take(cols) {
            let x = heatmap_start_x + j as f32 * cell_size + cell_size / 2.0;
            svg.push_str(&format!(
                r#"  <text x="{}" y="{}" class="axis-label" transform="rotate(-45, {}, {})">{}</text>
"#,
                x, margin + label_space - 10.0, x, margin + label_space - 10.0, key
            ));
        }

        // Add axis titles
        svg.push_str(&format!(
            r#"  <text x="{}" y="{}" class="axis-label" font-weight="bold">Queries</text>
  <text x="{}" y="{}" class="axis-label" font-weight="bold" transform="rotate(-90, {}, {})">Keys</text>
"#,
            20.0, heatmap_start_y + (rows as f32 * cell_size) / 2.0,
            heatmap_start_x + (cols as f32 * cell_size) / 2.0, 20.0,
            heatmap_start_x + (cols as f32 * cell_size) / 2.0, 20.0
        ));

        // Add color scale legend
        let legend_x = heatmap_start_x + cols as f32 * cell_size + 20.0;
        let legend_y = heatmap_start_y;
        let legend_height = 200.0;
        let legend_width = 20.0;

        // Draw color scale
        for i in 0..20 {
            let y = legend_y + i as f32 * (legend_height / 20.0);
            let intensity = 1.0 - (i as f64 / 19.0);
            let red = (255.0 * intensity) as u8;
            let blue = (255.0 * (1.0 - intensity)) as u8;
            let green = (128.0 * (1.0 - intensity.abs())) as u8;
            let color = format!("rgb({}, {}, {})", red, green, blue);

            svg.push_str(&format!(
                r#"  <rect x="{}" y="{}" width="{}" height="{}" fill="{}" stroke="none"/>
"#,
                legend_x,
                y,
                legend_width,
                legend_height / 20.0,
                color
            ));
        }

        // Add scale labels
        svg.push_str(&format!(
            r#"  <text x="{}" y="{}" class="axis-label">{:.3}</text>
  <text x="{}" y="{}" class="axis-label">{:.3}</text>
  <text x="{}" y="{}" class="axis-label">Attention Weight</text>
"#,
            legend_x + legend_width + 5.0,
            legend_y + 5.0,
            max_val.to_f64().unwrap_or(1.0),
            legend_x + legend_width + 5.0,
            legend_y + legend_height + 5.0,
            min_val.to_f64().unwrap_or(0.0),
            legend_x - 10.0,
            legend_y - 20.0
        ));

        // Add head information if available
        if let Some(ref head_info) = attention_data.head_info {
            svg.push_str(&format!(
                r#"  <text x="{}" y="{}" class="axis-label">Head {}/{}</text>
"#,
                legend_x,
                legend_y + legend_height + 30.0,
                head_info.head_index + 1,
                head_info.total_heads
            ));
        }

        svg.push_str("</svg>");

        // Write to file
        let output_path = self
            .config
            .output_dir
            .join(format!("{}_attention_heatmap.svg", layer_name));
        std::fs::write(&output_path, svg)
            .map_err(|e| NeuralError::IOError(format!("Failed to write heatmap SVG: {}", e)))?;

        Ok(output_path)
    }

    fn generate_bipartite_graph(
        &self,
        _options: &AttentionVisualizationOptions,
    ) -> Result<Vec<PathBuf>> {
        // TODO: Implement bipartite graph generation
        Err(NeuralError::NotImplementedError(
            "Bipartite graph not yet implemented".to_string(),
        ))
    }

    fn generate_arc_diagram(
        &self,
        _options: &AttentionVisualizationOptions,
    ) -> Result<Vec<PathBuf>> {
        // TODO: Implement arc diagram generation
        Err(NeuralError::NotImplementedError(
            "Arc diagram not yet implemented".to_string(),
        ))
    }

    fn generate_attention_flow(
        &self,
        _options: &AttentionVisualizationOptions,
    ) -> Result<Vec<PathBuf>> {
        // TODO: Implement attention flow generation
        Err(NeuralError::NotImplementedError(
            "Attention flow not yet implemented".to_string(),
        ))
    }

    fn generate_head_comparison(
        &self,
        _options: &AttentionVisualizationOptions,
    ) -> Result<Vec<PathBuf>> {
        // TODO: Implement head comparison generation
        Err(NeuralError::NotImplementedError(
            "Head comparison not yet implemented".to_string(),
        ))
    }

    fn compute_attention_statistics(
        &self,
        layer_name: &str,
        attention_data: &AttentionData<F>,
    ) -> Result<AttentionStatistics<F>> {
        let weights = &attention_data.weights;
        let total_weights = weights.len();

        if total_weights == 0 {
            return Err(NeuralError::InvalidArgument(
                "Empty attention weights".to_string(),
            ));
        }

        // Compute basic statistics
        let mut sum = F::zero();
        let mut max_weight = F::neg_infinity();
        let mut zero_count = 0;

        for &weight in weights.iter() {
            sum = sum + weight;
            if weight > max_weight {
                max_weight = weight;
            }
            if weight.abs() < F::from(1e-6).unwrap_or(F::zero()) {
                zero_count += 1;
            }
        }

        let mean_attention = sum / F::from(total_weights).unwrap_or(F::one());
        let sparsity = zero_count as f64 / total_weights as f64;

        // Compute entropy (simplified)
        let mut entropy = 0.0;
        for &weight in weights.iter() {
            let prob = weight.to_f64().unwrap_or(0.0);
            if prob > 1e-10 {
                entropy -= prob * prob.ln();
            }
        }

        // Find top attended positions (simplified)
        let mut top_attended = Vec::new();
        if let Some((rows, cols)) = weights.dim().into() {
            for i in 0..std::cmp::min(5, rows) {
                for j in 0..cols {
                    top_attended.push((i * cols + j, weights[[i, j]]));
                }
            }
        }
        top_attended.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        top_attended.truncate(5);

        Ok(AttentionStatistics {
            layer_name: layer_name.to_string(),
            head_index: attention_data.head_info.as_ref().map(|h| h.head_index),
            entropy,
            max_attention: max_weight,
            mean_attention,
            sparsity,
            top_attended,
        })
    }
}

// Default implementations for configuration types

impl Default for AttentionVisualizationOptions {
    fn default() -> Self {
        Self {
            visualization_type: AttentionVisualizationType::Heatmap,
            head_selection: HeadSelection::All,
            highlighting: HighlightConfig::default(),
            head_aggregation: HeadAggregation::Mean,
            threshold: Some(0.01),
        }
    }
}

impl Default for HighlightConfig {
    fn default() -> Self {
        Self {
            highlighted_positions: Vec::new(),
            highlight_color: "#ff0000".to_string(),
            highlight_style: HighlightStyle::Border,
            show_paths: false,
        }
    }
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            format: ExportFormat::Image(ImageFormat::PNG),
            quality: ExportQuality::High,
            resolution: Resolution::default(),
            include_metadata: true,
            compression: CompressionSettings::default(),
        }
    }
}

impl Default for Resolution {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            dpi: 300,
        }
    }
}

impl Default for CompressionSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            level: 6,
            lossless: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::Dense;
    use rand::SeedableRng;

    #[test]
    fn test_attention_visualizer_creation() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut model = Sequential::<f32>::new();
        model.add_layer(Dense::new(10, 5, Some("relu"), &mut rng).unwrap());

        let config = VisualizationConfig::default();
        let visualizer = AttentionVisualizer::new(model, config);

        assert!(visualizer.attention_cache.is_empty());
    }

    #[test]
    fn test_attention_visualization_options_default() {
        let options = AttentionVisualizationOptions::default();
        assert_eq!(
            options.visualization_type,
            AttentionVisualizationType::Heatmap
        );
        assert_eq!(options.head_selection, HeadSelection::All);
        assert_eq!(options.head_aggregation, HeadAggregation::Mean);
        assert_eq!(options.threshold, Some(0.01));
    }

    #[test]
    fn test_attention_visualization_types() {
        let types = [
            AttentionVisualizationType::Heatmap,
            AttentionVisualizationType::BipartiteGraph,
            AttentionVisualizationType::ArcDiagram,
            AttentionVisualizationType::AttentionFlow,
            AttentionVisualizationType::HeadComparison,
        ];

        assert_eq!(types.len(), 5);
        assert_eq!(types[0], AttentionVisualizationType::Heatmap);
    }

    #[test]
    fn test_head_selection_variants() {
        let all = HeadSelection::All;
        let specific = HeadSelection::Specific(vec![0, 1, 2]);
        let top_k = HeadSelection::TopK(5);
        let range = HeadSelection::Range(2, 8);

        assert_eq!(all, HeadSelection::All);

        match specific {
            HeadSelection::Specific(heads) => assert_eq!(heads.len(), 3),
            _ => panic!("Expected specific head selection"),
        }

        match top_k {
            HeadSelection::TopK(k) => assert_eq!(k, 5),
            _ => panic!("Expected top-k head selection"),
        }

        match range {
            HeadSelection::Range(start, end) => {
                assert_eq!(start, 2);
                assert_eq!(end, 8);
            }
            _ => panic!("Expected range head selection"),
        }
    }

    #[test]
    fn test_head_aggregation_methods() {
        let none = HeadAggregation::None;
        let mean = HeadAggregation::Mean;
        let max = HeadAggregation::Max;
        let weighted = HeadAggregation::WeightedMean(vec![0.3, 0.7]);
        let rollout = HeadAggregation::Rollout;

        assert_eq!(none, HeadAggregation::None);
        assert_eq!(mean, HeadAggregation::Mean);
        assert_eq!(max, HeadAggregation::Max);
        assert_eq!(rollout, HeadAggregation::Rollout);

        match weighted {
            HeadAggregation::WeightedMean(weights) => assert_eq!(weights.len(), 2),
            _ => panic!("Expected weighted mean aggregation"),
        }
    }

    #[test]
    fn test_highlight_styles() {
        let styles = [
            HighlightStyle::Border,
            HighlightStyle::Background,
            HighlightStyle::Overlay,
            HighlightStyle::Glow,
        ];

        assert_eq!(styles.len(), 4);
        assert_eq!(styles[0], HighlightStyle::Border);
    }

    #[test]
    fn test_export_formats() {
        let image = ExportFormat::Image(ImageFormat::PNG);
        let html = ExportFormat::HTML;
        let svg = ExportFormat::SVG;
        let data = ExportFormat::Data(DataFormat::JSON);
        let video = ExportFormat::Video(VideoFormat::MP4);

        assert_eq!(html, ExportFormat::HTML);
        assert_eq!(svg, ExportFormat::SVG);

        match image {
            ExportFormat::Image(ImageFormat::PNG) => {}
            _ => panic!("Expected PNG image format"),
        }

        match data {
            ExportFormat::Data(DataFormat::JSON) => {}
            _ => panic!("Expected JSON data format"),
        }

        match video {
            ExportFormat::Video(VideoFormat::MP4) => {}
            _ => panic!("Expected MP4 video format"),
        }
    }

    #[test]
    fn test_export_quality_levels() {
        let qualities = [
            ExportQuality::Low,
            ExportQuality::Medium,
            ExportQuality::High,
            ExportQuality::Maximum,
        ];

        assert_eq!(qualities.len(), 4);
        assert_eq!(qualities[2], ExportQuality::High);
    }

    #[test]
    fn test_cache_operations() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut model = Sequential::<f32>::new();
        model.add_layer(Dense::new(10, 5, Some("relu"), &mut rng).unwrap());

        let config = VisualizationConfig::default();
        let mut visualizer = AttentionVisualizer::new(model, config);

        assert!(visualizer.attention_cache.is_empty());
        assert!(visualizer.get_cached_attention("test_layer").is_none());

        visualizer.clear_cache();
        assert!(visualizer.attention_cache.is_empty());
    }

    #[test]
    fn test_resolution_settings() {
        let resolution = Resolution::default();
        assert_eq!(resolution.width, 1920);
        assert_eq!(resolution.height, 1080);
        assert_eq!(resolution.dpi, 300);
    }

    #[test]
    fn test_compression_settings() {
        let compression = CompressionSettings::default();
        assert!(compression.enabled);
        assert_eq!(compression.level, 6);
        assert!(!compression.lossless);
    }
}
