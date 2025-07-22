//! Model architecture visualization utilities
//!
//! This module provides utilities for visualizing the architecture
//! of neural network models.

use crate::error::{NeuralError, Result};
use crate::layers::Layer;
use crate::layers::Sequential;
use crate::utils::colors::{colorize, stylize, Color, ColorOptions, Style};
use ndarray::ScalarOperand;
use num_traits::Float;
use std::fmt::Debug;
/// Represents a node in the model architecture graph
#[derive(Debug, Clone)]
struct ModelNode {
    /// Layer name or description
    name: String,
    /// Input shape
    input_shape: Option<Vec<usize>>,
    /// Output shape
    output_shape: Option<Vec<usize>>,
    /// Number of parameters
    parameters: Option<usize>,
    /// Layer type
    layer_type: String,
    /// Additional properties
    properties: Vec<(String, String)>,
}
/// Options for model architecture visualization
pub struct ModelVizOptions {
    /// Width of the visualization
    pub width: usize,
    /// Show parameter counts
    pub show_params: bool,
    /// Show layer shapes
    pub show_shapes: bool,
    /// Show layer properties
    pub show_properties: bool,
    /// Color options
    pub color_options: ColorOptions,
}

impl Default for ModelVizOptions {
    fn default() -> Self {
        Self {
            width: 80,
            show_params: true,
            show_shapes: true,
            show_properties: true,
            color_options: ColorOptions::default(),
        }
    }
}

/// Create an ASCII text representation of a sequential model architecture
///
/// # Arguments
/// * `model` - The sequential model to visualize
/// * `input_shape` - Optional input shape to propagate through the model
/// * `title` - Optional title for the visualization
/// * `options` - Visualization options
/// # Returns
/// * `Result<String>` - ASCII representation of the model architecture
#[allow(dead_code)]
pub fn sequential_model_summary<
    F: Float + Debug + ScalarOperand + num_traits::FromPrimitive + std::fmt::Display,
>(
    model: &Sequential<F>,
    input_shape: Option<Vec<usize>>,
    title: Option<&str>,
    options: Option<ModelVizOptions>,
) -> Result<String> {
    let options = options.unwrap_or_default();
    // Width is used for column calculation later
    let colors = &options.color_options;
    let mut result = String::new();
    // Add title
    if let Some(title_text) = title {
        if colors.enabled {
            result.push_str(&stylize(title_text, Style::Bold));
        } else {
            result.push_str(title_text);
        }
        result.push_str("\n\n");
    }
    // Extract layer information
    let layer_infos = model.layer_info();
    if layer_infos.is_empty() {
        return Err(NeuralError::ValidationError(
            "Model has no layers".to_string(),
        ));
    }

    // Create nodes for each layer
    let mut nodes = Vec::new();
    // Add input node if _shape is provided
    if let Some(_shape) = input_shape.clone() {
        nodes.push(ModelNode {
            name: "Input".to_string(),
            input_shape: None,
            output_shape: Some(_shape),
            parameters: Some(0),
            layer_type: "Input".to_string(),
            properties: Vec::new(),
        });
    }

    // Add actual layer nodes
    for layer_info in &layer_infos {
        let layer_name = if layer_info.name.starts_with("Layer_") {
            let index = layer_info.index + 1;
            format!("Layer {index}")
        } else {
            layer_info.name.clone()
        };
        // Create properties from layer info
        let mut properties = Vec::new();
        if let Some(ref input_shape) = layer_info.input_shape {
            properties.push(("Input Shape".to_string(), format!("{input_shape:?}")));
        }
        if let Some(ref output_shape) = layer_info.output_shape {
            properties.push(("Output Shape".to_string(), format!("{output_shape:?}")));
        }

        let node = ModelNode {
            name: layer_name,
            input_shape: layer_info.input_shape.clone(),
            output_shape: layer_info.output_shape.clone(),
            parameters: Some(layer_info.parameter_count),
            layer_type: layer_info.layer_type.clone(),
            properties,
        };
        nodes.push(node);
    }

    // Try to propagate shapes if input _shape is provided
    if let Some(input_shape) = input_shape {
        // For now, simplified approach since we can't easily run the forward pass here
        // In a full implementation, this would use actual layer logic
        let mut current_shape = input_shape;
        for (i, node) in nodes.iter_mut().enumerate() {
            if i > 0 {
                // Skip input node
                node.input_shape = Some(current_shape.clone());
                // Very simplified _shape propagation (would need more detailed layer info)
                if node.layer_type == "Dense" {
                    if let Some(output_size) = extract_output_size(node) {
                        // For Dense layers, output _shape is (batch_size, output_size)
                        if !current_shape.is_empty() {
                            let mut output_shape = current_shape.clone();
                            if output_shape.len() > 1 {
                                let last_idx = output_shape.len() - 1;
                                output_shape[last_idx] = output_size;
                            } else {
                                output_shape = vec![output_size];
                            }
                            current_shape = output_shape.clone();
                            node.output_shape = Some(output_shape);
                        }
                    }
                } else {
                    // For other layer types, assume _shape is preserved
                    node.output_shape = Some(current_shape.clone());
                }
            }
        }
    }

    // Calculate total parameters
    let total_params: usize = nodes.iter().filter_map(|node| node.parameters).sum();
    // Determine column widths
    let name_width = nodes
        .iter()
        .map(|node| node.name.len())
        .max()
        .unwrap_or(10)
        .max(10);
    let type_width = nodes
        .iter()
        .map(|node| node.layer_type.len())
        .max()
        .unwrap_or(8)
        .max(8);
    let shape_width = if options.show_shapes {
        nodes
            .iter()
            .map(|node| {
                let input_str = node.input_shape.as_ref().map(|s| format!("{s:?}"));
                let output_str = node.output_shape.as_ref().map(|s| format!("{s:?}"));
                let input_len = input_str.as_ref().map(|s| s.len()).unwrap_or(0);
                let output_len = output_str.as_ref().map(|s| s.len()).unwrap_or(0);
                input_len.max(output_len)
            })
            .max()
            .unwrap_or(15)
            .max(15)
    } else {
        0
    };
    let params_width = if options.show_params {
        14 // Room for formatted parameter counts
    } else {
        0
    };

    // Add header
    let mut header = format!(
        "{:<width$} | {:<type_width$}",
        if options.color_options.enabled {
            stylize("Layer", Style::Bold).to_string()
        } else {
            "Layer".to_string()
        },
        if options.color_options.enabled {
            stylize("Type", Style::Bold).to_string()
        } else {
            "Type".to_string()
        },
        width = name_width,
        type_width = type_width
    );
    if options.show_shapes {
        header.push_str(&format!(
            " | {:<shape_width$}",
            if options.color_options.enabled {
                stylize("Output Shape", Style::Bold).to_string()
            } else {
                "Output Shape".to_string()
            },
            shape_width = shape_width
        ));
    }
    if options.show_params {
        header.push_str(&format!(
            " | {:<params_width$}",
            if options.color_options.enabled {
                stylize("Params", Style::Bold).to_string()
            } else {
                "Params".to_string()
            },
            params_width = params_width
        ));
    }

    let mut result = String::new();
    result.push_str(&header);
    result.push('\n');
    // Add separator
    let total_width = name_width
        + type_width
        + (if options.show_shapes {
            shape_width + 3
        } else {
            0
        })
        + (if options.show_params {
            params_width + 3
        } else {
            0
        })
        + 1;
    result.push_str(&"-".repeat(total_width));
    // Add layers
    for node in &nodes {
        // Layer name with color
        let mut line = if options.color_options.enabled {
            let styled_name = match node.layer_type.as_str() {
                "Input" => colorize(&node.name, Color::BrightCyan),
                "Dense" => colorize(&node.name, Color::BrightGreen),
                "Conv2D" => colorize(&node.name, Color::BrightMagenta),
                "RNN" | "LSTM" | "GRU" => colorize(&node.name, Color::BrightBlue),
                "BatchNorm" | "Dropout" => colorize(&node.name, Color::Yellow),
                _ => colorize(&node.name, Color::BrightWhite),
            };
            format!("{:<width$} | ", styled_name, width = name_width + 9) // Add space for ANSI codes
        } else {
            format!("{:<width$} | ", node.name, width = name_width)
        };
        // Layer type
        line.push_str(&format!(
            "{:<type_width$}",
            node.layer_type,
            type_width = type_width
        ));

        // Output _shape
        if options.show_shapes {
            let shape_str = if let Some(_shape) = &node.output_shape {
                format!("{_shape:?}")
            } else {
                "?".to_string()
            };
            line.push_str(&format!(" | {shape_str:<shape_width$}"));
        }

        // Parameters
        if options.show_params {
            if let Some(params) = node.parameters {
                let params_str = if params >= 1_000_000 {
                    let param_mb = params as f64 / 1_000_000.0;
                    format!("{param_mb:.2}M")
                } else if params >= 1_000 {
                    let param_kb = params as f64 / 1_000.0;
                    format!("{param_kb:.2}K")
                } else {
                    format!("{params}")
                };
                line.push_str(&format!(" | {params_str:<params_width$}"));
            } else {
                line.push_str(&format!(" | {question:<params_width$}", question = "?"));
            }
        }

        result.push_str(&line);
        result.push('\n');
        // Add properties if enabled
        if options.show_properties && !node.properties.is_empty() {
            for (key, value) in &node.properties {
                let prop_line = if options.color_options.enabled {
                    let styled_key = stylize(format!("  - {key}"), Style::Dim);
                    format!("{styled_key}: {value}")
                } else {
                    format!("  - {key}: {value}")
                };
                result.push_str(&prop_line);
                result.push('\n');
            }
        }
    }

    // Add summary information
    let trainable_params = total_params; // For now, assume all are trainable
    let formatted_total = format_params(total_params);
    let summary = format!("Total parameters: {formatted_total}");
    if options.color_options.enabled {
        result.push_str(&stylize(&summary, Style::Bold));
    } else {
        result.push_str(&summary);
    }
    result.push('\n');

    // Trainable parameters
    let formatted_trainable = format_params(trainable_params);
    let trainable_summary = format!("Trainable parameters: {formatted_trainable}");
    if options.color_options.enabled {
        result.push_str(&stylize(&trainable_summary, Style::Bold));
    } else {
        result.push_str(&trainable_summary);
    }
    result.push('\n');
    // Non-trainable parameters
    let non_trainable_params = total_params - trainable_params;
    let non_trainable_summary = format!(
        "Non-trainable parameters: {}",
        format_params(non_trainable_params)
    );
    if options.color_options.enabled {
        result.push_str(&stylize(&non_trainable_summary, Style::Bold));
    } else {
        result.push_str(&non_trainable_summary);
    }
    result.push('\n');

    Ok(result)
}
/// Creates an ASCII representation of the data flow through a sequential model
/// This visualization shows how data flows through the network layers,
/// including transformations in shape and any connections between layers.
#[allow(dead_code)]
pub fn sequential_model_dataflow<
    F: Float + Debug + ScalarOperand + num_traits::FromPrimitive + std::fmt::Display,
>(
    model: &Sequential<F>,
    input_shape: Vec<usize>,
    options: Option<ModelVizOptions>,
) -> Result<String> {
    let options = options.unwrap_or_default();
    let width = options.width;
    // Create nodes for visualization (input + layers)
    let layer_infos = model.layer_info();
    let mut nodes: Vec<ModelNode> = Vec::with_capacity(layer_infos.len() + 1);
    // Add input node
    nodes.push(ModelNode {
        name: "Input".to_string(),
        input_shape: None,
        output_shape: Some(input_shape.clone()),
        parameters: Some(0),
        layer_type: "Input".to_string(),
        properties: Vec::new(),
    });
    // Add layer nodes with simplified _shape propagation
    let mut current_shape = input_shape.clone();

    for (i, layer_info) in layer_infos.iter().enumerate() {
        let layer_name = if layer_info.name.starts_with("Layer_") {
            let index = i + 1;
            format!("Layer_{index}")
        } else {
            layer_info.name.clone()
        };
        let layer_type = layer_info.layer_type.clone();
        let mut properties: Vec<(String, String)> = Vec::new();
        if layer_info.parameter_count > 0 {
            properties.push((
                "Parameters".to_string(),
                layer_info.parameter_count.to_string(),
            ));
        }
        let input_shape = current_shape.clone();
        // Very simplified _shape inference
        let output_shape = match layer_type.as_str() {
            "Dense" => {
                if let Some(output_size) = properties
                    .iter()
                    .find(|(key, _)| key == "output_dim")
                    .map(|(_, value)| value.parse::<usize>().unwrap_or(0))
                {
                    if !current_shape.is_empty() {
                        let mut new_shape = current_shape.clone();
                        let last_idx = new_shape.len() - 1;
                        new_shape[last_idx] = output_size;
                        new_shape
                    } else {
                        vec![output_size]
                    }
                } else {
                    current_shape.clone()
                }
            }
            "Conv2D" => {
                if current_shape.len() >= 3 {
                    // Very simplified...in reality we'd need filter count, strides, etc.
                    current_shape.clone()
                } else {
                    current_shape.clone()
                }
            }
            _ => current_shape.clone(),
        };

        current_shape = output_shape.clone();

        let node = ModelNode {
            name: layer_name,
            input_shape: Some(input_shape),
            output_shape: Some(output_shape),
            parameters: Some(0), // Simplified for now
            layer_type,
            properties,
        };
        nodes.push(node);
    }
    // Draw the data flow diagram
    //
    // Format:
    //    ┌──────────────┐
    //    │    Input     │
    //    │  [batch, 28, 28, 1]  │
    //    └──────┬───────┘
    //           │
    //           ▼
    //    │    Conv2D    │
    //    │ [b, 26, 26, 32] │
    let mut result = String::new();
    let box_width = 20.min(width / 2);

    for (i, node) in nodes.iter().enumerate() {
        // Draw box top
        result.push_str(&" ".repeat((width - box_width) / 2));
        result.push('┌');
        result.push_str(&"─".repeat(box_width - 2));
        result.push('┐');
        result.push('\n');

        // Draw layer name
        let name = if node.layer_type == "Input" {
            node.layer_type.clone()
        } else {
            format!("{} ({})", node.layer_type, node.name)
        };
        let padded_name = format!("{name:^width$}", width = box_width - 2);
        result.push_str(&" ".repeat((width - box_width) / 2));

        let styled_name = if options.color_options.enabled {
            match node.layer_type.as_str() {
                "Input" => colorize(&padded_name, Color::BrightCyan),
                "Dense" => colorize(&padded_name, Color::BrightGreen),
                "Conv2D" => colorize(&padded_name, Color::BrightMagenta),
                "RNN" | "LSTM" | "GRU" => colorize(&padded_name, Color::BrightBlue),
                "BatchNorm" | "Dropout" => colorize(&padded_name, Color::Yellow),
                _ => padded_name.to_string(),
            }
        } else {
            padded_name
        };

        result.push('│');
        result.push_str(&styled_name);
        result.push('│');
        result.push('\n');

        // Draw _shape info
        if let Some(_shape) = &node.output_shape {
            let shape_str = format!("{_shape:?}");
            let padded_shape = format!("{shape_str:^width$}", width = box_width - 2);
            result.push_str(&" ".repeat((width - box_width) / 2));
            result.push('│');
            if options.color_options.enabled {
                result.push_str(&stylize(&padded_shape, Style::Dim));
            } else {
                result.push_str(&padded_shape);
            }
            result.push('│');
            result.push('\n');
        }
        // Draw box bottom
        result.push_str(&" ".repeat((width - box_width) / 2));
        result.push('└');
        result.push_str(&"─".repeat(box_width - 2));
        result.push('┘');
        result.push('\n');

        // Draw connector to next layer if not the last node
        if i < nodes.len() - 1 {
            result.push_str(&" ".repeat(width / 2));
            result.push('│');
            result.push('\n');
            result.push_str(&" ".repeat(width / 2));
            result.push('▼');
            result.push('\n');
        }
    }

    // Add summary if requested
    let total_params: usize = nodes.iter().filter_map(|node| node.parameters).sum();
    let formatted_total = format_params(total_params);
    let summary = format!("Total parameters: {formatted_total}");
    if options.color_options.enabled {
        result.push_str(&stylize(&summary, Style::Bold));
    } else {
        result.push_str(&summary);
    }
    result.push('\n');

    Ok(result)
}
// Helper function to extract output size from a layer's properties
#[allow(dead_code)]
fn extract_output_size(_node: &ModelNode) -> Option<usize> {
    if _node.layer_type == "Dense" {
        for (key, value) in &_node.properties {
            if key == "output_dim" {
                return value.parse::<usize>().ok();
            }
        }
    }
    None
}
// Helper function to extract useful properties from a layer
#[allow(dead_code)]
fn extract_layer_properties<F: Float + Debug + ScalarOperand>(
    layer: &(dyn Layer<F> + Send + Sync),
) -> Vec<(String, String)> {
    let mut properties = Vec::new();
    let description = layer.layer_description();
    // Very simple parsing of layer description
    // In a real implementation, we'd want direct API access to layer properties
    let parts: Vec<&str> = description.split(',').collect();
    for part in parts {
        let kv: Vec<&str> = part.split(':').collect();
        if kv.len() == 2 {
            let key = kv[0].trim().to_string();
            let value = kv[1].trim().to_string();
            if key != "type" && !key.is_empty() && !value.is_empty() {
                properties.push((key, value));
            }
        }
    }
    properties
}
// Helper function to format parameter counts
#[allow(dead_code)]
fn format_params(_params: usize) -> String {
    if _params >= 1_000_000 {
        format!(
            "{:.2}M ({} parameters)",
            _params as f64 / 1_000_000.0,
            _params
        )
    } else if _params >= 1_000 {
        let param_kb = _params as f64 / 1_000.0;
        format!("{param_kb:.2}K ({_params} parameters)")
    } else {
        format!("{_params} parameters")
    }
}
