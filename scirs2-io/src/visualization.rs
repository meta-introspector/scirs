//! Visualization tool integration for scientific data
//!
//! Provides interfaces and utilities for integrating with popular visualization
//! libraries and tools, enabling seamless data export and plotting capabilities.

use crate::error::{IoError, Result};
use crate::metadata::Metadata;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Visualization format types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisualizationFormat {
    /// Plotly JSON format
    PlotlyJson,
    /// Matplotlib script
    MatplotlibPython,
    /// Gnuplot script
    Gnuplot,
    /// D3.js JSON format
    D3Json,
    /// Vega-Lite specification
    VegaLite,
    /// Bokeh JSON format
    BokehJson,
    /// SVG format
    Svg,
    /// HTML with embedded visualization
    Html,
}

/// Plot type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PlotType {
    Line,
    Scatter,
    Bar,
    Histogram,
    Heatmap,
    Surface,
    Contour,
    Box,
    Violin,
    Pie,
    Area,
    Stream,
}

/// Axis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxisConfig {
    pub title: Option<String>,
    pub range: Option<[f64; 2]>,
    pub scale: Option<ScaleType>,
    pub tick_format: Option<String>,
    pub grid: bool,
}

impl Default for AxisConfig {
    fn default() -> Self {
        Self {
            title: None,
            range: None,
            scale: None,
            tick_format: None,
            grid: true,
        }
    }
}

/// Scale types for axes
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ScaleType {
    Linear,
    Log,
    SymLog,
    Sqrt,
    Power(f64),
}

/// Plot configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlotConfig {
    pub title: Option<String>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub x_axis: AxisConfig,
    pub y_axis: AxisConfig,
    pub z_axis: Option<AxisConfig>,
    pub color_scale: Option<String>,
    pub theme: Option<String>,
    pub annotations: Vec<Annotation>,
}

impl Default for PlotConfig {
    fn default() -> Self {
        Self {
            title: None,
            width: Some(800),
            height: Some(600),
            x_axis: AxisConfig::default(),
            y_axis: AxisConfig::default(),
            z_axis: None,
            color_scale: None,
            theme: None,
            annotations: Vec::new(),
        }
    }
}

/// Annotation for plots
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Annotation {
    pub text: String,
    pub x: f64,
    pub y: f64,
    pub arrow: bool,
}

/// Visualization builder
pub struct VisualizationBuilder {
    data: Vec<DataSeries>,
    config: PlotConfig,
    metadata: Metadata,
}

/// Data series for plotting
#[derive(Debug, Clone)]
pub struct DataSeries {
    pub name: Option<String>,
    pub x: Option<Vec<f64>>,
    pub y: Vec<f64>,
    pub z: Option<Vec<f64>>,
    pub plot_type: PlotType,
    pub style: SeriesStyle,
}

/// Style configuration for data series
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeriesStyle {
    pub color: Option<String>,
    pub line_style: Option<String>,
    pub marker: Option<String>,
    pub opacity: Option<f64>,
    pub size: Option<f64>,
}

impl Default for SeriesStyle {
    fn default() -> Self {
        Self {
            color: None,
            line_style: None,
            marker: None,
            opacity: Some(1.0),
            size: None,
        }
    }
}

impl VisualizationBuilder {
    /// Create a new visualization builder
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            config: PlotConfig::default(),
            metadata: Metadata::new(),
        }
    }

    /// Set plot title
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.config.title = Some(title.into());
        self
    }

    /// Set plot dimensions
    pub fn dimensions(mut self, width: u32, height: u32) -> Self {
        self.config.width = Some(width);
        self.config.height = Some(height);
        self
    }

    /// Configure X axis
    pub fn x_axis(mut self, title: impl Into<String>) -> Self {
        self.config.x_axis.title = Some(title.into());
        self
    }

    /// Configure Y axis
    pub fn y_axis(mut self, title: impl Into<String>) -> Self {
        self.config.y_axis.title = Some(title.into());
        self
    }

    /// Add a line plot
    pub fn add_line(mut self, x: &[f64], y: &[f64], name: Option<&str>) -> Self {
        self.data.push(DataSeries {
            name: name.map(|s| s.to_string()),
            x: Some(x.to_vec()),
            y: y.to_vec(),
            z: None,
            plot_type: PlotType::Line,
            style: SeriesStyle::default(),
        });
        self
    }

    /// Add a scatter plot
    pub fn add_scatter(mut self, x: &[f64], y: &[f64], name: Option<&str>) -> Self {
        self.data.push(DataSeries {
            name: name.map(|s| s.to_string()),
            x: Some(x.to_vec()),
            y: y.to_vec(),
            z: None,
            plot_type: PlotType::Scatter,
            style: SeriesStyle::default(),
        });
        self
    }

    /// Add a histogram
    pub fn add_histogram(mut self, values: &[f64], name: Option<&str>) -> Self {
        self.data.push(DataSeries {
            name: name.map(|s| s.to_string()),
            x: None,
            y: values.to_vec(),
            z: None,
            plot_type: PlotType::Histogram,
            style: SeriesStyle::default(),
        });
        self
    }

    /// Add a heatmap
    pub fn add_heatmap(mut self, z: Array2<f64>, name: Option<&str>) -> Self {
        let flat_z: Vec<f64> = z.iter().cloned().collect();
        self.data.push(DataSeries {
            name: name.map(|s| s.to_string()),
            x: Some(vec![z.shape()[1] as f64]),  // Store dimensions
            y: vec![z.shape()[0] as f64],
            z: Some(flat_z),
            plot_type: PlotType::Heatmap,
            style: SeriesStyle::default(),
        });
        self
    }

    /// Build and export to specified format
    pub fn export(self, format: VisualizationFormat, path: impl AsRef<Path>) -> Result<()> {
        let exporter = get_exporter(format);
        exporter.export(&self.data, &self.config, &self.metadata, path.as_ref())
    }

    /// Build and return as string
    pub fn to_string(self, format: VisualizationFormat) -> Result<String> {
        let exporter = get_exporter(format);
        exporter.to_string(&self.data, &self.config, &self.metadata)
    }
}

/// Trait for visualization exporters
trait VisualizationExporter {
    fn export(&self, data: &[DataSeries], config: &PlotConfig, metadata: &Metadata, path: &Path) -> Result<()>;
    fn to_string(&self, data: &[DataSeries], config: &PlotConfig, metadata: &Metadata) -> Result<String>;
}

/// Get appropriate exporter for format
fn get_exporter(format: VisualizationFormat) -> Box<dyn VisualizationExporter> {
    match format {
        VisualizationFormat::PlotlyJson => Box::new(PlotlyExporter),
        VisualizationFormat::MatplotlibPython => Box::new(MatplotlibExporter),
        VisualizationFormat::Gnuplot => Box::new(GnuplotExporter),
        VisualizationFormat::VegaLite => Box::new(VegaLiteExporter),
        _ => Box::new(PlotlyExporter), // Default to Plotly
    }
}

/// Plotly JSON exporter
struct PlotlyExporter;

impl VisualizationExporter for PlotlyExporter {
    fn export(&self, data: &[DataSeries], config: &PlotConfig, _metadata: &Metadata, path: &Path) -> Result<()> {
        let json_str = self.to_string(data, config, _metadata)?;
        let mut file = File::create(path).map_err(|e| IoError::Io(e))?;
        file.write_all(json_str.as_bytes()).map_err(|e| IoError::Io(e))?;
        Ok(())
    }

    fn to_string(&self, data: &[DataSeries], config: &PlotConfig, _metadata: &Metadata) -> Result<String> {
        let mut traces = Vec::new();
        
        for series in data {
            let trace = match series.plot_type {
                PlotType::Line | PlotType::Scatter => {
                    serde_json::json!({
                        "type": if matches!(series.plot_type, PlotType::Line) { "scatter" } else { "scatter" },
                        "mode": if matches!(series.plot_type, PlotType::Line) { "lines" } else { "markers" },
                        "name": series.name,
                        "x": series.x,
                        "y": series.y,
                        "line": {
                            "color": series.style.color,
                            "dash": series.style.line_style,
                        },
                        "marker": {
                            "symbol": series.style.marker,
                            "size": series.style.size,
                        },
                        "opacity": series.style.opacity,
                    })
                }
                PlotType::Histogram => {
                    serde_json::json!({
                        "type": "histogram",
                        "name": series.name,
                        "x": series.y,
                        "opacity": series.style.opacity,
                    })
                }
                PlotType::Heatmap => {
                    let cols = series.x.as_ref().unwrap()[0] as usize;
                    let rows = series.y[0] as usize;
                    let z_data: Vec<Vec<f64>> = series.z.as_ref().unwrap()
                        .chunks(cols)
                        .map(|chunk| chunk.to_vec())
                        .collect();
                    
                    serde_json::json!({
                        "type": "heatmap",
                        "name": series.name,
                        "z": z_data,
                        "colorscale": config.color_scale,
                    })
                }
                _ => continue,
            };
            traces.push(trace);
        }
        
        let layout = serde_json::json!({
            "title": config.title,
            "width": config.width,
            "height": config.height,
            "xaxis": {
                "title": config.x_axis.title,
                "range": config.x_axis.range,
                "showgrid": config.x_axis.grid,
            },
            "yaxis": {
                "title": config.y_axis.title,
                "range": config.y_axis.range,
                "showgrid": config.y_axis.grid,
            },
            "annotations": config.annotations.iter().map(|ann| {
                serde_json::json!({
                    "text": ann.text,
                    "x": ann.x,
                    "y": ann.y,
                    "showarrow": ann.arrow,
                })
            }).collect::<Vec<_>>(),
        });
        
        let plot_data = serde_json::json!({
            "data": traces,
            "layout": layout,
        });
        
        serde_json::to_string_pretty(&plot_data)
            .map_err(|e| IoError::SerializationError(e.to_string()))
    }
}

/// Matplotlib Python script exporter
struct MatplotlibExporter;

impl VisualizationExporter for MatplotlibExporter {
    fn export(&self, data: &[DataSeries], config: &PlotConfig, _metadata: &Metadata, path: &Path) -> Result<()> {
        let script = self.to_string(data, config, _metadata)?;
        let mut file = File::create(path).map_err(|e| IoError::Io(e))?;
        file.write_all(script.as_bytes()).map_err(|e| IoError::Io(e))?;
        Ok(())
    }

    fn to_string(&self, data: &[DataSeries], config: &PlotConfig, _metadata: &Metadata) -> Result<String> {
        let mut script = String::from("import matplotlib.pyplot as plt\nimport numpy as np\n\n");
        
        // Create figure
        script.push_str(&format!("fig, ax = plt.subplots(figsize=({}, {}))\n\n",
            config.width.unwrap_or(800) as f64 / 100.0,
            config.height.unwrap_or(600) as f64 / 100.0
        ));
        
        // Plot data
        for series in data {
            match series.plot_type {
                PlotType::Line => {
                    if let Some(x) = &series.x {
                        script.push_str(&format!("ax.plot({:?}, {:?}", x, series.y));
                        if let Some(name) = &series.name {
                            script.push_str(&format!(", label='{}'", name));
                        }
                        script.push_str(")\n");
                    }
                }
                PlotType::Scatter => {
                    if let Some(x) = &series.x {
                        script.push_str(&format!("ax.scatter({:?}, {:?}", x, series.y));
                        if let Some(name) = &series.name {
                            script.push_str(&format!(", label='{}'", name));
                        }
                        script.push_str(")\n");
                    }
                }
                PlotType::Histogram => {
                    script.push_str(&format!("ax.hist({:?}", series.y));
                    if let Some(name) = &series.name {
                        script.push_str(&format!(", label='{}'", name));
                    }
                    script.push_str(")\n");
                }
                _ => continue,
            }
        }
        
        // Configure axes
        if let Some(title) = &config.title {
            script.push_str(&format!("\nax.set_title('{}')\n", title));
        }
        if let Some(xlabel) = &config.x_axis.title {
            script.push_str(&format!("ax.set_xlabel('{}')\n", xlabel));
        }
        if let Some(ylabel) = &config.y_axis.title {
            script.push_str(&format!("ax.set_ylabel('{}')\n", ylabel));
        }
        
        script.push_str("\nax.grid(True)\n");
        script.push_str("ax.legend()\n");
        script.push_str("plt.tight_layout()\n");
        script.push_str("plt.show()\n");
        
        Ok(script)
    }
}

/// Gnuplot script exporter
struct GnuplotExporter;

impl VisualizationExporter for GnuplotExporter {
    fn export(&self, data: &[DataSeries], config: &PlotConfig, _metadata: &Metadata, path: &Path) -> Result<()> {
        let script = self.to_string(data, config, _metadata)?;
        let mut file = File::create(path).map_err(|e| IoError::Io(e))?;
        file.write_all(script.as_bytes()).map_err(|e| IoError::Io(e))?;
        Ok(())
    }

    fn to_string(&self, data: &[DataSeries], config: &PlotConfig, _metadata: &Metadata) -> Result<String> {
        let mut script = String::new();
        
        // Set terminal and output
        script.push_str("set terminal png size ");
        script.push_str(&format!("{},{}\n", config.width.unwrap_or(800), config.height.unwrap_or(600)));
        script.push_str("set output 'plot.png'\n\n");
        
        // Set title and labels
        if let Some(title) = &config.title {
            script.push_str(&format!("set title '{}'\n", title));
        }
        if let Some(xlabel) = &config.x_axis.title {
            script.push_str(&format!("set xlabel '{}'\n", xlabel));
        }
        if let Some(ylabel) = &config.y_axis.title {
            script.push_str(&format!("set ylabel '{}'\n", ylabel));
        }
        
        script.push_str("set grid\n\n");
        
        // Plot command
        script.push_str("plot ");
        let mut first = true;
        
        for (i, series) in data.iter().enumerate() {
            if !first {
                script.push_str(", ");
            }
            first = false;
            
            match series.plot_type {
                PlotType::Line => {
                    script.push_str(&format!("'-' using 1:2 with lines title '{}'", 
                        series.name.as_deref().unwrap_or(&format!("Series {}", i))));
                }
                PlotType::Scatter => {
                    script.push_str(&format!("'-' using 1:2 with points title '{}'", 
                        series.name.as_deref().unwrap_or(&format!("Series {}", i))));
                }
                _ => continue,
            }
        }
        script.push_str("\n\n");
        
        // Data blocks
        for series in data {
            if let Some(x) = &series.x {
                for (xi, yi) in x.iter().zip(series.y.iter()) {
                    script.push_str(&format!("{} {}\n", xi, yi));
                }
            }
            script.push_str("e\n");
        }
        
        Ok(script)
    }
}

/// Vega-Lite specification exporter
struct VegaLiteExporter;

impl VisualizationExporter for VegaLiteExporter {
    fn export(&self, data: &[DataSeries], config: &PlotConfig, _metadata: &Metadata, path: &Path) -> Result<()> {
        let spec = self.to_string(data, config, _metadata)?;
        let mut file = File::create(path).map_err(|e| IoError::Io(e))?;
        file.write_all(spec.as_bytes()).map_err(|e| IoError::Io(e))?;
        Ok(())
    }

    fn to_string(&self, data: &[DataSeries], config: &PlotConfig, _metadata: &Metadata) -> Result<String> {
        let mut data_values = Vec::new();
        
        for series in data {
            if let Some(x) = &series.x {
                for (xi, yi) in x.iter().zip(series.y.iter()) {
                    data_values.push(serde_json::json!({
                        "x": xi,
                        "y": yi,
                        "series": series.name.as_deref().unwrap_or("default"),
                    }));
                }
            }
        }
        
        let spec = serde_json::json!({
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "title": config.title,
            "width": config.width,
            "height": config.height,
            "data": {
                "values": data_values
            },
            "mark": "line",
            "encoding": {
                "x": {
                    "field": "x",
                    "type": "quantitative",
                    "title": config.x_axis.title,
                },
                "y": {
                    "field": "y",
                    "type": "quantitative",
                    "title": config.y_axis.title,
                },
                "color": {
                    "field": "series",
                    "type": "nominal"
                }
            }
        });
        
        serde_json::to_string_pretty(&spec)
            .map_err(|e| IoError::SerializationError(e.to_string()))
    }
}

/// Quick plotting functions
pub mod quick {
    use super::*;
    
    /// Quick line plot
    pub fn plot_line(x: &[f64], y: &[f64], output: impl AsRef<Path>) -> Result<()> {
        VisualizationBuilder::new()
            .title("Line Plot")
            .add_line(x, y, None)
            .export(VisualizationFormat::PlotlyJson, output)
    }
    
    /// Quick scatter plot
    pub fn plot_scatter(x: &[f64], y: &[f64], output: impl AsRef<Path>) -> Result<()> {
        VisualizationBuilder::new()
            .title("Scatter Plot")
            .add_scatter(x, y, None)
            .export(VisualizationFormat::PlotlyJson, output)
    }
    
    /// Quick histogram
    pub fn plot_histogram(values: &[f64], output: impl AsRef<Path>) -> Result<()> {
        VisualizationBuilder::new()
            .title("Histogram")
            .add_histogram(values, None)
            .export(VisualizationFormat::PlotlyJson, output)
    }
    
    /// Quick heatmap
    pub fn plot_heatmap(z: &Array2<f64>, output: impl AsRef<Path>) -> Result<()> {
        VisualizationBuilder::new()
            .title("Heatmap")
            .add_heatmap(z.clone(), None)
            .export(VisualizationFormat::PlotlyJson, output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_visualization_builder() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 1.0, 4.0, 9.0, 16.0];
        
        let result = VisualizationBuilder::new()
            .title("Test Plot")
            .x_axis("X values")
            .y_axis("Y values")
            .add_line(&x, &y, Some("y = x²"))
            .to_string(VisualizationFormat::PlotlyJson);
        
        assert!(result.is_ok());
        let json_str = result.unwrap();
        assert!(json_str.contains("Test Plot"));
        assert!(json_str.contains("y = x²"));
    }

    #[test]
    fn test_matplotlib_export() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 4.0, 9.0];
        
        let result = VisualizationBuilder::new()
            .title("Matplotlib Test")
            .add_scatter(&x, &y, Some("data"))
            .to_string(VisualizationFormat::MatplotlibPython);
        
        assert!(result.is_ok());
        let script = result.unwrap();
        assert!(script.contains("import matplotlib.pyplot"));
        assert!(script.contains("ax.scatter"));
    }
}